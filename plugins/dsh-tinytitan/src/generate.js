/**
 * Generate the harness's TinyTitan route with no checkout to shell out to.
 *
 * `tools/dsh_route.sh` is the one place that turns the installed models into the
 * `llm-pi-ai` route block, and it stays the source of truth wherever a checkout
 * exists. A plugin installed from a catalogue is a plain package beside no
 * checkout, though, so it cannot run that script; this module is the fallback.
 * It mirrors the shell tool exactly — the three header comment lines, the three
 * switches that are easy to get wrong by hand, the effort ladder each chat
 * template renders, and the line-based settings surgery — and
 * `test/generate.test.js` pins the block against the shell tool's own `--print`
 * output on the real catalog.
 *
 * Dependency-free on purpose (node builtins only): a package that needed a
 * checkout to build could not be the fallback for a checkout being absent.
 *
 * @module dsh-tinytitan/generate
 */
import { execFileSync } from "node:child_process";
import {
  accessSync,
  constants,
  copyFileSync,
  existsSync,
  readFileSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { delimiter, join } from "node:path";

import { scanModelsFolder } from "./catalog-scan.js";

/** The block's defaults, copied from `tools/dsh_route.sh`'s own. */
export const ROUTE_DEFAULTS = Object.freeze({
  port: 8080,
  provider: "tinytitan",
  context: 262144,
  maxTokens: 32768,
  reasoning: "medium",
});

/** The binary a built checkout or app ships. */
export const SERVER_BINARY_NAME = "TinyTitanServer";

/**
 * The order the shell tool offers thinking levels in.
 *
 * The catalog reports the levels a template renders, but not necessarily in
 * pi-ai's order; the shell parser re-sorts them by this list (keeping anything
 * newer, in catalog order), so the generated block matches its output.
 */
const LEVEL_ORDER = ["off", "on", "minimal", "low", "medium", "high", "xhigh", "max"];

/**
 * The shell's `field()`: a value that can print on one YAML line.
 *
 * The parser that feeds the route skips a model whose id, name or path is
 * missing or multi-line, so a malformed install cannot split the block across
 * lines. `null` is rejected here where Python would print `None`; a catalog is
 * JSON and a null there is a bug, not a name.
 */
function textField(value) {
  const text = value === undefined || value === null ? "" : String(value);
  if (text === "" || /[\t\r\n]/.test(text)) throw new Error("empty or multi-line field");
  return text;
}

/**
 * The levels of one model, in pi-ai's order, defaulting like the shell parser.
 *
 * The parser does `model.get("thinking") or ["off"]` and then iterates it, so a
 * missing, empty or other-falsy list means `off`, while a *string* is iterated
 * character by character. A truthy non-iterable is a TypeError there and skips
 * the model. Mirroring that keeps the block byte-identical even on a catalog a
 * hand-edit produced.
 */
function catalogLevels(thinking) {
  const listed = [];
  if (Array.isArray(thinking)) {
    if (thinking.length === 0) listed.push("off");
    else for (const level of thinking) listed.push(String(level));
  } else if (typeof thinking === "string") {
    if (thinking === "") listed.push("off");
    else for (const character of thinking) listed.push(character);
  } else if (thinking !== null && typeof thinking === "object") {
    const keys = Object.keys(thinking);
    if (keys.length === 0) listed.push("off");
    else listed.push(...keys);
  } else if (thinking === undefined || thinking === null || thinking === 0 || thinking === false) {
    listed.push("off");
  } else {
    throw new Error("thinking is not iterable");
  }
  return [
    ...LEVEL_ORDER.filter((level) => listed.includes(level)),
    ...listed.filter((level) => !LEVEL_ORDER.includes(level)),
  ];
}

/**
 * Normalize a catalog's `models` array into the rows the block is built from.
 *
 * GPU installs first, then CPU ones, mirroring the shell parser: a server
 * serves one resident model at a time, so the route only orders the picker, and
 * GPU is what a launcher starts by default. A row missing a field the block
 * needs is skipped rather than failing the whole route.
 *
 * @param models - the catalog's `models` array.
 * @returns one `{id, name, path, backend, family, engines, levels}` per servable model.
 */
export function catalogRows(models) {
  if (!Array.isArray(models)) throw new Error('catalog: "models" is not a list');
  const gpu = [];
  const cpu = [];
  for (const model of models) {
    let row;
    try {
      if (model === null || typeof model !== "object") throw new Error("not an object");
      const backend = textField(model.backend);
      if (backend !== "gpu" && backend !== "cpu") throw new Error(`backend ${backend}`);
      const quant = model.quant;
      if (
        quant === undefined ||
        quant === null ||
        quant === "" ||
        !Number.isFinite(Number(quant))
      ) {
        throw new Error("quant");
      }
      // Not used by the block, but the shell parser formats it and skips the
      // model when it cannot; dropping the same rows keeps the two in step.
      const size = model.size_gb;
      if (size !== undefined && size !== null && Number.isNaN(Number(size))) {
        throw new Error("size_gb");
      }
      row = {
        id: textField(model.id),
        name: textField(model.name),
        path: textField(model.path),
        backend,
        family: textField(model.family || "-"),
        engines: textField(model.engines || backend),
        levels: catalogLevels(model.thinking),
      };
    } catch {
      // The shell's parser prints "catalog: skipping …" on stderr and carries
      // on; the route is still useful for the installs that did parse.
      continue;
    }
    (row.backend === "gpu" ? gpu : cpu).push(row);
  }
  return [...gpu, ...cpu];
}

/**
 * One model's `reasoningEfforts` map.
 *
 * pi-ai's level vocabulary has no `on`, while a binary-thinking template renders
 * exactly off|on. Such a family's thinking mode is therefore offered as
 * `medium` with the wire value `on`, so the picker shows one thinking choice and
 * TinyTitan reads `on`.
 */
function effortsLines(levels) {
  const lines = ["          reasoningEfforts:"];
  for (const raw of levels) {
    const level = raw.replace(/ /g, "");
    if (level === "") continue;
    if (level === "off") lines.push("            off:");
    else if (level === "on") lines.push("            medium: on");
    else lines.push(`            ${level}: ${level}`);
  }
  return lines;
}

/**
 * The label the DSH picker renders.
 *
 * The picker shows `name` and nothing else, while a catalog's display name
 * carries no width -- `Qwen 3.5 2B` names both the 4-bit and the 8-bit install,
 * so the two rows arrived looking identical and the width could not be picked.
 * The routed width is always on the id (`..._4-Bit`), which is where the shell
 * tool reads it too, so the two implementations stay byte-identical.
 */
function routeLabel(name, id) {
  const match = /_(\d+)-Bit$/.exec(id);
  return match === null ? name : `${name} (${match[1]}-bit)`;
}

/** Build the block from already-normalized rows. */
function buildBlock(rows, options) {
  const port = String(options.port ?? ROUTE_DEFAULTS.port);
  const provider = String(options.provider ?? ROUTE_DEFAULTS.provider);
  const context = String(options.context ?? ROUTE_DEFAULTS.context);
  const maxTokens = String(options.maxTokens ?? ROUTE_DEFAULTS.maxTokens);
  const reasoning = String(options.reasoning ?? ROUTE_DEFAULTS.reasoning);

  const lines = [
    `# DeepSeek Harness route to the TinyTitan server on port ${port}.`,
    "# Generated by tools/dsh_route.sh from the installs under models/.",
    `# ${rows.length} served model(s); settings.yaml is hot-reloaded.`,
    "llm-pi-ai:",
    "  providers:",
    `    ${provider}:`,
    "      displayName: TinyTitan",
    "      api: openai-completions",
    `      baseURL: http://127.0.0.1:${port}/v1`,
    '      # pi-ai refuses a keyless route ("No API key for provider");',
    "      # TinyTitan has no authentication and ignores the header.",
    "      headers:",
    "        authorization: Bearer tinytitan-local",
    "      # Level for calls that name none: compaction and session titles.",
    "      # Use `off` to keep those unthinking, or install plugins/dsh-tinytitan,",
    "      # which forces it for them without turning chat off.",
    `      reasoning: ${reasoning}`,
    "      # TinyTitan emits nothing until the first token; pi-ai's own default",
    "      # abandons an idle stream after five minutes.",
    "      streamIdleTimeoutMs: 3600000",
    `      defaultContextWindow: ${context}`,
    `      defaultMaxTokens: ${maxTokens}`,
    "      models:",
  ];
  for (const row of rows) {
    lines.push(
      `        - id: ${row.id}`,
      `          name: ${routeLabel(row.name, row.id)}`,
      `          contextWindow: ${context}`,
      `          maxTokens: ${maxTokens}`,
      ...effortsLines(row.levels),
      "          compat:",
      "            # The only place TinyTitan reads the thinking switch; pi-ai's",
      "            # `qwen` format sends it top-level, where TinyTitan ignores it.",
      "            thinkingFormat: chat-template",
      "            chatTemplateKwargs:",
      "              enable_thinking: { $var: thinking.enabled }",
      "              reasoning_effort: { $var: thinking.effort }",
      "            maxTokensField: max_tokens",
      "            supportsUsageInStreaming: true",
    );
  }
  return `${lines.join("\n")}\n`;
}

/**
 * The `llm-pi-ai` block for a catalog, byte-for-byte as the shell tool prints it.
 * @param models - the catalog's `models` array.
 * @param options - `port`, `provider`, `context`, `maxTokens`, `reasoning`.
 * @returns the block, ending in a newline.
 */
export function generateBlock(models, options = {}) {
  return buildBlock(catalogRows(models), options);
}

/**
 * Whether a comment line belongs to a previous run's generated header.
 *
 * The header sits *above* `llm-pi-ai:`, so the section replacement below has to
 * remove it explicitly; without that, every refresh (the plugin runs at every
 * harness boot) would leave three more stale comment lines behind.
 */
export function generatedHeader(line) {
  const stripped = String(line).trim();
  if (stripped.startsWith("# DeepSeek Harness route to the ")) return true;
  if (stripped.startsWith("# Generated by tools/dsh_route.sh ")) return true;
  return /^# \d+ served model\(s\); settings\.yaml is hot-reloaded\.$/.test(stripped);
}

/** Split keeping line endings, the way Python's `splitlines(keepends=True)` does. */
function splitKeepingEnds(text) {
  if (text === "") return [];
  return text.match(/[^\n]*\n|[^\n]+$/g) ?? [];
}

/**
 * Replace the `llm-pi-ai` section in a settings file with the generated block.
 *
 * Line-based surgery, not a YAML round-trip: the file is the person's, with
 * their comments, and a parse-and-dump would rewrite all of it. A section ends
 * at the next line that starts in column 0 and is not a comment or blank. The
 * replacement starts at `llm-pi-ai:`, which is *below* the block's own three-line
 * header, so the previous refresh's header is removed explicitly rather than
 * left orphaned. When there is no section the block is appended.
 *
 * CRLF is normalized to LF first because the shell tool reads the file through
 * Python's universal newlines and rewrites it with LF, so a route refresh
 * normalizes the whole file; matching that keeps the two implementations
 * byte-identical on a file edited elsewhere.
 *
 * Pure: it returns the new text and writes nothing, so `writeRouteSettings` can
 * back the file up first and tests can compare a rewrite byte-for-byte.
 *
 * @param settingsText - the settings file.
 * @param block - the generated block.
 * @returns the settings text, with exactly one generated block.
 */
export function applyRouteToSettings(settingsText, block) {
  const body = String(block).endsWith("\n") ? String(block) : `${block}\n`;
  const lines = splitKeepingEnds(String(settingsText).replace(/\r\n|\r/g, "\n"));
  const out = [];
  let index = 0;
  let replaced = false;
  while (index < lines.length) {
    const line = lines[index];
    if (line.startsWith("llm-pi-ai:")) {
      index += 1;
      while (index < lines.length) {
        const following = lines[index];
        if (following.trim() === "" || following.startsWith(" ") || following.startsWith("\t")) {
          index += 1;
          continue;
        }
        break;
      }
      while (out.length > 0 && out[out.length - 1].trim() === "") out.pop();
      while (out.length > 0 && generatedHeader(out[out.length - 1])) {
        out.pop();
        while (out.length > 0 && out[out.length - 1].trim() === "") out.pop();
      }
      if (out.length > 0) out.push("\n");
      out.push(body);
      replaced = true;
      continue;
    }
    out.push(line);
    index += 1;
  }
  if (!replaced) {
    while (out.length > 0 && out[out.length - 1].trim() === "") out.pop();
    if (out.length > 0) out.push("\n");
    out.push(body);
  }
  return out.join("");
}

/**
 * Write the block into a settings file, after backing it up.
 *
 * A refresh that would change nothing writes nothing and makes no backup, so
 * the plugin's per-boot run does not pile up identical `.bak-*` files.
 *
 * @param options - `settingsPath`, `block`, `stamp`, `backup`.
 * @returns `{settingsPath, backup, changed}`.
 * @throws when the settings file does not exist — the shell tool refuses too,
 *   because creating a settings file from nothing would guess at the profile's
 *   other configuration.
 */
export function writeRouteSettings({
  settingsPath,
  block,
  stamp = new Date().toISOString().replace(/[:.]/g, "-"),
  backup = true,
} = {}) {
  if (!settingsPath || !existsSync(settingsPath)) {
    throw new Error(`no DSH settings file at ${settingsPath ?? "(no path)"}`);
  }
  const before = readFileSync(settingsPath, "utf8");
  return replaceSettingsFile({
    settingsPath,
    before,
    after: applyRouteToSettings(before, block),
    stamp,
    backup,
  });
}

/**
 * Back the file up and write the new text, unless nothing changed.
 *
 * Shared by the route block writer and the default-model repair so a refresh and
 * a repair cannot disagree about whether a no-op makes a `.bak-*`.
 */
function replaceSettingsFile({ settingsPath, before, after, stamp, backup }) {
  if (after === before) return { settingsPath, backup: null, changed: false };
  let backupPath = null;
  if (backup) {
    backupPath = `${settingsPath}.bak-${stamp}`;
    copyFileSync(settingsPath, backupPath);
  }
  writeFileSync(settingsPath, after);
  return { settingsPath, backup: backupPath, changed: true };
}

/** The model ids the `llm-pi-ai` block in these lines serves, in file order. */
function servedIdsInLines(lines) {
  const ids = [];
  let inside = false;
  for (const line of lines) {
    if (line.startsWith("llm-pi-ai:")) {
      inside = true;
      continue;
    }
    if (!inside) continue;
    if (line.trim() !== "" && !line.startsWith(" ") && !line.startsWith("\t")) break;
    const id = line.match(/^\s+-\s+id:\s*(\S+)\s*$/);
    if (id) ids.push(id[1]);
  }
  return ids;
}

/**
 * Point `agent-default-model` at a model the refreshed route actually serves.
 *
 * AUD-163's repair ran inside the settings-service branch, because that is the
 * branch the pinned harness takes. A profile that composes no settings service
 * takes the file branch instead (`index.js` calls `registerRoute`), and there
 * nothing kept the default in step either: the refresh rewrote `llm-pi-ai`, the
 * picker showed the live model, and every turn still went out with the old id and
 * came back `UNKNOWN_MODEL`. The same failure, one branch over.
 *
 * Read from the file rather than handed a list, because it has to work after
 * either writer — the shell tool or the built-in generator — has just refreshed
 * that same block, and the block is what says which models exist now.
 *
 * As narrow as the service repair, and for the same reason: this edits somebody
 * else's profile. A default naming a served model is a choice, a default
 * belonging to another provider is a choice, and a default that was never set is
 * the installer's job. Only a reference this repository wrote and can prove dead
 * is rewritten.
 *
 * @param settingsText - the settings file's text.
 * @param options - `provider`, the name this route writes under.
 * @returns `{text, status, reason, from, to}`; `text` is unchanged unless
 *   `status` is `repaired`.
 */
export function applyDefaultModelToSettings(
  settingsText,
  { provider = ROUTE_DEFAULTS.provider } = {},
) {
  const lines = splitKeepingEnds(String(settingsText).replace(/\r\n|\r/g, "\n"));
  const ids = servedIdsInLines(lines);
  if (ids.length === 0) {
    return { text: String(settingsText), status: "skipped", reason: "the route serves no models" };
  }
  let inside = false;
  let blockProvider = null;
  let modelLine = null;
  for (const [index, line] of lines.entries()) {
    if (line.startsWith("agent-default-model:")) {
      inside = true;
      continue;
    }
    if (!inside) continue;
    if (line.trim() !== "" && !line.startsWith(" ") && !line.startsWith("\t")) break;
    const providerLine = line.match(/^\s+provider:\s*(\S+)\s*$/);
    if (providerLine) blockProvider = providerLine[1];
    const modelMatch = line.match(/^(\s+model:\s*)(\S+)(\s*)$/);
    if (modelMatch && modelLine === null)
      modelLine = { index, value: modelMatch[2], indent: modelMatch[1] };
  }
  if (!inside) {
    return { text: String(settingsText), status: "skipped", reason: "no default model is set" };
  }
  if (modelLine === null) {
    return { text: String(settingsText), status: "skipped", reason: "the block names no model" };
  }
  if (blockProvider !== null && blockProvider !== provider) {
    return {
      text: String(settingsText),
      status: "kept",
      reason: `the default belongs to ${blockProvider}`,
    };
  }
  if (ids.includes(modelLine.value)) {
    return { text: String(settingsText), status: "kept", reason: "it names a served model" };
  }
  const out = lines.slice();
  out[modelLine.index] = `${modelLine.indent}${ids[0]}\n`;
  return { text: out.join(""), status: "repaired", from: modelLine.value, to: ids[0] };
}

/**
 * Read the settings file, repair a dead default in it, and write it back.
 *
 * @param options - `settingsPath`, `provider`, `stamp`, `backup`.
 * @returns `{status, …}` from {@link applyDefaultModelToSettings}, or `skipped`
 *   when there is no file to read.
 */
export function repairDefaultModelSettings({
  settingsPath,
  provider = ROUTE_DEFAULTS.provider,
  stamp = new Date().toISOString().replace(/[:.]/g, "-"),
  backup = true,
} = {}) {
  if (!settingsPath || !existsSync(settingsPath)) {
    return { status: "skipped", reason: `no settings file at ${settingsPath ?? "(no path)"}` };
  }
  const before = readFileSync(settingsPath, "utf8");
  const { text, status, reason, from, to } = applyDefaultModelToSettings(before, { provider });
  if (status !== "repaired") return { status, reason };
  const written = replaceSettingsFile({ settingsPath, before, after: text, stamp, backup });
  return { status, from, to, backup: written.backup };
}

/** Whether a path is a file this process may execute. */
export function defaultIsExecutable(path) {
  try {
    if (!statSync(path).isFile()) return false;
    accessSync(path, constants.X_OK);
    return true;
  } catch {
    return false;
  }
}

/** Whether a path is a directory. */
export function defaultIsDirectory(path) {
  try {
    return statSync(path).isDirectory();
  } catch {
    return false;
  }
}

/** The first `name` on `env.PATH`, or null. Injectable so tests need no real PATH. */
export function pathLookup(name, env = process.env, isExecutable = defaultIsExecutable) {
  for (const directory of String(env.PATH ?? "").split(delimiter)) {
    if (directory === "") continue;
    const candidate = join(directory, name);
    if (isExecutable(candidate)) return candidate;
  }
  return null;
}

/**
 * Find a TinyTitan server binary.
 *
 * Explicit config wins, then `TINYTITAN_SERVER`, then PATH, then the checkout's
 * release build. The checkout is last on purpose: it is the copy the installer
 * leaves behind, and the catalogue package this fallback exists for usually has
 * no `.build/` at all.
 *
 * There is deliberately no app-bundle candidate. The project has no GUI app, so
 * `~/Applications/TinyTitan.app` cannot exist, and a probe for it would only
 * make a failure harder to read.
 *
 * @param options - `explicit`, `env`, `repoRoot`, `isExecutable`.
 * @returns the path, or null when nothing is found.
 */
export function findServerBinary({
  explicit,
  env = process.env,
  repoRoot,
  isExecutable = defaultIsExecutable,
} = {}) {
  const candidates = [];
  if (explicit) candidates.push(String(explicit));
  if (env.TINYTITAN_SERVER) candidates.push(String(env.TINYTITAN_SERVER));
  const onPath = pathLookup(SERVER_BINARY_NAME, env, isExecutable);
  if (onPath !== null) candidates.push(onPath);
  if (repoRoot) {
    candidates.push(join(repoRoot, ".build", "arm64-apple-macosx", "release", SERVER_BINARY_NAME));
    candidates.push(join(repoRoot, ".build", "release", SERVER_BINARY_NAME));
  }
  for (const candidate of candidates) {
    if (isExecutable(candidate)) return candidate;
  }
  return null;
}

/**
 * Find the directory of installed models.
 *
 * Explicit config wins, then `TINYTITAN_MODELS_DIR`, then `<repoRoot>/models`.
 * There is deliberately no bundle guess: models live beside the checkout, so a
 * wrong directory would describe nothing rather than describe it wrongly.
 *
 * @param options - `explicit`, `env`, `repoRoot`, `isDirectory`.
 * @returns the path, or null when nothing is found.
 */
export function findModelsDir({
  explicit,
  env = process.env,
  repoRoot,
  isDirectory = defaultIsDirectory,
} = {}) {
  const candidates = [];
  if (explicit) candidates.push(String(explicit));
  if (env.TINYTITAN_MODELS_DIR) candidates.push(String(env.TINYTITAN_MODELS_DIR));
  if (repoRoot) candidates.push(join(repoRoot, "models"));
  for (const candidate of candidates) {
    if (isDirectory(candidate)) return candidate;
  }
  return null;
}

/**
 * Discover a server, read its catalog, and refresh the settings route.
 *
 * The catalog command loads no model: it only enumerates what is installed, so
 * this is safe to run at every harness boot.
 *
 * @param options - resolved config (`port`, `provider`, `repoRoot`, `dshHome`,
 *   optional `serverBinary`/`modelsDir`), plus injectable `env`, `run`, `log`,
 *   `isExecutable`, `isDirectory`, `stamp` and `backup` for tests.
 * @returns `{status, detail, …}` with status `written-self-contained`, `missing`
 *   or `failed`.
 */
export function collectRoute({
  port,
  provider,
  context,
  maxTokens,
  reasoning,
  repoRoot,
  serverBinary,
  modelsDir,
  env = process.env,
  run = execFileSync,
  log = () => {},
  isExecutable = defaultIsExecutable,
  isDirectory = defaultIsDirectory,
} = {}) {
  const directory = findModelsDir({ explicit: modelsDir, env, repoRoot, isDirectory });
  if (directory === null) {
    const detail = "no models directory (set modelsDir or TINYTITAN_MODELS_DIR)";
    log(`dsh-tinytitan: ${detail}; leaving the llm-pi-ai route as it is`);
    return { status: "missing", detail, serverBinary: null, modelsDir: null };
  }
  const binary = findServerBinary({ explicit: serverBinary, env, repoRoot, isExecutable });

  // The server is the authority on the catalog while it can answer. A profile
  // that installed models but has not built the server yet has nothing to ask,
  // so the folder is read directly -- the same rules, mirrored in
  // `catalog-scan.js` and pinned against the binary's own output by the tests.
  // A binary that exists but fails is treated the same way: a working picker
  // beats a stale route, and the reason goes to the log.
  let catalog = null;
  let source = "folder";
  if (binary !== null) {
    try {
      const stdout = run(binary, ["--catalog", "--models-dir", directory], {
        encoding: "utf8",
        stdio: ["ignore", "pipe", "pipe"],
      });
      const parsed = JSON.parse(String(stdout));
      if (parsed !== null && typeof parsed === "object" && Array.isArray(parsed.models)) {
        catalog = parsed;
        source = "server";
      } else {
        log('dsh-tinytitan: the server catalog carried no "models" list; reading models/ directly');
      }
    } catch (error) {
      const detail =
        String(error?.stderr ?? error?.message ?? error)
          .trim()
          .split("\n")[0] || "the server catalog could not be read";
      log(`dsh-tinytitan: the server catalog failed (${detail}); reading models/ directly`);
    }
  } else {
    log(
      "dsh-tinytitan: no TinyTitan server binary (set serverBinary or TINYTITAN_SERVER); " +
        "reading models/ directly",
    );
  }
  if (catalog === null) {
    const scanned = scanModelsFolder(directory, { env });
    for (const skip of scanned.skipped) {
      log(`dsh-tinytitan: skipping ${skip.path}: ${skip.reason}`);
    }
    catalog = { models: scanned.models };
  }
  const rows = catalogRows(catalog.models);
  if (rows.length === 0) {
    const detail = `the catalog describes no servable install under ${directory}`;
    log(`dsh-tinytitan: ${detail}; leaving the llm-pi-ai route as it is`);
    return { status: "failed", detail, serverBinary: binary, modelsDir: directory };
  }
  const block = buildBlock(rows, { port, provider, context, maxTokens, reasoning });
  return { status: null, block, rows, source, serverBinary: binary, modelsDir: directory };
}

/**
 * Refresh the route **file** with the built-in generator.
 *
 * The catalogue/self-contained path: a settings file the harness reads. Kept
 * for older harnesses and installs with no running settings service; DSH 0.2.0
 * uses {@link applyRouteThroughSettings} instead, because it removed
 * `settings.yaml`.
 *
 * @param options - {@link collectRoute} options plus `settingsPath`/`dshHome`,
 *   and injectable `stamp`/`backup` for tests.
 * @returns `{status, detail, …}` with status `written-self-contained`, `missing`
 *   or `failed`.
 */
export function generateRoute({
  settingsPath,
  dshHome,
  stamp = new Date().toISOString().replace(/[:.]/g, "-"),
  backup = true,
  ...options
} = {}) {
  const collected = collectRoute(options);
  if (collected.status) return collected;
  const { block, rows, source, serverBinary, modelsDir } = collected;

  const path = settingsPath ?? join(String(dshHome ?? ""), "settings.yaml");
  if (!existsSync(path)) {
    const detail = `no DSH settings file at ${path}`;
    options.log?.(`dsh-tinytitan: ${detail}; leaving the llm-pi-ai route as it is`);
    return {
      status: "missing",
      detail,
      serverBinary,
      modelsDir,
      settingsPath: path,
    };
  }
  try {
    const written = writeRouteSettings({ settingsPath: path, block, stamp, backup });
    const detail = written.changed ? "written" : "already current";
    options.log?.(
      `dsh-tinytitan: route refreshed with the built-in generator ` +
        `(${rows.length} model(s) from the ${source}, ${detail})`,
    );
    return {
      status: "written-self-contained",
      detail,
      serverBinary,
      modelsDir,
      settingsPath: path,
      models: rows.length,
      source,
      backup: written.backup,
    };
  } catch (error) {
    const detail = String(error?.message ?? error)
      .trim()
      .split("\n")[0];
    options.log?.(`dsh-tinytitan: route refresh failed: ${detail}`);
    return { status: "failed", detail, serverBinary, modelsDir, settingsPath: path };
  }
}

/**
 * Refresh the route through the harness `settings` service.
 *
 * DSH 0.2.0 removed `settings.yaml`; the active configuration lives in the
 * profile patch and is edited through the `settings` service. This runs the
 * same discovery as {@link generateRoute}, parses the generated block, and
 * merges it into the `llm-pi-ai` entry, so installing or removing a model
 * reaches the picker without a settings file. `settings.update` deep-merges
 * objects but replaces arrays, so a removed model's entry does not linger.
 *
 * The picker is not the whole story: `agent-default-model` is a separate
 * namespace, and refreshing only the provider left a stale default behind that
 * no longer named a served model — every turn then failed with `UNKNOWN_MODEL`.
 * {@link ensureDefaultModel} repairs that reference on the same pass.
 *
 * @param options - {@link collectRoute} options plus `settings` (the service).
 * @returns `{status, …}`; `applied`, or a discovery/`failed` result.
 */
export async function applyRouteThroughSettings({ settings, ...options } = {}) {
  const collected = collectRoute(options);
  if (collected.status) return collected;
  const { block, rows, source } = collected;
  try {
    if (!settings || typeof settings.update !== "function") {
      throw new Error("no settings service");
    }
    const [{ default: yaml }] = await Promise.all([import("js-yaml")]);
    const parsed = yaml.load(block);
    const namespace = parsed !== null && typeof parsed === "object" ? Object.keys(parsed)[0] : null;
    if (namespace === null || typeof parsed[namespace] !== "object") {
      throw new Error("the generated block is not a single settings section");
    }
    await settings.update(namespace, parsed[namespace]);
    const defaultModel = await ensureDefaultModel({
      settings,
      ids: rows.map((row) => row.id),
      provider: options.provider ?? ROUTE_DEFAULTS.provider,
      log: options.log,
    });
    options.log?.(
      `dsh-tinytitan: route applied through the settings service ` +
        `(${rows.length} model(s) from the ${source})`,
    );
    return { status: "applied", models: rows.length, source, defaultModel };
  } catch (error) {
    const detail = String(error?.message ?? error)
      .trim()
      .split("\n")[0];
    options.log?.(`dsh-tinytitan: route refresh failed: ${detail}`);
    return { status: "failed", detail };
  }
}

/** The namespace the harness opens a new session with. */
export const DEFAULT_MODEL_SETTINGS_NS = "agent-default-model";

/**
 * Repair a default model that names an install the route no longer serves.
 *
 * `tools/dsh_local.sh` writes `agent-default-model` once, at install time, and
 * the harness migrates it into the profile patch. Nothing kept it in step with
 * the route afterwards, so changing which model the server runs left the window
 * pointed at an id the provider does not have: the picker showed the live model,
 * the composer sent the old one, and every turn died with
 * `pi-ai provider "tinytitan" has no configured model <id>` / `UNKNOWN_MODEL`.
 * Discovered by a real `smoke` run, not by reading — the page looks configured.
 *
 * Deliberately narrow, because this runs inside somebody else's profile:
 * a default that names a served model is a choice and is left alone, however
 * surprising; a default belonging to another provider is a choice too; and a
 * default that was never set is the installer's job, not a defect to invent.
 * Only a reference this repository wrote and can now prove is dead gets rewritten.
 *
 * @param options - `settings` (the harness service), the served `ids`, the
 *   provider name, and a logger.
 * @returns `{status, …}` — `repaired`, `kept`, `skipped` or `failed`.
 */
export async function ensureDefaultModel({
  settings,
  ids = [],
  provider = ROUTE_DEFAULTS.provider,
  log = () => {},
} = {}) {
  if (settings === undefined || settings === null || typeof settings.update !== "function") {
    return { status: "skipped", reason: "no settings service" };
  }
  if (ids.length === 0) return { status: "skipped", reason: "the route serves no models" };
  let current;
  try {
    const forms = typeof settings.describe === "function" ? settings.describe() : [];
    const row = Array.isArray(forms)
      ? forms.find((form) => form?.ns === DEFAULT_MODEL_SETTINGS_NS)
      : null;
    current = row?.user;
  } catch (error) {
    const detail = String(error?.message ?? error)
      .trim()
      .split("\n")[0];
    return { status: "skipped", reason: detail };
  }
  const chosen = typeof current?.model === "string" ? current.model : "";
  if (chosen === "") return { status: "skipped", reason: "no default model is set" };
  if (chosen !== "" && ids.includes(chosen)) return { status: "kept", model: chosen };
  const chosenProvider = typeof current?.provider === "string" ? current.provider : "";
  if (chosenProvider !== "" && chosenProvider !== provider) {
    return { status: "kept", provider: chosenProvider, model: chosen };
  }
  const model = ids[0];
  try {
    await settings.update(DEFAULT_MODEL_SETTINGS_NS, { provider, model });
  } catch (error) {
    const detail = String(error?.message ?? error)
      .trim()
      .split("\n")[0];
    log(`dsh-tinytitan: could not repoint the default model: ${detail}`);
    return { status: "failed", reason: detail };
  }
  log(
    `dsh-tinytitan: default model ${provider}/${model} replaces ${chosen}, which no longer serves`,
  );
  return { status: "repaired", model, replaced: chosen };
}
