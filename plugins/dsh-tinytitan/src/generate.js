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
import * as fs from "node:fs";
import { basename, delimiter, dirname, join } from "node:path";

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
 * A catalog row this writer cannot describe, with a reason safe to print.
 *
 * Only this type is a skip. A row that trips something else — a getter that
 * throws, a value that cannot be stringified — is a defect in the catalog or in
 * this module, and it must surface rather than look like one more malformed
 * install that quietly disappears from the picker.
 */
class CatalogRowError extends Error {}

/**
 * The shell's `field()`: a value that can print on one YAML line.
 *
 * The parser that feeds the route skips a model whose id, name or path is
 * missing or multi-line, so a malformed install cannot split the block across
 * lines. `null` is rejected here where Python would print `None`; a catalog is
 * JSON and a null there is a bug, not a name. Anything else is kept — quoting it
 * is {@link yamlScalar}'s job, because stripping punctuation would rename the
 * model in the picker.
 */
function textField(value, field = "field") {
  const text = value === undefined || value === null ? "" : String(value);
  if (text === "" || /[\t\r\n]/.test(text)) {
    throw new CatalogRowError(`the ${field} is empty or multi-line`);
  }
  return text;
}

/**
 * A plain scalar that YAML reads back as exactly this string, or null.
 *
 * Conservative on purpose: YAML's plain style also hosts indicators, comments
 * and the core schema's non-string tokens, so the writer only leaves a value
 * bare when it cannot be read as anything else. A letter or `_` first (so no
 * `-`, `[`, `{`, `&`, `*`, `!`, `|`, `>`, `%`, `@`, quote or digit — and no
 * number, timestamp or sexagesimal), no `:`, `#`, `\`, quote or control
 * character inside, and no trailing space, which YAML strips.
 *
 * `on`, `off`, `yes` and `no` stay bare: YAML 1.2's core schema — the one
 * js-yaml and the harness's own parser implement — reads them as strings, and
 * the effort ladder already emits `off:` as a key on every route.
 */
const PLAIN_SCALAR = /^[A-Za-z_][A-Za-z0-9 _.,()+/@-]*$/;

/** The plain scalars YAML's core schema resolves to a boolean or null. */
const NON_STRING_SCALARS = new Set([
  "true",
  "True",
  "TRUE",
  "false",
  "False",
  "FALSE",
  "null",
  "Null",
  "NULL",
]);

/**
 * The one spelling a served model id may have.
 *
 * The block writes it on one `- id:` line, the default-model repair reads it
 * back with `\S+`, and `tools/dsh_route.sh --from-server` already refuses
 * anything outside this set, so the catalog path refuses the same rows rather
 * than writing an id the picker and the repair cannot agree on.
 */
const MODEL_ID = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;

/**
 * A string as one YAML scalar: bare when that is unambiguous, else quoted.
 *
 * The quoted form is a YAML double-quoted scalar with `\`, `"` and the control
 * characters escaped and everything else — including non-ASCII — left as it is.
 * `tools/dsh_route.sh`'s `yaml_scalar` produces the same bytes for the same
 * input, which is what keeps the two writers byte-identical; the escaping is
 * per byte on the shell side and per code point here, and those agree because
 * only ASCII controls are escaped.
 *
 * @param value - the string (or anything `String()` describes) to serialize.
 * @returns the scalar, without a trailing newline.
 */
export function yamlScalar(value) {
  const text = String(value);
  if (PLAIN_SCALAR.test(text) && !text.endsWith(" ") && !NON_STRING_SCALARS.has(text)) {
    return text;
  }
  let quoted = '"';
  for (const character of text) {
    const code = character.codePointAt(0);
    if (character === "\\") quoted += "\\\\";
    else if (character === '"') quoted += '\\"';
    else if (code < 0x20 || code === 0x7f) quoted += `\\x${code.toString(16).padStart(2, "0")}`;
    else quoted += character;
  }
  return `${quoted}"`;
}

/**
 * A whole number the route declares, as the block prints it.
 *
 * `port`, `context` and `maxTokens` land on their own YAML line, so a string
 * carrying `#`, `:` or a newline would rewrite the rest of the block. Resolved
 * config already hands these over as numbers; this is the chokepoint that keeps
 * `generateBlock` safe for any other caller. A canonical decimal string is
 * accepted because that is the shape the shell tool's arguments take.
 */
function integerField(field, value, { min = 1, max = Number.MAX_SAFE_INTEGER } = {}) {
  const text = String(value);
  const number =
    typeof value === "number" ? value : /^(?:0|[1-9][0-9]*)$/.test(text) ? Number(text) : NaN;
  if (!Number.isSafeInteger(number) || number < min || number > max) {
    throw new Error(`${field} must be a whole number, got ${text}`);
  }
  return String(number);
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
    throw new CatalogRowError("thinking is not iterable");
  }
  return [
    ...LEVEL_ORDER.filter((level) => listed.includes(level)),
    ...listed.filter((level) => !LEVEL_ORDER.includes(level)),
  ];
}

/**
 * A row's id, when it is one this writer would accept, for a diagnostic.
 *
 * A validated id is the one piece of a rejected row that is safe to print: it is
 * a plain token, it is what the picker and the shell parser would call the model,
 * and it lets an operator find the install. The reason never carries any other
 * field, so a name with a credential in it cannot reach the log.
 */
function rowIdentity(model) {
  if (model === null || typeof model !== "object") return "";
  const id = model.id === undefined || model.id === null ? "" : String(model.id);
  return MODEL_ID.test(id) ? ` (id ${id})` : "";
}

/**
 * A row that is already this module's normalized row, not a catalog row.
 *
 * `catalogRows` is the only place a raw row becomes
 * `{id, name, path, backend, family, engines, levels}`. That result is *not* a
 * catalog row — it has no `quant` and no `thinking` — so feeding it back used to
 * return nothing, and `generateBlock(catalogRows(rows))` was an empty route that
 * read exactly like a catalog of no installs. That is a caller error rather than
 * a malformed install, so it is refused loudly instead of skipped.
 */
export class CatalogRowShapeError extends Error {
  constructor(position) {
    super(
      `catalog: row ${position} is already a catalogRows() result; pass the catalog's ` +
        "raw models array (the shape --catalog prints), not the normalized rows",
    );
    this.name = "CatalogRowShapeError";
  }
}

/** Whether one entry is a normalized row rather than a catalog row. */
function normalizedRow(model) {
  return (
    model !== null &&
    typeof model === "object" &&
    !Array.isArray(model) &&
    Array.isArray(model.levels) &&
    model.quant === undefined &&
    model.thinking === undefined
  );
}

/**
 * Normalize a catalog's `models` array into the rows the block is built from.
 *
 * Two shapes exist and they are not interchangeable:
 *
 * - a **catalog row**, what `TinyTitanServer --catalog`, the folder scan and the
 *   harness's own listing produce: `{id, name, path, backend, quant, family?,
 *   engines?, thinking?, size_gb?}`;
 * - a **normalized row**, what this function returns:
 *   `{id, name, path, backend, family, engines, levels}`.
 *
 * Passing a normalized row back in throws `CatalogRowShapeError`: a second pass
 * would otherwise look like a catalog in which every install is malformed.
 *
 * GPU installs first, then CPU ones, mirroring the shell parser: a server
 * serves one resident model at a time, so the route only orders the picker, and
 * GPU is what a launcher starts by default. A row missing a field the block
 * needs is skipped rather than failing the whole route.
 *
 * @param models - the catalog's raw `models` array.
 * @param options - `log`, the channel a refused row's diagnostic goes to.
 * @returns one `{id, name, path, backend, family, engines, levels}` per servable model.
 * @throws {CatalogRowShapeError} when an entry is already a normalized row.
 */
export function catalogRows(models, { log = () => {} } = {}) {
  if (!Array.isArray(models)) throw new Error('catalog: "models" is not a list');
  const gpu = [];
  const cpu = [];
  for (const [position, model] of models.entries()) {
    if (normalizedRow(model)) throw new CatalogRowShapeError(position + 1);
    let row;
    try {
      if (model === null || typeof model !== "object") {
        throw new CatalogRowError("the entry is not an object");
      }
      const backend = textField(model.backend, "backend");
      if (backend !== "gpu" && backend !== "cpu") {
        throw new CatalogRowError("the backend is not gpu or cpu");
      }
      const quant = model.quant;
      if (
        quant === undefined ||
        quant === null ||
        quant === "" ||
        !Number.isFinite(Number(quant))
      ) {
        throw new CatalogRowError("quant is not a number");
      }
      // Not used by the block, but the shell parser formats it and skips the
      // model when it cannot; dropping the same rows keeps the two in step.
      const size = model.size_gb;
      if (size !== undefined && size !== null && Number.isNaN(Number(size))) {
        throw new CatalogRowError("size_gb is not a number");
      }
      // The block's `- id:` line is one `\S+` token to the picker, to the
      // default-model repair in this file and to the shell tool's parser, so an
      // id that is not one cannot be represented; a name is free-form and is
      // quoted by `yamlScalar` instead.
      const id = textField(model.id, "id");
      if (!MODEL_ID.test(id)) throw new CatalogRowError("the id is not a plain model id");
      row = {
        id,
        name: textField(model.name, "name"),
        path: textField(model.path, "path"),
        backend,
        family: textField(model.family || "-", "family"),
        engines: textField(model.engines || backend, "engines"),
        levels: catalogLevels(model.thinking),
      };
    } catch (error) {
      // The shell's parser prints "catalog: skipping …" on stderr and carries
      // on; the route is still useful for the installs that did parse. Only a
      // refusal this module raised is a skip — anything else is a defect.
      if (!(error instanceof CatalogRowError)) throw error;
      log(
        `dsh-tinytitan: skipping catalog row ${position + 1}${rowIdentity(model)}: ${error.message}`,
      );
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
    // A comma is the shell tool's own separator: it joins a row's levels with
    // one and splits them again, so a comma inside a level names two levels
    // there. Splitting here keeps the two writers byte-identical even on that
    // input, which is what "the shell tool is authoritative" means.
    for (const part of String(raw).split(",")) {
      const level = part.replace(/ /g, "");
      if (level === "") continue;
      // The names the ladder is documented in stay bare; anything a template
      // adds is serialized like every other free-form string, as both a key and
      // its wire value.
      if (level === "off") lines.push("            off:");
      else if (level === "on") lines.push("            medium: on");
      else lines.push(`            ${yamlScalar(level)}: ${yamlScalar(level)}`);
    }
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
  const port = integerField("port", options.port ?? ROUTE_DEFAULTS.port, { max: 65535 });
  const provider = yamlScalar(options.provider ?? ROUTE_DEFAULTS.provider);
  const context = integerField("context", options.context ?? ROUTE_DEFAULTS.context);
  const maxTokens = integerField("maxTokens", options.maxTokens ?? ROUTE_DEFAULTS.maxTokens);
  const reasoning = yamlScalar(options.reasoning ?? ROUTE_DEFAULTS.reasoning);

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
      `        - id: ${yamlScalar(row.id)}`,
      `          name: ${yamlScalar(routeLabel(row.name, row.id))}`,
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
 *
 * @param models - the catalog's raw `models` array, as `--catalog` prints it.
 *   The rows `catalogRows` returns are a different shape; passing them throws
 *   `CatalogRowShapeError` instead of building an empty block.
 * @param options - `port`, `provider`, `context`, `maxTokens`, `reasoning`.
 * @returns the block, ending in a newline.
 */
export function generateBlock(models, options = {}) {
  return buildBlock(catalogRows(models, { log: options.log }), options);
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
 * @param options - `settingsPath`, `block`, `stamp`, `backup`, and an injectable
 *   `io` (a `node:fs`-shaped object) so tests can fail a rename or fake another
 *   editor's save without touching a real profile.
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
  io = fs,
} = {}) {
  if (!settingsPath || !io.existsSync(settingsPath)) {
    throw new Error(`no DSH settings file at ${settingsPath ?? "(no path)"}`);
  }
  const before = io.readFileSync(settingsPath, "utf8");
  return replaceSettingsFile({
    settingsPath,
    before,
    after: applyRouteToSettings(before, block),
    stamp,
    backup,
    io,
  });
}

/** How many temporary names to try before giving up on an exclusive create. */
const TEMPORARY_ATTEMPTS = 16;

/** How many `-N` suffixes a backup name may take before it is an error. */
const BACKUP_ATTEMPTS = 1000;

/**
 * A temporary file beside the target, created exclusively.
 *
 * Same directory on purpose: a rename across filesystems is a copy, not an
 * atomic replacement. The name is audit-owned (`.<name>.tmp-<pid>-<stamp>`), so
 * the cleanup below can never remove somebody else's file.
 */
function openTemporary(io, target, stamp) {
  const directory = dirname(target);
  for (let attempt = 0; attempt < TEMPORARY_ATTEMPTS; attempt += 1) {
    const suffix = attempt === 0 ? "" : `-${attempt}`;
    const path = join(directory, `.${basename(target)}.tmp-${process.pid}-${stamp}${suffix}`);
    try {
      return { path, descriptor: io.openSync(path, "wx", 0o600) };
    } catch (error) {
      if (error?.code !== "EEXIST") throw error;
    }
  }
  throw new Error(`could not create a temporary file beside ${target}`);
}

/** A backup name that does not already exist, so two refreshes never collide. */
function backupPathFor(io, settingsPath, stamp) {
  const base = `${settingsPath}.bak-${stamp}`;
  let candidate = base;
  for (let taken = 1; io.existsSync(candidate); taken += 1) {
    if (taken >= BACKUP_ATTEMPTS) throw new Error(`no free backup name beside ${settingsPath}`);
    candidate = `${base}-${taken}`;
  }
  return candidate;
}

/**
 * Replace a settings file atomically, unless nothing changed.
 *
 * Shared by the route block writer and the default-model repair so a refresh and
 * a repair cannot disagree about whether a no-op makes a `.bak-*`, or about how
 * the file is replaced.
 *
 * The sequence is: write the new text to an exclusive same-directory temp file,
 * set the target's own mode on it, fsync and close it, re-read the target and
 * refuse if it is no longer the text this replacement was computed from, copy
 * that exact original to a collision-free backup, and rename the temp over the
 * target. Every failure before the rename leaves the original byte-identical;
 * a failure after it (there is none in this function) would not be a rollback,
 * so nothing here claims to roll back.
 *
 * The re-read is a best-effort compare-and-swap, not a lock: a non-cooperating
 * editor can still save between it and the rename. What it removes is the
 * unbounded window of the old truncate-and-write, which spanned the whole write.
 * A profile shared with an editor that does not cooperate is a case for the
 * editor's own locking, not for a lock file this plugin cannot make anyone take.
 *
 * @param options - `settingsPath`, the `before` text the new text was computed
 *   from, the `after` text, `stamp`, `backup`, and the `io` seam.
 * @returns `{settingsPath, backup, changed}`.
 */
function replaceSettingsFile({ settingsPath, before, after, stamp, backup, io = fs }) {
  if (after === before) return { settingsPath, backup: null, changed: false };
  // Resolve symlinks first, so the rename replaces the file the link points at
  // and the link itself survives; a profile kept in a dotfiles checkout stays a
  // symlink there.
  const target = io.realpathSync(settingsPath);
  const mode = io.statSync(target).mode & 0o7777;
  const temporary = openTemporary(io, target, stamp);
  let descriptor = temporary.descriptor;
  try {
    io.writeFileSync(descriptor, after);
    io.fchmodSync(descriptor, mode);
    io.fsyncSync(descriptor);
    io.closeSync(descriptor);
    descriptor = null;
    const current = io.readFileSync(target, "utf8");
    if (current !== before) {
      throw Object.assign(
        new Error(
          `the settings file at ${settingsPath} changed while the route was being written; ` +
            "nothing was replaced",
        ),
        { code: "ESETTINGSCHANGED" },
      );
    }
    const backupPath = backup ? backupPathFor(io, settingsPath, stamp) : null;
    // The backup is the exact original this replacement is based on, taken
    // after the recheck proved the file still holds those bytes.
    if (backupPath !== null) io.copyFileSync(target, backupPath);
    io.renameSync(temporary.path, target);
    return { settingsPath, backup: backupPath, changed: true };
  } catch (error) {
    // Before the rename the target was never opened for writing; only this
    // process's own temp file is removed. A backup already copied is a copy of
    // the intact original and is left in place.
    if (descriptor !== null) {
      try {
        io.closeSync(descriptor);
      } catch {
        // Already closed by the failed call, or never opened.
      }
    }
    try {
      io.unlinkSync(temporary.path);
    } catch {
      // The rename may have consumed it; nothing else to clean.
    }
    throw error;
  }
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
 * The replacement is the same atomic one the route writer uses
 * ({@link replaceSettingsFile}), so a repair that fails cannot leave a
 * half-written profile either.
 *
 * @param options - `settingsPath`, `provider`, `stamp`, `backup`, and the `io`
 *   seam tests inject.
 * @returns `{status, …}` from {@link applyDefaultModelToSettings}, or `skipped`
 *   when there is no file to read.
 */
export function repairDefaultModelSettings({
  settingsPath,
  provider = ROUTE_DEFAULTS.provider,
  stamp = new Date().toISOString().replace(/[:.]/g, "-"),
  backup = true,
  io = fs,
} = {}) {
  if (!settingsPath || !io.existsSync(settingsPath)) {
    return { status: "skipped", reason: `no settings file at ${settingsPath ?? "(no path)"}` };
  }
  const before = io.readFileSync(settingsPath, "utf8");
  const { text, status, reason, from, to } = applyDefaultModelToSettings(before, { provider });
  if (status !== "repaired") return { status, reason };
  const written = replaceSettingsFile({ settingsPath, before, after: text, stamp, backup, io });
  return { status, from, to, backup: written.backup };
}

/** Whether a path is a file this process may execute. */
export function defaultIsExecutable(path) {
  try {
    if (!fs.statSync(path).isFile()) return false;
    fs.accessSync(path, fs.constants.X_OK);
    return true;
  } catch {
    return false;
  }
}

/** Whether a path is a directory. */
export function defaultIsDirectory(path) {
  try {
    return fs.statSync(path).isDirectory();
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
  diagnostics = null,
  isExecutable = defaultIsExecutable,
  isDirectory = defaultIsDirectory,
} = {}) {
  // Boot and every watcher refresh share one budget, so a still-broken install
  // is named once rather than once per quiet period. The first failure always
  // prints, and a diagnostic that changes (another row, another reason) is new
  // information and prints too — only an exact repeat is dropped.
  const seen = diagnostics instanceof Set ? diagnostics : null;
  const report = (message) => {
    if (seen !== null) {
      if (seen.has(message)) return;
      seen.add(message);
    }
    log(message);
  };
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
      report(`dsh-tinytitan: skipping ${skip.path}: ${skip.reason}`);
    }
    catalog = { models: scanned.models };
  }
  const rows = catalogRows(catalog.models, { log: report });
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
 *   and injectable `stamp`/`backup`/`io` for tests.
 * @returns `{status, detail, …}` with status `written-self-contained`, `missing`
 *   or `failed`.
 */
export function generateRoute({
  settingsPath,
  dshHome,
  stamp = new Date().toISOString().replace(/[:.]/g, "-"),
  backup = true,
  io = fs,
  ...options
} = {}) {
  const collected = collectRoute(options);
  if (collected.status) return collected;
  const { block, rows, source, serverBinary, modelsDir } = collected;

  const path = settingsPath ?? join(String(dshHome ?? ""), "settings.yaml");
  if (!io.existsSync(path)) {
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
    const written = writeRouteSettings({ settingsPath: path, block, stamp, backup, io });
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
 * A route setup that writes `agent-default-model` once, at install time, leaves
 * the harness pointing at it after the migration into the profile patch. Nothing
 * kept it in step with the route afterwards, so changing which model the server runs left the window
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
