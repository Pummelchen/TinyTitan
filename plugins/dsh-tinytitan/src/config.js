/**
 * Plugin config: every field optional, with the environment as the fallback.
 *
 * A plugin that has to be configured before it does anything is one nobody
 * installs; the defaults are what a local TinyTitan server on the default port
 * needs, and every one of them can be overridden per profile.
 *
 * @module dsh-tinytitan/config
 */
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";

/** The checkout this plugin was authored in: `<repo>/plugins/dsh-tinytitan/src/config.js`. */
export const REPO_ROOT = resolve(fileURLToPath(new URL("../../..", import.meta.url)));

/** The DSH home, which is where the settings file and presets live. */
export function defaultDshHome() {
  return process.env.DSH_HOME || join(homedir(), ".dsh");
}

/** The route this plugin keeps current. */
export const DEFAULT_PROVIDER = "tinytitan";

/**
 * The route's default reasoning level when nothing overrides it.
 *
 * The level a route declares is what the harness asks for on every call that
 * names none of its own, so it has to agree with how the server was started.
 * `medium` (thinking on) against a server running `--reasoning off` is not a
 * harmless disagreement: a dense Qwen asked to think spends its entire output
 * budget inside the reasoning block and never emits an answer. Measured on the
 * 4B — thinking off: `finish: stop`, content "42", 3 tokens. Thinking on:
 * `finish: length`, 64/64 reasoning tokens, empty content.
 */
export const DEFAULT_REASONING = "medium";

/** The preset it generates, so it never has to touch a person's own. */
export const DEFAULT_PRESET_ID = "tinytitan";

/** The tool this plugin delegates to; its presence identifies a checkout. */
const TOOL = join("tools", "dsh_route.sh");

function hasTool(directory) {
  return directory !== "" && existsSync(join(directory, TOOL));
}

/**
 * Where a profile installed this plugin from, when it recorded a `file:` spec.
 *
 * A plugin installed into a profile is a hardlinked copy, so this module's own
 * path no longer points into the checkout it came from. The profile's
 * `package.json` does: its `dsh-tinytitan` dependency is the `file:` path the person
 * installed (`<repo>/plugins/dsh-tinytitan`), and the checkout root is above it.
 *
 * @param moduleUrl - this module's URL.
 * @returns candidate roots, nearest first.
 */
function fileDependencyRoots(moduleUrl) {
  // <profile>/node_modules/dsh-tinytitan/src/config.js -> <profile>/package.json
  const profile = resolve(fileURLToPath(new URL("../../../", moduleUrl)));
  const roots = [];
  try {
    const manifest = JSON.parse(readFileSync(join(profile, "package.json"), "utf8"));
    for (const spec of Object.values(manifest.dependencies ?? {})) {
      if (typeof spec !== "string" || !spec.startsWith("file:")) continue;
      roots.push(resolve(fileURLToPath(new URL(spec))));
    }
  } catch {
    // No profile manifest (running out of the checkout, or a copied package):
    // the other candidates still apply.
  }
  return roots;
}

/**
 * Find the TinyTitan checkout, by looking for `tools/dsh_route.sh`.
 *
 * Explicit config wins, then `TINYTITAN_REPO`, then this module's own location
 * (which works when the plugin runs out of the checkout), then the profile's
 * `file:` dependency (which works when it runs from a profile), then the
 * working directory.
 *
 * @param options - `explicit` root, `env`, `moduleUrl`.
 * @returns `{root, found}` — `root` is the first candidate even when none matched.
 */
export function findRepoRoot({
  explicit,
  env = process.env,
  moduleUrl = import.meta.url,
  cwd = process.cwd(),
} = {}) {
  const candidates = [];
  if (explicit) candidates.push(String(explicit));
  if (env.TINYTITAN_REPO) candidates.push(String(env.TINYTITAN_REPO));
  let directory = resolve(fileURLToPath(new URL(".", moduleUrl)));
  for (let level = 0; level < 6; level += 1) {
    candidates.push(directory);
    const parent = dirname(directory);
    if (parent === directory) break;
    directory = parent;
  }
  for (const dependency of fileDependencyRoots(moduleUrl)) {
    let root = dependency;
    for (let level = 0; level < 3; level += 1) {
      candidates.push(root);
      root = dirname(root);
    }
  }
  candidates.push(cwd);
  for (const candidate of candidates) {
    if (hasTool(candidate)) return { root: candidate, found: true };
  }
  return { root: candidates[0] ?? "", found: false };
}

/**
 * How long to wait after the last `models/` change before refreshing.
 *
 * An install writes thousands of files, so the route is rebuilt once per quiet
 * period rather than once per event.
 */
function resolveDebounce(value) {
  const ms = Number(value ?? 2000);
  if (!Number.isFinite(ms) || ms < 0) {
    throw new Error(`dsh-tinytitan: watchDebounceMs must be milliseconds, got ${value}`);
  }
  return ms;
}

/**
 * Resolve the plugin config.
 * @param config - the raw row config.
 * @returns the resolved config, with every field a value.
 */
export function resolveConfig(config = {}) {
  const port = Number(config.port ?? process.env.TINYTITAN_PORT ?? 8080);
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error(`dsh-tinytitan: port must be a port number, got ${config.port}`);
  }
  const provider = String(config.provider ?? DEFAULT_PROVIDER).trim();
  if (provider.length === 0) throw new Error("dsh-tinytitan: provider must not be empty");
  // TINYTITAN_REASONING is the fallback, for the same reason the port has one:
  // `tools/dsh_route.sh` defaults to `medium`, so a route refreshed at boot
  // would silently put back a level the caller had chosen against.
  const reasoning = String(
    config.reasoning ?? process.env.TINYTITAN_REASONING ?? DEFAULT_REASONING,
  ).trim();
  if (reasoning.length === 0) throw new Error("dsh-tinytitan: reasoning must not be empty");
  const presetId = String(config.presetId ?? DEFAULT_PRESET_ID).trim();
  if (presetId.length === 0) throw new Error("dsh-tinytitan: presetId must not be empty");
  const repoRoot = findRepoRoot({ explicit: config.repoRoot, env: process.env });
  // The self-contained generator's discovery order starts at explicit config and
  // then the environment; resolving both here keeps route.js free of the
  // fallback rules. Empty strings mean "not set", as with every other field.
  const serverBinary = config.serverBinary || process.env.TINYTITAN_SERVER || null;
  const modelsDir = config.modelsDir || process.env.TINYTITAN_MODELS_DIR || null;
  return {
    port,
    provider,
    reasoning,
    presetId,
    repoRoot: repoRoot.root,
    repoFound: repoRoot.found,
    dshHome: String(config.dshHome ?? defaultDshHome()),
    serverBinary: serverBinary === null ? null : String(serverBinary),
    modelsDir: modelsDir === null ? null : String(modelsDir),
    // Force the built-in generator even where tools/dsh_route.sh exists. The
    // checkout tool stays the default source of truth, so this is opt-in.
    selfContained: config.selfContained === true,
    // Three switches, so an operator can take one job at a time:
    registerRoute: config.registerRoute !== false,
    writeCompactionPreset: config.writeCompactionPreset !== false,
    // Keep watching `models/` after boot, so installing or deleting a model
    // reaches the picker without restarting the harness. A machine where the
    // folder is on a slow volume, or an operator who would rather refresh by
    // hand, can turn it off.
    watchModels: config.watchModels !== false,
    watchDebounceMs: resolveDebounce(config.watchDebounceMs),
    // Set the registry's `selectedDefault` only while the profile names none: an
    // explicit choice is never overwritten. Off means "register the preset and
    // let the person pick it on the Agent presets page".
    //
    // 0.1.6's `adoptDefaultPreset` (re-point the current default preset's stock
    // compaction row) has no 0.2.0 equivalent: presets are declared rows now, and
    // re-pointing a shipped one means freezing its whole plugin list in the
    // profile patch — the drift the generated preset exists to avoid. The
    // switch is gone rather than silently ignored.
    setDefaultWhenUnset: config.setDefaultWhenUnset !== false,
    log: typeof config.log === "function" ? config.log : null,
  };
}
