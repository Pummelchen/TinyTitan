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
import { DEFAULT_AUTO_GOAL_ROUNDS } from "./keep-going.js";
import { AUTONOMY_DEFAULT_ROUNDS } from "./preset.js";
import {
  DEFAULT_HANDOFF_HOPS,
  DEFAULT_HANDOFF_MAX_CHILDREN,
  DEFAULT_HANDOFF_WINDOW_RATIO,
} from "./handoff.js";

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
 * The compaction engine's headroom, when the operator names one.
 *
 * `null` (the default) leaves the field out of the generated preset, so the
 * harness's own default stands. A number is passed through: the engine subtracts
 * it from the message budget before applying `thresholdRatio`, so 0 puts that
 * ratio back in charge on a wide window and a small value keeps a guard on a
 * narrow one — see `buildTinytitanPlugins` for the arithmetic and the trap this
 * exists for.
 */
function resolveCompactionHeadroom(value) {
  if (value === undefined || value === null) return null;
  const tokens = Number(value);
  if (!Number.isInteger(tokens) || tokens < 0) {
    throw new Error(
      `dsh-tinytitan: compactionHeadroomTokens must be a whole number of tokens, got ${value}`,
    );
  }
  return tokens;
}

/**
 * A route token count the operator has declared, or `null` for "say nothing".
 *
 * `null` is load-bearing: both route writers fall back to the launcher's pin
 * (262144 window, 32768 cap — `ROUTE_DEFAULTS` in generate.js, and
 * `dsh_route.sh`'s own defaults), so naming nothing keeps the historical block
 * byte-for-byte. The value exists because a route narrowed by hand is not: the
 * next refresh — boot, or the `models/` watcher — rewrites the whole section, so
 * without this knob the only way to keep a smaller window is to never let the
 * plugin refresh again. `compactionHeadroomTokens` is the other half of that
 * choice, and `README.md` carries the arithmetic that makes a narrow window
 * dangerous.
 */
function resolveRouteTokenCount(field, value) {
  if (value === undefined || value === null || value === "") return null;
  const tokens = Number(value);
  if (!Number.isSafeInteger(tokens) || tokens <= 0) {
    throw new Error(`dsh-tinytitan: ${field} must be a positive token count, got ${value}`);
  }
  return tokens;
}

/**
 * Rounds an auto-created goal may run before the harness blocks it.
 *
 * Only read when `autoGoal` is on. The cap is deliberately much smaller than the
 * goal service's own default (256): it bounds prompts a person did not mark as
 * long-running work, while `/goal` keeps the harness default for the ones they
 * did. A positive integer is required — the value is a promise about how much
 * unattended work a stray prompt may buy, so it is never derived or guessed.
 */
function resolveAutoGoalRounds(value) {
  if (value === undefined || value === null || value === "") return DEFAULT_AUTO_GOAL_ROUNDS;
  const rounds = Number(value);
  if (!Number.isSafeInteger(rounds) || rounds <= 0) {
    throw new Error(`dsh-tinytitan: autoGoalRounds must be a positive integer, got ${value}`);
  }
  return rounds;
}

/**
 * Rounds one `ralph` run may take, when autonomy enables the loop.
 *
 * Only read when `autonomy` is on. The harness enforces its own ceiling on a
 * call override, so this is a policy default rather than a hard stop; a positive
 * integer is still required, because "keep going" without a number is not a
 * budget anyone can reason about.
 */
function resolveAutonomyRounds(value) {
  if (value === undefined || value === null || value === "") return AUTONOMY_DEFAULT_ROUNDS;
  const rounds = Number(value);
  if (!Number.isSafeInteger(rounds) || rounds <= 0) {
    throw new Error(`dsh-tinytitan: autonomyRounds must be a positive integer, got ${value}`);
  }
  return rounds;
}

/**
 * How many times one objective may be handed to a fresh context.
 *
 * Only read when `handoff` is on. Each hop is a child session with its own
 * context and model calls, so the chain needs a number: "keep going until it is
 * done" is a policy, not an unbounded budget.
 */
function resolveHandoffHops(value) {
  if (value === undefined || value === null || value === "") return DEFAULT_HANDOFF_HOPS;
  const hops = Number(value);
  if (!Number.isSafeInteger(hops) || hops <= 0) {
    throw new Error(`dsh-tinytitan: handoffHops must be a positive integer, got ${value}`);
  }
  return hops;
}

/**
 * The prompt-token budget that starts a handoff, or `null` to derive it.
 *
 * Only read when `handoff` is on. Unset means auto: `handoffWindowRatio` of the
 * routed model's declared context window, held at least the reserve above the
 * window's end — so a 256K route moves on at 131,072 (the reserve binds), a
 * 1,000,000-token route at 600,000, and a 1,048,576-token one at 629,145. An
 * explicit positive integer pins one number.
 * `0` is refused on purpose: a zero budget hands off at the first turn, which is
 * how an unbounded chain was found once already.
 */
function resolveHandoffAtTokens(value) {
  if (value === undefined || value === null || value === "") return null;
  const tokens = Number(value);
  if (!Number.isSafeInteger(tokens) || tokens <= 0) {
    throw new Error(`dsh-tinytitan: handoffAtTokens must be a positive integer, got ${value}`);
  }
  return tokens;
}

/**
 * The fraction of the routed model's context window the auto budget uses.
 *
 * Only read when `handoff` is on and `handoffAtTokens` is unset. It must stay
 * below the compaction trigger, or compaction continues the session in place
 * instead — and that trigger is a different fraction per shape, ~0.625 of a
 * 262,144-token window with this plugin's preset but 0.8 of a 1M one, so the
 * ratio clears the narrowest shape or it clears nothing.
 */
function resolveHandoffWindowRatio(value) {
  if (value === undefined || value === null || value === "") return DEFAULT_HANDOFF_WINDOW_RATIO;
  const ratio = Number(value);
  if (!Number.isFinite(ratio) || ratio <= 0 || ratio > 1) {
    throw new Error(
      `dsh-tinytitan: handoffWindowRatio must be above 0 and at most 1, got ${value}`,
    );
  }
  return ratio;
}

/**
 * How many handoff children may be alive at once.
 *
 * Only read when `handoff` is on. The per-goal cap bounds one chain; this is the
 * process-wide ceiling that makes a runaway impossible even across several
 * goals, which is the shape the one observed runaway took.
 */
function resolveHandoffMaxChildren(value) {
  if (value === undefined || value === null || value === "") return DEFAULT_HANDOFF_MAX_CHILDREN;
  const children = Number(value);
  if (!Number.isSafeInteger(children) || children <= 0) {
    throw new Error(`dsh-tinytitan: handoffMaxChildren must be a positive integer, got ${value}`);
  }
  return children;
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
  // Config, then environment, then nothing — the same order as `reasoning` and
  // for the same reason recorded above it: a refresh that is told neither writes
  // the writer's default over a window and cap the operator narrowed.
  const context = resolveRouteTokenCount(
    "context",
    config.context ?? process.env.TINYTITAN_CONTEXT,
  );
  const maxTokens = resolveRouteTokenCount(
    "maxTokens",
    config.maxTokens ?? process.env.TINYTITAN_MAX_TOKENS,
  );
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
    context,
    maxTokens,
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
    // The generated preset's compaction row carries no headroom unless one is
    // named here, so the harness's own default (65536) stands.
    compactionHeadroomTokens: resolveCompactionHeadroom(config.compactionHeadroomTokens),
    // Off by default: it changes what every manual prompt means (each one
    // becomes a persistent objective that continues by itself), which is a
    // choice an operator makes once for a profile rather than a default.
    autoGoal: config.autoGoal === true,
    autoGoalRounds: resolveAutoGoalRounds(config.autoGoalRounds),
    // Autonomy is the preset half of "work it until it is done": the policy text
    // that stops the model asking, the fresh-agent `ralph` loop, and optionally
    // the removal of the question tool. Off by default, because it changes what
    // every prompt in the profile means.
    autonomy: config.autonomy === true,
    autonomyRounds: resolveAutonomyRounds(config.autonomyRounds),
    autonomySuppressQuestions: config.autonomySuppressQuestions === true,
    // Handoff is what makes "no matter how long" survive a context wall: an
    // unfinished objective moves to a fresh child context before the window
    // fills. Off by default — it starts sessions on its own.
    handoff: config.handoff === true,
    handoffHops: resolveHandoffHops(config.handoffHops),
    // null means auto: handoffWindowRatio of the routed model's context window.
    handoffAtTokens: resolveHandoffAtTokens(config.handoffAtTokens),
    handoffWindowRatio: resolveHandoffWindowRatio(config.handoffWindowRatio),
    handoffMaxChildren: resolveHandoffMaxChildren(config.handoffMaxChildren),
    log: typeof config.log === "function" ? config.log : null,
  };
}
