/**
 * Register this plugin's agent preset from the shipped `standard` composition.
 *
 * DSH 0.2.0 replaced the harness-home preset files
 * (`~/.dsh/.agent-presets/<id>/agent.cordis.yml`, which no longer exist and are
 * no longer read) with *declared* presets: a row of
 * `@deepseek-ai/dsh-agent-preset` whose `config.plugins` is the agent-plane
 * entry list, mounted by `@deepseek-ai/dsh-agent-preset-registry`. The shipped
 * `standard` preset is now `@deepseek-ai/dsh-web-app/presets/standard.patch.yml`.
 *
 * So instead of copying a preset file and editing it line by line, this module
 * reads the shipped composition, drops the chat-noise rows the original plugin
 * removed, points the compaction row at this plugin's backend, and calls
 * `ctx.agentPresets.register(...)` — the same call the declarative plugin makes,
 * with the same `!!js` expression objects the loader hands it. Keeping the
 * shipped composition as the source of truth means the generated preset tracks
 * the upstream `standard` on every harness upgrade rather than drifting.
 *
 * {@link ensureDefaultPreset} is the other half of the old plugin's job: a
 * profile that has chosen no preset of its own starts on this one. An explicit
 * choice is never overwritten.
 *
 * @module dsh-tinytitan/preset
 */
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";

/** The backend module a preset row names. */
export const COMPACTION_BACKEND = "dsh-tinytitan/backend";

/** The profile entry whose `selectedDefault` names the preset a new session uses. */
export const PRESET_SETTINGS_NS = "agent-preset-registry";

/**
 * The rows that make an agent preset expensive for a prompt box, and are not
 * tool definitions — so `TINYTITAN_STRIP_CLI_PROMPT` cannot remove them.
 *
 * `agent-instructions` injects `AGENTS.md`/`CLAUDE.md` into every turn (up to
 * 64 KB by its own config), `skill-filesystem` supplies a skill tree and
 * `tool-skill` injects the skill catalog that lists it. Measured on a nine-word
 * question through the browser window, the harness request that reached the
 * engine after stripping was 4,222 tokens across four user messages; the
 * question itself is about ten of them. That is prefill the user pays for on
 * every turn, and at the 35B's measured ~65 tok/s it is a minute of it.
 */
export const CHAT_NOISE_ROWS = ["agent-instructions", "skill-filesystem", "tool-skill"];

/**
 * Rounds a `ralph` run may take when autonomy enables the loop.
 *
 * The shipped row is disabled and carries 64; the harness enforces its own
 * ceiling on a call override, so this is only the default this preset writes.
 */
export const AUTONOMY_DEFAULT_ROUNDS = 64;

/**
 * The prompt policy autonomy appends to the preset's `persona` row.
 *
 * It exists because the two halves of "keep going" are in different places: the
 * goal driver (and `ralph`) supply the *continuation*, while nothing in the
 * harness can stop a model from politely stopping to ask. This text is the
 * policy half. It is written to be read by the model, not by a person, so it
 * states the failure modes it is preventing rather than describing a feature.
 */
export const AUTONOMY_INSTRUCTIONS = [
  "Autonomy: you are working toward one objective, and the person wants it finished rather than discussed.",
  "- Do not ask questions you can answer yourself, and do not wait for a decision you can make. Choose the best technical solution, state it as an explicit assumption, and continue. The only things worth stopping for are a missing credential, spending the person's money, or destroying data they did not ask you to touch.",
  '- Keep going until the work is solved, tested and verified. "Done" means the change is in place, the relevant tests and gates pass, and you have shown the evidence: the command, its output and its exit status. Never report success without evidence, and never stop at a plan when the objective asks for the change.',
  "- The workspace is the authority. Re-read the code and the task state instead of trusting this conversation.",
  "- If you are genuinely blocked, say exactly what is blocked, what you already tried, and what a person would have to do. That is the only reason to stop.",
  "- When the work will not fit in one session, call the `ralph` tool with the objective and a round budget: it runs fresh-agent rounds against the immutable objective, with the shared workspace as the memory between them.",
].join("\n");

/** The shipped standard composition, as a patch file rather than a preset file. */
const STANDARD_PRESET = "@deepseek-ai/dsh-web-app/presets/standard.patch.yml";

/**
 * The shipped `standard` composition's path.
 * @returns the resolved path.
 */
export function standardPresetPath() {
  const require = createRequire(import.meta.url);
  return require.resolve(STANDARD_PRESET);
}

/**
 * Read the `preset-standard` row's child plugin list from the shipped patch.
 *
 * YAML is loaded with the harness's own `entryListSchema` so `!!js` rows become
 * the `{ __jsExpr }` objects the loader and the preset registry evaluate.
 * `js-yaml` and `@deepseek-ai/cordis-plugin-include` are imported lazily: this
 * module is imported at boot in the harness (where both resolve) and in the
 * package's own test run (where only the pure builder is exercised).
 *
 * @param options - `path` override, injectable for tests.
 * @returns the agent-plane entry list.
 */
export async function standardPlugins({ path = standardPresetPath() } = {}) {
  const [{ default: yaml }, { entryListSchema }] = await Promise.all([
    import("js-yaml"),
    import("@deepseek-ai/cordis-plugin-include"),
  ]);
  const document = yaml.load(readFileSync(path, "utf8"), { schema: entryListSchema });
  const entries = Array.isArray(document) ? document : [];
  const row = entries
    .flatMap((entry) => (entry && Array.isArray(entry.insert) ? entry.insert : []))
    .find((entry) => entry && entry.id === "preset-standard");
  const plugins = row?.config?.plugins;
  if (!Array.isArray(plugins)) {
    throw new Error(`dsh-tinytitan: no preset-standard plugin list in ${path}`);
  }
  return plugins;
}

/**
 * Build this plugin's plugin list from a standard composition.
 *
 * Pure: the chat-noise rows are dropped wherever they sit (top level or a
 * nested group) and the `compaction-basic` row is repointed at this plugin's
 * backend. Every other row, including its `!!js` expressions, is returned
 * unchanged.
 *
 * `headroomTokens` is written only when the caller supplies one. The engine's
 * own default is 65536, and it is subtracted from the message budget before the
 * `thresholdRatio` is applied: `threshold = min(window x ratio, window - maxTokens
 * - headroomTokens)`. On a declared window of 262,144 with our 32,768-token cap
 * that holds the trigger at ~62% of the window instead of the documented 80%,
 * and on a window below ~98,000 the pressure budget goes negative, which the
 * engine reports once and then never compacts at all. Leaving the field out
 * keeps the harness's own policy; `compactionHeadroomTokens` lets an operator
 * put the ratio back in charge (0) or pick their own guard.
 *
 * @param options - `plugins` (the standard list, from {@link standardPlugins}),
 *   `maxTokens`, `headroomTokens`, `backend`.
 * @returns a new entry list.
 */
export function buildTinytitanPlugins({
  plugins,
  maxTokens = 32768,
  headroomTokens = null,
  backend = COMPACTION_BACKEND,
  autonomy = false,
  autonomyRounds = AUTONOMY_DEFAULT_ROUNDS,
  autonomySuppressQuestions = false,
} = {}) {
  if (!Array.isArray(plugins)) {
    throw new TypeError("dsh-tinytitan: buildTinytitanPlugins needs a plugins array");
  }
  const drop = new Set(CHAT_NOISE_ROWS);
  // Autonomy instructs the model not to ask; removing the tool makes it
  // impossible instead of discouraged. That is a bigger hammer than it looks:
  // the shipped plan-mode instructions tell the model to use
  // `ask_user_question` for user-owned choices, so a session that needs plan
  // mode should keep the row and rely on the policy text alone.
  if (autonomy && autonomySuppressQuestions) drop.add("tool-ask-user");
  const walk = (rows) =>
    rows.flatMap((row) => {
      if (row === null || typeof row !== "object") return [row];
      if (drop.has(row.id)) return [];
      if (row.id === "compaction-basic") {
        const config = { ...(row.config ?? {}), maxTokens };
        if (headroomTokens !== null) config.headroomTokens = headroomTokens;
        return [{ ...row, name: backend, config }];
      }
      if (autonomy && row.id === "persona") {
        const config = { ...(row.config ?? {}) };
        config.suffix = [config.suffix, AUTONOMY_INSTRUCTIONS].filter(Boolean).join("\n\n");
        return [{ ...row, config }];
      }
      // `ralph` ships disabled: a fresh-agent loop is not what most sessions
      // want. Autonomy is the profile saying that this one does.
      if (autonomy && row.id === "tool-ralph") {
        return [
          { ...row, disabled: false, config: { ...(row.config ?? {}), maxRounds: autonomyRounds } },
        ];
      }
      if (row.group === true && Array.isArray(row.config)) {
        return [{ ...row, config: walk(row.config) }];
      }
      return [row];
    });
  return walk(plugins);
}

/**
 * Register this plugin's agent preset with the running harness.
 *
 * The `agentPresets` service may not exist yet, so callers reach this through
 * `ctx.inject(["agentPresets"], ...)`; when it is genuinely absent (a profile
 * with no registry) this logs and returns rather than throwing, because the
 * harness must boot with or without a preset.
 *
 * A refusal is a **return, never a throw**: this plugin mounts into somebody
 * else's profile, so a bad preset must not take the boot down.
 *
 * @param ctx - the harness context; `ctx.agentPresets.register` is the sink.
 * @param options - preset identity, token budget, compaction headroom, backend
 *   and logger.
 * @returns the registration promise, or `null` when no registry is available.
 */
export async function registerTinytitanPreset(
  ctx,
  {
    presetId = "tinytitan",
    name = "TinyTitan",
    description = "The standard agent with compaction that does not think.",
    order = 10,
    maxTokens = 32768,
    headroomTokens = null,
    backend = COMPACTION_BACKEND,
    autonomy = false,
    autonomyRounds = AUTONOMY_DEFAULT_ROUNDS,
    autonomySuppressQuestions = false,
    plugins,
    log = () => {},
  } = {},
) {
  const registry = ctx?.agentPresets;
  if (registry === undefined || registry === null || typeof registry.register !== "function") {
    log("dsh-tinytitan: the agentPresets service is unavailable; skipping the tinytitan preset");
    return null;
  }
  try {
    const built = buildTinytitanPlugins({
      plugins: plugins ?? (await standardPlugins()),
      maxTokens,
      headroomTokens,
      backend,
      autonomy,
      autonomyRounds,
      autonomySuppressQuestions,
    });
    const disposer = await registry.register({
      id: presetId,
      name,
      description,
      order,
      plugins: built,
    });
    log(`dsh-tinytitan: registered the ${presetId} agent preset`);
    return disposer;
  } catch (error) {
    log(
      `dsh-tinytitan: preset registration failed: ${error instanceof Error ? error.message : error}`,
    );
    return null;
  }
}

/**
 * Make this plugin's preset the one a new session starts on, while nobody has
 * chosen one.
 *
 * 0.1.6 had no notion of a selected preset at the settings plane: the plugin
 * wrote `agent-presets.default` into `settings.yaml`, and only when that file
 * named no default, so an explicit choice was never overwritten. 0.2.0 moved the
 * choice onto the `agent-preset-registry` row's volatile `selectedDefault`, which
 * is the same field the Agent presets page writes, so the same rule is applied
 * here: set it only when the profile carries no selection of its own.
 *
 * "No selection" is read through the settings service's own projection
 * (`describe().user`) rather than by parsing the patch, so a value inherited
 * from a bundle layer does not count as a choice. A read that fails leaves the
 * selection alone — this runs inside somebody else's profile, and losing a
 * person's choice is worse than not offering a default.
 *
 * @param options - `settings` (the harness service), the preset id, and a logger.
 * @returns `{status}` — `set`, `kept`, `skipped` or `failed`.
 */
export async function ensureDefaultPreset({
  settings,
  presetId = "tinytitan",
  log = () => {},
} = {}) {
  if (settings === undefined || settings === null || typeof settings.update !== "function") {
    return { status: "skipped", reason: "no settings service" };
  }
  let chosen;
  try {
    const forms = typeof settings.describe === "function" ? settings.describe() : [];
    const row = Array.isArray(forms) ? forms.find((form) => form?.ns === PRESET_SETTINGS_NS) : null;
    chosen = row?.user?.selectedDefault;
  } catch (error) {
    return { status: "skipped", reason: describeError(error) };
  }
  if (typeof chosen === "string" && chosen !== "") {
    return { status: "kept", selectedDefault: chosen };
  }
  try {
    await settings.update(PRESET_SETTINGS_NS, { selectedDefault: presetId });
  } catch (error) {
    log(`dsh-tinytitan: could not make ${presetId} the default preset: ${describeError(error)}`);
    return { status: "failed", reason: describeError(error) };
  }
  log(`dsh-tinytitan: ${presetId} is the default agent preset (nothing else was chosen)`);
  return { status: "set", selectedDefault: presetId };
}

/** One line from an unknown throwable, for a log line. */
function describeError(error) {
  return String(error instanceof Error ? error.message : error)
    .trim()
    .split("\n")[0];
}
