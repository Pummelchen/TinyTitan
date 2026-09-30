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
 * @module dsh-tinytitan/preset
 */
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";

/** The backend module a preset row names. */
export const COMPACTION_BACKEND = "dsh-tinytitan/backend";

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
 * @param options - `plugins` (the standard list, from {@link standardPlugins}),
 *   `maxTokens`, `backend`.
 * @returns a new entry list.
 */
export function buildTinytitanPlugins({
  plugins,
  maxTokens = 32768,
  backend = COMPACTION_BACKEND,
} = {}) {
  if (!Array.isArray(plugins)) {
    throw new TypeError("dsh-tinytitan: buildTinytitanPlugins needs a plugins array");
  }
  const drop = new Set(CHAT_NOISE_ROWS);
  const walk = (rows) =>
    rows.flatMap((row) => {
      if (row === null || typeof row !== "object") return [row];
      if (drop.has(row.id)) return [];
      if (row.id === "compaction-basic") {
        return [{ ...row, name: backend, config: { ...(row.config ?? {}), maxTokens } }];
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
 * @param options - preset identity, token budget, backend and logger.
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
    backend = COMPACTION_BACKEND,
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
      backend,
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
