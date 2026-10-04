/**
 * `dsh-tinytitan` — the DeepSeek Harness side of running models from a local TinyTitan
 * server.
 *
 * It does two things, both at boot, both idempotent:
 *
 * 1. **Keeps the route current.** The harness's own `llm-pi-ai` adapter serves
 *    these models — there is no adapter here — and this checkout's
 *    `tools/dsh_route.sh` is the one place that turns the installed models into
 *    that route's block. Running it means the model picker follows `models/`
 *    instead of a copy someone typed once. A catalogue install has no checkout
 *    to run, so `generate.js` produces the same block in-process from the
 *    server's catalog; the shell tool stays authoritative wherever it exists.
 * 2. **Mounts a compaction backend that does not think.** Compaction and session
 *    titles name no reasoning level, so they inherit the route's default; on a
 *    local thinking model that spends a summariser's own output cap on thinking
 *    and costs tens of seconds on every new session's title. The preset this
 *    plugin generates points that row at `dsh-tinytitan/backend`.
 *
 * Neither job patches the harness or replaces its adapter, so a harness upgrade
 * cannot desynchronise a copied protocol implementation — and when the two
 * upstream asks in this repository's `docs/dsh-upstream-asks.md` land, the second
 * job becomes unnecessary.
 *
 * @module dsh-tinytitan
 */
import { resolveConfig } from "./config.js";
import { applyRouteThroughSettings, findModelsDir } from "./generate.js";
import { registerRoute } from "./route.js";
import { ensureDefaultPreset, registerTinytitanPreset } from "./preset.js";
import { watchModels } from "./models-watch.js";
import { dshVersion, supportDecision } from "./support.js";
import { installAutoGoal } from "./keep-going.js";

/** Plugin name, as the harness registry shows it. */
export const name = "dsh-tinytitan";

export {
  DEFAULT_PRESET_ID,
  DEFAULT_PROVIDER,
  REPO_ROOT,
  findRepoRoot,
  resolveConfig,
} from "./config.js";
export { registerRoute, routeScript } from "./route.js";
export { DEFAULT_DEBOUNCE_MS, watchModels } from "./models-watch.js";
export { scanModelsFolder } from "./catalog-scan.js";
export {
  applyRouteThroughSettings,
  applyRouteToSettings,
  catalogRows,
  collectRoute,
  findModelsDir,
  findServerBinary,
  generateBlock,
  generateRoute,
  writeRouteSettings,
} from "./generate.js";
export {
  buildTinytitanPlugins,
  CHAT_NOISE_ROWS,
  ensureDefaultPreset,
  PRESET_SETTINGS_NS,
  registerTinytitanPreset,
  standardPlugins,
  standardPresetPath,
} from "./preset.js";
export {
  AUXILIARY_PURPOSES,
  auxiliaryThinkingOff,
  createAuxiliaryQuietCompaction,
} from "./compaction.js";
export {
  SUPPORTED_DSH_VERSION,
  dshVersion,
  packageFrom,
  siblingPackage,
  supportDecision,
} from "./support.js";
export {
  DEFAULT_AUTO_GOAL_ROUNDS,
  GOALS_SERVICE,
  installAutoGoal,
  isDirectHuman,
  messageText,
} from "./keep-going.js";

/**
 * The one route-refresh path, shared by boot and the `models/` watcher.
 *
 * DSH 0.2.0 removed `settings.yaml`; the route lives in the profile patch and is
 * written through the `settings` service. On that harness the service is always
 * present, so prefer it, and keep the file/shell path for a harness that has no
 * such service. Boot and the watcher must take the *same* branch: a refresh that
 * only worked at boot is how a model installed while the harness ran stopped
 * reaching the picker — the legacy path dies on a home with no settings file.
 *
 * `apply` and `legacy` are injectable so a test can pin which branch is taken.
 *
 * @param options - `resolved` config, the context, a logger, and the two writers.
 * @returns a zero-argument callback the watcher can call on every quiet period.
 */
export function routeRefresher({
  resolved,
  ctx,
  log = () => {},
  apply = applyRouteThroughSettings,
  legacy = registerRoute,
} = {}) {
  return (scoped = ctx) => {
    const settings = scoped?.get?.("settings");
    try {
      if (settings && typeof settings.update === "function") {
        void Promise.resolve(apply({ ...resolved, settings, log })).catch((error) => {
          log(
            `dsh-tinytitan: route refresh threw: ${error instanceof Error ? error.message : error}`,
          );
        });
      } else {
        legacy({ ...resolved, log });
      }
    } catch (error) {
      log(
        `dsh-tinytitan: route registration threw: ${error instanceof Error ? error.message : error}`,
      );
    }
  };
}
/**
 * Report a refusal where an operator will actually see it.
 *
 * `console.error` is the load-bearing sink, and that is a finding rather than a
 * preference: the harness collects plugin log records into its startup log and
 * prints them **only when the boot itself fails**, and its startup exporter is
 * registered with `levels: { default: 2 }`, so a host-logger `info` record never
 * reaches anything at all. A healthy boot therefore prints none of them — and a
 * refusal nobody can read is indistinguishable from a plugin that silently
 * stopped working, which is the failure this gate exists to prevent.
 *
 * The other two sinks keep the record in the deployment's own log when it has
 * one; an explicit `config.log` wins for tests and for operators who wired one.
 *
 * @param ctx - the harness context.
 * @param config - the raw row config.
 * @param message - the refusal line.
 */
function refuse(ctx, config, message) {
  if (typeof config.log === "function") config.log(message);
  if (typeof ctx?.logger?.error === "function") ctx.logger.error(message);
  console.error(message);
}

/**
 * Run the plugin.
 * @param ctx - the harness context; services are resolved through it.
 * @param config - the row config; see {@link resolveConfig}.
 * @param deps - injectable seams for tests: the `models/` watcher, the two route
 *   writers, the preset registrar, and the environment the models directory is
 *   resolved from. The harness passes none of them.
 */
export function apply(ctx, config = {}, deps = {}) {
  const {
    watch = watchModels,
    applyRoute = applyRouteThroughSettings,
    legacyRoute = registerRoute,
    preset = registerTinytitanPreset,
    env = process.env,
  } = deps;
  // The gate runs first, and before `resolveConfig`, so a harness this plugin
  // does not support cannot reach a single write. A refusal is a return rather
  // than a throw: the harness must boot, every other plugin must load, and
  // removing this one must leave nothing to undo.
  const harness = dshVersion();
  const decision = supportDecision(harness);
  if (!decision.run) {
    refuse(ctx, config, decision.refusal);
    return { refused: true, version: harness };
  }
  const log =
    typeof config.log === "function"
      ? config.log
      : (message) => {
          if (typeof ctx?.logger?.info === "function") ctx.logger.info(message);
          else console.log(message);
        };
  const resolved = resolveConfig(config);
  const refreshRoute = routeRefresher({
    resolved,
    ctx,
    log,
    apply: applyRoute,
    legacy: legacyRoute,
  });
  // A read-only home, a missing checkout or a failed write must not take the
  // profile down: the harness still works, only this convenience does not.
  if (resolved.registerRoute) {
    try {
      if (typeof ctx?.inject === "function") ctx.inject(["settings"], refreshRoute);
      else refreshRoute();
    } catch (error) {
      log(
        `dsh-tinytitan: route registration threw: ${error instanceof Error ? error.message : error}`,
      );
    }
  }
  if (resolved.writeCompactionPreset) {
    // The registry that owns the `agentPresets` service may activate after this
    // row, so register through `ctx.inject` when the context offers it and fall
    // back to a direct call otherwise. The preset is built from the shipped
    // `standard` composition and registers under `resolved.presetId`.
    const register = (scoped = ctx) => {
      // `registerTinytitanPreset` never rejects: it reports a failure through
      // `log`. The wrapper only guards the synchronous gap before its first
      // await (a malformed config or a throwing `register` call).
      void preset(scoped, {
        presetId: resolved.presetId,
        headroomTokens: resolved.compactionHeadroomTokens,
        log,
      })
        .then((registered) => {
          // Only a preset that actually registered may become the default: a
          // selection naming an id the registry does not know would break every
          // new session, and this order is what makes that impossible.
          if (registered === null || registered === undefined) return null;
          if (!resolved.setDefaultWhenUnset) return null;
          return ensureDefaultPreset({
            settings: scoped?.get?.("settings"),
            presetId: resolved.presetId,
            log,
          });
        })
        .catch((error) => {
          log(
            `dsh-tinytitan: preset registration threw: ${error instanceof Error ? error.message : error}`,
          );
        });
    };
    try {
      if (typeof ctx?.inject === "function") ctx.inject(["agentPresets"], register);
      else register();
    } catch (error) {
      log(
        `dsh-tinytitan: preset registration threw: ${error instanceof Error ? error.message : error}`,
      );
    }
  }
  // The keep-going switch: a manual prompt becomes a goal so the harness's own
  // round driver continues it, exactly as `/goal <prompt>` would. Off unless the
  // profile asks; a harness with no goal service logs why and boots unchanged.
  try {
    installAutoGoal({ ctx, resolved, log });
  } catch (error) {
    log(`dsh-tinytitan: autoGoal setup threw: ${error instanceof Error ? error.message : error}`);
  }
  // Boot writes the route once; a folder that changes during the session has to
  // reach the picker too, because installing a model and using it are the same
  // sitting. The watcher is closed on disposal so it cannot outlive the plugin.
  if (resolved.registerRoute && resolved.watchModels) {
    try {
      const modelsDir = findModelsDir({
        explicit: resolved.modelsDir,
        env,
        repoRoot: resolved.repoRoot,
      });
      const handle = watch({
        modelsDir,
        debounceMs: resolved.watchDebounceMs,
        log,
        // The same refresher boot used: on 0.2.0 that is the settings service,
        // not the shell tool, which needs a settings file that no longer exists.
        refresh: () => refreshRoute(ctx),
      });
      if (handle.watching && typeof ctx?.on === "function") {
        ctx.on("dispose", () => handle.close());
      }
    } catch (error) {
      log(`dsh-tinytitan: models watch threw: ${error instanceof Error ? error.message : error}`);
    }
  }
}

export default apply;
