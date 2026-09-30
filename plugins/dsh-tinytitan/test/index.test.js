/**
 * The wiring in `apply`: which route-refresh path boot and the watcher take, and
 * when the preset may become the default.
 *
 * Both are regressions this file exists to pin. Boot wrote the route through the
 * `settings` service while the watcher still shelled out to the 0.1.6 file path,
 * which dies on a 0.2.0 home with no `settings.yaml` — so a model installed while
 * the harness ran never reached the picker. And `agent-presets.default` from
 * 0.1.6 had no 0.2.0 successor in the port, so a profile that had chosen nothing
 * silently fell back to the shipped `standard` preset and lost the quiet
 * compaction.
 *
 * `apply` takes injectable seams for exactly this: none of the fakes below touch
 * the filesystem, a socket, or a harness.
 */
import assert from "node:assert/strict";
import test from "node:test";

import { apply, PRESET_SETTINGS_NS, routeRefresher, SUPPORTED_DSH_VERSION } from "../src/index.js";

/** The version gate reads this before anything else runs. */
function withSupportedVersion(body) {
  const previous = process.env.DSH_VERSION;
  process.env.DSH_VERSION = SUPPORTED_DSH_VERSION;
  try {
    return body();
  } finally {
    if (previous === undefined) delete process.env.DSH_VERSION;
    else process.env.DSH_VERSION = previous;
  }
}

/** A context whose `inject` resolves immediately and whose `get` returns services. */
function context(services = {}) {
  const ctx = {
    get: (name) => services[name],
    inject: (_names, run) => run(ctx),
    on: () => {},
    logger: {},
  };
  return ctx;
}

/** A settings service stub that records writes. */
function settings(forms = []) {
  const updates = [];
  return {
    updates,
    describe: () => forms,
    update: async (ns, patch) => updates.push([ns, patch]),
  };
}

test("the refresher prefers the settings service over the file path", () => {
  const applied = [];
  const legacy = [];
  const refresh = routeRefresher({
    resolved: { port: 8080 },
    ctx: context({ settings: { update: async () => {} } }),
    apply: (options) => applied.push(options),
    legacy: (options) => legacy.push(options),
  });
  refresh();
  assert.equal(applied.length, 1);
  assert.equal(legacy.length, 0);
  assert.equal(applied[0].port, 8080);
});

test("a harness without the settings service still gets the file path", () => {
  const applied = [];
  const legacy = [];
  const refresh = routeRefresher({
    resolved: { port: 8080 },
    ctx: context({}),
    apply: (options) => applied.push(options),
    legacy: (options) => legacy.push(options),
  });
  refresh();
  assert.equal(applied.length, 0);
  assert.equal(legacy.length, 1);
});

test("a refresh that throws or rejects is logged, never raised", async () => {
  const lines = [];
  const throwing = routeRefresher({
    resolved: {},
    ctx: context({ settings: { update: async () => {} } }),
    log: (message) => lines.push(message),
    apply: () => {
      throw new Error("sync boom");
    },
  });
  throwing();
  const rejecting = routeRefresher({
    resolved: {},
    ctx: context({ settings: { update: async () => {} } }),
    log: (message) => lines.push(message),
    apply: async () => {
      throw new Error("async boom");
    },
  });
  rejecting();
  await new Promise((resolve) => setImmediate(resolve));
  assert.ok(lines.some((line) => line.includes("sync boom")));
  assert.ok(lines.some((line) => line.includes("async boom")));
});

/** Run `apply` with a captured watcher and a service bag. */
function runApply({ services = {}, config = {}, deps = {} } = {}) {
  const lines = [];
  const watches = [];
  const result = withSupportedVersion(() =>
    apply(
      context(services),
      {
        modelsDir: "/models",
        registerRoute: true,
        writeCompactionPreset: true,
        watchModels: true,
        log: (message) => lines.push(message),
        ...config,
      },
      {
        watch: (options) => {
          watches.push(options);
          return { watching: true, close() {} };
        },
        applyRoute: async () => ({ status: "applied" }),
        legacyRoute: () => ({ status: "written" }),
        preset: async () => () => {},
        env: {},
        ...deps,
      },
    ),
  );
  return { lines, result, watches };
}

test("a models/ change mid-session takes the same path as boot", () => {
  const applied = [];
  const legacy = [];
  const { watches } = runApply({
    services: { settings: { update: async () => {} } },
    deps: {
      applyRoute: (options) => applied.push(options),
      legacyRoute: (options) => legacy.push(options),
    },
  });
  assert.equal(watches.length, 1, "the watcher must be started");
  // Boot already refreshed once; the watcher must take the settings branch too.
  assert.equal(applied.length, 1);
  watches[0].refresh();
  assert.equal(applied.length, 2);
  assert.equal(legacy.length, 0);
});

test("the watcher's refresh is the same callback boot used", () => {
  const applied = [];
  const { watches } = runApply({
    services: { settings: { update: async () => {} } },
    deps: { applyRoute: (options) => applied.push(options) },
  });
  const bootOptions = applied[0];
  watches[0].refresh();
  assert.deepEqual(applied[1], bootOptions);
});

test("the preset becomes the default after it registers", async () => {
  const bag = settings([{ ns: PRESET_SETTINGS_NS, user: {} }]);
  const { lines } = runApply({ services: { settings: bag } });
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(bag.updates, [[PRESET_SETTINGS_NS, { selectedDefault: "tinytitan" }]]);
  assert.ok(lines.some((line) => line.includes("default agent preset")));
});

test("a preset that did not register never becomes the default", async () => {
  const bag = settings([{ ns: PRESET_SETTINGS_NS, user: {} }]);
  let asked = 0;
  runApply({
    services: { settings: bag },
    deps: {
      preset: async () => {
        asked += 1;
        return null;
      },
    },
  });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(asked, 1, "the registrar is still the one that runs");
  assert.equal(bag.updates.length, 0, "an id the registry does not know must not be selected");
});

test("setDefaultWhenUnset: false registers the preset and nothing else", async () => {
  const bag = settings([{ ns: PRESET_SETTINGS_NS, user: {} }]);
  const { lines } = runApply({
    services: { settings: bag },
    config: { setDefaultWhenUnset: false },
  });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(bag.updates.length, 0);
  assert.equal(
    lines.some((line) => line.includes("default agent preset")),
    false,
  );
});

test("an explicit selection survives a boot", async () => {
  const bag = settings([{ ns: PRESET_SETTINGS_NS, user: { selectedDefault: "ptc" } }]);
  const { lines } = runApply({ services: { settings: bag } });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(bag.updates.length, 0);
  assert.equal(
    lines.some((line) => line.includes("default agent preset")),
    false,
  );
});
