/**
 * Building the `tinytitan` agent preset from the shipped `standard` composition.
 *
 * `buildTinytitanPlugins` is the pure half and is tested directly. The parser
 * (`standardPlugins`) needs the harness's own `js-yaml` and
 * `@deepseek-ai/cordis-plugin-include`, so it is exercised when those resolve
 * (inside a harness, or after `npm ci`) and skipped with a reason otherwise.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  AUTONOMY_INSTRUCTIONS,
  buildTinytitanPlugins,
  CHAT_NOISE_ROWS,
  COMPACTION_BACKEND,
  ensureDefaultPreset,
  PRESET_SETTINGS_NS,
  registerTinytitanPreset,
  standardPlugins,
} from "../src/preset.js";

const STANDARD_PATCH = fileURLToPath(new URL("./fixtures/standard-patch.yml", import.meta.url));

/** A minimal standard list, mirroring the shipped shape. */
function standardList() {
  return [
    {
      id: "persona",
      name: "@deepseek-ai/dsh-persona",
      config: { suffix: "Your working directory is {{cwd}}." },
    },
    { id: "agent-instructions", name: "@deepseek-ai/dsh-agent-instructions" },
    { id: "tool-bash", name: "@deepseek-ai/dsh-tool-bash", disabled: { __jsExpr: "x" } },
    {
      id: "compaction",
      name: "cordis:group",
      group: true,
      config: [
        { id: "compaction-basic", name: "@deepseek-ai/dsh-compaction-basic" },
        { id: "command-compact", name: "@deepseek-ai/dsh-command-compact" },
      ],
    },
    {
      id: "delegation",
      name: "cordis:group",
      group: true,
      config: [
        { id: "tool-subagent", name: "@deepseek-ai/dsh-tool-subagent" },
        {
          id: "tool-ralph",
          name: "@deepseek-ai/dsh-tool-ralph",
          disabled: true,
          config: { subagentProvider: "spawn", maxRounds: 64 },
        },
      ],
    },
    { id: "tool-ask-user", name: "@deepseek-ai/dsh-tool-ask-user" },
    { id: "skill-filesystem", name: "@deepseek-ai/dsh-skill-filesystem" },
    { id: "tool-skill", name: "@deepseek-ai/dsh-tool-skill" },
  ];
}

test("every chat-noise row is dropped, including nested ones", () => {
  const built = buildTinytitanPlugins({ plugins: standardList() });
  const ids = [];
  const collect = (rows) =>
    rows.forEach((row) => {
      ids.push(row.id);
      if (Array.isArray(row.config)) collect(row.config);
    });
  collect(built);
  for (const id of CHAT_NOISE_ROWS) assert.ok(!ids.includes(id), `${id} must be dropped`);
  assert.ok(ids.includes("persona"), "a kept row survives");
  assert.ok(ids.includes("command-compact"), "a kept group child survives");
});

test("the compaction backend row is repointed with the token budget", () => {
  const built = buildTinytitanPlugins({ plugins: standardList(), maxTokens: 1234 });
  const compaction = built.find((row) => row.id === "compaction");
  const basic = compaction.config.find((row) => row.id === "compaction-basic");
  assert.equal(basic.name, COMPACTION_BACKEND);
  assert.equal(basic.config.maxTokens, 1234);
  assert.equal(
    "headroomTokens" in basic.config,
    false,
    "the harness's own headroom policy stands unless one is named",
  );
});

test("a named compaction headroom reaches the row, 0 included", () => {
  const built = buildTinytitanPlugins({ plugins: standardList(), headroomTokens: 0 });
  const compaction = built.find((row) => row.id === "compaction");
  const basic = compaction.config.find((row) => row.id === "compaction-basic");
  assert.equal(basic.config.headroomTokens, 0, "0 must be written, not treated as unset");
  assert.equal(
    basic.config.maxTokens,
    32768,
    "maxTokens stays explicit: the engine derives it from headroom otherwise",
  );
});

test("unrelated rows and their expression objects pass through unchanged", () => {
  const source = standardList();
  const built = buildTinytitanPlugins({ plugins: source });
  const bash = built.find((row) => row.id === "tool-bash");
  assert.deepEqual(bash.disabled, { __jsExpr: "x" });
  assert.equal(built.find((row) => row.id === "persona").name, "@deepseek-ai/dsh-persona");
});

test("the input list is not mutated", () => {
  const source = standardList();
  const before = JSON.stringify(source);
  buildTinytitanPlugins({ plugins: source });
  assert.equal(JSON.stringify(source), before);
});

test("a non-array plugins argument is rejected, not silently ignored", () => {
  assert.throws(() => buildTinytitanPlugins({}), /plugins array/);
});

/** Depth-first lookup by row id, the way the transform walks the list. */
function findRow(rows, id) {
  for (const row of rows) {
    if (row === null || typeof row !== "object") continue;
    if (row.id === id) return row;
    if (Array.isArray(row.config)) {
      const nested = findRow(row.config, id);
      if (nested) return nested;
    }
  }
  return null;
}

test("autonomy writes the policy and enables the fresh-agent loop", () => {
  const built = buildTinytitanPlugins({
    plugins: standardList(),
    autonomy: true,
    autonomyRounds: 25,
  });
  const persona = findRow(built, "persona");
  assert.match(persona.config.suffix, /\{\{cwd\}\}/, "the shipped template survives");
  assert.ok(
    persona.config.suffix.includes(AUTONOMY_INSTRUCTIONS),
    "the policy is appended to the persona",
  );
  const ralph = findRow(built, "tool-ralph");
  assert.equal(ralph.disabled, false, "the loop ships disabled; autonomy is the opt-in");
  assert.equal(ralph.config.maxRounds, 25, "the configured budget reaches the row");
  assert.equal(ralph.config.subagentProvider, "spawn", "the provider selection is preserved");
  assert.ok(findRow(built, "tool-ask-user"), "the question tool stays unless removal is asked for");
});

test("questions are removed only when that is asked for, and only with autonomy", () => {
  const kept = buildTinytitanPlugins({ plugins: standardList(), autonomy: true });
  assert.ok(findRow(kept, "tool-ask-user"));

  const removed = buildTinytitanPlugins({
    plugins: standardList(),
    autonomy: true,
    autonomySuppressQuestions: true,
  });
  assert.equal(findRow(removed, "tool-ask-user"), null);

  const flaggedButOff = buildTinytitanPlugins({
    plugins: standardList(),
    autonomy: false,
    autonomySuppressQuestions: true,
  });
  assert.ok(findRow(flaggedButOff, "tool-ask-user"), "the flag is scoped to autonomy");
});

test("autonomy off leaves the policy rows exactly as shipped", () => {
  const built = buildTinytitanPlugins({ plugins: standardList() });
  assert.equal(
    findRow(built, "persona").config.suffix,
    "Your working directory is {{cwd}}.",
    "no policy text without the switch",
  );
  const ralph = findRow(built, "tool-ralph");
  assert.equal(ralph.disabled, true);
  assert.equal(ralph.config.maxRounds, 64, "the shipped budget is untouched");
});

test("standardPlugins reads the preset-standard declaration", async (t) => {
  let plugins;
  try {
    plugins = await standardPlugins({ path: STANDARD_PATCH });
  } catch (error) {
    if (error?.code === "ERR_MODULE_NOT_FOUND") {
      t.skip("js-yaml / @deepseek-ai/cordis-plugin-include are not installed here");
      return;
    }
    throw error;
  }
  assert.ok(Array.isArray(plugins) && plugins.length > 0);
  assert.ok(plugins.some((row) => row.id === "compaction"));
  assert.match(readFileSync(STANDARD_PATCH, "utf8"), /preset-standard/);
});

/** A settings service stub that records what it was asked to write. */
function settingsStub({ forms = [], failUpdate = null } = {}) {
  const updates = [];
  return {
    updates,
    describe: () => forms,
    update: async (ns, patch) => {
      if (failUpdate !== null) throw failUpdate;
      updates.push([ns, patch]);
    },
  };
}

/** The `agent-preset-registry` descriptor with (or without) a user selection. */
function registryForm(selectedDefault) {
  return {
    ns: PRESET_SETTINGS_NS,
    user: selectedDefault === undefined ? {} : { selectedDefault },
    value: { default: "standard", selectedDefault },
  };
}

test("the preset becomes the default only while nothing is selected", async () => {
  const settings = settingsStub({ forms: [registryForm(undefined)] });
  const result = await ensureDefaultPreset({ settings, presetId: "tinytitan", log: () => {} });
  assert.equal(result.status, "set");
  assert.deepEqual(settings.updates, [[PRESET_SETTINGS_NS, { selectedDefault: "tinytitan" }]]);
});

test("an explicit selection is left exactly as it is", async () => {
  const settings = settingsStub({ forms: [registryForm("ptc")] });
  const result = await ensureDefaultPreset({ settings, presetId: "tinytitan", log: () => {} });
  assert.equal(result.status, "kept");
  assert.equal(result.selectedDefault, "ptc");
  assert.equal(settings.updates.length, 0);
});

test("a form the service does not report is not treated as a choice", async () => {
  const settings = settingsStub({ forms: [{ ns: "some-other-entry" }] });
  const result = await ensureDefaultPreset({ settings, presetId: "tinytitan", log: () => {} });
  assert.equal(result.status, "set");
});

test("a failing write is reported, never thrown", async () => {
  const lines = [];
  const settings = settingsStub({
    forms: [registryForm(undefined)],
    failUpdate: new Error("read-only profile"),
  });
  const result = await ensureDefaultPreset({
    settings,
    presetId: "tinytitan",
    log: (message) => lines.push(message),
  });
  assert.equal(result.status, "failed");
  assert.ok(lines.some((line) => line.includes("read-only profile")));
});

test("registerTinytitanPreset hands the registry the built roster", async () => {
  const registered = [];
  const ctx = {
    agentPresets: {
      register: async (definition) => {
        registered.push(definition);
        return () => {};
      },
    },
  };
  const disposer = await registerTinytitanPreset(ctx, {
    presetId: "tinytitan",
    plugins: standardList(),
    headroomTokens: 0,
    log: () => {},
  });
  assert.equal(typeof disposer, "function");
  assert.equal(registered.length, 1);
  assert.equal(registered[0].id, "tinytitan");
  assert.equal(registered[0].name, "TinyTitan");
  const basic = registered[0].plugins
    .find((row) => row.id === "compaction")
    .config.find((row) => row.id === "compaction-basic");
  assert.equal(basic.name, COMPACTION_BACKEND);
  assert.equal(basic.config.maxTokens, 32768);
  assert.equal(basic.config.headroomTokens, 0);
});

test("registerTinytitanPreset reports a missing registry and a throwing one", async () => {
  const lines = [];
  const missing = await registerTinytitanPreset(
    {},
    { plugins: standardList(), log: (message) => lines.push(message) },
  );
  assert.equal(missing, null);
  assert.ok(lines.some((line) => line.includes("agentPresets service is unavailable")));

  const failing = await registerTinytitanPreset(
    {
      agentPresets: {
        register: async () => {
          throw new Error("duplicate agent preset: tinytitan");
        },
      },
    },
    { plugins: standardList(), log: (message) => lines.push(message) },
  );
  assert.equal(failing, null, "a bad preset must not take the boot down");
  assert.ok(lines.some((line) => line.includes("duplicate agent preset")));
});

test("a read that throws leaves the selection alone instead of guessing", async () => {
  const settings = {
    describe: () => {
      throw new Error("no document");
    },
    update: async () => {
      throw new Error("must not be called");
    },
  };
  const result = await ensureDefaultPreset({ settings, presetId: "tinytitan", log: () => {} });
  assert.equal(result.status, "skipped");
  assert.match(result.reason, /no document/);
});

test("no settings service means no write and no throw", async () => {
  for (const settings of [undefined, null, {}]) {
    const result = await ensureDefaultPreset({ settings, presetId: "tinytitan", log: () => {} });
    assert.equal(result.status, "skipped");
  }
});
