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
  buildTinytitanPlugins,
  CHAT_NOISE_ROWS,
  COMPACTION_BACKEND,
  standardPlugins,
} from "../src/preset.js";

const STANDARD_PATCH = fileURLToPath(new URL("./fixtures/standard-patch.yml", import.meta.url));

/** A minimal standard list, mirroring the shipped shape. */
function standardList() {
  return [
    { id: "persona", name: "@deepseek-ai/dsh-persona" },
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
