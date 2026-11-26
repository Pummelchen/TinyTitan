import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import fs, {
  chmodSync,
  existsSync,
  lstatSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  statSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import { registerRoute } from "../src/route.js";
import {
  applyDefaultModelToSettings,
  applyRouteThroughSettings,
  applyRouteToSettings,
  collectRoute,
  DEFAULT_MODEL_SETTINGS_NS,
  ensureDefaultModel,
  findModelsDir,
  findServerBinary,
  generateBlock,
  generateRoute,
  repairDefaultModelSettings,
  writeRouteSettings,
} from "../src/generate.js";

const REPO_ROOT = fileURLToPath(new URL("../../../", import.meta.url));
const SHELL = join(REPO_ROOT, "tools", "dsh_route.sh");
const MODELS_DIR = join(REPO_ROOT, "models");
// The release build the checkout comparison needs, named the way the shell tool
// names it: `tools/dsh_route.sh` resolves its server through `$TINYTITAN_BIN_DIR`
// and then the checkout's `.build/release` (SwiftPM keeps that as a link to the
// triple directory), so both sides of the comparison read the same binary and
// the assertion is about the two writers, not about discovery.
const SHELL_BINARY = join(REPO_ROOT, ".build", "release", "TinyTitanServer");

/** A two-model catalog with one binary and one four-level family. */
const FAKE = [
  {
    id: "alpha_4-Bit",
    name: "Alpha 4B",
    family: "f_dense",
    quant: 4,
    backend: "gpu",
    engines: "gpu,cpu",
    path: "/models/alpha_4Bit",
    thinking: ["off", "on"],
  },
  {
    id: "beta_4-Bit",
    name: "Beta 35B-A3B",
    family: "qwen38flash",
    quant: 4,
    backend: "gpu",
    engines: "gpu",
    path: "/models/beta_4Bit",
    thinking: ["off", "low", "medium", "xhigh"],
  },
];

function count(haystack, needle) {
  return haystack.split(needle).length - 1;
}

/** The catalog exactly as the shell tool would read it, with no ambient env. */
function readCatalog() {
  const stdout = execFileSync(SHELL_BINARY, ["--catalog", "--models-dir", MODELS_DIR], {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
  return JSON.parse(String(stdout)).models;
}

/** `tools/dsh_route.sh --print`, with the ambient TinyTitan env removed. */
function shellBlock(extra = []) {
  const env = { ...process.env };
  for (const name of [
    "TINYTITAN_CATALOG_JSON",
    "TINYTITAN_MODELS_DIR",
    "TINYTITAN_PORT",
    "TINYTITAN_SERVER",
    // The tool checks this before the checkout's own `.build/release`, so an
    // ambient one would have the two sides read different binaries and the
    // comparison would be about build versions rather than about the writers.
    "TINYTITAN_BIN_DIR",
  ]) {
    delete env[name];
  }
  return execFileSync("bash", [SHELL, "--print", ...extra], {
    cwd: REPO_ROOT,
    encoding: "utf8",
    env,
    stdio: ["ignore", "pipe", "pipe"],
  });
}

/**
 * `js-yaml`, or a skipped test.
 *
 * The writer parses the generated block with the harness's own copy, and CI
 * runs this suite before `npm ci` installs one here, so a missing module is a
 * skip rather than a product failure — the same rule the existing
 * settings-service tests follow.
 */
async function yamlModule(t) {
  try {
    return (await import("js-yaml")).default;
  } catch (error) {
    if (error?.code === "ERR_MODULE_NOT_FOUND") {
      t.skip("js-yaml is not installed here; the harness's own copy resolves it");
      return null;
    }
    throw error;
  }
}

/**
 * `tools/dsh_route.sh --print` over a scratch catalog file.
 *
 * A catalog with no `models/` install and no built server is still exercised by
 * the shell tool: `TINYTITAN_CATALOG_JSON` is the path its own tests use, and
 * it means the comparison is about the two writers rather than about a build.
 */
function shellBlockFor(catalog) {
  const catalogPath = join(mkdtempSync(join(tmpdir(), "dsh-tinytitan-catalog-")), "catalog.json");
  writeFileSync(catalogPath, JSON.stringify({ models: catalog }));
  const env = { ...process.env, TINYTITAN_CATALOG_JSON: catalogPath };
  delete env.TINYTITAN_MODELS_DIR;
  delete env.TINYTITAN_PORT;
  delete env.TINYTITAN_SERVER;
  return execFileSync("bash", [SHELL, "--print"], {
    cwd: REPO_ROOT,
    encoding: "utf8",
    env,
    stdio: ["ignore", "pipe", "pipe"],
  });
}

/** Whether this machine can run the checkout comparison at all. */
function shellComparable(t) {
  if (!existsSync(SHELL) || !existsSync(SHELL_BINARY) || !existsSync(MODELS_DIR)) {
    t.skip(
      `no ${SHELL_BINARY} or tools/dsh_route.sh or models/ here; skipping shell byte-equality`,
    );
    return false;
  }
  // A build with an empty models/ is not comparable: the shell tool exits 2
  // ("no catalog") while the pure builder happily produces an empty block. The
  // comparison needs at least one install to have something to compare.
  try {
    if (readCatalog().length === 0) {
      t.skip("models/ holds no install; skipping shell byte-equality");
      return false;
    }
  } catch (error) {
    t.skip(`the catalog could not be read (${error.message}); skipping shell byte-equality`);
    return false;
  }
  return true;
}

test("the generated block is byte-identical to tools/dsh_route.sh --print", (t) => {
  if (!shellComparable(t)) return;
  const models = readCatalog();
  assert.equal(generateBlock(models), shellBlock());
  assert.equal(
    generateBlock(models, {
      port: 8123,
      provider: "local",
      reasoning: "off",
      context: 131072,
      maxTokens: 8192,
    }),
    shellBlock([
      "--port",
      "8123",
      "--provider",
      "local",
      "--reasoning",
      "off",
      "--context",
      "131072",
      "--max-tokens",
      "8192",
    ]),
  );
});

/** A catalog that exercises GPU-first order, level re-sorting and skipped rows. */
const SYNTHETIC = [
  {
    id: "cpu_4-Bit",
    name: "CPU Model",
    family: "f_dense",
    quant: 4,
    backend: "cpu",
    engines: "gpu,cpu",
    path: "/models/cpu_4Bit",
    thinking: ["xhigh", "off", "medium"],
  },
  {
    id: "gpu-on_8-Bit",
    name: "GPU Binary",
    family: "qwen36",
    quant: 8,
    backend: "gpu",
    engines: "gpu",
    path: "/models/gpu_8Bit",
    thinking: ["on", "off"],
  },
  // Two widths of one model, named identically by the catalog. The picker
  // renders `name` and nothing else, so this pair is what proves the label
  // carries the width.
  {
    id: "twin_4-Bit",
    name: "Twin 2B",
    family: "f_dense",
    quant: 4,
    backend: "gpu",
    engines: "gpu",
    path: "/models/twin_4Bit",
    thinking: ["off", "on"],
  },
  {
    id: "twin_8-Bit",
    name: "Twin 2B",
    family: "f_dense",
    quant: 8,
    backend: "gpu",
    engines: "gpu",
    path: "/models/twin_8Bit",
    thinking: ["off", "on"],
  },
  // A string where a list belongs: the shell parser iterates it per character.
  {
    id: "string_4-Bit",
    name: "String Levels",
    family: "f_dense",
    quant: 4,
    backend: "cpu",
    engines: "cpu",
    path: "/models/string_4Bit",
    thinking: "off",
  },
  // Neither a bad backend nor a missing quant may reach the block.
  { id: "bad-backend", name: "Bad", family: "f", quant: 4, backend: "tpu", path: "/models/bad" },
  { id: "bad-quant", name: "Bad", family: "f", backend: "gpu", path: "/models/bad" },
];

test("the shell tool and the generator agree on a synthetic catalog", (t) => {
  if (!existsSync(SHELL)) {
    t.skip("no tools/dsh_route.sh here; skipping the synthetic-catalog comparison");
    return;
  }
  try {
    execFileSync("python3", ["--version"], { stdio: ["ignore", "ignore", "ignore"] });
  } catch {
    t.skip("python3 is not available for the shell tool's catalog parser");
    return;
  }
  const printed = shellBlockFor(SYNTHETIC);
  const block = generateBlock(SYNTHETIC);
  assert.equal(block, printed);
  assert.deepEqual(
    [...block.matchAll(/^ {8}- id: (.+)$/gm)].map((match) => match[1]),
    ["gpu-on_8-Bit", "twin_4-Bit", "twin_8-Bit", "cpu_4-Bit", "string_4-Bit"],
  );
  // One label per row, and the width in it: the picker shows `name` alone, so
  // two widths of one model would otherwise arrive as the same row twice.
  const labels = [...block.matchAll(/^ {10}name: (.+)$/gm)].map((match) => match[1]);
  assert.equal(new Set(labels).size, labels.length, "every picker label must be unique");
  assert.ok(
    labels.includes("Twin 2B (4-bit)") && labels.includes("Twin 2B (8-bit)"),
    `the width must be on the label: ${JSON.stringify(labels)}`,
  );
  // Levels follow pi-ai's order, not the catalog's.
  assert.ok(
    block.includes(
      "          reasoningEfforts:\n            off:\n            medium: medium\n            xhigh: xhigh\n",
    ),
  );
});

/**
 * A catalog whose every free-form field is YAML syntax.
 *
 * A catalog is JSON, so a name is any string a hand edit or a future manifest
 * put there: `Evil: name # x` changes the block's structure unquoted, `true`
 * parses as a boolean, `[x]` as a flow sequence, and a leading `-` starts a
 * sequence entry. The writer must keep them literals. The row with a space in
 * its id is refused (a served id is one `\S+` token in the block the picker and
 * the default-model repair both read) without taking the valid rows with it.
 */
const ADVERSARIAL = [
  {
    id: "colon_4-Bit",
    name: "Evil: name # not a comment",
    family: "f_dense",
    quant: 4,
    backend: "gpu",
    engines: "gpu",
    path: "/models/colon_4Bit",
    thinking: ["off", "on"],
  },
  {
    id: "nullish_4-Bit",
    name: "null",
    family: "f_dense",
    quant: 4,
    backend: "gpu",
    engines: "gpu",
    path: "/models/nullish_4Bit",
    thinking: ["off"],
  },
  {
    id: "bools_4-Bit",
    name: "true",
    family: "f_dense",
    quant: 4,
    backend: "gpu",
    engines: "gpu",
    path: "/models/bools_4Bit",
    thinking: ["off"],
  },
  {
    id: "flow_4-Bit",
    name: "[brackets] {braces} &anchor *star !tag |pipe >fold %pct @at",
    family: "f_dense",
    quant: 4,
    backend: "gpu",
    engines: "gpu",
    path: "/models/flow_4Bit",
    thinking: ["off"],
  },
  {
    id: "quotes_4-Bit",
    name: `he said "hi", then 'bye' \\ back`,
    family: "f_dense",
    quant: 4,
    backend: "gpu",
    engines: "gpu",
    path: "/models/quotes_4Bit",
    thinking: ["off"],
  },
  {
    id: "unicode_4-Bit",
    name: "ünïcode 日本語 — em dash",
    family: "f_dense",
    quant: 4,
    backend: "gpu",
    engines: "gpu",
    path: "/models/unicode_4Bit",
    thinking: ["off"],
  },
  {
    id: "leading_4-Bit",
    name: "- leading dash",
    family: "f_dense",
    quant: 4,
    backend: "gpu",
    engines: "gpu",
    path: "/models/leading_4Bit",
    thinking: ["off"],
  },
  {
    id: "levels_4-Bit",
    name: "Levels",
    family: "f_dense",
    quant: 4,
    backend: "gpu",
    engines: "gpu",
    path: "/models/levels_4Bit",
    thinking: ["off", "xhigh:custom", "hash#tag", "true", '"q"', "split,level"],
  },
  {
    id: "control_4-Bit",
    name: "bell \u0007 and del \u007f",
    family: "f_dense",
    quant: 4,
    backend: "gpu",
    engines: "gpu",
    path: "/models/control_4Bit",
    thinking: ["off"],
  },
  {
    id: "bad id",
    name: "Bad Id",
    family: "f_dense",
    quant: 4,
    backend: "gpu",
    engines: "gpu",
    path: "/models/bad_4Bit",
    thinking: ["off"],
  },
];

/** The literal YAML values the adversarial catalog must produce. */
const ADVERSARIAL_NAMES = [
  "Evil: name # not a comment (4-bit)",
  "null (4-bit)",
  "true (4-bit)",
  "[brackets] {braces} &anchor *star !tag |pipe >fold %pct @at (4-bit)",
  `he said "hi", then 'bye' \\ back (4-bit)`,
  "ünïcode 日本語 — em dash (4-bit)",
  "- leading dash (4-bit)",
  "Levels (4-bit)",
  "bell \u0007 and del \u007f (4-bit)",
];

test("adversarial catalog values round-trip as literals through a YAML parser", async (t) => {
  const yaml = await yamlModule(t);
  if (!yaml) return;
  const provider = "evil: provider # x";
  const block = generateBlock(ADVERSARIAL, { provider, reasoning: "off" });
  const parsed = yaml.load(block);
  // No row may add a key of its own at any level: the block is exactly the
  // route the writer meant to write.
  assert.deepEqual(Object.keys(parsed), ["llm-pi-ai"]);
  const providers = parsed["llm-pi-ai"].providers;
  assert.deepEqual(Object.keys(providers), [provider], "the provider name is one literal key");
  const route = providers[provider];
  assert.equal(route.reasoning, "off");
  assert.equal(route.defaultContextWindow, 262144);
  assert.equal(route.defaultMaxTokens, 32768);
  // The refused row is the only one missing, and its id is not in the block.
  assert.deepEqual(
    route.models.map((model) => model.name),
    ADVERSARIAL_NAMES,
  );
  assert.deepEqual(
    route.models.map((model) => model.id),
    ADVERSARIAL.filter((model) => model.id !== "bad id").map((model) => model.id),
  );
  assert.ok(!block.includes("bad id"));
  assert.equal(typeof route.models[0].maxTokens, "number");
  // `off` is a bare key with no value; `on` is the wire value of `medium`.
  assert.deepEqual(route.models[0].reasoningEfforts, { off: null, medium: "on" });
  // A level the level vocabulary does not name is still offered, as the literal
  // string it is, and `true` does not become the boolean or the other way round.
  const levels = route.models[7].reasoningEfforts;
  assert.deepEqual(Object.keys(levels), [
    "off",
    "xhigh:custom",
    "hash#tag",
    "true",
    '"q"',
    "split",
    "level",
  ]);
  assert.equal(levels["true"], "true");
  assert.equal(levels['"q"'], '"q"');
  assert.equal(levels["hash#tag"], "hash#tag");
});

test("route numbers are written as safe whole numbers, never as text", () => {
  // The three numeric lines come from resolved config, but `generateBlock` is
  // exported and a caller that hands it a string is one `#` away from
  // commenting out the rest of the block.
  assert.throws(() => generateBlock(FAKE, { port: "8080 # x" }), /port must be a whole number/);
  assert.throws(() => generateBlock(FAKE, { port: 70000 }), /port must be a whole number/);
  assert.throws(() => generateBlock(FAKE, { context: 1.5 }), /context must be a whole number/);
  assert.throws(() => generateBlock(FAKE, { context: 0 }), /context must be a whole number/);
  assert.throws(() => generateBlock(FAKE, { maxTokens: -1 }), /maxTokens must be a whole number/);
  assert.throws(
    () => generateBlock(FAKE, { maxTokens: Number.MAX_SAFE_INTEGER + 1 }),
    /maxTokens must be a whole number/,
  );
  // A canonical decimal string is the shape the shell tool's own arguments
  // take, and it is written as the number it is.
  assert.ok(
    generateBlock(FAKE, { port: "8123" }).startsWith(
      "# DeepSeek Harness route to the TinyTitan server on port 8123.\n",
    ),
  );
  assert.ok(
    generateBlock(FAKE, { context: "131072" }).includes("          contextWindow: 131072\n"),
  );
});

test("a written settings file parses to the catalog's literal values", async (t) => {
  const yaml = await yamlModule(t);
  if (!yaml) return;
  const directory = mkdtempSync(join(tmpdir(), "dsh-tinytitan-generate-"));
  const settingsPath = join(directory, "settings.yaml");
  // The person's own file, with a comment and their own keys: the replacement
  // is line-based and must keep both.
  writeFileSync(
    settingsPath,
    "# my profile\nui-theme:\n  preference: dark\nagent-presets:\n  default: qwen38\n",
  );
  const written = writeRouteSettings({
    settingsPath,
    block: generateBlock(ADVERSARIAL),
    stamp: "adversarial",
  });
  assert.equal(written.changed, true);
  const parsed = yaml.load(readFileSync(settingsPath, "utf8"));
  assert.deepEqual(parsed["ui-theme"], { preference: "dark" });
  assert.deepEqual(parsed["agent-presets"], { default: "qwen38" });
  const route = parsed["llm-pi-ai"].providers.tinytitan;
  assert.deepEqual(
    route.models.map((model) => model.name),
    ADVERSARIAL_NAMES,
  );
});

test("the shell tool and the generator agree on adversarial scalars", async (t) => {
  if (!existsSync(SHELL)) {
    t.skip("no tools/dsh_route.sh here; skipping the adversarial comparison");
    return;
  }
  try {
    execFileSync("python3", ["--version"], { stdio: ["ignore", "ignore", "ignore"] });
  } catch {
    t.skip("python3 is not available for the shell tool's catalog parser");
    return;
  }
  const yaml = await yamlModule(t);
  if (!yaml) return;
  const printed = shellBlockFor(ADVERSARIAL);
  const block = generateBlock(ADVERSARIAL);
  // Byte parity is the contract between the two writers, and it is what makes
  // the shell tool authoritative wherever a checkout exists.
  assert.equal(block, printed);
  // Both writers quote, and the escaped forms parse back to the literals.
  const route = yaml.load(printed)["llm-pi-ai"].providers.tinytitan;
  assert.deepEqual(
    route.models.map((model) => model.name),
    ADVERSARIAL_NAMES,
  );
  assert.ok(!printed.includes("bad id"), "an id that is not a token is refused by both");
});

/** A row the writer must refuse, with the diagnostic it must print for it. */
const MALFORMED = [
  [
    { id: "bad-backend_4-Bit", name: "B", backend: "tpu", quant: 4, path: "/m" },
    "the backend is not gpu or cpu",
  ],
  [{ id: "no-quant_4-Bit", name: "B", backend: "gpu", path: "/m" }, "quant is not a number"],
  [
    { id: "no-size_4-Bit", name: "B", backend: "gpu", quant: 4, size_gb: "many", path: "/m" },
    "size_gb is not a number",
  ],
  [
    { id: "bad id", name: "B", backend: "gpu", quant: 4, path: "/m" },
    "the id is not a plain model id",
  ],
  [
    { id: "no-name_4-Bit", backend: "gpu", quant: 4, path: "/m" },
    "the name is empty or multi-line",
  ],
  [{ id: "no-path_4-Bit", name: "B", backend: "gpu", quant: 4 }, "the path is empty or multi-line"],
  [
    { id: "bad-thinking_4-Bit", name: "B", backend: "gpu", quant: 4, path: "/m", thinking: 7 },
    "thinking is not iterable",
  ],
  [null, "the entry is not an object"],
];

test("a malformed catalog row is reported and the valid rows still serve", () => {
  // A broken install used to vanish from the picker with nothing said anywhere,
  // while the shell parser it mirrors prints `catalog: skipping …` on stderr. The
  // operator needs the same evidence from the built-in writer: which row, which
  // install when it has a usable id, and what was wrong.
  const messages = [];
  const models = [FAKE[0], ...MALFORMED.map(([model]) => model), FAKE[1]];
  const block = generateBlock(models, { log: (message) => messages.push(message) });
  assert.deepEqual(
    messages,
    MALFORMED.map(
      ([model, reason], index) =>
        `dsh-tinytitan: skipping catalog row ${index + 2}` +
        `${typeof model?.id === "string" && /^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(model.id) ? ` (id ${model.id})` : ""}` +
        `: ${reason}`,
    ),
  );
  // The two good rows are still there, in their own order, and nothing else is.
  assert.deepEqual(
    [...block.matchAll(/^ {8}- id: (.+)$/gm)].map((match) => match[1]),
    ["alpha_4-Bit", "beta_4-Bit"],
  );
});

test("a row diagnostic never quotes the catalog text it refused", () => {
  // The catalog is data, not a trusted message: an id or name can carry a
  // credential, a newline or the rest of the file. Only the reason vocabulary
  // this module owns, the row number and a *validated* id are printed.
  const messages = [];
  generateBlock(
    [
      {
        id: "leaky_4-Bit",
        name: "Evil: name # comment\napiKey: sk-live-SECRET",
        backend: "tpu",
        quant: 4,
        path: "/m",
      },
      { id: "not a token: sk-live-SECRET", name: "x", backend: "tpu", quant: 4, path: "/m" },
    ],
    { log: (message) => messages.push(message) },
  );
  assert.equal(messages.length, 2);
  for (const message of messages) {
    assert.ok(!message.includes("SECRET"), message);
    assert.ok(!message.includes("Evil"), message);
    assert.ok(!message.includes("\n"), message);
  }
  assert.match(
    messages[0],
    /^dsh-tinytitan: skipping catalog row 1 \(id leaky_4-Bit\): the backend is not gpu or cpu$/,
  );
  assert.match(
    messages[1],
    /^dsh-tinytitan: skipping catalog row 2: the backend is not gpu or cpu$/,
  );
});

test("an unexpected failure while reading a row is not swallowed", () => {
  // The old catch was bare: a programming error inside validation read exactly
  // like a malformed install and disappeared. Only this module's own typed
  // refusal is a skip; anything else is a defect and has to be visible.
  const exploding = {
    get backend() {
      throw new Error("catalog getter exploded");
    },
  };
  assert.throws(
    () => generateBlock([FAKE[0], exploding], { log: () => {} }),
    /catalog getter exploded/,
  );
});

test("an all-invalid catalog says so row by row and as a whole", () => {
  const home = mkdtempSync(join(tmpdir(), "dsh-tinytitan-home-"));
  const settingsPath = join(home, "settings.yaml");
  writeFileSync(settingsPath, "ui-theme:\n  preference: dark\n");
  const messages = [];
  const result = generateRoute({
    serverBinary: "/x/TinyTitanServer",
    modelsDir: "/models",
    settingsPath,
    env: { PATH: "" },
    isExecutable: () => true,
    isDirectory: () => true,
    run: () => JSON.stringify({ models: MALFORMED.map(([model]) => model) }),
    log: (message) => messages.push(message),
  });
  assert.equal(result.status, "failed");
  assert.match(result.detail, /describes no servable install/);
  assert.equal(messages.filter((message) => /skipping catalog row/.test(message)).length, 8);
  assert.ok(
    messages.some((message) => /describes no servable install/.test(message)),
    messages,
  );
  // A route that describes nothing must not replace the person's file.
  assert.equal(readFileSync(settingsPath, "utf8"), "ui-theme:\n  preference: dark\n");
});

test("a repeated refresh does not repeat the same catalog diagnostic", () => {
  // The watcher refreshes once per quiet period, so a still-broken install would
  // otherwise print the same lines on every pass. One budget per refresher: the
  // first failure is always printed, a new one is printed, a repeat is not.
  const messages = [];
  const diagnostics = new Set();
  const directory = mkdtempSync(join(tmpdir(), "dsh-tinytitan-models-"));
  const options = {
    serverBinary: "/x/TinyTitanServer",
    modelsDir: directory,
    env: { PATH: "" },
    isExecutable: () => true,
    isDirectory: () => true,
    diagnostics,
    log: (message) => messages.push(message),
  };
  const run = () => JSON.stringify({ models: [MALFORMED[0][0]] });
  collectRoute({ ...options, run });
  collectRoute({ ...options, run });
  const skips = () => messages.filter((message) => /skipping catalog row/.test(message));
  assert.equal(skips().length, 1, JSON.stringify(messages));

  // Installing another broken model is new information, so it is printed.
  collectRoute({
    ...options,
    run: () => JSON.stringify({ models: [MALFORMED[0][0], MALFORMED[1][0]] }),
  });
  assert.equal(skips().length, 2);
});

test("the block carries the header, the three switches and the effort ladder", () => {
  const block = generateBlock(FAKE);
  assert.ok(
    block.startsWith(
      "# DeepSeek Harness route to the TinyTitan server on port 8080.\n" +
        "# Generated by tools/dsh_route.sh from the installs under models/.\n" +
        "# 2 served model(s); settings.yaml is hot-reloaded.\n",
    ),
  );
  assert.ok(block.includes("      displayName: TinyTitan\n"));
  assert.ok(block.includes("        authorization: Bearer tinytitan-local\n"));
  assert.ok(block.includes("      reasoning: medium\n"));
  assert.ok(block.includes("      streamIdleTimeoutMs: 3600000\n"));
  assert.ok(block.includes("      defaultContextWindow: 262144\n"));
  assert.ok(block.includes("      defaultMaxTokens: 32768\n"));
  assert.ok(block.includes("            thinkingFormat: chat-template\n"));
  assert.ok(block.includes("              enable_thinking: { $var: thinking.enabled }\n"));
  assert.ok(block.includes("              reasoning_effort: { $var: thinking.effort }\n"));
  assert.ok(block.includes("            maxTokensField: max_tokens\n"));
  assert.ok(block.includes("            supportsUsageInStreaming: true\n"));
  // An `on`-only template offers that mode as `medium` with the wire value `on`,
  // because pi-ai's level vocabulary has no `on`.
  assert.ok(
    block.includes("          reasoningEfforts:\n            off:\n            medium: on\n"),
  );
  assert.ok(block.includes("            xhigh: xhigh\n"));
  assert.ok(!block.includes("            on:"));
  assert.ok(block.endsWith("            supportsUsageInStreaming: true\n"));

  const options = generateBlock(FAKE, {
    port: 8123,
    provider: "local",
    reasoning: "off",
    context: 131072,
    maxTokens: 8192,
  });
  assert.ok(options.includes("      baseURL: http://127.0.0.1:8123/v1\n"));
  assert.ok(options.includes("    local:\n"));
  assert.ok(options.includes("      reasoning: off\n"));
  assert.ok(options.includes("          contextWindow: 131072\n"));
  assert.ok(options.includes("          maxTokens: 8192\n"));
});

test("a stale generated header and section are replaced by exactly one block", () => {
  const block = generateBlock(FAKE);
  const stale = [
    "ui-theme:",
    "  preference: dark",
    "# DeepSeek Harness route to the TinyTitan server on port 1.",
    "# Generated by tools/dsh_route.sh from the installs under models/.",
    "# 99 served model(s); settings.yaml is hot-reloaded.",
    "llm-pi-ai:",
    "  providers:",
    "    stale-route:",
    "      displayName: old",
    "",
    "agent-presets:",
    "  default: qwen38",
    "",
  ].join("\n");
  const next = applyRouteToSettings(stale, block);
  assert.equal(count(next, "# DeepSeek Harness route to the "), 1);
  assert.equal(count(next, "# Generated by tools/dsh_route.sh "), 1);
  assert.equal(count(next, "served model(s); settings.yaml is hot-reloaded."), 1);
  assert.ok(next.includes("llm-pi-ai:"));
  assert.ok(!next.includes("stale-route"));
  assert.ok(!next.includes("port 1."));
  assert.ok(next.includes("ui-theme:\n  preference: dark\n"));
  assert.ok(next.includes("agent-presets:\n  default: qwen38\n"));
  // Running it on its own output must not move a byte.
  assert.equal(applyRouteToSettings(next, block), next);
});

test("a file with no route gets the block appended, and an empty one gets only it", () => {
  const block = generateBlock(FAKE);
  const plain = "ui-theme:\n  preference: dark\n";
  assert.equal(applyRouteToSettings(plain, block), `${plain}\n${block}`);
  assert.equal(applyRouteToSettings("", block), block);
});

test("CRLF settings are normalized to LF, as the shell tool's own read does", () => {
  const block = generateBlock(FAKE);
  const next = applyRouteToSettings(
    "ui-theme:\r\n  preference: dark\r\nllm-pi-ai:\r\n  stale: 1\r\n",
    block,
  );
  assert.ok(!next.includes("\r"));
  assert.ok(next.startsWith("ui-theme:\n  preference: dark\n\n"));
  assert.ok(!next.includes("stale: 1"));
  assert.equal(applyRouteToSettings(next, block), next);
});

test("a rewrite of a real file is byte-identical and makes one backup", () => {
  const directory = mkdtempSync(join(tmpdir(), "dsh-tinytitan-generate-"));
  const settingsPath = join(directory, "settings.yaml");
  writeFileSync(settingsPath, "ui-theme:\n  preference: dark\n");
  const block = generateBlock(FAKE);

  const first = writeRouteSettings({ settingsPath, block, stamp: "first" });
  assert.equal(first.changed, true);
  assert.equal(first.backup, `${settingsPath}.bak-first`);
  const once = readFileSync(settingsPath, "utf8");
  assert.ok(once.includes("llm-pi-ai:"));

  const second = writeRouteSettings({ settingsPath, block, stamp: "second" });
  assert.equal(second.changed, false);
  assert.equal(second.backup, null);
  assert.equal(readFileSync(settingsPath, "utf8"), once);
  assert.ok(!existsSync(`${settingsPath}.bak-second`));
});

test("a missing settings file is refused with a clear error", () => {
  const directory = mkdtempSync(join(tmpdir(), "dsh-tinytitan-generate-"));
  assert.throws(
    () => writeRouteSettings({ settingsPath: join(directory, "absent.yaml"), block: "x:\n" }),
    /no DSH settings file at .*absent\.yaml/,
  );
  assert.throws(() => writeRouteSettings({ block: "x:\n" }), /no DSH settings file/);
});

/** A `node:fs` seam that fails one call, for the failure-before-rename cases. */
function failingIo(name, error) {
  return {
    ...fs,
    [name]: () => {
      throw error;
    },
  };
}

/** Every entry in a directory, sorted, for "nothing was left behind" checks. */
function entries(directory) {
  return readdirSync(directory).sort();
}

test("a failure before the rename leaves the settings file byte-identical", () => {
  // The old writer called `writeFileSync` on the target: an ENOSPC, a crash or a
  // failing sync there truncated the person's profile, and a backup is a copy of
  // what was there, not a way back. The replacement is a temp file in the same
  // directory plus one rename, so every failure before that rename must leave
  // the original untouched — bytes and mode.
  const directory = mkdtempSync(join(tmpdir(), "dsh-tinytitan-atomic-"));
  const settingsPath = join(directory, "settings.yaml");
  const original = "ui-theme:\n  preference: dark\n";
  writeFileSync(settingsPath, original);
  chmodSync(settingsPath, 0o640);
  const block = generateBlock(FAKE);

  for (const [name, error] of [
    ["renameSync", Object.assign(new Error("EIO: rename failed"), { code: "EIO" })],
    ["fsyncSync", Object.assign(new Error("EIO: fsync failed"), { code: "EIO" })],
    ["openSync", Object.assign(new Error("EACCES: no temp"), { code: "EACCES" })],
  ]) {
    assert.throws(
      () =>
        writeRouteSettings({ settingsPath, block, stamp: "atomic", io: failingIo(name, error) }),
      new RegExp(error.message),
      name,
    );
    assert.equal(readFileSync(settingsPath, "utf8"), original, `${name}: the profile is intact`);
    assert.equal(statSync(settingsPath).mode & 0o777, 0o640, `${name}: the mode is intact`);
    assert.deepEqual(
      entries(directory).filter((entry) => entry.includes(".tmp-")),
      [],
      `${name}: no temporary file is left behind`,
    );
  }
});

test("a write that dies mid-way cannot truncate the profile", () => {
  // The reported shape: the target is truncated and half-written. Here the temp
  // write itself dies after a partial write, which is what a full disk looks
  // like; the target must not have been opened for writing at all.
  const directory = mkdtempSync(join(tmpdir(), "dsh-tinytitan-atomic-"));
  const settingsPath = join(directory, "settings.yaml");
  const original = "ui-theme:\n  preference: dark\n";
  writeFileSync(settingsPath, original);
  const block = generateBlock(FAKE);
  const io = {
    ...fs,
    writeFileSync: (descriptor, text) => {
      fs.writeFileSync(descriptor, String(text).slice(0, 16));
      throw Object.assign(new Error("ENOSPC: no space left on device"), { code: "ENOSPC" });
    },
  };
  assert.throws(() => writeRouteSettings({ settingsPath, block, stamp: "half", io }), /ENOSPC/);
  assert.equal(readFileSync(settingsPath, "utf8"), original);
  assert.deepEqual(
    entries(directory).filter((entry) => entry.includes(".tmp-")),
    [],
  );
});

test("an edit between the read and the replacement is refused, not overwritten", () => {
  // A separate editor can save the file after the writer read it. The old
  // writer then overwrote that edit with a block computed from the stale text.
  // The replacement re-reads the exact bytes it based its work on and refuses
  // when they moved.
  const directory = mkdtempSync(join(tmpdir(), "dsh-tinytitan-atomic-"));
  const settingsPath = join(directory, "settings.yaml");
  const original = "ui-theme:\n  preference: dark\n";
  writeFileSync(settingsPath, original);
  const block = generateBlock(FAKE);
  let reads = 0;
  const io = {
    ...fs,
    readFileSync: (...args) => {
      reads += 1;
      const text = fs.readFileSync(...args);
      // The first read is the writer's; the recheck sees the other editor's save.
      return reads === 1 ? text : `${text}ui-theme:\n  preference: light\n`;
    },
  };
  assert.throws(
    () => writeRouteSettings({ settingsPath, block, stamp: "race", io }),
    /changed while the route was being written; nothing was replaced/,
  );
  // The edit is still the file's content — the reader was faked, not the write.
  assert.equal(readFileSync(settingsPath, "utf8"), original);
  assert.ok(!existsSync(`${settingsPath}.bak-race`), "nothing is backed up for a refused write");
  assert.deepEqual(
    entries(directory).filter((entry) => entry.includes(".tmp-")),
    [],
  );
});

test("the backup holds the exact bytes replaced and never clobbers an older one", () => {
  const directory = mkdtempSync(join(tmpdir(), "dsh-tinytitan-atomic-"));
  const settingsPath = join(directory, "settings.yaml");
  const original = "ui-theme:\n  preference: dark\n";
  writeFileSync(settingsPath, original);
  // Two refreshes in the same second (the default stamp has second resolution in
  // the shell tool and millisecond in the plugin) must not overwrite each other.
  writeFileSync(`${settingsPath}.bak-same`, "an older backup\n");
  const written = writeRouteSettings({ settingsPath, block: generateBlock(FAKE), stamp: "same" });
  assert.equal(written.backup, `${settingsPath}.bak-same-1`);
  assert.equal(readFileSync(`${settingsPath}.bak-same`, "utf8"), "an older backup\n");
  assert.equal(readFileSync(written.backup, "utf8"), original);
});

test("a symlinked settings file stays a symlink", () => {
  // A profile that keeps settings.yaml in a dotfiles checkout is a symlink; the
  // in-place writer followed it, so the replacement has to as well, or the link
  // becomes a regular file and the person's checkout stops tracking it.
  const directory = mkdtempSync(join(tmpdir(), "dsh-tinytitan-atomic-"));
  const profile = join(directory, "profile");
  mkdirSync(profile);
  const target = join(profile, "settings.yaml");
  writeFileSync(target, "ui-theme:\n  preference: dark\n");
  const link = join(directory, "settings.yaml");
  symlinkSync(target, link);

  const written = writeRouteSettings({ settingsPath: link, block: generateBlock(FAKE) });
  assert.equal(written.changed, true);
  assert.equal(lstatSync(link).isSymbolicLink(), true);
  assert.ok(readFileSync(target, "utf8").includes("llm-pi-ai:"));
});

test("the profile's permission bits are preserved, never widened", () => {
  const directory = mkdtempSync(join(tmpdir(), "dsh-tinytitan-atomic-"));
  const settingsPath = join(directory, "settings.yaml");
  writeFileSync(settingsPath, "ui-theme:\n  preference: dark\n");
  chmodSync(settingsPath, 0o600);
  writeRouteSettings({ settingsPath, block: generateBlock(FAKE) });
  assert.equal(statSync(settingsPath).mode & 0o777, 0o600);
});

test("the default-model repair is replaced atomically too", () => {
  const directory = mkdtempSync(join(tmpdir(), "dsh-tinytitan-atomic-"));
  const settingsPath = join(directory, "settings.yaml");
  const before = profile(DEAD_DEFAULT);
  writeFileSync(settingsPath, before);
  const io = failingIo("renameSync", new Error("EIO: rename failed"));
  assert.throws(() => repairDefaultModelSettings({ settingsPath, stamp: "repair", io }), /EIO/);
  assert.equal(readFileSync(settingsPath, "utf8"), before);
  assert.deepEqual(
    entries(directory).filter((entry) => entry.includes(".tmp-")),
    [],
  );
});

test("findServerBinary follows explicit, env, PATH, then the checkout", () => {
  const onPath = join("/env", "bin", "TinyTitanServer");
  const explicit = "/explicit/TinyTitanServer";
  const repoArm = join("/repo", ".build", "arm64-apple-macosx", "release", "TinyTitanServer");
  const repoRelease = join("/repo", ".build", "release", "TinyTitanServer");
  const files = new Set([explicit, onPath, repoArm, repoRelease]);
  const isExecutable = (path) => files.has(path);
  const env = { TINYTITAN_SERVER: onPath, PATH: "/env/bin" };

  assert.equal(findServerBinary({ explicit, env, repoRoot: "/repo", isExecutable }), explicit);
  assert.equal(
    findServerBinary({ explicit: "/nope", env, repoRoot: "/repo", isExecutable }),
    onPath,
  );
  assert.equal(findServerBinary({ env, repoRoot: "/repo", isExecutable }), onPath);
  // Nothing on PATH: the checkout's own build is the fallback.
  assert.equal(findServerBinary({ env: { PATH: "" }, repoRoot: "/repo", isExecutable }), repoArm);
  assert.equal(
    findServerBinary({
      env: { PATH: "" },
      repoRoot: "/repo",
      isExecutable: (path) => path === repoRelease,
    }),
    repoRelease,
  );
  assert.equal(
    findServerBinary({ env: { PATH: "" }, repoRoot: "/repo", isExecutable: () => false }),
    null,
  );
  assert.equal(findServerBinary({ env: {}, isExecutable: () => false }), null);
});

test("findServerBinary finds a real executable on PATH with the default probe", () => {
  const directory = mkdtempSync(join(tmpdir(), "dsh-tinytitan-path-"));
  const binary = join(directory, "TinyTitanServer");
  writeFileSync(binary, "#!/bin/sh\nexit 0\n");
  chmodSync(binary, 0o755);
  assert.equal(findServerBinary({ env: { PATH: directory }, repoRoot: "/none" }), binary);
});

test("findModelsDir follows explicit, env, then the checkout", () => {
  const dirs = new Set(["/explicit/models", "/env/models", "/repo/models"]);
  const isDirectory = (path) => dirs.has(path);
  assert.equal(
    findModelsDir({
      explicit: "/explicit/models",
      env: { TINYTITAN_MODELS_DIR: "/env/models" },
      repoRoot: "/repo",
      isDirectory,
    }),
    "/explicit/models",
  );
  assert.equal(
    findModelsDir({
      explicit: "/nope",
      env: { TINYTITAN_MODELS_DIR: "/env/models" },
      repoRoot: "/repo",
      isDirectory,
    }),
    "/env/models",
  );
  assert.equal(findModelsDir({ env: {}, repoRoot: "/repo", isDirectory }), "/repo/models");
  assert.equal(findModelsDir({ env: {}, repoRoot: "/repo", isDirectory: () => false }), null);
  assert.equal(findModelsDir({ env: {}, isDirectory: () => false }), null);
});

test("generateRoute discovers the binary, parses stdout, and writes the block", () => {
  const home = mkdtempSync(join(tmpdir(), "dsh-tinytitan-generate-"));
  const modelsDir = mkdtempSync(join(tmpdir(), "dsh-tinytitan-models-"));
  const settingsPath = join(home, "settings.yaml");
  writeFileSync(settingsPath, "ui-theme:\n  preference: dark\n");
  const binary = join(home, "TinyTitanServer");
  writeFileSync(binary, "#!/bin/sh\nexit 0\n");
  chmodSync(binary, 0o755);

  const calls = [];
  const run = (command, args) => {
    calls.push({ command, args });
    return JSON.stringify({ models: FAKE });
  };
  const result = generateRoute({
    serverBinary: binary,
    modelsDir,
    settingsPath,
    port: 8080,
    provider: "tinytitan",
    run,
    env: { PATH: "" },
    log: () => {},
  });
  assert.equal(result.status, "written-self-contained");
  assert.equal(result.detail, "written");
  assert.equal(result.models, 2);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].command, binary);
  assert.deepEqual(calls[0].args, ["--catalog", "--models-dir", modelsDir]);
  assert.equal(
    readFileSync(settingsPath, "utf8"),
    applyRouteToSettings("ui-theme:\n  preference: dark\n", generateBlock(FAKE)),
  );

  // A second run over its own output is a no-op with no new backup.
  const again = generateRoute({
    serverBinary: binary,
    modelsDir,
    settingsPath,
    run,
    env: { PATH: "" },
    log: () => {},
  });
  assert.equal(again.status, "written-self-contained");
  assert.equal(again.detail, "already current");
  assert.equal(again.backup, null);
});

test("generateRoute reports absent prerequisites and a broken catalog", () => {
  // No models directory is the one prerequisite nothing can work around.
  const noModels = generateRoute({
    serverBinary: "/x/TinyTitanServer",
    env: { PATH: "" },
    isExecutable: () => true,
    isDirectory: () => false,
    log: () => {},
  });
  assert.equal(noModels.status, "missing");
  assert.match(noModels.detail, /no models directory/);

  // A binary that fails is a reason to read the folder, not to give up: an
  // empty folder says so instead, and the binary's own words are logged.
  const emptyDir = mkdtempSync(join(tmpdir(), "dsh-tinytitan-empty-models-"));
  const messages = [];
  const broken = generateRoute({
    serverBinary: "/x/TinyTitanServer",
    modelsDir: emptyDir,
    env: { PATH: "" },
    isExecutable: () => true,
    isDirectory: () => true,
    run: () => {
      throw new Error("exit 2");
    },
    log: (message) => messages.push(message),
  });
  assert.equal(broken.status, "failed");
  assert.match(broken.detail, /describes no servable install/);
  assert.ok(
    messages.some((message) => /server catalog failed \(exit 2\)/.test(message)),
    `the binary's own failure must be logged: ${JSON.stringify(messages)}`,
  );

  const empty = generateRoute({
    serverBinary: "/x/TinyTitanServer",
    modelsDir: emptyDir,
    env: { PATH: "" },
    isExecutable: () => true,
    isDirectory: () => true,
    run: () => JSON.stringify({ models: [] }),
    log: () => {},
  });
  assert.equal(empty.status, "failed");
  assert.match(empty.detail, /describes no servable install/);

  // A populated folder with no binary at all is the case the folder scan is
  // for: the route is written from what is on disk.
  const dshHome = mkdtempSync(join(tmpdir(), "dsh-tinytitan-home-"));
  writeFileSync(join(dshHome, "settings.yaml"), "ui-theme:\n  preference: dark\n");
  const directory = join(emptyDir, "qwen3.5_2B_4Bit");
  mkdirSync(join(directory, "tokenizer"), { recursive: true });
  writeFileSync(
    join(directory, "manifest.json"),
    JSON.stringify({
      magic: "SSDAI",
      modelID: "qwen3.5-2b",
      quant: { routedExpert: { weightBits: 4 } },
      arch: { family: "qwen3_5_dense", hiddenActivation: "silu" },
    }),
  );
  writeFileSync(join(directory, "tokenizer", "tokenizer.json"), "{}");
  const scanned = generateRoute({
    env: { PATH: "" },
    dshHome,
    modelsDir: emptyDir,
    serverBinary: null,
    isExecutable: () => false,
    backup: false,
    log: () => {},
  });
  assert.equal(scanned.status, "written-self-contained");
  assert.equal(scanned.source, "folder");
  assert.equal(scanned.models, 1);
  const written = readFileSync(join(dshHome, "settings.yaml"), "utf8");
  assert.ok(written.includes("- id: qwen3.5-2b_4-Bit"));
  assert.ok(written.includes("name: Qwen 3.5 2B (4-bit)"));
});

test("registerRoute falls back to the generator when the checkout tool is absent", () => {
  const repoRoot = mkdtempSync(join(tmpdir(), "dsh-tinytitan-repo-"));
  const dshHome = mkdtempSync(join(tmpdir(), "dsh-tinytitan-home-"));
  const modelsDir = mkdtempSync(join(tmpdir(), "dsh-tinytitan-models-"));
  writeFileSync(join(dshHome, "settings.yaml"), "ui-theme:\n  preference: dark\n");
  const binary = join(repoRoot, "TinyTitanServer");
  writeFileSync(binary, "#!/bin/sh\nexit 0\n");
  chmodSync(binary, 0o755);

  const messages = [];
  const result = registerRoute({
    repoRoot,
    dshHome,
    port: 8080,
    provider: "tinytitan",
    serverBinary: binary,
    modelsDir,
    env: { PATH: "" },
    run: () => JSON.stringify({ models: FAKE }),
    log: (message) => messages.push(message),
  });
  assert.equal(result.status, "written-self-contained");
  assert.ok(messages.some((message) => message.includes("using the built-in route generator")));
  assert.ok(readFileSync(join(dshHome, "settings.yaml"), "utf8").includes("llm-pi-ai:"));
});

/** `registerRoute` on a checkout with no `tools/dsh_route.sh`, i.e. the built-in writer. */
function throughGenerator({ settingsText = "ui-theme:\n  preference: dark\n", ...options } = {}) {
  const repoRoot = mkdtempSync(join(tmpdir(), "dsh-tinytitan-repo-"));
  const dshHome = mkdtempSync(join(tmpdir(), "dsh-tinytitan-home-"));
  const modelsDir = mkdtempSync(join(tmpdir(), "dsh-tinytitan-models-"));
  writeFileSync(join(dshHome, "settings.yaml"), settingsText);
  const binary = join(repoRoot, "TinyTitanServer");
  writeFileSync(binary, "#!/bin/sh\nexit 0\n");
  chmodSync(binary, 0o755);
  const result = registerRoute({
    repoRoot,
    dshHome,
    port: 8080,
    provider: "tinytitan",
    serverBinary: binary,
    modelsDir,
    env: { PATH: "" },
    run: () => JSON.stringify({ models: FAKE }),
    log: () => {},
    ...options,
  });
  assert.equal(result.status, "written-self-contained");
  return { settings: readFileSync(join(dshHome, "settings.yaml"), "utf8"), result };
}

test("the built-in writer declares the window and cap it was given", () => {
  // The generator is the writer DSH 0.2.0 machines reach when the checkout has
  // no tool, so it needs the same pin as the script branch: dropping the two
  // forwarded options here left every existing test green.
  const { settings } = throughGenerator({ context: 131072, maxTokens: 8192 });
  assert.ok(settings.includes("defaultContextWindow: 131072"), settings);
  assert.ok(settings.includes("defaultMaxTokens: 8192"), settings);
  // One row per served model; the lowercase spelling excludes the provider's
  // own `defaultContextWindow` / `defaultMaxTokens` lines.
  assert.equal(count(settings, "contextWindow: 131072"), 2, settings);
  assert.equal(count(settings, "maxTokens: 8192"), 2, settings);
});

test("the built-in writer keeps the launcher's pin when nothing is declared", () => {
  const { settings } = throughGenerator({});
  assert.ok(settings.includes("defaultContextWindow: 262144"), settings);
  assert.ok(settings.includes("defaultMaxTokens: 32768"), settings);
  assert.equal(count(settings, "contextWindow: 262144"), 2);
  assert.equal(count(settings, "maxTokens: 32768"), 2);
});

test("applyRouteThroughSettings merges the route into the llm-pi-ai entry", async (t) => {
  // The writer parses the generated block with the harness's own `js-yaml`,
  // which resolves inside a harness (and after `npm ci`) but not in the CI step
  // that runs these suites before the toolchain is installed. Skip there rather
  // than reporting a product failure for a missing test dependency.
  try {
    await import("js-yaml");
  } catch (error) {
    if (error?.code === "ERR_MODULE_NOT_FOUND") {
      t.skip("js-yaml is not installed here; the harness's own copy resolves it");
      return;
    }
    throw error;
  }
  const updates = [];
  const settings = {
    update: async (ns, patch) => {
      updates.push({ ns, patch });
    },
  };
  const result = await applyRouteThroughSettings({
    serverBinary: "/x/TinyTitanServer",
    modelsDir: "/models",
    env: { PATH: "" },
    isExecutable: () => true,
    isDirectory: () => true,
    run: () => JSON.stringify({ models: FAKE }),
    settings,
    log: () => {},
  });
  assert.equal(result.status, "applied");
  assert.equal(result.models, 2);
  assert.equal(updates.length, 1);
  assert.equal(updates[0].ns, "llm-pi-ai");
  assert.equal(updates[0].patch.providers.tinytitan.models.length, 2);
  // Nothing declared, so the launcher's pin stands. This is the branch a
  // caller reaches through `routeRefresher`, which spreads the whole resolved
  // config, so a key `resolveConfig` never emits would land here as `undefined`
  // and silently become 262144/32768 whatever the operator had narrowed to.
  assert.equal(updates[0].patch.providers.tinytitan.defaultContextWindow, 262144);
  assert.equal(updates[0].patch.providers.tinytitan.defaultMaxTokens, 32768);
  assert.equal(updates[0].patch.providers.tinytitan.models[0].contextWindow, 262144);
  assert.equal(updates[0].patch.providers.tinytitan.models[1].maxTokens, 32768);
});

test("a declared window and cap reach the settings patch at both levels", async (t) => {
  try {
    await import("js-yaml");
  } catch (error) {
    if (error?.code === "ERR_MODULE_NOT_FOUND") {
      t.skip("js-yaml is not installed here; the harness's own copy resolves it");
      return;
    }
    throw error;
  }
  const updates = [];
  const result = await applyRouteThroughSettings({
    serverBinary: "/x/TinyTitanServer",
    modelsDir: "/models",
    env: { PATH: "" },
    isExecutable: () => true,
    isDirectory: () => true,
    run: () => JSON.stringify({ models: FAKE }),
    context: 131072,
    maxTokens: 8192,
    settings: { update: async (ns, patch) => updates.push({ ns, patch }) },
    log: () => {},
  });
  assert.equal(result.status, "applied");
  const provider = updates[0].patch.providers.tinytitan;
  assert.equal(provider.defaultContextWindow, 131072);
  assert.equal(provider.defaultMaxTokens, 8192);
  // Every model row carries the same pair, as the shell tool's does.
  for (const model of provider.models) {
    assert.equal(model.contextWindow, 131072);
    assert.equal(model.maxTokens, 8192);
  }
});

test("applyRouteThroughSettings reports a missing settings service", async () => {
  const messages = [];
  const result = await applyRouteThroughSettings({
    serverBinary: "/x/TinyTitanServer",
    modelsDir: "/models",
    env: { PATH: "" },
    isExecutable: () => true,
    isDirectory: () => true,
    run: () => JSON.stringify({ models: FAKE }),
    log: (message) => messages.push(message),
  });
  assert.equal(result.status, "failed");
  assert.ok(messages.some((message) => message.includes("no settings service")));
});

/**
 * A settings service that reads back one profile form, the way the harness's
 * `describe()` does: `{ns, user}` is the user layer, which is the only layer
 * that counts as a choice.
 */
function stubSettings(user, { fail = false } = {}) {
  const updates = [];
  return {
    updates,
    describe: () => {
      if (fail) throw new Error("the settings service is not readable");
      return [{ ns: DEFAULT_MODEL_SETTINGS_NS, user }];
    },
    update: async (ns, patch) => {
      updates.push({ ns, patch });
    },
  };
}

test("a default model the route no longer serves is repointed", async () => {
  // The real case: the install moved from a 4B dense model to the 125B, the
  // route refresh updated the picker, and `agent-default-model` went on naming
  // the old id. The page looked configured and every turn failed with
  // `pi-ai provider "tinytitan" has no configured model qwen3.5-4b_4-Bit`.
  const settings = stubSettings({ provider: "tinytitan", model: "gone_4-Bit" });
  const messages = [];
  const result = await ensureDefaultModel({
    settings,
    ids: ["alpha_4-Bit", "beta_4-Bit"],
    log: (message) => messages.push(message),
  });
  assert.equal(result.status, "repaired");
  assert.equal(result.replaced, "gone_4-Bit");
  assert.equal(settings.updates.length, 1);
  assert.equal(settings.updates[0].ns, DEFAULT_MODEL_SETTINGS_NS);
  assert.deepEqual(settings.updates[0].patch, { provider: "tinytitan", model: "alpha_4-Bit" });
  // The name that was broken is named in the log, so a person can see what the
  // refresh decided and why it had to.
  assert.ok(messages.some((message) => message.includes("replaces gone_4-Bit")));
});

test("a default that names a served model is left alone", async () => {
  // Not the first id: a person who chose the second install made a choice, and
  // a refresh that "corrects" it to the first would take their selection away.
  const settings = stubSettings({ provider: "tinytitan", model: "beta_4-Bit" });
  const result = await ensureDefaultModel({
    settings,
    ids: ["alpha_4-Bit", "beta_4-Bit"],
  });
  assert.equal(result.status, "kept");
  assert.equal(settings.updates.length, 0);
});

test("another provider's default is not ours to rewrite", async () => {
  const settings = stubSettings({ provider: "deepseek-official", model: "deepseek-flash" });
  const result = await ensureDefaultModel({ settings, ids: ["alpha_4-Bit"] });
  assert.equal(result.status, "kept");
  assert.equal(settings.updates.length, 0);
});

test("a default that was never set is the installer's job", async () => {
  // Writing one here would be inventing a choice, and an install-time route writer
  // already owns first-run defaults.
  for (const user of [undefined, {}, { provider: "tinytitan" }]) {
    const settings = stubSettings(user);
    const result = await ensureDefaultModel({ settings, ids: ["alpha_4-Bit"] });
    assert.equal(result.status, "skipped");
    assert.equal(settings.updates.length, 0);
  }
});

test("an unreadable settings service leaves the default alone", async () => {
  const settings = stubSettings({ provider: "tinytitan", model: "gone_4-Bit" }, { fail: true });
  const result = await ensureDefaultModel({ settings, ids: ["alpha_4-Bit"] });
  assert.equal(result.status, "skipped");
  assert.ok(result.reason.includes("not readable"));
  assert.equal(settings.updates.length, 0);
});

test("a failed default write does not fail the route refresh", async () => {
  // The route is already applied at this point; reporting the whole refresh as
  // failed would hide that and send the caller down the "leaving the route as it
  // is" path.
  const settings = stubSettings({ provider: "tinytitan", model: "gone_4-Bit" });
  settings.update = async (ns) => {
    if (ns === DEFAULT_MODEL_SETTINGS_NS) throw new Error("the profile is read-only");
  };
  const messages = [];
  const result = await ensureDefaultModel({
    settings,
    ids: ["alpha_4-Bit"],
    log: (message) => messages.push(message),
  });
  assert.equal(result.status, "failed");
  assert.ok(messages.some((message) => message.includes("could not repoint the default model")));
});

test("the route refresh repairs a stale default in the same pass", async (t) => {
  try {
    await import("js-yaml");
  } catch (error) {
    if (error?.code === "ERR_MODULE_NOT_FOUND") {
      t.skip("js-yaml is not installed here; the harness's own copy resolves it");
      return;
    }
    throw error;
  }
  const settings = stubSettings({ provider: "tinytitan", model: "gone_4-Bit" });
  const result = await applyRouteThroughSettings({
    serverBinary: "/x/TinyTitanServer",
    modelsDir: "/models",
    env: { PATH: "" },
    isExecutable: () => true,
    isDirectory: () => true,
    run: () => JSON.stringify({ models: FAKE }),
    settings,
    log: () => {},
  });
  assert.equal(result.status, "applied");
  assert.equal(result.defaultModel.status, "repaired");
  assert.equal(settings.updates.length, 2);
  assert.equal(settings.updates[0].ns, "llm-pi-ai");
  assert.equal(settings.updates[1].ns, DEFAULT_MODEL_SETTINGS_NS);
  // The ids compared against are the ones just written, not a re-read of the
  // file: the repair must agree with the picker in the same boot.
  assert.deepEqual(
    settings.updates[1].patch,
    {
      provider: "tinytitan",
      model: settings.updates[0].patch.providers.tinytitan.models[0].id,
    },
    "the default names a model the refreshed route does not serve",
  );
});

/**
 * A profile whose route serves `alpha_4-Bit` and `beta_4-Bit`, with the default
 * block above it. This is the shape a prior install leaves behind: a real
 * `agent-default-model` section and a generated `llm-pi-ai` block.
 */
function profile(defaultLines) {
  return `ui-theme:\n  preference: dark\n\n${defaultLines}\n${generateBlock(FAKE)}`;
}

const DEAD_DEFAULT = "agent-default-model:\n  provider: tinytitan\n  model: gone_4-Bit\n";

test("a dead default in a settings file is repointed to a served model", () => {
  // The file-branch case (AUD-172): a profile with no settings service never
  // reaches `ensureDefaultModel`, so the refresh rewrote `llm-pi-ai` and left
  // `gone_4-Bit` as the default. The picker showed the live model and every turn
  // came back `UNKNOWN_MODEL`.
  const before = profile(DEAD_DEFAULT);
  const result = applyDefaultModelToSettings(before);
  assert.equal(result.status, "repaired");
  assert.equal(result.from, "gone_4-Bit");
  assert.equal(result.to, "alpha_4-Bit");
  assert.ok(result.text.includes("  model: alpha_4-Bit\n"), result.text);
  assert.ok(!result.text.includes("gone_4-Bit"), "the dead id must be gone");
  // Only that one line moves: the person's own theme and provider stay, and the
  // route block is untouched.
  assert.ok(result.text.startsWith("ui-theme:\n  preference: dark\n\n"));
  assert.ok(result.text.includes("  provider: tinytitan\n"));
  assert.equal(count(result.text, "llm-pi-ai:"), 1);
});

test("the repair leaves a default that names a served model alone", () => {
  // Not the first id: choosing the second install is a choice, and a refresh that
  // "corrects" it would take the selection away.
  for (const id of ["alpha_4-Bit", "beta_4-Bit"]) {
    const result = applyDefaultModelToSettings(
      profile(`agent-default-model:\n  provider: tinytitan\n  model: ${id}\n`),
    );
    assert.equal(result.status, "kept", id);
    assert.equal(
      result.text,
      profile(`agent-default-model:\n  provider: tinytitan\n  model: ${id}\n`),
    );
  }
});

test("another provider's default in a settings file is not ours to rewrite", () => {
  const before = profile(
    "agent-default-model:\n  provider: deepseek-official\n  model: deepseek-flash\n",
  );
  const result = applyDefaultModelToSettings(before);
  assert.equal(result.status, "kept");
  assert.ok(result.reason.includes("deepseek-official"), result.reason);
  assert.equal(result.text, before);
});

test("a default that was never set, and a route that serves nothing, are skipped", () => {
  // Three shapes where writing a default would be inventing a choice or where
  // there is nothing to compare against: no block at all, a block with no
  // `model:` line, and a file whose route names no model.
  const cases = [
    {
      text: `ui-theme:\n  preference: dark\n\n${generateBlock(FAKE)}`,
      reason: "no default model is set",
    },
    {
      text: profile("agent-default-model:\n  provider: tinytitan\n"),
      reason: "the block names no model",
    },
    { text: "agent-default-model:\n  model: gone_4-Bit\n", reason: "the route serves no models" },
  ];
  for (const one of cases) {
    const result = applyDefaultModelToSettings(one.text);
    assert.equal(result.status, "skipped", one.text);
    assert.equal(result.reason, one.reason);
    assert.equal(result.text, one.text);
  }
});

test("the default block is found whichever side of the route it sits on", () => {
  // The scan stops at the next column-0 key, so a block *after* `llm-pi-ai:`
  // must be read as carefully as one before it.
  const route = generateBlock(FAKE);
  const result = applyDefaultModelToSettings(`${route}\n${DEAD_DEFAULT}`);
  assert.equal(result.status, "repaired");
  assert.equal(result.to, "alpha_4-Bit");
  assert.ok(result.text.includes("  model: alpha_4-Bit\n"), result.text);
  assert.ok(result.text.startsWith("# DeepSeek Harness route to the "), "the route is untouched");
});

test("repairDefaultModelSettings rewrites the file once and backs it up", () => {
  const home = mkdtempSync(join(tmpdir(), "dsh-tinytitan-default-"));
  const path = join(home, "settings.yaml");
  writeFileSync(path, profile(DEAD_DEFAULT));
  const result = repairDefaultModelSettings({
    settingsPath: path,
    stamp: "2026-10-06T00-00-00-000Z",
  });
  assert.equal(result.status, "repaired");
  assert.equal(result.backup, `${path}.bak-2026-10-06T00-00-00-000Z`);
  assert.ok(existsSync(result.backup), "the person's file is kept before the write");
  assert.ok(readFileSync(path, "utf8").includes("  model: alpha_4-Bit\n"));
  // Nothing dead left, so the second pass is a no-op that makes no backup.
  const again = repairDefaultModelSettings({ settingsPath: path, stamp: "second" });
  assert.equal(again.status, "kept");
  assert.equal(again.backup, undefined);
  assert.equal(existsSync(`${path}.bak-second`), false);
});

test("repairDefaultModelSettings says so when there is no file to read", () => {
  const result = repairDefaultModelSettings({
    settingsPath: join(tmpdir(), "dsh-tinytitan-definitely-not-here", "settings.yaml"),
  });
  assert.equal(result.status, "skipped");
  assert.ok(result.reason.includes("no settings file"));
});

test("the built-in writer repairs a stale default in the same pass", () => {
  // AUD-172: the settings-service branch has done this since AUD-163, and the
  // file branch is what a profile with no settings service reaches.
  const { settings, result } = throughGenerator({ settingsText: profile(DEAD_DEFAULT) });
  assert.equal(result.defaultModel.status, "repaired");
  assert.equal(result.defaultModel.from, "gone_4-Bit");
  assert.ok(settings.includes("  model: alpha_4-Bit\n"), settings);
  assert.ok(!settings.includes("gone_4-Bit"), settings);
});

/** A checkout that does carry `tools/dsh_route.sh`, so `registerRoute` takes the script branch. */
function checkoutWithTool() {
  const repoRoot = mkdtempSync(join(tmpdir(), "dsh-tinytitan-repo-"));
  mkdirSync(join(repoRoot, "tools"), { recursive: true });
  writeFileSync(join(repoRoot, "tools", "dsh_route.sh"), "#!/usr/bin/env bash\n");
  return repoRoot;
}

test("the script branch repairs a stale default after the tool has written", () => {
  const dshHome = mkdtempSync(join(tmpdir(), "dsh-tinytitan-home-"));
  const path = join(dshHome, "settings.yaml");
  // The checkout tool has just refreshed the block; the injected `run` stands in
  // for it, so this pins what `registerRoute` does with the file afterwards.
  writeFileSync(path, profile(DEAD_DEFAULT));
  const messages = [];
  const result = registerRoute({
    repoRoot: checkoutWithTool(),
    port: 8080,
    provider: "tinytitan",
    dshHome,
    run: () => "replaced\n",
    log: (message) => messages.push(message),
  });
  assert.equal(result.status, "written");
  assert.equal(result.defaultModel.status, "repaired");
  assert.ok(readFileSync(path, "utf8").includes("  model: alpha_4-Bit\n"));
  assert.ok(
    messages.some((message) => message.includes("replaces gone_4-Bit")),
    messages,
  );
});

test("a default repair that cannot write does not fail a refresh that wrote", () => {
  // The route block is already on disk. Reporting the whole refresh as failed
  // would undo that, and the caller would then take the "leaving the route as it
  // is" path. The detail goes to the log instead.
  const dshHome = mkdtempSync(join(tmpdir(), "dsh-tinytitan-home-"));
  const path = join(dshHome, "settings.yaml");
  writeFileSync(path, profile(DEAD_DEFAULT));
  chmodSync(path, 0o000);
  const messages = [];
  const result = registerRoute({
    repoRoot: checkoutWithTool(),
    port: 8080,
    provider: "tinytitan",
    dshHome,
    run: () => "replaced\n",
    log: (message) => messages.push(message),
  });
  chmodSync(path, 0o644);
  assert.equal(result.status, "written");
  assert.equal(result.defaultModel.status, "failed");
  assert.ok(messages.some((message) => message.includes("could not repoint the default model")));
});
