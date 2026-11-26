/**
 * The folder scan must describe a models directory exactly as the server does.
 *
 * The scan exists so a profile that has installed models but has not built the
 * server still gets a picker. That only works if its rows are the server's rows,
 * so the first test below compares the block built from the scan with the block
 * built from `TinyTitanServer --catalog` on the same folder: the mirror is the
 * feature, and a divergence has to fail here rather than change what the picker
 * offers.
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import { scanModelsFolder } from "../src/catalog-scan.js";
import { generateBlock } from "../src/generate.js";

const REPO_ROOT = fileURLToPath(new URL("../../../", import.meta.url));
const MODELS_DIR = join(REPO_ROOT, "models");
const BINARY = join(REPO_ROOT, ".build", "release", "TinyTitanServer");

/** A `.ssdai` install the way the repacker writes one, minus the weights. */
function writeInstall(
  root,
  name,
  { modelID, family, bits, activation = "silu", tokenizer = true } = {},
) {
  const directory = join(root, name);
  mkdirSync(directory, { recursive: true });
  writeFileSync(
    join(directory, "manifest.json"),
    JSON.stringify({
      magic: "SSDAI",
      modelID,
      quant: { routedExpert: { weightBits: bits } },
      arch: { family, hiddenActivation: activation },
    }),
  );
  if (tokenizer) {
    mkdirSync(join(directory, "tokenizer"), { recursive: true });
    writeFileSync(join(directory, "tokenizer", "tokenizer.json"), "{}");
  }
  return directory;
}

/** A converted safetensors snapshot. */
function writeSnapshot(
  root,
  name,
  { modelType = "qwen3_5", bits = 8, modelID, displayName, complete = true } = {},
) {
  const directory = join(root, name);
  mkdirSync(directory, { recursive: true });
  writeFileSync(
    join(directory, "config.json"),
    JSON.stringify({
      model_type: modelType,
      quantization: bits === undefined ? undefined : { bits },
      model_id: modelID,
      display_name: displayName,
    }),
  );
  if (complete) {
    writeFileSync(join(directory, "model.safetensors.index.json"), "{}");
    writeFileSync(join(directory, "tokenizer.json"), "{}");
  }
  return directory;
}

/**
 * An Inkling 2 install: a TPP/1 plugin-backed variant.
 *
 * The Q4 plugin exports its tokenizer from the GGUF header it carries, so that
 * install has no `tokenizer/` folder at all; the NVFP4 plugin supplies no
 * tokenizer hook and needs one. `activation` is `situ`, which is what both
 * shipped manifests declare and what the family's kernels implement.
 */
function writeInkling(
  root,
  name,
  { modelID, activation = "situ", bits = 4, tokenizerFolder = true } = {},
) {
  const directory = join(root, name);
  mkdirSync(directory, { recursive: true });
  writeFileSync(
    join(directory, "manifest.json"),
    JSON.stringify({
      magic: "SSDAI",
      modelID,
      quant: { routedExpert: { weightBits: bits } },
      arch: { family: "inkling2", hiddenActivation: activation },
    }),
  );
  if (tokenizerFolder) {
    mkdirSync(join(directory, "tokenizer"), { recursive: true });
    writeFileSync(join(directory, "tokenizer", "tokenizer.json"), "{}");
  }
  return directory;
}

test("the scan and the server catalog produce the same picker block", (t) => {
  if (!existsSync(BINARY) || !existsSync(MODELS_DIR)) {
    t.skip("no built TinyTitanServer or models/ here; skipping the parity check");
    return;
  }
  const printed = execFileSync(BINARY, ["--catalog", "--models-dir", MODELS_DIR], {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
  const fromServer = JSON.parse(String(printed)).models;
  const fromFolder = scanModelsFolder(MODELS_DIR, { env: {} }).models;
  assert.ok(fromFolder.length > 0, "the parity check needs at least one installed model");
  // The raw rows on both sides: `generateBlock` normalizes its own input, and a
  // row that has already been through `catalogRows` is not a catalog row any
  // more (it carries `levels`, not `quant`/`thinking`), so feeding it back in
  // compares two empty blocks and asserts nothing.
  const block = generateBlock(fromFolder);
  assert.equal(block, generateBlock(fromServer));
  assert.equal(
    [...block.matchAll(/^ {8}- id: /gm)].length,
    fromFolder.length,
    "every scanned model must reach the picker",
  );
});

test("a folder is read the way the server reads it", () => {
  const root = mkdtempSync(join(tmpdir(), "dsh-tinytitan-scan-"));
  writeInstall(root, "qwen3.5_2B_4Bit", {
    modelID: "qwen3.5-2b",
    family: "qwen3_5_dense",
    bits: 4,
  });
  writeInstall(root, "qwen3.5_2B_8Bit", {
    modelID: "qwen3.5-2b",
    family: "qwen3_5_dense",
    bits: 8,
  });
  // An id the manifest already spelled a width into: the suffix is added once.
  writeInstall(root, "kat_4Bit", { modelID: "kat-coder-v2.5-4bit", family: "qwen36", bits: 4 });
  // An id outside the display-name table is listed by its id.
  writeInstall(root, "custom_4Bit", { modelID: "custom-model", family: "qwen36", bits: 4 });
  // The 4.0 Flash family lists like the 3.8 one, under its own name.
  writeInstall(root, "qwen4_4Bit", {
    modelID: "qwen4-omni-120b-think-4bit",
    family: "qwen4flash",
    bits: 4,
  });
  // Not models, for one reason each.
  writeInstall(root, "mtp_4Bit", {
    modelID: "qwen3.8-flash-next-mtp",
    family: "qwen38flash_mtp",
    bits: 4,
  });
  writeInstall(root, "incomplete_4Bit", {
    modelID: "half",
    family: "qwen36",
    bits: 4,
    tokenizer: false,
  });
  writeInstall(root, "gguf_4Bit", {
    modelID: "wrong-activation",
    family: "qwen36",
    bits: 4,
    activation: "gelu",
  });
  writeSnapshot(root, "qwen3.5_9B_8Bit", { modelID: "qwen3.5-9b", displayName: "Qwen 3.5 9B" });
  writeSnapshot(root, "llama_4Bit", { modelType: "llama" });
  writeSnapshot(root, "halfwritten_4Bit", { complete: false });
  mkdirSync(join(root, "loose-file-dir"), { recursive: true });
  mkdirSync(join(root, ".cache"), { recursive: true });
  writeFileSync(join(root, "qwen3.5_2B_4Bit.install.lock"), "");

  const { models, skipped } = scanModelsFolder(root, { env: {} });

  assert.deepEqual(
    models.map((model) => [
      model.id,
      model.backend,
      model.quant,
      model.engines,
      model.thinking.join(","),
    ]),
    [
      // GPU entries first, by id; then the CPU snapshot.
      ["custom-model_4-Bit", "gpu", 4, "gpu", "off,on"],
      ["kat-coder-v2.5_4-Bit", "gpu", 4, "gpu", "off,on"],
      ["qwen3.5-2b_4-Bit", "gpu", 4, "gpu,cpu", "off,on"],
      ["qwen3.5-2b_8-Bit", "gpu", 8, "gpu,cpu", "off,on"],
      ["qwen4-omni-120b-think_4-Bit", "gpu", 4, "gpu", "off,low,medium,xhigh"],
      ["qwen3.5-9b", "cpu", 8, "cpu", "off,on"],
    ],
  );
  assert.deepEqual(
    models.map((model) => model.name),
    [
      "custom-model",
      "KAT-Coder-V2.5-Dev 35B-A3B",
      "Qwen 3.5 2B",
      "Qwen 3.5 2B",
      "Qwen 4.0 Flash (Qwen4-Omni-120B-Think)",
      "Qwen 3.5 9B",
    ],
  );
  // Every directory that is not a model says why, so the caller can log it.
  assert.deepEqual(skipped.map((entry) => entry.path.split("/").pop()).sort(), [
    "gguf_4Bit",
    "halfwritten_4Bit",
    "incomplete_4Bit",
    "llama_4Bit",
    "loose-file-dir",
    "mtp_4Bit",
  ]);
  for (const entry of skipped) assert.ok(entry.reason.length > 0, entry.path);
  assert.ok(skipped.find((entry) => entry.path.endsWith("mtp_4Bit")).reason.includes("MTP"));
  assert.ok(
    skipped.find((entry) => entry.path.endsWith("incomplete_4Bit")).reason.includes("tokenizer"),
  );
  assert.ok(skipped.find((entry) => entry.path.endsWith("llama_4Bit")).reason.includes("llama"));
});

/**
 * The rows `TinyTitanServer --catalog` prints for this repository's installs.
 *
 * Measured on this Mac; the `sampling` and `size_gb` fields are the server's own
 * and do not reach the picker block. Passing the paths in keeps the rows' `path`
 * field honest for the synthetic installs.
 */
function measuredServerRows({ gguf, nvfp4, qwen }) {
  return [
    {
      id: "inkling-2-mini-gguf-q4",
      name: "Inkling 2 Mini UD-Q4_K_XL (Unsloth)",
      family: "inkling2",
      quant: 4,
      backend: "gpu",
      engines: "gpu",
      path: gguf,
      thinking: ["off", "low", "medium", "high", "xhigh"],
      sampling: { temperature: 0.6, top_p: 0.95, top_k: 20 },
      size_gb: 17.7,
    },
    {
      id: "inkling-2-mini-nvfp4",
      name: "Inkling 2 Mini",
      family: "inkling2",
      quant: 4,
      backend: "gpu",
      engines: "gpu",
      path: nvfp4,
      thinking: ["off", "low", "medium", "high", "xhigh"],
      sampling: { temperature: 0.6, top_p: 0.95, top_k: 20 },
      size_gb: 19.6,
    },
    {
      id: "qwen3.5-4b_4-Bit",
      name: "Qwen 3.5 4B",
      family: "qwen3_5_dense",
      quant: 4,
      backend: "gpu",
      engines: "gpu,cpu",
      path: qwen,
      thinking: ["off", "on"],
      sampling: { temperature: 0.7, top_p: 0.8, top_k: 20 },
      size_gb: 3.7,
    },
  ];
}

test("a plugin-backed Inkling 2 install scans as the server describes it", () => {
  // The server serves these through a TPP/1 plugin, not through a `ModelFamily`
  // the generic loader knows: the plugin answers with its own identity, display
  // name, width, reasoning levels and tokenizer hook. The folder scan has to
  // describe them identically or a profile with no built server sees a different
  // picker from one with it. These two rows are the shape `TinyTitanServer
  // --catalog` prints for this repository's installs, measured on this Mac (the
  // `sampling` and `size_gb` fields are the server's own and do not reach the
  // block).
  const root = mkdtempSync(join(tmpdir(), "dsh-tinytitan-inkling-"));
  const gguf = writeInkling(root, "inkling-2-mini_GGUF-Q4", {
    modelID: "inkling-2-mini-gguf-q4",
    tokenizerFolder: false,
  });
  const nvfp4 = writeInkling(root, "inkling-2-mini_NVFP4", { modelID: "inkling-2-mini-nvfp4" });
  const qwen = writeInstall(root, "qwen3.5_4B_4Bit", {
    modelID: "qwen3.5-4b",
    family: "qwen3_5_dense",
    bits: 4,
  });

  const { models, skipped } = scanModelsFolder(root, { env: {} });
  assert.deepEqual(skipped, []);
  assert.deepEqual(
    models.map((model) => [
      model.id,
      model.name,
      model.family,
      model.quant,
      model.backend,
      model.engines,
      model.thinking.join(","),
    ]),
    [
      // The plugin reports the manifest's own id, so no `_4-Bit` suffix; the
      // name is the plugin's display name, not the id; the ladder is the
      // template's (thinking off plus four effort levels, `high` included).
      [
        "inkling-2-mini-gguf-q4",
        "Inkling 2 Mini UD-Q4_K_XL (Unsloth)",
        "inkling2",
        4,
        "gpu",
        "gpu",
        "off,low,medium,high,xhigh",
      ],
      [
        "inkling-2-mini-nvfp4",
        "Inkling 2 Mini",
        "inkling2",
        4,
        "gpu",
        "gpu",
        "off,low,medium,high,xhigh",
      ],
      ["qwen3.5-4b_4-Bit", "Qwen 3.5 4B", "qwen3_5_dense", 4, "gpu", "gpu,cpu", "off,on"],
    ],
  );

  const server = measuredServerRows({ gguf, nvfp4, qwen });
  const block = generateBlock(models);
  assert.equal(
    block,
    generateBlock(server),
    "the picker block must be the server's, not a differently spelled one",
  );
  // Raw rows on both sides: `generateBlock` normalizes its own input, so feeding
  // it rows that already went through `catalogRows` would compare two empty
  // blocks. This pins that the two Inkling rows and the qwen row are really in
  // the block, under the ids the server prints.
  assert.deepEqual([...block.matchAll(/^ {8}- id: (.+)$/gm)].map((match) => match[1]).sort(), [
    "inkling-2-mini-gguf-q4",
    "inkling-2-mini-nvfp4",
    "qwen3.5-4b_4-Bit",
  ]);
  assert.ok(block.includes("          name: Inkling 2 Mini UD-Q4_K_XL (Unsloth)\n"), block);
  assert.ok(
    block.includes(
      "          reasoningEfforts:\n" +
        "            off:\n" +
        "            low: low\n" +
        "            medium: medium\n" +
        "            high: high\n" +
        "            xhigh: xhigh\n",
    ),
  );
});

test("a divergent raw row fails the block parity assertion", () => {
  // The parity test compares generated blocks, so it has to be able to tell two
  // raw row sets apart in the fields the picker renders — not merely count rows
  // or compare ids.
  const rows = measuredServerRows({ gguf: "/models/g", nvfp4: "/models/n", qwen: "/models/q" });
  const block = generateBlock(rows);
  const renamed = rows.map((row) => ({ ...row }));
  renamed[0].name = "Inkling 2 Mini UD-Q4 (copy)";
  assert.notEqual(generateBlock(renamed), block, "a label change must change the block");
  const reladdered = rows.map((row) => ({ ...row }));
  reladdered[1].thinking = ["off"];
  assert.notEqual(
    generateBlock(reladdered),
    block,
    "an effort ladder change must change the block",
  );
});

test("an Inkling 2 install the shipped plugins do not recognise is refused", () => {
  // The plugin probe matches the install's model id prefix, not its directory
  // name and not the family alone: an `inkling2` manifest outside both variants
  // resolves to no plugin and the server leaves it off the list. Guessing an
  // identity for it would offer a model that cannot be served.
  const root = mkdtempSync(join(tmpdir(), "dsh-tinytitan-inkling-"));
  writeInkling(root, "inkling-2-other", { modelID: "inkling-2-other" });
  const { models, skipped } = scanModelsFolder(root, { env: {} });
  assert.deepEqual(models, []);
  assert.equal(skipped.length, 1);
  assert.match(skipped[0].reason, /Inkling 2/);
});

test("two directories claiming one id list the first and skip the second", () => {
  const root = mkdtempSync(join(tmpdir(), "dsh-tinytitan-dupe-"));
  writeInstall(root, "a_4Bit", { modelID: "same-model", family: "qwen36", bits: 4 });
  writeInstall(root, "b_4Bit", { modelID: "same-model", family: "qwen36", bits: 4 });

  const { models, skipped } = scanModelsFolder(root, { env: {} });
  assert.equal(models.length, 1);
  assert.equal(models[0].id, "same-model_4-Bit");
  assert.equal(skipped.length, 1);
  assert.match(skipped[0].reason, /duplicate id same-model_4-Bit/);
});

test("an unreadable folder is reported rather than thrown", () => {
  const { models, skipped } = scanModelsFolder(join(tmpdir(), "dsh-tinytitan-absent-dir"));
  assert.deepEqual(models, []);
  assert.equal(skipped.length, 1);
  assert.match(skipped[0].reason, /cannot list/);
});
