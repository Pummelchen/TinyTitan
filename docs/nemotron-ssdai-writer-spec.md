# The `.ssdai` install format, as written by `TinyTitanRepack`

A code-verified specification of what the repacker **writes**, sufficient to
reproduce an install from Python that `TinyTitanRepack --verify-install` accepts
(and, where noted, that the runtime will also load).

Every claim below is anchored to the source that produces or checks it. Nothing
in this document was inferred from the prose docs alone; where `docs/ssdai-format.md`
and the code disagree, the code is quoted and the discrepancy is called out.

Sources read for this specification:

| Area | File |
| --- | --- |
| manifest / layout JSON writer | `sources/TinyTitanRepack/Core/Format/SSDAIJSON.swift` |
| binary index encoders | `sources/TinyTitanRepack/Core/Format/SSDAIEncoders.swift` |
| source tensor types | `sources/TinyTitanRepack/Core/Format/TensorMetadata.swift` |
| resident file writer | `sources/TinyTitanRepack/Core/Writing/ResidentWriter.swift` |
| plan value types (page size, entry/slice shapes) | `sources/TinyTitanRepack/Core/Planning/RepackPlanTypes.swift` |
| planner (ordering, strides, logical shapes) | `sources/TinyTitanRepack/Core/Planning/RepackPlanner+Planning.swift` |
| LZFSE layer writer | `sources/TinyTitanRepack/Core/Writing/CompressedExpertLayerWriter.swift` |
| rANS layer writer | `sources/TinyTitanRepack/Core/Writing/RANSExpertLayerWriter.swift` |
| shared writer helpers (SHA-256) | `sources/TinyTitanRepack/Core/Writing/WriterCore.swift` |
| layout round-trip validator | `sources/TinyTitanRepack/Core/Format/SSDAILayoutValidator.swift` |
| `--verify-install` | `sources/TinyTitanRepack/Core/Verification/*.swift` |
| wire types + structural validation | `sources/TinyTitanFormat/*.swift` |
| runtime dequantization / schema | `sources/TinyTitan/Infrastructure/ModelIO/Quantization.swift`, `Runtime/Inference/*` |
| reference installs | `models/ornith-1.5_35B_A3B_4Bit/`, `models/k2-horizon-4bit.ssdai/` |

All integers are **little-endian**. Page size is **16 384 bytes** everywhere
(`Layout.pageBytes`, `RepackPlanTypes.swift:9`; `SSDAIFormatV1.alignmentBytes`).
"No alignment" below means literally no padding.

---

## 0. Container layout and the minimum file set

```
<install>/
  manifest.json                     required, read first, hashed into the receipt
  model_weights.bin                 required by --verify-install (header+index decoded)
  packed_experts/
    layout.json                     required by --verify-install, even for dense installs
    layer_00.bin … layer_NN.bin     one per layer whose expertsPerLayer > 0
    layer_NN.idx / layer_NN.ridx    only for a compressed install
  tokenizer/
    tokenizer.json                  required by the runtime, ignored by --verify-install
    tokenizer_config.json           required by the runtime
    chat_template.jinja             optional
    special_tokens_map.json         optional
    config.json                     sidecar copied by the installer
  verified-install.json             receipt; written by install *and* by --verify-install
```

Files the manifest may **never** declare (`SSDAIInstallFileNames.reserved`,
`SSDAIFormatV1.swift:127-132`, enforced in `SSDAIManifestV1.swift:346-352`):
`manifest.json`, `verified-install.json`, or anything under them. The tokenizer
*directory* name alone is also refused as a `files` key (`key != "tokenizer"`).

`verified-install.json` is not hashed by anything and is rewritten by
`--verify-install` on every run (`VerifiedInstallTool.swift:96-99`), so a Python
writer may omit it entirely.

**Minimum set accepted by `--verify-install`:**

1. `manifest.json`
2. `model_weights.bin`
3. `packed_experts/layout.json`
4. `packed_experts/<layer.file>` for every layer with a non-zero expert count

`tokenizer/` is not required by the verifier (it is skipped wholesale by
`SSDAIDirectoryAccess.collectEntries`, `SSDAIDirectoryAccess.swift:218-221`, and
by `findUnexpectedEntries`, `VerifiedInstallTool.swift:376-379`).

---

## 1. `manifest.json`

Written by `SSDAIJSON.encodeManifest` (`SSDAIJSON.swift:30-70`) with
`JSONSerialization` options `[.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]`.

### 1.1 Top-level fields

| Field | JSON type | Required | Writer's value / source |
| --- | --- | --- | --- |
| `magic` | string | yes | `"SSDAI"` (`SSDAIJSON.swift:10`); reader also accepts `"GTURBO"` (`SSDAIFormatV1.swift:14-19`) |
| `versionMajor` | integer | yes | `1` (`SSDAIFormatV1.versionMajor`) |
| `versionMinor` | integer | yes | `1` (`SSDAIFormatV1.versionMinor`) |
| `flags` | object of bool | yes | exactly `streamingPresent`, `turboQuantKV`, `aneSharedExpert` (`SSDAIJSON.swift:53-57`); the latter two are hard-coded `false` (`SSDAIJSON.swift:382-389`) |
| `modelID` | string | yes | non-empty; `modelIDOverride ?? plan.matchedModelID ?? "unknown/snapshot"` (`RemoteStreamingRepacker+Output.swift:228`) |
| `sourceSnapshotHash` | string | optional | `"sha256:" + <hex sha256 of model.safetensors.index.json>` (`RemoteStreamingRepacker+Output.swift:229`) |
| `arch` | object | yes | see 1.2 |
| `quant` | object | yes for `--verify-install` | see 1.3 |
| `files` | object path → `{size, sha256}` | yes | see 1.4 |
| `expertsPerLayer` | integer | yes | first non-empty layer's count, else `0` (`RemoteStreamingRepacker+Output.swift:231-232`) |
| `numLayers` | integer | yes | `arch.numLayers` |
| `expertStride` | integer (u64) | yes | first non-empty layer's stride, else `0` (`RemoteStreamingRepacker+Output.swift:234`) |

Format-level constraints (`SSDAIManifestCodec.validate`, `SSDAIManifestV1.swift:262-381`):

- `versionMajor == 1`, `versionMinor >= 0`.
- Unknown `flags` keys are rejected (`:273-276`); `{}` is legal.
- `modelID` non-empty; `numLayers > 0` (`:281-283`).
- **Dense gate** (`:280-286`): if `arch.family == "qwen3_5_dense"` then
  `expertsPerLayer == 0` **and** `expertStride == 0`; *for every other family*
  `expertsPerLayer > 0` **and** `expertStride > 0`. `expertStride` must be a
  multiple of 16 384.
- `arch.numLayers == numLayers` and `arch.numExperts == expertsPerLayer` (`:291-295`).
- `arch` value constraints (`:297-317`): all of `hiddenSize`, `ffnIntermediate`,
  `numHeads`, `numKVHeads`, `numFullKVHeads`, `headDim`, `fullHeadDim`,
  `vocabSize` > 0; `slidingWindow >= 0`; `moeIntermediateSize > 0` unless dense;
  `topKExperts > 0 && <= numExperts` unless dense; finite, positive
  `ropeTheta`/`fullRopeTheta`; `partialRotaryFactor ∈ [0,1]`;
  `fullAttentionLayerMask.count == numLayers` and every element ∈ {0,1,2}.
- `quant` slots, when present, must have `weightBits ∈ 1...32`, `groupSize > 0`,
  non-empty `scheme`/`scaleType`/`biasType` (`:318-335`).

Note `sourceSnapshotHash` has **no** validator; it is free text.

### 1.2 `arch` object

Read back by `SSDAIManifestArchV1` (`SSDAIManifestV1.swift:13-206`). Non-optional
keys (a Python writer must emit all of them):

| Key | JSON type | Notes |
| --- | --- | --- |
| `hiddenSize` | int | > 0 |
| `ffnIntermediate` | int | dense FFN width; > 0 |
| `moeIntermediateSize` | int | per-expert FFN; > 0 unless dense |
| `numHeads` | int | |
| `numKVHeads` | int | |
| `numFullKVHeads` | int | |
| `headDim` | int | |
| `fullHeadDim` | int | |
| `vocabSize` | int | |
| `slidingWindow` | int | `>= 0`; `0` means "none" in every shipped install |
| `finalLogitSoftcap` | double | finite; `0` in both reference installs |
| `ropeTheta` | double | |
| `fullRopeTheta` | double | |
| `partialRotaryFactor` | double | |
| `numLayers` | int | must equal top-level `numLayers` |
| `numExperts` | int | must equal top-level `expertsPerLayer` |
| `topKExperts` | int | |
| `tieWordEmbeddings` | bool | |
| `attentionKEqV` | bool | |
| `hiddenActivation` | string | non-empty (`"silu"`) |
| `fullAttentionLayerMask` | array[int] | length `numLayers`, values 0/1/2 (0 sliding, 1 full, 2 gated-DeltaNet — `ArchInfo.swift:30-31`) |

Optional keys, all written by the repacker (`SSDAIJSON.swift:103-115`, always,
for every family):

`family` (string), `attnOutputGate`, `attentionScale`, `embeddingScaledBySqrtHidden`,
`routerScaled`, `ffnSandwichNorms`, `sharedExpertGated`, `ropeNeoxSubdim`
(`bool`/`double` as named), `linearNumKHeads`, `linearNumVHeads`,
`linearKeyHeadDim`, `linearValueHeadDim`, `linearConvKernelSize` (ints).

Extension-geometry block — gate is **geometry, not family**
(`ArchInfo.declaresExtensionGeometry`, `ArchInfo.swift:206-208`:
`hcCount > 0 || indexerBudget > 0 || !pleLayerIndices.isEmpty`;
written at `SSDAIJSON.swift:129-146`):

`hcCount`, `hcLowRank`, `indexerNumHeads`, `indexerNumKVHeads`,
`indexerHeadDim`, `indexerBudget`, `indexerCompressRatio` (int),
`pleLayerIndices` (array[int]), `pleEmbedDim`, `pleConvKernelSize`,
`pleNgramSize`, `pleVocabSizeBase`, `pleHeadsPerNgram`, `pleVocabDivisor` (int),
`routerNormTopK` (bool), `quantGroupSize` (int).

K2-only block (`SSDAIJSON.swift:147-166`): `movaNumExperts`, `movaTopKExperts`,
`normGroups`, `attentionGate` (string), `routerScaling`, `routerHasBias`,
`movaRouterScaling`, `movaRouterHasBias`, `movaActivation` (string),
`denseLayerIndices` (array[int]), `routerNormTopK`.

**Who enforces what.** `--verify-install` runs only `SSDAIManifestCodec.validate`,
which never inspects the optional extension keys — so a Nemotron manifest may
omit all of them and still verify. The *runtime* additionally cross-checks every
declared value against its compiled preset and, for `versionMinor >= 1` and a
family whose contract has the geometry, requires the block
(`ManifestReader.swift:330-360`). See §8.

### 1.3 `quant` object

Writer: `SSDAIJSON.quantObject` (`SSDAIJSON.swift:194-224`).

Five **slots** are always present, each `{weightBits:int, scheme:string,
scaleType:string, biasType:string, groupSize:int}`:

```
embedding, attention, router, sharedExpert, routedExpert
```

Writer's values: `scheme = plan.baseMode` (`"affine"`), `scaleType = "BF16"`,
`biasType = "BF16"`, `groupSize = plan.baseGroupSize` (`64`).

Slot widths are **derived from the payload**, not from config
(`RemoteStreamingRepacker+Output.swift:178-222`):

- start at `metadata.baseBits` for embedding/attention/routedExpert, `8` for
  router/sharedExpert;
- every quantized resident tensor updates the slot its name matches
  (`.embed_tokens.weight` → embedding; `.self_attn.{q,k,v,o}_proj.weight` and
  `.linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj}.weight` →
  attention; `.router.proj.weight` / `.mlp.gate.weight` → router;
  `.mlp.shared_expert.{gate,up,down}_proj.weight` → sharedExpert);
- `routedExpert` is finally overwritten from the first layer's first sub-tensor
  `bitsForWeights`.

Then one entry per **quantized resident tensor** is merged into the same object
(`SSDAIJSON.swift:214-222`), keyed by the tensor **stem** — the index name with
the trailing `.weight` removed (`:249-251`). Each override is the same
five-field object with that tensor's real width. Two collisions are refused at
write time (`:252-271`): a stem equal to a slot name, and one stem at two widths.

Unquantized tensors (norms, scalars, `A_log`, `dt_bias`) get **no** entry.

Decoder: `SSDAIManifestQuantV1` (`SSDAIManifestQuantV1.swift:36-140`). The five
slots are a `CodingKeys` enum; every other key is decoded as an override. An
override whose `weightBits` is not 4 or 8 is rejected **at decode**
(`:86-92`). Slots are not restricted to 4/8 by the wire type.

Real examples: Ornith 4-bit → `attention 4, embedding 8, routedExpert 4,
router 8, sharedExpert 4`, 252 overrides; Ornith 8-bit → `attention 8,
embedding 8, routedExpert 8, router 8, sharedExpert 8` (252 overrides);
K2 4-bit → `attention 4, embedding 8, routedExpert 4, router 8,
sharedExpert 8`, 3221 overrides.

### 1.4 `files` object

`SSDAIJSON.encodeManifest` builds it from `audit.outputFiles`
(`SSDAIJSON.swift:44-47`, `RemoteStreamingRepacker+Output.swift:223-225`), i.e.
**every file the run wrote except `manifest.json` and `verified-install.json`**:

- `model_weights.bin`
- `packed_experts/layer_NN.bin` — only layers with `expertsPerLayer > 0`
  (`RemoteStreamingRepacker+Remote.swift:281-288`); a dense layer gets no file
  and no entry
- `packed_experts/layout.json`
- `tokenizer/config.json`, `tokenizer/tokenizer.json`,
  `tokenizer/tokenizer_config.json` (required); `tokenizer/special_tokens_map.json`,
  `tokenizer/chat_template.jinja`, `tokenizer/chat_template.json` (optional, only
  if the source has them) (`RemoteStreamingRepacker+Output.swift:115-144`)
- passthrough files for `qwen38flash`: `ple_constants.json` (required),
  `ngram_table.bin` (optional) (`RepackPlanner.swift:20-47`)
- compressed installs also list `packed_experts/layer_NN.idx` (lzfse) or
  `.ridx` (rans)

Each value is exactly two keys: `{"size": <u64>, "sha256": "<64 hex chars>"}`
(`SSDAIManifestFileV1`, `SSDAIManifestV1.swift:3-11`). The validator requires
exactly 64 hex characters per digest (`:357-367`), rejects filesystem-equivalent
duplicate paths, reserved names, and a file path that collides with another
entry's directory prefix (`:336-380`).

Digests are lowercase hex of plain SHA-256 over the whole file
(`Sha256Stream`, `Sha256Stream.swift:7-19`; compared case-insensitively,
`VerifiedInstallTool.swift:72`).

**Ordering:** irrelevant to every reader; the writer sorts keys with
Foundation's collation (see §7.2).

### 1.5 What "the manifest digest" means

There is **no digest field inside `manifest.json`**. "The manifest digest" is
an external attestation:

- `verified-install.json.manifestSha256`, and
- `verified-install.json.files["manifest.json"].sha256`

both equal **SHA-256 over the exact bytes of `manifest.json`** (no
canonicalization, no whitespace stripping), lowercase hex. Written at install
time by `Sha256Stream.hashFile(path:)` (`RemoteStreamingRepacker+Output.swift:241`),
re-computed by `--verify-install` (`VerifiedInstallTool.swift:44`) and checked
against the receipt by the runtime (`VerifiedInstallReceipt.swift:138-140`).

Consequence: JSON whitespace/key-order are *not* part of the format contract —
only of the *receipt* of whatever bytes you wrote. If a Python writer emits its
own receipt, it must hash its own bytes; if it lets `--verify-install` write the
receipt, formatting is irrelevant.

---

## 2. `model_weights.bin`

Writer: `ResidentWriter` (`ResidentWriter.swift:11-101`) from the plan produced
by `RepackPlanner.planResidentFile` (`RepackPlanner+Planning.swift:140-241`).
Layout: one index region of `indexSize` bytes, then one contiguous payload
region of `residentSize` bytes. `fileSize == indexSize + residentSize` exactly.

### 2.1 Index header — 24 bytes at offset 0

`SSDAIBinary.writeIndexHeader` (`SSDAIEncoders.swift:12-23`); struct `<QQQ`:

| Offset | Type | Field |
| --- | --- | --- |
| 0 | u64 | `indexSize` — header + entry table + string table + zero padding to 16 384 |
| 8 | u64 | `residentSize` — payload bytes after the index |
| 16 | u64 | `entryCount` |

Reader constraints (`SSDAIResidentIndexCodec.decodeRegion`,
`SSDAIResidentIndexV1.swift:59-193`): `indexSize <= 32 MiB`
(`residentIndexMaxBytes`), `indexSize % 16384 == 0`,
`indexSize >= 24`, `24 + 72*entryCount <= indexSize`.
`--verify-install` does **not** require `fileSize == indexSize + residentSize`;
the runtime does (`Model+Loading.swift:196-205`).

### 2.2 Index entry — 72 bytes each at `24 + i*72`

`SSDAIBinary.writeIndexEntry` (`SSDAIEncoders.swift:26-53`); struct
`<IHBBQQIIIIQQQQ`:

| Offset | Type | Field |
| --- | --- | --- |
| 0 | u32 | `nameOffset` — **absolute file offset** of the UTF-8 name |
| 4 | u16 | `nameLen` — byte count, `<= 65535` |
| 6 | u8 | `dtype` (table below) |
| 7 | u8 | reserved, must be 0 |
| 8 | u64 | `fileOffset` — absolute offset of the weight bytes |
| 16 | u64 | `sizeBytes` — weight bytes (`> 0`) |
| 24 | u32 ×4 | `shape[0..3]`, row-major, zero-padded on the right |
| 40 | u64 | `scaleOffset` (0 when absent) |
| 48 | u64 | `scaleSize` (0 when absent) |
| 56 | u64 | `biasOffset` (0 when absent) |
| 64 | u64 | `biasSize` (0 when absent) |

`dtype`: `0 = u32`, `1 = bf16`, `2 = fp16`, `3 = fp32`
(`TensorMetadata.swift:7-12`, `SSDAIFormatV1.DType`).

Reader constraints: reserved byte 0; unknown dtype refused; name range inside
`[tableEnd, indexSize)`; name non-empty, valid UTF-8, unique; `shape[0] > 0`;
zero dimensions only trailing; `scaleSize == 0` implies `scaleOffset == 0`
(and same for bias); every non-empty range inside `[indexSize, indexSize+residentSize)`;
**ranges must not overlap** (`SSDAIResidentIndexV1.swift:95-192`).

Note the shape is **logical (dequantized)**: for a packed weight
`shape[1] = scale_groups * 64`, not the packed word count
(`RepackPlanner+Planning.swift:375-394`). `shape` has exactly 4 slots; a rank-1
tensor is `[n, 0, 0, 0]`, rank-2 `[rows, cols, 0, 0]`. Ranks above 4 are
truncated by the writer (`padTo4`, `:357-370`).

### 2.3 String table

Starts immediately after the entry table at `24 + 72*entryCount`. Names are
concatenated in entry order with **no separators and no padding**
(`RepackPlanner+Planning.swift:151-158`, `ResidentWriter.swift:69-79`). Each
entry's `nameOffset = 24 + 72*entryCount + runningLength`.

After the last name, the rest of `indexSize` is **zero bytes**
(`ResidentWriter.swift:56` initializes the buffer, then rounds `indexSize` up to
16 384).

Verified on the Ornith install: 613 entries, `tableStart = 44 160`,
names total 34 867 bytes, `tableEnd = 79 027`, `indexSize = 81 920`, 2 893 zero
bytes of padding.

### 2.4 Payload region and ordering

`fileOffset`/`scaleOffset`/`biasOffset` are **absolute** (they already include
`indexSize`). Per quantized entry the planner lays out, back to back with **no
per-tensor alignment**:

```
weight bytes  [fileOffset, fileOffset+sizeBytes)
scale bytes   [scaleOffset, scaleOffset+scaleSize)
bias bytes    [biasOffset,  biasOffset+biasSize)
```

(`RepackPlanner+Planning.swift:197-203`). Unquantized entries carry only the
weight bytes and zeros for the four scale/bias fields (`:216-228`).

Entry order is `lmResidentOrdering` (`RepackPlanner+Planning.swift:399-421`):
group 0 `language_model.model.embed_tokens.weight`; group 1 per-layer tensors by
layer then a Qwen slot rank; group 2 everything else; group 3
`language_model.model.norm.weight`; group 4 `language_model.lm_head.weight`.
The order has **no** semantic force — readers look tensors up by name — but a
byte-faithful clone should copy it. (It is a hard-coded Qwen template: K2's
`model.embed_tokens.weight` falls into group 2 and ends up after the layers.)

First entry's `fileOffset` is `indexSize` in both reference installs; the
payload then fills the file exactly to `indexSize + residentSize`.

### 2.5 Quantization: what the bytes mean

The repacker is a **byte copier**, not a quantizer: a source `u32 .weight` is
copied unchanged, and its `.scales`/`.biases` companions are copied into the
entry's `scaleOffset`/`biasOffset` (`docs/ssdai-format.md:383-391`;
`RepackPlanner+Planning.swift:177-213`). So the packing is MLX affine group-64:

- One row is `columns` logical values. Groups of **64 consecutive columns**
  share one scale and one bias.
- Packed weights: `rows * columns * bits / 8` bytes, row-major. 4-bit: two
  unsigned values per byte, **low nibble = even column**, high nibble = odd
  (`dequant_int4.metal:8-12`; `Quantization.swift:35-45, 89-100, 105-120`).
  8-bit: one unsigned byte per value (`Quantization.swift:122-193`).
- Scales: `rows * (columns/64)` BF16 values, row-major, i.e. row `r`'s
  `columns/64` scales are contiguous; same for biases.
- Dequantization the runtime performs:
  `w[r,c] = bf16(scale[r, c/64]) * q[r,c] + bf16(bias[r, c/64])`, with
  `q ∈ [0,15]` (4-bit) or `[0,255]` (8-bit). BF16 is the top 16 bits of an
  fp32 bit pattern (`Quantization.swift:26-28`).

Consistency check the format relies on: `scaleSize == biasSize ==
rows * (columns/64) * 2` (`docs/ssdai-format.md:237-239`); for Ornith's
248 320×2048 embedding that is 15 892 480 bytes each, which matches the file.

The repacker refuses any source whose `quantization.group_size != 64`
(`IndexLoader.swift:91-96`) and only accepts 4- and 8-bit widths at runtime
(`ManifestReader.swift:11-14`, `SSDAIManifestQuantV1.swift:16`).

One sharp edge: `dtype 0` is used for every packed weight **and** the logical
shape must be the dequantized one. `--verify-install` inverts that:
`implied_bits = (sizeBytes/4 * 32) / (shape[0] * shape[1])` and requires exactly
4 or 8 (`VerifiedInstallTool.swift:282-318`).

---

## 3. `packed_experts/layout.json` and `layer_NN.bin`

### 3.1 `layout.json`

Writer: `SSDAIJSON.encodeLayout` (`SSDAIJSON.swift:278-358`), same JSON options
as the manifest. Reader/validator: `SSDAIPackedExpertsLayoutV1`.

Top level:

| Field | Type | Required | Writer |
| --- | --- | --- | --- |
| `expertStride` | u64 | yes | first non-empty layer's stride, else 0; multiple of 16 384 |
| `numLayers` | int | yes | `arch.numLayers` |
| `expertsPerLayer` | int | yes | first non-empty layer's count, else 0 |
| `compression` | string | optional | absent for raw; `"lzfse"` or `"rans"` |
| `layers` | array | yes | one entry per transformer layer, always |

Per layer (`SSDAIJSON.swift:338-343`): `layer` (int), `file` (string, basename
only — no `/`, not `layout.json`), `expertsPerLayer` (int, the layer's own
count; `0` for a dense layer), `experts` (array).

Per expert (`:330-336`): `expert` (int), `offset` (u64), `size` (u64),
`tensors` (object). `physicalRank` is optional and never written by the planner.

Per tensor (`:313-329`), key = role for weights, `role + "_scales"`,
`role + "_biases"`; any other component becomes `role + "_" + component`:

```json
"gate":        {"offset": 0,       "size": 983040, "dtype": "U32",  "shape": [768,2560], "bits": 4},
"gate_scales": {"offset": 983040,  "size": 61440,  "dtype": "BF16", "shape": [768,40]},
"gate_biases": {"offset": 1044480, "size": 61440,  "dtype": "BF16", "shape": [768,40]}
```

`dtype` is the string `"U32"` or `"BF16"` only (`slice.dtype == 0 ? "U32" :
"BF16"`); `bits` is written only for weight slices.

### 3.2 `layer_NN.bin`

A flat array of `expertsPerLayer` fixed-size blobs:
`fileSize == expertsPerLayer * expertStride` (raw). Expert `e` starts at
`e * expertStride`; nothing else is in the file. Within a blob the sub-tensors
are laid out by the planner in this order (`RepackPlanner+Planning.swift:257-331`):

```
gate weights, gate scales, gate biases,
up   weights, up   scales, up   biases,
down weights, down scales, down biases
```

with `offsetInExpertBlob` being the running cursor. The writer never adds
alignment between sub-tensors. `expertStride = roundUpToPage(blobCursor)`, i.e.
the sum of the nine sizes rounded up to 16 384 (`:333`); the tail padding is
zeros because the layer file is `ftruncate`d.

Naming: `layer_%02d.bin` for layer indices < 100 (`:111-112`), so `layer_00.bin`
… `layer_47.bin`. (The runtime accepts `layer_%d.bin` or `layer_%02d.bin`,
`ManifestReader.swift:216-226`, but the writer always zero-pads to two digits.)

### 3.3 A dense layer inside a sparse install

K2 proves the accepted shape: layers 0-2 have `expertsPerLayer: 0`, an empty
`experts` array, and a nominal `file` name; **no `layer_00.bin` is written and
none is listed in the manifest**. Both `--verify-install`
(`PackedExpertLayoutVerification.swift:110-117`) and the runtime
(`ManifestReader.swift:204-226`) skip a layer whose expected size is zero.

### 3.4 A two-role (up/down, no gate) bundle

Representable in the format, verifiable, and **not loadable by this runtime**.

A Python writer can emit six tensors per expert — `up`, `up_scales`, `up_biases`,
`down`, `down_scales`, `down_biases` — with the same ordering rules and
`expertStride = roundUpToPage(sum of the six sizes)`. That passes:

- `--verify-install`'s `PackedExpertBytes.validate`
  (`PackedExpertLayoutVerification.swift:314-382`): it iterates whatever keys
  exist, requires every `U32` tensor's implied width to equal the
  `routedExpert.weightBits` slot, requires every `BF16` tensor's size to equal
  `elements * 2`, and requires the total (rounded up to 16 384) to equal the
  stride. Nothing is role-specific.
- The structural validator (`SSDAIPackedExpertsLayoutV1.swift:118-284`) also
  only checks ranges/overlap, not role names.

What assumes three roles:

- The planner refuses a 2-role source: `guard let gName = bundle["gate"], let
  uName = bundle["up"], let dName = bundle["down"]` else it throws "bundle
  incomplete" (`RepackPlanner+Planning.swift:94-110`).
- `routedExpertRole` only recognizes `.mlp.switch_mlp.` + `.gate_proj`/`.up_proj`/
  `.down_proj` (`RepackPlanner+Planning.swift:12-18`); a Nemotron expert name
  classifies as `unknown` and the planner throws (`RepackPlanner.swift:73`).
- The runtime requires all nine roles with fixed shapes
  (`Model+SchemaValidation.swift:259-338`); a missing one throws
  `"... is missing role gate"` (`:312-316`).
- `ModelExpertIO.routedExpertOffsets` maps the nine offsets and substitutes `0`
  for a missing key (`ModelExpertIO.swift:73-91`) — a silent alias if the schema
  check were ever bypassed.
- `Model.swift:193` and `Model+LayerStreamers.swift:361` read
  `subTensors["gate"]?.bits` for the routed width; `K2MoEStage.swift:51` builds
  three views (`gate`, `up`, `down`).
- Compression is role-agnostic: `ExpertBlobIndex.phase1Bytes(downOffset:…)`
  returns `nil` when there is no `down`, giving one frame per expert
  (`ExpertBlobIndex.swift:415-427`), and the rANS writer iterates
  `expert.tensors.keys.sorted()` (`RANSExpertLayerWriter.swift:34-56`).

### 3.5 Per-layer varying expert counts and widths: what passes where

`SSDAIV1StructuralValidator.validate` (`SSDAIPackedExpertsLayoutV1.swift:118-284`)
allows more than the writer emits:

- **Per-layer counts:** `layerExperts = layer.expertsPerLayer ??
  layout.expertsPerLayer` must satisfy `0 <= layerExperts <=
  layout.expertsPerLayer` and `layer.experts.count == layerExperts` (`:182-190`).
  So a layer may have fewer experts than the install-wide value (or zero), never
  more.
- **Per-expert extents:** `1 <= expert.size <= layout.expertStride`
  (`:224-229`); offsets must be non-overlapping (`:244-250`) and
  `offset + size <= expertsPerLayer * expertStride` (`:236-243`). Offsets are
  **not** required to be `rank * expertStride` — the "tiered" shape is legal
  here, and `expert.physicalRank` may map logical to physical.
- **Per-tensor widths:** `tensor.bits` may be any `1...32` (`:257`) and the U32
  extent only has to fit inside `expert.size` (`:263-270`). So per-expert,
  per-layer mixed widths are structurally legal (hot 4-bit / cold 2-bit).
- **Roles:** no role names are required — a 2-role `up`/`down` bundle passes.

`--verify-install` is **stricter** than the structural validator on width:
`PackedExpertBytes.validate` requires *every* U32 expert tensor's implied width
to equal the single `manifest.quant.routedExpert.weightBits`
(`PackedExpertLayoutVerification.swift:339-348`), unconditionally — so a tiered
4/2-bit layout passes the structural validator and the runtime
(`Model+SchemaValidation.swift:324-328` uses `tensor.bits ?? slot`) but **fails
`--verify-install`**. Uniform width per install is the only shape the verifier
accepts. Likewise per-expert `size` must equal `layout.expertStride` for every
expert (`PackedExpertLayoutVerification.swift:154-157`), and every expert id
must be `< layout.expertsPerLayer` (the install-wide value, `:146-149`).

Practical rule for a Python writer: one width per install, `size ==
expertStride` for every expert, `offset == expert * expertStride`, and
per-layer `expertsPerLayer` either the install-wide value or `0`.

### 3.6 Compressed layers (only if you need them)

`layout.json` gains `"compression": "lzfse"`, each `layer_NN.bin` holds frames,
and a sidecar `layer_NN.idx` (same basename, `.idx`) addresses them:

```
"TTXI" (4) | version u16 = 3 | exponent u8 (log2 blockBytes, 12..20) | reserved u8 0
| count u32 (experts)
| per expert, in decode order: (offset u64, length u32) per block
```

(`ExpertBlobIndex.swift:29-37, 164-190`). Blocks: phase 0 (`gate|up`) = bytes
`[0, down.offset)`, phase 1 = the tail, each cut into `blockBytes`
(default 262 144, power of two 4 KiB…1 MiB)
(`CompressedExpertLayerWriter.swift:9-74, 205-233`). The manifest must list the
`.bin` with its compressed size and the `.idx` with a non-zero size; the `.bin`
must be smaller than `expertsPerLayer * expertStride`
(`PackedExpertLayoutVerification.swift:191-277`).

rANS variant: `"compression": "rans"`, sidecar `layer_NN.ridx`, magic `"TTRX"`,
scale 14, one shared 256-symbol model per block index, block records
`(payloadOffset u64, payloadLength u32, rawOffset u32, symbolCount u32, stride u8,
role u8, reserved u16)` (`ExpertRANSIndex.swift:10-32, 111-143`).

---

## 4. `verified-install.json` (the receipt)

Writer: `VerifiedInstallReceiptWriter.encode`
(`VerifiedInstallReceiptWriter.swift:7-46`); reader:
`VerifiedInstallReceiptReader` (`VerifiedInstallReceipt.swift:76-210`).

| Field | Type | Value |
| --- | --- | --- |
| `schemaVersion` | int | `1` |
| `manifestSha256` | string | SHA-256 of `manifest.json` bytes, lowercase hex |
| `modelDirectoryPath` | string | `URL(fileURLWithPath: outputDir).standardizedFileURL.path` |
| `verificationTimestamp` | string | ISO-8601 UTC, second precision, e.g. `2026-10-17T22:39:14Z` |
| `toolVersion` | string | `"TinyTitanRepack"` at install; `"TinyTitanRepack verify-install"` when re-issued |
| `sourceRepoID` | string | optional; omitted when nil (install: HF repo id, or `"local/snapshot"`) |
| `sourceRevision` | string | install: resolved HF commit; **verify: `manifest.sourceSnapshotHash`** |
| `files` | object | path → `{size, sha256}`, for `manifest.json` **and** every manifest file |

Hashes: SHA-256 over whole-file bytes; sizes are the real `stat` sizes.

**Path binding** (`VerifiedInstallReceipt.swift:178-208`): the runtime compares
`receipt.modelDirectoryPath == directoryURL.standardizedFileURL.path` as exact
strings. `standardizedFileURL` collapses `.`/`..` but does **not** resolve
symlinks, and the comparison is case-sensitive. Moving or renaming the directory
therefore invalidates the receipt ("model directory mismatch"); re-running
`--verify-install` in place re-issues it.

**File-set equality** (`:123-131`): at load, `receipt.files` must be *exactly*
`manifest.files.keys ∪ {"manifest.json"}` — no missing entries, no extras. The
receipt never lists `verified-install.json` itself.

`--verify-install` ignores any pre-existing receipt and overwrites it
(`VerifiedInstallTool.swift:88-99`), so a Python writer does not need to produce
one. If it does (e.g. to pre-empt a runtime check), the recipe is:

1. write `manifest.json`;
2. `sha = sha256(manifest_bytes).hexdigest()`;
3. `files = {path: {size, sha256} for every manifest file}`, plus
   `files["manifest.json"] = {size: len(manifest_bytes), sha256: sha}`;
4. `modelDirectoryPath` = the final absolute path, byte-for-byte.

---

## 5. What `--verify-install` actually checks, in order

Entry point: `VerifiedInstallTool.run` (`VerifiedInstallTool.swift:38-105`),
CLI `TinyTitanRepack --verify-install --input-ssdai <dir>` (`Command/main.swift:286-297`).

1. **Open the root** with `O_DIRECTORY`; `rootPath` = `standardizedFileURL.path`
   (`SSDAIDirectoryAccess.swift:14-22`).
2. **Size and hash `manifest.json`** (before parsing): `fileSize`, then SHA-256.
3. **Decode the manifest twice** (`VerifiedInstallTool.swift:331-345`):
   `SSDAIManifestCodec.decodeUnchecked` + `validate` (all the §1.1 rules), then a
   plain `JSONDecoder` into the verifier's local `Manifest` mirror. Read is capped
   at 64 MiB (`metadataMaxBytes:22`), opened with `O_NOFOLLOW` at every component.
4. **Validate the packed-expert layout** (`PackedExpertLayoutVerification.swift:34-83`):
   - `manifest.files["packed_experts/layout.json"]` must exist;
   - read it (64 MiB cap) and decode the mirror (`expertStride`, `numLayers`,
     `expertsPerLayer` required; `compression` optional string; `layers[].file`
     and `layers[].experts[].tensors` required);
   - `layout.expertStride == manifest.expertStride`,
     `layout.numLayers == manifest.numLayers`,
     `layout.expertsPerLayer == manifest.expertsPerLayer`;
   - `expertStride % 16384 == 0`;
   - `layout.layers.count == layout.numLayers`;
   - `compression`, if present, must be `"lzfse"` or `"rans"`;
   - per layer (`:89-182`): `0 <= layer < numLayers`; `experts.count ==
     (layer.expertsPerLayer ?? layout.expertsPerLayer)`; `file` is a safe
     basename; `expected = count * stride`; if `expected == 0` stop here
     (dense layer); else `manifest.files["packed_experts/<file>"]` must exist
     and its `size` and the on-disk size must both equal `expected` (raw case);
   - per expert: id in range, no duplicate ids, `size == expertStride`,
     `offset % 16384 == 0`, `offset + size <= file size` (raw case);
   - `PackedExpertBytes.validate` (`:314-382`): for every tensor, in sorted key
     order — `U32`: `size*8/elements` must divide evenly, be in 1…32, equal the
     tensor's own `bits` if annotated, and **equal `manifest.quant.routedExpert.weightBits`**;
     `BF16`: `size == elements*2`; sum of all tensor sizes rounded up to 16 384
     must equal `expertStride`.
   - compressed layers additionally: `.bin` smaller than raw, sidecar listed
     with matching size, sidecar decodable, and the last frame ends inside the
     layer file (`:191-277`).
5. **Cross-check the resident widths** (`:118-158`, `:189-274`):
   - `manifest.quant` must exist;
   - `model_weights.bin` must be in `manifest.files`;
   - read 24 bytes → header; `indexSize <= 32 MiB`; read exactly `indexSize`
     bytes; `decodeRegion` runs all the structural rules in §2.2;
   - for each `dtype 0` entry: compute `implied = (size/4*32)/(shape[0]*shape[1])`
     (must divide; must be 4 or 8); resolve `declared` = override by stem, else
     (only when `expertsPerLayer == 0`) the embedding slot for stems ending
     `embed_tokens`/`lm_head` and the attention slot otherwise; else (MoE
     install, no override) **skip the tensor**;
   - for a dense install (`expertsPerLayer == 0`) also require the
     `routedExpert` slot to be one of the widths actually present (ties are
     accepted as a set).
6. **Hash every declared file** (`:49-85`), in `manifest.files.keys.sorted()`
   order: path validation (`..`, absolute, non-normalized, reserved), per-file
   cap 256 GiB, actual size must equal the manifest size, actual SHA-256 must
   equal the manifest digest (case-insensitive).
7. **Find unexpected entries** (`:356-384`) — files on disk that are neither
   declared nor `manifest.json`/the receipt nor under `tokenizer/`; `.DS_Store`
   exempt. **This is reported, not fatal**; `main.swift` doesn't even print it.
8. **Write the receipt** (`:88-99`) with the values in §4 and
   `fileCount = files.count + 1`.

Not checked by `--verify-install` (but checked by the runtime on load):

- `fileSize == indexSize + residentSize` for `model_weights.bin`
  (`Model+Loading.swift:196-205`);
- expert `offset == expert * expertStride`, expert ranges non-overlapping, and
  `packed_experts/*` capacity — those are in the runtime's structural validator
  (`SSDAIPackedExpertsLayoutV1.swift:204-282`) and the writer's own
  `SSDAILayoutValidator` (`SSDAILayoutValidator.swift:58-79`), not in the
  verifier;
- manifest/layout `crossValidate` (manifest file sizes vs expert ranges,
  `SSDAIPackedExpertsLayoutV1.swift:298-391`);
- `arch` extension values, `family` membership, tokenizer presence,
  `modelID` vs catalog;
- any file on disk not named by the manifest.

---

## 6. What a Python writer may omit or default

Legally omitted (must still produce a manifest `--verify-install` accepts):

| Thing | Why it's safe |
| --- | --- |
| `verified-install.json` | rewritten by the verifier; must not be in `files` |
| `tokenizer/*` | verifier skips the directory entirely |
| per-tensor `quant` overrides | a MoE install never resolves a resident width, so nothing is compared; a dense install only needs them where a tensor's width differs from the slot the fallback picks |
| `sourceSnapshotHash` | optional in the wire type; no validator |
| `arch.family` and every optional `arch` key | `SSDAIManifestCodec.validate` never reads them; only the dense gate reads `family` |
| `arch` extension block | same — enforced only by the runtime from `versionMinor >= 1` |
| `flags` contents | `{}` is legal; unknown keys are not |
| per-layer `expertsPerLayer` in `layout.json` | defaults to the install-wide value |
| `expert.expert` | defaults to the array position (but if any entry has it, all must — runtime structural rule, not verifier) |
| per-tensor `bits` in `layout.json` | verifier only compares it when present |
| `compression` | absent = raw |
| `expertStride` padding | must only be a multiple of 16 384 and match the tensor sum rounded up |

Cannot be omitted:

- `manifest.json`, `model_weights.bin`, `packed_experts/layout.json`, and a
  `layer_NN.bin` for every layer with experts;
- `quant` (the verifier's first guard);
- all 22 non-optional `arch` keys;
- the five quant slots;
- `files`, with correct `size` + `sha256` for everything it names;
- every `layout.json` layer entry (one per transformer layer, even dense ones);
- `magic`/`versionMajor`/`versionMinor`/`modelID`/`numLayers`/`expertsPerLayer`/
  `expertStride`/`flags`.

Defaults that matter if you *do* write them:

- `versionMinor: 0` is accepted by the format and would avoid the runtime's
  mandatory extension block; the current writer emits `1`.
- For a **dense** install the only accepted shape is
  `family == "qwen3_5_dense"`, `expertsPerLayer == 0`, `expertStride == 0`.
  Any other `family` string with zero experts is rejected by
  `SSDAIManifestV1.swift:280-286`, which is the single hardest Qwen-specific
  assumption in the format layer (see §8).
- Slots may legally hold `weightBits` 1…32 at the wire level, but per-tensor
  overrides must be 4 or 8, and the resident check requires 4 or 8.

---

## 7. Reproducing the writer's JSON byte-for-byte

Nothing in the format requires this; the receipt binds whatever bytes you wrote.
If you want it anyway, these are the observed rules (verified by round-tripping
the four reference JSON documents, not inferred).

### 7.1 Whitespace

`JSONSerialization` with `.prettyPrinted, .sortedKeys, .withoutEscapingSlashes`
produces, on macOS 26/27 + Swift 6.4:

- 2-space indent per level;
- object entry: `"key" : value` — a space **before and after** the colon;
- items separated by `,\n`; no trailing comma;
- empty object: `{\n\n<indent>}`; empty array: `[\n\n<indent>]` (blank line
  between the brackets);
- no trailing newline at end of file;
- `/` not escaped; UTF-8 emitted raw; `"` and `\` escaped.

Python equivalent:

```python
import json, re

def _collate(s):
    # Foundation's sortedKeys = localizedStandardCompare: case-insensitive,
    # numeric-aware, punctuation-insensitive.
    s = "".join(ch for ch in s.casefold() if ch.isalnum() or ch.isspace())
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", s)]

def dumps_foundation(obj, indent=0):
    pad = "  " * indent
    if isinstance(obj, dict):
        if not obj:
            return "{\n\n" + pad + "}"
        items = ",\n".join(
            f'{pad}  {json.dumps(k, ensure_ascii=False)} : '
            f"{dumps_foundation(obj[k], indent + 1)}"
            for k in sorted(obj, key=_collate))
        return "{\n" + items + "\n" + pad + "}"
    if isinstance(obj, list):
        if not obj:
            return "[\n\n" + pad + "]"
        items = ",\n".join(f"{pad}  {dumps_foundation(v, indent + 1)}" for v in obj)
        return "[\n" + items + "\n" + pad + "]"
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if isinstance(obj, int):
        return str(obj)
    if isinstance(obj, float):
        return str(int(obj)) if obj == int(obj) else repr(obj)   # see caveat
    if isinstance(obj, str):
        return json.dumps(obj, ensure_ascii=False)
    raise TypeError(type(obj))
```

The `_collate` function reproduces the key order of all five reference JSON
documents exactly (top level, `arch`, `quant`, `files`, `layout.json`,
receipt). In particular it puts `layers.2` before `layers.10` (numeric) and
`tokenizer_config.json` before `tokenizer.json` (punctuation-insensitive).

### 7.2 Floats (caveat)

Double formatting is **Foundation-version dependent** and is the one thing you
cannot reproduce portably. The reference manifests contain
`"attentionScale" : 0.08838834764831843` (16 significant digits, i.e. shortest
round-trip), while Swift 6.4's `JSONSerialization` emits
`0.088388347648318433` (17 digits, `%.17g`) for the same value, and
`9.9999999999999995e-08` for `1e-7`. Integral doubles print without `.0`
(`0`, `10000000`), and `-0.0` prints as `-0`.

None of this matters to `--verify-install` (it decodes to `Double`); it matters
only if you are diffing bytes against an install produced by a different
Foundation. Do not chase it: write the manifest, then let the verifier write the
receipt.

---

## 8. Qwen-specific assumptions a Nemotron-H install will trip over

### 8.1 In the format layer (these bite `--verify-install`)

1. **Dense is one family name.** `SSDAIManifestV1.swift:280-286`:
   `let isDense = manifest.arch.family == "qwen3_5_dense"`. A dense Nemotron
   (or a dense install of any other family, or a manifest that omits `family`)
   must therefore declare `expertsPerLayer > 0` and `expertStride > 0` to pass.
   There is no dense representation for any other family.
2. **Uniform expert count.** `SSDAIManifestV1.swift:291-292` requires
   `arch.numExperts == expertsPerLayer`; `RepackPlanner+Planning.swift:252` and
   `SSDAIJSON.swift:351` derive both from a single number. A Nemotron-H with
   different expert counts per layer cannot be expressed, except by declaring
   the maximum install-wide and giving smaller layers a per-layer
   `expertsPerLayer` (allowed by `SSDAIPackedExpertsLayoutV1.swift:182-190`,
   and by the verifier) — a layer cannot declare *more* than the install-wide
   value.
3. **Every layer needs a layout entry.** `layout.layers.count == numLayers`
   (`PackedExpertLayoutVerification.swift:55-58`) and one entry per transformer
   layer; a Mamba/attention mix must still be enumerated, with
   `expertsPerLayer: 0` and no `.bin` for the layers that have none.
4. **`fullAttentionLayerMask` has no value for a Mamba layer.** Only 0/1/2 are
   accepted (`SSDAIManifestV1.swift:312-313`); 2 means "gated-DeltaNet linear
   attention" (`ArchInfo.swift:30-31`). A Nemotron mixer that is neither
   sliding, full, nor GDN has to be encoded as one of these.
5. **The per-tensor override decoder rejects non-4/8 widths**
   (`SSDAIManifestQuantV1.swift:86-92`) even though slot widths up to 32 are
   legal.
6. **The resident width check requires 4 or 8 bits** and inverts the byte extent
   (`VerifiedInstallTool.swift:282-318`), so every resident `dtype 0` tensor must
   be 4- or 8-bit, with `shape[1]` the *logical* column count.

### 8.2 In the Swift writer (you bypass these by writing from Python)

The repacker cannot *produce* a Nemotron install without code changes:

- `RepackModelFamily` has no Nemotron case (`ArchInfo.swift:5-25`) and
  `ArchInfo.load` switches on `model_type` with five accepted values
  (`ArchInfo.swift:244-263`).
- `classify` recognizes only `language_model.*`, `model.*`, `lm_head.*`,
  `mtp.*`, and the qwen38 prefixes (`RepackPlanner.swift:67-155`). An unknown
  prefix throws `unknownTensorPrefix` (`RepackPlanner+Planning.swift:73`).
- Routed experts are recognized only by `.mlp.switch_mlp.` plus
  `.gate_proj`/`.up_proj`/`.down_proj` (`RepackPlanner+Planning.swift:12-18`),
  and a layer without all three throws (`:94-110`).
- Resident ordering and destination renaming hard-code Qwen names:
  `language_model.model.embed_tokens.weight`,
  `language_model.model.layers.<N>…`, `language_model.lm_head.weight`
  (`RepackPlanner+Planning.swift:399-421`), `qwenSlotRank`'s 22 name fragments
  (`:426-450`), and `residentDestinationName`'s `mtp.` → `language_model.model.`
  rewrite (`:454-482`). Names not matching fall into the "other" group — ordering
  only, not correctness — but the *slot derivation* in
  `RemoteStreamingRepacker+Output.swift:184-217` will silently leave the slots at
  their seeds for Nemotron names, and `routedExpert` at `baseBits`.
- `passthroughRequirements` is an exhaustive switch over the family enum
  (`RepackPlanner.swift:20-47`).
- `group_size` must be 64 (`IndexLoader.swift:91-96`).

### 8.3 In the runtime (why `--verify-install` acceptance ≠ loadable)

`--verify-install` can be made to accept a Nemotron install; the engine will
still refuse it:

- `ModelFamily` has six cases and no Nemotron (`ModelTypes.swift:8-42`);
  `TensorSchema.schema(for:)` is an exhaustive switch (`TensorSchema.swift:100-110`).
- `ArchConfig.resolved(forFamily:)` throws `unsupportedFamily` for any family
  without a preset except `qwen35Dense`/`k2Horizon`
  (`ArchConfig+Manifest.swift:49-54`).
- `ManifestReader` requires `model_weights.bin` + `packed_experts/layout.json`
  (`ManifestReader.swift:27-30`), a `quant` block (`:200-204`), and for
  `expertsPerLayer > 0` a `layer_%d.bin`/`layer_%02d.bin` per non-dense layer
  (`:216-226`), where "dense" comes from `arch.denseLayerIndices`.
- `validateQuant` pins `scheme == "affine"`, `scaleType == "bf16"`,
  `biasType == "bf16"`, `groupSize == 64`, and per-family router widths —
  `qwen36` allows only 8-bit (`ManifestReader.swift:230-289`).
- The routed-expert schema demands the three Qwen roles with the exact shapes
  `gate/up = [moeIntermediateSize, hiddenSize]`, `down = [hiddenSize,
  moeIntermediateSize]`, scales/biases `[rows, columns/64]`
  (`Model+SchemaValidation.swift:259-338`).
- `validateExecutableGeometry` bounds hidden size to the kernel tiles
  (`Model+Validation.swift:130-150`).

So the honest split is: **`--verify-install` verifies bytes and arithmetic, not
architecture.** A Python-written Nemotron install can pass it and still be
unloadable. If the goal is only the verifier, §1-§6 is the whole contract; if
the goal is to run it, the runtime needs a new `ModelFamily`, schema and
kernels, which no manifest can supply.

---

## 9. Reference: exact encoder calls

```python
import struct

PAGE = 16384
def page_up(n): return (n + PAGE - 1) // PAGE * PAGE

# ---- model_weights.bin ----
names = [...]                       # entry order, UTF-8
table = b"".join(n.encode() for n in names)
offs, cur = [], 0
for n in names:
    offs.append(cur); cur += len(n.encode())
index_size = page_up(24 + 72*len(names) + len(table))
buf = bytearray(index_size)         # zero padding to the page
struct.pack_into("<QQQ", buf, 0, index_size, resident_size, len(names))
tbl_base = 24 + 72*len(names)
for i, e in enumerate(entries):
    struct.pack_into(
        "<IHBBQQIIIIQQQQ", buf, 24 + 72*i,
        tbl_base + offs[i], len(names[i].encode()), e.dtype, 0,
        e.file_offset, e.size_bytes,
        *e.shape4,                              # [a,b,0,0]
        e.scale_offset, e.scale_size, e.bias_offset, e.bias_size)
buf[tbl_base:tbl_base+len(table)] = table

# ---- layout.json per expert ----
blob = b""
for role in ("gate", "up", "down"):
    for comp in ("weights", "scales", "biases"):
        blob += packed_bytes(role, comp)
expert_stride = page_up(len(blob))
```

`file_offset` for the first entry = `index_size`; each quantized tensor's three
ranges are contiguous (`fileOffset`, then `+sizeBytes` for scales, then
`+scaleSize` for biases); `resident_size = last_end - index_size`;
`file size = index_size + resident_size`.

Per-expert blob offsets are relative to the expert start; expert `e`'s blob is
at `e * expert_stride` in `layer_NN.bin`, and the file is exactly
`expertsPerLayer * expert_stride` bytes.

---

## 10. Known doc/code discrepancies found while writing this

- `docs/ssdai-format.md:162-163` says "Both are page-aligned to 16 KB". Only
  `indexSize` is page-aligned; the payload is contiguous and individual tensor
  offsets are not aligned (`RepackPlanner+Planning.swift:197-203`).
- `docs/ssdai-format.md:295-301` says a dense `layout.json` lists a layer per
  layer with a nominal file. Code agrees, and K2 shows the same shape for its
  three dense layers — but the writer also emits a per-layer `expertsPerLayer`
  key (`SSDAIJSON.swift:341`) that the doc does not mention; older installs
  (Ornith) lack it.
- `docs/ssdai-format.md:36-44` implies `--verify-install` accepts both magics;
  it does, via `SSDAIManifestCodec.validate`, which accepts `"GTURBO"` as well.
- The rANS path is reachable only through `--compress-experts`
  (`CompressExpertsTool.swift`), not through the normal repack flow.
