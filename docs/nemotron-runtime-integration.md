# Nemotron 3.5 Bolt — where a new family plugs into the runtime

Code-verified map of the TinyTitan runtime as it stands for the
**NVIDIA Nemotron 3.5 Bolt 74B-A9B** port (Mamba-2 + LatentMoE + GQA attention,
88 layers: 40 mamba / 40 MoE / 8 attention, text-only). Everything below was read
from the tree, not recalled; every claim carries a `file:line`.

**Baseline.** Read on branch `nemotron-3.5-bolt` at HEAD `03040160`
("nemotron: record the checkpoint and plan its conversion") on 2026-10-25. The
tree has since returned to `main` (`713c1619`); the three files that commit
touched — `K2GraphProbe.swift`, `K2SparseLayerProbe.swift`,
`K2Commands.swift` — were re-checked against `main` and their line numbers
below are the `main` ones. The companion files
(`docs/nemotron-35-bolt-port.md`, `tools/prepare_nemotron.py`) live on the
`nemotron-3.5-bolt` branch, not on `main`. No source file was modified for this
document.

**Companion records.** `docs/nemotron-ssdai-writer-spec.md` (on `main`) is the
format-side contract: what a Python writer can and cannot emit, and §8 of it is
the same Qwen-assumption list this document meets from the runtime side.
`docs/nemotron-35-bolt-port.md` (on the port branch) holds the checkpoint facts
(geometry, sizes, quantisation). This document does not repeat either; it maps
the *runtime*. Where a claim depends on the port record rather than on code it is
marked **(port record)**.

---

## 0. The load pipeline in one paragraph

`Model.load` is a single straight-line function with four separable stages, and
a new family must satisfy all four:

```
Model.load(directoryURL:device:expecting:…)            Model+Loading.swift:25
  ├─ manifest.json  bounded read + SHA                  :42–59
  ├─ receipt (optional policy)                          :62–99
  ├─ ManifestReader.decode(data:expecting:)             :101  → ManifestReader.swift:47
  │    └─ validate(manifest, against: expecting)        ManifestReader.swift:80
  │         ├─ validateArch        (field-by-field)     :289
  │         └─ validateQuant       (router widths)      :230
  ├─ model_weights.bin / packed_experts/layout.json     :115–148
  ├─ PackedExpertsLayoutReader.decode                   :174
  ├─ ResidentIndexReader.load                           :187
  ├─ Model.validateRuntimeSchema                        :189  → Model+Validation.swift:126
  │    ├─ validateExecutableGeometry                    Model+Validation.swift:73
  │    └─ switch config.family                          :148   ← every family
  ├─ ResidentBuffer (one mmap'd MTLBuffer)              :209
  └─ Model.buildBF16ReadableViews                       :219  → Model+Accessors.swift:315
```

The family does **not** get to skip the format: `manifest.quant` is mandatory
(`Model+Validation.swift:132–135`), `model_weights.bin` and
`packed_experts/layout.json` are both mandatory even for a model with no routed
experts (`Model+Loading.swift:115–121`, `ManifestReader.swift:27–30`), and the
resident index must account for the whole weights file
(`Model+Loading.swift:196–206`).

Trap worth naming up front: `Model.load`'s `expecting:` parameter defaults to
`.qwen36_35B_A3B` (`Model+Loading.swift:28`). A caller that forgets to pass the
resolved architecture gets an `archMismatch` on hidden size, not a family error.

---

## 1. What a new `ModelFamily` must implement

### 1.1 The four contracts

| Contract | Where | What it needs |
| --- | --- | --- |
| **Identity / manifest read** | `ModelFamily` enum `ModelTypes.swift:8`; `ManifestReader.peekIdentity` `ManifestReader.swift:143` | A raw value matching `manifest.arch.family` (`ManifestReader.swift:165–171`). The wire format does **not** validate the string — it is free-form (`SSDAIManifestV1.swift:36–39`) — so the enum is the only gate. |
| **Arch decode** | `ArchConfig.resolved(forFamily:directoryURL:)` `ArchConfig+Manifest.swift:43` | Either a preset in `ArchConfig.knownArchitectures` (`ModelTypes.swift:412–417`) **or** a manifest-declared branch. The guard at `ArchConfig+Manifest.swift:51` is an allow-list of exactly `.qwen35Dense` and `.k2Horizon`; anything else throws `ArchResolutionError.unsupportedFamily`. |
| **Schema validation** | `Model.validateRuntimeSchema` `Model+Validation.swift:126` | A `case` in the family switch (`:148`) plus a family validation body. `validateExecutableGeometry` (`:73`) runs *before* the switch and is not family-specific. |
| **Model handle** | `Model` struct `Model.swift:31`, `Model.init` `:270` | No code change: `Model.load` constructs it (`Model+Loading.swift:226–239`). Resident tensor access goes through `TensorSchema` (`Model+Accessors.swift:24–25`), so new tensor names are a schema file, not an accessor edit. |

### 1.2 Every switch / enum / table that must gain a case

Legend: **compile** = the build fails if missed (exhaustive `switch`, no
`default`); **runtime** = the build passes and the failure is a thrown error;
**silent** = the build passes and the model runs with a default nobody chose.

| # | Site | Kind | If missed |
| --- | --- | --- | --- |
| 1 | `ModelFamily` enum — `ModelTypes.swift:8–42` | enum | `peekIdentity` does not recognise the string and **falls back to qwen36** (`ManifestReader.swift:165–184`), then `archMismatch`, which `Engine.readManifest` flattens to `notAnInstall` (`Engine.swift:269–273`). Misleading, not silent-wrong: the family is never named. |
| 2 | `TensorSchema.schema(for:)` — `TensorSchema.swift:100–109` | compile | new family has no names at all. |
| 3 | `TensorSchema` stored roles + memberwise init — `TensorSchema.swift:16–98` | compile (all six literals) | adding a role means updating `qwen36`, `qwen36MTP`, `qwen38flash`, `qwen38flashMTP`, `qwen35Dense`, `k2Horizon`. A family with no such role points at an impossible name — the `qwen35Dense` precedent (`Qwen35DenseFamily.swift:36,40`). |
| 4 | `ArchConfig.knownArchitectures` — `ModelTypes.swift:412` | runtime | `ArchResolutionError.unsupportedFamily` unless the manifest path is opened. |
| 5 | `ArchConfig.resolved` allow-list — `ArchConfig+Manifest.swift:51` | runtime | same error; this is the single line that decides whether a manifest-declared architecture is readable at all. |
| 6 | `ArchConfig.from(manifest:family:)` + `layerShape` — `ArchConfig+Manifest.swift:61–127`, `:144–220` | runtime (`manifestOmitsField` / `manifestDeclaresInvalidValue`) | every convention (`attnOutputGate`, `ropeNeoxSubdim`, …) must be declared, not defaulted. |
| 7 | `ManifestArch` fields — `ManifestTypes.swift:19–88`; wire type `SSDAIManifestArchV1`; conversion `ManifestWireConversion.swift:21–88`; writer `TinyTitanRepack/Core/Format/SSDAIJSON.swift` | compile at the conversion | new geometry (Mamba shape, latent width, per-layer expert counts/widths) has nowhere to live. This is four files per field group. |
| 8 | `SSDAIManifestV1` validation — `SSDAIManifestV1.swift:280` (`isDense`), `:313` (mask must be 0/1/2), `:291–293` (`numLayers`/`numExperts` must equal the streaming metadata) | runtime (`invalid architecture values`) | a Mamba layer kind cannot be encoded **at all** today; and `arch.numExperts` is forced to the single install-wide `expertsPerLayer`, so Nemotron's 240–496 must be declared as the maximum with smaller per-layer counts in `layout.json` (see B3). |
| 9 | `Model.validateRuntimeSchema` switch — `Model+Validation.swift:148–198` | compile | — |
| 10 | `validateExecutableGeometry` — `Model+Validation.swift:73–124` | runtime | hiddenSize > 2816 with routed experts is refused (`:95–101`); mask value 0 refused (`:83–88`); `numHeads > 32` (`:111`); `headDim > 512` (`:116`). **Nemotron's 4096 hits the first one.** |
| 11 | `validateLayerTensors` / `validateRoleUniformity` / `validateFamilyQuantSupport` — `Model+SchemaValidation.swift:130`, `:21`, `:102` | runtime | only reached if the family branch does *not* return early. K2 returns early (`Model+Validation.swift:160`) and validates its own roles (`Model+Validation+K2.swift:19–81`). |
| 12 | `ManifestReader.validateQuant` switch — `ManifestReader.swift:235–255` | compile | router widths per family. |
| 13 | `ManifestReader.validateArch` + family extension checks — `ManifestReader.swift:289–393`, K2 pattern `:403–438` | runtime (`archMismatch`) | new geometry that differs from the runtime's is refused here — which is the point. |
| 14 | `RealForwardRunner.buildCore` refusal — `RealForwardRunner+BuildCore.swift:27–32` | runtime | without it the runner builds a Qwen-shaped graph from a foreign payload. **A new family must be refused here in phase 1.** |
| 15 | `ModelProfile.table` — `ModelProfile.swift:57–201` | silent | family default: `RuntimeConfiguration.decodeTuning` (`RuntimeConfiguration.swift:127–143`) → 8 GiB cache, prefetch 0; `GenerationDefaults.forFamily` (`SamplerTypes.swift:65–80`) → `house` sampling. |
| 16 | `ModelFamily.reasoningControl` — `ReasoningControl.swift:57–67` | compile | — |
| 17 | `ModelCatalog.probeInstall` switch — `ModelCatalog.swift:239–261` | compile | — |
| 18 | `ModelCatalog.displayNames` — `ModelCatalog+DisplayNames.swift:7–22` | silent | `/v1/models` and the launcher show the raw id. |
| 19 | `DoctorModelScanner.unsupportedInstallReason` — `DoctorModelScan.swift:93–105` | compile | — |
| 20 | `ServerModelIdentity.base` — `ModelSessionPlan.swift:61–70` | compile | — |
| 21 | `ServerInference.defaultPrefillChunkTokens` — `ServerInference.swift:25–54` | silent (`default: fallback`, and the server's load runtime hands it 128 — `RuntimeConfiguration.swift:399`) | short chunk; for a family with no profile row this is the shipped prefill size. |
| 22 | `ArchConfig.layerIsFull` / `layerIsLinear` / `hasLinearAttentionLayers` — `ModelTypes.swift:451–453`; consumers `KVCacheManager+Storage.swift:31–33`, `:71`, `GDNStateManager.swift:56`, `:89` | none (mask 3 reads as false/false) | **a mask value the helpers do not know is treated as full attention for KV sizing** (`maskValue != 0` → `isFull`, `KVCacheManager+Storage.swift:33`). Silent over-allocation. |
| 23 | `CPUModelFamily` — `CPUModelFamily.swift:18–80` (aliases `:25`, `reasoningControl` `:50`, `samplingDefaults` `:75`) | compile *if* CPU is used | `CPUModelFamily.refusal` names the gap (`:37–44`). |
| 24 | Repack family — `RepackModelFamily` `ArchInfo.swift:5–25`; `ArchInfo.load`'s five-value `model_type` switch `ArchInfo.swift:244–263`; `RepackPlanner.passthroughRequirements` `RepackPlanner.swift:20–47`; arch loaders `ArchInfo+Loaders.swift`; writer `SSDAIJSON.swift:147` | compile (planner switch), runtime (`configJsonInvalid`) at `ArchInfo.load` | no install can be produced by the Swift repacker. The port's converter bypasses it by writing from Python (`docs/nemotron-ssdai-writer-spec.md` §8.2), but `--verify-install` still has to accept the result. |

### 1.3 The three silent fallbacks that matter most

1. **Family inference** (`ManifestReader.swift:159–184`): an unrecognised
   `family` string is not an error — it is inferred from layer count, sliding
   window and mask, and defaults to `qwen36`. Nemotron's 88 layers will not
   match the MTP shape check, so it lands on `qwen36` and dies later on hidden
   size.
2. **Profile** (`ModelProfile.swift:206–244`): an untabled `(modelID, bits)`
   pair is *fine* — it takes the family default. Nothing warns.
3. **Sampling** (`SamplerTypes.swift:65–80`): `default: house` — the new
   family's card sampling is silently the house row until a `case` is added.

---

## 2. How `RealForwardRunner` decides it can run a family

### 2.1 The gate is one line

`RealForwardRunner+Requirements.swift` **does not** decide family viability — it
is only checked accessors for optional kernels/buffers (`:12–83`, e.g.
`requireElementwise`, `requireBuffer`). The entire family gate is:

```swift
// RealForwardRunner+BuildCore.swift:27
guard cfg.family != .k2Horizon else {
    throw ModelError.unsupportedArchitecture(
        detail: "K2-Horizon's GPU graph is not implemented yet; …")
}
```

It runs in phase one of the two-phase initializer
(`RealForwardRunner+Capabilities.swift:14–24`), *before* any scratch is sized.
Construction then proceeds unconditionally: profile resolution `:66`, KV cache
`:186`, every kernel `:198–318`, optional subsystems behind `cfg` flags
`:334–415`, RoPE `:416–418`. There is no second gate.

### 2.2 How K2 is refused end-to-end (the template)

| Layer | Behaviour | Ref |
| --- | --- | --- |
| Format | Accepted; mask all-1, `family: "k2_horizon"` is a legal string | `SSDAIManifestV1.swift:36`, `ArchInfo+Loaders+K2.swift:79` |
| Identity | `ModelFamily(rawValue:)` resolves | `ManifestReader.swift:165–171` |
| `Model.load` | Succeeds; validates K2's own schema and **returns early** before the Qwen layer checks | `Model+Validation.swift:149–160`, `Model+Validation+K2.swift:19–81` |
| Runner | Refused in `buildCore` | `RealForwardRunner+BuildCore.swift:27–32` |
| Catalog | Directory skipped with a reason, not listed | `ModelCatalog.swift:250–261` |
| Doctor | Reported as not servable | `DoctorModelScan.swift:95–100` |
| Bench | **Bypasses `RealForwardRunner` entirely** | `K2Commands.swift:147–214` |

The bench bypass is the important half:

```swift
// K2Commands.swift:150–155, 178
let config = try ArchConfig.resolved(forFamily: .k2Horizon, directoryURL: install)
let context = try MetalContext()
let model = try Model.load(directoryURL: install, device: context.device,
                           expecting: config, streamingMode: .pread(slotCount: slots))
…
let graph = try K2GraphProbe(model: model, context: context,
                             maxSequence: ids.count + limit + 1)
```

`K2GraphProbe` is a `package final class` (`K2GraphProbe.swift:11`) that owns one
`K2DenseLayer` per dense layer and one `K2SparseLayerProbe` per sparse layer
(`:29–41`), drives them one at a time, and carries the hidden state as `[Float16]`
between command buffers (`:59–75`). It conforms to **nothing**. That is the whole
seam: `Model` + `MetalContext` + the kernel wrappers, no runner.

### 2.3 Extension points that exist — and what they cannot do

What is genuinely pluggable today:

- **Tensor names** — one file in `Runtime/Family/`, no shared accessor edits
  (`TensorSchema.swift:3–11`).
- **Optional geometry** — `ArchConfig`'s sub-configs default to `.none`
  (`ModelTypes.swift:110–126`): `LinearAttentionConfig`, `HyperConnectionConfig`,
  `SparseIndexerConfig`, `PLEConfig`, `MoVAConfig`.
- **Per-subsystem kernels** — `GDN`, `HyperConnection`, `PLEBlock`, `QSAIndexer`,
  `MoVAValueStage` are separate classes built behind `cfg` flags in `buildCore`
  (`:334–415`) and consumed by `if let` branches in the decode loop.
- **Per-family validation** — one file + one `case` (K2 precedent).
- **Model accessors** — family-prefixed functions (`Model+Accessors.swift:48–105`)
  rather than editing shared ones.

What does **not** exist, and is what Nemotron needs:

- **No layer-block protocol.** The decode loop is one function with inline
  family conditionals: `decodeLayerLoop` (`RealForwardRunner+DecodeLayers.swift:13`),
  `isLinear = cfg.layerIsLinear(L)` (`:47`), `denseFFN = cfg.numExperts == 0`
  (`:55`), PLE/QSA/hyper-connection branches (`:88–130`, `:333–360`). A Mamba
  block is not a variant of any of them.
- **No third state kind.** KV is sized from the mask (`KVCacheManager+Storage.swift:31–77`);
  recurrent state exists only for mask-2 layers (`GDNStateManager.swift:56`)
  and is shaped for Gated-DeltaNet (`numVHeads × valueHeadDim × keyHeadDim`,
  `:56–66`). Mamba-2 needs `d_inner × state` plus a conv tail — same *idea*,
  different shape, and the manager is keyed on `layerIsLinear`.
- **The format cannot encode the block kind.** `SSDAIManifestV1.swift:313`
  accepts mask 0/1/2 only, and `ArchConfig.layerIsFull/Linear` know only 1/2
  (`ModelTypes.swift:451–452`).
- **Prefill, MTP, batched and fused decode would each need the same branch**, in
  `RealForwardRunner+PrefillLayer.swift:372–373`, `+FusedDecode.swift:364`,
  `+PrefillProjection.swift:192–193`, `+MTPVerifyRouted.swift:98`.

### 2.4 Dedicated graph vs extension points — the decision

**Use a dedicated graph class; do not extend `RealForwardRunner`.**

Cost of extending the runner: a fourth mask value threaded through the format,
KV sizing, GDN state, validation, and five separate per-layer loops; a Mamba
state manager; and a latent-MoE path in kernels specialised for `hiddenSize`.
Every one of those is a change to code the shipped families execute, and
the failure mode of getting one wrong is silent (wrong KV allocations, wrong
state layout) rather than a compile error.

Cost of a dedicated graph: one new class tree, unreachable from the server, and
one refusal line in `buildCore`. It leaves every existing path byte-identical.

The K2 port already paid for this decision and documented it
(`docs/plan-k2-horizon.md`, `Model+Validation.swift:149–153`).

---

## 3. The precedent for a text-only CPU path — split in three

**There is no single precedent that does (a) `.ssdai`, (b) `packed_experts`
streaming, and (c) a benchmark command.** The three exist, in three different
places, and that fact drives the plan in §6.

### 3.1 (b) + text-only decode + benchmark: the K2 CPU stack

`K2Snapshot` — `sources/TinyTitan/CPUEngine/K2Snapshot.swift`

| API | Line | Behaviour |
| --- | --- | --- |
| `init(directory:maxBytes:)` | `:54` | reads `config.json`, the safetensors index, maps every shard |
| `Configuration` | `:17–43` | 25 fields from the checkpoint's own config, no runtime enum |
| `has(_:)` / `floats(_:)` | `:150`, `:153` | tensor lookup and bf16/norm read |
| `bits(forStem:)` | `:159` | base width or per-stem override |
| `matrix(_:)` | `:164` | packed `UInt32` weights + bf16 scales/biases → `AffineSnapshot.Matrix` |
| `row(_:_:)` | `:209` | dequantizes one embedding row by hand |
| `WeightCache` | `:263–351` | LRU copy of hot expert matrices; budget `TINYTITAN_K2_CACHE_MB`, default 6 GiB (`:284–288`) |

`K2Model` — `sources/TinyTitan/CPUEngine/K2Model.swift`: `init(snapshot:threads:)`
`:33`, `reset()` `:50`, `step(token:needsLogits:)` `:60` (the whole 48-layer
stack), `generate` `:88`, `project` `:115` (quantized GEMV), `projectFloat`
`:136` (bf16 router), `groupedNorm` `:161`, `topK` `:197`. Attention and MLP
live in `K2Model+Attention.swift` / `K2Model+MLP.swift`.

Critically: **it streams routed experts from safetensors shards, not from
`packed_experts`.** There is no packed-expert reader in `CPUEngine/` at all
(grep for `packed_experts|PackedExpertsLayout` in `sources/TinyTitan/CPUEngine/`
returns nothing). The GPU-side equivalent is `Model.routedExpert(layer:expert:)`
(`Model+LayerStreamers.swift:19`), which returns an `MTLBuffer` slot — unusable
on the CPU.

`SideEngine` — `sources/TinyTitan/CPUEngine/SideEngine.swift`: the serving shape
for a CPU model. `SideEngineModel` protocol (`:194–211`: `reset`, `encode`,
`decode`, `generate`, `endOfSequence`, `contention`, `busyThreads`,
`idleThreads`, `threads`), `SideEngine` actor with a single-flight load
(`:249–338`, `judge` `:280`), and `CPUQwen35SideEngineModel` (`:351–417`) which
is the wrapper that knows how to open **both** a `.ssdai` install and a flat
snapshot (`:358–365`).

### 3.2 (a) `.ssdai` on the CPU: `AffineSnapshot(ssdai:)`

`AffineSnapshot+Loading.swift:113` reads a `.ssdai` install for the CPU engine:
manifest → `ArchConfig`-equivalent `Configuration` → resident index →
`Data(contentsOf:options:.alwaysMapped)` of `model_weights.bin` (`:173`) →
per-tensor widths from `manifest.quantOverrides` (`:186–196`). It is called from
three places that must stay consistent: `CPUQwen35SideEngineModel.init`
(`SideEngine.swift:362`), `TinyTitanBench.loadDenseSnapshot`
(`CPUCommands.swift:194–208`), and the batch/generation commands.

But it hard-refuses any family that is not dense:

```swift
// AffineSnapshot+Loading.swift:120
guard identity.family == .qwen35Dense else {
    throw SafeTensorsFile.Failure.malformed(
        "not a dense install: manifest declares \(identity.family.rawValue)")
}
```

So `.ssdai` reading exists on the CPU, but only for a model with **no** routed
experts and only under `CPUModelFamily.qwen35Dense` (`CPUModelFamily.swift:22`).

### 3.3 (c) the benchmark command pattern

`TinyTitanBench` dispatch is a string switch that runs **before** a Metal
context is created (`TinyTitanBench.swift:36`), then falls through to GPU
micro-benchmarks:

```swift
// CPUCommands.swift:269
static func runCPUCommand(_ name: String, iterations: Int) throws -> Bool {
    switch name {
    case "cpu35gen":   … :292
    case "k2gpu":      … :320   // GPU family probe, creates its own MetalContext
    case "k2":         … :333   // CPU family probe
    …
    default: return false
    }
}
```

`k2` is `K2Commands.runK2(snapshot:prompt:limit:)` (`K2Commands.swift:11`):
snapshot → model → tokenizer (or comma-separated token ids for a tokenizer-less
fixture, `:17–23`) → prefill loop → greedy decode loop → prints
prefill tok/s, decode tok/s, output, and optionally raw logits
(`TINYTITAN_K2_LOGITS`, `:80–82`).
`k2gpu` is `runK2GPU` (`:124`) → `runK2GPUAsync` (`:147`) with the same output
shape plus token ids.

### 3.4 Verdict for Nemotron

| Requirement | Existing precedent | Can Nemotron reuse it? |
| --- | --- | --- |
| (a) `.ssdai` text-only | `AffineSnapshot(ssdai:)` + `CPUQwen35` | **No** — refuses non-dense, and has no expert path |
| (b) `packed_experts` streaming | `Model.routedExpert` + `PreadExpertStreamer` (GPU only) | **Only on GPU** |
| (c) benchmark command | `TinyTitanBench` `k2` / `k2gpu` dispatch | **Yes**, verbatim pattern |

A CPU implementation of Nemotron would need a new packed-expert reader, a new
expert cache (there is no CPU slot cache, no `pread` slot ring, no prefetch
ring) and a new family in `CPUModelFamily` — i.e. the entire `PreadExpertStreamer`
stack re-implemented for CPU, on a model whose 4-bit install is ~46 GB
**(port record)**. The GPU path already has that stack.

---

## 4. What the CLI and server need for a new family

### 4.1 CLI

The CLI is thin and has **no CPU path and no resident-memory flag**:

- `--model <dir>` is the only model input (`Args.swift:165`); usage at `:154–220`.
- `run` always builds `Engine(directory:device:configuration:)`
  (`Run.swift:134–135`), which resolves the family from the manifest and calls
  `Model.load` (`Engine.swift:70–71`, via `resolveArch` `:251` and `readManifest` `:269`).
- The only memory knob is `EngineConfiguration.expertCacheBudgetBytes`
  (`EngineConfiguration.swift:22`); `--ram-budget` is server-only
  (`ServerArguments.swift:339–352`, usage `ServerArguments+Usage.swift:99`).

So "the CLI can serve a new family" reduces to: the engine's load path must
accept it, and `Engine.classify` must map its errors sensibly
(`Engine.swift:224–249` — `archMismatch`/`unsupportedArchitecture` become
`unsupportedFamily` at `:240–241`, so a misconfigured install is reported as a
family problem rather than as a corrupt one).

### 4.2 Catalog

`ModelCatalog.probeInstall` (`ModelCatalog.swift:232–287`):

- family switch `:239` — MTP draft heads skipped `:240–247`, K2 skipped
  `:250–261` (compile error if a case is missed);
- tokenizer required `:262–265`;
- id from `ServerModelIdentity.apiModelID` `:266–269`;
- name from `displayNames[base] ?? base` `:276`;
- engines `:286` — `[.gpu, .cpu]` only for `.qwen35Dense`.

`displayNames` (`ModelCatalog+DisplayNames.swift:7–22`) needs one row; the
catalog's `Entry` needs nothing else.

### 4.3 `ModelProfile`

Key is `(modelID, weightBits)` (`ModelProfile.swift:20–27`); resolution is
family default → table row → env (`:206–244`). For a shipped install you add two
rows (4-bit, 8-bit) to `ModelProfile.table` (`:57`), otherwise the family default
applies silently. `isTabled` (`:252`) and `summary` (`:255`) are how the log says
which happened.

### 4.4 Resident-memory gates

| Gate | Where | Relevance |
| --- | --- | --- |
| Page-cache policy vs physical memory | `Model.init` `Model.swift:295–302`, `PageCachePolicy.resolve` `PageCachePolicy.swift:77` | a 46/88 GB install on a 24 GB host takes the bounded reader automatically |
| Tuned budget clamped to RAM | `RuntimeConfiguration.affordableExpertCacheBudget` `:208–216` | third of RAM below 24 GiB, floor at the tune above |
| `--ram-budget` floor | `minimumProcessTargetBytes = 4 GiB` `:341`, `expertCacheSlotsFitting` `:351–370` | server only |
| Resident floor | `residentFloorBytes` `:328–330` | weights + 512 MiB runtime reserve |
| Slot budget from stride | `expertCacheSlots` `:249–304` | per-slot cost = `expertStride × numLayers`; **for Nemotron this is the number that decides whether 24 GB is viable at all** |

### 4.5 The serving seam does not exist yet

`ServerModelSession` stores a **concrete** `RealForwardRunner`
(`ServerModelSession.swift:71`, init `:160`), and `BatchedLogitProducer`
(`:33–36`), `StreamingMTP` and `RawCompletion` (`:144`) all name the concrete
type. The decode loop itself is already protocol-typed:
`runRawCompletion(producer: any LogitProducer, …)`
(`RawCompletion.swift:92`), with `LogitProducer` / `ContinuableLogitProducer` /
`ChunkedPrefillRunner` defined in `LogitProducer.swift:4–80`.

So a probe graph can drive `runRawCompletion` unchanged if it conforms to
`LogitProducer`, but it **cannot be served** until `ServerModelSession.runner`
becomes an abstraction. That refactor is out of scope for a first text-only
decode and is the reason to ship the bench command first.

---

## 5. Tests that enumerate shipped families/models

| Test | File:line | Assertion that needs a row |
| --- | --- | --- |
| Shipped profile rows | `tests/TinyTitan/Runtime/Configuration/ModelProfileTests.swift:6–11` (`shipped`), `:23` | `#expect(ModelProfile.table.count == 10)` — becomes 12 with two Nemotron widths, and the new id must appear in `shipped` or `everyShippedInstallHasItsOwnRow` fails |
| Display names | `tests/TinyTitanServer/ModelCatalogTests.swift:328–343` | `let shipped = […]` plus `#expect(ModelCatalog.displayNames.count == shipped.count)` — a new name without a new entry fails |
| Catalog probe | `tests/TinyTitanServer/ModelCatalogTests.swift:101–161` | `catalog.entries.map(\.id)` exact list, and the skipped-directory set |
| K2 skip reason | `tests/TinyTitanServer/ModelCatalogTests.swift:166–180` | the precedent for a "not served yet" assertion — clone it for Nemotron |
| Preset count | `tests/TinyTitan/Infrastructure/ModelIO/ArchConfigManifestTests.swift:87–103` | `presetFamilies` list and `#expect(ArchConfig.knownArchitectures.count == 4)` — 5 if Nemotron gets a preset |
| Family schema | `tests/TinyTitan/Runtime/FamilySchemaTests.swift` (whole file) | one pin per family; add `nemotronSchemaMatches…` and, if new roles are added, extend `layerRoles(of:)` `:128–143` so the "no two roles collide" check covers them |
| Doctor | `tests/TinyTitan/CLI/DoctorCommandTests.swift:282–292` | `unsupportedInstallReason(.k2Horizon)` non-nil, `.qwen36/.qwen38flash/.qwen35Dense` nil — add the Nemotron expectation |
| Serving defaults | `tests/TinyTitan/Runtime/K2SurfaceTests.swift:11–44` | the template for a new family's sampling/reasoning pin |
| Reasoning surfaces | `tests/TinyTitan/Runtime/ReasoningControlTests.swift:13–101` | per-family level lists |
| Launcher tables | `benchmark/test_launcher_install.py:39` (`STEM_KEYS`), `:53` (`LAUNCHER_NAMES`), `:283–299` | every stem the launcher can name; a new install needs both tables plus `tools/tinytitan_models.sh` (`:125–178`) and `tools/install_models.sh` `CATALOGUE` |

Not a test but a table that fails loudly at runtime: `ModelProfile.table` and
`ArchConfig.knownArchitectures` are dictionaries — nothing enumerates them.

---

## 6. Integration plan for a text-only Nemotron path

### 6.1 Blockers found in code (before any Nemotron-specific work)

These are hard stops in the current tree, in the order a load hits them:

| # | Blocker | Evidence |
| --- | --- | --- |
| B1 | **No Mamba block kind in the format.** Mask must be 0/1/2; `layerIsFull/Linear` know 1/2; mask 3 would be treated as full attention by KV sizing. The writer spec reaches the same conclusion from the writer side (`docs/nemotron-ssdai-writer-spec.md` §8.1.4). | `SSDAIManifestV1.swift:313`; `ModelTypes.swift:451–452`; `KVCacheManager+Storage.swift:31–33` |
| B2 | **hiddenSize 4096 is refused at load** with routed experts: the MoE staging tile is 2816. | `Model+Validation.swift:95–101`; `Metal/MoE/moe.metal:16` (`kMoEXMaxD = 2816`) |
| B3 | **Per-layer expert widths are not expressible in a verifiable install.** `expertStride` is install-global (`PackedExpertsLayout.swift:61`); the runtime checks every routed layer against one `config.moeIntermediateSize` (`Model+SchemaValidation.swift:265–269`); and `--verify-install` additionally demands `expert.size == expertStride`, `offset == expert * expertStride` and one width for every expert (`PackedExpertLayoutVerification.swift:154–157`, `:146–149`, `:339–348`). So the only shape that both loads and verifies is: stride = the widest layer's expert, every narrower expert padded to it, uniform width, with per-layer counts declared no larger than the manifest's `numExperts` (`SSDAIManifestV1.swift:291–293`). A true per-layer stride is a format extension. Per-layer *counts* are already legal (`LayerLayout.expertsPerLayer`, `PackedExpertsLayout.swift:47`). | as cited |
| B4 | **`relu²` does not exist as an expert activation.** `moe.metal` implements SiLU or `gelu_pytorch_tanh` only, selected by a function constant; `Elementwise` has silu/sigmoid/softplus, no relu². | `Metal/MoE/moe.metal:153–159`; `Kernels/Primitives/Elementwise.swift:18–40` |
| B5 | **Latent MoE**: the routed expert FFN input is the 1,024-wide latent, not the model hidden. The MoE wrapper is built with `specializedD = cfg.hiddenSize` (`RealForwardRunner+BuildCore.swift:266`) and the kernels assume that. | `RealForwardRunner+BuildCore.swift:259–269` |
| B6 | **Router** is grouped (8 groups, top-1 group by top-2 sum), bias-corrected and scaled ×5 — not the runtime's sigmoid top-k. There is a per-family precedent to copy: `K2RouterSelection` is a dedicated kernel (`k2_router_topk_select_kn`) for sigmoid + bias + top-k + normalize + scale, precisely because "those are different arithmetic, not a flag on the same one". Nemotron needs one group-selection step in front of it. | `Kernels/MoE/K2RouterSelection.swift:4–12` (rationale), `:18` (pipeline), `:22–35` (encode); Qwen router `Kernels/MoE/MoE.swift:237`; CPU readback pattern `K2SparseLayerProbe.swift:121–160` |
| B7 | **No RoPE** on the 8 attention layers. This is *good*: leave `ropeNeoxSubdim` false so `buildCore` makes no `RoPE` (`RealForwardRunner+BuildCore.swift:416`), but the format still requires `ropeTheta > 0` (`SSDAIManifestV1.swift:311`) and `validateArch` compares `partialRotaryFactor` (`ManifestReader.swift:317`) — write placeholders and document them. | as cited |
| B8 | **Serving seam** is concrete (`ServerModelSession.swift:71`). Phase 1 cannot serve. | §4.5 |

B1–B3 are format/schema work that must land before a `.ssdai` Nemotron install
can load at all; B4–B6 are probe-graph work; B7 is a convention; B8 is a
deliberate deferral.

### 6.2 Options

**Option A — dedicated GPU probe graph (K2GraphProbe pattern).**
One class tree over `Model` + `MetalContext`, reusing `RMSNorm`,
`AffineQuantGEMV` (4/8), `BF16GEMV`, `AffineQuantEmbeddingLookup`, `Attention`
and `Model.routedExpert`; per-block classes for mamba / latent-MoE / attention;
a `TinyTitanBench nemotron` command; `buildCore` refuses the family.

- New: ~6 files, ~900–1,300 lines.
- Reuses: the whole install/verification path, the expert slot cache, the
  prefetch ring, every quantization kernel.
- Blocked by: B1–B7 (B1–B3 are format work regardless of runtime choice).
- Risk: low to existing families — zero edits to shared decode/prefill loops.

**Option B — extend `RealForwardRunner`.**
Add mask 3, a `MambaStateManager`, latent-MoE branches in decode/prefill/fused/
MTP, and a `relu²` kernel.

- New: fewer *files*, far more *edits to shared code* (`+DecodeLayers`,
  `+PrefillLayer`, `+PrefillProjection`, `+FusedDecode`, `+MTPVerifyRouted`,
  `KVCacheManager+Storage`, `GDNStateManager`, `Model+SchemaValidation`).
- Gains: served by the server with no new seam; reuses chunked prefill, batching,
  prompt cache, snapshots.
- Risk: high — every change is on the path the shipped families execute,
  and the failure modes are silent (`KVCacheManager+Storage.swift:33`).

**Option C — build on `CPUEngine` (K2Model pattern).**
A `NemotronSnapshot`/`NemotronCPUModel` pair, dims from `config.json` or a
`.ssdai` manifest, experts read from `packed_experts` with a new CPU reader.

- New: 3–5 files before the expert stack it would have to grow, no Metal.
- Missing: any CPU packed-expert reader, any CPU slot cache/prefetch/`pread`
  ring, `AffineSnapshot(ssdai:)` refuses non-dense
  (`AffineSnapshot+Loading.swift:120`), `CPUModelFamily` has one case.
- Ceiling: a 46 GB install on a 24 GB host through the page cache, one token at
  a time, no speculative expert reads. Correct for a numerics probe, not for the
  published tok/s the port record asks for.

### 6.3 Recommendation

**Option A, staged: format first, then the probe graph, then the benchmark, then the catalogue.**

Rationale, in one line each:

1. The hard part is the install format (B1–B3), and that work is identical under
   every option — so choose the option that pays it once and adds the least
   runtime risk.
2. The expert-streaming machinery Nemotron needs already exists exactly once, on
   the GPU (`Model.routedExpert` → `PreadExpertStreamer`), and a 74B-A9B on a
   24 GB host cannot run without it.
3. K2 proved the pattern: load + validate in `Model`, refuse in `buildCore`,
   bypass the runner with a probe graph, publish numbers from a bench command,
   then wire serving later (`Model+Validation.swift:149–160`,
   `RealForwardRunner+BuildCore.swift:27`, `K2Commands.swift:147`).
4. Option B's only advantage — server support — is explicitly deferred by the
   port record (benchmarks first, "nothing called supported until … the runtime
   answers from them"), and it can be reached later by either promoting the probe
   graph behind `LogitProducer` or adding the runner branches with measurements
   in hand.

**Phase 0 — format and schema (no Nemotron math yet)**

Touch:
- `sources/TinyTitan/Infrastructure/ModelIO/ModelTypes.swift` — `ModelFamily`
  case (`:8`), `ArchConfig` fields/sub-config for Mamba + latent MoE,
  mask helpers (`:451`), `knownArchitectures` if a preset is used (`:412`).
- `sources/TinyTitanFormat/SSDAIManifestV1.swift` — mask value 3 (`:313`), new
  arch fields, `isDense` (`:280`).
- `sources/TinyTitan/Infrastructure/ModelIO/ManifestTypes.swift` + new fields.
- `sources/TinyTitan/Infrastructure/ModelIO/ManifestWireConversion.swift` — map
  the new fields (`:21`).
- `sources/TinyTitan/Infrastructure/ModelIO/ArchConfig+Manifest.swift` — allow
  the family (`:51`), require its geometry (`:61`, `:144`).
- `sources/TinyTitan/Infrastructure/ModelIO/ManifestReader.swift` — family
  detection (`:165`), quant widths (`:235`), required extension fields (`:403`).
- `sources/TinyTitan/Runtime/KVCache/KVCacheManager+Storage.swift` — mask 3 must
  not allocate KV (`:31–33`, `:71`).
- `sources/TinyTitan/Runtime/KVCache/GDNStateManager.swift` — keep mask 3 out of
  the GDN state (`:56`, `:89`).
- `sources/TinyTitan/Runtime/Inference/Model+Validation.swift` — geometry bound
  (`:95`), family case (`:148`).
- `sources/TinyTitan/Runtime/Family/TensorSchema.swift` + new
  `Runtime/Family/NemotronFamily.swift`.
- `sources/TinyTitan/Runtime/Inference/Model+Validation+Nemotron.swift` — its own
  schema (K2 file as the template).
- `sources/TinyTitan/Metal/MoE/moe.metal` — raise `kMoEXMaxD` to 4096 (`:16`;
  the staging array goes from 2,816 to 4,096 halfs, i.e. 5.6 KB → 8 KB of
  threadgroup memory, inside Metal's 32 KB budget) **or** exempt the family;
  raising is the honest fix and matches `gdn.metal:7`, which is already 4096.
- `sources/TinyTitanRepack/Core/Format/*` — `RepackModelFamily`, `ArchInfo`
  loaders, `SSDAIJSON` writer, planner switch (`RepackPlanner.swift:23`).

Verify: `swift build` (the exhaustive switches will find what is missing), a
synthetic 2-layer install through `Model.load`, `ArchConfigManifestTests`.

**Phase 1 — the probe graph**

Add, under `sources/TinyTitan/Runtime/Inference/`:
- `NemotronGraphProbe.swift` — block sequencing, hidden state as `[Float16]`,
  one command buffer per block (K2GraphProbe.swift:11–148 is the template).
- `NemotronMambaStage.swift` — in_proj (18,560 rows), conv1d + bias, SSM state
  update, gated RMSNorm, SiLU gate, D skip; holds `[d_inner × state]` + conv tail
  per layer.
- `NemotronMoEStage.swift` — latent down-projection (`fc1_latent_proj`), BF16
  router GEMV + group pre-pass + a `K2RouterSelection`-shaped top-9 select
  (or a CPU readback for the first cut), 9 expert FFNs at latent width, relu²,
  `fc2_latent_proj`, shared expert (`5,376`, relu²).
- `NemotronAttentionStage.swift` — GQA 32/2 × 128, scale `128^-0.5`, no RoPE,
  plain FP16 KV cache (the `K2DenseLayer` pattern).
- `NemotronHead.swift` — embedding, final norm, 8-bit head GEMV (the `K2Head`
  pattern).
- A `relu²` path: a small `.metal` function + Swift wrapper, or a
  `FC_MOE_ACT_RELU2` case in `moe.metal:153–159`.

Touch:
- `RealForwardRunner+BuildCore.swift` — one refusal line beside `:27`.
- `Model+Validation.swift` — nothing beyond Phase 0.

**Phase 2 — the benchmark command**

- `sources/TinyTitanBench/NemotronCommands.swift` — `runNemotron(install:prompt:limit:)`
  modelled on `K2Commands.runK2GPU` (`:124–214`): resolve arch → `Model.load`
  with `streamingMode: .pread(slotCount:)` → tokenizer → prefill loop → greedy
  decode → print prefill/decode tok/s and output.
- `sources/TinyTitanBench/CPUCommands.swift` — one dispatch case before
  `default` (`:320–344` is the pattern).
- `sources/TinyTitanBench/TinyTitanBench.swift` — nothing (dispatch happens in
  `runCPUCommand`, `:36`).

**Phase 3 — catalogue/serving (after the numbers exist)**

`ModelCatalog.swift:239`, `ModelCatalog+DisplayNames.swift:7`,
`ModelProfile.swift:57`, `DoctorModelScan.swift:93`,
`ServerModelIdentity` (`ModelSessionPlan.swift:61`), and — only if serving is
wanted — the `ServerModelSession.runner` abstraction (`:71`) plus a
`LogitProducer` conformance on the probe.

### 6.4 Trade-off summary

| | A: probe graph | B: extend runner | C: CPU engine |
| --- | --- | --- | --- |
| New files | 6–7 | 4–6 (stages + state manager) | 3–5 before the expert stack |
| Runtime shared-code edits | 1 line (refusal) | ~10 files on hot paths | 3 (snapshot guard, family enum, bench) |
| Format work (Phase 0) | required | required | required only for a `.ssdai` install; avoidable on a snapshot |
| Reuses expert streaming | ✅ `Model.routedExpert` | ✅ | ❌ must write it |
| Reuses `Model.load`/verification | ✅ | ✅ | ❌ new reader |
| Server-servable | ❌ (later) | ✅ | ❌ |
| 74B on a 24 GB host | ✅ bounded slots | ✅ | ⚠️ page-cache only |
| Risk to shipped families | none | high | low |
| Time to first text | **lowest** | highest | medium |

### 6.5 What not to do

- Do not run a Nemotron payload through `Model.load` before B1–B3: it fails at
  `validateExecutableGeometry` with a message about threadgroup tiles
  (`Model+Validation.swift:95–101`), which reads like a kernel bug.
- Do not add the family to `ModelProfile.table` before a width exists — the row
  is a promise that an install was measured (the KAT-Coder comment,
  `ModelProfile.swift:137–145`).
- Do not let `ModelCatalog` list the install before the bench runs. K2's refusal
  (`ModelCatalog.swift:250–261`) exists because "listing it would send a launch
  at a family whose runner refuses".
- Do not put Mamba layers behind mask 2. `layerIsLinear` drives
  `GDNStateManager` allocation and the GDN kernel path; a Mamba layer there is a
  silently wrong state layout, not an error.

---

## Appendix A — geometry the runtime must be given

From `docs/nemotron-35-bolt-port.md` §1 and `tools/prepare_nemotron.py`
(**port record**, not re-derived here) mapped to runtime fields:

| Nemotron | Runtime field | Note |
| --- | --- | --- |
| hidden 4,096 | `ArchConfig.hiddenSize` | exceeds `maximumThreadgroupTileWidth` (B2) |
| 88 layers, 40/40/8 | `fullAttentionLayerMask` + a new Mamba list, or mask 3 | B1 |
| 32 q / 2 kv × 128, no RoPE | `numHeads`, `numFullKVHeads`, `fullHeadDim`, `ropeNeoxSubdim = false` | `numHeads == Attention.maxQHeads` exactly (`Attention.swift:56`) |
| 240–496 experts, top-9 | `numExperts` (must equal the install-wide `expertsPerLayer`), `topKExperts` | declare 496; per-layer counts go in `layout.json` (`PackedExpertsLayout.swift:47`) and must stay ≤ 496 (`SSDAIManifestV1.swift:291–293`) |
| latent 1,024, expert width 1,536–2,560 | no field; `moeIntermediateSize` is single-valued | B3 |
| shared expert 5,376 | `intermediateSize` / `ffnIntermediate` | |
| relu² | no activation case | B4 |
| router: 8 groups, top-1, bias, ×5 | no fields | a `K2RouterSelection`-shaped variant + a group pre-pass (B6) |
| Mamba d_inner 8,192, 128×64, state 128, conv 4 + bias | no fields; closest analogue `LinearAttentionConfig` | new sub-config |
| vocab 131,072, untied | `vocabSize`, `tieWordEmbeddings = false` | 8-bit head → `AffineQuantGEMV`, not `LMHeadChainInt4` |
| rope_theta absent | `ropeTheta` must be > 0 in the format | B7 placeholder |

## Appendix B — file:line index of every claim

Load: `Model+Loading.swift:25,28,42,101,115,174,187,189,209,219,226`;
`ManifestReader.swift:27,47,80,143,165,230,235,289,317,403`;
`Model+Validation.swift:73,83,95,111,116,126,132,144,148,149,160,203,209`;
`Model+Validation+K2.swift:19`; `Model+SchemaValidation.swift:21,102,130,192,223,265`;
`ModelTypes.swift:8,41,49,110,412,451`; `TensorSchema.swift:16,100`;
`Qwen35DenseFamily.swift:36,40`; `ArchConfig+Manifest.swift:43,51,61,144`;
`SSDAIManifestV1.swift:36,280,291,311,313`; `ManifestWireConversion.swift:21`;
`ManifestTypes.swift:19`; `SSDAIManifestQuantV1.swift:86`;
`PackedExpertLayoutVerification.swift:146,154,339`; `ArchInfo.swift:244`;
`docs/nemotron-ssdai-writer-spec.md` (§3.5, §8).

Runner: `RealForwardRunner+BuildCore.swift:27,66,186,198,259,266,334,416`;
`RealForwardRunner+Capabilities.swift:14`; `RealForwardRunner+Requirements.swift:12`;
`RealForwardRunner+DecodeLayers.swift:13,47,55,88,333`;
`RealForwardRunner+PrefillLayer.swift:372`; `RealForwardRunner+PrefillProjection.swift:192`;
`RealForwardRunner+FusedDecode.swift:364`; `RealForwardRunner+MTPVerifyRouted.swift:98`;
`RealForwardRunner+MTP38.swift:34`; `RealForwardRunner+State.swift:122`;
`KVCacheManager+Storage.swift:31,33,71`; `GDNStateManager.swift:56,89`;
`Attention.swift:56,57`.

K2 probe: `K2GraphProbe.swift:11,21,29,59`; `K2SparseLayerProbe.swift:78,121,149`;
`K2DenseLayer.swift:11`; `K2Head.swift:11`; `K2Commands.swift:11,124,147,150,156,158,178`;
`CPUCommands.swift:194,269,292,320,333`; `TinyTitanBench.swift:36`;
`K2RouterSelection.swift:4,18,22`; `MoE.swift:237`.

CPU: `K2Snapshot.swift:17,54,150,159,164,209,263,284`;
`K2Model.swift:33,50,60,88,115,136,161,197`;
`SideEngine.swift:194,249,280,351,358`; `AffineSnapshot+Loading.swift:113,120,173,186`;
`CPUModelFamily.swift:18,22,25,37,50,75`.

CLI/catalog/profile: `Args.swift:154,165`; `Run.swift:134`; `Engine.swift:70,240,251,269`;
`EngineConfiguration.swift:22`; `ServerArguments.swift:339`;
`ModelCatalog.swift:232,239,240,250,262,266,276,286`;
`ModelCatalog+DisplayNames.swift:7`; `ModelProfile.swift:20,57,137,206,252,255`;
`RuntimeConfiguration.swift:127,208,249,328,341,351`;
`PageCachePolicy.swift:77`; `SamplerTypes.swift:65`;
`ReasoningControl.swift:57`; `DoctorModelScan.swift:93`;
`ModelSessionPlan.swift:61`; `ServerInference.swift:25`;
`ServerModelSession.swift:71,160`; `RawCompletion.swift:92,144`;
`LogitProducer.swift:4,41,70`.

Tests: `ModelProfileTests.swift:6,14,23`;
`ModelCatalogTests.swift:101,166,328,341`;
`ArchConfigManifestTests.swift:87,103`; `FamilySchemaTests.swift:128`;
`DoctorCommandTests.swift:282`; `K2SurfaceTests.swift:11`;
`ReasoningControlTests.swift:13`; `benchmark/test_launcher_install.py:39,53,283`.
