# TinyTitan 7.0 — one engine path per (model, quantization)

Status: **proposal**, not implemented. Written 2026-10-27 on `ornith-1.5-122b`.
Tracking issue: [#32](https://github.com/Pummelchen/TinyTitan/issues/32).
This document is the design contract for the 7.0 rewrite. It deliberately
reverses several 6.x decisions; where it does, the old decision is named so the
migration does not quietly keep both.

## 1. The non-negotiables

1. **One engine path per (LLM, quantization).** `Ornith-1.5-122B` at 4-bit and
   at 8-bit are two engines, not one engine with a width flag. A performance
   change, kernel change or tuning change for one must not alter the other, nor
   any other model, unless it is separately tested and approved for that
   variant.
2. **Duplication is allowed, and expected.** Code specific to one LLM and one
   quant may be copied. The target is isolation and a small, stable generic
   library — not DRY.
3. **Generic code is a library, not a framework.** Tokenizer, safetensors
   format, SSH/SSD install format, Metal context, HTTP, sampling primitives and
   the server transport are generic. They may be used by every variant. A
   generic component is not the primary product and must not grow model
   knowledge to avoid a copy.
4. **Swift first, scripts last.** Installation, conversion, repacking,
   verification, benchmarking and catalogue/docs generation live in the
   `tinytitan` binary. The runtime and the tools are one executable. Shell and
   Python are reduced to the build itself and, at most, one documented ANE
   export exception (§9).
5. **A variant is qualified, not merely compiled.** An engine path ships with
   its golden output, decode/prefill measurements, memory ceiling and
   Terminal-Bench record for that exact (model, quant, hardware) triple.

## 2. What 6.x does, and why this is a rewrite

6.x has one manifest contract and one quant contract, which is good, but the
model axis is a family switch and a union graph:

- `ModelFamily` is six cases; family-dependent behaviour is spread over twelve
  `switch` sites (schema, validation, graph, sampler, runtime defaults,
  identity, server, catalog).
- Optional features — hyper-connections, PLE, QSA indexer, MoVA — are fields on
  the shared `ArchConfig`/`ManifestArch` and properties on `Model`, so every
  family carries every feature.
- The model catalogue is duplicated across the installer shell, the menu shell,
  Swift display names, the `ModelProfile` table, `SupportedModelSource` and
  seven Python converters.
- Quantization is the cleaner axis: five slots plus per-tensor overrides,
  resolved per role, with `SlotGEMV` centralizing 4/8-bit dispatch. But the
  five slots are MoE-shaped, the dense family compensates with overrides, and
  the main decode still constructs concrete int4/affine kernels per role.
- The 122B addition needed a hybrid geometry resolution because `qwen36` was a
  preset; that is the seam this rewrite removes.

The lesson is not "6.x was wrong about everything". It is that **sharing a
graph across models trades isolation for less code**, and the project has now
decided that trade the other way.

## 3. The 7.0 unit: `EngineVariant`

The unit of code, test, tuning and release is `EngineVariant`, identified by
`<model-id>-<bits>bit`. One variant owns:

| Concern | Owned by the variant |
| --- | --- |
| Graph | its own layer loop, attention/GDN/MoE ordering, residual scheme |
| Kernels | its own Metal functions and threadgroup constants (copied if needed) |
| Quant | its `VariantQuant`: every tensor role and width it reads |
| Geometry | its manifest contract and cross-checks |
| Performance | cache budget, prefetch, chunk size, kernel switches |
| Sampling | temperature/top-k/top-p/penalties/thinking defaults |
| Converter recipe | tensor renames, bf16 keeps, head/embedding widths |
| Qualification | golden hashes, decode/ANE numbers, memory, Terminal-Bench |

Sketch (names are illustrative, the boundaries are the contract):

```swift
public protocol EngineVariant: Sendable {
    static var id: VariantID { get }
    static var source: SourcePin { get }          // repo + commit sha
    static var geometry: GeometryContract { get } // required, validated
    static var quant: VariantQuant { get }
    static var performance: PerformanceProfile { get }
    static var sampling: SamplingProfile { get }
    static var recipe: ModelRecipe { get }        // install/convert only

    init(context: MetalContext, model: LoadedModel) throws
    func prefill(_ tokens: [Int], state: inout SessionState) throws
    func decode(_ state: inout SessionState) throws -> Int
    static func qualify() throws -> Qualification
}
```

`EngineVariant` has **no** methods that take another model's name, no optional
feature flags, and no runtime `family` switch. The registry resolves a variant
id to a concrete factory once, at load; an unknown id is refused rather than
defaulted.

### Isolation rule

- A variant target may import `TinyTitanCore` and Foundation/Metal only.
- A variant may **not** import another variant. A build/lint check reads every
  `import` and fails on a cross-variant edge.
- A variant's Metal constants are private to its target. There is no shared
  `kMoEXMaxD`; a variant that needs 3072 compiles 3072.
- Shared library code is opt-in. A variant may copy a kernel or a tuning file
  from another variant; the copy carries a provenance header (source variant,
  commit, date) and is qualified on its own.
- A change inside `Engines/<Variant>/` can only rebuild that target. A change
  in `TinyTitanCore` is a breaking change and requires re-qualification of every
  variant that uses the changed symbol.

## 4. Repository layout

```
Sources/
  TinyTitanCore/                 # generic only: format, tokenizer, Metal context,
                                 # HTTP, sampling primitives, server transport
  TinyTitanCLI/                  # one binary: run/serve/install/convert/bench/...
  Engines/
    Ornith122B4Bit/              # target, own graph + kernels + tuning + recipe
    Ornith122B8Bit/
    Ornith35B4Bit/
    Ornith35B8Bit/
    Qwen36_35B4Bit/
    Qwen36_35B8Bit/
    Qwen38Flash4Bit/
    Qwen38Flash8Bit/
    Qwen35Dense2B4Bit/ ...       # dense CPU/GPU variants
    K2Horizon4Bit/               # when its runner exists, otherwise no target
  Registry/                      # generated lookup: VariantID -> factory
Models/
  registry.json                  # data, not code; one entry per model+variant
Tests/
  Core/
  Engines/<Variant>/             # golden, quant, bench, memory
docs/
  plan-tinytitan-7-engine.md
```

The number of targets is intentional. A target is the compiler-enforced
isolation boundary; directory-only separation would rely on review.

## 5. One registry, generated artefacts

`Models/registry.json` is the single source for:

- model id, display name, source repo and pinned sha;
- geometry and quant description (for validation, not for execution);
- variant id and Swift target;
- installed size and source size (for the disk guard);
- tokenizer/template identity;
- qualification record: golden hash, decode tok/s, ANE prefill tok/s,
  Terminal-Bench suite and score, hardware and date.

`tinytitan catalog` renders from it:

- the installer menu and `--choose` list;
- the CLI/launcher model list and display names;
- the README benchmark table, sorted by the declared ranking key;
- the wiki/installation table (via an explicit `--write`).

`tinytitan catalog --check` fails when a registry entry names a missing target,
when a target has no registry entry, or when a committed qualification is
missing for a shipped variant. This replaces the six parallel lists and the
eight-point checklist: adding a model is **one registry entry plus one target**.

## 6. Quantization: `QuantSpec` and `QuantRole`

The five MoE-shaped slots are retired. A variant declares the roles it actually
has:

```swift
struct QuantSpec {            // affine only in 7.0
    let bits: Int             // 2, 4 or 8; 6 is refused by name
    let group: Int            // compiled into the variant's kernels
    let scale: DType          // bf16
    let bias: DType           // bf16
}

enum QuantRole: String {
    case embedding, lmHead, attentionQO, attentionKV, gdnQKV, gdnZ, gdnOut,
         router, sharedExpert, routedExpert, hyperConnection, ple, indexer,
         movaValue
}

struct VariantQuant {
    let defaultSpec: QuantSpec
    let roles: [QuantRole: QuantSpec]      // e.g. embedding/lmHead at 8-bit
    let perTensor: [TensorStem: QuantSpec] // only where the variant needs it
}
```

Rules:

- The variant validates every tensor it reads against its own `VariantQuant`.
  There is no global role-uniformity rule; a variant that deliberately mixes
  widths inside a role declares it and tests it.
- The group size is a function constant of that variant's kernels. A group-32
  variant compiles group-32 kernels; it does not reinterpret a group-64 buffer.
- The concrete int4/int8/bf16 linear kernels are **internal to the variant**.
  A variant cannot accidentally read 8-bit bytes as nibbles, because it cannot
  obtain a kernel it did not compile for its own spec. The 6.x `SlotGEMV`
  incident is the reason this is a hard boundary, not a style preference.
- Promoted bf16 remains a tensor-dtype property, read from the tensor, never
  inferred from the role.
- 6-bit stays withdrawn unless a variant is explicitly built and qualified for
  it; the error names the reason.

## 7. Performance and sampling profiles

`ModelProfile` is split:

```swift
struct PerformanceProfile {       // constants in the variant target
    let expertCacheBytes: Int
    let prefetchDepth: Int
    let prefillChunk: Int
    let routerTopKSimd: Bool
    let attentionSimdPartial: Bool
    let keepCacheWired: Bool
}

struct SamplingProfile {          // constants in the variant target
    let temperature: Float
    let topK: Int
    let topP: Float
    let presencePenalty: Float
    let thinking: ThinkingDefaults
}
```

A tuning change is a source change in one variant, reviewed against that
variant's benchmark and golden output. Environment overrides are debug-only and
compile out of release builds. There is no string-keyed `(modelID, bits)` table
that can silently fall back to a family default; a missing constant is a
compile error in the variant.

## 8. Features live with the variant, not on `Model`

The union graph from 6.x is **not** the 7.0 target. Hyper-connections, PLE,
QSA and MoVA are Qwen3.8/K2 concepts; they live in those variants.

- `Model` becomes a tensor/residency handle: it knows names, buffers, offsets
  and dtypes, not PLE or hyper-connections.
- A Qwen3.8 variant reads its HC/PLE/indexer tensors with its own types.
- A K2 variant either has its own runner and target, or it has no variant at
  all. "Install-only" is not a shipped engine path.
- If two variants want the same optional block, one copies it from the other
  with provenance; that is cheaper than a shared abstraction that couples
  their release cadence.

## 9. One binary, Swift tooling, no scripts

`tinytitan` subcommands:

```
tinytitan run      --variant <id> --prompt ... [--max-new N]
tinytitan serve    --variant <id> --port 11434
tinytitan install  <model> --bits 4|8 [--source-sha <sha>]
tinytitan convert  --source <dir|repo> --recipe <id> --bits 4|8
tinytitan verify   <install-dir>
tinytitan benchmark --variant <id> [--json out.json] [--ane]
tinytitan catalog  [--markdown] [--check]
tinytitan doctor
tinytitan self-test --variant <id>|--all
```

Everything currently in `tools/prepare_*.py`, `tools/install_models.sh`,
`tools/tinytitan_models.sh`, `tools/internal-speeds.py` and the benchmark
scripts moves into Swift targets:

| Current | 7.0 |
| --- | --- |
| `prepare_agentworld.py`, `prepare_qwen38.py`, … | `Conversion/` in Swift: `SafetensorsReader`, `AffineQuantizer`, `ExpertFuser`, `ManifestWriter`, `ModelRecipe` |
| `install_models.sh`, `tinytitan_models.sh` | `tinytitan install` + `Models/registry.json` |
| `internal-speeds.py`, `tinytitan_benchmark.py` | `tinytitan benchmark` |
| `ane_sidecars.sh`, `export_ane_prefill.py` | `tinytitan ane export` (see below) |
| per-model `KEEP_BF16`/`HEAD_BITS` in Python | each variant's `ModelRecipe` |

The downloader is Swift `URLSession` with range requests, resumable chunks,
size validation and the pinned sha. The quantizer uses Accelerate/SIMD on
bf16/f32, group from the variant. The repacker writes the same `.ssdai` format
and receipt; `tinytitan verify` re-checks it. The disk guard is one Swift
function used by install and convert.

**The one honest exception:** authoring an ANE Core ML program currently goes
through `coremltools` (Python). 7.0 has two acceptable answers:

1. a Swift MIL/`.mlpackage` writer sufficient for the attention sidecar, with
   the same compile-and-load verification; or
2. if that is not funded, ANE prefill is not a shipped feature for 7.0, and the
   GPU path is the only path. A half-Python toolchain is not acceptable.

No shell is needed at runtime. `swift build` remains the build; a checked-in
`Package.swift` and CI are the only orchestration.

## 10. Tests and qualification

Per variant, in `Tests/Engines/<Variant>/`:

- **Golden**: fixed prompts, greedy decode, logits/text hash. A change to one
  variant cannot touch another's golden.
- **Quant fidelity**: every role's declared width is what the kernel reads;
  synthetic tensors detect an 8-bit-as-nibbles read.
- **Geometry**: the variant's manifest contract, including the fields it
  requires; a near-miss is refused.
- **Memory**: resident ceiling and expert-cache budget on the target machine.
- **Benchmark**: decode tok/s, prefill tok/s, TTFT, and ANE prefill when the
  variant ships a sidecar. Results are committed as the qualification record.
- **Terminal-Bench**: the published suite/score for the model, with the harness
  version and date; `tinytitan catalog` renders it into README.

Global gates:

- `tinytitan catalog --check`: registry ↔ target ↔ qualification consistency.
- `tools/lint.sh` equivalent in Swift/CI: no cross-variant imports, no model
  names in `TinyTitanCore`, no shell/Python in the runtime path.
- Matrix test: for every variant, every role resolves and every kernel width
  has a path. This is the test that would have caught the 6.x width bug.

A variant is **shipped** only when its qualification record is present and the
README row generated from it is committed.

## 11. Migration from 6.x

Do not rewrite in place. Freeze 6.x as the reference and port one variant at a
time, with a gate at each step:

| Phase | Work | Gate |
| --- | --- | --- |
| 0 | Freeze 6.x tag; inventory every (model, quant) actually shipped | list matches registry |
| 1 | Extract `TinyTitanCore` from 6.x, no model changes | 6.x tests green, core has no model names |
| 2 | Variant template + Qwen 3.5 2B 4-bit (smallest dense) | golden parity, bench parity |
| 3 | 2B 8-bit, 4B, 9B | parity per variant |
| 4 | Ornith 35B 4/8, Qwen 3.6 4/8, KAT, AgentWorld (copy the 35B graph) | parity, then delete the 35B family switch |
| 5 | Qwen 3.8 125B 4/8 (HC/PLE/QSA inside its variants) | parity |
| 6 | Ornith 122B 4/8 (3072/48, dedicated tile per variant) | parity + new qualification |
| 7 | MTP sidecars as variants of their own (no shared target kwargs) | acceptance rate parity |
| 8 | K2 own runner or drop it from the shipped matrix | graph exists or install-only removed |
| 9 | Delete the family enum switches, `ModelProfile` table, Python tools | `rg` finds no model names in core |

Parity means: same prompt set, greedy output hash identical or a recorded,
approved difference; decode within the 6.x noise floor; memory not worse;
receipt verifies.

## 12. The ordered changes

These are the changes in the order they should land, restated under the 7.0
constraints. (The earlier draft said "keep the union graph"; the isolation rule
supersedes that. The union graph is a 6.x reference, not a 7.0 target.)

1. **One registry** (`Models/registry.json`) generating the installer menu,
   CLI list, display names, profile scaffolding and docs, with
   `tinytitan catalog --check`. It removes the six parallel lists. It is data,
   not code; the binary reads it, and no shell generates it.
2. **Manifest-first geometry for every variant.** A preset is a validation
   contract, not a runtime geometry. The 122B's "qwen36 preset unless the
   manifest differs" branch disappears; every variant validates its manifest
   against its own contract.
3. **`QuantSpec` + `QuantRole` in the variant**, not five fixed slots plus role
   properties. Group size is a function constant; concrete linear kernels are
   internal to the variant so a width cannot be bypassed.
4. **Split `ModelProfile`** into `PerformanceProfile` and `SamplingProfile`,
   held as constants in the variant target. One lookup, no silent fallback.
5. **Features move into variants.** PLE/HC/QSA/MoVA leave `Model` and the
   shared config; Qwen3.8's variant owns them. K2 gets its own runner or is
   removed from the matrix.
6. **Matrix tests per variant**, not per family: synthetic manifests for every
   variant × role × width × head/embedding override, asserting every role
   resolves and every kernel width has a path.

## 13. Trade-offs we accept

- **More code, more targets, longer clean builds.** Accepted: the isolation is
  the product. A shared-kernel regression that silently changes a second model
  is the failure mode being bought out.
- **A fix may need to be applied twice.** Accepted. A copy carries provenance
  and is qualified independently; a shared fix requires all qualifications.
- **A shared library change is expensive.** That is the point: it should be.
- **Binary size grows.** Each variant links its kernels. Release builds can
  split into per-variant executables or plugins later if size becomes a
  problem; the source boundary stays.
- **ANE export may remain non-Swift or be dropped.** Stated explicitly rather
  than hidden behind a script.

## 14. Definition of done for 7.0

- [ ] Every shipped (model, quant) is a variant target with its own graph,
      kernels, quant spec, profiles and qualification.
- [ ] `TinyTitanCore` contains no model name, family switch, or quant width
      policy.
- [ ] No cross-variant imports; enforced in CI.
- [ ] One `tinytitan` binary performs run, serve, install, convert, verify,
      benchmark, catalog and self-test; no shell or Python in that path.
- [ ] `Models/registry.json` is the only list; `catalog --check` is green.
- [ ] README/wiki tables are generated from qualification records.
- [ ] Every variant has a golden hash, benchmark record, memory ceiling and
      Terminal-Bench row, dated and hardware-tagged.
- [ ] The 6.x family switches and `ModelProfile` table are deleted.

## Appendix A — inventory, 2026-10-27

Every LLM this repository supports, has an install for, or has started a port
for, with the two widths listed separately. "Shipped" means an install, a
qualified golden and a README benchmark row on `main`; "Install-only" means the
format and loader exist but no graph can serve it; "In progress" means work on
a branch that is not yet a 7.0 variant; "Installed" means an install made from
another branch is present on this machine; "Concept" means design only.

| Model | Architecture family | Dense/MoE | Total params | Active params | 4-bit | 8-bit | Notes |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| Qwen 3.5 2B (dense) | `qwen3_5_dense` | Dense | 2B | 2B | Shipped | Shipped | CPU or GPU |
| Qwen 3.5 4B (dense) | `qwen3_5_dense` | Dense | 4B | 4B | Shipped | Shipped | CPU or GPU |
| Qwen 3.5 9B (dense) | `qwen3_5_dense` | Dense | 9B | 9B | Shipped | Shipped | CPU or GPU |
| Qwen 3.6 35B-A3B | `qwen36` | MoE top-8 | 35B | 3B | Shipped | Shipped | MTP draft 4-bit sidecar |
| Ornith 1.5 35B-A3B | `qwen36` | MoE top-8 | 35B | 3B | Shipped | Shipped | MTP draft 4-bit sidecar |
| Qwen-AgentWorld 35B-A3B | `qwen36` | MoE top-8 | 35B | 3B | Shipped | Shipped | |
| KAT-Coder-V2.5-Dev 35B-A3B | `qwen36` | MoE top-8 | 35B | 3B | Shipped | Shipped | |
| Qwen 4.0 Flash Next (Qwen3.8-Flash-Next) 125B-A6B | `qwen4_exp` / `qwen38flash` | MoE top-10 | 125B | 6B | Shipped | Shipped | Checkpoint is `Qwen/Qwen3.8-Flash-Next`; MTP draft 4-bit sidecar |
| K2-Horizon-MoVA-36B-A4B | `k2_horizon` | MoE top-8 + MoVA top-4 | 36B | 4B | Install-only | Install-only | No GPU graph; cannot serve |
| Ornith 1.5 122B | `qwen36` (3072×48) | MoE top-8 | 122B | ≈12B* | In progress | In progress | On `main` since 2026-10-27; both widths installed and measured at a 12 GB target, golden baseline still missing |
| Laguna-S-2.1 (poolside) | `laguna` (3072×48) | MoE top-10 | 118B | 8B | Installed | Not installed | 62 GB 4-bit install; code on `laguna-s-2.1` |
| Nemotron-3.5-Bolt-74B-A9B (NVIDIA) | `nemotron_h` (4096×88) | MoE top-9 | 74B | 9B | Installed | Not installed | 43 GB 4-bit install; 8-bit built then removed under disk pressure; code on `nemotron-3.5-bolt` |
| DeepSeek-V4.1-Flash-1017 | `deepseek_v41` | MoE top-6 + Engram/CSA | 763B | n/a | No (policy) | No (policy) | Concept only; precision-preserving mixed install, no re-quantization |

\* **Ornith 122B active parameters are computed, not published:** 8 routed
experts at F=1024 over 48 layers plus the non-expert body is ≈12.1B active,
≈13.7B if the embedding and lm_head are counted. The model card names no
A-number. DeepSeek's card declares 763B total but no active count; the port is
a concept and not a supported width.

**Naming:** "Qwen 4.0 Flash Next" is the name used for the `qwen4_exp`
architecture; the shipped checkpoint is `Qwen/Qwen3.8-Flash-Next` 125B-A6B,
at both 4-bit and 8-bit. No separate Qwen 4.0 checkpoint exists in this tree.

Draft heads (MTP) are not independent LLMs and are not rows above: Qwen 3.6,
Ornith 1.5 35B and Qwen3.8-Flash-Next each ship a 4-bit draft sidecar, and the
122B checkpoint carries a draft that is not yet ported.

The 7.0 target is one variant per row and width: two for every "Shipped" or
"Installed" row, two for the 122B, two for K2 when a K2 runner exists, and no
variant for the DeepSeek concept until its precision policy changes.
