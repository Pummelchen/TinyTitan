# Audit coverage matrix (AUD-0007)

One row per module, one column per audit level. This is the artefact AUD-0007
asks for: what has actually been reviewed, at which level, against which
revision — and what has not. A cell is only marked reviewed when the review
exists and its evidence is named; everything else is `–` (not reviewed) or
`P` (partial, with the part that was read). `N` means not applicable, with the
reason.

The findings this pass produced are tracker rows in the wiki
(`Project-Tracker`, `AUD-0066`…`AUD-0085`); fixes cite them in their commit
subjects. Nothing in this file is a clean bill of health: the `–` cells are the
residual.

## Levels

| level | question |
|---|---|
| L0 | reproducible builds, pinned dependencies, CI security, ignored artefacts, full-history secrets |
| L1 | architecture, dependency cycles, contracts, configuration and error boundaries |
| L2 | API invariants: lifetime, cancellation, retries, queues, concurrency |
| L3 | line-by-line arithmetic, logic, encoding, timezone, return paths |
| L4 | authorisation, injection, file/URL trust, native memory, model-driven actions |
| L5 | allocation, I/O, bounded queues, latency on hot paths |
| L6 | failure and boundary behaviour, critical-path test coverage |
| L7 | health, startup validation, shutdown, rollback per service |

## Repository-wide (L0)

Reviewed 2026-11-27 at `84006bb8`:

- release build `swift build -c release` — exit 0, zero warnings, `arm64` only;
- `swift test --no-parallel` — 2,791 tests / 479 suites, zero failures;
- full TSan and full ASan over the same tree — exit 0, zero reports;
- coverage 34.81 % of instrumented lines (68,629/197,158, 1,342 files);
- `tools/lint.sh` 19/19 in a clean worktree; gitleaks over 2,920 commits clean;
  `pip-audit` on `benchmark/requirements.txt` clean;
- dependency pins: SwiftPM `exact`/`from` ranges in `Package.swift`, Python pins
  in `benchmark/requirements.txt`, the toolchain and tool versions pinned by the
  gates;
- ignored artefacts: `models/`, `.build/`, `AUDIT/`, `*.profraw` (added
  2026-11-27 after a coverage run left one in the tree).

## Modules

`R` reviewed, `P` partial, `–` not reviewed, `N` not applicable.

| module | tier | lines | L1 | L2 | L3 | L4 | L5 | L6 | L7 | evidence / residual |
|---|---|---|---|---|---|---|---|---|---|---|
| `TinyTitanKernelsC` | A | ~1.9k | R | – | R | R | P | R | N | every `.c` and header read line-by-line 2026-11-27; AUD-0078/0079/0080 fixed, AUD-0085 open; bench numbers exist for L5 but no review |
| `TinyTitanCore/Metal` (kernels) | A | — | P | – | R | P | – | R | N | attention, rmsnorm, gdn, kv-quant, dequant, gguf partial reviewed (AUD-0082 fixed); `logit`, `nemotron`, `tensorops`, `fused`, `primitive`, `rope`, `qsa`, `depth_decode`, `drafthead`, `ple` not reviewed |
| `TinyTitanCore` (Swift) | A | 10.3k | P | P | P | P | – | P | N | `JSONValue`/schema/grammar, manifest reader, format types, `Sampling`, `ChatDialect` read; AUD-0065/0077/0086 fixed, AUD-0084 open; tokenizer internals and `Generation/*` beyond the schema path not reviewed |
| `TinyTitanFormat` | A | 2.9k | R | P | R | R | – | R | N | manifest/format/index/layout/RANS/GGUF-header readers read; AUD-0074 closed by using its validator in the verifier; `SSDAIBlockEncodingV1`, `SSDAIVariantGeometryV1`, `Inkling2*` arch types skimmed |
| `Engines/Ornith35B4Bit` | A | 55.6k | P | P | P | P | – | P | N | reader/RANS/streaming, kernels cited in the native pass, plugin surface read; `RealForwardRunner+*`, prefetch ring, host-streaming prefill not reviewed |
| `Engines/DenseGPU` | A | 55.4k | P | P | P | P | – | P | N | as Ornith (19 files differ); `RoPE.swift` read (AUD-0084 open) |
| `Engines/Legacy6xRuntime` | A | 60.4k | P | P | P | P | – | P | N | the 6.x runtime, moved here by TT-134(e); audited only where the other rows reach — `Infrastructure/Streaming` (AUD-0060/0062), `Attention.swift`, `VerifiedInstallReceipt`; the rest is TT-132's porting surface |
| `Engines/Inkling2GGUFQ4` | A | 4.8k | P | P | R | R | – | R | N | plugin, install, sidecar, weights, decoder graph read; AUD-0065/0075/0076-class fixes here; MTP/`Decoder` internals partial |
| `Engines/Inkling2NVFP4` | A | 6.1k | P | P | R | R | – | R | N | as above; MTP table and weights read (AUD-0075/0084 fixed); the MTP path is kept but unused |
| `Engines/Qwen35Dense*` (6) | A | 3.5k each | P | P | P | P | – | P | N | `AffineSnapshot`, `CPUBatchedMatmul`, `CPUTensorOps` read (AUD-0083/0084 open); the six differ only in `Variant.swift` |
| `TinyTitanPluginAPI` | B | 0.6k | R | P | R | N | N | P | N | protocol, registry, probe read; no I/O, no threads of its own |
| `Plugins/TemplatePlugin` | B | 0.4k | R | P | R | N | N | P | N | golden base, read in full |
| `TinyTitanLib` | B | 9.8k | R | P | P | N | – | P | N | facade, `Session` lifecycle, request validator, plugin identity read; `LibraryContractTests` covers two engines/cancel/unload; tokenizer and prompt internals partial |
| `TinyTitanLegacyEngine` | A | 1.1k | R | P | R | N | – | P | N | the adapter boundary, read in full; TT-132 removes it |
| `TinyTitanVariants` | B | 0.6k | R | P | R | N | N | P | N | selection table read; `LegacyRuntimeBoundaryTests` pins the 6.x import set |
| `TinyTitanCPUEngine` | B | 0.5k | R | P | R | N | – | P | N | facade over the dense CPU backend, read |
| `TinyTitanServerCore` | A | 10.7k | R | R | R | R | P | R | R | all 41 files read line-by-line 2026-11-27; AUD-0066/0067/0068/0069/0071 fixed, 0070 did not reproduce and is pinned by a test, 0087 open; loopback policy, body/header bounds, response store, coordinator, model router verified clean |
| `TinyTitanRepackCore` / `TinyTitanRepack` | A | 13.1k | R | R | R | R | P | R | R | verification/remote/local/planning/format/workflow read; AUD-0072/0073/0074 fixed, 0075/0076 open; `ArchInfo+*`, `RemoteChunkPolicy`, progress display, RANS writer skimmed |
| `TinyTitanCLICore` / `TinyTitanCLI` | B | 3.2k | R | P | P | N | – | P | N | args, run, doctor, plugin-check read; calibration and off-disk routes partial |
| `ContinuityCore` | A | 4.3k | R | R | P | P | – | R | N | journal durability and append/compaction serialisation closed by the earlier audit tasks (0010–0016) and pinned by its suites; store beyond the journal partial |
| `TinyTitanMemory` | A | 5.3k | R | P | R | R | – | R | N | scope validation, discovery/listing (AUD-0061/0063/0064), store keys read; consolidation and bootstrap partial |
| `TinyTitanMemoryTool` | B | 0.2k | R | N | R | N | N | R | N | read in full; its failure paths are pinned by tests |
| `TinyTitanBench` / `TinyTitanValidationSupport` | B | 4.7k | P | N | P | N | N | P | N | `PassCostPrompt` read and pinned (TT-139); the rest is operator tooling and was not audited |
| `tools/*.py` (converter) | A | — | P | P | P | R | – | R | N | `prepare_qwen38` transport/ngram/driver read; AUD-0077 fixed; `prepare_laguna_s21`, `prepare_nemotron`, `prepare_*_mtp`, `install_models.sh`, `install_tinytitan.sh` not reviewed (`prepare_k2_horizon` was read and has since been deleted with K2's withdrawal, TT-086) |
| `tools/*.sh`, `benchmark/*.py` | B | — | P | N | P | P | N | P | N | the gates and the launcher suite are exercised on every run; not audited line-by-line |

## What this pass covered (2026-11-27)

Three read-only reviews plus the manager's own pass, at `92abaf92`–`84006bb8`:

1. **Server** — all 41 files line-by-line (L2/L3/L4/L6/L7). Trust boundary,
   body/header bounds, response-store accounting, coordinator and router
   arithmetic verified clean; five defects found (AUD-0066…0071) and four open
   investigations recorded on the parent row.
2. **Installer / format / converter** — `Verification/**`, `System/**`,
   `Remote/**`, `Local/**`, many `Planning/*`, the format readers, and the
   qwen38/k2 python. Path handling (`openat` with `O_NOFOLLOW`), resume ordering,
   fingerprint checks and credential handling verified clean; six defects found
   (AUD-0072…0077).
3. **Native / GPU** — every C file and header line-by-line, the Metal kernels
   with their Swift wrappers, the unsafe seams of four engine trees. Reader
   bounds, RANS bounds, GDN bounds, attention scratch sizing, `makeBuffer`
   sites and the `-Werror -Wconversion` build verified clean; nine findings
   (AUD-0078…0085, plus two closed by construction).
4. **Manager** — plugin registry/probe and the host boundary (AUD-0065, 0086),
   memory scope/discovery, the unused-declaration sweep (1,440 public/package
   declarations in `TinyTitanLib`/`TinyTitanCore`/`TinyTitanPluginAPI`, two
   unreachable and deleted), and the AUD-0072…0074 fixes.

## Residual, stated plainly

- The `–` cells above are unreviewed, not clean. The largest are the engine
  runtime bodies (`RealForwardRunner+*`, prefetch, host-streaming prefill), the
  Metal kernels not listed, the tokenizer internals, `TinyTitanBench`, and most
  of `tools/`.
- The 6.x runtime is the porting surface for TT-132 and is audited only where
  the rows above reach; porting replaces it rather than auditing it twice.
- Everything this pass found is fixed or closed except **AUD-0087** (three more
  unbounded maps in the memory actor, the sibling of AUD-0067) and the four open
  investigations recorded on the parent row. Each closed finding is a commit
  whose message carries the RED evidence; AUD-0070 did not reproduce and is
  pinned by a regression test instead.
- **K2 left the tree** under TT-086/TT-053 (202 files, −17,127 lines), and the
  `TinyTitanServerCore` row above no longer imports the 6.x runtime at all. The
  shared `K2MoEStage`/`K2Head` stay because Laguna's probe links them, and MTP is
  kept, tested and unused.
- AUD-0008 (independent host) still has to run after this matrix has no
  unreviewed Tier-A cell, which it does not yet.
