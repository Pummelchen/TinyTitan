# Repository layout

What lives where, why it is arranged this way, and the conventions a new file
has to follow. Written after a structural pass over the tree (2026-09-11) that
split the oversized files and flattened one inconsistent test path; every claim
here was checked against the tree, and the counts are from that pass.

## Top level

| Path | What it is | Committed |
| --- | --- | --- |
| `sources/` | The Swift package's targets, one directory per target | yes |
| `tests/` | Test targets, mirroring `sources/` path for path | yes |
| `docs/` | Engineering documentation: plans, profiles, the findings register, the runbooks | yes |
| `tools/` | Build, install, verification and conversion drivers (`*.sh`, `*.py`) | yes |
| `benchmark/` | Benchmark scripts, the golden outputs the baseline compares against, launch helpers | yes |
| `plugins/` | Client-side bundles for tools that drive the server; `plugins/dsh-tinytitan/` is the DeepSeek Harness one | yes |
| `assets/` | Brand assets (wordmark, slogans) | yes |
| `.build/`, `models/` | SwiftPM's build directory and the installed models | **no** — ignored, and never a source of truth |

`plugins/dsh-tinytitan/` is a Node package (the DeepSeek Harness bundle), the one
non-Swift, non-Python deliverable here: it ships a `cordis.patch.yml`, a plugin
entry, a compaction backend and its own `node --test` suite, and it is installed
into a DSH profile as a `file:` dependency. It is plain ESM JavaScript because
that is what the harness loads.

The two ignored directories are the only large ones (`models/` is the whole
point of the project and is hundreds of GB; `.build/` is disposable). Nothing
else in the tree is generated, so a clean checkout is the tree plus whatever
models you installed.

## Targets

`sources/` holds one directory per SwiftPM target, and the name of the
directory is the name of the target:

- **`TinyTitan`** — the runtime: the model, the forward runner, the kernels, the
  tokenizer. Subdirectories are *concerns*, not layers:
  `Kernels/` (Swift dispatch over the shaders), `Metal/` (the `.metal`
  sources), `Runtime/{Inference,Prefill,KVCache,Generation,Configuration,Family}`,
  `Infrastructure/{ModelIO,Streaming,Metal}`, `CPUEngine/`, `Tokenization/`.
- **`TinyTitanServer`, `TinyTitanCLI`, `TinyTitanRepack`, `TinyTitanMemoryTool`** — the four
  executables, each split into a SwiftPM-invisible `Command/` subdirectory
  (the `@main`/top-level entry) and a `Core/` library part that the tests
  import. `Package.swift` declares them as two targets each, with
  `exclude: ["Command"]` on the library half.
- **`TinyTitanFormat`, `TinyTitanMemory`, `ContinuityCore`** — the `.gturbo` format
  types, the memory layer, and the session/continuity engine. Each is a
  standalone library with its own README where its contract needs prose.
- **`TinyTitanBench`, `TinyTitanValidation`, `TinyTitanKernelsC`** — the benchmark
  harness, the validation/reference target, and the C kernels.

Two naming conventions follow from this and are worth stating, because both
were violated by exactly one file each and both violations were fixed in the
2026-09-11 pass:

1. **`main.swift` means top-level code.** A file with top-level statements is
   named `main.swift` (`TinyTitanCLI/Command`, `TinyTitanServer/Command`,
   `TinyTitanRepack/Command`, `TinyTitanMemoryTool`). A file whose entry point is
   `@main` is named after its type (`TinyTitanBench/TinyTitanBench.swift`,
   `ContinuityDemo/ContinuityDemo.swift`). `@main` in a `main.swift` happens to
   compile while the target is a single file and stops compiling the moment a
   second file joins the target — which is how `TinyTitanBench` was caught.
2. **Feature files are named for the type or the axis they extend**:
   `Model.swift` + `Model+Loading.swift`, `HTTPServerHandler.swift` +
   `HTTPServerHandler+{Routes,Chat,Responses,Anthropic,Plumbing}.swift`. A
   `+` name means "another file's type, one concern".

## Tests mirror sources

`tests/<Target>/...` mirrors the target's own directory shape, so a test is
found the same way the code is: `sources/TinyTitan/Runtime/Prefill/X.swift` is
tested by `tests/TinyTitan/Runtime/Prefill/...`. Where a target has a `Core/`
library half, the test path repeats it (`sources/TinyTitanServer/Core` ↔
`tests/TinyTitanServer/Core`).

One path was inconsistent and is now fixed: the `TinyTitan` runtime's test target
sat at `tests/TinyTitan/Core` while the runtime itself has no `Core/` level. It is
`tests/TinyTitan` now, with the target renamed `TinyTitanTestsCore` → `TinyTitanTests`.

`tests/TinyTitan/Runtime/qwen38_tensor_names.txt` and
`tests/TinyTitanRepack/Core/Support/qwen38_tensor_names.txt` are byte-identical
200 KB fixtures. **That duplication is required, not an oversight:** SwiftPM
resources belong to one target, the two files are resources of two different
test targets, and neither target may read the other's `Bundle.module`.
Removing one means one suite silently loses its fixture, so leave them.

## File size

There is no line-count ceiling on a *file*; the gate is on functions
(`tools/lint.sh`, 120 lines, with an inline `lint:allow-long <reason>`
exemption for orchestrators that are genuinely one sequence). File size is a
readability question, and the convention that came out of this pass is:

- **A file should hold one type, or one type plus the value types it speaks
  in.** When a file held a type and its supporting value types, those moved
  out (`ExpertCacheTypes.swift` out of `PreadExpertStreamer.swift`;
  `HTTPServerSupport.swift` out of `HTTPServer.swift`).
- **A file should hold one API surface or one phase.** `HTTPServerHandler` had
  grown to 2,111 lines covering routing, three API surfaces and the response
  plumbing; it is now six files, the largest 604 lines. The forward runner was
  already split by phase (`+Decode`, `+Prefill`, `+Residual`, `+MTP`), which is
  why those files stay large: each *is* one phase, and splitting a pipeline
  mid-sequence trades one long read for several functions with unwieldy
  signatures — the same argument its `lint:allow-long` comments already make.
- **Extracting a method to another file widens its access.** `private` is
  file-scoped in Swift, so a member reached from a new file of the same module
  becomes `internal`. That is the price of the split and the reason it is done
  only where the read improves; 95 members of `ServerHTTPHandler` and 4 members
  around `Model` were widened this pass, and nothing else changed.

The file-size rule is 500 physical lines per source file, comments and blanks
included. `find sources -name '*.swift' | xargs wc -l` listed 9 production
files above it on 2026-09-28; the largest are
`RealForwardRunner.swift` (1,449), `PreadExpertStreamer.swift` (1,392, one
class), `MemoryService.swift` (1,128), `MemoryBackend.swift` (643),
`SessionLog.swift` (642) and `RealForwardRunner+Decode.swift` (629).
Each is one
cohesive type or one phase of a pipeline; the next structural gain there is a
*design* change (a type doing two jobs), not a move, and none is currently doing
two jobs. Where a file is a phase plus a cluster of helpers around it, or a
group of independent value types or accessors, the cluster moves out on its own
(`+DecodeAttention.swift`, `+DecodeMoE.swift`, `+PrefillLayer.swift`,
`+PrefillMoE.swift`, `OpenAIWireTypes.swift`, `ResponsesAPIMapper.swift`,
`Model+Validation.swift`, `Model+Accessors.swift`).

Three files were split out of that list on 2026-09-15, each as pure code motion
after the seam was checked to be self-contained — no `private` member reached
across the new boundary, so no access was widened:

| New file | Lines | Out of |
| --- | ---: | --- |
| `TinyTitanServer/Core/StructuredOutputDiagnostics.swift` | 245 | `ServerInference.swift` (1,897 → 1,664) |
| `Runtime/Inference/RealForwardRunner+DecodeAttention.swift` | 268 | `+Decode.swift` (1,829 → 1,572) |
| `Runtime/Inference/RealForwardRunner+PrefillAttention.swift` | 362 | `+Prefill.swift` (1,875 → 1,524) |

Seven more files came out of `ServerInference.swift` on 2026-09-27, when it
carried 1,821 lines and held the public API, the coordinator, the per-generation
decode state and a 1,464-line `ServerModelSession` actor. Unlike the pass above
this one *did* widen access: `private` is file-scoped in Swift, so every member
reached from another file became `internal`. The split is pure code motion
otherwise — every original line is present verbatim in one of the new files:

| New file | Lines | Out of |
| --- | ---: | --- |
| `TinyTitanServer/Core/ServerInference.swift` (API, events, protocols) | 182 | itself (1,821) |
| `TinyTitanServer/Core/ServerModelSession.swift` | 186 | same |
| `TinyTitanServer/Core/ServerCoordinator.swift` | 164 | same |
| `TinyTitanServer/Core/ServerModelSession+Loading.swift` | 343 | same |
| `TinyTitanServer/Core/ServerModelSession+PromptCache.swift` | 269 | same |
| `TinyTitanServer/Core/ServerModelSession+Generation.swift` | 320 | same |
| `TinyTitanServer/Core/ServerModelSession+Diagnostics.swift` | 425 | same |

The same move took the stage code out of the decode phase file on 2026-09-28,
in two steps, leaving `+Decode.swift` at 629 lines: the token entry points and
the layer loop that calls the stages.

| New file | Lines | Out of |
| --- | ---: | --- |
| `Runtime/Inference/RealForwardRunner+DecodeMoE.swift` | 621 | `+Decode.swift` (1,609 → 1,004) |
| `Runtime/Inference/RealForwardRunner+DecodeGEMV.swift` | 201 | `+Decode.swift` (1,004 → 629) |
| `Runtime/Inference/RealForwardRunner+DecodeAttention.swift` | 299 → 491 | `+Decode.swift`, the `encodeDecodeAttention` dispatch |

No access widened — the moved declarations were already internal. The same pass
found `encodeLinearAttentionDecode`'s doc comment stranded in `+Decode.swift`
since the 2026-09-15 split moved its body; it now sits above the function it
describes.

The prefill phase file followed on the same day, `+Prefill.swift` 1,541 → 239,
which took it under the rule in one pass:

| New file | Lines | Out of |
| --- | ---: | --- |
| `Runtime/Inference/RealForwardRunner+PrefillLayer.swift` | 379 | the chunk orchestrator and the per-layer pass it dispatches |
| `Runtime/Inference/RealForwardRunner+PrefillMoE.swift` | 483 | the dense FFN and the routed-MoE tile stage |
| `Runtime/Inference/RealForwardRunner+PrefillProjection.swift` | 306 | the per-layer views, the affine GEMV wrapper and the final head |
| `Runtime/Inference/RealForwardRunner+PrefillKV.swift` | 177 | the KV cache writes and the quantized staging path |

No widening here either: `runPrefillLayer` is `private` and moved together with
its only caller, so it stayed private.

The OpenAI-compatible server models split the same way, by declaration group
rather than by phase, `OpenAIModels.swift` 1,046 → 235:

| New file | Lines | Out of |
| --- | ---: | --- |
| `TinyTitanServer/Core/OpenAIRequestValidator.swift` | 471 | the validator enum |
| `TinyTitanServer/Core/OpenAIRequestValidation.swift` | 194 | `ServerRequestError`, `ValidatedChatRequest` |
| `TinyTitanServer/Core/OpenAIWireTypes.swift` | 170 | the wire types, message content through template kwargs |

The two `private` members involved — `ValidatedChatRequest.copy` and the
validator's static helpers — moved with their own declarations, so nothing was
widened.

`ResponsesAPIModels.swift` (1,004 → 219) went the same way, keeping the request
types and sending the other three groups elsewhere:

| New file | Lines | Out of |
| --- | ---: | --- |
| `TinyTitanServer/Core/ResponsesAPIMapper.swift` | 354 | the request-to-chat mapper |
| `TinyTitanServer/Core/ResponsesAPIBuilder.swift` | 296 | the response builder and the `JSONValue` bridging |
| `TinyTitanServer/Core/ResponsesAPIStore.swift` | 155 | `ResponseStore` and `ResponsesAPIEcho` |

`AnthropicModels.swift` (721 → 134) followed the same shape, keeping the three
request types:

| New file | Lines | Out of |
| --- | ---: | --- |
| `TinyTitanServer/Core/AnthropicMapper.swift` | 469 | the Messages request mapping |
| `TinyTitanServer/Core/AnthropicBuilder.swift` | 131 | the response builders |

`Model+Loading.swift` (861 → 239) splits one extension by validation stage: the
load pipeline keeps the file, the schema checks move out.

| New file | Lines | Out of |
| --- | ---: | --- |
| `Runtime/Inference/Model+SchemaValidation.swift` | 328 | role uniformity, family quant support, layer tensors, routed-expert layout |
| `Runtime/Inference/Model+Validation.swift` | 302 | receipt layer layout, tile bounds, executable geometry, runtime schema, dense and layer schema |

Three members widened from `private` to internal because their callers stayed
behind or moved apart: `validateTrustedReceiptLayerLayout` (called by `load`),
`validateLayerTensors` and `validateRoutedExpertLayout` (called by
`validateLayerSchema`).

`Model.swift` (1,081 → 377) is the first split that cuts members *out of a type
declaration* rather than out of an extension, so the moved members became
extensions: stored properties cannot leave the declaration file, methods and
computed properties can.

| New file | Lines | Out of |
| --- | ---: | --- |
| `Runtime/Inference/Model+Accessors.swift` | 324 | the family schema accessor, every named tensor accessor and `bf16Readable` |
| `Runtime/Inference/Model+LayerStreamers.swift` | 266 | lazy routed-expert and streamer management |
| `Runtime/Inference/RuntimeSchemaChecks.swift` | 149 | the `RuntimeSchemaChecks` type |

No widening: the only two `private` members involved (`bf16Readable` and
`openLayerLocked`) moved together with their only callers.

`Tokenizer.swift` (818 → 491) followed the same shape — the struct keeps its
stored properties, the initializers and the encode/decode surface:

| New file | Lines | Out of |
| --- | ---: | --- |
| `Tokenization/Tokenizer+Helpers.swift` | 146 | the Jinja context, `ResolvedSpecialTokens`, the streaming-decoder check and special-token resolution |
| `Tokenization/Tokenizer+Loading.swift` | 138 | the sidecar resolution, the public factories and the load coordinator |
| `Tokenization/TokenizerTypes.swift` | 73 | the error type and the two reasoning-mode enums |

This split did widen four members — `templateContext`, `ResolvedSpecialTokens`,
`validateStreamingDecoder` and `resolveChatMLTokens` — because the initializer
that calls them stays in `Tokenizer.swift`, and it widened `imStartMark` and
`imEndMark` for the moved resolver. The load-source enum and coordinator moved
with their only user, so they stayed `private`.

`RealForwardRunner+Residual.swift` (797 → 294) kept the residual seam and sent
its two unrelated stages to their own files:

| New file | Lines | Out of |
| --- | ---: | --- |
| `Runtime/Inference/RealForwardRunner+QSA.swift` | 363 | the sparse-attention indexer, its selection, the decode and prefill encoders and the snapshot dumps |
| `Runtime/Inference/RealForwardRunner+PLE.swift` | 157 | the PLE n-gram block |

No widening: `writeQSACaches` moved with both of its callers, and the PLE view
helpers with theirs.

`RealForwardRunner+MTP.swift` (788 → 326) sent its two verify stages out:

| New file | Lines | Out of |
| --- | ---: | --- |
| `Runtime/Inference/RealForwardRunner+MTPVerifyRouted.swift` | 286 | the routed-MoE stage of the width-2 verify pass |
| `Runtime/Inference/RealForwardRunner+MTPVerify.swift` | 198 | the pair schedule and the argmax packaging |

No widening: `finishVerifyPair` moved with both of its callers, and the
`lint:allow-long` marker travelled with its declaration.

`ArchInfo.swift` (732 → 197) split by family: the struct and its stored fields
stay, the readers and their cross-checks go.

| New file | Lines | Out of |
| --- | ---: | --- |
| `TinyTitanRepack/Core/Format/ArchInfo+Loaders.swift` | 458 | the four per-family `config.json` readers |
| `TinyTitanRepack/Core/Format/ArchInfo+CrossChecks.swift` | 100 | the family cross-checks |

All four loaders and both cross-checks widened from `private` to internal —
`ArchInfo.load` stays in `ArchInfo.swift` and dispatches to them, and the
cross-checks are called from the loaders, which now live apart.

`RepackPlanner.swift` (720 → 138) keeps the plan types' public entry points and
sends the value types and the planning helpers elsewhere:

| New file | Lines | Out of |
| --- | ---: | --- |
| `TinyTitanRepack/Core/Planning/RepackPlanner+Planning.swift` | 488 | name parsing, per-file planning and resident ordering |
| `TinyTitanRepack/Core/Planning/RepackPlanTypes.swift` | 113 | `Layout` and the plan value types |

`routedExpertRole`, `layerIndex` and `isMultimodalTensorName` widened from
`private` to internal because the public `plan` entry point stays behind.

`HTTPServerHandler+Plumbing.swift` (671 → 284) split along the streaming seam;
the new file carries the same imports as the original, which a first build
caught as missing:

| New file | Lines | Out of |
| --- | ---: | --- |
| `TinyTitanServer/Core/HTTPServerHandler+Streaming.swift` | 408 | the SSE head, chat-event enqueueing, stream lifecycle and frame writers |
| `TinyTitanServer/Core/HTTPServerHandler+Plumbing.swift` | 284 | the outbox drainer, the low-level writers, deadlines and frame helpers |

No widening: this file held no `private` member at all.

`QSAIndexer.swift` (712 → 487) is the first split of a state-heavy class. The
scratch helpers move; the class declaration, its stored properties and the
encode paths stay:

| New file | Lines | Out of |
| --- | ---: | --- |
| `TinyTitan/Kernels/Attention/QSAIndexer+Scratch.swift` | 237 | the per-layer key buffers, `hiddenColumns` and the four grow-on-demand helpers |

This is the widest widening so far, and it is the price of a class split: the
six moved helpers are called by the public encode paths that stay, so they
became internal, and so did the twelve stored properties they read (`ctx`,
`selectPSO`, `rms`, `rope`, `gemv`, `rawKeys`, `pooled`, `scoresBuf`,
`keepBuf`, `keepIndexBuf`, `keepCountBuf`, `queryRowsBuf`), plus `encodePool`,
which the moved `layerBuffers` calls.

`KVCacheManager.swift` (703 → 470) split three independent groups out of the
class on the same terms:

| New file | Lines | Out of |
| --- | ---: | --- |
| `TinyTitan/Runtime/KVCache/KVCacheManager+Snapshot.swift` | 112 | segment lengths, payload append and restore |
| `TinyTitan/Runtime/KVCache/KVCacheManager+Validation.swift` | 80 | range, slot and view checks, and the residency advice |
| `TinyTitan/Runtime/KVCache/KVCacheManager+Views.swift` | 72 | the K/V view and range accessors |

Widened here: seven stored properties (`kBuffers`, `vBuffers`, `strides`,
`kinds`, `capacityTokens`, `positions`, `valueBytes`), the `fp16Size`
constant, `regionBase`, and the eight validation helpers whose callers stayed
behind.

Two more followed on 2026-09-28, both without any widening:

| New file | Lines | Out of |
| --- | ---: | --- |
| `TinyTitanBench/CPUQwenCommands.swift` | 339 | the Qwen3.5 side-engine commands: continuation, perplexity, generation and batch |
| `TinyTitan/Infrastructure/ModelIO/ModelTypes+Configs.swift` | 141 | the family configuration structs |
| `TinyTitan/Infrastructure/ModelIO/ModelTypes+Error.swift` | 73 | `ModelError` |

`CPUCommands.swift` 663 → 342 (an extension with no `private` member), and
`ModelTypes.swift` 654 → 451 (declaration groups around `ArchConfig`, which
keeps the file).

`ManifestReader.swift` (626 → 465) then shed its value types: they are read by
every importer and never by the reader's own validation, so
`ManifestTypes.swift` (168) now holds the file entry, arch block, quant slots
and the decoded document, and the reader keeps the two `private` validators.

The rule for the next split is the one the pass above followed: move a *cluster*
— an entry point with its own helpers — never half of one pipeline, and check
for `private` members on both sides of the seam first, because a file-scoped
`private` reached from another file has to become `internal`.

Three more followed on 2026-09-28, all with clean seams:

| New file | Lines | Out of |
| --- | ---: | --- |
| `TinyTitanServer/Core/HTTPServerHandler+ResponsesStream.swift` | 305 | the Responses stream opener, per-event enqueueing and the terminal flush |
| `TinyTitanBench/MetalMoEBenchmarks.swift` | 289 | the routed-MoE decode benchmark and its expert-offset mirror |
| `TinyTitan/Runtime/Generation/SamplerTypes.swift` | 221 | the generation defaults, knobs and the two path enums |

`HTTPServerHandler+Responses.swift` 541 → 255 and `MetalBenchmarks.swift`
625 → 354 are both extensions with no `private` member, and `Sampler.swift`
531 → 317 keeps the class while the value types move. No widening in any of the
three; the `lint:allow-long` marker on `runMoE` travelled with its declaration.

Two more value-type groups followed:

| New file | Lines | Out of |
| --- | ---: | --- |
| `TinyTitan/Runtime/Generation/StreamingMTPTypes.swift` | 258 | the MTP memory plan, error type, statistics, checkpoint shapes and verify schedule |
| `TinyTitan/Runtime/Configuration/RuntimeConfigurationTypes.swift` | 138 | the runtime configuration enums and its error type |

`StreamingMTP.swift` 524 → 273 and `RuntimeConfiguration.swift` 547 → 415, both
keeping their class or struct, both with no widening.

The two largest class splits then followed:

| New file | Lines | Out of |
| --- | ---: | --- |
| `TinyTitanRepack/Core/Remote/RemoteStreamingRepacker+Local.swift` | 407 | the local-snapshot repack path |
| `TinyTitanRepack/Core/Remote/RemoteStreamingRepacker+Remote.swift` | 371 | the prepared remote run |
| `TinyTitanRepack/Core/Remote/RemoteStreamingRepacker+Output.swift` | 256 | completed-range recovery and the output/manifest writers |
| `TinyTitanRepack/Core/Remote/RemoteStreamingRepackTypes.swift` | 90 | the options and result types |
| `TinyTitan/Runtime/Generation/RawCompletionHelpers.swift` | 217 | the MTP streaming pass and `sampleOnce` |

`RemoteStreamingRepacker.swift` 1,373 → 288 (three stored properties and all
twenty methods widened from `private` to internal, because every cluster calls
across the new file boundaries) and `RawCompletion.swift` 591 → 383 (two
file-private functions widened for the same reason). Both class files needed
their imports re-added after the cut, which the first build caught.

`ANEPrefillAttention.swift` (919 → 494) then shed its model lifecycle and
masks into `ANEPrefillAttention+Models.swift` (437): residency, compilation,
loading, release, `preload` and the causal/selection mask builders. The
widening was again broad — the fifteen stored properties, the seven private
helpers and `LoadedModelBox` (whose `private(set)` setter on `shadowTokens`
also had to go) — because the paths that stay (`eligibleChunk`, `appendShadow`,
`finishChunk`) call into the moved code. The new file's `import TinyTitan` was
a self-import and the build rejected it, so the original's three imports are
what it carries.

Two more declaration-group moves followed:

| New file | Lines | Out of |
| --- | ---: | --- |
| `TinyTitanServer/Core/ModelRouterError.swift` | 338 | the router's failure modes and their wire descriptions |
| `TinyTitan/Kernels/Prefill/MoE/PrefillGroupedRoutedMoETypes.swift` | 194 | the streamed-tile buffer indices, parameter blocks, fetch results and lifetime tracker |

`ModelRouter.swift` 540 → 207 and `PrefillGroupedRoutedMoE.swift` 548 → 363,
both with no widening. Each new file carries the original's imports — the
router's error type needs `import TinyTitan` because it reads `rawValue` off
the runtime's enums, which the first build caught.

And two more, both single-group cuts:

| New file | Lines | Out of |
| --- | ---: | --- |
| `TinyTitanServer/Core/ServerArguments+Usage.swift` | 117 | the `--help` text |
| `TinyTitanRepack/Core/Verification/VerifiedInstallManifest.swift` | 44 | the verifier's local manifest mirror |

`ServerArguments.swift` 544 → 436 (no widening) and
`VerifiedInstallTool.swift` 527 → 491 (the five mirror structs widened, since
the loader that returns them stays; the new file also needed
`import TinyTitanFormat`).

Two more type-group cuts, both without widening:

| New file | Lines | Out of |
| --- | ---: | --- |
| `ContinuityCore/Persistence/JournalTypes.swift` | 84 | the journal record enum, the journal protocol with its no-op defaults, the null journal and the error type |
| `TinyTitanServer/Core/ServerPromptStateTypes.swift` | 45 | the prompt-state store's configuration, save result and error |

`Journal.swift` 541 → 463 and `ServerPromptStateStore.swift` 512 → 473.

One more, on the file the earlier decode split created: the routed-MoE
finalisation moves out of `RealForwardRunner+DecodeMoE.swift` (621 → 495) into
`RealForwardRunner+DecodeMoEFinalize.swift` (137) — the pending-command
hand-off, the slot publication and the shared-expert commit. No widening;
`encodeDecodeRoutedMoE` keeps the `lint:allow-long` doc that justifies its
size.

`ContinuityEngine.swift` (586 → 429) then split the same way for an actor: its
two value types went to `ContinuityEngineTypes.swift` (49) and the journal
internals — observer installation, the record/restore path — to
`ContinuityEngine+Internals.swift` (128). `journalWriteFailed` stayed behind
with `journalFailure`, whose `private(set)` setter would otherwise have had to
become public; it widened to internal instead, since three of the moved
routines call it.

Two CPU-engine cuts followed: `CPUQwen35.swift` (527 → 492) sent the quantized
row reader and the bfloat widening to `CPUQwen35+Quant.swift` (48), and
`AffineSnapshot.swift` (538 → 443) sent the shard lookup and the
`floats`/`has`/`matrix` accessors to `AffineSnapshot+Access.swift` (107). The
snapshot split widened six stored properties, the nested `Storage` enum, the
static matrix helper and `stem(of:)` — one of them, `stem`, only surfaced as a
"use of local variable before its declaration" error, because an inaccessible
method made the compiler resolve the name to the local `let stem` instead.

`Attention.swift` (553 → 441) then sent its four pipeline builders to
`Attention+Pipelines.swift` (113): the simd partial builder, the specialized
cache lookup and the partial/combine builders. That split widened all
twenty-one stored properties and the four builders, since both the kept encode
paths and the moved builders call them.

## Generated and local files

`models/`, `.build/`, `.swiftpm/`, `benchmark/mock/`,
`benchmark/benchmark-results/`, `.qwen/` (the wiki clone), `.claude/` and
`memory/` are ignored. `benchmark/__pycache__` and `tools/__pycache__` are
Python's, ignored by the same rule as any `__pycache__`. `.DS_Store` is
ignored and should not be committed anywhere; the structural pass removed the
stray ones that had accumulated outside `.build/`.

## Where to start reading

- The runtime's entry point is `Model+Loading.swift` (`Model.load`) →
  `RealForwardRunner` → the phase files.
- The server's is `HTTPServer.swift` (the actor) → `HTTPServerHandler.swift`
  (the per-connection handler) → `HTTPServerHandler+Routes.swift`.
- The format is `docs/gturbo-format.md`; the memory layer is
  `sources/ContinuityCore/README.md` and `docs/agent-memory.md`.
- The state of the tree, including what is verified and what is not, is the
  project tracker in the wiki (`Project-Tracker`).
