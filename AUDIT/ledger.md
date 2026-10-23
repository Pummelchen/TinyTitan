# Audit ledger

Generated from `ledger.json` by `render-ledger.py`; edit the JSON, not this report.

Tasks: 39 | done: 2 | open: 36 | blocked: 1

BLOCKED is not a passing verification. The audit remains incomplete until Phase E passes.

| ID | Severity | Tier | Project | Location | Title | Status | Host |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AUD2-001 | S2 | A | TinyTitanLib | sources/TinyTitanLib/PromptCheckpointStore+Disk.swift:19-74; sources/TinyTitanLib/PromptCheckpointStore.swift:304-340 | Interior-checkpoint publication can leave uncounted staging/backup or incomplete record directories after a crash | OPEN | dev-mac (primary) |
| AUD2-002 | S2 | A | TinyTitanLib | sources/TinyTitanLib/PromptCheckpointStore.swift:199,305,378,403 | Corrupt metadata.json surfaces as .missing and is never quarantined or reclaimed | OPEN | dev-mac (primary) |
| AUD2-003 | S2 | A | TinyTitanLib | sources/TinyTitanLib/ServerModelSession+Loading.swift:346-372 | PromptCheckpointStore and ServerPromptStateStore share one storage config but keep independent disk budgets (up to 2x limit on disk) | OPEN | dev-mac (primary) |
| AUD2-004 | S2 | A | TinyTitanLib | sources/TinyTitanLib/PromptCheckpointStore+Disk.swift:94-110; sources/TinyTitanLib/ServerPromptStateStore+Disk.swift:12-37 | Checkpoint and pending turn-store readers materialize declared multi-GiB payloads before verifying hashes | OPEN | dev-mac (primary) |
| AUD2-005 | S2 | A | TinyTitanLib | sources/TinyTitanLib/CalibrationProfile.swift:215-250; sources/TinyTitanServer/Core/ServerArguments.swift:416-422 | Calibration profile application skips osVersion and (server path) modelID/family/weightBits checks | OPEN | dev-mac (primary) |
| AUD2-006 | S2 | A | TinyTitanLib | sources/TinyTitanLib/ServerPromptStateStore+Disk.swift:12-37 | readPayloadVerifying reads payload to EOF with no cap before size check (swapped/huge file read fully then rejected) | OPEN | dev-mac (primary) |
| AUD2-007 | S2 | A | TinyTitanLib | sources/TinyTitanLib/PromptCheckpointStore.swift:274,305,450 | Post-restart LRU order is directory-listing order; removeAll without prior index deletes nothing (latent: no production caller) | OPEN | dev-mac (primary) |
| AUD2-008 | S2 | A | TinyTitanCLI | sources/TinyTitanCLI/CalibrationCommand.swift:265-272 | calibrate --plan silently ignores --check/--model | OPEN | dev-mac (primary) |
| AUD2-009 | S2 | A | TinyTitan | sources/TinyTitan/Runtime/KVCache/KVCacheManager+HostStreamingPrefill.swift:410-416 | restoreStreamedSnapshot silently returns when length > hostArenaLengths[layer] (fail-open; currently unreachable behind KVCacheManager+Snapshot.swift:82 guard) | OPEN | dev-mac (primary) |
| AUD2-010 | S1 | A | TinyTitan | sources/TinyTitan/Infrastructure/Streaming/RANSDecode.swift:195-232; PreadExpertStreamer+RANS.swift:82-104 | Failed length-table pread leaves empty plans and zero payload size: expert decode silently skipped (stale weights) and payloadStride underestimated so the payload read overflows the staging MTLBuffer | OPEN | dev-mac (primary) |
| AUD2-011 | S1 | A | TinyTitanFormat/TinyTitan | sources/TinyTitanFormat/ExpertRANSIndex.swift:156-159; sources/TinyTitan/Infrastructure/Streaming/RANSDecode.swift:91-98,168 | scaleBits validated 8...16 but the decoder implements only 14 (16383/>>14/2^23 hardcoded): a sidecar with another scale passes validation and decodes to garbage weights | PROGRESS | dev-mac (primary) |
| AUD2-012 | S1 | A | TinyTitan | sources/TinyTitan/Infrastructure/Streaming/RANSDecode.swift:200-221,91-94 | Per-stream u16 lengths read from the payload are unbounded against the block run, and the kernel renorm loop has no end-of-stream bound: corrupt lengths read into neighbouring streams or past the payload buffer | TEST | dev-mac (primary) |
| AUD2-013 | S2 | A | TinyTitan | sources/TinyTitan/Infrastructure/Streaming/TwoBitExpand.swift:62-99 | expandExpertInPlace trusts layout ordering (target >= source, target+size <= slot) without validation; a violating layout silently clobbers unread source bytes | OPEN | dev-mac (primary) |
| AUD2-014 | S2 | A | TinyTitan | sources/TinyTitan/Infrastructure/Streaming/TwoBitExpand.swift:83; PreadExpertStreamer+RANS.swift:109-121 | Small numeric/latent items: expand3BitReversed drops sourceSize % 3 trailing bytes silently; 2-bit path assumes even sizes; UInt64->UInt32 narrowing at outputOffset+offsets[position] (safe only under per-slot MTLBuffer layout); payloadOffset=0 dead assignment | OPEN | dev-mac (primary) |
| AUD2-015 | S2 | B | plugins/dsh-tinytitan | plugins/dsh-tinytitan/src/keep-going.js:48; plugins/dsh-tinytitan/src/config.js (resolveAutoGoalRounds) | DEFAULT_AUTO_GOAL_ROUNDS raised 12 -> 99999 ("effectively unbounded"): the shipped default no longer bounds prompts a person did not mark as long-running; unattended overnight runs possible | AUDIT | dev-mac (primary) |
| AUD2-016 | S2 | A | TinyTitan | sources/TinyTitan/Infrastructure/Metal/GPUKeepWarmRunner.swift:69; sources/TinyTitanLib/ServerModelSession+Generation.swift:34 | MTLCommandQueue created and destroyed per request by the keep-warm runner (queue teardown documented as not deinit-safe elsewhere in the tree) | AUDIT | dev-mac (primary) |
| AUD2-017 | S2 | A | TinyTitanServer | sources/TinyTitanServer/Core/ServerArguments.swift:296; sources/TinyTitan/Runtime/Configuration/RuntimeConfiguration.swift:420 | --host-kv-window validates only > 0 while every other sizing flag names a range; a huge value passes parsing and fails only at allocation | OPEN | dev-mac (primary) |
| AUD2-018 | S1 | A | TinyTitan | sources/TinyTitan/Infrastructure/Streaming/RANSDecode.swift:229 | Audit patch double-counts stream headers and rejects valid rANS payloads | TEST | dev-mac (primary) |
| AUD2-019 | S1 | A | audit evidence | AUDIT/baseline.md:10 | Baseline package-test status was inferred from a masked pipeline and one bundle footer | PROGRESS | dev-mac (primary) |
| AUD2-020 | S1 | A | TinyTitan/TinyTitanBench | sources/TinyTitan/Infrastructure/Metal/GPUKeepWarmRunner.swift:100-127; sources/TinyTitanBench/GPUKeepWarmStream.swift:91-116 | Keep-warm worker and caller access the stop Bool without synchronization | TEST | dev-mac (primary) |
| AUD2-021 | S1 | A | TinyTitanBench | sources/TinyTitanBench/MetalHeadBenchmarks.swift:217-246 | A thrown cadence-benchmark command allocation leaves keep-warm worker running | START | dev-mac (primary) |
| AUD2-022 | S1 | A | toolchain verification | AUDIT/environment.md | Requested Swift 6.4.1 was missed by an Xcode-only toolchain inventory; Homebrew 6.4.1 builds and tests the package | DONE | dev-mac (primary) |
| AUD2-023 | S0 | A | TinyTitan | sources/TinyTitan/Infrastructure/Streaming/PreadExpertStreamer+RANS.swift:17-24; sources/TinyTitan/Runtime/Inference/RealForwardRunner+DecodeMoEPhase1.swift:133-177 | rANS publishes gate/up after the first wave while later experts are still undecoded | TEST | dev-mac (primary) |
| AUD2-024 | S0 | A | TinyTitan/TinyTitanFormat | sources/TinyTitan/Runtime/Inference/Model+RANS.swift:5-64; sources/TinyTitan/Runtime/Inference/Model+LayerStreamers.swift:136-137 | rANS sidecar output geometry is not bound to the tensor layout used by inference | TEST | dev-mac (primary) |
| AUD2-025 | S1 | A | TinyTitanFormat | sources/TinyTitanFormat/ExpertRANSIndex.swift:228-238 | Untrusted rANS payload offset addition can overflow before validation | TEST | dev-mac (primary) |
| AUD2-026 | S2 | C | benchmark | benchmark/tinytitan_m5_sweep.py:281-285 | M5 sweep discards the open temporary prompt handle and relies on garbage collection to close it | TEST | dev-mac (primary) |
| AUD2-027 | S1 | A | TinyTitanFormat | sources/TinyTitanFormat/ExpertRANSIndex.swift:197-259 | rANS sidecar permits overlapping compressed payload runs for distinct experts | TEST | dev-mac (primary) |
| AUD2-028 | S1 | A | TinyTitanLib | sources/TinyTitanLib/PromptCheckpointStore.swift:39,199,333,403 | Unvalidated checkpoint payload and auxiliary byte counts can overflow disk accounting | TEST | dev-mac (primary) |
| AUD2-029 | S2 | A | TinyTitanLib | sources/TinyTitanLib/PromptCheckpointStore.swift:259,274,363 | Checkpoint deletion and eviction discard accounting even when filesystem removal fails | START | dev-mac (primary) |
| AUD2-030 | S2 | A | TinyTitan | sources/TinyTitan/Runtime/Inference/Model+RANS.swift:14-29 | Intermediate canonical rANS guard rejects equivalent omitted and explicit-default width annotations | TEST | dev-mac (primary) |
| AUD2-031 | S1 | A | audit evidence/coverage | .github/workflows/ci.yml:1027-1046; AUDIT/baseline.md | SwiftPM coverage JSON reports only the last test bundle, not the full package | PROGRESS | dev-mac (primary) |
| AUD2-032 | S2 | A | audit evidence/scope | AUDIT/audit-brief.md; AUDIT/ledger.json:meta.prompt; audit_2310.md | The original 13-section audit brief was not in the repository; it is now preserved and reconciled | DONE | dev-mac (primary) |
| AUD2-033 | S1 | A | TinyTitan runtime/streaming | sources/TinyTitan/Infrastructure/Streaming/PreadExpertStreamer+IO.swift:259-275; sources/TinyTitan/Infrastructure/Streaming/PreadExpertStreamer+RANS.swift:11-27; sources/TinyTitan/Runtime/Inference/RealForwardRunner+DecodeMoEPhase1.swift:123-171 | LZFSE compressed plans publish the first stage after wave 0 while phase 1 is encoded for every miss | OPEN | dev-mac (primary) |
| AUD2-034 | S2 | A | TinyTitanServerCore | sources/TinyTitanServer/Core/HTTPServerHandler.swift:121-152; sources/TinyTitanServer/Core/HTTPServerHandler+Plumbing.swift:203-217 | A stalled request body is never timed out once the head is counted in flight | OPEN | dev-mac (primary) |
| AUD2-035 | S2 | A | TinyTitanCLI / TinyTitanLib | sources/TinyTitanCLI/Command/main.swift:49-52; sources/TinyTitanLib/Session.swift:52-59,94-105 | CLI Ctrl-C cancels the outer task, so an in-flight generation keeps running | OPEN | dev-mac (primary) |
| AUD2-036 | S2 | A | TinyTitan runtime/prefill | sources/TinyTitan/Runtime/Inference/RealForwardRunner+BuildCore.swift:305-372; sources/TinyTitan/Runtime/Inference/RealForwardRunner+PrefillLayer.swift:57,121; sources/TinyTitan/Runtime/Inference/RealForwardRunner+PLE.swift:90-117 | PLE prefill gather is bounded by the scratch chunk size, not by the PLEBlock rows it writes | OPEN | dev-mac (primary) |
| AUD2-037 | S1 | A | audit process | AUDIT/audit-brief.md (sections 5, 6, L5, L7); audit_2310.md | Four brief-mandated passes have no recorded evidence: facade hunt, Tier-A unused code, L5 performance and L7 ops | OPEN | dev-mac (primary) |
| AUD2-038 | S1 | A | audit process / Phase E | AUDIT/audit-brief.md (sections 1b, 11 Phase E, 12); AUDIT/environment.md | No independent host exists for the Phase E whole-repository verification | BLOCKED | dev-mac (primary) |
| AUD2-039 | S2 | B | build/toolchain | Package.swift:29; AUDIT/environment.md; AUDIT/baseline.md | Swift 6.4.1 keg adds 49 linker deployment-target warnings against a zero-warning baseline | OPEN | dev-mac (primary) |

## Evidence and dispositions

### AUD2-001

- **discovered-by:** subagent manual pass + main-agent verification
- **evidence-before:** Original in-place publication could leave payload without usable metadata. Continuation review confirms startup indexing skips malformed/payload-only directories, accepts complete .tmp-/.bak- records without canonical basename validation, and the backup/staging rename sequence has a crash gap.
- **fix-summary:** Pending patch stages writes and rolls back caught rename errors, but it does not recover crash leftovers. Disk methods moved unchanged into PromptCheckpointStore+Disk.swift to satisfy the500-line source limit; that split is not a behavioral fix.
- **evidence-after:** Existing aStaleStagingDirectoryIsNotIndexedOrServed checks invisibility and zero accounting, not reclamation. No crash-recovery proof; keep OPEN.

### AUD2-002

- **discovered-by:** subagent manual pass + main-agent verification
- **evidence-before:** Corrupt metadata lookup previously returned missing. Continuation full-file review confirms indexDiskIfNeeded still skips malformed or payload-only directories without reclaiming them; an exact-key lookup is required to trigger the new deletion path.
- **fix-summary:** Pending lazy-load patch reports corruption and attempts deletion. Startup-wide recovery remains incomplete; failed deletion is independently tracked as AUD2-029.
- **evidence-after:** Focused checkpoint tests exercise direct lookup quarantine, not reclamation of all stale entries. Keep OPEN.

### AUD2-003

- **discovered-by:** subagent manual pass + main-agent verification
- **evidence-before:** Both stores originally received the full disk budget. Continuation review of ServerModelSession+Loading.swift:364-391 confirms the pending half-budget split only accounts for concurrently enabled stores. Disabling checkpoints leaves their persisted files outside the active turn-store budget, allowing old half-budget checkpoints plus a full-budget turn store.
- **fix-summary:** Per-store arithmetic is patched for enabled checkpoints, but persisted enable/disable transitions remain unresolved.
- **evidence-after:** ServerModelSessionStorageBudgetTests verifies arithmetic, not occupancy across restart/disable. Transition reproduction not yet run; keep OPEN.

### AUD2-004

- **discovered-by:** subagent manual pass + main-agent verification
- **evidence-before:** Checkpoint readPayload calls read(upToCount: expectedBytes+1), potentially allocating the4GiB capture cap, then load hashes the Data. Pending ServerPromptStateStore+Disk.swift now also materializes expected+1 before hashing. Its comment incorrectly claims refusal before allocation. A read cap is not preverification allocation control.
- **fix-summary:** No fix to hash-before-large-allocation. The checkpoint source split preserved this behavior; AUD2-028 bounds numeric metadata but does not resolve this resource concern.
- **evidence-after:** Confirmed by full-file review; no large-allocation runtime probe in this continuation. Keep OPEN and include turn-store sibling.

### AUD2-005

- **discovered-by:** subagent manual pass + main-agent verification

### AUD2-006

- **discovered-by:** subagent manual pass + main-agent verification

### AUD2-007

- **discovered-by:** subagent manual pass + main-agent verification
- **evidence-before:** Directory listing order previously determined restart eviction. Pending patch sorts by filesystem modification dates, but touchDisk only updates the in-memory date (PromptCheckpointStore.swift); no persistent touch occurs. Thus load A after saving A/B, restart, save C still treats A as oldest.
- **fix-summary:** Pending patch indexes before removeAll/save and sorts restart records. Persisting access order remains unfixed.
- **evidence-after:** Existing restart test manually backdates files and passes, but does not exercise save/load/restart access persistence. Static residual confirmed; dedicated restart-access reproduction remains pending. Keep OPEN.

### AUD2-008

- **discovered-by:** subagent manual pass + main-agent verification

### AUD2-009

- **discovered-by:** subagent manual pass + main-agent verification

### AUD2-010

- **discovered-by:** RANS/C-kernel subagent pass + main-agent verification

### AUD2-011

- **discovered-by:** RANS/C-kernel subagent pass + main-agent verification
- **evidence-before:** New test tests/TinyTitanRepack/Core/Format/ExpertRANSIndexDecodeTests.swift failed pre-fix: scaleBits 13 and 15 both accepted ("an error was expected but none was thrown", 2 issues).
- **fix-summary:** ExpertRANSIndex.decode now requires scaleBits == ExpertRANSIndex.implementedScaleBits (14), the scale the GPU kernel and the reference decoder implement; RANSExpertLayerWriter.scaleBits is now defined as the same constant so reader and writer cannot drift.
- **evidence-after:** Post-fix run: 4 tests in 2 suites passed (new decode suite + the existing writer round-trip, which proves the constant alignment did not change the emitted bytes). Final focused rANS run passed19tests/3suites, and the post-format full serial suite exited0 with1974passed/11skipped (AUDIT/continuation-full-swift-tests-postformat.log). Full-scope review, clean commit and PhaseE remain pending.

### AUD2-012

- **discovered-by:** RANS/C-kernel subagent pass + main-agent verification
- **verification:** AUDIT/rans-block-red.log: after removing double-counted header, short-header and cross-block refusal assertions fail. AUDIT/rans-kernel-red.log: writer/GPU round-trip passes, corrupted first stream returns success and publishes (2 failed assertions). With per-block length>=8 guard and per-stream GPU payloadEnd+atomic failure checked before publish, AUDIT/rans-kernel-green.log: 11 tests in 3 suites pass, exit0. Recovery with original bytes also succeeds. Post-format full serial suite exited0 with1974passed/11skipped (AUDIT/continuation-full-swift-tests-postformat.log). Full-scope cold review, clean commit and PhaseE remain pending.

### AUD2-013

- **discovered-by:** RANS/C-kernel subagent pass + main-agent verification

### AUD2-014

- **discovered-by:** RANS/C-kernel subagent pass + main-agent verification

### AUD2-015

- **discovered-by:** delta-surface subagent + main-agent spot-check
- **disposition:** Claim disproven: autoGoal requires explicit true and ships false; 01bd8ad deliberately changed the opted-in cap and docs. Agent ran node --test plugins/dsh-tinytitan/test/keep-going.test.js plugins/dsh-tinytitan/test/index.test.js: 25 passed. Upstream cap termination is not live-verified. No unauthorized runtime-default change.

### AUD2-016

- **discovered-by:** delta-surface subagent + main-agent spot-check
- **disposition:** Claim disproven: every committed command buffer is joined before queue release; warning cited from MetalContext was a comment-only empty destructor. Separate confirmed race and benchmark cleanup defects are tracked as AUD2-020/021, not substituted for this claim.

### AUD2-017

- **discovered-by:** delta-surface subagent + main-agent spot-check

### AUD2-018

- **discovered-by:** main-agent cold review against actual writer
- **evidence-before:** RANSExpertLayerWriter.encodeBlock writes UInt16(data.count), including the eight-byte header; current guard checks streamOffset + 8 + length. Expected: accept exact block tiling, require each length >= 8, reject a stream crossing its own block even if within the expert run.
- **verification:** AUDIT/rans-header-red.log exit1: exact-fit 36-byte block wrongly rejected as stream ending44. AUDIT/rans-block-red.log: same positive test passes after eliminating extra8. AUDIT/rans-kernel-green.log exit0: 11tests3suites including actual writer-to-GPU round-trip. Earlier two-stream fixture corrected from64 to4096 symbols and payload helper now writes declared stream lengths; original misleading refusal pass withdrawn.

### AUD2-019

- **discovered-by:** main-agent evidence review
- **evidence-before:** Original run used swift test piped through tail; pipeline exit 0 does not prove swift exit 0. The 98-test footer is one bundle, not the full package. Baseline pass and zero-skip assertions withdrawn. Full Tier-A scope evidence must also be checked: a delta-only review does not satisfy the requested full audit.
- **fix-summary:** Replaced the masked assertion with complete serial logs and unmasked command statuses, reconciled all bundle events and explicit skips, retained the earlier signal5 failure and the formatting-during-run caveat, then reran after formatting.
- **evidence-after:** AUDIT/continuation-full-swift-tests-postformat.log exit0:26.28s build; seven bundle counts949/422/168/189/96/63/98 include11skips, leaving1974passed. All5318lines independently reviewed; main checked footers and skip inventory. No source changed during this run. All18lint gates passed. Coverage export scope gap is AUD2-031. Full current Tier-A L2/L3/L4 passes, exact6.4.1, model checks and clean-commit PhaseE remain incomplete; keep open.

### AUD2-020

- **discovered-by:** independent review verified by main-agent full-file read
- **evidence-before:** Ordinary Bool written by start/stop and read on a dispatch worker. Existing unchecked invariant claims stale reads are harmless, which does not make the Swift data race defined. keepwarm-tsan-red.log:153-235 records the actual race after an earlier invalid compilation attempt; only the running reproduction is RED evidence.
- **fix-summary:** Pending patch uses Atomic<Bool> release stores/acquire loads; semaphore signal/wait still joins the worker and synchronizes the dispatch count. Command buffers finish before the signal. The cadence structure is unchanged; no timing claim.
- **evidence-after:** Independent full-file narrow review found no new high-confidence defect for the supported one-start/one-stop lifecycle. Historical keepwarm-tsan-green.log:75-85 reports3tests exit0; it is not fresh final-tree sanitizer certification. Runner tests cover128fresh immediate joins and enabled/disabled modes, not the benchmark stream or forced in-flight dispatch shutdown. AUD2-021 remains unfixed. No concurrent start/stop guarantee inferred.

### AUD2-021

- **discovered-by:** independent review verified by main-agent full-file read
- **evidence-before:** start precedes throwing requireCommandBuffer/encoder allocation, but stop is only on the success path. The worker retains itself while running. Expected: exactly one joined stop on success and error without changing measured timing.

### AUD2-022

- **discovered-by:** main-agent version command plus independent toolchain inventory
- **evidence-before:** Earlier inventory: xcrun swift --version = Swift 6.4 (swiftlang-6.4.0.34.1), driver 1.168.6; xcodebuild -version = Xcode 27.1 (27A9275). XcodeDefault and CommandLineTools both reported 6.4.0 and the search concluded no 6.4.1 compiler existed. That search omitted Homebrew's keg-only formulae, so the conclusion was wrong.
- **fix-summary:** Located the Homebrew keg-only `swift` formula at /opt/homebrew/opt/swift (Cellar/swift/6.4.1) and verified it reports Swift 6.4.1 (swift-6.4.1-RELEASE). Built and tested the package with it through an explicit absolute path and a scratch path under .build/brew641, so Xcode's 6.4.0 and the main build tree were untouched. The Xcode-only negative finding is corrected, not hidden.
- **evidence-after:** /opt/homebrew/opt/swift/bin/swift --version: Swift version 6.4.1 (swift-6.4.1-RELEASE), target arm64-apple-macosx27.0.0; swift package --version: Swift Package Manager - Swift 6.4.1; brew list --versions swift: 6.4.1. Build (debug): exit 0, Build complete! (72.81 sec), 1258 tasks, zero compiler errors (AUDIT/brew641-build.log). Release build: exit 0, Build complete! (125.84 sec), 972 tasks, zero compiler errors (AUDIT/brew641-release-build.log). Full serial suite: exit 0, all seven bundles passed, 1985 counted tests with the same 11 skips as the 6.4.0 run, zero compiler errors (AUDIT/brew641-full-tests.log). Language standard unchanged and enforced: swiftLanguageModes [.v6] plus -warnings-as-errors. 49 linker deployment-target warnings remain and are filed as AUD2-039; libSwiftScan falls back to the frontend scanner.
- **commit:** 250a03a090a9f9d80473351be29bd9541195ee53

### AUD2-023

- **discovered-by:** independent rANS review, main-agent full-file verification
- **evidence-before:** executeRANSPlan forwards firstStage only when start==0, but the consumer immediately dispatches all miss slots. maximumWave is eight; Qwen3.8 routes ten experts. Expected: publish only once every miss has completed phase0, and never after a failed earlier wave. Earlier RED evidence and fixture corrections remain in AUDIT/rans-waves-red-confirmed.log and its companion logs.
- **fix-summary:** Pending streamer patch gates firstStage on the final wave, after successful phase0 completion of every preceding wave.
- **evidence-after:** Independent narrow rANS review found no further publication defect. AUDIT/rans-validation-final.log exit0 includes firstStagePublishesOnlyAfterEveryWavesFrontIsReady and aLaterWaveFailureDoesNotPublishAnIncompletePlan. Synthetic GPU evidence only; no live model check. Post-format full serial suite exited0 with1974passed/11skipped (AUDIT/continuation-full-swift-tests-postformat.log); exact6.4.1/model/PhaseE checks remain pending.

### AUD2-024

- **discovered-by:** independent rANS review, main-agent full-file verification
- **evidence-before:** Original reader checked individual output bounds and role<blockCount only. Earlier patch matched blocks against each expert's own metadata, but a coordinated later-expert layout and sidecar gate/down offset swap still passed. ModelExpertIO.routedExpertOffsets uses expert0 for all slots. AUDIT/rans-cross-expert-red.log: new coordinated-swap refusal failed because no error was thrown, exit1. Earlier geometry fixture caveats remain in the original logs.
- **fix-summary:** At layer open, require matching named tensor offsets, sizes, dtypes and shapes across experts, then verify each sidecar block's offset/count/stride/role against its tensor. Optional bits annotations are excluded because schema validation already resolves their effective widths and checks byte sizes; the overly strict intermediate check is AUD2-030.
- **evidence-after:** AUDIT/rans-validation-final.log exit0:19 tests/3 suites, including coordinated-swap refusal, eight sidecar-geometry mismatch cases, writer geometry with BF16 planes, equivalent implicit-width acceptance, GPU/reference round trips. Tests are synthetic; no live model/golden run. Independent follow-up verified schema/caller ordering with no actionable concern. Post-format full serial suite exited0 with1974passed/11skipped (AUDIT/continuation-full-swift-tests-postformat.log); exact6.4.1/model/PhaseE checks remain pending.

### AUD2-025

- **discovered-by:** main-agent full-file review
- **evidence-before:** payloadOffset is decoded as UInt64 and added to UInt32 length in a trapping expression before comparison with payloadSize. UInt64.max plus a positive length crashes instead of throwing badBlock. The original refusal test remained unfixed and crashed TinyTitanRepackTests in AUDIT/continuation-full-swift-tests.log (exit1, signal5), corroborating AUDIT/rans-overflow-red.log.
- **fix-summary:** Use addingReportingOverflow once, reject overflow or file overrun, and reuse the checked endpoint for continuity and expert-run checks.
- **evidence-after:** Original refusesOverflowingPayloadRange assertion unchanged and passes in AUDIT/rans-validation-green.log and rans-validation-final.log, both exit0. Independent narrow review found no issue in checked addition. Post-format full serial suite exited0 with1974passed/11skipped (AUDIT/continuation-full-swift-tests-postformat.log); exact6.4.1/model/PhaseE checks remain pending.

### AUD2-026

- **discovered-by:** Python compatibility matrix, verified by main-agent source and raw-log review
- **evidence-before:** AUDIT/python-matrix-3.14/test_tinytitan_m5_sweep.stderr.log and the 3.15 counterpart each report 12 ResourceWarning: Implicitly cleaning up messages. NamedTemporaryFile(...).name drops the live wrapper before reopening its path; 23 assertions passed despite these warnings. This is nondeterministic cleanup, not evidence of a permanent descriptor leak. New real-handle regression failed with handle.closed false, one assertion, exit1: AUDIT/m5-handle-red.log.
- **fix-summary:** Write through the NamedTemporaryFile context manager, closing the original handle explicitly before the sweep instead of reopening the path. One new test retains the real handle to distinguish explicit close from garbage collection; no existing assertions, helpers, fixtures or imports changed.
- **evidence-after:** 24-test suite passed on Python3.13.14,3.14.9,3.15.0, each exit0 with zero ResourceWarning (AUDIT/m5-handle-green-3.13.log and 3.14/3.15 counterparts). All three invoked from repository root with -W default::ResourceWarning -m unittest -v benchmark.test_tinytitan_m5_sweep. Other five NamedTemporaryFile sites in benchmark retain/close their handle or use a context manager. Independent narrow read-only review found no further change needed; /bin/bash tools/lint.sh python exited0 (AUDIT/m5-python-gate.log). All18repository lint gates passed, exit0 (AUDIT/continuation-lint-verified.log). No model run or performance claim; final clean-commit/PhaseE certification remains pending.

### AUD2-027

- **discovered-by:** independent rANS review, main-agent writer/reader verification and regression tests
- **evidence-before:** AUDIT/rans-cross-expert-red.log: refusesOverlappingExpertPayloads at offsets0 and2 both failed because no error was thrown, exit1. The writer emits distinct contiguous expert runs, but the reader only checked contiguity within each expert.
- **fix-summary:** Sort validated expert payload ranges and reject adjacent overlaps without assuming expert ordering in the payload file. Shared payloads are not emitted by the writer.
- **evidence-after:** TMPDIR="$PWD/AUDIT/scratch" xcrun swift test --no-parallel --filter 'RANS\|rANS' exited0, AUDIT/rans-validation-green.log: 18 tests in3 suites, including both overlap cases, original overflow refusal, writer/reference/GPU round trips and multi-wave publication. Final focused19-test run also passed (rans-validation-final.log); independent review found no further range-check concern. Post-format full serial suite exited0 with1974passed/11skipped (AUDIT/continuation-full-swift-tests-postformat.log); exact6.4.1/model/PhaseE checks remain pending.

### AUD2-028

- **discovered-by:** independent checkpoint review, main-agent full-file verification
- **evidence-before:** DiskMetadata decoding did not bound payloadBytes/auxiliaryBytes before signed disk-accounting sums. After strengthening only the two new test budgets to Int.max, AUDIT/checkpoint-payload-counts-red-valid.log exited1 with all three invalid payload counts accepted by indexing (six assertions); checkpoint-auxiliary-counts-red-valid.log exited1 after accepted negative/over-cap auxiliary counts and signal5 at Int.max. Original checkpoint-byte-counts-red.log is superseded because its1MiB budgets could mask over-cap acceptance through eviction.
- **fix-summary:** DiskMetadata.validate bounds payloadBytes to1...4GiB and auxiliaryBytes-or-zero to0...1GiB; all three decoded-metadata paths invoke it before accounting or payload reads. No capture limits or valid numeric behavior changed.
- **evidence-after:** TMPDIR="$PWD/AUDIT/scratch" xcrun swift test --no-parallel --filter PromptCheckpointStoreTests exited0:28 tests/1 suite,0.057s, AUDIT/checkpoint-byte-counts-green.log. Both new methods exercise six combinations with real16-byte fixtures and metadata-only mutations. Independent narrow review found no high-confidence defect; indexed-then-mutated metadata lacks a separate regression. Post-format full serial suite exited0 with1974passed/11skipped (AUDIT/continuation-full-swift-tests-postformat.log); exact6.4.1/model/PhaseE checks remain pending; this does not resolve the other checkpoint findings.

### AUD2-029

- **discovered-by:** independent checkpoint review, main-agent full-file verification
- **evidence-before:** evictDiskIfNeeded and remove drop the index entry and subtract bytes before try? removeItem; removeAll resets all state after ignored removal errors. Failed removals leave occupied bytes invisible to subsequent budgeting. Confirmed statically; immutable-directory reproduction not yet run. Turn-store deletion-failure tests are a sibling reference, not evidence for this store.

### AUD2-030

- **discovered-by:** independent review of continuation patch, main-agent schema verification and regression test
- **evidence-before:** Full SubTensorEntry dictionary equality compares optional bits literally. Schema validation accepts omitted bits as the manifest default and checks the resulting byte size. New modelLoaderAcceptsImplicitWidthWithIdenticalGeometry failed with indexCorrupt, AUDIT/rans-implicit-width-red.log exit1. The earlier18-test rANS pass did not cover this valid representation.
- **fix-summary:** Compare named tensor offset,size,dtype,shape, not optional width spelling. Existing model-open validation ensures effective widths agree with shapes and sizes. No manifest argument or public API change needed.
- **evidence-after:** AUDIT/rans-validation-final.log:19 tests/3 suites passed, exit0, including new acceptance and unchanged coordinated geometry-swap refusal. Independent follow-up verified the schema/caller ordering and found no actionable concern. Post-format full serial suite exited0 with1974passed/11skipped (AUDIT/continuation-full-swift-tests-postformat.log); exact6.4.1/model/PhaseE checks remain pending.

### AUD2-031

- **discovered-by:** main-agent inspection of actual coverage export after successful full serial suite
- **evidence-before:** continuation-full-swift-tests-final.log completed all seven bundles with exit0, but swift test --show-codecov-path names TinyTitan.json containing only27files:19ContinuityCore sources,7tests,1generated entry point. Its4095/4256 covered lines are not package-wide coverage. default.profdata contains11984functions versus503in that JSON. Original artifact preserved in AUDIT/continuation-codecov-last-bundle.json. CI uploads this path glob and checks presence only; remote artifact contents not checked.
- **fix-summary:** Locally exported the post-format merged profile with all seven current manifest test binaries, excluding tests/build/dependency paths. No stale removed-product bundles included; no cache purge or pipeline change. Durable CI export/scope validation remains unfixed.
- **evidence-after:** llvm-cov export exited0 with empty stderr (AUDIT/swift-coverage-multi-bundle.json; exact command in baseline.md). All395unique paths are under sources:51598/63995lines=80.63%, including17TinyTitanValidation reference files. Excluding that target gives378mapped shipping-source files:50611/62974lines=80.37%.27of422Swift/C files have no map, and all16Metal files/embedded GPU code remain outside LLVM coverage. No threshold or whole-repository coverage claim. Keep open for the durable export gap.

### AUD2-032

- **discovered-by:** exhaustive read-only search of AUDIT/, docs/ and the supplied session transcript
- **evidence-before:** The brief is named only as 'Attachment 9ca50da3 (pre-production audit + remediation, 13 sections)'. The supplied transcript begins at 'continue the audit' (line 3); later hash matches are ledger copies and report references. No attachment body or path exists in the workspace. Surviving legacy labels (L0 repository, L1 architecture, L2 module, L3 line, L4 security, L5/L6 performance+tests, L6 closure sweep, L7 ops) are descriptive only; no normative checklist, Phase B/C procedure, convergence threshold or Phase E exception rule was recovered.
- **fix-summary:** Brief preserved verbatim as AUDIT/audit-brief.md behind a provenance header naming the operator attachment, recovery date, size and SHA-256. The audit report now carries a section-by-section reconciliation against the brief instead of treating the recorded scope as a substitute. No requirement was reconstructed from memory.
- **evidence-after:** AUDIT/audit-brief.md is 111 lines / 18,443 bytes: a 19-line provenance header plus the 92-line brief. The trailing 17,800 bytes hash to 0390fa3d63e92f40d8eaedff2a1c1f74c5cbd07b670a42f32123663fa2aec3e2, matching the supplied attachment. Reconciliation identifies two requirements with no recorded pass (AUD2-037) and one unmet host requirement (AUD2-038); section 12 completion is therefore still unclaimed.
- **commit:** 0f61aa5ef46d0a1a9be5497610871a1279ab9ad8

### AUD2-033

- **discovered-by:** lead from the credit-terminated native-I/O reviewer, re-derived in the current tree by the main agent
- **evidence-before:** executeCompressedMisses walks plan.misses in waves of 8 and passes firstStage only when start == 0; decodeCompressedSet invokes it after that wave's phase-0 blocks decode. encodeDecodeEarlyMissFixup awaits firstStageCompletion and then encodes phase 1 over phase1MissSlots, which is populated from every index of plan.misses (RealForwardRunner+DecodeMoE.swift:180-194). With more than 8 misses, later waves' gate/up halves are not resident when GPU phase-1 work for their slots is committed. The rANS path was fixed for exactly this class in AUD2-023 and now passes firstStage only when end == plan.misses.count; the LZFSE path retains the earlier condition.
- **fix-summary:** No fix applied. Make the LZFSE condition match the rANS final-wave rule (end == plan.misses.count) or publish only after every wave's phase 0.
- **evidence-after:** Not reproduced live: the 2026-10-23 ornith-8 golden check is batch-1 greedy decode and passed byte-identical, which does not exercise a >8-miss plan. Treat as S0 if a shipped front end reaches that plan shape, as the rANS record implies.

### AUD2-034

- **discovered-by:** lead from the credit-terminated server reviewer, re-derived in the current tree by the main agent
- **evidence-before:** channelRead resets the 120-second idle deadline on every read and increments inFlightRequests when a valid head arrives. resetIdleDeadline's scheduled task closes only when inFlightRequests == 0 and is not re-armed after it fires. A client that sends headers plus a partial body therefore leaves inFlightRequests == 1 forever: the timer fires, declines to close, and nothing else closes the connection. The up-to-1 MiB body buffer and the connection are held indefinitely. The comment above resetIdleDeadline claims a stalled slowloris request is closed, which this path cannot do. Loopback-only binding limits the blast radius, but this is local denial of service and contradicts the stated S1 control.
- **fix-summary:** No fix applied. Add a body-completion deadline (or re-arm while a body is incomplete), distinct from the no-in-flight idle close, and keep the never-close-mid-generation rule.
- **evidence-after:** Verified by code path only; no live stalled-client probe was run.

### AUD2-035

- **discovered-by:** lead from the credit-terminated library reviewer, re-derived in the current tree by the main agent
- **evidence-before:** The SIGINT handler cancels only the task that drives run(args:). Session.respond wraps generation in an unstructured Task and awaits work.value; cancellation does not propagate into an unstructured task, and awaiting Task.value in a cancelled task does not throw. The CLI never calls Session.cancel(), which is the path the server uses. A long generation therefore ignores SIGINT until it finishes on its own.
- **fix-summary:** No fix applied. Either call session.cancel() from the SIGINT handler or replace runOneShot with a cancellation-forwarding structure and keep the .cancelled mapping.
- **evidence-after:** Observed live on 2026-10-23 against models/qwen3.5_4B_4Bit: SIGINT was delivered to the running CLI mid-decode (stdout 1,008 bytes); the process stayed alive and kept writing (2,037 bytes at +10 s, 2,468 at +15 s), then produced the normal [stop=maxTokens prefill=18tok new=800tok decode=35.95s tok/s=22.255] footer and exited. Cancellation was not observed. The probe's exit status was not captured because the process was started outside the measuring shell's job table; the normal completion footer is the evidence that generation ran to its limit.

### AUD2-036

- **discovered-by:** lead from the credit-terminated inference reviewer, re-derived in the current tree by the main agent
- **evidence-before:** PLEBlock allocates embedding = embedDim * maxRows with maxRows = runtimeConfiguration.prefillChunkTokens. executePrefillChunk checks tokens.count against scratch.layout.chunkTokens, which is built from the caller's PrefillRuntimeConfig; production(chunkTokens:) accepts any value in allowedPrefillChunkTokens independently of the runner's own configuration. gatherPLERowsPrefill then writes tokens.count rows of headCount*rowBytes into that buffer. A caller passing a chunk larger than the runner's configured prefillChunkTokens passes the scratch guard and writes past the PLE embedding allocation in host-visible shared Metal memory.
- **fix-summary:** No fix applied. Bound the gather by PLEBlock.maxRows (or size PLEBlock from the effective chunk) and refuse a config whose chunkTokens exceeds the runner's.
- **evidence-after:** Static analysis only. Shipped front ends appear to pass the engine's own prefillChunkTokens, so reachability requires a caller that constructs a mismatched PrefillRuntimeConfig; RealForwardRunner is public in the engine target but is not part of the TinyTitanLib facade.

### AUD2-037

- **discovered-by:** main-agent reconciliation of the recovered original brief against the recorded passes
- **evidence-before:** The recovered brief mandates a facade hunt (section 5: always-true validators or auth checks, hardcoded success returns, NotImplemented/fatalError/empty bodies on a production path, canned data on a non-test path, sleep standing in for work, localhost/test defaults, surfaces wired to nothing; a production-path facade is S0), unused-code coverage limited to Tier A modules and audit-orphaned code (section 6), an L5 performance pass over Tier A and profiler-identified hot paths, and an L7 ops pass per deployable service (structured errors, health checks, graceful shutdown, startup config validation, rollback path). The report has no section for any of the four, and no ledger record was derived from them. Proxies that do exist are not substitutes: the test-hollow and stdout-clean lint gates cover narrow slices, and the TSan/golden runs exercise neither L5 hot paths nor L7 service behaviour.
- **fix-summary:** None. The gap is recorded rather than filled with an inferred pass. No facade, unused-code, performance or ops conclusion is asserted.

### AUD2-038

- **discovered-by:** main-agent reconciliation of the recovered brief against the recorded hosts
- **evidence-before:** Sections 1b and 11 Phase E require the single final whole-repository verification to run from a fresh clone on one independent host, and section 12 makes that run a condition of completion. AUDIT/environment.md records exactly one host: the primary development Mac. No second Mac, VPS or container host was provisioned, and no remote verification was attempted. Asking before provisioning a VPS is itself a brief requirement, so this cannot be resolved unilaterally.
- **fix-summary:** None. No host was provisioned and no substitute local run is presented as Phase E.
- **blocked-reason:** {"options": ["Provide access to an independent Apple Silicon Mac (or an approved Linux VPS for non-Swift harness checks) for the single final Phase E run", "Explicitly waive sections 1b and 12's independent-host requirement in writing, accepting a primary-host-only final verification and a correspondingly weaker completion claim"], "owner": "repository operator", "reason": "Phase E requires a second, independent host; only the primary Mac is available and provisioning one needs operator approval", "tried": "Inventoried the recorded hosts in AUDIT/environment.md; confirmed every run to date used the primary Mac; did not provision or contact any remote host"}

### AUD2-039

- **discovered-by:** main-agent build and test under the required Homebrew Swift 6.4.1
- **evidence-before:** The committed baseline records a release build with zero warnings and -warnings-as-errors on every target. Package.swift declares platforms .macOS(.v26). The Homebrew 6.4.1 keg targets arm64-apple-macosx27.0.0 and links the host's Swift runtime dylibs, which are built for 27.0.
- **fix-summary:** No fix applied. The warnings are not source diagnostics and do not fail the build; every Swift target still compiles with -warnings-as-errors. Options, none taken unilaterally: raise the package deployment target to macOS 27 (a supported-OS change needing operator approval), use an Xcode-bundled 6.4.1 toolchain whose runtime matches (none is installed), or accept the warnings in writing as a property of the verification toolchain.
- **evidence-after:** AUDIT/brew641-build.log: 49 lines of the form `building for macOS-26.0, but linking with dylib '/usr/lib/swift/libswiftCore.dylib' which was built for newer version 27.0`, plus one libSwiftScan fallback notice. AUDIT/brew641-full-tests.log: the same 49 warnings, zero compiler errors, exit 0. The release build adds 41 warnings of the same class (972 tasks, 125.84 s, exit 0; AUDIT/brew641-release-build.log). The 6.4.0 Xcode build of the same tree had zero warnings. No baseline metric is silently rewritten; the increase is recorded here.

## Verification notes

- All nine behavioral S3 findings reclassified S2 per section 8; none waived or narrowed.
- At initial evidence review AUD2-011 returned from DONE to PROGRESS because focused commit, full-suite verification and cold review were missing. Later local verification is recorded in its evidence-after field; this historical correction does not imply closure.
- Checkpoint restart-LRU fixture originally used 17 bytes while expecting two 16-byte records: corrected to 32; original mixed failure is not valid RED evidence.
- Three expansion negative fixtures were actually valid layouts; corrected earlier. Their original failed runs are withdrawn as defect evidence.
- Calibration server tests now create manifest fixtures instead of nonexistent /tmp/model paths; existing assertions retained, OS/model/family/width mismatch assertions added. Fresh 41-test run passed with unmasked exit 0 in AUDIT/calibration-focused.log.
- Prior targeted test logs under /tmp were a workspace-policy deviation; new logs use AUDIT/.
- Continuation test inventory: one new Python handle-lifetime test; one new rANS overlap method, one coordinated-geometry refusal and one implicit-width acceptance; two new checkpoint byte-count methods with six cases each. Existing assertions/fixtures were not changed in this continuation; toolchain formatting touched the two rANS test files and wrapped four tensor calls in TwoBitExpertReadTests.swift. The untracked rANS test files already existed at continuation start. Initial lint diagnostics were truncated to30 lines, so the visible four drifted files were not an exhaustive inventory.
- The two new checkpoint test budgets were strengthened from1MiB to Int.max before production changes so eviction could not mask invalid-count acceptance. checkpoint-byte-counts-red.log is superseded by checkpoint-payload-counts-red-valid.log and checkpoint-auxiliary-counts-red-valid.log; no old fixture-based conclusion is reused.
- Checkpoint source split moved writeDisk/readPayload into PromptCheckpointStore+Disk.swift with internal-only access changes, recorded in docs/repository-layout.md. Numeric validation was a separate subsequent behavior change. No model, dependency or toolchain was installed; no worktree, commit or push created.
- All continuation results describe a dirty working tree at164e84a on Swift6.4.0, not clean-commit or requested Swift6.4.1 certification. Full current Tier-A L2/L3/L4 review and model-dependent checks remain incomplete.
- Fresh post-format full serial run exited0 with1974passed and11explicit skips (1985counted tests/286counted suites); all18lint gates passed. No source/test changes occurred during that rerun. Commands, complete bundle timing footers and source-backed skip gates are recorded in baseline.md.
- Actual SwiftPM JSON covered only the final test bundle. A local all-seven-object export now measures395mapped source files with explicit exclusions; this is not whole-repository coverage and the durable CI gap stays open as AUD2-031. No pipeline, permission or instruction configuration was changed.
- 2026-10-23 continuation: five new records filed (AUD2-032..036) from reviewer leads re-derived in the current tree; the eleven delegated module reviews all terminated on credit exhaustion before emitting findings, so no reviewer claim is recorded as evidence. AUD2-031's remote artifact was inspected: it contains one bundle and 27 files, confirming the last-bundle-only defect in CI output.
- 2026-10-23 brief recovery: the operator supplied the original 13-section brief. It is preserved verbatim in AUDIT/audit-brief.md with a provenance header; the trailing 17,800 bytes hash to the supplied SHA-256. AUD2-032 is closed as documentation. Reconciliation against the brief added AUD2-037 (section 5 facade hunt, section 6 Tier-A unused code, L5 performance and L7 ops have no recorded pass) and AUD2-038 (no independent host for the Phase E whole-repository verification). The ledger is still in Phase B discovery, so no convergence milestone is claimed and section 8's strictly-decreasing closure invariant is not yet due.
- 2026-10-23 Swift 6.4.1 correction: the earlier AUD2-022 inventory searched only Xcode/CommandLineTools and missed Homebrew's keg-only swift 6.4.1. The keg builds and tests the package (exit 0, all seven bundles, same 11 skips). Its 49 linker deployment-target warnings are filed as AUD2-039 because the baseline had zero warnings. Both the error and the resulting regression are recorded rather than smoothed over.
