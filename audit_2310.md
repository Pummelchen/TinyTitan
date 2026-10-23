# Pre-production audit — 23 October 2026

**Audit incomplete. Known defects remain. No finding is formally closed.**

This is the requested dated checklist and issue report, not a second project
backlog. [AUDIT/ledger.json](AUDIT/ledger.json) remains the audit's authoritative
record; the [task-table standard](docs/task-table-standard.md) and existing project
tracker are unchanged. The issue notes below preserve the ledger's IDs, severity,
tier and status, including disproven claims and audit-introduced regressions.

## Snapshot and interpretation

- Audit: `pre-production-audit-2026-10-23`; original brief: Attachment `9ca50da3`,
  13 sections.
- Branch: `audit/2026-10-23`; audit base: `4816a84`.
- Measured HEAD: `164e84a08de5d434f95a90d552ceeff608581caa`, **with uncommitted
  source and test changes**. That commit alone does not contain the tested fixes.
- **31 records: 0 DONE, 30 unresolved/non-blocked, 1 BLOCKED.**
- Exact statuses: **13 OPEN, 2 START, 3 PROGRESS, 10 TEST, 2 AUDIT, 1 BLOCKED**.
- Recorded severities: **2 S0, 12 S1, 17 S2**. These are ledger classifications,
  not a claim that all 31 records are confirmed outstanding product bugs.
- `OPEN`, `START` and `PROGRESS` remain unfinished; `TEST` means local remediation
  evidence exists, not closure. `AUDIT` retains a claim/disposition awaiting formal
  reconciliation. `BLOCKED` is not a passing verification.
- AUD2-015 and AUD2-016 were disproven as originally stated. AUD2-019, AUD2-022
  and AUD2-031 concern verification, rather than independent runtime defects.
- No issue has a remediation commit recorded. No fix is represented here as
  merged, released, or certified on a clean final revision.

Sources are [baseline](AUDIT/baseline.md), [environment](AUDIT/environment.md),
[scope and tiers](AUDIT/scope-tiers.md), and
[tool-enforcement proofs](AUDIT/tool-coverage.md). Raw logs and generated coverage
artifacts named below are **local evidence under the gitignored `AUDIT/` tree**;
not all are tracked or available in a clean clone. Results and limitations are
therefore included inline. Several linked source/test files are also still
untracked pending audit changes; adding this report does not add those changes.
Source links identify files, not a promise that old ledger line numbers still
match after pending edits.

## Audit completion checklist

Checked items describe recorded work at this snapshot. An unchecked full-scope
item does not mean no work happened there; it means completion is not evidenced.

### Preparation and trustworthy evidence

- [x] Record the audit branch/base, environment, scope, risk tiers and trust
  boundaries.
- [x] Record deliberate-violation tool proofs for Swift concurrency, force
  unwrapping, strict C, Python lint and secret detection.
- [x] Preserve the original baseline and explicitly withdraw its masked Swift
  test-pass and zero-skip claims.
- [x] Capture a complete, unmasked, post-format serial Swift run and reconcile all
  seven bundles, including explicit skips.
- [x] Record a complete 18-gate lint run without disabling failing gates.
- [x] Detect the last-bundle-only coverage export and produce a scoped local
  seven-object export with exclusions disclosed.
- [x] Record corrected fixtures, invalid RED evidence, audit regressions and
  remaining review gaps rather than silently replacing the history.
- [ ] Reconcile this operational checklist against every requirement in the
  original 13-section brief before claiming Phase E completion.
- [ ] Provide the requested Swift 6.4.1 compiler/host, or obtain an explicit change
  to that requirement; installed Swift 6.4.0 is not a substitute.

### Full current-tree review

Every Tier A surface requires the brief's complete L2/L3/L4 manual passes on the
current tree. Earlier audit closure, delta-only review, narrow patch review,
linters and test coverage do not discharge these items.

- [ ] TinyTitanKernelsC: native memory, pointer lifetime and Swift/C contracts.
- [ ] TinyTitan runtime: loading, streaming, mmap, Metal buffers/kernels, decode,
  KV cache, sampling and concurrency.
- [ ] TinyTitanFormat: manifest/index/receipt parsing and writer-reader symmetry.
- [ ] TinyTitanLib: public facade, Engine/Session lifetime, cancellation, prompt
  state and checkpoint persistence.
- [ ] TinyTitanServerCore: loopback HTTP validation, adapters and tool calls.
- [ ] TinyTitanRepackCore and command: install writes, verification, receipts,
  resume and discard behavior.
- [ ] Converter download/resume path, including production-install writes and
  cross-language format contracts.
- [ ] ContinuityCore: persistent memory database mutations and recovery.
- [ ] TinyTitanMemory and memory tool: persistent and model-output-driven writes.
- [ ] TinyTitanFleetCore and command: LAN interactions and shared group-key
  contract.
- [ ] Release/install tooling: irreversible operations, artifact identity and
  publication boundaries.
- [ ] Finish Tier B tools/findings-driven review for CLI, benchmarks, both DSH
  plugins and remaining shell tools, including the prescribed scanner coverage.
- [ ] Finish Tier C formatter/linter/secret-scan coverage for non-converter
  benchmarks, tests, docs, examples and assets.
- [ ] Reconcile per-finding tier assignments with module scope where they differ;
  this report preserves the existing assignments rather than silently lowering
  CLI/benchmark findings from A to B.
- [ ] Complete sibling-defect review across shared contracts: install/receipt
  files, auxiliary/ANE sidecars, facade consumers, prompt stores and fleet key.
- [ ] Verify trust boundaries end to end: untrusted HTTP and disk bytes, Swift/C
  pointers, Metal/mmap, memory DB writes, downloads/LAN peers and model tool
  actions constrained by client permissions.

### Remediation and regression proof

- [ ] Resolve checkpoint crash recovery, startup reclamation, persisted shared
  budgets, allocation-before-hash, persistent LRU and failed-deletion accounting.
- [ ] Fix the still-unhandled benchmark allocation-error cleanup path.
- [ ] Complete before/after evidence and independent review for pending
  calibration, CLI, snapshot, sizing, expansion and rANS read-error patches whose
  individual ledger evidence fields remain empty.
- [ ] Retain valid RED/GREEN evidence for every behavior change, both invalid-input
  refusal and valid-input acceptance, without weakening tests.
- [ ] Close the remaining targeted test gaps: indexed-then-mutated checkpoint
  metadata, benchmark-worker/in-flight shutdown, and BF16 writer-to-GPU behavior.
- [ ] Formally dispose of the two disproven claims without substituting their
  separate confirmed sibling defects.
- [ ] Record focused remediation commits and independent cold-review outcomes;
  locally passing patches must not be marked DONE prematurely.

### Final verification and Phase E

- [ ] Run the release build and full serial test suite on the final source
  revision and required compiler; preserve exact commands, exit codes and full
  timing footers.
- [ ] Rerun relevant TSan and C ASan/UBSan suites on that final revision; older
  sanitizer results do not certify later edits.
- [ ] Reconcile Python 3.13/3.14/3.15 runtime verification with the pinned
  dependency stack; syntax checks and the focused M5 suite are insufficient.
- [ ] Inspect final-revision dependency/security/CodeQL evidence and secret scans,
  rather than assuming configured CI jobs succeeded.
- [ ] Implement and verify durable multi-bundle CI coverage export and scope
  validation; inspect the actual resulting artifact. No pipeline change is
  authorized merely by this documentation request.
- [ ] Run applicable model/golden and Engine/Session contract checks using only
  already-installed models after the required host/process preflight; explicitly
  report missing installs or unsupported hardware as not checked.
- [ ] Exercise relevant release-only and opt-in paths, or record their precise
  missing preconditions. Do not relabel skipped tests as passes.
- [ ] Rerun all 18 lint gates, documentation checks and patch hygiene after the
  final changes; retain warnings/errors instead of masking tool status.
- [ ] Reconcile every issue with its evidence, disposition and commit; resolve
  blockers or obtain explicit operator decisions, without treating them as GREEN.
- [ ] Preserve required raw evidence in a reviewable delivery form; this document
  alone does not deliver the ignored logs or untracked fixes.
- [ ] Complete Phase E and clean-commit CI/cold review before integration through
  a PR to `main`. No merge or release is claimed by this report.

## Issue record

Each numbered subsection corresponds to exactly one ledger record. All next
steps are owned **here** unless an external owner is explicitly named. These are
snapshot notes, not newly assigned tracker IDs or permission to change shared
systems. Passing local tests never bypasses the final checklist above.

### AUD2-001 — Checkpoint publication leaves crash leftovers

**S2 · Tier A · OPEN · TinyTitanLib**

- **Location:** [checkpoint disk I/O](sources/TinyTitanLib/PromptCheckpointStore+Disk.swift)
  and [checkpoint store](sources/TinyTitanLib/PromptCheckpointStore.swift).
- **Finding:** Crashes can leave incomplete payload/metadata directories or
  uncounted staging/backup directories. Startup indexing skips malformed entries
  and can accept complete `.tmp-`/`.bak-` records without canonical-name checks.
- **Current evidence:** Pending staging/rename rollback handles caught errors,
  not crash recovery. The stale-staging test proves invisibility and zero
  accounting, not reclamation. Splitting disk methods was code motion only.
- **Next:** Reproduce interrupted publication/restart states and implement tested
  recovery, reclamation and canonical directory validation.

### AUD2-002 — Corrupt checkpoint metadata is not reclaimed at startup

**S2 · Tier A · OPEN · TinyTitanLib**

- **Location:** [checkpoint store](sources/TinyTitanLib/PromptCheckpointStore.swift).
- **Finding:** Corrupt metadata originally surfaced as missing and stayed on
  disk. Pending exact-key lookup reports corruption and attempts deletion, but
  startup indexing still skips malformed or payload-only directories.
- **Current evidence:** Direct-lookup quarantine tests do not establish global
  startup reclamation. Failed deletion is separately recorded as AUD2-029.
- **Next:** Add startup-wide corruption/reclamation tests and fix recovery without
  hiding bytes when deletion fails.

### AUD2-003 — Shared prompt storage can exceed its disk budget

**S2 · Tier A · OPEN · TinyTitanLib**

- **Location:** [session loading](sources/TinyTitanLib/ServerModelSession+Loading.swift).
- **Finding:** Both stores originally received the full shared disk allowance.
  The pending half-budget split only handles simultaneously enabled stores;
  disabling checkpoints leaves persisted checkpoint bytes outside the active
  turn-store budget.
- **Current evidence:**
  [storage-budget tests](tests/TinyTitanLib/ServerModelSessionStorageBudgetTests.swift)
  verify arithmetic, not actual occupancy across restart/disable transitions.
  The residual was reviewed statically; its transition reproduction is pending.
- **Next:** Reproduce save/restart/disable occupancy and enforce the combined
  limit over persisted as well as active stores.

### AUD2-004 — Large payloads are allocated before integrity verification

**S2 · Tier A · OPEN · TinyTitanLib**

- **Location:** [checkpoint disk I/O](sources/TinyTitanLib/PromptCheckpointStore+Disk.swift)
  and [turn-store disk I/O](sources/TinyTitanLib/ServerPromptStateStore+Disk.swift).
- **Finding:** Reading `expectedBytes + 1` into Data before hashing can materialize
  multi-GiB untrusted payloads. A read cap does not establish integrity before a
  large allocation; the turn-store comment overstates that protection.
- **Current evidence:** Confirmed by full-file review, without a large-allocation
  runtime probe. The numeric metadata guard in AUD2-028 does not fix this issue.
- **Next:** Design a bounded integrity-verification path for both readers and
  prove corrupt/oversized input is refused without the large materialization.

### AUD2-005 — Calibration applicability checks are incomplete

**S2 · Tier A · OPEN · TinyTitanLib**

- **Location:** [calibration profile](sources/TinyTitanLib/CalibrationProfile.swift)
  and [server arguments](sources/TinyTitanServer/Core/ServerArguments.swift).
- **Finding:** The reported paths omitted OS-version validation and, on the
  server, model ID, family and weight-bit compatibility.
- **Current evidence:** Pending validation patches and mismatch tests exist. The
  corrected manifest fixtures retained existing assertions; the recorded focused
  calibration run passed 41 tests, exit 0 (`calibration-focused.log`). This row's
  before/fix/after fields remain empty, so the original title is not a reliable
  description of every current guard and the patch is not formally closed.
- **Next:** Reconcile the exact checks and both entry paths with valid before/after
  evidence and independent review, then update the ledger disposition.

### AUD2-006 — Turn-store payload reader was unbounded until EOF

**S2 · Tier A · OPEN · TinyTitanLib**

- **Location:** [turn-store disk I/O](sources/TinyTitanLib/ServerPromptStateStore+Disk.swift).
- **Finding:** The original reader consumed a swapped or huge file fully before
  comparing its size with metadata.
- **Current evidence:** A pending `expected + 1` read cap addresses a different
  condition from the allocation-before-hash issue in AUD2-004. The individual
  before/fix/after evidence fields remain empty.
- **Next:** Prove oversized and swapped-file refusal on the actual reader and
  record focused evidence without conflating a size cap with preallocation
  integrity verification.

### AUD2-007 — Restart eviction does not preserve actual access order

**S2 · Tier A · OPEN · TinyTitanLib**

- **Location:** [checkpoint store](sources/TinyTitanLib/PromptCheckpointStore.swift).
- **Finding:** Restart LRU originally followed directory listing order, and
  `removeAll` without indexing could remove nothing (no production caller was
  identified for that latent case).
- **Current evidence:** Index-before-save/removeAll and sorting modification
  dates are patched, but access touches update only memory. Saving A/B, loading
  A, restarting and saving C can still evict A. The existing restart test uses
  manually backdated files, not persisted real access.
- **Next:** Add the save/load/restart reproduction and persist eviction order;
  retain the corrected 32-byte two-record fixture disclosure below.

### AUD2-008 — Calibration plan mode silently ignores conflicting options

**S2 · Tier A · OPEN · TinyTitanCLI**

- **Location:** [calibration command](sources/TinyTitanCLI/CalibrationCommand.swift).
- **Finding:** `calibrate --plan` silently ignored `--check`/`--model`.
- **Current evidence:** Pending command/test changes exist; this row's specific
  before/fix/after evidence remains blank. A full-suite pass alone does not prove
  every literal option combination has the intended behavior.
- **Next:** Verify the reported invocations, document accepted/refused combinations
  and reconcile focused regression evidence and review.

### AUD2-009 — Streamed snapshot restoration can fail silently

**S2 · Tier A · OPEN · TinyTitan**

- **Location:** [streamed restore](sources/TinyTitan/Runtime/KVCache/KVCacheManager+HostStreamingPrefill.swift)
  and [snapshot caller](sources/TinyTitan/Runtime/KVCache/KVCacheManager+Snapshot.swift).
- **Finding:** The inner restore returned silently when length exceeded the
  stored host-arena length. The outer snapshot guard makes the originally
  reported condition currently unreachable through that caller.
- **Current evidence:** A pending throwing path/caller change exists; the row's
  evidence fields remain empty. This is not a reproduced reachable inference
  failure merely because the inner behavior was wrong.
- **Next:** Verify propagation and all callers with focused tests; explicitly
  distinguish the guarded latent case from any reachable sibling.

### AUD2-010 — Failed rANS length reads can skip decoding or undersize staging

**S1 · Tier A · OPEN · TinyTitan**

- **Location:** [rANS decoder](sources/TinyTitan/Infrastructure/Streaming/RANSDecode.swift)
  and [rANS streamer](sources/TinyTitan/Infrastructure/Streaming/PreadExpertStreamer+RANS.swift).
- **Finding:** A failed length-table `pread` could leave empty plans and zero
  payload size, silently preserving stale weights and underestimating the Metal
  staging buffer needed for the subsequent payload read.
- **Current evidence:** Pending fail-closed changes and synthetic truncated-read
  tests exist. The final focused rANS run passed, but this row's individual
  before/fix/after fields have not been reconciled.
- **Next:** Trace length-table failure through buffer sizing, read and publication;
  retain focused failure evidence and independently verify refusal before use.

### AUD2-011 — rANS scale values unsupported by the decoder were accepted

**S1 · Tier A · PROGRESS · TinyTitanFormat/TinyTitan**

- **Location:** [rANS index](sources/TinyTitanFormat/ExpertRANSIndex.swift),
  [decoder](sources/TinyTitan/Infrastructure/Streaming/RANSDecode.swift) and
  [writer](sources/TinyTitanRepack/Core/Writing/RANSExpertLayerWriter.swift).
- **Finding:** Validation accepted scales 8–16 although decoding implements 14.
- **Current evidence:** Scale 13/15 refusal tests failed before the patch. Reader
  and writer now share the implemented scale-14 constant; the initial four-test
  run, final 19-test rANS run and full post-format suite passed. The original
  premature DONE state was withdrawn; it is not reinstated by these passes.
- **Next:** Complete full-scope review and focused commit/Phase E evidence while
  preserving writer-reader alignment and valid payload acceptance.

### AUD2-012 — rANS stream lengths and GPU renormalization were unbounded

**S1 · Tier A · TEST · TinyTitan**

- **Location:** [rANS decoder](sources/TinyTitan/Infrastructure/Streaming/RANSDecode.swift).
- **Finding:** Corrupt per-stream lengths could cross block/stream boundaries,
  and GPU renormalization could read beyond its own compressed stream.
- **Current evidence:** Valid RED runs cover short headers, block crossing and
  corrupt-stream success/publication. The patch requires at least the inclusive
  eight-byte header, bounds each GPU stream and checks an atomic failure signal
  before publishing. Eleven focused tests passed, including recovery after
  restoring original bytes; final focused and full runs also passed.
- **Next:** Complete full-scope cold review and required final/model verification;
  synthetic bounds tests do not certify live inference.

### AUD2-013 — In-place expansion trusts unsafe source/destination geometry

**S2 · Tier A · OPEN · TinyTitan**

- **Location:** [packed-weight expansion](sources/TinyTitan/Infrastructure/Streaming/TwoBitExpand.swift).
- **Finding:** An invalid target order or range can overwrite unread packed
  source bytes or exceed the slot during in-place expansion.
- **Current evidence:** A pending validator and tests exist. Three earlier
  negative fixtures described valid layouts; their original failed runs are
  withdrawn, not retained as defect evidence. The row's evidence fields are blank.
- **Next:** Reconcile valid refusal and acceptance cases for ordering, overlap and
  slot bounds, then review sibling expansion paths before disposition.

### AUD2-014 — Expansion group sizes and offset narrowing need separate proof

**S2 · Tier A · OPEN · TinyTitan**

- **Location:** [packed-weight expansion](sources/TinyTitan/Infrastructure/Streaming/TwoBitExpand.swift)
  and [rANS streamer](sources/TinyTitan/Infrastructure/Streaming/PreadExpertStreamer+RANS.swift).
- **Finding:** The recorded group contains silent three-bit trailing-byte loss,
  two-bit even-size assumptions, UInt64-to-UInt32 output-offset narrowing, and a
  dead zero assignment to payloadOffset.
- **Current evidence:** Some cases are latent or safe only under the per-slot
  Metal-buffer invariant. Individual evidence fields are empty; this is not four
  demonstrated crashes. Existing ID/grouping is preserved for reconciliation.
- **Next:** Prove or refute each subclaim with exact size/range invariants and
  tests, separating cleanups from behavioral defects without reusing this ID.

### AUD2-015 — Claimed runaway default is disproven

**S2 · Tier B · AUDIT · plugins/dsh-tinytitan**

- **Location:** [keep-going](plugins/dsh-tinytitan/src/keep-going.js) and
  [configuration](plugins/dsh-tinytitan/src/config.js).
- **Original claim:** Raising the round cap from 12 to 99,999 made ordinary
  prompts effectively unbounded by default.
- **Disposition/evidence:** `autoGoal` requires explicit true and ships false.
  Commit `01bd8ad` deliberately changed the opted-in cap and documentation.
  The recorded agent run of the keep-going/index suites passed 25 Node tests;
  upstream live cap termination was not verified. No default was changed here.
- **Next:** Finalize the evidence-backed disproven disposition, retaining the
  live-runtime limitation instead of substituting an unrequested default change.

### AUD2-016 — Claimed unsafe per-request Metal queue teardown is disproven

**S2 · Tier A · AUDIT · TinyTitan**

- **Location:** [keep-warm runner](sources/TinyTitan/Infrastructure/Metal/GPUKeepWarmRunner.swift)
  and [session generation](sources/TinyTitanLib/ServerModelSession+Generation.swift).
- **Original claim:** Creating/destroying a command queue per request was unsafe
  based on a warning elsewhere in the tree.
- **Disposition/evidence:** Committed command buffers are joined before queue
  release. The cited destructor was empty except for a comment, not proof of a
  teardown defect. The separate real race and cleanup defects are AUD2-020/021.
- **Next:** Finalize this claim's disproven disposition without treating either
  sibling fix as evidence for the original allegation.

### AUD2-017 — Host KV window accepts impractically large sizes

**S2 · Tier A · OPEN · TinyTitanServer**

- **Location:** [server arguments](sources/TinyTitanServer/Core/ServerArguments.swift),
  [runtime configuration](sources/TinyTitan/Runtime/Configuration/RuntimeConfiguration.swift)
  and [CLI arguments](sources/TinyTitanCLI/Args.swift).
- **Finding:** `--host-kv-window` checked only positivity, permitting huge values
  to fail much later at allocation instead of at input validation.
- **Current evidence:** Pending range checks/tests exist, but the individual
  before/fix/after evidence remains empty.
- **Next:** Verify documented boundaries and overflow handling through server,
  CLI and programmatic configuration; reconcile focused evidence and review.

### AUD2-018 — An audit patch double-counted rANS stream headers

**S1 · Tier A · TEST · TinyTitan**

- **Location:** [rANS decoder](sources/TinyTitan/Infrastructure/Streaming/RANSDecode.swift).
- **Finding:** Writer lengths already include the eight-byte header; adding it
  again rejected valid exact-fit blocks.
- **Current evidence:** `rans-header-red.log` records a 36-byte block rejected as
  ending at 44. Removing the extra eight bytes restored acceptance while keeping
  minimum-header and own-block bounds. The writer-to-GPU round trip passed in
  the 11-test GREEN run; later focused/full runs passed too. Earlier two-stream
  fixture corrections remain disclosed below.
- **Next:** Preserve positive writer-reader acceptance in final cold review and
  certification; do not resurrect the misleading original refusal result.

### AUD2-019 — Baseline test status and full-scope audit evidence were overstated

**S1 · Tier A · PROGRESS · audit evidence**

- **Location:** [baseline evidence](AUDIT/baseline.md).
- **Finding:** A pipeline hid Swift's exit status and its last 98-test footer was
  mistaken for the package total. Delta-only manual review also falls short of
  the required full current Tier A scope.
- **Current evidence:** The masked pass/zero-skip claims are withdrawn. The later
  unmasked post-format run recorded 1,974 passes and 11 skips across seven
  bundles; all 5,318 log lines were independently reviewed. All 18 lint gates
  passed. The earlier real signal-5 failure remains part of the record.
- **Next:** Complete the current-tree manual review and required compiler/model/
  clean-commit checks. Local evidence repair is not audit completion.

### AUD2-020 — Keep-warm stop flag has a data race

**S1 · Tier A · TEST · TinyTitan/TinyTitanBench**

- **Location:** [runtime runner](sources/TinyTitan/Infrastructure/Metal/GPUKeepWarmRunner.swift)
  and [benchmark worker](sources/TinyTitanBench/GPUKeepWarmStream.swift).
- **Finding:** An ordinary Bool was written by the caller and read by the worker
  without synchronization; an unchecked invariant did not make this defined.
- **Current evidence:** A running TSan reproduction reported the race; an earlier
  compilation failure is not RED evidence. The patch uses release/acquire atomic
  access and retains the semaphore join and command-buffer completion order.
  Historical TSan GREEN passed three tests; narrow review found no new defect
  for the supported one-start/one-stop lifecycle. No performance claim is made.
- **Next:** Verify the final tree under TSan and cover the benchmark worker and
  forced in-flight shutdown. Tests do not promise arbitrary concurrent start/stop.

### AUD2-021 — Benchmark allocation failure leaves keep-warm work running

**S1 · Tier A · START · TinyTitanBench**

- **Location:** [Metal head benchmarks](sources/TinyTitanBench/MetalHeadBenchmarks.swift).
- **Finding:** Worker start precedes throwing command-buffer/encoder allocations;
  joined stop exists only on the success path. The running worker retains itself.
- **Current evidence:** Independently reviewed and still **unfixed**. An atomic
  stop flag does not fix missing cleanup when the stop is never called.
- **Next:** Write a failing allocation-error cleanup test, then guarantee exactly
  one joined stop on both success and error without changing the measured timing
  boundary.

### AUD2-022 — Requested Swift 6.4.1 is unavailable

**S1 · Tier A · BLOCKED · toolchain verification**

- **Location:** [environment evidence](AUDIT/environment.md).
- **Finding/evidence:** Both XcodeDefault and CommandLineTools report
  `swiftlang-6.4.0.34.1`; no installed Swift 6.4.1 toolchain was found. Recorded
  local results establish only the installed compiler's behavior.
- **Owner:** Repository operator.
- **Next:** Provide an Apple Silicon host/toolchain reporting Swift 6.4.1, or
  explicitly revise the required patch version. No revision is assumed and no
  compiler installation is authorized by this report.

### AUD2-023 — rANS publishes before all expert waves are decoded

**S0 · Tier A · TEST · TinyTitan**

- **Location:** [rANS streamer](sources/TinyTitan/Infrastructure/Streaming/PreadExpertStreamer+RANS.swift)
  and [MoE consumer](sources/TinyTitan/Runtime/Inference/RealForwardRunner+DecodeMoEPhase1.swift).
- **Finding:** Publishing after the first wave exposes all miss slots even though
  later experts are not decoded. The wave limit is eight; Qwen3.8 routes ten.
- **Current evidence:** The patch publishes only on the final wave after earlier
  phase-zero work succeeds. Synthetic GPU tests prove complete-front publication
  and refusal after later-wave failure; narrow review found no further defect.
  Final focused/full suites passed, but no live model was checked.
- **Next:** Complete required model/end-to-end, full-scope and final-revision
  verification before closing this incorrect-results finding.

### AUD2-024 — rANS output geometry is not bound to inference's tensor layout

**S0 · Tier A · TEST · TinyTitan/TinyTitanFormat**

- **Location:** [rANS model validation](sources/TinyTitan/Runtime/Inference/Model+RANS.swift)
  and [layer opening](sources/TinyTitan/Runtime/Inference/Model+LayerStreamers.swift).
- **Finding:** Checking a sidecar against only its own expert metadata allowed a
  coordinated later-expert gate/down swap, while inference reused expert-zero
  offsets for all experts.
- **Current evidence:** The coordinated-swap refusal failed before canonical
  checks. The patch compares named offsets, sizes, dtypes and shapes across
  experts, then validates each sidecar block against its tensor. Nineteen focused
  tests passed, including eight geometry mismatches and valid implicit widths.
  BF16 geometry acceptance is not a BF16 writer-to-GPU round trip. No live model
  or golden run occurred; the final full model-free suite passed.
- **Next:** Close the missing end-to-end/BF16 and final-review evidence while
  retaining the valid-representation correction tracked as AUD2-030.

### AUD2-025 — rANS payload endpoint arithmetic can trap on overflow

**S1 · Tier A · TEST · TinyTitanFormat**

- **Location:** [rANS index](sources/TinyTitanFormat/ExpertRANSIndex.swift).
- **Finding:** Adding an untrusted UInt64 offset and positive length before
  validating the result can trap rather than throw a format error.
- **Current evidence:** The unchanged overflowing-range regression crashed the
  first continuation full suite with signal 5. Checked addition now rejects
  overflow/file overrun and reuses the endpoint for later validation. The original
  assertion passed in focused and post-format full runs; narrow review found no
  issue in the scoped arithmetic fix.
- **Next:** Preserve the crash reproduction and unchanged regression through
  final-review, focused-commit and required certification gates.

### AUD2-026 — M5 sweep relies on garbage collection to close a prompt handle

**S2 · Tier C · TEST · benchmark**

- **Location:** [M5 sweep](benchmark/tinytitan_m5_sweep.py) and
  [M5 tests](benchmark/test_tinytitan_m5_sweep.py).
- **Finding:** Taking only the temporary wrapper's filename abandoned the open
  handle. Python 3.14/3.15 emitted 12 ResourceWarnings each despite the original
  23-test suites passing; this proved implicit cleanup, not a permanent leak.
- **Current evidence:** One added test retains the real wrapper and failed on
  `handle.closed` before the fix. Writing inside its context manager gives
  explicit closure. All 24 tests passed on each of Python 3.13.14, 3.14.9 and
  3.15.0, exit 0 with no ResourceWarnings; sibling sites were reviewed.
- **Next:** Carry the unchanged regression and explicit-close patch through
  focused commit and Phase E. No model run or performance improvement is claimed.

### AUD2-027 — Different rANS experts can alias compressed payload runs

**S1 · Tier A · TEST · TinyTitanFormat**

- **Location:** [rANS index](sources/TinyTitanFormat/ExpertRANSIndex.swift).
- **Finding:** Within-expert contiguity checks did not reject exact or partial
  cross-expert overlap. The writer emits separate runs, not shared deduplication.
- **Current evidence:** Alias offset zero and partial-overlap offset two were
  both accepted in RED. Sorting validated expert runs and checking adjacent
  disjointness fixed both without assuming file order equals expert order.
  Focused refusal/round-trip/publication tests and the full suite passed.
- **Next:** Preserve the writer-reader contract and both overlap regressions in
  final review, commit and required end-to-end certification.

### AUD2-028 — Checkpoint metadata byte counts can overflow accounting

**S1 · Tier A · TEST · TinyTitanLib**

- **Location:** [checkpoint store](sources/TinyTitanLib/PromptCheckpointStore.swift)
  and [checkpoint tests](tests/TinyTitanLib/PromptCheckpointStoreTests.swift).
- **Finding:** Decoded payload/auxiliary counts reached signed accounting before
  range validation, permitting negative/over-cap values and overflow.
- **Current evidence:** Corrected RED budgets prevent eviction from hiding
  acceptance. Tests use real 16-byte payloads and metadata-only mutation, not huge
  allocations. The guard permits payloads from one byte through 4 GiB and
  auxiliary data from zero through 1 GiB at all three decoded-metadata paths,
  before accounting/read. Two new methods cover six cases each; the complete
  checkpoint suite passed 28 tests and the full Swift suite passed.
- **Next:** Add indexed-then-mutated metadata coverage and complete final
  certification. This numeric guard does not resolve other checkpoint findings.

### AUD2-029 — Failed checkpoint removal drops real bytes from accounting

**S2 · Tier A · START · TinyTitanLib**

- **Location:** [checkpoint store](sources/TinyTitanLib/PromptCheckpointStore.swift).
- **Finding:** Remove/evict drops the index entry and subtracts bytes before an
  ignored filesystem removal error; removeAll resets state despite such errors.
  Persisted occupied bytes then disappear from budget accounting.
- **Current evidence:** Confirmed statically and still unfixed. An immutable-
  directory reproduction has not run. Turn-store deletion tests are a sibling
  reference, not evidence that this store behaves correctly.
- **Next:** Reproduce a real filesystem deletion failure and retain truthful
  index/accounting state until removal succeeds or is explicitly reconciled.

### AUD2-030 — An intermediate geometry guard rejected equivalent bit widths

**S2 · Tier A · TEST · TinyTitan**

- **Location:** [rANS model validation](sources/TinyTitan/Runtime/Inference/Model+RANS.swift).
- **Finding:** Literal dictionary equality compared optional `bits` spelling;
  omitted/default width is valid after schema resolution and size checks.
- **Current evidence:** The new valid-input acceptance test failed with
  `indexCorrupt`; the preceding 18-test pass had missed it. Comparing geometry
  rather than optional spelling restored acceptance without weakening the
  coordinated-swap refusal. Final 19-test rANS and full Swift runs passed;
  independent follow-up verified schema/caller ordering.
- **Next:** Retain both acceptance and refusal tests through final certification;
  record this as an audit-introduced regression, not an original model defect.

### AUD2-031 — SwiftPM coverage JSON represents only the final test bundle

**S1 · Tier A · PROGRESS · audit evidence/coverage**

- **Location:** [CI workflow](.github/workflows/ci.yml) and
  [coverage evidence](AUDIT/baseline.md).
- **Finding:** After seven bundles passed, SwiftPM's JSON contained only 27 files
  associated with ContinuityCore, including tests/generated code. Its 96.22%
  cannot be reported as package or production coverage. CI checks artifact
  presence, not represented targets; remote artifact contents were not inspected.
- **Current evidence:** A successful local export explicitly names all seven
  current test binaries and excludes build/tests/dependencies. It maps 395 source
  files; excluding reference support gives 378 shipping-source files and
  50,611/62,974 covered lines (80.37%). Unmapped Swift/C files and all Metal code
  are disclosed below. No CI pipeline was modified; the durable gap is unfixed.
- **Next:** Obtain approval for the needed pipeline change, implement target/scope
  validation and inspect a fresh CI artifact. Do not invent a coverage threshold
  or call the local percentage whole-repository coverage.

## Recorded verification and its limits

These are recorded 2026-10-23 runs, not tests rerun merely to write this document.
Host evidence reports an M3 MacBook Pro with 24 GB RAM, macOS 27.0.1 (26A434),
Xcode 27.1 (27A9275), Swift `swiftlang-6.4.0.34.1`, driver 1.168.6. The measured
source was the dirty working tree at the HEAD above. Resource availability must
be rechecked before future model runs; an old inventory is not current preflight.

### Full serial Swift run

Exact command from the repository root:

```bash
TMPDIR="$PWD/AUDIT/scratch" env -u TINYTITAN_LIBRARY_CONTRACT_MODEL \
  xcrun swift test --no-parallel --enable-code-coverage \
  > AUDIT/continuation-full-swift-tests-postformat.log 2>&1
```

The command tool reported **exit 0**; the exit status is not printed in the raw
log. Build: **26.28 seconds**. All 5,318 lines were independently reviewed; no
source/test changes occurred during the run. There were no unexpected failures,
compiler/linker warnings or crashes; intentional error-path diagnostics remain.
This was not a sanitizer run.

Bundle footers, including skipped tests/suites:

- TinyTitanTests: **949 tests / 150 suites / 163.568 s**; 946 passed, 3 skipped.
- TinyTitanServerTests: **422 / 48 / 21.322 s**; 420 passed, 2 skipped.
- TinyTitanRepackTests: **168 / 25 / 5.286 s**; 167 passed, 1 skipped.
- TinyTitanMemoryTests: **189 / 23 / 0.524 s**; 189 passed, none skipped.
- TinyTitanLibTests: **96 / 16 / 1.125 s**; 91 passed, 5 skipped.
- TinyTitanFleetTests: **63 / 14 / 1.179 s**; 63 passed, none skipped.
- ContinuityCoreTests: **98 / 10 / 0.908 s**; 98 passed, none skipped.

Total: **1,974 passed + 11 skipped = 1,985 counted tests**. The 286 counted suites
include three skipped suites. Parameter cases are not additional footer tests.
The seven zero-test XCTest headings are not the substantive Swift Testing runs.

The 11 skipped tests, with reasons derived from source rather than raw log text:

1. `theGPUEngineContinuesTheOraclesOwnChecks()` — opt-in
   `TINYTITAN_DENSE_GPU_EQUIV`.
2. `denseInstallMatchesItsSnapshot()` — opt-in `TINYTITAN_DENSE_EQUIV`.
3. `tensorOps2DFullAttentionMatchesReferenceAtTileBoundaries(_:)` — Metal
   `.apple10`, unavailable on this M3.
4. `stubServer()` — manual `TINYTITAN_STUB_SERVER_SECONDS` lifetime.
5. `theRealInstallAnswersThroughTheFactoryAndTheAdapter()` —
   `TINYTITAN_SIDE_ENGINE_E2E` plus a non-debug build.
6. `The pinned checkpoint's own config.json parses and matches` — existing
   config named by `TINYTITAN_QWEN38_CONFIG`.
7. `twoEnginesOnOneDeviceRunAlternately()` — library contract model gate.
8. `twoGenerationsOnOneSessionBothComplete()` — same gate.
9. `cancellingMidDecodeEndsTheGeneration()` — same gate.
10. `unloadingShutsLiveSessionsDown()` — same gate.
11. `aToolLoopRoundTrips()` — same model gate.

For the final five tests, `TINYTITAN_LIBRARY_CONTRACT_MODEL` was explicitly unset.
These skips are not evidence that lifetime, cancellation or live tool loops work.

### Focused tests and lint

Recorded commands/results from the repository root:

```bash
TMPDIR="$PWD/AUDIT/scratch" xcrun swift test --no-parallel --filter 'RANS|rANS'
TMPDIR="$PWD/AUDIT/scratch" xcrun swift test --no-parallel --filter PromptCheckpointStoreTests
/bin/bash tools/lint.sh > AUDIT/continuation-lint-postverification.log 2>&1
```

- rANS: **19 tests / 3 suites, exit 0**; 11.37-second build, bundle footers
  13 tests / 0.209 s and 6 tests / 0.012 s (`rans-validation-final.log`).
- Checkpoint store: **28 tests / 1 suite, exit 0**; 10.46-second build,
  0.057-second suite (`checkpoint-byte-counts-green.log`).
- Lint: **18/18 gates passed, exit 0**. Source-length scan: 419 production Swift
  files, largest 500 lines; strict formatting: 692 Swift files. This does not
  establish complete manual or model verification.

M5 command, repeated with each interpreter listed in the environment record:

```bash
TMPDIR="$PWD/AUDIT/scratch" PYTHONDONTWRITEBYTECODE=1 \
  <interpreter> -W default::ResourceWarning -m unittest -v benchmark.test_tinytitan_m5_sweep
```

**24 tests, exit 0, zero ResourceWarnings** on each of Python 3.13.14 (0.046 s),
3.14.9 (0.071 s) and 3.15.0 (0.054 s); local logs are
`m5-handle-green-3.13.log`, `m5-handle-green-3.14.log` and
`m5-handle-green-3.15.log`. This focused success does not replace the full matrix.

The earlier 108-suite-per-interpreter matrix remains qualified:

- Python 3.13: 1,728 passes, 18 failures, 10 errors and 87 skips, **exit 1**;
  required converter/analysis dependencies were absent. Seven error entries were
  unittest loader placeholders. This runtime was not certified.
- Python 3.14/3.15: each reported 2,077 tests, zero failures/errors/skips and
  exit 0, but each emitted 12 old M5 ResourceWarnings. Their NumPy 2.5.4 was not
  the committed 2.5.3 pin, so neither certifies the pinned dependency stack.
- All 226 tracked Python files parsed on all three interpreters; that is syntax
  evidence only. No model/CoreML verification or dependency installation followed.

### Historical checks, not final-tree certification

The baseline records a release build of 589/589 modules in 137.8 seconds with no
warnings; converter suites passed 87 tests in 42.3 seconds; gitleaks scanned
1,981 commits/33.2 MB in 12.4 seconds without findings. These precede the final
pending source state and do not certify it.

Historical sanitizer evidence records TSan 224 tests / 29 suites and C ASan/UBSan
29 tests / 7 suites, exit 0 without sanitizer findings. The separate keep-warm
TSan GREEN records three tests. None is a fresh sanitizer run over every final
edit. Configured dependency/CodeQL jobs and earlier CI references are not proof
that clean final-commit CI passed.

### Corrected local coverage measurement

The original last-bundle JSON is preserved locally as
`AUDIT/continuation-codecov-last-bundle.json`: 27 files, 503 functions and
4,095/4,256 covered lines. The merged profile reports 11,984 functions. A
successful full test command therefore did not imply a complete JSON export.

The corrected command explicitly selects current manifest test binaries, avoiding
stale removed-product bundles still present in the build directory:

```bash
xcrun llvm-cov export --summary-only \
  --instr-profile=.build/out/Products/Debug/codecov/default.profdata \
  .build/out/Products/Debug/TinyTitanTests.xctest/Contents/MacOS/TinyTitanTests \
  --object=.build/out/Products/Debug/TinyTitanServerTests.xctest/Contents/MacOS/TinyTitanServerTests \
  --object=.build/out/Products/Debug/TinyTitanRepackTests.xctest/Contents/MacOS/TinyTitanRepackTests \
  --object=.build/out/Products/Debug/TinyTitanMemoryTests.xctest/Contents/MacOS/TinyTitanMemoryTests \
  --object=.build/out/Products/Debug/TinyTitanLibTests.xctest/Contents/MacOS/TinyTitanLibTests \
  --object=.build/out/Products/Debug/TinyTitanFleetTests.xctest/Contents/MacOS/TinyTitanFleetTests \
  --object=.build/out/Products/Debug/ContinuityCoreTests.xctest/Contents/MacOS/ContinuityCoreTests \
  --ignore-filename-regex='(^|/)(tests|\.build)/' \
  > AUDIT/swift-coverage-multi-bundle.json 2> AUDIT/swift-coverage-multi-bundle.stderr.log
```

Result: **exit 0, empty stderr**, 395 unique mapped source files, all under
`sources/`. Including 17 TinyTitanValidation reference-support files gives
**51,598/63,995 lines (80.63%)**. Excluding them leaves 378 mapped shipping-source
files: **50,611/62,974 lines (80.37%)**, 5,171/6,505 functions (79.49%), and
17,015/22,937 regions (74.18%). These are measurements, not acceptance thresholds.

Limits: 27 of 422 Swift/C files have no map: all seven benchmark files, six
executable entries, and QSASelection, FleetBrand, FleetUsage, ModelBackend,
ModelDescriptor, RunnerCounterSnapshot, InstallProgressDisplay, RemoteChunkPolicy,
VerifiedInstallManifest, ModelInstallProgress, BoundedScratch, ServerVersion,
OffByMultiples and Tolerances. Some contain only declarations; missing mapping is
not silently scored as either zero or full coverage. All 16 `.metal` files,
embedded GPU source, skipped model paths and release-only paths remain outside
what these percentages establish. Durable CI export/scope validation is still
unfixed; remote artifacts have not been inspected.

## Evidence corrections and test-change disclosure

- The original piped Swift invocation did not preserve a proven Swift exit code.
  Its final 98-test footer and zero-skip/pass assertions are withdrawn.
- The first continuation full run really failed: exit 1, signal 5 in the rANS
  overflow regression, after an 85.81-second build. Other surviving bundle
  footers do not turn that into a pass.
- A later successful run overlapped whitespace formatting of four tensor calls.
  It is not exact post-format evidence; the separate unchanged-source rerun above
  supplies that local evidence.
- The restart-LRU fixture used 17 bytes while expecting two 16-byte records.
  It was corrected to 32; the original mixed failure is not valid RED evidence.
- Three expansion negative fixtures were valid layouts. Their original failures
  are withdrawn; correcting those fixtures is not proof of the original claims.
- Calibration tests were changed to create actual manifests rather than use
  nonexistent `/tmp/model` paths. Existing assertions were retained; OS/model/
  family/width mismatch assertions were added. The 41-test rerun is the relevant
  recorded result, not the invalid-fixture runs.
- The rANS two-stream fixture was corrected from 64 to 4,096 symbols and its
  helper now writes declared stream lengths. The original misleading refusal
  pass is withdrawn. Header double-counting and optional-width equality were
  genuine audit-introduced regressions, separately recorded above.
- The two newly added checkpoint numeric tests initially used 1 MiB budgets,
  allowing eviction to hide acceptance. Only those new budgets were strengthened
  to `Int.max` before production changes. `checkpoint-byte-counts-red.log` is
  superseded by `checkpoint-payload-counts-red-valid.log` and
  `checkpoint-auxiliary-counts-red-valid.log`.
- The initial non-running keep-warm TSan compilation attempt is not RED;
  `keepwarm-tsan-red.log` contains the later actual running race reproduction.
- The continuation added six test methods: one Python handle test, one rANS
  overlap method, one coordinated-geometry refusal, one implicit-width acceptance
  and two checkpoint numeric methods with six cases each. No pre-existing
  assertions/helpers/fixtures changed in that continuation; earlier fixture
  corrections are disclosed separately above.
- Formatting touched the two already-untracked rANS test files and wrapped four
  calls in [TwoBitExpertReadTests](tests/TinyTitan/Infrastructure/Streaming/TwoBitExpertReadTests.swift).
  Their assertion values were unchanged. Initial formatter output was limited to
  30 lines, so its visible file list was never an exhaustive drift inventory.
- The checkpoint disk-method split, documented in
  [repository layout](docs/repository-layout.md), was pure code motion with only
  necessary private-to-internal access changes. Numeric validation followed
  separately; splitting the file did not repair persistence semantics.
- Earlier scratch/tool-proof files and targeted logs outside this checkout were
  workspace-policy deviations. Later scratch/logs use `AUDIT/` and
  `AUDIT/scratch`; no new model/dependency/toolchain installation, worktree,
  cache purge, commit or push was used to make verification pass.

## Continuation boundaries

Continue from the recorded issue evidence, not from a claim that passing tests
means all bugs are fixed. Preserve uncommitted work and the invalid-evidence
history. Do not close a static claim without the required reproduction/review,
replace a disproven claim with its sibling, or treat absent evidence as a pass.

The following repository constraints still apply:

> Installing a model is a separate, operator-requested job — never run it to
> satisfy a check, a gate, a benchmark or a release.

> Run package tests serially (`swift test --no-parallel`), passing extra arguments
> like `--filter` through. Run only one CLI or model-using test at a time.

> Verification uses only the models already installed under `models/`.

> Do not download a full checkpoint, duplicate the `.ssdai` model, create a
> worktree, or purge caches just to run tests, a gate or a release.

> Keep the server on `127.0.0.1`; it has no remote authentication or TLS, so do not
> proxy, tunnel, or expose it.

> A tool call from the local model never bypasses the client's normal permission
> policy.

> What needs a reason is optimization: do not start performance work, change
> runtime defaults, or alter numerics unless the user asked.

Before model use, apply the full preflight in [AGENTS.md](AGENTS.md): supported
macOS/toolchain, disk and memory pressure, completed selected install and no
existing model process. Do not kill somebody else's process or install a missing
model to satisfy a check. Final integration remains PR-based after Phase E;
writing or staging this report is not authorization to publish code or a release.
