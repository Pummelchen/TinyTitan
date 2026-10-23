# Pre-production audit — 23 October 2026

`[#AUD2-024 44/44 | done:17 open:26 blocked:1 new:2] DONE — the rANS cluster is
closed at 61de182: both S0s and seven related S1/S2 records, with geometry bound
to the tensor layout, publication held to the final wave, and the scale, stream
and overlap guards proven by tests`

**Audit incomplete. Known defects remain. No product defect is formally closed.**
Two documentation records (AUD2-022, AUD2-032) are closed; AUD2-037 is closed as
a process record once its four passes were run; AUD2-040 to AUD2-044 and the
rANS cluster (AUD2-010, 011, 012, 018, 023, 024, 025, 027, 030) are closed with
remediation commits, and none is merged, released or certified on a clean final
revision.

This is the requested dated checklist and issue report, not a second project
backlog. [AUDIT/ledger.json](AUDIT/ledger.json) remains the audit's authoritative
record; the [task-table standard](docs/task-table-standard.md) and existing project
tracker are unchanged. The issue notes below preserve the ledger's IDs, severity,
tier and status, including disproven claims and audit-introduced regressions.

## Snapshot and interpretation

- Audit: `pre-production-audit-2026-10-23`; original brief now preserved verbatim
  at [AUDIT/audit-brief.md](AUDIT/audit-brief.md) (operator attachment, 13
  sections, SHA-256 `0390fa3d…aec3e2`).
- Branch: `audit/2026-10-23`; audit base: `4816a84`.
- Measured HEAD: `61de182f1d3e141e92314f079cf1f0e606fcb647`, which carries the
  rANS cluster on top of the AUD2-044 fix and the AUD2-040 to AUD2-043 chain.
  **The tree also carries uncommitted source and test changes** belonging to
  records still in START/PROGRESS/TEST, so that commit alone does not contain
  every tested fix.
- **44 records: 17 DONE, 26 unresolved/non-blocked, 1 BLOCKED.**
- Exact statuses: **17 OPEN, 2 START, 2 PROGRESS, 3 TEST, 2 AUDIT, 17 DONE,
  1 BLOCKED**.
- Recorded severities: **2 S0, 16 S1, 25 S2, 1 S3**. These are ledger
  classifications, not a claim that all 44 records are confirmed product bugs.
- `OPEN`, `START` and `PROGRESS` remain unfinished; `TEST` means local remediation
  evidence exists, not closure. `AUDIT` retains a claim/disposition awaiting formal
  reconciliation. `BLOCKED` is not a passing verification.
- AUD2-015 and AUD2-016 were disproven as originally stated. AUD2-019, AUD2-022,
  AUD2-031, AUD2-037 and AUD2-038 concern verification, rather than independent
  runtime defects. AUD2-022 (Swift 6.4.1), AUD2-032 (brief recovery) and AUD2-037
  (the four brief-mandated passes) are DONE and close no product defect;
  AUD2-040 (stale diagnostic), AUD2-041 (`min_p` accepted and ignored), AUD2-042
  (five unreferenced declarations), AUD2-043 (an orphaned expert-I/O diagnostic)
  and AUD2-044 (late port validation) are DONE with remediation commits, as are
  the nine rANS records closed at `61de182`.
- AUD2-040 is `dc52729`, AUD2-041 is `9ca9942`, AUD2-042 is `ccfee98`, AUD2-043 is
  `c8d04c4`, AUD2-044 is `3ac4699` and the rANS cluster is `61de182`, all on the
  audit branch. Removal tickets
  TT-050 and TT-051 are filed in the project tracker (wiki commit `081335d`). No
  fix is represented here as merged, released, or certified on a clean final
  revision.

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
- [x] Reconcile this operational checklist against every requirement in the
  original 13-section brief before claiming Phase E completion. *(The brief was
  recovered and is preserved verbatim; the reconciliation table below records
  what is evidenced, what is missing and what is blocked. Reconciliation is not
  completion: see AUD2-038.)*
- [x] Verify the package under the requested Swift 6.4.1 compiler. *(Homebrew keg
  6.4.1 built the package and ran the full serial suite, both exit 0. Its 49
  linker warnings are tracked as AUD2-039.)*
- [ ] Run it on a clean final revision; the 6.4.1 runs above used the dirty tree.

### Reconciliation against the recovered brief

The brief is [preserved verbatim](AUDIT/audit-brief.md). Its requirements are
mapped below to the evidence that exists in this report. `Partial` means the
requirement was worked but its own definition of done is not met.

| Brief section | Requirement | State | Where |
| --- | --- | --- | --- |
| §0 | No weakened checks, branch/reporting discipline, no secrets, no narrowing | Met for all work recorded here | This report's corrections log; `git` history on `audit/2026-10-23` |
| §1 | One pinned toolchain per language; formatter, linter, CVE scanner, type checker | Partial | [environment](AUDIT/environment.md), [tool proofs](AUDIT/tool-coverage.md) |
| §1 | Swift 6 language mode, strict concurrency, warnings-as-errors, SwiftLint `--strict` | Met and proven by deliberate-violation build | [tool proofs](AUDIT/tool-coverage.md) |
| §1 | Swift 6.4 / Xcode 27 exactly | Met — Homebrew `swift` 6.4.1 builds and tests the package; 49 linker warnings tracked as AUD2-039 | AUD2-022, AUD2-039 |
| §1 | Strict C99 with the named hardening flags, proven to reject GNU extensions | Met and proven | [tool proofs](AUDIT/tool-coverage.md) |
| §1 | Python basic: Ruff format+lint, B/E722/S101/PT rules, bare-except proof, pip-audit | Partial — Ruff and the proof exist; the 3.13 matrix is not certified on the pinned stack | [baseline](AUDIT/baseline.md), [tool proofs](AUDIT/tool-coverage.md) |
| §1 | Tool-coverage proof for every delegated check | Met for the delegated checks that were listed | [tool proofs](AUDIT/tool-coverage.md) |
| §1 | Sanitizers run the suite (C ASan/UBSan, Swift TSan) | Partial — runs exist, none on the final edits | This report, historical checks |
| §1b | One primary host recorded | Met | [environment](AUDIT/environment.md) |
| §1b, §11, §12 | One independent host runs the single final verification | **Not met** | AUD2-038 (BLOCKED) |
| §2 | Inventory, 2-hop dependency graph, trust boundaries, tier table | Met | [scope and tiers](AUDIT/scope-tiers.md) |
| §3 | Baseline with actual language mode, warnings, tests, lint, CVEs, secret scan | Partial — baseline exists and was re-measured; the 6.4.1 build adds 49 linker warnings (AUD2-039) | [baseline](AUDIT/baseline.md) |
| §4 L0–L4 | Repository, architecture, module, line and security passes at tier depth | Partial — rANS, checkpoint, server, CLI, PLE and Metal areas reviewed; no complete Tier A batch review | Issue record below |
| §4 L5 | Performance pass on Tier A and profiler-identified hot paths | Met — no N+1, unbounded collection, missing stream or blocking request-path call; no new measurement taken or claimed | AUD2-037 |
| §4 L6 | Coverage gaps, failure paths, tests that assert nothing, cross-project contracts | Partial — coverage export corrected; contract tests partly run | AUD2-031, this report |
| §4 L7 | Ops pass per deployable service | Met — health, signals, graceful shutdown, structured errors and rollback checked; the one startup-validation gap is fixed | AUD2-037, AUD2-044 |
| §5 | Facade hunt with a production-path facade as S0 | Partial — the hunt ran 2026-10-23; no production-path facade found, both outcomes now fixed and DONE | AUD2-037, AUD2-040, AUD2-041 |
| §6 | Unused code limited to Tier A and audit-orphaned code | Met for the sweep — five unreferenced declarations deprecated behind TT-050/TT-051, one orphaned diagnostic restored and live-verified | AUD2-042, AUD2-043 |
| §7 | Severity order S0 → S3, nothing dropped | Met in classification; remediation is not done | Ledger |
| §8 | Ledger schema and S0/S1 gates | Met for the closed records — both S0s and the rANS S1s ran every stage; their fixes share one commit, a deviation recorded in the ledger notes | [ledger](AUDIT/ledger.md) |
| §8 | Non-terminal count strictly decreasing at every milestone | Not yet due — the run is still in Phase B discovery, and these milestones add the records discovery finds | Ledger notes |
| §9 | Ledger is the single source; report generated from it | Met | [ledger](AUDIT/ledger.md) |
| §10 | One focused commit per S0/S1; audit branch only | Partial — `dc52729`, `9ca9942`, `ccfee98`, `c8d04c4`, `3ac4699` and `61de182` are focused commits, but the rANS cluster merges nine interleaved records into one; the rest of the tree's fixes are still uncommitted | `git log` |
| §11 | Phase A inventory/baseline/environment | Met | Committed |
| §11 | Phase B L0–L7 loop | In progress — every pass now has recorded work | This report |
| §11 | Phase C drain to zero | In progress — 17 DONE, 26 non-terminal, 1 BLOCKED | Ledger |
| §11 | Phase D convergence | Not started — requires zero open | Ledger |
| §11, §12 | Phase E clean run on the independent host, open count zero | **Not met** | AUD2-038; 26 non-terminal records |
| §13 | The `[#id n/total …] STATUS — title` status line | Met | Status line at the top |

### Full current-tree review

Every Tier A surface requires the brief's complete L2/L3/L4 manual passes on the
current tree. Earlier audit closure, delta-only review, narrow patch review,
linters and test coverage do not discharge these items.

The 2026-10-23 continuation delegated eleven module-batch reviews to independent
subagents and **every one terminated on the operator's credit limit before
emitting a finding**. Their preambles were recovered; four reviewer leads were
re-derived in the tree and filed as AUD2-033…036, and a fifth lead (a claimed
`workspaceId` type error) was disproven. AUD2-032, the separate missing-brief
blocker, is now closed: the brief was recovered and preserved, and reconciling
against it exposed four unrun passes (AUD2-037) and the unmet independent-host
requirement (AUD2-038). That is targeted verification, not a substitute for the
listed passes: no batch produced a complete L2/L3/L4 review, so every item below
stays unchecked and the full current-tree review remains incomplete.

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
  timing footers. *(Partial: the release build and the serial suite both now run
  under the required 6.4.1 compiler, but on the dirty tree; the clean-final-
  revision run remains outstanding.)*
- [ ] Rerun relevant TSan and C ASan/UBSan suites on that final revision; older
  sanitizer results do not certify later edits.
- [ ] Reconcile Python 3.13/3.14/3.15 runtime verification with the pinned
  dependency stack; syntax checks and the focused M5 suite are insufficient.
- [ ] Inspect final-revision dependency/security/CodeQL evidence and secret scans,
  rather than assuming configured CI jobs succeeded.
- [ ] Implement and verify durable multi-bundle CI coverage export and scope
  validation; inspect the actual resulting artifact. No pipeline change is
  authorized merely by this documentation request. *(Partial: a fresh CI artifact
  was inspected and contains one bundle / 27 files; the export itself is still
  unfixed and no pipeline file was changed.)*
- [x] Run applicable model/golden and Engine/Session contract checks using only
  already-installed models after the required host/process preflight; explicitly
  report missing installs or unsupported hardware as not checked.
- [ ] Exercise relevant release-only and opt-in paths, or record their precise
  missing preconditions. Do not relabel skipped tests as passes.
- [x] Rerun all 18 lint gates, documentation checks and patch hygiene after the
  final changes; retain warnings/errors instead of masking tool status. *(This
  certifies the measured dirty tree; it is not a clean-commit certification.)*
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

**S1 · Tier A · DONE · TinyTitan**

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

**S1 · Tier A · DONE · TinyTitanFormat/TinyTitan**

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

**S1 · Tier A · DONE · TinyTitan**

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

**S1 · Tier A · DONE · TinyTitan**

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

### AUD2-022 — Requested Swift 6.4.1 was missed by an Xcode-only inventory

**S1 · Tier A · DONE · toolchain verification**

- **Location:** [environment evidence](AUDIT/environment.md) and
  [baseline](AUDIT/baseline.md).
- **Finding:** The earlier inventory queried XcodeDefault and CommandLineTools,
  found only `swiftlang-6.4.0.34.1`, and concluded that no 6.4.1 compiler existed.
  That search omitted Homebrew's keg-only formulae, so the conclusion was wrong:
  it was an audit false negative, not a missing toolchain.
- **Resolution:** Homebrew's `swift` formula 6.4.1 is installed at
  `/opt/homebrew/opt/swift` (Cellar `swift/6.4.1`, keg-only, 2.5 GB) and reports
  `Swift version 6.4.1 (swift-6.4.1-RELEASE)`, target
  `arm64-apple-macosx27.0.0`. `xcrun swift` remains Xcode's 6.4.0, so 6.4.1 runs
  invoke the keg by absolute path.
- **Evidence:** Debug build **exit 0**, `Build complete! (72.81 sec)`, 1,258
  tasks; release build **exit 0**, `Build complete! (125.84 sec)`, 972 tasks;
  both with zero compiler errors. Full serial suite **exit 0**, all seven bundles,
  1,985 counted tests, the same 11 skips as the 6.4.0 run, zero compiler errors.
  All used `--scratch-path .build/brew641*`, leaving the main build tree untouched.
  The language standard is unchanged: `swiftLanguageModes: [.v6]` plus
  `-warnings-as-errors` on every target.
- **Regression:** The keg adds **49 debug / 41 release linker deployment-target
  warnings** against a zero-warning baseline. Filed as AUD2-039 rather than
  smoothed over.
- **Commit:** `250a03a` on `audit/2026-10-23`.

### AUD2-023 — rANS publishes before all expert waves are decoded

**S0 · Tier A · DONE · TinyTitan**

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

**S0 · Tier A · DONE · TinyTitan/TinyTitanFormat**

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

**S1 · Tier A · DONE · TinyTitanFormat**

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

**S1 · Tier A · DONE · TinyTitanFormat**

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

**S2 · Tier A · DONE · TinyTitan**

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
  are disclosed below. A fresh CI artifact from run `39974492789`
  (`AUDIT/ci-coverage-39974492789/TinyTitan.json`) was then inspected: it holds
  **one bundle and 27 files**, confirming remotely that the defect is in CI output
  and not only in the local export. No CI pipeline was modified; the durable gap
  is unfixed.
- **Next:** Obtain approval for the needed pipeline change, implement target/scope
  validation and inspect a fresh CI artifact. Do not invent a coverage threshold
  or call the local percentage whole-repository coverage.

### AUD2-032 — The original 13-section audit brief was not in the repository

**S2 · Tier A · DONE · audit evidence/scope**

- **Location:** [audit ledger](AUDIT/ledger.json) metadata,
  [brief](AUDIT/audit-brief.md) and this report.
- **Finding:** Until 2026-10-23 the brief was named only as “Attachment
  `9ca50da3` (pre-production audit + remediation, 13 sections)”, and an
  exhaustive read-only search of `AUDIT/`, `docs/` and the supplied transcript
  found no attachment body or path.
- **Resolution:** The operator supplied the original brief. It is preserved
  verbatim at [AUDIT/audit-brief.md](AUDIT/audit-brief.md) behind a provenance
  header naming the source attachment, the recovery date, the size and the
  SHA-256. The trailing 17,800 bytes of the preserved file hash to
  `0390fa3d63e92f40d8eaedff2a1c1f74c5cbd07b670a42f32123663fa2aec3e2`, matching
  the supplied attachment byte for byte.
- **Result:** The reconciliation table near the top of this report now maps
  every brief section to its evidence. That reconciliation did not produce a
  clean bill: it found four mandated passes with no recorded work (AUD2-037) and
  one host requirement the audit cannot satisfy alone (AUD2-038). Section 12
  completion remains unclaimed.
- **Commit:** `0f61aa5` on `audit/2026-10-23`.

### AUD2-033 — LZFSE first-stage publication does not wait for the last wave

**S1 · Tier A · OPEN · TinyTitan runtime/streaming**

- **Location:** [compressed plan execution](sources/TinyTitan/Infrastructure/Streaming/PreadExpertStreamer+IO.swift)
  (lines 259–275), [rANS contrast](sources/TinyTitan/Infrastructure/Streaming/PreadExpertStreamer+RANS.swift)
  (line 24) and [early fixup](sources/TinyTitan/Runtime/Inference/RealForwardRunner+DecodeMoEPhase1.swift)
  (lines 123–171).
- **Finding:** `executeCompressedMisses` walks `plan.misses` in waves of 8 and
  passes `firstStage` only when `start == 0`; `decodeCompressedSet` invokes it
  after that first wave's phase-0 blocks decode. `encodeDecodeEarlyMissFixup`
  awaits `firstStageCompletion()` and then encodes phase 1 over
  `phase1MissSlots`, which is populated from **every** index of `plan.misses`.
  With more than 8 misses, the gate/up halves of later waves are not resident when
  GPU phase-1 work for their slots is committed. The rANS path was fixed for this
  exact class in AUD2-023 and now passes `firstStage` only when
  `end == plan.misses.count`; the LZFSE path retains the earlier condition.
- **Current evidence:** Confirmed statically in the current tree. Not reproduced
  live: the 2026-10-23 ornith-8 golden check is batch-1 greedy decode and passed
  byte-identical, which does not exercise a >8-miss plan. Treat as S0 if a shipped
  front end reaches that plan shape, as the rANS record implies.
- **Next:** No fix applied. Make the LZFSE condition match the rANS final-wave rule
  (`end == plan.misses.count`), or publish only after every wave's phase 0; then
  retain a >8-miss regression test.

### AUD2-034 — A stalled request body is never timed out once its head is in flight

**S2 · Tier A · OPEN · TinyTitanServerCore**

- **Location:** [HTTP handler](sources/TinyTitanServer/Core/HTTPServerHandler.swift)
  (lines 117–152) and [deadline plumbing](sources/TinyTitanServer/Core/HTTPServerHandler+Plumbing.swift)
  (line 207), against `idleReadTimeout = .seconds(120)`
  ([server](sources/TinyTitanServer/Core/HTTPServer.swift), line 15).
- **Finding:** `channelRead` resets the 120-second idle deadline on every read and
  increments `inFlightRequests` when a valid head arrives. `resetIdleDeadline`'s
  scheduled task closes only when `inFlightRequests == 0` and is not re-armed after
  it fires. A client that sends headers plus a partial body therefore leaves
  `inFlightRequests == 1` forever: the timer fires, declines to close, and nothing
  else closes the connection. The body buffer (up to the configured limit) and the
  connection are held indefinitely. The comment above `resetIdleDeadline` claims a
  stalled slowloris request is closed, which this path cannot do.
- **Current evidence:** Verified by code path only; no live stalled-client probe
  was run. Loopback-only binding limits the blast radius, but this is local denial
  of service and contradicts the stated control.
- **Next:** No fix applied. Add a body-completion deadline (or re-arm while a body
  is incomplete), distinct from the no-in-flight idle close, and keep the
  never-close-mid-generation rule.

### AUD2-035 — CLI Ctrl-C cancels the outer task, so an in-flight generation keeps running

**S2 · Tier A · OPEN · TinyTitanCLI / TinyTitanLib**

- **Location:** [CLI signal handler](sources/TinyTitanCLI/Command/main.swift)
  (lines 49–52) and [session one-shot](sources/TinyTitanLib/Session.swift)
  (lines 89–105).
- **Finding:** The SIGINT handler cancels only the task that drives `run(args:)`.
  `Session.respond` wraps generation in an unstructured `Task` and awaits
  `work.value`; cancellation does not propagate into an unstructured task, and
  awaiting `Task.value` from a cancelled task does not throw. The CLI never calls
  `Session.cancel()`, which is the path the server uses. A long generation
  therefore ignores SIGINT until it finishes on its own.
- **Current evidence:** Observed live on 2026-10-23 against the installed
  `models/qwen3.5_4B_4Bit`. SIGINT was delivered to the running CLI mid-decode
  (stdout 1,008 bytes); the process stayed alive and kept writing (2,037 bytes at
  +10 s, 2,468 at +15 s), then produced the normal
  `[stop=maxTokens prefill=18tok new=800tok decode=35.95s tok/s=22.255]` footer
  and exited. No cancellation occurred. The exit status was not captured because
  the process was started outside the measuring shell's job table; the normal
  completion footer is the evidence that generation ran to its limit.
- **Next:** No fix applied. Either call `session.cancel()` from the SIGINT handler,
  or replace `runOneShot` with a cancellation-forwarding structure and keep the
  `.cancelled` mapping; then add a probe that observes the interruption.

### AUD2-036 — PLE prefill gather is bounded by the scratch chunk, not the PLEBlock rows

**S2 · Tier A · OPEN · TinyTitan runtime/prefill**

- **Location:** [PLE block construction](sources/TinyTitan/Runtime/Inference/RealForwardRunner+BuildCore.swift)
  (line 309), [chunk guard](sources/TinyTitan/Runtime/Inference/RealForwardRunner+PrefillLayer.swift)
  (line 57), [gather](sources/TinyTitan/Runtime/Inference/RealForwardRunner+PLE.swift)
  (line 90), [PLEBlock allocation](sources/TinyTitan/Kernels/Primitives/PLEBlock.swift)
  (line 136) and [prefill config](sources/TinyTitan/Runtime/Prefill/PrefillRuntimeConfig.swift)
  (line 155).
- **Finding:** `PLEBlock` allocates `embedding = embedDim * maxRows` with
  `maxRows = runtimeConfiguration.prefillChunkTokens`.
  `executePrefillChunk` checks `tokens.count` against
  `scratch.layout.chunkTokens`, which is built from the caller's
  `PrefillRuntimeConfig`; `production(chunkTokens:)` accepts any value in
  `allowedPrefillChunkTokens` independently of the runner's own configuration.
  `gatherPLERowsPrefill` then writes `tokens.count` rows of `headCount*rowBytes`
  into that buffer. A caller passing a chunk larger than the runner's configured
  `prefillChunkTokens` passes the scratch guard and writes past the PLE embedding
  allocation in host-visible shared Metal memory.
- **Current evidence:** Static analysis only. Shipped front ends appear to pass the
  engine's own `prefillChunkTokens`, so reachability requires a caller that
  constructs a mismatched `PrefillRuntimeConfig`; `RealForwardRunner` is public in
  the engine target but is not part of the `TinyTitanLib` facade.
- **Next:** No fix applied. Bound the gather by `PLEBlock.maxRows` (or size
  `PLEBlock` from the effective chunk) and refuse a config whose `chunkTokens`
  exceeds the runner's.

### AUD2-037 — Four brief-mandated passes have no recorded evidence

**S1 · Tier A · DONE · audit process**

- **Location:** [original brief](AUDIT/audit-brief.md), §5, §6, L5 and L7; this
  report.
- **Finding:** The recovered brief requires four passes that this audit has not
  recorded. **§5 facade hunt** looks for work that is faked on a production path:
  always-true validators or auth checks, hardcoded success returns,
  `NotImplemented*`/`fatalError("TODO")`/empty bodies, canned data served outside
  tests, `sleep()` standing in for processing, localhost/test defaults and
  surfaces wired to nothing. A production-path facade is an S0 by the brief's own
  rule. **§6 unused code** is limited to Tier A modules and code this audit
  orphaned. **L5 performance** covers Tier A and profiler-identified hot paths.
  **L7 ops** covers each deployable service once: structured errors, health
  checks, graceful shutdown, startup config validation and rollback path.
- **Current evidence:** None of the four has a section here or a ledger record.
  Existing checks are narrow proxies, not the passes: the `test-hollow` and
  `stdout-clean` lint gates cover thin slices, and the TSan, golden and contract
  runs exercise neither L5 hot paths nor L7 service behaviour. Filing this record
  is deliberately not the same as running the passes, and no facade, unused-code,
  performance or ops conclusion is asserted anywhere in this report.
- **Progress (2026-10-23):** the **§5 facade hunt** has run over `sources/`.
  Sixteen trap sites, eight "not implemented" strings, fifteen sleep calls, the
  endpoint/`localhost` defaults, hardcoded success strings, canned data and the
  literal `return true` sites in the server, repack and fused-decode paths were
  each traced to their callers. No production-path facade was found: the traps
  are fail-loud construction guards or bench code, every sleep is a poll interval,
  retry backoff, a 250 µs batch-coalescing delay or keep-warm cadence, and the
  capability checks are real. The hunt produced two findings — AUD2-041 (S1, a
  sampling parameter documented as refused that the wire never reads) and
  AUD2-040 (S2, a diagnostic that names a feature as unimplemented when it is
  implemented and the guard is unreachable). Both are now DONE (`9ca9942`,
  `dc52729`), each with a failing-then-passing test.
- **Progress (2026-10-23, §6):** the **Tier A unused-code sweep** enumerated the
  declarations of the Tier A modules and this audit's orphaned code, resolving
  every candidate against `sources/`, `tests/` and `plugins/`. Five declarations
  have no reference anywhere but themselves (AUD2-042, S3): they are deprecated
  with the removal tickets TT-050 and TT-051 rather than deleted, which is §6's
  default and keeps the reachability proof unspent. One audit-adjacent
  regression surfaced (AUD2-043, S2): `0cdba76` removed the only printer of the
  `[decode expert io]` line when generation moved onto the library, leaving
  `TINYTITAN_DECODE_IO_TRACE`, the runner's baseline and `decodeExpertIO()` with
  no reporter while `benchmark/expert_cache_slots.py` still parsed the line as
  its primary metric. The reporter is restored in the library and the line was
  observed on a live run, with the benchmark's own regex matching it.
- **Progress (2026-10-23, L5/L7):** the last two passes ran at Tier A depth.
  **L5 performance** covered the expert-streaming hot path and its cache plan,
  the sampler, SSE emission, prefill tile scheduling and the C kernels: no N+1
  pattern (the request path issues no query; the single per-request disk read is
  one prompt-state file), no repeated-I/O defect on top of the design (the cache
  plan matches experts against resident slots in O(experts × slots) with arrays
  sized by those two counts, and the C reader's one allocation happens at open,
  not per read), no unbounded memory or collection (the response store is
  LRU-bounded at 256, the SSE outbox carries a capacity and an overflow flag, KV
  and prompt-cache memory come from the RAM budget), streaming where the brief
  asks, and no blocking call on the request path. No performance measurement was
  taken in this pass and none is claimed. **L7 ops** covered the one deployable
  service: `/health` answers from process state without loading a lazy model,
  SIGINT/SIGTERM run a graceful shutdown that cancels and awaits request tasks
  and exits 0, request errors are structured JSON, argument errors exit 2 on
  stderr, and rollback is the documented install-receipt re-issue. Startup
  configuration validation was complete except for the port, which was checked
  only by the bind that runs after the load — filed as AUD2-044 and fixed.
- **Resolution:** all four passes now have recorded evidence, and each defect
  they found has its own id and disposition: AUD2-040 and AUD2-041 (section 5),
  AUD2-042 and AUD2-043 (section 6), AUD2-044 (L7). Nothing was waived and no
  pass was replaced by a proxy.

### AUD2-038 — No independent host exists for the Phase E verification

**S1 · Tier A · BLOCKED · audit process / Phase E**

- **Location:** [original brief](AUDIT/audit-brief.md), §1b, §11 Phase E and §12;
  [environment](AUDIT/environment.md).
- **Finding:** The brief requires the single final whole-repository verification
  to run from a fresh clone on **one independent host**, and §12 makes that run a
  condition of completion. [The environment record](AUDIT/environment.md) names
  exactly one host, the primary development Mac. No second Mac, VPS or container
  host was provisioned, and no remote verification was attempted. The brief also
  requires asking before provisioning a VPS, so this cannot be resolved from
  inside the audit.
- **Current evidence:** Every run recorded here — the full serial suite, the
  release build, the lint gates, the coverage export, the golden check and the
  CLI probe — executed on the primary host against a dirty tree. None is a
  Phase E run.
- **Blocked on:** *repository operator*. Reason: an independent host is required
  and only the primary Mac is available. Tried: inventoried the recorded hosts and
  confirmed the scope of every run to date; no host was provisioned or contacted.
  Options: (1) provide access to an independent Apple Silicon Mac, or an approved
  Linux VPS for the non-Swift harness checks, for the single final Phase E run;
  (2) explicitly waive §1b and §12's independent-host requirement in writing,
  accepting a primary-host-only final verification and a correspondingly weaker
  completion claim.
- **Next:** The operator chooses an option above. Until then Phase E and §12
  completion are unclaimable, regardless of how many records reach DONE.

### AUD2-039 — Swift 6.4.1 keg adds 49 linker warnings to a zero-warning baseline

**S2 · Tier B · OPEN · build/toolchain**

- **Location:** [Package.swift](Package.swift) line 29 (`.macOS(.v26)`),
  [environment](AUDIT/environment.md) and [baseline](AUDIT/baseline.md).
- **Finding:** The Homebrew 6.4.1 keg targets `arm64-apple-macosx27.0.0` and
  links against the host's Swift runtime dylibs, which are built for macOS 27.0.
  The package declares macOS 26.0, so every link step emits
  `building for macOS-26.0, but linking with dylib '/usr/lib/swift/libswiftCore.dylib'
  which was built for newer version 27.0`. The 6.4.0 Xcode build of the same tree
  produced zero warnings, so this is a measurable regression against the baseline
  even though no source changed.
- **Current evidence:** `AUDIT/brew641-build.log` and
  `AUDIT/brew641-full-tests.log`: 49 warning lines, all of this one class, plus a
  `libSwiftScan` fallback notice. `AUDIT/brew641-release-build.log`: 41 warnings
  of the same class (972 tasks, 125.84 s). Zero compiler errors in every run;
  `-warnings-as-errors` still governs compilation, so these are linker notices
  rather than suppressed source diagnostics.
- **Options:** raise the deployment target to macOS 27 (changes the supported OS
  set, so it needs an operator decision), use an Xcode-bundled 6.4.1 toolchain
  whose runtime matches (none is installed), or accept the warnings in writing as
  a property of the audit's verification toolchain.
- **Next:** No change applied. The project's shipping toolchain remains Xcode 27 /
  Swift 6.4; this record exists so the warning delta is not lost.

### AUD2-040 — The QSA error text names a feature as unimplemented that is implemented

**S2 · Tier A · DONE · facade hunt / stale diagnostic**

- **Location:** [QSAExactness.swift:80-89](sources/TinyTitan/Runtime/Family/QSAExactness.swift#L80),
  raised by `requireQSAExact`
  ([RealForwardRunner+QSA.swift:396](sources/TinyTitan/Runtime/Inference/RealForwardRunner+QSA.swift#L396)).
- **Finding:** The error text says "The QSA indexer is not implemented yet" and
  tells the operator to shorten the prompt. The indexer is implemented
  ([QSAIndexer.swift](sources/TinyTitan/Kernels/Attention/QSAIndexer.swift)) and is
  constructed whenever a family enables one
  ([RealForwardRunner+BuildCore.swift:321](sources/TinyTitan/Runtime/Inference/RealForwardRunner+BuildCore.swift#L321),
  `try QSAIndexer(...)`); `staged` traps if the builder never sets it
  ([RealForwardRunner.swift:336](sources/TinyTitan/Runtime/Inference/RealForwardRunner.swift#L336)).
  `qsaExactness` is derived from the same flag, so the throw requires
  `qsaIndexer == nil && exactness != nil` — which the current builder cannot
  produce. The message is both unreachable today and wrong if it is ever reached:
  a real cause would be an indexer that failed to build, not an unimplemented
  feature, and shortening the prompt is not the remedy.
- **Current evidence:** Static call-chain trace above; `grep -rn "not implemented
  yet" tests/` returns nothing, so no test pins the string. The nearby comment at
  `requireQSADensePrefill` already states the guard "only fires for a family that
  declares a sparse indexer without one being built — which nothing does today",
  which the description contradicts.
- **Resolution (`dc52729`):** The description and doc comment now say this model's
  indexer was not built and point at the load diagnostics, keeping the guard as
  defence in depth. The new test `refusalNamesTheRealCause` pins that the text
  does not claim the feature is unimplemented. RED on the old text, GREEN after;
  `QSAExactnessTests` 8/8, full serial suite 7 bundles / 1,988 tests / exit 0, and
  all 18 lint gates pass.

### AUD2-041 — `min_p` is documented as refused but the request model never reads it

**S1 · Tier A · DONE · untrusted-input validation**

- **Location:** [OpenAIModels.swift](sources/TinyTitanLib/OpenAIModels.swift),
  [OpenAIRequestValidator.swift:300](sources/TinyTitanLib/OpenAIRequestValidator.swift#L300),
  [release-notes-v5.10.md:56](docs/release-notes-v5.10.md#L56).
- **Finding:** The release notes state "`min_p` is accepted but must be zero: the
  filter does not exist, and refusing beats sampling as if it did", and
  [adding-a-model.md](docs/adding-a-model.md) repeats that `min_p` is unsupported.
  The wire does neither: `OpenAIChatRequest` declares no `minP` field and no
  `min_p` coding key, and the server decodes bodies with a plain `JSONDecoder()`
  ([HTTPServerHandler+Chat.swift:25](sources/TinyTitanServer/Core/HTTPServerHandler+Chat.swift#L25)),
  which ignores unknown keys. The validator passes only
  `effectiveSampling.minP`, and every shipped row sets it to zero
  ([SamplerTypes.swift:18-21](sources/TinyTitan/Runtime/Generation/SamplerTypes.swift#L18)),
  so a client-sent `min_p` of any value reaches the sampler as `0` with no
  diagnostic. The engine's own refusal
  ([SamplerTypes.swift:179](sources/TinyTitan/Runtime/Generation/SamplerTypes.swift#L179))
  is reachable only from library callers; it is pinned by
  [SampleTopK64Tests.swift:24](tests/TinyTitan/Kernels/Sampling/SampleTopK64Tests.swift#L24),
  and nothing pins the HTTP path.
- **Current evidence:** Static proof above. Reproduction: a
  `/v1/chat/completions` body containing `"min_p":0.1` is accepted and sampled
  exactly as without it — the documented refusal never fires.
- **Resolution (`9ca9942`):** `OpenAIChatRequest` decodes `min_p`, and
  `OpenAIRequestValidator` refuses a non-zero value with a 400
  `unsupported_value`, as the release note promises;
  [adding-a-model.md](docs/adding-a-model.md) now says both filters are refused
  unless zero. RED: a `"min_p":0.1` body validated with minP `0.0`. GREEN:
  `nonZeroMinPIsRefused` and `zeroMinPIsAccepted` pass, 44 tests in the OpenAI
  validation suites, full serial suite 7 bundles / 1,988 tests / exit 0, all 18
  lint gates pass, and no S0/S1 gate item was waived.

### AUD2-042 — Five declarations have no caller anywhere in the tree

**S3 · Tier A · DONE · unused code**

- **Location:**
  [Model+LayerStreamers.swift:87-92](sources/TinyTitan/Runtime/Inference/Model+LayerStreamers.swift#L87),
  [Model.swift:185-192](sources/TinyTitan/Runtime/Inference/Model.swift#L185),
  [RealForwardRunner+Requirements.swift:40-47](sources/TinyTitan/Runtime/Inference/RealForwardRunner+Requirements.swift#L40),
  [ModelRouterError.swift:179-184](sources/TinyTitanServer/Core/ModelRouterError.swift#L179),
  [ServerMemory.swift:352-357](sources/TinyTitanServer/Core/ServerMemory.swift#L352).
- **Finding:** `releaseLayerStreamer(_:)`, `routedExpertBits(layer:expert:)`,
  `requireSharedScalarGateBuffer()`, `ModelRouter.reasoningChoice(for:)` and
  `ServerMemory.isFirstTurn(_:)` each resolve to exactly one line — their own
  declaration — across `sources/`, `tests/` and `plugins/`. A reference from a
  test target, the embedded example or a DeepSeek Harness plugin would have kept
  a declaration alive; there is none. The members are the remains of a
  layer-major prefill that did not land, a tiered install layout no shipped
  converter writes, and two convenience accessors whose call sites index the
  underlying maps directly.
- **Resolution (`ccfee98`):** all five are deprecated with
  `@available(*, deprecated, message:)` naming the removal ticket and the reason
  there is no caller — TT-050 for the engine declarations, TT-051 for the two
  ServerCore ones (wiki commit `081335d`). Nothing was deleted, which is §6's
  default: the removal tickets carry the deletion. `-warnings-as-errors` turns a
  future call into a build failure, so a reintroduced user is caught at the
  commit that adds it. Release build exit 0 with no warnings, full serial suite
  7 bundles / 1,990 tests / exit 0, 18/18 lint gates. S3 needs no TEST stage
  because nothing changed behavior.

### AUD2-043 — The decode-only expert I/O line lost its only emitter

**S2 · Tier A · DONE · observability**

- **Location:**
  [ServerModelSession+Diagnostics.swift:324-337](sources/TinyTitanLib/ServerModelSession+Diagnostics.swift#L324),
  [ExpertCacheDiagnostics.swift:82-101](sources/TinyTitanLib/ExpertCacheDiagnostics.swift#L82),
  [expert_cache_slots.py:48](benchmark/expert_cache_slots.py#L48).
- **Finding:** `TINYTITAN_DECODE_IO_TRACE` still selects the runner's statistics
  baseline and `decodeExpertIO()` still returns it, but no caller printed the
  result after `0cdba76` removed the four-line block in the CLI's run loop when
  generation moved onto the library. The consumer still expects it:
  `benchmark/expert_cache_slots.py` sets the flag, documents the printed shape
  and parses it with `IO` — the only source of hits, misses, hit rate and
  bytes/token — and `verdict()` exits 2 when a slot logs nothing. The flag, the
  baseline and the accessor therefore had no reporter while the sweep that
  depends on them would have reported `no data` for every slot.
- **Resolution (`c8d04c4`):** `emitGenerationDiagnostics` emits
  `ExpertCacheDiagnostics.decodeIOLine` whenever `decodeExpertIO()` is non-nil,
  outside the other diagnostic guards because the runner's own baseline already
  gates it. The formatter reproduces the shape the benchmark parses and omits
  the per-token figure when a run reports no new tokens. It lives in the library
  because the CLI reaches it through `Engine.load`, so there is one emitter.
  RED against the missing function, GREEN 8/8, full suite 7 bundles / 1,990
  tests / exit 0, 18/18 lint gates. Live: a `--max-new 8` run of the installed
  qwen3.8-flash-next 4-bit printed `[decode expert io] hits 2575 misses 785
  (76.6% hit) 2.02 GiB = 259.1 MiB/token`, and the benchmark's regex matched it
  with groups `('2575','785','76.6','2.02','259.1')`.

### AUD2-044 — An occupied --port is reported only after the full model load

**S2 · Tier A · DONE · ops / startup validation**

- **Location:**
  [main.swift:39-48](sources/TinyTitanServer/Command/main.swift#L39),
  [main.swift:206](sources/TinyTitanServer/Command/main.swift#L206),
  [HTTPServer.swift:125](sources/TinyTitanServer/Core/HTTPServer.swift#L125),
  [ServerPortCheck.swift](sources/TinyTitanServer/Core/ServerPortCheck.swift).
- **Finding:** `ServerArguments` validates the port's range but nothing asked
  whether it was free. The only listener binds at `HTTPServer.swift:125`, from a
  call that runs *after* `routing.preload()` has loaded the initial install —
  36.9 GB for the default ornith 8-bit. A second server started on a busy port
  therefore paid the whole load before it heard that the port was taken, and
  under `--lazy-load` the refusal landed in the first request instead. The
  brief's L7 startup-configuration item was satisfied for every option except
  this one.
- **Resolution (`3ac4699`):** `ServerPortCheck.isAvailable(port:)` probes
  127.0.0.1 with a socket, `SO_REUSEADDR` (matching the listener, so a restart
  whose connections are in `TIME_WAIT` is not falsely refused), `inet_pton` and
  a bind, treating only `EADDRINUSE` as unavailable and failing open on every
  other probe error so the real bind stays authoritative. `main` calls it as the
  first act of the serving block and exits 2 with the same `error: ` shape the
  argument errors use. RED against the missing type; GREEN 2/2 (listener present
  → unavailable; port free again after close); full suite 7 bundles / 1,992
  tests / exit 0; 18/18 lint gates. Live: an occupied 18451 made the release
  binary exit 2 in 0.43 s with `error: --port 18451 is already in use on
  127.0.0.1` and empty stdout, while the control on a free port served
  `/health {"status":"ok"}` and exited 0 on SIGTERM.

## Recorded verification and its limits

These are recorded 2026-10-23 runs, not tests rerun merely to write this document.
Host evidence reports an M3 MacBook Pro with 24 GB RAM, macOS 27.0.1 (26A434),
Xcode 27.1 (27A9275), Xcode's Swift `swiftlang-6.4.0.34.1` (driver 1.168.6), and
the Homebrew keg-only Swift 6.4.1. The measured source was the dirty working tree
at the HEAD above. Resource availability must be rechecked before future model
runs; an old inventory is not current preflight.

After the AUD2-040/041/042/043 fixes, two further runs were recorded on the same
host: the full serial suite (`swift test --no-parallel`, **7 bundles, 1,990
tests, 0 failures, exit 0**, `AUDIT/aud2-042-043-full-tests.log`) and all 18
lint gates (`tools/lint.sh`, exit 0, `AUDIT/aud2-042-043-lint.log`). The live
expert-I/O probe is `AUDIT/aud2-043-live.log`; it used the installed
qwen3.8-flash-next 4-bit under the machine's 12 GiB limit, and its preflight was
macOS 27.0.1, Xcode 27.1, 99 GiB free and 76% memory free with no other model
process running.

The AUD2-044 fix was verified the same way: full serial suite (`swift test
--no-parallel`, **7 bundles, 1,992 tests, 0 failures, exit 0**,
`AUDIT/aud2-044-full-tests.log`) and all 18 lint gates (`tools/lint.sh`, exit 0,
`AUDIT/aud2-044-lint.log`). Its live probe used the installed qwen3.5 4-bit under
`--lazy-load`, so no model was loaded in either arm; the refusal arm had a
python3 listener on 127.0.0.1:18451 and exited 2 in 0.43 s
(`AUDIT/aud2-044-live-occupied.err`), the control arm on 18452 answered
`/health {"status":"ok"}` and exited 0 on SIGTERM
(`AUDIT/aud2-044-live-free.out`, `-health.json`).

The rANS cluster at `61de182` was audited the same day: cold re-read of the whole
diff, then a focused rerun of the affected suites (**27 tests in 4 suites, exit
0**, `AUDIT/aud2-023-024-audit.log`); the full serial suite and the 18 lint gates
above ran on exactly that tree. AUD2-010's RED was demonstrated on the fix
itself — restoring its old `else { return }` made `refusesTruncatedPayload` fail
with `an error was expected but none was thrown` (`AUDIT/aud2-010-red.log`, exit
1) — and the committed file was restored before the paired GREEN (exit 0,
`AUDIT/aud2-010-green.log`). No live model run exercises these guards: they are
untrusted-sidecar paths, and the writer-to-GPU round trips in the suite stand in
for a real sidecar.

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
  files, largest 500 lines; strict formatting: 692 Swift files. A fresh full run
  after the last source changes (`full-audit-lint.log`) again exited **0** with
  every gate reporting its own count: 419 Swift files force-cast clean, 2,410
  functions (0 new offenders), 265 test files with no silent skip, 1,959 test
  bodies with none hollow, 169 allowlisted public declarations, 28 shell scripts
  under bash 3.2.57 and shellcheck 0.11.0, swiftlint 0.65.1 `--strict`, 692 Swift
  files clean under `swift-format`, both plugin packages under eslint 10.11.0 / prettier
  3.9.9, and ruff 0.16.7 with 227 scripts parsing under 3.13. This does not
  establish complete manual or model verification, and certifies the measured
  dirty tree rather than a clean commit.

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

### Release build, golden, contract and probe checks

Three further checks were recorded in this continuation, all on the same dirty
tree and installed models only:

```bash
TMPDIR="$PWD/AUDIT/scratch" xcrun swift build -c release \
  --disable-automatic-resolution > AUDIT/full-audit-release-build.log 2>&1
tools/golden-baseline.sh --check ornith-8 > AUDIT/golden-check-ornith8.log 2>&1
TINYTITAN_LIBRARY_CONTRACT_MODEL=models/qwen3.5_4B_4Bit \
  TMPDIR="$PWD/AUDIT/scratch" xcrun swift test --no-parallel \
  --filter LibraryContractTests > AUDIT/full-audit-library-contract.log 2>&1
```

- Release build: **exit 0**, `Build complete! (106.09 sec)`, final task counter
  **306/306**, **zero** `error:` and **zero** `warning:` lines.
- Golden: **`== ornith-8 ==` / `ok — output identical to baseline`**, `exit=0`.
  Ornith 1.5 8-bit is the correct target because its stored baseline was captured
  on this exact machine and OS; the qwen35-4b baseline was captured under macOS
  26.6.2 and is stale for this 27.0.1 host. The check is batch-1 greedy decode, so
  it does not exercise the >8-miss plan in AUD2-033.
- Library contract: **4 tests / 1 suite passed in 14.187 s, exit 0** against the
  installed `models/qwen3.5_4B_4Bit`. The four tests are
  `twoEnginesOnOneDeviceRunAlternately`, `twoGenerationsOnOneSessionBothComplete`,
  `cancellingMidDecodeEndsTheGeneration` and `unloadingShutsLiveSessionsDown`; the
  fifth, `aToolLoopRoundTrips`, was not part of this filtered run and remains
  unexercised. A pass here is not evidence for AUD2-035: the CLI path never calls
  `Session.cancel()`, whereas the suite exercises the cancellation API directly.
- CLI SIGINT probe: after the contract suite ended, the release binary was run
  against the installed 4B and SIGINT was sent mid-decode. At 1,008 stdout bytes
  the signal was delivered; the process stayed alive and output grew to 2,037
  bytes after 10 s and 2,468 after 15 s, then ended with the normal
  `[stop=maxTokens … new=800tok decode=35.95s tok/s=22.255]` footer. This is the
  first live confirmation of AUD2-035 rather than a source trace; the streams are
  `AUDIT/cli-sigint-probe.out` and `.err`.

### Swift 6.4.1 toolchain verification (AUD2-022)

The required 6.4.1 compiler is Homebrew's keg-only `swift` formula, not an Xcode
toolchain. Exact commands from the repository root:

```bash
/opt/homebrew/opt/swift/bin/swift --version          # 6.4.1 (swift-6.4.1-RELEASE)
TMPDIR="$PWD/AUDIT/scratch" /opt/homebrew/opt/swift/bin/swift build \
  --scratch-path .build/brew641 --disable-automatic-resolution
TMPDIR="$PWD/AUDIT/scratch" /opt/homebrew/opt/swift/bin/swift build -c release \
  --scratch-path .build/brew641-release --disable-automatic-resolution
TMPDIR="$PWD/AUDIT/scratch" env -u TINYTITAN_LIBRARY_CONTRACT_MODEL \
  /opt/homebrew/opt/swift/bin/swift test --scratch-path .build/brew641 --no-parallel
```

- Debug build: **exit 0**, `Build complete! (72.81 sec)`, 1,258 tasks, **zero
  compiler errors**, **49 linker warnings**.
- Release build: **exit 0**, `Build complete! (125.84 sec)`, 972 tasks, **zero
  compiler errors**, **41 linker warnings** (`AUDIT/brew641-release-build.log`).
- Tests: **exit 0**; seven bundles passed (949/150 suites in 61.754 s, 422/48 in
  19.696 s, 168/25 in 4.011 s, 189/23 in 0.456 s, 96/16 in 1.141 s, 63/14 in
  1.067 s, 98/10 in 0.917 s); **1,985 counted tests** with the same 11 skips as
  the 6.4.0 run; zero compiler errors.
- Lint after the runs: **18/18 gates, exit 0** (`AUDIT/brew641-lint.log`).
- All 49 warnings are `building for macOS-26.0, but linking with dylib …
  built for newer version 27.0`. They are recorded as AUD2-039, not as a pass.
- `xcrun swift` still reports 6.4.0, so the project's shipping toolchain is
  unchanged; the keg is used only by the explicit paths above.

An earlier attempt used `AUDIT/scratch/brew641` as the scratch path. The
`arch-path` and `swiftlint` gates then scanned SwiftPM's checkouts and generated
test entry points under that tree and failed. The scratch tree was removed and the
runs repeated under `.build/brew641`, which the gates exclude. No gate was
weakened, and no source was changed by this exercise.

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
- The delegated reviewer batches produced no findings because all eleven hit the
  credit limit first. No reviewer claim is treated as evidence; the four
  reviewer-derived records added here (AUD2-033…036) were re-derived in the tree
  by the main agent, and AUD2-032 came from the search for the brief.
- The brief itself has since been recovered from the operator. It is preserved
  verbatim at [AUDIT/audit-brief.md](AUDIT/audit-brief.md) and verified by hash;
  the earlier statement that it was unrecoverable described the state before that
  recovery, not the current one. Reconciliation added AUD2-037 and AUD2-038 and
  did not close any product record.
- The toolchain inventory made the same class of mistake: it searched Xcode's and
  CommandLineTools' toolchains only and reported that no Swift 6.4.1 compiler
  existed. Homebrew's keg-only `swift` 6.4.1 was installed all along. The negative
  finding is withdrawn; the 6.4.1 build and suite are recorded, and the 49 linker
  warnings they expose are filed as AUD2-039.
- A scratch build first lived under `AUDIT/scratch/brew641`, which made the
  `arch-path` and `swiftlint` gates scan SwiftPM checkouts and generated test
  entry points. The tree was removed and the runs repeated under `.build/brew641`.
  This was a self-created artifact and a wrong path choice, not a source defect;
  no gate was excluded to hide it.
- A reviewer lead alleging that `FleetModels.workspaceId` was a non-optional
  `String` while the plugin sends `null` is **disproven**: the Swift property is
  `String?` and the plugin writes `null` only for an unregistered workspace. It is
  recorded as a rejected lead, not filed as a defect.
- [Scope and tiers](AUDIT/scope-tiers.md) named the converter as
  `benchmark/prepare_qwen38.py`, but the converter is
  [tools/prepare_qwen38.py](tools/prepare_qwen38.py); `benchmark/` holds the
  suites and measure scripts. The four references were corrected in place. This
  was a documentation path error, not a scope or tier change: the converter
  remains Tier A.
- The statement that remote CI artifact contents had not been inspected is
  superseded: a fresh artifact was downloaded and shown to contain one bundle and
  27 files, which confirms AUD2-031 rather than closing it.

## Continuation boundaries

Continue from the recorded issue evidence, not from a claim that passing tests
means all bugs are fixed. Preserve uncommitted work and the invalid-evidence
history. Do not close a static claim without the required reproduction/review,
replace a disproven claim with its sibling, or treat absent evidence as a pass.

At this snapshot **the audit is not complete, and it cannot be certified complete
from inside the tree.** The blocks are external or unrun:

1. The brief is no longer missing: it is preserved at
   [AUDIT/audit-brief.md](AUDIT/audit-brief.md) and hash-verified. Reconciling
   against it exposed four mandated passes with no recorded work (AUD2-037, whose
   §5 facade hunt has since run) and an independent-host requirement the audit
   cannot satisfy alone (AUD2-038).
2. The eleven delegated Tier A reviewer batches all terminated on the operator's
   credit limit before emitting a finding, so no full L2/L3/L4 pass exists for any
   module. Four targeted findings were re-derived manually; that is not the same
   thing.
3. There is no second host for the Phase E verification (AUD2-038), so §12's
   completion condition cannot be met from this machine even if every other record
   were closed.

The Swift 6.4.1 question is no longer a blocker: Homebrew's keg-only 6.4.1 built
the package and passed the full serial suite (AUD2-022, DONE). It did surface a
new build-hygiene finding, AUD2-039.

The ledger stands at **36 non-terminal records** (19 OPEN, 2 START, 3 PROGRESS,
10 TEST, 2 AUDIT), four DONE records and one BLOCKED. The §5 facade hunt's two
findings are now DONE with remediation commits, so this milestone drains two
records; the run has still never reached Phase C, let alone Phase D, so no
convergence milestone is claimed and §8's invariant is not yet due.

Completed and evidenced here: recovery and hash verification of the original
brief, the debug and release builds plus the full serial suite under the required
Swift 6.4.1 compiler, the unmasked full serial suite under 6.4.0, the 6.4.0
release build, the 18-gate lint run, the corrected seven-object coverage export,
the CI artifact inspection, the ornith-8 golden check, four of the five
library-contract tests, the live CLI SIGINT probe, and the four brief-mandated
passes (AUD2-037). Still incomplete:
the module-by-module Tier A review,
clean-commit CI certification, fresh sanitizer runs on the final edits, the Python
dependency reconciliation, durable CI coverage, release-only/opt-in paths, and
remediation of every OPEN/START/PROGRESS record — including AUD2-035, which the
probe confirms but does not fix. This report is a faithful snapshot of that state,
not a Phase E certificate.

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
