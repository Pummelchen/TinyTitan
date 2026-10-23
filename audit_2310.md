# Pre-production audit — 23 October 2026

`[#AUD2-051 51/51 | done:36 open:14 blocked:1] DONE — the CI markdown-link step,
which had never run over these records, now reads code spans and fences as literal
text and resolves the 17 citations it used to fail on (AUD2-051); the coverage
artifact CI uploads is built from the manifest's seven test bundles and checked for
scope, so it no longer reads as package-wide while naming one bundle (AUD2-031);
every expansion-layout guard
now has a test that names it, proved by disabling the guard one at a time
(AUD2-013), and the rANS offset narrowing is a documented invariant rather than an
unexamined trust (AUD2-014); the persisted checkpoint access (AUD2-007), the
conflict refusal that names its own flags (AUD2-047), the calibration identity
binding (AUD2-005), the --plan conflict (AUD2-008), the host-KV window range
(AUD2-017), the shared disk budget (AUD2-003), the bounded payload read (AUD2-006)
and the fail-silent restore (AUD2-009) beneath it stand`

**Audit incomplete. Known defects remain. No product defect is formally closed.**
Two documentation records (AUD2-022, AUD2-032) are closed; AUD2-037 is closed as
a process record once its four passes were run; AUD2-040 to AUD2-044, the
keep-warm pair (AUD2-020, 021), AUD2-033, AUD2-026, AUD2-028, AUD2-046, AUD2-029,
the pending calibration/validation cluster (AUD2-005, 008, 017, 047), the shared
prompt-cache disk budget (AUD2-003), the two reader/restore corrections
(AUD2-006, 009), the persisted checkpoint access (AUD2-007), the two expansion
geometry records (AUD2-013, AUD2-014), the coverage-artifact scope record
(AUD2-031) and the audit's own markdown-link checker (AUD2-051) are closed with
remediation commits, and none is merged, released or certified on a clean final
revision. AUD2-048 (a
diagnostic asymmetry found while closing AUD2-007), AUD2-049 (the docs gate never
reads this audit's own ledger) and AUD2-050 (a type-level `#expect(throws:)`
cannot say which guard refused, so a negative test can pass for the wrong reason —
found twice inside AUD2-013's own suite and un-swept across the rest of `tests/`)
are opened in the same pass.

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
- Measured HEAD: `8acc6f8`, which carries the AUD2-013/014 expansion-test pins on
  top of the AUD2-007 persisted-access fix at `92da15a`, the
  AUD2-047 conflict diagnostic at `3191968`, the
  AUD2-003/006/009 disk-budget and reader fixes at `6130898`, the
  AUD2-005/008/017 calibration and validation fixes at `41d14ca`, the AUD2-029
  deletion-accounting fix at `7024d03`, the AUD2-046 indexing fix at `7740a42`, the
  AUD2-028 checkpoint validation at `c5e2e93`, the AUD2-026 sweep fix at
  `5deb70a`, the keep-warm fixes, the AUD2-033
  final-wave publish and its comment correction, the rANS cluster at `61de182`,
  the AUD2-044 fix and the AUD2-040 to AUD2-043 chain.
  Every fix recorded as DONE is committed; what remains uncommitted at this
  snapshot is this report and the ledger, not source. `9ca9942` is the exception
  disclosed in AUD2-017: it committed a test whose fix stayed in the working tree
  until `41d14ca`.
- **50 records: 34 DONE, 15 unresolved/non-blocked, 1 BLOCKED.**
- Exact statuses: **11 OPEN, 2 PROGRESS, 2 AUDIT, 34 DONE, 1 BLOCKED**.
- Recorded severities: **2 S0, 16 S1, 30 S2, 2 S3**. These are ledger
  classifications, not a claim that all 50 records are confirmed product bugs.
  Deviation: this line read "28 S2, 3 S3" at `615203a` and did not match the
  ledger it restates, which held 29 S2 and 2 S3 in those 49 records — AUD2-049
  is the finding that nothing checks it. Corrected here and re-derived from
  `AUDIT/ledger.json` rather than from the previous sentence.
- `OPEN` and `PROGRESS` remain unfinished. `AUDIT` retains a
  claim/disposition awaiting formal reconciliation. `BLOCKED` is not a passing
  verification.
- AUD2-015 and AUD2-016 were disproven as originally stated. AUD2-019, AUD2-022,
  AUD2-031, AUD2-037 and AUD2-038 concern verification, rather than independent
  runtime defects. AUD2-022 (Swift 6.4.1), AUD2-032 (brief recovery) and AUD2-037
  (the four brief-mandated passes) are DONE and close no product defect;
  AUD2-040 (stale diagnostic), AUD2-041 (`min_p` accepted and ignored), AUD2-042
  (five unreferenced declarations), AUD2-043 (an orphaned expert-I/O diagnostic)
  and AUD2-044 (late port validation) are DONE with remediation commits, as are
  the nine rANS records closed at `61de182`, AUD2-020, AUD2-021, AUD2-033,
  AUD2-026, AUD2-028, AUD2-046, AUD2-029, the calibration/validation cluster
  (AUD2-005, AUD2-008, AUD2-017), the shared prompt-cache disk budget
  (AUD2-003), the bounded turn-store read (AUD2-006), the fail-silent streamed
  restore (AUD2-009), the conflict diagnostic (AUD2-047), the persisted
  checkpoint access (AUD2-007) and the two expansion geometry records
  (AUD2-013, AUD2-014) — the last two closing on mutation proof rather than on a
  guard's existence. AUD2-048 (the turn store discards its own LRU touch failure)
  is opened as the S3 sibling of the AUD2-007 closure, and AUD2-050 (a
  type-level `#expect(throws:)` cannot name which guard refused) as the
  tree-wide sibling of the AUD2-013 one.
- AUD2-040 is `dc52729`, AUD2-041 is `9ca9942`, AUD2-042 is `ccfee98`, AUD2-043 is
  `c8d04c4`, AUD2-044 is `3ac4699`, AUD2-020 is `3842d8d`, AUD2-021 is `3098fdd`,
  AUD2-026 is `5deb70a`, AUD2-028 is `c5e2e93`, AUD2-046 is `7740a42`, AUD2-029 is
  `7024d03`, AUD2-005/008/017 are `41d14ca`, AUD2-003/006/009 are `6130898`,
  AUD2-007 is `92da15a`, AUD2-013/014 are `8acc6f8`,
  AUD2-033 is
  `71b40d7` (with the
  comment-only `d11be7c`) and the rANS cluster is
  `61de182`, all on the audit branch. Removal tickets
  TT-050 and TT-051 are filed in the project tracker (wiki commit `081335d`). No
  fix is represented here as merged, released, or certified on a clean final
  revision.

Sources are [baseline](AUDIT/baseline.md), [environment](AUDIT/environment.md),
[scope and tiers](AUDIT/scope-tiers.md), and
[tool-enforcement proofs](AUDIT/tool-coverage.md). Raw logs and generated coverage
artifacts named below are **local evidence under the gitignored `AUDIT/` tree**;
not all are tracked or available in a clean clone. Results and limitations are
therefore included inline. Every source and test change recorded as DONE is
committed on this branch; the only files that stay local are the raw logs named
above.
Source links identify files, not a promise that old ledger line numbers still
match after the fixes landed.

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
| §1 | Sanitizers run the suite (C ASan/UBSan, Swift TSan) | Partial — TSan is clean on the keep-warm paths (AUD2-020); no sanitizer run covers the whole final tree | AUD2-020, this report |
| §1b | One primary host recorded | Met | [environment](AUDIT/environment.md) |
| §1b, §11, §12 | One independent host runs the single final verification | **Not met** | AUD2-038 (BLOCKED) |
| §2 | Inventory, 2-hop dependency graph, trust boundaries, tier table | Met | [scope and tiers](AUDIT/scope-tiers.md) |
| §3 | Baseline with actual language mode, warnings, tests, lint, CVEs, secret scan | Partial — baseline exists and was re-measured; the 6.4.1 build adds 49 linker warnings (AUD2-039) | [baseline](AUDIT/baseline.md) |
| §4 L0–L4 | Repository, architecture, module, line and security passes at tier depth | Partial — rANS, checkpoint, server, CLI, PLE and Metal areas reviewed; no complete Tier A batch review | Issue record below |
| §4 L5 | Performance pass on Tier A and profiler-identified hot paths | Met — no N+1, unbounded collection, missing stream or blocking request-path call; no new measurement taken or claimed | AUD2-037 |
| §4 L6 | Coverage gaps, failure paths, tests that assert nothing, cross-project contracts | Partial — the coverage artifact is now package-scoped and scope-checked; contract tests partly run; Metal and 29 production files stay outside LLVM coverage | AUD2-031, this report |
| §4 L7 | Ops pass per deployable service | Met — health, signals, graceful shutdown, structured errors and rollback checked; the one startup-validation gap is fixed | AUD2-037, AUD2-044 |
| §5 | Facade hunt with a production-path facade as S0 | Partial — the hunt ran 2026-10-23; no production-path facade found, both outcomes now fixed and DONE | AUD2-037, AUD2-040, AUD2-041 |
| §6 | Unused code limited to Tier A and audit-orphaned code | Met for the sweep — five unreferenced declarations deprecated behind TT-050/TT-051, one orphaned diagnostic restored and live-verified | AUD2-042, AUD2-043 |
| §7 | Severity order S0 → S3, nothing dropped | Met in classification; remediation is not done | Ledger |
| §8 | Ledger schema and S0/S1 gates | Met for the closed records — both S0s and the rANS S1s ran every stage, and AUD2-020/021/033 each carry their own fix commit; the rANS fixes share one, a deviation recorded in the ledger notes | [ledger](AUDIT/ledger.md) |
| §8 | Non-terminal count strictly decreasing at every milestone | Met at this milestone (16 → 15), not across the run: the AUD2-007 pass moved it 15 → 16 because sibling discovery is expected to add records, and that reversal is disclosed in the closing summary rather than netted away | Ledger |
| §9 | Ledger is the single source; report generated from it | Met | [ledger](AUDIT/ledger.md) |
| §10 | One focused commit per S0/S1; audit branch only | Partial — `dc52729`, `9ca9942`, `ccfee98`, `c8d04c4`, `3ac4699`, `3842d8d`, `3098fdd`, `71b40d7`, `c5e2e93`, `7740a42`, `7024d03`, `41d14ca`, `6130898`, `92da15a`, `8acc6f8` and `61de182` are focused commits, but the rANS cluster merges nine interleaved records into one and `41d14ca`/`6130898`/`8acc6f8` each bundle S2 records, which §10 allows; `9ca9942` carries a test whose fix landed only in `41d14ca` (AUD2-017) | `git log` |
| §11 | Phase A inventory/baseline/environment | Met | Committed |
| §11 | Phase B L0–L7 loop | In progress — every pass now has recorded work | This report |
| §11 | Phase C drain to zero | In progress — 34 DONE, 15 non-terminal, 1 BLOCKED | Ledger |
| §11 | Phase D convergence | Not started — requires zero open | Ledger |
| §11, §12 | Phase E clean run on the independent host, open count zero | **Not met** | AUD2-038; 15 non-terminal records |
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

- [ ] Resolve the checkpoint records still open: crash leftovers and uncounted
  staging/backup directories (AUD2-001), corrupt metadata that is never
  reclaimed at startup (AUD2-002) and allocation before integrity
  verification (AUD2-004). The shared disk budget (AUD2-003), the bounded
  turn-store read (AUD2-006), the fail-silent streamed restore (AUD2-009), failed
  deletion accounting (AUD2-029) and the persisted eviction order (AUD2-007) are
  closed with commits on this branch.
- [ ] Sweep and close the S3 diagnostic asymmetry the AUD2-007 fix exposed in the
  turn store (AUD2-048): the same attribute write, reported by one store and
  discarded with `try?` by the other.
- [ ] Extend the `docs` gate's ledger-commit check to `AUDIT/ledger.json` and test
  it against a rotted pointer (AUD2-049): the audit's own evidence currently
  resolves nothing, while the gate's message says it resolved 222 references.
- [ ] Give the benchmark allocation-error cleanup path a test. The path itself is
  closed — AUD2-021 is DONE at `3098fdd`, with `keepWarmStopped` and a `defer` so
  every throw inside the cadence loop unwinds through a `stop()` — but its own
  ledger row discloses that no unit seam forces a command-buffer allocation
  failure, because `requireCommandBuffer` throws only when Metal returns nil and
  the production function takes the real queue. What is missing is an injected
  queue, not a fix; the current evidence is compile, review and the suite.
- [x] Complete before/after evidence for the expansion geometry records: AUD2-013
  and AUD2-014 now carry their `evidence-before`, `fix-summary` and
  `evidence-after` and close at `8acc6f8` on mutation proof — each of the eight
  layout guards disabled in turn, with the two tests that stayed green doing so
  because their fixture was refused by a different check. Independent (cold)
  review is still outstanding and belongs to the Phase E item below, not here.
  The calibration, CLI, snapshot and sizing patches in that group carry their own
  RED/GREEN runs (`41d14ca`, `6130898`, `3191968`).
- [ ] Sweep the tree-wide sibling of that defect (AUD2-050): 298 type-level
  `#expect(throws: <Type>.self)` refusals in 75 test files cannot name which guard
  fired, the form that let two expansion tests pass for the wrong reason.
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
  timing footers. *(Partial: the serial suite runs against a committed tree — most
  recently **7 bundles, 2,007 tests, exit 0** at `8acc6f8`, with the exact command,
  the `xcrun` Swift 6.4 (`swiftlang-6.4.0.34.1`) and `git rev-parse HEAD` stamped
  into `AUDIT/aud2-013-full-final.log`. What is still missing is the other half:
  the required-compiler **6.4.1 keg** runs were recorded on an earlier dirty tree,
  and a clean-checkout run on the final revision has not been made at all.)*
- [ ] Rerun relevant TSan and C ASan/UBSan suites on that final revision; older
  sanitizer results do not certify later edits.
- [ ] Reconcile Python 3.13/3.14/3.15 runtime verification with the pinned
  dependency stack; syntax checks and the focused M5 suite are insufficient.
- [ ] Inspect final-revision dependency/security/CodeQL evidence and secret scans,
  rather than assuming configured CI jobs succeeded.
- [x] Implement durable multi-bundle CI coverage export and scope
  validation. *(Done in `74b56e1`: `tools/coverage-export.sh` merges the seven
  manifest test bundles, refuses a bundle newer than the profile, and fails on a
  report whose file set any single bundle exceeds; the workflow uploads
  `.build/package-code-coverage.json`. Verified locally on a fresh coverage run
  and red-proved on the artifacts it catches. The step has **not** run in CI: the
  workflow triggers are `push: main` and `pull_request`, and this change lives on
  `audit/2026-10-23` only, so it is not yet a merged pipeline change and merging
  it stays the operator's call. The pre-fix remote artifact (one bundle / 27
  files) remains evidence of the defect.)*
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

**S2 · Tier A · DONE · TinyTitanLib**

- **Location:** [session loading](sources/TinyTitanLib/ServerModelSession+Loading.swift)
  (lines 17–45, 364–401).
- **Finding:** Both prompt stores are handed one `ServerPromptCacheStorageConfiguration`
  over one directory and each evicts only its own records, so each with the full
  `--prompt-cache-disk-mib` allowed the pair to occupy twice it. The residual the
  review added: with checkpoints disabled the checkpoint store is never
  constructed, so records left by an earlier enabled run are charged to nobody —
  a half-budget checkpoint cache beside a full-budget turn store reaches 1.5x.
- **Current evidence:** The split (`promptCachePerStoreDiskLimit`) gives each
  store half while both exist; `reclaimStaleCheckpoints` deletes what an earlier
  enabled run left and returns the bytes that refused deletion, which are then
  subtracted from the turn store's share; and with checkpoints on the store
  indexes at startup, so a restart over its half is evicted before the turn
  store fills the other. RED is compile-only and disclosed as such — the tests
  do not build against the pre-fix tree because the seam is new (exit 1,
  `AUDIT/aud2-003-red.log`) — so the bound rests on the arithmetic and on the
  four helper-level tests (exit 0, `AUDIT/aud2-003-green.log`), not on an
  occupancy reproduction across a restart.
- **Limits:** `load`'s startup indexing needs a model to reach and has no unit
  seam; it is recorded as reviewed and compiled. AUD2-004 (verify before
  materializing) and AUD2-001/002 (crash-gap and quarantine reachability) stay
  open and are not closed by this budget work.
- **Commit:** `6130898`. Not merged, released or verified on an independent host.

### AUD2-004 — Large payloads are allocated before integrity verification

**S2 · Tier A · OPEN · TinyTitanLib**

- **Location:** [checkpoint disk I/O](sources/TinyTitanLib/PromptCheckpointStore+Disk.swift)
  and [turn-store disk I/O](sources/TinyTitanLib/ServerPromptStateStore+Disk.swift).
- **Finding:** Reading `expectedBytes + 1` into Data before hashing can materialize
  multi-GiB untrusted payloads. A read cap does not establish integrity before a
  large allocation; the turn-store comment claimed it did.
- **Current evidence:** Confirmed by full-file review, without a large-allocation
  runtime probe. The numeric metadata guard in AUD2-028 does not fix this issue.
  The overstating comment was corrected in `6130898` alongside AUD2-006: the cap
  bounds the read to what the metadata declares, and the declared bytes are
  still materialized because the caller needs the payload.
- **Next:** Design a bounded integrity-verification path for both readers and
  prove corrupt/oversized input is refused without the large materialization.
  This needs a streaming loader: `readPayload` returns the bytes it verified,
  so verification cannot precede materialization through the current API.

### AUD2-005 — Calibration applicability checks are incomplete

**S2 · Tier A · DONE · TinyTitanLib**

- **Location:** [calibration profile](sources/TinyTitanLib/CalibrationProfile.swift)
  (lines 215–256), [server arguments](sources/TinyTitanServer/Core/ServerArguments.swift)
  (lines 418–440) and [CLI arguments](sources/TinyTitanCLI/Args.swift)
  (lines 428–457).
- **Finding:** `application(for:contextTokens:modelID:...)` compared machine
  identifier, memory and context only, and the server path passed no modelID at
  all: a profile measured on another macOS release, or for another family or
  weight width, was applied to the install in front of it.
- **Current evidence:** `osVersion` is now a refusal and `family`/`weightBits`
  are compared beside `modelID`; the CLI passes the identity it already peeked
  and the server peeks the manifest, skipping the binding only when no `--model`
  names an install (a catalog launch has no single model to bind to). RED: with
  the three source files reverted, the foreign-OS profile and all three
  foreign-identity profiles parse to a usable `ServerArguments` instead of
  throwing (7 tests, 4 issues, exit 1, `AUDIT/aud2-005-red.log`).
- **Sibling-arm disclosure:** the CLI arm's first test varied `modelID` only and
  **passed against the unfixed tree**, because `modelID` was already checked —
  an invalid RED, replaced by a parameterized test over `modelID`, `family` and
  `weightBits`, which then failed for exactly the two new arms (2 issues, exit 1,
  `AUDIT/aud2-005-cli-red.log`) and passes with the fix (17 tests, exit 0,
  `AUDIT/aud2-005-cli-green.log`). The server tests now build a toy install the
  real `ManifestReader` accepts instead of pointing at `/tmp/model`.
- **Commit:** `41d14ca`. Not merged, released or verified on an independent host.

### AUD2-006 — Turn-store payload reader was unbounded until EOF

**S2 · Tier A · DONE · TinyTitanLib**

- **Location:** [turn-store disk I/O](sources/TinyTitanLib/ServerPromptStateStore+Disk.swift)
  (lines 12–40).
- **Finding:** `readPayloadVerifying` looped `handle.read(upToCount: 1 MiB)` to
  EOF, appended everything into one `Data` and hashed it, and only then compared
  the length against the declared `payloadBytes` — a payload file swapped for a
  larger one after indexing was fully read and allocated before being refused.
- **Current evidence:** The reader now asks the handle for `expected + 1` bytes
  and stops there; further reads happen only to fill a short file, so over-sized
  input is refused on the 1-byte overshoot. The comment written with that change
  claimed the bytes were refused before being allocated, which AUD2-004 correctly
  called false: the declared bytes are still materialized because the caller needs
  the payload, and the cap bounds the read to what the metadata declares rather
  than to what the file is. The comment now says that, and the two issues are
  kept separate: this row is the read bound, AUD2-004 is verify-before-materialize
  and stays open.
- **Disclosed as a pin, not a RED:** the new test
  `swappedOversizedDiskPayloadIsRefusedAtTheDeclaredSize` indexes a valid record,
  grows its payload to a 16 MiB sparse file and asserts `loadSnapshot` fails
  closed and the record is removed (11 turn-store tests, exit 0,
  `AUDIT/aud2-006-green.log`) — but the pre-fix reader also refused that file,
  having first read all of it. The bound itself rests on the read call taking
  `expected + 1`, not on an assertion; memory is not observable from a unit test
  and no model run applies.
- **Commit:** `6130898`. Not merged, released or verified on an independent host.

### AUD2-007 — Restart eviction does not preserve actual access order

**S2 · Tier A · DONE · TinyTitanLib**

- **Location:** [checkpoint store](sources/TinyTitanLib/PromptCheckpointStore.swift#L352-L377),
  with the restart index it feeds in
  [`PromptCheckpointStore+Maintenance.swift`](sources/TinyTitanLib/PromptCheckpointStore+Maintenance.swift#L6).
- **Finding:** The listing-order and unindexed-`removeAll` halves of this row
  closed at `7740a42` (AUD2-046). What stayed was the access itself: `touchDisk`
  reassigned `record.modificationDate` inside the actor's state and never wrote
  to the filesystem, while `indexDiskIfNeeded` rebuilds a restarted store's LRU
  from the directories' modification dates. Saving A then B, loading A,
  restarting and filling the budget evicted A — the record that had just been
  read. The existing restart test backdated its directories by hand, so it
  proved the sort and never the touch.
- **Current evidence:** `touchDisk` now persists the date with
  `FileManager.setAttributes([.modificationDate:])` on the record's own
  directory, the way the turn store does. Only `load` reaches it, and the
  multi-prefix restore path
  ([`ServerModelSession+PromptCache.swift:333`](sources/TinyTitanLib/ServerModelSession+PromptCache.swift#L333))
  calls `load`, so a served cache hit is the access that moves the record: the
  defect was reachable in production, not latent. A failed attribute write keeps
  the older date — ordering, not data — and is reported through
  `ServerLog.diagnostic` rather than swallowed. RED: the new
  `aLoadedRecordKeepsItsPlaceAcrossARestart` reports 2 issues on the unfixed
  tree (34 tests, exit 1, `AUDIT/aud2-007-red.log`) — the loaded record was gone
  and the untouched one survived. GREEN: the same 34 tests pass, exit 0
  (`AUDIT/aud2-007-green.log`). The fixture pins both directories in the past
  (epoch 0 and epoch 1) so the outcome rests on the access alone and not on
  timestamp resolution.
- **Sibling asymmetry found while closing this row:** the turn store persists
  the same touch under `try?`, so its failure is silently discarded. Recorded
  as **AUD2-048**, open.
- **Commit:** `92da15a`. Not merged, released or verified on an independent host.

### AUD2-008 — Calibration plan mode silently ignores conflicting options

**S2 · Tier A · DONE · TinyTitanCLI**

- **Location:** [calibration command](sources/TinyTitanCLI/CalibrationCommand.swift)
  (lines 270–276).
- **Finding:** `run` checked `--check`, then `--calibrate`, then `--plan`, so
  `calibrate --plan --check <path>` printed the plan and discarded the check
  file, and `--plan --model <path>` ignored the model — with no diagnostic for
  either.
- **Current evidence:** The `--plan` branch refuses `--check` and `--model` with
  `CalibrationCommandError.conflictingInput` before printing. Precedence between
  the existing modes is unchanged; only the silently-ignored combination became an
  error. RED: the two new expectations report "an error was expected but none was
  thrown" (16 tests, 2 issues, exit 1, `AUDIT/aud2-008-red.log`); with the fix the
  same 16 tests pass (`AUDIT/aud2-008-green.log`).
- **Sibling defect found while closing this row:** all three conflict sites shared
  one message written for `--model` with `--check`, so the new `--plan` refusals
  told the user about a flag they had not passed. Recorded as **AUD2-047** and
  fixed in the same pass.
- **Commit:** `41d14ca`. Not merged, released or verified on an independent host.

### AUD2-009 — Streamed snapshot restoration can fail silently

**S2 · Tier A · DONE · TinyTitan**

- **Location:** [streamed restore](sources/TinyTitan/Runtime/KVCache/KVCacheManager+HostStreamingPrefill.swift)
  (lines 410–424) and [snapshot caller](sources/TinyTitan/Runtime/KVCache/KVCacheManager+Snapshot.swift#L96).
- **Finding:** `restoreStreamedSnapshot` returned silently when
  `hostK[layer]`/`hostV[layer]` was missing or `length` exceeded
  `hostArenaLengths[layer]`. The caller in `KVCacheManager+Snapshot` guards the
  segment length, so the originally reported condition is unreachable through it;
  the hazard is that a restore which copied nothing can still be reported as a
  successful restore, with the window map left describing rows the arena does not
  hold.
- **Current evidence:** The function now throws
  `InferenceStateSnapshotError.invalidLayout`, its sole caller propagates with
  `try`, and the comment records why the guard is a throw rather than a return: it
  is the backstop that keeps the caller's guard from becoming load-bearing.
  Behavior on every reachable path is unchanged.
- **Disclosed as a gap, not coverage:** there is no unit seam — reaching it needs
  a live `KVCacheManager` with a host arena, which a model-free test cannot
  construct, and the only reachable caller sits behind the guard this replaces.
  The signature change is verified by the build and by the full serial suite
  (7 bundles, 2,002 tests, exit 0, `AUDIT/aud2-pending-full.log`), not by a test
  that asserts the throw.
- **Commit:** `6130898`. Not merged, released or verified on an independent host.

### AUD2-010 — Failed rANS length reads can skip decoding or undersize staging

**S1 · Tier A · DONE · TinyTitan**

- **Location:** [rANS decoder](sources/TinyTitan/Infrastructure/Streaming/RANSDecode.swift)
  and [rANS streamer](sources/TinyTitan/Infrastructure/Streaming/PreadExpertStreamer+RANS.swift).
- **Finding:** A failed length-table `pread` could leave empty plans and zero
  payload size, silently preserving stale weights and underestimating the Metal
  staging buffer needed for the subsequent payload read.
- **Current evidence:** A failed length-table `pread` is collected as a failure and
  thrown at layer open, so an empty plan with a zero payload size can no longer
  reach the wave loop. The RED was demonstrated on the fix itself — restoring the
  old `else { return }` made `refusesTruncatedPayload` fail with "an error was
  expected but none was thrown" (`AUDIT/aud2-010-red.log`, exit 1) — and the
  committed file restored before the paired GREEN (exit 0,
  `AUDIT/aud2-010-green.log`). The cold re-read at the AUDIT stage confirmed writer,
  reader and kernel agree on the 8-byte header and that the overlap check compares
  adjacent runs in start order.
- **Next:** Preserve this evidence in final cold review and certification; no live
  model run exercises the guard, so an untrusted-sidecar run remains outstanding.

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

**S2 · Tier A · DONE · TinyTitan**

- **Location:** [packed-weight expansion](sources/TinyTitan/Infrastructure/Streaming/TwoBitExpand.swift#L73-L137),
  with the one map builder that feeds it
  ([`Model+LayerStreamers.swift:354-379`](sources/TinyTitan/Runtime/Inference/Model+LayerStreamers.swift#L354))
  and the [suite that pins it](tests/TinyTitan/Infrastructure/Streaming/TwoBitExpertReadTests.swift#L182).
- **Finding:** An invalid target order or range can overwrite unread packed
  source bytes or exceed the slot during in-place expansion.
- **Current evidence:** `validateExpansionLayout` went in at `61de182` with the
  rANS geometry work and runs once per expert while the layer's maps are built,
  before any expert is streamed, over the manifest's own sub-tensor offsets and
  sizes — untrusted file data. It refuses a negative offset or size, a target
  before its source, a region outside the slot, a 3-bit size that is not a whole
  number of groups, a rescaled size that is not a whole number of bf16 values, a
  target size that does not match the transform, a write into a lower tensor's
  unread source, and two tensors sharing a target region.
  What this row still lacked was proof that any of those eight refusals is
  *tested*. The suite at `615203a` held six tests, and reading it against the
  validator showed three guards with no case at all — a negative offset or size,
  an odd rescaled size, a 3-bit leftover group. Pins for the first two were
  written before the sweep (eight tests), then every guard was disabled in turn
  and `ExpansionLayoutValidationTests` re-run
  (`AUDIT/aud2-013-mutation.log`, `AUDIT/agent-notes/mutate-sweep-stdout.log`;
  drivers `AUDIT/agent-notes/mutate-expansion-guards.sh` and `mutate-one.py`).
  **Five of the eight failed exactly the test named for them** — M1 negative
  offset/size, M2 target before source, M5 odd rescale, M6 size match, M7
  unread-source overlap. Three proved nothing, and none of the three was a guard
  gap:
  - **M4** (3-bit whole group) left all eight green, because the tree's only
    3-bit negative case (source size 4, target size 6) is equally refused by the
    size-match guard, so nothing distinguished the two.
  - **M8** (shared target region) also left every test green, because
    `tensorsThatShareATargetRegionAreRefused`'s second tensor sat at target
    offset 4 with source offset 8 and the bounds guard at `:77-80` refused the
    map on its `targetOffset >= sourceOffset` clause — long before the overlap
    check it is named for. That guard is one `else` over four clauses sharing one
    message ("violates the slot bounds"), which is exactly why
    `#expect(throws: ModelError.self)` could not express that the test had been
    passing for the wrong reason.
  - **M3** was an invalid mutation, not a gap: as first written it *tightened*
    the bound to `<= -1`, which no layout can satisfy, so the run failed
    `sizesThatDoNotMatchTheTransformAreRefused` and the positive
    `theConverterLayoutIsValid` (3 issues) instead of the bounds test. **M3b**
    relaxes the bound instead and fails exactly both cases of
    `aRegionOutsideTheSlotIsRefused` (2 issues,
    `AUDIT/mutate-aud2-013-m3b-full.log`).
  Added `aThreeBitTensorWithALeftoverGroupIsRefused` (7 source bytes are two
  whole groups, so 8 target bytes satisfies every guard except the group check),
  rebuilt the shared-target fixture so both targets sit at or beyond their own
  source while the write regions overlap, and routed both through a
  `refusalDetail` helper that asserts the `indexCorrupt` detail names the guard.
  Re-running each now fails exactly the one test that names it — 1 issue in each
  run (`AUDIT/mutate-aud2-013-m4-recheck-full.log`,
  `AUDIT/mutate-aud2-013-m8-recheck-full.log`), not a broadening failure. The M1
  and M5 pins were found by reading the suite, not by mutation: the sweep only
  confirms them, and what mutation alone revealed is M4 and M8.
  `TwoBitExpand.swift` is byte-identical to HEAD after the sweep
  (`git diff --exit-code` on the file, clean), and the sweep's own log records
  the restore after every mutation.
- **Verification:** `ExpansionLayoutValidationTests` and `TwoBitExpertReadTests`
  together — 9 tests in 2 suites pass on the fixed tree
  (`AUDIT/aud2-013-green-9.log`); the three streaming suites together
  26 tests in 3 suites (`AUDIT/aud2-013-streaming-suite.log`); the full serial
  suite re-run on the commit this record names —
  `TMPDIR="$PWD/AUDIT/scratch" env -u TINYTITAN_LIBRARY_CONTRACT_MODEL xcrun swift
  test --no-parallel`, Xcode 27 Swift `swiftlang-6.4.0.34.1`, **7 bundles, 2,007
  tests, all passed, `FULL_EXIT=0`**, command and `git rev-parse HEAD` stamped in
  the log header (`AUDIT/aud2-013-full-final.log`; the earlier
  `AUDIT/aud2-013-full.log`/`.exit` hold the same tree before the commit) — and all
  18 lint gates exit 0 on the same tree (`AUDIT/aud2-013-lint-final.log`). No
  production behaviour changed in this record, so no golden-baseline run: the
  change is test-side plus one doc comment.
- **Commit:** `8acc6f8`. Not merged, released or verified on an independent host.

### AUD2-014 — Expansion group sizes and offset narrowing need separate proof

**S2 · Tier A · DONE · TinyTitan**

- **Location:** [expansion guards](sources/TinyTitan/Infrastructure/Streaming/TwoBitExpand.swift#L87-L105),
  [descriptor writes](sources/TinyTitan/Infrastructure/Streaming/PreadExpertStreamer+RANS.swift#L115-L124),
  [staging geometry](sources/TinyTitan/Infrastructure/Streaming/RANSDecode.swift#L262-L279).
- **Finding:** The recorded group contains silent three-bit trailing-byte loss,
  two-bit even-size assumptions, UInt64-to-UInt32 output-offset narrowing, and a
  dead zero assignment to payloadOffset.
- **Current evidence, sub-claim by sub-claim:**
  - **(a) 3-bit trailing bytes.** `expand3BitReversed` decodes
    `sourceSize / 3` groups, so a run with a leftover byte would leave it
    undecoded in the slot. The whole-group guard refuses such a map at load, and
    the mutation above is what shows a test now distinguishes that guard from the
    size-match guard. The truncation is unreachable, not merely unlikely.
  - **(b) 2-bit even sizes.** A 2-bit expansion needs no evenness of its own
    (`targetSize == sourceSize * 2` is exact for any size, and the size-match
    guard at `:105` enforces it — caught by M6); the evenness that does matter is
    the rescaled bf16 block, where `rescaleBF16InPlace` divides `bytes / 2`
    halves and an odd size would leave the last byte unscaled. Guarded at `:97`
    and pinned by `anOddRescaledTensorIsRefused` (caught by M5).
  - **(c) UInt64→UInt32 narrowing.** Two sites: the staging base
    `UInt32(position * resources.payloadStride)` (`:115`) and the slot base
    `UInt32(offsets[position])` (`:118`). The stride is
    `max(1, maximumPayload)` where every expert's run is clamped inside the
    payload file by `payloadEnd <= payloadSize` at decode
    ([`ExpertRANSIndex.swift:229-234`](sources/TinyTitanFormat/ExpertRANSIndex.swift#L229)),
    and the staging buffer is allocated at `stride * maximumWave` with
    `maximumWave = 8`, so overflowing a 32-bit base needs one expert's compressed
    run above 4 GiB / 8 = 512 MiB — a multi-gigabyte layer sidecar — and Swift's
    `UInt32` arithmetic traps on overflow rather than wrapping, so even that
    fails as a clean crash at layer open instead of a wrong descriptor.
    `offsets` is `[UInt64]` because Metal buffer offsets are, and every
    construction site in this tree gives zero: each slot is its own
    `makeBuffer(bytesNoCopy:)` with `bufferOffsets.append(0)`
    ([`PreadExpertStreamer.swift:340-350`](sources/TinyTitan/Infrastructure/Streaming/PreadExpertStreamer.swift#L340)),
    and the ring's speculative path passes
    `offsets: [UInt64](repeating: 0, count:)`
    ([`PreadExpertStreamer+Prefetch.swift:46`](sources/TinyTitan/Infrastructure/Streaming/PreadExpertStreamer+Prefetch.swift#L46)).
    No speculative guard was added — a `stride * 8 > UInt32.max` check would
    police a state the payload-file clamp already prevents, and the repo rule is
    not to validate scenarios that cannot happen. The constraint a future
    shared-buffer caller would break is now stated on `decodeRANSInto`.
    **Limitation, stated plainly:** this bound is arithmetic over the current
    format, not an empirical test — no test builds a 512 MiB expert run, because
    that needs a file of that size.
  - **(d) dead `payloadOffset = 0`.** Already gone: `git log -S` shows it
    introduced at `d651fd9` and removed at `61de182`, in the same hunk that
    replaced the fixed-width `region[index * 8 + n]` writes with
    `descriptorWords`-based ones. No `payloadOffset = 0` remains under `sources/`.
- **Verification:** the streaming suites and full serial suite named under
  AUD2-013 cover the guard halves; the narrowing half is a documented invariant
  plus this analysis, and the only production edit in this record is that
  comment, which cannot change numerics — so, again, no golden-baseline run.
- **Commit:** `8acc6f8`. Not merged, released or verified on an independent host.

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

**S2 · Tier A · DONE · TinyTitanServer**

- **Location:** [server arguments](sources/TinyTitanServer/Core/ServerArguments.swift)
  (lines 296–304) and [runtime configuration](sources/TinyTitan/Runtime/Configuration/RuntimeConfiguration.swift)
  (lines 420–428).
- **Finding:** `--host-kv-window` accepted any positive `Int` while every other
  sizing flag names its range, so `999999999` parsed, was stored, and failed only
  when the per-layer, per-slot arena was allocated — after the model had been
  opened. `RuntimeConfiguration` applied the same `> 0` test.
- **Current evidence:** Both now require 1 through
  `RuntimeConfiguration.maximumContextTokens`: a window larger than the largest
  context the engine serves cannot hold a meaningful selection, and the error text
  names the bound it enforces. RED: with the two source files reverted,
  `parsesHostKVWindow` records 2 issues — the parse and the `RuntimeConfiguration`
  init both accept `maximumContextTokens + 1` (exit 1, `AUDIT/aud2-017-red.log`);
  with the fix it passes (exit 0, `AUDIT/aud2-017-green.log`).
- **Deviation disclosed:** those expectations were committed in `9ca9942` while the
  source fix stayed in the working tree, so that test failed on every commit from
  `9ca9942` through `7024d03`, and the full-suite runs recorded against those
  commits passed only because of the uncommitted fix. `41d14ca` lands the fix, and
  the suite was re-run on the committed tree.
- **Limits:** the CLI arm does not carry this flag (`Args.swift` has no
  `--host-kv-window`), so the two validated surfaces are the server parse and the
  programmatic initializer; the bound itself is a sizing ceiling, not an overflow
  guard.
- **Commit:** `41d14ca`. Not merged, released or verified on an independent host.

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

**S1 · Tier A · DONE · TinyTitan/TinyTitanBench**

- **Location:** [runtime runner](sources/TinyTitan/Infrastructure/Metal/GPUKeepWarmRunner.swift)
  and [benchmark worker](sources/TinyTitanBench/GPUKeepWarmStream.swift).
- **Finding:** An ordinary Bool was written by the caller and read by the worker
  without synchronization; an unchecked invariant did not make this defined.
- **Resolution:** Both types hold `running` as an `Atomic<Bool>` (Synchronization),
  stored with `.releasing` in `start()`/`stop()` and loaded with `.acquiring` by
  the worker loop. The stop/join semaphore still publishes `dispatches`; the
  cadence structure is unchanged and no timing claim is made.
- **Evidence:** A running TSan reproduction reported the race; an earlier
  compilation failure was not RED evidence. Fresh sanitizer run on the final
  tree: `swift test --sanitize=thread --no-parallel --filter GPUKeepWarmRunnerTests`
  exited 0 with **3 tests passed and 0 ThreadSanitizer warnings**
  (`AUDIT/aud2-020-tsan.log`); the only later change to the file was a comment.
  An independent reviewer read both types and found no remaining unsynchronized
  access: `startedAt` is written before the `serial.async` that is the worker's
  release edge and read only after the semaphore join, and `dispatches` is
  published by the semaphore, not the atomic, so `.releasing`/`.acquiring` is
  stronger than the kill flag needs. Two latent, unreachable paths are recorded
  rather than fixed: `stop()` without `start()`, or a second `stop()`, would block
  on a drained semaphore, and a runner dropped without `stop()` is retained by
  its own `guard let self` loop; the only production call site stops exactly once.
  The three lifecycle tests can fail on a missing join but cannot fail if the
  atomic store were dropped -- they pin the join, TSan pins the ordering. No
  performance claim is made.
- **Commit:** `3842d8d` on `audit/2026-10-23`; the TSan run compiled its parent.

### AUD2-021 — Benchmark allocation failure leaves keep-warm work running

**S1 · Tier A · DONE · TinyTitanBench**

- **Location:** [Metal head benchmarks](sources/TinyTitanBench/MetalHeadBenchmarks.swift).
- **Finding:** Worker start precedes throwing command-buffer/encoder allocations;
  joined stop exists only on the success path. The running worker retains itself.
- **Resolution:** `runHeadCadence` sets a `keepWarmStopped` flag after
  `keepWarm?.start()` and a `defer` stops the worker when it is still set, so
  every throw inside the loop (`requireCommandBuffer`, a nil compute encoder)
  unwinds through a stop. The success path stops first and then sets the flag, so
  a stop happens exactly once.
- **Evidence:** No unit seam exists to force a command-buffer allocation failure:
  `requireCommandBuffer` throws only when Metal returns nil, which needs an
  injected queue and the production function takes the real one; no test was
  written and none is claimed. Evidence is compile, review and the suite: the
  TSan build compiled the fixed file, the full serial suite passed on the same
  tree, and an independent reviewer confirmed nothing throws between the
  success-path stop and the flag assignment, so a double stop is impossible. The
  bench executable is outside `TinyTitanLib`'s closure, so its existing prints do
  not reach the stdout-clean gate.
- **Commit:** `3098fdd` on `audit/2026-10-23`.

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

**S2 · Tier C · DONE · benchmark**

- **Location:** [M5 sweep](benchmark/tinytitan_m5_sweep.py) and
  [M5 tests](benchmark/test_tinytitan_m5_sweep.py).
- **Finding:** Taking only the temporary wrapper's filename abandoned the open
  handle. Python 3.14/3.15 emitted 12 ResourceWarnings each despite the original
  23-test suites passing; this proved implicit cleanup, not a permanent leak.
- **Current evidence:** One added test retains the real wrapper and failed on
  `handle.closed` before the fix. Writing inside its context manager gives
  explicit closure. All 24 tests passed on each of Python 3.13.14, 3.14.9 and
  3.15.0, exit 0 with no ResourceWarnings; sibling sites were reviewed. The
  Python gate and all 18 lint gates passed on the same tree.
- **Commit:** `5deb70a`. Not merged, released or verified on an independent host.

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

**S1 · Tier A · DONE · TinyTitanLib**

- **Location:** [checkpoint store](sources/TinyTitanLib/PromptCheckpointStore.swift),
  [checkpoint disk I/O](sources/TinyTitanLib/PromptCheckpointStore+Disk.swift)
  and [checkpoint tests](tests/TinyTitanLib/PromptCheckpointStoreTests.swift).
- **Finding:** Decoded payload/auxiliary counts reached signed accounting before
  range validation, permitting negative/over-cap values and overflow. A
  directory whose metadata could not be decoded was treated as absent, so it
  surfaced as `missing` and its bytes stayed uncounted and un-evictable.
- **Current evidence:** `DiskMetadata.validate()` bounds payloads to 1…4 GiB and
  auxiliary data to 0…1 GiB, and `save`, `load`, restart indexing and the
  filesystem lookup all call it before accounting or reading a payload. A record
  that cannot be read, decoded or validated is quarantined: the directory is
  deleted, the key logged, and `corrupt` thrown. Restart indexing orders by
  directory modification date and evicts after indexing; `removeAll` indexes
  first. RED was captured by mutation on the final tree (quarantine and sort
  removed: 3 tests, 5 issues, exit 1, `AUDIT/aud2-028-red-mutation.log`; index
  validation removed: the new `metadata(for:)` assertion fails,
  `AUDIT/aud2-028-red-index-validate.log`). The restored tree passes 28 tests,
  exit 0 (`AUDIT/aud2-028-green-final.log`), and six lint gates pass on it.
- **Reviewed limits:** The independent review found two indexing gaps — a record
  loaded before the first index is evicted first, and `diskIndexed` latches done
  when the listing fails — filed as AUD2-046. Its two byte-count asymmetries
  (`save` accepts a zero-byte payload that `validate` rejects; `validate` accepts
  `auxiliaryBytes == 0` that `readPayload` rejects) are unreachable through
  capture and are recorded, not changed. This commit does not close AUD2-001,
  AUD2-002 or AUD2-007.
- **Commit:** `c5e2e93`. Not merged, released or verified on an independent host.

### AUD2-029 — Failed checkpoint removal drops real bytes from accounting

**S2 · Tier A · DONE · TinyTitanLib**

- **Location:** [checkpoint maintenance](sources/TinyTitanLib/PromptCheckpointStore+Maintenance.swift),
  `evictDiskIfNeeded`, `remove` and `removeAll`.
- **Finding:** All three paths dropped the index entry and subtracted its bytes
  before the filesystem removal succeeded, and each ignored the removal error.
  `evictDiskIfNeeded` also removed the oldest key from the LRU either way, so a
  directory it could not unlink left the budget under-reporting occupied disk
  with nothing left to retry; `removeAll` replaced the whole state, so a
  directory that refused deletion vanished from both the index and the budget.
- **Current evidence:** Reproduced with a directory marked `UF_IMMUTABLE`
  (`FileManager.setAttributes([.immutable: true], …)`), which makes
  `removeItem` fail `EPERM` on any account — independent of runner permissions
  or Full Disk Access. The first RED attempt was **invalid and is disclosed**:
  `aFailedRemovalKeepsTheRecordCharged` read `contains(key)` before
  `diskBytes`, and `contains` re-indexes the directory, restoring the bytes and
  passing against unfixed code. Reordered so the accounting is read first, the
  three tests then failed on the unfixed tree for exactly the recorded reasons
  (3 tests, 6 issues, exit 1, `AUDIT/aud2-029-red.log`): bytes dropped after a
  refused unlink, an evicted-but-undeleted record uncharged, and a survivor
  lost from `removeAll`. `evictDiskIfNeeded` now stops the loop and restores the
  record instead of rotating past it; `remove` returns before touching the
  index; `removeAll` rebuilds state from the directories that remain.
- **Evidence after:** The checkpoint suite passes 33 tests
  (`AUDIT/aud2-029-green.log`); the full serial suite passes 7 bundles, 1,998
  tests, exit 0 (`AUDIT/aud2-029-full.log`) and all 18 lint gates exit 0
  (`AUDIT/aud2-029-lint.log`). The fix needed the AUD2-046 split to be
  extended: the three methods moved by code motion into
  `PromptCheckpointStore+Maintenance.swift`, with `diskIndexed` widened from
  `private` to internal and the split recorded in
  [repository layout](docs/repository-layout.md).
- **Commit:** `7024d03`. Not merged, released or verified on an independent host.

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

**S1 · Tier A · DONE · audit evidence/coverage**

- **Location:** [CI workflow](.github/workflows/ci.yml) (lines 1027–1060),
  [coverage export script](tools/coverage-export.sh) and
  [coverage evidence](AUDIT/baseline.md).
- **Finding:** After seven bundles passed, SwiftPM's JSON contained only 27 files
  associated with ContinuityCore, including tests/generated code. Its 96.22%
  cannot be reported as package or production coverage. CI checked artifact
  presence, not represented targets, and uploaded exactly that file:
  `path: .build/debug/codecov/*.json` under the name `swift-code-coverage`.
- **Fix:** `tools/coverage-export.sh` builds the report itself. Its object list
  comes from `swift package dump-package` rather than a glob of the build
  directory — this tree carries ten `.xctest` bundles for seven declared test
  targets, so a glob folds removed products into the number — and it refuses a
  bundle newer than `default.profdata`, because llvm-cov drops such an object
  silently and still exits with a report that parses. Two checks then guard the
  two ways the artifact can lie: **shape** (a report mapping no file, or any file
  outside `sources/`, fails) and **scope** (every bundle is exported alone and the
  merged file set must strictly contain each single-bundle set, then be strictly
  larger than the largest one). No constant threshold could do the scope check
  honestly: one bundle here maps 323 of the 397 merged files, so 397 is only
  whole next to the per-bundle numbers. CI runs the script between the suite and
  anything that rebuilds, and uploads `.build/package-code-coverage.json`.
  The operator's 2026-10-11 call stands: nothing asserts a percentage, so the
  artifact stays a signal; what is now checked is that it describes the package.
- **Evidence:** A fresh coverage run at `a9f680a` (`TEST_EXIT=0`, 2,007 tests in
  seven bundles: 956/427/168/189/106/63/98) exported `EXPORT_EXIT=0` with
  **397 source files, 51,762/64,190 lines = 80.64% of instrumented lines**,
  per-bundle 19/8/247/41/52/323/313 and `report: 397 source files, 74 more than
  the largest single bundle` (`AUDIT/aud2-031-coverage-run.log`, reproduced by
  `AUDIT/aud2-031-export-rerun.log`). Both red paths were proved on the artifacts
  the check exists to catch (`AUDIT/aud2-031-red-checks.log`, clean tree at
  `74b56e1`): the preserved SwiftPM JSON fails shape with *maps 8 file(s) outside
  sources/*, and a real single-bundle export — which **passes** shape at 323 files
  and 42,546/53,090 = 80.14%, a plausible-looking percentage — fails scope with
  *TinyTitanFleetTests maps 8 source file(s) the report does not*.
  The first full gate chain after the change failed on my own regression
  (an unguarded `"${OBJECTS[@]:1}"` slice, `AUDIT/aud2-031-lint.log`
  `LINT_EXIT=1`); after the fix all 18 gates pass, `LINT_EXIT=0`
  (`AUDIT/aud2-031-lint2.log`).
- **Not checked, stated:** the new step has not executed in CI. The workflow
  triggers are `push: branches: main` and `pull_request`, so the first CI run of
  this step will be the next PR, and the change is not merged: it lives on
  `audit/2026-10-23`, so the pipeline edit is still the operator's call at merge
  time. The downloaded pre-fix artifact from run
  `39974492789` is evidence of the defect, not of the fix. And 80.64% over 397
  files is not whole-repository coverage: 29 of 426 non-metal production files sit
  outside the profile — the whole `TinyTitanBench` target (7 files) and
  `TinyTitanMemoryTool`'s entry point, the other four executable `main.swift`s, five
  more `TinyTitanRepack` files, two `TinyTitanFleet` support files,
  `ServerVersion.swift`, three `TinyTitanLib` files, two `TinyTitanValidation`
  support files, two C headers, `ContinuityDemo.swift` and `QSASelection.swift` —
  and none of the 16 `.metal` files is instrumented by LLVM at all.
- **Commit:** `74b56e1` on `audit/2026-10-23`.

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

**S1 · Tier A · DONE · TinyTitan runtime/streaming**

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
- **Resolution:** `executeCompressedMisses` forwards `firstStage` only from the
  wave whose `end` is `plan.misses.count`, matching `executeRANSPlan`, so the
  publish fires when the last wave's fronts are resident and every miss slot the
  consumer covers has bytes. The doc comment above it no longer claims a
  per-layer wave with a publish inside each one.
- **Evidence:** RED on the same tree. The new test plans ten misses (two waves of
  eight), poisons every slot byte with `0xAA`, and records at first-stage time
  which slots had been written. With the old `start == 0` condition it failed
  with `frontsWritten → [true ×8, false, false]` at
  `CompressedExpertStreamerTests.swift:308`, exit 1 (`AUDIT/aud2-033-red.log`);
  with the fix the eight-test suite passed, exit 0
  (`AUDIT/aud2-033-green.log`). The condition was reverted and restored by
  editing, not by discarding work, and `git status` on the file is clean. Full
  serial suite on the fix commits: **7 bundles, 1,993 tests, exit 0**
  (`AUDIT/aud2-020-021-033-full.log`); **all 18 lint gates exit 0**
  (`AUDIT/aud2-020-021-033-lint.log`). The one later change was the comment-only
  `d11be7c`. Sibling sweep: the raw bounded path reads all misses in one batch and
  publishes after that set's fronts, so it does not share the defect; the tiered
  branch that skips the split yet still declares `publishesFirstStage` is filed
  separately as AUD2-045. No live model check: the recorded golden is batch-1
  decode and does not reach a two-wave plan, so this is a structural proof.
- **Commit:** `71b40d7`, with the comment correction in `d11be7c`.

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

### AUD2-045 — A tiered layer declares a first stage its read branch never publishes

**S2 · Tier A · OPEN · TinyTitan runtime/streaming**

- **Location:**
  [plan construction](sources/TinyTitan/Infrastructure/Streaming/PreadExpertStreamer+Plan.swift)
  and the [tiered branch](sources/TinyTitan/Infrastructure/Streaming/PreadExpertStreamer+IO.swift)
  of `executeBoundedReads`.
- **Finding:** The bounded fallback builds the operation with
  `publishesFirstStage: boundedReader != nil && layout.phase1Bytes != nil`.
  `executeBoundedReads` early-returns for a layer whose `expertSizes` are not
  uniform (the tiered format), reading each expert's own length and skipping the
  phase-1 split; that branch never calls `firstStage`. `encodeDecodeEarlyMissFixup`
  then awaits `firstStageCompletion` before encoding the miss fixup, so on such a
  layer the fixup waits for the whole load instead of overlapping the tail read.
  There is no hang — `firstStageCompletion` also resumes at `finish` — but the
  declared stage never exists. AUD2-033 fixed the same publish condition for the
  wave path; this is the branch that bypasses it.
- **Current evidence:** Found by the sibling sweep while closing AUD2-033. Not
  reproduced: the tiered install is unreachable in this tree — `Model.swift:181-188`
  records that the tiered width it serves has no converter (TT-050) — so no plan
  reaches the branch. Severity is S2 on reachability plus lost overlap, not on
  results: the fixup is still ordered after the load.
- **Next:** Decide at layout construction: either the tiered branch publishes
  after its whole set has landed (keeping `publishesFirstStage` true), or
  `publishesFirstStage` is false whenever the uniform-size split will not run.

### AUD2-046 — Lazy checkpoint indexing evicts a pre-index load first and latches on a failed listing

**S2 · Tier A · DONE · TinyTitanLib**

- **Location:** [checkpoint store](sources/TinyTitanLib/PromptCheckpointStore.swift),
  `indexDiskIfNeeded` and `load`'s filesystem restore.
- **Finding:** Two manifestations, one cause. A record `load` finds on disk
  before indexing has run was appended to an empty `diskLRU`; indexing then
  skipped that key and appended the older records behind it, so eviction removed
  the just-touched record first. Separately, `diskIndexed` was set true before
  `contentsOfDirectory` was attempted and its error folded into an empty list,
  so one failed listing (root removed or replaced, `EACCES`, `EIO`) disabled
  indexing for the store's lifetime: `removeAll` unlinked nothing and
  `contains`/`metadata` answered from an empty index. The turn store does not
  share either shape — it indexes at startup, resets the index, and retries.
- **Current evidence:** Both tests failed on the unfixed tree for exactly those
  reasons (2 tests, 4 issues, exit 1, `AUDIT/aud2-046-red.log`): the backdated
  record survived while the newest was evicted, and the record directory
  survived a `removeAll` after one failed listing. `indexDiskIfNeeded` now sets
  its flag only after a successful listing and merges the in-process LRU with
  the indexed records by date. The checkpoint suite passes 30 tests
  (`AUDIT/aud2-046-green.log`); the full serial suite passes 7 bundles, 1,995
  tests, exit 0 (`AUDIT/aud2-046-full.log`) and all 18 lint gates exit 0
  (`AUDIT/aud2-046-lint.log`). The failed-listing fixture replaces the root with
  a regular file (ENOTDIR), so it does not depend on runner permissions.
- **Commit:** `7740a42`. Not merged, released or verified on an independent host.

### AUD2-047 — A calibrate conflict is refused with a message about the wrong flags

**S2 · Tier A · DONE · TinyTitanCLI**

- **Location:** [calibration command](sources/TinyTitanCLI/CalibrationCommand.swift)
  (lines 126, 140–141 and 270–286).
- **Finding:** `CalibrationCommandError.conflictingInput` carried no payload and
  its `description` was written for one site: "calibrate accepts either --model
  or --check, not both". All three conflict sites threw it, so `--plan --check`,
  `--plan --model` and `--measure --check` each printed advice about flags the
  user never passed — the first two are the refusals AUD2-008 just added, so the
  new refusal was correct and its diagnostic was not.
- **Current evidence:** The case now carries the two flag names and renders
  "calibrate --plan and --check cannot be combined"; each of the four sites passes
  its own pair. Only the `--plan` branch's shape changed (one `||` test became
  two), the refusals themselves are the ones already made, and mode precedence is
  untouched. RED: `eachConflictRefusalNamesItsOwnFlags` covers four combinations
  and records 3 issues — `--check --plan`, `--model --plan` and `--check --measure`
  each fail because the message omits the flag the refusal is about (18 tests,
  exit 1, `AUDIT/aud2-047-red.log`). GREEN: the same 18 tests pass, exit 0
  (`AUDIT/aud2-047-green.log`).
- **Disclosed:** the `--model --check` case passed before and after the change; it
  is the one message that was already right, and it is kept in the test so a later
  edit cannot quietly drop it.
- **Commit:** `3191968`. Not merged, released or verified on an independent host.

### AUD2-048 — The turn store's LRU touch discards its own failure

**S3 · Tier A · OPEN · TinyTitanLib**

- **Location:** [turn store](sources/TinyTitanLib/ServerPromptStateStore.swift#L405-L422).
- **Finding:** `touchDisk` updates the in-memory record, then on its background
  disk queue writes the directory's modification date under `try?`. A failure —
  read-only volume, `EACCES`, `EIO`, the directory replaced under the store —
  leaves the date at save time, so the restarted LRU silently disagrees with the
  in-process one and the touch that exists to protect a hot record does nothing.
  Nothing is logged.
- **Current evidence:** None yet; the row exists because closing AUD2-007 made
  the two halves of one mechanism answer differently: the checkpoint store now
  reports the same failure through `ServerLog.diagnostic` (`92da15a`) while the
  turn store discards it. The fix is that one line's shape; the finding is the
  silence, not the error handling. No correctness or data-loss impact measured.
- **Next:** Sweep, then report the failure on the background queue.

### AUD2-049 — The docs gate never reads this audit's own ledger

**S2 · Tier A · OPEN · audit tooling**

- **Location:** [docs-facts](tools/docs-facts.py#L269-L289) and
  [`ledger_commit_evidence`](tools/docs-facts.py#L321-L360).
- **Finding:** `ledger_totals()` lists its ledgers with
  `git ls-files docs/audit-ledger.json` — a hardcoded path to the *prior*
  (2026-10-06) ledger — and `ledger_commit_evidence()` iterates exactly that
  list. `AUDIT/ledger.json`, which this report names the audit's authoritative
  record, is never opened. The gate prints "222 ledger commit reference(s)
  resolved" while reading none of this audit's 34 pointers, so a DONE row here
  naming a nonexistent sha, prose, or nothing at all would still report ok.
  AUD-209 is the reason the check exists: seven rotted commit fields in the
  ledger it *does* read.
- **Current evidence:** Checked by hand twice this session, because the gate
  cannot check it: all 34 `commit` references in `AUDIT/ledger.json` resolve
  under `git cat-file -e <sha>^{commit}` and no DONE row has an empty field (32
  before `8acc6f8` gave AUD2-013 and AUD2-014 theirs). Re-running
  `bash tools/lint.sh docs` after that commit prints the *same* 222 — every one
  of them from `docs/audit-ledger.json`. So the finding is that the gate would
  not see rot, not that any exists today.
- **Next:** Take ledger paths as a set that includes `AUDIT/ledger.json`, keep
  the prose exemption, judge only the structured `commit` field, and add the test
  that puts an unresolvable sha in a copy of this ledger and expects a FAIL —
  without it the fix is a path list nothing checks.

### AUD2-050 — A type-level `#expect(throws:)` cannot say which guard refused

**S2 · Tier A · OPEN · TinyTitan**

- **Location:** `tests/` — **302** type-level `#expect(throws: <Type>.self)`
  assertions across **75** files, 298 of them refusals (4 are the
  `throws: Never.self` acceptance form), against callees including
  [`ExpertRANSIndex.decode`](sources/TinyTitanFormat/ExpertRANSIndex.swift) (15
  throw sites behind an 8-case `Failure` enum),
  [`TwoBitExpand.validateExpansionLayout`](sources/TinyTitan/Infrastructure/Streaming/TwoBitExpand.swift#L73-L137)
  (6) and
  [`Model+SchemaValidation`](sources/TinyTitan/Runtime/Inference/Model+SchemaValidation.swift)
  (6).
- **Finding:** `#expect(throws: ModelError.self)` is satisfied by *any* refusal
  from the callee, so a fixture aimed at one guard passes when a different guard
  refuses it. The test reports green, the guard it names stays unpinned, and
  nothing in the output says so.
- **Current evidence:** Proven twice inside the suite closed under AUD2-013:
  disabling the 3-bit whole-group guard left every test passing because the only
  3-bit negative case is equally refused by the target-size guard, and
  `tensorsThatShareATargetRegionAreRefused` had been refused by the slot bounds
  long before the overlap check it names. Both were fixed there, with a
  `refusalDetail` helper that asserts the `indexCorrupt` detail names the guard.
  The rest of the tree has not been swept: the pool is counted on the current
  tests, not modelled — 39 `ModelError`, 37 `RepackError` and 32 written against a
  nested type (`ExpertRANSIndex.Failure` 4, `ExpertBlobIndex.Failure` 7,
  `NgramTableReader.Failure` 8 among 7 such types), with
  `grep -rnE 'expect\(throws: [A-Za-z_.]+\.self\)' tests/`. **That is a candidate
  pool, not a defect count:** the form is harmless where the callee has one throw
  site and wrong only where a second site can answer for the first. No test
  outside AUD2-013's own suite has been mutation-tested yet, and this record
  claims no product defect — it records that the tree's negative tests are
  un-verified in the specific way AUD2-013 proved two of them were.
  Deviation on the count, same class as the finding: the first measurement used
  `[A-Za-z]+` before `.self`, which cannot match a dotted name, and reported 270
  assertions in 68 files — it missed all 32 nested-type assertions, including the
  `ExpertRANSIndex.Failure` cases this record names as its worst instance.
  Re-measured with `[A-Za-z_.]+` to the 302 / 75 above, and the generator
  (`AUDIT/agent-notes/apply-aud2-013.py`) carries the same fix so the ledger row
  regenerates to what it claims.
- **Next:** Mutation-sweep the pool the way AUD2-013 was swept — disable one
  throw site at a time and require the test that names it to fail — and where a
  fixture is refused by the wrong site, move the fixture or assert the case or
  detail that names the guard. Fixing this in the suite this audit already
  touched is not enough; the same form is what hid the two wrong-reason tests.

### AUD2-051 — The CI markdown-link step fails on the audit's own records

**S2 · Tier A · DONE · audit evidence/CI**

- **Location:** [CI workflow](.github/workflows/ci.yml) (the
  `Check Markdown links` step, lines 1061–1116; 1061–1096 before this fix),
  17 links in this report, and `AUDIT/ledger.md:179`.
- **Finding:** The step's own script, run verbatim at `b719a4a` on a clean tree,
  reports `Checked 365 local link(s); 19 broken` and exits 1
  (`AUDIT/aud2-051-markdown-link-check.log`). Two distinct causes:
  - **17 links written as `path:lines`.** `[Foo.swift:352-377](sources/…/Foo.swift:352-377)`
    puts a `:352-377` suffix on the URL, which is neither a path nor a fragment:
    the step's `File.exist?` is false, and GitHub cannot resolve it for a reader
    either. The repository already has the other convention — `docs/` writes the
    readable text and an anchor URL
    (`[PreadExpertStreamer+Plan.swift:95-106](…/PreadExpertStreamer+Plan.swift#L95)`),
    and across tracked markdown 179 links use an `#L` anchor and 167 a plain path,
    against these 17 suffixes. Every one of the 17 cited ranges was checked against
    its file and is **in range**, so this is link syntax, not a stale citation.
  - **2 code spans read as links.** The step scans raw lines, so a Swift
    initializer inside backticks — `` `offsets: [UInt64](repeating: 0, count:)` ``
    at `audit_2310.md:695` in that baseline run (699 now), and the same text
    rendered into `AUDIT/ledger.md:178` (179 now) — matches the link regex,
    because `](` looks like a link to a scanner that has no notion of code spans.
    In CommonMark a code span is not a link.
- **Exposure:** `audit_2310.md` and the rendered `AUDIT/ledger.md` are tracked only
  on `audit/2026-10-23` (`git ls-tree main` shows neither), and the workflow
  triggers are `push: branches: main` and `pull_request`, so **no CI run has
  executed this step over these files yet** — the first PR that carries this branch
  fails it. That is why the defect is invisible rather than absent: the step has
  never been run against the documents the audit produced.
- **Severity rationale:** S2, not S1 — it turns CI red on the audit branch and makes
  17 citations unclickable; it ships no runtime defect. It is not waived: the
  checker must keep resolving every real link, and skipping a code span is not a
  weakening because a code span is never a link.
- **Fix:** two parts, one per cause. The step's ruby scanner now tracks fenced
  blocks (``` and `~~~`) and strips *paired* inline code spans from a line before
  matching, so literal Swift text is no longer read as a link; because only matched
  pairs are removed, a line with an unmatched backtick still has its remaining
  links checked, and a link after a fence close is still checked. In the records,
  the 17 URLs written as `path:352-377` became the convention the rest of the tree
  uses — readable text, anchor URL (`Foo.swift:352-377` with URL
  `Foo.swift#L352-L377`), so a reader gets a clickable citation and the checker
  resolves the path. Three illustrative link-shaped snippets in `AUDIT/ledger.json`
  — the AUD2-014 fix-summary's Swift initializer and the two examples this row
  quotes — are now backtick-wrapped, so the rendered ledger shows them as the code
  they are. No citation, number or claim changed, and each of the 17 ranges was
  re-checked against its file.
- **Evidence:** four runs in `AUDIT/aud2-051-verification.log`, each with the step's
  ruby extracted from the YAML itself (which also proves the job still parses), on
  ruby 2.6.10p210. On four fixtures (`broken.md`, `code.md`, `fenced.md`,
  `keep.md`) the OLD step reports 7 broken of 8 — four of them a code span, a
  double-backtick span and two fenced-block lines — while the NEW step reports
  exactly the three genuinely missing files and exits 1: `Checked 4 local
  link(s); 3 broken`. The check is therefore not weakened. Over this tree the NEW
  step reports `Checked 363 local link(s); 0 broken`, exit 0, and the OLD step over
  the SAME already-converted tree still reports 7 broken (all code spans here and
  in the rendered ledger), which isolates cause 2 from cause 1 instead of leaving
  the two fixes entangled. The counts reconcile: 370 checked before, 363 after — the
  7 code-span matches are no longer counted, the 17 converted links still are. The
  filed baseline was 19 of 365 at `b719a4a`; the filing text itself raised it to 24
  of 370 (`AUDIT/aud2-051-filing-run.log`) before any fix, because writing about the
  defect in markdown trips it. After these record edits, the step re-extracted from
  the committed workflow reports `Checked 364 local link(s); 0 broken`, exit 0
  (`AUDIT/aud2-051-final-link-check.log`) — 364 rather than 363 because the
  rewritten Location bullet links `ci.yml` itself — and the `docs` gate is green over
  the edited records (`AUDIT/aud2-051-final-docs-gate.log`). All 18 gates re-ran with
  the edited workflow in the
  tree: every gate ok, `LINT_EXIT=0` (`AUDIT/aud2-051-lint.log`).
- **Not checked, stated:** the step still has not run in CI over these files — the
  branch is unmerged and the triggers are `push` to `main` and `pull_request`. And
  the checker validates that the target *file* exists, never that an `#L` anchor
  still names the lines the prose claims; converting these links does not fix that,
  no line-number validation is claimed here, and the audit's own records remain read
  by no gate (AUD2-049).
- **Commit:** `d6b1842` on `audit/2026-10-23` (the checker); the record conversions
  ride in the records commit that names it.

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

The keep-warm pair and AUD2-033 were verified together on the same tree: the full
serial suite (`swift test --no-parallel`, **7 bundles, 1,993 tests, 0 recorded
issues, exit 0**, `AUDIT/aud2-020-021-033-full.log`) and all 18 lint gates
(`tools/lint.sh`, exit 0, `AUDIT/aud2-020-021-033-lint.log`). AUD2-020's race was
re-run under TSan on the final tree (**3 tests, 0 ThreadSanitizer warnings, exit
0**, `AUDIT/aud2-020-tsan.log`); AUD2-021 has no unit seam for a failed
allocation and is recorded as reviewed, compiled and suite-covered rather than
tested; AUD2-033's regression test was RED against the old condition
(`frontsWritten → [true ×8, false, false]`, exit 1, `AUDIT/aud2-033-red.log`) and
GREEN after the fix (exit 0, `AUDIT/aud2-033-green.log`). The comment-only
`d11be7c` followed the suite; no code changed after it.

The checkpoint batch at `c5e2e93` was verified the same way: full serial suite
(`swift test --no-parallel`, **7 bundles, 1,993 tests, 0 recorded issues, exit 0**,
`AUDIT/aud2-028-full.log`) and all 18 lint gates (`tools/lint.sh`, exit 0,
`AUDIT/aud2-028-lint.log`). Its RED is a post-hoc mutation on the final tree, not
a pre-fix run, and both mutations were reverted by editing with the file restored
before the GREEN: `AUDIT/aud2-028-red-mutation.log` (3 tests, 5 issues, exit 1)
and `AUDIT/aud2-028-red-index-validate.log` (the added `metadata(for:)`
assertion fails). The focused suite passed 28 tests, exit 0
(`AUDIT/aud2-028-green-final.log`). AUD2-026's three-interpreter Python run
predates `5deb70a`, the same source under test; no full Swift run was repeated
for it because it touches only the benchmark sweep.

AUD2-046 was written as a test-first pair on the unfixed tree: 2 tests, 4 issues,
exit 1 (`AUDIT/aud2-046-red.log`), and passed after the fix (30 checkpoint
tests, exit 0, `AUDIT/aud2-046-green.log`). The fix commit `7740a42` carries the
full serial suite (**7 bundles, 1,995 tests, exit 0**, `AUDIT/aud2-046-full.log`)
and all 18 lint gates (`AUDIT/aud2-046-lint.log`, 18/18).

AUD2-029's first RED run was invalid and is not counted: the bytes assertion ran
after `contains`, which re-indexes and restored what the test meant to measure,
so the test passed against unfixed code. After reordering, the three
immutable-directory tests failed for their recorded reasons (3 tests, 6 issues,
exit 1, `AUDIT/aud2-029-red.log`) and passed after the fix (33 checkpoint tests,
exit 0, `AUDIT/aud2-029-green.log`). The fix commit `7024d03` carries the full
serial suite (**7 bundles, 1,998 tests, exit 0**, `AUDIT/aud2-029-full.log`) and
all 18 lint gates (`AUDIT/aud2-029-lint.log`, 18/18).

Six records (AUD2-003, 005, 006, 008, 009, 017) were found already fixed in the
working tree with no ledger status — the previous session's uncommitted work. They
were taken over rather than trusted: each was mapped to its record, re-read in
full, and given a before/after run. Five landed at `41d14ca` and `6130898`; the
tree carrying both was verified with the full serial suite (**7 bundles, 2,002
tests, exit 0**, `AUDIT/aud2-pending-full.log`) and all 18 lint gates (exit 0,
`AUDIT/aud2-pending-lint.log`). Three disclosures ride with them: AUD2-003 has no
behavioral RED (its seam is new, so the unfixed tree fails to compile, not
asserts); AUD2-006's new test is a pin, not a RED, because the pre-fix reader also
refused the oversized file after reading all of it; and AUD2-009 has no unit seam
at all and is recorded as reviewed and compiled. The last of these is a real gap,
not a formality — a guard that catches a defect and has no test is half a fix, and
these rows say so instead of borrowing the suite pass as proof.

AUD2-017's deviation is a history defect, not just a missing test: its
expectations were committed in `9ca9942` while the source fix stayed uncommitted,
so `parsesHostKVWindow` failed on every commit from `9ca9942` through `7024d03` and
the full-suite runs recorded against those commits passed only because the
working tree carried the fix. `41d14ca` lands the fix and the suite was re-run on
that committed tree; the earlier "green" runs are withdrawn as evidence that those
commits were sound.

AUD2-047 was written test-first on the tree `6130898` left: 18 tests with 3 issues,
exit 1 (`AUDIT/aud2-047-red.log`), and 18 tests passing, exit 0
(`AUDIT/aud2-047-green.log`). The fix commit `3191968` was verified as the tree it
records — full serial suite **7 bundles, 2,003 tests, exit 0**
(`AUDIT/aud2-047-full.log`; the only two lines in that log matching `error:` are the
corrupt-record quarantine diagnostics the checkpoint test provokes on purpose) and
all 18 lint gates exit 0 (`AUDIT/aud2-047-lint.log`, 18/18). Nothing ran a model,
and `AUDIT/*.log` evidence stays in the gitignored audit tree, so these files are
not delivered with the report.

AUD2-007's test was written first and watched fail on the unfixed tree for the
reported reason (34 tests, 2 issues, exit 1, `AUDIT/aud2-007-red.log`), then
passed with the persisted touch (34 tests, exit 0, `AUDIT/aud2-007-green.log`).
The fix commit `92da15a` was re-verified as the tree it records after
`swift-format` reflowed the new fixture and a redundant comment line was dropped:
full serial suite **7 bundles, 2,004 tests, exit 0**
(`AUDIT/aud2-007-full.log`) and all 18 lint gates exit 0
(`AUDIT/aud2-007-lint.log`, 18/18), with the focused checkpoint suite re-run on
that committed tree. One scope note, because the row predates the split: the
title used to carry "(latent: no production caller)". That was the `removeAll`
half, which `7740a42` closed. The access half was never latent —
`ServerModelSession+PromptCache.swift:333` restores through `load` — and the
closure says so rather than narrowing the row.

AUD2-013 and AUD2-014 were proved rather than asserted, and the proving method is
recorded because it is the part that transfers: the validator at `61de182` was
left as it stood (no production edit except one doc comment) and each of its eight
guards was disabled in turn, `ExpansionLayoutValidationTests` re-run, the file
restored and verified byte-identical after every mutation
(`AUDIT/aud2-013-mutation.log`, `AUDIT/agent-notes/mutate-sweep-stdout.log`,
`AUDIT/mutate-*-full.log`). Five guards failed exactly the test named for them;
M4 and M8 failed nothing because their fixtures were refused by a different clause,
and M3 failed the wrong tests because a bound nothing can satisfy proves nothing.
The focused result is 9 tests in 2 suites exit 0
(`AUDIT/aud2-013-green-9.log`), the streaming suites 26 tests in 3 suites
(`AUDIT/aud2-013-streaming-suite.log`), and the commit `8acc6f8` was then re-run
whole: full serial suite **7 bundles, 2,007 tests, exit 0**
(`AUDIT/aud2-013-full-final.log`, whose header carries the exact command, the
`xcrun` Swift 6.4 `swiftlang-6.4.0.34.1` and the HEAD it ran on) and all 18 lint
gates exit 0 (`AUDIT/aud2-013-lint-final.log`, 18/18). `swift build -c release`
on that same tree reports `Build complete! (109.85 sec)` with `BUILD_EXIT=0`
(`AUDIT/aud2-013-release-build.log`, stamped with HEAD `8acc6f8` and the only
dirty files being this report and the two ledger files). The gates then ran a
second time after the report and ledger edits below: 18/18, `LINT_EXIT=0`
(`AUDIT/aud2-013-lint-final2.log`). The second run is what makes the edits above
safe to land without a third: no gate reads either file — `grep -rn audit_2310
tools/ .github/workflows/` returns nothing, and `tools/docs-facts.py` reads only
`docs/audit-ledger.json`, which is AUD2-049. No model ran: nothing in this pair
touches numerics, which is why no golden-baseline check is claimed.
Sibling audit under the brief's §5 turned up the tree-wide version of the same
test defect and it was filed as AUD2-050 rather than folded into this closure.

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
  27 files, which confirmed the defect. The durable fix landed later in the same
  day in `74b56e1`; that artifact is evidence of the defect, not of the fix,
  because no CI run has executed the new step yet.
- Two tests in `ExpansionLayoutValidationTests` had been **passing for the wrong
  reason**, and that is disclosed rather than quietly fixed:
  `tensorsThatShareATargetRegionAreRefused` was refused by the slot-bounds check
  (its second tensor sat at target offset 4 below source offset 8), never reaching
  the overlap guard it is named for, and the only 3-bit negative case (source 4 →
  target 6) was equally refused by the target-size check, so removing the
  whole-group guard changed nothing. Every test-file change closing AUD2-013 is
  inventoried: `refusalDetail` added (asserts the `indexCorrupt` *detail*, not
  just the error type); the shared-target fixture rebuilt so both targets sit at
  or beyond their own source; `aThreeBitTensorWithALeftoverGroupIsRefused`,
  `anOddRescaledTensorIsRefused` and `aNegativeSourceOffsetOrSizeIsRefused` added;
  one comment corrected ("Six source bytes are exactly two groups of four"). **No
  assertion was weakened and no test removed** — the sweep removed a production
  guard temporarily, never a test. Assertions that mutation showed were already
  honest kept their original `#expect(throws: ModelError.self)` form.
- One mutation was invalid as evidence and is disclosed as such: M3 as first
  written *tightened* the slot bound past any satisfiable value
  (`<= expertStride` became `<= -1`), so it failed
  `sizesThatDoNotMatchTheTransformAreRefused` and the positive
  `theConverterLayoutIsValid` — 3 issues, exit 1 (`AUDIT/aud2-013-mutation.log`) —
  and never isolated `aRegionOutsideTheSlotIsRefused`, because a bound nothing can
  satisfy makes every layout throw. M3b relaxes the bound instead and fails
  exactly that one test, both its cases (`AUDIT/mutate-aud2-013-m3b-full.log`, 2
  issues, lines 336 and 341). The sweep driver's line-based `grep -qF` check could
  also not verify a multi-line replacement, which is why M8 was re-run through
  `AUDIT/agent-notes/mutate-one.py`, an applier that asserts the pattern occurs
  exactly once and restores the file in a `finally`.
- The AUD2-050 count was first measured as 270 assertions in 68 files with
  `expect\(throws: [A-Za-z]+\.self\)`, a pattern that cannot match a dotted type
  name and so missed all 32 nested-type assertions — including the
  `ExpertRANSIndex.Failure` cases the record itself cites as its worst instance.
  Re-measured with `[A-Za-z_.]+` to 302 in 75 files, in both the ledger row and the
  generator script. The old numbers are withdrawn.

## Continuation boundaries

Continue from the recorded issue evidence, not from a claim that passing tests
means all bugs are fixed. Six records were found this session already fixed in the
working tree with no ledger status; they were taken over, re-proved and committed
rather than trusted — keep doing that, and never discard uncommitted work or the
invalid-evidence history. Do not close a static claim without the required
reproduction/review, replace a disproven claim with its sibling, or treat absent
evidence as a pass.

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

The ledger stands at **50 records: 34 DONE, 15 non-terminal (11 OPEN, 2 PROGRESS,
2 AUDIT) and 1 BLOCKED**. The count has fallen at every earlier recorded
milestone — 36 non-terminal at the last snapshot of this section, 21 after the
rANS and checkpoint batches, 15 before the AUD2-007 pass, which closed one record
and opened two to land at 16. It fell again here: AUD2-013 and AUD2-014 closed
with evidence and one new record (AUD2-050, the tree-wide sibling of the
wrong-reason tests AUD2-013's sweep proved) arrived, so 16 → 15. That is stated
rather than smoothed over — §8's invariant is a record moving to a terminal
status, and one net drain at one milestone is not convergence. The run has
still never reached Phase C, let alone Phase D, and no convergence milestone is
claimed. Every
non-terminal record left is either an S2 design gap that needs work
this branch has not done (AUD2-001, 002, 004, 034, 035, 036, 039,
045) or a gap in the audit's own tooling and tests (AUD2-049, AUD2-050), an S3
diagnostic asymmetry (AUD2-048), a claim awaiting formal reconciliation
(AUD2-015, 016), a verification record
(AUD2-019, 031), or the external host (AUD2-038).

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
remediation of every OPEN/PROGRESS/AUDIT record — including AUD2-035, which the
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
