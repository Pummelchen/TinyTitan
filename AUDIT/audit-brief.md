# Original audit brief

Preserved verbatim, because it is the specification this audit is measured
against and it was not previously in the repository.

| | |
| --- | --- |
| Source | operator attachment `64ce6918-ced4-4371-ab9a-ba09df9d00fb.txt` |
| Recovered | 2026-10-23 |
| Size | 17,800 bytes, 92 lines |
| SHA-256 | `0390fa3d63e92f40d8eaedff2a1c1f74c5cbd07b670a42f32123663fa2aec3e2` |

The digest covers the brief text below, from `ROLE You are performing` through
the final `open:0.` line. Nothing in this file is commentary on the brief; the
audit's own reconciliation against it lives in
[`../audit_2310.md`](../audit_2310.md).

---

ROLE You are performing a pre-production audit and remediation of this repository. It ships today. There is no time limit and no time budget. The run continues until the ledger reaches zero open tasks (see §12). A partial run is never "done".

=== 0. NON-NEGOTIABLES ===

Never weaken a check to make something pass. Forbidden as "fixes": deleting or skipping tests, loosening assertions, broadening catches, empty catch/rescue, # type: ignore/@ts-ignore/! force-unwrap to dodge an error, try!, #pragma warning disable, -Wno-*, @unchecked Sendable, lowering strictness (including downgrading the Swift language mode or SWIFT_STRICT_CONCURRENCY), or excluding files from analysis. A genuinely wrong check is fixed as its own numbered task with justification.
GIT: work only on branch audit/<date>. Never commit to main/master/release, never force-push, never rewrite history, never delete branches or tags. Merge only via PR after Phase E. Print the exact command plus its rollback into the ledger before any destructive-looking operation.
Never touch live runtime systems: no production DB writes, no migrations against a live DB, no restarts, no credential rotation, no deploys. Produce migration + rollback scripts as human tasks — these become BLOCKED(owner), which is a terminal, counted state.
Never print/log/commit/echo secrets. Report only "file:line contains a live-looking credential".
You may not close a task by narrowing its scope. Scope changes require a new task plus an explicit note on the original.
If in doubt, stop and write the doubt down before acting.
=== 1. TOOLCHAIN === Install one pinned toolchain per language present. Required per language: formatter, linter, dependency/CVE scanner, and a type checker where the language standard below requires one (the Python rules drop it). Add SAST and sanitizer/memory checker only for languages in Tier A code. Run the secret scanner over full history once (history is immutable under §0). Record name, version, install method and host in AUDIT/environment.md. Uninstallable tool → BLOCKED. Do not build a tool matrix across hosts; one pinned version per language is the standard. Prefer tools with an auto-fix mode and wire it in from the start — mechanical resolution is a primary cost lever, not an afterthought.

Language standards — enforced, not aspirational. A standard is only in force if the build actually fails without it. Naming a mode here enforces nothing; the build config must carry it, and you must prove it below.

Swift → Swift 6.4 (Xcode 27). Every Swift target compiles in Swift 6 language mode with complete strict concurrency and warnings-as-errors. Enforce in build config, not per-invocation: SWIFT_VERSION = 6.0 (Swift 6 language mode), SWIFT_STRICT_CONCURRENCY = complete, SWIFT_TREAT_WARNINGS_AS_ERRORS = YES, plus a committed swift-format config and a committed SwiftLint config run with --strict so linter warnings also fail the build. If Swift 6.4 (Xcode 27) is not installed on any host, that is BLOCKED — never substitute whichever Swift happens to be present, and never lower the language mode or relax strict concurrency to make code compile (a §0 violation).
C → strict C99. -std=c99 -pedantic-errors, warnings-as-errors, no GNU/compiler extensions, no implicit declarations, plus the hardening flags that -Wall -Wextra does not enable: -Wshadow -Wconversion -Wsign-conversion -Wcast-qual -Wwrite-strings -Wformat=2 -Wstrict-prototypes -Wmissing-prototypes.
Python → basic only. Python here is test and glue code, so keep it cheap. Pin one Python version, use Ruff for both formatting and linting (with --fix for the S3 sweep), and keep the scripts importable (ruff check clean or python -m compileall). Run pip-audit once at baseline if a requirements/lock file exists. Do not stand up mypy/pyright strict mode, per-file coverage targets, or a Python tool matrix — that cost is not repaid on test scripts. This is a de-scope by language, never by tier: a Python script that writes production data, handles credentials, or runs as a migration is Tier A and gets the full treatment regardless. The pitfalls worth covering — enable the Ruff rules that catch them rather than reading the scripts: B (flake8-bugbear), E722 (bare except), S101 (assert), PT (flake8-pytest-style).
bare except: or except Exception: pass swallowing a real failure
subprocess without check=True — the failure vanishes
time.sleep() used to synchronize → flaky
test order dependence or shared mutable module state
tests that assert nothing; assert used for validation that -O strips
is compared against ints/strings instead of ==
mutable default arguments (def f(x=[]))
open() without encoding=; datetime.now()/utcnow() without a timezone
hardcoded paths or cwd dependence The tool-coverage proof applies: introduce a bare except and confirm Ruff rejects it.
These standards are existing checks, not new work. Skipping them is how findings get silently lost.

Tool-coverage proof — required before any check is delegated to a tool. A human check delegated to a tool is only covered if that tool is actually configured to catch it. For every such delegation, introduce a deliberate violation in a scratch file and record the tool's output in AUDIT/tool-coverage.md. A tool that stays silent does not cover the check: restore the human check or fix the tool config. Delegating to an unproven tool is how findings get silently lost.

Language-standard proof — required, same gate as the tool-coverage proof above. For each language, introduce a violation the standard is supposed to reject and confirm the build fails:

Swift: a non-Sendable value captured across an actor or Task boundary (must raise a strict concurrency error in Swift 6 mode), and a force-unwrap (must fail SwiftLint --strict).
C: an implicit declaration, and a GNU extension such as a nested function or typeof (must fail under -pedantic-errors).
Record the violating snippet and the compiler/linter error in AUDIT/tool-coverage.md. A mode that happily compiles the violating code is not enforced — fix the build config before auditing anything.

Sanitizers run the suite, not just the build. For C, run the full test suite under ASan + UBSan; for Swift, run concurrency-heavy targets under swift test --sanitize=thread. Findings are S0/S1. Once ASan is in use, do not also run Valgrind — it costs hours for the same defect class.

=== 1b. HOSTS === Assign every project exactly one primary host at discovery: Apple-platform work on a Mac; Linux/x86 (Docker, C sanitizers, .NET, Python matrix) local or VPS. Baseline, fixes, and tests run on the primary host. One independent host runs the single final verification for the whole repo. At most one heavy job per 8 GB Mac; if a job OOMs, record it and lower parallelism — do not retry blindly. Ask before provisioning the VPS. Log everything installed on every remote host. The Mac carrying a Swift project must have Xcode 27 / Swift 6.4 installed — verify with xcodebuild -version and swift --version and record both in AUDIT/environment.md. Do not attempt Swift 6.4 strict-concurrency builds on Linux hosts or with a mismatched toolchain; that is BLOCKED with the missing-version reason, never a reason to lower the standard.

=== 2. SCOPE + RISK TIERS === 2.1 Enumerate projects, languages, build systems, entry points, primary host. 2.2 Dependency graph: direct module deps + cross-project contracts (env vars, schemas, HTTP/IPC, DB tables, queues, shared model interfaces). Depth cap: 2 hops. Anything coupling at 2 hops is automatically Tier A; do not recurse further. 2.3 Trust boundaries: untrusted input, credential holders, network-reachable surfaces, and every native-interop seam (Swift↔C via module map or bridging header) — usually the most bug-dense boundary in the repo. 2.4 Classify every module:

Tier A — deep manual. Authn/authz, secrets/credentials, untrusted-input parsing, network-facing endpoints, persistent-data or money mutation, irreversible operations (migrations, backfills), cross-project contracts with >1 consumer, unsafe/native memory code, native-interop boundaries (pointer lifetimes, escaping UnsafePointer, missing _Nonnull/_Nullable annotations, String(cString:) on non-terminated buffers, Unmanaged ownership), LLM output driving privileged actions. In a C99 repo, treat every module with native memory access as Tier A.
Tier B — tool-first. Remaining production code. Automated tools gate it. Manual review only on tool findings, coverage gaps, and complexity/change hotspots.
Tier C — scanner-only. Tests, docs, non-production scripts, examples, fixtures, vendored/generated code. Formatter/linter/secret-scan only. (Migration and backfill scripts are not here — they are Tier A on the strength of irreversible data mutation, whatever language they are written in.) Unsure → Tier A. Commit the inventory + tier table before fixing anything, and state the tier coverage explicitly in the final report so reduced inspection is disclosed, never hidden. Tiering reduces how much surface is human-read. It does not reduce how many findings get fixed: findings from all three tiers are enumerated and closed.
=== 3. BASELINE (once per project, primary host) === Per project: build result + warning counts, test pass/fail/skip + coverage %, linter/analyzer/type checker findings by rule, dependency CVEs, secret-scan result (full history, once). Commit as the regression yardstick. No later state may be worse on any metric without a justified numbered task. Do not duplicate the baseline across hosts. Record the language standard actually in force, not the one you intended: swift --version and xcodebuild -version, the resolved SWIFT_VERSION / SWIFT_STRICT_CONCURRENCY / SWIFT_TREAT_WARNINGS_AS_ERRORS build settings, and the exact C standard plus warning flags. A baseline produced in the wrong language mode is not a baseline.

=== 4. AUDIT PASSES (by tier) === Checklist rule: if a check's worst plausible finding is S3, it is not on this list. Findings are still enumerated at every tier.

L0 Repository — once, repo-wide: build reproducibility, pinned deps, lockfiles, CI config, .gitignore hygiene, committed artifacts or secrets.
L1 Architecture — Tier A + Tier B: module boundaries, circular deps, layering violations, contract drift between projects, error-propagation strategy, config management.
L2 Module — Tier A manual: public API correctness, invariants, resource lifecycle (files, sockets, handles, cursors, tasks), concurrency model, cancellation, timeouts, retries, and backpressure/idempotency on network or queue paths. Tier B: tool findings only.
L3 Line — Tier A full manual: logic errors, off-by-one, wrong operator/branch, silent truncation/overflow, float/decimal misuse, timezone and encoding bugs, unchecked return values, unhandled error paths. Tier B: sampled via coverage gaps, complexity hotspots and recently changed lines. Tier C: skip. Syntax, naming, shadowing, dead parameters, magic numbers and duplication are deliberately not read by a human — the formatter and linter own them.
L4 Security (not reduced) — Tier A manual + SAST. Tier B: SAST + dependency + secret scan. Tier C: secret scan. Injection (SQL/shell/path/template), deserialization, SSRF, TOCTOU, unsafe temp files, permissive file modes, weak crypto/RNG, missing authn/authz, secrets in code or logs, unvalidated external/LLM output used in a privileged context, prompt injection driving actions, unsafe blocks and native memory safety.
L5 Performance — Tier A + profiler-identified hot paths only: N+1 queries, repeated I/O, unbounded memory or collections, missing pagination/streaming, blocking calls on async paths.
L6 Tests — Tier A: coverage gaps on critical paths, failure-path and boundary cases, tests that assert nothing. Tier B: coverage threshold gate on critical paths. Cross-project integration tests required only where §2.2 found a contract with >1 consumer.
L7 Ops — once per deployable service (not per module): structured errors, health checks, graceful shutdown, startup config validation, rollback path.
=== 5. FACADE HUNT === Hunt only for things that fake work on a production path, because those are the ones that can reach a customer:

always-true validators or auth checks, hardcoded "success" returns
NotImplemented* / fatalError("TODO") / empty bodies on a production path
mocked, canned or fixture data served on a non-test path
sleep() standing in for real processing; fabricated latency or progress
config defaults pointing at localhost / test / example endpoints
UI or API surfaces wired to nothing Each of these becomes a task; a production-path facade is S0. TODO/FIXME/HACK/XXX/WIP/STUB markers and the words placeholder/dummy/sample/lorem/foo are not grepped as a task source. They record intent, and the search mostly returns test fixtures. If you want visibility, list marker counts once in an appendix — no tasks, no fixes.
=== 6. UNUSED CODE === No repo-wide reachability proofs. Cover only:

code in Tier A modules, and
code the audit's own changes orphaned. Default action is deprecate plus one removal ticket per module. Spend the reachability proof once per module sweep, covering all its candidates together, and only before an actual deletion. Everything else is a backlog chore, not a go-live risk.
=== 7. SEVERITY + WORK ORDER === S0 go-live blocker (data loss/corruption, security hole, crash on a normal path, wrong results, production placeholder). S1 high (incorrect edge case, leak, missing error handling, unsafe concurrency, missing critical test). S2 medium (performance, maintainability, weak typing, dup logic). S3 low (style, naming, comments, formatting). Order: all S0 → all S1 → S2 → S3. Never defer an S0. Nothing is dropped at any severity — S3 is resolved by the automated sweep, not by a waiver.

=== 8. LEDGER (single source of truth) === Committed AUDIT/ledger.md + AUDIT/ledger.json. Re-read at the start of every session and resume; never restart. Fields: id | severity | tier | project | file:line | title | category | status | host | discovered-by | evidence-before | fix-summary | evidence-after | commit | blocked-reason.

S0/S1 gates: START (reproduced or statically proven; expected-correct written down) → PROGRESS (the diff) → TEST (test failing before / passing after, full suite green on the primary host, no new warnings) → AUDIT (cold re-read + affected scanners re-run + language standard still in force + no baseline regression + no new placeholder) → DONE (commit on the audit branch).
S2 gates: OPEN → PROGRESS → TEST (only if behavior changed) → DONE.
S3 gates: OPEN → SWEPT (one commit covering a rule class via formatter/linter/codemod) → DONE. No test, no cold re-read. Behavioral S3 is reclassified S2.
Batch tasks: one task may cover many instances of the same rule; evidence-before/after carry the instance count and the command, not 200 log excerpts.
BLOCKED requires a named owner, the reason, what was tried, and 2+ options for a human. BLOCKED is terminal and counted; OPEN is not terminal.
Closure invariant: the count of non-terminal tasks is strictly decreasing at every milestone report. If it is not, stop and fix the loop before continuing.
=== 9. REPORTING === One source of truth: the ledger. Generate the milestone report from ledger.json; do not hand-maintain a second tracker. If a wiki page is required, it is generated output and never edited by hand.

=== 10. CHANGE DISCIPLINE === One task = one focused commit for S0/S1 (audit(<id>): <title> + what/why/evidence). S2/S3 may batch by rule class into a single sweep commit. Audit branch only, no force-push, every change revertible. Cross-project contract fixes update every consumer as linked sub-tasks before the parent is DONE. Clean up artifacts/containers you created; leave the fleet documented.

=== 11. EXECUTION === Interleave discovery and repair; do not freeze a giant backlog first.

Phase A — §2 inventory + tier table, §3 baseline, §1 environment record. Commit.
Phase B — Open the loop. Run the L0–L7 passes at tier-appropriate depth. Every finding is recorded in the ledger the moment it is found, numbered, then fixed in severity order as you go. When a fix reveals a new issue, record it immediately with a new id and fold it into the same loop. A larger area may be enumerated first for planning, but nothing sits unrecorded.
Phase C — Continue draining the ledger in severity order: S0 → S1 → S2 → S3, gates per §8. Batch S2/S3 by rule class.
Phase D (convergence) — After the ledger reaches zero open, run the discovery passes again (L4 + §5 sweep at minimum, plus the tiers touched by fixes). Fixes create new findings; this pass exists to catch them. Repeat D until a full sweep adds no new actionable task. Only then is the discovery loop closed.
Phase E — From a fresh clone on one independent host: clean build (zero warnings vs baseline), full suite green, coverage report, scanners clean or waived in writing, zero production placeholders, ledger open count = 0 (all DONE or BLOCKED-with-owner).
=== 12. DEFINITION OF COMPLETE === Complete only when Phase E passes end-to-end in a single clean run on the independent host and the ledger's open count is exactly zero — every task DONE or BLOCKED with a named owner, and the final convergence sweep added no new tasks. Any open S0/S1/S2/S3 task means the audit is not complete, regardless of how minor it is. Re-read this prompt before declaring completion.

=== 13. REPORT FORMAT === [#<id> <n>/<total> | done:<d> open:<o> blocked:<b> new:<x>] <STATUS> — <title> Example: [#0037 37/212 | done:36 open:174 blocked:2 new:9] TEST — fix unchecked read() return in ingest.c:118 The run ends with open:0.