# TinyTitan audit — agent execution manual

**Revision:** 7b0901d75b3b1e79498874babf93218a617a69ac (original product-source baseline; line references refer to this SHA). **Branch:** audit/2026-11-26. **Manual revision:** 1.

**Status:** 29 known tasks; done:0, open:26, blocked:3. Discovery is incomplete. This is a detailed work specification, NOT an audit result or release approval. No implementation, test execution or product change is represented by this manual. Generated from the `manual` fields in AUDIT/ledger.json, the canonical ledger. Keep task numbers stable; do not maintain a competing status tracker in this document.

## Execution contract for the next agent

1. Read AUDIT/ledger.json, AUDIT/inventory.md, AUDIT/environment.md, AUDIT/baseline.md, AUDIT/tool-coverage.md, AUDIT/discovery-coverage.md, AGENTS.md and the original user audit specification before acting. Verify branch/head, current status and current source locations. Line numbers are anchors, not a substitute for reading full functions and callers. The task-list author did not run the proposed commands.
2. Current instruction is planning only. Begin code changes only when the repository owner authorizes remediation. Work on the existing audit branch; never write to main/master/release, force-push, rewrite history or delete branches/tags. No production DB, deployment, live profile/server, credential rotation or irreversible operations. Use synthetic fixtures and isolated scratch stores. For destructive-looking operations record exact command and rollback before action.
3. Establish required host/toolchain and Phase A baseline before any product change. Tasks 0001, 0006, 0002, 0009, 0003 are enabling gates (some overlap; run 0009 scans/proofs once and reference baseline). Avoid redundant history scan or tool matrix. Prioritize verified S0, then S1, S2, S3; dependency order may require a prerequisite first. Task 0007 spans discovery and convergence and remains open until complete; 0008 last. Current manual covers known tasks, not all future defects.
4. START means static condition demonstrated, not necessarily runtime impact reproduced. OPEN investigation means first prove reachable behavior or enforced invariant; do not invent bugs or make speculative changes. Infrastructure BLOCKED is not a pass. Reopen blocker when access obtained; record named owner, reason, attempts, at least two options. Severity on candidate tasks is provisional; escalation/downgrade requires written evidence and no scope reduction. Do not silently close candidates.
5. For each S0/S1 task record expected behavior and failing regression or static proof before fix, focused diff, targeted failing-before/passing-after test and full primary-host suite, cold self-audit plus affected scanners and baseline comparison, then focused commit `audit(<id>): <title>`. S2 behavior changes need tests; S3 mechanical sweep by rule. A test must assert real behavior, not merely no throw. No deletion/skipping/weakened assertions, broadened/empty catches, force unwrap/try!, ignore flags, strictness downgrade, @unchecked Sendable workaround or analysis exclusions.
6. Commands are suggested from the inspected CI and repository; verify exact filters, supported options, fixture availability and working directories on pinned host. Always record exit status, version, duration, warning count, pass/fail/skip and evidence path. Use deterministic synchronization and bounded timeouts; no arbitrary sleeps, no unsafe giant allocations or live data. If a test could hang pre-fix, isolate in subprocess with deadline.
7. On Mac A full regression normally includes `swift build -c release`, `swift build --build-tests`, `tools/lint.sh`, `swift test --no-parallel --enable-code-coverage`, `tools/coverage-export.sh`, plus affected example/dependency checks and actual CI Python/Node groups. For concurrency `swift test --no-parallel --sanitize=thread`; C ASan+UBSan full suite with verified instrumentation. On Linux B use pinned Node/lockfile and `npm test`, `npm run lint`, `npm run format:check` in plugin directory; affected integration checks on Mac. Record no-new-warnings and baseline delta. A focused command never substitutes for full suite. Repo-specific test filters shown below are intended to be checked against actual suite names; a zero-test match is failure.
8. Keep a per-task evidence record (suggested AUDIT/evidence/<id>.md or linked redacted artifacts). Record current HEAD, files/line anchors, expected invariant, reproduction command/output, tests added, before/after results, full-suite and scanner outputs, coverage/baseline delta, self-audit findings, commit SHA, residual risk. Store only redacted logs; never commit secrets. Link any affected multi-consumer contract tasks before parent closure. Do not commit test fixtures with real data.
9. On every milestone derive counts from ledger.json; no task status change without evidence. During discovery, new tasks increase count and are recorded; during remediation drain nonterminal count strictly decreases at milestones. Re-read source and check fix does not introduce production facade or regression. Repeat convergence, then independently verify Phase E. No release certification until original completion criteria actually met.
10. Update ledger.json as canonical; regenerate human views after edits. This manual describes the intended implementation, not an instruction to apply a design despite conflicting source evidence. If design needs changing, state why, update numbered task and tests, preserve scope.

## Priority and dependency map

| ID | Severity | Status at manual creation | Project | Prerequisites / related gates | Name |
|---|---|---|---|---|---|
| [0001](#task-0001) | S0 | BLOCKED | Root SwiftPM / Apple projects | — | Provide primary Xcode 27 / Swift 6.4 Apple Silicon execution host |
| [0002](#task-0002) | S0 | OPEN | Repository | 0001, 0006 | Complete Phase A inventory, two-hop contracts, trust boundaries and tier validation |
| [0003](#task-0003) | S0 | OPEN | Repository | 0001, 0002, 0006, 0009 | Produce current pinned baseline and tool coverage proofs |
| [0006](#task-0006) | S0 | BLOCKED | Repository | — | Obtain executable clone and pinned non-Apple toolchains |
| [0007](#task-0007) | S0 | OPEN | Repository | 0002, 0003, 0009 | Complete L0–L7, facade and unused-code audit, remediation and convergence |
| [0008](#task-0008) | S0 | BLOCKED | Repository | 0001, 0002, 0003, 0007, 0009 | Run Phase E once on an independent fresh-clone host |
| [0010](#task-0010) | S0 | START | ContinuityCore | 0003, 0009 | Checkpoint replacement ignores durability failures |
| [0004](#task-0004) | S1 | START | Embedded and plugin examples | 0003 | Enforce required Swift settings in standalone example packages |
| [0005](#task-0005) | S1 | START | Library demo | 0003 | Enforce Swift 6 and warnings-as-errors in direct swiftc demo builds |
| [0009](#task-0009) | S1 | OPEN | Repository | 0001, 0002, 0006 | Verify sanitizer and dependency/SAST/secret-scan coverage for every applicable language |
| [0011](#task-0011) | S1 | START | ContinuityCore | 0003, 0010 | Prevent actor reentrancy from clearing unsynced append accounting |
| [0012](#task-0012) | S1 | OPEN | ContinuityCore | 0003, 0010, 0011 | Establish serialization of append versus compaction |
| [0013](#task-0013) | S1 | START | ContinuityCore | 0003, 0010, 0011 | Propagate asynchronous journal durability failures |
| [0014](#task-0014) | S1 | START | ContinuityCore | 0003, 0010 | Distinguish corrupt complete journal records from torn tail |
| [0015](#task-0015) | S1 | OPEN | ContinuityCore | 0002, 0003 | Validate journal file type, ownership and permissions |
| [0016](#task-0016) | S1 | START | ContinuityCore | 0003 | Handle short pread during tail repair without truncating unread bytes |
| [0017](#task-0017) | S1 | START | TinyTitanServer | 0003, 0009 | Bound HTTP header buffering before decoder completes |
| [0018](#task-0018) | S1 | START | TinyTitanServer | 0003 | Bound stored response memory by bytes and chain size |
| [0019](#task-0019) | S1 | OPEN | TinyTitanServer | 0002, 0003, 0020 | Validate loopback trust boundary, Host and browser-origin protections |
| [0020](#task-0020) | S1 | START | TinyTitanServer | 0003 | Parse media type exactly instead of accepting application/json prefix |
| [0021](#task-0021) | S1 | OPEN | TinyTitanKernelsC | 0002, 0003, 0009 | Prove allocation products and BLAS narrowing safe at Swift–C boundary |
| [0022](#task-0022) | S1 | OPEN | TinyTitanKernelsC | 0002, 0003, 0009 | Prove C reader offset bounds and destruction versus active batch lifecycle |
| [0023](#task-0023) | S1 | START | DSH plugin | 0003, 0006 | Serialize catalog and configuration fields as safe YAML scalars |
| [0024](#task-0024) | S1 | START | DSH plugin | 0003, 0006, 0023 | Make settings rewrite atomic and detect concurrent updates |
| [0027](#task-0027) | S1 | OPEN | TinyTitanFormat | 0002, 0003, 0009 | Verify bounded GGUF metadata parser resource use and duplicate handling |
| [0028](#task-0028) | S1 | OPEN | TinyTitanFormat | 0002, 0003 | Verify overflow-safe compressed expert block-count arithmetic |
| [0029](#task-0029) | S1 | START | ContinuityCore | 0003, 0010 | Guard against zero-progress write loops |
| [0025](#task-0025) | S2 | START | DSH plugin | 0003, 0006 | Report malformed catalog rows rather than silently dropping them |
| [0026](#task-0026) | S2 | START | DSH plugin | 0003, 0006 | Require safe integer compaction headroom |

Dependencies are suggested execution prerequisites, not authorization to skip a higher-priority defect. Related tasks that share code retain separate evidence and closure gates. Infrastructure/verification gate 0009 precedes baseline even though classified S1. For 0008, its own open status is excluded from its pre-run zero-open check; it closes only after verification succeeds.

## Task specifications

<a id="task-0001"></a>

### Task 0001 — Provide primary Xcode 27 / Swift 6.4 Apple Silicon execution host

**Project / component:** Root SwiftPM / Apple projects  
**Location:** `Package.swift:1`  
**Severity / tier:** S0 / A  
**Status at manual creation:** BLOCKED  
**Category:** verification prerequisite  
**Primary host:** Unassigned Apple Silicon Mac  
**Prerequisites:** None beyond execution contract.  
**Evidence classification:** Infrastructure blocker  

#### 1. Detailed problem description

All Apple-platform production code depends on the prescribed compiler and frameworks (Metal, Accelerate, Darwin). API access is not a shell and a Linux compiler is not a substitute. CI names an xcode-27 runner, but neither access, actual Xcode selection, hardware, nor installed toolchain has been demonstrated. Without a primary Mac, no Apple baseline or regression result is admissible. The repository owner is a proposed blocker owner, not a confirmed assignment.

**Recorded evidence:** Local host is Linux x86_64; xcodebuild absent; direct GitHub clone fails DNS. Existing CI specifies xcode-27 but no shell/runner access was established.

**Existing blocker:** Required primary Mac unavailable to this session.  
**Proposed owner:** Pummelchen (repository owner; assignment proposed)  
**Attempted:** Checked uname, xcodebuild and git ls-remote; read CI configuration.  
**Human options:** Provide access to an isolated Apple Silicon development Mac with verified Xcode 27 / Swift 6.4.; Run the audit in a local coding agent on such a Mac; retain logs in AUDIT.

#### 2. Proposed solution / implementation sequence

1. Obtain explicit access to an isolated Apple Silicon development host, or have the owner execute the audit there with a coding agent. Do not provision a VPS, touch an existing live server, or install packages globally without permission.
2. Verify the active Xcode is version 27 and its selected Swift compiler is 6.4.x, using the actual environment that will build. If more than one Xcode is installed, select a specific installation for this audit via DEVELOPER_DIR; record its path, not a global change.
3. Record host identifier, OS version, architecture, RAM, free scratch space, Xcode and Swift build identifiers, Git version, and tool paths in AUDIT/environment.md. Schedule at most one heavy job per 8 GB Mac. Assign Mac A as primary for Apple-integrated work.
4. Use an isolated checkout of the existing audit branch; verify remote URL and pinned source ancestry. Do not assume a runner label proves a working host. Keep credentials in the host's approved credential helper.

#### 3. Verification — regression and integration

1. Run `uname -sm`, `sw_vers`, `xcode-select -p`, `xcodebuild -version`, `xcrun --find swift`, `swift --version`, `xcrun swift --version`, `sysctl -n hw.memsize`, `df -h .` and `git --version`; archive exit codes and redacted output.
2. Confirm arm64 macOS, Xcode 27 and Swift 6.4.x from the active toolchain. If any mismatch, leave BLOCKED with exact reason and human options; do not run a substitute build.
3. Confirm the audit checkout resolves the recorded base and branch. Demonstrate scratch filesystem access and that no production endpoints or model stores are targeted.
4. A second Mac is tracked separately in 0008; do not count the same host as independent.

#### 4. Self-audit — cold review and closure

1. Cold-read environment record against actual command output; confirm no hostname or path secrets are exposed.
2. Recheck the selected toolchain at the start of baseline and final verification; ensure xcrun and swift resolve consistently.
3. Close this infrastructure task only after primary host is usable, not merely requested.

**Acceptance:** Complete the specific verification and self-audit steps above with recorded evidence; apply the global closure gates. A blocker or unrun check is not a passing result.

**Implementation status:** Not implemented by the manual author. See canonical ledger for later updates.

---

<a id="task-0002"></a>

### Task 0002 — Complete Phase A inventory, two-hop contracts, trust boundaries and tier validation

**Project / component:** Repository  
**Location:** `AUDIT/inventory.md:1`  
**Severity / tier:** S0 / A  
**Status at manual creation:** OPEN  
**Category:** audit gate  
**Primary host:** Unassigned Apple Silicon Mac  
**Prerequisites:** 0001, 0006  
**Evidence classification:** Audit gate / investigation  

#### 1. Detailed problem description

The existing inventory contains a complete Git path enumeration and manifest-derived graph, but 1,522 A / 998 C file assignments are provisional. A file category is not a source review. Two-hop target edges do not capture data schemas, runtime HTTP/IPC, plugin contracts, environment variables, filesystem persistence or native pointer ownership. Incorrect C categorization could hide credential handlers, migrations or executable production paths.

**Recorded evidence:** Tree and manifests enumerated; source-level contract mapping and individual classification not complete.

#### 2. Proposed solution / implementation sequence

1. Reconcile AUDIT/inventory.json with `git ls-tree -r --name-only` at the pinned source commit, then list additions since the audit base separately. Do not confuse generated audit documents with product source.
2. Enumerate all root and nested Package.swift targets and products, Node packages, direct swiftc scripts, C/Metal compilation paths, Python and shell entry points, CI/release scripts and configuration-only integrations. Record project, module, language, build mechanism, executable entry point and primary host.
3. For every production module derive direct dependencies from imports, manifest targets and actual calls, then expand exactly two hops. Add a separate contract table: producer, consumers, schema/protocol or format version, transport/storage, validation owner, failure policy, trust boundary, tests. Include server/CLI/DSH/OpenCode, SSDAI/GGUF/repack, ContinuityCore and memory, plugin registry, Swift–C and Metal ABI.
4. Assign tiers per actual behavior: untrusted parsing, credentials, network, persistence, privileged actions, native memory and shared contracts => A; other production => B; tests/docs/glue => C only if no A behavior. Unsure => A. Record the rationale for each module and any changed tier. No blanket classification based on extension.
5. Identify each deployable and its health/startup/shutdown/rollback surface. Commit inventory and tier table before product fixes.

#### 3. Verification — regression and integration

1. Compare tracked file sets programmatically: no missing/unclassified paths, no duplicate paths, and counts reconcile with the pinned Git tree. Explain legitimate generated/vendored files; do not exclude from required scans.
2. Cross-check root and nested package targets against resolved `swift package describe --type json` on the pinned Mac; independently inspect direct build scripts and Node package entry points.
3. Pick at least one example for each trust boundary and trace its producer to every consumer, including all consumers within two hops. Verify high-risk Python/shell tools are not C.
4. Check every contract has a named validation point and test location or an explicit missing-test task. Add numbered findings rather than burying them in inventory prose.

#### 4. Self-audit — cold review and closure

1. Cold-read all tier changes and the dependency graph; confirm uncertain modules remain A and no depth beyond two hops is claimed.
2. Verify inventory counts against a fresh tree enumeration and report actual inspection coverage separately.
3. Do not mark DONE until the inventory and contract mapping—not the entire later audit—are complete and committed.

**Acceptance:** Complete the specific verification and self-audit steps above with recorded evidence; apply the global closure gates. A blocker or unrun check is not a passing result.

**Implementation status:** Not implemented by the manual author. See canonical ledger for later updates.

---

<a id="task-0003"></a>

### Task 0003 — Produce current pinned baseline and tool coverage proofs

**Project / component:** Repository  
**Location:** `AUDIT/baseline.md:1`  
**Severity / tier:** S0 / A  
**Status at manual creation:** OPEN  
**Category:** audit gate  
**Primary host:** Unassigned Apple Silicon Mac  
**Prerequisites:** 0001, 0002, 0006, 0009  
**Evidence classification:** Audit gate / investigation  

#### 1. Detailed problem description

No current executable baseline exists. Prior comments about clean historical builds are not proof for this revision. Swift manifest declarations and lint config are not proof that actual compiler invocations reject violations. The missing baseline prevents meaningful regression claims. If a baseline is red, it must be recorded as red and translated into numbered tasks, not adjusted to pass.

**Recorded evidence:** No current build, test, coverage, dependency scan, full-history scan or negative control executed.

#### 2. Proposed solution / implementation sequence

1. Use the primary hosts from 0002, the pinned toolchain from 0001/0006 and scanner plan from 0009. Start from a clean isolated worktree at the audited source revision, with audit-only documentation separately identified. Use the same configuration and commands for subsequent regression comparisons.
2. Record resolved Swift compiler flags for every target via verbose builds or resolved Xcode settings when Xcode projects exist. SwiftPM requires explicit v6 and warnings-as-errors; strict concurrency is inherent in Swift 6 mode but must be proven by a failing snippet. Verify C99 and every required warning flag in actual invocations. Cover independent examples and direct compiler demos.
3. Run the repo CI-equivalent commands on the assigned primary hosts: `swift build -c release`, `swift build --build-tests`, `tools/lint.sh`, `tools/embedded-dependency-check.sh`, `tools/plugin-dependency-check.sh`, `swift test --no-parallel --enable-code-coverage`, `tools/coverage-export.sh`; inspect CI for its full Python and JS test groups and execute them rather than substituting a tiny smoke suite. Preserve pipe exit status when capturing logs.
4. Record per-project build errors/warnings, pass/fail/skip counts, coverage numerator/denominator and percentage, linter findings by rule, dependency/CVE and SAST results, one full-history secret scan and sanitizer results. Redact secrets before writing any logs. Missing metrics are NOT RUN, not zero.
5. For every delegated rule, create an isolated scratch violation using the exact production config. Required minimum: Swift non-Sendable actor/Task transfer, SwiftLint force unwrap, C implicit declaration and GNU nested function/typeof, Ruff bare except. Add rule-specific probes for other delegated checks. Record snippet, command, actual nonzero status and diagnostic; remove scratch inputs safely. If silent, repair the config as a numbered task or restore manual inspection.
6. Commit AUDIT/baseline.md and AUDIT/tool-coverage.md, recording red findings as distinct ledger tasks and preserving the initial yardstick.

#### 3. Verification — regression and integration

1. Check the build command actually fails on the intentional violations, not on missing dependencies or unrelated errors; after restoring scratch input the control builds/lints clean or matches the known baseline.
2. Reconcile summary counts with raw logs and coverage export; ensure tests marked skipped are listed with reasons, not treated as passing.
3. Verify scanner history scope and versions; don't repeat the full-history scan unless history changes or the initial run was invalid. Cross-check root vs independent project flags.
4. Ensure build settings for 0004/0005 are recorded as current deficiencies rather than silently repaired before baseline.

#### 4. Self-audit — cold review and closure

1. Reproduce a representative command and deliberate violation independently on the primary host, preserving the original baseline rather than replacing it.
2. Confirm all configured gates run with original strictness and no source exclusions were added. Baseline task can be DONE with accurately recorded red metrics and numbered issues; release cannot be certified until those issues satisfy their own gates.
3. Confirm no logs contain credential values and no temporary probes remain in source.

**Acceptance:** Complete the specific verification and self-audit steps above with recorded evidence; apply the global closure gates. A blocker or unrun check is not a passing result.

**Implementation status:** Not implemented by the manual author. See canonical ledger for later updates.

---

<a id="task-0006"></a>

### Task 0006 — Obtain executable clone and pinned non-Apple toolchains

**Project / component:** Repository  
**Location:** `AUDIT/environment.md:1`  
**Severity / tier:** S0 / A  
**Status at manual creation:** BLOCKED  
**Category:** verification prerequisite  
**Primary host:** Linux x86_64 primary for standalone Node and portable tooling  
**Prerequisites:** None beyond execution contract.  
**Evidence classification:** Infrastructure blocker  

#### 1. Detailed problem description

The prior session had Linux x86_64, Node 22.16.0 instead of plugin requirement >=24.9, direct GitHub DNS failure, and missing pinned tools. Connector source reads do not provide clone, process execution, history or package downloads. This blocks Node and other portable checks. Python 3.13.5 was observed but the current host still needs verification.

**Recorded evidence:** Linux local Git clone cannot resolve github.com; Node v22.16.0 does not meet plugin >=24.9; Ruff, gitleaks, shellcheck, pip-audit and CodeQL unavailable.

**Existing blocker:** No networked execution environment or installed pinned scanners available.  
**Proposed owner:** Pummelchen (repository owner; assignment proposed)  
**Attempted:** Checked tool availability and public git ls-remote; connector supports source/API access, not execution.  
**Human options:** Provide a networked isolated Linux host with repository checkout and pinned versions.; Use a preconfigured runner or local coding agent with the pinned dependencies; provide full results.

#### 2. Proposed solution / implementation sequence

1. Obtain a networked isolated Linux execution host or approved runner for standalone Node/portable work. Confirm repository fetch access without printing credentials. Do not provision VPS without owner approval.
2. Pin one exact patch release of Node 24 satisfying >=24.9, one Python 3.13 patch, and the repository's configured ESLint 10.11.0, Prettier 3.9.9, Ruff 0.16.7, ShellCheck 0.11.0, Gitleaks 8.30.1, pip-audit 2.10.1; confirm actual compatibility and availability. Root Apple tasks remain on Mac A.
3. Prefer existing lockfiles and verified official distributions; log install method, version, checksum where published, host and command. Use isolated venv/npm workspace; do not install arbitrary global packages or change project lockfiles merely to install audit tools.
4. Verify source revision and full Git history are accessible; keep the audit branch only. Establish package registry network access and deterministic dependency install using lockfile (npm ci). If a required tool cannot be installed, retain explicit BLOCKED owner, attempts and alternatives.

#### 3. Verification — regression and integration

1. Record `uname -sm`, `git --version`, `node --version`, `npm --version`, `python3 --version` and version outputs for each installed tool. Run `git ls-remote` against repo without secrets in command line; verify `git rev-parse --is-shallow-repository` is false for history scan clone.
2. From scratch plugin directory run `npm ci`, `npm test`, `npm run lint`, `npm run format:check` using pinned Node; these are availability smoke checks, full baseline belongs to 0003. Ensure no lockfile drift.
3. Confirm Python venv imports and pinned tools execute their help/version; capture exact install failure if unavailable.
4. Demonstrate source checkout is isolated, no production credentials or paths, and tool outputs can be captured redacted.

#### 4. Self-audit — cold review and closure

1. Review environment record versus actual executable paths; verify no tool matrix or host-dependent standard substitution.
2. Confirm Node patch pinned explicitly rather than CI major-only selection and versions match repo gates.
3. Leave blockers terminal only when truly inaccessible and assign owner/alternatives; reopen when host arrives.

**Acceptance:** Complete the specific verification and self-audit steps above with recorded evidence; apply the global closure gates. A blocker or unrun check is not a passing result.

**Implementation status:** Not implemented by the manual author. See canonical ledger for later updates.

---

<a id="task-0007"></a>

### Task 0007 — Complete L0–L7, facade and unused-code audit, remediation and convergence

**Project / component:** Repository  
**Location:** `AUDIT/ledger.md:1`  
**Severity / tier:** S0 / A  
**Status at manual creation:** OPEN  
**Category:** audit gate  
**Primary host:** Unassigned Apple Silicon Mac  
**Prerequisites:** 0002, 0003, 0009  
**Evidence classification:** Audit gate / investigation  

#### 1. Detailed problem description

The 29-entry ledger is not exhaustive: selected source excerpts were reviewed but entire inference engines, Metal, most library/runtime, memory and repack paths, CI/release and full scanner coverage remain outstanding. There is no basis for treating this list as complete or a clean bill of health. Original execution specification requires discovery, severity-order closure and convergence; the user's current request is only to document, not perform those changes.

**Recorded evidence:** Discovery only. Tier A source lines and scanners have not been exhaustively evaluated; no substantive remediation performed.

#### 2. Proposed solution / implementation sequence

1. Resume from this ledger and AUDIT/discovery-coverage.md; never replace existing IDs. Establish a coverage matrix with one row per module and columns L0–L7, tier, contracts, facade review, unused-code scope, scanner proof, reviewed revision and evidence. Record not-applicable reasons and actual inspection coverage.
2. L0 once: reproducible builds, dependency pins, lockfiles, CI security, ignored artifacts, full-history secrets. L1 A+B: architecture, cycles, contracts, config/error boundaries. L2 A manual: API invariants, lifetime, cancellation/retries/queues. L3 A line-by-line and B sampled: arithmetic, logic, encoding, timezone, return paths. L4 A manual+SAST/B scanner/C secrets: authorization, injection, file/URL trust, native memory, LLM actions. L5 A/hot paths: allocations, I/O, bounded queues and latency. L6 A failure/boundary/critical coverage and B critical gate. L7 each service: health, startup validation, shutdown and rollback.
3. Prioritize remaining modules: (a) ContinuityCore + TinyTitanMemory/MemoryTool, (b) HTTP handler/mapper/router/backend, (c) format/repack/remote/model install and credentials, (d) C/Metal and Swift unsafe seams, (e) engine variants, legacy/core tokenizer and runtime, (f) library/CLI/plugin registry, (g) DSH/OpenCode, (h) tools/release/benchmark. Identify migration or credential scripts as A regardless of language.
4. Hunt production facades with actual reachability, not TODO-marker counts. Review unused code in A and code orphaned by audit changes only. Never delete unless reachability proof and an authorized focused task; prefer deprecation/removal ticket.
5. Record each evidence-backed new finding immediately as next ID, with precise location, reproduction or static proof, proposed fix, verification, affected consumers and severity. Batch mechanical S3 by rule. Record possible issues as OPEN investigation rather than fabricating confirmed bugs.
6. When authorized to remediate, drain verified S0→S1→S2→S3 with gates; prerequisites may run before severity order. Repeat L4+facade and touched tiers until a complete sweep adds no actionable task. Leave unresolved or unreviewed items open; no artificial closure.

#### 3. Verification — regression and integration

1. Reconcile module coverage matrix against inventory (no missing Tier A modules); spot-check review evidence against source lines and commit IDs. Verify scanner delegation has negative controls.
2. Inspect task IDs, required fields, duplicate/linked tasks, severity ordering and task counts. New findings reopen convergence; record counts at each milestone and explain discovery growth before drain phase.
3. Check every production facade candidate has reachability proof and every critical contract has multi-consumer test or explicit numbered missing-test task.
4. After future authorized fixes, perform full re-discovery and require zero newly actionable findings; run Phase E only after open count is zero under original terminal-state definition.

#### 4. Self-audit — cold review and closure

1. Independent cold review of risk-tier coverage, especially native boundaries, persistent writes and model-driven privileged actions. Confirm no omitted file is represented as reviewed.
2. Check every S0/S1 closure has START/PROGRESS/TEST/AUDIT evidence and focused commit; no baseline regression or weakened checks.
3. Do not mark this meta-task complete merely because the currently known 29 entries have been processed.

**Acceptance:** Complete the specific verification and self-audit steps above with recorded evidence; apply the global closure gates. A blocker or unrun check is not a passing result.

**Implementation status:** Not implemented by the manual author. See canonical ledger for later updates.

---

<a id="task-0008"></a>

### Task 0008 — Run Phase E once on an independent fresh-clone host

**Project / component:** Repository  
**Location:** `AUDIT/ledger.md:1`  
**Severity / tier:** S0 / A  
**Status at manual creation:** BLOCKED  
**Category:** verification prerequisite  
**Primary host:** Unassigned Apple Silicon Mac  
**Prerequisites:** 0001, 0002, 0003, 0007, 0009  
**Evidence classification:** Infrastructure blocker  

#### 1. Detailed problem description

Phase E must run once from a fresh clone on an independent host after convergence. Neither access nor a complete primary-host audit exists. Reusing Mac A, cached artifacts, a historical CI job or a partial rerun would not satisfy independence or the one-clean-run gate. Owner assignment remains proposed.

**Recorded evidence:** No independent execution host available; earlier phases not complete.

**Existing blocker:** Independent host access and completed primary audit unavailable.  
**Proposed owner:** Pummelchen (repository owner; assignment proposed)  
**Attempted:** Read existing CI workflow; no independent execution performed.  
**Human options:** Use a second isolated Apple Silicon Mac with Xcode 27/Swift 6.4 and pinned tools after Phase D.; Use an independently provisioned macOS CI runner after Phase D, with complete fresh-clone logs and all required gates.

#### 2. Proposed solution / implementation sequence

1. Arrange a second isolated Mac C with Xcode 27/Swift 6.4, appropriate hardware, pinned tools and adequate disk; do not provision without authorization. For portable checks on final host use the same pinned versions, not a new standard.
2. Wait for 0007 convergence and closure of all other nonterminal tasks (excluding the Phase E gate itself). Record exact audit branch HEAD; fetch a fresh clone on Mac C using approved credentials and ensure no reused build outputs or model stores. Use synthetic/test model fixtures or explicitly approved immutable test artifacts.
3. Execute the complete baseline-equivalent build, test, lint, coverage, required sanitizers and applicable dependency/SAST/secret checks in a single clean run. The full-history scan is once per history under original specification; verify its previous evidence still covers final history and scan new commits if needed rather than pretending to rerun historical coverage.
4. Check final task ledger, warnings, no placeholders, no regression and documented scanner findings/waivers (no waiver of S0 or bypass of checks). Generate final report from ledger.json. Do not merge or deploy in this task.

#### 3. Verification — regression and integration

1. Record host independence, `git rev-parse HEAD`, clean checkout state, exact tool versions, command exit codes, complete test counts and coverage. Verify each required gate ran; missing job = not passed.
2. Compare final metrics to 0003 and linked justified changes. Confirm no skipped/disabled checks, no new actionable convergence finding and zero nonterminal remediation tasks.
3. If any step fails, open a new numbered finding and return to remediation/convergence; subsequent final attempt must again be a single clean fresh-clone run.
4. Close 0008 only after the complete successful run; then ledger open count must be exactly zero, with any blocked owner and options explicit. No release certification from an infrastructure BLOCKED state.

#### 4. Self-audit — cold review and closure

1. Cross-check independence from Mac A, source SHA, actual tool versions and no stale caches. Review final report against raw redacted artifacts.
2. Confirm the final pass did not mutate live systems and no default-branch merge occurred. Any PR follows owner review after Phase E.
3. Document decision and reasons if Phase E cannot complete.

**Acceptance:** Complete the specific verification and self-audit steps above with recorded evidence; apply the global closure gates. A blocker or unrun check is not a passing result.

**Implementation status:** Not implemented by the manual author. See canonical ledger for later updates.

---

<a id="task-0010"></a>

### Task 0010 — Checkpoint replacement ignores durability failures

**Project / component:** ContinuityCore  
**Location:** `sources/ContinuityCore/Persistence/Journal.swift:336-372`  
**Severity / tier:** S0 / A  
**Status at manual creation:** START  
**Category:** L2/L4 persistent data  
**Primary host:** Apple Silicon Mac A  
**Prerequisites:** 0003, 0009  
**Evidence classification:** static finding; runtime impact untested  

#### 1. Detailed problem description

FileJournal.writeCheckpoint writes a replacement then calls F_FULLFSYNC; if it fails the fallback fsync result is ignored (line 359). It then renames over the original journal, attempts to open and sync the parent directory, ignores failure and returns success (367–372). A failed replacement-data barrier must not be followed by replacement of the only durable original. A directory sync failure after rename has different semantics: replacement may be visible but durability uncertain. Existing code conflates these states. Data loss is conditional on I/O failure/crash; no runtime reproduction yet.

**Recorded evidence:** writeCheckpoint ignores fallback fsync result at 359; ignores directory open failure and directory fsync result at 367–371, then returns success. A failed data barrier does not prevent rename of the replacement.

#### 2. Proposed solution / implementation sequence

1. Read Journal.swift end-to-end, JournalError/JournalTypes, callers of compact/shutDown and existing JournalTests. Identify exact commit point (rename) and descriptor ownership. Coordinate design with 0011–0014, 0015 and 0029; keep focused task scope.
2. Introduce a small injectable syscall/filesystem interface or equivalent deterministic test seam for data sync, directory open/sync, rename, close and cleanup. Production default must call real syscalls. Capture errno immediately; do not use broad catch/try? to hide durability errors.
3. Use the existing strong Darwin barrier with checked fallback where supported. If both fail before rename, close and clean temporary file, return typed failure, keep original bytes/inode and append descriptor usable. Do not rename failed temp. If directory open/sync fails after rename, return explicit post-commit/uncertain-durability outcome; do not pretend rollback happened or blindly delete the new checkpoint.
4. Ensure compact reconciles descriptor with visible inode even when post-rename directory barrier failed; no later append should target an unlinked old inode. Preserve sidecar lock, O_NOFOLLOW and permissions. Decide whether directory sync unsupported is a separately documented policy, not unconditional success.
5. Propagate result through ContinuityCore/TinyTitanMemory callers and diagnostics. Add linked tasks for contract changes in every consumer; do not expose raw journal data in logs.

#### 3. Verification — regression and integration

1. Add deterministic `tests/ContinuityCore/JournalTests.swift` fault tests that fail on original code: strong sync + fallback fail before rename (old contents unchanged), fallback success (new checkpoint valid), parent open failure and parent sync failure after rename (failure/uncertainty surfaced, append targets visible file).
2. Exercise rename failure, temp cleanup failure, reopen failure and successful path; verify exclusive lock survives and replay yields expected records. Compare file bytes and inode identity where relevant. Test in a fresh temporary directory, never live memory.
3. Run `swift test --no-parallel --filter JournalTests`, full ContinuityCore and TinyTitanMemory test targets as identified by manifest, then `swift test --no-parallel --enable-code-coverage`, `tools/lint.sh`, and relevant TSan/SAST gates. Run crash/reopen tests in subprocess scratch fixture if feasible.
4. Preserve baseline warnings and language settings. A successful return must imply documented durability guarantees; after failure, caller must observe failure and recovery must not lose acknowledged records.

#### 4. Self-audit — cold review and closure

1. Cold-read every return path before/after rename; ensure both data and directory barrier outcomes checked and descriptor never closed/reused unsafely. Verify cleanup errors cannot mask original error.
2. Inspect regression tests for pre-fix failure, deterministic seams, actual disk outcomes and no sleeps or weakened assertions.
3. Re-read related tasks before DONE; do not close 0011/0012/0013 merely because this fix touches common code. Record focused commit and redacted logs.

**Acceptance:** Inject file and directory sync errors; preserve old journal before successful data sync; report uncertain durability after rename. Verify replay and crash recovery.

**Implementation status:** Not implemented by the manual author. See canonical ledger for later updates.

---

<a id="task-0004"></a>

### Task 0004 — Enforce required Swift settings in standalone example packages

**Project / component:** Embedded and plugin examples  
**Location:** `examples/embedded/Package.swift:15; examples/plugin/Package.swift:11`  
**Severity / tier:** S1 / C  
**Status at manual creation:** START  
**Category:** language-standard configuration  
**Primary host:** Unassigned Apple Silicon Mac  
**Prerequisites:** 0003  
**Evidence classification:** Static condition; runtime impact untested  

#### 1. Detailed problem description

The standalone examples/embedded and examples/plugin manifests use tools version 6.4, but do not explicitly carry the root package's shared warnings-as-errors policy or `swiftLanguageModes: [.v6]`. Tools version may default to Swift 6, so do not describe absence of an explicit setting as proof of actual Swift 5 compilation. The demonstrated gap is inconsistent declared enforcement and lack of negative controls for all example targets (EmbeddedDemo, EchoPlugin, PluginDemo).

**Recorded evidence:** Both example manifests require tools version 6.4 but do not declare explicit Swift 6 language mode or warnings-as-errors; tools version alone does not establish warning enforcement. Expected: all example targets enforce the same mode and warnings-as-errors, proven with negative controls.

#### 2. Proposed solution / implementation sequence

1. Inspect both manifests and root tinytitanLanguageStandard plus CI consumer scripts. Record actual example compiler invocation first.
2. Declare `swiftLanguageModes: [.v6]` in each example package and apply warnings-as-errors to every target through a shared local setting array; consider explicitly matching relevant root upcoming features if they are applicable and supported. Do not import root manifest code or change Swift tools version.
3. Use Swift 6 complete concurrency enforcement; verify rather than add redundant flags that the compiler rejects. Preserve product dependency declarations, package-relative paths and ability to be built out of tree.
4. If stronger settings surface warnings/concurrency defects, open linked numbered tasks and correct their root causes; no suppression, unsafe Sendable workaround or mode downgrade.

#### 3. Verification — regression and integration

1. On Mac A, run `swift build --package-path examples/embedded` and `swift build --package-path examples/plugin` with verbose logs and the two existing dependency-check scripts.
2. In scratch copies introduce a compiler-warning probe (e.g. `#warning`), then a proven non-Sendable crossing actor/Task probe in each example target; confirm the relevant compiler invocation fails for the intended reason. Remove probes and confirm all targets build.
3. Run actual example integration/registration checks in `tools/embedded-dependency-check.sh` and `tools/plugin-dependency-check.sh` and their prescribed isolated prerequisites; no live server.
4. Run the full primary-host suite and `tools/lint.sh`; compare warnings and consumer behavior with 0003.

#### 4. Self-audit — cold review and closure

1. Review both manifests target by target and verify no target escaped settings. Inspect resolved compiler arguments for v6 and warnings-as-errors.
2. Check new flags do not change supported dependency/link behavior; ensure deliberate test probes are absent from committed source.
3. Verify focused diff contains only task-related settings and justified linked fixes.

**Acceptance:** Complete the specific verification and self-audit steps above with recorded evidence; apply the global closure gates. A blocker or unrun check is not a passing result.

**Implementation status:** Not implemented by the manual author. See canonical ledger for later updates.

---

<a id="task-0005"></a>

### Task 0005 — Enforce Swift 6 and warnings-as-errors in direct swiftc demo builds

**Project / component:** Library demo  
**Location:** `examples/library-demo/build-static.sh:46; examples/library-demo/build-dynamic.sh:52`  
**Severity / tier:** S1 / C  
**Status at manual creation:** START  
**Category:** language-standard configuration  
**Primary host:** Unassigned Apple Silicon Mac  
**Prerequisites:** 0003  
**Evidence classification:** Static condition; runtime impact untested  

#### 1. Detailed problem description

The direct swiftc build scripts for library-demo invoke `swiftc -O` with include/linker inputs but without explicit Swift language version or warnings-as-errors. They bypass SwiftPM manifest flags. This makes the release consumer example a distinct compiler-standard surface; the current active default mode has not been measured.

**Recorded evidence:** Both scripts invoke swiftc -O with no explicit -swift-version 6, strict concurrency or -warnings-as-errors. Expected: both demo build paths carry the strict standard; prove with negative controls and test both linkage forms.

#### 2. Proposed solution / implementation sequence

1. Inspect build-static.sh and build-dynamic.sh, their shared main.swift, dist make-flags.sh and generated swiftc-flags.txt. Record current invocation before changing it.
2. Use the pinned Xcode toolchain and add `-swift-version 6 -strict-concurrency=complete -warnings-as-errors` to both actual swiftc invocations (if a redundant strict flag is rejected by pinned compiler, retain v6 and document proof of complete enforcement). Keep `-D STATIC_LINK`/`-D DYNAMIC_LINK`, @flag-file paths, rpath and output placement unchanged.
3. Validate the distribution locally produced from audited code. If additional warnings arise, create linked tasks. Do not modify release artifacts or published distributions.
4. Ensure the script fails before producing a misleading success message when compilation fails. Use scratch copies of distribution and demo for destructive operations already present in script; inspect them before running.

#### 3. Verification — regression and integration

1. Build a scratch static and dynamic library distribution with the existing `tools/build_library.sh` documented invocation; no publishing. Execute both demo build scripts against scratch dist.
2. Capture compiler arguments and run warning/non-Sendable negative controls in scratch demo main.swift; each script must fail nonzero for expected reason; restore and rebuild.
3. Inspect static/dynamic artifacts with appropriate macOS binary tooling (e.g. `otool -L`); verify dynamic rpath and resource bundle discovery, and run a bounded demo with a synthetic or approved local test model if required.
4. Run the full primary suite and lint; ensure no untracked generated binary, library, or probe committed.

#### 4. Self-audit — cold review and closure

1. Compare the two scripts to ensure flags cannot diverge, and confirm the generated flag-file cannot silently override strictness.
2. Review shell quoting and cleanup within touched code; preserve all existing error handling and tests.
3. Record exact demo smoke commands and test-model prerequisites; if unavailable, do not claim the runtime demo passed.

**Acceptance:** Complete the specific verification and self-audit steps above with recorded evidence; apply the global closure gates. A blocker or unrun check is not a passing result.

**Implementation status:** Not implemented by the manual author. See canonical ledger for later updates.

---

<a id="task-0009"></a>

### Task 0009 — Verify sanitizer and dependency/SAST/secret-scan coverage for every applicable language

**Project / component:** Repository  
**Location:** `AUDIT/environment.md:1`  
**Severity / tier:** S1 / A  
**Status at manual creation:** OPEN  
**Category:** audit gate  
**Primary host:** Unassigned Apple Silicon Mac  
**Prerequisites:** 0001, 0002, 0006  
**Evidence classification:** Audit gate / investigation  

#### 1. Detailed problem description

CI config declares Swift TSan, CodeQL, Gitleaks, pip-audit and npm audit, but no current scan results were obtained. ASan+UBSan full suite for C and actual language coverage for Swift, C, JS, Metal, Python and shell need to be established; do not assume a scanner supports every language. Swift dependency CVE coverage and exact tool pins need evidence. This is a gate, not a verified vulnerability.

**Recorded evidence:** CI declares Swift TSan, pip-audit, npm audit, gitleaks and CodeQL. Full C ASan+UBSan suite and applicable Swift/C/JS/Metal/Shell tool coverage not established; no scan output from current revision.

#### 2. Proposed solution / implementation sequence

1. Inventory each language and actual production tier. Map formatter, linter, dependency/CVE scanner, required type checker/compiler, and A-tier SAST/sanitizer tools. Use one pinned version per language; record unsupported tools as BLOCKED and restore manual coverage where a tool cannot detect a class.
2. Inspect `.github/workflows/ci.yml`, `.github/workflows/codeql.yml`, `.github/codeql/codeql-config.yml`, `.gitleaks.toml`, `Package.resolved`, plugin package-lock and benchmark/requirements. Determine what CodeQL actually analyzes and its exclusions/build paths; do not claim Metal or C coverage merely from a Swift scan.
3. Run repo-configured secret scan over full history once using pinned Gitleaks with redaction enabled; disable verbose output that could expose credentials. Store only path/line and redacted finding category. Do not rotate or rewrite history. Use only safe local scratch negative controls, never a real secret.
4. Run `pip-audit -r benchmark/requirements.txt`, `npm audit` for all lockfiles and an appropriate pinned Swift dependency advisory check; identify exact package/version/advisory/reachability. Add numbered tasks for actual findings, not hypothetical CVEs.
5. Run full C-containing test suite with ASan + UBSan using Xcode/SwiftPM-supported instrumentation. Confirm both C compilation and linkage contain intended sanitizer flags and execute the tests, not merely compile. If combined mode unsupported, use separate full runs and document evidence; no Valgrind duplicate. Run `swift test --no-parallel --sanitize=thread` on concurrency-heavy targets per repository CI and record results.
6. For each delegated check record deliberate safe negative controls with expected diagnostics. For Metal use supported compiler validation and GPU validation/testing plus manual review where no equivalent scanner exists. Keep primary-host constraints.

#### 3. Verification — regression and integration

1. Inspect sanitizer build logs and binary linkage and run a controlled scratch violation to prove each sanitizer catches its class. Confirm full suite pass/fail/skip counts and log locations.
2. Verify dependency audit coverage includes resolved versions rather than only direct manifest ranges; reconcile each finding with a task ID. Do not treat network/API failure as zero vulnerabilities.
3. Verify CodeQL/SAST analyzed languages and source coverage, redacted history scan scope, rule versions, lint negative controls, and no exclusions added to make a gate green.
4. Check every scanner gap has a named owner or manual coverage and exact recorded limitations; findings S0/S1 for sanitizer failures.

#### 4. Self-audit — cold review and closure

1. Cold-read tool matrix against inventory, actual outputs and 0003 coverage proofs. Confirm no tool is delegated unproven checks.
2. Verify no secret values or raw sensitive logs are committed. Compare before/after scanner results only after authorized fixes.
3. Close gate only when required coverage is demonstrated or explicitly terminal BLOCKED with owner and options under original rules.

**Acceptance:** Complete the specific verification and self-audit steps above with recorded evidence; apply the global closure gates. A blocker or unrun check is not a passing result.

**Implementation status:** Not implemented by the manual author. See canonical ledger for later updates.

---

<a id="task-0011"></a>

### Task 0011 — Prevent actor reentrancy from clearing unsynced append accounting

**Project / component:** ContinuityCore  
**Location:** `sources/ContinuityCore/Persistence/Journal.swift:132-150,188-207`  
**Severity / tier:** S1 / A  
**Status at manual creation:** START  
**Category:** L2 concurrency  
**Primary host:** Apple Silicon Mac A  
**Prerequisites:** 0003, 0010  
**Evidence classification:** static finding; runtime impact untested  

#### 1. Detailed problem description

performBarrier captures a descriptor, awaits a blocking queue, and then sets pendingSinceSync=0 if descriptor number is unchanged. Actor methods can be reentrant at await: a later append may occur after the actual flush but before completion resumes on the actor. That write is not necessarily durable yet can be cleared from accounting, potentially preventing a future flush. Descriptor equality is also not a generation identity if descriptors are reused. Existing comment that appends queue behind a suspended actor is not an enforceable guarantee.

**Recorded evidence:** performBarrier awaits a blocking queue then sets pendingSinceSync to zero if descriptor unchanged. Actor can accept append while suspended; those records may be written after barrier syscall but before completion. Comment claiming appends queue behind the await is not a Swift actor guarantee.

#### 2. Proposed solution / implementation sequence

1. Map append(single/batch), afterAppend, sync, performBarrier, settleBarrier, compact, truncate and shutdown. Determine actual ordering and concurrent callers; pair design with 0012 and 0013.
2. Replace reset-all bookkeeping with monotonic append and durable generations or a captured pending watermark. Capture watermark and stable descriptor generation before dispatch; a successful barrier advances durable watermark only to the captured point. Later writes remain pending even if the OS happened to flush them.
3. Serialize barrier operations or coalesce with explicitly owned in-flight state; distinguish task cancellation from syscall completion. Keep descriptor lifetime valid until syscall ends; use generation identity not only numeric fd. Ensure newer writes schedule a further barrier and max latency cannot be postponed.
4. Use actor-owned state or existing synchronization primitives; avoid @unchecked Sendable additions or blocking cooperative executor. Handle counters/overflow with deliberate strategy and preserve single/batch semantics and error propagation.

#### 3. Verification — regression and integration

1. Build a deterministic controllable barrier seam: append A, start barrier, signal actual flush completion but hold actor continuation, append B, then resume A barrier. Assert B still pending and a subsequent barrier clears it. Original code should fail the assertion.
2. Test overlapping sync requests, batch sizes, failed barrier retaining pending state, cancellation, descriptor replacement and generation rollover boundary if applicable. Avoid sleep-based synchronization; use explicit continuations/latches.
3. Retain tests `theBarrierLandsOnceAppendsStop`, `aBusyWriterCannotPostponeDurabilityForever`, `explicitSyncIsImmediate` and `compactionWaitsForAnInFlightBarrier`. Run `swift test --no-parallel --filter JournalTests`, full suite, lint and `swift test --no-parallel --sanitize=thread`.
4. Verify public diagnostics count is exact at each controlled ordering and no flush task leaks. Compare warnings and latency/behavior with baseline; no numerical/model change needed.

#### 4. Self-audit — cold review and closure

1. Cold-read every suspension point; prove invariant durableGeneration <= appendedGeneration and pending equals difference. Check replacement/close cannot race in-flight barrier.
2. Review test scheduling independence; test must fail on old code for the targeted interleaving, not pass probabilistically.
3. Ensure 0012 remains independently tested for actual append preservation across rename.

**Acceptance:** Deterministic barrier/append interleaving test; count only records captured by completed durability barrier, preserve newer pending writes.

**Implementation status:** Not implemented by the manual author. See canonical ledger for later updates.

---

<a id="task-0012"></a>

### Task 0012 — Establish serialization of append versus compaction

**Project / component:** ContinuityCore  
**Location:** `sources/ContinuityCore/Persistence/Journal.swift:294-329,105-124`  
**Severity / tier:** S1 / A  
**Status at manual creation:** OPEN  
**Category:** L2 concurrency / data loss risk  
**Primary host:** Apple Silicon Mac A  
**Prerequisites:** 0003, 0010, 0011  
**Evidence classification:** investigation candidate; not a confirmed defect  

#### 1. Detailed problem description

Investigation, not a confirmed reachable production defect. compact awaits settleBarrier and then async replacement; append is independently actor-callable and writes current descriptor. If append runs during replacement or between snapshot capture and reopening, it may be written to old inode and lost when checkpoint replaces path. External serialization could prohibit this; trace every caller before fixing. A snapshot taken before compaction admission can itself be stale even if I/O is serialized.

**Recorded evidence:** compact awaits a checkpoint replacement on a blocking queue; append writes to existing descriptor. Unless every caller provides external serialization, an append during suspension may land on the old inode and disappear after reopen. Caller contract not yet established.

#### 2. Proposed solution / implementation sequence

1. Trace FileJournal compact/append callers in ContinuityCore and TinyTitanMemory, including tasks, public protocols, tests and direct API use. Document exactly which actor/lock owns the logical snapshot and journal ordering, and whether caller suspension allows concurrent calls.
2. Build a deterministic compact seam that pauses before rename and before descriptor reopen. If reachable, classify actual data-loss path S0 and record a severity update, not a new duplicate task. If not reachable, document the enforced invariant and add regression coverage; don't invent a fix.
3. Prefer one explicit serialization contract for append/compact/truncate/shutdown, with nonblocking async admission and safe cancellation. Serialize snapshot capture with mutation or checkpoint watermark; either queue append until replacement is installed or replay a proven delta onto new journal. Do not hold locks across await or block executor.
4. Preserve ordering and sidecar lock, ensure no accepted append is dropped or duplicated, including when compaction fails pre/post rename or reopening fails. Link consumer changes if API contract changes. Avoid a fake success that skips compaction.

#### 3. Verification — regression and integration

1. Regression: append A, request checkpoint at known state, pause replacement, concurrently append B; resume, close/reopen and assert durable state contains A and B exactly once in intended order. Also test append while snapshot is being captured, batch append, competing compactions and shutdown.
2. Inject compaction pre-rename and post-rename failure; verify queued operations resolve correctly and no deadlock/old inode write. Prove no descriptor use-after-close under TSan.
3. If no defect exists due to caller contract, add a positive deterministic test proving enforced isolation and public API cannot bypass it; include all call sites in evidence. No speculative source change required.
4. Run JournalTests, memory/continuity integration, full suite and TSan; verify warnings and coverage unchanged or improved.

#### 4. Self-audit — cold review and closure

1. Read caller and callee together; distinguish actor isolation from serialization across suspension. Check snapshot watermark and append admission under every failure path.
2. Verify exact-once replay assertions, no time.sleep, no disabled test and no unbounded waiting.
3. Record proof of non-reachability or actual reproduced/fixed behavior; never mark DONE based only on a lock comment.

**Acceptance:** Trace all FileJournal callers; reproduce with controlled suspension; guarantee append preservation or explicitly enforce isolation.

**Implementation status:** Not implemented by the manual author. See canonical ledger for later updates.

---

<a id="task-0013"></a>

### Task 0013 — Propagate asynchronous journal durability failures

**Project / component:** ContinuityCore  
**Location:** `sources/ContinuityCore/Persistence/Journal.swift:140-146,393-400`  
**Severity / tier:** S1 / A  
**Status at manual creation:** START  
**Category:** L2 error propagation  
**Primary host:** Apple Silicon Mac A  
**Prerequisites:** 0003, 0010, 0011  
**Evidence classification:** static finding; runtime impact untested  

#### 1. Detailed problem description

Scheduled background barrier uses `try? await self.performBarrier()`; shutDown declared async throws also suppresses settleBarrier and performBarrier errors before closing descriptors. A storage failure can leave caller unaware, pending records unsynced, and shutdown apparently successful. Cancellation during sleep is expected and should be distinguished from I/O failure; do not report every normal cancellation as corruption.

**Recorded evidence:** Scheduled performBarrier uses try?; shutDown also discards errors from settleBarrier and performBarrier before closing and releasing lock, despite async throws signature.

#### 2. Proposed solution / implementation sequence

1. Trace diagnostics/degraded-state APIs and consumers of FileJournal sync/shutDown, including memory service shutdown. Define typed error or state for background durability failure, recovery and shutdown failure.
2. Change background task handling to explicitly catch cancellation only where intended, capture actual I/O failure in actor-owned observable state, preserve pending watermark and use bounded retry policy or surface failure until caller retries. No infinite retry or silent catch.
3. Ensure `shutDown` attempts flush, always releases resources in correct order even on error, and returns the original meaningful durability failure. If flush succeeds after an earlier background failure, define whether diagnostic remains recorded and when degraded state clears.
4. Use same barrier and descriptor lifetime policy as 0010/0011/0012. Update every affected caller to propagate/log safe error category and not claim persistence when failed. Avoid logging record contents.

#### 3. Verification — regression and integration

1. Inject persistent barrier failure and assert background failure becomes visible without dropping pending records; subsequent successful explicit sync clears pending with documented diagnostic behavior.
2. Test cancellation of scheduled sleep (no false failure), shutdown flush failure (throws, lock released, descriptors closed), successful shutdown/reopen and repeat shutdown according to contract. Ensure caller reports degraded memory or failure rather than normal success.
3. Add tests in JournalTests and relevant TinyTitanMemory failure suites; run targeted tests, full suite, TSan and lint on Mac A.
4. Compare baseline failures and warnings; no swallowed I/O error remains on touched path.

#### 4. Self-audit — cold review and closure

1. Trace error from syscall to public caller, including fallback, cancellation and cleanup. Verify cleanup cannot overwrite original error or falsely clear state.
2. Cold-read retries, scheduled task ownership and shutdown admission; ensure no retry loop continues after close.
3. Check logs contain no record contents or credentials and negative tests fail before fix.

**Acceptance:** Fault inject fsync errors; expose degraded status and shutdown error while guaranteeing resource cleanup; verify retry and pending count.

**Implementation status:** Not implemented by the manual author. See canonical ledger for later updates.

---

<a id="task-0014"></a>

### Task 0014 — Distinguish corrupt complete journal records from torn tail

**Project / component:** ContinuityCore  
**Location:** `sources/ContinuityCore/Persistence/Journal.swift:273-289`  
**Severity / tier:** S1 / A  
**Status at manual creation:** START  
**Category:** L3 persistent integrity  
**Primary host:** Apple Silicon Mac A  
**Prerequisites:** 0003, 0010  
**Evidence classification:** static finding; runtime impact untested  

#### 1. Detailed problem description

decodeRecords splits newline-delimited JSON and uses try? to silently drop every undecodable complete line. A torn unterminated last line is legitimately recoverable; an invalid complete middle record, incompatible version, or corrupt checkpoint is different. Returning a successful partial replay can cause later compaction to overwrite recoverable original bytes. Need preserve availability when appropriate, but not treat corruption as empty or successful replay.

**Recorded evidence:** decodeRecords silently drops every complete line whose JSON decoding fails, not just a final unterminated tail; no diagnostic is returned. Corruption or incompatible records may be lost before compaction.

#### 2. Proposed solution / implementation sequence

1. Read JournalRecord schema, FileJournal replay/read, tail recovery and caller compaction behavior. Decide compatibility policy: supported schema migration vs unsupported record vs malformed data vs unterminated tail. Do not silently narrow existing format support.
2. Introduce a replay result or typed error with safe record location/line, failure class, and recovered valid records if necessary. Avoid storing/printing raw memory contents. Preserve last unterminated tail recovery semantics; treat complete invalid records explicitly.
3. Require callers to avoid destructive compaction/checkpoint replacement when replay integrity is uncertain unless there is an explicit operator-approved recovery procedure. Preserve original file or use a safe read-only/degraded path, not live migration.
4. If API shape changes, enumerate FileJournal.read/replay consumers, ContinuityCore, memory CLI and tests; add linked contract tasks for all consumers. Prefer backward compatibility where semantics can remain correct.

#### 3. Verification — regression and integration

1. Add fixtures: valid lines before/after malformed complete middle line, invalid complete final line, unsupported record version, corrupt checkpoint, blank line, valid final newline and torn unterminated tail. Assert exact diagnostic and correct recovery policy; old code silently succeeds on complete corruption.
2. Verify a failed/degraded replay cannot trigger destructive compaction; compare original file bytes and ensure no records silently lost. Test read-only inspection and normal replay ordering.
3. Retain `aTornTailIsDroppedAndTheRestSurvives`, `anUnterminatedFinalLineIsDropped`, `anUnterminatedTailDoesNotCostTheRecordAfterIt`; do not weaken them. Run JournalTests, memory integration/full suite and lint.
4. Test any version compatibility path with historical synthetic fixture and document supported behavior; no production data migration.

#### 4. Self-audit — cold review and closure

1. Cold-read decoder and all consumers; ensure explicit error vs recovery states distinguish absent file, corrupt file and torn tail.
2. Inspect tests for exact preserved bytes and no use of try? that hides corruption. Review handling of untrusted record sizes if touched; record separate task for bounds gaps.
3. Verify no recovery logic leaks data or changes live journals.

**Acceptance:** Test middle-line corruption and incompatible records; report rejected record positions, preserve recoverability and avoid destructive compaction without policy.

**Implementation status:** Not implemented by the manual author. See canonical ledger for later updates.

---

<a id="task-0015"></a>

### Task 0015 — Validate journal file type, ownership and permissions

**Project / component:** ContinuityCore  
**Location:** `sources/ContinuityCore/Persistence/FileJournal+Opening.swift:40-79; sources/ContinuityCore/Persistence/Journal.swift:251-270`  
**Severity / tier:** S1 / A  
**Status at manual creation:** OPEN  
**Category:** L4 filesystem boundary  
**Primary host:** Apple Silicon Mac A  
**Prerequisites:** 0002, 0003  
**Evidence classification:** investigation candidate; not a confirmed defect  

#### 1. Detailed problem description

Investigation. Final-component O_NOFOLLOW and 0600 creation are helpful, but do not establish existing file owner/mode/type and do not prohibit hardlinks. readContents opens O_RDONLY without O_NOFOLLOW and reads until EOF. Directory symlink support is intentional, so a blanket path ban would be wrong. If workspace directories or files can be influenced by an untrusted local process, non-regular files can block or expose data; actual trust/reachability needs proof.

**Recorded evidence:** O_NOFOLLOW and 0600 apply to final write path/creation, but existing file mode/type is not checked in shown openers; readContents follows final symlinks and reads until EOF. Parent-directory symlinks are intentional. Determine whether untrusted workspaces can supply FIFO, device, hardlink or permissive files.

#### 2. Proposed solution / implementation sequence

1. Trace workspace selection and directory creation from server/CLI to MemoryService/ContinuityStore/FileJournal. Define operator-controlled directory symlink policy, trust in ancestor paths, ownership/mode requirements, multi-user access and existing journal compatibility.
2. Review lock, journal, temp and read-only open paths. For paths required to be regular files, use safe open flags and descriptor-based fstat before reading, writing or truncating. Consider nonblocking open for special files then reject by type; validate owner/mode and hardlink count if threat model requires. Avoid precheck-only TOCTOU and do not follow final symlink where disallowed.
3. If safe compatibility requires reading legacy permissions, report and fail safely or provide explicit operator repair instructions; never chmod live paths automatically. Preserve explicitly allowed operator directory symlinks and non-destructive read-only behavior.
4. Bound read size/streaming where necessary; coordinate corruption/compaction changes 0010/0014. If existing trust model rules out a scenario, record actual enforcement and regression tests rather than claiming vulnerability or expanding product scope.

#### 3. Verification — regression and integration

1. In a private temporary directory test valid regular file, final symlink, operator-approved directory symlink, FIFO (test with bounded deadline), directory, hardlink and existing permissive/wrong-owner file where practical without privileges. Assert refusal or documented allowed behavior before side effects.
2. Verify read-only path cannot follow disallowed link or block on FIFO; compare sentinel file bytes and permissions; no actual sensitive file used.
3. Retain journal ownership and lock tests; test race-sensitive operations via descriptor checks, not sleeps. Run JournalTests, memory suite, lint/SAST and TSan as applicable.
4. Document any platform-specific limitations and residual local trust assumptions. If evidence disproves candidate, attach enforced invariant and tests.

#### 4. Self-audit — cold review and closure

1. Cold-read open→fstat→use ordering; check no gap where unchecked descriptor is truncated. Review parent symlink behavior and permissions without broadening access.
2. Ensure failure does not delete, chmod or overwrite user data. Review all lock/append/replay/checkpoint paths, not only first open.
3. Confirm threat model and tests support severity/closure decision.

**Acceptance:** Trace workspace path trust; test non-regular files and existing insecure modes on scratch store; ensure no blocking or unintended read/write; preserve documented directory symlink support.

**Implementation status:** Not implemented by the manual author. See canonical ledger for later updates.

---

<a id="task-0016"></a>

### Task 0016 — Handle short pread during tail repair without truncating unread bytes

**Project / component:** ContinuityCore  
**Location:** `sources/ContinuityCore/Persistence/FileJournal+Opening.swift:105-129`  
**Severity / tier:** S1 / A  
**Status at manual creation:** START  
**Category:** L3 data integrity  
**Primary host:** Apple Silicon Mac A  
**Prerequisites:** 0003  
**Evidence classification:** static finding; runtime impact untested  

#### 1. Detailed problem description

dropDanglingTail scans backward by a requested chunk size. `pread` may return fewer bytes than requested; code searches only buffer[0..<read] then moves end to the chunk offset. The unread suffix is skipped; if it contains the last newline, keep may be too small and ftruncate may remove valid records. Negative EINTR is treated as failure, which is safe but needlessly brittle; zero progress likewise needs explicit handling. Actual normal disk short-read frequency is unknown.

**Recorded evidence:** Tail repair accepts any nonnegative read length, searches only the returned prefix, then sets end to offset, skipping an unread suffix if a short read occurs. Subsequent ftruncate may discard a complete record.

#### 2. Proposed solution / implementation sequence

1. Inspect FileJournal+Opening.swift and journal startup sequence. Preserve exclusive lock and same-descriptor scan/truncate invariant. Add a scoped read seam for fault injection.
2. Implement exact-range pread helper: loop while requested bytes remain, advancing pointer and offset by bytes actually read; retry EINTR, reject other errors, reject unexpected EOF/zero progress while scanning known fstat size. Use checked offsets/counts and maintain correct chunk boundaries.
3. Search a chunk only after all bytes are successfully read. If scan cannot prove the last complete newline, throw and leave original length untouched. Check ftruncate and preserve errno. Decide how to handle concurrent external modification under trust policy 0015.
4. Do not rewrite journal schema, drop complete records, or compensate by disabling tail recovery.

#### 3. Verification — regression and integration

1. Add deterministic tests returning short positive read, EINTR then success, unexpected zero/EOF, and hard error. Place last newline in the previously skipped suffix across chunk boundary. Compare entire file before/after; failure paths preserve original bytes.
2. Test empty file, file ending newline, no newline, exact 64 KiB boundary, multi-chunk valid history and torn suffix. Verify only proven dangling bytes are removed.
3. Run JournalTests and full suite, lint, relevant sanitizer/TSan; ensure no new warning and no sleep synchronization.
4. Capture test failing on old algorithm for the short-read fixture.

#### 4. Self-audit — cold review and closure

1. Cold-read offset math and loop termination; verify no underflow/overflow, no uninitialized bytes searched and no ftruncate before complete scan.
2. Review test seam is production-default real pread and does not hide failures.
3. Check linked journal recovery semantics with 0014.

**Acceptance:** Inject short reads and EINTR; require complete range reads or fail without truncating; verify byte-identical valid tail.

**Implementation status:** Not implemented by the manual author. See canonical ledger for later updates.

---

<a id="task-0017"></a>

### Task 0017 — Bound HTTP header buffering before decoder completes

**Project / component:** TinyTitanServer  
**Location:** `sources/TinyTitanServer/Core/HTTPServer.swift:20-37,102-120`  
**Severity / tier:** S1 / A  
**Status at manual creation:** START  
**Category:** L4 resource exhaustion  
**Primary host:** Apple Silicon Mac A  
**Prerequisites:** 0003, 0009  
**Evidence classification:** static finding; runtime impact untested  

#### 1. Detailed problem description

HTTPServer source explicitly states NIO accumulates an unbounded number of headers before handler sees the complete request head. The 16 KiB/128-field application cap and 64 connection cap therefore do not limit decoder buffering for a single incremental header stream. Idle deadline is activity-based; continuous small reads may postpone it. Loopback limits exposure, not local resource exhaustion. Confirm actual NIO version and pipeline behavior; avoid speculative blanket raw-byte cap that breaks bodies, pipelining or chunked requests.

**Recorded evidence:** Source explicitly acknowledges NIO accumulates unbounded small headers before application aggregate-limit checks. Pipeline config has no earlier aggregate bound. Server binds loopback, limiting exposure but not local denial of service.

#### 2. Proposed solution / implementation sequence

1. Inspect pinned SwiftNIO version and decoder APIs, HTTPServer pipeline and handler idle/connection caps. Trace header bytes before request-head emission and verify whether a supported decoder option solves aggregate limit.
2. Prefer a supported aggregate header byte/count limit at decoder layer. If unavailable, design a bounded HTTP framing guard that understands request boundaries and partial reads, or a verified parser configuration; do not duplicate HTTP incorrectly. Ensure initial line, CRLF boundaries, multiple headers, chunked bodies/trailers and pipelining remain valid. Consider absolute head deadline independent of activity, as linked task if needed.
3. Keep existing application 431 checks, connection cap, body cap and error behavior. If early rejection cannot safely produce 431, close deterministically and document behavior with tests; no unbounded buffering waiting for end-of-headers. Release resources on disconnect.
4. Keep loopback binding and no live endpoint test; measure memory using isolated server with synthetic backend.

#### 3. Verification — regression and integration

1. Extend `tests/TinyTitanServer/HTTPServerTests.swift` (`anOversizedHeaderBlockIsRefused` and stalled body tests) with fragmented over-limit header bytes before terminal CRLF, aggregate field-count boundary, single long header and slow-but-active partial head. Assert deterministic bound/closure and no generation started.
2. Test valid headers at both limits, normal request body > header cap within body cap, SSE, pipelining, HEAD, keep-alive, malformed framing and chunked requests/trailers if supported. Preserve response ordering.
3. Use NIOEmbedded or controlled loopback fixture with synthetic backend; measure buffered bytes or RSS via repeatable instrumentation, not live load. Prove memory bounded as input increases; bounded timeout, no huge stress payload necessary.
4. Run HTTPServerTests, full package suite, TSan and lint; inspect actual decoder config and code scanner.

#### 4. Self-audit — cold review and closure

1. Cold-read framing state machine across fragmented/pipelined requests and error/close transitions; check counters reset only at correct boundary and no integer overflow.
2. Review tests prove pre-decode bound rather than only rejection after complete head. Confirm body/SSE and legitimate clients still work.
3. No dependency upgrade without explicit compatibility task, lock update and full regression verification.

**Acceptance:** Add pre-decode aggregate header byte/count gate or prove equivalent decoder protection; test incrementally supplied oversized headers and bounded RSS.

**Implementation status:** Not implemented by the manual author. See canonical ledger for later updates.

---

<a id="task-0018"></a>

### Task 0018 — Bound stored response memory by bytes and chain size

**Project / component:** TinyTitanServer  
**Location:** `sources/TinyTitanServer/Core/ResponsesAPIStore.swift:17-54; sources/TinyTitanServer/Core/HTTPServerHandler+Responses.swift:43-53,70-87`  
**Severity / tier:** S1 / A  
**Status at manual creation:** START  
**Category:** L5 resource exhaustion  
**Primary host:** Apple Silicon Mac A  
**Prerequisites:** 0003  
**Evidence classification:** static finding; runtime impact untested  

#### 1. Detailed problem description

ResponseStore is globally count-bounded at default 256 entries but retains responseJSON, inputItems and outputItems, and previous_response_id copies flattened prior history into each new entry. Count-only cap permits large total retained memory and repeated history; model context validation may constrain some requests but does not prove a server-wide byte bound. Actual growth depends on storage sharing, request size and generated outputs. Existing item lookup also allocates combined arrays.

**Recorded evidence:** ResponseStore caps number of entries, not bytes; each continuation flattens prior conversation into storedInput, and entries retain input/output plus response JSON. A count cap does not bound accumulated history byte size. Generation context limits may reduce practical impact; measure.

#### 2. Proposed solution / implementation sequence

1. Trace response creation, store=false, streaming/nonstreaming put, references, retrieval/delete and context validation. Measure entry memory representation and realistic worst-case retained bytes; document limits independent of model context. Identify any shared arrays/data and avoid claiming exact RSS from logical size alone.
2. Introduce explicit configurable or justified total byte/entry weight budget and per-entry/chain limit. Use checked arithmetic, account replacement/delete/eviction under existing lock or an appropriate actor, and prevent unbounded accumulation. Consider bounded structural sharing only if lookup/eviction semantics remain correct.
3. Define deterministic policy for oversize entry and prior references: reject or evict with documented API semantics; do not silently report a stored response that cannot be retrieved. Preserve store=false behavior, insertion order, replacement refresh and retrieval/input_items/cancel semantics.
4. Avoid expensive serialization under lock or repeated scans on hot path; consider measured logical byte accounting with conservative overhead. Update response handlers and tests as needed; don't change API without linked consumer verification.

#### 3. Verification — regression and integration

1. Extend `tests/TinyTitanServer/ResponseStoreTests.swift` with mixed-sized entries, replacement larger/smaller, deletion, eviction by bytes and count, zero/invalid capacity, oversize single entry and arithmetic overflow boundaries. Assert total accounting never exceeds budget.
2. HTTP-level tests for repeated previous_response_id chain, long messages, streamed and non-streamed stored outputs, store=false, prior response eviction, GET/input_items and item_reference; verify documented error behavior and no silent truncation.
3. Measure retained logical bytes and RSS trend in isolated synthetic test, comparing baseline and fix. Avoid expecting RSS to drop immediately due to allocator caching.
4. Run ResponseStoreTests, API surface tests, full suite, lint and TSan; compare response protocol and warnings with baseline.

#### 4. Self-audit — cold review and closure

1. Cold-read lock invariants, count/weight updates on every mutation and no multiplication overflow; verify a rejected insertion cannot corrupt ordering/accounting.
2. Review tests prove aggregate limit under repeated chain and high-size entry, not only count cap. Confirm no hidden changes to generated text or model defaults.
3. Check public contract and any docs/clients are updated consistently.

**Acceptance:** Test long continuation chains and many near-limit responses; enforce aggregate byte budget and chain bounds; document eviction semantics.

**Implementation status:** Not implemented by the manual author. See canonical ledger for later updates.

---

<a id="task-0019"></a>

### Task 0019 — Validate loopback trust boundary, Host and browser-origin protections

**Project / component:** TinyTitanServer  
**Location:** `sources/TinyTitanServer/Core/HTTPServerHandler+Routes.swift:22-54,377-395; sources/TinyTitanServer/Core/HTTPServer.swift:125`  
**Severity / tier:** S1 / A  
**Status at manual creation:** OPEN  
**Category:** L4 local service  
**Primary host:** Apple Silicon Mac A  
**Prerequisites:** 0002, 0003, 0020  
**Evidence classification:** investigation candidate; not a confirmed defect  

#### 1. Detailed problem description

Investigation. Server binds 127.0.0.1 and is intentionally local/single-user. Existing route only checks Host header presence for HTTP/1.1 and media-type prefix, plus bodyless unload form guard. Stored response data and unload are privileged local surfaces. A loopback socket is accessible to other processes/users on the same host; browser origin and DNS rebinding risk depends on browser enforcement, proxy behavior and intended security boundary. No remote exploit or broken auth promise is established; do not add mandatory auth blindly and break documented clients.

**Recorded evidence:** Server explicitly binds loopback; route checks Host presence only and JSON content type via prefix. Review DNS rebinding and local multi-user process access, including stored responses and model unload. No remote exploit established.

#### 2. Proposed solution / implementation sequence

1. Read README/server-api/SECURITY and actual routes, listener config, client headers and browser assumptions; identify whether supported deployment includes tunnels/proxies/shared machines. Trace stored response and control endpoints. Document intended trust model and threat assumptions explicitly.
2. Test Host handling (canonical loopback names/IP, port, invalid/foreign hosts, duplicate Host), Origin/Referer when present, CORS/preflight and browser form/simple requests, HTTP/1.0 behavior if supported. Use isolated test clients only.
3. If reachable issue, enforce a strict trusted-host/origin policy appropriate to local server and explicit opt-ins for proxies/remote clients, preserving required CLI headers and JSON media types (0020). For shared-machine threat model, consider explicit configurable authentication and client migration as linked task; never rely on hardcoded dummy bearer as auth.
4. Protect control/memory endpoints and stored response retrieval consistently. Preserve local startup, legitimate programmatic clients and clear error responses; avoid exposing secret values in logs.

#### 3. Verification — regression and integration

1. Use NIOEmbedded/ephemeral loopback integration tests with synthetic backend; assert refused host/origin does not route, mutate state or release model; allowed CLI and documented clients continue to work.
2. Test unload bodyless POST without Content-Type per documented contract, invalid browser form, valid JSON, preflight, foreign Host and response retrieval; check HTTP/1.1 missing Host existing test.
3. If policy is no additional auth due to explicitly accepted local trust, document and test actual bind and constraints; do not mark exploit fixed when it was never established. Escalate confirmed exploitable defect severity based on actual impact.
4. Run HTTPServerTests, API contract/client tests, full suite, lint/SAST; verify no live server, browser exploit tooling or real user data involved.

#### 4. Self-audit — cold review and closure

1. Cold-read entry paths to all privileged routes and listener; validate trust model against actual behavior, including proxy/tunnel assumptions.
2. Check no CORS allow-all or hardcoded credential added and no current client disabled without owner-approved contract task.
3. Record candidate resolution as proof or verified fix, including residual accepted local-process risk.

**Acceptance:** Document single-user trust model; test allowed Host values, browser origin behavior, invalid media types and control endpoints; implement mitigation if threat model requires.

**Implementation status:** Not implemented by the manual author. See canonical ledger for later updates.

---

<a id="task-0020"></a>

### Task 0020 — Parse media type exactly instead of accepting application/json prefix

**Project / component:** TinyTitanServer  
**Location:** `sources/TinyTitanServer/Core/HTTPServerHandler+Routes.swift:42-54`  
**Severity / tier:** S1 / A  
**Status at manual creation:** START  
**Category:** L3 input validation  
**Primary host:** Apple Silicon Mac A  
**Prerequisites:** 0003  
**Evidence classification:** static finding; runtime impact untested  

#### 1. Detailed problem description

`lowercased().hasPrefix("application/json")` matches application/jsonx, application/json-foo and other invalid media types. JSON endpoint admission is broader than its declared contract. Bodyless unload has a separate absent-header allowance for CLI, and that distinction must remain. Actual browser-security implications are in 0019; do not claim them proven by prefix check alone.

**Recorded evidence:** hasPrefix("application/json") accepts unrelated media types such as application/jsonx. Expected exact media type with optional parameters; bodyless route policy remains separate.

#### 2. Proposed solution / implementation sequence

1. Create a single media-type parser/helper shared by `jsonBody` and `notAForm` where appropriate: trim optional whitespace, parse type/subtype case-insensitively, accept exactly application/json with valid optional parameters per documented contract. Decide supported +json variants explicitly, not by prefix.
2. Reject invalid/ambiguous duplicate Content-Type headers and malformed media type consistently with existing error envelope policy if contract requires; keep absent-header allowance only on documented bodyless unload route. Review all routes in switch including Anthropic/Responses.
3. Do not loosen body parsing or browser form protections; preserve JSON validation and error mapping.

#### 3. Verification — regression and integration

1. Extend `everyJsonRouteRefusesANonJsonContentType` and `unloadRefusesABrowserFormPost` with application/jsonx, invalid suffix, valid application/json, casing, whitespace, charset parameter, malformed parameter and absent header cases. Include duplicate-header behavior based on chosen policy.
2. Assert status and surface-specific error envelope, and no backend invocation on rejected media type. Confirm normal JSON, streaming, count_tokens, compact and bodyless unload contracts remain unchanged.
3. Run `swift test --no-parallel --filter HTTPServerTests`, full suite and `tools/lint.sh`. Demonstrate one old prefix-accepted request now fails for intended reason.

#### 4. Self-audit — cold review and closure

1. Cold-read all route admission checks to ensure no old prefix gate remains or divergence between surfaces. Verify absent media type is not newly allowed on JSON body routes.
2. Review parser tests cover boundary rather than just form content; no broad catch or new default success.
3. Coordinate trust-boundary test scope with 0019.

**Acceptance:** Test valid parameters, casing and invalid suffixes across API surfaces.

**Implementation status:** Not implemented by the manual author. See canonical ledger for later updates.

---

<a id="task-0021"></a>

### Task 0021 — Prove allocation products and BLAS narrowing safe at Swift–C boundary

**Project / component:** TinyTitanKernelsC  
**Location:** `sources/TinyTitanKernelsC/gdn_chunk.c:72-126`  
**Severity / tier:** S1 / A  
**Status at manual creation:** OPEN  
**Category:** L4 native memory  
**Primary host:** Apple Silicon Mac A  
**Prerequisites:** 0002, 0003, 0009  
**Evidence classification:** investigation candidate; not a confirmed defect  

#### 1. Detailed problem description

Investigation. `tinytitan_delta_rule_chunk` calculates 2*tokens and multiple size_t allocation products unchecked, then explicitly narrows tokens/dk/dv to int for BLAS. If caller geometry is unbounded, wraparound can underallocate and later memcpy/loop can write out of bounds; narrowing can yield invalid BLAS dimensions. If validated Swift shapes strictly bound all dimensions, runtime path may be safe today but C API contract still needs proof. The function's fallback returns 1 both when it produces sequential output and when dv is too large for fallback; caller interpretation also needs inspection.

**Recorded evidence:** Scratch sizes multiply size_t dimensions without overflow checks; BLAS dimensions explicitly narrow to int. C entry point checks only nonzero dimensions. Need caller reachability and shape bounds before classifying runtime impact.

#### 2. Proposed solution / implementation sequence

1. Locate every `tinytitan_delta_rule_chunk` declaration, caller, Swift wrapper and tests with `git grep -n`; trace geometry to model manifests/config and buffer lengths. Read all C function, header and BLAS wrapper, including allocation failure branch and strides. Record valid dimension/stride bounds and ownership.
2. If reachable, add checked `size_t` multiplication/addition helpers (strict C99 using SIZE_MAX checks), validate every allocation product and BLAS argument fits INT_MAX before cast. Check zero semantics, stride minima, total buffer span assumptions and error code contract. Reject before allocation/use; no compiler-extension or warning suppression.
3. Clarify distinction between success, fallback-produced output and no-output allocation failure; coordinate every caller and ensure no uninitialized readout or partially mutated state on error. If C API intentionally assumes checked Swift wrapper, prove and enforce wrapper preconditions and document header contract; add local checks where public misuse can occur.
4. Keep fast-path arithmetic and numerical ordering unchanged for valid dimensions; use existing performance/golden gates if any hot-path behavior changes. Do not allocate huge memory in tests.

#### 3. Verification — regression and integration

1. Construct isolated C unit/harness inputs with tiny safe buffers and invalid dimensions rejected before dereference: SIZE_MAX product overflow, 2*tokens overflow, INT_MAX+1 BLAS dimension, bad strides, zero and valid boundary. Use fault-injection allocation failure on both fallback-valid and fallback-invalid dv.
2. Trace Swift return handling and regression test invalid manifest-derived geometry with controlled fixture, ensuring clean typed error not trap. Verify valid numerical output matches existing reference/golden tests.
3. Run strict C99 flags from Package.swift, full C-containing suite under ASan+UBSan, relevant Swift tests/full suite, lint and warnings-as-errors. Confirm probes actually instrument C; no giant memory test.
4. If reachability disproven, document complete caller invariant and add boundary tests proving it; adjust severity only with evidence.

#### 4. Self-audit — cold review and closure

1. Cold-read every allocation and pointer offset; verify all arithmetic checked before cast/allocation, all scratch freed exactly once, no fallback writes invalid buffer.
2. Inspect header nullability/ownership and caller shape/stride contracts; verify no silent wrong-result on failure. Check C flags remain exact.
3. Run numerical identity check for touched paths; no speedup at cost of accuracy.

**Acceptance:** Trace all Swift callers and manifest-derived dimensions; test overflow/INT_MAX boundary in isolated sanitizer harness; reject unsafe geometry before allocating or narrowing.

**Implementation status:** Not implemented by the manual author. See canonical ledger for later updates.

---

<a id="task-0022"></a>

### Task 0022 — Prove C reader offset bounds and destruction versus active batch lifecycle

**Project / component:** TinyTitanKernelsC  
**Location:** `sources/TinyTitanKernelsC/expert_io.c:40-46,87-97,220-238,241-284`  
**Severity / tier:** S1 / A  
**Status at manual creation:** OPEN  
**Category:** L2/L4 native interop  
**Primary host:** Apple Silicon Mac A  
**Prerequisites:** 0002, 0003, 0009  
**Evidence classification:** investigation candidate; not a confirmed defect  

#### 1. Detailed problem description

Investigation. expert_io converts uint64 offsets to off_t then adds done, and multiplies expert ID by stride without local overflow guard. Worker pool destroy sets shutting_down, wakes work_ready, joins workers and frees state; active submit waits on work_done/outstanding, and queued submit may wait on batch_idle. If destroy races submit without an enforced ownership contract, possible deadlock/use-after-free. Swift wrapper may guarantee exclusive lifetime and range bounds; trace before changing. Offset safety and lifetime are separate subchecks; split linked tasks if repair requires distinct commits.

**Recorded evidence:** Offset cast to off_t and addition lack local checked arithmetic; destroy signals work_ready, joins workers and frees state while submit waits on work_done and outstanding. Need verify caller ownership prohibits concurrent destroy and validates ranges.

#### 2. Proposed solution / implementation sequence

1. Read full C implementation and tinytitan_expert_io.h; find every create/fetch/fetch_offsets/fetch_ranges/destroy Swift caller and resource ownership. Map lock/condition predicates, outstanding decrement, batch publication pointer lifetime, shutdown admission, fd ownership and cancellation.
2. For offsets, validate multiplication, conversion to representable off_t and end offset (offset+length) before publishing or issuing pread. Ensure `size_t`/ssize_t per-call count is safe, destination sizes enforced by wrapper contract, null pointers handled consistently. Return meaningful errno without partial unsafe writes.
3. For lifecycle, determine if public destroy requires no concurrent users; if enforced by Swift actor/refcount/ownership, test it and document C header precondition. If concurrency allowed, design explicit close admission, cancellation/finish of active batch, wake every waiter and wait for all users before destroy/free. Do not free mutexes/condvars while waiters exist. Ensure caller pointers stay alive until work finishes.
4. Audit error paths during worker init/partial thread creation, EINTR and short file; preserve worker behavior and performance. Add linked tasks if other defects found.

#### 3. Verification — regression and integration

1. Offset harness: zero count, max valid range, >off_t max, multiplication overflow, end overflow, short file, EINTR and valid batch. Reject invalid inputs before accessing small buffers or performing I/O; use sparse scratch file only if needed.
2. Deterministic lifecycle test: pause worker during batch, start second submit, request cancellation/close through supported wrapper, release worker and assert bounded completion and no UAF. Do not deliberately call destroy concurrently if contract forbids it; instead prove wrapper retains reader until all work ends.
3. Run C99 strict build, ASan+UBSan full suite, TSan on concurrency path, relevant Swift integration and full suite. Test all three fetch APIs and worker creation failure if touched.
4. Compare performance/golden numerical behavior on valid expert fetch path and ensure no leaked descriptors or threads.

#### 4. Self-audit — cold review and closure

1. Cold-read condition-variable loops and shutdown signals; prove no active or queued waiter references freed state and every pointer has documented lifetime.
2. Review checked offset math in all fetch modes and failure propagation into Swift. Check no `@unchecked Sendable` or unsafe pointer escape added.
3. Record actual caller reachability and separate conclusions for offset and lifetime; do not claim proven race without evidence.

**Acceptance:** Trace Swift wrapper and header ownership; test boundary offsets, cancellation and destruction during active batch under ASan/UBSan/TSan if contract permits; enforce lifecycle.

**Implementation status:** Not implemented by the manual author. See canonical ledger for later updates.

---

<a id="task-0023"></a>

### Task 0023 — Serialize catalog and configuration fields as safe YAML scalars

**Project / component:** DSH plugin  
**Location:** `plugins/dsh-tinytitan/src/generate.js:62-65,163-170,190-240; plugins/dsh-tinytitan/src/config.js:300-309`  
**Severity / tier:** S1 / A  
**Status at manual creation:** START  
**Category:** L4 configuration injection  
**Primary host:** Linux primary B  
**Prerequisites:** 0003, 0006  
**Evidence classification:** static finding; runtime impact untested  

#### 1. Detailed problem description

DSH generator validates text fields as nonempty/single-line, then interpolates catalog IDs/names, provider keys, reasoning and effort map keys/values unquoted into YAML. YAML syntax is not limited to newline injection; punctuation, comments, aliases/tags, boolean/null-like text or colon-space can change parse type/structure or make invalid settings. Config provider/reasoning only nonempty. Actual privileged action depends on harness settings interpretation, not proven; direct malformed YAML risk is demonstrated. Existing tests enforce byte parity with shell writer, so both writers are part of contract.

**Recorded evidence:** textField rejects newline/tab but permits YAML syntax; builder interpolates ids, names, provider keys, effort values and reasoning as unquoted YAML. Config provider/reasoning only checked for nonempty. Inputs containing YAML special characters can change structure or fail parse.

#### 2. Proposed solution / implementation sequence

1. Read `generate.js`, config.js, `tools/dsh_route.sh`, settings-service and file-based writer paths, catalog sources and `generate.test.js`. Enumerate every interpolated scalar/key and its expected type, allowed enum and trust source. Include port/context/maxTokens numeric fields and catalog effort vocabulary.
2. Use reliable YAML scalar serialization or explicit correctly escaped quoted scalar strategy for all free-form strings and map keys. Validate provider/reasoning/effort values against actual supported grammar/enum when appropriate; do not simply strip punctuation or corrupt display names. Ensure generated YAML parses into expected types and structure.
3. Maintain exact preservation of unrelated user settings and comments. Update shell writer and JS writer together (linked tasks if separate commits); preserve byte parity where it is an intentional contract or justify a separate task changing to semantic parity, never weaken existing tests just to pass.
4. Keep supported provider defaults, model IDs and picker labels unchanged semantically. Reject invalid catalog rows with diagnostics (0025); no raw secret in output. Coordinate atomic writer 0024.

#### 3. Verification — regression and integration

1. Extend `plugins/dsh-tinytitan/test/generate.test.js` with synthetic names/IDs/provider/effort containing colon-space, #, quotes, brackets, YAML null/boolean-like words, leading special characters, and unicode; assert parsed YAML equals intended literal values and no additional keys or altered types. For invalid enums assert clear rejection.
2. Check both JS generator and `tools/dsh_route.sh --print` with the same scratch catalog, preserving existing byte-identical fixtures unless contract change explicitly approved. Verify settings-service and file-based routes yield same intended values.
3. Run `cd plugins/dsh-tinytitan && npm test`, `npm run lint`, `npm run format:check` with pinned Node; run relevant Python/shell parity tests on assigned host and full primary gates. No real DSH profile.
4. Verify existing route replacement, default-model repair and valid thinking-level fixtures all pass. Demonstrate old code fails at least one semantic roundtrip test.

#### 4. Self-audit — cold review and closure

1. Cold-read every interpolated scalar/key in both writers and ensure no raw untrusted field bypasses serializer. Review handling of duplicate keys/rows and error reporting.
2. Inspect tests parse with actual YAML library and assert semantic type/value, not just substrings. Keep existing parity checks strong.
3. Confirm no privilege expansion or arbitrary file modification.

**Acceptance:** Round-trip adversarial scalar cases and malformed config with YAML parser; enforce intended scalar values; update linked shell writer parity tests as necessary.

**Implementation status:** Not implemented by the manual author. See canonical ledger for later updates.

---

<a id="task-0024"></a>

### Task 0024 — Make settings rewrite atomic and detect concurrent updates

**Project / component:** DSH plugin  
**Location:** `plugins/dsh-tinytitan/src/generate.js:346-379`  
**Severity / tier:** S1 / A  
**Status at manual creation:** START  
**Category:** L2 persistent config  
**Primary host:** Linux primary B  
**Prerequisites:** 0003, 0006, 0023  
**Evidence classification:** static finding; runtime impact untested  

#### 1. Detailed problem description

replaceSettingsFile reads existing settings, optionally copies a backup, then writeFileSync truncates and writes the target in place. Crash, disk-full or interrupted write can leave a partial settings file. A separate editor/process may change file after read and before write, and old snapshot can overwrite those edits. Backups are not atomicity or concurrency control. Symlink and file-mode preservation requirements need explicit contract. JS single-process synchronous call does not exclude external writers.

**Recorded evidence:** Read settings then optional copy backup followed by writeFileSync directly to target. Interrupted write can truncate user profile; concurrent editor changes after read can be overwritten. Backup does not make replacement atomic.

#### 2. Proposed solution / implementation sequence

1. Trace writeRouteSettings, default-model repair, settings service, watchers and shell writer. Determine who owns file, whether symlinks intentionally supported, existing permissions, backup behavior and multi-writer expectations.
2. Write validated new YAML to a uniquely created same-directory temp file (exclusive create), preserve intended owner/mode, flush and close, then atomically rename. Use a checked parent-directory durability barrier if promised; report failure accurately before/after rename. Do not truncate original directly or delete an unrelated path. Cleanup only audit-owned temps.
3. Detect concurrent changes against the exact original bytes/inode/version before replacement; define coordination with cooperating writers (lock or settings service). A pre-rename recheck alone cannot guarantee CAS against noncooperating editors; document residual race or use an actual coordinated protocol. On conflict re-read/recompute or fail clearly; never blindly overwrite.
4. Backup the exact original being replaced, not a different concurrent version; preserve no-op behavior and collision-safe backup names. Avoid widening file permissions; respect supported symlink semantics. Coordinate shell/JS behavior and keep settings-service path consistent.
5. Use explicit error propagation and stable logging; no silent success when backup/write/sync/rename failed. Do not modify a real user profile.

#### 3. Verification — regression and integration

1. Extend generate.test.js existing real-file rewrite and missing-file cases with isolated temporary files: failure before rename leaves original byte-identical, success gives complete validated YAML, rename/fsync failure reports correct state, backup matches original, no-op creates no backup.
2. Inject external edit between read and write; assert conflict detected or merge preserving unrelated edits under documented coordination. Test concurrent cooperating writer, backup collision, file permissions and approved symlink behavior. Avoid sleeps; use injected seam or synchronization.
3. Test route writer and default-model repair plus shell parity and settings service. Ensure invalid YAML from 0023 cannot replace settings. Use subprocess crash fixture if feasible, no real profile.
4. Run full plugin npm test/lint/format, relevant shell tests, full primary gates; verify no temp/backup artifacts committed.

#### 4. Self-audit — cold review and closure

1. Cold-read exact commit point, error outcomes, file mode and backup contents. Prove failure before rename cannot damage original; post-rename failure is not described as rollback.
2. Verify concurrency claim matches mechanism; do not label best-effort check a guarantee against arbitrary external writers.
3. Review temp cleanup/symlink paths and preserve unrelated user settings.

**Acceptance:** Fault inject interruption and concurrent edit; use safe replacement preserving file mode and detect conflicts; verify both route/default-model branches.

**Implementation status:** Not implemented by the manual author. See canonical ledger for later updates.

---

<a id="task-0027"></a>

### Task 0027 — Verify bounded GGUF metadata parser resource use and duplicate handling

**Project / component:** TinyTitanFormat  
**Location:** `sources/TinyTitanFormat/GGUFHeaderV3.swift:69-81,187-204`  
**Severity / tier:** S1 / A  
**Status at manual creation:** OPEN  
**Category:** L4 untrusted model parsing  
**Primary host:** Apple Silicon Mac A  
**Prerequisites:** 0002, 0003, 0009  
**Evidence classification:** investigation candidate; not a confirmed defect  

#### 1. Detailed problem description

Investigation. GGUFHeaderV3 caps tensor/key counts under 2^20, but duplicate metadata overwrites earlier values; tensor names may duplicate; long arrays iterate every element even when not retained. Cursor supports needMoreBytes and caller may retry with larger headers. Need prove actual maximum header size, cumulative time/memory, spec rules and range validation across local/remote consumers before declaring exploitable DoS or wrong model. Empty arrays may bypass type validation because element loop does not run.

**Recorded evidence:** Key/tensor counts are capped at <2^20; metadata duplicates overwrite earlier entries; arrays iterate each element even if not retained. Need caller header-size limits and duplicate/tensor contract before asserting vulnerability.

#### 2. Proposed solution / implementation sequence

1. Trace every GGUFHeaderV3 initializer/read caller and local/remote header growth policy, downstream tensor lookup/geometry and metadata interpretation. Inspect format specification/version contract already documented in repo; distinguish permitted duplicates from invalid ones. Map exact max input/header sizes.
2. Create explicit parser resource budgets: header bytes, metadata/tensor counts, aggregate array elements/string bytes and bounded retry/growth. For skipped fixed-width arrays use checked offset advance if compatible, validating element type even when empty; variable-size strings need bounded walk. Preserve valid long tokenizer arrays and semantics.
3. If duplicates invalid, reject with typed parse error before overwrite/ambiguous lookup; otherwise define deterministic policy and ensure downstream consumers agree. Validate dimensions, alignment, relative/absolute offsets and truncation without arithmetic traps. Do not silently truncate model metadata.
4. Use checked arithmetic for cursor bounds/needMoreBytes and size calculations; avoid large allocation solely from declared count. Add linked consumer fixes where validation/format contract changes. If existing caller limits prove safe, document and test those limits rather than speculative rewrite.

#### 3. Verification — regression and integration

1. Synthetic minimal GGUF fixtures: valid v2/v3, duplicate metadata key and tensor name, empty array with invalid element type, nested array, large declared array with short payload, oversized string/count, alignment invalid, offset overflow, rank invalid and truncated header. Assert typed result and bounded work.
2. Test local and remote callers on declared header cap, incremental needMoreBytes growth, malformed partial data and valid long tokenizer metadata. Use tiny inputs and instrument requested bytes/iteration count; no huge allocation or remote model download.
3. Fuzz bounded parser with deterministic seed and capped input on pinned host; no crash, unbounded allocation or hang. Run relevant format/repack tests, full suite, SAST and sanitizers for native consumers.
4. Verify valid known model fixture parses identically; no format compatibility regression. If candidate disproven, attach complete caller-limit proof and boundary tests.

#### 4. Self-audit — cold review and closure

1. Cold-read parser loops, allocation sources, count/offset conversion, duplicate policy and caller retry logic. Check no catch turns corrupt header into valid empty result.
2. Review fuzz/negative tests assert error type and resource bounds rather than mere process survival.
3. Confirm every consumer honors same limits and schema; record residual risk and classification.

**Acceptance:** Trace local and remote callers; test duplicate keys/tensors, large arrays, invalid empty array element types, malformed offsets, memory/time bounds.

**Implementation status:** Not implemented by the manual author. See canonical ledger for later updates.

---

<a id="task-0028"></a>

### Task 0028 — Verify overflow-safe compressed expert block-count arithmetic

**Project / component:** TinyTitanFormat  
**Location:** `sources/TinyTitanFormat/ExpertBlobIndex.swift:62-63`  
**Severity / tier:** S1 / A  
**Status at manual creation:** OPEN  
**Category:** L3 model parser  
**Primary host:** Apple Silicon Mac A  
**Prerequisites:** 0002, 0003  
**Evidence classification:** investigation candidate; not a confirmed defect  

#### 1. Detailed problem description

Investigation. ExpertBlobIndex uses ceiling division `(phaseBytes + blockBytes - 1) / blockBytes` at lines 62–63 after positivity guard. Addition may overflow at integer limits depending on actual type and upstream bounds; Int conversion may trap if result exceeds Int range. A correct validated caller could render it unreachable. Need inspect phase size, block exponent and format constraints before claiming corrupt-model crash.

**Recorded evidence:** Ceiling formula phaseBytes + blockBytes - 1 uses unchecked addition. Need prove bounds/types at all decode and encode call sites.

#### 2. Proposed solution / implementation sequence

1. Read ExpertBlobIndex.swift entire function/type definitions and every caller in writer/reader/planner. Trace phaseBytes and blockBytes from manifest/index geometry and max allowed sizes; confirm types and conversion ranges.
2. If unsafe, compute ceiling using quotient plus nonzero remainder (or checked add), check result fits target Int and format maximum block count. Preserve existing zero/invalid-input contract unless a separate validated contract task changes it; reject malformed data through typed error rather than trap or fake zero.
3. Check adjacent arithmetic for expected frame counts, header/table size and compression layout. Record new independent findings if present. Update all reader/writer consumers if signature becomes throwing; preserve file format and normal results.
4. If caller invariant is sufficient, record actual checked bounds and add regression tests. Do not change codec layout or quantization behavior without linked contract task.

#### 3. Verification — regression and integration

1. Add unit/format tests for phase 0, block 0, one byte, exact multiple, one past multiple, max supported phase, maximum representable integer and invalid huge input. Assert mathematically correct ceiling or typed rejection; no trap or giant allocation.
2. Use synthetic expert-index fixture with malformed phase geometry through reader path and valid writer→reader roundtrip for representative block exponents/frame counts. Verify valid serialized bytes unchanged.
3. Run relevant format/repack tests identified by `git grep`, full suite and `tools/lint.sh`; run sanitizer/format fuzz gates where applicable. Demonstrate pre-fix failure or enforced prior invariant.
4. Confirm every consumer handles failure rather than interpreting it as a valid empty expert.

#### 4. Self-audit — cold review and closure

1. Cold-read all arithmetic and conversions with actual types; verify positivity checks and result bounds, no unchecked forced conversion.
2. Review tests include near-integer-limit values without allocating corresponding buffers.
3. Confirm format compatibility, unchanged valid geometry and complete caller proof.

**Acceptance:** Trace phase size validation; test maximum phase lengths; use checked arithmetic or division/remainder where needed.

**Implementation status:** Not implemented by the manual author. See canonical ledger for later updates.

---

<a id="task-0029"></a>

### Task 0029 — Guard against zero-progress write loops

**Project / component:** ContinuityCore  
**Location:** `sources/ContinuityCore/Persistence/Journal.swift:155-170,410-424`  
**Severity / tier:** S1 / A  
**Status at manual creation:** START  
**Category:** L2 I/O lifecycle  
**Primary host:** Apple Silicon Mac A  
**Prerequisites:** 0003, 0010  
**Evidence classification:** static finding; runtime impact untested  

#### 1. Detailed problem description

Two Journal.swift writeFully loops retry negative EINTR and throw on other negative errors, then advance by written. A zero write while bytes remain leaves offset unchanged indefinitely. Regular-file write rarely returns zero for positive request, but an I/O seam or unusual filesystem may expose it; deterministic test required. Both append and checkpoint helper have same invariant.

**Recorded evidence:** Both writeFully loops advance offset only by write return value; zero return with remaining bytes neither throws nor advances, causing nontermination if such a write occurs.

#### 2. Proposed solution / implementation sequence

1. Inspect both instance/static writeFully and any shared POSIX helper; consider consolidating into a single scoped helper while preserving path-specific typed error and descriptor ownership. Use a deterministic write seam shared with journal durability fault tests.
2. For each positive remaining byte count require positive progress; on zero throw explicit JournalError.writeFailed (appropriate EIO or documented equivalent), capture errno only for negative return. Retry EINTR, continue short positive write, propagate hard error. Guard impossible return >remaining if seam permits; checked pointer offset.
3. Preserve empty Data no-op behavior. Partial append failure may leave torn tail; ensure caller reports failure and next append does not silently combine with partial record—trace and file linked task if needed. Checkpoint failure must not rename incomplete file (0010).
4. Do not add sleep, arbitrary retry counts for normal short writes, or a success fallback that drops bytes.

#### 3. Verification — regression and integration

1. Inject write outcomes: zero first, positive short then zero, EINTR then positive, multiple short writes, hard failure, empty data, full write. Assert deterministic termination, typed error and exact byte prefix/complete content.
2. Test both append and checkpoint paths; failure must leave original checkpoint intact, lock/resource cleanup correct, no acknowledged success after zero progress. Reopen scratch journal and verify torn-tail policy if partial write occurred.
3. Add JournalTests and run targeted suite with bounded deterministic timeout guard, full suite, lint and relevant sanitizer/TSan. Regression must fail before fix without hanging entire test runner (use controlled subprocess or bounded seam).

#### 4. Self-audit — cold review and closure

1. Cold-read loop progress proof for every branch and pointer offset; confirm both helpers covered and no duplicate unpatched loop.
2. Verify error reporting and cleanup propagate to memory callers; no catch suppresses failure. Check 0010 and 0016 related I/O invariants.
3. Confirm tests fail on original path and prove normal short-write semantics unchanged.

**Acceptance:** Inject zero-write result on scratch descriptor seam; fail with explicit I/O error instead of spinning; preserve EINTR retry.

**Implementation status:** Not implemented by the manual author. See canonical ledger for later updates.

---

<a id="task-0025"></a>

### Task 0025 — Report malformed catalog rows rather than silently dropping them

**Project / component:** DSH plugin  
**Location:** `plugins/dsh-tinytitan/src/generate.js:111-149`  
**Severity / tier:** S2 / A  
**Status at manual creation:** START  
**Category:** L2 diagnostics  
**Primary host:** Linux primary B  
**Prerequisites:** 0003, 0006  
**Evidence classification:** static finding; runtime impact untested  

#### 1. Detailed problem description

catalogRows catches every exception and continues; invalid catalog rows vanish without the stderr warning documented for shell parser. Valid rows should remain usable, but operator needs a safe diagnostic to discover broken model install. Broad catch may hide programming bugs, not just expected validation errors.

**Recorded evidence:** catalogRows broad catch continues with no diagnostic; comment says corresponding shell parser prints a skip message. Malformed install disappears silently from picker.

#### 2. Proposed solution / implementation sequence

1. Enumerate expected row-validation failures in generate.js/catalogLevels/textField; replace broad catch with explicit typed validation result/error handling for invalid data. Let unexpected programming or I/O failures surface with context rather than masking them.
2. Report a safe structured reason (row index, validated model ID if safe, invalid field/category) through existing logger/diagnostic channel. Do not include raw untrusted catalog contents or potential credentials. Keep return API compatible where possible, or link all consumers.
3. Preserve order, valid row selection and parity with shell parser; decide how empty all-invalid catalog is communicated. Deduplicate repeated watcher diagnostics if necessary without suppressing first failure. Coordinate 0023 safe scalar validation.

#### 3. Verification — regression and integration

1. Add JS tests for invalid backend, missing ID/name, bad numeric quant, invalid thinking structure and mixed valid/invalid rows; assert correct valid rows and exact safe diagnostic for each rejected row. Test unexpected exception is not swallowed.
2. Check all-invalid and empty catalog cases, repeated refresh and shell parity. Existing byte parity/output tests must remain meaningful; diagnostics should not pollute generated YAML/stdout if machine-consumed.
3. Run plugin npm test/lint/format and relevant shell parity checks; verify full suite due to behavior change.

#### 4. Self-audit — cold review and closure

1. Cold-read catch scope and logging redaction; no raw catalog or credentials logged. Confirm invalid input is not silently converted to fake model.
2. Review tests assert both retained valid rows and diagnostic, not merely absence of crash.
3. Check severity remains medium unless actual production wrong-result/security effect discovered.

**Acceptance:** Retain useful valid rows, report safe row-specific rejection reason; test malformed entries without exposing credentials.

**Implementation status:** Not implemented by the manual author. See canonical ledger for later updates.

---

<a id="task-0026"></a>

### Task 0026 — Require safe integer compaction headroom

**Project / component:** DSH plugin  
**Location:** `plugins/dsh-tinytitan/src/config.js:149-157`  
**Severity / tier:** S2 / A  
**Status at manual creation:** START  
**Category:** L3 numeric validation  
**Primary host:** Linux primary B  
**Prerequisites:** 0003, 0006  
**Evidence classification:** static finding; runtime impact untested  

#### 1. Detailed problem description

resolveCompactionHeadroom converts with Number then checks Number.isInteger and >=0. Unsafe IEEE-754 integer-valued numbers can pass, and coercion of booleans, empty string or whitespace may produce 0/1. Adjacent route token count uses Number.isSafeInteger. Such inputs can distort compaction arithmetic, but intended accepted input types (config vs env) must be checked; don't reject documented numeric strings by accident.

**Recorded evidence:** resolveCompactionHeadroom uses Number.isInteger, accepts integer-valued unsafe IEEE-754 numbers unlike resolveRouteTokenCount at 173–179.

#### 2. Proposed solution / implementation sequence

1. Trace configuration schema, env fallback if any, callers and arithmetic using headroom. Define supported input types, null/unset semantics, zero semantics, max useful range, and relation to context window. Distinguish exact numeric strings from accidental coercion.
2. Use `Number.isSafeInteger` plus nonnegative range and explicit input-type validation. If numeric strings are supported, parse only documented decimal syntax and verify safe value. Reject booleans, arrays, malformed strings, NaN, infinity and unsafe integers; keep valid 0 and null behavior.
3. Review subsequent subtraction/multiplication and context bounds; reject impossible configurations with clear field-specific error, not silently clamp unless existing contract explicitly requires clamp. Keep defaults unchanged.

#### 3. Verification — regression and integration

1. Find existing config tests (enumerate plugin/test; do not assume config.test.js exists). Add cases for undefined/null, 0, 1, MAX_SAFE_INTEGER, MAX_SAFE_INTEGER+1, negative, fraction, NaN/infinity, booleans, empty/whitespace, valid/invalid numeric strings and headroom exceeding context if relevant.
2. Exercise resulting compaction threshold calculations with narrow and wide valid contexts; assert exact intended thresholds and no negative/unbounded budget. Demonstrate old unsafe integer or coercion accepted and new behavior rejected.
3. Run plugin npm test/lint/format and full gates for behavior change; preserve existing valid defaults and documented config examples.

#### 4. Self-audit — cold review and closure

1. Cold-read coercion, bounds and downstream calculations; verify no silent precision loss or undesired type acceptance.
2. Check config error includes field but not secrets. Ensure tests reflect supported numeric strings/null rather than arbitrarily tightening external contract.
3. Compare valid baseline behavior and docs.

**Acceptance:** Boundary tests for MAX_SAFE_INTEGER, unsafe integer, nonnumeric and boolean inputs; document valid range and type.

**Implementation status:** Not implemented by the manual author. See canonical ledger for later updates.

## Final handoff checklist

- All 29 task numbers appear exactly once above; all have the required problem, solution, verification and self-audit sections.
- Record any new findings under new stable IDs. Do not mistake this manual for exhaustive inspection.
- Only accept DONE with actual evidence and focused commit where applicable. Remaining discovery under 0007 and Phase E under 0008 are mandatory.
- All audit files are planning/evidence only until an authorized agent runs the tasks. No production changes were made in creating this manual.
