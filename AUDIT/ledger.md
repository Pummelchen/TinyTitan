# TinyTitan audit ledger — discovery only

**Agent execution instructions:** [TASKS.md](TASKS.md) contains the detailed manual for every task, generated from the canonical `manual` fields in ledger.json. No remediation has been performed.

Pinned source revision: `7b0901d75b3b1e79498874babf93218a617a69ac`. User requested no fixes; this ledger is a task list, not an audit completion certificate. Source of truth: ledger.json. START = statically demonstrated condition (runtime tests not run); OPEN = investigation/gate, not confirmed defect; BLOCKED = existing environment blocker. Severity on candidates is provisional. Existing 0001–0009 retained; 0010 onward newly recorded. **done:0 open:26 blocked:3 new:20**.

| ID | Severity | Status | Project | Title |
|---|---|---|---|---|
| 0001 | S0 | BLOCKED | Root SwiftPM / Apple projects | Provide primary Xcode 27 / Swift 6.4 Apple Silicon execution host |
| 0002 | S0 | OPEN | Repository | Complete Phase A inventory, two-hop contracts, trust boundaries and tier validation |
| 0003 | S0 | OPEN | Repository | Produce current pinned baseline and tool coverage proofs |
| 0006 | S0 | BLOCKED | Repository | Obtain executable clone and pinned non-Apple toolchains |
| 0007 | S0 | OPEN | Repository | Complete L0–L7, facade and unused-code audit, remediation and convergence |
| 0008 | S0 | BLOCKED | Repository | Run Phase E once on an independent fresh-clone host |
| 0010 | S0 | START | ContinuityCore | Checkpoint replacement ignores durability failures |
| 0004 | S1 | START | Embedded and plugin examples | Enforce required Swift settings in standalone example packages |
| 0005 | S1 | START | Library demo | Enforce Swift 6 and warnings-as-errors in direct swiftc demo builds |
| 0009 | S1 | OPEN | Repository | Verify sanitizer and dependency/SAST/secret-scan coverage for every applicable language |
| 0011 | S1 | START | ContinuityCore | Prevent actor reentrancy from clearing unsynced append accounting |
| 0012 | S1 | OPEN | ContinuityCore | Establish serialization of append versus compaction |
| 0013 | S1 | START | ContinuityCore | Propagate asynchronous journal durability failures |
| 0014 | S1 | START | ContinuityCore | Distinguish corrupt complete journal records from torn tail |
| 0015 | S1 | OPEN | ContinuityCore | Validate journal file type, ownership and permissions |
| 0016 | S1 | START | ContinuityCore | Handle short pread during tail repair without truncating unread bytes |
| 0017 | S1 | START | TinyTitanServer | Bound HTTP header buffering before decoder completes |
| 0018 | S1 | START | TinyTitanServer | Bound stored response memory by bytes and chain size |
| 0019 | S1 | OPEN | TinyTitanServer | Validate loopback trust boundary, Host and browser-origin protections |
| 0020 | S1 | START | TinyTitanServer | Parse media type exactly instead of accepting application/json prefix |
| 0021 | S1 | OPEN | TinyTitanKernelsC | Prove allocation products and BLAS narrowing safe at Swift–C boundary |
| 0022 | S1 | OPEN | TinyTitanKernelsC | Prove C reader offset bounds and destruction versus active batch lifecycle |
| 0023 | S1 | START | DSH plugin | Serialize catalog and configuration fields as safe YAML scalars |
| 0024 | S1 | START | DSH plugin | Make settings rewrite atomic and detect concurrent updates |
| 0027 | S1 | OPEN | TinyTitanFormat | Verify bounded GGUF metadata parser resource use and duplicate handling |
| 0028 | S1 | OPEN | TinyTitanFormat | Verify overflow-safe compressed expert block-count arithmetic |
| 0029 | S1 | START | ContinuityCore | Guard against zero-progress write loops |
| 0025 | S2 | START | DSH plugin | Report malformed catalog rows rather than silently dropping them |
| 0026 | S2 | START | DSH plugin | Require safe integer compaction headroom |

## 0001 — Provide primary Xcode 27 / Swift 6.4 Apple Silicon execution host

- S0 / Tier A / BLOCKED / verification prerequisite
- Project: Root SwiftPM / Apple projects; host: Unassigned Apple Silicon Mac
- Location: Package.swift:1
- Evidence: Local host is Linux x86_64; xcodebuild absent; direct GitHub clone fails DNS. Existing CI specifies xcode-27 but no shell/runner access was established.
- Blocked: {"owner":"Pummelchen (repository owner; assignment proposed)","reason":"Required primary Mac unavailable to this session.","attempt":"Checked uname, xcodebuild and git ls-remote; read CI configuration.","options":["Provide access to an isolated Apple Silicon development Mac with verified Xcode 27 / Swift 6.4.","Run the audit in a local coding agent on such a Mac; retain logs in AUDIT."]}
- Fix, after evidence and fix commit: none.

## 0002 — Complete Phase A inventory, two-hop contracts, trust boundaries and tier validation

- S0 / Tier A / OPEN / audit gate
- Project: Repository; host: Unassigned Apple Silicon Mac
- Location: AUDIT/inventory.md:1
- Evidence: Tree and manifests enumerated; source-level contract mapping and individual classification not complete.
- Fix, after evidence and fix commit: none.

## 0003 — Produce current pinned baseline and tool coverage proofs

- S0 / Tier A / OPEN / audit gate
- Project: Repository; host: Unassigned Apple Silicon Mac
- Location: AUDIT/baseline.md:1
- Evidence: No current build, test, coverage, dependency scan, full-history scan or negative control executed.
- Fix, after evidence and fix commit: none.

## 0004 — Enforce required Swift settings in standalone example packages

- S1 / Tier C / START / language-standard configuration
- Project: Embedded and plugin examples; host: Unassigned Apple Silicon Mac
- Location: examples/embedded/Package.swift:15; examples/plugin/Package.swift:11
- Evidence: Both example manifests require tools version 6.4 but do not declare explicit Swift 6 language mode or warnings-as-errors; tools version alone does not establish warning enforcement. Expected: all example targets enforce the same mode and warnings-as-errors, proven with negative controls.
- Fix, after evidence and fix commit: none.

## 0005 — Enforce Swift 6 and warnings-as-errors in direct swiftc demo builds

- S1 / Tier C / START / language-standard configuration
- Project: Library demo; host: Unassigned Apple Silicon Mac
- Location: examples/library-demo/build-static.sh:46; examples/library-demo/build-dynamic.sh:52
- Evidence: Both scripts invoke swiftc -O with no explicit -swift-version 6, strict concurrency or -warnings-as-errors. Expected: both demo build paths carry the strict standard; prove with negative controls and test both linkage forms.
- Fix, after evidence and fix commit: none.

## 0006 — Obtain executable clone and pinned non-Apple toolchains

- S0 / Tier A / BLOCKED / verification prerequisite
- Project: Repository; host: Linux x86_64 primary for standalone Node and portable tooling
- Location: AUDIT/environment.md:1
- Evidence: Linux local Git clone cannot resolve github.com; Node v22.16.0 does not meet plugin >=24.9; Ruff, gitleaks, shellcheck, pip-audit and CodeQL unavailable.
- Blocked: {"owner":"Pummelchen (repository owner; assignment proposed)","reason":"No networked execution environment or installed pinned scanners available.","attempt":"Checked tool availability and public git ls-remote; connector supports source/API access, not execution.","options":["Provide a networked isolated Linux host with repository checkout and pinned versions.","Use a preconfigured runner or local coding agent with the pinned dependencies; provide full results."]}
- Fix, after evidence and fix commit: none.

## 0007 — Complete L0–L7, facade and unused-code audit, remediation and convergence

- S0 / Tier A / OPEN / audit gate
- Project: Repository; host: Unassigned Apple Silicon Mac
- Location: AUDIT/ledger.md:1
- Evidence: Discovery only. Tier A source lines and scanners have not been exhaustively evaluated; no substantive remediation performed.
- Scope: User requested discovery only; remediation and convergence are retained as later work, not performed in this session.
- Fix, after evidence and fix commit: none.

## 0008 — Run Phase E once on an independent fresh-clone host

- S0 / Tier A / BLOCKED / verification prerequisite
- Project: Repository; host: Unassigned Apple Silicon Mac
- Location: AUDIT/ledger.md:1
- Evidence: No independent execution host available; earlier phases not complete.
- Blocked: {"owner":"Pummelchen (repository owner; assignment proposed)","reason":"Independent host access and completed primary audit unavailable.","attempt":"Read existing CI workflow; no independent execution performed.","options":["Use a second isolated Apple Silicon Mac with Xcode 27/Swift 6.4 and pinned tools after Phase D.","Use an independently provisioned macOS CI runner after Phase D, with complete fresh-clone logs and all required gates."]}
- Fix, after evidence and fix commit: none.

## 0009 — Verify sanitizer and dependency/SAST/secret-scan coverage for every applicable language

- S1 / Tier A / OPEN / audit gate
- Project: Repository; host: Unassigned Apple Silicon Mac
- Location: AUDIT/environment.md:1
- Evidence: CI declares Swift TSan, pip-audit, npm audit, gitleaks and CodeQL. Full C ASan+UBSan suite and applicable Swift/C/JS/Metal/Shell tool coverage not established; no scan output from current revision.
- Fix, after evidence and fix commit: none.

## 0010 — Checkpoint replacement ignores durability failures

- S0 / Tier A / START / L2/L4 persistent data
- Project: ContinuityCore; host: Apple Silicon Mac A
- Location: sources/ContinuityCore/Persistence/Journal.swift:336-372
- Evidence: writeCheckpoint ignores fallback fsync result at 359; ignores directory open failure and directory fsync result at 367–371, then returns success. A failed data barrier does not prevent rename of the replacement.
- Acceptance: Inject file and directory sync errors; preserve old journal before successful data sync; report uncertain durability after rename. Verify replay and crash recovery.
- Fix, after evidence and fix commit: none.

## 0011 — Prevent actor reentrancy from clearing unsynced append accounting

- S1 / Tier A / START / L2 concurrency
- Project: ContinuityCore; host: Apple Silicon Mac A
- Location: sources/ContinuityCore/Persistence/Journal.swift:132-150,188-207
- Evidence: performBarrier awaits a blocking queue then sets pendingSinceSync to zero if descriptor unchanged. Actor can accept append while suspended; those records may be written after barrier syscall but before completion. Comment claiming appends queue behind the await is not a Swift actor guarantee.
- Acceptance: Deterministic barrier/append interleaving test; count only records captured by completed durability barrier, preserve newer pending writes.
- Fix, after evidence and fix commit: none.

## 0012 — Establish serialization of append versus compaction

- S1 / Tier A / OPEN / L2 concurrency / data loss risk
- Project: ContinuityCore; host: Apple Silicon Mac A
- Location: sources/ContinuityCore/Persistence/Journal.swift:294-329,105-124
- Evidence: compact awaits a checkpoint replacement on a blocking queue; append writes to existing descriptor. Unless every caller provides external serialization, an append during suspension may land on the old inode and disappear after reopen. Caller contract not yet established.
- Acceptance: Trace all FileJournal callers; reproduce with controlled suspension; guarantee append preservation or explicitly enforce isolation.
- Fix, after evidence and fix commit: none.

## 0013 — Propagate asynchronous journal durability failures

- S1 / Tier A / START / L2 error propagation
- Project: ContinuityCore; host: Apple Silicon Mac A
- Location: sources/ContinuityCore/Persistence/Journal.swift:140-146,393-400
- Evidence: Scheduled performBarrier uses try?; shutDown also discards errors from settleBarrier and performBarrier before closing and releasing lock, despite async throws signature.
- Acceptance: Fault inject fsync errors; expose degraded status and shutdown error while guaranteeing resource cleanup; verify retry and pending count.
- Fix, after evidence and fix commit: none.

## 0014 — Distinguish corrupt complete journal records from torn tail

- S1 / Tier A / START / L3 persistent integrity
- Project: ContinuityCore; host: Apple Silicon Mac A
- Location: sources/ContinuityCore/Persistence/Journal.swift:273-289
- Evidence: decodeRecords silently drops every complete line whose JSON decoding fails, not just a final unterminated tail; no diagnostic is returned. Corruption or incompatible records may be lost before compaction.
- Acceptance: Test middle-line corruption and incompatible records; report rejected record positions, preserve recoverability and avoid destructive compaction without policy.
- Fix, after evidence and fix commit: none.

## 0015 — Validate journal file type, ownership and permissions

- S1 / Tier A / OPEN / L4 filesystem boundary
- Project: ContinuityCore; host: Apple Silicon Mac A
- Location: sources/ContinuityCore/Persistence/FileJournal+Opening.swift:40-79; sources/ContinuityCore/Persistence/Journal.swift:251-270
- Evidence: O_NOFOLLOW and 0600 apply to final write path/creation, but existing file mode/type is not checked in shown openers; readContents follows final symlinks and reads until EOF. Parent-directory symlinks are intentional. Determine whether untrusted workspaces can supply FIFO, device, hardlink or permissive files.
- Acceptance: Trace workspace path trust; test non-regular files and existing insecure modes on scratch store; ensure no blocking or unintended read/write; preserve documented directory symlink support.
- Fix, after evidence and fix commit: none.

## 0016 — Handle short pread during tail repair without truncating unread bytes

- S1 / Tier A / START / L3 data integrity
- Project: ContinuityCore; host: Apple Silicon Mac A
- Location: sources/ContinuityCore/Persistence/FileJournal+Opening.swift:105-129
- Evidence: Tail repair accepts any nonnegative read length, searches only the returned prefix, then sets end to offset, skipping an unread suffix if a short read occurs. Subsequent ftruncate may discard a complete record.
- Acceptance: Inject short reads and EINTR; require complete range reads or fail without truncating; verify byte-identical valid tail.
- Fix, after evidence and fix commit: none.

## 0017 — Bound HTTP header buffering before decoder completes

- S1 / Tier A / START / L4 resource exhaustion
- Project: TinyTitanServer; host: Apple Silicon Mac A
- Location: sources/TinyTitanServer/Core/HTTPServer.swift:20-37,102-120
- Evidence: Source explicitly acknowledges NIO accumulates unbounded small headers before application aggregate-limit checks. Pipeline config has no earlier aggregate bound. Server binds loopback, limiting exposure but not local denial of service.
- Acceptance: Add pre-decode aggregate header byte/count gate or prove equivalent decoder protection; test incrementally supplied oversized headers and bounded RSS.
- Fix, after evidence and fix commit: none.

## 0018 — Bound stored response memory by bytes and chain size

- S1 / Tier A / START / L5 resource exhaustion
- Project: TinyTitanServer; host: Apple Silicon Mac A
- Location: sources/TinyTitanServer/Core/ResponsesAPIStore.swift:17-54; sources/TinyTitanServer/Core/HTTPServerHandler+Responses.swift:43-53,70-87
- Evidence: ResponseStore caps number of entries, not bytes; each continuation flattens prior conversation into storedInput, and entries retain input/output plus response JSON. A count cap does not bound accumulated history byte size. Generation context limits may reduce practical impact; measure.
- Acceptance: Test long continuation chains and many near-limit responses; enforce aggregate byte budget and chain bounds; document eviction semantics.
- Fix, after evidence and fix commit: none.

## 0019 — Validate loopback trust boundary, Host and browser-origin protections

- S1 / Tier A / OPEN / L4 local service
- Project: TinyTitanServer; host: Apple Silicon Mac A
- Location: sources/TinyTitanServer/Core/HTTPServerHandler+Routes.swift:22-54,377-395; sources/TinyTitanServer/Core/HTTPServer.swift:125
- Evidence: Server explicitly binds loopback; route checks Host presence only and JSON content type via prefix. Review DNS rebinding and local multi-user process access, including stored responses and model unload. No remote exploit established.
- Acceptance: Document single-user trust model; test allowed Host values, browser origin behavior, invalid media types and control endpoints; implement mitigation if threat model requires.
- Fix, after evidence and fix commit: none.

## 0020 — Parse media type exactly instead of accepting application/json prefix

- S1 / Tier A / START / L3 input validation
- Project: TinyTitanServer; host: Apple Silicon Mac A
- Location: sources/TinyTitanServer/Core/HTTPServerHandler+Routes.swift:42-54
- Evidence: hasPrefix("application/json") accepts unrelated media types such as application/jsonx. Expected exact media type with optional parameters; bodyless route policy remains separate.
- Acceptance: Test valid parameters, casing and invalid suffixes across API surfaces.
- Fix, after evidence and fix commit: none.

## 0021 — Prove allocation products and BLAS narrowing safe at Swift–C boundary

- S1 / Tier A / OPEN / L4 native memory
- Project: TinyTitanKernelsC; host: Apple Silicon Mac A
- Location: sources/TinyTitanKernelsC/gdn_chunk.c:72-126
- Evidence: Scratch sizes multiply size_t dimensions without overflow checks; BLAS dimensions explicitly narrow to int. C entry point checks only nonzero dimensions. Need caller reachability and shape bounds before classifying runtime impact.
- Acceptance: Trace all Swift callers and manifest-derived dimensions; test overflow/INT_MAX boundary in isolated sanitizer harness; reject unsafe geometry before allocating or narrowing.
- Fix, after evidence and fix commit: none.

## 0022 — Prove C reader offset bounds and destruction versus active batch lifecycle

- S1 / Tier A / OPEN / L2/L4 native interop
- Project: TinyTitanKernelsC; host: Apple Silicon Mac A
- Location: sources/TinyTitanKernelsC/expert_io.c:40-46,87-97,220-238,241-284
- Evidence: Offset cast to off_t and addition lack local checked arithmetic; destroy signals work_ready, joins workers and frees state while submit waits on work_done and outstanding. Need verify caller ownership prohibits concurrent destroy and validates ranges.
- Acceptance: Trace Swift wrapper and header ownership; test boundary offsets, cancellation and destruction during active batch under ASan/UBSan/TSan if contract permits; enforce lifecycle.
- Fix, after evidence and fix commit: none.

## 0023 — Serialize catalog and configuration fields as safe YAML scalars

- S1 / Tier A / START / L4 configuration injection
- Project: DSH plugin; host: Linux primary B
- Location: plugins/dsh-tinytitan/src/generate.js:62-65,163-170,190-240; plugins/dsh-tinytitan/src/config.js:300-309
- Evidence: textField rejects newline/tab but permits YAML syntax; builder interpolates ids, names, provider keys, effort values and reasoning as unquoted YAML. Config provider/reasoning only checked for nonempty. Inputs containing YAML special characters can change structure or fail parse.
- Acceptance: Round-trip adversarial scalar cases and malformed config with YAML parser; enforce intended scalar values; update linked shell writer parity tests as necessary.
- Fix, after evidence and fix commit: none.

## 0024 — Make settings rewrite atomic and detect concurrent updates

- S1 / Tier A / START / L2 persistent config
- Project: DSH plugin; host: Linux primary B
- Location: plugins/dsh-tinytitan/src/generate.js:346-379
- Evidence: Read settings then optional copy backup followed by writeFileSync directly to target. Interrupted write can truncate user profile; concurrent editor changes after read can be overwritten. Backup does not make replacement atomic.
- Acceptance: Fault inject interruption and concurrent edit; use safe replacement preserving file mode and detect conflicts; verify both route/default-model branches.
- Fix, after evidence and fix commit: none.

## 0025 — Report malformed catalog rows rather than silently dropping them

- S2 / Tier A / START / L2 diagnostics
- Project: DSH plugin; host: Linux primary B
- Location: plugins/dsh-tinytitan/src/generate.js:111-149
- Evidence: catalogRows broad catch continues with no diagnostic; comment says corresponding shell parser prints a skip message. Malformed install disappears silently from picker.
- Acceptance: Retain useful valid rows, report safe row-specific rejection reason; test malformed entries without exposing credentials.
- Fix, after evidence and fix commit: none.

## 0026 — Require safe integer compaction headroom

- S2 / Tier A / START / L3 numeric validation
- Project: DSH plugin; host: Linux primary B
- Location: plugins/dsh-tinytitan/src/config.js:149-157
- Evidence: resolveCompactionHeadroom uses Number.isInteger, accepts integer-valued unsafe IEEE-754 numbers unlike resolveRouteTokenCount at 173–179.
- Acceptance: Boundary tests for MAX_SAFE_INTEGER, unsafe integer, nonnumeric and boolean inputs; document valid range and type.
- Fix, after evidence and fix commit: none.

## 0027 — Verify bounded GGUF metadata parser resource use and duplicate handling

- S1 / Tier A / OPEN / L4 untrusted model parsing
- Project: TinyTitanFormat; host: Apple Silicon Mac A
- Location: sources/TinyTitanFormat/GGUFHeaderV3.swift:69-81,187-204
- Evidence: Key/tensor counts are capped at <2^20; metadata duplicates overwrite earlier entries; arrays iterate each element even if not retained. Need caller header-size limits and duplicate/tensor contract before asserting vulnerability.
- Acceptance: Trace local and remote callers; test duplicate keys/tensors, large arrays, invalid empty array element types, malformed offsets, memory/time bounds.
- Fix, after evidence and fix commit: none.

## 0028 — Verify overflow-safe compressed expert block-count arithmetic

- S1 / Tier A / OPEN / L3 model parser
- Project: TinyTitanFormat; host: Apple Silicon Mac A
- Location: sources/TinyTitanFormat/ExpertBlobIndex.swift:62-63
- Evidence: Ceiling formula phaseBytes + blockBytes - 1 uses unchecked addition. Need prove bounds/types at all decode and encode call sites.
- Acceptance: Trace phase size validation; test maximum phase lengths; use checked arithmetic or division/remainder where needed.
- Fix, after evidence and fix commit: none.

## 0029 — Guard against zero-progress write loops

- S1 / Tier A / START / L2 I/O lifecycle
- Project: ContinuityCore; host: Apple Silicon Mac A
- Location: sources/ContinuityCore/Persistence/Journal.swift:155-170,410-424
- Evidence: Both writeFully loops advance offset only by write return value; zero return with remaining bytes neither throws nor advances, causing nontermination if such a write occurs.
- Acceptance: Inject zero-write result on scratch descriptor seam; fail with explicit I/O error instead of spinning; preserve EINTR retry.
- Fix, after evidence and fix commit: none.
