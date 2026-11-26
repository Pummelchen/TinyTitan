# Audit ledger — initial discovery

Source: `AUDIT/ledger.json`. Generated from the same in-memory task objects for this commit. Base: `7b0901d75b3b1e79498874babf93218a617a69ac`. Status: INCOMPLETE. Gate records do not represent an exhaustive defect count.

**done:0 open:6 blocked:3 new:9**

| ID | Severity | Tier | Project | Status | Title |
|---|---|---|---|---|---|
| 0001 | S0 | A | Root SwiftPM / Apple projects | BLOCKED | Provide primary Xcode 27 / Swift 6.4 Apple Silicon execution host |
| 0002 | S0 | A | Repository | OPEN | Complete Phase A inventory, two-hop contracts, trust boundaries and tier validation |
| 0003 | S0 | A | Repository | OPEN | Produce current pinned baseline and tool coverage proofs |
| 0004 | S1 | C | Embedded and plugin examples | START | Enforce required Swift settings in standalone example packages |
| 0005 | S1 | C | Library demo | START | Enforce Swift 6 and warnings-as-errors in direct swiftc demo builds |
| 0006 | S0 | A | Repository | BLOCKED | Obtain executable clone and pinned non-Apple toolchains |
| 0007 | S0 | A | Repository | OPEN | Complete L0–L7, facade and unused-code audit, remediation and convergence |
| 0008 | S0 | A | Repository | BLOCKED | Run Phase E once on an independent fresh-clone host |
| 0009 | S1 | A | Repository | OPEN | Verify sanitizer and dependency/SAST/secret-scan coverage for every applicable language |

## 0001 — Provide primary Xcode 27 / Swift 6.4 Apple Silicon execution host

- Location: Package.swift:1
- Category: verification prerequisite
- Host: Unassigned Apple Silicon Mac
- Discovered by: Phase A static discovery
- Evidence before: Local host is Linux x86_64; xcodebuild absent; direct GitHub clone fails DNS. Existing CI specifies xcode-27 but no shell/runner access was established.
- Fix / after evidence / fix commit: not available.
- Owner: Pummelchen (repository owner; assignment proposed)
- Blocked: Required primary Mac unavailable to this session.
- Tried: Checked uname, xcodebuild and git ls-remote; read CI configuration.
- Human options: Provide access to an isolated Apple Silicon development Mac with verified Xcode 27 / Swift 6.4. / Run the audit in a local coding agent on such a Mac; retain logs in AUDIT.

## 0002 — Complete Phase A inventory, two-hop contracts, trust boundaries and tier validation

- Location: AUDIT/inventory.md:1
- Category: audit gate
- Host: Unassigned Apple Silicon Mac
- Discovered by: Phase A static discovery
- Evidence before: Tree and manifests enumerated; source-level contract mapping and individual classification not complete.
- Fix / after evidence / fix commit: not available.

## 0003 — Produce current pinned baseline and tool coverage proofs

- Location: AUDIT/baseline.md:1
- Category: audit gate
- Host: Unassigned Apple Silicon Mac
- Discovered by: Phase A static discovery
- Evidence before: No current build, test, coverage, dependency scan, full-history scan or negative control executed.
- Fix / after evidence / fix commit: not available.

## 0004 — Enforce required Swift settings in standalone example packages

- Location: examples/embedded/Package.swift:15; examples/plugin/Package.swift:11
- Category: language-standard configuration
- Host: Unassigned Apple Silicon Mac
- Discovered by: Phase A static discovery
- Evidence before: Both example manifests require tools version 6.4 but do not declare explicit Swift 6 language mode or warnings-as-errors; tools version alone does not establish warning enforcement. Expected: all example targets enforce the same mode and warnings-as-errors, proven with negative controls.
- Fix / after evidence / fix commit: not available.

## 0005 — Enforce Swift 6 and warnings-as-errors in direct swiftc demo builds

- Location: examples/library-demo/build-static.sh:46; examples/library-demo/build-dynamic.sh:52
- Category: language-standard configuration
- Host: Unassigned Apple Silicon Mac
- Discovered by: Phase A static discovery
- Evidence before: Both scripts invoke swiftc -O with no explicit -swift-version 6, strict concurrency or -warnings-as-errors. Expected: both demo build paths carry the strict standard; prove with negative controls and test both linkage forms.
- Fix / after evidence / fix commit: not available.

## 0006 — Obtain executable clone and pinned non-Apple toolchains

- Location: AUDIT/environment.md:1
- Category: verification prerequisite
- Host: Linux x86_64 primary for standalone Node and portable tooling
- Discovered by: Phase A static discovery
- Evidence before: Linux local Git clone cannot resolve github.com; Node v22.16.0 does not meet plugin >=24.9; Ruff, gitleaks, shellcheck, pip-audit and CodeQL unavailable.
- Fix / after evidence / fix commit: not available.
- Owner: Pummelchen (repository owner; assignment proposed)
- Blocked: No networked execution environment or installed pinned scanners available.
- Tried: Checked tool availability and public git ls-remote; connector supports source/API access, not execution.
- Human options: Provide a networked isolated Linux host with repository checkout and pinned versions. / Use a preconfigured runner or local coding agent with the pinned dependencies; provide full results.

## 0007 — Complete L0–L7, facade and unused-code audit, remediation and convergence

- Location: AUDIT/ledger.md:1
- Category: audit gate
- Host: Unassigned Apple Silicon Mac
- Discovered by: Phase A static discovery
- Evidence before: Discovery only. Tier A source lines and scanners have not been exhaustively evaluated; no substantive remediation performed.
- Fix / after evidence / fix commit: not available.

## 0008 — Run Phase E once on an independent fresh-clone host

- Location: AUDIT/ledger.md:1
- Category: verification prerequisite
- Host: Unassigned Apple Silicon Mac
- Discovered by: Phase A static discovery
- Evidence before: No independent execution host available; earlier phases not complete.
- Fix / after evidence / fix commit: not available.
- Owner: Pummelchen (repository owner; assignment proposed)
- Blocked: Independent host access and completed primary audit unavailable.
- Tried: Read existing CI workflow; no independent execution performed.
- Human options: Use a second isolated Apple Silicon Mac with Xcode 27/Swift 6.4 and pinned tools after Phase D. / Use an independently provisioned macOS CI runner after Phase D, with complete fresh-clone logs and all required gates.

## 0009 — Verify sanitizer and dependency/SAST/secret-scan coverage for every applicable language

- Location: AUDIT/environment.md:1
- Category: audit gate
- Host: Unassigned Apple Silicon Mac
- Discovered by: Phase A static discovery
- Evidence before: CI declares Swift TSan, pip-audit, npm audit, gitleaks and CodeQL. Full C ASan+UBSan suite and applicable Swift/C/JS/Metal/Shell tool coverage not established; no scan output from current revision.
- Fix / after evidence / fix commit: not available.

No task is marked DONE. Resume from these records, reopen blockers when the missing capability is supplied, and do not restart discovery. Phase A is not complete. No PR or deployment authorized by this snapshot.
