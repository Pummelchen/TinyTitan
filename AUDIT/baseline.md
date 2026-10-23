# Baseline — 2026-10-23, branch `audit/2026-10-23`, head `4816a84`

Regression yardstick per §3. Primary host: the development Mac (see
`environment.md`). No later state may be worse on any metric without a
justified numbered task.

| Metric | Result |
| --- | --- |
| `swift build -c release` | **complete, 589/589 modules, zero warnings** (137.8 s) — `-warnings-as-errors` on every target, so a warning is a build failure |
| `swift test --no-parallel` | **UNVERIFIED — earlier pass/zero-skip claim withdrawn.** Output was piped through `tail` without evidence that the Swift exit status was preserved; the last bundle's 98 tests / 10 suites is not the full package total. A new complete serial run with coverage and an unmasked exit code is required (AUD2-019). |
| `tools/lint.sh` (all 18 gates) | **all ok** — force-cast, unbounded-read, func-length, file-length, unchecked-sendable, converter, arch-path, stdout-clean, silent-test-skip, test-hollow, library-facade, docs, shell-portability (28 scripts under system bash 3.2.57), shellcheck 0.11.0, swiftlint 0.65.1 --strict, swift-format, javascript (eslint 10.11.0 / prettier 3.9.9 both plugin packages), python (ruff 0.16.7, 227 scripts) |
| Converter suites (`test_prepare_qwen38` + `test_qwen38_resume_e2e` + `test_prepare_agentworld`) | **87 tests, 0 failures** (42.3 s) |
| Secret scan (gitleaks 8.30.1, full history) | **1,981 commits scanned, no leaks found** (33.2 MB, 12.4 s) |
| Dependency CVEs | pip-audit runs in CI over `benchmark/requirements.txt` (numpy 2.5.3, safetensors 0.8.0, ml_dtypes 0.6.0 — "no known vulnerabilities" per CI record); Swift deps pinned by `Package.resolved` (swift-transformers, swift-nio 2.100.0 exact); npm audit in CI over both plugin lockfiles |
| Coverage | **SwiftPM-only export is partial.** It contains only ContinuityCore, not all seven bundles. The post-format multi-object measurement below covers 395 source files with explicit exclusions; it is not whole-repository coverage. AUD2-031 remains open for durable export/scope checks. A downloaded CI artifact (run `39974492789`) holds one bundle and 27 files — the same scope — confirming the defect remotely. No threshold gate. |

Language standard in force (recorded, not intended — proofs in
`tool-coverage.md`): Swift 6.4 / Swift 6 language mode (`swiftLanguageModes:
[.v6]`), `-warnings-as-errors` on 25/25 targets, SwiftLint 0.65.1 `--strict`;
C99 `-pedantic-errors` + full hardening set + `-Werror` on `TinyTitanKernelsC`;
Ruff 0.16.7 at the 3.13 floor; shellcheck 0.11.0 + bash-3.2 portability gate.

Model-gated checks NOT run at baseline (no gate may fetch a model): golden
baseline, `TINYTITAN_LIBRARY_CONTRACT_MODEL` suite. Golden baselines are
checked at release time against installs present under `models/` (ornith-4,
ornith-8, qwen38-4 on this host).

Sanitizer runs for this audit (§1: suite, not just build), all on the primary
host, branch `audit/2026-10-23`:

- **TSan** (`swift test --no-parallel --sanitize=thread --filter 'PromptCheckpoint|ServerPromptState|Calibration|ManagedModelBackend|ServerTermination|ContinuityCore|Streaming'`):
  224 tests / 29 suites, exit 0, **zero ThreadSanitizer diagnostics**.
- **ASan/UBSan** on the C-kernel suites (Int8AffineGEMV, DequantInt4/Int8 GEMV,
  CPUExpertFFN, FusedQKV, LMHeadChainInt4, EmbedLookup): 29 tests / 7 suites,
  exit 0, **zero AddressSanitizer/UBSan findings**.

## Prior-audit deltas (scope note)

The 2026-10-06 audit (210/210 closed) covers the tree up to `docs/audit-ledger.json`.
Since then: 163 commits touching `sources/` (+13,330 / −1,547 lines over 132
files) — releases v5.19–v6.0, the Project Maya work. The manual passes recorded
so far concentrate on those deltas. **That is not the full Tier-A L2/L3/L4
coverage required by `scope-tiers.md`.** The prior audit and whole-tree tools
cannot stand in for those current manual passes; their remaining coverage is
part of AUD2-019 and must stay open until evidenced.

## Continuation verification — 2026-10-23

These are working-tree measurements at `164e84a`, not clean-commit certification.
The original baseline above is historical, with its withdrawn test claim kept
visible.

- `/bin/bash tools/lint.sh > AUDIT/continuation-lint.log 2>&1` exited **1**:
  16 of 18 gates passed. `file-length` found `PromptCheckpointStore.swift` at
  556 lines; the printed `swift-format` diagnostics named four files from the
  pending audit fixes. No gate was disabled. This run predates the M5 cleanup.
  **Count correction:** `tools/lint.sh` prints only the first 30 formatter
  diagnostics; the visible four files were not an exhaustive count. A later
  pass exposed a fifth, `TwoBitExpertReadTests.swift`. Neither the original
  count nor the intermediate correction to four was a complete inventory.
- `env -u TINYTITAN_LIBRARY_CONTRACT_MODEL xcrun swift test --no-parallel
  --enable-code-coverage > AUDIT/continuation-full-swift-tests.log 2>&1`
  exited **1** after an 85.81-second build. Six bundles finished (947, 422,
  189, 94, 63 and 98 tests), but TinyTitanRepackTests crashed with signal 5
  in the existing overflowing-payload-range regression (AUD2-025). The
  surviving footers do not represent a completed package run. Model-gated
  and platform-gated skips remain visible in the log.
- The new cross-expert rANS refusal tests failed before the guards:
  `TMPDIR="$PWD/AUDIT/scratch" xcrun swift test --no-parallel --filter
  'refusesOverlappingExpertPayloads|modelLoaderRejectsConsistentlySwappedLaterExpertGeometry'
  > AUDIT/rans-cross-expert-red.log 2>&1` exited **1**. Two methods produced
  three failed assertions: exact payload alias, partial overlap, and a
  coordinated later-expert layout/sidecar offset swap were all accepted.
  The pre-existing overflow assertion and fixtures were not changed.
- AUD2-026: the new real-handle test failed before the fix (one assertion,
  exit **1**, `m5-handle-red.log`). After writing the temporary prompt through
  its context-managed handle, the updated M5 suite passed on Python 3.13.14,
  3.14.9 and 3.15.0, **24 tests per interpreter, exit 0, zero ResourceWarnings**.
  Logs: `m5-handle-green-3.13.log`, `m5-handle-green-3.14.log`,
  `m5-handle-green-3.15.log`. Exactly one test method was added; no pre-existing
  test, assertion, fixture or import changed. No model was run.
  Commands, from the repository root: `TMPDIR="$PWD/AUDIT/scratch"
  PYTHONDONTWRITEBYTECODE=1 <interpreter> -W default::ResourceWarning -m unittest
  -v benchmark.test_tinytitan_m5_sweep`, using the three absolute interpreter
  paths in `environment.md`. The RED used Python 3.15 and selected
  `benchmark.test_tinytitan_m5_sweep.PromptSizing.test_the_prompt_handle_is_explicitly_closed`.
  Independent narrow review found no further change needed; the whole-tree
  Python gate exited **0** (`/bin/bash tools/lint.sh python`,
  `m5-python-gate.log`).
- rANS follow-up: checked payload addition, cross-expert payload separation,
  and canonical decoded geometry passed the focused 18-test run
  (`rans-validation-green.log`, exit **0**). Independent review then found an
  overly strict comparison of optional width annotations. One new acceptance
  test reproduced that regression (`rans-implicit-width-red.log`, exit **1**).
  The corrected geometry comparison and all refusal tests passed the final
  **19 tests / 3 suites**, exit **0**, in `rans-validation-final.log`
  (11.37-second build; bundle footers 13 tests / 0.209 s and 6 / 0.012 s).
  Exact command: `TMPDIR="$PWD/AUDIT/scratch" xcrun swift test --no-parallel
  --filter 'RANS|rANS'`. Follow-up independent review verified schema validation
  resolves effective widths before layer opening; no actionable concern found.
  All inputs are synthetic: BF16 geometry acceptance is not an actual BF16
  writer-to-GPU round trip, and no live model or golden inference was checked.
- Checkpoint counts: two new parameterized methods exercise negative,
  over-cap and `Int.max` metadata through indexing and lazy loading. The initial
  1 MiB test budgets could let eviction hide over-cap indexing acceptance;
  **only those two new budgets changed to `Int.max`**, and the first RED log is
  superseded. `checkpoint-payload-counts-red-valid.log` exited **1** with six
  assertions; `checkpoint-auxiliary-counts-red-valid.log` exited **1** with
  accepted invalid counts followed by signal 5 at `Int.max`. Each command was
  `TMPDIR="$PWD/AUDIT/scratch" xcrun swift test --no-parallel --filter
  <invalidPayloadByteCountsAreRefused|invalidAuxiliaryByteCountsAreRefused>`
  with one named method per run. No large payload was allocated: the saved
  payload was 16 bytes and only JSON metadata was changed.
  After metadata bounds were enforced before accounting and reads, the complete
  `PromptCheckpointStoreTests` suite passed **28 tests / 1 suite**, exit **0**,
  0.057 s after a 10.46-second build (`checkpoint-byte-counts-green.log`).
  Independent review found no scoped defect; mutation after an entry is already
  indexed is not independently tested.
- Source-size repair: complete `writeDisk` and `readPayload` methods moved from
  `PromptCheckpointStore.swift` to `PromptCheckpointStore+Disk.swift` with only
  necessary private-to-internal visibility changes. Main-agent diff inspection
  confirmed the new file and access changes; existing behavior defects remain
  open. The numeric metadata guard was added separately afterward.
- Test-change inventory for this continuation: one new Python method, one new
  rANS overlap method, one coordinated-layout refusal, one implicit-width
  acceptance, and two checkpoint numeric methods. No pre-existing assertion,
  fixture or helper changed. Toolchain formatting also changed whitespace in
  the two rANS test files (both already untracked at continuation start) and
  wrapped four tensor calls in `TwoBitExpertReadTests.swift`. Their values and
  assertions were preserved. The only new-fixture correction was the disclosed
  checkpoint budget.
- `/bin/bash tools/lint.sh > AUDIT/continuation-lint-final.log 2>&1`
  exited **1**, with **17 of 18 gates passing**. Source-size and prior visible
  formatting issues were resolved; the remaining four line-length diagnostics
  were in `TwoBitExpertReadTests.swift`, then corrected with the toolchain
  formatter. No test fixture, assertion or gate was weakened.
- After that formatting-only change, `/bin/bash tools/lint.sh >
  AUDIT/continuation-lint-verified.log 2>&1` exited **0**: **all 18 gates passed**.
  Source-length scanned 419 files (maximum 500 lines); strict Swift formatting
  scanned 692 files. This does not imply complete manual audit coverage.
- Final re-run after the coverage and audit-record updates: `/bin/bash
  tools/lint.sh > AUDIT/continuation-lint-postverification.log 2>&1` exited
  **0**, again passing all 18 gates. `git diff --check` and ledger JSON validation
  also passed; the regenerated ledger records 31 rows, none DONE, one BLOCKED.

### Complete serial run and explicit skips

`TMPDIR="$PWD/AUDIT/scratch" env -u TINYTITAN_LIBRARY_CONTRACT_MODEL xcrun swift
 test --no-parallel --enable-code-coverage >
 AUDIT/continuation-full-swift-tests-final.log 2>&1` exited **0** after a
**62.01-second build**. The unmasked status came from the command tool; it is not
printed in the log. Independent read-only review reconciled all 5,504 log lines,
including every test start/completion and non-event diagnostic; the main agent
inspected the footers, skipped-suite output and coverage artifact. No actual test
failure, compiler warning, crash or sanitizer diagnostic was found; intentional
error-path diagnostics remain in the log. This was not a sanitizer run.

| Bundle (identity inferred from suite contents) | Footer tests | Passed | Skipped | Footer suites | Footer seconds |
| --- | --- | --- | --- | --- | --- |
| TinyTitanTests | 949 | 946 | 3 | 150 | 147.800 |
| TinyTitanServerTests | 422 | 420 | 2 | 48 | 21.525 |
| TinyTitanRepackTests | 168 | 167 | 1 | 25 | 5.662 |
| TinyTitanMemoryTests | 189 | 189 | 0 | 23 | 0.659 |
| TinyTitanLibTests | 96 | 91 | 5 | 16 | 1.120 |
| TinyTitanFleetTests | 63 | 63 | 0 | 14 | 1.233 |
| ContinuityCoreTests | 98 | 98 | 0 | 10 | 0.898 |
| Total | 1,985 | 1,974 | 11 | 286 | — |

The footers include skipped tests: **1,985 counted is not 1,985 executed passes**.
Parameterized cases are not additional footer tests. The 286 counted suites
include three skipped suites. Seven XCTest headings each reported zero tests and
zero failures; the substantive tests use Swift Testing. Footer lines are 2670,
4030, 4424, 4879, 5126, 5284 and 5504.

The log names skips without reasons. The gate descriptions below come from the
current test sources, not a claim that the missing capabilities ran:

| Skipped test(s) | Count | Declared gate |
| --- | --- | --- |
| `theGPUEngineContinuesTheOraclesOwnChecks()` | 1 | Opt-in `TINYTITAN_DENSE_GPU_EQUIV` |
| `denseInstallMatchesItsSnapshot()` | 1 | Opt-in `TINYTITAN_DENSE_EQUIV` |
| `tensorOps2DFullAttentionMatchesReferenceAtTileBoundaries(_:)` | 1 | Metal `.apple10` support; unavailable on this M3 |
| `stubServer()` | 1 | Manual server lifetime in `TINYTITAN_STUB_SERVER_SECONDS` |
| `theRealInstallAnswersThroughTheFactoryAndTheAdapter()` | 1 | `TINYTITAN_SIDE_ENGINE_E2E` and a non-debug build |
| `The pinned checkpoint's own config.json parses and matches` | 1 | Existing file named by `TINYTITAN_QWEN38_CONFIG` |
| `twoEnginesOnOneDeviceRunAlternately()`, `twoGenerationsOnOneSessionBothComplete()`, `cancellingMidDecodeEndsTheGeneration()`, `unloadingShutsLiveSessionsDown()` | 4 | `LibraryContractTests`; `TINYTITAN_LIBRARY_CONTRACT_MODEL` explicitly unset |
| `aToolLoopRoundTrips()` | 1 | `ToolCallingTests`; same model gate |

The four `TwoBitExpertReadTests.swift` calls were whitespace-formatted while this
run was in flight. No values/assertions changed, but that run was not exact
post-format source certification.

### Post-format full verification

The same command, redirected to
`AUDIT/continuation-full-swift-tests-postformat.log`, exited **0** after a
**26.28-second build**. It started after the formatting change; no source or test
file changed during it. All 5,318 lines were independently reviewed, with the
main agent checking the footers and explicit skips. Fresh event reconciliation
again found **1,974 passed, 11 skipped, 1,985 counted**, with no unmatched starts
or completions. Counts and skip names in the preceding table are unchanged.
There were no unexpected errors, compiler/linker warnings or crashes; the
intentional error-path diagnostics remain visible. This is not a sanitizer run.

| Bundle | Footer seconds | Footer line |
| --- | --- | --- |
| TinyTitanTests | 163.568 | 2484 |
| TinyTitanServerTests | 21.322 | 3844 |
| TinyTitanRepackTests | 5.286 | 4238 |
| TinyTitanMemoryTests | 0.524 | 4693 |
| TinyTitanLibTests | 1.125 | 4940 |
| TinyTitanFleetTests | 1.179 | 5098 |
| ContinuityCoreTests | 0.908 | 5318 |

The measured state is still dirty HEAD `164e84a`, on the installed Swift 6.4.0
compiler. It does not establish requested Swift 6.4.1, model/golden behavior,
clean-commit CI or complete current Tier-A manual coverage.

### Coverage export gap (AUD2-031)

`xcrun swift test --show-codecov-path` returned
`.build/out/Products/Debug/codecov/TinyTitan.json`. The 252,341-byte JSON, modified
at local time 17:16:24, has **27 files: 19 ContinuityCore source files, seven tests,
and one generated entry point**. Its 4,095/4,256 covered lines (96.22%) describe
only that mixed last-bundle scope, not package or production coverage. The original
artifact is preserved as `AUDIT/continuation-codecov-last-bundle.json`.

The merged `default.profdata` reports 11,984 functions, while that JSON reports
503. The committed CI upload at `.github/workflows/ci.yml:1027-1046` checks only
that a JSON file exists, not its target coverage. A remote CI artifact from run
`39974492789` was then downloaded and inspected
(`AUDIT/ci-coverage-39974492789/TinyTitan.json`, 251,307 bytes): **27 files, 503
functions, 4,095/4,256 lines (96.22%)** — the same last-bundle scope as the local
export, so the defect is in CI output and not only in this checkout. No pipeline
was modified. The build directory also contains stale
test bundles from removed products; those were not included by globbing.

After the post-format run, this local export exited **0**, producing
`AUDIT/swift-coverage-multi-bundle.json` (199,816 bytes) with empty stderr
(`AUDIT/swift-coverage-multi-bundle.stderr.log`). The seven binaries match the
current manifest's test targets, verified with `xcrun swift package dump-package`.
Command from the repository root (line wrapping only):

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

Every exported filename is unique and under `sources/`; tests, dependencies and
generated build files are excluded. The **395 mapped source files** include 17
`TinyTitanValidation` reference/test-support files, so the aggregate is not a pure
shipping-code metric. Source-tree line coverage is **51,598/63,995 (80.63%)**;
excluding that reference target, the 378 represented shipping-source files have
**50,611/62,974 lines (80.37%)**, 5,171/6,505 functions (79.49%), and
17,015/22,937 regions (74.18%). These are measurements, not thresholds.

| Source directory | Mapped files | Covered / instrumented lines |
| --- | --- | --- |
| ContinuityCore | 19 | 2,090 / 2,214 |
| TinyTitan | 197 | 25,336 / 31,858 |
| TinyTitanCLI | 4 | 564 / 905 |
| TinyTitanFleet | 8 | 1,308 / 1,625 |
| TinyTitanFormat | 7 | 1,129 / 1,431 |
| TinyTitanKernelsC | 3 | 308 / 373 |
| TinyTitanLib | 39 | 3,651 / 6,196 |
| TinyTitanMemory | 22 | 3,011 / 3,209 |
| TinyTitanRepack | 45 | 6,352 / 7,547 |
| TinyTitanServer | 34 | 6,862 / 7,616 |
| TinyTitanValidation (reference support) | 17 | 987 / 1,021 |

**Scope limits:** 27 of the current 422 Swift/C source files have no mapping in
this export: all seven benchmark files; six executable entry files; and
`QSASelection.swift`, `FleetBrand.swift`, `FleetUsage.swift`, `ModelBackend.swift`,
`ModelDescriptor.swift`, `RunnerCounterSnapshot.swift`, `InstallProgressDisplay.swift`,
`RemoteChunkPolicy.swift`, `VerifiedInstallManifest.swift`, `ModelInstallProgress.swift`,
`BoundedScratch.swift`, `ServerVersion.swift`, `OffByMultiples.swift`, and
`Tolerances.swift`. Some contain declarations rather than instrumentable functions;
absence is not silently counted as zero or complete coverage. All 16 `.metal`
files, embedded Metal source strings, release-only paths and skipped model paths
remain outside what these LLVM percentages establish. The local export repairs
this measurement's scope; AUD2-031 stays open for durable CI export/scope checks.
