# Audit environment — 2026-10-23

Primary host (Apple-platform work per §1b): the development Mac.
Branch: `audit/2026-10-23` (§0; merge to `main` only via PR after Phase E).

## Host

| Item | Value |
| --- | --- |
| Model | base 8-core M3 MacBook Pro, 24 GB |
| macOS | 27.0.1 (26A434) |
| Xcode | 27.1 (27A9275) |
| Swift (Xcode) | 6.4 (swiftlang-6.4.0.34.1), `xcrun swift` — the project's pinned toolchain |
| Swift (required 6.4.1) | Homebrew `swift` 6.4.1, keg-only: `/opt/homebrew/opt/swift` → `Cellar/swift/6.4.1`; `swift-6.4.1-RELEASE`, target `arm64-apple-macosx27.0.0` |

## Toolchain (one pinned version per language; §1)

| Language | Formatter | Linter | Dep/CVE scanner | Type checker | SAST | Sanitizer |
| --- | --- | --- | --- | --- | --- | --- |
| Swift | `swift-format` 604.0.0 (toolchain's own; committed `.swift-format`) | SwiftLint 0.65.1 `--strict` (pinned by `tools/lint.sh`) | — (no Swift dep scanner exists; deps pinned by `Package.resolved`, audited manually in L0) | Swift 6 language mode compiler (`swiftLanguageModes: [.v6]`, complete strict concurrency by default in that mode) | CodeQL (CI job on `main`) | `swift test --sanitize=thread` (CI `thread-sanitizer` job) |
| C | — | clang via build flags (see standard below) | — (no third-party C deps; kernels are in-tree) | clang (`-std=c99 -pedantic-errors -Werror`) | CodeQL (CI) | ASan + UBSan on the C kernel suite — see proof below |
| Python | Ruff 0.16.7 (formatter + linter, `--fix`) | Ruff 0.16.7 (pinned by `tools/lint.sh python`) | pip-audit (CI pins 2.10.1; local run recorded below) | none (de-scoped by §1 Python basic tier) | — | — |
| Shell | shfmt? no — `shell-portability` gate + shellcheck 0.11.0 (pinned) | ShellCheck 0.11.0 (pinned by `tools/lint.sh`) | — | — | — | — |
| JavaScript | Prettier (each plugin package's own) | ESLint (each plugin package's own, `npm ci` first) | `npm audit` (CI, both plugin lockfiles) | — | — | — |

Secret scanner: gitleaks 8.30.1 over full history — run in CI (`supply-chain` job,
verified clean over 1643 commits on 2026-10-11); re-run locally for this audit's
baseline (recorded in `baseline.md`).

Install methods: swiftlint, ruff, shellcheck, gitleaks via Homebrew on PATH;
pip-audit via a project venv (not on the user PATH — CI runs 2.10.1). Node v26.12.0
/ npm 12.2.0 for the plugin packages' own eslint + prettier (`npm ci` in
`plugins/dsh-lan-manager/` and `plugins/dsh-tinytitan/`). Swift 6.4.1 comes from
the Homebrew formula `swift` (keg-only, 2.5 GB, installed 2026-10-17); it is not
on `PATH`, so `xcrun swift` still resolves to Xcode's 6.4.0 and the 6.4.1 runs
below invoke `/opt/homebrew/opt/swift/bin/swift` explicitly.

## Language standards in force (build-config-enforced, §1)

- **Swift 6.4, Swift 6 language mode**: `Package.swift` line 320
  `swiftLanguageModes: [.v6]` — package-wide, not per-invocation. Swift 6 mode
  enables complete strict concurrency checking by default; no target downgrades
  it (no `StrictConcurrency` downgrade setting anywhere in the manifest).
- **`-warnings-as-errors`**: `Package.swift` lines 19–24, applied via
  `tinytitanLanguageStandard` to **every** target (25/25 at time of writing),
  so a new target cannot opt out.
- **SwiftLint `--strict`**: `tools/lint.sh` (pinned 0.65.1).
- **C99 strict**: `Package.swift` line 325 `cLanguageStandard: .c99`, plus
  `TinyTitanKernelsC` `cSettings` (lines 98–113): `-pedantic-errors -Wall
  -Wextra -Wshadow -Wconversion -Wsign-conversion -Wcast-qual -Wwrite-strings
  -Wformat=2 -Wstrict-prototypes -Wmissing-prototypes -Werror`.
- **Python**: Ruff 0.16.7 via `pyproject.toml`, run by `tools/lint.sh python`
  (parse at the declared 3.13 floor + ruff check).

Proofs that these standards actually fail the build on a violation are in
`tool-coverage.md` (§1 language-standard proof + tool-coverage proof).

## Requested compatibility extension (2026-10-23)

The operator requires Swift **6.4.1** and Python **3.13, 3.14 and 3.15**.

The earlier inventory searched Xcode's and CommandLineTools' toolchains only and
reported no 6.4.1 compiler. That negative finding was wrong: Homebrew's
keg-only `swift` formula **6.4.1** is installed and reports
`Swift version 6.4.1 (swift-6.4.1-RELEASE)`. Verified with the package:

```bash
/opt/homebrew/opt/swift/bin/swift --version
/opt/homebrew/opt/swift/bin/swift package --version
TMPDIR="$PWD/AUDIT/scratch" /opt/homebrew/opt/swift/bin/swift build \
  --scratch-path .build/brew641 --disable-automatic-resolution
TMPDIR="$PWD/AUDIT/scratch" env -u TINYTITAN_LIBRARY_CONTRACT_MODEL \
  /opt/homebrew/opt/swift/bin/swift test --scratch-path .build/brew641 --no-parallel
```

- Build: **exit 0**, `Build complete! (72.81 sec)`, 1,258 tasks,
  **zero compiler errors**.
- Tests: **exit 0**, seven bundles, 1,985 counted tests, the same 11 skips as the
  6.4.0 run, **zero compiler errors** (`AUDIT/brew641-full-tests.log`).
- The keg's SwiftPM emits `warning: Unable to locate libSwiftScan` and falls back
  to the frontend scanner; it does not affect the build result.
- **49 linker warnings** appear because the keg links the package's macOS 26.0
  deployment target against system Swift dylibs built for 27.0
  (`libswiftCore`, `libswift_Concurrency`, …). They are not source diagnostics;
  `-warnings-as-errors` still governs compilation. Recorded as AUD2-039 because
  the baseline had zero build warnings.
- The language standard is unchanged and still enforced: `swiftLanguageModes:
  [.v6]`, upcoming features, and `-warnings-as-errors` on every target.

Xcode's own toolchain remains 6.4.0; no Xcode-bundled 6.4.1 was found. AUD2-022
is closed on the 6.4.1 evidence above, with the linker-warning regression tracked
separately.

| Python runtime | Location | Converter dependencies |
| --- | --- | --- |
| 3.13.14 | `~/.local/share/uv/python/cpython-3.13.14-macos-aarch64-none/bin/python3.13` | absent |
| 3.14.9 | `/opt/homebrew/bin/python3.14` | NumPy 2.5.4, safetensors 0.8.0, ml_dtypes 0.6.0 |
| 3.15.0 | `/opt/homebrew/bin/python3.15` | NumPy 2.5.4, safetensors 0.8.0, ml_dtypes 0.6.0 |

All 226 tracked Python files compiled in memory on these three interpreters.
That establishes syntax only. The completed runtime matrix is recorded in
`python-matrix-summary.json`: 108 serial suite invocations per interpreter.
Python 3.14 and 3.15 each reported 2,077 tests, zero failures/errors/skips and
exit 0; each also emitted 12 M5-sweep `ResourceWarning` messages (AUD2-026).
Python 3.13 reported 1,728 passes, 18 failures, 10 errors and 87 skips, exit 1:
the missing converter/analysis dependencies leave that runtime verification
incomplete, not green. Seven reported errors are unittest loader placeholders.
The off-pin NumPy 2.5.4 stack on 3.14/3.15 does not certify the committed 2.5.3
stack; no dependency was installed or fetched to turn an unavailable gate into
a pass. Real-model and CoreML checks were not part of this matrix.

The AUD2-026 follow-up changed the M5 sweep and added one real-handle regression
test. The updated 24-test suite passed on all three interpreters with no
`ResourceWarning` (`m5-handle-green-3.13.log`, `m5-handle-green-3.14.log`,
`m5-handle-green-3.15.log`). The original matrix still describes the earlier
23-test source state, not this revised suite.

## Continuation verification limits

The complete serial Swift run now has an unmasked exit status and an explicit
skip inventory in `baseline.md`; `continuation-lint-verified.log` records all
18 lint gates passing. These remain dirty-tree results; the separate Swift 6.4.1
build and suite above change the compiler question, not the Phase E question.
No model or dependency was installed.

The post-format rerun also exited 0, with unchanged pass/skip counts. Its separate
multi-object coverage export is recorded in `baseline.md`: SwiftPM's own JSON
contains only the last bundle (AUD2-031), while the corrected local export names
all seven current test binaries and discloses unmapped sources and Metal. Neither
percentage establishes whole-repository coverage. A CI artifact from run
`39974492789` was downloaded and contains one bundle / 27 files, matching the
local last-bundle scope; the durable export gap is still open and no pipeline was
changed. The same continuation recorded a clean release build (306/306 tasks,
106.09 s, zero warnings), a byte-identical `ornith-8` golden and four passing
`LibraryContractTests` against the installed 4B; `audit_2310.md` carries the
commands and footers.

## Prior audit

The 2026-10-06 pre-production audit under this same prompt closed 210/210 rows
(ledger retained at `docs/audit-ledger.json`). This audit re-runs the full
procedure against the tree as of 2026-10-23 (`4816a84`), which carries releases
v5.19–v6.0 since that audit closed — the Maya work (interior checkpoints,
Qwen3.8 auxiliary payloads, calibration command, machine-size page-cache policy,
Qwen3.8 4-bit tuning) is the unaudited surface.
