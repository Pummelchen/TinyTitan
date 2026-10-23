# Baseline — the yardstick

Measured once, on the primary host, before any fix. Every later state must be at
least as good on every metric here; a regression needs a justified numbered task.
Companion records: [environment.md](environment.md) for the host and tool pins,
[tool-coverage.md](tool-coverage.md) for the proofs that these checks actually fire.

## Toolchain actually in force — not the one intended

| Measurement | Value |
| --- | --- |
| `sw_vers -productVersion` | 27.0.1 |
| `xcodebuild -version` | Xcode 27.0, build 27A266a |
| `swift --version` | Apple Swift 6.4 (swiftlang-6.4.0.34.1 clang-2100.3.34.1) |
| `swift --version` target triple | `arm64-apple-macosx27.0.0` |
| Swift language mode | `.v6` — `Package.swift:316` `swiftLanguageModes: [.v6]` |
| Warnings as errors | `.unsafeFlags(["-warnings-as-errors"])` — `Package.swift:23` |
| C standard | `.c99` — `Package.swift:321` `cLanguageStandard: .c99` |
| C warning flags | `-O2 -pedantic-errors -Wall -Wextra -Wshadow -Wconversion -Wsign-conversion -Wcast-qual -Wwrite-strings -Wformat=2 -Wstrict-prototypes -Wmissing-prototypes -Werror` — `Package.swift:95-108` |

The compiler's own target is `macosx27.0.0` while the shipped products deploy to
`macos26.0`; that is a deployment-target setting, not a toolchain mismatch, and it
is recorded here so a later reader does not mistake it for one.

## Build

| Command | Result | Warnings |
| --- | --- | --- |
| `swift build -c release` (clean, from `git clean` state) | Build complete! (111.64 s), 932 steps | **0** |
| `swift build` (debug) | Build complete | 0 |

Zero is not luck: `-warnings-as-errors` makes any warning a build failure, so the
count and the exit code are the same measurement.

## Swift test suite

`swift test --no-parallel --enable-code-coverage` — log
`/tmp/tt-audit/swift-test-baseline.log` (scratch, not committed).

| Metric | Value |
| --- | --- |
| Tests executed | **1528** |
| Suites | **234** |
| Test bundles | 7 |
| Failed | **0** |
| Skipped | **2** |
| Wall time | 09:04 → 13:05 run, suites individually sub-second to ~1 min |

The two skips are the model-gated tests (`theGPUEngineContinuesTheOraclesOwnChecks`,
`denseInstallMatchesItsSnapshot`); they need a `.ssdai` install and are covered by
the gated runs in the ledger, not by this suite.

## Coverage of production sources

`xcrun llvm-cov report <each .xctest binary> -instr-profile=.build/out/Products/Debug/codecov/default.profdata`,
rows restricted to `sources/`, aggregated per file as the best across bundles (a
source file can be exercised from more than one test bundle).

| Metric | Value |
| --- | --- |
| Files under `sources/` instrumented | 342 |
| Lines covered | **42182 / 54206 = 77.8%** |
| Regions covered | **14035 / 19635 = 71.5%** |

The coverage profile lands under `.build/out/Products/Debug/codecov/`, not
`.build/debug` — this build uses a shared `out` scratch path. An earlier probe that
searched only `.build/debug` found nothing and would have been recorded as "coverage
not instrumented", which is the wrong conclusion.

Lowest-covered production files, which is the part of the number that matters:

| File | Line coverage |
| --- | --- |
| `TinyTitanLib/ServerModelSession+Generation.swift` | 0.0% (0/441) |
| `TinyTitanLib/ServerModelSession+Loading.swift` | 0.0% (0/368) |
| `TinyTitanLib/ServerModelSession+PromptCache.swift` | 0.0% (0/209) |
| `TinyTitanLib/ServerModelSession+Diagnostics.swift` | 14.8% |
| `TinyTitanLib/ServerModelSession.swift` | 28.6% |
| `TinyTitanLib/Session.swift` | 30.5% |
| `TinyTitanLib/Engine.swift` | 33.3% |
| `TinyTitanCLI/Run.swift` | 39.1% |
| `TinyTitanLib/ModelSessionPlan.swift` | 41.6% |
| `TinyTitanFormat/SSDAIManifestV1.swift` | 55.6% |

Ten files have zero covered lines. Three of the four largest are the
`ServerModelSession` model-gated trio: the unit suite deliberately never loads a
model, so those lines only run under `LibraryContractTests` (needs
`TINYTITAN_LIBRARY_CONTRACT_MODEL`) or `tools/golden-baseline.sh`. The rest of the
zero set — `ArchConfig+Manifest.swift`, `ProcessMemory.swift`,
`SharedExpertAffineQuant.swift`, `PLEBlock.swift`, `Qwen35DenseFamily.swift`,
`RealForwardRunner+ANE.swift`, `WatchdogSupervisor.swift` — has no such excuse and is
the L6 finding.

## Linters and analysers

`tools/lint.sh` — eleven gates, all clean at baseline. The pins are the repo's own,
and each gate fails when a different version is on `PATH`:

| Gate | Result | Detail |
| --- | --- | --- |
| force-cast | ok | no undocumented `as!` / `try!` under `sources/` |
| func-length | ok | 14 baselined, **0 new**, 2091 scanned |
| unchecked-sendable | ok | 0 undocumented, 0 new |
| converter-expert-order | ok | experts file at its own index |
| arch-path | ok | no hardcoded `arm64-apple-macosx` build path |
| shell-portability | ok | 24 scripts, system bash 3.2.57 |
| shellcheck | ok | 0.11.0, 24 scripts, no warnings |
| swiftlint | ok | 0.65.1, `--strict` clean |
| swift-format | ok | toolchain `swift-format`, `--strict` clean |
| javascript | ok | eslint 10.11.0 + prettier 3.9.9, both plugin packages, node v26.10.0 |
| python | ok | ruff 0.16.7, check and format clean, parses under 3.13 |

One host deviation, recorded rather than hidden: this Mac's Homebrew ruff is
0.16.10, not the `RUFF_PIN=0.16.7` the gate demands, so `tools/lint.sh python` fails
on this host as configured. The pinned version was installed in a scratch venv and
put first on `PATH` for the run above — the pin itself was not moved. See
[environment.md](environment.md).

## Python — the converter and benchmark suites

`cd benchmark && /tmp/tt-audit/converter-venv/bin/python -m unittest
test_prepare_qwen38 test_qwen38_resume_e2e test_prepare_agentworld`

| Metric | Value |
| --- | --- |
| Interpreter | CPython 3.13.16 (the declared floor) in a venv |
| Pins | numpy 2.5.3, safetensors 0.8.0, ml_dtypes 0.6.0 — from `benchmark/requirements.txt` |
| Tests | **87**, `OK`, 36.488 s |
| Failed | 0 |
| Skipped | 0 |

Measured against the claim in `benchmark/requirements.txt:4-5` ("299 tests, 52
skipped"): **that comment is stale**. The three suites run 87 tests and skip none on
this host. The comment is a finding, not a number to reconcile.

Running the same command with the venv that lacks the three dependencies produces a
different and worse result — 75 tests, 74 skipped, **1 failure** — because
`test_qwen38_resume_e2e` asserts on converter output that the missing-dependency
message replaces. A dependency-absent run should skip, not fail; recorded as an L6
task.

## Dependency CVEs

| Language | Scanner | Version | Result |
| --- | --- | --- | --- |
| Python | `pip-audit -r benchmark/requirements.txt --no-deps` | 2.10.1 | No known vulnerabilities found (3 pins) |
| JavaScript | `npm audit` in `plugins/dsh-lan-manager` and `plugins/dsh-tinytitan` | npm 11.x / node v26.10.0 | 0 vulnerabilities (info/low/moderate/high/critical all 0), both packages |
| Swift | — | — | **No scanner exists in the toolchain**, and there is no `.github/dependabot.yml`, so no upstream CVE feed is consulted for `swift-nio 2.100.0` or `swift-transformers`. `Package.resolved` pins exact versions and CI builds from it; that is a pin, not a vulnerability check. Recorded as an L0 task with the gap stated. |

## Secret scan — full history, once

`gitleaks detect --source . --config .gitleaks.toml --redact` (gitleaks 8.30.1,
installed by Homebrew; the committed config is the repo's own).

| Metric | Value |
| --- | --- |
| Commits scanned | **1221** (full history) |
| Bytes scanned | ~18.25 MB |
| Leaks | **0** — "no leaks found", exit 0 |

History is immutable under §0, so this runs once and is never re-scanned into a
different answer.

## Not measured at baseline, and why

- **Golden baseline** (`tools/golden-baseline.sh --check`) and
  **`LibraryContractTests`** are model-gated. A model install is an
  operator-requested job and may never be fetched to satisfy a gate
  (`AGENTS.md` "Verification uses only the models already installed under
  `models/`"), so they are ledger tasks with their preconditions listed, not
  baseline numbers invented here.
- **Sanitizers** (TSan over the suite, ASan/UBSan over the C kernels) run after the
  S0/S1 fixes, because a sanitizer report against code that is about to change is
  re-measured work. Tracked as its own task.
- **VPS / Linux**: this project is Apple Silicon only, so the Linux host cannot
  produce any of these metrics and is not used for the baseline.
