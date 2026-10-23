# Baseline — 2026-10-23, branch `audit/2026-10-23`, head `4816a84`

Regression yardstick per §3. Primary host: the development Mac (see
`environment.md`). No later state may be worse on any metric without a
justified numbered task.

| Metric | Result |
| --- | --- |
| `swift build -c release` | **complete, 589/589 modules, zero warnings** (137.8 s) — `-warnings-as-errors` on every target, so a warning is a build failure |
| `swift test --no-parallel` | **exit 0, zero failures, zero skips** (full run; final bundle reported 98 tests / 10 suites green — full log was piped through `tail`, the exit code is the gate) |
| `tools/lint.sh` (all 18 gates) | **all ok** — force-cast, unbounded-read, func-length, file-length, unchecked-sendable, converter, arch-path, stdout-clean, silent-test-skip, test-hollow, library-facade, docs, shell-portability (28 scripts under system bash 3.2.57), shellcheck 0.11.0, swiftlint 0.65.1 --strict, swift-format, javascript (eslint 10.11.0 / prettier 3.9.9 both plugin packages), python (ruff 0.16.7, 227 scripts) |
| Converter suites (`test_prepare_qwen38` + `test_qwen38_resume_e2e` + `test_prepare_agentworld`) | **87 tests, 0 failures** (42.3 s) |
| Secret scan (gitleaks 8.30.1, full history) | **1,981 commits scanned, no leaks found** (33.2 MB, 12.4 s) |
| Dependency CVEs | pip-audit runs in CI over `benchmark/requirements.txt` (numpy 2.5.3, safetensors 0.8.0, ml_dtypes 0.6.0 — "no known vulnerabilities" per CI record); Swift deps pinned by `Package.resolved` (swift-transformers, swift-nio 2.100.0 exact); npm audit in CI over both plugin lockfiles |
| Coverage | `swift test --enable-code-coverage` rides the CI serial run; llvm-cov JSON uploaded as artifact — reported, not gated (operator decision, prior audit) |

Language standard in force (recorded, not intended — proofs in
`tool-coverage.md`): Swift 6.4 / Swift 6 language mode (`swiftLanguageModes:
[.v6]`), `-warnings-as-errors` on 25/25 targets, SwiftLint 0.65.1 `--strict`;
C99 `-pedantic-errors` + full hardening set + `-Werror` on `TinyTitanKernelsC`;
Ruff 0.16.7 at the 3.13 floor; shellcheck 0.11.0 + bash-3.2 portability gate.

Model-gated checks NOT run at baseline (no gate may fetch a model): golden
baseline, `TINYTITAN_LIBRARY_CONTRACT_MODEL` suite. Golden baselines are
checked at release time against installs present under `models/` (ornith-4,
ornith-8, qwen38-4 on this host).

Sanitizer runs for this audit (§1: suite, not just build): TSan and C-kernel
ASan/UBSan runs are scheduled as part of Phase B/E — recorded here when run.

## Prior-audit deltas (scope note)

The 2026-10-06 audit (210/210 closed) covers the tree up to `docs/audit-ledger.json`.
Since then: 163 commits touching `sources/` (+13,330 / −1,547 lines over 132
files) — releases v5.19–v6.0, the Project Maya work. This audit's manual Tier-A
passes concentrate on that unaudited surface; the tool gates above cover the
whole tree at every commit.
