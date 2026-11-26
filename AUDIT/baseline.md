# Baseline — NOT ESTABLISHED

Revision: `7b0901d75b3b1e79498874babf93218a617a69ac`. Root manifest declares tools version 6.4, Swift language mode v6, shared warnings-as-errors across 46 parsed targets, C99 and the strict C warning set. These are static declarations only. No resolved build settings, compiler invocations or negative-control outputs have been observed for this run. CI is configured on xcode-27 and documents earlier successful builds; neither is current-run evidence.

For every project, current build/warnings, tests pass/fail/skip, coverage, formatter/linter/analyzer findings, dependency CVEs and full-history secrets scan are **NOT RUN / UNKNOWN**, not zero. No sanitizer suite run. No comparison to historical metrics is claimed. Baseline must be taken on primary host(s) before fixes. See ledger 0001–0003, 0006 and 0009.

## Baseline measured on the primary host (2026-11-26)

Host and toolchain: see `AUDIT/evidence/0001.md`. This section is the current run's yardstick, not a historical claim.

| Gate | Command | Result |
| --- | --- | --- |
| Portable release build | `swift build -c release` | **exit 0**, 0 `warning:` lines, 202.59 s |
| Test-target build | `swift build --build-tests` | **exit 0**; 12 `warning:` lines in `tests/TinyTitan/Kernels/MoE/GGUFReadScalingTests.swift` → new task **0030** (red on warnings) |
| Lint, clean worktree | `tools/lint.sh` (20 gates, `npm ci` present) | **exit 0**, all gates `ok` |
| Release build warnings-as-errors | manifest `tinytitanLanguageStandard` on 46/46 targets | declared; Swift `Sendable` negative control fails as required |
| C strict C99 | `-pedantic-errors -Wall -Wextra … -Werror` | negative control fails as required (implicit decl, `typeof`, nested function) |
| Ruff | 0.16.7, config `pyproject.toml` | bare `except` negative control fails (`E722`); tree clean |
| SwiftLint | 0.65.1 `--strict` | force-unwrap negative control is an error (exit 2); tree clean |
| Secret scan | Gitleaks 8.30.1, full history | 2,791 commits, no leaks |
| Python deps | pip-audit 2.10.1 on `benchmark/requirements.txt` | no known vulnerabilities |
| Node deps | `npm audit --omit=dev` | 0 vulnerabilities |
| Node suite | `npm test` (Node 24.22.2) | **1 failure**: `catalog-scan` parity → new task **0031** |
| Full Swift suite | `swift test --no-parallel --enable-code-coverage` | running; counts recorded in `AUDIT/evidence/0003.md` when finished |
| Sanitizers | TSan / ASan+UBSan | not yet run (0009) |
| Model-gated suites | `models/` currently holds only the two Inkling 2 installs | those tests take their documented `.enabled(if:)` skips; not counted as passes |

The lint result is from a clean detached worktree at the audit HEAD with `npm ci` in
`plugins/dsh-tinytitan`; a lint run from the main worktree fails only in SwiftLint because the nested
agent worktrees put SwiftNIO's own checkouts inside the walk. That is an audit-tooling artifact, not a
product finding, and the final verification runs with those worktrees removed.
