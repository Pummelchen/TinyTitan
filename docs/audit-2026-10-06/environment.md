# Audit environment

Recorded at the start of the pre-production audit on 2026-10-06, per `AUDIT` prompt §1
and §1b. One pinned tool per language, on one primary host. Anything a gate pins is
checked against what is actually installed, not against what is expected to be there.

## Primary host

| | |
| --- | --- |
| Host | `MacBook Pro (local)` — the only Apple-silicon host in scope |
| OS | macOS 27.0.0 (Darwin), `sw_vers` productVersion 27.0 |
| Architecture | `arm64`, 8 cores, 24 GB physical RAM |
| Free disk on `/` | see `Baseline state` below — the model store is the dominant user |
| Checkout | `/Users/andreborchert/Downloads/TinyTitan` |
| Branch | `audit/2026-10-06`, based on `origin/main` at `9bb9051` |
| Model installs under `models/` | `qwen3.8-flash-next_125B_A6B_4Bit`, `qwen3.8-flash-next_125B_A6B_MTP_4Bit` |

This is an Apple-silicon Swift/Metal product, so the Mac is the primary host for the
build, the suite and every measurement (§1b). The Intel VPS is Linux/x86 and cannot
build or run this tree; it is therefore *not* a verification host for Phase E, and the
independent-host requirement is recorded as a finding rather than worked around.

## Toolchain

| Language | Tool | Version | Install method | Status vs the repo pin |
| --- | --- | --- | --- | --- |
| Swift | Xcode | 27.0 (Build 27A266a), `/Applications/Xcode.app` | preinstalled | matches the hard rule in `AGENTS.md` |
| Swift | Swift compiler | 6.4 (`swiftlang-6.4.0.34.1`), target `arm64-apple-macosx27.0.0` | the Xcode toolchain | matches |
| Swift | `swift-format` | the toolchain's own, at `xcrun --find swift-format` | Xcode | the committed `.swift-format` is the config |
| Swift | SwiftLint | 0.65.1 | `/opt/homebrew/bin/swiftlint` | matches `SWIFTLINT_PIN` in `tools/lint.sh:627` |
| C | clang | Homebrew clang 23.1.2 (arm64) | Homebrew | the C standard is set by `Package.swift`, not by the compiler on `PATH` |
| Shell | ShellCheck | 0.11.0 | `/opt/homebrew/bin/shellcheck` | matches `SHELLCHECK_PIN` in `tools/lint.sh:584` |
| Shell | bash | `/bin/bash` 3.2.57 (factory) | macOS | the portability target every script is held to |
| Python | python | 3.14.8 (Homebrew) | `/opt/homebrew/bin/python3` | above the `PYTHON_FLOOR="3.13"` the gate declares |
| Python | ruff (global) | 0.16.10 | `/opt/homebrew/bin/ruff` | **does not match** `RUFF_PIN="0.16.7"` |
| Python | ruff (audit venv) | 0.16.7 | `python3 -m venv /tmp/tt-audit/venv && pip install ruff==0.16.7` | matches; put on `PATH` ahead of Homebrew for every gate run |
| Python | ruff (pipx) | 0.16.7 | `~/.local/bin/ruff` -> `~/.local/pipx/venvs/ruff/bin/ruff` | matches; survives `/tmp` being cleared, unlike the venv above |
| Python | pip-audit | 2.10.1 | same venv | new tool, installed for the §3 CVE baseline |
| JS/TS | node / npm | v26.10.0 / 11.19.1 | Homebrew | the plugin packages pin their own eslint (`ESLINT_PIN="10.11.0"`) via committed lockfiles |
| Secrets | gitleaks | 8.30.1 | `/opt/homebrew/bin/gitleaks` | uses the committed `.gitleaks.toml` |

Nothing was installed into the global toolchain. The two new tools (pinned ruff,
pip-audit) first lived in `/tmp/tt-audit/venv`, the same way CI runs the converter suites
in a venv, so this host's Homebrew ruff stayed untouched and the gate still saw the
version it pins. The pinned ruff is now also a `pipx` install under `~/.local`, which is
outside Homebrew but survives the venv being cleaned up, since `/tmp` does not survive a
reboot. Either one works for a gate run; `PATH="$HOME/.local/bin:$PATH" tools/lint.sh
python` is the shorter of the two. Clean-up: `rm -rf /tmp/tt-audit` at the end of the run.

## Deviations found while writing this file

1. The global `ruff` is 0.16.10 while `tools/lint.sh` pins 0.16.7, so `tools/lint.sh
   python` **fails on this host as configured** — by design, the gate refuses a version
   it does not pin. Resolved for the audit by the venv above; the standard is the repo
   pin, so the pin is not moved to match the host.
2. `swift --version` reports target `arm64-apple-macosx27.0.0` while `Package.swift`
   sets the release deployment target to macOS 26.0. Not a defect: the compiler's host
   triple and the product's deployment target are different fields, and `AGENTS.md`
   states the shipped baseline explicitly.
