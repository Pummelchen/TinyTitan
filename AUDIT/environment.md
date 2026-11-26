# Environment — discovery only

Date: 2026-11-26. Base: `7b0901d75b3b1e79498874babf93218a617a69ac`.

- Available execution host: Linux x86_64, Git 2.47.3, GCC 14.2.0, Python 3.13.5, Node v22.16.0. No xcodebuild. Swift binary present but deliberately not used for Apple build or language proof. No tool installed by audit.
- Commands attempted: `uname -sm`, `command -v xcodebuild`, `git --version`, `gcc --version`, `python3 --version`, `node --version`, tool PATH checks, `GIT_TERMINAL_PROMPT=0 git ls-remote https://github.com/Pummelchen/TinyTitan.git HEAD` (failed DNS resolution). GitHub API connector can read/write repository but is not a build host.
- Unavailable in execution host PATH: xcodebuild, swiftlint, ruff, gitleaks, shellcheck, pip-audit, codeql, clang-format. No dependency downloads attempted after DNS failure.
- Existing repository configuration (not installed/verified here): Swift 6.4 / Xcode 27; swift-format from toolchain; SwiftLint 0.65.1; Ruff 0.16.7 / Python 3.13; ShellCheck 0.11.0; Gitleaks 8.30.1; pip-audit 2.10.1; Node >=24.9 (CI requests 24); ESLint 10.11.0 and Prettier 3.9.9 (plugin manifest). Existing CI downloads pinned tools on runner xcode-27; actual tool versions and output remain unverified. C formatter/linter/dependency and SAST coverage, Metal tooling and exact Node/Python patch pin still need determination.
- No runtime systems accessed. No credentials used from prompt or written into audit files. No remote host provisioned. Named owner for environment blockers proposed as repository owner Pummelchen, pending their confirmation.

## Primary host measured (2026-11-26)

The execution host for this run is the repository owner's Apple Silicon Mac. This supersedes the discovery-only Linux record above for tasks 0001 and 0006.

| Item | Measured value |
| --- | --- |
| Architecture / OS | `Darwin arm64`, macOS 27.2.1 (build 26B131) |
| RAM | 25,769,803,776 bytes (24 GiB) |
| Free scratch at start | 255 GiB on `/System/Volumes/Data` |
| Xcode | `/Applications/Xcode.app/Contents/Developer`, Xcode 27.2 (27B52) |
| Swift | Apple Swift 6.4.2 (swiftlang-6.4.2.1.9, clang-2100.4.1.1), target `arm64-apple-macosx27.2.0` |
| Node (pinned 24) | v24.22.2 at `/opt/homebrew/opt/node@24/bin`, npm 11.19.1 |
| Node (default PATH) | v26.12.1, npm 12.2.1 |
| Python | 3.15.2 (`python3`), plus 3.14; Ruff targets `py313` and parses the tree at that floor |
| Ruff | 0.16.7 |
| ShellCheck | 0.11.0 |
| SwiftLint | 0.65.1 |
| swift-format | 604.0.0 (toolchain) |
| Gitleaks | 8.30.1 |
| pip-audit | 2.10.1 (isolated venv, installed for this audit) |
| Docker / QEMU | not needed for this audit; no live system touched |

No credentials were written to audit files. `git rev-parse --is-shallow-repository` is `false` and the full history (2,851 commits at the audit HEAD) is present, so the full-history secret scan is valid. `pip-audit` was installed into an isolated venv, not the project, and no repository dependency was changed.

Independent Phase E host (task 0008) is **not** this Mac; a second Mac remains unassigned.
