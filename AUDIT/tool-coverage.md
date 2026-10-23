# Tool-coverage and language-standard proofs — 2026-10-23

Per §1: every check delegated to a tool is proven by introducing a deliberate
violation in a scratch file (never committed) and recording the tool's
refusal. A mode that happily accepts the violating code is not enforced.

Scratch directory: a mktemp dir outside the repository; nothing below is
committed. Tool versions on this host: see `environment.md`.

## Swift language standard (Swift 6.4, Swift 6 language mode, strict concurrency)

Enforced in build config: `Package.swift:320` `swiftLanguageModes: [.v6]` +
`-warnings-as-errors` on every target (lines 19–24).

**Proof 1 — non-Sendable value captured across a concurrency boundary.**

```swift
import Foundation
final class NotSendable { var x = 0 }
func f() {
    let b = NotSendable()
    Task { print(b.x) }
    print(b.x)          // post-use makes the transfer race
}
```

`swiftc -swift-version 6 -parse-as-library -c`:

```
error: sending value of non-Sendable type '() async -> ()' risks causing data
races [#RegionIsolation::SendingRisksDataRace]
```

Also proven with an explicit `@Sendable` closure capture
(`let c: @Sendable () -> Void = { _ = b.x }`):

```
error: capture of 'b' with non-Sendable type 'NotSendable' in a '@Sendable'
closure
```

Nuance recorded: a *pure transfer* (`Task.detached { print(c.x) }` with no
post-use) compiles — that is SE-0414 region-based isolation deliberately
permitting a disconnected-region transfer, not a gap in the standard. The
risky cases above are rejected.

**Proof 2 — force unwrap fails SwiftLint `--strict` (repo config).**

```swift
let x: Int? = 1
let y = x!
```

`swiftlint lint --strict --config .swiftlint.yml`:

```
forceunwrap.swift:2:10: error: Force Unwrapping Violation: Force unwrapping
should be avoided (force_unwrapping)
```

Exit code 2 (failure). The same rule family is what the repo's own
`force-cast` gate enforces over `sources/`.

## C language standard (strict C99, -pedantic-errors, hardened warnings, -Werror)

Enforced in build config: `Package.swift:325` `cLanguageStandard: .c99` +
`TinyTitanKernelsC` `cSettings` (lines 98–113).

**Proof 1 — implicit declaration** (`int call(void) { return g2(); }`):

```
error: call to undeclared function 'g2'; ISO C99 and later do not support
implicit function declarations [-Wimplicit-function-declaration]
```

**Proof 2 — GNU extension (nested function)**:

```
error: function definition is not allowed here
```

**Control note:** a clean file missing a prototype also fails
(`-Werror,-Wmissing-prototypes`), which is itself proof the flag set is live.
Both violations are rejected with `-Werror`; exit non-zero.

## Python (Ruff 0.16.7, repo `pyproject.toml`)

**Proof 1 — bare `except:` (E722):**

```
bare-except: Do not use bare `except`   → exit 1
```

**Proof 2 — `subprocess.run` without `check` (B603 via `subprocess-run-without-check`):**

```
subprocess-run-without-check: `subprocess.run` without explicit `check`
argument   → exit 1
```

Control file passes clean. The repo config (`[tool.ruff.lint]`) carries the
families the standard names: B (bugbear, incl. mutable defaults B006), E722,
S101, F632, PT (pytest-style), plus the correctness core.

## Secret scanner (gitleaks 8.30.1)

**Proof — a fake `ghp_…` token in a scratch file:**

```
Finding: token = "REDACTED   Secret: REDACTED
```

Clean control: "no leaks found". CI runs the same tool over full history
(`supply-chain` job). Full-history run for this audit's baseline recorded in
`baseline.md`.

## Sanitizers (§1: suite, not just build)

- **C (ASan + UBSan)**: the C kernels are exercised through the Swift test
  suite (`tests/TinyTitan/` MPP kernel suite). Run record in `baseline.md`.
- **Swift (TSan)**: `swift test --no-parallel --sanitize=thread --filter` over
  the concurrency-heavy targets (LibraryContractTests is model-gated; the
  fleet/memory/server suites carry the actor surface). Run record in
  `baseline.md`.

## Delegations and their coverage

| Human check | Delegated to | Proven above |
| --- | --- | --- |
| Bare except / subprocess checks / mutable defaults | Ruff | yes |
| Force unwrap / force cast | SwiftLint + `tools/lint.sh force-cast` | yes |
| Implicit decls / GNU extensions in C | clang `-pedantic-errors -Werror` | yes |
| Non-Sendable captures | Swift 6 mode compiler | yes |
| Secrets | gitleaks | yes |
| Shell portability (bash 3.2) | `tools/lint.sh shell` + shellcheck 0.11.0 | carried from prior audit's gates; spot-reverified in baseline |
