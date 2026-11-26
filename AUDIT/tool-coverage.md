# Tool coverage — NOT ESTABLISHED

No deliberate negative-control run was executed. Required probes on pinned primary hosts: Swift non-Sendable actor/Task boundary; SwiftLint force unwrap; C99 implicit declaration and GNU nested function/typeof; Ruff bare except; every other check delegated to a tool. Record exact snippets, versions, commands, nonzero exit and diagnostics, then remove scratch inputs. Existing configuration comments describing prior audit proofs do not substitute for current proof. For example manifests and direct compiler scripts see ledger 0004–0005. No check delegated as proven yet.

## Negative controls measured on the primary host (2026-11-26)

Every probe ran with the repository's own configuration and the pinned tool. Scratch inputs live outside the tree and were removed after the run; no probe code is committed.

| Delegated check | Probe | Command (abridged) | Result |
| --- | --- | --- | --- |
| Swift 6 Sendable enforcement | `final class NotSendable` passed to `func requireSendable<T: Sendable>` | `swiftc -swift-version 6 -typecheck` | **fails as required**: `error: type 'NotSendable' does not conform to the 'Sendable' protocol` (exit 1) |
| SwiftLint force unwrap | `let forced = optional!` | `swiftlint lint --strict --config .swiftlint.yml` | **fails as required**: `error: Force Unwrapping Violation ... (force_unwrapping)`, exit 2 |
| C99 implicit declaration + GNU extensions | call to `typeof(int)` / nested function / undeclared call | `clang -std=c99 -pedantic-errors -Wall -Wextra -Wshadow -Wconversion -Wsign-conversion -Wcast-qual -Wwrite-strings -Wformat=2 -Wstrict-prototypes -Wmissing-prototypes -Werror -c` | **fails as required**: 7 `error:` diagnostics, exit 1 |
| Ruff bare except | `try: pass` / `except: pass` | `ruff check --config pyproject.toml` | **fails as required**: `E722`, exit 1 |

Note on the Swift probe: a non-Sendable passed once across an actor/Task boundary is accepted under Swift 6's region-based isolation when it is not used again (by design, SE-0414). The definitive complete-concurrency probe is the `Sendable` generic constraint above, which fails. Do not weaken it to an actor-crossing snippet that Swift 6 legitimately accepts.
