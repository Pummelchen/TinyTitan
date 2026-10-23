# The Swift language standard

What this tree is written to, what the compiler enforces, and what was
deliberately left for its own pass. Measured on 2026-09-11 with Swift 6.3.3;
the numbers come from builds, not from reading release notes.

## The baseline

| | |
| --- | --- |
| Toolchain | Apple Swift **6.4** (`swiftlang-6.4.0.34.1`), macOS 27.0, arm64 |
| Manifest | `// swift-tools-version: 6.4` |
| Language mode | `swiftLanguageModes: [.v6]` -- Swift 6 mode, so strict concurrency and the graduated 6.0/6.1/6.2 features are already errors, not warnings |

Verified again on 2026-10-23 with **Swift 6.4.1** (`swift-6.4.1-RELEASE`,
Homebrew toolchain): `swift build --build-tests` 0 errors / 0 warnings and
`swift test --no-parallel` passes all **1,947 tests** (927 + 421 + 164 + 189 +
85 + 63 + 98). Swift 6.4.1 is a patch release -- compiler, stdlib and linker
fixes, no new language features -- so 6.4 remains the language baseline.

Features that are no longer "upcoming" -- they graduated into language mode 6
and are therefore already enforced -- were confirmed against the compiler by
probing `-enable-upcoming-feature <name>` and finding the flag retired:
`StrictConcurrency`, `ConciseMagicFile`, `ForwardTrailingClosures`,
`BareSlashRegexLiterals`, `IsolatedDefaultValues`, `DisableOutwardActorInference`,
`GlobalActorIsolatedTypesUsability`, `InferSendableFromCaptures`.

## Enforced: the three that are still upcoming and the tree is clean under

Declared once in `Package.swift` as `tinytitanLanguageStandard` and applied to
**every target** (25 of 25), so a target added later cannot quietly opt out:

```swift
.enableUpcomingFeature("InferIsolatedConformances")
.enableUpcomingFeature("ImmutableWeakCaptures")
.enableUpcomingFeature("MemberImportVisibility")
```

`MemberImportVisibility` was the only one that cost anything: **12 files** used
members of a module they did not import and now name it directly (`TinyTitan` in
five, `NIOCore` in four, `Metal` in two, `Tokenizers` in two). That is the
feature doing its job -- a member reached through a transitive import is a
dependency the file never declared. The first measurement of this was 13
diagnostics; the compiler stops at the first error per file, so the honest count
came from iterating builds until clean, and the loop is what found the rest.

Verified with the flags coming from the manifest (no `-Xswiftc`): build 0
errors / 0 warnings, `tools/lint.sh` clean, **1458 tests in 226 suites** pass,
and the golden baseline is byte-identical on Ornith 4-bit.

## Deliberately not adopted, with the measured cost

| Feature | Cost measured | Why not now |
| --- | --- | --- |
| `ExistentialAny` | 63 errors, **32,332 warnings** | Requiring `any` at every existential position is a sweep of that size across the tree. The public API already spells `any` in most positions (hence the small error count); the bulk is internal. It is a mechanical, reviewable pass, but a large one -- its own task with the full gates, not a flag flip. |
| `InternalImportsByDefault` | **918 errors** | Changes what every `import` in every target means, and the fix is a `public import` at each re-export. Worth doing for the same reason `MemberImportVisibility` was (fewer accidental re-exports), but 918 sites is its own pass. |
| `NonisolatedNonsendingByDefault` | 79 errors **and** a semantics change | "Approachable concurrency" makes a `nonisolated async` function run on the *caller's* actor. That moves work between executors, so it needs a dedicated pass over the concurrency-sensitive paths -- the area this project's audit treated most carefully -- plus a fresh golden baseline, not a flag. |

## Small adoptions

Found while measuring, each mechanical and behaviour-preserving:

- **Done 2026-10-23:** `Task.sleep(nanoseconds:)` -> `Task.sleep(for:)` with
  `Duration`; the last **9 sites** (3 in `sources/`, 6 in tests) now use the
  `Duration` form (`for: .milliseconds(5)`).
- `NSLock` -> `Synchronization.Mutex`: **28 sites**. Not a blanket change and not
  a deprecation: `Mutex`'s advantage is binding the state it protects into the
  mutex (`Mutex<State>`), which is a per-site design improvement rather than a
  rename. New code should use `Mutex` and bind the state; existing locks migrate
  when the code around them is touched for another reason.

## For new code

- `any` at every existential position, `some` where the concrete type is
  enough; the compiler will require it here once the sweep above lands.
- Typed throws (`throws(SomeError)`) where a function's failure set is closed
  and callers switch on it; untyped `throws` elsewhere.
- `Synchronization.Mutex` for new mutual exclusion, with the protected state
  inside it.
- `Duration`-based `Task.sleep(for:)` and timeouts.
- `nonisolated(unsafe)` only with the invariant written directly above it, in
  the `unchecked-invariant:` form the lint gate checks.
- A new file's entry point: `main.swift` only for top-level code; an `@main`
  type lives in a file named after it (see `docs/repository-layout.md`).
