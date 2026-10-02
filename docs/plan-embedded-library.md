# TinyTitan as an embedded LLM engine — plan

**Status: P1 is implemented (phases A1 and A2), and the library ships as
binaries.** Written 2026-10-02, after 5.15. **Revised 2026-10-02:** the
dependency premise in §2 was tested rather than assumed, and it is false — a
consumer package already resolves, builds, links and opens an install against the
released tag. P0 therefore shrank from "remove the blockers" to "keep it proven".
§6, §7 and §8 carry the correction; §2 carries the measurement.

**What is implemented.** The library is **`TinyTitanLib`** (the owner's decision;
the draft called it `TinyTitanKit`), a target and the one library product, with
the generation orchestration moved into it out of `TinyTitanServerCore`. The
facade in §4 is public and everything else in the target is `package`; the
orchestrator imports no NIO and writes nothing to stdout.

- **A1** — the server serves through the library.
- **A2** — so does the CLI: `Run.swift` builds an `Engine` and a `Session`, and
  holds no tokenizer, no `Model.load` and no runner. Its output is byte-identical
  to the pre-move binary on three saved baselines (two raw-completion prompts and
  one chat request), which is the check that the facade can express what the
  engine actually does rather than most of it.
- **S1.5** — every release carries `libTinyTitanLib.a` and
  `libTinyTitanLib.dylib` with their module set, module maps and resource
  bundle, in a second archive built by `tools/build_library.sh` and published by
  `release.sh`. A consumer compiled against the extracted archive (no SwiftPM)
  was run against a real install in both link forms before this was written.

The facade's remaining gaps are listed at the end of §4.

The ask: let another Swift program use TinyTitan as its LLM engine — the way a C++
program links a `.dll` — instead of shelling out to `TinyTitanCLI` or speaking HTTP
to `TinyTitanServer`. This is the design record for that, with the blockers named
rather than discovered later. It is a plan, not a promise: nothing in it is
supported until the release notes say so.

## 1. The goal, and what it is not

**Goal.** A program adds one package dependency (or links one binary), passes a
`.ssdai` directory and an `MTLDevice`, and streams tokens. No subprocess, no HTTP,
no model reimplementation, no fork of the engine: the same kernels, format reader,
KV cache and sampler the shipped CLI uses, behind a supported and versioned API.

**Not in this plan:**

- **A network API.** That already exists (`TinyTitanServer`, OpenAI-compatible).
  The library is for programs in the same process.
- **A GUI or a new user-facing product.** `AGENTS.md` says the product is the
  engine plus its loopback server; the library is a *packaging* of that engine,
  not a second product. The rule that keeps this honest is in §6: the CLI and the
  server must be rebuilt on the library, so no API exists that they do not use.
- **Cross-platform support.** The floor is macOS 26 on Apple silicon, and it stays
  that way in v1: the kernels are Metal + NEON, and the ANE sidecars are
  Core ML. A Linux story would be a separate project.
- **Weights distribution.** The engine is Apache-2.0; the weights are not part of
  the library and are never downloaded by it. The installer keeps that job.
- **Fine-tuning or training.**

## 2. Where the code is today

Three facts decide most of the work.

**Fact 1 — the engine is already a library target.** `Package.swift` ships
`.library(name: "TinyTitan")` and `.library(name: "TinyTitanFormat")` beside the
executables, and every executable is a thin shell over a `*Core` target
(`TinyTitanCLICore`, `TinyTitanServerCore`, `TinyTitanRepackCore`). The engine
entry point is already shaped for an embedder:

```swift
public static func load(
    directoryURL: URL,
    device: MTLDevice,                    // the caller owns the device
    expecting: ArchConfig = .qwen36_35B_A3B,
    streamingMode: ExpertStreamingMode = .pread(slotCount: 32),
    expertCachePolicy: ExpertCachePolicy = PreadExpertStreamer.cachePolicyDefault,
    integrityPolicy: ModelIntegrityPolicy? = nil,
    loadStats: UnsafeMutablePointer<ModelLoadStats>? = nil
) throws -> Model
```

**Fact 2 — the package is already consumable as a dependency. The opposite was
assumed, and it is wrong.** `AGENTS.md` says `.unsafeFlags` is "why this package
cannot be consumed as a dependency; it is an application package and nothing
depends on it", and the first draft of this plan repeated it. Measured
2026-10-02 on Swift 6.4 / Xcode 27, with consumer packages that are not this
repository:

| Consumer | Requirement | Result |
| --- | --- | --- |
| `examples/embedded` (in this repo), path dependency | `.package(path: "../..")` | resolves, builds, links, and opens `models/qwen3.6_35B_A3B_4Bit` through `Model.load` |
| throwaway package, `file://` URL at the `v5.15` tag | `from: "5.15.0"` | resolves and builds, debug and release |
| throwaway package, `https://github.com/Pummelchen/TinyTitan` | `from: "5.15.0"` | resolves, builds in both configurations, and `swift run` prints |

SwiftPM's refusal still exists as a diagnostic in that toolchain ("the target …
in product … contains unsafe build flags"), so the rule has not been deleted — it
did not fire for any of those consumers, in either configuration. Nothing needs
to be removed before a consumer can build, which is why P0 (§6, §8) is now a
regression test instead of a refactor. Both `.unsafeFlags` uses stay, and a
consumer inherits both:

| Where | Flags | Why it stays |
| --- | --- | --- |
| `tinytitanLanguageStandard`, applied to all 23 targets | `-warnings-as-errors` | A warning that only appears in a build log is a check nobody runs (the flag is part of the standard, not a preference) |
| `TinyTitanKernelsC` | `-O2`, `-pedantic-errors`, `-Wall -Wextra -Wshadow -Wconversion -Wsign-conversion -Wcast-qual -Wwrite-strings -Wformat=2 -Wstrict-prototypes -Wmissing-prototypes -Werror` | SwiftPM compiles C at `-Os` in release; `-O2` is a measured **1.21x** on the CPU int8 GEMV (2.35 → 1.94 ms per pass). A dependency's `cSettings` are applied in the consumer's build — verified with a fresh-scratch `swift build -v` in the fixture, which shows the kernel compiled at `-O2` |

The fixture and `tools/embedded-dependency-check.sh` are what keep that true:
they run the real thing and fail loudly if a later manifest or toolchain change
reinstate the refusal.

**Fact 3 — the shape is right, the boundaries are not.** The engine target has
1,328 `public` declarations — everything is public, nothing carries
`@available`, and there is no API policy. `TinyTitanFormat` is the other extreme
and is worth its own line: all 164 of its declarations are `package`, so the
library product of that name exposes **nothing** to a consumer — an embedder
cannot read or validate the manifest of the install it is being handed. Meanwhile
the parts an embedder needs
around the model are in the *server* target: prompt/chat templating
(`ServerModelSession+Loading`, `ServerPromptCache`, `OpenAIRequestValidator`) and
the request orchestrator (`ServerModelSession.generate(_:onEvent:) async throws ->
ServerCompletion`), which lives in `TinyTitanServerCore` — a target that depends on
NIO for its HTTP layer. A consumer of the library today would pull in an HTTP
server to render a Qwen chat template. NIO is *not* used by the generation path
itself (no NIO imports in the session/generation files), so this is a target
boundary that needs moving, not a rewrite.

Two more facts that matter for a binary distribution: the Metal shader library is
a target resource reached through `Bundle.module` (`resources: [.copy("Metal")]`),
and `Model` is not formally `Sendable` — several call sites use
`nonisolated(unsafe)`, and the server serialises work with a slot plus the
`ForwardStepGate` actor.

## 3. The target architecture

```
TinyTitanLib          the supported facade (public) + the orchestrator it drives (package)
   └── TinyTitan        the runtime: TinyTitanFormat + TinyTitanKernelsC + the forward runner
TinyTitanServerCore   HTTP + OpenAI/Anthropic translation, rebuilt ON TinyTitanLib
TinyTitanCLICore      the CLI, rebuilt ON TinyTitanLib
TinyTitanC (optional) the C ABI: tinytitan.h over an opaque handle
```

The runtime target keeps the name `TinyTitan` (the first draft called it
`TinyTitanEngine`; renaming it would churn every import in the tree for no
behavioural gain, and §9.5 is still open on naming). What matters is the layer
above it: `TinyTitanLib` is a **product**, and the runtime is not — see
`docs/repository-layout.md`.

- **`TinyTitanLib` is additive and thin.** It does not re-export engine internals;
  it names the handful of types a consumer may depend on. Everything else stays
  internal or `package`. This is what makes a versioning policy possible at all.
- **The server becomes a client of the library.** That is the cheapest way to
  prove the facade is sufficient: if `TinyTitanServerCore` needs something the
  facade cannot express, the facade is wrong, and the golden baselines — which run
  through the CLI — keep covering the new path byte-for-byte.
- **No hidden globals.** The caller owns the `MTLDevice`; the engine owns its
  buffers and caches; two engines in one process must work (§5).

## 4. The API to freeze (v1 sketch)

Names are provisional; the shapes are the point.

```swift
public struct EngineConfiguration: Sendable {
    public var contextWindow: Int = 262_144
    public var kvCachePrecision: KVCachePrecision = .eightBit
    public var expertCacheBudgetBytes: Int?         // nil = the install's own profile
    public var streamingMode: ExpertStreamingMode = .pread(slotCount: 32)
    public var integrityPolicy: ModelIntegrityPolicy = .fullSha256
    public var maxConcurrentGenerations: Int = 1
}

public actor Engine {
    public init(directory: URL, device: MTLDevice, configuration: EngineConfiguration = .init()) async throws
    public var model: ModelDescriptor { get }        // id, family, context, quant slots, byte counts
    public func session(system: String? = nil) -> Session
    public func unload() async
}

public actor Session {
    /// One generation at a time; the engine serialises across sessions.
    public func respond(
        to messages: [ChatMessage],
        options: GenerationOptions = .init(),
        onEvent: @Sendable (GenerationEvent) -> Void
    ) async throws -> GenerationSummary
    public func cancel() async
}

public struct GenerationOptions: Sendable {
    public var maxTokens: Int = 512
    public var temperature: Double = 0.6
    public var topP: Double = 0.95
    public var topK: Int = 20
    public var repetitionPenalty: Double = 1.0
    public var seed: UInt64?                          // nil = system entropy
    public var stop: [String] = []
    public var reasoningEffort: ModelReasoningEffort? // the same control the route uses
}

public enum GenerationEvent: Sendable {
    case promptProcessed(tokens: Int, cachedTokens: Int)
    case token(String)
    case finished(GenerationSummary)
}

public struct GenerationSummary: Sendable {
    public var text: String
    public var promptTokens: Int
    public var completionTokens: Int
    public var timeToFirstToken: Duration
    public var decodeTokensPerSecond: Double
    public var stopReason: StopReason             // .endOfText, .length, .stopString, .cancelled
    public var diagnostics: InferenceStateSnapshot?   // already public today
}

public enum TinyTitanError: Error, Sendable {
    case modelNotFound(URL)
    case notAnInstall(URL)                 // missing/invalid manifest
    case unsupportedFamily(family: String)
    case unsupportedFormat(magic: String)  // names what was found
    case integrityFailure(path: String, expected: String, actual: String)
    case contextWindowExceeded(prompt: Int, window: Int)
    case metalUnavailable(reason: String)
    case cancelled
    case engineShutDown
}
```

Decisions embedded in that sketch:

- **Two levels, engine and session.** The engine is expensive and holds the
  resident weights, the expert cache and the Metal pipelines; a session holds the
  conversation state and the KV cache. An app that talks to one user creates one
  of each; a server creates one engine and a session per request.
- **`onEvent` rather than `AsyncSequence`.** The engine's streaming is
  callback-shaped internally and the server already loops it; an `AsyncStream`
  wrapper is a convenience we can add later without changing the contract.
- **Errors are typed and specific**, because the difference between "not an
  install" and "unsupported family" is the difference between a user fixing a path
  and a user waiting for support — the same distinction `ModelError` already
  draws.
- **Nothing about the harness, the route or the catalog.** Those are the plugin's
  and the server's business.

What the facade still cannot express, implemented honestly rather than papered
over — each of these is a real gap, and closing one is additive:

| Gap | Why |
| --- | --- |
| `EngineConfiguration` has no streaming mode or integrity policy | `ServerModelSession.load` derives both from the install |
| `Engine(device:)` honours only the system default device | the loader builds its own `MetalContext`; a different device is refused with `.metalUnavailable` rather than silently ignored |
| No `promptProcessed` event | `ServerInferenceEvent` has no prompt event to forward |
| Reasoning text and tool calls are dropped | the event/summary types do not carry them, though the orchestrator does |
| `.unsupportedFormat` / `.integrityFailure` are declared but unreachable | `ModelError` is internal to the `TinyTitan` target, so those failures rethrow unclassified rather than being guessed from a string |
| Diagnostics cannot be switched off | they now go to stderr through `ServerLog.diagnostic()` rather than stdout, but an embedder still cannot silence them |
| One generation per session is not enforced | a second `respond` waits on the slot pool; the `.busy` decision is P2 |
| Validation rules are the server's | a `--messages-file` with more than four stop strings, or a `tool` role, is refused where the old CLI rendered it — the facade needs a request vocabulary that is not the OpenAI wire format |

Closed since A1, for the record: the configuration knobs the CLI needed
(`prefillChunkTokens`, `expertCacheSlots`, `ropeScaling`, `thinkingMode`,
`reasoningEffort`, `readAhead`, `forceLogitsHead`), the model's own sampling
defaults, presence penalty, timing on the summary, and a raw-completion entry
point — `Prompt.raw`, which drives the *same* orchestrator with the chat
template skipped rather than a second decode loop.

## 5. Concurrency and lifetime contract (must be written down and tested)

The engine is not thread-safe inside, and pretending otherwise is how a library
earns a reputation. The contract for v1:

1. `Engine` and `Session` are actors: the compiler serialises calls.
2. **One generation per session at a time.** A second `respond` on the same
   session waits (or throws `.busy` if the owner prefers — decide in P2).
3. **One engine, one `MTLDevice`.** The caller passes it; the engine never
   creates a hidden one. Two engines on the same device must work; the test that
   proves it creates two engines on the 4B install and runs them alternately.
4. **Cancellation is cooperative and prompt**: `cancel()` stops the decode loop at
   the next token boundary and the KV/expert state stays usable.
5. **Lifetime**: `unload()` releases the resident buffers; deinit does the same.
   Nothing is left mapped after the last reference goes away.

## 6. P0, corrected: there is no blocker left, only a claim to keep true

The original P0 was "remove the blockers" — move `-warnings-as-errors` and the C
kernel flags out of the manifest so the package could be consumed at all. §2
shows that premise is false on the current toolchain, so the work shrinks to the
part that was always the real acceptance test: a fixture package that is not this
repository, built on every change.

What the fixture proves today (`examples/embedded`, driven by
`tools/embedded-dependency-check.sh`):

| Question | Answer | How it shows |
| --- | --- | --- |
| Does a consumer resolve the package? | yes | the fixture's `swift build` |
| Does the link hold? | yes | the fixture names `Model` and links it |
| Can a consumer use the engine? | partly | `--model …` opens a real install through `Model.load`; it still cannot render a prompt or generate, which is P1 |
| Do the `-O2` kernels travel to the consumer? | yes | a fresh-scratch `swift build -v` in the fixture compiles `TinyTitanKernelsC` with the manifest's flags |

Two things the first draft wanted are therefore **not** needed for a source
consumer and move to the binary stages, where they are wanted for a different
reason: shipping the kernels as a prebuilt binary target (S2's XCFramework, S3's
C ABI) and enforcing the warning set at the script layer (still worth having for
the release gate, but no longer on the critical path to being consumable).

The remaining P0 work is small and belongs in CI: run
`tools/embedded-dependency-check.sh` in the test job, so a manifest change, a
toolchain that reinstates the refusal, or a target added with a new unsafe flag
fails there rather than in somebody's package.

## 7. Packaging ladder

| Stage | Deliverable | Consumer | Notes |
| --- | --- | --- | --- |
| **S1 — source package** | `swift build` against the git tag | Swift programs (SwiftPM), like a static library | Already works mechanically (§2): a consumer resolves, links and opens an install. What S1 still lacks is a *supported* API rather than 1,328 public declarations; no ABI promise, only API stability |
| **S2 — XCFramework** | `TinyTitanLib.xcframework` + `TinyTitanLib_TinyTitanLib.bundle` (Metal resources), built by `release.sh`, attached to the Release with a checksum | Swift programs that cannot or will not build from source | Needs `-enable-library-evolution` + `BUILD_LIBRARY_FOR_DISTRIBUTION` for module stability, and the resource bundle must sit where `Bundle` lookup finds it — for an XCFramework, beside the framework inside the artifact |
| **S3 — C ABI ("the DLL")** | `libtinytitan.dylib` + `tinytitan.h`, exported over an opaque handle | C, C++, Rust, Python (`ctypes`) | `@_cdecl` functions — `tt_engine_create`, `tt_session_respond(callback)`, `tt_session_cancel`, `tt_engine_destroy` — with the C surface owned and versioned by us; Swift's own ABI underneath is the implementation's business |
| **S4 — client of itself** | the server/CLI rebuilt on the library | — | Not a distribution stage but the discipline that keeps the API honest (§3) |

**S1.5 — the raw binaries, decided 2026-10-02.** Every release now carries
`libTinyTitanLib.a` and `libTinyTitanLib.dylib` beside the engine binaries, in a
second archive (`tinytitan-lib-X.Y-macos-arm64.tar.gz`) with its own checksum.
`tools/build_library.sh` builds and asserts it, and `release.sh` stages,
checksums and publishes it, refusing to publish if either artifact is missing
from the archive. Three things that make it what it is rather than a bare `.a`:

- **The static archive is merged.** `swift build` writes one archive per target,
  so `libTinyTitanLib.a` on its own has no runtime, no format reader, no C
  kernels and no tokenizer; the script merges every archive the release build
  produced with `libtool -static`, because that single file is what a static
  consumer links.
- **The dylib's name is its install name.** It is built as a separate dynamic
  product and linked with `-install_name @rpath/libTinyTitanLib.dylib`, so the
  shipped file can carry the plain name; `otool -D` is asserted instead of
  trusted.
- **The Metal bundle travels with it**, because `Bundle.module` resolves beside
  the consumer's executable. That is the §10 trap, and it is the reason the
  script fails closed when `default.metallib` is not in the staged bundle.

What this does **not** yet give a consumer is module stability: a `.swiftmodule`
from this toolchain imports in this toolchain. `-enable-library-evolution` and a
`.swiftinterface` are the XCFramework stage's job (S2), and until then the binary
form is for consumers building with Xcode 27 / Swift 6.4 — which is already the
package's floor.

Versioning: the library follows the release tag. API stability starts at the
first documented release (call it 6.0 or a `1.0` library version — an open
question, §9); before that, `@available` annotations and a `swift-api-digester`
diff in the dry run, so a break is a build failure rather than a surprise.

## 8. Work breakdown

| Phase | Work | Acceptance |
| --- | --- | --- |
| **P0** | Keep consumability proven: `examples/embedded` (done) plus a CI step that runs `tools/embedded-dependency-check.sh` | The fixture resolves, builds and links in CI; the released-tag variant is runnable by hand (`--tag 5.15.0`) |
| **P1** | Introduce `TinyTitanLib`: move the session orchestrator, templating and prompt shaping out of `TinyTitanServerCore`; make the engine NIO-free; rebuild the CLI and server on the facade | Goldens byte-identical through the refactored path (they are the regression net); `TinyTitanServerCore` imports the kit, not the engine internals; a public-surface gate lists what is supported |
| **P2** | Write and test the §5 contract | Tests: one generation per session, two engines on one device, cancel mid-decode, unload while idle |
| **P3** | XCFramework + resource bundle from `release.sh` | The fixture app links the XCFramework (no source) and generates on a clean machine; checksum in the Release; `Bundle` lookup proven by a run |
| **P4** | C ABI + header + a C++ sample | The sample streams tokens; the header compiles as C99 and C++17 |
| **P5** | Docs and process: the wiki's "Embedding TinyTitan" page (user-facing), a library section in `docs/repository-layout.md`, release-note entries, and the versioning policy in `RELEASE.md` | A reader who has never seen the repo can build the fixture from the wiki alone |

Order, corrected: P0 is a guard rather than a prerequisite — it was never what
stood between a consumer and the engine, and §2 is the measurement that says so.
**P1 is the first real work**: turning 1,328 public declarations into an API a
consumer can be promised. P3/P4 are packaging, and P5 is what turns the result
from "possible" into "supported".

Do not start P1 before §9.3 (separate package or a product of this one) and §9.5
(naming) are answered: both decide where the facade lives, and moving it later is
the expensive mistake this plan exists to avoid.

## 9. Open questions for the owner

The first two are cheaper than they looked, because §2 removed the prerequisite
they were weighed against: S1 is no longer "a week of manifest surgery", it is
the facade in P1.

1. **Source first, or binary first?** S1 is a week of work and proves the API;
   S2 is what a closed-source consumer actually needs. Doing S1 alone is the
   cheap experiment.
2. **Is the C ABI in scope for the first library release, or later?** It is the
   literal "DLL" ask, and it is also the most expensive promise to keep.
3. **Separate package or a product of this one?** A separate
   `tinytitan-kit` repository versions cleanly and cannot accidentally expose the
   app's gates; a product of this package keeps one release, one CI, one set of
   goldens. The plan above assumes the product-of-this-package route first.
4. **Does the CPU engine need to be embeddable without Metal?** The dense CPU path
   (`CPUQwen35`) runs without the GPU engine, so a CPU-only consumer is plausible;
   it would need its own facade and its own tests.
5. **Naming.** `TinyTitanLib` vs `TinyTitanEngine` vs reusing `TinyTitan` (which is
   taken by the current target).
6. **What does "supported" mean for the first release?** Which parts of the
   surface (streaming, sampling knobs, diagnostics, the ANE sidecar) carry the
   same warranty the CLI does, and which are explicitly experimental.

## 10. Traps, named now

- **The unsafe-flag claim was stale, and the flag set still matters.** The
  manifest's `.unsafeFlags` did *not* block any consumer measured in §2, so the
  old rule of thumb — "any unsafe flag re-blocks the dependency path" — is not
  what this toolchain does. That is the argument for testing rather than
  reasoning about it: the mechanism belongs to SwiftPM, it has changed before,
  and the fixture in CI turns a future change into a red build instead of a
  user's bug report. Adding a flag is still a decision, because a consumer now
  inherits it as part of the package's build contract.
- **`Bundle.module` under binary distribution.** A copied resource directory
  resolves in a SwiftPM build and *not* inside an XCFramework unless the bundle is
  packaged beside it. S3 must assert a real `Bundle` lookup, not just a link.
- **Sendable holes.** The engine's `nonisolated(unsafe)` sites are load-bearing
  for performance; the facade must not hand a `Model` (or any engine type) to a
  caller. Only actor-isolated `Engine`/`Session` cross the boundary.
- **The GPU is per-process.** A second engine on the same device shares it; the
  memory budget is therefore an app-level decision, which is why
  `EngineConfiguration` exposes the expert-cache budget rather than inventing its
  own.
- **The toolchain floor.** macOS 26 and Swift 6.4 are today's floor; a consumer on
  an older Xcode cannot even resolve the manifest. State it in the wiki page, and
  keep it in step with the release notes.
- **Model licensing stays the user's.** The library reads installs; it must not
  ship, fetch or license weights, and the wiki page has to say so in one sentence.
- **Drift between the app and the library.** If the CLI/server stop exercising a
  facade path, the library rots. The rule "the product is the app" is what keeps
  the surface honest; a library-only code path is a smell to be removed.
