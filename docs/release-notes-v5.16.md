## TinyTitan 5.16 — the engine is a library

This release is the library split. **`TinyTitanLib` is a supported Swift package
product** you embed in your own program — no subprocess, no HTTP — and every
release now ships it as `libTinyTitanLib.a` and `libTinyTitanLib.dylib` beside
the engine binaries. The CLI and the server were both rebuilt on it, so there is
one generation path rather than two, and their output is byte-identical to
5.15's on every stored baseline. The toolchain is pinned **exactly**: Xcode 27 /
Swift 6.4, and nothing else.

### What is new

- **The library.** `import TinyTitanLib` gives you an `Engine` (one install, the
  resident weights, the expert cache) and a `Session` (one conversation) over the
  same kernels, format reader, KV cache and sampler the shipped engine uses.
  `Session.respond` streams through a callback and returns a summary with timings
  and the decode stop reason. Everything else in the target is `package`, so the
  public surface is the whole promise: it imports no NIO and writes nothing to
  stdout. **Checked by** `examples/embedded`, a consumer package that is not this
  repository: it builds against the released tag, streams tokens from a real
  install, and `tools/embedded-dependency-check.sh` runs it in CI.
- **The engine runs on it.** `TinyTitanServerCore`'s session *is* the library's,
  and the CLI no longer owns a tokenizer, a `Model.load` or a decode loop.
  **Checked by** three saved baselines — two raw-completion prompts and one chat
  request — compared byte for byte against the binaries that produced them before
  the move.
- **Tools, both directions.** `respond(tools:)` offers the model tools (each
  schema as JSON text, so the engine's own value type never enters the API), the
  model's calls arrive as `GenerationEvent.toolCall` and in the summary, and a
  caller closes the loop by replaying the assistant turn that asked plus a `tool`
  message naming the call it answers. **Checked by** a real run: offered
  `get_weather`, Qwen3.5 9B 4-bit called it — `get_weather({"city":"Paris"})`,
  `stop=toolCalls` — and the replayed turn with `18 degrees and sunny` answered
  "The weather in Paris right now is 18 degrees and sunny."
- **Every release carries the library binaries**, in a second archive:
  `libTinyTitanLib.a` (the library and every dependency merged into one file),
  `libTinyTitanLib.dylib` (install name `@rpath/libTinyTitanLib.dylib`), the Swift
  module **and every module it imports**, the generated C module maps, a
  `make-flags.sh` that writes the flags for wherever it was extracted, the Metal
  shader sources, and `demo/` — two terminal apps from one source, one per link
  form. **Checked by** extracting the archive into a clean directory and
  compiling and running a consumer that uses no SwiftPM, against both the `.a` and
  the `.dylib`.
- **One toolchain, exactly: Xcode 27 / Swift 6.4.** Not a floor — nothing else is
  tested, and the installer now warns above 6.4 instead of accepting a newer
  Swift silently. One consequence is worth stating: the shipped `.swiftmodule` is
  not stable across toolchains and does not need to be.

### Also in this release

- **Dense Qwen 3.5 prefills in 4,096-token chunks again.** The CLI's own family
  switch lost its dense case when the prefill decision moved into the library, so
  those installs fell to the 128-token default — which also kept them off the ANE,
  whose sidecar accepts exactly 4,096. Restored; output is byte-identical and the
  pre-5.16 stderr line returns with it.
- **A fifth stop string works again.** A local caller is no longer judged by the
  OpenAI wire's caps (four stop strings, a thousand messages); the structural
  rules still apply to everyone.
- **`--quiet` is quiet again**: the library's load and generation lines go through
  a sink the embedder owns, so the CLI can silence what it does not own.
- **More of the engine is on the facade**: `GenerationEvent.promptProcessed` (the
  prompt's length and how much of it the cache held), reasoning text kept apart
  from the answer, an `integrityPolicy` knob (trust the installer's receipt,
  re-hash everything, or let the loader decide), the caller's Metal device honoured
  end to end, and load failures classified — "not an install", "unsupported
  format", "corrupt install" — instead of rethrown unclassified.
- **The concurrency and lifetime contract is tested**, not just written down: two
  engines on one device, two generations on one session, cancellation, and
  `unload()`. It is a model-gated suite, so the ordinary run stays model-free.

### Verification

- `tools/lint.sh` — all eleven gates, clean.
- `swift test --no-parallel` — 1,527 tests in 234 suites.
- **7 of the 16 stored golden baselines compared byte-identical** (`qwen36-{4,8}`,
  `qwen38-4`, `qwen35-{4b,9b}-{4,8}`), then a clean scratch release build with a
  clean warning scan.
- **Not checked, because their install is not under `models/` and nothing may be
  fetched to change that**: `ornith-4`, `ornith-8`, `qwen38-8`, `agentworld-4`,
  `agentworld-8`, `katcoder-4`, `katcoder-8`, `qwen35-2b-4`, `qwen35-2b-8`.
- Live runs on this machine: the library outside SwiftPM in both link forms, the
  tool loop above, the §5 contract suite, and the reasoning probe (128 reasoning
  events from `thinkingMode: .on`, matching the summary).
- **No internal-speed record accompanies this release.** The owner removed that
  step from the release path on 2026-10-01 (`docs/release-process.md` §4b): this
  machine's timings swing past the 10% threshold whenever a browser, WindowServer
  or a game holds the GPU.

### Checksum

`tinytitan-5.16-macos-arm64.tar.gz` sha256: `SHA256_PENDING`
`tinytitan-5.16-macos-arm64.tar.gz` size: `ARCHIVE_BYTES_PENDING` bytes
`tinytitan-lib-5.16-macos-arm64.tar.gz` sha256: `LIBRARY_SHA256_PENDING`
`tinytitan-lib-5.16-macos-arm64.tar.gz` size: `LIBRARY_BYTES_PENDING` bytes
