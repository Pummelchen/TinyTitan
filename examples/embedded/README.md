# Embedded demo

A consumer package: it depends on the **library product** `TinyTitanLib` and
embeds the engine in its own process — no subprocess, no HTTP. This is the
fixture behind [`docs/plan-embedded-library.md`](../../docs/plan-embedded-library.md),
and `tools/embedded-dependency-check.sh` builds and runs it.

```bash
cd examples/embedded
swift run EmbeddedDemo                                   # resolves and links the library
swift run EmbeddedDemo --model ../../models/qwen3.5_4B_4Bit \
  --prompt "The capital of France is" --max-tokens 12     # streams tokens
```

The whole API it uses is four types:

```swift
import TinyTitanLib

let engine = try await Engine(directory: installURL, device: device)
let session = await engine.session()
let summary = try await session.respond(
    to: [ChatMessage(role: .user, content: "The capital of France is")],
    options: GenerationOptions(maxTokens: 32, temperature: 0)
) { event in
    if case .token(let text) = event { print(text, terminator: "") }
}
print(summary.stopReason, summary.completionTokens)
```

The real dependency line a consumer writes is a released tag, not the relative
path this fixture uses:

```swift
.package(url: "https://github.com/Pummelchen/TinyTitan", from: "5.15.0")
```

The model store is gitignored and no weights ship with this repository, so CI can
only build the fixture; the `--model` path runs on a machine that has an install.
Two things the facade does not offer yet, both recorded in the plan: the loader
runs on the system default Metal device, and the orchestrator still writes its
telemetry to stdout, which an embedder cannot switch off.
