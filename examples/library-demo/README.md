# TinyTitanLib terminal demo

Two apps, one source file: the same program built against each form of the
library. They are here to show what using `TinyTitanLib` as a **linked binary**
looks like — no SwiftPM, no package manifest, one `swiftc` line.

| App | Built by | Links | What that means |
| --- | --- | --- | --- |
| `tinytitan-demo-static` | `./build-static.sh` | `libTinyTitanLib.a` | The library is copied into your executable: one file to ship, nothing to install beside it. |
| `tinytitan-demo-dynamic` | `./build-dynamic.sh` | `libTinyTitanLib.dylib` | The library stays a separate file, found through `@executable_path` at run time: several apps can share one copy and update it without being rebuilt. |

Everything else — the model loading, the prompt, the streaming callback — is the
same code in both, which is the point: the choice is a link-step decision, not an
API one.

## Build

You need a library distribution. Either extract a release archive
(`tinytitan-lib-<version>-macos-arm64.tar.gz`, attached to every release), or
stage one from the source repository with `tools/build_library.sh <version>`.

```bash
# from a release archive
tar xzf tinytitan-lib-5.16-macos-arm64.tar.gz
cd tinytitan-lib-5.16-macos-arm64/demo
./build-static.sh        # finds the library one directory up
./build-dynamic.sh

# from the source repository, with a local staging
tools/build_library.sh 5.16         # writes .build/library-dist
cd examples/library-demo
./build-static.sh                   # finds it at ../../.build/library-dist
./build-dynamic.sh
```

Both scripts also take an explicit path: `./build-static.sh /path/to/library-dist`.

They copy the resource bundles (and, for the dynamic app, the dylib) next to the
executable. That is not decoration: the runtime resolves its Metal shaders
through `Bundle.module`, which looks beside the *running binary*, so an app that
ships without the bundles starts and then fails on its first model load.

## Run

```bash
./tinytitan-demo-static  --model models/qwen3.5_9B_4Bit
./tinytitan-demo-dynamic --model models/qwen3.5_9B_4Bit
```

Defaults, so the interesting part is one command: the prompt is
`difference swift vs c++ in detail`, 256 tokens, and the sampling is the model's
own row unless you override it. Override any of it:

```bash
./tinytitan-demo-dynamic \
  --model models/qwen3.5_9B_4Bit \
  --prompt "difference swift vs c++ in detail" \
  --max-tokens 512 --temperature 0 --repetition-penalty 1.15
```

One honest note about this model: **qwen3.5 9B 4-bit repeats itself** on a long
list-style answer — "philosophy and philosophy", "control and control" — and the
CLI prints the same text for the same request, so it is the model rather than the
library. The demo therefore defaults to 256 tokens and leaves `repetitionPenalty`
at the engine's 1.0, so what you see is the engine's real behaviour;
`--repetition-penalty 1.15` is noticeably better when you want a longer answer.

Weights are not part of this archive and the library never downloads them: point
`--model` at an install made by the installer or by `tools/install_models.sh` in
the source repository.

The generated answer goes to **stdout** and the framing — what was loaded, the
sampling, the timings — to **stderr**, so the output is composable:

```bash
./tinytitan-demo-static --model models/qwen3.5_9B_4Bit > answer.txt
```

## What the code does

`main.swift` is the whole program, and it is deliberately only the library's
public API:

```swift
let engine = try await Engine(directory: URL(fileURLWithPath: modelDirectory),
                              device: MTLCreateSystemDefaultDevice()!)
let session = await engine.session()

let summary = try await session.respond(
    to: [ChatMessage(role: .user, content: prompt)],
    options: GenerationOptions(maxTokens: maxTokens, temperature: temperature)
) { event in
    if case .token(let text) = event { print(text, terminator: "") }
}
print(summary.completionTokens, summary.decodeStopReason)
```

- **`Engine`** owns one install, the resident weights, the expert cache and the
  Metal pipelines. It is expensive; an app creates one and keeps it.
- **`Session`** owns one conversation. It is cheap; a server creates one per
  request.
- **`respond(to:options:onEvent:)`** streams tokens through the callback and
  returns a `GenerationSummary` with the timings and the decode stop reason.
- **`Prompt.raw("…")`** instead of `[ChatMessage]` gets you a raw completion with
  no chat template applied — the demo uses the chat form, which is what a
  question like this wants.

The app prints which form it was built with, so the two binaries are never
confused for one another at the terminal.

## Requirements

Apple Silicon, macOS 26 or later, and a `.ssdai` install. The released binaries
are built by one Xcode (27 / Swift 6.4) and their module is not stable across
toolchains yet — building the package as a SwiftPM dependency remains the
supported route when you need to compile the library yourself. See the wiki's
[Library and Engine](https://github.com/Pummelchen/TinyTitan/wiki/Library-and-Engine)
page and the plan in `docs/plan-embedded-library.md`.
