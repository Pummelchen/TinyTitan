# Embedded demo

A consumer package: it depends on the TinyTitan package and links the engine, the
way another Swift program would. It is the fixture behind the dependency claim in
[the embedded-library plan](../../docs/plan-embedded-library.md), and
`tools/embedded-dependency-check.sh` builds and runs it.

```bash
cd examples/embedded
swift run EmbeddedDemo                                   # proves resolution and the link
swift run EmbeddedDemo --model ../../models/qwen3.6_35B_A3B_4Bit   # opens an install, needs one
```

What it can do today is deliberately small: it links the engine and can open a
`.ssdai` install. It cannot render a prompt or generate a token yet, because that
orchestration lives in `TinyTitanServerCore`, which pulls in NIO — the target
boundary the plan exists to move. When that changes, this file is where the new
surface shows up.

The real dependency line a consumer writes is a released tag, not the relative
path this fixture uses:

```swift
.package(url: "https://github.com/Pummelchen/TinyTitan", from: "5.15.0")
```

The model store is gitignored and no weights ship with this repository, so CI can
only build the fixture; the `--model` path runs on a machine that has an install.
