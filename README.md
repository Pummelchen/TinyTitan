<p align="center">
  <img width="1254" height="1254" alt="TinyTitan" src="https://github.com/user-attachments/assets/2a990de5-af24-48cd-b673-e875755741cc" />
</p>

# TinyTitan

[![Stars](.github/badges/stars.svg)](https://github.com/Pummelchen/TinyTitan/stargazers)
[![Views (14d)](.github/badges/views.svg)](https://github.com/Pummelchen/TinyTitan)
[![Last Commit](.github/badges/last-commit.svg)](https://github.com/Pummelchen/TinyTitan/commits/main)

**TinyTitan runs a wide range of open-weight LLMs, mixture-of-experts and
dense, locally on Apple Silicon, streaming the routed experts from SSD so a
model larger than your RAM still runs.** Decode is native Swift over hand-written Metal kernels; the prompt runs
on the GPU or, optionally, the Apple Neural Engine. It ships a CLI, an
installer/repacker, and a loopback OpenAI-compatible server with no MLX and no
GGUF dependency.

**Who it is for.** Every M-series Mac, not only the small ones. With RAM to
spare, TinyTitan keeps more of the model resident and runs as a fast local engine
built to match or beat the established Apple Silicon engines on the models it
supports. Measured head to head against llama.cpp's Metal build, it is
**1.26–1.40× faster** on decode and prefill (Inkling 2 Mini, 24 GB M3, both GGUF
quantizations). On a Mac with less RAM, the same engine streams the routed
experts from SSD, so a model the machine could not hold in memory still runs. It
also serves developers who want a loopback OpenAI- or Anthropic-compatible
endpoint for Codex, Claude Code, Qwen Code, OpenCode, Zed or the official SDKs.
It is a terminal-first tool: the engine and its server are the product. It is
**not** a fine-tuning toolkit, a vision model, a GUI app, or a way to expose a
model to your network — the server is `127.0.0.1`-only and has no
authentication.

## Supported models

Three installs are supported, and nothing else is:

| model | width | engine plugin |
| --- | --- | --- |
| Ornith 1.5 35B-A3B | 4-bit | `Ornith35B4BitPlugin` |
| Inkling 2 Mini 26B-A5B | UD-Q4_K_XL (Unsloth) | `Inkling2GGUFQ4Plugin` |
| Inkling 2 Mini 26B-A5B | Original NVFP4/MXFP8 | `Inkling2NVFP4Plugin` |

The benchmark table below is the same matrix: the three supported installs,
measured on one machine. A family that is not in the table is not a route — it
leaves this matrix when support does, and returns only if someone asks for it in
a GitHub issue. K2, Nemotron, Laguna, the Qwen 3.6/3.8 and AgentWorld/KAT-Coder
fine-tunes, Ornith 1.5 122B and the 8-bit Ornith width have all left; MTP
drafting is out of the supported paths (its code is kept, unused).

## Benchmarks

Peak decode on a base 8-core M3 MacBook Pro with 24 GB, after a warm-up pass.

| Model | Quantization | GPU | CPU | Terminal-Bench 2.x | Terminal-Bench 4.0 |
| --- | --- | ---: | ---: | ---: | ---: |
| Ornith 1.5 35B-A3B | 4-bit | **21.7 tok/s** | NA | 68% (2.1) | NA |
| Inkling 2 Mini 26B-A5B | UD-Q4_K_XL (Unsloth) | **17.9 tok/s** | NA | 72% (2.1) | 5.6% |
| Inkling 2 Mini 26B-A5B | Original (NVFP4/MXFP8) | **8.1 tok/s** | NA | 72% (2.1) | 5.6% |

## Quick start

Requires macOS 26+, Xcode 27 (Swift 6.4) and Apple Silicon.

Last verified: 2026-11-14 — the three commands below were run in a fresh
clone of this commit (`swift build` 940 targets, ~140 s; `--help` exit 0).

```bash
git clone https://github.com/Pummelchen/TinyTitan.git
cd TinyTitan
swift build -c release
.build/release/TinyTitanCLI --help
```

Install a model with `tools/install_models.sh --choose` (the default Ornith
1.5 35B-A3B 4-bit row is a 19.5 GB install), then run a prompt:

```bash
.build/release/TinyTitanCLI --model models/ornith-1.5_35B_A3B_4Bit \
  --prompt "What is the capital of Paris ?" --max-new 96
```

Start the server with `tools/server_launcher.sh`; it prompts for model, RAM
budget, context window (`16K`–`2M`), and KV-cache precision (`8`/`4`/`16`),
prints the base URL, and writes the client configuration. The server runs
standalone TPP/1.1 engine plugins for the three supported installs — Inkling 2
Mini (UD-Q4_K_XL and Original NVFP4/MXFP8) and Ornith 1.5 35B-A3B 4-bit. Any
other install in the catalogue is not a supported route (see **Supported
models** above). The full walkthrough is
[Getting Started](https://github.com/Pummelchen/TinyTitan/wiki/Getting-Started).

## What this repository delivers

1. **The LLM engine as a library — [`TinyTitanLib`](sources/TinyTitanLib).** Embed
   the engine in your own Swift program: an `Engine` and a `Session`, streaming
   tokens through a callback. Same kernels, format reader and sampler as the
   engine, in your process — no subprocess, no HTTP, no model reimplementation.
2. **The complete local LLM engine — `TinyTitanCLI` + `TinyTitanServer`.** Run a
   35B or 26B model from the terminal and serve it on a loopback
   OpenAI-/Anthropic-compatible endpoint for Codex, Claude Code, Qwen Code,
   OpenCode, Zed or the official SDKs. Native Swift and Metal: no MLX, no GGUF
   library.
3. **The `.ssdai` model format and its converter — `TinyTitanRepack` +
   [`tools/`](tools).** Stream a checkpoint into a hash-verified install that
   keeps the routed experts on SSD, so a model larger than your RAM still runs —
   and a truncated, swapped or edited install is refused rather than served.

Those three ship from one branch (`main`) and one release tag. The first two are
the **products** — a library and an engine — and the tooling ships beside them;
the split is spelled out under
[Library and engine](https://github.com/Pummelchen/TinyTitan/wiki/Library-and-Engine).

## Documentation & Credits

Everything is in the [Wiki](https://github.com/Pummelchen/TinyTitan/wiki).
TinyTitan is a focused fork of
[drumih/turbo-fieldfare](https://github.com/drumih/turbo-fieldfare); the full
acknowledgements are in [CREDITS.md](CREDITS.md).

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). Copyright (c) 2026 André Borchert.
