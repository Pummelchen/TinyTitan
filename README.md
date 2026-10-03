<p align="center">
  <img width="1254" height="1254" alt="TinyTitan" src="https://github.com/user-attachments/assets/2a990de5-af24-48cd-b673-e875755741cc" />
</p>

# TinyTitan

[![Stars](https://img.shields.io/github/stars/Pummelchen/TinyTitan?style=flat-square&logo=github&label=Stars&color=e3b341)](https://github.com/Pummelchen/TinyTitan/stargazers)
[![Views (14d)](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Pummelchen/TinyTitan/main/.github/traffic.json)](https://github.com/Pummelchen/TinyTitan)
[![Last Commit](https://img.shields.io/github/last-commit/Pummelchen/TinyTitan?style=flat-square&logo=git&label=Last%20Commit&color=2ea44f)](https://github.com/Pummelchen/TinyTitan/commits/main)

**TinyTitan runs Qwen-family MoE and dense text models locally on Apple Silicon,
streaming the routed experts from SSD so a model larger than your RAM still
runs.** Decode is native Swift over hand-written Metal kernels; the prompt runs
on the GPU or, optionally, the Apple Neural Engine. It ships a CLI, an
installer/repacker, and a loopback OpenAI-compatible server with no MLX and no
GGUF dependency.

**Who it is for.** People on an M-series Mac who want to run a 35B — or a 125B —
model locally on a machine that could never hold it in RAM, and developers who
want a loopback OpenAI- or Anthropic-compatible endpoint for Codex, Claude Code,
Qwen Code, OpenCode, Zed or the official SDKs. It is a terminal-first tool: the
engine and its server are the product, and a browser chat window is an optional
client of that server. It is **not** a fine-tuning toolkit, a vision model, a
GUI app, or a way to expose a model to your network — the server is
`127.0.0.1`-only and has no authentication.

## What this repository delivers

1. **The LLM engine as a library — [`TinyTitanLib`](sources/TinyTitanLib).** Embed
   the engine in your own Swift program: an `Engine` and a `Session`, streaming
   tokens through a callback. Same kernels, format reader and sampler as the
   engine, in your process — no subprocess, no HTTP, no model reimplementation.
2. **The complete local LLM engine — `TinyTitanCLI` + `TinyTitanServer`.** Run a
   35B or 125B model from the terminal and serve it on a loopback
   OpenAI-/Anthropic-compatible endpoint for Codex, Claude Code, Qwen Code,
   OpenCode, Zed or the official SDKs. Native Swift and Metal: no MLX, no GGUF.
3. **DeepSeek Harness plugins — [`plugins/dsh-tinytitan`](plugins/dsh-tinytitan),
   [`plugins/dsh-lan-manager`](plugins/dsh-lan-manager).** The harness finds the
   models you have installed and keeps its route current, with a quiet
   compaction backend; the LAN manager drives a fleet of harnesses from one
   console.
4. **A ready-made DSH bundle install — [`tools/dsh_local.sh`](tools/dsh_local.sh).**
   A pinned harness with the TinyTitan bundle already wired, entirely under
   `~/.tinytitan`: it never touches a `dsh`, a `~/.dsh` or a port you already use.
5. **The `.ssdai` model format and its converter — `TinyTitanRepack` +
   [`tools/`](tools).** Stream a checkpoint into a hash-verified install that
   keeps the routed experts on SSD, so a model larger than your RAM still runs —
   and a truncated, swapped or edited install is refused rather than served.

Those five ship from one branch (`main`) and one release tag. The first two are
the **products** — a library and an engine — and the rest are the bundles and
tooling that ship beside them; the split is spelled out under
[Library and engine](https://github.com/Pummelchen/TinyTitan/wiki/Library-and-Engine).

## Benchmarks

Peak decode on a base 8-core M3 MacBook Pro with 24 GB. `NA` means the CPU
engine does not serve that model: the MoE families stream their experts on the
GPU + ANE path, and only the dense Qwen 3.5 models run on either engine.
Per-version measurements and the method live on the wiki
([Benchmarks](https://github.com/Pummelchen/TinyTitan/wiki/Benchmarks) ·
[Benchmarking Guide](https://github.com/Pummelchen/TinyTitan/wiki/Benchmarking-Guide)).

| Model | Quantization | GPU | CPU |
| --- | --- | ---: | ---: |
| Qwen 3.5 2B (dense) | 4-bit | **53.73 tok/s** | **15.42 tok/s** |
| Qwen 3.5 2B (dense) | 8-bit | **32.77 tok/s** | **15.83 tok/s** |
| Qwen 3.5 4B (dense) | 4-bit | **26.18 tok/s** | **7.71 tok/s** |
| Qwen-AgentWorld 35B-A3B | 4-bit | **21.74 tok/s** | NA |
| Ornith 1.5 35B-A3B | 4-bit | **21.65 tok/s** | NA |
| Qwen 3.6 35B-A3B | 4-bit | **21.41 tok/s** | NA |
| KAT-Coder-V2.5-Dev 35B-A3B | 4-bit | **17.86 tok/s** | NA |
| Qwen 3.5 4B (dense) | 8-bit | **16.14 tok/s** | **7.04 tok/s** |
| Qwen 3.5 9B (dense) | 4-bit | **14.93 tok/s** | **4.07 tok/s** |
| Qwen 3.6 35B-A3B | 8-bit | **12.37 tok/s** | NA |
| Qwen-AgentWorld 35B-A3B | 8-bit | **12.28 tok/s** | NA |
| Ornith 1.5 35B-A3B | 8-bit | **11.93 tok/s** | NA |
| Qwen 3.5 9B (dense) | 8-bit | **8.90 tok/s** | **4.51 tok/s** |
| KAT-Coder-V2.5-Dev 35B-A3B | 8-bit | **6.91 tok/s** | NA |
| Qwen3.8-Flash-Next 125B-A6B | 4-bit | **5.46 tok/s** | NA |
| Qwen3.8-Flash-Next 125B-A6B | 8-bit | **2.10 tok/s** | NA |

## Documentation

- [Getting started](https://github.com/Pummelchen/TinyTitan/wiki/Getting-Started)
- [Installation and configuration](https://github.com/Pummelchen/TinyTitan/wiki/Installation-and-Configuration)
  — the model catalogue, environment variables, the launcher and the tools
- [Cookbook — one recipe per task](https://github.com/Pummelchen/TinyTitan/wiki/Cookbook)
- [Local server and API](https://github.com/Pummelchen/TinyTitan/wiki/OpenAI-Compatible-Server)
- [Runtime controls](https://github.com/Pummelchen/TinyTitan/wiki/Runtime-Controls) — every flag and default
- [Features](https://github.com/Pummelchen/TinyTitan/wiki/Features)
- [Library and engine](https://github.com/Pummelchen/TinyTitan/wiki/Library-and-Engine)
  — the library and the engine, and how the two products are split
- [System design](https://github.com/Pummelchen/TinyTitan/wiki/System-Design)
- [FAQ](https://github.com/Pummelchen/TinyTitan/wiki/FAQ)
- [Benchmarks](https://github.com/Pummelchen/TinyTitan/wiki/Benchmarks) ·
  [Benchmarking guide](https://github.com/Pummelchen/TinyTitan/wiki/Benchmarking-Guide)
- [Changelog](https://github.com/Pummelchen/TinyTitan/wiki/Changelog)
- [Project tracker](https://github.com/Pummelchen/TinyTitan/wiki/Project-Tracker) — open work only
- [Repository layout](docs/repository-layout.md) — where everything lives, and
  the naming and file-size conventions

## Credits

TinyTitan is a focused fork of
[drumih/turbo-fieldfare](https://github.com/drumih/turbo-fieldfare), whose
bounded-memory runtime, installer, CLI and local server this project builds on.
The Qwen 3.6 integration was created by
[NeelM0906](https://github.com/NeelM0906) in
[upstream PR #29](https://github.com/drumih/turbo-fieldfare/pull/29). Concise
mode is derived from the
[Nail-Qwen3.6-35B-A3B](https://huggingface.co/peculiar-ragdoll/Nail-Qwen3.6-35B-A3B-MLX)
chat template by [peculiar-ragdoll](https://huggingface.co/peculiar-ragdoll).

**Related research:** [TinyTitan Datacenter](https://github.com/Pummelchen/TinyTitan_Datacenter)
runs large MoE models across a cluster of Mac minis and Studios, keeping them on
SSD/NVMe for near-linear scale of decode throughput.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). Copyright (c) 2026 André Borchert.
