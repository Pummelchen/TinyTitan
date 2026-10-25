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
engine and its server are the product, and the DeepSeek Harness plugins are an
optional client of that server. It is **not** a fine-tuning toolkit, a vision model, a
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
4. **The `.ssdai` model format and its converter — `TinyTitanRepack` +
   [`tools/`](tools).** Stream a checkpoint into a hash-verified install that
   keeps the routed experts on SSD, so a model larger than your RAM still runs —
   and a truncated, swapped or edited install is refused rather than served.

Those four ship from one branch (`main`) and one release tag. The first two are
the **products** — a library and an engine — and the rest are the plugin and
tooling that ship beside them; the split is spelled out under
[Library and engine](https://github.com/Pummelchen/TinyTitan/wiki/Library-and-Engine).

## Quick start

Requires macOS 26+, Xcode 27 (Swift 6.4) and Apple Silicon.

```bash
git clone https://github.com/Pummelchen/TinyTitan.git
cd TinyTitan
swift build -c release
.build/release/TinyTitanCLI --help
```

Install a model with `tools/install_models.sh --choose` (the dense 2B row is
about 2.3 GB), then run a prompt:

```bash
.build/release/TinyTitanCLI --model models/qwen3.5_2B_4Bit \
  --prompt "Explain a mutex in one paragraph." --max-new 96
```

Start the server with `tools/server_launcher.sh`; it prints the base URL and
writes the client configuration. The full walkthrough is
[Getting Started](https://github.com/Pummelchen/TinyTitan/wiki/Getting-Started).

Last verified: 2026-10-23 — release build and both `--help` commands in a fresh
clone; the 2B install and a greedy 96-token CLI run on this host.

## Benchmarks

Peak decode on a base 8-core M3 MacBook Pro with 24 GB. `NA` means that engine
does not serve the model: the MoE families stream their experts on the GPU + ANE
path, only the dense Qwen 3.5 models run on either engine, and K2-Horizon is
served from its GPU probe by the server (single model, no prompt cache or
batching) rather than by the runner. Its rows below are `TinyTitanBench k2gpu`
with 32 greedy decode tokens and a warm expert cache: **12.4 tok/s at 4-bit and
2.0 tok/s at 8-bit**. The gap is memory capacity, not weight width: the 4-bit
install is 21 GB and fits this host's 24 GB, so its 64-slot-per-layer expert
cache holds most of each layer's 100 experts; the 8-bit install is 39 GB
(27.6 GB of experts), only 16 slots per layer fit, and the other seven of each
token's eight picks stream from disk. The served path is lower — it adds the
request loop, detokenising and sampling to the same probe. K2-Horizon's score
is from its
[model card](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B); no
Terminal-Bench 4.0 result is published for it. Per-version measurements and the
method live on the wiki
([Benchmarks](https://github.com/Pummelchen/TinyTitan/wiki/Benchmarks) ·
[Benchmarking Guide](https://github.com/Pummelchen/TinyTitan/wiki/Benchmarking-Guide)).

| Model | Quantization | GPU | CPU | Terminal-Bench 2.x | Terminal-Bench 4.0 |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.8-Flash-Next 125B-A6B | 4-bit | **5.46 tok/s** | NA | 86.14% (2.1) | 25.25% |
| Qwen3.8-Flash-Next 125B-A6B | 8-bit | **2.10 tok/s** | NA | 86.14% (2.1) | 25.25% |
| Ornith 1.5 35B-A3B | 4-bit | **21.65 tok/s** | NA | 67.8% (2.1) | NA |
| Ornith 1.5 35B-A3B | 8-bit | **11.93 tok/s** | NA | 67.8% (2.1) | NA |
| K2-Horizon-MoVA-36B-A4B | 4-bit | **12.4 tok/s** | NA | 58.6% (2.1) | NA |
| K2-Horizon-MoVA-36B-A4B | 8-bit | **2.0 tok/s** | NA | 58.6% (2.1) | NA |
| Qwen 3.6 35B-A3B | 4-bit | **21.41 tok/s** | NA | 52.5% (2.1) | NA |
| Qwen 3.6 35B-A3B | 8-bit | **12.37 tok/s** | NA | 52.5% (2.1) | NA |
| Qwen-AgentWorld 35B-A3B | 4-bit | **21.74 tok/s** | NA | 39.55% (2.0) | NA |
| Qwen-AgentWorld 35B-A3B | 8-bit | **12.28 tok/s** | NA | 39.55% (2.0) | NA |
| KAT-Coder-V2.5-Dev 35B-A3B | 4-bit | **17.86 tok/s** | NA | 32.6% (2.1) | NA |
| KAT-Coder-V2.5-Dev 35B-A3B | 8-bit | **6.91 tok/s** | NA | 32.6% (2.1) | NA |
| Qwen 3.5 4B (dense) | 4-bit | **22.81 tok/s** | **7.54 tok/s** | 25.8% (2.1) | NA |
| Qwen 3.5 4B (dense) | 8-bit | **14.10 tok/s** | **7.85 tok/s** | 25.8% (2.1) | NA |
| Qwen 3.5 9B (dense) | 4-bit | **12.88 tok/s** | **4.43 tok/s** | 9.2% (2.0) | NA |
| Qwen 3.5 9B (dense) | 8-bit | **7.80 tok/s** | **4.42 tok/s** | 9.2% (2.0) | NA |
| Qwen 3.5 2B (dense) | 4-bit | **44.41 tok/s** | **17.42 tok/s** | 3.0% (2.1) | NA |
| Qwen 3.5 2B (dense) | 8-bit | **29.64 tok/s** | **18.10 tok/s** | 3.0% (2.1) | NA |

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

**Project Maya:** the 6.0 checkpoint, calibration and machine-size cache work was
carried under that name. The live ledger is
[docs/maya-task-status.md](docs/maya-task-status.md); the closeout is
[docs/maya-closeout-2026-10-22.md](docs/maya-closeout-2026-10-22.md).

**Kapitsa:** the cross-session shared system-prefix cache and the M4+ tensor-ops
prefill direction were informed by [nace-ai/kapitsa](https://github.com/nace-ai/kapitsa).
No code is copied from it.

**Related research:** [TinyTitan Datacenter](https://github.com/Pummelchen/TinyTitan_Datacenter)
runs large MoE models across a cluster of Mac minis and Studios, keeping them on
SSD/NVMe for near-linear scale of decode throughput.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). Copyright (c) 2026 André Borchert.
