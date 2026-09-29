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

## Quickstart

**Prerequisites:** Apple Silicon (arm64, M1 or newer) and macOS 26 or later.
The one-command install needs nothing else — no Xcode, Homebrew, git, Node or
Python. Building from source additionally needs Swift 6.4+ (Xcode 27 or a
matching toolchain). Model storage is the real cost: about 19.5 GB for a 35B
4-bit install, about 162 GB for Qwen3.8-Flash-Next 4-bit.

**Nothing installed yet?** One command checks the Mac, downloads the published
`arm64` engine, offers to download a model, and leaves a `tinytitan` command
that starts the server:

```bash
bash -c "$(curl -fsSL https://raw.githubusercontent.com/Pummelchen/TinyTitan/main/tools/install_tinytitan.sh)"
```

Use that form, **not** `curl … | bash`: a pipe makes the script's stdin the
pipe, so it cannot ask anything and takes the default at every step — including
the model download. From a clone or an unzipped download,
`bash tools/install_tinytitan.sh` does the same and never needs `chmod +x`;
`--help` lists the flags. Everything it creates is under `~/.tinytitan` and
`~/.local/bin`.

**Already have a build and an installed model?** These three commands are the
whole loop:

```bash
tools/install_models.sh                                         # what is installed
.build/release/TinyTitanCLI --model models/qwen3.5_4B_4Bit \
  --prompt "The capital of France is" --max-new 32 --temperature 0
tools/server_launcher.sh --client server --model qwen35-4b --bits 4   # the API
```

Generated text goes to **stdout** and the timing footer to **stderr**, so a
pipeline sees only the answer. The model directory is whatever
`tools/install_models.sh` reports as `installed`.

*Last verified: 2026-09-28 on macOS 27.0 / Swift 6.4 / arm64 (Apple M3, 24 GB)
by running `tools/install_models.sh` and `--help`, the CLI command above with
`models/qwen3.5_4B_4Bit` (and `TinyTitanCLI/--help`), `swift run -c release
TinyTitanRepack --verify-install --input-gturbo models/qwen3.5_4B_4Bit`
("Verified 7 files"), `tools/server_launcher.sh --help` and `--client server
--model qwen35-4b --bits 4 --port 8084`, `TinyTitanServer --help`, `tools/dsh_route.sh`
(the print form), `memory_pressure -Q`, the `pgrep` process check,
`tools/install_tinytitan.sh --help`, `swift build -c release` in a clean clone,
and the server workflow in [Worked example](#worked-example). Commands this page
marks with a prerequisite were not executed here: the *full* one-command
installer calls the GitHub releases API and writes to `~/.tinytitan` and
`~/.local/bin`, a model download is tens of gigabytes, and `--client
codex|zed|…` rewrites that client's own provider config.*

## Install

### One command (recommended)

The installer in the Quickstart is the supported path. It verifies the
published checksum, is safe to re-run, and never deletes a model. `--web` also
sets up the optional browser chat window; `--from-source` builds the checkout
instead of downloading a release; `--model NAME` picks a model unattended. A
second run updates what is already installed instead of fetching it twice.

### From source

```bash
git clone https://github.com/Pummelchen/TinyTitan.git
cd TinyTitan
swift build -c release
```

`swift build -c release` needs Swift 6.4+ and an arm64 macOS 26+ machine. It
builds every product; add `--product TinyTitanServer` to build only the server
or `--product TinyTitanCLI` to build only the CLI.

### Install a model

`tools/install_models.sh` knows the whole catalogue, including the models that
are converted from a checkpoint rather than streamed:

```bash
tools/install_models.sh                     # what is installed, what is missing
tools/install_models.sh qwen35-4b           # dense Qwen 3.5 4B, 4-bit
tools/install_models.sh katcoder both       # both widths, ONE download
tools/install_models.sh --help              # sources, disk sizes, env vars
```

Every model installs at **4-bit and 8-bit**:

- **Qwen3.8-Flash-Next 125B-A6B**
- **KAT-Coder-V2.5-Dev 35B-A3B**
- **Qwen-AgentWorld 35B-A3B**
- **Ornith 1.5 35B-A3B**
- **Qwen 3.6 35B-A3B**
- **Qwen 3.5 9B / 4B / 2B** (dense, GPU or CPU)

Use `both` for two widths of the same model: the Qwen3.5-MoE checkpoints
convert both widths from one ~70 GB fetch, so installing one width at a time
fetches the checkpoint twice. See
[Getting Started](https://github.com/Pummelchen/TinyTitan/wiki/Getting-Started#3-install-a-model)
for the per-model sizes, the interrupted-download resume, and the optional
experimental MTP sidecar.

## Configuration

Most settings have a command-line flag; these environment variables are read by
the launcher and the tools.

| Variable | Effect |
| --- | --- |
| `TINYTITAN_PORT` | Port the launcher serves on (default `8080`). `--port` does the same. |
| `TINYTITAN_MEMORY` | `1` turns on persistent agent memory. Off by default. |
| `TINYTITAN_MEMORY_DIR` | Where the memory store lives (default `~/.tinytitan/memory`). |
| `TINYTITAN_WORK_DIR` | Where a download and its conversion are staged. Point it at another volume when the model volume is tight. |
| `TINYTITAN_BIN_DIR` | Where the tools look for the executables (a checkout uses `.build/release`). |
| `TINYTITAN_MODELS_DIR` | Where installs live (default `models/`). |
| `TINYTITAN_PREFILL_ANE` | `on` runs full-attention prefill blocks on the Neural Engine. |
| `HF_ENDPOINT` | Point a model download at a Hugging Face mirror. |
| `HF_TOKEN` | Only when Hugging Face explicitly asks for authentication. |

The server itself takes the runtime controls: `--ram-budget` (resident-memory
target, e.g. `8G`), `--max-context`, `--rope-scaling yarn`, `--kv-bits 4|8|16`,
`--prompt-cache-mode`, `--thinking on|off`. `--ram-budget` covers the whole
server process, not just the expert cache: the runtime subtracts the weight file
and a ~0.5 GB reserve, then steps down the expert-cache slot ladder to the
largest cache that fits. Full flag reference:
[Runtime Controls](https://github.com/Pummelchen/TinyTitan/wiki/Runtime-Controls).

## Worked example

This is a complete round trip — start the server, check it, ask for an answer,
and get structured output — run exactly as written. It uses the dense Qwen 3.5
4B install; substitute any directory that `tools/install_models.sh` reports as
`installed`.

**1. Start the server** (keep this terminal open):

```bash
.build/release/TinyTitanServer --model models/qwen3.5_4B_4Bit --port 8083
```

**2. From a second terminal, check it and read the model id**. Take the id from
the response rather than typing it:

```bash
curl --silent http://127.0.0.1:8083/health
curl --silent http://127.0.0.1:8083/v1/models
```

```text
{"status":"ok"}
{"object":"list","data":[{"id":"qwen3.5-4b_4-Bit", ...}, {"id":"qwen3.5-4b_4-Bit-fast", ...}]}
```

**3. Ask for an answer**:

```bash
curl --silent http://127.0.0.1:8083/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.5-4b_4-Bit",
    "messages": [{"role": "user", "content": "Reply with exactly READY."}],
    "temperature": 0,
    "max_completion_tokens": 16
  }'
```

```json
{
  "usage": {"prompt_tokens": 17, "completion_tokens": 2, "total_tokens": 19},
  "model": "qwen3.5-4b_4-Bit",
  "choices": [{"message": {"role": "assistant", "content": "READY"},
               "finish_reason": "stop", "index": 0}],
  "object": "chat.completion"
}
```

**4. Constrain the output to JSON**. The server compiles the schema into a
grammar that masks the sampler, so the bytes that come back are a well-formed
document of the requested shape:

```bash
curl --silent http://127.0.0.1:8083/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen3.5-4b_4-Bit",
       "messages": [{"role": "user", "content": "The colour red, as JSON."}],
       "response_format": {"type": "json_object"},
       "max_completion_tokens": 32}'
```

```json
{"choices":[{"message":{"content":"{\n\"The colour\": \"red\"\n}"}}]}
```

It constrains the **shape**, never the truth of the content. The supported
schema subset is small and explicit, and anything outside it is refused by
name; see [docs/structured-output.md](docs/structured-output.md).

**5. Stop the server** with `Ctrl-C`, or `pgrep -fl TinyTitanServer` to find
it. Run one model process at a time.

## Common workflows

- **A coding client against the local server.** `tools/server_launcher.sh`
  starts the API on its own or starts it and wires one client — Codex, Claude
  Code, Qwen Code, OpenCode or Zed — to the model the server advertises. It
  asks for the model, the width, the thinking level and an optional RAM target;
  `--client server` is API only.
  ```bash
  tools/server_launcher.sh                                       # interactive
  tools/server_launcher.sh --client codex --model ornith --bits 4  # server + Codex
  tools/server_launcher.sh --client zed --model qwen38 --bits 4 --ram 8
  ```
  The `--client` forms rewrite that client's own provider config, so run them
  only when you want that change; `--client server` touches no client. It
  serves on `127.0.0.1:8080` by default, and every other installed model
  stays available by name through the API, one resident at a time. See
  [Connect a client](https://github.com/Pummelchen/TinyTitan/wiki/OpenAI-Compatible-Server#connect-a-client).
- **A chat window, if you want one.** The installer's `--web`, or
  `tools/server_launcher.sh --web`, sets up TinyTitan's own pinned DeepSeek
  Harness under `~/.tinytitan` — a local browser page already pointed at the
  model, kept isolated from any DeepSeek Harness you run yourself. It is a
  client of the loopback server, not a bundled app; see
  [tools/dsh_local.sh](tools/dsh_local.sh).
- **The CLI directly.** `.build/release/TinyTitanCLI --help` lists generation,
  streaming and function-tool flags; `--quiet` drops the timing footer.
- **Persistent agent memory.** `TINYTITAN_MEMORY=1` gives the model memory that
  outlives a conversation, scoped per repository, with six memory tools the
  engine answers itself. It runs inside the server process, so there is no
  database to install; see [docs/agent-memory.md](docs/agent-memory.md).
- **Three client protocols on one server.** OpenAI Chat Completions, the OpenAI
  Responses API (stored responses, `previous_response_id`, the full event
  grammar) and the Anthropic Messages API (`/v1/messages`, `count_tokens`,
  streaming); see [docs/server-api.md](docs/server-api.md).
- **Long context.** Native RoPE supports up to 262,144 tokens; `--rope-scaling
  yarn` extends it to 512K or 1M, at a memory cost you can cap with
  `--kv-bits 4`.
- **The CPU engine.** The dense Qwen 3.5 installs run on either engine —
  `--cpu`, or the `@cpu` id the catalogue server lists.
- **A second machine on the LAN.** `ttlanmanager` plus the
  `dsh-lan-manager` plugin manage DeepSeek Harness workspaces across Macs; see
  [LAN Manager](https://github.com/Pummelchen/TinyTitan/wiki/LAN-Manager).

## Troubleshooting

The wiki [FAQ](https://github.com/Pummelchen/TinyTitan/wiki/FAQ) is the full
list. The ones that catch people first:

- **`error: model directory not found: <path>`** — `--model` named a path that is
  not a directory. Check the spelling against `tools/install_models.sh`, which
  prints the installed directory names.
- **`error: installed tokenizer is missing chat_template.jinja`** — the directory
  exists but is not a complete install; reinstall that model.
- **The model stopped loading after it moved.** The verified receipt is bound to
  the original absolute path; reissue it in place with
  `swift run -c release TinyTitanRepack --verify-install --input-gturbo <path>`.
  Never hand-edit `verified-install.json`.
- **The first token is slow.** The prompt must be prefilled before decode
  begins, and coding clients send thousands of tokens of instructions. Use
  prompt-state reuse for continued conversations, or the chat-only `-fast`
  alias for direct questions.
- **Two model processes at once.** Don't. `pgrep -fl 'TinyTitanServer|TinyTitanCLI'`
  should print nothing before you start a run.

A model run that fails for any other reason reports the command, hardware, RAM,
macOS and Swift version alongside the error; include the whole message when you
ask for help.

## Getting help

- **Bugs and feature requests:** [GitHub Issues](https://github.com/Pummelchen/TinyTitan/issues).
  The issue templates ask for the exact command, its output and your hardware.
- **Questions and how-to:** the [wiki](https://github.com/Pummelchen/TinyTitan/wiki)
  — [Getting Started](https://github.com/Pummelchen/TinyTitan/wiki/Getting-Started),
  the [Cookbook](https://github.com/Pummelchen/TinyTitan/wiki/Cookbook) and the
  [FAQ](https://github.com/Pummelchen/TinyTitan/wiki/FAQ).
- **Security:** do not open a public issue; see [SECURITY.md](SECURITY.md).
- **Contributing:** [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md).

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

## What it does

- **Bounded expert RAM.** The resident expert cache is sized per family from the
  model's own expert stride and clamped to **a third of physical memory**, so a
  smaller Mac is not handed a budget tuned on a larger one. The 125B model runs
  on 24 GiB because only a bounded slice of its experts is resident.
- **No MLX, no GGUF.** TinyTitan ships its own high-speed model format and a
  converter that builds it straight from the original weights.
- **Apple Neural Engine prefill.** `TINYTITAN_PREFILL_ANE=on` runs
  full-attention prefill blocks on the Neural Engine from a one-time exported
  Core ML sidecar, roughly halving long-prompt time to first token; short
  prompts and decode are untouched.
- **Compressed KV cache.** Live attention state can use 16-bit, 8-bit or 4-bit
  storage independently of the installed model quantization.
- **Thinking mode.** Ornith and Qwen support truthful Off/On reasoning control;
  their chat templates do not define Low/Medium/High effort levels.
- **Enforced structured output.** Chat Completions `response_format`, Responses
  `text.format` and Messages `output_config.format` compile to a byte-level
  grammar that masks the sampler on both engines.
- **MTP off by default.** Native speculative decoding remains experimental and
  disabled because measured Ornith runs showed no speed benefit; it currently
  requires greedy decoding, native RoPE, and prompt-cache reuse off.
- **Tiled Top-K sampling.** Production sampling (Top-K 1–64) runs a three-stage
  tiled GPU reduction, cutting per-token sampling cost from 15.5 ms to 1.4 ms
  with a token-for-token identical stream.
- **Follow-up cache.** Exact live and multi-prefix prompt-state reuse avoids
  repeating compatible prefill work across conversation turns.
- **Tested coding CLIs.** Codex, Claude Code, Qwen Code, OpenCode and the Zed
  editor are wired by the launcher; `--round clients` on the coder benchmark
  checks every client's wiring without loading a model.

## Documentation

- [Getting started](https://github.com/Pummelchen/TinyTitan/wiki/Getting-Started)
- [Cookbook — one recipe per task](https://github.com/Pummelchen/TinyTitan/wiki/Cookbook)
- [Features](https://github.com/Pummelchen/TinyTitan/wiki/Features)
- [Local server and launchers](https://github.com/Pummelchen/TinyTitan/wiki/OpenAI-Compatible-Server)
- [Runtime controls](https://github.com/Pummelchen/TinyTitan/wiki/Runtime-Controls)
- [FAQ](https://github.com/Pummelchen/TinyTitan/wiki/FAQ)
- [Changelog](https://github.com/Pummelchen/TinyTitan/wiki/Changelog)
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
