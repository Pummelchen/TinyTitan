# Audit scope and risk tiers — 2026-10-23

Per §2. Committing this inventory before fixing anything. Head: `4816a84`
(branch `audit/2026-10-23`).

## 2.1 Projects, languages, build systems, entry points

One SwiftPM package (`Package.swift`, swift-tools-version 6.4, platform macOS 26)
with two products (`TinyTitanLib` library, engine executables), plus two npm
plugin packages and a Python benchmark/converter tree.

| Project | Language | Build | Entry points |
| --- | --- | --- | --- |
| `sources/TinyTitanFormat` | Swift 6 | SwiftPM | `.ssdai` format types (library-internal) |
| `sources/TinyTitanKernelsC` | C99 | SwiftPM (clang, `-O2 -pedantic-errors -Werror …`) | `int4_affine_gemv`, `int8_affine_gemv`, `expert_io` |
| `sources/TinyTitan` | Swift 6 | SwiftPM | engine runtime: decode loop, Metal kernels, expert streaming, sampler |
| `sources/TinyTitanLib` | Swift 6 | SwiftPM | **public facade**: `Engine`, `Session`, `GenerationOptions`… |
| `sources/TinyTitanRepack` (Core + Command) | Swift 6 | SwiftPM | `TinyTitanRepack` executable (installer/verifier) |
| `sources/TinyTitanCLI` (+Command) | Swift 6 | SwiftPM | `TinyTitanCLI`, `tinytitan calibrate` |
| `sources/TinyTitanServer` (Core + Command) | Swift 6 | SwiftPM | `TinyTitanServer` — loopback OpenAI/Anthropic-compatible HTTP |
| `sources/TinyTitanBench` | Swift 6 | SwiftPM | `TinyTitanBench` |
| `sources/ContinuityCore`, `ContinuityDemo` | Swift 6 | SwiftPM | persistent agent memory core |
| `sources/TinyTitanMemory`, `TinyTitanMemoryTool` | Swift 6 | SwiftPM | memory surface + `tinytitan-memory` tool |
| `sources/TinyTitanFleet` (Core + Command) | Swift 6 | SwiftPM | `ttlanmanager` (LAN fleet manager) |
| `sources/TinyTitanValidation` | Swift 6 | SwiftPM | test/reference support only |
| `benchmark/` | Python 3.13 | `python3 -m unittest` | converter `prepare_qwen38.py` + test/measure scripts |
| `plugins/dsh-tinytitan` | JavaScript | npm (eslint + prettier) | DeepSeek Harness bundle (route writer, quiet compaction) |
| `plugins/dsh-lan-manager` | JavaScript | npm | DSH LAN manager front end for `ttlanmanager` |
| `tools/` | bash 3.2-compatible | — | `lint.sh`, `release.sh`, `install_models.sh`, `server_launcher.sh`, `golden-baseline.sh`, … |
| `tests/` | Swift 6 (swift-testing) | SwiftPM | mirrors `sources/` path for path; never loads a model |

Primary host: the development Mac (§1b). Dependencies: `swift-transformers`
(from 1.3.0, `Package.resolved`), `swift-nio` (exact 2.100.0). No other runtime
dependencies.

## 2.2 Dependency graph + cross-project contracts (2-hop cap)

Target graph: `KernelsC`/`Format` → `TinyTitan` → `TinyTitanLib` →
{`CLICore`, `ServerCore`}; `RepackCore` → `Format`; `Memory` → `ContinuityCore`;
`Memory` → `ServerCore`; `FleetCore` standalone; `Bench` → `TinyTitan`.

Contracts with >1 consumer (auto Tier A):

1. **`.ssdai` format + `verified-install.json` receipt** — written by
   `RepackCore`/`benchmark/prepare_qwen38.py`, read and *enforced* by the
   engine (trust decision), checked by `tools/release.sh`. Consumers: 3+.
2. **Loopback OpenAI-/Anthropic-compatible HTTP API** — served by
   `ServerCore`, consumed by arbitrary external clients (Zed, Codex, DSH
   plugin, curl). Untrusted-input surface by design.
3. **Fleet group key** — `FleetGroupKey.shippedDefault` (Swift) must equal
   `DEFAULT_GROUP_KEY` (plugin `config.js`); drift empties a peer table.
   Guarded by the `docs` gate (`tools/docs-facts.py`).
4. **ANE prefill sidecar + Qwen3.8 auxiliary checkpoint payload** — written by
   converter/installer, read by engine at prefill/restore; corruption must
   fail closed (v6.0 work).
5. **`TinyTitanLib` public facade** — external embedders (proven by
   `examples/embedded` + `tools/embedded-dependency-check.sh`).

## 2.3 Trust boundaries

- **Loopback HTTP server** (`ServerCore`): untrusted request bodies (JSON),
  no auth by design, `127.0.0.1`-only — boundary condition of the whole
  threat model.
- **`.ssdai` install files**: hash-verified at install, receipt-bound to an
  absolute path; the engine treats on-disk bytes as untrusted (a swapped or
  edited install must be refused, not served).
- **Swift↔C seam** (`module.modulemap` over `TinyTitanKernelsC`): pointer
  lifetimes across 609 lines of C99 — the classic bug-dense boundary.
- **Metal buffers / mmap** in `TinyTitan`: native memory access, Tier A.
- **Prompt cache / interior checkpoints** (v6.0): restored from SSD into
  decode state; corrupt records must fall back to full prefill, never serve
  wrong text.
- **Agent memory DB** (`ContinuityCore`/`TinyTitanMemory`): persistent data
  mutation, and LLM output (model tool calls) writing into it.
- **Converter downloads** (`benchmark/prepare_qwen38.py`): network fetch
  (HF/CDN) writing production installs; also `ttlanmanager` ↔ LAN harnesses.
- **LLM output driving privileged actions**: the DSH plugin executes harness
  tool calls; engine tool-calling path in `ServerCore`.

## 2.4 Tier table

| Module | Tier | Why |
| --- | --- | --- |
| `TinyTitanKernelsC` | **A** | C99 native memory (§2.4 rule: every native-memory module is Tier A) |
| `TinyTitan` (engine runtime) | **A** | unsafe memory, mmap, Metal, `.ssdai` byte parsing, concurrency (decode loop, slots) |
| `TinyTitanFormat` | **A** | untrusted-input parsing (format + manifest + receipt attestation) |
| `TinyTitanLib` | **A** | public facade; concurrency (Engine/Session lifetime, cancellation) |
| `TinyTitanServerCore` | **A** | network-facing endpoint, untrusted JSON, tool-calling path |
| `TinyTitanRepackCore` (+Command) | **A** | irreversible writes (installer), receipt attestation, resume/discard paths |
| `benchmark/prepare_qwen38.py` + its resume/download loop | **A** | writes production data (installs); network downloads; §1 Python carve-out does not de-scope this |
| `ContinuityCore` | **A** | persistent-data mutation (memory DB v2) |
| `TinyTitanMemory` (+Tool) | **A** | persistent-data mutation, LLM-output-driven writes |
| `TinyTitanFleetCore` (+Command) | **A** | network client to LAN harnesses; cross-project contract (group key) |
| `TinyTitanCLICore` (+Command) | **B** | local front end over the facade; parses operator argv, not remote input |
| `TinyTitanBench` | **B** | measurement driver, not on customer path |
| `plugins/dsh-tinytitan` | **B** | writes local harness config (audited by eslint/prettier + docs gate contract check); L4 scanner applies |
| `plugins/dsh-lan-manager` | **B** | local operator UI |
| `tools/release.sh`, `tools/install_models.sh` | **A** | irreversible/publish operations (deploy + install tooling; L7) |
| remaining `tools/*.sh` | **B** | gates/launchers (tool-first: shellcheck + portability gate own them) |
| `benchmark/` measurement scripts (non-converter) | **C** | non-production measurement/report code |
| `tests/` | **C** | scanner-only per §2.4 |
| `docs/`, `examples/`, `assets/` | **C** | docs, consumer-fixture package |

Unsure → Tier A applied where any doubt existed (e.g. FleetCore, memory tool).

Tier coverage disclosure (for the final report): Tier A modules get L2+L3+L4
manual passes; Tier B gets tools + findings-driven review only; Tier C gets
formatter/linter/secret-scan only. All findings from all tiers are enumerated
and closed (§2.4).
