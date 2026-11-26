# Phase A — initial inventory (provisional)

Base commit: `7b0901d75b3b1e79498874babf93218a617a69ac`. Git tree: 2,814 entries / 2520 blobs. All tracked files enumerated in inventory.json. Root package has 46 targets (14 test targets). Conservative file classification: Tier A: 1522, Tier C: 998. These are provisional assignments, not reviewed-file counts. Uncertain production surfaces are Tier A; validate the benchmark/example/tool classifications individually. No source-level review coverage claimed.

## Projects and primary hosts

| Project | Build / language | Entry points | Assigned primary host (not provisioned) |
|---|---|---|---|
| Root TinyTitan | SwiftPM 6.4, Swift, C99, Metal | TinyTitanServer, TinyTitanCLI, TinyTitanRepack, TinyTitanBench, ContinuityDemo, tinytitan-memory, library and plugin products | Apple Silicon Mac A / Xcode 27 |
| Embedded example | SwiftPM 6.4 | EmbeddedDemo | Mac A |
| Plugin example | SwiftPM 6.4 | EchoPlugin, PluginDemo | Mac A |
| Library demo | Swift direct / shell | build-static.sh, build-dynamic.sh | Mac A |
| Model conversion and integrated harness | Python, shell, Swift, C | tools/, benchmark/ | Mac A (Apple framework and installed-model integration) |
| DSH plugin | Node ESM, npm | plugins/dsh-tinytitan/src/index.js and backend.js | Linux primary B |
| OpenCode provider | JSON config | plugins/opencode-tinytitan/ | Mac A (server integration) |
| CI / release / docs tooling | shell, Python, YAML, HTML | .github/workflows/, tools/, docs/paper/ | Mac A |

Standalone portable Node audit requires Linux B; integrated server tests belong to Mac A. Independent Mac C not provisioned. No VPS requested or provisioned. Host labels are proposed assignments, not claims of access.

## Root target graph (manifest-derived, direct edges)

| Target | Kind | Provisional tier | Direct dependencies |
|---|---|---|---|
| TinyTitanFormat | target | A | — |
| TinyTitanKernelsC | target | A | — |
| TinyTitanCore | target | A | TinyTitanFormat, swift-transformers:Tokenizers, yyjson:yyjson |
| Qwen35Dense2B4Bit | target | A | TinyTitanCore, TinyTitanFormat, TinyTitanKernelsC |
| Qwen35Dense2B8Bit | target | A | TinyTitanCore, TinyTitanFormat, TinyTitanKernelsC |
| Qwen35Dense4B4Bit | target | A | TinyTitanCore, TinyTitanFormat, TinyTitanKernelsC |
| Qwen35Dense4B8Bit | target | A | TinyTitanCore, TinyTitanFormat, TinyTitanKernelsC |
| Qwen35Dense9B4Bit | target | A | TinyTitanCore, TinyTitanFormat, TinyTitanKernelsC |
| Qwen35Dense9B8Bit | target | A | TinyTitanCore, TinyTitanFormat, TinyTitanKernelsC |
| DenseGPU | target | A | TinyTitanCore, TinyTitanFormat, TinyTitanKernelsC, swift-transformers:Tokenizers, TinyTitanPluginAPI |
| Ornith35B4Bit | target | A | TinyTitanCore, TinyTitanFormat, TinyTitanKernelsC, swift-transformers:Tokenizers, TinyTitanPluginAPI |
| TinyTitanPluginAPI | target | A | TinyTitanCore |
| TemplatePlugin | target | A | TinyTitanPluginAPI, TinyTitanCore |
| Inkling2NVFP4 | target | A | TinyTitanCore, TinyTitanFormat, TinyTitanPluginAPI |
| Inkling2GGUFQ4 | target | A | TinyTitanCore, TinyTitanFormat, TinyTitanPluginAPI |
| TinyTitan | target | A | TinyTitanCore, TinyTitanFormat, TinyTitanKernelsC, swift-transformers:Tokenizers |
| TinyTitanLib | target | A | TinyTitanCore, TinyTitanFormat |
| TinyTitanLegacyEngine | target | A | TinyTitanCore, TinyTitan, TinyTitanLib |
| TinyTitanVariants | target | A | TinyTitanCore, TinyTitanLib, TinyTitanLegacyEngine, Ornith35B4Bit, DenseGPU, Inkling2GGUFQ4, Inkling2NVFP4, TinyTitanPluginAPI, TemplatePlugin, Qwen35Dense2B4Bit, Qwen35Dense2B8Bit, Qwen35Dense4B4Bit, Qwen35Dense4B8Bit, Qwen35Dense9B4Bit, Qwen35Dense9B8Bit |
| TinyTitanCPUEngine | target | A | TinyTitanCore, TinyTitanLib, TinyTitanLegacyEngine, TinyTitanVariants |
| TinyTitanRepackCore | target | A | TinyTitanCore, TinyTitanFormat |
| TinyTitanRepack | executableTarget | A | TinyTitanRepackCore |
| TinyTitanCLICore | target | A | TinyTitanCore, TinyTitan, TinyTitanLib, TinyTitanLegacyEngine, TinyTitanVariants, TinyTitanPluginAPI, TinyTitanCPUEngine |
| TinyTitanCLI | executableTarget | A | TinyTitanCLICore |
| ContinuityCore | target | A | — |
| ContinuityDemo | executableTarget | C | ContinuityCore |
| TinyTitanMemory | target | A | ContinuityCore |
| TinyTitanMemoryTool | executableTarget | A | TinyTitanMemory, ContinuityCore |
| TinyTitanServerCore | target | A | TinyTitanCore, TinyTitanLib, TinyTitanPluginAPI, TinyTitan, TinyTitanLegacyEngine, TinyTitanMemory, TinyTitanVariants, TinyTitanCPUEngine, swift-nio:NIOCore, swift-nio:NIOPosix, swift-nio:NIOHTTP1 |
| TinyTitanServer | executableTarget | A | TinyTitanServerCore, TinyTitanVariants, TinyTitanCPUEngine, TinyTitanPluginAPI |
| TinyTitanBench | executableTarget | C | TinyTitanCore, TinyTitan, TinyTitanLib, TinyTitanLegacyEngine, TinyTitanVariants |
| TinyTitanValidationSupport | target | C | TinyTitanCore |
| TinyTitanTests | testTarget | C | TinyTitanVariants, yyjson:yyjson, Qwen35Dense2B4Bit, Qwen35Dense2B8Bit, Qwen35Dense4B4Bit, Qwen35Dense4B8Bit, Qwen35Dense9B4Bit, Qwen35Dense9B8Bit, TinyTitanCore, TinyTitan, TinyTitanFormat, TinyTitanValidationSupport, TinyTitanRepackCore, TinyTitanCLICore, TinyTitanLib |
| TinyTitanRepackTests | testTarget | C | TinyTitanRepackCore, TinyTitanFormat, yyjson:yyjson |
| ContinuityCoreTests | testTarget | C | ContinuityCore, yyjson:yyjson |
| TinyTitanCoreTests | testTarget | C | TinyTitanCore, yyjson:yyjson |
| Qwen35Dense2B4BitTests | testTarget | C | Qwen35Dense2B4Bit, TinyTitanCore, yyjson:yyjson |
| Ornith35B4BitTests | testTarget | C | Ornith35B4Bit, TinyTitan, TinyTitanCore, yyjson:yyjson |
| Inkling2NVFP4Tests | testTarget | C | Inkling2NVFP4, TinyTitanCore |
| TinyTitanPluginAPITests | testTarget | C | TinyTitanPluginAPI, TinyTitanCore |
| TemplatePluginTests | testTarget | C | TemplatePlugin, TinyTitanPluginAPI, TinyTitanCore |
| Inkling2GGUFQ4Tests | testTarget | C | Inkling2GGUFQ4, TinyTitan, TinyTitanCore, TinyTitanPluginAPI |
| DenseGPUTests | testTarget | C | DenseGPU, TinyTitanCore, TinyTitan, yyjson:yyjson |
| TinyTitanMemoryTests | testTarget | C | TinyTitanMemory, ContinuityCore, yyjson:yyjson |
| TinyTitanServerTests | testTarget | C | yyjson:yyjson, TinyTitanCore, TinyTitanServerCore, TinyTitanPluginAPI, TinyTitanLib, TinyTitanMemory, TinyTitan, TinyTitanVariants, TinyTitanCPUEngine, swift-nio:NIOEmbedded |
| TinyTitanLibTests | testTarget | C | yyjson:yyjson, TinyTitanCore, TinyTitanLib, TinyTitan, TinyTitanLegacyEngine, TinyTitanVariants |

## Preliminary cross-project contracts and trust boundaries

- HTTP/JSON/SSE: server ↔ CLI/DSH/OpenCode/other clients; OpenAI/Anthropic responses and tool-call schemas; auth, limits, malformed input and cancellation require Tier A review.
- Model format and disk: repack/converters ↔ SSD AI/model manifests ↔ engines and library; untrusted downloads, file offsets, sizes, partial writes, rollback, symlinks, path traversal.
- Native memory: Swift ↔ TinyTitanKernelsC/NEON/Accelerate, Swift ↔ yyjson, Metal buffer/shader ABI and model data lifetimes; Tier A.
- Persistence: ContinuityCore ↔ TinyTitanMemory ↔ server/tools; journal, workspace and session state; Tier A.
- Plugin API ↔ registry/factory ↔ engines; model output and tool calls ↔ client actions; validate authorization and schema boundaries.
- Shell release/install drivers, environment variables and credentials, supply chain and dependency locks; Tier A pending source review.

The direct target graph is from Package.swift. Two-hop adjacency is in inventory.json; the contract list is preliminary, not a completed source-derived two-hop cross-project graph. No manual L0–L7 passes are complete. All non-test runtime targets except demo/benchmark/validation support are provisionally A; no Tier B reduction has been claimed.
