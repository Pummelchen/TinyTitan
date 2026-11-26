# Contract graph (source-derived)

Method: first-party `import` statements parsed from every Swift file in each target path, with external product names normalized. The manifest graph is in `AUDIT/inventory.json`; this page records where actual imports differ. Two hops = a module's direct first-party imports plus their first-party imports.

| Target | Direct first-party imports | Two-hop first-party closure |
|---|---|---|
| ContinuityCore | — | — |
| ContinuityDemo | ContinuityCore | — |
| DenseGPU | TinyTitanCore, TinyTitanFormat, TinyTitanKernelsC, TinyTitanPluginAPI | — |
| Inkling2GGUFQ4 | TinyTitanCore, TinyTitanFormat, TinyTitanPluginAPI | — |
| Inkling2NVFP4 | TinyTitanCore, TinyTitanFormat, TinyTitanPluginAPI | — |
| Ornith35B4Bit | TinyTitanCore, TinyTitanFormat, TinyTitanKernelsC, TinyTitanPluginAPI | — |
| Qwen35Dense2B4Bit | TinyTitanCore, TinyTitanKernelsC | TinyTitanFormat |
| Qwen35Dense2B8Bit | TinyTitanCore, TinyTitanKernelsC | TinyTitanFormat |
| Qwen35Dense4B4Bit | TinyTitanCore, TinyTitanKernelsC | TinyTitanFormat |
| Qwen35Dense4B8Bit | TinyTitanCore, TinyTitanKernelsC | TinyTitanFormat |
| Qwen35Dense9B4Bit | TinyTitanCore, TinyTitanKernelsC | TinyTitanFormat |
| Qwen35Dense9B8Bit | TinyTitanCore, TinyTitanKernelsC | TinyTitanFormat |
| TemplatePlugin | TinyTitanCore, TinyTitanPluginAPI | TinyTitanFormat |
| TinyTitan | TinyTitanCore, TinyTitanFormat, TinyTitanKernelsC | — |
| TinyTitanBench | TinyTitan, TinyTitanCore, TinyTitanLegacyEngine, TinyTitanLib, TinyTitanVariants | DenseGPU, Inkling2GGUFQ4, Inkling2NVFP4, Ornith35B4Bit, Qwen35Dense2B4Bit, Qwen35Dense2B8Bit, Qwen35Dense4B4Bit, Qwen35Dense4B8Bit, Qwen35Dense9B4Bit, Qwen35Dense9B8Bit, TemplatePlugin, TinyTitanFormat, TinyTitanKernelsC, TinyTitanPluginAPI |
| TinyTitanCLI | TinyTitanCLICore | TinyTitan, TinyTitanCPUEngine, TinyTitanCore, TinyTitanLegacyEngine, TinyTitanLib, TinyTitanPluginAPI, TinyTitanVariants |
| TinyTitanCLICore | TinyTitan, TinyTitanCPUEngine, TinyTitanCore, TinyTitanLegacyEngine, TinyTitanLib, TinyTitanPluginAPI, TinyTitanVariants | DenseGPU, Inkling2GGUFQ4, Inkling2NVFP4, Ornith35B4Bit, Qwen35Dense2B4Bit, Qwen35Dense2B8Bit, Qwen35Dense4B4Bit, Qwen35Dense4B8Bit, Qwen35Dense9B4Bit, Qwen35Dense9B8Bit, TemplatePlugin, TinyTitanFormat, TinyTitanKernelsC |
| TinyTitanCPUEngine | TinyTitanCore, TinyTitanLegacyEngine, TinyTitanLib, TinyTitanVariants | DenseGPU, Inkling2GGUFQ4, Inkling2NVFP4, Ornith35B4Bit, Qwen35Dense2B4Bit, Qwen35Dense2B8Bit, Qwen35Dense4B4Bit, Qwen35Dense4B8Bit, Qwen35Dense9B4Bit, Qwen35Dense9B8Bit, TemplatePlugin, TinyTitan, TinyTitanFormat, TinyTitanPluginAPI |
| TinyTitanCore | TinyTitanFormat | — |
| TinyTitanFormat | — | — |
| TinyTitanKernelsC | — | — |
| TinyTitanLegacyEngine | TinyTitan, TinyTitanCore, TinyTitanLib | TinyTitanFormat, TinyTitanKernelsC |
| TinyTitanLib | TinyTitanCore | TinyTitanFormat |
| TinyTitanMemory | ContinuityCore | — |
| TinyTitanMemoryTool | ContinuityCore, TinyTitanMemory | — |
| TinyTitanPluginAPI | TinyTitanCore | TinyTitanFormat |
| TinyTitanRepack | TinyTitanRepackCore | TinyTitanCore, TinyTitanFormat |
| TinyTitanRepackCore | TinyTitanCore, TinyTitanFormat | — |
| TinyTitanServer | TinyTitanCPUEngine, TinyTitanCore, TinyTitanLegacyEngine, TinyTitanLib, TinyTitanMemory, TinyTitanPluginAPI, TinyTitanServerCore, TinyTitanVariants | ContinuityCore, DenseGPU, Inkling2GGUFQ4, Inkling2NVFP4, Ornith35B4Bit, Qwen35Dense2B4Bit, Qwen35Dense2B8Bit, Qwen35Dense4B4Bit, Qwen35Dense4B8Bit, Qwen35Dense9B4Bit, Qwen35Dense9B8Bit, TemplatePlugin, TinyTitan, TinyTitanFormat |
| TinyTitanServerCore | TinyTitan, TinyTitanCPUEngine, TinyTitanCore, TinyTitanLegacyEngine, TinyTitanLib, TinyTitanMemory, TinyTitanPluginAPI, TinyTitanVariants | ContinuityCore, DenseGPU, Inkling2GGUFQ4, Inkling2NVFP4, Ornith35B4Bit, Qwen35Dense2B4Bit, Qwen35Dense2B8Bit, Qwen35Dense4B4Bit, Qwen35Dense4B8Bit, Qwen35Dense9B4Bit, Qwen35Dense9B8Bit, TemplatePlugin, TinyTitanFormat, TinyTitanKernelsC |
| TinyTitanValidationSupport | TinyTitanCore | TinyTitanFormat |
| TinyTitanVariants | DenseGPU, Inkling2GGUFQ4, Inkling2NVFP4, Ornith35B4Bit, Qwen35Dense2B4Bit, Qwen35Dense2B8Bit, Qwen35Dense4B4Bit, Qwen35Dense4B8Bit, Qwen35Dense9B4Bit, Qwen35Dense9B8Bit, TemplatePlugin, TinyTitanCore, TinyTitanLegacyEngine, TinyTitanLib, TinyTitanPluginAPI | TinyTitan, TinyTitanFormat, TinyTitanKernelsC |

## Manifest versus actual imports

- **Qwen35Dense2B4Bit**: actual-only —; manifest-only ['TinyTitanFormat'].
- **Qwen35Dense2B8Bit**: actual-only —; manifest-only ['TinyTitanFormat'].
- **Qwen35Dense4B4Bit**: actual-only —; manifest-only ['TinyTitanFormat'].
- **Qwen35Dense4B8Bit**: actual-only —; manifest-only ['TinyTitanFormat'].
- **Qwen35Dense9B4Bit**: actual-only —; manifest-only ['TinyTitanFormat'].
- **Qwen35Dense9B8Bit**: actual-only —; manifest-only ['TinyTitanFormat'].
- **TinyTitanLib**: actual-only —; manifest-only ['TinyTitanFormat'].
- **TinyTitanServer**: actual-only ['TinyTitanCore', 'TinyTitanLegacyEngine', 'TinyTitanLib', 'TinyTitanMemory']; manifest-only —.

Actual-only imports are module names the compiler can see through a dependency's transitive closure (SwiftPM permits it); they are a real coupling the manifest graph misses. Manifest-only entries are declared dependencies with no direct import in the target (used through a re-export or unused); neither is a runtime defect on its own, but both are recorded rather than smoothed over.

## Contract table (key boundaries)

| Contract | Producer → consumers | Format/version | Validation owner | Failure policy | Tests |
|---|---|---|---|---|---|
| HTTP request/response, SSE | TinyTitanServerCore → CLI clients, DSH, OpenCode, SDKs | HTTP/1.1 + OpenAI/Anthropic JSON/SSE | `HTTPServerHandler` routes + surface builders | typed error envelope; 4xx/5xx; connection close on framing faults | `tests/TinyTitanServer/APISurfaceHTTPTests.swift`, `HTTPServerTests` |
| Model catalog JSON | TinyTitanServerCore `ModelCatalog` → server `--catalog`, DSH `catalog-scan.js` | JSON rows (id/name/family/quant/backend/engines/path/thinking/sampling/size_gb) | `ModelCatalog` probe + `ManifestReader` | skip/refuse an install with a reason; mirror must match | `ModelCatalogTests`, `catalog-scan.test.js` |
| `.ssdai` install | TinyTitanRepack → engines/library | SSDAI manifest/layout + `verified-install.json` receipt | repack writer + install verifier | refuse truncated/swapped/edited installs; receipt bound to absolute path | `TinyTitanRepackTests`, golden baselines |
| GGUF header/index | TinyTitanFormat `GGUFHeaderV3`/`ExpertBlobIndex` → engines/repack | GGUF v2/v3; TTXI v2/v3 | parser bounds + typed errors | typed parse failure; no partial metadata | `TinyTitanRepackTests`, format tests |
| Journal / memory | ContinuityCore `FileJournal` → TinyTitanMemory, server memory endpoints, memory CLI | newline-delimited JSON records + checkpoint | `FileJournal` replay/compaction | typed durability/replay errors; no silent corruption | `JournalTests`, `TinyTitanMemoryTests` |
| Plugin registry (TPP/1.1) | `PluginRegistry` → variants/server/CLI | `EnginePlugin` protocol 1.1 | `PluginConformance` + registry | refuse unknown/mismatched major; explicit unsupported parameters | `PluginRegistryConformanceTests`, `TinyTitanPluginAPITests` |
| Swift ↔ C kernels | Swift engine targets → TinyTitanKernelsC | C ABI (`tinytitan_*`) | C preconditions + Swift wrappers | checked geometry; typed errno/return | kernel tests + ASan/UBSan |
| DSH settings YAML | DSH plugin/shell writer → DSH profile | YAML route block | `yamlScalar`/`yaml_scalar` | atomic replace; conflict refusal; safe diagnostics | `generate.test.js`, `test_dsh_route` |
