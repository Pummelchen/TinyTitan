# Scope, contracts and tiers

Discovery output for §2 of the audit standard: projects, languages, build systems,
entry points, the 2-hop dependency graph, trust boundaries, and the Tier A/B/C
classification. Committed before any fix, so the tier coverage disclosed in the
final report is the table that was actually worked from.

Host: the Mac described in [environment.md](environment.md). This is a single
project — one SwiftPM package, two products on one branch.

## 2.1 Projects, languages, build systems, entry points

| Surface | Language | Build | Entry point |
| --- | --- | --- | --- |
| `TinyTitanLib` (+ dynamic variant) | Swift 6.4 / `.v6` | `Package.swift:44-45`, `:128` | library façade: `Engine`, `Session` |
| `TinyTitan` (engine runtime) | Swift + Metal | `Package.swift:110` | embedded by the executables |
| `TinyTitanFormat`, `TinyTitanValidationSupport` | Swift | `:66`, `:228` | format decode, reference oracles |
| `TinyTitanKernelsC` | strict C99 | `:94`, `.c99` + `-Werror` set | C kernels behind the interop seam |
| `TinyTitanCLI`, `TinyTitanServer`, `TinyTitanRepack`, `TinyTitanBench`, `tinytitan-memory`, `ttlanmanager`, `ContinuityDemo` | Swift executables over `*Core` targets | `:47-59` | `Command/main.swift` / `@main` in each |
| `tools/*.sh` (installer, launcher, release, gates) | bash 3.2 | `tools/lint.sh` shell gates | `install_tinytitan.sh`, `server_launcher.sh` |
| `benchmark/*.py`, `tools/prepare_*.py` | Python 3.13 floor | `pyproject.toml`, `benchmark/requirements.txt` | unittest suites, the converter |
| `plugins/dsh-lan-manager`, `plugins/dsh-tinytitan` | JavaScript (ESM) | own `package.json` + lockfile, eslint/prettier | harness plugin entry + `/dsh-lan` router |
| `examples/embedded` | Swift | `tools/embedded-dependency-check.sh` | consumer package that proves the library is consumable |

## 2.2 Dependency graph — direct edges, 2 hops

Derived from `Package.swift` and from the `import` statements themselves.

```
TinyTitanKernelsC (C)        TinyTitanFormat (Foundation only)
        ^   ^                        ^  ^  ^  ^
        |   |                        |  |  |  |
Tokenizers  +-- TinyTitan -----------+  |  |  +-- TinyTitanRepackCore
                 ^   ^   ^              |  |         ^
                 |   |   |              |  |         |
                 |  TinyTitanLib <------+--+   (executables: Repack, CLI, Server,
                 |   ^   ^   ^          |            Bench, memory, fleet, demo)
                 |   |   |   |          |
            ServerCore  CLICore  Validation  Memory -> ContinuityCore
                 ^                              FleetCore (Foundation/Darwin only)
                 |
                NIO (swift-nio 2.100.0, exact pin) — only here
```

No import cycles. `TinyTitanFormat` depends on nothing but Foundation, which is what
makes it the shared wire contract between the installer and the runtime.

Cross-project contracts found (the >1-consumer ones are automatically Tier A):

| Contract | Producers | Consumers | Tier |
| --- | --- | --- | --- |
| `.ssdai` manifest JSON | `TinyTitanRepack/Core/Format/SSDAIJSON.swift:37-122` | `TinyTitanFormat/SSDAIManifestV1.swift:308-348` (wire), `TinyTitan/Infrastructure/ModelIO/ManifestReader.swift:448-465` (runtime) | **A** |
| `verified-install.json` receipt | `Repack/Core/Verification/VerifiedInstallReceiptWriter.swift:4-44` | `TinyTitan/Infrastructure/ModelIO/VerifiedInstallReceipt.swift:76-149`, path binding `:174-205` | **A** |
| `ple_constants.json` sidecar | `tools/ane_sidecars.sh`, the add-a-model runbook | `TinyTitan/Runtime/Family/PLEConstants.swift` | **A** |
| Loopback HTTP `/v1/*` | `TinyTitanServer/Core/HTTPServerHandler+Routes.swift` | CLI, `tools/golden-baseline.sh:205-228`, `server_launcher.sh:1513-1837`, `dsh_route.sh:134`, `dsh_local.sh:802`, `plugins/dsh-tinytitan/src/generate.js:206`, `docs/server-api.md` | **A** (>1 consumer) |
| `tinytitan_models.sh` catalogue (tab-separated) | `tools/tinytitan_models.sh:318-393` | `server_launcher.sh:499-620`, `install_models.sh`, `benchmark/test_launcher_install.py` | **A** |
| Env vars (`TINYTITAN_*`, `HF_TOKEN`, `DSH_LAN_*`, ~100 names) | launcher, benchmark scripts, the installer | the runtime, the memory suite, the plugins | B, except the model-path and receipt set → A |
| `/dsh-lan` router (a second, separate server) | `plugins/dsh-lan-manager/src/router.js:276-294` | `TinyTitanFleet/Command/main.swift:95-138` | A (network-facing) |

Depth stopped at 2 hops per §2.2; nothing was recursed past it.

## 2.3 Trust boundaries

1. **Untrusted input → the engine**: the request body of the loopback HTTP server.
   1 MiB cap, content-type gate, then `OpenAIRequestValidator` → tokenizer →
   sampler. No remote authentication and no TLS, which is why the bind is hardcoded
   to `127.0.0.1` (`HTTPServer.swift:125`) and `AGENTS.md` forbids proxying it.
2. **Untrusted input → the model files**: `.ssdai` manifest, resident index,
   safetensors headers, `ple_constants.json`, n-gram table, packed-expert layout.
   All parsed by hand-written decoders with explicit bounds. A malicious or corrupt
   install is the attacker model here, because installs can be copied in from
   another machine.
3. **Untrusted network → the installer**: `curl` of a release tarball and the tag
   archive from GitHub, then `tar` + `chmod +x` + execute.
4. **Native-interop seams** (the densest class, per §2.3):
   - `ParallelExpertReader.swift:77-142` — the C expert-IO handle:
     `OpaquePointer` stored `:43`, created `:77`, destroyed in `deinit` `:91-93`,
     and `dst.withMemoryRebound(to: UnsafeMutableRawPointer?.self)` at `:112`/`:140`
     whose layout equality is asserted only by a comment (`:109-111`). The C side
     requires pre-sized destinations (`tinytitan_expert_io.h:49-51`).
   - GEMV calls: `CPUTensorOps.swift:145/149`, `CPUExpertFFN.swift:105`,
     `Int8AffineGEMV.swift:70/99/117`.
   - `String(cString:)` at 8 sites (`ParallelExpertReader.swift:56/58`,
     `NgramTableReader.swift:33/45`, `Repack/Core/System/Posix.swift:175`,
     `SSDAIDirectoryAccess.swift:203`, `ContinuityCore/Persistence/Journal.swift`,
     `JournalTypes.swift:79-81`) — reading past the terminator is the known failure
     mode of that call.
   - `mmap` / raw pointers: `ProcessMemory.swift:17`, `CPUEngine/SafeTensors.swift:71/97/164`.
   - Metal buffer `contents().bindMemory(...)`: `Sampler.swift:104/249`,
     `QSAIndexer.swift:127-144`, `RealForwardRunner+DecodeMoE.swift:42/129-131`,
     `ANEPrefillAttention+Models.swift:192-278`.
5. **Credential holders**: `HF_TOKEN` (read only at `Repack/Command/main.swift:243`,
   sent host-pinned at `HuggingFaceRemote.swift:262`, and stripped off-host by both
   redirect policies — `RemoteRangeTransfer.swift:372-382`,
   `RemoteDownloadSession.swift:153-168`); `DSH_LAN_KEY`/`DSH_LAN_TOKEN`
   (`TinyTitanFleet/Command/main.swift:115-116`, `plugins/dsh-lan-manager/src/config.js:132`).
6. **Irreversible operations**: the repack write path (weight files),
   `verified-install.json` re-issue, model directory moves, and the release/tag
   push. Migration-equivalent, hence Tier A whatever language they are in.
7. **LLM output driving privileged actions**: the tool-call path — a model emitted
   `tool_calls` reach the client's execution policy. `AGENTS.md` states the server
   must not bypass it; the LAN manager's `/prompt` route composes an agent session
   from request fields (`router.js:423-430`), which is the same shape.

## 2.4 Tier table

Tier A — deep manual review. Tier B — tool-first, manual only on tool findings,
coverage gaps and hotspots. Tier C — formatter/linter/secret-scan only.

| Module / surface | Tier | Why |
| --- | --- | --- |
| `TinyTitanKernelsC` (expert IO, int4/int8 GEMV) | **A** | native memory; in a C99 repo every module with native memory access is A |
| Interop seam files (`ParallelExpertReader`, `CPUTensorOps`, `CPUExpertFFN`, `Int8AffineGEMV`, `NgramTableReader`, `Posix`, `SSDAIDirectoryAccess`) | **A** | pointer lifetimes, `String(cString:)`, layout-rebind assumptions |
| `TinyTitan/Infrastructure/ModelIO` (manifest, receipt, resident index, safetensors) | **A** | untrusted-input parsing + persistent-data integrity |
| `TinyTitanFormat` (all decoders) | **A** | the wire contract with >1 consumer |
| `TinyTitan/Runtime` forward runner, attention, MoE, MTP, ANE paths | **A** | wrong numerics = wrong results = S0 class; Metal raw-pointer binding |
| `TinyTitanRepack/Core` (remote download, writer, verification) | **A** | irreversible data mutation, credential use, network input |
| `TinyTitanServerCore` + `TinyTitanLib` HTTP/wire layer (validator, routes, sessions) | **A** | network-facing surface, request parsing, concurrency |
| `tools/install_tinytitan.sh`, `server_launcher.sh`, `release.sh`, `tinytitan_models.sh`, `install_models.sh`, `golden-baseline.sh` | **A** | download-and-execute, contracts with >1 consumer, release identity |
| `TinyTitanMemory` + `ContinuityCore` (journal, store, lock) | **A** | persistent state, file locking, path scoping |
| `TinyTitanFleet` + `plugins/dsh-lan-manager` | **A** | LAN-facing HTTP server, token handling, agent session composition |
| `TinyTitanLib` façade + `examples/embedded` | **A** | the supported public promise |
| `TinyTitanLib/Watchdogs`, `BatchedMemoryBudget`, `ServerPromptStateStore` | B | resource policy; tool-gated, reviewed on findings |
| `TinyTitan/Tokenization`, `TinyTitanCLICore`, `ContinuityDemo`, `TinyTitanBench`, `TinyTitanValidation` | B | correctness matters but the surface is internal |
| `tools/prepare_qwen38.py`, `prepare_agentworld.py`, `benchmark/*.py` | B → **A** where they write production data | the converter *writes model weights*, so `prepare_*` are Tier A on that strength |
| `plugins/dsh-tinytitan` (route writer, compaction) | B | client-side of a Tier A contract |
| `tools/*.sh` helper scripts not in the A list above, `docs/*.md`, `README.md`, `examples/*` (except the embedded fixture), CI workflow YAML | B/C | reviewed as they are touched |
| `tests/**`, fixtures, `.qwen/wiki`, vendored checkouts under `.build` | **C** | scanner-only |

Tier coverage disclosure: every module in the package is named above; there is no
unclassified surface. Tier A holds the entire inference, format, repack, server,
memory, fleet and installer surface — which is most of the tree by line count, so
"reduced inspection" here applies mainly to Tier B internals and Tier C, and the
Tier C reduction is limited to tests, docs and vendored code.

## Language-standard status at discovery

Already enforced in build config, and *proved* in
[tool-coverage.md](tool-coverage.md) — not merely named:

- Swift: `swiftLanguageModes: [.v6]` (`Package.swift:316`), `-warnings-as-errors`
  (`:23`), SwiftLint `--strict` with `force_unwrapping` and
  `implicitly_unwrapped_optional` opted in (`.swiftlint.yml:43-75`), committed
  `.swift-format`. Toolchain Xcode 27 / Swift 6.4 — the only supported one.
- C: `.c99` (`:321`) plus `-pedantic-errors` and the full hardening set
  (`:95-108`), `-Werror`.
- Python: ruff pinned 0.16.7 with `B`, `E722`, `S101`, `PT` selected
  (`pyproject.toml:1-35`) — the two standards §1 names that the config *claims* and
  does not deliver are recorded as AUD-103 rather than assumed covered.
- Bash: system bash 3.2.57 parse-and-run gate plus shellcheck 0.11.0.
