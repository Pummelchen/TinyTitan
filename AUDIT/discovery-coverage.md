# Discovery-only coverage and remaining audit work

Source revision: 7b0901d75b3b1e79498874babf93218a617a69ac. GitHub API retrieved selected files without a local checkout. Local Linux x86_64 DNS failure confirmed again; no Xcode. No compilation, tests, negative controls, sanitizers, history scan, SAST, dependency audit or fix performed. Prior infrastructure tasks still open. Source comments are not independent tool output.

## Reviewed for targeted findings (partial file review)

- Server: HTTPServer, HTTPServerHandler, Routes, Plumbing, Responses, ResponsesAPIStore.
- ContinuityCore: Journal, FileJournal+Opening.
- DSH plugin: backend, config, generate.
- TinyTitanFormat: all eleven Swift files screened; focused excerpts GGUFHeaderV3, ExpertBlobIndex.
- Native C: seven C files under TinyTitanKernelsC screened; focused excerpts expert_io and gdn_chunk.
- Repack/Core: Posix, InstallLock, RemoteRangeTransfer, RemoteDownloadSession, BoundedScratch screened; no closure claimed.
- Core: BoundedMetadataRead screened; memory: MemoryProjectFile, MemoryService+Workspaces, MemoryTools screened.

Screening is not full line-by-line review. Files with no recorded finding are not certified clean. Findings are source-supported conditions or explicit investigation candidates. No live systems touched.

## Remaining work packages (not defect counts)

Use AUDIT/inventory.json for full file/target enumeration. For every package below perform L0–L7 per tier, contract mapping two hops, facade checks on production paths and Tier A/orphan unused-code checks. Record discrete evidenced findings; do not count this checklist as defects.

| Work package | Required discovery |
|---|---|
| Build and supply chain | Reproducibility, all manifests, Swift/C/Metal/Node/Python/shell gates, resolved versions, lock integrity, pinned CI actions, CVEs, full-history redacted secrets scan, proof for every delegated rule |
| Server HTTP + APIs | Complete request/response/streaming validation, authorization and trust boundaries, cancellation, queue bounds, memory budgets, model unload, error mapping, API contracts, health/shutdown and tests |
| ContinuityCore | Full state machine, append/compact/flush/replay ordering, resource lifetimes, permissions, corruption, failure injection, session isolation |
| Memory + memory CLI | Workspace path mapping, retention/deletion, tool authorization, model-output validation, durable data, prompt injection, privacy, resource limits |
| Format + model I/O | All binary/JSON parsers, offset/shape arithmetic, integrity, decompression bounds, symlinks and truncated/malformed model tests |
| Repack/converters/install | Remote redirects and credential forwarding, range verification, resumability, atomic install, rollback, disk quotas, file modes, subprocess, cross-consumer format contracts |
| C/Metal + Swift interop | Full manual C and GPU kernel review, pointer lifetimes, FFI sizes/nullability, integer math, synchronization, ASan+UBSan full suite and concurrency TSan |
| Core + runtime | All engine variants and legacy engine; allocation and cache lifetimes, numerical correctness, tokenizer and tool parsers, cancellation, context limits, sampling and error paths; golden comparisons |
| Library/CLI/plugin API | Public contracts, engine selection, session lifecycle, plugin registration, integration with server, example packages and warnings gates |
| Client integrations | All DSH/OpenCode files, shell/JS writer parity, config persistence, prompt/tool actions, bounded HTTP, error handling, npm audit and tests |
| Tools/benchmark/release | Production-path installers, publishers, conversions and credential handlers Tier A; all other scripts/tests/docs scanner-only unless higher-risk behavior discovered; verify gates actually fail, test assertions, release reproducibility |
| Coverage and ops | Critical failure and boundary tests, multi-consumer contracts, independent verification after eventual repair; no deployment |

## Completion status

Discovery not complete. No claim that all tasks have been discovered. User requested task list only, so no fixes or remediation closure. Resume from ledger, expand inventory classification and evidence-backed findings rather than restart. All new tasks have fix fields unset.
