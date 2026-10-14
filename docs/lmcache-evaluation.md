# LMCache evaluated against TinyTitan

2026-10-14, for the ≥6 tok/s qwen38-at-12G campaign. Source: the LMCache
repository and documentation as of this date.

## What LMCache is

LMCache is a vendor-neutral **KV-cache layer** for serving engines — Python and
Rust, integrated into vLLM-class Python serving stacks. Its capabilities:

- **Tiered KV offload**: attention KV moves GPU → CPU RAM → SSD → Redis/S3/NIXL
  so the GPU keeps less resident.
- **Cross-request prefix dedup and sharing**: two requests sharing a prompt
  prefix compute its KV once (in-memory, on disk, or from a peer).
- **CacheBlend**: reuse of *non-prefix* KV — previously computed chunks are
  re-blended with a small selective recompute instead of a full prefill.
- **PD disaggregation**: prefill and decode run as separate roles on separate
  nodes, with KV shipped between them; P2P KV transfer between serving
  instances.

Every one of these attacks **TTFT and aggregate throughput under multi-tenant
load**: many requests, shared prefixes, more context than GPU memory. Its own
documentation claims TTFT and throughput wins; it names no single-request
decode-latency mechanism.

## What TinyTitan's bottleneck is

The campaign's measurements (docs/qwen38-decode-12gb-measurements.md) are
unambiguous: at `--ram 12G`, decode is bound on **streaming routed-expert
weights from SSD** — ~117 misses/token at ~2.7 MiB each, ~84 ms/token of
exposed GPU wait against ~132 ms of GPU body. Attention KV is resident in
Metal buffers; it is read from unified memory every token and is not the
constraint. §5n measured context length costing ~20% of decode speed, and the
mechanism is *indirect*: the 262K-token KV reservation crowds expert cache
slots out of RAM, dropping the expert hit rate (0.759 → 0.682).

TinyTitan also already has the LMCache feature that fits its workload: the
server's prompt cache is multi-prefix by default
(`--prompt-cache-mode multi-prefix`), reusing KV across requests in-process —
the single-user, loopback, tool-loop case where prefixes are long and shared.

## Concept-by-concept mapping

| LMCache concept | Maps to TinyTitan? | Decode tok/s effect |
| --- | --- | --- |
| Cross-request prefix dedup | Already shipped (multi-prefix prompt cache, in-process) | None new |
| KV offload to SSD tier | Would trade the KV reservation for expert-cache RAM — the §5n mechanism | Negative: decode reads KV **every token**; an SSD KV tier adds a second streaming consumer to the I/O class that is already the bottleneck |
| KV offload to CPU tier | No-op — Apple unified memory means there is no GPU/CPU split to exploit; KV is already in "CPU RAM" | None |
| CacheBlend (non-prefix reuse) | No caller: local tool-loop prompts are prefix-structured; the reuse is already captured by the prefix cache | TTFT only, and only for RAG-style workloads TinyTitan does not serve |
| PD disaggregation / P2P / Redis/S3 | Multi-node serving — out of scope (AGENTS.md: an engine and its loopback server, nothing else) | N/A |

## Verdict

**LMCache's problem is not TinyTitan's problem.** It exists to move attention
KV around a data-center serving fleet; TinyTitan's 12G decode wall is expert
*weight* streaming on a single Mac. Nothing in LMCache touches decode tok/s
for a single-stream local engine, and its central mechanism (offload KV to
slower tiers) runs *against* the physics here: unified memory already places
KV in the cheapest tier it could be moved to, and pushing KV to SSD would
contend with the expert reads that already own the disk.

**Testing it directly is not possible in any meaningful sense**: LMCache is a
Python/Rust layer bound to vLLM's paged-KV layout and scheduler. There is no
adapter surface — "testing it with TinyTitan" would mean reimplementing its
concepts in Swift/Metal, which the table above prices as neutral-to-harmful
for the stated goal.

The one idea worth keeping is **KV-as-a-budget-valve**: shrinking the KV
reservation (a `--max-context` cap) frees RAM for expert slots without any
new machinery. That is exactly what the #147 budget sweep measures; if a
smaller context reservation wins, the answer is a flag default, not a cache
layer.

Re-evaluate only if the workload changes: a multi-session deployment sharing
long non-prefix context (RAG over a document set) is where CacheBlend-class
reuse would start paying, and that deployment is currently out of scope.
