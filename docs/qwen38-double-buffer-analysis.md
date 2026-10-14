# Double-buffered decode: why the classic shape does not apply at 12G

2026-10-14, task #150 of the ≥6 tok/s qwen38-at-12G campaign. The candidate
under examination is the profile's §6C "double-buffered plan" — two layers in
flight, encoding layer L+1's phase-1 before layer L's fixup completes
(docs/qwen38-decode-profile-2026-09-05.md). This note walks the current
control plane and asks where a second in-flight layer could come from.

## The pipeline as shipped

Per layer, `decodeLayerLoop`
(sources/TinyTitan/Runtime/Inference/RealForwardRunner+DecodeLayers.swift):

1. Encode and commit `attnCB` / `tailCB` (attention + router) and the shared
   expert `sharedCB`.
2. **Host wait on `tailCB`** — the routing indices must reach the CPU before
   anything expert-related can be planned.
3. Read back the indices, plan the fetch, pin slots, and (under immediate
   submission) issue the miss reads — all inside the first statements of
   `encodeDecodeRoutedMoE`
   (sources/TinyTitan/Runtime/Inference/RealForwardRunner+DecodeMoE.swift).
4. Event-driven I/O: no host wait for the load. The routed command buffer
   does `encodeWaitForEvent` on the MTLIO completion token and the loop
   proceeds to the next layer.
5. The previous layer's routed CB is finalized non-blocking at the next
   layer's tail wait.

So demand reads already issue at the earliest physically possible point — the
routing readback — and the host never blocks on the load. The exposed wait is
the **GPU** waiting on the I/O event inside the routed CB.

## Why there is no GPU slack to double-buffer

Two layers of GPU work in flight requires work that does not depend on the
previous layer's routed output. In this architecture none exists within a
token:

- Layer L+1's attention entry reads `hidden`, which is written by layer L's
  routed tail (the residual exit encoded into the routed CB). The dependency
  is data, not queue order: a second Metal queue with an event fence
  serializes identically.
- The shared expert and the phase-1 hit subset are already committed ahead of
  the wait and overlap the I/O.
- Phase-2 reduce and the residual tail are the work that needs the fetched
  bytes; splitting them finer creates no new work.

The residual chain is strictly sequential. The GPU's idle time during expert
I/O cannot be filled by re-ordering; it can only be shrunk.

## What could shrink it, and what each is worth here

Exposed wait per token at 12G is ~84 ms against ~132 ms of GPU body
(docs/qwen38-decode-12gb-measurements.md, runner-stats anatomy). Per layer
that is ~2.4 ms of exposure against ~3.8 ms of GPU work and ~3 ms of demand
read time — the read starts mid-layer and finishes just after the GPU reaches
it. The four ways to close that gap:

| Lever | Status |
| --- | --- |
| Fewer bytes: raise the expert hit rate (budget axis) | **Open** — #147 budget sweep measures `--expert-cache-slots` × `--max-context` |
| Faster bytes: deeper queue depth, coalescing, layout | **Open** — #149 miss-read path; the disk runs ~1.4 GB/s effective against a multi-GB/s device |
| Earlier issue: speculation beyond the prefetch ring | **Closed** — second rank −3.3%, second horizon −1.4%, depth≠1 −2.8% (docs/qwen38-prefetch-predictor-study.md); capacity-bound evictions eat the gain |
| More GPU work per layer to hide the wait behind | **Does not exist** — residual chain, above |

The one variant of "earlier issue" not previously priced is a full top-k
speculative demand plan for L+1 issued at L's routed stage (the prediction is
already read back there), with the mispredicted delta fetched as a demand
fixup at L+1. It differs from the ring by pinning real cache slots for all
top-k predictions instead of filling capped ring slots. The ring's recorded
alternatives are the same idea with a budget; their losses were capacity
(evicting future hits for wrong guesses), not bandwidth — the disk has
headroom. At 12G the cache is the scarce resource, so the prior stands, but
the experiment is cheap if #149 shows bandwidth headroom converting into
exposed-wait reduction.

## Verdict

The classic double-buffer — two layers of GPU work in flight — is
structurally unavailable within a token: the residual chain leaves nothing
independent for the GPU to run while the expert event is pending. The
§6C-shaped gains have already been harvested piecemeal (immediate submission,
event-driven I/O with no host wait, phase-1 hit/fixup overlap, the one-deep
prefetch ring), and the deeper-speculation arms are measured losses.

**#150 closes as analysed, not implemented.** The exposed wait is priced at
~84 ms/token; the levers that move it are #147 (fewer bytes) and #149 (faster
bytes). Cross-token overlap, the only remaining source of independent work,
is #151's subject.
