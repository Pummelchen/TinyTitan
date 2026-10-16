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
| Fewer bytes: raise the expert hit rate (budget axis) | **Closed** — #147 measured, and §5z priced the shape: at shipped defaults the derived rung is the best runnable configuration on this 24 GB machine, the binding constraint is physical working set not budget policy, and every static partition of a fixed tier is worth ≤ ~1 point (§5o, §5p, §5z) |
| Faster bytes: deeper queue depth, coalescing, layout | **Closed** — #149 measured the read path at its own wall: io_ms 102.0/token for ~403 MB ≈ **3.95 GB/s** against the 3.92 GB/s four-thread ceiling, with `io_queue_ms` **0.41** — no queueing to buy back (§5d: 8–16 workers saturate at ~3.7, so depth buys nothing; §5m: coalescing priced at the 10–12 ms/token it could recover ≈ 2–3%, parked) |
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

### Re-read 2026-10-16: both rows above said "Open", and the condition just stated is now answered

Two corrections, because as written this note pointed the next session at a
closed lever.

- **The "~1.4 GB/s effective against a multi-GB/s device" premise was real and is
  not wrong — but it does not mean what the row inferred.** §5v sampled `disk0`
  at one-second resolution across a whole decode run and got a mean of
  **1 359.5 MB/s** (min 753.0, **0 idle samples**) — that is the device at
  **34.7 % duty**. Meanwhile §5o's runner line measures the engine's *own read
  path* at **3.95 GB/s against its 3.92 GB/s ceiling** with `io_queue_ms` 0.41.
  The two figures are not in tension: the read path runs flat out when it is
  asked to run, and the drive is idle between bursts because the bytes it would
  read in that gap are not known yet. That is exactly §5w's finding, and it is
  the reason the operator's continuous-feeder premise reads as a bandwidth
  problem and measures as a dependency-chain one. The headroom depth could
  reach is 3.9 % of *read throughput*, not of clock — §5d's worker probe is
  1 / 2 / 4 / 8 / 16 workers = 2.01 / 2.84 / **3.57** / 3.71 / 3.69 GB/s, so the
  engine's 4-worker reader sits 3.9 % under its own saturation point — and the
  "read more sequentially" shape the same probe contrasts is worse, not better:
  `dd` over a whole layer file measures 3.15 GB/s and 2.61 GB/s, *slower* than
  the strided parallel reader.
- **The L+1 top-k pin described above was priced, and it is worth less than the
  bar.** §5aa arm 4 is the oracle form of exactly this idea — protect what this
  session routes in its next W tokens, reserving no slot — and W=1 (the whole of
  what an L+1 pin can own, since MTP is off on the delivered path and the verify
  pass is width 2) buys **+0.80 / +0.95 / +1.19 points** of hit rate across three
  captures. The +7-point bar sits at **W=32**. §5aa's mechanism sentence covers
  the rest: inside a fixed tier eviction is zero-sum, and even a protection list
  learned with hindsight at zero slot cost tops out at +2.91…+3.44 points.

So the four-way table is complete: none of the four has an arm left to run on
this machine at this budget. What remains open in the campaign is the victim
rule itself (#177, the saturating use-count cap — §5aa's zero-sum finding is
about *protection lists*, not about the shipped rule's own count saturation) and
the RAM ceiling, and both are outside what this note was ever about.

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

Both of those two were measured the same day and closed — see the re-read
above — so this note's remaining contribution is the structural result: the
residual chain leaves no independent GPU work to schedule.
