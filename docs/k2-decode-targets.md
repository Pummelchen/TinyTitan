# K2 decode targets: >12 tok/s at 4-bit, >6 tok/s at 8-bit

Status: **4-bit is at the line, 8-bit is hardware-bound.** Measured on the base
M3 / 24 GB host with `TinyTitanBench k2gpu`, greedy, 32 decode tokens, warm
expert cache, `TINYTITAN_K2_REPEAT` (the repeat is the same prompt and token
stream, so it is the upper bound, not a serving mix).

| width | steady (runs 1-2) | best run | target |
| --- | ---: | ---: | ---: |
| 4-bit | 11.6-11.8 tok/s | 12.0-12.1 | > 12 |
| 8-bit | 1.7 tok/s | 1.9 | > 6 |

## Where a 4-bit decode token goes

`TINYTITAN_K2_TIMING=1`, 16-token runs, per token:

| part | seconds | what it is |
| --- | ---: | --- |
| busy | 0.048 | GPU: phase 1 0.032 (norm, q/k/gate, MoVA+values, RoPE, attention, gate, o_proj, shared expert) and the routed pass 0.016 |
| other | 0.028 | GPU idle between the 45 sparse layers' command buffers while the CPU feeds the next one |
| fetch | 0.002 | routed-expert loads — the 64-slot cache holds the stream's experts, so it is all hits |
| encode | 0.003 | CPU encoding of the kernels |

So the target is not arithmetic: the GPU is busy 48 ms of an 82 ms token and
idle for 28 ms. Any 4-bit win has to come from either fewer synchronisation
points per token or cheaper per-layer kernels.

### Tried

- **One command buffer per layer instead of two** (routed pass + the next
  layer's phase 1). Measured *slower*: 10.7 tok/s against 11.6 over 16-token
  warm runs. The gap only moves: the CPU still has to wait for the router ids,
  fetch, and encode the next routed pass. Reverted.
- **Synchronous all-hit expert path** — when the plan reports no misses, read
  the cache slots directly instead of going through the async load operation
  (which is a no-op for a hit but still suspends). Best runs went 12.0 → 12.1,
  within noise but free, and it keeps the token path's output byte-identical.
  Kept.

## Why 8-bit cannot reach 6 tok/s on this host

A decode token needs top-8 × 48 layers = 384 expert reads. At 5.4 MB per 8-bit
expert that is 2.1 GB/token if nothing is cached; the measured fetch is 0.33 s
of a 0.5 s token at the 16-slot cache (~4 GB), and the GPU itself is 0.10 s
(busy) plus 0.07 s of idle. Even with a *perfect* cache the GPU+idle floor is
~0.17 s/token = 5.9 tok/s, and the full expert set is 26 GB — it does not fit
beside the 11.6 GB of resident weights on a 24 GB machine.

Raising the slot count confirms the wall: 24 slots measured 0.71 tok/s, 32 and
40 measured 0.19-0.22 tok/s with 3 s of "other" per token — the machine swaps.
So 6 tok/s at 8-bit needs either a smaller expert footprint (a quality change,
rejected) or different hardware (≥48 GB unified memory).

## Next levers for 4-bit

1. **Fuse the per-layer dispatches.** q/k/gate are three GEMVs over the same
   `normed` row; RoPE is two dispatches; the k/v cache writes are two blit
   encoders. Cutting ~4 of ~18 dispatches per layer is ~2-3 ms/token, which is
   the gap to a steady 12.5.
2. **Cheaper attention for short contexts.** The full prefill attention kernel
   is dispatched once per layer regardless of `seqLen`; a decode-shaped variant
   may shave the 0.032 s phase.
3. **Fewer synchronisation points** without merging buffers: speculative
   prefetch of the next layer's experts from the routing profile while the
   current layer's routed pass runs, so the fetch and the ids read-back are
   already done when the layer completes.
