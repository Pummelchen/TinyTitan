# Compression for TinyTitan's models: what is optimal, and why the first attempt lost

Status: design note, 2026-10-19. The compressed format and converter exist; the
pipeline described at the end does **not**, so today's compressed installs are
still slower than raw and no install should be converted for speed.

## The short answer

The technique is viable; the implementation that was measured was not. On the real
expert frames this machine decodes LZFSE at **5.61 GB/s across 8 workers** (5.5x
from one core), and one layer's miss set decodes in **~1.5 ms** inside a layer
that takes **1.8 ms** (Ornith 4-bit) to **7.6 ms** (Qwen3.8 4-bit). The A/B's
-42.5% is a *pipeline* result: the compressed path serialised read and decode,
ran its decode at effectively one core, and dropped the phase-1 overlap the raw
path uses. A correctly pipelined version has a ceiling of the byte saving itself:

| install | expert bytes/token | saving | token | ceiling |
| --- | ---: | ---: | ---: | ---: |
| Ornith 4-bit | 224 -> 204 MB | 8.4% | 73 ms | **+7.7%** |
| Ornith 8-bit | 432 -> 393 MB | 8.4% | 231 ms | **+5.0%** |
| Qwen3.8 4-bit | 626 -> 570 MB | 8.4% | 362 ms | **+4.5%** |

Those are computed from measured miss rates and measured decode throughput, not
achieved: they are the ceiling for a pipeline that decodes entirely behind the
GPU. The original campaign estimate of 3-5% was therefore right about the prize
and wrong about it being automatic.

## Measured decode scaling

`benchmark/expert_decode_bench.swift`, Ornith 4-bit layer 20 (512 frames,
415 MB packed -> 452 MB raw), min of three passes, every frame verified:

| workers | output GB/s | input GB/s | full-layer pass |
| ---: | ---: | ---: | ---: |
| 1 | 1.02 | 0.94 | 444 ms |
| 2 | 1.98 | 1.82 | 228 ms |
| 4 | 3.76 | 3.45 | 120 ms |
| 8 | **5.61** | **5.14** | **81 ms** |

Layer-sized waves (the unit the router produces), same benchmark:

| experts decoded | 4 workers | 8 workers |
| ---: | ---: | ---: |
| 3 (6 frames) | 1.81 ms | **1.47 ms** |
| 8 (16 frames) | 3.66 ms | 2.67 ms |
| 10 (20 frames) | 4.82 ms | 3.98 ms |

Two facts follow. First, a single LZFSE stream is serial, so the only parallelism
is across frames — the format's independent per-phase frames are what make the
8-worker number reachable at all. Second, the wave must be about a layer's miss
count (3-6 here), not one expert at a time, or dispatch overhead dominates.

## The budget model

Per layer, with `m` misses, stride `s`, ratio `r = 0.911`:

- raw read: `m·s / 3.4 GB/s`
- compressed read: `m·s·r / 3.4 GB/s`
- decode: `m·s·r / 5.61 GB/s` (8 workers)

The win is the read difference; the condition is that decode fits inside the
layer and overlaps the GPU. Measured miss rates (this session, 8-16 tokens per
install, `TINYTITAN_RUNNER_STATS=1`) and the resulting budgets:

| | Ornith 4-bit | Ornith 8-bit | Qwen3.8 4-bit |
| --- | ---: | ---: | ---: |
| stride | 1.77 MB | 3.34 MB | 2.77 MB |
| top-k / layers | 8 / 40 | 8 / 40 | 10 / 48 |
| hit rate | 0.606 | 0.501 | 0.326 |
| misses/token | 126 | 129 | 226 |
| packed per layer | 5.1 MB | 9.8 MB | 11.9 MB |
| read, raw -> packed | 1.64 -> 1.50 ms | 3.18 -> 2.89 ms | 3.83 -> 3.49 ms |
| decode, one wave (8 workers) | 1.47 ms | ~2.2 ms (est.) | ~2.8 ms (est.) |
| layer wall | 1.82 ms | 5.77 ms | 7.55 ms |

Decode fits in every case, with 0.35 ms to 4.8 ms of slack (only Ornith 4-bit is
measured for decode; the other two scale by bytes from it). Ornith 4-bit is the
tight one: 1.5 ms of packed read plus 1.5 ms of decode is already 1.82 ms of
layer, so read and decode must overlap *and* the decode must ride behind the
preceding GPU work. 8-bit and 125B are comfortable.

## Codec: LZFSE is right, "parallel deflate" is not

| codec | ratio | decode/core | 8 workers | fits under a 3.4 GB/s read? |
| --- | ---: | ---: | ---: | --- |
| LZFSE | 91.1% | 1.02 GB/s | **5.6 GB/s** | yes, 1.6x headroom |
| zstd -1 | 90.8% | 1.11 GB/s | ~6.7 GB/s (est.) | yes, but a new dependency |
| zlib (deflate) | 91.1% | 0.27 GB/s | ~2.2 GB/s | **no** — slower than the disk |
| lz4 | 98.7% | ~20 GB/s | — | yes, but saves only 1.3% |

Deflate parallelises fine; the problem is throughput per core. The decoder has to
keep up with a 3.4 GB/s read stream, and zlib's 0.27 GB/s/core does not reach it
even at eight workers, so its decode can never hide. LZFSE clears the bar with
1.6x headroom and needs no dependency; zstd would buy 0.9 ratio points and more
headroom for an external dependency, which is a separate decision.

## Why the first attempt lost, in order of impact

1. **The phase-1 overlap was dropped.** The raw path reads gate|up, starts the
   GPU on it, and fetches the tail while phase 1 runs (`firstStage` /
   `publishesFirstStage`). The compressed path decoded *both* frames before
   returning, so the layer could not start until the whole expert was decoded.
2. **Read and decode were serialised per batch.** One `fetch` of ~4 experts, then
   a 4-worker decode, then the next batch. No double buffering, so the two costs
   added instead of overlapping.
3. **The wave was too small.** 4-expert waves at 4 workers; the measured 8-expert
   decode at 4 workers is 3.66 ms, at 8 workers 2.67 ms.
4. **The ring could not prefetch.** One slot, and its decode competed with the
   demand path; adoption fell from 21.75 to 0.7 per token, so every miss was paid
   synchronously.
5. **That also inverted the saving.** With no prefetch filling slots, misses rose
   from 111 to 126 per token and the compressed install read *more* bytes (204 MB)
   than the raw one (196 MB).

Net effect: ~204 MB of decode per token at ~1 GB/s effective, i.e. ~200 ms
against a ~73 ms token — which is what the A/B measured.

## The optimal shape

For the three MoE installs TinyTitan supports, and only for routed experts:

1. **Format.** One frame per phase (two per expert when the layer splits), which
   is what `ExpertBlobIndex` already addresses. Frames of 0.5-2 MB keep the ratio
   and give one wave per layer. A single per-layer stream would be undecodable in
   parallel and is the wrong shape.
2. **Scheduling.** Decode the layer's misses in one 8-worker wave; decode the
   front frames first, publish phase 1, decode the tails while gate|up runs. Read
   the next batch while the current one decodes (double buffer). The GPU is idle
   on the CPU during its own work; that is the budget the decode spends.
3. **Prefetch.** Optional once demand-side overlap works; if used, the ring must
   hold a layer's worth of experts (8 slots, ~13 MB at 4-bit) and decode a whole
   predicted set per wave. A one-slot ring cannot hide anything.
4. **Admission.** Routed experts only. The resident dense weights are
   RAM-resident and read every token (a decode cost with no read to save); the
   125B n-gram table is 71-77% compressible but that is a **footprint** win
   (25-30 GB of 96 GB) and needs block addressing, not a speed feature.

## The experiment that settles it

A bounded change, not another format:

1. In `executeCompressedMisses`, take the plan's `firstStage` callback and decode
   front frames -> publish -> tails, the way `executeBoundedReads` does.
2. Size the wave to the layer's misses (3-6) and decode it with 8 workers.
3. Read batch N+1 while batch N decodes.
4. A/B with `benchmark/compressed_experts_ab.py`; keep the existing conversion.

Success criteria: `io_hidden_pct >= 95`, expert misses back to the raw arm's count
(no ring regression), and decode at least par with raw. If it lands, the expected
result is +4-8% depending on install; if it does not, the per-layer budget above
says exactly which of read, decode, or latency-gap ate the saving.

## Status

The pipeline above is not built. The converter and the `.idx` read path stay as
opt-in, reproducible tools; today's honest number is the A/B's -42.5%, and this
note explains why that number is about the implementation rather than the idea.
