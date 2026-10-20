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

**Read and decode do not contend.** The same benchmark reads the layer's 415 MB
with `F_NOCACHE` while the 8-worker decode runs: 85 ms total, which is the
decode-only time, with the read finishing 25 ms into it (the pages were resident,
so that read ran at 20 GB/s; the engine's own stats put the real uncached rate at
~3.4 GB/s). The pipeline bound is `max(read, decode)`, not their sum — so at the
real rate the projected 204 MB read (~60 ms a token) dominates the 8-worker decode
(~38 ms), and the byte saving is what a correct pipeline collects.

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

## What was built, and what it measured

The pipeline above is now built and measured; it is still not a win:

| step | compressed tok/s (16-token run) | A/B vs raw |
| --- | ---: | ---: |
| one shared scratch buffer behind a lock (first build) | 3.2 | -42.5% |
| per-decode buffers, parallel plan decode | 4.2 | -- |
| v3 block-parallel frames (256 KiB, 8 blocks/expert) | 4.7 | -- |
| phase-1 publish wired into the async plan (`io_hidden` 9% -> 99.5%) | 4.9 | -- |
| prefetch ring issued before the demand fetch, own reader, depth 8 | 5.1 | -38.8% |
| concurrent front/tail reads on separate readers | 5.2 | **-40.7%** |

The last A/B: raw 16.62 vs compressed 9.85 tok/s over three paired rounds,
identical text. The overlap step is inside the noise of the machine drifting up
across the session (raw also moved 15.87 -> 16.62), so the pipeline recovered
~24% of the compressed arm's speed and hid the I/O, but the ratio did not move.

**Why no amount of scheduling closes it.** One layer of raw experts is a 1.65 ms
DMA read that the CPU does not touch, so it hides completely. One layer of
compressed experts is ~1.5 ms of the same kind of read *plus* ~0.5-0.9 ms of CPU
decode; overlap the two and the critical path is still ~1.5 ms, and the layer has
1.6-1.8 ms. There is no time to win: the 9% byte saving is paid back as CPU work
that the same orchestration thread would otherwise spend keeping the GPU fed.
Prefetching cannot fix it either -- a prefetched layer still costs the same ~2 ms
of read-plus-decode, which is not shorter than the raw read it replaces.

**The wall is the per-layer window, not the codec or the scheduler.** The raw
path's read is 1.65 ms per layer and fits inside the 1.8 ms the layer takes, so
it is hidden completely (`io_hidden_pct=100`). The compressed wave costs ~1.5 ms
of read plus ~0.9 ms of decode; even with the phase split, the read and decode
are serial inside one wave, so the layer sees ~2.4 ms of work against a 1.8 ms
window. Decoding is not the problem any more (10 blocks of a wave decode in
0.55 ms, 5.9 GB/s); the *load* is 0.7 ms too long, and the prefetch ring has
exactly one layer of lead -- 1.8 ms -- to hide a 2.4 ms job.

**What the literature says the fix is.** ZipMoE (ICML 2026) makes the same
observation and wins on *high-precision* weights: it splits BF16 into
low-entropy exponent chunks (lossless, 68-74% of the original) and hides
parallel CPU decompression behind I/O with a per-layer DAG; its baselines are
I/O-stalled offloaders, not an engine that already hides I/O. FlashMoE reports
decompression at 41 GB/s on the GPU ("essentially free"). Both point the same
way: decompression has to leave the critical path entirely -- on the GPU, or
behind a window much larger than one layer's 1.8 ms. For a 4-bit install the
prize is only 8.4% of the expert bytes, so a CPU decoder that must share the
window with the read cannot pay for itself.

## The 8-bit install, and why the target cannot be met on this matrix

Converted Ornith 8-bit to the same v3 format: **34,225,520,640 -> 32,630,820,322**
bytes, only **95.3%** -- a 4.7% saving, less than half the 4-bit install's 9%.
8-bit affine weights have more entropy per byte. Measured decode: raw 5.87 tok/s
against compressed 3.96 (-33%) on 16 tokens, so the wider 5.8 ms layer window
does not rescue it either.

That leaves the ceiling arithmetic, which is what actually closes this:

| install | byte saving | read share of the token | ceiling if *all* decode were hidden |
| --- | ---: | ---: | ---: |
| Ornith 4-bit | 9.0% | ~80% | ~7.2% (needs a load <= 1.65 ms; it is 2.4) |
| Ornith 8-bit | 4.7% | ~56% | ~2.6% |
| Qwen3.8 4-bit | ~8.5% | ~50% | ~4.3% |

A 4-bit install can clear +5% only if the load fits the window entirely; the
8-bit and 125B cannot clear it even then, because their expert read is a smaller
share of the token and (for 8-bit) there are fewer redundant bytes. The 125B was
not converted: it needs ~150 GB and the volume has 112 GB free.

## What the literature says the fix is -- and what it costs

ZipMoE (ICML 2026) makes the same observation and wins on *high-precision*
weights: it splits BF16 into low-entropy exponent chunks (lossless, 68-74% of
the original) and hides parallel CPU decompression behind I/O with a per-layer
DAG; its baselines are I/O-stalled offloaders, not an engine that already hides
I/O. FlashMoE reports decompression at 41 GB/s on the GPU ("essentially free").
Both point the same way: decompression has to leave the critical path entirely --
on the GPU -- and the model has to have 25-30% of redundant bytes to win, which
BF16 does and a 4-bit install does not.

## MTLIO compressed reads: verified, and slower than our own decoder

Apple exposes LZFSE containers through Metal I/O: `MTLIOCreateCompressionContext`
writes one, and `newIOHandleWithURL:compressionMethod:` reads it while the driver
decompresses. This was the obvious GPU-side hope, and the addressing works: a
4 MiB container read at offsets 0, 64 KiB, 1 MiB and 1 MiB+1000 returned the exact
source bytes, i.e. **reads are addressed in uncompressed space**, so a layer could
be a drop-in container and the existing expert offsets would still apply.

The throughput kills it. Loading eight expert-sized ranges (13.0 MB) from a
64 KiB-chunk container took **15.2 ms**, and splitting them across eight command
buffers changed nothing (15.5 ms): the driver decodes at ~0.85 GB/s, on one
thread. A 1 MiB chunk size was worse (23.3 ms). Our own block decoder reaches
**5.6 GB/s on eight workers**, so MTLIO's path would be roughly six times slower
than the CPU pipeline it was meant to replace. MTLIO compression is an asset
pipeline, not a decode accelerator.

With that measured, there is no public GPU decompression path left: the
GPU-decode route would mean writing an LZFSE (or rANS) decoder in Metal, which is
a project in its own right and would still be capped by the byte saving.

## A GPU-decodable codec: 9.8% at 6 GB/s

The online path was never going to work, so the codec was redesigned around the
GPU. The result is `benchmark/gpu_rans_prototype.swift`, measured on Ornith 4-bit
layer 20:

- **ratio 0.9017-0.9020** (9.8-9.83% smaller), decoded **bit-exact** on both CPU
  and GPU, 162 blocks checked byte for byte.
- **6.0-6.1 GB/s** on the GPU at 9+ experts; a 3-expert wave (5.3 MB) decodes in
  1.2-2.0 ms, against the 1.6 ms a 4-bit layer takes. 8-bit (5.8 ms window) and
  125B (7.6 ms) have ample room.

**Why this codec.** Two measurements decide it. The 4-bit weight bytes are 89% of
an expert and sit at 7.48 bits/byte -- an entropy bound of 0.936 that no codec
beats. The BF16 scales/biases are the other 11%: interleaved, a 64 KiB pair
measures 5.58 bits/byte (0.698), but its **byte planes** measure 7.89 (mantissa)
and **1.36** (exponent), together 0.579. That is ZipMoE's bit-field observation,
and it is where the last 2 points come from. Splitting the planes is free: the
decoder writes every other byte.

**The format** is one static order-0 model per tensor role (gate/up/down weights,
and each mantissa/exponent plane), 14-bit normalized frequencies, and 32-256
interleaved streams per block so a GPU thread decodes ~2 KiB. rANS has no
context, so streams are independent; each thread runs its own state machine and
the slot -> (symbol, frequency, cumulative) table is one 8-byte load per symbol.
A raw fallback for an incompressible plane would recover the last 0.2%.

**The install exists.** `TinyTitanRepack --compress-experts --codec rans` writes
each layer as a rANS payload plus a `.ridx` sidecar (one model per block index,
one block map per expert, blocks back to back so the runtime needs one read an
expert), `layout.json` says `"compression": "rans"`, the manifest lists the
sidecars, and `--verify-install` certifies them. Converting Ornith 4-bit:

| | |
| --- | ---: |
| expert payload, 40 layers | 18.119 -> **16.036 GB** |
| sidecars | 3.99 MB |
| **ratio** | **0.8852 (11.48% smaller)** |
| GPU check, layers 0/20/39 | 0 mismatched bytes of 5.3 MB each |
| GPU decode, 3-expert wave | 4.0-4.3 GB/s, 1.23-1.33 ms |
| GPU decode, 9 experts | 6.7 GB/s, 2.39 ms |

The convert is 11.5% rather than the prototype's 9.8% because the model is fitted
over all 256 experts of a layer and each BF16 tensor gets its own pair of planes.
The 3-expert wave decodes inside the 1.6-1.8 ms layer, which is the number the
engine needs.

**The runtime path exists and is bit-exact on all three MoE installs, and is
still a loss.** The streamer decodes rANS layers on the GPU, the prediction ring
now decodes predicted experts into its own buffers a layer ahead (adoption 0 ->
32 experts/token, hit rate 0.40 -> 0.48 on 4-bit), and layer open reads its
stream length tables in parallel (prefill 17.4 -> 5.8 s on 4-bit).

| install | expert payload | ratio | saving | raw tok/s | rANS tok/s | output |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Ornith 4-bit | 18.12 -> 16.04 GB | 0.8852 | 11.48% | 6.32 | 2.86 | identical |
| Ornith 8-bit | 34.23 -> 32.00 GB | 0.9350 | 6.50% | 6.56 | 2.20 | identical |
| Qwen3.8 4-bit | 68.05 -> 60.26 GB | 0.8856 | 11.44% | 5.18 | 1.40 | identical |

(16-token runs, `--temperature 0`; the rANS rows use ring depth 8. GPU check on
layer 20 of each: 0 mismatched bytes of 5.3 / 10.0 / 8.3 MB.) The three converted
installs were removed after measurement to reclaim disk; they are reproducible
from the raw installs with `TinyTitanRepack --compress-experts --codec rans`.

The cause is measured and it is not the codec: `waitUntilCompleted` on our
compute costs **8-10 ms for phase 0 and 14-18 ms for phase 1 in-engine**, while
the same kernel is 1.5 ms in isolation. The engine's GEMM pipeline keeps the
device saturated, and a decode submitted from another queue is charged that
backlog; the prefetch ring has one layer (~1.6 ms) of lead, which is less than
the 8 ms the device takes to run the kernel. Moving the decode onto the GPU was
the right direction but this engine already spends its GPU budget on the GEMMs.

What that leaves: the decode must not occupy the same GPU as the GEMM, which
means either a second GPU engine/queue priority that Metal does not expose, or
decoding during an idle window the pipeline does not have. On this engine the
measured conclusion is that a GPU-decoded install is a footprint feature (11.5%
smaller on the 4-bit and 125B, 6.5% on 8-bit), not a speed one.

## Status

The v3 format, the block decoder, the phase publish and the separate prefetch
reader are committed as the pipeline's foundation. The target of +5% decode is
**not met**: -40.7% on Ornith 4-bit and -33% on 8-bit. The measured ceiling says
the CPU pipeline cannot reach it on the 8-bit or 125B installs at all, and on
Ornith 4-bit only by getting the whole per-layer load under the 1.65 ms the raw
read takes. The research-backed route is GPU-side decompression, which is a new
project and would still be capped by the 9% byte saving.
