# Performance campaign, 2026-10-18 22:00 → 2026-10-19 08:00

Machine and scope: one 24 GiB M3 MacBook Pro, Ornith 1.5 35B-A3B in 4-bit and
8-bit, decode tok/s from the engine's own `decode_tok_s` line. Driver:
`benchmark/minor_wins_ab.py` (paired, interleaved, rotating arm order, discarded
warm-up run, `--max-tokens 128`, greedy, output sha256 compared across arms).

**Bar for a commit:** ≥ +0.5% median on **both** bit widths, with the counters
consistent. Anything below that is recorded here as tested and not shipped —
a number, not an opinion.

## Every lever tried, and what it actually gave

Measured on Ornith 35B-A3B, 4-bit and 8-bit, on the 24 GiB M3. "n/m" is positive
rounds out of paired rounds. A blank cell was not measured.

| lever | 4-bit | 8-bit | status |
| --- | ---: | ---: | --- |
| **Disk prompt cache, session resume** | **32.25 -> 2.48 s (13x)** | **56.31 -> 4.07 s (13.8x)** | **works; opt-in for privacy** |
| Multi-turn prompt cache (already shipped) | 32.16 -> 1.33 s | not measured | already the default behaviour |
| Page-cache path (`BOUNDED_IO=0`) | +0.5% (2/4), −2.1% (1/6), **+1.7% under 4 sequences** | **+4.47% (6/6)** | not shipped: trades a declared memory budget for an undeclared one |
| Concurrency width 4 | +4.3% fair | — | **reverted**: clamps to 1 at 262K context, and the apparent 1.32x was cold-cache bias |
| Read advice off | −4.7% (1/4) | −3.2% (1/3) | loss on both |
| Immediate submission | −6.5% (0/4) | +0.1% (2/3) | loss / wash |
| Per-layer I/O timeline | +0.3% (2/3), +1.7% under 4 seq | −1.1% (0/3) | no win on both |
| Expert slots 128 / 64 | −0.6% / −0.3% | −20.7% / −8.6% | the derived count is the optimum |
| KV cache 16-bit / 4-bit | −0.8% / −2.1%, output differs | not measured | loss, and not output-invariant |
| Metal I/O backend | −0.7% (0/3) | not measured | loss |
| Prefill chunk 512 / 1024 / 2048 (base 4096) | +23% / +9% / 0% prefill | not measured | 4096 is already optimal |
| Expert use cap 3 | −0.4% (0/3) | −0.6% (0/3) | wash |
| GPU keep-warm 500 / 2000 us | −0.5% (1/3) | not measured | wash |
| No pinning | −0.8% (0/3) | not measured | pinning pays |
| Max context 8K | −0.5% (1/3) | not measured | neutral |
| Prefetch ring **off** (control) | −8.1% (0/3) | not measured | confirms the shipped default is worth 8-17% |
| `sync=event` (control) | −29.2% (0/3) | not measured | confirms the shipped sync is right |
| Startup to first answer | 4.7 s | not measured | no problem (the 90-125 s I chased was a buffered-log artifact) |
| Attention fast path **off** (`ATTN_SIMD_PARTIAL=0`) | −0.02% (1/3) | not measured | the fast path is worth having |
| Router fast path **off** (`ROUTER_TOPK_SIMD=0`) | −0.02% (1/3) | not measured | the fast path is worth having |
| Sampler `generic` instead of `tiled` | −1.52% (1/3) | not measured | tiled is right |
| GDN in-proj `r16` / `xsh16` | +0.16% / +0.33% | not measured | flat — the historical +1.6% on Qwen3.8 does not reproduce |

Those last four close the configuration space: with them measured, every
`TINYTITAN_*` knob that can act on this GPU decode path (94 exist; 59 are
memory-service, tracing, other engines, or Qwen/MTP/ANE-only) has either been
measured here or closed by an earlier record.

**A measurement fix came out of it.** The six-arm screen first read −4 to −6% for
*every* arm, because each round starts a fresh server and the arm that runs first
pays a cold expert cache while the rest do not — with six arms and three rounds
the rotation does not cancel that. `--per-run-warmup` sends one discarded request
per run before the measured one, and the effects then land at ±0.3%. Any earlier
sweep with many arms and few rounds carries the same bias; the two-arm sweeps are
safe because each arm occupies each position equally.

**Net shipped: nothing.** The one candidate that looked like a win was reverted
after measurement showed it was an artifact, and the two real levers (disk
prompt cache, page cache) are policy decisions rather than code changes.

## Why the protocol looks like this

Three properties of this machine decide the method, and they are measured rather
than assumed:

- The **first server of a session is ~16% slower** than the ones after it (cold
  page cache over the 3.4 GB of weights), so every sweep starts with a discarded
  warm-up run and rotates which arm goes first each round.
- Round-to-round swings on the **base arm itself** reach 8-14% on 4-bit (the
  4-bit stack sweep's base ranged 14.13-17.46 tok/s), so a single A/B is
  worthless and every result below is a paired medians over 3-6 rounds.
- Python's `hash()` is salted per process, so text identity is checked with
  sha256. Every arm recorded here produced **byte-identical output** unless the
  table says otherwise.

## Results

| experiment | 4-bit | 8-bit | verdict |
| --- | ---: | ---: | --- |
| Page-cache path (`TINYTITAN_BOUNDED_IO=0`) | +0.5% (2/4), −0.6% (0/3), **−2.1% (1/6)** | **+5.2%** (2/3) | fails the 4-bit leg conclusively |
| Read advice off | −4.7% (1/4) | −3.2% (1/3) | loss on both |
| Immediate submission | −6.5% (0/4) | +0.1% (2/3) | loss at 4-bit, wash at 8-bit |
| Stack: page cache + immediate | −0.7% (2/6) | **+4.7%** (5/6) | fails the 4-bit leg |
| Metal I/O backend | −0.7% (0/3) | not run | loss |
| Per-layer I/O timeline | +0.3% (2/3) | not run | below the bar |
| GPU keep-warm, 500 us | −0.5% (1/3) | not run | wash |
| `gpu-residency` expert execution | 0.0% (1/3) | not run | wash |
| Expert use cap 3 | −0.4% (0/3) | −0.6% (0/3) | wash / slight loss |
| Expert slots 128 (derived is 96) | not run | **−20.7%** (0/3) | far worse — the derivation is the optimum |
| Expert slots 64 | not run | −8.6% (0/3) | worse |
| Expert slots 128 / 192 at 4-bit (derived 160) | −0.6% / −0.3% (0/3, 1/3) | — | the derivation is the optimum on both widths |
| Per-layer timeline | +0.3% (2/3) | −1.1% (0/3) | no win either width |
| `sync=event` | **−29.2%** (0/3) | not run | the comparison path, not the fast one |
| Max context 8192 (default 262144) | −0.5% (1/3) | not run | neutral — the KV reservation is already out of the slot budget |
| No pinning (`TINYTITAN_NO_PIN=1`) | −0.8% (0/3) | not run | pinning earns its 0.003 ms of preamble |
| GPU keep-warm 2000 us | −0.5% (1/3) | not run | wash |
| Expert use-count reset at handover | not run | not run | measured in-source as identical: first-64-token hit 68/82/87% with *and* without (`RealForwardRunner+Prefill.swift:16-22`) |
| KV cache 16-bit | −0.8% (1/3), **output differs** | not run | loss, and not output-invariant |
| KV cache 4-bit | −2.1% (0/3), **output differs** | not run | loss, and not output-invariant |
| Prefetch ring **off** (default is on) | **−8.1%** (0/3) | not run | not a candidate — a control that confirms the shipped default is worth 8-17% |
| Long prefill (2.2K), per-layer timeline | +1.87% (2/2) → **+0.15%** (2/3) | not run | **collapsed under 3 rounds — noise** |
| Long prefill (2.2K), use cap 3 | +1.45% (2/2) → **−0.21%** (2/3) | not run | **collapsed under 3 rounds — noise** |
| Long prefill, per-layer + use cap together | not run | not run | −0.36% (1/3) — no interaction |
| Long prefill, page cache | −0.04% (1/2) | — | nothing at this pressure |
| Prefill chunk 512 / 1024 / 2048 (base is 4096) | −3.5% / −0.3% / +1.2% decode; prefill **39.7 / 35.1 / 32.2 s** vs 32.2 s | not run | default 4096 is already optimal; smaller chunks cost up to +23% prefill |
| Long prefill, advice off | +0.01% (1/2) | — | nothing at this pressure |

Baselines for orientation on this machine: 4-bit ≈ 16.2 tok/s, 8-bit ≈ 9.85
tok/s, 128 output tokens, greedy, short prompt. On a 2.2K-token prompt the same
4-bit install decodes at ≈ 14.0 tok/s and the expert hit rate falls to 0.736 —
that is the coding-agent shape, and it is where the surviving candidates came
from.

The derived expert-cache slot counts differ by bit width on this host: **160**
slots at 4-bit, **96** at 8-bit, both from the same RAM rule.

**A two-round result is not a result on this machine.** The one genuinely useful
lesson so far: both long-prefill candidates looked like +1.5-1.9% at 2/2 and
became +0.15% and −0.21% at 2/3. The bar is three paired rounds before anything
is even discussed for a commit.

**Nothing has cleared the bar yet.** The one real effect (page cache) is
pressure-dependent: it pays 5% at 8-bit, where a token moves twice the expert
bytes, and is flat-to-negative at 4-bit.

## What the counters say about where headroom is

8-bit, one decode token, shipped defaults: `body_ms=91.7`, of which
`wait_ms=45.3` — half the token is the GPU waiting on expert I/O. `head_ms=7.5`,
`cb1_ms=6.1`, `rdadvise_ms=2.8`, `cache_plan_ms=1.8`, `gather_ms=0.0`. So the
only lever with real headroom is expert-traffic scheduling, and the sweep above
says the obvious scheduling knobs are already at their optimum on this model.

## Queue for the rest of the night

1. Long-prefill (2.2K) decode sweep — page cache, per-layer timeline, use cap,
   advice off. The cold-cache handover is the condition the history says pays.
2. Prefill-side (TTFT) knobs, reported separately from decode.
3. `TINYTITAN_EXPERT_IO_SYNC=event` and deeper prefetch at long context.
4. Anything that clears 4-bit then confirmed at 8-bit before commit.

## A second axis: aggregate throughput under concurrency

`--max-concurrent-sequences` defaults to **1**, so a second client waits. Measured
with `benchmark/concurrency_sweep.py`, one server, N clients released together,
128 tokens each, Ornith 4-bit:

| sequences | wall | aggregate | vs 1 seq | engine per-stream rate |
| ---: | ---: | ---: | ---: | --- |
| 1 | 10.7 s | 11.98 tok/s | — | 15.9 |
| 2 | 18.1 s | 14.19 tok/s | **1.18x** | 15.6, 19.8 |
| 4 | 32.3 s | 15.84 tok/s | **1.32x** | 15.7-20.1 |

Two things stand out. Serialised, four requests would take ~43 s; concurrent
they take 32.3 s. And the engine's own per-stream decode rate **rises** under
batching (15.9 → ~20 tok/s), because the batched kernels amortise weight reads
across sequences. The reservation is free on this host: `slots=160` and
`cache=10.55G` are identical with `--max-concurrent-sequences 1` and `4`.

The cost is latency: each client's wall time rises roughly linearly (10.7 s →
32.3 s for 128 tokens at four sequences). This is a throughput lever for a server
with several clients, not a latency lever for one. The worst-case client still
finishes sooner than it would queued: four serialised requests need ~43 s, and
concurrently they all finish by 32.3 s.

### Concurrency: a retraction, and a measurement trap

I shipped a default of four concurrent sequences (`51b180d`), claimed +32%/+24%
aggregate, and **reverted it**. Two things were wrong.

**The width does not build at the default context.** `BatchedMemoryBudget` sizes
each slot for `--max-context`; at the default 262,144 tokens that is 2,782 MiB
per slot against a 2,048 MiB budget, so the load prints
"batch width 4 exceeds the memory budget ... serving 1 at once" and the server
serves one generation at a time exactly as before. At 32K context it builds, and
at 8-bit it does not even then (per-slot 402 MiB against a **0 MiB** budget,
because the expert cache has taken the rest of RAM).

**The aggregate metric was biased by cold-cache amortisation.** Each cell started
a fresh server, so the 1-sequence cell measured one cold request while the
4-sequence cell measured one cold request and three warm ones. The "1.32x" was
the expert cache warming up, not batching. The fair comparison is four sequential
requests on a warm server against four concurrent ones — Ornith 4-bit, 32K
context, the only configuration where the width actually builds:

| arm | wall | aggregate |
| --- | ---: | ---: |
| 4 sequential, one server | 33.0 s | 15.51 tok/s |
| 4 concurrent, width 4 | 31.6 s | 16.18 tok/s |

**+4.3%, which is inside this machine's noise.** The driver now measures this
denominator itself (`--sequential-baseline`), and a second cell puts the fair
ratio at **1.17x** (concurrent 18.42, sequential 15.74) — the range across runs
is 1.04-1.17x, not the 1.37x the biased metric showed. Batching is not a lever on this
engine at this RAM budget; the expert cache already holds the throughput, and
every slot the batcher asks for is a slot the cache cannot have. The default is
back to 1 and the tests are back to pinning it.

### The knee is four, and the page cache helps under it

Raising the width further is not worth it: 8 sequences measure 16.70 tok/s
aggregate against 15.78 at 4 (+5.8%) while each client's wall time doubles
(61.4 s against 32.5 s for 128 tokens). Four is the knee, which is what shipped.

With four sequences in flight the page cache finally pays at 4-bit too:
**16.05 tok/s aggregate against 15.78 (+1.7%)**, consistent across both rounds
(16.04, 16.06). That is the one measurement that makes the footprint trade look
attractive at both widths — and it is still not a default I can flip: it
trades a declared memory budget for an undeclared one, which is a policy call
for the owner, not a benchmark result. It remains available as
`TINYTITAN_BOUNDED_IO=0`.

### The knobs do not behave differently under concurrency

Re-screened at the new default width (four sequences, aggregate tok/s, 4-bit
base 15.78):

| arm | aggregate | vs base | 8-bit | vs base |
| --- | ---: | ---: | ---: | ---: |
| Per-layer I/O timeline | 16.07, 16.05 | **+1.7%** | 8.94, 8.85 | **−1.0%** |
| Use cap 3 | 15.64, 15.74 | −0.5% | not run | — |
| Immediate submission | 15.78, 15.77 | 0.0% | not run | — |

The per-layer timeline looked like the second real lever — four sequences sharing
one completion timeline should make a single slow read gating every layer worse
— and it gains 1.7% at 4-bit. At 8-bit it loses 1.0%, the same split the
single-stream screen found, so it does not clear the bar either. The other two
are flat.

### What is already optimised, and why the rest is not reachable

Two things that look like candidates are already done:

- **Multi-turn prompt cache.** A 1,656-token prompt prefills in 32.16 s, and the
  follow-up turn with the assistant reply and a new question prefills in
  **1.34 s** -- 24x cheaper, because the stored KV covers the conversation
  prefix. The agent-loop shape, which resends a growing transcript, is the case
  this optimises and it works.
- **Prefill expert reads are already minimal.** The routed tile is 8 experts
  wide (capped at 16, `PrefillRoutedTileScheduler`), and 256 experts at 8 per
  tile means each expert's weights are read once per layer: 18.1 GB predicted
  from the geometry against 16.7 GiB measured. There is no re-read to remove.
  Prefill is GPU-bound at 94.5% occupancy; the only lever left there is a faster
  kernel, not a schedule.

What is *not* reachable at this RAM budget is batching, because the KV
reservation costs 2,782 MiB per slot at the 262K default and the batch budget is
2,048 MiB. That is a KV-reservation-policy change, not a knob.

### The disk prompt cache evicts at its budget

Verified functionally rather than by reading the code: a server started with
`--prompt-cache-disk-mib 100` took two ~80 MiB states and kept **one, 73 MiB** --
the older entry was evicted, so the budget is enforced rather than nominal
(`ServerPromptStateStore.saveSync` -> `evictDiskIfNeeded`, LRU by modification
date, and a snapshot larger than the budget is refused outright). The launcher's
default is 2,048 MiB, which holds roughly 25 states of this size.

### Both gates for compressed experts passed

**B1 -- bursty decompression is free too.** The steady churn test could not say
whether the real shape (one expert frame every layer) behaves the same, so
`benchmark/decompress_churn.py` paces two libzstd decompressors at 1.6 MiB every
1.8 ms through ctypes, with no process spawn:
`paced/clean = 1.148x` and `0.969x` across two interleaved rounds -- inside the
noise, on both sides of one.

**B2 -- the ratio generalises to the 125B.** Qwen3.8-Flash-Next 4-bit:
`layer_00` **90.9%**, `layer_10` **91.4%** at `zstd -1` (Ornith measured 90.8-91.4),
and its dense `model_weights.bin` **89.8%** -- better than Ornith's 97.4%, so the
residency-side saving is real on that model too.

**The codec is LZFSE, not zstd**, and the engine needs no new dependency for it.
Measured on the same 128 MiB expert sample:

| codec | ratio | decode | note |
| --- | ---: | ---: | --- |
| `lzfse` (system) | **91.7%** | **896 MiB/s** | zero dependencies |
| `zstd -1` | 90.8% | 1,113 MiB/s | needs a SwiftPM dependency |
| `lz4` (system) | 98.7% | 20,904 MiB/s | 1.3% saving, not worth the plumbing |
| `zlib` (system) | 91.1% | 272 MiB/s | too slow to decode |

LZFSE gives up 0.9 points against zstd and 24% of decode rate, and buys the
project out of a dependency: at ~0.9 GiB/s one core decodes the ~65 MiB a token
reads in ~73 ms, which two threads halve -- the configuration the paced test
measured free. Encoding runs at **205 MiB/s**, so repacking Ornith's 16.9 GiB is
~85 seconds and lands at **15.5 GiB**.

`ExpertBlobCodec` is in the tree with a round-trip test that pins the contract
the streamer will rely on: the geometry says how long a blob must be, and a blob
that decodes to anything else is refused rather than leaving a slot with the
previous expert's weights in its tail.

### The format constraint the plan missed: two frames per expert

The streamer can split an expert read at the `down` tensor's offset, so the GPU
starts on gate|up while the tail is still in flight. Both installed models use
it: `down` sits at **67%** of the stride on Ornith (1,179,648 of 1,769,472) and on
Qwen3.8 (1,843,200 of 2,768,896), so `phase1Bytes` is interior and the split is
live.

A single compressed blob cannot be split — decoding needs the whole frame — so a
compressed layer must store **two frames per expert**, one for gate|up and one
for the tail, each independently decodable, and the index must address both.
`ExpertBlobIndex` currently holds one entry per expert; it needs a second, or a
frame count. That is the next change, and it is the reason the format work is not
a straight swap of `pread` for `decompress`.

The rest of the plan is unchanged: per-expert frames in the repacker,
decompression into the slot buffer behind the ring, and the uncompressed path
kept readable.

### The compression verdict was wrong about the CPU, and the correction matters

The recorded verdict rejected compression on decompression cost. Its arithmetic
assumed the decompressor sits on the critical path *and* that the cores are doing
something else. The second half is false, and the first half is a scheduling
choice rather than a constraint:

- **The CPU is idle during decode.** Measured over a 128-token run: 4.36 s of CPU
  against 11.0 s of wall = **0.40 of 8 cores**, 5% of the machine.
- **A faster codec gives nearly the same ratio.** On a 256 MiB expert sample:
  `zstd -1` **90.8% at 1,113 MiB/s per core**, `-3` 90.3% at 1,067, `-12` 90.1% at
  985, `lz4` 100% at 5,120. The levels barely differ in ratio; level 1 is 13%
  faster than level 12 and is the right one here.
- **At the intensity the model needs, the churn is free.** Six decompressor
  threads (~6.6 GB/s of memory traffic) cost **-29% decode** (11.95 -> 8.34,
  12.48 -> 8.93 tok/s), but one thread (11.35) and two (11.21) are inside the
  11.09 tok/s baseline's noise. The model needs about 1 GB/s of output for
  65 MiB/token at 16 tok/s.

What that is worth: a 4-bit token reads ~65 MiB of experts, so a 90.8% ratio
saves ~6 MiB. At the measured read cost that is **~2-4 ms of a 74-86 ms token,
or roughly 3-5%** -- above the campaign's bar, and it does not need the cores the
host path uses.

Per layer the arithmetic is comfortable: ~1.6 MiB to decompress at 1.1 GB/s is
1.5 ms on one core, ~0.4 ms across four, against a layer that takes ~1.8 ms.
The prefetch ring already gives one layer of lead, so the decompression can ride
behind it rather than in front of the GPU.

This is a real candidate and a project rather than a knob: repack the expert
blobs as zstd frames (17 GiB -> ~15.5), decompress in the streamer's read path
into the slot buffer, keep the uncompressed format readable, then A/B. The one
thing the churn test cannot tell us is whether *bursty per-layer* decompression
is as benign as a steady background stream; that is the first thing the
prototype should measure.

### Compression, re-measured on Ornith (the Qwen verdict was for Qwen)

A user zipped the Ornith 35B 4-bit install and got **20 GB -> 18 GB, ~10%**. That
matches the recorded verdict's 89.56% for `zstd -12`, so the record holds -- but
the per-file breakdown differs enough to be worth writing down, and one file is a
trap:

| file | gzip -1 | note |
| --- | ---: | --- |
| `model_weights.bin` (1.79 GiB, dense) | **97.4%** | 2.6% -- the dense bf16 backbone barely compresses |
| `packed_experts/layer_00.bin` | **73.0%** | **not representative** |
| `packed_experts/layer_07.bin` | 91.4% | typical |
| `packed_experts/layer_20.bin` | 91.4% | typical |
| `packed_experts/layer_39.bin` | 91.2% | typical |

Layer 0 compresses far better than the rest; sampling it alone would have said
"27% is available" when the corpus gives 8.6%. That is the same mistake shape as
the naive-LFU simulation: one sample standing in for a population.

**Why 8.6% still does not go in the read path.** Deflate decompresses at ~490
MiB/s per core and **1,422 MiB/s across four**. A 4-bit decode token reads
~65 MiB of experts, so:

- reading 8.6% fewer bytes saves ~2 ms of a ~16 ms read,
- decompressing those 65 MiB costs **~44 ms even on four cores**,
- and the wait it would be replacing is **35-45 ms**.

It swaps one bottleneck for another **if the decompressor sits in front of the
GPU on the critical path**. The section above shows the cost is scheduling, not
intrinsic: at the intensity the model needs, background decompression measured
free. Where the bytes *are* worth compressing: transport and
install (20 -> 18 GB) and the 102 GB n-gram table (71-77%).

### Compressed expert blobs, built and measured: 13.76 -> 7.91 tok/s

The A1 project is now complete on both sides and measured, and the answer is no.
The format, the writer (`CompressedExpertLayerWriter`), the loader and the
streamer's decode path all exist and are tested; `TinyTitanRepack
--compress-experts` converts an installed raw model into a compressed copy
without re-downloading it. Converting Ornith 4-bit took **94.6 s** for 40 layers
and produced **18,119,393,280 -> 16,504,463,711 bytes** (91.1%, 8.9% saved) plus
246 KB of indices -- exactly the ratio the plan predicted.

**The A/B is what kills it.** `benchmark/compressed_experts_ab.py`, paired, three
measured rounds after one warm-up round, fresh CLI process each time, same
prompt/greedy/128 new tokens, the two installs alternating order and sharing
`model_weights.bin` through a hard link:

| round | raw tok/s | compressed tok/s |
| ---: | ---: | ---: |
| 1 | 14.31 | 8.20 |
| 2 | 13.59 | 7.87 |
| 3 | 13.76 | 7.91 |
| **median** | **13.76** | **7.91** |

**-42.5% decode**, with byte-identical responses in all six runs (LZFSE is
lossless; the format is correct, the speed is not).

**Why the estimate was wrong, in one number.** The plan assumed a token reads
~65 MiB of experts. Measured on this install: 2,654 misses over 21 forward
passes = **126 misses/token x ~1.61 MB packed = ~204 MB/token**, because the
10.55 GB cache holds 4 slots per layer while the router asks for 8. LZFSE
decodes at ~0.9 GB/s per core, so one token's decode is **~226 ms of CPU**
against a ~100 ms token. The churn test that cleared the CPU assumption paced
1.6 MiB every 1.8 ms -- one expert per layer -- and the engine misses four to
six.

Worse, the saving inverted: the raw arm missed 111 experts/token (196 MB), the
compressed arm 126 (204 MB packed). The prefetch ring, which normally fills
slots ahead of demand, could not keep up with decode (adoption 0.7/token against
21.75), so the compressed install read *more* bytes than the raw one. Even with
the hit rates equalised it would have read 8.4% fewer -- about 5 ms of a ~100 ms
token, against ~226 ms of decode CPU that the ring has to hide to get it.

The fixes that were tried, all measured, none close:

- one shared staging buffer behind a lock -> 3.19 tok/s; per-decode buffers from
  a pool, plan misses decoded in parallel -> 4.19;
- batched frame reads through the bounded reader (a per-expert `pread` measured
  8 ms against the raw path's batched 2 ms) -> 3.5-3.6 on 16 tokens;
- the prefetch ring decoding its predictions in parallel -> 4.19, adoption still
  0.7/token against the raw path's 21.75;
- a deeper ring (`TINYTITAN_PREFETCH_LAYER_DEPTH`, depth 8 everywhere) -> 5.39
  tok/s, hit rate 0.81. Still 2.5x behind raw.

**Verdict: keep the install raw.** The `--compress-experts` converter stays as
an opt-in measurement tool and the read path stays because it is inert without
an `.idx` sidecar, but no install should be converted for speed and no default
changes. The deeper-ring idea is not the rescue either: it is the same knob that
measured a loss on the raw path (§5al), so paying for it to make a slower format
less slow is not a trade this engine makes.

### Placement is where the smart reader can still win

The reader is already predictive: it runs the *next* layer's router on the current
residual and issues ranked reads (42.7 issued / 23.3 adopted per token). The
reading side is exhausted -- compression 0% with a codec fast enough (lz4), 89.6%
with one too slow (zstd -12 at 0.96 GB/s/core against a 45 ms wait), and the SSD
runs at 1.4-2 GB/s of 3.6 so a layout permutation has nothing to win.

What had not been measured is **placement**: which experts to *keep*. New
simulator, `benchmark/expert_placement_sim.py`, replays a committed route trace
(48 layers x 521 positions x top-10) against four policies at the shipped 96
slots:

| policy | hit rate |
| --- | ---: |
| naive all-time LFU | 66.89% |
| LRU | 80.13% |
| oracle (Belady) | **88.75%** |
| static hot set, 48 pins, learned in-sample | 84.61% |
| static hot set, 48 pins, **learned on another session** | **83.28%** |
| static hot set, 72 pins, learned on another session | **85.05%** |

Read against **LRU**, not against the naive LFU, because the shipped policy is
LFU *with a count reset at the prefill-to-decode handover* and a recency tie
break -- the reset is exactly what stops the stale-popularity pathology that
makes plain LFU look 13 points worse here. Against LRU the oracle gap is 8.6
points, and a static per-layer hot set learned from a **different session**
recovers **37%** of it at 48 pins and **57%** at 72 pins (+3.2 and +4.9 points).

Cross-session transfer is the point: the pins were learned on the 20261011 trace
and scored on 20261015, so this is not an in-sample artifact (the in-sample
version of the same arm reads 84.61%).

**Engine result: neither the pins nor the policy change survives.** Built both
behind flags and paired them on Ornith, holdout prompt, 64 new tokens:

| arm | 4-bit (3 rounds) | 4-bit (6 rounds) | 8-bit (6 rounds) |
| --- | ---: | ---: | ---: |
| pin file, 48 pins | +0.13% mean / −1.05% median (1/3) | — | — |
| LRU policy | +4.55% median (2/3) | **+0.84% median (4/6)** | **−0.36% median (3/6)** |
| LRU + pins | +4.19% median (2/3) | — | — |

The pin file loads (`expert_pins loaded layers=40 pins=1920`) and changes
nothing: it pins the top-48 by frequency, which capped-LFU with a recency tie
break already keeps. LRU looked like a 4.5% win at three rounds and came back to
+0.84% at six, then lost the 8-bit leg. Both engine changes were reverted; the
simulator and the pin-derivation tool stayed.

**And the simulation was wrong first.** Its LFU column had no recency tie-break,
so a pure count rank evicted the expert it had just admitted whenever everything
sat at count 1 — which is where it read 58.2% against LRU's 83.0% and produced a
+33.8-point oracle gap. Modelling the engine's actual tie-break moved LFU to
**81.7%**, LRU to 83.05%, Belady to 92.08% and the pins to 84.2-85.1%, and shrank
the gap to **10.4 points**. That corrected gap is the honest size of the
placement headroom, and the engine returned 0-1.3 points of it.

What that is worth in time is not yet measured. The simulation ranks policies; it
cannot predict the engine's absolute hit rate, because the runtime's ring
converts misses into hits and reports 81-85% at the same slot count where the
simulation says 67-80%. The next step is an engine arm, not a simulation: a pin
file the streamer honours, then a paired A/B at both bit widths.

### Shared prefix reuse: found the wall, and it is a design wall

The idea worth chasing was serving a *prompt-only* prefix from the cache, so two
sessions that share a prompt pay one prefill. Built behind
`TINYTITAN_PROMPT_PREFIX_CACHE=1`: capture the inference state at the prompt
boundary, publish a prompt-only entry, match it on the direct-prefix branch. It
compiled and it worked as far as it went -- the second request **hit**
(`prompt_cache hit tier=ram cached_tokens=1656`) -- but the wall clock did not
move: 35.0 / 33.9 / 35.4 s across three identical prompts, `cached=0` reported to
the client.

The reason is in the contract, not the cache:

    // The continuation API requires cached < prompt count (it must
    // prefill at least one token), so resume as a full prefill
    ServerModelSession+PromptCache.swift:262-272

and in `runRawCompletion`, which throws `invalidContinuation` unless
`0 < count < promptIds.count`. That guard is **not arbitrary**: prefilling the
last prompt token is what produces the logits the first generated token is
sampled from. A state captured at the prompt boundary carries KV and the GDN
recurrent state, but not those logits, so a zero-token resume has nothing to
sample from.

So the fix is not in the cache. It is to **persist the boundary logits in the
snapshot** -- one row, `vocab x bf16` = ~0.5 MB for a 248k vocabulary -- then
allow `cached == prompt count` and sample directly from the restored row. That
touches the snapshot descriptor (a versioned format), the capture path, the
restore path and the resume guard: a scoped project, not a knob, and not
something to land at the end of a long session. The prototype was reverted.

### Two engines do not beat one engine

The natural idea -- split the engine in two so each serves a request and the
waits overlap -- was measured rather than argued. One server on a 12 GiB budget
serving two prompts one after another, against two servers on 6 GiB each serving
the same two prompts concurrently:

| arm | wall | expert slots per engine |
| --- | ---: | ---: |
| one engine, 12 GiB, two requests sequentially | 23.3 s | 128 |
| two engines, 6 GiB each, concurrently | 23.2 s | **48** |

**1.00x.** Splitting the engine halves the expert cache (128 -> 48 slots each), so
each engine streams more and waits longer, which cancels the parallelism it
bought. This is the same reason in-process batching measured 1.04-1.17x and not
2x: decode is bound by expert bytes per token and the latency of individual
reads, not by idle compute. Two engines do not create bandwidth, and they split
the one thing that was helping.

Parameters for the same reason: 4-bit decode moves ~64 MiB of expert bytes per
token (8-bit ~154 MiB) against a 3.6 GB/s device, so the engine uses ~1-1.5 GB/s
-- there is bandwidth, but a token's reads are serialised behind each other and
behind the layer dependency.

### Startup is not a problem, and the first measurement said it was

I chased a 90-125 s "load time" for an hour before checking what a user actually
sees. The port answers `/v1/models` **1.6 s** after launch, a first request on a
cold server completes in **3.1 s**, and start-to-first-answer is **4.7 s**.

The 90-125 s figures were an artifact of the measurement, not the engine: they
waited for the `ready at` banner in a *redirected* log, and Swift's stdout is
block-buffered there, so the line only appears when the process flushes or dies.
Polling HTTP readiness instead gives the honest number. Nothing to optimise on
this axis, and no claim made from the bad number.

### The disk prompt cache works, and it is worth 13x on a resumed session

The prompt cache is memory-only by default (`prompt_cache_disk=off`), so the
value of `--prompt-cache-disk <dir>` had never been measured here. It is the
largest single number of the night, Ornith 4-bit, a 1,656-token prompt:

| case | prefill |
| --- | ---: |
| turn 1, cold | **32.25 s** |
| turn 2, same server (memory cache) | 1.33 s |
| the same conversation, **after a server restart** (disk cache) | **2.48 s** |

Confirmed at 8-bit with the same shape:

| case | prefill |
| --- | ---: |
| turn 1, cold | 56.31 s |
| turn 2, same server | 2.55 s |
| the same conversation, after a restart | **4.07 s** (13.8x) |

Resuming a session across a restart therefore costs 2.5 s instead of 32.2 s --
**13x less time to first token** -- because the stored KV is reloaded from disk
instead of recomputed. Two entries were written to the directory, so the write
path and the read path both work.

This is not shipped as a default, and **should not be one without a decision**,
for a reason that is not performance: the persisted state is derived from the
prompt, so defaulting it on writes conversation-derived data to disk on every
user's machine. It also carries a disk budget (8 GiB by default). It is a
one-flag win for any workload that restarts sessions with a shared prefix, which
is the coding-agent case:

    --prompt-cache-disk ~/.tinytitan/prompt-cache --prompt-cache-disk-mib 512

Measured at both widths: 32.25 -> 2.48 s at 4-bit, 56.31 -> 4.07 s at 8-bit.
`tools/server_launcher.sh` now forwards both flags **and turns the tier on by
default**, at `<checkout>/.cache/prompt-cache` with a 2048 MiB budget — the
directory is the project's own, so nothing is written outside it.
`--no-prompt-cache-disk` keeps state in RAM only, `--prompt-cache-disk <dir>`
relocates it, and a `--dry-run` creates nothing. The server binary still defaults
to off; this is a launcher decision.

## Status at 2026-10-19 00:00

**Nothing has cleared the bar after ~22 configurations.** The decode path on this
model is at a local optimum: every knob moves it by less than the base arm's own
round-to-round swing, and the three that *are* decisive (prefetch ring, expert
slot count, expert I/O synchronisation) are all already at their measured best —
disabling or mis-setting them costs 8-29%, which is how we know they are set
right.

The single real effect remains the page-cache path at 8-bit (+4.7-5.2%), and a
six-round 4-bit run settles its other leg at **−2.1% (1/6 positive)**. It is a
bit-width-dependent trade, not a blanket win: it pays where a token moves twice
the expert bytes and costs where it does not. That is a *profile* decision for
8-bit installs, not a default, and it does not clear the stated bar.

**Correction to the earlier accounting:** the prefill→decode warm-up I listed as
the biggest unbuilt item has a **withdrawn premise**. The 0.52 hit rate was the
runner's cumulative statistic with prefill's tile misses mixed in; measured
decode-only, the first 64 tokens hit 68/82/87% with and without the use-count
reset, identical to three decimals. The cache is not cold after prefill, and
keeping prefill's counts is already the shipped behaviour. That item is off the
list.

## Queue for the rest of the night

1. Expert slot count at 4-bit (derived 160; 128 and 192), to complete the
   symmetry with the 8-bit sweep.
2. Prefill-side (TTFT) experiments, reported separately: ANE is unavailable for
   this install (no CoreML sidecar), chunk sizes are already optimal.
3. Re-test the page-cache trade on the 4-bit leg with more rounds, to decide
   whether the 4-bit "loss" is real or a floor artifact — if it is neutral rather
   than negative, the trade qualifies on both widths and ships behind a profile
   flag.
4. Anything that clears 4-bit is confirmed at 8-bit before commit.

## Commits

**The strongest decode result of the night is the page cache at 8-bit**, settled
over six paired rounds: **+4.47% median, 6/6 positive** (mean +4.19%), with
`io_ms` falling and the response unchanged. It is negative at 4-bit (−2.1%, 1/6)
and positive there only under concurrency (+1.7%). And it does **not** carry to a
long prompt: on a 1.6K-token prompt at 8-bit the decode delta is −0.21% mean /
+0.91% median (2/3) and **prefill moves +0.3%** (56.5 s against 56.3 s), because
the prefill working set is 17 GB and no cache that fits in RAM can hold it. So
the honest shape is narrow: short-prompt decode at 8-bit, ~4.5%.

**Nothing has shipped.** `51b180d` (four generations by default) was reverted in
the same night: the width does not build at the default context, and the
aggregate gain it claimed was cold-cache amortisation, not batching — +4.3% in
the fair sequential-vs-concurrent comparison, which is noise.

Everything in the tables above is recorded and not shipped, because it measured
below +0.5% on both widths or was a loss on one.
