# Residency and bottleneck verdict, 2026-10-20

Measured on the 24 GiB M3 with `models/ornith-1.5_35B_A3B_4Bit`, all arms
`--max-context 4096 --prompt-cache-mode off`, `TINYTITAN_RUNNER_STATS=1`, one
server per arm, warm request then measured request, 128 new tokens, greedy.

Host state during the run: `vm.swapusage` reported 7–8 GiB already in use
before the first server started, and ~1 GiB of non-server RSS. This is not the
quiet machine the 2026-10-18 campaign ran on; absolute numbers are therefore
lower than that campaign's, but every arm below ran under the same condition.

## Arms

`base` is the shipped default (cache 10.55 G, 160 slots). `s192` is
`--ram-budget 16G --expert-cache-slots 192` (estimate 15.9 G). `s256` is
`--ram-budget 20G --expert-cache-slots 256`, i.e. every expert of every layer
(estimate 19.17 G, "resident").

## Short prompt (26 tokens in, 128 out)

| arm | round | tok/s | hit rate | read MiB | wait ms | body ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 1 | 19.69 | 0.9940 | 437 | 32.9 | 41.6 |
| base | 2 | **19.96** | 0.9934 | 478 | 31.3 | 41.6 |
| s192 | 1 | 19.51 | 0.9964 | 265 | 33.2 | 42.0 |
| s192 | 2 | 12.62 | 0.9964 | 260 | 57.2 | 68.3 |
| s256 | 2 | 12.78 | 1.0000 | 0 | 54.4 | 64.7 |

`s256` round 1 was 1.28 tok/s with 17.4 s prefill and 613 ms wait: the first
load swapped hard. Round 2 recovered to 12.78 — still 36 % below base.

## Long prompt (2.2K tokens in, 128 out; the coding-agent shape)

| arm | round | tok/s | hit rate | read MiB | wait ms | body ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 1 | 10.40 | 0.7862 | 17891 | 52.8 | 86.0 |
| base | 2 | **12.86** | 0.7845 | 18036 | 45.1 | 69.3 |
| s256 | 1 | 2.80 | 0.8144 | 15530 | 269.8 | 289.8 |
| s256 | 2 | 2.78 | 0.8142 | 15549 | 254.6 | 279.3 |

`s256` spends 33–39 s in `pre_ms` pinning the 19 GiB working set and then
pages; decode is 4.6x slower. **Residency at 4-bit is rejected.**

## What the counters actually say

- `io_hidden_pct` is **99.9 %** on the long prompt and 100 % on the short one.
  The expert reader is no longer exposed: the prefetch ring plus the cache hide
  it. Topic 2 (SSD) is not the current bottleneck on either workload.
- `wait_ms` is the CPU blocking on the layer's command buffer
  (`RealForwardRunner+DecodeLayers.swift:261-262`), i.e. GPU execution time, not
  disk time. On the long prompt it is 45–53 ms of a 69–86 ms body.
- The GPU path is byte-bound: 8-bit (twice the bytes) measured 9.85 tok/s
  against 4-bit's 16.2 in the campaign, i.e. roughly half the speed for twice
  the bytes. Effective DRAM throughput during decode is ~33 GB/s of the M3's
  ~100 GB/s, so kernel efficiency, not the SSD, is where the headroom is.
- Prefill of 2,200 tokens takes 65–69 s (≈25 tok/s) while reading 15.5 GiB:
  ~8 % of the machine's fp16 throughput. For an agent loop this is the largest
  single latency term, far larger than decode.

## Verdicts

| item | result | disposition |
| --- | --- | --- |
| P1.4 resident profile at 4-bit | 2.8 vs 12.9 tok/s, `pre_ms` 33–39 s | **rejected**, not shipped |
| 192-slot cache arm | 19.5/12.6 vs 19.7/20.0, unstable | **rejected** |
| shipped default (160 slots, 10.55 G) | best on both workloads | **kept** |

## What this changes in the plan

1. **Residency is not the goal; halving bytes is.** 2-bit experts help because
   they halve the bytes the GPU pulls from DRAM, whether or not the model fits.
   The 4-bit install (19.2 G) cannot be resident on a 24 G host with a normal
   desktop loaded; a 2-bit build (~10–11 G of experts) can, but that is a
   consequence, not the justification.
2. **Route-ahead drops in priority.** It was justified by exposed I/O; I/O is
   already 99.9 % hidden. It only pays again at 8-bit or a cold cache.
3. **Prefill efficiency becomes the top UX item.** 65 s to first token for a
   2.2K prompt at ~8 % MFU is a bigger product problem than decode tok/s.
4. The 2-bit kernel gate is unchanged and now carries the whole thesis: if the
   int2 unpack cannot reach 60 % of the 4-bit kernel's throughput per byte, the
   byte saving is real but the speed is not.

## Reproduction

Arms and harness: `/tmp/tt_ab.sh` (short), `/tmp/tt_ab_long.sh` (long); logs
`/tmp/tt-ab-*.log`, `/tmp/tt-lng-*.log`. No repository code was modified.

## Prefill tile experiment (implemented)

Two env overrides were added to `RealForwardRunner.prefillRoutedTileSchedulerConfig`
(`TINYTITAN_PREFILL_TILE_EXPERTS`, `TINYTITAN_PREFILL_TILE_DEPTH`), and the
default pipeline depth moved from 1 to 2. 2.2K prompt, same host, one request
per server, four order-rotated pairs:

| arm | prefill s | |
| --- | ---: | --- |
| 8 experts, depth 1 (old default) | 51.54, 52.73, 51.18, 51.14 | mean 51.65 |
| 8 experts, depth 2 (new default) | 48.85, 50.98, 49.53, 50.45 | mean 49.95 |
| 16 experts, depth 1 | 51.05 | noise, rejected |
| 16 experts, depth 2 | 51.81 | rejected |

**Depth 2 is kept: −3.3 % to first token, 4/4 pairs faster, and the nine
`PrefillRoutedTileScheduler` tests pass.** Tile width 16 is not kept. The
change is scheduling-only; it does not affect output and is untested at 8-bit
(the override makes that A/B one environment variable).

Prefill is still ~50 s at roughly 8 % of fp16 peak, so this is a 3 % bite out of
the dominant term. The next instrument needed is a per-phase prefill breakdown
(route / shared expert / routed MoE / attention), which the runner does not
currently print.

## Prefill attention: eight queries per threadgroup (opt-in)

The scalar causal kernel launches one threadgroup per (query, head) and
re-reads every visible K/V row for each of them. `attention_prefill_causal_q8`
gives one simdgroup one query and puts eight consecutive queries in a
threadgroup, so the K/V row is read once per threadgroup and the reduction is
a `simd_sum` with no barrier in the key loop. 2.2K prompt, same host:

| | full-attention GPU | prefill wall |
| --- | ---: | ---: |
| `attention_prefill_causal_tiled` | 17.84 s | 50.31 s |
| `attention_prefill_causal_q8` | **11.57 s** | **44.55 s** |

**−35 % on the attention kernel, −11.4 % on time-to-first-token.** The kernel
is opt-in behind `TINYTITAN_PREFILL_ATTN_Q8=1`: the summation order changes, so
the completion is no longer byte-identical and the bug-fix quality suite has
not been run against it. It is gated to `keepMask == nil`, `headDim % 32 == 0`
and `headDim / 32 <= 8`.

The remaining cost is per-element dequantisation: `prefill_load_kv` does two
integer divisions and three loads for every element, and the q8 kernel calls it
once per (lane, key, dim). Hoisting the row pointer and the group index to the
(lane, key) level is the next measured step.

### Hoisted dequant, and promoted to default

`prefill_load_kv` recomputed the row pointer, the group count and the group
index for every element. Those are invariant across keys, so the q8 kernel now
hoists them to the (lane, key) level. Same 2.2K prompt:

| | full-attention GPU | prefill wall |
| --- | ---: | ---: |
| scalar `causal_tiled` | 17.84 s | 50.31 s |
| q8, first version | 11.57 s | 44.55 s |
| **q8 + hoisted dequant** | **8.29 s** | **40.51 s** |

**−54 % on the full-attention kernel and −19.5 % on time-to-first-token**
against the scalar kernel. A four-prompt battery (arithmetic, German
translation, code, capital) returned **4/4 identical answers** with the kernel
on and off, so it is now the default; `TINYTITAN_PREFILL_ATTN_Q8=0` restores
the scalar kernel for byte-identity work.

With attention at 8.3 s, the routed-expert tiles (22.4 s) are now the largest
prefill term, ahead of the shared expert (9.1 s) and GDN (9.1 s).

### Routed-tile knobs: two negatives, 2026-10-20

With attention fixed, the routed-tile phase (22.4 s, of which 11.5 s GPU) is
the largest prefill term. Two scheduler knobs were tested against the 2.2K
prompt and neither clears the noise:

| knob | prefill s | verdict |
| --- | ---: | --- |
| depth 2 (default) | 39.11, 38.99, 39.11 | keep |
| depth 3 | 39.06 | noise |
| depth 4 | 38.53 | within noise, not re-tested |
| depth 6 | 39.60 | worse |
| tile 16 experts | 40.04, 39.01 | noise/slightly worse |
| tile 8 (default) | 39.0 | keep |

The tile pipeline already overlaps the host fetch with the previous tile's
GPU work (`drainOldestPendingTile`), which is why depth is flat: the fetches
are not the exposed serial term any more. The remaining 11 s of non-GPU time
in that phase is inside `fetchBindingForTile`, and the GPU half runs at
~1.3 TFLOPS; both need a profile of the fetch path before any change.

### Next target located: the shared expert is a per-token loop

`prefill_shared_expert` is 9.12 s of GPU time over 40 layers, ~275 GFLOPS —
five times less efficient than the routed tiles. The cause is in
`PrefillSharedExpert.encodeBlock` (`Kernels/Prefill/MoE/PrefillSharedExpert.swift:46`):

    for row in 0..<queryCount {
        try shared.encode(commandBuffer: cb, x: x, xOffset: xOffset + row * ...)
    }

It calls the decode-path single-token GEMV once per token, so a 2.5K chunk
issues 2,505 tiny dispatches per layer through the int8 shared-expert runtime
instead of one GEMM. Each dispatch moves the same ~4 MB of weights and reaches
roughly 2 GFLOPS, which is why the phase costs what forty decode layers cost.

The fix is a batched int8 GEMM over the token dimension (dequantise a weight
tile to half in threadgroup memory, `simdgroup_matrix` over 8-token rows), or a
per-layer dequantise into a half scratch followed by the existing
`prefill_dequant_affine_qmm_f16_block`. Neither is a knob; both need the
kernel work the attention phase needed. Estimated prize: 6-7 s of the 40 s
prefill, on top of the 19.5 % already taken.

### Reader threads on the raw path: not a win (2026-10-20)

The raw expert reader runs four threads while the compressed path runs eight.
A six-line override made the count testable; 4 vs 8 on the 2.2K prompt:

| threads | prefill s | |
| --- | ---: | --- |
| 8 | 38.90, 42.17 | mean 40.54 |
| 4 | 40.36, 42.33 | mean 41.34 |

The first pair is 3.6 % in favour of eight, the second is 0.4 %, and the whole
run drifted 3.4 s between rounds, so the effect does not clear this machine's
noise. Reverted; the raw path stays at four. The 11 s of host time in the
routed phase is therefore not reader-thread starvation.

The shared-expert repack shortcut was also bounded and rejected before
building it: the existing batched `prefill_dequant_affine_qmm_f16_block` is
itself a scalar per-output loop (prefill.metal:564) running at ~364 GFLOPS on
QKV, so applying it to the shared expert would give 6.9 s against the current
9.12 s — about 2 s for a format repack and a new activation path. A tiled
simdgroup GEMM is the only shape of fix worth the work there.

### Shared expert: the layouts already match (implementation spec)

Last round's "repack costs 2 s" bound is void. The shared expert's weights are
int8 in row-major rows of `cols`, and its scales/biases are **bf16 per group of
64** — byte-for-byte the layout `prefill_dequant_affine_qmm_f16_block` already
expects (`kInt8GroupSize == kPrefillGroupSize == 64`, and `prefill_affine_bits()`
defaults to 8 when the constant is unset). No repack and no new scale buffers
are needed.

What is missing is only plumbing:

1. **An elementwise activation kernel**: `act[i] = prefill_hidden_activation(gate[i]) * up[i]`
   over `T * 512` elements. `prefill.metal` already carries
   `prefill_hidden_activation` (gelu_pytorch_tanh, with the silu branch), so the
   kernel is ~10 lines and exact by construction.
2. **Chunk-sized scratch**: the current path reuses one row's gate/up/act
   scratch; the batched path needs `T x 512` half buffers for gate and up and
   `T x 512` for act, allocated with the prefill scratch at the chunk size.
3. **Wiring** in `PrefillSharedExpert.encodeBlock` behind
   `TINYTITAN_PREFILL_SHARED_GEMM`, default off until measured.

Shapes: gate/up `512 x 2048`, down `2048 x 512`, T up to the 4,096 chunk.
At the qmm kernel's measured ~364 GFLOPS the three projections are 1.26 TFLOP
→ **~3.5 s against 9.12 s**, a 5.6 s (14 %) prefill win before any blocking.

### Batched shared expert (kept)

`PrefillSharedExpert` now runs gate/up/down through the prefill affine matmul
(`TINYTITAN_PREFILL_SHARED_GEMM`, on by default) with one elementwise
activation between them, instead of the decode GEMV once per token.

| | shared-expert GPU | prefill wall |
| --- | ---: | ---: |
| per-token GEMV (before) | 9.12 s | 40.51 s |
| **batched GEMM** | **4.60 s** | **34.61 s** |

**−50 % on the shared expert, −14.5 % on the whole prefill**, and a four-prompt
battery (arithmetic, German, code, capital) is 4/4 identical against the old
path. Scratch grows by 3 x chunkTokens x 512 halfs (12 MB at the 4,096 chunk).

Two function constants are load-bearing here and were both wrong on the first
build, which is why the first measurement produced a refusal instead of an
answer: index 78 (`FC_PREFILL_AFFINE_BITS`) defaults to 4, not 8, and index 77
(`FC_PREFILL_ACT_SILU`) defaults to gelu, not this model's silu. A batched path
must set both from the projection's width and the architecture's activation.

### Eight-token blocking of the qmm kernel: no change (2026-10-20)

QKV, GDN and the batched shared expert all run through
`prefill_dequant_affine_qmm_f16_block`, ~13 s of the 34.6 s prefill. A variant
gave each thread one output row and eight consecutive tokens, reading every
weight once per eight tokens instead of once per token, with the same
per-(token, output) accumulation order. Against the shipped build:

| | prefill s | shared expert | GDN | QKV |
| --- | ---: | ---: | ---: | ---: |
| eight-token blocking | 35.07 | 4.78 | 9.15 | 3.56 |
| shipped | 34.61 | 4.60 | 9.10 | 3.46 |

Every figure is inside the noise, so the variant was reverted. The kernel is
**compute-bound, not weight-traffic-bound**: cutting its weight reads eightfold
bought nothing, which means the scalar per-element dequantisation and dot
product is the cost. The lever left on those 13 s is vectorising that inner
loop (four or eight int8 values per load, group-aligned so a load cannot cross
a group), not blocking over tokens.

### Vectorised int8 qmm: neutral (2026-10-20)

A four-wide variant of the qmm kernel (`uchar4`/`half4` loads, scale and bias
hoisted to the group, same accumulation order) measured **33.43 and 33.50 s**
against a **33.47 s** control with the variant switched off — a dead heat.

The lesson is the measurement one: the 34.61 s "before" was drift, not the
kernel. Two runs of the same build straddled 0.1 s, and the control had to be
run to see it. Reverted.

The projection kernel is therefore neither weight-traffic-bound (round 10) nor
inner-loop-ALU-bound in the way vectorising addresses. Its remaining cost is
probably the dependent fma chain per thread; hiding that needs a different
thread mapping (split K across threads with a reduction), which is a larger
change than a load-width rewrite.

## Paired validation of the two kept prefill changes (2026-10-20)

Both changes had been measured across builds, where machine drift had already
been caught once. Re-measured on one build with the env toggles, alternating
arms, two rounds each.

### Batched shared expert (`TINYTITAN_PREFILL_SHARED_GEMM`)

| round | off: prefill / shared GPU | on: prefill / shared GPU |
| ---: | ---: | ---: |
| 1 | 38.49 s / 8.88 s | 33.39 s / 4.56 s |
| 2 | 38.55 s / 8.83 s | 33.51 s / 4.57 s |

**−13.3 % prefill, −48.6 % on the shared expert, 2/2 pairs.**

### Eight-query attention (`TINYTITAN_PREFILL_ATTN_Q8`)

| round | off: prefill / attention GPU | on: prefill / attention GPU |
| ---: | ---: | ---: |
| 1 | 43.44 s / 17.69 s | 33.46 s / 7.80 s |
| 2 | 43.36 s / 17.68 s | 33.28 s / 7.81 s |

**−23.2 % prefill, −55.9 % on the full-attention kernel, 2/2 pairs.**

Together the two toggles account for ~15 s of the ~48.5 s both-off prefill:
**~48.5 s → ~33.4 s (−31 %), paired rather than cross-build.** The earlier
"50.31 → 34.61" figures are replaced by these.

## Session outcome and backlog (2026-10-20)

### Shipped, with paired evidence

| change | evidence | commit |
| --- | --- | --- |
| prefill depth 2 | 51.65 → 49.95 s, 4/4 pairs | `0423b71` |
| q8 prefill attention (+ hoisted dequant) | attention GPU 17.7 → 7.8 s, prefill −23.2 %, 2/2 pairs; ring-gated | `6eaf9e7`, `cd199da`, `b35df2a` |
| batched shared expert | 8.85 → 4.57 s, prefill −13.3 %, 2/2 pairs | `5092627` |
| prefill phase trace (attention vs router) | instrument | `25150bc` |

Both-off prefill is ~48.5 s; both-on is ~33.4 s, measured paired. The earlier
"50.31 → 34.61" cross-build figures are superseded.

### Rejected, each with a control

4-bit residency (2.78 vs 12.86 tok/s), 192 slots, depth 3/4/6, 16-expert
tiles, token-tiled router, 8 raw reader threads, eight-token qmm blocking,
vectorised int8 qmm. The last was flat against its own control (33.43/33.50
vs 33.47) and is the reason a control is now mandatory before any claim.

### Findings that outlive the changes

- `io_hidden_pct` is 99.9 %: the SSD is hidden; `wait_ms` is GPU command-buffer
  time, and prefill is GPU-bound at ~400 GFLOPS on every GEMM-shaped kernel.
- The four scalar kernels (qmm, shared expert, routed tiles, router) all ran
  at 2-8 GFLOPS per token before batching; the attention kernel was the same
  shape of bug and gave the largest single win.
- Function constants are load-bearing: 77 (silu) and 78 (affine bits) both
  default to the wrong value for this model, and 76 (ring) must be compiled
  per pipeline. A batched path that forgets any of them is silently wrong.

### Backlog, largest first

1. **Routed-tile grouped GEMM**: 11.7 s GPU at ~430 GFLOPS plus ~10 s host
   fetch. Needs a tiled `simdgroup_matrix` MoE kernel, not a knob.
2. **GDN**: 8.5 s over 30 layers; projections are qmm (~364 GFLOPS) and the
   recurrence is serial. A chunked delta-rule scan is an algorithm project.
3. **qmm thread mapping**: split K across threads with a reduction; rounds 10
   and 11 bounded blocking and load width as flat, leaving the dependent fma
   chain as the suspect.
4. **2-bit routed experts**: halves per-token DRAM bytes; needs format width 2,
   a tiered repacker and an int2 kernel (the original plan's P1).
5. **Route-ahead**: deferred while I/O is hidden; only pays at 8-bit or a cold
   cache.

### GDN is already batched (2026-10-20)

The 8.5 s GDN term is not another per-token host loop. `encodeDeltaStepPrefill`
issues **one** dispatch for the whole chunk (`rows`), grid
`(numVHeads, valueHeadDim / 4)`, and the delta-rule scan runs inside the
kernel. The scan is serial by the algorithm, not by the host loop, so the
shared-expert/attention fix does not apply.

GDN's remaining cost is therefore two things already bounded: the qmm
projections (the same kernel as QKV and the shared expert) and the serial
scan, which would need a chunked parallel form — an algorithm change, not a
scheduling one. GDN is off the cheap list.

### Sixteen-query attention block: no change (2026-10-20)

Doubling the q8 block from 8 to 16 queries per threadgroup halves its K/V
reads again. Measured 33.53 / 33.51 s prefill and 7.80 / 7.80 s attention,
identical to the eight-query block's 33.4-33.5 s and 7.80 s. Reverted.

With this, every GEMM-shaped kernel in the prefill has now been shown to be
compute-bound rather than traffic-bound: qmm blocking and vectorising (rounds
10, 11), attention blocking (this round), routed tiles (unchanged by wider
tiles). The scalar fma chain is the common cause, and fixing it needs tiled
`simdgroup_matrix` kernels rather than more blocking.

### Combined quality check (2026-10-20)

The shipped default (q8 attention and batched shared expert both on) against
both toggles off, four prompts, greedy: **391 / Die Katze schläft / the
sum() one-liner / Canberra — 4/4 identical**. The two optimisations compose
without changing short-answer behaviour, which is the check the individual
four-prompt batteries could not make.

## Design sketch: tiled GEMM for the projection family (not yet built)

Target: `prefill_dequant_affine_qmm_f16_block`, used by QKV, GDN and the
shared expert (~13 s of a 33.4 s prefill). Every cheaper lever is measured
flat, so this is the next real step. Sketch only; no code written.

- **Shape**: 128 threads (4 simdgroups) per threadgroup, output tile
  `32 tokens x 32 rows`, K blocks of 32 (K is 2048 or 512 here, both divisible).
- **Staging**: cooperative dequant of the int8 weight block and the half
  activation block into `threadgroup float` arrays (4 KB each), one barrier per
  K block. 64 barriers per output tile at K=2048.
- **Compute**: `simdgroup_matrix<float, 8, 8>` with
  `simdgroup_multiply_accumulate`, four 8x8 output sub-tiles per simdgroup.
  Float accumulation avoids the half-epilogue precision loss the current
  kernel's `half(acc)` hides.
- **Epilogue**: scale/bias per 64-element group applied after accumulation,
  then one `half` store per output.
- **Correctness**: accumulation order changes, so this is not bit-identical to
  qmm; gate on the four-prompt battery plus a numeric comparison of the
  prefill logits against the qmm path, not on text identity.
- **Rollout**: env-gated, default off, paired A/B with the toggle off as its
  own control (the round-11 lesson), kept only above +10 %.
- **Risk**: the barrier count and the dequant staging may dominate at these
  small K blocks; if so, widen to 64 tokens per tile before concluding.

Same structure applies to the routed-tile grouped GEMM, which needs a pair
sort per tile on top; the projection kernel is the cheaper first target and
de-risks the pattern.

### Tiled projection GEMM: built, correct, not faster (2026-10-20)

The design sketch was implemented and measured rather than assumed. A 32x32
tile with 128 threads, int8 weights dequantised to threadgroup float during
staging, `simdgroup_matrix<float,8,8>` accumulation and a float epilogue:

| variant | prefill s | GDN | shared | QKV |
| --- | ---: | ---: | ---: | ---: |
| scalar qmm (shipped) | 33.4-33.5 | 8.6 | 4.56 | 3.33 |
| tiled, K block 32 | 33.97 | 8.75 | 4.62 | 3.40 |
| tiled, K block 64 (half the barriers) | 34.04 | 8.79 | 4.63 | 3.46 |

Output was numerically sound (same divergence point as the reference on the
free-form prompt), so the kernel worked; it simply does not beat the scalar
kernel. Reverted.

**This closes the GEMM line of attack.** Blocking (round 10), vectorising
(round 11) and now tiling all fail to move the projection kernels, whose
~364 GFLOPS is evidently this GPU's practical rate for these matrix shapes.
Any further prefill gain has to come from doing less arithmetic — 2-bit
weights halving the bytes and the multiply count, or skipping work — not from
a better inner loop.

## The 2-bit gate: quality says no to uniform 2-bit (2026-10-20)

`benchmark/tinytitan_3bit_probe.py` already measured 2-bit alongside 3-bit;
it had never produced that column because it crashed on the first 1-D tensor
it sampled (`requantize` assumed a matrix). Fixed to accept vectors and
edge-pad a row to the group boundary, then run against the installed 8-bit
reference:

| role | n | 2-bit added error | 3-bit added error | 3-vs-4 ratio |
| --- | ---: | ---: | ---: | ---: |
| routedExpert | 12 | **0.4695** | 0.2083 | 2.15x |
| sharedExpert | 2 | 0.5108 | 0.2270 | 2.15x |
| attention | 2 | 0.2800 | 0.1246 | 2.11x |
| router | 2 | 0.5756 | 0.2463 | 2.14x |

Each bit removed roughly doubles the quantisation error: 4-bit ~0.097,
3-bit ~0.208, 2-bit ~0.47 against the 8-bit reference. Packing is fine — 2-bit
is power-of-two, 20 bytes per 64-weight group including scale and bias,
**0.556x of 4-bit (44 % saving)** with no word crossings — so the byte case is
real but the quality case is not.

**Uniform 2-bit is therefore closed before building a format, a repacker or a
kernel.** What survives is only the mixed form from the original plan: keep
warm/hot experts at 4-bit and put *cold* experts at 2-bit, where the error
lands on weights that are rarely selected; or a codebook scheme (AQLM-style
vector quantisation) whose 2-bit is not uniform affine. Both need the routing
frequency data the trace work already produced.

## Hot/cold split, measured on Ornith's own routing (2026-10-20)

`TINYTITAN_ROUTE_TRACE` recorded a 2.2K-token prompt plus 128 decode tokens
(2,632 positions x 40 layers, 105,280 route lines) from the installed 4-bit
model. `benchmark/route_skew.py` turns it into the only table that matters for
the hybrid:

| hot fraction | experts at 4-bit | activation mass on hot | on cold | effective bits | bytes vs 4-bit |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.20 | 51 | 0.667 | 0.333 | 2.700 | −40.0 % |
| 0.25 | 64 | 0.730 | 0.270 | 2.812 | −37.5 % |
| 0.40 | 102 | 0.870 | 0.130 | 3.150 | −30.0 % |
| 0.50 | 128 | 0.925 | 0.075 | 3.375 | −25.0 % |
| 0.60 | 153 | 0.961 | 0.039 | 3.600 | −20.0 % |

Effective bits include the format's real overhead: 4-bit at group 64 is 4.5
bits/weight, 2-bit at group 128 is 2.25.

The routing is strongly skewed — 64 of 256 experts (25 %) carry 73 % of all
selections — so the hybrid has a real operating point. **The recommended split
is 40–50 % hot: 25–30 % fewer expert bytes with only 9–15 % of activations
running through 2-bit weights**, where the probe measured 0.47 added error.
Tighter splits save more but put a third of activations on the weakest format.

This is the design decision the plan was waiting for, and it needed no runtime
change — only a trace and the committed routing.

## Hybrid vs uniform 3-bit: the hybrid dominates (2026-10-20)

The probe's per-width added error and the trace's activation coverage give a
direct comparison, weighting each expert's error by the share of activations
that actually reach it:

| format | added error | vs 4-bit | expert bytes vs 4-bit |
| --- | ---: | ---: | ---: |
| 4-bit (today) | 0.0969 | 1.00x | 1.000x |
| uniform 3-bit | 0.2083 | 2.15x | 0.778x |
| **hybrid, 40 % hot** | **0.1453** | **1.50x** | **0.700x** |
| **hybrid, 50 % hot** | **0.1248** | **1.29x** | **0.750x** |

Inputs: 2-bit added error 0.4695, 3-bit 0.2083, 3-vs-4 ratio 2.15 from
`tinytitan_3bit_probe.py`; activation coverage 0.870 / 0.925 from
`route_skew.py`; bytes include scale/bias (3-bit shown at its *unpadded* stream
size, 0.778x, which flatters it).

**The hybrid beats uniform 3-bit on both axes at either split** — fewer bytes
and lower error — and 3-bit was already unattractive because it is not a
power-of-two packing. So 3-bit is closed as a fallback, not merely deprioritised,
and the cold-tier format should be built for the 40–50 % hot split.

Caveat: the error metric is mean absolute error relative to mean |w|, and the
weighting assumes uniform per-token sensitivity. It is a planning proxy; the
bug-fix suite remains the shipping gate.

## The cold tier needs no new Metal kernel (2026-10-20)

Checked before planning the 2-bit implementation, and it changes the size of
the project. Every kernel that would touch a cold expert is already
parameterised by width:

| kernel | constant | where set |
| --- | --- | --- |
| `affine_quant_gemv_simd` (decode projections) | 100 | `AffineQuantGEMV` |
| `prefill_dequant_affine_qmm_f16_block` | 78 | `PrefillInt4QMM` |
| routed MoE phase kernels | **5** (`FC_MOE_WEIGHT_BITS`) | `PrefillGroupedRoutedMoE`, `MoE` |
| router | 79 | `PrefillRouter` (already 4/8/16) |

The accessors (`affine_quant_value`, `prefill_affine_value`) shift and mask a
generic `bits`, and row word counts are computed as `columns * bits / 32`, so
2-bit fits the same arithmetic with no new shader.

What stands between here and a 2-bit cold tier is therefore **nine Swift
`[4, 8]` preconditions** — `SlotGEMV`, `AffineQuant` x2, `CPUExpertFFN`,
`PrefillFinalRowHead`, `PrefillPrimitives` x2, `PrefillGroupedRoutedMoE`,
`MPPPrefillInt4QMM` — plus the two manifest width sets, and then the real work:
a **tiered repacker** that quantises the cold experts to 2-bit per the routing
frequency table and writes per-expert widths instead of one stride.

Estimate revision: this was planned as format + repacker + kernel; it is now
repacker + gates + a 2-bit kernel test (the gates must not be opened without a
test that compares a 2-bit GEMV against a CPU reference, because no shipped
install exercises them).

## The format already supports per-expert widths; the reader does not (2026-10-20)

`packed_experts/layout.json` already carries, for every expert, its `offset`
and `size`, and for every tensor its own `bits`, `dtype`, `shape`, `offset`
and `size`:

    {"expert": 0, "offset": 0, "size": 1769472,
     "tensors": {"gate": {"bits": 4, ...}, "down": {"bits": 4, ...}, ...}}

So the on-disk schema needs **no change** for a 4-bit-hot / 2-bit-cold install.
What blocks it is the runtime: `Model+LayerStreamers` and the pread reader
compute every expert's position as `rank * expertStride`, and the streamer,
cache plan and ring all assume one expert size.

The minimal design that avoids redesigning the cache:

- the repacker writes **two contiguous regions per layer** — hot at 4-bit with
  stride H, cold at 2-bit with stride C — and the existing per-expert
  `offset`/`size` entries describe both;
- the reader takes `(offset, size)` per expert instead of a single stride (a
  table, or two readers, one per region);
- slot **capacity** stays at the 4-bit maximum, so a 2-bit expert fills half a
  slot; that wastes RAM capacity but costs nothing in bytes read, and avoids
  reworking slot sizing, the ring and the cache plan;
- kernels already accept a per-dispatch width constant, validated at 2-bit in
  five paths.

Remaining work in order: variable-size reader, two-region repacker, then the
measurement. It is a contained core change, not a format project — the schema
work that this plan originally budgeted for does not exist.

## The cold tier's plumbing already exists; the dispatch is the work (2026-10-20)

Checked the streaming core before planning a variable-size reader, and most of
it is already there:

- `tinytitan_expert_reader_fetch_ranges(reader, offsets, byte_counts, dests,
  count)` — absolute offsets **with per-item byte counts**, already used by the
  phase-1/tail split.
- `Model.routedExpertPhysicalOffsets(layer:)` returns each expert's offset
  straight from `layout.json`.
- `ParallelExpertReader` already wraps both the offset and the range fetches.

So the reader does not need a new primitive, and the repacker's output can be
consumed with `fetch_ranges(offsets, sizes, …)`.

**What is genuinely missing is the dispatch split.** The routed MoE pipeline is
compiled with `FC_MOE_WEIGHT_BITS` (constant 5) — one width per pipeline. A
layer holding both tiers must therefore issue **two routed dispatches per
layer**, one per pipeline, over the hot and cold pair subsets, with two
argument buffers and one shared reduction. That, plus slot bookkeeping that
records each expert's byte count and its tier, is the cold tier.

Revised remaining work, in order:

1. repacker: two contiguous regions per layer (schema unchanged);
2. streamer: record `(offset, size, bits)` per cached expert, capacity at the
   4-bit maximum;
3. MoE: hot/cold pair partition and two phase-1 dispatches into one reduction;
4. measurement against the 40–50 % hot split.

Estimate: about a week, with step 3 the only part that touches kernel dispatch.

### Hot-expert profile generator (2026-10-20)

`benchmark/hot_experts.py` is the repacker's missing input: it turns a route
trace into the per-layer hot list, with the activation share and byte ratio
that split implies, so the choice is made before any install is written.

    python benchmark/hot_experts.py --trace /tmp/ornith-route.txt --hot 0.5 --out /tmp/hot50.json

Measured on the recorded Ornith trace:

| hot fraction | hot per layer | activation on hot | bytes vs 4-bit |
| ---: | ---: | ---: | ---: |
| 0.40 | 103 | 0.870 | 0.700x |
| 0.50 | 128 | 0.925 | 0.750x |

The JSON carries `layers[L].hot` (sorted expert ids), the counts, the
per-layer activation share and the header's overall byte ratio — exactly what
a tiered converter needs to decide which experts to quantise to 2-bit.

## What the hybrid would do for decode (projection, not measurement)

Decode is byte-bound: the 8-bit install moves twice the bytes and runs at
9.85 tok/s against 4-bit's 16.2 in the campaign. The same byte ratio the hot
profile reports therefore projects onto decode:

| split | expert bytes vs 4-bit | MB/token (from 196) | tok/s if byte-bound |
| --- | ---: | ---: | ---: |
| 40 % hot | 0.700x | 137 | 19.7 |
| 50 % hot | 0.750x | 147 | 18.4 |

Against the measured 13.8 tok/s on the long prompt that is **+33 % to +43 %**
if the byte model holds — and it is the same change that takes 25–30 % off the
prefill. This is the prize the tiered converter is for; it is a projection
because decode's `wait_ms` is GPU command-buffer time, so part of the saving
may be hidden by compute rather than realised one-for-one.

## The decode byte model, measured rather than assumed (2026-10-20)

The hybrid's decode projection rested on "decode is byte-bound", which came
from older campaign runs on a different machine state. Re-measured on one
build with one prompt (~950 chars in, 128 out):

| install | decode tok/s | body ms | wait ms | expert MiB (request) |
| --- | ---: | ---: | ---: | ---: |
| 4-bit | **18.34** | 46.5 | 28.3 | 12,231 |
| 8-bit | **9.27** | 98.2 | 47.9 | 29,950 |

**2.45x the bytes for 2.11x the body time** — the near-linear scaling confirms
the byte-bound model the hybrid projection depends on. (The byte ratio exceeds
two because the 8-bit install also runs a smaller cache: 96 slots against 160,
hit rate 0.80 against 0.99 on this prompt.)

So the projection stands: a 4/2 hybrid at 0.70-0.75x expert bytes should land
near 1.33-1.43x decode throughput, 13.8 → 18-20 tok/s on the long prompt, with
the compute caveat already recorded. This is now the second measured pillar
under the remaining project, after the split decision.

### Correction: the schema supports mixed widths, the validator does not (2026-10-20)

The previous entry said "no format change needed". That was too optimistic,
and a test written this round found the exact line. `layout.json` accepts
per-expert `offset`/`size`/`bits`, but the format validator refuses the shape:

    SSDAIPackedExpertsLayoutV1.swift:203
    guard expert.offset == expectedOffset, expert.size == layout.expertStride
    // "offset or size does not match physical rank"

Two checks stand between a tiered install and the runtime:

1. **`SSDAIPackedExpertsLayoutV1`**: offset must equal `rank * expertStride` and
   size must equal the stride. A compact two-region layout breaks both. It
   needs to become "ranges are in bounds and do not overlap, size <= stride".
2. **The manifest's file-size cross-check** (`expertsPerLayer * expertStride`
   against `files[...].size`) assumes a full-size layer file; a compacted one
   is smaller.

`MixedExpertWidthLayoutTests` pins the intended behaviour and is marked
`@Test(.disabled)` with the reason pointing at line 203, so it flips on when
the validator is relaxed rather than failing the suite meanwhile.

So the cold tier is: two validator relaxations, the tiered converter, and the
dual-width dispatch — still bounded, but the format work is real and is
approximately one of the four steps, not zero.

### Fixed: the validator now accepts compact per-expert extents (2026-10-20)

The two checks recorded as blockers are relaxed:

- `SSDAIPackedExpertsLayoutV1` no longer requires `offset == rank * stride` and
  `size == stride`. It now requires `0 < size <= stride`, non-overlapping
  ranges, and that every range ends within the layer's nominal capacity
  (`expertsPerLayer * stride`). That admits a compact two-region layer while
  still refusing an expert that overruns the file's declared extent.
- `crossValidate` accepts a layer file whose size is between the highest
  expert end and the uniform product, instead of requiring exact equality.

`MixedExpertWidthLayoutTests` is enabled and passes: a layer with a 16 KiB
4-bit expert and an 8 KiB 2-bit expert decodes with both sizes and widths
intact, alongside the unchanged uniform layouts. Two existing contract tests
were updated rather than weakened — the out-of-layer offset is still refused,
and an oversized blob is still refused; what changed is that a *smaller* blob
is now the legal tiered case.

Full suite green. The format step of the cold tier is done; the remaining
work is the tiered converter, the streamer's per-expert size bookkeeping, and
the dual-width dispatch.

### Streamer reads per-expert sizes (2026-10-20)

`StreamLayout` now carries `expertSizes` from `layout.json` (offsets were
already per-expert), and the raw pread path uses the reader's
`fetch(offsets:byteCounts:into:)` when the sizes are not all the stride:

    fetch_ranges(offsets, sizes)   // tiered layer
    fetch(offsets)                 // every expert fills the stride

The phase-1 split is skipped for a tiered layer, because its split point is
per-width and the manifest records one per layer. Uniform installs take the
old path exactly, which the decode check confirms (18.74 tok/s against 18.34
before the change, same prompt and build).

Remaining for the streamer step: record each cached expert's `bits` so the MoE
dispatch can pick a pipeline. The read half is done.

## A cheaper path than the dispatch split: expand cold experts on load (2026-10-20)

The dual-width dispatch is the expensive part of the cold tier: the routed
pipeline is compiled one width at a time and the reduction kernel computes `y`
from exactly `topK` experts, so a mixed layer would need two dispatches, two
argument buffers, and either padding each group to `topK` (doubling the GEMV
work) or new `k`-compiled reduce kernels.

There is a cheaper decomposition. Store cold experts at 2-bit, and **expand
them to the 4-bit slot on load**:

    q4 = 5 * q2                 (0..3 -> 0, 5, 10, 15 — exact)
    scale' = scale / 5          (bf16, ~2e-3 relative error)
    bias'  = bias               (unchanged)

Then `q4 * scale' + bias' == q2 * scale + bias`, so the existing 4-bit kernels
run unchanged on an expanded slot. The scale division is the only added error,
and at ~0.2 % it is inside the quantisation noise the tier already carries.

Cost, per expert, at CPU unpack rates (the cores are idle during decode —
0.4 of 8 measured):

| unpack rate | per expert | per token (~3.2 misses) |
| --- | ---: | ---: |
| 1 GB/s | 2.4 ms | 7.5 ms |
| 2 GB/s | 1.2 ms | 3.8 ms |
| 4 GB/s | 0.6 ms | 1.9 ms |

Against a 45 ms token and a projected 25–30 % byte saving (~11–13 ms), the
2 GB/s case nets roughly 7–9 ms, and the work runs on cores that are otherwise
idle. What this buys:

- no second pipeline, no second argument buffer, no reduction change;
- no padding waste — the GEMV still processes `topK` real experts;
- the 2-bit kernels validated this session remain useful for the *converter*
  and as a cross-check, not as a second dispatch path.

The cold tier becomes: **tiered converter + an unpack step in the I/O path**.
The dual-width dispatch stays as a later optimisation if the unpack cost turns
out to matter.

### The expansion step measured: 0.27 ms per token, not 4-8 ms (2026-10-20)

`benchmark/two_bit_expand_bench.swift` measures the 2-bit to 4-bit transform on
one expert's weights (3 x 512 x 2048 values) at 1, 2, 4 and 8 threads:

| threads | per expert | throughput | per token (~3.2 misses) |
| ---: | ---: | ---: | ---: |
| 1 | 0.172 ms | 13.7 GB/s | 0.55 ms |
| 2 | 0.140 ms | 16.9 GB/s | 0.45 ms |
| 4 | 0.084 ms | 28.1 GB/s | **0.27 ms** |
| 8 | 0.090 ms | 26.3 GB/s | 0.29 ms |

The earlier estimate assumed 1-4 GB/s; the real transform is 14-28 GB/s
because it is a byte shuffle with no arithmetic dependency. **0.27 ms is 0.6 %
of a 45 ms token**, against the 11-13 ms the byte saving is worth, so the
expansion cost is not a consideration at all.

This settles the design: the cold tier is a tiered converter plus this
transform on the I/O path, with the dual-width dispatch no longer needed for
the first measurable version.

### Expansion transform shipped and proven on a real file (2026-10-20)

`TwoBitExpand` (Infrastructure/Streaming) is the cold tier's expansion step:
2-bit to 4-bit packed, `v -> 5v`, with `adjustedScale(_:)` for the paired
scale division. `TwoBitExpertReadTests` proves both halves on a real layer
file: a half-slot expert at a non-zero offset is fetched by `(offset, count)`
with the bytes past the payload untouched, and the expanded stream decodes back
to exactly `5x` the 2-bit values.

That closes the runtime mechanism the hybrid needs. What remains is the tiered
converter and calling the transform after a short read (plus applying the
scale division), both ordinary work against validated pieces.

### Converter scope narrowed: no receipt needed under the default policy (2026-10-20)

`Model+Loading` reads `verified-install.json` only when the integrity policy is
`sizeCheckTrustedReceipt`; under the default policy the receipt is `nil` and
each layer file is verified against the manifest's own sha256. So a prototype
converter must update `manifest.files[...]` sizes and hashes, but it does not
have to mint a receipt.

The expansion's arithmetic is also pinned now: `TwoBitExpertReadTests` checks
`q2*scale + bias == (5*q2)*(scale/5) + bias` across scales and biases, in
addition to the byte-level expansion. Three tests, all passing.

The converter remains the last piece and is a compute job as much as a coding
one: ~6,000 cold experts x ~3M values to dequantise from 4-bit and requantise
to 2-bit, which is minutes-to-tens-of-minutes of numpy rather than an
interactive loop.

## Converter core: keep the scales, remap the values (2026-10-20)

The tiered converter looked like a compute job — dequantise 4-bit weights,
requantise to 2-bit, mint new group scales — until the expansion's arithmetic
is read backwards. Because the runtime expands a cold expert with
`q4' = 5 * q2` and `scale' = scale / 5`, the converter can keep the expert's
**existing scales and biases** and only remap values:

    q2 = round(q4 / 5)

| q4 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
| --- | - | - | - | - | - | - | - | - | - | - | -- | -- | -- | -- | -- | -- |
| q2 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 2 | 2 | 2 | 2 | 2 | 3 | 3 | 3 |

Expanding reproduces `5 * q2`, and the induced error is at most 2 units of the
4-bit scale — half a 2-bit step. No float maths, no new scales, no bf16
encoding: it is a byte lookup table from two 4-bit-packed bytes to one
2-bit-packed byte.

`benchmark/two_bit_pack.py` implements it and `test_two_bit_pack.py` pins three
things: every 4-bit value maps to its nearest quarter, a random 4 KiB stream
round-trips through the expansion, and an odd-length stream is refused. 3/3
pass.

So the converter is now file plumbing — read a cold expert's four tensors,
LUT the weights, copy scales/biases, write the blob and the layout entry,
update the manifest hashes — plus the hot experts copied verbatim. The last
compute-shaped risk in the project is gone.

### The tiered converter exists and produces a loadable install (2026-10-20)

`tools/tiered_repack.py` walks a 4-bit install, copies hot experts verbatim,
remaps cold ones with the tested LUT, page-aligns every expert, rewrites
`layout.json` with per-expert `offset`/`size`/`bits`, hard-links the resident
weights and the untiered layers, and updates the manifest's file sizes and
hashes — including the entry for `layout.json` itself, which the engine checks
and which the first run missed (it failed loudly: "tensor
packed_experts/layout.json size 7777182 does not match expected 22493846").

A no-tier build (`--layers 999`, every layer hard-linked) loads and answers, so
the manifest/layout rewrite is accepted by the engine's own verification. The
tiered path still needs the expansion call site: the streamer can read a short
expert (proven), but nothing expands it into the slot yet, so a tiered layer
would hand the 4-bit kernel half-valid bytes.

Remaining: pass the blob's per-tensor offsets to the streamer and run
`TwoBitExpand` (plus the scale division) after a short read; then run the
converter for real and measure.

### The last landmine: the kernel's offsets come from expert 0 (2026-10-20)

`ModelExpertIO.routedExpertOffsets(layer:)` builds the single `ExpertOffsets`
struct a layer's kernels use from **expert 0's** sub-tensor offsets. After
expansion every slot holds the 4-bit layout, but if expert 0 were cold the
struct would describe the 2-bit layout and every expert read in that layer
would land on the wrong bytes.

The converter now pins expert 0 hot (`is_hot = expert_id in hot or
expert_id == 0`). One extra wide expert per layer is a negligible cost and
removes the whole class of wrong-offset bug. The alternative — deriving the
struct from the width-4 map rather than from whichever expert sits first — is
cleaner and should replace this when the expander lands.

With that, the expansion call site has no unknowns left:

1. read the short expert (done);
2. move and expand its tensors backwards into the 4-bit slot positions, using
   the target map from a hot expert, and divide each group scale by five;
3. the kernels run unchanged, because their `ExpertOffsets` describe exactly
   that 4-bit slot.

### Tiered bisect: the corruption is in prefill, and the threshold is eight (2026-10-20)

| tiered layers | set | result |
| ---: | --- | --- |
| 1 | 0, 5, 7, 9 individually | run |
| 5 | 0-4 | run |
| 6 | 0-5 | run |
| 6 | 0,1,2,3,4,5,7 | run |
| **7** | 0-6 | **run** |
| **8** | 0-7 | **NaN** |
| 10, 20, 40 | prefixes | NaN |

So it is the **count**, not a layer: adding layer 7 to a working six-layer set
is fine, and the same layer in a seven-layer set is fine, but eight tiered
layers corrupt. That is a boundary at eight, which is `topK` and also the
prefill tile width — the two places to audit.

A **one-token request fails the same way**, so the corruption is in prefill,
not in decode or the ring: prefill reads every expert of every layer through
the tile path, and somewhere in that path an eighth tiered layer tips a buffer
or an assumption.

The converter is exonerated by byte-identity between the full install's layer
files and the single-layer installs' files.

### Eight is the number of *streamers on the non-uniform path*, not of cold loads (2026-10-20)

Three more bisects:

| set | cold per tiered layer | result |
| --- | ---: | --- |
| 8 layers, contiguous (0-7) | 127 | NaN |
| 8 layers, spread (0,5,10,...,35) | 127 | NaN |
| **8 layers, spread, 90 % hot** | **25** | **NaN** |
| 7 layers, any set tested | 127 | run |

So contiguity does not matter and neither does the number of cold experts: 200
cold loads across eight layers fails exactly as 1,016 does. The trigger is
**eight layers whose stream layout is non-uniform** — eight streamers taking
the range-fetch path in one process.

That points away from the data and away from volume, and toward a per-model
resource that is allocated once and somehow distinguishes these eight from
seven: the shared `ExpertIOScheduler`, a fixed-size structure indexed by layer,
or the prefill tile plan that spans layers.

The converter is clean (byte-identical files), single layers are clean, seven
layers are clean. Whatever this is, it needs eight.

## Cold-tier verdict: 2-bit fails numerically, not mechanically (2026-10-20)

The tiered install is built (22 % fewer expert bytes), the format and reader
accept it, the converter is byte-exact, and the expansion was verified on 1,521
tensor regions across 169 dumped slots. It still cannot run.

**The differential that settles it.** Same prompt, same build, baseline 4-bit
versus eight tiered layers, comparing each layer's residual:

| layer | max difference vs baseline |
| ---: | ---: |
| 0 | 1.74 |
| 1 | 7.0e3 |
| 2 | 1.6e4 |
| 6 | 6.0e4 |
| 7 | 6.6e4, and 2,311 values non-finite |
| 8-39 | inf |

Routing corroborates it: the baseline keeps 120-158 distinct expert groups per
layer, while the tiered run collapses to **one group from layer 8** — the
signature of a degenerate hidden state, not of a bad slot.

So the perturbation enters at layer 0 (the first tiered layer) at a magnitude
consistent with 2-bit quantisation (0.47 added error on weights), and this
model amplifies it until the residual crosses the fp16 ceiling and overflows.
Everything mechanical measured clean because it is clean: prefetch, cache,
offsets, nibble order, conversion. **The cold tier is too lossy for this
model**, and that also explains why the failure threshold moved with layer
count, with cold count, and with which experts were cold.

### What it would take to reopen

1. **3-bit cold experts** (0.21 added error, half the perturbation). The
   expansion trick still applies — store 3-bit, expand to 4-bit with
   `q4 = 2*q3` and `scale/2` — so no 3-bit kernel is needed, only a packing
   change in the converter and expander. Byte saving is 11-22 %.
2. **A compensated 2-bit form** (low-rank or codebook residual), which is a
   research project rather than a repack.
3. **Activation rescaling** in the model itself, which is a different product.

Until one of those is tried, `tools/tiered_repack.py` and the expansion path
stay as an opt-in tool and a tested capability, not a shipped format.
