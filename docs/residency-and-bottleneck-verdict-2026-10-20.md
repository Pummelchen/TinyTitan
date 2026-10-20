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
