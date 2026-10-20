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
