# The PLE decode gather runs on eight readers

**Status (2026-10-21): shipped, measured on the 125B.** Project Maya's PLE
reader fans a token's n-gram row reads across workers; TinyTitan's prefill
already did (`gatherConcurrently`), but the decode path called the serial
`gather` for its 16 rows. On the 125B that was **14–16 ms of every token**, the
same disk that is feeding the routed-expert stream, and the rows are
independent: 16 random reads whose only ordering requirement is the order they
land in the destination.

## Change

`NgramTableReader.gatherConcurrentlyRows(rows:into:threads:)` splits one
position's rows across up to eight readers, each writing its own destination
rows. `RealForwardRunner.gatherPLERows` calls it with
`TINYTITAN_PLE_GATHER_THREADS` (default 8; `1` restores the serial arm for
A/B). No engine math changed: the same rows land in the same slots.

## Measured

M3, 24 GiB, `qwen3.8-flash-next_125B_A6B_4Bit` (fp16 table), 1,691-token
prompt, 32 greedy tokens, `--max-context 4096 --prompt-cache-mode off`,
`TINYTITAN_RUNNER_STATS=1`, one server per arm, arms alternated:

| arm | gather ms/token | body ms/token | decode tok/s |
| --- | ---: | ---: | ---: |
| serial (threads 1), run 1 | 15.80 | 247.4 | 3.756 |
| **8 readers, run 1** | **13.16** | 243.1 | **3.819** |
| serial, run 2 | 14.98 | 239.8 | 3.873 |
| **8 readers, run 2** | **13.13** | 237.4 | **3.944** |

The gather is **14 % cheaper** and decode **+1.7 % at the median**
(3.815 → 3.882 tok/s); both pairs move the same way. The gain is smaller than
the serial read count suggests because the disk is already saturated by the
expert stream, so the concurrent reads queue rather than overlap freely. The
output is unchanged: the same 32 tokens (the gather only reads).

## Prefetch-ahead: tried, does not clear the noise

The rows for a token are known before its first layer runs, and the PLE layer
is layer 1, so the gather was moved onto a background queue at the start of
`produceToken` and awaited only at layer 1 (the same build, arms alternated,
`TINYTITAN_PLE_PREFETCH=0` as the control):

| arm | exposed gather | decode tok/s |
| --- | ---: | ---: |
| off, run 1 | 13.37 ms | 3.803 |
| **on, run 1** | 12.71 ms | 3.830 |
| off, run 2 | 13.16 ms | 3.846 |
| **on, run 2** | 12.62 ms | **3.978** |

The exposed gather barely moved (13.2-13.4 → 12.6-12.7 ms) and decode moved
3.825 → 3.904 tok/s at the median — one pair +0.7 %, one +2.9 %, inside this
host's ~3 % noise. The disk is the shared bottleneck: starting the read earlier
cannot create bandwidth while the routed-expert stream is already saturating
it, and the gather's concurrent reads compete with layer 0's expert reads. The
prefetch was reverted; the fan-out is the part that pays.
