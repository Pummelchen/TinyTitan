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

## What it would take to hide the rest

The rows for token N+1 are knowable as soon as token N is sampled — before the
next token's layer 0 runs. A prefetch issued there, into a double-buffered PLE
embedding, would overlap the remaining ~13 ms with the layer-0 attention
instead of paying it in front of layer 1. That is the next step; this change
is the cheap one that does not need a second buffer or a scheduler.
