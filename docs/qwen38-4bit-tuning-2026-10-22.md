# Qwen3.8-Flash-Next 125B-A6B 4-bit: tuning round (2026-10-22)

Status: one change kept — the 4-bit profile's prefill chunk is **2,048** instead
of 4,096. Every other runtime knob tested here measured a wash or a loss and
was left alone. All arms used the installed
`models/qwen3.8-flash-next_125B_A6B_4Bit` on the 24 GB M3, one process at a
time, with `TINYTITAN_RUNNER_STATS=1`.

The machine moves the same configuration by ±15% within a session, so the
arms below are paired and interleaved, and the direction is what is claimed,
not a single number.

## Expert-cache slots: keep 96

`--expert-cache-slots` at 64 / 96 / 112, 64 new tokens.

| workload | 64 slots | 96 slots | 112 slots |
| --- | ---: | ---: | ---: |
| 9-token raw prompt | 4.62 / 4.68 | 4.29 / 3.97 | 1.14 / 2.28 |
| ~200-token chat | 4.90 / 5.40 | 5.46 / 5.66 | — |
| 851-token chat | 4.86 / 4.56 | 5.83 / 4.90 | — |
| 3,942-token prompt | 3.86 | 4.49 | — |

The 9-token prompt alone prefers 64 because it pays 12 GiB of wired cache for a
tiny KV; every realistic chat prompt prefers 96, and 112 slots pages
catastrophically. `qwen3.8-flash-next` 4-bit keeps its 12 GiB / 96-slot row.
The two prior notes that made 64 the shipped value were measured on different
prompts; at the product shape (system + user turn) 96 wins, so the row is not
changed.

## Runtime flags: all left at their profile values

| arm | chat decode (two rounds) | verdict |
| --- | ---: | --- |
| shipped | 5.58 / 6.09 | — |
| `TINYTITAN_KEEP_WIRED=0` | 5.14 / 5.26 | **loss**, keep wired |
| `TINYTITAN_HC_FUSED=1` | 4.76 / 5.99 | wash/loss, keep off |
| `TINYTITAN_QSA_GPU_SELECT=1` | 5.87 / 5.84 | wash, keep off |
| `TINYTITAN_EXPERT_USE_CAP=3` | 5.53 / 5.59 | **−4%**, keep uncapped |
| `--rdadvise off` | 4.08 / 4.24 | **−11 to −14%**, keep advice on |
| `TINYTITAN_BOUNDED_IO=0` (page cache) | 4.69 / 4.84 | wash against 4.58 / 4.96, keep bounded |

The read-advice result is the notable one: the 24 GB 4-bit install still pays
the advice pass (9-10 ms/token), but switching it off costs more than it saves
(228 ms body against 201 ms), so the earlier "advice is dead work" note does
not generalize to this model/prompt. The page-cache arm is a wash here, not a
win; the install is four times RAM.

## Kept: prefill chunk 2,048 for the 4-bit row

The 4,096-token prefill reservation is held through decode and costs it.

| workload | 2,048 chunk | 4,096 chunk | prefill |
| --- | ---: | ---: | --- |
| ~200-token chat, 64 new | 6.11 / 6.79 | 5.58 / 6.09 | 11.4-11.9 s both |
| 851-token chat, 64 new | 5.51 / 4.90 | 4.77 / 4.65 | ~30 s both |
| 851-token chat, 8 new | 4.55 / 5.03 | 3.05 / 4.40 | ~30 s both |

The 851-token row is the interleaved A/B: prefill is unchanged at ~30 s and
decode is 5-16% faster. A 200-token chat repeat agrees in direction (+4-15%).
A 3,942-token prompt measured 28.5 tok/s prefill at both sizes, so the shorter
chunk does not cost prefill until the prompt is long.

The trade is documented in `defaultPrefillChunkTokens` already: on a 10k
prompt, 4,096 is ~11% faster to prefill (450 s against 506 s) and a 512-token
generation only saves ~2 s. That is why the **8-bit** row and the family
fallback keep 4,096; only the 4-bit row moves. An operator running 10k prompts
on the 4-bit install should pass `--prefill-chunk 4096`; a short-prompt,
long-generation workload is the case this change serves.

Verification:

- `tools/golden-baseline.sh --check qwen38-4` — **output identical to
  baseline**. The shorter chunk changes reduction boundaries but not the
  generated text.
- `ModelProfileTests`, `DefaultPrefillChunkTests`, `ExpertCacheBudgetTests` —
  27 + 2 tests pass.

## Rejected during the round

- **Chunk 1,024 vs 2,048** on the 851-token prompt with 64 new tokens:
  5.83/5.40 against 5.40/5.70, a wash (+1% at best) with the same prefill.
  2,048 stays; 1,024 buys nothing and doubles the chunk count on long prompts.
- **Releasing the prefill scratch before decode**
  (`TINYTITAN_RELEASE_PREFILL_SCRATCH=1`) was tried to keep 4,096's prefill
  and 2,048's decode. It lost in both rounds (3.82/5.77 tok/s at 2,048 against
  6.67/5.30 without; 3.19/3.86 at 4,096), and the experiment was reverted
  rather than shipped. Freeing and reallocating the Metal scratch churns the
  allocator more than the memory it returns.
- **Chunk 512** prefilled the 851-token prompt in 38.9 s against ~30 s for
  1,024-4,096, so it is out on prefill alone.

Only the chunk value changed; no new switch was kept. The measurement scripts
and raw logs are under the git-ignored `.build/` (`qwen38-chat-slots.sh`,
`qwen38-chunk64-ab.sh`, `qwen38-chunk1024-ab.sh`, `qwen38-cap-ab.sh`, and the
per-arm logs).
