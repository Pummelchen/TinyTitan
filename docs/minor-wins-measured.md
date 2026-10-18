# Minor wins, measured on Ornith 35B 4-bit and 8-bit

2026-10-18. Answers `docs/minor-wins-accounting.md` with measurements instead of
history. Driver: `benchmark/minor_wins_ab.py`. Raw rows:
`/tmp/titan-ab/mw-ornith*.json` (not committed), server logs
`.build/minor-wins-ornith*.log`.

## Method

Paired and interleaved: every round runs the base arm and each candidate arm
back to back, and the round's delta is `(arm - base) / base`. The arm order
rotates each round, because the **first server of a session is ~16% slower** than
the ones after it (cold page cache over the 3.4 GB of weights) — larger than any
effect here, and a discarded warm-up run precedes round 1. One greedy request per
run, `--max-tokens 128`, fresh server per arm (the knobs are read at start),
`TINYTITAN_RUNNER_STATS=1` for the counters. Output text is compared by sha256
across arms: every arm here produced **byte-identical text** (1 distinct digest
per sweep), so none of this buys speed by changing the answer.

What is testable today and what is not:

| item from the accounting doc | status |
| --- | --- |
| Page-cache path | env `TINYTITAN_BOUNDED_IO=0` — tested |
| Read advice off | env `TINYTITAN_RDADVISE_POLICY=off` — tested |
| Immediate submission | env `TINYTITAN_EXPERT_IO_SUBMISSION=immediate` — tested |
| Stack | page cache + immediate — tested |
| Eviction `lru` | **no env reader any more** (default `.lfu`, the switch was deleted) — needs code |
| Early hits | **deleted from the engine** (0 references) — needs the phase-1 path rebuilt |
| Serial fills | **deleted**, a comment survives — needs rebuilding |
| Prefill→decode warm-up | **never built** — see "not done" below |

## Results

Paired delta vs base, median (and positive rounds). 4-bit default = 4 rounds of
all arms; 4-bit focused and 8-bit focused = 6 rounds of base vs stack; 8-bit
default = 3 rounds of all arms.

| improvement | Ornith 4-bit | Ornith 8-bit | verdict |
| --- | ---: | ---: | --- |
| **Page cache** (`BOUNDED_IO=0`) | **+0.5%** (2/4) | **+5.2%** (2/3) | pays at 8-bit only |
| Read advice off | **−4.7%** (1/4) | **−3.2%** (1/3) | a loss on both |
| Immediate submission | **−6.5%** (0/4) | +0.1% (2/3) | loss at 4-bit, wash at 8-bit |
| **Stack** (page cache + immediate) | −0.7% (2/6) | **+4.7%** (5/6), mean +5.6% | pays at 8-bit only |
| Stack at a tight 6 GB budget | inconclusive (base swung 9.27-17.23 tok/s) | — | not measured |

The 8-bit stack is the only signal with the shape of a result: 5 of 6 paired
rounds positive, and the two medians across six rounds are 10.259 vs 9.848 tok/s.

## The mechanism, in the counters

| arm | tok/s (med) | `io_ms` (med) | hit rate |
| --- | ---: | ---: | ---: |
| 8-bit base | 9.848 | **33.99** | 0.8507 |
| 8-bit stack | 10.259 | **27.39** | 0.8579 |
| 4-bit base | 17.147 | 11.75 | 0.8871 |
| 4-bit stack | 15.844 | 11.09 | 0.8874 |

The 4-bit row pair is raw medians, and it overstates the loss: that sweep's base
arm itself ranged 14.13-17.46 tok/s across rounds, so the paired statistic
(−0.7%) is the one to quote. The io_ms medians are stable in both arms, which is
the part this table is for.

`io_ms` falls 19% at 8-bit with the hit rate essentially unchanged — the page
cache is serving demand reads the counters still count as misses. That is exactly
what Lever 4 described on Qwen3.8, and it reproduces here on a different family.

It also explains the bit-width split: 8-bit moves **twice the bytes per expert**,
so the demand path has real work for a cache to absorb; 4-bit does not, and the
page-cache path's own cost cancels the gain. The historical Qwen3.8 measurements
were 4-bit, at +4-5% — so this is a *pressure-dependent* effect, not a constant.

## What this means for the accounting doc

- The page-cache trade is real and reproducible on a second model, but only where
  expert traffic is heavy. It remains a **footprint decision**, not a bug: the
  measured +4.7% still costs an undeclared memory footprint.
- `advice-off` and `immediate` **do not reproduce as wins** on this model at
  either bit width. The history's +1.15% and +1.29% were 4/4 and 3/4 positive on
  Qwen3.8; here they are −4.7% and −6.5% at 4-bit. Treat those two entries in the
  accounting table as **not wins**, and the compounding ceiling drops by ~2 points.
- The rest of the accounting table's small items are unreachable without restoring
  deleted code, and the two that were deleted (early hits, serial fills) are now
  the *least* attractive: the family they belong to measured flat-to-negative
  here.

## Not done, and why

- **LRU** — the env reader was deleted; restoring it is a two-line change, but the
  policy family measured flat here, so it would buy another measurement rather
  than a win. Cheap to add if wanted.
- **Early hits (+2.22%, 1/2) and serial fills (+0.4%, 2/3)** — these need deleted
  kernels and paths put back. Rebuilding them to chase an effect that did not
  reproduce in the neighbouring arms is the wrong order of work.
- **Prefill→decode warm-up** — the largest unbuilt item, but it needs a long
  prompt (the cold-cache window is the first 64 decode tokens *after* a long
  prefill) and a cache-policy change inside the streamer. It is not measurable
  with this driver, which uses a short prompt, and it is a design change rather
  than a knob. This is the one I would build next, with a long-prompt harness.

## Caveats

Each run is a fresh process, so per-run load time is excluded from `decode_tok_s`
but page-cache state is not perfectly controlled; the discarded warm-up run and
the rotation mitigate that, they do not eliminate it. Two runs in the set are
visibly contaminated by machine transients (an advice-off run at 3.165 tok/s
against a ~9.8 base, and the whole 6 GB sweep), and they are reported rather than
dropped. The 8-bit effect is +4.7% on medians over six rounds; that is evidence,
not a guarantee, and it is measured on one machine at one prompt length.
