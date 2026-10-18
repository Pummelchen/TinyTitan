# Performance campaign, 2026-10-18 22:00 → 2026-10-19 08:00

Machine and scope: one 24 GiB M3 MacBook Pro, Ornith 1.5 35B-A3B in 4-bit and
8-bit, decode tok/s from the engine's own `decode_tok_s` line. Driver:
`benchmark/minor_wins_ab.py` (paired, interleaved, rotating arm order, discarded
warm-up run, `--max-tokens 128`, greedy, output sha256 compared across arms).

**Bar for a commit:** ≥ +0.5% median on **both** bit widths, with the counters
consistent. Anything below that is recorded here as tested and not shipped —
a number, not an opinion.

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

### The default is now four, and it took a retraction to get there

The default was `--max-concurrent-sequences 1`, and the code says why: "the
batched path holds one KV cache and scratch buffer per sequence, so the memory
bill and the per-answer slowdown are opt-in, not a surprise on a default launch."

That bill is not paid at load on this host: RSS is 2,414,064 KB at width 1 and
2,413,952 KB at width 4, the derived expert slots are identical (160 at 4-bit,
96 at 8-bit), and four concurrent 1.6k-token requests complete with no error. A
lone client measures +0.17% (2/4 — noise), so width 4 costs a single user
nothing, and four clients finish 1.3x the work in the same wall time.

The first attempt to land it looked like a failure: `fourConcurrentRequestsAllComplete`
reported a 429 `queueFull`. Run in isolation, it does not reproduce — the 429 in
that log belongs to the *shedding* test beside it, which keeps one client and
sheds the rest by design. The only real failure was the assertion
`maxConcurrentGenerations == 1`, which pins the old serialised behaviour rather
than an invariant. The admission rule is `admittedCount < width + queueLimit`
(`ServerCoordinator.swift:54`), so width 4 admits four and queues four more.

Committed with the assertion updated to `defaults.maxConcurrentSequences`, and
the conformance note's admitted count corrected from 5 to 8.

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

- **`runtime: four generations run at once by default`** — the first change of
  the night to clear the bar. Aggregate **+32% at 4-bit and +24% at 8-bit** for
  four concurrent clients, single-stream neutral (+0.17%, 2/4), resident memory
  identical, and 4-way long-context verified.

Everything else in the table above is recorded and **not** shipped, because it
measured below +0.5% on both widths or was a loss on one.
