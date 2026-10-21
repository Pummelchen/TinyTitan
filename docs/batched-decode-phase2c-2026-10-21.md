# Phase 2c batched decode: the four-row head, and the lockstep attempt

**Status (2026-10-21): the four-row head ships for affine heads; the full
lockstep layer loop was built, measured, and reverted.** This note records both
halves so the next attempt starts from the measurements rather than from the
plan's estimate. Nothing here changes the width-1 path or the width-2 head.

## What shipped: a four-row head for affine heads

`produceBatch` already stopped each row after its final norm and sent two rows
through the two-row head with one weight read. At width 4 that is two two-row
launches, so the head's weights are read twice. The affine head now has a
four-row kernel (`affine_quant_gemv4_simd`, `AffineQuantGEMV.encodeFourRows`);
`encodeBatchHead` uses it while the batch has four rows and falls back to the
two-row pair / single-row GEMV otherwise.

The four-row kernel computes each row with the two-row kernel's accumulation
order, so a width-4 batch's numbers are **exactly** what the two two-row
launches produced. `fourRowsMatchFourSingleRows` asserts both properties: exact
equality against two two-row launches, and the two-row kernel's reduction
tolerance against four single-row calls.

### Why the int4 head did not get one

The installed 4-bit checkpoints have an int4 head (`affineHead` is nil, the
`dequant_int4_gemv` kernels serve the head). A four-row int4 kernel was built
to the same shape. It is **not** shipped: at the 248,320 x 2,048 head it costs
about the two-row launch for twice the rows, so two pairs are as fast and the
extra kernel only risks the 4-bit path.

Kernel-only, `TinyTitanBench`, 200 iterations, three order-rotated runs,
medians (M3 24 GiB; the host was carrying ~7.6 GiB of swap during the campaign,
so absolute values are lower than a quiet machine's, but the arms ran under the
same condition):

| head | 1 row | 2 rows | 4 rows | per-row at 2 | per-row at 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| int4, shipped | 3.55 ms | 3.86 ms | 7.52 ms (not shipped) | 1.93 ms | 1.88 ms |
| affine 8-bit, shipped | 6.11 ms | 6.66 ms | 8.88 ms | 3.33 ms | 2.22 ms |
| affine 4-bit, probe | 5.74 ms | 6.78 ms | 8.96 ms | 3.39 ms | 2.24 ms |

The 8-bit head is the case that wins: four rows in one launch cost 8.88 ms
against 13.32 ms for the two two-row launches they replace, so a width-4 step
reads the head once instead of twice and saves 4.4 ms (33 % of the head stage).
The 4-bit int4 head is already near the memory ceiling at two rows (70-89 GB/s
of the M3's ~100 GB/s); adding two more rows' dequant + dot turns it
compute-bound and buys nothing. The 4-bit affine kernel is slower per row than
int4 to begin with, so it is not a substitute.

The four-row affine path could not be measured end to end on this machine: the
only 8-bit install is the Ornith 35B at 34 GB, larger than the 24 GiB host.
The kernel measurement is the evidence; a host with a 32/64 GB machine should
re-run the width-4 server pair on an 8-bit install before quoting an
end-to-end number.

## What was built and reverted: the lockstep layer loop

The plan's remaining Phase 2c is "carry B rows through the dense projections,
the FFN and a union expert fetch". A first implementation was built end to end:

- `produceBatch` chose a lockstep path when the family had no PLE, QSA,
  hyper-connections or activation dumps (Ornith and the toy qualify; Qwen3.8
  does not, because its indexer and n-gram latch are per-runner, not per-slot).
- A B-row `PrefillChunkScratchBuffers` held the residual, block inputs, attention
  output and FFN scratch.
- Each layer ran the entry norm for all rows, then each row's existing decode
  attention / GDN step with row-sliced input and output buffers (new optional
  `input`/`output` parameters on the decode attention encoders), then the
  post-attention norm for all rows, then the **prefill** FFN for all rows:
  `encodeRoutedMoEPrefill` already groups per-row routes into one union fetch
  and a grouped MoE pass, and `encodeDenseFFNPrefill` covers dense families.
- The final norm ran for all rows and the existing head pairs wrote the logits.

Gates on the toy fixture (`QwenRunnerTests`): `produceBatchMatchesPerRowLogits`
and a new width-4 test passed at 4 and 8 bits, and the lockstep path was
**bit-identical** to per-row `produce` (max abs logit difference 0.0 over three
steps per row). On the real Ornith 4-bit server, width 2, 64 new tokens, two
prompts:

```
serial  (width 1):  9.21 s + 6.11 s = 15.31 s
batched (width 2):  wall 15.21 s, per request 15.06 s / 15.20 s
```

Two things failed at once:

1. **The output text diverged** from the serial arm on both prompts, after the
   first sentence. The toy's bit-identity does not survive the real model; the
   likely cause is the batched prefill RMS norm (D=2,048 is many 256-thread
   blocks, while the toy's D=64 is one) and the prefill FFN's different
   reduction paths. This is a quality regression, not a speed trade the
   project accepts.
2. **There was no throughput win.** The batched wall time equalled the serial
   sum: the prefill FFN pipeline is roughly twice as expensive as two decode
   FFNs because `encodeRoutedMoEPrefill` commits and waits at the router and at
   every expert tile, losing the decode path's cross-layer prefetch ring and
   asynchronous routed command. Sharing the expert bytes did not compensate.

Both are fatal for shipping it. The path was removed rather than left behind an
environment flag: the default is byte-identical and fast, and a disabled
second implementation of the layer loop is a maintenance and correctness risk
with no measured upside. The attempt is preserved as a patch in the round's
scratch space, not in the tree.

## Second attempt: batched QKV/o_proj, still no win

A narrower attempt batched only the attention-side dense projections at width
2: both rows' entry norms went into one buffer, one two-row GEMV produced QKV
for both, one two-row GEMV produced o_proj from both rows' post-gate attention,
and each row then ran its own residual + decode FFN. The per-row attention
epilogue and FFN were kept, and the layer body could resume at the residual
when the batch had already run attention.

Correctness held: the two prompts produced the same text as the serial arm
after the row-residual handoff was fixed (the routed-MoE tail had been adding
row 0's delta to the runner-wide `hidden` instead of the row's buffer). That is
the first time a lockstep variant matched serial text on Ornith.

It was still rejected on speed. Ornith 4-bit, width 2, 64 new tokens, the same
two prompts:

```
serial  (width 1):  10.66 s + 6.79 s  (18.39 and 17.29 decode tok/s)
batched (width 2):  wall 19.31 s      (7.03 and 4.83 decode tok/s)
```

The projection batch saved weight reads but the loop waited on each row's
routed command before the next row's layer, so the decode path's cross-layer
prefetch ring and asynchronous routed command were lost. The wait cost was
larger than the projection saving. The attempt was reverted rather than left
behind a flag, for the same reason as the first one; the diff is preserved in
the round's scratch space.

A second refinement gave each row its own routed-MoE scratch (`routedX`,
`h1`, `h2`, `moeActs`, router outputs, shared-expert scratch) and carried the
pending routed commands across layers instead of draining at each layer
boundary. That made the batch numerically correct again -- and exposed one
final shared buffer, the phase-1 `routedX`/`moeActs`, which had stayed
runner-wide. With the whole per-row set, Ornith 4-bit width 2 produced the
serial text.

It was still rejected. Warm 64-token pairs were 13.75-14.18 s batched against
14.44-14.85 s serial, a small win from overlapping one row's prefill with the
other's decode; at 128 tokens the same pair was 22.65 s batched against
21.41 s serial. Decode throughput was 8.1 and 10.1 tok/s against 17-18 for
serial. The layer-boundary synchronization that makes the QKV/o_proj batch
possible is exactly what removes the per-row cross-layer routed overlap; the
short-generation wall win does not survive a longer generation, so the path
was reverted.

## What the measurements say the next attempt must do

- **Give each row its own routed-MoE scratch and carry its pending command
  across layers.** The second attempt proved the projection batching and the
  per-row FFN can be numerically exact; what it could not do was keep the
  decode path's overlap while two rows shared `routedX`, `h2`, `moeActs` and
  the router buffers. A B-row set of those is the prerequisite for a fair
  test.
- **Reuse the decode kernels, not the prefill ones.** The lockstep win has to
  come from sharing a weight read while every row keeps the arithmetic and the
  pipeline the width-1 path uses. `encodeTwoRows` proved that is possible for a
  GEMV; the FFN needs the decode path's asynchronous routed command and
  prefetch ring, which means extracting the decode FFN stage into a per-row
  function rather than calling the prefill MoE.
- **Batch the attention-side dense projections first.** QKV and o_proj are
  about 400 MB of weights per token on Ornith and already have a two-row
  kernel. Sharing them for two rows is a smaller, byte-identical slice with a
  clear measured target; it needs the lockstep loop and a split between the
  projection and the attention proper, not a new numerical path.
- **Keep the width-1 and width-2 gates.** `produceBatchMatchesPerRowLogits`,
  the width-4 test and the slot-independence test are the harness; the real
  model adds the byte-identical text check the toy cannot replace.

## Gates run for this piece

- `swift test --no-parallel --filter 'fourRows|produceBatch'`: 3 tests / 2
  suites, 0 failures (four-row kernel at 4 and 8 bits, width-2 and width-4
  batch parity).
- `swift test --no-parallel`: full suite, 0 failures (run before the
  lockstep revert and again after the four-row change).
- `tools/lint.sh`: all gates.
- `swift build -c release` with `assert_clean_build_log`: no compiler warnings.
