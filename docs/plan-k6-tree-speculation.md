# K6 phase A: tree speculation cost model and verdict

Status: **phase A complete — REJECT**. Tree decoding was not implemented, no
model or benchmark was run, and no engine code changed. The phase-A gate is the
project's +5% shipping bar: proceed to phase B only if the projected aggregate is
at least 1.05x on this M3 host. It is not.

Written 2026-10-24 on `main` at `af9ced3` (clean tree). The inputs are the
already-measured width-2 MTP path on this host; the tree terms are derived from
them, and every unmeasured quantity is labelled as such.

**Result.** With the measured 2.238x scalar-token cost of a two-row verify pass
and the measured 1.857 emitted tokens per pass at 85.7% acceptance, a tree is
projected at **0.830x** at W=2. The perfect-acceptance ceiling is **0.894x at
every width**, because a W-row pass costs `1.119 W` scalar tokens and can emit at
most W tokens. KFlash's claimed 8-17% more tokens per pass reaches 2.173 tokens,
still below the 2.238 break-even and the 2.3499 gate at W=2; at its own eight-row
width the projection is 0.243x. The gate stays off.

## 1. The question

The unit of account is one scalar decode token on the measured host: cost 1.0,
one token emitted. A speculative pass is worth building only when

    tokens emitted per pass / cost of the pass in scalar-token units >= 1.05.

T29 applied this to the linear width-2 MTP pass and got `1.857 / 2.238 = 0.830`.
K6 asks whether tree verification changes either term enough to clear the bar.
It changes the cost: a tree of W rows is W row-verifications, not one pass.

## 2. Measured inputs

All input figures in this section are from this repository's recorded
measurements. No number in this document was measured for K6 itself.

| Input | Value | Status | Source |
| --- | --- | --- | --- |
| Scalar decode token | 1.0 cost unit, 1 token | measured | Qwen3.8 4-bit, 256 greedy tokens, 2026-09-18 |
| Width-2 linear MTP pass | 2.238x scalar per pass | measured | same run, 86.9% acceptance, 1.869 emitted per pass; `speculation-cost-model-verdict.md` |
| Width-2 emitted tokens | 1.857 per pass at 85.7% acceptance | measured | Qwen3.8 4-bit, `--ram 8`, byte-identical function scenario, 2026-09-21; `qwen38-mtp-and-overlap-verdict.md` |
| 2026-09-21 run end to end | 2.175 vs 3.827 tok/s = 0.568x | measured | -43.2%; implies 3.267x per pass on that busy host state |
| Draft proposal phase | 15.1 ms per pass = 0.11x scalar | measured | Ornith 8-bit, 163 passes, 92.6% acceptance; `v4.4-decode-width-plan.md` B1 |
| Verify backbone phase | 305.9 ms per pass = 2.25x scalar | measured | same B1 table, against a 135.9 ms scalar token |
| Head plus commit phases | 18.4 + ~6 ms = 0.18x scalar | measured | same B1 table |
| T30 Ornith MTP, 64 tokens | 1.703 emitted per pass, 70.3%, 11.16 vs 11.76 tok/s | measured | 0.949x aggregate; `speculation-cost-model-verdict.md` T30 |
| T30 Ornith MTP, 128 tokens | 1.649 emitted per pass, 64.9%, 10.74 vs 12.21 tok/s | measured | 0.880x aggregate; same table |
| Host decode shape | 216.8 ms/token, GPU busy 102.4 ms, device busy 95.7 ms at 3.29 GB/s | measured | `qwen38-mtp-and-overlap-verdict.md` |
| KFlash tree | 8 verify rows, 8-17% more tokens per pass on short prompts | external, unverified | external README as stated in the K6 brief; not vendored or re-measured here |

The 2.238x and 1.857 come from different runs with slightly different acceptance
(86.9% and 85.7%). The brief mandates this pair. Using the self-consistent
2026-09-18 pair (1.869 at 2.238x) gives `1.869 / 2.238 = 0.835x`, still a loss;
the 2026-09-21 end-to-end run is worse (0.568x), not better. The verdict is
therefore insensitive to which measured pair is used.

## 3. The model

Let W be the number of rows verified in one pass. The measured linear pass has
W = 2 (`confirmed` and `draft`, one backbone traversal) and costs 2.238 scalar
tokens. The only per-row price these data support is the measured average:

    r = 2.238 / 2 = 1.119 scalar tokens per verified row.

A width-W tree verifies W rows, so with no batching discount

    C(W) = W * r + O_tree(W) = 1.119 W + O_tree(W).

`O_tree` is the tree attention and mask overhead. It is unmeasured; the model
sets it to 0, the most favourable value. Any real mask, gather or compaction cost
makes C(W) larger, so a rejection at `O_tree = 0` rejects every `O_tree > 0`.

**Why W rows, not one pass.** The measured 2.238x is already a two-row batched
pass, and it is 2.238x, not 1x. The measured phase split agrees with a roughly
linear per-row price: the width-2 verify backbone is 2.25x scalar (1.125 per row)
against a 1.0x scalar token. The only sublinear measured quantity is the
routed-expert union alone (1.585x for two rows on Qwen3.6, 0.79 per row), and it
excludes attention, the head, host work and the prefill-path penalty that the
2.238x whole pass includes. A batched tree would need a measured whole-pass
discount to price below W rows; none exists, so W is used.

**Drafting cost.** The 2.238x is the whole MTP "on" arm: proposal, checkpoint,
verify and commit/rollback are what the measured pass runs. The measured proposal
phase is 0.11x scalar per pass at width 2, so it is inside the 1.119 per-row
price and is not added again. A tree drafter that costs more than the linear
sidecar would add cost and lower the speedup, so this also treats drafting
favourably. A tree drafter costing a full scalar token per pass, about nine times
the measured 0.11x proposal, would move the W=8 KFlash projection from 0.243x to
`2.173 / (8.952 + 1.0) = 0.218x`, still nowhere near the gate.

**Emitted tokens.** A pass can emit at most one token per verified row. The
measured pair averages `1 + p` (p = 0.857 gives 1.857) for two rows, consistent
with row 0 always emitting its token and row 1's bonus being emitted when the
draft is accepted. At p = 1 two rows emit two tokens, which is T29's
`2 / 2.238 = 0.894` ceiling. In the measured accounting there is no extra token
beyond the rows, so at p = 1 a W-row pass emits W tokens, not W + 1. Therefore

    E(W) <= W.

This cap follows the measured `1 + p` accounting and T29's 0.894. As a
robustness check, if a future tree were credited with one bonus token beyond its
rows, the W=8 ceiling would be `9 / 8.952 = 1.005x`, still below the 1.05 gate;
only a two-row pass could clear under that accounting (`3 / 2.238 = 1.34x`), and
the measured two-row pass emits 1.857, not 3. The alternative accounting does not
rescue the tree either.

**Aggregate speedup.**

    S(W, E) = E / C(W) = E / (1.119 W)       (O_tree = 0)

Break-even is `E > 1.119 W`. The +5% gate is
`E >= 1.05 * 1.119 W = 1.17495 W`.

Because `E <= W`, `S <= 1 / 1.119 = 0.8937` at **every** width and **every**
acceptance. The measured per-row price is 11.9% above one scalar token, so the
pass cannot pay for itself even when every row is accepted. This is T29's 0.894
extended to trees: the width cancels.

## 4. Break-even and the +5% gate

| W | rows verified | C(W) scalar | break-even E (>) | gate E (>=) | max E (= W) | max S | gate shortfall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 2 | 2.238 | 2.238 | 2.3499 | 2 | 0.894 | 0.3499 tokens (14.9% of gate) |
| 4 | 4 | 4.476 | 4.476 | 4.6998 | 4 | 0.894 | 0.6998 tokens (14.9% of gate) |
| 8 | 8 | 8.952 | 8.952 | 9.3996 | 8 | 0.894 | 1.3996 tokens (14.9% of gate) |

The gate requires 1.175 emitted tokens per row; the structural maximum is 1.0.
The required E is 17.5% above the maximum at every W, and break-even itself
requires E 11.9% above the maximum. No tree width clears either line, so the
accepted-tokens-per-pass question has no solution under the measured per-row
price: W=2 needs more than 2.238 tokens from 2 rows, W=4 more than 4.476 from 4
rows, W=8 more than 8.952 from 8 rows.

## 5. Plausible emitted tokens per pass

The table prices several yield assumptions at each width. The "chain" rows are
the optimistic case where every row lies on the accepted path and acceptance is
the measured 0.857; a best-first tree's accepted path is shorter than W, so a
real tree is below the chain, not above it. The chain at W=2 is the measured
pair. KFlash's band is external and is not a measurement on this host.

| W | E scenario | E | C(W) | S = E / C | Status |
| ---: | --- | ---: | ---: | ---: | --- |
| 2 | measured linear, p = 0.857 | 1.857 | 2.238 | 0.830 | measured |
| 2 | perfect acceptance | 2.000 | 2.238 | 0.894 | derived ceiling |
| 4 | chain, p = 0.857 | 3.221 | 4.476 | 0.720 | assumed, optimistic |
| 4 | KFlash band, +8% to +17% | 2.006 to 2.173 | 4.476 | 0.448 to 0.485 | external plus assumed |
| 4 | perfect acceptance | 4.000 | 4.476 | 0.894 | derived ceiling |
| 8 | chain, p = 0.857 | 4.958 | 8.952 | 0.554 | assumed, optimistic |
| 8 | KFlash band, +8% to +17% | 2.006 to 2.173 | 8.952 | 0.224 to 0.243 | external plus assumed |
| 8 | perfect acceptance | 8.000 | 8.952 | 0.894 | derived ceiling |

KFlash's band is not a legal W=2 row: 2.173 exceeds the two-row maximum of 2. At
W=2 the only honest values are the measured 1.857 and the 2.0 ceiling, both
below 1.0x. At W=8, its own width, the band is 0.224 to 0.243x. The W=4 rows
are a transfer test only: the claim was made for eight rows, and four rows are
shown to make the point that the same yield costs less there but is still far
below the bar.

## 6. Does KFlash's 8-17% close the gap?

Break-even needs

    2.238 / 1.857 - 1 = 0.2052  ->  +20.5% more tokens per pass.

The +5% gate needs

    2.3499 / 1.857 - 1 = 0.2654  ->  +26.5% more tokens per pass.

KFlash claims +8% to +17%:

    1.857 * 1.08 = 2.0056 tokens per pass   (+8.0%)
    1.857 * 1.17 = 2.1727 tokens per pass   (+17.0%)

At the high end, 2.1727 is short of break-even by `2.238 - 2.1727 = 0.0653`
tokens (2.9% of 2.238) and short of the gate by
`2.3499 - 2.1727 = 0.1772` tokens (7.5%). In uplift points, +17% leaves 3.5 of
the 20.5 points needed for break-even and 9.5 of the 26.5 points needed for the
gate; at +8% the shortfalls are 12.5 and 18.5 points.

That comparison is deliberately generous: it holds the verify bill at the
measured two-row 2.238x while granting the tree its claimed yield. KFlash
verifies eight rows. At W=8 the bill is 8.952x and the same claim gives
`2.1727 / 8.952 = 0.243x` at the high end and 0.224x at the low end. At W=4 it
gives 0.485x and 0.448x. So the claim does not close the gap: under the most
favourable two-row cost basis it still misses even break-even, and under its own
eight-row cost it is a 4.1x slowdown.

The transfer is also doubtful for a non-arithmetic reason. Decode on this host is
expert-I/O and wait bound (95.7 ms of device busy against 83.1 ms exposed wait
per token); more verified rows add the union of their expert reads, and KFlash's
claim is for short prompts on a different engine.

## 7. Sensitivities

**7.1 The other measured host state is worse.** The 2026-09-21 run measured
0.568x end to end, which implies a 3.267x pass. Using 2.238x is the favourable
reading; the busier run gives 0.568 against 0.830.

**7.2 Ornith T30 is better and still loses.** Its implied pass costs are
`1.703 / 0.949 = 1.795x` and `1.649 / 0.880 = 1.875x`. The +5% cost ceilings at
those yields are `1.703 / 1.05 = 1.622x` and `1.649 / 1.05 = 1.570x`. Even the
best measured pass cost on this host is 9.6% above the cost the gate allows.

**7.3 Amortizing fixed costs does not rescue wide trees.** The B1 phase split
(Ornith 8-bit, whose pass total is 2.59x, so this is a cross-check rather than
the 4-bit anchor) prices the width-2 backbone at 1.125 per row and the head,
proposal and commit at 0.14, 0.11 and 0.04. If all three are treated as per-pass
fixed costs, then `C(W) = 1.125 W + 0.29`; W=8 costs 9.29, the chain yield gives
0.534x, perfect acceptance gives 0.861x, and the large-W limit is
`1 / 1.125 = 0.889x`. The measured marginal row is already above one scalar
token, so no amount of fixed-cost amortization reaches 1.0x.

**7.4 A hypothetical decode-native marginal is the only sub-scalar path.** The
only sub-scalar measured component is the routed-expert union, at 0.7925 per row
for two rows on Qwen3.6. A hypothetical pass of `0.30 + 0.7925 W` costs 6.64x at
W=8: the chain yield gives 0.747x, the KFlash high yield gives 0.327x, and only
perfect acceptance gives 1.205x. That is the retest target, not a measured
result: the whole pass, not the union component, has to reach that price.

**7.5 Tree overhead only lowers the number.** A 5% mask overhead on the W=8 pass
raises its cost to 9.40 and lowers the chain speedup to 0.528; 15% gives 0.482.

## 8. Verdict

**REJECT on paper. Phase B must not start.** No tree implementation, no engine
change and no model run.

| Projection | Aggregate | Gate |
| --- | ---: | --- |
| W=2 at the measured 1.857 tokens per pass | 0.830x | fails |
| W=4 at the optimistic chain yield 3.221 | 0.720x | fails |
| W=8 at the optimistic chain yield 4.958 | 0.554x | fails |
| Any W at perfect acceptance (the ceiling) | 0.894x | fails |
| KFlash's own eight-row yield, 2.173 | 0.243x | fails |
| T30 Ornith measured, best arm | 0.949x | fails |

The best number anywhere in the model is 0.894x, and it requires 100% acceptance
with the measured per-row cost. The measured projection is 0.830x. The +5% gate
would require a W-row pass to emit 1.175 tokens per row, which one-token-per-row
verification cannot do. The gate stays off.

## 9. Assumptions, measured versus assumed

| Assumption | Value | Status | Effect if wrong |
| --- | --- | --- | --- |
| Width-2 pass cost | 2.238x scalar | measured | The anchor; a lower cost helps, but the measured host already has this number |
| Width-2 emit rate | 1.857 at 85.7% | measured | The anchor; higher acceptance cannot exceed E=W |
| Per-row price r | 1.119x | derived assumption | Only W=2 is measured; the two rows are assumed representative |
| W rows cost W row-prices | C = 1.119 W | assumption | Only W=2 is measured; the measured two-row pass is already batched and still 2.238x |
| Tree mask overhead | 0 | assumption | Best case; any real overhead lowers S |
| Draft sidecar inside 2.238x | included | derived assumption | The measured on-arm includes proposal; a dearer tree drafter lowers S |
| E at most W | E <= W | derived | Follows the measured 1+p accounting and T29's 0.894 ceiling |
| Chain yield at p=0.857 | 3.221 at W=4, 4.958 at W=8 | assumption | Optimistic: every row on the accepted path; a tree's path is shorter |
| KFlash uplift | +8% to +17% | external, unverified | Not re-measured; if larger, Section 6 says how much larger it would have to be |
| Host decode is device bound | 95.7 ms device busy per token | measured elsewhere | More rows add union I/O; this is not a compute-bound GPU |

One contradiction is disclosed rather than hidden: 2.238x (2026-09-18) and 0.568x
end to end (2026-09-21) do not imply the same per-pass cost (2.238 vs 3.267). The
later run was on a busy machine and its absolute rates were marked indicative;
the model uses the cleaner 2.238x as the brief requires. Both readings reject,
so the contradiction does not change the verdict.

## 10. What would have to be true for a future retest

A retest is worth one run only if it can plausibly clear `E / C >= 1.05` with a
measured W-row pass. The concrete bars:

1. **A measured whole-pass cost at the achieved yield.** At the current
   `E = 1.857`, the pass must cost at most `1.857 / 1.05 = 1.769x`, a 21.0% cut
   from 2.238x. At perfect acceptance (`E = 2`) it must cost at most 1.905x, a
   14.9% cut. For the W=8 chain yield (`E = 4.958`) it must cost at most 4.722x,
   47.3% below 8.952x; for KFlash's 2.173 yield it must cost at most 2.069x,
   76.9% below 8.952x.
2. **A per-row price at or below 0.952x scalar at perfect acceptance.** That is
   the gate condition when every row is accepted; at the measured acceptance the
   bar is lower, for example 0.590x per row for the W=8 chain, or 0.259x for
   KFlash's eight-row yield. The measured whole-pass price is 1.119x.
3. **A faster verify path, measured as a whole pass.** The named cost terms are
   the prefill-path verify (1.6-1.7x where decode is 1.0x) and the per-pass host
   and commit work. A decode-native verify must take the two-row pass from
   2.238x to at most 1.769x at the measured yield, or to 1.905x only if
   acceptance is perfect.
4. **Higher acceptance alone cannot do it.** At W=2 the gate needs `1 + p >=
   2.3499`, so p >= 1.35, which is impossible; at any W the ceiling is 0.894. Any
   retest that only raises acceptance is not a retest of this result.
5. **Batched rows must be measured, not assumed.** A W=4 or W=8 whole pass has to
   show its own cost curve. The union component is sublinear, but the whole pass
   is not the union. The phase footer (`proposal`, `checkpoint`, `verify`,
   `verify_backbone`, `verify_head`, `verify_argmax`, `commit`, `rollback`)
   already records what a retest must report, per pass, at each width.
6. **A tree shape that actually reaches the row cap.** KFlash's eight rows at
   2.0 to 2.2 tokens are 0.25 to 0.27 tokens per row, far from the 1.175 the gate
   needs at the measured price. A best-first tree whose accepted path is not
   close to the number of rows cannot clear the gate even if verification gets
   cheaper.
7. **Quality and protocol.** Byte-identical greedy output on the text battery,
   the scenario hashes, paired interleaved arms, at least three rounds above the
   measured 1.7-5.7% noise floor, at 64 and 128 new tokens, with the cost and
   acceptance reported per pass rather than only tok/s.

The single sentence version: a retest needs a measured verify pass whose
marginal row costs less than one scalar token, and a tree that accepts nearly
every row on one path. Neither has been measured, and the second is far from
KFlash's reported yield.

## 11. Provenance

- No model or benchmark was run for this document, and no engine code changed.
- The measurement inputs are the records cited in Section 2; the arithmetic was
  checked with a standalone calculation, not a harness.
- There is no docs index in this repository: `docs/repository-layout.md` and
  `AGENTS.md` define the docs conventions but no index file, and open work lives
  in the wiki Project Tracker. This file is linked from
  `docs/speculation-cost-model-verdict.md` so the K6 extension is discoverable
  beside the linear result.
- Lint: `tools/lint.sh docs`.
