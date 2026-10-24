# Speculation cost model: estimator and gate (T28-T30)

Status: T28 measured; T29 rejected (no path to the required win); T30 now
measured and rejected — the Ornith MTP sidecar was installed from its pinned
Hugging Face source and the 64/128-token arms lose.

## T28 — distinct-expert growth estimator

`benchmark/t28_distinct_expert_estimator.py` reads a
`TINYTITAN_ROUTE_TRACE` (`position layer e0 ...`) and predicts the union of the
next `W` routes per layer. The first attempt used a Good-Turing singleton
estimate; 70% into a long trace most experts are still singletons, so it
predicted a near-worst-case window and scored barely better than "topK,
no growth". The shipped estimator uses the signal a draft window actually has —
the previous route:

    new per token = mean |route_t \ route_{t-1}|
    predicted(W)  = topK + (W - 1) * new per token    (capped at experts/layer)

Evaluated on `benchmark/expert-cache/route-trace-qwen38-4bit-20261011.txt.gz`
(48 layers, 413 positions, topK 10, first 70% fit, rest scored):

| window | predictions | estimator MAE | topK baseline MAE | topK*W bound MAE |
| ---: | ---: | ---: | ---: | ---: |
| 2 tokens | 5,904 | **1.881** | 6.063 | 3.937 |
| 4 tokens | 5,808 | **3.726** | 16.532 | 13.468 |

The error is in distinct experts, so window 2 is within two experts of the
actual union on a 10-expert route. The worst layers are L42/L45/L46 at ~2.5 for
window 2. The suite `benchmark/test_t28_distinct_expert_estimator.py` pins the
parser and both the overlapping and repeated-route cases.

This is a measurement tool, not a runtime dependency: it adds no per-token work
to the engine.

## T29 — the gate cannot clear its own bar

The task asks for a controller that enables speculation only on a >5% win,
using predicted distinct-expert growth plus acceptance. The measured cost of
the verify path settles it before the predictor is consulted:

- width-2 MTP emits **1.857 tokens per target pass** at **85.7% acceptance**;
- the same pass costs **2.238x a scalar token** (`docs/qwen38-speculation-and-compression-verdict.md`,
  and `docs/minor-wins-accounting.md` records the end-to-end **−43.2%**).

Even at **100% acceptance** the best possible ratio is `2 / 2.238 = 0.894`, a
10.6% loss before the draft and commit work is counted. Distinct-expert growth
can only add expert I/O on top of that; it cannot lower the verify-pass cost
below 1.86x. A gate that requires +5% therefore never fires on this engine, and
one that fires anyway would be shipping the measured loss.

Decision: **rejected**, and no controller code is kept. The objective is to
keep only code that improves TinyTitan's performance; a gate whose only
reachable state is "off" is dead code, and the measured evidence belongs in this
document rather than in the hot path.

## T30 — measured, and it loses

The sidecar is available after all. `tools/install_models.sh ornith15-mtp`
ranges the pinned `ornith-ai/Ornith-1.5-35B-A3B` revision
`e4dfb35a93d4b6822a811a7676f3488514abe7e2`, verifies shard 16 (4.18 GB) and
installs a 43 MB sidecar at `models/ornith-1.5_35B_A3B_MTP_4Bit` with a
verified receipt.

`benchmark/t30_mtp_ab.py` then ran three requests per arm, pure greedy, cache
off, Ornith 4-bit, at 64 and 128 new tokens:

| tokens | arm | aggregate tok/s | paired median ratio | acceptance | emitted/pass |
| ---: | --- | ---: | ---: | ---: | ---: |
| 64 | scalar | 11.76 | — | — | — |
| 64 | MTP | 11.16 | 0.858 | 70.3% | 1.703 |
| 128 | scalar | 12.21 | — | — | — |
| 128 | MTP | 10.74 | 0.866 | 64.9% | 1.649 |

Text hashes are identical across all six requests per token count
(`856f0857dfc36dc8` at 64, `4e5fd441610d4b39` at 128), so this is a speed
loss, not a quality trade. MTP is −5.1% aggregate at 64 and −12.0% at 128;
the paired medians are −14.2% and −13.4%. No arm clears the +5% gate.

This is the same conclusion T29's arithmetic predicted: at 70% acceptance the
verify pass has to cost less than ~1.7× a scalar token, and the measured
width-2 path costs 2.238×. The controller remains unwired; the sidecar is
installed for a future verify-path rewrite, not because it is enabled.

## K6 tree extension

[plan-k6-tree-speculation.md](plan-k6-tree-speculation.md) (2026-10-24) extends
this model to tree widths
2/4/8. A W-row pass is priced at W row-verifications from the measured 2.238×
two-row pass (1.119× per row, tree mask overhead set to 0 as the best case).
The projection is 0.830× at the measured 1.857 emit rate, and the
perfect-acceptance ceiling is 0.894× at every width because a pass can emit at
most one token per verified row. Trees are rejected on paper before any
implementation; the document records the break-even bars a future retest would
have to measure.
