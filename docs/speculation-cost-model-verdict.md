# Speculation cost model: estimator, gate, and the missing sidecar (T28-T30)

Status: T28 measured; T29 rejected (no path to the required win); T30 incomplete
because no MTP sidecar is installed and the rules forbid installing one to
satisfy a benchmark.

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

## T30 — no sidecar to benchmark

`T30` names Ornith 4-bit at 64/128 tokens with the controller on/off. This
checkout has no MTP sidecar: `models/` contains only the target installs, and
`benchmark/tinytitan_benchmark.py` expects
`models/ornith-1.5_35B_A3B_MTP_4Bit`, which is absent. Installing a model is an
operator-requested job and AGENTS.md forbids doing it to satisfy a benchmark, so
there is no honest arm to run.

The prior measurements on the available target (Qwen3.8 4-bit) are the same
verdict T29's arithmetic gives: **−43.2%** at 85.7% acceptance, output identical
on the code-shaped scenario and divergent on one prose prompt. The task is
recorded **incomplete (asset-gated)**, not passed: a 64/128-token Ornith MTP
run needs the sidecar installed and would be the first thing to re-measure if it
appears.
