# Qwen3.8 4-bit decode at 12 GB — measurements

Stage-0 record for `docs/plan-qwen38-decode-12gb.md`. Machine: MacBook M3
24 GB, macOS 26, Xcode 27 / Swift 6.4, release build, bounded pread reader.
Every number below was measured 2026-10-11; commands and exit codes included.

## 1. The capture

Model run (window free: `pgrep` guard empty, `memory_pressure -Q` 77% free,
install `models/qwen3.8-flash-next_125B_A6B_4Bit`):

```
TINYTITAN_ROUTE_TRACE=/tmp/qwen38-route.trace \
  .build/release/TinyTitanCLI \
  --model models/qwen3.8-flash-next_125B_A6B_4Bit \
  --prompt "Write a detailed essay about the history of computing, …" \
  --max-new 384
# exit 0; footer: prefill 30 tok / 7.17 s, decode 88.24 s, tok/s=4.352
```

Trace: 19,824 lines = 413 positions (30 prefill + 383 decode) × 48 layers ×
top-10 experts. Committed as
`benchmark/expert-cache/route-trace-qwen38-4bit-20261011.txt.gz` so every
figure below re-runs without a model:

```
gunzip -c benchmark/expert-cache/route-trace-qwen38-4bit-20261011.txt.gz > /tmp/qwen38-route.trace
python3 benchmark/flat_pair_cache_sim.py /tmp/qwen38-route.trace --slots-per-layer 64 --layers 48
```

The measured 4.352 tok/s reproduces the 4.0–4.7 baseline.

## 2. Control calibration (does the simulator reproduce the engine?)

Simulator uniform-LFU at 64 slots/layer over **decode-only** events (first 30
positions dropped): **65.7%** hit. The engine measures 71.7–78.1% at this
budget; the gap is prefetch adoption, which the simulator does not model —
the depth-1 predictor turns would-be misses into hits, worth ≈6–12 points
here. Calibration through the engine's own measured point (≈55 ms exposed I/O
at ≈78% hit, i.e. ≈40.4k misses of 183,840 decode expert-references) puts
uniform-64's 62,976 misses at ≈86 ms exposed ≈ 211 ms/token ≈ **4.7 tok/s** —
against 4.352 measured. The control reproduces within run-to-run noise once
calibrated; the raw 3.6 GB/s projection in the CLI output understates exposed
time ~3x and its tok/s figures are not the ones to quote.

## 3. Policy comparison at the fixed 12 GB budget (3,072 pairs)

Decode-only, same budget:

| policy | hit rate |
| --- | --- |
| uniform (today's semantics) | 65.7% |
| flat (layer, expert) pool | 65.9% |
| heat-proportional per-layer | 65.5% |

**Flat and heat are within noise of the control (+0.2 / −0.2 points).** B1's
cache-semantics hypothesis is dead on this trace, and the reason is
structural, not a tuning miss:

- Per-layer demand is nearly uniform: distinct experts per layer 146 / 259
  (median) / 409, sum 12,544 pairs over the decode. There is no hot layer to
  favor, so heat allocation has nothing to exploit, and a flat pool's extra
  freedom buys nothing a per-layer pool cannot.
- **The working set outruns the cache at any window**: a sliding window of
  16 decode tokens touches a median of **3,234** distinct (layer, expert)
  pairs — more than the entire 3,072-slot budget; 32 tokens ≈ 4,906; 64
  tokens ≈ 6,811. The cache is in constant churn; hit rate is set by
  capacity vs window, not by allocation cleverness.

## 4. Ceilings at this budget

| quantity | value |
| --- | --- |
| compulsory-miss bound (infinite cache) | 93.2% |
| Belady oracle @ 3,072 pairs (8.28 GiB) | **85.4%** |
| Belady @ 4,608 | 89.8% |
| Belady @ 6,144 (16.6 GiB — swaps) | 92.1% |
| Belady @ 9,216 | 93.2% (saturated) |
| uniform LFU @ 96 / 128 slots | 76.5% / 83.5% |

Best case for *any* online policy at 8.28 GiB sits between uniform's 65.7%
and Belady's 85.4%. Even at the full Belady 85.4%, calibrated exposure is
≈36 ms/token → ≈6.2 tok/s — short of the 7 target, and Belady is unreachable.

## 5. Verdict (per the plan's §2 acceptance rule)

**B1 is dead as modeled: the plan jumps to §4/§5 (PLE attribution, split
reads) and the §6 merge decision moves to the operator.** No cache-semantics
change reaches 7 tok/s at 12 GB on this workload; the levers that remain are
shrinking the corpus (expert merging 512→256–320) or the bytes per miss
(split gate+up/down reads, hc-factored QKV, lm_head 4-bit) — the first is
numerics-changing and needs sign-off, the second is byte-identical but
bounded (≈+0.3–0.5 tok/s at today's hit rate).

The trace also kills the §3 LFU-warmth sub-hypothesis for this shape: with a
working set that exceeds the cache within 16 tokens, cross-request warmth
evicts within a handful of tokens regardless.

## 5b. B2 (PLE decode-time traffic) — closed by arithmetic, no model run

The plan's §4 asked for a capture; the manifest answers it. From
`models/qwen3.8-flash-next_125B_A6B_4Bit/ple_constants.json`:
`ple_head_dim=160` → `rowBytes = 160 × 2 = 320`, `ngram_size=3`,
`ple_n_heads=16` → `gatherPLERows`
(`sources/TinyTitan/Runtime/Inference/RealForwardRunner+PLE.swift:13-26`)
reads **16 rows × 320 B = 5,120 B per token** from `ngram_table.bin`
(102.4 GB). Against 293–368 MB of expert traffic per token that is 0.0015%
— 1.4 µs at the 3.6 GB/s wall. The comment on the gather already said
"three orders of magnitude under"; the manifest pins it. The reader uses an
ordinary `pread` (page-cached), so those 5 KiB are usually not even device
reads. **B2 is not material; closed with these numbers.**

## 6. Provenance

- Simulator: `benchmark/flat_pair_cache_sim.py` (22 tests, CI-registered).
- Belady and window analyses: ad-hoc scripts over the committed trace; the
  Belady implementation used for §4 is the corrected form (evict the
  farthest-next-use resident unconditionally; a first version that skipped
  the insert when the newcomer's next use was nearer produced the nonsense
  46.7% and was discarded before any conclusion was drawn from it).
- The trace is one workload (one essay prompt, 383 decode tokens). A second
  capture over a different workload is cheap while the window stays free and
  would strengthen §5's verdict; the working-set arithmetic above is the
  load-bearing fact and is a property of the routing distribution, not the
  prompt.

## 5c. B3 (split blob reads + early miss fixup) — measured, accepted

Reader split (`fetch_ranges`), streamer `phase1Bytes` from the manifest's
`down` offset, first-stage publication on `ExpertLoadOperation`, and the
runner's early miss fixup: phase 1 over the miss slots is encoded and
committed as soon as the front range (gate+up) is resident and runs on the
GPU while the tail (down) fetch is still in flight; `routedCB` carries
phase 2 only. Committed at `87d072c`, `af26fc5`, `c4a431c`, `53d7b8d`.

Golden gate: `tools/golden-baseline.sh --check qwen38-4` — output identical
to baseline on the full chain (model run, window free). Model-free suites
and `tools/lint.sh` all green.

A/B, same binary pair, same prompt, `--max-new 384`, `TINYTITAN_RUNNER_STATS=1`
(two pairs, back to back; base = pre-3b `c4a431c`, treatment = `53d7b8d`):

| arm | tok/s | hit rate | io_hidden_pct | wait_ms/token |
| --- | ---: | ---: | ---: | ---: |
| base, run 1 | 4.162 | 74.67% | 14.09 | 86.55 |
| base, run 2 | 4.197 | 74.97% | 14.31 | 85.90 |
| B3, run 1 | 4.286 | 74.35% | 99.90 | 76.76 |
| B3, run 2 | 4.286 | 74.05% | 99.83 | 75.97 |

The plan's falsifier (§5): hit rate held equal (74.8% → 74.2%, inside the
pair's spread), and tok/s **and** `io_hidden_pct` rose together — satisfied.
The `io_hidden_pct` jump is the accounting seeing the fixup CB complete
inside the I/O window; the wall-clock number is the honest one: **+2.5%
(4.180 → 4.286 mean), ≈ +0.11 tok/s**, with per-layer wait down 11.5%
(86.2 → 76.4 ms). Below the §5 projection (+0.3–0.5 tok/s) but positive and
mechanistically explained: the fixup fills only the GPU-idle slice of the
tail fetch, and the front fetch's window remains unfilled at this hit rate.

Text outputs were not compared across arms: the CLI's default sampling is
temperature 1.0, so run-to-run text differs inside each arm; the golden
gate is the numerics verdict and it passed.

## 5d. Reader throughput vs the device wall — closed (2026-10-11)

The one remaining byte-identical candidate was raising the reader's raw
throughput. Premise checked before touching code: what does this SSD
actually deliver on the reader's exact access pattern — 2.64 MiB
expert-stride-aligned preads scattered over 48 layer files? A thread-pool
probe (`stride = 2,768,896`, random file + expert-aligned offset per read,
~2 GB per configuration, 48 layer files ≈ 68 GB so the page cache never
answers):

| workers | GB/s |
| ---: | ---: |
| 1 | 2.01 |
| 2 | 2.84 |
| 4 | 3.57 |
| 8 | 3.71 |
| 16 | 3.69 |

Saturation is ~3.7 GB/s at 8–16 workers; the engine's 4-worker reader
(3.57 GB/s in this probe, matching its in-engine effective throughput) is
already at the wall. The sequential contrast kills the alternative shape:
`dd` over a whole layer file measures 3.15 GB/s (`layer_47.bin`) and
2.61 GB/s (`layer_31.bin`) — sequential is *slower* than the reader's
strided parallelism, so "read more sequentially" is not a lever either.
The earlier assumption of a ~7 GB/s M3 SSD was wrong for this access
pattern; scattered 2.64 MiB reads are the binding constraint.

**The reader-throughput lever is closed with these numbers.** Combined
with B1 (cache policy dead) and B2 (PLE = 0.0015%), no byte-identical
lever of material size remains; what is left is §6, gated on an operator
decision.

## 5e. The ANE/CPU survey — every remaining co-executor closed or bounded (2026-10-11)

The question: with the SSD at its wall and the GPU's per-token busy time
~103 ms, can the ANE or the CPU cores take enough work off the GPU to move
tok/s? Re-surveyed against the post-B3 profile (4.286 tok/s ≈ 233 ms/token,
wait 76 ms, busy ~103 ms, `head_logits` 13.49 ms/call) before building
anything; what was already measured in `docs/cpu-coexecution-plan.md` stays
closed, and the two candidates that record never tested at *this* profile
got measured now.

Closed by earlier measurement (do not reopen):

- **CPU co-execution of experts:** GPU busy +44.9%, tok/s −22.6%; no split
  ratio won (`cpu-coexecution-plan.md:601`).
- **ANE concurrent load:** GPU busy +89.3%, tok/s −45.4% — the ANE is a
  *faster bus consumer* on the same die, and this profile is bus-bound
  (`cpu-coexecution-plan.md:692`).
- **ANE expert ops:** the GPU runs them 2.8x faster; the ANE collapses
  below 1 MiB of weights.
- **CPU spinning during I/O wait:** −15%.
- **Fused greedy head:** −3% (benched, not shipped).

Closed by arithmetic against the post-B3 profile:

- **CPU head GEMV:** the head is 606 MiB of 8-bit weights; at the M3's
  ~17 GiB/s CPU memory bandwidth that is ~35 ms of reads against the GPU's
  13.49 ms measured. The CPU is 2.6x slower on the one op it could
  plausibly take, before counting any transfer.
- **CPU or ANE expert work during the I/O window:** dead after B3, not
  before it. The early miss fixup already runs phase-1 compute *while* the
  tail fetch is in flight and `io_hidden_pct` is 99.90 (§5c run 1) — the
  overlap those co-executors were meant to supply is already banked on the
  GPU. What remains of the token is GPU-busy compute at 233 ms cadence,
  and both co-executors measurably make *that* slower (the two closure
  rows above).
- **ANE at all, at decode cadence:** the token's ~40 ANE-eligible ops at
  8–12 ms of per-handoff cost is 320–480 ms of handoff per 233 ms token.

**New measurement — the head's in-engine cost is cadence, not the kernel.**
The head GEMV is the largest per-call GPU cost in the profile (13.49
ms/call, 606 MiB stored → 46.6 GB/s) and the only op never probed at the
qwen38 shape. `TinyTitanBench head_affine8` now has
`TINYTITAN_BENCH_HEAD_CADENCE_MS` (committed `2c3fc36`): one launch per
command buffer with a wall-clock gap, mimicking decode's once-per-token
cadence. The identical kernel, identical buffers, identical PSO:

| arm | per launch |
| --- | ---: |
| back-to-back steady state | 9.42 ms (71.7 GB/s) |
| 50 ms gaps | 12.73 ms |
| 200 ms gaps | 15.49 ms |
| in-engine (`head_logits`) | 13.49 ms |

The in-engine 13.49 sits between the 50 and 200 ms cadence arms: the ~4 ms
delta over steady state is GPU-state (DVFS/power between launches), not
kernel inefficiency, not memory layout, not the mmap. Recoverable bound if
the state could somehow be kept warm: ~4 ms/token ≈ +1.8% — and nothing in
the engine controls GPU clock state, so this is a measured bound, not an
actionable lever.

**New measurement — the head on CPU and ANE, at the exact shape.** A
synthetic coremltools probe (8-bit kmeans-palettised `linear` at 248320 x
2560, batch 1, slope harness N=2 vs N=4, median of 40 predicts) measured
the head op on every unit pool coremltools can address:

| arm | per op |
| --- | ---: |
| ALL / CPU_AND_GPU / CPU_ONLY / CPU_AND_NE (N=2 graph) | 78.4–79.1 ms whole-graph, no arm differs |
| slope per op (any arm) | 38.95 ms → 16.3 GB/s of stored bytes |
| GPU in-engine (`head_logits`) | 13.49 ms → 46.6 GB/s |

Two verdicts, one honest caveat. (1) **CPU head: measured dead** — 38.95
ms/op is 2.9x the GPU's 13.49 ms, confirming the ~35 ms bandwidth
arithmetic above with a real number. (2) **ANE: no demonstrable
engagement.** The arm contrast did not discriminate (ALL vs CPU_AND_GPU
identical at both N), and even with the GPU excluded (`CPU_AND_NE`,
GPU_ONLY's complement — the only setting that forces an ANE-or-CPU choice)
the rate is unchanged and matches CPU bandwidth — so the ANE either did
not take this op (8-bit palettised batch-1 linear not ANE-eligible in this
form) or fell back silently; `~/.venvs/coreml-py311/bin/python` ran both
probes. The caveat: a non-engagement is not a measured ANE *rate*, so the
closing statement leans on the earlier record — ANE batch-1 is bounded at
the same ~64 GB/s wall (the head alone would then save ~4 ms/token ≈
+1.8% at best, before handoff cost and a mandatory golden re-capture for
the ANE's different accumulation numerics), and every measured ANE
co-execution so far made the token slower, not faster.

**The survey is closed.** Every ANE/CPU involvement option is now either
measured dead at this profile or bounded below 2% by a mechanism nothing
in the engine controls; decode tok/s at 12 GB remains SSD-bound, and the
open levers are the operator decisions in §6.

## 5f. B4 algebra check — the HC-factored QKV premise is false (2026-10-12)

Plan §6 item 1 ordered the algebra check before anything else: `W_eff =
W·B` precomputed offline is only valid if the hyper-connection read map
`normed → blockInput` equals a fixed linear `B @ normed`, bit-exact modulo
float order. The runtime (`HyperConnection.swift`) computes
`h = silu(Wdown @ normed/4)`, `mix = sigmoid(Wup @ h)`,
`input = mean_st(mix[st] ⊙ normed[st])` — so the map is linear only if the
gates are degenerate. The check ran over the real install's dequantized
weights (`tools/ssdai_reader.py`, layers 0/1/2/47, attn + mlp arms and the
model-level mixer):

| arm | \|Wdown\|max | \|Wup\|max | mix std/mean | linearity viol. | scale viol. |
| --- | ---: | ---: | ---: | ---: | ---: |
| layers.0.attn | 1.28 | 3.94 | 0.24 | 0.19 | 0.33 |
| layers.1.attn | 6.47 | 2.94 | 0.41 | 0.44 | 0.45 |
| layers.2.attn | 1.84 | 4.50 | 0.52 | 0.36 | 0.30 |
| layers.47.attn | 1.06 | 2.89 | 0.34 | 0.40 | 0.31 |
| layers.0.mlp | 2.78 | 8.56 | 0.44 | 0.29 | 0.23 |
| layers.1.mlp | 2.34 | 4.49 | 0.39 | 0.26 | 0.33 |
| layers.2.mlp | 4.13 | 6.19 | 0.49 | 0.32 | 0.23 |
| layers.47.mlp | 0.94 | 2.95 | 0.37 | 0.27 | 0.24 |
| model mixer | 1.35 | 4.16 | 0.71 | 0.39 | 0.36 |

The gates are live (weight maxima O(1)–O(10), mix moves 24–71% of its mean
across random tokens) and the linearity violation is 0.19–0.44 relative on
every arm — against a float32 noise floor below 1e-6. **No fixed `B`
exists; the B4 premise is false and the lever is closed at the algebra
gate.** No golden re-capture was spent, no engine code was touched; this
is the outcome the plan's check-first ordering exists to catch cheaply.
The plan's §6 item 1 and status header record the closure.

## 5g. gate|up sub-split — premise measured false (2026-10-12)

The mechanism hypothesis after §5c: at 74% hit rate × 10 routed experts ×
48 layers, a token misses ~2.6 experts per layer, so the B3 front phase
issues only ~2.6 ranges of 1.76 MiB to the 4-worker reader — possibly
undersaturated — while splitting gate|up at the real blob seam (up's
offset, 921,600 B) would double the range count. Premise checked with the
§5d thread-pool probe at the real geometry (F_NOCACHE preads at
expert-aligned offsets over the 48 real layer files, random file per job,
~1.5 GB per arm, 4-bit front = 1,843,200 B, blob = 2,768,896 B):

| arm | GB/s |
| --- | ---: |
| front as-is: 3 concurrent × 1.76 MiB | 3.25 |
| front as-is: 4 concurrent × 1.76 MiB | 3.36 |
| sub-split: 3 × 2 × 0.88 MiB (6 concurrent) | 3.22 |
| sub-split: 3 × 2 × 0.88 MiB (4 workers) | 3.33 |
| control: 6 × 0.88 MiB | 3.36 |
| control: 3 × 2.64 MiB whole blob | 3.38 |
| tail-split blob: 3 × 2 × 1.32 MiB | 3.32 |

Every arm lands in 3.2–3.4 GB/s, inside arm-to-arm noise, and the
sub-split arms are not faster than the as-is front. **Per-pread size
between 0.88 MiB and 2.64 MiB does not move this SSD's throughput, and the
front phase at ~3 concurrent ranges is already at the same wall as
everything else — the "undersaturated front" premise is false and the
sub-split lever is closed at its probe gate.** A pinned single-file run
(layer_00 only) reproduced the same verdict (2.92–3.17 GB/s, sub-split
slowest), so the conclusion is not an artifact of the scatter. No engine
code was touched, no golden gate was spent. Combined with §5d, the reader
is closed in both directions: more workers buy nothing (§5d) and finer
ranges buy nothing (this section).

## 5h. Expert merging — the 512 experts are mutually orthogonal (2026-10-12)

Plan §6 item 2 was the last lever with a big ceiling: merge 512 experts to
256–320 so the decode working set fits the 3,072-pair cache. The operator
set the constraint first: **no merge that makes the model dumber.** The
study therefore asks whether the experts cluster at all, on the real
install's dequantized 4-bit weights (`PackedExperts`, layers 0/6/12/24/36/
47, all 512 experts, gate+up+down). Method: pairwise relative Frobenius
distance `d(i,j) = ||Wi−Wj||_F / sqrt(||Wi||_F ||Wj||_F)` — for
uncorrelated equal-norm matrices this is √2; then a greedy closest-pair
matching (the merge schedule), then the functional cost: SwiGLU output of
the weight-averaged expert vs both originals over unit-RMS gaussian inputs
(RMSNorm makes the hidden state unit-RMS; real activations would need a
model run), plus the milder "drop expert i, route its tokens to j"
contrast.

Per layer, 131,072 pairs each; pooled medians across the six layers:

| quantity | median | p90 | max |
| --- | ---: | ---: | ---: |
| all-pair relative distance | 1.4143 | 1.4159 | 1.5282 |
| matched-pair (closest 256) distance | 1.4131 | 1.4135 | 1.4438 |
| merge→256: merged-expert output error | 0.947 | 0.971 | 1.162 |
| merge→320 (closest 192 pairs) error | 0.946 | 0.963 | 1.123 |
| drop contrast (route i's token to j) | 1.411 | 1.419 | 1.470 |

The verdict is structural, not marginal. **The experts are pairwise
orthogonal: the all-pair distance is √2 = 1.4142 with almost no spread,
and even the closest matched pairs (1.4131) sit a rounding error below
random.** There is no cluster structure to merge. The functional numbers
agree: weight-averaging the *closest* pair still destroys ~95% of the
routed expert's output, and the drop contrast is worse than the output
itself (1.41×). Merging 512→256–320 would make the model measurably
dumber — it fails the operator's constraint at the study gate, and the
quality-eval-on-a-merged-install step is never reached because there is
no merge schedule worth building.

Caveats, stated plainly: the study measures weight geometry and
gaussian-input function, not task quality; and a merged install would add
re-quantization error on top. Both push the same direction — neither could
rescue a √2 distance. With this, §6 item 2 is closed; the remaining
§6 items are lm_head 4-bit (B5, quality-gated) and the MTP re-verdict
(B6). The study's script was a one-off (`/tmp/merge_distance.py`); its
pooled-summary print crashed on a scalar shape after every per-layer row
had printed, so the per-layer table above is complete.

## 5i. B5 offline gate — lm_head 4-bit is not near-lossless (2026-10-12)

Plan §6 item 3 hypothesized that 4-bit group-64 affine might be
near-lossless for the vocabulary head. Measured on **real activations**:
one decode run (175 positions, Hamlet prompt, `--max-new 192`) with
`TINYTITAN_ACT_DUMP` captured the true pre-head hidden state (`mixer_out`)
and the engine's own 8-bit logits per position; the study
(`/tmp/b5_head_4bit.py`, model-free afterwards) simulated the repack's
4-bit affine scheme (asymmetric uint4, groups of 64 along the input dim,
BF16-rounded scale and bias) on the real 8-bit head weights and compared
logits.

Reference sanity first: the offline fp32 recomputation of the 8-bit head
matches the engine's dumped logits to 0.00021 relative — the proxy is
sound, so all the error below is the 4-bit step itself.

| metric (175 real decode positions) | value |
| --- | ---: |
| relative logit error (4-bit vs 8-bit) | 0.079 |
| greedy top-1 flips | 2 / 175 (1.1%) |
| positions where err@top1 exceeds the top-1/top-2 margin | 6 / 175 |
| ΔNLL on the generated token (perplexity multiplier) | mean +0.005 nats on greedy, p90 +0.59 on generated |
| KL(p8‖p4), T=1 | 0.0065 mean, 0.020 p90 |
| entropy shift | 0.345 → 0.334 |

**The "near-lossless" premise is false: 4-bit injects 7.9% relative logit
noise — ~370x the 8-bit reference error — and flips the greedy argmax on
1.1% of positions.** Over a 300-token greedy generation that is a ~96%
chance of at least one divergence, and every divergence is a visible
quality event (the golden text changes, which is why the plan priced a
golden re-capture). The perplexity-side damage is small (≤0.5% on greedy)
and the KL is modest, so this is not obviously a dumber model — but it is
measurably a *different* one, against an operator constraint that says no
quality reduction.

**Decision: declined at the operator decision (2026-10-12)** — "quality
loss cannot be accepted." Advancing B5 would have meant a repack change
(the engine already supports a 4-bit head end to end —
`lmHeadWeightBits` resolves per-role and the decode/prefill head kernels
are width-keyed, so no kernel work), a fresh ~68 GB install from the
pinned source checkpoint, the full quality panel (perplexity + prompt
panel), a golden re-capture, and the runner-stats A/B for what B5 buys
(the head GEMV read halves, ~640 → ~320 MiB/token ≈ +2-3% ceiling at the
67.8 GB/s the plan measured). The offline gate's job was to kill a bad
premise before that spend; it did: the premise is false (7.9% logit
noise, 1.1% greedy flips), and the operator declined any numerics
change regardless of the small perplexity damage. No repack build, no
install, no quality panel. This closes the §6 list and with it the
decode-12GB campaign, standing at the measured 4.286 tok/s.

## 5j. B6 re-affirmed closed by its own gate (2026-10-12)

The MTP re-verdict is conditional on the hit rate rising; with B1, the
reader (§5d/§5g), B4, §5h and the sub-split all closed, nothing in the
campaign moves it off 74.2–74.8%. The condition never fires; B6 stays
closed without a re-measurement.


## 5k. The CPU L1/L2 cache is not an accelerator for this engine (2026-10-13)

Asked whether the CPU cores' L1 (192+128 KB per P-core) or 16 MB P-cluster
L2 could be used as an accelerator for a critical area. Attribution from
the code, no model run needed:

- **No tensor math runs on the CPU in the main decode loop.** Router,
  GEMVs, attention, expert FFN and sampling are all Metal kernels; the
  `Sampler` touches the logits on the host only when a repetition/presence
  penalty or a structured constraint is active, which the default path is
  not (`sources/TinyTitan/Runtime/Generation/Sampler.swift:84-106`).
- **The one real CPU cost per token is dispatch latency, not data.** The
  profiled "≈20 ms/token turnaround" is GPU idle during readback, plan,
  pin, encode and commit (`docs/qwen38-decode-profile-2026-09-05.md`
  §C) — command-buffer scheduling, on which a cache has no effect. The
  early-hits work (profile doc §12) already shipped most of it down.
- **The CPU data the decode loop does touch is already trivially
  resident**: kernel arguments, expert plans, PLE bookkeeping — kilobytes
  against a 128 KB L1d.
- **The only CPU tensor machinery in the tree is the closed side
  engine.** `CPUEngine/` + `Int8AffineGEMV`/`CPUExpertFFN` exist for CPU
  co-execution, measured at **−22.6%** tok/s (`cpu-coexecution-plan.md`),
  and CPU spinning during I/O wait measured −15%. This profile is
  bus-bound: moving compute onto the CPU cores turns them into another
  memory-bus consumer competing with the SSD streaming that is the wall.

Verdict: the cache-residency lever prices at **~0%** here. The L1/L2
option only pays when a kernel repeatedly re-reads a working set that
fits the cache, and the only such compute available would have to be
*added* to the CPU — which is the measured-loss co-execution lever. Not
re-openable on the evidence.

## 5l. Low Power Mode re-baseline and turnaround decomposition (2026-10-13)

A runner-stats re-baseline captured **inside Low Power Mode** (`pmset -g`
`lowpowermode 1`, on battery), at the operator's explicit direction after
an LPM throttle was found to have invalidated an earlier attempt to
re-measure. **Scope caveat: these numbers are valid for comparison
against each other and against the code's structure only — not against
the 4.286 tok/s campaign record or any mains-power number.** Method is
the §5c CLI command, run twice back-to-back (run 1 warms, run 2 is the
measured arm):

| | prefill | decode | tok/s | ms/token |
| --- | --- | --- | ---: | ---: |
| LPM run 1 (warm) | 58 tok / 8.53 s | 384 tok / 112.59 s | **3.411** | 293.2 |
| LPM run 2 (measured) | 58 tok / 8.35 s | 384 tok / 128.28 s | **2.993** | 334.1 |
| campaign B3 arm, mains, for orientation only | — | — | 4.286 | 233.3 |

Runner-stats decomposition (all values ms/token):

| segment | run 1 | run 2 |
| --- | ---: | ---: |
| cb1 (encode) | 8.28 | 12.77 |
| io (miss I/O) | 106.86 | 117.98 |
| cb2 | 1.78 | 2.99 |
| rdadvise (issue) | 17.53 | 26.81 |
| wait (GPU + hidden I/O) | 118.96 | 124.38 |
| **body** | **262.13** | **299.48** |
| head | 21.54 | 22.91 |
| cache_plan | 5.37 | 8.65 |
| io_completion_to_fixup | 2.99 | 5.15 |
| gather | 2.96 | 3.74 |
| loop_sample | 1.92 | 1.92 |
| loop_other | 1.97 | 1.98 |
| embed | 0.70 | 0.90 |
| expert hit rate | 76.0% | 74.4% |
| io_hidden | 99.79% | 99.09% |
| expert load p50/p95/p99 | 4 / 8 / 16 | 4 / 8 / 16 |
| rdadvise calls / MiB per token | 102.5 / 272.4 | 110.1 / 334.0 |
| prefetch issued / adopted per token | 43.5 / 23.6 | 44.0 / 26.1 |

Findings:

- **The token's structure is unchanged under LPM.** Hit rate (74–76%),
  I/O hidden fraction (99+%), expert load percentiles and the segment
  ordering all match the campaign's B3 arm; the throttle scales every
  segment roughly together — clock throttling, not a new bottleneck.
- **LPM run-to-run spread is ±14% on decode time** (112.6 vs 128.3 s)
  while every CPU-side dispatch segment moved 30–50% between runs
  (rdadvise 17.5→26.8, cb1 8.3→12.8, plan 5.4→8.6). **This baseline
  cannot resolve a single-% candidate**: the noise floor is an order of
  magnitude above one percent. It prices structure, not micro-levers.
- **CPU-side dispatch segments total ≈44–66 ms/token (15–20% of the
  token)**, the same magnitude as the mains-power turnaround finding
  (profile doc §C). rdadvise issuing is the largest single segment, but
  the §8 closure covers the *policy* A/B (`7cc8b5e`: default beats off
  by ~10.6% at 4-bit, neutral on M5/8-bit; `db1c102`: off costs 8.5%,
  adaptive/bounded no win) — not the issue count. A call-count reduction
  (~105 `F_RDADVISE` syscalls/token for ~300 MiB advised) is the one
  uncovered seam this baseline names, and it is unpriceable at LPM's
  noise floor; it would need a mains window and an A/B to go further.

Verdict: no new lever opened. The LPM baseline confirms the delivered
build's structure and re-identifies the turnaround as the remaining
non-GPU cost; any single-% work against it requires mains power.

## 5m. rdadvise on/off on this family — measured, closure re-affirmed (2026-10-14)

§5l named the rdadvise issue count (~105 `F_RDADVISE` syscalls/token) as the
one seam §8's closure did not cover, and noted the closure's provenance is
ornith-era (`7cc8b5e`, `db1c102`) — this family at 12G had never been
arm'd. Mains window, §5c CLI command, `--rdadvise default` vs
`--rdadvise off`, two back-to-back runs per arm, `TINYTITAN_RUNNER_STATS=1`,
`--max-new 384`, prompt "Write a detailed essay about the history of
computing." (10 prefill tokens):

| arm | run | tok/s | hit rate | io_hidden_pct | wait_ms/token | rdadvise_ms/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| default | 1 | 4.313 | 77.79% | 99.84 | 89.02 | 11.60 |
| default | 2 | 4.805 | 77.74% | 99.93 | 74.83 | 9.72 |
| off | 1 | 4.380 | 77.26% | 84.26 | 81.17 | 0.00 |
| off | 2 | 4.538 | 77.53% | 84.90 | 79.42 | 0.00 |

Findings:

- **The mechanism is confirmed and the two effects roughly cancel.** Off
  removes the 9.7–11.6 ms/token issue cost — and io_hidden falls from
  99.8–99.9% to 84.3–84.9%, i.e. ~15% of the ~100–120 ms/token miss I/O
  stops hiding behind GPU work. Wall-clock: default mean 4.559 vs off
  mean 4.459 (+2.2% for default), but the run-1 pairing went the other
  way (4.313 vs 4.380). The difference is inside the run spread — the
  default arm's own pair spans 11% (4.313 → 4.805).
- **The ornith-era "off costs 8.5%" does not transfer to this family at
  this budget.** Here the policy is worth ≤2%, not 8.5%. The §8 closure
  ("family-dependent") is re-affirmed on measurement, and the shipped
  `default` policy stays: it is never worse and its hiding is real.
- **Issue-count coalescing prices at a theoretical ceiling of the 10–12
  ms/token it could recover while keeping the hiding — ≈2–3%, single-%
  territory**, below what a 384-token single run can resolve here. Parked
  as unproven; an A/B would need many more repeats per arm than four.
- **Warmth keeps climbing across back-to-back runs** (4.313 → 4.805
  inside one arm), so short single-run A/Bs on this model carry a warmth
  confound the campaign's more stable pairs did not show. Today's
  no-route-trace runs also reach 4.805, above the 4.286 record — that
  record was captured with `TINYTITAN_ROUTE_TRACE` writing ~19,824 lines
  to /tmp during the run; the trace capture itself is part of the gap.
  The 4.286 campaign number stands as recorded; a no-trace run is a
  lighter-instrumented environment, not a faster build.

Verdict: no lever opens. Default rdadvise stays; the issue-count seam is
priced at ≤2–3% and parked.

## 5n. Context length at 12G — decode pays ~20% for a long prompt (2026-10-14)

The wiki-derived recommendation that was not already closed: measure decode
at more than one context length. Run with `benchmark/tinytitan_longctx.py`,
qwen3.8 4-bit, `TINYTITAN_EXPERT_CACHE_BUDGET=12` (the harness passes it to the
launcher as `--ram 12`, which forwards `--ram-budget 12G` to the server), mains
power, build `d26d97a`, server v5.19 on
127.0.0.1:8091. One generation per cell, driver cells in order A, B, C:

| cell | prompt | max_new | prefill s | decode tok/s | finish |
| --- | --- | --- | ---: | ---: | --- |
| A: long+128 | 1,697 tok | 128 | 74.4 | **3.962** | length |
| B: short+1024 | 22 tok | 1024 | 4.4 | **4.271** | length |
| C: long+1024 | 1,697 tok | 1024 | 67.8 | **3.423** | stop @ 549 |

Runner stats, per-token, B vs C (the two 1024-token decodes):

| metric | B: short | C: long |
| --- | ---: | ---: |
| expert_hit_rate | 0.7588 | 0.6824 |
| expert_load p95 | 8 ms | 16 ms |
| wait_ms | 84.0 | 95.6 |
| cache_plan_ms | 3.68 | 5.64 |
| io_hidden_pct | 99.88 | 99.63 |
| rdadvise MiB/token | 300.2 | 191.7 |

Findings:

- **A long prompt costs ~20% of decode throughput at this budget**
  (4.271 → 3.423 on the same 1024-token decode length). The mechanism is
  the §B/§5n cache mechanism, not attention: the expert hit rate falls
  0.759 → 0.682 and the expert-load p95 doubles (8 → 16 ms) as the 96-slot
  per-(layer,expert) cache turns over against a larger live working set.
  I/O stays almost fully hidden (99.6–99.9%) — the loss surfaces as GPU
  wait, not as exposed I/O.
- **Prefill is ~40 ms/token at this context** (67.8–74.4 s for 1,697
  tokens) and warms ~9% on the second long prompt of the session
  (74.4 → 67.8), the same warmth climb §5m saw.
- **Cell C stopped at 549 tokens on EOS** — greedy on the long prompt ends
  its essay there; the 3.423 figure is measured over those 549 tokens, not
  truncated at 1024.
- Single run per cell: these are first-measurement figures, not settled
  means. But the direction is structural (hit rate is a counted
  quantity, not a clock reading), and the cache-churn mechanism is the
  same one §5/Stage-0 priced: capacity, not policy, bounds the long case.

Verdict: the wiki proposal's premise is confirmed — context length is a
real decode variable at 12G, worth ~20% between short and ~1.7K-token
prompt. It does not open a lever: raising cache capacity at this budget
is §8-closed (slots_128 swaps at 68% cost; the flat-pair cache is at
capacity by design), and no routing or policy change recovers the hit
rate the larger working set spends. The number is recorded as the
long-context figure future proposals have to argue against.

## 5o. Budget × context sweep — the working-set cliff, and what actually sits on it (2026-10-14)

Two questions in one protocol. The sweep question (#147): does any
(cache slots × max-context) combination beat the derived default on this
machine? The regression question (#152): v5.0 shipped qwen38 at a 12 GiB
budget — 96 slots — and measured 6.957 tok/s (wiki Changelog);
`808cea3` then clamped the budget to a third of physical memory, which
is 8 GiB — 64 slots — on this 24 GB machine, and the whole campaign ran
there. Did the clamp cost the configuration?

Protocol: release build `33b3a06`, CLI (no budget flag: the CLI takes only
`--expert-cache-slots`, so the derivation picks the slots the server would
pick), essay prompt, `--max-new 384`,
`TINYTITAN_RUNNER_STATS=1`, mains power. Mirrored arms, and the mirror
of the decisive arm was run after two *healthy* arms so a predecessor's
swap debt cannot be the explanation. 129.6 MB per slot per layer group
(expert stride × 48 layers), so 64 slots is 7.92 GiB, 96 is 12.15,
112 is 14.51, 128 is 16.58, 160 is 20.74; the resident floor is 3.72 GiB.

| arm | slots | max-context | decode tok/s (pair) |
| --- | ---: | ---: | --- |
| base | 64 (derived) | 262144 | **4.551 / 4.481** |
| c32 | 64 | 32768 | 4.543 / 4.175 |
| s96 | 96 | 262144 | 3.052 / 3.201 |
| s96_32k | 96 | 32768 | **4.929 / 4.966** |
| s112_32k | 112 | 32768 | 3.622 |
| s128_32k | 128 | 32768 | 0.999 / 1.473 |
| s160_32k | 160 | 32768 | 0.498 / 0.579 |

Findings:

- **96 slots at the shipped 262144 context is a 30% loss, mirrored**
  (3.05/3.20 against 4.55/4.48). #152's hypothesis is false on the
  current build: restoring the tuned 12 GiB budget — the exact experiment
  `808cea3`'s comment deferred — would be a regression, not a recovery.
- **96 slots at a 32K context cap is the only winning arm: +9%**
  (4.93/4.97 against the base pair's 4.52). Hit rate rises 0.775 →
  0.850, confirming the extra 32 slots genuinely hold experts.
- **The cliff is the machine's working set, and the QSA indexer is what
  the context cap actually frees.** Runner stats, s96b vs s96_32kb:
  io_ms identical (66.2 / 65.7), hit rate identical (0.8525 / 0.8531),
  io_queue_ms identical (0.38) — but wait_ms 201 vs 98 and body_ms 290
  vs 184. The disk does the same work at the same speed in both arms;
  the slow arm pages. The accounting: the QSA indexer preallocates its
  per-layer key buffers at *full context* on first use
  (`QSAIndexer+Scratch.swift`: "Sized to the full context on a layer's
  first use, unlike the KV cache which grows on demand"). From the
  installed manifest (indexerHeadDim 128, compressRatio 4, 12
  full-attention layers of 48): **0.94 GiB at 262144 against 0.12 GiB
  at 32768**. That 0.82 GiB is the entire difference between the two
  96-slot arms: 12.15 + 3.72 + 0.12 ≈ 16.0 GiB fits, +0.82 ≈ 16.8 GiB
  does not. The KV cache is not the variable — it starts at 8192 tokens
  and grows (`min(maxContext, initialCapacityTokens)`), identical in
  both arms.
- **64 slots is on the right side of the cliff either way**, which is
  why the context cap measured neutral there (4.54/4.18 against
  4.55/4.48, inside the ~8% mirror noise this prompt shows).
- **112 slots is over the cliff even at 32K** (3.62), and 128/160 swap
  to collapse as before. The last good rung on this machine is 96, and
  only with the 0.82 GiB freed.
- **Deviation noted:** the s128/s160 arms left 9.4 GB of swap debt that
  persisted through the sweep; the mirrored pairs (and s96b running
  after two healthy arms at the same 3.2 as s96) are what make the
  cliff attribution safe anyway.

Verdicts:

- **#147 closed.** At shipped defaults the derived 64 slots is the best
  runnable configuration on this 24 GB machine. The binding constraint
  is physical working set, not budget policy.
- **#152 closed — hypothesis false, mechanism found.** The third-of-RAM
  ceiling is the *right* answer for qwen38 at the shipped context on
  this machine; v5.0's 6.957 is not reachable by any configuration this
  sweep turned. What grew since is named: the QSA decode indexer
  (`f5b1720`, 2026-08-30) wires ~0.94 GiB at full context and postdates
  the v5.0 measurement window. The rest of the gap to 6.957 is not
  attributable to a single knob and is not chased here.
- **#149 closed by the base arm's own runner line.** io_ms = 102.0 per
  token for ~403 MB moved ≈ 3.95 GB/s against the device's measured
  3.92 GB/s four-thread ceiling (`ParallelExpertReader`); io_queue_ms
  0.41 — no queueing to buy back. The disk is at its wall; every
  alternative allocation of that bandwidth (deeper rings, more staged
  reads, two-layer lookahead) is already measured loss-making in the
  prefetch study. The miss-read path has no headroom.
- **#151 closed without an experiment.** The serial tail per token is
  head 12.2 ms + gather 2.5 + sample 1.0 + embed 0.45 against a ~220 ms
  body; overlapping it across tokens requires starting the next token's
  embedding before this token is sampled, which is speculation — and
  speculation is §8-closed (MTP), with the fused head architecturally
  excluded for this family (hyper-connections on). The residue is ~1%,
  below the 8% mirror noise floor.
- **The lever the sweep did open:** the QSA preallocation is the file's
  own deferred work ("growing on demand is worth doing… and is not
  worth the bookkeeping yet"). Growing the indexer buffers on demand,
  exactly as KV does, frees ~0.82 GiB at short contexts at *any*
  max-context — which is precisely the headroom the +9% arm needs at
  the shipped 262144 default, with numerics untouched (the buffers are
  storage; selection runs over visible keys only). That is the next
  task.

## 5p. The lever taken: QSA grows on demand, the ceiling floor, and the default is now 96 slots (2026-10-15)

§5o's open lever, implemented and measured. Two commits:

- **`45f55bb` — the QSA indexer buffers grow on demand**, exactly as the KV
  cache does: 8192 cells at first touch, doubling with a `memcpy` of the live
  prefix, capped at `maxContext`. The full-context preallocation (0.94 GiB
  across the 12 full-attention layers) becomes 0.12 GiB at short contexts at
  any max-context. Growth is safe for the same reason KV's is: every access
  funnels through the one accessor, growth completes before the needing
  token or chunk is encoded, and Metal retains an old buffer for any command
  buffer still in flight. Selection runs over visible keys only, so the
  numerics are storage-independent. Model-free suite:
  `QSAIndexerGrowthTests` (start small, whole when small fits, doubling and
  capping, contents preserved across a grow).
- **The ceiling floor** — the experiment `808cea3`'s comment deferred ("a
  floor at the tuned budget, so only machines below the tune are cut").
  `affordableExpertCacheBudget` now keeps `wanted` on any machine with at
  least the 24 GiB the budgets were tuned on, and keeps the third-of-RAM
  clamp below it. The test amendment is stated in the test itself: the old
  32 GiB assertion expected `(32<<30)/3` because the 96 slots it buys paged
  under the QSA preallocation; with that gone, the cut buys nothing on a
  large machine and the assertion now expects the tuned 12 GiB through. A
  new boundary test pins the floor inclusive at 24 GiB and the third still
  cutting at 20.

Verification, all on this 24 GB M3, release build, greedy essay prompt,
`--max-new 128`, temperature 0:

| arm | slots | decode tok/s | text |
| --- | ---: | ---: | --- |
| old build (pre-`45f55bb`), default | 64 | 4.811 | reference |
| new build, default | 64 | 4.720 | byte-identical |
| new build, `--expert-cache-slots 96`, full 262144 context | 96 | **5.156** | byte-identical |
| floor build, **no flags** — shipped default derivation | 96 | **5.045** | byte-identical to both |

The floor run's own log: `ram profile cache=11.88G slots=96
resident_floor=3.72G estimate=15.60G` — the default now lands on the rung
the sweep measured as the last good one, with no operator flag. Greedy text
is byte-identical across all four arms, and `tools/golden-baseline.sh
--check qwen38-4` reports the output identical to the recorded baseline
(ornith-8 not checked: no install on this machine, reported rather than
fetched).

Effect by family on a ≥24 GiB machine: qwen38 4-bit gets its tuned 12 GiB
(64 → 96 slots, **4.48–4.81 → 5.05–5.16**, +7–14% depending on the base
arm); qwen3.6 35B-A3B 4-bit gets its tuned 10 GiB back (the ~+1.3%
`808cea3` priced); Ornith 8-bit is unchanged (its 8 GiB is a third of 24
GiB exactly, so floor and clamp agree); machines under 24 GiB are
unchanged — the 8 GB mini evidence for the third still stands and still
applies there.

Campaign accounting against the ≥6 tok/s goal: the shipped default moves
from 4.48–4.55 (mirrored base, §5o) to 5.05–5.16. That is the largest
single measured step of the campaign, taken with numerics untouched and
greedy output byte-identical. It is still short of 6: §5o/#152 recorded
that the remaining gap to v5.0's 6.957 is not attributable to a single
knob, and this step does not change that record. Single-run caveat: the
96-slot figures are one run each against a prompt whose mirrored spread is
~8% — but the pre/post gap at 96 slots (3.05/3.20 → 5.16, +61–69%) and
the default-derivation move (64 → 96 slots at equal text) are structural,
not clock noise.

## 5q. The same tier-sizing rule applied upward: budgets that scale with the machine (2026-10-15)

§5p floored the expert-cache ceiling at the tune, so a machine with at least the
24 GiB the budgets were measured on keeps what it was tuned to. That fixed the
machine this repository has, and left the other half of the answer unfinished: a
tuned budget is still a **constant derived on one machine class**, so a 64 GiB
Mac was handed the same 96 slots as a 24 GiB one and left ~40 GiB of the tier it
could have filled. `docs/lmcache-evaluation.md` named the one concept worth
taking from LMCache — size the resident tier to the host rather than to a number
measured elsewhere — and this section applies it at home, to both resident tiers.

**Change 1 — the third becomes a floor as well as a ceiling.**
`affordableExpertCacheBudget` is now `min(wanted, RAM/3)` below the 24 GiB tune
and `max(wanted, RAM/3)` at or above it. The measured machine classes are
byte-identical: below 24 GiB the rule is untouched, so the 8 GB mini evidence
still stands; at 24 GiB, `max(12, 8)` is the tuned 12 GiB, which is §5p's 96
slots; at 32 GiB a third is 10.67 GiB, so the tune still wins. What changes is
everything above, where the third exceeds the tune and the machine can hold it:

| RAM | a third | qwen38 4-bit (its row asks 12 GiB) | 35B-A3B 4-bit installs (ask 10 GiB) | prompt-cache RAM tier |
| ---: | ---: | --- | --- | ---: |
| 8 GiB | 2.67 | 2.67 → 24 (min, unchanged) | 2.67 → 40 (unchanged — the measured mini optimum) | 256 MiB |
| 24 GiB | 8.00 | 12.00 → **96** (measured, §5p) | 10.00 → 160 (unchanged) | 256 MiB |
| 32 GiB | 10.67 | 12.00 → 96 (tune wins) | 10.67 → 160 (same rung) | 256 MiB |
| 48 GiB | 16.00 | 16.00 → 128 (new) | 16.00 → 256 (new) | 2 GiB (new) |
| 64 GiB | 21.33 | 21.33 → 160 (new) | 21.33 → 256 (new) | 2 GiB |
| 128 GiB | 42.67 | 42.67 → 256, the ladder's top rung | 42.67 → 256 | 2 GiB |

Slot counts are the nearest rung on the 8…256 ladder, then stepped down while
the footprint exceeds 1.15 × the budget — the same rule that produced the 96 §5p
shipped. qwen38 4-bit is 129.6 MB a slot (stride 2,768,896 × 48 layers), the 35B
4-bit geometry 67.5 MiB (1,769,472 × 40). The `wanted` figures are the installs'
own `ModelProfile` table rows, which is what the server feeds the derivation
(`ServerModelSession+Loading.swift:118`) — not the family fallback. Nothing
changes at 32 GiB in either family; 48 GiB is where the third first exceeds the
tune.

**Change 2 — the prompt cache's RAM tier is derived, not constant.** Its default
was a flat 256 MiB, which is a small machine's number. The cost of prompt state
is now measured rather than assumed: this run stored a 152-token prefix as
`state_bytes=117442560` (112 MiB), so **~0.74 MiB per prefix token** on qwen38
4-bit — which means the 256 MiB tier holds roughly 350 tokens and pushes any
longer tool-loop prefix into the SSD tier, a second streaming consumer beside the
expert reads that already own the disk, while the RAM above it sat idle. 2 GiB is
the same rule at a big machine's size: about 2,800 prefix tokens resident. The
default stops there rather than scaling on, because past a few GiB what binds is
the store's own 4 GiB per-snapshot capture ceiling and the retained-entry count,
not RAM; an operator who wants more names the flag, whose range reaches 4 GiB.
Below 48 GiB the 256 MiB stays, because every machine this project has measured
lives there and it is what those measurements priced.

**Why this is the TTFT lever and not only a decode one.** Prefill streams the
experts a prompt activates, so a resident slot that holds one of them removes a
disk read from the path to the first token; a prefix restore served from RAM
rather than the SSD tier does the same. The profile prices prefill at 17.9 s for a
262-token qwen38 prompt (`docs/qwen38-decode-profile-2026-09-05.md`), and §5n
measured that the coupling between the KV reservation and expert residency is
real but indirect — it runs through RAM, not through the disk.

**Safety properties, each one a place where §5o's cliff was reached by the other
direction:**

- No measured class changes: below the tune `min`, at the tune `max(12, 8) = 12`,
  and the prompt-cache default on every machine measured here (< 48 GiB) is still
  256 MiB.
- A third cannot exceed the half-of-physical line the launcher warns on
  (`BatchedMemoryBudget.slotBudgetBytes` refuses to wire more than half), so the
  growth can never trip that guard by construction. Dense families are inert:
  `expertCacheHeldBack` is 0 when `numExperts == 0`.
- Every operator override still wins: `--expert-cache-slots`, `--ram-budget`,
  `--prompt-cache-memory-mib`.
- Slot counts and tier sizes change residency, not arithmetic. No numerics path is
  touched, so greedy output is unchanged by construction, and the golden check is
  the proof rather than the promise.

**Not measured.** This repository's install is the 24 GiB M3 — exactly 24 GiB, so
the floor is inclusive and the rule is a deliberate no-op here. There is no 48, 64
or 128 GiB machine available, so every row at or above 48 GiB is arithmetic on
the constant ladder, pinned by test (`aboveTheTuneTheBudgetGrowsWithTheMachine`,
`promptCacheMemoryTierGrowsWithTheMachine`) rather than measured. §5o bounds how
far to trust them: on 24 GB the rungs above 96 collapsed on *physical working
set*, not budget policy. A 48 GB machine has the headroom that rung needs (15.84
GiB of cache + the 3.72 GiB resident floor + 0.12 GiB of QSA at short context,
against 48 GiB), so the arithmetic is not contradicted by anything measured — but
nobody should call 128 slots a measured number until it runs on a machine that
big, and a long-context prompt there should be re-checked against the same
accounting before it is believed.

**Verification.** Three changes, each with its test run failing first.

- **Change 1** — `aboveTheTuneTheBudgetGrowsWithTheMachine` failed against the
  floor-only rule (`sources/TinyTitan/Runtime/Configuration/RuntimeConfiguration.swift:199`)
  before the `max` arm existed, and passed after. The existing
  `clampNeverGrowsABudget` had to be amended rather than left alone: its single
  assertion *was* the old premise — 8 GiB of budget on a 128 GiB machine stays
  8 GiB — which this change reverses on purpose. It is renamed
  `clampNeverGrowsABudgetBelowTheTune` and now pins the side that must not move
  (16 GiB and 8 GiB machines, `physicalMemory == 0`, the tune itself), and the
  reversed claim is asserted by the new test rather than dropped. The derivation
  has one production caller (`sources/TinyTitanLib/ServerModelSession+Loading.swift:118`),
  so nothing else could have drifted.
- **Change 2** — the order here was wrong and is recorded as such: `parse`
  already derived the tier when the assertion was written, so the RED had to be
  produced by temporarily reverting the seed to a flat `256`, watching
  `tests/TinyTitanServer/OpenAIValidationTests.swift:566` fail on
  `big.promptCacheMemoryMiB == 2_048`, and restoring the line. `defaults()` also
  gained a pinned `physicalMemory: 24 << 30`: an unpinned assertion over a
  derived default reads whichever host ran the suite, which is the AUD-312
  defect class.
- **The launcher pinned the tier away** — found by asking where change 2 is
  actually *reachable*. `tools/server_launcher.sh` put
  `--prompt-cache-memory-mib 256` into every GPU command line, and the launcher
  is how the owners of the big machines this was written for start a server: a
  dry run against a synthetic 128 GiB host showed the pin, so the derivation
  would have reached no one who used the supported path. `test_launcher_ram.py`'s
  new `ThePromptCacheTierIsLeftToTheServer` failed on both the 24 GiB and the
  128 GiB sub-arm before `tools/server_launcher.sh:1332` changed, and passes now
  that the launcher names only the mode. `benchmark/tinytitan_profile.py`'s
  `DEFAULT_PROMPT_CACHE_MEMORY_MIB` had one reader — `test_tinytitan_profile.py`,
  which asserted the launcher's pin as a property of the benchmark protocol. That
  test was the wrong side, because the pin was the defect: it now asserts the
  launcher's command carries no tier, and the constant is gone.- **Suites and gates** — `swift test --no-parallel` after the last source edit:
  7 bundles, 1765 tests, exit 0. `tools/lint.sh` full battery: exit 0 over all
  18 gates. `swift build -c release`: exit 0, `lipo -archs` reports `arm64`
  alone. Python: `test_launcher_ram` plus `test_tinytitan_profile` — 35 tests,
  OK — and ruff clean at the pin.
- **On the real machine, configuration identity** — a no-flag server launch on
  this 24 GiB M3 logged `TinyTitan ram profile cache=11.88G slots=96
  resident_floor=3.72G estimate=15.60G` and bannered `expert_slots=96 …
  prompt_cache_memory_mib=256`: byte-identical to what the shipped build
  banners, so both derivations are the deliberate no-ops they claim to be where
  they must be. `tools/golden-baseline.sh --check qwen38-4` → *ok — output
  identical to baseline*, exit 0. Ornith 1.5 8-bit is **not checked**: this
  machine has no 8-bit install and a gate may not fetch one.
- **A protocol deviation, recorded** — the same live run clocked 0.617 tok/s and
  a 20.895 s TTFT where §5p measured 5.045 tok/s at the identical configuration.
  Root-caused before anything was concluded from it: the Mac was on battery at
  34–35% with `lowpowermode 1`, 8.2 GB of swap in use against 2.9 GB before the
  run, 8% of RAM free, load 12.4 left over from the suite and builds that
  preceded it. The protocol requires mains, so those two figures are **not
  measurements** and are quoted nowhere else in this file; the banner and
  `ram profile` lines above are clock-independent, which is why they stand. §5r
  takes the mains re-run of decode, TTFT and `wait_ms` this left open, and
  corrects one use of those lines: `ram profile` prints the *derived* budget
  even after an operator's `--expert-cache-slots` has replaced it, so it is not
  proof of the rung that actually ran.
- **One claim corrected by measurement** — this section first said the 2 GiB
  tier "holds the retained four prefixes at the store's 4 GiB capture ceiling".
  The run's own log disproved it: `prompt_cache stored tokens=152
  state_bytes=117442560 ram_bytes=117442560 disk_bytes=0` is 0.74 MiB per prefix
  token, so 256 MiB holds about 350 tokens and 2 GiB about 2,800 — not four
  multi-gigabyte prefixes. The doc comment on `defaultPromptCacheMemoryMiB`, this
  section and the wiki row now carry the measured figure.

## 5r. The mains re-measurement of the shipped 96-slot default (2026-10-15)

§5q left three numbers open and one of its own pieces of evidence invalid. All
four are settled here. Provenance: commit `0242f65`, tree clean, 24 GiB M3
(`hw.memsize` 25769803776), macOS 27.0.1, Swift 6.4
(swiftlang-6.4.0.34.1 clang-2100.3.34.1), `swift build -c release` exit 0 in
85.21 s with `lipo -archs` = `arm64`, **AC power with `lowpowermode 0`**,
`pgrep` guard empty before every arm, one model process at a time. Seven CLI
arms, all exit 0, all producing stdout sha256
`95aa4531b4ea2304a10dde8d079c68bb3b2bf2eea1335b21814fe3fca5c2d0f0` (2,065 bytes)
— greedy text identical across both rung and both batch.

Arm command, batch A (three identical arms of the shipped default, 15:21–15:26Z):

```
TINYTITAN_RUNNER_STATS=1 .build/release/TinyTitanCLI \
  --model models/qwen3.8-flash-next_125B_A6B_4Bit \
  --prompt "Write a detailed essay about the history of computing." \
  --max-new 384 --temperature 0
```

No `--ram`, no `--expert-cache-slots`: the derivation lands on 96 slots, and each
arm logged `ram profile cache=11.88G slots=96 resident_floor=3.72G
estimate=15.60G`. Batch B (15:27–15:33Z) is the same command with
`--expert-cache-slots 64` interleaved as 64, 96, 96, 64 so machine drift cancels
in the paired difference and the baseline runs first *and* last.

| batch | run | slots | decode tok/s | prefill s | wait_ms | io_ms | body_ms | hit | read MiB | evictions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 1 | 96 | 4.075 | 6.080 | 118.84 | 75.26 | 218.43 | 0.8456 | 75,868 | 31,293 |
| A | 2 | 96 | 4.406 | 3.824 | 111.73 | 73.41 | 207.10 | 0.8459 | 75,709 | 31,293 |
| A | 3 | 96 | 4.632 | 3.226 | 102.59 | 73.65 | 196.44 | 0.8452 | 76,047 | 31,293 |
| B | 1 | 64 | 4.245 | 3.171 | 84.64 | 104.76 | 216.67 | 0.7704 | 112,768 | 48,582 |
| B | 2 | 96 | **5.021** | 3.808 | 91.67 | 72.10 | 181.19 | 0.8448 | 76,245 | 31,293 |
| B | 3 | 96 | **5.074** | 3.069 | 93.02 | 68.76 | 179.41 | 0.8462 | 75,554 | 31,293 |
| B | 4 | 64 | 4.690 | 2.820 | 75.61 | 100.11 | 195.61 | 0.7716 | 112,190 | 48,582 |

**1. §5p's step is confirmed at §5o's own arm length.** Paired 96−64 deltas are
**+0.776** (runs 1→2) and **+0.384** (runs 4→3): two rounds, both positive. The
mechanism is in the counters, not the clock — 76 GB of expert reads against 112
GB (−32%), 31,293 evictions against 48,582, hit rate 0.845 against 0.771, and
`io_ms` 69–72 against 100–105. §5p measured 5.045 at `--max-new 128` with n=1 and
flagged that as a caveat; the caveat is discharged for the direction and the
magnitude at 384 tokens is +0.4 to +0.8 tok/s. This still does not reach 6.

**2. A finding that outranks the headline: the first arm after a load is
systematically slow.** Batch A rose monotonically on one identical configuration
(4.075 → 4.406 → 4.632, +13.7% from first to last); batch B's two 64-slot arms
differ by 10.5% in the same position order; and the server arm's first request
decoded 4.099 tok/s where its second reached 5.431 (+32%). `wait_ms` carries it —
102–119 in batch A against 92–93 in batch B at near-identical `io_ms` and hit
rate, so the engine did the same work and waited longer. The likely mechanism is
file-cache warmth of the expert stream; it is **untested**, and named as a
hypothesis rather than a conclusion. The consequence is procedural and applies
backwards through this whole file: cross-*batch* qwen38 tok/s comparisons are not
usable, and the "~8% mirror noise" §5o recorded understates this host — the same
configuration measured up to 24% apart across batches. Within-batch pairing is
what a qwen38 number has to rest on.

**3. `wait_ms` at the shipped default is 91.7–93.0 ms of a 179–181 ms body.** §5o
priced the double-buffered-plan axis against 98 ms at 96 slots with a 32K cap; at
the shipped 262144 context and the QSA fix the number is essentially unchanged,
and it is ~51% of the token. `io_hidden_pct` is 99.6–99.8% and `io_queue_ms`
0.28–0.41, so the disk is not queueing and the miss I/O is already hidden — what
remains exposed is the wait for the *next* layer's plan, which is exactly axis 1.

**4. TTFT at the shipped default on mains is 6.882 s.** The server arm launched
no flags on port 8085, bannered `model=qwen3.8-flash-next_4-Bit … expert_slots=96
prompt_cache=multi-prefix prompt_cache_memory_mib=256`, and served a 128-token
greedy chat request: first content delta at **6.882 s** (`prefill_s=6.808`),
second identical request 4.674 s (`prefill_s=4.514`). §5q's 20.895 s was 3.0×
this, which is what "battery artifact" means in numbers. The server was launched
and stopped by the arm's own script; no other model process existed.

**5. An identical replay is not a prompt-cache hit, and that is the design.** The
repeat request logged a *second* `prompt_cache stored tokens=149` entry and no
`hit` line, so this looked like a reuse failure until `ServerPromptCache.match`
was read: `sources/TinyTitanLib/ServerPromptCache.swift:172` requires
`renderedPromptIDs.count >= entry.kvPosition`, and the stored entry's KV-backed
prefix is 149 tokens — the prompt *and* the generation — against a 25-token replay,
so the direct-prefix arm cannot fire; `:180` then requires more messages than a
single-turn replay has. Reuse is for continuations (the tool loop the wiki
documents), where the next request *extends* the stored prefix. §5q's `cached=0`
is the same mechanism, not a regression — and it bounds what §5q's bigger RAM tier
buys: a long retained prefix in a multi-turn loop, not a replayed one-shot prompt.
The per-token cost re-measured here, 117,403,392 bytes over 149 tokens = 0.738
MiB per prefix token, matches §5q's 0.74 within 0.5%.

**6. The operator cap, and the machine.** On this 24 GiB M3 the rule is that no
run may take a RAM budget above 12 GiB, because above it the machine destabilises;
larger Macs keep the §5q host-sized derivation, which is what they were written
for. Every arm here complied: 11.88 GiB derived cache, nothing above it named, and
the 64-slot arms deliberately *under* it. Stability held across all seven CLI arms
and the server — every exit 0, swap peaking at 9.41 GB without a watchdog, a crash
report or a lost request. Note that 64 slots is the rung whose whole footprint
(7.92 + 3.72 = 11.64 GiB) fits under 12 GiB including the resident floor, while the
shipped 96 sits at 15.60 GiB estimated: the cap is on the cache tier, and which
reading an operator wants is their call, not a measurement's.

## 5s. The two RAM diagnostics fixed test-first, and the wiki arm refused (2026-10-15)

§5r recorded the `ram profile` defect as authorised-for but not granted; the
operator then said fix it. Two commits: `2c2a30b` (the applied count in both
lines) and `0355d2c` (the warning names the override when the override is what
misses the target). No clock in this file changes — this section is about what
the load-time log is allowed to be used as evidence of.

**1. Root cause.** `ram profile` and `ram target` printed the rung the
*derivation* chose, while the allocation used `requestedExpertCacheSlots ??
slotOverride ?? derived`. That precedence existed only as an inline expression
*below* the diagnostics, so the printed number and the allocated one were
computed twice by different code and had drifted: an operator who passed
`--expert-cache-slots 96` under `--ram-budget 12G` got 96 slots wired and a load
line asserting `slots=64`. A second defect surfaced only by probing the fixed
line against a live server — the warning called those 96 allocated slots "the
minimum cache" and said "the cache is already at its smallest". Both are false
when an override pinned the count, and both pointed the reader at the install
when the flag they set themselves is the thing they can change.

**2. The lines now** (verbatim: a no-flag CLI run with `--expert-cache-slots 64`,
and the `--ram-budget 12G --expert-cache-slots 96` server probe on port 8086):

```
TinyTitan ram profile cache=7.92G slots=64 derived_slots=96 resident_floor=3.72G estimate=11.64G (cache budget, not a process target)
TinyTitan ram target=12.00G cache=11.88G slots=96 derived_slots=64 resident_floor=3.72G estimate=15.60G
TinyTitan ram warning: 12.00G is below this install's 15.60G footprint (3.72G resident + the 96-slot cache named by --expert-cache-slots); the target alone allows 64 slots, so the gap is the override, not the install.
```

`slots=` is the applied count in every line; `derived_slots=` is emitted only
when an override replaced the derivation, so the shipped default line is
byte-for-byte what it was. `appliedSlots` is defined **once** in
`ServerModelSession+Loading.swift` and shared by the two diagnostics, the
`loadSlots` that sizes the cache, and the server banner. The text itself moved to
a new 76-line `sources/TinyTitanLib/ExpertCacheDiagnostics.swift` with six tests
pinning both branches of both lines, because these lines are the operator's only
record of the allocation and nothing downstream re-derives them.

**3. Which prior claims this corrects.** §5p's third arm (line 786,
`--expert-cache-slots 96` at full context) is the case the defect describes: the
table's 96 was right, but under the old code that run's own `ram profile` printed
`slots=64`, so the log could not have confirmed it — the rung was known only from
the flag. The line quoted at §5p (the floor run) and at §5q was legitimate,
because those were no-flag loads where derivation *is* the applied count. §5r's
finding 1 therefore closes: the identity evidence it needed is now produced
correctly rather than worked around. Nothing is re-measured for this, since every
rate here already rests on within-batch pairing plus rung-specific counters (hit
rate, evictions, `expert_read` MiB).

**4. Flag ownership, which two recipe lines here got wrong.** The runtime and
server take `--ram-budget <size>`; `tools/server_launcher.sh` takes `--ram <whole
GB>` and forwards it as `--ram-budget "${ram_gb}G"` (`server_launcher.sh:1339`),
and only when a budget was chosen at all — an unattended run passes nothing and
the install profile applies; the whole `TinyTitanCLI` target parses **no** RAM
flag of any spelling, only `--expert-cache-slots`. §5n and §5o cited a bare
`--ram` on the CLI, which is not
a command anyone could have run: it exits 2 with `error: unknown flag`. Both lines
now state what actually executes.

**5. Verification on the committed tree.** `swift test --no-parallel` → seven
target runs, 1,771 tests (815 / 408 / 157 / 189 / 41 / 63 / 98), each reporting
*passed*, no failure line. `tools/lint.sh` → exit 0, 18 gate sections clean.
`tools/golden-baseline.sh --check qwen38-4` → *ok — output identical to baseline*
(ornith-8 not checked: no install here, and a gate may not fetch one). Both lines
above came from real loads, the server probe launched by a pgrep-guarded script
that killed its own server on exit; `lipo -archs` on the release binaries is
`arm64`. The test-file changes are disclosed in `0355d2c`'s message: one existing
fixture (applied=8, derived=16) described a pair no loader can produce, so it was
retargeted to applied=derived=8 — the state it meant to pin — with its expected
text unchanged, and the override shape it accidentally covered became a new test.

**6. The wiki's Qwen3.8 decode row is not refreshed, and the refusal is the
finding.** The second authorised item was `.qwen/wiki/Benchmarks.md`'s 6.82 tok/s
for Qwen3.8-Flash-Next 4-bit (the 2026-08-30 batch) against §5r's 5.02–5.07.
Those are different protocols, so refreshing the row means running the wiki's own
instrument, not pasting a number: `benchmark/tinytitan_gate0_profile.py` with
`--quant 4bit --model-dir models/qwen3.8-flash-next_125B_A6B_4Bit --runs 5` at
shipped defaults (11.88 GiB cache, 96 slots). It refused before starting a server:

```
refusing to start: GPU is already 54% busy before TinyTitan launches (threshold 15%).
```

`Device Utilization %` held at 43–68% across a 30 s sample with load average
3.4–6.3, and the consumers are the operator's desktop, not this engine:
`Qoder Helper (Renderer)` 42–46% CPU, `WindowServer` 42%, three Chrome renderer
helpers, WhatsApp, Boom 3D. Everything else was compliant — AC power,
`lowpowermode 0`, macOS 27.0.1, Swift 6.4, 80% memory free, empty `pgrep` guard,
clean tree. `--allow-busy-gpu` exists to override the threshold to 100%, and its
own message says a contended result is not comparable with the published
benchmark, so it was not used and the row stands unchanged, reported as **not
measured**. Two procedure notes: this driver is the only one under `benchmark/`
that gates on GPU idleness, so §5r's CLI arms were taken without that check; and
its threshold is binary (15% or off) — there is no "record the contention and
carry on" mode, which is the right design for a published table but means a
wiki-protocol arm needs the GPU-holding apps quit first.

Two further things about that refused arm were found after the fact, and each would
have cost a quiet-machine window to discover the other way round.

- **The command as actually run named the wrong install.** It was invoked as
  `--quant 4bit --runs 5` with no `--model-dir`, and that driver's `4bit` key is
  `models/ornith-1.5_35B_A3B_4Bit` (`tinytitan_gate0_profile.py:75`) — the published
  workload was Ornith-only until `--model-dir` was added. This Mac has no Ornith
  install, so on a quiet GPU the arm would have died on a missing model directory
  after waiting all day for one. The refusal itself still stands, because the GPU
  preflight runs before the model is opened, but the Qwen3.8 row could never have
  been refreshed by that invocation. TT-049's next-step command on the tracker
  already carries the `--model-dir`; the command quoted above did not, and now does.
- **The budget claim is confirmed in the code rather than assumed.**
  `server_command` appends `--ram` only when `ram_budget` is not `None`, and the
  harness default is `os.environ.get("TINYTITAN_BENCH_RAM_BUDGET")` with no fallback
  — `None`, so the shipped derivation applies (`tinytitan_profile.py:29-30`,
  `:211`). That is what "no flag, 96 slots" rests on. The gate0 docstring still
  describes the profile as using an "8 GiB budget", which was true of the v4.1
  published batch and is not what the harness does now; left unfixed, because this
  window was not authorised to touch the driver, but no reader should take that
  line as the arm's configuration.

**7. What the page now says instead (wiki `c2bb180`).** The 6.82 figure was not
touched — it is still the v4.7 measurement, and only a wiki-protocol arm can change
it — but the page's own provenance was checked and published, because the
contradiction is older and wider than one row. `git log -S` on the wiki: all five
medians arrive in `eb84741` (2026-08-30, "Add the v4.7 benchmark section") and none
has been edited since, while the peak paragraph was rewritten on 2026-09-29
(`5d74cd6`) from the README matrix, whose rows for these models all arrived in
`69dd683` (2026-09-05). Four of the five table medians are *above* the peak the same
page then quotes for that model and width — 23.28 and 22.65 against "17–22" at
4-bit, 12.72 against "7–12" at 8-bit, 6.82 against "5.5" — and only Ornith 1.5 8-bit
at 11.89 sits inside its range. A median at shipped defaults cannot exceed a peak
taken across every setting, so the two sets cannot both describe one build, and the
table is the older of the pair by six days. For the 125B row §5r settles it
independently: 4.08–5.07 tok/s at the shipped 96-slot default, against 6.82. The
note states the batch, the contradiction, that range and its protocol, points at
TT-049, and pastes nothing into the table.

Two claims were caught and corrected while drafting that note, and are recorded
because both were plausible: that the README's `5.46` predated the table (it
postdates it by six days — the `5.46` in README on 2026-08-08 is a different
quantity, a "32 slots + pin" reference figure), and a positional "two lines below"
that the insertion itself invalidated. What the note does *not* establish is whether
the four 35B rows are also unreachable on the current engine: this Mac has no 35B
install, and no gate may fetch one, so on this machine TT-049 can only ever refresh
the 125B row.

The sibling sweep for the same defect found two engineering documents dividing by
that stale figure. `docs/ngram-table-sharing-plan.md` projected the 8-bit install at
"about 3.4 tok/s" as half of 6.82; corrected in the same commit as this section,
because the README already measures that build at **2.10 tok/s** — 0.39x the
README's own 4-bit row, so the projection was optimistic and its verdict (not worth
building) was right for a stronger reason than the arithmetic. The other is
`docs/qwen38-decode-20tps-concept.md:110`, which opens "Published decode is
**6.82 tok/s**" and reasons from it to a ~12 tok/s target. Left alone: it says
*published*, which remains true of the wiki row, it is a superseded concept
document, and §5r's 4.08–5.07 already displaces it for any live decision. Neither
file is guarded by a gate, so both readings are documents, not proofs.



## 5t. The wiki's own instrument, run on v5.20, on a contended GPU (2026-10-15)

§5s recorded the refused arm and left TT-049 open for a quiet machine. The
operator then said run it now, so it ran with the guard overridden. **This is not
a benchmark figure and the wiki row is unchanged** — see 4 and 5 for why.

**1. What ran, and against what build.**

```
python3 benchmark/tinytitan_gate0_profile.py --quant 4bit \
  --model-dir models/qwen3.8-flash-next_125B_A6B_4Bit --runs 5 --allow-busy-gpu
exit 0    # 1 warm-up + 5 measured runs, fresh server per run, 9m21s wall
```

Two launches were refused first, at GPU 53% and 70%, by the 15% preflight
(`tinytitan_gate0_profile.py:220`); the third went because `--allow-busy-gpu`
raises the threshold to 100%. Commit `b27f29e` (the v5.20 tag). The binary is
`.build/release/TinyTitanServer`, mtime 21:19, and carries the `5.20` version
literal; `b27f29e` changed only that literal and the notes file, so this is the
shipped code path. The rung is confirmed from the run's own server log rather
than assumed — `.build/benchmark-logs/gate0_4bit_5.log`:

```
TinyTitan ram profile cache=11.88G slots=96 resident_floor=3.72G estimate=15.60G (cache budget, not a process target)
```

No `derived_slots=`, so the derivation applied with no override: the shipped
default, exactly what the wiki table claims to measure.

**2. The numbers.** Median over five runs; each run is 512 tokens of continuous
prose at native 262K context, multi-prefix prompt cache on, 8-bit KV, top-k 20,
temperature 0.6, MTP off.

| run | decode tok/s | busy/token ms | occupancy | expert hit | expert read MiB | wait_ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 5.450 | 104.858 | 52.1% | 0.8718 | 73,459.5 | 96.754 |
| 2 | 5.491 | 100.333 | 50.3% | 0.8722 | 73,277.3 | 97.026 |
| 3 | 5.713 | 96.823 | 50.4% | 0.8719 | 73,409.4 | 91.634 |
| 4 | 6.102 | 92.234 | 51.0% | 0.8718 | 73,464.8 | 84.661 |
| 5 | 6.026 | 94.920 | 51.7% | 0.8719 | 73,422.6 | 86.497 |

**median 5.713 tok/s = 175.04 ms/token**; `prefill_s` 7.598–7.912 (median 7.79);
`io_ms` 51.195, `io_hidden` 99.71%, `gpu_share_of_decode` 65.5%, router readback
0.0162 ms, cache plan 3.4161 ms. Driver verdict: `BANDWIDTH-BOUND (ceiling
~1.35x)`, which is the same verdict §5c–§5r reached from the other direction.

**3. The finding that outranks the level: all five runs are byte-identical.**
`completion_sha256` is `aded9bd20fbde145` for every run, 440 completion tokens
each, and the counters move with it — hit rate spans 0.8718–0.8722 and expert
reads 73,277–73,465 MiB, both inside 0.25%. The work was provably the same in all
five arms, yet decode rose 5.450 → 6.102 across them, **+12%**, and
`busy_per_token` fell from 104.9 to 92.2 ms. That is §5r's first-arm-after-a-load
effect reproduced inside a single protocol with the bytes held fixed, and it is
the strongest evidence in this file that a *level* from a batch is a property of
the machine settling, not of the build. Only paired within-batch deltas carry
magnitudes; a median of five sequential identical arms carries a range.

**4. The contention, measured rather than assumed.** `Device Utilization %` was
sampled every 30 s for the whole run (19 samples, `.build/benchmark-logs` aside):
50–65%, never below 50%, load average 2.7–7.4. Everything else complied — AC
power, `lowpowermode 0`, macOS 27.0.1, Swift 6.4, 82% memory free, empty `pgrep`
guard before launch, clean tree, and no model process left after it
(`_terminate_all` ran in the driver's `finally`). Note what contention did *not*
do here: the hit rate stayed at 0.872, so the expert cache remained resident and
this is not the 0.277-collapse shape the preflight's comment describes. The
sharing shows up as CPU/GPU service time, not as an I/O failure — which is
exactly why a contended run looks superficially healthy in its counters.

**5. What this licenses, and what it does not.** Three figures now exist for this
model and width and none of them is the same measurement: this contended
gate0 median **5.713**, §5r's greedy 384-token CLI arms **4.08–5.07**, and the
wiki table's **6.82** from the v4.7 engine on 2026-08-30. Note that 5.713 already
sits *above* the 5.5 this page quotes as the model's peak and the README's 5.46 —
so the table-versus-peak contradiction the wiki's batch note describes is not
resolved by taking the table's own instrument; it is a protocol difference between
the peak row and the table row, still standing. What this run licenses is the
statement that **the 6.82 row cannot be defended as a current figure at this
machine's defaults** — it was measured on another engine, and the current build at
the shipped default under contention does not reach it. What it does not license
is replacing it: the replacement must come from the same command without
`--allow-busy-gpu`, on a GPU under 15%, and that window has never been available
on this Mac while its desktop is running. TT-049 stays open for that arm.

## 5u. The 32K context cap re-measured on v5.20 — §5o's winning arm is gone (2026-10-15)

§5o found exactly one configuration that beat that batch's default: 96 slots with
`--max-context 32768`, **+9%** (4.929/4.966 against 4.551/4.481), and it named the
mechanism — the QSA indexer preallocating 0.94 GiB at full context against 0.12
GiB at 32K, that 0.82 GiB being the whole margin between the 96-slot arm paging
(3.05) and fitting (4.93). §5p then shipped the fix (`45f55bb`): the buffers start
at `initialCapacityCells = 8_192` and double on demand
(`sources/TinyTitan/Kernels/Attention/QSAIndexer.swift:34`), which frees the same
0.82 GiB at *any* max-context. The question this section answers is narrow and it
is the one the operator asked — after that fix, does the cap still buy anything?

**Provenance and protocol.** Tree `3f92d47` (the `v5.20` tag plus a notes-only
commit), `swift build -c release` refreshed before the batch (Build complete,
82.35 s), clean `git status`. 24 GB M3, macOS 27.0.1, Swift 6.4, AC power,
`lowpowermode` off, empty model-process guard before launch, one CLI process at a
time, none left after. Seven cold runs: three interleaved rounds with the arm
order alternated (A B · B A · A B) plus a closing baseline run to bound drift.

```
TINYTITAN_RUNNER_STATS=1 .build/release/TinyTitanCLI \
  --model models/qwen3.8-flash-next_125B_A6B_4Bit \
  --prompt "Write a detailed essay about the history of computing." \
  --max-new 384 --temperature 0 --max-context 262144   # arm A — shipped default
                                          --max-context 32768    # arm B — capped
```

§5o's prompt and length, greedy as §5p ran it. Both arms derived the same tier:
every one of the seven logs carries `ram profile cache=11.88G slots=96
resident_floor=3.72G estimate=15.60G`, so the cap is not changing the slot count —
this is a pure memory-pressure test at equal capacity. Driver
`/tmp/v520_ctx_ab.sh`, data `/tmp/v520-ctx-ab/summary.tsv`.

**Deviations, all of them.** (1) The desktop held the GPU at 56% at run 1's launch
instant; runs 2-7 launched at 0%. Round 1's delta is therefore contaminated and
only rounds 2-3 are clean. (2) A 16-token smoke run preceded the batch to pin the
footer field names — not an arm, but it warmed the page cache for run 1. (3) The
harness pipeline exited 1 because `tee` opened its progress log before the script
created the directory; every one of the seven CLI runs exited 0, so the batch
itself succeeded. (4) The arms went through the CLI, which accepts any count up to
262144, while the server refuses a non-rung; 32768 is a rung, so the two paths
agree on this value.

| run | arm | ctx | tok/s | decode s | hit | wait_ms | body_ms | io_ms | gpu at launch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | A | 262144 | 4.884 | 78.63 | 0.8460 | 96.086 | 186.469 | 70.619 | 56% |
| 2 | B | 32768 | 5.061 | 75.88 | 0.8446 | 90.753 | 179.916 | 72.640 | 0% |
| 3 | B | 32768 | 5.067 | 75.79 | 0.8461 | 92.876 | 179.349 | 69.449 | 0% |
| 4 | A | 262144 | 5.072 | 75.70 | 0.8463 | 93.603 | 179.082 | 68.707 | 0% |
| 5 | A | 262144 | 5.111 | 75.13 | 0.8454 | 90.028 | 177.664 | 71.504 | 0% |
| 6 | B | 32768 | 5.086 | 75.50 | 0.8452 | 90.767 | 178.425 | 71.394 | 0% |
| 7 | A | 262144 | 5.031 | 76.33 | 0.8458 | 94.640 | 180.768 | 68.603 | 0% |

**Paired deltas, per round.** Round 1 (contended) **+3.62%**, round 2 **−0.10%**,
round 3 **−0.49%**: median **−0.10%** (−0.005 tok/s). The identical-arm band is
wider than the effect — the four A runs spread 4.884-5.111 (4.6%), or 5.031-5.111
(1.6%) once the contended run is excluded, and the three B runs spread 0.5%. A
direction needs two rounds of its own; here the two clean rounds point opposite
ways and both are inside the drift band. **The cap is neutral on v5.20.**

**The counters say the mechanism is gone, not just the clock.** `wait_ms` is the
quantity §5o used to prove the cap was relieving paging: it measured **201 ms
uncapped against 98 ms capped**. This batch measures **90.03-96.09 uncapped** (A
median 94.12, or 93.60 with the contended run excluded) against **90.75-92.88
capped** (B median 90.77); the per-round paired wait deltas are +5.33, +0.73 and
−0.74 ms, so the arms are not separable on it either way. The point is not the
1.4% between these two arms — it is that the ~100 ms/token the cap used to buy
back is already absent from the *un*Capped arm, which is `45f55bb` working, seen
from the other side. `io_ms` is flat across both arms (68.6-72.6) and the hit rate
does not separate them either (0.8446-0.8463), so there is no second effect hiding
under the first.

**Output invariance.** `sha256` of the generated text is `95aa4531b4ea2304…` in all
seven runs, 384 tokens each, `stop=maxTokens` throughout. The cap changed no token,
so nothing here is a numerics question.

**Verdicts.**
- **§5o's +9% arm is closed as a lever.** Do not recommend `--max-context 32768`
  for decode speed on a machine that keeps its tuned budget: the headroom it was
  buying is now released by default. It stays what the wiki documents it as — a
  capacity ceiling — and it still has the one real effect the code shows: because
  `capacity: maxContext` (`sources/TinyTitan/Runtime/Inference/RealForwardRunner+BuildCore.swift:310`),
  a cap bounds how far the indexer and KV buffers may grow, so a request above the
  window is refused rather than allowed to page. On a long-prompt workload that is
  a blast-radius control, not a speed control.
- **The configuration search for ≥6 tok/s on this Mac has no untried flag left.**
  The shipped default measures 5.03-5.11 here (median 5.05), agreeing with §5p's
  5.045 and sitting under §5t's contended 5.713 server-protocol median. Every
  candidate this campaign turned — budget (§8), context (§5u), rdadvise (§8),
  knobs (§8), MTP (§8), co-execution (§8) — is now measured closed.
- **New reading worth carrying:** during each run `memory_pressure -Q` falls to
  **8-11% free** (77-80% at the launch instants) while the 11.88 GiB tier is
  resident. That is the margin the 12 GiB budget is chosen to fit, and it is direct
  evidence the budget sits at the edge rather than comfortably inside: anything
  else opening on this machine is a few hundred megabytes away from pushing the
  expert cache out — the collapse shape the gate0 preflight hunts. It also means
  the paired-batch free-memory guard is measuring the wrong instant for this model:
  a launch-time 77% free says nothing about the 9% the run runs at.
