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

## 5v. The continuous weight feeder, tested on the drive itself — the drive is already continuous (2026-10-15)

The operator's idea, stated as the premise to be tested: *"it makes no sense to
make it read for a couple of seconds, then stop, compute and read again. the SSD
can read constantly while a user has an active session… the idea is to have a
constant weight feeder instead of a chunk-by-request reader."* That is a claim
about duty cycle, so it was measured as one rather than argued from §8's closed
lever list. Nothing in the probe touches the engine: the samplers are external,
and the decode arm runs the shipped default with **no flags at all** — the 12 GiB
tier in its logs is the host-sized derivation on this 24 GB Mac, not an override.

**Provenance and protocol.** Tree `b8002b0`, whose only commits past the `v5.20`
tag are docs-only (`git diff --stat v5.20..HEAD` = one `docs/` file), so the
binary is the released engine. 24 GB M3 (Mac15,3), macOS 27.0.1, Swift 6.4, AC
power, empty model-process guard before launch, one CLI process at a time. Two
arms, each a cold run: **A** = `TINYTITAN_RUNNER_STATS=1` decode, **B** = the same
plus `TINYTITAN_ROUTE_TRACE` for the offline simulator. Both sampled every second
by external `iostat -d -w 1` and an `ioreg` AGX poller. Prompt as §5u, `--max-new
512 --temperature 0`, exit 0 in both arms. Driver `/tmp/feeder_probe_run.sh`, data
`/tmp/feeder-probe/`. Preconditions at launch: 81% memory free, **swap already
5.47 GB used**, GPU reader 61%.

**Result 1 — the drive never stops, at one-second resolution.** Taking
arm A's own clock window (decode begins 9 s after launch, runs 113.8 s), the
`disk0` series over 112 one-second samples reads:

| mean | median | min | max | samples at zero | samples under 400 MB/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1359.5 MB/s | 1353.0 | 753.0 | 1685.7 | **0** | **0** |

Against the four-thread `ParallelExpertReader` ceiling recorded in §5o and
`docs/v4-core-design.md` (3.92 GB/s) the drive sits at **34.7% duty**, so
the operator's instinct that the SSD has spare capacity is correct. What this
table does *not* show, and what §5w found, is that "no sample at zero" was a
statement about the sampler's resolution: a 1 Hz series averages a 51 ms burst
into the second around it, so alternation at the scale that matters to a 217 ms
token is invisible here. Reading 2 was the mistake that followed from that.

**Result 2 — what a perfect feeder could still buy is 0.075%.** *Refuted by §5w,
kept because the arithmetic is the reason the refutation matters.* `io_ms=72.254`
is the time spent in miss reads per token and `io_hidden_pct=99.77` says how much
of it already runs under compute, so the exposure is `72.254 × 0.0023 = 0.166` ms
of a 222.27 ms token (1000/4.499), and a feeder that removed *all* remaining
exposure moves 4.499 → 4.502 tok/s. The engine's `expert_load_p50/p95/p99` of
2/8/16 ms and the `io_host_waits_avoided=8944` of 15,584 waits are the machinery
already spending that idle capacity. The flaw is the instrument, not the
arithmetic: `io_hidden_pct` compares the read's span with the completion
timestamps of the command buffers that ran alongside it, so it answers "did the
*host* have to block", and the routed command buffer is event-gated
(`sources/TinyTitan/Runtime/Inference/RealForwardRunner+DecodeMoE.swift:372`) —
the *device* blocks instead, and that is the time the token pays. §5w measures
the device side.

**Result 3 — the continuous feeder already exists, and its limit is RAM, not the
drive.** `prefetch_issued_per_token=39.54` against `prefetch_adopted_per_token=18.23`
means the ring issues ~20,244 speculative reads over the run (`spec_ops=20243`)
at full stride width — about 493 MB/s of pure prediction, on top of the 920 MB/s
the demand misses cost. Demand plus speculation is 1413 MB/s against the 1359.5
measured, a 4% cross-check that the two series account for the whole drive load.
The ring reports `free=0.99` — it runs with about one spare slot — so it is
throttled by where a read can *land*, not by bandwidth. This is the sentence the
feeder concept dies on: a constant reader already has the bandwidth it wants, and
cannot use more of it because the 96-slot tier has no room for the bytes.

**The misses are retention-shaped, not cold-read shaped.** Of 37,807 demand misses,
**33,239 are reloads** of experts already loaded once, against 42,535 evictions, at
hit rate 0.8472. Reading earlier does not help an expert that was evicted because
nothing was holding a slot for it; that is a placement and capacity question, and
capacity at 12 GiB is the operator's own ceiling on this machine.

**The offline feeder simulator, and why its rates are withheld.**
`benchmark/continuous_feeder_sim.py` (new) replays a route trace as a timeline —
one serial drive, the engine's LFU-with-LRU-tiebreak placement, and speculative
reads that may only issue in the intervals the demand reader leaves free — under
five arms: `burst` (control), `prev_token`, `recency3`, `hot_set8`, `recurrence`
(read an expert back before its settled recurrence interval says it is due). 26
tests in `benchmark/test_continuous_feeder_sim.py`, registered in CI's model-free
python list. On the real capture — committed as
`benchmark/expert-cache/route-trace-qwen38-4bit-v520-20261015.txt.gz`, arm B's,
24,528 layer steps after `--skip-steps 480`, 96 slots/layer — it reports:

```
burst       demand= 56433 spec=      0 cov=0.000 waste_MB=     0.0 exposed_ms=   9375.2 duty=0.446 REFUSED
prev_token  demand= 54908 spec=   2732 cov=0.027 waste_MB=   257.5 exposed_ms=   9117.1 duty=0.457 REFUSED
recency3    demand= 53136 spec=   7098 cov=0.058 waste_MB=   888.8 exposed_ms=   8682.3 duty=0.480 REFUSED
hot_set8    demand= 54158 spec=   4267 cov=0.040 waste_MB=   473.5 exposed_ms=   8658.0 duty=0.465 REFUSED
recurrence  demand= 54291 spec=  22533 cov=0.038 waste_MB=  9228.7 exposed_ms=   8138.5 duty=0.616 REFUSED
calibration: REJECTED (burst projects 5.72 against measured 4.50 tok/s, 27.1% off)
```

The control arm is 27.1% too fast against the clock measured in this same
section, which means the timeline is missing ~47 ms/token of per-step cost. The
tool now says so and exits 1 instead of printing quotable rates — the gate is tested
(`test_refuses_every_projection_when_the_control_arm_misses_the_measurement`).
Two readings survive the rejection, because they are within-arm and structural:
the `prev_token`/`recency` shape removes 0-6% of demand reads while the resident
set is the last-seen set — a due-soon predictor is the only shape that spends idle
drive time usefully — and `recurrence` buys its 13% exposure reduction at 9.2 GB
of wasted reads, 20× the waste of the next arm. Neither is a token-rate claim.

**Deviations, all of them.** (1) Arm B's 4.733 tok/s is **not** a clock: the trace
capture does an unbuffered `write()` per layer per token (~480 syscalls/token), so
its rate is perturbed by design and only its trace was used. (2) The `ioreg` AGX
"Device Utilization%" reader showed 61% with the machine idle and 69.6% mean during
decode, so it is not an occupancy instrument on this host and no conclusion here
rests on it — §5u's <15%-at-launch requirement cannot be met by that gauge. (3) One
run per arm, not a paired batch: the finding is a counter and a duty cycle, not a
few-percent clock delta. (4) Swap was already 5.47 GB used at launch. (5) The `disk0`
series is whole-volume, so it includes this machine's other I/O; the 4% agreement
with the engine's own byte counters is what bounds that error.

**Verdicts.**
- **The chunk-by-request → continuous-feeder change is closed by measurement, not
  by §8.** *Amended by §5w:* the prize was 0.075% only under `io_hidden_pct`, which
  measures host blocking, and the routed command buffer is event-gated so the device
  blocks instead. The device-side ledger prices the same concept at **52.8 ms of a
  217 ms token (24%)**, and §5w's verdicts keep the closure — the burst is already at
  97% of the drive's ceiling — on a different and correct basis. "0 idle samples in
  112" stands as a statement about a 1 Hz sampler, not about the drive.
- **The live headroom is host-side, and it is bigger than everything the SSD path
  can still give.** Named per-token spans in arm A read `head_ms=11.849`,
  `rdadvise_ms=7.863` over 35,430 calls, `wait_ms=109.113`, `body_ms=202.936`,
  `cache_plan_ms=4.877`, `gather_ms=2.329`, `loop_*=2.82` — and because `body_ms`
  overlaps `wait_ms` these cannot be summed into a budget. That the footer's spans
  do not decompose a 222 ms token is itself the next instrument to build, and it is
  where a CPU lane belongs: bookkeeping off the critical path, not tensor compute
  (which §8 and the operator's no-quality-loss rule both keep closed).
- `free=0.99` is the number to watch if this is ever reopened: it says the ring is
  one spare slot from starved, and any change that frees expert-cache slots without
  giving up residency converts directly into feeder room.


## 5w. The device-side token ledger — the feeder premise was right and §5v's instrument could not see it (2026-10-15)

**What this section overturns.** §5v priced the continuous-feeder idea at 0.075% of a
token on the strength of `io_hidden_pct=99.77`, and reported the drive's 1 Hz duty
series as proof that there was no burst/idle alternation to exploit. Both readings are
wrong, and they were wrong for one reason: each measured a *host* or a *sampler* where
the question is about the *device*. This section measures the device. It lands on the
operator's side of the argument — the SSD really does burst and then wait, and the wait
is 52.8 ms of a 217 ms token — and it also shows the prize is not collectable the way
§5v assumed, because the burst is already running at 97% of the drive's measured ceiling.

**Why `io_hidden_pct` cannot answer this question.** The counter is defined as
`100 × (1 − exposedIo/missIo)` (`sources/TinyTitanLib/ServerModelSession+Diagnostics.swift:324-329`),
and `totalExposedIoNanos` is built by comparing a read's span against the *completion
timestamps of the command buffers that ran alongside it*
(`sources/TinyTitan/Runtime/Inference/RealForwardRunner+DecodeMoE.swift:298-321`,
`RealForwardRunner+DecodeMoEFinalize.swift:44-58`). So it answers "did the **host**
have to block". It cannot answer "did the **GPU** go idle", because the routed command
buffer is event-gated on the device —
`routedCB.encodeWaitForEvent(token.event, value: token.value)` at
`sources/TinyTitan/Runtime/Inference/RealForwardRunner+DecodeMoE.swift:372` — which
means that when bytes are late, the device waits and the host does not, and the counter
reports the token as essentially fully hidden. The engine already exposes the device-side
truth: `recordKernelGPU` feeds `kernelGPUOccupancy()`, which merges intervals so
overlapping buffers are not double-counted, and `kernelGPUGaps()`, which charges each
true idle interval to the role pair it separates
(`sources/TinyTitan/Runtime/Inference/RealForwardRunner+Diagnostics.swift:42-140`).
Nobody had published those two lines for this build.

**Provenance and protocol.** Tree `8c61f14`; `git diff --stat v5.20..HEAD` touches
`benchmark/`, one workflow file and `docs/` and **no file under `sources/`**, so the
binary is the released v5.20 engine. 24 GB M3 (Mac15,3), macOS 27.0.1, Swift 6.4
(`arm64-apple-macosx27.0.0`), AC power, `System-wide memory free 83%`, empty
model-process guard before launch, one CLI process at a time, **no engine flags** — the
12 GiB tier is this host's own derivation. Prompt "Write a detailed essay about the
history of computing.", `--temperature 0`, greedy. Capture arm `--max-new 512` with
`TINYTITAN_RUNNER_STATS=1 TINYTITAN_KERNEL_STATS=1`; the four A/B arms `--max-new 384`
with the same two env vars. Exit 0 in all five. Swap was **7.63 GB used before the
batch and 7.90 GB after**, which is well above the comfort line and is reported as a
deviation below. Drivers `/tmp/occupancy_probe_run.sh` and `/tmp/gap_slot_ab.sh`, data
`/tmp/occupancy-probe/` and `/tmp/gap-slot-ab/`.

### The ledger of one decode token (capture arm, 4.601 tok/s)

`busy_ms=58689.240 span_ms=115393.980 occupancy=50.9% busy_share_of_decode=52.7%`,
against `decode_s=111.283` for 512 tokens:

| part of a 217.34 ms token | ms/token | how it was obtained |
| --- | ---: | --- |
| GPU busy | 113.00 | `busy_ms/512`, minus the 832 ms of kernels whose role names are `prefill_*` |
| GPU idle, in named gaps | 101.98 | sum of the six gap pairs whose counts are decode-sized (table below) |
| unattributed | 2.36 | 1.1% of the token |

The ledger closes to 1.1%, and it is a *partition* rather than a pile-up: `kernelGPUGaps()`
advances a single frontier, so no interval is counted twice and overlapping command
buffers manufacture no gap. The caveat is that `busy_per_token_ms` as printed divides the
whole session's busy time by *decode* tokens, so the prefill rows would be smeared across
512 tokens; the 832 ms subtraction is what makes the first line honest, and without it the
closure would look better than the measurement is (the printed figure is 114.63).

**GPU occupancy is 50.9%, and the idle is not spare worker capacity — it is two
mechanisms, and only one of them is the drive:**

| gap pair | ms/token | count | ms per occurrence |
| --- | ---: | ---: | ---: |
| `moe_phase1_hit → moe_phase1_miss_fixup` | **52.803** | 15,549 | 1.739 |
| `shared_expert → moe_phase1_hit` | 26.766 | 15,499 | 0.884 |
| `shared_expert → moe_phase1_2_routed` | 16.070 | 8,964 | 0.918 |
| `embed → attn_norm_qkv` | 3.008 | 511 | 3.014 |
| `moe_phase2_routed → attn_norm_qkv` | 1.842 | 15,298 | 0.062 |
| `moe_phase1_miss_fixup → moe_phase2_routed` | 1.493 | 15,564 | 0.049 |
| *(prefill-only pairs, excluded above)* | 6.238 | 295 | 10.83 |

The largest single line is the answer to the feeder question: **a quarter of every token
(52.8 of 217.34 ms) is the device sitting idle between the hit-phase kernels and the
miss-fixup kernels, waiting for bytes.** Of that, `io_completion_to_fixup_ms=1.6549` is
host submit latency — 3.1% — and the remaining 51.1 ms is the read itself.

### The burst is at the ceiling, so the bill is the bytes, not the pipe

`expert_read_mib / tokens` divided by the four-thread `ParallelExpertReader` ceiling
recorded in §5o and `docs/v4-core-design.md` (3.92 GB/s) predicts the gap without using
it. Five arms, spanning 1.87× in read volume:

| arm | slots | tok/s | MB read/token | predicted gap | measured gap | ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| armA | 96 (default) | 4.601 | 203.55 | 51.93 ms | 52.803 | 1.017 |
| r1a | 96 (default) | 5.015 | 207.63 | 52.97 ms | 56.348 | 1.064 |
| r2a | 96 (default) | 5.050 | 207.01 | 52.81 ms | 53.872 | 1.020 |
| r1b | 48 | 4.574 | 380.19 | 96.99 ms | 96.056 | 0.990 |
| r2b | 48 | 4.398 | 380.72 | 97.12 ms | 102.999 | 1.061 |

The prediction is `bytes ÷ 3.92 GB/s` and nothing else — no tuning, no fitted constant —
and it lands within 6.4% on every arm. Two more identities pin the mechanism:
`expert_read_mib / expert_misses` is **2.7689 MB in all five arms, exactly `expertStride`**,
so a miss costs one full stride and read volume is fully explained by the miss count; and
`gap_total / gap_count ÷ (experts_per_miss_layer × stride)` puts the *delivered rate inside
each burst* at 3.85, 3.68, 3.84, 3.95 and 3.69 GB/s — **mean 3.80 GB/s, 97% of the ceiling**.

So the operator's description of the machine is correct and the diagnosis that followed
from it is not. The SSD does stop and start — §5v's 1 Hz sampler simply could not see a
1.7 ms burst — but when it is asked for bytes it already delivers them at essentially the
full rate the drive was ever measured capable of, in bursts of 1.74–2.51 ms. There is no
idle pipe to fill that would make these particular bytes arrive sooner. What would help is
moving them *earlier*, which needs them to be known earlier, and the router decides which
experts a layer needs only after that layer's routing readback. The ring already tries to
beat that clock and lands 18.56 of 73.51 demand misses per token (25%) with `free=0.99` —
about one spare landing slot.

### The falsifier, run before measuring

The claim under test was *"the gap is the drive delivering bytes."* The falsifier written
down first: cutting the cache to 48 slots must raise the miss count and the gap together;
if the gap stays put while misses move, the reading in this section is wrong. Arms ran
interleaved `r1a(96), r1b(48), r2b(48), r2a(96)` — the shipped arm first and last, as the
drift check — and the counters were near-deterministic while the wall clock was not:

| paired round | miss/token 96 → 48 | gap/token 96 → 48 | tok/s 96 vs 48 | Δ |
| --- | ---: | ---: | ---: | ---: |
| round 1 | 74.99 → 137.31 (×1.831) | 56.348 → 96.056 (×1.705) | 5.015 vs 4.574 | +0.441 (+9.6%) |
| round 2 | 74.76 → 137.50 (×1.839) | 53.872 → 102.999 (×1.912) | 5.050 vs 4.398 | +0.652 (+14.8%) |

**The falsifier did not fire.** Misses ×1.83, gap ×1.70–1.91, same direction in both
rounds. Mean paired advantage of the shipped 96-slot default: **+0.546 tok/s, +12.2%**,
with the two rounds agreeing in direction and disagreeing in magnitude by 5 points of
percent — the spread that §5v's single runs would have had to guess. The response hash is
`95aa4531b4ea2304` in **all four** arms, so the slot count moves no numerics at all: this
is a scheduling and residency change with byte-identical output, and the 12.2% is pure
queueing.

**And the instrument correction, in its strongest form:** across those same four arms
`io_hidden_pct` reads 99.84, 99.90, 99.86 and 99.85 — a spread of 0.06 points — while the
clocks differ by 12.2% and the read-bound gap by 70%. `wait_ms` even *falls* (92.562 →
74.139) as `io_ms` *rises* (70.939 → 105.441), because at 48 slots the host is never the
one blocked. An instrument that varies by 0.06 points across a 12.2% effect cannot bound
that effect at 0.075%; §5v's Result 2 was reading the wrong clock.

### The two idle families trade against each other

The second family — `shared_expert → moe_phase1_*`, 42.835 ms/token in the capture arm —
is host turnaround: router readback, `cache_plan_ms=4.898`, `gather_ms=2.248`,
`rdadvise_ms=7.624` over 35,260 calls, `cb1_ms=5.079`, encode. The shared MLP
(`shared_expert` = 5.142 ms/token of GPU work) is what is queued to cover it, and it is
far too small. But this family is **not independent of the first**: at 48 slots it halves
to 19.930 and 20.386 ms/token while the read-bound gap doubles (those totals include a
`shared_expert → moe_phase1_miss_fixup` pair worth 0.975/1.124 ms/token that does not
appear at 96 slots at all). Per occurrence the dominant pair *falls*
from 0.730 to 0.393 ms. The reason is queue depth — when the drive is behind, the GPU has
already-queued work to chew on and the host's bookkeeping disappears under it; when the
drive is ahead, the host is exposed at the front of an empty queue. So the 102 ms of idle
is one queueing system, not three prizes, and a change that "fills" family 2 with work
that then has to wait on the same bytes buys nothing.

### Answer to the tri-worker premise

The premise was: GPU is busy but the ANE and CPU are idle, so reassign slices. The ledger
says the machine is 50.9% occupied and the whole idle half is attributable to (a) bytes
that are not yet known and (b) host bookkeeping the dependency chain will not let the GPU
start earlier. A second worker could only take a slice that does **not** depend on the
router readback, and the ledger names what those are: the shared MLP (5.1 ms/token,
already on the GPU, already queued ahead of the wait — `RealForwardRunner+DecodeLayers.swift:211-223`),
the sampler (0.285 ms/token), and the embeddings (0.106 ms/token). Together 5.5 ms of a
217 ms token, §8's CPU co-execution arm already measured that transfer at **−22.6%** on
bus contention, and the ANE decode path is unreachable. That is the honest size of the
tri-worker idea on this engine: the idle is a dependency chain and a drive, not an
unassigned worker.

**Deviations, all of them.** (1) The capture arm is a *capture* arm:
`TINYTITAN_KERNEL_STATS=1` records an interval per command buffer, so 4.601 tok/s is not
a published level and none of the rate claims above use it — the paired arms' 5.015/5.050
against 4.574/4.398 are the levels. (2) n=1 for the occupancy run; its role and gap
per-token figures are reproduced in shape by all four A/B arms but only the gap magnitude
was re-measured. (3) Swap was 7.63 GB used before the batch and 7.90 GB after, and the
48-slot arm ran with a smaller resident tier against that pressure — this is a plausible
cause of round 2's slower 48-slot pair, and it is why the paired *direction* rather than
the paired *magnitude* carries the finding. (4) The gap list is truncated at the engine's
own top-8, so the 2.36 ms residual includes small real gaps that were never printed; the
closure figure is therefore an upper bound on the attribution, not an exact balance.
(5) The `prefill_*` subtraction from `busy_ms` is by role name, not by timestamp, so any
prefill work recorded under a decode role name stays in the 113.00 line. (6) `--max-new`
differs between the capture arm (512) and the A/B (384); the counters are per token and
the two agree — 203.55 vs 207.63 MB/token, 52.803 vs 56.348 ms/token gap — but they are
not the same run and are not treated as one.

**Verdicts.**
- **The continuous-feeder premise is confirmed as a description and closed as a lever.**
  The drive does burst (1.74–2.51 ms) and idle, §5v's sampler could not see it, and the
  burst costs 52.8 ms/token — 24% of the token, not 0.075%. But the burst already runs at
  97% of the drive's measured ceiling, so the remaining lever is *earlier bytes*, not
  *faster bytes*, and earlier bytes need either prediction that lands (slot room, `free=0.99`)
  or a router result that arrives sooner. Neither is a feeder rewrite.
- **The slot count is worth 12.2% and it is the largest single knob measured in this
  campaign** — larger than every §6/§8 arm here. It is already at the maximum the
  operator's 12 GiB cap on this 24 GB Mac allows, and §5q's rule is what sets it. Nothing
  to ship; everything to remember if a bigger host appears, because this section says a
  host with room for ~200 more slots should be measured as a ~12% decode win, not a rounding error.
- **The open seam is the 42.8 ms/token of host turnaround, and it is the only part of the
  idle that is neither drive-bound nor RAM-bound.** §5m already priced the cheapest version
  of it (rdadvise issue-count coalescing) at ≤2–3% and parked it on paired evidence, so a
  real attempt needs a different shape: not fewer syscalls but bookkeeping issued a layer
  ahead of the wait. That is an implementation question, not a measurement, and it is where
  the next session should start.
- **`io_hidden_pct` is unfit for any future overlap claim.** Keep the counter — it is the
  right instrument for host blocking, and `io_host_waits_avoided=8964` of 15,564 waits is
  a real achievement — but a percentage that moves 0.06 points across a 12.2% clock change
  must not be used to bound one. The device-side lines (`TinyTitan kernel busy_ms=…`,
  `TinyTitan gap …`) are the ones to publish, and they are already in the build.

## 5x. The completion wait is already the cheapest primitive, and what an extra command buffer costs (2026-10-15)

§5w left ~0.44 ms/layer of GPU idle that no host counter names, and the decode loop blocks once
per layer on `cb.waitUntilCompleted()` (`RealForwardRunner+CommandBuffers.swift:16`). The obvious
suspect was the wait itself: a blocking wait sleeps, and waking a thread costs tens of
microseconds, so if the *wake* were the 0.44 ms, a spin-then-block wait would recover on the order
of 20 ms of a 217 ms token. Measured directly, it is not.

`/private/tmp/waitcost/main.swift` is a throwaway spike, not engine code: it commits a 4 KiB blit
and waits for it, 4 000 iterations per arm, on an idle machine on AC power with no model process
running (record in `/private/tmp/waitcost/run2.log`).

| arm | mean | p50 | p95 | p99 | min |
|---|---|---|---|---|---|
| `waitUntilCompleted()` | 126.3 µs | 101.6 | 179.2 | 850.0 | 74.3 |
| `cb.status` spin | 245.1 | 158.8 | 697.6 | 2438.1 | 80.0 |
| `sched_yield` spin | 253.1 | 165.4 | 721.9 | 2533.3 | 95.8 |
| blocking, repeated last | 157.7 | 103.1 | 252.1 | 904.8 | 76.5 |

**The blocking wait is the cheapest arm by 56% on p50 and its tail is 3× better than either spin.**
Polling `cb.status` costs more than sleeping on it: the driver's completion path beats a userspace
loop, and a spinning loop also holds a core the engine's own host bookkeeping needs. There is no
spin-first wait to ship. So the ~0.44 ms/layer from §5w is not a scheduler artefact — it is real work
the host is doing, which is where §5w's own verdict pointed, and it cannot be recovered by changing
*how* the engine waits.

### The floor the other direction has to pay

The same spike priced a command buffer with no wait at all: `encode` p50 3.9 µs (mean 22.3, p99
368.8), `commit` p50 0.5 µs, per-command-buffer host cost mean 22.9 µs — 6.59 ms/token at 6 command
buffers × 48 layers. So the open direction is the opposite of finer slicing: **fewer, larger command
buffers**. Every added commit-and-wait pair costs the ~74 µs wait floor this table shows plus ~23 µs
of host time, which is ~10 ms/token for one extra pair per layer — a tenth of the idle it would have
to explain.

**Deviations, all of them.** (1) The spike never loads a model, so it prices primitives, not the
token; the wait figures are a floor on wake cost with the GPU otherwise empty, not an estimate of a
decode layer's share. (2) An earlier, unsaved run of the same arms gave blocking p50 101.7 / min 73.3
and a `sched_yield` p50 of 137.2 against 165.4 here; the two spin arms trade places between runs, so
this section publishes only what is stable across both — the blocking arm is the cheapest, p50 ≈ 102
µs with a ~74 µs floor, and both spin arms are clearly worse. (3) `min` columns are the best of 4 000
samples, not a bound. (4) n=2 runs, same process shape, no interleaving — enough for an ordering, not
for a magnitude difference between the two spin arms.

## 5y. The miss replay is calibrated, and what it says first is that the slot *partition* costs more than the victim rule (2026-10-15)

§5w made capacity the largest knob on this machine (+12.2% decode for a 45% miss change, byte-identical
output), so the next feeder or placement idea should be screened offline before it costs a model window.
The screen already existed in `benchmark/continuous_feeder_sim.py` but was known-biased and loosely banded;
this section calibrates it, states what it may and may not be used for, and then runs the first experiment
through it.

### Calibration against the run that wrote the trace

Arm B of §5v is the capture that produced `route-trace-qwen38-4bit-v520-20261015.txt.gz`, so footer and
replay are one workload: 511 decode tokens, `expert_misses=37724`, `prefetch_adopted_per_token=18.40`.
The engine's own two counters bracket what a route trace can see — a replay must count more than the drive
*reads* (it cannot know which misses the prefetch ring had already staged) and no more than reads plus
adoptions (the placements it really made):

| quantity | engine | replay | error |
|---|---|---|---|
| drive reads/token (floor) | 73.82 | — | — |
| placements/token (ceiling) | 92.26 | **89.28** | −3.2% |
| 48-slot ÷ 96-slot miss ratio | 1.831 (§5w paired) | **1.794** | −2.0% |
| cold-cache consistency | placements − evictions = 4 610 | one cold cache = 48 × 96 = **4 608** | 2 slots |

The last row is the structural check: if the replay's placement count and the engine's eviction count do not
differ by about one cold cache, the two are not counting the same object. They do, to 2 slots in 4 610.

**Licensed — ranking placement rules at a fixed budget. Banned — quoting a token rate from it.** §5w's
0.637 ms of critical path per avoided miss was measured in drive *reads*; the ~15.5 placements/token the
ring served without one are invisible in a route trace and not fixed, because they depend on how deep the
ring ran. A replay delta has to be scaled by that unmeasured share before it becomes money, and the
timeline that would supply it is the refused model (§5v).

### Which detail of the engine's rule is load-bearing

The plan-faithful rule (`_PlanSlots`) and the textbook LFU-with-LRU-tiebreak (`_Slots`, the one the refused
timeline uses) differ by 69% on this trace — 89.28 against 151.20 placements/token. That gap was previously
described in this tree as belonging to three transcribed details of `makeExpertCachePlan`. Measured one at a
time, none of the three is it:

| arm | placements/token | vs shipped |
|---|---|---|
| plan rule as shipped | 89.28 | — |
| use count raised only on misses, not on hits | 88.24 | −1.17% |
| `useClock` advanced per expert instead of per plan | 89.31 | +0.03% |
| slots re-ranked for each miss instead of once per plan | 89.28 | +0.00% |
| expert's count **forgotten when its slot is evicted** | 102.17 | +14.4% |
| textbook `_Slots` rule, whole | 151.20 | +69.3% |

The load-bearing detail is count lifetime, and it is in the engine, not in the ranking: `expertUseCount` is a
per-layer array sized by `layout.expertsPerLayer` (`PreadExpertStreamer.swift:326`) and only
`resetExpertUseCounts()` zeroes it — which is itself env-gated (`TINYTITAN_EXPERT_USE_RESET=1`,
`RealForwardRunner+Prefill.swift:24`) and therefore **off** in every flag-free arm measured in this document.
A returning expert is priced by everything it has ever done. The textbook rule prices it at one.

**Withdrawn figures.** Four numbers published earlier in this campaign's working notes — 109.10, 111.64,
113.39 and 157.11 placements/token for a "textbook" replay — are withdrawn. The first three were three
different miss *accountings* of the same rule (no warm-up; warm-up counting every visit; warm-up counting
decode only), and 157.11 came from a probe that leaked residency: `_Slots.contains(e)` is `e in use_count`,
so restoring a count for an evicted expert makes it look resident, which is also how a broken variant of the
same probe once collapsed a miss count to 22.92. None of them is reproducible from the tree. The band claim
is now pinned by a test rather than by prose: `MissReplayTests` asserts the plan rule sits inside
[73.82, 96.87] and a drift arm asserts the textbook rule breaches the ceiling at 151.20 — the same assertion
run against the plan rule fails at 89.28, so the band discriminates.

### The ranking: capacity distribution beats victim order

Two throwaway rounds screened victim rules and budget shapes at the shipped 4 608-slot total
(`/tmp/placement-screen/`, uncommitted — see deviations). Ranking only, no token rate anywhere in these rows.

| rule and budget shape | placements/token | vs shipped |
|---|---|---|
| oracle static per-layer split (sum 4 587 slots) | 82.66 | −7.42% |
| LFU capped at 3 + one flat 4 608-slot pool | 82.97 | −7.07% |
| LRU + flat pool (capped at 3 or 8 — identical) | 85.50 | −4.24% |
| oracle static split, plain LFU inside it | 86.08 | −3.59% |
| LFU capped at 5, per-layer 96 | 86.95 | −2.61% |
| `lru_then_lfu`, per-layer 96 | 87.46 | −2.04% |
| **shipped: LFU, per-layer 96** | **89.28** | — |
| LRU, per-layer 96 | 91.05 | +1.98% |
| LFU + flat pool, uncapped | 104.56 | +17.11% |
| FIFO / random, per-layer 96 | 106.64 / 106.43 | +19.4% / +19.2% |

Read the middle of the table, not the top: the shipped rule is already the best *victim order* at this budget
(every pure alternative here is worse), and the wins come from **how the 4 608 slots are divided**, not from
who gets evicted. The per-layer decode working sets are very unequal — 174 to 415 distinct experts per layer,
mean 285, against a uniform 96-slot partition — and an oracle split of nearly the same total recovers 7.4% of
placements. A single flat pool with LRU recovers 4.2% without knowing the future; combining it with a
saturating count (cap 3) recovers 7.07%, which is within noise of the oracle. The uncapped LFU in a flat pool
is much *worse* (104.56): without a cap, one expert's lifetime count can defend a slot for the whole session,
and in a shared pool that starves every other layer — which is why the cap and the pool help together and the
cap alone is a 2% story.

Capacity ladder for reference (per-layer, LFU / LRU): 48 slots 160.13 / 157.77, 64 131.22 / 131.59, 96 89.28 /
91.05, 128 61.81 / 63.52, 192 34.77 / 34.53.

**Deviations, all of them.** (1) One trace, one prompt, 511 decode tokens; the §5w ratio it is calibrated
against came from a 384-token run, so the calibration compares a *response*, not a level. (2) The screen
scripts and their logs are uncommitted throwaways under `/tmp/placement-screen/`; only the shipped arm
(89.28), the 48-slot arm (160.13) and the drift arm (151.20) are reproducible from the tree, via
`benchmark/test_continuous_feeder_sim.py`. A reboot costs the ranking rows. (3) Deltas of 2–4% are inside the
instrument's own 3.2% level bias, and §5w showed a 45% miss change buying 12.2% of clock — a 4% placement win
may express as ~1% or as nothing once the ring and `free=0.99` absorb it. (4) The oracle split uses future
knowledge (per-layer distinct counts read off the same trace); it is published as the ceiling a static split
could reach, not as a policy. (5) The `window` arms at 32 and 128 tokens returned exactly the baseline because
the cutoff (`window × 48` rows) covers the whole session — those two are **inconclusive, not "no effect"**;
the window-8 arm (88.22, −1.19%) is the only one that tested the shape. (6) An earlier `lfu_decay` arm was a
no-op by construction — halving every count is a uniform rescaling and cannot change a ranking — caught before
any of it was reported. (7) The 128- and 192-slot rows exceed the operator's 12 GiB cap on this 24 GB Mac and
are offline arithmetic, not configurations to run here.

**Verdicts.**
- **The victim rule is not the open seam.** Shipped LFU beats LRU, FIFO and random at this budget, and every
  tiebreak reshuffle tried here moved placements by under 3%. Do not spend a model window on a cleverer LRU.
- **The partition is.** 512 experts per layer and a uniform 96 slots each, against per-layer working sets of
  174–415. Any real attempt should rebalance the *budget across layers* (or share one pool), not reorder
  evictions. Both are architectural here: `slotCount` is a `let` per streamer
  (`PreadExpertStreamer.swift:93`) and the plan guard assumes one uniform capacity per layer, so a non-uniform
  split is a load-time data-layout change, not a tuning knob. It needs the operator to ask for it.
  **— Corrected by §5z.** The streamers are already one per layer with their own `slotCount`, and the split is
  one expression (`Model+LayerStreamers.swift:168`); the "architectural" claim was wrong. §5z also shows this
  lever's own oracle ceiling is ~1 point of hit rate, so it is not worth the change.
- **One cheap live arm survives this screen**: a saturating use count (cap ~5) inside the comparator. It is a
  clamp in `makeExpertCachePlan`, changes only which slot is given up, and should be output-invariant —
  residency is a cache, not semantics, and §5w already showed a placement change holding the bytes identical.
  Offline it is worth −2.6% placements, which is *inside* the instrument's own 3.2% bias: that licenses one
  paired run, not a claim.
- **Nothing here is a tok/s number.** This section prices placements. The engine's clock is still what §5w and
  §5v measured, and no row above has been run on the machine.

## 5z. A heat split learned on one session does transfer to another — and the bar it has to clear is arithmetic, not empirical (2026-10-16)

§5y ended by naming the partition as the open seam and calling a non-uniform split architectural. Both halves of
that are wrong, and this section is the experiment that says so. It also answers the question §5y could not: a
split can only ship if it is learned from an **earlier** session, so the number that matters is not what an
oracle split wins on its own trace, but what a split carries across captures.

Everything below is reproducible from the tree. `benchmark/heat_split_sim.py` (new) plus
`benchmark/test_heat_split_sim.py` (17 tests) commit the screen that §5y left as a `/tmp` throwaway, together with
the third route capture it needed. Same licence as §5y: **ranking budget shapes at a fixed slot total. Banned:
quoting a token rate from any of it.**

### The captures

| capture | decode rows from | distinct experts/layer | demand | what it is |
|---|---|---|---|---|
| `20261011` | position 30 | 146..409, mean 261 | 480 req/token | pre-v5.20 build, the campaign's first trace |
| `v520-20261015` | position 10 | 174..415, mean 285 | 480 | §5y's arm B, the shipped 96-slot default |
| `v520-prompt2` | position 33 | 122..417, mean 269 | 480 | **new** — same build, a different prompt |

`v520-prompt2` is the falsifier's control: `benchmark/expert-cache/run-record-qwen38-4bit-v520-20261016-prompt2.txt`
fixes its provenance (machine, build, exit code, footer counters) and calibrates it the way §5y calibrated arm B.
Its replay is 102.10 placements/token against an engine bracket of 94.27 (drive reads) to 114.66 (reads plus
adoptions) — **−10.8%** from the ceiling. §5y's bias on its own capture was −3.2%. The replay's level error is
workload-dependent, which is a second, independent reason the tok/s ban is not caution but arithmetic: the bias is
larger than every effect this section measures.

Deviation on that run: the argv was never logged, so the run-record says so and the command line is treated as
unrecovered. The identity lines and counters in that file are the evidence; nothing else about the invocation is claimed.

### Do two sessions of the same model route the same experts?

| pair | profile correlation | shared rows | identical rows | picks in common |
|---|---|---|---|---|
| `20261011` × `v520-20261015` | +0.976 | 18 384 | **0** | 13.206% |
| `20261011` × `v520-prompt2` | +0.960 | 18 240 | **0** | 5.117% |
| `v520-20261015` × `v520-prompt2` | +0.957 | 18 384 | **0** | 3.209% |

Chance for a top-10 pick out of 512 is 1.953%. So the two same-prompt captures agree on 13.2% of picks — about 7×
chance, which is the prompt-memory an engine could in principle exploit — but the different-prompt pair sits at
3.2%, barely above chance, and **no row in any pair is routed identically**. Meanwhile the per-layer *shape*
correlates at +0.957 to +0.976 everywhere, including across the pair whose routes are nearly independent.

That is the result that decides whether a shipped table is data or memorisation: **which layer is busy is a property
of the model; which expert is busy is a property of the conversation.** A per-layer budget table is therefore
portable in the way a per-(layer, expert) table would not be.

### The transfer table

`split_slots` spends **exactly 4 608 slots** — the shipped tier — in proportion to one capture's heat profile, floor
10, ceiling 512. Each row replays that budget on a *different* capture. All figures are placements/token at cap
none, engine rule as shipped.

| scored on | shipped uniform | oracle (own trace) | transfer from the other two |
|---|---|---|---|
| `20261011` | 91.77 (80.88% hit) | 88.38 −3.69%, +0.71 pt | 88.58 (+0.66 pt, keeps 94.0%) / 88.25 (+0.73 pt, keeps 103.7%) |
| `v520-20261015` | 89.28 (81.40% hit) | 85.54 −4.19%, +0.78 pt | 85.32 (+0.82 pt, keeps 105.9%) / 84.91 (+0.91 pt, keeps 117.0%) |
| `v520-prompt2` | 102.10 (78.73% hit) | 97.69 −4.32%, +0.92 pt | 98.36 (+0.78 pt, keeps 84.9%) / 98.56 (+0.74 pt, keeps 80.3%) |

Six transfers, six better than the shipped uniform split, and four of them beat an oracle that read the future of
the very trace it is scored on. With the saturating count added (cap 3, cap 5) all eighteen transfers still win;
the keeps-fraction column, like-for-like at each cap, runs 74.4% to 128.3%.

### The bar, and why no budget shape reaches it

`docs/plan-qwen38-decode-12gb.md:150` sets B1's acceptance as **≥ +7 points of hit rate** over the reproduced
control. The measured transfers deliver **+0.66 to +1.52 points**. That is a factor of four to eleven short — and
the miss is not a transfer penalty, it is a ceiling: the *oracle* split, which knows the trace it is scored on, only
reaches +0.71 to +0.92 points. Every static partition of 4 608 slots is bounded by that.

The reason is units. Hit rate is `1 − placements / demand` with demand 480, so a percentage of placements buys
one-fifth of a point per percent: §5y's headline −7.4% of placements is 6.6 placements, 1.4 points. To buy +7 points
at this tier you must remove 33.6 placements/token, i.e. cut 89.28 by **37.6%**. The only arms in this campaign's
ladder that large are 128 slots/layer (−30.8%) and 192 (−61%), and §5y's deviation 7 already records that both blow
past the operator's 12 GiB cap on this 24 GB Mac.

So the plan's bar and §5y's percentage were never commensurable, and the honest reading is the strong one: **at a
fixed 4 608-slot budget, capacity *shape* is worth at most ~1 point of hit rate.** For contrast, §3/§4 measured
Belady at the same 4 608 slots at 89.8% hit — +8.4 points over the shipped 81.40%. The headroom at this budget is
real and it is nearly all in *when a slot is given up* relative to when it is needed, which is the one thing a static
partition cannot touch and a feeder can.

### Corrections to §5y, named

1. **"Both are architectural here: … a non-uniform split is a load-time data-layout change, not a tuning knob."**
   False, and the plan had already designed the seam. Verified in the engine:
   - `Model+LayerStreamers.swift:195-202` constructs **one `PreadExpertStreamer` per layer**, each with its own
     `slotCount:`, and each streamer allocates its own slot buffers from that number
     (`PreadExpertStreamer.swift:231-259`, `321-325`). Per-layer memory is already per-layer; nothing is a shared
     fixed-size table. The whole load-time change is line 168, `let effectiveSlotCount = slotCount`, which is the
     identity `docs/plan-qwen38-decode-12gb.md:182` says a heat table plugs into.
   - Two readers assume the count is uniform, and **both clear this split without modification**. The MTP pair
     schedule needs `slotCount >= 2 × topK` = 20 (`RealForwardRunner+MTPVerify.swift:67-68`); the prefill tile
     scheduler keeps its default 8 experts/tile for any `slotCount ≥ 16` at the shipped
     `maxPendingDepth = 1` (`PrefillRoutedTileScheduler.swift:36-56`, `RealForwardRunner.swift:238`). The three
     splits' minimum budgets are 53, 58 and 43 slots, so neither gate moves — which is why the floor in
     `split_slots` is inert: floors of 10, 20 and 32 produce byte-identical budgets, and only 48 changes anything
     (oracle 97.67 vs 97.69, keeps ±2%).
   - `Engine.swift:283-284` reports `slots × stride × layers` and would need `sum(budgets) × stride`; that is
     exactly equal, because `split_slots` conserves the total by construction (`test_heat_split_sim.py` pins it).
   - `ModelExpertIO.swift:147-152` documents the no-layer-argument property as deliberate; under a split it must
     return the **minimum** budget (or grow the argument), and `PreadExpertStreamer+Plan.swift:30`'s
     `experts.count <= slotCount` guard becomes per-layer — which it already effectively is, since it reads the
     streamer's own `slotCount`.
2. **§5y's top table row pooled two levers.** "oracle static per-layer split … −7.42%" was replayed with the
   saturating count *and* the split (`screen2.py`: `replay({"cap": 3}, budgets=scaled)`). The split-only arm in that
   same table is 86.08, −3.59%. Decomposed at the exact shipped total on `v520-20261015`: split alone **−4.19%**,
   cap 3 alone **−2.04%**, both **−8.04%**. The partition is a smaller share of §5y's headline than the prose implied.
3. **§5y's table has one arm listed twice.** "`lru_then_lfu`, per-layer 96 | 87.46" is the same rule as
   "LFU capped at 3 | 87.46" — the throwaway's rank key was `(min(count, 3), last)` (`screen.py:68`). It is not a
   second, independent victim order, so the ranking table's breadth overstated by one row how many distinct rules
   were tested.
4. **The cap is not a 2.6% arm.** §5y priced it from one capture and called it "inside the instrument's own 3.2%
   bias". Across three captures, cap 3 on the uniform split is −7.92% / −2.04% / −5.03%. On two of three it is well
   outside the bias. That strengthens, not weakens, the case for the one live arm §5y left open (task #177).
5. **Withdrawn:** the 92.79 placements/token figure used for `20261011`'s shipped arm in the working throwaways
   (`crossprofile3.py` carried the note). The committed instrument reads **91.77** on that capture and the tests pin
   it; `crossprofile3.py`'s number is not reproducible from the tree and is superseded.

### Deviations, all of them

1. Three captures, two prompts, one model. `v520-prompt2` is the only different-workload pair, so the shape
   transfer rests on two of three pairs being same-prompt.
2. Every transfer is scored with a cold cache and no warm-up beyond the prefill rows, the same convention as
   `replay_misses`; a live engine's cache at decode time is not cold.
3. The replay's level bias is −3.2% on `v520-20261015` and **−10.8%** on `v520-prompt2`, i.e. larger than the effect
   under test. Only the *sign* and the cross-capture agreement survive that; the magnitude does not.
4. `20261011` was captured on a pre-v5.20 build, so a shape agreement with the two v5.20 traces partly measures
   model constancy across a build change rather than session constancy.
5. No token rate appears in this section, and the one that could be inferred from §5w's 0.637 ms/miss must not be:
   that was priced in drive reads, and the ~15.5 placements/token the ring served without a read are invisible here.
6. The oracle column uses future knowledge of the trace it scores; it is published as the ceiling a static split can
   reach, not as a policy — which is the whole of its use here.
7. `v520-prompt2`'s command line is unrecovered (see the run-record). Its identity and counters are logged; nothing
   about its flags is claimed.

### Verdicts

- **The B1 heat split does not clear its own acceptance bar.** Six transfers all beat the shipped uniform split and
  four beat an oracle, and it is still a rejection: +0.66…+1.52 points against a required +7. It does not become a
  shipped default and does not get a model window on its own.
- **§5y's "architectural" verdict is withdrawn.** The change is one expression at
  `Model+LayerStreamers.swift:168` plus a table read, and the two consumers that assume a uniform count clear it at
  the split's own minimum. Anyone told in §5y that this needs a data-layout change was told wrong.
- **Do not spend the next screen on capacity shape.** At 4 608 slots the oracle ceiling of every partition is ~1
  point of hit rate, while Belady at the same slots is +8.4. The gap is temporal, not spatial. That is where the
  operator's constant-feeder idea actually points, and the counters already show a place to look: in the
  `v520-prompt2` capture the ring **issued 40.51 prefetches per token and had 20.39 adopted**, with `free=0.99` and
  `inflight=0.01` — it is issuing for twice the depth it keeps and still leaving slots idle. Nothing in this
  section measures why; that is the next question, and it is a feeder question rather than a budget question.
- **The live arm that survives unchanged** is the saturating use count (cap ~3–5) in `makeExpertCachePlan`: −2.04%
  to −7.92% placements across three captures, output-invariant by construction, and no model run yet. Task #177.
- **Nothing here is a tok/s number.** This section prices placements and hit rate, and says how many points of hit
  rate each lever can buy at this budget.

## 5aa. The constant feeder, screened in every shape it can take — it fails as a cache change, and the reason is eviction being zero-sum (2026-10-16)

§5z closed the *budget shape* lever and pointed the residual headroom somewhere else: Belady on the same 4 608 slots
is +8.4 points while every static partition is worth ~1, so the gap is in **when a slot is given up**, not in where
slots sit. The operator's premise is exactly that temporal idea — stop reading in bursts around each request and let
the drive stream weights continuously while a session is live. This section tests the premise instead of another
knob on the same lever.

The test follows from one constraint: **a feeder can only stream what it can name.** The engine ships no
per-conversation oracle, so the naming available is (a) an earlier session's statistics, or (b) this session's own
immediate future. Four arms, all inside the same 4 608-slot tier — none of them is a budget increase wearing another
name. Same licence as §5y and §5z: **ranking placement shapes at a fixed slot total; banned, quoting a token rate.**

One half of the premise was already measured on this machine, and it is worth naming before the arms. §5v sampled
`disk0` at one-second resolution across a whole decode run and got **0 idle samples** (mean 1 359.5 MB/s, min 753.0):
the drive does not stop and wait for compute. So the literal reading — bursty reads with dead wire between them — is
closed by counters, not by argument. What stayed open is the stronger reading: keep that continuously-busy wire, and
use it to read *different* weights, chosen ahead of the request. That is the thing the four arms test, and it is the
only version of the idea the route trace can price.

### What is in the tree

`benchmark/predictive_feeder_sim.py` (new) and `benchmark/test_predictive_feeder_sim.py` (new, 23 tests in 8 classes).
They reuse §5z's three committed captures, the engine's own `_PlanSlots` placement rule and `heat_split_sim`'s slot
tier, so every figure below is reproducible: `python3 benchmark/predictive_feeder_sim.py` prints these four tables
model-free and prints no token rate. Shipped-cache reference lines: hit
**80.9 / 81.4 / 78.7 %**, placements **91.77 / 89.28 / 102.10** per token, reads 242 / 236 / 270 MiB per token. Units,
again because everything here depends on them: demand is 480 requests/token, so **1 % of placements = 0.2 points of
hit rate**, and the plan's bar (`docs/plan-qwen38-decode-12gb.md:150`) of **+7 points** is **−33.6 placements/token**.

### Arm 1 — what an earlier session can name

Top-n per layer learned on one capture, scored on another. `demand cover` is the share of the other session's 480
requests per token; `miss cover` is the share of the placements the shipped rule actually makes — the only number a
feeder can act on, because a streamed expert still costs a read and the only thing that changes is whether that read
is exposed or already done.

| learned on → scored on | width | demand cover | miss cover |
|---|---|---|---|
| `20261011` → `v520-20261015` | 96 | 70.2 % | **26.5 %** |
| `20261011` → `v520-prompt2` | 96 | 42.4 % | **24.9 %** |
| `v520-20261015` → `20261011` | 96 | 72.0 % | **33.8 %** |
| `v520-20261015` → `v520-prompt2` | 96 | 36.8 % | **25.9 %** |
| `v520-prompt2` → `20261011` | 96 | 42.7 % | **28.7 %** |
| `v520-prompt2` → `v520-20261015` | 96 | 37.6 % | **28.6 %** |
| same six pairs | 192 | 61.2 – 89.9 % | 53.7 – 71.9 % |
| same six pairs | 288 | 73.0 – 95.8 % | 69.4 – 88.6 % |

At the tier's own width, **an earlier session names a quarter to a third of the misses the cache pays for**. A random
width-96 list would name 96/512 = 18.75 % of any requests by construction, so the signal is real but small. It grows
fast with width — and the width is where the idea dies: 192 and 288 named experts per layer are more than the tier
holds, so a wider list is not a cheaper feeder, it is a bigger cache under another name, and §5z's deviation 7 already
records that 128 slots/layer blows past the operator's 12 GiB cap on this Mac.

The two demand figures above 50 % are **not prediction**. They are the one *conversation* shared by two captures:
`20261011` and `v520-20261015` agree on 432 literal `(position, layer)` rows — positions 0..8 across all 48 layers —
and every pair involving `v520-prompt2` agrees on 0. Those shared rows are prefill rows, so they are not themselves
inside the scored demand; they are the marker that the two captures are the same conversation replayed on two builds.
§5z measured the same thing from the other side: 13.2 % picks in common for that pair against 3.2 % across prompts.

### The pre-registered bar, and the one disclosed rewrite

The bar was written before measuring as *"no earlier session's top-96 covers half of another session's 480 requests"*.
**It failed** (70.2 % and 72.0 %), and per the standing rule this names which side was wrong: the assertion, not the
premise. The demand denominator was the mistake — it credits a hot list for requests the live cache had already
resident-ised, which is why the *miss* column, not the demand column, is 25–34 %. The rewritten test
(`test_demand_coverage_tracks_how_much_route_the_two_captures_share`) asserts the **relation** in both directions:
a pair that shares literal rows must cover ≥ 50 %, and a pair that shares none must cover < 50 %. If a different-prompt
pair ever crosses 50 %, cross-session prediction is real and the test has to be rewritten again to say so.

The second falsifier — miss coverage < 50 % at tier width — was **not** rewritten and holds for all six pairs. Two
test-file changes are disclosed here: the rewrite above, and a wording correction in the same docstring made after
measuring, because the first version called the shared prefill rows the mechanism when they are only the marker. Both
are in a file that has never been committed; no earlier run is withdrawn, because none was published.

### Arm 2 — pinning a learned prefix costs slots the rest of the tier no longer has

P experts held for the whole session, taken **out of** the same 96-slot tier. Δ placements/token vs the shipped tier.

| learned on → scored on | P=12 | P=24 | P=48 | P=72 | P=96 |
|---|---|---|---|---|---|
| `20261011` → `v520-20261015` | −0.55 % | **−0.73 %** | +0.45 % | +1.96 % | +59.96 % |
| `v520-20261015` → `20261011` | +0.05 % | −0.05 % | **−0.71 %** | +0.10 % | +46.64 % |
| `20261011` → `v520-prompt2` | +4.05 % | +7.65 % | +16.42 % | +31.05 % | +170.89 % |
| `v520-20261015` → `v520-prompt2` | +5.11 % | +10.06 % | +19.90 % | +35.44 % | +196.92 % |
| `v520-prompt2` → `20261011` | +5.54 % | +11.22 % | +23.00 % | +41.12 % | +199.93 % |
| `v520-prompt2` → `v520-20261015` | +6.15 % | +12.70 % | +26.01 % | +49.24 % | +235.46 % |

The best honest arm in the whole table is **+0.14 points of hit rate** (−0.73 % and −0.71 %, the two same-conversation
transfers), and it is only reachable by a list learned from the same conversation. Every different-prompt transfer is
worse than shipping nothing, monotonically in P, up to 3.4× the reads at P=96 where no live slot remains. Cost, so the
ambition is legible: one pinned expert id held in all 48 layers is 0.124 GiB of pre-read, and a full 96-wide pin tier
is **11.88 GiB** — the entire expert-cache budget spent on a list learned from another session before a single row of
it is validated.

### Arm 3 — heat-biased eviction: same slots, no reservation, only a different victim

`_HeatedSlots` subclasses the shipped rule and changes exactly one thing: a protected slot's victim price is `+inf`
instead of its LFU rank, so it gives up its slot only after every unprotected one has. It claims no slot, so a list
wider than the tier is legal, and this is the arm that would have been free if nameability converted into placements.

| arm | W=96 | W=192 | W=288 | W=384 |
|---|---|---|---|---|
| `20261011` → `v520-20261015` | +1.88 % | **−3.04 %** | +0.87 % | +1.59 % |
| `v520-20261015` → `20261011` | **−6.81 %** | −6.18 % | −3.87 % | −3.20 % |
| `20261011` → `v520-prompt2` | +24.63 % | +23.33 % | +16.94 % | +16.46 % |
| `v520-20261015` → `v520-prompt2` | +22.43 % | +23.92 % | +16.71 % | +16.12 % |
| `v520-prompt2` → `20261011` | +11.52 % | +16.53 % | +9.61 % | +9.34 % |
| `v520-prompt2` → `v520-20261015` | +20.52 % | +24.22 % | +18.47 % | +17.85 % |
| **hindsight** `20261011` → itself | −15.24 % | −10.48 % | −2.82 % | −1.97 % |
| **hindsight** `v520-20261015` → itself | −16.15 % | −7.09 % | +1.15 % | +2.33 % |
| **hindsight** `v520-prompt2` → itself | −16.16 % | −11.31 % | −4.94 % | −3.89 % |

Best transfer **+1.30 points** (−6.81 %, same conversation). Best different-prompt transfer: there isn't one — the
worst is +24.63 % of placements (`20261011` → `v520-prompt2`), i.e. **−5.24 points** against its own control. And the
hindsight rows are the arm's ceiling with zero slot cost: protecting a list learned with perfect knowledge of the very
session being scored buys **+2.91 / +3.00 / +3.44 points**. Under half the bar, with the future already known.

That is the mechanism this whole screen found, and it is not a prediction-quality problem. **Eviction inside a fixed
tier is zero-sum.** Naming 62 % of a session's misses at width 192 does not reduce them, because holding those slots
means refusing to hold the ones the shipped rule would have chosen, and the shipped rule is adapting to the session
that is actually running. High nameability plus low placement reduction is the expected shape, not a surprise.

### Arm 4 — how much of THIS session's future the engine would have to own

The honest form of reading ahead: protect what this session routes in its next W tokens. It reserves no slot, so
`W=0` is exactly the shipped rule (pinned to 9 decimal places against `placements_per_token`) and every other window
reads as a delta on it. This is an **oracle** — the engine cannot see those rows yet — and it exists to bound what any
real predictor could ever be worth.

| capture | W=1 | W=2 | W=4 | W=8 | W=16 | W=32 |
|---|---|---|---|---|---|---|
| `20261011` | −4.19 % (+0.80 pt) | −6.88 % (+1.32) | −11.74 % (+2.24) | −18.92 % (+3.62) | −29.12 % (+5.57) | −40.90 % (**+7.82**) |
| `v520-20261015` | −5.08 % (+0.95 pt) | −8.26 % (+1.54) | −12.55 % (+2.33) | −18.34 % (+3.41) | −26.48 % (+4.93) | −37.91 % (**+7.05**) |
| `v520-prompt2` | −5.59 % (+1.19 pt) | −8.87 % (+1.89) | −13.75 % (+2.93) | −21.53 % (+4.58) | −30.73 % (+6.54) | −41.67 % (**+8.86**) |

Monotone and consistent across three captures: each doubling of the window buys more than the last, ≈ +0.8/+0.95/+1.19
points at one token and **+7.05 to +8.86 at 32** — which is where the plan's +7-point bar actually sits, and it
recovers most of the +8.4 that §5z attributed to Belady at the same slots. So the temporal headroom is real, and the
oracle prices it.

What the engine can actually own: nothing. MTP is off on the delivered path (§5t's config list, line 1259) and
speculation is §8-closed; and its verify pass is **width 2** — one draft token, `StreamingMTPTypes.swift:234` — so
even re-opening speculation buys W=1, worth **+0.80 to +1.19 points** (−4.2 to −5.6 % of placements). That is not close
to the bar on any reading, and §5y's licence note applies to it exactly as it applies to every other arm here: a
replay delta cannot be turned into a clock claim at all, because the ~15.5 placements/token the ring serves without a
drive read are invisible in a route trace and that share is unmeasured. The gap between what would clear the bar (32
tokens) and what is available (0, or 1 at the cost of re-opening a closed lever) is the whole result of this arm.

### One correction to §5z, named

§5z's last verdict bullet calls the ring counters "issuing for four times the depth it keeps". The two numbers it
quotes in the same sentence are 40.51 issued and 20.39 adopted per token — a ratio of **1.99**, not four. The wording
is corrected there; the counters themselves were right and are unchanged. The mistake does not soften §5z's point:
issuing 2× the depth it keeps while `inflight=0.01` is still a scheduling question that no budget arm can answer.

### Deviations, all of them

1. Three captures, two prompts, n = 1 conversation per prompt. The "same conversation" pair is one pre-v5.20 and one
   v5.20 build, so it partly measures model constancy across a build change (§5z deviation 4 applies unchanged).
2. Cold-cache replay, no warm-up beyond the prefill rows — §5z deviation 2. A live engine's cache at decode time is
   not cold, which flatters every arm that starts from an empty tier.
3. The replay's level bias is −3.2 % on `v520-20261015` and −10.8 % on `v520-prompt2` (§5z deviation 3). Only signs,
   monotonicities and cross-capture agreement survive that; no magnitude here is a machine number.
4. Arm 4 is an oracle by construction and cannot be implemented. It is published as a price, not a policy.
5. `_HeatedSlots` overrides only `_price`; `_PinnedSlots` composes the same rule at a smaller capacity. Both were
   checked to reproduce `placements_per_token` exactly at their neutral setting, so no delta in this section comes from
   a re-implemented cache rule.
6. Hot lists rank by request count with ties broken by ascending expert index, so a shipped prefix list would be
   byte-reproducible offline; that property is tested rather than assumed.
7. Heat widths above the tier (192/288/384) are legal *for that arm* because protection holds no slot, so those
   columns are judged on their own numbers. The same widths in Arm 1's table are not affordable, and are reported as
   nameability only.
8. The driver's pin-cost line first read "per layer" for a figure that is one pin spread over all 48 layers; the
   wording was corrected before commit. The first two sweep runs printed identical figures wherever their tables
   overlapped, so no number moved; deviation 10 records the third run.
9. No token rate appears in this section, and §5w's 0.637 ms per exposed read must not be used to infer one — same
   reasoning as §5z deviation 5: that was priced in drive reads, and the ~15.5 placements/token the ring served
   without one are invisible here.
10. The sweep has been run three times. The third run, after the footer rewrite and the formatter pass, reproduces the
    second **byte-for-byte across all four tables** and differs only in the two closing lines (the old one projected
    "3.6 GB/s → 274 experts/token" from an uncited rate; the new one names its two sources: §5d's 3.57 GB/s reader
    wall → 271 experts/token, and §5v's measured 1 359.5 MB/s drive → 103). Its own wall time did not reproduce: the
    same code path took 1 497 s wall / 883 s user the second time and 138.9 s / 136.3 s the third, an unexplained 6×
    on a machine that was idle in both cases, so no runtime is claimed anywhere in this section — and nothing in it
    depends on one, because the driver's output is a pure function of three committed files.

### Verdicts

- **The constant-feeder premise fails as a cache change**, in all three shapes it can take: pinning a learned prefix
  (+0.14 point best honest arm), heat-biased eviction (+1.30 across sessions, +3.44 even in hindsight), and prediction
  width (25–34 % of the misses the cache pays for at the tier's own width). None reaches +7 and none gets a model
  window.
- **Why it fails is the useful finding.** Not because misses are unnameable — at 192 names an unrelated session
  covers 54–72 % of them — but because a fixed tier makes eviction zero-sum: acting on that knowledge displaces the
  rule that was already adapting to the live session. Any future feeder screen should be judged against this
  asymmetry first, or it will re-report nameability as if it were savings.
- **The temporal headroom §5z pointed at is confirmed and is unreachable.** 32 tokens of lookahead recovers
  +7.05…+8.86 points, most of Belady's +8.4 on the same slots; one token, which is the widest thing this engine's own
  drafter could offer, is worth ≈ +1.
- **"Read constantly" survives only as scheduling, and this instrument cannot price it.** A read moved from exposed to
  overlapped is the same read, so an arm that changes no placement can still change the critical path — which is
  §5w's measurement, not a trace replay's. The ring's own counters say the burst-vs-stream question is not the one the
  engine is stuck on anyway: on `v520-prompt2` it **issued 40.51 prefetches per token and adopted 20.39**, with
  `free=0.99` and `inflight=0.01`. It is already issuing twice the depth it keeps, and half of what it reads is
  thrown away. That is a queue-depth and eviction-honesty question, and answering it needs the live paired A/B method,
  not more offline ranking.
- **The live arm that survives unchanged** is still the saturating use count (cap ~3–5) in `makeExpertCachePlan`:
  −2.04 to −7.92 % placements across three captures, output-invariant by construction, no model run yet. Task #177.
- **Nothing here is a tok/s number.** This section prices placements, says how many points of hit rate each feeder
  shape can buy at a fixed 4 608 slots, and closes the cache-change reading of the premise.

## 5ab. Four overnight premises, each closed by a measurement — and the empty scan that had to be re-run first (2026-10-16)

The overnight mandate was to try things out rather than re-run the plan's knob list. Five
premises were taken apart with model-free instruments — an SDK header inventory, a codec
entropy screen on real blob bytes, a codec *cost* screen on the same bytes, the committed
predictor study, and §5x's µs-scale host primitives. All five closed. None of them moved a
clock, and this section claims no clock.

### The scan that read nothing, and why it is first

The first version of the header inventory below was run as

```
find "$SDK/System/Library/Frameworks/Metal.framework/Headers" -name '*.h' | wc -l   # 0
grep -rli "accel" "$H"                                                              # (no output)
```

and it returned **nothing**, which is exactly what an absent API looks like. It was not
evidence: `Metal.framework/Headers` **is a symbolic link** (`stat -f '%HT'` → `Symbolic
Link`), and neither `find` nor `grep -r` follows a symlinked starting directory by default.
`ls "$H" | wc -l` over the same path reports **97** entries, and a direct `grep -c MTLIO`
on one of those files reports **18** hits, so the files were always readable — the walk was
blind. Re-run with `find -L`, which enumerates **96** headers, is the instrument every
figure below comes from.

This is the same defect shape `tools/lint.sh` hunts in `unbounded-read` and `test-hollow`:
*a scan that read nothing reporting a pass.* The earlier session's conclusion happened to
survive the re-run; the claim is nevertheless withdrawn and re-measured, because "right" and
"proven" are different properties and only the second one is reusable.

### GPU-side fetch planning is not expressible on this SDK

`docs/qwen38-decode-20tps-concept.md:96-102` leaves exactly two things that could move this
model, and the second is "GPU-side fetch planning, so the CPU never blocks on the router at
all… an architecture change". It is now answered from the API surface rather than from code,
on the toolchain this repository is required to use (macOS SDK **27.0**, Xcode 27 / Swift 6.4):

- `MTLIOCommandBuffer.h` exposes **three** load entry points — `loadBytes:` (46),
  `loadBuffer:` (56), `loadTexture:` (67) — and each one takes
  `sourceHandleOffset:(NSUInteger)`, a **host-supplied scalar baked in at encode time**.
- The device-selectable I/O family is absent: `MTLIOAccelTable`, `MTLIODevice`, `MTLHandle`,
  `MTLIOHandleDescriptor`, `MTLBufferIndirectHandleDescriptor` each appear in **0 of 96**
  headers. `MTL4AccelerationStructure.h` is ray tracing and has no I/O role.
- The complete `MTLIO*` symbol set in this SDK is the queue/buffer/status/compressor family:
  `MTLIOCommandQueue`, `MTLIOCommandBuffer`, `MTLIOFileHandle`, `MTLIOStatus*`,
  `MTLIOCompression*`, `MTLIOScratchBuffer*`, `MTLIOPriority*`. Nothing takes an offset from
  a buffer.

So the thing the architecture change needs — a kernel that decides *which file range* to
fetch after the router result lands on device — has no entry point here. Every read offset
must be known by the host before the command is encoded, which is the dependency the change
was meant to remove. What the SDK *does* offer and the engine already uses is device-gated
completion (`copyStatusToBuffer:` :81 then `signalEvent:` :139, wired together at
`MetalExpertReader.swift:114-117`), so the kernels can gate on bytes without a host
callback. That half is already shipped; the offset half is not buildable.

### The last predictor shape was already measured at 0.000

The recurrence idea — predict layer L's next-token experts from layer L's *previous* token's
selection — is the one prefetch-predictor shape not covered by §5aa's four feeder arms. It
turns out to be measured already, in the committed study, and its precision is zero on every
rank
([docs/qwen38-prefetch-predictor-study.md:44-45](qwen38-prefetch-predictor-study.md#L44-L45)):
`previous token's selection at that layer` = 0.000 / 0.000 / 0.000 / 0.000 P@1..4 and
0 / 0 / 0 adopted per token; `previous token's misses at that layer` the same; and mixing it
with the shipped probe changes nothing (0.552 / 0.450 / 0.379 / 0.328, identical to probe
alone). The mechanism is stated at lines 52-55: **the previous token's experts are always
already resident, which is why a temporal predictor stages nothing at all — the cache has
already absorbed that reuse.** §5aa arm 4 prices the same conclusion from the other side:
1 token of lookahead is worth +0.80…+1.19 points of hit rate and the plan's bar (+7) sits at
**32** tokens, which no part of this engine can own.

No test is needed and none was run: there is nothing left in the predictor shape to measure.

### Lossless expert compression — capacity measured, then cost measured, and the lever closes

This was the one genuinely new idea of the night: **gain slots without gaining RAM, without
touching numerics.** A lossless codec over the packed 4-bit blobs cannot change a single
value, so it is quality-exact by construction — the only capacity lever in this campaign that
is not in tension with the standing "never make the model dumber for tok/s" rule.

Screen on the real install (`python3 /tmp/expert_reducibility.py`, exit 0; layers
0/7/15/23/31/39/47 × experts 0/37/129/250/391/503 = 42 blobs = 116 293 632 B = 110.9 MiB,
i.e. 6 of 512 blobs per sampled layer):

| measured | value |
| --- | --- |
| codes used in the 4-bit space | **16/16** |
| order-0 nibble entropy | **3.7745 bits** = 94.36 % of 4 bits |
| order-1 conditional entropy | 3.7438 bits/nibble |
| single-valued code groups (32 B of codes each) | **0 / 3 225 600** = 0.000 % |
| zlib-9, whole blob | ratio **0.90555** → **+10.43 %** slots (96 → 106.0/layer) |
| zlib-9, code region only | 0.916 → +9.21 % (104.8) |
| zlib-9 per-blob spread | 0.901 … 0.907 |
| lzma-9, 8 blobs | **0.89493** → +11.74 % slots (0.8888 … 0.8956) |
| scale array / header | 0.945 / 0.707 |

Arithmetic pinned: `quantGroupSize 64` weights at 4 bits = 32 B of codes per group, so one
expert's 2 457 600 code bytes are 76 800 groups and the 42-blob sample is 3 225 600 groups.
The price comes from §5w's own measured slope — 48 → 96 slots (+100 %) bought **+12.2 %**
paired — so +10.43 % of capacity projects to **+1.27 % of clock**, under the 2.9 % drift
floor this campaign refuses to report as a win. lzma's +11.74 % projects to +1.43 %.

Then the cost, which is what actually kills it (`python3 /tmp/codec_cost.py`, exit 0, same
blobs, single thread, best of 5 rounds, Python's `zlib`/`lzma` i.e. the same C libraries a
Swift build would call):

| codec | decode throughput | a token's 203.55 MiB of reads costs | vs §5v drive demand 1 359.5 MB/s |
| --- | --- | --- | --- |
| zlib inflate | **458 MiB/s** (rounds 453 / 457 / 458 / 457 / 458) | **444 ms of CPU** | 0.35× |
| same, at perfect 8-core scaling (arithmetic, not measured) | 3 664 MiB/s | 55.6 ms = 26 % of the 217.34 ms token | 2.8× |
| lzma-9 inflate | 31 MiB/s | 6 572 ms | 0.02× |

One 2.64 MiB blob inflates in **5.207 ms**, which is 3× the 1.739 ms the drive spends
delivering a layer's *entire* miss burst, and inflating at the rate a burst arrives needs
**7.9×** the sustained pipe (3 622 MiB/s at §5w's measured 3.80 GB/s against 458 MiB/s of
inflate). So a lossless tier would move the bottleneck off the SSD and onto CPU cores that
already own the busiest host budget in the loop — and §5w's ledger says the token is 50.9 %
GPU-busy with 102 ms already accounted for by the drive and the dependency chain.

The one escape hatch was checked rather than assumed: the SDK's own compression surface is
**compress-only**. `MTLIOCompressor.h` declares three functions —
`MTLIOCreateCompressionContext` (:40), `MTLIOCompressionContextAppendData` (:46),
`MTLIOFlushAndDestroyCompressionContext` (:52), all macOS 13+ — with no decompression entry
point anywhere in the 96 headers. The hardware path that could beat a CPU inflate is not
nameable on this toolchain, so the codec lever is closed on both halves: +1.27 % projected
capacity for 2× a token's wall in CPU.

### What family 2's 42.8 ms is not, by arithmetic

§5w left the 42.835 ms/token of `shared_expert → moe_phase1_*` host turnaround open and
suggested "bookkeeping issued a layer ahead of the wait". §5x's µs measurements close that
suggestion off. Family 2 is 42.835 ms/token = **0.892 ms per layer**. §5x measured
encode p50 3.9 µs, commit p50 0.5 µs and a per-command-buffer host mean of 22.9 µs, so the
whole 6-CB-per-layer encoding budget is 6 × 48 × 22.9 µs = **6.59 ms/token** — 15 % of the
gap, and 0.892 ms per layer is **39×** the per-CB host mean.

Summing the named host counters on the same arm (line 1682) reaches the same place:
`cache_plan_ms=4.898` + `gather_ms=2.248` + `rdadvise_ms=7.624` + `cb1_ms=5.079` =
**19.85 ms/token**, i.e. 46 % of family 2 is named and the rest is queue serialization —
which §5w already explains: at 48 slots the same pair *falls* from 0.730 to 0.393 ms per
occurrence because the drive is behind and the host's bookkeeping disappears under already-
queued work. And §5x prices the proposed remedy: one extra commit-and-wait pair is ≈10 ms per
token, so a look-ahead command buffer spends ~10 to save ≤19.85, most of which is not waiting
on encoding. Task #179's premise is therefore answered as written, and the answer is no.

### Deviations, all of them

1. Both spikes live in `/tmp` and are **deliberately not committed**: the levers they tested
   are closed, so a new `benchmark/` driver would be a new file to audit for no usable gain.
   Their full printed output is embedded above so the numbers stay in the tree even when the
   scripts do not. They are throwaway probes, not engine code, and nothing under `sources/`
   changed in this section.
2. No model window was taken and no token rate appears in this section. Every clock-like
   figure quoted is a reference to an earlier section's measurement.
3. The entropy screen samples 6 of 512 blobs in each of 7 layers, and holds the same six
   expert ids fixed across those layers, so it is 42 blobs of 24 576 (0.17 %) and cannot
   separate layer variation from expert variation. The per-blob ratio spread (0.901 … 0.907)
   is that sample's variance, not a claim about the whole install.
4. The first entropy run was **broken and is withdrawn**: `entropy()` summed a `Counter`'s
   keys instead of its values and printed −29 141 861.85 bits (−728 546 546 % of 4 bits). The
   zlib/lzma ratios in that run were unaffected, the bug was fixed, and only the re-run's
   entropy lines are quoted.
5. The same run initially used a 128 B group where the format's is 32 B; the single-valued
   count and the group total changed with the fix. 76 800 groups/expert matches the manifest's
   scale count exactly, which is what confirms the corrected size.
6. Inflate throughput is single-threaded C-library throughput on this Mac. The 8-core row is
   labelled arithmetic and is **not** a measurement — no parallel inflate was run.
7. The capacity→clock projection uses §5w's 48→96 slope linearly across +10 %. The real
   slot/response curve is not linear and §5o's cliff sits elsewhere; the projection is quoted
   as an upper bound on a small step, not as a measurement of that step.
8. The SDK conclusion is about **this** SDK (macOS 27.0 from Xcode 27). A newer SDK could add
   the device-handle family; that is a toolchain question, and this repository's rule is that
   only the pinned toolchain's guarantees count as supported.
9. `nm -gU` over the framework binary returned nothing (the SDK ships a stub; the real image
   is in the dyld shared cache), so it is not part of the evidence. The header walk is.

### Verdicts

- **Four of the five overnight premises are closed by measurement, not by argument**: the
  recurrence predictor (already 0.000 on every rank), GPU-side fetch planning (no API for it
  on this SDK), lossless expert compression (+1.27 % projected capacity against 2× a token's
  wall in CPU), and look-ahead host bookkeeping (15 % of the gap it is meant to remove, and
  priced above its own saving).
- **The fifth is the instrument**: a blind `find`/`grep -r` walk over a symlinked SDK
  directory produced a confident, empty, wrong-shaped result. It is recorded here as the
  section's headline finding, not a footnote, because the conclusion it happened to support
  was already the right one — which is precisely when a broken gauge is dangerous.
- **No capacity route is left that does not need more RAM.** Slots can be bought with bytes
  (lossless: refused, sub-noise and CPU-expensive), with placement (5aa: zero-sum), with
  budget shape (5z: capped), or with hardware (not this Mac, and the operator's 12 GiB rule
  binds here). The concept document's remaining two levers are now both measured: more RAM,
  and nothing else.
- **The live work that survives this section unchanged** is #177 — the saturating use-count
  arm in `makeExpertCachePlan`, −2.04…−7.92 % placements offline, output-invariant by
  construction, still needing the paired live A/B — and #163, the wiki decode row.

## 5ac. Issue-count coalescing, priced by structure instead of by more repeats (2026-10-16)

§5m left one seam parked rather than closed: *"Issue-count coalescing prices at a
theoretical ceiling of the 10–12 ms/token it could recover while keeping the hiding —
≈2–3%, single-% territory … Parked as unproven; an A/B would need many more repeats
per arm than four."* That ceiling is the cost of the whole `F_RDADVISE` line item, and
an A/B was the only instrument proposed for finding how much of it coalescing can
actually reach. It does not need one. The merge condition is structural, so the
reachable fraction is computable from the committed route captures.

The mechanism, read rather than assumed: `sources/TinyTitan/Infrastructure/Streaming/RDAdvice.swift:33-38`
issues one `fcntl(fd, F_RDADVISE, &advice)` per call with a single `radvisory{ra_offset,
ra_count}`, so two expert ranges share a syscall only if their byte ranges touch —
i.e. only if their expert ids are consecutive inside the layer file. The live counters
confirm one call per miss read: `rdadvise_calls=53 712` over a 512-token decode is
104.9 calls/token against §5y's 89–102 placements/token at that prompt shape.

### What the bursts actually look like

`python3 /tmp/rdadvise_merge_probe.py` (throwaway, not committed, per §5ab deviation 1)
replays the shipped plan rule over the three committed captures with §5y's warm-up and
counts, for every `(token, layer)` miss burst, the maximal runs of ascending expert ids
at three hole tolerances. Gap 0 merges only strictly consecutive ids and therefore costs
no extra bytes; gap 1 and 2 buy fewer calls by reading the experts in between.

| capture | placements/token | bursts with 2+ misses | gap 0 calls/token | gap 1 net | gap 2 net |
| --- | ---: | ---: | ---: | ---: | ---: |
| `20261011` | 91.765 | 8 809/13 198 = 66.7 % | 91.21 → **+0.059 ms** | −0.291 ms | −1.036 ms |
| `v520-20261015` | 89.284 | 11 842/17 951 = 66.0 % | 88.77 → **+0.054 ms** | −0.189 ms | −0.833 ms |
| `v520-prompt2` | 102.104 | 9 840/14 071 = 69.9 % | 101.37 → **+0.077 ms** | −0.279 ms | −1.119 ms |

The nets are `ms/token` after the wasted bytes are priced against the drive: a merge
that drags in one extra expert costs 2.64 MiB = **0.66 ms of device time** at §5o's
measured 3.95 GB/s, while the call it saves is worth **0.105 ms** (`rdadvise_ms`
10.958/token ÷ 104.9 calls/token). So a byte-costing merge loses by ~6:1 before it
starts, which is why gap 1 and gap 2 are negative and only gap 0 is a candidate.

**Two thirds of bursts do hold two or more misses — and 0.6 % of them are side by
side.** That is the whole lever: 0.054–0.077 ms/token = **0.025–0.035 % of a 219 ms
token**, against the 2–3 % §5m parked it at.

### The only way to buy contiguity is to move the weights, and that is capped too

Contiguity is a property of the on-disk id order, and the id order is the repack's
choice. Relabelling is numerics-free by construction — the same bytes at a different
offset, with the routing path resolving id → offset through a table — so a repack could
place co-routed experts side by side and manufacture the adjacency the router does not
supply. `python3 /tmp/expert_layout_screen.py` (throwaway) builds a per-layer
co-occurrence graph from the routed rows, takes a greedy maximum-weight path cover
(heaviest edge first, degree ≤ 2, no cycles — deterministic and cheap enough to run
inside a repack), writes each path to consecutive offsets, and replays the shipped rule
under that permutation.

| layout learned on | evaluated on | placements/token | calls/token | saved |
| --- | --- | ---: | ---: | ---: |
| shipped order | any capture | unchanged | 91.21 / 88.77 / 101.37 | 0.025–0.035 % of clock |
| `20261011` | `20261011` (self) | **91.765** | 88.35 | 0.164 % |
| `20261011` | `v520-20261015` | **89.284** | 88.17 | 0.053 % |
| `20261011` | `v520-prompt2` | **102.104** | 100.90 | 0.058 % |
| `v520-20261015` | `v520-20261015` (self) | **89.284** | 86.21 | 0.147 % |
| `v520-prompt2` | `v520-prompt2` (self) | **102.104** | 98.26 | 0.184 % |
| best cross-session arm | — | — | — | **0.063 %** |

The self-check is the `placements/token` column: it is identical, to the last digit,
under every permutation. That is the proof that a relabel cannot change what the cache
does — placement depends on the arrival sequence, not on the labels — so the only thing
moving is the syscall count, exactly as intended.

A learned layout does create adjacency, and it does transfer, but only at ⅓ of its own
value: 0.147–0.184 % against the session it was trained on, 0.044–0.063 % against
another. The reason is the same one §5h found from the other side — the 512 experts are
mutually orthogonal, so their co-routing is diffuse: a layer file has 511 adjacencies
available against 130 816 possible pairs, and no ordering can make many of them hot.

### Deviations

1. Both probes are `/tmp` throwaways and are deliberately not committed: they read
   committed captures and change no shipped code, so there is nothing to register.
   Their full text is reproducible from the tables above.
2. **My first run of the merge probe was wrong and is recorded as such.** It skipped
   the prefill rows instead of planning them, so decode started on a cold cache and
   reported 96.89 / 90.77 / 105.45 placements/token — 5.6 / 1.5 / 3.3 above §5y's
   calibrated figures. The self-check against `placements_per_token` caught it before
   any merge number was read; after the warm-up fix all three match to 1e-6. The
   discard's ratios were not wildly off (0.62 % vs 0.61 % adjacency), which is exactly
   why the absolute check, not the plausibility of the ratio, is what earns the result.
3. 0.105 ms per call is the engine's own aggregate: it attributes all of
   `rdadvise_ms` to the `fcntl`, including whatever the read thread pays around it. If
   the true syscall is cheaper, the ceiling is lower, not higher.
4. A 219 ms token is the shipped-default wall from §5r, used only as a denominator;
   nothing here is a clock claim.
5. The device side argues against the lever as well. §5d measured `dd` over a whole
   layer file at 3.15 / 2.61 GB/s — *slower* than the engine's strided 3.57 GB/s — so
   merging short scattered ranges into longer sequential ones moves toward the access
   shape this drive handles worse. A pure syscall win of ≤0.18 % would have to survive
   that to reach the wall clock at all.

### Verdict

**§5m's parked seam closes at two orders of magnitude below its estimate.** Merging
what is already contiguous reaches 0.025–0.035 % of clock; merging what a learned
relabel makes contiguous reaches 0.147–0.184 % on its own session and 0.063 % on
another's, and §5d's sequential contrast says the device may take some of that back.
None of it is within reach of the 2–3 % that made the A/B look worth many repeats, so
no repeats are owed: the lever is closed by structure, and the parked row can be
retired without a model run.

**What survives unchanged** is #177 — the saturating use-count arm, whose live paired
A/B is the running subject of §5ad — and #163, the wiki decode row.
## 5ad. #177 live: the capped use-count is a measured loss — and the replay that recommended it was a first-request proxy (2026-10-16)

§5aa's one surviving candidate was the saturating use-count cap in
`makeExpertCachePlan`: clamp the *ranking key* `min(useCount, C)` while the engine's
per-expert count keeps accumulating, so a defended slot is priced by recent demand
instead of by everything it has done since load. It was −2.04…−7.92 % placements
offline and output-invariant by construction, and §5ab closed the night with it as the
one arm still owed a live paired A/B. The arm has now run. **It loses, and the reason
it loses is a hole in the offline screen itself.**

Provenance: `models/qwen3.8-flash-next_125B_A6B_4Bit`, `.build/release/TinyTitanServer`
sha256 `fdaa34830d33a1e52d436376e30f821947fd0455e342aba4c26c1a944b0f948d`, built at
`2ef9d00`; `git diff --name-only 2ef9d00..HEAD` names `ci.yml` and two files under
`docs/` and **nothing under `sources/`**, so that binary is the engine at `1307a75`.
Working tree clean at measurement. 24 GB Apple M3, macOS 27.0.1, Swift 6.4,
target `arm64-apple-macosx27.0.0`, AC power at 100 % (recorded at launch),
`memory_pressure -Q` 85 % free. Command:
`python3 benchmark/tinytitan_usecap_ab.py` → exit 0. 3 rounds × 3 arms × 2 requests of
512 tokens at `--ram 12G`, one fresh server per arm per round (9 servers, never two
alive), arms interleaved within each round, `default` carried first as the control.
The cap is `TINYTITAN_EXPERT_USE_CAP`
(`sources/TinyTitan/Infrastructure/Streaming/PreadExpertStreamer+Residency.swift:110-135`),
unset for the control arm — the driver *clears* it rather than leaving it, so an
operator's exported value cannot silently rename an arm.

### The wall clock, both pairings

| round | default r1 | default r2 | cap3 r1 | cap3 r2 | cap5 r1 | cap5 r2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.557 | 4.763 | 4.614 | 4.486 | 4.509 | 4.506 |
| 2 | 4.641 | 4.744 | 4.623 | 4.508 | 4.631 | 4.630 |
| 3 | 4.664 | 4.879 | 4.544 | 4.533 | 4.558 | 4.572 |

The driver pairs request *2* with request *2*, because that is the request a cache
policy can actually distinguish — and that pairing is uniformly negative:

| arm | r1 | r2 | r3 | mean |
| --- | ---: | ---: | ---: | ---: |
| cap3 (warm pairing) | −0.277 (−5.8 %) | −0.236 (−5.0 %) | −0.346 (−7.1 %) | **−0.286 (−6.0 %)** |
| cap5 (warm pairing) | −0.257 (−5.4 %) | −0.114 (−2.4 %) | −0.307 (−6.3 %) | **−0.226 (−4.7 %)** |
| cap3 (cold pairing, request 1) | +0.057 | −0.018 | −0.120 | −0.027 |
| cap5 (cold pairing, request 1) | −0.048 | −0.010 | −0.106 | −0.055 |

Six of six warm pairs negative; the cold pairs are inside ±0.12 tok/s and average
−0.6 % and −1.2 %. So the arm is not "slightly slower everywhere" — it is **neutral on
a cold cache and 5–7 % slower on a warm one**. That shape is the finding.

### The counters carry the mechanism, not the clock

| arm | hit rate range | mean hit | mean misses | mean read MiB |
| --- | --- | ---: | ---: | ---: |
| default | 0.7676 … 0.7872 | **0.7775** | 55 418 | 146 336.8 |
| cap3 | 0.7626 … 0.7643 | 0.7633 | 58 951 (+6.4 %) | 155 667.9 (+6.4 %) |
| cap5 | 0.7650 … 0.7667 | 0.7659 | 58 315 (+5.2 %) | 153 987.6 (+5.2 %) |

The default arm's second request gains **+1.8 points** of hit rate over its first
(0.7689 → 0.7864 at round 2): that is the cache remembering the previous request. The
cap arms gain nothing (cap3 0.7643 → 0.7626, i.e. −0.2 points). A clamp at 3 or 5 makes
every well-used expert indistinguishable from one used exactly that many times, so the
rule that defended the resident set across requests degenerates into recency — and
recency is exactly what §5y's `_Slots` textbook rule is, which lands at 151.20
misses/token where the shipped plan lands at 89.28. **The cap does not weaken the LFU
rank at the margin; it removes the memory that made the rank worth having.**
Response invariance held on every round: `b5c6c72d385f2f20` across default, cap3, cap5.

### Why the offline screen said the opposite, and what it now has to do

§5aa's replay ran each arm on **fresh caches over one session** — the analogue of
request 1, which is precisely where this arm is neutral-to-better. The second request
is not reproducible in that shape. Replaying the same route rows **twice over one cache
set** models it (greedy decoding makes the second request's routes identical, which the
hash equality above confirms), and the prediction flips:

| capture | pass | shipped | cap3 | cap5 |
| --- | --- | ---: | ---: | ---: |
| `20261011` | 1 | 91.765 | 84.496 (**−7.9 %**) | 84.734 (−7.7 %) |
| | 2 | 81.486 | 86.977 (**+6.7 %**) | 84.044 (+3.1 %) |
| `v520-20261015` | 1 | 89.284 | 87.458 (−2.0 %) | 86.951 (−2.6 %) |
| | 2 | 77.258 | 87.487 (**+13.2 %**) | 84.687 (+9.6 %) |
| `v520-prompt2` | 1 | 102.104 | 96.966 (−5.0 %) | 98.157 (−3.9 %) |
| | 2 | 90.752 | 95.397 (**+5.1 %**) | 93.493 (+3.0 %) |

Pass 1 reproduces the shipped baselines to the digit (91.765 / 89.284 / 102.104), so
this harness is §5y's, and only the second pass is new. The shipped rule's second pass
improves 11.1–13.5 % over its first — that is the warmth the live run measured as
+1.8 points of hit rate — and the cap arms' second passes do not improve at all
(+2.94 %, +0.03 %, −1.62 % for cap3). The offline second-pass penalty of
**+5.1…+13.2 %** placements brackets the live **+5.2…+6.4 %** miss penalty, which is
the agreement that matters: the counters predict each other across an offline replay and
a real server, and the wall clock is only the third witness.

### Deviations

1. **Swap was already 6 523.62 M of 8 192 M committed before the run started**, on
   mains, with the machine otherwise idle. Not removable without purging caches, which
   this repository forbids, so it is disclosed rather than fixed. It raises both arms
   equally and the counters move with the wall clock, not against it.
2. **Two single-threaded python screens totalling ~50 s of CPU ran inside the
   measurement window** — §5ac's merge probe and layout screen, while rounds 2–3 were
   in flight. Arms are interleaved *within* a round, so common-mode drift cancels in
   the paired delta, but round 3's losses are the largest of the three and I cannot
   exclude an asymmetric effect on it. That is why the verdict rests on the six-of-six
   sign, the counters and §5ad's own offline reproduction, not on the magnitude of
   −6.0 %. Re-running for a cleaner magnitude is not needed to decide the arm.
3. The driver's headline pairs the warm request. Quoted above alongside the cold
   pairing, because the difference between the two *is* the mechanism.
4. A 512-token request at ~4.6 tok/s is ~111 s per arm-request; 18 requests plus 9
   server starts is ~40 min of model time. Nothing else ran against the GPU.
5. `prefetch_issued_per_token` and `io_hidden_pct` are flat across arms (42.3–43.0 and
   99.7–99.9 %), so the cap does not disturb the prefetch ring — the loss is purely
   placement.

### Verdict

**The cap is refused.** `TINYTITAN_EXPERT_USE_CAP` stays unset by default — the shipped
rank remains the uncapped per-expert count that outlives its slot — and the flag and its
35 model-free tests stay in the tree as a measured-refused control, the same way
`--rdadvise off` stays after §5m.

**#177 closes as measured, and it leaves a rule behind**: for any candidate that changes
what the cache *remembers*, the admissible offline screen is the two-pass replay, not
the single-pass one. §5aa's numbers for this arm were not wrong — they described request
1, which the live run confirms is neutral. The campaign's other offline-screened policy
arms share the blind spot and should be re-read with that limit stated, which is what
this section exists to state.

**The rule is now code, not prose.** `two_pass_placements` in `benchmark/heat_split_sim.py`
replays the same rows twice over one shared cache set and returns both counts; its three
tests live in the already-CI-registered `benchmark/test_heat_split_sim.py` as
`TwoPassReplayTests`. The first pins pass 1 against `placements_per_token` over the same
events, so the two numbers cannot drift into different quantities and every §5y/5z/5ad
single-pass table stays comparable to a two-pass one; the second hand-traces the
remembering itself (a 3-expert working set over 3 slots costs 1.0 placements/token cold
and 0.0 warm); the third asserts the flip on the committed captures — `use_cap=3` beats
the shipped rank on pass 1 and loses on pass 2, on all three. One constraint inherited
from `_PlanSlots` and worth knowing before writing an arm: a row may not name more experts
than its layer has slots, which the engine cannot produce anyway (top-k 10 against 96) but
a hand-built fixture easily can — the failure is an `IndexError` inside the plan, not a
refusal.

## 5ae. The two-pass screen run over the campaign's backlog: the feeder family splits in two (2026-10-16)

§5ad left a rule and a backlog. This is the backlog. Three arms that were priced on fresh caches are
re-priced on both passes, at **cap none — the engine rule as shipped** — and every figure is placements
of token converted to hit-rate points on the 480-request denominator (1 placement = 0.2083 points, bar
= +7 points). Model-free: the three committed captures, python, no GPU, no clock, same licence as
§5y/§5z/§5aa.

Each arm was gated before it was read, by a self-check that **pass 1 reproduces the published
single-pass function to 1e-9**:

| arm | pass-1 gate | arms checked |
|---|---|---|
| §5z per-layer budget split | `heat_split_sim.placements_per_token` + §5z's own printed table | 9/9 |
| §5aa arm 3, heat-biased eviction | `predictive_feeder_sim.heated_placements` | 18/18 |
| §5aa arm 4, oracle look-ahead | `predictive_feeder_sim.lookahead_placements` | 21/21 |

That gate is not decoration — it is the reason this section's numbers are trustworthy. The first arm-3
screen counted with `placements += len(cache.plan(experts)) if decode else 0`, and Python evaluates only
the chosen branch of a conditional expression, so `plan()` never ran on a prefill row: the replay had
silently dropped §5y's warm-up and scored a cold cache. Its shipped baseline came out at 90.775
placements/token against the capture's published 89.284 — and 90.775 is *exactly* the discarded cold
figure §5ac disclosed and withdrew, which is what pointed the finger at my copy rather than at the
published value. Calling `plan` unconditionally and counting conditionally fixes it; all 18 arms then
reproduce to 1e-9. No number below comes from the broken copy.

### 1. The §5z heat split is warmth-neutral — and it never rested on the refused cap

| scored on | arm | pass 1 pts | pass 2 pts |
|---|---|---:|---:|
| `20261011` | uniform (shipped), 91.765 → 81.486 | — | — |
| | oracle: own profile | **+0.71** | **+0.87** |
| | transfer from `v520-20261015` | **+0.66** | **+0.75** |
| | transfer from `v520-prompt2` | **+0.73** | **+0.84** |
| `v520-20261015` | uniform (shipped), 89.284 → 77.258 | — | — |
| | oracle: own profile | **+0.78** | **+0.74** |
| | transfer from `20261011` | **+0.82** | **+0.80** |
| | transfer from `v520-prompt2` | **+0.91** | **+0.82** |
| `v520-prompt2` | uniform (shipped), 102.104 → 90.752 | — | — |
| | oracle: own profile | **+0.92** | **+0.99** |
| | transfer from `20261011` | **+0.78** | **+0.85** |
| | transfer from `v520-20261015` | **+0.74** | **+0.81** |

Pass 1 reproduces §5z's transfer table to the printed digit on all nine rows, which answers the second
question §5ad raised: **no, §5z never depended on `use_cap=3`.** Its own words already said "cap none,
engine rule as shipped" and the figures agree. The cap is only what `CrossCaptureTests` scores — a test
problem, not a documentation problem, filed as #185.

The first question: nine of nine split arms are positive on **both** passes. Six arms move up on the warm
pass (by +0.07…+0.16) and three move down (by −0.02…−0.09), so no arm moves by more than 0.16 points. That
is mechanism, not luck. `split_slots` changes **how many slots a layer holds** — a property of the cache's
size — and leaves the eviction rule's memory untouched, so it prices the same cold and warm. §5z's verdict
therefore stands unchanged by warmth: the six transfers land at +0.66…+0.91 on pass 1 and +0.75…+0.85 on
pass 2, against oracle ceilings of +0.71…+0.92 and +0.74…+0.99, which is every static partition of 4 608
slots sitting seven to eleven times under the +7-point bar.

### 2. Arm 4 keeps four fifths of its gain on a warm cache, and the bar it clears is one capture wide

| capture | W | pass 1 pts | pass 2 pts | retained | placements the 2nd request saves |
|---|---|---:|---:|---:|---:|
| `20261011` | 0 = shipped | — | — | — | 10.28 |
| | 1 | +0.80 | +0.66 | 82 % | 9.60 |
| | 8 | +3.62 | +2.90 | 80 % | 6.84 |
| | 16 | +5.57 | +4.47 | 80 % | 5.00 |
| | 32 | **+7.82** | **+6.40** | 82 % | 3.45 |
| `v520-20261015` | 0 = shipped | — | — | — | 12.03 |
| | 1 | +0.95 | +0.77 | 82 % | 11.21 |
| | 8 | +3.41 | +2.71 | 79 % | 8.64 |
| | 16 | +4.92 | +4.00 | 81 % | 7.56 |
| | 32 | **+7.05** | **+5.75** | 82 % | 5.78 |
| `v520-prompt2` | 0 = shipped | — | — | — | 11.35 |
| | 1 | +1.19 | +0.94 | 79 % | 10.18 |
| | 8 | +4.58 | +3.69 | 81 % | 7.08 |
| | 16 | +6.54 | +5.24 | 80 % | 5.11 |
| | 32 | **+8.86** | **+7.19** | 81 % | 3.30 |

Three results, and the middle one is the interesting one.

**The gain survives warmth.** Across 18 window/capture points the arm retains 79–82 % of its cold value
— a near-constant four fifths, independent of window size. So §5aa's temporal headroom was not an artifact
of cold caches, unlike everything else in the feeder family.

**The crossing is.** At W=32 the +7-point bar is cleared on pass 1 by 3 of 3 captures (+7.82/+7.05/+8.86)
and on pass 2 by **1 of 3** (+6.40/+5.75/+7.19). The bar is a threshold, so this does not soften the
conclusion into "still clears, roughly" — it moves two of three captures to the wrong side of it. And the
form the engine could actually own (§5aa: MTP is off, and its verify pass is width 2, so W=1 at best) is
**+0.66…+0.94 points warm** against +0.80…+1.19 cold.

**Protection and warmth are substitutes.** The shipped rule's second request saves 10.28/12.03/11.35
placements/token — offline warmth of 2.14/2.51/2.37 points against the +1.8 points §5ad measured live on the
same default arm. An engine that already protected this session's next 32 tokens saves only 3.45/5.78/3.30
on the same second request: the look-ahead had already consumed **66 %, 52 % and 71 %** of what the repeated
request would have delivered.
Both mechanisms buy the same thing — the right expert resident before it is named — so their gains are not
additive, and any future plan that prices "prediction" and "multi-turn warmth" together must subtract one
from the other.

### 3. Arm 3 is a cold-start discovery effect: every positive transfer flips sign

| learned on → scored on | W | pass 1 pts | pass 2 pts |
|---|---|---:|---:|
| **hindsight** `20261011` → itself | 96 | +2.91 | +1.12 |
| | 192 | +2.00 | −0.08 |
| **hindsight** `v520-20261015` → itself | 96 | +3.00 | +0.84 |
| | 192 | +1.32 | −0.92 |
| **hindsight** `v520-prompt2` → itself | 96 | +3.44 | +1.24 |
| | 192 | +2.41 | +0.07 |
| `v520-20261015` → `20261011` | 96 | **+1.30** | **−1.16** |
| | 192 | **+1.18** | **−0.96** |
| `20261011` → `v520-20261015` | 96 | −0.35 | −2.42 |
| | 192 | **+0.57** | **−1.67** |
| `20261011` → `v520-prompt2` | 96 | −5.24 | −8.49 |
| | 192 | −4.96 | −7.53 |
| `v520-20261015` → `v520-prompt2` | 96 | −4.77 | −7.73 |
| | 192 | −5.09 | −7.76 |
| `v520-prompt2` → `20261011` | 96 | −2.20 | −6.53 |
| | 192 | −3.16 | −5.95 |
| `v520-prompt2` → `v520-20261015` | 96 | −3.82 | −8.94 |
| | 192 | −4.51 | −7.75 |

All three transfers that were worth anything on a cold cache (**+1.30, +1.18, +0.57**) are worth *less than
nothing* on a warm one (−1.16, −0.96, −1.67), and the nine that were already negative get worse — up to
−8.94 points. The hindsight rows, the arm's ceiling with the session's own future already known and at zero
slot cost, keep 38 %/28 %/36 % at W=96 and **nothing at all at W=192** (+0.07…−0.92).

So §5aa's mechanism claim ("eviction inside a fixed tier is zero-sum") is not merely confirmed, it is
narrowed: what little arm 3 did convert was the cache **discovering its working set**, and a second pass
discovers it for free. A list learned from another session is redundant with the memory the shipped rule
already has — which is the same verdict §5ad reached for the cap from the live side, arriving here from the
offline side, and the reason both are now one finding rather than two.

### The rule the campaign gets out of this

The two passes separate policy arms by *what they change*, and that is worth stating before the next
candidate is screened:

- Arms that change **how fast the cache reaches its steady state** — cross-session victim protection
  (arm 3), a saturating count that flattens the rank (§5ad) — are priced almost entirely by pass 1 and
  collapse or flip on pass 2. Both arms of this class that the campaign measured are in that shape.
- Arms that change **what the steady state is** — per-layer budget size (§5z, ±0.16 points), own-future
  protection (arm 4, 79–82 % retained) — price similarly on both passes.

A single-pass offline screen therefore over-reports exactly the first class, and the first class is the one
that looks cheapest to build. Nothing of that class shipped, which is the only good news in the section.

### Deviations

1. **My first arm-3 replay dropped the warm-up** (the conditional-expression shape described above) and its
   figures are withdrawn, not adjusted. The corrected screen's 18/18 self-check is the evidence for §3.
2. **The arm-4 throwaway still contains the bug in its own arm-3 half**, so `feeder_two_pass_screen.py`
   exits 1 and the 12 arm-3 rows it prints are wrong — the corrected `/tmp/arm3_two_pass.py` rows are the
   ones used here. Its arm-4 loop calls `plan` first and counts second, which is why 21/21 of the arm-4
   checks pass. Disclosed because both scripts are in `/tmp` and may be re-run by someone who is not me.
3. **Arm 2 (pinned prefix) was not re-screened.** Its published best honest arm is +0.14 points, fifty times
   under the bar and reachable only from the same conversation; a sign flip cannot move that decision. This
   is the one backlog row closed by reasoning instead of measurement, and it is named as such.
4. **Pass 2 models a repeated identical request**, which is the shape §5ad confirmed live (greedy routes,
   hash `b5c6c72d385f2f20`). A multi-turn conversation whose next turn asks something different routes
   differently — §5z measured that agreement as 3.2–13.2 % of picks — so it lands *between* the two passes.
   Read both columns as a bracket on a session's value, not the warm one as the answer. For arm 3 that
   bracket is the whole story: its cold gain is real only to the extent turns repeat routes.
5. **Two §5aa intermediate points differ by 0.01** (+4.93 vs +4.92 at `v520-20261015` W=16; +2.93 vs +2.92
   at `v520-prompt2` W=4). Placements reproduce to 1e-9, so the difference is the doc's hand-converted
   rounded percentages: `predictive_feeder_sim.main()` prints points only at W=32. Not a measurement
   disagreement.
6. **Only `two_pass_placements` is shipped** (`85f810a`, and `test_heat_split_sim.py` is at 20 tests — §5z
   recorded 17, the three new ones are `TwoPassReplayTests`). §1 runs through it; §2 and §3 need their own
   two-pass loops, because a protection set is not a budget. If the campaign keeps pricing warm policies
   that loop shape belongs in `predictive_feeder_sim.py` — a decision, not a task for this section.
7. **No clock anywhere.** These are placement counts on three captures whose replay level error (§5z: −3.2 %
   to −10.8 % against the engine's own bracket) is larger than every effect above. The ban on quoting a
   token rate from them is arithmetic, not caution.
8. Provenance: tree `85f810a` clean, python 3.14.8, three committed route captures, no model loaded, no GPU
   work, mains power but nothing timed.

### Verdict

**The backlog §5ad named is re-priced, and the feeder family splits in two.** The budget-shape lever is
warmth-neutral and still seven to eleven times under the bar; arm 4's own-future protection survives warmth
at four fifths of its value but clears the +7-point bar on one capture of three rather than three of three;
arm 3's cross-session protection does not survive it at all. No shipped default changes: the 96-slot tier,
the uncapped rank and the flat per-layer budget are all steady-state levers, and this section is the check
that nothing in the cold-start class was quietly priced into one of them.

**What it does change** is the campaign's read on the temporal headroom. §5aa said the headroom is real but
owned by nobody; §5ae says the headroom is real *on a warm cache too*, which is the stronger claim — a
predictor that only paid off on cold starts would not be worth building, and one that keeps 80 % of its
points would. The blocker is unchanged and is not a measurement: the window that clears the bar is 32 tokens
and the window the engine can own is 1. **The reusable rule is the one to take forward**: price every future
policy arm on both passes and treat a pass-1-only gain as a cold-start effect until proven otherwise —
which is now the shape of two refusals (§5ad's cap, §5aa's arm 3) and would have saved the campaign a live
A/B if it had been stated first.

## 5af. Seven probes of the constant-feeder premise: the drive's wall is 3.3–3.4 GB/s, and the live session pays 17–53 % of decode for filling its idle clock (2026-10-16)

§5aa closed the feeder as a *cache* change (eviction inside a fixed tier is zero-sum) and §5v closed the
duty-cycle reading (34.7 % busy, and the idle clock is not spare bandwidth). Neither measured the thing the
operator's own words ask about: *"the sdd can read constantly while a user has an active session and can keep
on reading weights even if compute is doing stuff."* Whether that is worth doing depends on what the device
delivers when it is asked to read constantly, in the shapes a feeder would produce. Six model-free probes on
on the 48 `packed_experts/layer_NN.bin` files answer the device half of it. The seventh probe is not model-free: it runs
the same feeder beside a live decode session, because the offline arms can say what the device does under a constant
feed but not what a token does.

### The instrument

`/tmp/slice_sweep2.c` mirrors the shipped reader's terms and nothing else: `pread` on descriptors opened
`F_NOCACHE`, 2 MiB-aligned scratch, expert-stride-aligned offsets, a full-length read loop, per-read latency,
1–32 workers, and an arbitrary number of cooperating processes released by a shared gate file so their
windows overlap. Its aggregate for multi-process arms is `Σ bytes / (max end − min start)` — the union window,
not an average of averages. Worker seeds carry `getpid()`, because two processes chasing the same LBA
sequence would measure locality instead of the device.

### #186 — granularity × concurrency: the knee is in-flight bytes, and the shipped reader sits on it

Median of 5 interleaved rounds; control noise floor 3.258–3.385 GB/s (3.9 %). Rate at a given amount of
in-flight bytes:

| in-flight MiB | 0.17 | 0.66 | 2.64 | 5.28 | 10.56 | 21.12 | 42.25 | 84.50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| slice 2.641 MiB | — | — | 2.459 | 3.239 | **3.360** | 3.380 | 3.344 | 3.329 |
| slice 1.320 MiB | — | — | 2.132 | 3.073 | 3.318 | 3.352 | 3.359 | 3.274 |
| slice 0.660 MiB | — | — | 1.716 | 2.632 | 3.202 | 3.309 | 3.316 | 3.309 |
| slice 0.330 MiB | — | 0.999 | 1.676 | 2.347 | 2.856 | 3.052 | 3.039 | — |
| slice 0.165 MiB | 0.722 | 1.276 | 1.968 | 2.541 | 2.803 | 2.747 | — | — |

Three readings, all of them against the feeder idea:

1. **Throughput is set by in-flight bytes, not by how you cut them.** At 10.56 MiB in flight the three
   coarse slices deliver 3.202–3.360; at 21.12 they deliver 3.309–3.380. Granularity is free *down to
   0.66 MiB* and then costs: at the same 10.56 MiB, 0.330 MiB reads give 2.856 and 0.165 MiB give 2.541.
2. **The shipped shape is the knee.** Four workers × 2.64 MiB = 10.56 MiB is where the curve flattens; 8, 16
   and 32 workers move it by +0.020, −0.016 and −0.031 GB/s against that control. A reader with more
   concurrency in it would not read any faster.
3. **Extra workers buy latency, not bandwidth.** p50 per full read at slice 2.641 MiB: 1100 µs at w=1, 3260
   at w=4, 6534 at w=8, 13 182 at w=16, 26 793 at w=32 — linear in queue depth while the rate is flat.

The finer slices are not hypothetical: they are what B3's split blob reads issue. At w=4 a 1.32 MiB half-read
costs −0.019 (noise), and the campaign's own sub-0.66 MiB shapes cost −0.245 and −0.545 at their best.

### #187 — burst length: the wall is flat from 64 MiB to 6.4 GiB, so "constantly" is not slower, and not faster either

Control 203 MiB at 2.641 MiB × 4: 3.218–3.391 across 9 round-pairs (5.2 %).

| window MiB | 64 | 128 | 203 | 406 | 812 | 1625 | 3250 | 6500 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.641 MiB × 4 | 3.265 | 3.317 | 3.356 | 3.374 | 3.373 | 3.374 | 3.388 | 3.376 |
| paired Δ vs control | −0.074 | −0.017 | n/a | +0.008 | +0.013 | +0.008 | +0.022 | +0.004 |
| 0.165 MiB × 16 | 3.044 | 3.213 | 3.242 | 3.260 | 3.291 | 3.292 | 3.314 | 3.300 |

A continuous feed runs 32 s at 3.376 GB/s with no decay — last-bin / first-bin is 0.974–0.993 for every arm
of 406 MiB or more. So the premise that the drive slows while it works is false, and the corollary matters
more: a feeder that keeps it busy forever earns the *same* 3.4 GB/s the burst already gets. There is no
sustained-read bonus to be had. (The summarizer prints the 0.165 MiB shape as `0MiBx16`; it is `%.0f` of
0.165, not a zero-length read.)

### #188 — a co-running feeder is strictly zero-sum, and it lands on the critical path

The question the operator's idea actually needs answered is not "can the drive do two things" but "does the
second thing cost the first anything". 200.7 MiB demand burst at the shipped shape, against a feeder reading
2 GiB of its own, 9 interleaved rounds, burst row is the measurement:

| arm | burst GB/s | burst wall ms | paired Δ vs same-round solo | feeder GB/s |
| --- | ---: | ---: | ---: | ---: |
| solo | 3.342 / 3.305 | 63 / 64 | — | — |
| thin (1 feeder worker) | 2.705 / 2.673 | 78 / 79 | −0.619 / −0.628 | 2.33 / 2.32 |
| full (4 feeder workers) | 1.710 / 1.694 | 123 / 124 | −1.642 / −1.633 | 3.08 / 3.07 |

All nine replication rounds are negative in both feeder arms. The two-averages figure this first suggested —
"aggregate 5.0 GB/s" — was an artifact: the feeder's 2 GiB window is mostly uncontended, so summing two
medians over two different windows invents bandwidth. #189's union-window instrument is what replaced it, and
the naive sums are labelled as naive wherever they appear.

### #189 — processes do not break the wall either; 3.31–3.38 at every count

Union-window aggregate, 2 GiB per process, 5 interleaved rounds, control noise floor 5.1 %, launch skew
sub-millisecond:

| p1w4 | p2w2 | p2w4 | p3w4 | p4w2 | p4w4 | p6w4 | p8w1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3.385 | 3.378 | 3.383 | 3.375 | 3.370 | 3.334 | 3.331 | 3.308 |

Per-process share of the total falls as 1/P. Eight processes with one reader each get the same 3.31 that one
process with four readers gets. Whatever the 3.3–3.4 GB/s is, it is not a per-process limit, so no amount of
external feeding processes buys a faster device.

### #190 — splitting one demand burst across processes buys nothing

203 MiB delivered, gate-synchronised, 7 interleaved rounds: **63.2 ms** at 1×4, **63.0 ms** at 2×2 (paired
+0.023 — nothing), 64.5 ms at 4×1, 67.0 ms at 2×4, 71.5 ms at 6×2. A burst served by two cooperating
readers is not served faster than one.

### #191 — file scope: locality does not help, and sequential is the slowest arm

§5w reports the engine delivering 3.85/3.68/3.84/3.95/3.69 GB/s inside its own bursts (mean 3.80) at the
identical shape #186 measured at 3.36. The last structural difference between the two instruments is *which
files the concurrent reads touch*, so the address set was made the only variable — slice, worker count and
total bytes fixed at 2.641 MiB × 4 × 203 MiB — across two interleaved batches (9 rounds, then a 16-round
replication), with `spread` carried as the control in every round and the sharing arm's file rotated per
round so no arm is favoured by one platter range:

| scope | what it reads | GB/s (9 / 16 rounds) | paired Δ vs control |
| --- | --- | --- | --- |
| spread | random file, random expert | 3.310 / 3.310 | — |
| one | one file per worker | 3.320 / 3.317 | −0.005 / +0.029 |
| same | all four workers in one layer file | 3.218 / 3.234 | −0.100 / −0.049 |
| seq | consecutive expert strides in one file | 3.216 / 3.236 | −0.093 / −0.053 |

Two arms bracket the engine's own shapes: a token's ten routed experts live in one file (`same`), and a
token's 480 requests spread over all 48 (`spread`). Neither is faster, and maximal locality is ~3 % *slower*.
`seq` at −0.922 in one of its rounds is the same first-round warm-up shape seen in both batches — round 1's
control read 2.66–2.69 GB/s while every other arm of that round read high, which inflates all three of its
deltas; excluding round 1 the medians are `same` −0.06, `one` +0.03, `seq` −0.07. Short reads: 0 everywhere.

### The ceiling this section retires, and the one it replaces it with

Nothing the campaign can make this device do offline exceeds **3.39 GB/s** — across 30 slice × worker shapes
(#186), 8 burst lengths to 6.4 GiB (#187), 8 process counts (#189), 5 burst splits (#190) and 4 address sets
(#191). That contradicts three citations still standing in this doc and in `docs/v4-core-design.md`:

- the **3.92 GB/s** four-thread `ParallelExpertReader` ceiling (§5o),
- the engine's own **3.95 GB/s**,
- §5w's **3.80 GB/s "delivered inside each burst, 97 % of the ceiling"**.

§5w's number is a quotient, not a rate: `gap_total / gap_count` divided by `experts_per_miss_layer × stride`
credits the *whole token's demand volume* to the hit→fixup gap, while the ring's ~42.8 speculative reads per
token (§5v) are also flying in that window. Either the gap was not entirely demand-read time or not all of
those bytes flew inside it — both make the quotient larger than the device rate, and the device is what the
five scopes above just measured directly. **The planning number from here is 3.3–3.4 GB/s.** That does not
weaken §5w's conclusion, it strengthens it: if the bytes cost *more* device time than the section assumed,
the bill for a constant feed is bigger, not smaller, and moving them earlier is even more the only lever.

### Arithmetic on the idle clock the operator pointed at

How much idle clock there is depends on which of the campaign's two busy figures you believe: §5w's kernel
ledger leaves `217.34 − 95.7 = 121.6` ms of every token with the device not serving reads, while §5v's duty
sample (34.7 %) implies 141.9 ms. Take the smaller. Filling all of it at the wall measured here moves
`121.6 ms × 3.37 GB/s = 391 MiB` per token — roughly double the 203.55 MiB the token's demand misses actually
cost. So the idle clock is real and large enough; the reason the feeder still fails is not that there is
nothing to read, it is #188 (bytes read while a burst runs are taken from that burst) plus §5aa's landing
limit (42.8 issued per token, 18.56 adopted, `free=0.99`) — the tier has nowhere to put the extra bytes.

### #192 — the live session: the idle clock really is fillable, and filling it costs 17 % and 53 % of decode

The six offline probes share one limitation: their victim is a synthetic burst that never stops. A session is
different — §5v measured the device idle roughly two thirds of the time — so the same feeder was run beside ten
live greedy 384-token decodes at the shipped default (`--ram 12`, 96 slots, `--max-context 262144`,
`TINYTITAN_RUNNER_STATS=1`), in three interleaved rounds of control / 1-worker feeder / 4-worker feeder plus a
closing control, with `iostat -d -w 1 disk0` sampled across every arm:

| arm | tok/s median | paired Δ vs same-round control | hit rate | engine MiB/tok | device MB/s mean | % of the 3.37 GB/s wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 5.082 | — | 0.8457 | 197.3 | 1467.9 | 44 % |
| feeder ×1 | 4.216 | −0.870 (−17.1 %) | 0.8329 | 213.8 | 2724.9 | 81 % |
| feeder ×4 | 2.370 | −2.709 (−53.3 %) | 0.8112 | 241.5 | 3072.5 | 91 % |

Per-round paired deltas are −0.661, −1.008, −0.870 for ×1 and −2.717, −2.709, −2.650 for ×4: nine arm-pairs, all
negative, and no round in which a feeder helped. All ten arms emitted the same response hash
(`95aa4531b4ea2304…`), so this is scheduling and not numerics. The control's own spread is 4.6 %
(4.896–5.132), with round 1's control the low outlier — the same first-round warm-up #191 showed.

The premise is **confirmed** in the only sense the device can confirm it. Device utilisation rises from 44 % of
the wall to 81 % and 91 %, and the ×4 arm's busiest second is 3228 MB/s, 96 % of everything the offline probes
ever got out of this drive. The engine is not short of idle clock; it has never been using it. It is **refuted**
in the sense that matters: those bytes are not free, and they are worse than the fixed-pie split #188 measured.

**The demand volume is identical; only the classification moves.** `expert_hits + expert_misses` is 186 032 in
every arm of every round (484.5 per token), and round 1's `expert_evictions` (31 293) and `expert_reloads`
(22 986) are the same number in all three arms — the plan asked for the same experts in the same order and
chased the same slots. What changed is how many of them had already arrived: `prefetch_adopted_per_token` falls
19.01 → 12.52 → 4.66 while `prefetch_issued_per_token` barely moves, 39.45 → 37.98 → 35.93.

Three independent counters then agree to ±1 read, in all six feeder arms:

| round × arm | Δ demand misses | −Δ prefetch adoptions | Δ `expert_read_mib` ÷ stride |
| --- | ---: | ---: | ---: |
| R1 ×1 | +2 491 | 2 492 | 2 491.0 |
| R1 ×4 | +5 510 | 5 511 | 5 510.0 |
| R2 ×1 | +2 212 | 2 211 | 2 212.0 |
| R2 ×4 | +6 360 | 6 359 | 6 359.9 |
| R3 ×1 | +2 542 | 2 542 | 2 542.0 |
| R3 ×4 | +6 504 | 6 505 | 6 504.0 |

That is the mechanism, and it is in the code. A plan adopts a speculative read only if its buffer is ready when
the router's indices come back: `if let source = prefetched[...]` memcpys it into the slot, and anything without
a ready buffer falls through to `misses.append(index)`
([PreadExpertStreamer+Plan.swift:95-106](../sources/TinyTitan/Infrastructure/Streaming/PreadExpertStreamer+Plan.swift#L95)),
while every demand miss charges `experts.count × expertStride` to the byte counter
([PreadExpertStreamer+Residency.swift:381](../sources/TinyTitan/Infrastructure/Streaming/PreadExpertStreamer+Residency.swift#L381)).
A late speculative read is therefore **not used at all**: the same 2.641 MiB crosses the bus twice, once in the
window the feeder stole and once on the path the token waits for. That is why the engine reads *more* under a
co-tenant (197.3 → 213.8 → 241.5 MiB/token) — a fixed-bytes model like #188's cannot predict it, and it is the
reason a feeder is worse than zero-sum rather than merely zero-sum.

The time follows the bytes. Per token, `io_ms` goes 69.1 → 98.2 → 189.7 ms against token time 196.8 → 237.2 →
389.0 ms, while `wait_ms` (the host blocked on the GPU) only goes 92.3 → 98.2 → 124.0. The regression is the read
span, not a new stall — which is what a contention story should look like, and not what a scheduling-priority
story would.

What survives is the version of the operator's idea that the engine owns. Its ring issues 39.45 reads per token at
the shipped default and adopts 47 % of them, so about 21 reads — 55 MiB, 18 % of the 309.7 MiB/token this session
moves at the device — is read and then thrown away for arriving late. The argument that kills an external feeder
says the internal one is paying for its own lateness. Whether throttling it helps is #193, and
`TINYTITAN_PREDICTIVE_PREFETCH=0` has never been priced in this configuration:
`docs/qwen38-remaining-levers-plan.md:59` lists it as a check under hypothesis 1 and never ran it, while the
ring-depth study that retired two-and-three-ahead ran at `--ram-budget 8G` in September with no depth-0 arm.

### Deviations

1. **The probe's p99 was read out of bounds**, `all[n - n/100]` with `n < 100` evaluating to index `n`. It
   printed 0 for every short arm: #187's 64/128/203 MiB rows and #188's 76-read burst rows. Fixed to
   `all[n * 99 / 100]` and #188 re-run from the corrected binary (`/tmp/contention_rounds2.tsv`), so the
   #188 figures quoted here are from the fix. Rates, walls and p50 come from timestamps and a sorted array
   and were never affected. Note the converse degeneracy: with the fix, a 76-read row's p99 *is* its max, so
   #191's tail column is reported as max, not as a percentile.
2. **Binary provenance is not uniform.** #186 and #187 ran on the pre-edit binary (no pid in the seed, no
   gate, no bound on the gate wait). Both were single-process, so pid-in-seed and the gate are inert there;
   their rate/wall/p50 columns are valid. #188 was run once on the old binary and re-run on the corrected
   one; #189–#191 ran entirely on the corrected one.
3. **A `""` gate argument hung the smoke test.** An empty fifth argument was treated as a real path, so
   workers spun waiting for a file that never appears. Fixed (empty means no gate) and the spin is now
   bounded to 15 s with a loud exit, after it had cost two minutes. Probe defect, not a device finding.
4. **First-round warm-up in both #191 batches**, described above, is why per-round deltas are printed rather
   than only medians. #192's round-1 control is the same effect (4.896 against a 5.082 median), which is why
   its deltas are paired within a round and not against the batch median.
5. **Machine state before the live batch**: `sysctl vm.swapusage` read 6.19 GiB of 7.17 GiB already in use,
   `memory_pressure -Q` 82–83 % free, load 3.25, AC power, no model process. The swap is not something these
   probes created and it is sampled per arm in #192.
6. **Provenance.** The offline arms (#186–#191) ran at tree `b5fba5e`. 24 GB M3 MacBook Pro, macOS 27.0.1,
   Xcode 27 / Swift 6.4 toolchain present but not exercised — the probes are C, built `cc -O2 -Wall -Wextra`,
   and `swift build` was not re-run. No model loaded, no GPU work, no numerics touched. #192 is the exception
   on every one of those points: it loaded the installed 4-bit Qwen3.8-Flash-Next through the release binary
   built at 04:29 from the last source commit `9c858cd`, with `b5fba5e` (a benchmark-only change) as `HEAD`
   and this document the only dirty file. Its numbers are therefore attributable to a committed engine, but
   the tree they were measured in was not clean, and the standing rule is not to publish from one — so the
   #192 rows are a measurement of `9c858cd`, recorded here as such, not a shipped-artifact benchmark.
   Everything in §5af is a throwaway under `/tmp`; nothing shipped.
7. **The #192 driver averaged `iostat` over the wrong denominator.** It divided by every row past `NR>2`,
   including the trailing blank and zero rows, so each arm's device mean was understated by roughly 9–10 %.
   Every device figure in the #192 table was recomputed from the raw `.iostat` files with `NR>3 && $3+0>0`.
   The wall clocks, hit rates and byte counters came from the driver and are unaffected.
8. **`iostat`'s first data row is the since-boot average.** The driver's per-arm MiB total included it, adding
   ~47 MiB to ~121 206. The recomputed totals the table uses exclude it.
9. **The device window is the whole arm, not the decode loop.** `iostat` runs from before model load to after
   the last token, so the device MiB/token column is session-total bytes ÷ generated tokens and carries the
   load phase. The paired arm-to-arm deltas are still the right comparison; the absolute 197–241 MiB/token is
   an upper bound. The `engine MiB/tok` column is the engine's own counter and has no such inflation.
10. **The feeder's own achieved rate was not captured** — its stdout went to `/dev/null` — so its bytes are
    derived by difference between the recomputed device total and the engine counter, not read from the probe.
    Its process liveness and wall were captured.
11. **The driver's `load1` field is malformed** (`tr -d '{} '` collapses `vm.loadavg`'s spaces into
    `3.693.293.25`). Load was read off the raw sysctl for the record; no conclusion in this section rests on it.
12. **Swap moved during the live batch**: 7.35–7.85 GiB used across the ten arms against 6.19 GiB before it.
    §5v's warning is that swap movement, not the device, is what distorts a live arm. Here the regression is
    monotone in feeder intensity (−17.1 %, then −53.3 %) and the closing control returned to 5.132, which is
    what contention predicts and not what a swap cliff looks like — but the swap was there, and it is the
    strongest reason to re-run this arm on a machine that has not been fed before adopting anything from it.

### Verdict

**The drive wall is 3.3–3.4 GB/s and the shipped reader is already at it.** In-flight bytes set throughput
and 10.56 MiB is the knee (#186); sustained feeding is neither slower nor faster than a burst (#187); a
co-tenant's bytes come straight out of the demand burst, by 0.62–1.64 GB/s (#188); and no process count
(#189), burst split (#190) or address set (#191) moves the wall by more than the control's own noise. The
constant-feeder concept therefore dies in the offline limit for the same reason it died in the cache: the
machine can already be kept busy, and keeping it busy buys bytes the engine has nowhere to put. #192 then
tested the one version the offline probes could not see — the same feeder scheduled into a *live session's*
gaps instead of into a synthetic burst — and the session pays for it: −17.1 % decode with one worker, −53.3 %
with four, nine negative pairs out of nine, identical output hash. The idle clock really is fillable (44 % →
81 % → 91 % of the wall) and filling it is **worse than the fixed pie #188 measured**, because a speculative
read that arrives late is not used at all: the same expert crosses the bus again on the path the token waits
for. Refused as shipped behaviour. What survives of the operator's idea is the internal half of it — the
engine's own ring discards ~21 reads per token, and §5ag prices that and keeps the ring.

## 5ag. #193 prices the engine's own feeder by switching it off: it earns +5.6 % while spending 54 MiB per token on reads it throws away (2026-10-16)

§5af refused the *external* constant feeder and left the internal one unpriced. The shipped decode path already
does what the operator described — it keeps the drive reading while the GPU computes — through a one-slot staging
ring ([ExpertPrefetchRing.swift:154](../sources/TinyTitan/Runtime/Inference/ExpertPrefetchRing.swift#L154),
built with `slotCount: ringDepth` at
[RealForwardRunner+BuildCore.swift:99](../sources/TinyTitan/Runtime/Inference/RealForwardRunner+BuildCore.swift#L99)).
`TINYTITAN_PREDICTIVE_PREFETCH=0` turns it off and has never been run at this configuration —
`docs/qwen38-remaining-levers-plan.md:59` lists it as a check under hypothesis 1 and never ran it, and the
September depth study that retired two- and three-ahead ran at `--ram-budget 8G` with no depth-0 arm.

Three paired rounds of ring-on (shipped) against ring-off, plus a closing ring-on, greedy 384 tokens, `--ram 12`,
96 slots, `--max-context 262144`, mains power, one process at a time, `iostat -d -w 1 disk0` across every arm:

| arm | tok/s | paired Δ vs same-round ring-on | hit | `io_ms` | `wait_ms` | engine MiB/tok | device MiB/tok | issued/pt | adopted/pt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ring on | 5.079 | — | 0.8459 | 68.3 | 92.6 | 197.1 | 309.1 | 39.45 | 18.86 |
| ring off | 4.793 | **−0.286 (−5.6 %)** | 0.8070 | 85.3 | 88.1 | 246.9 | 254.7 | 0.00 | 0.00 |

Per-round deltas −0.204, −0.492, −0.286: three pairs, all negative. Control spread 1.9 % (5.027–5.124) and the
closing ring-on returned to 5.043, inside it. All seven arms emitted the same hash (`95aa4531b4ea2304…`).

**The ring is worth 5.6 % of decode, and it is worth it while moving 18 % fewer bytes than it saves.** That
sentence is the finding, and the counters close on it exactly:

- Turning the ring off removes **20.59 issued reads per token** (39.45 − 18.86 adopted) and the device total
  falls by **54.4 MiB per token** — `20.59 × 2.641 MiB = 54.4`, to the read.
- Turning it off adds **+7 242 demand misses** over the session, which is **18.86 per token** — the adopted
  count, exactly. `18.86 × 2.641 = 49.8 MiB` per token returns to the demand path, and the engine's own
  `expert_read_mib` rises 197.1 → 246.9.
- So the ring's trade is: spend 54.4 MiB/token of device traffic to move 49.8 MiB/token off the critical path.
  `io_ms` 85.3 → 68.3 is the payment landing, and it is worth 5.6 % of the clock.

**Refused: do not throttle or remove the speculative ring.** The half of it that is discarded is the price of
the half that arrives ahead of the router, and §5af just measured what happens to a token when something else
competes for those bytes instead.

**The two shapes this arm might have suggested are already closed, and saying so is not redundant — #193 is
their first re-confirmation at the shipped tier.** `docs/qwen38-prefetch-predictor-study.md` scored the staged
*choice* against ground truth from the prefetch trace, at the September `--ram 8` meaning (64 cache slots), and
found the shipped probe the best ordering available: P@1 0.552, P@2 0.450, P@3 0.379, while frequency
re-ranking, the previous token's route and the previous token's misses are neutral or worse — the temporal
predictors stage *nothing*, because the cache has already absorbed that reuse. Its monotone decay by rank also
settles what this section would otherwise have had to open: the list the ring truncates with `prefix(count)`
([ExpertPrefetchRing.swift:88-98](../sources/TinyTitan/Runtime/Inference/ExpertPrefetchRing.swift#L88))
is already in router-weight order, so the staged expert is the highest-weight non-resident prediction rather
than an arbitrary one, and the caller's residency walk decides where it stops. The same study priced the
*volume* question and lost every arm: 2 slots −3.3 %, 3 slots −10.7 %, mixed horizon −1.4 %, L+2 horizon −3.7 %,
all byte-identical.

What #193 adds is the two things that study did not have: a depth-0 arm at all, and the current 96-slot tier at
5.08 tok/s against its 4.56. The conclusion survives the move — the ring earns +5.6 % here — and one number
moves. Adoption per issued read is 18.86 / 39.45 = **0.478** at 96 slots against 23.3 / 42.8 = **0.544** at 64,
and issued reads fall 42.8 → 39.45. Both halves of that move are what a bigger cache predicts: fewer
predictions are worth staging, and more of the ones that are get loaded by the demand path before the staged
read is wanted. That is an inference from two counters in two batches, not a measurement, and it is the kind
that must not be carried into a decision — the study's own note that its `--ram 8` rows are not this tier is
exactly why. #194 scores it from a trace at the shipped configuration.

The ring's own footer also says which failure mode a deferral would have to avoid: over 18 001 `begin` calls the
single slot is observed free 0.99 of the time and in flight 0.01, with a mean speculative load of 1.21 ms
against a 3.75 ms layer window (`body_ms 179.9 / 48`). A staged read nearly always *finishes* in time; what it
fails to be is the right expert. So lateness is not the loss channel at depth 1, and any deferral is spending
the one thing the ring actually has — its headroom.

The shape neither batch tested is **ordering at fixed volume**: keep the 39.45 speculative reads a token, and
issue each one after that layer's demand burst has taken service, so the same bytes land in the idle window
instead of at the head of the queue. It can add no device traffic and remove none, which makes it the only
remaining version of the operator's "read constantly" premise that §5af's contention result does not already
kill — and the only version the engine itself owns. Its risk is named above: it trades headroom for queue
position, and adoption depends on the headroom. That is #194.

### Deviations

1. **Seven arms, not eight.** The plan was three rounds of the pair plus a closing control; the closing control is
   ring-on only, so it is a drift check and carries no pair. Reported as measured.
2. **Provenance.** `HEAD 8e82098` (§5af, docs-only); the binary is the release build from `9c858cd` at 04:29, the
   same one §5af's #192 ran, and the tree was clean at launch. 24 GB M3 MacBook Pro, macOS 27.0.1, Swift 6.4.
   Nothing was changed in engine code to run this arm — the toggle is the existing env switch — so both arms are
   the shipped engine, and the only unshipped thing here is the record.
3. **The device window includes model load**, as in #192, so `device MiB/tok` is session-total ÷ 384 and the
   309.1 / 254.7 absolutes carry the load phase. The *difference* between the two arms is what the arithmetic
   above uses, and the ring-off arm's 7.8 MiB/token residual is the load and non-expert traffic, which is
   consistent with #192's control.
4. **`engine MiB/tok` is demand-path bytes only.** `recordSuccessfulLoadsUnlocked` charges
   `experts.count × expertStride` on the demand load path and the adoption path only bumps the load count
   ([PreadExpertStreamer+Residency.swift:379-399](../sources/TinyTitan/Infrastructure/Streaming/PreadExpertStreamer+Residency.swift#L379)),
   so speculative reads never appear in `expert_read_mib`. That is why the engine/device gap is the ring's
   traffic and not a discrepancy — the gap in the ring-on arm (112.0 MiB/token) is `39.45 × 2.641 = 104.2`
   plus the 7.8 MiB residual, which closes to 0.4 %.
5. **Swap** 7.66–7.92 GiB (ring on) and 7.69–7.82 GiB (ring off): both arms ran under the same pressure, and the
   closing control sat inside the round range, so no arm is explained by a swap move. Load and pressure at launch
   were as recorded for §5af; no model process was running before or after the batch.
6. **`adopted_pt` medians are 18.86 where §5af quoted 19.01** for the same shipped configuration. Different batch,
   same arm, 0.8 % apart — inside the control spread of both batches, and the identity claims above use each
   batch's own numbers rather than a cross-batch mix. Never compare a qwen38 clock across batches.
7. **`ExpertPrefetchRing` has no unit test.** A repo-wide search finds the type only in `sources/` and `docs/`.
   What *is* covered is the config seam — `tests/TinyTitan/Runtime/Configuration/ModelProfileTests.swift:116` and
   `ExpertCacheBudgetTests.swift:211-240` assert `prefetchDepth` 1 and 0, which is the arm #193 ran — but nothing
   pins the ring's `.completed` gate or its `prefix(count)` truncation. Any change #194 makes has to start by
   testing the adoption contract, not by changing it.

## 5ah. #194 reads the ring's shortfall off a trace: 93 % of the discarded reads were the wrong expert, 7 % were right but late, and the late pool is too small to measure (2026-10-16)

§5ag left one number unexplained and one idea alive. The number: the ring issues 39.45-39.59 reads a token and
adopts 47-48 % of them, so 20.5 reads a token are paid for and thrown away. The idea: if a meaningful share of
those 20.5 were *correct* reads that merely missed their layer, then re-ordering them — issuing after the layer's
demand burst has taken service, or awaiting them briefly at plan time — would convert them at zero extra bytes.
§5ag called that the only untested shape. This section tests it, and it does not survive the test.

Two greedy 384-token decodes of the same prompt at the shipped default (`--ram 12`, 96 slots,
`--max-context 262144`, `TINYTITAN_RUNNER_STATS=1`), each writing `TINYTITAN_PREFETCH_TRACE`: one with the ring
on, one with `TINYTITAN_PREDICTIVE_PREFETCH=0`. 18 384 records each (383 positions × 48 layers), and the two
responses are byte-identical (`md5 c172890df88cdbbfa0627bd320d16614`).

### The offline replica, and why its output can be trusted

`ExpertPrefetchRing.begin` is re-implemented offline exactly: at layer L the staged expert is the **first
element of `next_layer_prediction` that is not in the target layer's `resident` list**, taken from the record at
(`position`, L+1) — the same set the caller passes in
([RealForwardRunner+DecodeMoE.swift:322-349](../sources/TinyTitan/Runtime/Inference/RealForwardRunner+DecodeMoE.swift#L322)),
`model.routedExpertResidentIDs(layer: L + 1)`. Each staged read is then classified against that layer's own
record: *correct* if it is in `experts` (the exact route), *adopted* if it is in `experts` and **not** in
`misses` (the trace's `misses` are post-adoption, so a read the plan took from the ring is not a miss), *late* if
it is in `experts` and still in `misses`.

| check | replica | engine's own counter |
| --- | ---: | ---: |
| staged reads per token | 39.73 | `prefetch_issued_per_token` 39.59 |
| adopted reads per token | **19.25** | `prefetch_adopted_per_token` **19.25** |
| demand misses removed per token | **19.25** (33 709 − 26 316 between the two arms) | `expert_misses` 35 901 − 28 508 = **7 393** = 19.25/token |
| speculative MiB/token implied | 104.9 | §5ag's engine/device gap, 112.0 − 7.8 residual = 104.2 |

Three counters agree to the read. The replica over-counts *issues* by 54 reads (0.35 %) because `active`
deduplication and the rare busy slot are not modelled; every one of those 54 is a wrong read, so the adopted and
miss identities are exact while the issued figure is a 0.35 % upper bound.

### The decomposition: 92.7 % precision, 7.3 % timing

| per token | reads | MiB | share of the 20.48-read shortfall |
| --- | ---: | ---: | ---: |
| issued | 39.73 | 104.9 | — |
| correct **and** adopted | 19.25 | 50.8 | — |
| correct but not adopted (**late**) | **1.50** | 3.95 | **7.3 %** |
| not in the route at all (**wrong**) | **18.98** | 50.1 | **92.7 %** |

The identity closes on the counts: 15 257 issued = 7 393 adopted + 575 late + 7 289 wrong, and 7 968 correct =
7 393 adopted + 575 late. Nothing in the ring is unaccounted for.

**The lateness channel is spread over the layer stack, not concentrated at a boundary**: 45 of the 47 target
layers have at least one late read (only layers 1 and 31 have none), the largest count is 33 at layer 38 (9.6 %
of that layer's 345 issues), and layers 45-47 together carry 19 of 575. So there is no structural discard to
fix — not the token-boundary
`beginToken()` reclaim, not the last layer's slot. The ring footer agrees: over 18 001 begins the single slot is
free 1.00 of the time and submitted, in flight and held are all 0.00, with `spec_load_ms` 1.22 against a 3.91 ms
layer window (`body_ms 187.974 / 48`). Reads land inside the window; they are simply not the expert the router
asks for.

### What that prices, and why #195 is not run

The pool a perfect await could convert is 1.50 reads/token = 3.95 MiB/token of duplicate crossings. Three
routes to a price, all from this campaign's own measurements:

1. **Linear in the byte savings #193 just measured.** §5ag: removing the ring's 49.8 MiB/token of *demand*
   savings costs 5.6 %. 3.95 / 49.8 × 5.6 % = **+0.44 %**.
2. **The exchange rate.** 2.641 MiB at the §5af wall (3.3-3.4 GB/s) is 0.78-0.80 ms, so 1.50 reads = 1.20 ms of
   a 196.8 ms token = **+0.61 %**.
3. **A two-anchor per-MiB price model.** #193 gives `49.8d − 104.2c = 5.6` and #192's 1-worker feeder gives
   `322.9c + 17.1(d − c) = 17.1`, where `c` is the cost of a speculative MiB and `d` the value of an avoided
   demand MiB. Solving: `c = 0.042 %/MiB`, `d = 0.201 %/MiB`, so awaiting a late read is worth 1.50 × 2.641 ×
   0.201 = **+0.80 %** before its own cost. That model is fitted to those two arms, so it reproduces them by
   construction; its useful output is the break-even below, not the 0.80 %.

Route 3's upside is the ceiling, not the expectation: the await puts the read's remaining load time on the
critical path, and 0.3-0.5 % of token time is a plausible cost for it, so the honest range is **+0.3 % to
+0.8 %**. This campaign's control spread in the same configuration is 1.9 % (§5ag) and 4.6 % (§5af). **No arm
run here can resolve the effect it predicts.** #195 — ordering at fixed volume, and its bounded-await cousin —
is screened out before being built, and the ring's adoption contract therefore needs no test for a change that
will not be made. That is the answer §5ag's last deviation line was holding open.

### Two things the trace says that no prior record does

**Rank order is the right staging rule, measured.** Precision by the rank of the first non-resident prediction
falls monotonically: rank 1 0.754 (2 151 issues), 2 0.687 (1 999), 3 0.611 (1 916), 4 0.560 (1 835), 5 0.501
(1 647), 6 0.415 (1 478), 7 0.403 (1 320), 8 0.325 (1 143). The ring already stages the best-ranked
non-resident candidate, so within one read a layer there is no better *choice* available in its own list.
Overall precision here is 0.522 — the September study's 0.552 at the 64-slot tier, same shape, slightly worse.

**Per-layer precision varies 4×, and it is not actionable.** Layer 6's staged reads are 0.773 correct; layer
42's are 0.310 and **layer 47's 0.181** (66 correct of 364 issues). Under the route-3 prices a read is worth
issuing at all when its precision exceeds `c/d` = 0.21, so only layer 47 falls below the line — and dropping its
staging is worth **+0.02 %** (it saves 2.50 MiB/token of speculative bytes and gives up 0.44 MiB/token of demand
savings). The reason layer 47 routes so much less predictably is *not* established here; the probe reads layer
46's residual and misses 82 % of the time at the layer above it, and no candidate explanation was tested. It is
recorded as a finding and not as a lever, because the screen above says it cannot be cashed at depth 1.

### The size of what is actually left

Scored against the ring-off arm's true demand need, **34.65 of the 48 layer-slots a token have at least one
demand miss** (histogram: 5 080 layers need nothing, 4 573 need one, 3 370 need two, 2 272 need three, and 20
need all ten). At depth 1 the ring can adopt at most one read per layer, so that is the ceiling on adoption:
**19.25 achieved against 34.65 available**. A perfect next-token predictor would issue *fewer* reads than the
shipped ring does (34.65 against 39.73) and adopt every one of them, worth roughly +4.6 % by route 1's
arithmetic — and it would remove 13 MiB/token of traffic rather than adding it.

That is the campaign's answer to the feeder premise, and it is the same answer the September study gave from
the other direction. `docs/qwen38-prefetch-predictor-study.md` measured live per-layer frequency, the previous
token's route and its misses, an EMA of probe scores, their unions, a second rank, a second horizon and a second
layer of lookahead; none beat the shipped probe, and every wider arm lost 1.4-10.7 %. What remains after §5af,
§5ag and §5ah is therefore not a scheduling question at all: **the bytes the ring throws away are the price of
the 47 % it keeps, the price is set by prediction quality, and prediction quality is a model-side problem this
engine cannot manufacture by re-ordering reads.** The decode line for qwen38 4-bit on this machine is closed at
the shipped tier by three independent measurements — remove the ring and lose 5.6 %, feed the device from
outside and lose 17-53 %, aim the ring better and find that 93 % of what is left is already known to be wrong.

### Deviations

1. **These two arms are traced, so their clocks are not the shipped ones.** 4.865 (ring on) against 4.496 (ring
   off) is −7.6 %, where §5ag's untraced pair measured −5.6 %; `TINYTITAN_PREFETCH_TRACE` writes 18 384 JSON
   lines in both arms and is not part of the shipped path. The pair is used for *structure* (which is what #194
   asks) and never for magnitude. The classification counters agree with §5ag to a tenth of a point: hit rate
   0.8468 / 0.8070 here against 0.8459 / 0.8070 there.
2. **383 positions, not 384.** The trace covers positions 10-392; the last decode step is not recorded, so the
   replica denominators are 383 × 48 records while the per-token figures divide 384. That is a ≤0.26 % skew on
   any count that ends in the untraced step.
3. **`late` means "in the route and not adopted", which is the union of three causes.** A still-in-flight buffer
   at `readyBuffers`, a failed operation and a slot reclaimed before its plan all look the same from the trace.
   The footer's `inflight = 0.00` is sampled at the *next* `begin`, which runs after that layer's plan, so it
   cannot exclude in-flight-at-`readyBuffers`; what it does exclude is a read still running at the end of its
   own layer. So 1.50/token is an **upper bound** on genuine lateness, which makes the screen above conservative
   in the direction that matters.
4. **The residency filter is read one instant late.** `resident` in the trace is captured before that layer's
   plan; the ring queried the same set at the previous layer. For depth 1 the two are identical — nothing writes
   layer L+1's cache between them, because adoption happens inside layer L+1's plan, which the snapshot precedes.
   At depth 2 or 3 they would differ, so this replica is only valid for the shipped geometry.
5. **The ceiling is scored against the ring-off arm on purpose.** The ring's own adoptions shrink the demand need
   visible in its own trace (68.53 misses/token against 87.78), so measuring headroom on the ring-on records
   would score it against a need it has already eaten.
6. **The scorer is a throwaway.** `/tmp/score194.py` is not in the tree; the algorithm is specified above
   completely enough to rebuild it, and the trace facility it reads is shipped. Nothing in `sources/` changed to
   produce this section, so no test was added and the golden baseline is **not checked** (docs-only record, as in
   §5af and §5ag).
7. **Single round per arm.** #194 is an offline scoring exercise over two traces, not a paired timing batch, so
   the interleave-and-control rule does not apply to it; the only clock it touches is deviation 1's, and that is
   reported as contaminated. The tree was clean at `HEAD cd229f6`, the binary is the 04:29 release build, AC
   power at 100 %, `memory_pressure -Q` 86 % free, swap 6.65 GiB used at launch and 6.63 GiB after, no other
   model process before or after either arm.

## 5ai. #195 separates MTP's byte term from its time term at the shipped tier: a width-2 verify pass moves 0.79x the device traffic the shipped path moves per token, so the recorded loss is a time story — and the number a rewrite would have to beat is 1.582x (2026-10-16)

§5ah ends by naming prediction quality as the only lever left, and the one trained next-token predictor in this
tree is the MTP draft head. §8 carries MTP as closed on three records — 2026-09-18 (a pair pass at **2.238x** a
scalar token, 86.9 % acceptance), 2026-09-21 at `--ram 8` / 64 slots on a machine Chrome was using (−32.3 % and
−46.9 % paired, and the project's own B3 gate at **−43.2 %** with 85.7 % acceptance and byte-identical output),
and the archived 8-bit case (1.926 emitted per pass, −2 %). Two things about those records are checkable without
a model run: whether the *byte* half of the price was ever separated from the *time* half, and whether either
half moved when the tier went 64 → 96 slots and B1 lifted the hit rate 0.7577 → 0.8459.

**The trace answers the byte half.** A width-W verify plan loads the union of its W rows' selections as one
cache plan (`RealForwardRunner+MTPVerify.swift:64-68`), so its expert traffic is `|union(experts) \ resident|`.
Consecutive trace positions are consecutive decode tokens and `experts` is the target router's exact selection,
so that union is readable from #194's ring-off trace without running MTP. The replica is validated the way §5ah
validated its own: `misses == experts \ resident` holds on **18 384 of 18 384 records**, and the demand it
reconstructs is 88.013 reads/token against the footer's 87.78.

**Route locality is the reason the union is cheap, and it is measured here for the first time.** Of the second
row's 10 selections at a layer, over **18 336** sliding consecutive-pair layer-slots (both rows present, residency
read at the first row as a verify plan would read it; the three columns partition the 10 by construction and sum
to 10.000):

| where the next token's expert is | per layer | note |
| --- | ---: | --- |
| also selected by the previous token | **3.820** | independence over 512 experts would give 0.195 |
| not in it, but already resident | **4.439** | |
| needs a fresh read | **1.741** | the disjoint-pair tiling the width table uses puts the same marginal at 1.750 |
| control: 10 tokens apart (17 904 slots) | 1.878 / 5.652 / 2.471 | locality decays but does not vanish |

That closes a question the September study left as an assertion. It found the previous token's route scores
P@1 0.000 as a prefetch ranking and wrote "the previous token's experts are always already resident, which is
why a temporal predictor stages nothing at all". Now both halves have a number: the reuse is real (38 % of a
token's selections), and the cache is what absorbs it, so there is nothing left for a temporal predictor to
stage. The same locality is why MTP's second row costs 1.74 reads rather than 10.

**Width table** — `E` is the emitted tokens per pass at the recorded acceptance 0.869 (`E = Σ pᵏ`, capped at
1/(1−p) = 7.63), and the two ratio columns are the shipped path's own counters from §5ag: 309.1 MiB/token of
device traffic, of which 197.1 MiB is demand-path:

| W | passes | reads/layer/pass | MiB/pass | E | MiB per emitted token | vs 309.1 (device) | vs 197.1 (demand) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 383 | 1.834 | 232.4 | 1.000 | 232.41 | 0.752 | 1.179 |
| 2 | 191 | 3.584 | 454.3 | 1.869 | 243.07 | **0.786** | 1.233 |
| 3 | 127 | 5.254 | 666.0 | 2.624 | 253.79 | 0.821 | 1.288 |
| 4 | 95 | 6.879 | 872.0 | 3.280 | 265.81 | 0.860 | 1.349 |
| 5 | 76 | 8.447 | 1070.7 | 3.851 | 278.06 | 0.900 | 1.411 |
| 6 | 63 | 10.044 | 1273.0 | 4.346 | 292.91 | 0.948 | 1.486 |
| 7 | 54 | 11.621 | 1472.9 | 4.777 | 308.35 | 0.998 | 1.564 |
| 8 | 47 | 13.069 | 1656.5 | 5.151 | 321.58 | 1.040 | 1.632 |
| 13 | 29 | 20.358 | 2580.4 | 6.403 | 402.98 | 1.304 | 2.045 |
| 42 | 9 | 52.414 | 6643.5 | 7.613 | 872.70 | 2.823 | 4.428 |

**What that says about the closure.** The union does keep growing per emitted token — 232.4 → 872.7 MiB from
W=1 to W=42 — so `RealForwardRunner+MTPVerify.swift:31-34`'s stated reason is directionally right, and this
table is the first quantification of it on qwen38 (their 5.18x at W=13 and 11.25x at W=42 are Qwen3.6-35B-A3B,
256 experts, top-K 8, a different cache). But against the baseline a shipped token actually pays, the growth is
not what fails MTP: **a pair pass costs 0.786x the shipped path's device traffic per emitted token**, and the
byte race stays favourable to W=7 (308.35 against 309.1 MiB) before W=8 turns it. The reason is that the two
mechanisms are substitutes for the same locality: the ring spends 54.4 MiB/token on reads it throws away
(§5ag), and the draft's second row needs only 1.74 fresh reads because the cache already holds 8.26 of its
10 selections.

So the −17 % of 2026-09-18 and the −43 % of 2026-09-21 cannot be a byte story, and the record already knew
that — "the union is not where it goes", with `verify_routed_pair` at 1.45x the scalar routed GPU time. The cost
lives in the two **time** terms the same header names: two rows through the 32-token prefill kernels at
1.6-1.7x where the decode path is 1.0x, and ~200 ms/pass of host and commit (0.93x a token), where the
sequential fetch awaits nothing at width 2. Neither term depends on the cache tier, which is why 64 → 96 slots
and +8.9 points of hit rate could not have flipped them. **MTP stays closed.**

**The number to beat, if anyone revisits it.** Starting from 5.079 tok/s (§5ag, shipped), break-even needs a
pass costing ≤ E = 1.869x a token and 6.0 tok/s needs **≤ 1.582x** (1.869 ÷ 6.0/5.079). The best width-2 pass
ever measured anywhere in this tree is 1.585x — on Qwen3.6, and as a union model, not a qwen38 decode-native
kernel. So the design target is a 29 % cut in the pass's time from 2.238x, and since the byte term is already
favourable, the whole cut has to come out of the prefill-path inefficiency and the per-pass host work. That is
a much better-specified question than the "large rewrite against a gate that fails by 43 points" the 2026-09-21
verdict recorded, and it is the one thing this section leaves open.

**Not run: the live arm, refused by its own guard.** The arm that would price the two time terms at the shipped
tier — `benchmark/tinytitan_mtp_phases.py --target models/qwen3.8-flash-next_125B_A6B_4Bit --sidecar
models/qwen3.8-flash-next_125B_A6B_MTP_4Bit --ram-budget 12G --verify-arm pair --presence-penalty 0 --pairs 1
--warmups 1` — reached `gate0.preflight(15)` and stopped: "GPU is already 51 % busy before TinyTitan launches".
`idle_gpu_utilization()` (`ioreg -c AGXAccelerator`, "Device Utilization %") read 51 % then 45 % on a second
sample, no model process was live, and **`--allow-busy-gpu` was not passed**, so nothing was measured and no
number in this section comes from a contended GPU. The sidecar is present and its manifest guard passed:
`models/qwen3.8-flash-next_125B_A6B_MTP_4Bit/{manifest.json, model_weights.bin 51 681 440 B, packed_experts/}`,
which the driver mounts at `--mtp-memory-mib 384`. It is still the right next run — it also settles the one
correctness finding the tree has never explained, the prose-scenario divergence at character 553 that §8's own
verdict says must be explained "before quoting any speed number".

Deviations.

1. **The byte denominators are three, and the table shows two of the comparisons.** 232.41 MiB/token is this
   trace's own demand (ring off, 96 slots), and is the denominator the union is measured against in the same
   records; 309.1 is §5ag's device traffic including 104.2 MiB of speculation and a 7.8 MiB load/non-expert
   residual; 197.1 is the shipped demand path. So the only strictly internal figure is 243.07 ÷ 232.41 =
   **1.046x** — a pair costs 4.6 % more demand bytes than a scalar token — while the 0.786x headline is the
   comparison against the shipped alternative the reader is choosing between, and it inherits §5ag's deviation
   that the device counter includes model load, which makes it *harder* for MTP, not easier.
2. **The first pass at this scored the wrong ratio.** Dividing the union by W (rows) instead of E (emitted
   tokens) produced 0.977x at W=2 and, at W=42, 0.681x — a flattering full-acceptance bound that would have
   read as "widening is byte-free". The correction is why the table divides by E and reports W=42 at 2.823x.
3. **The tables were re-derived from one loop each before this commit, and four cells changed.** The first
   draft mixed outputs: the locality table's resident-only and fresh columns (4.352, 1.828) came from a
   disjoint-pair loop while its shared column and slot count came from the sliding one, and the W=13 / W=42
   `MiB/pass` cells (2493.6, 6399.9) do not reproduce from the width loop, which gives 2580.4 and 6643.5. The
   republished numbers come from `/tmp/score195c.py` (widths, disjoint tiling, row-0 residency) and
   `/tmp/locality195.py` (locality, sliding, same snapshot). Nothing the argument rests on moved: the
   MiB-per-emitted-token column and both ratio columns are what the single method already gave, and the
   sliding and disjoint marginals agree to 0.009 reads (1.741 against 1.750). **W=13 is 29 disjoint passes and
   W=42 is 9**, so those two rows are sample-thin; W ≤ 8 has 47 passes or more.
4. **E assumes independent acceptance.** Real qwen38 acceptance is measured per pass (85.7-92.6 %) and streaks
   are autocorrelated; the pass also pays all W rows' bytes up front, so a draft rejected at rank 2 of 6 has
   already staged rows 3-6. That waste is invisible to this accounting, and it grows with W — one more reason
   the short drafts are the only byte-favourable ones.
5. **Residency comes from a ring-off demand trace.** A live MTP session loads two tokens' experts per pass, so
   its cache holds more of the next union and its misses are fewer: the reads in this table are an **upper
   bound** on the byte bill.
6. **The draft's own traffic is not here.** The 384 MiB sidecar has its own `packed_experts` and runs a proposal
   pass per verification pass; its bytes, and every non-byte term (KV, PLE, head, commit, rollback), are outside
   this accounting. `tinytitan_mtp_phases.py` reads them as `proposal=` milliseconds, not as bytes.
7. **One session, one prose prompt, positions 10-392.** Same trace as #194, so the same ≤0.26 % denominator
   skew and the same prompt-shape limit; the code-shaped decode in the September study is not represented.
8. **`io_ms` is not converted to a rate anywhere in this section.** §5ag's marginal 17 ms for 54.4 MiB is 3.20
   GB/s and matches the saturated device, but its absolute io_ms/bytes ratio does not, so claims here are byte
   claims only.
9. **My own launch masked the driver's exit status** by piping it through `tee`, so the background task reported
   exit 0 while the driver raised `SystemExit`. The refusal text in the log is the evidence; the status was
   lost. That is the AUD-230 shape, in my command line rather than in the driver.
10. **Nothing in `sources/` changed.** `/tmp/score195.py`, `/tmp/score195b.py`, `/tmp/score195c.py` and
    `/tmp/locality195.py` are throwaways, so no test was added and the golden baseline is **not checked**
    (docs-only, as in §5af-§5ah). Provenance: tree clean at `HEAD 0ef95b7`, 04:29 release binary with no
    `sources/` commit since (`git log -1 -- sources/` = 04:28:32), AC 100 %, `memory_pressure -Q` 85 % free,
    swap 6.41 GiB, load average 3.42, no model process before or after.

## 5aj. #196 makes the leakage rule structural in the shipped screen: a bounded strictly-past recency ranking names experts that really are selected 65.4 % of the time and still finds only 16 worth-staging picks in 18 001 links, because the cache already holds 17 985 of them (2026-10-16)

§5ah ends by naming prediction quality as the only lever left, and §5ai closes the last predictor-shaped question
that could be answered without a GPU (MTP's byte half). Everything that campaign used to rank candidates was a
throwaway in `/tmp` — including the script that scored the previous window's temporal arms. So this task's
question was not "is there a better ordering", asked a fourth time, but the one thing that would make the answer
reusable: **can the shipped screen be made incapable of answering it dishonestly.** The standing instruction
here is "do not rely just on old tests", and the old tests could not have caught what this section reports —
the falsifier paragraph below runs them against a deliberately leaky build and they pass.

**The rule, stated once.** When the ring issues a plan for layer *L+1* at position *p*, the engine knows the
earlier positions' selections at layer *L+1* and position *p*'s layers ≤ *L*. It does not know *(p, L+1)* — that
is what it is trying to guess. Any feature that reads the target row (its selection, its misses, its residency)
is not a predictor, and on this trace reading it inflates a 152-issue, 13-useful arm into 1 994 issues at
precision 0.9975, which is why the rule had to become code rather than a comment.

**What landed** (`894ecfa`, all of it in `benchmark/`):

| piece | where | what it forbids or enables |
| --- | --- | --- |
| `Link` | `tinytitan_prefetch_trace_analysis.py:40` | one *L → L+1* plan, carrying only `source`, `target`, `top_m`, `counts`, `history` — a ranker cannot reach anything else |
| three uniform rankers | `:97`, `:102`, `:110` | `recorded`, `transition`, `recent`, all `Link -> list[int]` |
| `RANKERS` / `PREDICTORS` | `:120-121` | one derived name list feeds `--predictor`'s `choices` *and* `analyze`'s refusal, so an arm cannot exist in the CLI and not in the guard |
| position-boundary publish | `:164-165` | history is appended **after** every link at that position is scored — the leak is structurally unavailable |
| `--window` | `:184-194` | depth of strictly-past history, default 8, refuses ≤ 0 |
| name refusal | `:131-132` | `unknown predictor 'frequency'` raised, not silently scored as `transition` |

Five tests now pin it (`test_tinytitan_prefetch_trace_analysis.py:17, 33, 42, 55, 67`): the pre-existing
transition-prior recall test, the unknown-name refusal, a positive test that `recent` credits a past selection
the probe misses, the no-target-row test, and the window-bound test. All five pass; the python gate is clean
(`ruff 0.16.7`, check and format, 201 scripts parse under 3.13).

**The falsifier, run rather than asserted.** I copied module and tests to `/tmp/leakcheck` and moved the publish
above the scoring loop — a one-line change that makes the target position's own selection visible to the ranker.
Three tests fail, with the exact messages `3 != 1`, `1 != 0` and `4 != 0`, and **the two pre-existing tests stay
`ok`**. That is the "old tests" finding in concrete form: the suite that shipped before this commit certifies a
leaky screen, because it only ever scored arms that cannot read the future.

### The shipped screen's own numbers, on #194's ring-off trace

18 384 records, 383 positions, 48 layers, identity `misses == experts \ resident` holding on 18 384 of 18 384,
18 001 scored links (layer 47 has no *L+1*), 32 046 actual misses:

| predictor | top_m | window | issues | useful | recall | precision | selections named (of 10) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| recorded (shipped probe) | 1 | — | 2 151 | 1 622 | 0.0506 | 0.7541 | **0.919** |
| recorded | 2 | — | 4 899 | 3 486 | 0.1088 | 0.7116 | 1.798 |
| recorded | 4 | — | 12 064 | 7 587 | 0.2368 | 0.6289 | 3.402 |
| recent | 1 | 4 | 21 | 1 | 0.00003 | 0.0476 | **0.654** |
| recent | 1 | 8 | **16** | 1 | 0.00003 | 0.0625 | 0.654 |
| recent | 1 | 16 | 9 | 0 | 0.0 | 0.0 | 0.650 |
| recent | 2 | 8 | 44 | 1 | 0.00003 | 0.0227 | 1.192 |
| recent | 4 | 8 | 152 | 13 | 0.0004 | 0.0855 | 2.085 |
| transition | 1 | — | 36 | 30 | 0.0009 | 0.8333 | 0.722 |
| transition | 2 | — | 88 | 71 | 0.0022 | 0.8068 | 1.324 |

The last column is the screen's `raw_topk_agreement` × 10 — the mean number of the ten actually-selected experts
that the plan names. At top_m = 1 it is also a probability, and the honest way to read those three rows is as
probabilities against a chance level of 10/512 = **1.95 %** (one pick from 512 landing among the 10): the probe
hits 91.9 % of the time, the transition prior 72.2 %, and a bounded strictly-past recency ranking **65.4 %, 33×
chance**.

**The finding is the issue column, not the precision column.** `recent` is not a bad guesser: 65.4 % of its
single picks are experts the router really does select at *L+1*, 33× chance, and the transition prior reaches
72.2 %. It is a bad *staging* rule, because 17 985 of its 18 001 picks are already resident when the plan is
issued — so only **16 links out of 18 001** offer it anything to read, and one of those 16 turns out to be
needed. Deeper history makes that worse, not better (window 16 → 9 issues, 0 useful), and widening does too: at
depth 4 `recent` reaches 0.0004 recall where the probe reaches 0.2368, 592× the demand.

That is §5ai's locality table read from the other side. It measured that the previous token already holds 3.820
of the next token's 10 selections and the cache covers 4.439 more, leaving 1.741 fresh reads; this section
measures the consequence for a predictor — **recurrence and cache residency are the same mechanism**, so a
temporal ranking spends its entire vocabulary on experts the 96-slot ring already has. That is why the September
study's `previous token` arms scored P@1 0.000 and why the three temporal shapes in §5w staged nothing live: not
a scoring artifact, an empty order.

### What the leak would have bought

Same trace, leaky module, `recent` only (`recorded` and `transition` are byte-identical between the two modules,
so the leak is specific to the arm that reads history):

| arm | clean: issues / useful / precision | leaky: issues / useful / precision |
| --- | --- | --- |
| `recent` top_m=1 w=4 | 21 / 1 / 0.048 | 213 / 213 / **1.0000** |
| `recent` top_m=1 w=8 | 16 / 1 / 0.063 | 137 / 137 / **1.0000** |
| `recent` top_m=1 w=16 | 9 / 0 / 0.000 | 119 / 119 / 1.0000 |
| `recent` top_m=2 w=8 | 44 / 1 / 0.023 | 466 / 464 / 0.9957 |
| `recent` top_m=4 w=8 | 152 / 13 / 0.0855 | **1 994 / 1 989 / 0.9975** |

A leaky recency arm reads as precision 1.000 because it names the target row's own selections and then discards
the resident ones — the definition of "predicting" a miss. Published as-is that table says *a bounded recency
ring matches the probe's recall at 8× fewer reads*, and it would have been the basis for building an engine arm
that stages nothing. The honest figure for the same arm is 152 issues, 13 useful.

### Three precisions that are not three measurements of one thing

§5ah's rank table and the September study both quote precision, and this screen adds a third figure; they do not
line up and the reason is definitional, not a disagreement:

| figure | what it is | tier |
| --- | --- | --- |
| 0.552 | study probe P@1 — precision of the first non-resident pick of the whole list, scored against a residency simulation calibrated to the **ring-on** footer (`issued 43.0` against measured `42.74`) | 64 slots |
| 0.522 | §5ah's *pooled* precision over its per-rank table (ranks 1-8 alone carry 13 489 issues) | 96 slots |
| 0.7541 | this screen at `--top-m 1` — the probe list truncated to 1 **before** the residency filter | 96 slots |

The checkable cross-validation is the count, not the ratio: the screen's top_m=1 issues **2 151** and §5ah's
rank-1 row issues **2 151** on the same trace, so the shipped tool reproduces the throwaway that produced §5ah.
No truncation depth reproduces 0.522 — the curve runs 0.7541 (1), 0.7116 (2), 0.6679 (3), 0.6289 (4), 0.5851
(5), 0.5424 (6), 0.4687 (8), 0.3995 (10) — because that figure pools ranks rather than deepening a plan, and
its per-rank denominators (2 151, 1 999, 1 916, …) are a different partition of the same issues. Two consequences
for how §5ah line 3396 should be read: its "same shape, slightly worse" sentence compares the study's *rank-1*
figure with this campaign's *pooled* one, so it is not a tier comparison, and the 0.522-vs-0.552 direction it
asserts is not supported by anything in that table. Both numbers it names are reproduced; the comparison between
them is the defect, and it is recorded here rather than edited into §5ah, which is a published record.

### Where this leaves the feeder question

§5ah established the size of the prize (a perfect next-token predictor issues 34.65 reads/token, fewer than the
ring's 39.73, and adopts all of them — ≈ +4.6 %). This section establishes, on the shipped tool rather than a
scratch script, that **nothing issue-time feasible is a candidate**: the two engine-buildable strictly-past
rankings offer 16 and 36 worth-staging picks in 18 001 links, and the only ranking with volume is the probe's own
list, which the ring already stages in that order. Combined with §5ah's "rank order is the right staging rule,
measured" — the first non-resident pick is the best pick available in the list — the predictor lever is closed on
precision, not on timing, and it is closed with a guard in the tree so that the next arm proposed against it has
to pass the same three tests.

**The GPU guard, measured rather than assumed — and one prior reading corrected.** §5w's deviation (2) recorded
that the `ioreg` AGX "Device Utilization %" reader "showed 61 % with the machine idle … so it is not an occupancy
instrument on this host". Three samples taken for this section, with the sub-counters, say otherwise: 52 / 71 /
74 %, with `Tiler`, `Renderer` and `Device` agreeing to ≤ 1 point every time, while `WindowServer` holds 43.8 %
CPU and `Qoder Helper (Renderer)` 42.3 %. A latched gauge does not move with three correlated counters and a live
compositor. So the reading §5w took as a broken instrument was real co-tenant GPU work, gate0's 15 % refusal is
correct, and **every remaining timing arm in this campaign — the live MTP pair, #163's wiki row — stays blocked
while this app is running**, which is also why the operator's "the macbook is idle, stop checking" holds for
*my* processes (the `pgrep` guard is empty, nothing of mine runs) but not for the desktop's. `--allow-busy-gpu`
was not passed and nothing in this section comes from a contended GPU.

Deviations.

1. **The recall denominator is 18 001 scored links, not 18 384 layer-slots.** Layer 47 has no *L+1* target, so
   it is absent from every recall figure here, and it is the hungriest layer in the trace: 33 709 misses over all
   48 layers against 32 046 over scored links, i.e. **1 663 misses (4.9 % of all demand) sit on the unscored
   layer**, at 4.34 per position against the scored average of 1.78. Demand here is therefore 1.780 misses per
   link, which is *not* §5ai's 1.834 per layer (that figure is 33 709 ÷ 18 384); the two differ by exactly the
   missing layer.
2. **`recent`'s precision is not measurable at its own issue count.** 1 useful of 16 issues, and 0 of 9 at
   window 16: the quoted 0.0625 and 0.0000 have no error bar worth printing. What is solid is the *count* — 16
   and 9 against the probe's 2 151, a 134× gap that no sampling noise reaches.
3. **The ranking measured is "recency among everything", filtered to nonresident — not a recency ranking of the
   nonresident set.** That alternative (an LRU-style "stage the oldest-seen thing") was not scored, and it would
   carry no next-token information at all; it is a cache-policy idea, not a predictor, and #196's subject is
   predictors.
4. **No clocks and no bytes appear in this section.** The screen is offline and non-mutating by design, so
   nothing here is a tok/s claim, and no §6 or §8 closure is reopened or overturned by it.
5. **One session, one prose prompt, positions 10-392.** Same trace as #194 and #195, so the same ≤ 0.26 %
   denominator skew and the same prompt-shape limit; the code-shaped decode the September study scored is not
   represented, and its 0.554 row is not comparable to this table for the reasons above.
6. **The leaky module is a throwaway, not a shipped arm.** It exists only at `/tmp/leakcheck/` and was deleted
   after this section; nothing in `benchmark/` can score an arm that reads its target row.
7. **The transition prior's own arm is not the engine's.** `rank_transition` accumulates *L → L+1* counts from
   strictly-past positions, which is what a runtime table could learn, but the shipped probe is a trained
   next-token head, so this row is a lower bound on what a learned prior could do — not a measurement of the
   head's alternatives.
8. **Nothing in `sources/` changed.** `894ecfa` touches two files under `benchmark/`, so no engine behaviour
   moves and the golden baseline is **not checked** (docs-and-python only, as in §5af-§5ai). Test inventory: 3
   tests added, 2 pre-existing tests unmodified, both green. Provenance: tree clean at `HEAD 894ecfa`, AC 100 %,
   `memory_pressure -Q` 83 % free, swap 5.49 GiB used, load average 4.45, Swift 6.4 / macOS 27.0.1, `pgrep` guard
   empty before and after, and no model process at any point in this task.
9. **Every `top_m` row in this section was scored under the withdrawn v4.3 issue policy, and §5ak corrects what
   the axis means.** Before `bc4c648` the screen capped the ranking at `top_m` *then* dropped the resident names,
   which is what v4.3 did and not what ships; the shipped ring is handed the whole list and caps the *reads*. So
   the 0.7541-at-`top_m`=1 row is a marginal count of §5ah's rank table, not the engine's issue total, and the
   "2 151 = 2 151" cross-validation in this section proves less than it appears to. §5ak adds the policy axis,
   reproduces the shipped total exactly, and states which figures here survive the correction.

## 5ak. #197 puts the issue policy inside the screen: the shipped filter-then-cap rule reproduces §5ah's live ring to the count, and the depth ladder that reproduction exposes is decided by the engine's own series rather than by the screen (2026-10-16)

§5aj's deviation (9) is this section's premise, and it was checked rather than assumed. The engine hands the ring
the **whole** ranked prediction and lets the ring's own size cap the reads
([RealForwardRunner+DecodeMoE.swift:334-348](../sources/TinyTitan/Runtime/Inference/RealForwardRunner+DecodeMoE.swift#L334)),
and `ExpertPrefetchRing.begin` then computes `wanted = experts.filter { !resident.contains($0) && … }`,
`count = min(wanted.count, free.count)` and stages `wanted.prefix(count)`
([ExpertPrefetchRing.swift:88-98](../sources/TinyTitan/Runtime/Inference/ExpertPrefetchRing.swift#L88)).
That is **filter first, cap second**. The screen had been doing the opposite — cap the ranking at `top_m`, then
drop the resident names — which is exactly the v4.3 behaviour the same comment records as the reason v4.3 lost.
One flag name was sweeping two different axes, and every `top_m` figure in §5aj belongs to the withdrawn one.

### What landed

| piece | where | what it does |
| --- | --- | --- |
| `Link.candidate_limit` | [benchmark/tinytitan_prefetch_trace_analysis.py:50](../benchmark/tinytitan_prefetch_trace_analysis.py#L50) | every ranker may expose `max(top_m, len(prediction))` names, so the two policies differ **only** in the order of cap and filter |
| `stage_filter_first` | [:125](../benchmark/tinytitan_prefetch_trace_analysis.py#L125) | the shipped rule — drop resident names, then take the first `top_m` |
| `stage_truncate_first` | [:135](../benchmark/tinytitan_prefetch_trace_analysis.py#L135) | the withdrawn v4.3 rule — take the first `top_m` names, then drop resident |
| `analyze(…, policy="filter")` | [:151-157](../benchmark/tinytitan_prefetch_trace_analysis.py#L151) | unknown names raise; the default is the shipped arm, so a reader gets the engine's behaviour unless they ask for the other one |
| `--policy`, `report["policy"]` | [:196](../benchmark/tinytitan_prefetch_trace_analysis.py#L196), [:218](../benchmark/tinytitan_prefetch_trace_analysis.py#L218) | the axis is named in the artefact, not implied by a flag's default |
| the pinning test | [benchmark/test_tinytitan_prefetch_trace_analysis.py:82](../benchmark/test_tinytitan_prefetch_trace_analysis.py#L82) | one fixture, both policies: the pool's rank-1 name is resident and rank-2 is a real miss, so shipped stages 1 read and 1 useful while v4.3 stages 0 and 0 |

Test-first, in full: the new test was written before the implementation and failed with
`TypeError: analyze() got an unexpected keyword argument 'policy'` — the expected RED, since the axis did not
exist. It is now green along with the five pre-existing tests (6/6, `python3 -m unittest`, serial). The five are
unmodified; their fixtures leave every target's `resident` empty, so the two policies coincide on them and the
shipped default moves none of their assertions. `ruff format` reflowed two lines of *my own new test* (not a
pre-existing file) and the suite was re-run to green after it; `tools/lint.sh python` reports
`ok (ruff 0.16.7, check and format clean, 201 scripts parse under 3.13)`. Commit `bc4c648`.

### Both policies on the #194 ring-off trace, `recorded` ranking, 18 001 scored links

| `top_m` | filter issues | filter useful | filter precision | truncate issues | truncate useful | truncate precision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | **15 257** | **7 968** | **0.5223** | **2 151** | **1 622** | **0.7541** |
| 2 | 26 594 | 12 660 | 0.4760 | 4 899 | 3 486 | 0.7116 |
| 3 | 34 356 | 15 300 | 0.4453 | 8 212 | 5 485 | 0.6679 |
| 4 | 39 274 | 16 768 | 0.4269 | 12 064 | 7 587 | 0.6289 |
| 5 | 42 130 | 17 509 | 0.4156 | 16 437 | 9 617 | 0.5851 |
| 6 | 43 737 | 17 863 | 0.4084 | 21 239 | 11 521 | 0.5424 |
| 7 | 44 604 | 18 011 | 0.4038 | 26 548 | 13 411 | 0.5052 |
| 8 | 45 063 | 18 079 | 0.4012 | 32 283 | 15 130 | 0.4687 |
| 9 | 45 277 | 18 106 | 0.3999 | 38 550 | 16 693 | 0.4330 |
| 10 | 45 339 | 18 113 | 0.3995 | 45 339 | 18 113 | 0.3995 |

**The shipped arm is the one the engine already produced.** `filter` at `top_m` = 1 reads 15 257 times and finds
7 968 correct names, and §5ah's offline replica of `ExpertPrefetchRing.begin` — built from the live run, validated
against three engine counters — reports **15 257 issued** and **7 968 correct** on the same trace
(§5ah line 3356), with overall precision 0.522 against this table's 0.5223. Identical to the read. `truncate` at
`top_m` = 1 reads 2 151 times at 0.7541 — which is §5ah's **rank-1 marginal row** (2 151 issues, 0.754 precision,
line 3393), a slice of a partition, not an issue total. That is what §5aj's "2 151 = 2 151" cross-validation
actually matched — one marginal row of the rank table, while the engine's own total, which the shipped tool now
reproduces exactly, went unscored by it for a whole section. The partition reconciles completely: the per-rank
marginals sum to 13 489 over ranks 1-8 (the figure §5aj quotes) and to 15 257 over all ten, which is the shipped
total. The two policies meet at `top_m` = 10 because the recorded pool is exactly ten names wide, so beyond that
the cap cannot bind and only the filter order differs.

### Under the shipped rule `top_m` is ring depth, not pool width

The caller never truncates, so the pool is always the full prediction; the only thing `top_m` can be is
`slotCount`, the number of reads in flight — the knob `ringDepth = max(1, profile.prefetchDepth)` sets in the
build path. So the two rows at `top_m` = 1 are the two arms of the v4.3 withdrawal, and the gap between them is
its size: **7.09× more reads** (15 257 against 2 151) and 4.91× more correct names. §5aj's precision curve
0.7541 → 0.3995 is the withdrawn arm's *precision against pool width*; the shipped arm's *precision against ring
depth* is 0.5223 → 0.3995, and the two are not interchangeable at any depth below ten.

### The depth ladder, priced with this campaign's own money

Counts from the `filter` column, divided by 384 tokens as §5ah does, `2.641 MiB` per read. `adopted-equiv`
multiplies `useful` by the one measured conversion in the record — 19.25 adopted of 20.75 correct at depth 1 =
**0.9277** — because the screen counts a correct-but-late read as adopted and §5ah measured that 1.50 reads/token
never were. `ceiling` is `Σ min(depth, |misses|)`, the reads a *perfect* predictor could use at that depth.

| depth | reads/tok | spec MiB/tok | useful/tok | adopted-equiv | demand MiB avoided | marginal precision of slot *n* | ceiling reads/tok | cum % at route-3 prices |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 39.73 | 104.9 | 20.75 | 19.25 | 50.8 | 0.5223 (pooled) | 33.67 | **+5.81** |
| 2 | 69.26 | 182.9 | 32.97 | 30.58 | 80.8 | 0.4139 | 55.53 | +8.55 |
| 3 | 89.47 | 236.3 | 39.84 | 36.96 | 97.6 | 0.3402 | 68.78 | +9.70 |
| 4 | 102.28 | 270.1 | 43.67 | 40.51 | 107.0 | 0.2985 | 76.26 | +10.16 |
| 5 | 109.71 | 289.8 | 45.60 | 42.30 | 111.7 | 0.2594 | 80.12 | **+10.29** |
| 6 | 113.90 | 300.8 | 46.52 | 43.16 | 114.0 | 0.2203 | 82.03 | +10.27 |
| 7 | 116.16 | 306.8 | 46.90 | 43.51 | 114.9 | 0.1707 | 82.90 | +10.21 |
| 8 | 117.35 | 309.9 | 47.08 | 43.68 | 115.4 | 0.1482 | 83.26 | +10.17 |
| 9 | 117.91 | 311.4 | 47.15 | 43.75 | 115.5 | 0.1262 | 83.42 | +10.14 |
| 10 | 118.07 | 311.8 | 47.17 | 43.77 | 115.6 | 0.1129 | 83.45 | +10.13 |

The screen's depth-1 row validates; its extrapolation to depth > 1 is **falsified by the engine**, and the reason
is structural rather than a calibration question. Price a saved demand MiB `d` and a spent speculative MiB `c`
from the two arms that measure them:

- **#193's ring-off↔on arm** gives `49.8d − 104.2c = 5.6`. Read at `c = 0` it gives `d = 0.1124 %/MiB`, and then
  the ladder is monotone: depth 10 is the best arm at **+13.0 %**. Take #192's `c = 0.0530` as a flat cost instead
  and the same equation gives `d = 0.2233`. §5ah's route-3 pair splits the difference by modelling the feeder as
  partly *substituting* demand reads (`322.9c + 17.1(d − c) = 17.1`), which solves to `c = 0.042, d = 0.201` —
  and that is the pair in the last column: **cumulative maximum +10.29 % at depth 5**, depth 2 worth +2.74 % more
  than depth 1.
- Both readings are wrong about depth, and wrong in the same direction: **no price pair this campaign can produce
  puts the peak at depth 1.** The route-3 pair peaks at **+10.29 % (depth 5)**, #192's flat cost gives a lower
  curve that still peaks at **+9.6 % (depth 4)**, and the `c = 0` corner is monotone to **+13.0 % (depth 10)**. The
  live ladder in the build path measured **M=1 +12.2 % (hit 78.7 %), M=2 +6.0 % (77.9 %), M=4 −9.8 % (77.1 %)**
  against the same prefetch-off control in the same batch: depth 2 is 6.2 points *worse* than depth 1 and depth 4
  is 22 points worse, while all three priced ladders say both are better — depth 2 by 2.6-3.4 points and depth 4
  by 3.8-6.3.

What the screen cannot see is what the build-path comment names: the drive is saturated while it reads, so a
second speculative read steals service from the demand burst it was meant to anticipate and lands too late to be
adopted — which is why the live *hit* column falls with depth while this table's byte totals rise. The `useful`
counter has no arrival time in it, so depth beyond one is outside what the screen measures. **The predictor lever
is screened by this tool; the depth lever is closed by the engine's own paired series, and this section adds
nothing to it.**

### The one figure that does carry, and one §5aj magnitude that moves

At depth 1 the shipped ring reads 15 257 times where 12 930 links have any demand miss at all: **2 327 reads
(6.06/token, 16.0 MiB/token) are traffic a perfect depth-1 predictor would never generate**, and of the 12 930
links that needed something the single pick is right on 7 968 — **61.6 % of what depth 1 can reach, 24.9 % of all
demand**. That is the prize §5aj named, now on the shipped denominator.

§5aj's other headline needed the same correction and one number in it was off by an order of magnitude. Under the
shipped policy the two engine-buildable strictly-past rankings do not offer "16 and 36 worth-staging picks":
`recent` issues **607** reads (92 useful, precision 0.1516) and `transition` **805** (514 useful, precision
0.6385) at depth 1, and both saturate at 740 and 1 705 reads by depth 10. The qualitative conclusion is untouched
— 805 against the 12 930 links that need a read is 6.2 %, and the probe's own list is the only ranking with
volume — but the quoted magnitudes belonged to the withdrawn arm. §5aj's deviation (9) points here.

Deviations.

1. **Per-token figures divide by 384, per-link figures by 18 001.** Same convention as §5ah (its 39.73 reads/token
   is 15 257 ÷ 384 while the trace holds 383 positions), and it makes every per-token number here ~0.26 % high
   against a per-position reading.
2. **The 0.9277 adopted conversion is measured at depth 1 only.** Applying it to depths 2-10 is *generous* to them:
   lateness must grow with depth for the reason this section is about, so the true cum curve is flatter and lower
   than the last column, and its peak is earlier. The live ladder already says where the peak is; nothing here
   moves it.
3. **Three ring behaviours stay unmodelled**, all named in §5ah: `active` deduplication, the rare busy slot, and
   arrival time. The first two make the depth-1 issue count a 0.35 % upper bound; the third is the whole
   depth argument.
4. **The live M=1/M=2/M=4 series is quoted from the tree, not re-measured here**, and it comes from its own batch:
   per the campaign rule its +12.2 % is not compared with #193's +5.6 %, only with its own M=2 and M=4 arms. The
   "hit" column is quoted as the comment labels it; which counter it names was not established.
5. **`raw_topk_agreement` is a prefix property of the ranking, not of the issued set**, so at equal `top_m` it is
   identical under both policies and none of §5aj's selection-named figures needed re-running.
6. **Only `recorded` is the engine's ranking.** The `filter` rows for `transition` and `recent` are screen-only
   arms of synthetic rankers; the shipped ring never sees them.
7. **No clocks, and nothing in `sources/` moved**: `bc4c648` touches two files under `benchmark/`, so no engine
   behaviour changes and the golden baseline is **not checked** (docs-and-python, as in §5af-§5aj). Provenance:
   code at `HEAD bc4c648` with only this doc file modified when the environment was read, AC 100 %,
   `memory_pressure -Q` 83 % free, swap 5 458 MiB used, load average 2.57, Swift 6.4 / macOS 27.0.1, `pgrep` guard
   empty before and after, no model process at any point, and the trace is the #194 artefact pair under
   `/tmp/trace194/` (both arms' responses byte-identical, `md5` recorded in §5ah).
8. **The link form in this repository's prose is repo-relative, and `file:///` absolute links break CI.** The
   markdown-link check (`.github/workflows/ci.yml:984-1018`) resolves a link relative to the document's own
   directory and skips only `http(s)://`, `#` and `mailto:` — so a `file:///Users/…` target is read as a path
   under `docs/` and reported broken. This section's 9 references and **7 more that already sat on `main`**
   (2 in §5af, 4 in §5ag, 1 in §5ah) were in that form and are now `../sources/…` or `../benchmark/…`; the
   checker reports "Checked 66 local link(s); 0 broken". The pre-existing rows are the AUD-180 class — a doc
   reference no local gate caught — found again, and `tools/lint.sh docs` does not cover link targets.

## 5al. #198 scores the one axis the live depth series never varied: slot 2's marginal precision spans 8.1x across the 47 links and is stable within a session (r = 0.922), so uniform-depth closure does not close it — but the campaign's two price models put a layer-selective arm on opposite sides of zero, and only a live paired run can settle it (2026-10-16)

§5ak closed **depth as a uniform knob**: one number, applied to every layer, decided by the engine's own paired
M=1/M=2/M=4 series. That series never varied the other axis in the same line of the build path — depth is *one
scalar for all 47 links* ([RealForwardRunner+BuildCore.swift:87](../sources/TinyTitan/Runtime/Inference/RealForwardRunner+BuildCore.swift#L87),
handed to the ring at `slotCount: ringDepth` line 99) — and the §5z result that survived the whole campaign says
why a uniform number is suspect: **which layer is busy is a property of the model; which expert is busy is a
property of the conversation**, and six cross-trace per-layer budget transfers all beat the shipped uniform split.
So the question #198 asks is narrow and unasked before: is the second read worth staging *at some layers*?

### First finding: a counterfactual depth arm must be scored on the ring-off trace

The screen ran on `on.jsonl` first and returned a depth-1 precision of **0.0377** — not the 0.5223 §5ak
established for the shipped arm. That is not a bug in either section; it is what the two traces record.
In `on.jsonl` the engine's `misses` field is written **after** the ring has already staged its read, so a
counterfactual arm evaluated there is scored against a demand set the shipped prefetcher already consumed:

| scored on | depth | issues | useful | precision | recall | `actual_misses` |
|---|---|---|---|---|---|---|
| `off.jsonl` (ring off) | 1 | 15 257 | 7 968 | **0.5223** | 0.2486 | 32 046 |
| `off.jsonl` | 2 | 26 594 | 12 660 | 0.4760 | 0.3951 | 32 046 |
| `on.jsonl` (ring on) | 1 | 15 257 | **575** | 0.0377 | 0.0233 | 24 653 |
| `on.jsonl` | 2 | 26 594 | 5 267 | 0.1981 | 0.2136 | 24 653 |

The issue counts agree to the count across the two substrates (15 257 and 26 594 — they are properties of the
prediction list, not of what ran), while the useful column collapses. Every number below is on `off.jsonl`, the
same 18 384 observations and 18 001 scored links as §5ak. The on-trace stays the right substrate for *adoption
and lateness* (§5ah/§5ai) and the wrong one for *counterfactual value*; nothing in the repo says which is which,
so it is recorded here.

### Method

No production code changed and nothing new was added to `benchmark/`. The shipped screen
([../benchmark/tinytitan_prefetch_trace_analysis.py](../benchmark/tinytitan_prefetch_trace_analysis.py)) was run
at `--top-m` 1, 2, 3, 4, 6, 10 with `--predictor recorded --policy filter` on `off.jsonl`, and the per-link
counters differenced between adjacent depths:

    slot-k marginal precision = (useful@k − useful@k−1) / (predicted_nonresident@k − predicted_nonresident@k−1)

That is the value of the k-th read *alone*, on the shipped issue policy — filter-then-cap means `--top-m` is ring
depth, per §5ak. The differencing, the two price models and the gate tables are arithmetic, done in throwaway
scripts under `/tmp/depthsel/` (`slot2.py`, `gates.py`, `split.py`, `layeraxis.py`, `verify.py`); they are not
proposed as a shipped tool.

### The spread is real and it is 8-10x, across all three slots

| slot | min | p25 | median | p75 | max | ratio |
|---|---|---|---|---|---|---|
| 1 — the shipped read (`prec1`) | 0.085 | 0.410 | 0.567 | 0.667 | 0.836 | **9.8x** |
| 2 — one deeper read (`prec2`) | 0.093 | 0.284 | 0.454 | 0.533 | 0.752 | **8.1x** |
| 3 — a third (`prec3`) | 0.064 | 0.248 | 0.372 | 0.478 | 0.660 | **10.4x** |

All 47 L→L+1 links have a slot-2 read available and none saturates at depth 3 either — the smallest slot-2
increment over the whole trace is 69 issues (slot 3: 19, slot 4: 10), so the deeper ring is never offered a layer
it cannot fill. Slot 2 pooled over the trace is 4 692 useful against 11 337 extra reads = **0.4139** — below the
median of the per-link distribution, which is exactly what a spread of that size means: the uniform arm averages a
good half and a bad half into one number that is neither. The eight most valuable slot-2 links:

| link | `prec2` | extra useful/tok | extra reads/tok | its `prec1` |
|---|---|---|---|---|
| L 6 | 0.752 | +0.55 | +0.73 | 0.836 |
| L21 | 0.708 | +0.59 | +0.83 | 0.809 |
| L37 | 0.704 | +0.49 | +0.70 | 0.820 |
| L 5 | 0.634 | +0.51 | +0.80 | 0.773 |
| L28 | 0.600 | +0.32 | +0.53 | 0.769 |
| L40 | 0.596 | +0.17 | +0.28 | 0.757 |
| L 7 | 0.565 | +0.37 | +0.66 | 0.705 |
| L44 | 0.563 | +0.15 | +0.27 | 0.531 |

Two structural facts come with it. **One ordering, not two:** `prec1` and `prec2` correlate at Pearson **r = 0.851**
across the 47 links (`prec2` vs `prec3` at **0.899**), so a layer that is a good use of a first read is a good use
of a second — one per-layer quality list serves both gates. **Concentration:** ranked by slot-2 useful reads, the
top 11 links carry 1 865 of 4 692 = **39.7 %** and the top 23 carry 3 187 = **67.9 %**, for 2.26-14.83 extra
reads/token instead of uniform depth 2's 29.52.

### And it is stable within the session, which is the part §5z could not show

The obvious objection is that a per-layer statistic from one 384-token capture is that conversation's noise. A
position split tests it: the trace's positions 10-201 and 202-392 are scored separately (this is a clean
out-of-sample test for a *layer-selection* rule, because `recorded` reads only the trace's own prediction list and
the gate is fitted on one half and applied unchanged to the other):

| per-link statistic | half-1 mean | half-2 mean | Pearson r across the 47 links |
|---|---|---|---|
| `prec2` (slot-2 marginal) | 0.428 | 0.418 | **0.922** |
| `prec1` (slot-1) | 0.514 | 0.538 | **0.939** |

A gate fitted on half 1 and never shown half 2 selects 24 / 17 / 9 / 6 links at `prec2` ≥ 0.45 / 0.50 / 0.55 /
0.60 and realises **0.540 / 0.568 / 0.620 / 0.672** on half 2. All four clear the optimistic break-even derived
below (0.4579) and none reaches the pessimistic one (0.8403): what survives the split is the *ranking* — the
layers that aimed well in the first half aim well in the second — while whether that earns anything stays the
price dispute below.

### Two price models for one extra read, and they disagree in sign

The screen cannot say whether 0.546 realised precision is good, because the campaign has two defensible answers.
Both are anchored on the same live pair quoted in §5ak — **M=1 +12.2 %, M=2 +6.0 %**, one batch, one control, so
neither crosses batches.

1. **Optimistic.** Solve a benefit `B` per adopted-equivalent read/token and a cost `K` per speculative
   read/token from the two ladder arms and this trace's own counts: `B = +5.142`, `K = −2.184`. Break-even
   marginal precision `= −K / (B × 0.9277)` = **0.4579**. It reproduces both anchors by construction, and
   **22 of the 47 links clear it**, carrying 2 739 of the 4 692 slot-2 useful reads (58.4 %).
2. **Pessimistic.** Hold the benefit at §5ak's route-3 price (`d = 0.201 %` per demand MiB avoided, times
   stride, times the 0.9277 conversion → 0.4925 % per adopted-equiv read/token) and charge the whole measured
   6.2-point depth-1→2 loss to the 11 337 extra reads → **0.4138 % per extra read/token**. Break-even
   **0.8403** — and **no link reaches it**; the best is 0.752.

The optimistic fit is the one to distrust, and the ladder itself proves it. Its depth-4 falsifier: predicted
`u4 × 0.9277 × B + r4 × K` = **−15.1 %** against the **measured −9.8 %** (and +12.2 / +6.0 reproduced exactly).
The error is specific — a *constant* per-read cost under-prices the third and fourth reads, because each one
contends longer with the demand burst — which is precisely the mechanism §5ak named. It says nothing about
whether a *per-layer* constant is right, but it is the same extrapolation class the selective arm is asking for.

### Arm B — depth 2 only at the links that aim best

Using the whole trace to fit the gate (the live arm would have to ship a list; see the seam below):

| gate on `prec2` | links kept | extra reads/tok | extra MiB/tok | realised precision | optimistic net | pessimistic net |
|---|---|---|---|---|---|---|
| 0.00 — uniform depth 2 | 47 | 29.52 | 78.0 | 0.414 | **−6.20 %** (the ladder, by construction) | −6.20 % |
| 0.30 | 35 | 21.13 | 55.8 | 0.499 | +4.16 % | −3.55 % |
| 0.40 | 30 | 17.67 | 46.7 | 0.527 | +5.82 % | −2.73 % |
| **0.45** | **25** | **14.83** | **39.2** | **0.546** | **+6.21 %** | **−2.15 %** |
| 0.50 | 15 | 8.41 | 22.2 | 0.602 | +5.76 % | −0.99 % |
| 0.55 | 9 | 5.21 | 13.8 | 0.648 | +4.73 % | −0.49 % |
| 0.60 | 5 | 3.60 | 9.5 | 0.684 | +3.88 % | −0.28 % |
| 0.65 | 3 | 2.26 | 6.0 | 0.721 | +2.84 % | −0.13 % |

The gate's shape is the interesting part: at `prec2 ≥ 0.45` it keeps **66.2 % of slot-2's useful reads for 50.2 %
of its bytes**, so the pessimistic model's objection — speculative traffic — is the one thing it demonstrably
reduces, while its own break-even claim (0.8403) is a bar no arm of *any* depth can clear. A third reading,
route-3's own prices, is worth naming because it disagrees with both: under `c = 0.042 / d = 0.201` the selective
arm is +0.55 to +2.85 % against the shipped depth-1 arm, but so is *uniform* depth 2 (+2.74 %), i.e. route-3 says
the layer gate buys 0.11 points and depth itself is free — the model §5ak already rejected on live evidence.

### Arm A — depth 0, i.e. skipping the shipped read at the bad layers

The mirror question (the ring is already spending there; is the *first* read worth it?) was priced the same way and
answers no. Under route-3's prices, keeping only links with `prec1` ≥ threshold:

| gate on `prec1` | links kept | reads/tok | useful/tok | vs shipped |
|---|---|---|---|---|
| 0.20 | 42 | 35.34 | 20.09 | **+0.16 %** |
| 0.30 | 41 | 34.38 | 19.88 | +0.16 % |
| 0.40 | 36 | 30.23 | 18.49 | −0.06 % |
| 0.50 | 30 | 24.66 | 16.06 | −0.64 % |
| 0.60 | 18 | 15.03 | 10.60 | −2.26 % |

Nothing above +0.16 % is worth a code change, and the reason is arithmetic rather than subtle: route-3 prices a
demand MiB avoided about **4.8x** above a speculative MiB spent, so the reads that are lost are worth more than the
traffic that is saved. Under the optimistic pair the same gate models +6.4 to +10.5 % — which is the same
sign-disagreement as arm B, and is a *larger* extrapolation (arm A removes 4-25 reads/token where the ladder only
ever moved between 39.73 and 69.25), so neither model is being believed here either.

### What would settle it, and the seam it needs

One live paired arm: **selective depth 2 on a fixed link list versus the shipped uniform depth 1**, interleaved
A/B/A within rounds, same batch, same control, greedy responses hash-checked for invariance. Its predicted band
is −2.2 % to +6.2 % against a known uniform result of −6.2 % — i.e. the arm is only worth running because the
screen says the layer axis could be worth ~12 points relative to the uniform arm that was measured and rejected.

It is not implementable by flipping a knob, and this is the concrete finding for whoever runs it:

- `ExpertPrefetchRing.begin(model:layer:experts:resident:currentLayer:)`
  ([ExpertPrefetchRing.swift:65-114](../sources/TinyTitan/Runtime/Inference/ExpertPrefetchRing.swift#L65)) takes
  **no depth argument** — the depth arrives once, as the ring's `slotCount`, and `wanted.prefix(count)` caps every
  link identically.
- The ring carries **global counters only** — `issuedReads` (:51), `begins` (:58), `reclaimedOps` (:61) and the
  `observed*` gauges (:54-57). There is no per-link state, so an *online* gate that learns which layers aim badly
  has nothing to key on; it would be new bookkeeping, not a filter.
- What is already reachable is the **offline-fitted list**, which is how §5z's per-layer budgets ship: one value
  per layer in the profile, read at build time, handed to `begin` as a per-layer cap. That is the shape the live
  arm needs, and it costs a `depth:` argument (or a profile array) plus the guard that a per-layer depth outside
  `1...topKExperts` is refused the way line 89's scalar check already refuses it.

### Verdict

**The layer axis is genuinely untested, the effect is large (8.1x), stable within a session (r = 0.922, and a
half-trace gate transfers to the other half at 0.540-0.672), and concentration is real (top 23 links = 67.9 % of
slot 2's value). §5ak's closure was of uniform depth and does not automatically extend to it.** It stays a
*question* rather than a lever, because the campaign's two price models put the same arm at +6.21 % and −2.15 %,
the optimistic one fails an out-of-sample check at depth 4, and the pessimistic one's break-even (0.8403) is above
anything any link achieves — a bar that would also condemn the shipped depth-1 arm, which earns +12.2 % live.
No engine change is proposed, no default moved, nothing shipped.

Deviations.

1. **Every figure is from one 384-token capture, one session, one prompt shape.** The half-split test is
   out-of-sample *within* a session; it is not a cross-session portability claim. §5z's +0.957/+0.960/+0.976
   per-layer shape correlation across three captures is the evidence that the layer axis travels, and it is
   quoted, not re-measured here.
2. **`--top-m` differencing assumes the shipped filter-then-cap policy at each depth.** Depth k−1's issue set is a
   prefix of depth k's because both use the same ranking and the same `resident`/`active` filters, and the
   candidate side is checked rather than assumed: no link runs out of names at any depth used here (smallest
   per-link increments 69 / 19 / 10 reads for slots 2 / 3 / 4). What is **not** checked is the runtime free-slot
   pool — the rare busy slot §5ah names — which could make a depth-k read fail to stage where depth k−1's did;
   §5ak's 0.35 % bound is carried here rather than re-derived.
3. **The two price models are two readings of one three-point ladder, not two measurements.** Both are anchored
   on +12.2 % / +6.0 % / −9.8 % quoted from the tree; §5ak deviation 4 still applies — the "hit" column's counter
   was never established. The optimistic fit is used for its break-even and its falsifier, not as a forecast.
4. **`split.py`'s gate-transfer nets (+2.49 to +2.99 %) are not comparable to the arm-B table's nets.** That script
   prices its two anchors as depth-1 = +5.6 % (#193's batch) and depth-2 = −0.6 % (that batch's +12.2 % minus this
   ladder's 6.2 points), which composes across batches against the campaign rule, and it prints its break-even as
   `−K/B` (0.4441) while its own net formula multiplies benefit by 0.9277, so its stated bar is 0.4788 under its
   convention. **Only the price-free parts of that script are carried above**: the two half-to-half correlations
   and the realised precisions 0.540/0.568/0.620/0.672. Its nets are recorded as a direction and are not quoted
   in the verdict or the tables.
5. **The per-link `prec2` used for the gates is a whole-trace statistic, so the arm-B table is in-sample fitted.**
   The half-split rows are the honest estimate of what a shipped list would achieve; the table is the menu of what
   the axis could reach, and the two are deliberately separated.
6. **Adoption is still not in any of this.** `useful` has no arrival time (§5ak's closing sentence), so slot 2's
   extra reads are credited as if they always land in time. The live ladder's falling hit column with depth is
   the counter-evidence, and it is the reason the recommendation is a live arm rather than a default change.
7. **No clocks, and nothing in `sources/` or `benchmark/` moved.** The golden baseline is **not checked**
   (docs-only, as in §5af-§5ak); no model process ran at any point, `pgrep` guard empty before and after.
   Provenance: analysis run against `HEAD 760b62a` with this doc file as the only modification, the trace being
   the #194 artefact pair under `/tmp/trace194/` (`off.jsonl`, mtime 2026-10-16 07:47, unchanged since capture),
   AC power. Environment at the time the screen ran: battery 100 %, `memory_pressure -Q` 83 % free, swap
   5 434 MiB used, load average 2.35, Swift 6.4 / macOS 27.0.1. Re-verified at 13:05 with the machine busier
   (97 % charging, 72 % free, swap 3 968 MiB, load 4.50): **every number above reproduced identically**, because
   the analysis is a deterministic read of a frozen file — which is the check that makes the load readings
   irrelevant here rather than a protocol deviation. The determinism claim is itself tested, not assumed: the
   shipped screen was re-invoked at 13:07 on `off.jsonl` at `--top-m` 1 and 2 and its JSON output is
   **byte-identical** to the 09:55 artefacts every table above was built from (`md5` of the trace
   `67d221e2546baca4b26d1e66aaa360ae`).
8. **The per-token denominator is 384, inherited with all its error from §5ak deviation 1** — every `reads/tok`,
   `MiB/tok` and `useful/tok` figure here is ~0.26 % high against a per-position reading, and the same convention
   makes the two price models' `B`/`K` internally consistent with each other but 0.26 % off a strict per-token
   truth. It cancels in every comparison inside this section, since all arms share the denominator.

## 5am. #200 tests the constant-feeder premise in its other dimension and closes it on arithmetic rather than price: of the 70 043 expert names that survive one token, **70 043 are already resident and 0 are missed**, so a token-ahead feeder has literally nothing to fetch (2026-10-16)

The operator's brief for this campaign is a *shape* of reading, not a knob: "the SSD can read constantly while a
user has an active session and can keep on reading weights even if compute is doing stuff… instead of a
chunk-by-request reader". Two sections have now priced one dimension of that — **how many reads per link**
(§5ak closed uniform depth, §5al found the layer axis undecided). This section tests the other dimension, which
no section has measured: **how far ahead in time a read can be issued at all.** A feeder that runs ahead in time
is the only version of "constant" that could fill the idle windows, so the question is whether the future is
predictable far enough out to be worth reading for.

It is not, and the reason is stronger than a bad precision number. On the same #194 ring-off trace
(18 384 observations, 383 decode positions 10-392, 48 layers), for each horizon `p` the same-layer top-10 route
was compared between position `t` and `t+p`, and both a persistence ranker (this token's score-ordered route
names the read for `t+p`) and a strictly-past recency ranker were scored depth-1 under the shipped filter-then-cap
policy — the same rule §5ak put in the screen, and the same §5al substrate rule applies: ring-off, or the
counterfactual is scored against demand the shipped prefetcher already ate.

### Route churn between tokens is real — and entirely invisible to the cache

| horizon `p` | links | mean top-10 overlap | per-layer mean overlap min/p25/med/p75/max |
|---|---|---|---|
| 1 | 18 336 | **0.382** | 0.075 / 0.276 / 0.399 / 0.466 / 0.654 |
| 2 | 18 288 | 0.299 | 0.072 / 0.204 / 0.302 / 0.365 / 0.566 |
| 3 | 18 240 | 0.274 | 0.072 / 0.189 / 0.277 / 0.323 / 0.521 |
| 4 | 18 192 | 0.257 | 0.075 / 0.178 / 0.254 / 0.306 / 0.489 |

Only 38 % of the ten names a layer demands at `t+1` were also routed at `t` (the overlap is scored against
`t+1`'s set, which is the one a feeder would have to hit), and the layer axis is there again (some layers repeat
65 % of their picks, one repeats 7.5 %). Now the part that decides the lever — of the names that *do* carry over,
what does the cache still hold at the moment they are demanded:

| horizon `p` | carried names | already resident at `t+p` | in `t+p`'s miss set |
|---|---|---|---|
| 1 | 70 043 | **70 043 (1.000)** | **0 (0.000)** |
| 2 | 54 682 | 53 716 (0.982) | 966 (0.018) |

Not a rate near one — exactly one, with no exception in 70 043 observations, on the *pre-plan* residency capture
([RealForwardRunner+Diagnostics.swift:151-153](../sources/TinyTitan/Runtime/Inference/RealForwardRunner+Diagnostics.swift#L151):
"`resident` is captured before cache planning so a later miss reservation cannot make the trace falsely report an
expert as absent"). The complement carries the lesson: **every demand miss at `t+1` is a name that was not routed
at `t`.** So a one-token-ahead feeder would issue reads only for experts that are already hits, and would need to
name the misses out of a set that does not contain them. There is no price model, no adoption discount and no
noise floor that turns that into traffic worth paying for — it is the rare case in this campaign where the screen
settles a lever by itself.

### What the two past-visible rankers get, and what the shipped link gets instead

| horizon `p` | persistence slot-1 issues / useful / precision | persistence slot-2 marginal | recency slot-1 precision |
|---|---|---|---|
| 1 | **0 / 0 / —** (nothing non-resident to stage) | — | 0.0267 |
| 2 | 7 355 / 564 / **0.0767** | 0.0638 | 0.0268 |
| 3 | 8 855 / 607 / 0.0685 | 0.0591 | 0.0308 |
| 4 | 9 391 / 536 / 0.0571 | 0.0521 | 0.0382 |

Against the shipped link — **same token, one layer ahead, `recorded` ranker: 15 257 issues, 7 968 useful,
precision 0.5223, recall 0.2486** (§5ak) — the best cross-token arm is 6.8x worse at depth 1 and reaches 564
useful reads out of 33 709 demand misses in the trace (**1.7 % recall**) while the shipped ring reaches 24.9 %.
The asymmetry explains itself: the pipeline inside one token gives the ring a *causal* input (layer L's route is
known while layer L computes, and layer L+1's demand is one kernel away), whereas a token-ahead feeder's input is
a statistical guess about a route that has not been drawn yet. Depth-1 lookahead in **layers** is the only place
where this model's future is readable.

### The axis this trace cannot answer, named so the next capture fixes it

"Constant while compute runs" has a third reading that is *not* closed here: not more reads per link (§5ak/§5al)
and not future tokens (closed above), but **more slack within the token** — issuing for layer L+2 while layer L
computes, so a read has two layers of compute to land in instead of one. The trace has a field for exactly that,
`next2_layer_prediction`, and in this capture it is **empty in all 18 384 rows**. So the slack axis is unscreened,
and the cheapest untried experiment in the campaign is the one that would screen it: a re-capture with that field
populated (the MTP work in §5ai reads it, so it is writable), which costs one model run and no code.

Deviations.

1. **The overlap and precision figures are route-set arithmetic on one capture** — one prompt shape, 383 decode
   positions, `--ram 12`, 96 slots, no clocks and no model process. The 1.000 residency result is a joint property
   of *this* slot budget and *this* eviction order, not of the route. Measured on the same file: each layer's own
   resident view averages 91.9-95.9 of the 96 slots while admitting only **1.78 demand misses per scored link**
   (32 046 over 18 001; 1.83 per observation over all 18 384), so a token's arrivals are about 2 % of what the
   layer already holds, and the order deciding who is dropped is LFU by accumulated use count
   ([ExpertCacheTypes.swift:176](../sources/TinyTitan/Infrastructure/Streaming/ExpertCacheTypes.swift#L176)) —
   LRU is the configured alternative and was not run, so neither order has been varied here. A tighter budget,
   where those two arrivals are a larger share, is what would break the 1.000; §5r-§5s's budget work is where that
   trade lives. The smallest resident set anywhere in the capture is 28, at positions 10-12 — the cold-start ramp,
   not steady state.
2. **"Every miss at `t+1` was not routed at `t`" is a statement about the route, not about predictors.** A
   transition-table ranker can still name some of those misses — that is §5aj's `transition` arm, which does it at
   horizon 0 (805 issues, 0.6385 precision) and is what #196's leakage guard exists to keep honest. This section
   closes the *persistence* route to them, not every conceivable ranking.
3. **Recency and persistence are ranked here from trace-visible past state only** (`t`'s route, and the strictly
   past window of 8 for recency), matching the shipped screen's rule. Both are scored against `t+p`'s own
   `resident` and `misses`, which are the ring-off cache's — a token-ahead feeder would have changed that state by
   now. The first-order effect is nil at `p=1` because that arm issues zero reads; at `p≥2` it is unmodelled, and
   none of those arms is anywhere near worth modelling.
4. **Two different miss denominators appear in this section and both are right**: 33 709 is the sum over all
   18 384 observations, while §5ak's 32 046 counts only the 18 001 scored `L -> L+1` links (layer 47's demand is
   never a scored target). The recall figures here use the first, §5ak's uses the second.
5. **The `next2_layer_prediction` emptiness is a property of the capture, not of the engine.** Nothing was
   re-run to fill it, because that is a model run and none was available; the claim checked is the count (0 of
   18 384 rows carry a name).
6. **Nothing in `sources/` or `benchmark/` moved.** Docs-only, as in §5af-§5al, so the golden baseline is
   **not checked**. Provenance: `HEAD 3bd3314` with this file the only modification, read at 13:21 on AC power
   (99 % finishing charge), `memory_pressure -Q` 70 % free, swap 3 904 MiB used, load average 3.51, Swift 6.4 /
   macOS 27.0.1, `pgrep` guard empty before and after, no model process at any point. The machine state is
   recorded for completeness, not because it can move a number: every figure here is a deterministic read of the
   same frozen file §5al was built on (`/tmp/trace194/off.jsonl`, md5 `67d221e2546baca4b26d1e66aaa360ae`). The
   arithmetic lives in throwaways under
   `/tmp/horizon/` (`decay.py` for the arms and the mechanism table, `shape.py` and `reshape.py` for the churn
   and resident-view counts in deviation 1), deliberately not added to `benchmark/`: they reuse
   [../benchmark/tinytitan_prefetch_trace_analysis.py](../benchmark/tinytitan_prefetch_trace_analysis.py)'s
   `load_trace` and `analyze` rather than duplicating a second screen, and nothing in them is a shipped arm.

## 5an. #201 asks whether the two-layer-slack axis §5am left unscreened is blocked by predictability: a learned hop-2 ranking holds hop-1 precision (ratio 1.019-1.061), and the capture it called for then shows the engine's real probe does not (2026-10-16)

§5am closed feeding *across tokens* and ended on a sentence that needs qualifying here: "Depth-1 lookahead in
**layers** is the only place where this model's future is readable." What §5am actually measured is that a name
carried from one token to the next is already resident by the time it is demanded. It did not measure whether
layer L+2 is readable *from* layer L — it reported that the trace field which would say so, `next2_layer_prediction`,
is empty in all 18 384 rows of the #194 capture. That is the third reading of the operator's "keep reading weights
even if compute is doing stuff": not more reads per link (§5ak), not future tokens (§5am), but **more slack within
the token** — issue for L+2 while layer L computes, and a read gets two layers of compute to land in instead of one.

The `L -> L+2` link turns out not to need the engine's field at all. It is just the pair of observations at
(`t`, `L`) and (`t`, `L+2`), and the frozen ring-off capture has 17 618 of them against 18 001 hop-1 links — the
same 383 positions, 46 layers instead of 47. So the predictability question is screenable offline, on the
substrate §5al requires (ring off) and the policy §5ak put in the screen (filter-then-cap).

### Three arms, and one of them exists to prove the script leaks nothing

All three rank with the shipped `rank_transition`, on tables keyed by layer, and every link is scored before the
position's pairs are folded in — the shipped screen's anti-leakage order
([../benchmark/tinytitan_prefetch_trace_analysis.py](../benchmark/tinytitan_prefetch_trace_analysis.py#L189)),
reproduced rather than trusted:

- **C** hop-1: `L`'s route ranks names for `L+1` through an `L -> L+1` co-occurrence table. This is the decay
  denominator, and it is also the control — the shipped screen scores the same arm independently.
- **A** hop-2, weakest: `L`'s route ranks names for `L+2` through an `L -> L+2` table. No prediction at all, just
  the same learned co-occurrence one layer further out.
- **B** hop-2 through the probe: the engine's own recorded `L+1` prediction (`source.next_layer_prediction`, the
  list the shipped ring actually reads) is composed with the learned `L+1 -> L+2` table, so each name the real
  probe offers is carried one layer further. Because the layer sweep at a position runs ascending, the table B
  reads has only strictly-past positions in it — the current position's `L+1` pairs are folded in after B scores.

| depth | arm C hop-1 | arm A hop-2 | arm B hop-2 | A/C | B/C | shipped `recorded` | A extrapolated | B extrapolated |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.6385 (805 / 514) | **0.6506** (601 / 391) | 0.5727 (557 / 319) | **1.019** | 0.897 | 0.5223 (15 257 / 7 968) | **0.5321** | 0.4684 |
| 2 | 0.6160 (1 190 / 733) | **0.6534** (854 / 558) | 0.5649 (793 / 448) | **1.061** | 0.917 | 0.4760 (26 594 / 12 660) | **0.5050** | 0.4366 |
| 4 | 0.6078 (1 545 / 939) | **0.6384** (1 048 / 669) | 0.5567 (997 / 555) | **1.050** | 0.916 | 0.4269 (39 274 / 16 768) | 0.4484 | 0.3911 |

Precision is issues / useful under the shipped policy; the last two columns are the ratio applied to the shipped
`recorded` arm, not a measurement of it. The control passes exactly: arm C reproduces the shipped screen's
`transition` row to the count at all three depths (805 / 514 / 0.6385, 1 190 / 733 / 0.6160, 1 545 / 939 / 0.6078),
so this script's table order and its `candidate_limit` behave like the shipped one's.

**One hop out, precision does not decay — it rises slightly and holds.** The ratio is above 1 at every depth, and
the warm subset (positions ≥ 60, away from the cold-start ramp) says the same thing: 0.6535 against C's 0.6425 at
depth 1, 0.6562 against 0.6185 at depth 2, 0.6414 against 0.6110 at depth 4. That is the §5am result in reverse:
**across tokens the future is unreadable, across layers it is nearly as readable as the next layer is.** What does
decay is recall (0.0128 against 0.0160) and raw top-k agreement (0.0684 against 0.0722), which is expected — the
set two layers out is a different set, so a ranker that is precise about it is precise about fewer of the misses.
**What this does not license is the inference one would most want from it** — that the engine's own probe decays the
same way one hop out. That is a claim about a different ranker, and §5ao measures it at 0.808 rather than 1.019.

### The demand pool is not the binding constraint either

| horizon | scored links | target-side demand misses | per link |
|---|---|---|---|
| 1 (`L -> L+1`) | 18 001 | 32 046 | 1.78 |
| 2 (`L -> L+2`) | 17 618 | 30 649 | 1.74 |

And on this substrate every one of those is genuinely available to an early feeder, by structure rather than by
counting: with `TINYTITAN_PREDICTIVE_PREFETCH=0` nothing else can have staged them, so a name in `L+2`'s miss set
is absent from the moment `L` computes until `L+2` demands it. Two honest limits sit next to that. The misses are
counted against the *target's own* pre-plan resident view, so a live arm running *beside* the shipped 1-layer ring
would see its room shrink by whatever the ring already staged — §5ah measured the ring converting 19.25 of ~39.5
reads a token, and that overlap is not modelled here. One statistic that would have looked good in this section was
measured and then dropped: the share of `L+2`'s misses whose expert id also appears in `L`'s resident view, or in
`L+1`'s route. Both are cross-layer comparisons over an id space, and the cache is keyed by (layer, expert), so
"resident at `L`" says nothing about slot `(L+2, id)`; the 0.190 / 0.182 figures that came out of it are properties
of the id union, not of the feeder's room, and they are recorded here only so the next session does not reuse them.

### What the two price models make of it, and what only a capture can settle

§5al left this campaign with two break-even precisions for one extra read per link: **0.4579** on the optimistic
constant (`B = +5.142`, `K = −2.184` per read, CONV 0.9277) and **0.8403** on the pessimistic one. The hop-2
extrapolation sits above the first at depths 1 and 2 (0.5321, 0.5050) and at depth 4 below both (0.4484); arm B
clears the optimistic line only marginally (0.4684). So the screen said what §5al's arm B said: **the
predictability objection is void, and the price question is still open.** The asymmetry that decided §5am was
absent — a feeder issued at layer L has a causal input either way, because layer L+2's router can be run on L's
residual rather than on a route that has not been drawn yet.

What the screen could not supply was the *arm a shipped engine would actually run*: not a learned co-occurrence
table but the second probe's own ranking, and whether that ranking still aims one layer further out. The flag for
it already exists and needs no code —
[RealForwardRunner.swift:156-160](../sources/TinyTitan/Runtime/Inference/RealForwardRunner.swift#L156)
reads `TINYTITAN_PROBE2_TRACE=1`,
[RealForwardRunner+DecodeLayers.swift:179](../sources/TinyTitan/Runtime/Inference/RealForwardRunner+DecodeLayers.swift#L179)
encodes layer L+2's router on `hidden: routedX` (L's residual) into a dedicated buffer, and
[RealForwardRunner+DecodeMoE.swift:80](../sources/TinyTitan/Runtime/Inference/RealForwardRunner+DecodeMoE.swift#L80)
is its **only** consumer: the value reaches the trace line and nowhere else. The #194 capture simply did not set
the flag.

**That run was made in the same session, and it refutes the extrapolation.** The engine's real hop-2 probe scores
0.4223 against the hop-1 probe's 0.5223 — a ratio of 0.808, not 1.019 — and its non-duplicate reads are worth
0.240, below both break-evens. §5an's learned-table arms therefore describe how *a table* behaves one hop out, not
how *the probe* does, and the two disagree in sign. The measurement is §5ao; this section stands as the record of
the screen that came to the wrong conclusion and of the check that caught it.

Deviations.

1. **Both hop-2 arms are learned-table arms on one capture** — one essay prompt, 383 decode positions, `--ram 12`,
   96 slots, ring off, no clocks. Arm A issues 601 reads where the shipped ring issues 15 257, so its precision
   rests on a sample 25x smaller than the number it is being used to discount, and a table arm is only as
   conservative as its sparsity: 601 issues in 17 618 links means it declines to name almost everything.
2. **The extrapolated columns are arithmetic, not measurement, and they were wrong.** They transfer a ratio
   measured between two learned-table arms onto a recorded-probe arm whose ranker is different, which assumes the
   two kinds of ranker decay at the same rate one layer out. §5ao ran the capture that tests the assumption and it
   failed: the measured hop-2 recorded precision is **0.4223**, a ratio of 0.808 to the same probe at hop 1, not
   the 1.019 predicted. The columns above are kept because the mistake is the finding — a learned co-occurrence
   table is not a proxy for the router probe, and the campaign's screens should not treat one as the other.
3. **The control is the reason to believe the order.** Arm C equals the shipped screen's independent `transition`
   row at all three depths to the read, which is what rules out the leakage this script could most easily have
   introduced (a hop-2 table folded in before its own links were scored). No second check on that exists.
4. **Precision is not adoption and not speed.** Nothing in this section measures whether a read issued at `L`
   lands before `L+2` demands it, how much queue it competes with, or what it evicts; §5ah's late pool (7 % of
   discarded reads) is the shape of that question and it was already too small to measure there.
5. **Warm subset uses `WARM_START = 60` positions**, chosen as comfortably past the ramp §5am located at
   positions 10-12; the all/warm pair is reported side by side rather than one being the headline, because the
   difference between them never exceeded 0.006 precision.
6. **Nothing in `sources/` or `benchmark/` moved.** Docs-only, as in §5af-§5am, so the golden baseline is **not
   checked**. Provenance: `HEAD d4189f6` with this file the only modification, `memory_pressure -Q` 72 % free,
   AC power 100 % finishing charge, load average and swap as read at the capture's start below, Swift 6.4 /
   macOS 27.0.1, `pgrep` guard empty before the offline work. Every figure in the tables above is a deterministic
   read of the same frozen file §5al and §5am were built on (`/tmp/trace194/off.jsonl`, md5
   `67d221e2546baca4b26d1e66aaa360ae`). The arithmetic lives in throwaways under `/tmp/slack/` (`hop2.py` for the
   three arms and the extrapolation, `room.py` for the miss-pool counts; its cross-layer "already resident" column
   is the statistic deviation above rejects), deliberately not added to `benchmark/`: they reuse
   [../benchmark/tinytitan_prefetch_trace_analysis.py](../benchmark/tinytitan_prefetch_trace_analysis.py)'s
   `load_trace`, `Link`, `rank_transition`, `stage_filter_first` and `analyze` rather than duplicating a second
   screen, and nothing in them is a shipped arm. The capture this section calls for was run in the same session and
   its figures are §5ao; the machine readings above were taken before it, and none of §5an's numbers is a clock.

## 5ao. #202 runs the capture §5an called for and prices the slack-in-layers axis: the engine's real two-layer-ahead probe keeps 0.808 of the one-layer probe's precision (0.4223 against 0.5223), and once its duplicated reads are removed it is at 0.240 — below both of §5al's break-evens (2026-10-16)

§5an ended on an inference it could not test and told the next reader exactly what would test it. This section is
that run. One model run, no code change: the flag
[RealForwardRunner.swift:156-160](../sources/TinyTitan/Runtime/Inference/RealForwardRunner.swift#L156)
already exists, the L+2 router encode at
[RealForwardRunner+DecodeLayers.swift:179](../sources/TinyTitan/Runtime/Inference/RealForwardRunner+DecodeLayers.swift#L179)
already runs on layer L's residual, and
[RealForwardRunner+DecodeMoE.swift:80](../sources/TinyTitan/Runtime/Inference/RealForwardRunner+DecodeMoE.swift#L80)
is still the only consumer — the value reaches the trace line and nowhere else, so nothing here is a shipped arm
and nothing here needed a rebuild.

```
TINYTITAN_RUNNER_STATS=1 TINYTITAN_PREFETCH_TRACE=/tmp/trace200/probe2.jsonl \
TINYTITAN_PROBE2_TRACE=1 TINYTITAN_PREDICTIVE_PREFETCH=0 \
  .build/release/TinyTitanCLI --model models/qwen3.8-flash-next_125B_A6B_4Bit \
  --prompt "Write a detailed essay about the history of computing." \
  --max-new 384 --temperature 0 --max-context 262144
# exit 0; 18 384 trace rows; md5 caf3128e9ddc7fdd1e0394bc63b20bc9 (14 701 537 bytes)
```

### Two controls, both of them exact

The field is populated where #194's was empty: **17 618 of 18 384 rows carry a full ten-name
`next2_layer_prediction`**, and the 766 that do not are exactly layers 46 and 47 of each of the 383 positions — the
rows with no `L+2` to predict. Two things had to be checked before any of that could be scored.

1. **Does the field name the layer it claims to?** Mean overlap with the actual top-10 route of the layer it should
   describe versus its neighbour: **5.952 of 10 against layer `L+2`'s route, 0.191 of 10 against `L+1`'s.** The
   alignment is the documented one, not an off-by-one artefact of where the readback lands.
2. **Did turning the probe on change the engine the numbers are measured on?** It did not, and this is the check
   that matters most, because every precision figure below is a claim about cache state. Scored per observation
   against the frozen `/tmp/trace194/off.jsonl` (md5 `67d221e2546baca4b26d1e66aaa360ae`), all four traced fields —
   `experts`, `misses`, `resident`, `next_layer_prediction` — are **identical on all 18 384 rows**, and the run's
   own counters repeat the #194 ring-off arm to the read: `expert_hits=150131`, `expert_misses=35901`,
   `expert_evictions=31293`, `expert_reloads=22986`, `expert_read_mib=94801.1`, `hit_fixup_layers=13304`,
   `io_host_waits=13304`, `rdadvise_calls=33518`, `prefetch_issued_per_token=0.00`. Route identity across 18 384
   observations is also the response-identity evidence §5ah got from an md5 of the text; the greedy decode
   followed the same tokens, so the two captures are the same engine run with one more field written.

### The real probe, one hop out

Same file, same `recorded` ranker, same shipped filter-then-cap policy, `analyze` on the hop-1 link and the hand
score below on the hop-2 link — and the hop-1 arm re-scored **on the new file** reproduces §5ak's figures exactly
(15 257 / 7 968 / 0.5223 / 0.2486), which is what lets the two columns be read as one comparison.

| measure | hop 1 (`L -> L+1`) | hop 2 (`L -> L+2`) | ratio |
|---|---|---|---|
| names of the real top-10 route hit, of 10 | 6.709 | 5.952 | 0.887 |
| slot-1 name present in the route | 0.9193 | 0.8677 | 0.944 |
| rank-ordered (positionwise) matches | 0.1575 | 0.1272 | 0.808 |
| issues / useful under the policy, depth 1 | 15 257 / 7 968 | 15 455 / 6 527 | 1.013 / 0.819 |
| **precision** | **0.5223** | **0.4223** | **0.808** |
| recall of the target's demand misses | 0.2486 | 0.2130 | 0.857 |

Depth 2 and 4 behave the same way they do on hop 1, only lower — 27 444 / 10 240 / 0.3731 (recall 0.3341) and
41 610 / 13 560 / 0.3259 (recall 0.4424), against hop-1's 0.4760 and 0.4269; the warm subset (positions ≥ 60) is
0.4096 / 0.3611 / 0.3142. **§5an's extrapolation was wrong in the direction that mattered.** Its learned-table arms
predicted a ratio of 1.019 and a shipped-arm landing at 0.5321; the probe measured 0.808 and 0.4223. The set
accuracy only falls 11 % (6.709 → 5.952) and slot-1 accuracy only 6 % (0.9193 → 0.8677), so the loss is not mainly
in naming the route — it is in naming route members that are *still absent* when demanded, which is the thing a
feeder has to hit. That is where a co-occurrence table and a router probe part company, and it is why the screen's
answer could not be trusted.

### 42 % of the early reads are reads the engine was already going to make

The two probes are not alternatives — a feeder running the hop-2 probe would run *beside* the shipped ring, and the
shipped ring stages for layer `L+2` one layer later. Both arms therefore name experts in the same (layer, expert)
key space, so the overlap is a real duplication count rather than the cross-layer id union §5an had to reject:

| depth | early issues | also issued by the ring | marginal issues | marginal useful | **marginal precision** | ring's own, same target |
|---|---|---|---|---|---|---|
| 1 | 15 455 | 6 443 (0.417) | 9 012 | 2 160 | **0.2397** | 14 874 issues / 7 811 useful |
| 2 | 27 444 | 12 137 (0.442) | 15 307 | 2 522 | 0.1648 | 25 828 / 12 402 |
| 4 | 41 610 | 18 716 (0.450) | 22 894 | 2 546 | 0.1112 | 37 748 / 16 376 |

Read the depth-1 row as the arm a live engine would run: adding the second probe to the shipped ring costs **9 012
extra reads to remove 2 160 more demand misses — precision 0.240** — while raising total coverage from 7 811 to
9 971 useful reads (+27.7 %) on 14 874 to 23 886 issues (+60.6 %). Against §5al's two price models for one extra
read per link, the break-evens are **0.4579** (optimistic `B = +5.142`, `K = −2.184`, CONV 0.9277) and **0.8403**
(pessimistic). **0.240 is below both, and so is the unadjusted 0.4223** — for the first time in this stretch of the
campaign the two models agree on a sign, because the arm is not near either line.

### The other reading of "constant", and what it does not buy

The 6 443 duplicated reads are the version of the operator's idea that this section cannot price as a cost: those
bytes are spent by the shipped engine anyway, and knowing them one layer sooner moves them out of the demand burst
without adding a byte. That is genuinely the "keep reading while compute works" shape. It is also the one the
campaign already priced from the other side: §5ah found **7 %** of the ring's discarded reads were correct-but-late
and called the pool too small to measure, and this run's own `io_hidden_pct` is 99.89 — the demand reads are almost
entirely hidden already, so a read that arrives earlier than an already-hidden read hides nothing. The slack arm
therefore has no traffic argument and no latency argument left in this configuration. What would reopen either is a
**tighter slot budget** (§5r-§5s's axis, where arrivals are a larger share of what a layer holds and reads stop
being hidden) or the **layer-selective depth** §5al left undecided — not a longer horizon.

Deviations.

1. **One capture, one prompt, one configuration** — 383 decode positions, `--ram 12`, 96 slots, ring off, LFU order,
   greedy. The duplication counts and precisions are deterministic reads of that file, and §5al's r = 0.922
   session-stability result is what makes them worth reading as a shape rather than a one-off; they are still one
   prompt's shape.
2. **Marginal precision is a counterfactual against a ring that was not running.** The substrate rule (§5al) forces
   ring off, so the "also issued by the ring" column is what the shipped ring *would* have staged from the same
   pre-plan `resident` snapshot, not what a live co-running pair would each find still absent. A live arm would see
   the earlier read land, and the ring would then skip it — which moves bytes between arms rather than adding them,
   and is exactly why the marginal column, not the total, is the one compared to the break-evens.
3. **The run's clock is not a magnitude and is not used as one.** It read `decode_tok_s=2.636` against the #194
   ring-off arm's 4.496, with `body_ms` 354.7 against 204.2, `wait_ms` 229.7 against 96.1 and the engine's own
   `router_readback_ms` 0.0401 against 0.0194. That is a cross-batch comparison — the paired-benchmarking rule this
   campaign follows forbids reading it as an effect — and the second router dispatch plus a ten-index host readback
   per layer per token is tracing overhead a shipped feeder would not pay, since the prediction would stay on
   device. What can be said from it is only that the capture is not a speed run, which is what §5an predicted
   before making it. Nothing here prices the probe itself; if a future arm needs that, it is a paired A/B.
4. **The two 0.808s in the probe table are different quantities that happen to agree.** Rank-ordered match rate and
   policy precision both fall by that factor; nothing makes one follow from the other, and the coincidence is not
   evidence.
5. **`next2_layer_prediction` is still trace-only.** This section measures what a feeder *would have aimed at*; it
   does not add a feeder. `sources/` did not move, so the golden baseline is **not checked**.
6. **Nothing in `sources/` or `benchmark/` moved.** Docs-only, as in §5af-§5an, so the golden baseline is **not
   checked**. Provenance: `HEAD d4189f6` with this file the only modification, and the tree's sources unchanged
   since `9c858cd` (04:28:32) against a CLI built at 04:29, so the binary is HEAD's sources. Preconditions read
   once immediately before the run: `pgrep` guard empty, `memory_pressure -Q` 72 % free, AC power 100 % finishing
   charge, 283 GiB free on the data volume, Swift 6.4 / macOS 27.0.1 on this campaign's 24 GB M3 MacBook Pro, one
   model process start to finish. The screen arithmetic lives in `/tmp/slack/probe2.py` (controls, both horizons,
   the duplication count) and `/tmp/slack/hop2.py` (§5an's arms), deliberately not added to `benchmark/`: they reuse
   [../benchmark/tinytitan_prefetch_trace_analysis.py](../benchmark/tinytitan_prefetch_trace_analysis.py)'s
   `load_trace`, `Link`, `stage_filter_first`, `analyze` and `Counters` rather than duplicating a second screen, and
   nothing in them is a shipped arm. The capture is `/tmp/trace200/` and, like every other trace in this campaign,
   is not committed.

## 5ap. #202 runs the paired live arm §5al asked for: the gate-0.45 layer list measures −8.4 % over four interleaved rounds, and the counters name the term the screen had no way to price — the 33 extra reads a token do not add adopted reads, they remove 10.5 of them, exactly one for each demand hit the arm loses (2026-10-16)

§5al's closing requirement was quoted by §5ao and is worth restating exactly, because this section is either its
fulfilment or its refusal:

> One live paired arm: **selective depth 2 on a fixed link list versus the shipped uniform depth 1**, interleaved
> A/B/A within rounds, same batch, same control, greedy responses hash-checked for invariance.

Its predicted band was −2.15 % to +6.21 % against a measured uniform-depth result of −6.2 %. Four rounds ran. Every
one of them is below the band's floor.

### The instrument needed one change first

The seam was already in the tree from `dcebe75` — `TINYTITAN_PREFETCH_LAYER_DEPTH`
([../sources/TinyTitan/Runtime/Configuration/PrefetchDepthProfile.swift:17](../sources/TinyTitan/Runtime/Configuration/PrefetchDepthProfile.swift#L17),
read at [../sources/TinyTitan/Runtime/Inference/RealForwardRunner+BuildCore.swift:104](../sources/TinyTitan/Runtime/Inference/RealForwardRunner+BuildCore.swift#L104),
applied as `min(freeSlots, depthByTargetLayer[layer] ?? slots.count)` at
[../sources/TinyTitan/Runtime/Inference/ExpertPrefetchRing.swift:116](../sources/TinyTitan/Runtime/Inference/ExpertPrefetchRing.swift#L116))
— and it is default-off: with the variable unset the parse gets `nil` and every link keeps the shipped depth.

What the sweep could not do was answer the hash half of that sentence. It compared rates without proving the arms
generated the same text, so a rate could have been a rate at different work. `2a06f7b` adds the check: the streamed
completion is reassembled from `delta.content` only — `reasoning_content` is a separate channel and is deliberately
not hashed — at
[../benchmark/tinytitan_knob_sweep.py:314](../benchmark/tinytitan_knob_sweep.py#L314), hashed per arm at `:342`,
and `verdict()` now refuses a page whose arms disagree (`CONTESTED`, status 1) or where an arm streamed nothing
(status 2). 47 tests in [../benchmark/test_knob_sweep_verdict.py](../benchmark/test_knob_sweep_verdict.py),
ruff 0.16.7 clean.

```
SWEEP_TOKENS=384 TINYTITAN_BENCH_RAM_BUDGET=12G \
  python3 benchmark/tinytitan_knob_sweep.py --arms base,layer_depth25,base_again
# arm env: TINYTITAN_PREFETCH_LAYER_DEPTH = the 25-link gate-0.45 list, 119 chars,
#   built at benchmark/tinytitan_knob_sweep.py:70 from §5al's table
# rounds 3-4: exit 0, "response text identical across 3 arms (md5 3c0c3391)"
```

The ring evidence line confirms the arm configured what it claims: `slots=2 links_deep=25 free=1.90` against the
baseline's `slots=1 links_deep=0 free=0.99`, in every round.

### Four interleaved rounds

| round | base tok/s | depth-2-on-25 tok/s | paired delta | drift control | text hash |
|---|---|---|---|---|---|
| 1 | 4.403 | 4.072 | **−7.5 %** | −1.1 % | not taken (pre-`2a06f7b`) |
| 2 | 4.370 | 3.973 | **−9.1 %** | +3.9 % | not taken (pre-`2a06f7b`) |
| 3 | 4.451 | 4.086 | **−8.2 %** | +0.2 % | `3c0c3391`, all three arms |
| 4 | 4.464 | 4.075 | **−8.7 %** | +0.5 % | `3c0c3391`, all three arms |

Mean −8.4 %, spread 1.6 points, four of four negative, and the two hash-verified rounds alone land at −8.2 % and
−8.7 %. The drift control does not explain the ordering: round 2's control got *faster* across the batch (+3.9 %),
which if anything flatters the middle arm, and that round is the most negative. The greedy text is also identical
*between* rounds, so the four rounds are four clocks on one answer rather than four answers.

**§5al's Arm B is refused: the layer gate does not turn depth 2 from a loss into a win.**

### Where the screen was wrong is the useful part

Round 4's per-token counters, the hash-verified one (round 1 agrees to two decimals on every line):

| per token | base | depth 2 on 25 links | delta |
|---|---|---|---|
| prefetch issued | 42.76 | 76.12 | **+33.36** |
| prefetch adopted | 22.80 | 12.35 | **−10.45** |
| adoption ratio | 0.533 | 0.162 | |
| demand hits (384 tok) | 145 313 | 141 298 | −4 015 |
| demand misses (384 tok) | 42 330 | 46 345 | +4 015 |
| demand total | 187 643 | 187 643 | 0 |
| `expert_read_mib` | 111 777.7 | 122 379.8 | +10 602.1 = 27.61/token |
| `io_ms` / `body_ms` | 100.406 / 205.540 | 116.461 / 225.691 | +16.06 / +20.15 |

Read the two bold lines together. 4 015 ÷ 384 = **10.456**, and the adopted column fell by **10.45**. The identity
holds in every round, to the printed precision: round 1 11.5625 against 11.56, round 3 11.1745 against 11.17, and
round 2 7.8411 against 7.84 (scored against its control, for the reason in deviation 2). The demand total is
187 643 in both arms of every round.

So the extra 33 issues per token do not add adopted reads. **They remove them, one for one with the demand hits the
arm loses.** That is §5al deviation 6 answered on the machine it was written about — *"Adoption is still not in any
of this. `useful` has no arrival time … the live ladder's falling hit column with depth is the counter-evidence"* —
and it is why the screen's 0.546 realised precision could never be spent: the screen credits a slot-2 read when its
name is in the route, and charges nothing for the slot-1 read that no longer gets adopted.

### The screen missed in both directions, and the cheap one favours the arm anyway

| quantity | §5al's gate-0.45 screen | measured (rounds 1/3/4) |
|---|---|---|
| extra issues/token | 14.83 | 33.06 / 33.25 / 33.36 |
| extra MiB/token | 39.2 | 30.53 / 29.51 / 27.61 |
| extra useful reads/token | +8.1 | **−11.56 / −11.17 / −10.45** |
| net | −2.15 % … +6.21 % | −7.5 % … −9.1 % |

The arm spends ~30 % *fewer* bytes than forecast while issuing 2.25× the ops: 0.83 MiB moved per extra issue against
the 2.64 MiB a demand miss costs, so most of the extra issues carried little or no read. The likeliest reading is
that a live ring sees its own earlier arrivals in `resident` and drops them, which is exactly what a ring-off trace
cannot show; that sentence is a hypothesis, and the ratio is the measurement. It does not matter for the verdict —
the cheaper-than-predicted traffic still cost 8.4 %, because the loss is not in the bytes, it is in adoption.
Against §5al's two price models, whose break-evens were 0.4579 (optimistic) and 0.8403 (pessimistic), this arm's
whole-config precision is 0.162 and its *marginal* precision is negative. Both models agree on the sign for the
first time in the campaign, and the arm is not near either line.

### What the counters cannot say

`expert_evictions` reads **51 086** and `expert_reloads` **43 662** in every arm of every round, while the hit and
miss columns move by thousands. Reading the increment site explains the shape of the problem: line 75 of
[../sources/TinyTitan/Infrastructure/Streaming/PreadExpertStreamer+Plan.swift:75](../sources/TinyTitan/Infrastructure/Streaming/PreadExpertStreamer+Plan.swift#L75)
ticks an eviction per demand-miss slot the plan reassigns, so the column describes the demand plan rather than the
cache's total pressure — and a speculative arrival that pushed an adopted expert out would not necessarily appear
in it. The consequence for this page is narrow and it is a refusal, not a mechanism: the displacement channel —
slot pressure or arrival lateness — is **not decided by these counters**, and §5ap does not claim it. What is
established is the one-for-one adoption↔hit identity and the 8.4 % that comes with it.

`io_hidden_pct` is deliberately not quoted. §5w established the counter is unfit for an overlap claim, and the
99.7–99.8 readings in these logs would otherwise read as "the extra traffic was free" — which round 4's `body_ms`
(+20.15 ms against a rate-implied +21.4 ms, agreeing to 6 %) contradicts.

### Verdict

**The layer axis is closed by measurement.** §5ak closed uniform depth; this closes the selective reading of it,
so §5al's finding that the *spread* is real (8.1×, r = 0.922, transfers half-to-half at 0.540-0.672) stands as a
description of the routes and fails as a lever: concentrating a bad read does not make it a good read. Nothing
shipped — the seam stays default-off and no shipped configuration sets it, no runtime default moved, and no
numerics changed.

One reopen is left. §5ao named two conditions that could still make an earlier read pay: the layer-selective depth
(now refused) and a **slot budget tight enough that a staged arrival is a larger share of what a layer holds**. The
second is an interaction and neither single-knob arm answers it, so the cell that turns both knobs at once — 64
slots with the same 25 links deep — is added in `5502ccb` as `slots64_depth25` and is measured next.

Deviations.

1. **Rounds 1 and 2 predate the hash instrument.** They are paired, controlled and same-batch, but they do not
   satisfy §5al's invariance clause. Rounds 3 and 4 do, and they carry the conclusion on their own (−8.2 %,
   −8.7 %); the four-round mean is reported with the two unhashed rounds labelled as such.
2. **Round 2's baseline footer is gone.** The harness writes one log path per arm name
   (`.build/benchmark-logs/sweep_<arm>.log`), so round 3 overwrote it before it was copied. Its rate, hit, io and
   waits cells survive in `/tmp/seam202/pass2.log`; its full counters do not, which is why round 2's adoption
   identity is scored against that round's control arm instead of its baseline. Procedural fix adopted and used
   from round 3 on: copy each round's three logs to `/tmp/seam202/pass<N>/` before starting the next.
3. **A partial log was caught before it was read.** The salvaged `pass2/sweep_base.log` holds one footer at 32
   tokens (`decode_tok_s=4.581`, `hit=0.6406`, `misses=6 715`) — a warm-up, not a measurement — because the
   extractor took `rates[-1]` from a file still being written. It is excluded from every table here. The extractor
   bug (silent `rates[-1]` over a log with one footer) is in `/tmp`, not in `benchmark/`, and is not shipped.
4. **One prompt, one session, one 384-token measured request per arm per round**, greedy, `--ram 12`, 96 slots,
   48 layers, top-k 10 of 512, the shipped 96-slot default as the baseline. Cross-round agreement measures this
   machine's noise floor, not prompt portability; §5al deviation 1 still applies to the gate list itself, which was
   fitted on a whole trace.
5. **The 2.25× issue discrepancy is reported, not explained.** The byte/issue ratio that accompanies it is a
   measurement; the residency-filter reading attached to it is a hypothesis and is labelled as one.
6. **`expert_evictions` and `expert_reloads` carry no information for this claim**, for the reason above. They are
   still printed by the sweep's table; a reader should not treat their invariance as evidence that the cache
   behaved identically.
7. **Provenance.** Rounds 1-2 ran at HEAD `b8bb1fa`, rounds 3-4 at HEAD `2a06f7b`; both differ from current `main`
   only by benchmark python rows. `sources/` was unchanged across all four rounds and across the golden run: every
   seam file was last written by 14:23:53, [../sources/TinyTitanLib/ServerModelSession+Diagnostics.swift](../sources/TinyTitanLib/ServerModelSession+Diagnostics.swift)
   and the rest of the tree were untouched after that, the release binaries on disk are 14:37:28 (CLI) and
   14:37:36 (server), and `find sources -newer` against them is empty — so whatever binary each round used, it
   carries those sources. `tools/golden-baseline.sh --check qwen38-4` at 14:31:39 read
   `ok — output identical to baseline` against that same source state. This section moves no sources; the two
   python commits it cites are ruff-clean with 47 tests, so the golden baseline is **not re-run** for them and is
   **not** needed by them. Preconditions read immediately before the rounds: `pgrep` guard empty,
   `memory_pressure -Q` 73-74 % free, AC power 100 % charged, 281-283 GiB free on the data volume, Swift 6.4 /
   macOS 27.0.1 on this campaign's 24 GB M3 MacBook Pro, one model process at a time, `--ram 12` throughout.
   Logs live in `/tmp/seam202/pass<N>/` and, like every other capture in this campaign, are not committed.

