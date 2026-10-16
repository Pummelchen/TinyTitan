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
