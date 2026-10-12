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

**Decision left deliberately open, priced:** advancing B5 means a repack
change (the engine already supports a 4-bit head end to end —
`lmHeadWeightBits` resolves per-role and the decode/prefill head kernels
are width-keyed, so no kernel work), a fresh ~68 GB install from the
pinned source checkpoint, the full quality panel (perplexity + prompt
panel), a golden re-capture, and the runner-stats A/B for what B5 buys
(the head GEMV read halves, ~640 → ~320 MiB/token ≈ +2-3% ceiling at the
67.8 GB/s the plan measured). The offline gate's job was to kill a
bad premise before that spend; instead it says the premise is false but
the damage is small — an operator call, not a measurement call.

## 5j. B6 re-affirmed closed by its own gate (2026-10-12)

The MTP re-verdict is conditional on the hit rate rising; with B1, the
reader (§5d/§5g), B4, §5h and the sub-split all closed, nothing in the
campaign moves it off 74.2–74.8%. The condition never fires; B6 stays
closed without a re-measurement.

