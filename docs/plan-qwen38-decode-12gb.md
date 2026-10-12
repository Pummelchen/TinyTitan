# Qwen3.8-Flash-Next 4-bit decode at `--ram 12G` — test plan

**Status: B2 and B3 closed (2026-10-11).** Stage 0 put flat and heat within
0.2 points of the uniform control at the fixed 12 GB budget: a 16-token
decode window already touches ~3,234 distinct (layer, expert) pairs against
a 3,072-slot cache, so the cache is in constant churn and no per-pair policy
can beat capacity — B1 dead as modeled. B2 closed by arithmetic: the PLE
gather reads 5,120 B/token against 293–368 MB of expert traffic (0.0015%).
B3 measured and accepted: split gate+up/down reads with an early miss fixup
give +2.5% tok/s (4.180 → 4.286 mean over two A/B pairs) with the
hit-rate-equal + `io_hidden_pct`-up falsifier satisfied, golden
byte-identical — below the §5 projection of +0.3–0.5 tok/s, recorded as
measured (`docs/qwen38-decode-12gb-measurements.md` §5c). B4 closed at its
own algebra gate (2026-10-12): on the real install's weights the
hyper-connection read map violates linearity by 19–44% on every arm
probed (fp32 rounding <1e-6), so no fixed `B` exists and `W_eff = W·B` is
not an identity (`docs/qwen38-decode-12gb-measurements.md` §5f). The B3
follow-up gate|up sub-split closed at its probe gate the same day: finer
ranges (0.88 MiB) do not raise the reader's throughput over the as-is
1.76 MiB front at the real ~3-miss parallelism (all arms 3.2–3.4 GB/s,
inside noise — §5g), so the reader is closed in both directions (§5d:
workers; §5g: range size). The
remaining headroom is §6, which is gated on an explicit operator decision
that has
not been given yet. §3 stands only as the recorded design that stage 0
rejected. Written 2026-10-11, after v5.19. This is a plan, not a promise:
nothing here is supported until a measurement says so and the release notes
carry it. The analysis it builds on is §1's measured state; every stage is
gated on the previous stage's number.

**Goal.** Take decode from the measured 4.0–4.7 tok/s (5.8–6.1 warm) at
`--ram 12G` on a 24 GB M3 to 6–7 tok/s, without raising the RAM target.

**The one-line economics.** At 12 GB the token is ≈170 ms: 105 ms GPU busy,
≈55 ms exposed expert I/O, ≈20 ms per-layer CPU turnaround, ≈5 ms other. The
compute floor is 79 ms = 12.6 tok/s, so 7 tok/s (143 ms) needs the exposed I/O
down to ≈25 ms — roughly a third of what it is. At 3.6 GB/s of real SSD
bandwidth that means a hit rate the current 64-slot uniform cache does not
reach: measured 71.7–78.1%, Belady oracle 86.3% *under the same uniform
semantics*, first-touch bound 93.6%.

## Ground rules (apply to every stage)

- **Cache-policy work must stay byte-identical.** Verify with
  `PATH="$HOME/.local/bin:$PATH" tools/golden-baseline.sh --check qwen38-4`
  before and after; the baseline is compared for byte equality, and a
  cache-policy change that alters a byte is a numerics change and stops.
  (qwen38-4 is the only installed golden target here; the other 15 baselines
  are *not checked* and are never fetched — that is stated in the notes, not
  fixed.)
- **Numerics work needs two keys before code: the operator's sign-off and a
  golden re-capture.** §6 stages do not start without both.
- **The falsifier for any performance arm**: `TINYTITAN_RUNNER_STATS` must
  show `io_hidden_pct` **and** throughput rising together. An arm that raises
  one and not the other is a failed arm; the pairing has never been observed
  and treating either alone as a win is the error the prefetch study warned
  about.
- **Model runs need the window**: `pgrep -fl
  'TinyTitanServer|TinyTitanCLI|TinyTitanPackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'`
  empty, `memory_pressure -Q` acceptable, macOS 26+, Xcode 27 / Swift 6.4.
  The operator pauses and restarts their own server; we never start or kill a
  model process. Fleet CLI probes stay pinned at `--peer 127.0.0.1:1`.
- **No model, dataset or dependency is fetched to satisfy a stage.** A stage
  that cannot run reports *not measured*, with the reason.
- **No closed lever re-opens.** §8 lists them; the plan routes around every
  one.
- **Python drivers obey the audit rules**: no import-time work, refuse an
  unreadable number at the read, fail rather than report a skip, count what
  ran before printing a verdict, and every run is reported with its command
  and exit code.
- **Every stage lands test-first** (the TDD skill's red-green cycle) and
  commits straight to `main`, one push per closed stage so CI coverage is
  created and certified on the final head only.

## §1 Measured baseline (the numbers every stage argues against)

Machine: MacBook M3 24 GB, macOS 26+, Xcode 27 / Swift 6.4, bounded pread
reader (F_NOCACHE, 4 threads), SSD wall 3.6 GB/s. Install:
`models/qwen3.8-flash-next_125B_A6B_4Bit` (the only 125B install; golden
target `qwen38-4`).

| Quantity | Value | Source |
| --- | --- | --- |
| tok/s cold / warm | 4.0–4.7 / 5.8–6.1 | server footer, repeated runs |
| token decomposition | 105 GPU + 55 exposed I/O + 20 turnaround + 5 other (ms) | `docs/qwen38-decode-profile-2026-09-05.md` |
| GPU occupancy | 35–37% | same |
| hit rate @ 64 slots / 12G | 71.7–78.1% | `benchmark/expert-cache/slots-qwen38-4bit-20260918T1423.json` |
| SSD reads per token | 293–368 MB | same |
| Belady oracle @ 96 slots | 86.3% | same study |
| first-touch bound | 93.6% | same |
| reload share of misses | 57% | runner stats |
| token-to-token expert reuse | 38% | route traces |
| compute floor | 79 ms = 12.6 tok/s | profile doc |
| overlap ceiling | 65% hit → 7.7 tok/s; 78% → 12.3; 86.5% → 20.1 (perfect overlap) | `docs/qwen38-decode-20tps-concept.md` |
| only 12.6 ms of I/O hidden today | — | same |

Slot mechanics that stage 1 changes: `--ram 12G` → cache budget 12 − resident
floor 3.72 GiB = 8.28 GiB → 64 slots on the ladder
(`RuntimeConfiguration.swift:28-30`); `cachePerSlotBytes = expertStride ×
numLayers` (`sources/TinyTitanLib/ServerModelSession+Loading.swift:126-129`)
— **a slot caches one expert ID across all 48 layers**, and every layer's
streamer is instantiated with the *same* slot count
(`sources/TinyTitan/Runtime/Inference/Model+LayerStreamers.swift:181-188`).
Total cache is 64 × 48 = 3,072 expert-stride regions (7.92 GiB), allocated
uniformly no matter how a layer's routing actually distributes.

## §2 Stage 0 — offline trace simulation (model-free, decides the order)

**Hypothesis.** The uniform per-layer slot semantics (§1) is the binding
constraint, not the 8.28 GiB: a flat per-`(layer, expert)`-pair pool with the
same total bytes, and/or heat-proportional per-layer allocation, reaches a
hit rate the uniform cache cannot. Belady's 86.3% was measured *under* the
uniform semantics, so it is a floor for a different scheme, not a ceiling.

**Instrument.** The route trace already exists:
`TINYTITAN_ROUTE_TRACE=<file>` records `position layer e0 e1 … e9` per decode
layer (`sources/TinyTitan/Runtime/Inference/RealForwardRunner+Diagnostics.swift:131-151`).
One trace over a real workload is the entire input. No model is touched.

**Deliverables.**

1. `benchmark/flat_pair_cache_sim.py` — a pure simulator with three policies
   over the same trace and the same total budget:
   - `uniform` — replays today's semantics exactly (per-layer streamer, 64
     slots each, LFU with count-zeroing at the prefill→decode handoff), as
     the control that must reproduce the measured 71.7–78.1% within a few
     points. If the control does not reproduce, nothing downstream counts.
   - `flat` — one global pool of (layer, expert) pairs, LFU; per-pair
     footprint is one expertStride, budget 3,072 pairs.
   - `heat` — per-layer slot counts allocated ∝ a layer's distinct-expert
     demand from the trace (capped at 512, sum ≤ 3,072), then per-layer LFU
     within the allocation.
   Output: hit rate per policy, per-layer distinct-expert counts and entropy,
   bytes-read per token per policy, and a projection through the measured
   token model (linear in exposed bytes at 3.6 GB/s, GPU 105 ms, turnaround
   20 ms). Every projection is labelled a projection.
2. `benchmark/test_flat_pair_cache_sim.py` — unittest suite for the pure
   functions (LRU/LFU replay on hand-made traces where the answer is
   computable by hand, allocation sum/cap/determinism, malformed-trace
   refusal at the read).

**Acceptance.** The simulator picks the stage-1 design:
- `flat` or `heat` ≥ +7 points of hit rate over the reproduced control
  → stage 1 is worth building, and the better of the two names its shape.
- Both within noise of the control → B1 is dead as modeled; the plan jumps to
  §5 (B2 measurement) and the merge decision in §6 moves forward *only* with
  the operator.
- The projection also decides whether 7 tok/s is reachable by cache alone or
  needs a §6 byte-cut stacked on it.

**How to run.** With the window free and the server down:
`TINYTITAN_ROUTE_TRACE=/tmp/qwen38-route.trace` on one real workload through
a CLI decode run (CLI, not server — the trace flag is process-global and the
CLI is the shorter path), then the simulator offline. The trace is a few MB;
the run is the only model cost.

## §3 Stage 1 — implement the winning cache shape (byte-identical)

Named after stage 0's verdict; the shape below assumes `heat` won, with
`flat` as the documented fallback. Either way the numerics cannot move: the
same bytes land in the same slot layout, only *which* expert occupies a slot
changes.

**Touch points.**

- `sources/TinyTitan/Runtime/Configuration/RuntimeConfiguration.swift` — new
  pure function
  `expertCacheSlotAllocation(layers:expertsPerLayer:totalSlots:perLayerDistinct:) -> [Int]`
  (sum = totalSlots as far as the per-layer cap allows, cap 512, deterministic
  tie-break by layer index, monotone in demand). This is where the ladder and
  `expertCacheSlotsFitting` already live; the total-budget computation is
  untouched.
- `sources/TinyTitan/Runtime/Inference/Model+LayerStreamers.swift:181-188` —
  layer L's streamer gets `allocation[L]` instead of the uniform
  `effectiveSlotCount`. A per-layer heat table reaches the model through one
  new optional parameter on the load path; absent (the default), every value
  is `totalSlots / layers` and behavior is exactly today's.
- `sources/TinyTitanLib/ServerModelSession+Loading.swift` — reads the heat
  table when `TINYTITAN_EXPERT_CACHE_HEAT` names a simulator-produced JSON
  file (per-layer distinct counts from stage 0). No flag, no change: the
  uniform path is the default and the golden check runs on it.
- `sources/TinyTitan/Infrastructure/Streaming/PreadExpertStreamer+Residency.swift:104-108`
  — the LFU warmth change: stop incrementing `expertUseCount` from prefill
  plans and delete the handoff zeroing, so counts reflect decode history and
  persist across requests. The zeroing comment documents why prefill's
  hundreds-of-plans-per-expert had to be flushed; not incrementing during
  prefill removes the cause instead of the symptom. `resetExpertUseCounts()`
  stays (public API) but becomes a no-op with the reason recorded — or is
  deleted with its callers if nothing else names it; decide by grep at
  implementation time. **The existing test
  `resetExpertUseCountsLetsDecodeEvictPrefillLeftovers`
  (`tests/TinyTitan/Infrastructure/Streaming/PreadExpertStreamerTests+CachePlanning.swift:304`)
  encodes the old behavior and will be rewritten with the change — declared
  here, not slipped in: the zeroing existed because prefill inflated the
  counts, and with prefill no longer inflating them the test's premise
  (leftovers that outrank decode) is void.**

**Tests** (all model-free; `tests/` mirrors `sources/`):

- `tests/TinyTitan/Runtime/Configuration/RuntimeConfigurationTests.swift` —
  allocation sum, cap, monotonicity, determinism, and that the absent-heat
  default equals the uniform distribution.
- `tests/TinyTitan/Infrastructure/Streaming/PreadExpertStreamerTests+CachePlanning.swift`
  — per-layer slot counts actually reach the streamer (a streamer built with
  `slotCount: 3` plans 3, not the old 64); LFU counts survive a request
  boundary; a prefill-plan burst does not outrank decode's own use counts.
- Golden gate: byte-identical `--check qwen38-4` on the default path before
  the flag exists and after.

**Measurement.** With the heat file from stage 0 and the window free:
slot A/B through the existing harness
(`tinytitan_slots_ab.py` / `expert_cache_slots.py`) comparing uniform-64
against heat-allocated at the same 8.28 GiB. Acceptance: hit rate ≥ stage 0's
simulated number minus 3 points, *and* warm tok/s ≥ +0.7 over the 5.8–6.1
baseline, *and* the falsifier pairing (`io_hidden_pct` up with it). A miss on
any of the three sends the stage back to the simulator with the measured
trace, not to tuning.

**Commit discipline.** One commit per red-green cycle; the stage closes with
the measurement table committed to
`docs/qwen38-decode-12gb-measurements.md` (created at first measurement).

## §4 Stage 1.5 — B2: attribute the PLE n-gram table's decode-time SSD traffic

Cheap measurement, no code change: `TINYTITAN_DECODE_IO_TRACE` already
records per-decode I/O; run it over the same workload and split the token's
293–368 MB into expert reads vs PLE n-gram table reads. If PLE steals >10 ms
of the exposed window, it joins the overlap budget and the stage-1 projection
gets a correction term; if not, record *not material* with the number and
move on. The decision rule mirrors the audit's: a lever measured dead is
closed with its number, not left open.

## §5 Stage 2 — B3: split the expert blob read (gate+up before down)

The expert blob is one `expertStride` (2.64 MiB) region read whole; the MoE
decode path consumes gate+up (≈2/3) in phase 1 and down (≈1/3) in phase 2.
Reading gate+up first lets phase-1 compute start ≈18 ms of SSD time earlier
per missing expert.

**Touch points.** `ParallelExpertReader` gains a two-range read
(`fetchSplits(offsets:into:)` or equivalent), `PreadExpertStreamer` plans the
split for misses, the Metal staging path keeps its whole-region contract
(documented, not changed — staging is not the default backend). The slot
still publishes only when the whole region landed; residency semantics are
untouched.

**Tests.** `ParallelExpertReaderTests.swift`: split fetch returns the same
bytes as a whole-region fetch on a synthetic file (byte comparison), handles
a short first range, and reports partial progress; streamer test: a miss
still ends `.resident` with identical slot bytes.

**Measurement.** Same A/B harness, hit rate held equal, tok/s and
`io_hidden_pct` up. Expected from geometry: ≈1/3 of the exposed window on
pure-miss tokens, i.e. ≈+0.3–0.5 tok/s at today's hit rate and less after
stage 1 — measured before it is believed.

## §6 Stage 3+ — numerics levers (GATED: operator sign-off + golden re-capture)

None of these start without the operator's explicit yes. Each is listed with
its algebra-check-first rule, because a wrong premise here costs a golden
re-capture.

1. **HC-factored QKV (B4) — CLOSED at the algebra gate (2026-10-12).**
   The premise is false: the mixin is data-dependent by construction
   (`sigmoid(Wup @ silu(Wdown @ normed/4))`, measured mix spread 0.24–0.71
   of its mean across tokens) and the map `normed → blockInput` violates
   linearity by 0.19–0.44 relative on every arm probed (layers 0/1/2/47,
   attn + mlp, and the model-level mixer) against a real-tensor float32
   noise floor <1e-6. No fixed `B` exists; §5f of the measurements doc
   holds the table. No golden re-capture was spent.
2. **Expert merging 512→256–320 in the repack.** Shrinks the streamed corpus
   (68 → ~45 GB) and the per-token miss count together; the only lever that
   makes 7 tok/s comfortable if stage 0 says cache work tops out below it.
   Numerics-changing by construction; needs the merge-distance study before
   the operator can decide.
3. **lm_head 4-bit (B5).** 644 MiB/token of 8-bit logits at 67.8 GB/s;
   4-bit halves the bytes. Quality gate required (perplexity + a fixed prompt
   panel), golden re-capture.
4. **MTP re-verdict (B6)** only after stage 1: the −43% gate failure was
   measured at 78% hit; the verify-pair economics change if the hit rate
   rises. Re-measure, never re-argue.

## §7 Verification matrix

| Change class | Golden | Runner stats | Slot A/B | Sign-off |
| --- | --- | --- | --- | --- |
| Stage 0 simulator | n/a (no engine) | n/a | n/a | no |
| Stage 1 cache shape | byte-identical required | required, falsifier pairing | required | no |
| Stage 1.5 PLE attribution | unchanged | required | no | no |
| Stage 2 split reads | byte-identical required | required | required | no |
| §6 any | **re-capture** | required | required | **required** |

## §8 Closed levers this plan must not re-open

CPU co-execution (−22.6%, bus contention); ANE decode (unreachable; ANE load
during decode grew GPU busy +89.3%); expert compression (GPU unpack floor);
MTP as shipped (−43%); rdadvise (family-dependent); attention kernel tuning
(retired, including the three-times-reverted `attn_norm_qkv` dispatch work —
42.5 ms stands as measured); the 15-arm knob sweep; concurrency-2 at 12 GB
(swap collapse). Any proposal that collides with this list is refused at
triage, with the doc it died in named.
