# MTP and I/O overlap: the gate result and the budget (Qwen3.8 4-bit, --ram 8)

> **Note on `--ram` provenance, 2026-09-22.** `--ram-budget` (the launcher's
> `--ram`) now names a target for the *whole process*, not the expert cache: the
> cache gets the target minus the resident weights and a measured runtime reserve.
> Every `--ram 8` measurement in this document was taken under the old meaning,
> where 8 GB was 64 slots of cache and ~11.6 GiB of real use; today `--ram 8` is
> a ~32-slot / ~7.8 GiB configuration and `--ram 12` is the flag that reproduces
> the old one. See Lever 10 of
> `benchmark/internal-speeds/v2-qwen38-4bit-telemetry.txt`.

Written 2026-09-21. Two items from the last proposal were tried: the MTP
speculation path (item 1) and more I/O/compute overlap (item 3). Both are closed
by measurement. All runs used `--ram-budget 8G`.

## Item 1 — the MTP path

### The project's own gate could not run

`benchmark/tinytitan_mtp_b3_qualification.py` called
`ph.one_run(args.quant, mtp, tag)` — three positional arguments, the first a
model *key* — while `tinytitan_mtp_phases.one_run` takes
`(target, sidecar, mtp, tag, ram_budget)`. So the Track B3 gate raised a
`TypeError` before it could measure anything.

Fixed in this commit:

- the wrapper now resolves and forwards `target`, `sidecar` and `--ram-budget`
  (and accepts `--target`/`--sidecar` so it can qualify any install, not just the
  Ornith pair its constants name);
- both harnesses gained `--presence-penalty` (default **0**), which Qwen3.8
  requires: the instruct row applies presence 1.5 to a thinking-off request, and
  the decode loop only uses the draft when `RawCompletion.isPureGreedy`
  (temperature 0 **and** presencePenalty 0 **and** repetitionPenalty 1) holds.
  Without pinning 0, both arms measure the scalar decode and the gate is vacuous
  — which is exactly what an earlier run of mine showed (256 scalar passes in
  both arms with the sidecar loaded).

### The gate result

`--quant 4bit --scenario function --ram-budget 8G --blocks 1 --presence-penalty 0`.
The machine was busy (Chrome), so the harness's `--allow-busy-gpu` was needed and
the absolute rates are indicative; both arms ran under the same load, interleaved.

| arm | runs (tok/s) | median |
| --- | --- | ---: |
| scalar | 3.828 / 3.826 | **3.827** |
| MTP, pair verify | 2.165 / 2.185 | **2.175** |

- acceptance **85.7%**, emitted per pass **1.857**
- **DELTA −43.2%**; the gate requires ≥ +10% at acceptance ≥ 0.65 → **FAIL**
- **output identical: YES** — `073c5aeda43630b5` in all four runs

Two things follow. First, the speed verdict is settled by the project's own gate
at high acceptance: −43%, worse than the archived −15%, and the 2026-09-18 phase
attribution already localises the cost (the verify runs through 32-token prefill
kernels at 1.6-1.7x where the model assumes 1.0x, plus ~200 ms per pass of host
and commit time). A decode-native verify would have to bring a pass from ~2.2x a
scalar token below 1.86x *just to break even* — a large rewrite against a gate
that currently fails by 43 points. Second, **the identity half passed on this
scenario**, which refines the earlier prose-prompt divergence (character 553):
that divergence is prompt-specific rather than universal, so it stays an open
correctness question but does not block code-shaped prompts.

## Item 3 — the overlap budget says there is nothing to overlap with

From the shipped `--ram 8` decode logs (256 tokens, ring on):

| quantity | per token |
| --- | ---: |
| token | **216.8 ms** |
| GPU busy | 102.4 ms (occupancy 43.8%) |
| device busy (`io_ms`) | 95.7 ms = 315.0 MiB at **3.29 GB/s** |
| exposed wait | 83.1 ms |
| other inter-kernel gaps | ~31 ms |
| I/O hidden behind compute | 12.6 ms |

Ring off, same conditions: device busy 116.7 ms = 376.6 MiB at 3.23 GB/s, exposed
wait 91.4 ms, hidden 25.3 ms. So the ring cuts device bytes 16% (377 -> 315 MiB
per token) and exposed wait 9%, and **the device runs at its measured saturation
in both arms** (3.2-3.3 GB/s, the knee found independently with `F_NOCACHE`).

The token is therefore `compute + I/O` in series, not `max(compute, I/O)`: 102.4 ms
of GPU work and 95.7 ms of device work with only 12.6 ms hidden. Perfect overlap
would give ~102-130 ms per token, **7.7-9.8 tok/s**.

The only mechanism that can move expert I/O off the critical path is *prediction*,
because a layer's expert identity depends on that same layer's router output. The
prefetch ring is that mechanism and it is at its measured optimum: M=1 wins,
M=2 is −6.6%, the probe-margin gate −4.2%, M=4 −9.8%. Its remaining limit is
prediction quality — 42.8 reads issued per token against 23.0 adopted (54%).

What will not help, each measured: more I/O threads (the device saturates at four
outstanding reads), splitting each miss (p50 1.04 vs 1.10 ms), separate or
throttled prefetch pools (the ring-off arm shows the same 4 ms p50 with *higher*
wait), and reading fewer bytes by cache (64 slots is the measured optimum) or by
precision/expert dropping (excluded by the quality and RAM rules).

## Bottom line

- **MTP**: fails its own gate by 43 points with byte-identical output on the
  tested scenario. Do not build the decode-native verify for this model unless a
  cheaper verify path can be shown to beat a scalar token per pass, which nothing
  measured suggests.
- **Overlap**: the exposed wait is device time for bytes that cannot be known
  earlier, and the device is saturated while it works. Further gain needs a faster
  device or a better predictor, not more engine parallelism.

## Decision, 2026-11-27 — MTP is dropped

MTP drafting is withdrawn from the product, not deferred. The reason is the
engine's own design: routed experts are streamed from SSD, so a speculative
verify pass reads weights that are not resident, and a width-_W_ pass costs more
than the _≤ W_ tokens it accepts can save. That matches what the gate above
measured — 43 points short with byte-identical output — and it is structural,
not a tuning gap: no cheaper verify path can make a streamed weight read free.

What this closes:

- `TT-085` (MTP sidecars as variants) and `TT-061` (wide verify for Qwen3.8) are
  dropped from the tracker, not left blocked; 7.0 spec phase 7 is withdrawn and
  `docs/plan-wide-verify.md` is marked superseded.
- The Qwen/Ornith sidecar installers were already removed (TT-092); Inkling's
  bundled head remains in its plugin as an opt-in that defaults off.

**The code stays.** This is a scope decision, not a removal: the `--mtp-model`
server flag and its plumbing (`SessionBuildOptions.mtpModelDirectory`,
`EngineFactories` sidecar resolution), `StreamingMTP` / wide-verify in the
runtime, `InklingMTP*` and the per-engine MTP kernels all remain in the tree,
tested, and are simply not part of any shipped path. They are deliberately kept
rather than classified as unused code — nothing in the audit should propose
deleting them — so that enabling a draft head later, on hardware or a model
where speculation pays, does not mean rebuilding it.
