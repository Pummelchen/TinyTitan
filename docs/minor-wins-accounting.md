# Minor wins: what was measured, what was skipped, and what it would have been worth

2026-10-18. Sources: `benchmark/internal-speeds/v2-qwen38-4bit-telemetry.txt`
(Levers 1-11), `docs/qwen38-decode-20tps-concept.md`,
`docs/qwen38-rewrite-verdict.md`, `docs/qwen38-mtp-and-overlap-verdict.md`,
`docs/qwen38-remaining-levers-plan.md`, `.qwen/wiki/Project-Tracker.md`.

**Update, 2026-10-18:** these entries have now been measured on Ornith 35B, see
`docs/minor-wins-measured.md`. `advice-off` and `immediate` do **not** reproduce
as wins (−4.7% and −6.5% at 4-bit), the page cache reproduces as +4.7% but only
at 8-bit where expert traffic is heavy, and early hits / serial fills stay
unrebuilt. The compounding ceiling in this document therefore drops by ~2 points.

## Verdict first

The premise is half right, and the half that is wrong matters.

**Right:** this project did measure several 1-2% changes, call them a wash, and
delete the knobs. Those can never accumulate now, because the code that would
have carried them is gone. That is a real loss and the habit is worth fixing.

**Wrong:** there is no pile of ten silenced 1-2% wins. The record contains
**five** measured-positive-but-unshipped items between +0.4% and +2.2%, plus one
policy refusal at +4-5%. Their arithmetic ceiling is **+11.0%**; a realistic
expectation is **+3-5%**. Getting to +10-20% needs ten independent items that
each survive measurement, and that is not what the history shows.

The larger finding is elsewhere: the two biggest unbuilt items are **not** 1-3%.
A prefill→decode cache warm-up was identified on 2026-09-04 and never built, and
the tracker records a temporal placement opportunity worth +8.4 hit points. Those
dwarf everything in the tables below.

## Why 1-2% is a special band here

The percentage is not the reason these were skipped. The **noise floor** is:

| measurement | value |
| --- | --- |
| quiet-machine floor, 3 baseline runs, bit-identical counters | 0.07 tok/s (**1.7%**) |
| base-arm drift across one session (paired sweeps) | 4.478-4.596 (**2.6%**) |
| shipped arm moving between two rounds | 4.369-4.620 (**5.7%**) |
| same configuration, one session, different hours | 4.37-5.00 (**14%**) |

A +1.15% effect sits *below* the quiet floor, not near it. That is why four
positive paired rounds were not treated as proof — on this host they aren't. The
honest statement is not "we skipped a 1.15% win", it is "1.15% is unmeasurable
here at any sample size we paid for".

## Table 1 — measured positive, not shipped

| item | measured | strength | why it did not ship | recoverable today | compounds? |
| --- | --- | --- | --- | --- | --- |
| Page-cache path (`TINYTITAN_BOUNDED_IO=0`) | **+5.4%** short (3 rounds, 3/3), **+3.9%** long (2/2), identical output | strong | Rejected by decision: it makes the memory footprint undeclared. The reader's own comment prices the default at "15-30% of decode throughput" | **Switch** — knob present | Yes — but it is a **policy** choice, not an oversight |
| Early hits (`TINYTITAN_EARLY_HITS=1`) | **+2.22%** (1/2 rounds) | weak | Mean smaller than the shipped arm's own 5.7% round swing; knob **deleted** from the tree | **Rebuild** — 0 references in `sources/`; needs the phase-1 early path, its kernel selection and pool handle back | Maybe |
| Eviction policy `lru` | **+2.14%** (1/2) | weak | Same; `lru` survives as runtime config but is not the default | **Switch** — `TINYTITAN_EXPERT_CACHE_POLICY` present | Maybe |
| Immediate submission (`EXPERT_IO_SUBMISSION`) | **+1.29%** (3/4) | weak | Contradicted: the v4.2 record has it **slower** on the representative real-model A/B; still flag-selectable | **Switch** — flag present | Maybe, but the sign is workload-dependent |
| Read advice off (`RDADVISE_POLICY=off`) | **+1.15%** (4/4) | medium | Left unchanged. The advice is dead work — it warms a page cache the `F_NOCACHE` reader never reads | **Switch** — knob present | Yes, but it is a **substitute** for the page cache, not additive |
| Serial fills (`PARALLEL_IO=0`) | **+0.4%** (2/3) | very weak | The serial path was deleted as a losing knob — so the shipped path is the one that measured 0.4% *slower* | **Rebuild** — only a comment survives (`PreadExpertStreamer+Plan.swift:160`) | Negligible |

Four of the six are a configuration flip today; two would need deleted code put
back. That split matters more than the percentages: the page-cache item four rows
up is a decision, while "early hits" is a rebuild for a 2.22% arm that was 1/2
positive — which is why it is not worth doing on its own.

Two of these six are inverted or contradictory rather than simply skipped: the
serial-fill path was deleted while measuring slightly faster, and immediate
submission reverses sign between a short prompt and the representative run.

## Table 2 — measured negative (these were not skipped wins)

| item | result |
| --- | --- |
| MTP speculation | **-43.2%** at 85.7% acceptance; project's own gate requires +10% |
| 96 expert slots | **-10.8%** (11 samples) and **-14%** (3 rounds), despite hit rate 0.757→0.827 |
| Layer-major prefill | reads **-92%**, wall **2.7x slower**; deleted from the engine |
| CPU expert co-execution | **-22.6%** |
| Expert I/O `sync=event` | **-11.9%**, triples `wait_ms` |
| Prefetch ahead 2 / top-M 2 / margin 0.03 / per-expert | **-2.8% / -6.6% / -4.2% / -2.0 to -3.3%** |
| Attention/tail command-buffer merge | **-1.8%**, reverted (removing the buffer removed the overlap) |
| HC fusion, QSA GPU select | **wash** (bit-exact but moves no bytes) |
| GDN in-proj layouts | +0.3% / +1.6% / -1.0% against a **3.2% drift** |

This is the bulk of the history, and it is the reason the campaign closed: most
candidates were not small, they were negative.

## Table 3 — identified, never built

| item | the number on it | status |
| --- | --- | --- |
| Prefill→decode cache warm-up | **premise withdrawn** — see below | The 0.52 was the runner's *cumulative* statistic with prefill's tile misses mixed in. Measured decode-only on a 3.7k prompt, the first 64 decode tokens hit 68% / 82% / 87% at tokens 0-16 / 16-32 / 32-64 **with and without** the use-count reset, identical to three decimals (`RealForwardRunner+Prefill.swift:16-22`). The cache is not cold after prefill, the leftovers decode reuses outweigh what it evicts, and keeping them is the shipped default |
| Temporal placement feeder | Belady over the same 4,608 slots reaches **+8.4 hit points** | Open question; needs a feeder. The only remaining structural item on this list |
| Static per-layer slot partition | capped at **~+1 hit point** against a **+7** bar | Rejected on its own bar, correctly |

## The arithmetic, and why it is not 10-20%

Ten independent 1.5% wins compound to **+16.1%**; ten at 2.0% to **+21.9%**. The
arithmetic in the question is right. The inventory is what does not support it.

From the five small items in Table 1:

| scenario | result |
| --- | ---: |
| arithmetic sum | +7.2% |
| multiplicative, all five | +7.4% |
| plus the page-cache trade (dropping advice-off, which it replaces) | **+11.0%** |
| keep half of every effect (the noise hypothesis) | +5.4% |
| keep a quarter | +2.7% |

So the honest range is **+3% to +11%**, and +11% requires believing two arms
that were 1/2 positive and one that reverses sign on a different workload. The
+16% and +22% figures are what you get if the *number* of items were ten; the
record has five, and one of those five is a policy refusal rather than a miss.

## What is actually worth changing

1. **Measure the stack, not the items.** This is the real lesson. A 1% effect is
   below this host's floor, so it can never be accepted individually — but five
   of them installed together are a 5% effect, which *is* measurable. The
   project tested each knob alone against a 1.7-5.7% floor, which is a design
   that guarantees every small true win is rejected. Stack the plausible ones
   behind one flag, measure the stack, and only then bisect what to keep.
2. **Accept on counters, not wall clock.** Every one of these changes leaves a
   counter signature (`expert_read_mib`, `hit_rate`, `io_ms`, `rdadvise_ms`). A
   stack that moves counters the right way and wall clock by an unmeasurable
   amount is still evidence; wall clock alone at this noise level is not.
3. **Decide the page cache deliberately.** +4-5% is the largest single item in
   the record and it is refused for a footprint reason. If the RAM rule can be
   restated as "declared and enforced" rather than "bounded at any cost", this
   is the cheapest 4-5% on the machine.
4. **Build the warm-up.** It is the biggest unbuilt item, it was identified
   six weeks ago, and the counter evidence (0.52 vs 0.82 hit rate over the
   first 64 tokens) is not ambiguous in the way a 1% wall-clock claim is.

## Caveats

Every delta above was measured on its own build, prompt and machine state. They
are **not** a controlled stack: several touch the same mechanism (ring, slot
table, page cache), at least two are substitutes rather than additions, and none
was re-measured after later changes landed. The compounding table is arithmetic
over those numbers, not a forecast, and the +11.0% row is the ceiling of a
hypothesis rather than an expectation.
