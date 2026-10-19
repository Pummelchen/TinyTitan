# What to do next, after the 2026-10-18/19 campaign

Ordered by expected value per unit of work. Every number is measured in this tree;
the evidence is in `docs/perf-campaign-2026-10-18.md`.

## A. Build now — measured headroom, no further justification needed

- **Compressed expert blobs (`zstd -1`).** 90.8% ratio at 1,113 MiB/s per core,
  decompression measured free at one and two threads, expected **3-5%** decode.
  Path: per-expert frames, decompress into the slot buffer behind the prefetch
  ring, keep the uncompressed format readable, then A/B at both widths. Cost: a
  17 GB repack plus streamer and format work.
- **Page cache as an opt-in profile for 8-bit installs.** **+4.47% median over six
  paired rounds, 6/6 positive** — but short-prompt only, and -2.1% at 4-bit, so a
  preset rather than a default. `TINYTITAN_BOUNDED_IO=0` already exists; the work
  is a profile row and the documented caveat.
- **Disk prompt cache** is already the launcher default (13x / 13.8x time-to-first
  token on a resumed session). Left: a size/GC policy check against its 2 GiB
  budget, and one long-session confirmation at 8-bit.

## B. One measurement before committing to A1

- **Bursty versus steady decompression.** The churn test that cleared the CPU
  assumption was a steady stream; real decompression is one expert per layer.
  Expected to be gentler (same average bandwidth, idle gaps) but unmeasured.
- **Per-file ratio on the 125B install**, to confirm the 90.8% generalises rather
  than being an Ornith number.

## C. Scoped projects — known wall or unknown payoff

- **Cross-request prefix sharing.** Blocked precisely: the continuation contract
  requires `0 < cached < prompt length`, because the last prompt token produces
  the first logits. Fix: persist the boundary logits (~0.5 MB) in the snapshot and
  allow a zero-token resume. Prize: ~33 s per duplicate prompt.
- **Cross-request expert warm start** — pre-load a predicted hot set when a request
  arrives, overlapping prefill. The pin file alone measured null, so this needs the
  *timing* argument rather than the placement one.
- **Co-activation disk layout** — predicted low value: the SSD runs at 1.4-2 of
  3.6 GB/s, so the wait is not bandwidth-bound. Only worth it with evidence that
  it is.

## D. Cheap housekeeping

- **A forwarding test for the launcher's prompt-cache flags.** The refusals are
  pinned; forwarding is verified only by dry run.
- **n-gram table compression (71-77%, ~25-30 GB)** for the 125B — footprint, not
  speed.

## E. Closed — do not spend time here again

- More knob sweeps: ~40 configurations, and only the two items in A clear +0.5% on
  both widths.
- Compression outside the expert blobs: the dense backbone is 97.4%.
- Prefetch depth, top-M, margin, LRU, pin files, hop-2 prediction, batching at full
  context, MTP: measured negative or below break-even.
