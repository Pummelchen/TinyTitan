# What to do next, after the 2026-10-18/19 campaign

Ordered by expected value per unit of work. Every number is measured in this tree;
the evidence is in `docs/perf-campaign-2026-10-18.md`.

## A. Build now — measured headroom, no further justification needed

- **Page cache as an opt-in profile for 8-bit installs.** **+4.47% median over six
  paired rounds, 6/6 positive** — but short-prompt only, and -2.1% at 4-bit, so a
  preset rather than a default. `TINYTITAN_BOUNDED_IO=0` already exists; the work
  is a profile row and the documented caveat.
- **Disk prompt cache** is already the launcher default (13x / 13.8x time-to-first
  token on a resumed session). Left: a size/GC policy check against its 2 GiB
  budget, and one long-session confirmation at 8-bit.

## B. Closed since this list was written

B1 and B2 were superseded by building A1 end to end and running it: the real
answer is a 42% loss, not a 3-5% win (see E).

## C. Scoped projects — known wall or unknown payoff

- **Cross-request prefix sharing.** Blocked precisely: the continuation contract
  requires `0 < cached < prompt length`, because the last prompt token produces
  the first logits. Fix: persist the boundary logits (~0.5 MB) in the snapshot and
  allow a zero-token resume. Prize: ~33 s per duplicate prompt.
- **Cross-request expert warm start** — **implemented 2026-10-20** as a
  persisted routing profile (`docs/expert-profile-warm-start-2026-10-20.md`):
  the decode counts are saved beside the install, seed the cache's LFU score at
  the next start, and are restored at the prefill-to-decode handover. Measured
  +40.6 % decode on a repeated 2.2K prompt, +29 % on a different prompt in the
  same domain, +17 % across domains, with 63-98 % fewer expert reloads. The pin
  file alone measured null; this is the timing argument it lacked.
- **Co-activation disk layout** — predicted low value: the SSD runs at 1.4-2 of
  3.6 GB/s, so the wait is not bandwidth-bound. Only worth it with evidence that
  it is.

## D. Cheap housekeeping

- **A forwarding test for the launcher's prompt-cache flags.** The refusals are
  pinned; forwarding is verified only by dry run.
- **n-gram table compression** — **implemented 2026-10-20** as a 4-bit affine
  table (`tools/quantize_ngram_table.py`, `docs/ple-table-4bit-2026-10-20.md`):
  102.4 GB → 32.0 GB (31.2 % of fp16), install 162 GB → ~92 GB with the source
  removed, decode +3 % at the median (noise) and a byte-identical 32-token
  output. Footprint, not speed, as expected.

## E. Closed — do not spend time here again

- **Compressed expert blobs.** Built, tested and measured: 8.9% fewer expert
  bytes (Ornith 4-bit, 18.12 -> 16.50 GB) but **13.76 -> 7.91 tok/s decode,
  -42.5%** over three paired rounds, identical text. A token misses ~190 MB here
  (127 misses x ~1.5 MB), not the ~65 MiB the estimate assumed, so LZFSE's
  ~0.9 GB/s/core is 211 ms of a 100 ms token. Parallel decode, batched reads,
  parallel ring decode and a depth-8 ring were each measured; none came close.
  `TinyTitanRepack --compress-experts` and the `.idx` read path stay as
  opt-in/reproducible, not recommended.

- More knob sweeps: ~40 configurations, and only the two items in A clear +0.5% on
  both widths.
- Compression outside the expert blobs: the dense backbone is 97.4%.
- Prefetch depth, top-M, margin, LRU, pin files, hop-2 prediction, batching at full
  context, MTP: measured negative or below break-even.
