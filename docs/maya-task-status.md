# Maya work: task status

Source plan: `docs/maya-remaining-work-2026-10-21.md`. This file is the live
status ledger; a task is done only when its acceptance test and benchmark
evidence exist.

| Task | Status | Evidence |
| --- | --- | --- |
| T1 baseline harness | done | `benchmark/maya_batch_ab.py`; Ornith 4-bit widths 1/2/4 at 64/128/256, 5 measured + 3 warmup each, 0 text mismatches. Artifacts in `.build/maya-batch-ab-ornith4-t{64,128,256}.json`. |
| T2 width-2/4 aggregate mode | done | Same harness. Aggregate tok/s (width 1 / 2 / 4): 64 -> 9.02 / 9.45 / 10.40; 128 -> 12.89 / 13.68 / 13.54; 256 -> 14.55 / 15.68 / 14.01. Width 2 wins all three; width 4 loses at 256. This is the existing head-only batch, not fused decode. |
| T3 fused decode design | done | `docs/fused-batched-decode-design.md` (pipeline, buffer ownership, prefetch overlap, B=1 identity, gates). |
| T4 B-row scratch | pending | — |
| T5 batched QKV/o_proj | pending | — |
| T6 batched FFN/prefetch | pending | — |
| T7 server wiring | pending | — |
| T8 fused tests | pending | — |
| T9 real parity | pending | — |
| T10 fused benchmark | pending | — |
| T11 128K host-KV parity | pending | — |
| T12 1M start/memory | pending | — |
| T13 512K/128K/1M timings | pending | — |
| T14 host-KV guidance | pending | — |
| T15 calibrate interface | pending | — |
| T16 calibration arms | pending | — |
| T17 profile precedence | pending | — |
| T18 calibration tests | pending | — |
| T19 multi-Mac validation | pending (hardware) | — |
| T20 checkpoint design | pending | — |
| T21 checkpoint format | pending | — |
| T22 edited-turn resume | pending | — |
| T23 checkpoint tests | pending | — |
| T24 100K checkpoint benchmark | pending | — |
| T25 page-cache policy | pending | — |
| T26 policy tests | pending | — |
| T27 large-memory benchmark | pending (hardware) | — |
| T28 distinct-expert estimator | pending | — |
| T29 speculation gate | pending | — |
| T30 MTP controller benchmark | pending | — |
| T31 closeout | pending | — |

## New tasks found during execution

Add tasks here with the next free T number; keep the main plan in sync.

- None yet.
