# Maya work: task status

Source plan: `docs/maya-remaining-work-2026-10-21.md`. This file is the live
status ledger; a task is done only when its acceptance test and benchmark
evidence exist.

| Task | Status | Evidence |
| --- | --- | --- |
| T1 baseline harness | in progress | `benchmark/maya_batch_ab.py`; 64-token run saved in `.build/maya-batch-ab-ornith4-t64.json` (widths 1/2/4, 5 measured + 3 warmup, 0 text mismatches). 128/256 pending. |
| T2 width-2/4 aggregate mode | in progress | same harness; 64-token aggregate: width 1 9.02 tok/s, width 2 9.45, width 4 10.40 (head-only batch, not fused). 128/256 pending. |
| T3 fused decode design | pending | — |
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
