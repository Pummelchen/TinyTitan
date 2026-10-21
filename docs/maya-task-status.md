# Maya work: task status

Source plan: `docs/maya-remaining-work-2026-10-21.md`. This file is the live
status ledger; a task is done only when its acceptance test and benchmark
evidence exist.

| Task | Status | Evidence |
| --- | --- | --- |
| T1 baseline harness | done | `benchmark/maya_batch_ab.py`; Ornith 4-bit widths 1/2/4 at 64/128/256, 5 measured + 3 warmup each, 0 text mismatches. Artifacts in `.build/maya-batch-ab-ornith4-t{64,128,256}.json`. |
| T2 width-2/4 aggregate mode | done | Same harness. Aggregate tok/s (width 1 / 2 / 4): 64 -> 9.02 / 9.45 / 10.40; 128 -> 12.89 / 13.68 / 13.54; 256 -> 14.55 / 15.68 / 14.01. Width 2 wins all three; width 4 loses at 256. This is the existing head-only batch, not fused decode. |
| T3 fused decode design | done | `docs/fused-batched-decode-design.md` (pipeline, buffer ownership, prefetch overlap, B=1 identity, gates). |
| T4 B-row scratch | done | `BatchedDecodeScratch` (T4) with width 2/4 tests; width 1 unchanged, scratch is opt-in. |
| T5 batched QKV/o_proj | done | `encodeBatchGatedAttention` + `BatchAttentionTests`: two rows, per-row KV slots, batch o_proj; output matches the per-row gated path within 1e-3. `batchMatchesPerRowAtDecodePositions` adds width 4 at four different positions against per-slot multi-token KV. Full logits parity remains T8/T9. |
| T6 batched FFN/prefetch | in progress | Toy fused path (batch attention + verify-pair FFN) matched per-row and was measured on Ornith; real text was corrupt, so the path was reverted. Attention is cleared (T32/T33); the corruption is in the fused loop's wiring, not the stage. Re-land under T34. |
| T7 server wiring | pending | Fused path reverted before server landing. |
| T8 fused tests | in progress | Toy stage tests pass, including width 4 at different positions. The fused loop itself still needs the T34 layer-level gate. |
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
| T32 real-shape batch-attention diff | done | The reported diff does not reproduce. An independent layer-3 probe on Ornith 4-bit (width 2/4, positions 100/60 and 100/60/130/20) shows batch attention == per-row: normed 0.0, packed q 2.4e-4, v 0.0, post-gate attn 2.4e-4, o 2.4e-4, K/V slot bytes 0.0. The in-loop probe ran after the fused FFN's hidden handoff, and its `first=1269` is constant across unrelated positions; it was not a stage comparison. See `docs/batched-decode-phase2c-2026-10-21.md` T32/T33. |
| T33 instrument q/k/attn stages | done | No wrong tensor in the q/k/attention/o chain; the stage is exact to fp16 rounding on the real install and is pinned in-tree by `BatchAttentionTests.batchMatchesPerRowAtDecodePositions`. The corrupt Ornith text is in the reverted loop's wiring around the stage (T34). |
| T34 re-land fused decode loop | pending | Rebuild `produceBatchFused` from the verified stage, with a layer-level hidden-parity gate before any text run and the attention probe before the FFN. The corrupt run still reproduces from the preserved 22:56 binary in `.build/t6-fused-repro/`; `TINYTITAN_ACT_DUMP` makes the fused path ineligible, so the gate must be fused-safe. |

## New tasks found during execution

Add tasks here with the next free T number; keep the main plan in sync.

- T34: re-land the fused decode loop around the verified attention stage (added 2026-10-21 after T32/T33 showed the stage is exact and the T6 corruption was in the loop's wiring).
- T33: split the T32 probe into q-scratch, k/v cache and post-gate attention tensors (added after the layer-3 probe showed the attention output diverges, first index 1269). Outcome: no tensor in that chain diverges; the probe's placement after the FFN made its numbers unusable.
- T32: isolate the real-dimension batch-attention divergence (added 2026-10-21 after the fused Ornith run produced corrupt text). A first pass found and fixed the fused loop using hardcoded position 0 instead of the row positions; the re-run was still corrupt. The wiring is reverted. Outcome: the corruption is real but is not the attention stage.
