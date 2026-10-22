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
| T5 batched QKV/o_proj | done | `encodeBatchGatedAttention` + `BatchAttentionTests`: two rows, per-row KV slots, batch o_proj; output matches the per-row gated path within 1e-3. `batchMatchesPerRowAtDecodePositions` adds width 4 at four different positions against per-slot multi-token KV. The multi-row GEMVs now accumulate scale/bias in the single-row order, so per-row outputs are bit-identical, not just within 1e-3. |
| T6 batched FFN/prefetch | done | The fused loop's FFN is a decode-kernel union stage (one readback, one union plan/fetch, per-row phase 1/2, carried routed command), bit-identical at width 2 and width 4. Prefetch and shared-expert-overlap variants were measured and reverted. |
| T7 server wiring | rejected | Not wired: T10 found no durable win on this host. The path stays opt-in behind `TINYTITAN_FUSED_DECODE=1`. |
| T8 fused tests | done | Toy width-2 and width-4 logit parity, both batch-stage suites, and the 8/8 real-model text battery; the fused loop is asserted to have run. |
| T9 real parity | done | B=2: bit-identical logits and all 8 four-prompt cases (64/128) byte-identical. B=4: bit-identical logits for 4 rows and 4/4 texts identical at 64. Qwen3.8 4-bit does not fit this host. |
| T10 fused benchmark | rejected | No durable win: width 2 is parity at 64/128; width 4 is +9.4% at 64 (59.4 s vs 65.0 s for four prompts) but a loss at 128 (129-131 s for three requests vs 109 s serial for four) under 4.97 GB of swap. Not wired (T7 rejected); the opt-in path stays for a 64 GB retest. |
| T11 128K host-KV parity | blocked (model) | Needs a QSA install to exercise window crossings; the Qwen3.8 4-bit install does not fit this 24 GB host and Ornith has no indexer, where a streamed window at 128K equals the whole context. |
| T12 1M start/memory | blocked (model) | Same gate as T11. The KVCacheManager-level 1M arena/window test exists (`HostKVStreamingTests`). |
| T13 512K/128K/1M timings | blocked (model) | Same gate as T11. |
| T14 host-KV guidance | pending | Depends T13; cannot be written against measurements this host can produce. |
| T15 calibrate interface | done | `CalibrationProfile` (v1 schema, validation, machine identity), `tinytitan calibrate --model/--check/--out`, help listed in `TinyTitanCLI --help`; 10 schema tests + 5 CLI tests. Drafts are `measured: false`, knobs empty; T16 fills them. |
| T16 calibration arms | done | Keep rule, measurement driver and `calibrate --plan` landed. Real Ornith run (32 tokens, 2 repeats): baseline 7.91 tok/s, prefill-chunk-128 +0.5% rejected by the 3% floor, cache-slots-96 +4.1% rejected inside the 6.3% noise band; the produced profile is `measured: true` with no knobs. `docs/calibration-product.md` records the protocol and the arm/switch split; profile at `.build/t16-ornith-calibration.json`. |
| T17 profile precedence | done | `--calibration-profile <path>` on the CLI and server loads with the bounded reader, validates, fills only unset `prefillChunkTokens`/`expertCacheSlots` (explicit flags win) and logs the applied line plus the knobs with no runtime switch. CLI + server precedence tests: fill, explicit-wins, switchless-reported, corrupt-refused. |
| T18 calibration tests | done | Schema round-trip + every refusal (10), keep floor/noise/slower/non-positive/interleaving/merge/refusals (8), CLI precedence fill/explicit/no-profile/corrupt/foreign-machine/wrong-model/wrong-context/switchless (16), server versions of the same (7). Decision tests inject samples, so no test depends on wall-clock luck. Profiles are checked against machine, memory, context and install model before application. |
| T19 multi-Mac validation | pending (hardware) | — |
| T20 checkpoint design | done | `docs/interior-checkpoints-design.md`: coarse points (`systemEnd`, `lastTurnStart`, existing `turnEnd`), payload = `InferenceStateSnapshot` plus prefix tokens and domain/content hashes (no logits for interior), 100K int8 arithmetic (~1.06 GB for system-end + last-turn-start, ~2.15 GB with the turn end, ~3.65 GB for a 16K grid), eviction order and restore gates. Explicitly no per-16K default: message-shaped requests only diverge at turn boundaries. |
| T21 checkpoint format | in progress | `PromptCheckpointStore` + `PromptCheckpointMetadata`/`PromptCheckpointKey`: content-addressed by SHA-256(domain + prefix tokens), kind and conversation key, `InferenceStateSnapshot` payload, position/prefix/key/domain validation, memory+disk LRU with atomic writes and SHA-verified loads, corrupt records deleted. 10 store tests (round-trip, hash sensitivity, refusals, domain mismatch, eviction, disk persistence, corruption). Remaining: capture the points during prefill and restore into a live cache. |
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
| T34 re-land fused decode loop | rejected (wall clock) | Correctness is done: width 2 and width 4 are bit-identical to the per-row decode, with 8/8 and 4/4 text parity. The wall clock is not: width 2 parity, width 4 +9.4% at 64 and a loss at 128 under memory pressure. Per T10 the path is not wired; it stays opt-in for a 64 GB retest. |

Phase 2 (T11-T14) is model/hardware-blocked here: the only QSA install is
Qwen3.8 4-bit, which does not fit 24 GB, and Ornith cannot exercise a streamed
window crossing. Phase 3 (T15) is therefore the next feasible task; T16 follows.

## New tasks found during execution

Add tasks here with the next free T number; keep the main plan in sync.

- T34: re-land the fused decode loop around the verified attention stage (added 2026-10-21 after T32/T33 showed the stage is exact and the T6 corruption was in the loop's wiring).
- T33: split the T32 probe into q-scratch, k/v cache and post-gate attention tensors (added after the layer-3 probe showed the attention output diverges, first index 1269). Outcome: no tensor in that chain diverges; the probe's placement after the FFN made its numbers unusable.
- T32: isolate the real-dimension batch-attention divergence (added 2026-10-21 after the fused Ornith run produced corrupt text). A first pass found and fixed the fused loop using hardcoded position 0 instead of the row positions; the re-run was still corrupt. The wiring is reverted. Outcome: the corruption is real but is not the attention stage.
