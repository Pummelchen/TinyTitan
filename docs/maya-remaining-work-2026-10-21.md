# Maya transfer: remaining work plan

Status: plan only; nothing in this list is implemented by this commit. The
audited current state is in `docs/project-maya-scan-2026-10-20.md`.

This list covers only work that is not already shipped. B (quantised PLE
table and gather), D (persisted expert profile) and F (prompt-progress
streaming) are complete and excluded. The non-transfer list in the scan is
excluded by design, not pending.

Sizes: S <= 1 day, M 2-3 days, L 1-2 weeks, XL multi-week.

## Phase 0 - Measurement foundation

**T1 - Reproducible decode baseline harness.** Fixed two-prompt A/B harness
with per-run nonce, 3 warmup + 5 measured runs, median/spread and output hash;
width-1 baselines at 64/128/256 new tokens. Acceptance: same binary/prompts
across arms, text hash compared, spread reported. Size M.

**T2 - Width-2/4 aggregate benchmark mode.** Extend T1 to two/four concurrent
requests; report wall, per-row latency and aggregate tok/s. Acceptance: B=1
matches T1; B=2/4 selectable without code changes. Depends T1. Size S.

## Phase 1 - A. Fused batched decode

**T3 - New batched decode design note.** B-row pipeline that keeps the decode
prefetch ring and asynchronous routed command across a layer barrier. The
reverted per-row-scratch and verify-pair variants are evidence, not a base.
Acceptance: names every shared/private buffer and the overlap invariant; B=1
identity and rollback defined. Depends T1. Size M.

**T4 - B-row decode scratch and plumbing.** Per-row hidden/normed/QKV/attention/o
buffers and a B-row residual/FFN scratch set. Width-1 allocations unchanged.
Acceptance: width-1 byte-identical; memory scales with B. Depends T3. Size L.

**T5 - Batched QKV/o_proj, per-row attention.** One projection per layer across
rows; per-row epilogue/attention/KV writes; host-KV selection preserved.
Acceptance: toy logits within 1e-3; real text identical at B=2; no slot
cross-talk. Depends T4. Size L.

**T6 - Batched FFN with union fetch and decode-style prefetch.** One union
expert plan/fetch per layer, per-row phase 1/2, pending carried so next-layer
work overlaps; no synchronous per-layer drain. Acceptance: text identical at
B=2; GPU profile shows fetch/compute overlap; no shared scratch or expert-lease
conflict. Depends T4, T5. Size XL.

**T7 - Server/produceBatch wiring at width 2 and 4.** Select fused path for
eligible families, fall back per row otherwise, keep B=1 byte-identical.
Acceptance: server width 2/4 uses the path; ineligible models work. Depends T6.
Size M.

**T8 - Fused-path tests.** Toy B=2/4 parity, slot independence, GDN, host-KV
window, prefill->decode, reset. Assert the fused path ran. Acceptance: width-1
goldens unchanged; no silent fallback. Depends T5, T6. Size M.

**T9 - Real-model text parity.** Ornith 4-bit (and Qwen3.8 4-bit if it fits) at
B=2/4, 64/128 tokens, seeded prompts. Acceptance: byte-identical text; any
divergence blocks T10. Depends T7. Size M.

**T10 - Fused benchmark and ship/revert decision.** 64/128/256 token aggregate
tok/s and wall against serial. Acceptance: durable win, not a short-pair
artifact; otherwise revert and record. Depends T9. Size M.

## Phase 2 - C. Host KV at target scale

**T11 - Real 128K parity.** Streamed vs contiguous at 128K; sampled logits and
greedy continuation. Acceptance: parity at project tolerance or identical
text; window crossings exercised. Depends T1. Size M.

**T12 - Real 1M start and memory.** Start 1M on the 24 GB host; record Metal
window bytes, arena RSS, peak memory. Acceptance: no OOM; Metal cost is the
window. Depends T11. Size L.

**T13 - 512K prefill and 128K/1M decode timings.** Compare to contiguous; report
window-copy share. Acceptance: reproducible under T1; regressions documented.
Depends T12. Size L.

**T14 - Host-KV operating guidance.** Window sizing, selectionWidth + 1 rule,
QSA requirement, host-RAM accounting, benchmark table. Acceptance: docs and
help agree with measurements. Depends T13. Size S.

## Phase 3 - E. Calibration product

**T15 - `tinytitan calibrate` interface and profile schema.** Per (machine,
model, context) profile, schema, CLI, env/flag interaction. Acceptance: schema
round-trips; invalid profiles refused; help lists command. Depends T1. Size M.

**T16 - Candidate arms and 3% keep rule.** Prefill tile depth, q8 attention,
cache slots, reader threads, bounded I/O on fresh prompts; keep only >3% with a
noise guard. Acceptance: deterministic runner; sub-noise wins rejected; profile
produced. Depends T15. Size L.

**T17 - Profile load and precedence.** Apply at startup, explicit flags/env win,
log applied settings. Acceptance: precedence tests; no silent override. Depends
T16. Size M.

**T18 - Calibration tests.** Schema, keep/reject, precedence, corrupt profile,
unsupported machine, no-profile path. Acceptance: tests fail on unpatched code;
no wall-clock luck. Depends T17. Size M.

**T19 - Multi-Mac validation.** Three Macs; win on >=1 knob on >=1 machine.
Acceptance: per-machine table; a single machine reports incomplete, not pass.
Depends T18. Size L. Hardware-gated.

## Phase 4 - G. Interior checkpoints

**T20 - Checkpoint granularity and payload design.** Coarse points (system
prompt end, last-turn start), KV/GDN/logits/cursor payload, eviction limits.
Acceptance: explains the 100K granularity trade; no per-16K default. Depends
T1. Size M.

**T21 - Interior checkpoint format and save/restore.** Extend prompt-cache store
with interior checkpoints keyed by prefix hash; restore into a live cache.
Acceptance: round-trip; no restore on hash mismatch; limits enforced. Depends
T20. Size L.

**T22 - Edited-last-turn resume.** Longest matching interior prefix; no
regeneration of prior turns. Acceptance: end-to-end edited-turn test; fallback
to full prefill when no checkpoint. Depends T21. Size L.

**T23 - Checkpoint correctness tests.** Prefix matching, edited turn, corrupt
payload, stale model/config, limits, concurrency. Acceptance: no wrong-prefix
restore; failures explicit. Depends T22. Size M.

**T24 - 100K checkpoint benchmark.** None / 16K / 32K on a 100K read.
Acceptance: coarse placement measured; keep only if faster for edited-turn
case. Depends T23. Size M.

## Phase 5 - H. Page-cache policy

**T25 - Machine-size-aware page-cache policy.** F_NOCACHE on small hosts; enable
caching when RAM comfortably exceeds install size (proposed 2x); explicit
override. Acceptance: policy table and precedence; no change on 24 GB. Depends
T15. Size M.

**T26 - Policy tests.** Boundaries, override precedence, install-size
calculation, unreadable memory. Acceptance: deterministic; no silent enable.
Depends T25. Size S.

**T27 - Large-memory benchmark.** Caching on/off on a 64/128 GB Mac.
Acceptance: recorded win/loss; unavailable hardware leaves task incomplete.
Depends T26. Size M. Hardware-gated.

## Phase 6 - I. Speculation cost model

**T28 - Distinct-expert growth estimator.** From route traces, predict distinct
experts a draft window would touch, per layer. Acceptance: tested against
recorded traces; error reported. Depends T1. Size M.

**T29 - Speculation controller and gate.** Gate MTP on predicted distinct-expert
growth plus acceptance; default off; enable only on >5% win. Acceptance:
controller tests; off path byte-identical. Depends T28. Size M.

**T30 - MTP controller benchmark.** Controller on/off, Ornith 4-bit, 64/128
tokens. Acceptance: keep only durable >5% win; otherwise record why off.
Depends T29. Size M.

## Phase 7 - Closeout

**T32 - Isolate the real-dimension batch-attention divergence.** The T5
toy-shaped stage and the T6 toy fused path agree with per-row, but on Ornith
4-bit the same fused path produced corrupt text. Add a layer-level real-shape
comparison (batched stage vs the per-row gated path, same hidden/KV inputs) and
diff the activation at each boundary before T6 is re-wired.
Acceptance: the first differing tensor is named, with a minimal reproducer.
Depends T5, T6. Size M.

**T33 - Instrument q/k/attn stages.** Extend T32's layer probe to dump the
per-row q scratch, the K/V rows written to each slot, and the post-gate
attention before o_proj; compare each against the per-row path.
Acceptance: the first wrong tensor is named; no whole-model text run.
Depends T32. Size M.

**T34 - Re-land the fused decode loop around the verified attention stage.**
T32/T33 cleared `encodeBatchGatedAttention` (q, K/V, post-gate attention and
o_proj match the per-row path within fp16 rounding on the real install), so the
corrupt T6 run came from the loop around it. Rebuild `produceBatchFused` with
one change at a time and a layer-level gate before any text run: (1) batched
attention only, per-row FFN and residual add, diff each layer's hidden against
the serial path for two rows; (2) then the verify-pair FFN with the row-hidden
handoff to/from the pair scratch diffed at every boundary; (3) only then the
whole-model text battery. The attention probe must sit directly after
`encodeBatchRowAttention` and before the FFN, on the same hidden the stage
consumed.
Acceptance: layer-level hidden parity, then identical text at 64/128 tokens on
the four-prompt battery; keep only if the wall clock also beats serial.
Depends T33, T6. Size L.

**T31 - Final scorecard, gates, commit.** Record each outcome; run
`swift test --no-parallel`, `tools/lint.sh all`, release build + warning gate;
commit. Acceptance: all green; tree clean; every task done, rejected with
evidence, or hardware-blocked. Depends all shipped tasks. Size S.

## Execution order and gates

T1-T2 first; then T3-T10 (largest value), T11-T14, T15-T19, T20-T24, T25-T27,
T28-T30, T31. Calibration may run in parallel with fused-decode design, but
T1/T2 gate every measurement. T19 and T27 cannot be truthfully completed
without the named hardware. T34 follows T32/T33 and gates the T6 retry; T7/T10
depend on it.
