# Qwen3.8-Flash-Next 4-bit: task-list verification

Status: every optimization from `docs/maya-task-status.md` that can run on the
installed 125B-A6B 4-bit install was exercised on it. One profile value changed
(2,048 prefill chunk); one correctness guard was added (QSA/PLE models serve
one sequence at a time); the rest are rejected or blocked on this model, with
the measurements below.

Host: Apple M3, 24 GB, macOS 27.0.1, model
`models/qwen3.8-flash-next_125B_A6B_4Bit` (96 GB install). Commands and raw
logs are under the git-ignored `.build/` (`qwen38-*`).

## Matrix

| Task / optimization | Qwen3.8 4-bit result |
| --- | --- |
| T1/T2 width-1/2 aggregate | **No win.** At 96 slots the batched path is clamped by `BatchedMemoryBudget` (2 slots → 1 served). At 64 slots a width-2 head-only run produced **different text** from width 1 — QSA/PLE state is per-sequence, not slot-keyed. |
| T5/T6/T34 fused decode | **Rejected on correctness.** At 64 slots the fused width-2 run was admitted by the memory budget and produced different text in every measured row; the fused loop is not verified on Qwen3.8. |
| T15-T18 calibration | **Measured on Qwen3.8 4-bit.** `calibrate --measure --tokens 32 --repeats 2`: baseline 4.12 tok/s; `prefill-chunk-128` +4.0% rejected inside the 4.7% noise band; `cache-slots-96` −3.9% rejected. Profile written `measured: true`, `source: calibrate/1`, no knobs. |
| T20-T24 interior checkpoints | **Now available on Qwen3.8.** `InferenceAuxiliaryState` carries the QSA pooled blocks/raw tail and the PLE convolution window beside the snapshot; a 1,407-token edited last turn went from 44.38 s to 2.76 s (16.1x, identical text hash), with capture 58.64 s against 44.19 s. The turn-cache live tier still hits (`cached_tokens=36`, append 4.93 s vs 5.23 s). See `docs/qwen38-checkpoint-auxiliary.md`. |
| T25-T26 page cache | Policy correctly selects **bounded** (96 GB install against 48 GB threshold). `TINYTITAN_BOUNDED_IO=0` measured a wash (4.69/4.84 vs 4.58/4.96 tok/s), so no override. |
| T27 large-memory | Blocked (hardware), unchanged. |
| T28-T30 MTP | No sidecar for this install; unchanged. |
| T31 gates | Full suite/lint/golden green for the changed profile (see the closeout). |

## Calibration on Qwen3.8 (T16)

    .build/release/TinyTitanCLI calibrate \
        --model models/qwen3.8-flash-next_125B_A6B_4Bit \
        --max-context 8192 --measure --tokens 32 --repeats 2 \
        --out .build/t16-qwen38-calibration.json

    arm prefill-chunk-128: rejected: +4.0% is inside the 4.7% noise band
                           (baseline 4.12, arm 4.28 tok/s)
    arm cache-slots-96:    rejected: -3.9% is not an improvement
                           (baseline 4.12, arm 3.96 tok/s)

The profile is `measured: true` with no knobs. The chunk arm is the useful
part: 128 is the calibration arm's chunk, not the 2,048 default, and it did
not clear the gate. The slots arm duplicates the profile's 96, so its −3.9% is
the run noise being measured against itself.

## The two findings

### Batching is not safe on QSA/PLE

`BatchedMemoryBudget` already clamped width 2 at 96 slots (per-slot 213 MiB,
budget 0 at the 12 GiB cache). Dropping the cache to 64 slots made the width
admit, and then the head-only batched path and the fused path both produced
**different tokens** from width 1. The cause is the QSA indexer/PLE block: the
indexer's pooled keys are per layer, not per slot, and `resetSequence` only
clears slot 0. Rather than ship wrong output, `ServerModelSession.load` now
serves one sequence at a time for `sparseIndexer.enabled || ple.enabled`,
logging:

    TinyTitan batch width 2 is not supported for a QSA/PLE model
    (per-sequence indexer/ngram state is not slot-keyed); serving 1 at once

A re-run after the guard has `text_mismatches: []`; the width-2 requests queue
behind width 1. This is a correctness guard, not a performance claim. Slot-
keying the indexer/PLE is the work that would make batching real on this model.

### The only kept optimization is the 2,048 prefill chunk

See `docs/qwen38-4bit-tuning-2026-10-22.md`: same prefill, 5-16% faster decode
on 200- and 851-token prompts, golden output identical; 4,096 is kept for the
8-bit row and long-prefill operators.

## Rejected measurements, for the record

| arm | measurement | verdict |
| --- | --- | --- |
| 64 cache slots | short raw prompt 4.65 vs 4.13; realistic chat 5.15 vs 5.56; long prompt 3.86 vs 4.49 | keep 96 |
| 112 cache slots | 1.14 / 2.28 tok/s | pages, reject |
| `TINYTITAN_KEEP_WIRED=0` | 5.14/5.26 vs 5.58/6.09 | keep wired |
| `TINYTITAN_HC_FUSED=1` | 4.76/5.99 vs 5.58/6.09 | keep off |
| `TINYTITAN_QSA_GPU_SELECT=1` | 5.87/5.84 vs 5.58/6.09 | wash, keep off |
| `TINYTITAN_EXPERT_USE_CAP=3` | 5.53/5.59 vs 5.92/5.71 | reject |
| `--rdadvise off` | 4.08/4.24 vs 4.58/4.96 | keep advice on |
| `TINYTITAN_BOUNDED_IO=0` | 4.69/4.84 vs 4.58/4.96 | wash, keep bounded |
| `--kv-bits 4` | 6.12/6.08 vs 5.88/5.90 | +3.6% but numerics differ, reject |
| release prefill scratch | 3.82/5.77 vs 6.67/5.30 | loss, reverted |
