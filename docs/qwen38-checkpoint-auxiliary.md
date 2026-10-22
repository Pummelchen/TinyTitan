# Qwen3.8-Flash-Next interior checkpoints: auxiliary state

Status: implemented and measured on the installed 125B-A6B 4-bit model.
Interior checkpoints now restore on Qwen3.8 as well as Ornith.

## Why a snapshot alone is not enough

`InferenceStateSnapshot` carries the KV rows and the Gated-DeltaNet state. A
Qwen3.8 prefix also owns two per-sequence subsystems:

- the **QSA indexer**, whose raw keys and pooled blocks decide which keys the
  sparse attention may see. Restoring without them leaves the indexer ranking
  blocks from the previous conversation: fluent, wrong output.
- the **PLE convolution window**, `(kernel - 1) * dilation` rows of n-gram
  embedding history. Replaying the prefix through the PLE block is the whole
  prefill again.

The old code refused restore outright for both. The auxiliary payload carries
them beside the snapshot instead of changing the snapshot format, so every
existing Ornith record and `InferenceStateSnapshot` version stays valid.

## What is stored

`InferenceAuxiliaryState`: a version, the KV position (must equal the
snapshot's), and a little-endian payload.

**QSA:** per layer, the full pooled blocks up to the prefix and the raw keys of
the **last, possibly partial, block only**. Earlier raw keys are not needed: a
block is pooled once, when its last member arrives, and never read raw again.
That makes the QSA payload scale with blocks, not tokens — about 4.3 MiB for the
1,407-token benchmark, rather than the QSA key cache's full size.

**PLE:** the valid `history` rows of `xpad[xpadIndex]` plus the ping-pong index;
the padding rows are scratch and are not stored.

`PromptCheckpointStore` writes this as `auxiliary.bin` beside `state.bin`, with
its own SHA-256, memory/disk accounting and `maximumAuxiliaryBytes` (1 GiB).
The disk metadata gains two optional fields so v1 records without auxiliary
state keep loading; a rewrite with no auxiliary removes the old file. Corrupt
or missing auxiliary data deletes the record rather than restoring it.

Restore order: KV/GDN snapshot first (which moves the cursor to the prefix
length), then QSA/PLE. `restoreInferenceState(snapshot)` without auxiliary still
refuses on a QSA/PLE runner; only the combined call accepts.

## Tests

- Toy QSA runner (`ArchConfig.qwenToyQSA`): prefix snapshot + auxiliary restored
  into a fresh runner, suffix prefill produces exactly the uninterrupted run's
  logits. A bare snapshot still throws and resets.
- Store: auxiliary survives memory and disk round-trips, a corrupt auxiliary
  file deletes the record, and rewriting without auxiliary drops the old file
  and its bytes.
- Real model: the 1,407-token edited-turn run below has the same output hash in
  the no-checkpoint and checkpoint arms (`b344d80e24a3…`).

## Benchmark

`benchmark/maya_checkpoint_ab.py`, Qwen3.8 4-bit, 96 slots, chunk 2,048,
`--max-context 8192`, 1,407 prompt tokens, two measured edited requests, same
edited text.

| arm | capture prefill | edited prefill (median) | edited wall | speedup |
| --- | ---: | ---: | ---: | ---: |
| no checkpoints | 44.19 s | **44.38 s** | 44.62 s | 1.0x |
| system end + last-turn start | 58.64 s | **2.76 s** | 2.94 s | **16.1x** (wall 15.2x) |

The edited request restores 1,385 of 1,407 tokens. Text hashes are identical
across both arms and both runs.

The capture request costs **+14.45 s** against no checkpoints. That is the
first-request price of splitting the prefill at the two boundaries and copying
the snapshots; it is paid once and recovered by a single edit (41.6 s saved).
This is a break-even-on-first-edit feature, not a free one, and it is the honest
reason the interior tier is most valuable for agent retries rather than a
conversation that is never edited.

## What is not done

`ServerPromptStateStore` — the turn-end cache — still stores snapshots without
auxiliary state, so a Qwen3.8 turn-end restore after a process restart (a
non-live hit) still falls back to a full prefill. Interior checkpoints are the
edited-turn path and are complete; carrying the same auxiliary payload into the
turn store is the remaining step for restart-surviving appends.
