# Host KV with a VRAM window: the design and the wall

**Status (2026-10-21): design; not implemented.** This is the third piece of
the Project Maya transfer (see `docs/project-maya-scan-2026-10-20.md`, item C).
Maya keeps each attention layer's KV in pinned host memory with a window of
positions in VRAM; the Qwen3.8 table in its study is the same model family this
repository runs. This note records what TinyTitan would have to change, why it
is not a one-session change, and what the first test would be.

## What the engine has today

- `KVCacheManager` allocates one MTLBuffer region per layer and slot, at the
  configured precision (4-bit for the 125B install). `encodeFullAttentionDecode`
  reads it at the slot's row for the token's position; prefill writes it through
  its own epilogue (`copyPrefillKVToCache`).
- Qwen3.8's full-attention layers are sparse: the QSA indexer selects a bounded
  set of positions (`keepMask`/`keepIndices`), and the attention kernel reads
  the selected rows from the same buffer.
- The 2M plan (`docs/plan-longrope-2m.md`) already says 1M at 4-bit KV is
  6.3 GB and 2M is 12.3 GB, against a 15.6 GB resident model — the KV, not the
  weights, is what stops a 24 GB host from accepting a long window.
- The prompt cache snapshots and restores KV, but that is a disk tier for
  repeated prompts, not a live window.

## The change

1. **Host arena.** Per (slot, layer), a host allocation of
   `maxContext * bytesPerToken` — about 6.3 GB at 1M/4-bit for the 12 full
   layers. It is the authoritative store; it is not a copy of a full VRAM cache.
2. **Window.** A VRAM buffer of `windowTokens * bytesPerToken` (for example 64K
   tokens ≈ 0.4 GB per layer-slot) holding the rows the next attention call can
   read. The window is filled from the host arena, not re-read from SSD.
3. **Selection-aware copy.** After the indexer names a token's positions, the
   selected 4-row blocks that are not in the window are copied host→VRAM; the
   attention call reads the window. Prompts read a layer's cache through a
   staging copy filled while earlier layers compute (Maya's measured 4-9×
   prefill win at 512K).
4. **Write path.** Every token's new KV row is written to both the host arena
   and the window (or to the window with a lazy write-back). The host write is a
   memcpy on the same die; there is no PCIe hop on Apple Silicon, which is why
   the window is a *working-set* device, not a bandwidth one.
5. **Eviction.** A clock/LRU over the window's blocks; the selected set is
   usually small (QSA top-k), so the window can hold far more than one token's
   selection and re-use it across steps.

## What it buys on this hardware

- A 1M context at 4-bit KV stops competing with the expert cache for Metal's
  working set: 6.3 GB moves to host memory, and the window (0.4-1 GB) is the
  only VRAM cost. The 2M plan's arithmetic is the same, one tier down.
- Decode pays the selected-block copies per layer per token; on Apple Silicon
  those are memory-fabric reads, not PCIe, so the cost is bounded by the
  selection size (QSA top-k), not the context length.
- Prefill with a chunk larger than the window pays a staged copy per layer;
  that is the 4-9× prompt case Maya measured, and it needs the staging buffer
  to be filled while earlier layers run.

## Why it is not in this pass

The attention kernel's `keepMask` is a mask over the existing buffer, not an
indirection through a window. A selected-block copy needs either

- a kernel variant that reads a compacted window plus a block map, or
- a position remap in the mask and a window that is a dense copy of the
  selected rows,

and neither is a local change: `encodeFullAttentionDecode`, the QSA decode and
prefill paths, `KVCacheManager`'s region layout, and the snapshot/restore code
all assume one contiguous per-slot buffer. Estimating from the existing split
work, it is a multi-week change with a new kernel, a new cache layout and a
new test matrix (short-context parity, 128K/1M start, decode and prefill
timings). Shipping a half-integrated KV tier would risk silent wrong answers at
long context, which is the one class of bug this project refuses to trade for a
number.

## First test when it is built

1. Short-context parity: the same prompt at 4K gives byte-identical logits to
   the contiguous KV (the existing `QwenRunnerTests` harness).
2. Window boundary: a 128K prompt crosses the window repeatedly; logits match a
   contiguous 128K run.
3. Memory: the process's Metal allocation at 1M is the window, not the full KV;
   the host arena is the difference.
4. Speed: decode at 128K/1M against the contiguous path, and prefill at 512K.
