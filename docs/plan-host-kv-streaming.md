# Host KV with a VRAM window: the design and the wall

**Status (2026-10-21): implemented (decode and chunked prefill).** This is the
third piece of the Project Maya transfer (see
`docs/project-maya-scan-2026-10-20.md`, item C). Maya keeps each attention
layer's KV in pinned host memory with a window of positions in VRAM; the
Qwen3.8 table in its study is the same model family this repository runs.
`docs/host-kv-streaming-2026-10-21.md` records the seams, the command line and
the acceptance matrix; this note keeps the original design and now states what
shipped against it.

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

## How it is built

The complete tree changed the two things this note said stood in the way.

- `KVCacheManager` now has an optional host tier (`windowTokens`). For the
  streamed full-attention layers `kBuffers`/`vBuffers` are a window of that
  many rows, the authoritative rows live in a lazily faulted `mmap` arena, and
  the engine keeps a logical-position -> window-row map plus a row -> occupant
  reverse map. SWA layers keep their ring and linear layers keep no rows, so
  their layout is unchanged.
- `attention_decode_window_partial(_simd)` reads a compact selection with a
  row indirection (`sel_row`), so the attention cost is the selection, not the
  context. The contiguous kernel is unchanged when the tier is off.
- The tiled prefill kernels take `useKeep == 3`: `keepIdx` stays logical (for
  the visibility check) and a parallel `keepRows` names the window row. A
  prefill chunk writes into a per-layer staging buffer, and `advance` moves
  the chunk to the arena after the command buffers complete.
- Selection-aware gathering protects every row assigned in one sweep, so a
  later allocation cannot evict an earlier indirection entry. Dirty rows are
  written back on eviction and at `advance`.
- Snapshot/restore reads and writes the arena, not the window.
- `--host-kv-window <tokens>` is the switch. The decode path needs the CPU
  selection (the GPU selection deliberately does not read back), and the
  prefill chunk is capped at `windowTokens / selectionWidth` so the union of a
  chunk's selections fits the window.

The first test when it was designed is below; it is implemented, scaled so it
runs in the ordinary suite, and the exact command for a real 128K/1M run is in
the implementation note.

## Acceptance tests

1. Short-context parity: the same rows at 4K give the same attention as the
   contiguous KV. `HostKVStreamingTests.shortContextParity` runs the same data
   through both layouts at 4,096 positions; `QwenRunnerTests` pins the runner
   path on the toy fixture.
2. Window boundary: `windowBoundaryMatchesContiguousSelection` crosses a
   32-row window at 131,072 positions and compares against contiguous attention
   over the same mask; `QwenRunnerTests.hostKVStreamsQSADecodeAndPrefill`
   exercises the real QSA selection and asserts the window was evicted.
3. Memory: `memoryIsTheWindowNotTheFullKV` compares a 1,048,576-token arena
   against a 64-row window and asserts the Metal cost is the window.
4. Selection-bounded copies: `copiesStaySelectionBounded` asserts a repeated
   selection copies nothing and a disjoint one copies its keys, not the
   context. The large-context timing is a benchmark run, not a unit test; see
   the implementation note for the command.
