# Host KV with a VRAM window: implementation notes

**Status: implemented.** The design is `docs/plan-host-kv-streaming.md`; this
note records what the tree actually does, how to turn it on, and what the
acceptance tests prove.

## Storage

For each streamed full-attention layer and slot:

- **Host arena.** `KVCacheManager` maps `maxContext * stride` bytes with
  anonymous `mmap` (`MAP_PRIVATE | MAP_ANON`). Pages are faulted only when a
  row is written, so a 1M/4-bit arena is address space first and resident
  memory only as the conversation grows. It is the authoritative store; no
  copy of the full KV lives in Metal.
- **VRAM window.** `kBuffers`/`vBuffers` hold `windowTokens` rows instead of
  the full context. SWA layers keep their ring and linear layers have no rows,
  so only `kind == .full` layers are streamed.
- **Maps.** A per-layer `logical -> windowRow` map (`UInt32.max` = absent) and
  a per-layer `windowRow -> logical` reverse map. A clock cursor allocates
  rows; a dirty bit marks a row not yet written to the arena.
- **Protection.** One selection sweep can assign up to `windowTokens` rows, so
  every row it assigns is protected until the sweep ends. Without that, a
  later allocation could evict an earlier key and leave the kernel reading the
  wrong row.
- **Write-back.** A dirty row is copied to the arena when it is evicted and at
  `advance`, after the step's command buffers have completed.

## Attention

Decode uses a compact selection and an indirection row per key:

- `attention_decode_window_partial` and
  `attention_decode_window_partial_simd` take `sel_index` (logical position)
  and `sel_row` (window row), chunk the selection, and compute the same
  online-softmax recurrence as the contiguous split-KV kernel. The combine
  pass is unchanged.
- The dense path inside QSA's exact window passes no index list; the gather
  expands `0..<visible` and the same kernel runs. A non-QSA model can only use
  the tier while its whole context fits the window; past it the runner throws
  rather than attending to the wrong rows.

Prefill extends the tiled kernels with `useKeep == 3`:

- `keepIdx` remains the logical key (for the causal/visibility check) and a
  parallel `keepRows` buffer holds the window row. Both are per query row.
- The chunk's KV writes go to a per-layer staging buffer sized to the chunk
  (`beginPrefillWrite`); `preparePrefillWindow` marks the union of every row's
  selection, assigns the union once, copies history from the arena on the CPU,
  and queues staging -> window blits on the chunk's command buffer.
- `advance` copies the staged rows to the arena. A snapshot therefore reads
  the arena and never depends on the window's residency.
- The chunk size is capped at `windowTokens / selectionWidth`, the worst-case
  union bound. This is correctness, not tuning: larger chunks would overflow
  the window. Typical QSA selections overlap heavily, so the actual union is
  smaller; the cap is the ceiling.

## Switch

```text
--host-kv-window <tokens>
```

- Off by default; the layout is byte-identical to the contiguous cache.
- Positive token count. The window must exceed the model's selection width by
  at least one row (the in-flight token); `KVCacheManager` refuses a smaller
  one.
- Batched slots get their own window and arena regions, like the contiguous
  layout.
- A bounded server counts both the window and the arena: the per-slot budget
  adds `worstCaseHostArenaBytes` beside the Metal window, so enabling the tier
  does not silently over-commit host RAM.
- `--prefill-chunk` remains the caller's ceiling; the streaming tier only
  lowers the effective chunk.
- ANE prefill is bypassed while the tier is engaged: the sidecar expects the
  contiguous cache.

## Acceptance matrix

The scaled tests run in `swift test --filter HostKVStreamingTests`:

| Test | What it proves |
| --- | --- |
| `shortContextParity` | 4,096 positions, streamed vs contiguous, exact attention. |
| `windowBoundaryMatchesContiguousSelection` | 131,072 positions, a 32-row window, repeated crossing; attention matches contiguous over the same mask, and arena rows match. |
| `memoryIsTheWindowNotTheFullKV` | 1,048,576-token arena vs a 64-row Metal window; the Metal bytes are the window. |
| `copiesStaySelectionBounded` | A repeated selection copies zero bytes; a disjoint selection copies its keys, not the context. |
| `prefillWindowMatchesContiguousSelection` | A staged prefill chunk matches the contiguous tiled selection, and its rows reach the arena. |

The runner-level tests are
`QwenRunnerTests.hostKVWindowMatchesContiguousLogits` (toy, no indexer) and
`QwenRunnerTests.hostKVStreamsQSADecodeAndPrefill` (QSA toy, window 8, asserts
eviction and selection copies happened).

For a real-model timing run, use the serving path with a window and compare
against the same prompt without it:

```bash
swift run -c release tinytitan-server --model <install> --max-context 131072 \
  --kv-bits 4 --host-kv-window 65536 --prefill-chunk 128

swift run -c release tinytitan-server --model <install> --max-context 131072 \
  --kv-bits 4 --prefill-chunk 128
```

The first prompt should be decoded with the same greedy seed; the acceptance
check is logits/text parity at a short prompt plus the decode and prefill
timings. At 1M, the same comparison uses `--max-context 1048576` and a window
of 65,536 rows; at 512K prefill, the prefill chunk is capped internally to the
window/selection bound.

## Limits

- Streaming is for full-attention rows. A dense family with no selection can
  only stream while the visible context fits the window, because it has no
  bounded key set to gather.
- The decode gather needs the selection on the host, so the GPU QSA selection
  is disabled while the tier is on. The CPU selection is the reference path.
- The arena is not counted against Metal's allocation; the window is. A
  session that enables the tier trades VRAM for host address space and resident
  rows, which is the point, and should be sized against the machine's RAM.
- The window is allocated per layer and slot; a very wide batch multiplies the
  window and arena, exactly as the contiguous layout multiplies the full KV.
