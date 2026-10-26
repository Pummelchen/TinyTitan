# Plan: K2-Horizon runner phase (batched prefill, ANE)

Status: **phase 1 pays on 4-bit.** Every layer has a chunked form —
`K2DenseBatch` for the dense prefix, `K2SparseBatch` for the sparse MoVA
layers — and `K2GraphProbe.forwardChunk` drives the whole graph layer-major
over a chunk. The sparse batch fetches the union of a chunk's selected experts
once per layer (495 fetches for a 163-token prompt against the token path's
7,335), and with that a 163-token prompt measured a median **17.5 s against
23.8 s** token-by-token over eight paired rounds (best 13.4 against 22.5), the
greedy stream byte-identical to the token path. The runtime therefore defaults
chunk 16 on 4-bit installs; 8-bit is faster too (77.9 s against 107-134 s) but
its greedy output changes, because the affine GEMM's reduction order is not the
GEMV's, so it stays on the token path until that is fixed. The family still has
no ANE sidecar.

## Progress

- **Affine GEMM** (done) — `affine_quant_gemm_simd` in
  `Metal/Quant/dequant_affine.metal`, bound by `Kernels/Quant/AffineQuantGEMM`.
  One simdgroup per row, the row's words/scales/biases in registers, a token
  loop, output token-major `[tokens][rows]`. 4- and 8-bit; the lane mapping is
  the coalesced GEMV's, so a chunk of one token equals the GEMV result and the
  tests compare a chunk against N GEMV calls and against a dequantized dot
  product. The shape guard is strict: a row must have a whole number of
  lane-group iterations (for 4-bit, `columns/64` divisible by 4) and at most
  32 of them — a half-covered row would otherwise dispatch and write nothing.
- **Batched dense layer** (done, called by the chunk entry) — `K2DenseBatch`
  runs one dense layer over a chunk: the grouped norm (the kernel already takes a token
  count), the four attention and three MLP projections through the GEMM with a
  per-token GEMV fallback for shapes that do not tile, RoPE per token (its
  scalar position cannot advance across a chunk), causal attention per token
  with buffer offsets, and flat elementwise stages over the chunk. One blit
  writes the chunk's K/V rows into consecutive cache positions.
  `K2DenseBatchTests.chunkMatchesTheTokenPathAcrossTwoChunks` compares it with
  `K2DenseLayer` for positions 0–3 and 4–7 on the sparse toy and they agree.
  The batch takes the decoder layer's `kCache`/`vCache` when a caller passes
  them, so a chunk prefill and a later token decode read the same rows;
  `chunkThenTokenSharesTheCache` writes 0–3 through the batch, decodes 4
  through the layer, and matches a token-only run over all five positions.
- **Chunked graph entry** (done) — `K2GraphProbe.forwardChunk` embeds a chunk,
  runs every layer layer-major (dense through `K2DenseBatch`, sparse through
  `K2SparseBatch`, both sharing the decode layers' caches) and returns the last
  token's logits. `chunkPrefillMatchesTokenByToken` checks a four-token chunk
  and two two-token chunks against four `forward` calls on the toy. Rows stay
  on the GPU between layers: the sparse path reads its input straight out of
  the chunk buffer and blits its output into the alternate one, so a chunk
  costs no CPU `[Float16]` conversions (the CPU-row version measured 36.46 s on
  a 163-token prompt against 31.84 s token-by-token).
- **Batched sparse layer** (done, default on 4-bit) — `K2SparseBatch` runs one
  sparse layer over a chunk: grouped norm, q/k/gate and o projections, the
  softplus gate, both residuals and the post-attention norm are batched
  (through the affine GEMM when the shape tiles, per-token GEMV otherwise),
  while RoPE, causal attention, the MoVA router/values and MoE routing stay per
  token. `K2SparseBatchTests.chunkMatchesTheTokenPath` compares it element-wise
  with `K2SparseLayerProbe` on the sparse toy.
- **Expert fetch: union per chunk, awaited per group** — the layer's expert
  cache is a ring of 64 (4-bit) or 16 (8-bit) slots that a fetch reuses, so a
  fetch must never run ahead of a dispatch still reading a slot. An earlier
  version committed all sixteen expert buffers and awaited them at the end; the
  same chunk size then produced different tokens from run to run at 8-bit
  (measured: two different outputs for chunk 16, two more for chunk 8). The
  phase now groups consecutive tokens while the union of their selected ids
  fits the ring, fetches the union once, and dispatches the group's tokens
  against it in one command buffer before the next fetch. On the 163-token
  prompt that is 495 fetches against 7,335 for the token path — one per layer
  per chunk, because sixteen tokens' top-8 union fits the 64-slot 4-bit cache
  (a four-token group needs at most 32 slots at either width).
  Measurements, 163-token prompt, greedy, paired:

  | install | token path | chunk 16 |
  | --- | ---: | ---: |
  | 4-bit | 23.8 s median (22.5-35.1, 8 rounds) | 17.5 s median (13.4-30.0) |
  | 8-bit | 107-134 s | 77.9 s |

  4-bit is **1.36x** by median (1.68x best against best) and its 24-token
  greedy stream is byte-identical to the token path at chunk 8 and 16 (the
  96-token stream at 16 matched too), so chunk 16 is its default. 8-bit is
  1.4-1.7x, but its greedy output on the golden prompt flips three tokens: the
  affine GEMM partitions a 64-element group across lanes differently from the
  GEMV (4 elements per lane at 8-bit against 2), so the reduction order — and
  the logits at close calls — differ. A speed change may not buy a different
  answer, so 8-bit stays on the token path.
- **Next**: make the 8-bit chunk legal. A GEMM whose per-token accumulation
  mirrors the GEMV's lane-for-lane (one 64-element group across all 32 lanes,
  two elements each, the row's words held in registers and scales/biases read
  per token) would give it the same reduction order, hence bit-identical
  logits, and retire the width caveat. The chunk still spends most of its time
  streaming each token's routed experts; the next fetch-amortisation step is
  batching the routed experts' GEMM across the tokens that selected the same
  expert, so an expert's weights are read once per group instead of once per
  token.

## What exists today

| Piece | Where | Evidence |
| --- | --- | --- |
| Install format, identity, structural validation | `tools/prepare_k2_horizon.py`, `tools/fuse_k2_experts.py`, repacker | both installs verify; `models/k2-horizon-{4,8}bit.ssdai` |
| GPU graph (dense + sparse MoVA layers, streamed routed experts) | `Runtime/Inference/K2*`, `Kernels/MoE/K2*`, `Metal/MoE/moe.metal` | toy tests + real 36B output |
| Shared generation runtime | `TinyTitanLib/K2ProbeRuntime.swift` | server, CLI and bench all drive it |
| Server (single model and catalogue), streaming, tools, stops, Responses, `json_schema`, token counting | `TinyTitanServer/Core/K2ProbeBackend.swift`, `ModelCatalog.swift`, `ModelRouter.swift` | `tools/k2-serve-smoke.sh` (seven checks) |
| CLI | `TinyTitanCLI/RunK2.swift` | raw and chat prompts, timing footer |
| Decoder / tokenizer surfaces | `Tokenization/*` | K2 dialect, multi-turn history, wrapped tool calls, text-spelled think tags |
| Goldens | `benchmark/golden/k2-horizon-{4,8}bit.txt`, `tools/golden-baseline.sh k2-4/k2-8`, `tools/release.sh` | release-gated |
| Invariants | `K2GraphProbeTests.continuingAfterAPrefixMatchesOneFullPass`, KV-reuse and residency tests | CI |

Measured on this host (base M3 MacBook Pro, 24 GB):

- `TinyTitanBench k2gpu`, 4-bit, warm expert cache: 10–13 tok/s decode.
- Served K2, warm: ~6.1 tok/s (32 tokens in 8.36 s on the second request).
- Served K2, cold: a 173-token prompt took 38 s before the first token, 30 s
  with a 64-expert warm start; short prompts taken 13–15 s.
- Prefill is ~0.2 s/token because every token re-reads the active weights and
  re-waits on expert fetches.

## Why the probe cannot just batch

Every kernel the probe uses is shaped for one token: `AffineQuantGEMV` reads a
weight matrix once per token, the attention kernel takes `seqLen` keys for one
query, MoVA/GEMV runs four value experts for one hidden vector, and the MoE
pass reads each selected expert once per token. Batching is therefore new
kernel work, not a loop change — the same conclusion the Qwen fused-expert
retrofit reached from the other side (those kernels stage 16 rows of a
512-wide expert per threadgroup and were slower on K2's 768×2560 shape).

## Phase 1: batched prefill

Process the prompt in chunks (start at 16–64 tokens) **layer-major**: all
tokens through layer 0, then layer 1, and so on. The K/V cache is written once
per chunk per layer, and each layer's weights are read once per chunk instead
of once per token.

### Kernels

1. **Affine GEMM (4/8-bit), the foundation.** One simdgroup per output row,
   the row's packed weights held in registers (K = 2560 4-bit is 320 words, ten
   per lane), then a loop over the chunk's tokens: each lane reads its slice of
   that token's `x`, accumulates, `simd_sum`, lane 0 writes `Y[token][row]`.
   Weights are read once per chunk per row; `x` (chunk × K halfs) stays in L2.
   The same kernel serves q/k/gate/o, the dense MLP, the shared expert and the
   per-expert MoE projections (expert weights are already packed per role).
   Scales and biases are per 64-element group; load them inside the token loop
   (cheap, cached) or hold them per lane if registers allow.
2. **Batched grouped RMSNorm.** `rmsnorm_bf16w_grouped` is per row; either loop
   N rows in one command buffer or add a row-count parameter. The norm is not
   the bottleneck, so the dispatch loop is acceptable for the first slice.
3. **Batched causal attention.** K2 is 32 query heads / 8 KV heads, head dim
   128, full RoPE, softplus gate, and a context that can exceed any chunk. The
   existing prefill attention kernels are Qwen-shaped; check whether their
   head/stride parameters cover 32/8/128 before writing a K2 variant. Causal
   masking over the chunk plus the KV cache is the one genuinely new shape.
4. **Batched MoVA values.** Each token selects its own four of 64 value
   experts. Two options: per-token GEMV dispatches inside one command buffer
   (no weight reuse, simplest) or a kernel where each row looks up its token's
   selected ids and reads that expert — the second is what makes the batch
   worth it. The router itself is a bf16 GEMV over the chunk (kernel 1).
5. **MoE.** Routing stays per token. Fetch the union of selected experts for
   the chunk once per layer (the existing `planRoutedExperts` /
   `fetchRoutedExperts` take an expert list), then run each expert's GEMM over
   the tokens that chose it. Tokens that chose different sets still share a
   weight read within their subset.
6. **Head.** `lm_head` GEMM over the chunk is only needed for the last token of
   the prompt; keep the existing single-token head call.

### Data flow

- `hidden` becomes an `N × 2560` half buffer; `q/k/gate/v` and `attn` follow.
- Positions are `base + 0..<N`; the KV cache already indexes by absolute
  position, so chunked prefill is just the same writes in bigger groups.
- The runtime's K/V reuse must understand chunks: a cached prefix of `m` tokens
  is reusable when `m` is a multiple of the chunk (or the first chunk is
  re-run), otherwise fall back to the token path.

### Expected win

A 173-token prompt is ~35 s today; the weight traffic is what a batched pass
amortizes. Even at half the ideal bandwidth, the prompt should land near
5–12 s, and a 500-token prompt from ~100 s to well under a minute. Measure with
`TinyTitanBench k2gpu` and the server (a `timing` line already splits prefill
from decode), and keep the cold/warm distinction: expert I/O is unchanged by
batching and will dominate first-token latency until the cache is warm.

### Verification

1. Prefill logits for one chunk must match the token-by-token path on the toy
   install, token for token, inside fp16 tolerance — the same contract the
   layer tests already use. Add it to `K2GraphProbeTests`.
2. On the real installs, greedy output for the golden prompt must match the
   stored token ids; if reduction order changes inside the new kernels, treat
   it as a deliberate numerics change, re-capture both goldens through
   `tools/golden-baseline.sh k2-4 k2-8`, and say so in the commit.
3. `tools/k2-serve-smoke.sh` and the CLI raw/chat paths must keep working.

## Phase 2: ANE prefill sidecar

Convert the parts that are static shapes: embeddings, dense layers 0–2, the
QKV/gate projections, and the attention block for a fixed chunk. MoVA and MoE
are data-dependent routing and belong on the GPU until there is an ANE-side
gather schedule. `tools/ane_sidecars.sh` and the Qwen sidecars are the
template, but a K2 sidecar is only worth building after phase 1 numbers show
the GPU prefill is still the bottleneck — on this host the expert cache and
memory pressure were bigger factors than arithmetic.

## Phase 3: runner integration and cache

The probe runtime is deliberately not `RealForwardRunner`: no prompt cache, no
batching across requests, no runner statistics, plain FP16 caches. If K2 is to
be served beside other models at scale, the choice is either to grow the probe
runtime into that role or to add a K2 branch to the runner. The probe runtime
now has the seams (graph ownership, chunked prefill would slot into it), so the
recommended route is to extend it and keep the runner untouched until the
catalogue needs cross-family scheduling. Catalogue residency already works
(one resident model, lazy load, idle unload via `ManagedModelBackend`).

## Gates to keep green

- `swift test` (2,000+ tests; the K2 toy suites are the numeric contract).
- `tools/lint.sh` (swift-format, lengths, docs, shellcheck, JS, python).
- `tools/golden-baseline.sh --check k2-4 k2-8` on a machine with the installs.
- `tools/k2-serve-smoke.sh` (models, chat, SSE, Responses, tool call, stop
  string, context bound).
- `TinyTitanCLI` raw and chat runs against the 4-bit install.

## First concrete steps

1. `affine_gemm` kernel in `Metal/MoE/` (or `Metal/Quant/`), one simdgroup per
   row and a token loop, with a unit test against N applications of the
   existing GEMV on random 4-bit data.
2. Batched dense-layer prefill only (layers 0–2 on the toy) through the new
   kernel, compared with the token path in `K2GraphProbeTests`.
3. Batched attention for the chunk; only then the sparse layer's MoVA and MoE.
4. Wire the chunked path into `K2ProbeRuntime.prefill` behind a chunk-size
   setting, keep the token path as the fallback, and measure the 173-token
   prompt before touching the goldens.
