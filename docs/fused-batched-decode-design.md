# Fused batched decode: design for T3

Status: design; no engine change. Scope is the T3-T10 slice in
`docs/maya-remaining-work-2026-10-21.md`. The three reverted attempts in
`docs/batched-decode-phase2c-2026-10-21.md` are evidence for what not to do:
they shared projections while keeping a per-row FFN and synchronised at every
layer boundary, which removed the decode prefetch overlap.

## Target

B = 2 and B = 4 rows, one token each, on pre-norm MoE families with no PLE, no
QSA and no hyper-connections (Ornith 4-bit first). GDN layers may remain
per-row for the first landing; full-attention layers carry the batching win.
B = 1 stays on the existing path and must remain byte-identical.

## Invariant

No layer boundary is a CPU barrier. The only waits are:

1. router logits readback (CPU needs the ids), and
2. the union expert fetch completion for the layer that is about to run
   phase 1.

Everything else is queued behind the previous command buffer on the same
queue. In particular, the next layer's entry norm and QKV may be encoded before
the current layer's routed command completes, because the GPU queue already
orders them and every buffer they touch is row-private or batch-private.

## Per-layer pipeline

For each layer L:

1. **Entry norm, all rows.** `encodeResidualEntryDecode` writes row r's
   post-entry residual into `normed[r * D]` using `hidden[r * D]`.
2. **QKV, all rows.** One two-row (B=2) or two two-row (B=4) GEMV per
   projection reads the weights once. Outputs are `[B, qRows]`, `[B, kvDim]`,
   `[B, kvDim]`; gated q is the packed `[query ; gate]` row.
3. **Per-row attention epilogue and attention.** Split q/gate, per-head q/k
   norm, RoPE, write K/V to each row's slot, run the existing attention kernel
   per row into `attn[r * qDim]`. No weight is shared here except the attention
   kernel's code; the math is per sequence.
4. **o_proj, all rows.** One batched GEMV reads o_proj once into
   `o[r * D]`.
5. **Residual + post-attn norm, all rows.** Add `o[r]` to `hidden[r]`, then
   write the MLP input to `routedX[r]`.
6. **Router, all rows.** Batched router encode (`queryCount = B`) writes
   `routeIDs[B, topK]` and `routeWeights[B, topK]`; commit and read back.
   Build the union of experts and one cache plan.
7. **Next-layer prefetch.** Run the existing next-router probe per row against
   `routedX[r]`, union the predicted ids, and enqueue the L+1 prefetch before
   the current FFN. This is the overlap that the reverted lockstep lost.
8. **Batched FFN.**
   - Shared expert: one batched projection set over `[B, D]` into `[B, F]`,
     one activation, one down projection into `h1[B, D]`.
   - Routed: the union fetch for L, then per-row phase-1 and phase-2 kernels
     (the decode kernels, not the prefill tile scheduler). Phase 2 takes
     `h1[r]` as its residual and writes `h2[r]`.
   - Residual: add `h2[r]` to `hidden[r]`.
9. **Queue the next layer.** Do not wait for step 8 before encoding L+1's
   entry norm/QKV. The union fetch for L+1 is awaited only when step 8 of L+1
   needs the blobs.

For GDN layers, phase 1 landing may run the whole layer per row, with the
batch's row-private hidden buffers and the existing slot state. That keeps the
first landing small; batching GDN projections is a follow-up task if the
full-attention win justifies it.

## Buffers

Batch-private (one allocation per runner, sized B):

| Buffer | Contents |
| --- | --- |
| `hidden[B, D]` | authoritative residual per row |
| `normed[B, D]` | entry/QKV input |
| `q[B, qRows]`, `k[B, kvDim]`, `v[B, kvDim]` | projected QKV |
| `attn[B, qDim]`, `o[B, D]` | attention output and o_proj |
| `routedX[B, D]`, `h1[B, D]`, `h2[B, D]` | MLP input, shared branch, routed branch |
| `routeIDs[B, topK]`, `routeWeights[B, topK]` | batched router |
| `nextIDs[B, topK]` | next-layer probe for prefetch |

Row-private scratch is limited to the attention epilogue and the per-row
phase-1/2 scratch that cannot be shared by definition. No buffer is reused by
two rows in the same layer.

## Union fetch and prefetch

- One `planRoutedExperts(layer, experts: union)` per layer, one lease.
- One fetch completion gates that layer's phase 1; it may overlap the current
  layer's shared-expert GPU work and the previous layer's tail.
- The next-layer router probe is a predictor, not authority: a wrong prediction
  costs a miss, never a wrong answer.
- The prefetch ring must be sized for the union, not for one row. If the union
  exceeds the ring's slots, the extra ids are demand reads and the plan records
  the overflow.

## B=1 identity

The width-1 path is not refactored to go through the batch path. `produceBatch`
with one row calls `produce` exactly as today. The first fused tests assert
byte-identical logits against the width-1 baseline; any shared helper must be
shown not to change width-1 arithmetic.

## Gates

1. Toy B=2/4: logits match per-row reference within the existing 1e-3 gate;
   B=1 byte-identical.
2. Ornith 4-bit B=2/4: identical text on the fixed 4-prompt battery at
   64/128 tokens, seeded and nonce-stable.
3. Profiling: next-layer expert fetch starts before the current FFN's phase 2
   and the routed command does not block the next layer's QKV encode.
4. Benchmark: aggregate tok/s beats serial at 64/128/256 and no wall regression.
   If it does not, revert the code and keep the measurements in the note.
5. Host KV and GDN are regression-tested, but neither is required for the
   first fused landing.

## Risks

- Union expert cache pressure: more distinct experts per layer may evict the
  row working set. The plan measures hit rate and bytes, not just tok/s.
- Router readback remains a CPU barrier; a batched router reduces dispatches,
  not the round trip.
- Per-row attention leaves the attention kernels unshared. That is deliberate:
  softmax/KV are per sequence and were never the batching target.
- B=4 doubles the union width. If the ring cannot hold it, B=4 may be a loss
  even when B=2 wins; the decision is per width, not per feature.
