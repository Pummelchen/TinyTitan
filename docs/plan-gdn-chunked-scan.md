# Plan: a chunked scan for the gated-DeltaNet recurrence

Status: **implemented and measured.** The prototype is
`benchmark/gdn_chunked_scan.py`; it reproduces the scalar recurrence exactly
(max error ~1e-14 on random cases). The engine kernel is
`tinytitan_delta_rule_chunk` in `sources/TinyTitanKernelsC/gdn_chunk.c`, wired
into `gatedDeltaNetBatch` in chunks of 8 tokens by default
(`TINYTITAN_GDN_CHUNK` overrides the length).

## Why

The batched CPU prefill made the projections cheap and left the delta rule as
the cost of a long prompt: a 7K-token profile put the gated-DeltaNet path at
about two thirds of prefill and attention at about a tenth. The rule runs one
token at a time because each token reads the state the previous one wrote, and
it reads and writes the head's whole `dv * dk` state for every token. For the
2B that is 16 heads x 128 x 128 floats = 1 MB per layer per token across 18
linear-attention layers, so a 7K prompt streams roughly 250 GB of state.

A chunked scan reads the state once per chunk instead of once per token.

## The algebra

Scalar recurrence as `tinytitan_delta_rule` runs it:

    S_t = alpha_t * S_{t-1}
    c_t = beta_t * (v_t - S_t @ k_t)
    S_t = S_t + outer(c_t, k_t)
    o_t = S_t @ q_t

With `A_t = prod_{i<=t} alpha_i`, unrolling gives

    S_t = A_t * S_0 + sum_{s<=t} (A_t / A_s) * outer(c_s, k_s)

so `S_0` is touched only by `S_0 K^T` and `S_0 Q^T` (two products per chunk).
The within-chunk part needs `K K^T`, `K Q^T`, and a triangular pass that
computes each `c_t` and `o_t` from the earlier corrections:

    c_t = beta_t * (v_t - alpha_t * [A_{t-1} S_0 k_t
          + sum_{s<t} (A_{t-1}/A_s) c_s (k_s . k_t)])
    o_t = alpha_t * [A_{t-1} S_0 q_t
          + sum_{s<t} (A_{t-1}/A_s) c_s (k_s . q_t)]
          + c_t (k_t . q_t)

and the final state is

    S_chunk = A_chunk * S_0 + sum_s (A_chunk / A_s) * outer(c_s, k_s).

`benchmark/gdn_chunked_scan.py` checks both forms on random cases (dv, dk and
chunk are CLI flags) and exits non-zero if they differ by more than 1e-9.

## Implemented, and what it measured

The kernel computes `S_0 K^T | S_0 Q^T`, the Gram matrices and the final
low-rank update as BLAS calls (the same wrapper the projections use); only the
triangular pass is scalar. One correctness trap mattered: `A_t/A_s` must be
computed as the backward product of the alphas between `s` and `t`, not by
dividing `A_t` by `A_s` -- with a strong decay `A_s` underflows to zero and
`0/0` poisons the state. The first version divided, and produced `!!!!!!!!` at
chunk 32 on the real model while chunks 8 and 16 happened to pass.

Measured on the 2B 4-bit, greedy, chunk 8:

| Prompt tokens | Before the chunked scan | After |
| --- | ---: | ---: |
| 641 | 4.1 s | **3.9 s** |
| 2,201 | 10.7 s | **8.9 s** |
| ~4,000 | 18.9 s | **18.2 s** |
| ~7,000 | 35.1 s | **30.6 s** |

Greedy text is identical at 641 and 2,201 tokens, and on the 4B and both 9B
widths. The win is smaller than the earlier "two thirds of prefill" suggested:
after the per-token kernel was threaded and the SiLU vectorized, the state
stream was no longer the binding cost. A fresh 7K profile puts the chunk
kernel itself at about 7% of main-thread samples; the gated-DeltaNet layer's
projections -- which go through the same BLAS path -- dominate it. The chunked
scan is therefore kept for the traffic it saves, not as the next big lever.

## What it buys and what it costs

- State traffic falls by the chunk length: one read and one write per chunk
  instead of per token.
- The within-chunk work grows: the triangular pass is O(C^2 * dv) per chunk,
  and the Gram matrices are O(C^2 * dk) to build and C^2 floats to hold. At
  C = 64 and the 2B's dv = dk = 128 that is a few hundred kilobytes per head,
  and it replaces per-token state streaming with cache-resident vectors.
- The projections and the rest of the layer are untouched; this only replaces
  the token loop in `gatedDeltaNetBatch`.

## Acceptance

1. The C kernel matches the scalar kernel to float32 rounding on the same
   random cases (a test beside `CPUBatchedPrefillTests`).
2. The engine's greedy output stays byte-identical at 641, 2,201 (chunk
   boundary), 4K and 7K tokens against the current batched path — the same gate
   the batched prefill itself passed.
3. Measured prefill time at 2K and 7K: the state traffic is the claim, so a
   win that does not show there is a failed implementation, not noise.
4. Chunk sizes below the crossover stay on the existing per-token kernel;
   short chunks are expected to lose to the scalar loop.

## Risks

- The summation order changes (Gram-matrix products instead of per-row dot
  products), so bitwise equality is not expected; the gate is the generated
  text, as it was for the dequantized projections.
- The per-token `alpha_t`/`beta_t` make the triangular pass more than a plain
  solve; the prototype is the specification and should be kept as the oracle.
- `C^2` memory per head is bounded but must be checked against the side
  engine's resident budget when it runs beside a 35B.
