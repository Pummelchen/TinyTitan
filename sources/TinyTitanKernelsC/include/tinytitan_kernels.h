#ifndef TINYTITAN_KERNELS_H
#define TINYTITAN_KERNELS_H

#include <stddef.h>
#include <stdint.h>

// Re-exported so the Swift module surfaces every C entry point.
#include "tinytitan_expert_io.h"

/// `out[r] = sum_i (q[r][i] * scale[r][g(i)] + bias[r][g(i)]) * x[i]`
///
/// Affine INT4 GEMV over a `rows`-by-`n` matrix in the packed `.ssdai`
/// layout: nibbles low-first (element 2k in the low half of byte k), one BF16
/// scale and one BF16 bias per group of 64 elements, `scales` and `biases`
/// each `rows * (n / 64)` entries in row-major order.
///
/// `n` must be a multiple of 64. `out` is written, not accumulated.
///
/// Accumulation is factored as `scale * sum(q*x) + bias * sum(x)` per group,
/// the same factoring `moe.metal` uses, so each group rounds once. That is an
/// algebraic claim, not a bitwise one: which element each lane owns, how many
/// accumulators it has, and how the row is reduced horizontally all differ
/// between the CPU and the GPU, so a CPU row and a Metal row agree to a
/// rounding error rather than bit for bit.
void tinytitan_int4_affine_gemv(const uint8_t *weights,
                            const uint16_t *scales,
                            const uint16_t *biases,
                            const float *x,
                            size_t rows,
                            size_t n,
                            float *out);

/// `out[r] = sum_i (q[r][i] * scale[r][g(i)] + bias[r][g(i)]) * x[i]`
///
/// Affine INT8 GEMV over a `rows`-by-`n` matrix: one byte per element, one
/// BF16 scale and one BF16 bias per group of 64, `scales` and `biases` each
/// `rows * (n / 64)` entries in row-major order.
///
/// `n` must be a multiple of 64. `out` is written, not accumulated. Rows are
/// independent, so a caller threading over row ranges advances `weights`,
/// `scales`, `biases` and `out` together and passes its own `rows`.
///
/// Factored as `scale * sum(q*x) + bias * sum(x)` per group, as in the 4-bit
/// kernel. The reduction *inside* a group is not the same -- four accumulators
/// here against one chain there -- so the two widths of one model agree to a
/// rounding error, not bit for bit.
void tinytitan_int8_affine_gemv(const uint8_t *weights,
                            const uint16_t *scales,
                            const uint16_t *biases,
                            const float *x,
                            size_t rows,
                            size_t n,
                            float *out);

/// Materialize a quantized matrix as FP32, for a batched multiplication.
///
/// Same layout and same affine factoring as the GEMV kernels (`q*scale+bias`
/// per group of 64), but every element is written instead of consumed in
/// place. `out` is `rows*n` floats; the caller splits rows across threads.
void tinytitan_dequant_affine4(const uint8_t *weights,
                               const uint16_t *scales,
                               const uint16_t *biases,
                               size_t rows,
                               size_t n,
                               float *out);

void tinytitan_dequant_affine8(const uint8_t *weights,
                               const uint16_t *scales,
                               const uint16_t *biases,
                               size_t rows,
                               size_t n,
                               float *out);

/// Row-major matrix multiplication: `out = a * b`, with `b` transposed when
/// `transposed_b` is non-zero (stored `columns`-by-`inner`), otherwise stored
/// `inner`-by-`columns`. A thin wrapper over the platform BLAS so the
/// deprecated CBLAS declaration stays out of Swift.
void tinytitan_sgemm_rowmajor(int transposed_b, int rows, int columns, int inner,
                              const float *a, const float *b, float *out);

/// The gated-DeltaNet recurrence for one token.
///
/// `state` is `heads * dv * dk` floats and is updated in place; `readout` is
/// `heads * dv` floats and is written. `key` and `query` are `keyHeads * dk`
/// wide, and head `h` reads key head `h / (heads / keyHeads)` -- the
/// consecutive-value-heads mapping the Swift path uses. `a`/`b` are the two
/// per-head projections, `decayLog`/`decayBias` the per-head decay inputs.
void tinytitan_delta_rule(const float *query,
                          const float *key,
                          const float *value,
                          const float *a,
                          const float *b,
                          const float *decayLog,
                          const float *decayBias,
                          size_t heads,
                          size_t keyHeads,
                          size_t dk,
                          size_t dv,
                          float inverse,
                          float *state,
                          float *readout);

/// The same rule over a contiguous head range, so a caller can split the
/// independent heads across threads. Head indices are absolute; `state` and
/// `readout` are the full arrays and each worker writes its own heads.
void tinytitan_delta_rule_range(const float *query,
                                const float *key,
                                const float *value,
                                const float *a,
                                const float *b,
                                const float *decayLog,
                                const float *decayBias,
                                size_t heads,
                                size_t keyHeads,
                                size_t dk,
                                size_t dv,
                                float inverse,
                                size_t firstHead,
                                size_t headCount,
                                float *state,
                                float *readout);

#endif /* TINYTITAN_KERNELS_H */
