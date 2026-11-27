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
///
/// `a` must be a **contiguous** row-major `rows`-by-`inner` matrix: the leading
/// dimension is `inner`, with no padding between rows. A strided operand is not
/// supported and would be read as if it were contiguous, silently producing
/// wrong values (AUD-0085). `b` is contiguous too: `inner`-by-`columns`, or
/// `columns`-by-`inner` when transposed.
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

/// The same recurrence over a chunk of tokens for one head, using the unrolled
/// form `benchmark/gdn_chunked_scan.py` checks. `queries`, `keys` and `values`
/// point at token 0 of this head, with the strides between tokens; `alphas`
/// and `betas` are per token for this head. `readouts` is written
/// `tokens`-by-`dv`, `state` is updated in place.
///
/// Every dimension, stride and allocation product is validated before anything
/// is allocated or written, because this entry point cannot see the caller's
/// buffers. A stride smaller than the width it indexes, a zero alpha/beta
/// stride, a `2 * tokens` or `size_t` product that would wrap, or a BLAS extent
/// above `INT_MAX` is refused with `TINYTITAN_DELTA_RULE_CHUNK_INVALID_GEOMETRY`
/// and leaves `state` and `readouts` untouched. A zero token or head width is a
/// no-op and returns `TINYTITAN_DELTA_RULE_CHUNK_OK`.
///
/// The return values are:
/// - `TINYTITAN_DELTA_RULE_CHUNK_OK` — the BLAS path produced the result.
/// - `TINYTITAN_DELTA_RULE_CHUNK_FELL_BACK` — the scratch allocation failed,
///   and the fixed-buffer sequential fallback produced the result; it is valid.
/// - `TINYTITAN_DELTA_RULE_CHUNK_INVALID_GEOMETRY` — rejected as above.
/// - `TINYTITAN_DELTA_RULE_CHUNK_ALLOCATION_FAILED` — the scratch allocation
///   failed and `dv` is past the fallback's fixed buffer, so nothing was
///   written. The caller must treat `state` and `readouts` as unwritten.
///
/// Valid geometry keeps the historical arithmetic and summation order exactly:
/// the checks only reject input the old code would have run off the end for.
#define TINYTITAN_DELTA_RULE_CHUNK_OK 0
#define TINYTITAN_DELTA_RULE_CHUNK_FELL_BACK 1
#define TINYTITAN_DELTA_RULE_CHUNK_INVALID_GEOMETRY 2
#define TINYTITAN_DELTA_RULE_CHUNK_ALLOCATION_FAILED 3

int tinytitan_delta_rule_chunk(const float *queries,
                               size_t queryStride,
                               const float *keys,
                               size_t keyStride,
                               const float *values,
                               size_t valueStride,
                               const float *alphas,
                               size_t alphaStride,
                               const float *betas,
                               size_t betaStride,
                               size_t tokens,
                               size_t dk,
                               size_t dv,
                               float *state,
                               float *readouts,
                               size_t readoutStride);

#endif /* TINYTITAN_KERNELS_H */
