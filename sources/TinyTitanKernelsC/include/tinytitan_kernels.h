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

/// The same recurrence over a chunk of tokens for one head, using the unrolled
/// form `benchmark/gdn_chunked_scan.py` checks. `queries`, `keys` and `values`
/// point at token 0 of this head, with the strides between tokens; `alphas`
/// and `betas` are per token for this head. `readouts` is written
/// `tokens`-by-`dv`, `state` is updated in place. Returns 0 on success and 1
/// if its scratch allocation failed (the sequential fallback still ran when
/// `dv` fits the fixed buffer).
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

/// Nemotron 3.5 Bolt's CPU elementwise kernels. See nemotron_ops.c.
void tinytitan_rmsnorm_f32(const float *x, const float *weight, size_t n,
                           float epsilon, float *out);
void tinytitan_rmsnorm_gated_f32(const float *x, const float *gate,
                                 const float *weight, size_t n, size_t group,
                                 float epsilon, float *out);
void tinytitan_silu_f32(const float *x, size_t n, float *out);
void tinytitan_relu2_f32(const float *x, size_t n, float *out);
/// `out[r] = sum_i bf16(weights[r][i]) * x[i]`, for the MoE router.
void tinytitan_bf16_gemv(const uint16_t *weights, const float *x,
                         size_t rows, size_t n, float *out);
/// One row of an affine group-64 embedding table, dequantized to FP32.
void tinytitan_affine_embedding_row(const uint8_t *weights,
                                    const uint16_t *scales,
                                    const uint16_t *biases, size_t columns,
                                    size_t bits, size_t row, float *out);
/// Shift the depthwise history, append `input`, convolve, add bias, SiLU.
void tinytitan_mamba_conv_step(const float *input, const uint16_t *weight,
                               const uint16_t *bias, size_t channels,
                               size_t kernel, float *state, float *out);
/// One Mamba-2 selective-scan step over a contiguous head range.
void tinytitan_mamba_ssm_range(const float *x, const float *b, const float *c,
                               const float *dt, const uint16_t *aLog,
                               const uint16_t *skipD, const uint16_t *dtBias,
                               size_t heads, size_t headDim, size_t stateSize,
                               size_t groups, float timeStepMin,
                               size_t firstHead, size_t headCount,
                               float *state, float *y);

#endif /* TINYTITAN_KERNELS_H */
