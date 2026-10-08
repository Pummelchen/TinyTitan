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

#endif /* TINYTITAN_KERNELS_H */
