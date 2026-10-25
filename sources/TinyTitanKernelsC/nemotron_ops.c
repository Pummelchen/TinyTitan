// Nemotron 3.5 Bolt's non-GEMM CPU kernels.
//
// The 74B-A9B text tower alternates Mamba-2 blocks, LatentMoE blocks and eight
// full-attention blocks. Its matrix-vector products go through the affine
// GEMV kernels in this target; what is left is the per-token elementwise work
// those layers do -- RMS norms (plain and Mamba's gated one), the depthwise
// causal convolution, the SSM state update, SiLU and relu^2, a BF16 GEMV for
// the router, and an affine embedding row lookup.
//
// Every function is a straight transcription of
// `modeling_nemotron_h.py` at the pinned BF16 revision; the comments name the
// reference line each one follows where the order of operations matters.

#include "include/tinytitan_kernels.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static inline float nemotron_bf16(uint16_t bits) {
    union { uint32_t u; float f; } c;
    c.u = ((uint32_t)bits) << 16;
    return c.f;
}

static inline float nemotron_silu(float x) {
    return x / (1.0f + expf(-x));
}

/// `softplus(x)` with the reference's overflow behaviour: PyTorch's softplus
/// switches to the identity above its threshold, and `expf` would otherwise
/// overflow for the largest dt values this checkpoint produces.
static inline float nemotron_softplus(float x) {
    if (x > 20.0f) return x;
    return log1pf(expf(x));
}

void tinytitan_rmsnorm_f32(const float *x, const float *weight, size_t n,
                           float epsilon, float *out) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) sum += x[i] * x[i];
    const float rstd = 1.0f / sqrtf(sum / (float)n + epsilon);
    for (size_t i = 0; i < n; ++i) out[i] = weight != NULL ? x[i] * rstd * weight[i]
                                                           : x[i] * rstd;
}

/// Mamba's `MambaRMSNormGated` with `norm_before_gate=False`:
/// `out = RMSNorm(x, group) * SiLU(gate)`.
void tinytitan_rmsnorm_gated_f32(const float *x, const float *gate,
                                 const float *weight, size_t n, size_t group,
                                 float epsilon, float *out) {
    for (size_t start = 0; start < n; start += group) {
        size_t end = start + group;
        if (end > n) end = n;
        float sum = 0.0f;
        for (size_t i = start; i < end; ++i) sum += x[i] * x[i];
        const float rstd =
            1.0f / sqrtf(sum / (float)(end - start) + epsilon);
        for (size_t i = start; i < end; ++i) {
            out[i] = weight != NULL ? x[i] * rstd * weight[i] : x[i] * rstd;
            out[i] *= nemotron_silu(gate[i]);
        }
    }
}

void tinytitan_silu_f32(const float *x, size_t n, float *out) {
    for (size_t i = 0; i < n; ++i) out[i] = nemotron_silu(x[i]);
}

void tinytitan_relu2_f32(const float *x, size_t n, float *out) {
    for (size_t i = 0; i < n; ++i) {
        const float v = x[i] > 0.0f ? x[i] : 0.0f;
        out[i] = v * v;
    }
}

void tinytitan_bf16_gemv(const uint16_t *weights, const float *x,
                         size_t rows, size_t n, float *out) {
    // Deliberately scalar: there is no NEON bfloat16 widening here, and
    // reading the bits as half (which the VCVT instruction does) would be
    // silently wrong. The router is 496 x 4096 at most, forty times a token.
    for (size_t r = 0; r < rows; ++r) {
        const uint16_t *row = weights + r * n;
        float acc = 0.0f;
        for (size_t i = 0; i < n; ++i) acc += nemotron_bf16(row[i]) * x[i];
        out[r] = acc;
    }
}

void tinytitan_affine_embedding_row(const uint8_t *weights,
                                    const uint16_t *scales,
                                    const uint16_t *biases, size_t columns,
                                    size_t bits, size_t row, float *out) {
    const size_t groups = columns / 64;
    const uint8_t *rowWeights = bits == 4 ? weights + row * (columns / 2)
                                          : weights + row * columns;
    const uint16_t *rowScales = scales + row * groups;
    const uint16_t *rowBiases = biases + row * groups;
    for (size_t g = 0; g < groups; ++g) {
        const float scale = nemotron_bf16(rowScales[g]);
        const float bias = nemotron_bf16(rowBiases[g]);
        for (size_t k = 0; k < 64; ++k) {
            const size_t index = g * 64 + k;
            unsigned value;
            if (bits == 4) {
                const uint8_t byte = rowWeights[index >> 1];
                value = (index & 1) ? (unsigned)(byte >> 4) : (unsigned)(byte & 0x0F);
            } else {
                value = rowWeights[index];
            }
            out[index] = (float)value * scale + bias;
        }
    }
}

void tinytitan_mamba_conv_step(const float *input, const uint16_t *weight,
                               const uint16_t *bias, size_t channels,
                               size_t kernel, float *state, float *out) {
    // `state` is channels x kernel, oldest first. Shift, append, then the
    // depthwise dot the reference computes after `torch.roll`.
    for (size_t c = 0; c < channels; ++c) {
        float *history = state + c * kernel;
        for (size_t k = 0; k + 1 < kernel; ++k) history[k] = history[k + 1];
        history[kernel - 1] = input[c];
        float acc = 0.0f;
        for (size_t k = 0; k < kernel; ++k) {
            acc += history[k] * nemotron_bf16(weight[c * kernel + k]);
        }
        if (bias != NULL) acc += nemotron_bf16(bias[c]);
        out[c] = nemotron_silu(acc);
    }
}

void tinytitan_mamba_ssm_range(const float *x, const float *b, const float *c,
                               const float *dt, const uint16_t *aLog,
                               const uint16_t *skipD, const uint16_t *dtBias,
                               size_t heads, size_t headDim, size_t stateSize,
                               size_t groups, float timeStepMin,
                               size_t firstHead, size_t headCount,
                               float *state, float *y) {
    const size_t headsPerGroup = heads / groups;
    for (size_t h = firstHead; h < firstHead + headCount; ++h) {
        const size_t group = h / headsPerGroup;
        const float a = -expf(nemotron_bf16(aLog[h]));
        const float dtValue = fmaxf(
            nemotron_softplus(dt[h] + nemotron_bf16(dtBias[h])), timeStepMin);
        const float *bGroup = b + group * stateSize;
        const float *cGroup = c + group * stateSize;
        const float *xh = x + h * headDim;
        float *stateHead = state + h * headDim * stateSize;
        float *yh = y + h * headDim;
        for (size_t d = 0; d < headDim; ++d) {
            const float decay = expf(dtValue * a);
            const float dB = dtValue * xh[d];
            float *row = stateHead + d * stateSize;
            float acc = 0.0f;
            for (size_t n = 0; n < stateSize; ++n) {
                row[n] = row[n] * decay + dB * bGroup[n];
                acc += row[n] * cGroup[n];
            }
            yh[d] = acc + nemotron_bf16(skipD[h]) * xh[d];
        }
    }
}
