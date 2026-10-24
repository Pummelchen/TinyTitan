// Dequantization of the affine quantized matrices, for the batched CPU
// prefill path.
//
// The decode kernels read the packed weights in place: one token means one
// pass over the bytes, so materializing FP32 weights would only add work.
// A batched multiplication is the opposite -- one pass over the weights for
// many tokens -- and there the unpacking has to happen once, into a buffer
// the matrix engine can consume. These two functions are that unpacking, and
// they match the GEMV kernels' layout exactly: unsigned lanes, element 2k in
// the low nibble of byte k, one BF16 scale and bias per group of 64.
//
// They are deliberately free of threading; the caller splits rows, as it does
// for the GEMVs, because rows are independent and the row stride is the only
// pointer arithmetic involved.

#include "include/tinytitan_kernels.h"

#if defined(__APPLE__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
// The classic CBLAS interface is deprecated on macOS 13.3 in favour of the
// ILP64 headers, but this project compiles for one architecture and one
// integer width, and the batched path needs the call. Keeping the deprecated
// call here -- rather than in Swift, where a deprecation warning is an error
// with no local suppression -- leaves one place to revisit if the interface
// moves.
#include <Accelerate/Accelerate.h>
#pragma clang diagnostic pop
#endif

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#define TINYTITAN_DEQUANT_GROUP 64

static inline float tinytitan_bf16_to_float(uint16_t bits) {
    union { uint32_t u; float f; } c;
    c.u = ((uint32_t)bits) << 16;
    return c.f;
}

#if defined(__ARM_NEON)
// Sixteen unsigned bytes -> sixteen floats with this group's affine pair.
static inline void tinytitan_store_scaled_16(uint8x16_t bytes, float *out,
                                             float scale, float bias) {
    const uint16x8_t lo = vmovl_u8(vget_low_u8(bytes));
    const uint16x8_t hi = vmovl_u8(vget_high_u8(bytes));
    const float32x4_t q0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo)));
    const float32x4_t q1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo)));
    const float32x4_t q2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi)));
    const float32x4_t q3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi)));
    const float32x4_t s = vdupq_n_f32(scale);
    const float32x4_t b = vdupq_n_f32(bias);
    vst1q_f32(out + 0, vfmaq_f32(b, q0, s));
    vst1q_f32(out + 4, vfmaq_f32(b, q1, s));
    vst1q_f32(out + 8, vfmaq_f32(b, q2, s));
    vst1q_f32(out + 12, vfmaq_f32(b, q3, s));
}
#endif

void tinytitan_dequant_affine8(const uint8_t *weights,
                               const uint16_t *scales,
                               const uint16_t *biases,
                               size_t rows,
                               size_t n,
                               float *out) {
    const size_t groups = n / TINYTITAN_DEQUANT_GROUP;
    for (size_t r = 0; r < rows; ++r) {
        const uint8_t *row = weights + r * n;
        const uint16_t *scale_row = scales + r * groups;
        const uint16_t *bias_row = biases + r * groups;
        float *out_row = out + r * n;
        for (size_t g = 0; g < groups; ++g) {
            const float scale = tinytitan_bf16_to_float(scale_row[g]);
            const float bias = tinytitan_bf16_to_float(bias_row[g]);
            const uint8_t *q = row + g * TINYTITAN_DEQUANT_GROUP;
            float *o = out_row + g * TINYTITAN_DEQUANT_GROUP;
#if defined(__ARM_NEON)
            for (size_t k = 0; k < TINYTITAN_DEQUANT_GROUP; k += 16) {
                tinytitan_store_scaled_16(vld1q_u8(q + k), o + k, scale, bias);
            }
#else
            for (size_t k = 0; k < TINYTITAN_DEQUANT_GROUP; ++k) {
                o[k] = (float)q[k] * scale + bias;
            }
#endif
        }
    }
}

void tinytitan_dequant_affine4(const uint8_t *weights,
                               const uint16_t *scales,
                               const uint16_t *biases,
                               size_t rows,
                               size_t n,
                               float *out) {
    const size_t groups = n / TINYTITAN_DEQUANT_GROUP;
    const size_t packed_per_group = TINYTITAN_DEQUANT_GROUP / 2;
    for (size_t r = 0; r < rows; ++r) {
        const uint8_t *row = weights + r * (n / 2);
        const uint16_t *scale_row = scales + r * groups;
        const uint16_t *bias_row = biases + r * groups;
        float *out_row = out + r * n;
        for (size_t g = 0; g < groups; ++g) {
            const float scale = tinytitan_bf16_to_float(scale_row[g]);
            const float bias = tinytitan_bf16_to_float(bias_row[g]);
            const uint8_t *q = row + g * packed_per_group;
            float *o = out_row + g * TINYTITAN_DEQUANT_GROUP;
#if defined(__ARM_NEON)
            // Low nibble first: byte k holds element 2k in its low half and
            // 2k+1 in its high half. Interleaving the two nibble vectors is
            // exactly that order.
            for (size_t k = 0; k < packed_per_group; k += 16) {
                const uint8x16_t packed = vld1q_u8(q + k);
                const uint8x16_t lo = vandq_u8(packed, vdupq_n_u8(0x0F));
                const uint8x16_t hi = vshrq_n_u8(packed, 4);
                tinytitan_store_scaled_16(vzip1q_u8(lo, hi), o + k * 2, scale, bias);
                tinytitan_store_scaled_16(vzip2q_u8(lo, hi), o + k * 2 + 16, scale, bias);
            }
#else
            for (size_t k = 0; k < packed_per_group; ++k) {
                const uint8_t byte = q[k];
                o[k * 2] = (float)(byte & 0x0F) * scale + bias;
                o[k * 2 + 1] = (float)(byte >> 4) * scale + bias;
            }
#endif
        }
    }
}

/// `out[rows, columns] = a[rows, inner] * b`, with `b` either transposed
/// (stored columns-by-inner, the weight-matrix case) or not (stored
/// inner-by-columns, the score-times-values case).
///
/// A thin wrapper rather than a Swift call: `cblas_sgemm` is deprecated on
/// this SDK against the ILP64 headers, and a deprecation in Swift is an
/// error with no local suppression.
void tinytitan_sgemm_rowmajor(int transposed_b, int rows, int columns, int inner,
                              const float *a, const float *b, float *out) {
#if defined(__APPLE__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    cblas_sgemm(CblasRowMajor, CblasNoTrans,
                transposed_b != 0 ? CblasTrans : CblasNoTrans,
                rows, columns, inner,
                1.0f, a, inner,
                b, transposed_b != 0 ? inner : columns,
                0.0f, out, columns);
#pragma clang diagnostic pop
#else
    (void)transposed_b;
    (void)rows;
    (void)columns;
    (void)inner;
    (void)a;
    (void)b;
    (void)out;
#endif
}
