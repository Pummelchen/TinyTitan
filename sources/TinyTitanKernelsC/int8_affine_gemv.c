// The decode inner loop for the CPU side-engine.
//
// Sibling of int4_affine_gemv.c: same affine format, same factoring, one
// byte per element instead of two elements per byte. It exists because the
// small resident model is built at 8-bit -- quantization is not a suspect
// while correctness is still being established -- and because a dense 2B
// spends nearly all of a token in exactly this loop.
//
// The two kernels are deliberately parallel in structure, and both factor one
// group as `scale * sum(q*x) + bias * sum(x)`. That factoring is the contract;
// the reduction inside a group is not shared, so reordering one kernel moves
// the last bits of its own width rather than breaking an exact agreement
// between the 4-bit and 8-bit builds. Keep them parallel anyway: a reader
// comparing the two should be looking at the factoring, not at an accident.
//
// Bandwidth, not arithmetic, is the ceiling here: at 8-bit a 2B model reads
// about 2.4 GB per token, and an M3 sustains well under its headline figure
// once the GPU is also reading. Everything below is arranged so the weight
// stream is sequential and read exactly once.

#include "include/tinytitan_kernels.h"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#define TINYTITAN_GROUP_SIZE 64
// Groups per row for the shapes this kernel serves: 2048/64 = 32 for the
// attention and gate/up projections, 6144/64 = 96 for down. Generous; wider
// rows recompute sum(x) per row rather than being wrong.
#define TINYTITAN_MAX_GROUPS_8 256

static inline float tinytitan_bf16_8(uint16_t bits) {
    union { uint32_t u; float f; } c;
    c.u = ((uint32_t)bits) << 16;
    return c.f;
}

void tinytitan_int8_affine_gemv(const uint8_t *weights,
                            const uint16_t *scales,
                            const uint16_t *biases,
                            const float *x,
                            size_t rows,
                            size_t n,
                            float *out) {
    const size_t groups = n / TINYTITAN_GROUP_SIZE;

#if defined(__ARM_NEON)
    // sum(x) per group depends only on x, so it is hoisted out of the row
    // loop exactly as in the 4-bit kernel: with 6144 rows to serve,
    // recomputing it per row is 6144 redundant passes over x.
    float group_xsum[TINYTITAN_MAX_GROUPS_8];
    const size_t xsum_groups = groups <= TINYTITAN_MAX_GROUPS_8 ? groups : 0;
    for (size_t g = 0; g < xsum_groups; ++g) {
        const float *xg = x + g * TINYTITAN_GROUP_SIZE;
        float32x4_t s = vdupq_n_f32(0.0f);
        for (size_t i = 0; i < TINYTITAN_GROUP_SIZE; i += 4) {
            s = vaddq_f32(s, vld1q_f32(xg + i));
        }
        group_xsum[g] = vaddvq_f32(s);
    }

    for (size_t r = 0; r < rows; ++r) {
        const uint8_t *w_row = weights + r * n;
        const uint16_t *s_row = scales + r * groups;
        const uint16_t *b_row = biases + r * groups;
        float acc = 0.0f;

        for (size_t g = 0; g < groups; ++g) {
            const uint8_t *wg = w_row + g * TINYTITAN_GROUP_SIZE;
            const float *xg = x + g * TINYTITAN_GROUP_SIZE;
            // Four accumulators rather than one: the FMA latency on these
            // cores is longer than its throughput, so a single chain stalls.
            float32x4_t d0 = vdupq_n_f32(0.0f), d1 = vdupq_n_f32(0.0f);
            float32x4_t d2 = vdupq_n_f32(0.0f), d3 = vdupq_n_f32(0.0f);
            float32x4_t xs = vdupq_n_f32(0.0f);
            const int have_xsum = (xsum_groups != 0);

            // 64 bytes per group; 16 bytes -> 16 elements per iteration.
            for (size_t k = 0; k < TINYTITAN_GROUP_SIZE; k += 16) {
                const uint8x16_t q8 = vld1q_u8(wg + k);
                const uint16x8_t q16_lo = vmovl_u8(vget_low_u8(q8));
                const uint16x8_t q16_hi = vmovl_u8(vget_high_u8(q8));
                const float32x4_t q0 =
                    vcvtq_f32_u32(vmovl_u16(vget_low_u16(q16_lo)));
                const float32x4_t q1 =
                    vcvtq_f32_u32(vmovl_u16(vget_high_u16(q16_lo)));
                const float32x4_t q2 =
                    vcvtq_f32_u32(vmovl_u16(vget_low_u16(q16_hi)));
                const float32x4_t q3 =
                    vcvtq_f32_u32(vmovl_u16(vget_high_u16(q16_hi)));

                const float *xp = xg + k;
                const float32x4_t x0 = vld1q_f32(xp);
                const float32x4_t x1 = vld1q_f32(xp + 4);
                const float32x4_t x2 = vld1q_f32(xp + 8);
                const float32x4_t x3 = vld1q_f32(xp + 12);

                d0 = vfmaq_f32(d0, q0, x0);
                d1 = vfmaq_f32(d1, q1, x1);
                d2 = vfmaq_f32(d2, q2, x2);
                d3 = vfmaq_f32(d3, q3, x3);
                if (!have_xsum) {
                    xs = vaddq_f32(xs, x0);
                    xs = vaddq_f32(xs, x1);
                    xs = vaddq_f32(xs, x2);
                    xs = vaddq_f32(xs, x3);
                }
            }
            // (d0+d1)+(d2+d3) is this kernel's own grouping, and it is not the
            // 4-bit kernel's: there every lane chains all sixteen of its
            // products into one accumulator. Same products, different
            // association, so the two widths disagree in the last bit on most
            // rows -- measured in Int8AffineGEMVTests, and asserted nowhere as
            // an equality. What the widths share is the *factoring* below.
            const float32x4_t dot = vaddq_f32(vaddq_f32(d0, d1), vaddq_f32(d2, d3));
            const float xsum = have_xsum ? group_xsum[g] : vaddvq_f32(xs);
            acc += tinytitan_bf16_8(s_row[g]) * vaddvq_f32(dot)
                 + tinytitan_bf16_8(b_row[g]) * xsum;
        }
        out[r] = acc;
    }
#else
    // Portable fallback. TinyTitan targets Apple Silicon, so this exists to keep
    // the file compilable rather than as a path anyone is expected to take.
    for (size_t r = 0; r < rows; ++r) {
        const uint8_t *w_row = weights + r * n;
        float acc = 0.0f;
        for (size_t g = 0; g < groups; ++g) {
            const uint8_t *wg = w_row + g * TINYTITAN_GROUP_SIZE;
            const float *xg = x + g * TINYTITAN_GROUP_SIZE;
            float dot = 0.0f, xsum = 0.0f;
            for (size_t k = 0; k < TINYTITAN_GROUP_SIZE; ++k) {
                dot += (float)wg[k] * xg[k];
                xsum += xg[k];
            }
            acc += tinytitan_bf16_8(scales[r * groups + g]) * dot
                 + tinytitan_bf16_8(biases[r * groups + g]) * xsum;
        }
        out[r] = acc;
    }
#endif
}
