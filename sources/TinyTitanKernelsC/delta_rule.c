// The gated-DeltaNet recurrence for one token, on the CPU.
//
// The batched prefill made the projections cheap and left this loop as the
// bottleneck: per token, per layer it is 16 heads times 128 rows of two
// 128-wide dot products, and the Swift scalar loop spent most of the prefill
// inside it. The arithmetic is unchanged -- decay the state row, read the
// stored value, correct towards the new one, read the query -- only the
// reduction is vectorized, so the result agrees to a rounding error with the
// scalar path rather than bit for bit.
//
// One token at a time is deliberate: the recurrence reads the state the
// previous token wrote, so there is no batch dimension here to widen.

#include "include/tinytitan_kernels.h"

#include <math.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static inline float tinytitan_softplus(float x) {
    return log1pf(expf(-fabsf(x))) + fmaxf(x, 0.0f);
}

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
                          float *readout) {
    if (heads == 0 || keyHeads == 0 || dk == 0 || dv == 0 || heads % keyHeads != 0) {
        return;
    }
    const size_t repeats = heads / keyHeads;
    for (size_t head = 0; head < heads; ++head) {
        const float beta = 1.0f / (1.0f + expf(-b[head]));
        const float x = a[head] + decayBias[head];
        const float decay = expf(-expf(decayLog[head]) * tinytitan_softplus(x));
        const float *keyRow = key + (head / repeats) * dk;
        const float *queryRow = query + (head / repeats) * dk;
        for (size_t index = 0; index < dv; ++index) {
            float *row = state + (head * dv + index) * dk;
            float stored = 0.0f;
            float sum = 0.0f;
#if defined(__ARM_NEON)
            float32x4_t decayed = vdupq_n_f32(0.0f);
            for (size_t column = 0; column < dk; column += 4) {
                const float32x4_t scaled = vmulq_n_f32(vld1q_f32(row + column), decay);
                vst1q_f32(row + column, scaled);
                decayed = vfmaq_f32(decayed, scaled, vld1q_f32(keyRow + column));
            }
            stored = vaddvq_f32(decayed);
            const float correction = (value[head * dv + index] - stored) * beta;
            float32x4_t read = vdupq_n_f32(0.0f);
            for (size_t column = 0; column < dk; column += 4) {
                const float32x4_t updated = vfmaq_n_f32(
                    vld1q_f32(row + column), vld1q_f32(keyRow + column), correction);
                vst1q_f32(row + column, updated);
                read = vfmaq_f32(read, updated, vld1q_f32(queryRow + column));
            }
            sum = vaddvq_f32(read);
#else
            for (size_t column = 0; column < dk; ++column) {
                row[column] *= decay;
                stored += row[column] * keyRow[column];
            }
            const float correction = (value[head * dv + index] - stored) * beta;
            for (size_t column = 0; column < dk; ++column) {
                row[column] += keyRow[column] * correction;
                sum += row[column] * queryRow[column];
            }
#endif
            readout[head * dv + index] = sum * inverse;
        }
    }
}
