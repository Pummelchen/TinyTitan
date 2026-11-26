// The gated-DeltaNet recurrence over a chunk of tokens, for one head.
//
// The token-ordered kernel (`delta_rule.c`) reads and writes the head's whole
// dv*dk state for every token; a 2B prompt therefore streams about 1 MB per
// layer per token. This kernel unrolls the same recurrence over a chunk and
// touches the state twice per chunk instead:
//
//   S_t = alpha_t * S_{t-1}
//   c_t = beta_t * (v_t - S_t k_t)
//   S_t = S_t + c_t k_t^T
//   o_t = S_t q_t
//
// Unrolled with A_t = prod_{i<=t} alpha_i:
//
//   S_t = A_t S_0 + sum_{s<=t} (A_t/A_s) c_s k_s^T
//
// S_0 is then used only by S_0 K^T and S_0 Q^T, and the within-chunk part
// needs the Gram matrices K K^T and K Q^T plus a triangular pass over c_s.
// The Python prototype `benchmark/gdn_chunked_scan.py` is the specification
// and checks this algebra against the scalar loop.
//
// The state products, the Gram matrices and the final low-rank update are
// matrix multiplications, so they go through the same BLAS wrapper the
// projections use; only the triangular pass is scalar. The summation order is
// not the scalar loop's, so the result agrees to a rounding error, not bit for
// bit.

#include "include/tinytitan_kernels.h"
#include "include/tinytitan_testing.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define TINYTITAN_CHUNK_FALLBACK_DV 256

// Strict C99 checked size_t arithmetic. The dimensions arrive through a C entry
// point that cannot see the caller's buffers, so a product that would wrap must
// be refused before it sizes an allocation: `2 * tokens * dk * sizeof(float)`
// wrapping to a small number is exactly how an under-allocation turns into a
// memcpy past the end.
static int tinytitan_checked_add_size(size_t left, size_t right, size_t *out) {
    if (right > SIZE_MAX - left) {
        return 0;
    }
    *out = left + right;
    return 1;
}

static int tinytitan_checked_mul_size(size_t left, size_t right, size_t *out) {
    if (left != 0 && right > SIZE_MAX / left) {
        return 0;
    }
    *out = left * right;
    return 1;
}

/// Bytes for `rows * columns` floats, checked as a product and then as a byte
/// count. Both halves are needed: the element product can fit while the byte
/// product does not.
static int tinytitan_scratch_bytes(size_t rows, size_t columns, size_t *out) {
    size_t elements = 0;
    if (!tinytitan_checked_mul_size(rows, columns, &elements)) {
        return 0;
    }
    if (elements > SIZE_MAX / sizeof(float)) {
        return 0;
    }
    *out = elements * sizeof(float);
    return 1;
}

static void tinytitan_delta_rule_chunk_sequential(const float *queries, size_t queryStride,
                                                  const float *keys, size_t keyStride,
                                                  const float *values, size_t valueStride,
                                                  const float *alphas, size_t alphaStride,
                                                  const float *betas, size_t betaStride,
                                                  size_t tokens, size_t dk, size_t dv,
                                                  float *state, float *readouts,
                                                  size_t readoutStride) {
    float correction[TINYTITAN_CHUNK_FALLBACK_DV];
    for (size_t t = 0; t < tokens; ++t) {
        const float alpha = alphas[t * alphaStride];
        const float beta = betas[t * betaStride];
        const float *keyRow = keys + t * keyStride;
        const float *queryRow = queries + t * queryStride;
        for (size_t i = 0; i < dv; ++i) {
            float *row = state + i * dk;
            float stored = 0.0f;
            for (size_t column = 0; column < dk; ++column) {
                row[column] *= alpha;
                stored += row[column] * keyRow[column];
            }
            correction[i] = beta * (values[t * valueStride + i] - stored);
            for (size_t column = 0; column < dk; ++column) {
                row[column] += keyRow[column] * correction[i];
            }
            float read = 0.0f;
            for (size_t column = 0; column < dk; ++column) {
                read += row[column] * queryRow[column];
            }
            readouts[t * readoutStride + i] = read;
        }
    }
}

/// The body of `tinytitan_delta_rule_chunk`, with the scratch allocator passed
/// in so the allocation-failure paths are testable without a real OOM.
static int delta_rule_chunk_impl(const float *queries, size_t queryStride,
                                 const float *keys, size_t keyStride,
                                 const float *values, size_t valueStride,
                                 const float *alphas, size_t alphaStride,
                                 const float *betas, size_t betaStride,
                                 size_t tokens, size_t dk, size_t dv,
                                 float *state, float *readouts, size_t readoutStride,
                                 tinytitan_scratch_alloc_fn allocate) {
    if (tokens == 0 || dk == 0 || dv == 0) {
        return TINYTITAN_DELTA_RULE_CHUNK_OK;
    }
    // A stride below the width it indexes makes every row overlap the next one,
    // which is a caller error rather than a geometry this kernel can honour.
    if (queryStride < dk || keyStride < dk || valueStride < dv
        || readoutStride < dv || alphaStride == 0 || betaStride == 0) {
        return TINYTITAN_DELTA_RULE_CHUNK_INVALID_GEOMETRY;
    }
    // Every allocation below is sized from these products; compute and check
    // them first so an impossibly large geometry is refused before the
    // allocator is asked for anything.
    size_t twoTokens = 0;
    if (!tinytitan_checked_add_size(tokens, tokens, &twoTokens)) {
        return TINYTITAN_DELTA_RULE_CHUNK_INVALID_GEOMETRY;
    }
    size_t keysQueriesBytes = 0;
    size_t productsBytes = 0;
    size_t gramBytes = 0;
    size_t correctionsBytes = 0;
    size_t weightedBytes = 0;
    size_t updateBytes = 0;
    size_t decayBytes = 0;
    if (!tinytitan_scratch_bytes(twoTokens, dk, &keysQueriesBytes)
        || !tinytitan_scratch_bytes(dv, twoTokens, &productsBytes)
        || !tinytitan_scratch_bytes(tokens, twoTokens, &gramBytes)
        || !tinytitan_scratch_bytes(tokens, dv, &correctionsBytes)
        || !tinytitan_scratch_bytes(dv, tokens, &weightedBytes)
        || !tinytitan_scratch_bytes(dv, dk, &updateBytes)
        || !tinytitan_scratch_bytes(tokens, 1, &decayBytes)) {
        return TINYTITAN_DELTA_RULE_CHUNK_INVALID_GEOMETRY;
    }
    // Every BLAS extent below is narrowed to `int`; `twoTokens` is both an
    // extent and a leading dimension. Reject before the cast.
    if (tokens > (size_t)INT_MAX / 2
        || dk > (size_t)INT_MAX
        || dv > (size_t)INT_MAX) {
        return TINYTITAN_DELTA_RULE_CHUNK_INVALID_GEOMETRY;
    }

    float *keysQueries = allocate(keysQueriesBytes);
    float *products = allocate(productsBytes);
    float *gram = allocate(gramBytes);
    float *corrections = allocate(correctionsBytes);
    float *weighted = allocate(weightedBytes);
    float *update = allocate(updateBytes);
    float *decay = allocate(decayBytes);
    if (keysQueries == NULL || products == NULL || gram == NULL || corrections == NULL
        || weighted == NULL || update == NULL || decay == NULL) {
        free(keysQueries);
        free(products);
        free(gram);
        free(corrections);
        free(weighted);
        free(update);
        free(decay);
        // The sequential fallback has a fixed stack buffer, so it can only
        // stand in when dv fits it. When it does, the result is complete and
        // valid and the caller may use it; when it does not, nothing has been
        // written and the caller must not use state or readouts.
        if (dv <= TINYTITAN_CHUNK_FALLBACK_DV) {
            tinytitan_delta_rule_chunk_sequential(
                queries, queryStride, keys, keyStride, values, valueStride,
                alphas, alphaStride, betas, betaStride, tokens, dk, dv, state,
                readouts, readoutStride);
            return TINYTITAN_DELTA_RULE_CHUNK_FELL_BACK;
        }
        return TINYTITAN_DELTA_RULE_CHUNK_ALLOCATION_FAILED;
    }

    // Contiguous [K ; Q] rows, because the source rows are strided by the
    // head layout.
    for (size_t t = 0; t < tokens; ++t) {
        memcpy(keysQueries + t * dk, keys + t * keyStride, dk * sizeof(float));
        memcpy(keysQueries + (tokens + t) * dk, queries + t * queryStride, dk * sizeof(float));
    }

    // A_t, the running product of the per-token decay.
    float product = 1.0f;
    for (size_t t = 0; t < tokens; ++t) {
        product *= alphas[t * alphaStride];
        decay[t] = product;
    }

    // products[:, t] = S_0 k_t and products[:, C+t] = S_0 q_t.
    tinytitan_sgemm_rowmajor(1, (int)dv, (int)twoTokens, (int)dk, state, keysQueries, products);
    // gram[s, t] = k_s.k_t and gram[s, C+t] = k_s.q_t.
    tinytitan_sgemm_rowmajor(
        1, (int)tokens, (int)twoTokens, (int)dk, keysQueries, keysQueries, gram);

    // The triangular pass: each token's correction and readout from S_0 and
    // the corrections that came before it.
    for (size_t t = 0; t < tokens; ++t) {
        const float previous = t > 0 ? decay[t - 1] : 1.0f;
        const float alpha = alphas[t * alphaStride];
        const float beta = betas[t * betaStride];
        const float *valueRow = values + t * valueStride;
        if (t > 0) {
            for (size_t i = 0; i < dv; ++i) {
                products[i * twoTokens + t] *= previous;
                products[i * twoTokens + tokens + t] *= previous;
            }
        }
        // A_{t-1}/A_s is the product of alpha_{s+1}..alpha_{t-1}, computed
        // backwards: dividing by A_s would divide by a product that can have
        // underflowed, and 0/0 would poison the state.
        float weight = 1.0f;
        for (size_t s = t; s-- > 0;) {
            if (s + 1 < t) {
                weight *= alphas[(s + 1) * alphaStride];
            }
            const float keyWeight = weight * gram[s * twoTokens + t];
            const float queryWeight = weight * gram[s * twoTokens + tokens + t];
            const float *correction = corrections + s * dv;
            for (size_t i = 0; i < dv; ++i) {
                products[i * twoTokens + t] += keyWeight * correction[i];
                products[i * twoTokens + tokens + t] += queryWeight * correction[i];
            }
        }
        const float keyQuery = gram[t * twoTokens + tokens + t];
        for (size_t i = 0; i < dv; ++i) {
            const float correction = beta * (valueRow[i] - alpha * products[i * twoTokens + t]);
            corrections[t * dv + i] = correction;
            readouts[t * readoutStride + i] =
                alpha * products[i * twoTokens + tokens + t] + correction * keyQuery;
        }
    }

    // S_chunk = A_chunk * S_0 + sum_s (A_chunk / A_s) c_s k_s^T. The low-rank
    // part is one GEMM; the state scaling and the add are elementwise.
    float weight = 1.0f;
    for (size_t s = tokens; s-- > 0;) {
        if (s + 1 < tokens) {
            weight *= alphas[(s + 1) * alphaStride];
        }
        for (size_t i = 0; i < dv; ++i) {
            weighted[i * tokens + s] = weight * corrections[s * dv + i];
        }
    }
    tinytitan_sgemm_rowmajor(0, (int)dv, (int)dk, (int)tokens, weighted, keysQueries, update);
    const float last = decay[tokens - 1];
    for (size_t i = 0; i < dv; ++i) {
        float *stateRow = state + i * dk;
        const float *updateRow = update + i * dk;
        for (size_t column = 0; column < dk; ++column) {
            stateRow[column] = last * stateRow[column] + updateRow[column];
        }
    }

    free(keysQueries);
    free(products);
    free(gram);
    free(corrections);
    free(weighted);
    free(update);
    free(decay);
    return TINYTITAN_DELTA_RULE_CHUNK_OK;
}

int tinytitan_delta_rule_chunk(const float *queries, size_t queryStride,
                               const float *keys, size_t keyStride,
                               const float *values, size_t valueStride,
                               const float *alphas, size_t alphaStride,
                               const float *betas, size_t betaStride,
                               size_t tokens, size_t dk, size_t dv,
                               float *state, float *readouts, size_t readoutStride) {
    return delta_rule_chunk_impl(
        queries, queryStride, keys, keyStride, values, valueStride,
        alphas, alphaStride, betas, betaStride, tokens, dk, dv, state,
        readouts, readoutStride, malloc);
}

int tinytitan_delta_rule_chunk_with_allocator(
    const float *queries, size_t queryStride,
    const float *keys, size_t keyStride,
    const float *values, size_t valueStride,
    const float *alphas, size_t alphaStride,
    const float *betas, size_t betaStride,
    size_t tokens, size_t dk, size_t dv,
    float *state, float *readouts, size_t readoutStride,
    tinytitan_scratch_alloc_fn allocate) {
    if (allocate == NULL) {
        return TINYTITAN_DELTA_RULE_CHUNK_INVALID_GEOMETRY;
    }
    return delta_rule_chunk_impl(
        queries, queryStride, keys, keyStride, values, valueStride,
        alphas, alphaStride, betas, betaStride, tokens, dk, dv, state,
        readouts, readoutStride, allocate);
}
