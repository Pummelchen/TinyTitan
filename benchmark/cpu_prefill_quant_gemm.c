// Phase-0 benchmark for a batched CPU prefill on the quantized path.
//
// cpu_prefill_gemm.c measured FP32 GEMM shapes and answered "is the matrix
// engine worth using at all?". It did not answer the question the engine
// actually faces: the CPU engine reads 4/8-bit affine weights through
// tinytitan_int{4,8}_affine_gemv, one token at a time, and a batched path has
// to get those weights into a GEMM without paying more for the conversion than
// the batching saves.
//
// So this harness compares two arms on the shapes the dense Qwen 3.5 models
// really use, at the real quantization widths:
//
//   A. today: the shipping int8/int4 GEMV called once per prompt token, with
//      the weight matrix streamed in full for every token (row-split across
//      the performance cores, as Int8AffineGEMV.threaded does).
//   B. batched: dequantize the weight matrix to FP32 once, then one
//      cblas_sgemm over all M tokens (the public Accelerate path, AMX on
//      M1-M3 and SME on M4+).
//
// Both arms cover the linear projections and FFN of one layer only --
// attention, norms, the final head and sampling are excluded, exactly as in
// cpu_prefill_gemm.c -- so the projected prompt times are a lower bound.
//
// Arm B's dequantization is charged to the same run. It is per prompt, so the
// M=512 numbers below overstate its cost for a 2K/8K prompt: the state in the
// report is the measured one, not the amortized one.
//
// Build and run:
//
//   clang -O3 -mcpu=native -fblocks -framework Accelerate \
//       -I sources/TinyTitanKernelsC/include \
//       benchmark/cpu_prefill_quant_gemm.c \
//       sources/TinyTitanKernelsC/int8_affine_gemv.c \
//       sources/TinyTitanKernelsC/int4_affine_gemv.c \
//       -o .build/cpu_prefill_quant_gemm
//   .build/cpu_prefill_quant_gemm [tokens]

#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <dispatch/dispatch.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>
#include <time.h>

#include "tinytitan_kernels.h"

#define GROUP 64
#define MAX_PROJECTIONS 4

typedef struct {
    const char *name;
    int hidden, q, kv, inter, layers;
} Model;

typedef struct {
    const char *name;
    int rows, n;
} Projection;

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static float bf16_to_float(uint16_t bits) {
    uint32_t u = (uint32_t)bits << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static uint16_t float_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return (uint16_t)(u >> 16);
}

static uint64_t rng_state = 0x9e3779b97f4a7c15ull;

static uint32_t rng(void) {
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng_state = x;
    return (uint32_t)(x >> 16);
}

static void *allocate(size_t bytes) {
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0 || p == NULL) {
        fprintf(stderr, "allocation of %zu bytes failed\n", bytes);
        exit(1);
    }
    memset(p, 0, bytes);
    return p;
}

static int performance_threads(void) {
    int32_t count = 0;
    size_t size = sizeof(count);
    if (sysctlbyname("hw.perflevel0.logicalcpu", &count, &size, NULL, 0) == 0 && count > 0) {
        return count;
    }
    return 4;
}

// Convert one 64-element group of unsigned bytes with the group's affine
// pair. This is the same `q * scale + bias` the GEMV kernel applies.
static inline void dequant_group_u8(const uint8_t *q, float *out, float scale, float bias) {
    const float32x4_t sv = vdupq_n_f32(scale);
    const float32x4_t bv = vdupq_n_f32(bias);
    for (int k = 0; k < GROUP; k += 16) {
        const uint8x16_t bytes = vld1q_u8(q + k);
        const uint16x8_t lo = vmovl_u8(vget_low_u8(bytes));
        const uint16x8_t hi = vmovl_u8(vget_high_u8(bytes));
        const float32x4_t q0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo)));
        const float32x4_t q1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo)));
        const float32x4_t q2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi)));
        const float32x4_t q3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi)));
        vst1q_f32(out + k + 0, vfmaq_f32(bv, q0, sv));
        vst1q_f32(out + k + 4, vfmaq_f32(bv, q1, sv));
        vst1q_f32(out + k + 8, vfmaq_f32(bv, q2, sv));
        vst1q_f32(out + k + 12, vfmaq_f32(bv, q3, sv));
    }
}

static void dequant8_rows(const uint8_t *weights, const uint16_t *scales,
                          const uint16_t *biases, int rows, int n, float *out,
                          int first, int last) {
    const int groups = n / GROUP;
    for (int r = first; r < last; ++r) {
        const uint8_t *wr = weights + (size_t)r * n;
        const uint16_t *sr = scales + (size_t)r * groups;
        const uint16_t *br = biases + (size_t)r * groups;
        float *orow = out + (size_t)r * n;
        for (int g = 0; g < groups; ++g) {
            dequant_group_u8(wr + g * GROUP, orow + g * GROUP,
                             bf16_to_float(sr[g]), bf16_to_float(br[g]));
        }
    }
}

// 4-bit packs element 2k in the low half of byte k and 2k+1 in the high half.
// Expand to bytes first (low, high, low, high...) then reuse the 8-bit path.
static void dequant4_rows(const uint8_t *weights, const uint16_t *scales,
                          const uint16_t *biases, int rows, int n, float *out,
                          int first, int last) {
    const int groups = n / GROUP;
    for (int r = first; r < last; ++r) {
        const uint8_t *wr = weights + (size_t)r * (n / 2);
        const uint16_t *sr = scales + (size_t)r * groups;
        const uint16_t *br = biases + (size_t)r * groups;
        float *orow = out + (size_t)r * n;
        for (int g = 0; g < groups; ++g) {
            const uint8_t *packed = wr + g * (GROUP / 2);
            uint8_t bytes[GROUP];
            for (int k = 0; k < GROUP / 2; k += 16) {
                const uint8x16_t p = vld1q_u8(packed + k);
                const uint8x16_t lo = vandq_u8(p, vdupq_n_u8(0x0F));
                const uint8x16_t hi = vshrq_n_u8(p, 4);
                vst1q_u8(bytes + k * 2, vzip1q_u8(lo, hi));
                vst1q_u8(bytes + k * 2 + 16, vzip2q_u8(lo, hi));
            }
            dequant_group_u8(bytes, orow + g * GROUP,
                             bf16_to_float(sr[g]), bf16_to_float(br[g]));
        }
    }
}

static void dequant_rows(int bits, const uint8_t *weights, const uint16_t *scales,
                         const uint16_t *biases, int rows, int n, float *out,
                         int threads) {
    if (threads <= 1) {
        if (bits == 4) {
            dequant4_rows(weights, scales, biases, rows, n, out, 0, rows);
        } else {
            dequant8_rows(weights, scales, biases, rows, n, out, 0, rows);
        }
        return;
    }
    const int chunk = (rows + threads - 1) / threads;
    dispatch_apply((size_t)threads, DISPATCH_APPLY_AUTO, ^(size_t i) {
        const int first = (int)i * chunk;
        const int last = first + chunk < rows ? first + chunk : rows;
        if (first >= rows) {
            return;
        }
        if (bits == 4) {
            dequant4_rows(weights, scales, biases, rows, n, out, first, last);
        } else {
            dequant8_rows(weights, scales, biases, rows, n, out, first, last);
        }
    });
}

// Arm A: the shipping kernel, run once per token with the rows split across
// the performance cores (Int8AffineGEMV.threaded's shape).
static void gemv_rows(int bits, const uint8_t *weights, const uint16_t *scales,
                      const uint16_t *biases, const float *x, int rows, int n,
                      float *out, int threads) {
    const int groups = n / GROUP;
    const int chunk = (rows + threads - 1) / threads;
    if (threads <= 1) {
        if (bits == 4) {
            tinytitan_int4_affine_gemv(weights, scales, biases, x, (size_t)rows, (size_t)n, out);
        } else {
            tinytitan_int8_affine_gemv(weights, scales, biases, x, (size_t)rows, (size_t)n, out);
        }
        return;
    }
    dispatch_apply((size_t)threads, DISPATCH_APPLY_AUTO, ^(size_t i) {
        const int first = (int)i * chunk;
        const int last = first + chunk < rows ? first + chunk : rows;
        if (first >= rows) {
            return;
        }
        const size_t count = (size_t)(last - first);
        if (bits == 4) {
            tinytitan_int4_affine_gemv(weights + (size_t)first * (n / 2),
                                       scales + (size_t)first * groups,
                                       biases + (size_t)first * groups,
                                       x, count, (size_t)n, out + first);
        } else {
            tinytitan_int8_affine_gemv(weights + (size_t)first * n,
                                       scales + (size_t)first * groups,
                                       biases + (size_t)first * groups,
                                       x, count, (size_t)n, out + first);
        }
    });
}

static void fill_weights(uint8_t *weights, uint16_t *scales, uint16_t *biases,
                         size_t packed_bytes, int entries) {
    for (size_t i = 0; i < packed_bytes; ++i) {
        weights[i] = (uint8_t)rng();
    }
    // A small positive scale keeps the accumulated values in a realistic
    // range; bias is zero, as the converter leaves it for symmetric groups.
    for (int i = 0; i < entries; ++i) {
        scales[i] = float_to_bf16(0.0125f);
        biases[i] = float_to_bf16(0.0f);
    }
}

static void run_model(const Model *model, int tokens, int bits, int threads) {
    Projection projections[MAX_PROJECTIONS] = {
        {"qkv", model->q + 2 * model->kv, model->hidden},
        {"o", model->hidden, model->q},
        {"gate_up", 2 * model->inter, model->hidden},
        {"down", model->hidden, model->inter},
    };

    double total_a = 0.0;
    double total_dequant = 0.0;
    double total_b = 0.0;
    double worst_diff = 0.0;
    double peak = 0.0;

    printf("\n== %s, %d-bit, M=%d tokens, %d threads ==\n", model->name, bits,
           tokens, threads);
    printf("%-8s %7s %7s %10s %10s %10s %10s %9s %10s\n", "proj", "rows", "n",
           "gemv_s", "dequant_s", "sgemm_s", "batched_s", "speedup", "maxdiff");

    for (int p = 0; p < MAX_PROJECTIONS; ++p) {
        const int rows = projections[p].rows;
        const int n = projections[p].n;
        const int groups = n / GROUP;
        const size_t packed = bits == 4 ? (size_t)rows * n / 2 : (size_t)rows * n;

        uint8_t *weights = allocate(packed);
        uint16_t *scales = allocate((size_t)rows * groups * sizeof(uint16_t));
        uint16_t *biases = allocate((size_t)rows * groups * sizeof(uint16_t));
        float *xf = allocate((size_t)tokens * n * sizeof(float));
        float *ya = allocate((size_t)tokens * rows * sizeof(float));
        float *yb = allocate((size_t)tokens * rows * sizeof(float));
        float *wf = allocate((size_t)rows * n * sizeof(float));

        fill_weights(weights, scales, biases, packed, rows * groups);
        for (size_t i = 0; i < (size_t)tokens * n; ++i) {
            xf[i] = ((float)(rng() % 2001) - 1000.0f) / 10000.0f;
        }

        const double a_start = now_seconds();
        for (int t = 0; t < tokens; ++t) {
            gemv_rows(bits, weights, scales, biases, xf + (size_t)t * n, rows, n,
                      ya + (size_t)t * rows, threads);
        }
        const double a_seconds = now_seconds() - a_start;

        const double d_start = now_seconds();
        dequant_rows(bits, weights, scales, biases, rows, n, wf, threads);
        const double dequant_seconds = now_seconds() - d_start;

        // C[M x rows] = X[M x n] * W^T, W stored row-major as rows x n.
        const double g_start = now_seconds();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, tokens, rows, n, 1.0f,
                    xf, n, wf, n, 0.0f, yb, rows);
        const double sgemm_seconds = now_seconds() - g_start;
        const double batched_seconds = dequant_seconds + sgemm_seconds;

        for (size_t i = 0; i < (size_t)tokens * rows; ++i) {
            const double diff = fabs((double)ya[i] - (double)yb[i]);
            const double mag = fabs((double)ya[i]);
            if (diff > worst_diff) {
                worst_diff = diff;
            }
            if (mag > peak) {
                peak = mag;
            }
        }

        total_a += a_seconds;
        total_dequant += dequant_seconds;
        total_b += batched_seconds;

        printf("%-8s %7d %7d %10.4f %10.4f %10.4f %10.4f %8.1fx %10.3g\n",
               projections[p].name, rows, n, a_seconds, dequant_seconds,
               sgemm_seconds, batched_seconds, a_seconds / batched_seconds,
               worst_diff);

        free(weights);
        free(scales);
        free(biases);
        free(xf);
        free(ya);
        free(yb);
        free(wf);
    }

    // The loop above ran one layer, so a whole-model token is that sum times
    // the layer count. (The first version of this summary forgot the factor
    // and printed a per-layer number as a per-token one.)
    const double per_token_a_ms = total_a * 1000.0 / tokens * model->layers;
    const double per_token_b_ms = total_b * 1000.0 / tokens * model->layers;
    const double per_layer_a_ms = total_a * 1000.0 / tokens;
    const double per_layer_b_ms = total_b * 1000.0 / tokens;

    printf("one layer: gemv %.3f ms/token, dequant+sgemm %.3f ms/token\n",
           per_layer_a_ms, per_layer_b_ms);
    printf("all %d layers, projections only: gemv %.1f ms/token, "
           "dequant+sgemm %.1f ms/token, speedup %.1fx\n",
           model->layers, per_token_a_ms, per_token_b_ms, total_a / total_b);
    printf("projected prompt (linear layers only): 2K %.1f s -> %.1f s, "
           "8K %.1f s -> %.1f s\n",
           per_token_a_ms * 2048 / 1000.0, per_token_b_ms * 2048 / 1000.0,
           per_token_a_ms * 8192 / 1000.0, per_token_b_ms * 8192 / 1000.0);
    printf("arm B dequant share: %.1f%%, worst |A-B| %.3g against peak %.3g\n",
           100.0 * total_dequant / total_b, worst_diff, peak);
}

int main(int argc, char **argv) {
    const int tokens = argc > 1 ? atoi(argv[1]) : 512;
    if (tokens <= 0) {
        fprintf(stderr, "usage: %s [tokens]\n", argv[0]);
        return 2;
    }
    const int threads = performance_threads();
    const Model models[] = {
        {"Qwen 3.5 2B (dense)", 2048, 2048, 512, 6144, 24},
        {"Qwen 3.5 4B (dense)", 2560, 4096, 1024, 9216, 32},
    };
    printf("CPU prefill quantization benchmark: shipping per-token GEMV vs "
           "dequantize+Accelerate, projections and FFN only.\n");
    for (size_t m = 0; m < sizeof(models) / sizeof(models[0]); ++m) {
        for (int bits = 8; bits >= 4; bits -= 4) {
            run_model(&models[m], tokens, bits, threads);
        }
    }
    return 0;
}
