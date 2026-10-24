// Microbenchmark: batched CPU prefill GEMM shapes for the dense Qwen 3.5
// models, comparing Accelerate's BLAS (AMX/SME-backed on Apple Silicon) with a
// hand-written 4x4 NEON FP32 kernel.
//
// This is a measurement harness, not an engine path. It answers one question:
// if the CPU engine batched prompt tokens into GEMMs instead of decoding them
// one at a time, what would the linear layers cost, and how does that compare
// with today's per-token prefill?
//
// Build and run on this Mac:
//
//   clang -O3 -mcpu=native -framework Accelerate \
//       benchmark/cpu_prefill_gemm.c -o .build/cpu_prefill_gemm
//   .build/cpu_prefill_gemm
//
// The projection is for the linear projections/FFN only: attention, norms,
// the final head and sampling are excluded, so treat the projected TTFT as a
// lower bound. Decode rates are the README's CPU numbers for the 4-bit
// installs (2B 17.42 tok/s, 4B 7.54 tok/s), i.e. today a prompt is decoded
// token by token with its logits discarded.
//
// Optional raw AMX is deliberately not built here: the parked probe lives on
// the closed experiment branch and shipping its .word encodings is a separate
// decision. Accelerate is the public route and already dispatches to AMX on
// M1-M3 and SME on M4+.

#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    const char *name;
    int hidden;
    int q;
    int kv;
    int inter;
    int layers;
    double decode_tps;
} Model;

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void gemm_neon(int M, int N, int K, const float *A, const float *B, float *C) {
    int i = 0;
    for (; i + 4 <= M; i += 4) {
        int j = 0;
        for (; j + 4 <= N; j += 4) {
            float32x4_t c0 = vdupq_n_f32(0.0f);
            float32x4_t c1 = vdupq_n_f32(0.0f);
            float32x4_t c2 = vdupq_n_f32(0.0f);
            float32x4_t c3 = vdupq_n_f32(0.0f);
            const float *a0 = A + (size_t)(i + 0) * K;
            const float *a1 = A + (size_t)(i + 1) * K;
            const float *a2 = A + (size_t)(i + 2) * K;
            const float *a3 = A + (size_t)(i + 3) * K;
            for (int k = 0; k < K; ++k) {
                float32x4_t b = vld1q_f32(B + (size_t)k * N + j);
                c0 = vfmaq_n_f32(c0, b, a0[k]);
                c1 = vfmaq_n_f32(c1, b, a1[k]);
                c2 = vfmaq_n_f32(c2, b, a2[k]);
                c3 = vfmaq_n_f32(c3, b, a3[k]);
            }
            vst1q_f32(C + (size_t)(i + 0) * N + j, c0);
            vst1q_f32(C + (size_t)(i + 1) * N + j, c1);
            vst1q_f32(C + (size_t)(i + 2) * N + j, c2);
            vst1q_f32(C + (size_t)(i + 3) * N + j, c3);
        }
        for (; j < N; ++j) {
            float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            for (int k = 0; k < K; ++k) {
                float b = B[(size_t)k * N + j];
                c0 += A[(size_t)(i + 0) * K + k] * b;
                c1 += A[(size_t)(i + 1) * K + k] * b;
                c2 += A[(size_t)(i + 2) * K + k] * b;
                c3 += A[(size_t)(i + 3) * K + k] * b;
            }
            C[(size_t)(i + 0) * N + j] = c0;
            C[(size_t)(i + 1) * N + j] = c1;
            C[(size_t)(i + 2) * N + j] = c2;
            C[(size_t)(i + 3) * N + j] = c3;
        }
    }
    for (; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0;
            for (int k = 0; k < K; ++k) acc += A[(size_t)i * K + k] * B[(size_t)k * N + j];
            C[(size_t)i * N + j] = acc;
        }
    }
}

static void fill(float *p, size_t n) {
    for (size_t i = 0; i < n; ++i) p[i] = (float)((i % 97) - 48) * 0.001f;
}

static double median3(double a, double b, double c) {
    if (a > b) { double t = a; a = b; b = t; }
    if (b > c) { b = c; }
    return a > b ? a : b;
}

static double timed_blas(int M, int N, int K, const float *A, const float *B, float *C) {
    const int repeats = 3;
    double t[3];
    for (int r = 0; r < repeats; ++r) {
        double t0 = now_seconds();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B, N, 0.0f,
                    C, N);
        t[r] = now_seconds() - t0;
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B, N, 0.0f, C,
                N);
    return median3(t[0], t[1], t[2]);
}

static double timed_neon(int M, int N, int K, const float *A, const float *B, float *C) {
    const int repeats = 3;
    double t[3];
    for (int r = 0; r < repeats; ++r) {
        double t0 = now_seconds();
        gemm_neon(M, N, K, A, B, C);
        t[r] = now_seconds() - t0;
    }
    gemm_neon(M, N, K, A, B, C);
    return median3(t[0], t[1], t[2]);
}

// One layer at M prompt tokens: returns FLOPs and measured seconds per arm.
static double bench_shape(const char *label, int M, int K, int N, double *blas_s, double *neon_s) {
    size_t a_n = (size_t)M * K, b_n = (size_t)K * N, c_n = (size_t)M * N;
    float *A = malloc(a_n * sizeof(float));
    float *B = malloc(b_n * sizeof(float));
    float *C = malloc(c_n * sizeof(float));
    float *D = malloc(c_n * sizeof(float));
    if (!A || !B || !C || !D) { fprintf(stderr, "out of memory for %s\n", label); exit(1); }
    fill(A, a_n);
    fill(B, b_n);
    *blas_s = timed_blas(M, N, K, A, B, C);
    *neon_s = timed_neon(M, N, K, A, B, D);
    double max_err = 0;
    for (size_t i = 0; i < c_n; ++i) {
        double e = fabs((double)C[i] - (double)D[i]);
        if (e > max_err) max_err = e;
    }
    double flops = 2.0 * M * K * N;
    printf("  %-10s M=%d K=%d N=%d  BLAS %7.2f GF/s (%6.3f s)  NEON %7.2f GF/s (%6.3f s)"
           "  max|diff|=%.4g\n",
           label, M, K, N, flops / *blas_s / 1e9, *blas_s, flops / *neon_s / 1e9, *neon_s,
           max_err);
    free(A); free(B); free(C); free(D);
    return flops;
}

static void run_model(const Model *m, int M) {
    printf("\n%s: hidden=%d q=%d kv=%d inter=%d layers=%d, prompt=%d tokens\n", m->name,
           m->hidden, m->q, m->kv, m->inter, m->layers, M);
    double tb = 0, tn = 0, flops = 0, s;
    flops += bench_shape("qkv", M, m->hidden, m->q + 2 * m->kv, &s, &tn); tb += s;
    {
        double n;
        flops += bench_shape("o", M, m->q, m->hidden, &s, &n); tb += s; tn += n;
    }
    {
        double n;
        flops += bench_shape("gate+up", M, m->hidden, 2 * m->inter, &s, &n); tb += s; tn += n;
    }
    {
        double n;
        flops += bench_shape("down", M, m->inter, m->hidden, &s, &n); tb += s; tn += n;
    }
    double per_layer_flops = flops;
    printf("  layer total: BLAS %.2f GF/s (%.3f s)  NEON %.2f GF/s (%.3f s)\n",
           per_layer_flops / tb / 1e9, tb, per_layer_flops / tn / 1e9, tn);

    for (int target = 0; target < 2; ++target) {
        int prompt = target == 0 ? 2048 : 8192;
        double total_flops = per_layer_flops * m->layers * ((double)prompt / M);
        double blas_ttft = total_flops / (per_layer_flops / tb) ;
        double neon_ttft = total_flops / (per_layer_flops / tn);
        double today = prompt / m->decode_tps;
        printf("  projected %5d-token linear-layer prefill: BLAS %8.2f s (%.1fx vs today)"
               "  NEON %8.2f s (%.1fx); today ~%8.2f s\n",
               prompt, blas_ttft, today / blas_ttft, neon_ttft, today / neon_ttft, today);
    }
}

int main(void) {
    const Model models[] = {
        {"Qwen 3.5 2B 4-bit", 2048, 2048, 512, 6144, 24, 17.42},
        {"Qwen 3.5 4B 4-bit", 2560, 4096, 1024, 9216, 32, 7.54},
    };
    printf("CPU prefill GEMM microbenchmark (FP32, M=1024); projected TTFT is lower-bound:"
           " linear layers only\n");
    for (size_t i = 0; i < sizeof(models) / sizeof(models[0]); ++i) run_model(&models[i], 1024);
    return 0;
}
