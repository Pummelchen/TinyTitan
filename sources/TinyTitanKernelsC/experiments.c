#include "tinytitan_experiments.h"

#include <stdint.h>
#include <string.h>

#if defined(__aarch64__) && defined(__APPLE__)
#include <pthread.h>
#include <setjmp.h>
#include <signal.h>
#include <unistd.h>
#define TT_AMX_TARGET 1
#else
#define TT_AMX_TARGET 0
#endif

// `DC ZVA` zeroes one cache block and rounds an unaligned address down, so the
// alignment guard below is a correctness requirement, not an optimisation.
void tinytitan_zero_fill(void *dst, size_t n) {
    unsigned char *base = (unsigned char *)dst;
    if (base == NULL || n == 0) {
        return;
    }
#if TT_AMX_TARGET
    if (n >= 256 && (((uintptr_t)base & (uintptr_t)63) == 0)) {
        unsigned char *cursor = base;
        unsigned char *end = base + n;
        while ((size_t)(end - cursor) >= 64) {
            __asm__ __volatile__("dc zva, %0" :: "r"(cursor) : "memory");
            cursor += 64;
        }
        if (cursor < end) {
            memset(cursor, 0, (size_t)(end - cursor));
        }
        return;
    }
#endif
    memset(base, 0, n);
}

#if TT_AMX_TARGET

// The encodings are the reverse-engineered Apple AMX space (0x0020xxxx). They
// are deliberately confined to this file: no engine path calls them yet.
#define TT_AMX_SET() \
    __asm__ __volatile__("nop\n\tnop\n\tnop\n\t.word 0x00201220" ::: "memory")
#define TT_AMX_CLR() \
    __asm__ __volatile__("nop\n\tnop\n\tnop\n\t.word 0x00201221" ::: "memory")

// `operand` is a GPR holding the instruction's operand bits (pointer in the low
// 56 bits, flags above). The `0%0` dance bakes that register's number into the
// instruction word, which is what the encoding requires.
#define TT_AMX_OP(op, operand)                                                 \
    __asm__ __volatile__(                                                      \
        ".word (0x201000 + (" #op " << 5) + 0%0 - ((0%0 >> 4) * 6))"           \
        :: "r"((uint64_t)(operand))                                            \
        : "memory")

static sigjmp_buf tt_amx_jump;
static volatile sig_atomic_t tt_amx_armed = 0;
static pthread_once_t tt_amx_once = PTHREAD_ONCE_INIT;
static int tt_amx_result = 0;

static void tt_amx_on_sigill(int signal_number) {
    (void)signal_number;
    if (tt_amx_armed != 0) {
        tt_amx_armed = 0;
        siglongjmp(tt_amx_jump, 1);
    }
    // A SIGILL from anywhere else while the probe is installed must not be
    // swallowed: restore the default action and re-raise so the crash report
    // names the real fault rather than an exit code.
    (void)signal(SIGILL, SIG_DFL);
    (void)kill(getpid(), SIGILL);
    _exit(127);
}

static void tt_amx_probe(void) {
    struct sigaction action;
    struct sigaction previous;
    memset(&action, 0, sizeof action);
    action.sa_handler = tt_amx_on_sigill;
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;
    if (sigaction(SIGILL, &action, &previous) != 0) {
        return;
    }
    tt_amx_armed = 1;
    if (sigsetjmp(tt_amx_jump, 1) == 0) {
        TT_AMX_SET();
        TT_AMX_CLR();
        tt_amx_result = 1;
    }
    tt_amx_armed = 0;
    (void)sigaction(SIGILL, &previous, NULL);
}

int tinytitan_amx_available(void) {
    (void)pthread_once(&tt_amx_once, tt_amx_probe);
    return tt_amx_result;
}

int tinytitan_amx_sgemm16(const float *at, size_t lda,
                          const float *b, size_t ldb,
                          float *c, size_t ldc,
                          size_t m, size_t n, size_t k) {
    if (c == NULL || m == 0 || n == 0) {
        return -1;
    }
    if ((m % 16) != 0 || (n % 16) != 0) {
        return -1;
    }
    if (k > 0 && (at == NULL || b == NULL)) {
        return -1;
    }
    if (tinytitan_amx_available() == 0) {
        return -1;
    }
    if (k == 0) {
        for (size_t row = 0; row < m; ++row) {
            memset(c + row * ldc, 0, n * sizeof(float));
        }
        return 0;
    }

    TT_AMX_SET();
    for (size_t m0 = 0; m0 < m; m0 += 16) {
        for (size_t n0 = 0; n0 < n; n0 += 16) {
            for (size_t kk = 0; kk < k; ++kk) {
                uint64_t gx = (uint64_t)(uintptr_t)(at + kk * lda + m0);
                uint64_t gy = (uint64_t)(uintptr_t)(b + kk * ldb + n0);
                // Bank 0 only: one outer product per k, accumulated in Z. The
                // first product skips Z (bit 27) so the tile starts from zero
                // without clearing the whole AMX state per tile.
                uint64_t fma = (kk == 0) ? ((uint64_t)1 << 27) : (uint64_t)0;
                TT_AMX_OP(0, gx);
                TT_AMX_OP(1, gy);
                TT_AMX_OP(12, fma);
            }
            float tile[16] __attribute__((aligned(64)));
            for (size_t j = 0; j < 16; ++j) {
                // f32 matrix mode uses every fourth Z row: 0, 4, ..., 60.
                uint64_t gs =
                    (uint64_t)(uintptr_t)tile | (((uint64_t)(4 * j)) << 56);
                TT_AMX_OP(5, gs);
                for (size_t i = 0; i < 16; ++i) {
                    c[(m0 + i) * ldc + (n0 + j)] = tile[i];
                }
            }
        }
    }
    TT_AMX_CLR();
    return 0;
}

#else

int tinytitan_amx_available(void) {
    return 0;
}

int tinytitan_amx_sgemm16(const float *at, size_t lda,
                          const float *b, size_t ldb,
                          float *c, size_t ldc,
                          size_t m, size_t n, size_t k) {
    (void)at;
    (void)lda;
    (void)b;
    (void)ldb;
    (void)c;
    (void)ldc;
    (void)m;
    (void)n;
    (void)k;
    return -1;
}

#endif
