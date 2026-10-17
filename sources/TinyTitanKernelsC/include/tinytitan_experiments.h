#ifndef TINYTITAN_EXPERIMENTS_H
#define TINYTITAN_EXPERIMENTS_H

#include <stddef.h>

/// Whether the undocumented Apple AMX instruction set executes on this core.
///
/// Probes once per process by issuing `AMX_SET` / `AMX_CLR` behind a `SIGILL`
/// handler: 1 means the instructions ran, 0 means they faulted (or this is not
/// Apple arm64). The probe installs and restores the handler under
/// `pthread_once`, so concurrent callers are safe, but the result is cached for
/// the process lifetime — a hypervisor or a later OS change cannot flip it.
///
/// AMX state is per thread, and `AMX_SET` faults when the thread already has it
/// enabled. Call this before handing the thread to Accelerate; a probe that
/// runs while a library holds AMX reports 0 falsely.
int tinytitan_amx_available(void);

/// Experimental fp32 GEMM using AMX: `c = a * b`.
///
/// `at` is **A transposed**, `at[k * lda + row]`, because AMX loads a column of
/// A as one 64-byte vector. `b` is row-major `b[k * ldb + col]`. `m` and `n`
/// must be multiples of 16 (one outer-product tile); `k` may be any value,
/// including 0, which yields a zero matrix. Returns 0 on success, -1 when AMX
/// is unavailable or a dimension is unsupported.
///
/// This is an experiment, not a supported kernel: it does not save or restore
/// AMX state (the caller owns the thread), it assumes no other library has AMX
/// enabled, and it is not wired into any engine path.
int tinytitan_amx_sgemm16(const float *at, size_t lda,
                          const float *b, size_t ldb,
                          float *c, size_t ldc,
                          size_t m, size_t n, size_t k);

/// Zeroes `n` bytes at `dst`, using `DC ZVA` when the address is 64-byte
/// aligned.
///
/// `DC ZVA` zeroes one cache block, which on Apple M-series is **64 bytes**, not
/// the 128-byte line `hw.cachelinesize` reports. Striding by 128 would leave
/// half the range dirty, so this walks in 64-byte steps and falls back to
/// `memset` for a misaligned base or a short range (a `DC ZVA` on an unaligned
/// address rounds down and would zero bytes before `dst`).
void tinytitan_zero_fill(void *dst, size_t n);

#endif
