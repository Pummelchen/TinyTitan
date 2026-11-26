#ifndef TINYTITAN_TESTING_H
#define TINYTITAN_TESTING_H

// Fault-injection seams for the C kernels' tests.
//
// Nothing in the shipping runtime calls these: the production entry points use
// `malloc` and `pread` directly, and the defaults here are those functions.
// They exist because the failure paths they exercise cannot be reached with a
// real allocation or a real syscall on a small, safe fixture:
//
// - refusing a scratch allocation needs an allocator that returns NULL, since
//   any geometry that would make `malloc` fail also makes the fallback write
//   gigabytes;
// - a retried `EINTR` needs a `pread` that returns it, since a regular file
//   read on APFS does not.
//
// A test that sets a seam must restore it before it returns. The gate and the
// allocator are not safe to change while a call is in flight.

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

/// The allocator used by `tinytitan_delta_rule_chunk_with_allocator`.
typedef void *(*tinytitan_scratch_alloc_fn)(size_t bytes);

/// As `tinytitan_delta_rule_chunk`, but every scratch allocation comes from
/// `allocate` instead of `malloc`, so a test can fail it deterministically.
/// The geometry checks and the return codes are exactly the production ones.
/// `allocate` must not be NULL.
int tinytitan_delta_rule_chunk_with_allocator(
    const float *queries, size_t queryStride,
    const float *keys, size_t keyStride,
    const float *values, size_t valueStride,
    const float *alphas, size_t alphaStride,
    const float *betas, size_t betaStride,
    size_t tokens, size_t dk, size_t dv,
    float *state, float *readouts, size_t readoutStride,
    tinytitan_scratch_alloc_fn allocate);

/// `read_one`, the reader's per-item loop, with an injectable `pread`.
///
/// It validates `byte_offset`/`length` exactly as the worker path does, then
/// loops over `read_fn` handling a short count and `EINTR`. Returns 0 or an
/// `errno` value. `read_fn` must not be NULL.
int tinytitan_expert_reader_testing_read_one(
    int fd, void *dst, size_t length, uint64_t byte_offset,
    ssize_t (*read_fn)(int, void *, size_t, off_t));

/// A function the worker calls after claiming an item and before reading it.
typedef void (*tinytitan_worker_gate_fn)(void);

/// Installs (or clears, with NULL) the worker gate, so a lifecycle test can
/// pause the pool at a known point instead of sleeping. Set it before a fetch
/// starts and clear it after every fetch has returned.
void tinytitan_expert_reader_testing_set_gate(tinytitan_worker_gate_fn gate);

#endif /* TINYTITAN_TESTING_H */
