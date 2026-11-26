// Parallel routed-expert reader. See include/tinytitan_expert_io.h for why this
// exists: the device sustains ~3.2 GB/s on four concurrent expert-sized reads and
// v3.x was reaching ~0.62 GB/s by fetching one at a time on the calling thread.

#include "include/tinytitan_expert_io.h"
#include "include/tinytitan_testing.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define TINYTITAN_IO_MAX_THREADS 16

// The worker gate the lifecycle test installs, or NULL in every shipping run.
static tinytitan_worker_gate_fn tinytitan_worker_testing_gate = NULL;

// Strict C99 checked arithmetic for the read range. `pread` takes the offset as
// a signed `off_t`; a `uint64_t` that does not fit it, or an offset plus length
// that runs past `off_t`'s last byte, must be refused before the cast rather
// than handed to the kernel as a wrapped negative position.
static int tinytitan_checked_add_u64(uint64_t left, uint64_t right, uint64_t *out) {
    if (right > UINT64_MAX - left) {
        return 0;
    }
    *out = left + right;
    return 1;
}

static int tinytitan_checked_mul_u64(uint64_t left, uint64_t right, uint64_t *out) {
    if (left != 0 && right > UINT64_MAX / left) {
        return 0;
    }
    *out = left * right;
    return 1;
}

/// True when `value` round-trips through `off_t`, which is signed on every
/// POSIX target, so a value above `OFF_MAX` lands negative and is refused.
static int tinytitan_offset_fits_off_t(uint64_t value) {
    off_t converted = (off_t)value;
    return converted >= 0 && (uint64_t)converted == value;
}

/// 0 when `byte_offset` and the `length` bytes after it are all addressable,
/// otherwise `EOVERFLOW`.
///
/// Darwin's `pread` refuses a count above `INT_MAX` with `EINVAL` even though
/// POSIX allows up to `SSIZE_MAX`, so the count is bounded here and the failure
/// is typed instead of reaching the kernel. The last byte, not the one-past
/// end, is what `off_t` has to hold.
static int tinytitan_validate_read(uint64_t byte_offset, size_t length) {
    if (length == 0) {
        return 0;
    }
    if (length > (size_t)INT_MAX) {
        return EOVERFLOW;
    }
    if (!tinytitan_offset_fits_off_t(byte_offset)) {
        return EOVERFLOW;
    }
    uint64_t last = 0;
    if (!tinytitan_checked_add_u64(byte_offset, (uint64_t)length - 1, &last)) {
        return EOVERFLOW;
    }
    if (!tinytitan_offset_fits_off_t(last)) {
        return EOVERFLOW;
    }
    return 0;
}

struct tinytitan_expert_reader {
    int fds[TINYTITAN_IO_MAX_THREADS];
    int threads;
    size_t expert_stride;

    pthread_t workers[TINYTITAN_IO_MAX_THREADS];
    pthread_mutex_t lock;
    pthread_cond_t work_ready;   // a batch was published, or shutdown
    pthread_cond_t work_done;    // outstanding hit zero
    pthread_cond_t batch_idle;   // prior submitter cleared the published batch

    // Current batch. Valid only while outstanding > 0.
    const uint32_t *expert_ids;   // one of these three is non-NULL
    const uint64_t *offsets;
    const uint32_t *byte_counts;  // present only for range batches
    void *const *destinations;
    size_t count;
    size_t next_index;           // claimed by workers
    size_t outstanding;          // published minus completed
    int first_errno;
    int shutting_down;
    uint64_t generation;         // so a worker cannot re-run a finished batch
};

/// Reads one expert with `pread`, looping because a short read is legal. The
/// range is validated here as well as in the public entry points, so the
/// invariant is local to the call that hands the offset to the kernel.
static int read_one_with(int fd, void *dst, size_t stride, uint64_t byte_offset,
                         ssize_t (*read_fn)(int, void *, size_t, off_t)) {
    int range = tinytitan_validate_read(byte_offset, stride);
    if (range != 0) {
        return range;
    }
    unsigned char *out = (unsigned char *)dst;
    off_t base = (off_t)byte_offset;
    size_t done = 0;
    while (done < stride) {
        ssize_t got = read_fn(fd, out + done, stride - done, base + (off_t)done);
        if (got > 0) {
            done += (size_t)got;
            continue;
        }
        if (got == 0) {
            return EIO;               // short file: the caller's offsets are wrong
        }
        if (errno == EINTR) {
            continue;
        }
        return errno;
    }
    return 0;
}

static int read_one(int fd, void *dst, size_t stride, uint64_t byte_offset) {
    return read_one_with(fd, dst, stride, byte_offset, pread);
}

int tinytitan_expert_reader_testing_read_one(
    int fd, void *dst, size_t length, uint64_t byte_offset,
    ssize_t (*read_fn)(int, void *, size_t, off_t)) {
    if (read_fn == NULL) {
        return EINVAL;
    }
    return read_one_with(fd, dst, length, byte_offset, read_fn);
}

static void *worker_main(void *arg) {
    tinytitan_expert_reader *r = (tinytitan_expert_reader *)arg;
    // Each worker owns one descriptor, so index it by position in the pool.
    int slot = -1;
    pthread_mutex_lock(&r->lock);
    for (int i = 0; i < r->threads; ++i) {
        if (pthread_equal(r->workers[i], pthread_self())) {
            slot = i;
            break;
        }
    }
    pthread_mutex_unlock(&r->lock);
    if (slot < 0) {
        return NULL;
    }
    int fd = r->fds[slot];

    pthread_mutex_lock(&r->lock);
    for (;;) {
        while (!r->shutting_down && r->next_index >= r->count) {
            pthread_cond_wait(&r->work_ready, &r->lock);
        }
        if (r->shutting_down) {
            break;
        }
        size_t index = r->next_index++;
        size_t length = r->byte_counts != NULL
            ? (size_t)r->byte_counts[index]
            : r->expert_stride;
        uint64_t offset = r->offsets != NULL
            ? r->offsets[index]
            : (uint64_t)r->expert_ids[index] * (uint64_t)r->expert_stride;
        void *dst = r->destinations[index];
        pthread_mutex_unlock(&r->lock);

        // Testing seam: a lifecycle test pauses the pool at a known point.
        if (tinytitan_worker_testing_gate != NULL) {
            tinytitan_worker_testing_gate();
        }

        int rc = read_one(fd, dst, length, offset);

        pthread_mutex_lock(&r->lock);
        if (rc != 0 && r->first_errno == 0) {
            r->first_errno = rc;
        }
        if (--r->outstanding == 0) {
            pthread_cond_signal(&r->work_done);
        }
    }
    pthread_mutex_unlock(&r->lock);
    return NULL;
}

tinytitan_expert_reader *tinytitan_expert_reader_create(const char *path,
                                               size_t expert_stride,
                                               int threads,
                                               int bypass_cache,
                                               int *out_errno) {
    if (out_errno) {
        *out_errno = 0;
    }
    if (path == NULL || expert_stride == 0) {
        if (out_errno) { *out_errno = EINVAL; }
        return NULL;
    }
    if (threads < 1) { threads = 1; }
    if (threads > TINYTITAN_IO_MAX_THREADS) { threads = TINYTITAN_IO_MAX_THREADS; }

    tinytitan_expert_reader *r = (tinytitan_expert_reader *)calloc(1, sizeof(*r));
    if (r == NULL) {
        if (out_errno) { *out_errno = ENOMEM; }
        return NULL;
    }
    r->expert_stride = expert_stride;
    r->threads = threads;
    for (int i = 0; i < TINYTITAN_IO_MAX_THREADS; ++i) {
        r->fds[i] = -1;
    }

    // One descriptor per worker: pread does not use the shared offset, but
    // separate descriptors keep the kernel's per-fd state uncontended.
    for (int i = 0; i < threads; ++i) {
        r->fds[i] = open(path, O_RDONLY);
        if (r->fds[i] >= 0 && bypass_cache) {
            // Bounded memory is a correctness contract. Silently continuing
            // without F_NOCACHE would create an undeclared page-cache working
            // set, so an unsupported descriptor must fail the reader.
            if (fcntl(r->fds[i], F_NOCACHE, 1) != 0) {
                int err = errno;
                close(r->fds[i]);
                r->fds[i] = -1;
                for (int j = 0; j < i; ++j) { close(r->fds[j]); }
                free(r);
                if (out_errno) { *out_errno = err; }
                return NULL;
            }
        }
        if (r->fds[i] < 0) {
            int err = errno;
            for (int j = 0; j < i; ++j) { close(r->fds[j]); }
            free(r);
            if (out_errno) { *out_errno = err; }
            return NULL;
        }
    }

    if (pthread_mutex_init(&r->lock, NULL) != 0) {
        goto sync_init_failed;
    }
    if (pthread_cond_init(&r->work_ready, NULL) != 0) {
        pthread_mutex_destroy(&r->lock);
        goto sync_init_failed;
    }
    if (pthread_cond_init(&r->work_done, NULL) != 0) {
        pthread_cond_destroy(&r->work_ready);
        pthread_mutex_destroy(&r->lock);
        goto sync_init_failed;
    }
    if (pthread_cond_init(&r->batch_idle, NULL) != 0) {
        pthread_cond_destroy(&r->work_done);
        pthread_cond_destroy(&r->work_ready);
        pthread_mutex_destroy(&r->lock);
        goto sync_init_failed;
    }

    // Publish thread identities under the lock before any worker looks for its
    // own slot, otherwise a fast worker can fail to find itself.
    pthread_mutex_lock(&r->lock);
    int started = 0;
    for (int i = 0; i < threads; ++i) {
        if (pthread_create(&r->workers[i], NULL, worker_main, r) != 0) {
            break;
        }
        started++;
    }
    if (started < threads) {
        // Bring up whatever started, then fail: a partial pool would silently
        // read at a fraction of the requested rate.
        r->shutting_down = 1;
        r->threads = started;
        pthread_cond_broadcast(&r->work_ready);
        pthread_mutex_unlock(&r->lock);
        for (int i = 0; i < started; ++i) { pthread_join(r->workers[i], NULL); }
        for (int j = 0; j < threads; ++j) { close(r->fds[j]); }
        pthread_cond_destroy(&r->work_ready);
        pthread_cond_destroy(&r->work_done);
        pthread_cond_destroy(&r->batch_idle);
        pthread_mutex_destroy(&r->lock);
        free(r);
        if (out_errno) { *out_errno = EAGAIN; }
        return NULL;
    }
    pthread_mutex_unlock(&r->lock);
    return r;

sync_init_failed:
    for (int j = 0; j < threads; ++j) { close(r->fds[j]); }
    free(r);
    if (out_errno) { *out_errno = ENOMEM; }
    return NULL;
}

void tinytitan_expert_reader_destroy(tinytitan_expert_reader *r) {
    if (r == NULL) {
        return;
    }
    pthread_mutex_lock(&r->lock);
    r->shutting_down = 1;
    pthread_cond_broadcast(&r->work_ready);
    pthread_mutex_unlock(&r->lock);
    for (int i = 0; i < r->threads; ++i) {
        pthread_join(r->workers[i], NULL);
    }
    for (int i = 0; i < TINYTITAN_IO_MAX_THREADS; ++i) {
        if (r->fds[i] >= 0) { close(r->fds[i]); }
    }
    pthread_cond_destroy(&r->work_ready);
    pthread_cond_destroy(&r->work_done);
    pthread_cond_destroy(&r->batch_idle);
    pthread_mutex_destroy(&r->lock);
    free(r);
}

static int submit_batch(tinytitan_expert_reader *r,
                        const uint32_t *expert_ids,
                        const uint64_t *offsets,
                        const uint32_t *byte_counts,
                        void *const *destinations,
                        size_t count) {
    pthread_mutex_lock(&r->lock);
    // A caller owns the published pointers until its workers finish and it
    // clears the batch below. Without this predicate a second caller could
    // overwrite those pointers while the first caller was waiting, corrupting
    // destinations and leaving both callers blocked on the wrong outstanding
    // count.
    while (r->count > 0 && !r->shutting_down) {
        pthread_cond_wait(&r->batch_idle, &r->lock);
    }
    if (r->shutting_down) {
        pthread_mutex_unlock(&r->lock);
        return ECANCELED;
    }
    r->offsets = offsets;
    r->expert_ids = expert_ids;
    r->byte_counts = byte_counts;
    r->destinations = destinations;
    r->count = count;
    r->next_index = 0;
    r->outstanding = count;
    r->first_errno = 0;
    r->generation++;
    pthread_cond_broadcast(&r->work_ready);
    while (r->outstanding > 0) {
        pthread_cond_wait(&r->work_done, &r->lock);
    }
    int rc = r->first_errno;
    // Leave the batch empty so idle workers block instead of spinning on a
    // consumed batch.
    r->count = 0;
    r->next_index = 0;
    r->expert_ids = NULL;
    r->offsets = NULL;
    r->byte_counts = NULL;
    r->destinations = NULL;
    pthread_cond_broadcast(&r->batch_idle);
    pthread_mutex_unlock(&r->lock);
    return rc;
}

int tinytitan_expert_reader_fetch(tinytitan_expert_reader *r,
                              const uint32_t *expert_ids,
                              void *const *destinations,
                              size_t count) {
    if (r == NULL || (count > 0 && (expert_ids == NULL || destinations == NULL))) {
        return EINVAL;
    }
    if (count == 0) { return 0; }
    // The id form turns an index into an absolute position here, so the product
    // and the span it addresses are checked before a batch is published. An
    // unguarded product can wrap to a small offset and read the wrong expert.
    for (size_t i = 0; i < count; ++i) {
        uint64_t offset = 0;
        if (!tinytitan_checked_mul_u64(
                (uint64_t)expert_ids[i], (uint64_t)r->expert_stride, &offset)) {
            return EOVERFLOW;
        }
        int range = tinytitan_validate_read(offset, r->expert_stride);
        if (range != 0) {
            return range;
        }
    }
    return submit_batch(r, expert_ids, NULL, NULL, destinations, count);
}

int tinytitan_expert_reader_fetch_offsets(tinytitan_expert_reader *r,
                                     const uint64_t *offsets,
                                     void *const *destinations,
                                     size_t count) {
    if (r == NULL || (count > 0 && (offsets == NULL || destinations == NULL))) {
        return EINVAL;
    }
    if (count == 0) { return 0; }
    for (size_t i = 0; i < count; ++i) {
        int range = tinytitan_validate_read(offsets[i], r->expert_stride);
        if (range != 0) {
            return range;
        }
    }
    return submit_batch(r, NULL, offsets, NULL, destinations, count);
}

int tinytitan_expert_reader_fetch_ranges(tinytitan_expert_reader *r,
                                    const uint64_t *offsets,
                                    const uint32_t *byte_counts,
                                    void *const *destinations,
                                    size_t count) {
    if (r == NULL
        || (count > 0
            && (offsets == NULL || byte_counts == NULL || destinations == NULL))) {
        return EINVAL;
    }
    for (size_t i = 0; i < count; ++i) {
        if (byte_counts[i] == 0) {
            return EINVAL;
        }
        int range = tinytitan_validate_read(offsets[i], (size_t)byte_counts[i]);
        if (range != 0) {
            return range;
        }
    }
    if (count == 0) { return 0; }
    return submit_batch(r, NULL, offsets, byte_counts, destinations, count);
}


int tinytitan_expert_reader_threads(const tinytitan_expert_reader *r) {
    return r == NULL ? 0 : r->threads;
}

void tinytitan_expert_reader_testing_set_gate(tinytitan_worker_gate_fn gate) {
    tinytitan_worker_testing_gate = gate;
}
