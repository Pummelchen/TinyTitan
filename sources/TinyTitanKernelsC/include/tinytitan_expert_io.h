#ifndef TINYTITAN_EXPERT_IO_H
#define TINYTITAN_EXPERT_IO_H

#include <stddef.h>
#include <stdint.h>

/// Parallel reader for routed-expert blocks.
///
/// v3.x fetched misses one at a time on the calling thread and reached ~0.62 GB/s.
/// Measured on the same machine and the same file, expert-sized (1.688 MiB) reads
/// sustain ~2.0 GB/s on one thread and ~3.2 GB/s on four, and random offsets are
/// as fast as sequential ones -- NVMe does not care about locality at this
/// granularity, so nothing needs reordering. The gap was concurrency, not the
/// device.
///
/// Reads go through `pread` on per-thread descriptors: no shared file offset, no
/// lseek, and no mmap, so a fetch cannot fault the GPU's address space or leave
/// page-cache pressure behind for the rest of the machine to pay. That matters
/// here because the point of streaming is to keep RAM free.
typedef struct tinytitan_expert_reader tinytitan_expert_reader;

/// Opens `path` with `threads` independent descriptors.
///
/// Returns NULL if the file cannot be opened or the arguments are invalid;
/// `out_errno` receives the failure cause when non-NULL. `threads` is clamped to
/// [1, 16] -- four saturates this device and more does not help.
/// `bypass_cache` sets `F_NOCACHE`, keeping expert reads out of the unified
/// buffer cache.
///
/// This is a policy choice, not an optimisation. With the page cache in play,
/// repeated reads of a hot expert are served from RAM and are very fast -- but the
/// cache grows without bound and consumes exactly the memory this project exists to
/// leave free. With it bypassed, the slot cache is the *only* cache, so the
/// declared RAM budget is the true footprint and throughput is the device's.
///
/// Default to bypassing for a predictable footprint; allow the page cache when the
/// caller would rather have the free speed.
tinytitan_expert_reader *tinytitan_expert_reader_create(const char *path,
                                               size_t expert_stride,
                                               int threads,
                                               int bypass_cache,
                                               int *out_errno);

void tinytitan_expert_reader_destroy(tinytitan_expert_reader *reader);

/// Reads `count` experts, `expert_ids[i]` into `destinations[i]`, and blocks
/// until every one has completed.
///
/// Each destination must hold `expert_stride` bytes. Returns 0 on success, or the
/// first `errno` observed by any worker; on failure the contents of the
/// destinations are undefined and the caller must not use them.
int tinytitan_expert_reader_fetch(tinytitan_expert_reader *reader,
                              const uint32_t *expert_ids,
                              void *const *destinations,
                              size_t count);

/// As `tinytitan_expert_reader_fetch`, but the caller supplies absolute byte offsets.
///
/// Callers that lay experts out as `index * stride` can use the id form; the
/// streamer cannot, because its regions carry a per-layer base and a container
/// offset, so an id would silently address the wrong layer.
int tinytitan_expert_reader_fetch_offsets(tinytitan_expert_reader *reader,
                                     const uint64_t *offsets,
                                     void *const *destinations,
                                     size_t count);

/// As `tinytitan_expert_reader_fetch_offsets`, but item `i` reads
/// `byte_counts[i]` bytes from `offsets[i]` into the *front* of
/// `destinations[i]` instead of a full expert stride.
///
/// The expert blob is gate|up|down contiguous, so a caller that wants the
/// gate+up share to land before `down` fetches the front range, runs its
/// phase-1 work when this returns, and fetches the tail (at
/// `offset + byte_count`, into `destination + byte_count`) while that work
/// runs. Every `byte_counts[i]` must be at least 1.
int tinytitan_expert_reader_fetch_ranges(tinytitan_expert_reader *reader,
                                    const uint64_t *offsets,
                                    const uint32_t *byte_counts,
                                    void *const *destinations,
                                    size_t count);

/// Threads actually in use, after clamping.
int tinytitan_expert_reader_threads(const tinytitan_expert_reader *reader);

#endif /* TINYTITAN_EXPERT_IO_H */
