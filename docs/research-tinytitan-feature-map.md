# Feature map: session research → TinyTitan code sections

2026-10-17. Maps the two research notes
([undocumented instructions](research-apple-silicon-hidden-instructions.md),
[cache/bandwidth/compression](research-apple-silicon-cache-bandwidth.md)) onto
the actual tree. Machine: Apple M3, 24 GB, macOS 27.0.1.

The short version: **most of the research does not apply to TinyTitan.** The
streaming and CPU paths already make the right choices, and the remaining items
are either conditional on a design change or measurement-only.

## Verdict table

| Feature | Code section | Verdict |
|---|---|---|
| QoS ≥ `.default` for memory-bound work | `ModelExpertIO.swift:250` (`.userInitiated`), `ExpertLoadOperation.swift:252` (workers `.userInitiated`), `PreadExpertStreamer+IO.swift:191` (`concurrentPerform`) | **Already satisfied — keep.** `BACKGROUND` would cost 3–4× (18 vs 63 GB/s read); nothing in the demand path is there today. |
| 4 readers, `F_NOCACHE`, `F_RDADVISE` | `expert_io.c:145`, `RDAdvice.swift:36`, `ParallelExpertReader.swift:9-13` | **At the measured ceiling — keep.** 3.92 GB/s at 4 threads = 3.92 at 8. More threads would not help. |
| 16 KB page alignment / mmap | `ResidentBuffer.swift:22`, `PreadExpertStreamer.swift:228` | **Already correct.** Page-aligned resident mapping and slot pool. |
| 128 B cache line | — | **No action.** No CPU-shared counters today; only relevant if atomics land on one line. |
| **AMX** | `CPUQwen35.swift:14` — the CPU engine deliberately has **no batched prefill**; prompts are decode with logits discarded | **Defer.** AMX is a GEMM engine; today's CPU work is bandwidth-bound GEMV. Payoff requires a new batched CPU prefill (C shim, 16×16 tiles, probe, NEON fallback). |
| `DC ZVA` | Only if manual zero-fill is added | **No action, but note the trap:** the ZVA block is **64 B**, not the 128 B `hw.cachelinesize`; a stride-128 loop leaves half the buffer dirty. `memset(0)` already takes the fast path (81 vs 65 GB/s). |
| PMU (`PL2_CACHE_*`, `LD_SRC_LL_CACHE`, `LD_SRC_MEMSYS`) | `TinyTitanBench` | **Optional measurement.** The only way to know whether the 16.88 GiB expert stream hits the SLC; kperf, root, no kext. |
| Shared vs private Metal scratch | `QSAIndexer.swift:115`, `Attention.swift:132` (shared) vs `LMHeadChainInt4.swift:52` (private) | **Measure.** Shared is correct for CPU-visible data; GPU-only scratch in private may avoid CPU cache pollution. |
| Non-temporal `LDNP`/`STNP` | — | **Skip.** Measured inside noise (36.7 vs 35.0 GB/s copy). |
| GXF/SPRR/APRR, MUL53, WKdm `wkdmc`/`wkdmd` | — | **Not applicable.** No JIT, no 53-bit math, no kernel code. |
| `MTLStorageModeMemoryless` | — | **Not applicable.** Render targets only; this engine is compute-only. |
| Compressed-memory tuning | — | **No API.** `vm_stat` shows the compressor working (~1.8:1); apps cannot opt pages in or out. |

## Verified call paths

- `fetchRoutedExperts` → `DispatchQueue.global(qos: .userInitiated)` →
  `loadExpertsCached` → plan → `executeCachedPreads` → `concurrentPerform`.
  `concurrentPerform` inherits the caller's QoS, so the demand read path is
  `.userInitiated` end to end.
- `ExpertIOScheduler` workers are created with `qos: .userInitiated`.
- `streamersQueue` is default QoS and only does bookkeeping; the C reader
  threads are created there, so they inherit default (above the background
  cliff, below user-initiated — acceptable for disk-bound work).
- `CPUQwen35` has no batched prefill by design (`CPUQwen35.swift:14-16`), which
  is what makes AMX inapplicable without a new code path.

## What is already right and should not be "optimised"

The staging copy in `MetalExpertStagingPool` is hazard-driven and the path is
disk-bound at 3.9 GB/s; the extra copy is not the bottleneck. The CPU backend
already learned the `madvise` lesson (a hint cost 2.6× throughput,
`CPUModelBackend.swift:67`). The resident weights are file-backed `mmap`
(clean, evictable), which is the right memory class for a 5 GB working set.

## Next steps, in order

1. Keep the QoS guard: do not move generation, expert I/O or CPU GEMV onto a
   `.background`/`.utility` queue. No code change needed today.
2. Optional: add a kperf counter run to `TinyTitanBench` for SLC/PL2 attribution.
3. Only if a batched CPU prefill is added: prototype AMX behind a C shim, with a
   runtime probe and the existing NEON/Accelerate path as fallback.
