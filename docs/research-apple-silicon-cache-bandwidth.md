# Apple M1–M3: cache, bandwidth and compression tricks

Research note, 2026-10-17. Companion to
[research-apple-silicon-hidden-instructions.md](research-apple-silicon-hidden-instructions.md).
All measurements on this machine: **Apple M3 (4 P + 4 E), 24 GB, macOS 27.0.1**,
`clang -O2`, best of 3–5 runs, QoS `USER_INTERACTIVE` unless stated.

Short version: there is **no Apple-specific cache-control interface** in user
space — no way locking, no SLC control, nothing beyond the standard ARM
`PRFM`/non-temporal hints. What is hidden and useful is the *geometry*
(128-byte lines, per-cluster L2, 16 KB pages), a set of **EL1-only chicken
bits** that reconfigure the caches,
**PMU events** that are the only way to see the SLC, the **hardware WKdm
compression instructions**, and a few measurable tricks (QoS placement,
`DC ZVA`, read-vs-copy asymmetry).

---

## 1. Verified cache geometry

| Resource | P-core | E-core |
|---|---|---|
| L1I | 192 KB | 128 KB |
| L1D | 128 KB | 64 KB |
| L2 (per cluster) | 16 MB | 4 MB |
| Line size | **128 B** (`hw.cachelinesize`) | 128 B |
| Page size | **16 KB** (`hw.pagesize`) | 16 KB |

`sysctl hw.l3cachesize` does not exist and `hw.cachesize` stops at L2: **the SLC
is not described by any public API.** It exists (it is what keeps CPU/GPU/ANE
traffic off DRAM) but software can neither size it nor target it. The only
observability is the PMU (section 6).

## 2. The 128 B line vs the 64 B `DC ZVA` block (a real trap)

`hw.cachelinesize` says 128. `DC ZVA` — the zero-fill instruction every
`memset(0)` wants — zeroes **64 bytes**, not 128. Measured by filling a buffer
with `0x5A`, zeroing at different strides, and counting non-zero bytes:

| Stride | Non-zero bytes left (1 MB buffer) |
|---|---|
| 32 | 0 |
| 64 | 0 |
| **128** | **524,288 (half the buffer)** |
| 256 | 786,432 |

So code that steps by the advertised cache line size leaves every other 64-byte
block untouched. Use a 64-byte stride for `DC ZVA`, and 128 for ordinary line
alignment/padding.

`CTR_EL0` — which would tell you the real block size — is **trapped at EL0**
(`mrs ctr_el0` raises `SIGILL`), so the only portable source is the 64 B
empirical value (or the commpage/sysctl line size, which is wrong for this).

All the usual maintenance ops *are* allowed at EL0 on macOS, though:

| Instruction | EL0 result |
|---|---|
| `dc zva` | ok |
| `dc cvac` | ok |
| `dc civac` | ok |
| `dc cvau` | ok |
| `ic ivau` | ok |

That is how JITs publish code and flush data without a syscall. It also means
you can invalidate your own lines — useful before timing, dangerous if you
invalidate lines another thread is using.

## 3. Measured bandwidth (256 MB–2 GB buffers)

| Path | GB/s | Notes |
|---|---|---|
| `memset(p, 0)` (libc) | 80–85 | consistent with a `DC ZVA` fast path |
| explicit `DC ZVA` loop, 64 B stride | 81 | verified all-zero afterwards |
| `memset(p, 0xAA)` | 64–66 | pattern fill has to write data |
| read (sum), 1 thread | 63 | `USER_INTERACTIVE`/`DEFAULT` |
| read, 4 threads | 77 | near the practical fabric limit |
| copy (`ldp`/`stp`), 1 thread | 38–41 | **does not scale**: 4 threads = 38 |
| nominal DRAM | 100 | LPDDR5, M3 |

Two conclusions:

- **Practical bandwidth is ~80 GB/s, not 100.** Four threads reading saturate it.
- **Copies cost twice.** Copy traffic is one read plus one write, so a 40 GB/s
  copy already uses the ~80 GB/s fabric. Adding threads cannot help; removing
  the copy (zero-copy, shared buffers, in-place work) can.

Non-temporal hints (`LDNP`/`STNP`) made no meaningful difference in the copy
loop here: 36.7 GB/s vs 35.0 GB/s for `LDP`/`STP`, inside run-to-run noise. The
hardware has "LDNT/STNT widgets" (section 5), but macOS does not expose a way to
tune them, and at this working-set size the prefetcher already does the job.

## 4. The QoS cliff: 3–4× bandwidth for free

Same code, same buffer, only the QoS class changed:

| QoS | read GB/s | copy GB/s |
|---|---|---|
| `BACKGROUND` | 18 | 10.5 |
| `UTILITY` | 57–60 | 39–40 |
| `DEFAULT` | 63 | 41 |
| `USER_INITIATED` | 64 | 42 |
| `USER_INTERACTIVE` | 60–63 | 40–42 |

`BACKGROUND` threads are placed on the E-cluster and lose ~3.5× read and ~4×
copy bandwidth. Anything bandwidth-bound — model loading, weight packing,
prefill — should run at `DEFAULT` or above, and should not be left at
`BACKGROUND` "to be polite". Note this is a *scheduling* effect, not a cache
control: the E-cluster's L2 is 4 MB instead of 16 MB and its fabric path is
narrower.

The same placement effect explains the AMX result in the companion note: the
AMX block lives per cluster, so P-cluster threads and E-cluster threads do not
see the same matrix throughput.

## 5. Cache behaviour that only the kernel can change

The Asahi reverse-engineering wiki documents Apple's `HID0–HID21` "chicken
bits" (EL1 only). The cache-relevant ones:

| Register | Bit | Effect when set |
|---|---|---|
| `HID0` | 21 | disable **AMX cache fusion** |
| `HID0` | 36 | disable **cache fusion** |
| `HID0` | 25 | limit I-cache prefetch to one "brn" |
| `HID0` | 62:60 | I-cache prefetch depth |
| `HID3` | 25 | disable `DC ZVA` |
| `HID4` | 11 | disable `DC MVA` ops |
| `HID4` | 44 | disable `DC SW` L2 ops |
| `HID5` | 57 | disable **full-line write** |
| `HID5` | 61 | disable **fill 2C merge** |
| `HID4` | 1 | disable **STNT** widget (non-temporal stores) |
| `HID11` | 59 | disable **LDNT** widget (non-temporal loads) |

These are not a programming interface, but they prove the hardware has
full-line-write, fill-merging and non-temporal paths that firmware can turn
off. On macOS the only way to touch them is a kernel extension / debug boot.

Also EL1-only: the **PMU** (`PMCR0–4`, `PMC0–9`, uncore `UPMC0–15`) and the
**CTRR** text-protection region.

## 6. What the PMU can see (the only SLC visibility)

macOS ships per-CPU event tables in `/usr/share/kpep/*.plist` (26 files on this
machine) for `kperf` / Instruments' CPU Counters. Besides the standard ARM
events they contain Apple-private ones:

- `PL2_CACHE_ACCESS`, `PL2_CACHE_MISS` (+ `_LD`, `_ST`, `_INSTRUCTION`, `_MMU`)
  — the private per-cluster L2.
- `LD_SRC_PL2_CACHE_NONSPEC`, `LD_SRC_LL_CACHE_NONSPEC`,
  `LD_SRC_MEMSYS_NONSPEC`, `LD_SRC_MEMSYS_OTHERDIE_NONSPEC` — where a load was
  actually served from (private L2, last-level/SLC, DRAM, other die).
- `*_NT_*` events — non-temporal loads/stores.
- `LDST_UNIT_WAITING_*` and, on the newer SME-capable chips,
  `CORE_WAITING_SME_ENGINE_CYCLE` — stall attribution.

`LD_SRC_*` is the closest thing to an SLC hit-rate counter a developer can get;
it is root-only (kperf) but does not require a kext.

## 7. Compression

Three different things are called "compression" on Apple Silicon:

1. **Hardware WKdm page compression** — `wkdmc` (compress) and `wkdmd`
   (decompress) in the `0x0020` opcode space, page-granular, physical-address
   operands (see the companion note). Kernel-only; XNU's compressor uses the
   WKdm algorithm, so these are presumably its hardware fast path (not
   confirmed by Apple).
2. **macOS compressed memory** — visible in `vm_stat` and
   `sysctl vm.compressor.*`. On this machine: 105,961 pages stored in
   58,838 pages of space (~1.8:1), with counters for compressions,
   decompressions and swap-outs. Applications cannot opt a page in or out
   directly; the compressor decides.
3. **GPU bandwidth compression** — Apple GPUs keep render targets in tile
   memory and compress texture/render-target traffic. The developer-facing
   lever is Metal: `MTLStorageModeMemoryless` for transient render targets
   (never touches DRAM), `storageModePrivate` + blit instead of shared
   round-trips, and compressed pixel formats (ASTC/BC) for textures. This is
   the biggest real bandwidth win available to app code, and it is documented.

Zero-fill is a fourth, cheaper trick: `DC ZVA` at 81 GB/s beats a pattern fill
at 65 GB/s and leaves the line in cache, so a freshly zeroed buffer is also warm.

## 8. Practical checklist

1. Pad and align shared structures to **128 B**, not 64 — false sharing is
   per 128 B line.
2. Zero with `DC ZVA`, **stride 64 B**; `memset(0)` already does something
   equivalent.
3. Run bandwidth-bound work at `DEFAULT` QoS or above; never `BACKGROUND`.
4. Prefer reads over copies; copies hit the fabric twice and cap at ~40 GB/s.
   Use zero-copy/shared buffers, and do work in place where possible.
5. Use all P-cores for streaming reads (63 → 77 GB/s); more than ~4 threads
   adds nothing.
6. Use `ic ivau` / `dc cvac` directly for JIT-style publishing instead of
   syscalls, but only on your own lines.
7. For GPU work, use `MTLStorageModeMemoryless` and private storage; that is
   where the real bandwidth savings are.
8. Measure with `kperf`/Instruments CPU Counters if you need SLC/PL2 attribution;
   there is no other SLC visibility.

## 9. Sources

- Local measurements (this M3): `sysctl`, `vm_stat`, microbenchmarks in
  `/tmp` (`fillbench`, `copyscale`, `qosbench`, `zva`, `cacheops`).
- [Asahi Linux — ARM System Registers (HID bits, PMU, CTRR)](https://github.com/AsahiLinux/docs/wiki/HW:ARM-System-Registers)
- [Asahi Linux — Apple Proprietary Instructions (`wkdmc`/`wkdmd`)](https://asahilinux.org/docs/hw/cpu/apple-instructions/)
- [Apple vs. Oranges: Apple M-series for HPC (cache sizes, 100 GB/s, M4 SME)](https://arxiv.org/abs/2502.05317)
- [corsix/amx — AMX reference (AMX cache-fusion context)](https://github.com/corsix/amx)
- [Apple — Metal `MTLStorageMode`](https://developer.apple.com/documentation/metal/mtlstoragemode)
