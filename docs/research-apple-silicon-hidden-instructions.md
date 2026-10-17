# Apple M1–M3: undocumented instructions and what developers can do with them

Research note, 2026-10-17. Scope: CPU-level undocumented instruction set
extensions and control registers on Apple M1, M2 and M3.
Verified locally on this machine: **Apple M3 (4 P-cores + 4 E-cores, 24 GB),
macOS 27.0.1 (26A434)**.

Short version: yes, several were found. **Two are executable from ordinary user
processes on macOS today, no entitlement and no kext — AMX (the matrix
coprocessor) and MUL53 — and both were verified on this machine.** The others
are privileged (kernel/firmware) control instructions or hidden system
registers. No backdoor-style "secret instruction" has been found.

---

## 1. The proprietary opcode space: `0x0020_xxxx`

Apple's undocumented A64 instructions all live in a reserved encoding hole
starting at `0x00200000`. The Asahi Linux reverse-engineering wiki keeps the
canonical map ([Apple Proprietary Instructions](https://asahilinux.org/docs/hw/cpu/apple-instructions/)):

| Encoding | Name | What it does | Reachable from |
|---|---|---|---|
| `0x00200000–0x002007FF` | **MUL53** (`mul53lo.2d`, `mul53hi.2d`) | 2×53-bit multiply on vector doubles; low/high 53-bit halves; present since A11 | **EL0 — verified on M3** |
| `0x00200800 \| rD<<5 \| rS` | **`wkdmc`** | Hardware WKdm page compression (source page → compressed buffer) | kernel (physical pages) |
| `0x00200C00 \| rD<<5 \| rS` | **`wkdmd`** | WKdm page decompression | kernel |
| `0x00201000–0x002012DF` | **AMX** | Matrix/vector coprocessor: ~30 opcodes | **EL0 — verified on M3** |
| `0x00201400` | **`gexit`** | Leave guarded execution mode | EL1 (needs enable) |
| `0x00201420 \| imm5` | **`genter`** | Enter guarded execution mode; `imm5` lands in `ESR_GLx[5:0]` | EL1 (needs enable) |
| `0x00201440 \| rA` | **`at_as1elx`** | Address translate, result returned in the same register (PAR_EL1-like) | EL1 |
| `0x00201460–63` | **`sdsb {osh,nsh,ish,sy}`** | DSB-family barrier (exact semantics not public); `ish` used by the iBoot trampoline | EL1/firmware |

`AMX_SET`/`AMX_CLR` are themselves `0x00201220`/`0x00201221` — inside the range
the wiki still labels a "hole", because the map predates their identification.

Accessibility was probed on this M3, not just read from the docs:

| Probe | Result |
|---|---|
| `mul53lo.2d` / `mul53hi.2d` with 128-bit reference (3 cases, incl. 2^53−1 operands) | correct low and high halves — **runs at EL0** |
| AMX `set`/`ldx`/`ldy`/`fma32`/`stz`/`clr`, 16×16 fp32 outer product | correct, 0 mismatches — **runs at EL0** |
| second `AMX_SET` on the same thread | `SIGILL` (confirms non-reentrancy) |
| `genter` | `SIGILL` (confirms the EL1 gate) |

---

## 2. AMX — the one you can actually use

### 2.1 What it is

A matrix coprocessor attached to each CPU cluster, fed by otherwise-invalid
A64 instructions issued from the core. Apple never documents it; it is what
Accelerate/BNNS dispatch SGEMM to on M1–M3.

Architectural state is 5 KB per thread:

- `X[8]`, `Y[8]` — 64-byte registers; each pool is also a 512-byte circular
  buffer, so instructions can address any 64-byte window (`x_offset`/`y_offset`).
- `Z[64]` — a 64×64-byte grid; the accumulator.

Data types: f16/f32/f64, bf16 (M2+), i8/i16 with mixed-width
multiply-accumulate (e.g. f16×f16→f32, i8×i16→i32).

### 2.2 Instruction families

Full reference: [corsix/amx](https://github.com/corsix/amx)
(`Instructions.md`, `RegisterFile.md`, per-instruction files).

| Family | Ops | Notes |
|---|---|---|
| Setup | `set` (17/0), `clr` (17/1) | `set` is **not re-entrant**: faults if AMX already enabled on the thread; Accelerate prefixes it with three `nop`s. |
| Memory | `ldx`, `ldy`, `ldz`, `ldzi`, `stx`, `sty`, `stz`, `stzi` | 64 B single, 128 B pair, 256 B quad (M2+); M3 adds non-consecutive register targets. |
| FP matrix | `fma16/32/64`, `fms16/32/64` (bit 63 = 0) | `z[j][i] += x[i] * y[j]` — full 16×16 (f32) outer product per instruction; 7-bit write masks, X/Y/Z skip flags. |
| FP vector | same ops, bit 63 = 1 | `z[_][i] += x[i] * y[i]`. |
| Int matrix/vector | `mac16`, `matfp`, `matint`, `vecfp`, `vecint` | 9-bit masks, indexed loads, shuffles, min/max, `sqrdmlah`, `popcnt`, shifts, saturation. |
| Data movement | `extrx`, `extry`, `extrh`, `extrv` | Move/extract between X/Y and Z with optional shift/saturation. |
| LUT | `genlut` | Indexed loads / gather-like table generation. |

Encodings are `0x00201000 | (op << 5) | reg`, with the real operands in the
selected general-purpose register (pointer in bits 0–55, flags above).

### 2.3 Version differences

- **M1**: 7-bit write masks; "load multiple" = 2 registers; one AMX block per cluster (P-cluster and a lower-throughput E-cluster block).
- **M2**: adds **bf16**; 256-byte loads; 64-bit PMU counters.
- **M3**: adds an extra mode to `ldx`/`ldy`/`matint` and non-consecutive register
  loads (`ldx` bit 61).
- **M4**: Accelerate's fp32 path moves to ARM **SME** per a 2026 study; AMX is
  still present per reverse-engineers but some `extr*`/`vec*` modes changed
  semantics (low offset bits ignored). Treat M1–M3 as the AMX generation.

### 2.4 Using it from userspace (verified)

Minimal, working C (compiled with `clang -O2` on this M3, result checked
against a scalar reference):

```c
#include "aarch64.h"   /* corsix/amx, MIT: encoding macros */

AMX_SET();                    /* brackets the AMX region; not re-entrant */
AMX_LDX((uint64_t)a);         /* 64 B -> X[0]  (word 0x00201000 | reg) */
AMX_LDY((uint64_t)b);         /* 64 B -> Y[0]  (word 0x00201020 | reg) */
AMX_FMA32(0);                 /* z[j][i] += x[i]*y[j]; word 0x00201180 | reg */
for (int j = 0; j < 16; j++)  /* f32 matrix mode uses Z rows 0,4,...,60 */
    AMX_STZ(((uint64_t)(4 * j) << 56) | (uintptr_t)&out[j][0]);
AMX_CLR();
```

The operand register number has to be baked into the instruction word, which is
why the macros exist; `set`/`clr` are plain `0x00201220`/`0x00201221` (with
Accelerate's three-`nop` prefix).

Measured on this machine:

- 16×16 fp32 outer product: correct (0 mismatches).
- ALU-only loop over the four f32 Z banks: **1.80–1.83 TFLOPS** fp32
  (8 M `FMA32` instructions, no loads).
- Literature cross-check: 1.48 TFLOPS on M1 Max (corsix), 1.67 TFLOPS on
  M2 Pro (independent microbenchmark), but a *real* single-thread GEMM with
  loads falls to **610–680 GFLOPS** because the inner loop is load-issue
  bound; multi-thread across both AMX blocks recovers ~1.48 TFLOPS aggregate
  on M1.

### 2.5 Who already uses it

- **Accelerate / BNNS / BNNS Graph** — Apple's own SGEMM path on M1–M3.
- **Modular MAX** ships `apple_amx_intrinsics.mojo` in production
  ([source](https://github.com/modular/modular/blob/max/v26.6/max/kernels/src/linalg/arch/cpu/apple_amx_intrinsics.mojo)).
- **llama.cpp prefill**: a direct-AMX bit-exact kernel took full-forward
  throughput from 291 → 420 tok/s at 128-token prefill (1.44×) on M1
  ([arXiv:2606.25426](https://arxiv.org/abs/2606.25426)).
- **Post-quantum crypto**: NTRU polynomial multiplication
  ([eprint 2024/002](https://eprint.iacr.org/archive/2024/002/1704885524.pdf))
  and Saber/FrodoKEM ([eprint 2024/195](https://eprint.iacr.org/2024/195))
  set speed records on M1/M3 using AMX.

### 2.6 Caveats

- Undocumented and unsupported: no ABI, no intrinsics, no guarantee it exists
  on the next chip (M4 already changed the picture).
- Per-thread state is 5 KB; `set`/`clr` bracketing is cheap but not free, and
  `set` faults (`SIGILL`, verified) if AMX is already enabled on the thread —
  including if a library such as Accelerate enabled it and did not clear it.
- Inline asm only — Swift cannot emit it directly; you need a C/asm shim.
- Don't ship it blind: App Store review, future OS changes and profiler/sanitizer
  blind spots are real risks. It is best treated as an opt-in fast path with a
  runtime probe and a scalar/NEON fallback.

### 2.7 MUL53 — the other userspace-usable one

`mul53lo.2d` / `mul53hi.2d` (`0x00200000 | m<<5 | d`, `0x00200400 | m<<5 | d`)
multiply two 53-bit values per 128-bit vector lane, returning the low 53 bits in
`Vd` or the product shifted right by 53. Two instructions reconstruct the full
106-bit product — useful for exact double-double arithmetic or big-integer
crypto, and present since A11. Verified on this M3 against a `__int128`
reference: low and high halves correct for three operand sets including
2^53−1. It is niche, but it is genuinely available from EL0 and can be probed
safely in a forked child.

---

## 3. Guarded execution: `genter`/`gexit`, SPRR, APRR

This is the second real discovery, but it is mostly **kernel-level**.

- **GXF (guarded execution)** adds lateral exception levels `GL1`/`GL2` next to
  EL1/EL2. They use the *same* page tables with *different* permissions.
  `genter` switches EL→GL and takes the PC from `S3_6_C15_C8_1` (`GXF_ENTER_EL1`);
  `gexit` returns. Guarded mode
  has its own `ELR/FAR/ESR/SPSR/VBAR/TPIDR` (`S3_6_C15_C10_*`) plus `ASPSR`.
- **SPRR** turns the PTE permission bits (`AP[1:0]`, `UXN`, `PXN`) into a 4-bit
  index into a 16-entry permission table; each entry holds four permission bits
  `{GL1, GL0, EL1, EL0}`. That lets software flip a page between `r-x` and `rw-`
  for one mode without touching page tables or flushing TLBs.
- **APRR** is the older equivalent, reached from EL0 through the JIT mechanism
  (`SYS_APL_APRR_EL0`, `S3_4_C15_C2_0`): 16 × 4-bit `{X,W,R}` entries indexed by the PTE's
  `{XN,PXN,AP0,AP1}`. This is what `pthread_jit_write_protect_np()` toggles, and
  why MAP_JIT write-protection is a few-nanosecond register write instead of an
  `mprotect` syscall.

Developer-facing consequence: fast W^X for JITs is available through the
documented API ([Apple: porting JIT compilers](https://developer.apple.com/documentation/apple-silicon/porting-just-in-time-compilers-to-apple-silicon)).
On this machine `pthread_jit_write_protect_supported_np()` returns `1`.
`genter`/`gexit` themselves need EL1 enablement and are not callable from
third-party userspace (verified: `genter` raises `SIGILL`).

---

## 4. Hidden system registers

All privileged (kernel drivers, or `kperf` on macOS) except the APRR pair:

- **PMU**: `PMC0–9`, `PMCR0–4`, `PMESR0/1`, plus uncore `UPMC0–15`/`UPMCR0`.
  M1 counters are 48-bit, M2+ 64-bit. Used by Linux's `apple_m1_pmu` driver;
  event IDs are undocumented and reverse-engineered.
- **HID0–HID21 / EHID***: "chicken bits". Examples: `HID0[21]` disables AMX
  cache fusion, `HID0[20]` the loop buffer, `HID9[16]` enables a **TSO memory
  model** (x86-like ordering), `HID7`/`HID9` hold speculation and errata
  workarounds. Not a supported programming interface, but they show how much
  behavior is switchable in hardware.
- **CTRR**: "Configurable Text Read-only Region" — locks a physical text range
  read-only/executable at EL1/EL2.
- **APRR_JIT_ENABLE_EL2 / APRR_JIT_MASK_EL2**: the hypervisor-side enable for
  the JIT permission mechanism.

---

## 5. What is *not* there

- No backdoor or "magic" instruction that unlocks extra GPU/ANE throughput.
  The hidden surface is a coprocessor ISA plus privileged control registers.
- No ARM **SME/SVE** on M1–M3 (`hw.optional.arm.FEAT_SME = 0` on this M3).
  SME appears to arrive with M4 (the 2026 study finds M4 Accelerate dispatching
  SGEMM to SME), which is why the matrix path changes there.
- The ANE and GPU have their own undocumented ISAs, but they are not reachable
  as CPU instructions; they are reached through private frameworks/firmware.

---

## 6. Relevance to TinyTitan

TinyTitan already has GPU and ANE prefill paths and a CPU engine; AMX is the
missing CPU-side matrix path.

- **Where it could help**: the dense CPU engine's prefill (currently NEON),
  CPU-only/CI runs, and cases where the GPU is busy or thermally throttled.
  Published microbenchmarks put AMX roughly an order of magnitude above one
  NEON core (102 → 1475 GFLOPS on M1 Max), and a bit-exact fp32 AMX GEMM can
  beat Accelerate's own SGEMM at prefill shapes (1.17–1.58× per the 2026 study).
- **Where it will not**: large prefill on a healthy GPU/ANE. AMX peaks around
  1.5–1.8 TFLOPS fp32 ALU-only (1.8 measured on this M3) and ~0.6–0.7 TFLOPS in
  a real single-thread GEMM; the GPU and ANE are far ahead for big GEMMs. AMX
  wins on small/medium GEMMs and on CPU-only paths, not on the main prefill path.
- **Shape of the work**: a C/asm microkernel (Swift needs a shim), 16×16 f32
  tiles with 4-way Z-bank ILP, B pre-packed once at model load, `Nc=64` panels
  so both AMX blocks (P- and E-cluster) are fed, and `Kc` sized to the 128 KB L1.
  Runtime-probe AMX, fall back to NEON, and keep output bit-identical so it can
  be validated against the existing engine.
- **Cheapest experiment**: a standalone benchmark that multiplies the same
  shapes the CPU engine sees, AMX vs Accelerate vs current NEON, before touching
  the engine.

---

## 7. Sources

- [corsix/amx — instruction set reference, register file, emulation code](https://github.com/corsix/amx)
- [Asahi Linux — Apple Proprietary Instructions](https://asahilinux.org/docs/hw/cpu/apple-instructions/)
- [Asahi Linux — SPRR and GXF](https://asahilinux.org/docs/hw/cpu/sprr-gxf/)
- [Asahi Linux wiki — ARM System Registers](https://github.com/AsahiLinux/docs/wiki/HW:ARM-System-Registers)
- [Above the Inner Loop: Exceeding Accelerate at LLM Prefill GEMM on the M1 AMX (2026)](https://arxiv.org/abs/2606.25426)
- [Modern iOS Security Features — SPTM, TXM, Exclaves (2025)](https://arxiv.org/abs/2510.09272)
- [FLOP: Breaking the Apple M3 CPU via False Load Output Predictions (USENIX Security '25)](https://www.usenix.org/conference/usenixsecurity25/presentation/kim-jason)
- [Modular MAX — apple_amx_intrinsics.mojo](https://github.com/modular/modular/blob/max/v26.6/max/kernels/src/linalg/arch/cpu/apple_amx_intrinsics.mojo)
- [Apple — Porting just-in-time compilers to Apple silicon](https://developer.apple.com/documentation/apple-silicon/porting-just-in-time-compilers-to-apple-silicon)
- [Explore AMX instructions (microbenchmarks)](https://medium.com/@zhen8838/explore-amx-instructions-unlock-the-performance-of-apple-silicon-e0276a293aec)
