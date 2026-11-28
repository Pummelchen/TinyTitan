# Decode baseline and stall map, 2026-11-28

Machine: base 8-core M3 MacBook Pro, 24 GB, macOS 27.2.1, Swift 6.4.2,
`swift build -c release` (portable arm64 baseline). Load average 2.6-3.2,
swap ~0.9 GB, no Chrome. Golden identity first:
`tools/golden-baseline.sh --check ornith-4` -> `ok — output identical to baseline`.

## Decode throughput

Greedy, chat-framed prompt, one process per run.

| install | new tokens | decode tok/s |
| --- | ---: | ---: |
| Ornith 1.5 35B-A3B 4-bit | 128 | 15.1-15.6 |
| Ornith 1.5 35B-A3B 4-bit | 256 | 15.4-16.5 |
| Ornith 1.5 35B-A3B 4-bit | 512 | 16.9 |
| Inkling 2 Mini UD-Q4_K_XL | 128 | 10.3 |
| Inkling 2 Mini NVFP4 | 128 | 3.1 |

The recorded reference for Ornith's 160-slot profile is 20.55/21.04 tok/s
(`ModelProfile` table comment, 2026-09-05). This host reads ~15.5-16.9.

## Where the time goes (Ornith, 128 tokens, `TINYTITAN_KERNEL_STATS=1`)

- GPU busy **43.8 ms/token**, decode occupancy **62%** — ~38% of decode is
  GPU idle in gaps between command buffers.
- Expert I/O: 96.9% hit at 512 tokens, **16.8 MiB/token** read, p99 load 4 ms,
  `io_hidden_pct=100`.
- Largest GPU idle transitions, per token:

| transition | ms/token |
| --- | ---: |
| `shared_expert -> moe_phase1_2_routed` | 11.16 |
| `moe_phase1_hit -> moe_phase1_miss_fixup` | 11.12 |
| `shared_expert -> moe_phase1_hit` | 4.13 |
| `moe_phase2_routed -> attn_norm_qkv` | 1.45 |

The second row is the on-demand miss wait: about ten misses per token at the
p50 1 ms load, after the phase-1 hit pass has finished. The first is host
encode/scheduling between the shared-expert and routed command buffers.

## Defaults and knobs measured (paired where it matters)

| knob | result |
| --- | --- |
| expert cache 160 slots (default) vs 192 vs 256 | 160 best (16.0 median); 192 15.3; 256 11.7 |
| predictive prefetch on vs off | wash (16.06 vs 16.03) |
| prefetch ring depth 1/2/3 | wash (16.3 / 16.2 / 16.1) |
| `--kv-bits 8` vs `4` | wash (15.4-15.6) |
| `--prefill-chunk` 512/1024/2048/4096 | wash on decode (15.4-15.6) |
| `--rdadvise` default/bounded/adaptive/off | wash (15.8-16.0) |
| pinning on vs `TINYTITAN_NO_PIN=1` | pinning +5% (16.5 vs 15.7) |
| native M3 build (`tools/build_m3.sh`) vs portable | identical (15.5) |

## Micro-benchmarks (`TinyTitanBench`, 300 iterations, quiet GPU)

| arm | GB/s |
| --- | ---: |
| `baseline` (QKV GEMV) | 62.7-74.7 |
| `gdn_inproj` | 66.7-74.8 |
| `head_affine8` | 74.7-76.5 (recorded 89.9) |
| `head_int4` | 75.8-77.0 (recorded 91.5) |
| `moe` | 28.3-39.6 — **not trustworthy**, see below |

**Defect found while measuring:** every routed-MoE bench arm writes an
all-zero `acts` buffer (`acts_fnv=bcc31dc5` is FNV-1a over 8,192 zero bytes),
including `moe_phase1_{r16,v2,subset}` and the specialized pipeline. Binding
the declared `io_status` buffer and `useResource` on the blobs did not change
it, so the harness or the kernel's shape binding is wrong; the `moe` GB/s and
the `gpu.routed_moe_gbps` row it feeds are therefore unverified.

## Inkling: the KV precision is the biggest single lever

Both Inklings are read at the shipped default `--kv-bits 8`. The NVFP4 install
does not merely lose a few percent:

| install | kv-bits | new tokens | decode tok/s |
| --- | ---: | ---: | ---: |
| Inkling 2 Mini NVFP4 | 8 (default) | 32 | 3.6 |
| Inkling 2 Mini NVFP4 | 4 | 32 | 8.1 |
| Inkling 2 Mini NVFP4 | 16 | 32 | 0.8 |
| Inkling 2 Mini UD-Q4_K_XL | 8 | 64 | 10.7 |
| Inkling 2 Mini UD-Q4_K_XL | 4 | 64 | 11.2 |

The recorded README figure for NVFP4 is 8.1 tok/s, i.e. what the 4-bit cache
gives — so the 8-bit cache path (`kv_finish_q8` / `attention_*_q8`) is the
outlier, not the machine. Its output is coherent (`kv-bits 4` answers "Paris"),
but 4-bit KV is a precision change and cannot become the default without the
quality evidence the profile table asks for.

## Next targets

1. The `moe_phase1_hit -> miss_fixup` gap: start miss loads earlier or widen
   the overlap window, so the phase-1 hit pass covers their latency.
2. The `shared_expert -> routed` gap: take host encoding of the routed buffer
   off the critical path (it is ~11 ms of the 70 ms token).
3. Inkling NVFP4 at 3.1 tok/s against a recorded 8.1 is the largest single gap
   in the shipped matrix, and it is the 8-bit KV path: profile `kv_finish_q8`
   and the q8 attention kernel against their q4 counterparts, then fix the slow
   one. Making 4 the default is a precision change and needs its quality check.
