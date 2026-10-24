# The `lowmem` 2-bit routed-expert profile

Status: **implemented, experimental, quality unverified.** Do not describe a
`lowmem` install as supported until a real-model quality/perplexity run exists.

## What already existed when this profile was added

- `packed_experts/layout.json` already records a per-tensor `bits` width
  (`SSDAISubTensorV1.bits`), and `SSDAIPackedExpertsLayoutV1` already accepts a
  compact expert whose `size` is smaller than `expertStride`.
- The MoE kernels already take the routed width as a function constant and
  accept 2, 4 and 8 bits:
  - decode: `moe_weight_bits()` / `moe_affine_value` in
    `sources/TinyTitan/Metal/MoE/moe.metal`;
  - decode dispatch: `MoE.init(routedWeightBits:)` in
    `sources/TinyTitan/Kernels/MoE/MoE.swift`, which compiles `FC_MOE_WEIGHT_BITS`
    and picks the `moe_affine_*` pipelines for any width other than 4;
  - prefill: `PrefillGroupedRoutedMoE(weightBits:)` accepts `[2, 4, 8]`.
- `MoEFusedFFNTests.affineRoutedPipelineSupportsQwenBits` runs the fused
  pipeline at 2, 4 and 8 bits (with zero-filled weights, so it does not pin the
  byte order).
- A *tiered* 2-bit path also existed: `tools/tiered_repack.py` remaps cold
  experts, and `TwoBitExpand` expands them into a 4-bit slot at load.

What did **not** exist: a manifest/profile that names 2-bit routed experts,
loader validation for that shape, a converter that writes a uniform native
2-bit install, catalog/doctor handling, and a model profile row.

## The profile

`manifest.json` gains an optional `"memoryProfile": "lowmem"`. Only the
non-default value is written, so a standard repack is byte-identical to what it
was before. A `lowmem` manifest:

- sets `quant.routedExpert.weightBits = 2`;
- keeps `embedding`, `attention`, `router` and `sharedExpert` at 4 or 8 bits
  (6-bit is withdrawn and refused);
- stores every routed expert at 2 bits, one uniform `expertStride` per layer,
  with no compression (the compressed readers never run the width path).

The loader refuses the two dangerous near-misses:

- a standard manifest with `routedExpert.weightBits = 2` (it would be expanded
  as if it were a cold tier);
- a `lowmem` manifest whose layout holds a 4-bit expert, a compact offset or a
  compression marker.

An install's API id comes from `routedExpert.weightBits`, so a `lowmem` install
is `qwen3.6-35b-a3b_2-Bit` and cannot collide with the 4-bit build.

## Arithmetic

A 4-bit group decodes as `q4 * scale + bias`, `q4` in `0...15`. The 2-bit
remap is:

```
q2     = round(q4 / 5)          # 0...3
scale2 = 5 * scale             # bf16
bias2  = bias
```

Then `q2 * scale2 + bias ≈ q4 * scale + bias`, with at most two 4-bit steps of
error (`|5*q2 - q4| <= 2`). The 2-bit blob is packed **low element first**
(bit `2k` holds element `k`), which is what `moe_affine_value` reads. This is
the opposite order from `TwoBitExpand`'s stream (top pair first), so the two
remaps are deliberately separate functions.

Note: the pre-existing *tiered* converter keeps the source scale and the
expansion divides it by five, which makes an expanded cold expert five times
smaller than its 4-bit source. That is an existing defect in the tiered path,
outside this profile; it is not changed here. The native `lowmem` path uses
`5 * scale` and does not depend on the expansion.

## Build, verify, serve

```bash
# From an existing 4-bit .ssdai install (no 40B download in this task):
python3 tools/tiered_repack.py --profile lowmem \
    --model models/ornith-1.5_35B_A3B_4Bit \
    --out models/ornith-1.5_35B_A3B_2Bit-lowmem

# The uniform native layout can be receipted:
swift run -c release TinyTitanRepack --verify-install \
    --input-ssdai models/ornith-1.5_35B_A3B_2Bit-lowmem

# Serving is opt-in because quality is unverified:
TINYTITAN_ALLOW_EXPERIMENTAL_LOWMEM=1 swift run -c release TinyTitanServer \
    --model models/ornith-1.5_35B_A3B_2Bit-lowmem
```

The converter hard-links `model_weights.bin`, copies the tokenizer and any root
passthrough files, rewrites every layer file, `layout.json` and the manifest's
sizes/hashes. It refuses to overwrite an existing output directory.

`doctor` lists a `lowmem` install with the note that it is experimental, and
its fit uses the install's actual half-size `expertStride` plus the profile
row's declared expert-cache budget (`ModelProfile` rows keyed `(model, 2)`
halve the 4-bit row's bytes, which keeps the slot count constant).

## Quality: no run was done here, and the prior evidence is negative

No real-model perplexity or generation run was made for this profile. The
project's earlier 2-bit experiment (`docs/residency-and-bottleneck-verdict-
2026-10-20.md`) measured a 2-bit *tiered* install diverging from layer 0 and
concluded uniform 2-bit quality is too lossy (the separate probe put the
added weight error at ~0.47 against an 8-bit reference). A later run must
establish:

1. that the native path reproduces the intended weights (the unit tests here
   cover the arithmetic and byte order, not a model);
2. perplexity and a task-quality comparison against 4-bit at the same context;
3. decode throughput and RAM against the 4-bit build on the same host.

Until then the catalog's opt-in gate is the honest state: the profile loads and
is testable, and is not served by default.

## Unverified

- Real-model quality, perplexity and throughput of a `lowmem` install.
- The uniform-native converter on a real 35B/125B install (only toy layouts
  and the Python packing tests were run here).
- Whether the fitted `expertCacheBudgetBytes` (half the 4-bit row) is the right
  operating point; it is arithmetic, not a measurement.
- The pre-existing tiered expansion's 5x scale defect, which needs its own
  investigation before the tiered quality verdict is trusted.
