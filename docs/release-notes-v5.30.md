## TinyTitan 5.30 — prefill, 31 % faster

This release is about the wait before the first token. On the 2.2K-token
agent-shaped prompt the prefill phase drops from **~48.5 s to ~33.4 s** with the
two new paths below enabled, and nothing about decode changes. All numbers are
paired A/B against the previous default, with rotating arm order and a control.

### Eight-query prefill attention

The scalar causal kernel launched one threadgroup per (query, head) and re-read
every visible K/V row for each of them — roughly 100 GB of cache traffic for a
2.5K-token chunk. `attention_prefill_causal_q8` gives one simdgroup one query
and puts eight consecutive queries in a threadgroup, so a K/V row is read once
per group, and the reduction becomes a `simd_sum` with no barrier in the key
loop. The KV dequantisation invariants (row pointer, group count, group index)
are hoisted out of the inner loop.

- full-attention GPU: **17.7 s → 7.8 s** per 2.2K chunk, 2/2 paired rounds;
- prefill wall: **43.4 s → 33.4 s** (−23.2 %), 2/2;
- 8-bit install, short prompt: prefill **10.61 s → 9.71 s** and 9.41 s → 9.40 s.

The kernel is gated to no selection mask, `headDim % 32 == 0`, `headDim / 32 <= 8`,
and **`kvRingCapacity == 0`**. The ring gate is a correctness fix: the ring
capacity is function constant 76, and the q8 pipeline is compiled without it, so
a ringed chunk would have used the logical key instead of `key % ring`. The
prefill attention suite caught it; ringed chunks fall back to the scalar kernel.
`TINYTITAN_PREFILL_ATTN_Q8=0` restores the old kernel.

### The shared expert stops running one token at a time

`PrefillSharedExpert` called the decode single-token GEMV once per row — 2,505
dispatches a layer for a 2K chunk, each re-reading the same int8 weights. It now
runs gate/up/down through the prefill affine matmul with one elementwise
activation between them; the shared expert's int8 rows and its group-64 bf16
scales/biases are that kernel's layout already.

- shared-expert GPU: **8.9 s → 4.6 s**, 2/2 paired rounds;
- prefill wall: **38.5 s → 33.4 s** (−13.3 %), 2/2.

`TINYTITAN_PREFILL_SHARED_GEMM=0` restores the per-token path.

### A deeper routed-tile pipeline

The routed-tile scheduler shipped a one-tile lookahead; a second pending tile
measured **51.65 s → 49.95 s** over four order-rotated rounds (4/4 faster).

### Also in this release

- `TINYTITAN_PREFILL_TILE_EXPERTS` and `TINYTITAN_PREFILL_TILE_DEPTH` expose the
  routed-tile shape for measurement; the shipped default is 8 experts, depth 2.
- `TURBO_FIELDFARE_PHASES` now splits the prefill trace into attention and
  router, so the two stages can be told apart in a run.
- `benchmark/tinytitan_3bit_probe.py` no longer crashes on the first 1-D tensor
  it samples; its added-bits column is produced for the first time.

### Checks not run

Golden baselines are re-run only against installs already present under
`models/`; the release runbook never fetches a model. On the cut host the three
installed targets were checked, and all three produced output identical to
their stored baseline: `ornith-4`, `ornith-8` and `qwen38-4`.

Every other target has no install on this machine, so it was **not checked**:
`qwen36-4`, `qwen36-8`, `qwen38-8`, `agentworld-4`, `agentworld-8`,
`katcoder-4`, `katcoder-8`, `qwen35-2b-4`, `qwen35-2b-8`, `qwen35-4b-4`,
`qwen35-4b-8`, `qwen35-9b-4` and `qwen35-9b-8`.

### Performance

Measured on the 24 GiB M3 with paired arms on the commits listed above, not
re-run at cut time:

| phase | before | 5.30 |
| --- | ---: | ---: |
| prefill, 2.2K prompt | ~48.5 s | **~33.4 s** |
| full-attention GPU | 17.7 s | 7.8 s |
| shared-expert GPU | 8.9 s | 4.6 s |
| decode | unchanged | unchanged |

The decode path is untouched by this release: cold tiers at 2-bit and 3-bit were
built and measured, and both were **rejected on measurement** — the model's
hidden states sit close enough to the fp16 ceiling that quantisation noise
compounds into overflow (2-bit fails at 8 tiered layers, 3-bit at 40). The tooling
and the full record live in `docs/residency-and-bottleneck-verdict-2026-10-20.md`.

### Checksum

`tinytitan-5.30-macos-arm64.tar.gz` sha256: `SHA256_PENDING`
`tinytitan-5.30-macos-arm64.tar.gz` size: `ARCHIVE_BYTES_PENDING` bytes
`tinytitan-lib-5.30-macos-arm64.tar.gz` sha256: `LIBRARY_SHA256_PENDING`
`tinytitan-lib-5.30-macos-arm64.tar.gz` size: `LIBRARY_BYTES_PENDING` bytes
`tinytitan-5.30-tools.tar.gz` sha256: `TOOLS_SHA256_PENDING`
`tinytitan-5.30-tools.tar.gz` size: `TOOLS_BYTES_PENDING` bytes
