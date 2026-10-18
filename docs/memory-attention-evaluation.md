# Memory Attention (arXiv 2609.28399) evaluated against TinyTitan

2026-10-18. Source: arXiv 2609.28399v2, *Memory Attention*, Jiale Kang
(Yuanshi Intelligence), 14 Oct 2026. Read in full (v2 HTML).

## Verdict first

**Nothing to implement today, and one thing to steal for the n-gram path.**

MA is a *training-time* architecture change: it deletes `W_V` and rebuilds
values from a vocabulary-sized table that has to be learned jointly with the
rest of the model. TinyTitan runs published Qwen-family checkpoints and has no
training path, so no existing model can adopt it. The paper's headline
mechanism — CPU offload of the tables — is also already *less* interesting here
than it is on an H800, for a reason specific to these models (below).

What is worth taking:

1. A retraining-free, measured ~1% win for the existing 95.4 GiB n-gram gather:
   put the prefill reads on a parallel pool, 3.68 s -> 0.32 s per 2K chunk
   (§"What transfers today"). Small, but it is measured and safe.
2. A note for the format/repacker roadmap: a token-indexed table is already a
   first-class artifact, so an MA-style model would be unusually cheap to
   support — and unusually well matched to this engine's strengths.
3. A useful external data point that the "capacity in storage, not in dense
   compute" bet is being taken seriously by the architecture literature.

## What the paper does

Per attention layer, replace the value projection with

```
V = K + Norm(E[s])          E ∈ R^{N x d_v},  N = vocabulary
```

`K` keeps the context dependence (it is the key projection, pre-RoPE); the
table row supplies the token-specific part. At inference `Norm` is folded into
the table row-by-row, so online value construction is **a gather plus an add** —
no matmul. Because the row index is just the token ID, rows can be prefetched
before the hidden states exist, which supports keeping the table in CPU memory
(MA-Offload). MA-Recall is an analysed-only extension that drops the persistent
V cache and reconstructs `V = K + E[s]` from cached keys and token IDs, halving
KV storage.

Reported results: at matched *token* budgets, +0.6 to +1.2 points average
downstream over seven zero-shot benchmarks, large NIAH gains (e.g. 62.4 → 98.2
on niah_single_3 at 1K), and offload latency that is within noise of the
baseline. The gains come with 3–8x more parameters.

## What the paper does not establish

These matter before anyone spends project time on it:

- **No parameter-matched control.** 373M Standard vs 1,135M MA is the first
  pair; the authors say so plainly ("they do not isolate the contribution of
  its structure from the increase in parameter capacity"). A 3x-bigger baseline
  is not in the paper.
- **Toy scale, no MoE.** 24 layers, hidden 1,024/2,048, 10–20B tokens.
  MoE and larger backbones are explicitly "ongoing".
- **The latency result is a tie, not a win.** Prefill 90.686 vs 90.970 ms,
  decode 17.189 vs 17.636 ms, single H800, one operating point, prototype.
- **Offload is measured against PCIe, not against SSD.** The 3 GiB of BF16
  tables moved to CPU RAM. That is the easy case: host-to-device over PCIe
  Gen5 x16 is tens of GB/s, and the table stays in a random-access medium.
  TinyTitan's tables live on an SSD it measures at ~3.6 GB/s.

## Why the offload story is weaker here

The reference geometry in this tree — `qwen3.8-flash-next_125B_A6B_4Bit`:

| quantity | value |
| --- | --- |
| layers / layers with attention values | 48 / **12** (36 are GDN linear, no V at all) |
| vocabulary `N` | 248,320 |
| value width `d_v = numKVHeads x headDim` | 2 x 256 = **512** |
| MA table per layer | 127.1M params (0.24 GiB bf16) |
| MA table, all 12 layers | **2.84 GiB bf16 / 0.71 GiB int4** |
| table bytes read per decoded token | 12 KiB bf16 / 3 KiB int4 |
| that, at 5.46 tok/s | **0.07 MB/s** |
| routed-expert bytes per decoded token | **1.24 GiB** |

The table is ~1:100,000 of the bytes the decode loop already streams. There is
nothing to overlap, because there is nothing to hide: a 0.71 GiB INT4 table
does not need to be offloaded at all, it needs to be resident. MA-Offload is a
solution to a problem this engine does not have with these shapes.

Deleting `W_V` is likewise a rounding error *in traffic terms*: 15.73M params
across the 12 attention layers is 7.5 MiB read per token at INT4, **0.59%** of
per-token weight traffic. The paper's own framing agrees — reduced
value-construction arithmetic "does not uniformly translate into lower
latency".

The one place the arithmetic could matter is a dense model at batch 1, where
`W_V` is re-read for every token and there is no expert streaming to hide
behind. That is not the workload this repo is optimizing.

## The one real prize, and why it is still not action

MA-Recall halves the V cache: at 100k context the V half of the 12 attention
layers is 1.14 GiB, so ~0.57 GiB comes back. Note *why* that would be worth
something here: not bandwidth — `docs/lmcache-evaluation.md` already
establishes KV is resident and not the constraint — but **RAM budget**, since
context length costs decode speed indirectly by crowding expert-cache slots
(0.759 → 0.682 hit rate). 0.57 GiB is a few more expert slots.

It is still not actionable: the table does not exist in any shipped checkpoint,
and the reconstruction cost (`L x B x R x d_v` adds, plus RoPE handling) is
paid per step against a decode loop that is already 84 ms/token of exposed
expert wait. Revisit only if a model trained with MA ever ships.

## What transfers today, without retraining

**Put the prefill n-gram gather on a parallel read pool.** The paper's
transfer-budget formula (eq. 18) counts its payload *"without caching or
deduplication"*, i.e. one logical read per row. The equivalent question here is
what those reads cost in wall-clock when they are issued serially.

`NgramTableReader.gather` issues one `pread` per row, serially, with
`F_NOCACHE`. A token needs `headsPerNgram * (ngramSize - 1) = 16` rows of 320 B.
`gatherPLERowsPrefill` loops the chunk and calls that same serial gather per
token, so a 2K chunk is **32,768 serial `pread`s on the prefill critical path**,
before layer 0. Decode pays the same 16 reads per token, already measured at
`gather_ms = 2.3`.

Measured directly against the installed 95.4 GiB table, `F_NOCACHE`, on this
machine. Each 320 B read pulls a 4 KiB page:

| variant, 2048 tokens | wall | per token | speedup |
| --- | ---: | ---: | ---: |
| serial, as today | **3,680 ms** | 1.80 ms | 1.0x |
| serial, rows sorted by offset | 3,881 ms | 1.90 ms | **0.9x — no help** |
| 2 threads | 1,644 ms | 0.80 ms | 2.2x |
| 4 threads | 884 ms | 0.43 ms | 4.2x |
| 8 threads | 491 ms | 0.24 ms | 7.5x |
| 16 threads | **317 ms** | 0.15 ms | 11.6x |

Single 320 B cold read: p50 **103 us**, p90 112 us — which matches 16 reads x
103 us = 1.65 ms/token against the decode counter, so the micro-benchmark is
measuring the same thing the engine does.

Two things follow, and the first one contradicts what this doc said before it
was measured:

- **Sorting does not help.** 32K rows sampled from 16 segments of ~20M rows have
  essentially no adjacency to exploit; sorting costs 5% and buys nothing. The
  access pattern is random by construction.
- **Concurrency is the whole win.** 103 us per read against a 320 B payload is
  latency, not bandwidth: the chunk requests 10.5 MiB (134 MiB of pages) over
  3.7 s, ~36 MiB/s against an SSD measured at 3.6 GB/s. The device queue is
  under-filled; the wait is round-trips, not throughput. Depth 16 is 11.6x and
  still scaling.

Deduplication is a second-order lever, not the first: two different 3-grams
collide in a 20M-row segment with probability ~0, so the only duplicate rows come
from *repeated* 3-grams (boilerplate, indentation, a shared system prompt). That
is real but bounded, and it needs instrumentation to size. At the measured
per-read cost, 10% duplication is worth ~0.35 s; changing the scheduling is
worth 3.3 s.

**What it is worth, measured in the engine: ~2.3% of prefill.** The section
above was written from a micro-benchmark and a 1%-ish guess. Both were wrong, so
here is the measurement that replaced them.

`gather_ms` used to time only the decode gather, so prefill cost was invisible.
It now covers both paths. With that in place, the same 412-token prompt on the
release build (6,592 row reads, 16 per token), two runs per arm:

| gather depth | `gather_ms` | vs serial |
| --- | ---: | ---: |
| serial (previous code) | 647, 673 (660) | 1.0x |
| 4 readers | 176, 192 (184) | 3.6x |
| **8 readers (shipped)** | 101, 117, 109, 125 (109-117) | **5.6x** |
| 16 readers | 109, 113 (111) | 6.0x |

Depth 8 is the knee: sixteen buys 4 ms and no more. The gather was 660 ms of a
~23.5 s prefill (2.8%) and is now ~0.11 s (0.5%), so the change recovers about
**0.54 s, or 2.3%, of prefill** — per 412 tokens, and it scales with prompt
length. Release prefill runs ~59 ms/token, not the 145 ms/token the 58-token
figure in `docs/qwen38-decode-12gb-measurements.md` implies; that run was
dominated by fixed cost.

The end-to-end total is not separable from run-to-run noise (23.0-25.5 s across
arms), which is why the counter, not `prefill_s`, is the evidence here.
Deduplication stays second order: only repeated 3-grams collide, and the
scheduling change was worth 6x on its own.

Decode was left serial. Its gather is 16 reads and its cost is workload- and
cache-dependent: 4.4-4.6 ms across 65 generated tokens on a filler-word prompt
(the n-grams repeat, so they come back warm), against ~1.6 ms/token when the
rows are cold. Parallelising a handful of reads whose cost is usually a cache
hit would add dispatch overhead to the hot loop for at most ~1% in the cold
case — and in that case it competes with expert I/O for SSD queue slots. Not
worth it without a workload that shows otherwise.

**Status: implemented.** `NgramTableReader.gatherConcurrently(batches:chunkStride:into:threads:)`
fans independent batches across `concurrentPerform` workers (default depth 8),
and `gatherPLERowsPrefill` resolves the chunk's rows in token order, then fetches
them concurrently, accumulating into `gather_ms`.

Verified 2026-10-18: `swift build` clean (all products); 8/8 in
`NgramTableReaderTests`, including byte-for-byte equivalence with the serial
gather plus a per-batch addressing spot check, and the single-batch and
empty-input degenerate sizes; `func-length`, `file-length`,
`unchecked-sendable`, `force-cast` and `docs` gates green; and the 125B install
runs to exit 0 with coherent output, reproducing 109-125 ms for the gather where
the previous code measured 647-673 ms.

## Why this architecture suits this engine (roadmap note)

MA models have the property TinyTitan is built around: **their extra capacity is
token-indexed, so it is addressable one step ahead and off the critical path.**
That is exactly the `NgramTableReader` argument, already written down in the
source:

> a row id depends only on token ids ... The rows for token N+1 are therefore
> knowable the moment token N is sampled, so this read can be issued entirely
> off the decode critical path — where an expert miss cannot be, because routing
> is not known until the layer runs.

The install format already carries a token-indexed table as a first-class
artifact: `ngram_table.bin` is 95.4 GiB of a 56-file manifest, with
`ple_constants.json` as its geometry sidecar, and the read path is a
`pread`-based row gather. That gather is fp16-only: `EmbedLookupInt4` is the
*resident* token-embedding lookup, not a quantized variant of this path, so an
INT4 MA table would need a quantized gather it does not have today. An MA table
is a strict subset of that: `N x d_v` per attention layer, one sidecar, 0.71 GiB
total at INT4. If a Qwen-family or TinyTitan-native model ever ships MA (or
Bank of Values, its coarser cousin), the engine-side work is a new table kind in
the repacker plus a reader — not new machinery.

That is a reason to keep the table artifact generic in the repacker, not a
reason to build anything now.

## Recommendation

| | |
| --- | --- |
| Adopt MA as an architecture | No — no training path, no shipped checkpoint |
| Implement MA-Offload | No — 0.71 GiB INT4 table is resident-sized here |
| MA-Recall / KV halving | Not yet — needs a trained model; 0.57 GiB at 100k ctx |
| Parallelize the prefill n-gram gather | **Done** — 660 ms -> 109 ms per 412-token prefill at depth 8, 2.3% of prefill |
| Dedup the n-gram gather | Second order — only repeated 3-grams; instrument before building |
| Sort the gather by row offset | **No** — measured 5% slower, no adjacency to exploit |
| Keep the token-indexed table artifact generic | **Yes** — cheap now, unblocks an MA/BoV model later |
