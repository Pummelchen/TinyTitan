# Project Maya — what transfers to TinyTitan

Read-only scan of [mw00/project-maya](https://github.com/mw00/project-maya)
(MIT), cloned 2026-10-20 at `main`. No Maya code is copied here; the goal is to
separate the ideas worth rebuilding in Swift/Metal from the ones TinyTitan has
already measured and closed.

## Why this repository is unusually relevant

Maya's engine is a rewrite of [Strata](https://github.com/Niko1221/Strata), and
Strata was **a native engine for Qwen3.8-Flash-Next** — the same architecture
TinyTitan runs at 125B-A6B. Maya's own port study says so explicitly
(`docs/GLM5-FLASH.md` §2: 48 layers, 512 experts, top-10, PLE n-gram table, MTP,
QSA). So its expert tiers, KV streaming and speculation stack were built around
our model's shape, not a generic transformer. That is why the comparison is
worth the time.

## What TinyTitan already has, so this scan does not re-propose it

| Maya mechanism | TinyTitan equivalent | Status here |
| --- | --- | --- |
| VRAM expert cache with aged-use eviction | aging-LFU with a decayed count and half-life, LRU as the alternative (`ExpertCacheTypes.swift`) | shipped |
| Predict next layer's experts, prefetch | prefetch ring, route-ahead | measured: I/O 99.9 % hidden; closed |
| Prompt/conversation cache | disk prompt cache, 13× TTFT on a resumed session | shipped |
| Several requests at once | `--max-concurrent-sequences`, `produceBatch` | admitted, **not fused** (see A) |
| CPU computes some experts | `CPUExpertFFN` + `docs/cpu-coexecution-plan.md` | measured: ~6 % ceiling, CPU work slows the GPU; parked |
| Page cache instead of `F_NOCACHE` | `TINYTITAN_BOUNDED_IO=0` profile | +4.47 % at 8-bit, −2.1 % at 4-bit (24 GiB) |
| 2–3-bit experts / hot-cold tiers | tiered repack, 2-bit and 3-bit cold tiers | closed: model overflows fp16 |
| MTP / speculative decode | MTP head, speculation off | measured negative/below break-even |
| Quantised PLE table | planned (`docs/ngram-table-sharing-plan.md`) | not built — see B |

The useful part of Maya is therefore not its tier *capacity* story (a 24 GiB
Mac has no spare pool to tier into) but four specific mechanisms TinyTitan has
not built, plus a product habit.

## A. Fused batched decode — the largest open win

Maya: `--parallel N` runs **one token of each request through the model
together**. Every dense weight is read once for all rows, attention and routed
experts run per request. Measured aggregate: ×1.41 for two and ×1.35 for four
on a V100 that streams most experts from RAM; ×1.67 / ×2.02 on a machine that
holds every expert in GPU memory (`docs/SETTINGS.md` "Several requests at
once").

TinyTitan today: the server admits up to `--max-concurrent-sequences`, but
`RealForwardRunner.produceBatch` is documented as "the correctness-first batched
entry point" and **loops `produce()` once per row** — there is no fused forward
pass, and a grep finds no caller. So concurrency time-slices and does not share
the 1.92 GB of `model_weights.bin` that every token re-reads. The continuous
batching design (`docs/plan-continuous-batching.md`) states the amortisation
argument but the engine half is not there.

Why it matters here: decode is byte-bound on the GPU's DRAM traffic — 4-bit
18.34 tok/s vs 8-bit 9.27, 2.45× the bytes for 2.11× the body time
(`docs/residency-and-bottleneck-verdict-2026-10-20.md`). A dense-weight read
shared by B rows cuts the dominant per-token bytes by up to B; expert reads are
per-row but route sets overlap, and the prefill kernels already run `t` rows.

The shape of the TinyTitan version:

1. one `produceBatch` that issues the embed, the dense projections and the head
   once for B rows (reuse `encodeAffineProjection(..., tokenCount: B)`);
2. per-row attention and KV (the slots already exist), one expert-fetch plan
   per layer over the **union** of the rows' routes, one grouped MoE pass;
3. logits per row, and the byte-identity gate at B = 1.

**Measured 2026-10-20 — the first slice, in `TinyTitanBench`.** The head is the
largest single dense op in decode, and the engine already carries a two-row
kernel for MTP. `head_affine8_two_rows` runs the same Qwen3.8 head shape
(248,320 x 2,560, 8-bit) through it:

| arm | per launch | per row | head throughput |
| --- | ---: | ---: | ---: |
| `head_affine8` (1 row) | 8,020 us | 8,020 us | 84.2 GB/s |
| `head_affine8_two_rows` (2 rows) | 8,645 us | **4,322 us** | **+85.6 %** |

The weights are read once for both rows, so a pair costs almost what one row
costs. At B = 2 that removes ~4-7 ms/token of the head from an Ornith or
Qwen3.8 decode, 7-12 % of the token, once the server's concurrent path drives
the batched head. The rest of the plan above is unchanged; this measurement is
what makes the head worth wiring first, before the layer path is restructured.

Effort is a project, not a knob, and it is the one idea on this list that
changes a product story ("several agent sessions on one Mac") rather than a
percentage.

## B. Quantise the PLE n-gram table (102.4 GB → ~29 GB)

Maya/Strata carry the same Qwen3.8-Flash-Next PLE table at **28.8 GB**
(`docs/GLM5-FLASH.md` §2), read through an unbuffered async reader with a
bounded row cache (`src/ngram/ple_reader.cpp`). 28.8 GB from a 95.4 GiB fp16
source is about a 6-bit-per-weight footprint, and Maya's DeepSeek engine says
its Engram tables ship explicitly at Q6_K. TinyTitan's table is
`ngram_table.bin` = **102,400,491,520 bytes** (95.4 GiB fp16) on the installed
125B — 63 % of that install, and the plan notes it is identical in every
quantisation.

A ~6-bit table is ~3.5× smaller: ~29 GB, saving ~73 GB per 125B install, and
bringing the table inside the page cache of 64 GB machines instead of being a
102 GB stream. The row is 320 bytes (160 fp16), so the reader is already
row-addressed; a quantised row plus its per-block scales is a small change to
`NgramTableReader` and the converter, not to the engine's layout.

Maya's version also keeps the PLE gathers out of the normal read path with a
dedicated reader and latency stats — TinyTitan's reader is already separate.
This is the cheapest large win on the list: disk, download, page-cache pressure,
and a smaller gather footprint.

## C. KV streaming: host-resident cache, VRAM window

Maya: each attention layer's latent KV lives in pinned RAM; N positions stay in
VRAM (`--kv-resident`, ≥20480). After the indexer picks a token's 2,048
positions, the selected 4-position blocks not in VRAM are copied in; prompts
read the layer's cache from a staging copy filled while earlier layers compute
(`docs/SETTINGS.md`). At 512K on a 24 GB 3090 this grows the expert pool from
714 to 1,176 slots, makes prompts 4–9× faster, and **1M context starts only
with it**; decode +5 %; 1M on three GPUs +16 %.

TinyTitan's own arithmetic says 1M at 4-bit KV is 6.3 GB and 2M is 12.3 GB,
against a model the plan already calls 15.6 GB resident —
`docs/plan-longrope-2m.md` concludes 2M does not fit 24 GB. Maya's design is the
missing piece: the KV is the part that grows with context, and it is the only
large allocation that can live outside the GPU working set.

The Mac translation differs from Maya's: there is no PCIe hop, so "host KV" is
a non-MTL allocation or a shared buffer, and the window copy is a memcpy over
the same memory. The win is not copy bandwidth, it is the **Metal working-set
budget**: freeing 6.3 GB at 1M is 6.3 GB more expert cache, and it lets a 24 GB
Mac accept a context it currently refuses. This is a capability feature, not a
percentage.

## D. Persisted expert profile, warm start and promotion fill

Maya keeps `<pack>/expert-profile.bin`: routing counts that grow with every
answer and are saved after each one; a new pack is seeded with a shipped
profile whose counts weigh about 2,000 tokens of real use; at start the
most-routed experts go to VRAM and are read ahead into RAM
(`docs/SETTINGS.md` "the starting expert profile"). At runtime,
`MAYA_GLM_PROMOTE` moves experts between tiers per token, `PROMOTE_MIN` gates a
promotion on an **aged** route count so one-off routes stop churning the pool,
and `PROMOTE_FILL` refills slots a prompt borrowed.

TinyTitan's cache is aging-LFU and already has the right score; what it lacks is
persistence. Its measured long-prompt hit rate is 0.786 against 0.994 on a short
prompt, and the open item is exactly "cross-request expert warm start — the pin
file alone measured null, so this needs the timing argument". Maya's design is
that argument: the profile is not a static pin, it is a running count written
after every request and used to order the first reads of the next one. The
cheap experiment is a per-install profile file loaded into the existing LFU
scores at runner start, plus a bounded read-ahead of the top-N experts during
load; measure TTFT and decode on a cold process for a repeated agent prompt.

**Implemented 2026-10-20** (the first piece of this scan to ship): the counts
are saved beside the install, seed the LFU score at open, and are restored at
the prefill-to-decode handover; a long-lived server adopts the counts as its
next prior. Measured +40.6 % decode on a repeated 2.2K prompt, +29 % on a
different prompt in the same domain, +17 % across domains, with 63–98 % fewer
expert reloads. Details and the paired tables:
`docs/expert-profile-warm-start-2026-10-20.md`.

## E. Calibration as a product feature

Maya's `--calibrate` spends 10–15 minutes measuring the real machine — CPU vs
link split, thread counts, read chunk sizes — on prompts it has not seen before,
keeps a setting only when it is **more than 3 % faster**, and stores the result
per (PC, model, context) in `calibration.json`.

TinyTitan's defaults are measured on one 24 GiB M3 and its own docs say an M1,
M4 or a different SSD is untested. The knobs that matter here are few (prefill
tile depth, q8 attention, cache slots, reader threads, bounded I/O), so a
`tinytitan calibrate` could be much shorter than Maya's and would replace "the
defaults were measured on the author's machine" with a per-machine answer. The
3 % keep rule and the fresh-prompt discipline are the parts worth copying.

## F. Prompt-progress streaming

Maya streams how much of a prompt has been read (`"return_progress": true`) on
chat completions. TinyTitan's prefill is 33.4 s for a 2.2K agent prompt — the
largest single latency term in the product — and the client sees nothing until
it finishes. The engine already emits per-chunk phase timings; exposing a
`prompt_progress` event (tokens read / total) is small and directly improves the
agent experience.

**Implemented 2026-10-20.** `"return_progress": true` on a streamed chat
completion now emits a `chat.completion.chunk` with empty `choices` and
`{"prompt_progress":{"tokens":done,"total":total}}` for every prefill chunk
before the first content delta; without the flag the stream is byte-identical
to before. The flag survives validation and derived requests, and the
Anthropic and Responses streams ignore the event (they have no shape for it).

## G. Coarse conversation checkpoints and slots

Maya keeps conversation states on SSD (`MAYA_GLM_SLOTS=4`) so returning to one
does not re-read its prompt, and interior KV checkpoints in RAM
(`MAYA_GLM_CKPTS`) so an edited last turn restores the longest matching prefix.
Its measured lesson is about **granularity**: on a 100K-token read, checkpoints
every 16K cost 148.8 s, every 32K 134.9 s, and none 93.2 s — each piece pays a
fixed fill cost, so coarser is faster (`docs/SETTINGS.md`).

TinyTitan's disk prompt cache is prefix-only and needs boundary logits for a
zero-token resume. Interior checkpoints would cover the "edit the last message"
case; Maya's numbers say to place them coarsely (system prompt end, last turn
start) rather than every N tokens.

## H. Large-memory Macs: stop bypassing the page cache

Maya's hard-won rule is "retention is the whole game, and random-access hints
are the enemy": an expert mapping with `FILE_FLAG_RANDOM_ACCESS` measured
**1.93 GB/s, 14× off**, because it tells the cache manager to drop the pages
again (`docs/GLM5-FLASH.md` §4). The mapping is left hint-free, and
`posix_fadvise(WILLNEED)` prefetches the routed blobs at layer start.

TinyTitan does the opposite by design: expert and n-gram reads open with
`F_NOCACHE`, so nothing enters the unified buffer cache. That is right on a
24 GiB Mac, where the page cache competes with the 10.55 GB expert pool and the
measured page-cache profile was −2.1 % at 4-bit. It is the wrong default on a
64 GB or 128 GB Mac, where the whole 19 GB install could stay resident and
decode would stop touching the SSD. A machine-size-aware switch (or a
calibration result) that turns caching on above, say, 2× the install size is a
cheap experiment with a potentially large payoff on hardware TinyTitan supports
but has never measured.

## I. Speculation: the cost model, not the drafter

Maya's DFlash2 adds a 0.7 GB drafter and a confidence-gated window; it is +19 %
on one 3090 and +44 % on nine GPUs, but slower where most experts come from RAM,
so the installer **measures it and keeps it only when >5 % faster**. Its
controller prices a window with a cost model that includes the number of
**distinct experts** a longer window touches
(`src/spec/controller.cpp`), which is precisely why speculation struggles on a
byte-bound engine.

TinyTitan measured MTP negative and turned it off. The transferable piece is
not DFlash2 but the controller's cost model: if speculation is ever revisited,
gate it on predicted distinct-expert growth, not on acceptance alone. Low
priority against A–D.

## What does not transfer

- **CPU expert lane.** TinyTitan built and measured it
  (`docs/cpu-coexecution-plan.md`): the GPU is 2.1× faster per expert, the CPU
  share caps at ~6 %, and concurrent CPU work made the GPU 45 % slower in the
  decisive test. Maya's lane exists because a discrete GPU pays PCIe; Apple
  Silicon has no such hop. Parked, correctly.
- **Residency / complete-pool mode.** Measured on this host: 4-bit resident is
  2.8 vs 12.9 tok/s. No tier policy fixes a 19 GB working set on a 24 GB host.
- **2–3-bit experts.** Maya's per-expert bit allocation is a real idea, but
  TinyTitan's own cold tiers are closed numerically: 2-bit fails at eight
  tiered layers, 3-bit at forty.
- **Multi-GPU split, tensor parallel, PCIe lanes, pinned host memory, CUDA
  graphs, Windows page-file/VRAM reserve.** No hardware analogue here.
- **Disk-skip of low-weight experts.** It changes answers; TinyTitan's golden
  gates and product promise assume it does not. Interesting only as an explicit
  cold-start mode, and even then a product decision.

## Suggested order

| # | Experiment | Gate |
| --- | --- | --- |
| 1 | Adopt Maya's 3 %/fresh-prompt discipline in a first `tinytitan calibrate` over the existing knobs | a per-machine profile beats the shipped default on ≥1 of 3 Macs tested |
| 2 | Persist the LFU profile and read ahead the top-N at load | cold-process TTFT and decode on a repeated agent prompt |
| 3 | Q6_K PLE table | output equivalence on a fixed battery + install shrinks ~73 GB |
| 4 | Fused `produceBatch` (B = 2, 4) | aggregate tok/s vs serial; B = 1 byte-identical |
| 5 | Host KV + VRAM window | 1M context starts on 24 GB; attention parity at 128K |

Items 1–3 are cheap and independently useful; 4 and 5 are the projects.
