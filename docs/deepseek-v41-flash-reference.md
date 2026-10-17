# DeepSeek-V4.1-Flash — checkpoint reference

Everything this project has established about
[`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash),
in one place: identity, geometry, every tensor family with its shapes and
encodings, the reference forward's algorithm, the chat/prompt encoding, the
vision tower, the KV-cache design, and the open questions.

**Scope.** This is a *reference*, not a plan. The integration plan — what
TinyTitan would have to build, at what size, under which precision policy — is
[`deepseek-v41-flash-port.md`](deepseek-v41-flash-port.md). Read that for
decisions; read this for facts.

**Revision note, 2026-10-17.** The plan now targets
[`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).
Everything recorded here was read from the 2026-09-10 `…-Flash` checkpoint, and
that fact base still stands: 1017's text and vision geometry, its 48-shard
510.3 GB layout, its 763,205,315,794 declared parameters and its dtype mix are
identical to the base's. What differs is prompt encoding — 1017 ships no Jinja
template, while the base gained one on 2026-10-16 — and the benchmark scores.
`deepseek-v41-flash-port.md` §1.1 carries the measured delta.

**Provenance.** Every value was read from the published checkpoint or from the
official `inference/` reference code shipped in the same repository. Retrieval
was by HTTP: the HF API (`/api/models/...?blobs=true`), the recursive file tree,
`config.json`, `tokenizer_config.json`, `model.safetensors.index.json`, and the
`inference/` Python sources; plus the safetensors headers of individual shards,
fetched with ranged requests. Each item below is tagged:

- **CONFIRMED** — read directly from a source; the source is named.
- **INFERRED** — my reasoning over confirmed values; the reasoning is stated.
- **NOT FOUND** — looked for and absent; an absence, not a value.

Nothing is guessed. Where two sources disagree, both are recorded (§13).

**Naming: one convention in this project, one mapping table.** The repository
ships **two** config files that use *different names for the same fields* — root
`config.json` (transformers naming, and the published contract) and
`inference/config.json` (the reference implementation's own naming).

**This project uses the root `config.json` names everywhere**, in documentation
and in any code written from it. That is the file the checkpoint actually ships
and the file a converter reads. The reference names are quoted only where the
behaviour being described lives in the reference code, and never as an
alternative name for the same field. The mapping, with which side is
authoritative:

| Field | **Authoritative** — root `config.json` | Reference-only alias — `inference/config.json` |
| --- | --- | --- |
| rope dim | `qk_rope_head_dim` | `rope_head_dim` |
| window | `sliding_window` | `window_size` |
| shared KV owners | `kv_source_layer_ids` | `kv_source_layers` |
| shared index owners | `index_source_layer_ids` | `index_source_layers` |
| candidate source | `candidate_source_layer_id` | `candidate_source_layer` |
| router score | `scoring_func` | `score_func` |
| route scale | `routed_scaling_factor` | `route_scale` |
| expert width | `moe_intermediate_size` | `moe_inter_dim` |
| draft layers | `num_nextn_predict_layers` | `n_mtp_layers` |
| top-k experts | `num_experts_per_tok` | `n_activated_experts` |
| engram pad | `engram_pad_token_id` | `engram_pad_id` |
| draft top-k | `dspark_num_experts_per_tok` | `dspark_n_activated_experts` |
| KV sharing, per layer | `kv_source_layer_ids` (list) | `kv_source_layer` (scalar, reference-internal) |
| quantization | `quantization_config` object | hardcoded constants in `model.py` |
| topk method | `topk_method: "noaux_tc"` | field absent |

Rules that follow, so the two files cannot be confused again:

1. A field's authority is the **root `config.json`**. If a value ever disagrees
   between the two files, the root file wins and the discrepancy is a bug to
   record, not a choice to make.
2. `inference/config.json` exists to configure the reference *script*; it is not
   a second description of the model. Never read a geometry value out of it
   without checking the root file.
3. `quantization_config` has **no counterpart** in the reference config; the
   reference hardcodes the same values as constants in `model.py` (§7).
4. `topk_method` is absent from the reference entirely; its semantics are
   implemented in `Gate` and documented in §4.
5. `kv_source_layer` (singular) is a reference-internal scalar, not a rename of
   `kv_source_layer_ids`. The same applies to `candidate_source_layer`,
   `index_source_layers` and `candidate_topk_blocks`.

Upstream is unchanged by any of this: the two files are published artefacts, and
this section is a reader's convention, not an edit to them. If the two naming
schemes are ever a problem for the checkpoint itself, that is DeepSeek's call,
not this project's.

---

## 1. Identity and size

| | Value | Source |
| --- | --- | --- |
| Repo | `deepseek-ai/DeepSeek-V4.1-Flash` | HF API — **CONFIRMED** |
| Pinned sha | `dba1be0a40aa45a94ad051997016db3960a90277` | HF API — **CONFIRMED** |
| Last modified | 2026-09-10 | HF API — **CONFIRMED** |
| `gated` | `false` | HF API — **CONFIRMED** |
| License | MIT | `LICENSE`, card `license: mit` — **CONFIRMED** |
| Pipeline tag | `image-text-to-text` | HF API — **CONFIRMED** |
| Tags | `transformers`, `safetensors`, `deepseek_v41`, `text-generation`, `image-text-to-text`, `license:mit`, `eval-results`, `endpoints_compatible`, `8-bit`, `fp8`, `region:us` | HF API — **CONFIRMED** |
| `architectures` | `["DeepseekV41ForCausalLM"]` | `config.json` — **CONFIRMED** |
| `model_type` | `deepseek_v41` | `config.json` — **CONFIRMED** |
| Nested model types | `deepseek_v41_text`, `deepseek_v41_vision` | `config.json` — **CONFIRMED** |
| `transformers_version` | `5.6.0` | `config.json` — **CONFIRMED** |
| `dtype` (declared) | `bfloat16` | `config.json` — **CONFIRMED** (but see §7: it is a declaration, not the storage format) |
| Popularity | 3,736 likes, 621,396 downloads, 64 discussion threads | HF API — **CONFIRMED** |

### File inventory

48 safetensors shards, `model.safetensors.index.json` (7,470,294 bytes),
`config.json` (3,311), `tokenizer.json` (6,367,257), `tokenizer_config.json`
(801). **CONFIRMED.**

- **No `generation_config.json`** — absent from the recursive tree. **NOT FOUND.**
- **No chat template** — no `chat_template` key in `tokenizer_config.json`, and
  the card states "This release does not include a Jinja-format chat template."
  **CONFIRMED.**
- **`DeepSeek_V41_Tech_Report.pdf`** is present but is a Git-LFS pointer
  (`sha256 ba68e2e40408125ae6d2f63a9a241b61c73910691c74ec1a2a7023c851eac08d`,
  1,809,802 bytes); `web_fetch` returns the pointer, not the PDF. **UNREADABLE.**

Repo `usedStorage`: 510,311,613,821 bytes (≈ 475 GiB). Index `total_size`:
510,286,023,000. **CONFIRMED.**

Shard sizes (bytes): one 970 MB, one 1.32 GB, thirty-eight ≈ 7.39 GB, one
1.32 GB, 2.65 GB, 2.57 GB, 2.71 GB, and **two at 101.5 GB each (shards 47 and
48)**. Those two large shards contain **six tensors each, all Engram**.

### Parameters

| Figure | Value | Source |
| --- | --- | ---: |
| safetensors metadata total | 763,205,315,794 | HF API — **CONFIRMED** |
| BF16 | 1,976,441,856 | HF API — **CONFIRMED** |
| F32 | 42,307,282 | HF API — **CONFIRMED** |
| F8_E4M3 | 204,015,223,296 | HF API — **CONFIRMED** |
| I8 | 557,171,343,360 | HF API — **CONFIRMED** |
| card claim, backbone | "552B backbone parameters" | card — **CONFIRMED as a claim** |
| card claim, Engram | "196B parameters" | card — **CONFIRMED as a claim** |
| card claim, active | "8B active per token during prefill and 16B during decode" | card — **CONFIRMED as a claim** |
| 552 + 196 | 748 B | card's two figures; the metadata's 763.2 B is **15.2 B higher** — reduced but not closed, see the subsection above |

**INFERRED:** the `I8` bucket is the routed experts stored as packed FP4,
because `557,171,343,360 = 384·3·5120·2304·40 + 128·3·5120·2304·3` exactly — the
routed-expert value count for 40 backbone layers × 384 experts plus 3 MTP layers
× 128 experts.

### The 15.2 B gap, reduced as far as the published data allows

The card claims 552 B backbone + 196 B Engram. The metadata reports 763.2 B.
`552 + 196 = 748`, so **15.2 B is unaccounted for**. What can be established:

- **The card's 196 B Engram figure is exactly reconstructible** from the
  published tensor shapes — it matches Engram *values* to **0.6%**:

  | Engram component | Count |
  | --- | ---: |
  | `2 × 384M rows × 256` values | 196.6 B |
  | `2 × 384M rows × 8` E8M0 scales | 6.1 B |
  | `2 × ([25600, 6144] + [800, 192])` | 0.3 B |
  | **values + scales** | **203.0 B** |

  So the "196 B" counts **values only**, excluding the per-block scales, and the
  reconstruction confirms it to within rounding. That also means a reader adding
  Engram to a backbone figure must not add the scales twice.
- **The vision tower is not the gap.** `vision.*` + `aligner.*` is **0.485 B**
  elements (measured from the shard-1 safetensors header: 411,842,560 +
  73,410,560), which is far too small.
- **The metadata's per-dtype buckets do not sum to the card's components**, so
  the two are not two views of one count: the `F8_E4M3` bucket alone is 204 B,
  while Engram values are 196.6 B — leaving only 7.4 B for every attention,
  shared-expert and MTP projection. Those projections measure ≈ 8.3 B elements
  from their own shapes (13 per layer × 40 layers ≈ 6.09 B, plus MTP and shared
  experts), so the buckets do not reconcile with the tensor inventory either.

**Conclusion, and it is deliberately limited:** the card's Engram figure is
verifiable and correct on its own terms; the backbone figure is not
reconstructible from published shapes by the same counting method; and the
metadata's dtype buckets disagree with both. The discrepancy is **in the
published figures**, not in this document's measurements — the file sizes
(510.3 GB total, 288.8 GB experts, 202.8 GB Engram) reconcile exactly with the
shard layout and are the authoritative quantities. Reading 763.2 B as the model's
parameter count and 510.3 GB as its size are both safe; building an arithmetic
argument on 552 vs 748 vs 763.2 is not. The tech report is the only source that
could settle it and it is unreadable (§13).

---

## 2. Text architecture

**40 layers**, presented by the card as a *Causal Encoder-Decoder*: a 20-layer
causal encoder plus a 20-layer decoder. **CONFIRMED (card).** **INFERRED**
mapping: layer 20 is the sole `kv_source_layer` of the decoder, so its
`Compressor` projects the final encoder hidden state into the global KV that
layers 21–39 reuse.

| Field | Value |
| --- | ---: |
| `num_hidden_layers` | 40 |
| `hidden_size` | 5120 |
| `vocab_size` | 129280 |
| `num_attention_heads` | 64 |
| `num_key_value_heads` | **1** |
| `head_dim` | 512 |
| `qk_rope_head_dim` | 64 |
| `q_lora_rank` | 1280 |
| `o_lora_rank` | 1024 |
| `o_groups` | 8 |
| `n_routed_experts` | 384 |
| `n_shared_experts` | 1 |
| `num_experts_per_tok` | 6 |
| `moe_intermediate_size` | 2304 |
| `hidden_act` | silu |
| `swiglu_limit` | 10.0 |
| `rms_norm_eps` | 1e-20 |
| `attention_bias` | false |
| `attention_dropout` | 0.0 |
| `use_cache` | true |
| `tie_word_embeddings` | **false** |
| `max_position_embeddings` | 1,048,576 |
| `rope_theta` | 10000 |
| `rope_scaling` | yarn, factor 16, `original_max_position_embeddings` 65536, `beta_fast` 32, `beta_slow` 1 |
| `compress_rope_theta` | 160000 |
| `sliding_window` | 128 |
| `scoring_func` | `sqrtsoftplus` |
| `topk_method` | `noaux_tc` |
| `norm_topk_prob` | true |
| `routed_scaling_factor` | 1.5 |
| `hc_mult` | 4 |
| `hc_sinkhorn_iters` | 20 |
| `hc_eps` | 1e-06 |
| `num_nextn_predict_layers` | 3 |

**All CONFIRMED** from `config.json`.

Context: 1 M tokens; the card says sparse attention was trained at 64 K and then
extended to 1 M, over 45 T pre-training tokens. **CONFIRMED (card).**

---

## 3. Attention — Compressed Sparse Attention 2 (CSA2)

The labels **MLA / MHA / GQA are NOT FOUND** anywhere in the card or reference
for this model; the card's name is **CSA2** and the mechanism below is what
`inference/model.py` implements. **CONFIRMED.**

### 3.1 Q / K / V projections

**Not** DeepSeek-V3-style MLA. This model has no separate qk_nope/qk_rope
projection tensors.

- **Q is low-rank:** `wq_a` Linear(5120 → 1280) → `q_norm` RMSNorm(1280) →
  `wq_b` ColumnParallelLinear(1280 → 64·512). **CONFIRMED.**
- **KV is not low-rank:** `wkv` Linear(5120 → 512) → `kv_norm` RMSNorm(512),
  **one shared KV head for all 64 query heads**. **CONFIRMED.**
- `nope_head_dim = head_dim − rope_head_dim` (448) is computed in the reference
  and **never used** — the split is positional: the **last 64** of the 512 dims
  are RoPE-rotated, and the attention output receives the **inverse** rotation on
  those same dims so the cache can stay in one rotated form. **CONFIRMED.**
- **`o_groups` = 8:** `wo_a` = ColumnParallelLinear(64·512/8 = 4096 →
  8·1024 = 8192), **bf16**; `wo_b` = RowParallelLinear(8192 → 5120). Computed as
  a block-diagonal grouped projection: `o.view(b,s,8,-1)` →
  `einsum("bsgd,grd->bsgr", o, wo_a.weight.view(8,1024,4096))` → flatten to 8192
  → `wo_b`. Group *g* sees only its own 8 contiguous heads. **CONFIRMED.**
- `attn_sink` = learnable per-head fp32 bias `[64]` added into the softmax
  denominator by the kernel. `softmax_scale = head_dim^−0.5` (512^−0.5), **not**
  the rope-dim scale. **CONFIRMED.**

### 3.2 Two KV sources, one attention call

Every attention layer concatenates two key sources and makes **one** `sparse_attn`
call. Selection is **index-based**, not a learned attention gate. **CONFIRMED.**

**1. Sliding window — every layer, per-layer cache.**
`window_size` 128. `get_window_topk_idxs` builds a ring-buffer index matrix
`[b,m,topk]` with −1 for empty slots; prefill gives one causal window row per
query, decode gives the single query the whole ring oldest-first. Order is
irrelevant, since the kernel treats slots independently. Cache:
`window_kv_cache [B, 128, 512]`. **CONFIRMED.**

**2. Compressed KV — only where `compress_ratios[layer] > 0`.**
A `Compressor` pools `compress_ratio` consecutive tokens into **one** 512-d
latent through a learned softmax gate:

```
kv    = wkv(x)                 # fp32
score = wgate(x)               # fp32
pooled = Σ (kv · softmax(score over the group))
latent = RMSNorm(pooled)
```

`ratio == 1` degenerates to `norm(wkv(x))` — bf16, no gate, no fp32. Incomplete
groups persist across decode steps in `kv_state`/`score_state`; the compressor
returns `None` until a group completes. It returns the latent **pre-RoPE**,
because the indexer needs it unrotated. Cache:
`compress_kv_cache [B, max_seq_len // ratio, 512]`. **CONFIRMED.**

`compress_ratios` (43 entries = 40 backbone + 3 MTP):
`[0, 0, 18×2 (layers 2–19), 20×1 (layers 20–39), 0, 0, 0]`. So layers 0–1 and
all three MTP layers are window-only. **CONFIRMED.**

### 3.3 Hierarchical lightning indexer

Runs **only** on `index_source_layer_ids` = [2, 8, 14, 20, 24, 28, 32, 36].
**CONFIRMED.**

- `index_n_heads` 32, `index_head_dim` 128, `index_topk` 512,
  `softmax_scale = 128^−0.5`. **CONFIRMED.**
- `wq_b` derives its query from the **same** `qr` as the attention query;
  `weights_proj` is 5120 → 32 heads, bf16. **CONFIRMED.**
- `wk` (owners only): 512 → 128 plus RMSNorm(128). **CONFIRMED.**
- Score: `einsum("bshd,btd->bsht", q, index_k)` → **relu** → × head weights →
  sum over heads (all_reduce if sharded). **CONFIRMED.**
- Unreachable compressed positions are masked to −inf, where
  `compress_lens = end_pos // ratio` — "a block becomes visible once the query
  has passed its last token". **CONFIRMED.**
- `topk(min(index_topk, end_pos // ratio))`, then indices are **re-sorted into
  position order**, unreachable → −1, valid ones shifted by `offset` (the
  window-KV length). Returns int32 `[b, s, topk]`. **CONFIRMED.**

### 3.4 Two-level candidate pre-filter

- Level 1: `select_candidate_blocks` scores each block by its **max position
  logit**, pins the partially-filled newest block, keeps the top
  `candidate_topk_blocks` = 2048 blocks (`candidate_block_size` = 8) → bool mask.
- Level 2: layers with `0 <= candidate_source_layer < layer_id` (i.e. 21–39)
  mask their own index scores to those blocks **before** top-k.
- `candidate_source_layer < 0` disables the mechanism.

**CONFIRMED.** Config: `candidate_source_layer_id` 20, `candidate_topk_blocks`
2048, `candidate_block_size` 8.

### 3.5 Cross-layer sharing — the structurally new part

A `SharedAttentionRuntime` singleton (`shared_attn`) carries `compress_kv`,
`index_k`, `topk_idxs` and `candidates` down the stack. **CONFIRMED.**

- Only the four `kv_source_layer_ids` = [2, 8, 14, 20] build a `Compressor` and
  own a `compress_kv_cache`. Groups: **2 → layers 3–7, 8 → 9–13, 14 → 15–19,
  20 → 21–39**.
- Only the kv-source subset of the index sources owns an indexer K cache.
- Only the index sources run an `Indexer` and publish `topk_idxs`; **every other
  layer reads the last published object.**
- **Window/SWA KV is per-layer** — every attention layer has its own ring.
- MTP layers are excluded (`is_backbone = layer_id < n_layers`).

**Which component selects which keys:** the `Indexer` selects the ≤ 512
compressed positions per query; `get_window_topk_idxs` selects the 128 window
slots; `sparse_attn` attends exactly over the concatenation. There is no
logit-based eviction inside attention. **CONFIRMED.**

**What is cached:** `window_kv_cache [B, 128, 512]`; `compress_kv_cache
[B, max_seq_len//ratio, 512]`; indexer `k_cache [B, max_seq_len//ratio, 128]`.
Compressed latents rotate at their group's **first** token position
(`j·ratio`) with `compress_rope_theta` 160000 + YaRN; ratio-0 layers use base
`rope_theta` 10000 with YaRN disabled. **CONFIRMED.**

**`Full` / `Reindex` / `Reuse`** are the card's names for three static layer
modes. The mapping is **INFERRED** from the source-layer lists: Full =
{2, 8, 14, 20} (own compressor + own indexer), Reindex = {24, 28, 32, 36} (own
indexer, shared KV), Reuse = all remaining layers.

**"SWA Bounded Replay"** — the card describes reconstructing missing SWA KV by
replaying only the most recent `n_win` tokens instead of persisting SWA KV, for
≈ 1/8 the persistent footprint. **No code for it exists in `inference/`** — card
prose only. **NOT FOUND.**

---

## 4. Mixture of experts

- **384 routed experts + exactly 1 shared expert per MoE layer** (`assert
  n_shared_experts == 1`), **6 routed activated** per token. MTP layers use
  **128 routed, top-3**. There are **no dense FFN layers** — all 40 backbone
  blocks are MoE. **CONFIRMED.**
- Router: `scores = softplus(linear(x.float(), weight.float()) / gate_temp).sqrt()`
  with `gate_temp` default 1.0. **CONFIRMED.**
- `topk_method: "noaux_tc"` does not exist as a field in the reference, but its
  semantics are implemented: `indices = topk(scores + bias)`; `weights =
  scores.gather(indices)` — the **unbiased** scores; normalized by
  `sum + 1e-20` when `norm_topk_prob` and topk > 1; then `weights *= 1.5`
  (`routed_scaling_factor`). **The selection bias never scales the weights.**
  **CONFIRMED.**
- The gate carries **two** bias vectors of shape `[384]`: `bias` (renamed from
  `e_score_correction_bias`) and `bias_vl`, the latter tied to "training
  `noaux_tc_for_vl`" and used when vision is enabled. **CONFIRMED.** When
  exactly a text-only forward should use which one is **NOT FOUND** (§13).
- Expert = SwiGLU `w1`/`w2`/`w3` at `moe_intermediate_size` 2304. The shared
  expert is a plain `Expert(dim, 2304)`; routed experts are FP4. Routed outputs
  are summed in **fp32**, all_reduced, then the shared expert is added.
  **CONFIRMED.**
- **`swiglu_limit` = 10.0**: `up = clamp(up, −10.0, +10.0)` (both sides);
  `gate = clamp(gate, max=10.0)` (upper only); then `silu(gate)·up`. The
  reference comment ties the clamps to keeping fp8/fp4 activations in range.
  **CONFIRMED.**

---

## 5. Engram — conditional memory

The largest and least conventional subsystem: **196.6 B parameters, 202.8 GB as
stored, 40% of the checkpoint.**

### 5.1 Structure

- Two tables, at `engram_layer_ids` = **[1, 14]**. **CONFIRMED.**
- Row counts `engram_num_embeddings` = **[384006168, 384016682]** — **~384 M rows
  per layer**, not 16 M. **CONFIRMED.**
- `engram_head_dim` 256, `engram_n_heads` 8, `engram_max_ngram_size` 4,
  `engram_compressed_vocab_size` 99092, `engram_pad_token_id` 2. **CONFIRMED.**
- `engram_vocab_size` = 16,000,000 is **not** the row count — it is the starting
  value for a prime-bucket search. **INFERRED** from `inference/engram.py`: per
  layer the table is partitioned into `(max_ngram_size − 1) × engram_n_heads`
  = 3 × 8 = **24 prime-sized buckets**, 48 primes across two layers, drawn in
  order and never reused, each ≈ 16.0 M rows (`384006168 / 24 ≈ 16,000,257`).
- **Engram row size:** 256 E4M3 bytes + 8 E8M0 bytes = **264 bytes/row**.
  **INFERRED** from the confirmed shapes.

### 5.2 Tensors

| Tensor | Dtype | Shape |
| --- | --- | --- |
| `layers.{1,14}.engram.embed.weight` | F8_E4M3 | `[part_rows, 256]` |
| `layers.{1,14}.engram.embed.scale` | F8_E8M0 | `[part_rows, 8]` |
| `layers.{1,14}.engram.wkv.weight` | F8_E4M3 | `[25600, 6144]` |
| `layers.{1,14}.engram.wkv.scale` | F8_E8M0 | `[800, 192]` |
| `layers.{1,14}.engram.q_weight` | BF16 | `[4, 5120]` |
| `layers.{1,14}.engram.k_weight` | BF16 | `[4, 5120]` |

**CONFIRMED** from the safetensors headers of shards 47 and 48. Note
`wkv` in = `n_hash_cols · 256` = 24 · 256 = 6144, out =
`dim · (hc_mult + 1)` = 5120 · 5 = 25600.

### 5.3 The lookup

1. **Token compression.** Every token id maps to a compressed id via
   `build_compressed_token_map`: NFKC → NFD → strip accents → lowercase →
   collapse whitespace → strip; partial-UTF8 tokens are keyed raw; the map is
   asserted to be exactly `engram_compressed_vocab_size` = 99092 entries.
   **CONFIRMED.** This is derived from the tokenizer and is runtime state, not
   weights.
2. **Image spans become `DEAD` (−1)**; look-back stops at the sequence start or
   at any DEAD token; blocked slots are filled with `pad_id` 2. **CONFIRMED.**
3. For shifts 0–3, `token = cache[pos − shift]`. The multipliers are per
   (layer, shift), **odd**, drawn from a numpy RNG seeded `10007 · layer_id`,
   bounded by `int64max / (compressed_vocab · 2)`. **CONFIRMED.**
4. `rolling = t0·m0; for i in 1..3: rolling ^= ti·mi; hash_i = rolling % prime_i`
   → 3 n-gram sizes × 8 heads = **24 hash ids**, then offset by the bucket
   cumsum. Output `[B, L, 2, 24]`. **CONFIRMED.**

### 5.4 Injection into the residual stream

Happens **before** the block body: `h = layer.engram(h, hashes[:,:,idx,:],
engram_mask)`, with `h` shaped `[B, L, hc_mult, dim]` and the mask false on
image tokens. Inside:

```
kv     = wkv(embed(hash_ids).flatten(-2))     # key [B,L,4,5120], value [B,L,5120]
weight = q_weight · k_weight
rstd   = rsqrt(mean(h²) + eps) · rsqrt(mean(key²) + eps)
dot    = Σ(h · weight · key) · rstd · dim^−0.5
gate   = sigmoid(copysign(sqrt(max(|dot|, 1e−6)), dot))    # signed sqrt, then sigmoid
out    = h + gate[..., None] · value[..., None, :]
```

One shared 5120-d value is added to all four hc copies, gated per (token, hc
copy). The signed square root before the sigmoid is annotated in the source as
"matching the training kernel". Embedding rows are FP8 and dequantized per
32-block on lookup. **CONFIRMED.**

---

## 6. Hyper-connections and the DSpark draft head

### 6.1 Single-pass mHC (Sinkhorn-normalized)

`hc_mult` 4 residual-stream copies, `hc_sinkhorn_iters` 20, `hc_eps` 1e-6.
The card calls it "Single-Pass mHC (revised residual-stream mixing with an
efficient Mega-mHC kernel)". **CONFIRMED.**

Per block, fp32 parameters: `hc_attn_fn` / `hc_ffn_fn` `[24, 20480]`,
`hc_attn_base` / `hc_ffn_base` `[24]`, `hc_attn_scale` / `hc_ffn_scale` `[3]`,
where `mix_hc = (2 + hc_mult)·hc_mult = 24`. **CONFIRMED.**

```
mixes = Linear(flatten(x), hc_fn) · rsqrt(mean(x²) + norm_eps)     # [b,s,24]
pre[j]  = sigmoid(mixes[j]   · scale[0] + base[j])   + eps         # j = 0..3
post[j] = 2 · sigmoid(mixes[j+4] · scale[1] + base[j+4])
comb[j,k] = mixes[8+4j+k] · scale[2] + base[8+4j+k]
comb = softmax(comb, −1) + eps;  comb /= (comb.sum(0) + eps)
# then 19 more iterations alternating row- and column-normalisation
#   -> doubly stochastic
```

`hc_pre` collapses `y = Σ_hc pre[hc]·x[hc]`; `hc_post` expands
`y[hc_out] = post[hc_out]·x + Σ_hc_in comb[hc_out, hc_in]·residual[hc_in]`.

**Scheduling quirk (CONFIRMED, easy to get wrong):** a sub-block's coefficients
are consumed by the **next** one. Attention collapses with the passed-in
`pre_mix` from the previous layer's FFN, while attention's own mixes feed the
FFN; the FFN returns `ffn_pre` for the next block. The stream starts as
identity one-hot on copy 0 (`make_identity_pre_mix`) and is collapsed with the
final `pre_mix` before the head.

### 6.2 DSpark / MTP

`num_nextn_predict_layers` = 3, `dspark_block_size` = 5, `dspark_target_layer_ids`
= [37, 38, 39], `dspark_markov_rank` = 256, `dspark_n_routed_experts` = 128 with
`dspark_num_experts_per_tok` = 3, `dspark_noise_token_id` = 128799.
**CONFIRMED.**

- `DSparkBlock` lives in the `mtp.*` checkpoint namespace with
  `layer_id = n_layers + stage`. Stage 0 adds `main_proj`
  Linear(5120·3 = 15360 → 5120) + `main_norm`; the last stage adds `norm`,
  `markov_head` and `confidence_head`. MTP **ties** its `embed`/`head` to the
  backbone's. **CONFIRMED.**
- `DSparkAttention` asserts `compress_ratio == 0` → window-only. Prefill seeds
  `window_kv_cache` from `main_x` and returns `x`. Decode uses
  `get_dspark_topk_idxs = cat(arange(min(win, start_pos+1)), win +
  arange(block_size))`, i.e. each draft query attends the main model's window KV
  **plus all 5 block positions**. **CONFIRMED.**
- Draft flow (`forward_spec`): `draft_input_ids` is `[B, 5]` filled with
  `noise_token_id`, position 0 being the accepted token; embeds are replicated
  to `hc_mult`; `main_x = main_norm(main_proj(main_hidden))` where `main_hidden`
  concatenates the hc-mean of the **attention inputs** of layers 37/38/39 — the
  MTP head reads the attention *input* of its target layers, not their output.
  `forward_head` then runs the 3 MTP blocks and iterates the Markov head:
  `logits_bias, markov_embed = markov_head(output_ids[:,i]);
  logits[:,i] += logits_bias; output_ids[:,i+1] = sample(logits[:,i], T)`.
  `DSparkConfidenceHead` = Linear(dim + 256 → 1, fp32) over
  `cat([hidden, markov_embeds])`. Net effect: **5 semi-autoregressive drafts in
  one pass, with a Markov bigram correction and a confidence score.**
  **CONFIRMED.**
- **The repo implements the forward path only.** `generate.py` is plain
  autoregressive (`for cur_pos in ...: model.forward(...)`); `forward_spec` is
  called only by the `model.py` self-test; `inference/README.md` states
  "Generation itself is plain autoregressive sampling." **The
  confidence-scheduled verification/accept loop is NOT FOUND** — there is no
  reference accept/reject algorithm to port. **CONFIRMED absence.**

---

## 7. Quantization as published

The checkpoint is **already quantized**. `config.json`'s `quantization_config`:

```json
{ "quant_method": "fp8", "activation_scheme": "dynamic",
  "weight_block_size": [32, 32], "scale_fmt": "ue8m0", "expert_dtype": "fp4" }
```

`inference/config.json` carries `"dtype": "fp8"`, `"expert_dtype": "fp4"`, and
`model.py` hardcodes `default_dtype = float8_e4m3fn`, `fp8_block_size = 32`,
`fp4_block_size = 32`, `scale_fmt = "ue8m0"`, `scale_dtype =
float8_e8m0fnu`. **All CONFIRMED.**

### 7.1 Which tensors are in which format

| Format | Tensors |
| --- | --- |
| **FP8 E4M3** (+ E8M0 scale) | `wq_a`, `wq_b`, `wkv`, `wo_b`, shared-expert `w1/w2/w3`, DSpark `main_proj`, Engram `embed` and `wkv` |
| **FP4 E2M1** (packed 2/byte, + E8M0 scale) | **all routed experts** `w1/w2/w3`, in the backbone **and** in the MTP layers |
| **BF16** | `wo_a` (the reference dequantizes it to bf16), vision tower (plain `nn.Linear`), `indexer.wk`, `indexer.weights_proj`, `ffn.gate.weight`, all norms |
| **F32** | `attn_sink`, all `hc_*`, `DSparkConfidenceHead.proj`, gate biases |
| **bf16/fp32 depending on ratio** | `Compressor.wkv` — bf16 at ratio 1, fp32 (with fp32 `wgate`) at ratio > 1 |

**CONFIRMED.**

### 7.2 FP8 layout

Weight `[out, in]` with `scale` `[ceil(out/32), ceil(in/32)]` in E8M0.
Activations are quantized dynamically **per row per 32 along K**, E8M0
power-of-two (`round_scale=True` → `2^ceil(log2(amax/448))`), clamped to ±448,
minimum amax 1e-4. `fp8_gemm_kernel` asserts `block_size ∈ {32, 128}` and
applies activation and weight scales to a separate fp32 accumulator.
**CONFIRMED.**

### 7.3 FP4 layout (routed experts)

- Weight stored `[out, in//2]` as `float4_e2m1fn_x2` — **two values per byte,
  packed along K**. `convert.py` asserts the scale shape is exactly
  `(out_dim, in_dim // 32)`, E8M0. **CONFIRMED.**
- E2M1 range is **±6.0** (`fp4_max = 6.0`); nonzero floor `6·2⁻⁹`. **CONFIRMED.**
- `fp4_gemm` is an FP8-activation × FP4-weight GEMM: the activation is FP8 with
  a 1×32 scale; the kernel casts FP4 → FP8 **through FP32**, runs FP8×FP8, then
  multiplies the accumulator by the per-row activation scale and the
  per-row/per-32-K-block weight scale. **CONFIRMED.**
- `convert.py` also supports `--expert-dtype fp8`, losslessly casting E2M1 →
  E4M3 with a shared per-(32,32)-block exponent offset (`MAX_OFFSET_BITS = 6`,
  since `6·2⁶ = 384 < 448`). **CONFIRMED.**

### 7.4 Tensors whose stored shapes look wrong at first glance

`experts.*.w2` is `[5120, 1152]` with scale `[5120, 72]`. 1152 = 2304/2, i.e.
packed FP4 along K; 72 = 2304/32, the per-32 scale count. `w1`/`w3` are
`[2304, 2560]` with scale `[2304, 160]` where 2560 = 5120/2 and 160 = 5120/32.
**CONFIRMED.** The practical hazard: **2304 is not a multiple of 64**, the group
size every affine path in TinyTitan assumes.

### 7.5 KV-cache precision

- Compressed main KV: **FP4 E2M1 with one E4M3 scale per 16 channels**.
- Indexer Q/K: **FP4 with E8M0 per 32**.
- Window KV: **FP8 E4M3 → dequant**.
- All four paths use `inplace=True`, i.e. fused quant+dequant, so the buffer
  keeps the original dtype at quantized precision. **CONFIRMED.**
- Card figures: **≈ 890 bytes per token** globally, ≈ 1/4 of DeepSeek-V4-Flash;
  ≈ 1/8 persistent with SWA Bounded Replay; Figure 1(b) states ≈ 4× and ≈ 437×
  reductions vs V4-Flash and V1 (asset `assets/dsv41_kv_cache.png`).
  **CONFIRMED (card).**
- The official release note independently states **1/4 the HBM and 1/8 the SSD
  storage** vs the previous generation. **CONFIRMED.**

---

## 8. Vision — DeepSeek-ViT

**Present in the repo** (unlike checkpoints that declare a `vision_config` and
ship language weights only): `vision.*` and `aligner.*` tensors exist, ≈ 0.26 GB
BF16. **CONFIRMED.**

### 8.1 Tower

`vision_n_layers` 32, `vision_dim` 1024, `vision_n_heads` 16 (head_dim 64),
`vision_inter_dim` 2816, patch 14, `downsample_ratio` 3, `max_image_tokens`
1024, `min_pixels` 295936 (= 544²), `max_wh_ratio` null, `rope_theta` 10000.
Card: trained from scratch, 2D-RoPE, 3×3 pixel-unshuffle. **CONFIRMED.**

- **Patch encoding:** `PatchEmbed.proj = nn.Linear(3·14·14 = 588 → 1024)` on
  `x.flatten(1)`; pixels normalised `((x/255) − 0.5) / 0.5` in bf16, split into
  `[n_vit_h·n_vit_w, 3, 14, 14]`. **CONFIRMED.**
- **2D RoPE:** `rope_dim = 1024/16/2 = 32` per axis; cos/sin built from
  `(hpos, wpos) · inv_freq`, applied split-half
  (`x1·cos − x2·sin`, `x2·cos + x1·sin`). **CONFIRMED.**
- **Blocks:** RMSNorm → **full bidirectional** attention
  (`F.scaled_dot_product_attention`, **no mask**; q/k/v from one `wqkv`
  Linear, `wo` out) → RMSNorm → SwiGLU MLP (`w1` 1024 → 5632, chunked into
  gate/up, `w2` 2816 → 1024). Final RMSNorm. **CONFIRMED.**
- **Aligner (projector):** 3×3 pixel-unshuffle (`view(n_h,n_w,-1)` → pad →
  `F.unfold(kernel=3, stride=3)`), in_dim = 1024·9 = **9216**, then
  `w1` Linear(9216 → **5120**) → GELU → `w2` Linear(5120 → 5120). It **does**
  map to the text hidden size. **CONFIRMED.**

### 8.2 Visual-token injection

- Tokens per image: `n_llm_h · (n_llm_w + 1) + 2`, laid out
  `[IMAGE_START] + ([IMAGE] × n_llm_w + [IMAGE_NEW_LINE]) × n_llm_h +
  [IMAGE_END]`. **CONFIRMED.**
- **Every** position of that span carries `image_token_id` **129264** in
  `input_ids`; positions are distinguished **only** by `token_type`
  (TEXT = −1; IMAGE_START / IMAGE / IMAGE_NEW_LINE / IMAGE_END = 0…3).
  **CONFIRMED.**
- START/END/NEW_LINE take learned `image_start` / `image_end` /
  `image_newline` `[5120]` embeddings; IMAGE slots take aligner rows in
  **row-major reading order**. **CONFIRMED.**
- `prepare_vl_inputs` requires the placeholder count to equal the image count
  and asserts the tokenizer's placeholder id (if known) is 129264.
  **CONFIRMED.**
- **Resize planning:** upscale if `0 < w·h < min_pixels` (295936), pad to a
  patch multiple, then `solve_resize_ratio`/`safe_resize` shrink until the token
  count is ≤ 1024. **CONFIRMED.**
- Input handling: single PIL images only, from bytes/base64/data-URL/URL/path,
  RGB, `ImageOps.pad` with grey (127,127,127). **Video: NOT FOUND** — no
  temporal handling in the processor, no mention in `vision.py` or the card, and
  the pipeline tag is image-text-to-text. **CONFIRMED absence.**

---

## 9. Tokenizer, chat encoding, sampling

### 9.1 Tokenizer

`tokenizer_class` = `PreTrainedTokenizerFast`, vocab 129280.
**CONFIRMED.**

| Token | Id |
| --- | ---: |
| `<｜begin▁of▁sentence｜>` (bos) | 0 |
| `<｜end▁of▁sentence｜>` (eos, **also pad**) | 1 |
| `pad_token_id` | 2 |
| `image_token_id` | 129264 |
| `dspark_noise_token_id` | 128799 |
| `engram_pad_token_id` | 2 |

`unk_token` is `null`; `add_bos_token` and `add_eos_token` are both **false**;
`clean_up_tokenization_spaces` false; `legacy` true; `model_max_length`
1048576. **CONFIRMED.**

Special tokens: `<｜User｜>`, `<｜Assistant｜>`, `<｜System｜>`,
`<｜latest_reminder｜>`, `<think>`/`</think>`, `｜DSML｜`,
`<｜deepseek_image｜>`. Quick-instruction tokens: `<｜action｜>`, `<｜title｜>`,
`<｜query｜>`, `<｜authority｜>`, `<｜domain｜>`, `<｜read_url｜>`. Roles:
`system`, `user`, `assistant`, `tool`, `latest_reminder`. **CONFIRMED.**

### 9.2 Prompt encoding — there is no template

**CONFIRMED:** the card states the release "does not include a Jinja-format chat
template", and `tokenizer_config.json` has no `chat_template` key. Encoding lives
in `encoding/encoding.py` (a self-contained reference with tests,
`encoding/README.md`) plus the external `deepseek-recipe` library.
`generate.py` imports `encode_case`, `encode_messages`,
`parse_message_from_completion_text`, `parse_tagged_text`, `to_json`.

- **Thinking mode** emits `<think>…</think>` before the answer and injects
  `<｜System｜>Reasoning Effort: {budget} (range 1-100, the higher the value, the
  more thorough the reasoning)` once at index 0. **Chat mode** places `</think>`
  immediately after `<｜Assistant｜>`. **CONFIRMED.**
- `reasoning_effort` is an integer 1–100 with aliases low→50, high→75
  (**default**), max→100. `drop_thinking` (default true) strips reasoning from
  earlier assistant turns, and is **auto-disabled when tools are present**.
  **CONFIRMED.**
- **Tool calls changed in V4.1.** Wrapped in `<｜DSML｜ calls>` — note the
  **leading space** — with `<｜DSML｜ invoke name="...">` and
  `<｜DSML｜ parameter name="..." string="true|false">…</｜DSML｜ parameter>`.
  V4 used `<｜DSML｜tool_calls>` with no leading space. `string="true"` means a
  raw string, `string="false"` a JSON value. Results are wrapped
  `<｜User｜><tool_result>…</tool_result>`. Namespaces are addressed as
  `search::lookup`. `merge_tool_messages` folds `tool` messages into the
  preceding user message, sorted by the assistant's tool-call order.
  **CONFIRMED.**

### 9.3 Sampling

- Card recommendations: **temperature 1.0, top_p 0.95 or 1.0**, context 1 M,
  max_tokens ≥ 256K; instruct evaluations used temperature 1.0 / top_p 0.95.
  **CONFIRMED.**
- `generate.py` defaults: temperature 1.0, thinking mode `chat`,
  `max_new_tokens` 200 (CLI) / 100 (main signature). The `inference/README`
  interactive example passes `--temperature 0.6`. **CONFIRMED.**
- Sampler is Gumbel-max
  (`softmax(logits/T).div_(exponential(1)).argmax()`), greedy at temperature 0.
  **CONFIRMED.**
- **No `generation_config.json` exists**, so there is no checkpoint-authored
  sampling default. **CONFIRMED absence.**

---

## 10. Reference implementation and serving

- `inference/` is described by its own README as "a readable reference
  implementation rather than a production serving engine". **CONFIRMED.**
- `convert.py --model-parallel "${MP}"` with `MP = 8` in `run.sh` (a comment
  shows `MP = 4` as an override); launched with
  `torchrun --nproc-per-node "${MP}"`; multi-node via standard
  `--nnodes/--node-rank/--master-addr/--master-port`. **CONFIRMED.**
- **CUDA-only** runtime (`torch.cuda.set_device`, nccl,
  `expandable_segments:True`). **CONFIRMED.**
- `requirements.txt`: `torch>=2.10.0`, `transformers`, `tokenizers`,
  `safetensors>=0.7.0`, `numpy`, `sympy`, `Pillow`, **`tilelang==0.1.8`**,
  `tqdm`. The kernels route through tilelang. **CONFIRMED.**
- The `model.py` self-test exercises "the real dense-fp8 / MoE-fp4 kernels".
  **CONFIRMED.**
- **No hardware requirement is stated anywhere**: no minimum GPU memory, no
  recommended GPU count, no vLLM/SGLang version. The official release note says
  only "Planning a large-scale deployment with 2,000 GPUs + a storage cluster?
  Let's talk." — an invitation, not a requirement. **NOT FOUND.**
- Directories shipped besides the weights: `encoding/` (reference encoder +
  tests), `evaluation/` (including `dsh-minimal.patch`), `inference/`, `assets/`.
  **CONFIRMED.**

---

## 11. Community quantizations

**CONFIRMED to exist** via the HF model-search API. Listing them is not an
endorsement, and none is an official MLX release.

- **4-bit / INT4:** `INCModel3/DeepSeek-V4.1-Flash-W4A16-Engram-AutoRound`
  (experts INT4 GPTQ, Engram INT4), `INCModel3/...-MXFP4-Engram-AutoRound`,
  `bot-lab-21/...-EXL3-3.5bpw-Pollard`,
  `diffbot/...-EXL3-2.0bpw-2x-RTX-PRO-6000`, `Mia-AiLab/...-EXL3-2.9bpw`,
  `dealignai/...-UNCENSORED-EXL3-2.9bpw`, `sfxnz/...-EXL3`,
  `jeet0733/...-EXL3-C4h36`.
- **4-bit MLX:** `pipenetwork/DeepSeek-V4.1-Flash-MLX-mixed-4_8bit`,
  `Jundot/DeepSeek-V4.1-Flash-oQ4e-mtp`.
- **FP4 / NVFP4:** `nvidia/DeepSeek-V4.1-Flash-NVFP4` (NVIDIA ModelOpt), plus
  mirrors and derivatives (`s-zaizen`, `LibertAIDAI`, `msuiche`,
  `TechnoBaptist`, `aidendle94/...-Engram`, `MomonV/...-UNCENSORED-NVFP4`).
- **GGUF:** `antirez/deepseek-v4.1-flash-gguf`, `vcruz305/...-GGUF`,
  `smalinin/...-GGUF` (vision), `pfeifferj/...-GSQ-RCO-GGUF`,
  `taurusduan/...-GSQ-RCO-GGUF`, `Lucebox/...-ROCMFP23-GGUF`,
  `mxxm-t/...-GGUF`, `apetersson/...-MixedQ2-GGUF`,
  `kernelpool/...-MXFP4-GGUF`, `AMAImedia/...-FP8-GGUF`,
  `audreyt/...-Abliterated-GGUF`, `DevQuasar/deepseek-ai....-GGUF`.
- **MLX 2–3 bit:** `orcarouter/...-Uncensored-MLX`,
  `Vontra/...-MLX-2bit-MTP`, `OpensourceWTF/...-MTPLX-streaming-q2`,
  `nanguoyu/...-minirun`.
- **8-bit:** many identical FP8 mirrors (`Terom/`, `Fileportz/`,
  `kwakuobeng/`, `Vaibhavhome30/`, …) plus uncensored FP8 derivatives
  (`dealignai/...-UNCENSORED-FP8`).
- **AWQ: NOT FOUND** in what was read (the search output was truncated, so this
  is "not found", not proof of absence).

**Community discussion threads** (titles via the discussion index API; bodies
were JS-rendered and unreadable, so the claims inside are **unverified**):
#38 "Reading DeepSeek V4.1 Flash's config.json: Where Sparse Attention Lives";
#37 "Runs on one RTX 5090 (31.8 GiB) + 125.7 GiB RAM via a llama.cpp fork: GGUF,
report and numbers"; #43 deployment on 4× A100 80GB; #28 "Running on 4x RTX PRO
6000 with NVMe offload for ngram"; #56 a VRAM-fit Space
(`yash-711/deepseek-v41-flash-fit`).

---

## 12. Confirmed quantities worth having in one place

| Quantity | Value |
| --- | ---: |
| Total stored | 510.3 GB |
| Safetensors parameters | 763,205,315,794 |
| Card: backbone + Engram | 552 B + 196 B |
| Card: active per token | 8 B prefill / 16 B decode |
| Routed experts | 384 per layer, top-6 |
| Routed-expert parameters | ≈ 288.8 B (271.8 B weights) |
| Engram parameters | ≈ 196.6 B |
| Engram stored | ≈ 202.8 GB |
| Engram rows | 384,006,168 + 384,016,682 |
| Engram row size | 264 bytes (256 E4M3 + 8 E8M0) |
| Engram rows per token | 48 (2 layers × 8 heads × 3 n-gram sizes) |
| Engram bytes per token | ≈ 12.7 KB |
| KV cache | ≈ 890 bytes/token |
| Context | 1,048,576 tokens |
| Vision tower + aligner elements | 485,253,120 (0.485 B) |
| Weighted average precision | ≈ 5.9 bits/parameter (510.3 GB ÷ 763.2 B) |

The last row is **INFERRED** arithmetic, not a stated figure: it is what the
published file averages across 204.0 B FP8 values, 557.2 B FP4-packed values,
and the bf16/f32 tensors and scales. It is a size-to-parameter ratio, and it is
the reason an affine 8-bit re-encoding of this checkpoint comes out *larger*
than the checkpoint (see the port document's §2).

---

## 13. Open questions and unknowns

Recorded so they are not silently filled in later.

1. **The parameter count does not reconcile.** The card says 552 B backbone +
   196 B Engram = **748 B**; the safetensors metadata says **763.2 B**; the gap
   is 15.2 B. Reduced as far as published data allows in §1: the Engram figure
   is exactly reconstructible (values only, scales excluded), the vision tower
   is only 0.485 B and is not the gap, and the metadata's own dtype buckets
   disagree with the tensor inventory. Only the tech report could close it, and
   it is unreadable.
2. **The tech report is unreadable.** `DeepSeek_V41_Tech_Report.pdf` is an
   LFS pointer. Anything only in it is unknown.
3. **`bias_vl`.** The router carries a second bias vector used "when vision is
   enabled"; when exactly it applies, and whether a text-only forward should use
   `bias` or `bias_vl`, is **NOT FOUND**.
4. **SWA Bounded Replay** is card prose with no reference implementation.
5. **The DSpark confidence-scheduled verification loop** is absent; only the
   draft forward exists.
6. **`tie2eid` / `tid2eid`** are checkpoint name components special-cased in
   `convert.py` that never appear in `inference/model.py`; their role is
   **NOT FOUND**.
7. **`dspark_noise_token_id` 128799**'s token string is not identified.
8. **`Full` / `Reindex` / `Reuse`** are prose names; the layer mapping in §3.5 is
   **inferred** from the source-layer lists.
9. **No hardware requirement is stated** by the publisher (§10).
10. **`nope_head_dim`** is computed and never used; whether that is vestigial or
    intended for a path not shipped is unknown.
11. **The `transformers` implementation** for this architecture was not read
    (the model requires `transformers` 5.6.0); everything here comes from the
    repository's own reference code plus the configs.

---

## Sources

- HF API: `https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4.1-Flash?blobs=true`
- Tree: `https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4.1-Flash/tree/main?recursive=true`
- `config.json`, `tokenizer_config.json`, `model.safetensors.index.json` (raw, sha `dba1be0a…`)
- safetensors headers of shards 1, 3, 43, 45, 48 (ranged HTTP requests)
- `inference/`: `model.py`, `vision.py`, `engram.py`, `image_processor.py`,
  `generate.py`, `convert.py`, `kernel.py`, `config.json`, `README.md`,
  `requirements.txt`, `run.sh`
- `encoding/encoding.py`, `encoding/README.md`
- card `README.md`; `LICENSE`
- release note: `https://api-docs.deepseek.com/news/news260910/`
- discussion index: `https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4.1-Flash/discussions?p=0`
