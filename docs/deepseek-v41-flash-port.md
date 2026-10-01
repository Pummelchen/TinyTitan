# DeepSeek-V4.1-Flash port — integration concept

**Status: concept. Nothing here is implemented, and nothing is supported.**

**Owner decision, 2026-09-25: the weights stay in their original form. If
something must be converted or modified, it may not degrade the model in any
area.** The consequence is that there is no 4-bit build and no 8-bit build of
this model — re-quantizing is a degradation and is ruled out. There is one
install, at the checkpoint's own mixed precision. §6 is the policy; §11 is what
still needs deciding (storage and throughput, not precision).

Requested: integrate `deepseek-ai/DeepSeek-V4.1-Flash` into TinyTitan at 4-bit and
8-bit, each derived from the original 16-bit. The request's premise does not
survive contact with the checkpoint; §2 says why.

This document is the source-verified design record that a later implementation
session works against, in the shape of
[`qwen38-flash-next-port.md`](qwen38-flash-next-port.md). Every geometry value,
tensor name, dtype and byte figure below was read from the published checkpoint
(its `config.json`, its `model.safetensors.index.json`, the safetensors headers
of the shards that carry the interesting tensors) or from the official
`inference/` reference code in the same repository. Anything inferred rather
than read is labelled **inferred**. Anything unverified is listed as an open
unknown in §12 rather than guessed at.

The full fact base behind this plan — every tensor family with its shapes and
encodings, the reference forward's algorithm, the chat and tool-call encoding,
the vision tower, the KV-cache design, the community quantizations and the
unresolved questions — is kept separately in
[`deepseek-v41-flash-reference.md`](deepseek-v41-flash-reference.md). Read that
for facts; this document is the plan that follows from them.

The conclusion, stated up front so it is not buried: **this is not a wiring
job.** It is not a `MODELS` row in `prepare_agentworld.py`, and it is not a
sibling of the Qwen3.8-Flash-Next port. Four of its subsystems — Engram,
Compressed Sparse Attention 2, the FP4-expert quantization format, and a
missing chat template — have no counterpart anywhere in this tree, and one of
them (Engram) alone is larger than every model this project has ever shipped.

---

## 1. The checkpoint, verified

| | Value | Read from |
| --- | --- | --- |
| Repo | `deepseek-ai/DeepSeek-V4.1-Flash` | HF API |
| Pinned sha | `dba1be0a40aa45a94ad051997016db3960a90277` | HF API (`lastModified` 2026-09-10) |
| `gated` | `false` | HF API |
| License | MIT | `LICENSE`, card `license: mit` |
| `architectures` / `model_type` | `DeepseekV41ForCausalLM` / `deepseek_v41` | `config.json` |
| Nested configs | `text_config.model_type` `deepseek_v41_text`, `vision_config.model_type` `deepseek_v41_vision` | `config.json` |
| Shards / tensors | 48 safetensors / 96,085 index entries | index + tree API |
| Size on disk | **510.3 GB** (`total_size` 510,286,023,000; tree total 510,311,613,821) | index + tree API |
| Declared parameters | **763,205,315,794** | safetensors metadata (HF API) |
| Card's parameter claim | 552 B backbone + 196 B Engram = 748 B | card README |
| Context | 1,048,576 (`max_position_embeddings`) | `config.json` |
| `tie_word_embeddings` | `false` — the head is a real tensor (`head.weight`, `[129280, 5120]`, BF16) | `config.json` + shard 43 header |
| Chat template | **absent.** "This release does not include a Jinja-format chat template." | card README |
| `generation_config.json` | **absent** — the repo has no such file | tree API |

The card's 748 B and the safetensors metadata's 763.2 B do not reconcile, and
neither source explains the gap. Both figures are recorded; the tech report that
would settle it is an unreadable Git-LFS pointer (§12).

### Text geometry

| Field | Value |
| --- | --- |
| `num_hidden_layers` | 40 (card: 20-layer causal encoder + 20-layer decoder) |
| `hidden_size` | 5120 |
| `vocab_size` | 129280 |
| `num_attention_heads` / `num_key_value_heads` | 64 / **1** |
| `head_dim` / `qk_rope_head_dim` | 512 / 64 |
| `q_lora_rank` / `o_lora_rank` / `o_groups` | 1280 / 1024 / 8 |
| `n_routed_experts` / `n_shared_experts` / `num_experts_per_tok` | 384 / 1 / 6 |
| `moe_intermediate_size` | 2304 |
| `scoring_func` / `topk_method` | `sqrtsoftplus` / `noaux_tc` |
| `norm_topk_prob` / `routed_scaling_factor` | `true` / 1.5 |
| `hidden_act` / `swiglu_limit` | silu / **10.0** |
| `rms_norm_eps` | 1e-20 |
| `rope_scaling` | yarn, factor 16, `original_max_position_embeddings` 65536, theta 10000 |
| `sliding_window` | 128 |
| `compress_ratios` | 43 entries: `[0, 0, 18×2, 20×1, 0, 0, 0]` |
| `kv_source_layer_ids` / `index_source_layer_ids` | [2,8,14,20] / [2,8,14,20,24,28,32,36] |
| `index_n_heads` / `index_head_dim` / `index_topk` | 32 / 128 / 512 |
| `candidate_source_layer_id` / `candidate_topk_blocks` / `candidate_block_size` | 20 / 2048 / 8 |
| `hc_mult` / `hc_sinkhorn_iters` / `hc_eps` | 4 / 20 / 1e-6 |
| `engram_layer_ids` / `engram_num_embeddings` | [1, 14] / [384006168, 384016682] |
| `engram_vocab_size` / `engram_max_ngram_size` / `engram_n_heads` / `engram_head_dim` | 16,000,000 / 4 / 8 / 256 |
| `num_nextn_predict_layers` / `dspark_*` | 3 draft layers at 128 experts top-3, `dspark_block_size` 5 |

### The tensor inventory, by category

Counted from `model.safetensors.index.json`; the checkpoint's own top-level
namespaces are `layers.*`, `mtp.*`, `vision.*`, `aligner.*`, `embed.weight`,
`head.weight`, `norm.weight`, `image_start`, `image_end`, `image_newline`.

| Category | Tensors | Stored bytes (from shard headers) |
| --- | ---: | ---: |
| routed experts (40 layers × 384 experts × w1/w2/w3) | 92,160 | ≈ 288.8 GB |
| Engram (2 layers × 6 tensors) | 12 | **≈ 202.8 GB** |
| attention (wq_a, wq_b, wkv, wo_a, wo_b, sink, norms) | 603 | ≈ 6.45 GB |
| hyper-connections (`hc_*`, 6 per layer) | 240 | ≈ 1.77 GB |
| shared experts | 240 | ≈ 1.77 GB |
| router gate (`weight`, `bias`, `bias_vl` per layer) | 120 | ≈ 0.90 GB |
| MTP/DSpark (3 layers, 128 experts each) | 2,401 | ≈ 6.35 GB |
| vision tower + aligner | ≈ 263 | ≈ 0.26 GB |
| `embed.weight` | 1 | 1.32 GB |
| `head.weight` | 1 | 1.32 GB |
| norms, image specials, `norm.weight` | ≈ 45 | ≈ 0.31 GB |

Per-tensor shapes that matter, read from the safetensors headers:

```
layers.0.attn.wq_a.weight    F8_E4M3  [1280, 5120]     scale F8_E8M0 [40, 160]
layers.0.attn.wq_b.weight    F8_E4M3  [32768, 1280]    scale F8_E8M0 [1024, 40]
layers.0.attn.wkv.weight     F8_E4M3  [512, 5120]      scale F8_E8M0 [16, 160]
layers.0.attn.wo_a.weight    F8_E4M3  [8192, 4096]     scale F8_E8M0 [256, 128]
layers.0.attn.wo_b.weight    F8_E4M3  [5120, 8192]     scale F8_E8M0 [160, 256]
layers.0.attn.attn_sink      F32      [64]
layers.0.ffn.gate.weight     BF16     [384, 5120]
layers.0.ffn.gate.bias       F32      [384]
layers.0.ffn.gate.bias_vl    F32      [384]
layers.0.ffn.experts.0.w1.weight  I8   [2304, 2560]    scale F8_E8M0 [2304, 160]
layers.0.ffn.experts.0.w2.weight  I8   [5120, 1152]    scale F8_E8M0 [5120, 72]
layers.0.ffn.experts.0.w3.weight  I8   [2304, 2560]    scale F8_E8M0 [2304, 160]
layers.0.hc_attn_fn          F32      [24, 20480]
layers.14.engram.embed.weight F8_E4M3 [384016682, 256] scale F8_E8M0 [384016682, 8]
layers.14.engram.wkv.weight  F8_E4M3  [25600, 6144]    scale F8_E8M0 [800, 192]
layers.14.engram.q_weight    BF16     [4, 5120]
layers.14.engram.k_weight    BF16     [4, 5120]
embed.weight                 BF16     [129280, 5120]
head.weight                  BF16     [129280, 5120]
```

Two whole shards — `model-00047` and `model-00048`, 101.5 GB each — contain
**six tensors each, and all six are Engram**. The Engram tables are 40% of this
checkpoint's bytes.

---

## 2. First correction: there is no 16-bit source to convert from

The request says "each converted from original 16-Bit". That premise does not
hold for this checkpoint, and the difference changes the conversion design.

`config.json` declares `"dtype": "bfloat16"`, but that is a declaration, not the
storage format. The checkpoint's own `quantization_config` says:

```json
{ "quant_method": "fp8", "activation_scheme": "dynamic",
  "weight_block_size": [32, 32], "scale_fmt": "ue8m0", "expert_dtype": "fp4" }
```

The safetensors metadata agrees: 204.0 B of `F8_E4M3` values, 557.2 B of `I8`
values, only 2.0 B BF16 and 42.3 M F32. The `I8` bucket is the routed experts —
`557,171,343,360` is exactly `384·3·5120·2304·40 + 128·3·5120·2304·3`, the routed
expert value count — stored as **packed FP4** at two nibbles per byte.

So the masters are:

- **Attention projections, shared experts, DSpark `main_proj`, Engram `embed`/`wkv`:
  FP8 E4M3** with `F8_E8M0` block scales.
- **Routed experts (backbone and MTP): FP4 E2M1**, packed along K at two
  values per byte, scaled per 32 along K by an E8M0 power-of-two exponent.
- **`wo_a` is BF16** in the published checkpoint even though the reference code
  treats it as the BF16 path — `convert.py` dequantizes it back.
- Router (`ffn.gate.weight`) BF16, `attn_sink` and all `hc_*` F32, every norm
  BF16.

Three consequences, all of which belong in the document rather than in a
surprise at the end of a 510 GB download:

1. **"16-bit" is a reconstruction, not a source.** Any `--bits 16` path means
   dequantizing FP8/FP4 back to BF16 — producing a *larger* artifact than the
   checkpoint it came from, at 748–763 B × 2 bytes ≈ 1.5 TB. That is not
   achievable on this host and is not worth building.
2. **The averaging matters for what "8-bit" means.** The card is tagged
   "8-bit precision", and its FP8 half really is 8-bit, but 61% of the
   parameters are FP4. Weighted across the file the published checkpoint
   averages ≈ 5.9 bits per parameter (510.3 GB over 763.2 B). A TinyTitan
   8-bit build at affine group 64 (8.25 effective bits) is therefore **not**
   "the same 8-bit" — it is roughly 40% larger than the checkpoint it converts.
3. **FP4 experts do not fit the affine scheme at all.** TinyTitan's `.ssdai`
   quant slots are affine (`weightBits`/`scheme`/`scaleType`/`biasType`/
   `groupSize`), and its converters emit affine group-64 snapshots. FP8
   E4M3 + E8M0 per-32×32 and packed FP4 E2M1 + E8M0 per-32 are different
   formats. So the port either re-quantizes affinely — changing the stored
   values — or it grows block-FP8 and FP4 E2M1 decode kernels and writes the
   checkpoint's own blocks through. **The owner has ruled out the first option:
   nothing may degrade the model.** §6 is the policy that follows.

---

## 3. What the port actually has to build

In dependency order. Each of these is a subsystem, not a config flag.

### 3.1 No chat template — a hard blocker, and the cheapest to fix

TinyTitan's runtime renders prompts from the model's own `chat_template.jinja`
and errors with "installed tokenizer is missing chat_template.jinja" without it.
This release ships **no Jinja template at all**; prompt encoding lives in
`encoding/encoding.py` (a self-contained Python reference with tests) and the
external `deepseek-recipe` library.

Supplying one is a prerequisite for *any* install, at any width, and it is the
one piece of work here that is genuinely small — but it is also the one that
must be exact, because the template is simultaneously the prompt format, the
tool-call dialect and the reasoning-marker system. `encoding/README.md` gives
the pieces a template has to reproduce:

- roles `system`, `user`, `assistant`, `tool`, `latest_reminder`;
- tokens `<｜begin▁of▁sentence｜>` (0), `<｜end▁of▁sentence｜>` (1, also pad),
  `<｜User｜>`, `<｜Assistant｜>`, `<｜System｜>`, `<｜latest_reminder｜>`,
  `<think>`/`</think>`, `｜DSML｜`, `<｜deepseek_image｜>`;
- **thinking mode**: `<think>…</think>` before the answer, plus a one-time
  `<｜System｜>Reasoning Effort: {budget} (range 1-100…)` injection at index 0;
  chat mode emits `</think>` immediately after `<｜Assistant｜>`;
- **tool calls, changed in V4.1**: `<｜DSML｜ calls>` (leading space),
  `<｜DSML｜ invoke name="…">`, `<｜DSML｜ parameter name="…" string="true|false">`
  — V4 used `<｜DSML｜tool_calls>` with no leading space, so a template copied
  from a V4 sibling would be wrong in a way that still parses sometimes;
- tool results as `<｜User｜><tool_result>…</tool_result>`;
- `drop_thinking` (default true) strips earlier-turn reasoning but is
  auto-disabled when tools are present.

`reasoning_effort` is an integer 1–100 with aliases low→50, high→75 (default),
max→100. TinyTitan's existing thinking levels must be mapped onto that, and the
mapping recorded as a deviation if it cannot be exact.

There is also **no `generation_config.json`**, so the sampling defaults come
from the card: temperature 1.0, top_p 0.95 or 1.0, max_tokens ≥ 256K,
context 1M. TinyTitan supports presence penalty `0.0` only; nothing here asks
for otherwise, which is one of the few things that goes smoothly.

### 3.2 Compressed Sparse Attention 2 (CSA2)

The card names it CSA2; the mechanism in `inference/model.py` is index-based
selection over two KV sources concatenated into one `sparse_attn` call.

**Sliding window (every layer, per-layer cache).** A ring buffer of 128 raw KV
slots per layer, `window_kv_cache [B, 128, 512]`. Prefill builds one causal
window per query; decode gives the single query the whole ring oldest-first.

**Compressed KV (only where `compress_ratios[layer] > 0`).** A `Compressor`
pools `compress_ratio` consecutive tokens into **one** 512-d latent through a
learned softmax gate: `kv = wkv(x)`, `score = wgate(x)` (both fp32), pooled as
`Σ kv·softmax(score over the group)`, then RMSNorm. `ratio == 1` degenerates to
`norm(wkv(x))` with no gate and no fp32. Incomplete groups persist in
`kv_state`/`score_state` across decode steps, and the compressor returns
**pre-RoPE** latents because the indexer needs them unrotated. Cache shape
`[B, max_seq_len // ratio, 512]`.

The layer schedule is `[0, 0, 18×2 (layers 2–19), 20×1 (layers 20–39), 0, 0, 0]`:
layers 0–1 and the three MTP layers are window-only.

**Hierarchical lightning indexer**, on `index_source_layer_ids`
[2,8,14,20,24,28,32,36] only: 32 heads of dim 128, `index_topk` 512,
scale 128^−0.5. `wq_b` derives its query from the same `qr` as attention Q;
`weights_proj` is 5120→32; `wk` (owners only) is 512→128 plus an RMSNorm.
Score is `einsum("bshd,btd->bsht")` → **relu** → weighted by head → summed.
Unreachable compressed positions are masked to −inf, `topk(min(512, end_pos//ratio))`
is taken, indices are re-sorted into position order, invalid entries become −1,
valid ones shift by the window length, returned as int32.

**Two-level candidate pre-filter:** `candidate_source_layer_id` 20 scores each
8-token block by its max position logit, pins the partially-filled newest block,
keeps the top 2048 blocks; layers 21–39 mask their own index scores to that
block set before top-k. A negative source id disables the level.

**Cross-layer sharing** is the part with no analogue in this tree. A
`SharedAttentionRuntime` singleton carries `compress_kv`, `index_k`, `topk_idxs`
and `candidates` down the stack: only the four `kv_source_layer_ids` own a
compressor cache (2→layers 3–7, 8→9–13, 14→15–19, 20→21–39); only the
kv-source subset of index sources owns an indexer K cache; only index sources
run an indexer and publish `topk_idxs`; **every other layer reads the last
published object.** Window KV stays per-layer. MTP layers are excluded
(`is_backbone = layer_id < n_layers`). The card's "Full / Reindex / Reuse"
modes are, **inferred**, exactly this partition: Full = {2,8,14,20},
Reindex = {24,28,32,36}, Reuse = the remaining layers.

Consequence for the runtime: the KV cache is no longer per-layer. It becomes a
small set of shared, ratio-scaled caches plus 40 per-layer windows, and the
decode loop acquires an inter-layer dependency — a layer cannot attend until an
earlier source layer has published. That reaches into the runner's command
graph, not just a kernel.

**Cache precision.** The card states ≈ **890 bytes per token** globally (≈ 1/4
of V4-Flash; ≈ 1/8 persistent with SWA Bounded Replay; ≈ 437× better than V1).
The mechanism is FP4 E2M1 main KV with one E4M3 scale per 16 channels, FP4
indexer Q/K with E8M0 per 32, and FP8 window KV, all as fused quant+dequant
(`inplace=True`). TinyTitan already has 4-bit/8-bit KV quantization, but this is
a different layout and a different granularity.

Note "SWA Bounded Replay" appears only in the card prose — **no code for it in
`inference/`** — so it is not something that can be lifted, only re-derived.

### 3.3 Engram — the decisive subsystem

Two tables at **layers 1 and 14**, `engram.embed.weight` FP8 E4M3 with shapes
`[384006168, 256]` and `[384016682, 256]` — **384 M rows each (768 M rows, 98.3 B
values per table), 196.6 B parameters, 202.8 GB as stored**, versus a card claim
of 196 B parameters. The two tables ship as shards 47 and 48 on their own,
101.5 GB each.

`engram_vocab_size` = 16,000,000 is *not* the row count; it is the starting
point of a prime-bucket search. **Inferred** from `inference/engram.py`: per
layer the table is partitioned into `(max_ngram_size − 1) × engram_n_heads`
= 3 × 8 = 24 prime-sized buckets, 48 primes across the two layers, drawn in
order and never reused, each ≈ 16.0 M rows (`384006168 / 24 ≈ 16,000,257`).

The lookup, per `inference/engram.py`:

1. Every token id maps to a **compressed** id via `build_compressed_token_map`
   (NFKC → NFD → strip accents → lowercase → collapse whitespace → strip;
   partial-UTF8 tokens keyed raw), asserted to be exactly
   `engram_compressed_vocab_size` 99092 entries. This is derived from the
   tokenizer and is **runtime state**, not weights.
2. Image spans become `DEAD` (−1); look-back stops at the sequence start or any
   DEAD token; blocked slots take `pad_id` 2.
3. For shifts 0–3, `token = cache[pos − shift]`; per (layer, shift) **odd**
   multipliers come from a fixed numpy RNG seeded `10007·layer_id`, bounded by
   `int64max / (compressed_vocab·2)`.
4. `rolling = t0·m0; for i in 1..3: rolling ^= ti·mi; hash_i = rolling % prime_i`
   → 3 n-gram sizes × 8 heads = 24 hash ids, offset by the bucket cumsum.
   Output `[B, L, 2, 24]`.
5. Injection happens **before** the block body:
   `kv = wkv(embed(hash_ids).flatten(-2))` → key `[B,L,4,5120]` + value
   `[B,L,5120]`; `weight = q_weight · k_weight`;
   `rstd = rsqrt(mean(h²)+eps)·rsqrt(mean(key²)+eps)`;
   `dot = Σ(h·weight·key)·rstd·dim^−0.5`;
   `gate = sigmoid(copysign(sqrt(max(|dot|,1e−6)), dot))`;
   output `h + gate[...,None]·value[...,None,:]` — one shared 5120-d value added
   to all four hc copies, gated per (token, hc copy). Embedding rows are FP8,
   dequantized per 32-block on lookup.

So Engram is not an embedding table that can sit in the resident file. It is
14.6 M rows *per output token* of a 203 GB table, whose indices depend on a
tokenizer-derived map and on a rolling hash across the four preceding tokens.

**In `prepare_agentworld.py`'s loader: no analogue.** `convert.py` in the
checkpoint repo special-cases two checkpoint name components, `tie2eid` and
`tid2eid`, that do not appear in `inference/model.py` at all; their role is an
open unknown (§12).

**Sizing reality**, computed from the verified shapes and TinyTitan's own
effective bits per weight (4.25 at 4-bit, 8.25 at 8-bit, affine group 64):

| Engram precision | Stored size |
| --- | ---: |
| 16-bit (what "convert from 16-bit" would mean) | **393 GB** |
| 8-bit affine | **203 GB** |
| 4-bit affine | **105 GB** |
| its own native FP8 E4M3 + E8M0/32 | **203 GB** |

Keeping Engram at 16-bit costs more than the entire published checkpoint
(510 GB) minus the experts. This is why §6 recommends leaving it at native FP8:
it costs exactly what the checkpoint already spends, it is the precision the
model was trained and evaluated at, and 8-bit affine buys nothing over it while
4-bit affine halves it at an unknown quality cost on a table whose rows are
lookup keys, not a smooth weight matrix.

### 3.4 Attention and MoE details that are new even without CSA2

- **Not MLA in the V3 sense.** Q *is* low-rank (`wq_a` 5120→1280, `q_norm`,
  `wq_b` 1280→64×512) but **KV is not**: a single `wkv` 5120→512 plus
  `kv_norm(512)`, one KV head shared by all 64 query heads. There are **no
  separate qk_nope/qk_rope projection tensors** — `nope_head_dim` is computed in
  the reference and never used. The split is purely positional: the **last 64**
  of the 512 dims are RoPE-rotated, and the attention output gets the inverse
  rotation on those dims so the cache can stay in one rotated form.
- **`o_groups` = 8.** `wo_a` is a grouped, block-diagonal projection
  (`[8192, 4096]`, 8 groups of 8 contiguous heads → 1024 each), then `wo_b`
  `[5120, 8192]`. Group *g* sees only its own eight heads.
- **`attn_sink`** is a learnable per-head fp32 bias `[64]` added into the
  softmax denominator. `softmax_scale = head_dim^−0.5` = 512^−0.5, not the
  rope-dim scale.
- **Router.** `scores = softplus(linear(x)/gate_temp).sqrt()`. Selection is
  `topk(scores + bias)` with the **unbiased** scores gathered as the weights,
  normalized when `norm_topk_prob`, then scaled by 1.5. The selection bias
  never scales the weights. `topk_method: noaux_tc` is not a field in the
  reference; the semantics above are what it implements. There is a second
  `gate.bias_vl` of the same shape used when vision is enabled — semantics of
  when it applies are an open unknown (§12), but a text-only build must decide
  which bias to use and say so.
- **`swiglu_limit` = 10.0**: `up = clamp(up, −10, +10)` both sides,
  `gate = clamp(gate, max=10)` upper only, then `silu(gate)·up`. Not optional —
  the reference comment ties it to keeping fp8/fp4 activations in range.
- **Hyper-connections are Sinkhorn-normalized** (as in the Qwen3.8 port, but
  with this model's constants `hc_mult` 4, 20 iterations, eps 1e-6), with a
  **scheduling quirk**: a sub-block's coefficients are consumed by the *next*
  one. Attention collapses with the previous FFN's `pre_mix`; the FFN returns
  the next block's. The stream starts as identity one-hot on copy 0. Whatever
  the Qwen3.8 port built must be checked against this ordering before reuse.
- **MTP/DSpark** is a 3-layer draft with 128 experts top-3, `dspark_block_size`
  5, a Markov bigram correction head and a confidence head, reading the
  **attention input** of target layers 37/38/39 (not their output). The
  reference repo implements only the forward path; its own README says
  "Generation itself is plain autoregressive sampling", and the
  confidence-scheduled verification loop is absent. So there is no reference
  accept/reject algorithm to port — only a forward one.

### 3.5 Vision — deferred, per decision

The vision tower is genuinely present in this repo (unlike some Qwen
checkpoints, which declare a `vision_config` and ship language weights only):
`vision.*` and `aligner.*` tensors exist, ≈ 0.26 GB BF16. Tower: 32 layers,
dim 1024, 16 heads of 64, intermediate 2816, patch 14, `downsample_ratio` 3,
`max_image_tokens` 1024, `min_pixels` 295936. `PatchEmbed.proj` is
`Linear(3·14·14 = 588 → 1024)`; 2D RoPE splits 1024/16/2 = 32 dims per axis;
blocks are **full bidirectional** attention (no mask); the aligner is a 3×3
pixel-unshuffle (`1024·9 = 9216`) → 5120 → GELU → 5120, i.e. it **does** map to
the text hidden size.

Image tokens are emitted as
`[IMAGE_START] + ([IMAGE]×w + [IMAGE_NEW_LINE])×h + [IMAGE_END]`, with
`n_llm_h·(n_llm_w+1) + 2` positions, **every one carrying `image_token_id`
129264** in `input_ids` and distinguished only by `token_type`
(TEXT = −1, START/IMAGE/NEW_LINE/END = 0..3). START/END/NEW_LINE take learned
`image_start`/`image_end`/`image_newline` [5120] embeddings; the IMAGE slots take
aligner rows in reading order.

**Decision: text-only first, vision deferred** — the Qwen3.8-Flash-Next
precedent. The consequence to state honestly is that a deferred vision path is
not a bolt-on here: the Engram hash map treats image spans as DEAD, the router
has a separate `bias_vl`, and the tokenizer's placeholder id is entangled with
the input pipeline. Deferring vision defers that coupling; it does not avoid it.

---

## 4. What this tree already has that the port can reuse

Recorded because it is the difference between "months" and "never", and because
`adding-a-model.md`'s first question is whether this is a wiring job.

| Need | Existing asset | Fit |
| --- | --- | --- |
| 4-bit/8-bit affine conversion | `tools/prepare_agentworld.py`, `prepare_qwen38.py` — streaming shard-at-a-time, group-64 affine, bf16 keep list, `--plan` mode | **Partial.** The pipeline is right; the *input* is FP8/FP4 and the tensor namespaces do not match any it knows |
| Streaming + bounded cache | the v4.1 expert streaming engine (`docs/v4.1-expert-streaming-engine.md`) — per-layer expert blobs, 16 KiB alignment, hit/fixup schedule, `F_NOCACHE` preads | **Strong.** This is the mechanism that makes a 763 B model conceivable at all |
| Sparse attention | `Kernels/Attention/QSAIndexer.swift`, `Runtime/Family/QSAExactness.swift` (Qwen3.8-Flash-Next) | **Partial.** Global, per-layer, dense-over-context indexer; CSA2 needs per-source-layer sharing, ratio-scaled compressed caches and a two-level candidate pre-filter |
| Hyper-connections | `Runtime/Family/Qwen38FlashFamily.swift`, `ModelProfile.hcFused` | **Partial.** Same idea, different constants and a different sub-block ordering |
| KV quantization | 4-bit/8-bit KV paths (`kvCachePrecision`, `KVCacheQuantizer`) | **Partial.** Different block layout and granularity |
| Dense FP8 / FP4 kernels | none | **Missing** |
| Engram | none | **Missing** |
| Chat template | renderer requires one | **Missing** |

---

## 5. Tensor mapping (checkpoint → resident index)

The checkpoint uses bare names — `layers.N.*`, `embed.weight`, `head.weight` —
which match **none** of the namespaces `prepare_agentworld.py` knows
(`model.language_model.*`, `lm_head.weight`, `model.visual.*`, `mtp.*`). This is
a new converter with a new rename table, not a new `MODELS` row.

| Checkpoint name | Resident stem | Slot |
| --- | --- | --- |
| `embed.weight` | `language_model.model.embed_tokens.weight` | `embedding` |
| `head.weight` | `language_model.lm_head.weight` | `embedding`/head |
| `norm.weight` | `language_model.model.norm.weight` | bf16 keep |
| `layers.N.attn.wq_a` / `wq_b` / `wkv` / `wo_b` | `language_model.model.layers.N.self_attn.*` | `attention` |
| `layers.N.attn.wo_a` | same | `attention`, **bf16 keep** (published bf16) |
| `layers.N.attn.q_norm` / `kv_norm` | same | bf16 keep |
| `layers.N.attn.attn_sink` | same | f32 keep |
| `layers.N.attn_norm` / `ffn_norm` | same | bf16 keep |
| `layers.N.ffn.gate.weight` | `language_model.model.layers.N.mlp.gate.weight` | `router`, bf16 keep |
| `layers.N.ffn.gate.bias` / `bias_vl` | same | f32 keep |
| `layers.N.ffn.shared_experts.w1/w2/w3` | `…mlp.shared_expert.*` | `sharedExpert` |
| `layers.N.ffn.experts.E.w1/w2/w3` | `…mlp.experts.E.*` | `routedExpert` (per-layer files) |
| `layers.N.hc_attn_*` / `hc_ffn_*` | `language_model.model.layers.N.hc_*` | f32 keep |
| `layers.{1,14}.engram.embed` | **new streaming section** | see §6.4 |
| `layers.{1,14}.engram.wkv` | new | `engram` |
| `layers.{1,14}.engram.q_weight` / `k_weight` | new | bf16 keep |
| `mtp.{0,1,2}.*` | `language_model.mtp.N.*` | new `dsparkMTP` family |
| `vision.*`, `aligner.*`, `image_start/end/newline` | — | **excluded** (text-only) |

The router also needs the derived Engram constants — the 99092-entry
compressed-token map, the 48 bucket primes and sizes, and the per-(layer, shift)
odd multipliers — which are **not weights** and must be produced by the
converter into a sidecar the runtime reads. Nothing in `.ssdai` carries
non-tensor constants today.

---

## 6. The precision policy — preserve, do not degrade

**Decision (owner, 2026-09-25): weights stay in their original form. If
something must be converted or modified, it is not allowed to degrade the model
in any area.**

That single constraint settles most of this section and removes an entire
branch of work. The rest of this section is what "preserve, do not degrade"
means concretely, where it is cheap, and where it is not.

### 6.1 What the decision rules out, and what it does not touch

Ruled out, permanently for this model:

- the affine 4-bit / 8-bit re-quantization the rest of the catalog calls
  "4-Bit" and "8-Bit" (`docs/ssdai-format.md`'s five quant slots, the
  `prepare_agentworld.py` pipeline). Re-encoding into affine group 64 changes
  the stored values;
- any width reduction on any tensor — including the Engram tables, whose 16-bit
  form (393 GB) is not even the issue: their *published* FP8 is the bar;
- approximations anywhere in the architecture. The indexer, the candidate
  filter, the Engram gate, the Sinkhorn iterations, `swiglu_limit`, the grouped
  `wo_a` and the shared-KV publication order are not candidates for
  simplification, however tempting an "approximately correct" shortcut looks on
  a forward this large.

Unaffected, and still permitted, because none of it changes a weight value:

- **converting the container** — `.ssdai` instead of safetensors, a single
  index, 16 KiB alignment, per-layer expert files;
- **streaming and caching** — the access pattern is not part of the model;
- **dtype widening for compute**, as long as it is exact. Decoding FP8 E4M3 and
  FP4 E2M1 into `fp16`/`bf16`/`fp32` and multiplying is what the official
  reference does too (`fp4_gemm` casts FP4→FP8 through FP32 and runs an FP8×FP8
  GEMM with an fp32 accumulator). This is not degradation; it is the reference
  arithmetic.

The bar this creates is unusually crisp: **the converted install must reproduce
the published checkpoint bit-for-bit in its stored values and scales.** Every
quantized value and every E8M0 exponent in the install must equal the
corresponding bytes in the source safetensors. That is checkable without a
model forward at all (§8), and it is by far the strongest verification this
project has ever had on a conversion.

### 6.2 What is preserved, tensor group by tensor group

| Tensor group | Published form | Preserved as | In the install |
| --- | --- | --- | --- |
| routed experts (288.8 GB) | FP4 E2M1 packed 2/byte along K, E8M0 scale per 32 | byte-identical | per-layer expert files, band 2 |
| Engram `embed` (196.6 GB) | FP8 E4M3, E8M0 per 32 along the 256-dim axis | byte-identical | **new streamed band** (§6.4) |
| Engram `wkv` | FP8 E4M3 + E8M0 | byte-identical | resident |
| attention `wq_a`/`wq_b`/`wkv`/`wo_b` | FP8 E4M3 + E8M0 per 32×32 | byte-identical | resident |
| `wo_a` | BF16 | byte-identical | resident |
| shared experts w1/w2/w3 | FP8 E4M3 + E8M0 | byte-identical | resident |
| MTP/DSpark experts | FP4 E2M1 + E8M0 | byte-identical | per-layer files |
| MTP `main_proj` | FP8 E4M3 | byte-identical | resident |
| `ffn.gate.weight` | BF16 | byte-identical | resident |
| `ffn.gate.bias`, `bias_vl` | F32 | byte-identical | resident |
| `attn_sink`, `hc_*`, `DSparkConfidenceHead.proj` | F32 | byte-identical | resident |
| every norm, `q_weight`, `k_weight` | BF16 | byte-identical | resident |
| `embed.weight`, `head.weight` | BF16 | byte-identical | resident |
| vision / aligner | BF16 | excluded (text-only, §3.5) | — |

Nothing in this table is re-encoded. The install is a re-containerization of the
checkpoint plus an access layout — which is exactly what "converted but the
quants unchanged" means.

### 6.3 The runtime work this forces

The runtime's only weight format today is affine: `dequant_affine.metal` binds
`kAffineGroupSize = 64` and consumes packed `uint` weights with a `bfloat`
scale **and a `bfloat` bias** per group, shared by int4 and int8. There is no
block-scaled FP8 decode and no FP4 E2M1 decode in the tree. Preserving the
quants therefore buys fidelity at the cost of new kernels, and this is the
accepted trade. There is also a trap that will catch a careless converter: it is
tempting to "reuse" the affine path for the tensors whose shapes happen to fit.
They do not fit — `w1`/`w3` are `[2304, 2560]` with a `[2304, 160]` E8M0 scale
where 160 = 5120/32, and `w2` is `[5120, 1152]` with `[5120, 72]` where
72 = 2304/32. Those scale shapes are the **per-32-block** count, not a
group-of-64 count, and **2304 is not a multiple of 64** — the group size every
affine path in this tree assumes. A converter that reshapes these into affine
groups because the arithmetic happens to divide will produce a file that loads,
passes shape checks and is wrong (`docs/ssdai-format.md`'s failure mode). The
`--plan` mode must therefore report every scale shape it sees and refuse on any
tensor it cannot classify, rather than sizing bytes and dividing.

The work, in order:

1. **A block-FP8 GEMV** — decode E4M3 with a per-32×32 E8M0 exponent into fp16
   or bf16, accumulate in fp32. E8M0 is a power of two, so scale application is
   an exact exponent add: cheap and bit-exact.
2. **A packed-FP4 E2M1 GEMV** — two nibbles per byte, E8M0 scale per 32 along K,
   and the same dynamic per-32 activation scaling the reference applies. This
   is the hot kernel: it runs for 6 of 384 experts × 40 layers on every decode
   step.
3. **A manifest extension** for non-affine schemes: `scheme` gains
   `blockFP8`/`packedFP4` values and `scaleType` gains an "E8M0 exponent"
   meaning. `SSDAIBinary` remains the single writer, per `docs/ssdai-format.md`.
   The existing per-tensor `weightBits` entries must still be written for every
   tensor, because the old silent-misread failure mode does not care which
   format is being misread.
4. **An FP8→dequant embedding lookup** for Engram, since the existing
   `EmbedLookupInt4` path is affine int4.
5. **Engram weights stay BF16** (`q_weight`, `k_weight`) and every gate and
   Sinkhorn coefficient stays F32 — no rounding of a decision to save bytes.

### 6.4 Engram: stream it, do not shrink it

Engram is the one place where "preserve" collides with arithmetic, and the
resolution is worth stating plainly.

The two tables are 768 M rows — 98.3 B values per table — and **202.8 GB as
published**. They cannot be resident beside a ≈ 22 GB core. The available
responses were:

| Option | Size | Degrades? |
| --- | ---: | --- |
| keep as published, streamed | 202.8 GB | **no** |
| convert to 16-bit | 393 GB | not on its own, but larger than the whole checkpoint minus experts — and pointless, since the info is not there to recover |
| re-quantize to 4-bit affine | 105 GB | **yes** — forbidden by §6.1 |
| drop Engram entirely | 0 | **yes** — it is 196 B of the model's parameters |

So Engram is **streamed at its published FP8**, on a new `.ssdai` band with the
same design as the expert files: one blob per bucket — the 24 prime-sized buckets
per layer, each ≈ 16.0 M rows and ≈ 4.2 GB — aligned to 16 KiB so a random row
inside one is a single 16 KiB-granular read. The rows stay 264 bytes and are
addressed individually; 16 KiB is the read granularity, **not** a per-row pad —
padding each of the 768 M rows to 16 KiB would be 12.6 TB. The access pattern is
favourable in a way the experts' is not — the four preceding tokens of a decode
step are already known, so the n-gram hash for the next position can be computed
one step ahead and its rows prefetched, which is what the existing
predictive-prefetch machinery is for.

Row size to budget with: 256 bytes of E4M3 plus 8 bytes of E8M0 per row = **264
bytes**. Two layers × 8 heads × 3 n-gram sizes = **48 rows per token = 12.7 KB
per token**; a 512-token generation streams ≈ 6.5 MB. That is small, and it is
the one part of this model's memory problem that has a cheap answer.

`docs/ssdai-format.md` describes exactly two regions (`index` + resident
payload) plus per-layer expert files; this is a third band.

### 6.5 The cost of preserving: it is time, not quality

Preserving the quants does not make the model fit — it makes the model
*faithful*, and pushes the whole problem onto I/O.

| Component | Preserved size | Can it be resident? |
| --- | ---: | --- |
| core (attention, router, shared experts, hc, embed/head, MTP core) | ≈ 22 GB | yes on a 64 GB+ machine |
| routed experts (40 × 384) | ≈ 288.8 GB | **no** — must stream |
| MTP/DSpark experts (3 × 128) | ≈ 7.3 GB | marginal |
| Engram (streamed) | ≈ 202.8 GB | **no** — streamed by design, §6.4 |
| **text-only total** | **≈ 510 GB** | the checkpoint's own size |

The per-token expert traffic is the number that decides whether this is usable:
6 of 384 experts × 40 layers = 240 expert activations per token. One expert is
`w1` + `w2` + `w3` = 3 × 5120 × 2304 = 35.4 M parameters, which at FP4 E2M1 with
the per-32 E8M0 scale is **18.8 MB** — and the same number falls out of §5's
288.8 GB divided by 40 × 384 experts (18.80 MB), so the geometry and the
checkpoint agree. That is **≈ 4.5 GB of random reads per token**.

| Underlying read throughput | Decode cost per token |
| ---: | ---: |
| 100 MB/s | ≈ 45 s |
| 1 GB/s | ≈ 4.5 s |
| 3 GB/s (fast NVMe) | ≈ 1.5 s |
| resident in RAM | bounded by compute, not I/O |

The expert cache does not rescue this the way it rescues the 35B models. The
cache is budgeted in `bytes per layer`, so the same 8 GiB that holds 64 of
Qwen3.8's 2.64 MB experts holds about **11** of these 18.8 MB ones: 2.9% of a
layer's 384 experts, against Qwen3.8's 12.5%. It is routing skew that turns
Qwen3.8's 12.5% coverage into the 60–85% hit rate measured on this host, and the
same skew applied to a 2.9% foothold still leaves the great majority of the 240
activations missing. Working the expectation through —
`240 × (1 − coverage)` misses per token at 18.8 MB each — puts a 24 GB Mac at
**≈ 4.4 GB/token, ≈ 1.5 s, ≈ 0.7 tok/s** under uniform routing, and at
≈ 3.8 GB / 1.3 s / 0.8 tok/s if routing skew lifts the hit rate from 2.9% to
15%; the answer barely moves, because at this coverage almost everything misses
either way. What a larger cache would buy is model-dependent — uniform routing
still leaves three quarters of the activations missing at 64 GiB, while the skew
that gives Qwen3.8 its measured 60–85% hit rate at 12.5% coverage would close
much of that gap — so treat anything above ~1 tok/s as contingent on a ≥96 GB
machine, not on this one.

Today's expert cache is 10–12 GiB (`ModelProfile.table`, measured on 35B
models) and the 162 GB Qwen3.8-Flash-Next install already streams on this host.
Even a 64 GiB cache covers only ≈ 22–24% of the expert bytes, so on this machine
the steady state is a miss on most of the 240 activations per token.
**Preserving the quants makes this a streaming-throughput problem with a ~510 GB
working set on a machine that cannot hold it**, and no amount of kernel work
changes that — only storage does.

This is the honest consequence of the decision, and it belongs in front of the
owner before a byte is downloaded, alongside the disk table (which is now the
same 510 GB for the source and ≈ 510 GB for the install — the conversion no
longer shrinks anything, so peak space is ≈ 1.1 TB and the snapshot cannot be
deleted early).

**What the install costs on disk.** Preserving the quants means copying bytes, so
the install is the checkpoint's own size: **≈ 510 GB**, of which 288.8 GB is
routed experts and 202.8 GB is Engram — those two are 96% of it, and neither
shrinks without re-quantizing. The `.ssdai` container adds ~0.8 GB of worst-case
stride padding across 92,160 experts and a few megabytes of index, manifest and
receipt. Peak space depends on how the conversion is fed: **≈ 521 GB** if it
streams shard by shard the way `TinyTitanRepack` installs the Qwen checkpoints
today (install plus the one ~10.6 GB shard in flight), or **≈ 1.02 TB** if a full
local snapshot has to exist first, which is the case this section's table
assumes.

### 6.6 What the catalog should say

There is no 4-bit build and no 8-bit build. There is one install, at the
model's own mixed precision: 8-bit dense, 4-bit experts. Under
`docs/ssdai-format.md`'s rules, `ManifestIdentity.weightBits` reads the
`routedExpert` slot, and the API id appends it — which would produce
`deepseek-v4.1-flash_4-Bit` and advertise a re-quantized width that does not
exist. That is a naming change to settle deliberately:

- either the id carries no width and the model is listed as
  `deepseek-v4.1-flash` with `displayNames` stating "native FP8/FP4",
- or a distinct marker is used that cannot be confused with the affine widths,
  and the catalog's width-based duplicate detection is taught about it.

The per-tensor `weightBits`/`scheme` entries must still be written for every
tensor, so a reader can never unpack an FP8 or FP4 tensor as affine.

## 7. The eight wiring points

### 7.0 The integration surface, subsystem by subsystem

`adding-a-model.md` frames the work as "wiring job or runtime job", and the
answer here is runtime — but it is clearer to say exactly which existing type
each DeepSeek subsystem extends, parallels, or has no counterpart for. Every path
below exists in this tree today unless marked **new**.

| DeepSeek subsystem | Nearest existing asset | What has to happen |
| --- | --- | --- |
| CSA2 window rings + compressed caches + indexer publication | `Kernels/Attention/QSAIndexer.swift`, `Metal/Attention/qsa.metal`, `Runtime/Family/QSAExactness.swift` (Qwen3.8's global indexer) | **Extend.** QSA selects globally per layer; CSA2 needs per-layer windows, four shared ratio-scaled caches, publish-before-consume ordering and a two-level candidate filter. `QSAExactness.swift` is the closest prior art for the verification shape, not for the mechanism |
| Engram lookup, hashing, gate | nothing | **New.** No hashing, no bucket table, no streamed embedding band exists. `Kernels/Quant/EmbedLookupInt4.swift` is an affine int4 lookup, not FP8, and is resident-only |
| Sinkhorn mHC | `Runtime/Family/Qwen38FlashFamily.swift`, `ModelProfile.hcFused` | **Extend.** Same idea, different constants (20 iterations, eps 1e-6) and a different sub-block scheduling order — check the ordering before reusing anything |
| MoE routing (`sqrtsoftplus`, `noaux_tc`, top-6 of 384, `swiglu_limit`) | `Kernels/MoE/`, `sources/TinyTitan/Kernels/Prefill/MoE/` | **Extend.** The existing families are softmax→topk at top-8; scoring, the selection-vs-weight bias split, and the SwiGLU clamp are new |
| Block-FP8 GEMV | `Kernels/Quant/DequantInt8GEMV.swift`, `Metal/Quant/dequant_affine.metal` | **New kernel + Swift binding.** The affine kernel is group-64 with scale *and* bias; FP8 is per-32×32 with an E8M0 exponent |
| Packed-FP4 E2M1 GEMV | `Kernels/Quant/DequantInt4GEMV.swift` (`dequant_int4.metal`) | **New kernel + Swift binding.** Int4 affine ≠ FP4 E2M1; the E2M1 range is ±6.0 and the scale is a power-of-two exponent |
| Native KV formats (FP4 E2M1/E4M3-16, FP4 E8M0/32, FP8) | `Kernels/Quant/KVCacheQuantizer.swift` | **Extend.** A different scheme at a different granularity; a cache written by the affine quantizer would pass shape checks and change every attention output |
| Engine family dispatch | `Runtime/Family/TensorSchema.swift`, `Infrastructure/ModelIO/ModelTypes.swift` (`ModelFamily`) | **Extend.** Add `deepseekV41` to `ModelFamily` and `TensorSchema`, plus a **new** `Runtime/Family/DeepseekV41Family.swift` for the forward |
| Manifest arch + quant scheme | `TinyTitanRepack/Core/Format/ArchInfo+Loaders.swift` (`loadQwen4Exp` is the first-of-family precedent), `SSDAIJSON.swift`, `SSDAIEncoders.swift` | **Extend.** A `loadDeepseekV41` branch, a `RepackModelFamily.deepseekV41` case, and `scheme`/`scaleType` values for non-affine formats. `SSDAIBinary` stays the single writer |
| Engram streaming band | `Runtime/Inference/ModelExpertIO.swift`, `RealForwardRunner+Decode.swift` (expert streaming and prefetch) | **Extend.** The mechanism is right; the working set and the row-grained access pattern are not |
| Per-install tuning row | `Runtime/Configuration/ModelProfile.swift` (`table`, keyed by model id + width) | **Extend.** Note the key carries a width, and §6.6 says this model has none — the row's key is a decision, not a default |
| Served id → name | `TinyTitanServer/Core/ModelCatalog.swift` (`displayNames`) | **Extend.** Same width caveat |
| Converter | `tools/prepare_agentworld.py`, `prepare_qwen38.py` | **New** `tools/prepare_dsv41.py`. The checkpoint's bare `layers.N.*` / `embed.weight` names match none of the namespaces the existing converters know |
| Install catalogue and launcher | `tools/install_models.sh`, `tools/tinytitan_models.sh`, `tools/server_launcher.sh` | **Extend.** One row, not two — there is no width pair |

Read the table as the answer to "how much of this is new": two kernels, one
family, one converter, one streamed band, and extensions to eight existing
types. That is the shape of a multi-phase runtime project, which is why §9
sequences it instead of scheduling it.

### 7.1 The eight wiring points themselves

Same eight as `adding-a-model.md`, with what each one costs here.

| # | Where | What changes |
| --- | --- | --- |
| 1 | `tools/prepare_agentworld.py` — `MODELS` | Not this file. A **new** `tools/prepare_dsv41.py` (namespaces differ), with the pinned sha `dba1be0a…` and its own rename table |
| 2 | `tools/install_models.sh` — `CATALOGUE` | Two rows and the preset→served-id `case` |
| 3 | `tools/tinytitan_models.sh` | key/stem/label `case` plus `ENGINES`, `THINKING`, `FAMILY`; unknown-model help; `TINYTITAN_ALL_MODELS` |
| 4 | `tools/server_launcher.sh` | model-key list in the header comment and the unknown-model error |
| 5 | `ModelProfile.swift` | One row per width; expert-cache budget and chunk must be *measured*, not inherited — this geometry matches nothing shipped |
| 6 | `ModelCatalog.swift` — `displayNames` | served id → human name |
| 7 | `tests/` | `ModelProfileTests.shipped` and the table count |
| 8 | ANE prefill sidecar | Likely **skip**: for a model this size the exporter has no graph for the new attention, and the Qwen3.8 precedent already shows the ANE losing where the GPU path attends sparsely (0.72×). Export explicitly to re-measure; never install by default |

Beyond the eight: `ArchInfo+Loaders.swift` needs a `loadDeepseekV41` branch and a
`RepackModelFamily` case (`deepseekV41`), a new `ModelFamily` case, a new
`TensorSchema` mapping, and a new `Runtime/Family/DeepseekV41Family.swift`.
`ArchInfo`'s `crossCheckProduction*` pattern means the new branch either matches
a shipped geometry exactly or is a runtime question answered *before*
conversion — here there is no shipped sibling, so the cross-check has to be
written fresh from the published config (there is precedent: `loadQwen4Exp`
does exactly this for a first-of-family model).

---

## 8. Verification — building a surrogate, not downloading 510 GB

`adding-a-model.md`'s bar (continuations, golden baseline, receipt, catalog,
first measured row) is the right gate but is unaffordable at the front of this
project: nothing can be installed before the storage question is answered, and a
510 GB snapshot behind every iteration is not a test loop.

**Gate 1 — the container, before any forward (§9 Phase 1).** Because the
conversion re-encodes nothing, its correctness is checkable without a model:
read each tensor from the source safetensors and from the `.ssdai` install,
dequantize both with the same routine, and compare in fp32. The values must be
identical, and because the scale exponents are preserved the comparison can
assert exact equality of every stored quantized value and every E8M0 exponent
rather than a tolerance. A mismatch is a converter bug, full stop. This is the
cheapest and highest-value test in the whole project, and it catches the
`ssdai-format` class of failure — a tensor written at one width and read at
another — at container level.

**Gate 2 — the kernels, on synthetic tensors (§9 Phase 2).** A block-FP8 GEMV
and a packed-FP4 E2M1 GEMV, each multiplied against the Python reference within
fp32 tolerance, on tensors whose values are known independently.

**Gate 3 — a synthetic surrogate, full forward.** The cheap path this tree
already supports in spirit — `tools/testdata/`, `TinyTitanValidation`'s
reference kernels, `qwen38_full_forward.py` and `qwen38_parity.py`:

1. Generate a tiny checkpoint with the *same tensor names, dtypes and config
   keys* but 4 layers and 8 experts, using the official `inference/model.py` as
   the reference forward. Random weights, real plumbing — and real FP8/FP4
   encodings, since the kernels are what is under test.
2. Prove: window ring buffer, compressor grouping at ratios 1 and 2,
   cross-layer KV publication order, the indexer's top-k + −1 masking, the
   two-level candidate filter, grouped `wo_a`, sink, `swiglu_limit`,
   Sinkhorn convergence to doubly stochastic, the Engram hash and gate.
3. Only then convert the real checkpoint (§9 Phase 6).

Additional gates the standard checklist does not cover and this model needs:

- **Layer-20 boundary.** Layers 21–39 read a KV projected from the final
  *encoder* hidden state; a test that crosses from layer 19 to 21 is the natural
  place for this to be silently wrong.
- **Ratio crossing.** A decode sequence that completes a compression group
  (`end_pos % ratio == 0`) and one that does not.
- **Window boundary.** Sequences shorter than 128 and longer than 128, since the
  ring's "oldest-first" ordering differs between prefill and decode.
- **Engram hash parity** against `encoding/`'s reference for the same token
  stream, including a DEAD (image-span) interruption and a `pad_id` fill.
- **KV-cache format parity.** The native cache is FP4 E2M1 with an E4M3 scale
  per 16 channels (main), FP4 with E8M0/32 (indexer) and FP8 (window). The
  existing affine KV quantizer is a different scheme, so a cache written by the
  wrong one would pass a shape check and quietly change every attention output.
- **End-to-end logits** against the reference on the surrogate once every
  subsystem is in, which is the first test that can catch an error the
  per-subsystem gates each missed.

---

## 9. Phased plan

Each phase has an exit condition; none of them is "it answers". The ordering
reflects the owner's preserve decision (§6): the decode kernels are no longer a
fidelity upgrade at the end, they are on the critical path from Phase 1.

- **Phase 0 — unblock the front door.** Derive `chat_template.jinja` from
  `encoding/` and prove it against the reference's own test cases. Nothing else
  can be installed without it. Small, exacting, and independent of the model.
  The two things it must not get wrong, both from
  [`deepseek-v41-flash-reference.md`](deepseek-v41-flash-reference.md) §9.2:
  the **V4.1 tool-call dialect** — `<｜DSML｜ calls>` with a **leading space**,
  where V4 used `<｜DSML｜tool_calls>` without one, with
  `<｜DSML｜ parameter … string="true|false">` distinguishing a raw string from a
  JSON value — and the **two thinking modes**, since chat mode emits `</think>`
  immediately after `<｜Assistant｜>` while thinking mode wraps the reasoning and
  injects the one-time `<｜System｜>Reasoning Effort: {budget} …` line. There is
  no upstream template to diff against, so the reference's `encoding/` tests are
  the only oracle.
- **Phase 1 — a bit-exact container, no forward.** A new `prepare_dsv41.py`
  with a `--plan` mode that fetches only `config.json` and the index, classifies
  every tensor, and refuses on anything it cannot map. Then the conversion
  itself, whose exit condition is **byte-exactness**: every quantized value and
  every E8M0 exponent in the output equals the source safetensors, proven by
  dequantizing both sides and diffing (§8). No runtime work is involved, and
  this is the cheapest possible place to catch a format mistake.
- **Phase 2 — the decode kernels.** Block-FP8 GEMV, packed-FP4 E2M1 GEMV, the
  FP8 embedding lookup, and the manifest `scheme`/`scaleType` extension with
  `SSDAIBinary` as the only writer. Exit condition: a synthetic tensor of each
  format multiplied against the Python reference within fp32 tolerance, and no
  tensor anywhere in the tree unpacked as affine by accident.
- **Phase 3 — surrogate parity.** §8's synthetic checkpoint through the full
  forward, against the official reference. This is where the real schedule risk
  lives: the indexer, the candidate filter, shared-KV publication order, the
  Engram gate, the Sinkhorn mixing.
- **Phase 4 — runtime attention.** `SharedAttentionRuntime` equivalents:
  per-layer windows, four shared compressed caches, indexer publication order,
  the candidate filter, and a streaming band for the caches.
- **Phase 5 — Engram.** The new `.ssdai` band, the tokenizer-derived constants
  sidecar, the streamed lookup, and the one-step-ahead prefetch (§6.4).
- **Phase 6 — a real install, storage and throughput first.** The operator
  decisions in §11 (≈ 1.1 TB peak space; expected decode throughput from
  §6.5). No snapshot can be deleted early, because the conversion no longer
  shrinks anything.
- **Deferred — vision.** Per the decision in §3.5.


---

## 10. Deviations to state in user-facing docs

- **No 16-bit source.** The conversion master is FP8/FP4; "16-bit" is a
  reconstruction. State it wherever "4-bit and 8-bit, converted from 16-bit"
  would otherwise be claimed.
- **There is no 4-bit build and no 8-bit build.** One install at the model's own
  precision (8-bit dense, 4-bit experts). Claiming a width would imply a
  re-quantization that is now explicitly forbidden (§6), and the catalog's
  width-appended id has to be handled deliberately (§6.6).
- **Nothing is re-quantized.** The install's quantized values and E8M0
  exponents equal the source safetensors byte for byte; that is the contract,
  and it is verifiable without a forward pass.
- **Chat template is ours, not the model's.** This release ships none, so the
  template is a TinyTitan artifact and must be versioned and tested accordingly.
- **No `generation_config.json`**; sampling comes from the card (temperature
  1.0, top_p 0.95) and is a documented choice, not the checkpoint's.
- **Presence penalty 0.0 only**, as everywhere in this runtime.
- **Thinking levels map onto `reasoning_effort` 1–100** with aliases
  low→50 / high→75 / max→100; any lossy mapping is stated.
- **`bias_vl` is unused in a text-only build**; which bias the reference applies
  is an open unknown (§12), so the choice is a recorded deviation.
- **Vision is excluded**, and the image-span/DEAD-token coupling is deferred
  rather than absent.
- **Decode is I/O-bound, not compute-bound.** With the quants preserved, the
  expert working set is ≈ 290 GB streamed through a cache that holds a fraction
  of it (§6.5). State the expected throughput honestly rather than letting a
  number from a re-quantized build stand in for it.

---

## 11. Open questions for the owner

1. **Storage and throughput.** Preserving the quants means the install is
   ≈ 510 GB and peak space is ≈ 1.1 TB (the conversion no longer shrinks
   anything, so no snapshot can be dropped early), against 292 GiB free. And
   §6.5's arithmetic puts decode at seconds per token unless the expert working
   set is largely resident. Expand storage, run on a different host, or accept
   the throughput?
2. **The catalog id (§6.6).** Drop the width from the served id, or introduce a
   marker that cannot be confused with the affine 4-Bit/8-Bit ids — knowing the
   catalog's duplicate detection is width-based.
3. **Engram streaming (§6.4).** Agree that the Engram tables stream at their
   published FP8 and are never narrowed? At 264 bytes per row and 48 rows per
   token, this is the cheap part of the memory problem; the alternative is
   forbidden by the no-degradation rule.
4. **Is a ≥ 1 TFLOP-scale model the intended target at all** on 24–128 GB
   machines, given that this model's *resident* core plus Engram is already
   beyond the smaller end even with perfect expert streaming?

---

## 12. Open unknowns (not to be guessed at implementation time)

- **The tech report is unreadable**: `DeepSeek_V41_Tech_Report.pdf` is a
  Git-LFS pointer (`sha256 ba68e2e4…`, 1,809,802 bytes). Anything only in it —
  including any reconciliation of 748 B vs 763.2 B — is unknown.
- **`bias_vl` semantics**: when the vision-gate bias is applied, and whether a
  text-only forward should use it, is not stated in anything read.
- **"SWA Bounded Replay"** appears only in card prose; no reference code.
- **The confidence-scheduled verification loop** for DSpark is not in the repo;
  only the forward path is.
- **`tie2eid` / `tid2eid`**: checkpoint name components special-cased in
  `convert.py` that do not appear in `inference/model.py`.
- **`dspark_noise_token_id` 128799**'s token string is not identified.
- **`Full`/`Reindex`/`Reuse`** naming is prose; the mapping in §3.2 is inferred
  from the source-layer lists.
- **No hardware requirement is stated** anywhere: no minimum memory, no GPU
  count, no vLLM/SGLang version. The only quantity is tensor-parallel examples
  (`MP=8`, `MP=4` as an override) for a CUDA-only reference stack.
- **Community quantizations exist** (many FP8 mirrors, NVFP4 from NVIDIA
  ModelOpt, EXL3/GPTQ/GGUF/MLX 2–4 bit variants, and a mixed 4/8-bit MLX
  checkpoint) — but none is an official MLX release, and this document takes no
  position on borrowing from them, consistent with the Qwen3.8-Flash-Next
  record's "wait for an official release" decision.

---

## Checklist

Modelled on `adding-a-model.md`, with every box unticked and the blockers named.
The preserve decision (§6) is taken; these are the consequences.

- [ ] Chat template derived from `encoding/` and proven against its tests —
      **blocker for every other item**
- [ ] Storage and throughput settled (≈ 1.1 TB peak and seconds-per-token unless
      the expert set is largely resident, §6.5 — or a different host)
- [ ] Gating, pinned sha, geometry, rope block, tie-embeddings and EOS ids read
      from the source — **done, §1**
- [ ] `--plan` classifies every tensor, and every scale shape is reported and
      read as a per-32 block count rather than reshaped into affine groups
      (§6.3)
- [ ] **Gate 1: byte-exact container** — every stored quantized value and E8M0
      exponent matches the source safetensors (§8)
- [ ] **Gate 2: block-FP8 and packed-FP4 E2M1 GEMV kernels** plus the FP8
      embedding lookup, and the manifest `scheme`/`scaleType` extension
- [ ] `ArchInfo.loadDeepseekV41` + `RepackModelFamily.deepseekV41` + a
      `ModelFamily` case + `TensorSchema` mapping
- [ ] `SharedAttentionRuntime` equivalent: per-layer windows, shared compressed
      caches, publication order, candidate filter
- [ ] Native KV-cache formats (FP4 E2M1/E4M3-16, FP4 E8M0/32, FP8), not the
      affine KV quantizer
- [ ] A third `.ssdai` band for Engram, streamed at published FP8, plus its
      constants sidecar
- [ ] The eight wiring points, with the catalog width question (§6.6) settled
- [ ] Surrogate parity against `inference/model.py`, including the layer-20,
      ratio and window boundaries
- [ ] Conversion and install — one install, not two widths; snapshots cannot be
      deleted early because the conversion does not shrink anything
- [ ] Continuations, golden target, receipt, catalog, launcher keys
- [ ] First measured row (TTFT, decode) with machine and commit stated
- [ ] README, wiki pages, tracker, roadmap — deviations and status stated
- [ ] Vision: deferred by decision, recorded as such
