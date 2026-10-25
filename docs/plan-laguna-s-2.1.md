# Plan: Poolside Laguna-S-2.1

Status: **runtime port and conversion in progress.** The install format, the
architecture reader, the resident schema, the decode probe and the converter
exist; the real 4-bit and 8-bit installs are being built, and the serving
runner does not have a Laguna graph yet.

The checkpoint is `poolside/Laguna-S-2.1`, OpenMDW-1.1, ungated, at sha
`0f573140834b11cfac0c2af97a101a7a69a13e22` (2026-08-19). Everything below was
read from that checkpoint's `config.json`, `configuration_laguna.py`,
`modeling_laguna.py`, the safetensors index and `generation_config.json`.

## The checkpoint

| | |
| --- | --- |
| Architecture | `LagunaForCausalLM`, `model_type: laguna` |
| Size | 118B total, ~8B active per token, 235,123,955,200 bytes in 46 shards |
| Tensors | 36,769 in the index (36,096 routed-expert tensors) |
| Layers | 48; layer 0 dense, layers 1-47 sparse |
| Hidden | 3,072 |
| Attention | 48 query heads on the 12 global layers, 72 on the 36 sliding-window layers; 8 KV heads; head dim 128; GQA; **per-head** softplus output gate (`g_proj`: hidden -> heads) |
| QK norms | per-head RMSNorm over 128 dims on q and k, before RoPE |
| Layout | 1:3 global-to-sliding, window 512; global layers `full_attention`, others `sliding_attention` |
| RoPE | global: YaRN, theta 500,000, factor 128 (original 8,192), beta 32/1, attention factor 1.4852030, partial 0.5; sliding: default RoPE, theta 10,000, partial 1.0 |
| MoE | 256 routed experts, top-10, expert width 1,024, one shared expert; sigmoid router with aux-loss-free `e_score_correction_bias`, top-k renormalised, routed output scaled by **2.5** |
| Norms | plain RMSNorm, eps 1e-6 |
| Vocabulary | 100,352, untied `lm_head` |
| Tokens | BOS/EOS id 2, EOS also 24; `generation_config.json` asks temperature 1.0, top-p 1.0, top-k 20 |
| Reasoning | `enable_thinking` boolean in the chat template; preserved thinking |

The 4-bit install is ~66.5 GB and the 8-bit ~124.9 GB by the converter's own
plan (`tools/prepare_laguna_s21.py --plan`); the source is 235 GB. This host
has 201 GB free after the converter intermediates from the previous port were
removed, so the two snapshots (178 GB) fit, and each install is written after
the snapshot is complete.

## What a layer does

**Layer 0 (dense).** `input_layernorm` -> GQA with QK norms, RoPE and the
per-head softplus gate -> `o_proj` -> residual -> `post_attention_layernorm`
-> plain SwiGLU MLP (3,072 -> 12,288) -> residual.

**Layers 1-47 (sparse).** The same attention, then `post_attention_layernorm`
feeds both the shared expert (3,072 -> 1,024, always on) and the router. The
router is bf16: sigmoid scores, the selection bias is added **only** to the
ranking, the chosen sigmoid scores are normalised over top-10, and the routed
sum is multiplied by 2.5 before the shared output is added. The routed experts
are fetched from SSD per token, exactly like the Qwen MoE path.

Two details are easy to get wrong and are pinned by tests:

- the gate is `softplus(x)` (plain, natural log), **not** K2's
  `softplus(x; beta = ln 2)`, and it is applied per head;
- the global and sliding layers use different RoPE dialects *and* different
  head counts, so neither can come from a single scalar in `ArchConfig`.

## The eight wiring points

1. **Converter** — `tools/prepare_laguna_s21.py` (new). Streams one layer at a
   time as byte ranges, fuses its 256 per-expert tensors into one rank-3
   `mlp.switch_mlp.*_proj`, quantizes at both widths and writes one safetensors
   file per layer and width. A completed layer is never rewritten, so a
   dropped 15-hour run resumes at the next layer. `--plan`, `--self-test`,
   `--source` (local) and `--from-hf` modes; the self-test covers the group-64
   round trip, the 8-bit embedding override, the fused shape and the index.
2. **Arch reader** — `RepackModelFamily.laguna`,
   `sources/TinyTitanRepack/Core/Format/ArchInfo+Loaders+Laguna.swift`, plus
   `crossCheckLaguna` for the production geometry.
3. **Planner** — routed experts classify through `mlp.switch_mlp`, layer 0 is
   the lone dense layer, and the tokenizer travels with the payload.
4. **Runtime identity** — `ModelFamily.laguna`, `ArchConfig.lagunaS21` (a
   checked preset: the manifest must agree on every generic field), the
   `LagunaGeometry` block for the per-layer heads, QK norms, per-head gate and
   the two RoPE dialects.
5. **Resident schema** — `Model+Validation+Laguna.swift` checks every tensor,
   both MLP shapes, the bf16 router and bias, and the packed-expert layout. It
   bypasses the shared `validateExecutableGeometry` gate only because Laguna
   has sliding layers and a 3,072 hidden width, both of which the probe
   supports and the serving runner refuses by name.
6. **Probe graph** — `LagunaLayer` + `LagunaGraphProbe`: embedding, 48 layers
   (dense and sparse), final norm and untied head, with a plain FP16 K/V cache
   and per-token expert fetches.
7. **Benchmark** — `TinyTitanBench lagunagpu <install> "<prompt>" [n]` prints
   prefill seconds, decode tokens/second, the token ids and the raw decode.
8. **Surfaces** — tokenizer names, the EOS set, sampling defaults and the
   boolean thinking switch are wired. The server catalogue reports the family
   as not served yet, because only the probe exists.

## Kernels

Reused unchanged: affine GEMV (4/8-bit), bf16 GEMV, grouped/per-head RMSNorm,
`encodeSWA` (window attention), `encodeFull`, `encodeNeoxSubdim` (including the
existing YaRN frequency table), `k2_router_topk_select_kn` (sigmoid + bias +
renorm + scale), `k2_moe_accumulate_fp16`, the embedding lookup and the head.

Two changes:

- `softplus_gate_per_head_mul_fp16` (`Metal/Primitives/utility.metal`): one
  plain softplus value per head, broadcast over 128 channels. K2's existing
  kernel uses beta = ln 2 and a per-channel gate, so it is a different
  function, not a flag on the same one.
- `Attention.maxQHeads` 32 -> 96: Laguna's sliding layers run 72 query heads
  against 8 KV heads. The bound sizes the split-KV reduction scratch; 96 keeps
  the existing Qwen families untouched and costs about 12 MB in one buffer per
  `Attention`.

## Measurements

Filled in as the runs finish. The benchmark prompt is the project's canonical
`The capital of France is`, greedy, with a plain tokenizer encode (no chat
template), so the number is comparable with the K2 and Qwen runs.

| Install | Prompt | Prefill | Decode | Output |
| --- | ---: | ---: | ---: | --- |
| 4-bit | | | | |
| 8-bit | | | | |

## Open

- Serving graph: the server does not run Laguna; `lagunagpu` is the probe.
- ANE prefill sidecar: not exported. The engine's ANE path is prefill-only;
  the project has no ANE *decode*, so the decode numbers above are the GPU
  probe's.
- Batched prefill: none; the probe decodes one token at a time.
- Golden baseline: not captured. A baseline is valid per (machine, build,
  model); capture it once the installs are stable and a serving path exists.
