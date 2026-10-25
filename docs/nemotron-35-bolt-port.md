# Nemotron 3.5 Bolt 74B-A9B — port record

Status: **in progress.** This records what was read from the published
checkpoint and the decisions the port follows. Nothing here is called
supported until both installs exist, the runtime answers from them, and the
decode numbers below are measured on this host.

Request: integrate
[`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB)
at 4-bit and 8-bit, each derived from the **16-bit original**
(`…-BF16`), and publish decode tok/s for both widths.

## 1. The source checkpoint

| | Value | Read from |
| --- | --- | --- |
| Quantized card | `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB`, sha `cf2effc5c4b794ee9fb8d34eab3d1d65ce64142f` | HF API |
| 16-bit original | `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16`, sha `e09a4cedb1787db1760375ee5539f222cae0c672` (2026-10-23) | HF API |
| `gated` / license | `false` / OpenMDW-1.1 | HF API, card |
| `architectures` | `NemotronH_Omni_Reasoning_V3`; text tower `model_type: nemotron_h_puzzle` under `llm_config` | `config.json` |
| Size | **149.77 GB** in 63 safetensors shards; 28,725 index tensors | index + headers |
| Total / active parameters | 74 B / 9 B | card |
| Context | 262,144 | `llm_config.max_position_embeddings` |
| Vocabulary | 131,072, untied head (`tie_word_embeddings: false`) | `llm_config` |
| Sampling | temperature 1.0, top-p 0.95; EOS ids 2 and 11, BOS 1 | `generation_config.json` |
| Terminal-Bench 2.1 (card) | **50.98** | card benchmark table |
| Terminal-Bench 4.0 | none published | card |

The 24 GB card is a native **IQ2** (2-bit, QAT) quantisation of the BF16
release, not a 16-bit checkpoint. The two widths this project ships are
therefore quantized here from the BF16 weights, not copied from the card.

### Text tower geometry (`llm_config`)

| | |
| --- | --- |
| Layers | **88**: 40 `mamba`, 40 `moe`, 8 `attention` (indices 7, 16, 25, 36, 47, 58, 69, 80) |
| Hidden | 4,096; pre-norm RMSNorm, eps 1e-5; residual added after the mixer |
| MoE | 40 layers; **240 / 304 / 368 / 432 / 496** routed experts per layer (13,824 total), top-9, latent width **1,024**, per-expert width 1,536–2,560 (per layer), shared expert 5,376, activation **relu²** |
| Router | sigmoid scores + `e_score_correction_bias`, 8 groups choose top-1 group by top-2 sum, top-9 globally, normalize top-k, ×5.0 |
| Mamba-2 | d_inner 8,192, 128 heads × 64, state 128, 8 groups, conv kernel 4 with bias, `in_proj` 18,560 = 8,192 gate + 10,240 conv (8,192 x + 1,024 B + 1,024 C) + 128 dt, gated RMSNorm group 1,024, SiLU, `softplus(dt + dt_bias)` clamped at 1e-4, `A = -exp(A_log)`, D skip |
| Attention | 32 query heads, 2 KV heads, head dim 128, **no RoPE** (the reference applies none), scale 128^-0.5, no bias, causal |
| Resident tensors | `language_model.backbone.{embeddings,norm_f,layers.*}`, `language_model.lm_head`; the MoE latent projections are `mixer.fc1_latent_proj` / `fc2_latent_proj` |
| Vision | C-RADIO v4-H tower + `mlp1` projector (1.31 GB + 0.38 GB); **excluded** from a text install |

The reference forward every implementation detail above was read from is
`modeling_nemotron_h.py` at the pinned BF16 sha. The Mamba-2 decode path is a
single-token state update, so a decode implementation needs no chunked-scan
kernel; prefill may use the same per-token update.

## 2. What TinyTitan must add

This is a new architecture, not a Qwen wiring job. The pieces:

1. `ArchInfo` loader for `nemotron_h_puzzle` (root nests the text config under
   `llm_config`, not `text_config`), a `RepackModelFamily.nemotronH`, and the
   extension fields the geometry needs (Mamba shape, latent MoE, per-layer
   expert counts and widths, attention mask).
2. A converter that reads the BF16 shards and writes 4-bit and 8-bit affine
   `.ssdai` installs. At 149.77 GB per pass, two separate passes would read
   300 GB; the converter therefore writes both widths in **one** pass, directly
   into the install layout, with source shards deleted as they are consumed.
3. A text-only runtime path: embedding → 88 blocks → final norm → head, with
   the 40 Mamba layers holding SSM and conv state per sequence and the 40 MoE
   layers streaming routed experts from `packed_experts`.
4. Catalogue/profile/tooling rows, tests, and the README table row at the
   card's Terminal-Bench 2.1 score, inserted in the table's descending order.

### Quantisation

Affine group-64, the format the repacker and every kernel already speak:
per-group min/scale, `q = round((x - min) / scale)` packed LSB-first into
32-bit lanes, scale and bias stored bf16. Routed experts, attention, Mamba and
the latent/shared projections are quantized; norms and the router stay bf16;
the embedding and untied head are kept at 8 bits in both installs, as the
other families do.

Expected installed size, computed from the index (group-64 overhead included):

| Build | Routed experts | Resident | Total (approx.) |
| --- | ---: | ---: | ---: |
| 4-bit | 37.2 GB | 8.6 GB | **45.8 GB** |
| 8-bit | 70.3 GB | 18.2 GB | **88.5 GB** |

Both installs are ~134 GB, so the host needs its disposable build artifacts
cleared first (`.build` is 47 GB here); the source is never staged whole.

### Decode design

Decode is memory-bound: ~9 B active parameters per token, so 4-bit reads
~4.5 GB of weights per token and 8-bit ~9 GB. The expert cache and prefetch
ring are the same levers the Qwen families use; the resident (non-expert)
weights are small enough to stay mapped.

## 3. Benchmarks

To be filled only from measurements on this host:

| Build | GPU decode tok/s | ANE decode tok/s |
| --- | ---: | ---: |
| 4-bit | TBD | TBD |
| 8-bit | TBD | TBD |

ANE note, to be stated in the README exactly as measured: this project's ANE
path is a **prefill** attention sidecar (v4.5) — the Neural Engine is not
programmable for the Mamba-2 recurrence, and Core ML cannot express the
streaming routed experts. A decode column is only published if a real path
exists; it is not replaced with a guess.
