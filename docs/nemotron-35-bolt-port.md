# Nemotron 3.5 Bolt 74B-A9B — port record

Status: **conversion running.** This records what was read from the published
checkpoint and the decisions the port follows. Nothing here is called
supported until both installs exist, the runtime answers from them, and the
decode numbers below are measured on this host.

## 0. Progress

Done and verified on the toy fixture:

- the branch `nemotron-3.5-bolt` and this record;
- `tools/prepare_nemotron.py`: geometry validation, per-layer expert census,
  the direct `.ssdai` writer for both widths, and a resumable parallel shard
  download;
- `tools/test_nemotron_writer.py`: a 3-layer toy install accepted by the real
  `TinyTitanRepack --verify-install` at 4 and 8 bits; its wrapper
  `benchmark/test_nemotron_install.py` is in the Python suite;
- `NemotronInstall` + `NemotronModel` + `NemotronCPU` + `NemotronMetal`, and
  `tools/nemotron_reference.py`, with parity 4.9e-7 (CPU) / 3.4e-6 (GPU) on the
  toy at both widths and across three tokens;
- `TinyTitanBench nemotron|nemotrongpu|nemotronlogits|nemotroninfo` and the
  `TinyTitanCLI` branch that serves a Nemotron install directly;
- `benchmark/nemotron_decode.py`, the four-arm measurement script.

Running now, and what follows it:

1. `tools/prepare_nemotron.py --install` is writing both installs from the
   pinned BF16 release (149.77 GB over a ~3-4 MB/s link, so hours). It resumes
   per shard and per 64 MB chunk.
2. When it finishes: `TinyTitanRepack --verify-install` on both directories,
   then `TinyTitanBench nemotroninfo` on each.
3. `benchmark/nemotron_decode.py --runs 1` for the four arms.
4. The README table rows and the measured table in §3.
5. A full `swift build -c release` and the Python gates, then the branch is
   ready for `main`.

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
| Layers | **88**: 40 `mamba`, 40 `moe`, 8 `attention` (indices 7, 16, 25, 36, 47, 58, 69, 78) |
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

Measured from the index and the planner (`--plan`), group-64 overhead included:

| Build | Routed experts (padded) | Resident | Total |
| --- | ---: | ---: | ---: |
| 4-bit | 40.77 GB | 5.07 GB | **45.83 GB** |
| 8-bit | 77.01 GB | 8.45 GB | **85.46 GB** |

The routed experts are padded to the widest layer's blob (`expertStride`
2,949,120 B at 4-bit, 5,570,560 B at 8-bit), because the verifier accepts one
stride per install and requires `fileSize = expertsPerLayer * expertStride`.
That costs 3.56 GB and 6.73 GB of page padding respectively. Both installs are
131.3 GB; the source is never staged whole.

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

## 4. The runtime, as implemented

`NemotronInstall` reads the install directly rather than through the Qwen-shaped
`Model.load`: the manifest's `arch.nemotronH` block is the geometry, the
resident index is parsed from `model_weights.bin`, and `packed_experts/layout.json`
gives each layer's two-role (`up`, `down`) expert blobs. The weight files are
mmap'd; a 74B model is never resident.

`NemotronModel` is one straight transcription of the reference forward:

- every block is pre-norm RMSNorm, mixer, residual;
- Mamba-2: `in_proj` → depthwise conv (4 taps) → SiLU → split x/B/C → selective
  scan with `softplus(dt + dt_bias)` clamped at 1e-4, `A = -exp(A_log)`, D skip →
  gated RMSNorm (group 1,024) → `out_proj`;
- LatentMoE: BF16 router → sigmoid → `e_score_correction_bias` → top-9 (one
  group, so the grouped mask is a no-op) → normalize → ×5 → `fc1_latent_proj`,
  nine `up`/`down` pairs on the 1,024-wide latent, weighted sum →
  `fc2_latent_proj`, plus the shared 5,376-wide relu² expert on the unprojected
  residual;
- attention: 32/2×128, **no RoPE**, scale 128^-0.5, causal softmax over a
  per-layer KV cache;
- final RMSNorm then the untied 8-bit head.

Two matvec backends, one graph:

| Backend | GEMV | Status |
| --- | --- | --- |
| `NemotronCPU` | `tinytitan_int4/int8_affine_gemv` (NEON), rows split across cores | reference |
| `NemotronMetal` | `dequant_int4_gemv_simd` / `dequant_int8_gemv_simd`, one no-copy `MTLBuffer` per mmap'd file, a layer's nine experts in one command buffer | measurement path |

The elementwise kernels (RMSNorm, gated RMSNorm, conv, SSM update, SiLU, relu²,
BF16 router GEMV, embedding row) are in `TinyTitanKernelsC/nemotron_ops.c`.

**Checked against an independent implementation.** `tools/nemotron_reference.py`
is a NumPy transcription of the same reference forward that reads the same
install. On a three-layer toy install (mamba, MoE, attention; random weights)
with an input of three tokens — so the Mamba conv/SSM state and the KV cache are
both exercising their second and third steps:

| Arm | max |logit difference| vs NumPy |
| --- | ---: |
| 4-bit CPU | **4.9e-7** |
| 4-bit Metal | **3.4e-6** (FP16 GEMV) |
| 8-bit CPU | **5.0e-7** |
| 8-bit Metal | **3.4e-6** |

The writer itself is checked before any 150 GB download: `tools/test_nemotron_writer.py`
builds that toy install from scratch and the real `TinyTitanRepack --verify-install`
accepts both widths.

Commands:

```bash
TinyTitanBench nemotroninfo models/nemotron-3.5-bolt_74B_A9B_4Bit
TinyTitanBench nemotron      <install> "The capital of France is" 4   # CPU
TinyTitanBench nemotrongpu   <install> "The capital of France is" 4   # Metal
TinyTitanBench nemotronlogits <install> 5,7,9 [--gpu]                 # no tokenizer
benchmark/nemotron_decode.py --install4 ... --install8 ... --runs 1
```

## 5. The Neural Engine

There is **no ANE decode path for this model, and this is a hardware limit, not
an omission**. TinyTitan's ANE path is a Core ML sidecar that computes a
full-attention block's prefill (`docs/v4.5-ane-prefill.md`); the ANE is
programmed by exporting a fixed graph, not by launching arbitrary compute.
Nemotron 3.5 Bolt needs three things Core ML cannot express here:

1. a per-token **selective-scan recurrence** over a 128×64×128 FP32 state in
   each of 40 layers — the ANE has no such op and the model is not a static
   graph;
2. **dynamic gather of 9 of 240–496 experts per layer** from an 80 GB blob —
   there is no ANE op for an indexed expert read, and a dense surrogate would
   be 74B parameters per token;
3. the routed experts do not fit in 24 GB of unified memory even at 4 bits.

A decode column is therefore reported as **NA with that reason**; it is not
replaced by a prefill-assisted number that would measure the GPU anyway.
