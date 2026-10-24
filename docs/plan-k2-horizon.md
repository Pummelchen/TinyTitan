# Plan: K2-Horizon-MoVA-36B-A4B

Status: **research complete; runtime not implemented.** This is not an
"adding a model" wiring job in the sense of `adding-a-model.md` — the
architecture is new. The checkpoint is `IFM/K2-Horizon-MoVA-36B-A4B`,
Apache-2.0, ungated, at sha `e9ce717642b2074082875be233c02c729ba47960`
(2026-10-14).

Everything below was read from the source at that sha: `config.json`,
`configuration_k2_horizon.py`, `modeling_k2_horizon.py`, the safetensors
index, `generation_config.json`, `tokenizer_config.json` and
`chat_template.jinja`. None of it has been run.

## The checkpoint

| | |
| --- | --- |
| Architecture | `K2HorizonForCausalLM`, `model_type: k2_horizon` |
| Size | 36B total, 4B active per token, 74.9 GB in 48 safetensors shards |
| Tensors | 16,998 in the index |
| Layers | 48; hidden 2,560 |
| Attention | 32 query heads / 8 KV heads, head dim 128, full RoPE on all 128 dims, theta 1e7 |
| Context | 524,288 native |
| MoE | 100 routed experts, top-8, expert width 768, one shared expert |
| MoVA | 64 value experts in the attention of sparse layers, top-4, each hidden -> 1,024 |
| Norms | grouped RMS norm, **2 groups** of 1,280, eps 1e-6 |
| Gate | attention output gate `hidden -> 4,096`, **softplus(beta = ln 2)** per element |
| Vocabulary | 250,624, untied `lm_head` |
| Tokens | BOS `<\|ifm\|begin_of_text\|>`, EOS `<\|ifm\|endoftext\|>` (ids 1 and 250,019) |

The quantized installs this would produce are roughly 18-19 GB at 4-bit and
36-37 GB at 8-bit; the source is 74.9 GB and this host has 178 GB free.

## Progress

- **Planning** — `tools/prepare_k2_horizon.py --plan` classifies all 16,998
  tensors and measures the snapshots (21.7 GB at 4-bit, 39.8 GB at 8-bit).
- **Conversion (local data layer)** — `--source/--output/--bits` writes an
  affine snapshot with the checkpoint's names, bf16 keeps and 8-bit
  embedding/head; `--self-test` converts a tiny K2-shaped fixture and checks
  the metadata and a round trip.
- **Arch reader** — `RepackModelFamily.k2Horizon`, the K2 extension fields on
  `ArchInfo` (MoVA experts, grouped norm, gate function, router bias and
  scaling, dense layers) and `loadK2Horizon`, tested against the pinned
  config's values.

Not started: the repacker's tensor classification and layer planning for the
two expert systems, the runtime family and kernels, the template/parsers, and
the catalogue/golden wiring.

## The conversion plan, measured

`tools/prepare_k2_horizon.py --plan` fetches the config, the index and the
safetensors headers (no weights) and classifies all 16,998 tensors. It reports:

| | 4-bit | 8-bit |
| --- | ---: | ---: |
| MoE experts (13,500 tensors) | 14.93 GB | 28.20 GB |
| MoVA value experts (2,880) | 4.25 GB | 8.02 GB |
| attention (285) | 0.94 GB | 1.76 GB |
| embedding + untied head | 1.36 GB | 1.36 GB |
| MLP + shared experts (141) | 0.20 GB | 0.38 GB |
| routers, norms, biases kept bf16 | 38 MB | 38 MB |
| **snapshot** | **~21.7 GB** | **~39.8 GB** |

Every quantized last dimension is a multiple of the 64-element group; the
only tensors that would have failed are the 100-wide router biases, which the
plan keeps at bf16. The install TinyTitanRepack writes afterwards is a second
copy, so the peak disk for the 8-bit path is the source download plus ~40 GB
of snapshot plus ~40 GB of install.

## The converter

`tools/prepare_k2_horizon.py` has two working modes beyond `--plan`:

- `--source <dir> --output <dir> --bits 4 8` converts a **local** HF-style
  checkpoint into the affine snapshot layout. Names are preserved (the K2
  reader is written against the checkpoint's spelling, so nothing is renamed or
  fused), quantized matrices become packed `uint32` weights plus bf16
  `.scales`/`.biases`, and routers, norms and biases stay bf16.
- `--self-test` builds a tiny K2-shaped fixture (two layers, four experts, two
  value experts), converts it at 4-bit and checks the base width, the group
  size, the 8-bit embedding override, a quantize/dequantize round trip, and
  that the router bias is unchanged bf16.

The measured plan output and the self-test both pass; the download/streaming
path for the real 74.9 GB checkpoint is not written yet, and no repacker or
runtime reader exists for this snapshot.

## What a layer does

Layers 0-2 are dense; layers 3-47 are sparse.

**Dense layer (0-2).** `input_layernorm` (grouped RMS) -> standard GQA
attention with full RoPE and the softplus gate -> residual ->
`post_attention_layernorm` -> plain SwiGLU MLP (2,560 -> 6,144) -> residual.
Ten tensors.

**Sparse layer (3-47), 377 tensors.**

1. Grouped RMS norm.
2. **MoVA attention** instead of `v_proj`:
   - `v_router.weight` (2,560 -> 64) and `v_router.bias`;
   - 64 `v_experts.N` matrices, each 2,560 -> 1,024;
   - routing is sigmoid, the bias is added **only to the selection scores**,
     the chosen four weights are the sigmoid scores, normalized over the four
     and multiplied by `router_scaling_factor` 2.5;
   - each selected expert's output is passed through SiLU, weighted and
     summed to produce the value states;
   - `q_proj`, `k_proj`, `o_proj`, full RoPE, then the same softplus gate.
3. Residual, grouped RMS norm.
4. **Sparse MoE**: router `gate.weight` (2,560 -> 100) plus `gate.bias`, again
   sigmoid with the bias used only for selection, top-8, normalized and
   scaled by 2.5; each expert is SwiGLU 2,560 -> 768 -> 2,560; the
   `shared_experts` SwiGLU (width 768) is added to the routed sum.

## Why this is a runtime project

TinyTitan serves `qwen36`, `qwen38flash`, `qwen35Dense` (and their MTP heads).
Nothing matches `k2_horizon`: the layer count, expert counts, router bias and
scaling, two-group norm, softplus gate and — above all — the MoVA value
experts are all new. The pieces that do carry over are the quantized
group-64 affine format, the expert streaming machinery (after it is
generalized to 100 + 64 expert lists) and the scheduler.

The work breaks down as:

1. **Converter and arch reader.** A `tools/prepare_k2_horizon.py` that maps
   the 16,998 tensor names into the quantized layout, plus a
   `loadK2Horizon` branch in `ArchInfo+Loaders.swift` and a repacker test
   against the real `config.json`. This is the first phase and can be tested
   without the runtime by writing a small fixture checkpoint.
2. **Family and dense path.** `ModelFamily.k2Horizon`, schema validation,
   grouped RMS norm, the softplus gate and the dense layers 0-2. This gets a
   converted model loading and running its first three layers correctly
   against a PyTorch reference.
3. **MoE routing.** Generalize the router to sigmoid + selection bias +
   top-k normalization + scaling, and to two routed expert sets per layer.
4. **MoVA attention.** A prefill and a decode path for routed value experts,
   including the streaming layout: top-4 of 64 experts per layer is about
   10.5M parameters read per token, on top of the MoE's ~47M.
5. **Surfaces.** The 52 KB chat template with `reasoning_effort` and
   `tool_call_format` (json/xml/xml_typed), the `k2_horizon` reasoning and
   tool-call parsers, catalogue entry, launcher keys, docs and a golden
   baseline. Recommended sampling is `temperature=1.0`, `top_p=0.95`,
   reasoning effort `high`; `frequency_penalty` and `min_p` are not
   implemented in this runtime and would have to be named as deviations.

Phases 3 and 4 are the expensive ones; the expert streaming layout is the
part most likely to force a change in the existing MoE kernel interface.

## Verification plan

- Logit parity against the HF `modeling_k2_horizon.py` on short prompts, the
  same way the dense CPU engine is checked against its numpy reference.
- Greedy continuation equality at 4-bit and 8-bit for a fixed prompt, then
  the release golden mechanism.
- A memory check: the 4-bit install plus KV cache is already ~19 GB; 512K
  context is the number to test, not to assume.

## Risks

- **MoVA is the unknown.** It is a second routed expert system inside
  attention, and the checkpoint's own code loops per expert; a batched,
  streamed implementation is new kernel work, not a parameter change.
- **Per-token expert traffic.** Top-4 MoVA + top-8 MoE reads about 58M
  parameters per token before attention; at 4-bit that is ~29 MB per token
  from disk unless the expert cache holds it.
- **Template and parsers.** The template is 52 KB and carries reasoning and
  three tool formats; the runtime's renderer and parsers would need a new
  dialect, not a reuse of Qwen's.
- **No in-tree oracle.** Unlike the Qwen families there is no reference
  implementation in this repository; the HF code is the only one, so the
  first run needs PyTorch installed and a reference logit dump.

## Next step

Phase 1: the converter and arch reader, plus a fixture checkpoint that
round-trips a handful of tensors. That is a bounded piece of work and the
only phase that does not depend on the runtime kernels. The 74.9 GB download
should wait until phase 2 can load the result, unless the disk is needed for
the fixture alone.
