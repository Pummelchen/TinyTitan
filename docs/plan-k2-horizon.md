# Plan: K2-Horizon-MoVA-36B-A4B

Status: **install format and runtime identity complete; the GPU graph is not
implemented.** This is not an "adding a model" wiring job in the sense of
`adding-a-model.md` — the architecture is new. The checkpoint is
`IFM/K2-Horizon-MoVA-36B-A4B`, Apache-2.0, ungated, at sha
`e9ce717642b2074082875be233c02c729ba47960` (2026-10-14).

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

- **Runtime** — `K2Snapshot` + `K2Model` run the text path on the CPU:
  grouped RMS norm, full RoPE, GQA with the softplus gate, dense SwiGLU for
  layers 0-2, sparse MoE (sigmoid router with selection bias, top-8,
  normalised and scaled by 2.5, plus the shared expert), and MoVA values
  (top-4 of 64 experts, SiLU before weighting). `TinyTitanBench k2` drives it.
  Checked against an independent numpy implementation of the same snapshot:
  per-layer hidden states and logits agree to about 1e-6 on a tiny fixture.
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
- **Repacking and installs** — the two expert systems are laid out (the 100
  MoE experts fused into one rank-3 tensor per layer and role, the MoVA value
  experts resident), and both installs are written and verified:
  `models/k2-horizon-4bit.ssdai` (21.8 GB) and
  `models/k2-horizon-8bit.ssdai` (39.9 GB), 53 files each.
- **Runtime identity and structural validation** — `ModelFamily.k2Horizon`
  exists, so `ManifestReader.peekIdentity` reports the family instead of
  falling through to `qwen36`. `ManifestArch`/`ArchConfig` decode the K2
  fields from `manifest.arch` (`normGroups`, `attentionGate`,
  `routerHasBias`, `routerScaling`, the MoVA set, `denseLayerIndices`);
  `ArchConfig.resolved(forFamily: .k2Horizon)` requires them by name rather
  than defaulting, and `TensorSchema.k2Horizon` carries the checkpoint's
  tensor names. Manifest and layout validation now read per-layer expert
  counts, so the three dense layers are not demanded a packed-expert file
  that was never written. `Model.load` now returns a handle for both installs
  (full SHA-256, all resident tensors and the packed layout validated); the
  catalogue and `doctor` still skip the family, and `RealForwardRunner` refuses
  it before sizing a Qwen-shaped pipeline set, because the graph below does not
  exist.
- **Resident schema** — `validateK2Schema` checks the whole install: embedding
  and head, the grouped-norm weights, every layer's q/k/o and separate
  softplus-gate projection, the dense layers' `v_proj` and SwiGLU MLP, the
  sparse layers' bf16 MoE router and selection bias, the 64 MoVA value experts,
  the 768-wide shared expert and the packed routed-expert layout.
  `validateRoutedExpertLayout` now lets a declared dense layer be empty and
  refuses experts on one; the trusted-receipt layer check has the same
  allowance (both mixed-install gaps it closed were found by loading the real
  installs). Both builds load under full SHA-256 and stop at the runner's graph
  refusal, which is the current end state.
- **Synthetic install** — `K2ToySynthetic` builds a K2 `.ssdai` (toy geometry,
  real manifest/layout/index machinery) so the loader path is covered in CI:
  the default is three dense layers for the dense-graph tests, and
  `write(sparse: true)` adds a fourth sparse layer with bf16 routers, four
  MoVA value experts and a packed two-expert MoE blob. `K2ToyInstallTests`
  resolves the architecture, runs the resident schema (both variants), maps
  the weights, and checks that the runner refuses the family. It is the
  fixture the future graph tests will reuse.
- **Dense layer on the GPU** — `K2DenseLayer` runs one dense layer on Metal
  with a plain FP16 K/V cache: grouped norm, q/k/v and the separate gate
  projection, RoPE, GQA over the cache, the softplus gate, o_proj, the SwiGLU
  MLP and both residuals. Its two tests compare every output element against
  the CPU formulas on the synthetic install, at one token and at a second
  token that has to attend over the cached first. It is probe-sized caching,
  not the runner's paged/quantized manager; the same commit adds
  `Elementwise.encodeSiluMul` and the K2 gate/dense-MLP accessors.
- **Dense graph** — `K2DenseGraph` chains the pieces into a whole dense-only
  token: embedding, the layer stack (each layer with its own K/V cache), final
  norm and head. The synthetic install runs token 7 through all three layers,
  and a two-token run checks the second token against every layer's cache; both
  logit vectors match the chained CPU reference. Sparse layers are refused by
  name, so a real 36B install still stops before the graph runs.
- **Sparse layer** — `K2SparseLayerProbe` runs one sparse layer end to end:
  grouped norm, q/k and the separate gate, MoVA values from the four resident
  experts, cached gated attention, the bf16 MoE router, the fetched routed
  expert and the shared expert, with both residuals. Its test on the sparse toy
  compares every output element against the CPU formulas, dequantizing the
  routed expert from the same fetched bytes the layer used.
- **MoE stage** — `K2MoEStage` runs the sparse block's resident form: the bf16
  router GEMV (float logits), the shared selection kernel, then each selected
  expert's SwiGLU (separate gate/up GEMV, fused SiLU multiply, down GEMV) with
  the router weights summed by `k2_moe_accumulate_fp16`, and the always-on
  shared expert added first. The test builds four 4-bit experts at top-2 on
  synthetic buffers and matches the CPU formula, including the selected ids.
  `K2MoEStage.expertWeights(blob:entry:)` converts a fetched packed blob into
  the three per-role views from the layout's sub-tensor table; a second test
  fetches the sparse toy's real experts through the streaming cache and
  matches a reference dequantized from the same bytes.
- **Input and output ends** — `K2Head` covers the embedding lookup
  (`AffineQuantEmbeddingLookup` at the install's width, no `sqrt(hidden)`
  scale), the grouped final norm and the untied `lm_head` GEMV, with a test
  against the CPU formulas on the synthetic install.
- **GPU primitives** — the grouped two-way RMS norm is the existing
  `rmsnorm_bf16w_grouped` kernel with `headDim 1280, groupsPerRow 2`; no new
  norm work was needed. Two kernels are new and tested in isolation:
  `softplus_gate_mul_fp16` (K2's `softplus(x; beta = ln 2)` attention gate,
  with the overflow-safe branch) and `k2_router_topk_select_kn` (sigmoid
  scores, bias applied only to the ranking, top-k renormalization, `2.5`
  scaling; the same kernel serves the 100-expert MoE at top-8 and the 64-expert
  MoVA router at top-4). `K2RouterSelection` and
  `Elementwise.encodeSoftplusGateMul` are the Swift entry points. The router's
  `routerNormTopK` now defaults to true for K2 (older manifests, including both
  installed ones, omit the key) and the repacker writes it for new installs.
- **MoVA value stage** — `MoVAValueStage` runs the bf16 value-router GEMV
  (float logits; `bf16_gemv_float_simd` is new because rounding logits to half
  before the comparison could change the chosen experts), the selection kernel,
  four value GEMVs from the resident experts (4- or 8-bit), and `mova_combine_fp16`
  (SiLU before weighting). It is tested end to end against the CPU formula:
  8 experts at top-4, selection ids and a weighted value vector.
- **Serving defaults** — K2 is an effort family: the template accepts
  low|medium|high with `high` as its default, while Qwen3.8 keeps
  low|medium|xhigh. `ModelReasoningEffort` gained `high`, and each
  `ModelReasoningControl` now carries the subset its own template accepts, so
  a request for the other family's level is refused instead of reaching a
  template that raises. K2 is the effort-only case: its generation prompt
  always opens a think block (`<ifm|think>`, `_fast` or `_faster` by effort), so
  `supportsThinkingOff` is false and a thinking-off request is refused rather
  than silently thinking. Sampling follows the model card: temperature 1.0,
  top-p 0.95, top-k 20.
- **Tool-call parsing** — `K2ToolCallParser` reads the checkpoint's
  `<ifm|tool_calls>` wrapper in all three formats its template offers (`json`,
  `xml`, default `xml`, and `xml_typed`), including several calls in one
  wrapper. Values that the template writes as JSON literals become typed
  arguments; the type tag is informational. Tests cover the three grammars,
  multi-call blocks, unknown tools and malformed/oversized input.
- **Surfaces not wired** — the 52 KB template renders through the generic
  tokenizer (`TINYTITAN_K2_CHAT=1`), but the K2 decoder is not selected yet:
  `StructuredAssistantDecoder` still looks for ChatML's `<think>`/`<tool_call>`
  markers instead of K2's `<ifm|think>` / `<ifm|tool_call>`, so the parser has no
  caller. The catalogue entry and a golden baseline are also missing.

Next: the GPU family graph that calls these stages (dense layers first), the
in-graph wiring of the MoVA stage for decode and prefill, the K2 decoder
(markers plus the tool parser), the ANE prefill sidecar.

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

## Runtime path and benchmark

`TinyTitanBench k2 <snapshot> "<prompt>" [n]` loads the snapshot, tokenizes
through swift-transformers' generic `AutoTokenizer` (K2's tokenizer is a plain
BPE and does not carry Qwen's special tokens), and prints prefill time and
decode tokens per second. The snapshots were produced by
`prepare_k2_horizon.py --from-hf`, which downloads the 74.9 GB checkpoint once
and writes the 4-bit and 8-bit snapshots side by side.

Measured on this host (M3, 24 GB), greedy:

| Snapshot | Prompt | Prefill | Decode | Output |
| --- | ---: | ---: | ---: | --- |
| 4-bit (20 GB) | 5 tokens | 4.03 s | 2.46 tok/s | " Paris. The capital of France is Paris" |
| 4-bit (20 GB) | 31 tokens | 12.74 s (2.4 tok/s) | 4.09 tok/s (4 tokens) | " The quick brown fox" |
| 8-bit (37 GB) | 5 tokens | 14.11 s | 0.82 tok/s | " Paris. The capital of France is Paris" |
| 8-bit (37 GB) | 8 tokens | 16.65 s | 0.91 tok/s (16 tokens) | " little girl named Lily. She loved..." |

Both widths answer the capital question correctly and continue coherently.
The 8-bit snapshot is larger than RAM, so the page cache streams it from disk;
that is most of the difference. Prefill is token-by-token, so TTFT is roughly
prompt length times the per-token cost.

The checkpoint's own 52 KB chat template renders through swift-transformers'
generic tokenizer as well: `TINYTITAN_K2_CHAT=1` applies it and the model then
answers in its instruction format (a chat request for "What is the capital of
France?" tokenizes to 17 prompt tokens and decodes reasoning-style text).

### Serving path: ANE prefill + GPU decode

The CPU engine is only for the Qwen 3.5 dense 2B/4B/9B. Every other model runs
ANE prefill and GPU decode, and K2 is no exception: its 36B model cannot be
served from the CPU path, and the temporary CPU server wiring was removed --
there is no `K2ModelBackend` and no `k2_horizon` CPU family any more. The CPU
model described above stays as a numeric reference and a measurement harness
(`TinyTitanBench k2`), not as a serving path; the K2 installs under `models/`
are identified but not served today.

### Why the CPU reference is ~2-3 tok/s

The layer loop is 300-400 ms per token and the 250,624-row head is 15-30 ms;
the engine is not spending its time in the logits. A token reads about 2.25 GB
of active weights (attention, top-8 MoE experts, top-4 MoVA experts, shared
expert), and the synthetic CPU GEMV measures 43 GB/s for a resident matrix -
so with the model in RAM the same work would be ~50 ms/token, around 19 tok/s.
The 4-bit snapshot is 20 GB and this host has 25.8 GB with the desktop already
using several; `iostat` during decode shows the SSD doing 1.4-1.7 GB/s, and
swap is 7.7 GB used. Decode is therefore memory-bound: the model does not fit,
so experts stream from disk every token. The 8-bit snapshot (37 GB) is worse
(0.8-0.9 tok/s). A CPU-wide weight cache was tried and made it slower (it
competes with the page cache and the swap), so it was removed. This is the
measurement that says the CPU is the wrong engine, not a tuning target.

### Phase 1 landed: the install format

The repacker now produces a K2 `.ssdai` install. Two changes made it work:

- the snapshot is rewritten with the 100 MoE experts fused into one rank-3
  tensor per layer and role (`model.layers.N.mlp.switch_mlp.*`) by
  `tools/fuse_k2_experts.py`; the MoVA value experts stay resident tensors
  (top-4 of 64 is small enough not to stream), as do attention, the gate and
  the router;
- the layout format gained an optional per-layer `expertsPerLayer`, because
  K2's first three layers are dense and the other 45 carry 100 experts. The
  writer, the structural validator and `--verify-install` all read it, and an
  install without it is unchanged.

Both installs are written and verified: `models/k2-horizon-4bit.ssdai`
(21.8 GB) and `models/k2-horizon-8bit.ssdai` (39.9 GB), 53 files each,
`--verify-install` green. The 4-bit manifest carries the K2 extension fields in
`manifest.arch`: `movaNumExperts 64`, `movaTopKExperts 4`, `normGroups 2`,
`attentionGate softplus`, `routerScaling 2.5`, `routerHasBias true`,
`denseLayerIndices [0,1,2]`.

### The GPU/ANE port

1. **Install format.** The repacker's planner stores one fused expert stack per
   layer; K2 has two expert systems per sparse layer (100 MoE + 64 MoVA) plus a
   separate attention gate and value router. The layer/expert layout and the
   manifest need that shape before a GPU loader can exist.
2. **GPU family and loader.** `ModelFamily.k2Horizon`, schema validation and
   the per-layer graph on the existing Metal engine.
3. **Kernels.** Grouped two-way RMS norm, the softplus attention gate, the
   router's selection bias and 2.5 scaling, and - the new work - the MoVA value
   experts (top-4 of 64 linear experts, SiLU, weighted sum) in prefill and
   decode, with the same expert-streaming discipline as the MoE path.
4. **ANE prefill.** An ANE sidecar graph for the K2 layer shape, exported and
   verified by `tools/ane_sidecars.sh`; a GPU install without it has no ANE
   prefill.
5. **Surfaces.** The 52 KB template with `reasoning_effort` and three tool
   formats, the `k2_horizon` reason/tool parsers, catalogue keys, receipt,
   docs and goldens.

What this path does **not** have yet: the GPU family graph and kernels,
server/CLI/catalog serving, the `k2_horizon` reason/tool parsers, a golden
baseline, and a batched prefill. The installs under `models/` carry their
`verified-install.json`, and the bench command is still the only entry point
that runs K2 at all.

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

The measured plan output and the self-test both pass. `--from-hf` drives the
streaming 74.9 GB download and writes the two snapshots, and the repacker then
turns a snapshot into a `.ssdai` install (phase 1 above); the runtime reader
for that install's identity and layout exists, the graph that would execute it
does not.

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

Phase 2, the GPU family graph. The install is already loadable far enough to
prove its identity, architecture, resident schema and per-layer expert layout;
what does not exist is the layer graph that consumes them. The bounded first
slice is the dense path: `ModelFamily.k2Horizon` on the existing Metal runner,
a per-layer graph that runs layers 0-2 (grouped norm, GQA with the softplus
gate, SwiGLU) and refuses the sparse layers, checked against the CPU
`K2Snapshot`'s hidden states on a tiny fixture. The MoVA stage, the router's
selection bias/scaling and the ANE prefill follow, in that order, because each
one is a kernel surface the dense path does not need.

The 74.9 GB source download is no longer on the critical path: the 4-bit and
8-bit installs exist and are the runtime's input.

