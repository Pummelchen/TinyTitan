# CPU prefill GEMM microbenchmark

Measured 2026-10-24 on the M3 / 24 GB host: would batching prompt tokens into
GEMMs make a CPU-only dense Qwen 3.5 prefill materially faster than today's
per-token decode loop?

This is a shape microbenchmark, not an engine path. The harness is
`benchmark/cpu_prefill_gemm.c`; it compares Accelerate's `cblas_sgemm` (the
public path that dispatches to AMX on M1–M3 and SME on M4+) with a hand-written
4×4 FP32 NEON kernel. FP32 is used for both arms so the comparison is about the
matrix engine, not quantization.

## Method

- Prompt length M = 1024 tokens; results are projected linearly to 2K/8K.
- Per layer: QKV (`hidden × (q + 2·kv)`), output projection (`q × hidden`),
  gate+up (`hidden × 2·intermediate`), down (`intermediate × hidden`).
- Shapes read from each install's `tokenizer/config.json`: 2B is 2048 hidden,
  24 layers, 6144 intermediate; 4B is 2560 hidden, 32 layers, 9216
  intermediate.
- Median of three timed runs after warm-up; max |BLAS − NEON| printed.
- “Today” is the README's CPU decode rate for the 4-bit install (2B 17.42
  tok/s, 4B 7.54), because the CPU engine currently pre-fills by decoding each
  prompt token with its logits discarded.

## Results

| Model | Arm | Layer GF/s | 2K linear prefill | 8K linear prefill |
| --- | --- | ---: | ---: | ---: |
| Qwen 3.5 2B 4-bit | Accelerate BLAS | ~1.18 TF/s | **4.02 s** (29× vs today) | **16.08 s** (29×) |
| Qwen 3.5 2B 4-bit | NEON 4×4 | 8.9 GF/s | 530.6 s (0.2×) | 2,122.6 s (0.2×) |
| Qwen 3.5 4B 4-bit | Accelerate BLAS | ~1.17 TF/s | **10.88 s** (25× vs today) | **43.52 s** (25×) |
| Qwen 3.5 4B 4-bit | NEON 4×4 | 8.6 GF/s | 1,482.8 s (0.2×) | 5,931.1 s (0.2×) |

“Today” for the same prompts is ~117.6 s / 470.3 s (2B) and ~271.6 s /
1,086.5 s (4B). BLAS and NEON agreed exactly on the harness inputs.

## What the numbers do and do not cover

- Only the linear projections and FFN are counted. Attention (quadratic),
  norms, the final head, sampling and KV work are excluded, so the projected
  prefill is a lower bound.
- M=1024 is projected linearly. GEMM efficiency usually improves with M, so
  the projection is conservative for 2K/8K.
- Accelerate chooses its own internal kernel; the 1.2 TF/s figure is what the
  public API delivers, not a claim about the hidden AMX encoding.
- No engine integration, quantization or quality check has been done. A
  batched CPU prefill must reproduce the current greedy output (or stay within
  the measured logit noise floor with zero top-1 divergence) before it ships.

## Recommendation, after the quantized follow-up

Accelerate wins by ~25–29× over today's per-token CPU prefill on the linear
layers, far above any plausible margin. That ratio is linear layers against the
*whole* per-token cost, though, so it is not a TTFT ratio: the non-linear share
of the token stays in place, and the shipping weights are 4/8-bit, not the FP32
this harness measured.

`cpu-prefill-quant-gemm.md` is the follow-up measurement: with the real
quantized kernels and the dequantization charged to the batched arm, the
projections are 10–14× faster and the real CPU prefill of the installed 2B is
47.4 ms/token, so batching every dense projection lands around 5–10× for TTFT
rather than 25×. The next step is a batched CPU prefill behind an opt-in flag,
using Accelerate for the GEMM with the existing NEON/serial path as fallback;
decode stays untouched. Raw AMX is not needed: Accelerate already reaches the
matrix hardware, and the M4+ path should use standard SME rather than
undocumented `.word` encodings.
