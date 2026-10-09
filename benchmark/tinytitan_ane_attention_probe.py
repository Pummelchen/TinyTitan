#!/usr/bin/env python3
"""Track A probe: the full-attention prefill block on the Neural Engine.

Builds the complete Qwen3.5-MoE full-attention block — packed QKV projection
with output gate, per-head q/k RMS norms, NeoX-subdim RoPE (64 of 256), GQA
SDPA (16 query heads over 2 KV heads) against a KV history, sigmoid output
gate, and the O projection — as one Core ML MIL program at the real shapes,
runs it with CPU_AND_NE and CPU_ONLY, and reports per-chunk latency.

Reference GPU numbers (measured this session, 6,103-token prefill, 4-bit,
TINYTITAN_KERNEL_STATS): the 10 full-attention layers cost 84.3 s of the 133.2 s
prefill — 4.21 s per layer-chunk. The go/no-go: the ANE must beat that per
layer-chunk by enough to survive integration overheads.

That mean is an average over 20 layer-chunks of two *unequal* shapes (4096:0 then
2007:4096), so it is not the cost of any one chunk and cannot be compared against
a row measured at a different one. Both sides are therefore reported in ms per 1k
layer-tokens (84.3 s / (6,103 x 10) = 1,381 ms per 1k layer-tokens) and the ratio
is printed per row. A row's attention cost per token grows with its history (the
quadratic term), so a history-bearing row reads a *smaller* gain than the same ANE
shows on a cold chunk — measured at chunk 512 on this M3, 95.9x with no history
against 62.6x with 1,024 — while the reference average mixes both shapes. The run
says so rather than leaving the mismatch to the reader.

Weights are random fp16 at the real shapes (throughput does not depend on
values). Numerics are sanity-checked against a float32 NumPy reference of the
same math against `MAX_MEAN_REL_ERROR`, and a non-finite gap — the failure mode
the fused SDPA showed on this machine — fails the run. Measured on this M3 with
the default list: 0.0141 at 1024:0, 0.0204 at 2048:0, 0.0292 at 4096:0 and 0.0998
at 2048:4096; that growth is uniform-random scores at total length 6,144, which
`docs/v4.4-decode-width-plan.md` records collapsing to 0.0002 on realistic ones.
Exact parity with the Metal kernels' conventions is integration work, not probe
work.

The CPU_ONLY arm is a wall-clock cross-check, not the decision, and it is only run
up to `CPU_ONLY_MAX_CHUNK`; its absence is printed and recorded with the reason.

Exit is 2 for a refused `--configs` or `--repeats`, 1 when any config errored,
timed nothing, failed the numerics ceiling, or could not be written, and 0 only
when every config measured and passed.

  ~/.venvs/coreml-py311/bin/python benchmark/tinytitan_ane_attention_probe.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Qwen3.5-MoE 35B-A3B full-attention geometry (ArchConfig.qwen36_35B_A3B).
D = 2048
N_Q_HEADS = 16
N_KV_HEADS = 2
HEAD_DIM = 256
Q_DIM = N_Q_HEADS * HEAD_DIM  # 4096
Q_PROJ_ROWS = 2 * Q_DIM  # packed query+gate
KV_DIM = N_KV_HEADS * HEAD_DIM  # 512
ROTARY = 64  # headDim * partialRotaryFactor(0.25)
THETA = 10_000_000.0
SCALE = 0.0625  # 256^-0.5
EPS = 1e-6

# The recorded GPU measurement this probe is judged against, and the shapes it
# averages: 84.3 s over the 10 full-attention layers of a 6,103-token prefill
# chunked 4096 then 2007, which is 20 layer-chunks.
GPU_LAYER_SECONDS = 84.3
GPU_PREFILL_TOKENS = 6103
FULL_ATTENTION_LAYERS = 10
GPU_CHUNK_SHAPES = "4096:0 then 2007:4096"
GPU_MS_PER_LAYER_CHUNK = GPU_LAYER_SECONDS * 1000 / (FULL_ATTENTION_LAYERS * 2)

WARM_UPS = 2
CPU_WARM_UPS = 1
CPU_REPEATS = 3
# Above this chunk the CPU_ONLY arm is not run. It is a cross-check that the ANE
# number is not simply CPU time, and a 4,096-token CPU pass costs minutes for no
# extra assurance. Measured here: 102.74 ms at 2048:0, 1.95x-2.37x at 1024-2048.
CPU_ONLY_MAX_CHUNK = 2048
# A garbage-detection bar, not a precision promise: the measured errors in the
# docstring sit under it, and the plan's realistic-distribution error is two
# orders lower.
MAX_MEAN_REL_ERROR = 0.15

ARTIFACT = "ane-attention-probe.json"
OUT_ENV = "TINYTITAN_ANE_PROBE_OUT"
DEFAULT_OUT = ROOT / ".build" / "benchmark-results" / ARTIFACT


def rope_tables(start: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    half = ROTARY // 2
    inv = THETA ** (-np.arange(half, dtype=np.float64) * 2 / ROTARY)
    pos = np.arange(start, start + count, dtype=np.float64)[:, None] * inv[None, :]
    return (np.cos(pos).astype(np.float16), np.sin(pos).astype(np.float16))


def make_weights(rng: np.random.Generator) -> dict[str, np.ndarray]:
    def w(rows, cols):
        return (rng.standard_normal((rows, cols)) * 0.02).astype(np.float16)

    return {
        "wq": w(Q_PROJ_ROWS, D),
        "wk": w(KV_DIM, D),
        "wv": w(KV_DIM, D),
        "wo": w(D, Q_DIM),
        "q_norm": np.abs(rng.standard_normal(HEAD_DIM) * 0.1 + 1).astype(np.float16),
        "k_norm": np.abs(rng.standard_normal(HEAD_DIM) * 0.1 + 1).astype(np.float16),
    }


def build_block(t: int, history: int, weights: dict[str, np.ndarray]):
    """One full-attention block: hidden [t, D] + K/V history -> output [t, D]
    plus the chunk's rotated K and raw V for the cache write.

    history == 0 builds a variant without history inputs: Core ML rejects
    zero-length tensor dimensions on model inputs."""
    total = history + t
    fp16 = ct.converters.mil.mil.types.fp16
    specs = [mb.TensorSpec(shape=(t, D), dtype=fp16)]
    if history > 0:
        specs += [
            mb.TensorSpec(shape=(1, N_KV_HEADS, history, HEAD_DIM), dtype=fp16),
            mb.TensorSpec(shape=(1, N_KV_HEADS, history, HEAD_DIM), dtype=fp16),
        ]
    specs += [
        mb.TensorSpec(shape=(t, ROTARY // 2), dtype=fp16),
        mb.TensorSpec(shape=(t, ROTARY // 2), dtype=fp16),
        mb.TensorSpec(shape=(1, 1, t, total), dtype=fp16),
    ]

    def body(hidden, k_hist, v_hist, cos_t, sin_t, mask):
        def rms_head(x, weight_name, heads):
            # x: [heads, seq, HEAD_DIM] per-head RMS norm with weight.
            sq = mb.mul(x=x, y=x)
            mean = mb.reduce_mean(x=sq, axes=[-1], keep_dims=True)
            denom = mb.rsqrt(x=mb.add(x=mean, y=np.float16(EPS)))
            return mb.mul(x=mb.mul(x=x, y=denom), y=weights[weight_name].reshape(1, 1, HEAD_DIM))

        def rope(x, heads, seq):
            # NeoX half-split on the first ROTARY dims; passthrough beyond.
            r1 = mb.slice_by_index(
                x=x,
                begin=[0, 0, 0],
                end=[heads, seq, ROTARY // 2],
                begin_mask=[True, True, False],
                end_mask=[True, True, False],
            )
            r2 = mb.slice_by_index(
                x=x,
                begin=[0, 0, ROTARY // 2],
                end=[heads, seq, ROTARY],
                begin_mask=[True, True, False],
                end_mask=[True, True, False],
            )
            rest = mb.slice_by_index(
                x=x,
                begin=[0, 0, ROTARY],
                end=[heads, seq, HEAD_DIM],
                begin_mask=[True, True, False],
                end_mask=[True, True, True],
            )
            cos_b = mb.reshape(x=cos_t, shape=[1, seq, ROTARY // 2])
            sin_b = mb.reshape(x=sin_t, shape=[1, seq, ROTARY // 2])
            o1 = mb.sub(x=mb.mul(x=r1, y=cos_b), y=mb.mul(x=r2, y=sin_b))
            o2 = mb.add(x=mb.mul(x=r2, y=cos_b), y=mb.mul(x=r1, y=sin_b))
            return mb.concat(values=[o1, o2, rest], axis=-1)

        # Projections: one matmul each, weights transposed at build time.
        packed = mb.matmul(x=hidden, y=weights["wq"].T)  # [t, 8192]
        k = mb.matmul(x=hidden, y=weights["wk"].T)  # [t, 512]
        v = mb.matmul(x=hidden, y=weights["wv"].T)  # [t, 512]

        # Split packed query+gate: per head, first half query, second gate.
        packed_h = mb.reshape(x=packed, shape=[t, N_Q_HEADS, 2 * HEAD_DIM])
        q = mb.slice_by_index(
            x=packed_h,
            begin=[0, 0, 0],
            end=[t, N_Q_HEADS, HEAD_DIM],
            begin_mask=[True, True, False],
            end_mask=[True, True, False],
        )
        gate = mb.slice_by_index(
            x=packed_h,
            begin=[0, 0, HEAD_DIM],
            end=[t, N_Q_HEADS, 2 * HEAD_DIM],
            begin_mask=[True, True, False],
            end_mask=[True, True, True],
        )

        q = mb.transpose(x=q, perm=[1, 0, 2])  # [16, t, 256]
        k_h = mb.transpose(
            x=mb.reshape(x=k, shape=[t, N_KV_HEADS, HEAD_DIM]), perm=[1, 0, 2]
        )  # [2, t, 256]
        v_h = mb.transpose(x=mb.reshape(x=v, shape=[t, N_KV_HEADS, HEAD_DIM]), perm=[1, 0, 2])

        q = rms_head(q, "q_norm", N_Q_HEADS)
        k_h = rms_head(k_h, "k_norm", N_KV_HEADS)
        q = rope(q, N_Q_HEADS, t)
        k_h = rope(k_h, N_KV_HEADS, t)

        k_new = mb.reshape(x=k_h, shape=[1, N_KV_HEADS, t, HEAD_DIM])
        v_new = mb.reshape(x=v_h, shape=[1, N_KV_HEADS, t, HEAD_DIM])
        if history > 0:
            k_all = mb.concat(values=[k_hist, k_new], axis=2)  # [1,2,total,256]
            v_all = mb.concat(values=[v_hist, v_new], axis=2)
        else:
            k_all, v_all = k_new, v_new

        # GQA: query head h reads KV head h // 8 — expand each KV head into
        # a contiguous block of 8, matching np.repeat on the head axis.
        rep = N_Q_HEADS // N_KV_HEADS

        def gqa_expand(x):
            x5 = mb.reshape(x=x, shape=[1, N_KV_HEADS, 1, total, HEAD_DIM])
            x5 = mb.concat(values=[x5] * rep, axis=2)
            return mb.reshape(x=x5, shape=[1, N_Q_HEADS, total, HEAD_DIM])

        k_g = gqa_expand(k_all)
        v_g = gqa_expand(v_all)

        # Decomposed attention, deliberately NOT the fused
        # scaled_dot_product_attention op: on this M3/macOS the fused op
        # produces NaN/inf on the ANE from sequence length 2048 even at tame
        # score scales (std 0.25), while matmul+softmax+matmul is clean and
        # slightly faster (isolated A/B: rel err inf vs 0.007 at 2048,
        # 58.4 vs 50.0 ms). The explicit scale matches attentionScale.
        q4 = mb.reshape(x=q, shape=[1, N_Q_HEADS, t, HEAD_DIM])
        scores = mb.matmul(x=q4, y=k_g, transpose_y=True)
        scores = mb.mul(x=scores, y=np.float16(SCALE))
        scores = mb.add(x=scores, y=mask)
        probs = mb.softmax(x=scores, axis=-1)
        attn = mb.matmul(x=probs, y=v_g)  # [1,16,t,256]

        gated = mb.mul(
            x=mb.transpose(x=attn, perm=[0, 2, 1, 3]),
            y=mb.sigmoid(x=mb.reshape(x=gate, shape=[1, t, N_Q_HEADS, HEAD_DIM])),
        )
        merged = mb.reshape(x=gated, shape=[t, Q_DIM])
        out = mb.matmul(x=merged, y=weights["wo"].T)  # [t, D]
        return out, k_new, v_new

    if history > 0:

        @mb.program(input_specs=specs, opset_version=ct.target.iOS18)
        def prog(hidden, k_hist, v_hist, cos_t, sin_t, mask):
            return body(hidden, k_hist, v_hist, cos_t, sin_t, mask)
    else:

        @mb.program(input_specs=specs, opset_version=ct.target.iOS18)
        def prog(hidden, cos_t, sin_t, mask):
            return body(hidden, None, None, cos_t, sin_t, mask)

    return ct.convert(
        prog,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )


def causal_mask(t: int, history: int) -> np.ndarray:
    total = history + t
    mask = np.zeros((1, 1, t, total), dtype=np.float16)
    for i in range(t):
        mask[0, 0, i, history + i + 1 :] = np.float16(-np.inf)
    return mask


def reference(hidden, k_hist, v_hist, cos_t, sin_t, mask, w):
    """Float32 NumPy of the same math, for a numerics sanity check."""
    h = hidden.astype(np.float32)
    packed = h @ w["wq"].T.astype(np.float32)
    k = h @ w["wk"].T.astype(np.float32)
    v = h @ w["wv"].T.astype(np.float32)
    t = h.shape[0]
    packed = packed.reshape(t, N_Q_HEADS, 2 * HEAD_DIM)
    q, gate = packed[..., :HEAD_DIM], packed[..., HEAD_DIM:]

    def rms(x, weight):
        return x / np.sqrt((x**2).mean(-1, keepdims=True) + EPS) * weight.astype(np.float32)

    def rope(x, cos_t, sin_t):
        r1, r2 = x[..., : ROTARY // 2], x[..., ROTARY // 2 : ROTARY]
        c = cos_t.astype(np.float32)[:, None, :]
        s = sin_t.astype(np.float32)[:, None, :]
        return np.concatenate([r1 * c - r2 * s, r2 * c + r1 * s, x[..., ROTARY:]], -1)

    q = rope(rms(q, w["q_norm"]), cos_t, sin_t).transpose(1, 0, 2)
    kh = k.reshape(t, N_KV_HEADS, HEAD_DIM)
    kh = rope(rms(kh, w["k_norm"]), cos_t, sin_t).transpose(1, 0, 2)
    vh = v.reshape(t, N_KV_HEADS, HEAD_DIM).transpose(1, 0, 2)
    k_all = np.concatenate([k_hist[0].astype(np.float32), kh], 1)
    v_all = np.concatenate([v_hist[0].astype(np.float32), vh], 1)
    rep = N_Q_HEADS // N_KV_HEADS
    k_g = np.repeat(k_all, rep, axis=0)
    v_g = np.repeat(v_all, rep, axis=0)
    scores = q * SCALE @ k_g.transpose(0, 2, 1) + mask[0, 0].astype(np.float32)
    p = np.exp(scores - scores.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    attn = (p @ v_g).transpose(1, 0, 2)
    out = (attn * (1 / (1 + np.exp(-gate)))).reshape(t, Q_DIM)
    return out @ w["wo"].T.astype(np.float32)


class ConfigError(ValueError):
    """A configuration the reader can fix without reading a traceback."""


def parse_configs(spec: str) -> list[tuple[int, int]]:
    """`(chunk, history)` pairs, or a refusal naming the flag and the entry."""
    if not spec.strip():
        raise ConfigError(f"--configs {spec!r} is empty; it takes CHUNK:HISTORY pairs")
    configs = []
    for entry in spec.split(","):
        parts = entry.split(":")
        if len(parts) != 2:
            raise ConfigError(f"--configs {spec!r}: entry {entry!r} is not CHUNK:HISTORY")
        try:
            chunk, history = (int(value) for value in parts)
        except ValueError:
            raise ConfigError(
                f"--configs {spec!r}: entry {entry!r} holds a token count that is not a number"
            ) from None
        if chunk < 1:
            raise ConfigError(
                f"--configs {spec!r}: entry {entry!r} has a chunk of {chunk}; a chunk is at least one token"
            )
        if history < 0:
            raise ConfigError(
                f"--configs {spec!r}: entry {entry!r} has history of {history}; the first chunk is 0"
            )
        configs.append((chunk, history))
    return configs


def parse_repeats(value: int) -> int:
    if value < 1:
        raise ConfigError(
            f"--repeats {value} times nothing; the median needs at least one timed run"
        )
    return value


def timed_samples(samples: list[float], warm_ups: int) -> list[float]:
    """The timed tail of a run, with its warm-ups dropped."""
    return samples[warm_ups:]


def median_ms(samples: list[float]) -> float:
    """The upper middle, which is what the shipped numbers were read from."""
    return sorted(samples)[len(samples) // 2]


def ms_per_1k_tokens(ms: float, tokens: int) -> float:
    return ms * 1000 / tokens


def ms_per_1k_layer_tokens() -> float:
    """The GPU reference in the same unit every row is given in."""
    layer_tokens = GPU_PREFILL_TOKENS * FULL_ATTENTION_LAYERS
    return ms_per_1k_tokens(GPU_LAYER_SECONDS * 1000, layer_tokens)


def speedup_over_gpu(ms: float, tokens: int) -> float:
    return ms_per_1k_layer_tokens() / ms_per_1k_tokens(ms, tokens)


def error_within_ceiling(rel) -> bool:
    """A non-finite gap is never within the ceiling, whatever the ceiling is."""
    return rel is not None and math.isfinite(rel) and rel <= MAX_MEAN_REL_ERROR


def cpu_model_for(spec, weights_dir):
    return ct.models.MLModel(spec, weights_dir=weights_dir, compute_units=ct.ComputeUnit.CPU_ONLY)


def block_output_name(model, chunk: int) -> str:
    """The block output, found by shape rather than by the name the converter chose.

    The generated names are positional, so the shape is the only stable handle on
    the tensor the timing and the numerics are about; the k/v cache outputs are the
    same program's other two.
    """
    wanted = (chunk, D)
    outputs = model.get_spec().description.output
    for entry in outputs:
        if tuple(entry.type.multiArrayType.shape) == wanted:
            return entry.name
    names = [entry.name for entry in outputs]
    raise ValueError(f"no output of shape {wanted} among {names}")


def timed_predictions(model, feed, repeats: int, warm_ups: int):
    samples = []
    out = None
    for _ in range(repeats + warm_ups):
        start = time.perf_counter()
        out = model.predict(feed)
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples, out


def measure_config(chunk, history, repeats, weights, rng, build, cpu_build) -> dict:
    """One `(chunk, history)` config: the row of what was actually measured.

    Everything the run cannot report is written into the row rather than raised,
    because the artifact is what the go/no-go is read from and a row that says
    `null` cannot tell a skipped arm from an errored one.
    """
    row = {
        "chunk": chunk,
        "history": history,
        "cpu_and_ne_ms": None,
        "cpu_only_ms": None,
        "cpu_only_status": "skipped",
        "cpu_only_note": "",
        "mean_rel_error": None,
        "error": None,
    }
    try:
        model = build(chunk, history, weights)
        hidden = (rng.standard_normal((chunk, D)) * 0.5).astype(np.float16)
        k_hist = (rng.standard_normal((1, N_KV_HEADS, history, HEAD_DIM)) * 0.5).astype(np.float16)
        v_hist = (rng.standard_normal((1, N_KV_HEADS, history, HEAD_DIM)) * 0.5).astype(np.float16)
        cos_t, sin_t = rope_tables(history, chunk)
        mask = causal_mask(chunk, history)
        feed = {"hidden": hidden, "cos_t": cos_t, "sin_t": sin_t, "mask": mask}
        if history > 0:
            feed.update(k_hist=k_hist, v_hist=v_hist)

        name = block_output_name(model, chunk)
        samples, out = timed_predictions(model, feed, repeats, WARM_UPS)
        row["cpu_and_ne_ms"] = median_ms(timed_samples(samples, WARM_UPS))

        got = np.asarray(out[name], dtype=np.float32)
        ref = reference(hidden, k_hist, v_hist, cos_t, sin_t, mask, weights)
        denom = np.abs(ref).mean()
        row["mean_rel_error"] = float(np.abs(got - ref).mean() / max(denom, 1e-9))

        if chunk > CPU_ONLY_MAX_CHUNK:
            row["cpu_only_note"] = f"chunk {chunk} is over CPU_ONLY_MAX_CHUNK={CPU_ONLY_MAX_CHUNK}"
        else:
            try:
                cpu = cpu_build(model.get_spec(), model.weights_dir)
                cpu_samples, _ = timed_predictions(cpu, feed, CPU_REPEATS, CPU_WARM_UPS)
                row["cpu_only_ms"] = median_ms(timed_samples(cpu_samples, CPU_WARM_UPS))
                row["cpu_only_status"] = "measured"
            except Exception as error:  # the cross-check failing is the row's to say
                row["cpu_only_status"] = "errored"
                row["cpu_only_note"] = f"{type(error).__name__}: {error}"
    except Exception as error:  # one config failing does not end the sweep
        row["error"] = f"{type(error).__name__}: {error}"
    return row


def row_status(row: dict) -> int:
    if row.get("error"):
        return 1
    if not error_within_ceiling(row.get("mean_rel_error")):
        return 1
    if row.get("cpu_only_status") == "errored":
        return 1
    return 0


def row_report(row: dict) -> tuple[list[str], int]:
    """The lines one config prints, and whether it costs the run."""
    status = row_status(row)
    if row.get("error"):
        return (
            [f"  NOT RUN: chunk {row['chunk']}, history {row['history']} -- {row['error']}"],
            status,
        )

    lines = []
    ms = row["cpu_and_ne_ms"]
    rate = ms_per_1k_tokens(ms, row["chunk"])
    gain = speedup_over_gpu(ms, row["chunk"])
    lines.append(
        f"  CPU_AND_NE {ms:9.2f} ms/chunk-layer   {rate:.1f} ms per 1k tokens"
        f"   {gain:.1f}x the GPU reference"
    )

    if row.get("cpu_only_ms") is not None:
        cpu_ms = row["cpu_only_ms"]
        lines.append(
            f"  CPU_ONLY     {cpu_ms:9.2f} ms   ratio {cpu_ms / ms:.2f}x against CPU_AND_NE"
        )
    elif row.get("cpu_only_status") == "errored":
        lines.append(
            f"  CPU_ONLY errored -- {row.get('cpu_only_note')}; "
            "the wall-clock cross-check did not run"
        )
    else:
        reason = (
            row.get("cpu_only_note")
            or f"chunk {row['chunk']} is over CPU_ONLY_MAX_CHUNK={CPU_ONLY_MAX_CHUNK}"
        )
        lines.append(
            f"  CPU_ONLY     not run: {reason} -- the arm is a wall-clock cross-check, not the go/no-go"
        )

    rel = row.get("mean_rel_error")
    if rel is None or not math.isfinite(rel):
        lines.append(
            "  mean rel err NOT MEASURED: the gap to the float32 reference is not a finite number"
            " -- the block output holds a non-finite value, the failure mode the fused SDPA showed"
        )
    elif rel > MAX_MEAN_REL_ERROR:
        lines.append(
            f"  mean rel err NOT MEASURED: {rel:.4f} is over the {MAX_MEAN_REL_ERROR} ceiling"
        )
    else:
        lines.append(f"  mean rel err {rel:.4f}   within the {MAX_MEAN_REL_ERROR} ceiling")
    return lines, status


def header_lines(configs, repeats: int) -> list[str]:
    return [
        f"ANE full-attention probe -- configs: {', '.join(f'{t}:{h}' for t, h in configs)}, "
        f"repeats: {repeats} timed run(s) after {WARM_UPS} warm-up(s)",
        f"weights: random fp16 at the real shapes; numpy {np.__version__}, "
        f"coremltools {getattr(ct, '__version__', 'unknown')}",
        f"CPU_ONLY arm runs for chunks up to {CPU_ONLY_MAX_CHUNK}; "
        f"numerics ceiling {MAX_MEAN_REL_ERROR}",
    ]


def reference_lines() -> list[str]:
    return [
        "",
        f"GPU reference (measured, 4-bit, this machine): the {FULL_ATTENTION_LAYERS} "
        f"full-attention layers cost {GPU_LAYER_SECONDS} s of a {GPU_PREFILL_TOKENS:,}-token "
        f"prefill chunked {GPU_CHUNK_SHAPES} = {GPU_MS_PER_LAYER_CHUNK:,.0f} ms per layer-chunk "
        f"averaged over {FULL_ATTENTION_LAYERS * 2} layer-chunks.",
        f"That mean spans two unequal chunk shapes, so no row above is comparable to it as "
        f"printed; both sides are given in ms per 1k layer-tokens, where the reference is "
        f"{ms_per_1k_layer_tokens():,.1f}.",
        "A row's attention cost per token grows with its history (the quadratic term), so a "
        "history-bearing row reads a smaller gain than the same ANE shows on a cold chunk, while "
        "the reference average mixes both shapes -- no row above is a like-for-like comparison "
        "of it.",
    ]


def out_path(env=None) -> pathlib.Path:
    """Where the artifact goes: the variable names a file, or a directory to fill."""
    mapping = os.environ if env is None else env
    value = str(mapping.get(OUT_ENV, "")).strip()
    if not value:
        return DEFAULT_OUT
    path = pathlib.Path(value)
    return path / ARTIFACT if path.is_dir() else path


def write_artifact(rows, path: pathlib.Path) -> None:
    """Strict JSON: a non-finite measurement is recorded as null, never as NaN."""
    clean = []
    for row in rows:
        item = dict(row)
        rel = item.get("mean_rel_error")
        if rel is not None and not math.isfinite(rel):
            item["mean_rel_error"] = None
            item["mean_rel_error_note"] = "the gap to the reference is not a finite number"
        clean.append(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs", default="1024:0,2048:0,4096:0,2048:4096", help="comma list of chunk:history"
    )
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    try:
        configs = parse_configs(args.configs)
        repeats = parse_repeats(args.repeats)
    except ConfigError as error:
        print(f"REFUSED: {error}")
        return 2

    rng = np.random.default_rng(41)
    weights = make_weights(rng)
    for line in header_lines(configs, repeats):
        print(line, flush=True)

    rows, statuses = [], []
    for chunk, history in configs:
        print(f"== chunk {chunk}, history {history} ==", flush=True)
        row = measure_config(chunk, history, repeats, weights, rng, build_block, cpu_model_for)
        lines, status = row_report(row)
        rows.append(row)
        statuses.append(status)
        for line in lines:
            print(line, flush=True)

    for line in reference_lines():
        print(line, flush=True)

    path = out_path()
    try:
        write_artifact(rows, path)
    except OSError as error:
        print(f"NOT WRITTEN: {path} -- {type(error).__name__}: {error}")
        statuses.append(1)
    else:
        print(f"wrote {path}")

    failed = sum(1 for value in statuses if value)
    if failed:
        print(
            f"PROBE INCOMPLETE: {failed} of {len(statuses)} check(s) failed -- "
            "the go/no-go cannot be read off this run."
        )
        return 1
    print(
        f"PROBE COMPLETE: {len(configs)} config(s) measured and within the "
        f"{MAX_MEAN_REL_ERROR} numerics ceiling."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
