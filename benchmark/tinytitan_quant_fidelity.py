#!/usr/bin/env python3
"""Is the community 4-bit quantization faithful to the official bf16 weights?

A full logit parity is impossible here (the bf16 checkpoint is ~360 GB against
288 GB free), but the question underneath it -- did this quantization preserve
the weights -- is answerable directly and cheaply: pull the same tensors from
both repos by HTTP range, dequantize the MLX affine blocks, and measure the
error against the official values.

Reference point: TinyTitan's own shipping 4-bit g64 models measure ~2% relative
error on attention tensors. Comparable error means sound; an order of
magnitude worse means broken.

Run it from a directory holding `off_index.json`, `rt_index.json` and
`rt_config.json` (the two `*_index.json` are a checkpoint's `weight_map`, the
`rt_config.json` its `quantization_config`). A target that cannot be read is
counted separately from a target that disagreed: the verdict names how many of
`TARGETS` were actually compared, and a run that compared none prints NOT
MEASURED and exits 1 rather than calling the build faithful.
"""

import json
import sys
import urllib.request
import numpy as np

OFF = "Qwen/Qwen3.8-Flash-Next"
MLX = "RockTalk/Qwen3.8-Flash-Next-MLX-4bit"
MLX_REV = "478474da92599ad0cf9f8bd447e658b29cb8480a"


def get(url, start=None, end=None):
    r = urllib.request.Request(url)
    if start is not None:
        r.add_header("Range", f"bytes={start}-{end}")
    with urllib.request.urlopen(r, timeout=180) as f:
        return f.read()


def url_for(repo, rev, shard):
    return f"https://huggingface.co/{repo}/resolve/{rev}/{shard}"


_hdr_cache = {}


def header(repo, rev, shard):
    key = (repo, shard)
    if key in _hdr_cache:
        return _hdr_cache[key]
    u = url_for(repo, rev, shard)
    n = int.from_bytes(get(u, 0, 7), "little")
    hdr = json.loads(get(u, 8, 8 + n - 1))
    _hdr_cache[key] = (hdr, 8 + n)
    return _hdr_cache[key]


def fetch(repo, rev, shard, name):
    hdr, base = header(repo, rev, shard)
    if name not in hdr:
        return None
    m = hdr[name]
    s, e = m["data_offsets"]
    raw = get(url_for(repo, rev, shard), base + s, base + e - 1)
    dt = {
        "BF16": np.uint16,
        "F16": np.float16,
        "F32": np.float32,
        "U32": np.uint32,
        "I32": np.int32,
        "U8": np.uint8,
    }[m["dtype"]]
    a = np.frombuffer(raw, dtype=dt)
    if m["dtype"] == "BF16":  # bf16 -> f32
        a = (a.astype(np.uint32) << 16).view(np.float32)
    return a.reshape(m["shape"])


def dequant(q, scales, biases, bits, group):
    """MLX affine: value = q * scale + bias, q packed little-endian into u32."""
    per = 32 // bits
    flat = q.reshape(-1)
    vals = np.empty((flat.size, per), dtype=np.float32)
    mask = (1 << bits) - 1
    for i in range(per):
        vals[:, i] = (flat >> (bits * i)) & mask
    out_rows = q.shape[0]
    vals = vals.reshape(out_rows, -1)  # [out, in]
    s = scales.astype(np.float32)
    b = biases.astype(np.float32)
    ng = s.shape[1]
    vals = vals.reshape(out_rows, ng, group)
    return (vals * s[:, :, None] + b[:, :, None]).reshape(out_rows, -1)


def verdict(checked: int, bad: int, skipped: int, total: int):
    """(lines, exit status) for a run that compared `checked` of `total` targets.

    `bad` counts tensors that were compared and disagreed, so it says nothing about
    a run that compared none: zero checked is the instrument unplugged, not a
    clean reading, and it has to be refused rather than printed as faithfulness.
    """
    if checked == 0:
        return (
            [
                f"NOT MEASURED: 0 of {total} target(s) were compared, "
                f"{skipped} unavailable -- the driver measured nothing"
            ],
            1,
        )
    lines = [
        f"VERDICT: {checked} of {total} tensor(s) compared, "
        + ("quantization is faithful" if bad == 0 else f"{bad} tensor(s) SUSPECT")
    ]
    if skipped:
        lines.append(f"({skipped} target(s) were unavailable and not compared)")
    return lines, 1 if bad else 0


TARGETS = [
    "model.language_model.layers.3.self_attn.q_proj.weight",
    "model.language_model.layers.3.self_attn.o_proj.weight",
    "model.language_model.layers.0.linear_attn.out_proj.weight",
    "model.language_model.layers.3.mlp.shared_expert.gate_proj.weight",
    "model.language_model.layers.3.mlp.gate.weight",
    "model.language_model.layers.19.self_attn.q_proj.weight",
]


def compare(name, off_idx, mlx_idx, qcfg):
    """(outcome, bits, err, max|w|, note) for one target.

    `outcome` is 'compared', 'shape' (dequantized against a reference of a
    different shape), or 'unavailable' with the reason in `note`. The distinction
    matters only because `bad` must count tensors that disagreed, never tensors
    that could not be read.
    """
    if name not in off_idx or name not in mlx_idx:
        return "unavailable", None, None, None, "missing"
    stem = name[: -len(".weight")]
    if stem + ".scales" not in mlx_idx or stem + ".biases" not in mlx_idx:
        return "unavailable", None, None, None, "no scales entry"
    bits = 4
    for key, value in qcfg.items():
        if isinstance(value, dict) and stem.endswith(key.split("model.language_model.")[-1]):
            bits = value["bits"]
    group = qcfg.get("group_size", 64)
    ref = fetch(OFF, "main", off_idx[name], name)
    quantised = fetch(MLX, MLX_REV, mlx_idx[name], name)
    scales = fetch(MLX, MLX_REV, mlx_idx[stem + ".scales"], stem + ".scales")
    biases = fetch(MLX, MLX_REV, mlx_idx[stem + ".biases"], stem + ".biases")
    if any(x is None for x in (ref, quantised, scales, biases)):
        return "unavailable", bits, None, None, "fetch fail"
    deq = dequant(quantised, scales, biases, bits, group)
    if deq.shape != ref.shape:
        return "shape", bits, None, None, f"SHAPE {deq.shape} vs {ref.shape}"
    err = float(np.linalg.norm(deq - ref) / max(np.linalg.norm(ref), 1e-9))
    return "compared", bits, err, float(np.abs(ref).max()), None


def main() -> int:
    try:
        off_idx = json.load(open("off_index.json", encoding="utf-8"))["weight_map"]
        mlx_idx = json.load(open("rt_index.json", encoding="utf-8"))["weight_map"]
        qcfg = json.load(open("rt_config.json", encoding="utf-8"))["quantization_config"]
    except OSError as exc:
        print(
            f"cannot read the index files ({exc.filename or exc}): run this from the "
            "directory holding off_index.json, rt_index.json and rt_config.json",
            file=sys.stderr,
        )
        return 2

    print(f"{'tensor':52s} {'bits':>4s} {'rel err':>9s} {'max|w|':>9s} {'verdict':>9s}")
    print("-" * 90)
    checked = bad = skipped = 0
    for name in TARGETS:
        outcome, bits, err, max_abs, note = compare(name, off_idx, mlx_idx, qcfg)
        label = name[-50:]
        if outcome == "unavailable":
            skipped += 1
            print(f"{label:52s} {'' if bits is None else f'{bits:>4d}'} {note:>9s}")
            continue
        if outcome == "shape":
            checked += 1
            bad += 1
            print(f"{label:52s} {bits:>4d} {note}")
            continue
        ok = err < (0.05 if bits == 4 else 0.02)
        checked += 1
        bad += 0 if ok else 1
        print(f"{label:52s} {bits:>4d} {err:9.4f} {max_abs:9.4f} {'OK' if ok else 'SUSPECT':>9s}")
    print("-" * 90)
    lines, status = verdict(checked, bad, skipped, len(TARGETS))
    for line in lines:
        print(line)
    return status


if __name__ == "__main__":
    sys.exit(main())
