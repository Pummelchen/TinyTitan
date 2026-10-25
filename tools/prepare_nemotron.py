#!/usr/bin/env python3
"""Plan (and later convert) NVIDIA Nemotron 3.5 Bolt 74B-A9B from its 16-bit release.

The checkpoint (`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16`, 149.77 GB in 63
shards) is a Mamba-2 + LatentMoE + attention hybrid, not a Qwen variant:

  - 88 layers: 40 mamba, 40 MoE, 8 attention (7, 16, 25, 36, 47, 58, 69, 80)
  - Mamba-2: d_inner 8192, 128 heads x 64, state 128, 8 groups, conv 4
  - LatentMoE: 240-496 routed experts per layer (13,824 total), top-9,
    latent 1024, per-expert width 1536-2560, shared expert 5376, relu^2
  - Attention: 32 query / 2 KV heads, head dim 128, no rotary

The converter writes an affine group-64 snapshot that `TinyTitanRepack` reads,
exactly like `prepare_qwen38.py` does. `--plan` fetches only the config, the
index and the safetensors headers, so a geometry or alignment mistake is found
before 150 GB is fetched.

    tools/prepare_nemotron.py --plan
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16"
REVISION = "e09a4cedb1787db1760375ee5539f222cae0c672"
BASE = f"https://huggingface.co/{REPO}/resolve/{REVISION}"
GROUP_SIZE = 64
RETRY = ("--retry", "5", "--retry-delay", "2", "--retry-all-errors")

# The production geometry this converter was written against. Anything that
# disagrees is a different model wearing the same name, and refusing it here is
# the cheap place to find out.
EXPECTED = {
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "vocab_size": 131072,
    "mamba_num_heads": 128,
    "mamba_head_dim": 64,
    "ssm_state_size": 128,
    "n_groups": 8,
    "conv_kernel": 4,
    "moe_latent_size": 1024,
    "moe_shared_expert_intermediate_size": 5376,
    "n_group": 1,
    "topk_group": 1,
    "chunk_size": 128,
}

ATTENTION_LAYERS = (7, 16, 25, 36, 47, 58, 69, 78)


def fetch(remote: str, byte_range: str | None = None) -> bytes:
    command = ["curl", "-sfL", "--max-time", "180", *RETRY]
    if byte_range is not None:
        command += ["-r", byte_range]
    result = subprocess.run(
        [*command, f"{BASE}/{remote}"], capture_output=True, check=False
    )
    if result.returncode != 0:
        raise SystemExit(
            f"cannot fetch {remote} (curl exited {result.returncode}): "
            f"{result.stderr.decode(errors='replace').strip()}"
        )
    return result.stdout


def fetch_json(remote: str) -> dict:
    return json.loads(fetch(remote))


def fetch_header(shard: str) -> dict:
    prefix = fetch(shard, "0-7")
    size = struct.unpack("<Q", prefix[:8])[0]
    if size <= 0 or size > 256 * 1024 * 1024:
        raise SystemExit(f"{shard}: implausible safetensors header size {size}")
    header = json.loads(fetch(shard, f"8-{8 + size - 1}"))
    header.pop("__metadata__", None)
    return header


def elements(shape: list[int]) -> int:
    total = 1
    for dimension in shape:
        total *= dimension
    return total


def affine_bytes(count: int, bits: int) -> int:
    """Packed weight bytes plus one bf16 scale and one bf16 bias per group."""
    return count * bits // 8 + (count // GROUP_SIZE) * 4


# --- tensor roles -----------------------------------------------------------

ROUTED = re.compile(
    r"^language_model\.backbone\.layers\.(\d+)\.mixer\.experts\.(\d+)\."
    r"(up_proj|down_proj)\.weight$"
)
RESIDENT = re.compile(r"^language_model\.backbone\.layers\.(\d+)\.")
VISION = ("vision_model.", "mlp1.")


def is_vision(name: str) -> bool:
    return name.startswith(VISION)


def is_mamba(name: str) -> bool:
    return bool(
        re.search(
            r"\.mixer\.(A_log|D|dt_bias|conv1d\.(weight|bias)|in_proj\.weight|out_proj\.weight|norm\.weight)$",
            name,
        )
    )


def is_attention(name: str) -> bool:
    return bool(re.search(r"\.mixer\.(q_proj|k_proj|v_proj|o_proj)\.weight$", name))


def expert_layers(headers: dict[str, dict]) -> dict[int, int]:
    counts: Counter[int] = Counter()
    for name in headers:
        match = ROUTED.match(name)
        if match:
            counts[int(match.group(1))] += 1
    # Up and down are both present, so each expert counts twice.
    return {layer: value // 2 for layer, value in counts.items()}


def width_of(name: str, width: int) -> int | None:
    """The width a resident tensor is carried at, or None for bf16.

    Norms, the router, the Mamba scalars and the depthwise conv keep the
    checkpoint's bf16: they are either decision tensors or their last dimension
    is not group-aligned. The embedding and the untied head are 8-bit in both
    installs, as every other family in this tree does.
    """
    if name.endswith("conv1d.weight"):
        # [conv_dim, 1, kernel] -- the last dimension is 4, not group-aligned.
        return None
    if name.endswith(
        (
            "norm.weight",
            "norm_f.weight",
            "gate.weight",
            "e_score_correction_bias",
            "A_log",
            "D",
            "dt_bias",
            "conv1d.bias",
        )
    ):
        return None
    if name in ("language_model.backbone.embeddings.weight", "language_model.lm_head.weight"):
        return 8
    return width


def validate_config(config: dict) -> dict:
    if config.get("model_type") != "nemotron_h_omni":
        raise SystemExit(f"config model_type is {config.get('model_type')!r}, expected nemotron_h_omni")
    text = config.get("llm_config")
    if not isinstance(text, dict):
        raise SystemExit("config has no llm_config (this checkpoint nests its text tower there)")
    problems = []
    for key, expected in EXPECTED.items():
        actual = text.get(key)
        if actual != expected:
            problems.append(f"{key}: config {actual!r} != expected {expected!r}")
    kinds = text.get("layers_block_type")
    if not isinstance(kinds, list) or len(kinds) != 88:
        problems.append("layers_block_type is not 88 entries")
    else:
        counts = Counter(kinds)
        if counts != {"mamba": 40, "moe": 40, "attention": 8}:
            problems.append(f"layer-type census {dict(counts)} != 40 mamba / 40 moe / 8 attention")
        attention = [index for index, kind in enumerate(kinds) if kind == "attention"]
        if tuple(attention) != ATTENTION_LAYERS:
            problems.append(f"attention layers {attention} != {list(ATTENTION_LAYERS)}")
    for layer in text.get("block_configs", []):
        if layer.get("block_type") == "moe" and layer.get("num_experts_per_tok") != 9:
            problems.append("a MoE block does not route top-9")
    if text.get("tie_word_embeddings") is not False:
        problems.append("tie_word_embeddings must be false (the head is a real tensor)")
    if text.get("moe_latent_size") != 1024:
        problems.append("moe_latent_size must be 1024")
    if problems:
        raise SystemExit("checkpoint does not match the ported geometry:\n  " + "\n  ".join(problems))
    return text


def plan() -> int:
    config = fetch_json("config.json")
    text = validate_config(config)
    index = fetch_json("model.safetensors.index.json")
    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    headers: dict[str, dict] = {}
    for shard in shards:
        headers.update(fetch_header(shard))
    if set(headers) != set(weight_map):
        missing = sorted(set(weight_map) - set(headers))
        extra = sorted(set(headers) - set(weight_map))
        raise SystemExit(
            f"index/header mismatch: {len(missing)} missing, {len(extra)} extra; "
            f"first missing {missing[:3]}, first extra {extra[:3]}"
        )

    layers = expert_layers(headers)
    if len(layers) != 40:
        raise SystemExit(f"expected 40 MoE layers with routed experts, found {len(layers)}")
    block_counts = [layer["n_routed_experts"] for layer in text["block_configs"] if layer["block_type"] == "moe"]
    if [layers[i] for i in sorted(layers)] != block_counts:
        raise SystemExit("per-layer expert counts disagree with block_configs")

    print(f"{REPO} @ {REVISION[:12]}")
    print(f"  {len(shards)} shards, {len(headers)} tensors")
    print(f"  88 layers: 40 mamba / 40 moe / 8 attention at {list(ATTENTION_LAYERS)}")
    print(
        f"  {sum(layers.values())} routed experts across {len(layers)} layers "
        f"({sorted(set(layers.values()))}), top-9"
    )

    # Every quantized tensor must have a group-aligned last dimension.
    unaligned = []
    for name, meta in headers.items():
        if is_vision(name):
            continue
        if width_of(name, 4) is None:
            continue
        if isinstance(meta.get("shape"), list) and meta["shape"][-1] % GROUP_SIZE:
            unaligned.append((name, meta["shape"]))
    if unaligned:
        raise SystemExit(
            f"{len(unaligned)} tensors are not group-{GROUP_SIZE} aligned; "
            f"first: {unaligned[:5]}"
        )

    for width in (4, 8):
        resident = 0
        experts = 0
        counts: dict[str, int] = defaultdict(int)
        for name, meta in headers.items():
            if is_vision(name):
                counts["vision (excluded)"] += elements(meta["shape"]) * 2
                continue
            if ROUTED.match(name):
                bits = width
                experts += affine_bytes(elements(meta["shape"]), bits)
                counts["routed experts"] += affine_bytes(elements(meta["shape"]), bits)
                continue
            bits = width_of(name, width)
            if bits is None:
                resident += elements(meta["shape"]) * 2
                counts["bf16 resident"] += elements(meta["shape"]) * 2
            else:
                resident += affine_bytes(elements(meta["shape"]), bits)
                counts[f"{bits}-bit resident"] += affine_bytes(elements(meta["shape"]), bits)
        total = resident + experts
        print(
            f"  {width}-bit: {total / 1e9:.2f} GB = resident {resident / 1e9:.2f} GB "
            f"+ routed experts {experts / 1e9:.2f} GB"
        )
        for key, value in sorted(counts.items(), key=lambda item: -item[1]):
            print(f"      {key:22s} {value / 1e9:8.2f} GB")

    print("  tensors by role:")
    roles: Counter[str] = Counter()
    for name in headers:
        if is_vision(name):
            roles["vision"] += 1
        elif ROUTED.match(name):
            roles["routed expert"] += 1
        elif is_attention(name):
            roles["attention"] += 1
        elif is_mamba(name):
            roles["mamba"] += 1
        elif ".mixer.fc1_latent_proj." in name or ".mixer.fc2_latent_proj." in name:
            roles["latent projection"] += 1
        elif ".mixer.shared_experts." in name:
            roles["shared expert"] += 1
        elif name.endswith("mixer.gate.weight") or name.endswith("e_score_correction_bias"):
            roles["router"] += 1
        elif name.endswith(".norm.weight") or name.endswith("norm_f.weight"):
            roles["norm"] += 1
        elif name.endswith("embeddings.weight") or name.endswith("lm_head.weight"):
            roles["embedding/head"] += 1
        else:
            roles[name] += 1
    for key, value in sorted(roles.items(), key=lambda item: -item[1]):
        print(f"      {key:22s} {value:6d}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="validate and measure; download no weights")
    args = parser.parse_args()
    if not args.plan:
        parser.error("only --plan is implemented yet")
    return plan()


if __name__ == "__main__":
    sys.exit(main())
