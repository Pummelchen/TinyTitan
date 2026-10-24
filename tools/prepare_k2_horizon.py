#!/usr/bin/env python3.13
"""Plan the K2-Horizon-MoVA-36B-A4B conversion from the index alone.

The checkpoint (IFM/K2-Horizon-MoVA-36B-A4B) is a new architecture, not a
Qwen variant: 48 layers, 100 routed experts at top-8 plus a shared expert, a
grouped two-way RMS norm, a softplus attention gate, and MoVA -- 64 routed
value experts inside the attention of every sparse layer. This script is the
first step of the port and follows `docs/adding-a-model.md` section 2: it
fetches only the config, the index and the safetensors headers, classifies
every tensor, checks the group alignment the affine quantizer needs, and
prints the output size for 4 and 8 bits.

It downloads no weights and writes no files.

    python3.13 tools/prepare_k2_horizon.py --plan
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = "IFM/K2-Horizon-MoVA-36B-A4B"
REVISION = "e9ce717642b2074082875be233c02c729ba47960"
BASE = f"https://huggingface.co/{REPO}/resolve/{REVISION}"
GROUP_SIZE = 64
RETRY = ("--retry", "5", "--retry-delay", "2", "--retry-all-errors")


def fetch(remote: str, byte_range: str | None = None) -> bytes:
    command = ["curl", "-sfL", "--max-time", "120", *RETRY]
    if byte_range is not None:
        command += ["-r", byte_range]
    result = subprocess.run([*command, f"{BASE}/{remote}"], capture_output=True, check=False)
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
    return json.loads(fetch(shard, f"8-{8 + size - 1}"))


def is_bf16(name: str) -> bool:
    """Decision tensors stay at the checkpoint's width."""
    return (
        name.endswith("layernorm.weight")
        or name == "model.norm.weight"
        or name.endswith("mlp.gate.weight")
        or name.endswith("mlp.gate.bias")
        or name.endswith("self_attn.v_router.weight")
        or name.endswith("self_attn.v_router.bias")
    )


def is_eight_bit(name: str) -> bool:
    """The embedding and the untied head keep 8-bit in both installs."""
    return name in ("model.embed_tokens.weight", "lm_head.weight")


def width_of(name: str, width: int) -> int | None:
    if is_bf16(name):
        return None
    if is_eight_bit(name):
        return 8
    return width


def role(name: str) -> str:
    if name.startswith("model.layers."):
        suffix = name.split(".", 3)[3]
        if suffix.endswith("layernorm.weight"):
            return "norm"
        if "v_experts." in suffix:
            return "mova expert"
        if suffix.startswith("self_attn."):
            return "attention"
        if "experts." in suffix and "shared_experts" not in suffix:
            return "moe expert"
        if suffix.startswith("mlp.gate"):
            return "moe router"
        return "mlp"
    if name == "lm_head.weight":
        return "head"
    if name == "model.embed_tokens.weight":
        return "embedding"
    return "norm"


def bytes_for(shape: list[int], width: int | None) -> int:
    elements = 1
    for dimension in shape:
        elements *= dimension
    if width is None:
        return elements * 2
    groups = elements // GROUP_SIZE
    return elements * width // 8 + groups * 4


def plan(widths: list[int]) -> None:
    config = fetch_json("config.json")
    index = fetch_json("model.safetensors.index.json")
    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    headers = {shard: fetch_header(shard) for shard in shards}

    print(f"{REPO} @ {REVISION[:12]}")
    print(
        f"  {config['num_hidden_layers']} layers, hidden {config['hidden_size']}, "
        f"{config['num_experts']} experts top-{config['num_experts_per_tok']}, "
        f"{config['mova_num_experts']} MoVA experts top-{config['mova_num_experts_per_tok']}"
    )
    print(f"  {len(shards)} shards, {len(weight_map)} tensors in the index")

    for width in widths:
        counts: dict[str, int] = {}
        totals: dict[str, int] = {}
        bf16_bytes = 0
        alignment_failures: list[str] = []
        seen = 0
        for header in headers.values():
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                shape = meta["shape"]
                bits = width_of(name, width)
                kind = role(name)
                counts[kind] = counts.get(kind, 0) + 1
                size = bytes_for(shape, bits)
                totals[kind] = totals.get(kind, 0) + size
                if bits is not None:
                    if shape[-1] % GROUP_SIZE:
                        alignment_failures.append(f"{name}: last dim {shape[-1]}")
                    bf16_bytes += 0 if bits is None else 0
                else:
                    bf16_bytes += size
                seen += 1
        if seen != len(weight_map):
            raise SystemExit(f"header walk saw {seen} tensors, index has {len(weight_map)}")
        total = sum(totals.values())
        print(f"\n  {width}-bit plan")
        for kind in sorted(counts):
            print(f"    {kind:<14} {counts[kind]:>6} tensors  {totals[kind] / 1e9:6.2f} GB")
        print(f"    bf16 kept      {bf16_bytes / 1e6:6.0f} MB")
        print(f"    snapshot      ~{total / 1e9:.1f} GB")
        if alignment_failures:
            print("    ALIGNMENT FAILURES:")
            for failure in alignment_failures[:10]:
                print(f"      {failure}")
        else:
            print(f"    every quantized last dim is a multiple of {GROUP_SIZE}")


# --- conversion -------------------------------------------------------------


def quantize_affine(value, bits: int):
    """Group-64 affine quantization, the format the repacker and runtime read."""
    import ml_dtypes
    import numpy as np

    value = value.astype(np.float32)
    if value.shape[-1] % GROUP_SIZE:
        raise SystemExit(f"last dimension {value.shape[-1]} is not group-aligned")
    grouped = value.reshape(*value.shape[:-1], value.shape[-1] // GROUP_SIZE, GROUP_SIZE)
    bias = grouped.min(axis=-1)
    high = grouped.max(axis=-1)
    levels = (1 << bits) - 1
    scale = np.where(high == bias, np.float32(1), (high - bias) / levels)
    scale = scale.astype(ml_dtypes.bfloat16)
    bias = bias.astype(ml_dtypes.bfloat16)
    quantized = (
        np.rint(
            (grouped - bias.astype(np.float32)[..., None]) / scale.astype(np.float32)[..., None]
        )
        .clip(0, levels)
        .astype(np.uint32)
        .reshape(value.shape)
    )
    lanes = 32 // bits
    words = quantized.reshape(*quantized.shape[:-1], quantized.shape[-1] // lanes, lanes)
    packed = np.zeros(words.shape[:-1], dtype=np.uint32)
    for lane in range(lanes):
        packed |= words[..., lane] << np.uint32(bits * lane)
    return packed, scale, bias


def bits_for(name: str, width: int) -> int | None:
    if is_bf16(name):
        return None
    if is_eight_bit(name):
        return 8
    return width


def convert(source: Path, output: Path, width: int) -> None:
    """Quantize a local HF-style checkpoint into the affine snapshot layout.

    Names are preserved: the K2 reader will be written against the
    checkpoint's own spelling, so nothing has to be renamed or fused here.
    """
    import numpy as np
    from safetensors import safe_open
    from safetensors.numpy import save_file

    config = json.loads((source / "config.json").read_text())
    index_path = source / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
    else:
        single = source / "model.safetensors"
        with safe_open(str(single), framework="numpy") as handle:
            weight_map = {name: single.name for name in handle.keys()}

    tensors: dict[str, np.ndarray] = {}
    overrides: dict[str, dict] = {}
    for name, shard in sorted(weight_map.items()):
        with safe_open(str(source / shard), framework="numpy") as handle:
            value = handle.get_tensor(name)
        bits = bits_for(name, width)
        if bits is None:
            import ml_dtypes

            tensors[name] = value.astype(ml_dtypes.bfloat16)
            continue
        packed, scale, bias = quantize_affine(value, bits)
        stem = name[: -len(".weight")]
        tensors[name] = packed
        tensors[stem + ".scales"] = scale
        tensors[stem + ".biases"] = bias
        if bits != width:
            overrides[stem] = {"bits": bits, "group_size": GROUP_SIZE}

    output.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output / "model.safetensors"), metadata={"format": "pt"})
    config = dict(config)
    config["quantization"] = {
        "bits": width,
        "group_size": GROUP_SIZE,
        "mode": "affine",
        **overrides,
    }
    (output / "config.json").write_text(json.dumps(config, indent=1))
    (output / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": sum(int(v.size) for v in tensors.values())},
                "weight_map": {name: "model.safetensors" for name in tensors},
            },
            indent=1,
        )
    )


# --- self test --------------------------------------------------------------


def build_fixture(root: Path) -> None:
    """A tiny K2-shaped checkpoint with random weights, for the converter test."""
    import numpy as np
    from safetensors.numpy import save_file

    rng = np.random.default_rng(7)
    hidden, heads, kv_heads, head_dim = 128, 4, 2, 32
    inter, moe_inter, vocab = 256, 64, 256
    q_width, kv_width = heads * head_dim, kv_heads * head_dim
    config = {
        "model_type": "k2_horizon",
        "architectures": ["K2HorizonForCausalLM"],
        "hidden_size": hidden,
        "intermediate_size": inter,
        "moe_intermediate_size": moe_inter,
        "num_hidden_layers": 2,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "rope_head_dim": head_dim,
        "vocab_size": vocab,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "mova_num_experts": 2,
        "mova_num_experts_per_tok": 1,
        "num_shared_experts": 1,
        "mlp_only_layers": [0],
        "decoder_sparse_step": 1,
        "tie_word_embeddings": False,
        "rope_parameters": {"rope_theta": 10000000.0},
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config, indent=1))

    def weight(*shape):
        return rng.normal(scale=0.02, size=shape).astype(np.float32)

    tensors: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": weight(vocab, hidden),
        "lm_head.weight": weight(vocab, hidden),
        "model.norm.weight": np.ones(hidden, dtype=np.float32),
        "model.layers.0.input_layernorm.weight": np.ones(hidden, dtype=np.float32),
        "model.layers.0.post_attention_layernorm.weight": np.ones(hidden, dtype=np.float32),
        "model.layers.0.self_attn.q_proj.weight": weight(q_width, hidden),
        "model.layers.0.self_attn.k_proj.weight": weight(kv_width, hidden),
        "model.layers.0.self_attn.v_proj.weight": weight(kv_width, hidden),
        "model.layers.0.self_attn.o_proj.weight": weight(hidden, q_width),
        "model.layers.0.self_attn.gate_proj.weight": weight(q_width, hidden),
        "model.layers.0.mlp.gate_proj.weight": weight(inter, hidden),
        "model.layers.0.mlp.up_proj.weight": weight(inter, hidden),
        "model.layers.0.mlp.down_proj.weight": weight(hidden, inter),
        "model.layers.1.input_layernorm.weight": np.ones(hidden, dtype=np.float32),
        "model.layers.1.post_attention_layernorm.weight": np.ones(hidden, dtype=np.float32),
        "model.layers.1.self_attn.q_proj.weight": weight(q_width, hidden),
        "model.layers.1.self_attn.k_proj.weight": weight(kv_width, hidden),
        "model.layers.1.self_attn.o_proj.weight": weight(hidden, q_width),
        "model.layers.1.self_attn.gate_proj.weight": weight(q_width, hidden),
        "model.layers.1.self_attn.v_router.weight": weight(2, hidden),
        "model.layers.1.self_attn.v_router.bias": np.zeros(2, dtype=np.float32),
        "model.layers.1.mlp.gate.weight": weight(4, hidden),
        "model.layers.1.mlp.gate.bias": np.zeros(4, dtype=np.float32),
        "model.layers.1.mlp.shared_experts.gate_proj.weight": weight(moe_inter, hidden),
        "model.layers.1.mlp.shared_experts.up_proj.weight": weight(moe_inter, hidden),
        "model.layers.1.mlp.shared_experts.down_proj.weight": weight(hidden, moe_inter),
    }
    for expert in range(2):
        tensors[f"model.layers.1.self_attn.v_experts.{expert}.weight"] = weight(kv_width, hidden)
    for expert in range(4):
        for role, shape in (
            ("gate_proj", (moe_inter, hidden)),
            ("up_proj", (moe_inter, hidden)),
            ("down_proj", (hidden, moe_inter)),
        ):
            tensors[f"model.layers.1.mlp.experts.{expert}.{role}.weight"] = weight(*shape)

    save_file(tensors, str(root / "model.safetensors"))
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {name: "model.safetensors" for name in tensors},
            }
        )
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def self_test() -> int:
    import ml_dtypes
    import numpy as np
    from safetensors.numpy import load_file

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source, output = root / "source", root / "out-4bit"
        build_fixture(source)
        convert(source, output, 4)
        config = json.loads((output / "config.json").read_text())
        require(config["quantization"]["bits"] == 4, "base width is not 4")
        require(config["quantization"]["group_size"] == 64, "group size is not 64")
        require(
            config["quantization"]["model.embed_tokens"]["bits"] == 8,
            "the embedding is not kept at 8-bit",
        )
        converted = load_file(str(output / "model.safetensors"))
        original = load_file(str(source / "model.safetensors"))

        name = "model.layers.0.mlp.down_proj.weight"
        stem = name[: -len(".weight")]
        packed = converted[name]
        scale = converted[stem + ".scales"]
        bias = converted[stem + ".biases"]
        lanes = 8
        levels = np.uint32(0x0F)
        words = packed
        unpacked = np.zeros((*words.shape[:-1], words.shape[-1] * lanes), dtype=np.uint32)
        for lane in range(lanes):
            unpacked[..., lane::lanes] = (words >> np.uint32(4 * lane)) & levels
        dequantized = unpacked.astype(np.float32) * np.repeat(
            scale.astype(np.float32), 64, axis=-1
        ) + np.repeat(bias.astype(np.float32), 64, axis=-1)
        error = float(np.abs(dequantized - original[name]).max())
        require(error < 0.01, f"4-bit round trip error {error}")

        kept = "model.layers.1.mlp.gate.bias"
        require(converted[kept].dtype == ml_dtypes.bfloat16, "the router bias is not bf16")
        require(
            np.array_equal(converted[kept].astype(np.float32), original[kept]),
            "the router bias changed",
        )

        eight = "model.embed_tokens.weight"
        require(converted[eight].dtype == np.uint32, "the embedding was not quantized")
        require(
            converted[eight[: -len(".weight")] + ".scales"].dtype == ml_dtypes.bfloat16,
            "the embedding scales are not bf16",
        )
        print("K2 converter self-test passed")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan", action="store_true", help="classify the index, download no weights"
    )
    parser.add_argument(
        "--self-test", action="store_true", help="convert a tiny fixture checkpoint"
    )
    parser.add_argument("--source", type=Path, help="local HF-style checkpoint directory")
    parser.add_argument("--output", type=Path, help="affine snapshot directory to write")
    parser.add_argument("--bits", nargs="+", type=int, choices=(4, 8), default=[4, 8])
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.source is not None or args.output is not None:
        if args.source is None or args.output is None:
            parser.error("--source and --output go together")
        for width in sorted(set(args.bits)):
            convert(args.source, args.output / f"{width}bit", width)
        return 0
    if not args.plan:
        parser.error("choose --plan, --source/--output, or --self-test")
    plan(sorted(set(args.bits)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
