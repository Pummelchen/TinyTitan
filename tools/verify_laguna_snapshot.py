#!/usr/bin/env python3
"""Check a converted Laguna snapshot against the shape contract, before repack.

The repacker validates what it reads while copying. That is too late to be
useful on a host where the copy consumes the snapshot as it goes: a missing
companion or a fused tensor with the wrong rank would be discovered after the
source file holding it had been deleted. This reads only the index and the
safetensors headers, checks every tensor the runtime will ask for, and prints
the category totals so they can be compared with `--plan`.

    python3 tools/verify_laguna_snapshot.py --snapshot .build/laguna-affine-4bit

It exits non-zero on the first contract violation and prints the offending
name.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(f"verify: {message}")


def expected_names(config: dict, bits: int) -> dict[str, tuple[str, tuple]]:
    """name -> (kind, shape) where kind decides the dtype check."""
    hidden = config["hidden_size"]
    head_dim = config["head_dim"]
    kv_heads = config["num_key_value_heads"]
    kv_dim = kv_heads * head_dim
    vocab = config["vocab_size"]
    dense = set(config.get("mlp_only_layers", []))
    heads_per_layer = (
        config.get("num_attention_heads_per_layer")
        or [config["num_attention_heads"]] * config["num_hidden_layers"]
    )
    layers = config["num_hidden_layers"]
    moe = config["moe_intermediate_size"]
    dense_width = config["intermediate_size"]
    experts = config["num_experts"]
    out: dict[str, tuple[str, tuple]] = {}

    def packed(rows: int, columns: int, width: int) -> tuple:
        return (rows, columns * width // 32)

    def aux(rows: int, columns: int, group: int) -> tuple:
        return (rows, columns // group)

    group = config.get("quantization", {}).get("group_size", 64)

    def add_matrix(name: str, rows: int, columns: int, width: int) -> None:
        stem = name[: -len(".weight")]
        out[name] = ("u32", packed(rows, columns, width))
        out[stem + ".scales"] = ("bf16", aux(rows, columns, group))
        out[stem + ".biases"] = ("bf16", aux(rows, columns, group))

    # The embedding and head keep 8 bits in every install.
    add_matrix("model.embed_tokens.weight", vocab, hidden, 8)
    if not config.get("tie_word_embeddings", False):
        add_matrix("lm_head.weight", vocab, hidden, 8)
    out["model.norm.weight"] = ("bf16", (hidden,))

    for layer in range(layers):
        prefix = f"model.layers.{layer}"
        out[f"{prefix}.input_layernorm.weight"] = ("bf16", (hidden,))
        out[f"{prefix}.post_attention_layernorm.weight"] = ("bf16", (hidden,))
        out[f"{prefix}.self_attn.q_norm.weight"] = ("bf16", (head_dim,))
        out[f"{prefix}.self_attn.k_norm.weight"] = ("bf16", (head_dim,))
        heads = heads_per_layer[layer]
        query = heads * head_dim
        add_matrix(f"{prefix}.self_attn.q_proj.weight", query, hidden, bits)
        add_matrix(f"{prefix}.self_attn.k_proj.weight", kv_dim, hidden, bits)
        add_matrix(f"{prefix}.self_attn.v_proj.weight", kv_dim, hidden, bits)
        add_matrix(f"{prefix}.self_attn.o_proj.weight", hidden, query, bits)
        add_matrix(f"{prefix}.self_attn.g_proj.weight", heads, hidden, bits)
        if layer in dense:
            add_matrix(f"{prefix}.mlp.gate_proj.weight", dense_width, hidden, bits)
            add_matrix(f"{prefix}.mlp.up_proj.weight", dense_width, hidden, bits)
            add_matrix(f"{prefix}.mlp.down_proj.weight", hidden, dense_width, bits)
        else:
            out[f"{prefix}.mlp.gate.weight"] = ("bf16", (experts, hidden))
            out[f"{prefix}.mlp.experts.e_score_correction_bias"] = (
                "bf16",
                (experts,),
            )
            add_matrix(f"{prefix}.mlp.shared_expert.gate_proj.weight", moe, hidden, bits)
            add_matrix(f"{prefix}.mlp.shared_expert.up_proj.weight", moe, hidden, bits)
            add_matrix(f"{prefix}.mlp.shared_expert.down_proj.weight", hidden, moe, bits)
            for role, rows, columns in (
                ("gate_proj", moe, hidden),
                ("up_proj", moe, hidden),
                ("down_proj", hidden, moe),
            ):
                name = f"{prefix}.mlp.switch_mlp.{role}.weight"
                stem = name[: -len(".weight")]
                out[name] = ("u32", (experts, *packed(rows, columns, bits)))
                out[stem + ".scales"] = (
                    "bf16",
                    (experts, *aux(rows, columns, group)),
                )
                out[stem + ".biases"] = (
                    "bf16",
                    (experts, *aux(rows, columns, group)),
                )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--bits", type=int, required=True)
    arguments = parser.parse_args()

    from safetensors import safe_open

    snapshot = arguments.snapshot
    config = json.loads((snapshot / "config.json").read_text())
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    quant = config.get("quantization", {})
    base_bits = quant.get("bits")
    if base_bits != arguments.bits:
        fail(f"config declares {base_bits}-bit, the command says {arguments.bits}-bit")
    expected = expected_names(config, arguments.bits)
    missing = sorted(set(expected) - set(weight_map))
    extra = sorted(set(weight_map) - set(expected))
    if missing:
        fail(f"{len(missing)} tensors missing; first: {missing[:5]}")
    if extra:
        fail(f"{len(extra)} unexpected tensors; first: {extra[:5]}")

    # Read one header per shard, not the payload.
    headers: dict[str, dict] = {}
    for shard in sorted(set(weight_map.values())):
        path = snapshot / shard
        if not path.exists():
            fail(f"index names a missing shard: {shard}")
        with safe_open(str(path), framework="numpy") as handle:
            headers[shard] = {
                name: (
                    handle.get_slice(name).get_dtype(),
                    tuple(handle.get_slice(name).get_shape()),
                )
                for name in handle.keys()
            }

    errors: list[str] = []
    totals: dict[str, int] = {}
    for name, (kind, shape) in expected.items():
        shard = weight_map[name]
        meta = headers[shard].get(name)
        if meta is None:
            errors.append(f"{name}: absent from its shard {shard}")
            continue
        dtype, actual = meta
        if tuple(actual) != tuple(shape):
            errors.append(f"{name}: shape {actual} != {shape}")
            continue
        want_dtype = "U32" if kind == "u32" else "BF16"
        if dtype != want_dtype:
            errors.append(f"{name}: dtype {dtype} != {want_dtype}")
            continue
        if name.endswith(".weight") and kind == "u32":
            totals["quantized"] = totals.get("quantized", 0) + 1
        elif kind == "bf16":
            totals["bf16"] = totals.get("bf16", 0) + 1
    if errors:
        for error in errors[:10]:
            print(f"  {error}")
        fail(f"{len(errors)} tensors do not match the contract")

    # A layer file must exist for every layer the plan names, including the
    # empty dense one the repacker never writes; only the count of non-empty
    # files is checked here.
    layers = [name for name in weight_map if ".mlp.switch_mlp." in name]
    sparse_layers = {int(name.split(".")[2]) for name in layers}
    dense = set(config.get("mlp_only_layers", []))
    want_sparse = set(range(config["num_hidden_layers"])) - dense
    if sparse_layers != want_sparse:
        fail(
            "fused expert layers do not match the dense/sparse split: "
            f"missing {sorted(want_sparse - sparse_layers)[:5]}, "
            f"extra {sorted(sparse_layers - want_sparse)[:5]}"
        )
    for layer in sorted(want_sparse):
        path = snapshot / f"layer-{layer:02d}.safetensors"
        if not path.exists():
            fail(f"layer {layer} has tensors in the index but no {path.name}")

    print(
        f"snapshot ok: {len(expected)} tensors, {arguments.bits}-bit, "
        f"group {quant.get('group_size')}, "
        f"{len(want_sparse)} sparse layers, "
        f"{len(set(weight_map.values()))} shards"
    )
    print(f"  quantized {totals.get('quantized', 0)}, bf16 {totals.get('bf16', 0)}")
    for key in ("model.embed_tokens", "lm_head"):
        if key in quant:
            print(f"  {key}: {quant[key]['bits']}-bit override")
    return 0


if __name__ == "__main__":
    sys.exit(main())
