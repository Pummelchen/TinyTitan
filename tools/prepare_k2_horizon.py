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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan", action="store_true", help="classify the index, download no weights"
    )
    parser.add_argument("--bits", nargs="+", type=int, choices=(4, 8), default=[4, 8])
    args = parser.parse_args()
    if not args.plan:
        parser.error("only --plan is implemented so far; conversion is the next step")
    plan(sorted(set(args.bits)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
