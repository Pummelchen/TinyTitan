#!/usr/bin/env python3.13
"""Fuse K2-Horizon's per-expert MoE tensors for the .ssdai repacker.

The snapshot the converter writes keeps the checkpoint's one-tensor-per-expert
spelling; the repacker's expert planner wants one rank-3 tensor per layer and
role. This rewrites a snapshot with the 100 experts of each sparse layer fused
into `model.layers.N.mlp.switch_mlp.{gate,up,down}_proj.*`; every other tensor
is copied through unchanged (the MoVA value experts stay resident tensors, not
a second streamed set, because top-4 of 64 is small enough to hold).

    python3 tools/fuse_k2_experts.py --source models/k2-horizon-affine-4bit \
        --output .build/k2-fused-4bit
"""

from __future__ import annotations

import argparse
import json
import mmap
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ml_dtypes  # noqa: E402
import numpy as np  # noqa: E402
from prepare_k2_horizon import ShardWriter  # noqa: E402

DTYPES = {
    "F32": np.float32,
    "BF16": ml_dtypes.bfloat16,
    "F16": np.float16,
    "U32": np.uint32,
    "U16": np.uint16,
    "I32": np.int32,
}


class Shard:
    """A read-only mmap of one safetensors file, bf16 included."""

    def __init__(self, path: Path):
        self.file = open(path, "rb")
        self.map = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        header_bytes = struct.unpack("<Q", self.map[:8])[0]
        self.header = json.loads(self.map[8 : 8 + header_bytes])
        self.base = 8 + header_bytes

    def tensor(self, name: str) -> np.ndarray:
        meta = self.header[name]
        start, end = meta["data_offsets"]
        raw = self.map[self.base + start : self.base + end]
        return np.frombuffer(raw, dtype=DTYPES[meta["dtype"]]).reshape(meta["shape"])


ROLES = ("gate", "up", "down")
EXPERT_MARKER = ".mlp.experts."


def is_moe(name: str) -> bool:
    return EXPERT_MARKER in name and ".shared_experts." not in name


def fuse(source: Path, output: Path) -> None:
    config = json.loads((source / "config.json").read_text())
    index = json.loads((source / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    dense = set(config.get("mlp_only_layers", []))
    layers = [n for n in range(config["num_hidden_layers"]) if n not in dense]

    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise SystemExit(f"{output} is not empty")
    writer = ShardWriter(output)
    opened: dict[str, Shard] = {}

    def handle(shard: str) -> Shard:
        if shard not in opened:
            opened[shard] = Shard(source / shard)
        return opened[shard]

    for name, shard in weight_map.items():
        if is_moe(name):
            continue
        writer.add(name, handle(shard).tensor(name))

    for layer in layers:
        for role in ROLES:
            stem = f"model.layers.{layer}.mlp.experts.{{}}.{role}_proj"
            weights = [
                handle(weight_map[f"{stem.format(e)}.weight"]).tensor(f"{stem.format(e)}.weight")
                for e in range(config["num_experts"])
            ]
            scales = [
                handle(weight_map[f"{stem.format(e)}.scales"]).tensor(f"{stem.format(e)}.scales")
                for e in range(config["num_experts"])
            ]
            biases = [
                handle(weight_map[f"{stem.format(e)}.biases"]).tensor(f"{stem.format(e)}.biases")
                for e in range(config["num_experts"])
            ]
            fused = f"model.layers.{layer}.mlp.switch_mlp.{role}_proj"
            writer.add(fused + ".weight", np.stack(weights, axis=0))
            writer.add(fused + ".scales", np.stack(scales, axis=0))
            writer.add(fused + ".biases", np.stack(biases, axis=0))
        print(f"fused layer {layer}", flush=True)

    writer.finish()
    (output / "config.json").write_text(json.dumps(config, indent=1))
    tokenizer = output / "tokenizer"
    tokenizer.mkdir(exist_ok=True)
    for item in (source / "tokenizer").iterdir():
        (tokenizer / item.name).write_bytes(item.read_bytes())
    # The repacker wants tokenizer.json beside config.json for a local
    # snapshot; the sidecar is what the runtime reads.
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "generation_config.json",
    ):
        if (source / "tokenizer" / name).exists():
            (output / name).write_bytes((source / "tokenizer" / name).read_bytes())
    print(f"wrote {output} ({len(writer.index)} tensors)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fuse(args.source, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
