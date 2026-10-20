#!/usr/bin/env python3
"""Build a tiered install from a 4-bit one: hot experts wide, cold experts 2-bit.

The runtime expands a cold expert with `q4' = 5 * q2` and `scale / 5`, so this
tool keeps each cold expert's existing scales and biases and only remaps values
with `q2 = round(q4 / 5)` (see benchmark/two_bit_pack.py). A hot expert's blob
is copied verbatim.

    python tools/tiered_repack.py \
        --model models/ornith-1.5_35B_A3B_4Bit \
        --hot-profile /tmp/hot50.json \
        --out models/ornith-1.5_35B_A3B_4Bit-tiered

Only the layer files and `packed_experts/layout.json` are rewritten; the
resident weights are hard-linked, the tokenizer is copied, and the manifest's
file sizes and sha256 are updated so the engine's own verification passes.
No `verified-install.json` is written (it is only required under the strict
integrity policy).

`--layers 0` is a smoke test: layer 0 is tiered and the rest are copied
verbatim, which produces a loadable install in a fraction of the time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "benchmark"))
from two_bit_pack import remap_4bit_to_2bit  # noqa: E402

ALIGN = 16_384


def sha256_of(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tier_layer(
    source: bytes, experts: list[dict], hot: set[int], layer: int
) -> tuple[bytes, list[dict]]:
    """Rewrite one layer. Returns the new file bytes and the new expert entries."""
    out = bytearray()
    entries: list[dict] = []
    for entry in experts:
        expert_id = entry["expert"]
        tensors = sorted(entry["tensors"].items(), key=lambda kv: kv[1]["offset"])
        blob = source[entry["offset"]:entry["offset"] + entry["size"]]
        # The runtime derives one ExpertOffsets struct per layer from expert 0
        # (ModelExpertIO.routedExpertOffsets). After expansion every slot holds
        # the 4-bit layout, but that struct is read from expert 0's entry, so
        # expert 0 must stay at the wide layout or the kernel is handed 2-bit
        # offsets. One extra 4-bit expert is a negligible cost for removing a
        # whole class of wrong-offset bug.
        is_hot = expert_id in hot or expert_id == 0
        new_tensors: dict[str, dict] = {}
        new_blob = bytearray()
        for name, tensor in tensors:
            data = blob[tensor["offset"]:tensor["offset"] + tensor["size"]]
            if is_hot or tensor.get("bits") != 4:
                # Hot weights and every scale/bias are copied unchanged.
                new_blob += data
                new_tensors[name] = {
                    "offset": len(new_blob) - len(data),
                    "size": len(data),
                    "dtype": tensor["dtype"],
                    "shape": tensor["shape"],
                }
                if "bits" in tensor:
                    new_tensors[name]["bits"] = tensor["bits"]
            else:
                remapped = remap_4bit_to_2bit(data)
                new_blob += remapped
                new_tensors[name] = {
                    "offset": len(new_blob) - len(remapped),
                    "size": len(remapped),
                    "dtype": tensor["dtype"],
                    "shape": tensor["shape"],
                    "bits": 2,
                }
        # Page-align every expert so a slot read never shares a page with the
        # previous expert's payload.
        aligned = (len(out) + ALIGN - 1) // ALIGN * ALIGN
        out += b"\x00" * (aligned - len(out))
        offset = len(out)
        out += new_blob
        entries.append({
            "expert": expert_id,
            "offset": offset,
            "size": len(new_blob),
            "tensors": new_tensors,
        })
    return bytes(out), entries


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--hot-profile", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="all",
                    help="'all' or a comma-separated list of layer indices to tier")
    args = ap.parse_args()

    model = pathlib.Path(args.model)
    out = pathlib.Path(args.out)
    manifest = json.loads((model / "manifest.json").read_text())
    layout = json.loads((model / "packed_experts" / "layout.json").read_text())
    profile = json.loads(pathlib.Path(args.hot_profile).read_text())

    tiered = set(range(len(layout["layers"])))
    if args.layers != "all":
        tiered = {int(x) for x in args.layers.split(",")}

    if out.exists():
        print(f"refusing to overwrite {out}", file=sys.stderr)
        return 1
    (out / "packed_experts").mkdir(parents=True)

    if not (out / "model_weights.bin").exists():
        try:
            os.link(model / "model_weights.bin", out / "model_weights.bin")
        except OSError:
            shutil.copy2(model / "model_weights.bin", out / "model_weights.bin")
    for name in ("tokenizer",):
        shutil.copytree(model / name, out / name)

    new_layers = []
    for layer in layout["layers"]:
        index = layer["layer"]
        source_path = model / "packed_experts" / layer["file"]
        destination = out / "packed_experts" / layer["file"]
        if index not in tiered:
            # Hard-link rather than copy: an untiered layer's bytes, size and
            # hash are already correct in the source manifest.
            try:
                os.link(source_path, destination)
            except OSError:
                shutil.copy2(source_path, destination)
                entry = manifest["files"][f"packed_experts/{layer['file']}"]
                entry["size"] = destination.stat().st_size
                entry["sha256"] = sha256_of(destination)
            new_layers.append(layer)
        else:
            hot = set(profile["layers"][str(index)]["hot"])
            data, entries = tier_layer(source_path.read_bytes(), layer["experts"], hot, index)
            destination.write_bytes(data)
            new_layers.append({"layer": index, "file": layer["file"], "experts": entries})
        if index in tiered:
            entry = manifest["files"][f"packed_experts/{layer['file']}"]
            entry["size"] = destination.stat().st_size
            entry["sha256"] = sha256_of(destination)
            print(f"layer {index:02d}: {entry['size']/1e6:.1f} MB")

    layout["layers"] = new_layers
    layout_bytes = json.dumps(layout, separators=(",", ":")).encode()
    (out / "packed_experts" / "layout.json").write_bytes(layout_bytes)
    # The manifest lists layout.json itself, so its size and hash move too.
    manifest["files"]["packed_experts/layout.json"] = {
        "size": len(layout_bytes),
        "sha256": hashlib.sha256(layout_bytes).hexdigest(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
