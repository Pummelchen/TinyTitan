#!/usr/bin/env python3
"""Build a lower-footprint install from a 4-bit one.

Two profiles:

- `tiered` (default): hot experts stay at 4 bits, cold experts are remapped to
  2 bits and the runtime expands each cold blob into a 4-bit slot with
  `q4' = 5 * q2` and `scale / 5`. Experts are not uniform and expert 0 stays
  wide because the expansion targets its offsets.

- `lowmem`: every routed expert is requantized to a **uniform native 2-bit**
  blob (`q2 = round(q4 / 5)`, `scale / 5`, bias unchanged) packed in the MoE
  kernels' little-endian order. No expansion runs: the kernels take the width
  as a function constant. The manifest declares
  `"memoryProfile": "lowmem"` and `routedExpert.weightBits = 2`, so the
  install advertises itself as `_2-Bit`.

The 2-bit quality result is negative: a model-wide cold tier was measured to
diverge (docs/residency-and-bottleneck-verdict-2026-10-20.md). The `lowmem`
profile is therefore experimental and the catalog refuses to serve it without
`TINYTITAN_ALLOW_EXPERIMENTAL_LOWMEM=1`.

    python tools/tiered_repack.py \
        --model models/ornith-1.5_35B_A3B_4Bit \
        --hot-profile /tmp/hot50.json \
        --out models/ornith-1.5_35B_A3B_4Bit-tiered

    python tools/tiered_repack.py --profile lowmem \
        --model models/ornith-1.5_35B_A3B_4Bit \
        --out models/ornith-1.5_35B_A3B_2Bit-lowmem

Only the layer files and `packed_experts/layout.json` are rewritten; the
resident weights are hard-linked, the tokenizer and any root passthrough files
are copied, and the manifest's file sizes and sha256 are updated. No
`verified-install.json` is written; issue one with
`TinyTitanRepack --verify-install --input-ssdai <out>`.

`--layers 0` is a smoke test for the tiered profile: layer 0 is tiered and the
rest are copied verbatim, which produces a loadable install in a fraction of
the time. The lowmem profile is all-or-nothing (`--layers all`).
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
from two_bit_pack import (  # noqa: E402
    rescale_bf16,
    remap_4bit_to_2bit,
    remap_4bit_to_2bit_native,
    remap_4bit_to_3bit,
)

ALIGN = 16_384


def sha256_of(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tier_layer(
    source: bytes, experts: list[dict], hot: set[int], layer: int, cold_bits: int = 2
) -> tuple[bytes, list[dict]]:
    """Rewrite one layer. Returns the new file bytes and the new expert entries."""
    out = bytearray()
    entries: list[dict] = []
    for entry in experts:
        expert_id = entry["expert"]
        tensors = sorted(entry["tensors"].items(), key=lambda kv: kv[1]["offset"])
        blob = source[entry["offset"] : entry["offset"] + entry["size"]]
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
            data = blob[tensor["offset"] : tensor["offset"] + tensor["size"]]
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
                if cold_bits == 3:
                    remapped = remap_4bit_to_3bit(data)
                else:
                    remapped = remap_4bit_to_2bit(data)
                new_blob += remapped
                new_tensors[name] = {
                    "offset": len(new_blob) - len(remapped),
                    "size": len(remapped),
                    "dtype": tensor["dtype"],
                    "shape": tensor["shape"],
                    "bits": cold_bits,
                }
        # Page-align every expert so a slot read never shares a page with the
        # previous expert's payload.
        aligned = (len(out) + ALIGN - 1) // ALIGN * ALIGN
        out += b"\x00" * (aligned - len(out))
        offset = len(out)
        out += new_blob
        entries.append(
            {
                "expert": expert_id,
                "offset": offset,
                "size": len(new_blob),
                "tensors": new_tensors,
            }
        )
    return bytes(out), entries


def lowmem_layer(source: bytes, experts: list[dict]) -> tuple[bytes, list[dict], int]:
    """Rewrite one layer as *uniform native 2-bit* experts.

    Unlike `tier_layer`, nothing is expanded later: the MoE kernels take the
    width as a function constant and read 2-bit values directly, so every
    weight is packed in the kernel's little-endian order and its group scale is
    multiplied by five (`q2*(5s) == (5*q2)*s`). Each expert occupies one
    uniform, page-aligned slot; there is no hot sentinel because expert 0 is
    also 2-bit and the offsets struct it provides is the layout every expert
    uses.

    Returns the new file bytes, the new expert entries, and the slot stride.
    """
    if not experts:
        return b"", [], 0
    ordered = sorted(experts, key=lambda entry: entry["expert"])
    template = ordered[0]
    template_tensors = sorted(template["tensors"].items(), key=lambda kv: kv[1]["offset"])
    # A uniform layer has the same tensor geometry in every expert; check that
    # before writing anything, because a per-expert size difference would make
    # the single stride below wrong for the rest of the file.
    for entry in ordered:
        if set(entry["tensors"]) != set(template["tensors"]):
            raise ValueError(
                f"expert {entry['expert']} has a different tensor set than expert "
                f"{template['expert']}; lowmem needs a uniform layer"
            )
        for name, tensor in template["tensors"].items():
            if entry["tensors"][name]["size"] != tensor["size"]:
                raise ValueError(
                    f"expert {entry['expert']} tensor {name} is "
                    f"{entry['tensors'][name]['size']} bytes, not {tensor['size']}"
                )

    def rewrite(entry: dict) -> tuple[bytearray, dict[str, dict]]:
        """One expert's blob and the tensor map every expert shares."""
        tensors: dict[str, dict] = {}
        blob = bytearray()
        base = entry["offset"]
        for name, tensor in template_tensors:
            component = name.rsplit("_", 1)[-1]
            data = source[base + tensor["offset"] : base + tensor["offset"] + tensor["size"]]
            bits: int | None
            if tensor["dtype"] == "U32":
                if tensor.get("bits") != 4:
                    raise ValueError(
                        f"{name} is {tensor.get('bits')}-bit; the lowmem converter "
                        f"only requantizes a 4-bit source install"
                    )
                data = remap_4bit_to_2bit_native(data)
                bits = 2
            elif component == "scales":
                # q2 = round(q4/5) spans 0...3, so the group needs 5x the
                # source scale to cover the same range:
                # q2*(5s) + b == (5*q2)*s + b.
                data = rescale_bf16(data, 5.0)
                bits = None
            else:
                bits = None
            tensors[name] = {
                "offset": len(blob),
                "size": len(data),
                "dtype": tensor["dtype"],
                "shape": tensor["shape"],
            }
            if bits is not None:
                tensors[name]["bits"] = bits
            blob.extend(data)
        return blob, tensors

    blobs: list[bytearray] = []
    shared_tensors: dict[str, dict] = {}
    for position, entry in enumerate(ordered):
        blob, tensors = rewrite(entry)
        if position == 0:
            shared_tensors = tensors
            blobs.append(blob)
        else:
            if tensors != shared_tensors:
                raise ValueError(
                    f"expert {entry['expert']} rewrote to a different tensor map; "
                    f"the layer is not uniform"
                )
            blobs.append(blob)

    lengths = {len(blob) for blob in blobs}
    if len(lengths) != 1:
        raise ValueError(f"lowmem experts disagree on blob length: {sorted(lengths)}")
    length = lengths.pop()
    stride = (length + ALIGN - 1) // ALIGN * ALIGN
    if stride == 0:
        raise ValueError("an expert blob is empty")
    padding = b"\x00" * (stride - length)
    out = bytearray()
    entries: list[dict] = []
    for position, (entry, blob) in enumerate(zip(ordered, blobs, strict=False)):
        out += blob
        out += padding
        entries.append(
            {
                "expert": entry["expert"],
                "offset": position * stride,
                "size": stride,
                "tensors": shared_tensors,
            }
        )
    return bytes(out), entries, stride


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--profile",
        choices=("tiered", "lowmem"),
        default="tiered",
        help="tiered: hot wide/cold expanded; lowmem: uniform native 2-bit",
    )
    ap.add_argument(
        "--hot-profile",
        help="tiered only: JSON with one hot set per layer",
    )
    ap.add_argument(
        "--cold-bits",
        type=int,
        default=2,
        choices=(2, 3),
        help="tiered only: width of the cold tier (2 or 3 bits)",
    )
    ap.add_argument("--layers", default="all", help="tiered only: 'all' or layer indices to tier")
    args = ap.parse_args()

    if args.profile == "tiered" and not args.hot_profile:
        ap.error("--profile tiered needs --hot-profile")
    if args.profile == "lowmem" and args.hot_profile:
        ap.error("--profile lowmem tiers every expert; drop --hot-profile")
    if args.profile == "lowmem" and args.layers != "all":
        ap.error("--profile lowmem is all-or-nothing; only --layers all is valid")
    if args.profile == "lowmem" and args.cold_bits != 2:
        ap.error("--profile lowmem is the 2-bit profile; --cold-bits 3 is tiered-only")

    model = pathlib.Path(args.model)
    out = pathlib.Path(args.out)
    manifest = json.loads((model / "manifest.json").read_text())
    layout = json.loads((model / "packed_experts" / "layout.json").read_text())
    hot_profile = (
        json.loads(pathlib.Path(args.hot_profile).read_text(encoding="utf-8"))
        if args.hot_profile
        else None
    )

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
    # Root passthrough files (the PLE constants and n-gram table) are payload
    # the manifest already hashes; a lowmem install of Qwen3.8-Flash-Next is
    # unloadable without them. Hard-link so a 20 GB table is not copied.
    for relative in manifest["files"]:
        if "/" in relative or relative in ("model_weights.bin", "manifest.json"):
            continue
        source_file = model / relative
        if not source_file.is_file():
            continue
        try:
            os.link(source_file, out / relative)
        except OSError:
            shutil.copy2(source_file, out / relative)

    new_layers = []
    strides: set[int] = set()
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
        elif args.profile == "lowmem":
            data, entries, stride = lowmem_layer(source_path.read_bytes(), layer["experts"])
            strides.add(stride)
            destination.write_bytes(data)
            new_layers.append({"layer": index, "file": layer["file"], "experts": entries})
        else:
            hot = set(hot_profile["layers"][str(index)]["hot"])
            data, entries = tier_layer(
                source_path.read_bytes(), layer["experts"], hot, index, cold_bits=args.cold_bits
            )
            destination.write_bytes(data)
            new_layers.append({"layer": index, "file": layer["file"], "experts": entries})
        if index in tiered:
            entry = manifest["files"][f"packed_experts/{layer['file']}"]
            entry["size"] = destination.stat().st_size
            entry["sha256"] = sha256_of(destination)
            print(f"layer {index:02d}: {entry['size'] / 1e6:.1f} MB")

    if args.profile == "lowmem":
        if len(strides) != 1:
            raise ValueError(f"lowmem layers disagree on stride: {sorted(strides)}")
        stride = strides.pop()
        # One uniform slot per expert, and the manifest/layout must both name
        # that stride: the runtime reads the slot capacity from the manifest
        # and each expert's own size from the layout.
        layout["expertStride"] = stride
        manifest["expertStride"] = stride
        manifest["memoryProfile"] = "lowmem"
        quant = manifest.get("quant")
        if not isinstance(quant, dict) or "routedExpert" not in quant:
            raise ValueError("source manifest has no quant.routedExpert slot")
        quant["routedExpert"]["weightBits"] = 2
        quant["routedExpert"]["scheme"] = "affine"
        quant["routedExpert"]["scaleType"] = "BF16"
        quant["routedExpert"]["biasType"] = "BF16"
        # The per-expert U32 tensors carry their own bits: 2.

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
