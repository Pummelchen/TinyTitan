#!/usr/bin/env python3
"""Convert poolside/Laguna-S-2.1 into TinyTitan's affine snapshot layout.

Laguna is a new architecture, not a Qwen variant: 48 layers, 256 routed
experts at top-10 plus one shared expert, GQA with 8 KV heads and a different
head count on the sliding-window layers, per-head softplus attention gating,
per-head QK RMS norms, a sigmoid router with an aux-loss-free selection bias
and a 2.5 routed scale, and two RoPE configurations (YaRN on the 12 global
layers, plain RoPE on the 36 sliding-window layers).

The checkpoint is 235 GB of bf16 in 46 shards, so this converter never stages
it: it fetches one *layer* at a time as byte ranges (a layer's 256 experts are
contiguous inside one or two shards), quantizes that layer at every requested
width, writes one safetensors file per width, and deletes the fetched bytes.
A completed layer file is never touched again, which is what makes an
interrupted 9-hour run resumable at layer granularity.

Experts are fused on the way in -- 256 separate gate/up/down tensors become
one rank-3 `mlp.switch_mlp.*_proj` per layer and role -- because that is the
shape the `.ssdai` repacker's expert planner consumes.

    # The plan only (config, index and shard headers; no weights).
    python3 tools/prepare_laguna_s21.py --plan

    # Convert from the Hub, both widths, resumable.
    python3 tools/prepare_laguna_s21.py --from-hf \
        --output .build/laguna-affine --bits 4 8

    # Convert a local HF-style checkpoint directory.
    python3 tools/prepare_laguna_s21.py --source <dir> --output <dir> --bits 4 8

    # Fixture test, no network.
    python3 tools/prepare_laguna_s21.py --self-test
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import tempfile
import threading
import time
from pathlib import Path

REPO = "poolside/Laguna-S-2.1"
REVISION = "0f573140834b11cfac0c2af97a101a7a69a13e22"
BASE = f"https://huggingface.co/{REPO}/resolve/{REVISION}"
GROUP_SIZE = 64
# 128 MiB range requests: small enough that a dropped connection costs a
# retry rather than the layer, large enough that the per-request redirect and
# TLS setup do not dominate a ~7.8 MB/s link.
CHUNK = 4 * 1024 * 1024
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
)


def log(message: str) -> None:
    print(message, flush=True)


# --- HTTP -------------------------------------------------------------------


def curl_bytes(url: str, start: int | None = None, end: int | None = None) -> bytes:
    command = [
        "curl",
        "-sfL",
        "--http1.1",
        "--max-time",
        "1800",
        "--retry",
        "5",
        "--retry-delay",
        "3",
        "--retry-all-errors",
    ]
    if start is not None:
        command += ["-r", f"{start}-{end}"]
    result = subprocess.run([*command, url], capture_output=True, check=False)
    if result.returncode != 0:
        raise SystemExit(
            f"cannot fetch {url} [{start}-{end}] (curl {result.returncode}): "
            f"{result.stderr.decode(errors='replace').strip()}"
        )
    return result.stdout


def fetch_json(remote: str) -> dict:
    return json.loads(curl_bytes(f"{BASE}/{remote}"))


def fetch_header(shard: str, timeout: int = 300) -> tuple[dict, int]:
    """The shard's tensor table and the file offset its data section starts at.

    `data_offsets` in a safetensors header are relative to the data section,
    not the file: the 8-byte length, the JSON header and its padding come
    first. Every range this script fetches is absolute, so the base travels
    with the header instead of being added at each call site.
    """
    prefix = curl_bytes(f"{BASE}/{shard}", 0, 7)
    if len(prefix) < 8:
        raise SystemExit(f"short header prefix for {shard}")
    size = struct.unpack("<Q", prefix[:8])[0]
    raw = curl_bytes(f"{BASE}/{shard}", 8, 8 + size - 1)
    return json.loads(raw), 8 + size


def download_piece(url: str, start: int, end: int, destination: Path) -> None:
    """Fetch the half-open range [start, end) into `destination`.

    `end` is exclusive, matching a safetensors header's `data_offsets[1]`.
    Treating it as inclusive asks for one byte past EOF on the last range of
    every shard; HF answers 416 and the whole resident phase fails on the last
    byte.
    """
    with open(destination, "wb") as out:
        position = start
        while position < end:
            stop = min(position + CHUNK, end)
            expected = stop - position
            part = destination.with_name(destination.name + ".part")
            attempt = 0
            while True:
                if part.exists():
                    part.unlink()
                result = subprocess.run(
                    [
                        "curl",
                        "-sfL",
                        "--http1.1",
                        "--max-time",
                        "1800",
                        "-r",
                        f"{position}-{stop - 1}",
                        "-o",
                        str(part),
                        url,
                    ],
                    check=False,
                )
                got = part.stat().st_size if part.exists() else 0
                if result.returncode == 0 and got == expected:
                    break
                attempt += 1
                if attempt >= 8:
                    raise SystemExit(
                        f"download failed for {url} [{position}-{stop - 1}] "
                        f"(curl {result.returncode}, {got}/{expected} bytes)"
                    )
                if got:
                    # Keep what arrived and continue from there.
                    out.write(part.read_bytes())
                    position += got
                    if position >= end:
                        break
                    stop = min(position + CHUNK, end)
                    expected = stop - position
                    log(f"    short read, continuing at {position}")
                time.sleep(min(2**attempt, 30))
            if part.exists():
                out.write(part.read_bytes())
                part.unlink()
            position += expected


def download_range(remote: str, start: int, end: int, destination: Path) -> None:
    """Fetch a half-open range over several connections, in order.

    One connection gets a small share of a link several sessions are using:
    measured on this host, parallel range reads of the same shard finish in a
    fraction of the wall time, and the sub-ranges concatenate in order, so
    nothing about the result changes. Short ranges stay on one connection to
    keep the request count down.
    """
    url = f"{BASE}/{remote}"
    length = end - start
    if length <= 0:
        raise SystemExit(f"empty download range [{start}, {end})")
    workers = min(64, max(1, length // (64 * 1024 * 1024)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == length:
        log(f"    {destination.name} is complete, skipping")
        return
    if workers == 1:
        download_piece(url, start, end, destination)
        return
    span = length // workers
    pieces: list[tuple[Path, int, int]] = []
    threads: list[threading.Thread] = []
    errors: list[BaseException] = []

    def fetch(index: int, low: int, high: int) -> None:
        path = destination.with_name(f"{destination.name}.w{index}")
        try:
            download_piece(url, low, high, path)
        except BaseException as error:  # surfaced on the main thread
            errors.append(error)

    for index in range(workers):
        low = start + index * span
        high = end if index == workers - 1 else low + span
        path = destination.with_name(f"{destination.name}.w{index}")
        pieces.append((path, low, high))
        thread = threading.Thread(target=fetch, args=(index, low, high), daemon=True)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    with open(destination, "wb") as out:
        for path, _, _ in pieces:
            with open(path, "rb") as source:
                while True:
                    block = source.read(16 * 1024 * 1024)
                    if not block:
                        break
                    out.write(block)
            path.unlink()
    expected = sum(high - low for _, low, high in pieces)
    if destination.stat().st_size != expected:
        raise SystemExit(
            f"{destination} assembled to {destination.stat().st_size} bytes, expected {expected}"
        )


# --- classification ---------------------------------------------------------


def is_bf16(name: str) -> bool:
    """Decision tensors stay at the checkpoint's bf16 width."""
    return (
        name.endswith("layernorm.weight")
        or name == "model.norm.weight"
        or name.endswith("self_attn.q_norm.weight")
        or name.endswith("self_attn.k_norm.weight")
        or name.endswith(".mlp.gate.weight")
        or name.endswith("e_score_correction_bias")
    )


def is_eight_bit(name: str) -> bool:
    """The embedding and the untied head keep 8-bit in both installs."""
    return name in ("model.embed_tokens.weight", "lm_head.weight")


def bits_for(name: str, width: int) -> int | None:
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
        if suffix.startswith("self_attn."):
            return "attention"
        if suffix == "mlp.gate.weight" or suffix.endswith("e_score_correction_bias"):
            return "moe router"
        if suffix.startswith("mlp.shared_expert."):
            return "shared expert"
        if ".mlp.experts." in suffix and ".shared_expert." not in suffix:
            return "moe expert"
        return "dense mlp"
    if name == "lm_head.weight":
        return "head"
    if name == "model.embed_tokens.weight":
        return "embedding"
    return "norm"


def is_expert(name: str) -> bool:
    """A routed expert's own tensor (`.mlp.experts.<index>.*`).

    The router's aux-loss-free selection bias also lives under
    `.mlp.experts.`; it carries no expert index and stays resident.
    """
    if not name.startswith("model.layers.") or ".mlp.experts." not in name:
        return False
    if ".shared_expert." in name:
        return False
    tail = name.split(".mlp.experts.", 1)[1]
    return tail.split(".", 1)[0].isdigit()


def expert_key(name: str) -> tuple[int, int, str] | None:
    """(layer, expert, role) for one per-expert tensor, or None."""
    if not is_expert(name):
        return None
    suffix = name.split(".", 3)[3]  # mlp.experts.<e>.<role>.weight
    parts = suffix.split(".")
    try:
        expert = int(parts[2])
        role_name = parts[3]
    except (IndexError, ValueError):
        return None
    if role_name not in ("gate_proj", "up_proj", "down_proj"):
        return None
    return int(name.split(".")[2]), expert, role_name


def bytes_for(shape: list[int], width: int | None) -> int:
    elements = 1
    for dimension in shape:
        elements *= dimension
    if width is None:
        return elements * 2
    if elements % GROUP_SIZE:
        raise SystemExit(f"tensor of {elements} elements is not group-aligned")
    return elements * width // 8 + (elements // GROUP_SIZE) * 4


# --- quantization -----------------------------------------------------------


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


# --- safetensors reading ----------------------------------------------------


class HeaderIndex:
    """Every tensor in the checkpoint: shard, absolute byte range, dtype, shape."""

    def __init__(self, headers: dict[str, tuple[dict, int]]):
        self.entries: dict[str, dict] = {}
        for shard, (header, base) in headers.items():
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                self.entries[name] = {
                    "shard": shard,
                    "start": base + meta["data_offsets"][0],
                    "end": base + meta["data_offsets"][1],
                    "dtype": meta["dtype"],
                    "shape": meta["shape"],
                }

    def names(self) -> list[str]:
        return list(self.entries)


def numpy_dtype(name: str):
    """The numpy/ml_dtypes dtype for a safetensors dtype string.

    `np.dtype("bfloat16")` is not a numpy dtype: bf16 comes from ml_dtypes,
    and passing the string to `frombuffer` raises "data type 'bfloat16' not
    understood" on the first real checkpoint tensor (the fixture is F32).
    """
    import ml_dtypes
    import numpy as np

    return {
        "F32": np.float32,
        "BF16": ml_dtypes.bfloat16,
        "F16": np.float16,
        "U32": np.uint32,
        "U16": np.uint16,
        "I32": np.int32,
    }[name]


class Segment:
    """One fetched byte range: the shard it came from, a path, a range, a base."""

    def __init__(self, shard: str, path: Path, start: int, end: int, base: int):
        self.shard = shard
        self.path = path
        self.start = start
        self.end = end
        self.base = base  # offset of `start` inside `path`

    def contains(self, shard: str, start: int, end: int) -> bool:
        return shard == self.shard and self.start <= start and end <= self.end


class TensorSource:
    """Reads tensors from a list of segments, local files or fetched ranges."""

    def __init__(self, segments: list[Segment]):
        self.segments = segments
        self._maps: dict[Path, object] = {}

    def _map(self, path: Path):
        import mmap

        if path not in self._maps:
            handle = open(path, "rb")
            self._maps[path] = (handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ))
        return self._maps[path][1]

    def tensor(self, entry: dict):
        import numpy as np

        dtype = numpy_dtype(entry["dtype"])
        for segment in self.segments:
            if segment.contains(entry["shard"], entry["start"], entry["end"]):
                mapping = self._map(segment.path)
                offset = segment.base + (entry["start"] - segment.start)
                raw = mapping[offset : offset + (entry["end"] - entry["start"])]
                array = np.frombuffer(raw, dtype=dtype)
                return array.reshape(entry["shape"])
        raise KeyError(f"no segment holds [{entry['start']}, {entry['end']})")

    def close(self) -> None:
        for handle, mapping in self._maps.values():
            mapping.close()
            handle.close()
        self._maps.clear()


def merge_ranges(ranges: list[tuple[int, int]], gap: int = 1 << 20) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


# --- conversion -------------------------------------------------------------


def quantized_tensor(name: str, value, width: int):
    """Return the entries for one tensor at one width."""
    bits = bits_for(name, width)
    if bits is None:
        import ml_dtypes

        return {name: value.astype(ml_dtypes.bfloat16)}
    packed, scale, bias = quantize_affine(value, bits)
    stem = name[: -len(".weight")]
    return {name: packed, stem + ".scales": scale, stem + ".biases": bias}


def write_safetensors(path: Path, tensors: dict) -> None:
    from safetensors.numpy import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    save_file(tensors, str(temporary))
    os.replace(temporary, path)


def write_config(output: Path, config: dict, width: int) -> None:
    text = dict(config)
    overrides = {}
    for name in ("model.embed_tokens", "lm_head"):
        if 8 != width:
            overrides[name] = {"bits": 8, "group_size": GROUP_SIZE}
    text["quantization"] = {
        "bits": width,
        "group_size": GROUP_SIZE,
        "mode": "affine",
        **overrides,
    }
    (output / "config.json").write_text(json.dumps(text, indent=1))


def write_index(output: Path, files: list[Path]) -> None:
    from safetensors import safe_open

    weight_map: dict[str, str] = {}
    total = 0
    for path in sorted(files):
        with safe_open(str(path), framework="numpy") as handle:
            for name in handle.keys():
                weight_map[name] = path.name
                total += 1
    (output / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total}, "weight_map": weight_map},
            indent=1,
        )
    )


class Converter:
    def __init__(
        self,
        output: Path,
        widths: list[int],
        config: dict,
        index: HeaderIndex,
    ):
        self.output = output
        self.widths = widths
        self.config = config
        self.index = index
        self.layers = config["num_hidden_layers"]
        self.dense = set(config.get("mlp_only_layers", []))

    def output_for(self, width: int) -> Path:
        return self.output.with_name(f"{self.output.name}-{width}bit")

    def layer_path(self, width: int, layer: int) -> Path:
        return self.output_for(width) / f"layer-{layer:02d}.safetensors"

    def resident_path(self, width: int) -> Path:
        return self.output_for(width) / "resident.safetensors"

    def prepare_directories(self) -> None:
        for width in self.widths:
            directory = self.output_for(width)
            directory.mkdir(parents=True, exist_ok=True)

    def write_resident(self, source: TensorSource, names: list[str]) -> None:
        for width in self.widths:
            path = self.resident_path(width)
            if path.exists():
                log(f"  resident {width}-bit exists, skipping")
                continue
            tensors: dict = {}
            for name in names:
                entry = self.index.entries[name]
                tensors.update(quantized_tensor(name, source.tensor(entry), width))
            write_safetensors(path, tensors)
            log(f"  wrote {path.name} ({width}-bit, {len(tensors)} tensors)")
            del tensors

    def write_layer(self, layer: int, source: TensorSource) -> None:
        """Fuse one layer's 256 experts and write one file per width."""
        import numpy as np

        roles = ("gate_proj", "up_proj", "down_proj")
        accumulators: dict[int, dict[str, list]] = {
            width: {role: [] for role in roles} for width in self.widths
        }
        for expert in range(self.config["num_experts"]):
            for role_name in roles:
                name = f"model.layers.{layer}.mlp.experts.{expert}.{role_name}.weight"
                entry = self.index.entries.get(name)
                if entry is None:
                    raise SystemExit(f"missing expert tensor {name}")
                value = source.tensor(entry)
                for width in self.widths:
                    accumulators[width][role_name].append(quantize_affine(value, width))
        for width in self.widths:
            path = self.layer_path(width, layer)
            tensors: dict = {}
            for role_name in roles:
                stem = f"model.layers.{layer}.mlp.switch_mlp.{role_name}"
                parts = accumulators[width][role_name]
                tensors[stem + ".weight"] = np.stack([p[0] for p in parts], axis=0)
                tensors[stem + ".scales"] = np.stack([p[1] for p in parts], axis=0)
                tensors[stem + ".biases"] = np.stack([p[2] for p in parts], axis=0)
                accumulators[width][role_name] = []
            write_safetensors(path, tensors)
            log(f"  wrote {path.name} ({width}-bit, {len(tensors)} tensors)")
            del tensors
        del accumulators

    def resident_names(self) -> list[str]:
        return [name for name in self.index.names() if not is_expert(name)]

    def expert_names(self, layer: int) -> list[str]:
        return [
            name for name in self.index.names() if is_expert(name) and expert_key(name)[0] == layer
        ]

    def finish(self) -> None:
        for width in self.widths:
            directory = self.output_for(width)
            files = [
                path for path in directory.glob("*.safetensors") if not path.name.endswith(".tmp")
            ]
            write_index(directory, files)
            write_config(directory, self.config, width)
            tokenizer = directory / "tokenizer"
            tokenizer.mkdir(exist_ok=True)
            log(f"  {directory.name}: {len(files)} shards")


# --- remote conversion ------------------------------------------------------


def remote_convert(output: Path, work: Path, widths: list[int], resume: bool) -> int:
    config = fetch_json("config.json")
    index = fetch_json("model.safetensors.index.json")
    shards = sorted(set(index["weight_map"].values()))
    work.mkdir(parents=True, exist_ok=True)
    header_file = work / "headers.json"
    if header_file.exists():
        headers = json.loads(header_file.read_text())
    else:
        log(f"fetching {len(shards)} shard headers")
        fetched = {shard: fetch_header(shard) for shard in shards}
        headers = {shard: [header, base] for shard, (header, base) in fetched.items()}
        header_file.write_text(json.dumps(headers))
    headers = {shard: (value[0], value[1]) for shard, value in headers.items()}
    header_index = HeaderIndex(headers)
    if len(header_index.entries) != len(index["weight_map"]):
        raise SystemExit(
            f"header walk saw {len(header_index.entries)} tensors, "
            f"index has {len(index['weight_map'])}"
        )
    converter = Converter(output, widths, config, header_index)
    converter.prepare_directories()

    # Resident tensors: attention, norms, the router, the shared expert, the
    # dense layer, the embedding and the head. They live in the first shards.
    resident_missing = [width for width in widths if not converter.resident_path(width).exists()]
    if resident_missing:
        names = converter.resident_names()
        log(f"resident: {len(names)} tensors")
        segments = fetch_segments(header_index, names, work / "resident-segments")
        source = TensorSource(segments)
        converter.write_resident(source, names)
        source.close()
        for segment in segments:
            segment.path.unlink(missing_ok=True)

    for layer in range(config["num_hidden_layers"]):
        if layer in converter.dense:
            continue
        done = all(converter.layer_path(width, layer).exists() for width in widths)
        if done and resume:
            log(f"layer {layer:02d}: exists, skipping")
            continue
        if done and not resume:
            raise SystemExit(
                f"{converter.layer_path(widths[0], layer)} exists; "
                "pass --resume or remove the output"
            )
        started = time.time()
        log(f"layer {layer:02d}: fetching {len(converter.expert_names(layer))} tensors")
        segments = fetch_segments(
            header_index, converter.expert_names(layer), work / f"layer-{layer:02d}"
        )
        source = TensorSource(segments)
        converter.write_layer(layer, source)
        source.close()
        for segment in segments:
            segment.path.unlink(missing_ok=True)
        log(f"layer {layer:02d}: done in {time.time() - started:.0f}s")

    converter.finish()
    for width in widths:
        directory = converter.output_for(width)
        tokenizer = directory / "tokenizer"
        tokenizer.mkdir(exist_ok=True)
        for remote in TOKENIZER_FILES:
            destination = tokenizer / remote
            if not destination.exists():
                data = curl_bytes(f"{BASE}/{remote}")
                destination.write_bytes(data)
                # The repacker's local import reads the tokenizer beside
                # config.json, not under tokenizer/; the runtime reads the
                # sidecar. Write both.
                (directory / remote).write_bytes(data)
            elif not (directory / remote).exists():
                (directory / remote).write_bytes(destination.read_bytes())
    log(f"converted {REPO} @ {REVISION[:12]} -> {output} at {widths}")
    return 0


def fetch_segments(index: HeaderIndex, names: list[str], prefix: Path) -> list[Segment]:
    """Fetch the ranges that hold `names`, one segment per shard run."""
    by_shard: dict[str, list[tuple[int, int]]] = {}
    for name in names:
        entry = index.entries[name]
        by_shard.setdefault(entry["shard"], []).append((entry["start"], entry["end"]))
    segments: list[Segment] = []
    # The shard number must be in the name: a layer's experts (and the whole
    # resident set) span more than one shard, and a per-shard position alone
    # made the second shard's first segment overwrite the first shard's.
    for shard_number, (shard, ranges) in enumerate(sorted(by_shard.items())):
        merged = merge_ranges(ranges)
        for position, (start, end) in enumerate(merged):
            path = prefix.with_name(f"{prefix.name}-{shard_number:02d}-{position:03d}.bin")
            log(f"    {shard} [{start}-{end}] ({(end - start) / 1e9:.2f} GB)")
            download_range(shard, start, end, path)
            segments.append(Segment(shard, path, start, end, 0))
    return segments


# --- local conversion -------------------------------------------------------


def local_convert(source: Path, output: Path, widths: list[int]) -> int:
    from safetensors import safe_open

    config = json.loads((source / "config.json").read_text())
    index_path = source / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
    else:
        single = source / "model.safetensors"
        with safe_open(str(single), framework="numpy") as handle:
            weight_map = {name: single.name for name in handle.keys()}
    shards = sorted(set(weight_map.values()))
    headers = {}
    for shard in shards:
        with safe_open(str(source / shard), framework="numpy") as handle:
            headers[shard] = {
                name: {
                    "data_offsets": None,
                    "dtype": handle.get_slice(name).get_dtype(),
                    "shape": list(handle.get_slice(name).get_shape()),
                }
                for name in handle.keys()
            }
    # `safe_open` exposes dtype and shape but not byte offsets; re-read the
    # JSON header for the offsets.
    for shard in shards:
        with open(source / shard, "rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            raw = json.loads(handle.read(size))
        for name, meta in raw.items():
            if name == "__metadata__":
                continue
            headers[shard][name]["data_offsets"] = meta["data_offsets"]
            headers[shard][name]["dtype"] = meta["dtype"]
            headers[shard][name]["shape"] = meta["shape"]
    with_base = {}
    for shard in shards:
        with open(source / shard, "rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
        with_base[shard] = (headers[shard], 8 + size)
    header_index = HeaderIndex(with_base)
    converter = Converter(output, widths, config, header_index)
    converter.prepare_directories()
    segments = [
        Segment(shard, source / shard, 0, (source / shard).stat().st_size, 0) for shard in shards
    ]
    tensor_source = TensorSource(segments)
    names = converter.resident_names()
    converter.write_resident(tensor_source, names)
    for layer in range(config["num_hidden_layers"]):
        if layer in converter.dense:
            continue
        converter.write_layer(layer, tensor_source)
    tensor_source.close()
    converter.finish()
    tokenizer = source / "tokenizer"
    if tokenizer.exists():
        for width in widths:
            directory = converter.output_for(width)
            destination = directory / "tokenizer"
            destination.mkdir(exist_ok=True)
            for item in tokenizer.iterdir():
                (destination / item.name).write_bytes(item.read_bytes())
                (directory / item.name).write_bytes(item.read_bytes())
    log(f"converted {source} -> {output} at {widths}")
    return 0


# --- plan -------------------------------------------------------------------


def plan(widths: list[int]) -> int:
    config = fetch_json("config.json")
    index = fetch_json("model.safetensors.index.json")
    shards = sorted(set(index["weight_map"].values()))
    headers = {shard: fetch_header(shard) for shard in shards}
    header_index = HeaderIndex(headers)
    log(f"{REPO} @ {REVISION[:12]}")
    log(
        f"  {config['num_hidden_layers']} layers, hidden {config['hidden_size']}, "
        f"{config['num_attention_heads']}/{config['num_key_value_heads']} heads, "
        f"{config['num_experts']} experts top-{config['num_experts_per_tok']}"
    )
    log(f"  {len(shards)} shards, {len(header_index.entries)} tensors")
    for width in widths:
        counts: dict[str, int] = {}
        totals: dict[str, int] = {}
        failures: list[str] = []
        for name in header_index.entries:
            entry = header_index.entries[name]
            kind = role(name)
            bits = bits_for(name, width)
            counts[kind] = counts.get(kind, 0) + 1
            try:
                size = bytes_for(entry["shape"], bits)
            except SystemExit:
                failures.append(f"{name}: {entry['shape']}")
                continue
            totals[kind] = totals.get(kind, 0) + size
        log(f"\n  {width}-bit plan")
        for kind in sorted(counts):
            log(f"    {kind:<14} {counts[kind]:>6} tensors  {totals.get(kind, 0) / 1e9:7.2f} GB")
        total = sum(totals.values())
        log(f"    snapshot      ~{total / 1e9:.1f} GB")
        if failures:
            log("    ALIGNMENT FAILURES:")
            for failure in failures[:10]:
                log(f"      {failure}")
        else:
            log(f"    every quantized last dim is a multiple of {GROUP_SIZE}")
    return 0


# --- self test --------------------------------------------------------------


def build_fixture(root: Path) -> None:
    """A tiny Laguna-shaped checkpoint with random weights, two shards."""
    import numpy as np
    from safetensors.numpy import save_file

    rng = np.random.default_rng(11)
    hidden, heads, kv_heads, head_dim = 128, 4, 2, 32
    full_heads, swa_heads = 4, 6
    inter, moe_inter, vocab = 256, 64, 256
    config = {
        "model_type": "laguna",
        "architectures": ["LagunaForCausalLM"],
        "vocab_size": vocab,
        "hidden_size": hidden,
        "intermediate_size": inter,
        "num_hidden_layers": 2,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "max_position_embeddings": 4096,
        "attention_bias": False,
        "rms_norm_eps": 1e-6,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": moe_inter,
        "shared_expert_intermediate_size": moe_inter,
        "norm_topk_prob": True,
        "moe_routed_scaling_factor": 2.5,
        "mlp_only_layers": [0],
        "decoder_sparse_step": 1,
        "tie_word_embeddings": False,
        "gating": "per-head",
        "sliding_window": 8,
        "layer_types": ["full_attention", "sliding_attention"],
        "gating_types": ["per_head", "per_head"],
        "num_attention_heads_per_layer": [full_heads, swa_heads],
        "rope_parameters": {
            "full_attention": {
                "rope_theta": 500000.0,
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 8192,
                "beta_slow": 1.0,
                "beta_fast": 32.0,
                "attention_factor": 1.1,
                "partial_rotary_factor": 0.5,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
            },
        },
        "eos_token_id": [2],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config, indent=1))

    def weight(*shape):
        return rng.normal(scale=0.02, size=shape).astype(np.float32)

    tensors: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": weight(vocab, hidden),
        "lm_head.weight": weight(vocab, hidden),
        "model.norm.weight": np.ones(hidden, dtype=np.float32),
    }
    for layer, layer_heads in ((0, full_heads), (1, swa_heads)):
        q_width = layer_heads * head_dim
        tensors[f"model.layers.{layer}.input_layernorm.weight"] = np.ones(hidden, np.float32)
        tensors[f"model.layers.{layer}.post_attention_layernorm.weight"] = np.ones(
            hidden, np.float32
        )
        tensors[f"model.layers.{layer}.self_attn.q_proj.weight"] = weight(q_width, hidden)
        tensors[f"model.layers.{layer}.self_attn.k_proj.weight"] = weight(
            kv_heads * head_dim, hidden
        )
        tensors[f"model.layers.{layer}.self_attn.v_proj.weight"] = weight(
            kv_heads * head_dim, hidden
        )
        tensors[f"model.layers.{layer}.self_attn.o_proj.weight"] = weight(hidden, q_width)
        tensors[f"model.layers.{layer}.self_attn.g_proj.weight"] = weight(layer_heads, hidden)
        tensors[f"model.layers.{layer}.self_attn.q_norm.weight"] = np.ones(head_dim, np.float32)
        tensors[f"model.layers.{layer}.self_attn.k_norm.weight"] = np.ones(head_dim, np.float32)
    tensors["model.layers.0.mlp.gate_proj.weight"] = weight(inter, hidden)
    tensors["model.layers.0.mlp.up_proj.weight"] = weight(inter, hidden)
    tensors["model.layers.0.mlp.down_proj.weight"] = weight(hidden, inter)
    tensors["model.layers.1.mlp.gate.weight"] = weight(4, hidden)
    tensors["model.layers.1.mlp.experts.e_score_correction_bias"] = np.zeros(4, np.float32)
    tensors["model.layers.1.mlp.shared_expert.gate_proj.weight"] = weight(moe_inter, hidden)
    tensors["model.layers.1.mlp.shared_expert.up_proj.weight"] = weight(moe_inter, hidden)
    tensors["model.layers.1.mlp.shared_expert.down_proj.weight"] = weight(hidden, moe_inter)
    for expert in range(4):
        for role_name, shape in (
            ("gate_proj", (moe_inter, hidden)),
            ("up_proj", (moe_inter, hidden)),
            ("down_proj", (hidden, moe_inter)),
        ):
            tensors[f"model.layers.1.mlp.experts.{expert}.{role_name}.weight"] = weight(*shape)

    # Two shards: resident in one, layer 1's experts in the other, so the test
    # covers a layer whose tensors live in a different file from its peers.
    resident = {
        name: value
        for name, value in tensors.items()
        if ".mlp.experts." not in name or "shared_expert" in name
    }
    experts = {
        name: value
        for name, value in tensors.items()
        if ".mlp.experts." in name and "shared_expert" not in name
    }
    save_file(resident, str(root / "model-00001-of-00002.safetensors"))
    save_file(experts, str(root / "model-00002-of-00002.safetensors"))
    weight_map = {name: "model-00001-of-00002.safetensors" for name in resident}
    weight_map.update({name: "model-00002-of-00002.safetensors" for name in experts})
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
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
        source = root / "source"
        build_fixture(source)
        output = root / "out"
        local_convert(source, output, [4])
        converted_dir = root / "out-4bit"
        config = json.loads((converted_dir / "config.json").read_text())
        require(config["quantization"]["bits"] == 4, "base width is not 4")
        require(config["quantization"]["group_size"] == 64, "group size is not 64")
        require(
            config["quantization"]["model.embed_tokens"]["bits"] == 8,
            "the embedding is not kept at 8-bit",
        )

        layer_file = converted_dir / "layer-01.safetensors"
        converted = load_file(str(layer_file))
        for name in converted:
            if "switch_mlp" not in name:
                require(False, f"unexpected tensor {name} in the layer file")
        stem = "model.layers.1.mlp.switch_mlp.down_proj"
        require(
            converted[stem + ".weight"].shape == (4, 128, 8),
            f"unexpected packed shape {converted[stem + '.weight'].shape}",
        )
        require(
            converted[stem + ".scales"].shape == (4, 128, 1),
            f"unexpected scales shape {converted[stem + '.scales'].shape}",
        )
        require(converted[stem + ".scales"].dtype == ml_dtypes.bfloat16, "scales are not bf16")

        # Round trip one expert's down projection against the checkpoint.
        original = load_file(str(source / "model-00002-of-00002.safetensors"))
        name = "model.layers.1.mlp.experts.0.down_proj.weight"
        packed = converted[stem + ".weight"][0]
        scale = converted[stem + ".scales"][0]
        bias = converted[stem + ".biases"][0]
        lanes = 8
        words = packed
        unpacked = np.zeros((words.shape[0], words.shape[1] * lanes), np.uint32)
        for lane in range(lanes):
            unpacked[:, lane::lanes] = (words >> np.uint32(4 * lane)) & np.uint32(0x0F)
        dequantized = unpacked.astype(np.float32) * np.repeat(
            scale.astype(np.float32), 64, axis=-1
        ) + np.repeat(bias.astype(np.float32), 64, axis=-1)
        error = float(np.abs(dequantized - original[name]).max())
        require(error < 0.02, f"4-bit round trip error {error}")

        resident = load_file(str(converted_dir / "resident.safetensors"))
        kept = "model.layers.1.mlp.experts.e_score_correction_bias"
        require(resident[kept].dtype == ml_dtypes.bfloat16, "the router bias is not bf16")
        require(
            np.array_equal(resident[kept].astype(np.float32), original[kept]),
            "the router bias changed",
        )
        require(
            resident["model.layers.1.mlp.gate.weight"].dtype == ml_dtypes.bfloat16,
            "the router weight is not bf16",
        )
        require(
            resident["model.embed_tokens.weight"].dtype == np.uint32,
            "the embedding was not quantized",
        )
        require(
            resident["model.embed_tokens.scales"].dtype == ml_dtypes.bfloat16,
            "the embedding scales are not bf16",
        )

        index = json.loads((converted_dir / "model.safetensors.index.json").read_text())
        require(
            index["weight_map"]["model.layers.1.mlp.switch_mlp.up_proj.weight"]
            == "layer-01.safetensors",
            "the layer index points at the wrong shard",
        )
        require(
            "model.layers.1.mlp.experts.0.gate_proj.weight" not in index["weight_map"],
            "a per-expert tensor survived into the snapshot",
        )

        # The 8-bit build is the same tensors at a wider group: no embedding
        # override (the embedding is already 8-bit), half the packed words.
        local_convert(source, root / "out8", [8])
        eight_dir = root / "out8-8bit"
        eight_config = json.loads((eight_dir / "config.json").read_text())
        require(eight_config["quantization"]["bits"] == 8, "base width is not 8")
        require(
            "model.embed_tokens" not in eight_config["quantization"],
            "the 8-bit build carries a pointless embedding override",
        )
        eight = load_file(str(eight_dir / "layer-01.safetensors"))
        require(
            eight[stem + ".weight"].shape == (4, 128, 16),
            f"unexpected 8-bit packed shape {eight[stem + '.weight'].shape}",
        )
        require(eight[stem + ".weight"].dtype == np.uint32, "the 8-bit expert is not packed")
        log("Laguna converter self-test passed")
        return 0


# --- CLI --------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--from-hf", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bits", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--work", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    for width in arguments.bits:
        if width not in (4, 8):
            parser.error(f"unsupported width {width}")
    if arguments.self_test:
        return self_test()
    if arguments.plan:
        return plan(arguments.bits)
    if arguments.from_hf:
        if arguments.output is None:
            parser.error("--from-hf needs --output")
        work = arguments.work or arguments.output.with_name(arguments.output.name + "-work")
        return remote_convert(arguments.output, work, arguments.bits, arguments.resume)
    if arguments.source is not None:
        if arguments.output is None:
            parser.error("--source needs --output")
        return local_convert(arguments.source, arguments.output, arguments.bits)
    parser.error("choose --plan, --from-hf, --source or --self-test")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
