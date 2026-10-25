#!/usr/bin/env python3
"""Plan and convert NVIDIA Nemotron 3.5 Bolt 74B-A9B from its 16-bit release.

The checkpoint (`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16`, 149.77 GB in 63
shards) is a Mamba-2 + LatentMoE + attention hybrid, not a Qwen variant:

  - 88 layers: 40 mamba, 40 MoE, 8 attention (7, 16, 25, 36, 47, 58, 69, 78)
  - Mamba-2: d_inner 8192, 128 heads x 64, state 128, 8 groups, conv 4
  - LatentMoE: 240-496 routed experts per layer (13,824 total), top-9,
    latent 1024, per-expert width 1536-2560, shared expert 5376, relu^2
  - Attention: 32 query / 2 KV heads, head dim 128, no rotary

This converter writes the `.ssdai` install directly -- both widths from one
download of the source. An affine safetensors snapshot that `TinyTitanRepack`
then repacks was rejected for this model: it doubles peak disk for an 8-bit
build (85 GB snapshot beside an 85 GB install) and would need a second 150 GB
download, while the runtime only ever reads the install. The format contract is
`docs/nemotron-ssdai-writer-spec.md`.

    tools/prepare_nemotron.py --plan
    tools/prepare_nemotron.py --install \
        --out4 models/nemotron-3.5-bolt_74B_A9B_4Bit \
        --out8 models/nemotron-3.5-bolt_74B_A9B_8Bit \
        --work .build/nemotron-src

Source shards are streamed one at a time and deleted after conversion, so peak
disk is both installs plus one or two shards. `--resume` adopts the shards a
killed run already converted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import ml_dtypes
import numpy as np
from safetensors import safe_open

REPO = "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16"
REVISION = "e09a4cedb1787db1760375ee5539f222cae0c672"
BASE = f"https://huggingface.co/{REPO}/resolve/{REVISION}"
GROUP_SIZE = 64
PAGE = 16_384
RETRY = ("--retry", "5", "--retry-delay", "2", "--retry-all-errors")
CHUNK = 64 * 1024 * 1024

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
EMBEDDING = "language_model.backbone.embeddings.weight"
HEAD = "language_model.lm_head.weight"
LAYERS = 88


# --- source access ----------------------------------------------------------


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


def fetch_optional(remote: str) -> bytes | None:
    result = subprocess.run(
        ["curl", "-sfL", "--max-time", "60", *RETRY, f"{BASE}/{remote}"],
        capture_output=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 and result.stdout else None


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


_SIZE_CACHE: dict[str, int] = {}


def remote_size(shard: str) -> int:
    if shard in _SIZE_CACHE:
        return _SIZE_CACHE[shard]
    result = subprocess.run(
        [*("curl", "-sfIL", "--max-time", "60", *RETRY), f"{BASE}/{shard}"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"cannot stat {shard}")
    for line in result.stdout.decode(errors="replace").splitlines():
        if line.lower().startswith("content-length:"):
            _SIZE_CACHE[shard] = int(line.split(":", 1)[1].strip())
    if shard not in _SIZE_CACHE:
        raise SystemExit(f"{shard}: no Content-Length")
    return _SIZE_CACHE[shard]


def download(shard: str, work: Path, log=print, connections: int = 6) -> Path:
    """Fetch one shard with parallel ranged requests, resuming chunk by chunk.

    The host's link is throttled per connection, so one curl gets ~1.4 MB/s and
    six get ~4.3 MB/s on the same shard. A chunk is recorded as done only after
    its bytes are written, so a killed run resumes at the first missing chunk
    rather than trusting a pre-truncated file's length.
    """
    destination = work / shard
    expected = remote_size(shard)
    state_path = work / f"{shard}.chunks.json"
    completed: set[int] = set()
    if state_path.exists():
        completed = {int(value) for value in json.loads(state_path.read_text())}
    fd = os.open(destination, os.O_RDWR | os.O_CREAT, 0o644)
    os.ftruncate(fd, expected)
    started = time.monotonic()
    done = sum(
        min(CHUNK, expected - offset)
        for offset in completed
    )
    lock = __import__("threading").Lock()

    def fetch(offset: int, end: int) -> None:
        nonlocal done
        cursor = offset
        while cursor <= end:
            result = subprocess.run(
                [
                    "curl", "-sfL", "--http1.1", "--max-time", "600", *RETRY,
                    "-r", f"{cursor}-{end}", f"{BASE}/{shard}",
                ],
                capture_output=True,
                check=False,
            )
            if result.returncode != 0 or not result.stdout:
                raise SystemExit(
                    f"cannot fetch {shard} bytes {cursor}-{end}: "
                    f"{result.stderr.decode(errors='replace').strip()}")
            pwrite_all(fd, result.stdout, cursor)
            cursor += len(result.stdout)
        with lock:
            completed.add(offset)
            done += end - offset + 1
            state_path.write_text(json.dumps(sorted(completed)))
            elapsed = time.monotonic() - started
            log(
                f"    {shard}: {done / 1e9:.2f}/{expected / 1e9:.2f} GB at "
                f"{done / 1e6 / max(elapsed, 1e-6):.1f} MB/s "
                f"({len(completed)}/{len(spans)} chunks)")

    spans = [
        (offset, min(offset + CHUNK, expected) - 1)
        for offset in range(0, expected, CHUNK)
    ]
    pending = [(start, end) for start, end in spans if start not in completed]
    try:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=connections) as pool:
            futures = [pool.submit(fetch, start, end) for start, end in pending]
            for future in futures:
                future.result()
    finally:
        os.fsync(fd)
        os.close(fd)
    if destination.stat().st_size != expected:
        raise SystemExit(
            f"{shard}: size {destination.stat().st_size} != expected {expected}")
    state_path.unlink(missing_ok=True)
    elapsed = time.monotonic() - started
    log(
        f"    fetched {shard} ({expected / 1e9:.2f} GB, "
        f"{expected / 1e6 / max(elapsed, 1e-6):.1f} MB/s)")
    return destination


# --- geometry ---------------------------------------------------------------


def elements(shape: list[int]) -> int:
    total = 1
    for dimension in shape:
        total *= dimension
    return total


def affine_bytes(count: int, bits: int) -> int:
    return count * bits // 8 + (count // GROUP_SIZE) * 4


ROUTED = re.compile(
    r"^language_model\.backbone\.layers\.(\d+)\.mixer\.experts\.(\d+)\."
    r"(up_proj|down_proj)\.weight$"
)
VISION = ("vision_model.", "mlp1.")


def is_vision(name: str) -> bool:
    return name.startswith(VISION)


def width_of(name: str, width: int) -> int | None:
    """The width a resident tensor is carried at, or None for the source dtype."""
    if name.endswith("conv1d.weight"):
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
    if name in (EMBEDDING, HEAD):
        return 8
    return width


def validate_config(config: dict) -> dict:
    if config.get("model_type") != "nemotron_h_omni":
        raise SystemExit(
            f"config model_type is {config.get('model_type')!r}, expected nemotron_h_omni")
    text = config.get("llm_config")
    if not isinstance(text, dict):
        raise SystemExit("config has no llm_config (this checkpoint nests its text tower there)")
    problems = []
    for key, expected in EXPECTED.items():
        actual = text.get(key)
        if actual != expected:
            problems.append(f"{key}: config {actual!r} != expected {expected!r}")
    kinds = text.get("layers_block_type")
    if not isinstance(kinds, list) or len(kinds) != LAYERS:
        problems.append(f"layers_block_type is not {LAYERS} entries")
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


def load_source(cache: Path | None = None) -> tuple[dict, dict, list[str], dict[str, dict]]:
    config = fetch_json("config.json")
    text = validate_config(config)
    index = fetch_json("model.safetensors.index.json")
    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    if cache is not None and cache.exists():
        cached = json.loads(cache.read_text())
        if cached.get("shards") == shards:
            return text, index, shards, cached["headers"]
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
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"shards": shards, "headers": headers}))
    return text, index, shards, headers


def expert_census(text: dict, headers: dict[str, dict]) -> dict[int, int]:
    counts: Counter[int] = Counter()
    for name in headers:
        match = ROUTED.match(name)
        if match:
            counts[int(match.group(1))] += 1
    layers = {layer: value // 2 for layer, value in counts.items()}
    block = [
        entry["n_routed_experts"]
        for entry in text["block_configs"]
        if entry["block_type"] == "moe"
    ]
    if len(layers) != 40:
        raise SystemExit(f"expected 40 MoE layers with routed experts, found {len(layers)}")
    if [layers[i] for i in sorted(layers)] != block:
        raise SystemExit("per-layer expert counts disagree with block_configs")
    return layers


def plan() -> int:
    cache = Path(".build/nemotron-src/headers.json")
    text, index, shards, headers = load_source(cache)
    layers = expert_census(text, headers)
    print(f"{REPO} @ {REVISION[:12]}")
    print(f"  {len(shards)} shards, {len(headers)} tensors")
    print(f"  88 layers: 40 mamba / 40 moe / 8 attention at {list(ATTENTION_LAYERS)}")
    print(
        f"  {sum(layers.values())} routed experts across {len(layers)} layers "
        f"({sorted(set(layers.values()))}), top-9"
    )
    unaligned = []
    for name, meta in headers.items():
        if is_vision(name) or width_of(name, 4) is None:
            continue
        if isinstance(meta.get("shape"), list) and meta["shape"][-1] % GROUP_SIZE:
            unaligned.append((name, meta["shape"]))
    if unaligned:
        raise SystemExit(
            f"{len(unaligned)} tensors are not group-{GROUP_SIZE} aligned; first: {unaligned[:5]}"
        )
    for width in (4, 8):
        resident = 0
        experts = 0
        for name, meta in headers.items():
            if is_vision(name):
                continue
            if ROUTED.match(name):
                experts += affine_bytes(elements(meta["shape"]), width)
                continue
            bits = width_of(name, width)
            if bits is None:
                resident += elements(meta["shape"]) * (
                    4 if meta["dtype"] == "F32" else 2)
            else:
                resident += affine_bytes(elements(meta["shape"]), bits)
        # The install format carries one expert stride per install; the
        # verifier requires every layer file to be count * that stride, so the
        # widest expert blob is charged to every layer.
        global_stride = 0
        for layer, count in layers.items():
            expert_width = text["block_configs"][layer]["moe_intermediate_size"]
            blob = 0
            for shape in ((expert_width, text["moe_latent_size"]),
                          (text["moe_latent_size"], expert_width)):
                rows, cols = shape
                blob += rows * cols * width // 8
                blob += rows * (cols // GROUP_SIZE) * 4
            global_stride = max(global_stride, round_up(blob))
        padded = sum(count * global_stride for count in layers.values())
        print(
            f"  {width}-bit: {(resident + padded) / 1e9:.2f} GB = "
            f"resident {resident / 1e9:.2f} GB + routed experts "
            f"{padded / 1e9:.2f} GB (stride {global_stride / 1e6:.2f} MB, "
            f"{(padded - experts) / 1e9:.2f} GB page padding)"
        )
    print(f"  index total_size {index['metadata']['total_size'] / 1e9:.2f} GB")
    return 0


# --- quantisation -----------------------------------------------------------


def quantize_affine(value: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Affine group-64: packed U32 weights, bf16 scale and bf16 bias per group.

    Same formula as prepare_qwen38.py / prepare_ornith_mtp.py: the group's
    minimum is its bias, `(max - min) / (2**bits - 1)` its scale, and lanes are
    packed least-significant first into 32-bit words.
    """
    value = value.astype(np.float32)
    if value.shape[-1] % GROUP_SIZE:
        raise ValueError(f"last dimension {value.shape[-1]} is not group-aligned")
    shape = (*value.shape[:-1], value.shape[-1] // GROUP_SIZE, GROUP_SIZE)
    grouped = value.reshape(shape)
    bias = grouped.min(axis=-1).astype(np.float32)
    high = grouped.max(axis=-1)
    levels = (1 << bits) - 1
    scale = np.where(high == bias, np.float32(1), (high - bias) / levels).astype(np.float32)
    scale_bf16 = scale.astype(ml_dtypes.bfloat16)
    bias_bf16 = bias.astype(ml_dtypes.bfloat16)
    quantized = (
        np.rint((grouped - bias[..., None]) / scale[..., None])
        .clip(0, levels)
        .astype(np.uint32)
        .reshape(value.shape)
    )
    lanes = 32 // bits
    words = quantized.reshape(*quantized.shape[:-1], quantized.shape[-1] // lanes, lanes)
    packed = np.zeros(words.shape[:-1], dtype=np.uint32)
    for lane in range(lanes):
        packed |= words[..., lane] << np.uint32(bits * lane)
    return packed, scale_bf16, bias_bf16


def bf16_bytes(value: np.ndarray) -> bytes:
    return np.ascontiguousarray(value).astype(ml_dtypes.bfloat16).tobytes()


def bf16_uint16(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(value).astype(ml_dtypes.bfloat16).view(np.uint16)


def pwrite_all(fd: int, data: bytes, offset: int) -> None:
    view = memoryview(data)
    while view:
        written = os.pwrite(fd, view, offset)
        if written <= 0:
            raise OSError("pwrite made no progress")
        view = view[written:]
        offset += written


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(64 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def round_up(value: int, page: int = PAGE) -> int:
    return ((value + page - 1) // page) * page


# --- install plan -----------------------------------------------------------


@dataclass
class ResidentTensor:
    name: str
    shape: tuple[int, ...]
    source_dtype: str
    bits: int | None
    offset: int = 0
    size: int = 0
    scale_offset: int = 0
    scale_size: int = 0
    bias_offset: int = 0
    bias_size: int = 0

    @property
    def index_dtype(self) -> int:
        if self.bits is not None:
            return 0
        return 1 if self.source_dtype == "BF16" else 3


@dataclass
class ExpertTensor:
    role: str
    shape: tuple[int, int]
    bits: int
    offset: int
    size: int
    scale_offset: int
    scale_size: int
    bias_offset: int
    bias_size: int


@dataclass
class LayerPlan:
    layer: int
    kind: str
    experts: int
    stride: int
    file_size: int
    tensors: list[ExpertTensor] = field(default_factory=list)

    @property
    def file_name(self) -> str:
        return f"layer_{self.layer:02d}.bin"


def build_install_plan(
    text: dict, headers: dict[str, dict], layers: dict[int, int], bits: int
) -> tuple[list[ResidentTensor], dict[int, LayerPlan], int]:
    resident_names = sorted(
        (name for name in headers if not is_vision(name) and not ROUTED.match(name)),
        key=resident_order,
    )
    index_size = round_up(
        24 + len(resident_names) * 72
        + sum(len(name.encode()) for name in resident_names))
    cursor = index_size
    resident: list[ResidentTensor] = []
    for name in resident_names:
        shape = tuple(headers[name]["shape"])
        width = width_of(name, bits)
        entry = ResidentTensor(
            name=name, shape=shape, source_dtype=headers[name]["dtype"], bits=width)
        if width is None:
            entry.size = elements(shape) * (4 if entry.source_dtype == "F32" else 2)
            entry.offset = cursor
            cursor += entry.size
        else:
            rows, cols = shape[0], shape[-1]
            entry.size = rows * cols * width // 8
            entry.scale_size = rows * (cols // GROUP_SIZE) * 2
            entry.bias_size = entry.scale_size
            entry.offset = cursor
            cursor += entry.size
            entry.scale_offset = cursor
            cursor += entry.scale_size
            entry.bias_offset = cursor
            cursor += entry.bias_size
        resident.append(entry)

    # First pass: every layer's own blob size.
    per_layer: dict[int, LayerPlan] = {}
    for layer, kind in enumerate(text["layers_block_type"]):
        if layer not in layers:
            per_layer[layer] = LayerPlan(
                layer=layer, kind=kind, experts=0, stride=0, file_size=0)
            continue
        block = text["block_configs"][layer]
        expert_width = block["moe_intermediate_size"]
        tensors: list[ExpertTensor] = []
        cursor = 0
        for role, shape in (
            ("up", (expert_width, text["moe_latent_size"])),
            ("down", (text["moe_latent_size"], expert_width)),
        ):
            rows, cols = shape
            tensor = ExpertTensor(
                role=role, shape=shape, bits=bits, offset=cursor,
                size=rows * cols * bits // 8, scale_offset=0,
                scale_size=rows * (cols // GROUP_SIZE) * 2,
                bias_offset=0, bias_size=0)
            cursor += tensor.size
            tensor.scale_offset = cursor
            cursor += tensor.scale_size
            tensor.bias_offset = cursor
            tensor.bias_size = tensor.scale_size
            cursor += tensor.bias_size
            tensors.append(tensor)
        per_layer[layer] = LayerPlan(
            layer=layer, kind=kind, experts=layers[layer],
            stride=round_up(cursor), file_size=0, tensors=tensors)

    # The format's verifier accepts one expert stride per install and requires
    # every layer file to be exactly count * that stride.
    stride = max(plan.stride for plan in per_layer.values())
    for plan_layer in per_layer.values():
        plan_layer.stride = stride if plan_layer.experts else 0
        plan_layer.file_size = plan_layer.experts * plan_layer.stride
    return resident, per_layer, stride


def resident_order(name: str) -> tuple:
    if name == EMBEDDING:
        return (0, 0, name)
    if name == HEAD:
        return (3, 0, name)
    match = re.match(r"language_model\.backbone\.layers\.(\d+)\.(.*)$", name)
    if match:
        return (1, int(match.group(1)), match.group(2))
    return (2, 0, name)


class Install:
    """One width's `.ssdai` payload, written in place while the source streams."""

    def __init__(self, directory: Path, bits: int, model_id: str, plan_data) -> None:
        self.directory = directory
        self.bits = bits
        self.model_id = model_id
        self.resident_plan, self.layer_plans, self.global_stride = plan_data
        self.resident = {entry.name: entry for entry in self.resident_plan}
        (directory / "packed_experts").mkdir(parents=True, exist_ok=True)
        (directory / "tokenizer").mkdir(parents=True, exist_ok=True)
        self.index_size = round_up(
            24 + len(self.resident_plan) * 72
            + sum(len(entry.name.encode()) for entry in self.resident_plan))
        last = max(
            (entry.offset + entry.size + entry.scale_size + entry.bias_size
             for entry in self.resident_plan),
            default=self.index_size)
        self.resident_size = last - self.index_size
        self.weights_fd = os.open(
            directory / "model_weights.bin",
            os.O_RDWR | os.O_CREAT | os.O_TRUNC,
            0o644,
        )
        os.ftruncate(self.weights_fd, self.index_size + self.resident_size)
        self.layer_fds: dict[int, int] = {}
        for layer, layer_plan in self.layer_plans.items():
            if layer_plan.experts:
                fd = os.open(
                    directory / "packed_experts" / layer_plan.file_name,
                    os.O_RDWR | os.O_CREAT | os.O_TRUNC,
                    0o644,
                )
                os.ftruncate(fd, layer_plan.file_size)
                self.layer_fds[layer] = fd

    def close(self) -> None:
        for fd in [self.weights_fd, *self.layer_fds.values()]:
            os.fsync(fd)
            os.close(fd)
        self.layer_fds.clear()

    # -- writes -------------------------------------------------------------

    def write_resident(self, name: str, value: np.ndarray) -> None:
        entry = self.resident[name]
        if entry.bits is None:
            if entry.source_dtype == "F32":
                data = np.ascontiguousarray(value).astype(np.float32).tobytes()
            else:
                data = bf16_bytes(value)
            if len(data) != entry.size:
                raise SystemExit(f"{name}: bytes {len(data)} != planned {entry.size}")
            pwrite_all(self.weights_fd, data, entry.offset)
            return
        self._write_quantized(
            self.weights_fd, value, entry.bits,
            entry.offset, entry.scale_offset, entry.bias_offset,
            entry.size, name,
        )

    def write_expert(self, layer: int, expert: int, role: str, value: np.ndarray) -> None:
        plan = self.layer_plans[layer]
        tensor = next(item for item in plan.tensors if item.role == role)
        base = expert * plan.stride
        if value.shape != tensor.shape:
            raise SystemExit(
                f"layer {layer} expert {expert} {role}: shape {value.shape} != {tensor.shape}")
        self._write_quantized(
            self.layer_fds[layer], value, tensor.bits,
            base + tensor.offset, base + tensor.scale_offset, base + tensor.bias_offset,
            tensor.size, f"layer {layer} expert {expert} {role}",
        )

    def _write_quantized(
        self, fd: int, value: np.ndarray, bits: int,
        weight_offset: int, scale_offset: int, bias_offset: int,
        weight_bytes: int, label: str,
    ) -> None:
        rows, cols = value.shape[0], value.shape[-1]
        packed_columns = cols * bits // 32
        rows_per_chunk = max(1, (32 * 1024 * 1024) // max(cols * 4, 1))
        produced = 0
        for start in range(0, rows, rows_per_chunk):
            chunk = np.ascontiguousarray(value[start:start + rows_per_chunk])
            packed, scale, bias = quantize_affine(chunk, bits)
            pwrite_all(
                fd, np.ascontiguousarray(packed).tobytes(),
                weight_offset + start * packed_columns * 4)
            pwrite_all(
                fd, bf16_uint16(scale).tobytes(),
                scale_offset + start * (cols // GROUP_SIZE) * 2)
            pwrite_all(
                fd, bf16_uint16(bias).tobytes(),
                bias_offset + start * (cols // GROUP_SIZE) * 2)
            produced += packed.size * 4
        if produced != weight_bytes:
            raise SystemExit(f"{label}: wrote {produced} weight bytes, planned {weight_bytes}")

    # -- finalise -----------------------------------------------------------

    def finish(self, text: dict, index_sha: str, source_config: dict) -> dict:
        pwrite_all(self.weights_fd, self._resident_index(), 0)
        os.fsync(self.weights_fd)
        self.close()
        self._layout()
        self._tokenizer(source_config)
        files = self._hash_files()
        manifest = self._manifest(text, index_sha, files)
        (self.directory / "manifest.json").write_text(
            json.dumps(manifest, indent=1, sort_keys=True))
        return {
            "directory": str(self.directory),
            "bits": self.bits,
            "modelID": self.model_id,
            "bytes": sum(entry["size"] for entry in files.values()),
            "files": len(files),
        }

    def _resident_index(self) -> bytes:
        header = struct.pack(
            "<QQQ", self.index_size, self.resident_size, len(self.resident_plan))
        entries = bytearray()
        strings = bytearray()
        name_base = 24 + len(self.resident_plan) * 72
        for entry in self.resident_plan:
            encoded = entry.name.encode()
            name_offset = name_base + len(strings)
            strings += encoded
            shape = list(entry.shape) + [0] * (4 - len(entry.shape))
            entries += struct.pack(
                "<IHBBQQIIIIQQQQ",
                name_offset, len(encoded), entry.index_dtype, 0,
                entry.offset, entry.size,
                shape[0], shape[1], shape[2], shape[3],
                entry.scale_offset, entry.scale_size,
                entry.bias_offset, entry.bias_size,
            )
        body = header + bytes(entries) + bytes(strings)
        if len(body) > self.index_size:
            raise SystemExit(f"resident index {len(body)} exceeds planned {self.index_size}")
        return body + b"\0" * (self.index_size - len(body))

    def _layout(self) -> None:
        layers = []
        for layer, layer_plan in self.layer_plans.items():
            experts = []
            if layer_plan.experts:
                for expert in range(layer_plan.experts):
                    tensors = {}
                    for tensor in layer_plan.tensors:
                        tensors[tensor.role] = {
                            "offset": tensor.offset,
                            "size": tensor.size,
                            "dtype": "U32",
                            "shape": list(tensor.shape),
                            "bits": tensor.bits,
                        }
                        tensors[f"{tensor.role}_scales"] = {
                            "offset": tensor.scale_offset,
                            "size": tensor.scale_size,
                            "dtype": "BF16",
                            "shape": [tensor.shape[0], tensor.shape[-1] // GROUP_SIZE],
                        }
                        tensors[f"{tensor.role}_biases"] = {
                            "offset": tensor.bias_offset,
                            "size": tensor.bias_size,
                            "dtype": "BF16",
                            "shape": [tensor.shape[0], tensor.shape[-1] // GROUP_SIZE],
                        }
                    experts.append({
                        "expert": expert,
                        "offset": expert * layer_plan.stride,
                        "size": layer_plan.stride,
                        "tensors": tensors,
                    })
            layers.append({
                "layer": layer,
                "file": layer_plan.file_name,
                "experts": experts,
                "expertsPerLayer": layer_plan.experts,
            })
        document = {
            "expertStride": self.global_stride,
            "numLayers": len(self.layer_plans),
            "expertsPerLayer": max(plan.experts for plan in self.layer_plans.values()),
            "layers": layers,
        }
        (self.directory / "packed_experts" / "layout.json").write_text(
            json.dumps(document, sort_keys=True))

    def _tokenizer(self, source_config: dict) -> None:
        target = self.directory / "tokenizer"
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
            (target / name).write_bytes(fetch(name))
        for name in ("special_tokens_map.json", "generation_config.json"):
            data = fetch_optional(name)
            if data:
                (target / name).write_bytes(data)
        (target / "config.json").write_text(json.dumps(source_config, indent=1))

    def _hash_files(self) -> dict[str, dict[str, object]]:
        files: dict[str, dict[str, object]] = {}
        for path in sorted(self.directory.rglob("*")):
            if not path.is_file() or path.name == "manifest.json":
                continue
            relative = path.relative_to(self.directory).as_posix()
            files[relative] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
        return files

    def _manifest(self, text: dict, index_sha: str, files: dict) -> dict:
        num_layers = len(self.layer_plans)
        expert_counts = [
            self.layer_plans[layer].experts if layer in self.layer_plans else 0
            for layer in range(num_layers)
        ]
        top_k = max(
            entry["num_experts_per_tok"]
            for entry in text["block_configs"] if entry["block_type"] == "moe")
        arch = {
            "hiddenSize": text["hidden_size"],
            "ffnIntermediate": text["moe_shared_expert_intermediate_size"],
            "moeIntermediateSize": max(
                entry["moe_intermediate_size"]
                for entry in text["block_configs"] if entry["block_type"] == "moe"),
            "numHeads": text["num_attention_heads"],
            "numKVHeads": text["num_key_value_heads"],
            "numFullKVHeads": text["num_key_value_heads"],
            "headDim": text["head_dim"],
            "fullHeadDim": text["head_dim"],
            "vocabSize": text["vocab_size"],
            "slidingWindow": 0,
            "finalLogitSoftcap": 0,
            "ropeTheta": 10000.0,
            "fullRopeTheta": 10000.0,
            "partialRotaryFactor": 0,
            "numLayers": num_layers,
            "numExperts": max(expert_counts),
            "topKExperts": top_k,
            "tieWordEmbeddings": False,
            "attentionKEqV": False,
            "hiddenActivation": "relu2",
            "fullAttentionLayerMask": [
                2 if kind == "mamba" else 1 if kind == "attention" else 0
                for kind in text["layers_block_type"]
            ],
            "family": "nemotron_h",
            "attnOutputGate": False,
            "attentionScale": text["head_dim"] ** -0.5,
            "embeddingScaledBySqrtHidden": False,
            "routerScaled": True,
            "ffnSandwichNorms": False,
            "sharedExpertGated": False,
            "ropeNeoxSubdim": False,
            "linearNumKHeads": 0,
            "linearNumVHeads": 0,
            "linearKeyHeadDim": 0,
            "linearValueHeadDim": 0,
            "linearConvKernelSize": 0,
            "routerNormTopK": True,
            "quantGroupSize": GROUP_SIZE,
            "routerScaling": text["routed_scaling_factor"],
            "routerHasBias": True,
            "nemotronH": {
                "layerKinds": list(text["layers_block_type"]),
                "attentionLayers": list(ATTENTION_LAYERS),
                "mambaNumHeads": text["mamba_num_heads"],
                "mambaHeadDim": text["mamba_head_dim"],
                "ssmStateSize": text["ssm_state_size"],
                "mambaGroups": text["n_groups"],
                "mambaConvKernel": text["conv_kernel"],
                "mambaIntermediate": text["mamba_num_heads"] * text["mamba_head_dim"],
                "moeLatentSize": text["moe_latent_size"],
                "moeSharedIntermediate": text["moe_shared_expert_intermediate_size"],
                "routerGroupCount": text["n_group"],
                "routerTopKGroup": text["topk_group"],
                "routedScalingFactor": text["routed_scaling_factor"],
                "expertCounts": expert_counts,
                "moeIntermediateSizes": [
                    entry.get("moe_intermediate_size", 0)
                    for entry in text["block_configs"]
                ],
                "moeNormTopK": bool(text["norm_topk_prob"]),
                "normEpsilon": text["layer_norm_epsilon"],
                "mambaEpsilon": text["layer_norm_epsilon"],
                "mambaHiddenActivation": text["mamba_hidden_act"],
                "timeStepMin": text["time_step_min"],
                "mambaConvBias": bool(text["use_conv_bias"]),
            },
        }
        slots = {
            "embedding": 8,
            "attention": self.bits,
            "router": 8,
            "sharedExpert": self.bits,
            "routedExpert": self.bits,
        }
        quant: dict[str, dict] = {
            key: {
                "weightBits": value, "scheme": "affine",
                "scaleType": "BF16", "biasType": "BF16",
                "groupSize": GROUP_SIZE,
            }
            for key, value in slots.items()
        }
        for entry in self.resident_plan:
            if entry.bits is None:
                continue
            stem = entry.name[: -len(".weight")] if entry.name.endswith(".weight") else entry.name
            quant[stem] = {
                "weightBits": entry.bits, "scheme": "affine",
                "scaleType": "BF16", "biasType": "BF16",
                "groupSize": GROUP_SIZE,
            }
        experts_per_layer = max(expert_counts)
        return {
            "magic": "SSDAI",
            "versionMajor": 1,
            "versionMinor": 1,
            "flags": {
                "streamingPresent": True,
                "turboQuantKV": False,
                "aneSharedExpert": False,
            },
            "modelID": self.model_id,
            "sourceSnapshotHash": f"sha256:{index_sha}",
            "arch": arch,
            "quant": quant,
            "files": files,
            "expertsPerLayer": experts_per_layer,
            "numLayers": num_layers,
            "expertStride": self.global_stride,
        }


# --- install orchestration --------------------------------------------------


def install(args) -> int:
    out4 = Path(args.out4)
    out8 = Path(args.out8)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    for directory in (out4, out8):
        if directory.exists() and any(directory.iterdir()) and not args.resume:
            raise SystemExit(f"{directory} is not empty; pass --resume to adopt it")
        directory.mkdir(parents=True, exist_ok=True)

    text, index, shards, headers = load_source(work / "headers.json")
    layers = expert_census(text, headers)
    source_config = fetch_json("config.json")
    index_sha = hashlib.sha256(fetch("model.safetensors.index.json")).hexdigest()

    plans = {
        4: build_install_plan(text, headers, layers, 4),
        8: build_install_plan(text, headers, layers, 8),
    }
    expected = {
        bits: sum(entry.size + entry.scale_size + entry.bias_size for entry in plan_data[0])
        + sum(layer.file_size for layer in plan_data[1].values())
        for bits, plan_data in plans.items()
    }
    needed = sum(expected.values()) + 2 * 4 * 1_073_741_824
    free = shutil.disk_usage(work).free
    print(
        f"installs need {sum(expected.values()) / 1e9:.2f} GB, "
        f"free {free / 1e9:.2f} GB, reserve 8 GB")
    if free < needed:
        raise SystemExit(
            f"not enough disk: need {needed / 1e9:.2f} GB, have {free / 1e9:.2f} GB")

    installs = {
        4: Install(out4, 4, "nemotron-3.5-bolt-74b-a9b", plans[4]),
        8: Install(out8, 8, "nemotron-3.5-bolt-74b-a9b", plans[8]),
    }
    progress_path = work / "progress.json"
    done: set[str] = set()
    if args.resume and progress_path.exists():
        done = set(json.loads(progress_path.read_text()).get("done", []))
        print(f"resuming: {len(done)} shards already converted")

    total_tensors = len(headers)
    written = 0
    started = time.monotonic()
    try:
        for number, shard in enumerate(shards, start=1):
            if args.limit_shards and number > args.limit_shards:
                break
            if shard in done:
                written += sum(
                    1 for name in headers if index["weight_map"][name] == shard)
                continue
            print(f"  [{number}/{len(shards)}] {shard}")
            path = download(shard, work)
            with safe_open(str(path), framework="np") as source:
                for name in source.keys():
                    if name not in headers or is_vision(name):
                        continue
                    value = source.get_tensor(name)
                    match = ROUTED.match(name)
                    if match:
                        layer = int(match.group(1))
                        expert = int(match.group(2))
                        role = match.group(3)
                        for target in installs.values():
                            target.write_expert(layer, expert, role, value)
                    else:
                        for target in installs.values():
                            target.write_resident(name, value)
                    written += 1
            path.unlink()
            done.add(shard)
            progress_path.write_text(json.dumps({"done": sorted(done)}))
            elapsed = time.monotonic() - started
            rate = written / max(elapsed, 1e-6)
            print(
                f"    {written}/{total_tensors} tensors, {rate:.0f} tensors/s, "
                f"~{((total_tensors - written) / rate) / 60:.1f} min left")
    except BaseException:
        for target in installs.values():
            for fd in [target.weights_fd, *target.layer_fds.values()]:
                try:
                    os.close(fd)
                except OSError:
                    pass
        raise

    results = []
    for bits, target in installs.items():
        print(f"  finalising {bits}-bit")
        results.append(target.finish(text, index_sha, source_config))
    progress_path.write_text(json.dumps({"done": sorted(done), "complete": True}))
    print(json.dumps(results, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="validate and measure; download no weights")
    parser.add_argument("--install", action="store_true", help="convert both widths")
    parser.add_argument("--out4", type=str)
    parser.add_argument("--out8", type=str)
    parser.add_argument("--work", type=str, default=".build/nemotron-src")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--limit-shards", type=int, default=0,
        help="convert only the first N shards, to check the install format")
    args = parser.parse_args()
    if args.plan:
        return plan()
    if args.install:
        if not args.out4 or not args.out8:
            parser.error("--install needs --out4 and --out8")
        return install(args)
    parser.error("choose --plan or --install")
    return 2


if __name__ == "__main__":
    sys.exit(main())
