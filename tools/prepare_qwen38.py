#!/usr/bin/env python3
"""Convert Qwen/Qwen3.8-Flash-Next into an affine snapshot TinyTitanRepack imports.

Why this exists: the installed 4-bit model came from a third-party MLX repack.
TinyTitan does not need one. It needs an affine group-64 layout, and it can produce
that from Qwen's own bf16 release, which is both better provenance and better
numerics -- quantizing once from the original weights beats inheriting somebody
else's quantization error.

The checkpoint is 360 GB across 131 shards. Nothing here holds the whole thing:
one shard is fetched, converted and deleted before the next is needed, with a
background thread fetching shard N+1 while shard N converts. Peak disk is the
output plus one shard.

Three things the official checkpoint does differently from the MLX repack the
runtime was built against. Each is a silent mis-load if assumed rather than
checked, so `--plan` asserts all three against the checkpoint's own headers:

1. Routed experts are *stacked and fused*: one `mlp.experts.gate_up_proj` of
   shape [512, 1280, 2560] per layer, where 1280 is gate and up concatenated,
   plus one `mlp.experts.down_proj`. TinyTitanRepack wants three separately named
   tensors matching `.mlp.switch_mlp.{gate,up,down}_proj.`, each of which may
   stay stacked over the 512 experts because the planner slices per expert.
2. Norms carry `.weight` here and do not in the repack the runtime reads, so
   `hc_norm.weight` becomes `hc_norm`. `linear_attn.norm.weight` keeps its
   suffix -- the schema asks for that one *with* `.weight`.
3. `lm_head` sits at the archive root, not under `model.language_model.`.

`ple_constants.json` is derived here rather than copied, using the algorithm in
the transformers reference (`_build_layer_multipliers`, `_find_nth_prime_after`).
The derivation reproduces the previously shipped constants exactly -- multipliers,
the sixteen prime head vocabularies and their offsets all match -- which is what
lets this run without any dependency on the third-party repack.

Usage:
    tools/prepare_qwen38.py --plan                    # validate, download nothing
    tools/prepare_qwen38.py --output DIR --work DIR   # convert
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue

try:
    import ml_dtypes
    import numpy as np
    from safetensors import safe_open
    from safetensors.numpy import save_file
except ImportError as exc:  # pragma: no cover - environment, not logic
    sys.exit(f"missing dependency: {exc}\n"
             f"  install them for the interpreter running this file: {sys.executable}\n"
             "    -m pip install safetensors numpy ml_dtypes\n"
             "  (or point TINYTITAN_PYTHON at another Python 3.10+)")
SOURCE = os.environ.get("TINYTITAN_SOURCE", "huggingface").lower()
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
REPO = "Qwen/Qwen3.8-Flash-Next"
BASE = f"{HF_ENDPOINT}/{REPO}/resolve/main"


def resolve_shard_url(path: str) -> str:
    if SOURCE == "modelscope":
        api = f"https://modelscope.cn/api/v1/models/Qwen/Qwen3.8-Flash-Next/repo?Revision=master&FilePath={path}"
        res = subprocess.run(["curl", "-s", "--max-time", "15", api], capture_output=True, text=True)
        if 'href="' in res.stdout:
            m = re.search(r'href="([^"]+)"', res.stdout)
            if m:
                return m.group(1).replace("&amp;", "&")
        return api
    return f"{BASE}/{path}"
GROUP_SIZE = 64
BITS_4, BITS_8 = 4, 8
# indexer_n_heads * indexer_head_dim from the config; the remainder of
# index_qk_proj's rows are the key projection.
INDEXER_QUERY_ROWS = 512
OUTPUT_SHARD_BYTES = 4 << 30

# --- PLE constants, per the transformers reference -------------------------

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _find_nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


def ple_constants(text_config: dict) -> dict:
    """Reproduce what the reference computes at init, as shippable data.

    `ple_layer_index` is the position *within* `ple_layer_ids`, not the layer
    number: the reference reads it as `ple_layer_ids.index(layer_idx + 1)`, so
    for the single PLE layer it is 0. Getting that wrong changes every hash
    multiplier and every n-gram id, silently.
    """
    ple_layer_index = 0
    heads = text_config["heads_per_ngram"] * 2      # two n-gram orders
    vocab_base = text_config["ngram_vocab_size_base"]
    sizes, offsets, total = [], [], 0
    for head in range(heads):
        size = _find_nth_prime_after(vocab_base - 1, ple_layer_index * heads + head + 1)
        sizes.append(size)
        offsets.append(total)
        total += size
    max_long = (1 << 63) - 1
    half_bound = max(1, (max_long // max(text_config["vocab_size"], 1)) // 2)
    base_seed = text_config.get("seed", 1234) + _PRIME_1 * ple_layer_index
    multipliers = [
        2 * (_splitmix64((base_seed + _SPLITMIX_GAMMA * (i + 1)) & _MASK64) % half_bound) + 1
        for i in range(text_config["ngram_size"])
    ]
    eos = text_config["eos_token_id"]
    return {
        "layer_multipliers": multipliers,
        "ngram_heads_offsets": offsets,
        "ngram_heads_vocab_sizes": sizes,
        "eos_token_id": eos[0] if isinstance(eos, list) else eos,
        "ngram_size": text_config["ngram_size"],
        "heads_per_ngram": text_config["heads_per_ngram"],
        "ple_n_heads": heads,
        "ple_head_dim": text_config["ple_embed_dim"] // heads,
        "table_file": "ngram_table.bin",
        "table_dtype": "float16",
    }


# --- quantisation ----------------------------------------------------------


def quantize_affine(value: np.ndarray, bits: int) -> tuple[np.ndarray, ...]:
    """Identical to prepare_ornith_mtp.quantize_affine, deliberately.

    Duplicated rather than imported so neither converter can drift silently;
    a change to one must be made in the other.
    """
    value = value.astype(np.float32)
    if value.shape[-1] % GROUP_SIZE:
        raise ValueError(f"last dimension {value.shape[-1]} is not group-aligned")
    shape = (*value.shape[:-1], value.shape[-1] // GROUP_SIZE, GROUP_SIZE)
    grouped = value.reshape(shape)
    bias = grouped.min(axis=-1)
    high = grouped.max(axis=-1)
    levels = (1 << bits) - 1
    scale = np.where(high == bias, np.float32(1), (high - bias) / levels)
    scale = scale.astype(ml_dtypes.bfloat16)
    bias = bias.astype(ml_dtypes.bfloat16)
    quantized = np.rint(
        (grouped - bias.astype(np.float32)[..., None])
        / scale.astype(np.float32)[..., None]
    ).clip(0, levels).astype(np.uint32).reshape(value.shape)
    lanes = 32 // bits
    words = quantized.reshape(*quantized.shape[:-1], quantized.shape[-1] // lanes, lanes)
    packed = np.zeros(words.shape[:-1], dtype=np.uint32)
    for lane in range(lanes):
        packed |= words[..., lane] << np.uint32(bits * lane)
    return packed, scale, bias


# --- naming ----------------------------------------------------------------


def is_multimodal(name: str) -> bool:
    return ".visual." in name or name.startswith("model.visual.")


def is_ngram(name: str) -> bool:
    return ".ngram_embedding.shard_" in name


# The checkpoint carries the PLE hash constants as int64 buffers beside the
# table. They are not weights, the repacker has no dtype for them, and this
# converter derives the same values into ple_constants.json from the algorithm
# in the transformers reference -- verified against these very buffers, which
# match exactly. Carrying them would only give the repacker something it must
# reject.
PLE_BUFFERS = ("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes")


def is_ple_buffer(name: str) -> bool:
    return name.startswith("model.language_model.layers.") \
        and ".ple.ple_embedding." in name \
        and name.rsplit(".", 1)[-1] in PLE_BUFFERS


# Names the runtime's schema spells without `.weight`, where the checkpoint
# has it. Not a family property -- it is what the repack the runtime was built
# against happened to write -- so each one is listed rather than guessed at
# from a pattern. `linear_attn.conv1d.weight` deliberately keeps its suffix:
# the schema asks for that one with it, and stripping it by pattern would
# break the GDN path.
STRIP_WEIGHT_SUFFIXES = (
    ".hc_norm",
    ".self_attn.q_norm",
    ".self_attn.k_norm",
    ".self_attn.indexer.q_layernorm",
    ".self_attn.indexer.k_layernorm",
    ".ple.conv1d",
    ".ple.norm_conv",
    ".ple.norm_key",
    ".ple.norm_query",
    ".pre_fc_norm_embedding",
    ".pre_fc_norm_hidden",
)


# RMSNorm weights the model reads as `1 + weight`.
#
# `Qwen3_5RMSNorm` initialises its parameter to *zeros* and computes
# `normalized * (1.0 + weight)`, so the checkpoint stores the offset from one,
# not the gain. The runtime multiplies by the stored value, so the +1 has to be
# folded in here -- which is exactly what MLX's conversion does, and why a
# build made from the MLX repack answered while one made from the original did
# not.
#
# Two absences are deliberate. `linear_attn.norm` is `Qwen3NextRMSNormGated`,
# which uses `weight` directly with no offset. `ple.conv1d` is a convolution
# kernel, not a norm. Folding either would be as wrong as not folding these.
#
# Getting this wrong is silent. Every tensor still matches the checkpoint
# byte for byte, every parity harness still agrees -- both sides read the same
# stored value -- and the model generates fluent nonsense. It cost a full
# investigation; see docs/qwen38-flash-next-port.md.
UNIT_OFFSET_NORM_SUFFIXES = (
    ".hc_norm",
    ".self_attn.q_norm",
    ".self_attn.k_norm",
    ".self_attn.indexer.q_layernorm",
    ".self_attn.indexer.k_layernorm",
    ".ple.norm_conv",
    ".ple.norm_key",
    ".ple.norm_query",
    ".pre_fc_norm_embedding",
    ".pre_fc_norm_hidden",
)


def fold_unit_offset(out_name: str, value: np.ndarray) -> np.ndarray:
    """Fold the implicit +1 into a zero-centred RMSNorm weight."""
    stem = out_name[: -len(".weight")] if out_name.endswith(".weight") else out_name
    if stem.endswith(UNIT_OFFSET_NORM_SUFFIXES):
        return (value.astype(np.float32) + 1.0).astype(value.dtype)
    return value


def rename(name: str) -> str:
    for suffix in STRIP_WEIGHT_SUFFIXES:
        if name.endswith(suffix + ".weight"):
            return name[: -len(".weight")]
    return name


# Families kept at the checkpoint's own bf16 in an 8-bit build.
#
# The 8-bit build exists for work where correctness outranks speed -- an
# overnight coding run on a machine doing nothing else -- so it is worth
# spending resident memory that a 4-bit build could not.
#
# Chosen by measurement, not by intuition. Every family measures 0.57-1.0%
# relative error at 8 bits, so the selection is by what the error *does*, not
# how large it is: the router and the indexer produce rankings, and a ranking
# either matches or it does not (the router's top-10 agrees with bf16 on 98.8%
# of tokens at 8 bits). The rest are the smallest tensors in the model, where
# group-64 quantization has the fewest values to amortise over and the error is
# correspondingly highest -- a 1-row gate has nothing to average.
#
# The cost is bounded and lands in the right place: ~108 MB of resident memory,
# 2.0% of the active parameters per token, and about 0.3% of decode. Resident
# weights come from RAM; the routed experts, which are 23x the parameters and
# would double SSD traffic, are deliberately not here.
PROMOTE_TO_BF16_AT_8BIT = (
    ".mlp.gate",                        # router: picks which experts run
    ".mlp.shared_expert_gate",          # 1 row of D; highest measured error
    ".block_inject_weight",             # 4 rows; the write gate for every layer
    ".linear_attn.in_proj_a",
    ".linear_attn.in_proj_b",
    ".ple.key_proj",
    ".self_attn.indexer.index_q_proj",
    ".self_attn.indexer.index_k_proj",
)


def promoted_to_bf16(name: str, width: int) -> bool:
    """Is this tensor kept unquantized in a build of this width?"""
    if width != BITS_8:
        return False
    stem = name[: -len(".weight")] if name.endswith(".weight") else name
    return stem.endswith(PROMOTE_TO_BF16_AT_8BIT)


def quant_bits(name: str, width: int = BITS_4) -> int | None:
    """Bits for a tensor, or None to copy it through unquantised.

    This mirrors the runtime's slot model *plus its per-tensor overrides*. `Model`
    derives a tensor's expected size from `manifest.quant.<slot>.weightBits` and
    builds one GEMV per *role*, so a role shares one width — but a tensor whose
    width differs from its slot carries an override keyed by stem, which the
    repacker writes (`GTurboJSON.quantObject`), the format validates
    (`GTurboManifestV1`), and the runtime resolves per tensor
    (`ManifestQuant.slot(forTensorNamed:overrides:fallback:)`). The dense
    Qwen 3.5 installs are built exactly that way: `mlp.*` at 4 bits against an
    8-bit slot, full-attention `k_proj`/`v_proj` at 8 against a 4-bit one.

    One pair is the exception, and it is the pair this converter promotes: the
    GDN `a`/`b` projections. The fused QKV+Z+A+B kernel takes a *bf16-or-slot*
    flag rather than a width and its quantized branch is int4, so the runtime
    refuses a quantized override for them by name
    (`Model.validateRoleUniformity`). A 16-bit override is honoured, which is
    the only width asked for here.

    The slots, matching what a working install carries:

        embedding     8   embed_tokens, lm_head
        router        8   mlp.gate
        attention   `width`   q/k/v/o, GDN, hyper-connection, indexer, PLE
        sharedExpert `width`  shared_expert.*, shared_expert_gate
        routedExpert `width`  switch_mlp.*

    This is why the router already sits at 8 bits in a 4-bit build: it has its
    own slot. The measurement in tools/precision_probe.py says the QSA indexer,
    the hyper-connection write gate and the PLE key projection deserve more
    precision -- the indexer picks the same keys 0.0% of the time at 4 bits
    against 49.5% at 8, for about 10 MB. Taking the whole attention slot to 8
    bits instead is measured and *works*, but it is 61% of the active parameters
    and +2.10 GB resident (Engineering Notes, "8-bit attention slot"); these
    tensors are ~10 MB. A per-tensor override is how they get the precision
    cheaply, and the runtime resolves one for all three families --
    `Model.hyperConnectionWeightBits`, `pleKeyWeightBits`,
    `qsaIndexerWeightBits`, uniform across the family because one kernel serves
    every layer. What this converter does not yet do is ask for it, and the
    8-bit indexer was measured and rejected on its own (2.4 points on marginal
    keys), so the promotion is a decision per tensor rather than a switch.
    """
    if not name.endswith(".weight"):
        return None
    if name.endswith("conv1d.weight"):
        return None
    if name.endswith("norm.weight"):
        return None
    # Checked before the slot rules below: promotion overrides the slot, which
    # is the whole point of it. The runtime reads these tensors' width from
    # their own dtype rather than from the slot they nominally belong to.
    if promoted_to_bf16(name, width):
        return None
    if name.endswith("embed_tokens.weight") or name == "lm_head.weight":
        return BITS_8                      # embedding slot
    # Both of these validate against `quant.router` in Model.swift. The scalar
    # gate does so explicitly and against expectation -- "quantized at the
    # ROUTER's bit width ... independent of the sharedExpert slot" -- which is
    # why grouping it with the shared expert produced a tensor half the size
    # the loader wanted.
    if name.endswith(".mlp.gate.weight") or name.endswith(".shared_expert_gate.weight"):
        return BITS_8                      # router slot
    return width                           # attention / sharedExpert / routedExpert


def outputs_for(name: str, shape: list[int]) -> list[tuple[str, list[int]]]:
    new = rename(name)
    if new.endswith(".mlp.experts.gate_up_proj"):
        stem = new[: -len("experts.gate_up_proj")] + "switch_mlp."
        experts, fused, hidden = shape
        return [(stem + "gate_proj.weight", [experts, fused // 2, hidden]),
                (stem + "up_proj.weight", [experts, fused // 2, hidden])]
    if new.endswith(".mlp.experts.down_proj"):
        stem = new[: -len("experts.down_proj")] + "switch_mlp."
        return [(stem + "down_proj.weight", list(shape))]
    if new.endswith(".self_attn.indexer.index_qk_proj.weight"):
        # Fused query and key in the checkpoint, separate in the runtime.
        # [640, 2560] is 4 heads x 128 of query followed by 1 kv head x 128 of
        # key, which the config's indexer_n_heads/indexer_kv_heads confirm.
        stem = new[: -len("index_qk_proj.weight")]
        rows, hidden = shape
        q = INDEXER_QUERY_ROWS
        return [(stem + "index_q_proj.weight", [q, hidden]),
                (stem + "index_k_proj.weight", [rows - q, hidden])]
    return [(new, list(shape))]



def write_config(config: dict, out: Path, tensor_names, width: int) -> dict:
    """Emit config.json with the `quantization` block TinyTitanRepack reads.

    The repacker resolves each tensor's width from `config.json -> quantization`:
    a base `bits`/`group_size`/`mode`, plus per-tensor overrides keyed by the
    tensor name with `.weight` stripped. Qwen's own config has no such block --
    it describes an unquantised model -- so copying it verbatim leaves the
    repacker with nothing to read and it refuses the snapshot.

    Every tensor whose width differs from the base is listed explicitly rather
    than relying on the repacker to re-derive the policy, so the snapshot
    records what it actually contains.
    """
    overrides = {}
    for name in tensor_names:
        if not name.endswith(".weight"):
            continue
        bits = quant_bits(name, width)
        if bits is None or bits == width:
            continue
        overrides[name[: -len(".weight")]] = {"bits": bits, "group_size": GROUP_SIZE}
    config = dict(config)
    config["quantization"] = {
        "bits": width, "group_size": GROUP_SIZE, "mode": "affine", **overrides,
    }
    (out / "config.json").write_text(json.dumps(config, indent=1))
    return config


# --- transport -------------------------------------------------------------


def fetch_header(shard: str) -> dict:
    url = resolve_shard_url(shard)
    raw = subprocess.run(["curl", "-sfL", "--max-time", "60", "-r", "0-7", url],
                         capture_output=True, check=True).stdout
    size = struct.unpack("<Q", raw[:8])[0]
    body = subprocess.run(["curl", "-sfL", "--max-time", "180", "-r", f"8-{8 + size - 1}", url],
                          capture_output=True, check=True).stdout
    return json.loads(body)


def download(shard: str, work: Path) -> Path:
    """Fetch one shard, resuming a partial file rather than restarting it."""
    dest = work / shard
    dest.parent.mkdir(parents=True, exist_ok=True)
    max_retries = 20
    for attempt in range(1, max_retries + 1):
        url = resolve_shard_url(shard)
        cmd = ["curl", "-fL", "--retry", "5", "--retry-delay", "3",
               "--retry-connrefused", "--retry-all-errors", "-C", "-",
               "--silent", "--show-error", "-o", str(dest), url]
        res = subprocess.run(cmd)
        if res.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
            return dest
        print(f"\n    [download retry {attempt}/{max_retries}] {shard} fetch failed, retrying in 5s...",
              file=sys.stderr, flush=True)
        time.sleep(5)
    raise RuntimeError(f"Failed to download {shard} after {max_retries} attempts")


# --- conversion ------------------------------------------------------------


class OutputWriter:
    """Accumulates converted tensors and flushes them as safetensors shards."""

    def __init__(self, out: Path):
        self.out = out
        self.out.mkdir(parents=True, exist_ok=True)
        self.block: dict[str, np.ndarray] = {}
        self.bytes = 0
        self.index: dict[str, str] = {}
        self.total = 0
        self.shard_no = 0

        existing_shards = sorted(self.out.glob("model-[0-9][0-9][0-9][0-9][0-9].safetensors"))
        if existing_shards:
            print(f"found {len(existing_shards)} existing output shards, resuming index...", flush=True)
            for shard_path in existing_shards:
                num = int(shard_path.stem.split("-")[1])
                if num > self.shard_no:
                    self.shard_no = num
                with shard_path.open("rb") as f:
                    hdr_size = struct.unpack("<Q", f.read(8))[0]
                    hdr = json.loads(f.read(hdr_size).decode("utf-8"))
                for key, meta in hdr.items():
                    if key == "__metadata__":
                        continue
                    self.index[key] = shard_path.name
                    offsets = meta["data_offsets"]
                    self.total += offsets[1] - offsets[0]
            print(f"  resumed up to shard {self.shard_no:05d} ({self.total / 1e9:.2f} GB, {len(self.index)} tensors)", flush=True)

    def add(self, name: str, value: np.ndarray) -> None:
        self.block[name] = value
        self.bytes += value.nbytes
        self.total += value.nbytes
        if self.bytes >= OUTPUT_SHARD_BYTES:
            self.flush()

    def flush(self) -> None:
        if not self.block:
            return
        self.shard_no += 1
        name = f"model-{self.shard_no:05d}.safetensors"
        save_file(self.block, str(self.out / name))
        for key in self.block:
            self.index[key] = name
        print(f"    wrote {name} ({self.bytes / 1e9:.2f} GB, {len(self.block)} tensors)",
              flush=True)
        self.block.clear()
        self.bytes = 0

    def finish(self) -> None:
        self.flush()
        # Rename to the N-of-M form the loaders expect now that M is known.
        final = {}
        for old_key, old_name in self.index.items():
            n = int(old_name.split("-")[1].split(".")[0])
            final[old_key] = f"model-{n:05d}-of-{self.shard_no:05d}.safetensors"
        for n in range(1, self.shard_no + 1):
            src = self.out / f"model-{n:05d}.safetensors"
            src.rename(self.out / f"model-{n:05d}-of-{self.shard_no:05d}.safetensors")
        (self.out / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {"total_size": self.total}, "weight_map": final}, indent=1))


def convert_shard(path: Path, writer: OutputWriter, ngram: "NgramTable",
                  width: int = BITS_4) -> None:
    with safe_open(path, framework="np") as src:
        for name in src.keys():
            if is_multimodal(name):
                continue
            if is_ngram(name):
                ngram.add(name, src.get_tensor(name))
                continue
            if is_ple_buffer(name):
                continue
            value = src.get_tensor(name)
            for out_name, _ in outputs_for(name, list(value.shape)):
                if out_name.endswith("switch_mlp.gate_proj.weight"):
                    piece = value[:, : value.shape[1] // 2, :]
                elif out_name.endswith("switch_mlp.up_proj.weight"):
                    piece = value[:, value.shape[1] // 2:, :]
                elif out_name.endswith("indexer.index_q_proj.weight"):
                    piece = value[:INDEXER_QUERY_ROWS]
                elif out_name.endswith("indexer.index_k_proj.weight"):
                    piece = value[INDEXER_QUERY_ROWS:]
                else:
                    piece = value
                bits = quant_bits(out_name, width)
                if bits is None:
                    writer.add(out_name, np.ascontiguousarray(
                        fold_unit_offset(out_name, piece)))
                    continue
                stem = out_name[: -len(".weight")]
                packed, scales, biases = quantize_affine(np.ascontiguousarray(piece), bits)
                writer.add(stem + ".weight", packed)
                writer.add(stem + ".scales", scales)
                writer.add(stem + ".biases", biases)


class NgramTable:
    """Assembles ngram_table.bin from the checkpoint's 128 table shards.

    The shards must be concatenated in *numeric* order. Their names sort
    lexically as shard_0, shard_1, shard_10, shard_100 ..., so sorting the
    strings would interleave the table and produce a model that loads, runs,
    and is quietly wrong.
    """

    def __init__(self, out: Path, expected_rows: int, dim: int,
                 reuse: Path | None = None):
        self.path = out / "ngram_table.bin"
        self.expected_rows = expected_rows
        self.dim = dim
        self.pending: dict[int, np.ndarray] = {}
        self.next_index = 0
        self.rows = 0
        self.reused = reuse is not None
        self.skipped = 0
        if reuse is None:
            self.handle = self.path.open("wb")
            return
        # Hardlink, not copy and not symlink. Both directories then hold a
        # reference to one inode: deleting either leaves the other intact,
        # which a symlink would not, and the runtime's F_NOCACHE reads do not
        # care about the extra link.
        self.handle = None
        expected_bytes = expected_rows * dim * 2
        actual = reuse.stat().st_size
        if actual != expected_bytes:
            raise ValueError(
                f"{reuse}: {actual} bytes, expected {expected_bytes} "
                f"({expected_rows} rows x {dim} x fp16). A table of the wrong "
                "size is a different model's, or a truncated copy.")
        if self.path.resolve() != reuse.resolve():
            if self.path.exists():
                self.path.unlink()
            os.link(reuse, self.path)

    @staticmethod
    def index_of(name: str) -> int:
        return int(name.rsplit(".shard_", 1)[1].split(".")[0])

    def add(self, name: str, value: np.ndarray) -> None:
        if self.reused:
            # Two of the 131 checkpoint shards carry n-gram rows alongside
            # ordinary tensors, so they are fetched even when the table is
            # reused and their n-gram names still arrive here. The linked
            # table already contains those rows -- the size check in __init__
            # is what guarantees it holds all of them -- so drop them.
            self.skipped += 1
            return
        self.pending[self.index_of(name)] = value
        while self.next_index in self.pending:
            block = self.pending.pop(self.next_index)
            if block.dtype == ml_dtypes.bfloat16:
                as_f32 = block.astype(np.float32)
                finite = np.isfinite(as_f32.astype(np.float16))
                if not finite.all():
                    raise ValueError(
                        f"{name}: bf16 -> fp16 overflows on "
                        f"{(~finite).sum()} values; the table format is fp16")
                block = as_f32.astype(np.float16)
            self.handle.write(np.ascontiguousarray(block).tobytes())
            self.rows += block.shape[0]
            self.next_index += 1

    def summary(self) -> str:
        """What the table cost this build, in the mode it actually ran."""
        if self.reused:
            return (f"linked, {self.expected_rows} rows "
                    f"({self.expected_rows * self.dim * 2 / 1e9:.1f} GB not written)")
        return f"{self.rows} rows"

    def finish(self) -> None:
        if self.reused:
            if self.pending:
                raise ValueError("n-gram shards arrived while reusing a table")
            if self.skipped:
                print(f"  {self.skipped} n-gram tensors in mixed shards ignored "
                      f"(already in the linked table)")
            return
        self.handle.close()
        if self.pending:
            raise ValueError(f"n-gram shards never became contiguous: "
                             f"{sorted(self.pending)[:5]} still pending")
        if self.rows != self.expected_rows:
            raise ValueError(f"n-gram table has {self.rows} rows, "
                             f"expected {self.expected_rows}")


# --- driver ----------------------------------------------------------------


# The repacker refuses a snapshot without these: a model imported from one has
# no other source for them, and an install that loads but cannot be prompted is
# worse than one that fails. `config.json` is written separately by
# `write_config`, which adds the quantization block the repacker reads.
TOKENIZER_FILES = (
    ("tokenizer.json", True),
    ("tokenizer_config.json", True),
    ("special_tokens_map.json", False),
    ("chat_template.jinja", False),
    ("chat_template.json", False),
)


def fetch_tokenizer(out: Path) -> None:
    """Copy the tokenizer beside the weights, so the snapshot stands alone."""
    for name, required in TOKENIZER_FILES:
        url = resolve_shard_url(name)
        result = subprocess.run(["curl", "-sfL", "--max-time", "300", url],
                                capture_output=True)
        if result.returncode != 0 or not result.stdout:
            if required:
                raise SystemExit(f"cannot fetch {name} from {REPO}; the "
                                 "snapshot would be rejected at repack")
            continue
        (out / name).write_bytes(result.stdout)
        print(f"  {name} ({len(result.stdout) / 1e6:.2f} MB)")


def plan(index: dict, width: int = BITS_4) -> None:
    wm = index["weight_map"]
    shards = sorted(set(wm.values()))
    text = {n: s for n, s in wm.items() if not is_multimodal(n)}
    ngram = [n for n in text if is_ngram(n)]
    print(f"repo     : {REPO}")
    print(f"shards   : {len(shards)}   tensors: {len(wm)} "
          f"({len(wm) - len(text)} multimodal skipped)")
    print(f"declared : {index['metadata']['total_size'] / 1e9:.1f} GB")
    print(f"n-gram   : {len(ngram)} table shards")
    probe = [s for s in shards if any(
        ".mlp.experts." in n for n, sh in text.items() if sh == s)][:1]
    problems: list[str] = []
    for shard in probe:
        for name, meta in fetch_header(shard).items():
            if name == "__metadata__" or is_multimodal(name) or is_ngram(name):
                continue
            for out_name, out_shape in outputs_for(name, meta["shape"]):
                bits = quant_bits(out_name, width)
                if bits and out_shape[-1] % GROUP_SIZE:
                    problems.append(f"{out_name} last dim {out_shape[-1]} unaligned")
                print(f"  {name} {meta['shape']}\n     -> {out_name} {out_shape} "
                      f"{'q' + str(bits) if bits else 'passthrough'}")
    print("\nPROBLEMS: " + "; ".join(problems) if problems
          else "\nplan validates against the checkpoint's own headers")


# The PLE constants a table is addressed by. The file is a hash table: its
# contents are meaningless under different multipliers, offsets or vocabulary
# sizes, and none of that is recoverable from the file itself, so a build may
# only reuse a table whose constants match its own.
REUSE_CONSTANT_KEYS = ("layer_multipliers", "ngram_heads_offsets",
                       "ngram_heads_vocab_sizes", "ngram_size",
                       "heads_per_ngram", "ple_head_dim")


def reusable_table_path(reuse: Path, constants: dict) -> Path:
    """The table to hardlink for `--reuse-ngram-table`, or SystemExit.

    `reuse` is an install directory (whose `ple_constants.json` is checked
    against this build's) or the table file itself. Exits rather than returning
    None on a bad one: silently falling back to fetching 128 shards and writing
    102 GB is the wrong default for a flag whose whole purpose is to save them,
    and a table read under the wrong constants produces garbage ids rather than
    an error.
    """
    if reuse.is_dir():
        sibling = reuse / "ple_constants.json"
        if sibling.exists():
            have = json.loads(sibling.read_text())
            for key in REUSE_CONSTANT_KEYS:
                if have.get(key) != constants.get(key):
                    raise SystemExit(
                        f"--reuse-ngram-table: {key} differs between "
                        f"{sibling} and this build; the table is addressed "
                        "by those constants and would be read wrongly")
        reuse = reuse / "ngram_table.bin"
    if not reuse.exists():
        raise SystemExit(f"--reuse-ngram-table: no such file: {reuse}")
    return reuse


def main() -> int:
    global SOURCE, HF_ENDPOINT, BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--bits", type=int, choices=(4, 8), default=4,
                    help="routed-expert width of the install being built")
    ap.add_argument("--rewrite-config", type=Path,
                    help="regenerate config.json for an existing snapshot")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--work", type=Path, help="scratch for in-flight shards")
    ap.add_argument("--reuse-ngram-table", type=Path, metavar="DIR_OR_FILE",
                    help="hardlink ngram_table.bin from an existing install or "
                         "snapshot instead of fetching the table shards. The "
                         "table is fp16 in every quantization and is fully "
                         "determined by the checkpoint, so two builds of the "
                         "same model cannot differ in it.")
    ap.add_argument("--index", type=Path)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--source", choices=("huggingface", "modelscope"),
                    default=SOURCE,
                    help="source provider for checkpoint shards (default: huggingface, optional: modelscope)")
    ap.add_argument("--endpoint", default=HF_ENDPOINT,
                    help="Hugging Face endpoint URL or mirror (default: https://huggingface.co)")
    args = ap.parse_args()

    SOURCE = args.source
    HF_ENDPOINT = args.endpoint.rstrip("/")
    BASE = f"{HF_ENDPOINT}/{REPO}/resolve/main"

    def fetch_json(path: Path | None, remote: str) -> dict:
        if path and path.exists():
            return json.loads(path.read_text())
        if SOURCE == "modelscope":
            url = f"https://modelscope.cn/api/v1/models/Qwen/Qwen3.8-Flash-Next/repo?Revision=master&FilePath={remote}"
        else:
            url = f"{HF_ENDPOINT}/{REPO}/raw/main/{remote}"
        return json.loads(subprocess.run(["curl", "-sfL", "--max-time", "60", url],
                                         capture_output=True, check=True).stdout)

    index = fetch_json(args.index, "model.safetensors.index.json")
    if args.rewrite_config:
        snap = args.rewrite_config
        names = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
        cfg_url = (f"https://modelscope.cn/api/v1/models/Qwen/Qwen3.8-Flash-Next/repo?Revision=master&FilePath=config.json"
                   if SOURCE == "modelscope" else f"{HF_ENDPOINT}/{REPO}/raw/main/config.json")
        cfg = json.loads(subprocess.run(
            ["curl", "-sfL", cfg_url],
            capture_output=True, check=True).stdout) if not (snap / "config.json").exists() \
            else json.loads((snap / "config.json").read_text())
        cfg.pop("quantization", None)
        out = write_config(cfg, snap, names.keys(), args.bits)
        n = len(out["quantization"]) - 3
        print(f"wrote {snap}/config.json: base {args.bits}-bit, {n} overrides")
        return 0

    if args.plan or not args.output:
        plan(index, args.bits)
        return 0

    config = fetch_json(args.config, "config.json")
    text_config = config["text_config"]
    work = args.work or (args.output.parent / "qwen38-shards")
    work.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)

    constants = ple_constants(text_config)
    (args.output / "ple_constants.json").write_text(json.dumps(constants, indent=1))
    rows = sum(constants["ngram_heads_vocab_sizes"])
    divisor = text_config.get("make_ngram_vocab_size_divisible_by", 128)
    padded = math.ceil(rows / divisor) * divisor

    wm = index["weight_map"]
    shards = sorted(set(wm.values()))
    by_shard: dict[str, list[str]] = {}
    for name, shard in wm.items():
        by_shard.setdefault(shard, []).append(name)

    writer = OutputWriter(args.output)

    expected_bytes = padded * constants["ple_head_dim"] * 2
    if args.reuse_ngram_table is None and args.output is not None:
        table_file = args.output / "ngram_table.bin"
        if table_file.exists() and table_file.stat().st_size == expected_bytes:
            print(f"found existing completed ngram_table.bin ({expected_bytes / 1e9:.1f} GB), reusing automatically", flush=True)
            args.reuse_ngram_table = table_file

    reuse = (None if args.reuse_ngram_table is None
             else reusable_table_path(args.reuse_ngram_table, constants))

    ngram = NgramTable(args.output, padded, constants["ple_head_dim"], reuse)

    if reuse is not None:
        # Skip the shards that carry nothing else. Two of the 131 hold n-gram
        # rows alongside ordinary tensors and are still fetched; convert_shard
        # already ignores the n-gram names inside them.
        skippable = {shard for shard, names in by_shard.items()
                     if all(is_ngram(n) for n in names)}
        shards = [s for s in shards if s not in skippable]
        print(f"reusing n-gram table: {reuse}")
        print(f"  hardlinked, {padded * constants['ple_head_dim'] * 2 / 1e9:.1f} GB not written")
        print(f"  {len(skippable)} of {len(by_shard)} checkpoint shards not fetched")

    if writer.index:
        def shard_is_complete(shard_name: str) -> bool:
            names = by_shard[shard_name]
            has_tensors = False
            for n in names:
                if is_multimodal(n) or is_ngram(n) or is_ple_buffer(n):
                    continue
                has_tensors = True
                stem_raw = rename(n)
                if stem_raw.endswith(".mlp.experts.gate_up_proj"):
                    st = stem_raw[: -len("experts.gate_up_proj")] + "switch_mlp."
                    targets = [st + "gate_proj.weight", st + "up_proj.weight"]
                elif stem_raw.endswith(".mlp.experts.down_proj"):
                    st = stem_raw[: -len("experts.down_proj")] + "switch_mlp."
                    targets = [st + "down_proj.weight"]
                elif stem_raw.endswith(".self_attn.indexer.index_qk_proj.weight"):
                    st = stem_raw[: -len("index_qk_proj.weight")]
                    targets = [st + "index_q_proj.weight", st + "index_k_proj.weight"]
                else:
                    targets = [stem_raw]

                for t in targets:
                    bits = quant_bits(t, args.bits)
                    if bits is None:
                        if t not in writer.index:
                            return False
                    else:
                        st = t[: -len(".weight")] if t.endswith(".weight") else t
                        if (st + ".weight") not in writer.index or \
                           (st + ".scales") not in writer.index or \
                           (st + ".biases") not in writer.index:
                            return False
            return has_tensors

        already_done = {s for s in shards if shard_is_complete(s)}
        if already_done:
            shards = [s for s in shards if s not in already_done]
            print(f"  {len(already_done)} checkpoint shards already converted in existing output shards, skipping them", flush=True)
            print(f"  remaining {len(shards)} checkpoint shards to fetch and convert", flush=True)

    # Fetch shard N+1 while shard N converts.
    queue: Queue = Queue(maxsize=1)

    def fetcher() -> None:
        for shard in shards:
            try:
                queue.put(download(shard, work))
            except Exception as exc:                     # noqa: BLE001
                queue.put(exc)
                return
        queue.put(None)

    threading.Thread(target=fetcher, daemon=True).start()
    done = 0
    while True:
        item = queue.get()
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        done += 1
        print(f"[{done}/{len(shards)}] {item.name}", flush=True)
        convert_shard(item, writer, ngram, args.bits)
        item.unlink()

    writer.finish()
    ngram.finish()
    # Written last: the quantization block lists every tensor whose width
    # differs from the base, which is only known once they have all been seen.
    write_config(config, args.output, writer.index.keys(), args.bits)
    print("tokenizer:")
    fetch_tokenizer(args.output)
    print(f"\naffine snapshot written to {args.output}")
    print(f"  {writer.shard_no} shards, {writer.total / 1e9:.1f} GB")
    print(f"  ngram_table.bin {ngram.summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
