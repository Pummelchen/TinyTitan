#!/usr/bin/env python3
"""Convert a Qwen3.8-Flash-Next PLE n-gram table from fp16 to 4-bit affine.

The table is 160 fp16 values per row, 320,001,446 rows on the shipped 125B
install: 95.4 GiB, and identical in every quantisation of the same checkpoint
(`docs/ngram-table-sharing-plan.md`). Project Maya's engine carries the same
Qwen3.8 table at 28.8 GB (its base engine is a Qwen3.8 implementation), which
is about 4.5 bits a value. This tool produces the same class of table for this
runtime: 4-bit affine, one bf16 scale and bias per group of 32 values, which
is 100 bytes a row (80 packed + 10 scale + 10 bias) and 32.0 GB for the shipped
geometry.

File layout (little-endian):

    40-byte header: magic "TTQ4" (0x34515454), version 1, rowDim, bits=4,
                    groupSize=32, rowBytes=100, rowCount (u64), headerBytes=40,
                    reserved
    rows:           packed nibbles, low nibble first (rowDim/2 bytes),
                    group scales as bf16 (groups * 2 bytes),
                    group biases as bf16 (groups * 2 bytes)

Usage:

    python3 tools/quantize_ngram_table.py --install models/<125B install> \
        --out models/<install>/ngram_table.q4

    # then point ple_constants.json at it (and, when the result is verified,
    # delete ngram_table.bin to reclaim the fp16 copy):
    python3 tools/quantize_ngram_table.py --install models/<install> \
        --out models/<install>/ngram_table.q4 --update-metadata

`--rows N` writes only the first N rows and sets that in the header; it exists
to test the format cheaply, and the reader will refuse the file for the real
install because its declared row count will not match the sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import struct
import sys

import numpy as np

MAGIC = 0x34515454  # "TTQ4", little-endian
VERSION = 1
GROUP = 32
BITS = 4
HEADER_BYTES = 40


def bf16(values: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even float32 -> bfloat16, returned as little-endian u16."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)
    return rounded.astype("<u2")


def decode_rows(blob: bytes, groups: int, row_dim: int) -> np.ndarray:
    """Decode one packed row to float32, the same arithmetic the reader uses."""
    packed = np.frombuffer(blob[: row_dim // 2], dtype=np.uint8)
    nibbles = np.empty(row_dim, dtype=np.uint8)
    nibbles[0::2] = packed & 0x0F
    nibbles[1::2] = packed >> 4
    scales = np.frombuffer(blob[row_dim // 2 : row_dim // 2 + groups * 2], dtype="<u2")
    biases = np.frombuffer(blob[row_dim // 2 + groups * 2 : row_dim // 2 + groups * 4], dtype="<u2")
    scale_f = (scales.astype(np.uint32) << np.uint32(16)).view(np.float32)
    bias_f = (biases.astype(np.uint32) << np.uint32(16)).view(np.float32)
    values = nibbles.astype(np.float32) * np.repeat(scale_f, GROUP)
    values += np.repeat(bias_f, GROUP)
    return values


def row_count(constants: dict) -> int:
    return int(constants["ngram_heads_offsets"][-1]) + int(constants["ngram_heads_vocab_sizes"][-1])


def convert(
    source: pathlib.Path,
    destination: pathlib.Path,
    rows: int,
    row_dim: int,
    chunk_rows: int,
) -> None:
    groups = row_dim // GROUP
    if groups * GROUP != row_dim:
        raise SystemExit(f"rowDim {row_dim} is not a multiple of {GROUP}")
    row_bytes = row_dim // 2 + groups * 4
    expected = rows * row_dim * 2
    actual = source.stat().st_size
    if actual < expected:
        raise SystemExit(f"{source} is {actual} bytes; {rows} rows need {expected}")

    header = struct.pack(
        "<IIIIIIQII",
        MAGIC,
        VERSION,
        row_dim,
        BITS,
        GROUP,
        row_bytes,
        rows,
        HEADER_BYTES,
        0,
    )
    mm = np.memmap(source, dtype="<f2", mode="r", shape=(rows, row_dim))
    written = 0
    print(
        f"converting {rows} rows x {row_dim} fp16 -> {row_bytes} B/row "
        f"({rows * row_bytes / 1e9:.1f} GB)",
        flush=True,
    )
    with destination.open("wb") as out:
        out.write(header)
        for start in range(0, rows, chunk_rows):
            stop = min(start + chunk_rows, rows)
            n = stop - start
            values = np.array(mm[start:stop], dtype=np.float32)
            grouped = values.reshape(n, groups, GROUP)
            low = grouped.min(axis=2)
            high = grouped.max(axis=2)
            scale = (high - low) / np.float32(15.0)
            scale = np.where(scale == 0, np.float32(1.0), scale)
            quantized = np.rint((grouped - low[:, :, None]) / scale[:, :, None])
            np.clip(quantized, 0, 15, out=quantized)
            q = quantized.astype(np.uint8)
            packed = (q[:, :, 0::2] | (q[:, :, 1::2] << np.uint8(4))).astype(np.uint8)
            scales = bf16(scale).view(np.uint8).reshape(n, groups * 2)
            biases = bf16(low).view(np.uint8).reshape(n, groups * 2)
            row = np.concatenate([packed.reshape(n, row_dim // 2), scales, biases], axis=1)
            out.write(row.tobytes())
            written = stop
            if (start // chunk_rows) % 20 == 0:
                print(f"  {written}/{rows} rows", flush=True)
    if written != rows:
        raise SystemExit(f"wrote {written} rows, expected {rows}")


def verify(source: pathlib.Path, destination: pathlib.Path, rows: int, row_dim: int) -> None:
    groups = row_dim // GROUP
    row_bytes = row_dim // 2 + groups * 4
    rng = np.random.default_rng(7)
    samples = sorted(set(int(v) for v in rng.integers(0, rows, size=8)))
    mm = np.memmap(source, dtype="<f2", mode="r", shape=(rows, row_dim))
    worst_mae = 0.0
    worst_max = 0.0
    with destination.open("rb") as handle:
        handle.seek(HEADER_BYTES)
        for row in samples:
            handle.seek(HEADER_BYTES + row * row_bytes)
            blob = handle.read(row_bytes)
            decoded = decode_rows(blob, groups, row_dim)
            original = np.array(mm[row], dtype=np.float32)
            error = np.abs(decoded - original)
            worst_mae = max(worst_mae, float(error.mean()))
            worst_max = max(worst_max, float(error.max()))
    print(
        f"verified {len(samples)} rows: mean abs error <= {worst_mae:.5g}, "
        f"max abs error <= {worst_max:.5g}"
    )


def sha256_of(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def update_metadata(install: pathlib.Path, table: pathlib.Path) -> None:
    constants_path = install / "ple_constants.json"
    constants = json.loads(constants_path.read_text())
    constants["table_file"] = table.name
    constants["table_dtype"] = "int4_affine_group32"
    constants_path.write_text(json.dumps(constants, indent=1) + "\n")

    manifest_path = install / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    files = manifest.get("files", {})
    entry = files.pop("ngram_table.bin", None)
    files[table.name] = {
        "sha256": sha256_of(table),
        "size": table.stat().st_size,
    }
    if entry is not None:
        # Keep any fields the old entry carried that this tool does not know.
        for key, value in entry.items():
            if key not in ("sha256", "size"):
                files[table.name][key] = value
    constants_entry = files.get("ple_constants.json")
    if constants_entry is not None:
        constants_entry["sha256"] = hashlib.sha256(constants_path.read_bytes()).hexdigest()
        constants_entry["size"] = constants_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"updated {constants_path.name} and {manifest_path.name}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", required=True, type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path)
    parser.add_argument("--rows", type=int, default=0, help="0: every row")
    parser.add_argument("--chunk-rows", type=int, default=131072)
    parser.add_argument("--update-metadata", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="remove ngram_table.bin after a verified conversion (reclaims the fp16 copy)",
    )
    args = parser.parse_args(argv)

    constants = json.loads((args.install / "ple_constants.json").read_text())
    row_dim = int(constants["ple_head_dim"])
    total = row_count(constants)
    rows = args.rows if args.rows > 0 else total
    if rows > total:
        raise SystemExit(f"--rows {rows} exceeds the table's {total}")
    source = args.install / constants.get("table_file", "ngram_table.bin")
    destination = args.out or (args.install / "ngram_table.q4")
    if destination == source:
        raise SystemExit("refusing to overwrite the source table")
    if destination.exists():
        raise SystemExit(f"{destination} already exists; remove it first")

    convert(source, destination, rows, row_dim, args.chunk_rows)
    groups = row_dim // GROUP
    row_bytes = row_dim // 2 + groups * 4
    size = destination.stat().st_size
    expected = HEADER_BYTES + rows * row_bytes
    if size != expected:
        raise SystemExit(f"{destination} is {size} bytes; expected {expected}")
    print(
        f"wrote {destination} ({size / 1e9:.2f} GB, "
        f"{size / source.stat().st_size:.1%} of the fp16 table)"
    )
    if args.verify or args.update_metadata or args.delete_source:
        verify(source, destination, rows, row_dim)
    if args.update_metadata:
        if rows != total:
            raise SystemExit("--update-metadata needs a full table (no --rows)")
        update_metadata(args.install, destination)
    if args.delete_source:
        if rows != total:
            raise SystemExit("--delete-source needs a full table")
        source.unlink()
        print(f"removed {source}; reclaimed {expected / 1e9:.1f} GB of the fp16 copy")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
