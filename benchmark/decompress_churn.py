#!/usr/bin/env python3
"""Does decompressing at a *layer's cadence* disturb decode?

The steady test (`docs/next-steps-2026-10-19.md`, item B1) cleared decompression
at one and two threads but could not say whether the real shape -- one
expert-sized frame every layer, ~1.6 MiB every ~1.8 ms -- behaves the same.

This runs a decode twice against freshly started servers, once alone and once
with `--threads` decompressors pacing themselves at `--cadence-ms`, and prints
both rates. Decompression goes through libzstd via ctypes rather than the CLI, so
process spawn (5-10 ms) cannot stand in for the burst.

    python3 benchmark/decompress_churn.py \
        --model ornith4=models/ornith-1.5_35B_A3B_4Bit \
        --sample models/ornith-1.5_35B_A3B_4Bit/packed_experts/layer_07.bin \
        --frame-mib 1.6 --cadence-ms 1.8 --threads 2
"""

from __future__ import annotations

import argparse
import ctypes
import http.client
import json
import pathlib
import subprocess
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROMPT = "Write a detailed technical explanation of how a compiler turns source code into a binary."


class Zstd:
    """The two calls this needs, through libzstd."""

    def __init__(self, library: str) -> None:
        self.lib = ctypes.CDLL(library)
        self.lib.ZSTD_compressBound.argtypes = [ctypes.c_size_t]
        self.lib.ZSTD_compressBound.restype = ctypes.c_size_t
        self.lib.ZSTD_compress.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.lib.ZSTD_compress.restype = ctypes.c_size_t
        self.lib.ZSTD_decompress.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self.lib.ZSTD_decompress.restype = ctypes.c_size_t
        self.lib.ZSTD_isError.argtypes = [ctypes.c_size_t]
        self.lib.ZSTD_isError.restype = ctypes.c_uint

    def compress(self, data: bytes, level: int = 1) -> bytes:
        bound = self.lib.ZSTD_compressBound(len(data))
        out = ctypes.create_string_buffer(bound)
        src = ctypes.create_string_buffer(data, len(data))
        size = self.lib.ZSTD_compress(out, bound, src, len(data), level)
        if self.lib.ZSTD_isError(size):
            raise RuntimeError("zstd compress failed")
        return out.raw[:size]


def start_server(path: str, port: int) -> subprocess.Popen:
    process = subprocess.Popen(
        [
            str(ROOT / ".build/release/TinyTitanServer"),
            "--model",
            path,
            "--port",
            str(port),
            "--thinking",
            "off",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=ROOT,
    )
    for _ in range(300):
        try:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            connection.request("GET", "/v1/models")
            connection.getresponse().read()
            connection.close()
            return process
        except OSError:
            time.sleep(0.5)
    return process


def one_decode(port: int, max_tokens: int) -> float:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    connection.request("GET", "/v1/models")
    model = json.loads(connection.getresponse().read())["data"][0]["id"]
    connection.close()
    started = time.time()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1800)
    connection.request(
        "POST",
        "/v1/chat/completions",
        json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": max_tokens,
                "temperature": 0,
            }
        ),
        {"Content-Type": "application/json"},
    )
    payload = json.loads(connection.getresponse().read())
    connection.close()
    tokens = payload["usage"]["completion_tokens"]
    return tokens / (time.time() - started)


def churn(zstd: Zstd, frame: bytes, plain_bytes: int, cadence_ms: float, stop: threading.Event):
    out = ctypes.create_string_buffer(plain_bytes)
    src = ctypes.create_string_buffer(frame, len(frame))
    period = cadence_ms / 1000.0
    while not stop.is_set():
        started = time.perf_counter()
        zstd.lib.ZSTD_decompress(out, plain_bytes, src, len(frame))
        elapsed = time.perf_counter() - started
        remaining = period - elapsed
        if remaining > 0:
            time.sleep(remaining)


def run(
    label: str,
    model_path: str,
    port: int,
    max_tokens: int,
    frame: bytes | None,
    plain_bytes: int,
    zstd: Zstd | None,
    cadence_ms: float,
    threads: int,
) -> float:
    process = start_server(model_path, port)
    stop = threading.Event()
    workers = []
    try:
        time.sleep(3)
        if frame is not None and threads:
            for _ in range(threads):
                worker = threading.Thread(
                    target=churn, args=(zstd, frame, plain_bytes, cadence_ms, stop), daemon=True
                )
                worker.start()
                workers.append(worker)
        rate = one_decode(port, max_tokens)
        print(f"  {label:<28} {rate:6.2f} tok/s")
        return rate
    finally:
        stop.set()
        for worker in workers:
            worker.join(timeout=5)
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
        time.sleep(3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="label=path")
    parser.add_argument("--sample", required=True, help="an expert layer file")
    parser.add_argument("--frame-mib", type=float, default=1.6)
    parser.add_argument("--cadence-ms", type=float, default=1.8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--port", type=int, default=8130)
    parser.add_argument("--library", default="/opt/homebrew/lib/libzstd.dylib")
    arguments = parser.parse_args()

    label, model_path = arguments.model.split("=", 1)
    zstd = Zstd(arguments.library)
    plain_bytes = int(arguments.frame_mib * 2**20)
    with open(arguments.sample, "rb") as handle:
        raw = handle.read(plain_bytes)
    frame = zstd.compress(raw, level=1)
    print(
        f"{label}: frame {len(raw) / 2**20:.2f} MiB -> {len(frame) / len(raw) * 100:.1f}%, "
        f"{arguments.threads} decompressor(s) at {arguments.cadence_ms} ms cadence"
    )

    for round_index in range(1, arguments.rounds + 1):
        # Interleave: the machine drifts, and an arm order fixed across rounds
        # would put the drift on one arm.
        if round_index % 2:
            clean = run(
                f"round {round_index} clean",
                model_path,
                arguments.port,
                arguments.max_tokens,
                None,
                plain_bytes,
                zstd,
                arguments.cadence_ms,
                0,
            )
            paced = run(
                f"round {round_index} paced",
                model_path,
                arguments.port,
                arguments.max_tokens,
                frame,
                plain_bytes,
                zstd,
                arguments.cadence_ms,
                arguments.threads,
            )
        else:
            paced = run(
                f"round {round_index} paced",
                model_path,
                arguments.port,
                arguments.max_tokens,
                frame,
                plain_bytes,
                zstd,
                arguments.cadence_ms,
                arguments.threads,
            )
            clean = run(
                f"round {round_index} clean",
                model_path,
                arguments.port,
                arguments.max_tokens,
                None,
                plain_bytes,
                zstd,
                arguments.cadence_ms,
                0,
            )
        print(f"round {round_index}: paced/clean = {paced / clean:.3f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
