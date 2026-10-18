#!/usr/bin/env python3
"""Aggregate throughput under concurrent requests, for one server configuration.

The engine's decode path exposes ~45 ms of expert I/O wait in a ~92 ms token on
8-bit, and the server ships `--max-concurrent-sequences 1`, so a second client
waits for the first. Two sequences in flight can overlap one's wait with the
other's compute, which is a throughput question rather than a latency one: the
per-request rate may fall while the aggregate rises.

    python3 benchmark/concurrency_sweep.py \
        --model ornith8=models/ornith-1.5_35B_A3B_8Bit \
        --sequences 1 2 4 --max-tokens 128 --out .build/concurrency-ornith8.json
"""

from __future__ import annotations

import argparse
import http.client
import json
import pathlib
import re
import subprocess
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROMPT = "Write a detailed technical explanation of how a compiler turns source code into a binary."


def wait_ready(port: int, process: subprocess.Popen, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            return False
        try:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            connection.request("GET", "/v1/models")
            response = connection.getresponse()
            response.read()
            connection.close()
            if response.status == 200:
                return True
        except OSError:
            pass
        time.sleep(0.5)
    return False


def model_id(port: int) -> str:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.request("GET", "/v1/models")
    payload = json.loads(connection.getresponse().read())
    connection.close()
    return payload["data"][0]["id"]


def one_request(
    port: int,
    name: str,
    max_tokens: int,
    results: dict,
    barrier: threading.Barrier,
    prompt_text: str,
):
    body = json.dumps(
        {
            "model": name,
            "messages": [{"role": "user", "content": prompt_text}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    )
    # All clients are released together, so the measurement window is the batch
    # and not the client startup order.
    barrier.wait()
    started = time.time()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1800)
    try:
        connection.request(
            "POST", "/v1/chat/completions", body, {"Content-Type": "application/json"}
        )
        payload = json.loads(connection.getresponse().read())
        results["payload"] = payload
    except OSError as error:  # noqa: BLE001 - reported, not raised
        results["error"] = str(error)
    finally:
        connection.close()
    results["elapsed"] = time.time() - started


def run_cell(
    label: str,
    path: str,
    sequences: int,
    port: int,
    max_tokens: int,
    rounds: int,
    prompt_text: str,
):
    rows = []
    for round_index in range(1, rounds + 1):
        log = ROOT / f".build/concurrency-{label}-s{sequences}-r{round_index}.log"
        log.unlink(missing_ok=True)
        with log.open("w") as handle:
            process = subprocess.Popen(
                [
                    str(ROOT / ".build/release/TinyTitanServer"),
                    "--model",
                    path,
                    "--thinking",
                    "off",
                    "--port",
                    str(port),
                    "--max-concurrent-sequences",
                    str(sequences),
                    # Room for every sequence plus the queue behind it.
                    "--queue-limit",
                    str(max(sequences, (sequences + 1) // 2)),
                ],
                stdout=handle,
                stderr=subprocess.STDOUT,
                cwd=ROOT,
                env={**__import__("os").environ, "TINYTITAN_RUNNER_STATS": "1"},
            )
        try:
            if not wait_ready(port, process, 300):
                rows.append(
                    {"sequences": sequences, "round": round_index, "error": "server failed"}
                )
                print(f"s{sequences} r{round_index} SERVER FAILED")
                continue
            name = model_id(port)
            results = [dict() for _ in range(sequences)]
            barrier = threading.Barrier(sequences)
            started = time.time()
            threads = [
                threading.Thread(
                    target=one_request,
                    args=(port, name, max_tokens, results[index], barrier, prompt_text),
                )
                for index in range(sequences)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            wall = time.time() - started
            tokens = sum(
                (r.get("payload") or {}).get("usage", {}).get("completion_tokens", 0)
                for r in results
            )
            errors = [r["error"] for r in results if "error" in r]
            row = {
                "sequences": sequences,
                "round": round_index,
                "wall_s": wall,
                "tokens": tokens,
                "aggregate_tok_s": tokens / wall if wall else 0.0,
                "errors": errors,
            }
            # The engine's own per-request rate, for the latency side of the trade.
            text = log.read_text(encoding="utf-8", errors="replace")
            rates = [float(m) for m in re.findall(r"decode_tok_s=([\d.]+)", text)]
            row["decode_tok_s"] = rates
            rows.append(row)
            print(
                f"s{sequences} r{round_index}  aggregate {row['aggregate_tok_s']:6.2f} tok/s  "
                f"wall {wall:6.1f}s  tokens {tokens:4d}  per-request {rates}"
            )
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
            time.sleep(2)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="label=path")
    parser.add_argument("--sequences", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--port", type=int, default=8094)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()

    label, path = arguments.model.split("=", 1)
    prompt_text = PROMPT
    if arguments.prompt_file:
        prompt_text = pathlib.Path(arguments.prompt_file).read_text(encoding="utf-8")
    rows = []
    for sequences in arguments.sequences:
        rows += run_cell(
            label,
            path,
            sequences,
            arguments.port,
            arguments.max_tokens,
            arguments.rounds,
            prompt_text,
        )

    baseline = [
        r["aggregate_tok_s"] for r in rows if r["sequences"] == 1 and "aggregate_tok_s" in r
    ]
    print("\n== aggregate throughput vs one sequence ==")
    for sequences in arguments.sequences:
        cell = [
            r["aggregate_tok_s"]
            for r in rows
            if r["sequences"] == sequences and "aggregate_tok_s" in r
        ]
        if not cell:
            continue
        mean = sum(cell) / len(cell)
        if sequences == 1:
            print(f"{sequences} seq  {mean:6.2f} tok/s  —")
        else:
            base = sum(baseline) / len(baseline) if baseline else mean
            print(f"{sequences} seq  {mean:6.2f} tok/s  {mean / base:5.2f}x")

    pathlib.Path(arguments.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\n{len(rows)} rows -> {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
