#!/usr/bin/env python3
"""T1/T2: reproducible width-1/2/4 batched-decode A/B.

Starts one server per width, warms it with fresh prompts, then measures the same
fixed prompts N times. Every measured prompt carries a per-run nonce so a
prompt-cache hit cannot silently remove prefill work. The output records wall
time, aggregate and per-row decode throughput, prefill/decode seconds parsed
from the server's own log line, and a SHA-256 of each generated text; the hashes
are compared across widths so a numerical divergence is a failed run, not a
number to report.

    python3 benchmark/maya_batch_ab.py \
        --model ornith4=models/ornith-1.5_35B_A3B_4Bit \
        --widths 1 2 4 --tokens 64 --runs 5 --warmups 3 \
        --out .build/maya-batch-ab-ornith4.json
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import http.client
import json
import os
import pathlib
import re
import signal
import statistics
import subprocess
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
HARNESS_LOCK = ROOT / ".build/maya-batch.lock"

PROMPTS = [
    (
        "Explain, step by step, how a CPU cache hierarchy works, including L1, "
        "L2, L3, inclusion, coherence, and the tradeoffs of associativity. "
    )
    * 8,
    (
        "Describe how a modern operating system schedules threads across cores, "
        "covering run queues, affinity, priority, and context switching. "
    )
    * 8,
    (
        "Walk through how a database engine plans and executes a join, including "
        "statistics, access paths, hash and merge joins, and spill behaviour. "
    )
    * 8,
    (
        "Explain how a TCP connection recovers from loss, covering congestion "
        "windows, fast retransmit, SACK, and the difference from flow control. "
    )
    * 8,
]


def running_servers() -> list[str]:
    """Every TinyTitanServer process on the host, as pgrep sees it.

    The benchmark starts one server per width and must never overlap with a
    manually started server. Two instances on a 24 GiB Mac can exhaust unified
    memory and take the machine down.
    """
    try:
        found = subprocess.run(
            ["pgrep", "-fl", "TinyTitanServer"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return []
    # `pgrep -f` also matches a shell command that merely mentions the
    # binary (this harness's own `pgrep`, or a wrapper). Only a line whose
    # executable is the server counts.
    return [
        line
        for line in found.stdout.splitlines()
        if line.strip()
        and "TinyTitanServer" in line
        and "bash -c" not in line
        and "pgrep" not in line
        and "python" not in line
    ]


def require_no_server(port: int) -> None:
    existing = running_servers()
    if existing:
        raise RuntimeError(
            "refusing to start: a TinyTitanServer is already running; "
            "running two can crash the machine:\n" + "\n".join(existing)
        )
    probe = http.client.HTTPConnection("127.0.0.1", port, timeout=0.5)
    try:
        probe.request("GET", "/v1/models")
        probe.getresponse().read()
    except OSError:
        return
    finally:
        probe.close()
    raise RuntimeError(f"refusing to start: port {port} already answers")


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        process.terminate()
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()
        process.wait(timeout=30)
    deadline = time.time() + 30
    while time.time() < deadline:
        if not running_servers():
            return
        time.sleep(0.5)
    raise RuntimeError(
        "a TinyTitanServer survived cleanup; refusing to start another:\n"
        + "\n".join(running_servers())
    )


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


def served_model_id(port: int) -> str:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.request("GET", "/v1/models")
    payload = json.loads(connection.getresponse().read())
    connection.close()
    return payload["data"][0]["id"]


def decode_rates(log: pathlib.Path) -> list[dict[str, float]]:
    text = log.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"generation prefill_s=([\d.]+) decode_s=([\d.]+) decode_tok_s=([\d.]+)",
        text,
    )
    return [{"prefill_s": float(a), "decode_s": float(b), "tok_s": float(c)} for a, b, c in matches]


def one_request(
    port: int,
    model: str,
    prompt: str,
    max_tokens: int,
    results: dict,
    barrier: threading.Barrier,
) -> None:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        }
    )
    barrier.wait()
    started = time.time()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2400)
    try:
        connection.request(
            "POST", "/v1/chat/completions", body, {"Content-Type": "application/json"}
        )
        payload = json.loads(connection.getresponse().read())
        results["payload"] = payload
    except OSError as error:  # noqa: BLE001 - recorded, not raised
        results["error"] = str(error)
    finally:
        connection.close()
    results["elapsed"] = time.time() - started


def run_batch(
    port: int,
    model: str,
    prompts: list[str],
    max_tokens: int,
    nonce: str,
) -> dict:
    results = [dict() for _ in prompts]
    barrier = threading.Barrier(len(prompts))
    threads = [
        threading.Thread(
            target=one_request,
            args=(port, model, prompt + f" [nonce {nonce}]", max_tokens, holder, barrier),
        )
        for prompt, holder in zip(prompts, results, strict=True)
    ]
    started = time.time()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.time() - started
    rows = []
    total_tokens = 0
    for holder in results:
        payload = holder.get("payload") or {}
        choices = payload.get("choices") or [{}]
        text = (choices[0].get("message") or {}).get("content", "")
        usage = payload.get("usage") or {}
        tokens = int(usage.get("completion_tokens") or 0)
        total_tokens += tokens
        rows.append(
            {
                "elapsed_s": holder.get("elapsed"),
                "error": holder.get("error"),
                "completion_tokens": tokens,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "text_prefix": text[:120],
            }
        )
    return {
        "nonce": nonce,
        "wall_s": wall,
        "completion_tokens": total_tokens,
        "aggregate_tok_s": total_tokens / wall if wall else 0.0,
        "rows": rows,
    }


def run_width(
    label: str,
    path: str,
    width: int,
    tokens: int,
    runs: int,
    warmups: int,
    port: int,
    max_context: int | None,
    serial_prompts: int,
) -> dict:
    log = ROOT / f".build/maya-batch-{label}-w{width}-t{tokens}.log"
    log.unlink(missing_ok=True)
    prompts = PROMPTS[: serial_prompts if width == 1 else width]
    require_no_server(port)
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
                str(width),
                "--queue-limit",
                str(max(4, width)),
            ]
            + (["--max-context", str(max_context)] if max_context else []),
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=ROOT,
            start_new_session=True,
        )
    try:
        if not wait_ready(port, process, 300):
            return {"width": width, "tokens": tokens, "error": "server failed"}
        model = served_model_id(port)
        for index in range(warmups):
            run_batch(port, model, prompts, tokens, f"warm-{index}")
        measured = [
            run_batch(port, model, prompts, tokens, f"meas-{index}") for index in range(runs)
        ]
        rates = decode_rates(log)
        walls = [row["wall_s"] for row in measured]
        aggregates = [row["aggregate_tok_s"] for row in measured]
        per_row = [
            tokens / result["elapsed_s"]
            for run in measured
            for result in run["rows"]
            if result.get("elapsed_s")
        ]
        return {
            "width": width,
            "tokens": tokens,
            "runs": measured,
            "wall_median_s": statistics.median(walls),
            "wall_spread_s": max(walls) - min(walls),
            "aggregate_median_tok_s": statistics.median(aggregates),
            "per_row_median_tok_s": statistics.median(per_row) if per_row else 0.0,
            "server_decode_rates": rates[-len(measured) :] if rates else [],
        }
    finally:
        stop_server(process)


def text_mismatches(payload: dict) -> list[dict]:
    """Compare each width's text to width 1 for the same prompt and nonce.

    A nonce is deliberately different between measured runs so the prompt cache
    cannot hide prefill. That means run 1 and run 2 are different prompts and
    must not be compared to each other; the reference key includes the nonce.
    """
    reference = {}
    for cell in payload["results"]:
        if cell.get("width") == 1:
            for run in cell.get("runs") or []:
                for index, row in enumerate(run["rows"]):
                    reference[(cell["tokens"], run["nonce"], index)] = row["text_sha256"]
    mismatches = []
    for cell in payload["results"]:
        if cell.get("width") == 1:
            continue
        for run in cell.get("runs") or []:
            for index, row in enumerate(run["rows"]):
                key = (cell["tokens"], run["nonce"], index)
                if key in reference and row["text_sha256"] != reference[key]:
                    mismatches.append(
                        {
                            "width": cell["width"],
                            "tokens": cell["tokens"],
                            "row": index,
                            "nonce": run["nonce"],
                        }
                    )
    return mismatches


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="label=path")
    parser.add_argument("--widths", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--tokens", nargs="+", type=int, default=[64])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--port", type=int, default=8093)
    parser.add_argument("--max-context", type=int, default=8192)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    HARNESS_LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock = HARNESS_LOCK.open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another maya_batch_ab.py is running; only one server at a time") from None
    require_no_server(args.port)
    label, path = args.model.split("=", 1)
    payload = {
        "model": label,
        "path": path,
        "runs": args.runs,
        "warmups": args.warmups,
        "server": str(ROOT / ".build/release/TinyTitanServer"),
        "results": [],
    }
    for tokens in args.tokens:
        for width in args.widths:
            print(f"width={width} tokens={tokens}", flush=True)
            payload["results"].append(
                run_width(
                    label,
                    path,
                    width,
                    tokens,
                    args.runs,
                    args.warmups,
                    args.port,
                    args.max_context,
                    max(args.widths),
                )
            )
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    mismatches = text_mismatches(payload)
    payload["text_mismatches"] = mismatches
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "text_mismatches": len(mismatches)}))
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
