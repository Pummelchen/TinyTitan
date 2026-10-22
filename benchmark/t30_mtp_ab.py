#!/usr/bin/env python3
"""T30: Ornith 4-bit MTP on/off at 64 and 128 generated tokens.

Two server arms per token count: scalar decode and native MTP. The prompt is
the multiplication table, which MTP predicts well, so acceptance is at the
high end rather than a prose worst case. Requests are pure greedy (presence 0)
because the decode loop only engages the draft under that condition. Text
hashes are compared between arms so a speed number can never be reported for
a different answer.

    python3 benchmark/t30_mtp_ab.py --pairs 2 --out .build/t30-mtp-ab.json
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import pathlib
import re
import signal
import subprocess
import time

from tinytitan_profile import pgrep_answer

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = ROOT / ".build/release/TinyTitanServer"
TARGET = ROOT / "models/ornith-1.5_35B_A3B_4Bit"
SIDECAR = ROOT / "models/ornith-1.5_35B_A3B_MTP_4Bit"
PORT = 8098
PROMPT = (
    "Write out the multiplication table for 7, from 7 x 1 to 7 x 30, "
    "one line each, in the exact format '7 x N = M'. Output only the "
    "table, nothing else."
)


def running_servers() -> list[str]:
    verdict, lines = pgrep_answer(["-fl", "TinyTitanServer"])
    if verdict == "unknown":
        raise RuntimeError("cannot check for a running TinyTitanServer: " + "; ".join(lines))
    return [line for line in lines if "bash -c" not in line and "python" not in line]


def stop(process: subprocess.Popen) -> None:
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


def wait_ready(process: subprocess.Popen, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            return False
        try:
            connection = http.client.HTTPConnection("127.0.0.1", PORT, timeout=2)
            connection.request("GET", "/v1/models")
            response = connection.getresponse()
            response.read()
            connection.close()
            if response.status == 200:
                return True
        except OSError:
            pass
        time.sleep(1)
    return False


def request(model: str, tokens: int) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0,
            "presence_penalty": 0,
            "seed": 41,
            "max_completion_tokens": tokens,
            "stream": False,
        },
        separators=(",", ":"),
    )
    started = time.time()
    connection = http.client.HTTPConnection("127.0.0.1", PORT, timeout=900)
    connection.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    response = connection.getresponse()
    raw = response.read()
    connection.close()
    elapsed = time.time() - started
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status}: {raw[:1000]!r}")
    payload = json.loads(raw)
    text = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content", "")
    return {
        "wall_s": elapsed,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
        "completion_tokens": (payload.get("usage") or {}).get("completion_tokens"),
    }


def run_arm(tokens: int, mtp: bool, pairs: int, log: pathlib.Path) -> dict:
    if running_servers():
        raise RuntimeError("another TinyTitanServer is running")
    log.unlink(missing_ok=True)
    cmd = [
        str(BIN),
        "--model",
        str(TARGET),
        "--port",
        str(PORT),
        "--max-context",
        "8192",
        "--thinking",
        "off",
        "--prompt-cache-mode",
        "off",
        "--prompt-cache-memory-mib",
        "0",
    ]
    if mtp:
        cmd += ["--mtp-model", str(SIDECAR), "--mtp-memory-mib", "384"]
    environment = dict(os.environ)
    environment["TINYTITAN_RUNNER_STATS"] = "1"
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=ROOT,
            env=environment,
            start_new_session=True,
        )
    try:
        if not wait_ready(process, 600):
            raise RuntimeError("server did not become healthy")
        model = None
        connection = http.client.HTTPConnection("127.0.0.1", PORT, timeout=10)
        connection.request("GET", "/v1/models")
        listing = json.loads(connection.getresponse().read())
        connection.close()
        model = listing["data"][0]["id"]
        rows = [request(model, tokens) for _ in range(pairs)]
    finally:
        stop(process)
    text = log.read_text(encoding="utf-8", errors="replace")
    rates = re.findall(
        r"generation prefill_s=([\d.]+) decode_s=([\d.]+) decode_tok_s=([\d.]+)",
        text,
    )
    mtp_rows = re.findall(
        r"TinyTitan mtp drafted=(\d+) accepted=(\d+) acceptance=([\d.]+)% "
        r"target_passes=(\d+) emitted_per_pass=([\d.]+) .*?"
        r"decode_s=([\d.]+) decode_tok_s=([\d.]+)",
        text,
    )
    for index, row in enumerate(rows):
        if index < len(rates):
            row["server_decode_tok_s"] = float(rates[index][2])
            row["server_decode_s"] = float(rates[index][1])
        if mtp and index < len(mtp_rows):
            row["drafted"] = int(mtp_rows[index][0])
            row["accepted"] = int(mtp_rows[index][1])
            row["acceptance"] = float(mtp_rows[index][2])
            row["emitted_per_pass"] = float(mtp_rows[index][4])
            row["server_decode_s"] = float(mtp_rows[index][5])
            row["server_decode_tok_s"] = float(mtp_rows[index][6])
    return {"tokens": tokens, "mtp": mtp, "rows": rows, "log": str(log)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = {
        "target": str(TARGET),
        "sidecar": str(SIDECAR),
        "prompt": PROMPT,
        "pairs": args.pairs,
        "results": [],
    }
    for tokens in args.tokens:
        for mtp in (False, True):
            arm = "on" if mtp else "off"
            print(f"tokens={tokens} mtp={arm}", flush=True)
            result = run_arm(
                tokens,
                mtp,
                args.pairs,
                ROOT / f".build/t30-mtp-{tokens}-{arm}.log",
            )
            payload["results"].append(result)
            out = pathlib.Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    # Compare text per token count, in arm order.
    mismatches = []
    for tokens in args.tokens:
        arms = [row for row in payload["results"] if row["tokens"] == tokens]
        if len(arms) == 2:
            for index, row in enumerate(arms[0]["rows"]):
                other = arms[1]["rows"][index]
                if row["text_sha256"] != other["text_sha256"]:
                    mismatches.append({"tokens": tokens, "row": index})
    payload["text_mismatches"] = mismatches
    pathlib.Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": args.out, "text_mismatches": len(mismatches)}))
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
