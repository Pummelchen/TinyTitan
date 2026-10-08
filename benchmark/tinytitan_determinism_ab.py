#!/usr/bin/env python3
"""Cross-process greedy determinism check: two fresh servers, greedy at
`max_tokens` (default 512) for the essay and digit-cycle prompts, comparing the
CONTENT deltas token by token (never the raw SSE bytes). A raw-byte comparison is
invalid: each response embeds a per-request chatcmpl id and created timestamp, so
identical content hashes differently across processes. first_diff_index=None means
the streams are identical token-for-token.

A pair of streams that carried no content is not a result: `extract_deltas` returns
[] for a response that is not an SSE stream, and two empty lists neither differ nor
differ in hash, so the conclusion prints only when every prompt streamed content on
both sides. A difference is a result and exits 0; a prompt that streamed nothing
exits 1.

Usage: python3 benchmark/tinytitan_determinism_ab.py [max_tokens]
"""

import hashlib
import http.client
import json
import os
import subprocess
import sys
import time

from tinytitan_profile import (
    DEFAULT_MODEL_PATH,
    benchmark_log_path,
    server_command,
    server_environment,
    resolve_api_model,
)

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BIN = os.path.join(BASE, ".build", "release", "TinyTitanServer")
MODEL = str(DEFAULT_MODEL_PATH)
PORT = 8114
PROMPTS = [
    ("essay", "Write a detailed essay about the history of computing."),
    (
        "digits",
        "Write the digits 1,2,3,4,5,6,7,8,9,0 over and over in sequence, separated by commas, without stopping.",
    ),
]


def extract_deltas(resp):
    """Content deltas only — the raw bytes contain the per-request chatcmpl id
    and created timestamp, so hashing/compare the stream itself is invalid."""
    deltas = []
    for raw in resp:
        line = raw.decode()
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :].strip()
        if payload == "[DONE]":
            break
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        for choice in obj.get("choices", []):
            content = choice.get("delta", {}).get("content")
            if content is not None:
                deltas.append(content)
    return deltas


def first_diff_index(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return None


def run_server(max_tokens):
    log = open(benchmark_log_path("tinytitan_determinism_server.log"), "w", encoding="utf-8")
    proc = subprocess.Popen(
        server_command(BIN, PORT, model=MODEL),
        env=server_environment(),
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    start = time.time()
    while time.time() - start < 120:
        if proc.poll() is not None:
            raise SystemExit("server failed to start")
        try:
            conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=1)
            conn.request("GET", "/health")
            if "ok" in conn.getresponse().read().decode():
                conn.close()
                break
            conn.close()
        except OSError:
            pass
        time.sleep(0.05)

    result = {}
    for name, prompt in PROMPTS:
        payload = json.dumps(
            {
                "model": resolve_api_model(PORT),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "top_p": 0.95,
                "top_k": 20,
                "presence_penalty": 0.0,
                "max_completion_tokens": max_tokens,
                "stream": True,
            }
        ).encode()
        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=1800)
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        deltas = extract_deltas(resp)
        conn.close()
        result[name] = deltas
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    time.sleep(0.5)
    return result


def parse_max_tokens(argv):
    """(tokens, error) for the driver's one argument, read when it runs.

    It used to be a module-level `int(sys.argv[1])`, which made every importer of
    this file run the cast against the *importing program's* argv.
    """
    if len(argv) < 2:
        return 512, None
    try:
        value = int(argv[1])
    except ValueError:
        return None, f"max_tokens must be an integer, got {argv[1]!r}"
    if value < 1:
        return None, f"max_tokens must be at least 1, got {value}"
    return value, None


def verdict(pairs, max_tokens):
    """(lines, status): the per-prompt rows, then the conclusion or why there is none.

    Two streams that carried no content look identical to the comparison — no first
    difference, and `sha256` of two empty answers is the same hash — so "identical"
    has to be earned by content on both sides before it can be called determinism.
    """
    lines, unmeasured = [], []
    for name, da, db in pairs:
        diff = first_diff_index(da, db)
        joined_a = "".join(da)
        joined_b = "".join(db)
        sha_a = hashlib.sha256(joined_a.encode()).hexdigest()[:16]
        sha_b = hashlib.sha256(joined_b.encode()).hexdigest()[:16]
        lines.append(
            f"{name}: tokens A={len(da)} B={len(db)} "
            + f"first_diff_index={diff} "
            + f"content_sha256={sha_a}=={sha_b} equal={sha_a == sha_b}"
        )
        if not da or not db:
            side = "A and B" if not da and not db else ("A" if not da else "B")
            lines.append(f"  NOT MEASURED: {name} carried no content deltas on side {side}")
            unmeasured.append(name)
    if unmeasured:
        lines.append(
            f"{len(unmeasured)} of {len(pairs)} prompt(s) streamed no content, so "
            "nothing was compared: determinism is not established and this run exits 1."
        )
        return lines, 1
    lines.append(
        f"None for both => {max_tokens}-token greedy is deterministic across fresh processes"
    )
    return lines, 0


def main():
    max_tokens, error = parse_max_tokens(sys.argv)
    if error:
        print(f"error: {error}", file=sys.stderr)
        print("usage: python3 benchmark/tinytitan_determinism_ab.py [max_tokens]", file=sys.stderr)
        return 2
    print(f"max_tokens={max_tokens} prompts={[n for n, _ in PROMPTS]}")
    print("server A...", flush=True)
    a = run_server(max_tokens)
    print("server B...", flush=True)
    b = run_server(max_tokens)
    lines, status = verdict([(name, a[name], b[name]) for name, _ in PROMPTS], max_tokens)
    for line in lines:
        print(line)
    return status


if __name__ == "__main__":
    sys.exit(main())
