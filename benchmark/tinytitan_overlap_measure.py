#!/usr/bin/env python3
"""GPU/IO overlap analysis: per-token stage splits (TINYTITAN_RUNNER_STATS) and
per-role GPU spans (TINYTITAN_KERNEL_STATS) on a warm cache. Two greedy
requests of `max_tokens` tokens; the second is the representative warm
measurement.

The three sections are exactly the three channels this reads: generation
footers, runner stage splits, and the kernel `role=`/`total_gpu_ms=` lines. The
engine also prints `TinyTitan kernel busy_ms=… span_ms=…`, which names the
busy/span gap directly, and this driver does not read it -- so the overlap is
inferred from the stage splits, not measured by that line.

    cd benchmark && ./tinytitan_overlap_measure.py [max_tokens]

`max_tokens` defaults to 512. This is a model run: it starts a server process,
so the machine preconditions in `AGENTS.md` apply and no second model process
may already be running.

A section that captured no line is not a zero measurement, it is an instrument
that did not run: the driver prints which section was empty and exits 1. Every
header carries the number of lines under it, because two requests were sent and
a section of one is a partial run.
"""

import http.client
import json
import os
import subprocess
import sys
import time

from tinytitan_profile import (
    DEFAULT_MODEL_PATH,
    benchmark_log_path,
    parse_max_tokens,
    resolve_api_model,
    server_command,
)

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BIN = os.path.join(BASE, ".build", "release", "TinyTitanServer")
MODEL = os.environ.get("TINYTITAN_BENCH_MODEL", str(DEFAULT_MODEL_PATH))
PORT = 8111
PROMPT = "Write a detailed essay about the history of computing."


def capture(lines):
    """(gen, runner, kernels) classified out of the server log."""
    gen, runner, kernels = [], [], []
    for line in lines:
        if "TinyTitan generation" in line and "decode_tok_s=" in line:
            gen.append(line.strip())
        if "TinyTitan runner" in line and "cb1_ms=" in line:
            runner.append(line.strip())
        if "TinyTitan kernel role=" in line:
            kernels.append(line.strip())
        if "TinyTitan kernel total_gpu_ms=" in line:
            kernels.append(line.strip())
    return gen, runner, kernels


def header(name: str, count: int) -> str:
    """A section header that says how many lines it holds, so a section of one is
    visibly a partial run rather than reading like the two it was sent for."""
    return f"=== {name} ({count} {'line' if count == 1 else 'lines'}) ==="


def verdict(gen, runner, kernels):
    """(lines, exit status) for what was captured.

    A section with no line is not a zero measurement: each one exists only if the
    server printed the matching channel, so an empty body means the instrument was
    unplugged -- a binary that dropped the format, an env var the release CLI no
    longer honours, or a run that never reached decode. Say which, and fail.
    """
    sections = [
        ("generation footers", "decode_tok_s=", gen),
        ("runner stage splits", "cb1_ms=", runner),
        ("kernel GPU roles", "TinyTitan kernel role=", kernels),
    ]
    lines, missing = [], []
    for name, wanted, captured in sections:
        lines.append(header(name, len(captured)))
        lines.extend(captured)
        if not captured:
            missing.append(f"NOT MEASURED: {name} -- no log line contained {wanted!r}")
    if missing:
        lines += ["", *missing]
    return lines, 1 if missing else 0


def wait_for_health(proc, port=PORT, timeout=120):
    """Poll /health until the server answers, or return whether it never did."""
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            return False
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
            conn.request("GET", "/health")
            if "ok" in conn.getresponse().read().decode():
                conn.close()
                return True
            conn.close()
        except OSError:
            pass
        time.sleep(0.05)
    return True


def request_twice(max_tokens, port=PORT):
    """Two streamed greedy requests; the content is discarded, the log is the product."""
    payload = json.dumps(
        {
            "model": resolve_api_model(port),
            "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 0.0,
            "max_completion_tokens": max_tokens,
            "stream": True,
        }
    ).encode()
    for i in range(2):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1800)
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        while resp.read(8192):
            pass
        conn.close()
        print(f"request {i + 1} done")


def main() -> int:
    max_tokens, error = parse_max_tokens(sys.argv)
    if error:
        print(f"usage: {os.path.basename(sys.argv[0])} [max_tokens] -- {error}", file=sys.stderr)
        return 2

    env = dict(os.environ)
    env["TINYTITAN_RUNNER_STATS"] = "1"
    env["TINYTITAN_KERNEL_STATS"] = "1"
    log_path = benchmark_log_path("tinytitan_overlap.log")
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            server_command(BIN, PORT, model=MODEL), env=env, stdout=log, stderr=subprocess.STDOUT
        )
        if not wait_for_health(proc):
            print("server exited early", file=sys.stderr)
            return 1
        try:
            request_twice(max_tokens)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    with open(log_path, encoding="utf-8") as handle:
        lines = handle.readlines()
    printed, status = verdict(*capture(lines))
    for line in printed:
        print(line)
    return status


if __name__ == "__main__":
    sys.exit(main())
