#!/usr/bin/env python3
"""Long-context decode measurement: KV pressure and decode-rate drift.

Three cells, three streaming requests against one server:

  A: the long prompt,  128 new tokens  -> prefill under KV pressure
  B: a short prompt, 1024 new tokens   -> sustained long generation
  C: the long prompt, 1024 new tokens  -> both at once

The long prompt is 8,875 characters, which is roughly 2,400 tokens -- the
number that matters is the server's own `prompt_tokens`, and every row prints
it, so the context this measures is read from the answer rather than guessed
from the string. Decode rates come from the server's footer lines, which are
authoritative; the per-chunk timestamps only say how the rate moved *within*
one generation, and their character counts are converted to tokens with that
request's own `usage.completion_tokens`, never with a guessed
characters-per-token.

Every cell owes one row and the run owes one footer per cell. A stream that
produced no content, a response with no usage chunk, a drift capture too short
to split in half and a log holding a different number of footers than the run
sent requests are each printed as `NOT MEASURED`, and `main()` returns 1 for
the run; the banner says `STOPPED`, not `COMPLETE`. A server that exits during
load is `NOT RUN` and no request is sent to it.

Usage: python3 benchmark/tinytitan_longctx.py [model] [--port N]

    python3 benchmark/tinytitan_longctx.py models/qwen3.5_4B_4Bit

With no argument it runs `TINYTITAN_BENCH_MODEL`, or the shipped default.
"""

import argparse
import http.client
import json
import os
import re
import subprocess
import sys
import time

from tinytitan_profile import (
    bench_model,
    benchmark_log_path,
    resolve_api_model,
    run_stamp,
    server_command,
    server_environment,
    wait_for_health,
)

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BIN = os.path.join(BASE, ".build", "release", "TinyTitanServer")
DEFAULT_PORT = 8091
SERVER_LOAD_TIMEOUT = 2400
REQUEST_TIMEOUT_SECONDS = 1800
FOOTER_RE = re.compile(r"decode_tok_s=([0-9.]+)")
MIN_DRIFT_SAMPLES = 4
USAGE = "usage: tinytitan_longctx.py [model] [--port N]"


def build_long_prompt():
    para = (
        "The TinyTitan inference engine executes a 40-layer mixture-of-experts "
        "network on Apple Silicon via Metal with routed-expert pread streaming. "
        "This paragraph repeats to build a long context for decode-pressure "
        "measurement. "
    )
    return "Continue the technical discussion: " + para * 40


LONG_PROMPT = build_long_prompt()
SHORT_PROMPT = "Write a detailed essay about the history of computing."

# (label, prompt, max_completion_tokens) -- the whole run is this table.
CELLS = (
    ("A:long+128", LONG_PROMPT, 128),
    ("B:short+1024", SHORT_PROMPT, 1024),
    ("C:long+1024", LONG_PROMPT, 1024),
)


class ConfigError(ValueError):
    """A run configuration the driver refuses rather than guesses at."""


def results_directory():
    return os.path.join(BASE, "benchmark", "benchmark-results")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="tinytitan_longctx.py",
        description="Long-context decode pressure and rate drift.",
    )
    parser.add_argument("model", nargs="?", default=None)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    model = args.model if args.model is not None else bench_model()
    if not str(model).strip():
        raise ConfigError(f"an empty model path is not an install -- {USAGE}")
    if args.port < 1024 or args.port > 65535:
        raise ConfigError(f"--port {args.port} is not a port a server can bind -- {USAGE}")
    return str(model), args.port


def launch_server(model, port):
    log_path = benchmark_log_path("longctx_server.log")
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            server_command(BIN, port, model=model),
            env=server_environment(),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    # The parent's copy of the descriptor closes when the `with` block exits;
    # the child holds its own duplicate and is the only writer left.
    return proc


def stop_server(proc):
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def rss_mb(pid):
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)])
        return int(out.strip()) / 1024
    except Exception:
        return None


def request(prompt, max_new, label, model_id, port):
    """Stream one request and return what was observed.

    Nothing is printed here: `row_report` decides whether the observation is a
    measurement, so a stream that produced nothing cannot be displayed as one.
    """
    payload = json.dumps(
        {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 0.0,
            "max_completion_tokens": max_new,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=REQUEST_TIMEOUT_SECONDS)
    start = time.time()
    conn.request(
        "POST", "/v1/chat/completions", body=payload, headers={"Content-Type": "application/json"}
    )
    resp = conn.getresponse()
    buf = ""
    usage = None
    ttft = None
    chunk_times = []
    cum = 0
    while True:
        chunk = resp.read(8192)
        if not chunk:
            break
        now = time.time() - start
        buf += chunk.decode("utf-8", errors="ignore")
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            line = line.strip()
            if not line.startswith("data:"):
                continue
            js = line[5:].strip()
            if js in ("[DONE]", ""):
                continue
            try:
                event = json.loads(js)
            except json.JSONDecodeError:
                continue
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                text = (choice.get("delta") or {}).get("content")
                if text:
                    if ttft is None:
                        ttft = now
                    cum += len(text)
        if cum and (not chunk_times or chunk_times[-1][1] != cum):
            chunk_times.append((now, cum))
    conn.close()
    wall = time.time() - start
    return {
        "label": label,
        "wall": wall,
        "ttft": ttft,
        "pt": int(usage["prompt_tokens"]) if usage and usage.get("prompt_tokens") else 0,
        "ct": int(usage["completion_tokens"]) if usage and usage.get("completion_tokens") else 0,
        "chunk_times": chunk_times,
    }


def _time_at(times, chars):
    """When the generation reached `chars`, interpolating between samples."""
    if chars <= times[0][1]:
        return times[0][0]
    # A sliding window: `times[1:]` is one shorter than `times` by construction.
    for (t_prev, c_prev), (t, c) in zip(times, times[1:], strict=False):
        if c >= chars:
            if c == c_prev:
                return t
            return t_prev + (chars - c_prev) / (c - c_prev) * (t - t_prev)
    return times[-1][0]


def segment_rate(times, lo, hi, ct):
    """Tokens per second over the [lo, hi) fraction of the generation's characters.

    The characters-to-tokens divisor is the request's own completion count, so
    the unit is measured. `rate_at` used to divide by a hardcoded 4.0 and call
    the result tok/s, and it took its first number from `times[:n//2]` -- the
    last half *of the first half*, which is the second quarter of the run.
    """
    if not times or ct <= 0:
        return None
    total_chars = times[-1][1]
    if total_chars <= 0:
        return None
    c0, c1 = total_chars * lo, total_chars * hi
    dt = _time_at(times, c1) - _time_at(times, c0)
    if dt <= 0:
        return None
    tokens = (c1 - c0) * ct / total_chars
    return tokens / dt


def row_report(label, row, footer_rate=None):
    """(lines, status) for one streamed request, judged on what it produced."""
    if row.get("ttft") is None:
        return (
            [
                f"NOT MEASURED: {label} -- the stream ended without a single content "
                "delta, so there is no first token to time and no rate to report."
            ],
            1,
        )
    if not row.get("ct"):
        return (
            [
                f"NOT MEASURED: {label} -- the stream ended with no usage chunk, so the "
                "token count that turns characters into a rate is unknown."
            ],
            1,
        )
    decode_span = row["wall"] - row["ttft"]
    client = row["ct"] / decode_span if decode_span > 0 else None
    rate = (
        f"decode={footer_rate:.2f} tok/s (server footer)"
        if footer_rate is not None
        else "decode=no server footer for this request"
    )
    estimate = f", client estimate {client:.2f} tok/s" if client is not None else ""
    return (
        [
            f"  [{label}] wall={row['wall']:.1f}s ttft={row['ttft']:.1f}s "
            f"{row['pt']} prompt tokens {row['ct']} completion tokens "
            f"{rate}{estimate}"
        ],
        0,
    )


def drift_report(label, times, ct):
    """(lines, status) for how the rate moved within one generation."""
    if len(times) < MIN_DRIFT_SAMPLES:
        return (
            [
                f"NOT MEASURED: {label} drift -- {len(times)} timestamped samples for a "
                f"generation that needs at least {MIN_DRIFT_SAMPLES} to split in half."
            ],
            1,
        )
    first = segment_rate(times, 0.0, 0.5, ct)
    last = segment_rate(times, 0.5, 1.0, ct)
    if first is None or last is None:
        return (
            [f"NOT MEASURED: {label} drift -- the capture spans no measurable time."],
            1,
        )
    total_chars = times[-1][1]
    span = times[-1][0] - times[0][0]
    per_token = total_chars / ct if ct else float("nan")
    return (
        [
            f"  {label} decode drift: first-half={first:.2f} tok/s "
            f"last-half={last:.2f} tok/s drift={first - last:.2f} tok/s",
            f"    ({total_chars} chars over {span:.1f}s at {per_token:.2f} chars/token, "
            f"from the request's own {ct} completion tokens)",
        ],
        0,
    )


def rss_report(base, after_a, after_c):
    """(lines, status) for the resident-size samples around the long prefills."""
    if base is None:
        return (
            ["NOT MEASURED: RSS -- the baseline `ps` sample failed, so no delta means anything."],
            1,
        )
    shown = (
        f"{after_a:.0f}" if after_a is not None else "unknown",
        f"{after_c:.0f}" if after_c is not None else "unknown",
    )
    lines = [f"\nRSS: base={base:.0f}MB afterA={shown[0]}MB afterC={shown[1]}MB"]
    if after_a is None or after_c is None:
        lines.append("  WARNING: a later RSS sample failed; that column is unknown, not zero.")
    return (lines, 0)


def footer_report(rates, expected):
    """(lines, status) for the server's own footer lines against the requests sent."""
    if len(rates) != expected:
        return (
            [
                f"NOT MEASURED: server footers -- {len(rates)} decode footers for "
                f"{expected} requests."
            ],
            1,
        )
    lines = [f"server footers: {len(rates)} of {expected} requests"]
    for label, rate in zip([cell[0] for cell in CELLS], rates, strict=True):
        lines.append(f"  [{label}] footer decode={rate:.2f} tok/s")
    return (lines, 0)


def read_footer_rates(log_path):
    rates = []
    try:
        with open(log_path, encoding="utf-8") as handle:
            for line in handle:
                match = FOOTER_RE.search(line)
                if match:
                    rates.append(float(match.group(1)))
    except OSError as exc:
        print(f"  WARNING: could not read the server log {log_path}: {exc}", flush=True)
    return rates


def run(model, port):
    print(f"# Model: {model}", flush=True)
    print(f"# Long prompt: {len(LONG_PROMPT)} characters", flush=True)
    print(f"# Port: {port}, cells: {', '.join(cell[0] for cell in CELLS)}", flush=True)
    proc = None
    statuses = []
    rows = {}
    base_rss = post_a_rss = post_c_rss = None
    try:
        try:
            proc = launch_server(model, port)
        except Exception as exc:
            print(
                f"NOT RUN: the server could not be launched -- {type(exc).__name__}: {exc}",
                flush=True,
            )
            return 1
        print(f"server pid {proc.pid}, waiting for /health ...", flush=True)
        if not wait_for_health(proc, port, timeout=SERVER_LOAD_TIMEOUT):
            print(
                f"NOT RUN: the server process exited before /health answered -- see "
                f"{benchmark_log_path('longctx_server.log')}",
                flush=True,
            )
            return 1
        model_id = resolve_api_model(port)
        base_rss = rss_mb(proc.pid)
        for index, (label, prompt, max_new) in enumerate(CELLS):
            try:
                rows[label] = request(prompt, max_new, label, model_id, port)
            except Exception as exc:
                print(f"FAILED: {label} -- {type(exc).__name__}: {exc}", flush=True)
                rows[label] = None
            if index == 0:
                post_a_rss = rss_mb(proc.pid)
        post_c_rss = rss_mb(proc.pid)
    finally:
        stop_server(proc)

    rates = read_footer_rates(benchmark_log_path("longctx_server.log"))
    lines = []
    for index, (label, _prompt, _max_new) in enumerate(CELLS):
        row = rows.get(label)
        if row is None:
            lines.append(f"NOT MEASURED: {label} -- the request never returned a row.")
            statuses.append(1)
            continue
        footer = rates[index] if len(rates) == len(CELLS) else None
        cell_lines, status = row_report(label, row, footer)
        lines.extend(cell_lines)
        statuses.append(status)
        if label.startswith(("B", "C")):
            drift_lines, drift_status = drift_report(label, row["chunk_times"], row["ct"])
            lines.extend(drift_lines)
            statuses.append(drift_status)
    rss_lines, rss_status = rss_report(base_rss, post_a_rss, post_c_rss)
    lines.extend(rss_lines)
    statuses.append(rss_status)
    footer_lines, footer_status = footer_report(rates, len(CELLS))
    lines.extend(footer_lines)
    statuses.append(footer_status)

    for line in lines:
        print(line, flush=True)

    status = 1 if any(value for value in statuses) else 0
    summary = {
        "model": model,
        "port": port,
        "cells": len(CELLS),
        "requests_sent": sum(1 for label in rows if rows.get(label) is not None),
        "status": status,
        "rows": {
            label: {
                "wall": row["wall"],
                "ttft": row["ttft"],
                "prompt_tokens": row["pt"],
                "completion_tokens": row["ct"],
                "samples": len(row["chunk_times"]),
            }
            if row
            else None
            for label, row in rows.items()
        },
        "server_footer_rates": rates,
    }
    outdir = results_directory()
    os.makedirs(outdir, exist_ok=True)
    stamp = run_stamp()
    summary_path = os.path.join(outdir, f"longctx-{stamp}.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nSaved: {summary_path}", flush=True)
    return status


def main():
    try:
        model, port = parse_args()
    except ConfigError as exc:
        print(f"REFUSED: {exc}", flush=True)
        return 2
    status = run(model, port)
    if status == 0:
        print("\n" + "=" * 78 + "\nLONG CONTEXT COMPLETE\n" + "=" * 78, flush=True)
    else:
        print("\n" + "=" * 78 + "\nLONG CONTEXT STOPPED\n" + "=" * 78, flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
