#!/usr/bin/env python3
"""Long-generation decode benchmark (code-generation style): 512-token greedy
coding prompt, server-footer decode rates, 1 warmup + 3 measured runs per
quant, the mean over the three measured runs only.

Usage: python3 benchmark/tinytitan_longgen.py [model1] [model2] ...

With no argument it runs `TINYTITAN_BENCH_MODEL` (or the shipped default). Each
install gets its own log, named after the install as well as its width, so two
4-bit installs on one command line do not truncate each other's capture. An arm
whose server never answered `/health` is reported and skipped rather than
posted at; an arm whose log carries fewer or more footers than it sent requests
is `NOT MEASURED`, and the sweep exits 1.

    python3 benchmark/tinytitan_longgen.py models/qwen3.5_4B_4Bit
"""

import http.client
import json
import os
import subprocess
import sys
import time

from tinytitan_profile import (
    bench_model,
    benchmark_log_path,
    resolve_api_model,
    server_command,
    server_environment,
    wait_for_health,
)

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BIN = os.path.join(BASE, ".build", "release", "TinyTitanServer")
PORT = 8114
MAX_TOKENS = 512
WARMUP_RUNS = 1
MEASURED_RUNS = 3
REQUESTS_PER_MODEL = WARMUP_RUNS + MEASURED_RUNS
SERVER_LOAD_TIMEOUT = 2400
REQUEST_TIMEOUT_SECONDS = 1800
FLUSH_SETTLE_SECONDS = 0.5
REQUEST_SETTLE_SECONDS = 0.5
WIDTHS = ("8bit", "6bit", "4bit")
USAGE = "usage: tinytitan_longgen.py [model...]"
PROMPT = (
    "Write a Python function that computes the Levenshtein distance between "
    "two strings using dynamic programming with O(min(m,n)) space, including "
    "a detailed docstring and comments. Then add a main block that tests it "
    "on several pairs of strings and prints the results. Then write a "
    "second function that uses it to find the closest match to a target "
    "string in a list of candidates, and demonstrate it."
)


class ConfigError(ValueError):
    """A run configuration the driver refuses rather than guesses at."""


def parse_args(argv):
    """The installs to sweep; no flags, because the only knob is which install."""
    models = []
    for arg in argv:
        if arg.startswith("-"):
            raise ConfigError(f"{arg} is not an option this driver takes")
        if not arg.strip():
            raise ConfigError("an empty model path is not an install")
        models.append(arg)
    return models or [bench_model()]


def model_label(model):
    """The width the install's own name carries, or its basename.

    A name carrying no width keeps its own name: the old ternary fell through to
    `4bit`, which is a quantisation claim about a run that did not make one.
    """
    lowered = model.lower().replace("-", "")
    label = next((width for width in WIDTHS if width in lowered), None)
    if label is None:
        label = os.path.basename(os.path.normpath(model)) or "model"
    return label


def tag_for(model):
    """The row's name: the width, and which install it was measured on.

    The width alone is not unique, and a row that cannot say which install it
    came from is not comparable with the row above it. An install whose name
    carries no width is already named by that name, so it is not said twice.
    """
    base = os.path.basename(os.path.normpath(model)) or "model"
    label = model_label(model)
    return base if label == base else f"{label}-{base}"


def log_name_for(model):
    return f"longgen_{tag_for(model)}.log"


def decode_rates(lines):
    """Every `decode_tok_s=` footer, in log order — the first one is the warm-up."""
    rates = []
    for line in lines:
        if "TinyTitan generation" in line and "decode_tok_s=" in line:
            rates.append(float(line.split("decode_tok_s=")[1].split()[0]))
    return rates


def completion_counts(lines):
    counts = []
    for line in lines:
        if "completed in" in line and "completion=" in line:
            counts.append(int(line.split("completion=")[1].split()[0]))
    return counts


def build_payload(api_model):
    return json.dumps(
        {
            "model": api_model,
            "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 0.0,
            "max_completion_tokens": MAX_TOKENS,
            "stream": True,
        }
    ).encode()


def arm_report(tag, rates, cts):
    """(lines, status): the mean is over the measured runs, never the warm-up."""
    if not rates:
        return (
            [
                f"NOT MEASURED: {tag} -- no log line carried decode_tok_s=, "
                f"though {REQUESTS_PER_MODEL} requests were sent"
            ],
            1,
        )
    if len(rates) != REQUESTS_PER_MODEL or len(cts) != REQUESTS_PER_MODEL:
        return (
            [
                f"NOT MEASURED: {tag} -- {len(rates)} decode footers and {len(cts)} "
                f"completion counts for {REQUESTS_PER_MODEL} requests. The first footer "
                f"is the warm-up and the mean is over the rest, so a footer that is "
                f"missing or extra moves both the mean and which run the warm-up was"
            ],
            1,
        )
    warm, measured = rates[0], rates[WARMUP_RUNS:]
    mean = sum(measured) / len(measured)
    row = (
        f"{tag}: mean={mean:.2f} tok/s over {MEASURED_RUNS} measured runs "
        f"(warm-up {warm:.2f}) rates=["
        + ", ".join(f"{rate:.2f}" for rate in measured)
        + f"] ct={cts[WARMUP_RUNS:]}"
    )
    return [row], 0


def request(port, payload, *, timeout=REQUEST_TIMEOUT_SECONDS):
    """One streamed POST, read to the end.

    The drain is the measurement: the engine prints its footer as the stream
    closes, so a request abandoned halfway leaves no `decode_tok_s=` line.
    """
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
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


def run_quant(model, tag, port=PORT):
    """One install against a fresh server: its two scraped channels, or None if the
    server never served a measurement. A server that does not answer `/health` is
    never posted at -- the old inline wait timed out *into* the run."""
    env = server_environment()
    env["TINYTITAN_RUNNER_STATS"] = "1"
    log_path = benchmark_log_path(log_name_for(model))
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            server_command(BIN, port, model=model),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            if not wait_for_health(proc, port, timeout=SERVER_LOAD_TIMEOUT):
                print(
                    f"{tag}: the server never answered /health within "
                    f"{SERVER_LOAD_TIMEOUT}s -- no request was sent",
                    flush=True,
                )
                return None
            api_model = resolve_api_model(port)
            print(f"{tag} model id: {api_model}", flush=True)
            payload = build_payload(api_model)
            for _ in range(REQUESTS_PER_MODEL):
                request(port, payload)
                time.sleep(REQUEST_SETTLE_SECONDS)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    time.sleep(FLUSH_SETTLE_SECONDS)
    with open(log_path, encoding="utf-8") as handle:
        lines = handle.readlines()
    return decode_rates(lines), completion_counts(lines)


def main() -> int:
    try:
        models = parse_args(sys.argv[1:])
    except ConfigError as error:
        print(f"REFUSED: {error} -- {USAGE}", file=sys.stderr)
        return 2
    print(
        f"long generation: {len(models)} install(s), {REQUESTS_PER_MODEL} requests each "
        f"({WARMUP_RUNS} warm-up + {MEASURED_RUNS} measured), {MAX_TOKENS} tokens, port {PORT}"
    )
    worst = 0
    for model in models:
        tag = tag_for(model)
        result = run_quant(model, tag)
        if result is None:
            printed, status = [f"ARM FAILED: {tag} -- no measurement was taken"], 1
        else:
            printed, status = arm_report(tag, *result)
        for line in printed:
            print(line, flush=True)
        worst = max(worst, status)
    return worst


if __name__ == "__main__":
    sys.exit(main())
