#!/usr/bin/env python3
"""A/B: expert read-ahead advice at the engine default vs `off`.

One pass, two arms, in the order `MODES` names them — `default` then `off` — each
against its own freshly started server that is reaped before the next one is
launched, so only one model process is ever alive. (An earlier version of this
docstring called that "interleaved", which it never was: one arm follows the
other and nothing alternates them. A single pass in a fixed order is the design
here, not a randomized one.)

Each arm sends the same 512-token greedy request twice and reports the server's
own footers — the generation rate, the per-token runner split, and the total GPU
time for the request. Those three are what the read-ahead policy is expected to
move. The engine's `TinyTitan kernel busy_ms=… span_ms=…` line is the one that
names the overlap directly and **no section reads it**, so the gpu column here is
a share of decode time, never the overlap ratio.

A channel that captured no line is not a zero measurement, it is an instrument
that did not run: the arm says which one and the exit status is 1. Every header
carries its line count, because two requests were sent and an arm with one footer
is a partial run.

The `default` arm clears `TINYTITAN_RDADVISE_POLICY` from the environment rather
than inheriting it. The runtime reads that variable *ahead of* the model's shipped
policy (`ServerModelSession+Loading.swift:198`), so an operator who had exported
one would have got `override vs off` on a page that said `default vs off`.

    python3 benchmark/tinytitan_rdadvise_ab.py        # no arguments; it takes none
    cd benchmark && python3 -m unittest test_tinytitan_rdadvise_ab
"""

import os
import subprocess
import sys
import time

from tinytitan_profile import (
    DEFAULT_MODEL_PATH,
    benchmark_log_path,
    request_twice,
    server_command,
    server_environment,
    wait_for_health,
)

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BIN = os.path.join(BASE, ".build", "release", "TinyTitanServer")
PORT = 8113
PROMPT = "Write a detailed essay about the history of computing."
MAX_TOKENS = 512
MODES = ("default", "off")
POLICY_ENV = "TINYTITAN_RDADVISE_POLICY"
# The child writes its footer to a pipe it owns; the reaped process flushes it, and
# this is the settle time the driver has always allowed before reading the log back.
FLUSH_SETTLE_SECONDS = 0.3

CHANNELS = (
    ("gen", "decode_tok_s="),
    ("runner", "cb1_ms="),
    ("gpu", "TinyTitan kernel total_gpu_ms="),
)


def bench_model() -> str:
    """The install this A/B runs against.

    `TINYTITAN_BENCH_MODEL` is how the other sweeps name their model; hardcoding
    the default here meant an operator pointing the benchmark tree at their own
    checkpoint got the shipped one under a label that named nothing.
    """
    return os.environ.get("TINYTITAN_BENCH_MODEL", str(DEFAULT_MODEL_PATH))


def arm_environment(mode: str, base=None) -> dict[str, str]:
    """The environment for one arm: both stat channels on, and the policy decided.

    `base` exists for the tests; production inherits `os.environ`, which is
    exactly why the `default` arm has to clear `POLICY_ENV` rather than leave it.
    """
    if mode not in MODES:
        raise ValueError(f"unknown arm {mode!r}, expected one of {', '.join(MODES)}")
    env = server_environment(base)
    env["TINYTITAN_RUNNER_STATS"] = "1"
    env["TINYTITAN_KERNEL_STATS"] = "1"
    if mode == "off":
        env[POLICY_ENV] = "off"
    else:
        env.pop(POLICY_ENV, None)
    return env


def capture(lines):
    """(gen, runner, gpu) classified out of the server log."""
    gen, runner, gpu = [], [], []
    for line in lines:
        if "TinyTitan generation" in line and "decode_tok_s=" in line:
            gen.append(line.strip())
        if "TinyTitan runner" in line and "cb1_ms=" in line:
            runner.append(line.strip())
        if "TinyTitan kernel total_gpu_ms=" in line:
            gpu.append(line.strip())
    return gen, runner, gpu


def _count(noun: str, rows) -> str:
    return f"{noun} {len(rows)} {'line' if len(rows) == 1 else 'lines'}"


def verdict(arms):
    """(lines, exit status) for arms given as `(mode, capture-or-None)`.

    An arm whose capture is None never produced a log worth reading, and an arm
    whose capture is a triple with an empty list in it produced a server that did
    not report the channel the arm exists to measure. Neither is a measurement, so
    neither may exit 0.
    """
    lines, status = [], 0
    for mode, sections in arms:
        if sections is None:
            lines.append(f"ARM FAILED: {mode} -- the server never answered /health")
            status = 1
            continue
        columns = list(zip(CHANNELS, sections, strict=True))
        header = ", ".join(_count(name, part) for (name, _), part in columns)
        lines.append(f"--- {mode} ({header}) ---")
        for _, part in columns:
            lines.extend(part)
        for (_name, wanted), part in columns:
            if not part:
                lines.append(f"NOT MEASURED: {mode} -- no log line contained {wanted!r}")
                status = 1
    return lines, status


def run(mode: str, port: int = PORT, max_tokens: int = MAX_TOKENS):
    """One arm against a fresh server: its capture, or None if it never loaded."""
    log_path = benchmark_log_path(f"tinytitan_rd_{mode}.log")
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            server_command(BIN, port, model=bench_model()),
            env=arm_environment(mode),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        if not wait_for_health(proc, port):
            return None
        try:
            request_twice(PROMPT, max_tokens, port)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    time.sleep(FLUSH_SETTLE_SECONDS)
    with open(log_path, encoding="utf-8") as handle:
        return capture(handle.readlines())


def main() -> int:
    if len(sys.argv) > 1:
        print(
            f"usage: {os.path.basename(sys.argv[0])} takes no arguments -- the arms are "
            f"{' and '.join(MODES)}, at {MAX_TOKENS} tokens",
            file=sys.stderr,
        )
        return 2

    print(f"rdadvise A/B: {bench_model()}, {MAX_TOKENS} tokens, one fresh server per arm")
    print("")
    printed, status = verdict([(mode, run(mode)) for mode in MODES])
    for line in printed:
        print(line)
    return status


if __name__ == "__main__":
    sys.exit(main())
