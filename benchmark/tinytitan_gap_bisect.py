#!/usr/bin/env python3
"""Bisect the unmeasured wall: a request-length sweep at 128 and 1024 tokens,
using authoritative server footers.

One pass, two arms, in the order `LEN_ARMS` names them — 128 then 1024 — each
against its own freshly started server that is reaped before the next one is
launched, so only one model process is ever alive. The control is the request
length and nothing else: both arms get the same environment, the same prompt and
the same two-request pattern, so the difference between the two pages is the
per-token work times the tokens.

This driver used to advertise "a length sweep and an rdadvise-off comparison"
and never ran the second half: it passed no extra environment and
`TINYTITAN_RDADVISE_POLICY` appeared nowhere in it. The policy A/B is
`tinytitan_rdadvise_ab.py`'s job and this page no longer claims it.

Each arm reports the server's own footers in four channels — the generation rate,
the per-token runner split, the per-kernel role table (whose `per_token_ms=`
column is what the sweep reads), and the total GPU time. A channel that captured
no line is not a zero measurement, it is an instrument that did not run: the arm
says which one and the exit status is 1. Every header carries its line count,
because two requests were sent and an arm with one footer is a partial run. A
server that never answers `/health` is reported as a failed arm and the sweep
carries on to the next one, rather than ending the run on the first.

Nothing runs at import: `main()` is the only code that opens a log or starts a
server, and it is behind the `__main__` guard.

    python3 benchmark/tinytitan_gap_bisect.py         # no arguments; it takes none
    cd benchmark && python3 -m unittest test_tinytitan_gap_bisect
"""

import os
import subprocess
import sys
import time

from tinytitan_profile import (
    bench_model,
    benchmark_log_path,
    channel_verdict,
    request_twice,
    server_command,
    server_environment,
    wait_for_health,
)

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BIN = os.path.join(BASE, ".build", "release", "TinyTitanServer")
PORT = 8112
PROMPT = "Write a detailed essay about the history of computing."
# (tokens, tag) — the two lengths the sweep spans, and the name each arm prints.
LEN_ARMS = ((128, "len128"), (1024, "len1024"))
# The child writes its footer to a pipe it owns; the reaped process flushes it, and
# this is the settle time the driver has always allowed before reading the log back.
FLUSH_SETTLE_SECONDS = 0.3

CHANNELS = (
    ("gen", "decode_tok_s="),
    ("runner", "cb1_ms="),
    ("kernel roles", "TinyTitan kernel role="),
    ("kernel total", "TinyTitan kernel total_gpu_ms="),
)


def arm_environment(tag: str, base=None) -> dict[str, str]:
    """The environment for one arm: both stat channels on, and no other control.

    `base` exists for the tests; production inherits `os.environ`. The two arms
    must differ only in length, so nothing here varies with `tag` except the name
    it validates, and the read-ahead policy is not this driver's to set.
    """
    known = {name for _, name in LEN_ARMS}
    if tag not in known:
        raise ValueError(f"unknown arm {tag!r}, expected one of {', '.join(sorted(known))}")
    env = server_environment(base)
    env["TINYTITAN_RUNNER_STATS"] = "1"
    env["TINYTITAN_KERNEL_STATS"] = "1"
    return env


def capture(lines):
    """(gen, runner, roles, total) classified out of the server log."""
    gen, runner, roles, total = [], [], [], []
    for line in lines:
        if "TinyTitan generation" in line and "decode_tok_s=" in line:
            gen.append(line.strip())
        if "TinyTitan runner" in line and "cb1_ms=" in line:
            runner.append(line.strip())
        if "TinyTitan kernel role=" in line:
            roles.append(line.strip())
        if "TinyTitan kernel total_gpu_ms=" in line:
            total.append(line.strip())
    return gen, runner, roles, total


def verdict(arms):
    """(lines, exit status) for arms given as `(tag, capture-or-None)`.

    An arm whose capture is None never produced a log worth reading, and an arm
    whose capture has an empty channel in it produced a server that did not report
    what that channel exists to measure. Neither is a measurement, so neither may
    exit 0. The page and status are `channel_verdict()`'s, shared with the
    read-ahead A/B so the two drivers cannot drift apart on what an empty arm
    means.
    """
    return channel_verdict(arms, CHANNELS)


def run(max_tokens: int, tag: str, port: int = PORT):
    """One length against a fresh server: its capture, or None if it never loaded."""
    log_path = benchmark_log_path(f"tinytitan_gap_{tag}.log")
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            server_command(BIN, port, model=bench_model()),
            env=arm_environment(tag),
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
            f"{', '.join(str(tokens) for tokens, _ in LEN_ARMS)} tokens",
            file=sys.stderr,
        )
        return 2

    lengths = ", then ".join(str(tokens) for tokens, _ in LEN_ARMS)
    print(f"length sweep: {bench_model()}, {lengths} tokens, one fresh server per arm")
    print("")
    printed, status = verdict([(tag, run(tokens, tag)) for tokens, tag in LEN_ARMS])
    for line in printed:
        print(line)
    return status


if __name__ == "__main__":
    sys.exit(main())
