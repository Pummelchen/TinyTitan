#!/usr/bin/env python3
"""A/B: expert-cache slot count (the 8 GB RAM lever). 32 vs 128 slots/layer
(128 x 40 x ~1.55 MiB ~= 7.9 GB), 512-token greedy essay, server footers.

`io_ms` is the decisive counter: a higher hit rate must shrink the pread wall on
the critical path (the pread window is GPU-idle except shared+phase-1). Each arm
boots its own server and is reaped before the next is launched, so the machine
never holds two.

An arm is a capture of three channels — the generation rate, the runner's `io_ms`
split, and the total GPU time — or a failure. This driver used to print
`--- slots=32 pin=None boot_s=... ---` over whatever those three lists held, with
no line count and no refusal, so a server that answered `/health` and reported no
counter at all produced one header line and exited 0. The header and the refusal
are now `channel_verdict()`'s, shared with the length sweep and the read-ahead A/B,
so the three drivers cannot disagree about what an empty arm means. A server that
dies before `/health` is one failed arm and the sweep carries on; it used to be
`sys.exit(1)` inside the per-arm function, which ended the run at the first dead
arm and reported neither.

Nothing runs at import: the slot list, the pin mode and the token count are read
by `main()`, so an unparseable `TINYTITAN_AB_TOKENS` is a refusal with exit 2
rather than a `ValueError` during import.

    python3 benchmark/tinytitan_slots_ab.py [pin|nopin] [slots...]
    cd benchmark && python3 -m unittest test_tinytitan_slots_ab
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
PORT = 8115
PROMPT = "Write a detailed essay about the history of computing."
DEFAULT_SLOTS = (32, 128)
# The pin A/B is the tighter lever, so it sweeps the smaller step by default.
DEFAULT_PIN_SLOTS = (32, 64)
TOKENS_ENV = "TINYTITAN_AB_TOKENS"
DEFAULT_MAX_TOKENS = 512
FLUSH_SETTLE_SECONDS = 0.3

CHANNELS = (
    ("gen", "decode_tok_s="),
    ("runner", "io_ms="),
    ("kernel total", "TinyTitan kernel total_gpu_ms="),
)

USAGE = "usage: tinytitan_slots_ab.py [pin|nopin] [slots...]"


class ConfigError(ValueError):
    """A command line or environment this driver will not guess at."""


def parse_args(argv):
    """`(slot list, pin)`, where `pin` is `True`, `False`, or `None` to leave the
    server's own default alone."""
    rest = list(argv)
    pin = None
    if rest and rest[0] in ("pin", "nopin"):
        pin = rest[0] == "pin"
        rest = rest[1:]
    if not rest:
        return list(DEFAULT_PIN_SLOTS if pin is not None else DEFAULT_SLOTS), pin
    try:
        slots = [int(value) for value in rest]
    except ValueError as error:
        raise ConfigError(f"slot counts must be integers, not {' '.join(rest)}") from error
    if any(count < 1 for count in slots):
        raise ConfigError(f"slot counts must be positive, not {slots}")
    return slots, pin


def max_tokens(env=None):
    """The request length, read when `main()` runs rather than at import."""
    source = os.environ if env is None else env
    raw = source.get(TOKENS_ENV) or str(DEFAULT_MAX_TOKENS)
    try:
        value = int(raw)
    except ValueError as error:
        raise ConfigError(f"{TOKENS_ENV} must be an integer, not {raw!r}") from error
    if value < 1:
        raise ConfigError(f"{TOKENS_ENV} must be positive, not {value}")
    return value


def tag_for(slots, pin):
    if pin is None:
        return f"slots={slots}"
    return f"slots={slots}-{'pin' if pin else 'nopin'}"


def arm_environment(slots, pin, base=None):
    """The environment for one arm: both stat channels on, this slot count, and the
    pin control only when the A/B is about pinning."""
    env = server_environment(base)
    env["TINYTITAN_RUNNER_STATS"] = "1"
    env["TINYTITAN_KERNEL_STATS"] = "1"
    env["TINYTITAN_EXPERT_CACHE_SLOTS"] = str(slots)
    if pin is True:
        env.pop("TINYTITAN_NO_PIN", None)
    elif pin is False:
        env["TINYTITAN_NO_PIN"] = "1"
    return env


def capture(lines):
    """(gen, runner, kernel total) classified out of the server log."""
    gen, runner, total = [], [], []
    for line in lines:
        if "TinyTitan generation" in line and "decode_tok_s=" in line:
            gen.append(line.strip())
        if "TinyTitan runner" in line and "io_ms=" in line:
            runner.append(line.strip())
        if "TinyTitan kernel total_gpu_ms=" in line:
            total.append(line.strip())
    return gen, runner, total


def verdict(arms):
    """(lines, exit status) for arms given as `(tag, capture-or-None)`."""
    return channel_verdict(arms, CHANNELS)


def run(slots, pin, tokens, port=PORT):
    """One slot count against a fresh server: its capture, or None if it never
    loaded. The boot time is printed as it is measured, because the slot count is
    also a memory setting and a slow load is the finding, not noise."""
    tag = tag_for(slots, pin)
    log_path = benchmark_log_path(f"tinytitan_slots_{tag}.log")
    started = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            server_command(BIN, port, model=bench_model()),
            env=arm_environment(slots, pin),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        if not wait_for_health(proc, port):
            print(f"{tag}: the server exited before /health answered", flush=True)
            return None
        print(f"{tag} boot_s={time.time() - started:.1f}", flush=True)
        try:
            request_twice(PROMPT, tokens, port)
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
    try:
        slots, pin = parse_args(sys.argv[1:])
        tokens = max_tokens()
    except ConfigError as error:
        print(f"REFUSED: {error} -- {USAGE}", file=sys.stderr)
        return 2

    print(
        f"slot A/B: {bench_model()}, {len(slots)} arm(s) "
        f"({', '.join(tag_for(count, pin) for count in slots)}), {tokens} tokens, "
        f"pin={pin}, port {PORT}"
    )
    arms = [(tag_for(count, pin), run(count, pin, tokens)) for count in slots]
    printed, status = verdict(arms)
    for line in printed:
        print(line)
    return status


if __name__ == "__main__":
    sys.exit(main())
