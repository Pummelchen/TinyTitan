#!/usr/bin/env python3
"""Peak-throughput probe: decode rate as a function of expert-cache locality.
Prompts rank from diverse routing (code) to maximally repetitive (digit cycles),
against one server that holds the whole ladder, each prompt sent twice — warm-up,
then the measured run — and read off the server's own footers.

Attribution is the whole job here. The log names no prompt: a footer is tied to
the request that produced it by *position*, so the driver requires exactly
`REQUESTS_PER_PROMPT` footers per prompt before it prints a single rate. That
count used to be ignored and the flat list sliced two at a time, which means one
footer missing anywhere in the run relabels everything after it — the row that
reads `essay: measured=44.00` was then the *digits* warm-up, and the last prompt
was accused of a failure it did not have. A run whose counts do not match is
reported `NOT MEASURED` with both numbers and exits 1, and nothing in it is
printed as a prompt's rate.

Nothing runs at import: `MAXTPUT_PROMPTS` is read by `main()`, so a selector that
names no prompt is a refusal with exit 2 rather than a `SystemExit` during import,
and a typo beside a real name is refused instead of silently narrowing the ladder.

Usage:
    python3 benchmark/tinytitan_maxthroughput.py [--mtp <draft-dir>] \\
        [--engine cpu|gpu] [model...]
    MAXTPUT_PROMPTS=essay python3 benchmark/tinytitan_maxthroughput.py
    cd benchmark && python3 -m unittest test_tinytitan_maxthroughput
"""

import os
import subprocess
import sys
import time

from tinytitan_profile import (
    bench_model,
    benchmark_log_path,
    request_twice,
    server_command,
    server_environment,
    wait_for_health,
)

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BIN = os.path.join(BASE, ".build", "release", "TinyTitanServer")
PORT = 8117
MAX_TOKENS = 512
REQUESTS_PER_PROMPT = 2
PROMPT_ENV = "MAXTPUT_PROMPTS"
# A 125B install streams off SSD for minutes before it serves. The 120 s budget
# this file used was sized for a model that is no longer the default and turned a
# slow load into "server failed to start".
SERVER_LOAD_TIMEOUT = 2400
# The child writes its footer to a pipe it owns; the reaped process flushes it.
FLUSH_SETTLE_SECONDS = 0.5
REQUEST_SETTLE_SECONDS = 0.3

PROMPTS = (
    (
        "code",
        "Write a Python function that computes the Levenshtein distance between two strings "
        "with a detailed docstring, then a second function using it to find the closest match "
        "in a list, with a demo main.",
    ),
    ("essay", "Write a detailed essay about the history of computing."),
    ("count", "Count from 1 to 1000, writing only the numbers separated by single spaces."),
    (
        "digits",
        "Write the digits 1,2,3,4,5,6,7,8,9,0 over and over in sequence, separated by commas, "
        "without stopping.",
    ),
)

WIDTHS = ("8bit", "6bit", "4bit")


class ConfigError(ValueError):
    """A command line or environment this driver will not guess at."""


def select_prompts(value):
    """The ladder, narrowed by `MAXTPUT_PROMPTS` and still in ladder order.

    The order is the measurement — diverse routing first, maximally repetitive
    last — so a selector written `digits,essay` narrows the set but does not
    reorder it. A name that matches no prompt is a refusal, including when the
    rest of the selector is valid: `essay,bogus` used to run one prompt under a
    page that read as a comparison.
    """
    if not value or not value.strip():
        return list(PROMPTS)
    wanted = [name.strip() for name in value.split(",") if name.strip()]
    known = {name for name, _ in PROMPTS}
    unknown = [name for name in wanted if name not in known]
    if unknown:
        raise ConfigError(
            f"no prompt named {', '.join(unknown)}; the ladder is "
            f"{', '.join(name for name, _ in PROMPTS)}"
        )
    return [prompt for prompt in PROMPTS if prompt[0] in set(wanted)]


def model_label(model, engine):
    """The row's name: the width the install advertises, plus the engine.

    The width is read off the directory because that is all the driver knows about
    the install, and a name carrying no width keeps its own basename — the old
    fallback printed `4bit` for any install whose name did not say, which is a
    quantisation claim about a run that did not make one.
    """
    lowered = model.lower().replace("-", "")
    label = next((width for width in WIDTHS if width in lowered), None)
    if label is None:
        label = os.path.basename(os.path.normpath(model)) or "model"
    return f"{label}-cpu" if engine == "cpu" else label


def parse_args(argv):
    """`(models, draft-head-dir, engine)`; `--engine cpu` is the only way to run
    the dense Qwen 3.5 models, and the label carries it so a CPU row cannot be
    mistaken for a GPU one."""
    models, mtp, engine = [], None, "gpu"
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg in ("--mtp", "--engine"):
            if index + 1 >= len(argv):
                raise ConfigError(f"{arg} needs its value")
            value = argv[index + 1]
            if arg == "--mtp":
                mtp = value
            else:
                if value not in ("cpu", "gpu"):
                    raise ConfigError(f"--engine must be cpu or gpu, not {value}")
                engine = value
            index += 2
        elif arg.startswith("-"):
            raise ConfigError(
                f"unknown flag {arg}; the flags are --mtp <draft-dir> and --engine cpu|gpu"
            )
        else:
            models.append(arg)
            index += 1
    return models, mtp, engine


def decode_rates(lines):
    """Every `decode_tok_s=` footer in the server log, in the order printed."""
    rates = []
    for line in lines:
        if "decode_tok_s=" in line and ("TinyTitan generation" in line or "TinyTitan mtp " in line):
            rates.append(float(line.split("decode_tok_s=")[1].split()[0]))
    return rates


def completion_counts(lines):
    """The token counts of the completed requests — the other half of the pair."""
    counts = []
    for line in lines:
        if "completed in" in line and "completion=" in line:
            counts.append(int(line.split("completion=")[1].split()[0]))
    return counts


def arm_report(label, prompts, rates, cts):
    """(lines, exit status) for one arm's scrape, or a refusal to attribute it.

    See the module docstring: positions are the only thing tying a footer to a
    prompt, so the counts have to match before any row may carry a rate.
    """
    expected = REQUESTS_PER_PROMPT * len(prompts)
    if not rates:
        return [
            f"NOT MEASURED: {label} -- no log line carried decode_tok_s=, "
            f"though {expected} requests were sent"
        ], 1
    if len(rates) != expected or len(cts) != expected:
        return [
            f"NOT MEASURED: {label} -- {len(rates)} decode footers and {len(cts)} completion "
            f"counts for {expected} requests. Rates are tied to prompts by position, so a "
            f"footer that is missing or extra shifts every row after it; no rate here can be "
            f"printed as a prompt's measurement."
        ], 1
    lines = []
    for index, (name, _) in enumerate(prompts):
        window = rates[index * REQUESTS_PER_PROMPT : (index + 1) * REQUESTS_PER_PROMPT]
        counts = cts[index * REQUESTS_PER_PROMPT : (index + 1) * REQUESTS_PER_PROMPT]
        lines.append(
            f"{label} {name}: measured={window[-1]:.2f} (warmup {window[0]:.2f}) ct={counts}"
        )
    return lines, 0


def run_quant(model, label, prompts, mtp_model=None, engine="gpu"):
    """One server, the whole ladder against it, and that arm's (lines, status)."""
    log_path = benchmark_log_path(f"maxtput_{label}.log")
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            server_command(BIN, PORT, model=model, mtp_model=mtp_model, engine=engine),
            env=server_environment(),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        if not wait_for_health(proc, PORT, timeout=SERVER_LOAD_TIMEOUT):
            return [f"ARM FAILED: {label} -- the server exited before /health answered"], 1
        try:
            for _, prompt in prompts:
                request_twice(prompt, MAX_TOKENS, PORT)
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
    return arm_report(label, prompts, decode_rates(lines), completion_counts(lines))


def main() -> int:
    try:
        models, mtp, engine = parse_args(sys.argv[1:])
        prompts = select_prompts(os.environ.get(PROMPT_ENV))
    except ConfigError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2

    chosen = models or [bench_model()]
    names = ", ".join(name for name, _ in prompts)
    print(
        f"peak-throughput probe: {len(chosen)} model(s) x {len(prompts)} prompt(s) ({names}) "
        f"x {REQUESTS_PER_PROMPT} requests at {MAX_TOKENS} tokens, engine={engine}, "
        f"port {PORT}"
    )
    if mtp:
        print(
            f"--mtp {mtp}: every model runs twice, the first arm *without* the draft head so "
            f"the delta is against a non-speculative baseline"
        )
    status = 0
    for model in chosen:
        print(f"--- {model} ---", flush=True)
        label = model_label(model, engine)
        arms = [(label, None)]
        if mtp:
            arms.append((f"{label}-mtp", mtp))
        for arm_label, draft in arms:
            lines, arm_status = run_quant(model, arm_label, prompts, mtp_model=draft, engine=engine)
            for line in lines:
                print(line, flush=True)
            status = status or arm_status
    return status


if __name__ == "__main__":
    sys.exit(main())
