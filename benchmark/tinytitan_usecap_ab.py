#!/usr/bin/env python3
"""Paired A/B: a saturating expert use-count cap against the shipped LFU.

The engine prices an expert-cache slot's eviction by how many times its expert has
ever been planned, so a hot expert decode stopped using defends its slot with
everything it did since the model loaded. A cap collapses the top of that ranking
instead: everyone at or above it prices the same, the tie falls to last use, and
the *recently* unused expert goes first. `TINYTITAN_EXPERT_USE_CAP` selects the
cap and is read once at streamer construction, so every arm below is a pure
environment toggle over one binary and one install — no rebuild between arms.

§5y measured it offline on the captured qwen38 route trace: cap 3 at 2.0–7.9 %
fewer placements per token, cap 5 at 2.6 %. Placements are not clock, and §5w
showed a 45 % miss change buying 12.2 % of wall, so this driver answers the one
question a replay cannot: does it move tok/s on this machine.

**Why paired and interleaved.** §5w's four arms of one identical configuration
spread 12.2 % inside a single session, which is larger than the effect being
hunted. Blocked arms would fold that drift into the delta, so each round runs
`default, cap3, cap5` back to back against three fresh servers — one alive at a
time — and the number quoted is the *within-round* delta against the control that
rode in that round. One round cannot carry a direction, so fewer than two complete
pairs fails the run and every per-round delta is printed, not only the mean.

The counters travel beside the clock: a tok/s win whose `expert_misses` and
`expert_read_mib` did not move is machine drift wearing a result. And residency is
a cache, not semantics, so every arm must answer with the same text — a differing
hash outranks every speed column on this page.

Two greedy requests per arm. The warm footer of each channel is the measurement
(AUD-232: averaging the cold one reports prefill as decode), and a counter that
never appeared is absent, not zero (AUD-276, AUD-283), so its round is dropped
rather than pooled at 0.

    TINYTITAN_BENCH_MODEL=models/qwen3.8-flash-next_125B_A6B_4Bit \\
      python3 benchmark/tinytitan_usecap_ab.py      # no arguments; it takes none
    cd benchmark && python3 -m unittest test_tinytitan_usecap_ab
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import statistics
import subprocess
import sys
import time

from tinytitan_profile import (
    bench_model,
    benchmark_log_path,
    channel_verdict,
    pgrep_answer,
    resolve_api_model,
    server_command,
    server_environment,
    wait_for_health,
)

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BIN = os.path.join(BASE, ".build", "release", "TinyTitanServer")
PORT = 8116
PROMPT = "Write a detailed essay about the history of computing."
MAX_TOKENS = 512
# Pinned, because the arm is a cache policy and a cache policy only means
# something at one cache size: at another budget the slots change, and the
# paired delta becomes a comparison between machines rather than between plans.
RAM_BUDGET = "12G"
ROUNDS = 3
BASELINE = "default"
ARMS = ("default", "cap3", "cap5")
CAP_ENV = "TINYTITAN_EXPERT_USE_CAP"
CAP_VALUE = {"cap3": "3", "cap5": "5"}
MODEL_PROCESSES = "TinyTitanServer|TinyTitanCLI"
MIN_PAIRS = 2
# The child writes its footer to a pipe it owns; the reaped process flushes it,
# and this is the settle time the sibling A/B drivers have always allowed.
FLUSH_SETTLE_SECONDS = 0.3
EMPTY_SHA = hashlib.sha256(b"").hexdigest()[:16]
CHANNELS = (
    ("gen", "decode_tok_s="),
    ("runner", "cb1_ms="),
    ("kernel", "total_gpu_ms="),
)


def arm_environment(arm: str, base=None) -> dict[str, str]:
    """The environment for one arm: both stat channels on, and the cap decided.

    `base` exists for the tests; production inherits `os.environ`, which is
    exactly why the control arm has to clear `CAP_ENV` rather than leave it — the
    runtime reads that variable at streamer construction, so an operator who had
    exported a cap would get `cap9 vs cap3` on a page that said `default vs cap3`.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}, expected one of {', '.join(ARMS)}")
    env = server_environment(base)
    env["TINYTITAN_RUNNER_STATS"] = "1"
    env["TINYTITAN_KERNEL_STATS"] = "1"
    if arm == BASELINE:
        env.pop(CAP_ENV, None)
    else:
        env[CAP_ENV] = CAP_VALUE[arm]
    return env


def server_command_for(binary, port, model=None):
    """The launcher invocation for one arm, at the one budget every arm shares."""
    return server_command(
        binary,
        port,
        model=bench_model() if model is None else model,
        ram_budget=RAM_BUDGET,
    )


def capture(lines):
    """(gen, runner, kernel) classified out of the server log."""
    gen, runner, kernel = [], [], []
    for line in lines:
        if "TinyTitan generation" in line and "decode_tok_s=" in line:
            gen.append(line.strip())
        if "TinyTitan runner" in line and "cb1_ms=" in line:
            runner.append(line.strip())
        if "TinyTitan kernel total_gpu_ms=" in line:
            kernel.append(line.strip())
    return gen, runner, kernel


def _field(line, name):
    """The number a footer token carries, or None when that footer never logged it."""
    marker = f"{name}="
    for token in line.split():
        if token.startswith(marker):
            try:
                return float(token[len(marker) :])
            except ValueError:
                return None
    return None


def figures(captured):
    """The arm's reading: the *warm* footer of each channel, absent kept absent.

    Two requests were sent, so the last generation line is the warm one; the
    counters are cumulative per request and the last runner line is the one that
    belongs to the same request as that footer.
    """
    gen, runner, kernel = captured
    warm_gen = gen[-1] if gen else ""
    warm_runner = runner[-1] if runner else ""
    warm_kernel = kernel[-1] if kernel else ""
    return {
        "decode_tok_s": _field(warm_gen, "decode_tok_s"),
        "io_ms": _field(warm_runner, "io_ms"),
        "wait_ms": _field(warm_runner, "wait_ms"),
        "expert_misses": _field(warm_runner, "expert_misses"),
        "expert_reloads": _field(warm_runner, "expert_reloads"),
        "expert_read_mib": _field(warm_runner, "expert_read_mib"),
        "total_gpu_ms": _field(warm_kernel, "total_gpu_ms"),
    }


def _delta(control, arm, key):
    """`arm - control` for one counter, or None when either failed to log it."""
    base, got = control.get(key), arm.get(key)
    if base is None or got is None:
        return None
    return got - base


def pair(rounds, arm, baseline=BASELINE):
    """Within-round rows for one arm: the clock delta and the counters beside it.

    A round is dropped — not zero-filled — when it has no control, no arm, or no
    rate for either: those are runs that did not measure this comparison, and
    pooling them would count them as if they had.
    """
    rows = []
    for index, readings in enumerate(rounds, start=1):
        control, measured = readings.get(baseline), readings.get(arm)
        if not control or not measured:
            continue
        base_rate, arm_rate = control.get("decode_tok_s"), measured.get("decode_tok_s")
        if not base_rate or not arm_rate:
            continue
        delta = arm_rate - base_rate
        rows.append(
            {
                "round": index,
                "delta_tok_s": delta,
                "pct": 100 * delta / base_rate,
                "delta_misses": _delta(control, measured, "expert_misses"),
                "delta_read_mib": _delta(control, measured, "expert_read_mib"),
            }
        )
    return rows


def _mean_delta(rows, key):
    """Mean of one counter's paired deltas, with the denominator that produced it."""
    values = [row[key] for row in rows if row[key] is not None]
    if not values:
        return "not logged"
    return f"{statistics.fmean(values):+.1f} over {len(values)}/{len(rows)} pairs"


def pairing_verdict(rounds, baseline=BASELINE):
    """(lines, status): each arm that ran needs `MIN_PAIRS` complete pairs.

    The per-round deltas stay on the page, because a round-to-round swing larger
    than the mean is the thing a reader needs in order to discount the mean.
    """
    lines, status = [], 0
    judged = [arm for arm in ARMS if arm != baseline and any(arm in r for r in rounds)]
    if not judged:
        return [f"NOT PAIRED: no capped arm ran beside {baseline}"], 1
    for arm in judged:
        rows = pair(rounds, arm, baseline)
        if len(rows) < MIN_PAIRS:
            lines.append(
                f"NOT PAIRED: {arm} -- {len(rows)} complete pair(s), a direction needs {MIN_PAIRS}"
            )
            status = 1
            continue
        deltas = ", ".join(
            f"r{row['round']} {row['delta_tok_s']:+.3f} ({row['pct']:+.1f}%)" for row in rows
        )
        spread = max(row["delta_tok_s"] for row in rows) - min(row["delta_tok_s"] for row in rows)
        lines.append(f"{arm} tok/s per round: {deltas}")
        lines.append(
            f"{arm} mean {statistics.fmean(row['delta_tok_s'] for row in rows):+.3f} "
            f"tok/s over {len(rows)} pairs, round-to-round spread {spread:.3f}"
        )
        lines.append(
            f"{arm} counters: misses {_mean_delta(rows, 'delta_misses')}, "
            f"read_mib {_mean_delta(rows, 'delta_read_mib')}"
        )
    return lines, status


def invariance(shas, baseline=BASELINE):
    """(lines, status) for one round's `{arm: response sha}`.

    Equality alone is not the claim: two arms that streamed nothing hash the
    same, so an empty control refuses the comparison instead of passing it.
    """
    missing = [arm for arm in ARMS if not shas.get(arm)]
    if missing:
        return [f"NOT COMPARED: {', '.join(missing)} streamed no response"], 1
    if shas[baseline] == EMPTY_SHA:
        return [f"NOT COMPARED: {baseline} streamed an empty response, so equal is empty"], 1
    odd = [arm for arm in ARMS if arm != baseline and shas[arm] != shas[baseline]]
    if odd:
        return [
            f"OUTPUT CHANGED: {', '.join(odd)} answered differently from {baseline} "
            f"-- a cache policy may not change the answer, and this outranks every "
            f"speed column above it"
        ], 1
    return [f"response invariance: equal across {', '.join(ARMS)} ({shas[baseline]})"], 0


def _extract_deltas(resp):
    """Content deltas only — the raw stream carries the per-request chatcmpl id
    and created timestamp, so hashing the bytes themselves is invalid."""
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


def generate_greedy_twice(arm, prompt, max_tokens, port):
    """Two greedy requests; returns both responses' text so the caller can hash it.

    `tinytitan_profile.request_twice` discards the content, because for those
    drivers the server log is the product. Here the answer is evidence too, and
    both requests are hashed together so a cold/warm difference is caught as well
    as an arm-to-arm one.
    """
    payload = json.dumps(
        {
            "model": resolve_api_model(port),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 0.0,
            "max_completion_tokens": max_tokens,
            "stream": True,
        }
    ).encode()
    texts = []
    for index in range(2):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1800)
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        texts.append("".join(_extract_deltas(conn.getresponse())))
        conn.close()
        print(f"{arm} request {index + 1} done", flush=True)
    return "".join(texts)


def run_arm(arm, round_, port=PORT, max_tokens=MAX_TOKENS):
    """One arm of one round against a fresh server: (capture, response sha) or (None, None)."""
    log_path = benchmark_log_path(f"tinytitan_usecap_r{round_}_{arm}.log")
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            server_command_for(BIN, port, model=bench_model()),
            env=arm_environment(arm),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        if not wait_for_health(proc, port):
            return None, None
        text = ""
        try:
            text = generate_greedy_twice(arm, PROMPT, max_tokens, port)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    time.sleep(FLUSH_SETTLE_SECONDS)
    with open(log_path, encoding="utf-8") as handle:
        return capture(handle.readlines()), hashlib.sha256(text.encode()).hexdigest()[:16]


def _refusal(answer, lines, label):
    """The two reasons a launch does not start, printed where an operator will see them."""
    reason = (
        "another model process is running; stop it first"
        if answer == "busy"
        else "the model-process guard could not answer what is running"
    )
    print(f"REFUSED: {label} -- {reason}", file=sys.stderr)
    for line in lines:
        print(f"  {line}", file=sys.stderr)


def main() -> int:
    if len(sys.argv) > 1:
        print(
            f"usage: {os.path.basename(sys.argv[0])} takes no arguments -- the arms are "
            f"{', '.join(ARMS)} at {MAX_TOKENS} tokens over {ROUNDS} interleaved rounds",
            file=sys.stderr,
        )
        return 2

    print(f"use-count cap A/B: {bench_model()}, {MAX_TOKENS} tokens, --ram {RAM_BUDGET}")
    print(f"{ROUNDS} rounds of {', '.join(ARMS)}, one fresh server per arm per round")
    print("")

    status = 0
    rounds = []
    for round_ in range(1, ROUNDS + 1):
        by_arm, shas, arms = {}, {}, []
        for arm in ARMS:
            answer, lines = pgrep_answer(["-f", MODEL_PROCESSES])
            if answer != "clear":
                _refusal(answer, lines, f"round {round_} {arm}")
                return 3
            print(f"== round {round_} {arm} ==", flush=True)
            captured, digest = run_arm(arm, round_)
            if captured is None:
                arms.append((f"r{round_} {arm}", None))
                continue
            arms.append((f"r{round_} {arm}", captured))
            by_arm[arm] = figures(captured)
            shas[arm] = digest
        printed, arm_status = channel_verdict(arms, CHANNELS)
        status |= arm_status
        for line in printed:
            print(line)
        printed, invariant_status = invariance(shas)
        status |= invariant_status
        for line in printed:
            print(line)
        rounds.append(by_arm)

    printed, pair_status = pairing_verdict(rounds)
    status |= pair_status
    print("")
    for line in printed:
        print(line)
    return status


if __name__ == "__main__":
    sys.exit(main())
