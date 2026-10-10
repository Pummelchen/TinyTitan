#!/usr/bin/env python3
"""Sweep decode knobs one at a time against a fixed baseline, on one model.

Each arm is a fresh server, because every knob here is read once at startup.
One prompt, one warm measured request, the server's own decode_tok_s footer
plus the runner counters that explain *why* an arm moved -- a rate without
io_hidden_pct or the hit rate is a number you cannot act on.

    python3 benchmark/tinytitan_knob_sweep.py                    # all arms
    python3 benchmark/tinytitan_knob_sweep.py --arms base,sync_event

The baseline is re-run last. This machine has +/-15% run-to-run spread and a
sweep takes hours, so a drifting baseline is the difference between a real 8%
win and a machine that got quieter. If the two baselines disagree by more than
the smallest win claimed, the sweep is inconclusive and says so -- in the exit
status as well as on the page. AUD-277.

    0  every arm ran, every published cell came from a run that logged it, and
       the drift control bounds the wins the page claims
    1  it measured, and something the page claims is contested -- the claim named
    2  the headline could not be computed, with the reason named
    3  another model process is running (the guard's own answer, pre-existing)

A counter no run logged is published as `--` in the table and `not logged` in
the live line, never as `0.0`: a hit rate that was never measured and prints as
zero reads as a cache that never hit.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tinytitan_profile import (  # noqa: E402
    benchmark_log_path,
    logged,
    pgrep_answer,
    server_command,
    server_environment,
)

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / ".build/release/TinyTitanServer"
MODEL = os.environ.get(
    "TINYTITAN_BENCH_MODEL", str(ROOT / "models/qwen3.8-flash-next_125B_A6B_4Bit")
)
PORT = 8131
# Read per run, not at import: an unparseable SWEEP_TOKENS used to raise out of
# the module before the guard answered or a refusal line printed.
DEFAULT_TOKENS = 256
BASELINE = "base"
DRIFT_CONTROL = "base_again"
PROMPT = "Write a detailed essay about the history of computing."

# Every arm is one env delta from the shipped defaults, so a win is
# attributable. Combinations come later, built from what wins here.
ARMS: list[tuple[str, dict[str, str], str]] = [
    ("base", {}, "shipped defaults"),
    (
        "sync_event",
        {"TINYTITAN_EXPERT_IO_SYNC": "event"},
        "GPU event instead of a host wait per layer (25.7 host waits/token)",
    ),
    (
        "submit_now",
        {"TINYTITAN_EXPERT_IO_SUBMISSION": "immediate"},
        "submit expert reads as planned, not deferred (io_hidden_pct 18.9%)",
    ),
    (
        "gpu_resident",
        {"TINYTITAN_DECODE_EXPERT_EXECUTION": "gpu-residency"},
        "GPU-side hit/miss classification (gpu_classified_* all zero today)",
    ),
    (
        "barrier",
        {"TINYTITAN_DECODE_EXPERT_EXECUTION": "barrier"},
        "control: the simple schedule, expected slower",
    ),
    # The cache-policy, cache-layout, prefetch-depth and retention arms are
    # gone with their knobs: each measured a wash or a loss, was documented, and
    # was then removed from the engine (their env vars no longer exist, so an arm
    # here would silently measure the baseline).
    (
        "prefetch_off",
        {"TINYTITAN_PREDICTIVE_PREFETCH": "0"},
        "control: confirm prefetch still earns its place",
    ),
    (
        "slots_128",
        {"TINYTITAN_EXPERT_CACHE_SLOTS": "128"},
        "hit rate 87.4% at 96; does the curve still climb?",
    ),
    (
        "slots_64",
        {"TINYTITAN_EXPERT_CACHE_SLOTS": "64"},
        "control: fewer slots must be worse if the cache matters",
    ),
    # --- second pass: knobs the first sweep listed but never ran -------------
    (
        "rdadvise_off",
        {"TINYTITAN_RDADVISE_POLICY": "off"},
        "rdadvise costs 5.1 ms/token, 3% of the budget",
    ),
    ("rdadvise_adaptive", {"TINYTITAN_RDADVISE_POLICY": "adaptive"}, "ditto"),
    ("rdadvise_bounded", {"TINYTITAN_RDADVISE_POLICY": "bounded"}, "ditto"),
    (
        "io_metal",
        {"TINYTITAN_EXPERT_IO_BACKEND": "metal"},
        "the other I/O backend, never measured on this model",
    ),
    ("bounded_io_off", {"TINYTITAN_BOUNDED_IO": "0"}, "unbounded reader footprint"),
    (
        "sampler_generic",
        {"TINYTITAN_SAMPLER_PATH": "generic"},
        "control: the tiled sampler should win",
    ),
    (
        "slots_112",
        {"TINYTITAN_EXPERT_CACHE_SLOTS": "112"},
        "the untested middle: 96 fits at 85.4%, 128 swaps at 89.8%",
    ),
    # --- 8-bit: the cache is sized in bytes, so a 1.89x expert stride buys
    # fewer slots. 64 slots at 8-bit is 16.1 GB of cache -- essentially the
    # footprint that cost 4-bit 68% at 128 slots. Sweep downward.
    ("slots_32", {"TINYTITAN_EXPERT_CACHE_SLOTS": "32"}, "8-bit: ~8 GB of cache"),
    ("slots_24", {"TINYTITAN_EXPERT_CACHE_SLOTS": "24"}, "8-bit: ~6 GB of cache"),
    ("slots_16", {"TINYTITAN_EXPERT_CACHE_SLOTS": "16"}, "8-bit: ~4 GB of cache"),
    # --- GDN in_proj kernel variants: gdn.metal has carried these since the
    # kernel was written and nothing ever selected them. attn_norm_qkv is
    # 29.8 ms/token at an effective 26.2 GB/s against ~100 GB/s peak, and every
    # row re-reads the 5 KiB x vector from device memory.
    ("gdn_xsh8", {"TINYTITAN_GDN_INPROJ": "xsh8"}, "stage x in threadgroup memory"),
    ("gdn_r16", {"TINYTITAN_GDN_INPROJ": "r16"}, "16 rows/threadgroup, x unstaged"),
    ("gdn_xsh16", {"TINYTITAN_GDN_INPROJ": "xsh16"}, "both"),
    # Each async piece was measured alone, where it pays its own overhead and
    # still cannot remove the host wait because the other two force one.
    # Together they are the only configuration that actually removes it.
    (
        "async_all",
        {
            "TINYTITAN_DECODE_EXPERT_EXECUTION": "gpu-residency",
            "TINYTITAN_EXPERT_IO_SYNC": "event",
            "TINYTITAN_EXPERT_IO_BACKEND": "metal",
        },
        "GPU classification + event sync + MTLIO together",
    ),
    (
        "async_event_metal",
        {"TINYTITAN_EXPERT_IO_SYNC": "event", "TINYTITAN_EXPERT_IO_BACKEND": "metal"},
        "event sync on the backend that signals the event natively",
    ),
    (
        "async_pread",
        {"TINYTITAN_DECODE_EXPERT_EXECUTION": "gpu-residency", "TINYTITAN_EXPERT_IO_SYNC": "event"},
        "GPU classification + event sync on pread -- avoids the MTLIO crash",
    ),
    ("base_again", {}, "drift check -- must match base"),
]

COUNTERS = (
    "expert_hit_rate",
    "io_hidden_pct",
    "io_ms",
    "wait_ms",
    "body_ms",
    "expert_evictions",
    "io_host_waits",
    "cache_plan_ms",
    "rdadvise_ms",
)

# The five counters the published table prints, in the order a reader needs them
# rather than the order of the columns: this file's own rule is that "a rate
# without io_hidden_pct or the hit rate is a number you cannot act on", so those
# two are the first the verdict looks for. `wait_ms`, `body_ms`, `cache_plan_ms`
# and `rdadvise_ms` are parsed but never printed, so no cell claims them.
PRINTED_CELLS = (
    "io_hidden_pct",
    "expert_hit_rate",
    "io_ms",
    "expert_evictions",
    "io_host_waits",
)


def measured_tokens(environ: dict[str, str]) -> tuple[int | None, str | None]:
    """The length of the measured request, or the reason this one refuses.

    A count of 0 asks the server for no tokens at all, so every arm "measures"
    the same nothing; an unparseable value used to die at import.
    """
    raw = environ.get("SWEEP_TOKENS")
    if raw is None:
        return DEFAULT_TOKENS, None
    try:
        value = int(raw)
    except ValueError:
        return None, f"SWEEP_TOKENS={raw} is not an integer"
    if value <= 0:
        return None, f"SWEEP_TOKENS={raw} measures no tokens"
    return value, None


def selected_arms(wanted: str) -> tuple[list[tuple[str, dict[str, str], str]], list[str]]:
    """The arms to run, in sweep order, and the names that matched no arm.

    A typo in `--arms` used to select nothing and the sweep printed no row at
    all, then exited 0 as if it had run an empty sweep it was reporting.
    """
    names = [a.strip() for a in wanted.split(",") if a.strip()]
    if not names:
        return list(ARMS), []
    known = {a[0] for a in ARMS}
    chosen = [a for a in ARMS if a[0] in set(names)]
    return chosen, [n for n in names if n not in known]


def cell(
    row: dict,
    key: str,
    scale: float = 1.0,
    spec: str = ".1f",
    suffix: str = "",
    refusal: str = "not logged",
) -> str:
    """One published counter, or the token that says no run logged it.

    The table passes `--`, the token the column already uses for a cell it
    cannot answer; the live line keeps the words.
    """
    value = row.get(key)
    return logged(None if value is None else value * scale, spec, suffix, refusal)


def wait_ready(proc, timeout=2400) -> str | None:
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            return None
        try:
            conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=2)
            conn.request("GET", "/v1/models")
            mid = json.loads(conn.getresponse().read())["data"][0]["id"]
            conn.close()
            return mid
        except OSError:
            time.sleep(5)
    return None


def request(model_id: str, tokens: int) -> None:
    payload = json.dumps(
        {
            "model": model_id,
            "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0,
            "top_p": 0.95,
            "top_k": 20,
            "max_completion_tokens": tokens,
            "stream": True,
        }
    ).encode()
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=3600)
    conn.request(
        "POST", "/v1/chat/completions", body=payload, headers={"Content-Type": "application/json"}
    )
    resp = conn.getresponse()
    while resp.read(8192):
        pass
    conn.close()


def run_arm(name: str, env_delta: dict[str, str], tokens: int) -> dict:
    log_path = benchmark_log_path(f"sweep_{name}.log")
    env = server_environment()
    # TINYTITAN_RUNNER_STATS is what makes a result explainable rather than just
    # faster or slower.
    env["TINYTITAN_RUNNER_STATS"] = "1"
    env.update(env_delta)
    with open(log_path, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            server_command(BIN, PORT, model=MODEL), env=env, stdout=fh, stderr=subprocess.STDOUT
        )
    try:
        model_id = wait_ready(proc)
        if model_id is None:
            tail = Path(log_path).read_text(encoding="utf-8").strip().splitlines()[-3:]
            return {"arm": name, "ok": False, "note": "; ".join(tail)[:160]}
        # An arm that crashes the server drops the stream mid-read. Record it
        # and continue: losing the rest of a multi-hour sweep to one bad
        # configuration is worse than losing the arm.
        try:
            request(model_id, 32)  # warm shaders and the cache
            request(model_id, tokens)  # measured
        except Exception as exc:
            alive = proc.poll() is None
            return {
                "arm": name,
                "ok": False,
                "note": f"{type(exc).__name__}: {exc}"[:110] + ("" if alive else " (server died)"),
            }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(3)

    text = Path(log_path).read_text(encoding="utf-8")
    rates = [float(m) for m in re.findall(r"TinyTitan generation .*?decode_tok_s=([\d.]+)", text)]
    if not rates:
        return {"arm": name, "ok": False, "note": "no decode footer"}
    out = {"arm": name, "ok": True, "tok_s": rates[-1]}
    runner = re.findall(r"TinyTitan runner (.+)", text)
    if runner:
        fields = dict(kv.split("=", 1) for kv in runner[-1].split() if "=" in kv)
        for c in COUNTERS:
            if c in fields:
                out[c] = float(fields[c])
    return out


def table(results: list[dict], baseline: float | None) -> str:
    head = (
        f"{'arm':<15}{'tok/s':>8}{'vs base':>9}{'hit%':>7}"
        f"{'io_hid%':>9}{'io_ms':>8}{'evict':>8}{'waits':>8}"
    )
    lines = [head, "-" * len(head)]
    for r in results:
        if not r.get("ok"):
            lines.append(f"{r['arm']:<15}{'FAILED':>8}   {r.get('note', '')[:50]}")
            continue
        delta = f"{(r['tok_s'] / baseline - 1) * 100:+.1f}%" if baseline else "--"
        lines.append(
            f"{r['arm']:<15}{r['tok_s']:>8.2f}{delta:>9}"
            f"{cell(r, 'expert_hit_rate', 100, refusal='--'):>7}"
            f"{cell(r, 'io_hidden_pct', refusal='--'):>9}"
            f"{cell(r, 'io_ms', refusal='--'):>8}"
            f"{cell(r, 'expert_evictions', spec='.0f', refusal='--'):>8}"
            f"{cell(r, 'io_host_waits', spec='.0f', refusal='--'):>8}"
        )
    return "\n".join(lines)


def verdict(results: list[dict], selected: list[str]) -> tuple[list[str], int]:
    """What the sweep may claim, and the status the page is judged by. AUD-277.

    Every arm used to be compared and then discarded: `main()` ended on
    `return 0` whatever it measured, and the sentence it printed after the drift
    compared that drift to nothing. The statuses are the ones this tree's other
    drivers use (AUD-273, AUD-274, AUD-275); the drift rule is the one this
    sweep's own header has always stated.
    """
    lines: list[str] = []
    measured = {r["arm"]: r for r in results if r.get("ok")}
    if not measured:
        return ["NOT MEASURED: no arm of this sweep measured anything"], 2
    if BASELINE not in selected:
        return ["NOT MEASURED: the selection has no baseline arm to compare to"], 2
    if BASELINE not in measured:
        return ["NOT MEASURED: no baseline arm measured"], 2
    for row in results:
        if not row.get("ok"):
            continue
        for key in PRINTED_CELLS:
            if key not in row:
                lines.append(f"NOT MEASURED: the {row['arm']} arm logged no {key}")
                return lines, 2
    if DRIFT_CONTROL in selected and DRIFT_CONTROL not in measured:
        lines.append(
            f"NOT MEASURED: {DRIFT_CONTROL}, the drift control, did not measure, "
            "so the sweep has no noise floor to bound its wins"
        )
        return lines, 2

    status = 0
    for row in results:
        if not row.get("ok"):
            lines.append(f"CONTESTED: {row['arm']} did not measure ({row.get('note', '')})")
            status = 1

    first = measured[BASELINE]["tok_s"]
    # The drift control is the baseline's own configuration, so its delta is the
    # drift and never a win the page claims.
    wins = [
        (row["arm"], (row["tok_s"] / first - 1) * 100)
        for row in results
        if row.get("ok") and row["arm"] not in (BASELINE, DRIFT_CONTROL) and row["tok_s"] > first
    ]
    if DRIFT_CONTROL not in measured:
        lines.append(
            f"CONTESTED: {DRIFT_CONTROL} was not run, so the noise floor is unknown "
            "and no win on this page is bounded by it"
        )
        return lines, 1

    last = measured[DRIFT_CONTROL]["tok_s"]
    drift = abs(last / first - 1) * 100
    lines.append(f"baseline drift over the sweep: {drift:.1f}% ({first:.2f} -> {last:.2f})")
    if not wins:
        lines.append(f"{drift:.1f}% of drift swallows nothing: no arm claims a win")
        return lines, status
    arm, smallest = min(wins, key=lambda w: w[1])
    if drift > smallest:
        lines.append(
            f"INCONCLUSIVE: baseline drift {drift:.1f}% is larger than the smallest "
            f"win claimed, {smallest:+.1f}% on {arm}"
        )
        return lines, 1
    lines.append(f"smallest win claimed {smallest:+.1f}% on {arm}, against {drift:.1f}% of drift")
    return lines, status


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="")
    args = ap.parse_args()
    arms, unknown = selected_arms(args.arms)
    if unknown:
        print(
            f"NOT MEASURED: --arms names no arm of this sweep: {', '.join(unknown)}",
            file=sys.stderr,
        )
        print(f"  known arms: {', '.join(a[0] for a in ARMS)}", file=sys.stderr)
        return 2

    tokens, reason = measured_tokens(os.environ)
    if tokens is None:
        print(f"NOT MEASURED: {reason}", file=sys.stderr)
        return 2

    answer, lines = pgrep_answer(["-f", "TinyTitanServer|TinyTitanCLI"])
    if answer != "clear":
        reason = (
            "another model process is running; stop it first"
            if answer == "busy"
            else "the model-process guard could not answer what is running"
        )
        print(reason, file=sys.stderr)
        for line in lines:
            print(f"  {line}", file=sys.stderr)
        return 3

    results: list[dict] = []
    baseline: float | None = None
    for name, delta, why in arms:
        print(f"\n== {name} == {why}", flush=True)
        r = run_arm(name, delta, tokens)
        results.append(r)
        if r.get("ok"):
            if name == BASELINE:
                baseline = r["tok_s"]
            print(
                f"   {r['tok_s']:.2f} tok/s"
                f"  hit={cell(r, 'expert_hit_rate', 100, suffix='%')}"
                f"  io_hidden={cell(r, 'io_hidden_pct', suffix='%')}"
                f"  io_ms={cell(r, 'io_ms')}",
                flush=True,
            )
        else:
            print(f"   FAILED: {r.get('note')}", flush=True)
        print("\n" + table(results, baseline), flush=True)

    lines, status = verdict(results, [a[0] for a in arms])
    print("\n" + "\n".join(lines), flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
