#!/usr/bin/env python3
"""Interleaved A/B for the Top-K sampler path, at the published v4.1 profile.

`generic` is the pre-change behavior: for any k != 64 the sampler fell through
to a single-threadgroup kernel that extracts Top-K in k full passes over a
262,144-entry vocabulary. `tiled` routes every k in 1...64 to the existing
three-stage reduction. The two are required to emit the same token — this
script checks that on the real model and measures what it costs.

Both arms run from **one binary in one machine state**, alternating
A/B/B/A with a fresh server process per run and a discarded warmup per arm,
because throughput on this Mac carries run-to-run spread wider than many
effects being tested and a sequential sweep warms the page cache as it goes.

The page carries three statuses and `main()` returns them: 0 when every figure
it published came from a run that logged it and the +10% gate cleared, 1 when it
measured and a claim failed (the gate, or digests that disagree), 2 when the
headline could not be computed -- a metric no run of an arm logged, an arm that
streamed no content, or a 0 tok/s median to divide by. A busy figure is a
separate metric line from the decode footer, and the sampler gap only appears
when the trace matched, so both are counted rather than defaulted to zero.

  python3 benchmark/tinytitan_sampler_ab.py --quant 4bit --pairs 2
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys

import tinytitan_gate0_profile as g0
from tinytitan_profile import (
    ROOT,
    arm_answered,
    arm_metric,
    byte_claim,
    metric_count,
)

ARMS = ("generic", "tiled")
GAP = "head_logits->embed"
GAP_KEY = f"{GAP} per_token_ms"
PAGE_METRICS = ("decode_tok_s", "busy_per_token_ms", GAP_KEY)


def one_run(quant: str, arm: str, tag: str) -> dict:
    port = g0.PORTS[quant]
    log_name = f"samplerab_{quant}_{arm}_{tag}.log"
    g0.launch(quant, port, log_name, sampler_path=arm)
    if not g0.wait_ready(port):
        g0._terminate_all()
        raise SystemExit(f"[{quant}/{arm}] server did not become healthy")
    result = g0.generate(port)
    # stdout is block-buffered into the log; footers land only on exit.
    g0._terminate_all()
    if result is None:
        raise SystemExit(f"[{quant}/{arm}] request failed")
    records = g0.parse_log(g0.benchmark_log_path(log_name))
    if not records:
        raise SystemExit(f"[{quant}/{arm}] no footer lines")
    record = records[-1]
    record.update(result)
    record["arm"] = arm
    return record


def _logged(row: dict, key: str):
    """The run's value for a published metric, or None when it never logged it.

    `busy_per_token_ms` arrives on the occupancy line and the sampler gap on a
    trace line, so either can be absent while the decode footer the rate comes
    from is present; `row.get(key, 0)` read that absence as a measured zero.
    """
    if key == GAP_KEY:
        return (row.get("gaps") or {}).get(GAP, {}).get("per_token_ms")
    return row.get(key)


def summarize(rows: list[dict]) -> dict:
    """Per-arm medians over the runs that logged each metric, plus the counts."""
    out: dict = {"arms": {}}
    for arm in ARMS:
        sel = [{**r, GAP_KEY: _logged(r, GAP_KEY)} for r in rows if r.get("arm") == arm]
        medians = {}
        counts = {}
        for key in PAGE_METRICS:
            value, counted, total = arm_metric(sel, key)
            medians[key] = None if value is None else round(value, 4)
            counts[key] = [counted, total]
        out["arms"][arm] = {
            "runs": len(sel),
            "answered": arm_answered(sel),
            "median_tok_s": medians["decode_tok_s"],
            "rates": [round(r["decode_tok_s"], 4) for r in sel],
            "median_busy_per_token_ms": medians["busy_per_token_ms"],
            "median_sample_gap_ms": medians[GAP_KEY],
            "counts": counts,
            "digests": sorted({r["completion_sha256"] for r in sel}),
        }
    generic, tiled = out["arms"]["generic"], out["arms"]["tiled"]
    # A 0 tok/s median is a refusal, not a denominator: dividing by it raises, and
    # the crash used to arrive before the page printed anything.
    if generic["median_tok_s"] and tiled["median_tok_s"] is not None:
        out["delta_percent"] = round((tiled["median_tok_s"] / generic["median_tok_s"] - 1) * 100, 2)
    else:
        out["delta_percent"] = None
    earned, identical, left, right = byte_claim(rows, *ARMS, digest_key="completion_sha256")
    out["output_identical"] = identical
    out["identity_earned"] = earned
    out["digest_sets"] = [left, right]
    return out


def _metric(value, spec: str, unit: str) -> str:
    if value is None:
        return "not logged"
    return f"{value:{spec}} {unit}"


def verdict(summary: dict, quant: str, gate_percent: float = 10.0):
    """(page, exit status) for a finished sampler A/B.

    0  both arms ran, every published figure came from a run that logged it, the
       two arms emitted the same bytes, and the arm cleared the gate;
    1  it measured and a claim failed -- `gate failed` names the delta,
       `output differs` names both digest sets, `PARTIAL` names the metric and
       how many of the arm's runs carried it;
    2  the headline could not be computed -- an arm with no runs, a metric no run
       of an arm logged, an arm that streamed no content, or a 0 tok/s median.
    """
    lines: list[str] = []
    reasons: list[str] = []
    lines.append("=" * 68)
    lines.append(f"SAMPLER A/B — {quant}, published v4.1 profile, one binary")
    lines.append("=" * 68)

    if not summary["arms"]["generic"]["runs"] and not summary["arms"]["tiled"]["runs"]:
        reasons.append("NOT MEASURED: no run at all -- the sweep ran no measured block")
    for arm in ARMS:
        d = summary["arms"][arm]
        if not d["runs"]:
            reasons.append(f"NOT MEASURED: the {arm} arm has no runs")
        elif not d["answered"]:
            reasons.append(
                f"NOT MEASURED: the {arm} arm streamed no content, so its digest "
                "is the hash of nothing"
            )
        for key in PAGE_METRICS:
            note = metric_count(arm, key, *d["counts"][key])
            if note:
                reasons.append(note)

    generic, tiled = summary["arms"]["generic"], summary["arms"]["tiled"]
    if generic["median_tok_s"] == 0:
        reasons.append(
            "NOT MEASURED: the generic arm's median is 0 tok/s, so no delta is computable"
        )
    for arm, d in (("generic", generic), ("tiled", tiled)):
        rate = "not logged" if d["median_tok_s"] is None else f"{d['median_tok_s']:7.3f} tok/s"
        lines.append(f"  {arm:<8} median {rate:>14}   runs {d['rates']}")
        lines.append(
            f"           busy/token "
            f"{_metric(d['median_busy_per_token_ms'], '.2f', 'ms')}   "
            f"sampler gap {_metric(d['median_sample_gap_ms'], '.2f', 'ms')}"
        )

    delta = summary["delta_percent"]
    if delta is not None:
        word = "PASS" if delta >= gate_percent else "FAIL"
        lines.append("")
        lines.append(f"  DELTA: {delta:+.2f}%   (gate is +{gate_percent:g}%: {word})")
        if word == "FAIL":
            reasons.append(f"gate failed: median delta {delta:+.2f}% is below +{gate_percent:g}%")

    lines.append("")
    if not summary["identity_earned"]:
        lines.append("  Output identical across every run of both arms: NOT MEASURED")
    else:
        lines.append(
            "  Output identical across every run of both arms: "
            f"{'YES' if summary['output_identical'] else 'NO'}"
        )
        if not summary["output_identical"]:
            left, right = summary["digest_sets"]
            lines.append(f"    generic digests {left}")
            lines.append(f"    tiled   digests {right}")
            lines.append(
                "    A sampling change that moves the token stream is a numerics "
                "change and must be re-baselined deliberately, not accepted here."
            )
            reasons.append(f"output differs: generic {left}, tiled {right}")

    if any(reason.startswith("NOT MEASURED") for reason in reasons):
        status = 2
    elif reasons:
        status = 1
    else:
        status = 0
    if reasons:
        lines.append("")
        lines.append("  VERDICT")
        lines.extend(f"    {reason}" for reason in reasons)
    lines.append("")
    lines.append(f"  sweep status {status}")
    return lines, status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant", choices=sorted(g0.QUANTS), default="4bit")
    parser.add_argument(
        "--pairs", type=int, default=2, help="A/B/B/A blocks; each block is 4 measured runs"
    )
    parser.add_argument("--allow-busy-gpu", action="store_true")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, g0._on_signal)
    signal.signal(signal.SIGTERM, g0._on_signal)

    rows: list[dict] = []
    if args.pairs < 1:
        # A block count that asks for no measured run cannot be answered by
        # launching servers, so the refusal is the page and nothing is started.
        print(f"[{args.quant}] --pairs {args.pairs} runs no measured block", flush=True)
    else:
        g0.preflight(100 if args.allow_busy_gpu else 15)
        try:
            # One discarded warmup per arm before any measurement, so neither arm
            # pays the first-touch cost the other does not.
            for arm in ARMS:
                print(f"[{args.quant}] warmup {arm}", flush=True)
                one_run(args.quant, arm, "warmup")
            for block in range(args.pairs):
                for arm in ("generic", "tiled", "tiled", "generic"):
                    row = one_run(args.quant, arm, f"b{block}_{len(rows)}")
                    rows.append(row)
                    print(
                        f"[{args.quant}] {arm:<7} "
                        f"{row['decode_tok_s']:.3f} tok/s  "
                        f"busy/token "
                        f"{_metric(row.get('busy_per_token_ms'), '.2f', 'ms')}  "
                        f"sample_gap "
                        f"{_metric(_logged(row, GAP_KEY), '.2f', 'ms')}  "
                        f"sha {row['completion_sha256']}",
                        flush=True,
                    )
        finally:
            g0._terminate_all()

    summary = summarize(rows)
    lines, status = verdict(summary, args.quant)
    print("\n".join(lines), flush=True)

    path = ROOT / f".build/benchmark-results/sampler-ab-{args.quant}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    artifact = {"quant": args.quant, **summary, "status": status}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2)
    print(f"\nwrote {path}")
    return status


if __name__ == "__main__":
    sys.exit(main())
