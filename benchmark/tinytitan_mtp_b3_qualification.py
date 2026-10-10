#!/usr/bin/env python3
"""Track B3: qualify MTP with the pair verify schedule against scalar decode.

Gate (docs/v4.4-decode-width-plan.md Track B3): on a scenario with acceptance
>= 0.65, MTP-on must beat the MTP-off scalar control by >= 10% median, with
byte-identical greedy output. Run per quantization; arms are interleaved
off/on/on/off with a fresh server per run and a discarded warmup per arm.

Two scenarios:
  table    — rote multiplication table; moderate acceptance (stress case)
  function — predictable docstring'd function; the published high-acceptance
             scenario class (92.6% in the wiki qualification)

  python3 benchmark/tinytitan_mtp_b3_qualification.py --quant 8bit --scenario function
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys

import tinytitan_gate0_profile as g0
import tinytitan_mtp_phases as ph
from tinytitan_profile import ROOT, arm_answered, arm_metric, byte_claim, logged, metric_count

SCENARIOS = {
    "table": ph.PROMPT,
    "function": (
        "Complete this Python file exactly. Output only code, no prose.\n\n"
        "def add(a, b):\n"
        '    """Return the sum of a and b."""\n'
        "    return a + b\n\n"
        "def subtract(a, b):\n"
        '    """Return the difference of a and b."""\n'
        "    return a - b\n\n"
        "def multiply(a, b):\n"
        '    """Return the product of a and b."""\n'
        "    return a * b\n\n"
        "Now write divide, modulo, and power in the identical style."
    ),
}


ARM_METRICS = {
    "off": ("decode_tok_s",),
    "on": ("decode_tok_s", "acceptance", "emitted_per_pass"),
}


def medians(rows):
    """(arms, {(arm, metric): (median, counted, total)}, delta percent).

    The page, the record file and the verdict all read this one table, so a
    number cannot be computed one way for stdout and another way for the status.
    """
    arms = {name: [r for r in rows if r["arm"] == name] for name in ARM_METRICS}
    got = {
        (name, key): arm_metric(arms[name], key)
        for name, keys in ARM_METRICS.items()
        for key in keys
    }
    off_rate = got[("off", "decode_tok_s")][0]
    on_rate = got[("on", "decode_tok_s")][0]
    delta = None if not off_rate or on_rate is None else (on_rate / off_rate - 1) * 100
    return arms, got, delta


def verdict(rows):
    """(page, exit status) for a finished B3 qualification.

    The gate is this file's own docstring: on a scenario with acceptance >= 0.65
    the MTP arm must beat the scalar control by >= 10% median, with byte-identical
    greedy output. Three outcomes, three statuses:

    0  in domain, the margin cleared it, and the two arms emitted the same bytes;
    1  in domain and it did not clear it, or the bytes differ;
    2  the gate could not be answered -- an arm with no runs, a metric no run
       logged, arms that streamed no content, or an acceptance below the 0.65
       the gate is conditioned on. The old code printed OUT OF DOMAIN and, unless
       the digests happened to differ, returned 0.
    """
    lines: list[str] = []
    reasons: list[str] = []
    arms, got, delta = medians(rows)
    off, on = arms["off"], arms["on"]
    for name, sel in arms.items():
        if not sel:
            reasons.append(f"NOT MEASURED: the {name} arm has no runs")
        elif not arm_answered(sel):
            reasons.append(
                f"NOT MEASURED: the {name} arm streamed no content, "
                "so its digest is the hash of nothing"
            )
    for (name, key), (_value, counted, total) in got.items():
        note = metric_count(name, key, counted, total)
        if note:
            reasons.append(note)

    off_rate = got[("off", "decode_tok_s")][0]
    on_rate = got[("on", "decode_tok_s")][0]
    accept = got[("on", "acceptance")][0]
    emitted = got[("on", "emitted_per_pass")][0]
    if off_rate is not None and on_rate is not None:
        lines.append(
            f"  scalar   median {off_rate:7.3f} tok/s  runs {[r.get('decode_tok_s') for r in off]}"
        )
        lines.append(
            f"  MTP      median {on_rate:7.3f} tok/s  runs {[r.get('decode_tok_s') for r in on]}"
        )
    if accept is not None and emitted is not None:
        lines.append(f"  acceptance {accept:.1f}%   emitted/pass {emitted:.3f}")
    if None in (off_rate, on_rate, accept, emitted):
        reasons.append("NOT MEASURED: an arm carried no rate, so no delta is computable")
    elif not off_rate:
        reasons.append("NOT MEASURED: the off arm's median is 0 tok/s, so no delta is computable")
    else:
        if accept < 65:
            word = f"OUT OF DOMAIN (acceptance {accept:.1f}% < 65%)"
            reasons.append(
                f"NOT MEASURED: acceptance {accept:.1f}% is below the 65% "
                "the +10% gate is conditioned on"
            )
        elif delta >= 10:
            word = "PASS"
        else:
            word = "FAIL"
            reasons.append(f"gate failed: median delta {delta:+.2f}% is below the +10% B3 margin")
        lines.append(f"  DELTA: {delta:+.2f}%   gate +10% at p>=0.65: {word}")

    earned, identical, off_digests, on_digests = byte_claim(rows)
    if earned:
        lines.append(
            f"  output identical: {'YES' if identical else 'NO'} "
            f"(off {off_digests}, on {on_digests})"
        )
        if not identical:
            reasons.append(f"output differs: off {off_digests}, on {on_digests}")
    else:
        lines.append("  output identical: NOT MEASURED (see the verdict)")

    if any(reason.startswith("NOT MEASURED") for reason in reasons):
        status = 2
    elif reasons:
        status = 1
    else:
        status = 0
    if reasons:
        lines.append("\n  VERDICT")
        lines.extend(f"    {reason}" for reason in reasons)
    lines.append(f"\n  qualification status {status}")
    return lines, status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant", choices=sorted(ph.MTP_MODELS), default="8bit")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="function")
    parser.add_argument(
        "--blocks", type=int, default=2, help="off/on/on/off blocks after the two warmups"
    )
    parser.add_argument(
        "--target", default=None, help="target model directory (e.g. the Qwen3.8 4-bit install)"
    )
    parser.add_argument(
        "--sidecar", default=None, help="MTP sidecar directory to pair with --target"
    )
    parser.add_argument("--ram-budget", default="8G", help="--ram-budget for the server")
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=0.0,
        help="presence penalty sent with the request; 0 keeps it pure greedy, which MTP requires",
    )
    parser.add_argument("--allow-busy-gpu", action="store_true")
    args = parser.parse_args()

    ph.PROMPT = SCENARIOS[args.scenario]
    ph.VERIFY_ARM = "pair"
    ph.PRESENCE_PENALTY = args.presence_penalty
    if args.target:
        target = (ph.ROOT / args.target).resolve()
        sidecar = (ph.ROOT / args.sidecar).resolve() if args.sidecar else ph.SIDECAR
        label = target.name
    else:
        target = ph.MTP_MODELS[args.quant]
        sidecar = ph.SIDECAR
        label = args.quant

    signal.signal(signal.SIGINT, g0._on_signal)
    signal.signal(signal.SIGTERM, g0._on_signal)
    g0.preflight(100 if args.allow_busy_gpu else 15)

    rows: list[dict] = []
    tag_prefix = f"b3_{args.scenario}"
    try:
        for mtp in (False, True):
            print(f"[{label}/{args.scenario}] warmup mtp={'on' if mtp else 'off'}", flush=True)
            ph.one_run(target, sidecar, mtp, f"{tag_prefix}_warmup", args.ram_budget)
        for block in range(args.blocks):
            for mtp in (False, True, True, False):
                row = ph.one_run(
                    target, sidecar, mtp, f"{tag_prefix}_b{block}_{len(rows)}", args.ram_budget
                )
                rows.append(row)
                extra = (
                    f"acc {logged(row.get('acceptance'), '.1f', '%')}" if row["arm"] == "on" else ""
                )
                print(
                    f"[{args.quant}/{args.scenario}] mtp={row['arm']:<3} "
                    f"{logged(row.get('decode_tok_s'), '7.3f')} tok/s  "
                    f"sha {row['sha256']}  {extra}",
                    flush=True,
                )
    finally:
        g0._terminate_all()

    print("\n" + "=" * 70)
    print(
        f"B3 QUALIFICATION — {args.quant}, scenario={args.scenario}, verify=pair, greedy, cache off"
    )
    print("=" * 70)
    lines, status = verdict(rows)
    for line in lines:
        print(line)

    _arms, got, delta = medians(rows)
    _earned, identical, _off_digests, _on_digests = byte_claim(rows)
    out = ROOT / (f".build/benchmark-results/mtp-b3-{args.quant}-{args.scenario}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    record = {
        "rows": rows,
        "status": status,
        "output_identical": identical,
    }
    if delta is not None:
        record["delta_percent"] = round(delta, 2)
    acceptance = got[("on", "acceptance")][0]
    if acceptance is not None:
        record["acceptance"] = acceptance
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    print(f"\nwrote {out}")
    return status


if __name__ == "__main__":
    sys.exit(main())
