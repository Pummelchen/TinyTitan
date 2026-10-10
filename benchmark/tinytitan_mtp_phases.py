#!/usr/bin/env python3
"""Track B1: attribute the MTP verify-pass cost, phase by phase.

The strongest historical MTP case (8-bit, 92.6% acceptance, 1.926 tokens per
pass) still lost 2%, which back-solves to a verify pass costing ~1.965x a
single decode token — against a union model predicting well under 1.3x once
`useTwoRowProjection` amortizes attention. This script measures where the
difference actually goes, using the per-pass phase counters recorded by
`StreamingMTPDecoder.advance` and printed by the server's `mtp-phases` footer.

Arms (interleaved off/on/on/off, fresh server per run, greedy, cache off in
both so the only variable is MTP):

  off — plain scalar decode; per-token cost is the denominator
  on  — draft proposal + checkpoint + width-2 verify + commit/rollback

Output: per-pass phase table, the implied verify multiple, and byte-identity
of the emitted text between arms.

  python3 benchmark/tinytitan_mtp_phases.py --quant 8bit --pairs 1
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import pathlib
import re
import signal
import statistics
import subprocess
import sys

import tinytitan_gate0_profile as g0
from tinytitan_profile import (
    DEFAULT_CONTEXT_TOKENS,
    DEFAULT_KV_BITS,
    ROOT,
    arm_answered,
    arm_metric,
    benchmark_log_path,
    byte_claim,
    logged,
    metric_count,
    server_environment,
    resolve_api_model,
)

# Rote continuation: maximally predictable so acceptance is high, mirroring
# the published "predictable function" scenario. 256 tokens is enough passes
# (~100+) to average the phase clocks.
PROMPT = (
    "Write out the multiplication table for 7, from 7 x 1 to 7 x 30, "
    "one line each, in the exact format '7 x N = M'. Output only the "
    "table, nothing else."
)
MAX_TOKENS = 256

# The request's presence penalty. 0 keeps the request pure greedy, which is the
# condition the decode loop checks before it will use the MTP draft at all
# (`RawCompletion.isPureGreedy`). Qwen3.8's instruct row sets 1.5 for a
# thinking-off request, so an MTP qualification on that family must pin 0 here or
# the draft path never engages and both arms measure the scalar decode.
PRESENCE_PENALTY = 0.0

MTP_MODELS = {
    "4bit": ROOT / "models/ornith-1.5_35B_A3B_4Bit",
    "8bit": ROOT / "models/ornith-1.5_35B_A3B_8Bit",
}
SIDECAR = ROOT / "models/ornith-1.5_35B_A3B_MTP_4Bit"
PORT = 8095

MTP_RE = re.compile(
    r"TinyTitan mtp drafted=(\d+) accepted=(\d+) acceptance=([0-9.]+)% "
    r"target_passes=(\d+) emitted_per_pass=([0-9.]+) "
    r"prefill_s=([0-9.]+) decode_s=([0-9.]+) decode_tok_s=([0-9.]+)"
)
PHASES_RE = re.compile(
    r"TinyTitan mtp-phases per_pass_ms proposal=([0-9.]+) checkpoint=([0-9.]+) "
    r"verify=([0-9.]+) verify_backbone=([0-9.]+) verify_head=([0-9.]+) "
    r"verify_argmax=([0-9.]+) commit=([0-9.]+) rollback=([0-9.]+) passes=(\d+)"
)


VERIFY_ARM = "pair"  # TINYTITAN_MTP_VERIFY for the mtp-on arm


def launch(
    target: pathlib.Path, sidecar: pathlib.Path, mtp: bool, log_name: str, ram_budget: str
) -> subprocess.Popen:
    binary = ROOT / ".build/release/TinyTitanServer"
    cmd = [
        str(binary),
        "--port",
        str(PORT),
        "--model",
        str(target),
        "--max-context",
        str(DEFAULT_CONTEXT_TOKENS),
        "--rope-scaling",
        "none",
        "--prompt-cache-mode",
        "off",
        "--prompt-cache-memory-mib",
        "0",
        "--ram-budget",
        ram_budget,
        "--kv-bits",
        str(DEFAULT_KV_BITS),
        "--thinking",
        "off",
    ]
    if mtp:
        cmd += ["--mtp-model", str(sidecar), "--mtp-memory-mib", "384"]
    env = server_environment()
    env["TINYTITAN_RUNNER_STATS"] = "1"
    env["TINYTITAN_KERNEL_STATS"] = "1"
    if mtp:
        env["TINYTITAN_MTP_VERIFY"] = VERIFY_ARM
    log = open(benchmark_log_path(log_name), "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    g0._servers.append(proc)
    return proc


def generate() -> dict | None:
    payload = json.dumps(
        {
            "model": resolve_api_model(PORT),
            "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0,
            "presence_penalty": PRESENCE_PENALTY,
            "seed": 41,
            "max_completion_tokens": MAX_TOKENS,
        },
        separators=(",", ":"),
    ).encode()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=900)
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        body = json.loads(conn.getresponse().read().decode())
        conn.close()
    except (OSError, ValueError) as exc:
        print(f"  ERROR: request failed: {exc}", flush=True)
        return None
    text = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    usage = body.get("usage") or {}
    return {
        "completion_tokens": usage.get("completion_tokens"),
        "sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
    }


def one_run(
    target: pathlib.Path, sidecar: pathlib.Path, mtp: bool, tag: str, ram_budget: str
) -> dict:
    arm = "on" if mtp else "off"
    log_name = f"mtpphases_{target.name}_{arm}_{tag}.log"
    launch(target, sidecar, mtp, log_name, ram_budget)
    if not g0.wait_ready(PORT):
        g0._terminate_all()
        raise SystemExit(f"[{target.name}/{arm}] server not healthy")
    result = generate()
    g0._terminate_all()
    if result is None:
        raise SystemExit(f"[{target.name}/{arm}] request failed")
    row: dict = {"arm": arm, **result}
    with open(benchmark_log_path(log_name), encoding="utf-8") as handle:
        text = handle.read()
    m = MTP_RE.search(text)
    if m:
        row.update(
            drafted=int(m.group(1)),
            accepted=int(m.group(2)),
            acceptance=float(m.group(3)),
            passes=int(m.group(4)),
            emitted_per_pass=float(m.group(5)),
            prefill_s=float(m.group(6)),
            decode_s=float(m.group(7)),
            decode_tok_s=float(m.group(8)),
        )
    else:
        gen = g0.GENERATION_RE.search(text)
        if gen:
            row.update(
                prefill_s=float(gen.group(1)),
                decode_s=float(gen.group(2)),
                decode_tok_s=float(gen.group(3)),
            )
    p = PHASES_RE.search(text)
    if p:
        row["phases"] = {
            k: float(p.group(i + 1))
            for i, k in enumerate(
                (
                    "proposal",
                    "checkpoint",
                    "verify",
                    "verify_backbone",
                    "verify_head",
                    "verify_argmax",
                    "commit",
                    "rollback",
                )
            )
        }
    # Per-role GPU time from the target runner, for the verify decomposition.
    roles: dict[str, float] = {}
    for rm in g0.ROLE_RE.finditer(text):
        roles[rm.group(1)] = roles.get(rm.group(1), 0) + float(rm.group(2))
    row["roles_total_ms"] = roles
    return row


def verdict(rows: list[dict]):
    """(page, exit status) for a finished off/on sweep.

    0  both arms ran, every run of them logged the footer the metrics come
       from, and the two arms emitted the same bytes;
    1  it measured and a claim failed -- `output differs` names the two digest
       sets, `PARTIAL` names the metric and how many of the arm's runs carried
       it;
    2  it could not be compared -- an arm with no runs, a metric no run of an
       arm logged, an arm that streamed no content, or a scalar rate of zero to
       divide by.

    The empty-content case is the one this driver used to certify: the sha256 of
    "" is one digest on both sides, so `output identical: YES` and status 0 came
    out of a sweep that generated nothing. `tinytitan_determinism_ab.py:156-165`
    refuses that shape for the streams it compares, and `arm_answered()` is that
    rule shared.
    """
    lines: list[str] = []
    reasons: list[str] = []
    off = [r for r in rows if r["arm"] == "off"]
    on = [r for r in rows if r["arm"] == "on"]
    for name, sel in (("off", off), ("on", on)):
        if not sel:
            reasons.append(f"NOT MEASURED: the {name} arm has no runs")
        elif not arm_answered(sel):
            reasons.append(
                f"NOT MEASURED: the {name} arm streamed no content, so its digest is the hash of nothing"
            )

    off_rate, off_n, off_total = arm_metric(off, "decode_tok_s")
    on_rate, on_n, on_total = arm_metric(on, "decode_tok_s")
    accept, acc_n, acc_total = arm_metric(on, "acceptance")
    emitted, em_n, em_total = arm_metric(on, "emitted_per_pass")
    for note in (
        metric_count("off", "decode_tok_s", off_n, off_total),
        metric_count("on", "decode_tok_s", on_n, on_total),
        metric_count("on", "acceptance", acc_n, acc_total),
        metric_count("on", "emitted_per_pass", em_n, em_total),
    ):
        if note:
            reasons.append(note)

    if None not in (off_rate, on_rate, accept, emitted):
        if not off_rate:
            reasons.append(
                "NOT MEASURED: the off arm's median is 0 tok/s, so no multiple is computable"
            )
        else:
            token_ms = 1000 / off_rate
            lines.append(f"  scalar decode      {off_rate:.3f} tok/s = {token_ms:.2f} ms/token")
            lines.append(
                f"  MTP decode         {on_rate:.3f} tok/s  "
                f"({(on_rate / off_rate - 1) * 100:+.1f}%)"
            )
            lines.append(f"  acceptance         {accept:.1f}%   emitted/pass {emitted:.3f}")

            phases = [r["phases"] for r in on if "phases" in r]
            if phases:
                if len(phases) < len(on):
                    reasons.append(
                        f"PARTIAL: the per-pass table is over {len(phases)} of {len(on)} on-runs"
                    )
                keys = phases[0].keys()
                pass_ms = {k: statistics.median([p[k] for p in phases]) for k in keys}
                # `verify` already contains backbone+head+argmax; the pass total is
                # the top-level phases only.
                total = sum(
                    pass_ms[k] for k in ("proposal", "checkpoint", "verify", "commit", "rollback")
                )
                lines.append(
                    f"\n  per-pass wall attribution (median across runs; "
                    f"one pass emits {emitted:.3f} tokens):"
                )
                for k in (
                    "proposal",
                    "checkpoint",
                    "verify_backbone",
                    "verify_head",
                    "verify_argmax",
                    "commit",
                    "rollback",
                ):
                    v = pass_ms[k]
                    lines.append(f"    {k:<18} {v:8.3f} ms   {v / token_ms:6.3f}x tokens")
                other = (
                    pass_ms["verify"]
                    - pass_ms["verify_backbone"]
                    - pass_ms["verify_head"]
                    - pass_ms["verify_argmax"]
                )
                lines.append(
                    f"    {'verify_other':<18} {other:8.3f} ms   {other / token_ms:6.3f}x tokens"
                )
                lines.append(f"    {'TOTAL':<18} {total:8.3f} ms   {total / token_ms:6.3f}x tokens")
                lines.append(
                    f"\n  break-even: pass must cost < {emitted:.3f}x a token; "
                    f"it costs {total / token_ms:.3f}x"
                )

    earned, identical, off_digests, on_digests = byte_claim(rows)
    if earned:
        lines.append(
            f"\n  output identical mtp-on vs mtp-off: "
            f"{'YES' if identical else 'NO'} "
            f"(off {off_digests}, on {on_digests})"
        )
        if not identical:
            reasons.append(f"output differs: off {off_digests}, on {on_digests}")
    else:
        lines.append("\n  output identical mtp-on vs mtp-off: NOT MEASURED (see the verdict)")

    if any(reason.startswith("NOT MEASURED") for reason in reasons):
        status = 2
    elif reasons:
        status = 1
    else:
        status = 0
    if reasons:
        lines.append("\n  VERDICT")
        lines.extend(f"    {reason}" for reason in reasons)
    lines.append(f"\n  sweep status {status}")
    return lines, status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quant",
        choices=sorted(MTP_MODELS),
        default="8bit",
        help="the Ornith pair by default; --target overrides it",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="target model directory, for a pair whose quant is "
        "not in MTP_MODELS (e.g. the Qwen3.8 4-bit install)",
    )
    parser.add_argument(
        "--sidecar", default=None, help="MTP sidecar directory to pair with --target"
    )
    parser.add_argument(
        "--ram-budget", default="8G", help="--ram-budget for the server; the Qwen3.8 row wants 12G"
    )
    parser.add_argument(
        "--warmups", type=int, default=1, help="discarded runs per arm (the old default was 1)"
    )
    parser.add_argument(
        "--verify-arm",
        choices=("pair", "tile"),
        default="pair",
        help="TINYTITAN_MTP_VERIFY for the mtp-on arm",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=0.0,
        help="presence penalty sent with the request; 0 keeps it "
        "pure greedy, which MTP requires (Qwen3.8's instruct "
        "row otherwise applies 1.5)",
    )
    parser.add_argument(
        "--pairs", type=int, default=1, help="off/on/on/off blocks after the warmups"
    )
    parser.add_argument("--allow-busy-gpu", action="store_true")
    args = parser.parse_args()
    global VERIFY_ARM, PRESENCE_PENALTY
    VERIFY_ARM = args.verify_arm
    PRESENCE_PENALTY = args.presence_penalty

    if args.target:
        target = (ROOT / args.target).resolve()
        sidecar = (ROOT / args.sidecar).resolve() if args.sidecar else SIDECAR
        label = target.name
    else:
        target = MTP_MODELS[args.quant]
        sidecar = SIDECAR
        label = args.quant
    if not (target / "verified-install.json").exists():
        raise SystemExit(f"not an installed target: {target}")
    if not (sidecar / "manifest.json").exists():
        raise SystemExit(f"no MTP sidecar at {sidecar}")

    signal.signal(signal.SIGINT, g0._on_signal)
    signal.signal(signal.SIGTERM, g0._on_signal)
    g0.preflight(100 if args.allow_busy_gpu else 15)

    rows: list[dict] = []
    try:
        for _ in range(args.warmups):
            for mtp in (False, True):
                print(f"[{label}] warmup mtp={'on' if mtp else 'off'}", flush=True)
                one_run(target, sidecar, mtp, "warmup", args.ram_budget)
        for block in range(args.pairs):
            for mtp in (False, True, True, False):
                row = one_run(target, sidecar, mtp, f"b{block}_{len(rows)}", args.ram_budget)
                rows.append(row)
                extra = (
                    f"acc {logged(row.get('acceptance'), '.1f', '%')}"
                    f" passes {logged(row.get('passes'), 'd')}"
                    if row["arm"] == "on"
                    else ""
                )
                print(
                    f"[{label}] mtp={row['arm']:<3} "
                    f"{logged(row.get('decode_tok_s'), '7.3f')} tok/s  "
                    f"sha {row['sha256']}  {extra}",
                    flush=True,
                )
    finally:
        g0._terminate_all()

    print("\n" + "=" * 70)
    print(f"MTP PHASE ATTRIBUTION — {label}, greedy, cache off, verify={args.verify_arm}")
    print("=" * 70)
    lines, status = verdict(rows)
    for line in lines:
        print(line)

    out = ROOT / (f".build/benchmark-results/mtp-phases-{label}-{args.verify_arm}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump({"rows": rows, "status": status}, handle, indent=2)
    print(f"\nwrote {out}")
    return status


if __name__ == "__main__":
    sys.exit(main())
