#!/usr/bin/env python3
"""Track A qualification: ANE prefill against the GPU path, interleaved.

Arms differ in exactly one variable: TINYTITAN_PREFILL_ANE. Fresh server per run,
one discarded warmup per arm, off/on/on/off blocks. The prompt is
prefill-dominated (~6.1K tokens of project documentation); a short greedy
continuation confirms decode runs correctly off the KV cache the ANE path
wrote.

Interpretation notes:
- prefill_s is the qualification metric.
- Output digests are expected to be STABLE WITHIN an arm (greedy) but to
  DIFFER BETWEEN arms: the ANE computes attention in fp16 with a different
  reduction order (~1% per-layer deviation). The script prints both digests
  and the first line of each arm's continuation so plausibility is visible.

  python3 benchmark/tinytitan_ane_prefill_ab.py --quant 4bit --pairs 1
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import signal
import subprocess
import sys

import tinytitan_gate0_profile as g0
from tinytitan_profile import (
    ROOT,
    arm_answered,
    arm_metric,
    benchmark_log_path,
    logged,
    metric_count,
    server_command,
    server_environment,
    resolve_api_model,
)

PORT = 8098
MODELS = {
    "4bit": ROOT / "models/ornith-1.5_35B_A3B_4Bit",
    "8bit": ROOT / "models/ornith-1.5_35B_A3B_8Bit",
}


# A pinned, self-contained prompt body. It was previously built from README.md
# and AGENTS.md, which made the benchmark silently non-reproducible: editing
# those files changed the prompt length (6,103 -> 6,593 tokens between two
# runs) and therefore the quadratic attention cost, so the arms of different
# sweeps were not comparable. A benchmark prompt must never be derived from
# files under active edit.
_PARAGRAPH = (
    "The runtime keeps routed mixture-of-experts weights on solid-state "
    "storage and loads only the experts that the router selects for each "
    "token, so a large model runs inside a small declared memory budget. "
    "Attention state is held in a compressed key-value cache whose precision "
    "is chosen independently of the weight precision. Prefill processes the "
    "prompt in fixed chunks, while decode emits one token at a time and is "
    "bounded by memory bandwidth rather than arithmetic. "
)


PARAGRAPHS = 119


def build_prompt() -> str:
    """Stable English prose, independent of repository files. The default 119
    repetitions measure 10,141 prompt tokens. Length is a real variable here,
    not a detail: the sidecar exposes history functions at 0/4096/8192/12288,
    so a 6.1K-token prompt loads two of them while a 10.1K one loads three,
    and each carries its own ANE arena. Compare only equal lengths."""
    return (
        "Summarize the following technical description in 40 words.\n\n" + _PARAGRAPH * PARAGRAPHS
    )


def launch(quant: str, ane: bool, log_name: str) -> None:
    binary = ROOT / ".build/release/TinyTitanServer"
    cmd = server_command(binary, PORT, model=MODELS[quant], cache_mode="off")
    env = server_environment()
    env["TINYTITAN_PREFILL_ANE"] = "on" if ane else "off"
    log = open(benchmark_log_path(log_name), "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    g0._servers.append(proc)


MAX_TOKENS = 48


def generate(prompt: str, max_tokens: int) -> dict | None:
    payload = json.dumps(
        {
            "model": resolve_api_model(PORT),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "seed": 41,
            "max_completion_tokens": max_tokens,
        },
        separators=(",", ":"),
    ).encode()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=1800)
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
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
        "first_line": text.strip().splitlines()[0][:90] if text.strip() else "(empty)",
    }


def one_run(quant: str, ane: bool, prompt: str, tag: str) -> dict:
    arm = "ane" if ane else "gpu"
    log_name = f"aneab_{quant}_{arm}_{tag}.log"
    launch(quant, ane, log_name)
    if not g0.wait_ready(PORT):
        g0._terminate_all()
        raise SystemExit(f"[{quant}/{arm}] server not healthy")
    result = generate(prompt, MAX_TOKENS)
    g0._terminate_all()
    if result is None:
        raise SystemExit(f"[{quant}/{arm}] request failed")
    row: dict = {"arm": arm, **result}
    with open(benchmark_log_path(log_name), encoding="utf-8") as handle:
        text = handle.read()
    gen = g0.GENERATION_RE.search(text)
    if gen:
        row.update(
            prefill_s=float(gen.group(1)),
            decode_s=float(gen.group(2)),
            decode_tok_s=float(gen.group(3)),
        )
    row["fallback"] = "ane-prefill fallback" in text
    return row


ARM_METRICS = {
    "gpu": ("prefill_s", "decode_tok_s"),
    "ane": ("prefill_s", "decode_tok_s"),
}


def medians(rows):
    """(arms, {(arm, metric): (median, counted, total)}) for a finished sweep.

    The page and the verdict read the same table, so the status cannot be
    computed on numbers the report never showed.
    """
    arms = {name: [r for r in rows if r["arm"] == name] for name in ARM_METRICS}
    got = {
        (name, key): arm_metric(arms[name], key)
        for name, keys in ARM_METRICS.items()
        for key in keys
    }
    return arms, got


def verdict(rows):
    """(page, exit status) for a finished ANE-vs-GPU prefill A/B.

    This driver's claims are the ones its own report states: the ANE arm must
    clear the published 1.5x gate, each arm's greedy output must be stable
    within the arm, and an arm that logged the runtime's GPU-fallback line did
    not run on the ANE at all.

    0  both arms measured, no run fell back, both arms stable, and the gate held;
    1  it measured and a claim failed -- the gate, an unstable arm, or a fallback;
    2  it could not be compared -- an arm with no runs, an arm whose runs logged
       no prefill time, or an arm that streamed no content. A ratio against a
       zero or missing prefill time is not a speedup, and the old code died on
       `StatisticsError` or divided by zero instead of saying so.
    """
    lines: list[str] = []
    reasons: list[str] = []
    arms, got = medians(rows)
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

    gpu_prefill = got[("gpu", "prefill_s")][0]
    ane_prefill = got[("ane", "prefill_s")][0]
    gpu_decode = got[("gpu", "decode_tok_s")][0]
    ane_decode = got[("ane", "decode_tok_s")][0]
    if None in (gpu_prefill, ane_prefill, gpu_decode, ane_decode) or not ane_prefill:
        reasons.append("NOT MEASURED: a prefill time is missing, so no speedup is computable")
    else:
        speedup = gpu_prefill / ane_prefill
        lines.append(
            f"  GPU prefill median {gpu_prefill:8.2f} s   "
            f"runs {[r.get('prefill_s') for r in arms['gpu']]}"
        )
        lines.append(
            f"  ANE prefill median {ane_prefill:8.2f} s   "
            f"runs {[r.get('prefill_s') for r in arms['ane']]}"
        )
        lines.append(
            f"  SPEEDUP: {speedup:.2f}x   (gate is >=1.5x: {'PASS' if speedup >= 1.5 else 'FAIL'})"
        )
        if speedup < 1.5:
            reasons.append(f"gate failed: {speedup:.2f}x is below the published 1.5x")
        lines.append(f"  decode after prefill: GPU {gpu_decode:.3f} vs ANE {ane_decode:.3f} tok/s")

    for name in ("gpu", "ane"):
        sel = arms[name]
        if not sel:
            continue
        digests = sorted({r["sha256"] for r in sel})
        stable = len(digests) == 1
        lines.append(f"  {name} digests {digests} ({'stable' if stable else 'UNSTABLE'})")
        lines.append(f"    continuation: {sel[0].get('first_line')}")
        if not stable:
            reasons.append(f"{name} digests differ within the arm: {digests}")

    if any(r.get("fallback") for r in arms["ane"]):
        lines.append(
            "  WARNING: an ANE run logged a GPU fallback — the arms did not measure what they claim"
        )
        reasons.append(
            "the ane arm logged a GPU fallback, so its prefill time is a GPU time "
            "and the ratio above is not the ANE's"
        )

    hard = [reason for reason in reasons if reason.startswith("NOT MEASURED")]
    if hard:
        status = 2
    elif reasons:
        status = 1
    else:
        status = 0
    if reasons:
        lines.append("\n  VERDICT")
        lines.extend(f"    {reason}" for reason in reasons)
    lines.append(f"\n  prefill status {status}")
    return lines, status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant", choices=sorted(MODELS), default="4bit")
    parser.add_argument("--pairs", type=int, default=1)
    parser.add_argument("--allow-busy-gpu", action="store_true")
    # 48 tokens is enough to prove decode still runs, but far too short to
    # separate a one-time expert-cache re-warm after prefill from a real
    # steady-state rate change: at 8-bit the two differ by 100x in cost.
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=48,
        help="generated tokens per run; raise it to measure "
        "steady-state decode rather than the transient",
    )
    parser.add_argument(
        "--paragraphs",
        type=int,
        default=119,
        help="prompt length in paragraph repetitions "
        "(119 = 10,141 tokens; 72 = about 6.1K, the "
        "length the v4.5 result was qualified at)",
    )
    args = parser.parse_args()
    global MAX_TOKENS, PARAGRAPHS
    MAX_TOKENS = args.max_tokens
    PARAGRAPHS = args.paragraphs

    signal.signal(signal.SIGINT, g0._on_signal)
    signal.signal(signal.SIGTERM, g0._on_signal)
    g0.preflight(100 if args.allow_busy_gpu else 15)

    prompt = build_prompt()
    rows: list[dict] = []
    try:
        for ane in (False, True):
            print(f"[{args.quant}] warmup {'ane' if ane else 'gpu'}", flush=True)
            one_run(args.quant, ane, prompt, "warmup")
        for block in range(args.pairs):
            for ane in (False, True, True, False):
                row = one_run(args.quant, ane, prompt, f"b{block}_{len(rows)}")
                rows.append(row)
                print(
                    f"[{args.quant}] {row['arm']:<3} "
                    f"prefill {logged(row.get('prefill_s'), '7.2f', ' s')}  "
                    f"decode {logged(row.get('decode_tok_s'), '6.3f', ' tok/s')}  "
                    f"sha {row['sha256']}" + ("  [FALLBACK]" if row.get("fallback") else ""),
                    flush=True,
                )
    finally:
        g0._terminate_all()

    gpu_prefill_row = next((r for r in rows if r["arm"] == "gpu"), None)
    print("\n" + "=" * 70)
    print(
        f"ANE PREFILL A/B — {args.quant}, "
        f"{gpu_prefill_row.get('prompt_tokens') if gpu_prefill_row else 'no gpu run'} "
        "prompt tokens, greedy, cache off"
    )
    print("=" * 70)
    lines, status = verdict(rows)
    for line in lines:
        print(line)

    out = ROOT / f".build/benchmark-results/ane-prefill-ab-{args.quant}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "status": status}, fh, indent=2)
    print(f"\nwrote {out}")
    return status


if __name__ == "__main__":
    sys.exit(main())
