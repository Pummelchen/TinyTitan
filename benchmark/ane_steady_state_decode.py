#!/usr/bin/env python3
"""Steady-state decode after the ANE handover transient (TT-007).

Every earlier ANE decode figure is a ~60-token window, because the pinned
qualification prompt hits end-of-turn early and `--max-new` never binds. That
window contains the one-time handover re-warm, so those figures are an *upper
bound* on what ANE prefill costs decode -- and the remaining cost is exactly
what TT-005 is about.

This measures the marginal rate instead of a window. The same arm is run at two
generation lengths on the same prompt, so

    steady-state tok/s = (new_big - new_small) / (decode_big - decode_small)

cancels every one-time cost both runs pay: the expert-cache re-warm, the first
token, the wiring walk. A length of 0 would be ideal; `--small` stays well past
the tokenizer/first-token effects and well inside the big run.

Arms alternate `gpu ane ane gpu` and the two lengths alternate inside each arm,
so thermal and page-cache drift land on both arms rather than on whichever ran
second. The prompt is the qualification body plus an instruction to keep
writing, so `--max-new` binds; a run that stopped early is reported and
excluded, because differencing a stopped run is meaningless. The default 140
paragraphs measure ~12,500 tokens, three ANE chunks, which is the interesting
end of the range: a longer ANE prefill leaves Core ML holding more arena space
(the 10.1K-token condition measured the largest gap). Use `--paragraphs 47`
(~4,230 tokens, one chunk) for the cheapest eligible prompt.

  python3 benchmark/ane_steady_state_decode.py \
      --model models/qwen-agentworld_35B_A3B_4Bit --small 64 --big 448 --record
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys
import tempfile

from tinytitan_profile import pgrep_answer

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLI = ROOT / ".build/release/TinyTitanCLI"
RESULTS = ROOT / "benchmark/ane-prefill"

# The same pinned body the qualification harness uses, so a number here is
# comparable with the figures it produced.
PARAGRAPH = (
    "The runtime keeps routed mixture-of-experts weights on solid-state "
    "storage and loads only the experts that the router selects for each "
    "token, so a large model runs inside a small declared memory budget. "
    "Attention state is held in a compressed key-value cache whose precision "
    "is chosen independently of the weight precision. Prefill processes the "
    "prompt in fixed chunks, while decode emits one token at a time and is "
    "bounded by memory bandwidth rather than arithmetic. "
)

# The qualification prompt asks for a 40-word summary and therefore stops at
# ~60 tokens. This asks for a long continuation and forbids stopping, which is
# what makes `--max-new` (rather than the model) the thing that ends a run.
INSTRUCTION = (
    "\n\nContinue the technical description above. Write at least 700 more "
    "words in the same style, without summarizing and without stopping early. "
    "Start immediately with the next sentence.\n"
)

FOOTER = re.compile(
    r"\[stop=(\S+) prefill=(\d+)tok/([\d.]+)s new=(\d+)tok decode=([\d.]+)s "
    r"tok/s=([\d.]+)\]"
)
WIRE = re.compile(
    r"\[wire\] setExpertCachePinned\((\w+)\) ([\d.]+) ms "
    r"early=(\w+) walked=(\d+)"
)


def build_prompt(paragraphs: int) -> str:
    return PARAGRAPH * paragraphs + INSTRUCTION


def run_cli(
    model: pathlib.Path,
    prompt: str,
    max_new: int,
    ane: bool,
    messages_file: pathlib.Path,
    wire_trace: bool = False,
) -> dict:
    """One arm at one length: a fresh process, so each run pays its own transient."""
    env = dict(os.environ)
    env["TINYTITAN_PREFILL_ANE"] = "on" if ane else "off"
    if wire_trace:
        env["TINYTITAN_WIRE_TRACE"] = "1"
    proc = subprocess.run(
        [
            str(CLI),
            "--model",
            str(model),
            "--messages-file",
            str(messages_file),
            "--max-new",
            str(max_new),
            "--temperature",
            "0",
        ],
        env=env,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )
    match = FOOTER.search(proc.stderr)
    if match is None:
        return {
            "arm": "ane" if ane else "gpu",
            "max_new": max_new,
            "failed": (proc.stderr or proc.stdout)[-400:],
        }
    stop, prompt_tokens, prefill_s, new, decode_s, rate = match.groups()
    wire = [
        {
            "pinned": bool(m.group(1) == "true"),
            "ms": float(m.group(2)),
            "early": m.group(3) == "true",
            "walked": int(m.group(4)),
        }
        for m in WIRE.finditer(proc.stderr)
    ]
    return {
        "arm": "ane" if ane else "gpu",
        "max_new": max_new,
        "stop": stop,
        "prompt_tokens": int(prompt_tokens),
        "prefill_s": float(prefill_s),
        "new_tokens": int(new),
        "decode_s": float(decode_s),
        "decode_tok_s": float(rate),
        "wire": wire,
        # A stopped run cannot be differenced: it generated fewer tokens than
        # asked, so its decode seconds measure a different amount of work.
        "complete": stop == "maxTokens" and int(new) == max_new,
    }


def median(rows: list[dict], key: str) -> float | None:
    values = [r[key] for r in rows if key in r and not r.get("failed")]
    return statistics.median(values) if values else None


def steady_state(rows: list[dict], arm: str, small: int, big: int) -> float | None:
    """Marginal tok/s over [small, big] for one arm, pairing each big run with
    the nearest complete small run of the same arm."""
    smalls = [r for r in rows if r["arm"] == arm and r["max_new"] == small and r.get("complete")]
    bigs = [r for r in rows if r["arm"] == arm and r["max_new"] == big and r.get("complete")]
    if not smalls or not bigs:
        return None
    small_med = statistics.median([r["decode_s"] for r in smalls])
    big_med = statistics.median([r["decode_s"] for r in bigs])
    if big_med <= small_med:
        return None
    return (big - small) / (big_med - small_med)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="models/qwen-agentworld_35B_A3B_4Bit",
        help="an installed model directory with an ANE sidecar",
    )
    parser.add_argument(
        "--small", type=int, default=64, help="short generation, past first-token effects"
    )
    parser.add_argument(
        "--big", type=int, default=448, help="long generation, the differencing upper end"
    )
    parser.add_argument(
        "--paragraphs",
        type=int,
        default=140,
        help="prompt body; 140 is ~12,500 tokens (three 4,096-token "
        "ANE chunks), 47 is ~4,230 (the cheapest eligible prompt)",
    )
    parser.add_argument(
        "--reps", type=int, default=1, help="gpu/ane/ane/gpu blocks; 1 gives two runs per arm"
    )
    parser.add_argument(
        "--wire-trace",
        action="store_true",
        help="TINYTITAN_WIRE_TRACE=1, to record the pin/unpin calls",
    )
    parser.add_argument("--label", default="tt007")
    parser.add_argument(
        "--record", action="store_true", help="write rows to benchmark/ane-prefill/"
    )
    args = parser.parse_args()

    model = (ROOT / args.model).resolve()
    if not (model / "verified-install.json").exists():
        raise SystemExit(f"not an installed model: {model}")
    if not (model / "ane_prefill").is_dir():
        raise SystemExit(f"no ANE sidecar under {model}")
    verdict, lines = pgrep_answer(["-fl", "TinyTitanCLI|TinyTitanServer|mlx_lm|mlx-lm"])
    if verdict != "clear":
        reason = (
            "a model process is already running"
            if verdict == "busy"
            else "the model-process guard could not answer what is running"
        )
        raise SystemExit(f"{reason}:\n" + "\n".join(f"  {line}" for line in lines))

    prompt = build_prompt(args.paragraphs)
    # `--messages-file` rather than `--prompt`: the chat template is what turns
    # this into "keep writing" for an instruct model, and it is the path the
    # qualification harness uses. Written once and reused by every run.
    messages = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", prefix="tt007-messages-", suffix=".json", delete=False
    )
    json.dump([{"role": "user", "content": prompt}], messages)
    messages.close()
    messages_file = pathlib.Path(messages.name)

    rows: list[dict] = []
    print(
        f"[tt007] {model.name}: prompt {len(prompt)} chars, "
        f"generations {args.small} and {args.big}" + (", wire trace" if args.wire_trace else ""),
        flush=True,
    )
    try:
        for _rep in range(args.reps):
            for ane in (False, True, True, False):
                for length in (args.small, args.big):
                    row = run_cli(model, prompt, length, ane, messages_file, args.wire_trace)
                    rows.append(row)
                    if row.get("failed"):
                        print(f"[{row['arm']:<3} {length:>4}] FAILED: {row['failed']}", flush=True)
                        continue
                    wire = row.get("wire") or []
                    wire_note = ""
                    if wire:
                        releases = sum(1 for w in wire if not w["pinned"])
                        pin_ms = sum(w["ms"] for w in wire if w["pinned"])
                        wire_note = (
                            f"  wire: {len(wire)} calls, {releases} release, "
                            f"{pin_ms:6.1f} ms pinned"
                        )
                    print(
                        f"[{row['arm']:<3} {length:>4}] prefill "
                        f"{row['prefill_s']:7.2f} s  decode {row['decode_s']:6.2f} s  "
                        f"new {row['new_tokens']:4d}  {row['decode_tok_s']:6.2f} tok/s  "
                        f"stop={row['stop']}"
                        + ("" if row["complete"] else "  [INCOMPLETE]")
                        + wire_note,
                        flush=True,
                    )
    finally:
        messages_file.unlink(missing_ok=True)

    gpu_ss = steady_state(rows, "gpu", args.small, args.big)
    ane_ss = steady_state(rows, "ane", args.small, args.big)
    gpu_prefill = median([r for r in rows if r["arm"] == "gpu"], "prefill_s")
    ane_prefill = median([r for r in rows if r["arm"] == "ane"], "prefill_s")

    print("\n" + "=" * 72)
    print(f"STEADY-STATE DECODE PAST THE ANE HANDOVER — {model.name}")
    print(
        f"  prompt {rows[0].get('prompt_tokens', '?')} tokens, greedy, "
        f"differenced over [{args.small}, {args.big}] generated tokens"
    )
    print("=" * 72)
    if gpu_prefill and ane_prefill:
        print(
            f"  prefill   GPU {gpu_prefill:7.2f} s   ANE {ane_prefill:7.2f} s   "
            f"speedup {gpu_prefill / ane_prefill:.2f}x"
        )
    for label, value in (("GPU", gpu_ss), ("ANE", ane_ss)):
        print(
            f"  decode    {label} steady state "
            + (f"{value:6.2f} tok/s" if value else "unavailable")
        )
    if gpu_ss and ane_ss:
        gap = (ane_ss - gpu_ss) / gpu_ss * 100.0
        print(f"  ANE vs GPU steady state: {gap:+.1f}% ({ane_ss:.2f} vs {gpu_ss:.2f} tok/s)")
        for length in (args.small, args.big):
            g = median(
                [r for r in rows if r["arm"] == "gpu" and r["max_new"] == length], "decode_tok_s"
            )
            a = median(
                [r for r in rows if r["arm"] == "ane" and r["max_new"] == length], "decode_tok_s"
            )
            if g and a:
                print(
                    f"    window up to {length:>4} tokens: "
                    f"ANE vs GPU {(a - g) / g * 100.0:+.1f}% "
                    f"({a:.2f} vs {g:.2f} tok/s) — contains the transient"
                )

    if args.record:
        RESULTS.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M")
        out = RESULTS / f"steady-state-{args.label}-{stamp}.json"
        out.write_text(
            json.dumps(
                {
                    "model": model.name,
                    "small": args.small,
                    "big": args.big,
                    "paragraphs": args.paragraphs,
                    "wire_trace": args.wire_trace,
                    "gpu_steady_tok_s": gpu_ss,
                    "ane_steady_tok_s": ane_ss,
                    "gpu_prefill_s": gpu_prefill,
                    "ane_prefill_s": ane_prefill,
                    "rows": rows,
                },
                indent=2,
            )
        )
        print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
