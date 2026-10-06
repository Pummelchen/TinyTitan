#!/usr/bin/env python3
"""Long-prompt correctness comparison for the ANE prefill path.

The A/B matrix answers *how much faster*; this answers *is it the same model*.
Both arms run the same long prompt greedily for `--max-new` tokens with one
variable, `TINYTITAN_PREFILL_ANE`, and the responses are compared as text: how
many leading characters agree, and where the first divergence is.

`--max-new 1` cannot answer this. A single greedy token usually agrees even when
the attention behind it is wrong, which is why the speed sweep's digests are a
determinism check and never evidence of correctness. A long continuation is what
makes a wrong mask visible.

A model whose full-attention layers select keys with a QSA indexer gets a third
arm for free: `causal`, which runs the ANE with the causal-only mask the fold
exists to replace (`TINYTITAN_ANE_MASK=causal`). It is the negative control, and
it is expected to diverge from the GPU arm while `on` does not — otherwise this
check is not measuring what it claims to.

  python3 benchmark/ane_prefill_correctness.py \
      --models qwen3.8-flash-next_125B_A6B_4Bit \
      --max-new 32 --label v5.6-ane-38 --record
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ane_prefill_ab_matrix as ab

ROOT = ab.ROOT
MODELS_DIR = ab.MODELS_DIR
CLI = ab.CLI
RESULTS = ab.RESULTS
FALLBACK_MARKER = ab.FALLBACK_MARKER
# The runtime prints this the first time it feeds the causal-only mask, so the
# control arm cannot be mistaken for the shipped path.
CAUSAL_MARKER = "TINYTITAN_ANE_MASK=causal"

ARMS = ("off", "on", "causal")


def sparse_indexed(model: str) -> bool:
    """Whether this install's full-attention layers select keys themselves."""
    manifest = json.loads((MODELS_DIR / model / "manifest.json").read_text())
    return int((manifest.get("arch") or {}).get("indexerBudget") or 0) > 0


def run_arm(
    model: str, arm: str, characters: int, chunk: int, max_new: int, timeout: int = 3600
) -> dict:
    env = os.environ.copy()
    env["TINYTITAN_PREFILL_ANE"] = "off" if arm == "off" else "on"
    if arm == "causal":
        env["TINYTITAN_ANE_MASK"] = "causal"
    command = [
        str(CLI),
        "--model",
        str(MODELS_DIR / model),
        "--prompt",
        ab.prompt(characters),
        "--max-new",
        str(max_new),
        "--temperature",
        "0",
        "--prefill-chunk",
        str(chunk),
    ]
    proc = subprocess.run(
        command, capture_output=True, text=True, env=env, cwd=ROOT, timeout=timeout, check=False
    )
    result: dict = {"arm": arm, "exit": proc.returncode}
    if proc.returncode != 0:
        tail = [line.strip() for line in proc.stderr.strip().splitlines() if line.strip()]
        result["error"] = " / ".join(tail[-3:])[:400] or f"exit {proc.returncode}"
        return result
    result["response"] = proc.stdout.strip()
    footer = ab.parse_footer(proc.stderr)
    if footer is not None:
        result["prefill_tokens"] = footer["prefill_tokens"]
        result["prefill_seconds"] = footer["prefill_seconds"]
        result["decode_tokens"] = footer["decode_tokens"]
        result["finish"] = footer["finish"]
    result["used_ane"] = FALLBACK_MARKER not in proc.stderr
    result["causal_warning"] = CAUSAL_MARKER in proc.stderr
    if not result["used_ane"]:
        result["fallback_reason"] = next(
            (line.strip() for line in proc.stderr.splitlines() if FALLBACK_MARKER in line),
            FALLBACK_MARKER,
        )
    return result


def common_prefix(a: str, b: str) -> int:
    limit = min(len(a), len(b))
    index = 0
    while index < limit and a[index] == b[index]:
        index += 1
    return index


def divergence(a: str, b: str, at: int) -> dict:
    """The two snippets around the first disagreement, for the record."""
    low = max(0, at - 30)
    return {"at_character": at, "off": a[low : at + 30], "arm": b[low : at + 30]}


def compare(reference: dict, arm: dict) -> dict:
    a, b = reference["response"], arm["response"]
    at = common_prefix(a, b)
    return {
        "identical": a == b,
        "common_prefix_characters": at,
        "characters": len(a),
        "first_divergence": None if a == b else divergence(a, b, at),
    }


def summarize(model: str, arms: dict[str, dict], max_new: int) -> dict:
    out: dict = {
        "model": model,
        "max_new_tokens": max_new,
        "measured": {
            name: {
                "decode_tokens": run.get("decode_tokens"),
                "prefill_seconds": run.get("prefill_seconds"),
                "finish": run.get("finish"),
            }
            for name, run in arms.items()
        },
        "comparisons": {},
    }
    # An arm that failed has no response to compare; report its error rather
    # than dying on the missing key.
    failed = [run for run in arms.values() if "error" in run]
    if failed:
        out["error"] = failed[0]["error"]
        return out
    for name, run in arms.items():
        if name != "off":
            out["comparisons"][name] = compare(arms["off"], run)

    # A comparison against an arm that fell back is a GPU-vs-GPU comparison
    # wearing an ANE label, which is how a model that never touched the Neural
    # Engine comes to look verified.
    for name in ("on", "causal"):
        if name in arms and "error" not in arms[name] and not arms[name]["used_ane"]:
            out["error"] = (
                f"the {name} arm fell back to the GPU: {arms[name].get('fallback_reason')}"
            )
            return out

    on = out["comparisons"].get("on")
    causal = out["comparisons"].get("causal")
    if on is not None and causal is not None:
        if not arms["causal"].get("causal_warning"):
            out["error"] = (
                "the causal control did not announce itself; it may have run the folded path"
            )
            return out
        if not arms["causal"]["used_ane"]:
            out["error"] = "the causal control did not reach the ANE"
            return out
        if causal["common_prefix_characters"] >= on["common_prefix_characters"]:
            out["warning"] = (
                f"the causal-only control ({causal['common_prefix_characters']} "
                f"chars) agrees with the GPU at least as far as the folded arm "
                f"({on['common_prefix_characters']} chars); the fold may not be "
                f"load-bearing on this prompt"
            )
    return out


def format_row(row: dict) -> str:
    if "error" in row:
        return f"{row['model']:<44} {'-':>9} {'-':>9}  {row['error']}"
    on = row["comparisons"].get("on", {})
    causal = row["comparisons"].get("causal")
    note = ""
    if "warning" in row:
        note = row["warning"]
    elif on.get("identical"):
        note = "identical to the GPU arm"
    return (
        f"{row['model']:<44} "
        f"{str(on.get('identical')):>9} "
        f"{on.get('common_prefix_characters', 0):>9}  "
        f"causal {causal['common_prefix_characters'] if causal else '-'}"
        f"  {note}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", required=True, help="install directory names under models/"
    )
    parser.add_argument(
        "--max-new",
        type=int,
        default=32,
        help="greedy tokens to generate per arm (default 32). "
        "One token cannot tell a wrong mask from a right "
        "one; the comparison needs a continuation",
    )
    parser.add_argument("--prompt-characters", type=int, default=ab.PROMPT_CHARACTERS)
    parser.add_argument("--prefill-chunk", type=int, default=ab.PREFILL_CHUNK)
    parser.add_argument("--label", default=None)
    parser.add_argument(
        "--record",
        action="store_true",
        help="write benchmark/ane-prefill/<label>.json after "
        "every model, so a held run keeps its rows",
    )
    args = parser.parse_args()

    results = []
    header = f"{'model':<44} {'identical':>9} {'prefix':>9}  causal divergence"
    print(header, flush=True)
    failed = False
    for name in args.models:
        arms: dict[str, dict] = {}
        wanted = list(ARMS) if sparse_indexed(name) else ["off", "on"]
        for arm in wanted:
            run = run_arm(name, arm, args.prompt_characters, args.prefill_chunk, args.max_new)
            arms[arm] = run
            if "error" in run:
                break
        row = summarize(name, arms, args.max_new)
        row["arms"] = {
            arm: {k: v for k, v in run.items() if k != "response"} for arm, run in arms.items()
        }
        row["responses"] = {arm: run.get("response", "") for arm, run in arms.items()}
        print(format_row(row), flush=True)
        if "error" in row:
            failed = True
        results.append(row)
        if args.record:
            RESULTS.mkdir(parents=True, exist_ok=True)
            label = args.label or datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y%m%dT%H%M%S"
            )
            path = RESULTS / f"ane-correctness-{label}.json"
            record = {
                "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "prompt_characters": args.prompt_characters,
                "prefill_chunk": args.prefill_chunk,
                "max_new_tokens": args.max_new,
                "results": results,
            }
            path.write_text(json.dumps(record, indent=2) + "\n")
            print(f"  wrote {path.relative_to(ROOT)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
