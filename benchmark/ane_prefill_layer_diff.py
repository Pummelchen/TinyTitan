#!/usr/bin/env python3
"""Layer-level comparison of the ANE prefill path against the GPU's, per arm.

`ane_prefill_correctness.py` compares generated *text*, which is the right
end-to-end question but a blunt instrument: on a prompt whose continuation is
nearly deterministic, a 32-token greedy comparison agrees even between the
folded mask and the causal-only one that is known to be wrong. Agreeing text is
therefore necessary, not sufficient.

This compares the runtime's own activation dump (`TINYTITAN_ACT_DUMP`) between
arms. **`prefill_logits` is the sensitive column**, and what the per-layer
ones actually measure was learned by measuring them:

- the dumps are live and post-layer — each differs from the previous layer by
  tens of percent, and on the dense 2B the ANE arm's `L3_after` differs from the
  GPU arm's by 0.94 %;
- changing the mask materially does move them: on Qwen 3.8, lowering the QSA
  budget from 2,048 to 8 moves `L3_after` by 10.2 % and the logits by 69 %;
- but the folded mask and the causal-only mask are **bit-identical at
  `L3_after`** on the 3.8 prompt while their logits differ by 13.6 %. That is not
  a dump artifact: at the first full-attention layer the blocks this indexer drops
  contribute below fp16 resolution, so the selection and the causal mask compute
  the same output there, and the divergence appears at later attention layers —
  which the per-layer columns cannot see (`dumpLayerLimit` is 3).

So the per-layer columns are a useful control (the gated-DeltaNet layers must
match exactly) but the logits are what separates the arms.

`--max-new 1` is deliberate and irrelevant to the comparison: the activations
come from the *prefill* chunk, so nothing about the decode length enters it.

  python3 benchmark/ane_prefill_layer_diff.py \
      --models qwen3.8-flash-next_125B_A6B_4Bit --label v5.6-ane-38 --record
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ane_prefill_ab_matrix as ab

ROOT = ab.ROOT
MODELS_DIR = ab.MODELS_DIR
CLI = ab.CLI
RESULTS = ab.RESULTS
ARMS = ("off", "on", "causal")


def sparse_indexed(model: str) -> bool:
    manifest = json.loads((MODELS_DIR / model / "manifest.json").read_text())
    return int((manifest.get("arch") or {}).get("indexerBudget") or 0) > 0


def run_arm(
    model: str, arm: str, characters: int, chunk: int, dump_root: pathlib.Path, timeout: int = 3600
) -> dict:
    env = os.environ.copy()
    env["TINYTITAN_PREFILL_ANE"] = "off" if arm == "off" else "on"
    if arm == "causal":
        env["TINYTITAN_ANE_MASK"] = "causal"
    env["TINYTITAN_ACT_DUMP"] = str(dump_root / arm)
    command = [
        str(CLI),
        "--model",
        str(MODELS_DIR / model),
        "--prompt",
        ab.prompt(characters),
        "--max-new",
        "1",
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
    result["used_ane"] = ab.FALLBACK_MARKER not in proc.stderr
    result["causal_warning"] = "TINYTITAN_ANE_MASK=causal" in proc.stderr
    footer = ab.parse_footer(proc.stderr)
    if footer is not None:
        result["prefill_seconds"] = footer["prefill_seconds"]
        result["prefill_tokens"] = footer["prefill_tokens"]
    return result


def dumps(root: pathlib.Path) -> dict[str, np.ndarray]:
    """Every activation this arm wrote, keyed `pos<N>/<name>`."""
    out: dict[str, np.ndarray] = {}
    for path in sorted(root.rglob("*.f16")):
        key = f"{path.parent.name}/{path.stem}"
        out[key] = np.fromfile(path, dtype="<f2").astype(np.float32)
    return out


def compare(reference: dict[str, np.ndarray], arm: dict[str, np.ndarray]) -> dict[str, dict]:
    """Mean relative error per activation, over what both arms wrote."""
    out: dict[str, dict] = {}
    for key in sorted(set(reference) & set(arm)):
        a, b = reference[key], arm[key]
        if a.shape != b.shape or a.size == 0:
            out[key] = {"error": f"shape {a.shape} vs {b.shape}"}
            continue
        base = float(np.abs(a).mean())
        out[key] = {
            "mean_relative_percent": float(np.abs(a - b).mean() / base * 100.0) if base else None,
            "max_absolute": float(np.abs(a - b).max()),
        }
    return out


def summarize(
    model: str, arms: dict[str, dict], arm_dumps: dict[str, dict[str, np.ndarray]]
) -> dict:
    failed = [run for run in arms.values() if "error" in run]
    if failed:
        return {"model": model, "error": failed[0]["error"]}
    for name in ("on", "causal"):
        if name in arms and not arms[name]["used_ane"]:
            return {"model": model, "error": f"the {name} arm fell back to the GPU"}
    out: dict = {"model": model, "arms": arms, "comparisons": {}}
    for name in ("on", "causal"):
        if name in arm_dumps and "off" in arm_dumps:
            out["comparisons"][name] = compare(arm_dumps["off"], arm_dumps[name])
    # The fold-specific signal: the two ANE arms differ only in whether the
    # selection is folded in, so their difference is exactly what the fold buys.
    if "on" in arm_dumps and "causal" in arm_dumps:
        out["comparisons"]["causal_vs_on"] = compare(arm_dumps["on"], arm_dumps["causal"])
    return out


def format_row(row: dict) -> str:
    if "error" in row:
        return f"{row['model']:<44} {row['error']}"
    parts = []
    for name in ("on", "causal", "causal_vs_on"):
        diffs = row["comparisons"].get(name)
        if diffs is None:
            continue
        logits = diffs.get("pos0/prefill_logits", {}).get("mean_relative_percent")
        control = max(
            (diffs.get(f"pos0/L{layer}_after", {}).get("mean_relative_percent") or 0.0)
            for layer in (0, 1, 2)
        )
        if logits is None:
            parts.append(f"{name}: no logits dump")
        else:
            parts.append(f"{name} logits {logits:.2f}% (linear ctrl {control:.3f}%)")
    return f"{row['model']:<44} " + "  ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument(
        "--attention-layer",
        type=int,
        default=3,
        help="the first full-attention layer the dump covers (the one the fold acts on)",
    )
    parser.add_argument("--prompt-characters", type=int, default=ab.PROMPT_CHARACTERS)
    parser.add_argument("--prefill-chunk", type=int, default=ab.PREFILL_CHUNK)
    parser.add_argument("--label", default=None)
    parser.add_argument("--record", action="store_true")
    parser.add_argument(
        "--keep-dumps", action="store_true", help="leave the dump directories behind for inspection"
    )
    args = parser.parse_args()

    attention_layers = [args.attention_layer]
    results = []
    for name in args.models:
        root = pathlib.Path(tempfile.mkdtemp(prefix="ane-layer-diff-"))
        arms: dict[str, dict] = {}
        arm_dumps: dict[str, dict[str, np.ndarray]] = {}
        wanted = list(ARMS) if sparse_indexed(name) else ["off", "on"]
        for arm in wanted:
            run = run_arm(name, arm, args.prompt_characters, args.prefill_chunk, root)
            arms[arm] = run
            if "error" in run:
                break
            arm_dumps[arm] = dumps(root / arm)
        row = summarize(name, arms, arm_dumps)
        row["dump_activations"] = {arm: sorted(d) for arm, d in arm_dumps.items()}
        print(format_row(row), flush=True)
        results.append(row)
        if not args.keep_dumps:
            shutil.rmtree(root, ignore_errors=True)
        else:
            print(f"  dumps kept at {root}", flush=True)
        if args.record:
            RESULTS.mkdir(parents=True, exist_ok=True)
            label = args.label or datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y%m%dT%H%M%S"
            )
            path = RESULTS / f"ane-layer-diff-{label}.json"
            record = {
                "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "prompt_characters": args.prompt_characters,
                "prefill_chunk": args.prefill_chunk,
                "attention_layers": attention_layers,
                "results": results,
            }
            path.write_text(json.dumps(record, indent=2) + "\n")
            print(f"  wrote {path.relative_to(ROOT)}", flush=True)
    return 0 if all("error" not in row for row in results) else 1


if __name__ == "__main__":
    sys.exit(main())
