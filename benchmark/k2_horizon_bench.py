#!/usr/bin/env python3.13
"""Run the K2-Horizon CPU benchmark for the 4-bit and 8-bit snapshots.

Wraps `TinyTitanBench k2 <snapshot> <prompt> <n>` and records the two numbers
it prints. A run is repeated so the first (cold page cache) pass can be seen
separately from the warm one.

    python3 benchmark/k2_horizon_bench.py \
        --snapshot-4bit .build/k2-horizon-affine-4bit \
        --snapshot-8bit .build/k2-horizon-affine-8bit \
        --out .build/k2-bench.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PROMPT = "The capital of France is"
PREFILL = re.compile(r"prompt (\d+) tokens: ([0-9.]+)s \(([0-9.]+) tok/s\)")
DECODE = re.compile(r"decode (\d+) tokens: ([0-9.]+)s \(([0-9.]+) tok/s\)")


def run(binary: Path, snapshot: Path, prompt: str, tokens: int) -> dict:
    result = subprocess.run(
        [str(binary), "k2", str(snapshot), prompt, str(tokens)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"{snapshot}: bench exited {result.returncode}\n{result.stderr.strip()}")
    prefill = PREFILL.search(result.stdout)
    decode = DECODE.search(result.stdout)
    if prefill is None or decode is None:
        raise SystemExit(f"{snapshot}: could not parse\n{result.stdout}")
    return {
        "prompt_tokens": int(prefill.group(1)),
        "prefill_seconds": float(prefill.group(2)),
        "prefill_tokens_per_second": float(prefill.group(3)),
        "decode_tokens": int(decode.group(1)),
        "decode_seconds": float(decode.group(2)),
        "decode_tokens_per_second": float(decode.group(3)),
        "output": result.stdout.strip().splitlines()[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path(".build/release/TinyTitanBench"))
    parser.add_argument("--snapshot-4bit", type=Path)
    parser.add_argument("--snapshot-8bit", type=Path)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = {"prompt": args.prompt, "tokens": args.tokens, "runs": {}}
    for width, snapshot in (("4-bit", args.snapshot_4bit), ("8-bit", args.snapshot_8bit)):
        if snapshot is None:
            continue
        runs = []
        for _ in range(args.repeats):
            run_result = run(args.binary, snapshot, args.prompt, args.tokens)
            print(f"{width}: {json.dumps(run_result)}", flush=True)
            runs.append(run_result)
        report["runs"][width] = runs
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=1))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
