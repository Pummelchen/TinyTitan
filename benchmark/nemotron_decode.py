#!/usr/bin/env python3
"""Measure Nemotron 3.5 Bolt decode throughput on both backends and both widths.

The two numbers this records are the ones the README carries: steady-state
decode tokens per second through the Metal GEMV kernels (`nemotrongpu`) and
through the C CPU kernels (`nemotron`), for the 4-bit and 8-bit installs.

Each arm loads the install in its own process, runs a fixed greedy continuation,
and is repeated `--runs` times; the table reports every run and the median.
Prefill is token-by-token, so TTFT is printed too but is not the headline.

    benchmark/nemotron_decode.py \
        --binary .build/release/TinyTitanBench \
        --install4 models/nemotron-3.5-bolt_74B_A9B_4Bit \
        --install8 models/nemotron-3.5-bolt_74B_A9B_8Bit \
        --prompt "The capital of France is" --limit 4 --runs 1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

DECODE = re.compile(r"decode (\d+) tokens: ([\d.]+)s \(([\d.]+) tok/s\)")
PREFILL = re.compile(r"prompt (\d+) tokens: ([\d.]+)s \(([\d.]+) tok/s\)")


def run_arm(
    binary: str, install: str, prompt: str, limit: int, gpu: bool, ignore_eos: bool
) -> dict:
    command = [binary, "nemotrongraph" if gpu else "nemotron", install, prompt, str(limit)]
    environment = dict(os.environ)
    if ignore_eos:
        environment["TINYTITAN_NEMOTRON_IGNORE_EOS"] = "1"
    result = subprocess.run(command, capture_output=True, text=True, check=False, env=environment)
    if result.returncode != 0:
        raise SystemExit(f"{command[1]} failed ({result.returncode}):\n{result.stderr[-4000:]}")
    decode = DECODE.search(result.stdout)
    prefill = PREFILL.search(result.stdout)
    if not decode:
        raise SystemExit(f"no decode line in:\n{result.stdout[-4000:]}")
    return {
        "backend": "gpu" if gpu else "cpu",
        "install": install,
        "prompt_tokens": int(prefill.group(1)) if prefill else None,
        "prefill_seconds": float(prefill.group(2)) if prefill else None,
        "decode_tokens": int(decode.group(1)),
        "decode_seconds": float(decode.group(2)),
        "decode_tps": float(decode.group(3)),
        "output": result.stdout.strip().splitlines()[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default=".build/release/TinyTitanBench")
    parser.add_argument("--install4", required=True)
    parser.add_argument("--install8", required=True)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("benchmark/nemotron_decode.json"))
    args = parser.parse_args()

    arms = []
    for label, install in (("4-bit", args.install4), ("8-bit", args.install8)):
        for gpu in (True, False):
            runs = [
                run_arm(args.binary, install, args.prompt, args.limit, gpu, args.ignore_eos)
                for _ in range(args.runs)
            ]
            arms.append({"quant": label, "runs": runs})

    table = []
    for arm in arms:
        rates = [run["decode_tps"] for run in arm["runs"]]
        table.append(
            {
                "quant": arm["quant"],
                "backend": arm["runs"][0]["backend"],
                "decode_tps": statistics.median(rates),
                "runs": rates,
                "prefill_seconds": arm["runs"][0]["prefill_seconds"],
                "output": arm["runs"][0]["output"],
            }
        )

    print("| Quant | Backend | decode tok/s (median) | runs | prefill s |")
    print("| --- | --- | ---: | --- | ---: |")
    for row in table:
        values = " / ".join(f"{value:.4f}" for value in row["runs"])
        prefill = f"{row['prefill_seconds']:.2f}" if row["prefill_seconds"] is not None else "NA"
        print(
            f"| {row['quant']} | {row['backend']} | **{row['decode_tps']:.4f}** "
            f"| {values} | {prefill} |"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"prompt": args.prompt, "limit": args.limit, "table": table}, indent=1)
    )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
