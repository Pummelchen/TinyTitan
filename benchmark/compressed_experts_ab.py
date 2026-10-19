#!/usr/bin/env python3
"""Paired A/B: raw expert layers versus LZFSE-compressed ones.

The compressed format saves ~8.3% of the routed-expert bytes (17 GiB -> 15.5 GiB
on Ornith 4-bit) and costs a decompression per expert on the prefetch thread.
Whether that is a win is a wall-clock question, so the two arms are the *same
install* in two formats, run alternately in one session:

    raw        models/<install>                     layer_NN.bin as packed
    compressed models/<install>-compressed          layer_NN.bin = LZFSE frames,
                                                    layer_NN.idx = the seek index

Each measurement is a fresh `TinyTitanCLI` process with the same prompt,
`--temperature 0` and a fixed `--max-new`, so decode starts from an empty expert
cache in both arms. The installs share `model_weights.bin` through a hard link,
so the page-cache state of the resident weights is the same on both sides -- and
the arms are alternated within a round so a machine that slowly warms up cannot
favour whichever arm always ran second.

The number the verdict is built on is the CLI's own footer,
`decode=<seconds> tok/s=<rate>`, not a recomputed one. Responses must also be
identical: LZFSE is lossless, so a different answer is a correctness finding,
not a speed result.

Exit status: 0 both arms measured and agreed, 1 they measured but disagreed or
regressed, 2 the A/B could not run (missing install, busy machine, no footer).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CLI = ROOT / ".build/release/TinyTitanCLI"
PROMPT = (
    "Write a detailed explanation of how a bicycle works, covering the frame, "
    "the wheels, the drivetrain and the brakes."
)
FOOTER = re.compile(
    r"\[stop=(\S+) prefill=(\d+)tok/([\d.]+)s new=(\d+)tok decode=([\d.]+)s tok/s=([\d.]+)\]"
)
BUSY = re.compile(r"TinyTitanServer|TinyTitanCLI|swiftpm-testing-helper|mlx_lm|mlx-lm")


def preflight(raw: pathlib.Path, compressed: pathlib.Path) -> None:
    if not CLI.exists():
        raise RuntimeError(f"{CLI} is missing; run: swift build -c release")
    for path in (raw, compressed):
        if not (path / "manifest.json").is_file():
            raise RuntimeError(f"{path} is not an installed model")
        if not (path / "packed_experts").is_dir():
            raise RuntimeError(f"{path} has no packed_experts/")
    if not (compressed / "packed_experts" / "layer_00.idx").is_file():
        raise RuntimeError(
            f"{compressed} has no .idx sidecars; convert it with "
            "TinyTitanRepack --compress-experts first"
        )
    busy = subprocess.run(
        ["pgrep", "-fl", BUSY.pattern], check=False, capture_output=True, text=True
    ).stdout.strip()
    if busy:
        raise RuntimeError(f"a model process is already running:\n{busy}")


def measure(model: pathlib.Path, max_new: int, timeout: int) -> dict:
    command = [
        str(CLI),
        "--model",
        str(model),
        "--prompt",
        PROMPT,
        "--max-new",
        str(max_new),
        "--temperature",
        "0",
    ]
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=timeout
    )
    if completed.returncode != 0:
        return {"error": f"cli exit {completed.returncode}", "stderr": completed.stderr[-600:]}
    match = FOOTER.search(completed.stderr)
    if not match:
        return {"error": "no [stop=...] footer", "stderr": completed.stderr[-600:]}
    return {
        "finish": match.group(1),
        "prefill_tokens": int(match.group(2)),
        "prefill_seconds": float(match.group(3)),
        "decode_tokens": int(match.group(4)),
        "decode_seconds": float(match.group(5)),
        "decode_tokens_per_second": float(match.group(6)),
        "response": completed.stdout.strip(),
    }


def median(rows: list[dict], key: str) -> float:
    return statistics.median(row[key] for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=pathlib.Path, required=True)
    parser.add_argument("--compressed", type=pathlib.Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--max-new", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT / ".build/benchmark-results/compressed-experts-ab.json",
    )
    args = parser.parse_args()
    if args.rounds < 3:
        print("refusing: fewer than 3 rounds cannot separate a delta from noise", file=sys.stderr)
        return 2
    try:
        preflight(args.raw, args.compressed)
    except (RuntimeError, subprocess.SubprocessError, OSError) as error:
        print(f"refusing: {error}", file=sys.stderr)
        return 2

    arms = {"raw": args.raw, "compressed": args.compressed}
    rows: dict[str, list[dict]] = {"raw": [], "compressed": []}
    order = list(arms)
    for index in range(args.warmups + args.rounds):
        measured = index >= args.warmups
        round_order = order if index % 2 == 0 else list(reversed(order))
        for arm in round_order:
            result = measure(arms[arm], args.max_new, args.timeout)
            result["arm"] = arm
            result["round"] = index - args.warmups
            if not measured:
                continue
            if "error" in result:
                print(f"failed: {arm} round {result['round']}: {result['error']}", file=sys.stderr)
            rows[arm].append(result)

    usable = all(rows[arm] and all("error" not in row for row in rows[arm]) for arm in arms)
    if not usable:
        print("the A/B did not run: an arm produced no usable figure", file=sys.stderr)
        return 2

    raw_rate = median(rows["raw"], "decode_tokens_per_second")
    compressed_rate = median(rows["compressed"], "decode_tokens_per_second")
    delta = (compressed_rate / raw_rate - 1.0) * 100.0
    disagreements = [
        row["round"]
        for raw_row, row in zip(rows["raw"], rows["compressed"], strict=False)
        if raw_row["response"] != row["response"]
    ]
    print(f"{'arm':<12}{'decode tok/s':>14}{'TTFT s':>10}{'bytes/token':>14}")
    for arm in arms:
        print(
            f"{arm:<12}"
            f"{median(rows[arm], 'decode_tokens_per_second'):>14.2f}"
            f"{median(rows[arm], 'prefill_seconds'):>10.2f}"
            f"{(median(rows[arm], 'decode_seconds') / median(rows[arm], 'decode_tokens')):>14.4f}"
        )
    print(f"\ncompressed vs raw: {delta:+.2f}% decode")
    if disagreements:
        print(f"responses differ in rounds {disagreements}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "protocol": {
                    "prompt": PROMPT,
                    "max_new": args.max_new,
                    "rounds": args.rounds,
                    "warmups": args.warmups,
                },
                "arms": {arm: str(path) for arm, path in arms.items()},
                "rows": rows,
                "median": {"raw": raw_rate, "compressed": compressed_rate},
                "delta_percent": delta,
                "response_mismatches": disagreements,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"results: {args.output}")
    return 0 if not disagreements else 1


if __name__ == "__main__":
    raise SystemExit(main())
