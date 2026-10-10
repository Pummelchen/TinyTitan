#!/usr/bin/env python3
"""Qualification matrix for v4.2 expert control-plane modes.

Every case uses a fresh server, identical requests, bounded pread unless the
case explicitly selects Metal I/O, and the response-equality gate inherited
from the v4.1 hit/fixup benchmark. Experimental cases are not promoted merely
because this script can run them.

The inherited gate is the neighbour module's classifier, run once per reference
and case, so a case answers 0, 1 or 2 rather than agreeing about nothing:

    0  every case compared against the reference and every published figure came
       from a footer that carried it
    1  it measured, and a named comparison is contested -- by a disagreement, by
       an arm that answered nothing, or by a pair that published no usable figure
    2  the measurement does not exist: a refused preflight, a case that died, no
       case answered, no figure readable anywhere, or no second case to compare
       the reference against -- reason printed, no artifact written
"""

from __future__ import annotations

import argparse
import json
import subprocess

import tinytitan_hit_fixup_ab as benchmark


CASES = (
    ("production-deferred", "hit-fixup", "pread", "host", "per-slot", "lfu", "deferred"),
    ("immediate-host", "hit-fixup", "pread", "host", "per-slot", "lfu", "immediate"),
    ("event-pread", "hit-fixup", "pread", "event", "per-slot", "lfu", "immediate"),
    ("event-pool", "hit-fixup", "pread", "event", "pool", "lfu", "immediate"),
    ("gpu-residency", "gpu-residency", "pread", "event", "pool", "lfu", "immediate"),
    ("gpu-residency-aging", "gpu-residency", "pread", "event", "pool", "aging-lfu", "immediate"),
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare v4.2 expert scheduling, event, pool, and residency modes"
    )
    parser.add_argument("--prompt", choices=tuple(benchmark.PROMPTS) + ("all",), default="all")
    parser.add_argument(
        "--case",
        action="append",
        dest="selected_cases",
        choices=tuple(case[0] for case in CASES),
        help="run only this case (repeatable)",
    )
    args = parser.parse_args()
    try:
        metadata = benchmark.preflight()
    except (RuntimeError, subprocess.SubprocessError, OSError) as error:
        return benchmark.refuse(str(error), [])
    prompts = (
        benchmark.PROMPTS if args.prompt == "all" else {args.prompt: benchmark.PROMPTS[args.prompt]}
    )
    cases = tuple(
        case for case in CASES if not args.selected_cases or case[0] in args.selected_cases
    )
    if len(cases) < 2:
        return benchmark.refuse(
            f"{cases[0][0]} has no reference to compare against; the matrix needs a "
            "second case before any comparison exists",
            [],
        )
    results: list[dict[str, object]] = []
    logs: dict[str, str] = {}
    try:
        for name, mode, backend, sync, layout, policy, submission in cases:
            for prompt_name, prompt in prompts.items():
                rows, log = benchmark.run_case(
                    mode,
                    prompt_name,
                    prompt,
                    io_backend=backend,
                    io_sync=sync,
                    cache_layout=layout,
                    cache_policy=policy,
                    io_submission=submission,
                )
                for row in rows:
                    row["case"] = name
                results.extend(rows)
                logs[f"{name}/{prompt_name}"] = str(log)
    except (RuntimeError, subprocess.SubprocessError, OSError, ValueError, KeyError) as error:
        return benchmark.refuse(str(error), results)

    reference = cases[0][0]
    names = tuple(prompts)
    mismatches: list[str] = []
    statuses: list[tuple[str, int]] = []
    for name, *_ in cases[1:]:
        print(f"\n{reference} vs {name}")
        lines, pair_status = benchmark.verdict(
            results, names, arm_key="case", arms=(reference, name)
        )
        for line in lines:
            print(line)
        statuses.append((name, pair_status))
        compared = benchmark._cases(results, names, arm_key="case", arms=(reference, name))
        for case in compared:
            if case["state"] == "differ":
                mismatches.append(f"{name} {case['prompt']}/{case['warmth']}")

    unmeasured = [name for name, status in statuses if status == 2]
    if not unmeasured:
        status = 1 if mismatches or any(one == 1 for _, one in statuses) else 0
    elif len(unmeasured) == len(statuses):
        return 2
    else:
        print(
            f"CONTESTED: {', '.join(unmeasured)} published no usable figure against "
            f"{reference}, so those comparisons do not exist"
        )
        status = 1

    output = benchmark.ROOT / ".build/benchmark-results/v4.2-control-plane-ab.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "configuration": {
                    "model": str(benchmark.DEFAULT_MODEL_PATH),
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "top_k": 20,
                    "presence_penalty": 0.0,
                    "seed": 41,
                    "max_completion_tokens": 128,
                },
                "cases": [case[0] for case in cases],
                "reference": reference,
                "logs": logs,
                "results": results,
                "comparison_status": dict(statuses),
                "status": status,
                "response_mismatches": mismatches,
                "passed": status == 0,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"results: {output}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
