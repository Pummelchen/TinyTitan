#!/usr/bin/env python3
"""Compare bounded F_NOCACHE pread with experimental Metal I/O.

Uses the same fresh-server cold/repeat protocol and response-equality gate as
the hit/fixup A/B, and its classifier: the two arms here are the I/O backends
rather than the decode schedules, so the rows are keyed on `io_backend` and the
same states -- compared, differ, empty-arm, unlogged, unusable -- decide the
answer. AUD-285 measured that this driver had been left on its own raw text
comparison, which passed a sweep where nothing answered, one where the backends
agreed on the words but not on the completion length, and one whose footers read
`decode_tok_s=nan`.

Metal I/O is not eligible as a production default unless these timing results
are paired with whole-machine memory-pressure evidence.

    0  every case compared, both backends answered, and every published figure
       came from a footer that carried it
    1  it measured, and the page is contested by a named disagreement or by cases
       it could not compare
    2  the A/B did not run: the driver's own refusal, no case answered, or no
       case published a usable figure, with the reason printed and no artifact
"""

from __future__ import annotations

import argparse
import json
import subprocess

import tinytitan_hit_fixup_ab as benchmark

BACKENDS = ("pread", "metal")


def main() -> int:
    argparse.ArgumentParser(
        description="Compare bounded pread and experimental Metal I/O"
    ).parse_args()
    try:
        metadata = benchmark.preflight()
    except (RuntimeError, subprocess.SubprocessError, OSError) as error:
        return benchmark.refuse(str(error), [])

    results: list[dict[str, object]] = []
    logs: dict[str, str] = {}
    prompts = tuple(benchmark.PROMPTS)
    try:
        for backend in BACKENDS:
            for prompt_name, prompt in benchmark.PROMPTS.items():
                rows, log = benchmark.run_case("hit-fixup", prompt_name, prompt, io_backend=backend)
                results.extend(rows)
                logs[f"{backend}/{prompt_name}"] = str(log)
    except (RuntimeError, subprocess.SubprocessError, OSError, ValueError, KeyError) as error:
        return benchmark.refuse(str(error), results)

    lines, status = benchmark.verdict(results, prompts, arm_key="io_backend", arms=BACKENDS)
    for line in lines:
        print(line)
    if status == 2:
        return 2

    cases = benchmark._cases(results, prompts, arm_key="io_backend", arms=BACKENDS)
    mismatches = [
        f"{case['prompt']}/{case['warmth']}" for case in cases if case["state"] == "differ"
    ]
    output = benchmark.ROOT / ".build/benchmark-results/io-backend-ab.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "warning": "Metal I/O requires separate page-cache/memory-pressure validation",
                "logs": logs,
                "results": results,
                "response_mismatches": mismatches,
                "status": status,
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
