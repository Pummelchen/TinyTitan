#!/usr/bin/env python3.13
"""Does precision buy quality? A checkable A/B between two installs of one model.

TT-025 wants to promote ~10 MB of precision-sensitive tensors to 8 bits inside a
4-bit install, because taking the whole attention block there costs +2.10 GB
resident. Before building anything, the question is whether precision buys
*quality* at all on these models — the Engineering Notes record weight-space
fidelity and selection stability and say plainly that no end-to-end quality
measurement exists.

So this is a first one. Twenty prompts with answers a script can check: arithmetic
including multi-step, a counting task, a syllogism, a format constraint, a
reversal, a calendar step, a fact. Greedy, fixed prompt, one generation each, so
two installs of the same model are compared on identical inputs.

    python3.13 benchmark/quant_quality_ab.py models/qwen3.5_4B_4Bit models/qwen3.5_4B_8Bit

It runs the release CLI once per (install, case) because that is the only runner
that needs no server and no port. It is a *floor*, not a suite: twenty short
answerable prompts cannot see a small perplexity difference, and a ceiling
result — both installs at 20/20 — is evidence that this instrument is too blunt,
not that precision does not matter. It is here so a promotion has something to
fail against before it costs a 360 GB fetch.

A case the CLI refuses is not a case the model answered wrongly. Refusals are
counted apart, the score's denominator is the cases that ran, and any refusal makes
the run exit 1: two installs that never ran would otherwise print as a tie, which
is the one reading that could not support any decision. A score of 0/20 from twenty
real replies still exits 0, because that is a result.

    cd benchmark && python3 -m unittest test_quant_quality_ab_exit
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLI = ROOT / ".build/release/TinyTitanCLI"
SYSTEM = "Answer with just the answer, nothing else."
MAX_NEW = 48

# (prompt, expected token, how to match)
CASES: list[tuple[str, str, str]] = [
    ("What is 17 + 28?", "45", "number"),
    ("What is 13 x 7?", "91", "number"),
    ("If x + 9 = 20, what is x?", "11", "number"),
    (
        "A train leaves at 14:20 and arrives at 16:05. How many minutes is the journey?",
        "105",
        "number",
    ),
    ("How many letter r are in the word strawberry?", "3", "number"),
    ("Which is the smallest: 0.7, 0.07, or 0.17?", "0.07", "number"),
    ("What is the capital of Australia?", "Canberra", "word"),
    ("What is 100 - 37?", "63", "number"),
    (
        "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops Lazzies?",
        "yes",
        "word",
    ),
    ("Write the numbers 1 to 5 as a JSON array.", "[1,2,3,4,5]", "json"),
    ("What is 12% of 250?", "30", "number"),
    ("Who wrote Pride and Prejudice?", "Austen", "word"),
    ("What is 2 to the power of 10?", "1024", "number"),
    ("Reverse the word stressed.", "desserts", "word"),
    ("If today is Wednesday, what day is it in 10 days?", "Saturday", "word"),
    ("What is the sum of the first five prime numbers?", "28", "number"),
    ("Which word does not belong: apple, banana, carrot, cherry?", "carrot", "word"),
    ("What is 9 squared minus 9?", "72", "number"),
    ("How many minutes are in 2.5 hours?", "150", "number"),
    ("What is 7 factorial?", "5040", "number"),
]


def completion_of(stdout: str) -> str:
    """The model's text, without the CLI's `[stop=…]` statistics line."""
    lines = [line for line in stdout.splitlines() if not line.startswith("[stop=")]
    return "\n".join(lines).strip()


def passed(reply: str, expected: str, kind: str) -> bool:
    text = reply.strip()
    if kind == "json":
        return re.sub(r"\s+", "", text) == expected
    if kind == "number":
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        if not numbers:
            return False
        # The last number is the answer when the model restates the question;
        # compare numerically so "45." and "45" agree.
        return any(abs(float(n) - float(expected)) < 1e-9 for n in numbers)
    return expected.lower() in text.lower()


class CaseRefused(RuntimeError):
    """The CLI answered nothing, so this case has no answer to score."""


def run_case(model: str, prompt: str) -> str:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
        json.dump(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}], handle
        )
        path = handle.name
    try:
        result = subprocess.run(
            [
                str(CLI),
                "--model",
                model,
                "--messages-file",
                path,
                "--max-new",
                str(MAX_NEW),
                "--temperature",
                "0",
            ],
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
    finally:
        Path(path).unlink(missing_ok=True)
    if result.returncode != 0:
        # The reason travels: a refusal that only prints `<exit 127>` is the shape
        # that let a broken install read as a score, and the CLI's own line — a moved
        # model directory, an unsupported flag — is what tells them apart.
        tail = [line.strip() for line in result.stderr.strip().splitlines() if line.strip()]
        reason = tail[-1] if tail else f"no output on exit {result.returncode}"
        raise CaseRefused(f"exit {result.returncode}: {reason[:200]}")
    return completion_of(result.stdout)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="two or more installs of one model")
    args = ap.parse_args()
    if not CLI.exists():
        print(
            f"build it first: swift build -c release --product TinyTitanCLI\n  missing {CLI}",
            file=sys.stderr,
        )
        return 2

    # A case that refused is not a case that was answered wrongly. `hits` is
    # therefore the answers that were actually measured, and `refused` rides beside it
    # so the denominator cannot absorb the failures the way `<exit N>` used to.
    results: dict[str, tuple[list[bool], int]] = {}
    for model in args.models:
        print(f"\n=== {model}")
        hits: list[bool] = []
        refused = 0
        for prompt, expected, kind in CASES:
            try:
                reply = run_case(model, prompt)
            except CaseRefused as case:
                refused += 1
                print(f"  ERR   want={expected:12s} {case}")
                continue
            ok = passed(reply, expected, kind)
            hits.append(ok)
            print(f"  {'ok  ' if ok else 'MISS'} want={expected:12s} got={reply[:40]!r}")
        results[model] = (hits, refused)

    print("\n=== totals")
    any_refused = False
    for model, (hits, refused) in results.items():
        line = f"  {model:44s} {sum(hits)}/{len(hits)}"
        if refused:
            any_refused = True
            line += f" measured, {refused} errored"
        print(line)
    # Zeroed cases are a result and exit 0; refused cases mean nothing was measured,
    # and a tie between two installs that never ran is the false finding this avoids.
    return 1 if any_refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
