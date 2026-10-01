#!/usr/bin/env python3.13
"""Does precision buy quality? Held-out perplexity, the sharper A/B.

TT-025's twenty-prompt check (`quant_quality_ab.py`) is a floor: two installs
of a 4B scored 18/20 and missed the *same* two cases, so it cannot see a small
perplexity effect. The Engineering Notes say a sharper check — held-out
perplexity — is cheap enough to run before concluding precision never matters.
This is it. A fixed text is scored through the CPU forward pass, once per
install, and because both passes see the *same token positions*, the
difference between two quantizations of one model is a paired number (mean
dNLL, standard error, t) rather than two means that could hide a real effect.

    python3.13 benchmark/quant_perplexity_ab.py \\
        models/qwen3.5_4B_4Bit .build/qwen35-4b-uniform.ssdai models/qwen3.5_4B_8Bit

The first install is the baseline; each other is compared against it. A run
whose token hash does not match the baseline is refused rather than averaged —
two installs scoring different text is not a comparison.

The default text is assembled from this repository's own documentation. It is
held out in the sense that matters here: the comparison is between two widths
of the same weights on identical tokens, so training contamination affects
both arms equally. `--text` names a file instead, and `--tokens` bounds the
work (1024 by default).
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / ".build/release/TinyTitanBench"
# Natural prose, not part of any golden, and cheap to read. Order is fixed so
# the token hash is reproducible.
CORPUS = [
    "docs/side-engine-tasks.md",
    "docs/agent-memory.md",
    "docs/handover-tinytitan.md",
    ".qwen/wiki/Engineering-Notes.md",
]

SUMMARY = re.compile(r"mean nll ([\d.]+)\s+perplexity ([\d.]+)\s+seconds ([\d.]+)")
HEADER = re.compile(r"token hash ([0-9a-f]+)")


def corpus_text(named: str | None) -> tuple[str, list[str]]:
    if named:
        return Path(named).read_text(encoding="utf-8"), [named]
    pieces, used = [], []
    for relative in CORPUS:
        path = ROOT / relative
        if not path.exists():
            continue
        pieces.append(path.read_text(encoding="utf-8"))
        used.append(relative)
    if not pieces:
        raise SystemExit(f"none of the default corpus files exist under {ROOT}")
    return "\n\n".join(pieces), used


def score(install: str, text: Path, tokens: int, nll_out: Path) -> dict:
    result = subprocess.run(
        [str(BENCH), "cpu35ppl", install, str(text), str(tokens), str(nll_out)],
        capture_output=True,
        text=True,
        timeout=7200,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"{install}: cpu35ppl exited {result.returncode}\n" + result.stderr[-2000:]
        )
    header = HEADER.search(result.stdout)
    summary = SUMMARY.search(result.stdout)
    if not header or not summary:
        raise SystemExit(f"{install}: could not read the bench output\n{result.stdout}")
    return {
        "install": install,
        "hash": header.group(1),
        "mean_nll": float(summary.group(1)),
        "perplexity": float(summary.group(2)),
        "seconds": float(summary.group(3)),
        "nlls": [float(line) for line in nll_out.read_text(encoding="utf-8").splitlines()],
    }


def paired(baseline: list[float], other: list[float]) -> tuple[float, float, float]:
    differences = [a - b for a, b in zip(baseline, other, strict=False)]
    count = len(differences)
    mean = sum(differences) / count
    if count < 2:
        return mean, 0.0, math.inf
    stderr = statistics.stdev(differences) / math.sqrt(count)
    return mean, stderr, mean / stderr if stderr > 0 else math.inf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "installs", nargs="+", help="two or more installs of one model; the first is the baseline"
    )
    ap.add_argument(
        "--tokens", type=int, default=1024, help="token positions to score (default 1024)"
    )
    ap.add_argument("--text", help="score this file instead of the default corpus")
    args = ap.parse_args()
    if not BENCH.exists():
        print(
            f"build it first: swift build -c release --product TinyTitanBench\n  missing {BENCH}",
            file=sys.stderr,
        )
        return 2

    body, sources = corpus_text(args.text)
    print("held-out text: " + ", ".join(sources))
    with tempfile.TemporaryDirectory() as tmp:
        text = Path(tmp) / "held-out.txt"
        text.write_text(body, encoding="utf-8")
        runs = []
        for index, install in enumerate(args.installs):
            print(f"\n=== {install}")
            run = score(install, text, args.tokens, Path(tmp) / f"nll{index}.txt")
            runs.append(run)
            print(
                f"  mean nll {run['mean_nll']:.6f}  perplexity "
                f"{run['perplexity']:.6f}  {run['seconds']:.1f}s  "
                f"hash {run['hash']}"
            )

    baseline = runs[0]
    print(f"\n=== paired against {baseline['install']}")
    print(f"{'install':44s} {'dNLL':>10s} {'±se':>9s} {'t':>8s} {'ppl ratio':>10s} {'n':>6s}")
    for run in runs[1:]:
        if run["hash"] != baseline["hash"]:
            print(f"{run['install']:44s} refused: token hash {run['hash']} != {baseline['hash']}")
            continue
        mean, stderr, t = paired(baseline["nlls"], run["nlls"])
        ratio = math.exp(mean)
        count = len(run["nlls"])
        print(f"{run['install']:44s} {mean:+10.6f} {stderr:9.6f} {t:8.2f} {ratio:10.6f} {count:6d}")
    print(
        "\ndNLL is baseline minus install, per token position; positive means the "
        "baseline is worse. t is the paired statistic over the same positions."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
