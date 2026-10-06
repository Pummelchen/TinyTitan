#!/usr/bin/env python3.13
"""Score T6 reply-check runs against the labels and the pre-registered gate.

The gate (`docs/t6-reply-check-offline.md`) is precision >= 95%, recall >= 50%,
and a silence false-alarm rate <= 5%. This reads one or more judge output files
(the JSONL `benchmark/side_engine_judges.py` writes) plus the review file that
`benchmark/t6_prose_cases.py` writes, and reports those numbers per judge.

    python3.13 benchmark/t6_prose_score.py \
        --review /tmp/t6-prose-review.tsv \
        --done "server-35B:/tmp/t6-server-done.jsonl" \
        --done "cpu-4B:/tmp/t6-4b-done.jsonl"
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def answer_of(completion: str) -> str:
    text = (completion or "").strip().upper()
    if not text:
        return ""
    token = text.split()[0].strip(".,:;\"'`*")
    return token if token in {"YES", "NO"} else f"<{token[:12]}>"


def load_labels(review: Path) -> dict[str, dict]:
    labels = {}
    with review.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["kind"] == "excluded":
                continue
            labels["r" + row["run"] + "s" + row["session"] + " " + row["key"]] = row
    return labels


def report(name: str, done: Path, labels: dict[str, dict]) -> dict:
    rows = [
        json.loads(line) for line in done.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    scored = []
    for row in rows:
        label = labels.get(row.get("note", ""))
        if label is None:
            continue
        scored.append((label, answer_of(row.get("completion", ""))))

    total = len(scored)
    right = sum(1 for label, answer in scored if answer == label["truth"])
    positives = [(label, a) for label, a in scored if label["truth"] == "YES"]
    negatives = [(label, a) for label, a in scored if label["truth"] == "NO"]
    fired = [(label, a) for label, a in scored if a == "YES"]
    true_positive = [(label, a) for label, a in fired if label["truth"] == "YES"]
    silent = [(label, a) for label, a in scored if label["kind"] == "silent"]
    silent_fired = [(label, a) for label, a in silent if a == "YES"]
    bad = [(label, a) for label, a in scored if not a or a.startswith("<")]

    precision = len(true_positive) / len(fired) if fired else 0.0
    recall = len(true_positive) / len(positives) if positives else 0.0
    silence_rate = len(silent_fired) / len(silent) if silent else 0.0

    print(f"\n== {name}  ({total} scored of {len(rows)} completions)")
    print(f"  accuracy          {right}/{total} = {100 * right / max(1, total):.1f}%")
    print(f"  YES precision     {len(true_positive)}/{len(fired)} = {100 * precision:.1f}%")
    print(f"  recall            {len(true_positive)}/{len(positives)} = {100 * recall:.1f}%")
    print(f"  YES half          {len(true_positive)}/{len(positives)}")
    print(
        f"  NO half           {len(negatives) - sum(1 for label, a in negatives if a == 'YES')}/{len(negatives)}"
    )
    print(f"  silence alarms    {len(silent_fired)}/{len(silent)} = {100 * silence_rate:.1f}%")
    print(f"  unparseable       {len(bad)}")
    print(
        f"  GATE              precision {'PASS' if precision >= 0.95 else 'FAIL'}"
        f"  recall {'PASS' if recall >= 0.50 else 'FAIL'}"
        f"  silence {'PASS' if silence_rate <= 0.05 else 'FAIL'}"
    )

    per_key: dict[str, list[int]] = {}
    for label, answer in scored:
        cell = per_key.setdefault(label["key"], [0, 0, 0, 0])  # n, right, fired, positives
        cell[0] += 1
        cell[1] += answer == label["truth"]
        cell[2] += answer == "YES"
        cell[3] += label["truth"] == "YES"
    print(f"  {'key':38s} {'n':>3s} {'right':>6s} {'fired':>6s} {'pos':>4s}")
    for key in sorted(per_key):
        n, ok, fire, pos = per_key[key]
        print(f"  {key:38s} {n:3d} {100 * ok / n:5.0f}% {fire:6d} {pos:4d}")
    return {
        "name": name,
        "total": total,
        "accuracy": right / max(1, total),
        "precision": precision,
        "recall": recall,
        "silence_rate": silence_rate,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", type=Path, required=True)
    ap.add_argument("--done", action="append", required=True, help="name:path")
    args = ap.parse_args()

    labels = load_labels(args.review)
    print(f"labels: {len(labels)}")
    summary = []
    for spec in args.done:
        name, _, path = spec.partition(":")
        summary.append(report(name, Path(path), labels))
    print("\n== summary")
    for row in summary:
        print(
            f"  {row['name']:14s} acc {100 * row['accuracy']:5.1f}%"
            f"  precision {100 * row['precision']:5.1f}%"
            f"  recall {100 * row['recall']:5.1f}%"
            f"  silence {100 * row['silence_rate']:4.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
