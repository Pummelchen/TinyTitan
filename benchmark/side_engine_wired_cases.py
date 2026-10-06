#!/usr/bin/env python3.13
"""The side-engine's wired judgements, on the case the wiring actually uses.

`benchmark/side_engine_tasks.py` measures T3 and T5 over pairs that either
share a key or are plainly about different things. The memory path asks a
narrower question, and it is the only one that reaches the engine:

- **T5 duplication**: a *new* key that says what an existing key already said,
  so the store keeps one address instead of two. A pair sharing a key never
  reaches the engine — the deterministic fold-equality check skips it first —
  so the same-key cases in the main benchmark cannot say whether the wiring
  works.
- **T3 contradiction**: a new key whose value cannot both be true with an
  existing one. The answer is advisory only: disagreement is not supersession,
  and T4, which would tell them apart, needs the stored rule supplied and has no
  source for it in the memory path yet.

    python3.13 benchmark/side_engine_wired_cases.py --prepare jobs.jsonl
    .build/release/TinyTitanBench cpu35batch <install> jobs.jsonl done.jsonl
    python3.13 benchmark/side_engine_wired_cases.py --score done.jsonl

The cases are authored here and marked so, because the point is the shape of
the pair rather than the fact: the book's own facts are the substrate, and
re-filing `characters/marcus/eyes = grey` as `characters/marcus/eye_colour`,
or setting two characters' eyes against each other, are what a consolidation
really produces.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tasks = _load("side_engine_tasks", "benchmark/side_engine_tasks.py")

# (existing key, existing value, new key, new value, truth, note)
T5_CASES = [
    (
        "characters/marcus/eyes",
        "grey",
        "characters/marcus/eye_colour",
        "grey",
        "YES",
        "re-filed eyes",
    ),
    ("setting/town", "Ashgrove", "setting/town_name", "Ashgrove", "YES", "re-filed town"),
    (
        "rules/ferry",
        "runs only on Sundays",
        "rules/ferry_schedule",
        "only Sundays",
        "YES",
        "re-filed ferry rule",
    ),
    (
        "characters/ines/role",
        "the town archivist",
        "characters/ines/job",
        "archivist",
        "YES",
        "re-filed role",
    ),
    (
        "state/inn",
        "burned to the ground",
        "state/inn_status",
        "destroyed by fire",
        "YES",
        "re-filed inn state",
    ),
    ("characters/marcus/eyes", "grey", "characters/ines/eyes", "green", "NO", "two characters"),
    ("setting/town", "Ashgrove", "rules/ferry", "runs only on Sundays", "NO", "different subjects"),
    (
        "characters/marcus/role",
        "the lighthouse keeper's son",
        "characters/marcus/eyes",
        "grey",
        "NO",
        "different attributes",
    ),
]

T3_CASES = [
    (
        "characters/marcus/eyes",
        "grey",
        "characters/marcus/eye_colour",
        "hazel",
        "YES",
        "same fact, two values",
    ),
    ("setting/town", "Ashgrove", "setting/town_name", "Millbrook", "YES", "same fact, two values"),
    ("characters/marcus/eyes", "grey", "characters/ines/eyes", "green", "NO", "two characters"),
    (
        "characters/marcus/eye_colour",
        "grey",
        "characters/marcus/eyes",
        "grey",
        "NO",
        "same fact, two keys",
    ),
]


def cases() -> list[dict]:
    jobs: list[dict] = []
    for task, question, rows in (
        ("T5", "Same fact?", T5_CASES),
        ("T3", "Do A and B disagree?", T3_CASES),
    ):
        for a_key, a_value, b_key, b_value, truth, note in rows:
            jobs.append(
                tasks.job(
                    task,
                    f"A: {a_key} = {a_value}\nB: {b_key} = {b_value}\n{question}",
                    truth,
                    note,
                    authored=True,
                )
            )
    return jobs


def prepare(path: Path) -> int:
    jobs = cases()
    path.write_text("\n".join(json.dumps(j) for j in jobs) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    for j in jobs:
        counts[j["task"]] = counts.get(j["task"], 0) + 1
    print(f"{len(jobs)} cases -> {path}")
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 0


def score(path: Path) -> int:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    by_task: dict[str, list[int]] = {}
    print(f"{'task':5s} {'truth':6s} {'answer':8s} note")
    for row in rows:
        answer = (row.get("completion") or "").strip().upper().split()
        answer = answer[0].strip(".,:;\"'") if answer else ""
        hit = answer == row["truth"]
        entry = by_task.setdefault(row["task"], [0, 0])
        entry[0] += hit
        entry[1] += 1
        print(
            f"{row['task']:5s} {row['truth']:6s} {answer:8s} {'' if hit else 'MISS '}{row['note']}"
        )
    print()
    failures = 0
    for task, (correct, total) in sorted(by_task.items()):
        print(f"{task}: {correct}/{total}")
        failures += correct != total
    return 0 if not failures else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", type=Path)
    ap.add_argument("--score", type=Path)
    args = ap.parse_args()
    if args.prepare:
        return prepare(args.prepare)
    if args.score:
        return score(args.score)
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
