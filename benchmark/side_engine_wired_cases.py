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

`--score` answers a status, and the totals it prints are only as wide as the
cases that earned them:

    0  every authored case ran, was answered, and every answer matched
    1  measured and contested: a case ran no rows or carried no judgement, a row
       from another case set came in, or a judgement disagreed with the truth
    2  the comparison does not exist: no rows, a file that cannot be read, or no
       row that belongs to this case set

Silence is not an answer, so a case whose row carries no completion leaves the
totals and is named rather than scored as a NO, and the ground truth is the
authored case list, not whatever rows arrived. A judgement is read from the
first word of the completion, upper-cased and stripped of punctuation, which is
how the recorded 4B and 9B runs were scored.
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


def refuse(reason: str) -> int:
    """Say why there is no comparison, print no totals, and answer 2."""
    print(f"\nNOT MEASURED: {reason}")
    return 2


def judgement(row: dict) -> str:
    """The first word of the completion, or nothing when there is no answer."""
    words = (row.get("completion") or "").strip().upper().split()
    return words[0].strip(".,:;\"'") if words else ""


def score(path: Path) -> int:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        return refuse(f"{path} could not be read: {error}")
    rows: list[dict] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as error:
            return refuse(f"{line[:60]} is not a judgement row ({error})")
        if not isinstance(row, dict) or not {"task", "truth", "prompt", "note"} <= set(row):
            return refuse(f"{line[:60]} holds no task, truth, prompt or note to score it against")
        rows.append(row)
    if not rows:
        return refuse(f"{path} holds no rows, so no case was answered")

    authored = {(row["task"], row["prompt"]): row for row in cases()}
    if not any((row["task"], row["prompt"]) in authored for row in rows):
        return refuse(
            f"none of the {len(rows)} row(s) is one of the {len(authored)} cases this driver "
            "prepared, so these answers are not this comparison's"
        )

    seen: dict[tuple[str, str], list[dict]] = {}
    unrecognised = 0
    for row in rows:
        key = (row["task"], row["prompt"])
        if key not in authored:
            unrecognised += 1
            continue
        seen.setdefault(key, []).append(row)

    print(f"{'task':5s} {'truth':6s} {'answer':8s} note")
    by_task: dict[str, list[int]] = {}
    missed = silent = unwritten = repeated = 0
    for key, job in authored.items():
        group = seen.get(key, [])
        if len(group) > 1:
            repeated += 1
        answer = judgement(group[0]) if group else ""
        if not group:
            unwritten += 1
            print(f"{job['task']:5s} {job['truth']:6s} {'-':8s} did not run  {job['note']}")
            continue
        if not answer:
            silent += 1
            print(f"{job['task']:5s} {job['truth']:6s} {'-':8s} no judgement  {job['note']}")
            continue
        entry = by_task.setdefault(job["task"], [0, 0])
        entry[1] += 1
        hit = answer == job["truth"]
        entry[0] += hit
        if not hit:
            missed += 1
        again = f"  (answered {len(group)} times, the first kept)" if len(group) > 1 else ""
        print(
            f"{job['task']:5s} {job['truth']:6s} {answer:8s} {'' if hit else 'MISS '}"
            f"{job['note']}{again}"
        )

    judged = sum(v[1] for v in by_task.values())
    print()
    for task in sorted(by_task):
        correct, total = by_task[task]
        print(f"{task}: {correct}/{total}")
    print(f"\njudged {judged} of {len(authored)} cases")
    if unrecognised:
        print(f"  unrecognised: {unrecognised} row(s) from another case set, never scored")
    if silent:
        print(f"  silence: {silent} case(s) carried no judgement, so they are out of the totals")
    if unwritten:
        print(f"  missing: {unwritten} case(s) ran no row")
    if repeated:
        print(f"  repeated: {repeated} case(s) answered more than once")
    if missed:
        print(f"  {missed} case answered differently, which is the wiring's result, not a gap")
    if not judged:
        return refuse(
            f"none of the {len(rows)} row(s) carried a judgement, so nothing was compared"
        )
    return 0 if not (missed or silent or unwritten or repeated or unrecognised) else 1


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
