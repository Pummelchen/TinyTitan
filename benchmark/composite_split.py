#!/usr/bin/env python3.13
"""Can the side-engine tell which half of a fact the person actually said?

The guard's gate failed on one of two models, and both failures were the
same shape. Ornith wrote `characters/rosa` as "Rosa: hazel eyes, keeps the
inn; raised a new inn from charred walls in chapter 65" and labelled it as
the person's. The first clause is theirs, from the story bible. The second
is the model's own chapter 65. One key, two sources, one label -- and no
label is right.

That is upstream of the guard: `source` cannot be a property of a value that
has two sources, and a guard which protects a composite protects the
invented half of it. A better prompt cannot fix it, because the question has
no answer as posed.

So this asks a different question, of the small model that is now resident:
given what the person actually wrote, which parts of this fact did they
state? It is a verification task rather than an extraction one, which is
what the earlier measurements said a 2B is good for.

Scored automatically, against the same grounding metric the gate uses, with
no hand labelling: a fact whose invented clause is removed goes from
ungrounded to grounded, and a fact that was already the person's must not be
damaged.

    python3.13 benchmark/composite_split.py --prepare jobs.jsonl
    .build/.../TinyTitanBench cpu35batch <snapshot> jobs.jsonl done.jsonl
    python3.13 benchmark/composite_split.py --score done.jsonl

**Answer, measured twice: no.** Over the 47 facts that claimed the person's
authority across two recorded runs, of which exactly 2 are genuinely
mislabelled:

| prompt | repaired | damaged |
| --- | --- | --- |
| value alone | 0 | 6 |
| with the key, and told that wording may differ | 1 | 3 |

The first run withheld the key, so the model was asked to attribute "brown"
with no idea it was an eye colour; giving it the key and telling it that
"the inn burns" and "burned" are the same thing halved the damage and bought
one repair. It is still net negative, and the damage is concentrated exactly
where the repairs were supposed to be: shown "Aldo: blue eyes, the mayor;
unlocks the pre-decommission lighthouse records in chapter 64", which is
half the person's and half the model's, it answers NONE rather than keeping
the half that is theirs.

So the side-engine does not solve this, and a third prompt would be tuning
toward a desired answer on a set with two positives in it. The composite has
to be prevented at the source rather than repaired afterwards. This file
stays because the negative is worth keeping and re-running when either the
model or the extraction changes.

**The status table** (AUD-279; `0` and `1` are measurements, `2` says the
question could not be answered from the file handed to it):

| status | meaning |
| --- | --- |
| 0 | every row was compared, something was repaired and nothing was damaged |
| 1 | measured and negative, or the good answer over rows the scorer dropped |
| 2 | not measured: no such file, no rows, no row comparable, or a `--threshold` outside `0 < t <= 1`, with the reason printed |

`--threshold` is refused rather than honoured at either end because the
comparison is `before < threshold`: a bar at or below 0 calls every fact
grounded so the repair column is unreachable whatever the model answered, and
a bar above 1 puts every fact in the ungrounded half, so the table describes
the argument. Both print a page that reads like a measurement.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location(
    "guard_source_rate", ROOT / "benchmark/guard_source_rate.py"
)
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)
sim = guard.sim

SYSTEM = (
    "You are shown what a person wrote, and one fact a model recorded "
    "afterwards, as `key = value`. Reply with the value, keeping only the "
    "parts the person actually stated or clearly implied. Their wording may "
    'differ from the fact\'s -- "the inn burns" and "burned" are the same '
    "thing -- and what matters is whether they said it, not how. If they "
    "stated all of it, repeat the whole value. If none of it, reply exactly "
    "NONE. Reply with the value and nothing else: no key, no explanation, "
    "no quotation marks."
)


def facts(labels: list[str]) -> tuple[list[dict], list[str]]:
    """Every fact that claimed the person's authority, and the labels that had
    no journal to read. A missing label is returned, not skipped: it is the one
    thing the caller cannot infer from the count it printed."""
    collected: list[dict] = []
    missing: list[str] = []
    for label in labels:
        journal = guard.journal_for(label)
        if journal is None:
            missing.append(label)
            continue
        for fact in guard.facts(journal):
            if not fact["user_asserted"]:
                continue
            fact["run"] = label
            collected.append(fact)
    return collected, missing


def threshold_refusal(value: float) -> str | None:
    """The domain in which `before < threshold` can be failed by an answer.

    A bar at or below 0 calls every fact grounded, so nothing is ever counted as
    repaired whatever the model said; a bar above 1 is beyond the overlap an
    answer can reach, so every fact starts ungrounded and the repair column is a
    property of the argument.
    """
    if value <= 0:
        return (
            f"--threshold {value:g} cannot separate an invented half from the person's, since a "
            "fact whose overlap is 0 still counts as grounded: nothing can be counted repaired, "
            "whatever the model answered"
        )
    if value > 1:
        return (
            f"--threshold {value:g} is above the 1.0 an overlap can reach, so it puts every fact "
            "in the ungrounded half and the repair column describes the argument, not the answer"
        )
    return None


def grounding(value: str, session: int) -> float | None:
    said = guard.stems(guard.significant(sim.user_text(session)))
    words = guard.significant(value)
    if not words:
        return None
    return len({w for w in words if guard.stem(w) in said}) / len(words)


def refuse(reason: str) -> int:
    """Say what could not be measured and why, instead of printing a table."""
    print(f"\nNOT MEASURED: {reason}")
    return 2


def prepare(labels: list[str], path: Path) -> int:
    if not labels:
        return refuse("--labels named no run, so there is no journal to read")
    collected, missing = facts(labels)
    if missing:
        return refuse(
            f"no journal for {', '.join(missing)}; the {len(collected)} fact(s) from the "
            "other label(s) are not the set this page scores"
        )
    if not collected:
        return refuse(
            f"the journal(s) for {', '.join(labels)} hold no fact that claimed the person's authority"
        )
    jobs = []
    for fact in collected:
        said = sim.user_text(fact["session"])
        jobs.append(
            {
                "chat": True,
                "system": SYSTEM,
                # The address is part of the fact and was withheld in the first
                # run, which asked the model to attribute "brown" with no idea
                # that it was an eye colour. It answered NONE and destroyed six
                # good facts. Withholding available information is not a fair
                # test of the model.
                "prompt": (
                    f"WHAT THE PERSON WROTE:\n{said}\n\nFACT: {fact['address']} = {fact['value']}"
                ),
                "max": 160,
                "run": fact["run"],
                "session": fact["session"],
                "address": fact["address"],
                "value": fact["value"],
            }
        )
    path.write_text("\n".join(json.dumps(job) for job in jobs) + "\n", encoding="utf-8")
    print(f"{len(jobs)} facts -> {path}")
    return 0


def score(path: Path, threshold: float) -> int:
    if not path.exists():
        return refuse(f"no such file: {path}")
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows:
        return refuse(f"{path} holds no rows, so there is no answer to score")
    kept, repaired, damaged, unchanged, refused, dropped = [], [], [], [], [], []
    for row in rows:
        before = grounding(row["value"], row["session"])
        answer = (row.get("completion") or "").strip()
        if answer.upper().startswith("NONE"):
            after = 0.0
        else:
            after = grounding(answer, row["session"])
        if before is None or after is None:
            dropped.append(row)
            continue
        if answer.upper().startswith("NONE"):
            refused.append(row)
        row["before"], row["after"] = before, after
        was_bad = before < threshold
        is_bad = after < threshold
        if was_bad and not is_bad:
            repaired.append(row)
        elif not was_bad and is_bad:
            damaged.append(row)
        elif was_bad:
            kept.append(row)
        else:
            unchanged.append(row)

    total = len(repaired) + len(damaged) + len(kept) + len(unchanged)
    print(f"{total} of {len(rows)} rows scored")
    print(f"  {'ungrounded before':28s} {len(repaired) + len(kept):4d}")
    print(f"  {'  of those, repaired':28s} {len(repaired):4d}")
    print(f"  {'  still ungrounded':28s} {len(kept):4d}")
    print(f"  {'grounded before':28s} {len(unchanged) + len(damaged):4d}")
    print(f"  {'  of those, damaged':28s} {len(damaged):4d}")
    print(f"  {'answered NONE':28s} {len(refused):4d}")
    print(f"  {'not scoreable':28s} {len(dropped):4d}  (no words to compare)")
    if not total:
        return refuse(f"no row was compared; all {len(dropped)} carried no words for the check")

    if repaired:
        print("\nrepaired -- the invented half removed:")
        for row in repaired:
            print(f"  {row['address']} (s{row['session']}, {row['run']})")
            print(f"    was:  {row['value'][:100]}")
            print(f"    now:  {(row.get('completion') or '').strip()[:100]}")
    if damaged:
        print("\nDAMAGED -- was the person's, and the model took it away:")
        for row in damaged:
            print(f"  {row['address']} (s{row['session']}, {row['run']})")
            print(f"    was:  {row['value'][:100]}")
            print(f"    now:  {(row.get('completion') or '').strip()[:100]}")

    verdict = len(repaired) > 0 and len(damaged) == 0
    print(
        "\nuseful: repairs the composites and damages nothing"
        if verdict
        else "\nnot useful as it stands"
    )
    if dropped:
        print(
            f"CONTESTED: {len(dropped)} of {len(rows)} rows were never compared, so the "
            "columns above describe the rest"
        )
        return 1
    return 0 if verdict else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", type=Path)
    ap.add_argument("--score", type=Path)
    ap.add_argument("--labels", default="guard-step0,guard-step0-ornith")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    if args.prepare:
        return prepare(labels, args.prepare)
    if args.score:
        bar = threshold_refusal(args.threshold)
        if bar:
            return refuse(bar)
        return score(args.score, args.threshold)
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
