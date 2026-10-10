#!/usr/bin/env python3.13
"""Would a caller get anything for a T7 judgement? Recall, measured.

T7 is 100% on the main benchmark's pairs and has no caller: `memory_search` is
a tool call the client's turn waits on, and a judgement costs 15.2 s on the 4B
(`docs/side-engine-tasks.md`). So the question a caller has to answer first is
what the ranking is worth. This compares the deterministic token ranking
`MemoryRanking` uses — a term in the key scores 3, in the value 2, and a fact
sharing no term is not returned at all — against asking the engine, for each
fact, whether it could answer the question.

The questions are authored to give the token ranking a hard time: they avoid
the target's own words where they can, name the attribute rather than the
holder, or ask about the value through the key.

    python3.13 benchmark/side_engine_recall.py --prepare jobs.jsonl
    .build/release/TinyTitanBench cpu35batch <install> jobs.jsonl done.jsonl
    python3.13 benchmark/side_engine_recall.py --score done.jsonl

Only an authored set can do this — the book's records have no questions
attached — so every case is marked as such.

--score answers a status, and the rates it prints are only as wide as the rows
that earned them:

    0  every fair question ran, answered every fact, and every answer was there
    1  a named fair question ran no rows, left facts unanswered, or carried rows
       with no judgement, so the rates cover part of the matrix
    2  the comparison does not exist: nothing readable, or no fair question
       answered in full
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tasks = _load("side_engine_tasks", "benchmark/side_engine_tasks.py")
BIBLE = tasks.BIBLE

# (question, the fact that answers it, what makes it hard for the token rank,
#  whether the label really is the fact that answers the question)
#
# The last two were labelled too loosely and the engine's NO is defensible:
# "the lighthouse keeper's son" says whose son Marcus is, not what he does for a
# living or who his family is. They stay in the run because removing a case
# after seeing the answer is how a benchmark stops meaning anything, and they
# are excluded from the totals for the same reason the model was right to
# refuse them.
QUESTIONS = [
    (
        "Does it ever rain in this town?",
        "rules/weather",
        "the answer is in the value, and 'town' points at another fact",
        True,
    ),
    (
        "How often does the boat cross the water?",
        "rules/ferry",
        "no word in common with the fact",
        True,
    ),
    (
        "Who keeps the town's records?",
        "characters/ines/role",
        "'town' gives a different fact the higher score",
        True,
    ),
    (
        "What colour are Marcus's eyes?",
        "characters/marcus/eyes",
        "the control: the token rank should win this one",
        True,
    ),
    (
        "What does Marcus do for a living?",
        "characters/marcus/role",
        "the label does not hold: the fact is not about his work",
        False,
    ),
    (
        "What is Marcus's family?",
        "characters/marcus/role",
        "the label does not hold: the fact says whose son he is",
        False,
    ),
]


def terms_of(text: str) -> list[str]:
    return [t for t in re.split(r"[^0-9a-z]+", text.lower()) if len(t) > 2]


def deterministic_rank(question: str) -> list[str]:
    """What `memory_search` returns, best first.

    Mirrors `MemoryRanking.rank` and its `textScore`, inverse document
    frequency included: a term in the key scores 3 and in the value 2, both
    scaled by `log(documents / occurrences) + 1` over these candidates. A fact
    sharing no term scores 0 and is not returned at all. Two terms the Swift
    scorer adds are not modelled here: tags, which the book's facts carry none
    of, and `record.importance`, which is raised on top of the text score and so
    can reorder what the store actually returns.
    """
    terms = terms_of(question)
    haystacks = {key: (key + " " + value).lower() for key, value in BIBLE.items()}
    frequency = {term: sum(1 for text in haystacks.values() if term in text) for term in terms}
    documents = max(1, len(BIBLE))
    scored = []
    for key, value in BIBLE.items():
        score = 0.0
        for term in terms:
            seen = frequency[term]
            if seen == 0:
                continue
            weight = math.log(documents / seen) + 1
            if term in key.lower():
                score += 3 * weight
            if term in value.lower():
                score += 2 * weight
        if score > 0:
            scored.append((score, key))
    scored.sort(key=lambda pair: -pair[0])
    return [key for _, key in scored]


def cases() -> list[dict]:
    jobs: list[dict] = []
    for question, target, why, _ in QUESTIONS:
        for key, value in BIBLE.items():
            job = tasks.job(
                "T7",
                f"QUESTION: {question}\nFACT: {key} = {value}\nCould this fact answer it?",
                "YES" if key == target else "NO",
                f"{target} <- {why}",
                authored=True,
            )
            job["question"] = question
            job["target"] = target
            job["fact"] = key
            jobs.append(job)
    return jobs


def prepare(path: Path) -> int:
    jobs = cases()
    path.write_text("\n".join(json.dumps(j) for j in jobs) + "\n", encoding="utf-8")
    print(f"{len(jobs)} cases -> {path}")
    print(f"  {len(QUESTIONS)} questions x {len(BIBLE)} facts")
    return 0


def recall(rank: list[str], target: str, k: int) -> bool:
    return target in rank[:k]


def refuse(reason: str) -> int:
    """Say why there is no comparison, print no rates, and answer 2."""
    print(f"\nNOT MEASURED: {reason}")
    return 2


def judgement(row: dict) -> bool | None:
    """The engine's answer to one fact, or None where it never gave one.

    A blank or absent completion used to read as `NO`, so a run whose engine died
    scored *worse* than one that answered and was wrong.
    """
    raw = (row.get("completion") or "").strip().upper()
    if not raw:
        return None
    return raw.startswith("YES")


def exclusions(group: list[dict], fair: bool) -> list[str]:
    """Why a fair question's rows cannot be counted, or [] when they can.

    The ranking is built over the whole BIBLE, so a fact with no judged row is
    ranked as though the engine had refused it; a short or silent group therefore
    has to be named rather than averaged in.
    """
    if not fair:
        return []
    if not group:
        return ["did not run"]
    reasons = []
    answered = {row["fact"] for row in group if judgement(row) is not None} & set(BIBLE)
    if len(answered) < len(BIBLE):
        reasons.append(f"answered {len(answered)} of {len(BIBLE)} facts")
    unjudged = sum(1 for row in group if judgement(row) is None)
    if unjudged:
        reasons.append(f"carried no judgement on {unjudged} of {len(group)} rows")
    return reasons


def score(path: Path) -> int:
    """Compare the token ranking with the engine's own YES/NO, per fair question.

    0  every fair question ran, answered every fact, and every answer was there
    1  a named fair question ran no rows, left facts unanswered, or carried rows
       with no judgement, so the rates are over part of the matrix
    2  the comparison does not exist: nothing readable, or no fair question
       answered in full
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        return refuse(f"{path} could not be read: {error}")
    rows = []
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as error:
            return refuse(f"{path} holds a line that is not json: {error}")
        if not isinstance(row, dict) or not {"question", "fact"} <= set(row):
            return refuse(f"{path} holds a row with no question or fact: {line[:60]}")
        rows.append(row)
    if not rows:
        return refuse(f"{path} holds no rows, so no question was answered")

    by_question: dict[str, list[dict]] = {}
    for row in rows:
        by_question.setdefault(row["question"], []).append(row)

    det_hits = {1: 0, 3: 0}
    t7_hits = {1: 0, 3: 0}
    fair_total = sum(1 for question in QUESTIONS if question[3])
    scored = 0
    print(f"{'question':46s} {'det@1':6s} {'det@3':6s} {'t7@1':6s} {'t7@3':6s}")
    for question, target, _, fair in QUESTIONS:
        group = by_question.get(question, [])
        if not group and not fair:
            continue
        reasons = exclusions(group, fair)
        det_rank = deterministic_rank(question)
        det = ["hit" if recall(det_rank, target, k) else "-" for k in (1, 3)]
        if reasons:
            t7 = ["-", "-"]
            mark = f"  (excluded: {', '.join(reasons)})"
        else:
            yes = {row["fact"]: judgement(row) for row in group}
            # T7 first: the facts it says could answer, then the rest, each block in
            # the store's own order.
            t7_rank = [key for key in BIBLE if yes.get(key)] + [
                key for key in BIBLE if not yes.get(key)
            ]
            t7 = ["hit" if recall(t7_rank, target, k) else "-" for k in (1, 3)]
            mark = "" if fair else "  (label does not hold, not counted)"
            if fair:
                scored += 1
                for k in (1, 3):
                    det_hits[k] += recall(det_rank, target, k)
                    t7_hits[k] += recall(t7_rank, target, k)
        print(f"{question[:44]:46s} {det[0]:6s} {det[1]:6s} {t7[0]:6s} {t7[1]:6s}{mark}")
    if not scored:
        return refuse(f"{path} answers none of the {fair_total} fair questions in full")
    print()
    print(f"scored over {scored} of {fair_total} fair questions")
    print(f"token ranking: recall@1 {det_hits[1]}/{scored}  recall@3 {det_hits[3]}/{scored}")
    print(f"side-engine:   recall@1 {t7_hits[1]}/{scored}  recall@3 {t7_hits[3]}/{scored}")
    return 1 if scored < fair_total else 0


def shares_stem(question: str, key: str, value: str) -> bool:
    """Does any question term share a four-character stem with the fact?

    A crude but sufficient line between a *lexical* miss — the answer's word is
    there in another form, `rain` against `rains` — and a *semantic* one, where
    nothing in the question resembles the fact at all (`boat` against `ferry`).
    Only the first kind is reachable by stemming or a bigger term list.
    """
    words = [w for w in re.split(r"[^0-9a-z]+", (key + " " + value).lower()) if w]
    for term in terms_of(question):
        if len(term) < 4:
            continue
        for word in words:
            if len(word) >= 4 and term[:4] == word[:4]:
                return True
    return False


def baseline() -> int:
    """What the token ranking scores with no model at all.

    Two regimes, and the difference is the whole finding. A question phrased in
    the store's own words is found perfectly — the key contains the term. A
    question that avoids the target's words is where it fails, and the failure
    is not a near miss: the top three fill up with facts that matched an
    incidental word.
    """
    mechanical = [(f"What is {key.replace('/', ' ')}?", key) for key in BIBLE]
    paraphrased = [(question, target) for question, target, _, fair in QUESTIONS if fair]
    for name, cases in (("mechanical", mechanical), ("paraphrased", paraphrased)):
        hits = {1: 0, 3: 0}
        for question, target in cases:
            rank = deterministic_rank(question)
            for k in (1, 3):
                hits[k] += recall(rank, target, k)
        print(
            f"{name:13s} n={len(cases):3d}  recall@1 {hits[1]}/{len(cases)}  "
            f"recall@3 {hits[3]}/{len(cases)}"
        )
    print("\nmechanical = the question uses the key's own words; paraphrased = it does not.")
    print("\nparaphrased, question by question — is the miss even lexical?")
    reachable = 0
    for question, target in paraphrased:
        rank = deterministic_rank(question)
        lexical = shares_stem(question, target, BIBLE[target])
        reachable += lexical
        print(
            f"  {'hit ' if target in rank[:1] else 'MISS'} "
            f"lexical={'yes' if lexical else 'no ':3s} top3={rank[:3]}"
        )
        print(f"       {question}  ->  {target}")
    misses = sum(
        1 for question, target in paraphrased if target not in deterministic_rank(question)[:1]
    )
    print(
        f"\n{misses} of {len(paraphrased)} missed at rank 1; "
        f"{reachable} of them share a stem with the fact, so at most those are "
        "reachable by stemming or a longer term list."
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", type=Path)
    ap.add_argument("--score", type=Path)
    ap.add_argument(
        "--baseline", action="store_true", help="score the token ranking alone, no model"
    )
    args = ap.parse_args()
    if args.prepare:
        return prepare(args.prepare)
    if args.score:
        return score(args.score)
    if args.baseline:
        return baseline()
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
