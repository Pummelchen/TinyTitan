#!/usr/bin/env python3.13
"""Do the side-engine's seven tasks actually work?

`docs/side-engine-tasks.md` argues that a 2B fails at composition and
succeeds at single decisions. T1 is measured -- 92% against 0 of 12 for the
composed version of the same job. This measures the other six the same way,
so the design is a finding rather than a claim.

    python3.13 benchmark/side_engine_tasks.py --prepare jobs.jsonl
    .build/.../TinyTitanBench cpu35batch <snapshot> jobs.jsonl done.jsonl
    python3.13 benchmark/side_engine_tasks.py --score done.jsonl

**Accuracy alone is not the result.** A model that always answers NO scores
well on a set that is mostly NO, and that is precisely how a small model
fails: it finds the cheap answer and gives it every time. So every task is
scored on both halves separately, and a task passes only when both are good.

Cases come from the book benchmark's own world -- its bible, its plot events
and the stores recorded from real runs -- because those have ground truth
that nobody wrote for this file. Where a case had to be authored, the task
says so, and authored cases are marked in the output.

**The status table** (AUD-279, shared with `composite_split.py`): `0` the page
measured what it was asked and every task is good on both halves; `1` it
measured and the answer is the negative one, or it is the good answer over
rows it never scored -- an unreadable row, an unlabelled one, or a row the run
left unanswered -- which are counted and named; `2` the question could not
be answered from the file handed to it -- no such file, no cases, no row
answered at all, or a T1 journal that is not there -- with the reason printed.
`--prepare` writes nothing when it refuses, because a pruned journal set
otherwise becomes a smaller case list that is reported as if it were the whole
one.

T1's ground truth is the same word-overlap check the guard's gate uses except
where `HAND_LABELS` carries the clause, and that check is the thing AUD-278 put
under a control, so a prepared file states how many of its T1 labels are
hand-made and how many the check derived.
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


book = _load("memory_book", "benchmark/memory_book.py")
guard = _load("guard_source_rate", "benchmark/guard_source_rate.py")
sim = guard.sim

ONE_WORD = "Do not explain. Do not quote. Answer with one word and nothing else."

SYSTEMS = {
    "T1": (
        "You decide whether one statement came from the person or not. "
        "Answer with exactly one word: YES or NO. YES means the person "
        "wrote it or clearly implied it. NO means it does not appear in "
        "what they wrote, however true it might be. " + ONE_WORD
    ),
    "T2": (
        "You decide whether one fact is worth keeping after this session "
        "ends. Answer with exactly one word: YES or NO. YES only for a "
        "standing fact a later session needs: a decision and its reason, a "
        "fixed attribute, a rule, a constraint, a preference, or the state "
        "of the work right now. NO for anything that reports what happened "
        "in this session instead of stating how things are: story text, "
        "narration, chapter content, a summary of what was written, a "
        "remark about the writing, a plan to write, an offer, or an "
        "acknowledgement. A fact that would only make sense to someone who "
        "read this session is NO. " + ONE_WORD
    ),
    "T3": (
        "You decide whether two statements disagree. Answer with exactly "
        "one word: YES or NO. YES means both cannot be true at once. NO "
        "means they can both be true, including when they are about "
        "different things, or when one simply says more than the other. "
        "Different wording for the same thing is NO. " + ONE_WORD
    ),
    "T4": (
        "Something has changed about one fact. You decide which kind of "
        "change it is. Answer with exactly one word: UPDATE or CONFLICT. "
        "UPDATE: the earlier value was true before and the newer one is "
        "true now, so both can be true in turn -- the world moved on. "
        "CONFLICT: a rule says this value never changes, or the two are "
        "about the same moment, so both cannot be true and one is wrong. A "
        "change of state -- a place burned, a person found, a service "
        "stopped -- is UPDATE. A change to something a rule fixes -- an "
        "eye colour under a rule that it never changes -- is CONFLICT. " + ONE_WORD
    ),
    "T5": (
        "You decide whether two facts say the same thing. Answer with "
        "exactly one word: YES or NO. YES means a reader learns nothing "
        "from the second that the first did not already tell them. NO "
        "means the second adds something, or is about something else. " + ONE_WORD
    ),
    "T6": (
        "You check one reply against one thing that is known. Answer with "
        "exactly one word: YES or NO. YES means the reply says something "
        "that cannot be true if the known fact is true. NO means it "
        "agrees, or does not touch on it at all. Silence is not a "
        "contradiction. " + ONE_WORD
    ),
    "T7": (
        "You decide whether one stored fact could answer one question. "
        "Answer with exactly one word: YES or NO. YES means the fact "
        "contains the answer, or part of it. NO means it does not, even "
        "if it is about the same subject. " + ONE_WORD
    ),
}

# The book's fixed attributes, as key/value pairs the model would really see.
BIBLE = {
    "characters/marcus/eyes": "grey",
    "characters/ines/eyes": "green",
    "characters/halvorsen/eyes": "brown",
    "characters/rosa/eyes": "hazel",
    "characters/aldo/eyes": "blue",
    "setting/town": "Ashgrove",
    "rules/weather": "it never rains",
    "rules/ferry": "runs only on Sundays",
    "characters/marcus/role": "the lighthouse keeper's son",
    "characters/ines/role": "the town archivist",
}
# Plot events: a state that legitimately changes, which is what separates an
# update from a conflict.
EVENTS = {
    "state/inn": ("standing", "burned to the ground"),
    "state/tomas": ("missing", "found alive in the lighthouse"),
    "state/ferry": ("running", "stopped running for good"),
}

# The rule that makes an eye-colour change a CONFLICT rather than an UPDATE.
# Supplied with the T4 cases because no model can know it from the two
# statements, and a caller must supply the same kind of rule from the store.
RULE = "RULE: eye colour is fixed and must never change."


# Two T1 labels the grounding check gets wrong. It counts a clause as the
# person's when half its words appear in what they wrote, and these borrow
# the words without the meaning: "archive" and "Marcus" are in the bible,
# spreading the archive and keeping promises are not, and nobody said
# chapters 61-70 were written. Every T1 clause was read by hand against the
# person's text on 2026-09-11; these two are the only labels that change.
# Two borderline ones stand as the check set them: "in hiding for years"
# embellishes what was said, and "anyone_left_ashgrove false" follows from
# a rule but was not said.
HAND_LABELS = {
    "characters/ines = spreads the archive openly and keeps promises to Marcus.": "NO",
    "chapters.photo/ch61-70 = Chapters 61-70 written": "NO",
}


def truth_of(row: dict) -> str:
    """A row's ground truth, with the hand corrections applied, so results
    recorded before a correction are scored against it too."""
    if row["task"] == "T1":
        statement = row["prompt"].rpartition("STATEMENT: ")[2].split("\n")[0]
        return HAND_LABELS.get(statement, row["truth"])
    return row["truth"]


def labelled(row: dict) -> str | None:
    """`truth_of` on a row that may not carry the fields it reads: None means
    this row cannot be scored either way, which the page has to count rather
    than crash over or drop from its denominator."""
    try:
        return truth_of(row)
    except KeyError:
        return None


def job(
    task: str, prompt: str, truth: str, note: str, authored: bool = False, system: str | None = None
):
    return {
        "chat": True,
        "system": system or SYSTEMS[task],
        "prompt": prompt,
        "max": 8,
        "task": task,
        "truth": truth,
        "note": note,
        "authored": authored,
    }


T1_LABELS = ("guard-step0", "guard-step0-ornith", "guard-confirm")


def journals() -> tuple[list[tuple[str, Path]], list[str]]:
    """The recorded runs T1 reads its clauses from, and the labels that have no
    journal. A missing label is returned rather than skipped: `--prepare` over
    two of the three is a smaller case set printed as if it were the whole one,
    and T1 is the task this file's headline number comes from."""
    paths: list[tuple[str, Path]] = []
    missing: list[str] = []
    for label in T1_LABELS:
        journal = guard.journal_for(label)
        if journal is None:
            missing.append(label)
            continue
        paths.append((label, journal))
    return paths, missing


def refuse(reason: str) -> int:
    print(f"\nNOT MEASURED: {reason}")
    return 2


def cases() -> list[dict]:
    jobs, _ = jobs_and_missing()
    return jobs


def jobs_and_missing() -> tuple[list[dict], list[str]]:
    jobs: list[dict] = []
    sim.user_text(10)

    # T1: every clause of every composite the recorded runs produced, with
    # the person's own words as the reference. Ground truth comes from the
    # same grounding check the guard's gate uses, except where HAND_LABELS
    # carries the clause, and the prepared file says which is which -- the
    # check is the thing AUD-278 put under a control.
    paths, missing = journals()
    for label, journal in paths:
        for fact in guard.facts(journal):
            if not fact["user_asserted"] or ";" not in fact["value"]:
                continue
            for clause in [c.strip() for c in fact["value"].split(";") if c.strip()]:
                words = guard.significant(clause)
                if not words:
                    continue
                stems = guard.stems(guard.significant(sim.user_text(fact["session"])))
                truth = len({w for w in words if guard.stem(w) in stems}) / len(words)
                key = f"{fact['address']} = {clause}"
                case = job(
                    "T1",
                    f"WHAT THE PERSON WROTE:\n{sim.user_text(fact['session'])}\n\n"
                    f"STATEMENT: {fact['address']} = {clause}\n"
                    f"Did the person state this?",
                    HAND_LABELS.get(key, "YES" if truth >= 0.5 else "NO"),
                    f"{fact['address']} / {label}",
                )
                case["label_source"] = "hand" if key in HAND_LABELS else "check"
                jobs.append(case)

    # T2: durable against not. The positives are the bible's own facts; the
    # negatives are lines of the novel the model wrote, which are exactly
    # what must not be stored.
    for key, value in BIBLE.items():
        jobs.append(job("T2", f"FACT: {key} = {value}\nKeep it?", "YES", key))
    for index, line in enumerate(
        [
            "Chapter 12: Ines turned the brittle pages and found the photograph.",
            "I will write the next ten chapters now.",
            "Chapter 34: the inn burned as the tide came in.",
            "Let me re-read chapter 8 before continuing.",
            "The prose in chapter 20 could be tightened.",
            "Chapter 51: Marcus walked to the harbour in the rain.",
            "Here are chapters 41 to 50 as requested.",
            "I have finished the section you asked for.",
            "Chapter 63: the certificate lay in the drawer, unsigned.",
            "That completes the ten chapters.",
        ],
        start=1,
    ):
        jobs.append(
            job(
                "T2",
                f"FACT: session/note{index} = {line}\nKeep it?",
                "NO",
                f"narration {index}",
                authored=True,
            )
        )

    # T3: a contradiction is the same key with an incompatible value. A
    # non-contradiction is two different facts, or the same fact reworded --
    # the case a fold-equality check gets right and a careless model does not.
    keys = list(BIBLE)
    for key in keys[:5]:
        wrong = "hazel" if BIBLE[key] != "hazel" else "grey"
        jobs.append(
            job(
                "T3",
                f"A: {key} = {BIBLE[key]}\nB: {key} = {wrong}\nDo A and B disagree?",
                "YES",
                key,
            )
        )
    for first, second in zip(keys[:5], keys[5:10], strict=False):
        jobs.append(
            job(
                "T3",
                f"A: {first} = {BIBLE[first]}\nB: {second} = {BIBLE[second]}\nDo A and B disagree?",
                "NO",
                f"{first} vs {second}",
            )
        )

    # T4: the plot events really are updates -- the person said the inn
    # burned. An eye colour changing is a conflict, because the bible says it
    # never does -- and that is not knowable from the two statements alone, so
    # the rule the store holds is supplied with them, as a caller must
    # (`SideEngineJudgement.supersession`'s `rule`). Without it the CONFLICT
    # half is unanswerable and the model is guessing.
    for key, (before, after) in EVENTS.items():
        jobs.append(
            job(
                "T4",
                f"{RULE}\nEARLIER: {key} = {before}\nNOW: {key} = {after}\nWhich is it?",
                "UPDATE",
                key,
            )
        )
    for key in list(BIBLE)[:3]:
        jobs.append(
            job(
                "T4",
                f"{RULE}\nEARLIER: {key} = {BIBLE[key]}\nNOW: {key} = hazel\nWhich is it?",
                "CONFLICT",
                key,
            )
        )

    # T5: the same fact reworded against two different facts.
    for key, value in list(BIBLE.items())[:4]:
        subject = key.split("/")[1]
        jobs.append(
            job(
                "T5",
                f"A: {key} = {value}\n"
                f"B: notes/{subject} = {subject}'s eyes are {value}\n"
                f"Same fact?",
                "YES",
                key,
                authored=True,
            )
        )
    for first, second in zip(keys[:4], keys[4:8], strict=False):
        jobs.append(
            job(
                "T5",
                f"A: {first} = {BIBLE[first]}\nB: {second} = {BIBLE[second]}\nSame fact?",
                "NO",
                f"{first} vs {second}",
            )
        )

    # T6: a reply that contradicts a known fact, against one that does not
    # touch it. Silence must not read as contradiction -- the failure that
    # would make a checker fire on every reply.
    for key, value in list(BIBLE.items())[:4]:
        subject = key.split("/")[1]
        jobs.append(
            job(
                "T6",
                f"KNOWN: {key} = {value}\n"
                f"REPLY: {subject.title()} looked up, hazel eyes catching "
                f"the light.\nDoes the reply contradict what is known?",
                "YES" if value != "hazel" else "NO",
                key,
                authored=True,
            )
        )
    for key, value in list(BIBLE.items())[:4]:
        jobs.append(
            job(
                "T6",
                f"KNOWN: {key} = {value}\n"
                f"REPLY: The ferry did not come that morning, and the "
                f"harbour stayed empty.\n"
                f"Does the reply contradict what is known?",
                "NO",
                key,
                authored=True,
            )
        )

    # T7: the quiz's own questions against the key that answers them, and
    # against a key about the same person that does not.
    asks = {
        "characters/marcus/eyes": "What colour are Marcus's eyes?",
        "setting/town": "Which town is the story set in?",
        "rules/ferry": "When does the ferry run?",
        "characters/ines/role": "What does Ines do?",
    }
    for key, question in asks.items():
        jobs.append(
            job(
                "T7",
                f"QUESTION: {question}\nFACT: {key} = {BIBLE[key]}\nCould this fact answer it?",
                "YES",
                key,
            )
        )
    wrong = {
        "characters/marcus/eyes": "characters/marcus/role",
        "setting/town": "rules/weather",
        "rules/ferry": "characters/aldo/eyes",
        "characters/ines/role": "characters/ines/eyes",
    }
    for key, question in asks.items():
        other = wrong[key]
        jobs.append(
            job(
                "T7",
                f"QUESTION: {question}\nFACT: {other} = {BIBLE[other]}\nCould this fact answer it?",
                "NO",
                f"{key} vs {other}",
            )
        )
    return jobs, missing


# Two things people believe about prompting, neither of them measured here
# before: that telling a model the stakes are high makes it try harder, and
# that a model can say how sure it is. Both are cheap to test on T1, which
# has 38 cases with ground truth, and both would be worth having.
#
# Stakes framing is folklore. It might work; it might also just make the
# model hedge, which on a binary question means moving toward whichever
# answer feels safer. The way to find out is to run the same cases three
# ways.
VARIANTS = {
    "plain": "",
    "stakes": (
        " This is a production system. A wrong answer corrupts a "
        "person's saved memory, and they will not know it happened. "
        "Be certain before you answer."
    ),
    "confidence": (
        " After your one-word answer, add a space and a "
        "confidence percentage from 0 to 100, like: YES 90"
    ),
}


# The second draft, from reading what the first got wrong. The 2B answered
# one word every time on T2, T4, T5 and T6, and missed nine T1 clauses
# written as shorthand. What those have in common is form, not difficulty:
# the facts reached it as key paths and name_attribute pairs, and it could
# not match them against prose. T3 and T7, which never ask it to, passed.
# So v2 changes how a fact arrives -- as a sentence -- and spells out each
# boundary it collapsed on. T3, T6 and T7 keep their wording.
SYSTEMS_V2 = {
    **SYSTEMS,
    "T1": (
        "You decide whether one statement came from the person or not. "
        "Answer with exactly one word: YES or NO. YES means the person "
        "wrote it or clearly implied it, however the statement is "
        "written: shorthand such as name_attribute value counts. NO means "
        "it does not appear in what they wrote, however true it might "
        "be. " + ONE_WORD
    ),
    "T2": (
        "You decide whether one note is worth keeping after this session "
        "ends. Answer with exactly one word: YES or NO. YES for decisions "
        "and the reasons behind them, fixed attributes, rules, "
        "constraints, and current state. NO for a line of the story "
        "itself, for anything about the writing -- what was written, what "
        "comes next, what could be better -- and for conversation, "
        "reasoning, code, and anything true only right now. " + ONE_WORD
    ),
    "T4": (
        "Something has changed about one fact. You decide which kind of "
        "change it is. Answer with exactly one word: UPDATE or CONFLICT. "
        "UPDATE means the world moved on and the newer one is the current "
        "state. CONFLICT means the two cannot both have been true, and "
        "one of them is wrong. A change the rule forbids is always "
        "CONFLICT. " + ONE_WORD
    ),
    "T5": (
        "You decide whether two facts say the same thing. Answer with "
        "exactly one word: YES or NO. YES means a reader learns nothing "
        "from the second that the first did not already tell them, "
        "however differently it is worded. NO means the second adds "
        "something, or is about something else. " + ONE_WORD
    ),
}


def sentence(key: str, value: str) -> str:
    """A stored fact without its filing path: `characters/marcus/eyes =
    grey` becomes "Marcus's eyes: grey." Mechanical on purpose -- the engine
    would have to render every key it ever stores this way, with no model
    in the loop and no hand-written prose."""
    value = value.rstrip(".")
    namespace, *parts = [p.replace("_", " ") for p in key.split("/")]
    if namespace in ("session", "notes"):
        return value[:1].upper() + value[1:] + "."  # already prose
    if len(parts) >= 2:
        return f"{parts[0].title()}'s {' '.join(parts[1:])}: {value}."
    return f"{parts[0].capitalize()}: {value}."


def as_sentences(prompt: str) -> str:
    """The prompt with each `LABEL: key = value` line rendered by `sentence`,
    so v2 asks exactly v1's cases and only the form changes. The upper-case
    label test keeps the person's own text, which has `name: value` lines of
    its own, untouched."""
    lines = []
    for line in prompt.split("\n"):
        label, colon, rest = line.partition(": ")
        key, equals, value = rest.partition(" = ")
        if colon and equals and "/" in key and label.isupper():
            line = f"{label}: {sentence(key, value)}"
        lines.append(line)
    return "\n".join(lines)


def v2_cases() -> list[dict]:
    jobs = []
    for case in cases():
        prompt = as_sentences(case["prompt"])
        if case["task"] == "T4" and not prompt.startswith("RULE:"):
            prompt = f"{RULE}\n{prompt}"
        jobs.append(dict(case, prompt=prompt, system=SYSTEMS_V2[case["task"]], variant="v2"))
    return jobs


def confidence_cases() -> list[dict]:
    """T1's cases again, under each variant, so the three are comparable."""
    jobs = []
    for case in cases():
        if case["task"] != "T1":
            continue
        for name, extra in VARIANTS.items():
            variant = dict(case)
            variant["system"] = case["system"] + extra
            variant["variant"] = name
            variant["max"] = 12 if name == "confidence" else 8
            jobs.append(variant)
    return jobs


def score_variants(path: Path) -> int:
    if not path.exists():
        return refuse(f"no such file: {path}")
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows:
        return refuse(f"{path} holds no variant cases, so the three are not compared")
    answered = [row for row in rows if (row.get("completion") or "").strip()]
    if not answered:
        return refuse(
            f"no completion in any of the {len(rows)} row(s): a 0% arm is not an answer, "
            "and the comparison is between answers"
        )
    groups: dict[str, list] = {}
    for row in rows:
        groups.setdefault(row.get("variant", "plain"), []).append(row)

    print(f"{'variant':12s} {'n':>3s} {'correct':>8s}   {'YES cases':>12s} {'NO cases':>10s}")
    absent: list[str] = []
    for name in VARIANTS:
        group = groups.get(name) or []
        if not group:
            absent.append(name)
            print(f"{name:12s} {0:3d} {'--':>8s}   {'--':>12s} {'--':>10s}")
            continue
        halves: dict[str, list[int]] = {}
        for row in group:
            answer = (row.get("completion") or "").strip().upper()
            word = answer.split()[0].strip(".,:;\"'") if answer else ""
            entry = halves.setdefault(truth_of(row), [0, 0])
            entry[0] += word == truth_of(row)
            entry[1] += 1
        correct = sum(v[0] for v in halves.values())
        total = sum(v[1] for v in halves.values())
        yes = halves.get("YES", [0, 0])
        no = halves.get("NO", [0, 0])
        print(
            f"{name:12s} {total:3d} {100 * correct / total:7.0f}%   "
            f"{yes[0]:>5d}/{yes[1]:<6d} {no[0]:>5d}/{no[1]:<4d}"
        )

    # Calibration: a confidence figure is only worth having if being sure
    # means being right. If the model says 90 whether it is right or wrong,
    # it is a decoration.
    group = groups.get("confidence") or []
    buckets: dict[str, list[int]] = {}
    unparsed = 0
    for row in group:
        answer = (row.get("completion") or "").strip().upper()
        parts = answer.replace("%", "").split()
        if len(parts) < 2 or not parts[1].isdigit():
            unparsed += 1
            continue
        hit = parts[0].strip(".,:;") == truth_of(row)
        value = int(parts[1])
        band = (
            "100"
            if value >= 100
            else ("90-99" if value >= 90 else ("70-89" if value >= 70 else "under 70"))
        )
        entry = buckets.setdefault(band, [0, 0])
        entry[0] += hit
        entry[1] += 1
    decoration = False
    if buckets:
        print("\ncalibration -- is being sure the same as being right?")
        print(f"  {'stated':10s} {'n':>3s} {'actually right':>15s}")
        for band in ("100", "90-99", "70-89", "under 70"):
            if band not in buckets:
                continue
            hit, seen = buckets[band]
            print(f"  {band:10s} {seen:3d} {100 * hit / seen:14.0f}%")
        print(f"  ({unparsed} answers carried no readable number)")
        spread = [100 * v[0] / v[1] for v in buckets.values() if v[1] >= 3]
        if len(spread) >= 2 and max(spread) - min(spread) < 10:
            print("  the bands do not separate: the number is a decoration.")
            decoration = True
    print(f"\nvariants scored {len(VARIANTS) - len(absent)}/{len(VARIANTS)}")
    if absent:
        print(
            f"CONTESTED: {', '.join(absent)} answer(ed) nothing, so the arms above are not "
            "the comparison of three the page claims"
        )
        return 1
    return 1 if decoration else 0


def publish(path: Path, jobs: list[dict]) -> int:
    """Write a jobs file, or refuse and write nothing. A label whose journal is
    gone shrinks T1 silently, and the operator then scores the shrunken set."""
    _, missing = journals()
    if missing:
        return refuse(
            f"no journal for {', '.join(missing)}, so T1 is built from fewer runs than the "
            "page describes"
        )
    t1 = [case for case in jobs if case["task"] == "T1"]
    if not t1:
        return refuse("the journals that were read hold no composite fact, so T1 has no cases")
    path.write_text("\n".join(json.dumps(case) for case in jobs) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    for case in jobs:
        counts[case["task"]] = counts.get(case["task"], 0) + 1
    print(f"{len(jobs)} cases -> {path}")
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    hand = sum(1 for case in t1 if case.get("label_source") == "hand")
    print(f"  T1 ground labels: {hand} hand-labelled, {len(t1) - hand} from the grounding check")
    return 0


def prepare(path: Path) -> int:
    return publish(path, cases())


def score(path: Path) -> int:
    if not path.exists():
        return refuse(f"no such file: {path}")
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows:
        return refuse(f"{path} holds no cases, so no task was answered")
    tasks: dict[str, list] = {}
    unread: list[dict] = []
    unlabelled: list[dict] = []
    unanswered: list[dict] = []
    for row in rows:
        if "task" not in row:
            unread.append(row)
            continue
        if labelled(row) is None:
            unlabelled.append(row)
            continue
        if not (row.get("completion") or "").strip():
            unanswered.append(row)
            continue
        tasks.setdefault(row["task"], []).append(row)

    scored = sum(len(group) for group in tasks.values())
    print(f"{scored} of {len(rows)} rows scored")
    if not scored:
        if unanswered:
            return refuse(
                f"no completion in any of the {len(rows)} row(s): a 0% task is not an "
                "answer, and this table is about answers"
            )
        if unlabelled:
            return refuse(
                f"no row carried a ground label ({len(unlabelled)} row(s) could not be read)"
            )
        return refuse(f"no row carried a task ({len(unread)} row(s) could not be read)")

    print(f"{'task':5s} {'n':>3s} {'correct':>8s}   per-answer accuracy      unparseable")
    failures = 0
    for task in sorted(tasks):
        group = tasks[task]
        by_truth: dict[str, list[int]] = {}
        unparseable = 0
        for row in group:
            answer = (row.get("completion") or "").strip().upper()
            answer = answer.split()[0].strip(".,:;\"'") if answer else ""
            expected = truth_of(row)
            legal = {"YES", "NO"} if expected in ("YES", "NO") else {"UPDATE", "CONFLICT"}
            if answer not in legal:
                unparseable += 1
                by_truth.setdefault(expected, [0, 0])[1] += 1
                continue
            hit = answer == expected
            entry = by_truth.setdefault(expected, [0, 0])
            entry[0] += hit
            entry[1] += 1
        correct = sum(v[0] for v in by_truth.values())
        total = sum(v[1] for v in by_truth.values())
        halves = "  ".join(f"{k}: {v[0]}/{v[1]}" for k, v in sorted(by_truth.items()))
        # Both halves must be good. One-sided accuracy is the shape of a
        # model answering the same word every time.
        worst = min((v[0] / v[1]) for v in by_truth.values() if v[1])
        mark = " " if worst >= 0.7 else "*"
        failures += worst < 0.7
        print(
            f"{task:5s} {total:3d} {100 * correct / total:7.0f}%{mark}  {halves:28s} "
            f"{unparseable:>3d}"
        )
    print(f"  {'unread':28s} {len(unread):4d}  (no task on the row)")
    print(f"  {'unlabelled':28s} {len(unlabelled):4d}  (no ground label to score against)")
    print(f"  {'unanswered':28s} {len(unanswered):4d}  (the run gave no answer for the row)")
    print(
        "\n* one half below 70%: the model is not reading the question, "
        "whatever the overall figure says."
    )
    if failures:
        print(f"{failures} task(s) not ready.")
    else:
        print("every task good on both halves.")
    if unanswered:
        print(
            f"CONTESTED: {len(unanswered)} of {len(rows)} rows never answered, so the table "
            "above is over the rows that did"
        )
        return 1
    remainder = len(unread) + len(unlabelled)
    if remainder:
        print(
            f"CONTESTED: {remainder} of {len(rows)} rows were never scored, so the table "
            "describes the rest"
        )
        return 1
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", type=Path)
    ap.add_argument("--score", type=Path)
    ap.add_argument(
        "--prepare-variants",
        type=Path,
        help="T1 again under stakes framing and with a confidence "
        "figure, to test two beliefs about prompting",
    )
    ap.add_argument("--score-variants", type=Path)
    ap.add_argument(
        "--prepare-v2",
        type=Path,
        help="every case again with facts as sentences and the "
        "second-draft prompts; score with --score",
    )
    args = ap.parse_args()
    if args.prepare:
        return prepare(args.prepare)
    if args.prepare_v2:
        return publish(args.prepare_v2, v2_cases())
    if args.prepare_variants:
        return publish(args.prepare_variants, confidence_cases())
    if args.score_variants:
        return score_variants(args.score_variants)
    if args.score:
        return score(args.score)
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
