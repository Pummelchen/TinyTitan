#!/usr/bin/env python3.13
"""How often does the extraction mislabel who said something?

The memory guard rests on one bit per fact: did the *person* assert this, or
did the model derive it? With the guard on, a fact labelled `user` can no
longer be silently superseded. That is protection when the label is right
and damage when it is wrong -- a fact the model invented and then had
protected is worse than today's behaviour, not merely different. So the
label's error rate has to be measured before the guard is switched on
anywhere, and this is that measurement.

It reads a recorded book run's journal, which holds every fact the engine
wrote together with the provenance author the label became, and scores each
fact claiming the person's authority
against what the person actually put in front of the model in that session:
the story bible plus the plot events delivered up to that point
(`memory_sim.user_text`). A fact claiming to come from the user whose
substance is nowhere in the user's own words is a mislabel.

    python3.13 benchmark/guard_source_rate.py                    # newest run
    python3.13 benchmark/guard_source_rate.py --label guard-step0
    python3.13 benchmark/guard_source_rate.py --show             # each fact

**This finds candidates; it does not decide.** Word overlap cannot judge a
fact whose claim lives in its key and whose value is a boolean --
`continuity/marcus_knows_photo = false` is the bible's hard rule and neither
word appears in what the person wrote. Scored on the value alone it reported
nine mislabels on a run that had none; scored on key and value together it
reported six across three runs, and hand review found all six correct. The
flags are worth reading and the count is not worth trusting.

So the gate is: run this, read every flag, and record the verdict. The list
is small -- two to nine per run -- which is what makes that practical.

The gate itself, from `docs/plan-memory-guard-and-shadow.md`: under 5%
mislabelled, and no invented fact labelled `user` at all.

The answer is three statuses, the ones this tree's other drivers use (AUD-273
through AUD-277), because the page has three trust conditions of its own and a
run that fails one must not read as a pass:

    0  every fact carrying authority was scored, the control discriminated, and
       nothing is flagged
    1  it measured, and facts wearing the person's label are flagged -- or a
       fact carrying authority could not be compared at all
    2  the gate cannot be judged, with the reason named: no recorded run, no
       journal, no facts, nothing scoreable, a `--threshold` outside the domain
       where the comparison means something, or a control that does not
       discriminate

`3` is not used: this driver starts no process, so it has no guard to answer 3.

One exception is worth naming, because the control rule looks like it should
catch it: a run where *nothing* grounds -- every fact flagged, the control at
zero against the person's zero -- is a `1` with a caveat, not a `2`. That is the
worst answer the gate can get and every fact on it is named; reporting it as
"this run proved nothing" would turn a scandal into an inconclusive run.

The scoring is deliberately generous to the model. A fact counts as
grounded if its *distinctive* words -- the ones carrying the claim, not the
scaffolding -- appear in the user's text. Paraphrase passes; only a value
the user never said anywhere fails. That direction is the safe one: it
under-reports mislabels, so a rate this measures as low could be lower still
but never higher.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / ".build/benchmark-logs"

# The bar the plan's own hand-checked numbers were taken at. Read per run, and
# refused outside the domain where the comparison can be failed: AUD-278.
DEFAULT_THRESHOLD = 0.5

_spec = importlib.util.spec_from_file_location("memory_sim", ROOT / "benchmark/memory_sim.py")
sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sim)

# Words that carry no claim. A fact grounded only in these is grounded in
# nothing, so they are removed before the overlap is taken.
SCAFFOLD = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "in",
    "on",
    "at",
    "of",
    "to",
    "and",
    "or",
    "but",
    "that",
    "this",
    "it",
    "its",
    "as",
    "by",
    "for",
    "with",
    "from",
    "not",
    "no",
    "be",
    "been",
    "being",
    "he",
    "she",
    "they",
    "his",
    "her",
    "their",
    "who",
    "which",
    "what",
    "when",
    "where",
    "chapter",
    "character",
    "story",
    "book",
    "novel",
    "user",
    "assistant",
    "model",
    "fact",
    "facts",
    "memory",
    "session",
}
WORD = re.compile(r"[a-z0-9]+")


def significant(text: str) -> set[str]:
    return {w for w in WORD.findall(text.lower()) if w not in SCAFFOLD and len(w) > 2}


# Inflection, not paraphrase. The first two candidates this script produced
# were both of this shape and both wrong: the bible says "Rosa, hazel eyes"
# and the extraction wrote "hazels"; the user's session-4 event says "Rosa's
# inn burns to the ground" and the extraction wrote "burned". Both facts are
# the user's, and an exact-token comparison called them inventions.
#
# A blunt prefix would fix those and blur real differences with them, so this
# strips the English suffixes that carry no claim instead. It is deliberately
# generous in one direction only: a value that survives it has no
# morphological relative anywhere in what the person wrote, which is what an
# invention looks like. Everything it flags is printed, because "no invented
# fact labelled user" is a claim a person checks, not a number.
SUFFIXES = ("ing", "ed", "es", "s", "d")


def atomic(value: str) -> bool:
    """Whether a value looks like one fact rather than several.

    A mirror of `MemoryRecord.isAtomic`, because this measures what ships:
    the store refuses to let a composite value carry the person's authority,
    so a composite is not a candidate for mislabelling. If the two ever
    disagree, this number stops describing the guard.
    """
    if ";" in value:
        return False
    if len(value) > 120:
        return False
    return re.search(r"[.!?]\s+\S", value) is None


def stem(word: str) -> str:
    for suffix in SUFFIXES:
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def stems(words: set[str]) -> set[str]:
    return {stem(w) for w in words}


def newest_label() -> str | None:
    runs = sorted(
        LOGS.glob("memval-scratch-*/book-auto-r*"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not runs:
        return None
    return runs[0].parent.name.removeprefix("memval-scratch-")


def journal_for(label: str) -> Path | None:
    paths = [
        Path(p)
        for p in glob.glob(str(LOGS / f"memval-scratch-{label}/book-auto-r*/tinytitan/*/*.ndjson"))
    ]
    real = [p for p in paths if "_global" not in p.name and p.stat().st_size > 0]
    return max(real, key=lambda p: p.stat().st_size) if real else None


def facts(journal: Path) -> list[dict]:
    """Every fact the engine wrote, in order, with its provenance flag.

    Sessions are numbered by the order they first wrote, the order the
    harness ran them, so a fact can be scored against what the person had
    said by that point and not against the whole book.
    """
    written = []
    for line in journal.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        item = record.get("memory", {}).get("_0")
        if not item:
            continue
        provenance = item.get("provenance") or {}
        written.append(
            {
                "session_id": provenance.get("sessionID", ""),
                "address": f"{item['namespace'].removeprefix('k.')}/{item['key']}",
                "value": str(item.get("value", "")),
                # The record's `isUserAsserted` is not stored as a field: the
                # store translates it into the provenance author, which is what
                # the guard reads back on the next write. So that is what has to
                # be scored -- reading the record field here would silently find
                # nothing and report a perfect run.
                "user_asserted": (provenance.get("author") == "user"),
            }
        )
    order: list[str] = []
    for fact in written:
        if fact["session_id"] not in order:
            order.append(fact["session_id"])
    index = {session: number for number, session in enumerate(order, start=1)}
    for fact in written:
        fact["session"] = index[fact["session_id"]]
    return written


def threshold_refusal(value: float) -> str | None:
    """Why this grounding bar cannot produce an answer, or None if it can.

    AUD-278. The comparison is `overlap >= threshold`, so a threshold at or
    below zero is cleared by a fact that shares no word with anything the person
    wrote, and one above 1 is unreachable by a fact that shares every word.
    Either way the run answers a question about its own argument, and the flag
    list the procedure is built around reading is either empty or total.
    """
    if value <= 0:
        return (
            f"--threshold {value:g} grounds every fact, since an overlap of 0 "
            "already clears it; this run cannot report a mislabel whatever the "
            "model wrote"
        )
    if value > 1:
        return (
            f"--threshold {value:g} is above the 1.0 an overlap can reach, so "
            "every fact is flagged and no flag means anything"
        )
    return None


def gate_verdict(
    claimed: int,
    scored: int,
    flagged: int,
    control_total: int,
    control_grounded: int,
) -> tuple[list[str], int]:
    """The page's answer, on the three statuses its trust conditions imply.

    A rate is only a rate over the facts that were compared, and the control is
    what says the scorer can tell the person's words from the model's: the
    docstring's own rule is that if the two rates are close, "this measurement
    is not measuring anything". Saying that in a line and returning 0 anyway is
    AUD-278, so the rule is what decides the status here.
    """
    lines: list[str] = []
    if scored == 0:
        lines.append(
            f"NOT MEASURED: nothing the scorer could compare -- all {claimed} "
            "fact(s) carrying authority have no distinctive words, so no "
            "comparison was made and none is being certified"
        )
        return lines, 2
    if control_total == 0:
        lines.append(
            "NOT MEASURED: no model-labelled fact to control against, so the "
            "scorer's ability to tell the person's words from the model's is "
            "unproven on this run"
        )
        return lines, 2
    user_grounded = (scored - flagged) / scored
    control_rate = control_grounded / control_total
    # The control discriminates when fewer of the model's facts ground than of the
    # person's. A higher rate means the grounded bit is inverted or noise; an equal
    # one means it is not carrying the label. Zero against zero is the exception, and
    # the reason it has to be: that is a run where every authority-carrying fact was
    # flagged, the worst answer this gate can get. Downgrading it to "nothing proved"
    # would turn a scandal into an inconclusive run, and every fact is named, so the
    # operator can read it.
    if control_rate > user_grounded or (control_rate == user_grounded and user_grounded > 0):
        lines.append(
            f"NOT MEASURED: the control does not discriminate -- {control_grounded}/"
            f"{control_total} of the model's own facts ({control_rate:.0%}) ground as "
            f"the person's against {user_grounded:.0%} of the facts labelled user, so "
            "the grounded bit is not carrying the label and neither number means anything"
        )
        return lines, 2
    if flagged:
        lines.append(
            f"CONTESTED: {flagged} fact(s) wearing the person's label that the "
            "person never said -- read each one; the gate is the list, not this number"
        )
        if user_grounded == 0:
            lines.append(
                "  no fact grounded at all in this run, so the other reading is that "
                "the journal's vocabulary does not match the book's -- read the list "
                "before treating the rate as the model inventing"
            )
    if scored != claimed:
        lines.append(
            f"CONTESTED: {claimed - scored} of the {claimed} facts carrying "
            "authority were never compared to the person's words, so the clean "
            "answer covers the rest"
        )
    if flagged or scored != claimed:
        return lines, 1
    return lines, 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default=None)
    ap.add_argument("--show", action="store_true")
    ap.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="fraction of a value's distinctive words that must appear in the user's own text",
    )
    args = ap.parse_args()

    reason = threshold_refusal(args.threshold)
    if reason is not None:
        print(f"NOT MEASURED: {reason}")
        return 2

    label = args.label or newest_label()
    if label is None:
        print(f"NOT MEASURED: no recorded book runs under {LOGS}")
        return 2
    journal = journal_for(label)
    if journal is None:
        print(f"NOT MEASURED: no journal for label {label} under {LOGS}")
        return 2

    written = facts(journal)
    if not written:
        print(f"NOT MEASURED: {journal} holds no facts")
        return 2

    def score(fact: dict) -> float | None:
        """How much of a fact is traceable to the person's own words.

        The **address and the value together**, because a fact is both. The
        first version of this scored the value alone and reported nine
        mislabels on a run that had none: every one was a boolean whose key
        carries the claim -- `rules/marcus_must_not_learn_photo_before_
        chapter_60 = true` is the person's rule from the story bible, and
        "true" appears nowhere in what they wrote. Scoring half a fact
        measures nothing.
        """
        said = stems(significant(sim.user_text(fact["session"])))
        words = significant(fact["address"].replace("/", " ").replace("_", " "))
        words |= significant(fact["value"])
        if not words:
            return None
        return len({w for w in words if stem(w) in said}) / len(words)

    # Only the facts that would actually carry authority. A composite value
    # cannot have one source, so the store demotes it before the guard ever
    # sees it -- and scoring demoted facts would report a risk that is not
    # taken.
    labelled = [f for f in written if f["user_asserted"]]
    claimed = [f for f in labelled if atomic(f["value"])]
    demoted = len(labelled) - len(claimed)
    mislabelled, grounded = [], []
    for fact in claimed:
        overlap = score(fact)
        if overlap is None:
            continue
        fact["overlap"] = overlap
        (grounded if overlap >= args.threshold else mislabelled).append(fact)

    # The control. A scorer generous enough to ground anything would report a
    # perfect run whatever the model did, so the facts labelled *model* are
    # put through the same test: they are the model's own output, and most of
    # them should not be traceable to the person's words. If the two rates
    # are close, this measurement is not measuring anything.
    derived = [f for f in written if not f["user_asserted"]]
    derived_scores = [s for s in (score(f) for f in derived) if s is not None]
    derived_grounded = sum(1 for s in derived_scores if s >= args.threshold)

    print(f"{label}: {journal.name}\n")
    print(f"  facts written                {len(written):5d}")
    print(
        f"  labelled user                {len(labelled):5d}  ({len(labelled) / len(written):.0%})"
    )
    print(f"  labelled model               {len(written) - len(labelled):5d}")
    print(f"  demoted, value not atomic    {demoted:5d}  (a composite cannot have one source)")
    print(f"  carrying authority           {len(claimed):5d}")
    if not claimed:
        print(
            "\n  nothing was labelled user; the guard would never fire, "
            "and the gate cannot be judged from this run"
        )
        return 2

    scored = len(grounded) + len(mislabelled)
    print(
        f"  of those, scored {scored} of {len(claimed)}  "
        f"({len(claimed) - scored} carried no distinctive words)"
    )
    if scored:
        print(
            f"  of those, mislabelled        {len(mislabelled):5d}  "
            f"({len(mislabelled) / scored:.1%})"
        )
    if derived_scores:
        print(
            f"\n  control: model-labelled facts that would also score as "
            f"the user's: {derived_grounded}/{len(derived_scores)} "
            f"({derived_grounded / len(derived_scores):.0%})"
        )
        print("  (a rate near the user rate would mean the test does not discriminate)")

    lines, status = gate_verdict(
        len(claimed), scored, len(mislabelled), len(derived_scores), derived_grounded
    )
    for line in lines:
        print(f"  {line}")
    if scored:
        print(
            f"\n  {len(mislabelled)} candidate(s) below the grounding "
            f"threshold -- read them; the count is not the verdict"
        )

    if mislabelled:
        print(f"\n{len(mislabelled)} facts claiming the user's authority that the user never said:")
        for fact in mislabelled:
            print(
                f"  s{fact['session']:<2d} {fact['address']:38s} "
                f"overlap={fact['overlap']:.0%}  {fact['value'][:70]}"
            )
    if args.show:
        print(f"\n{len(grounded)} grounded:")
        for fact in grounded:
            print(
                f"  s{fact['session']:<2d} {fact['address']:38s} "
                f"overlap={fact['overlap']:.0%}  {fact['value'][:70]}"
            )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
