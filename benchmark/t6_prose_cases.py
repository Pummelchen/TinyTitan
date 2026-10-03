#!/usr/bin/env python3.13
"""T6 (reply check) cases built from real replies instead of authored ones.

`docs/side-engine-tasks.md` measured T6 on eight authored cases. The gate for
wiring it asks for precision on *free-text* replies, and the recorded
master-benchmark runs carry 30 of them -- the photograph world, three runs of
ten sessions, memory-on arm -- with ground truth in
`benchmark/master_scenarios.py`. This builds the cases, with labels nobody
wrote for this file:

* a **claim** case is a reply that asserts a value for the fact's key. The
  truth is YES when the asserted value differs from the known one, NO when it
  matches.
* a **silent** case is a reply that never touches the fact. The truth is NO,
  because silence is not a contradiction.

Every claim case goes to the review file and is read by hand before the run.
Silent cases are sampled the same way.

    python3.13 benchmark/t6_prose_cases.py --runs 1 --out /tmp/t6.jsonl \
        --review /tmp/t6-review.tsv
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / ".build/benchmark-logs/memory-photograph-qwen36-4bit"

COLOURS = "grey|gray|green|hazel|brown|blue|amber|black"
WEEKDAYS = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"
NEGATIONS = r"\b(?:not|never|no|nobody|no one|without)\b"

# Pairs a plain reading does not settle, excluded from the primary metric and
# reported as a count. A label the person would argue about is not a label.
EXCLUDE = {
    (1, 2, "characters/marcus/knows_photo"): "the photograph's truth 'pressing down on him'",
    (1, 5, "state/ferry"): "the ferry did not run -- a day, or the service?",
    (1, 6, "state/ferry"): "the ferry had not run on Sunday -- a day, or the service?",
}

CHARACTER_FORMS = {
    "marcus": ["marcus"],
    "ines": ["ines"],
    "halvorsen": ["halvorsen"],
    "rosa": ["rosa"],
    "aldo": ["aldo"],
}
POSSESSIVE = {"marcus": "his", "ines": "her", "halvorsen": "his", "rosa": "her", "aldo": "his"}


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scenarios = _load("master_scenarios", "benchmark/master_scenarios.py")
tasks = _load("side_engine_tasks", "benchmark/side_engine_tasks.py")


def strip_quiz(text: str) -> str:
    """The reply without its leading quiz block: T6 checks prose."""
    match = re.match(r"\s*```(?:json)?\s*\{.*?\}\s*```", text, re.DOTALL)
    return text[match.end() :].strip() if match else text.strip()


def lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def spans(pattern: str, text: str) -> list[tuple[str, str]]:
    """(captured value, the line it was found in) for one pattern."""
    out = []
    for match in re.finditer(pattern, text, re.IGNORECASE):
        groups = match.groupdict()
        value = (groups.get("value") or match.group(0)).lower().strip()
        start = text.rfind("\n", 0, match.start()) + 1
        end = text.find("\n", match.end())
        out.append((value, text[start : end if end != -1 else len(text)].strip()))
    return out


def eye_claims(prose: str, character: str) -> list[tuple[str, str]]:
    forms = "|".join(re.escape(form) for form in CHARACTER_FORMS[character])
    claims = []
    for pattern in (
        rf"\b(?:{forms})(?:'s|’s)?\s+(?:\w+[ ,]\s*){{0,2}}(?P<value>{COLOURS})\s+(?:eyes|gaze|irises)\b",
        rf"\b(?P<value>{COLOURS})\s+(?:eyes|gaze|irises)\s+of\s+(?:{forms})\b",
        rf"\b(?:{forms})(?:'s|’s)?\s+eyes\s+(?:were|are|was|had been)\s+(?P<value>{COLOURS})\b",
    ):
        claims += spans(pattern, prose)
    possessive = POSSESSIVE[character]
    for line in lines(prose):
        for match in re.finditer(
            rf"\b{possessive}\s+(?P<value>{COLOURS})\s+(?:eyes|gaze|irises)\b",
            line,
            re.IGNORECASE,
        ):
            before = line[: match.start()].lower()
            nearest = None
            for name, forms_ in CHARACTER_FORMS.items():
                for form in forms_:
                    at = before.rfind(form)
                    if at != -1 and (nearest is None or at > nearest[0]):
                        nearest = (at, name)
            if nearest and nearest[1] == character:
                claims.append((match.group("value").lower(), line))
    return claims


def town_claims(prose: str) -> list[tuple[str, str]]:
    named = spans(r"(?i)\b(?:town|village|city|port)\s+of\s+(?P<value>[A-Z][a-zA-Z]{2,})", prose)
    if named:
        return named
    return spans(r"(?P<value>Ashgrove)", prose)


def rain_claims(prose: str) -> list[tuple[str, str]]:
    claims = []
    for _value, line in spans(
        r"\b(?P<value>rain(?:ed|s|ing|fall)?|downpour|drizzle[ds]?|showers?)\b", prose
    ):
        negated = re.search(
            rf"(?i){NEGATIONS}[^.!?\n]{{0,30}}?\brain(?:ed|s|ing|fall)?\b", line
        ) or re.search(
            r"(?i)\brain(?:ed|s|ing|fall)?\b[^.!?\n]{0,30}?\b(?:never|did not|didn't|no)\b", line
        )
        remembered = re.search(
            r"(?i)\b(?:memory|legend|stories|story|idea|thought|myth|notion|rumou?r)s?\s+of\s+rain",
            line,
        )
        if negated or remembered:
            continue
        claims.append(("rains", line))
    return claims


def ferry_day_claims(prose: str) -> list[tuple[str, str]]:
    """A ferry day is the ferry's, not the weather's.

    "the fog rolled in thick on Thursday, and the ferry stopped" is not a
    claim that the ferry runs on Thursdays, and the corpus is full of lines
    shaped like it. The weekday counts only after the ferry ("runs only on
    Sundays") or named as its day ("Thursdays, the day the ferry ran").
    """
    claims = []
    for line in lines(prose):
        positions = [match.start() for match in re.finditer(r"(?i)\bferry\b", line)]
        if not positions:
            continue
        for match in re.finditer(rf"(?i)\b(?P<value>{WEEKDAYS})s?\b", line):
            after = any(0 <= match.start() - at <= 40 for at in positions)
            named = False
            for at in positions:
                if match.end() < at:
                    between = line[match.end() : min(len(line), at + 5)]
                    if re.search(r"(?i)\b(?:day|when)\b[^.!?]{0,20}?\bthe\s+ferry\b", between):
                        named = True
            if after or named:
                claims.append((match.group("value").lower(), line))
    return claims


def state_claims(prose: str, subject: str, yes_words: str, no_words: str):
    """One state fact: the two sides claim different values.

    A yes-match sitting inside a no-match is the no-match's ("stopped
    running" contains "running"), not a competing claim. A bare negation
    ("the ferry did not come that morning") is deliberately not read as a
    state change -- it is a specific day, not the service.
    """
    claims = []
    for line in lines(prose):
        if subject not in line.lower():
            continue
        subject_spans = [m.span() for m in re.finditer(rf"(?i)\b{subject}\b", line)]

        def near(span, bounds=tuple(subject_spans)) -> bool:
            return any(start - 90 <= span[0] <= end + 90 for start, end in bounds)

        no_spans = [
            match.span()
            for match in re.finditer(rf"(?i)\b(?:{no_words})\b", line)
            if near(match.span())
        ]
        negated = [
            match.span()
            for match in re.finditer(rf"(?i)\b(?:not|no longer|never)\s+(?:{yes_words})\b", line)
            if near(match.span())
        ]
        no_spans += negated
        for match in re.finditer(rf"(?i)\b(?:{yes_words})\b", line):
            if not near(match.span()):
                continue
            if any(start <= match.start() and match.end() <= end for start, end in no_spans):
                continue
            claims.append(("__yes__", line))
        claims += [("__no__", line)] * len(no_spans)
    return claims


def knows_photo_claims(prose: str) -> list[tuple[str, str]]:
    """Only knowledge of *what the photograph shows* is the fact.

    Knowing the photograph matters, or holding it, is compatible with not
    knowing what is in it -- and "had not left ... until he understood the
    photograph" says the opposite of knowing.
    """
    claims = []
    knowing = (
        r"(?:knows|knew|learned|learns|discovered|realised|realized|understands|understood)"
        r"\s+(?:exactly\s+)?what\s+(?:the\s+)?(?:photograph|photo)\s+(?:shows|showed|contains|contained)"
    )
    # "Marcus knew the photograph showed the truth" is the same claim as the
    # sentence above, and the corpus uses both.
    understanding = (
        r"(?:knows|knew|learned|learns|discovered|realised|realized|understands|understood)"
        r"[^.!?]{0,45}?\b(?:photograph|photo)\b[^.!?]{0,45}?\b(?:showed|shows|was|is|held|holds"
        r"|meant|means|contained|contains|revealed|reveals|truth|meaning|contents)\b"
    )
    truth = r"(?:knew|knows|learned|learns|understood|understands)\s+the\s+truth"
    for line in lines(prose):
        if "marcus" not in line.lower():
            continue
        has_photo = bool(re.search(r"(?i)photograph|photo", line))
        for match in re.finditer(rf"(?i)\b(?:{knowing}|{understanding}|{truth})\b", line):
            if "truth" in match.group(0).lower() and not has_photo:
                continue
            before = line[max(0, match.start() - 40) : match.start()]
            if re.search(r"(?i)\b(?:not|never|until|before|no one|nobody)\b\s*\w*\s*$", before):
                claims.append(("false", line))
            else:
                claims.append(("true", line))
        if re.search(
            r"(?i)\bmarcus\b[^.!?\n]{0,60}?\b(?:does not know|did not know|doesn't know"
            r"|didn't know|unaware|had yet to learn)\b[^.!?\n]{0,40}?\b(?:photograph|photo)\b",
            line,
        ):
            claims.append(("false", line))
    return claims


def confessed_claims(prose: str) -> list[tuple[str, str]]:
    """A confession is a claim unless it is a rumour or still unspoken."""
    claims = []
    for value, line in state_claims(
        prose,
        "halvorsen",
        "confess(?:ed|es|ion)|admitted",
        "kept (?:it|the secret)|did not confess|never confessed|had not confessed"
        "|did not admit|had not admitted|hid (?:it|the secret)",
    ):
        if value == "__yes__" and re.search(
            r"(?i)\b(?:rumou?r|rumou?red|alleged|reportedly|unspoken)\b", line
        ):
            continue
        claims.append((value, line))
    return claims


def ferry_running_claims(prose: str) -> list[tuple[str, str]]:
    return state_claims(
        prose,
        "ferry",
        "ran|runs|running|arrived|docked|crossed",
        "stopped running|no longer (?:ran|runs|running)|gone for good|cancelled|ceased",
    )


def left_claims(prose: str) -> list[tuple[str, str]]:
    claims = []
    for line in lines(prose):
        if "ashgrove" not in line.lower():
            continue
        if re.search(
            r"(?i)\b(?:wonder|wondered|wondering|whether|if anyone|had ever|might|would)\b", line
        ):
            continue
        if re.search(r"(?i)\b(?:left|departed|sailed away|moved away|fled)\b", line):
            claims.append(("false" if re.search(rf"(?i){NEGATIONS}", line) else "true", line))
    return claims


def state_detector(subject: str, yes_words: str, no_words: str, yes: str, no: str):
    """Map a state detector's __yes__/__no__ onto the fact's own values."""

    def detect(prose: str) -> list[tuple[str, str]]:
        out = []
        for value, line in state_claims(prose, subject, yes_words, no_words):
            out.append((yes if value == "__yes__" else no, line))
        return out

    return detect


FACTS = [
    (
        "marcus_eyes",
        "characters/marcus/eyes",
        lambda t: t,
        lambda t: t,
        lambda p: eye_claims(p, "marcus"),
    ),
    (
        "ines_eyes",
        "characters/ines/eyes",
        lambda t: t,
        lambda t: t,
        lambda p: eye_claims(p, "ines"),
    ),
    (
        "halvorsen_eyes",
        "characters/halvorsen/eyes",
        lambda t: t,
        lambda t: t,
        lambda p: eye_claims(p, "halvorsen"),
    ),
    (
        "rosa_eyes",
        "characters/rosa/eyes",
        lambda t: t,
        lambda t: t,
        lambda p: eye_claims(p, "rosa"),
    ),
    (
        "aldo_eyes",
        "characters/aldo/eyes",
        lambda t: t,
        lambda t: t,
        lambda p: eye_claims(p, "aldo"),
    ),
    ("town", "setting/town", lambda t: t, lambda t: t.lower(), town_claims),
    (
        "weather_rule",
        "rules/weather",
        lambda t: t,
        lambda t: "rains" if "never" not in t else "never rains",
        rain_claims,
    ),
    ("ferry_day", "rules/ferry", lambda t: t, lambda t: t.lower(), ferry_day_claims),
    (
        "marcus_knows_photo",
        "characters/marcus/knows_photo",
        lambda t: (
            "Marcus knows what the photograph shows"
            if t
            else "Marcus does not know what the photograph shows"
        ),
        lambda t: "true" if t else "false",
        knows_photo_claims,
    ),
    (
        "tomas_status",
        "state/tomas",
        lambda t: "found alive in the lighthouse" if t == "found" else "missing",
        lambda t: t,
        state_detector(
            "tomas",
            "found alive|was found|had been found|has been found|been found|resurfaced"
            "|returned to Ashgrove|hiding in the lighthouse",
            "missing|vanished|disappeared|unaccounted",
            "found",
            "missing",
        ),
    ),
    (
        "inn_status",
        "state/inn",
        lambda t: "burned to the ground" if t == "burned" else "standing",
        lambda t: t,
        state_detector(
            "inn",
            "burn(?:ed|t|ing)?|in ruins|ashes|destroyed|razed|gutted|reduced to",
            "still standing|remained standing|was standing|stood intact|left standing"
            "|stood untouched|stood firm|intact|untouched|unharmed|survived",
            "burned",
            "standing",
        ),
    ),
    (
        "halvorsen_confessed",
        "characters/halvorsen/confessed",
        lambda t: "Halvorsen has confessed the forgery" if t else "Halvorsen has not confessed",
        lambda t: "true" if t else "false",
        confessed_claims,
    ),
    (
        "ferry_running",
        "state/ferry",
        lambda t: "running" if t else "stopped running for good",
        lambda t: "true" if t else "false",
        state_detector(
            "ferry",
            "ran|run|runs|running|arrived|docked|crossed",
            "stopped running|no longer (?:ran|runs|running|run)|gone for good|cancelled|ceased"
            "|suspended|halted",
            "true",
            "false",
        ),
    ),
    (
        "anyone_left_ashgrove",
        "state/anyone_left",
        lambda t: "someone has left Ashgrove" if t else "no one has left Ashgrove",
        lambda t: "true" if t else "false",
        left_claims,
    ),
]


def replies(run: int) -> list[tuple[int, str]]:
    out = []
    for session in range(1, 11):
        path = LOGS / f"photograph-auto-r{run}-{session:02d}.md"
        if not path.exists():
            raise SystemExit(f"missing recorded reply: {path}")
        out.append((session, path.read_text(encoding="utf-8")))
    return out


def build(runs: list[int]):
    truth = scenarios.SCENARIOS["photograph"]["truth"]
    jobs, review, stats = [], [], {}
    for run in runs:
        for session, raw in replies(run):
            prose = strip_quiz(raw)
            known = truth(session)
            for quiz, key, render, canon, detect in FACTS:
                if (run, session, key) in EXCLUDE:
                    review.append(
                        {
                            "run": run,
                            "session": session,
                            "key": key,
                            "kind": "excluded",
                            "truth": "-",
                            "known": render(known[quiz]),
                            "claims": "",
                            "span": EXCLUDE[(run, session, key)],
                        }
                    )
                    stats.setdefault(
                        key, {"claim": 0, "silent": 0, "yes": 0, "no": 0, "excluded": 0}
                    )
                    stats[key]["excluded"] += 1
                    continue
                expected = canon(known[quiz])
                found = detect(prose)
                values = sorted({value for value, _ in found})
                label = "YES" if any(value != expected for value in values) else "NO"
                kind = "claim" if found else "silent"
                jobs.append(
                    tasks.job(
                        "T6",
                        f"KNOWN: {key} = {render(known[quiz])}\n"
                        f"REPLY: {prose}\nDoes the reply contradict what is known?",
                        label,
                        f"r{run}s{session} {key}",
                    )
                )
                review.append(
                    {
                        "run": run,
                        "session": session,
                        "key": key,
                        "kind": kind,
                        "truth": label,
                        "known": render(known[quiz]),
                        "claims": "|".join(values),
                        "span": " || ".join(line[:220] for _, line in found[:3]),
                    }
                )
                counts = stats.setdefault(
                    key, {"claim": 0, "silent": 0, "yes": 0, "no": 0, "excluded": 0}
                )
                counts[kind] += 1
                counts["yes" if label == "YES" else "no"] += 1
    return jobs, review, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, nargs="+", default=[1])
    ap.add_argument("--out", type=Path, default=Path("/tmp/t6-prose-jobs.jsonl"))
    ap.add_argument("--review", type=Path, default=Path("/tmp/t6-prose-review.tsv"))
    args = ap.parse_args()

    jobs, review, stats = build(args.runs)
    args.out.write_text("\n".join(json.dumps(j) for j in jobs) + "\n", encoding="utf-8")
    header = ["run", "session", "key", "kind", "truth", "known", "claims", "span"]
    lines_out = ["\t".join(header)]
    for row in review:
        lines_out.append("\t".join(str(row[column]).replace("\t", " ") for column in header))
    args.review.write_text("\n".join(lines_out) + "\n", encoding="utf-8")

    print(f"cases: {len(jobs)}  (runs {args.runs})  -> {args.out}")
    print(f"review: {args.review}")
    for key, counts in stats.items():
        print(
            f"  {key:38s} claim {counts['claim']:3d}  silent {counts['silent']:3d}"
            f"  excluded {counts.get('excluded', 0):2d}  YES {counts['yes']:3d}  NO {counts['no']:3d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
