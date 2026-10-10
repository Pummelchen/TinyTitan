#!/usr/bin/env python3
"""Tests for `side_engine_wired_cases.py`: AUD-289.

The driver answers one question -- does the wiring get the right judgement on
the case shape the memory path actually sends -- over 8 authored duplication
pairs and 4 contradiction pairs, one YES/NO judgement each.

Pre-fix, measured by writing fixture `done.jsonl` files and calling the real
`score()`: no model, no server, nothing fetched.

1. a file with no rows printed the bare `task truth answer note` header, no
   totals at all, and returned 0 -- `failures = 0` over an empty `by_task` is the
   pass status, so a run where nothing was judged certified the wiring;
2. the per-task totals are built from whatever rows arrived, so a file holding
   only the 8 T5 rows printed `T5: 8/8` and returned 0 with T3 never run, and a
   file holding 5 of the 8 T5 pairs printed `T5: 5/5` and returned 0 as well --
   the denominator is the survivors, which is the AUD-233 shape in the driver
   whose table `docs/side-engine-tasks.md` publishes;
3. a row with no `completion` is answered `""`, compared against the truth, and
   printed as a `MISS`: a dead engine reads as an engine that disagreed;
4. a row repeated by a resumed run counted twice, so `T5: 9/9` is a possible
   line over 8 cases;
5. a row from a different case set -- another driver's jsonl -- counted into the
   task's total against a truth this file never authored;
6. a missing path, a half-written line and a row with no `prompt` each escaped as
   a traceback at exit 1, the status this family uses for a measured disagreement.

    0  every authored case ran, was answered, and every answer matched
    1  measured and contested: a case ran no rows or carried no judgement, a row
       this driver did not author came in, or a judgement disagreed with the truth
    2  the comparison does not exist: no rows, a file that cannot be read, or no
       row that belongs to this case set

The first-word reading is the shipped one and is pinned here rather than
tightened: `strip().upper()` then the first word with `.,:;"'` stripped, which is
how the recorded 4B and 9B runs were scored.
"""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import tempfile
import unittest

import side_engine_wired_cases as driver

CASES = driver.cases()
TASKS = sorted({row["task"] for row in CASES})


def full_run(*, drop=(), blank=(), wrong=(), absent_field=()):
    """Every authored case, answered, less the shapes the tests ask for.

    `drop` removes rows (a batch that died before a task ran), `blank` empties
    the completion, `absent_field` removes the key, and `wrong` flips the answer.
    """
    rows = []
    for job in CASES:
        row = dict(job)
        note = note_of(row)
        if note in drop:
            continue
        if note in blank or note in absent_field:
            if note in absent_field:
                rows.append(row)
                continue
            row["completion"] = ""
        else:
            row["completion"] = "NO" if note in wrong else row["truth"]
        rows.append(row)
    return rows


def note_of(row: dict) -> str:
    return row["note"]


def run_on(text: str):
    """Call the real `score()` on a file whose contents are `text`."""
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "done.jsonl"
        path.write_text(text, encoding="utf-8")
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            status = driver.score(path)
        return status, buffer.getvalue()


def drive(rows):
    lines = [json.dumps(row) for row in rows]
    return run_on("\n".join(lines) + "\n" if lines else "")


def drive_missing():
    """Call the real `score()` on a path that does not exist."""
    with tempfile.TemporaryDirectory() as tmp:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            status = driver.score(pathlib.Path(tmp) / "absent.jsonl")
        return status, buffer.getvalue()


class MeasuredRun(unittest.TestCase):
    def test_every_case_answered_right_answers_zero(self):
        status, text = drive(full_run())
        self.assertEqual(status, 0, text)
        for task in TASKS:
            total = sum(1 for row in CASES if row["task"] == task)
            self.assertIn(f"{task}: {total}/{total}", text)
        self.assertIn(f"judged {len(CASES)} of {len(CASES)} cases", text)
        self.assertNotIn("NOT MEASURED", text)
        self.assertNotIn("MISS", text)

    def test_a_lowercase_padded_or_rambling_answer_is_the_shipped_reading(self):
        """The engine answers in words, and the first word is the judgement.

        `strip().upper()` and the first-word rule are what the recorded 4B and 9B
        runs were scored with; dropping either reads a YES as a MISS.
        """
        rows = full_run()
        for row in rows:
            row["completion"] = f"  {row['completion'].lower()}, and it is the same fact  "
        status, text = drive(rows)
        self.assertEqual(status, 0, text)
        self.assertNotIn("MISS", text)

    def test_an_answer_that_only_mentions_yes_is_not_a_yes(self):
        """A rambling refusal whose last word is YES is a NO, not a hit."""
        rows = full_run()
        for row in rows:
            if row["truth"] == "YES":
                row["completion"] = "no, that is not the same fact, YES it is not"
        status, text = drive(rows)
        self.assertEqual(status, 1, text)
        self.assertIn("MISS", text)
        self.assertNotIn("NOT MEASURED", text)

    def test_a_judgement_that_disagrees_is_contested_not_missing(self):
        """Status 2 means the comparison does not exist, as it does next door.

        A wrong answer is a measurement of the wiring, so it is a contested run:
        the sibling `side_engine_tasks.py:756` answers 1 for a task that is not
        ready and reserves 2 for `refuse()`.
        """
        status, text = drive(full_run(wrong=["re-filed eyes"]))
        self.assertEqual(status, 1, text)
        self.assertIn("MISS", text)
        self.assertNotIn("NOT MEASURED", text)
        self.assertIn("1 case answered differently", text)


class SilenceIsNotAnAnswer(unittest.TestCase):
    def test_a_blank_completion_leaves_the_rate_and_is_named(self):
        status, text = drive(full_run(blank=["re-filed ferry rule"]))
        self.assertEqual(status, 1, text)
        row = next(line for line in text.splitlines() if "re-filed ferry rule" in line)
        self.assertIn("no judgement", row, text)
        self.assertEqual(row.split()[2], "-", text)
        self.assertIn(f"judged {len(CASES) - 1} of {len(CASES)} cases", text)
        self.assertIn("T5: 7/7", text)
        self.assertNotIn("T5: 7/8", text)

    def test_an_absent_completion_field_is_the_same_silence(self):
        """A row the engine never wrote for is not a NO about the pair."""
        rows = full_run()
        for row in rows:
            if note_of(row) == "re-filed inn state":
                row.pop("completion", None)
        status, text = drive(rows)
        self.assertEqual(status, 1, text)
        self.assertIn("no judgement", text)
        self.assertNotIn("MISS re-filed inn state", text)

    def test_a_run_where_no_case_answered_is_not_a_comparison(self):
        status, text = drive(full_run(blank=[note_of(row) for row in CASES]))
        self.assertEqual(status, 2, text)
        self.assertIn("NOT MEASURED", text)
        for task in TASKS:
            self.assertNotIn(f"{task}: ", text)


class PartOfTheMatrix(unittest.TestCase):
    def test_a_task_that_never_ran_is_named_and_answers_one(self):
        """The defect at its plainest: half the table printed as the whole one."""
        rows = [row for row in full_run() if row["task"] != "T3"]
        status, text = drive(rows)
        self.assertEqual(status, 1, text)
        self.assertIn("did not run", text)
        self.assertIn(f"judged {len(CASES) - 4} of {len(CASES)} cases", text)
        self.assertNotIn("T3: 0/0", text)

    def test_a_short_task_group_names_the_cases_that_are_missing(self):
        missing = ("re-filed town", "different attributes")
        rows = [row for row in full_run() if note_of(row) not in missing]
        status, text = drive(rows)
        self.assertEqual(status, 1, text)
        self.assertEqual(text.count("did not run"), len(missing), text)
        self.assertIn(f"judged {len(CASES) - len(missing)} of {len(CASES)} cases", text)

    def test_a_row_repeated_by_a_resumed_run_is_named_and_counted_once(self):
        rows = full_run()
        rows.append(dict(rows[0]))
        status, text = drive(rows)
        self.assertEqual(status, 1, text)
        self.assertIn("repeated", text)
        self.assertIn("T5: 8/8", text)

    def test_a_row_from_another_case_set_is_named_and_not_scored(self):
        """Someone scored the wrong jsonl: its truth is not this driver's."""
        rows = full_run()
        foreign = dict(rows[0])
        foreign["prompt"] = "A: characters/marcus/hat = blue\nB: nope = no\nSame fact?"
        foreign["truth"] = "NO"
        foreign["completion"] = "YES"
        rows.append(foreign)
        status, text = drive(rows)
        self.assertEqual(status, 1, text)
        self.assertIn("unrecognised", text)
        self.assertIn("T5: 8/8", text)


class NoComparison(unittest.TestCase):
    def test_an_empty_file_refuses_instead_of_answering_zero(self):
        status, text = drive([])
        self.assertEqual(status, 2, text)
        self.assertIn("NOT MEASURED", text)
        self.assertIn("holds no rows", text)
        for task in TASKS:
            self.assertNotIn(f"{task}: ", text)

    def test_a_missing_file_answers_with_a_status_not_a_traceback(self):
        status, text = drive_missing()
        self.assertEqual(status, 2, text)
        self.assertIn("NOT MEASURED", text)

    def test_a_line_that_is_not_json_answers_with_a_status_not_a_traceback(self):
        line = json.dumps(full_run()[0])
        status, text = run_on(f"{line}\nthe batch wrote half a row he\n")
        self.assertEqual(status, 2, text)
        self.assertIn("NOT MEASURED", text)

    def test_a_row_with_no_prompt_answers_with_a_status_not_a_traceback(self):
        """A row that cannot be matched to a case is not this comparison."""
        row = dict(CASES[0])
        row.pop("prompt")
        row["completion"] = "YES"
        status, text = drive([row])
        self.assertEqual(status, 2, text)
        self.assertIn("NOT MEASURED", text)

    def test_rows_from_another_case_set_are_not_a_comparison(self):
        rows = []
        for row in CASES:
            other = dict(row)
            other["prompt"] = f"{row['prompt']}\nasked somewhere else"
            other["completion"] = row["truth"]
            rows.append(other)
        status, text = drive(rows)
        self.assertEqual(status, 2, text)
        self.assertIn("NOT MEASURED", text)
        self.assertIn("this driver prepared", text)


class PreparedCases(unittest.TestCase):
    def test_prepare_writes_every_authored_case(self):
        """The ground truth `score()` holds a run against is the same list."""
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "jobs.jsonl"
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                status = driver.prepare(path)
            self.assertEqual(status, 0, buffer.getvalue())
            written = [
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
            ]
            self.assertEqual(len(written), len(CASES))
        status, text = drive([{**row, "completion": row["truth"]} for row in written])
        self.assertEqual(status, 0, text)


if __name__ == "__main__":
    unittest.main()
