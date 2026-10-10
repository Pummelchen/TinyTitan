#!/usr/bin/env python3
"""Tests for `side_engine_recall.py`: AUD-288.

The driver answers one question -- would a caller get anything for a T7
judgement -- by comparing the deterministic token ranking against the engine's
own YES/NO answers, per question, at recall@1 and recall@3.

Pre-fix, measured by writing fixture `done.jsonl` files and calling the real
`score()`: no model, no server, nothing fetched.

1. a file with no rows at all printed `recall@1 0/0` twice and returned 0, so a
   rate over a denominator of nothing was a result;
2. a run that answered one of the four fair questions printed `1/1` and `0/1` and
   returned 0 -- the denominator is whatever survived, which is the AUD-233 shape
   in the driver that decides whether a 15.2 s judgement is worth paying for;
3. rows whose `completion` field is absent were read as `NO`. The whole side-engine
   column is built from `row.get("completion") or ""`, so a dead run produced a
   *worse* side-engine score than a live one, with no line on the page saying the
   engine answered nothing. Measured: a group of ten rows with no completion at
   all scored identically to the same group answering honestly, and returned 0;
4. the same silent `NO` from a shorter cause -- a group that holds 3 of the 10
   facts ranks the other 7 as though the engine had refused them;
5. a missing file, a half-written line and a row that names no question each
   escaped as a traceback at exit 1, which is the status this page uses for
   "measured and contested".

    0  every fair question ran, answered every fact, and every answer was there
    1  a named fair question ran no rows, left facts unanswered, or carried rows
       with no judgement, so the comparison is over part of the matrix
    2  the comparison does not exist: no rows, a file that cannot be read, or no
       fair question answered in full

The `fair` flag on each question is the author's own label for whether the target
really is the answer, and questions labelled unfair stay out of the score; that is
what the "(label does not hold, not counted)" column means and these tests keep it.
"""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import tempfile
import unittest

import side_engine_recall as driver

FAIR = [q for q in driver.QUESTIONS if q[3]]
UNFAIR = [q for q in driver.QUESTIONS if not q[3]]
RAMBLING_NO = "no, that is not the YES this asks for"


def group(question, target, *, completions="honest", limit=None):
    """One question's rows over every fact, the way `prepare` writes them.

    `limit` truncates the group, which is what a batch that died mid-run leaves.
    """
    keys = list(driver.BIBLE)
    if limit is not None:
        keys = keys[:limit]
    rows = []
    for key in keys:
        row = {"question": question, "fact": key, "target": target}
        if completions == "honest":
            row["completion"] = "YES" if key == target else "NO"
        elif completions == "blank":
            row["completion"] = ""
        elif completions == "absent":
            pass
        rows.append(row)
    return rows


def full_run(*, skip=(), dead=(), blank=(), short=()):
    rows = []
    for question, target, _, fair in driver.QUESTIONS:
        if not fair:
            continue
        if question in skip:
            continue
        mode = "blank" if question in blank else "absent" if question in dead else "honest"
        limit = 3 if question in short else None
        rows.extend(group(question, target, completions=mode, limit=limit))
    return rows


def run_on(text):
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
    def test_a_run_where_every_fair_question_answered_answers_zero(self):
        status, text = drive(full_run())
        self.assertEqual(status, 0, text)
        self.assertIn(f"of {len(FAIR)} fair questions", text)
        self.assertNotIn("What does Marcus do", text)

    def test_a_question_whose_label_does_not_hold_stays_out_of_the_score(self):
        rows = full_run()
        rows.extend(group(UNFAIR[0][0], UNFAIR[0][1]))
        status, text = drive(rows)
        self.assertEqual(status, 0, text)
        self.assertIn("label does not hold", text)
        self.assertIn(f"of {len(FAIR)} fair questions", text)

    def test_an_unfair_question_is_dropped_for_its_label_not_its_rows(self):
        """A dead unfair group is still the label's problem, not a missing score.

        The `fair` flag is the author's judgement about the case, so it is what
        keeps the row out of the totals; the run quality of an uncounted row has
        to stay a note, not become an exclusion.
        """
        rows = full_run()
        rows.extend(group(UNFAIR[0][0], UNFAIR[0][1], completions="absent"))
        status, text = drive(rows)
        self.assertEqual(status, 0, text)
        self.assertIn("label does not hold", text)
        self.assertNotIn("excluded", text)

    def test_a_lowercase_answer_is_a_judgement(self):
        """The engine answers in words, not in the fixture's caps.

        `strip().upper()` is what `docs/side-engine-tasks.md` measured; a run that
        answered "yes." is a YES, and dropping either half reads it as a NO.
        """
        rows = []
        for question, target, _, fair in driver.QUESTIONS:
            if not fair:
                continue
            for key in driver.BIBLE:
                raw = "  yes, it could  " if key == target else "no."
                rows.append({"question": question, "fact": key, "completion": raw})
        status, text = drive(rows)
        self.assertEqual(status, 0, text)
        self.assertIn(f"{len(FAIR)} of {len(FAIR)} fair questions", text)
        self.assertIn("side-engine:   recall@1 4/4", text)

    def test_an_answer_that_only_mentions_yes_is_not_a_yes(self):
        """The answer is read from its first word, which is how it shipped.

        The job asks for exactly one word, and a run that rambles can put YES
        anywhere inside a refusal; reading it from anywhere would rank a fact the
        engine refused above the facts it never mentioned.
        """
        rows = []
        for question, target, _, fair in driver.QUESTIONS:
            if not fair:
                continue
            for key in driver.BIBLE:
                rows.append(
                    {
                        "question": question,
                        "fact": key,
                        "completion": RAMBLING_NO if key == target else "no",
                    }
                )
        status, text = drive(rows)
        self.assertEqual(status, 0, text)
        first = next(iter(driver.BIBLE))
        expected = sum(1 for _, target, _, fair in driver.QUESTIONS if fair and target == first)
        self.assertIn(f"side-engine:   recall@1 {expected}/{len(FAIR)}", text)


class PartialRun(unittest.TestCase):
    def test_a_fair_question_that_never_ran_is_named_not_erased(self):
        status, text = drive(full_run(skip=[FAIR[2][0]]))
        self.assertEqual(status, 1, text)
        self.assertIn("did not run", text)
        self.assertIn(f"{len(FAIR) - 1} of {len(FAIR)} fair questions", text)
        line = next(row for row in text.splitlines() if "excluded" in row)
        self.assertEqual(line.split("(excluded")[0].split()[-2:], ["-", "-"], text)

    def test_a_judgement_that_was_never_written_is_not_a_no(self):
        """The defect in its sharpest shape: a dead engine scored as a wrong engine."""
        status, text = drive(full_run(dead=[FAIR[0][0]]))
        self.assertEqual(status, 1, text)
        self.assertIn("carried no judgement", text)
        self.assertIn(f"{len(FAIR) - 1} of {len(FAIR)} fair questions", text)

    def test_an_empty_completion_is_not_a_judgement_either(self):
        status, text = drive(full_run(blank=[FAIR[1][0]]))
        self.assertEqual(status, 1, text)
        self.assertIn("carried no judgement", text)

    def test_the_rows_that_went_silent_are_counted_not_rounded_up(self):
        """A question whose engine died part-way says how many died.

        Every silent row also leaves a fact unanswered, so the two counts have to
        be measured separately or the page overstates the gap.
        """
        rows = full_run()
        for row in [r for r in rows if r["question"] == FAIR[0][0]][:3]:
            row["completion"] = ""
        status, text = drive(rows)
        self.assertEqual(status, 1, text)
        self.assertIn(f"carried no judgement on 3 of {len(driver.BIBLE)} rows", text)
        self.assertIn(f"answered {len(driver.BIBLE) - 3} of {len(driver.BIBLE)} facts", text)

    def test_a_group_that_answered_part_of_the_question_is_named(self):
        """A batch that died mid-question leaves the same silent NO.

        `t7_rank` is built over the whole BIBLE, so a fact with no row is ranked
        as though the engine had said NO about it.
        """
        status, text = drive(full_run(short=[FAIR[3][0]]))
        self.assertEqual(status, 1, text)
        self.assertIn(f"answered 3 of {len(driver.BIBLE)} facts", text)

    def test_a_row_about_a_fact_the_store_does_not_hold_does_not_fill_the_matrix(self):
        rows = full_run()
        rows = [
            r for r in rows if not (r["question"] == FAIR[0][0] and r["fact"] == "rules/weather")
        ]
        rows.append({"question": FAIR[0][0], "fact": "characters/marcus/hat", "completion": "YES"})
        status, text = drive(rows)
        self.assertEqual(status, 1, text)
        self.assertIn(f"answered {len(driver.BIBLE) - 1} of {len(driver.BIBLE)} facts", text)


class NoComparison(unittest.TestCase):
    def test_an_empty_file_refuses_instead_of_printing_a_zero_denominator(self):
        status, text = drive([])
        self.assertEqual(status, 2, text)
        self.assertIn("NOT MEASURED", text)
        self.assertIn("holds no rows", text)
        self.assertNotIn("recall@1 0/0", text)

    def test_rows_for_only_unfair_questions_are_not_a_comparison(self):
        rows = [row for q in UNFAIR for row in group(q[0], q[1])]
        status, text = drive(rows)
        self.assertEqual(status, 2, text)
        self.assertIn("NOT MEASURED", text)

    def test_a_run_where_no_engine_answered_anything_refuses(self):
        status, text = drive(full_run(dead=[q[0] for q in FAIR]))
        self.assertEqual(status, 2, text)
        self.assertIn("NOT MEASURED", text)

    def test_a_missing_file_answers_with_a_status_not_a_traceback(self):
        status, text = drive_missing()
        self.assertEqual(status, 2, text)
        self.assertIn("NOT MEASURED", text)

    def test_a_line_that_is_not_json_answers_with_a_status_not_a_traceback(self):
        lines = json.dumps({"question": FAIR[0][0], "fact": "rules/weather", "completion": "YES"})
        status, text = run_on(f"{lines}\nthe batch wrote half a line he\n")
        self.assertEqual(status, 2, text)
        self.assertIn("NOT MEASURED", text)

    def test_a_row_that_names_no_question_answers_with_a_status_not_a_traceback(self):
        status, text = run_on(json.dumps({"fact": "rules/weather", "completion": "YES"}) + "\n")
        self.assertEqual(status, 2, text)
        self.assertIn("NOT MEASURED", text)


if __name__ == "__main__":
    unittest.main()
