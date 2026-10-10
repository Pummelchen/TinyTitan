#!/usr/bin/env python3.13
"""AUD-279: the two drivers that score the side-engine over the guard's journals.

Both were found by AUD-278's sibling sweep, and both were measured before being
filed: every shape below is what the shipped `main()` printed, driven with the
journal lookup patched so no model, no server and nothing fetched is involved.

The statuses are the family's (AUD-273 through AUD-278): `0` the page measured
what it was asked and the answer is the good one, `1` it measured and the answer
is the negative one or a named part of it was never compared, `2` the question
could not be answered from this file and the reason is named. Pre-fix,
`composite_split.py` used `2` for "not useful" -- a real measured negative -- and
had no `1` at all, while `side_engine_tasks.py`'s variant scorer ended on an
unconditional `return 0` and its `--score` certified "every task good on both
halves" over a file holding no cases.

One shape was found while verifying the fix, not before filing it: `--score` over
the very file this driver's `--prepare` writes -- 60 readable, labelled rows and
no model answering any of them -- printed `T2 20 0%*` for all six tasks and
exited 1, an absent answer counted as a miss inside the denominator, while
`score_variants()` in the same file refused exactly that shape. A row the run
left unanswered is now its own counted bucket: the answers are scored over their
own denominator and the rest contests the page.

The rule the threshold arguments are held to is AUD-278's: `0 < threshold <= 1`.
`composite_split.py` compares `before < threshold`, so a bar at or below 0 makes
"ungrounded before" unreachable -- nothing can be counted as repaired, whatever
the model answered -- and a bar above 1 makes every row ungrounded, so the repair
column is a property of the argument rather than of the answer. Neither is a
measurement, and both print a table that reads like one.

What is deliberately *not* filed here: `side_engine_tasks.py:197-199` derives a
T1 case's ground label from the same word-overlap grounding check whose
reliability AUD-278 put under a control. Re-labelling those clauses by hand is a
measurement job for the operator -- the file already records two hand
corrections, and the Swift suite pins its prompts -- so what is fixed here is that
the page must say how many T1 labels are hand-made and how many come from the
check itself, rather than presenting all of them as ground truth.
"""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import composite_split as cs
import side_engine_tasks as tasks


def row(value: str, session: int, completion: str) -> dict:
    return {
        "value": value,
        "session": session,
        "completion": completion,
        "run": "r",
        "address": "k/a",
    }


def counter(out: str, label: str) -> str:
    """The number on the page's `label` counter line, so a fixture that lands a
    row in the wrong bucket cannot pass on a substring alone."""
    lines = [one for one in out.splitlines() if one.strip().startswith(label)]
    if not lines:
        raise AssertionError(f"no {label!r} counter line in:\n{out}")
    return lines[0].split()[len(label.split())]


class ThresholdDomain(unittest.TestCase):
    """`before < threshold` can only be failed by a bar inside (0, 1]."""

    def drive_score(self, rows: list[dict], threshold: str) -> tuple[int, str, pathlib.Path]:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "done.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            buf = io.StringIO()
            with (
                mock.patch.object(
                    cs, "grounding", side_effect=lambda v, s: 0.0 if v == "bruxelize" else 1.0
                ),
                mock.patch(
                    "sys.argv",
                    ["composite_split.py", "--score", str(path), "--threshold", threshold],
                ),
                contextlib.redirect_stdout(buf),
                contextlib.redirect_stderr(buf),
            ):
                status = cs.main()
            return status, buf.getvalue(), path

    def test_a_bar_at_zero_cannot_call_anything_ungrounded(self):
        # Pre-fix: `ungrounded before 0`, `of those, repaired 0` and "not useful
        # as it stands" -- the repair column is unreachable at any bar <= 0.
        status, out, _ = self.drive_score([row("bruxelize", 1, "hazel")], "0")
        self.assertEqual(status, 2)
        self.assertIn("NOT MEASURED: --threshold 0", out)
        self.assertIn("cannot separate an invented half from the person's", out)
        self.assertNotIn("not useful", out)

    def test_a_bar_above_one_flags_everything(self):
        status, out, _ = self.drive_score([row("bruxelize", 1, "hazel")], "2")
        self.assertEqual(status, 2)
        self.assertIn("NOT MEASURED: --threshold 2", out)
        self.assertIn("puts every fact in the ungrounded half", out)

    def test_the_strictest_honest_bar_is_accepted(self):
        # 1.0 is the boundary: a fact sharing every word is grounded, anything
        # less is ungrounded, and the comparison can still be failed. Driven
        # through main() because that is where an operator's argument is read;
        # score() takes the bar without consulting it.
        status, out, _ = self.drive_score([row("hazel", 1, "hazel")], "1.0")
        self.assertNotEqual(status, 2, out)
        self.assertNotIn("NOT MEASURED", out)

    def test_the_refusal_comes_before_the_file_is_read(self):
        with mock.patch(
            "sys.argv", ["composite_split.py", "--score", "/no/such/file", "--threshold", "0"]
        ):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                status = cs.main()
        self.assertEqual(status, 2)
        self.assertIn("--threshold 0", buf.getvalue())
        self.assertNotIn("no such file", buf.getvalue())


class ScoreableRows(unittest.TestCase):
    """A row the scorer cannot compare must not vanish from the page."""

    def drive(self, rows: list[dict], threshold: float = 0.5) -> tuple[int, str]:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "done.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                status = cs.score(path, threshold)
            return status, buf.getvalue()

    def test_unscoreable_rows_are_counted_and_named(self):
        rows = [row("bruxelize", 1, "hazel"), row("it", 1, "hazel")]
        with mock.patch.object(
            cs, "grounding", side_effect=lambda v, s: None if v == "it" else 0.0
        ):
            status, out = self.drive(rows)
        self.assertIn("not scoreable", out)
        self.assertIn("(no words to compare)", out)
        self.assertEqual(counter(out, "not scoreable"), "1")
        self.assertIn("1 of 2 rows scored", out)
        self.assertEqual(status, 1)

    def test_a_clean_answer_over_an_uncounted_remainder_is_contested(self):
        # One repair, nothing damaged: the good answer, but over a row the scorer
        # dropped, which is exactly the shape AUD-278 refused to certify as a 0.
        rows = [row("bruxelize", 1, "hazel"), row("hazel", 1, "hazel"), row("it", 1, "it")]
        with mock.patch.object(cs, "grounding", side_effect=self.overlap):
            status, out = self.drive(rows)
        self.assertIn("CONTESTED", out)
        self.assertEqual(status, 1)
        self.assertIn("repairs the composites and damages nothing", out)

    @staticmethod
    def overlap(value: str, session: int) -> float | None:
        return {"bruxelize": 0.0, "hazel": 1.0}.get(value)

    def test_a_run_that_changed_nothing_is_not_useful(self):
        # Every fact answered unchanged: nothing repaired, nothing damaged. The
        # verdict is the absence of damage plus the presence of a repair, so a
        # page that dropped the first half of that condition would certify this.
        rows = [row("hazel", 1, "hazel"), row("grey", 2, "grey")]
        with mock.patch.object(cs, "grounding", return_value=1.0):
            status, out = self.drive(rows)
        self.assertIn("not useful as it stands", out)
        self.assertEqual(status, 1)

    def test_a_file_with_no_rows_refuses_instead_of_answering(self):
        # Pre-fix: "0 facts that claimed the person's authority" and "not useful
        # as it stands" -- a verdict about the repair over nothing.
        status, out = self.drive([])
        self.assertEqual(status, 2)
        self.assertIn("NOT MEASURED", out)
        self.assertIn("holds no rows", out)
        self.assertNotIn("not useful", out)

    def test_a_file_where_nothing_can_be_compared_refuses(self):
        with mock.patch.object(cs, "grounding", return_value=None):
            status, out = self.drive([row("it", 1, "it"), row("is", 1, "is")])
        self.assertEqual(status, 2)
        self.assertIn("NOT MEASURED", out)
        self.assertIn("no row was compared", out)

    def test_a_measured_negative_is_one_and_not_a_refusal(self):
        # Two damaged facts is the answer the docstring records; it must not read
        # as "this run proved nothing".
        rows = [row("hazel", 1, "NONE"), row("grey", 1, "NONE")]
        with mock.patch.object(cs, "grounding", return_value=1.0):
            status, out = self.drive(rows)
        self.assertEqual(status, 1)
        self.assertIn("not useful as it stands", out)

    def test_a_missing_file_is_a_refusal_not_a_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                status = cs.score(pathlib.Path(tmp) / "absent.jsonl", 0.5)
        self.assertEqual(status, 2)
        self.assertIn("NOT MEASURED", buf.getvalue())
        self.assertIn("no such file", buf.getvalue())


class PrepareRefusesShrunkenJournals(unittest.TestCase):
    """`facts()` skips a label whose journal is missing; the caller must not."""

    def drive(
        self,
        labels: list[str],
        journals: dict[str, pathlib.Path],
        facts: dict[str, list[dict]],
    ):
        """`facts` is keyed by label and each label gets its own journal path, so
        the count the page prints is the count this fixture put in that label's
        journal rather than one list shared across the labels."""
        owner = {str(path): label for label, path in journals.items()}
        buf = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "jobs.jsonl"
            with (
                mock.patch.object(
                    cs.guard, "journal_for", side_effect=lambda label: journals.get(label)
                ),
                mock.patch.object(
                    cs.guard,
                    "facts",
                    side_effect=lambda j: facts.get(owner.get(str(j), ""), []),
                ),
                mock.patch.object(cs.sim, "user_text", return_value="Rosa: hazel eyes"),
                mock.patch(
                    "sys.argv",
                    ["composite_split.py", "--prepare", str(path), "--labels", ",".join(labels)],
                ),
                contextlib.redirect_stdout(buf),
            ):
                status = cs.main()
            written = path.exists()
        return status, buf.getvalue(), written

    def test_a_missing_label_refuses_and_writes_nothing(self):
        # Pre-fix: "1 facts -> jobs.jsonl" and exit 0 over the one label that
        # resolved, so a typo silently shrinks the file the operator then scores.
        status, out, written = self.drive(
            ["guard-step0", "guard-step0-ornth"],
            {"guard-step0": pathlib.Path("/tmp/guard-step0.ndjson")},
            {
                "guard-step0": [
                    {"session": 1, "address": "k/a", "value": "hazel", "user_asserted": True}
                ]
            },
        )
        self.assertEqual(status, 2)
        self.assertIn("NOT MEASURED", out)
        self.assertIn("guard-step0-ornth", out)
        self.assertFalse(written)

    def test_no_authority_fact_refuses(self):
        status, out, written = self.drive(
            ["guard-step0"], {"guard-step0": pathlib.Path("/tmp/guard-step0.ndjson")}, {}
        )
        self.assertEqual(status, 2)
        self.assertIn("no fact that claimed the person's authority", out)
        self.assertFalse(written)

    def test_an_empty_label_list_refuses(self):
        status, out, written = self.drive([], {}, {})
        self.assertEqual(status, 2)
        self.assertIn("NOT MEASURED", out)
        self.assertFalse(written)

    def test_all_labels_present_writes_and_reports_the_count(self):
        facts = {
            "a": [{"session": 1, "address": "k/a", "value": "hazel", "user_asserted": True}],
            "b": [{"session": 2, "address": "k/b", "value": "grey", "user_asserted": True}],
        }
        status, out, written = self.drive(
            ["a", "b"],
            {"a": pathlib.Path("/tmp/a.ndjson"), "b": pathlib.Path("/tmp/b.ndjson")},
            facts,
        )
        self.assertEqual(status, 0)
        self.assertTrue(written)
        self.assertIn("2 facts", out)


class SideEnginePrepare(unittest.TestCase):
    def drive_prepare(
        self, journals: dict[str, bool], t1_facts: list[dict]
    ) -> tuple[int, str, bool]:
        buf = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "jobs.jsonl"
            with (
                mock.patch.object(
                    tasks.guard,
                    "journal_for",
                    side_effect=lambda label: (
                        pathlib.Path("/tmp/j") if journals.get(label) else None
                    ),
                ),
                mock.patch.object(tasks.guard, "facts", return_value=t1_facts),
                contextlib.redirect_stdout(buf),
            ):
                status = tasks.prepare(path)
            written = path.exists()
        return status, buf.getvalue(), written

    def test_a_missing_journal_refuses_before_writing(self):
        # Pre-fix: `cases()` `continue`s over the label, prepare prints the count
        # and exits 0, and the file has no T1 case at all -- the task whose
        # headline number the docstring quotes.
        status, out, written = self.drive_prepare(
            {"guard-step0": True, "guard-step0-ornith": True, "guard-confirm": False},
            [{"address": "k/a", "value": "hazel; burned", "session": 1, "user_asserted": True}],
        )
        self.assertEqual(status, 2)
        self.assertIn("NOT MEASURED", out)
        self.assertIn("guard-confirm", out)
        self.assertFalse(written)

    def test_no_composite_fact_refuses(self):
        status, out, written = self.drive_prepare(
            {"guard-step0": True, "guard-step0-ornith": True, "guard-confirm": True}, []
        )
        self.assertEqual(status, 2)
        self.assertIn("no composite fact", out)
        self.assertFalse(written)

    def test_the_page_says_which_labels_came_from_the_check(self):
        # T1's ground truth is derived from the grounding check unless
        # HAND_LABELS carries the clause, and the check is the thing under
        # audit -- so the prepared file's own summary must name the split. The
        # clause is read out of HAND_LABELS rather than retyped, so this follows
        # the table instead of freezing one row of it.
        hand_address, hand_clause = next(iter(tasks.HAND_LABELS)).split(" = ", 1)
        facts = [
            {
                "address": hand_address,
                "value": f"{hand_clause}; the inn burned",
                "session": 1,
                "user_asserted": True,
            }
        ]
        status, out, written = self.drive_prepare(
            {"guard-step0": True, "guard-step0-ornith": True, "guard-confirm": True}, facts
        )
        self.assertEqual(status, 0)
        self.assertTrue(written)
        # Three journals, each producing one hand-labelled clause and one the
        # check derived.
        self.assertIn("3 hand-labelled, 3 from the grounding check", out)


class SideEngineScore(unittest.TestCase):
    def drive(self, rows: list[dict]) -> tuple[int, str]:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "done.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                status = tasks.score(path)
            return status, buf.getvalue()

    def case(self, task: str, completion: str, truth: str = "YES") -> dict:
        return {"task": task, "truth": truth, "prompt": "P", "completion": completion}

    def test_an_empty_file_is_not_every_task_good(self):
        # Pre-fix: the header printed, "every task good on both halves." and 0.
        status, out = self.drive([])
        self.assertEqual(status, 2)
        self.assertIn("NOT MEASURED", out)
        self.assertIn("holds no cases", out)
        self.assertNotIn("every task good", out)

    def test_a_row_without_a_task_is_counted_not_dropped(self):
        status, out = self.drive([self.case("T2", "YES"), {"truth": "YES", "completion": "YES"}])
        self.assertIn("unread", out)
        self.assertIn("1 of 2 rows scored", out)
        # Which bucket, not merely that one of them is non-zero: a task-less row
        # still carries a truth, so testing `truth_of` on it would file it as
        # unlabelled and the page would name the wrong cause.
        self.assertEqual(counter(out, "unread"), "1")
        self.assertEqual(counter(out, "unlabelled"), "0")
        self.assertEqual(status, 1)

    def test_a_measured_negative_is_one_and_a_refusal_is_two(self):
        rows = [self.case("T2", "NO", "YES") for _ in range(3)]
        status, out = self.drive(rows)
        self.assertEqual(status, 1)
        self.assertIn("not ready", out)
        with tempfile.TemporaryDirectory() as tmp:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                missing = tasks.score(pathlib.Path(tmp) / "absent.jsonl")
        self.assertEqual(missing, 2)
        self.assertIn("no such file", buf.getvalue())

    def test_a_row_whose_truth_nothing_supplies_is_unread(self):
        # `truth_of` reads row["truth"] for every task and the statement text for
        # T1, so a row carrying neither cannot be scored either way.
        status, out = self.drive([{"task": "T2", "prompt": "P", "completion": "YES"}])
        self.assertEqual(status, 2)
        self.assertIn("no row carried a ground label", out)
        # The count is part of the refusal: a page that says the row could not be
        # read while reporting none was is describing a different file.
        self.assertIn("1 row(s) could not be read", out)

    def test_a_run_where_nothing_answered_refuses_rather_than_scoring_zero(self):
        # A `--prepare` job file handed to `--score`: every row is readable and
        # labelled, and no model answered any of it. Measured pre-fix on the real
        # 60-case file this driver's own `--prepare` writes: the table printed
        # `T2 20 0%*` for all six tasks and exited 1, which reads as a measured
        # failure of the model rather than an absent one -- and `score_variants`
        # in this same file refuses that shape.
        rows = [{"task": "T2", "truth": "YES", "prompt": "P"} for _ in range(3)]
        status, out = self.drive(rows)
        self.assertEqual(status, 2)
        self.assertIn("NOT MEASURED", out)
        self.assertIn("no completion in any of the 3 row(s)", out)
        self.assertNotIn("0%*", out)
        self.assertNotIn("not ready", out)

    def test_a_partly_unanswered_run_scores_the_answers_and_contests_the_rest(self):
        # The answered rows are a measurement and stay one, over their own
        # denominator; the unanswered ones cannot be averaged into it as misses.
        rows = [
            self.case("T2", "YES"),
            self.case("T2", "NO", "NO"),
            {"task": "T2", "truth": "YES", "prompt": "P"},
        ]
        status, out = self.drive(rows)
        self.assertEqual(status, 1)
        self.assertEqual(counter(out, "unanswered"), "1")
        self.assertIn("2 of 3 rows scored", out)
        self.assertIn("CONTESTED: 1 of 3 rows never answered", out)


class VariantScorer(unittest.TestCase):
    def drive(self, rows: list[dict]) -> tuple[int, str]:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "done.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                status = tasks.score_variants(path)
            return status, buf.getvalue()

    def variant(self, name: str, completion: str, truth: str = "YES") -> dict:
        return {
            "task": "T1",
            "variant": name,
            "truth": truth,
            "prompt": "P",
            "completion": completion,
        }

    def test_no_rows_is_not_a_clean_exit(self):
        # Pre-fix: the header printed over nothing and `return 0`.
        status, out = self.drive([])
        self.assertEqual(status, 2)
        self.assertIn("NOT MEASURED", out)
        # The empty file has no completions either, so naming which refusal fired
        # is what keeps the two refusals from standing in for each other.
        self.assertIn("holds no variant cases", out)

    def test_a_run_where_nothing_answered_measured_nothing(self):
        rows = [
            self.variant("plain", ""),
            self.variant("stakes", ""),
            self.variant("confidence", ""),
        ]
        status, out = self.drive(rows)
        self.assertEqual(status, 2)
        self.assertIn("no completion", out)

    def test_a_missing_variant_is_named_and_contested(self):
        rows = [self.variant("plain", "YES") for _ in range(4)]
        status, out = self.drive(rows)
        self.assertEqual(status, 1)
        self.assertIn("CONTESTED: stakes, confidence answered nothing", out)
        self.assertIn("variants scored 1/3", out)

    def test_a_decoration_verdict_reaches_the_status(self):
        # The calibration section prints "the number is a decoration" and,
        # pre-fix, returned 0 anyway. Two bands, three answers each, the same
        # hit rate: the stated confidence separates nothing.
        rows = [self.variant("plain", "YES"), self.variant("stakes", "YES")]
        rows += [self.variant("confidence", word) for word in ("YES 100", "YES 100", "NO 100")]
        rows += [self.variant("confidence", word) for word in ("YES 90", "YES 90", "NO 90")]
        status, out = self.drive(rows)
        self.assertIn("decoration", out)
        self.assertEqual(status, 1)

    def test_all_three_variants_answered_and_separated_is_zero(self):
        rows = [self.variant(name, "YES") for name in ("plain", "stakes") for _ in range(3)]
        rows += [self.variant("confidence", "YES 100") for _ in range(3)]
        rows += [self.variant("confidence", "NO 20", truth="YES") for _ in range(3)]
        status, out = self.drive(rows)
        self.assertEqual(status, 0, out)
        self.assertNotIn("decoration", out)
        # The exact line, not a bare "3/3": the accuracy column prints "3/3" for a
        # variant that answered three cases, so the substring passes pre-fix.
        self.assertIn("variants scored 3/3", out)


if __name__ == "__main__":
    unittest.main()
