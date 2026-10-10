#!/usr/bin/env python3.13
"""The verdict `side_engine_judges.py` answers, and what each status means.

AUD-291 measured this driver against HEAD before it was written. Five shapes, all
of them wrong in the same direction -- a run that proved something reading as a
run that proved everything:

    a done file with no rows            `summarize()` -> {}, and the verdict is
                                        `1 if any(entry["refused"] ...) else 0`,
                                        which is False over an empty map: 0
    a judge that wrote two of sixty     `{'T2': {total: 2, refused: 0}}`: 0
    a missing done file                 `FileNotFoundError` out of `read_text`: 1
    a half-written line                 `JSONDecodeError`: 1
    a row that names no task            `KeyError`, and no `prompt` reaches
                                        `tasks.truth_of` as a second `KeyError`: 1

`line()` also printed `measured nothing: every request refused` over the first
shape, where no request had run at all, so the wording named a cause the file
contradicted.

The statuses these tests pin are the ones `side_engine_tasks.py`,
`side_engine_recall.py` and `side_engine_wired_cases.py` print:

    0  every job ran, every judgement was written, and the percentages cover them
    1  measured and contested: a job wrote no row, more rows arrived than there
       are jobs, or a request refused, so the score covers part of the run
    2  NOT MEASURED -- no done file, no rows in it, a row that is not readable as
       a case, or no jobs to judge; the reason prints and no table does

The reading itself is preserved and pinned below: `truth_of`, the first word of
the completion, the case-fold and the `.,:;"'` strip are untouched, so a recorded
accuracy figure still means what it meant. No model, no server, no engine binary
and nothing fetched: only `run_cpu` is patched, and it writes a fixture file.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import importlib.util

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


driver = _load("side_engine_judges", "benchmark/side_engine_judges.py")
CASES = driver.tasks.cases()
TASKS = sorted({case["task"] for case in CASES})


def rows_for(selected: list[dict], mode: str = "right") -> list[dict]:
    """Done-file rows over `selected`, answered the way the mode says."""
    rows = []
    for case in selected:
        row = dict(case)
        truth = row["truth"]
        if mode == "right":
            row["completion"] = truth
        elif mode == "wrong":
            row["completion"] = "NO" if truth == "YES" else "YES"
        elif mode == "blank":
            row["completion"] = ""
        elif mode == "refused":
            row["completion"] = ""
            row["error"] = "URLError: connection refused"
        elif mode == "untouched":
            row.pop("completion", None)
        rows.append(row)
    return rows


def drive(
    plan: dict[str, object],
    *,
    jobs: list[dict] | None = None,
    jobs_file: object = "authored",
) -> tuple[int, str]:
    """Call the real `main()` with only `run_cpu` patched.

    `plan` maps a judge spec to what that judge's runner leaves behind: a list of
    rows, a string written verbatim, `None` for an empty file, or `ABSENT` for a
    runner that writes nothing at all.
    """
    jobs = CASES if jobs is None else jobs
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        if jobs_file == "authored":
            handle = root / "tasks.jsonl"
            handle.write_text("\n".join(json.dumps(job) for job in jobs) + "\n", encoding="utf-8")
            jobs_path = handle
        elif jobs_file is None:
            jobs_path = root / "absent.jsonl"
        elif isinstance(jobs_file, list):
            jobs_path = root / "jobs.jsonl"
            jobs_path.write_text("\n".join(jobs_file) + "\n", encoding="utf-8")
        else:
            jobs_path = root / "empty.jsonl"
            jobs_path.write_text("", encoding="utf-8")
        argv = ["side_engine_judges.py", "--jobs", str(jobs_path)]
        for spec in plan:
            argv += ["--judge", spec]

        def fake_run_cpu(target: str, job_list: list[dict], out: Path) -> float:
            entry = plan[f"cpu:{target}"]
            if entry is ABSENT:
                return 12.0
            if isinstance(entry, list):
                out.write_text(
                    "\n".join(json.dumps(row) for row in entry) + "\n" if entry else "",
                    encoding="utf-8",
                )
            else:
                out.write_text(str(entry), encoding="utf-8")
            return 12.0

        buffer = io.StringIO()
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(driver, "run_cpu", fake_run_cpu),
            contextlib.redirect_stdout(buffer),
        ):
            status = driver.main()
        return status, buffer.getvalue()


ABSENT = object()
CLEAN = "cpu:models/qwen3.5_4B_4Bit"
SECOND = "cpu:models/qwen3.5_9B_4Bit"


class MeasuredRun(unittest.TestCase):
    def test_a_run_where_every_job_answered_answers_zero(self):
        status, text = drive({CLEAN: rows_for(CASES)})
        self.assertEqual(status, 0, text)
        self.assertIn(f"judged {len(CASES)} of {len(CASES)} jobs", text)
        self.assertIn(f"tasks good on both halves: {len(TASKS)}/{len(TASKS)}", text)
        self.assertNotIn("NOT MEASURED", text)
        self.assertNotIn("judgements refused", text)

    def test_a_wrong_answer_is_the_measurement_not_a_gap(self):
        """A judge that disagrees is a result, not a run that proved nothing.

        The driver exists to compare judges, so only silence and absence contest
        its status; accuracy is what it prints, and prints as 0% over a full run.
        """
        status, text = drive({CLEAN: rows_for(CASES, "wrong")})
        self.assertEqual(status, 0, text)
        self.assertIn(f"judged {len(CASES)} of {len(CASES)} jobs", text)

    def test_the_answer_is_read_the_way_it_was_before(self):
        """Preservation pin: case-fold, whitespace and punctuation still parse.

        The fix touches the verdict and the reading of absence only. An answer of
        `  yes. ` is a YES, which is how every recorded figure was produced.
        """
        rows = []
        for case in CASES:
            row = dict(case)
            row["completion"] = f"  {case['truth'].lower()}, it is.  "
            rows.append(row)
        status, text = drive({CLEAN: rows})
        self.assertEqual(status, 0, text)
        self.assertIn(f"tasks good on both halves: {len(TASKS)}/{len(TASKS)}", text)

    def test_a_half_good_half_is_flagged_and_the_tally_names_it(self):
        """The `*` is a threshold, so a test has to put a task under it.

        A judge right on one answer half and 2/4 on the other is a result, not a
        gap -- status stays 0 -- but the table must not call that task good: 0.5
        is below the 0.7 both-halves bar the published tables read as good.
        """
        rows = []
        flipped = 0
        for case in CASES:
            row = dict(case)
            if case["task"] == "T5" and case["truth"] == "YES" and flipped < 2:
                row["completion"] = "NO"
                flipped += 1
            else:
                row["completion"] = case["truth"]
            rows.append(row)
        self.assertEqual(flipped, 2)
        status, text = drive({CLEAN: rows})
        self.assertEqual(status, 0, text)
        self.assertIn(f"tasks good on both halves: {len(TASKS) - 1}/{len(TASKS)}", text)
        line = next(row for row in text.splitlines() if row.startswith("T5 "))
        self.assertTrue(line.endswith("*"), line)

    def test_a_refused_request_contests_the_run_and_stays_out_of_the_score(self):
        rows = rows_for(CASES)
        rows[0]["error"] = "URLError: connection refused"
        rows[0]["completion"] = ""
        status, text = drive({CLEAN: rows})
        self.assertEqual(status, 1, text)
        self.assertIn("refused 1", text)
        self.assertIn(f"judged {len(CASES) - 1} of {len(CASES)} jobs", text)


class AbsenceIsNotRefusal(unittest.TestCase):
    def test_a_done_file_with_no_rows_refuses_instead_of_answering_zero(self):
        """The defect in its sharpest shape: nothing written, everything passed.

        `failures = 0` over an empty per-task map is the pass status, and the
        wording printed beside it claimed every request had refused.
        """
        status, text = drive({CLEAN: []})
        self.assertEqual(status, 2, text)
        self.assertIn("NOT MEASURED", text)
        self.assertIn("holds no rows", text)
        self.assertIn(f"none of the {len(CASES)} jobs", text)
        self.assertNotIn("every request refused", text)
        self.assertNotIn("tasks good on both halves", text)

    def test_a_run_that_refused_everything_is_a_refusal_and_says_so(self):
        """Where the old wording was true, it stays true and stays status 1.

        A file of refusals did run, and naming that is different from naming a
        file that holds nothing; both print no score, only one calls it measured.
        """
        status, text = drive({CLEAN: rows_for(CASES, "refused")})
        self.assertEqual(status, 1, text)
        self.assertIn("every request refused", text)
        self.assertNotIn("NOT MEASURED", text)

    def test_a_row_with_no_completion_is_absence_not_a_no(self):
        """A run that went silent names silence, not refusals.

        The old wording said `every request refused` whatever the file held; a
        run whose rows carry no answer and no error refused nothing, and saying
        it did blames the transport for what the judge never wrote.
        """
        rows = rows_for(CASES, "untouched")
        status, text = drive({CLEAN: rows})
        self.assertEqual(status, 1, text)
        self.assertIn(f"judged 0 of {len(CASES)} jobs", text)
        self.assertIn(f"{len(CASES)} rows carried no answer", text)
        self.assertNotIn("every request refused", text)


class CoverageOfTheJobs(unittest.TestCase):
    def test_two_rows_of_a_sixty_job_run_are_named_not_a_pass(self):
        """`len(jobs)` reached the s/judgement rate and never the verdict.

        A run whose engine died after two cases scored its survivors and exited
        0, so the head-to-head the driver exists to run was certified from 3% of
        it. The percentages still print -- they are a measurement -- but they
        name how much of the matrix they cover.
        """
        status, text = drive({CLEAN: rows_for(CASES[:2])})
        self.assertEqual(status, 1, text)
        self.assertIn("judged 2 of 60 jobs", text)
        self.assertIn("60 jobs, so 58 of the jobs wrote no row", text)
        self.assertNotIn("so -", text)

    def test_more_rows_than_jobs_contests_the_run(self):
        """A resumed run that appended rather than replaced duplicates verdicts."""
        rows = rows_for(CASES) + rows_for(CASES[:3])
        status, text = drive({CLEAN: rows})
        self.assertEqual(status, 1, text)
        self.assertIn("63 rows for 60 jobs", text)
        self.assertIn("row repeats a job", text)
        self.assertNotIn("wrote no row", text)

    def test_a_second_judge_that_wrote_no_file_refuses_the_comparison(self):
        rows = rows_for(CASES)
        status, text = drive({CLEAN: rows, SECOND: ABSENT})
        self.assertEqual(status, 2, text)
        self.assertIn("NOT MEASURED", text)
        self.assertEqual(text.count("tasks good on both halves"), 1, text)


class UnreadableInput(unittest.TestCase):
    def test_a_missing_done_file_answers_a_status_not_a_traceback(self):
        status, text = drive({CLEAN: ABSENT})
        self.assertEqual(status, 2, text)
        self.assertIn("NOT MEASURED", text)

    def test_a_half_written_line_answers_a_status_not_a_traceback(self):
        lines = "\n".join(json.dumps(row) for row in rows_for(CASES))
        status, text = drive({CLEAN: f'{lines}\n{{"task": "T1", "comple\n'})
        self.assertEqual(status, 2, text)
        self.assertIn("NOT MEASURED", text)
        self.assertIn("not json", text)

    def test_a_row_that_names_no_case_answers_a_status_not_a_traceback(self):
        """A file from another driver is not this driver's evidence.

        `task` is indexed to build the map and `prompt` is parsed for the truth,
        so both raised out of `summarize()` and left exit 1 -- the status a
        contested comparison carries -- for a comparison that never existed.
        """
        for row in (
            {"prompt": "p", "truth": "YES", "completion": "YES"},
            {"task": "T1", "completion": "YES"},
        ):
            status, text = drive({CLEAN: [row] * len(CASES)})
            self.assertEqual(status, 2, text)
            self.assertIn("NOT MEASURED", text)

    def test_a_missing_jobs_file_answers_a_status_not_a_traceback(self):
        status, text = drive({CLEAN: rows_for(CASES)}, jobs_file=None)
        self.assertEqual(status, 2, text)
        self.assertIn("NOT MEASURED", text)

    def test_a_half_written_jobs_file_answers_a_status_not_a_traceback(self):
        """The input guard is the same one, on the file the cases come from.

        A `--prepare` that died mid-write leaves a truncated last line; parsing it
        raised `JSONDecodeError` and exited 1, the status for a run that measured
        and disagreed, over a run that never started.
        """
        lines = [json.dumps(job) for job in CASES]
        lines[-1] = '{"task": "T7", "prompt": "half a ca'
        status, text = drive({CLEAN: rows_for(CASES)}, jobs_file=lines)
        self.assertEqual(status, 2, text)
        self.assertIn("NOT MEASURED", text)
        self.assertIn("not json", text)

    def test_a_jobs_file_with_no_cases_refuses(self):
        status, text = drive({CLEAN: rows_for(CASES)}, jobs_file="")
        self.assertEqual(status, 2, text)
        self.assertIn("NOT MEASURED", text)


if __name__ == "__main__":
    unittest.main()
