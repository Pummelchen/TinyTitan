"""`memory_mini.py`'s fidelity row and what a refused request does to it.

The instrument asks a small resident model to extract the book's facts session
by session, then quizzes the store it built. A session whose request refused
used to be absent from the record, and `report()` walked sessions 2..10 anyway,
scoring every quiz cell whose fact the model never had a chance to write: a
server that answered nothing printed the same 0% as a model that answered
everything and kept nothing. These tests drive the real `report()` over
synthesized result records, with no model and no server.

    cd benchmark && python3 -m unittest test_memory_mini_report -v
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mini = _load("memory_mini", "benchmark/memory_mini.py")
CELLS_PER_SESSION = len(mini.book.QUIZ_KEYS)


def answered(session: int) -> dict:
    """A session that replied, and produced no JSON — a real measured zero."""
    return {
        "facts": {},
        "seconds": 1.0,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "raw": "no json here",
    }


def refused(session: int) -> dict:
    return {
        "error": "URLError: <urlopen error [Errno 61] Connection refused>",
        "facts": {},
        "seconds": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.results = pathlib.Path(self.temp.name) / "results"
        self.results.mkdir()
        self._results = mini.RESULTS
        mini.RESULTS = self.results
        self.addCleanup(setattr, mini, "RESULTS", self._results)
        # The runs come from recorded engine logs under .build/, which a fresh
        # clone does not have; the quiz only needs a run to exist.
        self._load_runs = mini.sim.load_runs
        mini.sim.load_runs = lambda: [{"name": "r1"}]
        self.addCleanup(setattr, mini.sim, "load_runs", self._load_runs)

    def write_record(self, sessions: dict[int, dict]) -> None:
        record = {
            "model": "350m-extract",
            "job": "full",
            "runs": {"r1": {str(session): value for session, value in sessions.items()}},
        }
        (self.results / "350m-extract-full.json").write_text(json.dumps(record), encoding="utf-8")

    def draw(self) -> tuple[int, str]:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = mini.report()
        return code, buffer.getvalue()

    def test_a_refused_session_stops_the_quiz_it_degrades(self):
        # Sessions 1 and 4..10 answered; 3 refused. The store the quiz reads from
        # session 3 onward is missing that session's facts, so those cells are not
        # the model's answer to score.
        self.write_record(
            {
                session: refused(session) if session == 3 else answered(session)
                for session in range(1, 11)
            }
        )
        code, out = self.draw()
        self.assertIn(f"1 refused, {CELLS_PER_SESSION} cells scored", out)
        self.assertEqual(code, 1, "a run with a refused request must not report a clean score")

    def test_a_run_where_every_request_refused_measures_nothing(self):
        self.write_record({session: refused(session) for session in range(1, 11)})
        code, out = self.draw()
        self.assertIn("measured nothing", out)
        self.assertIn("10 refused, 0 cells scored", out)
        row = next(line for line in out.splitlines() if line.strip().startswith("350m-extract"))
        self.assertNotIn("%", row, "a run that answered nothing must not print a scored fidelity")
        self.assertRegex(
            row, r"full\s+-\s+-\s+-\s+-\s+\d+\s+10 refused", "every measured column is a dash"
        )
        self.assertEqual(code, 1)

    def test_a_measured_zero_is_a_result_and_exits_clean(self):
        # Every request answered with prose the parser found no JSON in: that is
        # the model failing the job, which is what the instrument is for.
        self.write_record({session: answered(session) for session in range(1, 11)})
        code, out = self.draw()
        self.assertIn(f"{9 * CELLS_PER_SESSION} cells scored", out)
        self.assertNotIn("refused", out)
        row = next(line for line in out.splitlines() if line.strip().startswith("350m-extract"))
        self.assertRegex(row, r"\d+%", "a run that answered must print its fidelity")
        self.assertEqual(code, 0)


class RunModelTests(unittest.TestCase):
    """The refusal has to reach the record, or `report()` cannot know about it."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.results = pathlib.Path(self.temp.name) / "results"
        self._results = mini.RESULTS
        mini.RESULTS = self.results
        self.addCleanup(setattr, mini, "RESULTS", self._results)
        self._load_runs = mini.sim.load_runs
        mini.sim.load_runs = lambda: [{"name": "r1", "text": {1: "a session", 2: "another"}}]
        self.addCleanup(setattr, mini.sim, "load_runs", self._load_runs)

    def test_a_refused_session_is_written_into_the_record(self):
        import urllib.error

        def refuse(system: str, prompt: str):
            raise urllib.error.URLError("connection refused")

        self._complete = mini.complete
        mini.complete = refuse
        self.addCleanup(setattr, mini, "complete", self._complete)
        with contextlib.redirect_stdout(io.StringIO()):
            mini.run_model("350m-extract", 1, ("full",))
        record = json.loads((self.results / "350m-extract-full.json").read_text())
        sessions = record["runs"]["r1"]
        self.assertEqual(set(sessions), {"1", "2"})
        for value in sessions.values():
            self.assertIn("URLError", value["error"])


class ResultsTreeTests(unittest.TestCase):
    """AUD-241: seven of the eight memory reports take their results tree from
    `TINYTITAN_MEMVAL_RESULTS`; `memory_mini` hardcoded it, so a reader who named
    a tree got a verdict computed over a tree they never chose -- and the report
    globbed every `.json` in the directory it did read, so a file another tool
    wrote there ended the run on a traceback instead of a named refusal."""

    def temp(self) -> pathlib.Path:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        results = pathlib.Path(holder.name) / "mini"
        results.mkdir()
        return results

    def import_with_env(self, results: pathlib.Path):
        env = {"TINYTITAN_MEMVAL_RESULTS": str(results)}
        with mock.patch.dict(os.environ, env, clear=True):
            module = _load(f"memory_mini_at_{results.name}", "benchmark/memory_mini.py")
        module.sim.load_runs = lambda: [{"name": "r1"}]
        return module

    def draw(self, module) -> tuple[object, str]:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            try:
                code = module.report()
            except Exception as error:
                code = f"RAISED {type(error).__name__}: {error}"
        return code, buffer.getvalue()

    def test_the_env_names_the_tree_the_module_reads(self):
        results = self.temp()
        self.assertEqual(self.import_with_env(results).RESULTS, results)

    def test_a_record_in_the_named_tree_is_reported(self):
        results = self.temp()
        record = {"model": "350m-extract", "job": "full", "runs": {"r1": {"1": answered(1)}}}
        (results / "350m-extract-full.json").write_text(json.dumps(record), encoding="utf-8")
        code, printed = self.draw(self.import_with_env(results))
        self.assertEqual(code, 0, printed)
        self.assertIn("350m-extract", printed)

    def test_the_refusal_names_the_tree_the_reader_named(self):
        results = self.temp()
        code, printed = self.draw(self.import_with_env(results))
        self.assertEqual(code, 1, printed)
        self.assertIn("NOT MEASURED", printed)
        self.assertIn(str(results), printed)

    def test_a_foreign_json_is_named_and_refused_not_a_crash(self):
        results = self.temp()
        module = self.import_with_env(results)
        # The shape a sibling report writes: a list of per-session rows, not a
        # mini record. `record["runs"]` over it is a TypeError.
        (results / "pong-4bit-r1.json").write_text(
            json.dumps([{"self_truth": 1}]), encoding="utf-8"
        )
        code, printed = self.draw(module)
        self.assertNotIsInstance(code, str, printed)
        self.assertEqual(code, 1, printed)
        self.assertIn("pong-4bit-r1.json", printed)

    def test_a_shared_tree_names_the_foreign_file_and_still_reports(self):
        """A named tree may hold both tools' records. The foreign file is named and
        left out; the mini row that is there still counts, so a reader who shares a
        directory is told about it rather than denied their own result."""
        results = self.temp()
        record = {"model": "350m-extract", "job": "full", "runs": {"r1": {"1": answered(1)}}}
        (results / "350m-extract-full.json").write_text(json.dumps(record), encoding="utf-8")
        (results / "pong-4bit-r1.json").write_text(
            json.dumps([{"self_truth": 1}]), encoding="utf-8"
        )
        code, printed = self.draw(self.import_with_env(results))
        self.assertEqual(code, 0, printed)
        self.assertIn("pong-4bit-r1.json", printed)
        self.assertIn("350m-extract", printed)


if __name__ == "__main__":
    unittest.main()
