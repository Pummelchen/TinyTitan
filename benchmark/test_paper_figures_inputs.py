"""Pins the inputs `docs/paper/figures.py` reads before it draws the paper's charts.

The script builds every figure from result files under
`.build/benchmark-logs/memory-{book,value}-<label>/<arm>-r<run>.json`, looping a
fixed grid: three labels, four arms, three runs. Two shapes in that loop turn a
missing run into a measured result instead of a refusal:

    if not p.exists():
        continue                      # the arm loses a bar, the script exits 0
    ...
    if not log.exists():
        return 0, 0                   # an absent log reports as zero tokens

Measured on this commit against a temporary checkout holding a real copy of the
script and no logs at all: it prints `book {"v1": {}, ...}` and `pong {...}`,
then dies at line 322 with `ZeroDivisionError: division by zero` -- an unhandled
crash three functions past the input it could not read, naming no file. One run
file out of the grid is worse, because nothing crashes: the script prints its
four tables, writes all eight SVGs, and exits 0 with that repeat averaged away.
`installs` on the same complete-but-partial tree printed five of six models as
`None`, and `pong` printed `{}` for every arm. Absent `server-*.log` files take
the third shape: `consolidation` answers `0, 0`, so the cost split reports a run
that consolidated nothing.

That is AUD-240's zero-run certification and AUD-276's measured-zero in the one
file outside `benchmark/` that reads the same trees -- `docs/` was never swept
for it, and it is where the paper's charts come from.

The tests drive the script through a synthetic repository root because `ROOT` is
derived from `__file__` (`parents[2]`), which is also why the fixture has to be
three directories deep. Nothing here loads a model or touches a live server: the
two drivers the script imports are report readers, and the fixture gives them
synthetic JSON.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
FIGURES = REPO / "docs" / "paper" / "figures.py"
DRIVERS = ("memory_book.py", "memory_value.py", "memval_env.py")
LABELS = (
    "v1",
    "v2",
    "qwen36-4bit",
    "qwen36-8bit",
    "ornith-4bit",
    "ornith-8bit",
    "agentworld-4bit",
    "agentworld-8bit",
)
BOOK_ARMS = ("summary", "auto", "minimal", "full")
PONG_ARMS = ("control", "auto", "minimal", "full")


def book_row(session: int) -> dict:
    return {
        "session": session,
        "answers": {},
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "seconds": 1.5,
    }


def pong_rows() -> list[dict]:
    rules = json.dumps({"field_width": 8, "win_score": 7})
    return [
        {"stage": "swift", "content": rules},
        {"stage": "repack", "content": rules},
    ]


class PaperFigureInputTests(unittest.TestCase):
    def setUp(self):
        # <tmp>/docs/paper/figures.py so that parents[2] is <tmp>: the script
        # resolves its own repository root from its location.
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="paper-figures-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        paper = self.tmp / "docs" / "paper"
        paper.mkdir(parents=True)
        shutil.copy(FIGURES, paper / "figures.py")
        bench = self.tmp / "benchmark"
        bench.mkdir()
        for name in DRIVERS:
            shutil.copy(REPO / "benchmark" / name, bench / name)
        self.logs = self.tmp / ".build" / "benchmark-logs"
        self.logs.mkdir(parents=True)

    def write_complete_tree(self) -> None:
        for label in LABELS:
            for arm in BOOK_ARMS:
                d = self.logs / f"memory-book-{label}"
                d.mkdir(exist_ok=True)
                for run in (1, 2, 3):
                    (d / f"{arm}-r{run}.json").write_text(json.dumps([book_row(2), book_row(3)]))
                    (d / f"server-{arm}-r{run}.log").write_text(
                        "memory consolidated prompt=10 completion=2\n"
                    )
            for arm in PONG_ARMS:
                d = self.logs / f"memory-value-{label}"
                d.mkdir(exist_ok=True)
                for run in (1, 2, 3):
                    (d / f"{arm}-r{run}.json").write_text(json.dumps(pong_rows()))

    def run_figures(self) -> subprocess.CompletedProcess:
        env = {key: value for key, value in os.environ.items() if not key.startswith("TINYTITAN_")}
        env["PYTHONPATH"] = str(self.tmp / "benchmark")
        return subprocess.run(
            ["python3", str(self.tmp / "docs" / "paper" / "figures.py")],
            capture_output=True,
            text=True,
            cwd=self.tmp / "docs" / "paper",
            env=env,
            timeout=180,
            check=False,
        )

    def assert_refused(self, proc: subprocess.CompletedProcess, fragment: str) -> str:
        output = proc.stdout + proc.stderr
        self.assertIn("ERROR", output, f"a refusal has to say it refused: {output}")
        self.assertNotEqual(
            proc.returncode, 0, f"figures.py certified a run it did not read: {output}"
        )
        self.assertIn(fragment, output, f"the refusal has to name the input: {output}")
        return output

    def test_a_missing_results_tree_is_refused(self):
        """No logs at all should name the first input the grid needs, not die with
        a ZeroDivisionError three functions later."""
        proc = self.run_figures()
        output = self.assert_refused(proc, "memory-book-v1/summary-r1.json")
        self.assertNotIn("ZeroDivisionError", output, "the refusal is a crash, not a guard")
        self.assertEqual(
            sorted(path.name for path in (self.tmp / "docs" / "paper" / "fig").glob("*.svg")),
            [],
            "a refused build still wrote figure files",
        )

    def test_a_single_missing_run_is_refused(self):
        """The grid is fixed at three runs, so one absent repeat is not a shorter
        sample the chart may average -- it is a bar drawn from two of three."""
        self.write_complete_tree()
        gone = self.logs / "memory-book-v2" / "auto-r2.json"
        gone.unlink()
        proc = self.run_figures()
        self.assert_refused(proc, "memory-book-v2/auto-r2.json")

    def test_a_missing_value_run_is_refused(self):
        self.write_complete_tree()
        gone = self.logs / "memory-value-qwen36-4bit" / "minimal-r3.json"
        gone.unlink()
        proc = self.run_figures()
        self.assert_refused(proc, "memory-value-qwen36-4bit/minimal-r3.json")

    def test_a_missing_server_log_is_refused(self):
        """`return 0, 0` for an absent log is the measured-zero shape: the cost
        split then reports a run that consolidated nothing."""
        self.write_complete_tree()
        gone = self.logs / "memory-book-v1" / "server-full-r1.log"
        gone.unlink()
        proc = self.run_figures()
        self.assert_refused(proc, "server-full-r1.log")

    def test_a_complete_tree_still_builds(self):
        """The guard must not become a refusal of every build: with the full grid
        present the script exits 0 and writes its figures."""
        self.write_complete_tree()
        proc = self.run_figures()
        output = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 0, output)
        self.assertNotIn("ERROR", output, output)
        written = sorted(path.name for path in (self.tmp / "docs" / "paper" / "fig").glob("*.svg"))
        self.assertTrue(written, "the accepted build wrote no figure")
        self.assertIn("book {", output, output)
        for table in ("pong", "cost", "installs"):
            self.assertNotIn(
                "None",
                output.split(f"{table} ")[1].split("\n")[0],
                f"{table} still reports an arm it did not read: {output}",
            )
            self.assertNotIn(
                "{} ",
                output.split(f"{table} ")[1].split("\n")[0],
                f"{table} printed an empty table: {output}",
            )


if __name__ == "__main__":
    unittest.main()
