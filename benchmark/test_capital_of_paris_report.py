"""Gates `benchmark/capital_of_paris_report.py`'s completeness and provenance.

The driver turns the smartness matrix's raw rows into the wiki page. Until AUD-223
it printed the protocol line as an equation — `{models} x {prompts} x {repeats} =
**{N} measured runs**` — where `N` was a row count taken independently of the three
factors beside it. Measured: a matrix whose last three runs never returned printed
`2 model/engine combinations x 2 prompts x 3 repeats = **9 measured runs**`, and the
section beneath it then asserted **"No run failed": every request was served**,
because that sentence only looks at rows whose `status` is not `ok` and a run that
never returned has no row at all. The page certified a complete matrix over three
missing runs, which is exactly the number a reader quotes.

The same block told the reader the rows came from
`benchmark/benchmark-results/capital-of-paris-20260911T1935/results-v2-3x2.jsonl`
regardless of which file was passed: the path read is never printed, so a fresh
session's rows publish under a September session's provenance — and with no argument
at all the driver read `/tmp/smartness_v2.jsonl`, an ambient file that proves nothing
about the run being reported.

They drive the real `main()` over synthesized rows in a `TemporaryDirectory`, so no
model, no server and no archive file is involved.

    cd benchmark && python3 -m unittest test_capital_of_paris_report -v
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "capital_of_paris_report", ROOT / "benchmark" / "capital_of_paris_report.py"
)
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)

COMBOS = [("Qwen 3.5 4B", 4, "Metal"), ("Qwen 3.5 9B", 8, "CPU")]
PROMPTS = ["Capital of Paris", "What is the capital of France?"]
REPEATS = 3


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="capital-of-paris-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def write_rows(self, total: int, *, status: str = "ok") -> pathlib.Path:
        """`total` rows of a 2 x 2 x 3 matrix, in harness order, as JSONL.

        The harness writes one object per *completed* run, so a `total` below the
        12 the plan implies is a run that died before writing anything — a refused
        model, a killed session — rather than a row that says it failed.
        """
        path = self.tmp / "results.jsonl"
        written, rows = 0, []
        for label, quant, engine in COMBOS:
            for prompt in PROMPTS:
                for repeat in range(1, REPEATS + 1):
                    written += 1
                    if written > total:
                        continue
                    rows.append(
                        {
                            "label": label,
                            "quant": quant,
                            "engine": engine,
                            "prompt": prompt,
                            "repeat": repeat,
                            "status": status,
                            "cold": repeat == 1,
                            "load_s": 10.0,
                            "ttft_s": 0.5,
                            "decode_tok_s": 20.0,
                            "e2e_tok_s": 15.0,
                            "completion_tokens": 9,
                            "finish": "stop",
                            "content": "Paris",
                            "reasoning": "",
                        }
                    )
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return path

    def run_driver(self, argv: list[str]) -> tuple[object, str, str]:
        buffer_out, buffer_err = io.StringIO(), io.StringIO()
        original = sys.argv
        sys.argv = ["capital_of_paris_report.py", *argv]
        try:
            with contextlib.redirect_stdout(buffer_out), contextlib.redirect_stderr(buffer_err):
                status = report.main()
        finally:
            sys.argv = original
        return status, buffer_out.getvalue(), buffer_err.getvalue()

    def protocol_line(self, output: str) -> str:
        lines = [line for line in output.splitlines() if "measured run" in line]
        self.assertEqual(len(lines), 1, "\n".join(lines))
        return lines[0]

    def test_a_short_matrix_is_not_printed_as_an_equation(self):
        """9 rows of a 12-row plan. The product of the factors is 12; the count is 9;
        printing `12 = **9**` states an arithmetic falsehood on the page that is the
        measurement's record."""
        _, output, _ = self.run_driver([str(self.write_rows(9))])
        self.assertNotIn("3 repeats = **9 measured runs**", output)
        self.assertIn("9 of 12", self.protocol_line(output))

    def test_runs_that_never_returned_are_named_as_missing(self):
        """The reader has to see which cells are short, not only how many: three
        missing repeats of one cell and one missing repeat of three cells are
        different findings about the same instrument."""
        _, output, _ = self.run_driver([str(self.write_rows(9))])
        self.assertIn(
            "Qwen 3.5 9B 8-bit CPU", output[:900] + output[output.find("- 2 model") :][:900]
        )
        self.assertIn("What is the capital of France?", output)

    def test_a_short_matrix_is_not_certified_as_served(self):
        """The sentence that made this an S3 rather than a typo: `No run failed` is
        computed from rows that carry a failing `status`, so a missing row is not
        counted and the page affirms that every request was served."""
        _, output, _ = self.run_driver([str(self.write_rows(9))])
        self.assertNotIn("every request was served", output)

    def test_a_complete_matrix_still_prints_the_equation(self):
        """The guard the fix must not swallow: 12 of 12 is a clean, correct protocol
        line, and a full run must not lose the arithmetic it used to state."""
        _, output, _ = self.run_driver([str(self.write_rows(12))])
        self.assertIn("3 repeats = **12 measured runs**", self.protocol_line(output))
        self.assertIn("every request was served", output)

    def test_a_failing_row_is_still_reported_as_a_failure(self):
        """A run that returned `status: error` is the case the old sentence was
        written for, and it must keep its detail after the missing-row fix."""
        path = self.write_rows(12, status="error")
        _, output, _ = self.run_driver([str(path)])
        self.assertIn("12 run(s) failed", output)
        self.assertIn("Qwen 3.5 4B 4-bit Metal", output)

    def test_the_report_names_the_file_it_read(self):
        """Provenance is a measured fact about this run, not a sentence about a
        September session: the page has to carry the path the rows came from."""
        path = self.write_rows(12)
        _, output, _ = self.run_driver([str(path)])
        self.assertIn(str(path), output)
        self.assertNotIn("results-v2-3x2.jsonl", output)

    def test_no_path_is_a_usage_error_not_an_ambient_file(self):
        """With no argument the driver opened `/tmp/smartness_v2.jsonl`. A stale file
        from an earlier session then renders as a report of the run being reported,
        and on a clean machine it dies in a traceback instead of saying what it wants.
        """
        status, output, error = self.run_driver([])
        self.assertIn(status, (1, 2), "a report with no input must not exit 0")
        self.assertNotIn("Capital of Paris", output)
        self.assertNotIn("Traceback", error)
        self.assertIn("results.jsonl", error)

    def test_importing_the_module_does_not_run_the_report(self):
        """The module called `main()` at import, so it could not be tested, imported
        or reused without writing a wiki page to whoever's stdout it landed in — and
        it read the *importing program's* `argv[1:]` as its row files, which is how a
        library import turns into someone else's measurement.

        The guard passes an argv the mutant would act on: a valid rows file as the
        `-c` program's own argument. A module that runs on import prints the page
        instead of staying silent.
        """
        path = self.write_rows(12)
        result = subprocess.run(
            [sys.executable, "-c", "import capital_of_paris_report", str(path)],
            cwd=str(ROOT / "benchmark"),
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "PYTHONPATH": str(ROOT / "benchmark")},
        )
        self.assertEqual(result.returncode, 0, result.stderr[:400])
        self.assertNotIn("Capital of Paris", result.stdout)
        self.assertNotIn("measured runs", result.stdout)


if __name__ == "__main__":
    unittest.main()
