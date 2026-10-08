"""Gates `benchmark/quant_quality_ab.py`'s score and exit status.

The driver answers twenty checkable prompts from each of two installs and prints
`sum/20` per install. Until AUD-214 a run in which the CLI refused every prompt —
a moved model directory, a stale binary, a flag the release CLI no longer takes —
printed `0/20` for both installs and exited **0**, because `run_case()` turns a
non-zero exit into the string `<exit N>` and `passed()` scores that string as a
wrong answer. That is not a quiet log: the number is the product. Two installs
that never ran read as two installs that tie, and this driver exists so that a
precision promotion has something to fail against.

The distinction these tests hold is the one AUD-213 made in `memval_master.sh`: a
bad *score* is a result and exits 0, while a run that *did not measure* is not a
score at all. So the denominator is the number of cases that ran, and the exit
status follows the refusals rather than the report.

They drive the real `main()` against a stub CLI, so no model, no server and no GPU
is involved — the same shape as `test_golden_baseline_capture.py` and
`test_memval_matrix_exit.py`. The stub is invoked through the real `subprocess`, so
the exit codes and stdout the driver reads are real ones.

    cd benchmark && python3 -m unittest test_quant_quality_ab_exit -v
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import pathlib
import shutil
import stat
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "quant_quality_ab", ROOT / "benchmark" / "quant_quality_ab.py"
)
ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ab)

STUB = """#!/usr/bin/env python3
import json
import os
import sys

# Stands in for .build/release/TinyTitanCLI. Reads the prompt the driver wrote to
# --messages-file and obeys the first rule in TINYTITAN_STUB_REPLIES whose key is
# "*" or a substring of the prompt; an action of "error" is a CLI that refused.
argv = sys.argv[1:]
with open(argv[argv.index("--messages-file") + 1], encoding="utf-8") as handle:
    prompt = json.load(handle)[1]["content"]
with open(os.environ["TINYTITAN_STUB_REPLIES"], encoding="utf-8") as handle:
    for line in handle:
        key, _, action = line.rstrip("\\n").partition("\\t")
        if key == "*" or key in prompt:
            if action == "error":
                sys.stderr.write("error: trusted receipt invalid: model directory mismatch\\n")
                sys.exit(127)
            sys.stdout.write(action + "\\n")
            sys.exit(0)
sys.stderr.write("stub has no rule\\n")
sys.exit(9)
"""

MODELS = ["models/install_under_test_4Bit", "models/install_under_test_8Bit"]
ALL_RULES = "*\terror\n"
WRONG_RULES = "*\tbanana\n"


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="quant-quality-ab-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        stub = self.tmp / "stub-cli"
        stub.write_text(STUB)
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
        self.stub = stub
        self.replies = self.tmp / "replies"

    def run_driver(self, rules: str, models: list[str] = MODELS) -> tuple[int, str]:
        """The real `main()`, with the CLI path and the reply table swapped out.

        The stub is a subprocess of this process, so the reply table travels through
        the environment rather than an argument, and `subprocess.run` in the driver
        inherits it.
        """
        self.replies.write_text(rules)
        buffer = io.StringIO()
        original_cli, original_argv = ab.CLI, sys.argv
        ab.CLI = self.stub
        sys.argv = ["quant_quality_ab.py", *models]
        os.environ["TINYTITAN_STUB_REPLIES"] = str(self.replies)
        try:
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                status = ab.main()
        finally:
            ab.CLI = original_cli
            sys.argv = original_argv
            os.environ.pop("TINYTITAN_STUB_REPLIES", None)
        return status, buffer.getvalue()

    def totals_line(self, output: str, model: str) -> str:
        """The one `=== totals` row for this install. Matching on the model name alone
        also returns the `=== <model>` section header, which carries a `/` too."""
        lines = [line for line in output.splitlines() if line.startswith(f"  {model}")]
        self.assertEqual(len(lines), 1, output)
        return lines[0]

    def test_a_refused_cli_is_not_a_score(self):
        """Every prompt refused is the broken-instrument case, and it must not print
        as a measurement. A `0/20` here is what a reader would quote as a result."""
        status, output = self.run_driver(ALL_RULES)
        self.assertEqual(status, 1, "a run that measured nothing must not exit 0")
        self.assertIn(" 0/0 measured, 20 errored", self.totals_line(output, MODELS[0]))

    def test_a_refused_case_names_itself_and_carries_the_reason(self):
        """Which prompt refused, and what the CLI said when it refused. The old code
        discarded stderr and printed `<exit 127>`, so a moved model directory and a
        wrong answer looked the same in the log as well as in the score."""
        _, output = self.run_driver(ALL_RULES, MODELS[:1])
        self.assertEqual(sum(1 for line in output.splitlines() if line.startswith("  ERR")), 20)
        self.assertIn("trusted receipt invalid", output)

    def test_both_installs_report_their_own_refusals(self):
        status, output = self.run_driver(ALL_RULES)
        self.assertEqual(status, 1)
        self.assertIn(" 0/0 measured, 20 errored", self.totals_line(output, MODELS[1]))

    def test_the_refused_cases_are_counted_not_merged_into_misses(self):
        """The count is the point: a banner that says "some cases errored" would let a
        20-case refusal and a 1-case refusal print the same."""
        _, output = self.run_driver(ALL_RULES)
        self.assertNotIn("19/20", output)
        self.assertNotIn("0/20", output)

    def test_one_refusal_leaves_the_other_cases_measured(self):
        """The denominator follows what ran, so nineteen real answers are not thrown
        away with the one prompt the CLI refused."""
        rules = "17 + 28\terror\n*\tbanana\n"
        status, output = self.run_driver(rules)
        self.assertEqual(status, 1, "a partial refusal must still fail the run")
        self.assertIn(" 0/19 measured, 1 errored", self.totals_line(output, MODELS[0]))

    def test_a_measured_run_that_answers_badly_exits_zero(self):
        """A score of 0/20 from twenty real replies is a finding, not a crash. This is
        the guard the fix must not swallow: refusals and wrong answers are different.
        """
        status, output = self.run_driver(WRONG_RULES)
        self.assertEqual(status, 0, "a bad score is a result")
        self.assertIn(" 0/20", self.totals_line(output, MODELS[0]))
        self.assertNotIn("errored", output)

    def test_a_measured_run_that_answers_correctly_exits_zero(self):
        """The ceiling case the module docstring names: both installs at 20/20 is a
        valid reading of the instrument, so the counting path must survive the fix."""
        rules = "".join(f"{prompt}\t{expected}\n" for prompt, expected, _ in ab.CASES)
        status, output = self.run_driver(rules)
        self.assertEqual(status, 0, output)
        self.assertIn(" 20/20", self.totals_line(output, MODELS[0]))

    def test_a_missing_cli_is_still_refused_before_any_case_runs(self):
        """The pre-existing guard, pinned so the new exit status does not replace it:
        no binary means no run, and the driver says so rather than score nothing."""
        buffer = io.StringIO()
        original_cli, original_argv = ab.CLI, sys.argv
        ab.CLI = self.tmp / "not-a-binary"
        sys.argv = ["quant_quality_ab.py", *MODELS]
        try:
            with contextlib.redirect_stderr(buffer):
                status = ab.main()
        finally:
            ab.CLI = original_cli
            sys.argv = original_argv
        self.assertEqual(status, 2)
        self.assertIn("build it first", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
