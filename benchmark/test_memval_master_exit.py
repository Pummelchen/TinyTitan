"""Gates `benchmark/memval_master.sh`'s exit status.

The master driver runs the ten memory scenarios one at a time through
`memval_run.sh master`, then prints the aggregate report. Each scenario's status
went into a `##### master <scenario> exit=N` line and nowhere else, and the
script's last command is `memory_master.py report-all` -- which returns None, so
its exit status is 0 whether or not a single scenario produced a result. A run in
which every scenario refused therefore exited 0, the same defect AUD-212 closed
for `memval_matrix.sh` and in the same family as AUD-118, AUD-126 and AUD-210.

These tests drive a copy of the script inside a temporary checkout with a stub
`memval_run.sh`, so no model, no server and no GPU is involved -- the same fixture
shape as `test_memval_matrix_exit.py`, `test_golden_baseline_capture.py` and
`test_release_ci_green.py`. The stub answers with the status the test wrote for its
scenario, so one scenario can fail, or all ten, without touching the repository.
`report-all` runs for real against the temporary tree: it reads the JSON the arms
would have stored, finds none, and prints its header -- which is precisely the
shape that made the old exit code a lie, so pinning it is the point of tests four
and five.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
DRIVER = REPO / "benchmark" / "memval_master.sh"
# The python files the aggregate report needs: `memory_master.py` loads
# `master_scenarios` by path and imports `memval_env` for its results tree, so a
# checkout without all three dies in the report rather than running it, and the
# report's own exit status is half of what is under test.
REPORT_SOURCES = ("memory_master.py", "master_scenarios.py", "memval_env.py")
SCENARIOS = [
    "photograph",
    "pong",
    "ledger",
    "pigeon",
    "contract",
    "compound_k",
    "vantage",
    "kitchen",
    "cohort",
    "filing",
]

STUB = """#!/usr/bin/env bash
# Stands in for benchmark/memval_run.sh: prints what the real one prints and exits
# whatever status the test wrote into the statuses file for this scenario.
key="${TINYTITAN_MASTER_SCENARIO}"
line="$(grep -F -- "$key " "$STATUSES" | head -1)"
status="${line#"$key "}"
[ -n "$status" ] || status="${STATUS:-0}"
echo "master stub ran $key -> exit $status"
exit "$status"
"""


class MasterExitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="memval-master-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        (self.tmp / "benchmark").mkdir()
        (self.tmp / ".build").mkdir()
        shutil.copy(DRIVER, self.tmp / "benchmark" / "memval_master.sh")
        for name in REPORT_SOURCES:
            shutil.copy(REPO / "benchmark" / name, self.tmp / "benchmark" / name)
        stub = self.tmp / "benchmark" / "memval_run.sh"
        stub.write_text(STUB)
        stub.chmod(0o755)
        (self.tmp / "statuses").write_text("")

    def run_driver(self, *names: str, status: str = "0", per_scenario: dict | None = None):
        statuses = ""
        for key, code in (per_scenario or {}).items():
            statuses += f"{key} {code}\n"
        (self.tmp / "statuses").write_text(statuses)
        env = dict(os.environ, STATUS=status, STATUSES=str(self.tmp / "statuses"))
        argv = ["/bin/bash", str(self.tmp / "benchmark" / "memval_master.sh"), *names]
        return subprocess.run(
            argv, capture_output=True, text=True, env=env, timeout=120, check=False
        )

    def summary_lines(self, output: str, marker: str) -> list[str]:
        return [line for line in output.splitlines() if marker in line]

    def test_a_clean_run_exits_zero(self):
        proc = self.run_driver()
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        done = self.summary_lines(proc.stdout, "MASTER DONE")
        self.assertEqual(len(done), 1, proc.stdout)
        self.assertIn("all 10 scenarios exited 0", done[0])

    def test_one_failed_scenario_fails_the_run(self):
        proc = self.run_driver(per_scenario={"pong": "1"})
        self.assertEqual(proc.returncode, 1, "a scenario that failed must not exit 0")

    def test_every_scenario_failing_fails_the_run(self):
        """The reproduction: ten refusals, an empty report, and the exit code the
        audit found was 0."""
        proc = self.run_driver(status="1")
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(len(self.summary_lines(proc.stdout, "MASTER DONE")), 0, proc.stdout)

    def test_the_failed_scenarios_are_named(self):
        proc = self.run_driver(per_scenario={"pong": "1", "filing": "2"})
        failed = self.summary_lines(proc.stdout, "MASTER FAILED")
        self.assertEqual(len(failed), 1, proc.stdout)
        # A leading space, because " 2 of 10" is a substring of "12 of 10" and the
        # mutation this pins is a summary that counts the wrong total.
        self.assertIn(" 2 of 10 scenarios failed", failed[0])
        self.assertIn("pong(exit 1)", failed[0])
        self.assertIn("filing(exit 2)", failed[0])
        self.assertNotIn("photograph", failed[0])

    def test_the_denominator_is_the_scenarios_that_ran(self):
        """The driver takes scenario names as arguments, so `of 10` would be wrong for
        a two-scenario run -- and a wrong total is how a truncated run reads as a
        clean one."""
        proc = self.run_driver("pong", "filing", per_scenario={"filing": "1"})
        failed = self.summary_lines(proc.stdout, "MASTER FAILED")
        self.assertEqual(len(failed), 1, proc.stdout)
        self.assertIn(" 1 of 2 scenarios failed", failed[0])
        self.assertNotIn("of 10", failed[0])
        self.assertEqual(proc.returncode, 1)

    def test_the_report_still_runs_after_a_failure(self):
        """A non-zero verdict must not swallow the record: the aggregate is what an
        operator reads, and the failed scenarios are named in the summary above it."""
        proc = self.run_driver(per_scenario={"pong": "1"})
        self.assertIn("=== all master scenarios", proc.stdout)
        after = proc.stdout.split("=== all master scenarios", 1)[1]
        # `scenario` alone is worthless here: the failure summary that follows the
        # report contains `scenarios failed`, and that is a substring match. Pin the
        # report's own column row, which nothing else in the output prints.
        header = [line for line in after.splitlines() if "carryable" in line and "cost s" in line]
        self.assertEqual(len(header), 1, after)
        self.assertEqual(proc.returncode, 1)

    def test_the_log_still_carries_each_scenario_status(self):
        """The per-scenario `exit=N` line is what a reader diffs; the new summary must
        not replace it."""
        proc = self.run_driver(per_scenario={"pong": "3"})
        lines = self.summary_lines(proc.stdout, "master pong exit=")
        self.assertEqual(len(lines), 1, proc.stdout)
        self.assertIn("exit=3", lines[0])
        self.assertEqual(proc.returncode, 1)


if __name__ == "__main__":
    unittest.main()
