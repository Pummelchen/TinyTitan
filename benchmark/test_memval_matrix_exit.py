"""Gates `benchmark/memval_matrix.sh`'s exit status.

The matrix is a day-long run of twelve `memval_run.sh` invocations across every
35B install and arm set. Its only report is the log it tees, and until AUD-212 the
per-arm exit code went into a `##### … DONE … (exit N)` line and nowhere else, so
the script's own status was the status of the final `echo`.

These tests drive a copy of the script inside a temporary checkout with a stub
`memval_run.sh`, so no model, no server and no GPU is involved — the same shape as
`benchmark/test_golden_baseline_capture.py` and `test_release_ci_green.py`, which
stub the CLI and `gh` for the same reason. The stub answers with a status taken
from a file, so a test can make one arm fail, or all twelve, without touching the
repository.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO / "benchmark" / "memval_matrix.sh"

STUB = """#!/usr/bin/env bash
# Stands in for benchmark/memval_run.sh: prints what the real one prints and exits
# whatever status the test wrote into the statuses file, keyed by the arm it ran.
key="${TINYTITAN_MEMVAL_MODEL}:${TINYTITAN_MEMVAL_QUANT} ${1}"
line="$(grep -F -- "$key " "$STATUSES" | head -1)"
status="${line#"$key "}"
[ -n "$status" ] || status="${STATUS:-0}"
echo "memval stub ran $key -> exit $status"
exit "$status"
"""


class MatrixExitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="memval-matrix-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        (self.tmp / "benchmark").mkdir()
        shutil.copy(SCRIPT, self.tmp / "benchmark" / "memval_matrix.sh")
        stub = self.tmp / "benchmark" / "memval_run.sh"
        stub.write_text(STUB)
        stub.chmod(0o755)
        (self.tmp / "statuses").write_text("")
        (self.tmp / ".build").mkdir()

    def run_matrix(
        self, status: str = "0", per_arm: dict | None = None
    ) -> subprocess.CompletedProcess:
        statuses = ""
        for key, code in (per_arm or {}).items():
            statuses += f"{key} {code}\n"
        (self.tmp / "statuses").write_text(statuses)
        env = dict(os.environ, STATUS=status, STATUSES=str(self.tmp / "statuses"))
        # A day-long matrix is twelve arms; the stub makes each one instant, and
        # TINYTITAN_MEMVAL_RUNS is passed through so a trimmed run stays trimmed.
        return subprocess.run(
            ["/bin/bash", str(self.tmp / "benchmark" / "memval_matrix.sh")],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
            check=False,
        )

    def test_a_clean_matrix_exits_zero(self):
        proc = self.run_matrix()
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("MATRIX DONE", proc.stdout)

    def test_one_failed_arm_fails_the_matrix(self):
        proc = self.run_matrix(per_arm={"qwen36:8 pong": "1"})
        self.assertEqual(proc.returncode, 1, "an arm that failed must not exit 0")

    def test_every_arm_failing_fails_the_matrix(self):
        proc = self.run_matrix(status="1")
        self.assertEqual(proc.returncode, 1)

    def failed_line(self, output: str) -> str:
        lines = [line for line in output.splitlines() if "MATRIX FAILED" in line]
        self.assertEqual(len(lines), 1, output)
        return lines[0]

    def test_the_failed_arms_are_named(self):
        """The operator reads the last screenful of a day-long run, not the log, so
        the summary has to name the arms and say how many of the twelve they were."""
        proc = self.run_matrix(per_arm={"qwen36:8 pong": "1", "ornith:4 book": "2"})
        line = self.failed_line(proc.stdout)
        # A leading space, because "2 of 12" is a substring of "12 of 12": the
        # mutation this is here to catch is exactly a summary that counts the wrong total.
        self.assertIn(" 2 of 12 arms failed", line)
        self.assertIn("qwen36:8 pong(exit 1)", line)
        self.assertIn("ornith:4 book(exit 2)", line)
        self.assertNotIn("ornith:8 book", line)

    def test_the_log_still_carries_each_arm_status(self):
        """The per-arm `DONE (exit N)` line is what a reader diffs; the new
        summary must not replace it."""
        proc = self.run_matrix(per_arm={"qwen36:8 pong": "1"})
        log = (self.tmp / ".build" / "benchmark-logs" / "memval-matrix.log").read_text()
        done = [line for line in log.splitlines() if "qwen36:8 pong DONE" in line]
        self.assertEqual(len(done), 1, log)
        self.assertTrue(done[0].endswith("(exit 1)"), done[0])
        self.assertEqual(proc.returncode, 1)


if __name__ == "__main__":
    unittest.main()
