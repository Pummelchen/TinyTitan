"""Gates `benchmark/memval_master.sh`'s exit status.

The master driver runs the ten memory scenarios one at a time through
`memval_run.sh master`, then prints the aggregate report. Each scenario's status
went into a `##### master <scenario> exit=N` line and nowhere else, and the
script's last command is `memory_master.py report-all`. The claim here used to be
that report-all "returns None, so its exit status is 0" -- that went stale when
AUD-240 gave the reports a status (`raise SystemExit(report_all(...))`), and the
sentence was kept as if it were still the reason. It was the *discarded* status
that kept the defect alive: the driver ran the report as a statement, so a run in
which every scenario exited 0 while the aggregate found nothing to pool still
printed `MASTER DONE ... all 10 scenarios exited 0` with exit 0 (AUD-244). The
driver now reads that status, which is what
`test_a_scenario_that_exited_0_but_wrote_nothing_is_refused` and
`test_the_refused_report_is_named_in_the_summary` pin. The defect is in the same
family as AUD-212, closed for `memval_matrix.sh`, and as AUD-118, AUD-126 and
AUD-210.

These tests drive a copy of the script inside a temporary checkout with a stub
`memval_run.sh`, so no model, no server and no GPU is involved -- the same fixture
shape as `test_memval_matrix_exit.py`, `test_golden_baseline_capture.py` and
`test_release_ci_green.py`. The stub answers with the status the test wrote for its
scenario, so one scenario can fail, or all ten, without touching the repository.
`report-all` runs for real against the temporary tree: it reads the JSON the arms
would have stored, and its own status is half of what is under test -- which is why
the stub leaves a record per arm by default, and why `RECORDS=0` exists to take
them away again.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

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
# Stands in for benchmark/memval_run.sh: prints what the real one prints, exits
# whatever status the test wrote into the statuses file for this scenario, and --
# when RECORDS=1, the default -- leaves one result record per arm where
# `memory_master.py report-all` looks for it. Writing the record is what makes the
# stub a scenario that *measured* something: without it, a clean run of this driver
# is a run in which nothing was measured, which is the defect under test rather than
# the fixture's job. RECORDS=0 reproduces it: every scenario exits 0 and no record
# exists, so only the aggregate report can notice.
key="${TINYTITAN_MASTER_SCENARIO}"
root="$(cd "$(dirname "$0")/.." && pwd)"
line="$(grep -F -- "$key " "$STATUSES" | head -1)"
status="${line#"$key "}"
[ -n "$status" ] || status="${STATUS:-0}"
echo "master stub ran $key -> exit $status"
if [ "${RECORDS:-1}" = 1 ]; then
  dir="$root/.build/benchmark-logs/memory-$key-stub"
  mkdir -p "$dir"
  for arm in summary auto; do
    cat > "$dir/$key-$arm-r1.json" <<'RECORD'
[{"session": 1, "prompt_tokens": 10, "completion_tokens": 5, "seconds": 1.0,
  "summary_seconds": 0.0, "consolidation_wait": 0.0, "finish_reason": "stop",
  "answers": {}}]
RECORD
  done
fi
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

    def run_driver(
        self,
        *names: str,
        status: str = "0",
        per_scenario: dict | None = None,
        records: str = "1",
    ):
        statuses = ""
        for key, code in (per_scenario or {}).items():
            statuses += f"{key} {code}\n"
        (self.tmp / "statuses").write_text(statuses)
        env = dict(
            os.environ,
            STATUS=status,
            STATUSES=str(self.tmp / "statuses"),
            RECORDS=records,
        )
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

    def test_a_scenario_that_exited_0_but_wrote_nothing_is_refused(self):
        """AUD-244. Every scenario can exit 0 while the aggregate has nothing to
        pool -- the arms were told to run and no record reached the tree. The
        report already returns 1 for that (AUD-240), but the driver called
        `report-all` as a statement, so the verdict came only from the scenario
        statuses and the run printed `MASTER DONE ... all 10 scenarios exited 0`
        over a `NOT MEASURED` line with exit 0."""
        proc = self.run_driver(records="0")
        output = proc.stdout + proc.stderr
        self.assertIn("NOT MEASURED", output, output)
        self.assertEqual(
            len(self.summary_lines(proc.stdout, "MASTER DONE")),
            0,
            "a run that measured nothing is not a done run",
        )
        self.assertEqual(proc.returncode, 1, output)

    def test_the_refused_report_is_named_in_the_summary(self):
        """The operator reads the last screenful of a day-long run. A non-zero exit
        with no line saying which part refused reads as an infrastructure failure,
        so the summary has to name the report and carry its status."""
        proc = self.run_driver(records="0")
        failed = self.summary_lines(proc.stdout, "MASTER FAILED")
        self.assertEqual(len(failed), 1, proc.stdout)
        self.assertIn("report-all", failed[0])
        self.assertIn("no result record", failed[0])

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

    def plant_decoy_tree(self) -> str:
        """A populated results tree the run under test never wrote to.

        Returns the leaf path to name in `TINYTITAN_MEMVAL_RESULTS`: the report
        walks that path's parent, so the decoy leaves below it are all in scope.
        Records carry one session each -- the shape the stub writes -- so the
        pool is non-empty and `report-all` returns 0.
        """
        root = self.tmp / "decoy"
        for name in SCENARIOS:
            leaf = root / f"memory-{name}-other-install"
            leaf.mkdir(parents=True)
            for arm in ("summary", "auto"):
                row = {
                    "session": 1,
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "seconds": 1.0,
                    "summary_seconds": 0.0,
                    "consolidation_wait": 0.0,
                    "finish_reason": "stop",
                    "answers": {},
                }
                (leaf / f"{name}-{arm}-r1.json").write_text(json.dumps([row]))
        return str(root / "memory-photograph-other-install")

    def test_an_ambient_results_tree_does_not_move_the_report_off_this_runs_tree(self):
        """AUD-250. `TINYTITAN_MEMVAL_RESULTS` is a documented knob, and the driver
        inherits it: so does this fixture, because `run_driver` starts from
        `os.environ`. With a tree named that is not the temporary checkout's,
        `report-all` answered from records the stub never wrote -- measured with
        the repository's own recorded runs, which printed photograph's 116/135
        carryable and 2743 s and `MASTER DONE ... all 10 scenarios exited 0` at
        exit 0 over a run that wrote nothing.

        The same slip from an operator's shell does the real thing: a day-long
        sweep would be gated by a report over a tree it never wrote. Either way
        the refusal this suite exists for is disarmed. The fix is in the driver,
        which names the tree it aggregates, so this fixture deliberately does not
        pin the variable itself -- pinning it here would make the test answer for
        the script's own behaviour.
        """
        decoy = self.plant_decoy_tree()
        with mock.patch.dict(os.environ, {"TINYTITAN_MEMVAL_RESULTS": decoy}):
            proc = self.run_driver(records="0")
        output = proc.stdout + proc.stderr
        self.assertIn("NOT MEASURED", output, output)
        self.assertEqual(proc.returncode, 1, output)
        # The precise tree, not merely a path: the decoy sits under `self.tmp` too,
        # so any shorter assertion would read as satisfied by the wrong tree.
        self.assertIn(str(self.tmp / ".build/benchmark-logs"), output, output)

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
