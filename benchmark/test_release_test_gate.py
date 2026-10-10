"""Gates what `tools/release.sh` accepts as a passing test run.

The release's `gates` step runs the whole suite and then decides from one grep:

    swift test --no-parallel 2>&1 | tee "$STAGE_ROOT.testlog" | grep -E 'Test run with' \\
      || true
    grep -q 'Test run with .* passed' "$STAGE_ROOT.testlog" || die "..."

Both halves are wrong, and each was measured before it was written down:

1. **`swift test --no-parallel` prints one summary line per test target.** A real
   unfiltered run on Swift 6.4 (/tmp/aud260-full.log) carries seven of them —
   799/127, 406/47, 157/22, 189/23, 35/10, 63/14, 97/10. `grep -q` asks whether
   *some* line matches, so the gate is satisfied by six `failed` lines and one
   `passed` line: a release proceeds to the build, the archive and the tag while
   most of the suite is reporting failures.
2. **the run's own status is thrown away** by that `|| true`, and `set -uo pipefail`
   in this script has no `-e`, so nothing else looks at it either. A `swift test`
   that dies before printing any summary — the compile-error shape — leaves the
   gate reading a log with no `Test run with` line, which the `grep -q` does
   refuse, but only by accident: a run that exits non-zero *after* printing a
   passing line for one target is accepted.

These tests drive the script's own verdict function, extracted verbatim along with
the `die` it calls, so the instrument cannot drift from what the script runs and
the extraction refuses if either is rewritten. The seven-line logs are the real
lines with the counts kept; the mixed pass/fail log is those same lines with six
rewritten to the measured failure shape (`Test run with 20 tests in 1 suite
failed after 0.005 seconds with 1 issue.`, /tmp/aud255-mutantA.log). That mix is
**constructed, not observed**: producing it for real would mean breaking six test
targets and running the full suite, which changes nothing about what the grep
does. Nothing here runs `swift test`, builds, or loads a model.

    cd benchmark && python3 -m unittest test_release_test_gate -v
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "release.sh"

# Verbatim from a real `swift test --no-parallel`, one line per test target.
SEVEN_PASSED = [
    "  Test run with 799 tests in 127 suites passed after 112.625 seconds.",
    "  Test run with 406 tests in 47 suites passed after 21.659 seconds.",
    "  Test run with 157 tests in 22 suites passed after 5.798 seconds.",
    "  Test run with 189 tests in 23 suites passed after 0.484 seconds.",
    "  Test run with 35 tests in 10 suites passed after 1.107 seconds.",
    "  Test run with 63 tests in 14 suites passed after 1.150 seconds.",
    "  Test run with 97 tests in 10 suites passed after 0.898 seconds.",
]

# The measured failure shape, with the counts of six of the seven passing lines.
FAILED = [
    "  Test run with 799 tests in 127 suites failed after 112.625 seconds with 4 issues.",
    "  Test run with 406 tests in 47 suites failed after 21.659 seconds with 2 issues.",
    "  Test run with 157 tests in 22 suites failed after 5.798 seconds with 1 issue.",
    "  Test run with 189 tests in 23 suites failed after 0.484 seconds with 1 issue.",
    "  Test run with 35 tests in 10 suites failed after 1.107 seconds with 3 issues.",
    "  Test run with 63 tests in 14 suites failed after 1.150 seconds with 1 issue.",
]


def _function(name: str, pattern: str) -> str:
    """One function from the script, text and all."""
    match = re.search(pattern, SCRIPT.read_text(), re.M | re.S)
    if match is None:
        raise AssertionError(
            f"{SCRIPT} no longer holds a {name} function to drive -- the release's "
            "test verdict is decided somewhere else again"
        )
    return match.group(0)


def run_verdict(log: str, status: int) -> tuple[int, str, str]:
    """Feed `log` and the run's `status` to the script's own verdict function."""
    body = "\n".join(
        [
            _function("die", r"^die\(\) \{[^}]*\}$"),
            _function("test_gate_verdict", r"^test_gate_verdict\(\) \{.*?^\}$"),
            f'test_gate_verdict "$STDIN_LOG" {status}',
        ]
    )
    prelude = (
        # The script's own options: `set -uo pipefail`, with no `-e`.
        "set -uo pipefail\n"
        "STDIN_LOG=$(mktemp); trap 'rm -f \"$STDIN_LOG\"' EXIT\n"
        'cat > "$STDIN_LOG"\n'
    )
    proc = subprocess.run(
        ["/bin/bash", "-c", prelude + body + "\n"],
        input=log,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


class ReleaseTestGateVerdictTests(unittest.TestCase):
    def test_one_passing_target_of_seven_is_not_a_pass(self) -> None:
        """The grep accepts the run if any single target's line says passed."""
        log = "\n".join(FAILED + [SEVEN_PASSED[-1]]) + "\n"
        status, _, err = run_verdict(log, 1)
        self.assertNotEqual(status, 0, f"the release accepted a run six targets failed: {err}")
        self.assertIn("fail", err.lower(), err)
        self.assertIn("6", err, f"the refusal should count the failures it saw: {err}")

    def test_the_run_status_reaches_the_verdict(self) -> None:
        """Every line says passed, but the run exited non-zero: that is a refusal."""
        log = "\n".join(SEVEN_PASSED) + "\n"
        status, _, err = run_verdict(log, 1)
        self.assertNotEqual(
            status, 0, "the verdict reads only the log, so a non-zero run still releases"
        )
        self.assertIn("1", err, f"the refusal should name the status it got: {err}")

    def test_a_run_that_reported_no_summary_is_refused(self) -> None:
        """A build error prints no summary line at all."""
        log = "error: emit-module command failed with exit code 1\n"
        status, _, err = run_verdict(log, 1)
        self.assertNotEqual(status, 0, f"the release accepted a run that reported nothing: {err}")
        self.assertIn("no", err.lower(), err)

    def test_a_clean_run_over_seven_targets_passes_and_says_how_many(self) -> None:
        """The refusal must not cost the operator the count when the run is real."""
        log = "\n".join(SEVEN_PASSED) + "\n"
        status, out, err = run_verdict(log, 0)
        self.assertEqual(status, 0, f"a genuine clean run was refused: {err}")
        self.assertIn("7", out, f"the pass line should count the targets it read: {out}")

    def test_the_gate_hands_the_verdict_the_status_it_just_ran_with(self) -> None:
        """The status has to cross from the pipeline to the verdict by name."""
        text = SCRIPT.read_text()
        calls = [line for line in text.splitlines() if re.search(r"test_gate_verdict\s+\S", line)]
        self.assertEqual(len(calls), 1, calls)
        self.assertRegex(calls[0], r"\$\{?PIPESTATUS\[0\]\}?|test_status")

    def test_the_gate_no_longer_swallows_what_it_gates(self) -> None:
        """`|| true` on the `swift test` line is the defect, not its preamble."""
        text = SCRIPT.read_text()
        # The gates statement, continuation lines and all: a whole-file scan for
        # "swift test" also matches the refusal's own message, which says nothing
        # about what the gate swallows.
        gate = text[text.index('step "gates"') : text.index("# The golden baseline")]
        self.assertIn("swift test --no-parallel", gate)
        self.assertNotIn("|| true", gate)


if __name__ == "__main__":
    unittest.main()
