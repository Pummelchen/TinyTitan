"""Gates what `tools/repack_dense.sh` accepts as a passing equivalence run.

Step 5 of that script is the only step that proves the `.ssdai` reader
interprets the bytes right -- steps 3 and 4 prove the bytes and the receipt --
and it runs as one statement whose status `set -e` propagates:

    TINYTITAN_DENSE_EQUIV=1 ... swift test --no-parallel \\
        --filter DenseSSDAIEquivalenceTests
    echo "all checks passed. ..."

Two of `swift test`'s own behaviours make that line able to print a pass over a
run that compared nothing, both measured on Swift 6.4 (the transcripts are in
the docstring of the close, /tmp/aud261-vacuous.log and /tmp/aud261-skipped.log):

1. a ``--filter`` that matches no test at all **exits 0**. It writes
   ``warning: No matching test cases were run`` to stderr and no summary line.
   So renaming ``DenseSSDAIEquivalenceTests`` -- which several closes in this
   audit have done to other suites -- turns the gate into a warning and the
   script's next line into ``all checks passed``.
2. the summary line counts **skipped** tests as passed. Run against the
   model-gated ``LibraryContractTests`` with no model, where the suite and all
   four of its tests print ``skipped.``, the run still ends with
   ``Test run with 4 tests in 1 suite passed after 0.001 seconds.`` at exit 0.
   ``TINYTITAN_DENSE_EQUIV`` is exactly the kind of condition that skips, so an
   equivalence gate that never loaded a model reports as one that did.

These tests drive the script's own verdict function, extracted verbatim so the
instrument cannot drift from what the script runs, with canned logs. They start
no model, run no test target, and touch nothing on disk: the function's only
input is the log it is handed.

    cd benchmark && python3 -m unittest test_repack_dense_gate_verdict -v
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "repack_dense.sh"

# Captured from the real `swift test --filter NoSuchSuiteZZZ` run: exit 0, this
# on stderr, and no `Test run with` line anywhere.
VACUOUS = "warning: No matching test cases were run\n"

# Captured from the real `swift test --filter LibraryContractTests` run with no
# model installed: every test skipped, exit 0, and the summary still says passed.
ALL_SKIPPED = (
    "  Suite LibraryContractTests skipped.\n"
    "  Test twoEnginesOnOneDeviceRunAlternately() skipped.\n"
    "  Test twoGenerationsOnOneSessionBothComplete() skipped.\n"
    "  Test cancellingMidDecodeEndsTheGeneration() skipped.\n"
    "  Test unloadingShutsLiveSessionsDown() skipped.\n"
    "  Test run with 4 tests in 1 suite passed after 0.001 seconds.\n"
)

# The shape a real equivalence pass has: the gate's own test line, then its
# suite, then the summary.
REAL_PASS = (
    "  Test denseInstallMatchesItsSnapshot() passed after 12.543 seconds.\n"
    "  Suite DenseSSDAIEquivalenceTests passed after 12.544 seconds.\n"
    "  Test run with 1 test in 1 suite passed after 12.544 seconds.\n"
)

# One pair compared, a second one skipped: less was proved than the headline.
PARTIAL_SKIP = (
    "  Test denseInstallMatchesItsSnapshot() passed after 12.543 seconds.\n"
    "  Test secondPairStillMatches() skipped.\n"
    "  Test run with 2 tests in 1 suite passed after 12.544 seconds.\n"
)


def verdict_function() -> str:
    """The script's own verdict function, text and all."""
    match = re.search(r"^gate_verdict\(\) \{.*?^\}\n", SCRIPT.read_text(), re.M | re.S)
    if match is None:
        raise AssertionError(
            f"{SCRIPT} no longer holds a gate_verdict function to drive -- "
            "the equivalence gate's verdict is decided somewhere else again"
        )
    return match.group(0)


def run_verdict(log: str) -> tuple[int, str, str]:
    """Feed `log` to the extracted function on /bin/bash 3.2.57."""
    harness = "set -euo pipefail\n" + verdict_function() + "\ngate_verdict\n"
    proc = subprocess.run(
        ["/bin/bash", "-c", harness],
        input=log,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


class RepackDenseGateVerdictTests(unittest.TestCase):
    def test_a_run_that_matched_no_test_is_not_a_pass(self) -> None:
        """The vacuous `--filter` case: swift test exits 0 and ran nothing."""
        status, out, err = run_verdict(VACUOUS)
        self.assertNotEqual(status, 0, f"the verdict accepted a run of zero tests: {out!r}")
        self.assertIn("no tests", err.lower(), err)

    def test_a_run_where_everything_skipped_is_not_a_pass(self) -> None:
        """The summary line says passed while every test printed skipped."""
        status, _, err = run_verdict(ALL_SKIPPED)
        self.assertNotEqual(status, 0, "the verdict trusted 'Test run with ... passed'")
        self.assertIn("skip", err.lower(), err)

    def test_a_partial_skip_is_not_a_pass(self) -> None:
        """A gate that skipped one of its pairs proved less than it claims."""
        status, _, err = run_verdict(PARTIAL_SKIP)
        self.assertNotEqual(status, 0, "the verdict ignored a skipped test")
        self.assertIn("skip", err.lower(), err)

    def test_a_run_that_actually_compared_logits_passes_and_is_shown(self) -> None:
        """The refusal must not cost the operator the evidence when it is real."""
        status, out, err = run_verdict(REAL_PASS)
        self.assertEqual(status, 0, f"a genuine pass was refused: {err}")
        self.assertIn("denseInstallMatchesItsSnapshot() passed", out)

    def test_the_gate_is_still_piped_through_the_verdict(self) -> None:
        """The refusal has to sit on the line that runs the gate, not beside it."""
        text = SCRIPT.read_text()
        # Match the flag, not the name: the header comment names the suite too.
        invocations = [
            line for line in text.splitlines() if "--filter DenseSSDAIEquivalenceTests" in line
        ]
        self.assertEqual(len(invocations), 1, invocations)
        self.assertIn("gate_verdict", invocations[0])

    def test_the_verdict_is_still_the_last_word_before_the_pass_line(self) -> None:
        """`all checks passed` may only be printed after the verdict ran."""
        text = SCRIPT.read_text()
        verdict_at = text.index("gate_verdict\n")
        pass_at = text.index("all checks passed")
        self.assertLess(verdict_at, pass_at)


if __name__ == "__main__":
    unittest.main()
