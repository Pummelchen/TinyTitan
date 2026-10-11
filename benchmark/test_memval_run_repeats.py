"""Gates the repeat counts `benchmark/memval_run.sh` reads from the environment.

`RUNS="${TINYTITAN_MEMVAL_RUNS:-3}"` and `FIRST_RUN="${TINYTITAN_MEMVAL_FIRST_RUN:-1}"`
feed `$(( FIRST_RUN + RUNS - 1 ))`, and a value that is not a positive count turns
that list into one that counts *down*. Measured on this machine's /bin/bash 3.2.57
with the real /usr/bin/seq, against the script's own expression:

    RUNS=0    upper=0    seq 1 0     -> "1" then "0"   (2 iterations, run 0)
    RUNS=-1   upper=-1   seq 1 -1    -> "1", "0", "-1" (3 iterations, numbered down)
    FIRST_RUN=0           seq 0 2    -> "0", "1", "2"  (renumbers the interleaved repeats)

A leading zero reaches that same arithmetic and is read two ways in one line:
`$(( ))` takes it as octal, /usr/bin/seq takes it as decimal.

    RUNS=00    -> seq 1 0    -> "r1" "r0"    (2 iterations: the countdown above, by another spelling)
    RUNS=010   -> seq 1 8    -> r1..r8       (ten digits named, eight arms ran)
    FIRST_RUN=010 RUNS=3    -> seq 010 10 -> "r10" (three repeats numbered 10,11,12 became one)
    RUNS=09    -> "value too great for base", 0 iterations, rc 0, run continues past the loop

A set-but-blank value does *not* reach that arithmetic: `:-` substitutes for null as
well as unset, so `TINYTITAN_MEMVAL_RUNS=""` runs the documented three repeats. That
is the honest shape and it is asserted below rather than refused -- unlike a results
tree, a default run count still measures what the report then says it measured.

These tests drive a copy of the script inside a temporary checkout holding a stub
`tools/server_launcher.sh` and no release binary, so nothing here can start a
model: the guard under test runs before the script looks for a server, and a test
that reaches the binary check instead of the guard fails with "no release binary".
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO / "benchmark" / "memval_run.sh"

LAUNCHER_STUB = """#!/usr/bin/env bash
# Stands in for tools/server_launcher.sh so the script gets past its own
# `[[ -x launcher ]]` check. If a run reaches this line the count guard did not
# fire -- and it still cannot launch anything: it reports the invasion and dies.
echo "STUB LAUNCHER INVOKED" >&2
exit 1
"""


class RepeatCountTests(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="memval-run-repeats-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        (self.tmp / "benchmark").mkdir()
        shutil.copy(SCRIPT, self.tmp / "benchmark" / "memval_run.sh")
        tools = self.tmp / "tools"
        tools.mkdir()
        launcher = tools / "server_launcher.sh"
        launcher.write_text(LAUNCHER_STUB)
        launcher.chmod(0o755)
        (self.tmp / ".build").mkdir()

    def run_script(self, **counts: str) -> subprocess.CompletedProcess:
        env = {key: value for key, value in os.environ.items() if not key.startswith("TINYTITAN_")}
        env["PATH"] = os.environ["PATH"]
        env["HOME"] = os.environ.get("HOME", str(self.tmp))
        env.update(counts)
        return subprocess.run(
            ["/bin/bash", str(self.tmp / "benchmark" / "memval_run.sh"), "book"],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
            check=False,
        )

    def assert_refused(self, proc: subprocess.CompletedProcess, name: str) -> str:
        output = proc.stdout + proc.stderr
        self.assertNotIn(
            "STUB LAUNCHER INVOKED",
            output,
            "a repeat count the script could not read still started a launch",
        )
        self.assertNotIn(
            "no release binary",
            output,
            f"the run got as far as the binary check, so {name} was never refused",
        )
        self.assertEqual(proc.returncode, 2, output)
        self.assertIn(name, output, "a refusal has to name the variable it refused")
        return output

    def test_a_zero_repeat_count_is_refused(self):
        """`seq 1 0` is not an empty list, it is a countdown: two runs, the second
        named run 0, and the script exits 0 over both records."""
        proc = self.run_script(TINYTITAN_MEMVAL_RUNS="0")
        self.assert_refused(proc, "TINYTITAN_MEMVAL_RUNS")

    def test_a_negative_repeat_count_is_refused(self):
        proc = self.run_script(TINYTITAN_MEMVAL_RUNS="-1")
        self.assert_refused(proc, "TINYTITAN_MEMVAL_RUNS")

    def test_a_non_numeric_repeat_count_is_refused(self):
        """Bash does die on these, but from inside the arithmetic, after the
        directories are made, and with a message about syntax rather than the knob."""
        output = self.assert_refused(
            self.run_script(TINYTITAN_MEMVAL_RUNS="1.5"), "TINYTITAN_MEMVAL_RUNS"
        )
        self.assertNotIn("syntax error", output, output)

    def test_a_zero_first_run_is_refused(self):
        """The documented numbering starts at 1; 0 makes `seq 0 2`, which writes
        `{arm}-r0.json` and renumbers the interleaved repeats the recipe exists for."""
        proc = self.run_script(TINYTITAN_MEMVAL_FIRST_RUN="0")
        self.assert_refused(proc, "TINYTITAN_MEMVAL_FIRST_RUN")

    def test_a_zero_padded_repeat_count_is_refused(self):
        """`00` is the countdown the guard above already refuses, spelled with two
        digits: only `== 0` matches, so it survives and `seq 1 $((1+00-1))` runs r1
        and r0. `010` is read as octal by the arithmetic and as ten by the digits the
        operator typed, so eight arms run beside a count that says ten. `09` parses
        for neither reader: the loop expands to nothing, the day's matrix runs zero
        arms and carries on to the report with exit 0."""
        for value in ("00", "010", "09"):
            output = self.assert_refused(
                self.run_script(TINYTITAN_MEMVAL_RUNS=value), "TINYTITAN_MEMVAL_RUNS"
            )
            self.assertNotIn("value too great for base", output, f"{value}: {output}")
            self.assertIn("leading zero", output, f"{value}: {output}")

    def test_a_zero_padded_first_run_is_refused(self):
        """Same helper, same arithmetic, and the two readers disagree harder here:
        `seq` takes `010` as the decimal start and `$(( ))` as the octal 8, so three
        repeats numbered 10, 11, 12 come back as one run named 10."""
        for value in ("00", "010", "09"):
            output = self.assert_refused(
                self.run_script(TINYTITAN_MEMVAL_FIRST_RUN=value), "TINYTITAN_MEMVAL_FIRST_RUN"
            )
            self.assertNotIn("value too great for base", output, f"{value}: {output}")

    def test_a_blank_count_takes_its_documented_default(self):
        """`:-` substitutes for null as well as unset, so a blank is the default --
        three repeats and numbering from 1, both of which the report then shows."""
        for name in ("TINYTITAN_MEMVAL_RUNS", "TINYTITAN_MEMVAL_FIRST_RUN"):
            proc = self.run_script(**{name: ""})
            output = proc.stdout + proc.stderr
            self.assertNotIn("STUB LAUNCHER INVOKED", output, name)
            self.assertIn("no release binary", output, f"{name}: {output}")
            self.assertNotIn("TINYTITAN_MEMVAL", output, f"{name} was refused: {output}")

    def test_the_documented_counts_are_accepted(self):
        """The recipe in docs/master-benchmark-results.md is RUNS=1 with
        FIRST_RUN=2, and the trimmed pass is RUNS=2 FIRST_RUN=2. Both have to reach
        the run, or the guard is a refusal of everything."""
        for counts in (
            {"TINYTITAN_MEMVAL_RUNS": "1", "TINYTITAN_MEMVAL_FIRST_RUN": "2"},
            {"TINYTITAN_MEMVAL_RUNS": "2", "TINYTITAN_MEMVAL_FIRST_RUN": "2"},
            {"TINYTITAN_MEMVAL_RUNS": "3"},
        ):
            proc = self.run_script(**counts)
            output = proc.stdout + proc.stderr
            self.assertNotIn("STUB LAUNCHER INVOKED", output, counts)
            self.assertIn("no release binary", output, f"{counts} was refused: {output}")

    def test_an_unset_count_still_takes_its_default(self):
        """Refusing a bad value must not turn the default into a requirement to
        set one: with neither variable present the script runs on 3 x 1."""
        proc = self.run_script()
        self.assertIn("no release binary", proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main()
