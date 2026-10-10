"""AUD-266: the release's compiler-warning gate must refuse a log it cannot read.

`tools/build_library.sh:81` and `tools/release.sh:317` each decided "this build
emitted no compiler warnings" with the same line:

    grep -qE '^[^ ]+\\.(swift|metal|c|h|m|mm):[0-9]+:[0-9]+: warning:' "$LOG" \\
      && die "the build emitted compiler warnings"

grep answers three different questions with three different statuses -- 0 found a
warning, 1 found none, 2 could not read the file -- and this shape can only act on
0. Measured on this machine, in a script that sets `set -euo pipefail` exactly like
`build_library.sh` (and in `release.sh`, which sets `set -uo pipefail` without the
`-e`): a missing log, a 0-byte log, and a log the process may not open each left
grep at status 2 or 1, `&&` never reached `die`, and the script printed
REACHED_STAGING and exited 0. That is the release publishing a binary distribution
while its only warning check reports nothing, and it is AUD-265's root cause in the
neighbouring script: a verdict taken from a command whose failure status the caller
cannot see. `grep`'s own "Permission denied" goes to stderr two lines above a
`== staged ==` header, which is how a skip like this reads as a pass.

The second hole is vacuity, and it needs no fault at all: a truncated log -- what a
`tee` onto a full disk leaves -- is a file that records no build, and "no warning
line in it" is not evidence about warnings. Every one of the eleven real build logs
still on this disk (`.build/releases/*.buildlog`, `.build/*/*.buildlog`,
`.build/library-dist.buildlog`) is plain ASCII and carries at least one line
starting `Build complete!`, so that marker is what separates a log that holds a
finished build from one that holds a fragment.

These tests drive the extracted function over constructed logs, so they build
nothing and run no compiler. The warning *pattern* is deliberately not widened
here: the set of extensions is exactly the ones this package compiles (measured --
no `.cpp`, `.cc`, `.cxx` or `.S` under `sources/`), and a broader net would change
what a release refuses to ship, which is the operator's call rather than a test's.
"""

import pathlib
import shutil
import subprocess
import tempfile
import unittest

HELPER = pathlib.Path(__file__).resolve().parent.parent / "tools" / "assert-build-log.sh"

CLEAN_TAIL = "Build complete! (118.85 sec)\n"
COMPILER_WARNING = (
    "/Users/x/TinyTitan/sources/TinyTitan/Foo.swift:12:5: warning: "
    "variable 'bar' was never used; consider replacing with '_' or removing it\n"
)


def drive(log: pathlib.Path) -> subprocess.CompletedProcess:
    """Source the helper and call the gate, in a script that mirrors each caller."""
    script = f'. "{HELPER}"; assert_clean_build_log "{log}" probe && echo NOT_REFUSED'
    return subprocess.run(["/bin/bash", "-c", script], capture_output=True, text=True, check=False)


class BuildLogGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = pathlib.Path(tempfile.mkdtemp(prefix="tt-buildlog-"))
        self.addCleanup(shutil.rmtree, self.dir, True)

    def write(self, name: str, body: str) -> pathlib.Path:
        path = self.dir / name
        path.write_text(body)
        return path

    def test_clean_log_passes_and_says_what_it_read(self) -> None:
        log = self.write("clean.log", "[448 / 930] Compiling TinyTitan Engine.swift\n" + CLEAN_TAIL)
        result = drive(log)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("NOT_REFUSED", result.stdout)
        out = result.stdout + result.stderr
        self.assertIn("no compiler warnings", out)
        # The count is the evidence that a build was read, so it has to be a number
        # the reader can see -- `wc -l` pads, and a report of "read,   2 line(s)" is
        # the same claim while looking like nothing was counted.
        self.assertIn("read, 2 line(s)", out)

    def test_a_compiler_warning_is_refused_and_named(self) -> None:
        log = self.write("warn.log", COMPILER_WARNING + CLEAN_TAIL)
        result = drive(log)
        self.assertEqual(1, result.returncode, result.stdout + result.stderr)
        self.assertNotIn("NOT_REFUSED", result.stdout)
        self.assertIn("probe", result.stderr)
        self.assertIn("warning", result.stderr)

    def test_a_missing_log_is_refused_as_missing(self) -> None:
        result = drive(self.dir / "never-written.log")
        self.assertEqual(1, result.returncode, result.stdout)
        self.assertNotIn("NOT_REFUSED", result.stdout)
        self.assertIn("no build log", result.stderr)

    def test_an_unreadable_log_is_not_read_as_a_clean_build(self) -> None:
        log = self.write("sealed.log", COMPILER_WARNING + CLEAN_TAIL)
        log.chmod(0o000)
        self.addCleanup(log.chmod, 0o644)
        result = drive(log)
        self.assertEqual(1, result.returncode, result.stdout + result.stderr)
        self.assertNotIn("NOT_REFUSED", result.stdout)
        self.assertIn("cannot read", result.stderr)

    def test_an_empty_log_refuses_rather_than_passing(self) -> None:
        log = self.write("empty.log", "")
        result = drive(log)
        self.assertEqual(1, result.returncode, result.stdout + result.stderr)
        self.assertNotIn("NOT_REFUSED", result.stdout)
        self.assertIn("Build complete!", result.stderr)

    def test_a_truncated_log_refuses_rather_than_passing(self) -> None:
        # What a `tee` onto a full disk leaves: real build lines, no finished build.
        log = self.write("cut.log", "[448 / 930] Compiling TinyTitan Engine.swift\n[449 / ")
        result = drive(log)
        self.assertEqual(1, result.returncode, result.stdout + result.stderr)
        self.assertNotIn("NOT_REFUSED", result.stdout)
        self.assertIn("Build complete!", result.stderr)

    def test_a_marker_that_is_not_a_line_start_does_not_count(self) -> None:
        # The anchor is what makes the check "a build finished here" rather than "the
        # phrase appears somewhere in this text".
        log = self.write(
            "prose.log",
            "note: the phrase Build complete! appears in this fragment\n[449 / 930] linking\n",
        )
        result = drive(log)
        self.assertEqual(1, result.returncode, result.stdout + result.stderr)
        self.assertNotIn("NOT_REFUSED", result.stdout)
        self.assertIn("Build complete!", result.stderr)

    def test_a_broken_pattern_refuses_rather_than_reporting_clean(self) -> None:
        # grep answers an invalid expression with status 2, the same status as "I
        # cannot read this", so a corrupted pattern cannot be mistaken for a clean
        # build. This is the branch that catches someone editing the pattern itself.
        log = self.write("clean.log", "[448 / 930] Compiling TinyTitan Engine.swift\n" + CLEAN_TAIL)
        script = (
            f'. "{HELPER}"; BUILD_WARNING_PATTERN="(unclosed"; '
            f'assert_clean_build_log "{log}" probe && echo NOT_REFUSED'
        )
        result = subprocess.run(
            ["/bin/bash", "-c", script], capture_output=True, text=True, check=False
        )
        self.assertEqual(1, result.returncode, result.stdout + result.stderr)
        self.assertNotIn("NOT_REFUSED", result.stdout)
        # "cannot check", not "cannot read": the file opened fine, the pattern did
        # not run against it, and a message that blamed the log would misreport it.
        self.assertIn("cannot check", result.stderr)
        self.assertIn("status 2", result.stderr)

    def test_every_real_shape_of_warning_the_net_hunts_is_matched(self) -> None:
        # The extension set is this package's compiled set, measured from sources/.
        for name in ("Foo.swift", "Kernels.c", "Shader.metal", "Bridge.h", "Hook.m", "Patch.mm"):
            log = self.write(
                "shape.log",
                f"/Users/x/TinyTitan/sources/TinyTitan/{name}:3:4: warning: note\n" + CLEAN_TAIL,
            )
            result = drive(log)
            self.assertEqual(
                1, result.returncode, f"{name} slipped through: {result.stdout}{result.stderr}"
            )

    def test_a_warning_marker_that_is_not_a_diagnostic_does_not_refuse(self) -> None:
        # `Build complete!` logs quote target names; a line that merely contains the
        # word must not be mistaken for a diagnostic or the gate fails every release.
        log = self.write(
            "prose.log", "[448 / 930] Compiling warning: not a diagnostic path\n" + CLEAN_TAIL
        )
        result = drive(log)
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)

    def test_sourcing_it_still_stops_a_caller_that_has_no_errexit(self) -> None:
        # release.sh runs `set -uo pipefail`: no `-e`, so the only thing that keeps a
        # warned build from being staged is the helper's own `exit`.
        log = self.write("warn.log", COMPILER_WARNING + CLEAN_TAIL)
        script = f'set -uo pipefail\n. "{HELPER}"\nassert_clean_build_log "{log}" probe\necho NOT_REACHED\n'
        runner = self.write("caller.sh", script)
        result = subprocess.run(
            ["/bin/bash", str(runner)], capture_output=True, text=True, check=False
        )
        self.assertEqual(1, result.returncode, result.stdout + result.stderr)
        self.assertNotIn("NOT_REACHED", result.stdout)

    def test_sourcing_it_stops_the_errexit_caller_too(self) -> None:
        # build_library.sh runs `set -euo pipefail`, where a non-zero status inside a
        # function must not abort before the message is printed.
        log = self.write("empty.log", "")
        script = f'set -euo pipefail\n. "{HELPER}"\nassert_clean_build_log "{log}" probe\necho NOT_REACHED\n'
        runner = self.write("caller.sh", script)
        result = subprocess.run(
            ["/bin/bash", str(runner)], capture_output=True, text=True, check=False
        )
        self.assertEqual(1, result.returncode, result.stdout + result.stderr)
        self.assertNotIn("NOT_REACHED", result.stdout)
        self.assertIn("Build complete!", result.stderr)


class BuildLogCallerTests(unittest.TestCase):
    """The gate only protects the release if the two callers actually call it."""

    def callers(self):
        tools = HELPER.parent
        return [tools / "build_library.sh", tools / "release.sh"]

    def test_the_pattern_lives_in_one_file(self) -> None:
        # Both callers carried the same grep verbatim, which is how a fix to one
        # leaves the other exactly as broken. Single-source it or it drifts back.
        holders = [
            path.name
            for path in sorted(HELPER.parent.glob("*.sh"))
            if "metal|c|h|m|mm" in path.read_text()
        ]
        self.assertEqual(["assert-build-log.sh"], holders)

    def test_each_caller_sources_the_helper_and_calls_it(self) -> None:
        for path in self.callers():
            text = path.read_text()
            self.assertIn("assert-build-log.sh", text, f"{path.name} does not source the gate")
            self.assertIn("assert_clean_build_log", text, f"{path.name} does not call the gate")


if __name__ == "__main__":
    unittest.main()
