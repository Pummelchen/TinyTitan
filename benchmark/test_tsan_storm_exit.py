"""Gates `tools/tsan-storm.sh`'s verdict when its instrumented runs refuse.

The storm is the harness the audit's sanitizer conclusion rests on: it runs the
sanitized server test bundle many times at once and says whether the TT-001 race
reproduced. Its verdict was derived from one question only — did any log contain
`WARNING: ThreadSanitizer` — while every run's exit status went into `wait` and
nowhere else. A helper that dies before running a single test produces no warning
and no pass, so `tools/tsan-storm.sh --runs 1 --parallel 1 --filter NoSuchSuiteZZZ`
answered `Clean: no report in 1 instrumented runs` and exited 0 against runs that
each exited 69. The script's own header reserves exit 2 for a setup error; this is
one, and the header already promised it.

These tests drive a copy of the script in a temporary checkout whose
`xcode-select -p` is a stub answering with a fake SDK root, so the helper it
launches is a script the test controls: it can refuse, pass, or report a race
without Xcode, without a built bundle, and without a model.

    cd benchmark && python3 -m unittest test_tsan_storm_exit -v
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
DRIVER = REPO / "tools" / "tsan-storm.sh"

STUB_HELPER = """#!/usr/bin/env bash
# Stands in for swiftpm-testing-helper. MODE decides what the instrumented run does:
# refuse (die before running anything), pass, report the race it exists to find,
# or refuse-second — pass for whoever claims the lock first, refuse for the rest,
# which is how a single refusal among parallel runs is arranged.
mode="$(cat "$MODE_FILE")"
if [ "$mode" = refuse-second ] && ! mkdir "$CLAIM_DIR" 2>/dev/null; then
  mode=refuse
fi
case "$mode" in
  refuse)
    exit 69
    ;;
  report)
    echo "WARNING: ThreadSanitizer: data race (pid=1234)"
    echo "  SUMMARY: ThreadSanitizer: data race main"
    exit 66
    ;;
  *)
    echo "Test run with 117 tests passed"
    exit 0
    ;;
esac
"""


class StormVerdictTests(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="tsan-storm-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        sdk = self.tmp / "sdk"
        helper = (
            sdk / "Toolchains/XcodeDefault.xctoolchain/usr/libexec/swift/pm/swiftpm-testing-helper"
        )
        helper.parent.mkdir(parents=True)
        helper.write_text(STUB_HELPER)
        helper.chmod(0o755)
        (sdk / "Platforms/MacOSX.platform/Developer/Library/Frameworks/XCTest.framework").mkdir(
            parents=True
        )
        (sdk / "Platforms/MacOSX.platform/Developer/usr/lib").mkdir(parents=True)

        xcode_select = self.tmp / "bin/xcode-select"
        xcode_select.parent.mkdir()
        xcode_select.write_text(f'#!/usr/bin/env bash\necho "{sdk}"\n')
        xcode_select.chmod(0o755)

        bundle = (
            self.tmp
            / ".build/out/Products/Debug/TinyTitanServerTests.xctest/Contents/MacOS"
            / "TinyTitanServerTests"
        )
        bundle.parent.mkdir(parents=True)
        bundle.write_text("#!/usr/bin/env bash\nexit 0\n")
        bundle.chmod(0o755)
        # The script looks for the runtime beside the bundle's MacOS directory,
        # at Contents/Frameworks, so the stub goes exactly where it looks.
        runtime = bundle.parent.parent / "Frameworks/libclang_rt.tsan_osx_dynamic.dylib"
        runtime.parent.mkdir(parents=True)
        runtime.write_text("stub")

        tools = self.tmp / "tools"
        tools.mkdir()
        shutil.copy(DRIVER, tools / "tsan-storm.sh")
        shutil.copy(REPO / "tools/tsan-suppressions.txt", tools / "tsan-suppressions.txt")

        self.mode = self.tmp / "mode"
        self.mode.write_text("pass\n", encoding="utf-8")
        self.claims = 0

    def run_storm(self, mode: str, *args: str) -> subprocess.CompletedProcess:
        self.mode.write_text(f"{mode}\n", encoding="utf-8")
        claim = self.tmp / f"claim-{self.claims}"
        self.claims += 1
        env = dict(
            os.environ,
            PATH=f"{self.tmp / 'bin'}:{os.environ['PATH']}",
            MODE_FILE=str(self.mode),
            CLAIM_DIR=str(claim),
            # The storm keeps its logs under $TMPDIR on the error paths these tests
            # exercise. Pointing TMPDIR at the test's own directory is what makes
            # them hermetic: the kept logs are cleaned with everything else.
            TMPDIR=str(self.tmp),
        )
        return subprocess.run(
            ["/bin/bash", str(self.tmp / "tools/tsan-storm.sh"), *args],
            capture_output=True,
            text=True,
            env=env,
            cwd=self.tmp,
            timeout=120,
            check=False,
        )

    def test_a_run_that_executed_nothing_is_not_a_clean_verdict(self):
        result = self.run_storm("refuse", "--runs", "1", "--parallel", "1")
        self.assertNotEqual(result.returncode, 0, "a storm that ran no test must not exit clean")
        self.assertNotIn("Clean:", result.stdout, f"refused runs read as clean: {result.stdout}")
        self.assertEqual(result.returncode, 2, "the header reserves 2 for a setup error")
        self.assertIn("69", result.stdout + result.stderr, "the refusal names the status it got")

    def test_one_refused_run_among_clean_ones_still_fails_the_storm(self):
        result = self.run_storm("pass", "--runs", "1", "--parallel", "2")
        self.assertEqual(result.returncode, 0, "the fixture itself: two passing runs read clean")
        # Exactly one of two parallel runs refuses: the stub helper lets the first
        # process to claim the lock pass and makes the other exit 69.
        result = self.run_storm("refuse-second", "--runs", "1", "--parallel", "2")
        self.assertEqual(
            result.returncode, 2, f"one refusal among two runs: {result.stdout}{result.stderr}"
        )
        self.assertNotIn("Clean:", result.stdout)
        self.assertIn("1 of 2", result.stdout + result.stderr)

    def test_a_storm_that_runs_nothing_is_not_a_clean_verdict(self):
        result = self.run_storm("pass", "--runs", "0", "--parallel", "1")
        self.assertEqual(result.returncode, 2, f"--runs 0 read as: {result.stdout}")
        self.assertNotIn("Clean:", result.stdout)

    def test_a_non_numeric_run_count_is_refused_before_anything_launches(self):
        result = self.run_storm("pass", "--runs", "two")
        self.assertEqual(result.returncode, 2, f"--runs two read as: {result.stdout}")
        self.assertNotIn("Running", result.stdout)

    def test_a_reproduced_race_still_exits_one(self):
        result = self.run_storm("report", "--runs", "1", "--parallel", "1")
        self.assertEqual(result.returncode, 1)
        self.assertIn("REPRODUCED", result.stdout)

    def test_the_clean_verdict_counts_the_runs_that_actually_ran(self):
        result = self.run_storm("pass", "--runs", "2", "--parallel", "1")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("Clean: no report in 2 instrumented runs", result.stdout)


if __name__ == "__main__":
    unittest.main()
