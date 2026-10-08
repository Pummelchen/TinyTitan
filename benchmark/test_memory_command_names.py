"""Gates the command dispatch of every ``benchmark/memory_*.py`` driver.

Each of these drivers is run by a person typing a word: an arm name to start a
multi-hour model run, or a report word to read what is on disk. All nine accept
any word they do not know and answer with **the report** -- measured before the
fix, seven of them exited 0 over `zzz-not-a-command` having printed a whole
table (`benchmark/memory_value.py` and its siblings: ``arm run stage prompt
completion seconds wait parameters`` with no rows), so an operator who mistyped
the arm they meant to run got a clean-looking report of a measurement that never
happened. The module docstring of `memory_value.py` made that reachable without
a typo: it documents ``python3 benchmark/memory_value.py memory``, and ``memory``
is not one of its arms, so the documented way to run the memory arm never ran it.

``memory_small_model.py`` is the same defect with a worse edge: it treats
anything that is not `book` as the other driver, so a typo there does not report
nothing -- it launches a live `control` arm of `memory_value` against whatever
port is configured, which is a different measurement from the one asked for.

These tests drive each driver's real ``__main__`` in a subprocess with a bogus
word and a temporary results directory, and assert the refusal is loud. Nothing
here loads a model or starts a server: every port is pointed at a closed port and
the only command word a test is allowed to pass is one the driver must refuse.

    cd benchmark && python3 -m unittest test_memory_command_names -v
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent

BOGUS = "zzz-not-a-command"

# driver -> (the words its refusal must name, the documented read-only word that
# must still be dispatched). The read-only words are the ones that only touch the
# results directory; an arm name is never passed, because that would run a model.
DRIVERS = {
    "memory_book.py": (["summary", "auto", "minimal", "full", "report"], "report"),
    "memory_correct.py": (["control", "auto", "report"], "report"),
    "memory_master.py": (["summary", "auto", "report", "report-all", "stats"], "report"),
    "memory_mini.py": (["serve", "run", "verify", "report"], "report"),
    "memory_projects.py": (["control", "auto", "report"], "report"),
    "memory_sim.py": (["validate", "tail", "detail", "compare"], "compare"),
    "memory_small_model.py": (["book", "value"], None),
    "memory_volume.py": (["control", "auto", "full", "report"], "report"),
    "memory_value.py": (["control", "auto", "minimal", "full", "report"], "report"),
}

REFUSAL = "unknown command"


def env(temp: str) -> dict[str, str]:
    return dict(
        os.environ,
        TINYTITAN_PORT="1",
        TINYTITAN_MINI_PORT="1",
        TINYTITAN_MEMVAL_RESULTS=f"{temp}/results",
        TINYTITAN_MINI_RESULTS=f"{temp}/mini",
    )


class CommandDispatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def drive(self, driver: str, *words: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(REPO / "benchmark" / driver), *words],
            capture_output=True,
            text=True,
            env=env(self.temp.name),
            cwd=REPO,
            timeout=180,
            check=False,
        )

    def test_an_unknown_command_is_refused_not_answered_with_a_report(self):
        for driver, (names, _) in DRIVERS.items():
            with self.subTest(driver=driver):
                result = self.drive(driver, BOGUS)
                combined = result.stdout + result.stderr
                self.assertNotEqual(
                    result.returncode, 0, f"{driver} exited 0 over {BOGUS!r}: {combined[:200]}"
                )
                self.assertIn(REFUSAL, combined, f"{driver} refused without saying so")
                self.assertIn(BOGUS, combined, f"{driver} did not name the word it refused")
                for name in names:
                    self.assertIn(
                        name, combined, f"{driver}'s refusal did not list {name!r} as a choice"
                    )
                self.assertNotIn(
                    "Traceback",
                    combined,
                    f"{driver} answered by crashing rather than refusing: {combined[:200]}",
                )

    def test_a_typo_does_not_launch_a_model_arm(self):
        """The `memory_small_model.py` shape: its fallback is the other driver's arm."""
        result = self.drive("memory_small_model.py", BOGUS)
        combined = result.stdout + result.stderr
        self.assertNotIn(
            "against port",
            combined,
            f"a bogus word reached the launch banner: {combined[:300]}",
        )

    def test_the_read_only_word_is_still_dispatched(self):
        for driver, (_, read_only) in DRIVERS.items():
            if read_only is None:
                continue
            with self.subTest(driver=driver):
                result = self.drive(driver, read_only)
                combined = result.stdout + result.stderr
                self.assertNotIn(
                    REFUSAL,
                    combined,
                    f"{driver} refused its own documented {read_only!r} command",
                )


if __name__ == "__main__":
    unittest.main()
