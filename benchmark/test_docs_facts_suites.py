"""AUD-211: a python suite that nobody lists in CI simply never runs, and nothing said so.

`.github/workflows/ci.yml` invokes the benchmark suites by name -- `python3 -m unittest
test_a test_b …` -- so registration is a hand-maintained list, and the failure mode is
silent: a suite committed and never listed passes locally, is quoted in a commit message
as evidence, and runs on no machine ever. This audit already filed that shape for the
launcher suites (AUD-126) and registered its own two suites, yet on the day this gate
was written `git ls-files benchmark/test_*.py` counted 36 and the workflow named 19, of
which ten suites with real assertions ran in no CI step and in no gate.

`tools/docs-facts.py`'s `python_suite_registration()` is the rule that stops it being
filed again: every tracked `benchmark/test_*.py` is either named in the workflow or
listed in `tools/python-suites-unrun.txt` with the reason it cannot run there; a stale
exemption, a dead name in the workflow, an exemption with no reason, and a repository
with no suites at all all fail. The tests below drive it against a fixture repository,
and the last test drives it against this one, because a gate that only ever reads a
fixture proves nothing about the tree it ships in.

Run from `benchmark/`:

    python3 -m unittest test_docs_facts_suites
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
GATE = REPO / "tools" / "docs-facts.py"
CI_REL = ".github/workflows/ci.yml"
EXEMPT_REL = "tools/python-suites-unrun.txt"


def gate_module():
    spec = importlib.util.spec_from_file_location("docs_facts_suites_gate", GATE)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {GATE} as a module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


module = gate_module()


class Fixture:
    """A throwaway git repository whose only tracked facts are a workflow, some
    suites, and the exemption file. Nothing here touches this history."""

    def __init__(self, ci_lines: str, suites=(), exemptions: str | None = None):
        self.root = pathlib.Path(tempfile.mkdtemp(prefix="suites-gate-"))
        self.write(
            CI_REL,
            f"jobs:\n  test:\n    steps:\n      - run: |\n          cd benchmark\n{ci_lines}",
        )
        for name in suites:
            self.write(f"benchmark/{name}.py", "import unittest\n")
        if exemptions is not None:
            self.write(EXEMPT_REL, exemptions)
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        subprocess.run(["git", "add", "-A"], cwd=self.root, check=True)
        env = dict(
            os.environ,
            GIT_AUTHOR_NAME="f",
            GIT_AUTHOR_EMAIL="f@f",
            GIT_COMMITTER_NAME="f",
            GIT_COMMITTER_EMAIL="f@f",
        )
        subprocess.run(["git", "commit", "-q", "-m", "fixture"], cwd=self.root, check=True, env=env)

    def write(self, rel: str, text: str) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def result(self):
        return module.python_suite_registration(root=str(self.root))

    def close(self):
        import shutil

        shutil.rmtree(self.root, ignore_errors=True)


def step(*names: str) -> str:
    return "          python3 -m unittest " + " ".join(names) + "\n"


class SuiteRegistrationTests(unittest.TestCase):
    def tearDown(self):
        if getattr(self, "fixture", None):
            self.fixture.close()

    def listed(self, result):
        """The FAIL lines out of `python_suite_registration`'s answer, whether the
        caller unpacked the (lines, registered, exempt) tuple or passed it whole."""
        lines = result[0] if isinstance(result, tuple) else result
        return [line for line in lines if line.startswith("FAIL")]

    def test_a_registered_suite_reports_nothing(self):
        self.fixture = Fixture(step("test_alpha"), suites=["test_alpha"])
        lines, registered, exempt = self.fixture.result()
        self.assertEqual(self.listed(lines), [], lines)
        self.assertEqual((registered, exempt), (1, 0))

    def test_an_unregistered_suite_is_named(self):
        self.fixture = Fixture(step("test_alpha"), suites=["test_alpha", "test_beta"])
        lines, _, _ = self.fixture.result()
        fails = self.listed(lines)
        self.assertEqual(len(fails), 1, lines)
        self.assertIn("benchmark/test_beta.py", fails[0])

    def test_an_exemption_with_a_reason_is_accepted(self):
        self.fixture = Fixture(
            step("test_alpha"),
            suites=["test_alpha", "test_needs_model"],
            exemptions="test_needs_model\tstarts a server; a model run, never fetched in CI\n",
        )
        lines, registered, exempt = self.fixture.result()
        self.assertEqual(self.listed(lines), [], lines)
        self.assertEqual((registered, exempt), (1, 1))

    def test_an_exemption_without_a_reason_fails(self):
        self.fixture = Fixture(
            step("test_alpha"),
            suites=["test_alpha", "test_beta"],
            exemptions="test_beta\n",
        )
        # Two findings, not one: the row is refused *and* the suite is still
        # unregistered, because an exemption that says nothing exempts nothing.
        fails = self.listed(self.fixture.result()[0])
        self.assertEqual(len(fails), 2, fails)
        self.assertIn("reason", fails[0])
        self.assertIn("test_beta", fails[1])

    def test_a_stale_exemption_fails(self):
        """A suite that has been registered must leave the list, or the list becomes
        the place suites go to be forgotten."""
        self.fixture = Fixture(
            step("test_alpha", "test_beta"),
            suites=["test_alpha", "test_beta"],
            exemptions="test_beta\tleft over from before it was registered\n",
        )
        fails = self.listed(self.fixture.result())
        self.assertEqual(len(fails), 1, fails)
        self.assertIn("stale", fails[0])

    def test_an_exemption_for_a_missing_file_fails(self):
        self.fixture = Fixture(
            step("test_alpha"),
            suites=["test_alpha"],
            exemptions="test_ghost\tsuite deleted, row left behind\n",
        )
        fails = self.listed(self.fixture.result())
        self.assertEqual(len(fails), 1, fails)
        self.assertIn("test_ghost", fails[0])

    def test_a_dead_name_in_the_workflow_fails(self):
        self.fixture = Fixture(step("test_alpha", "test_renamed_away"), suites=["test_alpha"])
        fails = self.listed(self.fixture.result())
        self.assertEqual(len(fails), 1, fails)
        self.assertIn("test_renamed_away", fails[0])

    def test_a_repository_with_no_suites_fails(self):
        """A gate that found nothing to check must not read as a pass.

        Asserting only the count here would survive the guard being deleted: the
        fixture's workflow names test_alpha, and with no tracked suites that name is
        dead, so the dead-name arm reports one FAIL wearing the same number. Pin the
        message, because which arm fired is the whole point.
        """
        self.fixture = Fixture(step("test_alpha"), suites=[])
        fails = self.listed(self.fixture.result())
        self.assertEqual(len(fails), 1, fails)
        self.assertIn("no benchmark/test_*.py is tracked", fails[0])

    def test_a_workflow_that_cannot_be_read_fails(self):
        self.fixture = Fixture(step("test_alpha"), suites=["test_alpha"])
        (self.fixture.root / CI_REL).unlink()
        fails = self.listed(self.fixture.result())
        self.assertEqual(len(fails), 1, fails)
        self.assertIn(CI_REL, fails[0])

    def test_a_continued_invocation_registers_every_suite_it_lists(self):
        """The real workflow wraps its `python3 -m unittest` lines with backslashes;
        reading only the first line would leave the wrapped names unregistered and
        invite someone to fix the gate by widening it to the whole file."""
        self.fixture = Fixture(
            "          python3 -m unittest test_alpha \\\n            test_beta\n",
            suites=["test_alpha", "test_beta"],
        )
        lines, registered, exempt = self.fixture.result()
        self.assertEqual(self.listed(lines), [], lines)
        self.assertEqual((registered, exempt), (2, 0))

    def test_a_name_in_a_comment_registers_nothing(self):
        """A mention is not a run. The workflow's own comment about a suite family is
        the shape that makes this worth pinning: a gate that read comments would pass
        on the very suites this finding is about."""
        self.fixture = Fixture(
            "# the suite test_beta is discussed here\n" + step("test_alpha"),
            suites=["test_alpha", "test_beta"],
        )
        fails = self.listed(self.fixture.result())
        self.assertEqual(len(fails), 1, fails)
        self.assertIn("test_beta", fails[0])

    def test_this_repository_is_registered(self):
        """The arm has to be true of the tree it ships in, or it is a fixture that
        flatters itself."""
        lines, registered, exempt = module.python_suite_registration(root=str(REPO))
        self.assertEqual([line for line in lines if line.startswith("FAIL")], [], lines)
        tracked = subprocess.run(
            ["git", "ls-files", "benchmark/test_*.py"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        self.assertEqual(registered + exempt, len(tracked), (registered, exempt, tracked))
        self.assertGreaterEqual(registered, 29)


if __name__ == "__main__":
    unittest.main()
