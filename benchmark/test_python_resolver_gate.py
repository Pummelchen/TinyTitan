#!/usr/bin/env python3.13
"""AUD-295: `tools/lib/python.sh` tests the interpreters it finds and none it is told.

The library's own header says the candidates are tried newest-first and "each one
is *tested*: it must be at least the required version and must import the
dependencies", and that "`TINYTITAN_PYTHON` overrides the search entirely". The
second sentence is about the *search*; measured at HEAD `5a3c026` it turns out to
cover the tests as well, so the one interpreter chosen by the operator is the one
interpreter never checked:

    TINYTITAN_PYTHON=                     status  returned
    a path that does not exist            0       that path
    an interpreter below the version floor 0      that path
    an interpreter without the stack      0       that path
    a directory                           0       that path

`tools/install_models.sh:446` then hands the value straight to
`"$python" tools/prepare_*.py`, which is the raw "command not found" the call site's
comment says resolving once exists to prevent -- and it arrives mid-install, after
the download has been arranged. A second, smaller defect is in the refusal the
search does print: its "tried (newest first)" line names six candidates while the
loop tries seven, so the message under-reports the search it just performed. `python`
is the omitted name, and the search can select it.

These tests pin both: the override answers non-zero and names the value and the
reason, and the refusal's list is derived from the library's own `command -v` lines
rather than restated. The screens that already work are pinned unchanged -- newest
first, the version floor, executability -- so a change to any of them is a finding.

No model, no install, nothing fetched. Every case sources the real library in
/bin/bash (3.2.57 here, the shell the portability gate cares about) against stub
interpreters in a temp directory; a stub runs the library's probe for real and only
claims a version and a package set.
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
LIBRARY = ROOT / "tools" / "lib" / "python.sh"
NAMES = ["python3.14", "python3.13", "python3.12", "python3.11", "python3.10", "python3", "python"]

SHIM = """#!{python}
import sys

CLAIM = "{version}"
HAS_STACK = {stack}

args = sys.argv[1:]
if args[:1] == ["-c"]:
    major, _, minor = CLAIM.partition(".")
    sys.version_info = (int(major), int(minor), 0, "final", 0)
    if not HAS_STACK:
        for gone in ("numpy", "ml_dtypes", "safetensors"):
            sys.modules[gone] = None
    exec(compile(args[1], "<probe>", "exec"), {{"__name__": "__main__"}})
    raise SystemExit(0)
print("Python " + CLAIM)
"""


def _write_stub(directory: pathlib.Path, name: str, version: str, stack: bool) -> pathlib.Path:
    path = directory / name
    path.write_text(SHIM.format(python=sys.executable, version=version, stack=repr(stack)))
    path.chmod(0o755)
    return path


class ResolverCase(unittest.TestCase):
    """Run the real library in bash against stub interpreters in a temp dir."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="aud295-")
        self.addCleanup(self._tmp.cleanup)
        # Resolved once so the paths in PATH, the paths the stubs write, and the
        # path `command -v` answers with are the same string: /tmp is a symlink
        # here, and a comparison between the two forms fails for no reason.
        self.bin = (pathlib.Path(self._tmp.name) / "bin").resolve()
        self.bin.mkdir()

    def stub(self, name: str, version: str = "3.13", stack: bool = True) -> pathlib.Path:
        return _write_stub(self.bin, name, version, stack)

    def resolve(self, extra: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        env = {
            "PATH": f"{self.bin}{os.pathsep}/usr/bin:/bin",
            "HOME": os.environ.get("HOME", "/"),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        env.update(extra or {})
        return subprocess.run(
            [
                "/bin/bash",
                "--norc",
                "--noprofile",
                "-c",
                'source "$1"; tinytitan_resolve_python',
                "_",
                str(LIBRARY),
            ],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )


class TheOverrideIsTestedLikeAnythingElse(ResolverCase):
    """The five values the library used to return without asking them anything."""

    @staticmethod
    def reason(stderr: str) -> str:
        """What the refusal says the interpreter itself answered.

        Read out of its own line on purpose: the refusal also prints the
        requirements, so asserting a keyword against the whole message would be
        satisfied by the `need:` line rather than by the cause.
        """
        for line in stderr.splitlines():
            if "it answered:" in line:
                return line.split("it answered:", 1)[1].strip()
        raise ValueError(f"the refusal names no cause:\n{stderr}")

    def test_a_path_that_does_not_exist_is_refused_with_a_reason(self) -> None:
        missing = str(self.bin / "no-such-interpreter")
        answer = self.resolve({"TINYTITAN_PYTHON": missing})
        self.assertNotEqual(answer.returncode, 0, f"accepted a nonexistent path: {answer.stdout!r}")
        self.assertIn(missing, answer.stderr)
        self.assertIn("such file", self.reason(answer.stderr).lower())

    def test_an_interpreter_below_the_floor_is_refused_by_name(self) -> None:
        old = str(self.stub("python3.9claim", "3.9").resolve())
        answer = self.resolve({"TINYTITAN_PYTHON": old})
        self.assertNotEqual(answer.returncode, 0, "accepted 3.9 against a 3.10 floor")
        self.assertIn(old, answer.stderr)
        self.assertIn("3.10", answer.stderr)

    def test_an_interpreter_without_the_analysis_stack_is_refused_with_a_reason(self) -> None:
        bare = str(self.stub("python3.13bare", "3.13", stack=False).resolve())
        answer = self.resolve({"TINYTITAN_PYTHON": bare})
        self.assertNotEqual(
            answer.returncode, 0, "accepted an interpreter that cannot import ml_dtypes"
        )
        self.assertIn(bare, answer.stderr)
        cause = self.reason(answer.stderr).lower()
        # Which of the three packages is missing is the machine's choice -- the
        # probe imports them in one statement, so the refusal is held to naming a
        # missing module, not to naming one particular module.
        self.assertTrue(
            "modulenotfound" in cause or "no module named" in cause,
            f"the refusal does not say the stack is missing: [{self.reason(answer.stderr)}]",
        )

    def test_an_interpreter_that_cannot_be_run_is_refused_with_a_reason(self) -> None:
        unrunnable = self.stub("python3.13noexec", "3.13")
        unrunnable.chmod(0o644)
        answer = self.resolve({"TINYTITAN_PYTHON": str(unrunnable.resolve())})
        self.assertNotEqual(answer.returncode, 0, "accepted a file that is not executable")
        self.assertIn(str(unrunnable.resolve()), answer.stderr)
        cause = self.reason(answer.stderr).lower()
        # Either the library's own screen or the shell's answer to an EACCES exec
        # names the cause; a refusal that says nothing does not.
        self.assertTrue(
            "executable" in cause or "permission" in cause,
            f"the refusal does not say why it cannot run: [{self.reason(answer.stderr)}]",
        )

    def test_a_directory_is_not_an_interpreter(self) -> None:
        adir = self.bin / "somedir"
        adir.mkdir()
        answer = self.resolve({"TINYTITAN_PYTHON": str(adir)})
        self.assertNotEqual(answer.returncode, 0, f"accepted a directory: {answer.stdout!r}")
        self.assertIn(str(adir), answer.stderr)

    def test_a_usable_override_is_still_returned_verbatim(self) -> None:
        good = str(self.stub("python3.13", "3.13").resolve())
        answer = self.resolve({"TINYTITAN_PYTHON": good})
        self.assertEqual(answer.returncode, 0, answer.stderr)
        self.assertEqual(answer.stdout, good)


class TheSearchWasAlreadyRight(ResolverCase):
    """Pins, so a change to the screens the loop applies is a finding."""

    def test_the_newest_usable_candidate_wins_when_two_of_them_work(self) -> None:
        newest = self.stub("python3.14", "3.14").resolve()
        self.stub("python3.13", "3.13")
        answer = self.resolve()
        self.assertEqual(answer.returncode, 0, answer.stderr)
        self.assertEqual(answer.stdout, str(newest))

    def test_newest_first_and_the_dependency_screen_reach_past_a_stackless_candidate(self) -> None:
        self.stub("python3.14", "3.14", stack=False)
        older = self.stub("python3.13", "3.13").resolve()
        answer = self.resolve()
        self.assertEqual(answer.returncode, 0, answer.stderr)
        self.assertEqual(answer.stdout, str(older))

    def test_the_version_floor_reaches_past_a_candidate_below_it(self) -> None:
        self.stub("python3.14", "3.9")
        older = self.stub("python3.13", "3.13").resolve()
        answer = self.resolve()
        self.assertEqual(answer.returncode, 0, answer.stderr)
        self.assertEqual(answer.stdout, str(older))

    def test_a_candidate_that_is_not_executable_is_skipped(self) -> None:
        self.stub("python3.14", "3.14", stack=False)
        older = self.stub("python3.12", "3.12").resolve()
        unrunnable = self.stub("python3.13", "3.13")
        unrunnable.chmod(0o644)
        answer = self.resolve()
        self.assertEqual(answer.returncode, 0, answer.stderr)
        self.assertEqual(answer.stdout, str(older))

    def test_the_bare_python_name_is_a_candidate_and_can_win(self) -> None:
        for name in NAMES[:-1]:
            self.stub(name, "3.9")
        self.stub("python", "3.13")
        answer = self.resolve()
        self.assertEqual(answer.returncode, 0, answer.stderr)
        self.assertEqual(answer.stdout, str(self.bin / "python"))

    def test_nothing_qualifying_answers_nonzero_and_prints_nothing_on_stdout(self) -> None:
        for name in NAMES:
            self.stub(name, "3.9")
        answer = self.resolve()
        self.assertNotEqual(answer.returncode, 0)
        self.assertEqual(answer.stdout, "")
        self.assertIn("no usable Python interpreter", answer.stderr)


class TheRefusalReportsItsOwnSearch(ResolverCase):
    def test_the_tried_line_names_every_candidate_the_loop_tries(self) -> None:
        lines = LIBRARY.read_text()
        tried = re.findall(r"command -v ([a-z0-9.]+)", lines)
        self.assertGreaterEqual(len(tried), 2, "no candidate list to compare against")
        for name in NAMES:
            self.assertIn(name, tried, f"{name} is not in the loop; update NAMES in this suite")
        answer = self.resolve()  # no stubs at all: every name fails
        self.assertNotEqual(answer.returncode, 0)
        listed = re.search(r"tried \(newest first\): ([^&\n]+)", answer.stderr)
        self.assertIsNotNone(listed, f"the refusal does not list what it tried:\n{answer.stderr}")
        # Compared as tokens, not substrings: "python3.14" contains "python", so a
        # substring test would pass over the omission it is here to catch.
        printed = listed.group(1).split()
        self.assertEqual(
            printed,
            tried,
            "the refusal does not report the search it just made, in the order it made it",
        )

    def test_the_refusal_prints_on_stderr_and_names_the_requirements(self) -> None:
        answer = self.resolve()
        self.assertNotEqual(answer.returncode, 0)
        self.assertEqual(answer.stdout, "")
        self.assertIn("3.10", answer.stderr)
        self.assertIn("numpy, ml_dtypes, safetensors", answer.stderr)


class TheLibraryContractItself(ResolverCase):
    def test_the_floor_and_the_dependency_string_are_the_documented_ones(self) -> None:
        env = {
            "PATH": f"{self.bin}{os.pathsep}/usr/bin:/bin",
            "HOME": os.environ.get("HOME", "/"),
        }
        answer = subprocess.run(
            [
                "/bin/bash",
                "--norc",
                "--noprofile",
                "-c",
                'source "$1"; '
                'echo "$TINYTITAN_PYTHON_MIN_MAJOR.$TINYTITAN_PYTHON_MIN_MINOR"; '
                'echo "$TINYTITAN_PYTHON_DEPS"; '
                "TINYTITAN_PYTHON=/bin/false tinytitan_python_note",
                "_",
                str(LIBRARY),
            ],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        self.assertEqual(answer.returncode, 0, answer.stderr)
        floor, deps, note = answer.stdout.splitlines()
        self.assertEqual((floor, deps), ("3.10", "import numpy, ml_dtypes, safetensors"))
        self.assertIn("/bin/false", note)
        self.assertIn("pip install", note)

    def test_the_library_is_sourced_not_executed(self) -> None:
        self.assertIn("Sourced, never executed", LIBRARY.read_text())


if __name__ == "__main__":
    unittest.main()
