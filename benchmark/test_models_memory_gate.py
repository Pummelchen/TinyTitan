"""AUD-301: the memory switch has two readers, and one of them reads only `1`.

Measured, before the fix:

| reader | source | words it reads as ON |
|---|---|---|
| engine | MemoryConfiguration.swift:236 | `1`, `on`, `true`, case-insensitively |
| launcher library | tools/tinytitan_models.sh:509 | the literal `1` and nothing else |

The engine's alias set is not an accident: `on` is pinned by a test
(tests/TinyTitanMemory/MemoryConfigurationTests.swift:61), and the sibling
`TINYTITAN_MEMORY_TOOLS` switch carries the same aliases with a comment saying
they predate the surface. So the library is the wrong side, and this suite pins
the library to the engine rather than the other way round.

What the divergence costs: `tinytitan_export_memory_environment` is the only
thing that exports `TINYTITAN_MEMORY_DIR`, `TINYTITAN_MEMORY_NAMESPACE` and
`TINYTITAN_WORKSPACE_DIR`. Returning early leaves all three unset, so the engine
falls through to its own defaults -- `~/.tinytitan/memory` instead of the
`<TinyTitan>/memory` beside `models/` that the docs promise, and workspace
`default` (MemoryConfiguration.swift:184), one file for every project. That is
the exact outcome the junk-drawer refusal in the same function exists to
prevent, and because the refusal keys off `TINYTITAN_WORKSPACE_DIR`, unsetting
it also means neither side refuses: the store silently mixes.

The tests source the real library and call the real function in a `/bin/bash`
subprocess -- 3.2.57 on a factory Mac -- with HOME and the launch directory in a
scratch tree. Nothing here loads a model.

The mutation sweep killed eleven of eleven shapes, but only after a survivor
corrected the suite: dropping the `export TINYTITAN_MEMORY="${...:-0}"` line
passed everything, because the engine reads an absent name as off too. The suite
was not looking at the child's environment at all, so the two pass-through tests
added for it are what make that line mean something. One more shape -- the
named-workspace escape hatch -- had no test either, and the sweep found it.

    cd benchmark && python3 -m unittest test_models_memory_gate -v
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import tempfile
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
LIBRARY = REPO_ROOT / "tools" / "tinytitan_models.sh"

# Sourced, called, then the three exports it owns are printed. The function can
# `exit 2` from the junk-drawer refusal, which ends the child before the prints:
# that is why the exit status is checked separately from the body.
SCRIPT = """
set -uo pipefail
source "$1"
tinytitan_export_memory_environment "$2"
printf 'DIR=%s\\n'   "${TINYTITAN_MEMORY_DIR-UNSET}"
printf 'NS=%s\\n'    "${TINYTITAN_MEMORY_NAMESPACE-UNSET}"
printf 'WS=%s\\n'    "${TINYTITAN_WORKSPACE_DIR-UNSET}"
printf 'MEM=%s\\n'   "${TINYTITAN_MEMORY-UNSET}"
"""


class MemorySeam(unittest.TestCase):
    """Drive one value of TINYTITAN_MEMORY through the real library."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = pathlib.Path(self._tmp.name).resolve()
        self.home = root / "home" / "andre"
        self.project = self.home / "work" / "my-project"
        self.home.mkdir(parents=True)
        self.project.mkdir(parents=True)

    def run_library(
        self, launch_dir: pathlib.Path, memory_value: str | None
    ) -> subprocess.CompletedProcess[str]:
        environment = dict(os.environ)
        environment["HOME"] = str(self.home)
        if memory_value is None:
            environment.pop("TINYTITAN_MEMORY", None)
        else:
            environment["TINYTITAN_MEMORY"] = memory_value
        return subprocess.run(
            ["/bin/bash", "-c", SCRIPT, "bash", str(LIBRARY), str(launch_dir)],
            text=True,
            capture_output=True,
            check=False,
            env=environment,
            cwd=str(launch_dir),
        )

    def exports(self, memory_value: str | None) -> dict[str, str]:
        result = self.run_library(self.project, memory_value)
        self.assertEqual(result.returncode, 0, result.stderr)
        return {
            line.split("=", 1)[0]: line.split("=", 1)[1]
            for line in result.stdout.splitlines()
            if "=" in line
        }

    def assert_setup_matches_the_flag(self, memory_value: str) -> None:
        """ON is ON: the store, the namespace and the workspace all land."""
        exported = self.exports(memory_value)
        self.assertTrue(
            exported["DIR"].endswith("/memory"),
            f"{memory_value!r} left the store at {exported['DIR']!r}",
        )
        self.assertEqual(exported["NS"], "tinytitan")
        self.assertEqual(exported["WS"], str(self.project))

    def test_one_sets_up_the_store(self) -> None:
        self.assert_setup_matches_the_flag("1")

    def test_on_sets_up_the_store(self) -> None:
        self.assert_setup_matches_the_flag("on")

    def test_true_sets_up_the_store(self) -> None:
        self.assert_setup_matches_the_flag("true")

    def test_uppercase_on_sets_up_the_store(self) -> None:
        # The engine lowercases before comparing, so ON means the same to it.
        self.assert_setup_matches_the_flag("ON")

    def test_zero_leaves_the_store_to_the_engine(self) -> None:
        for value in ("0", "off", "", None):
            with self.subTest(value=value):
                self.assertEqual(self.exports(value)["DIR"], "UNSET")

    def test_a_word_the_engine_reads_as_off_stays_off(self) -> None:
        # `yes` is not in either reader's ON set. The fix must not invent a
        # refusal for it: the engine runs with memory off, and so does this.
        self.assertEqual(self.exports("yes")["DIR"], "UNSET")

    def test_an_unset_switch_is_exported_as_an_explicit_zero(self) -> None:
        # The library's first line is `export TINYTITAN_MEMORY="${...:-0}"`, and
        # the mutation sweep showed nothing observed it: the engine reads an
        # absent name as off, so dropping the normalisation changes no verdict.
        # It still matters, because the environment the server inherits is what
        # its own diagnostics and any child of it report.
        self.assertEqual(self.exports(None)["MEM"], "0")

    def test_the_word_is_passed_through_unchanged(self) -> None:
        for value in ("1", "on", "ON", "0", "off", "yes"):
            with self.subTest(value=value):
                self.assertEqual(self.exports(value)["MEM"], value)

    def test_an_operator_can_rename_the_store(self) -> None:
        # The exports are :-defaults, so a directory already chosen stays put.
        # The workspace assertion is what makes this mean anything: before the
        # fix this test passed over a function that returned early and left the
        # inherited variable standing, which is not the same as respecting it.
        environment = dict(os.environ)
        environment.update(
            HOME=str(self.home),
            TINYTITAN_MEMORY="on",
            TINYTITAN_MEMORY_DIR=str(self.home / "elsewhere"),
        )
        result = subprocess.run(
            ["/bin/bash", "-c", SCRIPT, "bash", str(LIBRARY), str(self.project)],
            text=True,
            capture_output=True,
            check=False,
            env=environment,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"DIR={self.home / 'elsewhere'}", result.stdout)
        self.assertIn(f"WS={self.project}", result.stdout)


class TheJunkDrawerRefusalReachesEverySpelling(unittest.TestCase):
    """Launching from $HOME mixes every project into one store; it refuses."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.home = pathlib.Path(self._tmp.name).resolve() / "home" / "andre"
        self.home.mkdir(parents=True)

    def refuse(self, memory_value: str) -> subprocess.CompletedProcess[str]:
        environment = dict(os.environ)
        environment.update(HOME=str(self.home), TINYTITAN_MEMORY=memory_value)
        return subprocess.run(
            ["/bin/bash", "-c", SCRIPT, "bash", str(LIBRARY), str(self.home)],
            text=True,
            capture_output=True,
            check=False,
            env=environment,
        )

    def test_one_refuses(self) -> None:
        result = self.refuse("1")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("not a project directory", result.stderr)

    def test_on_refuses(self) -> None:
        # Before the fix this ran the launch: memory on, store in
        # ~/.tinytitan/memory, workspace "default", and no refusal from either
        # side -- because the refusal keys off the export this skipped.
        result = self.refuse("on")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("not a project directory", result.stderr)
        self.assertIn("TINYTITAN_MEMORY", result.stderr)

    def test_a_named_workspace_is_not_refused(self) -> None:
        # The refusal's own advice: name the workspace and launch from anywhere.
        # Passing (this arm predates the fix) and here because the sweep showed
        # the `if [[ -z ... ]]` guard around it had no test at all.
        environment = dict(os.environ)
        environment.update(
            HOME=str(self.home),
            TINYTITAN_MEMORY="on",
            TINYTITAN_MEMORY_WORKSPACE="novel",
        )
        result = subprocess.run(
            ["/bin/bash", "-c", SCRIPT, "bash", str(LIBRARY), str(self.home)],
            text=True,
            capture_output=True,
            check=False,
            env=environment,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("DIR=UNSET", result.stdout)
        self.assertTrue(result.stdout.startswith("DIR="), result.stdout)
        self.assertTrue(result.stdout.splitlines()[0].endswith("/memory"))

    def test_zero_does_not_refuse(self) -> None:
        # Off is off: a reason that begins "TINYTITAN_MEMORY=1 but" must never
        # stop a launch that never asked for memory.
        result = self.refuse("0")
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
