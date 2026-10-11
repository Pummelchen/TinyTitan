"""The memory switch's readers, and the one definition they all have to call.

Measured, before the AUD-301 fix, for the two readers that then existed:

| reader | source | words it reads as ON |
|---|---|---|
| engine | MemoryConfiguration.swift:236 | `1`, `on`, `true`, case-insensitively |
| launcher library | tools/tinytitan_models.sh:509 | the literal `1` and nothing else |

A third reader appeared in the same seam (AUD-302): the launcher's own reporting,
which looked only at the `--memory` flag's shell variable.

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

AUD-302 is the same switch with a third reader: `tools/server_launcher.sh`
reported memory in the plan and the banner from the `--memory` flag's shell
variable alone, while `docs/agent-memory.md` tells the operator to use the
environment spelling, which since this fix does turn memory on. The two report
sites now call `tinytitan_memory_requested`, so the ON-set has one definition
besides the engine's, and the suites here drive both spellings through the real
launcher (`--dry-run`, which starts nothing) and through the two reporting blocks
lifted verbatim from it.

The mutation sweep killed eleven of eleven shapes, but only after a survivor
corrected the suite: dropping the `export TINYTITAN_MEMORY="${...:-0}"` line
passed everything, because the engine reads an absent name as off too. The suite
was not looking at the child's environment at all, so the two pass-through tests
added for it are what make that line mean something. One more shape -- the
named-workspace escape hatch -- had no test either, and the sweep found it. A
third, from the AUD-302 sweep, reads the switch without `:-0`: it still answers
OFF for an unset value, so only watching stderr catches it, because the launcher
runs under `set -u` and would print `unbound variable` on every ordinary launch.

    cd benchmark && python3 -m unittest test_models_memory_gate -v
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import tempfile
import unittest

import launcher_fixture
from test_launcher_port import run_launcher

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
LIBRARY = REPO_ROOT / "tools" / "tinytitan_models.sh"
LAUNCHER = REPO_ROOT / "tools" / "server_launcher.sh"

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


ON_SET_SCRIPT = """
set -uo pipefail
source "$1"
if tinytitan_memory_requested; then echo ON; else echo OFF; fi
"""


class TheOneReadingOfTheSwitch(unittest.TestCase):
    """AUD-302: a third reader appeared, so the ON-set needs one definition.

    The engine reads `TINYTITAN_MEMORY` (1/on/true, case-insensitive) and, since
    AUD-301, so does `tinytitan_export_memory_environment`. The launcher's own
    reporting reads only the `--memory` flag's shell variable, so
    `TINYTITAN_MEMORY=1 tools/server_launcher.sh` -- the spelling
    `docs/agent-memory.md` documents under "Setup" -- starts a server with
    memory on, with the repo-scoped store exported, and says nothing about it in
    the plan or the banner. This pins the predicate every reader is meant to
    call, in the shape the engine reads.
    """

    def asked(self, memory_value: str | None) -> str:
        environment = dict(os.environ)
        environment.pop("TINYTITAN_MEMORY", None)
        if memory_value is not None:
            environment["TINYTITAN_MEMORY"] = memory_value
        result = subprocess.run(
            ["/bin/bash", "-c", ON_SET_SCRIPT, "bash", str(LIBRARY)],
            text=True,
            capture_output=True,
            check=False,
            env=environment,
        )
        # The launcher runs under `set -u`, so a predicate that reads the switch
        # without a default does not merely mis-report an unset value: it prints
        # `TINYTITAN_MEMORY: unbound variable` on every ordinary launch. The
        # mutation sweep showed nothing watched stderr, so this is what pins it.
        self.assertEqual(result.stderr, "")
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.strip()

    def test_the_engine_s_on_words_are_all_requested(self) -> None:
        for value in ("1", "on", "true", "ON", "True", "TRUE"):
            with self.subTest(value=value):
                self.assertEqual(self.asked(value), "ON")

    def test_everything_the_engine_reads_as_off_is_not_requested(self) -> None:
        # "" is unset-as-read; "  on" and "yes" are off to the engine too, so
        # this must not ask for more than the engine does.
        for value in ("0", "off", "", "yes", "2", "  on", "1 on", None):
            with self.subTest(value=value):
                self.assertEqual(self.asked(value), "OFF")


# The launcher's two reporting lines, lifted verbatim so they can run with no
# model: the flag-to-environment bridge, and the export call followed by the
# banner that reads it.
BRIDGE_START = 'if [[ "$MEMORY" == "1" ]]; then\n  export TINYTITAN_MEMORY=1'
BRIDGE_END = "\nfi\n"
BANNER_START = 'tinytitan_export_memory_environment "$PWD"'
BANNER_END = "# 10) Start the server"

# The launcher's argument parser sets MEMORY; this hands the extracted bridge the
# same variable from the environment so a test can drive either spelling. The
# placeholders are replaced rather than formatted: the slice is full of shell
# braces, and str.format would read every one of them as a field.
LAUNCHER_SLICE_SCRIPT = """
set -uo pipefail
source "@LIBRARY@"
MEMORY="${TEST_MEMORY:-0}"
@BRIDGE@
@BANNER@
"""


def launcher_report_script() -> str:
    """The launcher's bridge and banner, verbatim, sourced with the real library."""
    text = LAUNCHER.read_text(encoding="utf-8")
    try:
        start = text.index(BRIDGE_START)
        bridge = text[start : text.index(BRIDGE_END, start) + len(BRIDGE_END)]
        start = text.index(BANNER_START)
        banner = text[start : text.index(BANNER_END, start)]
    except ValueError as error:  # pragma: no cover - a marker moved
        raise AssertionError(f"launcher reporting line moved: {error}") from error
    return (
        LAUNCHER_SLICE_SCRIPT.replace("@LIBRARY@", str(LIBRARY))
        .replace("@BRIDGE@", bridge)
        .replace("@BANNER@", banner)
    )


class TheLauncherReportsEverySpelling(unittest.TestCase):
    """What the operator is told about memory must match what runs."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.installs = launcher_fixture.SyntheticInstalls().create()
        cls.addClassCleanup(cls.installs.destroy)
        cls.model = cls.installs.first_gpu()

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.project = pathlib.Path(self._tmp.name).resolve() / "work" / "my-project"
        self.project.mkdir(parents=True)

    def memory_lines(self, stdout: str) -> list[str]:
        return [line for line in stdout.splitlines() if "Memory:" in line]

    def plan(self, args: tuple[str, ...], memory_value: str | None) -> list[str]:
        """The real launcher's plan, from a dry run that starts nothing."""
        environment = {"TINYTITAN_MEMORY": memory_value or ""}
        run = run_launcher(
            self.installs,
            "--client",
            "server",
            "--model",
            self.model,
            *args,
            env=environment,
        )
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        return self.memory_lines(run.stdout)

    def banner(
        self, test_memory: str, memory_value: str | None, *, cache_mib: str | None = None
    ) -> list[str]:
        """The launcher's bridge and banner, verbatim, against the real library."""
        environment = dict(os.environ)
        environment["HOME"] = str(self.project.parent)
        environment["TEST_MEMORY"] = test_memory
        for name in ("TINYTITAN_MEMORY", "TINYTITAN_MEMORY_CACHE_MIB"):
            environment.pop(name, None)
        if memory_value is not None:
            environment["TINYTITAN_MEMORY"] = memory_value
        if cache_mib is not None:
            environment["TINYTITAN_MEMORY_CACHE_MIB"] = cache_mib
        result = subprocess.run(
            ["/bin/bash", "-c", launcher_report_script()],
            text=True,
            capture_output=True,
            check=False,
            env=environment,
            cwd=str(self.project),
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return self.memory_lines(result.stdout)

    def test_the_flag_is_reported_in_the_plan(self) -> None:
        self.assertEqual(len(self.plan(("--memory",), None)), 1)

    def test_the_environment_is_reported_in_the_plan(self) -> None:
        # The spelling docs/agent-memory.md:322 tells the operator to use.
        for value in ("1", "on", "TRUE"):
            with self.subTest(value=value):
                self.assertEqual(len(self.plan((), value)), 1)

    def test_a_switch_that_is_off_is_not_reported(self) -> None:
        # Reporting memory that is off would be the same bug wearing the other
        # half of the coat: the line has to follow the switch, not lead it.
        for value in ("0", "off", ""):
            with self.subTest(value=value):
                self.assertEqual(self.plan((), value), [])

    def test_the_banner_follows_the_flag(self) -> None:
        lines = self.banner("1", None)
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("Memory: on", lines[0])

    def test_the_banner_follows_the_environment(self) -> None:
        # The property is that the report cannot depend on the spelling: the line
        # the flag produces is the line the environment has to produce. (The line
        # names the workspace with `basename "$PWD"`, so this compares whole
        # lines rather than reaching for a path it never prints.)
        expected = self.banner("1", None)
        for value in ("1", "on", "true"):
            with self.subTest(value=value):
                self.assertEqual(self.banner("0", value), expected)

    def test_the_banner_names_the_store_the_server_gets(self) -> None:
        # The line's whole purpose: the operator sees which directory holds the
        # facts, and which workspace names it.
        lines = self.banner("0", "on", cache_mib="64")
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("/memory", lines[0])
        self.assertIn("cap 64 MiB", lines[0])
        self.assertIn("workspace my-project", lines[0])

    def test_an_undocumented_word_is_not_reported(self) -> None:
        self.assertEqual(self.banner("0", "yes"), [])


if __name__ == "__main__":
    unittest.main()
