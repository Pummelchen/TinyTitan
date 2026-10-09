#!/usr/bin/env python3
"""Tests for the memory_* drivers' report commands, model-free.

Nothing here starts a server, loads a model or sends a request: every report path
reads result JSON out of a temp directory and prints. The child that proves the
imports touch no network patches `urlopen` before importing, and the suites set
`TINYTITAN_MEMVAL_RESULTS` to a scratch directory before the first import so no
test reads the operator's real `.build/benchmark-logs/`.

What the suite is about, in the drivers' own words and their defects (measured
over an empty results directory, `/tmp/aud240-empty-reports.log`):

1. All seven report commands printed their table headers over zero runs and
   exited 0. `memory_projects` printed `Leaks, one line each:` followed by
   `none` -- the isolation verdict the file exists for, passed by measuring
   nothing. `memory_master.aggregate()` printed `carryable 0/0 n/a` rows for both
   arms. `memory_mini` printed its `the 35B spends 45-55 s` cost claim and
   returned 0. `memory_correct` was the one that said anything
   (:501 `No results in ...`), and it still exited 0.
2. Six of the summaries skip an arm that wrote nothing with `if not rows:
   continue` (or `if not row` / `if not per_run`), so a report over one arm of
   four is indistinguishable from a report over four.
3. `memory_master.score_run([])` reads `results[0]` (:220), so one run file that
   holds an empty list turns the whole report into an IndexError traceback and
   loses every other arm.
4. No guard propagated a report's status: `report()` returned None and the
   dispatch called it as a statement, so the exit said nothing about what was
   measured. (`memory_mini.py:473` was already `raise SystemExit(report())`,
   which is the shape the rest are corrected to.)

The verdict strings are `NOT MEASURED` for a report that has nothing to say and
the artifact directory named in the same line, because a reader who is told only
`no results` cannot tell which directory they are looking at.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]

MODULES = (
    "memory_book",
    "memory_correct",
    "memory_master",
    "memory_projects",
    "memory_value",
    "memory_volume",
)
# The results directory each module reads, by the name it binds at import.
DIR_ATTR = {name: "OUT" for name in MODULES}
DIR_ATTR["memory_mini"] = "RESULTS"


@contextlib.contextmanager
def captured():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


class ReportHarness(unittest.TestCase):
    """Import the drivers once against a scratch root, then point each at its own dir."""

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.TemporaryDirectory()
        env = {
            "TINYTITAN_MEMVAL_RESULTS": cls.root.name,
            "TINYTITAN_MEMVAL_MEMDIR": cls.root.name,
            "TINYTITAN_MEMVAL_RUN": "1",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            cls.mods = {name: importlib.import_module(name) for name in MODULES}
            cls.mods["memory_mini"] = importlib.import_module("memory_mini")

    @classmethod
    def tearDownClass(cls):
        cls.root.cleanup()

    def scratch(self, name):
        """A fresh, named results directory bound to the module under test."""
        path = pathlib.Path(self.root.name) / name / self.id().split(".")[-1]
        if path.exists():
            for entry in path.iterdir():
                entry.unlink()
        path.mkdir(parents=True, exist_ok=True)
        setattr(self.mods[name], DIR_ATTR[name], path)
        return path

    def call(self, name, function="report", *args):
        with captured() as buf:
            try:
                status = getattr(self.mods[name], function)(*args)
            except Exception as error:  # a report that raises is exactly what is being pinned
                status = f"RAISED {type(error).__name__}: {error}"
        return status, buf.getvalue()

    def arm_file(self, name, arm, results):
        mod = self.mods[name]
        run = os.environ.get("TINYTITAN_MEMVAL_RUN", "1")
        stem = arm if name != "memory_master" else f"{mod.NAME}-{arm}"
        path = getattr(mod, DIR_ATTR[name]) / f"{stem}-r{run}.json"
        path.write_text(json.dumps(results))
        return path


class TestEmptyReportRefuses(ReportHarness):
    def test_a_report_with_no_results_returns_a_status(self):
        for name in MODULES + ("memory_mini",):
            with self.subTest(module=name):
                self.scratch(name)
                status, printed = self.call(name)
                self.assertEqual(
                    status,
                    1,
                    f"{name}.report() exited {status!r} over a directory with nothing in it",
                )

    def test_the_refusal_says_nothing_was_measured(self):
        for name in MODULES + ("memory_mini",):
            with self.subTest(module=name):
                self.scratch(name)
                _status, printed = self.call(name)
                self.assertIn("NOT MEASURED", printed)

    def test_the_refusal_names_the_directory_it_looked_in(self):
        for name in MODULES + ("memory_mini",):
            with self.subTest(module=name):
                path = self.scratch(name)
                _status, printed = self.call(name)
                self.assertIn(
                    str(path),
                    printed,
                    f"{name} refused without saying which directory it read",
                )

    def test_the_refusal_names_the_arms_it_expected(self):
        for name in MODULES:
            with self.subTest(module=name):
                self.scratch(name)
                _status, printed = self.call(name)
                for arm in self.mods[name].ARMS:
                    self.assertIn(
                        arm,
                        printed,
                        f"{name} refused without naming {arm}, one of the arms it can read",
                    )


class TestTheIsolationVerdict(ReportHarness):
    def test_the_leak_line_does_not_certify_a_run_that_did_not_happen(self):
        self.scratch("memory_projects")
        _status, printed = self.call("memory_projects")
        lines = [line.strip() for line in printed.splitlines()]
        self.assertNotIn(
            "none",
            lines,
            "memory_projects printed its 'no project leaked' verdict over zero results",
        )

    def test_the_leak_line_reports_the_count_it_was_computed_from(self):
        self.scratch("memory_projects")
        self.arm_file("memory_projects", "control", [])
        _status, printed = self.call("memory_projects")
        self.assertIn("0 result", printed)


class TestAnArmIsNeverSkippedSilently(ReportHarness):
    def test_every_arm_appears_when_only_one_of_them_ran(self):
        for name in MODULES:
            with self.subTest(module=name):
                mod = self.mods[name]
                self.scratch(name)
                self.arm_file(name, mod.ARMS[0], [])
                _status, printed = self.call(name)
                for arm in mod.ARMS[1:]:
                    self.assertIn(
                        arm,
                        printed,
                        f"{name} omitted {arm} from a report that shows {mod.ARMS[0]}",
                    )


class TestAMalformedRunDoesNotEndTheReport(ReportHarness):
    def test_master_reports_the_other_arm_when_one_run_is_empty(self):
        self.scratch("memory_master")
        self.arm_file("memory_master", "summary", [])
        self.arm_file("memory_master", "auto", [])
        status, printed = self.call("memory_master")
        self.assertNotIsInstance(
            status,
            str,
            f"memory_master.report() raised over a run file holding an empty list: {status}",
        )


class TestTheMasterSweepCommandsRefuse(ReportHarness):
    """`report-all` and `stats` read every scenario, so an empty tree is theirs too."""

    def empty_root(self):
        path = pathlib.Path(self.root.name) / "no-scenarios" / self.id().split(".")[-1]
        path.mkdir(parents=True, exist_ok=True)
        return path

    def test_stats_refuses_a_tree_with_no_runs(self):
        root = self.empty_root()
        status, printed = self.call("memory_master", "aggregate", root)
        self.assertEqual(
            status,
            1,
            f"aggregate() exited {status!r} over a root holding no run of any scenario",
        )
        self.assertIn("NOT MEASURED", printed)

    def test_stats_does_not_print_a_pooled_row_it_did_not_compute(self):
        root = self.empty_root()
        _status, printed = self.call("memory_master", "aggregate", root)
        self.assertNotIn(
            "0/0",
            printed,
            "aggregate() printed its pooled carryable/foundation rows over zero worlds",
        )

    def test_report_all_refuses_a_tree_with_no_runs(self):
        root = self.empty_root()
        status, printed = self.call("memory_master", "report_all", root)
        self.assertEqual(
            status,
            1,
            f"report_all() exited {status!r} over a root holding no run of any scenario",
        )
        self.assertIn("NOT MEASURED", printed)


class TestTheAggregateTableScoresEveryScenario(ReportHarness):
    """`report-all` reads all ten scenarios, so it may not read one scenario's spec.

    `score_run()` takes no spec: it uses the module globals `NAME`/`SPEC`, which are
    bound once at import from `TINYTITAN_MASTER_SCENARIO` -- and that is exactly how
    `memval_master.sh` invokes the aggregate (`TINYTITAN_MASTER_SCENARIO=photograph
    ... report-all`). `aggregate()` saves and swaps those globals per scenario;
    `report_all()` did not, so every row but the named one was scored against the
    wrong key set. Measured over one tree holding a full-mark record for each arm of
    `photograph` and `ledger` (`/tmp/aud247-red.log`): with photograph named the table
    read `photograph 45/45 81/81` and `ledger 0/35 0/63`; with ledger named the same
    tree read `photograph 0/27 0/27` and `ledger 21/21 21/21` -- exit 0 either way.

    The zeroes are the finding, and they are not honest zeroes. `score_run` counts a
    denominator per key the *named* spec expects, so ledger's row divides by
    photograph's key counts (5 carryable and 9 foundation x ledger's 7 scored
    sessions = 35 and 63) and answers every one of them "missed" because the stored
    keys are ledger's names. A row of that shape reads as a scenario that ran and
    scored nothing, which is the opposite of the `0/0` that would say "not measured".
    A day-long sweep therefore printed a table of plausible failures for nine
    scenarios and one real result, for whichever name the shell happened to set.

    The expected denominators are arithmetic, not a golden: a scenario with `k` keys,
    `c` of them carryable and `n` sessions scores `c * (n - 1)` carryable and
    `(k - c) * (n - 1)` foundation over one record, because `score_run` only counts
    sessions after the first.
    """

    def full_mark_records(self, name, root):
        """One record per arm under `root`, every key answered with that scenario's own truth.

        `root` is passed in rather than drawn from `scratch()` because a tree holds
        every scenario's directory: `scratch()` clears a reused directory by
        unlinking its entries, which cannot remove the `memory-<scenario>-*`
        directory that belongs inside the tree `report_all` globs.
        """
        spec = importlib.import_module("master_scenarios").SCENARIOS[name]
        rows = [
            {
                "session": session,
                "prompt_tokens": 1000,
                "completion_tokens": 50,
                "seconds": 1.0,
                "summary_seconds": 0.5,
                "consolidation_wait": 0.2,
                "finish_reason": "stop",
                "self_truth": None,
                "answers": spec["truth"](session),
            }
            for session in range(1, spec["sessions"] + 1)
        ]
        for arm in ("summary", "auto"):
            directory = root / f"memory-{name}-measure"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"{name}-{arm}-r1.json").write_text(json.dumps(rows))
        return root

    def denominators(self, printed, name):
        row = [line for line in printed.splitlines() if line.split()[:2] == [name, "summary"]]
        self.assertEqual(len(row), 1, printed)
        # Columns are scenario, arm, runs, carryable, stale, foundation, invalid, cost.
        parts = row[0].split()
        return [parts[3], parts[5]]

    def name_the_scenario(self, name):
        mod = self.mods["memory_master"]
        spec = importlib.import_module("master_scenarios").SCENARIOS[name]
        saved = (mod.NAME, mod.SPEC)
        self.addCleanup(setattr, mod, "NAME", saved[0])
        self.addCleanup(setattr, mod, "SPEC", saved[1])
        mod.NAME, mod.SPEC = name, spec

    def tree(self):
        """One results tree holding a full-mark run of both scenarios, both arms."""
        root = self.scratch("memory_master")
        self.full_mark_records("photograph", root)
        self.full_mark_records("ledger", root)
        return root

    def test_every_row_is_scored_against_its_own_scenario(self):
        root = self.tree()
        self.name_the_scenario("photograph")
        status, printed = self.call("memory_master", "report_all", root)
        self.assertEqual(status, 0, printed)
        # photograph, 14 keys / 5 carryable / 10 sessions -> 5*9 and 9*9
        self.assertEqual(self.denominators(printed, "photograph"), ["45/45", "81/81"], printed)
        # ledger, 6 keys / 3 carryable / 8 sessions -> 3*7 and 3*7
        self.assertEqual(
            self.denominators(printed, "ledger"),
            ["21/21", "21/21"],
            "ledger's row was scored against photograph's key set, so it printed "
            "photograph's denominators and called every ledger key missed",
        )

    def test_the_named_scenario_survives_the_table(self):
        """The fix mutates module state, so the restore is part of it.

        `report_all()` now writes `NAME`/`SPEC` for every row it prints. A caller that
        prints the table and then does something scenario-scoped -- which is what
        `aggregate()` in the same module does -- would otherwise inherit the last
        scenario alphabetically rather than the one it was told.
        """
        root = self.tree()
        mod = self.mods["memory_master"]
        self.name_the_scenario("photograph")
        status, printed = self.call("memory_master", "report_all", root)
        self.assertEqual(status, 0, printed)
        self.assertEqual(mod.NAME, "photograph")
        self.assertIs(mod.SPEC, importlib.import_module("master_scenarios").SCENARIOS["photograph"])

    def test_the_table_does_not_depend_on_which_scenario_is_named(self):
        root = self.tree()
        self.name_the_scenario("photograph")
        _status, named_photograph = self.call("memory_master", "report_all", root)
        self.name_the_scenario("ledger")
        _status, named_ledger = self.call("memory_master", "report_all", root)
        self.assertEqual(
            [line for line in named_photograph.splitlines() if line.strip()],
            [line for line in named_ledger.splitlines() if line.strip()],
            "the aggregate over one tree changed because the environment named a "
            "different scenario -- the rows describe the run, not the shell",
        )


class TestEveryReportCommandPropagates(ReportHarness):
    def test_the_guard_raises_with_the_report_status(self):
        for name in MODULES + ("memory_mini",):
            with self.subTest(module=name):
                source = pathlib.Path(f"{name}.py").read_text(encoding="utf-8")
                guard = source.split('if __name__ == "__main__":', 1)[1]
                self.assertIn("raise SystemExit(report())", guard)

    def test_master_propagates_its_report_commands(self):
        source = pathlib.Path("memory_master.py").read_text(encoding="utf-8")
        guard = source.split('if __name__ == "__main__":', 1)[1]
        self.assertIn("raise SystemExit(report_all(OUT.parent))", guard)

    def test_the_report_functions_are_declared_to_return_a_status(self):
        for name in MODULES + ("memory_mini",):
            with self.subTest(module=name):
                source = pathlib.Path(f"{name}.py").read_text(encoding="utf-8")
                body = source.split("def report(", 1)[1].split("\ndef ", 1)[0]
                self.assertRegex(
                    body,
                    r"return \d|return status|return report_status",
                    f"{name}.report() has no status to propagate",
                )


class TestABlankResultsTreeRefuses(unittest.TestCase):
    """AUD-243: `TINYTITAN_MEMVAL_RESULTS=` is not "unset", and it names no directory.

    `Path(os.environ.get(ENV, default))` accepts the empty string, so a driver
    sets its results tree to `Path('')`, which is the directory the command
    happens to be run in. Measured from an unrelated cwd:

        cd /tmp && TINYTITAN_MEMVAL_RESULTS= python3 .../memory_book.py report
        NOT MEASURED: no results in .

    The reader is shown a directory that identifies nothing, and a writer given
    the same shell slip drops its run records into the cwd of wherever it was
    launched. A value that was typed but came out empty is a mistake, so it is
    refused by name; unset still means the driver's own default tree.
    """

    ENV = "TINYTITAN_MEMVAL_RESULTS"

    def load(self, name, env):
        """A fresh import under `env`, so the driver's binding line runs.

        Returns the outcome, the bound module or the refusal, and the
        environment the import left behind — read inside the patch, because
        `memory_small_model` publishes its tree there rather than returning it.
        """
        with mock.patch.dict(os.environ, env, clear=True):
            spec = importlib.util.spec_from_file_location(
                f"aud243_{name}", ROOT / "benchmark" / f"{name}.py"
            )
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except SystemExit as error:
                return "REFUSED", error, dict(os.environ)
            except OSError as error:  # a spec that does not resolve is a broken test, not a finding
                self.fail(f"could not import {name}: {error}")
            return "BOUND", module, dict(os.environ)

    def bound_tree(self, name, env):
        outcome, value, _ = self.load(name, env)
        self.assertEqual(outcome, "BOUND", f"{name} did not bind: {value}")
        return getattr(value, DIR_ATTR[name])

    def test_a_blank_tree_is_refused_instead_of_becoming_the_cwd(self):
        for name in MODULES + ("memory_mini", "memory_small_model"):
            with self.subTest(module=name):
                outcome, error, _ = self.load(name, {self.ENV: ""})
                self.assertEqual(
                    outcome, "REFUSED", f"{name} bound a blank tree to the current directory"
                )
                self.assertIn(self.ENV, str(error))

    def test_a_whitespace_only_tree_is_refused_too(self):
        for name in MODULES + ("memory_mini",):
            with self.subTest(module=name):
                outcome, error, _ = self.load(name, {self.ENV: "   "})
                self.assertEqual(outcome, "REFUSED")
                self.assertIn(self.ENV, str(error))

    def test_an_unset_tree_still_binds_the_drivers_own_default(self):
        tree = self.bound_tree("memory_book", {})
        self.assertEqual(tree, ROOT / ".build/benchmark-logs/memory-book")

    def test_a_named_tree_still_binds_as_it_did(self):
        named = str(ROOT / ".build/benchmark-logs/two")
        self.assertEqual(self.bound_tree("memory_volume", {self.ENV: named}), pathlib.Path(named))

    def test_small_model_names_its_own_default_when_unset(self):
        """It publishes the tree for the drivers it imports, so it must set one."""
        outcome, _, env = self.load("memory_small_model", {})
        self.assertEqual(outcome, "BOUND")
        self.assertIn("memory-small", env[self.ENV])


class TestABlankRunTokenRefuses(unittest.TestCase):
    """AUD-245: `TINYTITAN_MEMVAL_RUN=` is how a day of repeats becomes one run.

    Result files are named `{arm}-r{RUN}.json` so repeats can be averaged --
    `memory_book.py`'s own comment says results are kept per run for exactly that
    -- and every report walks the tree with the glob `*-r*.json`. A blank token
    keeps the glob matching but drops the distinguishing part, so the second
    repeat overwrites the first into the same file, and the report's run count is
    the number of files it found. Three measured repeats therefore print as one
    run, and the reader has no way to see which it was.
    """

    ENV = "TINYTITAN_MEMVAL_RUN"

    def load(self, name, env):
        with mock.patch.dict(os.environ, env, clear=True):
            spec = importlib.util.spec_from_file_location(
                f"aud245_{name}", ROOT / "benchmark" / f"{name}.py"
            )
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except SystemExit as error:
                return "REFUSED", error
            except OSError as error:
                self.fail(f"could not import {name}: {error}")
            return "BOUND", module

    def bound(self, name, env):
        outcome, value = self.load(name, env)
        self.assertEqual(outcome, "BOUND", f"{name} did not bind: {value}")
        return value.RUN

    def test_a_blank_run_token_is_refused_by_name(self):
        for name in MODULES:
            with self.subTest(module=name):
                outcome, error = self.load(name, {self.ENV: ""})
                self.assertEqual(
                    outcome, "REFUSED", f"{name} bound a blank run token, so repeats share one file"
                )
                self.assertIn(self.ENV, str(error))

    def test_a_whitespace_run_token_is_refused_too(self):
        for name in MODULES:
            with self.subTest(module=name):
                outcome, error = self.load(name, {self.ENV: "  "})
                self.assertEqual(outcome, "REFUSED")
                self.assertIn(self.ENV, str(error))

    def test_an_unset_token_still_binds_the_default_run(self):
        self.assertEqual(self.bound("memory_book", {}), "1")

    def test_a_named_token_still_binds_as_it_did(self):
        self.assertEqual(self.bound("memory_volume", {self.ENV: "3"}), "3")


class TestImportIsClean(unittest.TestCase):
    def test_importing_a_driver_touches_no_network(self):
        child = """
import os, sys, urllib.request
urllib.request.urlopen = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("network"))
sys.path.insert(0, os.getcwd())
import {name}
print("imported")
"""
        root = tempfile.TemporaryDirectory()
        with mock.patch.dict(
            os.environ,
            {"TINYTITAN_MEMVAL_RESULTS": root.name, "TINYTITAN_MEMVAL_MEMDIR": root.name},
            clear=True,
        ):
            for name in MODULES + ("memory_mini",):
                with self.subTest(module=name):
                    import subprocess

                    done = subprocess.run(
                        ["python3", "-c", child.format(name=name)],
                        capture_output=True,
                        check=False,
                        text=True,
                    )
                    self.assertEqual(done.returncode, 0, done.stderr[-400:])
                    self.assertIn("imported", done.stdout)
        root.cleanup()


if __name__ == "__main__":
    unittest.main()
