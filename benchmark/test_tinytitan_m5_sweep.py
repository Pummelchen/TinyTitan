#!/usr/bin/env python3
"""Tests for benchmark/tinytitan_m5_sweep.py, model-free.

The driver runs the CLI once per configuration, so the whole suite is one patched
`subprocess.run` and a temp directory. Nothing here starts a server, loads a model
or converts anything: `run_once` only ever sees a fake `CompletedProcess`, and the
`--cli` argument points at an empty temp file whose only job is to exist.

What the suite is about, in the driver's own words and their defects:

1. `main()` had no return at all and the guard discarded whatever it returned, so
   a sweep in which every single run errored exited 0. The class is AUD-231/232/237.
2. `--slots abc` and `--slots ""` reached a bare `int()`, which is a traceback rather
   than a refusal naming the flag (the AUD-237 `--configs` shape), and `--slots 0` was
   accepted by that same `int()` and passed straight to the CLI.
3. A `KeyboardInterrupt` mid-sweep wrote the partial results and still exited 0, so
   an interrupted sweep and a completed one were indistinguishable in CI.
4. The missing-CLI message exited 1 through `sys.exit(str)` while a sweep in which
   every run failed exited 0, so "you did not build it" and "the model failed" were
   not merely shareable codes -- the refusal was the *stricter* of the two.
5. The results file had no output variable: it always landed in the repository's
   `benchmark/benchmark-results/`, while the three model-free harnesses beside it
   (`TINYTITAN_PROBE_OUT`, `TINYTITAN_ANE_PROBE_OUT`, `TINYTITAN_ANE_REHEARSAL_OUT`)
   each take one. The directory is git-ignored, so this is not a dirty tree -- it is
   synthetic rows written where the operator's real sweeps live.
6. The printed footer said `results written to <path>` with no count, so a reader of
   the log could not see how many of the planned runs were in it.
7. An error row keeps its configuration: `run_once` returns `{"slots": ..., "chunk":
   ..., "error": ...}` and the writer emits those two first. That part already worked
   and is pinned here, because it is the only thing that makes a failed arm
   attributable once it is a row in a file.

The mutation pass proved item 5 from the other side: the one mutant that ignores the
output variable wrote its synthetic rows into `benchmark/benchmark-results/` while it
was failing, so the override is what keeps a test run out of the directory the
operator's real sweeps land in.
"""

import contextlib
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import tinytitan_m5_sweep as sweep  # noqa: E402

BENCH = pathlib.Path(__file__).resolve().parent
DRIVER = BENCH / "tinytitan_m5_sweep.py"
FOOTER = "prefill={pt}tok/{ps}s new={nt}tok decode={ds}s tok/s={tps}\n"


def footer(prefill_tokens=8192, prefill_s=12.5, new_tokens=256, decode_s=5.0, tok_per_s=51.2):
    return FOOTER.format(pt=prefill_tokens, ps=prefill_s, nt=new_tokens, ds=decode_s, tps=tok_per_s)


class FakeProcess:
    def __init__(self, returncode=0, stderr=""):
        self.returncode = returncode
        self.stdout = ""
        self.stderr = stderr


def temp_dir():
    path = pathlib.Path(tempfile.mkdtemp(prefix="m5-sweep-"))
    return path


def flag_map(cmd):
    """One CLI invocation as flag -> value.

    `cmd[0]` is the binary, so the flags start at index 1. Strict, because a flag
    with no value is exactly the argv defect this class is looking for.
    """
    return dict(zip(cmd[1::2], cmd[2::2], strict=True))


class Runner:
    """Drive the real main() with the CLI subprocess replaced by replies.

    `replies` is one entry per planned run: a stderr string means a measured run,
    a `FakeProcess` with a non-zero returncode means the CLI refused, and a
    `KeyboardInterrupt` raised at that run stands in for Ctrl-C mid-sweep.
    """

    def __init__(self, replies):
        self.replies = list(replies)
        self.commands = []

    def run(self, cmd, **kwargs):
        self.commands.append((cmd, kwargs))
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, str):
            return FakeProcess(0, reply)
        return reply

    def cli_argv(self, directory, argv):
        cli = directory / "TinyTitanCLI"
        cli.write_text("")
        return [
            "--model",
            str(directory / "model.ssdai"),
            "--cli",
            str(cli),
            "--slots",
            "32,64",
            "--chunks",
            "512",
        ] + argv

    def call(self, argv=(), replies=None, env=None):
        directory = temp_dir()
        out_dir = directory / "results"
        runner = Runner(replies if replies is not None else self.replies)
        argv = self.cli_argv(directory, list(argv))
        buffer = io.StringIO()
        patcher = mock.patch.object(sweep.subprocess, "run", runner.run)
        env_map = {"TINYTITAN_M5_SWEEP_OUT": str(out_dir)}
        if env is not None:
            env_map = {k: v for k, v in env.items() if v is not None}
        with (
            mock.patch.object(sys, "argv", ["tinytitan_m5_sweep.py"] + argv),
            mock.patch.dict(os.environ, env_map, clear=True),
            patcher,
            contextlib.redirect_stdout(buffer),
        ):
            try:
                status = sweep.main()
            except SystemExit as exit_error:
                status = exit_error
        printed = buffer.getvalue()
        csv_path = out_dir / "m5_sweep_results.csv"
        csv_text = csv_path.read_text() if csv_path.exists() else None
        self.runner = runner
        return status, printed, csv_text


class StatusIsReturned(unittest.TestCase):
    def setUp(self):
        self.r = Runner([footer(), footer(), footer()])

    def test_a_sweep_that_measured_every_run_returns_0(self):
        status, _, _ = self.r.call()
        self.assertEqual(status, 0, "main() must return its status, not print one")

    def test_main_returns_an_int_on_every_path(self):
        source = DRIVER.read_text()
        body = source.split("def main() -> int:")[1].split('if __name__ == "__main__":')[0]
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("return "):
                value = stripped[len("return ") :].strip()
                self.assertTrue(
                    value.isdigit() or value == "status",
                    f"`{stripped}` hands the guard nothing usable",
                )

    def test_the_guard_propagates_the_status(self):
        self.assertIn("sys.exit(main())", DRIVER.read_text())
        self.assertNotIn("    main()\n", DRIVER.read_text())

    def test_a_sweep_where_every_run_errored_does_not_return_0(self):
        self.r.replies = [FakeProcess(1, "route index missing")] * 3
        status, printed, csv_text = self.r.call()
        self.assertEqual(status, 1)
        self.assertIn("NOT MEASURED", printed)
        self.assertIsNotNone(csv_text, "the errors are the result; the file still gets written")

    def test_one_errored_config_costs_the_run(self):
        self.r.replies = [footer(), FakeProcess(1, "kv cache too small"), footer()]
        status, printed, _ = self.r.call()
        self.assertEqual(status, 1)
        self.assertIn("ERROR", printed)

    def test_a_clean_exit_with_no_timing_footer_is_an_error_not_a_measurement(self):
        # The CLI can exit 0 and still say nothing the sweep can record; a row of
        # zeros would otherwise count as a measurement of a real configuration.
        self.r.replies = [FakeProcess(0, "generation finished, no footer line")] + [footer()] * 2
        status, printed, _ = self.r.call()
        self.assertEqual(status, 1)
        self.assertIn("no timing footer", printed)
        self.assertIn("2 of 3", printed)

    def test_an_interrupted_sweep_does_not_claim_a_completed_one(self):
        self.r.replies = [footer(), KeyboardInterrupt(), footer()]
        status, printed, csv_text = self.r.call()
        self.assertEqual(status, 1)
        self.assertIn("interrupted: 2 run(s) never attempted", printed)
        self.assertIsNotNone(csv_text)
        self.assertEqual(csv_text.count("\n"), 2, "the header plus the one run that finished")


class Refusals(unittest.TestCase):
    """A refusal is status 2 with a line that names the flag -- never a traceback.

    `main()` returns rather than raising: a status the guard can propagate is the
    whole point of the fix, so every case here asserts on the returned int.
    """

    def setUp(self):
        # replies that error rather than exhausting, so a driver that refuses to
        # refuse fails on the status instead of raising IndexError in the fixture.
        self.r = Runner([FakeProcess(1, "ran anyway")] * 6)

    def test_a_non_numeric_slot_is_refused_naming_the_flag(self):
        status, printed, csv_text = self.r.call(["--slots", "32,abc"])
        self.assertEqual(status, 2, printed)
        self.assertIn("--slots", printed)
        self.assertNotIn("Traceback", printed)
        self.assertIsNone(csv_text, "a refusal plans no runs and writes no results")

    def test_a_zero_chunk_is_refused_before_any_run_starts(self):
        status, printed, _ = self.r.call(["--chunks", "0"])
        self.assertEqual(status, 2, printed)
        self.assertIn("--chunks", printed)
        self.assertEqual(self.r.runner.commands, [])

    def test_an_empty_list_is_refused_as_a_list_and_not_as_an_int_error(self):
        status, printed, _ = self.r.call(["--slots", ""])
        self.assertEqual(status, 2, printed)
        self.assertIn("--slots", printed)
        self.assertNotIn("invalid literal", printed)

    def test_a_missing_cli_names_the_build_command_and_returns_2(self):
        status, printed, _ = self.r.call(["--cli", str(temp_dir() / "nowhere")])
        self.assertEqual(status, 2, printed)
        self.assertIn("swift build -c release", printed)


class WhatGetsWritten(unittest.TestCase):
    def setUp(self):
        # default plan is three runs: slots 32 and 64 at chunk 4096, then chunk 512
        # at slots 64. One of them measures.
        self.r = Runner([footer(), FakeProcess(1, "no footer here"), FakeProcess(1, "nor here")])

    def test_the_output_variable_moves_the_results_file(self):
        _, _, csv_text = self.r.call()
        self.assertIsNotNone(csv_text)

    def test_the_absent_variable_falls_back_to_the_repository_results_dir(self):
        # `clear=True` with no key set means TINYTITAN_M5_SWEEP_OUT is genuinely
        # absent, which is the case that must not write into a tracked directory
        # from a test run: the default has to be a path the repository owns and
        # the harnesses already share.
        default = sweep.results_path({})
        self.assertEqual(default.name, "m5_sweep_results.csv")
        self.assertTrue(
            str(default).endswith(os.path.join("benchmark-results", "m5_sweep_results.csv"))
        )

    def test_a_relative_output_value_stays_under_the_repository(self):
        # A relative value resolved against the working directory would land in
        # whatever directory the sweep happened to be launched from.
        path = sweep.results_path({sweep.OUT_ENV: "sweeps"})
        self.assertTrue(str(path).startswith(str(sweep.repository_root())), str(path))
        self.assertEqual(path.parent.name, "sweeps")
        self.assertEqual(path.name, sweep.ARTIFACT)

    def test_an_output_value_ending_in_csv_is_the_file_itself(self):
        path = sweep.results_path({sweep.OUT_ENV: "sweeps/run.csv"})
        self.assertEqual(path.name, "run.csv")
        self.assertEqual(path.parent.name, "sweeps")

    def test_the_footer_says_how_many_runs_are_in_the_file(self):
        _, printed, _ = self.r.call()
        self.assertIn("1 of 3", printed)

    def test_an_error_row_keeps_its_configuration(self):
        _, _, csv_text = self.r.call(
            ["--sweep", "slots", "--slots", "32"], replies=[FakeProcess(1, "boom")]
        )
        rows = [line for line in csv_text.splitlines()[1:] if line.strip()]
        self.assertEqual(len(rows), 1, csv_text)
        self.assertTrue(rows[0].startswith("32,4096,"), rows[0])
        self.assertIn("boom", rows[0])


class WhatRunsForEachConfig(unittest.TestCase):
    """Pins the argv the CLI is called with, so a sweep means what its header says."""

    def test_the_slot_sweep_pins_chunk_4096_and_the_chunk_sweep_slots_64(self):
        r = Runner([footer(), footer(), footer()])
        r.call()
        passed = [flag_map(cmd) for cmd, _ in r.runner.commands]
        # two slot runs at chunk 4096, then one chunk run at slots 64
        self.assertEqual([p["--expert-cache-slots"] for p in passed], ["32", "64", "64"])
        self.assertEqual([p["--prefill-chunk"] for p in passed], ["4096", "4096", "512"])

    def test_the_timeout_is_stated_on_every_call(self):
        r = Runner([footer()])
        r.call(["--sweep", "slots"], replies=[footer(), footer()])
        for _, kwargs in r.runner.commands:
            self.assertEqual(kwargs["timeout"], sweep.RUN_TIMEOUT_S)
            self.assertFalse(kwargs["check"])


class PromptSizing(unittest.TestCase):
    def test_a_context_too_small_for_the_prompt_is_refused_before_the_sweep(self):
        r = Runner([FakeProcess(1, "ran anyway")] * 6)
        status, printed, _ = r.call(["--prompt-tokens", "8192", "--max-context", "4096"])
        self.assertEqual(status, 2, printed)
        self.assertNotEqual(status, 1, "a refusal is not the status an errored run gives")
        self.assertEqual(r.runner.commands, [])

    def test_the_drift_between_estimate_and_tokenizer_is_printed(self):
        r = Runner([footer(prefill_tokens=12000), footer(prefill_tokens=12000)])
        _, printed, _ = r.call(
            ["--prompt-tokens", "8192"], replies=[footer(prefill_tokens=12000)] * 3
        )
        self.assertIn("tokenizer produced 12000", printed)
        self.assertIn("estimate off by", printed)

    def test_a_prompt_file_is_written_and_removed(self):
        written = {}

        def capture(content):
            written["file"] = content
            return "{}"

        r = Runner([footer(), footer(), footer()])
        with mock.patch.object(sweep.json, "dumps", side_effect=lambda obj: capture(obj) or "{}"):
            r.call()
        self.assertIn("user", json.dumps(written.get("file")))
        first_run = r.runner.commands[0][0]
        prompt = pathlib.Path(flag_map(first_run)["--messages-file"])
        self.assertFalse(prompt.exists(), "the prompt outlives the sweep")


class NoImportSideEffects(unittest.TestCase):
    def test_importing_the_driver_runs_nothing(self):
        child = f"""
import builtins, subprocess, sys, pathlib
sys.path.insert(0, {str(pathlib.Path(__file__).resolve().parent)!r})
seen = []
real_open = builtins.open
def watched(*a, **k):
    seen.append(str(a[0]))
    return real_open(*a, **k)
builtins.open = watched
real_run = subprocess.run
def refused(*a, **k):
    raise AssertionError("a CLI ran at import: " + str(a[0]))
subprocess.run = refused
import tinytitan_m5_sweep as sweep
builtins.open = real_open
if seen:
    raise SystemExit("import opened: " + repr(seen))
if getattr(sweep, "PROMPT_FILE", None):
    raise SystemExit("a prompt file survived the fix")
print("imported clean")
"""
        script = temp_dir() / "child.py"
        script.write_text(child)
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(pathlib.Path(__file__).resolve().parents[1]),
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("imported clean", result.stdout)


if __name__ == "__main__":
    unittest.main()
