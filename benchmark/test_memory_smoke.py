#!/usr/bin/env python3
"""Tests for benchmark/memory_smoke.py, driven model-free.

The smoke check asks a real model four questions about memory and prints one
verdict. These tests pin the ways that verdict can be reached without the
questions being answered, and they need no model: every request is served by a
fake `urlopen` and every file by a temporary directory.

1. A request that never answered was scored as a passed check. `main()` reads
   the status of `novel-1` and nothing else, so a 500 -- or a 200 with an empty
   message -- to the isolation probe leaves `reply` the empty string, which
   cannot contain "ashgrove", and the run prints `SMOKE OK: placement by
   declared directory` on the strength of a request that never ran.
2. A check that cannot run is not a check that passed. `if MEMDIR.exists():`
   skips the whole journal section with no line at all, and `MEMDIR =
   Path(os.environ.get(..., ""))` turns an unset variable into `Path('.')`, so a
   direct run recursively scans the directory it was launched from and can pass
   the section on files that have nothing to do with the run. The sibling driver
   does both of these right -- `memory_projects.py:364` guards on the value and
   `:449` aborts naming the variables to set.
3. A missing log was reported as the model's failure. `consolidation_lines()`
   returns `[]` when `TINYTITAN_MEMVAL_SERVER_LOG` names nothing, and `main()`
   then sleeps 180 seconds and appends "no consolidation of the novel session",
   which blames the engine for a file that was never there.
4. A refusal was a traceback. `model_id()` has no error path, so a server that
   is not listening ends the run with an uncaught `URLError`.
5. The verdict had no exit code of its own. `main()` returns `None` and the
   guard calls bare `main()`; the only exit path is a `raise SystemExit(1)`
   inside it. Configuration was frozen at import, so an operator's environment
   is invisible to a run that sets it after the file is read.

No server is contacted and no port is opened: `urllib.request.urlopen`,
`time.sleep` and the two environment variables are faked, and `time.monotonic`
is stepped by hand.
"""

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmark"
sys.path.insert(0, str(BENCH))

import memory_smoke as ms  # noqa: E402

CONSOLIDATION = "2026-10-08 memory consolidated session=photograph-1 facts=1 in 0.9s"
NO_FACTS = "2026-10-08 memory consolidated session=photograph-1 facts=0 in 0.4s"
FACT = "Ashgrove"
CHAT_CALLS = 3


def row(label, *, status=200, content="", prompt_tokens=149, completion_tokens=12, error=None):
    return {
        "label": label,
        "status": status,
        "content": content,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "error": error,
    }


class Resp(io.BytesIO):
    """A `urlopen` result: readable as JSON, with the status the driver reads."""

    def __init__(self, payload, status=200):
        raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        super().__init__(raw)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def completion(content, *, prompt_tokens=149, completion_tokens=12):
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def models_payload():
    return {"data": [{"id": "TinyTitan-Ornith"}]}


class Gateway:
    """The fake transport: `/v1/models` plus the three chat requests.

    The script is keyed by cell label; a value that is an exception class or
    instance is raised in its place, which is how a refusal and a mid-stream
    drop are reproduced without a socket.
    """

    def __init__(self, script, *, models=None):
        self.script = script
        self.models = models if models is not None else models_payload()
        self.calls = []
        self.urls = []

    def __call__(self, request, timeout=None):
        url = getattr(request, "full_url", request)
        if url.endswith("/models"):
            if isinstance(self.models, Exception):
                raise self.models
            return Resp(self.models)
        self.urls.append(url)
        body = json.loads(request.data.decode())
        label = classify(body)
        self.calls.append((label, body))
        outcome = self.script.get(label, self.script.get("default"))
        if isinstance(outcome, Exception):
            raise outcome
        if outcome is None:
            raise AssertionError(f"no scripted answer for {label}")
        return Resp(outcome)


def classify(body):
    system = body["messages"][0]["content"]
    user = body["messages"][1]["content"]
    if "widget" in system:
        return "code-1"
    if user.startswith("Store this"):
        return "novel-1"
    if user.startswith("What do you already know"):
        return "novel-2"
    raise AssertionError(f"unrecognized request: {user[:40]}")


HEALTHY = {
    "novel-1": completion("Confirmed: in this novel the town is Ashgrove and it never rains."),
    "code-1": completion("nothing"),
    "novel-2": completion("The novel is set in Ashgrove, where it never rains."),
}


class ImportTests(unittest.TestCase):
    def test_importing_the_module_makes_no_request_and_opens_no_port(self):
        """The child's guard raises if import-time code touches the network."""
        with tempfile.TemporaryDirectory() as tmp:
            guards = (
                "import builtins, urllib.request, sys\n"
                "def trip(*a, **k):\n"
                "    raise SystemExit('import made a request')\n"
                "urllib.request.urlopen = trip\n"
                "_open = builtins.open\n"
                "def guarded_open(name, *a, **k):\n"
                "    if 'server' in str(name) or '.log' in str(name):\n"
                "        raise SystemExit('import opened the server log')\n"
                "    return _open(name, *a, **k)\n"
                "builtins.open = guarded_open\n"
                "import importlib.util\n"
                "sys.path.insert(0, str(__import__('pathlib').Path(sys.argv[1]).parent))\n"
                "spec = importlib.util.spec_from_file_location('ms', sys.argv[1])\n"
                "mod = importlib.util.module_from_spec(spec)\n"
                "spec.loader.exec_module(mod)\n"
                "print('imported clean')\n"
            )
            script = Path(tmp) / "guard.py"
            script.write_text(guards, encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(script), str(BENCH / "memory_smoke.py")],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("imported clean", proc.stdout)

    def test_the_documented_runner_command_exists_and_takes_smoke(self):
        """The docstring's one command line has to be a command that runs."""
        runner = BENCH / "memval_run.sh"
        self.assertTrue(runner.exists(), "the docstring names a script that is not here")
        text = runner.read_text(encoding="utf-8")
        self.assertIn("smoke", text)

    def test_the_stated_request_count_matches_the_requests_it_sends(self):
        doc = (ms.__doc__ or "").lower()
        self.assertIn("three requests", doc)


class ReportTests(unittest.TestCase):
    """Each report must distinguish 'checked and clean' from 'could not check'."""

    def test_a_failed_isolation_request_is_not_a_passing_isolation_check(self):
        lines, status = ms.isolation_report(row("code-1", status=500, content=""), saw_fact=False)
        self.assertEqual(status, 1)
        self.assertIn("NOT MEASURED", "\n".join(lines))

    def test_an_empty_answer_is_reported_as_empty_and_not_as_a_clean_check(self):
        lines, status = ms.isolation_report(row("code-1", content=""), saw_fact=False)
        self.assertEqual(status, 1)
        self.assertIn("no content", "\n".join(lines))

    def test_a_status_that_is_not_200_is_blocked_even_with_a_body(self):
        """The error branch must not be masked by the empty-content branch."""
        lines, status = ms.isolation_report(row("code-1", status=503, content="degraded"), False)
        self.assertEqual(status, 1)
        self.assertIn("HTTP 503", "\n".join(lines))

    def test_a_row_that_raised_names_the_error_rather_than_a_status_it_never_had(self):
        lines, status = ms.recall_report(
            row("novel-2", status=None, content="", error="the request raised ConnectionReset: x"),
            saw_fact=False,
        )
        self.assertEqual(status, 1)
        self.assertIn("ConnectionReset", "\n".join(lines))
        self.assertNotIn("HTTP None", "\n".join(lines))

    def test_a_leaked_fact_fails_the_isolation_check(self):
        lines, status = ms.isolation_report(
            row("code-1", content="the town is Ashgrove"), saw_fact=True
        )
        self.assertEqual(status, 1)
        self.assertIn("saw", "\n".join(lines))

    def test_a_answered_and_clean_isolation_check_passes(self):
        lines, status = ms.isolation_report(row("code-1", content="nothing"), saw_fact=False)
        self.assertEqual(status, 0)
        self.assertIn("code-1", "\n".join(lines))

    def test_a_failed_recall_request_is_not_reported_as_a_miss(self):
        """A 500 cannot miss the fact; it never asked."""
        lines, status = ms.recall_report(row("novel-2", status=502, content=""), saw_fact=False)
        self.assertEqual(status, 1)
        self.assertIn("NOT MEASURED", "\n".join(lines))

    def test_a_recall_that_missed_is_named_as_a_miss(self):
        lines, status = ms.recall_report(row("novel-2", content="I know nothing"), saw_fact=False)
        self.assertEqual(status, 1)
        self.assertIn("did not recall", "\n".join(lines))

    def test_a_recall_of_the_fact_passes(self):
        lines, status = ms.recall_report(row("novel-2", content="Ashgrove"), saw_fact=True)
        self.assertEqual(status, 0)

    def test_a_short_prompt_is_reported_as_a_fragment_that_is_not_there(self):
        lines, status = ms.prompt_floor_report(
            row("novel-1", content="Confirmed: Ashgrove.", prompt_tokens=61)
        )
        self.assertEqual(status, 1)
        self.assertIn("61", "\n".join(lines))
        self.assertIn(str(ms.PROMPT_FLOOR_TOKENS), "\n".join(lines))

    def test_a_prompt_at_the_floor_is_a_measurement(self):
        lines, status = ms.prompt_floor_report(
            row("novel-1", content="Confirmed: Ashgrove.", prompt_tokens=149)
        )
        self.assertEqual(status, 0)

    def test_no_log_configured_is_not_the_model_failing_to_consolidate(self):
        lines, status = ms.consolidation_report(None, [])
        self.assertEqual(status, 1)
        text = "\n".join(lines)
        self.assertIn(ms.LOG_ENV, text)
        self.assertNotIn("no consolidation of the novel session", text)

    def test_a_log_that_does_not_exist_says_the_log_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "server-arm.log"
            lines, status = ms.consolidation_report(missing, [])
            self.assertEqual(status, 1)
            self.assertIn("server log", "\n".join(lines).lower())

    def test_a_log_with_no_consolidation_line_blames_the_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "server-arm.log"
            log.write_text("boot\n", encoding="utf-8")
            lines, status = ms.consolidation_report(log, [])
            self.assertEqual(status, 1)
            self.assertIn("no consolidation", "\n".join(lines))

    def test_a_consolidation_that_wrote_no_facts_is_a_failed_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "server-arm.log"
            log.write_text(NO_FACTS + "\n", encoding="utf-8")
            lines, status = ms.consolidation_report(log, ms.consolidation_lines(log))
            self.assertEqual(status, 1)
            self.assertIn("no facts", "\n".join(lines))

    def test_a_consolidation_with_facts_is_a_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "server-arm.log"
            log.write_text(CONSOLIDATION + "\n", encoding="utf-8")
            lines, status = ms.consolidation_report(log, [CONSOLIDATION])
            self.assertEqual(status, 0)

    def test_no_memdir_configured_does_not_scan_the_launch_directory(self):
        lines, status = ms.journal_report(None, [])
        self.assertEqual(status, 1)
        self.assertIn(ms.MEMDIR_ENV, "\n".join(lines))

    def test_a_memdir_that_does_not_exist_is_a_check_that_did_not_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "memory"
            lines, status = ms.journal_report(missing, [])
            self.assertEqual(status, 1)
            self.assertIn("NOT CHECKED", "\n".join(lines))

    def test_both_workspaces_present_is_a_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            lines, status = ms.journal_report(Path(tmp), ["photograph-1.ndjson", "widget-1.ndjson"])
            self.assertEqual(status, 0, "\n".join(lines))

    def test_a_missing_workspace_is_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            lines, status = ms.journal_report(Path(tmp), ["widget-1.ndjson"])
            self.assertEqual(status, 1)
            self.assertIn("photograph", "\n".join(lines))


class DriverTests(unittest.TestCase):
    """Drive the real main() with the transport and the clock faked."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.where = Path(self.tmp.name)
        self.memdir = self.where / "memory"
        self.memdir.mkdir()
        (self.memdir / "photograph-1.ndjson").write_text("{}\n", encoding="utf-8")
        (self.memdir / "widget-1.ndjson").write_text("{}\n", encoding="utf-8")
        self.log = self.where / "server-auto-r1.log"
        self.log.write_text("boot\n", encoding="utf-8")

    def run_main(self, *, script=HEALTHY, env=None, gateway_models=None, sleep_step=None):
        variables = {
            ms.MEMDIR_ENV: str(self.memdir),
            ms.LOG_ENV: str(self.log),
            "TINYTITAN_PORT": "8096",
            **(env or {}),
        }
        gateway = Gateway(script, models=gateway_models)
        self.gateway = gateway
        sleeps = []

        def fake_sleep(seconds):
            sleeps.append(seconds)
            if sleep_step is not None:
                sleep_step()

        patches = [
            mock.patch.dict(os.environ, variables, clear=False),
            mock.patch.object(ms.urllib.request, "urlopen", new=gateway),
            mock.patch.object(ms.time, "sleep", new=fake_sleep),
        ]
        argv_backup = list(sys.argv)
        sys.argv = ["memory_smoke.py"]
        out, err = io.StringIO(), io.StringIO()
        status = None
        try:
            for patcher in patches:
                patcher.start()
            with redirect_stdout(out), redirect_stderr(err):
                status = ms.main()
        except SystemExit as exc:  # the pre-fix guard raises rather than returning
            status = exc.code
            escaped = True
        else:
            escaped = False
        finally:
            for patcher in reversed(patches):
                try:
                    patcher.stop()
                except RuntimeError:
                    pass
            sys.argv = argv_backup
        return status, out.getvalue() + err.getvalue(), gateway.calls, sleeps, escaped

    def write_consolidation(self, line=CONSOLIDATION):
        self.log.write_text("boot\n" + line + "\n", encoding="utf-8")

    def test_the_healthy_run_passes_every_check_and_returns_zero(self):
        self.write_consolidation()
        status, output, calls, _sleeps, escaped = self.run_main()
        self.assertFalse(escaped, "main() raised instead of returning its status")
        self.assertEqual(status, 0, output)
        self.assertIn("SMOKE OK", output)
        self.assertEqual([label for label, _body in calls], ["novel-1", "code-1", "novel-2"])

    def test_a_500_on_the_isolation_probe_does_not_pass_the_isolation_check(self):
        """The headline defect: an unanswered request cannot show no leakage."""
        self.write_consolidation()
        script = dict(HEALTHY, **{"code-1": urllib.error.HTTPError("", 500, "boom", {}, None)})
        status, output, _calls, _sleeps, _escaped = self.run_main(script=script)
        self.assertEqual(status, 1)
        self.assertIn("NOT MEASURED", output)
        self.assertNotIn("SMOKE OK", output)

    def test_an_empty_isolation_answer_is_not_a_clean_check(self):
        self.write_consolidation()
        script = dict(HEALTHY, **{"code-1": completion("")})
        status, output, _calls, _sleeps, _escaped = self.run_main(script=script)
        self.assertEqual(status, 1)
        self.assertNotIn("SMOKE OK", output)

    def test_a_request_that_raised_is_a_report_not_a_traceback(self):
        self.write_consolidation()
        script = dict(HEALTHY, **{"novel-2": urllib.error.URLError("connection reset")})
        status, output, _calls, _sleeps, escaped = self.run_main(script=script)
        self.assertFalse(escaped, "the driver's transport error escaped main()")
        self.assertEqual(status, 1)
        self.assertIn("NOT MEASURED", output)

    def test_a_server_that_is_not_listening_runs_no_request(self):
        gateway_models = urllib.error.URLError("connection refused")
        status, output, calls, _sleeps, escaped = self.run_main(gateway_models=gateway_models)
        self.assertFalse(escaped, "a refused connection was an uncaught traceback")
        self.assertEqual(status, 1)
        self.assertEqual(calls, [])
        self.assertIn("NOT RUN", output)

    def test_the_journal_check_never_reads_the_launch_directory(self):
        """`Path(os.environ.get(..., ""))` is `Path('.')`, which exists."""
        self.write_consolidation()
        decoys = self.where / "cwd"
        decoys.mkdir()
        (decoys / "photograph-9.ndjson").write_text("{}\n", encoding="utf-8")
        (decoys / "widget-9.ndjson").write_text("{}\n", encoding="utf-8")
        cwd = os.getcwd()
        os.chdir(decoys)
        try:
            status, output, _calls, _sleeps, _escaped = self.run_main(env={ms.MEMDIR_ENV: ""})
        finally:
            os.chdir(cwd)
        self.assertEqual(status, 1)
        self.assertIn(ms.MEMDIR_ENV, output)
        self.assertNotIn("SMOKE OK", output)

    def test_a_memdir_that_does_not_exist_is_not_a_skipped_section(self):
        self.write_consolidation()
        status, output, _calls, _sleeps, _escaped = self.run_main(
            env={ms.MEMDIR_ENV: str(self.where / "no-such-memory")}
        )
        self.assertEqual(status, 1)
        self.assertIn("NOT CHECKED", output)
        self.assertNotIn("SMOKE OK", output)

    def test_a_missing_log_is_not_reported_as_a_write_the_model_did_not_make(self):
        status, output, _calls, sleeps, _escaped = self.run_main(
            env={ms.LOG_ENV: str(self.where / "gone.log")}
        )
        self.assertEqual(status, 1)
        self.assertNotIn("no consolidation of the novel session", output)
        self.assertNotIn(ms.CONSOLIDATION_POLL_SECONDS, sleeps)
        self.assertLessEqual(sum(sleeps), ms.JOURNAL_SETTLE_SECONDS)

    def test_an_unconfigured_log_waits_nothing_and_names_the_variable(self):
        status, output, _calls, sleeps, _escaped = self.run_main(env={ms.LOG_ENV: ""})
        self.assertEqual(status, 1)
        self.assertNotIn(ms.CONSOLIDATION_POLL_SECONDS, sleeps)
        self.assertLessEqual(sum(sleeps), ms.JOURNAL_SETTLE_SECONDS)
        self.assertIn(ms.LOG_ENV, output)

    def test_a_consolidation_that_wrote_nothing_fails_the_check_after_waiting(self):
        status, output, _calls, sleeps, _escaped = self.run_main()
        self.assertEqual(status, 1)
        self.assertIn("no consolidation", output)
        self.assertGreaterEqual(sum(sleeps), 60, "it gave up before the idle window")

    def test_a_consolidation_with_no_facts_is_a_failed_check(self):
        self.write_consolidation(NO_FACTS)
        status, output, _calls, _sleeps, _escaped = self.run_main()
        self.assertEqual(status, 1)
        self.assertIn("no facts", output)

    def test_a_prompt_below_the_floor_is_a_memory_fragment_that_is_absent(self):
        self.write_consolidation()
        script = dict(HEALTHY, **{"novel-1": completion("Confirmed: Ashgrove.", prompt_tokens=61)})
        status, output, _calls, _sleeps, _escaped = self.run_main(script=script)
        self.assertEqual(status, 1)
        self.assertIn("61", output)
        self.assertNotIn("SMOKE OK", output)

    def test_the_recall_that_missed_fails_the_check(self):
        self.write_consolidation()
        script = dict(HEALTHY, **{"novel-2": completion("I know nothing about this novel.")})
        status, output, _calls, _sleeps, _escaped = self.run_main(script=script)
        self.assertEqual(status, 1)
        self.assertIn("did not recall", output)

    def test_a_leaked_fact_fails_the_check_even_though_nothing_errored(self):
        self.write_consolidation()
        script = dict(HEALTHY, **{"code-1": completion("The town is Ashgrove.")})
        status, output, _calls, _sleeps, _escaped = self.run_main(script=script)
        self.assertEqual(status, 1)
        self.assertIn("saw", output)

    def test_the_port_is_read_when_the_run_starts_not_when_the_file_is_imported(self):
        """Two runs, two ports: an import-time freeze would send both to the first."""
        self.write_consolidation()
        status, output, calls, _sleeps, escaped = self.run_main()
        self.assertFalse(escaped)
        self.assertEqual(status, 0, output)
        self.assertEqual(len(calls), CHAT_CALLS)
        self.assertTrue(all("8096" in url for url in self.gateway.urls[1:]))
        self.write_consolidation()
        status, output, calls, _sleeps, escaped = self.run_main(env={"TINYTITAN_PORT": "9123"})
        self.assertEqual(status, 0, output)
        self.assertTrue(all("9123" in url for url in self.gateway.urls[1:]), output)

    def test_a_port_that_is_not_a_number_is_refused_before_any_request(self):
        self.write_consolidation()
        status, output, calls, _sleeps, escaped = self.run_main(
            env={"TINYTITAN_PORT": "not-a-port"}
        )
        self.assertFalse(escaped, "a bad port raised instead of refusing")
        self.assertEqual(status, 2)
        self.assertEqual(calls, [])
        self.assertIn("REFUSED", output)

    def test_a_port_no_server_can_bind_is_refused_before_any_request(self):
        self.write_consolidation()
        status, output, calls, _sleeps, escaped = self.run_main(env={"TINYTITAN_PORT": "80"})
        self.assertFalse(escaped)
        self.assertEqual(status, 2)
        self.assertEqual(calls, [])
        self.assertIn("is not a port a server can bind", output)

    def test_main_returns_an_int_and_the_guard_exits_with_it(self):
        source = (BENCH / "memory_smoke.py").read_text(encoding="utf-8")
        self.assertIn("sys.exit(main())", source)
        self.assertNotIn("\n    main()\n", source)
        self.assertNotIn("raise SystemExit(1)", source)

    def test_the_three_environment_inputs_are_read_inside_functions(self):
        source = (BENCH / "memory_smoke.py").read_text(encoding="utf-8")
        for line in source.splitlines():
            if line.startswith((" ", "\t", "#")) or "os.environ" not in line:
                continue
            self.fail(f"the environment is read at module scope: {line.strip()}")

    def test_nothing_is_written_into_the_repository_by_a_run(self):
        self.write_consolidation()
        before = set(os.listdir(BENCH))
        status, output, _calls, _sleeps, _escaped = self.run_main()
        self.assertEqual(set(os.listdir(BENCH)), before)

    def test_the_floor_it_checks_is_the_number_its_comment_measures(self):
        """The comment says the answered prompt measured 149; the floor must sit
        under that and above a request with no fragment, or the check is a guess."""
        source = (BENCH / "memory_smoke.py").read_text(encoding="utf-8")
        self.assertIn(str(ms.PROMPT_FLOOR_TOKENS), source)
        self.assertGreater(ms.PROMPT_FLOOR_TOKENS, 100)
        self.assertLess(ms.PROMPT_FLOOR_TOKENS, 149)
        self.assertIn("149", source)


class TransportTests(unittest.TestCase):
    def test_a_body_that_is_not_json_is_reported_and_not_a_crash(self):
        with (
            mock.patch.object(
                ms.urllib.request, "urlopen", return_value=Resp(b"<html>gateway error</html>")
            ),
            redirect_stdout(io.StringIO()),
        ):
            result = ms.ask("m", ms.NOVEL, "Store this in memory for later sessions.", "novel-1")
        self.assertEqual(result["content"], "")
        self.assertIsNotNone(result["error"])

    def test_a_refusal_mid_run_is_a_row_with_an_error_and_no_status(self):
        with (
            mock.patch.object(
                ms.urllib.request,
                "urlopen",
                side_effect=urllib.error.URLError("connection reset"),
            ),
            redirect_stdout(io.StringIO()),
        ):
            result = ms.ask("m", ms.CODE, "What do you know?", "code-1")
        self.assertIsNone(result["status"])
        self.assertIn("connection reset", result["error"])


if __name__ == "__main__":
    unittest.main()
