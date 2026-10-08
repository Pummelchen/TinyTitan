"""Gates `benchmark/tinytitan_longgen.py`'s mean, its attribution and its status.

The driver answers one question — what decode rate does a 512-token greedy
code-generation prompt hold over a long run — by sending one warm-up and three
measured requests against one server and reading the server's own
`decode_tok_s=` footers. Five things made that answer unreadable, and every one
of them is measured here without a model, a server or a port:

1. The docstring's protocol is "1 warmup + 3 measured runs", and the driver
   applied it to only one of its two columns: `ct={cts[1:] if len(cts) > 1 else
   cts}` drops the warm-up's completion count, while `rates` keeps every footer
   and `mean = sum(rates) / len(rates)` averages all four. The headline
   `mean=42.50 tok/s` was therefore three runs' rate wearing a warm-up's number,
   and the two columns on the same line did not describe the same requests.
2. Nothing counted footers against requests. Three footers for four requests
   averaged three; a single footer printed `ct=[...]` *including the warm-up*
   through the `len(cts) > 1` branch, and the mean moved with whichever footer
   the server happened not to print.
3. An arm whose log carried no footer printed `rates=[] mean=0.00 tok/s ct=[]` —
   `if rates else 0` turns an empty measurement into a number — and `main()`
   returns nothing anywhere in the file, so a sweep that measured nothing exited
   **0**. A `{label}: FAILED` arm was a line on stdout and the same exit 0.
4. The health wait was inlined and its budget expired *into the run*: the `while`
   loop breaks only on a healthy reply, so a server still loading after 120 s
   fell through to `resolve_api_model()` and the POSTs — and `resolve_api_model`
   answers a fallback id when it cannot reach a server, so the driver sent a
   512-token generation to a server that never said it was ready and reported
   whatever the log happened to hold.
5. The default install came from `DEFAULT_MODEL_PATH` rather than `bench_model()`,
   so `TINYTITAN_BENCH_MODEL` was ignored; a path naming no width was labelled
   `4bit`, a quantisation claim about a run that did not make one; and the log
   name carried only that label, so two 4-bit installs on one command line — the
   documented usage — wrote the same `longgen_4bit.log`, and the second arm
   truncated the first one's capture.

    cd benchmark && python3 -m unittest test_tinytitan_longgen -v
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import tinytitan_profile

ROOT = pathlib.Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "tinytitan_longgen", ROOT / "benchmark" / "tinytitan_longgen.py"
)
lg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lg)

GEN = "TinyTitan generation prefill_s=4.912 decode_s=11.900 decode_tok_s={rate}"
NO_RATE = "TinyTitan generation started"
COMPLETED = "request resp_0123456789 completed in 10.4s prompt=31 cached=0 completion={ct}"


def footers(rates):
    return [GEN.format(rate=rate) for rate in rates]


class LabelTests(unittest.TestCase):
    """What an arm is called, in the row and in the log."""

    def test_a_width_in_the_path_is_the_label(self):
        self.assertEqual(lg.model_label("models/ornith-1.5_35B_A3B_8Bit"), "8bit")
        self.assertEqual(lg.model_label("models/qwen3.6_35B_A3B_6bit"), "6bit")

    def test_a_path_naming_no_width_keeps_its_own_name(self):
        """The old ternary fell through to `4bit` for any path that did not say, so
        an install whose name carries no width was reported at one it never ran at."""
        self.assertEqual(lg.model_label("models/qwen3.8-flash-next_125B_A6B_4Bit"), "4bit")
        self.assertEqual(
            lg.model_label("models/ornith-1.5_35B_A3B_final"), "ornith-1.5_35B_A3B_final"
        )

    def test_the_tag_names_the_install_as_well_as_the_width(self):
        """Two 4-bit installs must not print two rows that read identically."""
        self.assertEqual(lg.tag_for("models/a_4Bit"), "4bit-a_4Bit")

    def test_a_width_less_install_is_not_named_twice(self):
        """Its directory name already *is* its label, so joining the two would print
        the same name twice in the row and in the log file."""
        self.assertEqual(lg.tag_for("models/ornith-1.5_35B_A3B_final"), "ornith-1.5_35B_A3B_final")

    def test_the_log_name_carries_the_tag_so_two_same_width_installs_do_not_share_it(self):
        first = lg.log_name_for("models/a_4Bit")
        second = lg.log_name_for("models/b_4Bit")
        self.assertNotEqual(first, second)
        self.assertIn("longgen", first)


class ScrapeTests(unittest.TestCase):
    """Which log lines are the measurement."""

    def test_a_generation_footer_is_a_rate(self):
        self.assertEqual(lg.decode_rates(footers([43.21])), [43.21])

    def test_a_generation_line_without_a_rate_is_not_a_footer(self):
        self.assertEqual(lg.decode_rates([NO_RATE]), [])

    def test_the_completion_counter_is_a_separate_channel(self):
        self.assertEqual(lg.completion_counts([COMPLETED.format(ct=512)]), [512])

    def test_a_line_without_a_completion_count_is_not_scraped(self):
        self.assertEqual(lg.completion_counts(footers([43.21])), [])

    def test_footers_are_kept_in_log_order(self):
        self.assertEqual(lg.decode_rates(footers([50.0, 40.0, 41.0])), [50.0, 40.0, 41.0])


class ReportTests(unittest.TestCase):
    """The row an arm prints, and whether it is allowed to print one."""

    def report(self, rates, cts=None):
        counts = cts if cts is not None else [512] * len(rates)
        return lg.arm_report("4bit-a_4Bit", rates, counts)

    def test_a_full_arm_is_a_mean_over_the_measured_runs_only(self):
        """Four footers, the first one the warm-up: the mean is 40.00, not 42.50,
        and the line names the number it dropped rather than hiding it."""
        lines, status = self.report([50.0, 40.0, 40.0, 40.0])
        self.assertEqual(status, 0, lines)
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("mean=40.00", lines[0])
        self.assertNotIn("mean=42.50", lines[0])
        self.assertIn("warm-up 50.00", lines[0])

    def test_the_measured_rates_are_the_three_after_the_warmup(self):
        lines, _ = self.report([50.0, 41.0, 42.0, 43.0])
        self.assertIn("[41.00, 42.00, 43.00]", lines[0])

    def test_the_completion_counts_drop_the_same_request_the_rates_drop(self):
        """Both columns on one line must describe the same requests, or the row
        reads as three runs' rate against four runs' tokens."""
        lines, _ = self.report([50.0, 40.0, 40.0, 40.0], [9, 11, 12, 13])
        self.assertIn("ct=[11, 12, 13]", lines[0])
        self.assertNotIn("9,", lines[0])

    def test_the_documented_protocol_is_one_warmup_and_three_measured_runs(self):
        self.assertEqual((lg.WARMUP_RUNS, lg.MEASURED_RUNS), (1, 3))
        self.assertEqual(lg.REQUESTS_PER_MODEL, 4)

    def test_a_short_footer_list_is_not_measured(self):
        """Three footers for four requests used to average three and print a row;
        the missing one is a run the mean does not contain, and nothing said so."""
        lines, status = self.report([50.0, 40.0, 40.0])
        self.assertEqual(status, 1, lines)
        self.assertTrue(all("mean=" not in line for line in lines), lines)
        self.assertIn("NOT MEASURED", lines[0])
        self.assertIn("3 decode footers", lines[0])

    def test_the_count_guard_names_the_requests_that_were_sent(self):
        lines, _ = self.report([50.0, 40.0, 40.0])
        self.assertIn("4 requests", lines[0])

    def test_an_extra_footer_is_not_measured(self):
        lines, status = self.report([50.0, 40.0, 40.0, 40.0, 39.0])
        self.assertEqual(status, 1, lines)
        self.assertIn("NOT MEASURED", lines[0])

    def test_a_footer_without_a_matching_completion_count_is_not_measured(self):
        """The two channels are scraped independently, so one of them drifting is
        exactly as silent as a dropped footer."""
        lines, status = self.report([50.0, 40.0, 40.0, 40.0], [11, 12])
        self.assertEqual(status, 1, lines)
        self.assertIn("2 completion counts", lines[0])

    def test_no_footer_at_all_is_not_measured(self):
        lines, status = self.report([])
        self.assertEqual(status, 1, lines)
        self.assertIn("no log line carried decode_tok_s=", lines[0])
        self.assertNotIn("mean=0.00", lines[0])


class ArgumentTests(unittest.TestCase):
    """Which installs the sweep runs, decided in `main()`, not at import."""

    def test_no_arguments_run_the_bench_model(self):
        with mock.patch.object(lg, "bench_model", return_value="models/env-choice"):
            self.assertEqual(lg.parse_args([]), ["models/env-choice"])

    def test_the_environment_reaches_the_default_and_not_the_shipped_path(self):
        with mock.patch.dict(os.environ, {"TINYTITAN_BENCH_MODEL": "/models/operator-choice"}):
            self.assertEqual(lg.parse_args([]), ["/models/operator-choice"])

    def test_named_models_are_kept_in_the_order_given(self):
        self.assertEqual(lg.parse_args(["models/a", "models/b"]), ["models/a", "models/b"])

    def test_an_empty_argument_is_refused(self):
        with self.assertRaises(lg.ConfigError):
            lg.parse_args([""])

    def test_a_flag_that_does_not_exist_is_refused(self):
        with self.assertRaises(lg.ConfigError):
            lg.parse_args(["--tokens", "512"])

    def test_the_refusal_names_the_argument_it_refuses(self):
        with self.assertRaises(lg.ConfigError) as caught:
            lg.parse_args(["-x"])
        self.assertIn("-x", str(caught.exception))


class DriverTests(unittest.TestCase):
    """`main()` end to end, with every process and socket faked.

    The fake server answers one footer per request, in the arm's own log, so a
    footer the engine would not have printed is expressible by giving an arm a
    shorter rate list than the number of requests it is sent.
    """

    def run_main(
        self,
        argv=(),
        rates=None,
        counts=None,
        healthy=True,
        bench_model=None,
    ):
        """(status, output, spawned, sent, logs, health_kwargs)."""
        rates = rates if rates is not None else {}
        counts = counts if counts is not None else {}
        tmp = pathlib.Path(tempfile.mkdtemp())
        state = {
            "active": None,
            "current": None,
            "cursor": 0,
            "spawned": [],
            "sent": [],
            "logs": [],
            "health": [],
        }

        class FakeProc:
            def __init__(self, command, **kwargs):
                self.stopped = False

            def poll(self):
                return None

            def terminate(self):
                self.stopped = True

            def wait(self, timeout=None):
                return 0

            def kill(self):
                pass

        def fake_command(*args, **kwargs):
            state["spawned"].append(kwargs)
            state["current"] = kwargs.get("model")
            state["cursor"] = 0
            return ["fake-server", *args[1:]]

        def fake_log_path(name):
            state["logs"].append(name)
            path = tmp / name
            path.touch()
            state["active"] = str(path)
            return str(path)

        def fake_env(*args, **kwargs):
            return {}

        def fake_health(proc, port, **kwargs):
            state["health"].append({"port": port, **kwargs})
            return healthy

        def fake_request(port, payload):
            state["sent"].append((port, payload))
            arm = state["current"]
            queue = rates.get(arm, ())
            if state["cursor"] < len(queue):
                with open(state["active"], "a", encoding="utf-8") as handle:
                    handle.write(GEN.format(rate=queue[state["cursor"]]) + "\n")
                    count = counts.get(arm)
                    if count is None or state["cursor"] < len(count):
                        ct = 512 if count is None else count[state["cursor"]]
                        handle.write(COMPLETED.format(ct=ct) + "\n")
            state["cursor"] += 1

        def fake_api_model(port, **kwargs):
            return "fake-model"

        def fake_bench_model():
            return bench_model

        original_argv = sys.argv
        sys.argv = ["tinytitan_longgen.py", *argv]
        buffer = io.StringIO()

        patches = [
            mock.patch.object(lg.subprocess, "Popen", new=FakeProc),
            mock.patch.object(lg, "server_command", new=fake_command),
            mock.patch.object(lg, "server_environment", new=fake_env),
            mock.patch.object(lg, "benchmark_log_path", new=fake_log_path),
            mock.patch.object(lg, "wait_for_health", new=fake_health),
            mock.patch.object(lg, "request", new=fake_request),
            mock.patch.object(lg, "resolve_api_model", new=fake_api_model),
            mock.patch.object(lg, "FLUSH_SETTLE_SECONDS", 0),
            mock.patch.object(lg, "REQUEST_SETTLE_SECONDS", 0),
        ]
        if bench_model is not None:
            patches.append(mock.patch.object(lg, "bench_model", new=fake_bench_model))
            patches.append(
                mock.patch.object(tinytitan_profile, "bench_model", new=fake_bench_model)
            )
        try:
            with contextlib.ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                    try:
                        status = lg.main()
                    except SystemExit as exc:
                        status = f"SystemExit({exc.code})"
        finally:
            sys.argv = original_argv
        return (
            status,
            buffer.getvalue(),
            state["spawned"],
            state["sent"],
            state["logs"],
            state["health"],
        )

    def test_a_measured_arm_exits_zero(self):
        status, output, spawned = self.run_main(
            ["models/a_4Bit"], rates={"models/a_4Bit": [50.0, 40.0, 40.0, 40.0]}
        )[:3]
        self.assertEqual(status, 0, output)
        self.assertEqual(len(spawned), 1)
        self.assertIn("mean=40.00", output)

    def test_the_request_count_is_one_warmup_and_three_measured_runs(self):
        sent = self.run_main(["models/a_4Bit"], rates={"models/a_4Bit": [50.0, 40.0, 40.0, 40.0]})[
            3
        ]
        self.assertEqual(len(sent), lg.REQUESTS_PER_MODEL)

    def test_the_long_generation_prompt_is_the_one_the_docstring_names(self):
        """The arm exists to hold a 512-token decode going; a short prompt would
        measure a different run and the printed line would still say 512."""
        sent = self.run_main(["models/a_4Bit"], rates={"models/a_4Bit": [50.0, 40.0, 40.0, 40.0]})[
            3
        ]
        body = sent[0][1]
        self.assertIn(b"Levenshtein", body)
        self.assertIn(b'"max_completion_tokens": 512', body)
        self.assertIn(b'"temperature": 0', body)
        self.assertIn(b'"stream": true', body)

    def test_a_server_that_never_answers_health_sends_no_request(self):
        """The inlined wait expired *into* the run; a sweep must not POST a
        512-token generation at a server that never said it was ready."""
        status, output, _, sent, _, _ = self.run_main(["models/a_4Bit"], healthy=False)
        self.assertEqual(status, 1, output)
        self.assertEqual(sent, [], output)
        self.assertIn("ARM FAILED", output)

    def test_the_reason_names_the_dead_process_and_not_a_timeout(self):
        """`wait_for_health()` returns False for exactly one thing: the process
        exited. A slow-but-alive load returns True, so a message blaming the
        timeout sends the operator to the wrong cause."""
        _, output, _, _, _, _ = self.run_main(["models/a_4Bit"], healthy=False)
        self.assertIn("exited before /health answered", output)
        self.assertNotIn("never answered /health within", output)

    def test_the_health_budget_is_the_server_load_timeout_and_reaches_the_wait(self):
        """A long-generation arm loads the install behind that wait, and the
        profile's 120 s default is the budget that timed out into the run."""
        health = self.run_main(
            ["models/a_4Bit"], rates={"models/a_4Bit": [50.0, 40.0, 40.0, 40.0]}
        )[5]
        self.assertGreaterEqual(lg.SERVER_LOAD_TIMEOUT, 600)
        self.assertEqual(health[0]["timeout"], lg.SERVER_LOAD_TIMEOUT)
        self.assertEqual(health[0]["port"], lg.PORT)

    def test_a_footerless_arm_is_not_reported_as_a_zero_rate(self):
        status, output = self.run_main(["models/a_4Bit"], rates={})[:2]
        self.assertEqual(status, 1, output)
        self.assertIn("NOT MEASURED", output)
        self.assertNotIn("mean=0.00", output)

    def test_a_short_footer_list_is_not_reported_as_a_mean(self):
        status, output = self.run_main(
            ["models/a_4Bit"], rates={"models/a_4Bit": [50.0, 40.0, 40.0]}
        )[:2]
        self.assertEqual(status, 1, output)
        self.assertNotIn("mean=", output)

    def test_one_failed_arm_does_not_end_the_sweep(self):
        """The arms *are* the comparison; the second install still has to be tried
        when the first one's server dies."""
        status, output, spawned = self.run_main(
            ["models/a_4Bit", "models/b_4Bit"],
            rates={"models/b_4Bit": [50.0, 40.0, 40.0, 40.0]},
            healthy=False,
        )[:3]
        self.assertEqual(status, 1, output)
        self.assertIn("ARM FAILED", output)
        self.assertEqual(len(spawned), 2, output)

    def test_a_sweep_of_measured_arms_exits_zero(self):
        status, output, _, _, logs, _ = self.run_main(
            ["models/a_4Bit", "models/b_4Bit"],
            rates={
                "models/a_4Bit": [50.0, 40.0, 40.0, 40.0],
                "models/b_4Bit": [50.0, 41.0, 41.0, 41.0],
            },
        )
        self.assertEqual(status, 0, output)
        self.assertEqual(len(set(logs)), 2, logs)
        self.assertIn("mean=40.00", output)
        self.assertIn("mean=41.00", output)

    def test_two_same_width_installs_never_read_the_same_log(self):
        """Both arms run against one fixture log per arm; if they shared a file the
        second `w` would truncate the first arm's capture before it was read."""
        _, output, _, _, logs, _ = self.run_main(
            ["models/a_4Bit", "models/b_4Bit"],
            rates={
                "models/a_4Bit": [50.0, 40.0, 40.0, 40.0],
                "models/b_4Bit": [50.0, 41.0, 41.0, 41.0],
            },
        )
        self.assertEqual(len(logs), 2, output)
        self.assertEqual(len(set(logs)), 2, logs)

    def test_the_environment_choice_reaches_the_server_command(self):
        """`MODEL` was read from `DEFAULT_MODEL_PATH` directly, so the operator's
        `TINYTITAN_BENCH_MODEL` selected nothing but the label."""
        spawned = self.run_main(
            [],
            bench_model="/models/operator-choice",
            rates={"/models/operator-choice": [50.0, 40.0, 40.0, 40.0]},
        )[2]
        self.assertEqual(spawned[0]["model"], "/models/operator-choice")

    def test_the_row_names_the_install_and_not_only_the_width(self):
        status, output = self.run_main(
            ["models/a_4Bit"], rates={"models/a_4Bit": [50.0, 40.0, 40.0, 40.0]}
        )[:2]
        self.assertIn("a_4Bit", output, status)

    def test_a_refused_configuration_exits_two_and_prints_the_usage(self):
        status, output = self.run_main(["--tokens", "512"])[:2]
        self.assertEqual(status, 2, output)
        self.assertIn("REFUSED", output)
        self.assertIn(lg.USAGE, output)

    def test_the_completion_column_comes_from_the_arm_under_test(self):
        counts = self.run_main(
            ["models/a_4Bit", "models/b_4Bit"],
            rates={
                "models/a_4Bit": [50.0, 40.0, 40.0, 40.0],
                "models/b_4Bit": [50.0, 41.0, 41.0, 41.0],
            },
            counts={
                "models/a_4Bit": [1, 2, 3, 4],
                "models/b_4Bit": [5, 6, 7, 8],
            },
        )[1]
        self.assertIn("ct=[2, 3, 4]", counts)
        self.assertIn("ct=[6, 7, 8]", counts)


class ImportTests(unittest.TestCase):
    """The module must be importable before it is configured."""

    def test_importing_the_module_starts_no_server_and_opens_no_log(self):
        program = (
            "import sys, builtins, http.client, subprocess\n"
            "sys.path.insert(0, " + repr(str(ROOT / "benchmark")) + ")\n"
            "def guard(*a, **k):\n"
            "    raise RuntimeError('a side effect ran during import')\n"
            "subprocess.Popen = guard\n"
            "http.client.HTTPConnection = guard\n"
            "real_open = builtins.open\n"
            "def no_log(name, *a, **k):\n"
            "    if 'benchmark-logs' in str(name):\n"
            "        raise RuntimeError('opened a log at import: ' + str(name))\n"
            "    return real_open(name, *a, **k)\n"
            "builtins.open = no_log\n"
            "import tinytitan_longgen\n"
            "print('imported clean')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            cwd=str(ROOT / "benchmark"),
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("imported clean", result.stdout)


if __name__ == "__main__":
    unittest.main()
