"""Gates `benchmark/tinytitan_gap_bisect.py`'s arms, verdict and exit status.

The driver answers one question — how much of the wall is fixed per request and
how much is per token — by running the same prompt at 128 tokens and at 1024 on
two fresh servers and reading the server's own footers. Four things made that
answer unreadable, and they are the seam AUD-225, AUD-226 and AUD-227 were
filed on:

1. It ran at *import*: `run(128, "len128")` and `run(1024, "len1024")` sat at
   module scope, and `run()` opens a log and `Popen`s a `TinyTitanServer`, so
   importing the module started **two** model processes. It is also why the file
   had no test.
2. It printed `--- {tag} (max_tokens=N) ---` over whatever the capture held, with
   no line count and no refusal, so a sweep in which neither server reported a
   single counter read as two clean measurements and exited 0.
3. Its only exit was `sys.exit(1)` *inside* `run()`, for a server that died before
   `/health`. One dead arm ended the whole sweep, and the surviving arm was never
   reported — the reader could not tell which of the two lengths failed.
4. The page it writes promised more than the code delivered: the docstring said
   the run was a length sweep "and an rdadvise-off comparison", but the driver
   never passes `extra_env`, never sets `TINYTITAN_RDADVISE_POLICY`, and its two
   tags are `len128` and `len1024`. The policy A/B is
   `tinytitan_rdadvise_ab.py`'s job; this driver measures one control, the
   request length, and now says so.

No model, no server and no GPU is involved in any of this: `Popen`, the health
wait, the request loop, `server_command()` (which reads the install manifest) and
the log path are all faked, and the fake server is fed fixture lines in the shape
the Swift engine prints them. The gap numbers themselves remain the operator's to
schedule.

    cd benchmark && python3 -m unittest test_tinytitan_gap_bisect -v
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import pathlib
import subprocess
import sys
import unittest
from unittest import mock

import tinytitan_profile

ROOT = pathlib.Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "tinytitan_gap_bisect", ROOT / "benchmark" / "tinytitan_gap_bisect.py"
)
gb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gb)

GEN = "TinyTitan generation prefill_s=4.912 decode_s=11.900 decode_tok_s=43.210"
GEN_NO_RATE = "TinyTitan generation started"
RUNNER = "TinyTitan runner cb1_ms=1.210 io_ms=0.440 cb2_ms=0.980"
KERNEL_ROLE = "TinyTitan kernel role=gemv gpu_ms=0.310 per_token_ms=0.021 count=64"
KERNEL_TOTAL = "TinyTitan kernel total_gpu_ms=12.700 gpu_share_of_decode=10.7%"
BUSY = "TinyTitan kernel busy_ms=9.800 span_ms=11.200"

FULL_LOG = [GEN, GEN, RUNNER, RUNNER, KERNEL_ROLE, KERNEL_TOTAL]
POLICY = "TINYTITAN_RDADVISE_POLICY"
EMPTY = ([], [], [], [])


class CaptureTests(unittest.TestCase):
    """Which footer line belongs to which channel of the sweep."""

    def test_each_line_lands_in_the_section_the_prose_names(self):
        gen, runner, roles, total = gb.capture(FULL_LOG)
        self.assertEqual(gen, [GEN, GEN])
        self.assertEqual(runner, [RUNNER, RUNNER])
        self.assertEqual(roles, [KERNEL_ROLE])
        self.assertEqual(total, [KERNEL_TOTAL])

    def test_a_generation_line_without_a_decode_rate_is_not_a_footer(self):
        """`decode_tok_s=` is the half that carries the measurement; a bare
        `TinyTitan generation` line is a progress print."""
        gen, _, _, _ = gb.capture([GEN_NO_RATE])
        self.assertEqual(gen, [])

    def test_the_per_role_table_and_the_total_are_different_channels(self):
        """The length sweep is read off the per-role `per_token_ms=` column, and
        the total is the denominator; one without the other is not the sweep."""
        _, _, roles, total = gb.capture([KERNEL_ROLE])
        self.assertEqual(len(roles), 1)
        self.assertEqual(total, [])

    def test_the_busy_and_span_channel_reaches_no_section(self):
        """Pinned as scope: the driver never claimed that channel and must not
        quietly start counting it as one of its four."""
        _, _, roles, total = gb.capture([BUSY])
        self.assertEqual(roles, [])
        self.assertEqual(total, [])


class ArmTests(unittest.TestCase):
    """What the two arms share, and what the page may therefore say about them."""

    def test_the_bench_model_env_override_is_honoured(self):
        """The three sibling sweeps name their model through
        `TINYTITAN_BENCH_MODEL`; this one hardcoded the shipped path, so an
        operator pointing the tree at their own checkpoint got the default under a
        label that named nothing."""
        with mock.patch.dict(os.environ, {"TINYTITAN_BENCH_MODEL": "/models/some-125b"}):
            self.assertEqual(gb.bench_model(), "/models/some-125b")

    def test_the_fallback_is_the_shipped_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(gb.bench_model(), str(tinytitan_profile.DEFAULT_MODEL_PATH))

    def test_both_arms_turn_both_stat_channels_on(self):
        """`per_token_ms=` comes from the kernel stats and `cb1_ms=` from the
        runner stats; without either the sweep has nothing to bisect."""
        for _tokens, tag in gb.LEN_ARMS:
            env = gb.arm_environment(tag)
            self.assertEqual(env["TINYTITAN_KERNEL_STATS"], "1")
            self.assertEqual(env["TINYTITAN_RUNNER_STATS"], "1")

    def test_the_driver_never_sets_the_policy_it_says_it_does_not_run(self):
        """The docstring used to advertise an rdadvise-off comparison. The length
        sweep runs one control — the request length — so neither arm may touch the
        read-ahead policy."""
        for _, tag in gb.LEN_ARMS:
            self.assertNotIn(POLICY, gb.arm_environment(tag))

    def test_an_unknown_arm_is_refused_not_run_as_a_third_setting(self):
        with self.assertRaises(ValueError):
            gb.arm_environment("len2048")


class VerdictTests(unittest.TestCase):
    """The page and the status, with no server involved."""

    def verdict(self, arms):
        lines, status = gb.verdict(arms)
        return "\n".join(lines), status

    def test_a_full_pair_of_arms_prints_both_and_exits_zero(self):
        output, status = self.verdict([(tag, gb.capture(FULL_LOG)) for _, tag in gb.LEN_ARMS])
        self.assertEqual(status, 0, output)
        self.assertNotIn("NOT MEASURED", output)
        self.assertIn("--- len128", output)
        self.assertIn("--- len1024", output)

    def test_the_header_carries_its_own_line_count(self):
        output, _ = self.verdict([("len128", gb.capture(FULL_LOG))])
        self.assertIn("gen 2", output)
        self.assertIn("kernel roles 1", output)
        self.assertIn("kernel total 1", output)

    def test_an_arm_that_captured_nothing_is_named_and_fails(self):
        output, status = self.verdict([("len128", EMPTY), ("len1024", gb.capture(FULL_LOG))])
        self.assertEqual(status, 1, output)
        self.assertIn("NOT MEASURED: len128", output)
        self.assertNotIn("NOT MEASURED: len1024", output)

    def test_every_empty_channel_is_named_with_the_string_it_wanted(self):
        output, _ = self.verdict([("len128", EMPTY)])
        self.assertEqual(output.count("NOT MEASURED"), 4, output)
        for marker in ("decode_tok_s=", "cb1_ms=", "TinyTitan kernel role=", "total_gpu_ms="):
            self.assertIn(marker, output)

    def test_a_dead_arm_is_reported_rather_than_ending_the_sweep(self):
        """The old `sys.exit(1)` inside `run()` meant a server that died before
        `/health` took the other arm down with it, unread."""
        output, status = self.verdict([("len128", None), ("len1024", gb.capture(FULL_LOG))])
        self.assertEqual(status, 1, output)
        self.assertIn("ARM FAILED: len128", output)
        self.assertIn("--- len1024", output)


class DriverTests(unittest.TestCase):
    """`main()` end to end, with every process and socket faked."""

    def run_main(self, argv=(), log_lines=None, healthy=True):
        """(status, output, spawned, events) driving the real `main()`.

        The fake server writes `log_lines` into the log the driver opens when it is
        terminated, so the capture path is the driver's own; `spawned` records the
        argv/env of every `Popen` attempted and `sent` the requests the driver made.
        """
        import tempfile

        tmp = tempfile.mkdtemp()
        log = pathlib.Path(tmp) / "server.log"
        spawned = []
        events = []
        sent = []

        class FakeProc:
            def __init__(self, command, **kwargs):
                spawned.append((command, kwargs.get("env") or {}))
                events.append("spawn")
                self.stopped = False
                self.stream = open(log, "w", encoding="utf-8")

            def poll(self):
                return None

            def terminate(self):
                if self.stopped:
                    return
                events.append("terminate")
                self.stopped = True
                for line in log_lines or []:
                    self.stream.write(line + "\n")
                self.stream.flush()

            def wait(self, timeout=None):
                return 0

            def kill(self):
                self.terminate()

        buffer = io.StringIO()
        original_argv = sys.argv
        sys.argv = ["tinytitan_gap_bisect.py", *argv]
        try:

            def fake_health(proc, *args, **kwargs):
                return healthy

            def fake_requests(prompt, max_tokens, port):
                sent.append((prompt, max_tokens, port))

            def fake_command(*args, **kwargs):
                return ["fake-server", kwargs.get("model")]

            def fake_model():
                return "/models/fake"

            def fake_log_path(name):
                return str(log)

            with (
                mock.patch.object(gb.subprocess, "Popen", new=FakeProc),
                mock.patch.object(gb, "wait_for_health", new=fake_health),
                mock.patch.object(gb, "request_twice", new=fake_requests),
                mock.patch.object(gb, "server_command", new=fake_command),
                mock.patch.object(gb, "bench_model", new=fake_model),
                mock.patch.object(gb, "benchmark_log_path", new=fake_log_path),
                mock.patch.object(gb, "FLUSH_SETTLE_SECONDS", 0),
                contextlib.redirect_stdout(buffer),
                contextlib.redirect_stderr(buffer),
            ):
                status = gb.main()
        finally:
            sys.argv = original_argv
        return status, buffer.getvalue(), spawned, events, sent

    def test_a_full_pair_of_arms_is_a_measurement_and_exits_zero(self):
        status, output, spawned, _, _ = self.run_main(log_lines=FULL_LOG)
        self.assertEqual(status, 0, output)
        self.assertIn("--- len128", output)
        self.assertIn("--- len1024", output)
        self.assertEqual(len(spawned), 2)

    def test_the_arms_are_two_fresh_servers_run_one_after_another(self):
        """`spawn, terminate, spawn, terminate` — the second length starts only
        once the first server is reaped, so the machine never holds two."""
        _, _, spawned, events, _ = self.run_main(log_lines=FULL_LOG)
        self.assertEqual(len(spawned), 2)
        self.assertEqual(events, ["spawn", "terminate", "spawn", "terminate"])

    def test_each_arm_requests_its_own_length(self):
        """The one control the sweep varies. A driver that sent the same length to
        both servers would print two identical rows and call it a bisect."""
        _, _, _, _, sent = self.run_main(log_lines=FULL_LOG)
        self.assertEqual(sent, [(gb.PROMPT, 128, gb.PORT), (gb.PROMPT, 1024, gb.PORT)])

    def test_an_arm_that_captured_nothing_fails_the_run(self):
        """The headline: with no capture at all the old driver printed two headers
        and exited 0."""
        status, output, _, _, _ = self.run_main(log_lines=[])
        self.assertEqual(status, 1, output)
        self.assertIn("NOT MEASURED", output)

    def test_a_server_that_never_becomes_healthy_is_reported_per_arm(self):
        status, output, spawned, _, _ = self.run_main(log_lines=FULL_LOG, healthy=False)
        self.assertEqual(status, 1, output)
        self.assertIn("ARM FAILED: len128", output)
        self.assertIn("ARM FAILED: len1024", output)
        self.assertEqual(len(spawned), 2)

    def test_the_page_names_the_model_and_both_lengths(self):
        status, output, _, _, _ = self.run_main(log_lines=FULL_LOG)
        self.assertEqual(status, 0, output)
        self.assertIn("/models/fake", output)
        self.assertIn("128", output)
        self.assertIn("1024", output)

    def test_the_named_model_is_the_one_handed_to_the_launcher(self):
        """`bench_model()` only fixes anything if it reaches `server_command`: a
        driver that reads the override and then passes the shipped path prints the
        operator's model while launching someone else's."""
        status, output, spawned, _, _ = self.run_main(log_lines=FULL_LOG)
        self.assertEqual(status, 0, output)
        for command, _env in spawned:
            self.assertEqual(command, ["fake-server", "/models/fake"])

    def test_an_argument_is_refused_rather_than_ignored(self):
        status, output, spawned, _, _ = self.run_main(argv=["256"])
        self.assertEqual(status, 2, output)
        self.assertEqual(spawned, [])
        self.assertIn("usage", output.lower())


class ImportTests(unittest.TestCase):
    """The side effect that made this file untestable, guarded rather than run."""

    def test_importing_the_module_starts_no_server_and_opens_no_log(self):
        """A child guards `subprocess.Popen` and `builtins.open` before importing,
        so the mutant that restores the module-scope calls trips the guard instead
        of loading a model — and the same child run against this file proves the
        guard is not hollow."""
        child = r"""
import builtins, pathlib, subprocess, sys

target = sys.argv[1]

def boom_popen(*args, **kwargs):
    raise RuntimeError("Popen ran at import: %r" % (args[0],))

real_open = builtins.open
def boom_open(file, *args, **kwargs):
    name = str(file)
    if name.endswith(".log"):
        raise RuntimeError("opened a log at import: %s" % name)
    return real_open(file, *args, **kwargs)

subprocess.Popen = boom_popen
builtins.open = boom_open
sys.path.insert(0, str(pathlib.Path(target).parent))
import tinytitan_gap_bisect  # noqa: F401
print("imported clean")
"""
        completed = subprocess.run(
            [sys.executable, "-c", child, str(ROOT / "benchmark" / "tinytitan_gap_bisect.py")],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("imported clean", completed.stdout)


if __name__ == "__main__":
    unittest.main()
