"""Gates `benchmark/tinytitan_rdadvise_ab.py`'s arms, verdict and exit status.

The driver exists to answer one question — does expert read-ahead advice change
the overlap counters — by running the same request twice, once with the engine's
default policy and once with `off`. Three things made that answer unreadable, all
of them the same seam AUD-218, AUD-223, AUD-224 and AUD-225 were filed on.

1. It ran at *import*: `for mode in ("default", "off"): run(mode)` sat at module
   scope, and `run()` opens a log and `Popen`s a `TinyTitanServer`, so importing
   the module started a model process — two of them, in fact, and it is also why
   the file had no test.
2. It printed `--- {mode} ---` over whatever the capture held, with no line count
   and no refusal, and the file's only `sys.exit` was the one for a server that
   died before `/health`. An A/B in which neither arm reached decode therefore
   read as two clean measurements and exited 0.
3. The arm labelled `default` did not necessarily measure the default. The engine
   reads `TINYTITAN_RDADVISE_POLICY` from the process environment *ahead of* the
   model's shipped value (`ServerModelSession+Loading.swift:198`), and the driver
   built its env from `os.environ`, so an operator who had exported a policy got
   `override vs off` on a page that said `default vs off` — and the page named no
   policy, model or length, so the reader could not see it.

No model, no server and no GPU is involved in any of this: `Popen`, the health
wait, the request loop, `server_command()` (which reads the install manifest) and
the log path are all faked, and the driver's logs are fed fixture lines in the
shape the Swift engine prints them. The A/B numbers themselves remain the
operator's to schedule.

    cd benchmark && python3 -m unittest test_tinytitan_rdadvise_ab -v
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
    "tinytitan_rdadvise_ab", ROOT / "benchmark" / "tinytitan_rdadvise_ab.py"
)
ra = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ra)

GEN = "TinyTitan generation prefill_s=4.912 decode_s=11.900 decode_tok_s=43.210"
RUNNER = "TinyTitan runner cb1_ms=1.210 io_ms=0.440 cb2_ms=0.980"
KERNEL_ROLE = "TinyTitan kernel role=gemv gpu_ms=0.310 per_token_ms=0.021 count=64"
KERNEL_TOTAL = "TinyTitan kernel total_gpu_ms=12.700 gpu_share_of_decode=10.7%"
BUSY = "TinyTitan kernel busy_ms=9.800 span_ms=11.200"

FULL_LOG = [GEN, GEN, RUNNER, RUNNER, KERNEL_TOTAL]
POLICY = "TINYTITAN_RDADVISE_POLICY"


class CaptureTests(unittest.TestCase):
    """The classification the refactor must not have changed."""

    def test_each_line_lands_in_the_section_the_prose_names(self):
        gen, runner, gpu = ra.capture(FULL_LOG)
        self.assertEqual(gen, [GEN, GEN])
        self.assertEqual(runner, [RUNNER, RUNNER])
        self.assertEqual(gpu, [KERNEL_TOTAL])

    def test_a_generation_line_without_a_decode_rate_is_not_a_footer(self):
        """`decode_tok_s=` is the half that carries the measurement; a bare
        `TinyTitan generation` line is a progress print."""
        gen, _, _ = ra.capture(["TinyTitan generation started"])
        self.assertEqual(gen, [])

    def test_the_per_role_kernel_lines_are_not_the_arm_total(self):
        """The docstring promises the overlap counters, and this driver reads the
        per-request total. `role=` lines belong to the overlap driver."""
        _, _, gpu = ra.capture([KERNEL_ROLE])
        self.assertEqual(gpu, [])

    def test_the_busy_and_span_channel_reaches_no_section(self):
        """Pinned as scope, not as a passing claim: `busy_ms=… span_ms=…` is the
        line that names the overlap directly and no section reads it, so this
        driver's gpu columns are a share of decode, never the overlap itself."""
        gen, runner, gpu = ra.capture([BUSY])
        self.assertEqual((gen, runner, gpu), ([], [], []))


class ArmEnvironmentTests(unittest.TestCase):
    """What each arm actually asks the engine to do."""

    def test_both_arms_ask_for_the_two_stat_channels(self):
        for mode in ra.MODES:
            env = ra.arm_environment(mode, base={})
            self.assertEqual(env["TINYTITAN_RUNNER_STATS"], "1", mode)
            self.assertEqual(env["TINYTITAN_KERNEL_STATS"], "1", mode)

    def test_the_off_arm_names_the_policy_off(self):
        self.assertEqual(ra.arm_environment("off", base={})[POLICY], "off")

    def test_an_inherited_policy_does_not_become_the_default_arm(self):
        """The engine reads `TINYTITAN_RDADVISE_POLICY` from the environment ahead
        of the model's shipped value, so an exported override would make the arm
        labelled `default` measure something else entirely."""
        env = ra.arm_environment("default", base={POLICY: "sequential"})
        self.assertNotIn(POLICY, env)

    def test_the_engine_defaults_the_arm_leaves_untouched_are_still_passed_through(self):
        env = ra.arm_environment("default", base={"TINYTITAN_THINKING_MODE": "off"})
        self.assertEqual(env["TINYTITAN_THINKING_MODE"], "off")

    def test_an_unknown_arm_is_refused_rather_than_run_as_default(self):
        with self.assertRaises(ValueError):
            ra.arm_environment("sequential", base={})

    def test_the_bench_model_env_override_is_honoured(self):
        """Three sibling drivers read `TINYTITAN_BENCH_MODEL`; this one hardcoded
        the 8-bit default, so an operator sweeping their own install got the
        default under a label that named nothing."""
        with mock.patch.dict(os.environ, {"TINYTITAN_BENCH_MODEL": "/models/some-125b"}):
            self.assertEqual(ra.bench_model(), "/models/some-125b")

    def test_with_no_override_the_default_install_is_named(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(ra.bench_model(), str(tinytitan_profile.DEFAULT_MODEL_PATH))


class VerdictTests(unittest.TestCase):
    """The page and the status, with no server involved."""

    def verdict(self, arms):
        lines, status = ra.verdict(arms)
        return "\n".join(lines), status

    def test_a_full_pair_of_arms_prints_both_and_exits_zero(self):
        output, status = self.verdict(
            [
                ("default", ra.capture(FULL_LOG)),
                ("off", ra.capture(FULL_LOG)),
            ]
        )
        self.assertEqual(status, 0, output)
        self.assertNotIn("NOT MEASURED", output)
        self.assertIn("--- default", output)
        self.assertIn("--- off", output)

    def test_an_arm_that_captured_nothing_is_named_and_fails(self):
        output, status = self.verdict([("default", ([], [], [])), ("off", ra.capture(FULL_LOG))])
        self.assertEqual(status, 1, output)
        self.assertIn("NOT MEASURED: default", output)

    def test_every_empty_channel_is_named_with_the_string_it_wanted(self):
        output, _ = self.verdict([("off", ([], [], []))])
        self.assertEqual(output.count("NOT MEASURED"), 3, output)
        for marker in ("decode_tok_s=", "cb1_ms=", "total_gpu_ms="):
            self.assertIn(marker, output)

    def test_the_header_carries_its_own_line_count(self):
        output, _ = self.verdict([("default", ra.capture(FULL_LOG))])
        self.assertIn("gen 2", output)
        self.assertIn("gpu 1", output)

    def test_a_shape_mismatch_is_not_a_measurement(self):
        """An arm that ran but whose log the parser matched nowhere is the
        instrument unplugged, so `([], [], [])` must fail even as the only arm."""
        _, status = self.verdict([("default", ([], [], []))])
        self.assertEqual(status, 1)


class DriverTests(unittest.TestCase):
    """`main()` end to end, with every process and socket faked."""

    def run_main(self, argv=(), log_lines=None, healthy=True, inherited_policy=None):
        """(status, output, spawned) driving the real `main()`.

        The fake server writes `log_lines` into the log the driver opens when it
        is terminated, so the capture path is the driver's own; `spawned` records
        the argv/env of every `Popen` the driver attempted.
        """
        import tempfile

        tmp = tempfile.mkdtemp()
        log = pathlib.Path(tmp) / "server.log"
        spawned = []
        events = []

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
        sys.argv = ["tinytitan_rdadvise_ab.py", *argv]
        env_patch = {}
        if inherited_policy is not None:
            env_patch[POLICY] = inherited_policy
        try:

            def fake_health(proc, *args, **kwargs):
                return healthy

            def fake_requests(*args, **kwargs):
                return None

            def fake_command(*args, **kwargs):
                return ["fake-server", kwargs.get("model")]

            def fake_model():
                return "/models/fake"

            def fake_log_path(name):
                return str(log)

            with (
                mock.patch.object(ra.subprocess, "Popen", new=FakeProc),
                mock.patch.object(ra, "wait_for_health", new=fake_health),
                mock.patch.object(ra, "request_twice", new=fake_requests),
                mock.patch.object(ra, "server_command", new=fake_command),
                mock.patch.object(ra, "bench_model", new=fake_model),
                mock.patch.object(ra, "benchmark_log_path", new=fake_log_path),
                mock.patch.object(ra, "FLUSH_SETTLE_SECONDS", 0),
                mock.patch.dict(os.environ, env_patch),
                contextlib.redirect_stdout(buffer),
                contextlib.redirect_stderr(buffer),
            ):
                status = ra.main()
        finally:
            sys.argv = original_argv
        return status, buffer.getvalue(), spawned, events

    def test_a_full_pair_of_arms_is_a_measurement_and_exits_zero(self):
        status, output, spawned, events = self.run_main(log_lines=FULL_LOG)
        self.assertEqual(status, 0, output)
        self.assertIn("--- default", output)
        self.assertIn("--- off", output)
        self.assertEqual(len(spawned), 2)

    def test_the_arms_are_two_fresh_servers_run_one_after_another(self):
        """`spawn, terminate, spawn, terminate` — the docstring's "fresh servers"
        means one live model process at a time, and a second spawn before the
        first arm was reaped would put two of them on the machine."""
        _, _, spawned, events = self.run_main(log_lines=FULL_LOG)
        self.assertEqual(len(spawned), 2)
        self.assertEqual(events, ["spawn", "terminate", "spawn", "terminate"])

    def test_the_default_arm_clears_a_policy_the_operator_exported(self):
        status, output, spawned, events = self.run_main(
            log_lines=FULL_LOG, inherited_policy="sequential"
        )
        self.assertEqual(status, 0, output)
        self.assertNotIn(POLICY, spawned[0][1])
        self.assertEqual(spawned[1][1][POLICY], "off")

    def test_an_arm_that_captured_nothing_fails_the_run(self):
        """The headline: with no capture at all the old driver printed two headers
        and exited 0."""
        status, output, _, _ = self.run_main(log_lines=[])
        self.assertEqual(status, 1, output)
        self.assertIn("NOT MEASURED", output)

    def test_the_page_names_the_model_and_the_length_it_ran(self):
        status, output, _, _ = self.run_main(log_lines=FULL_LOG)
        self.assertEqual(status, 0, output)
        self.assertIn("/models/fake", output)
        self.assertIn("512", output)

    def test_the_named_model_is_the_one_handed_to_the_launcher(self):
        """`bench_model()` only fixes anything if it reaches `server_command`: a
        driver that reads the override and then passes the shipped path prints the
        operator's model while launching someone else's."""
        status, output, spawned, _ = self.run_main(log_lines=FULL_LOG)
        self.assertEqual(status, 0, output)
        for command, _env in spawned:
            self.assertEqual(command, ["fake-server", "/models/fake"])

    def test_an_argument_is_refused_rather_than_ignored(self):
        status, output, spawned, events = self.run_main(argv=["1024"])
        self.assertEqual(status, 2, output)
        self.assertEqual(spawned, [])
        self.assertIn("usage", output.lower())

    def test_a_server_that_never_becomes_healthy_is_reported_per_arm(self):
        status, output, _, _ = self.run_main(log_lines=FULL_LOG, healthy=False)
        self.assertEqual(status, 1, output)
        self.assertNotIn("--- default", output)


class ImportTests(unittest.TestCase):
    """The side effect that made this file untestable, guarded rather than run."""

    def test_importing_the_module_starts_no_server_and_opens_no_log(self):
        """A child guards `subprocess.Popen` and `builtins.open` before importing,
        so the mutant that restores the module-scope loop trips the guard instead
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
import tinytitan_rdadvise_ab  # noqa: F401
print("imported clean")
"""
        completed = subprocess.run(
            [sys.executable, "-c", child, str(ROOT / "benchmark" / "tinytitan_rdadvise_ab.py")],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("imported clean", completed.stdout)


if __name__ == "__main__":
    unittest.main()
