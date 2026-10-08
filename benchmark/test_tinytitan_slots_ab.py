"""Gates `benchmark/tinytitan_slots_ab.py`'s arms, verdict and exit status.

The driver answers one question — how many expert-cache slots does the 8 GB RAM
budget need — by booting one server per slot count and reading the server's own
counters. Four things made that answer unreadable:

1. `run()` printed `--- slots=32 pin=None boot_s=... ---` and then whatever its
   three captures held, with no line count and no refusal. Measured on the file
   before the fix: a server that answered `/health` and printed no counter at all
   produced exactly one line — the header — and the process exited 0.
2. Its only exit was `sys.exit(1)` *inside* `run()`, for a server that died before
   `/health`. Measured: with `[32, 128]` queued, the first dead arm printed
   `server exited early` and the sweep stopped — the surviving arm was never
   reported, so the reader could not tell which slot count failed to load.
3. `main()` returns nothing and the `__main__` guard calls `main()` and discards
   it, so even the arms it did notice were an exit code nobody read.
4. `MODEL` and `MAX_TOKENS` were read from the environment at module scope, so the
   configuration was frozen before `main()` ran and an unparseable
   `TINYTITAN_AB_TOKENS` was a traceback rather than a refusal.

No model, no server and no GPU is involved: `Popen`, the health wait, the request
pair, `server_command()` (which reads the install manifest) and the log path are
all faked, and the fake server writes fixture lines in the shape the Swift engine
prints them.

    cd benchmark && python3 -m unittest test_tinytitan_slots_ab -v
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
    "tinytitan_slots_ab", ROOT / "benchmark" / "tinytitan_slots_ab.py"
)
sab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sab)

GEN = "TinyTitan generation prefill_s=4.912 decode_s=11.900 decode_tok_s=43.210"
GEN_NO_RATE = "TinyTitan generation started"
RUNNER = "TinyTitan runner cb1_ms=1.210 io_ms=0.440 cb2_ms=0.980"
KERNEL_ROLE = "TinyTitan kernel role=gemv gpu_ms=0.310 count=64"
KERNEL_TOTAL = "TinyTitan kernel total_gpu_ms=12.700 gpu_share_of_decode=10.7%"
FULL_LOG = [GEN, GEN, RUNNER, RUNNER, KERNEL_TOTAL]
EMPTY = ([], [], [])


class CaptureTests(unittest.TestCase):
    """Which footer line belongs to which channel of the A/B."""

    def test_each_line_lands_in_the_section_the_prose_names(self):
        gen, runner, total = sab.capture(FULL_LOG)
        self.assertEqual(gen, [GEN, GEN])
        self.assertEqual(runner, [RUNNER, RUNNER])
        self.assertEqual(total, [KERNEL_TOTAL])

    def test_a_generation_line_without_a_decode_rate_is_not_a_footer(self):
        gen, _, _ = sab.capture([GEN_NO_RATE])
        self.assertEqual(gen, [])

    def test_a_runner_line_without_the_io_split_is_not_the_counter_being_compared(self):
        """`io_ms=` is the decisive channel — the pread wall on the critical path.
        A runner line that does not carry it measures nothing here."""
        _, runner, _ = sab.capture(["TinyTitan runner cb1_ms=1.210"])
        self.assertEqual(runner, [])

    def test_the_per_role_table_and_the_total_are_different_channels(self):
        _, _, total = sab.capture([KERNEL_ROLE])
        self.assertEqual(total, [])


class ArmTests(unittest.TestCase):
    """What one arm varies, and what both arms must always share."""

    def test_the_slot_count_reaches_the_environment(self):
        self.assertEqual(sab.arm_environment(64, None)["TINYTITAN_EXPERT_CACHE_SLOTS"], "64")

    def test_both_stat_channels_are_on(self):
        env = sab.arm_environment(32, None)
        self.assertEqual(env["TINYTITAN_RUNNER_STATS"], "1")
        self.assertEqual(env["TINYTITAN_KERNEL_STATS"], "1")

    def test_pin_pops_the_no_pin_control_rather_than_asserting_a_default(self):
        env = sab.arm_environment(32, True, base={"TINYTITAN_NO_PIN": "1"})
        self.assertNotIn("TINYTITAN_NO_PIN", env)

    def test_nopin_sets_it(self):
        self.assertEqual(sab.arm_environment(32, False)["TINYTITAN_NO_PIN"], "1")

    def test_an_unpinned_arm_touches_neither_control(self):
        """`pin=None` means "leave the server's own default alone", which is the
        only way the slot sweep is comparable with the runs that came before it."""
        env = sab.arm_environment(128, None, base={"TINYTITAN_NO_PIN": "1"})
        self.assertEqual(env["TINYTITAN_NO_PIN"], "1")


class ConfigTests(unittest.TestCase):
    """The environment and the command line, read when `main()` runs."""

    def test_the_slot_list_from_the_command_line(self):
        self.assertEqual(sab.parse_args(["32", "128"]), ([32, 128], None))

    def test_the_default_pair_is_the_published_one(self):
        self.assertEqual(sab.parse_args([]), ([32, 128], None))

    def test_pin_and_nopin_are_the_first_argument(self):
        self.assertEqual(sab.parse_args(["pin", "32"]), ([32], True))
        self.assertEqual(sab.parse_args(["nopin"]), ([32, 64], False))

    def test_not_a_number_is_refused_not_raised(self):
        with self.assertRaises(sab.ConfigError):
            sab.parse_args(["thirty-two"])

    def test_an_empty_slot_list_is_refused(self):
        with self.assertRaises(sab.ConfigError):
            sab.parse_args(["pin", ""])

    def test_the_token_count_is_parsed_from_the_environment(self):
        self.assertEqual(sab.max_tokens({}), 512)
        self.assertEqual(sab.max_tokens({"TINYTITAN_AB_TOKENS": "96"}), 96)

    def test_an_unparseable_token_count_is_refused(self):
        with self.assertRaises(sab.ConfigError):
            sab.max_tokens({"TINYTITAN_AB_TOKENS": "512x"})


class VerdictTests(unittest.TestCase):
    """The page and the status, with no server involved."""

    def verdict(self, arms):
        """`arms` is what the driver hands `verdict()`: `(tag, capture-or-None)`."""
        lines, status = sab.verdict(arms)
        return "\n".join(lines), status

    def test_a_full_pair_of_arms_prints_both_and_exits_zero(self):
        output, status = self.verdict(
            [
                (sab.tag_for(32, None), sab.capture(FULL_LOG)),
                (sab.tag_for(128, None), sab.capture(FULL_LOG)),
            ]
        )
        self.assertEqual(status, 0, output)
        self.assertNotIn("NOT MEASURED", output)
        self.assertIn("slots=32", output)
        self.assertIn("slots=128", output)

    def test_the_header_carries_its_own_line_count(self):
        output, _ = self.verdict([(sab.tag_for(32, None), sab.capture(FULL_LOG))])
        self.assertIn("gen 2", output)
        self.assertIn("runner 2", output)
        self.assertIn("kernel total 1", output)

    def test_an_arm_that_captured_nothing_is_named_and_fails(self):
        """The headline: with no capture at all the old driver printed its header
        and exited 0."""
        output, status = self.verdict(
            [(sab.tag_for(32, None), EMPTY), (sab.tag_for(128, None), sab.capture(FULL_LOG))]
        )
        self.assertEqual(status, 1, output)
        self.assertIn("NOT MEASURED", output)
        self.assertNotIn("NOT MEASURED: slots=128", output)

    def test_a_dead_arm_is_reported_rather_than_ending_the_sweep(self):
        output, status = self.verdict(
            [(sab.tag_for(32, None), None), (sab.tag_for(128, None), sab.capture(FULL_LOG))]
        )
        self.assertEqual(status, 1, output)
        self.assertIn("ARM FAILED: slots=32", output)
        self.assertIn("--- slots=128", output)

    def test_the_pin_mode_is_part_of_the_tag_so_two_tables_cannot_be_merged(self):
        """`slots=32` under pin and under nopin are different runs of the same
        lever; one tag for both is one page that cannot be read."""
        self.assertEqual(sab.tag_for(32, None), "slots=32")
        self.assertEqual(sab.tag_for(32, True), "slots=32-pin")
        self.assertEqual(sab.tag_for(32, False), "slots=32-nopin")


class DriverTests(unittest.TestCase):
    """`main()` end to end, with every process and socket faked."""

    def run_main(self, argv=(), env=None, log_lines=None, healthy=True, bench_model=None):
        """(status, output, spawned, sent) driving the real `main()`.

        The fake server writes `log_lines` into the log when terminated, so the
        capture path is the driver's own; `spawned` records the kwargs of every
        `server_command()` call and `sent` the requests made.
        """
        tmp = pathlib.Path(tempfile.mkdtemp())
        log = tmp / "server.log"
        spawned = []
        sent = []

        class FakeProc:
            def __init__(self, command, **kwargs):
                self.stream = open(log, "w", encoding="utf-8")

            def poll(self):
                return None

            def terminate(self):
                for line in log_lines or []:
                    self.stream.write(line + "\n")
                self.stream.flush()

            def wait(self, timeout=None):
                return 0

            def kill(self):
                pass

        def fake_command(bin_path, port, **kwargs):
            spawned.append(kwargs)
            return ["fake-server"]

        def fake_requests(prompt, max_tokens, port):
            sent.append((prompt, max_tokens, port))

        def fake_health(proc, port, **kwargs):
            if isinstance(healthy, list):
                return healthy.pop(0)
            return healthy

        def fake_log_path(name):
            return str(log)

        def fake_env(*args, **kwargs):
            return {}

        def fake_bench_model():
            return bench_model

        patches = [
            # Determinism: the driver's two environment knobs are read by
            # `main()` now, so an operator's `TINYTITAN_*` cannot widen or narrow
            # a test's arms. `clear=True` means an unset variable is genuinely
            # unset rather than inherited from this shell.
            mock.patch.dict(os.environ, env or {}, clear=True),
            mock.patch.object(sab.subprocess, "Popen", new=FakeProc),
            mock.patch.object(sab, "server_command", new=fake_command),
            mock.patch.object(sab, "server_environment", new=fake_env),
            mock.patch.object(sab, "benchmark_log_path", new=fake_log_path),
            mock.patch.object(sab, "request_twice", new=fake_requests),
            mock.patch.object(sab, "wait_for_health", new=fake_health),
            mock.patch.object(sab, "FLUSH_SETTLE_SECONDS", 0),
        ]
        if bench_model is not None:
            patches.append(
                mock.patch.object(tinytitan_profile, "bench_model", new=fake_bench_model)
            )
            patches.append(mock.patch.object(sab, "bench_model", new=fake_bench_model))
        original_argv = sys.argv
        sys.argv = ["tinytitan_slots_ab.py", *argv]
        buffer = io.StringIO()
        try:
            with contextlib.ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                    try:
                        status = sab.main()
                    except SystemExit as e:
                        status = f"SystemExit({e.code})"
        finally:
            sys.argv = original_argv
        return status, buffer.getvalue(), spawned, sent

    def test_a_full_pair_of_arms_is_a_measurement_and_exits_zero(self):
        status, output, spawned, _ = self.run_main(log_lines=FULL_LOG)
        self.assertEqual(status, 0, output)
        self.assertIn("slots=32", output)
        self.assertIn("slots=128", output)
        self.assertEqual(len(spawned), 2)

    def test_each_arm_sends_its_warmup_and_measured_pair(self):
        """`request_twice()` *is* the pair — warm-up then the warm measurement — so
        one call per arm is what the header's line counts are read against."""
        _, _, _, sent = self.run_main(log_lines=FULL_LOG)
        self.assertEqual(len(sent), 2)

    def test_a_server_that_dies_fails_its_arm_without_ending_the_sweep(self):
        """The old `sys.exit(1)` inside `run()` stopped the sweep at the first dead
        arm, so the second slot count was never even tried."""
        status, output, spawned, _ = self.run_main(log_lines=FULL_LOG, healthy=False)
        self.assertEqual(status, 1, output)
        self.assertEqual(len(spawned), 2, output)
        self.assertIn("ARM FAILED: slots=32", output)
        self.assertIn("ARM FAILED: slots=128", output)

    def test_each_arm_boots_with_its_own_slot_count(self):
        """The one control the A/B varies. Both arms running the same slots would
        print two identical sections and call it a comparison."""
        seen = []

        original = sab.arm_environment

        def record(slots, pin, **kwargs):
            seen.append(slots)
            return original(slots, pin, **kwargs)

        with mock.patch.object(sab, "arm_environment", new=record):
            status, output, _, _ = self.run_main(argv=["64", "128"], log_lines=FULL_LOG)
        self.assertEqual(status, 0, output)
        self.assertEqual(seen, [64, 128])

    def test_an_arm_that_captured_nothing_fails_the_run(self):
        status, output, _, _ = self.run_main(log_lines=[])
        self.assertEqual(status, 1, output)
        self.assertIn("NOT MEASURED", output)

    def test_one_dead_arm_and_one_live_one_are_both_reported(self):
        """The mixed case is the one the old `sys.exit(1)` inside `run()` destroyed:
        the first slot count failed to load, and the second — which would have
        answered — never ran, so the page said nothing about either."""
        status, output, spawned, _ = self.run_main(log_lines=FULL_LOG, healthy=[False, True])
        self.assertEqual(status, 1, output)
        self.assertEqual(len(spawned), 2, output)
        self.assertIn("ARM FAILED: slots=32", output)
        self.assertIn("--- slots=128", output)
        self.assertNotIn("ARM FAILED: slots=128", output)

    def test_the_token_count_from_the_environment_reaches_the_request(self):
        _, _, _, sent = self.run_main(
            argv=[], env={"TINYTITAN_AB_TOKENS": "96"}, log_lines=FULL_LOG
        )
        self.assertEqual({tokens for _, tokens, _ in sent}, {96})

    def test_an_unparseable_token_count_is_a_refusal_not_a_traceback(self):
        status, output, _, sent = self.run_main(env={"TINYTITAN_AB_TOKENS": "512x"})
        self.assertEqual(status, 2, output)
        self.assertEqual(sent, [])
        self.assertIn("REFUSED", output)

    def test_a_bad_slot_argument_is_a_refusal_before_any_server_starts(self):
        status, output, _, _ = self.run_main(argv=["thirty-two"])
        self.assertEqual(status, 2, output)
        self.assertIn("usage", output.lower())

    def test_the_named_model_is_the_one_handed_to_the_launcher(self):
        """`MODEL` was frozen from the environment at import; a driver that reads
        the override and then launches someone else's install prints the
        operator's model over a stranger's numbers."""
        status, output, spawned, _ = self.run_main(
            log_lines=FULL_LOG, bench_model="/models/operator-choice"
        )
        self.assertEqual(status, 0, output)
        self.assertEqual(len(spawned), 2)
        for kwargs in spawned:
            self.assertEqual(kwargs["model"], "/models/operator-choice")

    def test_the_pin_mode_is_carried_into_the_arm_tag(self):
        status, output, _, _ = self.run_main(argv=["pin", "32"], log_lines=FULL_LOG)
        self.assertEqual(status, 0, output)
        self.assertIn("pin", output)

    def test_main_returns_an_int_and_the_guard_exits_with_it(self):
        source = (ROOT / "benchmark" / "tinytitan_slots_ab.py").read_text(encoding="utf-8")
        self.assertIn("sys.exit(main())", source)
        self.assertNotIn("\n    main()\n", source)

    def test_no_module_constant_holds_configuration_nothing_can_change(self):
        self.assertFalse(hasattr(sab, "MODEL"))


class ImportTests(unittest.TestCase):
    """What happens to a program that merely imports this driver."""

    GUARD = """
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
import tinytitan_slots_ab  # noqa: F401
print("imported clean")
"""

    def import_child(self, tokens=None):
        env = {"PATH": os.environ["PATH"]}
        if tokens is not None:
            env["TINYTITAN_AB_TOKENS"] = tokens
        return subprocess.run(
            [
                sys.executable,
                "-c",
                self.GUARD,
                str(ROOT / "benchmark" / "tinytitan_slots_ab.py"),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(ROOT / "benchmark"),
            env=env,
            check=False,
        )

    def test_importing_the_module_starts_no_server_and_opens_no_log(self):
        completed = self.import_child()
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("imported clean", completed.stdout)

    def test_an_unparseable_token_count_is_not_a_failure_at_import(self):
        """`MAX_TOKENS` was cast in module scope, so one bad environment variable
        made this driver unimportable — for a test, for a sibling, for anyone."""
        completed = self.import_child("512x")
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("imported clean", completed.stdout)


if __name__ == "__main__":
    unittest.main()
