"""Gates `benchmark/tinytitan_usecap_ab.py`'s arms, pairing, verdict and status.

The question the driver answers is whether a saturating expert-use-count cap
changes decode tok/s live, after §5y measured it offline on the captured route
trace. The answer only means anything if four properties hold, and each is a
class of defect this repository has already filed (AUD-218, AUD-227, AUD-232,
AUD-276, AUD-281):

1. **An arm that ran nothing is not a measurement.** Every channel is counted and
   an empty capture fails the run.
2. **The headline is the warm request.** Two greedy requests are sent per arm and
   the cold one is dominated by prefill, so averaging the pair — or reading the
   first — would report the warm-up as the result. `figures()` reads the *last*
   footer of each channel, and one test pins that it is not the mean.
3. **A counter that did not appear is absent, not zero.** `io_ms=0.000` is a
   measurement; a missing key is an instrument that did not run, so `figures()`
   yields `None` and the pairing drops the round rather than reading 0.
4. **The cap may not change the answer.** Residency is a cache, so every arm must
   produce the same text; a differing hash outranks any speed number.

Pairing is the point of the file: the machine drifts by more than the effect
(§5w: four arms, 12.2% clock spread, identical hashes), so the number that is
quoted is the *within-round* delta, and a direction needs at least two rounds.
`min_pairs()` is what refuses a single round.

No model, no server, no GPU: `Popen`, the health wait, the requests,
`server_command()` and the log path are faked, and the driver's logs are fed
fixture lines in the shape the Swift engine prints them.

    cd benchmark && python3 -m unittest test_tinytitan_usecap_ab -v
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "tinytitan_usecap_ab", ROOT / "benchmark" / "tinytitan_usecap_ab.py"
)
uc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(uc)

CAP_ENV = "TINYTITAN_EXPERT_USE_CAP"


def gen_line(tok_s: float) -> str:
    return f"TinyTitan generation prefill_s=41.100 decode_s=111.300 decode_tok_s={tok_s:.3f}"


def runner_line(misses: int, read_mib: float, io_ms: float = 92.562) -> str:
    return (
        f"TinyTitan runner cb1_ms=5.079 io_ms={io_ms:.3f} wait_ms=70.939 "
        f"expert_hit_rate=0.9212 expert_hits=44200 expert_misses={misses} "
        f"expert_reloads=612 expert_read_mib={read_mib:.1f} "
        "io_hidden_pct=99.85 cache_plan_ms=4.898 gather_ms=2.248"
    )


KERNEL = "TinyTitan kernel total_gpu_ms=1234.500 gpu_share_of_decode=50.9%"
FULL_LOG = [gen_line(4.100), gen_line(4.601), runner_line(38000, 203.6), KERNEL]


def measure(tok_s=4.6, misses=38000, read_mib=203.6, io_ms=92.5, gpu_ms=1234.5):
    """A `figures()` result, spelled the way the driver builds one."""
    return {
        "decode_tok_s": tok_s,
        "io_ms": io_ms,
        "wait_ms": 70.939,
        "expert_misses": misses,
        "expert_reloads": 612,
        "expert_read_mib": read_mib,
        "total_gpu_ms": gpu_ms,
    }


class ArmEnvironmentTests(unittest.TestCase):
    def test_every_arm_turns_on_both_stat_channels(self):
        for arm in uc.ARMS:
            env = uc.arm_environment(arm, base={})
            self.assertEqual(env["TINYTITAN_RUNNER_STATS"], "1", arm)
            self.assertEqual(env["TINYTITAN_KERNEL_STATS"], "1", arm)

    def test_each_capped_arm_names_its_own_cap(self):
        self.assertEqual(uc.arm_environment("cap3", base={})[CAP_ENV], "3")
        self.assertEqual(uc.arm_environment("cap5", base={})[CAP_ENV], "5")

    def test_an_inherited_cap_does_not_become_the_default_arm(self):
        """The engine reads `TINYTITAN_EXPERT_USE_CAP` at streamer construction,
        so an operator who had exported one would get `cap3 vs cap9` on a page
        that said `default vs cap3`."""
        env = uc.arm_environment("default", base={CAP_ENV: "9"})
        self.assertNotIn(CAP_ENV, env)

    def test_an_unknown_arm_is_refused_rather_than_run_as_default(self):
        with self.assertRaises(ValueError):
            uc.arm_environment("cap7", base={})

    def test_the_launcher_is_handed_the_ram_budget_the_page_claims(self):
        """Every arm must land on the same 96-slot shipped profile: an arm that
        lost the budget would measure a different cache in a different memory
        tier, and the paired delta would be a machine comparison, not a policy
        one. `server_command()` itself is the shared helper and is faked here —
        CI has no install to resolve — so what is pinned is that the driver
        passes it, and that the constant is the 12 GiB this campaign runs at."""
        self.assertEqual(uc.RAM_BUDGET, "12G")
        with mock.patch.object(uc, "server_command", autospec=True) as fake:
            fake.return_value = ["fake-server"]
            uc.server_command_for(uc.BIN, uc.PORT, model="/models/fake")
        fake.assert_called_once_with(
            uc.BIN, uc.PORT, model="/models/fake", ram_budget=uc.RAM_BUDGET
        )


class CaptureTests(unittest.TestCase):
    def test_each_line_lands_in_the_channel_the_prose_names(self):
        gen, runner, kernel = uc.capture(FULL_LOG)
        self.assertEqual(gen, [gen_line(4.100), gen_line(4.601)])
        self.assertEqual(runner, [FULL_LOG[2]])
        self.assertEqual(kernel, [KERNEL])

    def test_a_generation_line_without_a_decode_rate_is_not_a_footer(self):
        """`decode_tok_s=` is the half that carries the measurement; a bare
        `TinyTitan generation` line is a progress print."""
        gen, _, _ = uc.capture(["TinyTitan generation started"])
        self.assertEqual(gen, [])


class FiguresTests(unittest.TestCase):
    def test_the_warm_footer_is_the_arm_and_not_the_mean_of_the_pair(self):
        """AUD-232: a driver that averaged its warm-up into the headline reported
        the cold request's prefill as if it were decode."""
        gen, runner, kernel = uc.capture(FULL_LOG)
        got = uc.figures((gen, runner, kernel))
        self.assertEqual(got["decode_tok_s"], 4.601)

    def test_the_counter_columns_are_read_as_numbers(self):
        gen, runner, kernel = uc.capture(FULL_LOG)
        got = uc.figures((gen, runner, kernel))
        self.assertEqual(got["expert_misses"], 38000)
        self.assertEqual(got["expert_read_mib"], 203.6)
        self.assertEqual(got["io_ms"], 92.562)
        self.assertEqual(got["total_gpu_ms"], 1234.5)

    def test_a_counter_that_never_appeared_is_none_not_zero(self):
        """AUD-276/AUD-283: a measured zero and an absent metric are different
        facts, and pooling them turns an unplugged instrument into a result."""
        gen, _, _ = uc.capture([gen_line(4.601)])
        got = uc.figures((gen, [], []))
        self.assertIsNone(got["io_ms"])
        self.assertIsNone(got["expert_misses"])
        self.assertIsNone(got["total_gpu_ms"])
        self.assertEqual(got["decode_tok_s"], 4.601)


class PairingTests(unittest.TestCase):
    def test_a_pair_is_the_within_round_delta(self):
        rounds = [{"default": measure(tok_s=4.601), "cap3": measure(tok_s=4.750)}]
        rows = uc.pair(rounds, "cap3")
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["delta_tok_s"], 0.149, places=6)
        self.assertAlmostEqual(rows[0]["pct"], 100 * 0.149 / 4.601, places=6)

    def test_the_counter_delta_rides_with_the_clock_delta(self):
        """The mechanism claim is misses and bytes, not wall clock: a clock win
        with unmoved counters is machine drift wearing a result."""
        rounds = [
            {
                "default": measure(tok_s=4.601, misses=38000, read_mib=203.6),
                "cap3": measure(tok_s=4.750, misses=36900, read_mib=197.7),
            }
        ]
        row = uc.pair(rounds, "cap3")[0]
        self.assertEqual(row["delta_misses"], -1100)
        self.assertAlmostEqual(row["delta_read_mib"], -5.9, places=6)

    def test_a_round_missing_the_baseline_is_dropped_and_named(self):
        rounds = [{"cap3": measure()}, {"default": measure(), "cap3": measure()}]
        rows = uc.pair(rounds, "cap3")
        self.assertEqual([row["round"] for row in rows], [2])

    def test_a_round_whose_arm_measured_nothing_is_not_a_pair(self):
        rounds = [{"default": measure(), "cap3": {"decode_tok_s": None}}]
        self.assertEqual(uc.pair(rounds, "cap3"), [])

    def test_one_round_cannot_carry_a_direction(self):
        rounds = [{"default": measure(), "cap3": measure()}]
        _, status = uc.pairing_verdict(rounds)
        self.assertEqual(status, 1)

    def test_two_agreeing_rounds_are_a_result(self):
        rounds = [
            {"default": measure(tok_s=4.601), "cap3": measure(tok_s=4.750)},
            {"default": measure(tok_s=4.590), "cap3": measure(tok_s=4.740)},
        ]
        lines, status = uc.pairing_verdict(rounds)
        self.assertEqual(status, 0, "\n".join(lines))
        self.assertIn("cap3", "\n".join(lines))
        self.assertIn("mean", "\n".join(lines))

    def test_the_page_prints_every_round_not_only_the_mean(self):
        """A swing larger than the mean is the thing a reader needs to see, so
        the per-round deltas are on the page, not hidden behind the average."""
        rounds = [
            {"default": measure(tok_s=4.601), "cap3": measure(tok_s=4.750)},
            {"default": measure(tok_s=4.500), "cap3": measure(tok_s=4.510)},
            {"default": measure(tok_s=4.600), "cap3": measure(tok_s=4.780)},
        ]
        lines, status = uc.pairing_verdict(rounds)
        self.assertEqual(status, 0, "\n".join(lines))
        body = "\n".join(lines)
        for delta in ("+0.149", "+0.010", "+0.180"):
            self.assertIn(delta, body)

    def test_a_capped_arm_with_no_pairs_at_all_fails(self):
        rounds = [{"default": measure()}]
        _, status = uc.pairing_verdict(rounds)
        self.assertEqual(status, 1)


class InvarianceTests(unittest.TestCase):
    def test_identical_hashes_across_arms_are_a_pass(self):
        text = "the same essay"
        shas = {arm: sha(text) for arm in uc.ARMS}
        lines, status = uc.invariance(shas)
        self.assertEqual(status, 0, "\n".join(lines))
        self.assertIn("equal", "\n".join(lines).lower())

    def test_one_differing_arm_fails_and_is_named(self):
        """Residency is a cache, not semantics: a different answer means the cap
        changed what the model saw, and that outranks every speed column."""
        shas = {"default": sha("a"), "cap3": sha("b"), "cap5": sha("a")}
        lines, status = uc.invariance(shas)
        self.assertEqual(status, 1, "\n".join(lines))
        self.assertIn("cap3", "\n".join(lines))

    def test_an_empty_response_does_not_prove_invariance(self):
        """`sha256` of two empty answers is one hash, so an arm that produced no
        text could pass a pure equality check."""
        shas = {"default": uc.EMPTY_SHA, "cap3": uc.EMPTY_SHA, "cap5": uc.EMPTY_SHA}
        _, status = uc.invariance(shas)
        self.assertEqual(status, 1)


class DriverTests(unittest.TestCase):
    """`main()` end to end, with every process and socket faked."""

    def run_main(
        self,
        argv=(),
        log_lines=None,
        healthy=True,
        inherited_cap=None,
        texts=None,
        pgrep=None,
    ):
        """(status, page, popens, events, requests, guard calls) driving the real `main()`.

        `pgrep` is the model-process guard's answer, `("clear", [])` when the
        caller does not care, because every test that gets that far expects the
        run to start.
        """
        pgrep = pgrep or ("clear", [])
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="tinytitan-usecap-ab-"))
        log = tmp / "server.log"
        spawned = []
        events = []

        class FakeProc:
            def __init__(self, command, **kwargs):
                spawned.append((command, kwargs.get("env") or {}))
                events.append("spawn")
                self.stopped = False
                self.stream = kwargs.get("stdout")

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
        original_argv = list(sys.argv)
        sys.argv = ["tinytitan_usecap_ab.py", *argv]
        env_patch = {}
        if inherited_cap is not None:
            env_patch[CAP_ENV] = inherited_cap
        texts = texts or {arm: "the same essay" for arm in uc.ARMS}
        seen = []
        guards = []

        def fake_request(arm, prompt, max_tokens, port):
            seen.append(arm)
            return texts[arm]

        def fake_pgrep(argv):
            guards.append(list(argv))
            return pgrep

        try:
            with (
                mock.patch.object(uc.subprocess, "Popen", new=FakeProc),
                mock.patch.object(uc, "wait_for_health", return_value=healthy),
                mock.patch.object(uc, "generate_greedy_twice", new=fake_request),
                mock.patch.object(uc, "pgrep_answer", new=fake_pgrep),
                mock.patch.object(uc, "server_command", new=_fake_server_command),
                mock.patch.object(uc, "bench_model", return_value="/models/fake"),
                mock.patch.object(uc, "benchmark_log_path", return_value=str(log)),
                mock.patch.object(uc, "FLUSH_SETTLE_SECONDS", 0),
                mock.patch.dict(os.environ, env_patch),
                contextlib.redirect_stdout(buffer),
                contextlib.redirect_stderr(buffer),
            ):
                status = uc.main()
        finally:
            sys.argv = original_argv
            log.unlink(missing_ok=True)
            tmp.rmdir()
        return status, buffer.getvalue(), spawned, events, seen, guards

    def test_a_full_run_is_a_measurement_and_exits_zero(self):
        status, output, spawned, *_ = self.run_main(log_lines=FULL_LOG)
        self.assertEqual(status, 0, output)
        self.assertNotIn("NOT MEASURED", output)
        self.assertNotIn("NOT PAIRED", output)
        self.assertEqual(len(spawned), len(uc.ARMS) * uc.ROUNDS)

    def test_the_arms_interleave_within_each_round(self):
        """Machine drift is the enemy here, so the order is round-major: the
        control rides in every round beside the arms it is compared to, and a
        blocked `default, default, default, cap3, cap3, cap3` would fold drift
        into the delta instead of cancelling it out."""
        _, _, _, events, seen, _ = self.run_main(log_lines=FULL_LOG)
        self.assertEqual(seen, list(uc.ARMS) * uc.ROUNDS)
        self.assertEqual(events, ["spawn", "terminate"] * len(seen))

    def test_the_default_arm_clears_a_cap_the_operator_exported(self):
        status, output, spawned, *_ = self.run_main(log_lines=FULL_LOG, inherited_cap="9")
        self.assertEqual(status, 0, output)
        first = spawned[: len(uc.ARMS)]
        by_arm = {arm: env for arm, (_command, env) in zip(uc.ARMS, first, strict=True)}
        self.assertNotIn(CAP_ENV, by_arm["default"])
        self.assertEqual(by_arm["cap3"][CAP_ENV], "3")
        self.assertEqual(by_arm["cap5"][CAP_ENV], "5")

    def test_every_launch_carries_the_ram_budget(self):
        """The 12 GiB cap is both an operator rule and the profile the 96-slot
        cache comes from; an arm that lost it would measure a different cache."""
        _, _, spawned, *_ = self.run_main(log_lines=FULL_LOG)
        for command, _env in spawned:
            self.assertEqual(command[command.index("--ram") + 1], "12G")

    def test_the_page_names_the_model_length_and_budget_it_ran(self):
        status, output, *_ = self.run_main(log_lines=FULL_LOG)
        self.assertEqual(status, 0, output)
        self.assertIn("/models/fake", output)
        self.assertIn("512", output)
        self.assertIn("12G", output)

    def test_an_arm_that_captured_nothing_fails_the_run(self):
        status, output, *_ = self.run_main(log_lines=[])
        self.assertEqual(status, 1, output)
        self.assertIn("NOT MEASURED", output)

    def test_a_silent_channel_fails_the_run_even_when_the_clock_did_not(self):
        """AUD-273 and AUD-275: a driver prints its refusal and then returns 0
        over it. Here the rate arrived for every arm, the responses are identical
        and the pairing is complete, so the only defect on the page is the two
        channels that logged nothing — and that alone has to reach the status."""
        status, output, *_ = self.run_main(log_lines=[gen_line(4.601)])
        self.assertIn("NOT MEASURED", output)
        self.assertNotIn("NOT PAIRED", output)
        self.assertIn("response invariance: equal", output)
        self.assertEqual(status, 1, output)

    def test_a_single_round_is_refused(self):
        """§5w's arms spread 12.2% on the same configuration, so one round of
        one arm cannot carry a direction and must not print a verdict that reads
        as one."""
        with mock.patch.object(uc, "ROUNDS", 1):
            status, output, *_ = self.run_main(log_lines=FULL_LOG)
        self.assertEqual(status, 1, output)
        self.assertIn("NOT PAIRED", output)

    def test_a_differing_response_fails_the_run(self):
        """Residency is a cache. An arm that answers differently did not merely
        evict differently, and that outranks every speed column on the page."""
        status, output, *_ = self.run_main(
            log_lines=FULL_LOG, texts={"default": "a", "cap3": "b", "cap5": "a"}
        )
        self.assertEqual(status, 1, output)
        self.assertIn("cap3", output)

    def test_an_argument_is_refused_rather_than_ignored(self):
        status, output, spawned, _, seen, _ = self.run_main(argv=["1024"])
        self.assertEqual(status, 2, output)
        self.assertEqual(spawned, [])
        self.assertEqual(seen, [])
        self.assertIn("usage", output.lower())

    def test_a_server_that_never_becomes_healthy_is_reported_per_arm(self):
        status, output, *_ = self.run_main(log_lines=FULL_LOG, healthy=False)
        self.assertEqual(status, 1, output)
        self.assertNotIn("--- default", output)
        self.assertIn("ARM FAILED", output)

    def test_a_live_model_process_refuses_the_run_before_any_spawn(self):
        """AGENTS.md allows one model process, and this driver starts its own
        before every arm, so the guard is a precondition of each launch rather
        than a note at the top of a run that lasts an hour."""
        status, output, spawned, _, seen, guards = self.run_main(
            log_lines=FULL_LOG, pgrep=("busy", ["4242 TinyTitanServer --port 8113"])
        )
        self.assertEqual(status, 3, output)
        self.assertEqual(spawned, [])
        self.assertEqual(seen, [])
        self.assertIn("model process", output)
        self.assertIn("TinyTitanServer|TinyTitanCLI", " ".join(guards[0]))

    def test_a_guard_that_cannot_answer_is_not_read_as_clear(self):
        """AUD-268: an erroring or missing pgrep reported "nothing running" and
        the run began beside a live model — not knowing is not a clear."""
        status, output, spawned, *_ = self.run_main(
            log_lines=FULL_LOG, pgrep=("unknown", ["pgrep exited 2"])
        )
        self.assertEqual(status, 3, output)
        self.assertEqual(spawned, [])


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _fake_server_command(binary, port, **kwargs):
    command = ["fake-server", kwargs.get("model")]
    if kwargs.get("ram_budget") is not None:
        command += ["--ram", str(kwargs["ram_budget"])]
    return command


class ImportTests(unittest.TestCase):
    def test_importing_the_module_starts_no_server_and_opens_no_log(self):
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
import tinytitan_usecap_ab  # noqa: F401
print("imported clean")
"""
        completed = subprocess.run(
            [sys.executable, "-c", child, str(ROOT / "benchmark" / "tinytitan_usecap_ab.py")],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("imported clean", completed.stdout)


if __name__ == "__main__":
    unittest.main()
