"""Gates `benchmark/tinytitan_maxthroughput.py`'s ladder, attribution and status.

The driver answers one question — what decode rate does a prompt's routing
locality buy — by running a four-prompt ladder against one server and reading the
server's own `decode_tok_s=` footers. Four things made that answer unreadable:

1. It scraped every footer in the log into one flat list and sliced it two at a
   time by position, while nothing in the log names the prompt that produced a
   footer. Measured on this file before the fix: two prompts, the first warm-up
   footer missing, and the page printed `essay: measured=44.00 (warmup 50.00)` —
   where 44.00 is the *digits* warm-up rate — followed by `digits: missing rate
   footer (request failed?)`. One dropped footer relabels every row after it and
   accuses a prompt that ran fine.
2. `run_quant()` and `main()` return no status anywhere in the file and the
   `__main__` guard calls `main()` and discards it, so a run in which no footer
   reached decode at all exits 0.
3. `MAXTPUT_PROMPTS` was applied at module scope, and a name that matched nothing
   `raise SystemExit`d *during import* — a test could not load the module, and an
   operator with a typo in their selector got a refusal from `import`. The
   selector also accepted `essay,bogus` silently, so the ladder printed two rows
   and claimed the second one was digits.
4. `--mtp <dir>` documented a speculative arm measured "against its own
   non-speculative baseline", and passed `mtp_model=mtp` to *both* arms; the
   baseline was already speculating. `MODEL` at module scope names an install no
   code reads, and the default install came from `DEFAULT_MODEL_PATH` rather than
   `bench_model()`, so `TINYTITAN_BENCH_MODEL` was ignored under a label that
   named only a bit-width.

No model, no server and no GPU is involved: `Popen`, the health wait, the request
pair, `server_command()` (which reads the install manifest) and the log path are
all faked, and the fake server appends footer lines in the shape the Swift engine
prints them.

    cd benchmark && python3 -m unittest test_tinytitan_maxthroughput -v
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
    "tinytitan_maxthroughput", ROOT / "benchmark" / "tinytitan_maxthroughput.py"
)
mtm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mtm)

GEN = "TinyTitan generation prefill_s=4.912 decode_s=11.900 decode_tok_s={rate}"
MTP_GEN = "TinyTitan mtp draft_tok_s=90.0 decode_s=10.100 decode_tok_s={rate}"
NO_RATE = "TinyTitan generation started"
COMPLETED = "request resp_0123456789 completed in 10.4s prompt=31 cached=0 completion=512"
LADDER = [name for name, _ in mtm.PROMPTS]


def footers(rates):
    return [GEN.format(rate=rate) for rate in rates]


class SelectTests(unittest.TestCase):
    """Which prompts the ladder runs — decided in `main()`, not at import."""

    def test_no_selector_keeps_the_whole_ladder(self):
        self.assertEqual([name for name, _ in mtm.select_prompts(None)], LADDER)

    def test_an_empty_selector_keeps_the_whole_ladder(self):
        self.assertEqual([name for name, _ in mtm.select_prompts("")], LADDER)

    def test_a_named_subset_keeps_only_that_prompt(self):
        self.assertEqual([name for name, _ in mtm.select_prompts("essay")], ["essay"])

    def test_the_ladder_order_survives_a_reversed_selector(self):
        """The ranking is routing diversity from diverse to repetitive; the rows
        must stay in ladder order whoever typed the selector in what order."""
        chosen = [name for name, _ in mtm.select_prompts("digits,essay")]
        self.assertEqual(chosen, ["essay", "digits"])

    def test_a_selector_tolerates_spaces_and_empty_entries(self):
        self.assertEqual(
            [name for name, _ in mtm.select_prompts(" essay , digits ")], ["essay", "digits"]
        )

    def test_a_name_that_matches_nothing_is_refused(self):
        with self.assertRaises(mtm.ConfigError):
            mtm.select_prompts("bogus")

    def test_a_typo_beside_a_real_name_is_refused_not_dropped(self):
        """`essay,bogus` used to narrow the ladder to essay in silence, so the
        page read as a two-prompt comparison that never ran."""
        with self.assertRaises(mtm.ConfigError):
            mtm.select_prompts("essay,bogus")

    def test_the_refusal_names_the_prompts_the_ladder_knows(self):
        with self.assertRaises(mtm.ConfigError) as caught:
            mtm.select_prompts("bogus")
        for name in LADDER:
            self.assertIn(name, str(caught.exception))


class ScrapeTests(unittest.TestCase):
    """Which log lines are the measurement."""

    def test_a_generation_footer_is_a_rate(self):
        self.assertEqual(mtm.decode_rates(footers([43.21])), [43.21])

    def test_a_speculative_footer_counts_as_a_rate_too(self):
        self.assertEqual(mtm.decode_rates([MTP_GEN.format(rate=57.0)]), [57.0])

    def test_a_generation_line_without_a_rate_is_not_a_footer(self):
        """`decode_tok_s=` is the half that carries the measurement; a bare
        `TinyTitan generation` line is a progress print."""
        self.assertEqual(mtm.decode_rates([NO_RATE]), [])

    def test_the_completion_counter_is_a_separate_channel(self):
        self.assertEqual(mtm.completion_counts([COMPLETED]), [512])

    def test_a_line_without_a_completion_count_is_not_scraped(self):
        self.assertEqual(mtm.completion_counts([footers([43.21])[0]]), [])


class ReportTests(unittest.TestCase):
    """Attribution: which footer belongs to which prompt."""

    def test_a_complete_pair_is_the_only_shape_that_exits_zero(self):
        prompts = [("essay", "p"), ("digits", "q")]
        rates = [50.0, 49.0, 44.0, 43.0]
        lines, status = mtm.arm_report("4bit", prompts, rates, [512] * 4)
        self.assertEqual(status, 0, lines)
        self.assertIn("essay: measured=49.00", "\n".join(lines))
        self.assertIn("digits: measured=43.00", "\n".join(lines))
        self.assertIn("(warmup 50.00)", "\n".join(lines))

    def test_a_missing_footer_refuses_the_row_rather_than_shifting_every_later_one(self):
        """The measurement this suite exists for: three footers for two prompts
        used to print the second prompt's warm-up as the first prompt's measured
        rate. No row may carry a number the log cannot attribute."""
        prompts = [("essay", "p"), ("digits", "q")]
        lines, status = mtm.arm_report("4bit", prompts, [50.0, 44.0, 46.0], [512] * 6)
        self.assertEqual(status, 1, lines)
        self.assertIn("NOT MEASURED", "\n".join(lines))
        self.assertNotIn("measured=", "\n".join(lines))

    def test_the_refusal_prints_both_counts_so_the_reader_can_do_the_arithmetic(self):
        prompts = [("essay", "p"), ("digits", "q")]
        lines, _ = mtm.arm_report("4bit", prompts, [50.0, 44.0, 46.0], [512] * 6)
        text = "\n".join(lines)
        self.assertIn("3", text)
        self.assertIn("4", text)

    def test_an_extra_footer_refuses_too(self):
        """Five footers for four requests means the same thing — the pairing is
        lost — even though every prompt looks covered."""
        prompts = [("essay", "p")]
        lines, status = mtm.arm_report("4bit", prompts, [50.0, 49.0, 48.0], [512] * 3)
        self.assertEqual(status, 1, lines)
        self.assertNotIn("measured=", "\n".join(lines))

    def test_a_run_with_no_footer_is_not_a_measurement(self):
        prompts = [("essay", "p")]
        lines, status = mtm.arm_report("4bit", prompts, [], [])
        self.assertEqual(status, 1, lines)
        self.assertIn("NOT MEASURED", "\n".join(lines))

    def test_an_empty_log_is_diagnosed_as_counters_never_arriving(self):
        """An empty log and a miscounted one are different failures: nothing at
        all came, versus something came and cannot be placed. The count-mismatch
        branch would tell this reader the pairing is lost when there is nothing to
        pair."""
        prompts = [("essay", "p")]
        lines, status = mtm.arm_report("4bit", prompts, [], [])
        self.assertEqual(status, 1, lines)
        text = "\n".join(lines)
        self.assertIn("decode_tok_s=", text)
        self.assertIn(str(mtm.REQUESTS_PER_PROMPT * len(prompts)), text)
        self.assertNotIn("shifts every row", text)

    def test_the_completion_column_is_dropped_when_it_cannot_be_paired(self):
        """The rate is the product; a `ct=` column the log cannot pair must not be
        printed as if it belonged to a prompt."""
        prompts = [("essay", "p"), ("digits", "q")]
        lines, status = mtm.arm_report("4bit", prompts, [50.0, 49.0, 44.0, 43.0], [512])
        self.assertEqual(status, 1, lines)
        self.assertNotIn("measured=", "\n".join(lines))

    def test_the_label_is_carried_so_a_cpu_row_cannot_be_read_as_a_gpu_one(self):
        prompts = [("essay", "p")]
        lines, _ = mtm.arm_report("4bit-cpu", prompts, [50.0, 49.0], [512, 512])
        self.assertTrue(all(line.startswith("4bit-cpu ") for line in lines), lines)


class LabelTests(unittest.TestCase):
    """The one-word width the page prints for an install."""

    def test_the_width_is_read_off_the_install_name(self):
        self.assertEqual(mtm.model_label("models/ornith-1.5_35B_A3B_8Bit", "gpu"), "8bit")
        self.assertEqual(mtm.model_label("models/x_4Bit", "gpu"), "4bit")
        self.assertEqual(mtm.model_label("models/x_6bit", "gpu"), "6bit")

    def test_the_engine_is_carried_in_the_label(self):
        self.assertEqual(mtm.model_label("models/x_8Bit", "cpu"), "8bit-cpu")

    def test_an_install_that_names_no_width_is_not_labelled_a_width_it_did_not_claim(self):
        """ "8bit"/"6bit"/"4bit" else "4bit" meant any path without a width token
        was printed as a 4-bit run."""
        self.assertNotIn(
            mtm.model_label("models/my-uncased-checkpoint", "gpu"), ("4bit", "6bit", "8bit")
        )


class ArgumentTests(unittest.TestCase):
    """What the command line accepts, refused before a server starts."""

    def test_no_arguments_run_the_named_bench_model(self):
        models, mtp, engine = mtm.parse_args([])
        self.assertEqual(models, [])
        self.assertIsNone(mtp)
        self.assertEqual(engine, "gpu")

    def test_positional_arguments_are_the_models(self):
        models, _, _ = mtm.parse_args(["models/a_4Bit", "models/b_8Bit"])
        self.assertEqual(models, ["models/a_4Bit", "models/b_8Bit"])

    def test_mtp_takes_a_directory_and_leaves_the_engine_alone(self):
        models, mtp, engine = mtm.parse_args(["--mtp", "models/x_MTP_4Bit", "models/a_4Bit"])
        self.assertEqual(mtp, "models/x_MTP_4Bit")
        self.assertEqual(models, ["models/a_4Bit"])
        self.assertEqual(engine, "gpu")

    def test_the_cpu_engine_is_an_explicit_choice(self):
        self.assertEqual(mtm.parse_args(["--engine", "cpu"])[2], "cpu")

    def test_an_unknown_engine_is_refused(self):
        with self.assertRaises(mtm.ConfigError):
            mtm.parse_args(["--engine", "metal"])

    def test_a_flag_without_its_value_is_refused(self):
        with self.assertRaises(mtm.ConfigError):
            mtm.parse_args(["--engine"])
        with self.assertRaises(mtm.ConfigError):
            mtm.parse_args(["--mtp"])

    def test_an_unknown_flag_is_refused_rather_than_run_as_a_model(self):
        with self.assertRaises(mtm.ConfigError):
            mtm.parse_args(["--speculative"])


class DriverTests(unittest.TestCase):
    """`main()` end to end, with every process and socket faked."""

    def run_main(self, argv=(), rates=None, healthy=True, bench_model=None, env=None):
        """(status, output, spawned, sent) driving the real `main()`.

        `rates` maps a prompt name to the footers the fake server printed for its
        two requests, so a dropped warm-up is expressible; `bench_model` is the
        value `tinytitan_profile.bench_model()` should report, and passing `None`
        leaves the real one — and the environment it reads — in place.
        """
        rates = rates if rates is not None else {}
        names = {text: name for name, text in mtm.PROMPTS}
        tmp = pathlib.Path(tempfile.mkdtemp())
        log = tmp / "server.log"
        log.touch()
        spawned = []
        sent = []

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

        def fake_requests(prompt, max_tokens, port):
            name = names.get(prompt, prompt)
            sent.append((name, max_tokens, port))
            with open(log, "a", encoding="utf-8") as handle:
                for rate in rates.get(name, ()):
                    handle.write(GEN.format(rate=rate) + "\n")
                    handle.write(COMPLETED + "\n")

        def fake_command(*args, **kwargs):
            spawned.append(kwargs)
            return ["fake-server", *args[1:]]

        def fake_health(proc, port, **kwargs):
            return healthy

        def fake_bench_model():
            return bench_model

        original_argv = sys.argv
        sys.argv = ["tinytitan_maxthroughput.py", *argv]
        buffer = io.StringIO()

        def fake_env(*args, **kwargs):
            return {}

        def fake_log_path(name):
            return str(log)

        patches = [
            # Determinism: an empty selector is the whole ladder, so a selector
            # left in the operator's environment cannot narrow the run out from
            # under a test that expects all four prompts.
            mock.patch.dict(os.environ, {"MAXTPUT_PROMPTS": "", **(env or {})}, clear=False),
            mock.patch.object(mtm.subprocess, "Popen", new=FakeProc),
            mock.patch.object(mtm, "server_command", new=fake_command),
            mock.patch.object(mtm, "server_environment", new=fake_env),
            mock.patch.object(mtm, "benchmark_log_path", new=fake_log_path),
            mock.patch.object(mtm, "request_twice", new=fake_requests),
            mock.patch.object(mtm, "wait_for_health", new=fake_health),
            mock.patch.object(mtm, "FLUSH_SETTLE_SECONDS", 0),
            mock.patch.object(mtm, "REQUEST_SETTLE_SECONDS", 0),
        ]
        if bench_model is not None:
            patches.append(
                mock.patch.object(tinytitan_profile, "bench_model", new=fake_bench_model)
            )
            patches.append(mock.patch.object(mtm, "bench_model", new=fake_bench_model))
        try:
            with contextlib.ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                    try:
                        status = mtm.main()
                    except SystemExit as e:
                        status = f"SystemExit({e.code})"
        finally:
            sys.argv = original_argv
        return status, buffer.getvalue(), spawned, sent

    def test_a_full_ladder_exits_zero(self):
        status, output, spawned, sent = self.run_main(
            ["models/x_4Bit"], rates={name: [50.0, 49.0] for name in LADDER}
        )
        self.assertEqual(status, 0, output)
        self.assertEqual(len(spawned), 1)
        self.assertEqual(len(sent), len(LADDER))

    def test_every_prompt_is_sent_once_as_a_warmup_and_measured_pair(self):
        """`request_twice()` *is* the pair, so one call per prompt at the
        documented length is what `arm_report` counts two footers per prompt
        against; a driver that sent one request per prompt would be reporting a
        cold rate as the measurement."""
        _, _, _, sent = self.run_main(
            ["models/x_4Bit"], rates={name: [50.0, 49.0] for name in LADDER}
        )
        self.assertEqual({tokens for _, tokens, _ in sent}, {mtm.MAX_TOKENS})
        self.assertEqual(len(sent), len(LADDER))
        self.assertEqual(mtm.REQUESTS_PER_PROMPT, 2)

    def test_the_selector_narrows_the_ladder_and_the_requests(self):
        """The narrowing must reach both the rows printed and the requests sent; a
        selector that narrowed only the report would still run the whole ladder."""
        status, output, _, sent = self.run_main(
            ["models/x_4Bit"],
            rates={"essay": [50.0, 49.0]},
            env={"MAXTPUT_PROMPTS": "essay"},
        )
        self.assertEqual(status, 0, output)
        self.assertEqual([name for name, _, _ in sent], ["essay"])
        self.assertNotIn("digits", output)

    def test_a_run_where_nothing_reached_decode_exits_nonzero(self):
        """The headline: a footer-less ladder printed four "missing rate footer"
        lines and exited 0, because `main()` returned nothing to the guard."""
        status, output, _, _ = self.run_main(["models/x_4Bit"], rates={})
        self.assertEqual(status, 1, output)
        self.assertIn("NOT MEASURED", output)

    def test_a_dropped_warmup_footer_does_not_move_a_rate_onto_another_prompt(self):
        """The measurement from the docstring: essay ran once instead of twice, so
        the log holds three footers for four requests. The old driver sliced that
        flat list two at a time and published the next prompt's warm-up as this
        prompt's rate."""
        status, output, _, _ = self.run_main(
            ["models/x_4Bit"], rates={"essay": [50.0], "digits": [44.0, 43.0]}
        )
        self.assertEqual(status, 1, output)
        self.assertNotIn("measured=", output)

    def test_a_server_that_never_answers_health_is_reported_and_fails_the_run(self):
        status, output, spawned, _ = self.run_main(["models/x_4Bit"], healthy=False)
        self.assertEqual(status, 1, output)
        self.assertEqual(len(spawned), 1)
        self.assertIn("ARM FAILED", output)

    def test_the_env_model_is_the_install_that_launches(self):
        """The default came from `DEFAULT_MODEL_PATH`, so a sweep that named its
        model through `TINYTITAN_BENCH_MODEL` launched the shipped install and
        printed a page that named neither."""
        status, _, spawned, _ = self.run_main(rates={}, bench_model="/models/operator-choice")
        self.assertEqual(len(spawned), 1)
        self.assertEqual(spawned[0]["model"], "/models/operator-choice")
        self.assertEqual(status, 1, "a footer-less run must still fail")

    def test_the_page_names_the_install(self):
        _, output, _, _ = self.run_main(["/models/operator-choice_8Bit"], rates={})
        self.assertIn("/models/operator-choice_8Bit", output)

    def test_the_mtp_baseline_runs_without_the_draft_head(self):
        """`--mtp` documented a non-speculative baseline and handed the draft head
        to both arms, so the delta was speculative against speculative."""
        status, output, spawned, _ = self.run_main(
            ["--mtp", "models/x_MTP_4Bit", "models/x_4Bit"],
            rates={name: [50.0, 49.0] for name in LADDER},
        )
        self.assertEqual(status, 0, output)
        self.assertEqual(len(spawned), 2)
        self.assertIsNone(spawned[0]["mtp_model"])
        self.assertEqual(spawned[1]["mtp_model"], "models/x_MTP_4Bit")

    def test_the_second_mtp_arm_is_labelled_so_the_two_rows_cannot_be_mixed(self):
        _, output, spawned, _ = self.run_main(
            ["--mtp", "models/x_MTP_4Bit", "models/x_4Bit"],
            rates={name: [50.0, 49.0] for name in LADDER},
        )
        self.assertEqual(len(spawned), 2)
        self.assertIn("-mtp", output)

    def test_a_refused_selector_exits_two_before_any_server_starts(self):
        """`MAXTPUT_PROMPTS` used to `raise SystemExit` at import and accept a typo
        beside a real name; a refusal belongs to the exit status, not the import."""
        status, output, spawned, _ = self.run_main(
            ["models/x_4Bit"],
            rates={},
            env={"MAXTPUT_PROMPTS": "essay,bogus"},
        )
        self.assertEqual(status, 2, output)
        self.assertEqual(spawned, [])
        self.assertIn("REFUSED", output)

    def test_an_unknown_engine_is_refused_before_any_server_starts(self):
        status, output, spawned, _ = self.run_main(["--engine", "metal", "models/x_4Bit"])
        self.assertEqual(status, 2, output)
        self.assertEqual(spawned, [])

    def test_the_cpu_engine_reaches_the_launcher_and_the_label(self):
        """A CPU row on a page of GPU rows is a different measurement; the label is
        the only thing on the page that says which engine ran."""
        status, output, spawned, _ = self.run_main(
            ["--engine", "cpu", "models/x_8Bit"],
            rates={name: [50.0, 49.0] for name in LADDER},
        )
        self.assertEqual(status, 0, output)
        self.assertEqual(spawned[0]["engine"], "cpu")
        self.assertIn("8bit-cpu", output)

    def test_main_returns_an_int_and_the_guard_exits_with_it(self):
        """The nine drivers in this seam called `main()` and threw the status away;
        the guard is where the exit code is made, so it is gated by text."""
        source = (ROOT / "benchmark" / "tinytitan_maxthroughput.py").read_text(encoding="utf-8")
        self.assertIn("sys.exit(main())", source)
        self.assertNotIn("\n    main()\n", source)

    def test_no_module_constant_names_an_install_that_nothing_reads(self):
        """`MODEL` was computed at import and never used, so it read like the
        driver's model setting while `main()` picked a different one."""
        self.assertFalse(hasattr(mtm, "MODEL"))


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
import tinytitan_maxthroughput  # noqa: F401
print("imported clean")
"""

    def import_child(self, selector=None):
        env = {"PATH": os.environ["PATH"]}
        if selector is not None:
            env["MAXTPUT_PROMPTS"] = selector
        return subprocess.run(
            [
                sys.executable,
                "-c",
                self.GUARD,
                str(ROOT / "benchmark" / "tinytitan_maxthroughput.py"),
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

    def test_a_selector_that_matches_nothing_is_not_a_refusal_at_import(self):
        """`MAXTPUT_PROMPTS` narrowed the ladder in module scope and raised
        `SystemExit` there, so an import of this driver could kill whoever
        imported it — including this suite."""
        completed = self.import_child("bogus")
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("imported clean", completed.stdout)


if __name__ == "__main__":
    unittest.main()
