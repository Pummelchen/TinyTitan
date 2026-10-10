#!/usr/bin/env python3
r"""AUD-281: the head-to-head driver refuses the figures it cannot read.

`tinytitan_vs_competitors.py` publishes the claim "faster than any other similar
project", so every number in its table has to be a number. It read none of them
as numbers: the TinyTitan footer was parsed with `float(...)` on whatever token
followed `decode_tok_s=` and returned through `"" if rate else "no decode
footer"`, which is a truthiness test on a float. Measured on the current tree
with only the module's leaves patched:

  * `decode_tok_s=nan` -> `(nan, 100.0, '')`: published as a measurement, and a
    sweep whose repeats all logged a nan prints `nan tok/s` and exits 0.
  * `decode_tok_s=0.00` -> `(0.0, 100.0, 'no decode footer')`: a measured zero
    reported as a missing footer, and the note then discarded because `ok` is
    `rate is not None`.
  * `decode_tok_s=abc` -> `ValueError`, and `decode_tok_s=` with nothing after
    it -> `IndexError`, straight out of the driver with no table and no status.

The same rule was missing on the competitor side, and fairness is that file's
own subject: `float(m.group(1))` on a `[\d.]+` token raises for `..`, Ollama's
`eval_count / eval_duration` raises `ZeroDivisionError` when the duration is 0
and publishes a nan when it is one, and LM Studio's `float(stats rate)` raises
for `"n/a"`. One engine refusing a figure and another publishing it as a rate is
the asymmetry the header warns about.

`main()` returned 0 whatever it measured: a sweep where every cell failed, and a
sweep where every selected engine was skipped, both printed a table of dashes
and exited 0. The status table is the house one -- 0 every cell measured and
clean, 1 measured and contested by a named remainder, 2 the measurement does not
exist.

Everything here is model-free: the driver's own leaves are patched, so no
server, no binary, no engine and nothing fetched.
"""

from __future__ import annotations

import contextlib
import http.client
import io
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import tinytitan_vs_competitors as vc

FOOTER = "TinyTitan generation prefill_tok_s=900.0 decode_tok_s={rate} ttft_ms=12\n"


class TinyTitanFooter(unittest.TestCase):
    """The rate the server's own footer printed, read as a number or refused."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="aud281-footer-"))
        (self.tmp / ".build").mkdir(parents=True, exist_ok=True)
        vc.ROOT = self.tmp

    def drive(self, text):
        """The real `run_tinytitan` over a log the fake server writes.

        A figure the driver cannot read is a refusal it should publish, not an
        exception that ends the sweep, so a raise is reported as one here rather
        than escaping the test.
        """
        try:
            return self._run(text)
        except Exception as exc:
            return "RAISED", 0.0, f"{type(exc).__name__}: {exc}"

    def _run(self, text):
        real = (vc.subprocess.Popen, vc.sample_rss, vc.time.time, http.client.HTTPConnection)

        class _Server:
            pid = 4242
            returncode = 0

            def poll(self):
                return None

            def terminate(self):
                pass

            def wait(self, timeout=None):
                pass

            def kill(self):
                pass

        class _Response:
            def __init__(self, body):
                self._body, self._at = body.encode(), 0

            def read(self, size=None):
                if size is None or size < 0:
                    chunk, self._at = self._body[self._at :], len(self._body)
                else:
                    chunk, self._at = self._body[self._at : self._at + size], self._at + size
                return chunk

        class _Connection:
            def __init__(self, *args, **kwargs):
                pass

            def request(self, *args, **kwargs):
                pass

            def getresponse(self):
                return _Response(json.dumps({"data": [{"id": "test-model"}]}))

            def close(self):
                pass

        def fake_popen(command, *, stdout=None, stderr=None, **kwargs):
            stdout.write(text)
            stdout.flush()
            return _Server()

        vc.subprocess.Popen = fake_popen
        vc.sample_rss = lambda pid: 100.0
        vc.time.time = lambda: 0.0
        http.client.HTTPConnection = _Connection
        try:
            return vc.run_tinytitan("a prompt")
        finally:
            (
                vc.subprocess.Popen,
                vc.sample_rss,
                vc.time.time,
                http.client.HTTPConnection,
            ) = real

    def test_a_finite_footer_rate_is_published_as_the_measurement(self):
        self.assertEqual(self.drive(FOOTER.format(rate="41.50")), (41.5, 100.0, ""))

    def test_a_measured_zero_is_published_and_does_not_claim_a_missing_footer(self):
        rate, _, note = self.drive(FOOTER.format(rate="0.00"))
        self.assertEqual(rate, 0.0)
        self.assertEqual(note, "")

    def test_a_nan_rate_is_refused_and_never_reaches_the_table(self):
        rate, _, note = self.drive(FOOTER.format(rate="nan"))
        self.assertIsNone(rate, "a nan was returned as a measurement")
        self.assertIn("nan", note)
        self.assertIn("finite", note)

    def test_an_infinite_rate_is_refused(self):
        rate, _, note = self.drive(FOOTER.format(rate="inf"))
        self.assertIsNone(rate, "an infinite rate was returned as a measurement")
        self.assertIn("finite", note)

    def test_a_non_numeric_rate_is_refused_rather_than_raising(self):
        rate, _, note = self.drive(FOOTER.format(rate="abc"))
        self.assertIsNone(rate)
        self.assertIn("abc", note)
        self.assertIn("number", note)

    def test_an_empty_rate_token_is_refused_rather_than_raising(self):
        rate, _, note = self.drive("TinyTitan generation decode_tok_s=\n")
        self.assertIsNone(rate)
        self.assertIn("decode footer", note)
        self.assertIn("(nothing)", note, "an absent figure needs to say it was absent")

    def test_a_footer_with_no_decode_figure_is_absent(self):
        rate, _, note = self.drive("TinyTitan generation prefill_tok_s=900.0 rss=1\n")
        self.assertIsNone(rate)
        self.assertIn("no decode footer", note)

    def test_a_line_that_is_not_a_tinytitan_footer_is_not_read_as_one(self):
        rate, _, note = self.drive("other engine decode_tok_s=99.0\n")
        self.assertIsNone(rate)
        self.assertIn("no decode footer", note)

    def test_the_mtp_footer_line_is_a_footer_too(self):
        self.assertEqual(
            self.drive("TinyTitan mtp pass decode_tok_s=33.0\n")[0],
            33.0,
        )

    def test_the_last_decode_footer_wins(self):
        text = FOOTER.format(rate="10.0") + FOOTER.format(rate="20.0")
        self.assertEqual(self.drive(text)[0], 20.0)


class CompetitorFigures(unittest.TestCase):
    """The same figure rule on the engines we compare ourselves against."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="aud281-competitor-"))
        vc.pgrep_answer = lambda *args, **kwargs: ("busy", ["1234"])
        self.real_run = vc.subprocess.run

    def tearDown(self):
        vc.subprocess.run = self.real_run

    def subprocess_output(self, text):
        vc.subprocess.run = lambda *args, **kwargs: type(
            "Completed", (), {"stdout": text, "stderr": "", "returncode": 0}
        )()

    def respond(self, payload):
        body = json.dumps(payload).encode()

        class _Response:
            def __init__(self):
                self._at = 0

            def read(self, size=None):
                if size is None or size < 0:
                    chunk, self._at = body[self._at :], len(body)
                else:
                    chunk, self._at = body[self._at : self._at + size], self._at + size
                return chunk

        class _Connection:
            def __init__(self, *args, **kwargs):
                pass

            def request(self, *args, **kwargs):
                pass

            def getresponse(self):
                return _Response()

            def close(self):
                pass

        http.client.HTTPConnection = _Connection

    def drive(self, runner, prompt="a prompt"):
        real = (vc.sample_rss, http.client.HTTPConnection)
        vc.sample_rss = lambda pid: 50.0
        try:
            return runner(prompt)
        except Exception as exc:
            return "RAISED", 0.0, f"{type(exc).__name__}: {exc}"
        finally:
            vc.sample_rss, http.client.HTTPConnection = real

    def test_llamacpp_publishes_the_rate_it_reported(self):
        self.subprocess_output("eval time = 12345.0 ms ( 41.39 tokens per second)")
        self.assertEqual(self.drive(vc.run_llamacpp)[:1], (41.39,))

    def test_llamacpp_refuses_a_figure_that_is_not_a_number(self):
        self.subprocess_output("eval time = 12345.0 ms ( .. tokens per second)")
        rate, _, note = self.drive(vc.run_llamacpp)
        self.assertIsNone(rate, "a non-numeric figure was returned as a rate")
        self.assertIn("llama.cpp", note)
        self.assertIn("not a number", note)

    def test_llamacpp_still_refuses_when_it_reported_nothing(self):
        self.subprocess_output("llama-cli: some error")
        rate, _, note = self.drive(vc.run_llamacpp)
        self.assertIsNone(rate)
        self.assertIn("no decode rate", note)

    def test_mlx_refuses_a_figure_that_is_not_a_number(self):
        self.subprocess_output("Generation: 512 tokens, 1.2.3 tokens-per-sec")
        rate, _, note = self.drive(vc.run_mlx)
        self.assertIsNone(rate)
        self.assertIn("MLX-LM", note)
        self.assertIn("not a number", note)

    def test_mlc_refuses_a_figure_that_is_not_a_number(self):
        self.subprocess_output("decode: 1.2.3 tok/s")
        rate, _, note = self.drive(vc.run_mlc)
        self.assertIsNone(rate)
        self.assertIn("MLC-LLM", note)
        self.assertIn("not a number", note)

    def test_ollama_publishes_the_rate_it_computed(self):
        self.respond({"eval_count": 512, "eval_duration": 10_000_000_000})
        self.assertEqual(self.drive(vc.run_ollama)[0], 51.2)

    def test_ollama_refuses_stats_it_cannot_divide_rather_than_raising(self):
        self.respond({"eval_count": 512})
        rate, _, note = self.drive(vc.run_ollama)
        self.assertIsNone(rate, "a missing duration was allowed to leave the driver")
        self.assertIn("no eval stats", note)

    def test_ollama_refuses_a_zero_duration_rather_than_raising(self):
        self.respond({"eval_count": 0, "eval_duration": 0})
        rate, _, note = self.drive(vc.run_ollama)
        self.assertIsNone(rate, "a division by zero was allowed to leave the driver")
        self.assertIn("eval_duration", note)

    def test_ollama_refuses_a_rate_that_computed_to_nan(self):
        self.respond({"eval_count": 512, "eval_duration": float("nan")})
        rate, _, note = self.drive(vc.run_ollama)
        self.assertIsNone(rate)
        self.assertIn("ollama", note)
        self.assertIn("finite", note)

    def test_lmstudio_refuses_a_figure_that_is_not_a_number(self):
        self.respond({"stats": {"tokens_per_second": "n/a"}})
        rate, _, note = self.drive(vc.run_lmstudio)
        self.assertIsNone(rate, "float() on a string was allowed to leave the driver")
        self.assertIn("LM Studio", note)
        self.assertIn("not a number", note)

    def test_lmstudio_refuses_a_nan_figure(self):
        self.respond({"stats": {"tokens_per_second": float("nan")}})
        rate, _, note = self.drive(vc.run_lmstudio)
        self.assertIsNone(rate)
        self.assertIn("finite", note)

    def test_lmstudio_still_refuses_a_missing_stats_block(self):
        self.respond({"stats": {}})
        rate, _, note = self.drive(vc.run_lmstudio)
        self.assertIsNone(rate)
        self.assertIn("tokens_per_second", note)


class SweepStatus(unittest.TestCase):
    """`main()` answers with the status its table describes."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="aud281-status-"))
        self.real_engines = vc.ENGINES
        self.real_pgrep = vc.pgrep_answer

    def tearDown(self):
        vc.ENGINES = self.real_engines
        vc.pgrep_answer = self.real_pgrep

    def engine(self, name, *, ready=True):
        path = None if ready else self.tmp / "no-such-model"
        return vc.Engine(name, None, path, "setup", "model")

    def runner(self, rate, note):
        def run(prompt):
            return rate, 100.0, "" if rate is not None else note

        return run

    def drive(self, rates, *, skipped=(), repeats=1):
        """rates: {engine key: figure that runner reports}. skipped: keys not ready."""
        keys = list(rates) + list(skipped)
        vc.ENGINES = {}
        for key in keys:
            name = {"tinytitan": "TinyTitan", "mlx": "MLX-LM", "llamacpp": "llama.cpp"}[key]
            if key in rates:
                rate, note = rates[key]
                vc.ENGINES[key] = (self.engine(name), self.runner(rate, note))
            else:
                vc.ENGINES[key] = (self.engine(name, ready=False), self.runner(41.5, ""))
        vc.pgrep_answer = lambda *args, **kwargs: ("clear", [])
        out, err = io.StringIO(), io.StringIO()
        argv = [
            str(pathlib.Path("tinytitan_vs_competitors.py")),
            "--engines",
            ",".join(keys),
            "--repeats",
            str(repeats),
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(err),
        ):
            status = vc.main()
        return status, out.getvalue(), err.getvalue()

    def test_a_sweep_that_measured_every_cell_exits_zero(self):
        status, out, err = self.drive({"tinytitan": (41.5, "")})
        self.assertEqual(status, 0, out + err)
        self.assertNotIn("CONTESTED", out)
        self.assertNotIn("NOT MEASURED", out)

    def test_the_summary_table_puts_the_measured_figure_in_its_cell(self):
        status, out, err = self.drive({"tinytitan": (41.5, "")})
        self.assertIn(f"{'code':<8}" + f"{41.5:>14.2f}", out, "the table cell is not the rate")

    def test_a_measured_zero_cell_still_exits_zero(self):
        status, out, err = self.drive({"tinytitan": (0.0, "")})
        self.assertEqual(status, 0, out + err)
        self.assertIn(" 0.00 tok/s", out)
        self.assertNotIn("no decode footer", out)

    def test_a_refused_cell_exits_one_and_names_the_engine_prompt_and_reason(self):
        status, out, err = self.drive(
            {
                "tinytitan": (41.5, ""),
                "mlx": (None, "logged nan, which is not a finite rate"),
            }
        )
        self.assertEqual(status, 1, out + err)
        self.assertIn("CONTESTED", out)
        self.assertIn("MLX-LM", out)
        self.assertIn("nan", out)
        self.assertNotIn("CONTESTED: TinyTitan", out)

    def test_the_summary_table_puts_a_dash_in_the_cell_it_cannot_measure(self):
        status, out, err = self.drive(
            {
                "tinytitan": (41.5, ""),
                "mlx": (None, "logged nan, which is not a finite rate"),
            }
        )
        self.assertIn(f"{'code':<8}" + f"{'-':>14}" + f"{41.5:>14.2f}", out)

    def test_a_sweep_where_nothing_was_measured_refuses_rather_than_passing(self):
        status, out, err = self.drive(
            {
                "tinytitan": (None, "no decode footer"),
                "mlx": (None, "logged nan, which is not a finite rate"),
            }
        )
        self.assertEqual(status, 2, out + err)
        self.assertIn("NOT MEASURED", out)
        self.assertIn("every cell failed", out)

    def test_a_sweep_with_no_engine_selected_names_the_reason(self):
        vc.pgrep_answer = lambda *args, **kwargs: ("clear", [])
        out, err = io.StringIO(), io.StringIO()
        argv = [str(pathlib.Path("tinytitan_vs_competitors.py")), "--engines", ""]
        with (
            mock.patch.object(sys, "argv", argv),
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(err),
        ):
            status = vc.main()
        self.assertEqual(status, 2, out.getvalue() + err.getvalue())
        self.assertIn("no engine was selected", out.getvalue())

    def test_a_selected_engine_that_was_skipped_leaves_the_sweep_contested(self):
        status, out, err = self.drive({"tinytitan": (41.5, "")}, skipped=("llamacpp",))
        self.assertEqual(status, 1, out + err)
        self.assertIn("CONTESTED", out)
        self.assertIn("llama.cpp", out)
        self.assertIn("skipped", out)
        self.assertIn("model", out)
        self.assertIn("no-such-model", out, "a refusal needs to name what is missing")

    def test_a_sweep_where_every_engine_was_skipped_is_not_measured(self):
        status, out, err = self.drive({}, skipped=("tinytitan", "mlx"))
        self.assertEqual(status, 2, out + err)
        self.assertIn("NOT MEASURED", out)
        self.assertIn("no selected engine was ready", out)


if __name__ == "__main__":
    unittest.main()
