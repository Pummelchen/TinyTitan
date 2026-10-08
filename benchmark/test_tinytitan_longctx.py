#!/usr/bin/env python3
"""Tests for benchmark/tinytitan_longctx.py, driven model-free.

The long-context driver reports three numbers a reader cannot check from the
line that prints them, and it exits 0 whatever happens. These tests pin:

1. A run that measured nothing is not a pass. `main()` returned nothing and the
   guard called bare `main()`, so every request failing still ended with
   `--- server footers ---` and exit 0; and the drift section printed only
   `if len(times) >= 4`, so a generation that streamed no content printed no
   drift line and no failure either.
2. The drift headline measured the wrong quarter. `rate_at(times[:n//2], 0.5)`
   takes the last half *of the first half* -- the second quarter of the run --
   and prints it as `first-half`, so on a generation decaying 40 -> 10 tok/s the
   driver showed 9.64 tok/s of drift where the two halves differ by 15.04.
3. The unit was a guess. Both rates are characters per second divided by a
   hardcoded `4.0 # ~4 chars/token` and printed as `tok/s`, while the same
   request's `usage` carries the token count that makes the conversion measured.
4. A failure was a traceback. `ttft` is `None` until the first content delta, so
   a stream that produced none raised in the f-string; `rss_mb` returns `None`
   and the RSS line formats it the same way; and `proc.terminate()` sat after
   the requests, so any raise left a model server holding the port.
5. The configuration was frozen: `MODEL = str(DEFAULT_MODEL_PATH)` ignored
   `TINYTITAN_BENCH_MODEL`, and the inline `/health` poll never called
   `proc.poll()`, so an exited server and a still-loading one both raised
   `RuntimeError("server not ready")` after 120 s.

No model is loaded here and no port is opened: the driver's own `main()` runs
with `Popen`, `wait_for_health`, `server_command()`, `server_environment()`,
`resolve_api_model()`, `benchmark_log_path()`, `subprocess.check_output` and
`http.client.HTTPConnection` faked and `time.time` stepped by hand.
"""

import http.client
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmark"
sys.path.insert(0, str(BENCH))

import tinytitan_longctx as lctx  # noqa: E402

GEN = "TinyTitan generation prefill_s=12.100 decode_s=20.000 decode_tok_s={rate}"
STEP_S = 0.05
REQUESTS = 3  # the driver's three cells: A, B, C


def sse_body(ct, *, content=True, usage=True, chars_per_token=8.0):
    """An SSE stream of `ct` completion tokens, `chars_per_token` per token.

    `content=False` streams no deltas and `usage=False` no usage chunk, which
    are two different failures: one has no first token to time, the other no
    token count to convert characters with.
    """
    lines = []
    per = int(chars_per_token * 4)
    step = max(1, max(ct, 1) // 20)
    emitted = 0
    while content and emitted < ct:
        take = min(step, ct - emitted)
        emitted += take
        delta = {"choices": [{"delta": {"content": "x" * (per * take)}}]}
        lines.append("data: " + json.dumps(delta) + "\n\n")
    if usage:
        lines.append(
            "data: "
            + json.dumps({"usage": {"completion_tokens": ct, "prompt_tokens": 2449}})
            + "\n\n"
        )
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


class Clock:
    """A `time.time()` stand-in that advances one slice per call."""

    def __init__(self, step=STEP_S):
        self.now = 1000.0
        self.step = step

    def __call__(self):
        self.now += self.step
        return self.now


class Resp:
    def __init__(self, body, slice_size=64):
        self.body = body
        self.pos = 0
        self.slice = slice_size

    def read(self, size=None):
        if size is None or size < 0:
            out, self.body = self.body, b""
            return out
        # A real SSE response arrives in chunks smaller than the caller asked
        # for, so the driver's per-read timestamps get more than one sample.
        take = min(size, self.slice)
        out = self.body[self.pos : self.pos + take]
        self.pos += take
        return out


class Conn:
    """One fake connection: `GET /health` answers, `POST` streams `body`."""

    body = b""
    healthy = True
    posts = []

    def __init__(self, host, port, timeout=None):
        self.port = port

    def request(self, method, path, body=None, headers=None):
        self.method = method
        if method == "POST":
            Conn.posts.append((self.port, json.loads(body.decode())))
        return self

    def getresponse(self):
        if self.method == "GET":
            return Resp(b'{"status": "ok"}' if Conn.healthy else b"not ready")
        return Resp(Conn.body)

    def close(self):
        pass


class ImportTests(unittest.TestCase):
    def test_importing_the_module_starts_no_server_and_opens_no_log(self):
        """The child's guard raises if import-time code touches Popen or a socket."""
        with tempfile.TemporaryDirectory() as tmp:
            guards = (
                "import builtins, http.client, subprocess, sys\n"
                "def trip(*a, **k):\n"
                "    raise SystemExit('import ran an external effect')\n"
                "subprocess.Popen = trip\n"
                "subprocess.check_output = trip\n"
                "http.client.HTTPConnection = trip\n"
                "_open = builtins.open\n"
                "def guarded_open(name, *a, **k):\n"
                "    if 'longctx' in str(name) or '.log' in str(name):\n"
                "        raise SystemExit('import opened a log')\n"
                "    return _open(name, *a, **k)\n"
                "builtins.open = guarded_open\n"
                "import importlib.util\n"
                "sys.path.insert(0, str(__import__('pathlib').Path(sys.argv[1]).parent))\n"
                "spec = importlib.util.spec_from_file_location('lctx', sys.argv[1])\n"
                "mod = importlib.util.module_from_spec(spec)\n"
                "spec.loader.exec_module(mod)\n"
                "print('imported clean')\n"
            )
            script = Path(tmp) / "guard.py"
            script.write_text(guards, encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(script), str(BENCH / "tinytitan_longctx.py")],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("imported clean", proc.stdout)

    def test_the_load_wait_is_sized_for_a_real_model(self):
        self.assertGreaterEqual(lctx.SERVER_LOAD_TIMEOUT, 600)


def piecewise(chars_per_token=4.0):
    """200 tokens at 40 tok/s then 200 at 10 tok/s, sampled every 0.05 s.

    The halves are then exact: the first 800 characters took 5.0 s and the
    second 800 took 20.0 s, so a correct segment function owes 40.00 and 10.00
    tok/s and a 30.00 tok/s drift.
    """
    times, t, chars = [(0.0, 0)], 0.0, 0
    for rate, seconds in ((40.0, 5.0), (10.0, 20.0)):
        for _ in range(int(seconds / STEP_S)):
            t += STEP_S
            chars += rate * STEP_S * chars_per_token
            times.append((round(t, 4), round(chars)))
    return times


class SegmentTests(unittest.TestCase):
    def test_the_first_segment_is_the_first_half_not_the_second_quarter(self):
        """The pre-fix `rate_at(times[:n//2], 0.5)` covered t in [5.0, 9.9] of a
        25 s run -- the second quarter -- and printed it as `first-half`."""
        times = piecewise()
        first = lctx.segment_rate(times, 0.0, 0.5, ct=400)
        last = lctx.segment_rate(times, 0.5, 1.0, ct=400)
        self.assertAlmostEqual(first, 40.0, delta=0.2)
        self.assertAlmostEqual(last, 10.0, delta=0.2)

    def test_the_conversion_uses_the_request_own_token_count(self):
        """At 8 chars/token the hardcoded `/ 4.0` doubled every rate; the request's
        own `usage.completion_tokens` is the measured divisor."""
        times = piecewise(chars_per_token=8.0)
        rate = lctx.segment_rate(times, 0.0, 1.0, ct=400)
        self.assertAlmostEqual(rate, 400 / 25.0, delta=0.3)

    def test_a_segment_without_samples_or_without_tokens_is_none_not_a_crash(self):
        self.assertIsNone(lctx.segment_rate([], 0.0, 0.5, ct=100))
        self.assertIsNone(lctx.segment_rate([(1.0, 10)], 0.0, 0.5, ct=0))
        self.assertIsNone(lctx.segment_rate([(1.0, 10), (1.0, 20)], 0.0, 0.5, ct=100))


class DriftTests(unittest.TestCase):
    def test_a_short_capture_is_reported_as_not_measured(self):
        times = [(0.05 * i, 20 * i) for i in range(3)]
        lines, status = lctx.drift_report("B", times, ct=1024)
        self.assertEqual(status, 1)
        self.assertIn("NOT MEASURED", lines[0])
        self.assertIn("B", lines[0])

    def test_the_drift_line_names_both_halves_and_the_conversion(self):
        times = piecewise()
        lines, status = lctx.drift_report("B", times, ct=400)
        self.assertEqual(status, 0)
        text = "\n".join(lines)
        self.assertIn("first-half=40.00 tok/s", text)
        self.assertIn("last-half=10.00 tok/s", text)
        self.assertIn("chars/token", text)

    def test_the_drift_it_prints_is_the_drift_between_the_halves(self):
        times = piecewise()
        lines, status = lctx.drift_report("B", times, ct=400)
        self.assertIn("drift=30.00", "\n".join(lines))

    def test_a_generation_with_no_tokens_is_not_a_drift_measurement(self):
        times = piecewise()
        lines, status = lctx.drift_report("C", times, ct=0)
        self.assertEqual(status, 1)
        self.assertIn("NOT MEASURED", "\n".join(lines))


class RowTests(unittest.TestCase):
    def test_a_stream_that_produced_no_content_is_not_a_measurement(self):
        """The pre-fix line `ttft={ttft:.1f}s` raised TypeError on `None`, so a
        stream with a usage chunk and no deltas ended the run with a traceback."""
        row = {"wall": 5.0, "ttft": None, "pt": 2449, "ct": 128, "chunk_times": []}
        lines, status = lctx.row_report("A:long+128", row)
        self.assertEqual(status, 1)
        text = "\n".join(lines)
        self.assertIn("NOT MEASURED", text)
        self.assertIn("without a single content delta", text)

    def test_a_row_without_a_usage_chunk_says_so(self):
        row = {"wall": 1.0, "ttft": 0.1, "pt": 0, "ct": 0, "chunk_times": [(0.2, 8)]}
        lines, status = lctx.row_report("A:long+128", row)
        self.assertEqual(status, 1)
        self.assertIn("no usage chunk", "\n".join(lines))

    def test_a_measured_row_leads_with_the_server_footer_and_labels_the_client_estimates(self):
        row = {"wall": 21.0, "ttft": 1.0, "pt": 2449, "ct": 128, "chunk_times": [(2.0, 40)]}
        lines, status = lctx.row_report("A:long+128", row, footer_rate=40.0)
        self.assertEqual(status, 0)
        text = "\n".join(lines)
        self.assertIn("decode=40.00 tok/s (server footer)", text)
        self.assertIn("client estimate", text)
        self.assertIn("2449 prompt tokens", text)


class RssTests(unittest.TestCase):
    def test_a_sample_that_failed_is_printed_as_unknown(self):
        lines, status = lctx.rss_report(1000.0, None, 1200.0)
        self.assertIn("afterA=unknown", "\n".join(lines))
        self.assertEqual(status, 0)

    def test_a_base_sample_that_failed_is_a_measurement_that_did_not_happen(self):
        lines, status = lctx.rss_report(None, 1000.0, 1000.0)
        self.assertEqual(status, 1)
        self.assertIn("NOT MEASURED", "\n".join(lines))


class FooterTests(unittest.TestCase):
    def test_the_footers_are_counted_against_the_requests(self):
        lines, status = lctx.footer_report([40.0, 38.0], expected=3)
        self.assertEqual(status, 1)
        self.assertIn("2 decode footers for 3 requests", "\n".join(lines))

    def test_a_full_set_of_footers_is_a_measurement(self):
        lines, status = lctx.footer_report([40.0, 38.0, 20.0], expected=3)
        self.assertEqual(status, 0)
        self.assertIn("server footers: 3 of 3 requests", "\n".join(lines))


class DriverTests(unittest.TestCase):
    """Drive the real main() with every external effect faked."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.where = Path(self.tmp.name)
        self.log_dir = self.where / "logs"
        self.log_dir.mkdir(exist_ok=True)
        self.log_path = self.log_dir / "longctx_server.log"
        self.results = self.where / "results"
        self.results.mkdir(exist_ok=True)
        self.footer_rates = []
        Conn.posts = []
        Conn.body = sse_body(128)
        Conn.healthy = True

    def write_footers(self, rates):
        """What the server log will hold once the run is over.

        The driver truncates the log when it spawns the server, so the fake
        child writes these lines at spawn -- the same order a real server's
        footer lines reach the file the driver handed it.
        """
        self.footer_rates = list(rates)

    def run_main(self, argv=(), *, ready=True, rss=None, env=None):
        state = {"spawned": [], "health": [], "terminated": []}
        spawned_footers = self.footer_rates
        spawned_log_path = self.log_path

        class FakeProc:
            pid = 4242

            def __init__(self, cmd, **kwargs):
                state["spawned"].append(cmd)
                lines = [GEN.format(rate=r) for r in spawned_footers]
                spawned_log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            def poll(self):
                return None

            def terminate(self):
                state["terminated"].append("terminate")

            def wait(self, timeout=None):
                return 0

            def kill(self):
                pass

        def fake_health(proc, port, **kwargs):
            state["health"].append(("wait_for_health", kwargs.get("timeout")))
            return ready

        def fake_command(binary, port, **kwargs):
            return ["fake-server", str(port), str(kwargs.get("model", ""))]

        def fake_environment(*args, **kwargs):
            return {}

        def fake_check_output(cmd, **kwargs):
            if rss is None:
                return b"1234560\n"
            return str(int(rss * 1024)).encode() + b"\n"

        def fake_resolve_api_model(port, **kwargs):
            return "served-model"

        def fake_log_path(name):
            return str(self.log_dir / name)

        def fake_results():
            return str(self.results)

        def no_sleep(seconds):
            return None

        # The model name reaches the driver the way it does in a real run: the
        # environment, read by `bench_model()`, not a patch of the driver itself.
        model_env = {"TINYTITAN_BENCH_MODEL": "/models/default-install", **(env or {})}
        patches = [
            mock.patch.dict(os.environ, model_env, clear=False),
            mock.patch.object(lctx.subprocess, "Popen", new=FakeProc),
            mock.patch.object(lctx.subprocess, "check_output", new=fake_check_output),
            mock.patch.object(lctx, "server_command", new=fake_command),
            mock.patch.object(lctx, "server_environment", new=fake_environment),
            mock.patch.object(lctx, "benchmark_log_path", new=fake_log_path),
            mock.patch.object(lctx, "results_directory", new=fake_results),
            mock.patch.object(lctx, "resolve_api_model", new=fake_resolve_api_model),
            mock.patch.object(lctx, "wait_for_health", new=fake_health),
            mock.patch.object(http.client, "HTTPConnection", new=Conn),
            mock.patch.object(lctx.time, "sleep", new=no_sleep),
            mock.patch.object(lctx.time, "time", new=Clock()),
        ]

        argv_backup = list(sys.argv)
        sys.argv = ["tinytitan_longctx.py", *argv]
        out, err = io.StringIO(), io.StringIO()
        try:
            for patcher in patches:
                patcher.start()
            with redirect_stdout(out), redirect_stderr(err):
                status = lctx.main()
        finally:
            for patcher in reversed(patches):
                try:
                    patcher.stop()
                except RuntimeError:
                    pass
            sys.argv = argv_backup
        return status, out.getvalue() + err.getvalue(), state

    def test_a_measured_run_returns_zero_and_prints_every_cell(self):
        self.write_footers([40.0, 39.0, 21.0])
        status, output, state = self.run_main()
        self.assertEqual(status, 0, output)
        self.assertEqual(len(state["spawned"]), 1)
        self.assertEqual(state["health"][0][1], lctx.SERVER_LOAD_TIMEOUT)
        for label, _prompt, _max_new in lctx.CELLS:
            self.assertIn(label, output)
        self.assertIn("first-half=", output)
        self.assertIn("COMPLETE", output)
        self.assertEqual(state["terminated"], ["terminate"])

    def test_the_requests_are_the_cells_the_header_names(self):
        self.write_footers([40.0, 39.0, 21.0])
        status, output, state = self.run_main()
        self.assertEqual(len(Conn.posts), REQUESTS)

    def test_the_server_is_terminated_when_a_request_raises(self):
        """The orphan-port defect: `terminate()` sat after the requests, so any
        raise left a model server holding the port and ended the run."""

        class Boom(Exception):
            pass

        def exploding_request(*args, **kwargs):
            raise Boom("stream died")

        self.write_footers([])
        escaped = False
        with mock.patch.object(lctx, "request", new=exploding_request):
            try:
                status, output, state = self.run_main()
            except Boom:
                escaped = True
                status, output, state = None, "", {"terminated": []}
        self.assertFalse(escaped, "main() let the driver's own error escape")
        self.assertEqual(status, 1)
        self.assertIn("FAILED", output)
        self.assertEqual(state["terminated"], ["terminate"])

    def test_a_run_where_no_stream_answered_is_not_complete(self):
        Conn.body = sse_body(128, usage=False)
        self.write_footers([])
        status, output, state = self.run_main()
        self.assertEqual(status, 1)
        self.assertIn("NOT MEASURED", output)
        self.assertNotIn("COMPLETE", output)

    def test_a_stream_with_a_usage_chunk_and_no_deltas_is_not_complete(self):
        Conn.body = sse_body(128, content=False)
        self.write_footers([40.0, 39.0, 21.0])
        status, output, state = self.run_main()
        self.assertEqual(status, 1, output)
        self.assertIn("without a single content delta", output)
        self.assertNotIn("COMPLETE", output)

    def test_a_short_footer_log_is_reported(self):
        self.write_footers([40.0])
        status, output, state = self.run_main()
        self.assertEqual(status, 1)
        self.assertIn("decode footers for 3 requests", output)
        self.assertNotIn("COMPLETE", output)

    def test_a_server_that_never_became_ready_runs_no_request(self):
        Conn.healthy = False
        status, output, state = self.run_main(ready=False)
        self.assertEqual(status, 1)
        self.assertEqual(Conn.posts, [])
        self.assertIn("NOT RUN", output)
        self.assertNotIn("COMPLETE", output)
        self.assertEqual(state["terminated"], ["terminate"])

    def test_the_default_install_comes_from_the_environment(self):
        self.write_footers([40.0, 39.0, 21.0])
        status, output, state = self.run_main(
            env={"TINYTITAN_BENCH_MODEL": "/models/operator-choice"},
        )
        self.assertEqual(status, 0, output)
        self.assertIn("/models/operator-choice", state["spawned"][0][-1])

    def test_an_empty_model_in_the_environment_is_refused_not_launched(self):
        """`TINYTITAN_BENCH_MODEL=` is not an install; `str(DEFAULT_MODEL_PATH)`
        used to paper over it by quietly benchmarking a different model."""
        self.write_footers([40.0, 39.0, 21.0])
        status, output, state = self.run_main(env={"TINYTITAN_BENCH_MODEL": ""})
        self.assertEqual(status, 2)
        self.assertIn("REFUSED", output)
        self.assertEqual(state["spawned"], [])

    def test_a_port_no_server_can_bind_is_refused(self):
        self.write_footers([40.0, 39.0, 21.0])
        status, output, state = self.run_main(argv=("--port", "80"))
        self.assertEqual(status, 2)
        self.assertIn("REFUSED", output)
        self.assertEqual(state["spawned"], [])

    def test_the_prompt_is_described_in_characters_and_the_tokens_come_from_the_server(self):
        self.write_footers([40.0, 39.0, 21.0])
        status, output, state = self.run_main()
        self.assertIn("2449 prompt tokens", output)
        self.assertNotIn("10k tokens", output)

    def test_the_drift_number_is_a_token_rate_derived_from_usage(self):
        self.write_footers([40.0, 39.0, 21.0])
        status, output, state = self.run_main()
        self.assertIn("chars/token", output)

    def test_a_run_writes_its_summary_where_the_operator_pointed_it(self):
        self.write_footers([40.0, 39.0, 21.0])
        status, output, state = self.run_main()
        summaries = list(self.results.glob("*.json"))
        self.assertEqual(len(summaries), 1)
        summary = json.loads(summaries[0].read_text(encoding="utf-8"))
        self.assertEqual(summary["cells"], REQUESTS)
        self.assertEqual(summary["status"], 0)

    def test_the_log_descriptor_the_driver_opened_is_closed(self):
        """The pre-fix driver did `log = open(...)` and never closed it, so the
        parent held a second write handle to the file its child was writing."""
        real_open = open
        opened = []

        class Handle:
            def __init__(self, path, mode, encoding=None):
                self.path = str(path)
                self.mode = mode
                self.fh = real_open(path, mode, encoding=encoding)
                self.close_called = False
                opened.append(self)

            def write(self, text):
                return self.fh.write(text)

            def flush(self):
                self.fh.flush()

            def close(self):
                self.close_called = True
                self.fh.close()

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.close()
                return False

            def __iter__(self):
                return iter(self.fh)

        def fake_open(path, mode="r", encoding=None, **kwargs):
            if "b" in mode or not str(path).endswith((".log", ".json")):
                return real_open(path, mode, encoding=encoding, **kwargs)
            return Handle(path, mode, encoding=encoding)

        self.write_footers([40.0, 39.0, 21.0])
        with mock.patch("builtins.open", new=fake_open):
            status, output, state = self.run_main()
        self.assertEqual(status, 0, output)
        log_handles = [h for h in opened if h.path.endswith("longctx_server.log") and "w" in h.mode]
        self.assertTrue(log_handles, "the driver opened no write handle for the server log")
        for handle in log_handles:
            self.assertTrue(handle.close_called, f"{handle.path} left open")

    def test_nothing_is_written_into_the_repository_by_a_run(self):
        self.write_footers([40.0, 39.0, 21.0])
        before = set(os.listdir(BENCH))
        status, output, state = self.run_main()
        self.assertEqual(set(os.listdir(BENCH)), before)

    def test_main_returns_an_int_and_the_guard_exits_with_it(self):
        source = (BENCH / "tinytitan_longctx.py").read_text(encoding="utf-8")
        self.assertIn("sys.exit(main())", source)
        self.assertNotIn("\n    main()\n", source)

    def test_no_frozen_constant_holds_what_the_operator_can_name(self):
        source = (BENCH / "tinytitan_longctx.py").read_text(encoding="utf-8")
        self.assertNotIn("MODEL = str(DEFAULT_MODEL_PATH)", source)
        self.assertNotIn("def rate_at(", source)
        self.assertNotIn("DEFAULT_MODEL_PATH", source)

    def test_the_docstring_states_the_prompt_length_the_code_builds(self):
        """The old docstring called a 9,795-character prompt a "10k-token" one, so
        the header the reader trusts described a context four times longer than
        the request that was actually sent."""
        doc = lctx.__doc__
        stated = re.search(r"is ([0-9,]+) characters", doc)
        self.assertIsNotNone(stated, doc)
        self.assertEqual(int(stated.group(1).replace(",", "")), len(lctx.LONG_PROMPT))
        self.assertNotIn("10k tokens", doc)
        self.assertNotIn("10k-token", doc)
        self.assertIn("prompt_tokens", doc)


class StreamTests(unittest.TestCase):
    def test_read_of_the_whole_body_is_not_sliced_by_a_negative_size(self):
        body = b'{"status": "ok"}'
        self.assertEqual(Resp(body).read(None), body)

    def test_the_clock_steps_so_a_zero_duration_is_not_the_common_case(self):
        clock = Clock(step=0.5)
        self.assertAlmostEqual(clock() - 1000.5, 0.0)


if __name__ == "__main__":
    unittest.main()
