"""Gates `benchmark/tinytitan_benchmark.py`'s denominators, its cells and its status.

The driver answers one question per cell -- what does this cache/MTP profile
decode at -- by warming a server, sending every prompt (twice for a cache cell),
and reading the server's own `decode_tok_s=` footers out of the log. Four things
made that answer unreadable, and every one is measured here without a model, a
server or a port:

1. The denominator was the survivor count. `expected_sends = len(results) * (2 if
   warm_sends else 1)` is computed *after* the loop that `continue`s past every
   prompt whose stream ended without a usage chunk, so a cell in which all twelve
   prompts failed compares zero footers against zero sends, matches exactly,
   prints no warning, and reports `Answers verified: 0/0`. Four failures out of
   twelve printed four `FAILED` lines and a `7/8` score over the survivors -- the
   failures erased their own evidence.
2. `main()` returns nothing anywhere in the file and `__main__` calls it without
   `sys.exit`, so the sweep always exits 0 -- including a run in which every cell
   was skipped with `Port {port} FAILED to become ready!`, which still reached the
   unconditional `{label} COMPLETE` footer.
3. `wait_ready(port, attempts=10)` never looked at the process. A server that
   exited during load was polled ten times (~100 s of sleeps) and reported as
   "failed to become ready" rather than as dead, and a server that was merely slow
   -- an 8-bit MoE streaming experts off SSD -- was declared dead after 100 s and
   skipped. The profile's `wait_for_health()` distinguishes exactly those two, and
   takes the load budget as an argument.
4. The run was configured from frozen paths: the default install came from
   `DEFAULT_MODEL_PATH` so `TINYTITAN_BENCH_MODEL` was ignored; a path naming no
   width was labelled `4bit` by the ternary in `main()`; and the MTP cell always
   passed `models/ornith-1.5_35B_A3B_MTP_4Bit`, so a matrix over another install
   drafted with a different model's head.
5. The per-row footer join was `zip(results, rates)`: twelve rows against twenty-four
   footers, so row 2 wore prompt 1's warm send and only the first six prompts'
   rates were ever attached.

    cd benchmark && python3 -m unittest test_tinytitan_benchmark -v
"""

from __future__ import annotations

import contextlib
import http.client
import importlib.util
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "tinytitan_benchmark", ROOT / "benchmark" / "tinytitan_benchmark.py"
)
bt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bt)

GEN = "TinyTitan generation prefill_s=4.912 decode_s=11.900 decode_tok_s={rate}"
COMPLETED = "request resp_01 completed in 10.4s prompt=24 cached=0 completion=20"
BENCH = "/models/operator-choice"
PROMPTS = len(bt.PROMPTS)


def sse(text, usage=None):
    frames = [json.dumps({"choices": [{"delta": {"content": text}}]})]
    if usage is not None:
        frames.append(json.dumps({"choices": [], "usage": usage}))
    return ("".join(f"data: {f}\n\n" for f in frames) + "data: [DONE]\n\n").encode()


def reply(usage=True):
    """The usage payload a good send answers with.

    One answer that satisfies every prompt's expected string, so a clean run
    verifies 12 of 12 and a failure means what it says.
    """
    return sse(
        'Paris 408 BLUE neutral ana@example.com {"name": "Maya", "age": 31} no 7391 '
        "not provided disk full hello world 123! max(nums)",
        {
            "prompt_tokens": 24,
            "completion_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 0},
        }
        if usage
        else None,
    )


class ImportTests(unittest.TestCase):
    """The module must be importable before it is configured."""

    def test_importing_the_module_starts_no_server_and_opens_no_log(self):
        program = (
            "import sys, builtins, http.client, subprocess\n"
            + "sys.path.insert(0, "
            + repr(str(ROOT / "benchmark"))
            + ")\n"
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
            "import tinytitan_benchmark\n"
            "print('imported clean')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(ROOT),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("imported clean", result.stdout)

    def test_the_server_load_budget_is_a_load_and_not_a_hundred_seconds(self):
        """The old wait was ten attempts of ten seconds; an 8-bit MoE streaming
        its experts off SSD is not dead at 100 s."""
        self.assertGreaterEqual(bt.SERVER_LOAD_TIMEOUT, 600)


class LabelTests(unittest.TestCase):
    """What a cell is called, and which install it ran."""

    def test_a_width_in_the_path_is_the_label(self):
        self.assertEqual(bt.quant_label("models/ornith-1.5_35B_A3B_8Bit"), "8bit")
        self.assertEqual(bt.quant_label("models/qwen3.6_35B_A3B_6bit"), "6bit")
        self.assertEqual(bt.quant_label("models/qwen3.5_4B_4Bit"), "4bit")

    def test_a_path_naming_no_width_keeps_its_own_name(self):
        """The ternary in `main()` fell through to `4bit`, which is a quantisation
        claim about an install that never made one."""
        self.assertEqual(
            bt.quant_label("models/ornith-1.5_35B_A3B_final"), "ornith-1.5_35B_A3B_final"
        )


class SendTests(unittest.TestCase):
    """How many sends a cell owes, counted before any of them can fail."""

    def test_a_cache_cell_owes_two_sends_per_prompt(self):
        self.assertEqual(bt.sends_per_prompt("multi-prefix"), 2)

    def test_a_plain_cell_owes_one_send_per_prompt(self):
        self.assertEqual(bt.sends_per_prompt("off"), 1)

    def test_the_expected_sends_count_prompts_not_survivors(self):
        """The defect: `expected_sends` was `len(results) * n` after the loop that
        drops every failed prompt, so twelve failures made the check 0 == 0."""
        self.assertEqual(bt.expected_sends("multi-prefix"), PROMPTS * 2)
        self.assertEqual(bt.expected_sends("off"), PROMPTS)


class CellReportTests(unittest.TestCase):
    """The cell's verdict: lines and status, from the rows and the footers."""

    def report(self, rows, rates, cache_mode="off", label="cache_off_mtp_off_8bit"):
        return bt.cell_report(label, rows, rates, cache_mode=cache_mode)

    def test_a_full_cell_is_measured_and_exits_zero(self):
        lines, status = self.report(list(range(PROMPTS)), [40.0] * PROMPTS)
        self.assertEqual(status, 0, lines)
        self.assertTrue(any("FOOTER DECODE" in line for line in lines), lines)

    def test_missing_rows_are_named_rather_than_divided_away(self):
        """`Answers verified: 7/8` is a perfect score over eight survivors when
        twelve prompts were supposed to run."""
        lines, status = self.report(list(range(PROMPTS - 4)), [40.0] * 8)
        self.assertEqual(status, 1, lines)
        text = "\n".join(lines)
        self.assertIn(f"{PROMPTS - 4} of {PROMPTS}", text)
        self.assertIn("produced no row", text)

    def test_a_cell_with_no_rows_at_all_is_not_measured(self):
        lines, status = self.report([], [])
        self.assertEqual(status, 1, lines)
        text = "\n".join(lines)
        self.assertIn(f"0 of {PROMPTS}", text)

    def test_footers_are_counted_against_the_prompts_that_should_have_run(self):
        lines, status = self.report(list(range(PROMPTS)), [40.0] * (PROMPTS - 3))
        self.assertEqual(status, 1, lines)
        self.assertIn(f"{PROMPTS - 3} decode footers for {PROMPTS} sends", "\n".join(lines))

    def test_a_footer_for_every_send_is_a_measurement(self):
        lines, status = self.report(list(range(PROMPTS)), [40.0] * PROMPTS)
        self.assertEqual(status, 0, lines)
        self.assertFalse(any("NOT MEASURED" in line for line in lines), lines)

    def test_an_extra_footer_is_refused_too(self):
        lines, status = self.report(list(range(PROMPTS)), [40.0] * (PROMPTS + 1))
        self.assertEqual(status, 1, lines)
        self.assertIn(f"{PROMPTS + 1} decode footers for {PROMPTS} sends", "\n".join(lines))


class MtpTests(unittest.TestCase):
    """Which draft head the MTP cell runs with."""

    def test_the_draft_head_the_operator_named_is_the_one_launched(self):
        cmd = bt.mtp_arguments("/drafts/ornith.head")
        self.assertEqual(cmd, ["--mtp-model", "/drafts/ornith.head", "--mtp-memory-mib", "384"])

    def test_a_cell_with_no_draft_head_is_refused_not_guessed(self):
        """`models/ornith-1.5_35B_A3B_MTP_4Bit` was passed whatever the model under
        benchmark was, so a Qwen cell drafted with an Ornith head."""
        self.assertIsNone(bt.mtp_arguments(None))


class FooterJoinTests(unittest.TestCase):
    """Which footer line belongs to which row."""

    def rows(self):
        return [{"capability": name} for name, _, _ in bt.PROMPTS]

    def test_a_cache_row_carries_its_warm_send(self):
        """`zip(results, rates)` paired row 2 with the *second* send of prompt 1,
        so every row in a cache cell carried a neighbouring prompt's rate."""
        rows = self.rows()
        rates = [float(i) for i in range(PROMPTS * 2)]
        bt.attach_footer_rates(rows, rates, cache_mode="multi-prefix")
        self.assertEqual(rows[0]["footer_decode_tok_s"], rates[1])
        self.assertEqual(rows[1]["footer_decode_tok_s"], rates[3])
        self.assertEqual(rows[-1]["footer_decode_tok_s"], rates[-1])

    def test_a_plain_row_carries_its_only_send(self):
        rows = self.rows()
        rates = [float(i) for i in range(PROMPTS)]
        bt.attach_footer_rates(rows, rates, cache_mode="off")
        self.assertEqual(rows[0]["footer_decode_tok_s"], rates[0])
        self.assertEqual(rows[-1]["footer_decode_tok_s"], rates[-1])

    def test_a_log_that_does_not_match_attaches_no_rate(self):
        rows = self.rows()
        bt.attach_footer_rates(rows, [40.0] * (PROMPTS - 1), cache_mode="off")
        self.assertFalse(any("footer_decode_tok_s" in row for row in rows))


class DriverTests(unittest.TestCase):
    """`main()` end to end, with every process and socket faked."""

    def run_main(
        self,
        argv=(),
        ready=True,
        fail_first=0,
        footers=None,
        env=None,
        bench=None,
        where=None,
        footer_on_failed_send=False,
        fail_second_sends=0,
    ):
        """(status, output, spawned, posts, health_calls) driving the real `main()`.

        `ready` is what `wait_for_health()` answers; the profile defines `False` as
        one thing only -- the process exited.
        """
        tmp = pathlib.Path(where) if where else pathlib.Path(tempfile.mkdtemp())
        tmp.mkdir(parents=True, exist_ok=True)
        log = tmp / "server.log"
        log.touch()
        state = {"spawned": [], "posts": 0, "health": [], "footers": 0}
        footer_budget = footers if footers is not None else 10**6
        sends = PROMPTS * 2

        class FakeProc:
            def __init__(self, command, **kwargs):
                state["spawned"].append(dict(kwargs, command=list(command)))

            def poll(self):
                return None

            def terminate(self):
                pass

            def wait(self, timeout=None):
                return 0

            def kill(self):
                pass

            @property
            def pid(self):
                return 4242

        def fake_command(binary, port, **kwargs):
            return ["fake-server", str(port), str(kwargs.get("model", ""))]

        def fake_log_path(name):
            return str(log)

        class Resp:
            def __init__(self, body):
                self.body = body

            def read(self, size=-1):
                if not self.body:
                    return None
                if size is None or size < 0:
                    out, self.body = self.body, b""
                    return out
                out, self.body = self.body[:size], self.body[size:]
                return out

        def is_failing(post):
            """Which sends answer without a usage chunk.

            `fail_first` drops the first sends of a run, whose prompts then never
            send again; `fail_second_sends` drops only the warm send of the first
            `n` cache prompts, so every send is posted and every footer logged.
            """
            if 0 < fail_first and 1 < post <= 1 + min(fail_first, sends + 1):
                return True
            return post in {2 * index + 1 for index in range(1, fail_second_sends + 1)}

        class Conn:
            def __init__(self, host, port, timeout=None):
                pass

            def request(self, method, path, body=None, headers=None):
                self.kind = f"{method} {path}"
                if self.kind == "GET /health":
                    state["health"].append(self.kind)
                if self.kind.startswith("POST"):
                    state["posts"] += 1
                    failing = is_failing(state["posts"])
                    writes = not failing or footer_on_failed_send
                    if writes and state["footers"] < footer_budget:
                        with open(log, "a", encoding="utf-8") as handle:
                            handle.write(GEN.format(rate=40.0 + state["posts"] / 10) + "\n")
                            handle.write(COMPLETED + "\n")
                            state["footers"] += 1

            def getresponse(self):
                if self.kind == "GET /health":
                    return Resp(b"ok" if ready else b"loading")
                if self.kind == "GET /v1/models":
                    return Resp(b'{"data":[{"id":"served-model"}]}')
                if is_failing(state["posts"]):
                    return Resp(sse("Paris", None))
                return Resp(reply())

            def close(self):
                pass

        def fake_health(proc, port, **kwargs):
            state["health"].append(("wait_for_health", kwargs.get("timeout")))
            return ready

        def fake_api_model(port, **kwargs):
            return "served-model"

        def fake_bench_model():
            return bench

        def fake_environment(*args, **kwargs):
            return {}

        def no_sleep(seconds):
            return None

        def fake_results_directory():
            return str(tmp / "results")

        patches = [
            mock.patch.dict(os.environ, {"TINYTITAN_BENCH_MODEL": "", **(env or {})}, clear=False),
            mock.patch.object(bt.subprocess, "Popen", new=FakeProc),
            mock.patch.object(bt, "server_command", new=fake_command),
            mock.patch.object(bt, "server_environment", new=fake_environment),
            mock.patch.object(bt, "benchmark_log_path", new=fake_log_path),
            mock.patch.object(bt, "resolve_api_model", new=fake_api_model),
            mock.patch.object(http.client, "HTTPConnection", new=Conn),
            mock.patch.object(time, "sleep", new=no_sleep),
            mock.patch.object(bt, "results_directory", new=fake_results_directory),
        ]
        if bench is not None:
            patches.append(mock.patch.object(bt, "bench_model", new=fake_bench_model))
        patches.append(mock.patch.object(bt, "wait_for_health", new=fake_health))

        out = io.StringIO()
        original = sys.argv
        sys.argv = ["tinytitan_benchmark.py", *argv]
        try:
            with contextlib.ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                    try:
                        status = bt.main()
                    except SystemExit as exc:
                        status = f"SystemExit({exc.code})"
        finally:
            sys.argv = original
        return status, out.getvalue(), state["spawned"], state["posts"], state["health"]

    def test_a_full_production_cell_exits_zero(self):
        status, output, spawned, posts, _ = self.run_main(["models/a_8Bit"])
        self.assertEqual(status, 0, output)
        self.assertEqual(len(spawned), 1)
        self.assertEqual(posts, 1 + PROMPTS * 2 + 2, output)
        self.assertIn("COMPLETE", output)

    def test_the_health_wait_gets_the_load_budget_not_a_hundred_seconds(self):
        _, _, _, _, health = self.run_main(["models/a_8Bit"])
        waits = [call for call in health if isinstance(call, tuple)]
        self.assertTrue(waits, health)
        self.assertEqual(waits[0][1], bt.SERVER_LOAD_TIMEOUT)

    def test_a_cell_whose_server_died_is_not_run_and_sends_no_request(self):
        """`Port 8081 FAILED to become ready!` then `continue` skipped the cell and
        still reached the unconditional `COMPLETE` footer, with no status anywhere."""
        status, output, _, posts, _ = self.run_main(["models/a_8Bit"], ready=False)
        self.assertEqual(status, 1, output)
        self.assertEqual(posts, 0, output)
        self.assertIn("NOT RUN", output)
        self.assertIn("1 of 1 cells not measured", output)
        self.assertNotIn("COMPLETE", output)

    def test_the_reason_names_the_dead_process_and_its_log(self):
        """`wait_for_health()` answers False for one thing only: the process exited.
        Blaming the load budget sends the operator to the wrong cause."""
        _, output, _, _, _ = self.run_main(["models/a_8Bit"], ready=False)
        self.assertIn("exited before /health answered", output)
        self.assertNotIn("never answered /health within", output)

    def test_every_cell_failing_to_launch_is_not_a_complete_matrix(self):
        status, output, _, _, _ = self.run_main(
            ["models/a_8Bit", "--matrix", "--mtp-model", "/drafts/head"], ready=False
        )
        self.assertEqual(status, 1, output)
        self.assertNotIn("MATRIX COMPLETE", output)
        self.assertIn("NOT RUN", output)
        self.assertIn("3 of 3 cells not measured", output)

    def test_a_prompt_that_failed_to_produce_a_row_is_counted(self):
        """Twelve prompts, four streams end without a usage chunk: the rows drop
        to eight and the old denominator dropped with them."""
        status, output, _, _, _ = self.run_main(["models/a_8Bit"], fail_first=4)
        self.assertEqual(status, 1, output)
        self.assertIn(f"{PROMPTS - 4} of {PROMPTS}", output)

    def test_a_cell_where_every_prompt_failed_is_not_measured(self):
        status, output, _, _, _ = self.run_main(["models/a_8Bit"], fail_first=10**6, footers=0)
        self.assertEqual(status, 1, output)
        self.assertIn(f"0 of {PROMPTS}", output)
        # The number owed is stated even when nothing survived, which is exactly
        # what the survivor denominator used to erase.
        self.assertIn(f"0 decode footers for {2 * PROMPTS} sends", output)
        self.assertNotIn("COMPLETE", output)

    def test_a_short_footer_log_is_refused(self):
        status, output, _, _, _ = self.run_main(["models/a_8Bit"], footers=6)
        self.assertEqual(status, 1, output)
        # Six footer lines, the first of them the warm-up, so five remain for the
        # twenty-four sends a cache cell owes.
        self.assertIn(f"{5} decode footers for {2 * PROMPTS} sends", output)

    def test_the_matrix_runs_the_three_cells_it_promises(self):
        status, output, spawned, _, _ = self.run_main(
            ["models/a_8Bit", "--matrix", "--mtp-model", "/drafts/head"]
        )
        self.assertEqual(status, 0, output)
        self.assertEqual(len(spawned), 3, output)
        self.assertIn("MATRIX COMPLETE", output)

    def test_the_mtp_cell_uses_the_draft_head_it_was_given(self):
        status, output, spawned, _, _ = self.run_main(
            ["models/qwen3.6_35B_A3B_8Bit", "--matrix", "--mtp-model", "/drafts/head"]
        )
        self.assertEqual(status, 0, output)
        mtp = [cmd for call in spawned for cmd in [call["command"]] if "--mtp-model" in cmd]
        self.assertEqual(len(mtp), 1, spawned)
        self.assertIn("/drafts/head", mtp[0])

    def test_a_matrix_without_a_draft_head_refuses_before_anything_launches(self):
        """It used to pass `models/ornith-1.5_35B_A3B_MTP_4Bit` for every model, so
        a Qwen matrix drafted with an Ornith head and still called itself
        complete."""
        status, output, spawned, _, _ = self.run_main(["models/qwen3.6_35B_A3B_8Bit", "--matrix"])
        self.assertEqual(status, 2, output)
        self.assertEqual(spawned, [], output)
        self.assertIn("--mtp-model", output)

    def test_the_default_install_is_the_one_the_operator_nominated(self):
        status, output, spawned, _, _ = self.run_main(
            [], env={"TINYTITAN_BENCH_MODEL": BENCH}, bench=BENCH
        )
        self.assertEqual(status, 0, output)
        self.assertEqual(len(spawned), 1)
        self.assertIn("operator-choice", output)
        self.assertIn(BENCH, [str(arg) for call in spawned for arg in call["command"]])

    def test_a_server_footer_for_a_stream_that_failed_is_still_a_send(self):
        """The server logged all twenty-four footers while four client streams ended
        without a usage chunk: four rows are missing and no footer is, so the rates
        are paired against the twenty-four sends owed, not the eight rows that
        survived. The survivor denominator used to make that a different cell."""
        status, output, _, _, _ = self.run_main(
            ["models/a_8Bit"], fail_second_sends=4, footer_on_failed_send=True
        )
        self.assertEqual(status, 1, output)
        self.assertIn(f"{PROMPTS - 4} of {PROMPTS}", output)
        self.assertNotIn("decode footers", output)
        self.assertIn("FOOTER DECODE", output)

    def read_summary(self, where):
        cells = sorted((where / "results").glob("bench-*/aggregate.json"))
        self.assertEqual(len(cells), 1, [str(cell) for cell in cells])
        return json.loads(cells[0].read_text(encoding="utf-8"))["summary"]

    def test_the_saved_summary_reports_the_cell_that_measured(self):
        """`aggregate.json` is what a reader keeps, and `avg_decode_tok_s` falls
        back to nothing now rather than to the client-side average -- a field named
        for the server's own rate must not publish a client number."""
        where = pathlib.Path(tempfile.mkdtemp()) / "runs"
        status, output, *_ = self.run_main(["models/a_8Bit"], where=where)
        self.assertEqual(status, 0, output)
        summary = self.read_summary(where)
        self.assertTrue(summary["measured"], summary)
        self.assertEqual(summary["sends_expected"], PROMPTS * 2, summary)
        self.assertEqual(summary["prompts_configured"], PROMPTS, summary)
        self.assertIsNotNone(summary["avg_decode_tok_s"], summary)

    def test_the_saved_summary_says_so_when_the_cell_did_not_measure(self):
        where = pathlib.Path(tempfile.mkdtemp()) / "runs"
        status, output, *_ = self.run_main(
            ["models/a_8Bit"], fail_first=10**6, footers=0, where=where
        )
        self.assertEqual(status, 1, output)
        summary = self.read_summary(where)
        self.assertFalse(summary["measured"], summary)
        self.assertIsNone(summary["avg_decode_tok_s"], summary)

    def test_main_returns_an_int_and_the_guard_exits_with_it(self):
        source = (ROOT / "benchmark" / "tinytitan_benchmark.py").read_text(encoding="utf-8")
        self.assertIn("sys.exit(main())", source)
        self.assertNotIn("\n    main()\n", source)

    def test_no_frozen_constant_holds_what_the_operator_can_name(self):
        self.assertFalse(hasattr(bt, "MODEL_ID"))
        self.assertFalse(hasattr(bt, "wait_ready"))

    def test_the_matrix_row_is_labelled_from_the_install_not_guessed(self):
        _, output, _, _, _ = self.run_main(["models/ornith-1.5_35B_A3B_final"])
        self.assertIn("ornith-1.5_35B_A3B_final", output)
        self.assertNotIn("Quantization: 4bit", output)


class StreamTests(unittest.TestCase):
    """The request seam, so a failure means what it says."""

    def drive_stream(self, body):
        class Resp:
            def __init__(self, payload):
                self.body = payload

            def read(self, size=-1):
                if not self.body:
                    return None
                out, self.body = self.body[:size], self.body[size:]
                return out

        class Conn:
            def __init__(self, host, port, timeout=None):
                pass

            def request(self, *args, **kwargs):
                pass

            def getresponse(self):
                return Resp(body)

            def close(self):
                pass

        with mock.patch.object(http.client, "HTTPConnection", new=Conn):
            return bt.send_request_stream([{"role": "user", "content": "x"}])

    def test_a_stream_with_a_usage_chunk_is_a_row(self):
        row = self.drive_stream(reply())
        self.assertIsNotNone(row)
        self.assertEqual(row[3], 20)

    def test_a_stream_without_a_usage_chunk_is_not_a_row(self):
        self.assertIsNone(self.drive_stream(sse("Paris", None)))


if __name__ == "__main__":
    unittest.main()
