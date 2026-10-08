"""Gates `benchmark/capital_of_paris_smartness.py`'s config, rows and exit status.

The driver is the run behind the wiki's Capital-of-Paris-Smartness page: for every
(model, prompt, repeat) it sends a warm-up on a model change and one measured
streaming request to a server the operator already started, appends a JSON row to
`$RESULTS`, and prints one line per row. Four things made its answer unreadable.

1. It ran at *import*: `main()` at module scope (:177) and the run matrix read at
   :39. Measured on the pre-fix file — `python3 -c "import capital_of_paris_smartness"`
   with `RUNS` pointed at one model and a port nothing listens on printed an
   `error` row, appended it to `$RESULTS`, and returned; with `RUNS` unset it died
   with `KeyError: 'RUNS'`, and with `MAXTOK=abc` with `ValueError`. An importer
   cannot configure a crash, which is also why the file had no test.
2. Its exit status ignored every row. `main()` returns nothing and nothing counts
   the rows it wrote, so a whole matrix of failures exited **0** — measured with
   `PORT=8399`, where every row came back `URLError: ... Connection refused`.
   That is the AUD-212/AUD-218 class, and here it feeds a reader:
   `capital_of_paris_report.py` (AUD-223) renders its "every request was served"
   line out of exactly these rows.
3. An empty matrix passed vacuously: `RUNS='[]'` printed nothing, wrote no results
   file and exited 0.
4. A failed row printed `load=Nones` and `cold=None`, because the two keys are set
   only on the success path and the print reads them anyway.

No server, no model and no port are involved here: `post()` is the one seam every
request goes through and the suite fakes it, so the matrix loop, the row fields,
the results file and the exit status are all the driver's own. The numbers on the
wiki page remain the operator's to schedule.

    cd benchmark && python3 -m unittest test_capital_of_paris_smartness -v
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "capital_of_paris_smartness", ROOT / "benchmark" / "capital_of_paris_smartness.py"
)
cps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cps)

RUNS_ONE = json.dumps([["m1", "gpu", "Model One", "4"]])
RUNS_TWO = json.dumps([["m1", "gpu", "Model One", "4"], ["m2@cpu", "cpu", "Model Two", "8"]])


def sse(content="Paris", tokens=3):
    """The lines a streamed greedy request actually produces."""
    return iter(
        [
            b'data: {"choices":[{"delta":{"content":"' + content.encode() + b'"}}]}\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
            b'"usage":{"completion_tokens":%d}}\n' % tokens,
            b"data: [DONE]\n",
        ]
    )


class FakeResponse:
    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self.lines)

    def read(self):
        return b""


class ConfigTests(unittest.TestCase):
    """The variables the driver reads, refused rather than crashed on."""

    def run_with(self, env):
        buffer = io.StringIO()
        with (
            mock.patch.dict(os.environ, env, clear=True),
            contextlib.redirect_stdout(buffer),
            contextlib.redirect_stderr(buffer),
        ):
            return cps.main(), buffer.getvalue()

    def test_a_non_integer_maxtok_is_refused_not_a_traceback(self):
        status, output = self.run_with({"MAXTOK": "abc", "RUNS": RUNS_ONE})
        self.assertEqual(status, 2, output)
        self.assertIn("MAXTOK", output)

    def test_a_zero_or_negative_count_is_refused(self):
        for key, value in (("MAXTOK", "0"), ("REPEATS", "-1"), ("PORT", "0")):
            status, output = self.run_with({key: value, "RUNS": RUNS_ONE})
            self.assertEqual(status, 2, f"{key}={value}: {output}")
            self.assertIn(key, output)

    def test_a_missing_run_matrix_is_a_refusal_that_names_it(self):
        """The old file raised `KeyError: 'RUNS'`, which names the variable only
        to whoever reads a traceback."""
        status, output = self.run_with({"PROMPTS": '["hi"]'})
        self.assertEqual(status, 2, output)
        self.assertIn("RUNS", output)

    def test_a_malformed_run_row_is_refused_with_its_index(self):
        status, output = self.run_with({"RUNS": json.dumps([["m1", "gpu", "only three"]])})
        self.assertEqual(status, 2, output)
        self.assertIn("0", output)

    def test_prompts_must_be_a_list_of_strings(self):
        status, output = self.run_with({"RUNS": RUNS_ONE, "PROMPTS": '{"a": 1}'})
        self.assertEqual(status, 2, output)
        self.assertIn("PROMPTS", output)

    def test_a_prompt_that_is_not_a_string_is_refused(self):
        """A number in the array would reach the request body and come back as a
        400 row, which reads like a server fault rather than a bad run variable."""
        status, output = self.run_with({"RUNS": RUNS_ONE, "PROMPTS": '["Capital of Paris", 5]'})
        self.assertEqual(status, 2, output)
        self.assertIn("PROMPTS", output)

    def test_a_bare_prompt_string_is_still_one_prompt(self):
        with mock.patch.dict(os.environ, {"PROMPT": "Capital of Paris"}, clear=True):
            self.assertEqual(cps.parse_prompts(os.environ), ["Capital of Paris"])

    def test_the_prompts_array_is_parsed_in_order(self):
        env = {"PROMPTS": '["a", "b", "c"]'}
        self.assertEqual(cps.parse_prompts(env), ["a", "b", "c"])

    def test_the_default_prompt_is_the_one_the_wiki_page_ran(self):
        self.assertEqual(cps.parse_prompts({}), [cps.DEFAULT_PROMPT])


class ImportTests(unittest.TestCase):
    """The side effect that made this file untestable, guarded rather than run."""

    def test_importing_the_module_serves_no_request_and_writes_no_results(self):
        """A child guards `urllib.request.urlopen` and `builtins.open` before
        importing, with `RUNS` supplied so the pre-fix file would really run the
        matrix, and proves the guard is not hollow against the current file."""
        child = r"""
import builtins, pathlib, sys, urllib.request

target = sys.argv[1]
results = sys.argv[2]

def boom_urlopen(*args, **kwargs):
    raise RuntimeError("a request ran at import")

real_open = builtins.open
def boom_open(file, *args, **kwargs):
    if str(file) == results:
        raise RuntimeError("the results file was written at import")
    return real_open(file, *args, **kwargs)

urllib.request.urlopen = boom_urlopen
builtins.open = boom_open
sys.path.insert(0, str(pathlib.Path(target).parent))
import capital_of_paris_smartness  # noqa: F401
print("imported clean")
"""
        with tempfile.TemporaryDirectory() as tmp:
            results = str(pathlib.Path(tmp) / "results.jsonl")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    child,
                    str(ROOT / "benchmark" / "capital_of_paris_smartness.py"),
                    results,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env={
                    "PATH": os.environ["PATH"],
                    "RUNS": RUNS_ONE,
                    "RESULTS": results,
                    "PORT": "8399",
                },
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("imported clean", completed.stdout)


class MatrixTests(unittest.TestCase):
    """`main()` end to end, with every request faked."""

    def run_main(self, env=None, responses=None, fail=None, usage=True):
        """(status, output, rows, payloads) driving the real `main()`.

        `responses` maps a prompt to the SSE lines the fake server returns;
        `fail` is an exception the fake raises instead, for the error paths;
        `usage=False` streams deltas that never carry a token count.
        """
        tmp = tempfile.mkdtemp()
        results = pathlib.Path(tmp) / "results.jsonl"
        payloads = []
        responses = responses or {}

        def fake_post(payload, timeout=1800):
            payloads.append(payload)
            if fail is not None:
                raise fail
            if not usage:
                return FakeResponse(
                    [b'data: {"choices":[{"delta":{"content":"x"},"finish_reason":"stop"}]}\n']
                )
            return FakeResponse(responses.get(payload["messages"][0]["content"], sse()))

        full_env = {
            "RUNS": RUNS_ONE,
            "PROMPTS": '["Capital of Paris"]',
            "REPEATS": "1",
            "MAXTOK": "128",
            "PORT": "8091",
            "RESULTS": str(results),
        }
        full_env.update(env or {})
        buffer = io.StringIO()
        with (
            mock.patch.dict(os.environ, full_env, clear=True),
            mock.patch.object(cps, "post", new=fake_post),
            contextlib.redirect_stdout(buffer),
            contextlib.redirect_stderr(buffer),
        ):
            status = cps.main()
        rows = []
        if results.exists():
            rows = [json.loads(line) for line in results.read_text().splitlines() if line]
        return status, buffer.getvalue(), rows, payloads

    def test_a_clean_matrix_writes_every_row_and_exits_zero(self):
        status, output, rows, _ = self.run_main(
            {"RUNS": RUNS_TWO, "PROMPTS": '["a", "b"]', "REPEATS": "2"}
        )
        self.assertEqual(status, 0, output)
        self.assertEqual(len(rows), 2 * 2 * 2)

    def test_the_summary_counts_the_rows_it_planned(self):
        """`2 x 2 x 2 = 8` is arithmetic the reader cannot do from the rows alone,
        so the page states it and the row count has to match."""
        status, output, rows, _ = self.run_main(
            {"RUNS": RUNS_TWO, "PROMPTS": '["a", "b"]', "REPEATS": "2"}
        )
        self.assertEqual(status, 0, output)
        self.assertIn("8 of 8", output)

    def test_a_row_whose_request_fails_is_recorded_and_the_run_fails(self):
        error = urllib.error.HTTPError(
            "http://127.0.0.1:8091/v1/chat/completions",
            500,
            "server error",
            {},
            io.BytesIO(b"no such model"),
        )
        status, output, rows, _ = self.run_main(fail=error)
        self.assertEqual(status, 1, output)
        self.assertEqual(rows[0]["status"], "http_error")
        self.assertIn("500", rows[0]["error"])
        self.assertIn("1 failed", output)

    def test_every_row_failing_does_not_exit_zero(self):
        """The headline: a whole matrix against a port nothing listens on printed
        errors and exited 0."""
        status, output, rows, _ = self.run_main(fail=urllib.error.URLError("connection refused"))
        self.assertEqual(status, 1, output)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "error")
        self.assertIn("0 ok, 1 failed", output)

    def test_an_empty_matrix_measures_nothing(self):
        status, output, rows, _ = self.run_main({"RUNS": "[]"})
        self.assertEqual(status, 1, output)
        self.assertEqual(rows, [])
        self.assertIn("NOT MEASURED", output)

    def test_the_model_is_warmed_only_when_it_changes(self):
        """Two repeats of a resident model are back-to-back measurements; a second
        warm-up would make `cold=False` a lie and cost a full prefill. `MAXTOK` is
        deliberately not the default so the row pins the value where it is used."""
        _, _, rows, payloads = self.run_main({"RUNS": RUNS_ONE, "REPEATS": "2", "MAXTOK": "99"})
        self.assertEqual([r["cold"] for r in rows], [True, False])
        self.assertEqual([r["load_s"] for r in rows][1], 0.0)
        self.assertEqual([p["max_tokens"] for p in payloads], [1, 99, 99])

    def test_the_warm_up_uses_the_prompt_under_test(self):
        _, _, _, payloads = self.run_main({"PROMPTS": '["the real prompt"]'})
        self.assertEqual(payloads[0]["messages"][0]["content"], "the real prompt")

    def test_a_failed_row_prints_n_a_rather_than_none_with_a_unit_suffix(self):
        """`load=Nones` was the old shape: the key is only set on the success path,
        and the line appended an `s` to `None` anyway."""
        status, output, _, _ = self.run_main(fail=urllib.error.URLError("refused"))
        self.assertEqual(status, 1, output)
        self.assertIn("load=n/a", output)
        self.assertNotIn("Nones", output)

    def test_a_measured_row_carries_the_stream_fields_the_report_reads(self):
        _, _, rows, _ = self.run_main({"PROMPTS": '["Capital of Paris"]'})
        row = rows[0]
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["completion_tokens"], 3)
        self.assertEqual(row["content"], "Paris")
        self.assertEqual(row["finish"], "stop")
        self.assertIsNotNone(row["ttft_s"])
        self.assertIsNotNone(row["decode_tok_s"])

    def test_a_stream_that_never_reports_usage_leaves_the_count_none(self):
        """A row with no token count is a run that measured no rate, not a rate of
        zero -- the report must be able to tell them apart."""
        _, _, rows, _ = self.run_main(usage=False)
        self.assertEqual(rows[0]["completion_tokens"], None)
        self.assertIsNone(rows[0]["decode_tok_s"])

    def test_the_port_reaches_the_url_every_request_is_sent_to(self):
        """`post()` builds its URL from `BASE`, so a parsed `PORT` that never
        reaches it would send the whole matrix to the default port."""
        status, output, _, _ = self.run_main({"PORT": "8123"})
        self.assertEqual(status, 0, output)
        self.assertEqual(cps.BASE, "http://127.0.0.1:8123")
        self.assertIn("127.0.0.1:8123", output)

    def test_the_row_line_names_the_label_quant_engine_and_repeat(self):
        status, output, _, _ = self.run_main(
            {"RUNS": json.dumps([["m2@cpu", "cpu", "Model Two", "8"]])}
        )
        self.assertEqual(status, 0, output)
        self.assertIn("Model Two", output)
        self.assertIn("8-bit", output)
        self.assertIn("cpu", output)
        self.assertIn("repeat=1", output)


class WarmTests(unittest.TestCase):
    """The two request helpers, with the socket faked."""

    def test_a_response_that_is_not_an_iterable_of_data_lines_yields_no_tokens(self):
        def fake_post(payload, timeout=1800):
            return FakeResponse([b"event: ping\n", b"\n", b"data: not json\n"])

        with mock.patch.object(cps, "post", new=fake_post):
            row = cps.measure("m", "p")
        self.assertIsNone(row["completion_tokens"])
        self.assertIsNone(row["ttft_s"])


if __name__ == "__main__":
    unittest.main()
