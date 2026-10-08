"""The judge comparison's parsing and scoring.

A judge spec is `cpu:<install>` or `server:<url>:<model>`, and the URL contains
colons that a naive split eats; `summarize` decides the number the whole
comparison rests on. Both are pinned here, with no model; the refusal tests use a
closed port and a stub HTTP server on loopback, never a live TinyTitan.

    cd benchmark && python3 -m unittest test_side_engine_judges -v
"""

from __future__ import annotations

import contextlib
import http.server
import importlib.util
import io
import json
import pathlib
import socket
import sys
import tempfile
import threading
import unittest
import unittest.mock

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


judges = _load("side_engine_judges", "benchmark/side_engine_judges.py")


class ParseJudgeTests(unittest.TestCase):
    def test_a_cpu_judge_is_an_install(self):
        self.assertEqual(
            judges.parse_judge("cpu:models/qwen3.5_4B_4Bit"),
            ("cpu", "models/qwen3.5_4B_4Bit", None),
        )

    def test_a_server_judge_keeps_the_url_whole(self):
        kind, url, model = judges.parse_judge(
            "server:http://127.0.0.1:8080/v1:qwen3.6-35b-a3b_4-Bit"
        )
        self.assertEqual(kind, "server")
        self.assertEqual(url, "http://127.0.0.1:8080/v1")
        self.assertEqual(model, "qwen3.6-35b-a3b_4-Bit")

    def test_an_unknown_spec_is_refused(self):
        with self.assertRaises(ValueError):
            judges.parse_judge("gpu:something")


class SummarizeTests(unittest.TestCase):
    def _rows(self) -> pathlib.Path:
        payload = [
            {"task": "T2", "truth": "YES", "prompt": "p", "completion": "YES"},
            {"task": "T2", "truth": "YES", "prompt": "q", "completion": "No."},
            {"task": "T2", "truth": "NO", "prompt": "r", "completion": "maybe"},
            {"task": "T5", "truth": "YES", "prompt": "s", "completion": "YES"},
            {"task": "T5", "truth": "NO", "prompt": "t", "completion": "NO"},
        ]
        handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".jsonl", delete=False)
        handle.write("\n".join(json.dumps(row) for row in payload))
        handle.close()
        return pathlib.Path(handle.name)

    def test_an_unparseable_answer_counts_against_the_half_it_belongs_to(self):
        path = self._rows()
        try:
            per_task = judges.summarize(path)
        finally:
            path.unlink(missing_ok=True)
        self.assertEqual(per_task["T2"]["total"], 3)
        self.assertEqual(per_task["T2"]["correct"], 1)  # YES; "No." is NO
        self.assertEqual(per_task["T2"]["halves"]["YES"], [1, 2])
        self.assertEqual(per_task["T2"]["halves"]["NO"], [0, 1])  # "maybe" refused
        self.assertEqual(per_task["T5"]["correct"], 2)
        self.assertEqual(per_task["T5"]["halves"]["NO"], [1, 1])


if __name__ == "__main__":
    unittest.main()


class RefusalTests(unittest.TestCase):
    """A request that never answered has no answer to score.

    `run_server` used to write `<error …>` into the same `completion` field a real
    reply goes in, so `summarize` counted it as a wrong judgement: a server that
    answered nothing printed the same score as one that answered everything badly.
    """

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def _rows(self, rows: list[dict]) -> pathlib.Path:
        handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".jsonl", delete=False)
        handle.write("\n".join(json.dumps(row) for row in rows))
        handle.close()
        self.addCleanup(pathlib.Path(handle.name).unlink, missing_ok=True)
        return pathlib.Path(handle.name)

    @staticmethod
    def _job_row(task: str, truth: str, **extra: object) -> dict:
        return {"task": task, "truth": truth, "prompt": f"{task} p", **extra}

    def test_a_refused_request_is_not_scored_as_a_wrong_answer(self):
        path = self._rows(
            [
                self._job_row("T2", "YES", completion="YES"),
                self._job_row("T2", "NO", completion="", error="URLError: refused"),
            ]
        )
        per_task = judges.summarize(path)
        self.assertEqual(per_task["T2"]["total"], 1)
        self.assertEqual(per_task["T2"]["correct"], 1)
        self.assertEqual(per_task["T2"]["refused"], 1)
        self.assertEqual(per_task["T2"]["halves"], {"YES": [1, 1]})

    def test_a_run_where_every_request_refused_measures_nothing(self):
        path = self._rows(
            [
                self._job_row("T2", truth, completion="", error="URLError: refused")
                for truth in ("YES", "NO")
            ]
        )
        per_task = judges.summarize(path)
        self.assertEqual(per_task["T2"]["total"], 0)
        self.assertEqual(per_task["T2"]["refused"], 2)
        self.assertEqual(per_task["T2"]["halves"], {})
        # line() used to divide by the measured count and take min() over the
        # halves, so a run with nothing measured crashed instead of reporting.
        buffer = io.StringIO()
        raised = []
        with contextlib.redirect_stdout(buffer):
            try:
                judges.line("server:m", per_task, 3.0, 2)
            except Exception as failure:  # the shape pinned here: report, do not crash
                raised.append(failure)
        out = buffer.getvalue()
        self.assertEqual(raised, [], "a run with nothing measured must report, not raise")
        self.assertIn("refused 2", out)
        self.assertIn("measured nothing", out)

    def test_a_refused_request_records_the_refusal_not_an_answer(self):
        # A closed port: the real urllib failure, with no model and no server.
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        out = pathlib.Path(self.temp.name) / "done.jsonl"
        jobs = [
            {"task": "T2", "system": "s", "prompt": "p", "truth": "YES", "max": 8},
            {"task": "T2", "system": "s", "prompt": "q", "truth": "NO", "max": 8},
        ]
        judges.run_server(f"http://127.0.0.1:{port}/v1", "m", jobs, out)
        rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["completion"], "")
            self.assertIn("URLError", row["error"])

    def test_main_exits_failed_when_a_judgement_was_refused(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        jobs = pathlib.Path(self.temp.name) / "jobs.jsonl"
        jobs.write_text(
            "\n".join(
                json.dumps(
                    {"task": "T2", "system": "s", "prompt": f"p{i}", "truth": "YES", "max": 8}
                )
                for i in range(3)
            )
            + "\n",
            encoding="utf-8",
        )
        argv = [
            "side_engine_judges.py",
            "--jobs",
            str(jobs),
            "--judge",
            f"server:http://127.0.0.1:{port}/v1:m",
        ]
        with (
            unittest.mock.patch.object(sys, "argv", argv),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(judges.main(), 1, "a run that measured nothing must not exit 0")

    def test_main_exits_clean_when_every_judgement_answered(self):
        class Reply(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - the http.server name
                length = int(self.headers["Content-Length"])
                self.rfile.read(length)
                body = json.dumps({"choices": [{"message": {"content": "YES"}}]}).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):  # keep the test output clean
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Reply)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 5)
        port = server.server_address[1]
        jobs = pathlib.Path(self.temp.name) / "answered.jsonl"
        jobs.write_text(
            "\n".join(
                json.dumps(
                    {"task": "T2", "system": "s", "prompt": f"p{i}", "truth": "YES", "max": 8}
                )
                for i in range(2)
            )
            + "\n",
            encoding="utf-8",
        )
        argv = [
            "side_engine_judges.py",
            "--jobs",
            str(jobs),
            "--judge",
            f"server:http://127.0.0.1:{port}/v1:m",
        ]
        with (
            unittest.mock.patch.object(sys, "argv", argv),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(judges.main(), 0)
