"""The judge comparison's parsing and scoring.

A judge spec is `cpu:<install>` or `server:<url>:<model>`, and the URL contains
colons that a naive split eats; `summarize` decides the number the whole
comparison rests on. Both are pinned here, with no model and no server.

    cd benchmark && python3 -m unittest test_side_engine_judges -v
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import tempfile
import unittest

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
