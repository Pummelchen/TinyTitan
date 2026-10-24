"""Model-free tests for benchmark/quality_floor.py's capture gate.

Every fixture is a synthetic logit capture in a temporary directory. No model,
tokenizer, server or binary is started, and no test asserts on wall-clock time.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "quality_floor", ROOT / "benchmark" / "quality_floor.py"
)
quality_floor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(quality_floor)


def record(prompt_id, position, logits):
    return {"prompt_id": prompt_id, "position": position, "logits": list(logits)}


class GateCase(unittest.TestCase):
    """A temp directory, capture writers, and an in-process gate invocation."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = pathlib.Path(self._tmp.name)

    def write_jsonl(self, name, records):
        path = self.dir / name
        body = "\n".join(json.dumps(item) for item in records)
        path.write_text(body + "\n", encoding="utf-8")
        return str(path)

    def write_json(self, name, document):
        path = self.dir / name
        path.write_text(json.dumps(document), encoding="utf-8")
        return str(path)

    def run_gate(self, *args):
        out = io.StringIO()
        err = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            status = quality_floor.main([str(arg) for arg in args])
        return status, out.getvalue(), err.getvalue()

    def assert_refused(self, status, out, err, needle):
        self.assertEqual(status, 2, out)
        self.assertIn("NOT MEASURED:", out)
        self.assertIn(needle, out)
        self.assertNotIn("Traceback", out + err)


class BoundaryTests(GateCase):
    def test_pass_at_the_exact_floor_boundary(self):
        reference = self.write_jsonl("ref.jsonl", [record("p", 0, [0.0, 0.0])])
        candidate = self.write_jsonl("cand.jsonl", [record("p", 0, [0.5, 0.0])])
        status, out, err = self.run_gate(reference, candidate, "--reference-delta", "0.25")
        self.assertEqual(status, 0, out + err)
        self.assertIn("noise floor: 0.5", out)
        self.assertIn("max |delta|: 0.5", out)
        self.assertIn("verdict: PASS", out)
        self.assertIn("OVERALL: PASS", out)

    def test_fail_beyond_the_boundary(self):
        reference = self.write_jsonl("ref.jsonl", [record("p", 0, [0.0, 0.0])])
        candidate = self.write_jsonl("cand.jsonl", [record("p", 0, [0.5000001, 0.0])])
        status, out, err = self.run_gate(reference, candidate, "--reference-delta", "0.25")
        self.assertEqual(status, 1, out + err)
        self.assertIn("verdict: FAIL", out)
        self.assertIn("exceeds the floor", out)

    def test_reference_repeat_measures_the_floor(self):
        reference = self.write_jsonl("ref.jsonl", [record("p", 0, [0.0, 0.0])])
        repeat = self.write_jsonl("repeat.jsonl", [record("p", 0, [0.2, 0.0])])
        candidate = self.write_jsonl("cand.jsonl", [record("p", 0, [0.4, 0.0])])
        status, out, err = self.run_gate(reference, candidate, "--reference-repeat", repeat)
        self.assertEqual(status, 0, out + err)
        self.assertIn("noise floor: 0.4 (2 x 0.2", out)
        self.assertIn("reference repeat:", out)
        self.assertIn("max |delta|: 0.4", out)

    def test_reference_repeat_floor_is_inclusive_but_not_rounded(self):
        reference = self.write_jsonl("ref.jsonl", [record("p", 0, [0.0, 0.0])])
        repeat = self.write_jsonl("repeat.jsonl", [record("p", 0, [0.2, 0.0])])
        candidate = self.write_jsonl("cand.jsonl", [record("p", 0, [0.4000001, 0.0])])
        status, out, err = self.run_gate(reference, candidate, "--reference-repeat", repeat)
        self.assertEqual(status, 1, out + err)
        self.assertIn("verdict: FAIL", out)

    def test_self_comparison_is_a_zero_floor_not_a_free_pass(self):
        reference = self.write_jsonl("ref.jsonl", [record("p", 0, [0.0, 0.0])])
        same = self.write_jsonl("same.jsonl", [record("p", 0, [0.0, 0.0])])
        status, out, _ = self.run_gate(reference, same)
        self.assertEqual(status, 0, out)
        self.assertIn("self-comparison", out)
        self.assertIn("noise floor: 0", out)
        moved = self.write_jsonl("moved.jsonl", [record("p", 0, [0.001, 0.0])])
        status, out, _ = self.run_gate(reference, moved)
        self.assertEqual(status, 1, out)


class Top1Tests(GateCase):
    def test_top1_divergence_fails_even_inside_the_floor(self):
        reference = self.write_jsonl("ref.jsonl", [record("p", 0, [0.0, 0.0])])
        candidate = self.write_jsonl("cand.jsonl", [record("p", 0, [0.0, 0.1])])
        status, out, err = self.run_gate(reference, candidate, "--reference-delta", "0.5")
        self.assertEqual(status, 1, out + err)
        self.assertIn("top-1 divergences: 1", out)
        self.assertIn("p position 0: reference 0, candidate 1", out)
        self.assertIn("verdict: FAIL", out)

    def test_ties_break_towards_the_lower_token_id(self):
        reference = self.write_jsonl("ref.jsonl", [record("p", 0, [1.0, 1.0])])
        same_tie = self.write_jsonl("same.jsonl", [record("p", 0, [1.0, 1.0])])
        self.assertEqual(self.run_gate(reference, same_tie)[0], 0)
        broken_tie = self.write_jsonl("broken.jsonl", [record("p", 0, [0.9, 1.0])])
        status, out, _ = self.run_gate(reference, broken_tie)
        self.assertEqual(status, 1, out)
        self.assertIn("reference 0, candidate 1", out)

    def test_reference_repeat_must_be_top1_stable(self):
        reference = self.write_jsonl("ref.jsonl", [record("p", 0, [0.0, 0.0])])
        repeat = self.write_jsonl("repeat.jsonl", [record("p", 0, [0.0, 0.1])])
        candidate = self.write_jsonl("cand.jsonl", [record("p", 0, [0.0, 0.0])])
        status, out, err = self.run_gate(reference, candidate, "--reference-repeat", repeat)
        self.assert_refused(status, out, err, "not top-1 stable")


class CompatibilityTests(GateCase):
    def test_different_prompt_sets_are_refused(self):
        reference = self.write_jsonl("ref.jsonl", [record("p", 0, [0.0, 0.0])])
        candidate = self.write_jsonl("cand.jsonl", [record("q", 0, [0.0, 0.0])])
        status, out, err = self.run_gate(reference, candidate)
        self.assert_refused(status, out, err, "prompt set")
        self.assertIn("missing p", out)
        self.assertIn("extra q", out)

    def test_different_position_counts_are_refused(self):
        reference = self.write_jsonl(
            "ref.jsonl",
            [record("p", 0, [0.0, 0.0]), record("p", 1, [0.0, 0.0])],
        )
        candidate = self.write_jsonl("cand.jsonl", [record("p", 0, [0.0, 0.0])])
        status, out, err = self.run_gate(reference, candidate)
        self.assert_refused(status, out, err, "positions")

    def test_different_vocabulary_lengths_are_refused(self):
        reference = self.write_jsonl("ref.jsonl", [record("p", 0, [0.0, 0.0, 0.0])])
        candidate = self.write_jsonl("cand.jsonl", [record("p", 0, [0.0, 0.0])])
        status, out, err = self.run_gate(reference, candidate)
        self.assert_refused(status, out, err, "vocabulary length")

    def test_reference_repeat_with_a_different_prompt_is_refused(self):
        reference = self.write_jsonl("ref.jsonl", [record("p", 0, [0.0, 0.0])])
        repeat = self.write_jsonl("repeat.jsonl", [record("q", 0, [0.0, 0.0])])
        candidate = self.write_jsonl("cand.jsonl", [record("p", 0, [0.0, 0.0])])
        status, out, err = self.run_gate(reference, candidate, "--reference-repeat", repeat)
        self.assert_refused(status, out, err, "prompt set")


class MalformedCaptureTests(GateCase):
    def test_empty_input_is_refused(self):
        reference = self.dir / "ref.jsonl"
        reference.write_text("", encoding="utf-8")
        candidate = self.write_jsonl("cand.jsonl", [record("p", 0, [0.0])])
        status, out, err = self.run_gate(reference, candidate)
        self.assert_refused(status, out, err, "empty")

    def test_nan_is_refused_and_not_a_traceback(self):
        reference = self.write_jsonl("ref.jsonl", [record("p", 0, [0.0, 0.0])])
        candidate = self.dir / "cand.jsonl"
        candidate.write_text(
            '{"prompt_id": "p", "position": 0, "logits": [0.0, NaN]}\n',
            encoding="utf-8",
        )
        status, out, err = self.run_gate(reference, candidate)
        self.assert_refused(status, out, err, "not finite")

    def test_non_numeric_logit_is_refused(self):
        reference = self.write_jsonl("ref.jsonl", [record("p", 0, [0.0, 0.0])])
        candidate = self.dir / "cand.jsonl"
        candidate.write_text(
            '{"prompt_id": "p", "position": 0, "logits": [0.0, "x"]}\n',
            encoding="utf-8",
        )
        status, out, err = self.run_gate(reference, candidate)
        self.assert_refused(status, out, err, "not a number")

    def test_huge_integer_logit_is_refused_without_a_traceback(self):
        reference = self.write_jsonl("ref.jsonl", [record("p", 0, [0.0, 0.0])])
        candidate = self.dir / "cand.jsonl"
        candidate.write_text(
            '{"prompt_id": "p", "position": 0, "logits": [0.0, ' + "1" * 400 + "]}\n",
            encoding="utf-8",
        )
        status, out, err = self.run_gate(reference, candidate)
        self.assert_refused(status, out, err, "too large")

    def test_duplicate_position_is_refused(self):
        reference = self.write_jsonl("ref.jsonl", [record("p", 0, [0.0, 0.0])])
        candidate = self.write_jsonl(
            "cand.jsonl",
            [record("p", 0, [0.0, 0.0]), record("p", 0, [0.0, 0.0])],
        )
        status, out, err = self.run_gate(reference, candidate)
        self.assert_refused(status, out, err, "repeats prompt")

    def test_position_gap_is_refused(self):
        reference = self.write_jsonl(
            "ref.jsonl",
            [record("p", 0, [0.0, 0.0]), record("p", 2, [0.0, 0.0])],
        )
        candidate = self.write_jsonl(
            "cand.jsonl",
            [record("p", 0, [0.0, 0.0]), record("p", 1, [0.0, 0.0]), record("p", 2, [0.0, 0.0])],
        )
        status, out, err = self.run_gate(reference, candidate)
        self.assert_refused(status, out, err, "not contiguous")

    def test_empty_logits_are_refused(self):
        reference = self.write_jsonl("ref.jsonl", [record("p", 0, [0.0, 0.0])])
        candidate = self.write_jsonl("cand.jsonl", [record("p", 0, [])])
        status, out, err = self.run_gate(reference, candidate)
        self.assert_refused(status, out, err, "non-empty array")


class ContractTests(GateCase):
    def test_missing_capture_refuses_with_a_named_reason(self):
        reference = self.write_jsonl("ref.jsonl", [record("p", 0, [0.0])])
        status, out, err = self.run_gate(reference, self.dir / "missing.jsonl")
        self.assert_refused(status, out, err, "cannot read capture")

    def test_one_capture_is_not_a_comparison(self):
        reference = self.write_jsonl("ref.jsonl", [record("p", 0, [0.0])])
        status, out, err = self.run_gate(reference)
        self.assert_refused(status, out, err, "at least one candidate")

    def test_two_floor_sources_are_refused(self):
        reference = self.write_jsonl("ref.jsonl", [record("p", 0, [0.0])])
        candidate = self.write_jsonl("cand.jsonl", [record("p", 0, [0.0])])
        repeat = self.write_jsonl("repeat.jsonl", [record("p", 0, [0.0])])
        status, out, err = self.run_gate(
            reference, candidate, "--reference-repeat", repeat, "--reference-delta", "0.1"
        )
        self.assert_refused(status, out, err, "not both")

    def test_negative_and_non_finite_reference_deltas_are_refused(self):
        reference = self.write_jsonl("ref.jsonl", [record("p", 0, [0.0])])
        candidate = self.write_jsonl("cand.jsonl", [record("p", 0, [0.0])])
        for delta in ("-0.1", "nan", "inf"):
            status, out, err = self.run_gate(reference, candidate, "--reference-delta", delta)
            self.assert_refused(status, out, err, "finite and >= 0")

    def test_malformed_json_is_refused(self):
        reference = self.write_jsonl("ref.jsonl", [record("p", 0, [0.0])])
        candidate = self.dir / "cand.json"
        candidate.write_text("{not json", encoding="utf-8")
        status, out, err = self.run_gate(reference, candidate)
        self.assert_refused(status, out, err, "not valid JSON")

    def test_json_array_and_captures_object_are_accepted(self):
        reference = self.write_json("ref.json", [record("p", 0, [0.0, 0.0])])
        candidate = self.write_json("cand.json", {"captures": [record("p", 0, [0.0, 0.0])]})
        status, out, err = self.run_gate(reference, candidate)
        self.assertEqual(status, 0, out + err)

    def test_multiple_candidates_are_all_reported(self):
        reference = self.write_jsonl("ref.jsonl", [record("p", 0, [0.0, 0.0])])
        good = self.write_jsonl("good.jsonl", [record("p", 0, [0.1, 0.0])])
        bad = self.write_jsonl("bad.jsonl", [record("p", 0, [0.0, 0.5])])
        status, out, err = self.run_gate(reference, good, bad, "--reference-delta", "0.5")
        self.assertEqual(status, 1, out + err)
        self.assertIn(f"candidate: {good}", out)
        self.assertIn(f"candidate: {bad}", out)
        self.assertEqual(out.count("verdict: PASS"), 1)
        self.assertEqual(out.count("verdict: FAIL"), 1)
        self.assertIn("OVERALL: FAIL", out)

    def test_json_report_has_the_gate_fields(self):
        reference = self.write_jsonl("ref.jsonl", [record("p", 0, [0.0, 0.0])])
        candidate = self.write_jsonl("cand.jsonl", [record("p", 0, [0.25, 0.0])])
        status, out, err = self.run_gate(
            reference, candidate, "--reference-delta", "0.25", "--json"
        )
        self.assertEqual(status, 0, out + err)
        report = json.loads(out)
        self.assertTrue(report["passed"])
        self.assertEqual(report["noise_floor"]["reference_delta"], 0.25)
        self.assertEqual(report["noise_floor"]["floor"], 0.5)
        self.assertEqual(report["candidates"][0]["max_abs_delta"], 0.25)
        self.assertEqual(report["candidates"][0]["top1_divergences"], [])
        self.assertTrue(report["candidates"][0]["within_floor"])


if __name__ == "__main__":
    unittest.main()
