#!/usr/bin/env python3
"""T28 tests: the distinct-expert estimator's predictions and its parser.

The estimator is a measurement tool, so the failure that matters is a silent
one: a parser that drops layers, or a prediction that is the naive topK for
every window so the reported error says nothing. Both are pinned here with
synthetic traces whose next-window union is known exactly.

    cd benchmark && python3 -m unittest test_t28_distinct_expert_estimator -v
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

BENCH = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))

import t28_distinct_expert_estimator as estimator  # noqa: E402


class TraceParsingTests(unittest.TestCase):
    def test_reads_positions_layers_and_experts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = pathlib.Path(directory) / "trace.txt"
            trace.write_text(
                "0 0 1 2\n0 1 3\n1 0 2 4\n1 1 4\n",
                encoding="utf-8",
            )
            by_layer = estimator.read_trace(trace)
        self.assertEqual(set(by_layer), {0, 1})
        self.assertEqual(
            by_layer[0],
            [(0, frozenset({1, 2})), (1, frozenset({2, 4}))],
        )
        self.assertEqual(by_layer[1], [(0, frozenset({3})), (1, frozenset({4}))])

    def test_ignores_short_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = pathlib.Path(directory) / "trace.txt"
            trace.write_text("0 0\n1 0 1\n", encoding="utf-8")
            by_layer = estimator.read_trace(trace)
        self.assertEqual(by_layer, {0: [(1, frozenset({1}))]})


class PredictionTests(unittest.TestCase):
    def test_overlap_is_predicted_better_than_the_naive_baseline(self) -> None:
        # Two routes alternate. The next window's union is {1,2,3} (3 experts)
        # while a no-growth predictor says 2.
        entries = [
            (position, frozenset({1, 2} if position % 2 == 0 else {2, 3})) for position in range(40)
        ]
        report = estimator.evaluate({0: entries}, window=2, split=0.5)
        self.assertGreater(report["predictions"], 0)
        self.assertEqual(report["mae"], 0.0)
        self.assertAlmostEqual(report["baseline_mae"], 1.0, places=6)

    def test_a_repeated_route_predicts_top_k(self) -> None:
        entries = [(position, frozenset({7, 8})) for position in range(20)]
        report = estimator.evaluate({0: entries}, window=2, split=0.5)
        self.assertEqual(report["mae"], 0.0)
        self.assertEqual(report["baseline_mae"], 0.0)

    def test_no_predictions_before_there_is_history(self) -> None:
        entries = [(position, frozenset({position + 1})) for position in range(6)]
        report = estimator.evaluate({0: entries}, window=2, split=0.9)
        # The split leaves fewer than two positions, so there is no window to
        # score; reporting 0 predictions is honest, not a 0.0 MAE pass.
        self.assertEqual(report["predictions"], 0)
        self.assertIsNone(report["mae"])


if __name__ == "__main__":
    unittest.main()
