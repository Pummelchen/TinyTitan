#!/usr/bin/env python3
"""Tests for benchmark/hot_experts.py — the tiered repacker's profile input.

Runs the script on a synthetic trace in a temp directory: no model, no network.
The arithmetic asserted here is the arithmetic the repacker will trust when it
decides which experts to widen.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

SCRIPT = pathlib.Path(__file__).with_name("hot_experts.py")
PYTHON = sys.executable


def run_profile(trace_text: str, hot: float, experts: int = 256) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        trace = pathlib.Path(tmp) / "trace.txt"
        out = pathlib.Path(tmp) / "hot.json"
        trace.write_text(trace_text)
        result = subprocess.run(
            [
                PYTHON,
                str(SCRIPT),
                "--trace",
                str(trace),
                "--hot",
                str(hot),
                "--experts",
                str(experts),
                "--out",
                str(out),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise AssertionError(f"exit {result.returncode}: {result.stderr}")
        return json.loads(out.read_text())


class HotExpertsTests(unittest.TestCase):
    def test_skewed_layer_keeps_its_frequent_experts(self) -> None:
        # Layer 0: expert 5 is chosen every token, expert 7 every other token,
        # the rest once. Four positions, topK 2.
        lines = []
        for position in range(4):
            second = 7 if position % 2 == 0 else 9
            lines.append(f"{position} 0 5 {second}\n")
        profile = run_profile("".join(lines), hot=0.01, experts=256)
        layer = profile["layers"]["0"]
        self.assertEqual(layer["hotCount"], 3)  # ceil(0.01 * 256) = 3
        self.assertIn(5, layer["hot"])
        self.assertIn(7, layer["hot"])
        # 5 appears 4 times, 7 twice, 9 twice: 8 of 8 selections are covered.
        self.assertAlmostEqual(layer["activationShare"], 1.0)
        self.assertAlmostEqual(profile["activationShareOnHot"], 1.0)
        self.assertEqual(profile["activationShareOnCold"], 0.0)

    def test_bytes_ratio_follows_the_split(self) -> None:
        lines = [f"{p} 0 {p % 8} {(p + 1) % 8}\n" for p in range(8)]
        profile = run_profile("".join(lines), hot=0.5)
        self.assertAlmostEqual(profile["effectiveBitsPerExpertWeight"], 3.375)
        self.assertAlmostEqual(profile["bytesVsFourBit"], 0.75)
        # The trace only ever selects 8 distinct experts, so the profile can
        # only widen those; the rest are cold by absence of evidence.
        self.assertEqual(profile["layers"]["0"]["hotCount"], 8)
        self.assertEqual(profile["layers"]["0"]["coldCount"], 248)

    def test_rejects_a_fraction_outside_the_unit_interval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = pathlib.Path(tmp) / "trace.txt"
            trace.write_text("0 0 1 2\n")
            for bad in ("0", "1"):
                result = subprocess.run(
                    [
                        PYTHON,
                        str(SCRIPT),
                        "--trace",
                        str(trace),
                        "--hot",
                        bad,
                        "--out",
                        str(pathlib.Path(tmp) / "x.json"),
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 1)
                self.assertIn("between 0 and 1", result.stdout)


if __name__ == "__main__":
    unittest.main()
