#!/usr/bin/env python3
"""The ANE exporter's history list refuses what it cannot export.

`docs/adding-a-model.md` step 8 is the only thing that names this tool, and the
tool imports Core ML at module scope, so its argument arithmetic is extracted
into `tools/ane_export_args.py` where a stock `python3` can test it. The shape
that mattered: `range(0, max_history + 1, chunk)` is *empty* for a negative
`--max-history` that is a multiple of the chunk, so the exporter went on to
build a multi-function package with no functions and the failure named the
conversion instead of the argument (AUD-0117).
"""

from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import ane_export_args  # noqa: E402


class HistoryVariantsTests(unittest.TestCase):
    def test_the_default_shape_is_every_chunk_boundary(self):
        self.assertEqual(
            ane_export_args.history_variants(12288, 4096),
            [0, 4096, 8192, 12288],
        )

    def test_zero_history_is_one_zero_variant(self):
        self.assertEqual(ane_export_args.history_variants(0, 4096), [0])

    def test_a_negative_history_is_refused_by_name(self):
        # The defect: -4096 is a multiple of the chunk, so the old check passed
        # and the list was empty. An export with no variants is not an export.
        with self.assertRaises(ValueError) as raised:
            ane_export_args.history_variants(-4096, 4096)
        self.assertIn("--max-history", str(raised.exception))
        self.assertIn("-4096", str(raised.exception))

    def test_a_negative_history_is_refused_even_when_not_a_multiple(self):
        with self.assertRaises(ValueError):
            ane_export_args.history_variants(-1, 4096)

    def test_a_non_multiple_is_refused(self):
        with self.assertRaises(ValueError) as raised:
            ane_export_args.history_variants(1000, 4096)
        self.assertIn("multiple of --chunk", str(raised.exception))

    def test_a_zero_chunk_is_refused(self):
        with self.assertRaises(ValueError):
            ane_export_args.history_variants(4096, 0)


if __name__ == "__main__":
    unittest.main()
