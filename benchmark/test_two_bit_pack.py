#!/usr/bin/env python3
"""Tests for the tiered converter's 4-bit to 2-bit remap."""

from __future__ import annotations

import unittest

import numpy as np

from two_bit_pack import expanded_q4, remap_4bit_to_2bit


class TwoBitPackTests(unittest.TestCase):
    def test_every_four_bit_value_maps_to_its_nearest_quarter(self) -> None:
        # Build the 4-bit stream low-nibble-first: values 0..15 in order.
        values = list(range(16))
        stream = bytes((values[i] & 0xF) | ((values[i + 1] & 0xF) << 4) for i in range(0, 16, 2))
        two_bit = remap_4bit_to_2bit(stream)
        expanded = expanded_q4(two_bit)
        expected = np.rint(np.array(values) / 5.0).astype(np.int64) * 5
        np.testing.assert_array_equal(expanded, expected)
        # Rounding error is never more than half a 2-bit step (2.5 -> 2).
        self.assertLessEqual(int(np.abs(np.array(values) - expanded).max()), 2)

    def test_a_random_stream_round_trips_through_the_expansion(self) -> None:
        rng = np.random.default_rng(7)
        source = rng.integers(0, 256, size=4096, dtype=np.uint8).tobytes()
        two_bit = remap_4bit_to_2bit(source)
        self.assertEqual(len(two_bit), len(source) // 2)
        expanded = expanded_q4(two_bit)
        # Recover the source 4-bit values, low nibble first, and compare.
        original = []
        for byte in source:
            original.extend((byte & 0xF, (byte >> 4) & 0xF))
        expected = np.rint(np.array(original) / 5.0).astype(np.int64) * 5
        np.testing.assert_array_equal(expanded, expected)

    def test_an_odd_length_stream_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            remap_4bit_to_2bit(b"\x00\x01\x02")


if __name__ == "__main__":
    unittest.main()
