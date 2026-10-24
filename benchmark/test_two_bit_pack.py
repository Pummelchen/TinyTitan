#!/usr/bin/env python3
"""Tests for the tiered converter's 4-bit to 2-bit remap."""

from __future__ import annotations

import unittest

import numpy as np

from two_bit_pack import (
    rescale_bf16,
    expanded_q4,
    native_2bit_values,
    remap_4bit_to_2bit,
    remap_4bit_to_2bit_native,
)


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


class NativeTwoBitPackTests(unittest.TestCase):
    """The lowmem profile's kernel-order packing."""

    def test_native_order_matches_the_value_order_it_came_from(self) -> None:
        values = list(range(16))
        stream = bytes((values[i] & 0xF) | ((values[i + 1] & 0xF) << 4) for i in range(0, 16, 2))
        packed = remap_4bit_to_2bit_native(stream)
        self.assertEqual(len(packed), len(stream) // 2)
        expected = np.rint(np.array(values) / 5.0).astype(np.int64)
        np.testing.assert_array_equal(native_2bit_values(packed), expected)

    def test_native_and_expansion_streams_hold_the_same_values(self) -> None:
        rng = np.random.default_rng(11)
        source = rng.integers(0, 256, size=2048, dtype=np.uint8).tobytes()
        native = remap_4bit_to_2bit_native(source)
        expansion = remap_4bit_to_2bit(source)
        # Same quarter values, different byte order: low pair first for the
        # kernel, high pair first for the expander.
        np.testing.assert_array_equal(native_2bit_values(native), expanded_q4(expansion) // 5)
        self.assertNotEqual(native, expansion)

    def test_a_native_odd_length_stream_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            remap_4bit_to_2bit_native(b"\x00\x01\x02")

    def test_scaling_by_five_makes_the_remap_decode_like_the_source(self) -> None:
        def bf16(value: float) -> int:
            raw = np.float32(value).view(np.uint32)
            lsb = (raw >> 16) & 1
            return int((raw + 0x7FFF + lsb) >> 16)

        scales = [0.5, 1.0, 7.5, 0.03125]
        source = b"".join(int(bf16(s)).to_bytes(2, "little") for s in scales)
        scaled = rescale_bf16(source, 5.0)
        raw = np.frombuffer(scaled, dtype=np.uint16).astype(np.uint32) << 16
        for original, out in zip(scales, raw.view(np.float32), strict=False):
            # The stored scale is bf16, so the product is within one bf16 ulp
            # of the exact 5x. `q2 * (5s)` is what reproduces the source group.
            self.assertLessEqual(abs(float(out) - original * 5.0), max(1e-6, original * 5.0 * 1e-3))

    def test_an_odd_length_scale_block_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            rescale_bf16(b"\x00\x01\x02", 5.0)


if __name__ == "__main__":
    unittest.main()
