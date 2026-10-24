#!/usr/bin/env python3
"""Tests for the lowmem profile's per-layer converter."""

from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "benchmark"))

from tiered_repack import lowmem_layer  # noqa: E402
from two_bit_pack import native_2bit_values  # noqa: E402


def bf16(value: float) -> int:
    raw = np.float32(value).view(np.uint32)
    lsb = (raw >> 16) & 1
    return int((raw + 0x7FFF + lsb) >> 16)


def from_bf16(bits: list[int]) -> np.ndarray:
    raw = np.array(bits, dtype=np.uint32) << 16
    return raw.view(np.float32)


def packed_4bit(values: list[int]) -> bytes:
    out = bytearray((len(values) + 1) // 2)
    for index, value in enumerate(values):
        out[index // 2] |= (value & 0xF) << ((index % 2) * 4)
    return bytes(out)


class LowMemoryLayerTests(unittest.TestCase):
    rows = 2
    columns = 64
    stride = 256

    def make_layer(self) -> tuple[bytes, list[dict]]:
        """Two experts, each with gate/up/down at 4 bits, plus scales/biases."""
        weight_values = self.rows * self.columns
        weight_bytes = weight_values // 2
        aux_bytes = self.rows * 2
        tensors: dict[str, dict] = {}
        cursor = 0
        for role in ("gate", "up", "down"):
            tensors[role] = {
                "offset": cursor,
                "size": weight_bytes,
                "dtype": "U32",
                "shape": [self.rows, self.columns],
                "bits": 4,
            }
            cursor += weight_bytes
            tensors[f"{role}_scales"] = {
                "offset": cursor,
                "size": aux_bytes,
                "dtype": "BF16",
                "shape": [self.rows, 1],
            }
            cursor += aux_bytes
            tensors[f"{role}_biases"] = {
                "offset": cursor,
                "size": aux_bytes,
                "dtype": "BF16",
                "shape": [self.rows, 1],
            }
            cursor += aux_bytes

        payload = bytearray()
        experts = []
        for expert in range(2):
            blob = bytearray()
            for role_offset, (role, tensor) in enumerate(
                sorted(tensors.items(), key=lambda kv: kv[1]["offset"])
            ):
                if tensor["dtype"] == "U32":
                    # Distinct per expert and per role, covering 0...15.
                    values = [
                        (expert * 3 + role_offset + index) % 16 for index in range(weight_values)
                    ]
                    data = packed_4bit(values)
                elif role.endswith("scales"):
                    data = b"".join(
                        int(bf16(0.1 + expert * 0.05)).to_bytes(2, "little")
                        for _ in range(self.rows)
                    )
                else:
                    data = b"".join(int(bf16(0.5)).to_bytes(2, "little") for _ in range(self.rows))
                blob.extend(data)
            if len(blob) > self.stride:
                raise ValueError("fixture expert blob exceeds its slot")
            offset = len(payload)
            payload.extend(blob)
            payload.extend(b"\x00" * (self.stride - len(blob)))
            experts.append(
                {"expert": expert, "offset": offset, "size": self.stride, "tensors": tensors}
            )
        return bytes(payload), experts

    def test_every_expert_gets_its_own_native_two_bit_blob(self) -> None:
        source, experts = self.make_layer()
        data, entries, stride = lowmem_layer(source, experts)
        self.assertEqual(stride % 16_384, 0)
        self.assertEqual(len(entries), 2)
        self.assertEqual({entry["size"] for entry in entries}, {stride})
        self.assertEqual([entry["offset"] for entry in entries], [0, stride])
        self.assertEqual(len(data), 2 * stride)

        tensors = entries[0]["tensors"]
        for role in ("gate", "up", "down"):
            self.assertEqual(tensors[role]["bits"], 2)
        self.assertEqual(tensors["gate"]["size"], self.rows * self.columns // 4)

        # The two experts must not share weights: read expert 0 and 1 back and
        # compare their decoded values.
        def gate_values(entry: dict) -> list[int]:
            start = entry["offset"] + entry["tensors"]["gate"]["offset"]
            size = entry["tensors"]["gate"]["size"]
            return [int(v) for v in native_2bit_values(data[start : start + size])]

        self.assertNotEqual(gate_values(entries[0]), gate_values(entries[1]))
        # And the values are round(q4/5) of the source, in order.
        source_gate = [((0 * 3 + 0 + index) % 16) for index in range(self.rows * self.columns)]
        expected = [int(round(value / 5.0)) for value in source_gate]
        self.assertEqual(gate_values(entries[0]), expected)

    def test_scales_are_multiplied_by_five_and_decode_the_source_group(self) -> None:
        source, experts = self.make_layer()
        data, entries, _ = lowmem_layer(source, experts)
        tensor = entries[0]["tensors"]["gate_scales"]
        start = entries[0]["offset"] + tensor["offset"]
        new_scales = np.frombuffer(data[start : start + tensor["size"]], dtype=np.uint16)
        values = from_bf16(new_scales.tolist())
        for value in values:
            self.assertAlmostEqual(float(value), 5 * 0.1, delta=5 * 0.1 * 1e-3)

        # q2 * (5s) is the same product as (5*q2) * s, so the 4-bit source
        # group is reproduced within the remap's two-step bound.
        q4 = [0, 5, 10, 15]
        for source_q in q4:
            q2 = int(round(source_q / 5.0))
            self.assertLessEqual(abs(q2 * 5 * 0.1 - source_q * 0.1), 2 * 0.1 + 1e-6)

    def test_a_per_expert_size_difference_is_refused(self) -> None:
        source, experts = self.make_layer()
        experts[1]["tensors"] = dict(experts[1]["tensors"])
        expert_one_tensors = dict(experts[1]["tensors"])
        expert_one_tensors["gate"] = dict(expert_one_tensors["gate"])
        expert_one_tensors["gate"]["size"] -= 2
        experts[1]["tensors"] = expert_one_tensors
        with self.assertRaises(ValueError):
            lowmem_layer(source, experts)

    def test_an_eight_bit_source_is_refused(self) -> None:
        source, experts = self.make_layer()
        for role in ("gate", "up", "down"):
            experts[0]["tensors"][role]["bits"] = 8
        with self.assertRaises(ValueError):
            lowmem_layer(source, experts)


if __name__ == "__main__":
    unittest.main()
