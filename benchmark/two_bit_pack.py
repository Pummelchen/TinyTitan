#!/usr/bin/env python3
"""The tiered converter's value transform: 4-bit packed weights to 2-bit.

The runtime expands a cold expert with `q4 = 5 * q2` and `scale / 5`, so the
converter does not need to dequantise or mint new scales at all. It keeps the
expert's existing scales and biases, and only remaps values:

    q2 = round(q4 / 5)

That is a pure byte transform, verified here against the expansion: expanding
the result reproduces `5 * q2`, and the induced error is at most 2 units of the
4-bit scale (half a 2-bit step).

The 4-bit stream packs two values per byte, low nibble first (`affine_quant_value`
shifts by `index * bits`, so element 0 is the low nibble). The 2-bit stream
packs four values per byte, most significant pair first, matching
`TwoBitExpand` in the runtime.
"""

from __future__ import annotations

import numpy as np

# q2 for each possible 4-bit value, as a byte lookup.
_Q4_TO_Q2 = np.rint(np.arange(16) / 5.0).astype(np.uint8)

# One input byte carries two 4-bit values; one output byte carries four 2-bit
# values. Precompute byte-pair -> output-byte so the inner loop is a lookup.
_PAIR_LUT = np.zeros(65536, dtype=np.uint8)
for _b0 in range(256):
    for _b1 in range(256):
        _v = (
            _Q4_TO_Q2[_b0 & 0xF],
            _Q4_TO_Q2[(_b0 >> 4) & 0xF],
            _Q4_TO_Q2[_b1 & 0xF],
            _Q4_TO_Q2[(_b1 >> 4) & 0xF],
        )
        _PAIR_LUT[(_b0 << 8) | _b1] = (_v[0] << 6) | (_v[1] << 4) | (_v[2] << 2) | _v[3]


def remap_4bit_to_2bit(data: bytes) -> bytes:
    """Remap a packed 4-bit weight stream to 2-bit. `len(data)` must be even."""
    if len(data) % 2:
        raise ValueError("4-bit stream must be an even number of bytes")
    if not data:
        return b""
    source = np.frombuffer(data, dtype=np.uint8)
    pairs = (source[0::2].astype(np.uint16) << 8) | source[1::2].astype(np.uint16)
    return _PAIR_LUT[pairs].tobytes()


def expanded_q4(data: bytes) -> np.ndarray:
    """The 4-bit values a 2-bit stream expands to: `5 * q2`, low pair first."""
    out = []
    for byte in data:
        out.extend((5 * ((byte >> 6) & 0x3), 5 * ((byte >> 4) & 0x3),
                    5 * ((byte >> 2) & 0x3), 5 * (byte & 0x3)))
    return np.array(out, dtype=np.int64)


# --- 3-bit cold tier -------------------------------------------------------
#
# A 3-bit value expands to 4-bit with q4 = 2*q3 and scale/2, so
# (2*q3)*(s/2) + b == q3*s + b. Eight values fill exactly three bytes
# (8 x 3 bits = 24), which keeps every group byte-aligned and avoids the
# word-crossing that made 6-bit unattractive.

def remap_4bit_to_3bit(data: bytes) -> bytes:
    """Remap a packed 4-bit weight stream to packed 3-bit.

    `len(data)` must be a multiple of 4: four 4-bit bytes carry eight values,
    which become three 3-bit bytes.
    """
    if len(data) % 4:
        raise ValueError("4-bit stream length must be a multiple of 4 bytes")
    source = np.frombuffer(data, dtype=np.uint8)
    values = np.empty(source.size * 2, dtype=np.uint8)
    values[0::2] = source & 0x0F          # element 2k is the low nibble
    values[1::2] = source >> 4
    # round(q4/2), clamped to 7: q4=15 would round to 8, which does not
    # fit three bits; 14 is one step away, the same error as any other
    # rounding boundary.
    q3 = np.minimum(7, (values.astype(np.uint16) + 1) // 2).astype(np.uint8)
    groups = q3.reshape(-1, 8)
    # Eight 3-bit values fill one 24-bit little-endian word: value i occupies
    # bits [3i, 3i+3), so a group is exactly three bytes with no crossings.
    word = np.zeros(groups.shape[0], dtype=np.uint32)
    for index in range(8):
        word |= groups[:, index].astype(np.uint32) << (3 * index)
    packed = np.empty((groups.shape[0], 3), dtype=np.uint8)
    packed[:, 0] = word & 0xFF
    packed[:, 1] = (word >> 8) & 0xFF
    packed[:, 2] = (word >> 16) & 0xFF
    return packed.tobytes()


def expanded_3bit_q4(data: bytes) -> np.ndarray:
    """The 4-bit values a 3-bit stream expands to: 2*q3, low pair first."""
    out = []
    for index in range(0, len(data), 3):
        word = data[index] | (data[index + 1] << 8) | (data[index + 2] << 16)
        for slot in range(8):
            out.append(2 * ((word >> (slot * 3)) & 0x7))
    return np.array(out, dtype=np.int64)
