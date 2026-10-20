#!/usr/bin/env python3
"""Hot/cold split from a committed route trace.

A route trace line is `position layer e1..e10` (see
`RealForwardRunner+Diagnostics.recordRouteTrace`). For each candidate fraction
of experts kept at 4-bit, this reports the share of *activations* those experts
carry and the effective bits per weight when the rest go to 2-bit.

    TINYTITAN_ROUTE_TRACE=/tmp/route.txt <server> ...
    python benchmark/route_skew.py --trace /tmp/route.txt

Effective widths include the scale/bias overhead the format actually pays:
4-bit at group 64 is 4.5 bits per weight, 2-bit at group 128 is 2.25.
"""

from __future__ import annotations

import argparse
import collections
import math
import pathlib


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--hot-bits", type=float, default=4.5)
    ap.add_argument("--cold-bits", type=float, default=2.25)
    args = ap.parse_args()

    counts: dict[int, collections.Counter[int]] = collections.defaultdict(
        collections.Counter
    )
    selections: collections.Counter[int] = collections.Counter()
    for line in pathlib.Path(args.trace).open():
        parts = line.split()
        if len(parts) < 3:
            continue
        layer = int(parts[1])
        for token in parts[2:]:
            counts[layer][int(token)] += 1
            selections[layer] += 1
    if not counts:
        print("no routes in trace")
        return 1

    print(f"layers {len(counts)}  experts {args.experts}")
    print(
        f"{'hot frac':>8} {'hot experts':>12} {'hot act. mass':>14} "
        f"{'cold act. mass':>15} {'eff bits':>9} {'vs 4-bit':>9}"
    )
    for fraction in (0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75):
        coverages = []
        for layer, counter in counts.items():
            total = selections[layer]
            if total == 0:
                continue
            keep = max(1, math.ceil(fraction * args.experts))
            covered = sum(v for _, v in counter.most_common(keep))
            coverages.append(covered / total)
        mass = sum(coverages) / len(coverages)
        bits = fraction * args.hot_bits + (1 - fraction) * args.cold_bits
        print(
            f"{fraction:>8.2f} {int(fraction * args.experts):>12} "
            f"{mass:>14.3f} {1 - mass:>15.3f} {bits:>9.3f} "
            f"{1 - bits / args.hot_bits:>8.1%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
