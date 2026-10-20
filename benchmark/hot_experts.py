#!/usr/bin/env python3
"""Emit the per-layer hot-expert list a tiered repacker consumes.

Input is a route trace (`position layer e1..e10`). Output is the profile the
4-bit/2-bit split needs: for each layer, the experts kept at the wide tier,
chosen by selection frequency, plus the activation mass they carry so the
caller can see what the split costs before repacking.

    TINYTITAN_ROUTE_TRACE=/tmp/route.txt <server> ...
    python benchmark/hot_experts.py --trace /tmp/route.txt --hot 0.5 \
        --out /tmp/hot.json

The split fraction follows the measured decision in
docs/residency-and-bottleneck-verdict-2026-10-20.md: 0.4-0.5 hot keeps 13-7.5%
of activations on the 2-bit tier for a 30-25% byte saving.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument(
        "--hot", type=float, default=0.5, help="fraction of experts per layer kept at the wide tier"
    )
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if not 0.0 < args.hot < 1.0:
        print("--hot must be between 0 and 1")
        return 1

    counts: dict[int, collections.Counter[int]] = collections.defaultdict(collections.Counter)
    totals: collections.Counter[int] = collections.Counter()
    for line in pathlib.Path(args.trace).open(encoding="utf-8"):
        parts = line.split()
        if len(parts) < 3:
            continue
        layer = int(parts[1])
        for token in parts[2:]:
            counts[layer][int(token)] += 1
            totals[layer] += 1
    if not counts:
        print("no routes in trace")
        return 1

    keep = max(1, math.ceil(args.hot * args.experts))
    layers = {}
    covered_total = 0
    selected_total = 0
    for layer in sorted(counts):
        ranked = [expert for expert, _ in counts[layer].most_common(keep)]
        covered = sum(counts[layer][e] for e in ranked)
        layers[layer] = {
            "hot": sorted(ranked),
            "hotCount": len(ranked),
            "coldCount": args.experts - len(ranked),
            "activationShare": round(covered / totals[layer], 4),
        }
        covered_total += covered
        selected_total += totals[layer]

    header = {
        "trace": args.trace,
        "hotFraction": args.hot,
        "expertsPerLayer": args.experts,
        "activationShareOnHot": round(covered_total / selected_total, 4),
        "activationShareOnCold": round(1 - covered_total / selected_total, 4),
        "effectiveBitsPerExpertWeight": round(args.hot * 4.5 + (1 - args.hot) * 2.25, 4),
        "bytesVsFourBit": round((args.hot * 4.5 + (1 - args.hot) * 2.25) / 4.5, 4),
        "layers": layers,
    }
    pathlib.Path(args.out).write_text(json.dumps(header, separators=(",", ":")), encoding="utf-8")
    print(
        f"layers {len(layers)}  hot {keep}/{args.experts} per layer  "
        f"activation on hot {header['activationShareOnHot']:.3f}  "
        f"bytes {header['bytesVsFourBit']:.3f}x  -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
