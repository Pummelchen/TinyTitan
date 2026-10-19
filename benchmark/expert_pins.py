#!/usr/bin/env python3
"""Derive a per-layer expert pin file from a route trace.

The trace is the engine's own `TINYTITAN_ROUTE_TRACE` output — `position layer
e1..eK` — so this needs no new capture format. The output is the file
`ExpertPins` reads: one line per layer, `layer expert expert ...`, holding that
layer's most-requested experts.

Keep the profile and the workload apart. A pin file learned from the trace of
one prompt and spent on the same prompt measures the file against itself; the
simulation that justified this (`benchmark/expert_placement_sim.py`) deliberately
scored pins learned on a *different* session, and the same discipline applies
here.

    TINYTITAN_ROUTE_TRACE=/tmp/route.txt TinyTitanServer ...   # capture
    python3 benchmark/expert_pins.py --trace /tmp/route.txt --pins 48 --out /tmp/pins.txt
    TINYTITAN_EXPERT_PINS=/tmp/pins.txt TinyTitanServer ...    # spend
"""

from __future__ import annotations

import argparse
import collections
import gzip
import pathlib


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--pins", type=int, default=48, help="experts pinned per layer")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--positions",
        type=int,
        default=0,
        help="learn from the first N positions only (0 = the whole trace). Useful "
        "to keep the profile away from the positions it is scored on",
    )
    arguments = parser.parse_args()

    trace = pathlib.Path(arguments.trace)
    opener = gzip.open if trace.suffix == ".gz" else open
    counts: dict[int, collections.Counter[int]] = collections.defaultdict(collections.Counter)
    with opener(trace, "rt", encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 3:
                continue
            position, layer = int(fields[0]), int(fields[1])
            if arguments.positions and position >= arguments.positions:
                continue
            counts[layer].update(int(x) for x in fields[2:])

    lines = []
    for layer in sorted(counts):
        hot = [expert for expert, _ in counts[layer].most_common(arguments.pins)]
        lines.append(" ".join(str(x) for x in [layer, *hot]))
    pathlib.Path(arguments.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"{len(lines)} layers x {arguments.pins} pins -> {arguments.out} "
        f"(learned from {sum(sum(c.values()) for c in counts.values())} routed rows)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
