#!/usr/bin/env python3
"""How much of the oracle's placement headroom a *practical* hot set recovers.

The engine's expert cache is at its measured optimum count, and the tracker
records that Belady over the same slots reaches +8.4 hit points -- headroom no
online evictor can reach, because it needs the future. The question this script
answers is the useful one: how much of that gap does a **static per-layer hot
set**, learned offline from a routing trace, actually recover?

It replays a committed route trace (`position layer e1..e10`) against four
policies at a fixed slot count:

    lfu      the shipped policy (least frequently used, all-time counts)
    lru      least recently used
    pinK     the K most-requested experts of that layer are pinned; the rest are
             LRU. K comes from the trace's own frequency table, which is the
             "offline profile" a real deployment would ship.
    belady   the oracle: evict whichever expert is used farthest in the future

Read the output as *relative*: a trace has no prefetch adoption, so its absolute
hit rate is below the runtime's (the engine's ring converts misses into hits, and
`benchmark/expert_cache_slots.py` records the runtime disagreeing with simulated
LRU for exactly that reason). What transfers is the ordering of policies and the
fraction of the oracle gap a pinned set closes.

    python3 benchmark/expert_placement_sim.py \
        --trace benchmark/expert-cache/route-trace-qwen38-4bit-v520-20261015.txt.gz \
        --slots 96 --pins 24 48 72
"""

from __future__ import annotations

import argparse
import collections
import gzip
import pathlib


def load_trace(path: pathlib.Path) -> dict[int, list[list[int]]]:
    """layer -> ordered list of the expert sets requested at each position."""
    layers: dict[int, dict[int, list[int]]] = collections.defaultdict(dict)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            # position, layer, then however many experts the model routes to
            # (top-8 on Ornith, top-10 on Qwen3.8), so the width is not a constant.
            if len(parts) < 3:
                continue
            position, layer = int(parts[0]), int(parts[1])
            layers[layer][position] = [int(x) for x in parts[2:]]
    return {layer: [rows[p] for p in sorted(rows)] for layer, rows in layers.items()}


def replay(
    sequence: list[list[int]], slots: int, policy: str, pinned: set[int] | None = None
) -> tuple[int, int]:
    """(hits, requests) for one layer under one policy.

    `lfu` models the engine's policy, not a textbook one: the engine breaks an
    equal-count tie by *last use*, and without that tie-break a pure count rank
    evicts the expert it just admitted whenever everything sits at count 1 --
    which is where a simulation of naive LFU loses ~25 points to LRU and the
    shipped engine does not.
    """
    pinned = pinned or set()
    resident: set[int] = set()
    count: dict[int, int] = {}
    last: dict[int, int] = {}
    hits = 0

    future: dict[int, list[int]] = collections.defaultdict(list)
    if policy == "belady":
        for position, experts in enumerate(sequence):
            for expert in experts:
                future[expert].append(position)
    cursor: dict[int, int] = collections.defaultdict(int)

    for position, experts in enumerate(sequence):
        for expert in experts:
            if expert in resident:
                hits += 1
                count[expert] = count.get(expert, 0) + 1
                last[expert] = position
                continue
            if len(resident) >= slots:
                candidates = [e for e in resident if e not in pinned]
                if not candidates:
                    continue
                if policy == "belady":
                    victim, farthest = None, -1
                    for e in candidates:
                        occurrences = future[e]
                        index = cursor[e]
                        while index < len(occurrences) and occurrences[index] <= position:
                            index += 1
                        cursor[e] = index
                        nxt = occurrences[index] if index < len(occurrences) else 10**9
                        if nxt > farthest:
                            victim, farthest = e, nxt
                elif policy == "lru" or policy == "pin":
                    victim = min(candidates, key=lambda e: last[e])
                else:  # lfu, with the engine's recency tie-break
                    victim = min(candidates, key=lambda e: (count[e], last[e]))
                if victim is None:
                    continue
                resident.discard(victim)
            resident.add(expert)
            count[expert] = count.get(expert, 0) + 1
            last[expert] = position
    return hits, sum(len(experts) for experts in sequence)


def top_experts(sequence: list[list[int]], count: int) -> set[int]:
    counts: collections.Counter[int] = collections.Counter()
    for experts in sequence:
        counts.update(experts)
    return {expert for expert, _ in counts.most_common(count)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--slots", type=int, default=96)
    parser.add_argument("--pins", type=int, nargs="+", default=[24, 48, 72])
    parser.add_argument(
        "--train-trace",
        default=None,
        help="learn the pinned hot set from THIS trace and score on --trace. Without "
        "it the pins are learned from the scored trace, which is optimistic: a real "
        "deployment learns the profile from one session and spends it on another",
    )
    arguments = parser.parse_args()

    trace_path = pathlib.Path(arguments.trace)
    layers = load_trace(trace_path)
    positions = len(next(iter(layers.values())))
    print(
        f"trace {trace_path.name}: {len(layers)} layers x {positions} positions, "
        f"{sum(len(s) for s in layers.values())} routed rows"
    )

    results: dict[str, float] = {}
    for policy in ("lfu", "lru", "belady"):
        hits = reqs = 0
        for sequence in layers.values():
            h, r = replay(sequence, arguments.slots, policy)
            hits += h
            reqs += r
        results[policy] = hits / reqs * 100
        print(f"  {policy:<8} {results[policy]:5.2f}% hit ({hits}/{reqs})")

    trained = load_trace(pathlib.Path(arguments.train_trace)) if arguments.train_trace else None
    if trained is not None:
        print(f"  pins learned from {pathlib.Path(arguments.train_trace).name}")
    for pin in arguments.pins:
        if pin >= arguments.slots:
            continue
        hits = reqs = 0
        for layer, sequence in layers.items():
            source = trained.get(layer) if trained is not None else sequence
            pinned = top_experts(source, pin) if source is not None else set()
            h, r = replay(sequence, arguments.slots, "pin", pinned)
            hits += h
            reqs += r
        results[f"pin{pin}"] = hits / reqs * 100
        print(f"  pin{pin:<5} {results[f'pin{pin}']:5.2f}% hit ({hits}/{reqs})")

    oracle = results["belady"]
    shipped = results["lfu"]
    print(
        f"\nslots={arguments.slots}: shipped lfu {shipped:.2f}%, "
        f"oracle {oracle:.2f}% (+{oracle - shipped:.2f} points)"
    )
    for pin in arguments.pins:
        key = f"pin{pin}"
        if key in results and oracle > shipped:
            recovered = (results[key] - shipped) / (oracle - shipped) * 100
            print(f"  {key}: {results[key]:.2f}% -- recovers {recovered:+.1f}% of the oracle gap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
