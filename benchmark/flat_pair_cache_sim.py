"""Offline simulator for the Qwen3.8 routed-expert cache, stage 0 of
docs/plan-qwen38-decode-12gb.md.

Input is a TINYTITAN_ROUTE_TRACE capture (`position layer e0 e1 ...` per
decode layer). Three policies replay over the same trace and the same total
budget:

    uniform  today's per-layer slot cache, LFU with LRU tiebreak (control)
    flat     one global pool of (layer, expert) pairs
    heat     per-layer slot counts proportional to observed demand

Every tok/s figure this prints is a PROJECTION through the measured token
model (GPU 105 ms, turnaround 20 ms, SSD 3.6 GB/s), not a measurement.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

GPU_MS = 105.0
TURNAROUND_MS = 20.0
SSD_BYTES_PER_SEC = 3.6e9


class TraceFormatError(Exception):
    """The trace file is missing, empty, or malformed at a named line."""


@dataclass
class SimResult:
    policy: str
    hits: int
    misses: int
    per_layer_slots: dict = field(default_factory=dict)
    per_layer_distinct: dict = field(default_factory=dict)

    @property
    def requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.requests if self.requests else 0.0


def parse_trace(path: str) -> list[tuple[int, int, list[int]]]:
    try:
        with open(path, encoding="utf-8") as handle:
            raw_lines = handle.read().splitlines()
    except OSError as error:
        raise TraceFormatError(f"cannot read trace {path}: {error}") from error
    if not raw_lines:
        raise TraceFormatError(f"trace {path} is empty")
    events: list[tuple[int, int, list[int]]] = []
    for number, line in enumerate(raw_lines, start=1):
        if not line.strip():
            continue
        fields = line.split()
        try:
            position = int(fields[0])
            layer = int(fields[1])
            experts = [int(value) for value in fields[2:]]
        except ValueError as error:
            raise TraceFormatError(
                f"trace {path} line {number} has a field that is not an integer: {error!r}"
            ) from error
        if len(experts) == 0:
            raise TraceFormatError(f"trace {path} line {number} names no experts")
        if any(value < 0 for value in experts):
            raise TraceFormatError(f"trace {path} line {number} has a negative expert id")
        events.append((position, layer, experts))
    if not events:
        raise TraceFormatError(f"trace {path} holds no events")
    return events


def _replay_lfu(
    events: list[tuple[int, int, list[int]]],
    capacity_for: callable,
    pool_for: callable,
) -> tuple[int, int, dict]:
    """LFU with LRU tiebreak, per pool chosen by `pool_for`.

    Mirrors the engine's eviction (`shouldEvictSlot`): the least-frequent
    expert goes first, ties fall to least-recent use. One `use` is one event
    naming the expert, matching one cache plan. `pool_for` maps a layer to a
    pool id — the layer itself for the uniform/heat policies, one constant
    for the flat policy.
    """
    hits = 0
    misses = 0
    counts: dict = {}
    last_use: dict = {}
    resident: dict = {}
    clock = 0
    for _, layer, experts in events:
        pool = pool_for(layer)
        capacity = capacity_for(pool)
        held = resident.setdefault(pool, {})
        for expert in experts:
            clock += 1
            key = (layer, expert)
            counts[key] = counts.get(key, 0) + 1
            last_use[key] = clock
            if key in held:
                hits += 1
                continue
            misses += 1
            if len(held) < capacity:
                held[key] = True
                continue
            victim = min(held, key=lambda k: (counts[k], last_use[k]))
            del held[victim]
            held[key] = True
    return hits, misses, counts


def simulate_uniform(
    trace: list[tuple[int, int, list[int]]],
    slots_per_layer: int,
    experts_per_layer: int,
) -> SimResult:
    if slots_per_layer <= 0:
        raise ValueError("slots_per_layer must be positive")
    if experts_per_layer <= 0:
        raise ValueError("experts_per_layer must be positive")
    distinct: dict[int, set] = {}
    for _, layer, experts in trace:
        distinct.setdefault(layer, set()).update(experts)
    hits, misses, _ = _replay_lfu(trace, lambda layer: slots_per_layer, lambda layer: layer)
    return SimResult(
        policy="uniform",
        hits=hits,
        misses=misses,
        per_layer_slots={layer: slots_per_layer for layer in distinct},
        per_layer_distinct={layer: len(seen) for layer, seen in distinct.items()},
    )


def simulate_flat(
    trace: list[tuple[int, int, list[int]]],
    total_pairs: int,
    experts_per_layer: int,
) -> SimResult:
    if total_pairs <= 0:
        raise ValueError("total_pairs must be positive")
    if experts_per_layer <= 0:
        raise ValueError("experts_per_layer must be positive")
    distinct: dict[int, set] = {}
    for _, layer, experts in trace:
        distinct.setdefault(layer, set()).update(experts)
    keys_seen = set()

    def capacity_for(_pool: int) -> int:
        return total_pairs

    hits, misses, _ = _replay_lfu(trace, capacity_for, lambda layer: 0)
    for _, layer, experts in trace:
        for expert in experts:
            keys_seen.add((layer, expert))
    return SimResult(
        policy="flat",
        hits=hits,
        misses=misses,
        per_layer_slots={},
        per_layer_distinct={layer: len(seen) for layer, seen in distinct.items()},
    )


def allocate_heat_slots(
    per_layer_distinct: dict,
    total_slots: int,
    experts_per_layer: int,
) -> dict:
    """Slots to the layer with the largest unmet demand, ties to the lower
    index. Deterministic, sum <= total_slots, never above the cap."""
    if total_slots <= 0:
        raise ValueError("total_slots must be positive")
    if experts_per_layer <= 0:
        raise ValueError("experts_per_layer must be positive")
    demand = {
        layer: min(int(count), experts_per_layer)
        for layer, count in per_layer_distinct.items()
        if int(count) > 0
    }
    allocation = {layer: 0 for layer in demand}
    remaining = total_slots
    while remaining > 0:
        best_layer = None
        best_ratio = 0.0
        for layer in sorted(demand):
            deficit = demand[layer] - allocation[layer]
            if deficit <= 0:
                continue
            ratio = deficit / demand[layer]
            if ratio > best_ratio:
                best_ratio = ratio
                best_layer = layer
        if best_layer is None:
            break
        allocation[best_layer] += 1
        remaining -= 1
    return allocation


def simulate_heat(
    trace: list[tuple[int, int, list[int]]],
    total_pairs: int,
    experts_per_layer: int,
    layers: int,
) -> SimResult:
    if total_pairs <= 0:
        raise ValueError("total_pairs must be positive")
    if layers <= 0:
        raise ValueError("layers must be positive")
    distinct: dict[int, set] = {}
    for _, layer, experts in trace:
        if not 0 <= layer < layers:
            raise ValueError(f"trace names layer {layer}, outside 0..{layers - 1}")
        distinct.setdefault(layer, set()).update(experts)
    per_layer_distinct = {layer: len(seen) for layer, seen in distinct.items()}
    allocation = allocate_heat_slots(
        per_layer_distinct=per_layer_distinct,
        total_slots=total_pairs,
        experts_per_layer=experts_per_layer,
    )
    hits, misses, _ = _replay_lfu(
        trace, lambda layer: allocation.get(layer, 0), lambda layer: layer
    )
    return SimResult(
        policy="heat",
        hits=hits,
        misses=misses,
        per_layer_slots=allocation,
        per_layer_distinct=per_layer_distinct,
    )


def project_token_ms(
    exposed_bytes: float,
    gpu_ms: float = GPU_MS,
    turnaround_ms: float = TURNAROUND_MS,
    ssd_bytes_per_sec: float = SSD_BYTES_PER_SEC,
) -> float:
    if exposed_bytes < 0:
        raise ValueError("exposed_bytes must not be negative")
    if ssd_bytes_per_sec <= 0:
        raise ValueError("ssd_bytes_per_sec must be positive")
    return gpu_ms + turnaround_ms + exposed_bytes / ssd_bytes_per_sec * 1000.0


def _token_exposed_bytes(result: SimResult, expert_stride: int) -> int:
    """Bytes a token reads on average, assuming one pass over all layers."""
    layers = len(result.per_layer_distinct) or 1
    return round(result.misses * expert_stride / max(1, result.requests / layers))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay a route trace under three expert-cache policies."
    )
    parser.add_argument("trace", help="TINYTITAN_ROUTE_TRACE capture")
    parser.add_argument(
        "--slots-per-layer",
        type=int,
        default=None,
        help="uniform control width (default: derived from --total-pairs)",
    )
    parser.add_argument(
        "--total-pairs",
        type=int,
        default=None,
        help="total budget in expert slots (default: slots-per-layer x layers)",
    )
    parser.add_argument("--experts-per-layer", type=int, default=512)
    parser.add_argument(
        "--layers",
        type=int,
        default=None,
        help="layer count (default: highest layer in the trace + 1)",
    )
    parser.add_argument("--expert-stride", type=int, default=2_768_896)
    args = parser.parse_args(argv)

    try:
        trace = parse_trace(args.trace)
    except TraceFormatError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    layers = args.layers or max(layer for _, layer, _ in trace) + 1
    total_pairs = args.total_pairs
    if total_pairs is None:
        total_pairs = (args.slots_per_layer or 64) * layers
    slots_per_layer = args.slots_per_layer or max(1, total_pairs // layers)

    try:
        uniform = simulate_uniform(
            trace, slots_per_layer=slots_per_layer, experts_per_layer=args.experts_per_layer
        )
        flat = simulate_flat(
            trace, total_pairs=total_pairs, experts_per_layer=args.experts_per_layer
        )
        heat = simulate_heat(
            trace,
            total_pairs=total_pairs,
            experts_per_layer=args.experts_per_layer,
            layers=layers,
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"trace events: {len(trace)}  layers: {layers}  budget: {total_pairs} slots")
    for result in (uniform, flat, heat):
        exposed = _token_exposed_bytes(result, args.expert_stride)
        token_ms = project_token_ms(exposed)
        rate = 1000.0 / token_ms
        print(
            f"{result.policy:8s} hits={result.hits} misses={result.misses} "
            f"hit_rate={result.hit_rate:.3f} "
            f"exposed/token={exposed / 1e6:.1f}MB "
            f"PROJECTION token={token_ms:.1f}ms rate={rate:.2f} tok/s"
        )
    print("note: rates are projections through the measured token model, not measurements")
    return 0


if __name__ == "__main__":
    sys.exit(main())
