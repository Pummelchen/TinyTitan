#!/usr/bin/env python3
"""T28: predict the distinct experts a draft window touches, per layer.

Reads a `TINYTITAN_ROUTE_TRACE` file (`position layer e0 e1 ...`, one decode
layer per line) and evaluates a Good-Turing-style estimator against the trace's
own later tokens:

    new per token  = mean |route_t / route_{t-1}|   (only after the split)
    predicted(W)   = topK + (W-1) * new per token    (capped at experts/layer)

A global Good-Turing estimate was tried first and was barely better than
predicting `topK`: 70% into a long trace most experts are still singletons, so
it predicts a near-worst-case window. Adjacent routes are the signal the draft
window actually has -- the draft continues the same context, so the previous
route's overlap is a much better predictor of the next one.

The first 70% of positions fit; the rest are scored by the union of the next W
routes against the prediction made before that window. The report includes the
naive "topK, no growth" baseline and W = 2/4, which is the draft-and-verify
window the controller would price.

    python3 benchmark/t28_distinct_expert_estimator.py \
        benchmark/expert-cache/route-trace-qwen38-4bit-20261011.txt.gz \
        --out .build/t28-estimator-qwen38.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import statistics
from collections import defaultdict


def read_trace(path: pathlib.Path) -> dict[int, list[tuple[int, frozenset[int]]]]:
    opener = gzip.open if path.suffix == ".gz" else open
    by_layer: dict[int, list[tuple[int, frozenset[int]]]] = defaultdict(list)
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 3:
                continue
            position, layer = int(fields[0]), int(fields[1])
            experts = frozenset(int(value) for value in fields[2:])
            by_layer[layer].append((position, experts))
    for entries in by_layer.values():
        entries.sort(key=lambda item: item[0])
    return dict(by_layer)


class LayerStats:
    def __init__(self) -> None:
        self.previous: frozenset[int] | None = None
        self.new_total = 0
        self.transitions = 0

    def observe(self, experts: frozenset[int]) -> None:
        if self.previous is not None:
            self.new_total += len(experts - self.previous)
            self.transitions += 1
        self.previous = experts

    def predict(self, top_k: int, experts_per_layer: int, window: int) -> float:
        if self.transitions == 0:
            return float(top_k)
        new_per_token = self.new_total / self.transitions
        predicted = top_k + (window - 1) * new_per_token
        return min(float(experts_per_layer), max(float(top_k), predicted))


def evaluate(
    by_layer: dict[int, list[tuple[int, frozenset[int]]]],
    window: int,
    split: float,
) -> dict:
    per_layer = []
    all_errors: list[float] = []
    baseline_errors: list[float] = []
    upper_errors: list[float] = []
    for layer, entries in sorted(by_layer.items()):
        top_k = max(len(experts) for _, experts in entries)
        experts_per_layer = max(expert for _, experts in entries for expert in experts) + 1
        positions = [position for position, _ in entries]
        split_position = positions[int(len(positions) * split)] if positions else 0
        stats = LayerStats()
        errors: list[float] = []
        baselines: list[float] = []
        uppers: list[float] = []
        for index, (position, experts) in enumerate(entries):
            if position >= split_position:
                ahead = entries[index : index + window]
                if len(ahead) == window:
                    actual = len(set().union(*(row[1] for row in ahead)))
                    predicted = stats.predict(top_k, experts_per_layer, window)
                    errors.append(abs(predicted - actual))
                    baselines.append(abs(top_k - actual))
                    uppers.append(abs(min(experts_per_layer, top_k * window) - actual))
            stats.observe(experts)
        if errors:
            per_layer.append(
                {
                    "layer": layer,
                    "top_k": top_k,
                    "predictions": len(errors),
                    "mae": statistics.fmean(errors),
                    "baseline_mae": statistics.fmean(baselines),
                    "upper_bound_mae": statistics.fmean(uppers),
                }
            )
            all_errors.extend(errors)
            baseline_errors.extend(baselines)
            upper_errors.extend(uppers)
    return {
        "window_tokens": window,
        "split_fraction": split,
        "layer_count": len(per_layer),
        "predictions": len(all_errors),
        "mae": statistics.fmean(all_errors) if all_errors else None,
        "baseline_mae": (statistics.fmean(baseline_errors) if baseline_errors else None),
        "upper_bound_mae": (statistics.fmean(upper_errors) if upper_errors else None),
        "per_layer": per_layer,
        "worst_layers": sorted(per_layer, key=lambda row: row["mae"], reverse=True)[:5],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=pathlib.Path)
    parser.add_argument("--split", type=float, default=0.7)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args()
    by_layer = read_trace(args.trace)
    top_k = max(len(experts) for entries in by_layer.values() for _, experts in entries)
    payload = {
        "trace": str(args.trace),
        "lines": sum(len(entries) for entries in by_layer.values()),
        "positions": max(position for entries in by_layer.values() for position, _ in entries) + 1,
        "layers": len(by_layer),
        "top_k": top_k,
        "windows": [evaluate(by_layer, window, args.split) for window in (2, 4)],
    }
    print(
        "T28 estimator "
        f"trace={args.trace.name} layers={payload['layers']} "
        f"positions={payload['positions']} top_k={top_k}"
    )
    for row in payload["windows"]:
        print(
            f"window={row['window_tokens']} predictions={row['predictions']} "
            f"mae={row['mae']:.3f} baseline_mae={row['baseline_mae']:.3f} "
            f"upper_bound_mae={row['upper_bound_mae']:.3f}"
        )
        worst = ", ".join(
            f"L{layer['layer']}={layer['mae']:.2f}" for layer in row["worst_layers"][:3]
        )
        print(f"  worst layers: {worst}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
