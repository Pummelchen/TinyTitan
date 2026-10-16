#!/usr/bin/env python3
"""Analyze the non-mutating TinyTitan v4.3 predictive-prefetch trace.

The trace is collected with ``TINYTITAN_PREFETCH_TRACE=/path/to/trace.jsonl``.
This tool intentionally models cheap, online predictors over that trace; it
never changes model routing or runtime cache state. Its purpose is to decide
whether a real, bounded staging-ring implementation has enough useful
nonresident-miss recall to warrant integration.

A screen is only worth what its ranking is worth at issue time, so every
predictor here reads strictly earlier positions and nothing else. That rule is
structural rather than conventional: the ``recent`` ranker's history is only
published at position boundaries, after every link at that position is scored.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
from dataclasses import dataclass
from typing import Iterable

History = dict[int, collections.deque]
TransitionCounts = dict[int, dict[int, collections.Counter]]


@dataclass(frozen=True)
class Observation:
    position: int
    layer: int
    experts: tuple[int, ...]
    misses: frozenset[int]
    resident: frozenset[int]
    next_layer_prediction: tuple[int, ...] = ()


@dataclass(frozen=True)
class Link:
    """One L -> L+1 plan, and only what an online predictor may know about it."""

    source: Observation
    target: Observation
    top_m: int
    counts: TransitionCounts
    history: History


@dataclass
class Counters:
    links: int = 0
    actual_misses: int = 0
    predicted: int = 0
    useful: int = 0
    overlap: int = 0
    actual_experts: int = 0

    def report(self) -> dict[str, float | int]:
        return {
            "links": self.links,
            "actual_misses": self.actual_misses,
            "predicted_nonresident": self.predicted,
            "useful_prefetches": self.useful,
            "raw_topk_agreement": self.overlap / self.actual_experts
            if self.actual_experts
            else 0.0,
            "actual_miss_recall": self.useful / self.actual_misses if self.actual_misses else 0.0,
            "prefetch_precision": self.useful / self.predicted if self.predicted else 0.0,
        }


def load_trace(path: pathlib.Path) -> list[Observation]:
    observations: list[Observation] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            observations.append(
                Observation(
                    position=int(value["position"]),
                    layer=int(value["layer"]),
                    experts=tuple(int(expert) for expert in value["experts"]),
                    misses=frozenset(int(expert) for expert in value["misses"]),
                    resident=frozenset(int(expert) for expert in value["resident"]),
                    next_layer_prediction=tuple(
                        int(expert) for expert in value.get("next_layer_prediction", [])
                    ),
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"{path}:{number}: invalid trace observation: {error}") from error
    return sorted(observations, key=lambda item: (item.position, item.layer))


def rank_recorded(link: Link) -> list[int]:
    """The shipped probe's own ranking, as recorded in the trace."""
    return list(link.source.next_layer_prediction[: link.top_m])


def rank_transition(link: Link) -> list[int]:
    """Co-occurrence of the target layer's experts with this row's selections."""
    scores: collections.Counter = collections.Counter()
    for expert in link.source.experts:
        scores.update(link.counts[link.source.layer][expert])
    return [expert for expert, _ in scores.most_common(link.top_m)]


def rank_recent(link: Link) -> list[int]:
    """Recency-decayed frequency of the target layer's strictly-past selections."""
    scores: collections.Counter = collections.Counter()
    for age, experts in enumerate(link.history[link.target.layer]):
        weight = 1.0 / (1 + age)
        for expert in experts:
            scores[expert] += weight
    return [expert for expert, _ in scores.most_common(link.top_m)]


RANKERS = {"recorded": rank_recorded, "transition": rank_transition, "recent": rank_recent}
PREDICTORS = tuple(RANKERS)


def analyze(
    observations: Iterable[Observation],
    top_m: int,
    predictor: str = "transition",
    window: int = 8,
) -> dict[str, object]:
    """Evaluate a trace-only L -> L+1 predictor without future leakage."""
    if predictor not in PREDICTORS:
        raise ValueError(f"unknown predictor {predictor!r}")
    ranker = RANKERS[predictor]
    by_position: dict[int, dict[int, Observation]] = collections.defaultdict(dict)
    for observation in observations:
        by_position[observation.position][observation.layer] = observation

    counts: TransitionCounts = collections.defaultdict(
        lambda: collections.defaultdict(collections.Counter)
    )
    history: History = collections.defaultdict(lambda: collections.deque(maxlen=window))
    overall = Counters()
    per_link: dict[int, Counters] = collections.defaultdict(Counters)
    for position in sorted(by_position):
        layers = by_position[position]
        for layer in sorted(layers):
            source = layers[layer]
            target = layers.get(layer + 1)
            if target is None:
                continue
            candidates = ranker(Link(source, target, top_m, counts, history))
            issued = [expert for expert in candidates if expert not in target.resident]
            useful = set(issued).intersection(target.misses)
            overlap = set(candidates).intersection(target.experts)
            for bucket in (overall, per_link[layer]):
                bucket.links += 1
                bucket.actual_misses += len(target.misses)
                bucket.predicted += len(issued)
                bucket.useful += len(useful)
                bucket.overlap += len(overlap)
                bucket.actual_experts += len(target.experts)
            for source_expert in source.experts:
                counts[layer][source_expert].update(target.experts)
        for seen_layer, row in layers.items():
            history[seen_layer].appendleft(row.experts)

    return {
        "predictor": predictor,
        "top_m": top_m,
        "window": window,
        "overall": overall.report(),
        "per_layer_link": {
            str(layer): bucket.report() for layer, bucket in sorted(per_link.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=pathlib.Path)
    parser.add_argument("--top-m", type=int, default=4)
    parser.add_argument("--predictor", choices=PREDICTORS, default="recorded")
    parser.add_argument(
        "--window",
        type=int,
        default=8,
        help="positions of strictly-past history the recent predictor may read",
    )
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.top_m < 1:
        parser.error("--top-m must be positive")
    if args.window < 1:
        parser.error("--window must be positive")
    report = analyze(load_trace(args.trace), args.top_m, args.predictor, args.window)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
