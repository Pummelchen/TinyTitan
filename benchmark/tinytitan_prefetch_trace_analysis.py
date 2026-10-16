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
from dataclasses import dataclass, field
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

    @property
    def candidate_limit(self) -> int:
        """Names a ranker may expose: the engine scans its whole predicted list."""
        return max(self.top_m, len(self.source.next_layer_prediction))


@dataclass
class Counters:
    links: int = 0
    actual_misses: int = 0
    predicted: int = 0
    useful: int = 0
    overlap: int = 0
    actual_experts: int = 0
    slot_predicted: list[int] = field(default_factory=list)
    slot_useful: list[int] = field(default_factory=list)

    def tally_slot(self, index: int, *, hit: bool) -> None:
        """Record one issued read at its issue index, so a depth is judged per slot."""
        while len(self.slot_predicted) <= index:
            self.slot_predicted.append(0)
            self.slot_useful.append(0)
        self.slot_predicted[index] += 1
        self.slot_useful[index] += 1 if hit else 0

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
    return list(link.source.next_layer_prediction[: link.candidate_limit])


def rank_transition(link: Link) -> list[int]:
    """Co-occurrence of the target layer's experts with this row's selections."""
    scores: collections.Counter = collections.Counter()
    for expert in link.source.experts:
        scores.update(link.counts[link.source.layer][expert])
    return [expert for expert, _ in scores.most_common(link.candidate_limit)]


def rank_recent(link: Link) -> list[int]:
    """Recency-decayed frequency of the target layer's strictly-past selections."""
    scores: collections.Counter = collections.Counter()
    for age, experts in enumerate(link.history[link.target.layer]):
        weight = 1.0 / (1 + age)
        for expert in experts:
            scores[expert] += weight
    return [expert for expert, _ in scores.most_common(link.candidate_limit)]


def stage_filter_first(link: Link, ranked: list[int]) -> list[int]:
    """The shipped rule: drop what the cache holds, then fill the free slots.

    `RealForwardRunner+DecodeMoE.swift` hands the whole ranked prediction to the
    ring and lets the ring size cap it, because about 85 % of a top-M prefix is
    already resident: capping first starves the plan instead of aiming it.
    """
    return [expert for expert in ranked if expert not in link.target.resident][: link.top_m]


def stage_truncate_first(link: Link, ranked: list[int]) -> list[int]:
    """The withdrawn v4.3 rule: take the top M names, then drop the resident ones."""
    return [expert for expert in ranked[: link.top_m] if expert not in link.target.resident]


RANKERS = {"recorded": rank_recorded, "transition": rank_transition, "recent": rank_recent}
PREDICTORS = tuple(RANKERS)
STAGERS = {"filter": stage_filter_first, "truncate": stage_truncate_first}
POLICIES = tuple(STAGERS)


# Both sides of a ring's price, measured on the device ledger by §5bb (build
# `a8c6fdd`, four mirrored arms, byte-identical output). Each round is a complete
# price book: the device idle its shipped depth-1 ring removed, and the exposed time
# the ring's own line (`shared_expert -> moe_phase1_2_routed`) grew, reduced to a
# prize per adopted read and a cost per issued read.
#
# The byte floor is deliberately NOT the cost side. One read is 0.74066 ms of *drive
# busy* time, but §5bb's arms show the drive is not the wall -- `io_host_waits_avoided`
# moved while the host never blocked, and the ring's speculative bytes overlap compute.
# Pricing a read at the byte floor therefore charges it for serialized time the machine
# does not spend, and §5bb's first version of this screen did exactly that.
@dataclass(frozen=True)
class PriceBook:
    """One §5bb round's device ledger, reduced to what a depth decision needs."""

    gap_line_ms_per_token: float
    events_converted_per_token: float
    ms_per_event: float
    adopted_reads_per_token: float
    issued_reads_per_token: float
    ring_line_ms_per_token: float

    @property
    def ms_per_adopted_read(self) -> float:
        """The countable prize: whole fixup events converted, per adopted read."""
        return self.events_converted_per_token * self.ms_per_event / self.adopted_reads_per_token

    @property
    def ms_per_adopted_read_whole_line(self) -> float:
        """The upper reading: all of the measured gap-line fall, per adopted read."""
        return self.gap_line_ms_per_token / self.adopted_reads_per_token

    @property
    def ms_per_issued_read(self) -> float:
        """The exposed cost: growth of the ring's own device line, per issue."""
        return self.ring_line_ms_per_token / self.issued_reads_per_token


ROUND1 = PriceBook(16.523, 4.62, 2.184, 19.58, 39.53, 6.828)
ROUND2 = PriceBook(17.981, 4.48, 2.159, 19.20, 39.46, 4.409)
EXPERT_READ_MIB = 2.7689
DRIVE_GB_PER_S = 3.92
MIB_TO_MB = 1.048576


def _pairings() -> list[tuple[float, float, float]]:
    """(break-even precision, prize per adopted read, cost per issued read), within round."""
    pairings: list[tuple[float, float, float]] = []
    for book in (ROUND1, ROUND2):
        cost = book.ms_per_issued_read
        for prize in (book.ms_per_adopted_read, book.ms_per_adopted_read_whole_line):
            pairings.append((cost / prize, prize, cost))
    return pairings


def _corner(side: str) -> tuple[float, float, float]:
    """One corner of the band: (break-even, prize per adopted read, cost per issued read)."""
    pairings = _pairings()
    if side == "pessimistic":
        return max(pairings, key=lambda item: item[0])
    return min(pairings, key=lambda item: item[0])


def economics() -> dict[str, float | dict[str, float]]:
    """The byte floor, and the adoption band between which depth is undecidable offline."""
    low, high = _corner("optimistic")[0], _corner("pessimistic")[0]
    return {
        "drive_busy_ms_per_read": EXPERT_READ_MIB * MIB_TO_MB / DRIVE_GB_PER_S,
        "break_even_low": low,
        "break_even_high": high,
        "pessimistic": {
            "ms_per_adopted_read": _corner("pessimistic")[1],
            "ms_per_issued_read": _corner("pessimistic")[2],
        },
        "optimistic": {
            "ms_per_adopted_read": _corner("optimistic")[1],
            "ms_per_issued_read": _corner("optimistic")[2],
        },
    }


def _net_per_link(bucket: Counters, predicted: int, useful: int, side: str) -> float:
    if not bucket.links:
        return 0.0
    _, prize, cost = _corner(side)
    return (useful * prize - predicted * cost) / bucket.links


def slot_report(bucket: Counters) -> list[dict[str, float | int | bool]]:
    """Precision and net device time per issue index, so depth has a stop rule.

    A slot is only called when both corners of §5bb's band agree. Between them the
    offline screen cannot decide, and the honest answer is that a live arm is needed --
    which is what §5aq found when it priced deeper rings on wall clock alone.
    """
    slots = []
    for index, (predicted, useful) in enumerate(
        zip(bucket.slot_predicted, bucket.slot_useful, strict=True)
    ):
        net_worst = _net_per_link(bucket, predicted, useful, "pessimistic")
        net_best = _net_per_link(bucket, predicted, useful, "optimistic")
        slots.append(
            {
                "slot": index,
                "predicted": predicted,
                "useful": useful,
                "precision": useful / predicted if predicted else 0.0,
                "net_worst_ms_per_link": net_worst,
                "net_best_ms_per_link": net_best,
                "pays_under_every_reading": bool(predicted) and net_worst > 0.0,
                "loses_under_every_reading": bool(predicted) and net_best < 0.0,
            }
        )
    return slots


def first_loss_making_slot(slots: list[dict[str, float | int | bool]]) -> int | None:
    """The first issue index no reading of §5bb's ledger defends, or None."""
    for slot in slots:
        if slot["loses_under_every_reading"]:
            return int(slot["slot"])
    return None


def first_unproven_slot(slots: list[dict[str, float | int | bool]]) -> int | None:
    """The first issue index the ledger cannot prove pays, so a live arm must."""
    for slot in slots:
        if slot["predicted"] and not slot["pays_under_every_reading"]:
            return int(slot["slot"])
    return None


def analyze(
    observations: Iterable[Observation],
    top_m: int,
    predictor: str = "transition",
    window: int = 8,
    policy: str = "filter",
) -> dict[str, object]:
    """Evaluate a trace-only L -> L+1 predictor without future leakage."""
    if predictor not in PREDICTORS:
        raise ValueError(f"unknown predictor {predictor!r}")
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    ranker = RANKERS[predictor]
    stager = STAGERS[policy]
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
            link = Link(source, target, top_m, counts, history)
            ranked = ranker(link)
            issued = stager(link, ranked)
            useful = set(issued).intersection(target.misses)
            overlap = set(ranked[:top_m]).intersection(target.experts)
            for bucket in (overall, per_link[layer]):
                bucket.links += 1
                bucket.actual_misses += len(target.misses)
                bucket.predicted += len(issued)
                bucket.useful += len(useful)
                bucket.overlap += len(overlap)
                bucket.actual_experts += len(target.experts)
                for index, expert in enumerate(issued):
                    bucket.tally_slot(index, hit=expert in useful)
            for source_expert in source.experts:
                counts[layer][source_expert].update(target.experts)
        for seen_layer, row in layers.items():
            history[seen_layer].appendleft(row.experts)

    slots = slot_report(overall)
    return {
        "predictor": predictor,
        "policy": policy,
        "top_m": top_m,
        "window": window,
        "economics": economics(),
        "slots": slots,
        "first_loss_making_slot": first_loss_making_slot(slots),
        "first_unproven_slot": first_unproven_slot(slots),
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
    parser.add_argument(
        "--policy",
        choices=POLICIES,
        default="filter",
        help=(
            "how a ranking becomes reads: filter drops cache-resident names before the ring "
            "size caps the plan (what the engine does), truncate caps the ranking first "
            "(the withdrawn v4.3 ring)"
        ),
    )
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.top_m < 1:
        parser.error("--top-m must be positive")
    if args.window < 1:
        parser.error("--window must be positive")
    report = analyze(load_trace(args.trace), args.top_m, args.predictor, args.window, args.policy)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
