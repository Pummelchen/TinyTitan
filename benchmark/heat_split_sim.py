"""Screen a per-layer expert-cache budget split offline, across three route captures.

`continuous_feeder_sim` is the calibrated replay (§5y). It said the shipped victim order is
already the best one at the shipped 4 608-slot budget, and that the open seam is the
*partition*: 96 slots per layer against per-layer working sets of 122-417 distinct experts.
The only way to test a partition offline is to spend it by a heat profile -- and the only
heat profile an engine could ship is one measured on an EARLIER session. So the question
this module answers is not "how much can a split win", it is "does a split learned on one
capture still win on another", because a number that only wins against its own trace is
overfit and worthless.

Three committed captures, two of them the same prompt across a build change and one a
different workload whose route rows have nothing in common with the other two.

    python3 benchmark/heat_split_sim.py            # the whole table, ~1 minute
    python3 -m unittest test_heat_split_sim        # the falsifiers, in benchmark/

Licensed, as §5y licenses the replay: ranking budget shapes at a fixed slot total. Banned:
quoting a token rate from any of it.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from continuous_feeder_sim import _PlanSlots, load_trace

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expert-cache")
LAYERS = 48
SLOTS_PER_LAYER = 96
TOTAL_SLOTS = LAYERS * SLOTS_PER_LAYER
EXPERTS_PER_LAYER = 512

#: name -> (capture, first decode position). The boundary is the prefill length the run's
#: own footer reported, because the engine's decode counters exclude those rows.
CAPTURES = {
    "20261011": (os.path.join(CACHE, "route-trace-qwen38-4bit-20261011.txt.gz"), 30),
    "v520-20261015": (os.path.join(CACHE, "route-trace-qwen38-4bit-v520-20261015.txt.gz"), 10),
    "v520-prompt2": (
        os.path.join(CACHE, "route-trace-qwen38-4bit-v520-20261016-prompt2.txt.gz"),
        33,
    ),
}


def heat_profile(events, layers=LAYERS, start_position=0):
    """Distinct experts each layer routed, counting only rows at or after the boundary."""
    seen = [set() for _ in range(layers)]
    for position, layer, experts in events:
        if position >= start_position:
            seen[layer].update(experts)
    return [len(layer_seen) for layer_seen in seen]


def split_slots(profile, total_slots=TOTAL_SLOTS, floor=10, ceiling=EXPERTS_PER_LAYER):
    """Spend `total_slots` across layers in proportion to `profile`, exactly.

    The total is what the engine's tier buys, so a split never costs a byte more than the
    shipped one -- it only moves slots from cold layers to hot ones. `floor` keeps a cold
    layer from being starved to zero (it still routes its top-k every token), `ceiling`
    keeps any layer from holding more slots than it has experts. The rounding remainder
    goes to the heaviest layers, in the same deterministic order, so the answer depends on
    the profile and nothing else.
    """
    layers = len(profile)
    if floor * layers > total_slots or ceiling * layers < total_slots:
        raise ValueError(
            f"{total_slots} slots cannot be split over {layers} layers "
            f"between a floor of {floor} and a ceiling of {ceiling}"
        )
    weighted = [float(count) for count in profile]
    mass = sum(weighted) or 1.0
    raw = [total_slots * weight / mass for weight in weighted]
    budget = [min(ceiling, max(floor, int(value))) for value in raw]
    order = sorted(range(layers), key=lambda index: (-raw[index], index))
    cursor = 0
    while sum(budget) < total_slots:
        index = order[cursor % layers]
        if budget[index] < ceiling:
            budget[index] += 1
        cursor += 1
    cursor = 0
    while sum(budget) > total_slots:
        index = order[cursor % layers]
        if budget[index] > floor:
            budget[index] -= 1
        cursor += 1
    return budget


def placements_per_token(events, budgets, start_position=0, use_cap=None):
    """Demand placements per decode token under the engine's own plan rule.

    One cache per layer, sized by `budgets`, replayed exactly as `replay_misses` does;
    `use_cap` clamps the ranking key only, the way a saturating count would.

    The layer count comes from the trace, not from a constant, so a hand-traced
    single-layer fixture replays here while a 48-budget split still cannot be quietly
    scored against a trace that reaches a layer nobody budgeted for.
    """
    deepest = max((layer for _, layer, _ in events), default=-1)
    if deepest >= len(budgets):
        raise ValueError(
            f"trace reaches layer {deepest} but only {len(budgets)} per-layer budgets were given"
        )
    cache = [_PlanSlots(budgets[layer], use_cap=use_cap) for layer in range(len(budgets))]
    misses = 0
    tokens = set()
    for position, layer, experts in events:
        step = cache[layer].plan(experts)
        if position >= start_position:
            tokens.add(position)
            misses += len(step)
    return misses / max(1, len(tokens))


def two_pass_placements(events, budgets, start_position=0, use_cap=None):
    """Placements per decode token, for each of two identical passes over one cache set.

    §5ad's admissible screen for a policy that changes what the cache *remembers*. A single
    pass replays a cold cache, so it measures request 1; a saturating use count only pays
    against a holder it would otherwise have over-defended, and an over-defended holder is
    by definition a stale one, which cannot exist before the cache has run. Greedy decoding
    makes the second request route the same tokens (§5ad's hash equality), so replaying the
    same rows twice is the warm session, not a guess at one.

    Pass 1 starts exactly where `placements_per_token` starts -- cold caches, prefill rows
    planned but not counted -- so the two numbers are comparable by construction. A row may
    not name more experts than its layer has slots, which is the same limit the shipped plan
    runs under (top-k 10 against 96 slots) and both functions inherit from `_PlanSlots`.
    """
    deepest = max((layer for _, layer, _ in events), default=-1)
    if deepest >= len(budgets):
        raise ValueError(
            f"trace reaches layer {deepest} but only {len(budgets)} per-layer budgets were given"
        )
    cache = [_PlanSlots(budgets[layer], use_cap=use_cap) for layer in range(len(budgets))]
    tokens = max(1, len({position for position, _, _ in events if position >= start_position}))
    passes = []
    for _ in range(2):
        misses = 0
        for position, layer, experts in events:
            step = cache[layer].plan(experts)
            if position >= start_position:
                misses += len(step)
        passes.append(misses / tokens)
    return passes[0], passes[1]


def demand_per_token(events, start_position=0):
    """Expert requests each decode token makes -- the denominator of a hit rate.

    Derived from the trace instead of hardcoded as layers x top-k, so the plan's
    "+7 points of hit rate" acceptance bar is converted against the demand this capture
    actually routed. For qwen38 it is 480 (48 layers x 10).
    """
    requests = 0
    tokens = set()
    for position, _, experts in events:
        if position >= start_position:
            tokens.add(position)
            requests += len(experts)
    return requests / max(1, len(tokens))


def route_rows(events, start_position):
    """`(position, layer) -> experts`, for decode rows only."""
    rows = {}
    for position, layer, experts in events:
        if position < start_position:
            continue
        if (position, layer) in rows:
            raise ValueError(f"position {position} layer {layer} appears twice in the capture")
        rows[(position, layer)] = tuple(experts)
    return rows


def route_agreement(rows_a, rows_b):
    """How far two captures agree where their decode positions coincide.

    Returns `(shared rows, exactly identical rows, share of the picks in common)`. This is
    the overfit test: if two captures routed identically, a profile learned on one would
    just be the other read in advance. Compare the third number with `chance_pick_overlap`
    -- a level transfer could reach by accident.
    """
    common = rows_a.keys() & rows_b.keys()
    if not common:
        return 0, 0, float("nan")
    exact = sum(1 for key in common if rows_a[key] == rows_b[key])
    picks = sum(len(set(rows_a[key]) & set(rows_b[key])) for key in common)
    width = sum(len(rows_b[key]) for key in common)
    return len(common), exact, picks / width


def chance_pick_overlap(rows):
    """What two unrelated routings of this width would share, as a fraction of picks.

    Two independent top-k draws out of `EXPERTS_PER_LAYER` overlap on k * k / E of k
    requests, i.e. k / E. Computed from the capture's own width, not a constant.
    """
    picks = sum(len(row) for row in rows.values()) / max(1, len(rows))
    return picks / EXPERTS_PER_LAYER


def profile_correlation(a, b):
    """Pearson r of two per-layer heat profiles -- how similar the shapes are."""
    mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
    numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b, strict=True))
    denominator = math.sqrt(sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b))
    return numerator / denominator


def main(argv=None):
    loaded = {}
    for name, (path, start) in CAPTURES.items():
        trace = load_trace(path)
        loaded[name] = (trace, start, heat_profile(trace, start_position=start))
        profile = loaded[name][2]
        print(
            f"{name:16s} decode from position {start:3d}  heat "
            f"{min(profile)}..{max(profile)} mean {sum(profile) / LAYERS:.0f}"
            f"  demand {demand_per_token(trace, start):.0f} requests/token"
        )
    print("\ncapture pairs (profile shape correlation, and how far their routes agree):")
    row_sets = {name: route_rows(trace, start) for name, (trace, start, _) in loaded.items()}
    chance = chance_pick_overlap(next(iter(row_sets.values())))
    names = list(loaded)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            positions, exact, picks = route_agreement(row_sets[first], row_sets[second])
            print(
                f"  {first:16s} {second:16s} r {profile_correlation(loaded[first][2], loaded[second][2]):+.3f}"
                f"  {positions} shared rows, {exact} identical, picks agree {picks:6.3%}"
                f" (chance {chance:.3%})"
            )
    print()
    uniform_budget = [SLOTS_PER_LAYER] * LAYERS
    # Every arm is scored against the shipped arm at ITS OWN cap. An earlier table divided
    # a capped transfer by the uncapped uniform and reported 86.5% keeps for the pair that
    # is 74.4% like-for-like; the two arms differ by the cap, so the denominator has to.
    for cap in (None, 3, 5):
        print(f"=== cap {'none (engine as shipped)' if cap is None else cap} ===\n")
        for target, (trace, start, profile) in loaded.items():
            demand = demand_per_token(trace, start)
            base = placements_per_token(trace, uniform_budget, start, use_cap=cap)
            oracle = placements_per_token(trace, split_slots(profile), start, use_cap=cap)
            print(
                f"scored on {target:16s} uniform {base:7.2f} ({1 - base / demand:.2%} hit)  "
                f"oracle {oracle:7.2f} ({oracle / base - 1:+6.2%}, "
                f"{(1 - oracle / demand - (1 - base / demand)) * 100:+.2f} points)"
            )
            for source, (_, _, source_profile) in loaded.items():
                if source == target:
                    continue
                value = placements_per_token(trace, split_slots(source_profile), start, use_cap=cap)
                kept = (base - value) / (base - oracle)
                print(
                    f"    transfer from {source:16s} {value:7.2f} "
                    f"({value / base - 1:+6.2%}, "
                    f"{(1 - value / demand - (1 - base / demand)) * 100:+.2f} points)  "
                    f"keeps {kept:6.1%} of the oracle"
                )
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
