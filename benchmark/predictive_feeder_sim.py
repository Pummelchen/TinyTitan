"""Can a continuous SSD feeder name, in advance, the experts the cache is about to miss?

§5z closed the *budget shape* lever: at 4 608 slots every static partition is worth about one
point of hit rate, while Belady on the same slots is worth eight. The gap it left is temporal,
and the operator's premise is the temporal idea -- stop reading in bursts around each request
and stream weights constantly while a session is live, so a read is overlapped instead of
exposed.

A feeder cannot invent a request, so the only thing it can stream is what it can name. The only
naming it has, in an engine that ships no per-conversation oracle, is an EARLIER session. This
module measures that against the two quantities that matter:

* `predictable_share` -- of the 480 expert requests a decode token makes, how many are in a
  previous session's top-n for that layer. Demand coverage, the optimistic view.
* `miss_coverage` -- of the placements the shipped cache rule actually makes, how many were
  nameable that way. This is the one a feeder can act on: a streamed expert still costs a
  read, so the only win is a read that would have been exposed and is now already done.

Three arms then act on the answer, all inside the same 4 608-slot tier so none of them is a
budget increase wearing another name:

* `pinned_placements` holds P experts forever, and P comes out of the tier.
* `heated_placements` reserves nothing and only changes who is evicted when a slot is needed.
* `lookahead_placements` protects what THIS session routes in its next W tokens -- an oracle,
  and the only arm here that reaches the plan's +7-point bar, which is how it prices the
  temporal seam: in tokens of lookahead the engine cannot have.

    python3 -m unittest test_predictive_feeder_sim   # the falsifiers
    python3 benchmark/predictive_feeder_sim.py       # the sweep, model-free; two runtimes in §5aa dev. 10

Reuses the committed captures, the engine's own placement rule and §5z's slot tier from
`heat_split_sim`. Model-free, and no token rate is quoted anywhere in it.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from continuous_feeder_sim import _PlanSlots, load_trace
from heat_split_sim import CAPTURES, SLOTS_PER_LAYER, demand_per_token, placements_per_token

#: Qwen3.8-Flash-Next: 512 routed experts per layer, top-k 10, 48 MoE layers.
EXPERTS_PER_LAYER = 512

#: One expert's weights on disk, and the three rates the feasibility envelope needs. Every one of
#: them is a campaign measurement, not an assumption: 3.57 GB/s is §5d's four-worker reader
#: "matching its in-engine effective throughput", 1.3595 GB/s is §5v's measured `disk0` mean across
#: a whole decode run, and 4.742 is this capture's own `decode_tok_s` from
#: `expert-cache/run-record-qwen38-4bit-v520-20261016-prompt2.txt`.
EXPERT_STRIDE_BYTES = 2_768_896
READER_WALL_BYTES_PER_SECOND = 3.57e9
DRIVE_MEASURED_BYTES_PER_SECOND = 1_359_500_000
CAPTURE_DECODE_TOKENS_PER_SECOND = 4.742

#: Sweep widths. Pins are capped by the tier (they hold slots); heat protection is not, because
#: it only expresses a preference; lookahead is in tokens of the session's own future.
NAME_WIDTHS = (96, 192, 288)
PIN_WIDTHS = (0, 12, 24, 36, 48, 60, 72, 84, 96)
HEAT_WIDTHS = (96, 192, 288, 384)
LOOKAHEAD_WIDTHS = (0, 1, 2, 4, 8, 16, 32)


def _layer_count(events):
    """Layers the trace actually reaches, so a one-layer fixture stays a one-layer fixture."""
    return max((layer for _, layer, _ in events), default=-1) + 1


def _check(hot, events, what):
    layers = _layer_count(events)
    if layers > len(hot):
        raise ValueError(
            f"trace reaches layer {layers - 1} but {what} named only {len(hot)} layers"
        )


def hot_sets(events, start_position, size):
    """Each layer's `size` most-requested experts, from the decode rows only.

    Ranked by request count, ties broken by ascending expert index, so the same capture
    always yields the same list on any machine -- a shipped hot list has to be reproducible.
    `size` is capped at `EXPERTS_PER_LAYER`: naming more experts than a layer has would claim
    coverage it cannot deliver.
    """
    if not 1 <= size <= EXPERTS_PER_LAYER:
        raise ValueError(f"hot list width must be between 1 and {EXPERTS_PER_LAYER}, got {size}")
    _check([0] * _layer_count(events), events, "the trace")
    counts = [{} for _ in range(_layer_count(events))]
    for position, layer, experts in events:
        if position < start_position:
            continue
        for expert in experts:
            counts[layer][expert] = counts[layer].get(expert, 0) + 1
    return [
        {
            expert
            for expert, _ in sorted(layer_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:size]
        }
        for layer_counts in counts
    ]


def predictable_share(hot, events, start_position):
    """Share of decode requests that fall in that layer's hot list -- demand coverage."""
    _check(hot, events, "the hot list")
    requests = covered = 0
    for position, layer, experts in events:
        if position < start_position:
            continue
        requests += len(experts)
        covered += sum(1 for expert in experts if expert in hot[layer])
    if not requests:
        return float("nan")
    return covered / requests


def miss_coverage(events, budgets, hot, start_position):
    """Share of the shipped rule's placements that a hot list could have named.

    Replayed with the engine's own `_PlanSlots` under `budgets`, so the misses are the ones
    the cache really pays for. Prefill rows warm the cache and are not scored, exactly as in
    `placements_per_token`; hits are never counted, because a hit is a read no feeder had to
    make.
    """
    _check(hot, events, "the hot list")
    if len(budgets) < len(hot):
        raise ValueError(
            f"{len(hot)} layers of hot list cannot be replayed against {len(budgets)} budgets"
        )
    cache = [_PlanSlots(budgets[layer], use_cap=None) for layer in range(len(hot))]
    placements = nameable = 0
    for position, layer, experts in events:
        misses = cache[layer].plan(experts)
        if position < start_position:
            continue
        placements += len(misses)
        nameable += sum(1 for expert in misses if expert in hot[layer])
    if not placements:
        return float("nan")
    return nameable / placements


class _PinnedSlots:
    """One layer's tier split into a prefix that never evicts and the plan rule over the rest.

    The prefix is resident by construction, so a request for it is a hit with no read and the
    expert can never be a victim. Everything else runs through `_PlanSlots` over the slots that
    are left -- one call per layer step, so the clock still ticks once per plan and the use
    counts still rise for every named expert, exactly as the engine's rule does.
    """

    def __init__(self, capacity, pinned, use_cap=None):
        if capacity < 1:
            raise ValueError(f"a layer needs at least one slot, got {capacity}")
        if len(pinned) > capacity:
            raise ValueError(f"{len(pinned)} pinned experts cannot live in a {capacity}-slot tier")
        self.pinned = set(pinned)
        self.live = _PlanSlots(capacity - len(self.pinned), use_cap=use_cap)

    def plan(self, experts):
        misses = [expert for expert in experts if expert not in self.pinned]
        # Nothing left to hold them with: an expert with no slot to land in is a fresh read
        # every time it is asked for, which is what the engine would do.
        if self.live.capacity == 0:
            return misses
        return self.live.plan(misses)


def pinned_placements(events, capacity, pinned, start_position=0, use_cap=None):
    """Demand placements per decode token when each layer pins a set that never evicts.

    `capacity` stays the shipped tier and the pins are taken OUT of it, so every arm here costs
    the same bytes of RAM as the one the engine runs today -- that is what makes this a fair
    comparison rather than a budget increase wearing a different name.
    """
    layers = _layer_count(events)
    if layers > len(pinned):
        raise ValueError(
            f"trace reaches layer {layers - 1} but pins named only {len(pinned)} layers"
        )
    caches = [_PinnedSlots(capacity, names, use_cap=use_cap) for names in pinned]
    placements = 0
    tokens = set()
    for position, layer, experts in events:
        step = caches[layer].plan(experts)
        if position >= start_position:
            tokens.add(position)
            placements += len(step)
    return placements / max(1, len(tokens))


class _HeatedSlots(_PlanSlots):
    """The engine's plan rule with one change: a predicted expert is the last slot to give up.

    Nothing is reserved and nothing is pre-read, so the tier is byte-for-byte the shipped one.
    A protected slot's rank is infinity, which sorts it behind every unprotected slot and lets
    the usual `(count, recency)` tie-break decide among protected ones -- so a layer whose slots
    are ALL protected still finds a victim instead of stalling.
    """

    def __init__(self, capacity, protect=(), use_cap=None):
        super().__init__(capacity, use_cap=use_cap)
        self.protect = set(protect)

    def _price(self, slot):
        base = super()._price(slot)
        if base == -1 or self.slot_expert[slot] not in self.protect:
            return base
        return float("inf")


def heated_placements(events, capacity, protect, start_position=0, use_cap=None):
    """Demand placements per decode token when victims are chosen with a heat preference.

    `protect[layer]` is a set of expert ids an EARLIER session named; it may be wider than the
    tier, because it claims no slot.
    """
    layers = _layer_count(events)
    if layers > len(protect):
        raise ValueError(
            f"trace reaches layer {layers - 1} but heat named only {len(protect)} layers"
        )
    caches = [_HeatedSlots(capacity, names, use_cap=use_cap) for names in protect]
    placements = 0
    tokens = set()
    for position, layer, experts in events:
        step = caches[layer].plan(experts)
        if position >= start_position:
            tokens.add(position)
            placements += len(step)
    return placements / max(1, len(tokens))


def _slide(pending, experts, delta):
    """Add or remove one row's experts from a sliding window's membership counts."""
    for expert in experts:
        pending[expert] = pending.get(expert, 0) + delta
        if pending[expert] == 0:
            del pending[expert]


def lookahead_placements(events, capacity, window, start_position=0, use_cap=None):
    """Demand placements per decode token with the next `window` tokens' routes protected.

    The honest form of reading ahead: not a list from another session, but this session's own
    near future, which is what a feeder would have to know to keep the drive busy. Protection
    is recomputed before every layer step and slides by one token at a time; it reserves no
    slot, so `window=0` is exactly the shipped rule and every other window reads as a delta on
    it. This is an oracle -- the engine cannot see these rows yet -- and it exists to bound what
    any real predictor could ever be worth.
    """
    layers = _layer_count(events)
    if window < 0:
        raise ValueError(f"a lookahead window cannot be negative, got {window}")
    rows = [[] for _ in range(layers)]
    for position, layer, experts in events:
        rows[layer].append((position, list(experts)))
    caches = [_HeatedSlots(capacity, (), use_cap=use_cap) for _ in range(layers)]
    placements = 0
    tokens = set()
    for layer, layer_rows in enumerate(rows):
        layer_rows.sort(key=lambda row: row[0])
        cache = caches[layer]
        pending = {}
        width = len(layer_rows)
        for index in range(1, min(window + 1, width)):
            _slide(pending, layer_rows[index][1], +1)
        for index, (position, experts) in enumerate(layer_rows):
            cache.protect = set(pending)
            step = cache.plan(experts)
            if position >= start_position:
                tokens.add(position)
                placements += len(step)
            if index + 1 < width:
                _slide(pending, layer_rows[index + 1][1], -1)
            if index + 1 + window < width:
                _slide(pending, layer_rows[index + 1 + window][1], +1)
    return placements / max(1, len(tokens))


def drive_experts_per_token(bytes_per_second, tokens_per_second, expert_stride_bytes):
    """How many experts the drive could stream for one token, rounded down.

    A feasibility envelope, not a plan: it says whether an idea fits on the wire at all.
    Either rate at zero means nothing arrives, so it returns 0 instead of dividing.
    """
    if bytes_per_second <= 0 or tokens_per_second <= 0 or expert_stride_bytes <= 0:
        return 0
    return int(bytes_per_second / tokens_per_second // expert_stride_bytes)


def main(argv=None):
    loaded = {}
    for name, (path, start) in CAPTURES.items():
        loaded[name] = (load_trace(path), start)
    layers = _layer_count(next(iter(loaded.values()))[0])
    uniform = [SLOTS_PER_LAYER] * layers
    stride = EXPERT_STRIDE_BYTES

    print(f"slot tier per layer {SLOTS_PER_LAYER}, {layers} layers, expert stride {stride} B")
    print("ranking only -- no token rate is quoted anywhere in this output.\n")

    print("=== what an EARLIER session can name, per list width (96 = the shipped tier)")
    print(
        f"{'learned on':16s} {'scored on':16s} {'width':>5s} {'demand cover':>13s} {'miss cover':>11s}"
    )
    for learn_name, (learn_trace, learn_start) in loaded.items():
        for width in NAME_WIDTHS:
            hot = hot_sets(learn_trace, start_position=learn_start, size=width)
            for score_name, (score_trace, score_start) in loaded.items():
                if score_name == learn_name:
                    continue
                print(
                    f"{learn_name:16s} {score_name:16s} {width:5d}"
                    f" {predictable_share(hot, score_trace, score_start):13.1%}"
                    f" {miss_coverage(score_trace, uniform, hot, score_start):11.1%}"
                )

    print("\n=== pinned prefix: P experts held forever, taken OUT of the same 96-slot tier")
    print(
        f"{'learned on':16s} {'scored on':16s} "
        + " ".join(f"{f'P={p}':>9s}" for p in PIN_WIDTHS)
        + "  pre-read GiB per pin"
    )
    for learn_name, (learn_trace, learn_start) in loaded.items():
        for score_name, (score_trace, score_start) in loaded.items():
            if score_name == learn_name:
                continue
            base = pinned_placements(score_trace, SLOTS_PER_LAYER, [set()] * layers, score_start)
            cells = []
            for pins in PIN_WIDTHS:
                hot = hot_sets(learn_trace, start_position=learn_start, size=max(pins, 1))
                value = pinned_placements(
                    score_trace, SLOTS_PER_LAYER, hot if pins else [set()] * layers, score_start
                )
                cells.append(f"{value / base - 1:+8.2%} ")
            print(f"{learn_name:16s} {score_name:16s} " + " ".join(cells))
    print("    (P=96 leaves no live slot: every unpinned request is a cold read.)")
    print(
        f"    one pin is an expert id held in all {layers} layers, so it costs "
        f"{layers * stride / 2**30:.03f} GiB of pre-read; a full {SLOTS_PER_LAYER}-wide tier "
        f"is {SLOTS_PER_LAYER * layers * stride / 2**30:.2f} GiB."
    )

    print("\n=== heat-biased eviction: same 96 slots, no reservation, only a victim preference")
    print(
        f"{'learned on':16s} {'scored on':16s} " + " ".join(f"{f'W={w}':>9s}" for w in HEAT_WIDTHS)
    )
    for learn_name, (learn_trace, learn_start) in loaded.items():
        for score_name, (score_trace, score_start) in loaded.items():
            if score_name == learn_name:
                continue
            base = heated_placements(score_trace, SLOTS_PER_LAYER, [set()] * layers, score_start)
            cells = []
            for width in HEAT_WIDTHS:
                hot = hot_sets(learn_trace, start_position=learn_start, size=width)
                value = heated_placements(score_trace, SLOTS_PER_LAYER, hot, score_start)
                cells.append(f"{value / base - 1:+8.2%} ")
            print(f"{learn_name:16s} {score_name:16s} " + " ".join(cells))
    print("    self arm, same capture learned and scored -- this is hindsight, an upper bound:")
    for name, (trace, start) in loaded.items():
        base = heated_placements(trace, SLOTS_PER_LAYER, [set()] * layers, start)
        cells = []
        for width in HEAT_WIDTHS:
            value = heated_placements(trace, SLOTS_PER_LAYER, hot_sets(trace, start, width), start)
            cells.append(f"{value / base - 1:+8.2%} ")
        print(f"  {name:16s} {name:16s} " + " ".join(cells))

    print("\n=== how far ahead THIS session would have to be known (an oracle, not a predictor)")
    print(
        f"{'capture':16s} "
        + " ".join(f"{f'W={w}':>9s}" for w in LOOKAHEAD_WIDTHS)
        + "   points at W=32"
    )
    for name, (trace, start) in loaded.items():
        demand = demand_per_token(trace, start)
        base = lookahead_placements(trace, SLOTS_PER_LAYER, 0, start)
        cells = []
        best = None
        for window in LOOKAHEAD_WIDTHS:
            value = lookahead_placements(trace, SLOTS_PER_LAYER, window, start)
            cells.append(f"{value / base - 1:+8.2%} ")
            best = (1 - value / demand) - (1 - base / demand)
        print(f"{name:16s} " + " ".join(cells) + f"   {best * 100:+.2f} pts of hit rate")

    print("\ncache as shipped, for reference:")
    shipped = []
    for name, (trace, start) in loaded.items():
        demand = demand_per_token(trace, start)
        placements = placements_per_token(trace, uniform, start)
        shipped.append((name, demand, placements))
        print(
            f"  {name:16s} hit {1 - placements / demand:.1%}  placements {placements:.2f}/token"
            f"  reads {placements * stride / 2**20:.0f} MiB/token"
        )
    busiest = max(shipped, key=lambda row: row[2])
    print(
        f"\nwire: the reader's own wall of {READER_WALL_BYTES_PER_SECOND / 1e9:.2f} GB/s carries "
        f"{drive_experts_per_token(READER_WALL_BYTES_PER_SECOND, CAPTURE_DECODE_TOKENS_PER_SECOND, stride)}"
        f" experts/token at this capture's rate, and the drive as measured "
        f"({DRIVE_MEASURED_BYTES_PER_SECOND / 1e9:.2f} GB/s) already carries "
        f"{drive_experts_per_token(DRIVE_MEASURED_BYTES_PER_SECOND, CAPTURE_DECODE_TOKENS_PER_SECOND, stride)}"
    )
    print(
        f"      against a demand of {busiest[1]:.0f} experts/token and {busiest[2]:.0f} placements/token "
        f"on the busiest capture ({busiest[0]}): the headroom on the wire is real, and none of the arms "
        "above names anything to spend it on"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
