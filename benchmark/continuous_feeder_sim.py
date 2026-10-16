"""Continuous weight feeder: a timeline model of decode against one drive.

The shipped engine reads experts in bursts -- the router resolves a layer, the
host blocks on that layer's misses, the drive runs flat out for a few
milliseconds, and then the drive sits idle while the GPU finishes the token.
This module asks the operator's question: if the drive kept reading through
that idle time, pulling experts the history says are coming, how much of the
blocked I/O would simply stop being on the critical path?

It is a timeline, not a hit-rate replay. The drive is one serial server that
demand reads preempt, the expert cache is the engine's own LFU-with-LRU-tiebreak
placement, and a feeder may only issue reads in the intervals the demand reader
leaves free. Every number printed as a rate is a projection through that model,
never a measurement.
"""

import argparse
import bisect
import gzip
import sys
from collections import defaultdict

# Measured on this install: 48 layers, 512 experts per layer, top-10, a 2.64 MiB
# expert blob, 96 cache slots per layer, 3.92 GB/s at four pread threads, and
# ~108.6 ms of GPU busy time spread over a token's 48 layer steps.
EXPERT_STRIDE = 2_768_896
BANDWIDTH_BYTES_PER_MS = 3_920_000.0
GPU_MS_PER_STEP = 108.6 / 48
HOST_MS_PER_STEP = 1.0


class TraceFormatError(Exception):
    """A route-trace line does not carry a position, a layer and a top-k."""


class FeederConfig:
    def __init__(
        self,
        expert_stride=EXPERT_STRIDE,
        read_bytes=EXPERT_STRIDE,
        bandwidth_bytes_per_ms=BANDWIDTH_BYTES_PER_MS,
        slots_per_layer=96,
        gpu_ms_per_step=GPU_MS_PER_STEP,
        host_ms_per_step=HOST_MS_PER_STEP,
        max_preempt_delay_ms=None,
        max_speculative_reads=None,
    ):
        if bandwidth_bytes_per_ms <= 0:
            raise ValueError("bandwidth_bytes_per_ms must be positive")
        if slots_per_layer <= 0:
            raise ValueError("slots_per_layer must be positive")
        if read_bytes <= 0 or read_bytes > expert_stride:
            raise ValueError("read_bytes must be between 1 and expert_stride")
        if gpu_ms_per_step < 0 or host_ms_per_step < 0:
            raise ValueError("per-step times must not be negative")
        self.expert_stride = expert_stride
        self.read_bytes = read_bytes
        self.bandwidth_bytes_per_ms = bandwidth_bytes_per_ms
        self.slots_per_layer = slots_per_layer
        self.gpu_ms_per_step = gpu_ms_per_step
        self.host_ms_per_step = host_ms_per_step
        self.max_preempt_delay_ms = max_preempt_delay_ms
        self.max_speculative_reads = max_speculative_reads

    @property
    def read_ms(self):
        return self.read_bytes / self.bandwidth_bytes_per_ms


class NoFeeder:
    """Shipped burst-only behaviour: the drive reads when the router asks."""

    name = "burst"

    def observe(self, position, layer, experts):
        pass

    def candidates(self, position, layer, layer_count):
        return []


class _RouteHistory:
    """Bookkeeping for policies that predict from the route sets seen so far.

    `observe` is called as each step is reached, so a policy can only ever read
    the current position's resolved layers and everything before them -- which
    is exactly what the engine's router probe knows.
    """

    def __init__(self):
        self.seen = defaultdict(dict)

    def observe(self, position, layer, experts):
        self.seen[layer][position] = list(experts)

    def positions_before(self, layer, position):
        return sorted((p for p in self.seen.get(layer, {}) if p <= position), reverse=True)


class PrevToken(_RouteHistory):
    """What this layer routed last time, one or more layers ahead of the cursor."""

    name = "prev_token"

    def __init__(self, depth=1):
        super().__init__()
        self.depth = depth

    def candidates(self, position, layer, layer_count):
        out = []
        for ahead in range(1, self.depth + 1):
            target = (layer + ahead) % max(1, layer_count)
            for past in self.positions_before(target, position)[:1]:
                out.extend((target, e) for e in self.seen[target][past])
        return out


class RecencyWindow(PrevToken):
    """The union of the last `width` positions' routes, `depth` layers ahead."""

    name = "recency"

    def __init__(self, width=3, depth=1):
        super().__init__(depth=depth)
        self.width = width

    def candidates(self, position, layer, layer_count):
        out, seen = [], set()
        for ahead in range(1, self.depth + 1):
            target = (layer + ahead) % max(1, layer_count)
            for past in self.positions_before(target, position)[: self.width]:
                for expert in self.seen[target][past]:
                    if (target, expert) not in seen:
                        seen.add((target, expert))
                        out.append((target, expert))
        return out


class _RankedLayers:
    """Per-layer ranking, recomputed only when that layer is routed again.

    A candidate scan over every layer at every step is quadratic in the trace
    and takes minutes on a real capture, so each policy ranks one layer at a
    time and caches it until that layer's routes change.
    """

    def __init__(self, per_layer=8, horizon=1):
        self.per_layer = per_layer
        self.horizon = horizon
        self.version = defaultdict(int)
        self.ranked = {}

    def _rank(self, target):
        raise NotImplementedError

    def _touch(self, target):
        self.version[target] += 1
        self.ranked.pop(target, None)

    def candidates(self, position, layer, layer_count):
        out, seen = [], set()
        for ahead in range(1, self.horizon + 1):
            target = (layer + ahead) % max(1, layer_count)
            if target not in self.ranked:
                self.ranked[target] = self._rank(target)
            for expert in self.ranked[target][: self.per_layer]:
                if (target, expert) in seen:
                    continue
                seen.add((target, expert))
                out.append((target, expert))
        return out


class HotSet(_RankedLayers):
    """The per-layer frequency ranking, which is what the cache already keeps."""

    name = "hot_set"

    def __init__(self, per_layer=8, horizon=1):
        super().__init__(per_layer=per_layer, horizon=horizon)
        self.counts = defaultdict(lambda: defaultdict(int))

    def observe(self, position, layer, experts):
        for expert in experts:
            self.counts[layer][expert] += 1
        self._touch(layer)

    def _rank(self, target):
        return [e for e, _ in sorted(self.counts[target].items(), key=lambda kv: (-kv[1], kv[0]))]


def _consecutive_gaps(positions):
    """Position-to-position gaps; one fewer element than the input."""
    return [b - a for a, b in zip(positions, positions[1:], strict=False)]


class Recurrence(_RankedLayers):
    """Read an expert back before the interval it has settled into says it is
    due. A continuous feeder needs this: the last-seen set is resident by
    construction, so only a due-soon prediction spends idle drive time on
    something the next burst will actually want."""

    name = "recurrence"

    def __init__(self, per_layer=8, horizon=1):
        super().__init__(per_layer=per_layer, horizon=horizon)
        self.hits = defaultdict(lambda: defaultdict(list))

    def observe(self, position, layer, experts):
        for expert in experts:
            self.hits[layer][expert].append(position)
        self._touch(layer)

    def _rank(self, target):
        due = []
        for expert, hits in self.hits[target].items():
            if len(hits) < 2:
                continue
            gaps = _consecutive_gaps(hits)
            due.append((hits[-1] + sum(gaps) / len(gaps), -len(hits), expert))
        return [e for _, _, e in sorted(due)]

    def candidates(self, position, layer, layer_count):
        out, seen = [], set()
        for ahead in range(1, self.horizon + 1):
            target = (layer + ahead) % max(1, layer_count)
            if target not in self.ranked:
                self.ranked[target] = self._rank(target)
            hits = self.hits[target]
            taken = 0
            for expert in self.ranked[target]:
                if taken >= self.per_layer:
                    break
                seen_hits = hits[expert]
                if len(seen_hits) < 2 or (target, expert) in seen:
                    continue
                gaps = _consecutive_gaps(seen_hits)
                if seen_hits[-1] + sum(gaps) / len(gaps) < position:
                    continue
                seen.add((target, expert))
                out.append((target, expert))
                taken += 1
        return out


class _Slots:
    """One layer's slot set, placed by the engine's rule: evict the least
    frequently used, breaking ties toward the least recently used."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.use_count = {}
        self.last_use = {}
        self.clock = 0

    def contains(self, expert):
        return expert in self.use_count

    def touch(self, expert):
        self.clock += 1
        if expert in self.use_count:
            self.use_count[expert] += 1
            self.last_use[expert] = self.clock
            return None
        if len(self.use_count) < self.capacity:
            self.use_count[expert] = 1
            self.last_use[expert] = self.clock
            return None
        victim = min(self.use_count, key=lambda e: (self.use_count[e], self.last_use[e]))
        del self.use_count[victim]
        del self.last_use[victim]
        self.use_count[expert] = 1
        self.last_use[expert] = self.clock
        return victim


class _PlanSlots:
    """One layer's cache, placed the way `makeExpertCachePlan` places it.

    The three points below are transcribed from Plan.swift, not chosen for style.
    Together with the lifetime of the count they are what separates this from the
    textbook rule in `_Slots` (the one the refused timeline uses), which replayed
    on the 2026-10-15 capture with the same warm-up lands at 151.20 misses/token
    where this lands at 89.28 -- outside the band the engine's own counters put it
    in. Dropping any one of the three alone moves the count by at most 1.2%
    (88.24 / 89.31 / 89.28); letting an expert's count die with its slot, which is
    what the textbook rule does, moves it to 102.17 (+14.4%). The engine does not
    do that: `expertUseCount` is a per-layer array sized by the expert count
    (PreadExpertStreamer.swift:326) and only `resetExpertUseCounts()` zeroes it, so
    a returning expert is priced by everything it has ever done.

    * the use count is per *expert*, and is raised for every expert the router
      named -- resident or not (Plan.swift:65-67), so a hit is worth one use
      exactly as a read is;
    * `useClock` advances once per plan, so the ten experts of one layer step
      are co-equal in recency and the next eviction among them is decided by
      count, not by whichever landed last inside the step;
    * every slot is ranked once, before any of this step's arrivals land, and the
      k-th miss takes the k-th weakest slot. An unfilled slot ranks weakest of
      all, because the engine's comparator puts `slotExpert == -1` before any
      expert id -- that is where "fill the free slots first" actually comes from.
    """

    def __init__(self, capacity, use_cap=None):
        self.capacity = capacity
        self.use_count = {}
        self.slot_expert = {}
        self.slot_last_use = {}
        self.clock = 0
        # `None` is the engine as shipped. A number clamps only the *ranking key*,
        # not the count: the engine's `expertUseCount` keeps accumulating
        # (PreadExpertStreamer.swift:326), so a cap prices a slot by its recent
        # demand rather than by everything it has done since the model loaded.
        self.use_cap = use_cap

    def plan(self, experts):
        """Place one layer step and return the experts that were not resident."""
        self.clock += 1
        hit_slots = [slot for slot, expert in self.slot_expert.items() if expert in set(experts)]
        held = set(hit_slots)
        misses = [expert for expert in experts if expert not in self.slot_expert.values()]
        # Evictable: every slot this step did not just hit, ranked weakest first,
        # before any of the step's arrivals land. An unfilled slot has no expert
        # to price, and the engine's comparator sorts `slotExpert == -1` ahead of
        # any id, so free slots are the cheapest victims.
        evictable = sorted(
            (slot for slot in range(self.capacity) if slot not in held),
            key=lambda slot: (
                self._price(slot),
                self.slot_last_use.get(slot, 0),
            ),
        )
        for expert in experts:
            self.use_count[expert] = self.use_count.get(expert, 0) + 1
        for slot in hit_slots:
            self.slot_last_use[slot] = self.clock
        for index, expert in enumerate(misses):
            self.slot_expert[evictable[index]] = expert
            self.slot_last_use[evictable[index]] = self.clock
        return misses

    def contains(self, expert):
        return expert in self.slot_expert.values()

    def _price(self, slot):
        """What this step's ranking charges for the slot: an unfilled slot is free.

        With a cap the two ranks merge and the tie falls to `slot_last_use`, which is
        how a saturating count hands a defended slot back to recency.
        """
        if slot not in self.slot_expert:
            return -1
        count = self.use_count[self.slot_expert[slot]]
        return count if self.use_cap is None else min(count, self.use_cap)

    @property
    def resident_count(self):
        return len(self.slot_expert)


class ReplayResult:
    def __init__(self, fields):
        self.__dict__.update(fields)


def replay_misses(events, layers, slots_per_layer=96, start_position=0):
    """Count the cache misses the engine's own placement rule takes on a trace.

    A feeder policy can only win by moving bytes off the critical path, and the
    cheapest way to screen one offline is its miss count at a fixed slot budget.
    Calibrated against arm B of the 2026-10-15 capture -- the run that wrote the
    trace, so footer and replay are one workload -- the answer is a licence and a
    ban:

    * Licensed -- ranking placement rules at a fixed budget. The replay counts
      89.28 placements/token against the engine's own 92.26 (reads plus ring
      adoptions), 3.2% low, and its miss/token rises 1.794x from 96 slots to 48
      where 5w's paired arms measured 1.831x -- 2.0% error in the response a
      policy change actually moves. The textbook `_Slots` rule replayed on the
      same trace with the same warm-up gives 151.20/token, outside the engine's
      band, which is what makes those two falsifiers bite.
    * Banned -- quoting a token rate from it. The replay sees placements; 5w's
      0.637 ms per avoided miss was measured in drive *reads*, and the 15.5
      placements/token the prefetch ring served without one are invisible in a
      route trace and not fixed -- they depend on how deep the ring ran. So a
      replay delta must be scaled by an unmeasured share before it can be priced,
      and the timeline that would supply it is the refused model (5v).

    Steps at positions before `start_position` warm the cache and place experts
    exactly as later steps do, but none of their traffic is counted: they are
    the prefill rows of a capture, which the engine's decode counters exclude.
    """
    _validate(events, layers)
    cache = [_PlanSlots(slots_per_layer) for _ in range(layers)]
    loaded = set()
    misses = first_visits = reloads = hits = 0
    max_resident = 0
    positions = set()
    for position, layer, experts in events:
        warmup = position < start_position
        if not warmup:
            positions.add(position)
        step_misses = cache[layer].plan(experts)
        if cache[layer].resident_count > max_resident:
            max_resident = cache[layer].resident_count
        if not warmup:
            misses += len(step_misses)
            for expert in step_misses:
                if (layer, expert) in loaded:
                    reloads += 1
                else:
                    first_visits += 1
            hits += len(experts) - len(step_misses)
        for expert in experts:
            loaded.add((layer, expert))
    tokens = len(positions)
    return ReplayResult(
        dict(
            misses=misses,
            first_visits=first_visits,
            reloads=reloads,
            hits=hits,
            tokens=tokens,
            misses_per_token=misses / max(1, tokens),
            hit_rate=hits / max(1, hits + misses),
            max_resident_per_layer=max_resident,
        )
    )


class SimResult:
    def __init__(self, fields, read_ms):
        self.__dict__.update(fields)
        self.read_ms = read_ms


def _validate(events, layers):
    if layers <= 0:
        raise ValueError("layers must be positive")
    for position, layer, experts in events:
        if not 0 <= layer < layers:
            raise ValueError(f"trace names layer {layer}, outside 0..{layers - 1}")
        if not experts:
            raise ValueError(f"trace event {position} {layer} routes to no expert")


def simulate(events, layers, config=None, policy=None, control=None):
    """Replay the trace as a timeline and report what the drive and the host did.

    `control` is the same trace run with `NoFeeder`; passing it turns the
    result's `coverage` into the share of burst reads the feeder removed.
    """
    config = config or FeederConfig()
    policy = policy if policy is not None else NoFeeder()
    _validate(events, layers)
    events = list(events)

    # Event indices that route each (layer, expert). A speculative read is
    # wasted when no later step ever asks for that pair again.
    routed_at = defaultdict(list)
    for index, (_position, layer, experts) in enumerate(events):
        for expert in experts:
            routed_at[(layer, expert)].append(index)

    slots = [_Slots(config.slots_per_layer) for _ in range(layers)]
    in_flight_pairs = set()
    speculative_issues = []
    demand_reads = 0
    demand_bytes = speculative_bytes = wasted_bytes = 0
    exposed_ms = 0.0
    clock = drive_free_at = 0.0
    drive_holds_speculation = False
    speculative_reads = 0

    positions = sorted({position for position, _, _ in events})
    for index, (position, layer, experts) in enumerate(events):
        step_start = clock
        policy.observe(position, layer, experts)

        misses = [e for e in experts if not slots[layer].contains(e)]
        if misses and drive_holds_speculation:
            # Demand arrives while the drive is still mid-speculation, so it
            # waits what is left of that one read -- or the configured cap,
            # which aborts the read and throws its bytes away. This is the cost
            # the wider prefetch arms measured, and the one a feeder has to pay.
            held = max(0.0, drive_free_at - step_start)
            cap = config.max_preempt_delay_ms
            wait = held if cap is None else min(held, cap)
            if cap is not None and held > cap:
                wasted_bytes += config.read_bytes
            exposed_ms += wait
            drive_free_at = step_start + wait
        drive_holds_speculation = False
        for expert in misses:
            drive_free_at = max(drive_free_at, step_start) + config.read_ms
            demand_reads += 1
            demand_bytes += config.read_bytes
            evicted = slots[layer].touch(expert)
            if evicted is not None:
                in_flight_pairs.discard((layer, evicted))
        demand_done = max(drive_free_at, step_start) if misses else step_start

        gpu_done = max(step_start + config.gpu_ms_per_step, demand_done)
        exposed_ms += max(0.0, demand_done - (step_start + config.gpu_ms_per_step))
        clock = gpu_done + config.host_ms_per_step

        # The drive is idle from demand_done to clock: the GPU is still on this
        # layer and the host is doing the next one's bookkeeping. That interval
        # is the whole subject of this model.
        budget = clock - demand_done
        limit = config.max_speculative_reads
        for target, expert in policy.candidates(position, layer, layers):
            if budget < config.read_ms:
                break
            if limit is not None and speculative_reads >= limit:
                break
            pair = (target, expert)
            if slots[target].contains(expert) or pair in in_flight_pairs:
                continue
            in_flight_pairs.add(pair)
            speculative_issues.append((index, pair))
            drive_free_at += config.read_ms
            budget -= config.read_ms
            speculative_reads += 1
            speculative_bytes += config.read_bytes
            drive_holds_speculation = True
            evicted = slots[target].touch(expert)
            if evicted is not None:
                in_flight_pairs.discard((target, evicted))

    for issue_index, (target, expert) in speculative_issues:
        later = routed_at[(target, expert)]
        if bisect.bisect_right(later, issue_index) == len(later):
            wasted_bytes += config.read_bytes

    drive_busy_ms = (demand_reads + speculative_reads) * config.read_ms
    window_ms = max(clock, drive_free_at)
    removed = 0 if control is None else max(0, control.demand_reads - demand_reads)
    fields = dict(
        policy=getattr(policy, "name", "feeder"),
        positions=len(positions),
        steps=len(events),
        demand_reads=demand_reads,
        demand_bytes=demand_bytes,
        speculative_reads=speculative_reads,
        speculative_bytes=speculative_bytes,
        wasted_bytes=wasted_bytes,
        exposed_ms=exposed_ms,
        drive_busy_ms=drive_busy_ms,
        window_ms=window_ms,
        window_bytes=window_ms * config.bandwidth_bytes_per_ms,
        token_ms=clock / max(1, len(positions)),
        coverage=(removed / control.demand_reads) if control else 0.0,
        duty=drive_busy_ms / max(1e-9, window_ms),
    )
    return SimResult(fields, config.read_ms)


def load_trace(path):
    """Read a TINYTITAN_ROUTE_TRACE capture: `position layer e0 ... e(k-1)`."""
    opener = gzip.open if path.endswith(".gz") else open
    events = []
    with opener(path, "rt", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 3:
                raise TraceFormatError(
                    f"{path}:{line_number}: expected 'position layer expert...', got {line!r}"
                )
            try:
                position = int(fields[0])
                layer = int(fields[1])
                experts = [int(field) for field in fields[2:]]
            except ValueError as error:
                raise TraceFormatError(f"{path}:{line_number}: not an integer ({error})") from error
            if position < 0 or layer < 0 or any(e < 0 for e in experts):
                raise TraceFormatError(f"{path}:{line_number}: negative index")
            events.append((position, layer, experts))
    if not events:
        raise TraceFormatError(f"{path}: no route events")
    return events


ARMS = (
    ("burst", lambda: NoFeeder()),
    ("prev_token", lambda: PrevToken(depth=6)),
    ("recency3", lambda: RecencyWindow(width=3, depth=4)),
    ("hot_set8", lambda: HotSet(per_layer=8)),
    ("recurrence", lambda: Recurrence(per_layer=8, horizon=6)),
)

# A burst arm that cannot reproduce the measured clock means the timeline is
# missing per-step cost, so its rates are no longer projections of the engine.
CALIBRATION_TOLERANCE = 0.10


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Project the continuous weight feeder onto a route trace."
    )
    parser.add_argument("trace", help="TINYTITAN_ROUTE_TRACE capture, plain or .gz")
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--slots-per-layer", type=int, default=96)
    parser.add_argument(
        "--skip-steps",
        type=int,
        default=0,
        help="drop leading steps (skip the prefill rows of a capture)",
    )
    parser.add_argument("--measured-tok-s", type=float, default=None)
    args = parser.parse_args(argv)

    if args.measured_tok_s is not None and args.measured_tok_s <= 0:
        print("error: --measured-tok-s must be positive", file=sys.stderr)
        return 1

    try:
        events = load_trace(args.trace)[args.skip_steps :]
    except TraceFormatError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    config = FeederConfig(slots_per_layer=args.slots_per_layer)
    try:
        control = simulate(events, layers=args.layers, config=config)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        f"events={len(events)} positions={control.positions} "
        f"layers={args.layers} slots={args.slots_per_layer} "
        f"read_ms={config.read_ms:.3f} gpu={config.gpu_ms_per_step:.3f} "
        f"host={config.host_ms_per_step:.3f}"
    )
    projected = 1000.0 / max(1e-9, control.token_ms)
    label = "PROJ"
    if args.measured_tok_s is None:
        gate = (
            "calibration: no --measured-tok-s given, so these projections are "
            "unchecked against any clock"
        )
    else:
        error = abs(projected - args.measured_tok_s) / args.measured_tok_s
        if error <= CALIBRATION_TOLERANCE:
            gate = (
                f"calibration: within {CALIBRATION_TOLERANCE:.0%} (measured "
                f"{args.measured_tok_s:.2f} against burst projection "
                f"{projected:.2f} tok/s)"
            )
        else:
            gate = (
                f"calibration: REJECTED (burst projects {projected:.2f} against "
                f"measured {args.measured_tok_s:.2f} tok/s, {error:.1%} off) -- this "
                "timeline is missing per-step cost, so the rates are withheld and only "
                "the between-arm deltas below are quotable"
            )
            label = "REFUSED"
    for name, build in ARMS:
        result = simulate(
            events,
            layers=args.layers,
            config=config,
            policy=build(),
            control=control,
        )
        print(
            f"{name:11s} demand={result.demand_reads:6d} "
            f"spec={result.speculative_reads:7d} cov={result.coverage:.3f} "
            f"waste_MB={result.wasted_bytes / 1e6:8.1f} "
            f"exposed_ms={result.exposed_ms:9.1f} duty={result.duty:.3f} "
            f"{label} token={result.token_ms:7.2f}ms "
            f"rate={1000.0 / max(1e-9, result.token_ms):5.2f} tok/s"
        )
    print(gate)
    print(
        "note: burst is the shipped arm; every rate is a projection through "
        "this timeline, not a measurement"
    )
    return 1 if label == "REFUSED" else 0


if __name__ == "__main__":
    sys.exit(main())
