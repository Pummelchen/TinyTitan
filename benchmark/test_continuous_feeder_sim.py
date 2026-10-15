"""Tests for the continuous-weight-feeder timeline model.

Each test names the production change that would make it fail: a drive that is
one serial, demand-preempted server; a feeder that may only issue reads in the
intervals the demand reader leaves free; and a cache placed by the engine's own
LFU-with-LRU-tiebreak rule.
"""

import gzip
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO

from continuous_feeder_sim import (
    FeederConfig,
    HotSet,
    NoFeeder,
    PrevToken,
    RecencyWindow,
    Recurrence,
    TraceFormatError,
    _Slots,
    load_trace,
    main,
    replay_misses,
    simulate,
)

# Layer 0 asks for a different pair every token and cycles back with period 3.
# The pair is always evicted before its turn comes round again, so a feeder
# that reads it back early is doing useful work, and one that reads the
# last-seen pair is not: the last-seen pair is resident by construction.
CYCLE = [
    "0 0 1 2",
    "1 0 3 4",
    "2 0 5 6",
    "3 0 1 2",
    "4 0 3 4",
    "5 0 5 6",
    "6 0 1 2",
    "7 0 3 4",
]

# Expert 9 is routed at position 0 and never again. Expert 6 likewise.
DEAD = [
    "0 0 9 6 1 2",
    "1 0 1 2",
    "2 0 2 1",
    "3 0 1 2",
]


def write_trace(lines):
    handle, path = tempfile.mkstemp(suffix=".txt")
    os.close(handle)
    with open(path, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    return path


class TraceLoadingTests(unittest.TestCase):
    def test_plain_and_gzipped_files_parse_to_the_same_events(self):
        plain = write_trace(CYCLE)
        gz = plain + ".gz"
        with open(plain, "rb") as src, gzip.open(gz, "wb") as dst:
            dst.write(src.read())
        try:
            self.assertEqual(load_trace(plain), load_trace(gz))
            self.assertEqual(load_trace(plain)[0], (0, 0, [1, 2]))
            self.assertEqual(load_trace(plain)[2], (2, 0, [5, 6]))
        finally:
            os.unlink(plain)
            os.unlink(gz)

    def test_rejects_a_line_without_a_top_k(self):
        path = write_trace(["0 0"])
        with self.assertRaises(TraceFormatError):
            load_trace(path)
        os.unlink(path)

    def test_rejects_a_non_numeric_expert(self):
        path = write_trace(["0 0 3 q"])
        with self.assertRaises(TraceFormatError):
            load_trace(path)
        os.unlink(path)

    def test_rejects_a_non_numeric_position(self):
        path = write_trace(["x 0 3"])
        with self.assertRaises(TraceFormatError):
            load_trace(path)
        os.unlink(path)


class GeometryValidationTests(unittest.TestCase):
    def test_rejects_a_non_positive_bandwidth(self):
        with self.assertRaises(ValueError):
            FeederConfig(bandwidth_bytes_per_ms=0.0)

    def test_rejects_a_non_positive_slot_count(self):
        with self.assertRaises(ValueError):
            FeederConfig(slots_per_layer=0)

    def test_rejects_a_read_larger_than_the_stride(self):
        with self.assertRaises(ValueError):
            FeederConfig(expert_stride=100, read_bytes=101)

    def test_rejects_a_layer_outside_the_geometry(self):
        events = [(0, 3, [1])]
        with self.assertRaises(ValueError):
            simulate(events, layers=3)


class PolicyHistoryTests(unittest.TestCase):
    """A policy may only ever see routes the replay has already reached."""

    def test_prev_token_proposes_the_layer_aheads_last_route_set(self):
        policy = PrevToken(depth=1)
        policy.observe(0, 0, [1, 2])
        policy.observe(0, 1, [3, 4])
        self.assertEqual(policy.candidates(1, 0, 2), [(1, 3), (1, 4)])

    def test_prev_token_sees_nothing_before_the_first_route(self):
        policy = PrevToken(depth=1)
        policy.observe(0, 0, [1, 2])
        self.assertEqual(policy.candidates(0, 0, 2), [])

    def test_recency_window_widens_only_by_the_requested_positions(self):
        policy = RecencyWindow(width=2, depth=1)
        for position, experts in [(0, [9, 7]), (1, [7]), (2, [7, 8])]:
            policy.observe(position, 1, experts)
        policy.observe(3, 0, [1])
        self.assertEqual(sorted(pair[1] for pair in policy.candidates(3, 0, 2)), [7, 8])
        wide = RecencyWindow(width=3, depth=1)
        for position, experts in [(0, [9, 7]), (1, [7]), (2, [7, 8])]:
            wide.observe(position, 1, experts)
        self.assertEqual(sorted(pair[1] for pair in wide.candidates(2, 0, 2)), [7, 8, 9])

    def test_no_feeder_proposes_nothing(self):
        policy = NoFeeder()
        policy.observe(0, 0, [1, 2])
        self.assertEqual(policy.candidates(1, 0, 48), [])

    def test_recurrence_needs_two_sightings_before_it_predicts(self):
        policy = Recurrence(per_layer=8, horizon=6)
        policy.observe(0, 0, [1, 2])
        self.assertEqual(policy.candidates(1, 0, 1), [])
        policy.observe(3, 0, [1, 2])
        self.assertEqual(sorted(pair[1] for pair in policy.candidates(3, 0, 1)), [1, 2])


class ControlArmTests(unittest.TestCase):
    """The burst-only arm is shipped behaviour, so every projection is measured
    against it and against no other baseline."""

    def setUp(self):
        self.path = write_trace(CYCLE)
        self.events = load_trace(self.path)
        self.config = FeederConfig(slots_per_layer=4)

    def tearDown(self):
        os.unlink(self.path)

    def test_burst_only_arm_pays_for_every_demand_miss(self):
        result = simulate(self.events, layers=1, config=self.config, policy=NoFeeder())
        # Eight steps, two experts each, four slots: the cycle always turns
        # over, so nothing is resident when it is asked for again.
        self.assertEqual(result.demand_reads, 16)
        self.assertEqual(result.speculative_reads, 0)
        self.assertEqual(result.wasted_bytes, 0)
        self.assertEqual(result.coverage, 0.0)

    def test_burst_only_leaves_the_drive_idle_over_the_window(self):
        result = simulate(self.events, layers=1, config=self.config, policy=NoFeeder())
        self.assertLess(result.drive_busy_ms, result.window_ms)
        self.assertLess(result.duty, 1.0)

    def test_burst_hides_its_reads_under_the_gpu_when_they_are_short(self):
        result = simulate(self.events, layers=1, config=self.config, policy=NoFeeder())
        self.assertEqual(result.exposed_ms, 0.0)


class FeederArmTests(unittest.TestCase):
    def setUp(self):
        self.path = write_trace(CYCLE)
        self.events = load_trace(self.path)
        self.config = FeederConfig(slots_per_layer=4)
        self.control = simulate(self.events, layers=1, config=self.config, policy=NoFeeder())

    def tearDown(self):
        os.unlink(self.path)

    def test_a_due_soon_feeder_removes_demand_reads(self):
        fed = simulate(
            self.events,
            layers=1,
            config=self.config,
            policy=Recurrence(per_layer=8, horizon=6),
            control=self.control,
        )
        self.assertGreater(fed.speculative_reads, 0)
        self.assertLess(fed.demand_reads, self.control.demand_reads)
        self.assertGreater(fed.coverage, 0.0)

    def test_a_last_seen_feeder_cannot_remove_anything(self):
        """The resident set is the last-seen set, so the ring's own prediction
        shape has nothing to gain from more drive time. This is the finding the
        deeper prefetch arms already paid for."""
        fed = simulate(
            self.events,
            layers=1,
            config=self.config,
            policy=PrevToken(depth=1),
            control=self.control,
        )
        self.assertEqual(fed.demand_reads, self.control.demand_reads)
        self.assertEqual(fed.coverage, 0.0)

    def test_no_arm_reads_more_bytes_than_the_drive_can_carry(self):
        fed = simulate(
            self.events,
            layers=1,
            config=self.config,
            policy=Recurrence(per_layer=8, horizon=6),
            control=self.control,
        )
        self.assertLessEqual(fed.demand_bytes + fed.speculative_bytes, fed.window_bytes + 1e-9)

    def test_demand_pays_at_most_one_read_for_speculation(self):
        """A speculative read may delay demand by one read's time per step and
        no more, because the drive is modelled as one serial server."""
        config = FeederConfig(
            slots_per_layer=4,
            read_bytes=1000,
            bandwidth_bytes_per_ms=1000.0,
            max_speculative_reads=10**6,
        )
        fed = simulate(
            self.events,
            layers=1,
            config=config,
            policy=Recurrence(per_layer=8, horizon=6),
            control=simulate(self.events, layers=1, config=config),
        )
        self.assertLessEqual(fed.exposed_ms, fed.steps * config.read_ms + 1e-9)

    def test_a_speculation_cap_leaves_the_demand_arm_untouched(self):
        capped = FeederConfig(slots_per_layer=4, max_speculative_reads=0)
        fed = simulate(
            self.events,
            layers=1,
            config=capped,
            policy=Recurrence(per_layer=8, horizon=6),
            control=self.control,
        )
        self.assertEqual(fed.speculative_reads, 0)
        self.assertEqual(fed.demand_reads, self.control.demand_reads)


class WasteTests(unittest.TestCase):
    def test_a_read_of_an_expert_no_route_ever_asks_for_again_is_wasted(self):
        path = write_trace(DEAD)
        events = load_trace(path)
        config = FeederConfig(slots_per_layer=2)
        result = simulate(events, layers=1, config=config, policy=HotSet(per_layer=4))
        os.unlink(path)
        self.assertGreater(result.speculative_reads, 0)
        self.assertGreater(result.wasted_bytes, 0)


class CalibrationGateTests(unittest.TestCase):
    """The timeline is only worth projecting once its burst arm reproduces the
    measured clock. A control arm that misses the measurement by more than the
    tolerance means the model is missing per-step cost, and every rate it
    prints after that is a number the reader will quote."""

    def run_main(self, argv):
        out = StringIO()
        with redirect_stdout(out):
            status = main(argv)
        return status, out.getvalue()

    def test_refuses_every_projection_when_the_control_arm_misses_the_measurement(
        self,
    ):
        path = write_trace(CYCLE)
        try:
            status, text = self.run_main(
                [path, "--layers", "1", "--slots-per-layer", "4", "--measured-tok-s", "1.0"]
            )
        finally:
            os.unlink(path)
        self.assertEqual(status, 1)
        self.assertNotIn("PROJ", text)
        self.assertIn("calibration: REJECTED", text)

    def test_keeps_the_projections_when_the_control_arm_calibrates(self):
        path = write_trace(CYCLE)
        try:
            events = load_trace(path)
            control = simulate(events, layers=1, config=FeederConfig(slots_per_layer=4))
            measured = 1000.0 / control.token_ms
            status, text = self.run_main(
                [
                    path,
                    "--layers",
                    "1",
                    "--slots-per-layer",
                    "4",
                    "--measured-tok-s",
                    f"{measured:.6f}",
                ]
            )
        finally:
            os.unlink(path)
        self.assertEqual(status, 0)
        self.assertIn("PROJ", text)
        self.assertIn("calibration: within", text)

    def test_a_measurement_is_optional_and_the_arms_still_run(self):
        path = write_trace(CYCLE)
        try:
            status, text = self.run_main([path, "--layers", "1", "--slots-per-layer", "4"])
        finally:
            os.unlink(path)
        self.assertEqual(status, 0)
        self.assertIn("PROJ", text)
        self.assertIn("calibration: no --measured-tok-s", text)

    def test_refuses_a_measured_rate_that_cannot_be_a_clock(self):
        path = write_trace(CYCLE)
        try:
            status, text = self.run_main(
                [path, "--layers", "1", "--slots-per-layer", "4", "--measured-tok-s", "0"]
            )
        finally:
            os.unlink(path)
        self.assertEqual(status, 1)
        self.assertNotIn("PROJ", text)


HERE = os.path.dirname(os.path.abspath(__file__))
V520_TRACE = os.path.join(HERE, "expert-cache", "route-trace-qwen38-4bit-v520-20261015.txt.gz")
# The trace's prompt is "Write a detailed essay about the history of computing.",
# which the engine's own footer records as prefill=10tok, so decode starts at
# position 10. Counting from 0 would price the prefill's cold cache as decode.
PREFILL_POSITIONS = 10
DECODE_TOKENS = 511
# Arm B of docs/qwen38-decode-12gb-measurements.md 5v -- the run that WROTE this
# trace, so footer and replay are the same workload:
#   expert_misses=37724  prefetch_adopted_per_token=18.40  new=512tok
# A placement replay counts a miss the engine served out of the prefetch ring's
# staging as a drive read, so the engine's read count is the replay's floor and
# reads-plus-adoptions is its ceiling.
ENGINE_READ_MISSES_PER_TOKEN = 37724 / DECODE_TOKENS
ENGINE_PLACEMENTS_PER_TOKEN = (37724 + 18.40 * 512) / DECODE_TOKENS
# 5w's interleaved A/B on byte-identical output: 137.3 misses/token at 48 slots
# against 74.99 at 96. How a policy responds to capacity is the quantity an
# offline screen has to get right, and it is measurable on the real engine.
ENGINE_SLOT_RATIO = (28795 / 384 + 62.3) / (28795 / 384)


class MissReplayTests(unittest.TestCase):
    """A pure miss replay, deliberately without the timeline.

    The timeline is refused (its control arm projects 27.1% off a measured clock),
    but a miss *count* is a structural property of the trace and the placement
    rule, and 5w measured the conversion from a miss to critical-path time
    directly: 62.3 fewer misses per token bought 39.7 ms of gap, so one avoided
    miss is 0.637 ms. These tests pin the replay to `makeExpertCachePlan`'s own
    semantics and then to what the engine actually did, before any policy is
    allowed to be compared against it.
    """

    def setUp(self):
        self.events = load_trace(V520_TRACE)

    def decode_replay(self, slots_per_layer=96):
        return replay_misses(
            self.events,
            layers=48,
            slots_per_layer=slots_per_layer,
            start_position=PREFILL_POSITIONS,
        )

    def textbook_replay(self, slots_per_layer=96):
        """`_Slots` driven by the replay's own warm-up and counting, as a drift arm.

        Lives here, not in the module: nothing in the shipped analysis replays with
        this rule, and the only thing it is good for is proving that the band above
        can fail.
        """
        cache = [_Slots(slots_per_layer) for _ in range(48)]
        misses = 0
        positions = set()
        for position, layer, experts in self.events:
            absent = [expert for expert in experts if not cache[layer].contains(expert)]
            for expert in experts:
                cache[layer].touch(expert)
            if position >= PREFILL_POSITIONS:
                positions.add(position)
                misses += len(absent)
        return misses / len(positions)

    def test_a_first_visit_and_a_reload_are_both_misses_and_a_resident_expert_is_not(self):
        # One layer, two slots. Hand-traced through the engine's rule: 1 and 2
        # arrive (two first visits), 2 is hit (its count rises to 2), 3 then
        # evicts 1 because 1 is the weaker slot at one use, and 1 comes back as a
        # miss that is a reload -- not a first visit, and not a hit either.
        path = write_trace(["0 0 1", "1 0 2", "2 0 2", "3 0 3", "4 0 1"])
        try:
            result = replay_misses(load_trace(path), layers=1, slots_per_layer=2)
        finally:
            os.unlink(path)
        self.assertEqual(result.misses, 4)
        self.assertEqual(result.first_visits, 3)
        self.assertEqual(result.reloads, 1)
        self.assertEqual(result.hits, 1)
        self.assertEqual(result.tokens, 5)

    def test_a_hit_raises_the_use_count_so_a_hot_expert_outlives_a_cold_one(self):
        # `makeExpertCachePlan` bumps `expertUseCount` for every expert the router
        # named, resident or not (Plan.swift:65-67). If a hit left the count where
        # it was, experts 1, 2 and 3 would tie at one use with the two churned
        # newcomers and be evicted themselves; because each of them was hit once,
        # they outrank every single-use arrival and the cold experts 4 and 5 are
        # what the cache gives up -- so neither is ever seen twice.
        path = write_trace(["0 0 1 2 3", "1 0 1 2 3", "2 0 4", "3 0 5", "4 0 1 2 3 6"])
        try:
            result = replay_misses(load_trace(path), layers=1, slots_per_layer=4)
        finally:
            os.unlink(path)
        self.assertEqual(result.misses, 6)
        self.assertEqual(result.first_visits, 6, "a hot expert was evicted and re-read")
        self.assertEqual(result.reloads, 0, "a resident expert was treated as absent")
        self.assertEqual(result.hits, 6)

    def test_no_layer_holds_more_resident_experts_than_its_slot_count(self):
        self.assertLessEqual(self.decode_replay().max_resident_per_layer, 96)

    def test_the_replay_over_states_drive_reads_by_what_the_prefetch_ring_hid(self):
        """The absolute level is known-biased, and pinned so it stays known.

        A route trace records what the router asked for, never what the ring had
        already staged, so every replay miss is counted as a drive read. The
        replay must sit above the engine's read count and below its total
        placements -- if it leaves that band, the placement rule has drifted and
        no delta from it may be quoted.

        Measured here: 89.28 placements/token, 21% above the engine's 73.82 reads
        and 3.2% below its 92.26 placements. The 5% headroom over the ceiling is
        not slack for its own sake -- the drift arm below replays the textbook
        rule on this same trace and shows it breaching the ceiling at
        151.20/token, which is what this test is hunting.
        """
        result = self.decode_replay()
        self.assertGreater(result.misses_per_token, ENGINE_READ_MISSES_PER_TOKEN)
        self.assertLess(
            result.misses_per_token,
            ENGINE_PLACEMENTS_PER_TOKEN * 1.05,
            msg=(
                f"replay {result.misses_per_token:.2f}/token against the engine's "
                f"{ENGINE_READ_MISSES_PER_TOKEN:.2f} reads and "
                f"{ENGINE_PLACEMENTS_PER_TOKEN:.2f} placements: outside the "
                "ring-hidden band, so the placement rule no longer tracks the engine"
            ),
        )

    def test_the_replay_reproduces_the_engines_measured_response_to_slot_count(self):
        """The falsifier, on the quantity an offline screen actually needs.

        5w changed the capacity on the real machine and got 1.831x the misses per
        token at 48 slots against 96. A replay that reproduces that ratio can
        rank two placement rules at a fixed budget; one that does not cannot, and
        its absolute numbers are worthless either way.

        Measured here: 160.13 at 48 against 89.28 at 96, 1.794x -- 2.0% off the
        machine's own ratio.
        """
        wide = self.decode_replay(96).misses_per_token
        narrow = self.decode_replay(48).misses_per_token
        self.assertLess(
            abs(narrow / wide - ENGINE_SLOT_RATIO) / ENGINE_SLOT_RATIO,
            0.10,
            msg=(
                f"replay says {narrow / wide:.3f}x for 48 against 96 slots, the "
                f"engine measured {ENGINE_SLOT_RATIO:.3f}x"
            ),
        )

    def test_the_band_rejects_a_rule_that_prices_a_returning_expert_at_one(self):
        """The band above is a drift detector, and this arm shows it firing.

        `_Slots` is the rule the refused timeline uses. It keys residency by the
        expert and drops the expert's count when it is evicted, so a returning
        expert starts over at one use -- where the engine's `expertUseCount` is a
        per-layer array that outlives the slot and is zeroed only by
        `resetExpertUseCounts()` (PreadExpertStreamer.swift:326). Replayed on the
        same trace with the same warm-up that rule counts 151.20 placements/token,
        64% over the engine's 92.26, where the plan-faithful replay counts 89.28
        and sits inside the band. If this test ever passes quietly while the
        plan-faithful replay also breaches the ceiling, the band has stopped
        discriminating and no ranking built on it may be quoted.
        """
        textbook = self.textbook_replay()
        self.assertGreater(
            textbook,
            ENGINE_PLACEMENTS_PER_TOKEN * 1.05,
            msg=(
                f"textbook rule counts {textbook:.2f}/token; the ceiling is "
                f"{ENGINE_PLACEMENTS_PER_TOKEN * 1.05:.2f} -- the band no longer "
                "separates the engine's placement rule from the textbook one"
            ),
        )


if __name__ == "__main__":
    unittest.main()
