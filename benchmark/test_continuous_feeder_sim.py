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
    load_trace,
    main,
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


if __name__ == "__main__":
    unittest.main()
