"""Tests for the offline flat-pair expert-cache simulator (stage 0).

Every expected value here is computable by hand from the policy definition;
none is taken from a simulator output. Run:

    cd benchmark && python3 -m unittest test_flat_pair_cache_sim -v
"""

import os
import tempfile
import unittest

from flat_pair_cache_sim import (
    TraceFormatError,
    allocate_heat_slots,
    parse_trace,
    project_token_ms,
    simulate_flat,
    simulate_heat,
    simulate_uniform,
)


def write_trace(lines):
    fd, path = tempfile.mkstemp(suffix=".trace")
    with os.fdopen(fd, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


class ParseTraceTests(unittest.TestCase):
    def test_parses_position_layer_experts(self):
        path = write_trace(["0 0 7 11", "0 1 3", "1 0 7 11 250"])
        events = parse_trace(path)
        self.assertEqual(
            events,
            [(0, 0, [7, 11]), (0, 1, [3]), (1, 0, [7, 11, 250])],
        )

    def test_refuses_a_malformed_field_naming_the_line(self):
        path = write_trace(["0 0 7", "1 0 8 x"])
        with self.assertRaises(TraceFormatError) as caught:
            parse_trace(path)
        self.assertIn("line 2", str(caught.exception))

    def test_refuses_a_line_with_no_experts(self):
        path = write_trace(["0 0"])
        with self.assertRaises(TraceFormatError):
            parse_trace(path)

    def test_refuses_a_missing_file(self):
        with self.assertRaises(TraceFormatError):
            parse_trace("/nonexistent/qwen38.trace")

    def test_refuses_an_empty_trace(self):
        path = write_trace([])
        with self.assertRaises(TraceFormatError):
            parse_trace(path)


class UniformTests(unittest.TestCase):
    """The control: today's per-layer slot cache, LFU with LRU tiebreak."""

    def test_lfu_hits_a_reused_expert(self):
        # layer 0, 2 slots: load 1, load 2, hit 1, evict 2 for 3.
        trace = [(0, 0, [1]), (1, 0, [2]), (2, 0, [1]), (3, 0, [3])]
        result = simulate_uniform(trace, slots_per_layer=2, experts_per_layer=512)
        self.assertEqual((result.hits, result.misses), (1, 3))

    def test_layers_have_independent_pools(self):
        # A hot expert in layer 0 must not survive because layer 1 is quiet.
        trace = [(0, 0, [1]), (1, 1, [1]), (2, 0, [1])]
        result = simulate_uniform(trace, slots_per_layer=1, experts_per_layer=512)
        # Layer 0: 1 miss then 1 hit. Layer 1: 1 miss.
        self.assertEqual((result.hits, result.misses), (1, 2))

    def test_lfu_prefers_the_frequently_used_expert_for_eviction(self):
        # 2 slots: 1, 2, 1, 1 -> counts 1:3, 2:1; loading 3 evicts 2 (both
        # LRU orders would evict 2 as well, so make LFU disagree with LRU):
        # 1, 2, 2, 1 -> counts 1:2, 2:2, tie -> LRU evicts 1; then 3 evicts 2.
        trace = [(0, 0, [1]), (1, 0, [2]), (2, 0, [2]), (3, 0, [1]), (4, 0, [3])]
        result = simulate_uniform(trace, slots_per_layer=2, experts_per_layer=512)
        self.assertEqual((result.hits, result.misses), (2, 3))

    def test_refuses_a_zero_slot_count(self):
        with self.assertRaises(ValueError):
            simulate_uniform([], slots_per_layer=0, experts_per_layer=512)


class FlatTests(unittest.TestCase):
    """One global pool of (layer, expert) pairs."""

    def test_a_pair_survives_across_layers_of_a_shared_budget(self):
        # 2 pairs total: (0,1) and (1,1) both stay because the pool is shared,
        # where a 1-slot-per-layer uniform cache would evict (0,1).
        trace = [(0, 0, [1]), (1, 1, [1]), (2, 0, [1]), (3, 1, [1])]
        result = simulate_flat(trace, total_pairs=2, experts_per_layer=512)
        self.assertEqual((result.hits, result.misses), (2, 2))

    def test_flat_evicts_the_globally_least_frequent_pair(self):
        # 1 pair: (0,1) used twice, then (1,2) once, then (0,1) again -> hit.
        trace = [(0, 0, [1]), (1, 0, [1]), (2, 1, [2]), (3, 0, [1])]
        result = simulate_flat(trace, total_pairs=1, experts_per_layer=512)
        self.assertEqual((result.hits, result.misses), (1, 3))

    def test_keys_distinguish_layers(self):
        # 2 pairs: (0,1) and (1,1) are different keys even with the same expert.
        trace = [(0, 0, [1]), (1, 1, [1]), (2, 0, [1])]
        result = simulate_flat(trace, total_pairs=2, experts_per_layer=512)
        self.assertEqual((result.hits, result.misses), (1, 2))


class HeatAllocationTests(unittest.TestCase):
    def test_sum_never_exceeds_total(self):
        demand = {layer: 100 + layer for layer in range(48)}
        allocation = allocate_heat_slots(
            per_layer_distinct=demand, total_slots=100, experts_per_layer=512
        )
        self.assertEqual(sum(allocation.values()), 100)

    def test_respects_the_per_layer_cap(self):
        demand = {0: 512, 1: 512}
        allocation = allocate_heat_slots(
            per_layer_distinct=demand, total_slots=4096, experts_per_layer=512
        )
        self.assertEqual(allocation[0], 512)
        self.assertEqual(allocation[1], 512)

    def test_a_quieter_layer_gets_fewer_slots(self):
        demand = {0: 200, 1: 20}
        allocation = allocate_heat_slots(
            per_layer_distinct=demand, total_slots=220, experts_per_layer=512
        )
        self.assertGreater(allocation[0], allocation[1])

    def test_unnamed_layers_get_zero(self):
        allocation = allocate_heat_slots(
            per_layer_distinct={0: 100}, total_slots=50, experts_per_layer=512
        )
        self.assertEqual(allocation[0], 50)
        self.assertNotIn(1, allocation)

    def test_deterministic_under_a_tie(self):
        demand = {0: 10, 1: 10}
        first = allocate_heat_slots(
            per_layer_distinct=demand, total_slots=11, experts_per_layer=512
        )
        second = allocate_heat_slots(
            per_layer_distinct=demand, total_slots=11, experts_per_layer=512
        )
        self.assertEqual(first, second)
        # The tie-break gives the surplus to the lower layer index.
        self.assertEqual(first[0], 6)
        self.assertEqual(first[1], 5)

    def test_refuses_a_negative_or_zero_total(self):
        with self.assertRaises(ValueError):
            allocate_heat_slots(per_layer_distinct={0: 1}, total_slots=0, experts_per_layer=512)


class HeatSimulationTests(unittest.TestCase):
    def test_allocation_reaches_the_streamers(self):
        # Layer 0 is hot (distinct 3), layer 1 quiet (distinct 1); with 3
        # total pairs the hot layer must hold 2 and still hit its repeat.
        trace = [(0, 0, [1]), (0, 0, [2]), (1, 0, [3]), (2, 0, [1]), (3, 1, [1]), (4, 0, [1])]
        result = simulate_heat(trace, total_pairs=3, experts_per_layer=512, layers=2)
        self.assertEqual(result.per_layer_slots, {0: 2, 1: 1})
        # Layer 0: miss(1), miss(2), miss(3) evicts 1 (all counts tie at 1,
        # LRU tiebreak), miss(1), hit(1). Layer 1: miss. LFU-first with an
        # LRU tiebreak, matching the engine's shouldEvictSlot.
        self.assertEqual((result.hits, result.misses), (1, 5))


class ProjectionTests(unittest.TestCase):
    def test_token_ms_is_gpu_plus_turnaround_plus_exposed_io(self):
        # 100 MB at 3.6 GB/s is ~27.78 ms; 105 + 20 + 27.78 = 152.78.
        ms = project_token_ms(exposed_bytes=100_000_000)
        self.assertAlmostEqual(ms, 105 + 20 + 100_000_000 / 3.6e9 * 1000, places=2)

    def test_zero_exposed_bytes_hits_the_compute_floor(self):
        self.assertEqual(project_token_ms(exposed_bytes=0), 125.0)

    def test_refuses_a_negative_size(self):
        with self.assertRaises(ValueError):
            project_token_ms(exposed_bytes=-1)


if __name__ == "__main__":
    unittest.main()
