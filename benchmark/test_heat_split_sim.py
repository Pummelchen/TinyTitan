"""Tests for the per-layer slot-budget splitter.

Why this exists: the replay of `continuous_feeder_sim` is calibrated (§5y), and the first
thing it said is that at the shipped 4 608-slot budget the *victim order* is already the
best one available, while the *partition* of that budget across the 48 layers is uniform
and the per-layer working sets are not. These tests pin the splitter that turns a per-layer
heat profile into slot counts, and -- the one that matters -- the claim that a profile
measured on one capture is worth most of an oracle's gain on another.

Model-free: three committed route traces, no install, no GPU.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from heat_split_sim import (
    CAPTURES,
    EXPERTS_PER_LAYER,
    chance_pick_overlap,
    demand_per_token,
    heat_profile,
    load_trace,
    placements_per_token,
    profile_correlation,
    route_agreement,
    route_rows,
    split_slots,
    two_pass_placements,
)


def events(lines):
    return [
        (int(parts[0]), int(parts[1]), [int(value) for value in parts[2:]])
        for parts in (line.split() for line in lines)
    ]


class SplitSlotsTests(unittest.TestCase):
    def test_the_split_spends_exactly_the_budget_it_is_given(self):
        profile = [100, 50, 50]
        budget = split_slots(profile, total_slots=96)
        self.assertEqual(sum(budget), 96)
        self.assertEqual(len(budget), 3)

    def test_a_hotter_layer_never_gets_fewer_slots_than_a_colder_one(self):
        budget = split_slots([10, 20, 40], total_slots=90)
        self.assertLessEqual(budget[0], budget[1])
        self.assertLessEqual(budget[1], budget[2])

    def test_the_floor_holds_and_the_heaviest_layers_absorb_the_remainder(self):
        # 1 000 slots over a profile where one layer is 500x the others: the cold layers
        # cannot go below the floor, and the rounding remainder must not be lost.
        budget = split_slots([500] + [1] * 47, total_slots=1_000, floor=8)
        self.assertEqual(sum(budget), 1_000)
        self.assertGreaterEqual(min(budget[1:]), 8)
        self.assertGreater(budget[0], budget[1])

    def test_a_budget_that_cannot_clear_the_floor_is_refused(self):
        with self.assertRaises(ValueError):
            split_slots([1, 1, 1], total_slots=9, floor=4)


class HeatProfileTests(unittest.TestCase):
    def test_it_counts_distinct_experts_per_layer_and_skips_the_prefill_rows(self):
        lines = [
            "0 0 1 1 2",  # prefill: same expert twice, must count once and not at all
            "1 0 3 4",
            "2 0 4",
            "1 1 7 8 9",
        ]
        profile = heat_profile(events(lines), layers=2, start_position=1)
        self.assertEqual(profile, [2, 3])


class UseCapTests(unittest.TestCase):
    def test_clamping_the_count_hands_the_slot_to_recency_when_both_holders_are_over_the_cap(
        self,
    ):
        # Hand-traced. Two slots. A is used 5 times and B 7, then only A is touched for
        # the last four steps, so A is the more recent holder while B is the heavier one.
        # Step 12 brings C: uncapped LFU gives up A (5 < 7) and B stays; with the count
        # clamped at 4 both holders rank 4, so the tie falls to recency and the OLDER
        # holder -- B -- is given up, which makes the next step's B a miss. Three misses
        # uncapped, four clamped, and the difference is the one arrival that proves the
        # clamp reaches the ranking key rather than the count itself.
        lines = [
            "0 0 0 1",
            "1 0 1",
            "2 0 1",
            "3 0 1",
            "4 0 1",
            "5 0 1",
            "6 0 1",
            "7 0 0",
            "8 0 0",
            "9 0 0",
            "10 0 0",
            "11 0 2",
            "12 0 1",
        ]
        trace = events(lines)
        uncapped = placements_per_token(trace, [2], start_position=0)
        clamped = placements_per_token(trace, [2], start_position=0, use_cap=4)
        self.assertAlmostEqual(uncapped, 3 / 13)
        self.assertAlmostEqual(clamped, 4 / 13)

    def test_no_cap_reproduces_the_engine_rule_exactly(self):
        # The default must stay the rule §5y calibrated: this replay and the one in
        # continuous_feeder_sim have to agree on a real capture, slot for slot.
        from continuous_feeder_sim import replay_misses

        path, start = CAPTURES["v520-20261015"]
        trace = load_trace(path)
        mine = placements_per_token(trace, [96] * 48, start_position=start)
        theirs = replay_misses(trace, layers=48, slots_per_layer=96, start_position=start)
        self.assertAlmostEqual(mine, theirs.misses_per_token, places=9)


class CrossCaptureTests(unittest.TestCase):
    """The claim §5z rests on: layer heat is a property of the model, not of the session.

    The floor is a majority, not a fitted threshold. An earlier draft asserted `keeps
    >= 0.8`, taken from a table that divided a capped transfer by the UNCAPPED uniform
    and so counted the cap's own gain as the split's. Measured like-for-like (every arm
    against the shipped arm at its own cap) the six transfers at cap 3 -- the arm this
    test scores -- span 74.4% to 126.5%, and 11 of the 18 across all three caps keep MORE
    than an oracle that read the future. The worst case is `v520-20261015 ->
    v520-prompt2`: a donor profile from another prompt, whose routes agree with the
    target's on 3.2% of picks. 74.4% is the worst of three captures, and three captures do
    not license an 80% promise, so the assertion states the claim that survives the worst
    measurement: a foreign profile delivers most of what an oracle would. The magnitudes
    themselves are §5z's table, not a test's -- and all of them are well under the plan's
    own acceptance bar of +7 points of hit rate, which is a §5z finding, not something a
    green suite means.
    """

    def test_a_split_never_spends_more_slots_than_the_tier_buys(self):
        # The whole point of screening at a fixed total: a split moves slots from cold
        # layers to hot ones, it cannot quietly ask for a bigger cache.
        for name, (path, start) in CAPTURES.items():
            budget = split_slots(
                heat_profile(load_trace(path), layers=48, start_position=start),
                total_slots=4_608,
            )
            self.assertEqual(sum(budget), 4_608, name)
            self.assertEqual(len(budget), 48, name)

    def test_a_profile_from_another_capture_keeps_most_of_the_oracle_and_beats_uniform(self):
        for target, (target_path, start) in CAPTURES.items():
            trace = load_trace(target_path)
            own = heat_profile(trace, layers=48, start_position=start)
            uniform = placements_per_token(trace, [96] * 48, start_position=start, use_cap=3)
            oracle = placements_per_token(
                trace, split_slots(own, total_slots=4_608), start_position=start, use_cap=3
            )
            self.assertLess(oracle, uniform, f"{target}: the oracle split must beat uniform")
            for source, (source_path, source_start) in CAPTURES.items():
                if source == target:
                    continue
                foreign = heat_profile(
                    load_trace(source_path), layers=48, start_position=source_start
                )
                transferred = placements_per_token(
                    trace,
                    split_slots(foreign, total_slots=4_608),
                    start_position=start,
                    use_cap=3,
                )
                self.assertLess(transferred, uniform, f"{source} -> {target} is no gain")
                kept = (uniform - transferred) / (uniform - oracle)
                self.assertGreaterEqual(
                    kept,
                    0.5,
                    f"{source} -> {target} keeps only {kept:.1%} of the oracle's gain",
                )


class TwoPassReplayTests(unittest.TestCase):
    """§5ad's screen rule, as code: a policy that changes what the cache *remembers* must
    be replayed twice over one cache set, because one pass is request 1.

    The reason this exists is a specific mistake. The capped use-count (#177) looked like
    a −2.0…−7.9 % win offline on a single pass and measured −4.7…−6.0 % on a live paired
    A/B -- it was a loss the whole time. A saturating count only pays when it meets an
    expert whose unsaturated rival would have defended a slot; that is a *stale* holder,
    and a stale holder can only exist in a cache that already ran. So the gain lives
    entirely in the difference between a cold cache and a warm one, which is exactly the
    difference a single pass cannot see. The second pass is what the operator's session
    actually runs in.
    """

    def test_the_first_pass_is_the_single_pass_replay_over_the_same_events(self):
        # Pins the warm-up semantics: a prefill row is planned but not counted, and pass 1
        # starts on the same cold cache `placements_per_token` starts on. If this drifted,
        # every pass-1 number quoted from a two-pass run would be a different quantity
        # from the single-pass ones §5y/5z/5ad tabulated. Hand-traced to 1.5: the prefill
        # row fills both slots, then {2,3} takes one miss and {1,4} both.
        trace = events(["0 0 1 2", "1 0 2 3", "2 0 1 4"])
        first, _ = two_pass_placements(trace, budgets=[2], start_position=1)
        self.assertAlmostEqual(first, placements_per_token(trace, [2], start_position=1))
        self.assertAlmostEqual(first, 1.5)

    def test_the_second_pass_asks_for_fewer_placements_because_the_cache_remembers(self):
        # A 3-expert working set over a 3-slot layer, requested in a rotating order. Pass 1
        # pays for each expert exactly once -- {1,2} misses both, {2,3} misses 3 into the
        # still-free slot 2 (an unfilled slot ranks weakest, which is where "fill the free
        # slots first" comes from), {3,1} is a clean hit -- so 3 placements over 3 tokens.
        # Pass 2 replays the same rows over the cache pass 1 left, and asks for nothing.
        # Same rows, same budget, same rule; the only difference is that it already ran.
        trace = events(["1 0 1 2", "2 0 2 3", "3 0 3 1"])
        first, second = two_pass_placements(trace, budgets=[3])
        self.assertAlmostEqual(first, 1.0)
        self.assertAlmostEqual(second, 0.0)

    def test_a_capped_use_count_wins_the_first_pass_and_loses_the_second(self):
        # The §5ad result, asserted on the committed captures rather than a fixture built
        # to demonstrate it. On all three captures the cap ranks better on pass 1 and
        # worse on pass 2, which is the signature of a policy whose only advantage is
        # paying less for a stale holder -- and the reason #177 was refused.
        for name, (path, start) in CAPTURES.items():
            trace = load_trace(path)
            uncapped = two_pass_placements(trace, [96] * 48, start_position=start)
            capped = two_pass_placements(trace, [96] * 48, start_position=start, use_cap=3)
            self.assertLess(capped[0], uncapped[0], f"{name}: no first-pass cap win")
            self.assertGreater(capped[1], uncapped[1], f"{name}: cap holds up on pass 2")


class EvidenceTests(unittest.TestCase):
    """The two numbers §5z's whole argument rests on are pinned here, not in prose."""

    def test_demand_is_counted_from_the_trace_so_a_hit_rate_has_a_real_denominator(self):
        # Two decode tokens over two layers: 2 + 1 then 3 + 1 = 7 requests, so 3.5 each.
        trace = events(["0 0 1 2", "0 1 3", "1 0 4 5 6", "1 1 7"])
        self.assertEqual(demand_per_token(trace, start_position=0), 3.5)

    def test_demand_ignores_the_prefill_rows(self):
        trace = events(["0 0 1 2 3 4", "1 0 5"])
        self.assertEqual(demand_per_token(trace, start_position=1), 1.0)

    def test_a_repeated_position_and_layer_is_refused(self):
        # The overlap below assumes one row per (position, layer); a capture that breaks
        # that would otherwise silently lose rows and report an overlap it never measured.
        with self.assertRaises(ValueError):
            route_rows(events(["0 0 1 2", "0 0 3 4"]), start_position=0)

    def test_two_captures_of_one_prompt_share_their_early_routes(self):
        rows_a = route_rows(events(["0 0 1 2", "1 0 3 4"]), start_position=0)
        rows_b = route_rows(events(["0 0 1 2", "1 0 5 6"]), start_position=0)
        positions, exact, picks = route_agreement(rows_a, rows_b)
        self.assertEqual((positions, exact), (2, 1))
        self.assertAlmostEqual(picks, 0.5)

    def test_the_chance_level_of_a_pick_overlap_is_this_width_over_this_layer_size(self):
        # Two independent top-k draws out of 512 share k/E of their picks. Pinned because
        # §5z calls the measured 3.2-13.2% "close to" and "7x" chance -- both are
        # statements about this denominator.
        rows = route_rows(events(["0 0 1 2", "1 0 3 4"]), start_position=0)
        self.assertAlmostEqual(chance_pick_overlap(rows), 2 / EXPERTS_PER_LAYER)

    def test_no_two_captures_route_the_same_way_more_than_a_fifth_of_the_time(self):
        # The falsifier for "the transferred profile just memorised the target". If a
        # donor agreed with its target on most picks, the transfer would prove nothing
        # about portability. Measured, over ~18 000 coinciding rows per pair: not ONE row
        # is identical, and the picks agree 3.2-13.2% against a 1.953% chance level.
        chance = chance_pick_overlap(
            route_rows(load_trace(CAPTURES["v520-20261015"][0]), CAPTURES["v520-20261015"][1])
        )
        names = list(CAPTURES)
        for index, first in enumerate(names):
            for second in names[index + 1 :]:
                path_a, start_a = CAPTURES[first]
                path_b, start_b = CAPTURES[second]
                positions, exact, picks = route_agreement(
                    route_rows(load_trace(path_a), start_a),
                    route_rows(load_trace(path_b), start_b),
                )
                self.assertGreater(positions, 1_000, f"{first} vs {second} barely coincides")
                self.assertEqual(exact, 0, f"{first} vs {second} has identical rows")
                self.assertLess(picks, 0.20, f"{first} vs {second} agrees on {picks:.1%}")
                self.assertGreater(
                    picks, chance, f"{first} vs {second} is indistinguishable from noise"
                )

    def test_the_heat_shapes_agree_even_though_the_routes_do_not(self):
        # The other half of the same argument: which LAYER is busy is stable across
        # sessions even though WHICH expert is not -- that is what a shipped table sells.
        profiles = {
            name: heat_profile(load_trace(path), layers=48, start_position=start)
            for name, (path, start) in CAPTURES.items()
        }
        names = list(profiles)
        for index, first in enumerate(names):
            for second in names[index + 1 :]:
                self.assertGreater(
                    profile_correlation(profiles[first], profiles[second]),
                    0.9,
                    f"{first} vs {second} shapes do not agree",
                )

    def test_a_qwen38_decode_token_routes_four_hundred_and_eighty_experts(self):
        # 48 layers x top-k 10. §5z converts placements/token to a hit rate with this
        # denominator, so the conversion is pinned rather than assumed.
        for name, (path, start) in CAPTURES.items():
            demand = demand_per_token(load_trace(path), start)
            self.assertEqual(demand, 480.0, name)


if __name__ == "__main__":
    unittest.main()
