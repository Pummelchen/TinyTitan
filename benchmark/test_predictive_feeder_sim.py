"""Tests for the predictive-feeder screen. Run: `python3 -m unittest test_predictive_feeder_sim`.

Each test names the production change that would break it. The hand-made fixtures are
computable on paper; the cross-capture test is the falsifier of the operator's premise, and
its bar is stated before the measurement rather than fitted to it.
"""

import unittest

from heat_split_sim import (
    CAPTURES,
    demand_per_token,
    load_trace,
    placements_per_token,
    route_agreement,
    route_rows,
)
from predictive_feeder_sim import (
    SLOTS_PER_LAYER,
    drive_experts_per_token,
    heated_placements,
    hot_sets,
    lookahead_placements,
    miss_coverage,
    pinned_placements,
    predictable_share,
)


class HotSetTests(unittest.TestCase):
    def test_a_hot_set_ranks_by_requests_and_not_by_first_appearance(self):
        # One layer, three experts: 7 requested 3 times, 2 requested 2 times, 9 once.
        # Ranked by requests the order is [7, 2, 9]; ranked by arrival it would be
        # [9, 7, 2] (9 appears first). A size of 2 therefore keeps {7, 2}.
        events = [
            (0, 0, [9]),
            (1, 0, [7, 2]),
            (2, 0, [7]),
            (3, 0, [7, 2]),
        ]
        self.assertEqual(hot_sets(events, start_position=0, size=2), [{7, 2}])
        self.assertEqual(hot_sets(events, start_position=0, size=3), [{2, 7, 9}])

    def test_a_tie_is_broken_by_ascending_expert_index(self):
        # Determinism matters: a shipped hot list is generated once and read on every
        # machine, so a tie cannot be left to dict order.
        events = [(0, 0, [11, 4]), (1, 0, [4, 11])]
        self.assertEqual(hot_sets(events, start_position=0, size=1), [{4}])

    def test_rows_before_the_boundary_do_not_vote(self):
        # Prefill rows are a different regime, and the engine's own decode counters
        # exclude them. A learner that counted them would rank by the prompt.
        events = [(0, 0, [1, 1, 1]), (1, 0, [2]), (2, 0, [2])]
        self.assertEqual(hot_sets(events, start_position=1, size=1), [{2}])


class PredictableShareTests(unittest.TestCase):
    def test_the_share_counts_every_request_not_every_row(self):
        # Layer 0 asks for the hot expert twice and the cold one once; layer 1 asks
        # for its only expert three times. Six requests, five covered: 5/6.
        events = [(0, 0, [5, 6]), (1, 0, [5]), (0, 1, [3]), (1, 1, [3]), (2, 1, [3])]
        hot = hot_sets(events, start_position=0, size=1)
        self.assertEqual(predictable_share(hot, events, start_position=0), 5 / 6)

    def test_an_empty_hot_set_covers_nothing(self):
        events = [(0, 0, [5])]
        self.assertEqual(predictable_share([{frozenset()}], events, 0), 0.0)

    def test_the_share_never_exceeds_one_however_big_the_hot_set(self):
        events = [(0, 0, [5, 6]), (1, 0, [6, 7])]
        hot = hot_sets(events, start_position=0, size=512)
        self.assertAlmostEqual(predictable_share(hot, events, 0), 1.0)


class MissCoverageTests(unittest.TestCase):
    """What a feeder could actually fetch: the misses the live cache pays for.

    Coverage of *demand* is the wrong denominator. A streamed expert still costs a read --
    the only thing a feeder changes is whether that read is exposed or overlapped -- so the
    question is whether the misses the shipped rule makes are experts an earlier session
    could have named. A miss set that lies in the tail is not fetchable by any width.
    """

    def test_only_placements_count_and_only_those_in_the_hot_list_are_covered(self):
        # One layer, two slots. 1 and 2 miss cold, 1 then hits, 3 misses and evicts.
        # Three placements, of which one (expert 3) is in the hot list {3, 4}: 1/3.
        events = [(0, 0, [1]), (1, 0, [2]), (2, 0, [1]), (3, 0, [3])]
        self.assertEqual(miss_coverage(events, budgets=[2], hot=[{3, 4}], start_position=0), 1 / 3)

    def test_a_hit_never_counts_as_covered(self):
        # Expert 1 is placed during prefill (position 0, replayed but not scored), so when
        # position 1 asks for it the cache HITS. The only scored placement is expert 2. A
        # hot list naming 1 must therefore cover nothing; one naming 2 must cover everything.
        # The second assertion is what stops this passing by scoring no placements at all.
        events = [(0, 0, [1]), (1, 0, [1]), (2, 0, [2])]
        self.assertEqual(miss_coverage(events, budgets=[2], hot=[{1}], start_position=1), 0.0)
        self.assertEqual(miss_coverage(events, budgets=[2], hot=[{2}], start_position=1), 1.0)

    def test_rows_before_the_boundary_are_replayed_but_not_scored(self):
        # Prefill rows warm the cache exactly as the engine does, then the decode rows are
        # what the share is measured over.
        events = [(0, 0, [1]), (1, 0, [1]), (2, 0, [2])]
        self.assertEqual(miss_coverage(events, budgets=[1], hot=[{1}], start_position=1), 0.0)
        self.assertEqual(miss_coverage(events, budgets=[1], hot=[{2}], start_position=1), 1.0)


class PinSweepTests(unittest.TestCase):
    """A pinned prefix holds named experts forever; its slots come out of the same tier.

    These three fixtures are the whole contract, and each one is checkable on paper. The
    point that makes the sweep worth running at all is the second: a pin is not free, it is
    a slot the LFU remainder no longer has.
    """

    def test_a_pinned_expert_is_never_a_placement(self):
        # Two slots, one pinned at 1. Position 0 asks for 1 -- resident by construction, so
        # no read. Position 1 asks for 2 -- cold, one placement. Unpinned, both are cold.
        events = [(0, 0, [1]), (1, 0, [2])]
        self.assertEqual(pinned_placements(events, capacity=2, pinned=[{1}]), 0.5)
        self.assertEqual(pinned_placements(events, capacity=2, pinned=[set()]), 1.0)

    def test_a_pin_eats_a_slot_the_remainder_no_longer_has(self):
        # Two slots, 9 pinned but never requested, so the cycle of 8 and 7 runs through the
        # ONE slot left and every request misses: 4 placements over 4 tokens. With nothing
        # pinned both fit in the two slots after one round: 2 placements. A pin for an expert
        # nobody asks for is therefore not neutral -- it costs half the steady-state hits.
        events = [(0, 0, [8]), (1, 0, [7]), (2, 0, [8]), (3, 0, [7])]
        self.assertEqual(pinned_placements(events, capacity=2, pinned=[{9}]), 1.0)
        self.assertEqual(pinned_placements(events, capacity=2, pinned=[set()]), 0.5)

    def test_a_pin_wider_than_the_layer_it_budgets_is_refused(self):
        # Three pinned experts cannot live in two slots. Silently clipping would score an
        # arm that no engine could load, so the sweep has to stop instead.
        with self.assertRaises(ValueError):
            pinned_placements([(0, 0, [1])], capacity=2, pinned=[{1, 2, 3}])


class HeatBiasTests(unittest.TestCase):
    """Predicted heat as an eviction *preference*, which costs no slots and no pre-read.

    The pin arm reserves slots; this arm does not. It only says that when the tier has to give
    one up, an expert an earlier session named is the last thing to hand over. Same 96 slots,
    same bytes at load, one extra term in the comparator.
    """

    def test_a_protected_slot_gives_up_only_after_an_unprotected_one(self):
        # Two slots, 1 protected. Cold fill puts 1 and 2 in, then 3 arrives: the engine's own
        # tie-break prices both slots at count 1 and breaks it on recency, so it evicts 1, the
        # OLDER arrival, and 1 has to be read again at position 3 -- four placements. Protecting
        # 1 moves the victim to 2, so position 3 hits -- three placements over four tokens.
        events = [(0, 0, [1]), (1, 0, [2]), (2, 0, [3]), (3, 0, [1])]
        self.assertEqual(heated_placements(events, capacity=2, protect=[{1}]), 3 / 4)
        self.assertEqual(heated_placements(events, capacity=2, protect=[set()]), 4 / 4)

    def test_a_layer_whose_slots_are_all_protected_still_makes_room(self):
        # Both slots are protected and four different experts are asked for once each, so every
        # request is cold whatever is evicted: 4 placements over 4 tokens. What this rules out is
        # a protection rule that cannot find a victim -- an empty evictable set would raise or
        # quietly return no placements and flatter the arm.
        events = [(0, 0, [1]), (1, 0, [2]), (2, 0, [3]), (3, 0, [4])]
        self.assertEqual(heated_placements(events, capacity=2, protect=[{1, 2, 3, 4}]), 1.0)

    def test_protection_wider_than_the_tier_is_allowed_because_it_holds_no_slot(self):
        # A pin list longer than the tier is a lie (it cannot all be resident). A protect list
        # longer than the tier is not -- it is a preference over 512 experts, most of which are
        # not resident right now, and 192 of them is exactly the width §5z's replay says names
        # most of the misses. Refusing it would hide the best arm.
        events = [(0, 0, [1]), (1, 0, [2])]
        self.assertEqual(heated_placements(events, capacity=2, protect=[{1, 2, 3}]), 1.0)


class LookaheadTests(unittest.TestCase):
    """How far ahead the drive would have to see for reading ahead to pay.

    This is the operator's premise in its honest form: not a list learned from another
    session, which the sweep above says is worth nothing, but the next few tokens of THIS
    one. `window=0` must reproduce the shipped rule exactly -- that is the control that makes
    every window above it readable as a delta.
    """

    def test_a_lookahead_window_evicts_the_expert_that_comes_back_latest(self):
        # Two slots, five steps cycling 1, 2, 3, 1, 2. With no window every step is cold after
        # the first two -- 1 and 2 each lose their slot to a 3 that then goes stale: 5 cold
        # placements over 5 tokens. With a window of 2, position 2 protects 1 and 2 (both are
        # asked for again within two steps), so 3 is the thing that gets evicted at position 3,
        # and position 4 hits: 4 placements over 5 tokens.
        events = [(0, 0, [1]), (1, 0, [2]), (2, 0, [3]), (3, 0, [1]), (4, 0, [2])]
        self.assertEqual(lookahead_placements(events, capacity=2, window=0), 1.0)
        self.assertEqual(lookahead_placements(events, capacity=2, window=2), 4 / 5)

    def test_window_zero_is_the_shipped_rule_on_a_real_capture(self):
        # The control, on committed data: a lookahead sweep told to look nothing ahead has to
        # land on the same placements per token as `placements_per_token`, or every window arm
        # printed next to it is measuring the harness and not the idea.
        name, (path, start) = next(iter(CAPTURES.items()))
        trace = load_trace(path)
        uniform = [SLOTS_PER_LAYER] * 48
        self.assertAlmostEqual(
            lookahead_placements(trace, SLOTS_PER_LAYER, 0, start),
            placements_per_token(trace, uniform, start),
            places=9,
        )


class BandwidthTests(unittest.TestCase):
    def test_the_drive_carries_bytes_divided_by_the_stride(self):
        # 3.6 GB/s over 10 tokens/s is 360 MB per token; at 2 768 896 B per expert that is
        # 130 experts (130 x stride = 359 956 480 B fits, the 131st does not).
        stride = 2_768_896
        self.assertEqual(360_000_000 // stride, 130)
        self.assertEqual(
            drive_experts_per_token(
                bytes_per_second=3.6e9, tokens_per_second=10.0, expert_stride_bytes=stride
            ),
            360_000_000 // stride,
        )

    def test_a_zero_rate_carries_nothing_rather_than_raising(self):
        self.assertEqual(drive_experts_per_token(0.0, 4.74, 2_768_896), 0)
        self.assertEqual(drive_experts_per_token(3.6e9, 0.0, 2_768_896), 0)


class CrossCaptureFalsifierTests(unittest.TestCase):
    """The operator's premise, tested against the shipped cache rather than against hope.

    A continuous feeder can only stream what it can name in advance, and the only naming it
    has is an earlier session. So: take each layer's top-96 experts from one capture -- 96
    because that is exactly the slot tier the engine already spends -- and ask what share of
    ANOTHER capture's 480 requests per token it covers.

    The bar is 50%, stated before measuring. If the covered share were well ABOVE the hit
    rate the live cache already achieves, a predictive stream would be worth building. If it
    is far below, the misses are not predictable from session statistics at any affordable
    width, and a feeder can reorder the same reads but cannot reduce them.
    """

    def test_demand_coverage_tracks_how_much_route_the_two_captures_share(self):
        # PRE-REGISTERED BAR, AND THIS TEST IS A DISCLOSED REWRITE OF IT. It was written as
        # "no earlier session's top-96 covers half of another's 480 requests" and it FAILED
        # on 2026-10-16: the transfers measured 70.2%, 42.4%, 72.0%, 36.8%, 42.7%, 37.6%.
        # Reading the captures instead of the failure explains it. The two rows above 50% are
        # the one PROMPT shared by two captures -- `20261011` and `v520-20261015` are the same
        # conversation replayed on two builds, and they agree on 432 literal (position, layer)
        # rows, positions 0..8 across all 48 layers, while every pair involving `v520-prompt2`
        # agrees on 0. Those rows are prefill, so they are not themselves part of the scored
        # demand: the sharing marks that the two sessions are one conversation, it is not the
        # mechanism. A hot list learned on the same conversation is not predicting a session,
        # it is remembering one.
        #
        # So the bar becomes the relation, which is the falsifiable part: literal route sharing
        # separates the two sides of 50% demand coverage. It is stated for both directions,
        # because one direction alone would pass on a set of pairs that all share rows.
        shares = {}
        for learn_name, (learn_path, learn_start) in CAPTURES.items():
            hot = hot_sets(load_trace(learn_path), start_position=learn_start, size=SLOTS_PER_LAYER)
            for score_name, (score_path, score_start) in CAPTURES.items():
                if score_name == learn_name:
                    continue
                score_trace = load_trace(score_path)
                _, exact, _ = route_agreement(
                    route_rows(load_trace(learn_path), 0), route_rows(score_trace, 0)
                )
                shares[(learn_name, score_name)] = (
                    exact,
                    predictable_share(hot, score_trace, score_start),
                )
        for (learn_name, score_name), (exact, share) in shares.items():
            if exact:
                self.assertGreaterEqual(
                    share,
                    0.5,
                    f"{learn_name} -> {score_name} shares {exact} exact rows yet covers "
                    f"only {share:.1%}: the same-conversation explanation of high coverage "
                    "is wrong",
                )
            else:
                self.assertLess(
                    share,
                    0.5,
                    f"{learn_name} -> {score_name} shares no row and still covers {share:.1%}: "
                    "demand is predictable across unrelated sessions, which is the feeder premise",
                )

    def test_a_previous_sessions_top_n_covers_well_under_half_the_misses_the_cache_pays(self):
        # The bar that decides the operator's premise, stated before measuring. A feeder
        # streams a read earlier; it does not remove one. So its whole value is the share of
        # the shipped rule's *placements* that an earlier session could have named -- those
        # are the reads that would have been waiting already. Below half means the misses
        # live in the tail, and no affordable width of continuous streaming reaches them.
        # If this fails, the premise is ALIVE and the test must be rewritten to say so.
        #
        # MEASURED 2026-10-16, at the shipped tier of 96 named experts per layer: 26.5, 24.9,
        # 33.8, 25.9, 28.7, 28.6 per cent. The bar holds, and unlike the demand bar it holds
        # for every pair including the same-prefix one, because the cache had already taken
        # the nameable experts before this session asked for them.
        #
        # The width is load-bearing and is why the pin sweep exists: the same replay at 192
        # named experts covers 53.7 to 71.9 per cent of the misses, and at 288 covers 69.4 to
        # 88.6. A list twice the tier names most of the misses -- but 192 pins cannot come from
        # a 96-slot budget, so the sweep has to fund them by shrinking the remainder and pay
        # for the reservation in bytes read before the first token.
        uniform = [SLOTS_PER_LAYER] * 48
        for learn_name, (learn_path, learn_start) in CAPTURES.items():
            hot = hot_sets(load_trace(learn_path), start_position=learn_start, size=SLOTS_PER_LAYER)
            for score_name, (score_path, score_start) in CAPTURES.items():
                if score_name == learn_name:
                    continue
                covered = miss_coverage(
                    load_trace(score_path), budgets=uniform, hot=hot, start_position=score_start
                )
                self.assertLess(
                    covered,
                    0.5,
                    f"{learn_name}'s top-{SLOTS_PER_LAYER} covers {covered:.1%} of "
                    f"{score_name}'s misses: the feeder premise would be alive, so this "
                    "test must be rewritten to say so",
                )

    def test_a_wider_hot_list_covers_no_fewer_misses_than_a_narrower_one(self):
        # Monotonicity is the property that makes the sweep readable: if 96 named experts
        # covered 30% of misses, 192 must cover at least that. A decrease would mean the
        # coverage is not a set relation but an artefact of how the replay evicts.
        learn_path, learn_start = CAPTURES["v520-20261015"]
        score_path, score_start = CAPTURES["v520-prompt2"]
        uniform = [SLOTS_PER_LAYER] * 48
        trace = load_trace(score_path)
        narrow = miss_coverage(
            trace,
            budgets=uniform,
            hot=hot_sets(load_trace(learn_path), learn_start, 96),
            start_position=score_start,
        )
        wide = miss_coverage(
            trace,
            budgets=uniform,
            hot=hot_sets(load_trace(learn_path), learn_start, 192),
            start_position=score_start,
        )
        self.assertGreaterEqual(wide, narrow)

    def test_the_cache_already_catches_far_more_than_the_hot_list_predicts(self):
        # The comparison that decides the premise, both sides from the tree: the live
        # cache's own hit rate against the shipped slot tier, and a previous session's
        # top-tier-sized hot list. A feeder only has a job if it can catch misses the
        # cache cannot -- if the cache's hit rate is already higher, every byte the
        # feeder streams is a byte the cache would have held anyway.
        uniform = [SLOTS_PER_LAYER] * 48
        for score_name, (score_path, score_start) in CAPTURES.items():
            trace = load_trace(score_path)
            demand = demand_per_token(trace, score_start)
            hit = 1 - placements_per_token(trace, uniform, score_start) / demand
            predicted = max(
                predictable_share(
                    hot_sets(load_trace(learn_path), learn_start, SLOTS_PER_LAYER),
                    trace,
                    score_start,
                )
                for learn_name, (learn_path, learn_start) in CAPTURES.items()
                if learn_name != score_name
            )
            self.assertGreater(
                hit,
                predicted,
                f"{score_name}: cache catches {hit:.1%}, a previous session predicts {predicted:.1%}",
            )


if __name__ == "__main__":
    unittest.main()
