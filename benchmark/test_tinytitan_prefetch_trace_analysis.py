import importlib.util
import pathlib
import sys
import unittest


MODULE_PATH = pathlib.Path(__file__).with_name("tinytitan_prefetch_trace_analysis.py")
SPEC = importlib.util.spec_from_file_location("prefetch_analysis", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH} as a module")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PrefetchTraceAnalysisTests(unittest.TestCase):
    def test_online_transition_prior_reports_nonresident_miss_recall(self) -> None:
        observations = [
            MODULE.Observation(0, 0, (1, 2), frozenset({1, 2}), frozenset(), (7, 8)),
            MODULE.Observation(0, 1, (7, 8), frozenset({7, 8}), frozenset()),
            MODULE.Observation(1, 0, (1, 2), frozenset(), frozenset({1, 2}), (7, 9)),
            MODULE.Observation(1, 1, (7, 9), frozenset({7, 9}), frozenset()),
        ]

        report = MODULE.analyze(observations, top_m=2, predictor="recorded")
        overall = report["overall"]
        self.assertEqual(overall["links"], 2)
        self.assertEqual(overall["actual_misses"], 4)
        self.assertEqual(overall["useful_prefetches"], 4)
        self.assertEqual(overall["actual_miss_recall"], 1.0)
        self.assertEqual(overall["prefetch_precision"], 1.0)

    def test_analyze_refuses_an_unknown_predictor_name(self) -> None:
        observations = [
            MODULE.Observation(0, 0, (1,), frozenset({1}), frozenset(), (7,)),
            MODULE.Observation(0, 1, (7,), frozenset({7}), frozenset()),
        ]

        with self.assertRaisesRegex(ValueError, r"unknown predictor 'frequency'"):
            MODULE.analyze(observations, top_m=2, predictor="frequency")

    def test_recent_predictor_credits_a_past_selection_the_probe_misses(self) -> None:
        observations = [
            MODULE.Observation(0, 0, (1, 2), frozenset({1, 2}), frozenset(), (7, 8)),
            MODULE.Observation(0, 1, (9, 10), frozenset({9, 10}), frozenset()),
            MODULE.Observation(1, 0, (3, 4), frozenset({3, 4}), frozenset(), (7, 8)),
            MODULE.Observation(1, 1, (9, 11), frozenset({9}), frozenset({11})),
        ]

        recorded = MODULE.analyze(observations, top_m=2, predictor="recorded")["overall"]
        recent = MODULE.analyze(observations, top_m=2, predictor="recent")["overall"]
        self.assertEqual(recorded["useful_prefetches"], 0)
        self.assertEqual(recent["useful_prefetches"], 1)

    def test_recent_predictor_never_reads_the_target_position(self) -> None:
        # The one row a predictor may not consult is the row it is planning: an
        # expert credited from it would score a hit the engine cannot know.
        observations = [
            MODULE.Observation(0, 0, (1,), frozenset({1}), frozenset()),
            MODULE.Observation(0, 1, (5,), frozenset({5}), frozenset()),
        ]

        overall = MODULE.analyze(observations, top_m=2, predictor="recent")["overall"]
        self.assertEqual(overall["predicted_nonresident"], 0)
        self.assertEqual(overall["useful_prefetches"], 0)

    def test_recent_predictor_window_excludes_selections_older_than_the_bound(self) -> None:
        observations = []
        for position, experts in ((0, (5,)), (1, (7,)), (2, (8,)), (3, (7,))):
            observations.append(
                MODULE.Observation(position, 0, (position,), frozenset(), frozenset())
            )
            observations.append(
                MODULE.Observation(position, 1, experts, frozenset(experts), frozenset())
            )

        narrow = MODULE.analyze(observations, top_m=2, predictor="recent", window=1)["overall"]
        wide = MODULE.analyze(observations, top_m=2, predictor="recent", window=2)["overall"]
        self.assertEqual(narrow["useful_prefetches"], 0)
        self.assertEqual(wide["useful_prefetches"], 1)

    def test_screen_stages_the_first_non_resident_pick_as_the_engine_does(self) -> None:
        observations = [
            MODULE.Observation(0, 0, (1,), frozenset({1}), frozenset(), (7, 8)),
            MODULE.Observation(1, 0, (1,), frozenset(), frozenset({1}), (7, 8)),
            MODULE.Observation(1, 1, (8,), frozenset({8}), frozenset({7})),
        ]

        shipped = MODULE.analyze(observations, top_m=1, predictor="recorded")["overall"]
        withdrawn = MODULE.analyze(observations, top_m=1, predictor="recorded", policy="truncate")[
            "overall"
        ]
        self.assertEqual(shipped["predicted_nonresident"], 1)
        self.assertEqual(shipped["useful_prefetches"], 1)
        self.assertEqual(withdrawn["predicted_nonresident"], 0)
        self.assertEqual(withdrawn["useful_prefetches"], 0)

    def test_slot_level_precision_separates_the_marginal_read_from_the_first(self) -> None:
        # §5bb priced the device prize per converted fixup event, so a ring's depth
        # is decidable per issued slot: one prediction that hits and two that do
        # not must not pool into a single 33% that judges nothing.
        observations = [
            MODULE.Observation(0, 0, (1,), frozenset({1}), frozenset(), (7, 8, 9)),
            MODULE.Observation(0, 1, (7,), frozenset({7}), frozenset()),
        ]

        report = MODULE.analyze(observations, top_m=3, predictor="recorded")
        slots = report["slots"]
        self.assertEqual([slot["slot"] for slot in slots], [0, 1, 2])
        self.assertEqual([slot["predicted"] for slot in slots], [1, 1, 1])
        self.assertEqual([slot["useful"] for slot in slots], [1, 0, 0])
        self.assertEqual([slot["precision"] for slot in slots], [1.0, 0.0, 0.0])

    def test_break_even_bar_is_the_byte_floor_over_the_measured_event_value(self) -> None:
        economics = MODULE.economics()
        self.assertAlmostEqual(economics["ms_per_issued_read"], 0.7407, places=4)
        self.assertAlmostEqual(economics["ms_per_converted_event"], 2.17, places=4)
        self.assertAlmostEqual(
            economics["break_even_precision"],
            economics["ms_per_issued_read"] / economics["ms_per_converted_event"],
            places=6,
        )

    def test_a_slot_below_the_bar_is_named_as_the_stop_rule_for_the_ring_depth(self) -> None:
        hit_then_miss = [
            MODULE.Observation(0, 0, (1,), frozenset({1}), frozenset(), (7, 8, 9)),
            MODULE.Observation(0, 1, (7, 8), frozenset({7, 8}), frozenset()),
        ]

        report = MODULE.analyze(hit_then_miss, top_m=3, predictor="recorded")
        slots = report["slots"]
        self.assertEqual([slot["useful"] for slot in slots], [1, 1, 0])
        self.assertEqual([slot["clears_break_even"] for slot in slots], [True, True, False])
        self.assertEqual(report["first_loss_making_slot"], 2)
        self.assertAlmostEqual(slots[0]["net_ms_per_link"], 2.17 - 0.74066, places=4)


if __name__ == "__main__":
    unittest.main()
