"""The master-prompt scenarios and their scoring.

An authoring bug in a scenario is invisible in a model run — it just looks like
a judge failure — so the timeline, the quiz and the two sets are checked here
with no model. `score_run` is pinned on a synthetic run, including the stale
count, because that is the number the suite is for.

    cd benchmark && python3 -m unittest test_memory_master -v
"""

from __future__ import annotations

import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scenarios = _load("master_scenarios", "benchmark/master_scenarios.py")
master = _load("memory_master", "benchmark/memory_master.py")


class ScenarioTests(unittest.TestCase):
    def test_every_scenario_is_consistent(self):
        self.assertEqual(scenarios.check(), 0)

    def test_all_ten_are_present_and_named(self):
        self.assertEqual(len(scenarios.SCENARIOS), 10)
        for name, spec in scenarios.SCENARIOS.items():
            self.assertTrue(spec["sessions"] >= 2, name)
            self.assertTrue(spec["carryable"], f"{name}: nothing carryable")
            self.assertTrue(spec["brief"], name)

    def test_foundation_and_carryable_partition_the_keys(self):
        for name, spec in scenarios.SCENARIOS.items():
            self.assertEqual(
                sorted(spec["foundation"] + spec["carryable"]), sorted(spec["keys"]), name
            )
            self.assertFalse(set(spec["foundation"]) & set(spec["carryable"]), name)

    def test_a_changed_key_really_changes_and_a_foundation_key_never_does(self):
        for name, spec in scenarios.SCENARIOS.items():
            if spec["self_chosen"]:
                continue
            for key in spec["carryable"]:
                values = {repr(spec["truth"](s)[key]) for s in range(1, spec["sessions"] + 1)}
                self.assertGreater(len(values), 1, f"{name}: {key} never changes")
            for key in spec["foundation"]:
                values = {repr(spec["truth"](s)[key]) for s in range(1, spec["sessions"] + 1)}
                self.assertEqual(len(values), 1, f"{name}: {key} changes")


class MatchingTests(unittest.TestCase):
    def test_numbers_match_numerically(self):
        self.assertTrue(scenarios.hit(60, "60 days"))
        self.assertTrue(scenarios.hit(4200, "4200"))
        self.assertFalse(scenarios.hit(4200, "4180"))

    def test_booleans_accept_words(self):
        self.assertTrue(scenarios.hit(True, "yes"))
        self.assertTrue(scenarios.hit(False, "No."))
        self.assertFalse(scenarios.hit(True, "false"))

    def test_strings_match_as_a_phrase(self):
        self.assertTrue(scenarios.hit("postgres", "Postgres 16"))
        self.assertTrue(scenarios.hit("point in time", "recognised point in time"))
        self.assertFalse(scenarios.hit("england", "singapore"))

    def test_an_empty_answer_is_not_a_hit(self):
        self.assertFalse(scenarios.hit("postgres", ""))
        self.assertFalse(scenarios.hit("postgres", "   "))

    def test_a_truncated_answer_is_not_a_hit(self):
        self.assertFalse(scenarios.hit("point in time", "point"))

    def test_an_empty_expected_value_is_not_a_hit(self):
        self.assertFalse(scenarios.hit("", "Postgres 16"))

    def test_an_absent_answer_is_not_a_hit(self):
        """A control: memory_master asks for a key the model never wrote, so the
        answer that reaches the scorer is None, not a string."""
        self.assertFalse(scenarios.hit("postgres", None))
        # Without the None guard the absent answer becomes the text str(None)
        # produces, so any expected value inside that word is scored as recalled.
        self.assertFalse(scenarios.hit("on", None))
        self.assertFalse(scenarios.hit("ne", None))


class ScoreTests(unittest.TestCase):
    def _photograph_run(self):
        spec = scenarios.SCENARIOS["photograph"]
        session_one = dict(spec["truth"](1))
        session_five = dict(spec["truth"](5))
        # The inn burned in session 4; this answer is the old value.
        stale = dict(session_five, inn_status="standing")
        base = {
            "prompt_tokens": 500,
            "completion_tokens": 100,
            "seconds": 10.0,
            "consolidation_wait": 0.0,
        }
        return [
            dict(base, session=1, answers=session_one),
            dict(base, session=5, answers=stale),
        ]

    def test_carryable_miss_and_stale_are_counted(self):
        master.SPEC = scenarios.SCENARIOS["photograph"]
        run = master.score_run(self._photograph_run())
        last = run["sessions"][-1]
        self.assertEqual(last["foundation"], [9, 9])  # nothing else regressed
        self.assertEqual(last["carryable"], [4, 5])  # inn_status is the miss
        self.assertEqual(last["stale"], 1)  # and it is the old value
        self.assertEqual(last["wrong"], ["inn_status"])

    def test_a_session_with_no_quiz_is_excluded_not_scored_as_misses(self):
        master.SPEC = scenarios.SCENARIOS["photograph"]
        spec = scenarios.SCENARIOS["photograph"]
        base = {
            "prompt_tokens": 500,
            "completion_tokens": 2500,
            "seconds": 10.0,
            "consolidation_wait": 0.0,
            "finish_reason": "length",
        }
        run = master.score_run(
            [
                dict(base, session=1, answers=dict(spec["truth"](1))),
                dict(base, session=5, answers={}),
            ]
        )
        last = run["sessions"][-1]
        self.assertTrue(last["invalid"])
        self.assertEqual(last["invalid_reason"], "truncated at the token ceiling")
        self.assertEqual(last["carryable"], [0, 0])  # excluded, not 0/5
        self.assertEqual(last["stale"], 0)
        self.assertEqual(last["wrong"], [])

    def test_a_completed_reply_without_a_quiz_says_so(self):
        master.SPEC = scenarios.SCENARIOS["photograph"]
        spec = scenarios.SCENARIOS["photograph"]
        base = {
            "prompt_tokens": 500,
            "completion_tokens": 400,
            "seconds": 10.0,
            "consolidation_wait": 0.0,
            "finish_reason": "stop",
        }
        run = master.score_run(
            [
                dict(base, session=1, answers=dict(spec["truth"](1))),
                dict(base, session=5, answers={}),
            ]
        )
        self.assertEqual(run["sessions"][-1]["invalid_reason"], "no quiz in a completed reply")

    def test_a_self_chosen_scenario_scores_against_session_one(self):
        master.SPEC = scenarios.SCENARIOS["pong"]
        rules = {
            "field_width": 800,
            "field_height": 600,
            "win_score": 11,
            "ball_start_speed": 5,
            "ball_speed_increment": 0.5,
            "ball_max_speed": 15,
            "paddle_speed": 8,
        }
        base = {
            "prompt_tokens": 2000,
            "completion_tokens": 900,
            "seconds": 30.0,
            "consolidation_wait": 0.0,
        }
        run = master.score_run(
            [
                dict(base, session=1, answers=dict(rules), self_truth=dict(rules)),
                dict(base, session=2, answers=dict(rules)),
                dict(base, session=3, answers=dict(rules, paddle_speed=6)),
            ]
        )
        self.assertEqual(run["sessions"][1]["carryable"], [3, 3])
        self.assertEqual(run["sessions"][2]["carryable"], [2, 3])
        self.assertEqual(run["sessions"][2]["stale"], 0)  # 6 was never the value


class QuizExtractionTests(unittest.TestCase):
    def test_the_full_key_block_wins_over_an_earlier_partial_one(self):
        keys = ["a", "b", "c"]
        text = (
            '```json\n{"a": 1}\n```\nwork mentions {"a": 1, "b": 2}\n'
            '```json\n{"a": 1, "b": 2, "c": 3}\n```'
        )
        self.assertEqual(scenarios.extract_quiz(text, keys), {"a": 1, "b": 2, "c": 3})

    def test_a_partial_block_is_still_a_fallback(self):
        self.assertEqual(scenarios.extract_quiz('```json\n{"a": 1}\n```', ["a", "b"]), {"a": 1})

    def test_the_quiz_is_asked_for_first(self):
        prompt = scenarios.quiz_prompt(scenarios.SCENARIOS["pigeon"])
        self.assertIn("Begin your reply", prompt)
        self.assertIn("first thing in your reply", prompt)


class ConsolidationWaitTests(unittest.TestCase):
    """The wait must end on a decision, not only on a distillation.

    Counting only `memory consolidated session=` made a skipped consolidation
    look like work in flight, and the wait then burned its whole limit on a
    session the engine had already finished — harness overhead recorded as
    memory cost.
    """

    def _with_log(self, contents: str):
        import tempfile

        directory = tempfile.TemporaryDirectory()
        log = pathlib.Path(directory.name) / "server.log"
        log.write_text(contents)
        saved = master.SERVER_LOG
        master.SERVER_LOG = str(log)
        self.addCleanup(setattr, master, "SERVER_LOG", saved)
        self.addCleanup(directory.cleanup)
        return log

    def test_a_store_write_summary_is_not_an_engine_decision(self):
        self._with_log("[t] memory memory session=s-1 consolidated 3 records\n")
        self.assertEqual(master.consolidation_outcomes(), (0, 0))

    def test_a_skip_ends_the_wait(self):
        log = self._with_log("")
        before = master.consolidation_outcomes()
        log.write_text(
            "[t] memory consolidation skipped session=s-1: 10 new characters, nothing to distil\n"
        )
        self.assertLess(master.wait_for_consolidation(before, limit=30), 5)
        self.assertEqual(master.consolidation_outcomes(), (0, 1))

    def test_a_distillation_ends_the_wait(self):
        log = self._with_log("")
        before = master.consolidation_outcomes()
        log.write_text("[t] memory consolidated session=s-2 turns=1 facts=1 seconds=9\n")
        self.assertLess(master.wait_for_consolidation(before, limit=30), 5)
        self.assertEqual(master.consolidation_outcomes(), (1, 0))


if __name__ == "__main__":
    unittest.main()
