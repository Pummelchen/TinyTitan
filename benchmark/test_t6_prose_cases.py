"""The T6 prose case builder's labels and the gate scorer.

The measurement is only as good as its labels, and every label comes from a
detector that a hand review already corrected. The traps that cost a wrong
label are pinned here, with no model and no server, so a later edit cannot
quietly reintroduce one.

    cd benchmark && python3 -m unittest test_t6_prose_cases -v
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cases = _load("t6_prose_cases", "benchmark/t6_prose_cases.py")
score = _load("t6_prose_score", "benchmark/t6_prose_score.py")


class StripQuizTests(unittest.TestCase):
    def test_the_quiz_block_is_removed_and_the_prose_kept(self):
        reply = '```json\n{"town": "Ashgrove"}\n```\n\n# Chapter 1\nMarcus walked.'
        prose = cases.strip_quiz(reply)
        self.assertNotIn("Ashgrove", prose)
        self.assertIn("Marcus walked.", prose)

    def test_no_quiz_block_leaves_the_reply_alone(self):
        self.assertEqual(
            cases.strip_quiz("# Chapter 1\nMarcus walked."), "# Chapter 1\nMarcus walked."
        )


class EyeClaimTests(unittest.TestCase):
    def test_a_name_carries_its_own_colour(self):
        self.assertIn(
            "grey", [value for value, _ in cases.eye_claims("Marcus's grey eyes.", "marcus")]
        )

    def test_a_possessive_belongs_to_the_nearest_name(self):
        line = "Dr Halvorsen's grey eyes held the room."
        self.assertEqual([], cases.eye_claims(line, "marcus"))
        self.assertIn("grey", [value for value, _ in cases.eye_claims(line, "halvorsen")])

    def test_her_colour_is_not_his(self):
        line = "Marcus watched Ines, her green eyes bright."
        self.assertNotIn("green", [value for value, _ in cases.eye_claims(line, "marcus")])


class RainClaimTests(unittest.TestCase):
    def test_never_rained_is_not_a_claim(self):
        self.assertEqual(
            [], cases.rain_claims("The sun beat down, for it never rained in this town.")
        )

    def test_a_memory_of_rain_is_not_a_claim(self):
        self.assertEqual(
            [], cases.rain_claims("A place where the memory of rain was a faded legend.")
        )

    def test_living_with_the_rain_is_a_claim(self):
        self.assertEqual(
            ["rains"],
            [value for value, _ in cases.rain_claims("They had learned to live with the rain.")],
        )


class FerryDayTests(unittest.TestCase):
    def test_a_plural_weekday_after_the_ferry_counts(self):
        self.assertEqual(
            ["sunday"],
            [v for v, _ in cases.ferry_day_claims("The ferry runs only on Sundays.")],
        )

    def test_a_weekday_named_as_the_ferrys_day_counts(self):
        self.assertEqual(
            ["thursday"],
            [
                v
                for v, _ in cases.ferry_day_claims(
                    "The fog would only thicken on Thursdays, the day the ferry was supposed to run."
                )
            ],
        )

    def test_the_weathers_weekday_is_not_the_ferrys(self):
        self.assertEqual(
            [],
            cases.ferry_day_claims(
                "The fog rolled in thick on Thursday, and the ferry stopped running for good."
            ),
        )


class ConfessionTests(unittest.TestCase):
    def test_a_verb_is_a_claim(self):
        self.assertEqual(
            ["__yes__"], [v for v, _ in cases.confessed_claims("Halvorsen confessed the forgery.")]
        )

    def test_a_rumour_is_not(self):
        self.assertEqual(
            [],
            cases.confessed_claims("The rumor of his confession had made the silence worse."),
        )

    def test_an_unspoken_confession_is_not(self):
        self.assertEqual(
            [],
            cases.confessed_claims("Halvorsen's confession remained unspoken beneath the silence."),
        )


class StateClaimTests(unittest.TestCase):
    def test_speculation_about_being_alive_is_not_found(self):
        # The reply does assert "still missing" -- that is a claim, and the
        # detector is right to report it. What must not appear is "found":
        # "who knew if he was still alive?" is speculation, not a state change.
        claims = cases.state_detector(
            "tomas", "found alive|was found|has been found", "missing|vanished", "found", "missing"
        )
        values = [
            value for value, _ in claims("Tomas was still missing; who knew if he was still alive?")
        ]
        self.assertEqual(values, ["missing"])

    def test_an_inn_that_stood_firm_is_standing(self):
        claims = cases.state_detector(
            "inn",
            "burn(?:ed|t|ing)?|ashes",
            "still standing|stood firm|intact",
            "burned",
            "standing",
        )
        self.assertEqual(
            ["standing"], [v for v, _ in claims("The inn stood firm against the wind.")]
        )


class KnowsPhotoTests(unittest.TestCase):
    def test_knowing_what_it_shows_is_a_claim(self):
        self.assertEqual(
            ["true"],
            [v for v, _ in cases.knows_photo_claims("Marcus knew what the photograph showed.")],
        )

    def test_the_rule_is_not_a_claim(self):
        self.assertEqual(
            [],
            cases.knows_photo_claims(
                "Marcus must not learn what the photograph shows before chapter 60."
            ),
        )

    def test_unaware_is_the_negative_claim(self):
        self.assertEqual(
            ["false"],
            [
                v
                for v, _ in cases.knows_photo_claims(
                    "Marcus, still unaware of the photograph's true meaning, walked on."
                )
            ],
        )


class AnswerParsingTests(unittest.TestCase):
    def test_one_word_answers(self):
        self.assertEqual("YES", score.answer_of("YES"))
        self.assertEqual("NO", score.answer_of(" no."))
        self.assertEqual("YES", score.answer_of("Yes 90"))

    def test_an_error_is_not_an_answer(self):
        self.assertTrue(score.answer_of("<error timed out>").startswith("<"))


class GateMetricTests(unittest.TestCase):
    def test_precision_recall_and_silence(self):
        with tempfile.TemporaryDirectory() as tmp:
            review = pathlib.Path(tmp) / "review.tsv"
            review.write_text(
                "run\tsession\tkey\tkind\ttruth\tknown\tclaims\tspan\n"
                "1\t1\tk/yes\tclaim\tYES\t-\t\t\n"
                "1\t2\tk/missed\tclaim\tYES\t-\t\t\n"
                "1\t3\tk/silent\tsilent\tNO\t-\t\t\n"
                "1\t4\tk/wrong\tclaim\tNO\t-\t\t\n"
            )
            done = pathlib.Path(tmp) / "done.jsonl"
            done.write_text(
                "\n".join(
                    json.dumps({"note": note, "completion": answer})
                    for note, answer in [
                        ("r1s1 k/yes", "YES"),
                        ("r1s2 k/missed", "NO"),
                        ("r1s3 k/silent", "YES"),
                        ("r1s4 k/wrong", "NO"),
                    ]
                )
                + "\n"
            )
            result = score.report("t", done, score.load_labels(review))
        self.assertEqual(result["total"], 4)
        self.assertAlmostEqual(result["precision"], 0.5)  # fired twice, one right
        self.assertAlmostEqual(result["recall"], 0.5)  # one of two positives
        self.assertAlmostEqual(result["silence_rate"], 1.0)  # the one silent case fired


if __name__ == "__main__":
    unittest.main()
