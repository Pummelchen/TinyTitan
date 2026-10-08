"""The T6 prose case builder's labels and the gate scorer.

The measurement is only as good as its labels, and every label comes from a
detector that a hand review already corrected. The traps that cost a wrong
label are pinned here, with no model and no server, so a later edit cannot
quietly reintroduce one.

    cd benchmark && python3 -m unittest test_t6_prose_cases -v
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import pathlib
import sys
import tempfile
import unittest
import unittest.mock

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
    """What the gate scorer counts, and what it exits with.

    A row whose request never answered has no answer to score: `side_engine_judges`
    records the refusal in an `error` field (and, before AUD-216, inside
    `completion` as `<error …>`, which is what the judge files already on disk
    look like). Either way it must stay out of every denominator. And the gate is
    pre-registered in `docs/t6-reply-check-offline.md`, so a `GATE FAIL` line that
    exits 0 is a verdict the harness declines to carry.
    """

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)

    def _files(self, *cases: tuple[str, str, str, str, str | None]):
        """A review file and a judge output file from `(key, kind, truth, answer, error)`."""
        review = self.root / "review.tsv"
        lines = ["run\tsession\tkey\tkind\ttruth\tknown\tclaims\tspan"]
        rows = []
        for session, (key, kind, truth, completion, error) in enumerate(cases, start=1):
            lines.append(f"1\t{session}\t{key}\t{kind}\t{truth}\t-\t\t")
            row: dict[str, object] = {"note": f"r1s{session} {key}", "completion": completion}
            if error:
                row["error"] = error
            rows.append(row)
        review.write_text("\n".join(lines) + "\n", encoding="utf-8")
        done = self.root / "done.jsonl"
        done.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return review, done

    def _report(self, *cases):
        review, done = self._files(*cases)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            result = score.report("t", done, score.load_labels(review))
        return result, buffer.getvalue()

    def _main(self, *cases) -> tuple[int, str]:
        review, done = self._files(*cases)
        buffer = io.StringIO()
        argv = ["t6_prose_score.py", "--review", str(review), "--done", f"t:{done}"]
        with (
            unittest.mock.patch.object(sys, "argv", argv),
            contextlib.redirect_stdout(buffer),
        ):
            return score.main(), buffer.getvalue()

    def test_precision_recall_and_silence(self):
        result, _ = self._report(
            ("k/yes", "claim", "YES", "YES", None),
            ("k/missed", "claim", "YES", "NO", None),
            ("k/silent", "silent", "NO", "YES", None),
            ("k/wrong", "claim", "NO", "NO", None),
        )
        self.assertEqual(result["total"], 4)
        self.assertAlmostEqual(result["precision"], 0.5)  # fired twice, one right
        self.assertAlmostEqual(result["recall"], 0.5)  # one of two positives
        self.assertAlmostEqual(result["silence_rate"], 1.0)  # the one silent case fired

    def test_a_refused_row_is_not_scored_as_a_wrong_answer(self):
        result, out = self._report(
            ("k/yes", "claim", "YES", "YES", None),
            ("k/missed", "claim", "YES", "", "URLError: refused"),
        )
        self.assertEqual(result["total"], 1, f"a refusal was scored: {out}")
        self.assertEqual(result["refused"], 1)
        self.assertAlmostEqual(result["accuracy"], 1.0)
        self.assertAlmostEqual(result["recall"], 1.0)

    def test_a_refusal_written_into_the_completion_is_a_refusal_too(self):
        # The shape every judge file written before AUD-216 carries.
        result, _ = self._report(
            ("k/yes", "claim", "YES", "YES", None),
            (
                "k/missed",
                "claim",
                "YES",
                "<error <urlopen error [Errno 61] Connection refused>>",
                None,
            ),
        )
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["refused"], 1)

    def test_a_run_where_every_row_refused_measures_nothing(self):
        result, out = self._report(
            ("k/yes", "claim", "YES", "", "URLError: refused"),
            ("k/missed", "claim", "YES", "", "URLError: refused"),
            ("k/wrong", "claim", "NO", "", "URLError: refused"),
        )
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["refused"], 3)
        self.assertFalse(result["measured"])
        self.assertIn("measured nothing", out)
        self.assertNotIn("accuracy          0/0", out, f"0% printed for no measurement: {out}")

    def test_a_gate_that_fails_is_the_exit_code(self):
        code, out = self._main(
            ("k/yes", "claim", "YES", "YES", None),
            ("k/wrong", "claim", "NO", "YES", None),
        )
        self.assertIn("GATE", out)
        self.assertIn("FAIL", out)
        self.assertEqual(code, 1, f"a failed pre-registered gate exited {code}: {out}")

    def test_a_gate_that_passes_exits_clean(self):
        code, out = self._main(
            ("k/yes", "claim", "YES", "YES", None),
            ("k/yes2", "claim", "YES", "YES", None),
            ("k/wrong", "claim", "NO", "NO", None),
            ("k/silent", "silent", "NO", "NO", None),
        )
        self.assertNotIn("FAIL", out)
        self.assertEqual(code, 0, f"a gate that passed must not fail the run: {out}")

    def test_a_refused_row_fails_the_exit_code(self):
        code, out = self._main(
            ("k/yes", "claim", "YES", "YES", None),
            ("k/wrong", "claim", "NO", "NO", None),
            ("k/silent", "silent", "NO", "NO", None),
            ("k/missed", "claim", "YES", "", "URLError: refused"),
        )
        self.assertEqual(code, 1, f"a run that measured less than it claims exited {code}: {out}")


if __name__ == "__main__":
    unittest.main()
