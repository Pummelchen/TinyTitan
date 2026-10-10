#!/usr/bin/env python3
"""Tests for `guard_source_rate.py`: AUD-278.

The guard rests on one bit per fact -- did the person assert this, or did the
model derive it -- and this driver measures how often that label is wrong before
the guard is switched on anywhere. Its docstring and `docs/test-plan.md` step 16
say the same thing about what the answer is for: "run this, read every flag, and
record the verdict", because word overlap cannot judge a boolean whose claim
lives in its key. So the page has three trust conditions of its own: the scorer
has to be able to fail (a scorer "generous enough to ground anything would report
a perfect run whatever the model did"), every authority-carrying fact has to
actually be compared to something, and the control -- the model's own facts put
through the same test -- has to discriminate.

Pre-fix, none of the three reached the exit status. Measured by driving the real
`main()` with only `LOGS` redirected at a fixture journal, no model, no server,
no socket, nothing fetched:

1. `--threshold 0` and `--threshold -1` over a journal holding a real invented
   fact printed `of those, mislabelled 0 (0.0%)` and exited 0 -- an operator-set
   knob makes `overlap >= threshold` true for every fact, so the gate's pass is
   guaranteed by construction and the flag list the procedure depends on is empty;
2. `--threshold 2` put a fact with `overlap=100%` in "the user never said" list;
3. a journal whose authority-carrying facts carry no distinctive words scored
   none of them, and exited 0 with `mislabelled 0 (0.0%)` on a page that says
   `carrying authority 2` -- a run that compared nothing answering "no invented
   fact wearing the person's label";
4. with 4 such facts among 6, the rate printed `1  (16.7%)` over a denominator
   of facts of which only 2 were scored, and nothing on the page said so;
5. the control printing `2/2 (100%)` beside its own line "a rate near the user
   rate would mean the test does not discriminate" left the status at 0;
6. a journal with no model-labelled fact at all skipped the control line
   silently, so the scorer's discrimination was simply unproven at exit 0;
7. `1` means "no recorded runs / no journal / no facts" (a refusal) and `2` means
   both "nothing was labelled user, the gate cannot be judged" (a refusal) and
   "these facts are inventions" (a measured, contested answer).

The statuses are the three this tree's drivers already use (AUD-273, AUD-274,
AUD-275, AUD-276, AUD-277), and `3` is not used here because this driver starts
no process:

    0  every authority-carrying fact was scored, the control discriminated, and
       nothing was flagged
    1  it measured, and facts wearing the person's label were flagged -- each named
    2  the gate could not be judged, with the reason named: no run, no journal, no
       facts, nothing scoreable, a threshold that cannot fail or cannot be met, an
       unproven or non-discriminating control

A threshold is in the domain the comparison means something for when
`0 < threshold <= 1`: at or below 0 every fact is grounded, above 1 none can be,
and 1 itself is the strictest honest bar -- a fact whose every distinctive word
the person wrote.

The control has one exception, and it is tested on both sides of it. Refusing any
run whose control rate is not *strictly below* the user rate would catch the
zero-against-zero page -- every authority-carrying fact flagged -- and report the
worst result the gate can get as "this run proved nothing". That is the softening
this whole family of findings is about, so the rule is: a control that grounds at a
higher rate, or at an equal non-zero rate, refuses the page; zero against zero is a
`1` that names every fact and adds the other reading to the page.
"""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))

import guard_source_rate as g  # noqa: E402


def rec(namespace, key, value, session, author):
    return {
        "memory": {
            "_0": {
                "namespace": namespace,
                "key": key,
                "value": value,
                "provenance": {"sessionID": session, "author": author},
            }
        }
    }


# Vocabulary taken from `memory_sim`'s own bible and plot events, which is what
# `user_text()` returns: "rosa"/"hazel" and "rosa"/"inn"/"burned" are the
# person's words, while `zqpl/wxrt_cipher = bruxelize` is nobody's.
GROUNDED = rec("k.rosa", "eye_colour", "hazel", "s1", "user")
INVENTED = rec("k.zqpl", "wxrt_cipher", "bruxelize", "s1", "user")
# Address and value carry only scaffold words, so score() has nothing to compare.
UNSCOREABLE = rec("k.the", "is", "it", "s1", "user")
MODEL_GROUNDED = rec("k.rosa", "inn", "burned", "s2", "model")
MODEL_INVENTED = rec("k.vqmm", "ztqr_quilt", "flummoxed", "s2", "model")


def drive(records, argv, *, labelled=True):
    """The real `main()` over a fixture journal in a temp tree.

    Only `LOGS` moves: the scorer, the journal reader and `memory_sim`'s user
    text are the shipped ones, so every figure on the page is one the tool
    computed. `labelled=False` builds a tree with no runs at all.
    """
    with tempfile.TemporaryDirectory() as tmp:
        logs = pathlib.Path(tmp)
        if labelled:
            run = logs / "memval-scratch-probe" / "book-auto-r1" / "tinytitan" / "sess"
            run.mkdir(parents=True)
            (run / "a.ndjson").write_text(
                "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
            )
        buffer = io.StringIO()
        with (
            mock.patch.object(g, "LOGS", logs),
            mock.patch.object(sys, "argv", ["guard_source_rate.py", *argv]),
            contextlib.redirect_stdout(buffer),
            contextlib.redirect_stderr(buffer),
        ):
            status = g.main()
        return status, buffer.getvalue(), logs


def rate_line(out, phrase="mislabelled"):
    for line in out.splitlines():
        if phrase in line:
            return line
    raise AssertionError(f"no {phrase!r} line in:\n{out}")


class ThresholdDomain(unittest.TestCase):
    def test_a_threshold_that_grounds_everything_refuses_by_name(self):
        self.assertIn("--threshold", g.threshold_refusal(0.0))
        self.assertIn("0", g.threshold_refusal(0.0))

    def test_a_negative_threshold_refuses_too(self):
        self.assertIn("--threshold", g.threshold_refusal(-1.0))

    def test_a_threshold_no_overlap_can_reach_refuses_too(self):
        self.assertIn("--threshold", g.threshold_refusal(1.5))

    def test_the_strictest_honest_bar_and_the_default_are_both_accepted(self):
        # 1.0 asks for every distinctive word and 0.5 is what the plan's own
        # hand-checked numbers were taken at; neither can be reached only by
        # construction.
        self.assertIsNone(g.threshold_refusal(1.0))
        self.assertIsNone(g.threshold_refusal(0.5))

    def test_the_whole_bar_grounds_a_fact_sharing_every_word(self):
        # `overlap >= threshold` is the boundary the docstring means by "the
        # strictest honest bar": at 1.0 a fact whose every distinctive word the
        # person wrote still grounds. With a strict `>` the page would call the
        # person's own bible an invention.
        status, out, _ = drive([GROUNDED, MODEL_INVENTED], ["--threshold", "1.0"])
        self.assertEqual(status, 0, out)
        self.assertNotIn("the user never said", out)

    def test_the_refusal_comes_before_the_journal_is_read(self):
        # No run tree exists at all, so a page that reached the journal would
        # say so; naming the threshold proves the refusal is first.
        status, out, _ = drive([], ["--threshold", "0"], labelled=False)
        self.assertEqual(status, 2, out)
        self.assertIn("--threshold", out)
        self.assertNotIn("no recorded book runs", out)


class ScoredFacts(unittest.TestCase):
    def test_a_fact_with_no_distinctive_words_is_counted_as_unscored(self):
        status, out, _ = drive([GROUNDED, UNSCOREABLE, MODEL_INVENTED], [])
        self.assertEqual(status, 1, out)
        self.assertIn("scored 1 of 2", out)
        self.assertIn("1 carried no distinctive words", out)
        self.assertIn("were never compared to the person's words", out)

    def test_the_rate_is_over_the_facts_that_were_scored(self):
        # One flagged fact and one that was never compared: the honest answer is
        # 1 of 1, not 1 of 2.
        status, out, _ = drive([INVENTED, UNSCOREABLE, MODEL_INVENTED], [])
        self.assertEqual(status, 1, out)
        self.assertIn("(100.0%", rate_line(out))
        self.assertNotIn("16.7%", out)

    def test_a_run_that_scored_nothing_refuses_instead_of_passing(self):
        status, out, _ = drive([UNSCOREABLE, UNSCOREABLE, MODEL_INVENTED], [])
        self.assertEqual(status, 2, out)
        self.assertIn("NOT MEASURED", out)
        self.assertIn("nothing the scorer could compare", out)

    def test_a_threshold_refusal_is_named_and_scores_nothing(self):
        for value in ("0", "-1", "2"):
            status, out, _ = drive([GROUNDED, INVENTED, MODEL_INVENTED], ["--threshold", value])
            self.assertEqual(status, 2, out)
            self.assertIn("--threshold", out)
            self.assertNotIn("0.0%", out)


class TheControl(unittest.TestCase):
    def test_a_control_that_does_not_discriminate_refuses_the_page(self):
        # Both classes of fact ground at 100%, which is the docstring's own
        # description of a measurement that is not measuring anything.
        status, out, _ = drive([GROUNDED, GROUNDED, MODEL_GROUNDED, MODEL_GROUNDED], [])
        self.assertEqual(status, 2, out)
        self.assertIn("does not discriminate", out)
        self.assertIn("2/2", out)

    def test_a_discriminating_control_leaves_a_clean_run_at_zero(self):
        # Control on the same shape: the person's facts ground, the model's do
        # not, and nothing is flagged.
        status, out, _ = drive([GROUNDED, MODEL_INVENTED], [])
        self.assertEqual(status, 0, out)
        self.assertIn("control:", out)

    def test_a_journal_with_no_control_fact_refuses_as_unproven(self):
        status, out, _ = drive([GROUNDED], [])
        self.assertEqual(status, 2, out)
        self.assertIn("no model-labelled fact", out)

    def test_a_flagged_run_with_a_non_discriminating_control_is_a_refusal(self):
        # The flags are the page's product, and a scorer that grounds the model's
        # own inventions just as readily produces no evidence from them.
        status, out, _ = drive([INVENTED, MODEL_GROUNDED], [])
        self.assertEqual(status, 2, out)
        self.assertIn("does not discriminate", out)

    def test_every_fact_flagged_is_contested_not_unmeasured(self):
        # Zero against zero is the control rule's one exception: a run where the
        # person's facts ground 0% and the model's ground 0% is the worst measured
        # answer the gate has, and every fact on it is named. Turning it into
        # "nothing proved" would soften a scandal into an inconclusive run.
        status, out, _ = drive([INVENTED, MODEL_INVENTED], [])
        self.assertEqual(status, 1, out)
        self.assertNotIn("NOT MEASURED", out)
        self.assertIn("no fact grounded at all", out)
        self.assertIn("zqpl/wxrt_cipher", out)

    def test_unscored_facts_do_not_inflate_the_rate_the_control_is_barred_against(self):
        # The bar is the scored set: 3 facts carry authority, 2 could be compared,
        # 1 of those grounds, so the person's rate is 50% and a control grounding
        # 1 of 2 is at it. Dividing by the facts that were never compared would
        # read 67%, let that control pass, and report the flags as evidence.
        status, out, _ = drive(
            [GROUNDED, INVENTED, UNSCOREABLE, MODEL_GROUNDED, MODEL_INVENTED], []
        )
        self.assertEqual(status, 2, out)
        self.assertIn("does not discriminate", out)
        self.assertIn("1/2 of the model's own facts (50%) ground as the person's against 50%", out)

    def test_a_control_that_grounds_more_than_the_persons_is_a_refusal(self):
        # The worse-than-equal direction: half the person's facts ground and every
        # model fact does, so the grounded bit runs against the label.
        status, out, _ = drive([GROUNDED, INVENTED, MODEL_GROUNDED, MODEL_GROUNDED], [])
        self.assertEqual(status, 2, out)
        self.assertIn("does not discriminate", out)
        self.assertIn("(100%) ground as the person's against 50%", out)


class Statuses(unittest.TestCase):
    def test_facts_wearing_the_persons_label_are_contested_not_unmeasured(self):
        status, out, _ = drive([GROUNDED, INVENTED, MODEL_INVENTED], [])
        self.assertEqual(status, 1, out)
        self.assertIn("zqpl/wxrt_cipher", out)
        self.assertIn("bruxelize", out)

    def test_a_measured_clean_run_is_zero(self):
        status, out, _ = drive([GROUNDED, MODEL_INVENTED], [])
        self.assertEqual(status, 0, out)

    def test_no_recorded_runs_refuses_with_the_directory_named(self):
        status, out, logs = drive([], [], labelled=False)
        self.assertEqual(status, 2, out)
        self.assertIn(str(logs), out)

    def test_a_label_with_no_journal_refuses_and_names_the_label(self):
        status, out, _ = drive([GROUNDED], ["--label", "nosuch"])
        self.assertEqual(status, 2, out)
        self.assertIn("nosuch", out)

    def test_a_journal_holding_no_facts_refuses(self):
        status, out, _ = drive([{}, {"memory": {}}], [])
        self.assertEqual(status, 2, out)
        self.assertIn("no facts", out)

    def test_the_provenance_author_is_what_carries_the_label(self):
        # The store translates `isUserAsserted` into the provenance author, and
        # reading the record field instead would silently find nothing and
        # report a perfect run. A fact asserting both must land on the refusal.
        both = {
            "memory": {
                "_0": {
                    "namespace": "k.rosa",
                    "key": "eye_colour",
                    "value": "hazel",
                    "isUserAsserted": True,
                    "provenance": {"sessionID": "s1", "author": "model"},
                }
            }
        }
        status, out, _ = drive([both], [])
        self.assertEqual(status, 2, out)
        self.assertIn("nothing was labelled user", out)


class PublishedLinesStillPrint(unittest.TestCase):
    """Controls: the shapes a measured run has always printed must survive."""

    def test_the_counter_block_still_names_every_row(self):
        status, out, _ = drive([GROUNDED, INVENTED, MODEL_INVENTED], [])
        for row in (
            "facts written",
            "labelled user",
            "labelled model",
            "demoted, value not atomic",
            "carrying authority",
        ):
            self.assertIn(row, out)
        self.assertIn("facts claiming the user's authority that the user never said:", out)

    def test_a_flagged_fact_keeps_its_line_shape(self):
        status, out, _ = drive([GROUNDED, INVENTED, MODEL_INVENTED], [])
        self.assertIn(
            "s1  zqpl/wxrt_cipher                       overlap=0%  bruxelize",
            out,
        )

    def test_show_still_prints_the_grounded_list(self):
        status, out, _ = drive([GROUNDED, MODEL_INVENTED], ["--show"])
        self.assertEqual(status, 0, out)
        self.assertIn("1 grounded:", out)
        self.assertIn("rosa/eye_colour", out)
        self.assertIn("overlap=100%", out)

    def test_a_composite_value_is_still_counted_as_demoted(self):
        composite = rec("k.rosa", "notes", "hazel eyes; grey skies", "s1", "user")
        status, out, _ = drive([composite, MODEL_INVENTED], [])
        self.assertIn(
            "demoted, value not atomic        1  (a composite cannot have one source)",
            out,
        )
        self.assertIn("nothing was labelled user", out)


if __name__ == "__main__":
    unittest.main()
