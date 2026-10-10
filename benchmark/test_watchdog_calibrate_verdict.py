#!/usr/bin/env python3.13
"""AUD-292: what `watchdog_calibrate.py` is willing to certify.

The driver counts false positives for its own thresholds and its target is zero,
so the status is the whole report. Measured against HEAD at `1b25ef2`, with only
the three corpus readers patched (no log tree opened, nothing fetched):

    replies, no exchanges              exchanges, no conversations   status
    stub row prints `no corpus`        pingpong row prints `no corpus`   0
    stub row shows 0/1 exchanges       pingpong shows 0/N conversations  0

The two runs are indistinguishable by status, and the first never applied the
stub rule at all -- it printed `zero false positives on 1 replies` while saying
so beside itself. The pingpong watchdog is not in `clean` even when its corpus
is there: three identical tool calls in one recorded conversation, which is the
false positive the threshold exists to avoid, answers 0.

These tests pin the driver's own codes rather than renumbering them -- 0 clean,
1 the calibration did not get to run, 2 measured and not clean -- because AUD-292
records the numbering as the operator's decision, not the auditor's. What changes
is that a rule which saw no corpus can no longer certify the run, and a pingpong
trip is the false positive it is.

No model, no server, no engine binary, no log tree and nothing fetched: the
corpora are in-memory fixtures and `main()` and `selftest()` are the real ones.
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import watchdog_calibrate as driver  # noqa: E402

# Varied enough that the rolling-window detector cannot find a repeated phrase,
# and over the 96-byte stub threshold, so a present corpus reads as clean.
TEXT = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron "
    "pi rho sigma tau upsilon phi chi psi omega one two three four five six seven eight"
)


def reply(source="run/a.json", label="ok", text=TEXT):
    return {"text": text, "source": source, "label": label}


def exchange(source="run/a.json", text=TEXT, finish="stop", request_bytes=4000):
    return {"text": text, "finish": finish, "request_bytes": request_bytes, "source": source}


def conversation(tripped=False):
    call = ("search", '{"q":"weather"}')
    calls = [call] * 3 if tripped else [call, ("fetch", '{"u":"1"}'), ("write", '{"p":"2"}')]
    return calls


def run_main(replies, exchanges, conversations, argv=("watchdog_calibrate.py",)):
    buffer = io.StringIO()
    with (
        mock.patch.object(sys, "argv", list(argv)),
        mock.patch.object(driver, "replies", return_value=replies),
        mock.patch.object(driver, "exchanges", return_value=exchanges),
        mock.patch.object(driver, "tool_conversations", return_value=conversations),
        contextlib.redirect_stdout(buffer),
    ):
        status = driver.main()
    return status, buffer.getvalue()


def row(text: str, name: str) -> str:
    """The printed table's own line for one watchdog.

    The status and the incomplete note are one thing; the row is another. A test
    that only greps the whole output for `no corpus` is satisfied by the note
    beside it, so it reads the table's claim about the rule, not the claim.
    """
    return next(line for line in text.splitlines() if line.strip().startswith(name))


class CleanCertification(unittest.TestCase):
    def test_a_full_corpus_that_trips_nothing_answers_zero(self):
        """Preservation pin: the clean case stays 0 and stays worded as clean."""
        status, text = run_main([reply()], [exchange()], [conversation()])
        self.assertEqual(status, 0, text)
        self.assertIn("zero false positives", text)
        self.assertNotIn("no corpus", text)
        self.assertIn("over 1 recorded exchanges", row(text, "stub"))
        self.assertIn("0.0%", row(text, "pingpong"))

    def test_a_measured_trip_still_answers_two(self):
        """Preservation pin: the codes are not renumbered here.

        AUD-292 records that 2 means a measured disagreement in this driver and
        NOT MEASURED in the drivers AUD-273 through AUD-291 answer, and leaves
        which way round is the operator's call; this test holds the first.
        """
        short = exchange(text="a stub", finish="stop", request_bytes=4000)
        status, text = run_main([reply()], [short], [conversation()])
        self.assertEqual(status, 2, text)
        self.assertIn("NOT clean", text)
        self.assertIn("1 100.0% over 1 recorded exchanges", " ".join(row(text, "stub").split()))

    def test_a_measured_trip_outranks_a_rule_that_saw_nothing(self):
        """The incomplete note must not swallow a false positive it also has.

        Measured on this machine's own recorded corpus: the loop rule trips once,
        the pingpong corpus is not there, and a verdict that returns 1 for the
        missing corpus leaves `NOT clean` unsaid -- trading one honest printout
        for a status that hides the trip.
        """
        tripping = "x" * 200 + " the same phrase over and over again " * 8
        tripping += "tail " * 200
        status, text = run_main([reply(text=tripping)], [exchange(text=tripping)], [])
        self.assertEqual(status, 2, text)
        self.assertIn("NOT clean", text)
        self.assertIn("calibration incomplete", text)
        self.assertIn("no corpus", row(text, "pingpong"))

    def test_a_pingpong_trip_is_a_false_positive_and_not_a_clean_run(self):
        """The tool-call watchdog fires on a recorded conversation and answers 0."""
        status, text = run_main([reply()], [exchange()], [conversation(tripped=True)])
        self.assertEqual(status, 2, text)
        self.assertIn("NOT clean", text)
        self.assertIn("1 100.0%", " ".join(row(text, "pingpong").split()))


class RulesThatSawNoCorpus(unittest.TestCase):
    def test_no_recorded_exchanges_cannot_certify_the_stub_rule(self):
        status, text = run_main([reply()], [], [conversation()])
        self.assertEqual(status, 1, text)
        self.assertIn("no corpus", row(text, "stub"))
        self.assertIn("0.0%", row(text, "pingpong"))
        self.assertNotIn("zero false positives", text)

    def test_no_recorded_conversations_cannot_certify_the_pingpong_rule(self):
        status, text = run_main([reply()], [exchange()], [])
        self.assertEqual(status, 1, text)
        self.assertIn("no corpus", row(text, "pingpong"))
        self.assertIn("over 1 recorded exchanges", row(text, "stub"))
        self.assertNotIn("zero false positives", text)

    def test_the_stall_rule_is_declared_not_applicable_rather_than_absent(self):
        """Why stall stays out of the list: it is calibrated against timings.

        The docstring says its threshold comes from measured decode rates, so an
        empty text corpus is not a rule that failed to run. The row must keep
        saying that, or the absence reads as a measurement like the others.
        """
        status, text = run_main([reply()], [exchange()], [conversation()])
        self.assertEqual(status, 0, text)
        self.assertIn("not applicable", text)


class AbsentCorpus(unittest.TestCase):
    def test_no_replies_at_all_answers_one_and_names_the_directory(self):
        """The guard that exists stays: a calibration with nothing to read."""
        status, text = run_main([], [exchange()], [conversation()])
        self.assertEqual(status, 1, text)
        self.assertIn("no recorded replies", text)

    def test_selftest_without_its_fixture_answers_one_with_the_reason(self):
        missing = pathlib.Path(tempfile.mkdtemp()) / "watchdog-cases.json"
        buffer = io.StringIO()
        with (
            mock.patch.object(driver, "FIXTURE", missing),
            contextlib.redirect_stdout(buffer),
        ):
            status = driver.selftest()
        self.assertEqual(status, 1, buffer.getvalue() + f" (status {status})")
        self.assertIn("missing fixture", buffer.getvalue())


class SelftestAgreement(unittest.TestCase):
    def test_a_fixture_case_the_port_disagrees_on_is_counted_and_costs_the_run(self):
        """The port is the point of `--selftest`, so its verdict has to reach it."""
        cases = [
            {"name": "agreeing", "text": TEXT, "repeat": 1, "loopTrips": False},
            {"name": "disagreeing", "text": TEXT, "repeat": 1, "loopTrips": True},
        ]
        path = pathlib.Path(tempfile.mkdtemp()) / "watchdog-cases.json"
        path.write_text(json.dumps({"cases": cases}), encoding="utf-8")
        buffer = io.StringIO()
        with (
            mock.patch.object(driver, "FIXTURE", path),
            contextlib.redirect_stdout(buffer),
        ):
            status = driver.selftest()
        out = buffer.getvalue()
        self.assertEqual(status, 1, out)
        self.assertIn("MISMATCH disagreeing", out)
        self.assertIn("2 fixture cases, 1 mismatches", out)

    def test_a_fixture_the_port_agrees_with_answers_zero(self):
        cases = [{"name": "agreeing", "text": TEXT, "repeat": 1, "loopTrips": False}]
        path = pathlib.Path(tempfile.mkdtemp()) / "watchdog-cases.json"
        path.write_text(json.dumps({"cases": cases}), encoding="utf-8")
        buffer = io.StringIO()
        with (
            mock.patch.object(driver, "FIXTURE", path),
            contextlib.redirect_stdout(buffer),
        ):
            status = driver.selftest()
        self.assertEqual(status, 0, buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
