#!/usr/bin/env python3
"""Tests for `tinytitan_hit_fixup_ab.py`: AUD-280.

The driver answers one question: does the hit/fixup decode schedule produce the
same text as the barrier schedule while the server's runner and generation
footers record what each arm cost? Its docstring states both halves -- "response
text must match across modes before results pass" and "The server's
TINYTITAN_RUNNER_STATS footer supplies cache and I/O measurements".

Pre-fix only the first half existed, and it was the whole verdict. Measured on a
`git archive` of the committed tree by calling the real `main()` with the module's
own leaves patched (the process guard, the health wait, the memory probe, the HTTP
request, the log path) -- no model, no server, no build, no socket, nothing
fetched:

1. every response empty: `passed: true` and exit 0. Text equality between two
   arms that answered nothing is an agreement about nothing;
2. identical text, different lengths (`128` completion tokens in one arm, `96` in
   the other): `passed: true`, exit 0. `usage` is recorded and never read;
3. `decode_tok_s=nan`, `expert_hit_rate=nan`: exit 0. The footers are required by
   count and read by nothing, so the measurement the A/B exists for is unchecked;
4. `decode_tok_s=0.00` beside a run that claims `completion_tokens: 128`: exit 0,
   a self-contradiction the driver cannot see because it parses no figure;
5. a log with no footers, a refused preflight, a server that died, an HTTP 500:
   each escapes as a `RuntimeError` traceback and exits **1** -- the same status a
   measured disagreement carries -- and writes no artifact, so the cases that
   already ran are lost without a line saying which.

The statuses are the three this driver family already uses (AUD-273 through
AUD-279):

    0  every case compared, both arms answered, every published figure came from a
       footer that carried it, and the two arms agree
    1  it measured, and the page is contested -- by a named disagreement or by
       cases it could not compare, even when the disagreement covers every case
    2  the measurement does not exist: the driver's own refusal, every case
       answered nothing, or every case's figures unreadable, with the reason
       printed and no artifact written

Refusal tokens follow AUD-276: a table cell that has no figure prints the `--`
this driver's own refusal column uses, a live line uses the words.
"""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))

import tinytitan_hit_fixup_ab as hf  # noqa: E402

PROMPTS = ("short", "medium", "long")
WARMTH = ("cold", "warm")
MODES = ("barrier", "hit-fixup")


def footer(tok="41.20", rate="0.7500", hits="100", misses="33", fix="12"):
    return {
        "generation": f"TinyTitan generation decode_tok_s={tok} prefill_tok_s=1200.0",
        "runner": (
            f"TinyTitan runner expert_hit_rate={rate} expert_hits={hits} "
            f"expert_misses={misses} hit_fixup_layers={fix}"
        ),
    }


def row(
    mode,
    prompt,
    warmth,
    *,
    content="a real paragraph of routed output.",
    tokens=128,
    footers=None,
):
    footers = footers or footer()
    return {
        "mode": mode,
        "prompt": prompt,
        "warmth": warmth,
        "io_backend": "pread",
        "io_sync": "host",
        "io_submission": "deferred",
        "wall_seconds": 3.1,
        "content": content,
        "usage": {"completion_tokens": tokens},
        "baseline_memory": {"server_rss_kib": 1000, "machine_free_percent": 60},
        "memory_after": {"server_rss_kib": 1100, "machine_free_percent": 59},
        "generation_footer": footers["generation"],
        "runner_footer": footers["runner"],
    }


def sweep(**overrides):
    """Every case in both arms, with `overrides` applied to the rows."""
    rows = []
    for mode in MODES:
        for prompt in PROMPTS:
            for warmth in WARMTH:
                rows.append(row(mode, prompt, warmth, **overrides))
    return rows


def cell(lines, label):
    """The answer a published counter line carries, read from the line itself."""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(label):
            return stripped[len(label) :].strip()
    raise AssertionError(f"no {label!r} line in:\n" + "\n".join(lines))


class AnsweredSweep(unittest.TestCase):
    def test_a_sweep_where_every_case_answered_and_agrees_passes(self):
        lines, status = hf.verdict(sweep(), PROMPTS)
        self.assertEqual(status, 0, "\n".join(lines))
        self.assertEqual(cell(lines, "cases compared"), "6 of 6")

    def test_the_page_prints_each_arm_s_measured_rate(self):
        lines, status = hf.verdict(sweep(), PROMPTS)
        self.assertEqual(status, 0)
        case = next(line for line in lines if line.strip().startswith("short cold"))
        self.assertIn("41.20 tok/s", case)
        self.assertIn("75.0% hit", case)

    def test_a_text_disagreement_is_a_measured_negative_not_a_refusal(self):
        rows = sweep()
        for item in rows:
            if (
                item["mode"] == "hit-fixup"
                and item["prompt"] == "medium"
                and item["warmth"] == "warm"
            ):
                item["content"] = "a different paragraph entirely."
        lines, status = hf.verdict(rows, PROMPTS)
        self.assertEqual(status, 1, "\n".join(lines))
        self.assertEqual(cell(lines, "cases compared"), "5 of 6")
        self.assertIn("medium/warm text differs", "\n".join(lines))
        self.assertNotIn("NOT MEASURED", "\n".join(lines))

    def test_a_length_disagreement_is_a_finding_the_page_names(self):
        rows = sweep()
        for item in rows:
            if item["mode"] == "hit-fixup":
                item["usage"] = {"completion_tokens": 96}
        lines, status = hf.verdict(rows, PROMPTS)
        self.assertEqual(status, 1, "\n".join(lines))
        self.assertIn("128 vs 96", "\n".join(lines))
        self.assertIn("cases compared 0 of 6", "\n".join(lines))


class UnansweredCases(unittest.TestCase):
    def test_two_arms_answering_nothing_is_not_an_agreement(self):
        lines, status = hf.verdict(sweep(content=""), PROMPTS)
        self.assertEqual(status, 2, "\n".join(lines))
        self.assertIn("NOT MEASURED", "\n".join(lines))
        self.assertNotIn("cases compared 6 of 6", "\n".join(lines))

    def test_one_unanswered_case_contests_the_page_instead_of_passing(self):
        rows = sweep()
        for item in rows:
            if item["prompt"] == "long" and item["warmth"] == "cold":
                item["content"] = ""
        lines, status = hf.verdict(rows, PROMPTS)
        self.assertEqual(status, 1, "\n".join(lines))
        self.assertEqual(cell(lines, "cases compared"), "5 of 6")
        self.assertIn("CONTESTED", "\n".join(lines))
        self.assertIn("long/cold answered nothing", "\n".join(lines))

    def test_an_answer_of_only_spaces_is_not_an_answer(self):
        lines, status = hf.verdict(sweep(content="   "), PROMPTS)
        self.assertEqual(status, 2, "\n".join(lines))
        self.assertIn("NOT MEASURED", "\n".join(lines))

    def test_an_unanswered_case_is_not_reported_as_a_disagreement(self):
        rows = sweep()
        for item in rows:
            if item["mode"] == "hit-fixup" and item["prompt"] == "short":
                item["content"] = ""
        lines, status = hf.verdict(rows, PROMPTS)
        self.assertEqual(status, 1, "\n".join(lines))
        joined = "\n".join(lines)
        self.assertNotIn("text differs", joined)
        self.assertIn("hit-fixup short answered nothing", joined)

    def test_no_case_rows_at_all_refuses_instead_of_passing(self):
        lines, status = hf.verdict([], PROMPTS)
        self.assertEqual(status, 2, "\n".join(lines))
        self.assertIn("no case rows", "\n".join(lines))
        self.assertNotIn("cases compared", "\n".join(lines))

    def test_no_prompt_cases_refuses_instead_of_passing_zero_of_zero(self):
        lines, status = hf.verdict(sweep(), ())
        self.assertEqual(status, 2, "\n".join(lines))
        self.assertIn("no case rows to compare", "\n".join(lines))
        self.assertNotIn("cases compared 0 of 0", "\n".join(lines))


class UnreadableFigures(unittest.TestCase):
    def test_a_footer_figure_that_is_not_a_number_refuses_the_run(self):
        lines, status = hf.verdict(sweep(footers=footer(tok="nan", rate="nan")), PROMPTS)
        self.assertEqual(status, 2, "\n".join(lines))
        self.assertIn("NOT MEASURED", "\n".join(lines))
        self.assertIn("decode_tok_s=nan", "\n".join(lines))

    def test_a_zero_rate_beside_a_full_completion_is_not_a_measurement(self):
        rows = sweep(footers=footer(tok="0.00"))
        lines, status = hf.verdict(rows, PROMPTS)
        self.assertEqual(status, 2, "\n".join(lines))
        self.assertIn("0.00 tok/s while it reported 128 completion tokens", "\n".join(lines))

    def test_a_missing_footer_key_leaves_the_cell_blank_and_contests_the_page(self):
        rows = sweep()
        for item in rows:
            if item["prompt"] == "medium":
                item["runner_footer"] = "TinyTitan runner expert_hits=10 expert_misses=2"
        lines, status = hf.verdict(rows, PROMPTS)
        self.assertEqual(status, 1, "\n".join(lines))
        self.assertEqual(cell(lines, "cases compared"), "4 of 6")
        self.assertIn("medium", "\n".join(lines))
        self.assertIn("-- hit", "\n".join(lines))

    def test_a_generation_footer_without_a_decode_key_prints_a_blank_cell(self):
        rows = sweep()
        for item in rows:
            if item["prompt"] == "short":
                item["generation_footer"] = "TinyTitan generation prefill_tok_s=1200.0"
        lines, status = hf.verdict(rows, PROMPTS)
        self.assertEqual(status, 1, "\n".join(lines))
        self.assertEqual(cell(lines, "cases compared"), "4 of 6")
        self.assertIn("-- tok/s", "\n".join(lines))
        self.assertIn("hit-fixup short/cold logged no decode_tok_s", "\n".join(lines))

    def test_footer_value_reads_a_figure_or_refuses_it(self):
        line = footer(tok="12.50", rate="0.2500")["runner"]
        self.assertEqual(hf.footer_value(line, "expert_hit_rate"), 0.25)
        self.assertIsNone(hf.footer_value(line, "io_ms"))
        self.assertIsNone(hf.footer_value(footer(tok="nan")["generation"], "decode_tok_s"))
        self.assertIsNone(hf.footer_value(footer(tok="inf")["generation"], "decode_tok_s"))
        self.assertIsNone(hf.footer_value(footer(tok="abc")["generation"], "decode_tok_s"))

    def test_one_unreadable_figure_names_the_reason_and_stays_contested(self):
        rows = sweep()
        bad = footer(tok="nan", rate="nan")
        for item in rows:
            if item["prompt"] == "long" and item["warmth"] == "warm":
                item["generation_footer"] = bad["generation"]
                item["runner_footer"] = bad["runner"]
        lines, status = hf.verdict(rows, PROMPTS)
        joined = "\n".join(lines)
        self.assertEqual(status, 1, joined)
        self.assertEqual(cell(lines, "cases compared"), "5 of 6")
        self.assertIn("a published figure could not be read as a number", joined)

    def test_one_blank_cell_names_the_reason_as_well_as_the_cell(self):
        rows = sweep()
        for item in rows:
            if item["prompt"] == "short" and item["warmth"] == "cold":
                item["runner_footer"] = "TinyTitan runner expert_hit_layers=0"
        joined = "\n".join(hf.verdict(rows, PROMPTS)[0])
        self.assertIn("prints -- because no footer logged it", joined)
        self.assertIn("logged no expert_hit_rate", joined)

    def test_a_generation_line_without_a_decode_rate_is_not_a_footer(self):
        with tempfile.TemporaryDirectory() as directory:
            log = pathlib.Path(directory) / "server.log"
            log.write_text(
                "TinyTitan generation tokens=128 wall_seconds=3.1\n"
                + footer()["generation"]
                + "\n",
                encoding="utf-8",
            )
            generation, _runner = hf.parse_footers(log)
            self.assertEqual(len(generation), 1)


class DecodeRateAndLengthControls(unittest.TestCase):
    """The two figures the first cut of the fix still let slip through.

    Measured after `verdict()` existed: a sweep whose every footer read
    `decode_tok_s=0.00` beside a real answer still exited 0, because the rule that
    caught it required a non-zero token claim, and a response with no `usage` at
    all still counted as compared, because the length check it never ran was not
    published as missing.
    """

    def test_a_zero_rate_beside_an_answer_measured_no_decode(self):
        rows = sweep(footers=footer(tok="0.00"))
        for item in rows:
            item["usage"] = {}
        lines, status = hf.verdict(rows, PROMPTS)
        self.assertEqual(status, 2, "\n".join(lines))
        self.assertIn("NOT MEASURED", "\n".join(lines))
        self.assertIn("0.00 tok/s while the response carried text", "\n".join(lines))

    def test_a_response_that_claimed_no_tokens_leaves_the_length_check_unrun(self):
        rows = sweep()
        for item in rows:
            if item["mode"] == "hit-fixup":
                item["usage"] = {}
        lines, status = hf.verdict(rows, PROMPTS)
        self.assertEqual(status, 1, "\n".join(lines))
        self.assertEqual(cell(lines, "cases compared"), "0 of 6")
        self.assertIn("completion length was never compared", "\n".join(lines))
        self.assertIn("claimed no completion token count", "\n".join(lines))
        self.assertIn("-- claimed", "\n".join(lines))


class PartialRowSets(unittest.TestCase):
    """A case that ran only one arm is refused, never counted out of a smaller total.

    AUD-233's shape: a denominator built from whatever rows arrived reports a
    complete sweep over the survivors, so `cases compared 4 of 4` reads as clean
    while two arms never ran.
    """

    def test_a_missing_arm_raises_rather_than_shrinking_the_denominator(self):
        rows = [
            item
            for item in sweep()
            if not (item["mode"] == "hit-fixup" and item["prompt"] == "medium")
        ]
        with self.assertRaises(KeyError):
            hf.verdict(rows, PROMPTS)


class MainDrivesTheSweep(unittest.TestCase):
    """End to end through `main()`, with only the module's leaves patched."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="hit-fixup-ab-test-"))
        (self.tmp / "logs").mkdir(parents=True, exist_ok=True)
        self.state = {"handle": None, "mode": None, "calls": 0}

    def drive(
        self, *, content="a real paragraph.", tokens=128, other_tokens=None, footers=None, fail=None
    ):
        footers = footers or footer()
        real = (
            hf.ROOT,
            hf.PROMPTS,
            hf.preflight,
            hf.wait_until_healthy,
            hf.memory_snapshot,
            hf.resolve_api_model,
            hf.benchmark_log_path,
            hf.request,
            hf.server_command,
            subprocess.Popen,
        )
        state = self.state

        def fake_popen(command, *, env=None, stdout=None, stderr=None, text=None):
            state["handle"] = stdout.name
            state["mode"] = env["TINYTITAN_DECODE_EXPERT_EXECUTION"]
            return _FakeProcess()

        def fake_request(prompt):
            state["calls"] += 1
            claimed = tokens
            if other_tokens is not None and state["mode"] == "hit-fixup":
                claimed = other_tokens
            if fail == "request":
                raise RuntimeError("request failed HTTP 500: server error")
            if fail == "footers" or (fail == "footers-late" and state["calls"] > 2):
                return {
                    "wall_seconds": 1.0,
                    "content": content,
                    "usage": {"completion_tokens": claimed},
                }
            handle = pathlib.Path(state["handle"])
            handle.write_text(
                handle.read_text(encoding="utf-8")
                + footers["generation"]
                + "\n"
                + footers["runner"]
                + "\n",
                encoding="utf-8",
            )
            return {
                "wall_seconds": 1.0,
                "content": content,
                "usage": {"completion_tokens": claimed},
            }

        def log_path(name):
            return str(self.tmp / "logs" / name)

        hf.ROOT = self.tmp
        hf.PROMPTS = dict.fromkeys(PROMPTS, "a prompt")
        hf.preflight = lambda: {"commit": "abc", "model": "m"}
        hf.wait_until_healthy = lambda process: None
        hf.memory_snapshot = lambda pid: {"server_rss_kib": 1, "machine_free_percent": 50}
        hf.resolve_api_model = lambda port, **kw: "test-model"
        hf.benchmark_log_path = log_path
        hf.request = fake_request
        hf.server_command = lambda server, port: ["/bin/true"]
        subprocess.Popen = fake_popen
        if fail == "preflight":
            hf.preflight = lambda: (_ for _ in ()).throw(
                RuntimeError("release server missing; run swift build -c release")
            )
        out, err = io.StringIO(), io.StringIO()
        argv = [str(ROOT / "benchmark" / "tinytitan_hit_fixup_ab.py")]
        try:
            with (
                mock.patch.object(sys, "argv", argv),
                contextlib.redirect_stdout(out),
                contextlib.redirect_stderr(err),
            ):
                status = hf.main()
        finally:
            (
                hf.ROOT,
                hf.PROMPTS,
                hf.preflight,
                hf.wait_until_healthy,
                hf.memory_snapshot,
                hf.resolve_api_model,
                hf.benchmark_log_path,
                hf.request,
                hf.server_command,
                subprocess.Popen,
            ) = real
        artifact = self.tmp / ".build/benchmark-results/hit-fixup-ab.json"
        return {
            "status": status,
            "out": out.getvalue(),
            "err": err.getvalue(),
            "artifact": json.loads(artifact.read_text()) if artifact.exists() else None,
        }

    def test_a_clean_sweep_exits_zero_and_records_the_status_it_returned(self):
        result = self.drive()
        self.assertEqual(result["status"], 0, result["out"] + result["err"])
        self.assertEqual(result["artifact"]["status"], 0)
        self.assertEqual(result["artifact"]["cases_compared"], 6)
        self.assertEqual(result["artifact"]["cases_total"], 6)
        self.assertEqual(result["artifact"]["uncompared_cases"], [])
        self.assertIn("cases compared 6 of 6", result["out"])

    def test_a_contested_sweep_writes_the_record_with_the_status_it_returned(self):
        result = self.drive(other_tokens=96)
        self.assertEqual(result["status"], 1, result["out"] + result["err"])
        self.assertEqual(result["artifact"]["status"], 1)
        self.assertFalse(result["artifact"]["passed"])
        self.assertEqual(result["artifact"]["cases_compared"], 0)
        self.assertEqual(result["artifact"]["cases_total"], 6)
        self.assertEqual(
            result["artifact"]["uncompared_cases"],
            [f"{prompt}/{warmth} differ" for prompt in PROMPTS for warmth in WARMTH],
        )
        self.assertEqual(
            result["artifact"]["response_mismatches"],
            [f"{prompt}/{warmth}" for prompt in PROMPTS for warmth in WARMTH],
        )

    def test_answering_nothing_does_not_pass(self):
        result = self.drive(content="")
        self.assertEqual(result["status"], 2, result["out"] + result["err"])
        self.assertIn("NOT MEASURED", result["out"])
        self.assertIsNone(result["artifact"])

    def test_a_missing_footer_log_refuses_instead_of_tracing_back(self):
        result = self.drive(fail="footers")
        self.assertEqual(result["status"], 2, result["err"])
        self.assertIn("NOT MEASURED", result["out"])
        self.assertIn("missing benchmark footers", result["out"])
        self.assertIsNone(result["artifact"])

    def test_a_refusal_after_a_completed_case_names_that_case(self):
        result = self.drive(fail="footers-late")
        self.assertEqual(result["status"], 2, result["out"] + result["err"])
        self.assertIn("missing benchmark footers", result["out"])
        self.assertIn(
            "cases that ran before the refusal: barrier short/cold, barrier short/warm",
            result["out"],
        )
        self.assertIsNone(result["artifact"])

    def test_a_refused_preflight_refuses_with_its_own_reason(self):
        result = self.drive(fail="preflight")
        self.assertEqual(result["status"], 2, result["err"])
        self.assertIn("release server missing", result["out"])
        self.assertNotIn("barrier", result["out"])

    def test_a_request_that_fails_names_the_case_that_ran_before_it_died(self):
        result = self.drive(fail="request")
        self.assertEqual(result["status"], 2, result["err"])
        self.assertIn("request failed HTTP 500", result["out"])
        self.assertIn("no case ran before the refusal", result["out"])

    def test_the_verdict_status_is_the_answer_not_a_traceback(self):
        result = self.drive(footers=footer(tok="nan", rate="nan"))
        self.assertEqual(result["status"], 2, result["out"] + result["err"])
        self.assertIn("NOT MEASURED", result["out"])


class _FakeProcess:
    pid = 4242

    def poll(self):
        return None

    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


if __name__ == "__main__":
    unittest.main()
