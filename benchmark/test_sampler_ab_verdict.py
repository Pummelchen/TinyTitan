#!/usr/bin/env python3
"""Tests for `tinytitan_sampler_ab.py`, whose page cannot fail: AUD-275.

The driver answers one question -- does the tiled Top-K path emit the same token
as the `generic` fall-through, and what does it cost -- and it printed the answer
and returned 0 whatever the two arms had measured. Nothing here starts a server,
a model, a build or a download: every shape was produced by driving the real
`main()` with its own `one_run()` stubbed, and the page itself is driven through
`summarize()` + `verdict()`.

Measured on the committed tree before the fix (twelve shapes, the driver's own
`main()` called):

1. no run logged `busy_per_token_ms`: every line printed `busy/token 0.00 ms` and
   the summary printed `busy/token 0.00 ms`, rc=0;
2. half the runs logged it and half did not: the median was taken over a list
   `row.get("busy_per_token_ms", 0)` had padded with zeros, so the page published
   `busy/token 25.00 ms` -- a figure between the two real ones and derived from
   nothing, rc=0;
3. no run logged the `head_logits->embed` gap: `sample_gap 0.00 ms` and
   `sampler gap 0.00 ms`, rc=0 -- that gap is the mechanism the sweep exists to
   compare;
4. `busy_per_token_ms` present but null: `TypeError: unsupported format string
   passed to NoneType.__format__` in the first progress line, so the sweep died
   mid-run set;
5. `--pairs 0`: `StatisticsError: no median for empty data` -- no page, no
   artifact, and the same for `--pairs -1`;
6. the `generic` arm at 0 tok/s in every run: eight `0.000 tok/s` lines, then
   `ZeroDivisionError: division by zero` at `:103`, before any of the summary
   printed;
7. both arms streaming nothing (`completion_tokens: 0`) at one digest:
   `Output identical across every run of both arms: YES`, rc=0 -- the sha256 of
   "" is a digest, so a sweep that generated nothing certifies the sampler path;
8. the printed gate: `DELTA: +0.00%   (gate is +10%: FAIL)` with rc=0, because
   `:135` reads only `output_identical`. The exit status can still be 1, and the
   only way to get it is the digests differing (shapes 9 and 10 below);
9. one digest per arm with the arms at different ones: rc=1 with the two sets
   named;
10. one arm at two digests across its own runs: rc=1.

The statuses are the three `tools/model-guard.sh`,
`tools/qwen35_reference.py:411` and `tools/reconcile_snapshot.py` already use, and
the counting primitives are the `tinytitan_profile` ones AUD-273 added and AUD-274
reused rather than re-derived per driver:

    0  every published figure came from a run that measured it, and the gate cleared
    1  it measured, and something the page claims is contested -- with the claim named
    2  the page's headline could not be computed, with the reason named

Whether a failed speed gate should be a refusal (2) rather than a contested claim
(1) is the operator's open question from AUD-273; 1 follows the sibling that
already answers it, `tinytitan_mtp_b3_qualification.py:125`.
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
BENCH = ROOT / "benchmark"
sys.path.insert(0, str(BENCH))

import tinytitan_profile as prof  # noqa: E402
import tinytitan_sampler_ab as ab  # noqa: E402

GAP = "head_logits->embed"
ABSENT = object()
FULL = {
    "decode_tok_s": 4.0,
    "busy_per_token_ms": 50.0,
    "gaps": {GAP: {"per_token_ms": 1.0}},
    "completion_sha256": "aa",
    "completion_tokens": 64,
}
NO_BUSY = {k: v for k, v in FULL.items() if k != "busy_per_token_ms"}
NO_GAP = {**FULL, "gaps": {}}
NULL_BUSY = {**FULL, "busy_per_token_ms": None}
NULL_BUSY_CHANGES = {"busy_per_token_ms": None}
NO_BUSY_CHANGES = {"busy_per_token_ms": ABSENT}


def arm_rows(arm: str, count: int, **changes):
    """`count` runs of one arm. A change of `ABSENT` removes the key, which is how
    a footer line the server never printed is shaped."""
    row = dict(FULL)
    for key, value in changes.items():
        if value is ABSENT:
            row.pop(key, None)
        else:
            row[key] = value
    return [dict(row, arm=arm) for _ in range(count)]


def clean_rows(**arms):
    """Two arms of four runs each, the shape a real sweep publishes."""
    rows = []
    for arm in ("generic", "tiled"):
        rows.extend(arm_rows(arm, 4, **arms.get(arm, {})))
    return rows


def page(rows, gate=10.0):
    return ab.verdict(ab.summarize(rows), "4bit", gate)


class SamplerAbVerdict(unittest.TestCase):
    def test_a_clearing_sweep_is_zero(self):
        rows = clean_rows(tiled={"decode_tok_s": 5.0})
        lines, status = page(rows)
        out = "\n".join(lines)
        self.assertEqual(status, 0)
        self.assertIn("DELTA: +25.00%", out)
        self.assertIn("gate is +10%: PASS", out)
        self.assertIn("Output identical across every run of both arms: YES", out)

    def test_a_speed_gate_that_did_not_clear_is_contested_and_named(self):
        lines, status = page(clean_rows())
        out = "\n".join(lines)
        self.assertEqual(status, 1)
        self.assertIn("gate is +10%: FAIL", out)
        self.assertIn("gate failed: median delta +0.00% is below +10%", out)

    def test_a_metric_no_run_of_an_arm_logged_refuses_the_page(self):
        lines, status = page(clean_rows(generic=NO_BUSY_CHANGES, tiled=NO_BUSY_CHANGES))
        out = "\n".join(lines)
        self.assertEqual(status, 2)
        for arm in ("generic", "tiled"):
            self.assertIn(f"NOT MEASURED: the {arm} arm -- no run logged busy_per_token_ms", out)
        self.assertIn("busy/token not logged", out)
        self.assertNotIn("busy/token 0.00 ms", out)

    def test_a_median_padded_with_runs_that_never_answered_names_its_denominator(self):
        rows = []
        for arm in ("generic", "tiled"):
            rows.extend(arm_rows(arm, 2))
            rows.extend(arm_rows(arm, 2, **NO_BUSY_CHANGES))
        lines, status = page(rows)
        out = "\n".join(lines)
        self.assertEqual(status, 1)
        for arm in ("generic", "tiled"):
            self.assertIn(
                f"PARTIAL: the {arm} arm's busy_per_token_ms median is over 2 of 4 runs", out
            )
        # The published median is over the two runs that answered, not over the
        # two zeros `row.get(key, 0)` invented: 50.00, never the 25.00 measured.
        self.assertIn("busy/token 50.00 ms", out)
        self.assertNotIn("busy/token 25.00 ms", out)

    def test_the_sampler_gap_it_exists_to_compare_is_counted_the_same_way(self):
        lines, status = page(clean_rows(generic=NO_GAP, tiled=NO_GAP))
        out = "\n".join(lines)
        self.assertEqual(status, 2)
        for arm in ("generic", "tiled"):
            self.assertIn(f"NOT MEASURED: the {arm} arm -- no run logged {GAP} per_token_ms", out)
        self.assertIn("sampler gap not logged", out)
        self.assertNotIn("sampler gap 0.00 ms", out)

    def test_a_null_metric_is_not_a_measured_zero_and_does_not_end_the_page(self):
        lines, status = page(clean_rows(generic=NULL_BUSY_CHANGES, tiled=NULL_BUSY_CHANGES))
        self.assertEqual(status, 2)
        self.assertIn("no run logged busy_per_token_ms", "\n".join(lines))

    def test_a_zero_median_denominator_is_named_instead_of_divided_by(self):
        rows = clean_rows(generic={"decode_tok_s": 0.0}, tiled={"decode_tok_s": 4.0})
        lines, status = page(rows)
        out = "\n".join(lines)
        self.assertEqual(status, 2)
        self.assertIn(
            "NOT MEASURED: the generic arm's median is 0 tok/s, so no delta is computable", out
        )
        self.assertNotIn("DELTA:", out)

    def test_identical_digests_from_an_empty_stream_are_not_a_claim(self):
        rows = clean_rows(generic={"completion_tokens": 0}, tiled={"completion_tokens": 0})
        lines, status = page(rows)
        out = "\n".join(lines)
        self.assertEqual(status, 2)
        for arm in ("generic", "tiled"):
            self.assertIn(
                f"NOT MEASURED: the {arm} arm streamed no content, so its digest "
                "is the hash of nothing",
                out,
            )
        self.assertIn("Output identical across every run of both arms: NOT MEASURED", out)
        self.assertNotIn("both arms: YES", out)

    def test_digests_the_arms_disagree_on_are_contested_with_both_sets_named(self):
        rows = clean_rows(tiled={"completion_sha256": "bb"})
        lines, status = page(rows)
        out = "\n".join(lines)
        self.assertEqual(status, 1)
        self.assertIn("output differs: generic ['aa'], tiled ['bb']", out)

    def test_an_arm_that_repeated_a_digest_it_also_broke_is_contested(self):
        rows = arm_rows("generic", 3) + arm_rows("generic", 1, completion_sha256="cc")
        rows += arm_rows("tiled", 4)
        lines, status = page(rows)
        self.assertEqual(status, 1)
        self.assertIn("output differs", "\n".join(lines))

    def test_an_arm_with_no_runs_at_all_is_named(self):
        lines, status = page(arm_rows("generic", 4))
        out = "\n".join(lines)
        self.assertEqual(status, 2)
        self.assertIn("NOT MEASURED: the tiled arm has no runs", out)

    def test_no_run_at_all_is_refused_rather_than_printed_empty(self):
        lines, status = page([])
        out = "\n".join(lines)
        self.assertEqual(status, 2)
        self.assertIn("NOT MEASURED: no run at all", out)


class SamplerAbRun(unittest.TestCase):
    """The real `main()` over a stubbed `one_run`, so the sweep itself is pinned."""

    def drive(self, generic, tiled, argv):
        """Run the real `main()` with `one_run` answering per arm.

        Each `--pairs` block is `generic tiled tiled generic`, so a block count of
        2 hands each arm four measured runs after its discarded warmup; `generic`
        and `tiled` are that arm's four measured rows, or one row repeated. The
        warmup answers `FULL` for both arms because it is not a measured run --
        the driver never appends it.
        """

        def expand(rows):
            return list(rows) if isinstance(rows, list) else [rows] * 4

        queues = {"generic": expand(generic), "tiled": expand(tiled)}
        counted = {"generic": 0, "tiled": 0}
        calls = []

        def one_run(quant, arm, tag):
            calls.append((quant, arm, tag))
            if tag == "warmup":
                return {**FULL, "arm": arm}
            index = counted[arm]
            counted[arm] += 1
            return {**queues[arm][min(index, len(queues[arm]) - 1)], "arm": arm}

        tmp = pathlib.Path(tempfile.mkdtemp(prefix="aud275-"))
        buf = io.StringIO()
        argv = ["tinytitan_sampler_ab", "--quant", "4bit"] + argv
        with (
            mock.patch.object(ab, "one_run", one_run),
            mock.patch.object(ab, "ROOT", tmp),
            mock.patch.object(ab.g0, "preflight", return_value=None),
            mock.patch.object(ab.g0, "_terminate_all", return_value=None),
            mock.patch.object(sys, "argv", argv),
            contextlib.redirect_stdout(buf),
        ):
            status = ab.main()
        return status, buf.getvalue(), calls, tmp / ".build/benchmark-results/sampler-ab-4bit.json"

    def test_zero_pairs_refuses_with_a_page_instead_of_crashing(self):
        status, out, calls, artifact = self.drive(FULL, FULL, ["--pairs", "0"])
        self.assertEqual(status, 2)
        self.assertIn("NOT MEASURED: no run at all", out)
        self.assertIn("sweep status 2", out)
        self.assertEqual([tag for (_q, _a, tag) in calls], [])
        self.assertTrue(artifact.exists())
        self.assertEqual(json.loads(artifact.read_text())["status"], 2)

    def test_a_negative_pair_count_refuses_the_same_way(self):
        status, out, _calls, _artifact = self.drive(FULL, FULL, ["--pairs", "-1"])
        self.assertEqual(status, 2)
        self.assertIn("NOT MEASURED: no run at all", out)

    def test_a_progress_line_names_a_metric_the_run_never_answered(self):
        status, out, _calls, _artifact = self.drive(NO_BUSY, NO_BUSY, ["--pairs", "2"])
        self.assertEqual(status, 2)
        lines = [line for line in out.splitlines() if "tok/s  busy/token" in line]
        self.assertEqual(len(lines), 8)
        for line in lines:
            self.assertIn("busy/token not logged", line)
            self.assertNotIn("busy/token 0.00 ms", line)

    def test_a_null_metric_does_not_stop_the_sweep_halfway(self):
        status, _out, calls, _artifact = self.drive(NULL_BUSY, NULL_BUSY, ["--pairs", "2"])
        self.assertEqual(status, 2)
        self.assertEqual(len(calls), 10)

    def test_the_page_status_is_the_exit_status(self):
        status, out, _calls, artifact = self.drive(FULL, FULL, ["--pairs", "2"])
        self.assertEqual(status, 1)
        self.assertIn("gate is +10%: FAIL", out)
        self.assertIn("sweep status 1", out)
        self.assertEqual(json.loads(artifact.read_text())["status"], 1)

    def test_a_sweep_that_cleared_the_gate_returns_zero(self):
        fast = {**FULL, "decode_tok_s": 5.0}
        status, out, _calls, artifact = self.drive(FULL, fast, ["--pairs", "2"])
        self.assertEqual(status, 0)
        self.assertIn("DELTA: +25.00%", out)
        self.assertEqual(json.loads(artifact.read_text())["status"], 0)

    def test_the_artifact_publishes_no_median_the_runs_did_not_measure(self):
        _status, _out, _calls, artifact = self.drive(NO_BUSY, NO_BUSY, ["--pairs", "2"])
        arms = json.loads(artifact.read_text())["arms"]
        for arm in ("generic", "tiled"):
            self.assertIsNone(arms[arm]["median_busy_per_token_ms"])
            self.assertEqual(arms[arm]["counts"]["busy_per_token_ms"], [0, 4])
            self.assertEqual(arms[arm]["counts"]["decode_tok_s"], [4, 4])
            self.assertEqual(arms[arm]["median_tok_s"], 4.0)

    def test_the_artifact_median_is_over_the_runs_that_answered(self):
        rows = [FULL, FULL, NO_BUSY, NO_BUSY]
        _status, _out, _calls, artifact = self.drive(rows, rows, ["--pairs", "2"])
        arms = json.loads(artifact.read_text())["arms"]
        for arm in ("generic", "tiled"):
            self.assertEqual(arms[arm]["median_busy_per_token_ms"], 50.0)
            self.assertEqual(arms[arm]["counts"]["busy_per_token_ms"], [2, 4])


class ByteClaimReadsTheDriversDigestKey(unittest.TestCase):
    def test_a_digest_key_other_than_sha256_is_the_callers_to_name(self):
        rows = arm_rows("generic", 2) + arm_rows("tiled", 2)
        earned, identical, left, right = prof.byte_claim(
            rows, "generic", "tiled", digest_key="completion_sha256"
        )
        self.assertTrue(earned)
        self.assertTrue(identical)
        self.assertEqual(left, ["aa"])
        self.assertEqual(right, ["aa"])

    def test_the_default_key_still_reads_what_the_mtp_drivers_write(self):
        rows = [
            {"arm": "off", "sha256": "aa", "completion_tokens": 8},
            {"arm": "on", "sha256": "aa", "completion_tokens": 8},
        ]
        self.assertEqual(prof.byte_claim(rows)[1:3], (True, ["aa"]))


if __name__ == "__main__":
    unittest.main()
