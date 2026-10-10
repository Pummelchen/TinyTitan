#!/usr/bin/env python3
"""Tests for `tinytitan_v42_control_plane_ab.py`: AUD-286.

The matrix asks whether five experimental expert control-plane configurations
produce the same text as the production one. Its own docstring says a case is not
promoted just because the script can run it, which only holds if the comparison
behind "same text" is sound.

Pre-fix it was not. Measured on this tree by calling the real `main()` with the
neighbour module's own leaves patched (the preflight, `run_case`, `ROOT`) -- no
model, no server, no build, no socket, nothing fetched:

1. every case answering nothing: `passed: true`, exit 0, and the same artifact
   written, so a matrix that proved nothing was recorded as a promotion;
2. cases agreeing on the words while logging different completion lengths:
   `passed: true`, exit 0;
3. a footer reading `decode_tok_s=nan`: `passed: true`, exit 0. The rows carry
   the footers `run_case` attached and the driver parses none of them, so the
   timings this matrix exists to compare are unchecked;
4. a refused preflight or a case that died mid-sweep: a `RuntimeError` traceback
   at exit 1 -- the status a measured disagreement carries -- and no line naming
   what stopped it;
5. `--case` with one name: no comparison is possible, yet `passed: true`, exit 0;
6. no page: the driver prints only the artifact path, so nothing on stdout says
   which case was compared against which.

That is the AUD-280 set again, one driver further along, and the third copy after
AUD-285. The fix runs the parent classifier once per (reference, case) pair
through its `arm_key`/`arms` parameters instead of writing a fourth verdict.

    0  every case compared against the reference, every one answered, and every
       published figure came from a footer that carried it
    1  it measured, and a named comparison is contested -- by a disagreement, by
       an arm that answered nothing, or by a pair that published no usable figure
    2  the measurement does not exist: the driver's own refusal, no case
       answered, no figure readable anywhere, or no second case to compare the
       reference against -- reason printed, no artifact written
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

import tinytitan_hit_fixup_ab as hf  # noqa: E402
import tinytitan_v42_control_plane_ab as ab  # noqa: E402

PROMPTS = ("short", "medium", "long")
WARMTH = ("cold", "warm")
CASES = tuple(name for name, *_ in ab.CASES)
REFERENCE = CASES[0]
ANSWER = "a real paragraph of routed output."
SCRIPT = str(ROOT / "benchmark" / "tinytitan_v42_control_plane_ab.py")
OUTPUT = ".build/benchmark-results/v4.2-control-plane-ab.json"


def footer(tok="41.20", rate="0.7500"):
    return {
        "generation": f"TinyTitan generation decode_tok_s={tok} prefill_tok_s=1200.0",
        "runner": f"TinyTitan runner expert_hit_rate={rate} expert_hits=100 expert_misses=33",
    }


def row(case, prompt, warmth, *, content=ANSWER, tokens=128, footers=None):
    config = next(item for item in ab.CASES if item[0] == case)
    footers = footers or footer()
    return {
        "mode": config[1],
        "case": case,
        "prompt": prompt,
        "warmth": warmth,
        "io_backend": config[2],
        "io_sync": config[3],
        "cache_layout": config[4],
        "cache_policy": config[5],
        "io_submission": config[6],
        "wall_seconds": 3.1,
        "content": content,
        "usage": {"completion_tokens": tokens},
        "generation_footer": footers["generation"],
        "runner_footer": footers["runner"],
    }


def sweep(**by_case):
    """Every case, prompt and warmth; each key names one case's row overrides.

    An unknown key raises rather than being ignored: a misspelled case name would
    otherwise hand the driver the unaltered sweep and report it as a finding.
    """
    unknown = set(by_case) - set(CASES)
    if unknown:
        raise AssertionError(f"sweep() override names no case: {sorted(unknown)}")
    rows = []
    for case in CASES:
        for prompt in PROMPTS:
            for warmth in WARMTH:
                rows.append(row(case, prompt, warmth, **by_case.get(case, {})))
    return rows


def drive(rows, tmp, fail_on=None, extra_argv=()):
    """Run the driver's real `main()` over fixture rows. Returns (status, stdout, artifact).

    `fail_on` is a `(case, prompt)` pair whose case raises the way a dead server or
    a footer-less log does, so the refusal path is driven rather than assumed. The
    fake `run_case` matches the configuration the driver actually sent and raises
    if none matches, so a dropped knob cannot be tested against the wrong case.
    """
    nested = pathlib.Path(tmp)

    def run_case(mode, prompt_name, prompt, **config):
        for name, case_mode, backend, sync, layout, policy, submission in ab.CASES:
            if (case_mode, backend, sync, layout, policy, submission) != (
                mode,
                config["io_backend"],
                config["io_sync"],
                config["cache_layout"],
                config["cache_policy"],
                config["io_submission"],
            ):
                continue
            if fail_on == (name, prompt_name):
                raise RuntimeError(f"missing benchmark footers in {name}-{prompt_name}.log")
            picked = [r for r in rows if r["case"] == name and r["prompt"] == prompt_name]
            return picked, nested / f"{name}-{prompt_name}.log"
        raise AssertionError(f"no case matches the configuration sent: {mode} {config}")

    buffer = io.StringIO()
    with (
        mock.patch.object(sys, "argv", [SCRIPT, *extra_argv]),
        mock.patch.object(hf, "ROOT", nested),
        mock.patch.object(hf, "preflight", return_value={"model": "fixture"}),
        mock.patch.object(hf, "run_case", run_case),
        contextlib.redirect_stdout(buffer),
    ):
        status = ab.main()
    artifact = json.loads((nested / OUTPUT).read_text()) if (nested / OUTPUT).exists() else None
    return status, buffer.getvalue(), artifact


class DeadMatrix(unittest.TestCase):
    def test_a_matrix_where_nothing_answered_refuses_instead_of_passing(self):
        rows = sweep(**{case: {"content": ""} for case in CASES})
        with tempfile.TemporaryDirectory() as tmp:
            status, text, _ = drive(rows, tmp)
        self.assertEqual(status, 2, text)
        self.assertIn("NOT MEASURED", text)

    def test_a_refusal_writes_no_artifact(self):
        rows = sweep(**{case: {"content": ""} for case in CASES})
        with tempfile.TemporaryDirectory() as tmp:
            drive(rows, tmp)
            self.assertFalse((pathlib.Path(tmp) / OUTPUT).exists())

    def test_a_case_whose_figures_are_not_numbers_is_not_a_pass(self):
        rows = sweep(**{"gpu-residency": {"footers": footer(tok="nan", rate="nan")}})
        with tempfile.TemporaryDirectory() as tmp:
            status, text, artifact = drive(rows, tmp)
        self.assertEqual(status, 1, text)
        self.assertIn("gpu-residency short/cold logged decode_tok_s=nan", text)
        self.assertIn("CONTESTED", text)
        self.assertFalse(artifact["passed"])

    def test_a_matrix_where_no_figure_is_readable_refuses(self):
        bad = footer(tok="nan", rate="nan")
        rows = sweep(**{case: {"footers": bad} for case in CASES})
        with tempfile.TemporaryDirectory() as tmp:
            status, text, artifact = drive(rows, tmp)
        self.assertEqual(status, 2, text)
        self.assertIn("no case published a usable figure", text)
        self.assertIsNone(artifact)

    def test_a_case_that_died_mid_matrix_names_the_cases_that_ran(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, text, artifact = drive(sweep(), tmp, fail_on=("event-pool", "medium"))
        self.assertEqual(status, 2, text)
        self.assertIn("NOT MEASURED: missing benchmark footers", text)
        self.assertIn("cases that ran before the refusal", text)
        self.assertIsNone(artifact)

    def test_a_refused_preflight_answers_with_a_status_not_a_traceback(self):
        buffer = io.StringIO()
        with (
            mock.patch.object(sys, "argv", [SCRIPT]),
            mock.patch.object(hf, "preflight", side_effect=RuntimeError("no model install")),
            contextlib.redirect_stdout(buffer),
        ):
            status = ab.main()
        text = buffer.getvalue()
        self.assertEqual(status, 2, text)
        self.assertIn("NOT MEASURED: no model install", text)

    def test_a_single_case_matrix_reports_it_compared_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, text, artifact = drive(sweep(), tmp, extra_argv=["--case", "gpu-residency"])
        self.assertEqual(status, 2, text)
        self.assertIn("no reference", text)
        self.assertIsNone(artifact)


class ContestedMatrix(unittest.TestCase):
    def test_cases_that_agree_on_words_but_not_on_length_are_contested(self):
        rows = sweep(**{"gpu-residency-aging": {"tokens": 96}})
        with tempfile.TemporaryDirectory() as tmp:
            status, text, artifact = drive(rows, tmp)
        self.assertEqual(status, 1, text)
        self.assertIn("completion length differs: 128 vs 96", text)
        self.assertEqual(
            sorted(artifact["response_mismatches"]),
            sorted(f"gpu-residency-aging {p}/{w}" for p in PROMPTS for w in WARMTH),
        )

    def test_a_case_disagreeing_on_text_is_contested_by_name(self):
        rows = sweep(**{"event-pread": {"content": "a different paragraph entirely."}})
        with tempfile.TemporaryDirectory() as tmp:
            status, text, artifact = drive(rows, tmp)
        self.assertEqual(status, 1, text)
        self.assertIn(f"{REFERENCE} vs event-pread", text)
        self.assertIn("short/cold text differs", text)
        self.assertEqual(
            sorted(artifact["response_mismatches"]),
            sorted(f"event-pread {p}/{w}" for p in PROMPTS for w in WARMTH),
        )
        self.assertEqual(artifact["status"], 1)

    def test_the_reference_answering_nothing_is_contested_against_every_case(self):
        rows = sweep(**{REFERENCE: {"content": ""}})
        with tempfile.TemporaryDirectory() as tmp:
            status, text, _ = drive(rows, tmp)
        self.assertEqual(status, 1, text)
        self.assertIn("not comparable", text)


class LiveMatrix(unittest.TestCase):
    def test_a_matrix_where_every_case_agrees_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, text, artifact = drive(sweep(), tmp)
        self.assertEqual(status, 0, text)
        self.assertTrue(artifact["passed"])
        self.assertEqual(artifact["response_mismatches"], [])

    def test_the_page_names_the_reference_and_the_case_it_was_compared_to(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, text, _ = drive(sweep(), tmp, extra_argv=["--prompt", "short"])
        self.assertEqual(status, 0)
        lines = text.splitlines()
        page = lines.index(f"{REFERENCE} vs event-pread") + 1
        line = next(row for row in lines[page:] if row.startswith("short cold"))
        self.assertEqual(line.count("41.20 tok/s"), 2, line)
        self.assertIn(REFERENCE, line)
        self.assertIn("event-pread", line)

    def test_a_selected_pair_uses_its_first_case_as_the_reference(self):
        """The reference is the first *selected* case, not the catalogue's first row.

        A control at HEAD: it exists to pin the re-rooting, so a reference read from
        CASES rather than cases dies here instead of surviving the sweep.
        """
        with tempfile.TemporaryDirectory() as tmp:
            status, text, artifact = drive(
                sweep(),
                tmp,
                extra_argv=["--case", "event-pool", "--case", "gpu-residency"],
            )
        self.assertEqual(status, 0, text)
        self.assertIn("event-pool vs gpu-residency", text)
        self.assertNotIn("production-deferred", text)
        self.assertEqual(artifact["reference"], "event-pool")

    def test_the_artifact_keeps_every_row_and_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, _, artifact = drive(sweep(), tmp)
        self.assertEqual(status, 0)
        self.assertEqual(len(artifact["results"]), 36)
        self.assertEqual(sorted(artifact["comparison_status"]), sorted(CASES[1:]))
        self.assertEqual(
            sorted(artifact["logs"]),
            sorted(f"{c}/{p}" for c in CASES for p in PROMPTS),
        )


if __name__ == "__main__":
    unittest.main()
