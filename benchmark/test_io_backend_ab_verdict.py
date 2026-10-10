#!/usr/bin/env python3
"""Tests for `tinytitan_io_backend_ab.py`: AUD-285.

The driver answers one question: does the experimental Metal I/O backend produce
the same text as the bounded `F_NOCACHE` pread backend? Its docstring says that
answer "requires separate page-cache/memory-pressure validation", which is only
worth recording if the text comparison itself is sound.

Pre-fix it was not. Measured on this tree by calling the real `main()` with the
neighbour module's own leaves patched (the preflight, `run_case`, `ROOT`) -- no
model, no server, no build, no socket, nothing fetched:

1. every arm answering nothing: `passed: true`, exit 0;
2. both backends printing the same words while logging different completion
   lengths: `passed: true`, exit 0;
3. a footer reading `decode_tok_s=nan`: `passed: true`, exit 0. The rows carry
   the generation and runner footers `run_case` attached and the driver parses
   none of them, so the measurement the A/B exists for is unchecked;
4. a refused preflight or a server that died mid-sweep: a `RuntimeError`
   traceback and exit 1 -- the status a measured disagreement carries -- with no
   line naming what stopped it.

That is the AUD-280 set, in the driver AUD-280's sibling note called unique to
the hit/fixup A/B. It was not: this file compares raw `content` the same way and
was never brought onto the classifier that note pointed at. The fix threads the
arm identity through that classifier instead of writing a second one, so these
tests are also the proof that the parameterised parent still answers for
`barrier` against `hit-fixup` (`test_hit_fixup_ab_verdict.py`).

    0  every case compared, both backends answered, and every published figure
       came from a footer that carried it
    1  it measured, and the page is contested by a named disagreement or by cases
       it could not compare
    2  the measurement does not exist: the driver's own refusal, every case
       answered nothing, or no case published a usable figure -- reason printed,
       no artifact written
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
import tinytitan_io_backend_ab as ab  # noqa: E402

PROMPTS = ("short", "medium", "long")
WARMTH = ("cold", "warm")
BACKENDS = ("pread", "metal")
ANSWER = "a real paragraph of routed output."


def footer(tok="41.20", rate="0.7500"):
    return {
        "generation": f"TinyTitan generation decode_tok_s={tok} prefill_tok_s=1200.0",
        "runner": f"TinyTitan runner expert_hit_rate={rate} expert_hits=100 expert_misses=33",
    }


def row(backend, prompt, warmth, *, content=ANSWER, tokens=128, footers=None):
    footers = footers or footer()
    return {
        "mode": "hit-fixup",
        "prompt": prompt,
        "warmth": warmth,
        "io_backend": backend,
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


def sweep(**by_backend):
    """Every backend, prompt and warmth; each key names one arm's row overrides.

    An unknown key raises rather than being ignored: a misspelled arm name would
    otherwise hand the driver the unaltered sweep and report it as a finding.
    """
    unknown = set(by_backend) - set(BACKENDS)
    if unknown:
        raise AssertionError(f"sweep() override names no backend: {sorted(unknown)}")
    rows = []
    for backend in BACKENDS:
        for prompt in PROMPTS:
            for warmth in WARMTH:
                overrides = by_backend.get(backend, {})
                rows.append(row(backend, prompt, warmth, **overrides))
    return rows


def drive(rows, tmp, fail_on=None):
    """Run the driver's real `main()` over fixture rows. Returns (status, stdout, artifact).

    `fail_on` is a `(backend, prompt)` pair whose case raises the way a dead server
    or a footer-less log does, so the refusal path is driven rather than assumed.
    """
    nested = pathlib.Path(tmp)
    output = nested / ".build/benchmark-results/io-backend-ab.json"

    def run_case(mode, prompt_name, prompt, io_backend="pread", **kwargs):
        if fail_on == (io_backend, prompt_name):
            raise RuntimeError(f"missing benchmark footers in {io_backend}-{prompt_name}.log")
        picked = [r for r in rows if r["io_backend"] == io_backend and r["prompt"] == prompt_name]
        return picked, nested / f"{io_backend}-{prompt_name}.log"

    buffer = io.StringIO()
    with (
        mock.patch.object(sys, "argv", [str(ROOT / "benchmark" / "tinytitan_io_backend_ab.py")]),
        mock.patch.object(hf, "ROOT", nested),
        mock.patch.object(hf, "preflight", return_value={"model": "fixture"}),
        mock.patch.object(hf, "run_case", run_case),
        contextlib.redirect_stdout(buffer),
    ):
        status = ab.main()
    artifact = json.loads(output.read_text()) if output.exists() else None
    return status, buffer.getvalue(), artifact


class DeadSweep(unittest.TestCase):
    def test_a_sweep_where_nothing_answered_refuses_instead_of_passing(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, text, _ = drive(sweep(pread={"content": ""}, metal={"content": ""}), tmp)
        self.assertEqual(status, 2, text)
        self.assertIn("NOT MEASURED", text)
        self.assertIn("answered nothing", text)

    def test_a_refusal_writes_no_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            drive(sweep(pread={"content": ""}, metal={"content": ""}), tmp)
            self.assertFalse(
                (pathlib.Path(tmp) / ".build/benchmark-results/io-backend-ab.json").exists()
            )

    def test_a_footer_figure_that_is_not_a_number_is_not_a_pass(self):
        rows = sweep(metal={"footers": footer(tok="nan", rate="nan")})
        with tempfile.TemporaryDirectory() as tmp:
            status, text, _ = drive(rows, tmp)
        self.assertEqual(status, 2, text)
        self.assertIn("no case published a usable figure", text)
        self.assertIn("metal short/cold logged decode_tok_s=nan", text)

    def test_a_case_that_died_mid_sweep_names_the_cases_that_ran(self):
        rows = sweep()
        with tempfile.TemporaryDirectory() as tmp:
            status, text, artifact = drive(rows, tmp, fail_on=("metal", "medium"))
        self.assertEqual(status, 2, text)
        self.assertIn("NOT MEASURED: missing benchmark footers", text)
        self.assertIn("cases that ran before the refusal", text)
        self.assertIn("hit-fixup short/cold", text)
        self.assertIsNone(artifact)

    def test_a_refused_preflight_answers_with_a_status_not_a_traceback(self):
        buffer = io.StringIO()
        with (
            mock.patch.object(
                sys, "argv", [str(ROOT / "benchmark" / "tinytitan_io_backend_ab.py")]
            ),
            mock.patch.object(hf, "preflight", side_effect=RuntimeError("no model install")),
            contextlib.redirect_stdout(buffer),
        ):
            status = ab.main()
        text = buffer.getvalue()
        self.assertEqual(status, 2, text)
        self.assertIn("NOT MEASURED: no model install", text)


class ContestedSweep(unittest.TestCase):
    def test_arms_that_agree_on_words_but_not_on_length_are_contested(self):
        rows = sweep(metal={"tokens": 96})
        with tempfile.TemporaryDirectory() as tmp:
            status, text, artifact = drive(rows, tmp)
        self.assertEqual(status, 1, text)
        self.assertIn("completion length differs: 128 vs 96", text)
        self.assertFalse(artifact["passed"])
        self.assertEqual(artifact["status"], 1)

    def test_one_backend_answering_and_the_other_not_is_contested_by_name(self):
        rows = sweep(metal={"content": ""})
        with tempfile.TemporaryDirectory() as tmp:
            status, text, _ = drive(rows, tmp)
        self.assertEqual(status, 1, text)
        self.assertIn("metal short", text)
        self.assertIn("not comparable", text)

    def test_a_text_disagreement_still_names_the_case(self):
        rows = sweep(metal={"content": "a different paragraph entirely."})
        with tempfile.TemporaryDirectory() as tmp:
            status, text, artifact = drive(rows, tmp)
        self.assertEqual(status, 1, text)
        self.assertIn("short/cold text differs", text)
        self.assertEqual(
            sorted(artifact["response_mismatches"]),
            sorted(f"{p}/{w}" for p in PROMPTS for w in WARMTH),
        )


class LiveSweep(unittest.TestCase):
    def test_a_sweep_where_both_backends_answered_and_agree_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, text, artifact = drive(sweep(), tmp)
        self.assertEqual(status, 0, text)
        self.assertTrue(artifact["passed"])
        self.assertEqual(artifact["response_mismatches"], [])

    def test_the_page_prints_each_backend_s_measured_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, text, _ = drive(sweep(), tmp)
        self.assertEqual(status, 0)
        line = next(row for row in text.splitlines() if row.startswith("short cold"))
        self.assertEqual(line.count("41.20 tok/s"), 2, line)
        self.assertIn("pread", line)
        self.assertIn("metal", line)

    def test_the_artifact_keeps_every_row_and_both_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, _, artifact = drive(sweep(), tmp)
        self.assertEqual(status, 0)
        self.assertEqual(len(artifact["results"]), 12)
        self.assertEqual(
            sorted(artifact["logs"]),
            sorted(f"{b}/{p}" for b in BACKENDS for p in PROMPTS),
        )


if __name__ == "__main__":
    unittest.main()
