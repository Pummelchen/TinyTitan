"""The ANE-vs-GPU prefill harness: what it reads, and what it refuses to claim.

`benchmark/ane_prefill_ab_matrix.py` decides the number a release would quote,
so the two things that can make it lie are pinned here:

- the **footer parse** — prefill seconds come from the CLI's own footer, and a
  run that produced no footer must not be mistaken for a fast one;
- the **speedup** — an ANE arm that printed the runtime's GPU-fallback line did
  not use the ANE, so its time is a GPU time and no ratio may be reported from
  it. That is the check that would have caught the "3.8 was tested" illusion,
  where the switch was on and the model ran entirely on the GPU.

No model, no binary, no ANE.

Run from this directory, like the other benchmark tests:

    cd benchmark && python3 -m unittest test_ane_prefill_ab_matrix -v
"""

from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))

import ane_prefill_ab_matrix as ab  # noqa: E402


def arm(seconds: float, used_ane: bool = True, digest: str = "d") -> dict:
    return {
        "prefill_seconds": seconds,
        "prefill_tokens": 6027,
        "used_ane": used_ane,
        "response_sha256": digest,
        "response_head": "for",
        "exit": 0,
    }


def record(off: list[dict], on: list[dict]) -> dict:
    return {"model": "qwen3.5_2B_4Bit", "arms": {"off": off, "on": on}}


class ParseFooterTests(unittest.TestCase):
    def test_reads_prefill_seconds_and_tokens(self):
        footer = "[stop=maxTokens prefill=6027tok/74.25s new=1tok decode=0.00s tok/s=2109.811]"
        parsed = ab.parse_footer("loading\n" + footer + "\n")
        self.assertEqual(parsed["finish"], "maxTokens")  # the stop reason
        self.assertEqual(parsed["prefill_tokens"], 6027)
        self.assertAlmostEqual(parsed["prefill_seconds"], 74.25)

    def test_no_footer_is_none_rather_than_zero(self):
        self.assertIsNone(ab.parse_footer("error: something went wrong\n"))


class SummaryTests(unittest.TestCase):
    def test_median_prefill_and_speedup(self):
        summary = ab.summarize(record([arm(74.0), arm(74.4)], [arm(47.5), arm(47.7)]))
        self.assertAlmostEqual(summary["off"]["prefill_seconds_median"], 74.2)
        self.assertAlmostEqual(summary["on"]["prefill_seconds_median"], 47.6)
        self.assertAlmostEqual(summary["speedup"], 74.2 / 47.6, places=3)

    def test_an_arm_that_fell_back_reports_no_speedup(self):
        # The runtime said "using the GPU path", so the time is a GPU time.
        summary = ab.summarize(record([arm(74.0)], [arm(73.9, used_ane=False)]))
        self.assertFalse(summary["on"]["used_ane"])
        self.assertNotIn("speedup", summary)

    def test_a_mixed_arm_is_not_treated_as_using_the_ane(self):
        summary = ab.summarize(
            record([arm(74.0)], [arm(47.5, used_ane=True), arm(74.0, used_ane=False)])
        )
        self.assertFalse(summary["on"]["used_ane"])
        self.assertNotIn("speedup", summary)

    def test_a_model_the_ane_cannot_serve_keeps_its_gpu_time(self):
        # Any model without a usable sidecar — an install that has not been
        # exported yet, a refusal, or a family the exporter does not build for
        # — must still report its GPU number rather than no number at all.
        summary = ab.summarize(
            {
                "model": "qwen3.8-flash-next_125B_A6B_4Bit",
                "arms": {"off": [arm(200.0)], "on": []},
                "ane_unavailable": "no sidecar",
            }
        )
        self.assertAlmostEqual(summary["off"]["prefill_seconds_median"], 200.0)
        self.assertEqual(summary["ane_unavailable"], "no sidecar")
        self.assertNotIn("speedup", summary)

    def test_a_failed_measurement_is_reported_not_summarized(self):
        summary = ab.summarize(
            {"model": "m", "arms": {"off": [], "on": []}, "error": "off warm-up: no sidecar"}
        )
        self.assertIn("no sidecar", summary["error"])

    def test_differing_digests_are_kept_visible(self):
        # The ANE is not bit-identical to the GPU by construction; the summary
        # records both digests rather than pretending the arms agree.
        summary = ab.summarize(record([arm(74.0, digest="gpu")], [arm(47.5, digest="ane")]))
        self.assertEqual(summary["off"]["digests"], ["gpu"])
        self.assertEqual(summary["on"]["digests"], ["ane"])


class RecordTests(unittest.TestCase):
    """A held run must keep the rows it already earned, and resume cleanly."""

    def test_a_new_record_carries_what_a_reader_needs(self):
        record = ab.new_record(repeats=2, characters=23_000, max_new=1, chunk=1_024)
        self.assertEqual(record["repeats_per_arm"], 2)
        self.assertEqual(record["prompt_characters"], 23_000)
        self.assertEqual(record["max_new_tokens"], 1)
        self.assertEqual(record["prefill_chunk"], 1_024)  # the exported width
        self.assertEqual(record["results"], [])

    def test_a_prompt_under_one_chunk_reports_no_speedup(self):
        # Under 4,096 tokens the ANE cannot engage, so both arms are the GPU.
        # Reporting "1.0x" would read as a finding about the ANE.
        short = [dict(a, prefill_tokens=3_000) for a in [arm(50.0)]]
        fast = [dict(a, prefill_tokens=3_000) for a in [arm(20.0)]]
        summary = ab.summarize(record(short, fast))
        self.assertIn("prompt_too_short", summary)
        self.assertNotIn("speedup", summary)

    def test_the_default_prompt_clears_one_chunk(self):
        # The floor follows the configured chunk, and the shipped default must
        # clear it — otherwise the sweep measures two GPU arms.
        floor = int(ab.PREFILL_CHUNK * ab.CHARACTERS_PER_TOKEN)
        self.assertGreaterEqual(ab.PROMPT_CHARACTERS, floor)
        self.assertGreaterEqual(23_000, int(1_024 * ab.CHARACTERS_PER_TOKEN))

    def test_a_failed_row_is_not_counted_as_done(self):
        # A refusal must stay re-attemptable: treating it as done would keep
        # the failure forever.
        record = {
            "results": [
                {"model": "qwen3.5_2B_4Bit", "off": {}},
                {"model": "m", "error": "no sidecar"},
            ]
        }
        self.assertEqual(ab.stored_models(record), {"qwen3.5_2B_4Bit"})

    def test_storing_replaces_a_model_row_rather_than_duplicating_it(self):
        record = {"results": [{"model": "a", "marker": 1}]}
        ab.store_result(record, {"model": "a", "marker": 2})
        self.assertEqual([r["marker"] for r in record["results"]], [2])

    def test_storing_keeps_the_other_models(self):
        record = {"results": [{"model": "a"}]}
        ab.store_result(record, {"model": "b"})
        self.assertEqual([r["model"] for r in record["results"]], ["a", "b"])


class MeasuredZeroTests(unittest.TestCase):
    """AUD-283: a footer that logged 0 is a measurement, not an absence.

    Measured pre-fix: `summarize` tested both medians for truthiness, so an ANE
    arm whose runs logged `prefill=6027tok/0.00s` lost its `speedup` and its
    row printed `-` with an empty note -- the same blank cell the fallback row
    prints, so a timer that never started read as an arm that fell back. The
    prompt-length guard had the same shape on `prefill_tokens`: a run logging
    `0tok` skipped the `prompt_too_short` note the driver exists to give, and
    still published a speedup for a prompt the ANE cannot serve.
    """

    def test_a_zero_divisor_median_names_the_zero_instead_of_printing_a_blank(self):
        summary = ab.summarize(record([arm(74.0)], [arm(0.0)]))
        self.assertNotIn("speedup", summary)
        self.assertIn("median is 0 s", summary["no_speedup"])
        self.assertIn("median is 0 s", ab.format_row(summary))

    def test_a_zero_off_median_is_still_a_computable_speedup(self):
        # The divisor is the ANE arm; a GPU arm that logged 0.00 s gives 0.00x,
        # which is a figure the page should show rather than hide.
        summary = ab.summarize(record([arm(0.0)], [arm(47.0)]))
        self.assertAlmostEqual(summary["speedup"], 0.0)

    def test_a_prompt_that_logged_zero_tokens_is_still_under_one_chunk(self):
        short = [dict(arm(50.0), prefill_tokens=0), dict(arm(20.0), prefill_tokens=0)]
        summary = ab.summarize(record(short, short))
        self.assertIn("prompt_too_short", summary)
        self.assertNotIn("speedup", summary)


if __name__ == "__main__":
    unittest.main()
