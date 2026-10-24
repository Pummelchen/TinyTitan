"""Model-free tests for benchmark/tinytitan_bench_compare.py.

Fixtures are synthetic ``aggregate.json`` objects in a temporary directory. No
model, server, binary or benchmark process is started, and nothing asserts on
wall-clock time.
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

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "tinytitan_bench_compare", ROOT / "benchmark" / "tinytitan_bench_compare.py"
)
compare = importlib.util.module_from_spec(_spec)
# dataclasses resolves annotations through sys.modules[cls.__module__]; a module
# executed without being registered makes that lookup None on Python 3.13.
sys.modules[_spec.name] = compare
_spec.loader.exec_module(compare)

PROMPTS = [("Basic fact", 25), ("Arithmetic", 27), ("Logic", 30)]


def aggregate(
    config,
    *,
    prompts=None,
    avg=8.0,
    measured=True,
    provenance=None,
    repeats=None,
    total_requests=None,
    verified=None,
    avg_decode="default",
):
    rows = prompts if prompts is not None else PROMPTS
    results = [
        {"capability": name, "prompt_tokens": tokens, "verified": True} for name, tokens in rows
    ]
    if avg_decode == "default":
        decode_value = avg
    else:
        decode_value = avg_decode
    summary = {
        "prompts_configured": len(rows),
        "measured": measured,
        "total_requests": len(rows) if total_requests is None else total_requests,
        "avg_decode_tok_s": decode_value,
        "avg_footer_decode_tok_s": decode_value,
    }
    if verified is not None:
        summary["verified"] = verified
    if repeats is not None:
        summary["repeats"] = repeats
    data = {
        "config": config,
        "cache_mode": "multi-prefix",
        "mtp_config": "off",
        "runtime_profile": {"context_tokens": 262144, "kv_bits": 8, "thinking": "off"},
        "results": results,
        "summary": summary,
    }
    if provenance is not None:
        data["provenance"] = provenance
    return data


class CompareCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = pathlib.Path(self._tmp.name)

    def write(self, name, data):
        path = self.dir / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return str(path)

    def run_compare(self, *args):
        out = io.StringIO()
        err = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            status = compare.main([str(arg) for arg in args])
        return status, out.getvalue(), err.getvalue()

    def two_results(self, *, first_avg=8.0, second_avg=9.5, provenance=True):
        first_provenance = (
            {
                "git_commit": "abc1234",
                "chip": "Apple M3",
                "macos": "15.0",
                "flags": "cache=multi-prefix mtp=off",
                "repeats": 3,
                "model": "ornith-4bit",
            }
            if provenance
            else None
        )
        second_provenance = (
            {
                "git_commit": "abc1234",
                "chip": "Apple M3",
                "macos": "15.0",
                "flags": "cache=multi-prefix mtp=off",
                "repeats": 3,
                "model": "ornith-8bit",
            }
            if provenance
            else None
        )
        reference = self.write(
            "off.json",
            aggregate("cache_off_mtp_off_4bit", avg=first_avg, provenance=first_provenance),
        )
        candidate = self.write(
            "on.json",
            aggregate("cache_on_mtp_off_4bit", avg=second_avg, provenance=second_provenance),
        )
        return reference, candidate

    def assert_refused(self, status, out, err, needle):
        self.assertEqual(status, 2, out + err)
        self.assertIn("NOT MEASURED:", out)
        self.assertIn(needle, out)
        self.assertNotIn("Traceback", out + err)


class ReportTests(CompareCase):
    def test_markdown_table_lists_each_config_and_metric(self):
        reference, candidate = self.two_results()
        status, out, err = self.run_compare(reference, candidate)
        self.assertEqual(status, 0, out + err)
        self.assertIn("# TinyTitan bench comparison", out)
        self.assertIn("3 prompts, 82 prompt tokens", out)
        self.assertIn("| Config | Model | Cache | MTP | Prompts |", out)
        self.assertIn("cache_off_mtp_off_4bit", out)
        self.assertIn("cache_on_mtp_off_4bit", out)
        self.assertIn("8.00", out)
        self.assertIn("9.50", out)
        self.assertEqual(out.count("3/3"), 4)

    def test_json_report_carries_workload_and_provenance(self):
        reference, candidate = self.two_results()
        status, out, err = self.run_compare(reference, candidate, "--json")
        self.assertEqual(status, 0, out + err)
        payload = json.loads(out)
        self.assertTrue(payload["complete"])
        self.assertEqual(payload["workload"]["prompts"], ["Basic fact", "Arithmetic", "Logic"])
        self.assertEqual(payload["workload"]["prompt_tokens_total"], 82)
        self.assertEqual(len(payload["results"]), 2)
        provenance = payload["results"][0]["provenance"]
        self.assertEqual(provenance["git_commit"], "abc1234")
        self.assertEqual(provenance["chip"], "Apple M3")
        self.assertEqual(provenance["macos"], "15.0")
        self.assertEqual(provenance["repeats"], "3")
        self.assertEqual(provenance["model"], "ornith-4bit")
        self.assertIn("cache=multi-prefix", provenance["flags"])

    def test_absent_provenance_is_not_invented(self):
        reference, candidate = self.two_results(provenance=False)
        status, out, err = self.run_compare(reference, candidate)
        self.assertEqual(status, 0, out + err)
        self.assertIn("not recorded", out)
        status, out, _ = self.run_compare(reference, candidate, "--json")
        provenance = json.loads(out)["results"][0]["provenance"]
        self.assertIsNone(provenance["git_commit"])
        self.assertIsNone(provenance["chip"])
        self.assertIsNone(provenance["macos"])
        self.assertIsNone(provenance["repeats"])
        self.assertIn("cache=multi-prefix", provenance["flags"])
        self.assertIn("kv_bits=8", provenance["flags"])

    def test_legacy_top_level_provenance_keys_are_read(self):
        first = aggregate("first")
        first.update({"commit": "def5678", "chip": "Apple M2", "macos": "14.6"})
        second = aggregate("second", avg=9.0)
        status, out, err = self.run_compare(
            self.write("first.json", first), self.write("second.json", second)
        )
        self.assertEqual(status, 0, out + err)
        self.assertIn("def5678", out)
        self.assertIn("Apple M2", out)
        self.assertIn("14.6", out)

    def test_legacy_results_without_measured_or_configured_are_comparable(self):
        first = aggregate("first")
        second = aggregate("second", avg=9.0)
        for data in (first, second):
            data["summary"].pop("measured")
            data["summary"].pop("prompts_configured")
        status, out, err = self.run_compare(
            self.write("first.json", first), self.write("second.json", second)
        )
        self.assertEqual(status, 0, out + err)
        self.assertIn("Notes:", out)
        self.assertIn("prompts_configured not recorded", out)
        self.assertIn("measured not recorded", out)
        status, out, _ = self.run_compare(
            self.write("first.json", first), self.write("second.json", second), "--json"
        )
        self.assertEqual(status, 0, out)
        payload = json.loads(out)
        self.assertTrue(payload["complete"])
        self.assertIn("prompts_configured not recorded", payload["results"][0]["notes"][0])

    def test_legacy_run_that_cannot_establish_completeness_exits_one(self):
        first = aggregate("first")
        second = aggregate("second", avg=9.0)
        for data in (first, second):
            data["summary"].pop("measured")
            data["summary"].pop("prompts_configured")
        second["summary"]["total_requests"] = 2
        status, out, err = self.run_compare(
            self.write("first.json", first), self.write("second.json", second)
        )
        self.assertEqual(status, 1, out + err)
        self.assertIn("do not establish a complete run", out)

    def test_json_output_has_no_markdown(self):
        reference, candidate = self.two_results()
        status, out, _ = self.run_compare(reference, candidate, "--json")
        self.assertEqual(status, 0, out)
        self.assertTrue(out.lstrip().startswith("{"))
        self.assertNotIn("# TinyTitan", out)

    def test_the_table_is_stable_across_runs(self):
        reference, candidate = self.two_results()
        first = self.run_compare(reference, candidate)
        second = self.run_compare(reference, candidate)
        self.assertEqual(first[0], 0)
        self.assertEqual(first[1], second[1])
        self.assertEqual(first[2], second[2])


class InputTests(CompareCase):
    def test_missing_file_refuses_with_a_named_reason(self):
        reference, _ = self.two_results()
        status, out, err = self.run_compare(reference, self.dir / "missing.json")
        self.assert_refused(status, out, err, "does not exist")

    def test_malformed_json_refuses(self):
        reference, _ = self.two_results()
        broken = self.dir / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        status, out, err = self.run_compare(reference, broken)
        self.assert_refused(status, out, err, "not valid JSON")

    def test_huge_average_is_refused_without_a_traceback(self):
        reference = self.write("ref.json", aggregate("ref"))
        huge = aggregate("huge", avg=10**400)
        status, out, err = self.run_compare(reference, self.write("huge.json", huge))
        self.assert_refused(status, out, err, "too large")

    def test_non_object_json_refuses(self):
        reference, _ = self.two_results()
        status, out, err = self.run_compare(reference, self.write("list.json", []))
        self.assert_refused(status, out, err, "not a JSON object")

    def test_missing_schema_field_refuses(self):
        reference, _ = self.two_results()
        data = aggregate("no-results")
        data.pop("results")
        status, out, err = self.run_compare(reference, self.write("no-results.json", data))
        self.assert_refused(status, out, err, "results")

    def test_one_result_is_not_a_comparison(self):
        reference, _ = self.two_results()
        status, out, err = self.run_compare(reference)
        self.assert_refused(status, out, err, "at least two")

    def test_directory_holding_aggregate_json_is_accepted(self):
        reference, _ = self.two_results()
        directory = self.dir / "second"
        directory.mkdir()
        (directory / "aggregate.json").write_text(
            json.dumps(aggregate("second", avg=9.0)), encoding="utf-8"
        )
        status, out, err = self.run_compare(reference, directory)
        self.assertEqual(status, 0, out + err)
        self.assertIn("second", out)


class WorkloadTests(CompareCase):
    def test_different_prompt_sets_refuse(self):
        reference = self.write("ref.json", aggregate("ref"))
        candidate = self.write(
            "cand.json",
            aggregate(
                "cand",
                prompts=[("Basic fact", 25), ("Arithmetic", 27), ("Extraction", 30)],
            ),
        )
        status, out, err = self.run_compare(reference, candidate)
        self.assert_refused(status, out, err, "same workload")
        self.assertIn("missing Logic", out)
        self.assertIn("extra Extraction", out)

    def test_different_prompt_token_counts_refuse(self):
        reference = self.write("ref.json", aggregate("ref"))
        candidate = self.write(
            "cand.json",
            aggregate(
                "cand",
                prompts=[("Basic fact", 25), ("Arithmetic", 27), ("Logic", 31)],
            ),
        )
        status, out, err = self.run_compare(reference, candidate)
        self.assert_refused(status, out, err, "prompt tokens")
        self.assertIn("Logic", out)

    def test_different_configured_counts_refuse(self):
        reference = self.write("ref.json", aggregate("ref"))
        candidate_data = aggregate("cand", prompts=[("Basic fact", 25), ("Arithmetic", 27)])
        candidate = self.write("cand.json", candidate_data)
        status, out, err = self.run_compare(reference, candidate)
        self.assert_refused(status, out, err, "configured 2 prompts")

    def test_repeated_prompt_in_one_result_refuses(self):
        reference = self.write("ref.json", aggregate("ref"))
        data = aggregate("cand")
        data["results"].append({"capability": "Logic", "prompt_tokens": 30})
        status, out, err = self.run_compare(reference, self.write("cand.json", data))
        self.assert_refused(status, out, err, "repeats prompt")


class IncompleteTests(CompareCase):
    def test_unmeasured_run_reports_and_exits_one(self):
        reference = self.write("ref.json", aggregate("ref"))
        candidate = self.write("cand.json", aggregate("cand", measured=False))
        status, out, err = self.run_compare(reference, candidate)
        self.assertEqual(status, 1, out + err)
        self.assertIn("# TinyTitan bench comparison", out)
        self.assertIn("Incomplete runs:", out)
        self.assertIn("measured=false", out)
        status, out, _ = self.run_compare(reference, candidate, "--json")
        payload = json.loads(out)
        self.assertFalse(payload["complete"])
        self.assertFalse(payload["results"][1]["complete"])
        self.assertEqual(payload["results"][1]["incomplete"], ["the run reports measured=false"])

    def test_total_requests_mismatch_exits_one(self):
        reference = self.write("ref.json", aggregate("ref"))
        candidate = self.write("cand.json", aggregate("cand", total_requests=2, avg=9.0))
        status, out, err = self.run_compare(reference, candidate)
        self.assertEqual(status, 1, out + err)
        self.assertIn("total_requests=2 but 3 rows recorded", out)

    def test_missing_average_exits_one_and_prints_na(self):
        reference = self.write("ref.json", aggregate("ref"))
        candidate = self.write("cand.json", aggregate("cand", measured=True, avg_decode=None))
        status, out, err = self.run_compare(reference, candidate)
        self.assertEqual(status, 1, out + err)
        self.assertIn("no average decode rate recorded", out)
        self.assertIn("n/a", out)

    def test_verified_is_derived_from_rows_when_summary_omits_it(self):
        reference = self.write("ref.json", aggregate("ref"))
        data = aggregate("cand", avg=9.0)
        data["results"][2]["verified"] = False
        candidate = self.write("cand.json", data)
        status, out, err = self.run_compare(reference, candidate)
        self.assertEqual(status, 0, out + err)
        self.assertIn("| 2/3 |", out)


if __name__ == "__main__":
    unittest.main()
