"""The internal-speed record and its release comparison.

`tools/internal-speeds.py` writes one JSON record per release and diffs it
against a previous one; that diff is the release gate for engine speed. These
tests pin the two things that decide whether the gate is trustworthy:

- **which record it compares against.** Records are per model, and only a
  qwen36 install can carry an ANE sidecar, so the newest file on disk is not
  necessarily a comparable one. A 125B MoE's decode rate must never become the
  4B's baseline by being newest.
- **what a missing ANE number says.** "This model is not qwen36" is only true
  of a dense install; a qwen36 install that simply has no sidecar yet is a
  different situation and must not read alike.

They need no model, no built binary and no ANE sidecar.

Run from this directory, like the other benchmark tests:

    cd benchmark && python3 -m unittest test_internal_speeds -v
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import pathlib
import shutil
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_tool():
    """`tools/internal-speeds.py` is not importable by name (hyphen)."""
    spec = importlib.util.spec_from_file_location(
        "internal_speeds", ROOT / "tools" / "internal-speeds.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


internal_speeds = load_tool()


def record(model: str, prompt: str = internal_speeds.DEFAULT_PROMPT) -> dict:
    return {
        "model": {"path": model, "prompt": prompt},
        "quality": {"keyword_coverage": 0.5, "response_sha256": "abc"},
    }


class AnePromptTests(unittest.TestCase):
    def test_the_prompt_is_long_enough_for_more_than_one_ane_chunk(self):
        # ~4 characters per token is the pessimistic end for English; the
        # measurement must fill at least one 4,096-token chunk.
        prompt = internal_speeds.ane_prompt()
        self.assertGreaterEqual(len(prompt), 20_000)
        self.assertTrue(prompt.startswith(internal_speeds.ANE_PROMPT_SENTENCE))
        self.assertEqual(prompt, internal_speeds.ane_prompt())  # deterministic

    def test_a_fallback_line_is_reported_as_not_using_the_ane(self):
        used, fallback = internal_speeds.ane_usage(
            "loading model\n"
            "TinyTitan ane-prefill fallback: chunk at 0 (+8) outside sidecar "
            "coverage (chunk 4096, max prompt 16384); using the GPU path\n"
            "[stop=eos prefill=8tok/0.21s new=4tok decode=0.3s tok/s=13.3]\n"
        )
        self.assertFalse(used)
        self.assertIn("using the GPU path", fallback)

    def test_a_clean_run_is_reported_as_using_the_ane(self):
        used, fallback = internal_speeds.ane_usage(
            "loading model\n[stop=eos prefill=8192tok/3.10s new=1tok decode=0.1s tok/s=10.0]\n"
        )
        self.assertTrue(used)
        self.assertIsNone(fallback)


class AneCacheTests(unittest.TestCase):
    """A cold ANE compile cache makes the first measured run incomparable."""

    def setUp(self):
        self.dir = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.real_root = internal_speeds.ROOT
        internal_speeds.ROOT = self.dir
        self.addCleanup(setattr, internal_speeds, "ROOT", self.real_root)
        self.sidecar = self.dir / "models" / "m" / "ane_prefill"
        self.sidecar.mkdir(parents=True)

    def test_no_compiled_directory_is_cold(self):
        self.assertFalse(internal_speeds.ane_cache_is_warm("models/m"))

    def test_any_compiled_version_counts_as_warm(self):
        (self.sidecar / "compiled-v1").mkdir()
        self.assertTrue(internal_speeds.ane_cache_is_warm("models/m"))

    def test_a_missing_sidecar_is_cold_rather_than_an_error(self):
        self.assertFalse(internal_speeds.ane_cache_is_warm("models/absent"))


class MissingAneReasonTests(unittest.TestCase):
    def test_qwen36_without_a_sidecar_is_not_called_another_family(self):
        reason = internal_speeds.missing_ane_reason("models/qwen-agentworld_35B_A3B_4Bit", "qwen36")
        self.assertIn("qwen-agentworld_35B_A3B_4Bit", reason)
        self.assertIn("export_ane_prefill.py", reason)
        # The exporter *can* serve this model; only the sidecar is absent.
        self.assertNotIn("supports the qwen36 family only", reason)

    def test_a_dense_model_is_told_to_export_a_sidecar(self):
        # Since the exporter reads its geometry from the manifest, the dense
        # family is servable — the absence is a missing sidecar, not a family
        # the exporter cannot describe.
        reason = internal_speeds.missing_ane_reason("models/qwen3.5_4B_4Bit", "qwen3_5_dense")
        self.assertIn("export_ane_prefill.py", reason)
        self.assertNotIn("no graph for", reason)

    def test_qwen38_is_told_the_ane_measured_slower(self):
        # The family became servable when the runtime learned to fold the QSA
        # selection into the mask, and then measured: the ANE loses, so the
        # record says so rather than telling the operator to export one.
        reason = internal_speeds.missing_ane_reason(
            "models/qwen3.8-flash-next_125B_A6B_4Bit", "qwen38flash"
        )
        self.assertIn("does not pay", reason)
        self.assertIn("0.72x", reason)
        # Not a missing sidecar: one would make the default path slower.
        self.assertNotIn("export one", reason)

    def test_the_mtp_draft_is_not_prefilled_on_the_ane(self):
        reason = internal_speeds.missing_ane_reason(
            "models/qwen3.8-flash-next_125B_A6B_MTP_4Bit", "qwen38flash_mtp"
        )
        self.assertIn("MTP draft", reason)
        # Not a missing sidecar: one would never be loaded.
        self.assertNotIn("export one", reason)


class NewestBaselineTests(unittest.TestCase):
    def setUp(self):
        self.dir = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.real_results = internal_speeds.RESULTS
        internal_speeds.RESULTS = self.dir
        self.addCleanup(setattr, internal_speeds, "RESULTS", self.real_results)

    def write(self, name: str, payload: dict) -> pathlib.Path:
        path = self.dir / name
        path.write_text(json.dumps(payload))
        return path

    def test_picks_the_newest_record_for_the_same_model_and_prompt(self):
        self.write("a-4b.json", record("models/qwen3.5_4B_4Bit"))
        newest = self.write("b-4b.json", record("models/qwen3.5_4B_4Bit"))
        # Newer, but a different model: never the 4B's baseline.
        self.write("c-moe.json", record("models/qwen-agentworld_35B_A3B_4Bit"))
        out = self.dir / "d-4b.json"

        self.assertEqual(
            internal_speeds.newest_baseline(
                out, "models/qwen3.5_4B_4Bit", internal_speeds.DEFAULT_PROMPT
            ),
            str(newest),
        )

    def test_the_record_being_written_is_never_its_own_baseline(self):
        path = self.write("only.json", record("models/qwen3.5_4B_4Bit"))
        self.assertIsNone(
            internal_speeds.newest_baseline(
                path, "models/qwen3.5_4B_4Bit", internal_speeds.DEFAULT_PROMPT
            )
        )

    def test_a_different_prompt_is_not_comparable(self):
        self.write("other-prompt.json", record("models/qwen3.5_4B_4Bit", "explain quicksort"))
        self.assertIsNone(
            internal_speeds.newest_baseline(
                self.dir / "new.json", "models/qwen3.5_4B_4Bit", internal_speeds.DEFAULT_PROMPT
            )
        )

    def test_no_previous_record_at_all(self):
        self.assertIsNone(
            internal_speeds.newest_baseline(
                self.dir / "new.json", "models/qwen3.5_4B_4Bit", internal_speeds.DEFAULT_PROMPT
            )
        )

    def test_an_unreadable_record_is_skipped_not_fatal(self):
        (self.dir / "broken.json").write_text("{not json")
        good = self.write("good.json", record("models/qwen3.5_4B_4Bit"))
        self.assertEqual(
            internal_speeds.newest_baseline(
                self.dir / "new.json", "models/qwen3.5_4B_4Bit", internal_speeds.DEFAULT_PROMPT
            ),
            str(good),
        )


class ModelFamilyTests(unittest.TestCase):
    def setUp(self):
        self.dir = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.real_root = internal_speeds.ROOT
        internal_speeds.ROOT = self.dir
        self.addCleanup(setattr, internal_speeds, "ROOT", self.real_root)

    def test_reads_the_family_from_the_manifest(self):
        model = self.dir / "models" / "m"
        model.mkdir(parents=True)
        (model / "manifest.json").write_text(json.dumps({"arch": {"family": "qwen36"}}))
        self.assertEqual(internal_speeds.model_family("models/m"), "qwen36")

    def test_a_missing_or_malformed_manifest_is_none(self):
        model = self.dir / "models" / "m"
        model.mkdir(parents=True)
        self.assertIsNone(internal_speeds.model_family("models/m"))
        (model / "manifest.json").write_text("{not json")
        self.assertIsNone(internal_speeds.model_family("models/m"))


class ModelTotalBytesTests(unittest.TestCase):
    """A MoE keeps most of its bytes outside `model_weights.bin`."""

    def setUp(self):
        self.dir = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.real_root = internal_speeds.ROOT
        internal_speeds.ROOT = self.dir
        self.addCleanup(setattr, internal_speeds, "ROOT", self.real_root)
        self.model = self.dir / "models" / "m"
        self.model.mkdir(parents=True)

    def declare(self, files: dict) -> None:
        (self.model / "manifest.json").write_text(json.dumps({"files": files}))

    def test_sums_every_declared_file_including_the_packed_experts(self):
        self.declare(
            {"model_weights.bin": {"size": 1_000}, "packed_experts/layer_00.bin": {"size": 9_000}}
        )
        self.assertEqual(internal_speeds.model_total_bytes("models/m"), 10_000)

    def test_falls_back_to_the_resident_weights_without_a_usable_manifest(self):
        (self.model / "model_weights.bin").write_bytes(b"x" * 512)
        self.assertEqual(internal_speeds.model_total_bytes("models/m"), 512)
        self.declare({})
        self.assertEqual(internal_speeds.model_total_bytes("models/m"), 512)

    def test_a_declared_zero_total_falls_back_rather_than_reporting_zero(self):
        (self.model / "model_weights.bin").write_bytes(b"x" * 64)
        self.declare({"model_weights.bin": {"size": 0}})
        self.assertEqual(internal_speeds.model_total_bytes("models/m"), 64)

    def test_nothing_to_measure_is_zero(self):
        self.assertEqual(internal_speeds.model_total_bytes("models/m"), 0)


class DefaultLabelTests(unittest.TestCase):
    """Two models recorded at one commit must not resolve to one file."""

    def test_the_default_model_keeps_the_bare_describe(self):
        self.assertEqual(
            internal_speeds.default_label("v5.6-3-gabc", internal_speeds.DEFAULT_MODEL),
            "v5.6-3-gabc",
        )

    def test_another_model_is_suffixed_so_it_cannot_clobber_the_baseline(self):
        label = internal_speeds.default_label("v5.6-3-gabc", "models/qwen-agentworld_35B_A3B_4Bit")
        self.assertEqual(label, "v5.6-3-gabc-qwen-agentworld_35B_A3B_4Bit")
        self.assertNotEqual(
            label, internal_speeds.default_label("v5.6-3-gabc", internal_speeds.DEFAULT_MODEL)
        )


class CompareTests(unittest.TestCase):
    def candidate(self, **overrides) -> dict:
        base = {
            "generation": {
                "prefill_tokens_per_second": 100.0,
                "decode_tokens_per_second": 50.0,
                "effective_decode_gbps": 60.0,
                "ttft_seconds": 1.0,
                "decode_seconds": 5.0,
                "total_seconds": 6.0,
            },
            "gpu": {"qkv_gemv_gbps": 70.0, "routed_moe_gbps": 44.0, "gdn_inproj_gbps": 77.0},
            "cpu": {"best_gbps": 43.0},
            "quality": {"keyword_coverage": 0.5, "response_sha256": "abc"},
        }
        for dotted, value in overrides.items():
            section, field = dotted.split("__")
            base[section][field] = value
        return base

    def test_an_identical_record_passes(self):
        self.assertTrue(internal_speeds.compare(self.candidate(), self.candidate(), 10.0))

    def test_a_small_dip_stays_within_the_threshold(self):
        self.assertTrue(
            internal_speeds.compare(
                self.candidate(), self.candidate(generation__decode_tokens_per_second=46.0), 10.0
            )
        )

    def test_a_bandwidth_regression_fails(self):
        self.assertFalse(
            internal_speeds.compare(
                self.candidate(), self.candidate(gpu__routed_moe_gbps=30.0), 10.0
            )
        )

    def test_a_latency_regression_fails(self):
        self.assertFalse(
            internal_speeds.compare(
                self.candidate(), self.candidate(generation__ttft_seconds=1.5), 10.0
            )
        )

    def test_a_quality_drop_fails(self):
        self.assertFalse(
            internal_speeds.compare(
                self.candidate(), self.candidate(quality__keyword_coverage=0.2), 10.0
            )
        )

    def test_a_record_whose_probes_all_failed_is_refused(self):
        """A candidate in which nothing was measured is not a candidate that
        regressed in nothing."""
        broken = {
            "model": {"path": "models/qwen3.5_4B_4Bit", "prompt": "x"},
            "environment": {"git_describe": "v5.18"},
            "gpu": {"error": "qkv_gemv: exit 1", "output": "…"},
            "cpu": {"error": "cpugemv: exit 1", "output": "…"},
            "generation": {"error": "cli exit 1", "stderr": "…"},
        }
        self.assertFalse(internal_speeds.compare(self.candidate(), broken, 10.0))

    def test_the_number_compared_is_printed(self):
        """The table is the only evidence a reader has that a comparison ran, so
        the count has to be in the output rather than inferred from row height."""
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            internal_speeds.compare(self.candidate(), self.candidate(), 10.0)
        printed = buffer.getvalue()
        self.assertIn("10 metric(s) compared", printed)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            internal_speeds.compare(
                self.candidate(),
                {
                    "model": {"path": "models/qwen3.5_4B_4Bit", "prompt": "x"},
                    "generation": {"error": "cli exit 1"},
                },
                10.0,
            )
        self.assertIn("0 metric(s) compared", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
