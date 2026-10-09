#!/usr/bin/env python3
"""Tests for benchmark/tinytitan_ane_attention_probe.py, driven Core ML-free.

The probe answers a go/no-go -- "is the full-attention block fast enough on the
Neural Engine to be worth integrating?" -- and it already was:
`docs/v4.4-decode-width-plan.md` records 26.7x and `docs/v4.5-ane-prefill.md`
shipped the opt-in. Every number that decision turned on was printed by a file
that cannot tell a measurement from a failure. These tests pin what the pre-fix
driver did, plus the geometry the verdict rests on.

1. The deciding comparison was never made. The docstring sets the bar ("the ANE
   must beat that per layer-chunk by enough") and the footer prints
   `4,215 ms per layer-chunk`, but the rows are per-chunk times at whatever
   `--configs` says, and 4,215 ms is `84.3 s / 20 layer-chunks` averaged over a
   6,103-token prefill whose two chunks are 4096 and 2007 tokens -- unequal
   shapes, quadratic cost. Measured on this M3 at chunk 512 the pre-fix file
   prints `7.55 ms` directly above that footer; any ratio a reader forms there is
   off by the chunk-size difference. Both sides are now normalized to ms per 1k
   layer-tokens from the recorded measurement
   (84.3 s / (6,103 tokens x 10 layers) = 1,381 ms per 1k layer-tokens), the
   ratio is printed per row, and the history caveat is named.
2. The numerics "sanity check" has no criterion and no status. The shipped
   default run on this machine reports `mean rel err 0.0998` for 2048:4096 and
   exits 0 -- the plan explains that growth (uniform random scores at total
   length 6,144) -- but the file calls it a check, and a check that cannot fail
   is a print. Non-finite is worse: `max(nan, 1e-9)` is `nan`, so a block output
   that goes NaN -- the exact ANE failure mode the docstring says was routed
   around -- prints `mean rel err nan`, writes a literal `NaN` into the artifact
   (not valid JSON), and exits 0.
3. The CPU_ONLY arm skips itself silently. `if t <= 2048:` at :317 means the
   default list's `4096:0` row prints no comparison and nothing explains the
   absence (measured: `CPU_AND_NE 214.18 ms/chunk-layer   mean rel err 0.0292`),
   and the artifact's `cpu_only_ms: null` cannot be told apart from an arm that
   errored. `round(cpu_ms, 2) if cpu_ms else None` at :336 reports a measured
   0.0 ms as absent.
4. A configured run crashes instead of refusing. `--repeats 0` builds the model,
   discards two warm-ups over zero timed iterations and dies at :309 with
   `IndexError: list index out of range` (measured); a malformed `--configs` is a
   bare `ValueError` naming neither the flag nor the value.
5. One errored config loses the whole run. The artifact is written after the loop
   at :352, so a `predict` that raises on the last of four configs leaves no JSON
   at all -- including the three rows already measured.
6. The artifact path is relative to the working directory with no `mkdir`, so the
   shipped command run from anywhere else measures everything and then dies at
   :352 with `FileNotFoundError: '.build/benchmark-results/ane-attention-probe.json'`
   (measured from `/tmp`, exit 1, no artifact, after the run spent its time).
7. `def main() -> int` at :263 ends on `return 0` (:355) as its only return and
   the guard at :359 discards it, so none of the above can cost the run anything.

No Core ML program is built by these tests: `coremltools` is faked in
`sys.modules` before the driver is imported, and `build_block` / `cpu_model_for`
are patched, so the parsing, the arithmetic, the accounting and the status are
exercised against the real driver code while `predict` is a numpy call. The pure
geometry (`rope_tables`, `causal_mask`, the packed QKV interleave) is asserted
against real numpy, because a mask or a position offset that does not match the
engine's makes the probe time a different block than the one that shipped.
"""

import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmark"
sys.path.insert(0, str(BENCH))

FAKED = [
    "coremltools",
    "coremltools.converters",
    "coremltools.converters.mil",
    "coremltools.converters.mil.mil",
    "coremltools.converters.mil.mil.types",
]

# The driver imports `coremltools` and `coremltools.converters.mil.Builder`. A
# MagicMock answers both: `mb.program(...)` is a decorator that returns a mock,
# and every `mb.*` op the MIL body calls returns one. Nothing below the module
# scope runs in these tests, because `build_block` is patched.
for _name in FAKED:
    sys.modules.setdefault(_name, mock.MagicMock(name=_name))

import tinytitan_ane_attention_probe as probe  # noqa: E402

D = probe.D
HEAD_DIM = probe.HEAD_DIM


class FakeModel:
    """A converted program: named outputs with shapes, and a `predict` that answers.

    With no `array` the block output echoes the hidden state it was fed, so the
    numerics check against the echoed reference is exactly 0 and a test only has
    to supply an array to make the gap nonzero or non-finite.
    """

    def __init__(self, out_name, array=None, shape=None, extra=(), fail_after=None):
        self.out_name = out_name
        self.array = array
        self.shape = tuple(array.shape) if array is not None else tuple(shape or ())
        self.extra = tuple(extra)
        self.fail_after = fail_after
        self.calls = 0
        self._outputs = [Output(out_name, self.shape)] + [Output(n, a.shape) for n, a in self.extra]
        self.weights_dir = "weights"

    @property
    def output_description(self):
        return {o.name: None for o in self._outputs}

    def get_spec(self):
        return types.SimpleNamespace(description=types.SimpleNamespace(output=self._outputs))

    def predict(self, feed, *args, **kwargs):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("the ANE refused this program")
        value = self.array
        if value is None:
            value = np.asarray(feed["hidden"], dtype=np.float32)
        row = {self.out_name: value}
        for name, arr in self.extra:
            row[name] = arr
        return row


class Output:
    def __init__(self, name, shape):
        self.name = name
        self.type = types.SimpleNamespace(multiArrayType=types.SimpleNamespace(shape=list(shape)))


def clean_row(**overrides):
    row = {
        "chunk": 1024,
        "history": 0,
        "cpu_and_ne_ms": 15.84,
        "cpu_only_ms": 37.47,
        "cpu_only_status": "measured",
        "cpu_only_note": "",
        "mean_rel_error": 0.0141,
        "error": None,
    }
    row.update(overrides)
    return row


def run_main(*args, model=None, ref_array=None, build_error=None, out_env=None):
    """Drive the real main() with the two Core ML builders patched.

    Returns `(status, stdout, artifact text)`. With no `model` every config gets
    a fresh fake program whose block output echoes its hidden state, and the
    float32 reference is patched to echo it back too, so the numerics gap is
    exactly 0: these tests judge the accounting, not fp16 parity, which is what
    Core ML itself is for.
    """
    buf = io.StringIO()
    ref = ref_array if ref_array is not None else (model.array if model is not None else None)

    def echo(hidden, *rest, **kwargs):
        return np.asarray(hidden, dtype=np.float32)

    def build_block(t, history, weights):
        if build_error is not None:
            raise build_error
        return model if model is not None else FakeModel("block_output", shape=(t, D))

    def cpu_build(spec, weights_dir):
        return model if model is not None else FakeModel("block_output")

    env = dict(os.environ)
    if out_env is not None:
        env[probe.OUT_ENV] = str(out_env)
    prev = sys.argv
    sys.argv = ["tinytitan_ane_attention_probe.py", *args]
    try:
        with (
            mock.patch.object(probe, "build_block", build_block),
            mock.patch.object(probe, "cpu_model_for", cpu_build),
            mock.patch.object(probe, "reference", echo if ref is None else (lambda *a, **k: ref)),
            mock.patch.dict(os.environ, env, clear=True),
            redirect_stdout(buf),
        ):
            status = probe.main()
    finally:
        sys.argv = prev
    path = probe.out_path(env)
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    return status, buf.getvalue(), text


def model_for(t, array=None, **kwargs):
    return FakeModel("block_output", array=array, shape=(t, D), **kwargs)


def scratch():
    return pathlib.Path(tempfile.mkdtemp())


class ConfigParsingTests(unittest.TestCase):
    def test_the_shipped_default_list_parses(self):
        self.assertEqual(
            probe.parse_configs("1024:0,2048:0,4096:0,2048:4096"),
            [(1024, 0), (2048, 0), (4096, 0), (2048, 4096)],
        )

    def test_a_blank_config_list_is_refused_naming_the_flag(self):
        for value in ("", "   "):
            with self.subTest(value=value):
                with self.assertRaises(probe.ConfigError) as caught:
                    probe.parse_configs(value)
                self.assertIn("--configs", str(caught.exception))

    def test_a_config_that_is_not_chunk_colon_history_is_refused_naming_the_value(self):
        for value in ("1024", "1024:0:8", "a:0", "1024:x", "1024:0,"):
            with self.subTest(value=value):
                with self.assertRaises(probe.ConfigError) as caught:
                    probe.parse_configs(value)
                self.assertIn("--configs", str(caught.exception))
                self.assertIn(value, str(caught.exception))

    def test_the_refusal_names_a_wrong_shape_as_a_shape_and_not_as_a_non_number(self):
        # An entry with the wrong field count is a shape error even though the
        # field it does hold is a number; calling it "not a number" sends the
        # reader to the wrong key -- the wrong-cause class of AUD-193.
        for value in ("1024", "1024:0:8", "1024:0,"):
            with self.subTest(value=value):
                with self.assertRaises(probe.ConfigError) as caught:
                    probe.parse_configs(value)
                self.assertIn("CHUNK:HISTORY", str(caught.exception))
                self.assertNotIn("not a number", str(caught.exception))
        for value in ("a:0", "1024:x"):
            with self.subTest(value=value):
                with self.assertRaises(probe.ConfigError) as caught:
                    probe.parse_configs(value)
                self.assertIn("not a number", str(caught.exception))

    def test_a_chunk_below_one_token_is_refused(self):
        for value in ("0:0", "-512:0"):
            with self.subTest(value=value):
                with self.assertRaises(probe.ConfigError):
                    probe.parse_configs(value)

    def test_a_history_below_zero_is_refused(self):
        with self.assertRaises(probe.ConfigError):
            probe.parse_configs("1024:-1")

    def test_a_history_of_zero_is_the_first_chunk(self):
        self.assertEqual(probe.parse_configs("1024:0"), [(1024, 0)])

    def test_repeats_below_one_are_refused_naming_the_flag(self):
        for value in (0, -3):
            with self.subTest(value=value):
                with self.assertRaises(probe.ConfigError) as caught:
                    probe.parse_repeats(value)
                self.assertIn("--repeats", str(caught.exception))

    def test_two_repeats_is_accepted(self):
        self.assertEqual(probe.parse_repeats(2), 2)


class MedianTests(unittest.TestCase):
    def test_the_median_is_the_upper_middle_of_an_even_count(self):
        # The pre-fix driver indexed sorted(times)[len(times) // 2]. The value is
        # pinned so a later change to the estimator is a deliberate one, not a
        # silent shift in the number the go/no-go is read from.
        self.assertEqual(probe.median_ms([4.0, 1.0, 3.0, 2.0]), 3.0)
        self.assertEqual(probe.median_ms([5.0, 1.0, 3.0]), 3.0)

    def test_warm_ups_are_dropped_before_the_median(self):
        # The pre-fix loop discarded the first two of `repeats + 2` predicts by
        # index; what reaches the median is pinned so a changed warm-up count is a
        # deliberate one, not a silent shift of the headline.
        self.assertEqual(probe.timed_samples([1.0, 2.0, 3.0, 4.0, 5.0], 2), [3.0, 4.0, 5.0])
        self.assertEqual(probe.WARM_UPS, 2)
        self.assertEqual(len(probe.timed_samples([0.0] * (probe.WARM_UPS + 5), probe.WARM_UPS)), 5)


class ReferenceNormalizationTests(unittest.TestCase):
    def test_the_gpu_reference_is_normalized_from_the_recorded_measurement(self):
        # 84.3 s of full-attention time over 10 layers x 6,103 tokens.
        self.assertAlmostEqual(probe.ms_per_1k_layer_tokens(), 1381.3, places=1)

    def test_the_per_layer_chunk_mean_is_still_the_measured_number(self):
        self.assertAlmostEqual(probe.GPU_MS_PER_LAYER_CHUNK, 4215.0, places=0)

    def test_a_row_is_normalized_by_its_own_token_count(self):
        self.assertAlmostEqual(probe.ms_per_1k_tokens(214.18, 4096), 52.3, places=1)

    def test_the_measured_4096_row_is_the_26x_the_plan_records(self):
        # Corroboration, not a new claim: the plan's real-weight rehearsal measured
        # 26.7x on these shapes, and normalizing this file's own measured 214.18 ms
        # at chunk 4,096 against the recorded GPU number gives 26.4x.
        self.assertAlmostEqual(probe.speedup_over_gpu(214.18, 4096), 26.4, places=1)

    def test_the_reference_mean_and_the_normalized_rate_are_the_same_measurement(self):
        # 4,215 ms per layer-chunk of 3,051.5 tokens average is the same 1,381 ms
        # per 1k layer-tokens the rows are compared against.
        self.assertAlmostEqual(
            probe.ms_per_1k_layer_tokens(),
            probe.ms_per_1k_tokens(probe.GPU_MS_PER_LAYER_CHUNK, probe.GPU_PREFILL_TOKENS / 2),
            places=1,
        )


class NumericsCeilingTests(unittest.TestCase):
    def test_the_ceiling_sits_above_the_worst_error_this_machine_measured(self):
        # Measured on this M3 with the shipped default list: 0.0141 at 1024:0,
        # 0.0204 at 2048:0, 0.0292 at 4096:0 and 0.0998 at 2048:4096. A ceiling
        # under the last would fail the probe's own default run.
        self.assertGreater(probe.MAX_MEAN_REL_ERROR, 0.0998)

    def test_the_ceiling_is_tight_enough_to_be_a_check(self):
        # The pre-fix file printed and passed anything finite; an error of 0.5 or
        # worse is not the same block, so the check has to fire well before that.
        self.assertLess(probe.MAX_MEAN_REL_ERROR, 0.5)

    def test_a_non_finite_error_is_never_within_the_ceiling(self):
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                self.assertFalse(probe.error_within_ceiling(value))


class RowStatusTests(unittest.TestCase):
    def test_a_measured_and_checked_row_is_clean(self):
        self.assertEqual(probe.row_status(clean_row()), 0)

    def test_a_row_whose_block_errored_costs_the_run(self):
        self.assertEqual(probe.row_status(clean_row(error="RuntimeError: boom")), 1)

    def test_a_non_finite_error_costs_the_run(self):
        self.assertEqual(probe.row_status(clean_row(mean_rel_error=float("nan"))), 1)

    def test_an_error_over_the_ceiling_costs_the_run(self):
        self.assertEqual(probe.row_status(clean_row(mean_rel_error=0.9)), 1)

    def test_a_cpu_only_arm_that_errored_costs_the_run(self):
        self.assertEqual(
            probe.row_status(
                clean_row(cpu_only_ms=None, cpu_only_status="errored", cpu_only_note="boom")
            ),
            1,
        )

    def test_a_cpu_only_arm_skipped_by_the_chunk_cap_does_not(self):
        # A stated policy about a secondary arm, not a failed measurement.
        self.assertEqual(
            probe.row_status(clean_row(chunk=4096, cpu_only_ms=None, cpu_only_status="skipped")),
            0,
        )


class ReportLineTests(unittest.TestCase):
    def lines(self, **overrides):
        lines, _status = probe.row_report(clean_row(**overrides))
        return "\n".join(lines)

    def test_a_measured_row_prints_the_normalized_comparison(self):
        text = self.lines(chunk=4096, cpu_and_ne_ms=214.18)
        self.assertIn("214.18 ms", text)
        self.assertIn("52.3 ms per 1k tokens", text)
        self.assertIn("26.4x the GPU reference", text)

    def test_the_cpu_only_cap_is_stated_where_the_number_was(self):
        text = self.lines(chunk=4096, cpu_only_ms=None, cpu_only_status="skipped")
        self.assertIn("CPU_ONLY", text)
        self.assertIn(str(probe.CPU_ONLY_MAX_CHUNK), text)
        self.assertIn("not the go/no-go", text)

    def test_an_errored_cpu_only_arm_says_errored_not_absent(self):
        text = self.lines(
            cpu_only_ms=None, cpu_only_status="errored", cpu_only_note="RuntimeError: boom"
        )
        self.assertIn("CPU_ONLY errored", text)
        self.assertIn("RuntimeError: boom", text)

    def test_a_measured_cpu_only_arm_prints_its_ratio(self):
        text = self.lines(cpu_and_ne_ms=52.77, cpu_only_ms=102.74)
        self.assertIn("1.95x", text)

    def test_a_cpu_only_arm_that_measured_zero_milliseconds_is_still_printed(self):
        # The pre-fix artifact wrote `round(cpu_ms, 2) if cpu_ms else None`, so a
        # measurement of 0 read as an arm that never ran.
        text = self.lines(cpu_and_ne_ms=12.0, cpu_only_ms=0.0)
        self.assertIn("ratio 0.00x", text)
        self.assertNotIn("not run", text)

    def test_a_non_finite_error_is_named_rather_than_printed_as_a_number(self):
        text = self.lines(mean_rel_error=float("nan"))
        self.assertIn("NOT MEASURED", text)
        self.assertIn("not a finite number", text)
        self.assertNotIn("nan", text.lower())

    def test_an_error_over_the_ceiling_names_the_ceiling(self):
        text = self.lines(mean_rel_error=0.9)
        self.assertIn("NOT MEASURED", text)
        self.assertIn(str(probe.MAX_MEAN_REL_ERROR), text)

    def test_an_errored_config_says_which_one(self):
        text = self.lines(chunk=1024, error="RuntimeError: the ANE refused this program")
        self.assertIn("NOT RUN", text)
        self.assertIn("chunk 1024", text)
        self.assertIn("RuntimeError: the ANE refused this program", text)

    def test_the_footer_says_the_reference_averages_unequal_chunks(self):
        _status, out, _artifact = run_main(
            "--configs", "1024:0", "--repeats", "1", model=model_for(1024), out_env=scratch()
        )
        self.assertIn("4,215 ms", out)
        self.assertIn("6,103", out)
        self.assertIn("2007", out)
        self.assertIn("1,381", out)

    def test_a_history_bearing_row_states_the_quadratic_caveat(self):
        _status, out, _artifact = run_main(
            "--configs", "1024:2048", "--repeats", "1", model=model_for(1024), out_env=scratch()
        )
        self.assertIn("quadratic", out)
        # Measured at chunk 512: history lowers the gain (95.9x -> 62.6x), so the
        # caveat must say the row is penalized, not that it flatters the ANE.
        self.assertIn("smaller gain", out)
        self.assertNotIn("flatters", out)


class MainStatusTests(unittest.TestCase):
    def setUp(self):
        self.tmp = scratch()

    def test_a_run_that_measured_and_checked_every_config_exits_zero(self):
        status, out, artifact = run_main(
            "--configs", "1024:0,2048:0", "--repeats", "1", out_env=self.tmp
        )
        self.assertEqual(status, 0, out)
        self.assertIn("PROBE COMPLETE", out)
        self.assertNotIn("NOT MEASURED", out)
        self.assertEqual(len(json.loads(artifact)), 2)

    def test_a_nan_block_output_exits_nonzero_and_writes_no_nan(self):
        array = np.full((256, D), np.nan, dtype=np.float32)
        status, out, artifact = run_main(
            "--configs",
            "256:0",
            "--repeats",
            "1",
            model=model_for(256, array=array),
            out_env=self.tmp,
        )
        self.assertEqual(status, 1, out)
        self.assertIn("PROBE INCOMPLETE", out)
        # The artifact stays strict JSON: the pre-fix file wrote a literal NaN,
        # which RFC 8259 does not allow and a strict parser rejects.
        self.assertNotIn("NaN", artifact)
        json.loads(artifact, parse_constant=lambda name: self.fail(f"artifact holds {name}"))

    def test_an_error_over_the_ceiling_exits_nonzero(self):
        ref = np.ones((256, D), dtype=np.float32)
        status, out, _artifact = run_main(
            "--configs",
            "256:0",
            "--repeats",
            "1",
            model=model_for(256, array=ref * 10.0),
            ref_array=ref,
            out_env=self.tmp,
        )
        self.assertEqual(status, 1, out)
        self.assertIn("NOT MEASURED", out)

    def test_a_refused_configuration_exits_two_and_builds_nothing(self):
        status, out, artifact = run_main("--configs", "1024", out_env=self.tmp)
        self.assertEqual(status, 2, out)
        self.assertIn("REFUSED", out)
        self.assertIn("--configs", out)
        self.assertEqual(artifact, "")

    def test_repeats_of_zero_are_refused_instead_of_dividing_an_empty_list(self):
        # The pre-fix driver built the model, then died at the median with
        # IndexError: list index out of range.
        status, out, _artifact = run_main("--configs", "256:0", "--repeats", "0", out_env=self.tmp)
        self.assertEqual(status, 2, out)
        self.assertIn("--repeats", out)

    def test_one_errored_config_does_not_lose_the_rows_already_measured(self):
        good = model_for(256)
        seen = []

        def build_block(t, history, weights):
            seen.append(t)
            if len(seen) == 2:
                raise RuntimeError("the second config failed to build")
            return good

        buf = io.StringIO()
        prev = sys.argv
        sys.argv = ["x", "--configs", "256:0,512:0", "--repeats", "1"]
        try:
            with (
                mock.patch.object(probe, "build_block", build_block),
                mock.patch.object(probe, "cpu_model_for", return_value=good),
                mock.patch.object(
                    probe,
                    "reference",
                    lambda hidden, *a, **k: np.asarray(hidden, dtype=np.float32),
                ),
                mock.patch.dict(
                    os.environ, {probe.OUT_ENV: str(self.tmp / probe.ARTIFACT)}, clear=True
                ),
                redirect_stdout(buf),
            ):
                status = probe.main()
        finally:
            sys.argv = prev
        rows = json.loads((self.tmp / probe.ARTIFACT).read_text(encoding="utf-8"))
        self.assertEqual(status, 1, buf.getvalue())
        self.assertEqual([r["chunk"] for r in rows], [256, 512])
        self.assertIn("RuntimeError", rows[1]["error"])
        self.assertIsNone(rows[0]["error"])

    def test_a_config_with_no_output_of_the_block_shape_is_named_not_a_traceback(self):
        model = FakeModel("something_else", np.ones((7, 7), dtype=np.float32))
        status, out, _artifact = run_main(
            "--configs", "256:0", "--repeats", "1", model=model, out_env=self.tmp
        )
        self.assertEqual(status, 1, out)
        self.assertIn("(256, 2048)", out)

    def test_a_blocked_artifact_write_costs_the_run_and_names_the_path(self):
        locked = self.tmp / "locked"
        locked.mkdir()
        locked.chmod(0o500)
        target = locked / probe.ARTIFACT
        try:
            status, out, _artifact = run_main(
                "--configs", "256:0", "--repeats", "1", model=model_for(256), out_env=target
            )
        finally:
            locked.chmod(0o700)
        self.assertEqual(status, 1, out)
        self.assertIn(str(target), out)

    def test_a_predict_that_raises_becomes_a_row_not_a_lost_run(self):
        model = model_for(256, fail_after=3)
        status, out, artifact = run_main(
            "--configs", "256:0", "--repeats", "3", model=model, out_env=self.tmp
        )
        self.assertEqual(status, 1, out)
        self.assertIn("NOT RUN", out)
        rows = json.loads(artifact)
        self.assertIn("RuntimeError", rows[0]["error"])

    def test_a_chunk_over_the_cpu_only_cap_reports_the_reason_instead_of_nothing(self):
        # Pre-fix the arm was behind `if t <= 2048:` with no else: the 4096 row
        # printed no comparison line and its artifact said `"cpu_only_ms": null`,
        # which is exactly what an errored arm looked like too.
        status, out, artifact = run_main("--configs", "4096:0", "--repeats", "1", out_env=self.tmp)
        self.assertEqual(status, 0, out)
        self.assertIn("not run", out)
        self.assertIn(str(probe.CPU_ONLY_MAX_CHUNK), out)
        self.assertNotIn("errored", out)
        rows = json.loads(artifact)
        self.assertIsNone(rows[0]["cpu_only_ms"])
        self.assertEqual(rows[0]["cpu_only_status"], "skipped")


class WarmUpAccountingTests(unittest.TestCase):
    """The warm-ups dropped where the headline number is read, not only in the helper."""

    def scripted(self, durations):
        values = []
        now = 0.0
        for duration in durations:
            values.append(now)
            now += duration
            values.append(now)
        state = {"i": 0}

        def counter():
            index = state["i"]
            state["i"] += 1
            return values[index] if index < len(values) else 1e9

        return counter

    def test_the_reported_median_covers_the_timed_runs_only(self):
        # The ANE arm makes WARM_UPS + 1 calls and the CPU arm CPU_WARM_UPS +
        # CPU_REPEATS; warm-ups are 1 s each and every timed run 100 ms, so a
        # median that let a warm-up in reads 1000, not 100.
        ane = [1.0] * probe.WARM_UPS + [0.1]
        cpu = [1.0] * probe.CPU_WARM_UPS + [0.1] * probe.CPU_REPEATS
        tmp = scratch()
        model = model_for(256)
        with mock.patch.object(probe.time, "perf_counter", self.scripted(ane + cpu)):
            status, out, _artifact = run_main(
                "--configs", "256:0", "--repeats", "1", model=model, out_env=tmp
            )
        self.assertEqual(status, 0, out)
        rows = json.loads((tmp / probe.ARTIFACT).read_text(encoding="utf-8"))
        self.assertAlmostEqual(rows[0]["cpu_and_ne_ms"], 100.0, places=6)
        self.assertAlmostEqual(rows[0]["cpu_only_ms"], 100.0, places=6)
        self.assertIn("100.00 ms/chunk-layer", out)
        self.assertEqual(model.calls, probe.WARM_UPS + 1 + probe.CPU_WARM_UPS + probe.CPU_REPEATS)


class ExitContractTests(unittest.TestCase):
    """The seam this driver was filed for: a `__main__` block that discards the status.

    `main()` is tested through the patched builders above, but nothing in the
    interpreter runs it there, so the guard itself has to be read from the file:
    a bare `main()` makes every refusal, every errored config and every lost
    artifact exit 0.
    """

    def test_the_guard_forwards_the_status_and_nothing_else_runs(self):
        source = (BENCH / "tinytitan_ane_attention_probe.py").read_text(encoding="utf-8")
        self.assertEqual(source.count('if __name__ == "__main__":'), 1)
        guard = source.split('if __name__ == "__main__":')[1]
        self.assertIn("sys.exit(main())", guard)
        self.assertNotIn("\n    main()", guard)
        self.assertNotIn("raise SystemExit", guard)

    def test_main_returns_a_status_on_every_path(self):
        source = (BENCH / "tinytitan_ane_attention_probe.py").read_text(encoding="utf-8")
        body = source.split("def main() -> int:")[1].split("if __name__ ==")[0]
        self.assertNotIn("return None", body)
        for status in ("return 0", "return 1", "return 2"):
            self.assertIn(status, body)


class ArtifactPathTests(unittest.TestCase):
    def test_the_default_path_is_anchored_at_the_repository_not_the_directory(self):
        self.assertEqual(
            probe.DEFAULT_OUT,
            ROOT / ".build" / "benchmark-results" / "ane-attention-probe.json",
        )

    def test_the_environment_can_name_a_different_one(self):
        env = dict(os.environ, **{probe.OUT_ENV: "/tmp/elsewhere/probe.json"})
        self.assertEqual(probe.out_path(env), pathlib.Path("/tmp/elsewhere/probe.json"))

    def test_a_blank_environment_variable_uses_the_default(self):
        self.assertEqual(probe.out_path({probe.OUT_ENV: "  "}), probe.DEFAULT_OUT)

    def test_the_directory_is_created_before_the_write(self):
        nested = scratch() / "a" / "b"
        probe.write_artifact([{"chunk": 1}], nested / probe.ARTIFACT)
        self.assertTrue((nested / probe.ARTIFACT).is_file())

    def test_a_run_from_a_foreign_directory_writes_where_it_prints(self):
        # The pre-fix driver resolved `.build/...` against the working directory:
        # measured from /tmp it printed every row and then died with FileNotFoundError.
        target = scratch()
        prev = pathlib.Path.cwd()
        os.chdir(BENCH)
        try:
            status, out, artifact = run_main(
                "--configs", "256:0", "--repeats", "1", model=model_for(256), out_env=target
            )
        finally:
            os.chdir(prev)
        self.assertEqual(status, 0, out)
        self.assertIn(str(target / probe.ARTIFACT), out)
        self.assertTrue((target / probe.ARTIFACT).is_file())
        self.assertTrue(artifact)


class HeaderTests(unittest.TestCase):
    def test_the_header_names_the_configs_repeats_and_warm_ups(self):
        _status, out, _artifact = run_main(
            "--configs", "1024:0", "--repeats", "2", model=model_for(1024), out_env=scratch()
        )
        self.assertIn("configs: 1024:0", out)
        self.assertIn("repeats: 2", out)
        self.assertIn("numpy", out)
        self.assertIn("warm-up", out)

    def test_the_header_records_that_the_weights_are_random(self):
        _status, out, _artifact = run_main(
            "--configs", "1024:0", "--repeats", "1", model=model_for(1024), out_env=scratch()
        )
        self.assertIn("random fp16", out)


class ImportSideEffectTests(unittest.TestCase):
    def test_importing_the_probe_builds_nothing_and_writes_nothing(self):
        child = r"""
import sys, builtins
from unittest import mock
for name in FAKED:
    sys.modules.setdefault(name, mock.MagicMock(name=name))
sys.path.insert(0, BENCH)
opened = []
real_open = builtins.open
def watching(file, *a, **k):
    opened.append(str(file))
    return real_open(file, *a, **k)
builtins.open = watching
import tinytitan_ane_attention_probe
coremltools = sys.modules["coremltools"]
print("IMPORTED")
print("CONVERTS", coremltools.convert.call_count)
print("OPENS", [p for p in opened if "ane-attention-probe" in p])
"""
        source = (
            child.replace("FAKED", repr(FAKED))
            .replace("BENCH", repr(str(BENCH)))
            .replace("\\\n", "")
        )
        proc = subprocess.run(
            [sys.executable, "-c", source],
            cwd=str(scratch()),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("IMPORTED", proc.stdout)
        self.assertIn("CONVERTS 0", proc.stdout)
        self.assertIn("OPENS []", proc.stdout)


class GeometryTests(unittest.TestCase):
    """The arithmetic the timing is about, asserted against real numpy."""

    def test_rope_tables_cover_half_the_rotary_width_per_row(self):
        cos_t, sin_t = probe.rope_tables(0, 3)
        self.assertEqual(cos_t.shape, (3, probe.ROTARY // 2))
        self.assertEqual(sin_t.shape, (3, probe.ROTARY // 2))

    def test_rope_tables_stay_on_the_unit_circle(self):
        cos_t, sin_t = probe.rope_tables(0, 8)
        self.assertTrue(
            np.allclose(
                cos_t.astype(np.float64) ** 2 + sin_t.astype(np.float64) ** 2, 1.0, atol=1e-3
            )
        )

    def test_rope_positions_continue_from_the_history_not_from_zero(self):
        # A chunk after 100 tokens of history is at positions 100.., and the
        # engine reads it that way; restarting the table would make the probe
        # time a different block than the one that shipped.
        cos_t, _ = probe.rope_tables(100, 3)
        first, _ = probe.rope_tables(0, 3)
        self.assertFalse(np.allclose(cos_t.astype(np.float64), first.astype(np.float64), atol=1e-3))

    def test_the_causal_mask_allows_history_and_the_chunk_below_the_diagonal(self):
        mask = probe.causal_mask(4, 8)
        self.assertEqual(mask.shape, (1, 1, 4, 12))
        self.assertTrue(np.all(mask[0, 0, 0, :9] == 0))
        self.assertTrue(np.all(np.isneginf(mask[0, 0, 0, 9:])))
        self.assertTrue(np.all(mask[0, 0, 3, :] == 0))

    def test_the_packed_projection_holds_query_then_gate_per_head(self):
        weights = probe.make_weights(np.random.default_rng(1))
        self.assertEqual(weights["wq"].shape, (2 * probe.N_Q_HEADS * HEAD_DIM, D))
        self.assertEqual(weights["wk"].shape, (probe.N_KV_HEADS * HEAD_DIM, D))
        self.assertEqual(weights["wo"].shape, (D, probe.N_Q_HEADS * HEAD_DIM))


if __name__ == "__main__":
    unittest.main()
