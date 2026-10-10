"""AUD-276: six drivers print a metric the run never logged as a measured zero.

The pages were fixed by AUD-273, AUD-274 and AUD-275: a headline whose runs never
logged the key now refuses at status 2, or reports `PARTIAL` with its count. What
was left is the *printed* figure, and it is not the same thing. Measured here by
driving the real code with the run functions stubbed (`/tmp/aud276/measure.py`,
`/tmp/aud276/matrix.py`), no model and no server:

  * `tinytitan_mtp_phases.py:421,427` and `tinytitan_mtp_b3_qualification.py:204,207`
    print `acc 0.0% passes 0` for a run whose server logged no MTP footer, and
    `0.000 tok/s` for a run whose log matched neither the MTP line nor the
    generation line. Both drivers' `one_run` builds those keys inside
    `if m:` / `if gen:` (`tinytitan_mtp_phases.py:178-196`), so the key is absent
    exactly when the path did not engage -- which is the one variable the sweep
    exists to see. The page then refuses correctly (`rc=2`,
    `NOT MEASURED: the on arm -- no run logged acceptance`) while the line the
    operator watched for the whole sweep said the draft path scored zero.
  * `tinytitan_ane_prefill_ab.py:306,307` prints `prefill    0.00 s` for a run
    with no `prefill_s`, and `decode  0.000` for one with no `decode_tok_s` -- and
    a prefill of 0.00 s is the opposite of a finding, since it is the fastest
    number the tool can report.
  * `tinytitan_gate0_profile.py:430,431` prints `busy_per_token=0.000 ms,
    occupancy=0.0%` for a run whose log carried no occupancy line, in the same
    file whose table renders that cell as `not logged` (`_cell`, :524, used ~12
    times). One file, two answers to one question, and the live line is the wrong
    one.
  * `ane_prefill_ab_matrix.py:229` is the strongest of the five: it is not a live
    line, it is a published table. `summarize()` refuses to compute a ratio when
    the ANE arm fell back, when the prompt is under one chunk, and when the ANE
    cannot serve the model at all -- `test_ane_prefill_ab_matrix.py:66-126` pins
    `assertNotIn("speedup", summary)` for all three -- and `format_row()` prints
    `0.000` in the speedup column for every one of them:

        fell-back        0.86      0.91    0.000     False  ANE arm fell back to the GPU
        ane-unavailable  0.85         -    0.000      None  ANE unavailable: ...

    `0.000` reads as "the ANE made prefill infinitely slower", a measurement the
    driver decided not to make. The same function prints `-` for a missing value
    in its three other columns, and for the whole row when the run refused at
    `error` (`- - - - off arm: exit 127`), which is the internal evidence that
    the `0.000` is the defect and not the house choice.

The fix is one primitive, `tinytitan_profile.logged(value, spec, suffix)`, doing
what gate0's `_cell` and `tinytitan_sampler_ab._metric` already do privately, so
the rule has one home; the matrix's column prints the `-` its own row already
uses. Every test below asserts the *figure*, not a keyword, because the point is
that a fabricated number was published.

Run from `benchmark/`:

    python3 -m unittest test_logged_metric_cells

"""

from __future__ import annotations

import contextlib
import io
import pathlib
import sys
import tempfile
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))

import ane_prefill_ab_matrix as matrix  # noqa: E402
import tinytitan_ane_prefill_ab as ane  # noqa: E402
import tinytitan_gate0_profile as g0  # noqa: E402
import tinytitan_mtp_b3_qualification as b3  # noqa: E402
import tinytitan_mtp_phases as ph  # noqa: E402
import tinytitan_profile as prof  # noqa: E402
import tinytitan_sampler_ab as sampler  # noqa: E402

TMP = pathlib.Path(tempfile.mkdtemp(prefix="aud276-cells-"))


def setUpModule() -> None:
    (TMP / "target").mkdir(exist_ok=True)
    (TMP / "sidecar").mkdir(exist_ok=True)
    (TMP / "target/verified-install.json").write_text("{}")
    (TMP / "sidecar/manifest.json").write_text("{}")
    g0.preflight = lambda *a, **k: None


def mtp_row(arm, sha, *, rate=None, footer=False):
    """One row in the shape `tinytitan_mtp_phases.one_run` builds.

    `rate=None` is the log that matched neither regex, `footer=False` the run
    whose server logged no `mtp` line -- both leave the key absent rather than 0.
    """
    row = {"arm": arm, "sha256": sha, "completion_tokens": 256}
    if rate is not None:
        row["decode_tok_s"] = rate
        row["prefill_s"] = 1.0
    if footer:
        row.update(acceptance=90.0, passes=40, emitted_per_pass=1.8)
    return row


def ane_row(arm, sha, *, prefill=None, rate=None):
    row = {
        "arm": arm,
        "sha256": sha,
        "prompt_tokens": 10141,
        "completion_tokens": 256,
        "first_line": "7 x 1 = 7",
        "fallback": False,
    }
    if prefill is not None:
        row["prefill_s"] = prefill
    if rate is not None:
        row["decode_tok_s"] = rate
    return row


def drive(main, patch_module, rows, argv):
    """Run a driver's real `main()` over stubbed runs and return (status, stdout)."""
    original = patch_module.one_run
    queue = list(rows)
    saved = []
    patch_module.one_run = lambda *a, **k: queue.pop(0)
    for mod in (ph, b3, ane, sampler):
        if hasattr(mod, "ROOT"):
            saved.append((mod, mod.ROOT))
            mod.ROOT = TMP
    argv_saved = sys.argv[1:]
    sys.argv = ["driver"] + argv
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            status = main()
    finally:
        sys.argv = argv_saved
        patch_module.one_run = original
        for mod, root in saved:
            mod.ROOT = root
    return status, buf.getvalue()


MTP_ARGV = ["--target", str(TMP / "target"), "--sidecar", str(TMP / "sidecar"), "--pairs", "1"]
B3_ARGV = ["--scenario", "function", "--blocks", "1"]
ANE_ARGV = ["--quant", "4bit", "--pairs", "1"]
SAMPLER_ARGV = ["--quant", "8bit", "--pairs", "1"]


def progress(out, marker):
    """The driver's live lines -- the ones carrying this marker."""
    return [line for line in out.splitlines() if marker in line]


class LoggedPrimitive(unittest.TestCase):
    """One rule, one home: an unlogged metric is words, never a number."""

    def test_an_unlogged_metric_is_not_a_zero(self):
        self.assertEqual(prof.logged(None, ".3f"), "not logged")

    def test_a_logged_metric_keeps_its_format(self):
        self.assertEqual(prof.logged(1.5, ".3f"), "1.500")
        self.assertEqual(prof.logged(1.5, "7.3f"), "  1.500")

    def test_a_unit_only_appears_beside_a_number(self):
        self.assertEqual(prof.logged(0.5, ".2f", " ms"), "0.50 ms")
        self.assertEqual(prof.logged(None, ".2f", " ms"), "not logged")

    def test_a_count_is_a_count(self):
        self.assertEqual(prof.logged(40, "d"), "40")
        self.assertEqual(prof.logged(None, "d"), "not logged")


class MtpPhasesProgressLine(unittest.TestCase):
    def test_a_run_with_no_footer_prints_no_acceptance_of_zero(self):
        rows = [mtp_row("off", "warmup", rate=10.0), mtp_row("on", "warmup", rate=10.0)]
        rows += [mtp_row(arm, "aa", rate=10.0) for arm in ("off", "on", "on", "off")]
        _status, out = drive(ph.main, ph, rows, MTP_ARGV)
        lines = progress(out, "acc ")
        self.assertTrue(lines, f"no progress line printed: {out}")
        for line in lines:
            self.assertNotIn("acc 0.0%", line)
        self.assertIn("acc not logged passes not logged", out)

    def test_a_run_with_no_rate_prints_no_zero_rate(self):
        rows = [mtp_row("off", "warmup", rate=10.0), mtp_row("on", "warmup", rate=10.0)]
        rows += [mtp_row(arm, "aa", footer=True) for arm in ("off", "on", "on", "off")]
        _status, out = drive(ph.main, ph, rows, MTP_ARGV)
        lines = progress(out, "tok/s")
        self.assertTrue(lines, f"no progress line printed: {out}")
        for line in lines:
            self.assertNotIn("0.000 tok/s", line)
        self.assertIn("not logged tok/s", out)

    def test_a_measured_run_still_prints_its_figures(self):
        rows = [mtp_row("off", "warmup", rate=10.0), mtp_row("on", "warmup", rate=10.0)]
        rows += [
            mtp_row(arm, "aa", rate=10.0, footer=(arm == "on"))
            for arm in ("off", "on", "on", "off")
        ]
        _status, out = drive(ph.main, ph, rows, MTP_ARGV)
        self.assertIn(" 10.000 tok/s", out)
        self.assertIn("acc 90.0% passes 40", out)


class B3ProgressLine(unittest.TestCase):
    """b3 reaches the neighbour's `one_run`, so the stub patches `ph`, not `b3`."""

    def test_a_run_with_no_footer_prints_no_acceptance_of_zero(self):
        rows = [mtp_row("off", "warmup", rate=10.0), mtp_row("on", "warmup", rate=10.0)]
        rows += [mtp_row(arm, "aa", rate=10.0) for arm in ("off", "on", "on", "off")]
        _status, out = drive(b3.main, ph, rows, B3_ARGV)
        lines = progress(out, "acc ")
        self.assertTrue(lines, f"no progress line printed: {out}")
        for line in lines:
            self.assertNotIn("acc 0.0%", line)
        self.assertIn("acc not logged", out)

    def test_a_run_with_no_rate_prints_no_zero_rate(self):
        rows = [mtp_row("off", "warmup", rate=10.0), mtp_row("on", "warmup", rate=10.0)]
        rows += [mtp_row(arm, "aa", footer=True) for arm in ("off", "on", "on", "off")]
        _status, out = drive(b3.main, ph, rows, B3_ARGV)
        lines = progress(out, "tok/s")
        self.assertTrue(lines, f"no progress line printed: {out}")
        for line in lines:
            self.assertNotIn("0.000 tok/s", line)


class AnePrefillProgressLine(unittest.TestCase):
    def test_a_run_with_no_prefill_time_prints_no_zero_seconds(self):
        rows = [
            ane_row("gpu", "warmup", prefill=1.0, rate=20.0),
            ane_row("ane", "warmup", prefill=1.0, rate=20.0),
        ]
        rows += [ane_row(arm, "aa", rate=20.0) for arm in ("gpu", "ane", "ane", "gpu")]
        _status, out = drive(ane.main, ane, rows, ANE_ARGV)
        lines = progress(out, "prefill ")
        self.assertTrue(lines, f"no progress line printed: {out}")
        for line in lines:
            self.assertNotIn("0.00 s", line)
        self.assertIn("prefill not logged", out)

    def test_a_run_with_no_rate_prints_no_zero_rate(self):
        rows = [
            ane_row("gpu", "warmup", prefill=1.0, rate=20.0),
            ane_row("ane", "warmup", prefill=1.0, rate=20.0),
        ]
        rows += [ane_row(arm, "aa", prefill=1.0) for arm in ("gpu", "ane", "ane", "gpu")]
        _status, out = drive(ane.main, ane, rows, ANE_ARGV)
        lines = progress(out, "decode ")
        self.assertTrue(lines, f"no progress line printed: {out}")
        for line in lines:
            self.assertNotIn("0.000", line)
        self.assertIn("decode not logged", out)


class Gate0ProgressLine(unittest.TestCase):
    """gate0's table already answers `not logged`; its live line answered 0.000."""

    def run_one(self, record):
        saved = {
            k: getattr(g0, k)
            for k in (
                "launch",
                "wait_ready",
                "resolve_api_model",
                "_terminate_all",
                "time",
                "benchmark_log_path",
                "parse_log",
                "generate",
            )
        }
        g0.launch = lambda *a, **k: None
        g0.wait_ready = lambda port, attempts=1: True
        g0.resolve_api_model = lambda port: "test-model"
        g0._terminate_all = lambda: None
        g0.time = types.SimpleNamespace(sleep=lambda seconds: None)
        g0.benchmark_log_path = lambda name: str(TMP / name)
        g0.parse_log = lambda path: [dict(record)]
        g0.generate = lambda port: {
            "wall_s": 3.0,
            "completion_tokens": 64,
            "prompt_tokens": 8,
            "completion_sha256": "aa",
        }
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                g0.run_quant("4bit", 1)
        finally:
            for key, value in saved.items():
                setattr(g0, key, value)
        return buf.getvalue()

    def test_a_run_with_no_occupancy_line_prints_no_zero(self):
        out = self.run_one({"prefill_s": 1.0, "decode_s": 2.0, "decode_tok_s": 4.0})
        self.assertNotIn("busy_per_token=0.000 ms", out)
        self.assertNotIn("occupancy=0.0%", out)
        self.assertIn("busy_per_token=not logged", out)
        self.assertIn("occupancy=not logged", out)

    def test_a_run_that_logged_occupancy_still_prints_it(self):
        out = self.run_one(
            {
                "prefill_s": 1.0,
                "decode_s": 2.0,
                "decode_tok_s": 4.0,
                "busy_per_token_ms": 0.125,
                "occupancy_pct": 61.5,
            }
        )
        self.assertIn("busy_per_token=0.125 ms", out)
        self.assertIn("occupancy=61.5%", out)


def sampler_row(arm, sha, rate):
    return {
        "arm": arm,
        "sha256": sha,
        "completion_sha256": sha,
        "decode_tok_s": rate,
        "completion_tokens": 64,
    }


class Gate0TableCells(unittest.TestCase):
    """gate0's eleven table cells: the refactor must not change what they print."""

    def summary(self, **median):
        return {
            "quant": "4bit",
            "runs": 1,
            "spread": {"decode_tok_s": "n/a"},
            "median": median,
            "counts": {},
            "roles_per_token_ms": {"decode": 1.0},
            "gaps_per_token_ms": {"embed->router": 0.5},
            "completion_sha256": ["aa"],
        }

    def page(self, **median):
        lines, _status = g0.report([self.summary(**median)])
        return "\n".join(lines)

    def test_an_unlogged_metric_is_words_in_the_table(self):
        page = self.page(decode_tok_s=4.0)
        self.assertIn("GPU busy/token    not logged", page)
        self.assertIn("queue occupancy   not logged", page)
        self.assertNotIn("busy/token    0.000 ms", page)

    def test_a_logged_metric_keeps_its_figure(self):
        page = self.page(decode_tok_s=4.0, busy_per_token_ms=0.125, occupancy_pct=61.5)
        self.assertIn("GPU busy/token    0.125 ms", page)
        self.assertIn("queue occupancy   61.5%", page)


class SamplerPageCells(unittest.TestCase):
    """A refactor guard for the other private duplicate, `_metric`."""

    def test_an_unlogged_busy_metric_prints_words_not_zero(self):
        rows = []
        for arm in ("generic", "tiled"):
            rows += [sampler_row(arm, "warmup", 30.0)]
        for arm in ("generic", "tiled", "tiled", "generic"):
            rows += [sampler_row(arm, "aa" if arm == "generic" else "bb", 30.0)]
        _status, out = drive(sampler.main, sampler, rows, SAMPLER_ARGV)
        lines = [line for line in out.splitlines() if "busy/token" in line]
        self.assertTrue(lines, f"no progress line printed: {out}")
        for line in lines:
            self.assertNotIn("busy/token 0.00 ms", line)
            self.assertNotIn("sample_gap 0.00 ms", line)
        self.assertIn("busy/token not logged", out)
        self.assertIn("sample_gap not logged", out)


class MatrixSpeedupColumn(unittest.TestCase):
    """The published table, driven through the real `summarize()`."""

    @staticmethod
    def arm_run(seconds, used_ane, tokens=6000):
        return {
            "prefill_seconds": seconds,
            "prefill_tokens": tokens,
            "used_ane": used_ane,
            "response_sha256": "aa",
            "response_head": "x",
        }

    def cell(self, record):
        return matrix.format_row(matrix.summarize(record))

    def assertNoFabricatedSpeedup(self, line):
        self.assertNotIn("0.000", line, f"a ratio never computed was published: {line}")
        columns = line.split()
        self.assertEqual(columns[3], "-", f"speedup column reads {columns[3]!r}: {line}")

    def test_an_ane_arm_that_fell_back_publishes_no_speedup(self):
        line = self.cell(
            {
                "model": "fell-back",
                "arms": {
                    "off": [self.arm_run(0.85, True), self.arm_run(0.87, True)],
                    "on": [self.arm_run(0.90, False), self.arm_run(0.92, False)],
                },
            }
        )
        self.assertIn("ANE arm fell back to the GPU", line)
        self.assertNoFabricatedSpeedup(line)

    def test_a_prompt_under_one_chunk_publishes_no_speedup(self):
        line = self.cell(
            {
                "model": "short-prompt",
                "arms": {
                    "off": [self.arm_run(0.85, True, tokens=1000)],
                    "on": [self.arm_run(0.86, True, tokens=1000)],
                },
            }
        )
        self.assertIn("under one 4096-token chunk", line)
        self.assertNoFabricatedSpeedup(line)

    def test_a_model_the_ane_cannot_serve_publishes_no_speedup(self):
        line = self.cell(
            {
                "model": "ane-unavailable",
                "arms": {"off": [self.arm_run(0.85, True)], "on": []},
                "ane_unavailable": "sidecar geometry refused",
            }
        )
        self.assertIn("ANE unavailable", line)
        self.assertNoFabricatedSpeedup(line)

    def test_a_row_that_refused_wholesale_still_prints_dashes(self):
        line = self.cell(
            {"model": "off-failed", "arms": {"off": [], "on": []}, "error": "off arm: exit 127"}
        )
        self.assertIn("off arm: exit 127", line)
        self.assertNoFabricatedSpeedup(line)

    def test_a_measured_ratio_is_still_published_as_a_number(self):
        line = self.cell(
            {
                "model": "healthy",
                "arms": {
                    "off": [self.arm_run(0.85, True), self.arm_run(0.87, True)],
                    "on": [self.arm_run(0.45, True), self.arm_run(0.47, True)],
                },
            }
        )
        self.assertIn("1.870", line)


if __name__ == "__main__":
    unittest.main()
