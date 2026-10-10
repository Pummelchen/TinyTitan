#!/usr/bin/env python3
"""Tests for the four sweep drivers whose page cannot fail: AUD-274.

Each of these answers a question -- what does the ANE handover cost a decode
token, which slot count is worth its cache footprint, is decode bandwidth-bound
or dependency-stalled, does gathering the selection beat scoring it -- and each
printed its answer and returned 0 whatever it had measured. The shapes below
were measured by driving each real `main()` with its run function stubbed, so
nothing here starts a model, a server, a build, a Core ML program or a download.

1. `ane_steady_state_decode`. Eight failed runs print `prompt ? tokens` and two
   `steady state unavailable` lines and exit 0 (measured); an arm that produced
   nothing is one of those two lines; a run set where every run stopped early
   still prints a prefill speedup built from rows the driver itself marks
   `[INCOMPLETE]` and exits 0; the ANE prefill median over 2 of 4 runs prints
   `speedup 2.00x` naming no denominator; and `--reps 0` dies with
   `IndexError: list index out of range` (measured) after the header has printed.
2. `expert_cache_slots`. Every slot failed prints the table with its column
   headers and no rows and exits 0 (measured); a slot that produced nothing is
   simply absent from the published table (`if not group: continue`) with nothing
   explaining the gap; a group whose runs logged no IO line dies with
   `StatisticsError: no median for empty data` (measured) after the header; and a
   run whose IO line parsed without a per-token figure dies with
   `TypeError: unsupported format string passed to NoneType.__format__` (measured)
   in the middle of the sweep, so the later slots never run.
3. `tinytitan_gate0_profile`. The file's whole output is which optimization track
   to fund, decided by `busy >= 45` / `busy <= 35` over
   `m["busy_per_token_ms"] or 0` -- so a run set that never logged the metric
   prints `GPU busy/token 0.000 ms (0.0% of token)` and
   `VERDICT: DEPENDENCY-STALLED (Track B is the game)` and exits 0 (measured).
   A missing `occupancy_pct` instead dies mid-page with `TypeError` (measured),
   leaving a half-printed profile, and a run set with no output digest prints
   `output digest []` with no flag.
4. `ane_gather_probe`. When an arm records an error the four inputs to the ratio
   are incomplete and the `verdict inputs:` block is skipped with no line at all
   (measured) and exit 0; `--repeats 0` times nothing, `statistics.median([])`
   raises inside the arm's own `try`, and that lands in `ane_error` as
   `StatisticsError: ` -- an argument-test mislabelled as a Core ML failure. The
   ratio's own gate read its inputs with `and`, so a dense arm that logged `0.0 s`
   skipped the block while every figure still counted as measured: that shape is
   the one a first mutation sweep left alive, and it now names a zero denominator
   as a contested claim instead of dividing by it.

The three statuses are the ones `tools/model-guard.sh`,
`tools/qwen35_reference.py:411` and `tools/reconcile_snapshot.py` already use:

    0  every published figure came from a run that measured it
    1  it measured, and something the page claims is contested -- with the claim named
    2  the page's headline could not be computed, with the reason named

The counting primitive is `tinytitan_profile.arm_metric` / `metric_count`, added
for AUD-273 and shared here rather than re-derived per driver.
"""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import statistics
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmark"
sys.path.insert(0, str(BENCH))

# The gather probe converts a Core ML program at module scope, so the fakes go in
# before it is imported; `mb.program(...)` answers a mock decorator, which is the
# same stub test_tinytitan_ane_attention_probe.py installs.
FAKED = [
    "coremltools",
    "coremltools.converters",
    "coremltools.converters.mil",
    "coremltools.converters.mil.mil",
    "coremltools.converters.mil.mil.types",
    "coremltools.target",
]
for _name in FAKED:
    sys.modules.setdefault(_name, mock.MagicMock(name=_name))

import ane_steady_state_decode as ss  # noqa: E402
import ane_gather_probe as gp  # noqa: E402
import expert_cache_slots as slots  # noqa: E402
import tinytitan_gate0_profile as g0  # noqa: E402
import tinytitan_profile as prof  # noqa: E402

MESSAGES = ROOT / ".build/tt011-messages.json"
GATHER_RESULTS = ROOT / ".build/aud274-gather-test"


def installed_model(files: dict[str, str]) -> pathlib.Path:
    root = pathlib.Path(tempfile.mkdtemp(prefix="aud274-model-"))
    for name, body in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    return root


# ------------------------------------------------------------------ steady state
SS_MODEL = installed_model({"verified-install.json": "{}", "ane_prefill/x": "1"})
SMALL, BIG = 64, 448


def ss_row(arm: str, length: int, *, complete: bool = True, failed: bool = False):
    if failed:
        return {"arm": arm, "max_new": length, "failed": "cli exited without a footer"}
    return {
        "arm": arm,
        "max_new": length,
        "stop": "maxTokens" if complete else "endOfTurn",
        "prompt_tokens": 12500,
        "prefill_s": 2.0 if arm == "gpu" else 1.0,
        "new_tokens": length,
        "decode_s": length / 16.0,
        "decode_tok_s": 16.0 if arm == "gpu" else 14.0,
        "wire": [],
        "complete": complete,
    }


def ss_table(**arms):
    """{(arm, length): [rows...]} for the two lengths, keyed by arm outcome.

    One `--reps` block is `gpu ane ane gpu` x two lengths, so every (arm, length)
    key answers two CLI calls; the fixture hands out two rows per key so a
    denominator is a real count of runs the driver actually made, not an
    artifact of the stub running dry. `ok` gives two complete runs at the default
    rates, `failed` a CLI that never printed a footer, `incomplete` runs that
    stopped at end-of-turn, and `reverse` an ANE arm whose long run decoded no
    longer than its short one.
    """
    table = {}
    for arm in ("gpu", "ane"):
        how = arms.get(arm, "ok")
        for length in (SMALL, BIG):
            if how == "failed":
                row = ss_row(arm, length, failed=True)
            elif how == "incomplete":
                row = ss_row(arm, length, complete=False)
            elif how == "reverse":
                decode = 40.0 if length == SMALL else 30.0
                row = {**ss_row(arm, length), "decode_s": decode}
            else:
                row = ss_row(arm, length)
            table[(arm, length)] = [dict(row), dict(row)]
    return table


def ss_pairs(**arms):
    return [row for rows in ss_table(**arms).values() for row in rows]


def drive_ss(table: dict, argv: list[str] | None = None):
    queues = {key: list(rows) for key, rows in table.items()}

    def run_cli(model, prompt, max_new, ane, messages_file, wire_trace=False):
        arm = "ane" if ane else "gpu"
        queue = queues.get((arm, max_new), [])
        if queue:
            return queue.pop(0)
        return {"arm": arm, "max_new": max_new, "failed": "stub ran out"}

    buf = io.StringIO()
    with (
        mock.patch.object(ss, "run_cli", run_cli),
        mock.patch.object(ss, "pgrep_answer", return_value=("clear", [])),
        mock.patch.object(
            sys,
            "argv",
            ["ane_steady_state_decode", "--model", str(SS_MODEL)] + (argv or []),
        ),
        contextlib.redirect_stdout(buf),
    ):
        status = ss.main()
    return status, buf.getvalue()


# ------------------------------------------------------------------- slot sweep
SLOT_MODEL = installed_model(
    {
        "verified-install.json": "{}",
        "manifest.json": json.dumps({"expertStride": 4096, "arch": {"numLayers": 48}}),
    }
)
SLOT_LIST = [64, 96, 128]

ANSWERED = {
    "hits": 900,
    "misses": 100,
    "hit_pct": 90.0,
    "read_gib": 1.2,
    "mib_per_token": 3.3,
    "stop": "maxTokens",
    "prompt_tokens": 4500,
    "prefill_s": 3.0,
    "new_tokens": 256,
    "decode_s": 16.0,
    "decode_tok_s": 16.0,
    "max_rss_gib": 12.0,
    "swap_delta_gib": 0.1,
}


def drive_slots(by_slot: dict, argv: list[str] | None = None):
    queues = {s: list(by_slot.get(s, [])) for s in SLOT_LIST}

    def run_once(model, messages, slot_count, max_new):
        row = {
            "slots": slot_count,
            "swap_before_gib": 0.0,
            "swap_after_gib": 0.1,
            "swap_delta_gib": 0.1,
        }
        row.update(queues[slot_count].pop(0) if queues[slot_count] else {"failed": "stub ran out"})
        return row

    buf = io.StringIO()
    with (
        mock.patch.object(slots, "run_once", run_once),
        mock.patch.object(slots, "swap_used_gib", return_value=0.0),
        mock.patch.object(slots, "pgrep_answer", return_value=("clear", [])),
        mock.patch.object(
            sys,
            "argv",
            [
                "expert_cache_slots",
                "--model",
                str(SLOT_MODEL),
                "--rounds",
                "1",
                "--characters",
                "512",
                "--slots",
                ",".join(map(str, SLOT_LIST)),
            ]
            + (argv or []),
        ),
        contextlib.redirect_stdout(buf),
    ):
        status = slots.main()
    return status, buf.getvalue()


# --------------------------------------------------------------------- gate 0
GATE_ROW = {
    "prefill_s": 1.2,
    "decode_s": 16.0,
    "decode_tok_s": 16.0,
    "roles": {"attention": {"gpu_ms": 1.0, "per_token_ms": 0.5, "count": 10}},
    "gaps": ("io", {"total_ms": 1.0, "per_token_ms": 0.2, "count": 10}),
    "busy_per_token_ms": 50.0,
    "occupancy_pct": 88.0,
    "busy_share_of_decode_pct": 70.0,
    "gpu_share_of_decode": 0.7,
    "io_ms": 0.1,
    "wait_ms": 0.05,
    "router_readback_ms": 0.01,
    "cache_plan_ms": 0.02,
    "io_hidden_pct": 40.0,
    "expert_hit_rate": 0.9,
    "completion_sha256": "abcd1234",
}
del GATE_ROW["gaps"]
GATE_ROW["gaps"] = {"io": {"total_ms": 1.0, "per_token_ms": 0.2, "count": 10}}


def drive_gate0(rows: list[dict], argv: list[str] | None = None):
    out = pathlib.Path(tempfile.mkdtemp(prefix="aud274-gate0-")) / "profile.json"

    def run_quant(quant, runs):
        return g0.summarize(quant, rows)

    buf = io.StringIO()
    with (
        mock.patch.object(g0, "preflight", return_value=None),
        mock.patch.object(g0, "_terminate_all", return_value=None),
        mock.patch.object(g0, "run_quant", run_quant),
        mock.patch.object(
            sys, "argv", ["gate0_profile", "--quant", "4bit", "--out", str(out)] + (argv or [])
        ),
        contextlib.redirect_stdout(buf),
    ):
        status = g0.main()
    artifact = json.loads(out.read_text()) if out.exists() else None
    return status, buf.getvalue(), artifact


# ---------------------------------------------------------------- gather probe
def drive_gather(rows: list[dict], argv: list[str] | None = None):
    names = ("dense", "gather")
    by_name = dict(zip(names, rows, strict=False))
    called = []

    def measure(name, geom, repeats, seed):
        called.append(name)
        return by_name.get(name, {})

    GATHER_RESULTS.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    with (
        mock.patch.object(gp, "measure", measure),
        mock.patch.object(gp, "RESULTS", GATHER_RESULTS),
        mock.patch.object(sys, "argv", ["ane_gather_probe"] + (argv or [])),
        contextlib.redirect_stdout(buf),
    ):
        status = gp.main()
    return status, buf.getvalue(), called


class SteadyStateVerdict(unittest.TestCase):
    def test_a_run_set_where_every_run_failed_is_a_refusal_not_an_unavailable_line(self):
        status, out = drive_ss(ss_table(gpu="failed", ane="failed"))
        self.assertEqual(status, 2)
        self.assertIn("NOT MEASURED", out)
        self.assertNotIn("unavailable", out)

    def test_an_arm_that_produced_nothing_names_the_arm_and_the_window(self):
        status, out = drive_ss(ss_table(gpu="ok", ane="failed"))
        self.assertEqual(status, 2)
        self.assertIn("NOT MEASURED: the ane arm", out)
        self.assertIn("[64, 448]", out)

    def test_a_median_over_part_of_an_arm_prints_its_own_denominator(self):
        table = ss_table()
        table[("ane", BIG)] = [ss_row("ane", BIG), ss_row("ane", BIG, failed=True)]
        status, out = drive_ss(table)
        self.assertEqual(status, 1)
        self.assertIn("over 1 of 2", out)

    def test_a_page_built_from_nothing_at_all_does_not_die_on_its_own_header(self):
        status, out = drive_ss({}, ["--reps", "0"])
        self.assertEqual(status, 2)
        self.assertIn("no run", out.lower())

    def test_a_differenced_window_that_did_not_grow_is_a_contested_claim(self):
        status, out = drive_ss(ss_table(ane="reverse"))
        self.assertEqual(status, 1)
        self.assertIn("did not outlast", out)
        self.assertNotIn("NOT MEASURED", out)

    def test_runs_that_all_stopped_early_are_named_even_though_they_answered(self):
        status, out = drive_ss(ss_table(gpu="incomplete", ane="incomplete"))
        self.assertEqual(status, 2)
        self.assertIn("complete", out.lower())

    def test_a_measured_sweep_still_passes(self):
        status, out = drive_ss(ss_table())
        self.assertEqual(status, 0)
        self.assertIn("steady state", out)
        self.assertIn("sweep status 0", out)


class SlotSweepVerdict(unittest.TestCase):
    def test_a_sweep_where_every_slot_failed_is_a_refusal_with_a_status(self):
        status, out = drive_slots({s: [{"failed": "no footer"}] for s in SLOT_LIST})
        self.assertEqual(status, 2)
        self.assertIn("NOT MEASURED", out)

    def test_a_slot_that_produced_nothing_still_holds_a_row_in_the_table(self):
        status, out = drive_slots(
            {64: [dict(ANSWERED)], 96: [{"failed": "no footer"}], 128: [dict(ANSWERED)]}
        )
        self.assertEqual(status, 2)
        self.assertIn("96", out.split("====" * 2)[-1])
        self.assertIn("NOT MEASURED", out)

    def test_a_metric_no_run_logged_is_a_count_not_an_exception(self):
        thin = {k: v for k, v in ANSWERED.items() if k not in ("hit_pct", "mib_per_token")}
        status, out = drive_slots({s: [dict(thin)] for s in SLOT_LIST})
        self.assertNotIsInstance(out, type(None))
        self.assertEqual(status, 2)
        self.assertIn("hit_pct", out)

    def test_a_metric_logged_as_unknown_prints_its_denominator_instead_of_crashing(self):
        none_io = {**ANSWERED, "mib_per_token": None}
        status, out = drive_slots({s: [dict(none_io)] for s in SLOT_LIST})
        self.assertEqual(status, 2)
        self.assertIn("mib_per_token", out)

    def test_a_partial_group_names_how_many_of_how_many_ran(self):
        status, out = drive_slots(
            {
                64: [dict(ANSWERED), {**ANSWERED, "hit_pct": 80.0}],
                96: [dict(ANSWERED), {"failed": "no footer"}],
                128: [dict(ANSWERED), dict(ANSWERED)],
            },
            ["--rounds", "2"],
        )
        self.assertEqual(status, 1)
        self.assertIn("1 of 2", out)
        self.assertNotIn("NOT MEASURED", out)

    def test_a_sweep_that_ran_no_round_refuses(self):
        status, out = drive_slots({}, ["--rounds", "0"])
        self.assertEqual(status, 2)
        self.assertIn("no run", out.lower())

    def test_a_measured_sweep_still_passes(self):
        status, out = drive_slots(
            {s: [dict(ANSWERED), dict(ANSWERED)] for s in SLOT_LIST}, ["--rounds", "2"]
        )
        self.assertEqual(status, 0)
        self.assertIn("96", out)
        self.assertIn("sweep status 0", out)


class Gate0ReportVerdict(unittest.TestCase):
    def test_a_missing_busy_metric_does_not_become_a_dependency_verdict(self):
        rows = [{k: v for k, v in GATE_ROW.items() if k != "busy_per_token_ms"}]
        status, out, _artifact = drive_gate0(rows)
        self.assertEqual(status, 2)
        self.assertNotIn("DEPENDENCY-STALLED", out)
        self.assertIn("busy_per_token_ms", out)

    def test_a_missing_occupancy_is_named_rather_than_raising_mid_page(self):
        rows = [{k: v for k, v in GATE_ROW.items() if k != "occupancy_pct"}]
        status, out, _artifact = drive_gate0(rows)
        self.assertEqual(status, 2)
        self.assertIn("occupancy_pct", out)
        self.assertIn("VERDICT", out)

    def test_no_measured_run_is_a_refusal_not_a_zero_throughput_profile(self):
        status, out, _artifact = drive_gate0([])
        self.assertEqual(status, 2)
        self.assertNotIn("VERDICT: DEPENDENCY-STALLED", out)
        self.assertNotIn("VERDICT: BANDWIDTH-BOUND", out)

    def test_runs_with_no_output_digest_do_not_certify_identity(self):
        rows = [{k: v for k, v in GATE_ROW.items() if k != "completion_sha256"}]
        status, out, _artifact = drive_gate0(rows)
        self.assertEqual(status, 2)
        self.assertIn("digest", out)

    def test_two_runs_that_disagreed_on_their_bytes_are_contested(self):
        rows = [dict(GATE_ROW), {**GATE_ROW, "completion_sha256": "beef5678"}]
        status, out, _artifact = drive_gate0(rows)
        self.assertEqual(status, 1)
        self.assertIn("RUNS DISAGREE", out)

    def test_a_fully_logged_run_set_still_reaches_a_track_verdict(self):
        status, out, _artifact = drive_gate0([dict(GATE_ROW), dict(GATE_ROW)])
        self.assertEqual(status, 0)
        self.assertIn("VERDICT: BANDWIDTH-BOUND", out)


class GatherProbeVerdict(unittest.TestCase):
    ANSWERED_PAIR = [
        {
            "convert_seconds": 1.0,
            "ane_predict_seconds": 0.5,
            "ane_load_seconds": 1.0,
            "cpu_predict_seconds": 0.4,
        },
        {
            "convert_seconds": 1.0,
            "ane_predict_seconds": 30.0,
            "ane_load_seconds": 1.2,
            "cpu_predict_seconds": 0.9,
        },
    ]

    def test_arms_that_recorded_errors_still_get_a_named_refusal_line(self):
        errored = [
            {"convert_seconds": 1.0, "ane_error": "StatisticsError: "},
            {"convert_seconds": 1.0, "ane_error": "StatisticsError: "},
        ]
        status, out, _called = drive_gather(errored)
        self.assertEqual(status, 2)
        self.assertIn("ane_predict_seconds", out)
        self.assertIn("StatisticsError", out)

    def test_one_arm_that_never_answered_is_named_and_costs_the_run(self):
        rows = [dict(self.ANSWERED_PAIR[0]), {"convert_seconds": 1.0, "ane_error": "boom"}]
        status, out, _called = drive_gather(rows)
        self.assertEqual(status, 2)
        self.assertIn("gather", out)
        self.assertNotIn("gather/dense prediction", out)

    def test_a_zero_second_reference_is_a_named_refusal_not_a_skipped_ratio(self):
        dense = {**self.ANSWERED_PAIR[0], "ane_predict_seconds": 0.0, "ane_load_seconds": 0.0}
        status, out, _called = drive_gather([dense, dict(self.ANSWERED_PAIR[1])])
        self.assertEqual(status, 1)
        self.assertIn("divide by zero", out)
        self.assertNotIn("gather/dense prediction", out)

    def test_an_error_alongside_a_measured_ratio_is_contested_not_clean(self):
        dense = {**self.ANSWERED_PAIR[0], "plan_error": "PassPipelineError: x"}
        gather = dict(self.ANSWERED_PAIR[1])
        status, out, _called = drive_gather([dense, gather])
        self.assertEqual(status, 1)
        self.assertIn("plan_error", out)
        self.assertIn("gather/dense prediction", out)

    def test_zero_repeats_refuses_before_any_arm_is_measured(self):
        status, out, called = drive_gather([], ["--repeats", "0"])
        self.assertEqual(status, 2)
        self.assertIn("repeats", out)
        self.assertEqual(called, [])

    def test_a_measured_pair_still_prints_both_ratios(self):
        status, out, _called = drive_gather([dict(r) for r in self.ANSWERED_PAIR])
        self.assertEqual(status, 0)
        self.assertIn("gather/dense prediction  60.00x", out)


class StatusReachesTheCaller(unittest.TestCase):
    """The wiring claim: `main()` is the status, not a bare `return 0`."""

    def test_the_steady_state_pages_status_is_what_the_driver_returns(self):
        table = ss_table(gpu="ok", ane="failed")
        lines, page_status = ss.verdict(ss_pairs(gpu="ok", ane="failed"), SMALL, BIG, "model")
        status, out = drive_ss(table)
        self.assertEqual(status, page_status)
        self.assertIn(lines[-1].strip(), out)

    def test_the_slot_sweep_pages_status_is_what_the_driver_returns(self):
        footprint = {s: 4096 * 48 * s / 1_073_741_824 for s in SLOT_LIST}
        rows = [{"slots": s, **ANSWERED} for s in SLOT_LIST]
        lines, page_status = slots.verdict(rows, SLOT_LIST, footprint, 256, "model")
        status, out = drive_slots({s: [dict(ANSWERED)] for s in SLOT_LIST})
        self.assertEqual(status, page_status)
        self.assertIn(lines[-1].strip(), out)

    def test_the_gate0_pages_status_is_what_the_driver_returns_and_the_artifact_carries(self):
        rows = [dict(GATE_ROW)]
        lines, page_status = g0.report([g0.summarize("4bit", rows)])
        status, out, artifact = drive_gate0(rows)
        self.assertEqual(status, page_status)
        self.assertIn(lines[-1].strip(), out)
        self.assertEqual(artifact["status"], status)

    def test_the_gather_pages_status_is_what_the_driver_returns(self):
        errored = [{"ane_error": "boom"}, {"ane_error": "boom"}]
        lines, page_status = gp.verdict(errored)
        status, out, _called = drive_gather(errored)
        self.assertEqual(status, page_status)
        self.assertIn(lines[-1].strip(), out)


class ArmMetricCountsUnknownValues(unittest.TestCase):
    """The shared primitive has to count a logged-but-unknown value as not measured."""

    def test_a_key_present_with_a_none_value_is_not_counted(self):
        rows = [{"hit_pct": None}, {"hit_pct": 90.0}]
        value, counted, total = prof.arm_metric(rows, "hit_pct")
        self.assertEqual(value, 90.0)
        self.assertEqual((counted, total), (1, 2))

    def test_a_group_whose_only_value_is_none_is_a_count_of_zero_not_a_type_error(self):
        rows = [{"hit_pct": None}, {"hit_pct": None}]
        value, counted, total = prof.arm_metric(rows, "hit_pct")
        self.assertIsNone(value)
        self.assertEqual((counted, total), (0, 2))

    def test_the_median_of_a_real_set_is_unchanged(self):
        rows = [{"decode_tok_s": 9.0}, {"decode_tok_s": 11.0}, {"decode_tok_s": 13.0}]
        value, counted, total = prof.arm_metric(rows, "decode_tok_s")
        self.assertEqual(value, statistics.median([9.0, 11.0, 13.0]))
        self.assertEqual((counted, total), (3, 3))


if __name__ == "__main__":
    unittest.main()
