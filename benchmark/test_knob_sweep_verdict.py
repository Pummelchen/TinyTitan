#!/usr/bin/env python3
"""Tests for `tinytitan_knob_sweep.py`: AUD-277.

The sweep answers which decode knob earns its keep on one model, and its own
docstring states the two things that make an answer trustworthy -- every arm is
compared to a baseline, and the baseline is re-run last so a machine that got
quieter cannot be read as a win: "If the two baselines disagree by more than the
smallest win claimed, the sweep is inconclusive and says so."

Pre-fix, none of that reached the exit status. `main()` ended on `return 0`
whatever it had measured, and the sentence it printed after computing the drift
compared the drift to nothing. Measured on the committed tree by calling the real
`main()` with `run_arm` stubbed and the process guard replaced -- no model, no
server, no socket, nothing fetched:

1. every arm failed: two `FAILED` rows, exit 0;
2. `--arms bse` (a typo): zero arms ran, no output at all, exit 0;
3. `--arms base,bse`: one arm ran and the typo vanished silently, exit 0;
4. no run logged a runner-stats line: the published table read
   `hit% 0.0  io_hid% 0.0  io_ms 0.0  evict 0  waits 0` for every arm and the
   live line `hit=0.0%` -- an explanation column that was never measured
   published as a rate of zero, which reads as "the cache never hit";
5. one of the five counters absent while the others measured: the missing column
   printed `0.0` beside a real `87.4` hit rate;
6. `base` failed while the other arms measured: every `vs base` cell printed
   `--` (correct) and the sweep exited 0, having compared nothing;
7. `base_again` failed: no drift line printed at all, exit 0, so the sweep's own
   noise floor was simply absent from its report;
8. `base_again` at +30.0% against a `+3.0%` claimed win: the page printed
   `baseline drift over the sweep: 30.0%` and
   `Any win smaller than this is inside the noise and is not a win.` and exited
   0 -- the docstring's inconclusive rule computed, printed, and discarded;
9. `SWEEP_TOKENS=0` and `SWEEP_TOKENS=-5`: accepted, so the "measured" request
   asks for no tokens at all; `SWEEP_TOKENS=abc` dies with a bare `ValueError`
   traceback from module import, before the guard and before any refusal line.

The statuses are the three this tree's drivers already use (AUD-273, AUD-274,
AUD-275):

    0  every arm ran, every published cell came from a run that logged it, and
       the drift control bounds the wins the page claims
    1  it measured, and something the page claims is contested -- with the claim named
    2  the headline could not be computed, with the reason named

`3` stays the process guard's own answer, which is what `main()` already returns
when another model process is running. Refusal tokens follow the rule AUD-276
landed: a table cell uses the token that table already uses (`--`, which is what
`vs base` printed for a missing baseline), a live line uses the words
(`not logged`).
"""

from __future__ import annotations

import contextlib
import io
import os
import pathlib
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))

import tinytitan_knob_sweep as ks  # noqa: E402

# The five counters the published table prints. `wait_ms`, `body_ms`,
# `cache_plan_ms` and `rdadvise_ms` are parsed but never printed, so no cell of
# this table claims them and their absence is not a refusal.
CELLS = ("expert_hit_rate", "io_hidden_pct", "io_ms", "expert_evictions", "io_host_waits")
TRIO = ("base", "sync_event", "base_again")
COUNTERS = {
    "expert_hit_rate": 0.874,
    "io_hidden_pct": 18.9,
    "io_ms": 3.2,
    "expert_evictions": 4.0,
    "io_host_waits": 25.7,
}


def arm(name, rate=4.0, **changes):
    """One measured arm row. A change of `None` removes the key, which is how a
    counter the server never printed is shaped."""
    row = {"ok": True, "tok_s": rate, **COUNTERS}
    for key, value in changes.items():
        if value is None:
            row.pop(key)
        else:
            row[key] = value
    return {"arm": name, **row}


def failed(name, note="server not healthy"):
    return {"arm": name, "ok": False, "note": note}


def selection(names):
    """Arm names in sweep order, the way `main()` holds its selection."""
    wanted = set(names)
    return [a[0] for a in ks.ARMS if a[0] in wanted]


def row_line(out, name):
    for line in out.splitlines():
        if line.startswith(f"{name:<15}") or line.startswith(name):
            return line.split()
    raise AssertionError(f"no {name} row in:\n{out}")


def drive(rows, argv, environ=None, answer=None):
    """The real `main()`, with only the arm runner and the process guard replaced.
    `answer` is what the guard reports; a run of it is recorded in `seen`, so a
    refusal shape can prove nothing was launched *and* no server was asked for."""
    answer = answer or ("clear", [])
    seen = []

    def fake_run_arm(name, env_delta, tokens):
        seen.append((name, tokens))
        for row in rows:
            if row["arm"] == name:
                return dict(row)
        raise AssertionError(f"stub asked for unplanned arm {name}")

    def guard(*args, **kwargs):
        seen.append(("guard", None))
        return answer

    buffer = io.StringIO()
    env = dict({"SWEEP_TOKENS": "256"}, **(environ or {}))
    arms = [a for a in ks.ARMS if a[0] in {r["arm"] for r in rows}]
    with (
        mock.patch.object(ks, "run_arm", fake_run_arm),
        mock.patch.object(ks, "pgrep_answer", guard),
        mock.patch.object(ks, "ARMS", arms),
        mock.patch.dict(os.environ, env, clear=True),
        mock.patch.object(sys, "argv", ["tinytitan_knob_sweep.py", *argv]),
        contextlib.redirect_stdout(buffer),
        contextlib.redirect_stderr(buffer),
    ):
        status = ks.main()
    return status, buffer.getvalue(), seen


class MeasuredTokens(unittest.TestCase):
    def test_the_default_is_the_shipped_length(self):
        self.assertEqual(ks.measured_tokens({}), (256, None))

    def test_a_positive_value_is_used_as_given(self):
        self.assertEqual(ks.measured_tokens({"SWEEP_TOKENS": "64"}), (64, None))

    def test_zero_asks_for_no_measured_tokens_and_refuses_by_name(self):
        tokens, reason = ks.measured_tokens({"SWEEP_TOKENS": "0"})
        self.assertIsNone(tokens)
        self.assertIn("SWEEP_TOKENS=0", reason)

    def test_a_negative_value_refuses_too(self):
        tokens, reason = ks.measured_tokens({"SWEEP_TOKENS": "-5"})
        self.assertIsNone(tokens)
        self.assertIn("SWEEP_TOKENS=-5", reason)

    def test_an_unparseable_value_refuses_instead_of_raising(self):
        tokens, reason = ks.measured_tokens({"SWEEP_TOKENS": "abc"})
        self.assertIsNone(tokens)
        self.assertIn("abc", reason)


class SelectedArms(unittest.TestCase):
    def test_an_empty_selection_is_every_arm_in_sweep_order(self):
        arms, unknown = ks.selected_arms("")
        self.assertEqual([a[0] for a in arms], [a[0] for a in ks.ARMS])
        self.assertEqual(unknown, [])

    def test_a_selection_keeps_sweep_order_not_flag_order(self):
        arms, unknown = ks.selected_arms("base_again,base,sync_event")
        self.assertEqual([a[0] for a in arms], ["base", "sync_event", "base_again"])
        self.assertEqual(unknown, [])

    def test_a_name_that_matches_no_arm_comes_back_as_unknown(self):
        arms, unknown = ks.selected_arms("bse")
        self.assertEqual(arms, [])
        self.assertEqual(unknown, ["bse"])

    def test_a_typo_next_to_a_real_name_is_still_named(self):
        arms, unknown = ks.selected_arms("base,bse")
        self.assertEqual([a[0] for a in arms], ["base"])
        self.assertEqual(unknown, ["bse"])


class TableCell(unittest.TestCase):
    def test_a_measured_row_prints_the_cells_it_always_printed(self):
        # The control: telling an unlogged figure from a zero costs a measured
        # row nothing at all.
        rows = [arm("base", 4.0), arm("sync_event", 4.12)]
        out = ks.table(rows, 4.0)
        self.assertIn(
            "sync_event         4.12    +3.0%   87.4     18.9     3.2       4      26",
            out,
        )

    def test_a_counter_no_run_logged_is_not_published_as_a_zero(self):
        out = ks.table([arm("io_metal", 4.0, **dict.fromkeys(CELLS, None))], 4.0)
        self.assertEqual(
            row_line(out, "io_metal"),
            ["io_metal", "4.00", "+0.0%", "--", "--", "--", "--", "--"],
        )

    def test_one_missing_column_does_not_zero_the_ones_that_answered(self):
        row = arm("sync_event", 4.12, io_hidden_pct=None, expert_evictions=None)
        out = ks.table([arm("base", 4.0), row], 4.0)
        self.assertEqual(
            row_line(out, "sync_event"),
            ["sync_event", "4.12", "+3.0%", "87.4", "--", "3.2", "--", "26"],
        )

    def test_the_header_still_names_every_column(self):
        head = ks.table([], None).splitlines()[0]
        for column in ("arm", "tok/s", "vs base", "hit%", "io_hid%", "io_ms", "evict", "waits"):
            self.assertIn(column, head)

    def test_the_live_line_refuses_with_the_words_not_a_zero(self):
        rows = [arm("sync_event", 4.0, **dict.fromkeys(CELLS, None))]
        status, out, _ = drive(rows, ["--arms", "sync_event"])
        line = next(line for line in out.splitlines() if "tok/s" in line)
        self.assertIn("hit=not logged", line)
        self.assertIn("io_ms=not logged", line)
        self.assertNotIn("hit=0.0%", line)

    def test_the_live_line_still_prints_a_counter_that_answered(self):
        # Control on the same line: 87.4% is a measurement and must not vanish.
        rows = [arm("sync_event", 4.0, io_hidden_pct=None)]
        status, out, _ = drive(rows, ["--arms", "sync_event"])
        line = next(line for line in out.splitlines() if "tok/s" in line)
        self.assertIn("hit=87.4%", line)
        self.assertIn("io_ms=3.2", line)
        self.assertIn("io_hidden=not logged", line)


class VerdictStatus(unittest.TestCase):
    def test_a_sweep_where_nothing_measured_refuses(self):
        rows = [failed("base"), failed("sync_event"), failed("base_again")]
        lines, status = ks.verdict(rows, selection(TRIO))
        self.assertEqual(status, 2, "\n".join(lines))
        self.assertIn("NOT MEASURED: no arm of this sweep measured anything", "\n".join(lines))

    def test_a_clean_sweep_whose_drift_bounds_its_wins_is_zero(self):
        rows = [arm("base", 4.00), arm("sync_event", 4.12), arm("base_again", 4.02)]
        lines, status = ks.verdict(rows, selection(TRIO))
        out = "\n".join(lines)
        self.assertEqual(status, 0, out)
        self.assertIn("baseline drift over the sweep: 0.5%", out)

    def test_drift_larger_than_the_smallest_claimed_win_is_named_and_contested(self):
        rows = [arm("base", 4.00), arm("sync_event", 4.12), arm("base_again", 5.20)]
        lines, status = ks.verdict(rows, selection(TRIO))
        out = "\n".join(lines)
        self.assertEqual(status, 1, out)
        self.assertIn("INCONCLUSIVE: baseline drift 30.0%", out)
        self.assertIn("+3.0% on sync_event", out)

    def test_a_drift_that_swallows_no_claim_is_not_a_refusal(self):
        # Control: nothing on the page claims a win, so there is nothing for the
        # drift to invalidate and the sweep still has to be able to say 0.
        rows = [arm("base", 4.00), arm("sync_event", 3.80), arm("base_again", 5.20)]
        lines, status = ks.verdict(rows, selection(TRIO))
        out = "\n".join(lines)
        self.assertEqual(status, 0, out)
        self.assertIn("no arm claims a win", out)

    def test_an_arm_that_measures_the_baseline_rate_claims_no_win(self):
        # The boundary under `>`: an arm that lands exactly on the baseline is
        # not a claim, so it cannot be the smallest win the drift swallows.
        rows = [arm("base", 4.00), arm("sync_event", 4.00), arm("base_again", 4.20)]
        lines, status = ks.verdict(rows, selection(TRIO))
        out = "\n".join(lines)
        self.assertEqual(status, 0, out)
        self.assertIn("no arm claims a win", out)

    def test_the_smallest_claimed_win_bounds_the_sweep_not_the_largest(self):
        # The rule is the drift against the *smallest* win claimed. Reading the
        # largest instead would call this page conclusive because one arm
        # claimed +12.0%, while the +3.0% claim sits under a 5.0% floor.
        rows = [
            arm("base", 4.00),
            arm("sync_event", 4.12),
            arm("submit_now", 4.48),
            arm("base_again", 4.20),
        ]
        lines, status = ks.verdict(rows, selection(TRIO + ("submit_now",)))
        out = "\n".join(lines)
        self.assertEqual(status, 1, out)
        self.assertIn("INCONCLUSIVE: baseline drift 5.0%", out)
        self.assertIn("+3.0% on sync_event", out)

    def test_a_drift_exactly_equal_to_the_smallest_win_is_not_larger_than_it(self):
        # "disagree by more than the smallest win" is a `>`, and the page that
        # bounds its claims must not overstate it: 5.0% of drift against a 5.0%
        # win is neither inconclusive nor "larger than".
        rows = [arm("base", 4.00), arm("sync_event", 4.20), arm("base_again", 4.20)]
        lines, status = ks.verdict(rows, selection(TRIO))
        out = "\n".join(lines)
        self.assertEqual(status, 0, out)
        self.assertIn("smallest win claimed +5.0% on sync_event, against 5.0% of drift", out)
        self.assertNotIn("larger", out)

    def test_a_failed_arm_while_others_measured_is_named_and_contested(self):
        rows = [arm("base", 4.0), failed("sync_event", "BrokenPipe"), arm("base_again", 4.01)]
        lines, status = ks.verdict(rows, selection(TRIO))
        out = "\n".join(lines)
        self.assertEqual(status, 1, out)
        self.assertIn("CONTESTED: sync_event did not measure (BrokenPipe)", out)

    def test_no_baseline_measured_makes_the_whole_comparison_unmeasured(self):
        rows = [failed("base", "disk full"), arm("sync_event", 4.12), arm("base_again", 4.02)]
        lines, status = ks.verdict(rows, selection(TRIO))
        out = "\n".join(lines)
        self.assertEqual(status, 2, out)
        self.assertIn("NOT MEASURED: no baseline arm measured", out)

    def test_a_selection_without_a_baseline_is_a_refusal_not_an_empty_delta(self):
        rows = [arm("sync_event", 4.12)]
        lines, status = ks.verdict(rows, selection(("sync_event",)))
        out = "\n".join(lines)
        self.assertEqual(status, 2, out)
        # The cause is the selection, not a baseline that failed to come up: the
        # operator asked for one arm and nothing was there to compare it to. A
        # looser "no baseline" assertion let M11 survive on the other branch's
        # message, so the exact reason is what is pinned.
        self.assertIn("NOT MEASURED: the selection has no baseline arm to compare to", out)

    def test_a_selected_drift_control_that_failed_refuses_the_page(self):
        rows = [arm("base", 4.0), arm("sync_event", 4.12), failed("base_again", "server died")]
        lines, status = ks.verdict(rows, selection(TRIO))
        out = "\n".join(lines)
        self.assertEqual(status, 2, out)
        self.assertIn("base_again", out)
        self.assertNotIn("Any win smaller than this", out)

    def test_a_selection_without_the_drift_control_says_its_floor_is_unknown(self):
        rows = [arm("base", 4.0), arm("sync_event", 4.12)]
        lines, status = ks.verdict(rows, selection(("base", "sync_event")))
        out = "\n".join(lines)
        self.assertEqual(status, 1, out)
        self.assertIn("noise floor", out)
        self.assertNotIn("Any win smaller than this", out)

    def test_a_printed_counter_no_arm_logged_refuses_the_page(self):
        missing = dict.fromkeys(CELLS, None)
        rows = [
            arm("base", 4.00, **missing),
            arm("sync_event", 4.12, **missing),
            arm("base_again", 4.02, **missing),
        ]
        lines, status = ks.verdict(rows, selection(TRIO))
        out = "\n".join(lines)
        self.assertEqual(status, 2, out)
        self.assertIn("NOT MEASURED: the base arm logged no io_hidden_pct", out)

    def test_a_counter_missing_from_one_arm_of_three_is_named_by_arm(self):
        rows = [
            arm("base", 4.00),
            arm("sync_event", 4.12, io_host_waits=None),
            arm("base_again", 4.02),
        ]
        lines, status = ks.verdict(rows, selection(TRIO))
        out = "\n".join(lines)
        self.assertEqual(status, 2, out)
        self.assertIn("the sync_event arm logged no io_host_waits", out)
        self.assertNotIn("the base arm logged no", out)


class MainWiring(unittest.TestCase):
    def test_the_sweep_returns_its_verdict_instead_of_zero(self):
        rows = [arm("base", 4.00), arm("sync_event", 4.12), arm("base_again", 5.20)]
        status, out, seen = drive(rows, ["--arms", "base,sync_event,base_again"])
        self.assertEqual(status, 1, out)
        self.assertIn("INCONCLUSIVE", out)
        self.assertEqual(
            [name for name, tokens in seen if name != "guard"],
            ["base", "sync_event", "base_again"],
        )

    def test_every_arm_failing_is_a_refusal_not_a_clean_exit(self):
        rows = [failed("base"), failed("sync_event"), failed("base_again")]
        status, out, _ = drive(rows, ["--arms", "base,sync_event,base_again"])
        self.assertEqual(status, 2, out)

    def test_a_typo_in_arms_refuses_before_the_guard_and_before_any_arm(self):
        status, out, seen = drive([arm("base", 4.0)], ["--arms", "base,bse"])
        self.assertEqual(status, 2, out)
        self.assertIn("bse", out)
        self.assertEqual(seen, [])

    def test_a_zero_token_count_refuses_before_the_guard_and_before_any_arm(self):
        status, out, seen = drive([arm("base", 4.0)], [], environ={"SWEEP_TOKENS": "0"})
        self.assertEqual(status, 2, out)
        self.assertIn("SWEEP_TOKENS=0", out)
        self.assertEqual(seen, [])

    def test_an_unparseable_token_count_refuses_without_a_traceback(self):
        status, out, seen = drive([arm("base", 4.0)], [], environ={"SWEEP_TOKENS": "abc"})
        self.assertEqual(status, 2, out)
        self.assertIn("abc", out)
        self.assertNotIn("Traceback", out)
        self.assertEqual(seen, [])

    def test_the_measured_token_length_reaches_the_arm_runner(self):
        rows = [arm("base", 4.00), arm("sync_event", 4.12), arm("base_again", 4.02)]
        status, out, seen = drive(
            rows, ["--arms", "base,sync_event,base_again"], environ={"SWEEP_TOKENS": "96"}
        )
        self.assertEqual(status, 0, out)
        self.assertEqual({tokens for name, tokens in seen if name != "guard"}, {96})

    def test_the_process_guard_still_answers_three(self):
        rows = [arm("base", 4.00), arm("sync_event", 4.12), arm("base_again", 4.02)]
        status, out, seen = drive(rows, [], answer=("busy", ["17514 TinyTitanServer"]))
        self.assertEqual(status, 3, out)
        self.assertIn("17514", out)
        self.assertEqual(seen, [("guard", None)])


if __name__ == "__main__":
    unittest.main()
