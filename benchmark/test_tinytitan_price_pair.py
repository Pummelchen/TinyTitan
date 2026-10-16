#!/usr/bin/env python3
"""Tests for `tinytitan_price_pair.py`: #207.

The two-price model says a paired decode arm costs `x` ms per issued speculative
read and `y` ms per adopted one, so `dm = x*du + y*dv` for a row
`(du, dv) = (dissued, -dadopted)`. §5as chose which arms to run by the *angle*
between two such rows, §5at re-derived that geometry from published counters, and
both left the arithmetic itself in a throwaway. This screen closes that seam: it
reads the page `tinytitan_knob_sweep.py` prints and answers `x`, `y`, the
condition number and the band verdict with an exit status.

Nine things it must not be able to get wrong, each pinned below:

1. **The sign.** Rows live in the (`dissued`, `-dadopted`) plane, so the second
   coordinate is positive where a step makes adopted reads *fall*. §5at found the
   mis-signed pair collapses from 41.4 degrees to 9.75 -- the campaign's best
   design turned into one of its worst -- so the convention is asserted, not
   implied.
2. **Which table of a page is the page.** The sweep prints its table once per arm,
   cumulatively, so a four-arm batch emits four tables and only the last is the
   batch. Solving off the first prices one arm against nothing.
3. **The degenerate row.** `base_again` is the drift control and its counters
   equal the baseline's, so its row is the zero vector and any pair containing it
   is singular.
4. **What an out-of-band price means.** `y` outside 0.788-1.301 ms is not a
   measurement of an expensive read; it is the two-price model failing, and the
   screen says so with a status of its own.
5. **How many digits the page actually carries.** The sweep prints `tok/s` to two
   decimals and §5at quoted its prices from rates with three, so the same trio
   solves `y=1.598` off a sweep page and `y=1.585` off the published one. The
   screen has to bound its own answer by the digits it was given, and refuse to
   decide a band call that falls inside that bound.
6. **That conditioning is not signal (#208).** The angle bounds how well a pair
   is posed and the digits bound how well it is known; neither bounds whether the
   row moved. On §5at's real six-row page the widest pair (84.25 deg, `y=18.189`)
   rested on a row whose whole step was inside the same page's drift control, so
   the control is never a partner and a row inside its step is not signal -- and
   every pair of surviving rows is a candidate, not a window over sorted names.
7. **That one page may hold two answers.** After the bar that page prices the
   same adopted read at 0.796 ms (inside the band) and 0.570 ms (below it). A
   screen that prints one number and a status cannot say the page is the problem;
   designs that land on opposite sides of a band edge refuse it.
8. **That a price is only known as well as the machine is quiet (#209).** The
   drift sample bounds how far a row moves with no config change, which is a width
   on the *solve*, not only a gate on the row list: `y ±0.034 ms` printed next to a
   control that just moved a row 11.17 ms is a bound on the digits and not on the
   answer. A `p*` whose own width spans the gate's `L/S` cannot call either side,
   and a held-out row's residual is the only test of the model that is not Cramer
   agreeing with itself — so both are scored, with statuses.
9. **That an unmeasured drift is not a zero one (#215).** A page with no
   re-measured baseline put 0.0 into every drift width, which is the number a page
   whose control moved *nothing* prints, and it then priced `y=1.002`, called it
   admissible, drew §5ar's six-link gate and exited 0 on digits alone — a pass paid
   out for the channel it never read, on a Mac whose measured drift (§5az: the gate
   wants a baseline repeating to 0.75 ms) is an order of magnitude above its
   decimals. Such a page still prices and still classifies, because #208 ruled a
   missing control is not a refusal to solve, but it draws no gate, and its caveat
   names the row it lacks instead of promising decimals it cannot spend on a half
   it has not seen. The refusal is added after #209's rather than returned in front
   of it: a page can carry both defects, and the interval sentence is a finding
   about that page which the license rule has no business eating.

The statuses are the ones this tree's drivers already use, widened by #209 at the
second one because a page can now solve a price and still not be allowed to spend
it, and again by #215 because a page that never read its drift may not spend it either:

    0  the design identifies, `y` is inside the band, and the gate is drawn
    1  it identified a price, and that price or a held-out row refutes the model
    2  it could not solve, or the page cannot license a gate draw — its own widths
       straddle the threshold, or it sampled no drift — with the reason named

Every fixture page is built by calling the real `ks.table()`, so these tests fail
the moment the page the sweep prints and the page this screen reads stop
agreeing. No model, no server, no socket, nothing fetched.
"""

from __future__ import annotations

import io
import pathlib
import re
import sys
import unittest
from contextlib import redirect_stdout

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))

import tinytitan_knob_sweep as ks  # noqa: E402
import tinytitan_price_pair as pp  # noqa: E402

# §5ag's shipped-tier ring-on row, used as the baseline of every fixture so the
# prices solved here are comparable with the ones §5at published. The band bounds
# are §5ar's, quoted verbatim from that section's table.
BAND_TEXT = "0.788-1.301 ms"


def arm(name, tok_s, issued, adopted):
    """One row of the sweep's own table, as far as this screen reads it."""
    return {
        "arm": name,
        "ok": True,
        "tok_s": tok_s,
        "expert_hit_rate": 0.8459,
        "io_hidden_pct": 18.9,
        "io_ms": 68.3,
        "expert_evictions": 40000.0,
        "io_host_waits": 25.7,
        "prefetch_issued_per_token": issued,
        "prefetch_adopted_per_token": adopted,
    }


def page(*arms):
    return ks.table(list(arms), arms[0]["tok_s"])


# The identifying trio of §5as's reopen. Its counters are that section's published
# deltas (+33.28 issued, -9.42 adopted) against this baseline, and `base_again` is
# in the page with the baseline's own counters, so the drift control is a zero row
# and the best-pair search has to step over it. The rates are §5ag's, which carry
# three decimals, but `ks.table()` prints two -- so this page's depth step reads
# 30.42 ms/token where §5at quoted 30.23 from the rates themselves.
RING_ON = arm("base", 5.079, 39.45, 18.86)
RING_OFF = arm("prefetch_off", 4.793, 0.00, 0.00)
DEPTH25 = arm("layer_depth25", 4.403, 72.73, 9.44)
CONTROL = arm("base_again", 5.043, 39.45, 18.86)
TRIO = page(RING_ON, RING_OFF, DEPTH25, CONTROL)

# The same trio at the precision §5at solved it, as a measurements-doc table. This
# is the page whose prices are published, so it is the page that pins them.
DOC3 = """
| arm | tok/s | issued/pt | adopted/pt |
|---|---|---|---|
| base | 5.079 | 39.45 | 18.86 |
| prefetch_off | 4.793 | 0.00 | 0.00 |
| layer_depth25 | 4.403 | 72.73 | 9.44 |
| base_again | 5.043 | 39.45 | 18.86 |
"""

# One decimal: the half-ulp is 0.05 tok/s, ~4.5 ms/token on a step, and the solve
# lands 0.06 ms outside the band's top edge -- well inside its own resolution.
DOC1 = """
| arm | tok/s | issued/pt | adopted/pt |
|---|---|---|---|
| base | 5.1 | 39.45 | 18.86 |
| prefetch_off | 5.0 | 0.00 | 0.00 |
| layer_depth25 | 4.4 | 72.73 | 9.44 |
"""

# The in-band trio of `test_an_in_band_y_is_admissible_and_the_gate_is_scored_from_it`,
# rounded to one decimal like a human restating it in prose: it still solves inside
# the band (17.39 and 12.77 ms of step -> y = 1.004, x = 0.238), but now 0.05 tok/s of
# half-ulp is 4.3 ms of step and the band puts only 0.216 ms between y and the near
# edge. The clause has to fire on the side of the band that agrees with the model, too
# -- a caveat that only ever prints next to a refutation is a caveat that has never
# been tested. And because the same 4.3 ms reaches the ratio, x carries ±0.121 and y
# ±0.283, so p* runs 0.091 to 0.498 across the gate's 0.155: #209 made that a refusal.
DOC1_IN = """
| arm | tok/s | issued/pt | adopted/pt |
|---|---|---|---|
| base | 5.0 | 42.75 | 22.85 |
| prefetch_off | 4.7 | 0.00 | 0.00 |
| layer_depth25 | 4.6 | 76.03 | 13.43 |
"""

# The in-band trio `test_an_in_band_y_is_admissible_and_the_gate_is_scored_from_it`
# was built on, twice: once as the page stands (three decimals, no re-measured
# baseline, so the drift is unmeasured) and once with `base_again` carrying base's
# counters verbatim. That row is the difference between the two: it is the page's
# only measurement of how far the machine moves a row that did not change, and its
# step of exactly zero is a MEASURED zero, which is not the same statement as the
# first page's silence. #215.
INBAND_NO_CONTROL = """
| arm | tok/s | issued/pt | adopted/pt |
|---|---|---|---|
| base | 5.000 | 42.75 | 22.85 |
| prefetch_off | 4.665 | 0.00 | 0.00 |
| layer_depth25 | 4.628 | 76.03 | 13.43 |
"""

INBAND = """
| arm | tok/s | issued/pt | adopted/pt |
|---|---|---|---|
| base | 5.000 | 42.75 | 22.85 |
| prefetch_off | 4.665 | 0.00 | 0.00 |
| layer_depth25 | 4.628 | 76.03 | 13.43 |
| base_again | 5.000 | 42.75 | 22.85 |
"""

# DOC1_IN again with the zero control added: a page whose drift IS sampled, and
# sampled at nothing. Its bound is still the printing, and the printing is still
# what would narrow it -- which is the sentence #215 must keep saying here and stop
# saying on a page that never read the channel.
DOC1_IN_SAMPLED = """
| arm | tok/s | issued/pt | adopted/pt |
|---|---|---|---|
| base | 5.0 | 42.75 | 22.85 |
| prefetch_off | 4.7 | 0.00 | 0.00 |
| layer_depth25 | 4.6 | 76.03 | 13.43 |
| base_again | 5.0 | 42.75 | 22.85 |
"""

# §5at's own rendered page: six arms, and the first table this screen has ever
# read that was assembled from two batches (§5ag's ring-off row, §5aq's five
# others). Rates and counters are that section's, printed as the sweep prints
# them. §5au scored it by hand and found the selection rule picking the 84.25 deg
# pair, which rests on `slots_64`'s step of (-0.03, +0.21) and 3.93 ms -- every
# one of those inside what the same page's `base_again` moved with no config
# change at all, (-0.18, +1.80) and 11.17 ms. This page is the bug report.
PAGE5AT = page(
    arm("base", 4.55, 42.75, 22.85),
    arm("slots_64", 4.47, 42.72, 22.64),
    arm("layer_depth25", 4.00, 76.03, 13.43),
    arm("slots64_depth25", 4.11, 76.22, 14.58),
    arm("base_again", 4.33, 42.57, 21.05),
    arm("prefetch_off", 4.79, 0.00, 0.00),
)

# The same page with `slots64_depth25` removed, so after the low-signal row is
# stepped over exactly one design survives: `layer_depth25 x prefetch_off`, the
# 43.93 deg pair §5au prices y=0.796 -- inside the band. One design cannot
# disagree with itself, so this is the page that has to price something.
PAGE5AT_ONE = page(
    arm("base", 4.55, 42.75, 22.85),
    arm("slots_64", 4.47, 42.72, 22.64),
    arm("layer_depth25", 4.00, 76.03, 13.43),
    arm("base_again", 4.33, 42.57, 21.05),
    arm("prefetch_off", 4.79, 0.00, 0.00),
)

# A page whose drift control moved MORE than every arm it is pooled with, so the
# signal bar would keep it, and whose widest pair is the control x an arm. The
# control is the baseline re-measured: its row is drift by construction, and a
# solve that rests on it is pricing the machine and calling it a config change.
BIG_DRIFT = page(
    arm("base", 4.55, 42.75, 22.85),
    arm("base_again", 4.30, 30.75, 10.85),
    arm("layer_depth25", 4.00, 76.03, 13.43),
    arm("prefetch_off", 4.79, 0.00, 0.00),
)

# Three rows in alphabetical order, where the widest pair is the first and the
# last. A candidate set built by walking a sorted list two at a time never sees
# it, and "the widest pair a table offers" is not a property of the arm names.
WIDEST_NOT_ADJACENT = page(
    arm("base", 5.000, 40.00, 20.00),
    arm("a_first", 4.900, 50.00, 19.00),
    arm("m_middle", 4.800, 50.00, 10.00),
    arm("z_last", 4.700, 41.00, 10.00),
)

# A page where the drift control moved more than everything else, so the bar has
# no row left to pair. `base_again` moved (+10.00, +10.00) and +22.22 ms; both
# arms move less in all three coordinates. The refusal is the screen's own doing
# here, not the page's, and it has to say which.
ALL_INKED = page(
    arm("base", 5.000, 40.00, 20.00),
    arm("base_again", 4.500, 50.00, 10.00),
    arm("a_in", 4.900, 45.00, 16.00),
    arm("b_in", 4.950, 35.00, 24.00),
)


def holdout_page(rate):
    """A priceable trio plus a fourth arm held out of the solve, priced `rate`.

    The ring-off x depth pair on these rows prices x=0.264 / y=0.946, so the
    held-out row (+50.00, +15.00) is predicted to cost 27.4 ms of token. One
    decimal of `tok/s` resolves that row to under a millisecond, so a page that
    prints it 29 ms slower than its own two prices say is not a rounding
    question -- it is the model failing on a row it never saw. #209 found the
    screen printing that residual and handing over no status for it.

    `base_again` carries base's counters and rate verbatim, so its step is zero
    and the bound stays what the printing gives it. It is here because #215 says
    a page without it may not score the gate: this fixture's whole point is that
    the gate *is* scored on a consistent held-out row, and an unsampled drift
    would refuse that for a reason that has nothing to do with the residual.
    """
    return page(
        arm("base", 5.00, 40.00, 20.00),
        arm("prefetch_off", 4.80, 0.00, 0.00),
        arm("layer_depth25", 4.60, 70.00, 10.00),
        arm("layer_depth35", rate, 90.00, 5.00),
        arm("base_again", 5.00, 40.00, 20.00),
    )


HOLD_BAD = holdout_page(3.90)
HOLD_OK = holdout_page(4.40)


def best_line(out: str) -> str:
    lines = [line for line in out.splitlines() if "best pair" in line]
    if len(lines) != 1:
        raise ValueError(f"expected one 'best pair' line, got {len(lines)} in:\n{out}")
    return lines[0]


def run(text, *argv):
    out = io.StringIO()
    with redirect_stdout(out):
        status = pp.main(list(argv), stdin=text)
    return status, out.getvalue()


class Sign(unittest.TestCase):
    """A step that loses adoptions has a POSITIVE second coordinate.

    The convention is §5ar's own equation E1, `18.86 y - 39.45 x = 11.7 ms`, for
    the ring-on -> ring-off step: adoptions fall 18.86 -> 0, so that row's second
    coordinate is +18.86 and its first is -39.45. §5at's geometry table prints
    that row negated, which reads the same `sin theta` and solves the opposite
    verdict, so `Prices.test_it_solves_the_published_wall_prices_off_a_three_decimal_page`
    pins these coordinates to the `x`/`y` the published prices were solved from.
    """

    def rows(self):
        return pp.rows(pp.read(page(RING_ON, DEPTH25))[0], "base")

    def test_a_step_that_makes_adopted_reads_fall_is_positive(self):
        row = self.rows()["layer_depth25"]
        self.assertAlmostEqual(row.du, 33.28, places=6)
        self.assertAlmostEqual(row.dv, 9.42, places=6)

    def test_the_ring_off_step_also_loses_adoptions_so_its_row_is_e1(self):
        row = pp.rows(pp.read(TRIO)[0], "base")["prefetch_off"]
        self.assertAlmostEqual(row.du, -39.45, places=6)
        self.assertAlmostEqual(row.dv, 18.86, places=6)

    def test_a_step_that_makes_adopted_reads_rise_is_negative(self):
        wider = arm("ring_wider", 4.900, 60.00, 25.14)
        row = pp.rows(pp.read(page(RING_ON, wider))[0], "base")["ring_wider"]
        self.assertAlmostEqual(row.du, 20.55, places=6)
        self.assertAlmostEqual(row.dv, -6.28, places=6)


class ReadingAPage(unittest.TestCase):
    def test_it_reads_the_last_table_of_a_cumulative_sweep_page(self):
        cumulative = "\n".join(page(*[RING_ON, RING_OFF, DEPTH25][:n]) + "\n" for n in (1, 2, 3))
        rounds = pp.read(cumulative)
        self.assertEqual(len(rounds), 1)
        self.assertEqual(sorted(rounds[0].arms), ["base", "layer_depth25", "prefetch_off"])

    def test_two_complete_tables_are_two_rounds_of_one_batch(self):
        self.assertEqual(len(pp.read(TRIO + "\n" + TRIO)), 2)

    def test_a_markdown_table_from_the_measurements_doc_parses(self):
        doc = """
| arm | tok/s | vs base | hit | `io_ms` | issued/pt | adopted/pt |
|---|---|---|---|---|---|---|
| base | 4.55 | -- | 0.7745 | 99.4 | 42.75 | 22.85 |
| layer_depth25 | 4.00 | -12.1 % | 0.7552 | 117.0 | 76.03 | 13.43 |
"""
        row = pp.rows(pp.read(doc)[0], "base")["layer_depth25"]
        self.assertAlmostEqual(row.du, 33.28, places=6)
        self.assertAlmostEqual(row.dv, 9.42, places=6)

    def test_a_page_that_logged_no_issued_column_refuses(self):
        status, out = run(page(arm("base", 5.0, None, None), DEPTH25))
        self.assertEqual(status, 2)
        self.assertIn("NOT MEASURED", out)
        self.assertIn("issued/pt", out)

    def test_a_page_without_the_named_baseline_refuses(self):
        status, out = run(page(RING_OFF, DEPTH25))
        self.assertEqual(status, 2)
        self.assertIn("base", out)


class Geometry(unittest.TestCase):
    def test_the_depth_pair_is_refused_because_its_rows_are_2_6_degrees_apart(self):
        # §5as's own out-of-sample row: same direction as the depth-2 row, so the
        # pair cannot price anything however carefully it is solved.
        depth325 = arm("layer_depth325", 4.238, 94.78, 5.84)
        status, out = run(page(RING_ON, DEPTH25, depth325))
        self.assertEqual(status, 2)
        self.assertIn("REFUSED", out)
        self.assertIn("deg", out)

    def test_the_identifying_pair_is_the_ring_and_the_depth_arm_not_the_control(self):
        status, out = run(TRIO)
        best = [line for line in out.splitlines() if "best pair" in line]
        self.assertEqual(len(best), 1, out)
        self.assertIn("prefetch_off", best[0])
        self.assertIn("layer_depth25", best[0])
        self.assertNotIn("base_again", best[0])
        self.assertRegex(best[0], r"41\.\d deg")
        self.assertRegex(best[0], r"cond 1\.\d")
        self.assertEqual(status, 1, out)  # identified, and the price refutes the model

    def test_the_candidate_set_is_every_pair_of_rows_not_a_window_of_names(self):
        # a_first x z_last is 78.5 deg apart; m_middle sits between the two in
        # sorted order and pairs with each at 39.3. An enumeration that steps a
        # sorted list two at a time can only ever see the 39.3 deg pairs, and
        # "the widest pair a table offers" is not a property of arm-name order.
        _, out = run(WIDEST_NOT_ADJACENT)
        best = best_line(out)
        self.assertIn("a_first", best)
        self.assertIn("z_last", best)
        self.assertRegex(best, r"7[0-9]\.\d deg")


class Signal(unittest.TestCase):
    """A row has to move more than the page's own drift before it is signal.

    The angle bounds the *conditioning* of a design and the digits bound its
    precision; neither bounds whether the row moved at all. §5au showed a real
    page where the widest pair rested on a row whose whole step -- both counters
    and the wall -- was smaller than the drift control's, and priced an adopted
    read at 18.189 ms because of it. `base_again` is the same configuration
    re-measured, so its step IS this page's drift sample, and it is not an arm.
    """

    def test_a_row_inside_the_page_s_own_drift_sample_is_not_signal(self):
        status, out = run(PAGE5AT)
        self.assertIn("signal: drift sample base_again", out)
        self.assertIn("not signal, stepped over: slots_64", out)
        self.assertNotIn("18.189", out)
        best = best_line(out)
        self.assertIn("layer_depth25", best)
        self.assertIn("prefetch_off", best)
        self.assertNotIn("slots_64", best)
        self.assertEqual(status, 2, out)  # and what is left refuses the page

    def test_the_drift_control_is_never_a_solve_partner(self):
        # Here the control moved (-12, +12) and 12.8 ms -- more than every arm it
        # is pooled with, so the signal bar keeps it. It is still the baseline
        # re-measured: a pair built on it prices the machine, not a config change.
        # Widest otherwise: base_again x layer_depth25 at 60.8 deg.
        _, out = run(BIG_DRIFT)
        best = best_line(out)
        self.assertNotIn("base_again", best)
        self.assertIn("layer_depth25", best)
        self.assertIn("prefetch_off", best)
        self.assertRegex(best, r"4[0-9]\.\d deg")

    def test_surviving_designs_that_straddle_a_band_edge_refuse_the_page(self):
        # Two designs, 43.9 deg and 42.0 deg, both well-conditioned, both priced
        # off this page: y=0.796 inside the band and y=0.570 below its low edge.
        # One of them is what a cross-batch page says about an adopted read, and
        # which one a single-number rule prints is not evidence for it.
        status, out = run(PAGE5AT)
        self.assertEqual(status, 2, out)
        self.assertIn("do not agree", out)
        self.assertIn("y=0.796", out)
        self.assertIn("y=0.570", out)
        self.assertNotIn("ADMISSIBLE", out)
        self.assertNotIn("p*=", out)

    def test_a_page_without_the_control_row_says_it_has_no_drift_sample(self):
        # No drift sample is not a refusal: the bar has nothing to key to, so the
        # screen says so and solves, and the price is the one §5au printed for it.
        status, out = run(DOC1)
        self.assertIn("signal: no drift sample", out)
        self.assertIn("not on this page", out)
        self.assertIn("y=1.362", out)
        self.assertEqual(status, 1, out)

    def test_one_surviving_design_prices_exactly_as_it_did_unbarred(self):
        # The bar changes which rows may be paired; it must not touch the
        # arithmetic. layer_depth25 x prefetch_off on these rows is §5au's
        # x=0.683 / y=0.796, and with one design left there is nothing to
        # disagree about, so the band call stands. #209 amended the STATUS: this
        # page's own control moved a row 11.17 ms, which difference-mode puts 0.832
        # ms on y -- more than y is -- so the gate draw is refused even though the
        # price and its band classification print exactly as before.
        status, out = run(PAGE5AT_ONE)
        self.assertIn("not signal, stepped over: slots_64", out)
        self.assertIn("x=0.683 y=0.796", out)
        self.assertIn("ADMISSIBLE", out)
        self.assertIn("cannot decide the gate", out)
        self.assertNotIn("do not agree", out)
        self.assertEqual(status, 2, out)

    def test_a_zero_drift_sample_bars_no_row(self):
        # TRIO's control carries the baseline's counters verbatim, so its step is
        # zero and nothing can lie inside it. The fixture pages the campaign's
        # published prices were pinned on must solve exactly as before the bar.
        status, out = run(TRIO)
        self.assertIn("no row lies inside it", out)
        self.assertIn("y=1.598", out)
        self.assertNotIn("do not agree", out)
        self.assertEqual(status, 1, out)

    def test_a_page_whose_every_row_is_inked_says_the_bar_emptied_it(self):
        # The screen's own doing, not the page's: three rows were priceable and
        # the drift sample covered all of them. "Nothing priceable" would be true
        # of the arithmetic and false about the cause.
        status, out = run(ALL_INKED)
        self.assertEqual(status, 2, out)
        self.assertIn("not signal, stepped over: a_in, b_in", out)
        self.assertIn("the drift sample stepped over 2 row(s), leaving 0 with signal", out)
        self.assertNotIn("nothing priceable", out)


class Prices(unittest.TestCase):
    def test_it_solves_the_published_wall_prices_off_a_three_decimal_page(self):
        # §5at's wall read of these two rows is x 0.460 / y 1.58-1.59, solved from
        # rates it quoted to three decimals. DOC3 is that page.
        _, out = run(DOC3, "--column", "wall")
        self.assertIn("column wall", out)
        self.assertIn("x=0.460", out)
        self.assertIn("y=1.58", out)

    def test_a_column_the_page_does_not_carry_refuses_rather_than_guessing(self):
        status, out = run(TRIO, "--column", "body_ms")
        self.assertEqual(status, 2)
        self.assertIn("body_ms", out)

    def test_an_out_of_band_y_is_reported_as_a_refutation_not_a_price(self):
        status, out = run(TRIO)
        self.assertEqual(status, 1)
        self.assertIn("REFUTED", out)
        self.assertIn(BAND_TEXT, out)

    def test_an_in_band_y_is_admissible_and_the_gate_is_scored_from_it(self):
        # x=0.199 and y=1.002 built into the rates: the pair must return them, and
        # score §5ar's six-link gate as paying, because p* 0.199 > 0.82/5.30. #215
        # moved this page to the named INBAND fixture and gave it a `base_again` row
        # carrying base's counters verbatim: without that row the page has measured
        # nothing of how far the machine moves a row, and the gate score below is a
        # pass drawn from its decimals alone. The unsampled twin is what #215 refuses.
        status, out = run(INBAND, "--s", "5.30", "--l", "0.82", "--token-ms", "205.5")
        self.assertEqual(status, 0, out)
        self.assertIn("ADMISSIBLE", out)
        self.assertIn("p*=0.199", out)
        self.assertIn("pays", out)

    def test_the_sweep_prints_two_decimals_and_says_what_that_costs(self):
        # The same trio as the sweep prints it solves y=1.598, not 1.585, because
        # `table()` renders tok/s at two decimals and the step is a reciprocal.
        # The screen must name the half-ulp it read and bound its own answer.
        _, out = run(TRIO)
        self.assertIn("±0.005", out)
        printed = float(re.search(r"y=([\d.]+)", out).group(1))
        bound = float(re.search(r"y is known only to ±([\d.]+) ms", out).group(1))
        self.assertLessEqual(abs(printed - 1.585), bound, out)

    def test_a_three_decimal_page_bounds_the_price_tighter_than_a_two_decimal_one(self):
        _, wide = run(TRIO)
        _, tight = run(DOC3)
        self.assertIn("±0.0005", tight)
        self.assertLess(
            float(re.search(r"y is known only to ±([\d.]+) ms", tight).group(1)),
            float(re.search(r"y is known only to ±([\d.]+) ms", wide).group(1)),
        )

    def test_a_page_whose_bound_crosses_the_band_edge_refuses_to_decide_it(self):
        status, out = run(DOC1)
        self.assertEqual(status, 1, out)
        self.assertIn("cannot decide", out)

    def test_a_bound_wider_than_the_band_on_the_admissible_side_says_that_too(self):
        # #209 turned this page's status from 0 to 2: it keeps saying the band call
        # is not separable, and now it also refuses to draw the gate that sentence
        # always claimed it would not draw. #215 moved the gate refusal on THIS page
        # one check earlier -- it carries no drift control, so the widths it would
        # have been refused for are not even its own -- which is why the width-based
        # refusal is pinned on the sampled twin above.
        status, out = run(DOC1_IN, "--s", "5.30", "--l", "0.82", "--token-ms", "205.5")
        self.assertEqual(status, 2, out)
        self.assertIn("ADMISSIBLE", out)
        self.assertIn("cannot decide", out)

    def test_it_reports_how_much_one_millisecond_of_wall_moves_y(self):
        _, out = run(TRIO)
        self.assertRegex(out, r"1 ms of wall moves y by 0\.0\d")

    def test_the_out_of_sample_arm_is_predicted_and_its_residual_reported(self):
        depth325 = arm("layer_depth325", 4.200, 94.78, 5.84)
        _, out = run(page(RING_ON, RING_OFF, DEPTH25, depth325), "--predict", "layer_depth325")
        self.assertIn("out-of-sample layer_depth325", out)
        self.assertIn("residual", out)
        self.assertIn("predicted", out)


class DriftAndHoldout(unittest.TestCase):
    """#209: the page's own drift bounds the price, and a held-out row is scored.

    The digits bound how finely a row is printed and the angle bound how well two
    of them are posed, but neither bounds how far the *machine* moves a row between
    two runs of the same configuration — and the page says exactly how far it moved
    one, because `base_again` is the baseline re-measured. Before #209 the drift
    sample only chose which rows could be paired, and never entered the bound: the
    screen printed "±0.034 ms" for a `y` on a page whose own control had just moved
    a row by 11.17 ms, and the held-out residual was built, printed and thrown away,
    so a model that missed a row it never solved by 29 ms still exited 0.
    """

    def test_the_bound_names_the_digits_and_the_drift_apart(self):
        # This page's control moved 11.17 ms with no config change; its digits put
        # only ±0.034 ms on y, and the drift puts ±0.832, which is more than y is.
        # A single number cannot say which of those two the reader is priced by.
        _, out = run(PAGE5AT_ONE)
        self.assertIn("drift:", out)
        digits = float(re.search(r"the digits put ±([\d.]+)", out).group(1))
        drift = float(re.search(r"drift alone puts ±([\d.]+) ms on y", out).group(1))
        total = float(re.search(r"digits and drift together put ±([\d.]+) ms on y", out).group(1))
        self.assertGreater(drift, digits, out)
        self.assertGreater(total, drift, out)

    def test_the_digits_line_prices_only_the_printing(self):
        # §5au published this page's bound on y as ±0.034 ms -- the rounding's own
        # doing, and the sentence that printed it named the number of decimals it
        # came from. Folding the drift into that same number attributes 0.832 ms of
        # machine to 0.005 tok/s of printing, which is the defect this section
        # exists to close, so the digits line keeps the digits and the sum is said
        # once, as a sum.
        _, out = run(PAGE5AT_ONE)
        self.assertIn("y is known only to ±0.034 ms and x to ±0.015", out)
        self.assertIn("digits and drift together put ±0.866 ms on y", out)

    def test_the_band_caveat_says_which_half_of_the_bound_printing_can_reach(self):
        # On this page 0.832 of the 0.866 is the machine, so telling the reader
        # "more printed decimals is what would settle both" prescribes a cure for
        # a third of the illness and hides the other two thirds.
        _, out = run(PAGE5AT_ONE)
        self.assertIn("its digits ±0.034 ms and its own drift ±0.832", out)
        self.assertIn("printing more decimals would not settle it", out)
        self.assertNotIn("more printed decimals is what would settle both", out)

    def test_a_page_with_no_drift_sample_puts_the_whole_bound_on_the_printing(self):
        # The arithmetic half, which #215 leaves standing: with no re-measured
        # baseline on the page, the bound it prints IS the printing, so the two
        # numbers have to agree exactly and no drift term may quietly be charged as
        # zero. What used to be asserted here was the sentence's other half -- "and
        # more decimals is the cure" -- and that is the premise #215 refuses, so it
        # is gone and the equality is what remains.
        _, out = run(DOC1)
        self.assertIn("cannot decide", out)
        self.assertIn("bound on y is ±0.310 ms -- the printing alone, ±0.310 ms", out)

    def test_x_s_drift_sensitivity_is_the_difference_mode_not_the_single_row_max(self):
        # Both rows drift on their own, so their errors add: (|du1|+|du2|)/|det|.
        # §5ax had to derive this by hand because the screen printed only the
        # single-row maximum, and 0.028 (TRIO's x) is not 0.019 (that maximum) or
        # 0.009 (the common mode, which largely cancels).
        _, out = run(TRIO)
        self.assertRegex(out, r"difference-mode sensitivities 0\.073 ms of y and 0\.028 ms of x")

    def test_a_p_star_whose_own_width_spans_the_gate_is_refused(self):
        # This page prices y=1.004 inside the band and p*=0.237 above L/S=0.155, so
        # before #209 this scored "-> pays" and exited 0. Its own printed decimals
        # put ±0.121 ms on x and ±0.283 ms on y, which makes p* run 0.091 to 0.498
        # straight across the threshold: the page cannot say which side the gate is
        # on, and the band call standing does not make the gate call one. #215 moved
        # it to the sampled twin, because on the unsampled DOC1_IN the refusal now
        # comes one check earlier and for a different reason -- the drift was never
        # read at all -- which that section's own test pins.
        status, out = run(DOC1_IN_SAMPLED, "--s", "5.30", "--l", "0.82", "--token-ms", "205.5")
        self.assertEqual(status, 2, out)
        self.assertIn("ADMISSIBLE", out)
        self.assertIn("cannot decide the gate", out)
        self.assertIn("L/S = 0.155", out)
        self.assertNotIn("p*=", out)
        self.assertNotIn("-> pays", out)
        self.assertNotIn("-> does not pay", out)

    def test_a_drift_width_larger_than_y_itself_refuses_the_gate_as_unbounded(self):
        # Here the page's own control moves y by more than y is, so the ratio is not
        # finite and no threshold test on it means anything. The width comes from
        # drift, not from coarse decimals: the same rows printed to three decimals
        # would still refuse.
        status, out = run(PAGE5AT_ONE)
        self.assertEqual(status, 2, out)
        self.assertIn("cannot decide the gate", out)
        self.assertIn("not finite", out)

    def test_a_page_that_never_sampled_the_drift_does_not_score_the_gate(self):
        # The first page this screen was handed priced an adopted read, drew the
        # six-link gate and exited 0 on digits alone. What it never did is measure how
        # far the machine moves a row with no config change in it: it carries no
        # `base_again`, so `_solve` was handed a drift width of exactly 0.0, which is
        # the number it would have printed had the control been there and moved
        # nothing. Silence was read as a measurement, and the campaign's own pages say
        # this Mac's drift is nowhere near 0.004 ms: §5az found the gate needs a
        # baseline repeating to 0.75 ms, "which this machine has never given". The
        # price and its band call print -- #208's ruling that a missing control is not
        # a refusal to solve -- and the gate is refused for the reason, with its own
        # status.
        status, out = run(INBAND_NO_CONTROL, "--s", "5.30", "--l", "0.82", "--token-ms", "205.5")
        self.assertEqual(status, 2, out)
        self.assertIn("ADMISSIBLE", out)
        self.assertIn("x=0.199 y=1.002", out)
        # Not "no drift sample", which the signal line printed before this fix too:
        # this is the verdict's own reason, and it is what the status hangs on.
        self.assertIn("carried no re-measured baseline", out)
        self.assertNotIn("p*=", out)
        self.assertNotIn("pays", out)

    def test_an_unsampled_drift_is_not_the_same_page_as_a_measured_zero(self):
        # The two pages differ by one row, and the row's step is zero, so every
        # number either page solves is the same. Only the license differs: one read
        # the machine and found it still, the other did not look. If the fixture
        # above refuses and this one exits 0, the rule is about the channel, not
        # about the arithmetic.
        sampled, out_sampled = run(INBAND, "--s", "5.30", "--l", "0.82", "--token-ms", "205.5")
        unsampled, out_unsampled = run(
            INBAND_NO_CONTROL, "--s", "5.30", "--l", "0.82", "--token-ms", "205.5"
        )
        self.assertIn("x=0.199 y=1.002", out_sampled)
        self.assertIn("x=0.199 y=1.002", out_unsampled)
        self.assertEqual(sampled, 0, out_sampled)
        self.assertEqual(unsampled, 2, out_unsampled)

    def test_the_remedy_on_an_unsampled_page_names_the_row_it_lacks(self):
        # DOC1's bound is 0.310 ms and it is all printing: no control was on the
        # page, so no drift entered it. Telling that reader "more printed decimals
        # is what would settle both" prescribes the cure for the half it measured and
        # calls the half it never read settled.
        _, out = run(DOC1)
        self.assertIn("cannot decide", out)
        self.assertIn("the printing alone, ±0.310 ms", out)
        self.assertIn("no re-measured baseline", out)
        self.assertNotIn("more printed decimals is what would settle both", out)

    def test_a_measured_zero_drift_still_points_at_the_printing(self):
        # The same sentence on a page that DID sample the drift and found nothing:
        # here more decimals genuinely is what would narrow the bound, because the
        # other half has been read and is zero. Without this row the #215 branch
        # would swallow the case it was meant to distinguish.
        status, out = run(DOC1_IN_SAMPLED, "--s", "5.30", "--l", "0.82", "--token-ms", "205.5")
        self.assertIn("cannot decide", out)
        self.assertIn("more printed decimals is what would settle both", out)
        self.assertNotIn("no re-measured baseline", out)
        self.assertEqual(status, 2, out)

    def test_the_missing_license_does_not_swallow_the_interval_refusal(self):
        # DOC1_IN carries both defects: one decimal makes `p*` straddle the threshold
        # (#209) and no re-measured baseline means its drift was never read (#215).
        # The license is the more basic refusal, but the interval is a finding about
        # this page, and a rule that returns first would cost the screen the sentence
        # #209 was shipped to print -- so both lines print and one status answers.
        status, out = run(DOC1_IN)
        self.assertEqual(status, 2, out)
        self.assertIn("cannot decide the gate", out)
        self.assertIn("carried no re-measured baseline", out)

    def test_a_held_out_row_beyond_the_page_s_own_width_refutes_the_model(self):
        # The two prices fit the two rows they were solved from exactly — that is
        # what Cramer means — so the only test of the model is a row it never saw.
        # This one misses by 29.00 ms against the 1.694 ms the page cannot resolve:
        # 0.53 ms is that row's own printed step and the rest is how far the two
        # prices' widths reach into the prediction, so the miss is a refutation and
        # not a rounding.
        status, out = run(HOLD_BAD, "--predict", "layer_depth35")
        self.assertEqual(status, 1, out)
        self.assertIn("REFUTED OUT OF SAMPLE", out)
        self.assertNotIn("p*=", out)  # a refuted model is not allowed to draw the gate

    def test_a_held_out_row_inside_the_page_s_own_width_is_scored_as_consistent(self):
        # The same page with the held-out arm where the model puts it: the residual
        # is inside the width the row itself carries, so it is evidence the model
        # predicted a row it never solved, and the gate stays scored.
        status, out = run(HOLD_OK, "--predict", "layer_depth35")
        self.assertEqual(status, 0, out)
        self.assertIn("out-of-sample consistent", out)
        self.assertIn("pays", out)
        self.assertNotIn("REFUTED OUT OF SAMPLE", out)


class Rounds(unittest.TestCase):
    def test_one_round_says_a_magnitude_needs_more(self):
        _, out = run(TRIO)
        self.assertIn("rounds 1", out)
        self.assertIn("magnitude", out)

    def test_per_round_prices_are_printed_and_the_median_classifies(self):
        shifted = page(
            arm("base", 5.079, 39.45, 18.86),
            arm("prefetch_off", 4.800, 0.00, 0.00),
            arm("layer_depth25", 4.410, 72.73, 9.44),
            arm("base_again", 5.043, 39.45, 18.86),
        )
        _, out = run(TRIO + "\n" + shifted)
        self.assertEqual(out.count("round "), 2, out)
        self.assertIn("median", out)
        self.assertIn("spread", out)


if __name__ == "__main__":
    unittest.main()
