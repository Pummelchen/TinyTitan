#!/usr/bin/env python3
"""Tests for `tinytitan_price_pair.py`: #207.

The two-price model says a paired decode arm costs `x` ms per issued speculative
read and `y` ms per adopted one, so `dm = x*du + y*dv` for a row
`(du, dv) = (dissued, -dadopted)`. §5as chose which arms to run by the *angle*
between two such rows, §5at re-derived that geometry from published counters, and
both left the arithmetic itself in a throwaway. This screen closes that seam: it
reads the page `tinytitan_knob_sweep.py` prints and answers `x`, `y`, the
condition number and the band verdict with an exit status.

Five things it must not be able to get wrong, each pinned below:

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

The statuses are the ones this tree's drivers already use:

    0  the design identifies and `y` is inside the admissible band
    1  it identified a price, and that price refutes the model
    2  it could not solve, with the reason named

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
# the band (m1 12.77, m2 17.39 -> y = 1.004), but now 0.05 tok/s of half-ulp is 4.3 ms
# of step and the band puts only 0.22 ms between y and the near edge. The clause has
# to fire on the side of the band that agrees with the model, too -- a caveat that
# only ever prints next to a refutation is a caveat that has never been tested.
DOC1_IN = """
| arm | tok/s | issued/pt | adopted/pt |
|---|---|---|---|
| base | 5.0 | 42.75 | 22.85 |
| prefetch_off | 4.7 | 0.00 | 0.00 |
| layer_depth25 | 4.6 | 76.03 | 13.43 |
"""


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
        # score §5ar's six-link gate as paying, because p* 0.199 > 0.82/5.30.
        status, out = run(
            """
| arm | tok/s | issued/pt | adopted/pt |
|---|---|---|---|
| base | 5.000 | 42.75 | 22.85 |
| prefetch_off | 4.665 | 0.00 | 0.00 |
| layer_depth25 | 4.628 | 76.03 | 13.43 |
""",
            "--s",
            "5.30",
            "--l",
            "0.82",
            "--token-ms",
            "205.5",
        )
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
        status, out = run(DOC1_IN, "--s", "5.30", "--l", "0.82", "--token-ms", "205.5")
        self.assertEqual(status, 0, out)
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
