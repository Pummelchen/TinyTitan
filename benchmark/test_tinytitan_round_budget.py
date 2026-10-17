#!/usr/bin/env python3
"""Tests for `tinytitan_round_budget.py`: #216.

§5ar closed its withholding arm by saying its magnitude "is smaller than §5ap's
four-round control spread", and that the sign "needs on the order of 30-60 clean
rounds". §5az's deviation 3 stated the reachable-bound result honestly — "they
say no page this quiet can score the gate, not that some page that quiet exists"
— but left the follow-up question unanswered: **would more rounds settle it?**
That is a arithmetic question about a pool of pages, and this screen answers it
offline from pages already on disk.

The whole question is which bound a pool may carry, and there are three answers
with three different provenances:

1. **median / widest bound** — what `tinytitan_price_pair.py` does today: the
   median of the rounds' prices, bounded by the *widest* bound in the pool,
   because "the median of several solves is no better known than its least known
   member". Distribution-free, and it can never shrink by adding rounds: one
   wild control caps the pool forever.
2. **mean / triangle bound** — the mean of the rounds, bounded by the *mean* of
   their bounds. Still distribution-free (a triangle inequality, no independence
   assumption), and it does improve with rounds — but only toward the average
   bound, never toward zero.
3. **mean / sqrt(n)** — `t · sd / sqrt(n)`, the one that converges to zero and is
   therefore the rule that would license "30-60 rounds". It assumes each round's
   error is independent zero-mean noise, and the pool itself carries n samples of
   that error: the signed step each round's drift control took. Drift that points
   the same way every round is a bias, and averaging rounds does not cancel a
   bias, so the screen tests the assumption on the pool's own controls and only
   prints the sqrt(n) budget if they license it.

Ten things it must not be able to get wrong, each pinned below:

1. **Rule 1 must equal the shipped screen**, to the digit, on the same page. Two
   screens that price one pool differently are two bugs, not a choice, so the
   fixture is §5ax's real published pair of tables and the assertion is the
   `±0.805` that screen printed.
2. **The mean bound goes with the mean, not the median.** The triangle inequality
   bounds an average; applying it to a median would understate the median's own
   error, which rule 1 exists to avoid.
3. **A distribution-free bound does not converge to zero.** Adding rounds that
   each carry ±0.5 ms of their own leaves a pool at ±0.5 ms forever. A test that
   three identical rounds shrink rule 2 would be a test of the wrong rule.
4. **Unanimous controls refuse sqrt(n).** §5ax's two controls are −3.527 and
   −11.701 ms: the machine moved one way, hard, in both. That is the campaign's
   whole problem in two numbers, and it must cost the sqrt(n) rule its licence.
5. **Two rounds cannot establish independence however their signs fall** — the
   shipped screen's own line is "a direction is defensible; a magnitude wants 3
   or more", and a rule that estimates an sd from two points is quoting itself.
6. **A pool member that never sampled its drift is not a wide member, it is an
   unread one.** #215 made that a refusal for a page; a pool that averaged it in
   would let the unread half hide inside a mean.
7. **A pair the pool does not offer in every round prices nothing.** Solving
   `A x B` in two rounds and `A x C` in a third is three prices of two different
   questions, and their median is meaningless.
8. **A separable y outside the band is a refutation, not a settlement.** The
   status the shipped screen gives it, given here too, because no amount of
   pooling rescues a model the rows falsify.
9. **A drift measured at exactly zero every round is a measured zero, not a
   missing sample** (§5as). The steadiest baseline in the campaign must license
   the shrinking rule; refusing it would make perfect stability the least poolable
   evidence there is, and would put the guard on the wrong quantity.
10. **Repeats of one price are not a scatter.** If the pool's prices do not move
    while only its baseline does, `t · sd / sqrt(n)` is `t · 0 / sqrt(n) = 0`, and
    a bound of ±0.000 ms/token is the absence of a measurement dressed as the
    tightest one in the campaign. Rule 3 has to refuse itself on that pool.

The statuses mirror `tinytitan_price_pair.py`:

    0  a licensed rule settles what the pages alone could not, and is named
    1  the pool's price refutes the two-price model
    2  no licensed rule settles it — or the pool cannot be priced at all — with
       the rule that would and the rounds it needs named

Fixtures are the sweep's own `ks.table()` or the markdown shape §5ax was
published in. No model, no server, nothing fetched.
"""

from __future__ import annotations

import io
import pathlib
import sys
import unittest
from contextlib import redirect_stdout

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))

import tinytitan_knob_sweep as ks  # noqa: E402
import tinytitan_price_pair as pp  # noqa: E402
import tinytitan_round_budget as rb  # noqa: E402

BAND_TEXT = "0.788-1.301 ms"

# §5ax's two clean rounds, published verbatim in
# docs/qwen38-decode-12gb-measurements.md: the identifying trio plus its
# out-of-sample row, with a re-measured baseline in every round. The shipped
# screen solves these to x=0.120 y=0.897 (bound ±0.264) and x=0.224 y=0.837
# (bound ±0.805) and refuses the band call, because 0.867 sits only 0.079 ms
# inside the near edge. Everything this file asserts about "the pool" is this
# pool.
AX = """
| arm | tok/s | issued/pt | adopted/pt |
|---|---|---|---|
| base | 4.42 | 42.65 | 22.12 |
| prefetch_off | 4.15 | 0.00 | 0.00 |
| layer_depth25 | 4.15 | 75.69 | 10.15 |
| layer_depth325 | 3.69 | 102.85 | 6.67 |
| base_again | 4.49 | 42.51 | 21.86 |

| arm | tok/s | issued/pt | adopted/pt |
|---|---|---|---|
| base | 4.32 | 42.58 | 21.36 |
| prefetch_off | 4.17 | 0.00 | 0.00 |
| layer_depth25 | 4.03 | 75.67 | 10.33 |
| layer_depth325 | 3.85 | 102.80 | 5.86 |
| base_again | 4.55 | 42.56 | 21.64 |
"""

PAIR = ("layer_depth25", "prefetch_off")


def arm(name, tok_s, issued, adopted):
    """One row of the sweep's own table, as far as the shipped screen reads it."""
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


def table(*arms):
    """A one-round page in the sweep's own printed shape."""
    return ks.table(list(arms), arms[0]["tok_s"])


# Three quiet rounds of the §5ax trio, printed to three decimals with a
# re-measured baseline that did not move: the shipped screen's bound on `y`
# collapses to its digits, ±0.004 ms, which clears the band margin this pool has
# (0.109 ms) and puts p* at 0.131-0.137. Used where a pool *does* settle.
QUIET = """
| arm | tok/s | issued/pt | adopted/pt |
|---|---|---|---|
| base | 4.420 | 42.65 | 22.12 |
| prefetch_off | 4.150 | 0.00 | 0.00 |
| layer_depth25 | 4.150 | 75.69 | 10.15 |
| base_again | 4.420 | 42.65 | 22.12 |
"""

# The same trio with `prefetch_off` a full 1.4 tok/s slower, which pulls the
# adopted price above the band's top edge with a bound of a hair: a separable
# answer that falsifies the model rather than pricing an expensive read.
OUTSIDE = """
| arm | tok/s | issued/pt | adopted/pt |
|---|---|---|---|
| base | 4.420 | 42.65 | 22.12 |
| prefetch_off | 3.000 | 0.00 | 0.00 |
| layer_depth25 | 4.100 | 75.69 | 10.15 |
| base_again | 4.420 | 42.65 | 22.12 |
"""


# §5ax's two published rounds followed by §5at's published page, which is what a
# pool of *prices* is: each table is solved against its own baseline, so no clock is
# compared across batches even though the rounds come from different ones. The third
# round's baseline re-measured 4.8 % slower (+11.167 ms) where §5ax's two moved
# faster (−3.527, −11.701), so this pool's drift samples have mixed signs and a
# spread wider than their mean step: the shrinking rule IS licensed here. It is the
# only fixture in this file that reaches that verdict, and the screen's closing line
# must not blame its refusal on drift that licensed it.
POOL3 = (
    AX
    + """
| arm | tok/s | issued/pt | adopted/pt |
|---|---|---|---|
| base | 4.55 | 42.75 | 22.85 |
| prefetch_off | 4.79 | 0.00 | 0.00 |
| layer_depth25 | 4.00 | 76.03 | 13.43 |
| slots_64 | 4.47 | 42.72 | 22.64 |
| base_again | 4.33 | 42.57 | 21.05 |
"""
)


def run(*argv, stdin=None):
    """Drive `main()` and hand back (status, stdout) the way the sibling suite does."""
    out = io.StringIO()
    with redirect_stdout(out):
        status = rb.main(list(argv), stdin)
    return status, out.getvalue()


def sols(text, pair=PAIR):
    """The pool this screen builds from a page, for direct assertion."""
    rounds = pp.read(text)
    return rb.pool(rounds, pair, "base", pp.WALL, "base_again", "page")


class SharedArithmetic(unittest.TestCase):
    """The pool screen must not be a second opinion about one page's numbers."""

    def test_the_median_rule_reproduces_the_shipped_screens_bound(self):
        got = sols(AX)
        self.assertEqual(len(got), 2)
        self.assertAlmostEqual(got[0].y, 0.897, places=3)
        self.assertAlmostEqual(got[1].y, 0.837, places=3)
        self.assertAlmostEqual(got[0].y_bound, 0.264, places=3)
        self.assertAlmostEqual(got[1].y_bound, 0.805, places=3)
        med = rb.median_rule(got)
        self.assertAlmostEqual(med.y, 0.867, places=3)
        # The shipped screen's own line for this page, to the millimetre.
        self.assertAlmostEqual(med.y_bound, 0.805, places=3)

    def test_the_signed_control_survives_where_the_shipped_screen_stores_a_magnitude(self):
        # `Drift.dm` is a width, so the shipped screen rightly abs()s it; a pool has
        # to know which way the machine went, because a one-way drift is a bias.
        got = sols(AX)
        self.assertLess(got[0].control, 0.0)
        self.assertLess(got[1].control, 0.0)
        self.assertAlmostEqual(abs(got[0].control), 3.527, places=3)
        self.assertAlmostEqual(abs(got[1].control), 11.701, places=3)


class PoolingRules(unittest.TestCase):
    """Three widths on one pool, and what each one is allowed to assume."""

    def test_the_mean_bound_is_the_mean_of_the_bounds_and_uses_the_mean_price(self):
        got = sols(AX)
        med = rb.median_rule(got)
        mean = rb.mean_rule(got)
        want = (got[0].y_bound + got[1].y_bound) / 2.0
        self.assertLessEqual(abs(mean.y_bound - want), 0.001)
        # Bounds go up to the millimetre, never down, so a printed bound cannot
        # understate the pool's: §5ax's two rounds average 0.5345 and print ±0.535.
        self.assertGreaterEqual(mean.y_bound, want)
        self.assertLess(mean.y_bound, med.y_bound)
        # n=2 makes the mean and the median the same number; the rule still has to
        # say which estimator it bounded, because the triangle inequality does not
        # bound a median.
        self.assertEqual(mean.estimator, "mean")
        self.assertEqual(med.estimator, "median")

    def test_a_distribution_free_bound_converges_to_the_average_bound_not_to_zero(self):
        # Three rounds, two of them as quiet as §5ax's first: the wide control is
        # still in the pool, so no distribution-free rule may hand back a budget
        # that shrinks away from it.
        quiet = [
            rb.Sol("p", 1, PAIR, 47.3, 0.120, 0.897, 0.016, 0.264, -3.527),
            rb.Sol("p", 2, PAIR, 47.3, 0.120, 0.897, 0.016, 0.264, -3.527),
            rb.Sol("p", 3, PAIR, 45.1, 0.224, 0.837, 0.032, 0.805, -11.701),
        ]
        mean = rb.mean_rule(quiet)
        want = (0.264 * 2 + 0.805) / 3.0
        self.assertLessEqual(abs(mean.y_bound - want), 0.001)
        self.assertGreater(mean.y_bound, 0.4)

    def test_the_sqrt_n_budget_is_t_times_sd_over_root_n_and_is_refused_by_unanimous_controls(self):
        got = sols(AX)
        ok, why = rb.license(got)
        self.assertFalse(ok)
        self.assertIn("same direction", why)
        self.assertIn("-3.527", why)
        budget = rb.sqrt_n_rule(got)
        self.assertFalse(budget.licensed)
        # 0.079 ms is this pool's margin: how far y=0.867 sits inside 0.788.
        self.assertEqual(budget.n_needed, 4)
        self.assertGreater(budget.y_bound, 0.3)
        self.assertLess(budget.y_bound, 0.5)


class Licensing(unittest.TestCase):
    """What the pool's own drift samples say about averaging them."""

    def test_two_rounds_cannot_license_a_magnitude_however_their_signs_fall(self):
        # The shipped screen's rule: "a direction is defensible; a magnitude wants
        # 3 or more". Mixed signs at n=2 must not be read as independence.
        pool = [
            rb.Sol("p", 1, PAIR, 47.0, 0.120, 0.900, 0.016, 0.050, -3.527),
            rb.Sol("p", 2, PAIR, 47.0, 0.220, 0.830, 0.032, 0.050, 11.701),
        ]
        ok, why = rb.license(pool)
        self.assertFalse(ok)
        self.assertIn("three", why)

    def test_a_drift_bigger_than_the_round_to_round_scatter_refuses_the_rule(self):
        # Signs mixed and n=4, but the controls' net step dwarfs their spread: the
        # machine has a direction, and a direction does not average out.
        pool = [
            rb.Sol("p", i, PAIR, 47.0, 0.120, 0.860 + i * 0.010, 0.016, 0.050, d)
            for i, d in enumerate((1.0, -0.1, 1.1, 0.8), start=1)
        ]
        ok, why = rb.license(pool)
        self.assertFalse(ok)
        self.assertIn("direction", why)
        self.assertIn("0.548", why)

    def test_three_rounds_with_scattered_signed_controls_license_it(self):
        pool = [
            rb.Sol("p", i, PAIR, 47.0, 0.120, 0.860 + i * 0.010, 0.016, 0.050, d)
            for i, d in enumerate((2.0, -3.0, 1.0, -2.5, 3.5), start=1)
        ]
        ok, why = rb.license(pool)
        self.assertTrue(ok, why)
        budget = rb.sqrt_n_rule(pool)
        self.assertTrue(budget.licensed)
        self.assertLess(budget.y_bound, 0.05)

    def test_the_drift_moving_while_the_prices_do_not_is_still_no_scatter(self):
        # The licensing test above reads the controls; this reads the prices. Here
        # the re-measured baseline moved in both directions across five rounds, so
        # the drift samples are licensed, but every round solved the pair to exactly
        # y=0.897. `t * sd / sqrt(n)` on those is `t * 0 / sqrt(n)` = 0, and a bound
        # of ±0.000 ms/token would print as the tightest measurement in the campaign
        # while being the absence of one. n copies of one price is one price.
        pool = [
            rb.Sol("p", i, PAIR, 47.0, 0.120, 0.897, 0.016, 0.264, d)
            for i, d in enumerate((2.0, -3.0, 1.0, -2.5, 3.5), start=1)
        ]
        ok, why = rb.license(pool)
        self.assertTrue(ok, why)
        budget = rb.sqrt_n_rule(pool)
        self.assertFalse(budget.licensed)
        self.assertIn("price", budget.why)
        self.assertNotIn("settles", budget.why)
        self.assertIsNone(budget.n_needed)

    def test_a_baseline_that_never_moved_is_a_measured_zero_not_a_missing_sample(self):
        # §5as's rule: a control that steps exactly zero is a MEASURED zero, which
        # is a different statement from a page that never read the channel. Here every
        # round re-measured the baseline and every round got the same number, while
        # the pair's price moved. The drift channel therefore measured no bias, the
        # prices carry the scatter the rule divides, and the licence has to say so --
        # refusing here would make the steadiest pool in the campaign the least
        # poolable one, and would hide the real guard (scatter in the prices) behind
        # one on the wrong quantity.
        pool = [
            rb.Sol("p", i, PAIR, 47.0, 0.120, 0.860 + i * 0.010, 0.016, 0.050, 0.0)
            for i in range(1, 6)
        ]
        ok, why = rb.license(pool)
        self.assertTrue(ok, why)
        self.assertIn("never moved", why)
        budget = rb.sqrt_n_rule(pool)
        self.assertTrue(budget.licensed, budget.why)
        # sd(0.870..0.910) = 0.01581 ms/token, t(4) = 2.776, n = 5 -> 0.01963 ms,
        # which prints rounded up at the millimetre like every other bound here.
        self.assertGreater(budget.y_bound, 0.019)
        self.assertLess(budget.y_bound, 0.021)


class PoolRefusals(unittest.TestCase):
    """A pool is only a pool when every member answers the same question."""

    def test_a_round_that_never_sampled_its_drift_is_not_a_pool_member(self):
        # `layer_depth999` is here so the reader keeps this table: a table whose arm
        # set is a proper subset of a later one is the sweep's own cumulative
        # emission, not a round, and dropping it would silently test the wrong thing.
        page = """
| arm | tok/s | issued/pt | adopted/pt |
|---|---|---|---|
| base | 4.42 | 42.65 | 22.12 |
| prefetch_off | 4.15 | 0.00 | 0.00 |
| layer_depth25 | 4.15 | 75.69 | 10.15 |
| layer_depth999 | 4.25 | 88.00 | 8.00 |
"""
        status, text = run(stdin=page + "\n" + AX)
        self.assertEqual(status, 2)
        self.assertIn("no re-measured baseline", text)
        self.assertIn("table 1", text)

    def test_a_pair_one_round_does_not_offer_prices_nothing(self):
        status, text = run("--pair", "layer_depth25", "layer_depth325", stdin=AX)
        self.assertIn("layer_depth25 x layer_depth325", text)
        self.assertEqual(status, 2)

    def test_one_round_is_the_page_it_came_from_not_a_pool(self):
        status, text = run(stdin=AX.split("\n\n")[0])
        self.assertEqual(status, 2)
        self.assertIn("one round", text)

    def test_the_pair_reaches_the_screen_through_every_round_or_the_pick_names_another(self):
        best, why = rb.pick_pair([pp.read(AX)[0]], "base", pp.WALL, "base_again")
        self.assertIsNotNone(best)
        self.assertEqual(set(best), set(PAIR))
        self.assertIn("47.", why)


class Verdicts(unittest.TestCase):
    """The status the pool earns, and the sentence that says why."""

    def test_a_pool_no_licensed_rule_can_settle_refuses_and_names_what_would(self):
        status, text = run(stdin=AX)
        self.assertEqual(status, 2)
        self.assertIn("cannot settle", text)
        self.assertIn("±0.535", text)
        self.assertIn("4 rounds", text)
        self.assertIn("not licensed", text)

    def test_a_pool_whose_mean_bound_clears_the_margin_settles_the_band(self):
        status, text = run(stdin=QUIET * 3)
        self.assertEqual(status, 0, text)
        self.assertIn("settles", text)
        self.assertIn("mean / triangle", text)

    def test_a_separable_y_outside_the_band_is_a_refutation_not_a_settlement(self):
        status, text = run(stdin=OUTSIDE * 3)
        self.assertEqual(status, 1, text)
        self.assertIn("REFUTED", text)
        self.assertIn(BAND_TEXT, text)

    def test_the_gate_question_is_owed_its_own_answer_after_the_band_one(self):
        # The quiet pool settles `y` to ±0.004 ms, well inside its 0.109 ms margin,
        # but a gate whose L/S = 0.134 falls inside p*'s own 0.131-0.137: the band
        # call separates and the gate draw still cannot, and the screen has to say
        # which of the two it paid rather than calling the pool a settlement.
        status, text = run("--l", "0.71", stdin=QUIET * 3)
        self.assertEqual(status, 2, text)
        self.assertIn("the band call separates", text)

    def test_a_pool_that_licenses_the_shrinking_rule_is_not_blamed_on_its_drift(self):
        # Three rounds, drift samples -3.527, -11.701, +11.167 ms: mixed signs and a
        # spread wider than their mean step, so rule 3 is licensed and its own line
        # reads as a width, not a refusal. The closing sentence used to hard-code
        # "refused by the pool's own drift samples" whatever the rows above it said,
        # which told this pool the opposite of its own evidence.
        status, text = run(stdin=POOL3)
        self.assertEqual(status, 2, text)
        self.assertNotIn("not licensed", text)
        self.assertNotIn("refused by the pool's own drift samples", text)
        self.assertIn("licensed", text)


if __name__ == "__main__":
    unittest.main()
