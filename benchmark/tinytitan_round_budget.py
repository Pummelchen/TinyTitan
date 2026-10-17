#!/usr/bin/env python3
"""#216: would more rounds settle what this page could not?

`tinytitan_price_pair.py` prices one batch: it solves `dm = x*du + y*dv` off two
arms, bounds each price by that page's printed digits plus that page's own
re-measured baseline, and refuses a band or gate call that falls inside the
width. §5ax ran two clean rounds and it refused both questions — `y=0.867` sits
0.079 ms inside the near band edge and the bound on it is ±0.805 ms. That leaves
the campaign's actual open question unanswered: §5ar said the sign of its
withholding arm "needs on the order of 30-60 clean rounds", and §5az stated the
reachable-bound result as "no page this quiet can score the gate, not that some
page that quiet exists". Whether rounds *can* settle it is arithmetic on a pool
of pages, and this screen does it offline on pages already on disk.

The whole answer is which bound a pool may carry. Three rules, three
provenances:

    1  median / widest   the shipped screen's rule: median of the prices, bound
                         by the widest bound in the pool, because a median of
                         several solves is no better known than its least known
                         member. Distribution-free — and it never improves,
                         because one wild control caps the pool forever.
    2  mean / triangle   mean of the prices, bound by the mean of the bounds
                         (`|mean(z_r) - z| <= mean(|z_r - z|) <= mean(b_r)`).
                         Still distribution-free. It improves with rounds, but
                         only toward the average bound, never toward zero.
    3  mean / sqrt(n)    `t * sd / sqrt(n)`, the only rule that converges to
                         zero and therefore the only one that could license
                         "30-60 rounds". It assumes each round's error is
                         independent zero-mean noise.

Assumption 3 is not a matter of faith here, because the pool measures it. Each
round carries a drift control — `base_again`, the baseline re-measured, whose
step is what the machine did to a row with no config change in it — and n rounds
therefore hand over n signed samples of exactly the error the rule averages.
Drift that points the same way in every round is a bias, and averaging rounds
does not cancel a bias, so this screen tests the rule against the pool's own
controls and prints its budget only when they license it — and only when the
pool's *prices* move too, because `t*0/sqrt(n)` is zero and a bound of ±0.000 ms
is the absence of a measurement wearing the costume of the tightest one. On §5ax's
pool the two controls are −3.527 and −11.701 ms: same direction, three times
apart, which is the machine's story and the reason the sqrt(n) number is a
planning estimate rather than a bound.

Statuses, matching the shipped screen:

    0  a licensed rule settles what the pages alone could not, and is named
    1  the pool's price refutes the two-price model
    2  no licensed rule settles it — or the pool cannot be priced at all — with
       the rule that would and the rounds it needs named

Reads the same pages `tinytitan_price_pair.py` reads, and imports its solve, its
band and its step arithmetic rather than restating them, so the two screens
cannot disagree about a price.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import statistics
import sys
from dataclasses import dataclass

import tinytitan_price_pair as pp

BAND_TEXT = f"{pp.BAND_LO:.3f}-{pp.BAND_HI:.3f} ms"
MAX_ROUNDS = 400

# Two-sided 95 % Student-t by degrees of freedom, for the round budget. Small on
# purpose: a rule that needs 30 rounds is beyond the point, and the tail values
# (1.960) are what the table converges to, not a promise about a real pool.
T95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    12: 2.179,
    15: 2.131,
    20: 2.086,
    24: 2.064,
    30: 2.042,
}


def t_two_sided_95(df: int) -> float:
    """The table's value for `df`, and the value it converges to beyond 30."""
    if df in T95:
        return T95[df]
    if df < 1:
        return T95[1]
    for key in sorted(T95):
        if df < key:
            return T95[key]
    return 1.960


def _ceil_ms(ms: float) -> float:
    """Round a bound up to the millimetre, so printing it cannot understate it."""
    return math.ceil(ms * 1000.0) / 1000.0


@dataclass
class Sol:
    """One round's solve of one named pair, as this screen pools it.

    `control` is that round's drift sample signed, in ms/token: the shipped screen
    keeps `Drift.dm` as a magnitude because a width has no direction, and the
    direction is the one thing a pool has to know. `y_bound` and `x_bound` are that
    round's own total width, digits and drift together, as the shipped screen
    licenses them.
    """

    page: str
    index: int
    pair: tuple[str, str]
    deg: float
    x: float
    y: float
    x_bound: float
    y_bound: float
    control: float | None


@dataclass
class Budget:
    """A pooled price, its width, and what that width is allowed to assume."""

    rule: str
    estimator: str
    x: float
    y: float
    x_bound: float
    y_bound: float
    licensed: bool
    why: str
    n_needed: int | None = None


@dataclass
class Call:
    """What one budget says about the two questions the campaign asks."""

    in_band: bool
    edge: float
    gap: float
    band_separable: bool
    gate_lo: float
    gate_hi: float | None
    gate_separable: bool
    need: float


def score(b: Budget, need: float) -> Call:
    """Band edge, gate interval, and whether this budget's width clears either."""
    in_band = pp.BAND_LO <= b.y <= pp.BAND_HI
    edge = min((pp.BAND_LO, pp.BAND_HI), key=lambda e: abs(b.y - e))
    gap = abs(b.y - edge)
    lo, hi = pp.pi_interval(b.x, b.x_bound, b.y, b.y_bound)
    return Call(
        in_band,
        edge,
        gap,
        b.y_bound < gap,
        lo,
        hi,
        hi is not None and not lo <= need <= hi,
        need,
    )


def license(sols: list[Sol]) -> tuple[bool, str]:
    """Do the pool's own drift samples support averaging its rounds?

    Four tests, in the order that says the most: the pool has to have read the
    control at all, then have to show the drift with no direction — either mixed
    signs with a spread at least as wide as its own mean step, or a drift measured
    and never moved (§5as: a control that steps exactly zero is a MEASURED zero,
    which is not the silence of a page that did not read the channel) — and finally
    have to have enough rounds for a magnitude (the shipped screen's own line is
    that a direction is defensible at two rounds and a magnitude wants three or
    more). A pool that fails any of these may still be pooled distribution-free; it
    may not buy a sqrt(n). What this reads is the *drift*; a pool licensed here can
    still have no scatter in its prices, and `sqrt_n_rule` refuses that itself.
    """
    missing = [s for s in sols if s.control is None]
    if missing:
        names = ", ".join(f"{s.page} table {s.index}" for s in missing)
        return False, f"{names} carried no re-measured baseline, so the pool has an unread half"
    steps = [float(s.control) for s in sols]  # type: ignore[arg-type]
    said = ", ".join(f"{d:+.3f}" for d in steps)
    if all(d > 0 for d in steps) or all(d < 0 for d in steps):
        return False, (
            f"the pool's own drift samples ({said} ms) moved the same direction in every round, "
            "and averaging rounds cancels nothing that has a direction"
        )
    if len(steps) < 3:
        return False, (
            f"a pool of {len(steps)} round(s) cannot license a magnitude: the shipped screen's "
            "own rule is three or more, and an sd from two points is the two points talking "
            "to themselves"
        )
    mean = statistics.fmean(steps)
    sd = statistics.stdev(steps)
    # Unanimity above took every equal-but-nonzero sample, so an sd of zero here can
    # only mean every round re-measured its baseline and got the same number back:
    # a measured zero drift, which is the steadiest evidence the channel can give.
    if sd == 0.0:
        return True, (
            f"the drift never moved: every re-measured baseline stepped {steps[0]:+.3f} ms, "
            "so the channel measured no bias rather than failing to measure"
        )
    if sd < abs(mean):
        return False, (
            f"the pool's drift samples ({said} ms) have a net step of {mean:+.3f} ms wider than "
            f"their round-to-round spread of {sd:.3f} ms, so the machine has a direction and a "
            "bias does not average out"
        )
    return True, f"drift samples ({said} ms) have no direction: mixed signs, spread the wider"


def median_rule(sols: list[Sol]) -> Budget:
    """Rule 1: what `tinytitan_price_pair.py` already prints for this pool."""
    ys = [s.y for s in sols]
    xs = [s.x for s in sols]
    return Budget(
        "median / widest",
        "median",
        statistics.median(xs),
        statistics.median(ys),
        _ceil_ms(max(s.x_bound for s in sols)),
        _ceil_ms(max(s.y_bound for s in sols)),
        True,
        "distribution-free: the median is no better known than its least known round",
    )


def mean_rule(sols: list[Sol]) -> Budget:
    """Rule 2: the triangle inequality, which bounds a mean and not a median."""
    ys = [s.y for s in sols]
    xs = [s.x for s in sols]
    return Budget(
        "mean / triangle",
        "mean",
        statistics.fmean(xs),
        statistics.fmean(ys),
        _ceil_ms(statistics.fmean(s.x_bound for s in sols)),
        _ceil_ms(statistics.fmean(s.y_bound for s in sols)),
        True,
        "distribution-free: |mean(z_r) - z| <= mean(b_r), so more rounds converge on the "
        "average bound, never on zero",
    )


def sqrt_n_rule(sols: list[Sol]) -> Budget:
    """Rule 3: the only budget that shrinks, and the one that has to earn it.

    The margin it has to clear is the band's own: how far the pool's central price
    sits inside the nearer edge, which is the number the shipped screen refused on.
    `sd` is the pool's round-to-round scatter, so a pool of one has no scatter to
    divide and no budget to claim — and neither has a pool of repeats, which is
    what the licensing test on the drift samples cannot see on its own.
    """
    ok, why = license(sols)
    ys = [s.y for s in sols]
    xs = [s.x for s in sols]
    med_y = statistics.median(ys)
    edge = min((pp.BAND_LO, pp.BAND_HI), key=lambda e: abs(med_y - e))
    margin = abs(med_y - edge)
    sd_y = statistics.stdev(ys) if len(ys) > 1 else 0.0
    sd_x = statistics.stdev(xs) if len(xs) > 1 else 0.0
    n = len(sols)
    t = t_two_sided_95(n - 1)
    hw_y = t * sd_y / math.sqrt(n)
    hw_x = t * sd_x / math.sqrt(n)
    needed = None
    if sd_y > 0.0:
        for k in range(n, MAX_ROUNDS + 1):
            if t_two_sided_95(k - 1) * sd_y / math.sqrt(k) <= margin:
                needed = k
                break
    if not ok:
        refuse = f"not licensed: {why}"
    elif sd_y == 0.0:
        refuse = (
            f"the pool's {n} prices are all {med_y:.3f} ms/token: its drift samples moved "
            "between rounds but the pair did not, so t*sd/sqrt(n) is t*0/sqrt(n) = 0 and the "
            "absence of a measurement would print as the tightest budget in the campaign. "
            "n copies of one price is one price, and a scatter never measured cannot be "
            "divided by root n"
        )
    else:
        refuse = None
    if refuse is not None:
        return Budget(
            "mean / sqrt(n)",
            "mean",
            statistics.fmean(xs),
            statistics.fmean(ys),
            _ceil_ms(hw_x),
            _ceil_ms(hw_y),
            False,
            refuse,
            needed,
        )
    return Budget(
        "mean / sqrt(n)",
        "mean",
        statistics.fmean(xs),
        statistics.fmean(ys),
        _ceil_ms(hw_x),
        _ceil_ms(hw_y),
        True,
        f"licensed by the pool's own controls, and it assumes what they can only test: "
        f"independent per-round error. sd from {n} rounds, so the budget moves again at "
        f"{n + 1} -- re-measure it rather than trusting the tail",
        needed,
    )


def pick_pair(rounds, baseline, column, control) -> tuple[tuple[str, str] | None, str]:
    """The pair every round offers, widest on the tightest round.

    A pool has to answer one question, so a pair is common only where every table
    prices it; among common pairs the binding number is the *smallest* angle in the
    pool, because the round that solves it worst is the one that sets the width.
    """
    sets = []
    for rnd in rounds:
        cleared = pp.best_pair(rnd, baseline, column, None, control).cleared
        sets.append({(p.a, p.b): p.deg for p in cleared})
    common = set.intersection(*(set(s) for s in sets)) if sets else set()
    if not common:
        if len(rounds) == 1:
            return None, "one round offered no design that reaches the shipped angle bar"
        said = "; ".join(
            f"table {i} offers {', '.join(f'{a} x {b}' for a, b in sorted(s)) or 'nothing'}"
            for i, s in enumerate(sets, start=1)
        )
        return None, f"no pair is priced by every round: {said}"
    pair = max(common, key=lambda k: min(s[k] for s in sets))
    return pair, (
        f"{pair[0]} x {pair[1]} is priced by every round, tightest at "
        f"{min(s[pair] for s in sets):.1f} deg"
    )


def pool(rounds, pair, baseline, column, control, label) -> list[Sol]:
    """Solve the named pair in every round, as the shipped screen licenses it."""
    want = set(pair)
    out: list[Sol] = []
    for index, rnd in enumerate(rounds, start=1):
        sel = pp.best_pair(rnd, baseline, column, None, control)
        price = next((p for p in sel.cleared if {p.a, p.b} == want), None)
        if price is None:
            continue
        step = None
        if control in rnd.arms and baseline in rnd.arms:
            step = pp.dm_step(rnd.arms[control], rnd.arms[baseline], column)
        out.append(
            Sol(
                label,
                index,
                (price.a, price.b),
                price.deg,
                price.x,
                price.y,
                price.x_bound,
                price.y_bound,
                step,
            )
        )
    return out


def _budget_line(b: Budget, call: Call) -> str:
    if not b.licensed:
        return f"  {b.rule:<17}: y={b.y:.3f} ±{b.y_bound:.3f} -> {b.why}"
    if not call.band_separable:
        said = (
            f"cannot settle the band, its bound is {b.y_bound:.3f} ms against a margin of "
            f"{call.gap:.3f} ms"
        )
    elif not call.in_band:
        said = f"settles it OUTSIDE {BAND_TEXT} -> the model is refuted"
    elif not call.gate_separable:
        run = f"{call.gate_lo:.3f} to {call.gate_hi:.3f}"
        said = f"settles the band, gate p* runs {run} across L/S = {call.need:.3f}"
    else:
        side = "pays" if b.x / b.y > call.need else "does not pay"
        said = f"settles the band inside {BAND_TEXT}, and the gate {side}"
    return f"  {b.rule:<17}: y={b.y:.3f} ±{b.y_bound:.3f} -> {said}"


def main(argv: list[str] | None = None, stdin: str | None = None) -> int:
    ap = argparse.ArgumentParser(description="pool rounds and ask whether they settle it")
    ap.add_argument("pages", nargs="*", help="sweep pages to pool; default is stdin")
    ap.add_argument("--baseline", default="base", help="the arm every row is stepped against")
    ap.add_argument(
        "--column",
        default=pp.WALL,
        help=f"where the step's ms/token comes from: {pp.WALL} or "
        "any per-token column the page carries",
    )
    ap.add_argument("--control", default="base_again", help="each round's drift sample")
    ap.add_argument("--pair", nargs=2, metavar=("A", "B"), help="the arms to solve in every round")
    ap.add_argument("--s", type=float, default=pp.DEFAULT_S, help="reads suppressed per token")
    ap.add_argument("--l", type=float, default=pp.DEFAULT_L, help="adoptions lost per token")
    args = ap.parse_args(argv)

    texts: list[tuple[str, str]] = []
    if args.pages:
        for name in args.pages:
            try:
                texts.append((name, pathlib.Path(name).read_text(encoding="utf-8")))
            except OSError as exc:
                print(f"NOT MEASURED: cannot read {name}: {exc}")
                return 2
    else:
        texts.append(("<stdin>", stdin if stdin is not None else sys.stdin.read()))

    rounds: list[tuple[str, int, object]] = []
    for name, text in texts:
        for index, rnd in enumerate(pp.read(text), start=1):
            rounds.append((name, index, rnd))
    if not rounds:
        print(
            "NOT MEASURED: no page in this pool carries a table with an `arm` header and the "
            "`tok/s`, `issued/pt` and `adopted/pt` columns a price solve reads"
        )
        return 2
    for name, index, rnd in rounds:
        if args.baseline not in rnd.arms:  # type: ignore[union-attr]
            print(f"NOT MEASURED ({name} table {index}): no {args.baseline} arm to price against")
            return 2
        missing = pp.missing_cells(rnd)
        if missing:
            print(f"NOT MEASURED ({name} table {index}): " + "; ".join(missing))
            return 2

    if len(rounds) < 2:
        print(
            f"cannot settle: this pool is one round ({rounds[0][0]} table {rounds[0][1]}), and "
            "one round is the page it came from -- what it cannot resolve about its own price "
            "does not become resolvable by calling it a pool"
        )
        return 2

    pair = tuple(args.pair) if args.pair else None
    if pair is None:
        pair, why = pick_pair([r for _, _, r in rounds], args.baseline, args.column, args.control)
        if pair is None:
            print(f"NOT MEASURED: {why}")
            return 2
        print(f"pair: {why}")
    else:
        print(f"pair: {pair[0]} x {pair[1]}, named on the command line")

    names: list[str] = []
    for name, _, _ in rounds:
        if name not in names:
            names.append(name)
    sols = pool(
        [r for _, _, r in rounds], pair, args.baseline, args.column, args.control, " + ".join(names)
    )
    if len(sols) != len(rounds):
        priced = {s.index for s in sols}
        gaps = [f"{n} table {i}" for n, i, _ in rounds if i not in priced]
        print(
            f"NOT MEASURED: {', '.join(gaps)} does not price {' x '.join(pair)} -- a pair "
            "singular, below the angle bar or resting on a row inside its own drift is not a "
            "round of this pool, and averaging it in would price a question the page did not ask"
        )
        return 2

    need = args.l / args.s
    print(f"pool of {len(sols)} round(s), column {args.column}, gate L/S = {need:.3f}")
    for s in sols:
        step = "unread" if s.control is None else f"{s.control:+.3f} ms"
        print(
            f"  {s.page} table {s.index}: x={s.x:.3f} y={s.y:.3f} y bound ±{s.y_bound:.3f} "
            f"deg {s.deg:.1f} drift sample {step}"
        )

    med = median_rule(sols)
    budgets = [med, mean_rule(sols), sqrt_n_rule(sols)]
    for b in budgets:
        print(_budget_line(b, score(b, need)))
    last = budgets[-1]
    if last.n_needed is not None and last.n_needed > len(sols):
        print(
            f"  the sqrt(n) budget would clear this margin at {last.n_needed} rounds of the same "
            f"scatter -- {len(sols)} here"
        )
    if not last.licensed:
        print(
            f"  and that number is a planning estimate only: {last.why}. If more rounds are "
            "pooled, their drift samples have to keep pointing different ways for it to mean "
            "anything about this Mac rather than about the arithmetic"
        )

    licensed = [b for b in budgets if b.licensed]
    calls = {b.rule: score(b, need) for b in licensed}
    settled = [
        b
        for b in licensed
        if calls[b.rule].band_separable and calls[b.rule].in_band and calls[b.rule].gate_separable
    ]
    if settled:
        print(
            f"settles: {', '.join(sorted(b.rule for b in settled))} prices this pool past both "
            f"the {BAND_TEXT} margin and the gate, which no single round here could do"
        )
        return 0
    refuted = [b for b in licensed if calls[b.rule].band_separable and not calls[b.rule].in_band]
    if refuted:
        print(
            f"REFUTED: {', '.join(b.rule for b in refuted)} puts y={refuted[0].y:.3f} "
            f"separably outside {BAND_TEXT}, so this pool falsifies the two-price model and no "
            "amount of pooling rescues a model the rows contradict"
        )
        return 1
    band_only = [b for b in licensed if calls[b.rule].band_separable]
    if band_only:
        print(
            "cannot settle: the band call separates, but which side of L/S = "
            f"{need:.3f} a gate draw lands on is still inside the pooled width, so a gate "
            "score from this pool would be an answer about its arithmetic"
        )
        return 2
    if last.licensed:
        print(
            "cannot settle: no licensed rule narrows y enough -- the widest the pool may claim "
            f"distribution-free is the mean of its bounds, and the rule that shrinks, licensed by "
            f"this pool's own drift samples, is still ±{last.y_bound:.3f} against a margin of "
            f"{calls[last.rule].gap:.3f} ms"
        )
        return 2
    print(
        "cannot settle: no licensed rule narrows y enough -- the widest the pool may claim "
        "distribution-free is the mean of its bounds, and the rule that shrinks is refused by "
        "the pool's own drift samples"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
