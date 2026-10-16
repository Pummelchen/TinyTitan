#!/usr/bin/env python3
"""#207: solve the two-price pair off the page the knob sweep already prints.

The two-price model (§5ar, §5as, §5at) charges a paired decode arm `x` ms of wall
for every speculative read it issues and `y` ms for every adopted read it loses:

    dm = x * du + y * dv        (du, dv) = (dissued, -dadopted) against the baseline

Two such rows are two equations in two unknowns, so a pair prices both — but only
if the rows are far enough apart to be independent, which is what the angle
between them measures. §5as chose arms by that angle and §5at re-derived it from
published counters; both left the arithmetic in a throwaway. This is the shipped
screen: it reads the page `tinytitan_knob_sweep.py` prints (or a markdown table
with the same columns from the measurements doc), names the column it solved
from, reports the condition number and how far 1 ms of wall moves `y`, bounds the
answer by the two widths the page itself carries — how finely its rows were
printed, and how far its own re-measured baseline moved a row with no config
change in it — refuses a band or gate call that falls inside that bound, and scores
a held-out row's residual as the only test of the model that is not Cramer
agreeing with itself. Its answers use the statuses this tree's other drivers do:

    0  the design identifies, `y` is inside the band, and the gate is drawn
    1  it identified a price, and that price or a held-out row refutes the model
    2  it could not solve, or its own widths cannot draw the gate, reason named

The angle bounds a design's *conditioning*; the digits and the drift together bound
its *precision*; neither bounds whether the row moved at all. §5au fed this screen a
real page and its widest-angle rule priced an adopted read at 18.189 ms off a row
whose entire step was smaller than the same page's drift control, so the control
(`base_again`, the baseline re-measured) is never a solve partner, a row lying inside
its step is not signal, and the designs that survive are all printed: when they land
on opposite sides of a band edge the page is refused, because which one the widest
angle happens to pick is not evidence for it. #209 added the other half of the same
control: it is also how far the machine shifts each input the solve reads, so it
enters the bound instead of only the row list.

Nothing here is measured on hardware — it is arithmetic over a page somebody else
printed — so it runs in CI with no model, no server, no socket, nothing fetched.
"""

from __future__ import annotations

import argparse
import itertools
import math
import pathlib
import re
import statistics
import sys
from dataclasses import dataclass, field

# §5ar's band, from that section's table: an adopted read removes exactly one
# 2.641 MiB expert read from the critical path, so `y` is that read's service
# time — the low end at this drive's measured 3.35 GB/s, the high end at a
# pessimistic 2.03 GB/s. A `y` outside it is the model failing, not a costly read.
BAND_LO = 0.788
BAND_HI = 1.301
SERVICE_TEXT = "one 2.641 MiB read at 2.03-3.35 GB/s, the measured wall"

# §5as's own design rule for a trio, quoted by it as the reason a depth-3 arm was
# added: below this the two rows are one row twice, however carefully the solve.
MIN_ANGLE = 20.0

# §5ar's six-link depth-0 list, the out-of-sample split-half of its screen 1:
# 5.30 reads suppressed for 0.82 adoptions lost per token.
DEFAULT_S = 5.30
DEFAULT_L = 0.82

WALL = "wall"
RATE = "rate"
ISSUED = "issued"
ADOPTED = "adopted"
PRINTED = {RATE: "tok/s", ISSUED: "issued/pt", ADOPTED: "adopted/pt"}

# Printed column label -> the key this screen stores it under. A label that is
# not here keeps its own name, so `--column` can address any per-token column a
# page carries (body_ms, wait_ms) without this file guessing what it means.
CANON = {
    "tok/s": RATE,
    "issued/pt": ISSUED,
    "adopted/pt": ADOPTED,
    "prefetch_issued_per_token": ISSUED,
    "prefetch_adopted_per_token": ADOPTED,
    "vs base": "vs_base",
}

_SEP = re.compile(r"^[\s|:-]+$")


def _canon(label: str) -> str:
    return CANON.get(label.strip().strip("`"), label.strip().strip("`"))


@dataclass
class Arm:
    """One printed row, kept as the page's own cells."""

    name: str
    cells: dict[str, float | None] = field(default_factory=dict)

    def need(self, key: str) -> float:
        """A cell this screen has already checked is logged; callers do."""
        return float(self.cells[key])  # type: ignore[arg-type]


@dataclass
class Round:
    """One complete table of one batch."""

    arms: dict[str, Arm]
    columns: list[str] = field(default_factory=list)
    # How many decimals each printed column carries, its coarsest row's. A price
    # read off a 2-decimal rate column is not the same instrument as one read off
    # the 3-decimal rates the published prices were solved from, and this is the
    # only place on the page that says which one a reader has.
    decimals: dict[str, int] = field(default_factory=dict)


@dataclass
class Row:
    """A step's place in the (dissued, -dadopted) plane."""

    arm: str
    du: float
    dv: float


@dataclass(frozen=True)
class Drift:
    """What the page's own re-measured baseline says the machine does to one row.

    The three fields are that row's step in the column the solve reads and in the
    two counters. Since no config change is in it, every input the solve reads is
    as uncertain as this, and it enters the price bound through the same
    derivatives the printed digits do. `None` when the page carries no such row.
    """

    control: str
    dm: float
    du: float
    dv: float


@dataclass
class Price:
    """One pair's solve, with the geometry that licenses it."""

    a: str
    b: str
    deg: float
    cond: float
    x: float
    y: float
    dy_per_ms: float
    # Half-widths the page puts on each price, in ms, split by where they come from:
    # `*_digits` is how finely the row was printed, `*_drift` is how far the same
    # page's re-measured control moved a row with no config change, and `*_bound` is
    # the total a reader is allowed. Required, not defaulted: a Price that carries no
    # bound would read as an exact one.
    x_digits: float
    y_digits: float
    x_drift: float
    y_drift: float
    x_bound: float
    y_bound: float
    # Difference-mode sensitivity: ms of price per ms of width on ONE row, with the
    # two rows' widths independent so their errors add. `dy_per_ms` above is the
    # single-row maximum, which understates what a moving machine does to a solve;
    # the common mode, where both rows shift together, largely cancels in `det`.
    x_diff_per_ms: float
    y_diff_per_ms: float


@dataclass
class OffSample:
    """A row the solve never saw, priced by the model and measured on the same page.

    Two rows pin two prices exactly, so they can never test the model; a third is the
    only thing that can. `width` is everything this page cannot resolve about that
    row's own step, which is what the residual has to clear before it is a finding.
    """

    arm: str
    du: float
    dv: float
    want: float
    measured: float
    width: float


@dataclass
class Selection:
    """What one table offers a solve, and what this screen stepped over to offer it.

    `cleared` is every design that both solves and reaches `MIN_ANGLE`, widest
    first, not only the selected one: §5au showed a page whose designs price the
    same read on opposite sides of a band edge, and a screen that prints one
    number cannot say that the page it read is the reason.
    """

    best: Price | None = None
    cleared: list[Price] = field(default_factory=list)
    offers: int = 0
    signal: str = ""
    notes: list[str] = field(default_factory=list)
    # The page's own re-measured baseline, which _solve turns into widths. None when
    # no such row was on the page, which leaves the bound on the printed digits alone.
    drift: Drift | None = None


def _num(text: str) -> float | None:
    body = text.strip().strip("`").replace(",", "").replace("%", "")
    if body in ("", "--", "-", "FAILED", "not logged", "n/a", "None"):
        return None
    try:
        return float(body)
    except ValueError:
        return None


def _cells(line: str) -> list[str]:
    """A printed line's raw cells, pipe-separated markdown or whitespace columns."""
    if line.lstrip().startswith("|"):
        return [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
    return line.split()


def _decimals(cell: str) -> int | None:
    """How many decimals one printed cell carries; None when it is not a number."""
    body = cell.strip().strip("`").rstrip("%").strip()
    if _num(body) is None:
        return None
    return len(body.partition(".")[2])


def _is_separator(line: str) -> bool:
    return "-" in line and _SEP.match(line.strip()) is not None


def _header(line: str) -> list[str] | None:
    """The column labels of a table header, or None when this line is not one."""
    if line.lstrip().startswith("|"):
        labels = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
    else:
        tokens = line.split()
        if len(tokens) < 3:
            return None
        labels = []
        i = 0
        while i < len(tokens):
            # The sweep's `vs base` header spans two words over one column.
            if tokens[i] == "vs" and i + 1 < len(tokens):
                labels.append("vs base")
                i += 2
            else:
                labels.append(tokens[i])
                i += 1
    if not labels or _canon(labels[0]) != "arm":
        return None
    return labels


def _row(line: str, labels: list[str]) -> tuple[str, dict, dict] | None:
    """One data row as (arm name, cells, printed decimals), or None when not one."""
    if _is_separator(line):
        return None
    cells = _cells(line)
    if len(cells) != len(labels):
        return None
    name = cells[0].strip()
    if not name or _canon(name) == "arm":
        return None
    pairs = list(zip(labels, cells, strict=True))
    keyed = {_canon(lab): _num(cell) for lab, cell in pairs}
    # A row with no rate is not an arm that measured: it is a `FAILED` line, a
    # wrapped note, or prose that happens to have the right number of words.
    if keyed.get(RATE) is None:
        return None
    return name, keyed, {_canon(lab): _decimals(cell) for lab, cell in pairs}


def read(text: str) -> list[Round]:
    """Every complete table of a page, oldest first.

    The sweep prints its table once per arm, cumulatively, so a four-arm batch
    emits four tables and only the last is the batch. A table whose arm set is a
    proper subset of a later one is therefore a partial, not a round; two tables
    over the same arms are two rounds of one batch.
    """
    lines = text.splitlines()
    rounds: list[Round] = []
    i = 0
    while i < len(lines):
        labels = _header(lines[i])
        if labels is None or i + 1 >= len(lines) or not _is_separator(lines[i + 1]):
            i += 1
            continue
        arms: dict[str, Arm] = {}
        decs: dict[str, int] = {}
        j = i + 2
        while j < len(lines):
            line = lines[j]
            if not line.strip() or _header(line) is not None:
                break
            row = _row(line, labels)
            if row is not None:
                arms[row[0]] = Arm(row[0], row[1])
                for key, n in row[2].items():
                    # Coarsest row wins, so a column printed two ways cannot be
                    # read as the more precise of the two.
                    if n is not None and (key not in decs or n < decs[key]):
                        decs[key] = n
            j += 1
        if arms:
            rounds.append(Round(arms, [_canon(lab) for lab in labels], decs))
        i = max(j, i + 1)
    return [
        r
        for k, r in enumerate(rounds)
        if not any(set(r.arms) < set(later.arms) for later in rounds[k + 1 :])
    ]


def rows(rnd: Round, baseline: str) -> dict[str, Row]:
    """Each arm's row against the baseline.

    `du` is signed arm minus baseline; `dv` is the negation of that same step, so
    a move that makes adopted reads *fall* has a positive second coordinate. That
    is §5ar's E1 (`18.86 y - 39.45 x = 11.7 ms`) for the ring-on -> ring-off step,
    and the prices published from it are only reproducible this way round: §5at's
    geometry table prints the same row negated, which reads the identical angle
    and solves the opposite verdict.
    """
    base = rnd.arms[baseline]
    out: dict[str, Row] = {}
    for name, arm in rnd.arms.items():
        if name == baseline:
            continue
        pair = (arm.cells.get(ISSUED), arm.cells.get(ADOPTED))
        anchor = (base.cells.get(ISSUED), base.cells.get(ADOPTED))
        if any(v is None for v in pair + anchor):
            continue
        out[name] = Row(name, pair[0] - anchor[0], anchor[1] - pair[1])  # type: ignore[operator]
    return out


def _dm(arm: Arm, base: Arm, column: str) -> float | None:
    """The step's cost in ms/token, read from the named column."""
    if column == WALL:
        if arm.cells.get(RATE) is None or base.cells.get(RATE) is None:
            return None
        return 1000.0 / arm.need(RATE) - 1000.0 / base.need(RATE)
    if arm.cells.get(column) is None or base.cells.get(column) is None:
        return None
    return float(arm.cells[column] - base.cells[column])  # type: ignore[operator]


def _digits(rnd: Round, column: str) -> int:
    """How many decimals the page prints the column a step is read from.

    `wall` is not a printed column: it is the reciprocal of `tok/s`, so its digits
    are the rate column's. A column this screen cannot find carries no digits at
    all, and zero is the honest default -- the widest bound, the one that refuses
    to decide anything.
    """
    return rnd.decimals.get(RATE if column == WALL else column, 0)


def _ulp(rnd: Round, column: str) -> float:
    """Half the last place of a printed column, in that column's own units."""
    return 0.5 * 10.0 ** -_digits(rnd, column)


def _dm_ulp(rnd: Round, arm: Arm, base: Arm, column: str) -> float:
    """What the page's digits put on one row's step, in ms/token.

    A rate column enters the step through a reciprocal, `1000/r`, so the same
    half-ulp costs more wall the slower the arm is: `d(1000/r) = 1000 dr / r^2`.
    """
    if column == WALL:
        return _ulp(rnd, RATE) * (1000.0 / arm.need(RATE) ** 2 + 1000.0 / base.need(RATE) ** 2)
    return 2.0 * _ulp(rnd, column)


def _round_up(ms: float) -> float:
    """A bound rounded up to the millimetre, so printing it cannot understate it."""
    return math.ceil(ms * 1000.0) / 1000.0


def _solve(
    r1: Row,
    r2: Row,
    m1: float,
    m2: float,
    widths: tuple[float, float, float, float],
    drift: Drift | None,
) -> Price | None:
    """Cramer over the two rows, with the condition number and the page's own bound.

    `widths` is the half-width the page's digits put on each input the solve reads:
    the two steps, then one `du` and one `dv`. `drift` is how far the page's own
    re-measured control moved the same four inputs with no config change in it. Each
    width is multiplied by the derivative of the price it enters, so each sum is the
    width that source allows the answer, and their total is the bound a reader has to
    work to.
    """
    det = r1.du * r2.dv - r1.dv * r2.du
    n1, n2 = math.hypot(r1.du, r1.dv), math.hypot(r2.du, r2.dv)
    if det == 0.0 or n1 == 0.0 or n2 == 0.0:
        return None
    sin = min(1.0, abs(det) / (n1 * n2))
    hm1, hm2, hdu, hdv = widths
    dm, ddu, ddv = (drift.dm, drift.du, drift.dv) if drift else (0.0, 0.0, 0.0)
    d2 = det * det
    ny = r1.du * m2 - m1 * r2.du  # y's numerator
    nx = m1 * r2.dv - r1.dv * m2  # x's numerator, the same Cramer numerator as Price.x
    # Each tuple is the price's derivative with respect to one input, in the order
    # m1, m2, du1, du2, dv1, dv2. The determinant carries the row coordinates, so
    # their digits reach the price through it as well as through the numerator.
    y_d = (
        -r2.du / det,
        r1.du / det,
        (m2 * det - ny * r2.dv) / d2,
        (-m1 * det + ny * r1.dv) / d2,
        ny * r2.du / d2,
        -ny * r1.du / d2,
    )
    x_d = (
        r2.dv / det,
        -r1.dv / det,
        -nx * r2.dv / d2,
        nx * r1.dv / d2,
        (m1 * det - nx * r1.du) / d2,
        (-m2 * det + nx * r2.du) / d2,
    )
    # Both rows are read off the same columns, so each counter width applies twice;
    # the drift step is the same machine moving either row, so `dm` does too.
    w_digits = (hm1, hm2, hdu, hdu, hdv, hdv)
    w_drift = (dm, dm, ddu, ddu, ddv, ddv)
    y_digits, y_drift = _spread(y_d, w_digits), _spread(y_d, w_drift)
    x_digits, x_drift = _spread(x_d, w_digits), _spread(x_d, w_drift)
    return Price(
        a=r1.arm,
        b=r2.arm,
        deg=math.degrees(math.asin(sin)),
        cond=1.0 / sin,
        x=nx / det,
        y=ny / det,
        # |d y / d m| is the other row's du over the determinant, per row.
        dy_per_ms=max(abs(r2.du / det), abs(r1.du / det)),
        x_digits=_round_up(x_digits),
        y_digits=_round_up(y_digits),
        x_drift=_round_up(x_drift),
        y_drift=_round_up(y_drift),
        x_bound=_round_up(x_digits + x_drift),
        y_bound=_round_up(y_digits + y_drift),
        x_diff_per_ms=(abs(r1.dv) + abs(r2.dv)) / abs(det),
        y_diff_per_ms=(abs(r1.du) + abs(r2.du)) / abs(det),
    )


def _spread(derivs: tuple[float, ...], widths: tuple[float, ...]) -> float:
    """The half-width these input widths put on a price.

    First-order propagation with the inputs independent, which is the worst case: a
    machine that moved one row and not the other is what the drift sample measures,
    so their errors add rather than cancel."""
    return sum(abs(d) * w for d, w in zip(derivs, widths, strict=True))


def _band_class(y: float) -> str:
    """Which side of the admissible band a price lands on."""
    if y < BAND_LO:
        return "below"
    if y > BAND_HI:
        return "above"
    return "inside"


def _drift_sample(
    rnd: Round, step_rows: dict[str, Row], priced: dict[str, float], control: str, column: str
) -> tuple[dict[str, Row], list[str], str, Drift | None]:
    """The rows the page's own drift cannot explain, the ones it inked, the account,
    and the drift itself as a `Drift`.

    `control` is the sweep's `base_again`: the baseline re-measured, so its step is
    this page's drift with no config change in it. A row that moves less than that
    -- in both counters and in the column the solve reads -- is inside the drift,
    and a pair built on it is conditioning without signal.

    The same control step is the page's own measurement of how far one row's inputs
    move with no config change, so it is returned for _solve to carry into the bound.
    #209 found the screen printing `y is known only to ±0.034 ms` on a page whose
    control had just moved a row 11.17 ms, because the drift chose the row list and
    never reached the price.
    """
    if control not in rnd.arms:
        return (
            step_rows,
            [],
            f"no drift sample ({control} is not on this page), so no row was checked",
            None,
        )
    row, ms = step_rows.get(control), priced.get(control)
    if row is None or ms is None:
        label = PRINTED.get(column, column)
        return (
            step_rows,
            [],
            f"no drift sample ({control} logged no {label} step or no counters), "
            "so no row was checked",
            None,
        )
    live = {
        n: r
        for n, r in step_rows.items()
        if n != control
        and not (
            abs(r.du) <= abs(row.du) and abs(r.dv) <= abs(row.dv) and abs(priced[n]) <= abs(ms)
        )
    }
    barred = sorted(set(step_rows) - set(live) - {control})
    inside = (
        f"inside it, not signal, stepped over: {', '.join(barred)}"
        if barred
        else "no row lies inside it"
    )
    moved = f"{row.du:+.2f}, {row.dv:+.2f}"
    drift = Drift(control=control, dm=abs(ms), du=abs(row.du), dv=abs(row.dv))
    return (
        live,
        barred,
        f"drift sample {control} moved ({moved}) and {ms:+.2f} ms; {inside}",
        drift,
    )


def best_pair(
    rnd: Round, baseline: str, column: str, holdout: str | None, control: str = "base_again"
) -> Selection:
    """Everything one table offers a solve: the widest row with signal, and the rest.

    Every pair of surviving rows is a candidate -- a sorted list walked two at a
    time is a window over arm names, not a search.
    """
    all_rows = {n: r for n, r in rows(rnd, baseline).items() if n != holdout}
    base = rnd.arms[baseline]
    ms = {n: _dm(rnd.arms[n], base, column) for n in all_rows}
    unmeasured = sorted(n for n, v in ms.items() if v is None)
    priced = {n: float(v) for n, v in ms.items() if v is not None}
    step_rows, inked, signal, drift = _drift_sample(rnd, all_rows, priced, control, column)
    hrow = {n: _dm_ulp(rnd, rnd.arms[n], base, column) for n in priced}
    hcoords = (_ulp(rnd, ISSUED), _ulp(rnd, ADOPTED))
    candidates = [
        p
        for p in (
            _solve(
                step_rows[a],
                step_rows[b],
                priced[a],
                priced[b],
                (hrow[a], hrow[b], *hcoords),
                drift,
            )
            for a, b in itertools.combinations(sorted(step_rows), 2)
        )
        if p is not None
    ]
    notes = []
    if unmeasured:
        notes.append(f"no {column} step for {', '.join(unmeasured)}")
    zero = sorted(n for n, r in all_rows.items() if r.du == 0.0 and r.dv == 0.0 and n != control)
    if zero:
        notes.append(f"zero row, stepped over as a second drift control: {', '.join(zero)}")
    if not candidates:
        if inked and len(step_rows) < 2:
            notes.append(
                f"the drift sample stepped over {len(inked)} row(s), leaving "
                f"{len(step_rows)} with signal: a pair needs two"
            )
        elif len(priced) < 2:
            notes.append(f"only {len(priced)} priceable arm(s): a pair needs two")
    cleared = sorted(
        (p for p in candidates if p.deg >= MIN_ANGLE), key=lambda p: p.deg, reverse=True
    )
    best = max(candidates, key=lambda p: p.deg) if candidates else None
    return Selection(
        best=best,
        cleared=cleared,
        offers=len(candidates),
        signal=signal,
        notes=notes,
        drift=drift,
    )


def _designs_line(sel: Selection) -> str:
    """Each design the page can price, named, so a reader sees the disagreement."""
    said = ", ".join(f"y={p.y:.3f} ({p.a} x {p.b}, {p.deg:.1f} deg)" for p in sel.cleared)
    spread = f"; y spread {max(p.y for p in sel.cleared) - min(p.y for p in sel.cleared):.3f} ms"
    return (
        f"designs: {len(sel.cleared)} of {sel.offers} solvable pair(s) reach "
        f"{MIN_ANGLE:.0f} deg: {said}{spread if len(sel.cleared) > 1 else ''}"
    )


def _disagreement(sel: Selection) -> str | None:
    """Refuse a page whose own designs do not agree on which side of the band it is on."""
    if len(sel.cleared) < 2:
        return None
    kinds = {_band_class(p.y) for p in sel.cleared}
    if len(kinds) < 2:
        return None
    band = f"{BAND_LO:.3f}-{BAND_HI:.3f} ms"
    said = ", ".join(f"y={p.y:.3f} is {_band_class(p.y)} it" for p in sel.cleared)
    return (
        f"REFUSED: the {len(sel.cleared)} designs this page can price do not agree on the band "
        f"({band}): {said}, so this page does not have one price for an adopted read and the "
        "widest angle picking among them is a property of the arm names, not of the machine; "
        "nothing here is a measurement of y"
    )


def _missing(rnd: Round) -> list[str]:
    out = []
    for key in (RATE, ISSUED, ADOPTED):
        if key not in rnd.columns:
            out.append(f"the page has no {PRINTED[key]} column")
            continue
        unlogged = sorted(n for n, a in rnd.arms.items() if a.cells.get(key) is None)
        if unlogged:
            out.append(f"the {', '.join(unlogged)} arm(s) logged no {PRINTED[key]}")
    return out


def _digits_line(rnd: Round, column: str, pair: Price) -> str:
    """What the page's own printed precision says the answer is worth.

    Only the digits: the drift is a different width of a different thing, and the
    two together get their own line. A total printed under this sentence would
    read as though the number of decimals cost a reader a quarter of a token.
    """
    dec = _digits(rnd, column)
    key = RATE if column == WALL else column
    label = PRINTED.get(key, key)
    return (
        f"digits: the page prints {label} to {dec} decimal{'s' if dec != 1 else ''} "
        f"(±{0.5 * 10.0**-dec:.{dec + 1}f}), so y is known only to ±{pair.y_digits:.3f} ms "
        f"and x to ±{pair.x_digits:.3f} ms"
    )


def _drift_line(sel: Selection, pair: Price, column: str) -> str | None:
    """What the page's own machine says the answer is worth, named apart from its digits.

    The digits bound how finely a row was printed; only the re-measured baseline
    bounds how far the machine moves a row that did not change. On §5au's page the
    first was ±0.034 ms and the second a quarter of a token, so one number would have
    been a bound on the wrong thing.
    """
    drift = sel.drift
    if drift is None:
        return None
    label = PRINTED.get(column, column)
    return (
        f"drift: {drift.control} is this page's baseline re-measured, so its step "
        f"(±{drift.dm:.3f} ms of {label}, ±{drift.du:.2f} issued, ±{drift.dv:.2f} adopted) is "
        f"what the machine does to a row with no config change in it; difference-mode "
        f"sensitivities {pair.y_diff_per_ms:.3f} ms of y and {pair.x_diff_per_ms:.3f} ms of x "
        f"per ms of drift, so drift alone puts ±{pair.y_drift:.3f} ms on y and "
        f"±{pair.x_drift:.3f} on x where the digits put ±{pair.y_digits:.3f} and "
        f"±{pair.x_digits:.3f}; the digits and drift together put ±{pair.y_bound:.3f} ms on y "
        f"and ±{pair.x_bound:.3f} on x, and that is the bound this page may price by"
    )


def _pi_interval(x: float, wx: float, y: float, wy: float) -> tuple[float, float | None]:
    """The interval `x/y` covers once each price carries its own width.

    An empty lower half -- `y - wy <= 0` -- means the ratio is unbounded: the page
    cannot say the adopted read costs anything at all, so no threshold on `p*` means
    anything either, and the caller gets None rather than a made-up number.
    """
    if y - wy <= 0.0:
        return ((x - wx) / y if y > 0 else 0.0, None)
    return ((x - wx) / (y + wy), (x + wx) / (y - wy))


def _off_sample(o: OffSample, token_ms: float) -> tuple[str, bool]:
    """Score a held-out row: the only test of the model that is not Cramer agreeing
    with itself, bounded by everything this page cannot resolve."""
    res = o.measured - o.want
    refuted = abs(res) > o.width
    said = (
        f"out-of-sample {o.arm}: rows ({o.du:+.2f}, {o.dv:+.2f}) predicted {o.want:+.2f} ms, "
        f"measured {o.measured:+.2f} ms, residual {res:+.2f} ms/token "
        f"({res / token_ms * 100:+.2f} % of a token) against "
        f"{o.width:.3f} ms this page cannot resolve"
    )
    if refuted:
        return (
            said + f": REFUTED OUT OF SAMPLE, the pair that fits its own two rows exactly misses a "
            f"row it never solved by {abs(res) / o.width:.1f}x the page's own width, so the "
            "two-price model does not hold on this page and no gate draw from it is an answer",
            True,
        )
    return (
        said + ": out-of-sample consistent, a model fitted to two rows predicted a third it never "
        "solved to inside the page's own width, which is the strongest thing this design can "
        "say about it",
        False,
    )


def _cannot_decide(
    bound: float,
    digits: float,
    drift: float,
    gap: float,
    med_y: float,
    edge: float,
    band: str,
) -> str:
    """The one call the band edge forces, said whichever side of the edge `y` is on.

    Naming which half of the bound dominates says what would settle it: a page
    printed too coarsely is fixed by more decimals, and a page whose own control
    moved a quarter of a token is not, so the same width in two different
    provenances calls for two different pieces of advice.
    """
    side = "inside" if BAND_LO <= med_y <= BAND_HI else "outside"
    if drift > 0.0:
        parts = f"its digits ±{digits:.3f} ms and its own drift ±{drift:.3f}"
        remedy = (
            "more printed decimals is what would settle both"
            if digits >= drift
            else "printing more decimals would not settle it, because the machine's own drift "
            "is the wider half of what this page cannot resolve"
        )
    else:
        parts = f"the printing alone, ±{digits:.3f} ms"
        remedy = "more printed decimals is what would settle both"
    return (
        f"cannot decide: this page's bound on y is ±{bound:.3f} ms -- {parts} -- and it sits "
        f"only {gap:.3f} ms {side} the nearer band edge ({edge:.3f} of {band}), so neither this "
        "band call nor a gate draw from it is separable from what the page can resolve; "
        f"{remedy}"
    )


def _cannot_decide_gate(
    lo: float, hi: float | None, need: float, med_x: float, wx: float, med_y: float, wy: float
) -> str:
    """Refuse the six-link gate when the page's own widths straddle its threshold."""
    if hi is None:
        said = f"from {lo:.3f} up, unbounded, because y - ±{wy:.3f} is not finite"
        why = "the page does not price an adopted read at all, so no ratio of it has a value"
    else:
        said = f"from {lo:.3f} to {hi:.3f}"
        why = "the threshold lies inside the ratio's own interval"
    return (
        f"cannot decide the gate: x={med_x:.3f} ±{wx:.3f} and y={med_y:.3f} ±{wy:.3f} make p* run "
        f"{said} against the six-link gate's L/S = {need:.3f}, and {why}; the band call above "
        "stands on its own, but which side of this threshold a gate draw lands on is not "
        "separable from the widths this page carries, so no gate score is printed"
    )


def _verdict(
    pairs: list[Price],
    token_ms: float,
    saved: float,
    lost: float,
    off: tuple[OffSample, ...] = (),
) -> tuple[list[str], int]:
    """The band call, the gate call the widths license, and the out-of-sample score."""
    ys = [p.y for p in pairs]
    xs = [p.x for p in pairs]
    med_y, med_x = statistics.median(ys), statistics.median(xs)
    # The widest bound any round puts on each price: the median of several solves is
    # no better known than the least known of them. `worst` is that round, kept whole
    # so a caveat can say what its bound is made of rather than only how wide it is.
    worst = max(pairs, key=lambda p: p.y_bound)
    bound = worst.y_bound
    xbound = max(p.x_bound for p in pairs)
    edge = min((BAND_LO, BAND_HI), key=lambda e: abs(med_y - e))
    gap = abs(med_y - edge)
    band = f"{BAND_LO:.3f}-{BAND_HI:.3f} ms"
    lines = [
        f"median x={med_x:.3f} y={med_y:.3f}  "
        f"spread x={max(xs) - min(xs):.3f} y={max(ys) - min(ys):.3f}"
    ]
    need = lost / saved
    if not BAND_LO <= med_y <= BAND_HI:
        lines.append(
            f"REFUTED: y={med_y:.3f} is outside the admissible band {band} ({SERVICE_TEXT}), so "
            "this page falsifies the two-price linear model rather than pricing an expensive "
            "read; the six-link gate is not scored from a refuted price"
        )
        if bound >= gap:
            lines.append(
                _cannot_decide(bound, worst.y_digits, worst.y_drift, gap, med_y, edge, band)
            )
        refuted = [_off_sample(o, token_ms) for o in off]
        lines.extend(t for t, _ in refuted)
        return lines, 1
    lines.append(f"ADMISSIBLE: y={med_y:.3f} is inside {band}, so the two rows price both reads")
    if bound >= gap:
        lines.append(_cannot_decide(bound, worst.y_digits, worst.y_drift, gap, med_y, edge, band))
    scored = [_off_sample(o, token_ms) for o in off]
    lines.extend(t for t, _ in scored)
    if any(r for _, r in scored):
        lines.append(
            "the six-link gate is not scored from a model this page's own held-out row refutes, "
            "however well its two solved rows fit"
        )
        return lines, 1
    lo, hi = _pi_interval(med_x, xbound, med_y, bound)
    gate_scored = hi is not None and not lo <= need <= hi
    if gate_scored:
        pstar = med_x / med_y
        gain = med_x * saved - med_y * lost
        lines.append(
            f"p*={pstar:.3f} against the six-link gate's L/S = {lost:.2f}/{saved:.2f} = "
            f"{need:.3f} -> {'pays' if pstar > need else 'does not pay'} "
            f"({gain:+.2f} ms of a {token_ms:.1f} ms token, {gain / token_ms * 100:+.2f} %)"
        )
    else:
        lines.append(_cannot_decide_gate(lo, hi, need, med_x, xbound, med_y, bound))
    return lines, 0 if gate_scored else 2


def main(argv: list[str] | None = None, stdin: str | None = None) -> int:
    ap = argparse.ArgumentParser(description="price x and y off a knob-sweep page")
    ap.add_argument("--page", help="the sweep page to read; default is stdin")
    ap.add_argument("--baseline", default="base", help="the arm every row is stepped against")
    ap.add_argument(
        "--column",
        default=WALL,
        help=f"where the step's ms/token comes from: {WALL} (1000/tok/s) or any per-token "
        "column the page carries, e.g. body_ms",
    )
    ap.add_argument("--predict", help="an arm held out of the solve and priced as out-of-sample")
    ap.add_argument(
        "--control",
        default="base_again",
        help="the drift control's arm name: its step is this page's own drift, so it is "
        "never a solve partner and a row whose step lies inside it is not signal",
    )
    ap.add_argument("--s", type=float, default=DEFAULT_S, help="reads suppressed per token")
    ap.add_argument("--l", type=float, default=DEFAULT_L, help="adoptions lost per token")
    ap.add_argument("--token-ms", type=float, help="default: the baseline's own 1000/tok/s")
    args = ap.parse_args(argv)

    text = stdin
    if text is None:
        if args.page:
            try:
                text = pathlib.Path(args.page).read_text(encoding="utf-8")
            except OSError as exc:
                print(f"NOT MEASURED: cannot read --page {args.page}: {exc}")
                return 2
        else:
            text = sys.stdin.read()

    rounds = read(text)
    if not rounds:
        print(
            "NOT MEASURED: the page carries no table with an `arm` header and the "
            "`tok/s`, `issued/pt` and `adopted/pt` columns a price solve reads"
        )
        return 2

    for index, rnd in enumerate(rounds, start=1):
        if args.column != WALL and args.column not in rnd.columns:
            others = [c for c in rnd.columns if c not in ("arm", "vs_base")]
            print(
                f"NOT MEASURED (table {index}): the page carries no {args.column} column, so "
                f"--column {args.column} would be a guess; its per-token columns are "
                f"{', '.join(others) or 'none'}"
            )
            return 2
        if args.baseline not in rnd.arms:
            print(
                f"NOT MEASURED (table {index}): no {args.baseline} arm to price against; "
                f"this table's arms are {', '.join(sorted(rnd.arms))}"
            )
            return 2
        missing = _missing(rnd)
        if missing:
            print(f"NOT MEASURED (table {index}): " + "; ".join(missing))
            return 2

    if args.predict and all(args.predict not in r.arms for r in rounds):
        print(f"NOT MEASURED: --predict {args.predict} names no arm on this page")
        return 2

    token_ms = args.token_ms
    if token_ms is None:
        token_ms = 1000.0 / rounds[0].arms[args.baseline].need(RATE)

    print(f"column {args.column}  (ms/token stepped against {args.baseline})")
    pairs: list[Price] = []
    off: list[OffSample] = []
    for index, rnd in enumerate(rounds, start=1):
        sel = best_pair(rnd, args.baseline, args.column, args.predict, args.control)
        pair, notes = sel.best, sel.notes
        detail = [f"  signal: {sel.signal}"]
        if sel.cleared:
            detail.append(f"  {_designs_line(sel)}")
        if pair is None:
            why = "; ".join(notes) if notes else "nothing priceable"
            print(f"table {index}: REFUSED: no non-degenerate pair of rows to solve ({why})")
            print("\n".join(detail))
            return 2
        if pair.deg < MIN_ANGLE:
            print(
                f"table {index}: REFUSED: the widest pair here, {pair.a} x {pair.b}, is "
                f"{pair.deg:.1f} deg apart (condition {pair.cond:.1f}), below the "
                f"{MIN_ANGLE:.0f} deg a two-row solve needs; no prices"
            )
            print("\n".join(detail))
            return 2
        pairs.append(pair)
        print(
            f"round {index}: best pair {pair.a} x {pair.b} at {pair.deg:.1f} deg "
            f"cond {pair.cond:.1f}  x={pair.x:.3f} y={pair.y:.3f}"
        )
        print(f"1 ms of {args.column} moves y by {pair.dy_per_ms:.3f} ms")
        print(_digits_line(rnd, args.column, pair))
        drift_line = _drift_line(sel, pair, args.column)
        if drift_line is not None:
            print(f"  {drift_line}")
        print("\n".join(detail))
        if notes:
            print(f"  stepped over: {'; '.join(notes)}")
        refused = _disagreement(sel)
        if refused is not None:
            print(f"table {index}: {refused}")
            return 2
        if args.predict and args.predict in rnd.arms:
            step = rows(rnd, args.baseline).get(args.predict)
            measured = _dm(rnd.arms[args.predict], rnd.arms[args.baseline], args.column)
            if step is not None and measured is not None:
                # The residual is a difference of two numbers, each with its own width:
                # this row's step, digits plus whatever the machine moved it, and the
                # prediction, which inherits both prices' bounds through the row's
                # coordinates. A page that cannot resolve 5 ms may not call a 5 ms miss
                # a refutation.
                dm = sel.drift.dm if sel.drift else 0.0
                width = (
                    _dm_ulp(rnd, rnd.arms[args.predict], rnd.arms[args.baseline], args.column)
                    + dm
                    + abs(step.du) * pair.x_bound
                    + abs(step.dv) * pair.y_bound
                )
                off.append(
                    OffSample(
                        arm=args.predict,
                        du=step.du,
                        dv=step.dv,
                        want=pair.x * step.du + pair.y * step.dv,
                        measured=measured,
                        width=_round_up(width),
                    )
                )

    if len(pairs) == 1:
        print("rounds 1: a direction needs at least 2 paired tables and a magnitude needs more")
    else:
        print(f"rounds {len(pairs)}: a direction is defensible; a magnitude wants 3 or more")
    lines, status = _verdict(pairs, token_ms, args.s, args.l, tuple(off))
    for line in lines:
        print(line)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
