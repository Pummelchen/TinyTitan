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
answer by the digits the page actually carries — the sweep prints `tok/s` to two,
which is ±0.03 ms on `y` for the campaign's own trio, and a band call that falls
inside that bound is refused rather than decided — and answers with the status
this tree's other drivers use:

    0  the design identifies and `y` is inside the admissible band
    1  it identified a price, and that price refutes the two-price model
    2  it could not solve, with the reason named

Nothing here is measured on hardware — it is arithmetic over a page somebody else
printed — so it runs in CI with no model, no server, no socket, nothing fetched.
"""

from __future__ import annotations

import argparse
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
    # Half-widths the page's printed digits put on each price, in ms. Required, not
    # defaulted: a Price that carries no bound would read as an exact one.
    x_bound: float
    y_bound: float


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
    r1: Row, r2: Row, m1: float, m2: float, widths: tuple[float, float, float, float]
) -> Price | None:
    """Cramer over the two rows, with the condition number and the page's own bound.

    `widths` is the half-width the page's digits put on each input the solve reads:
    the two steps, then one `du` and one `dv`. Each is multiplied by the derivative
    of the price it enters, so the sum is the width the printed page allows the
    answer rather than an estimate of it.
    """
    det = r1.du * r2.dv - r1.dv * r2.du
    n1, n2 = math.hypot(r1.du, r1.dv), math.hypot(r2.du, r2.dv)
    if det == 0.0 or n1 == 0.0 or n2 == 0.0:
        return None
    sin = min(1.0, abs(det) / (n1 * n2))
    hm1, hm2, hdu, hdv = widths
    d2 = det * det
    ny = r1.du * m2 - m1 * r2.du  # y's numerator
    nx = m1 * r2.dv - r1.dv * m2  # x's numerator, the same Cramer numerator as Price.x
    # Each part is (d price / d input, that input's half-width), in the order
    # m1, m2, du1, du2, dv1, dv2. The determinant carries the row coordinates, so
    # their digits reach the price through it as well as through the numerator.
    y_parts = [
        (-r2.du / det, hm1),
        (r1.du / det, hm2),
        ((m2 * det - ny * r2.dv) / d2, hdu),
        ((-m1 * det + ny * r1.dv) / d2, hdu),
        (ny * r2.du / d2, hdv),
        (-ny * r1.du / d2, hdv),
    ]
    x_parts = [
        (r2.dv / det, hm1),
        (-r1.dv / det, hm2),
        (-nx * r2.dv / d2, hdu),
        (nx * r1.dv / d2, hdu),
        ((m1 * det - nx * r1.du) / d2, hdv),
        ((-m2 * det + nx * r2.du) / d2, hdv),
    ]
    return Price(
        a=r1.arm,
        b=r2.arm,
        deg=math.degrees(math.asin(sin)),
        cond=1.0 / sin,
        x=nx / det,
        y=ny / det,
        # |d y / d m| is the other row's du over the determinant, per row.
        dy_per_ms=max(abs(r2.du / det), abs(r1.du / det)),
        x_bound=_round_up(sum(abs(v) * w for v, w in x_parts)),
        y_bound=_round_up(sum(abs(v) * w for v, w in y_parts)),
    )
    return Price(
        a=r1.arm,
        b=r2.arm,
        deg=math.degrees(math.asin(sin)),
        cond=1.0 / sin,
        x=(m1 * r2.dv - r1.dv * m2) / det,
        y=(r1.du * m2 - m1 * r2.du) / det,
        # |d y / d m| is the other row's du over the determinant, per row.
        dy_per_ms=max(abs(r2.du / det), abs(r1.du / det)),
    )


def best_pair(rnd: Round, baseline: str, column: str, holdout: str | None):
    """The widest non-degenerate pair a table offers, and what it stepped over."""
    step_rows = {n: r for n, r in rows(rnd, baseline).items() if n != holdout}
    base = rnd.arms[baseline]
    ms = {n: _dm(rnd.arms[n], base, column) for n in step_rows}
    unmeasured = sorted(n for n, v in ms.items() if v is None)
    priced = {n: float(v) for n, v in ms.items() if v is not None}
    hrow = {n: _dm_ulp(rnd, rnd.arms[n], base, column) for n in priced}
    hcoords = (_ulp(rnd, ISSUED), _ulp(rnd, ADOPTED))
    names = sorted(priced)
    candidates = [
        p
        for p in (
            _solve(
                step_rows[a],
                step_rows[b],
                priced[a],
                priced[b],
                (hrow[a], hrow[b], *hcoords),
            )
            for a, b in zip(names, names[1:], strict=False)
        )
        if p is not None
    ]
    notes = []
    if unmeasured:
        notes.append(f"no {column} step for {', '.join(unmeasured)}")
    zero = sorted(n for n, r in step_rows.items() if r.du == 0.0 and r.dv == 0.0)
    if zero:
        notes.append(f"zero row, stepped over as the drift control: {', '.join(zero)}")
    if not candidates and len(names) < 2:
        notes.append(f"only {len(names)} priceable arm(s): a pair needs two")
    best = max(candidates, key=lambda p: p.deg) if candidates else None
    return best, notes


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
    """What the page's own printed precision says the answer is worth."""
    dec = _digits(rnd, column)
    key = RATE if column == WALL else column
    label = PRINTED.get(key, key)
    return (
        f"digits: the page prints {label} to {dec} decimal{'s' if dec != 1 else ''} "
        f"(±{0.5 * 10.0**-dec:.{dec + 1}f}), so y is known only to ±{pair.y_bound:.3f} ms "
        f"and x to ±{pair.x_bound:.3f} ms"
    )


def _cannot_decide(bound: float, gap: float, med_y: float, edge: float, band: str) -> str:
    """The one call the band edge forces, said whichever side of the edge `y` is on."""
    side = "inside" if BAND_LO <= med_y <= BAND_HI else "outside"
    return (
        f"cannot decide: the page's own digits put ±{bound:.3f} ms on y and it sits only "
        f"{gap:.3f} ms {side} the nearer band edge ({edge:.3f} of {band}), so neither this "
        "band call nor a gate draw from it is separable from the rounding of the rates "
        "the page was read from; more printed decimals is what would settle both"
    )


def _verdict(
    pairs: list[Price], token_ms: float, saved: float, lost: float
) -> tuple[list[str], int]:
    ys = [p.y for p in pairs]
    xs = [p.x for p in pairs]
    med_y, med_x = statistics.median(ys), statistics.median(xs)
    # The widest bound any round's own digits put on `y`: the median of several
    # solves is no better known than the least known of them.
    bound = max(p.y_bound for p in pairs)
    edge = min((BAND_LO, BAND_HI), key=lambda e: abs(med_y - e))
    gap = abs(med_y - edge)
    band = f"{BAND_LO:.3f}-{BAND_HI:.3f} ms"
    lines = [
        f"median x={med_x:.3f} y={med_y:.3f}  "
        f"spread x={max(xs) - min(xs):.3f} y={max(ys) - min(ys):.3f}"
    ]
    if not BAND_LO <= med_y <= BAND_HI:
        lines.append(
            f"REFUTED: y={med_y:.3f} is outside the admissible band {band} ({SERVICE_TEXT}), so "
            "this page falsifies the two-price linear model rather than pricing an expensive "
            "read; the six-link gate is not scored from a refuted price"
        )
        if bound >= gap:
            lines.append(_cannot_decide(bound, gap, med_y, edge, band))
        return lines, 1
    pstar = med_x / med_y
    need = lost / saved
    gain = med_x * saved - med_y * lost
    lines.append(f"ADMISSIBLE: y={med_y:.3f} is inside {band}, so the two rows price both reads")
    lines.append(
        f"p*={pstar:.3f} against the six-link gate's L/S = {lost:.2f}/{saved:.2f} = {need:.3f} -> "
        f"{'pays' if pstar > need else 'does not pay'} ({gain:+.2f} ms of a {token_ms:.1f} ms "
        f"token, {gain / token_ms * 100:+.2f} %)"
    )
    if bound >= gap:
        lines.append(_cannot_decide(bound, gap, med_y, edge, band))
    return lines, 0


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
    for index, rnd in enumerate(rounds, start=1):
        pair, notes = best_pair(rnd, args.baseline, args.column, args.predict)
        if pair is None:
            why = "; ".join(notes) if notes else "nothing priceable"
            print(f"table {index}: REFUSED: no non-degenerate pair of rows to solve ({why})")
            return 2
        if pair.deg < MIN_ANGLE:
            print(
                f"table {index}: REFUSED: the widest pair here, {pair.a} x {pair.b}, is "
                f"{pair.deg:.1f} deg apart (condition {pair.cond:.1f}), below the "
                f"{MIN_ANGLE:.0f} deg a two-row solve needs; no prices"
            )
            return 2
        pairs.append(pair)
        print(
            f"round {index}: best pair {pair.a} x {pair.b} at {pair.deg:.1f} deg "
            f"cond {pair.cond:.1f}  x={pair.x:.3f} y={pair.y:.3f}"
        )
        print(f"1 ms of {args.column} moves y by {pair.dy_per_ms:.3f} ms")
        print(_digits_line(rnd, args.column, pair))
        if notes:
            print(f"  stepped over: {'; '.join(notes)}")
        if args.predict and args.predict in rnd.arms:
            step = rows(rnd, args.baseline).get(args.predict)
            measured = _dm(rnd.arms[args.predict], rnd.arms[args.baseline], args.column)
            if step is not None and measured is not None:
                want = pair.x * step.du + pair.y * step.dv
                print(
                    f"out-of-sample {args.predict}: rows ({step.du:+.2f}, {step.dv:+.2f}) "
                    f"predicted {want:+.2f} ms, measured {measured:+.2f} ms, "
                    f"residual {measured - want:+.2f} ms/token "
                    f"({(measured - want) / token_ms * 100:+.2f} % of a token)"
                )

    if len(pairs) == 1:
        print("rounds 1: a direction needs at least 2 paired tables and a magnitude needs more")
    else:
        print(f"rounds {len(pairs)}: a direction is defensible; a magnitude wants 3 or more")
    lines, status = _verdict(pairs, token_ms, args.s, args.l)
    for line in lines:
        print(line)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
