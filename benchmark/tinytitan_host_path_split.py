#!/usr/bin/env python3
"""Split one decode run's host path into waiting, reading and bookkeeping.

§5w left `shared_expert -> moe_phase1_*` -- 42.835 ms/token of GPU idle -- open
and suggested the host's bookkeeping was sitting behind it, priced against four
named counters. This screen prices it against the whole host path, using the two
nestings the decode loop actually has.

Inside the layer loop, `body_ms` spans the entire iteration
(`RealForwardRunner+DecodeLayers.swift:46` to `+DecodeMoE.swift:472`) and
contains `wait_ms` (the router readback) and `io_ms` (the miss pread), so

    bookkeeping = body - wait - io

is the host time inside the loop that is neither waiting nor reading.

Outside it, the token is longer than the 48 layer bodies stacked together, and
the difference is the per-token prologue and epilogue: `embed_ms`, `gather_ms`,
`pre_ms`, `head_ms` and the three `loop_*` timers accumulate there
(`RealForwardRunner+Decode.swift:156,197,203`, `RawCompletion.swift:305,340,362`),
so they are priced against

    off the loop = wall - body

and never against `bookkeeping`. Summing the two envelopes gives the absolute
bound: every millisecond of the token that is not a blocked wait.

Statuses:
  0  family 2 exceeds even the absolute bound: no host-path change reaches it
  1  host time off the blocked waits reaches family 2: an arm is worth pricing
  2  the page cannot be read: a required line is absent, or an identity fails

Reads a diagnostics log (the run's stderr). Nothing here loads a model, starts a
server or fetches anything.
"""

from __future__ import annotations

import argparse
import re
import sys

# The three terms the in-body identity is made of.
ENVELOPE = ("body_ms", "wait_ms", "io_ms")
# Timers accumulated inside the 48 per-layer body windows.
IN_BODY = (
    "cb1_ms",
    "cb2_ms",
    "rdadvise_ms",
    "cache_plan_ms",
    "router_readback_ms",
)
# Timers accumulated once per token, outside them.
OFF_LOOP = (
    "head_ms",
    "head_fused_ms",
    "embed_ms",
    "gather_ms",
    "pre_ms",
    "loop_sample_ms",
    "loop_progress_ms",
    "loop_other_ms",
)
# Sub-parts of `io_ms`; reported for context, never added to a split.
NESTED_IN_IO = ("io_queue_ms", "io_completion_to_fixup_ms")
# Sub-parts of `pre_ms`, same.
NESTED_IN_PRE = ("pre_release_ms", "pre_pin_ms", "pre_reserve_ms")
# Latency percentiles, not totals: they do not sum to anything.
PERCENTILES = ("expert_load_p50_ms", "expert_load_p95_ms", "expert_load_p99_ms")

CLASSIFIED = ENVELOPE + IN_BODY + OFF_LOOP + NESTED_IN_IO + NESTED_IN_PRE + PERCENTILES

FAMILY2_PREFIX = "shared_expert->"
RUNNER_RE = re.compile(r"^TinyTitan runner (?P<body>.*)$")
GAP_RE = re.compile(
    r"^TinyTitan gap (?P<a>\S+)->(?P<b>\S+) total_ms=(?P<total>[\d.]+) "
    r"per_token_ms=(?P<per>[\d.]+) count=(?P<count>\d+)"
)
# The footer prints `new=` before `decode=`, so the two are matched separately
# rather than as one ordered pattern.
DECODE_RE = re.compile(r"decode=(?P<decode>[\d.]+)s")
NEW_RE = re.compile(r"new=(?P<tokens>\d+)tok")


def _ms_terms(text: str) -> dict[str, float]:
    """Every `key=value` pair on the runner line whose key ends in `_ms`."""
    out: dict[str, float] = {}
    for key, value in re.findall(r"(\w+)=([\d.]+)", text):
        if not key.endswith("_ms"):
            continue
        try:
            out[key] = float(value)
        except ValueError:
            continue
    return out


def parse(stdin: str) -> tuple[dict[str, float], list[tuple[str, float]], float | None]:
    runner: dict[str, float] = {}
    gaps: list[tuple[str, float]] = []
    wall: float | None = None
    for line in stdin.splitlines():
        line = line.strip()
        m = RUNNER_RE.match(line)
        if m:
            runner = _ms_terms(m.group("body"))
            continue
        m = GAP_RE.match(line)
        if m:
            gaps.append((f"{m.group('a')}->{m.group('b')}", float(m.group("per"))))
            continue
        d = DECODE_RE.search(line)
        n = NEW_RE.search(line)
        if d and n and wall is None:
            tokens = int(n.group("tokens"))
            if tokens > 0:
                wall = float(d.group("decode")) * 1000.0 / tokens
    return runner, gaps, wall


def main(argv: list[str] | None = None, stdin: str | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log", required=True, help="diagnostics log, or - for stdin")
    args = ap.parse_args(argv)
    if args.log == "-":
        text = stdin if stdin is not None else sys.stdin.read()
    else:
        with open(args.log, encoding="utf-8", errors="replace") as handle:
            text = handle.read()

    runner, gaps, wall = parse(text)
    if not runner:
        print("NOT MEASURED: no `TinyTitan runner` line, so the host path is unreadable")
        return 2
    missing_env = [k for k in ENVELOPE if k not in runner]
    if missing_env:
        print(
            "NOT MEASURED: the runner line has no "
            + ", ".join(missing_env)
            + ", so the envelope cannot be split"
        )
        return 2
    family2 = [g for g in gaps if g[0].startswith(FAMILY2_PREFIX)]
    if not family2:
        print(
            "NOT MEASURED: no `shared_expert->` gap line, so family 2 is not in this "
            "page (TINYTITAN_KERNEL_STATS=1 prints it)"
        )
        return 2
    if wall is None:
        print(
            "NOT MEASURED: no `new=<n>tok ... decode=<s>s` footer, so the token has no "
            "wall to subtract the layer loop from and the bound would silently narrow"
        )
        return 2

    body, wait, io_ms = (runner[k] for k in ENVELOPE)
    bookkeeping = body - wait - io_ms
    if bookkeeping < 0:
        print(
            f"NOT MEASURED: body={body:.3f} is smaller than wait={wait:.3f} + "
            f"io={io_ms:.3f}, which breaks the nesting the split is built on -- "
            "these counters no longer describe one layer iteration"
        )
        return 2
    off_loop = wall - body
    if off_loop < 0:
        print(
            f"NOT MEASURED: body={body:.3f} ms/token exceeds the wall={wall:.3f} it is "
            "supposed to nest inside, so the layer loop no longer fits the token and "
            "`wall - body` cannot be read as host time"
        )
        return 2

    fam = sum(v for _, v in family2)
    named_in = sum(runner.get(k, 0.0) for k in IN_BODY)
    named_out = sum(runner.get(k, 0.0) for k in OFF_LOOP)
    host_off_waits = bookkeeping + off_loop
    unassigned = sorted(k for k in runner if k not in CLASSIFIED)

    def row(label: str, value: float, note: str) -> None:
        print(f"{label:<14}: {value:.3f} ms/token  ({note})")

    row("body", body, "the whole layer iteration")
    row("wait", wait, "blocked on the router readback, inside body")
    row("io", io_ms, "blocked on the miss pread, inside body")
    row("bookkeeping", bookkeeping, "body - wait - io: host work in the loop that is neither")
    row("family 2", fam, ", ".join(a for a, _ in family2))
    row(
        "named in body",
        named_in,
        f"{len(IN_BODY)} counters; {100 * named_in / bookkeeping:.1f}% of the in-body envelope",
    )
    row("wall", wall, "one token of decode end to end")
    row("off the loop", off_loop, "wall - body: the token outside the layer bodies")
    row(
        "named off loop",
        named_out,
        f"{len(OFF_LOOP)} counters; {100 * named_out / off_loop:.1f}% of that envelope",
    )
    row(
        "host off waits",
        host_off_waits,
        "bookkeeping + off the loop: every millisecond that is not a blocked wait",
    )
    if unassigned:
        print(
            f"{'UNASSIGNED':<14}: the footer carries "
            + ", ".join(f"{k}={runner[k]:.3f}" for k in unassigned)
            + " -- this screen has not classified them, so the identities below are "
            "incomplete"
        )
    for label, keys in (
        ("nested in io", NESTED_IN_IO),
        ("nested in pre", NESTED_IN_PRE),
        ("percentiles", PERCENTILES),
    ):
        present = [k for k in keys if k in runner]
        if present:
            print(
                f"{label:<14}: "
                + ", ".join(f"{k}={runner[k]:.3f}" for k in present)
                + " (not added to a split)"
            )
    for label, named, envelope in (
        ("body = wait + bookkeeping + named in body", named_in, bookkeeping),
        ("wall = body + off the loop, named off loop", named_out, off_loop),
    ):
        residual = abs(envelope - named) / envelope if envelope > 0 else float("inf")
        print(
            f"{'identity':<14}: {label}: the two differ by {residual:.1%} of the envelope"
            + ("" if residual <= 0.5 else "  (more than half an envelope)")
        )

    of_token = f"{100 * bookkeeping / wall:.1f}% of the token"
    if host_off_waits < fam:
        print(
            f"VERDICT: a host-path change inside the layer loop can address at most "
            f"{bookkeeping:.3f} ms/token of its {fam:.3f} "
            f"({100 * bookkeeping / fam:.0f}%, {of_token})"
        )
        print(
            f"            the absolute bound is worse: {host_off_waits:.3f} ms/token is "
            f"everything that is not a blocked wait, and family 2's {fam:.3f} exceeds "
            f"even that by {fam - host_off_waits:.3f}. The rest is queue serialization "
            "behind the same bytes family 1 names, and the host is not on it"
        )
        return 0
    if bookkeeping >= fam:
        print(
            f"VERDICT: bookkeeping {bookkeeping:.3f} ms/token -- {of_token} -- reaches "
            f"family 2's {fam:.3f}: a host-path arm is worth pricing, and the work that "
            "would pay for it is inside the layer loop, where the idle is"
        )
        return 1
    print(
        f"VERDICT: family 2's {fam:.3f} is below the absolute bound "
        f"{host_off_waits:.3f}, so a host-path arm is worth pricing -- but not inside "
        f"the loop: bookkeeping is only {bookkeeping:.3f}, so the host time that could "
        f"reach it is the {off_loop:.3f} ms/token off the layer bodies"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
