"""Host-relative GPU contention gauge (engineering note §5bd).

The shipped benchmark preflight refuses a run above a fixed 15 % device utilisation, which this
Mac has never satisfied: the 2026-10-16 probe measured an idle floor of 56-68 % and a sustained
1 830 GFLOP/s matmul load at 90-97 %. The reader is therefore not blind -- its bands are disjoint
-- but its zero point is the desktop's, not the machine's, so an absolute threshold is a claim
about a host that is not this one.

This tool measures the host it is on: a floor window, a run window, and a verdict that says which
of the two the data can actually support. It changes no gate and refuses nothing; a run is bounded
by what it reports. Model-free -- the pure functions are what the suite pins, and only
`read_samples` and `main` touch `ioreg`.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence

UTILIZATION_KEY = "Device Utilization %"
MIN_SAMPLES = 3  # fewer than this cannot describe a band, so it cannot describe a verdict


def parse_utilization(text: str) -> list[int]:
    """Every device-utilization value in one `ioreg` dump, in the order the nodes answered."""
    return [int(m) for m in re.findall(rf'"{re.escape(UTILIZATION_KEY)}"=(\d+)', text)]


def summarize(samples: Sequence[int]) -> dict | None:
    """The band a list of readings describes, or None when it describes none."""
    if not samples:
        return None
    ordered = sorted(samples)
    return {
        "n": len(ordered),
        "median": statistics.median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "spread": ordered[-1] - ordered[0],
    }


def verdict(floor: dict | None, run: dict | None) -> str:
    """Classify a run window against the floor window it was measured beside.

    The bar is the floor's own spread, registered before the probe ran: a run is only called
    elevated or quiet when it clears that, because a host whose idle readings swing 12 points
    cannot support a smaller claim. Anything inside it is `indistinguishable`, which is not a
    pass -- it is the statement that this host's background hides the effect being looked for.
    """
    if floor is None or run is None or floor["n"] < MIN_SAMPLES or run["n"] < MIN_SAMPLES:
        return "unreadable"
    delta = run["median"] - floor["median"]
    if abs(delta) > floor["spread"]:
        return "elevated" if delta > 0 else "quiet"
    return "indistinguishable"


def report(floor: dict | None, run: dict | None, absolute_limit: int) -> dict:
    """The two bands, the verdict, and whether the old absolute threshold would have passed."""
    called = verdict(floor, run)
    return {
        "verdict": called,
        "floor_median": floor["median"] if floor else None,
        "floor_spread": floor["spread"] if floor else None,
        "run_median": run["median"] if run else None,
        "delta": (run["median"] - floor["median"]) if floor and run else None,
        "absolute_limit": absolute_limit,
        "meets_absolute_limit": called != "unreadable" and run["median"] <= absolute_limit,
    }


def split_command(text: str) -> list[str]:
    """An operator-typed load command, split into argv. No shell ever sees it."""
    return shlex.split(text)


def read_samples(seconds: float, interval: float) -> list[int]:
    """Sample the accelerator through one window; one value per read, the highest node wins."""
    samples: list[int] = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            out = subprocess.run(
                ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "AGXAccelerator"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            out = ""
        values = parse_utilization(out)
        if values:
            samples.append(max(values))
        time.sleep(max(interval, 0.05))
    return samples


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--floor-seconds", type=float, default=20.0)
    parser.add_argument("--run-seconds", type=float, default=20.0)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--absolute-limit", type=int, default=15)
    parser.add_argument(
        "--cmd",
        default=None,
        help="load to sample beside, as argv; it is launched directly and must exit on its own",
    )
    args = parser.parse_args(argv)

    load = split_command(args.cmd) if args.cmd else []
    if load and shutil.which(load[0]) is None:
        print(f"refusing to sample: {load[0]!r} is not an executable on PATH", file=sys.stderr)
        return 2

    floor = summarize(read_samples(args.floor_seconds, args.interval))
    proc = subprocess.Popen(load) if load else None
    run = summarize(read_samples(args.run_seconds, args.interval))
    status = proc.wait() if proc else 0
    row = report(floor, run, args.absolute_limit)
    row["floor_band"] = [floor["min"], floor["max"]] if floor else None
    row["run_band"] = [run["min"], run["max"]] if run else None
    row["cmd_status"] = status
    print(json.dumps(row, sort_keys=True))
    return 0 if row["verdict"] != "unreadable" else 1


if __name__ == "__main__":
    raise SystemExit(main())
