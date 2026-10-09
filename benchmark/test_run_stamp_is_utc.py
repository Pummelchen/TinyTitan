"""Gates the shape of every results-file stamp in `benchmark/` and `tools/`.

Two drivers built their output name from local wall-clock time while eight built it
from UTC, and this Mac's zone is Europe/Berlin, whose clock repeats the hour
02:00-02:59:59 on the last Sunday of October. Measured in the probe
(`.build/aud248/probe.py`, run as `TZ=Europe/Berlin python3 .build/aud248/probe.py`):
the instants 2026-10-25T00:15:00Z and 01:15:00Z -- one hour apart, both real -- both
formatted to `20261025T021500`, so `tinytitan_benchmark.py:606`'s
`os.makedirs(outdir, exist_ok=True)` reused the first run's directory and the
`aggregate.json` written at :608 held only the second run. The probe prints
`records lost: True`. `tinytitan_longctx.py:444` has the same shape with a file rather
than a directory: `open(summary_path, "w")` truncates the earlier `longctx-*.json`.

A name that identifies two runs is not a cosmetic offset. It is the run's identity, and
the tree already settled the question once -- every other stamping site reads UTC.

The suite is model-free: it calls the naming helper with fixed instants and reads
source text, so it starts no server and loads nothing.
"""

from __future__ import annotations

import datetime
import os
import pathlib
import re
import time
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
BENCHMARK = REPO / "benchmark"
TOOLS = REPO / "tools"
STAMP_FORMAT = "%Y%m%dT%H%M%S"

# 2026-10-25 is the European fall-back: 00:15Z and 01:15Z are both 02:15 locally.
FIRST = datetime.datetime(2026, 10, 25, 0, 15, tzinfo=datetime.timezone.utc)
SECOND = FIRST + datetime.timedelta(hours=1)


class RepeatsTheHour(unittest.TestCase):
    """Base for the suites that must run as if the machine were set to Europe/Berlin.

    It carries no tests itself, so it collects nothing on its own.
    """

    def setUp(self):
        self.saved_tz = os.environ.get("TZ")
        os.environ["TZ"] = "Europe/Berlin"
        time.tzset()
        self.addCleanup(self.restore)

    def restore(self):
        if self.saved_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = self.saved_tz
        time.tzset()


class TestTheSharedStamp(RepeatsTheHour):
    def setUp(self):
        super().setUp()
        import tinytitan_profile

        self.stamp = tinytitan_profile.run_stamp

    def utc(self, moment):
        return moment.strftime(STAMP_FORMAT)

    def test_two_runs_in_the_repeated_hour_get_different_stamps(self):
        first, second = self.stamp(FIRST.timestamp()), self.stamp(SECOND.timestamp())
        self.assertNotEqual(
            first,
            second,
            "both instants are 02:15 on the same local date, so a local stamp is not "
            "an identity -- and nothing told the second run it had taken the first's name",
        )

    def test_the_stamp_is_the_utc_time_of_the_instant(self):
        self.assertEqual(self.stamp(FIRST.timestamp()), self.utc(FIRST))
        self.assertEqual(self.stamp(SECOND.timestamp()), self.utc(SECOND))

    def test_the_stamp_keeps_the_documented_shape(self):
        self.assertRegex(self.stamp(FIRST.timestamp()), r"^\d{8}T\d{6}$")

    def test_the_stamp_follows_the_instant_not_the_machine_zone(self):
        """A stamp that read the zone would change with TZ; a UTC stamp must not.

        Both calls format the same instant, and the only thing between them is the
        machine's own offset, which Berlin applies to this hour as +02:00 and to the
        next as +01:00.
        """
        self.assertEqual(self.stamp(FIRST.timestamp()), self.stamp(SECOND.timestamp() - 3600))
        shifted = self.stamp(FIRST.timestamp() + 86400)
        self.assertNotEqual(shifted, self.stamp(FIRST.timestamp()))

    def test_no_argument_stamps_the_present(self):
        utc_now = datetime.datetime.now(datetime.timezone.utc)
        rendered = self.stamp()
        self.assertRegex(rendered, r"^\d{8}T\d{6}$")
        self.assertIn(rendered, {self.stamp(utc_now.timestamp()), self.stamp()})
        self.assertNotEqual(
            rendered,
            time.strftime(STAMP_FORMAT),
            "the default path must not be the local clock it is replacing",
        )


class TestTheDriversUseTheSharedStamp(RepeatsTheHour):
    """The two sites that read local time, pinned by what they now call."""

    def production(self, name):
        return (BENCHMARK / name).read_text(encoding="utf-8")

    def test_the_two_result_names_come_from_the_shared_stamp(self):
        for name in ("tinytitan_benchmark.py", "tinytitan_longctx.py"):
            with self.subTest(module=name):
                source = self.production(name)
                self.assertNotIn(
                    "time.strftime",
                    source,
                    f"{name} formats a results name from local wall-clock time, which "
                    "repeats an hour on this zone's fall-back and collides two runs",
                )
                self.assertIn("run_stamp(", source, f"{name} names its results by hand")

    def test_the_tree_has_no_local_time_stamp_left(self):
        """The guard: a name-shaped stamp must be UTC-anchored, in either language.

        `date +%s` is an epoch and needs no `-u`; a bare `$(date)` is a log line, not a
        name. Only a format that carries `%Y` names a file, so only those are checked.
        """
        offenders = []
        for directory, pattern, probe in (
            (BENCHMARK, "*.py", "strftime"),
            (TOOLS, "*.py", "strftime"),
            (BENCHMARK, "*.sh", "date +"),
            (TOOLS, "*.sh", "date +"),
        ):
            for path in sorted(directory.glob(pattern)):
                if path.name.startswith("test_"):
                    continue
                lines = path.read_text(encoding="utf-8").splitlines()
                for number, line in enumerate(lines, start=1):
                    if probe not in line:
                        continue
                    # A call may wrap, and the timezone sits on either side of it.
                    window = " ".join(lines[max(number - 3, 0) : number + 1])
                    if "%Y" not in window:
                        continue
                    anchored = "timezone.utc" in window or (
                        probe == "date +" and re.search(r"\bdate\s+-u\s+\+", line)
                    )
                    if not anchored:
                        offenders.append(f"{path.name}:{number}: {line.strip()}")
        self.assertEqual(
            offenders,
            [],
            "a results or backup name built from local time silently reuses one name "
            "for two runs when the zone repeats an hour",
        )


if __name__ == "__main__":
    unittest.main()
