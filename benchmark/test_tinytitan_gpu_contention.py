"""Tests for the host-relative GPU contention gauge (§5bd).

Every fixture here is a shape the real `ioreg` reader produced on the 2026-10-16 probe: three
nodes answering one read, a 0 sample landing immediately after a load stops, and an idle floor of
56-68 % that no absolute threshold can be satisfied under. The suite is model-free and never calls
`ioreg` -- it pins the parsing and the verdict, which is where the shipped single-sample `max` rule
goes wrong.
"""

import contextlib
import io
import statistics
import unittest

import tinytitan_gpu_contention as gauge


class ParseTests(unittest.TestCase):
    def test_parse_reads_every_node_in_one_dump(self) -> None:
        # One `ioreg` read answers from more than one AGXAccelerator node; a reader that keeps
        # only the first would report a different machine than the one being benchmarked.
        text = (
            '"Device Utilization %"=62\n'
            "other stuff\n"
            '"Device Utilization %"=58\n'
            '"Tiler Utilization %"=61\n'
        )
        self.assertEqual(gauge.parse_utilization(text), [62, 58])

    def test_parse_returns_empty_on_a_dump_that_names_no_utilization(self) -> None:
        self.assertEqual(gauge.parse_utilization("no keys here"), [])


class SummarizeTests(unittest.TestCase):
    def test_median_survives_the_zero_sample_that_follows_a_load(self) -> None:
        # Measured: the first read after a saturating window returns 0. A mean would drop the
        # floor by ~10 points and hand a contended host a clean label.
        summary = gauge.summarize([0, 62, 65, 64, 61])
        self.assertEqual(summary["median"], 62)
        self.assertEqual(summary["n"], 5)
        self.assertEqual(summary["spread"], 65)  # max - min, and the 0 is inside that range

    def test_summarize_refuses_an_empty_sample_list(self) -> None:
        self.assertIsNone(gauge.summarize([]))


class VerdictTests(unittest.TestCase):
    def _floor(self, samples: list[int]) -> dict:
        return gauge.summarize(samples)

    def test_a_run_inside_the_floor_band_is_indistinguishable_not_clean(self) -> None:
        # The campaign's actual situation: decode measured 69.6 % against a floor of 56-68 %,
        # i.e. inside the floor's own spread. Reporting that as "quiet" is the defect.
        floor = self._floor([56, 61, 64, 66, 68])
        run = gauge.summarize([67, 69, 70, 68, 66])
        self.assertEqual(gauge.verdict(floor, run), "indistinguishable")

    def test_a_run_more_than_the_floor_spread_above_the_floor_is_elevated(self) -> None:
        floor = self._floor([56, 61, 64, 66, 68])
        run = gauge.summarize([88, 90, 92, 91, 89])
        self.assertEqual(gauge.verdict(floor, run), "elevated")

    def test_a_run_below_the_floor_is_quiet(self) -> None:
        floor = self._floor([56, 61, 64, 66, 68])
        run = gauge.summarize([40, 41, 44, 42, 43])
        self.assertEqual(gauge.verdict(floor, run), "quiet")

    def test_a_side_with_too_few_samples_refuses_instead_of_passing(self) -> None:
        # A gate that read nothing must not report a pass -- the rule every AUD row here applies
        # to a scan over an empty walk.
        floor = self._floor([56, 61, 64, 66, 68])
        self.assertEqual(gauge.verdict(floor, gauge.summarize([70, 71])), "unreadable")
        self.assertEqual(gauge.verdict(None, gauge.summarize([70, 71, 72])), "unreadable")

    def test_verdict_reads_the_registered_rule_from_the_boundaries(self) -> None:
        # delta exactly equal to the spread is still inside it; one above is outside.
        floor = self._floor([60, 60, 64, 66, 66])  # median 64, spread 6
        self.assertEqual(
            gauge.verdict(floor, gauge.summarize([67, 68, 70, 69, 71])), "indistinguishable"
        )
        self.assertEqual(gauge.verdict(floor, gauge.summarize([71, 72, 70, 73, 74])), "elevated")


class CommandTests(unittest.TestCase):
    def test_a_command_is_split_into_argv(self) -> None:
        self.assertEqual(
            gauge.split_command("python3 probe.py --flag 'a b'"),
            ["python3", "probe.py", "--flag", "a b"],
        )

    def test_a_shell_operator_stays_a_literal_argument(self) -> None:
        # Nothing is handed to a shell, so `&&` is an argument to the named executable rather
        # than a second command. The gauge may report a wrong load; it may not run one.
        self.assertEqual(gauge.split_command("true && false"), ["true", "&&", "false"])

    def test_an_empty_command_splits_to_nothing(self) -> None:
        self.assertEqual(gauge.split_command("   "), [])

    def test_an_unlaunchable_load_is_refused_not_traced(self) -> None:
        # The load command is operator input, so a typo is a boundary: a status and one stderr line
        # rather than a traceback that buries the report.
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            status = gauge.main(
                [
                    "--cmd",
                    "/nonexistent/tinytitan-gauge-load",
                    "--floor-seconds",
                    "0",
                    "--run-seconds",
                    "0",
                ]
            )
        self.assertEqual(status, 2)
        self.assertIn("/nonexistent/tinytitan-gauge-load", err.getvalue())


class ReportTests(unittest.TestCase):
    def test_report_names_the_threshold_the_host_cannot_meet(self) -> None:
        floor = gauge.summarize([56, 61, 64, 66, 68])
        run = gauge.summarize([67, 69, 70, 68, 66])
        row = gauge.report(floor, run, absolute_limit=15)
        self.assertEqual(row["verdict"], "indistinguishable")
        self.assertFalse(row["meets_absolute_limit"])
        self.assertEqual(statistics.median([56, 61, 64, 66, 68]), row["floor_median"])

    def test_report_refuses_to_call_an_unreadable_floor_a_pass(self) -> None:
        row = gauge.report(None, gauge.summarize([1, 2, 3]), absolute_limit=15)
        self.assertEqual(row["verdict"], "unreadable")
        self.assertFalse(row["meets_absolute_limit"])

    def test_the_absolute_limit_field_can_read_true(self) -> None:
        # Characterization test, not a RED: it pins that the field is not a constant False, which
        # is the only way the refusal it describes stays checkable. Killing it needs the production
        # change `meets_absolute_limit = False`.
        row = gauge.report(
            gauge.summarize([56, 61, 64, 66, 68]), gauge.summarize([3, 4, 5, 4, 3]), 15
        )
        self.assertEqual(row["verdict"], "quiet")
        self.assertTrue(row["meets_absolute_limit"])


if __name__ == "__main__":
    unittest.main()
