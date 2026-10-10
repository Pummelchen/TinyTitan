"""AUD-273: the three MTP/ANE A/B drivers print their verdict and return 0.

`benchmark/tinytitan_mtp_phases.py` exists to answer one question -- does the MTP
arm emit the same bytes as the scalar arm -- and it answers it on stdout:

    output identical mtp-on vs mtp-off: NO (off ['aaaa'], on ['bbbb'])

Then `main()` reaches `return 0` (tinytitan_mtp_phases.py:372), so

    python3 benchmark/tinytitan_mtp_phases.py ... && echo "mtp is identical"

prints its sentence over a run that proved the opposite. The sibling
`tinytitan_mtp_b3_qualification.py:160` computes the *same expression*
(`off_digests == on_digests and len(off_digests) == 1`, :121 against :360) and
already returns `0 if identical else 1`, so this is not a new policy: one of the
three drivers carries its identity claim, and none of the three carries the rest.
Measured on the real `main()` with stubbed runs, no model and no server:

  * b3 prints `DELTA: +1.00%   gate +10% at p>=0.65: FAIL` and returns 0 (:160
    reads only `identical`), and prints OUT OF DOMAIN and returns 0;
  * ane-prefill prints `SPEEDUP: 1.25x (gate is >=1.5x: FAIL)`, prints
    `WARNING: an ANE run logged a GPU fallback -- the arms did not measure what
    they claim` and returns 0 (:234-237, :244);
  * all three certify two arms that streamed *no content at all* as identical
    (the sha256 of "" is one digest on both sides) and return 0, which is the
    case `tinytitan_determinism_ab.py:156-165` names in its own docstring and
    refuses: "'identical' has to be earned by content on both sides".

The denominator is the second defect. All three close the sweep with

    def med(sel, k):
        return statistics.median([r[k] for r in sel if k in r])   # :308, :206, :112

so an on-arm run whose server logged no MTP footer -- a run in which MTP did not
engage, which the script's own docstring says is the only variable between arms
-- still enters the rate median while `acceptance` is medianed over the one
survivor, with no count printed anywhere. Measured: `acceptance 90.0%` over 1 of
2 on-runs, exit 0. When *no* run of an arm carries the key, the comprehension is
empty and the driver dies mid-report, after the headline rates printed, with
`StatisticsError: no median for empty data`.

The statuses are the tree's three, as in `tools/model-guard.sh`,
`tools/qwen35_reference.py:411` and the parity harness (AUD-269):

    0  both arms measured, and every claim of the sweep holds
    1  it measured, and a claim failed -- with the claim named
    2  it could not be compared, with the reason named

Every verdict here goes through a `verdict(rows) -> (lines, status)` function,
the shape `tinytitan_determinism_ab.py:137`, `tinytitan_slots_ab.py:133` and
`tinytitan_rdadvise_ab.py:102` already use, so the number a caller reads is the
number the driver computed. The counting itself is one pair of primitives the
three drivers now share -- `arm_metric`/`metric_count` for the denominator and
`arm_answered`/`byte_claim` for the identity claim, in `tinytitan_profile.py`,
which all three already import -- so a fourth A/B driver cannot reintroduce the
private `med()` that hides its own survivor count.

Run from `benchmark/`:

    python3 -m unittest test_mtp_phases_verdict

Thirty-two tests pass. Twenty-seven mutants of the fixed code -- the four
primitives and each driver's gate, content, denominator, digest and
exit-status branches -- were each applied alone and run against this suite, and
all twenty-seven were killed. Three survived the first sweep and are the reason
three of these tests exist: `byte_claim` losing its one-digest-per-arm condition
(no row had tested an arm that disagreed with itself while agreeing with the
other arm), the per-pass attribution table hiding that it medians one of two
on-runs (the fixture that exercised it dropped the MTP footer too, so the
acceptance note already covered the status), and `ane_prefill_ab` dividing by a
prefill time of 0.0 s (the missing-time test used a *missing* key, which the
`None` check still caught). They were proven RED against their mutant and GREEN
against the fixed file, the way `test_reconcile_snapshot_verdict.py` records it.

"""

from __future__ import annotations

import contextlib
import hashlib
import io
import pathlib
import statistics
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))

import tinytitan_ane_prefill_ab as ane  # noqa: E402
import tinytitan_gate0_profile as g0  # noqa: E402
import tinytitan_mtp_b3_qualification as b3  # noqa: E402
import tinytitan_mtp_phases as ph  # noqa: E402
import tinytitan_profile as prof  # noqa: E402

EMPTY_SHA = hashlib.sha256(b"").hexdigest()[:16]
PHASE_KEYS = (
    "proposal",
    "checkpoint",
    "verify",
    "verify_backbone",
    "verify_head",
    "verify_argmax",
    "commit",
    "rollback",
)

TMP = pathlib.Path(tempfile.mkdtemp(prefix="aud273-verdict-"))


def setUpModule() -> None:
    (TMP / "target").mkdir()
    (TMP / "sidecar").mkdir()
    (TMP / "target/verified-install.json").write_text("{}")
    (TMP / "sidecar/manifest.json").write_text("{}")
    g0.preflight = lambda *a, **k: None


def mtp_row(arm, sha, rate, *, tokens=256, footer=True, phases=True):
    """One MTP A/B row in the shape `tinytitan_mtp_phases.one_run` builds.

    `footer=False` is the case the driver hides: the server ran, answered, and
    logged no `mtp` footer, because the draft path never engaged.
    """
    row = {
        "arm": arm,
        "sha256": sha,
        "decode_tok_s": rate,
        "prefill_s": 1.0,
        "completion_tokens": tokens,
    }
    if footer:
        row.update(acceptance=90.0, passes=40, emitted_per_pass=1.8)
        if phases and arm == "on":
            row["phases"] = dict.fromkeys(PHASE_KEYS, 1.0)
    return row


def ane_row(arm, sha, prefill, *, tokens=256, first="7 x 1 = 7", fallback=False):
    return {
        "arm": arm,
        "sha256": sha,
        "prefill_s": prefill,
        "decode_tok_s": 20.0,
        "prompt_tokens": 10141,
        "completion_tokens": tokens,
        "first_line": first,
        "fallback": fallback,
    }


def drive(main, patch_module, rows, argv):
    """Run a driver's real `main()` over stubbed runs and report its status."""
    original = patch_module.one_run
    queue = list(rows)
    saved = []
    patch_module.one_run = lambda *a, **k: queue.pop(0)
    for mod in (ph, b3, ane):
        if hasattr(mod, "ROOT"):
            saved.append((mod, mod.ROOT))
            mod.ROOT = TMP
    argv_saved = sys.argv[1:]
    sys.argv = ["driver"] + argv
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            status = main()
    finally:
        sys.argv = argv_saved
        patch_module.one_run = original
        for mod, root in saved:
            mod.ROOT = root
    return status, buf.getvalue()


MTP_ARGV = [
    "--target",
    str(TMP / "target"),
    "--sidecar",
    str(TMP / "sidecar"),
    "--pairs",
    "1",
]
ANE_ARGV = ["--quant", "4bit", "--pairs", "1"]
B3_ARGV = ["--scenario", "function", "--blocks", "1"]
WARM = [mtp_row("off", "warmup", 10.0), mtp_row("on", "warmup", 10.0)]
AWARM = [ane_row("gpu", "warmup", 10.0), ane_row("ane", "warmup", 10.0)]


def joined(lines):
    return "\n".join(lines)


class ArmMetric(unittest.TestCase):
    """The counting primitive the three drivers share."""

    def test_median_over_the_rows_that_carry_the_key(self):
        rows = [mtp_row("on", "a", 9.0), mtp_row("on", "b", 11.0)]
        value, counted, total = prof.arm_metric(rows, "decode_tok_s")
        self.assertEqual(value, statistics.median([9.0, 11.0]))
        self.assertEqual((counted, total), (2, 2))

    def test_counted_is_returned_so_a_survivor_denominator_is_visible(self):
        rows = [mtp_row("on", "a", 9.0, footer=False), mtp_row("on", "b", 11.0)]
        _value, counted, total = prof.arm_metric(rows, "acceptance")
        self.assertEqual((counted, total), (1, 2))

    def test_no_row_carries_the_key_is_a_count_not_an_exception(self):
        rows = [mtp_row("on", "a", 9.0, footer=False), mtp_row("on", "b", 11.0, footer=False)]
        value, counted, total = prof.arm_metric(rows, "acceptance")
        self.assertIsNone(value)
        self.assertEqual((counted, total), (0, 2))


class ByteClaim(unittest.TestCase):
    """The identity primitive the two MTP drivers share."""

    def test_an_arm_that_is_not_reproducible_does_not_agree_with_itself(self):
        # Two off runs that disagree, and two on runs disagreeing the same way, are
        # the same *set* on both sides. `identical` means one digest, per arm, which
        # is the condition the drivers already had before the fix (`len(== 1`).
        rows = [
            mtp_row("off", "aaaa", 10.0),
            mtp_row("off", "bbbb", 10.1),
            mtp_row("on", "aaaa", 9.0),
            mtp_row("on", "bbbb", 9.1),
        ]
        earned, identical, off_digests, on_digests = prof.byte_claim(rows)
        self.assertTrue(earned)
        self.assertFalse(identical)
        self.assertEqual(off_digests, ["aaaa", "bbbb"])
        self.assertEqual(on_digests, ["aaaa", "bbbb"])


class MtpPhasesVerdict(unittest.TestCase):
    def test_agreeing_arms_that_answered_hold(self):
        rows = [
            mtp_row("off", "aaaa", 10.0),
            mtp_row("on", "aaaa", 9.0),
            mtp_row("on", "aaaa", 9.1),
            mtp_row("off", "aaaa", 10.1),
        ]
        lines, status = ph.verdict(rows)
        self.assertEqual(status, 0)
        self.assertIn("output identical mtp-on vs mtp-off: YES", joined(lines))

    def test_a_divergence_refuses_and_names_both_digest_sets(self):
        rows = [
            mtp_row("off", "aaaa", 10.0),
            mtp_row("on", "bbbb", 9.0),
            mtp_row("on", "bbbb", 9.1),
            mtp_row("off", "aaaa", 10.1),
        ]
        lines, status = ph.verdict(rows)
        self.assertEqual(status, 1)
        text = joined(lines)
        self.assertIn("output identical mtp-on vs mtp-off: NO", text)
        self.assertIn("['aaaa']", text)
        self.assertIn("['bbbb']", text)

    def test_a_run_that_logged_no_mtp_footer_is_counted_not_hidden(self):
        rows = [
            mtp_row("off", "aaaa", 10.0),
            mtp_row("on", "aaaa", 9.0, footer=False),
            mtp_row("on", "aaaa", 9.1),
            mtp_row("off", "aaaa", 10.1),
        ]
        lines, status = ph.verdict(rows)
        self.assertEqual(status, 1)
        text = joined(lines)
        self.assertIn("acceptance", text)
        self.assertIn("1 of 2", text)

    def test_the_per_pass_table_names_its_own_denominator(self):
        # An on-run can carry its `mtp` footer and still log no phase breakdown.
        # The attribution table then medians one run and calls it the sweep's, which
        # is the same hidden denominator one row up -- so the count is printed even
        # when every rate, acceptance and emitted/pass figure reached both runs.
        rows = [
            mtp_row("off", "aaaa", 10.0),
            mtp_row("on", "aaaa", 9.0, phases=False),
            mtp_row("on", "aaaa", 9.1),
            mtp_row("off", "aaaa", 10.1),
        ]
        lines, status = ph.verdict(rows)
        self.assertEqual(status, 1)
        self.assertIn("per-pass table is over 1 of 2", joined(lines))

    def test_a_zero_scalar_median_is_a_refusal_not_a_division(self):
        rows = [
            mtp_row("off", "aaaa", 0.0),
            mtp_row("on", "aaaa", 9.0),
            mtp_row("on", "aaaa", 9.1),
            mtp_row("off", "aaaa", 0.0),
        ]
        lines, status = ph.verdict(rows)
        self.assertEqual(status, 2)
        text = joined(lines)
        self.assertIn("off arm's median is 0 tok/s", text)
        self.assertNotIn("scalar decode", text)

    def test_no_run_with_a_footer_refuses_instead_of_raising(self):
        rows = [
            mtp_row("off", "aaaa", 10.0),
            mtp_row("on", "aaaa", 9.0, footer=False),
            mtp_row("on", "aaaa", 9.1, footer=False),
            mtp_row("off", "aaaa", 10.1),
        ]
        with self.assertRaises(statistics.StatisticsError):
            # the shape the fix replaces, pinned so a mutant cannot reintroduce it
            statistics.median(
                [r["acceptance"] for r in rows if r["arm"] == "on" and "acceptance" in r]
            )
        lines, status = ph.verdict(rows)
        self.assertEqual(status, 2)
        text = joined(lines)
        self.assertIn("NOT MEASURED", text)
        self.assertIn("acceptance", text)

    def test_an_arm_with_no_runs_is_named(self):
        rows = [mtp_row("off", "aaaa", 10.0), mtp_row("off", "aaaa", 10.1)]
        lines, status = ph.verdict(rows)
        self.assertEqual(status, 2)
        self.assertIn("on arm", joined(lines))

    def test_two_arms_that_streamed_no_content_are_not_identity(self):
        rows = [
            mtp_row("off", EMPTY_SHA, 10.0, tokens=0),
            mtp_row("on", EMPTY_SHA, 9.0, tokens=0),
            mtp_row("on", EMPTY_SHA, 9.0, tokens=0),
            mtp_row("off", EMPTY_SHA, 10.0, tokens=0),
        ]
        lines, status = ph.verdict(rows)
        self.assertEqual(status, 2)
        text = joined(lines)
        self.assertIn("no content", text)
        self.assertNotIn("output identical mtp-on vs mtp-off: YES", text)

    def test_the_published_page_is_still_there(self):
        rows = [
            mtp_row("off", "aaaa", 10.0),
            mtp_row("on", "aaaa", 9.0),
            mtp_row("on", "aaaa", 9.1),
            mtp_row("off", "aaaa", 10.1),
        ]
        lines, _status = ph.verdict(rows)
        text = joined(lines)
        self.assertIn("scalar decode", text)
        self.assertIn("per-pass wall attribution", text)
        self.assertIn("break-even", text)


class MtpB3Verdict(unittest.TestCase):
    def test_a_passing_qualification_holds(self):
        rows = [
            mtp_row("off", "aaaa", 10.0),
            mtp_row("on", "aaaa", 12.0),
            mtp_row("on", "aaaa", 12.1),
            mtp_row("off", "aaaa", 10.1),
        ]
        lines, status = b3.verdict(rows)
        self.assertEqual(status, 0)
        self.assertIn("gate +10% at p>=0.65: PASS", joined(lines))

    def test_a_failed_gate_refuses_and_names_the_margin(self):
        rows = [
            mtp_row("off", "aaaa", 10.0),
            mtp_row("on", "aaaa", 10.1),
            mtp_row("on", "aaaa", 10.1),
            mtp_row("off", "aaaa", 10.0),
        ]
        lines, status = b3.verdict(rows)
        self.assertEqual(status, 1)
        text = joined(lines)
        self.assertIn("gate +10% at p>=0.65: FAIL", text)
        self.assertIn("+1.00%", text)

    def test_out_of_domain_is_not_a_pass(self):
        rows = [
            mtp_row("off", "aaaa", 10.0),
            mtp_row("on", "aaaa", 12.0),
            mtp_row("on", "aaaa", 12.0),
            mtp_row("off", "aaaa", 10.0),
        ]
        for row in rows:
            if row["arm"] == "on":
                row["acceptance"] = 40.0
        lines, status = b3.verdict(rows)
        self.assertEqual(status, 2)
        self.assertIn("OUT OF DOMAIN", joined(lines))

    def test_a_divergence_still_refuses(self):
        rows = [
            mtp_row("off", "aaaa", 10.0),
            mtp_row("on", "bbbb", 12.0),
            mtp_row("on", "bbbb", 12.0),
            mtp_row("off", "aaaa", 10.0),
        ]
        lines, status = b3.verdict(rows)
        self.assertEqual(status, 1)
        self.assertIn("output identical: NO", joined(lines))

    def test_two_arms_that_streamed_no_content_are_not_identity(self):
        rows = [
            mtp_row("off", EMPTY_SHA, 10.0, tokens=0),
            mtp_row("on", EMPTY_SHA, 12.0, tokens=0),
            mtp_row("on", EMPTY_SHA, 12.0, tokens=0),
            mtp_row("off", EMPTY_SHA, 10.0, tokens=0),
        ]
        lines, status = b3.verdict(rows)
        self.assertEqual(status, 2)
        text = joined(lines)
        self.assertIn("no content", text)
        self.assertNotIn("output identical: YES", text)

    def test_an_arm_with_no_runs_is_named(self):
        rows = [mtp_row("on", "aaaa", 12.0), mtp_row("on", "aaaa", 12.0)]
        lines, status = b3.verdict(rows)
        self.assertEqual(status, 2)
        self.assertIn("off arm has no runs", joined(lines))


class AnePrefillVerdict(unittest.TestCase):
    def test_a_clean_speedup_holds(self):
        rows = [
            ane_row("gpu", "aaaa", 10.0),
            ane_row("ane", "aaaa", 4.0),
            ane_row("ane", "aaaa", 4.0),
            ane_row("gpu", "aaaa", 10.0),
        ]
        lines, status = ane.verdict(rows)
        self.assertEqual(status, 0)
        self.assertIn("SPEEDUP: 2.50x", joined(lines))

    def test_a_failed_gate_refuses(self):
        rows = [
            ane_row("gpu", "aaaa", 10.0),
            ane_row("ane", "aaaa", 8.0),
            ane_row("ane", "aaaa", 8.0),
            ane_row("gpu", "aaaa", 10.0),
        ]
        lines, status = ane.verdict(rows)
        self.assertEqual(status, 1)
        self.assertIn("gate is >=1.5x: FAIL", joined(lines))

    def test_a_fallback_arm_refuses_because_it_measured_nothing_of_its_own(self):
        rows = [
            ane_row("gpu", "aaaa", 10.0),
            ane_row("ane", "aaaa", 4.0, fallback=True),
            ane_row("ane", "aaaa", 4.0, fallback=True),
            ane_row("gpu", "aaaa", 10.0),
        ]
        lines, status = ane.verdict(rows)
        self.assertEqual(status, 1)
        self.assertIn("did not measure what they claim", joined(lines))

    def test_an_unstable_arm_is_named(self):
        rows = [
            ane_row("gpu", "aaaa", 10.0),
            ane_row("ane", "bbbb", 4.0),
            ane_row("ane", "cccc", 4.0),
            ane_row("gpu", "aaaa", 10.0),
        ]
        lines, status = ane.verdict(rows)
        self.assertEqual(status, 1)
        self.assertIn("ane", joined(lines))

    def test_no_prefill_time_is_a_refusal_not_a_crash(self):
        rows = [
            {"arm": "gpu", "sha256": "aaaa", "decode_tok_s": 20.0, "completion_tokens": 1},
            {"arm": "ane", "sha256": "aaaa", "decode_tok_s": 20.0, "completion_tokens": 1},
            {"arm": "ane", "sha256": "aaaa", "decode_tok_s": 20.0, "completion_tokens": 1},
            {"arm": "gpu", "sha256": "aaaa", "decode_tok_s": 20.0, "completion_tokens": 1},
        ]
        with self.assertRaises(statistics.StatisticsError):
            statistics.median(
                [r["prefill_s"] for r in rows if r["arm"] == "gpu" and "prefill_s" in r]
            )
        lines, status = ane.verdict(rows)
        self.assertEqual(status, 2)
        self.assertIn("prefill_s", joined(lines))

    def test_a_zero_prefill_time_is_a_refusal_not_a_division(self):
        # A logged prefill of 0.0 s is not a fast ANE: it is a run whose timer
        # never started. Dividing the GPU's 10 s by it would print a 100.00x
        # speedup, and the gate would PASS, over a sweep that measured no ANE time.
        rows = [
            ane_row("gpu", "aaaa", 10.0),
            ane_row("ane", "aaaa", 0.0),
            ane_row("ane", "aaaa", 0.0),
            ane_row("gpu", "aaaa", 10.0),
        ]
        lines, status = ane.verdict(rows)
        self.assertEqual(status, 2)
        text = joined(lines)
        self.assertIn("prefill time is missing", text)
        self.assertNotIn("SPEEDUP", text)

    def test_a_missing_arm_is_named(self):
        rows = [ane_row("gpu", "aaaa", 10.0), ane_row("gpu", "aaaa", 10.0)]
        lines, status = ane.verdict(rows)
        self.assertEqual(status, 2)
        self.assertIn("ane arm has no runs", joined(lines))

    def test_two_arms_that_streamed_no_content_are_not_stability(self):
        rows = [
            ane_row("gpu", EMPTY_SHA, 10.0, tokens=0, first="(empty)"),
            ane_row("ane", EMPTY_SHA, 4.0, tokens=0, first="(empty)"),
            ane_row("ane", EMPTY_SHA, 4.0, tokens=0, first="(empty)"),
            ane_row("gpu", EMPTY_SHA, 10.0, tokens=0, first="(empty)"),
        ]
        lines, status = ane.verdict(rows)
        self.assertEqual(status, 2)
        self.assertIn("no content", joined(lines))


class StatusReachesTheCaller(unittest.TestCase):
    """The wiring: what `main()` returns is what the verdict computed."""

    def test_mtp_phases_main_refuses_a_divergence(self):
        rows = WARM + [
            mtp_row("off", "aaaa", 10.0),
            mtp_row("on", "bbbb", 9.0),
            mtp_row("on", "bbbb", 9.1),
            mtp_row("off", "aaaa", 10.1),
        ]
        status, out = drive(ph.main, ph, rows, MTP_ARGV)
        self.assertIn("output identical mtp-on vs mtp-off: NO", out)
        self.assertEqual(status, 1)

    def test_mtp_phases_main_still_returns_zero_for_a_sweep_that_holds(self):
        rows = WARM + [
            mtp_row("off", "aaaa", 10.0),
            mtp_row("on", "aaaa", 9.0),
            mtp_row("on", "aaaa", 9.1),
            mtp_row("off", "aaaa", 10.1),
        ]
        status, out = drive(ph.main, ph, rows, MTP_ARGV)
        self.assertIn("output identical mtp-on vs mtp-off: YES", out)
        self.assertEqual(status, 0)

    def test_b3_main_refuses_a_failed_gate(self):
        rows = WARM + [
            mtp_row("off", "aaaa", 10.0),
            mtp_row("on", "aaaa", 10.1),
            mtp_row("on", "aaaa", 10.1),
            mtp_row("off", "aaaa", 10.0),
        ]
        status, out = drive(b3.main, ph, rows, B3_ARGV)
        self.assertIn("gate +10% at p>=0.65: FAIL", out)
        self.assertEqual(status, 1)

    def test_b3_main_still_refuses_a_divergence(self):
        rows = WARM + [
            mtp_row("off", "aaaa", 10.0),
            mtp_row("on", "bbbb", 12.0),
            mtp_row("on", "bbbb", 12.0),
            mtp_row("off", "aaaa", 10.0),
        ]
        status, _out = drive(b3.main, ph, rows, B3_ARGV)
        self.assertEqual(status, 1)

    def test_ane_main_refuses_the_arm_that_fell_back(self):
        rows = AWARM + [
            ane_row("gpu", "aaaa", 10.0),
            ane_row("ane", "aaaa", 4.0, fallback=True),
            ane_row("ane", "aaaa", 4.0, fallback=True),
            ane_row("gpu", "aaaa", 10.0),
        ]
        status, out = drive(ane.main, ane, rows, ANE_ARGV)
        self.assertIn("did not measure what they claim", out)
        self.assertEqual(status, 1)


if __name__ == "__main__":
    unittest.main()
