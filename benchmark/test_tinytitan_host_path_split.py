"""§5bg's host-path split: does the GPU's family-2 idle sit behind the host?

§5w left 42.835 ms/token of `shared_expert -> moe_phase1_*` GPU idle as "the open
seam ... where the next session should start", and priced it against four named
host counters. This screen prices it against the whole host path, using the
nesting the layer loop actually has: `body_ms` (the entire layer iteration,
`RealForwardRunner+DecodeLayers.swift:46` -> `+DecodeMoE.swift:472`) contains
`wait_ms` (blocked on the router readback) and `io_ms` (blocked on the miss
pread), so what is left -- `body - wait - io` -- is the host time inside the
layer loop that is neither waiting nor reading.

But the footer's counters do not all accumulate inside that window. `head_ms`,
`embed_ms`, `gather_ms`, `pre_ms` and the three `loop_*` timers are accumulated
once per token, outside the 48 per-layer bodies (`RealForwardRunner+Decode.swift:156,197,203`,
`RawCompletion.swift:305,340,362`), so they are priced against the other
envelope, `wall - body`. Each named set is reported against its own envelope,
and both are summed once, as `host off waits`, to bound the seam absolutely.

It ships as a screen rather than as a paragraph because the identities are the
thing worth keeping: `body = wait + io + bookkeeping` and `wall = body + off the
loop`, each ~89% accounted for by the counters that belong to it. A future
counter that nests differently, a layer loop longer than the token that contains
it, or a footer that gains a term this screen does not assign must break an
identity and be reported, not quietly summed into a split.

Fixtures are the verbatim diagnostic lines one real 512-token capture arm
printed. No model, no server, nothing fetched. Rows are read by label rather
than by column width, so re-formatting the table cannot silently pass.
"""

import io
import re
import unittest
from contextlib import redirect_stdout

import tinytitan_host_path_split as hp

# The capture arm of §5bg, copied verbatim from the run's stderr. Every number
# below is what the shipped build printed; the test pins the screen to it.
RUNNER = (
    "TinyTitan runner cb1_ms=4.976 io_ms=73.676 cb2_ms=1.124 head_ms=11.605 "
    "head_fused_ms=0.000 rdadvise_ms=7.876 wait_ms=105.340 body_ms=199.960 "
    "rdadvise_calls=35525 rdadvise_mib=94320.5 expert_hit_rate=0.8468 "
    "expert_hits=209561 expert_misses=37911 expert_read_mib=100108.7 "
    "io_hidden_pct=99.67 hit_fixup_layers=15613 router_readback_ms=0.0206 "
    "cache_plan_ms=4.6565 io_queue_ms=0.3607 io_completion_to_fixup_ms=1.6075 "
    "io_host_waits=15613 io_host_waits_avoided=8915 embed_ms=0.664 "
    "gather_ms=2.446 loop_sample_ms=0.747 loop_progress_ms=0.094 "
    "loop_other_ms=1.204 pre_ms=0.239 pre_release_ms=0.002 pre_pin_ms=0.237 "
    "pre_reserve_ms=0.002"
)
GAPS = [
    "TinyTitan gap moe_phase1_hit->moe_phase1_miss_fixup total_ms=28388.8 "
    "per_token_ms=55.447 count=15598",
    "TinyTitan gap shared_expert->moe_phase1_hit total_ms=13824.5 per_token_ms=27.001 count=15552",
    "TinyTitan gap shared_expert->moe_phase1_2_routed total_ms=8194.4 "
    "per_token_ms=16.005 count=8915",
    "TinyTitan gap embed->attn_norm_qkv total_ms=1633.6 per_token_ms=3.191 count=511",
]
WALL = "[stop=maxTokens prefill=10tok/4.36s new=512tok decode=112.16s tok/s=4.565]"
LOG = "\n".join([RUNNER] + GAPS + [WALL])


def run(text: str) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = hp.main(["--log", "-"], stdin=text)
    return code, buf.getvalue()


def cell(out: str, label: str) -> float:
    """The `ms/token` value printed on the row whose label is `label`."""
    for line in out.splitlines():
        head, sep, rest = line.partition(":")
        if sep and head.strip() == label:
            m = re.search(r"([\d.]+) ms/token", rest)
            if m:
                return float(m.group(1))
            raise AssertionError(f"row {label!r} carries no ms/token value: {line!r}")
    raise AssertionError(f"no row labelled {label!r} in:\n{out}")


class ReadsTheNesting(unittest.TestCase):
    def test_the_host_path_splits_into_waiting_reading_and_bookkeeping(self):
        code, out = run(LOG)
        self.assertEqual(code, 0, out)
        # body = wait + io + bookkeeping, and the last is the host time inside
        # the layer loop that is neither waiting nor reading:
        # 199.960 - 105.340 - 73.676 = 20.944 ms/token.
        self.assertEqual(cell(out, "body"), 199.960)
        self.assertEqual(cell(out, "wait"), 105.340)
        self.assertEqual(cell(out, "io"), 73.676)
        self.assertEqual(cell(out, "bookkeeping"), 20.944)

    def test_counters_timed_outside_the_layer_loop_are_not_bookkeeping(self):
        _, out = run(LOG)
        # cb1 4.976 + cb2 1.124 + rdadvise 7.876 + cache_plan 4.6565 + readback
        # 0.0206 = 18.653 ms/token accumulate inside the 48 body windows. The
        # other six timers (embed 0.664, gather 2.446, pre 0.239, loop_* 0.747 +
        # 0.094 + 1.204, head 11.605) are per token and outside them, so they
        # belong to wall - body = 19.102, not to the 20.944. An earlier draft
        # summed both sets into one 24.047 "named terms" row against the
        # in-body envelope; that number must not come back.
        self.assertEqual(cell(out, "named in body"), 18.653)
        self.assertEqual(cell(out, "off the loop"), 19.102)
        self.assertEqual(cell(out, "named off loop"), 16.999)
        self.assertNotIn("24.047", out)

    def test_each_named_set_is_reported_against_its_own_envelope(self):
        _, out = run(LOG)
        # 18.653 / 20.944 = 89.1% and 16.999 / 19.102 = 89.0%: neither set
        # over-counts its envelope, and both leave ~11% of it unnamed. Those are
        # two independent ~89% figures, which is the screen's evidence that the
        # scopes it guessed really are the ones the counters accumulate in.
        self.assertIn("89.1%", out)
        self.assertIn("89.0%", out)

    def test_family_two_is_the_gap_lines_that_start_at_the_shared_expert(self):
        _, out = run(LOG)
        # 27.001 + 16.005: the two lines §5w charged to host turnaround. The
        # 55.447 ms/token miss-fixup gap is family 1 and is not part of it.
        self.assertEqual(cell(out, "family 2"), 43.006)
        rows = [line for line in out.splitlines() if line.startswith("family 2")]
        self.assertEqual(len(rows), 1, out)
        self.assertNotIn("moe_phase1_hit->moe_phase1_miss_fixup", rows[0])


class AnswersThePremise(unittest.TestCase):
    def test_a_host_path_cannot_explain_family_two_when_bookkeeping_is_smaller(self):
        code, out = run(LOG)
        self.assertEqual(code, 0, out)
        # The seam §5w left open is bounded by in-body bookkeeping, not by
        # family 2: 20.944 of 43.006 ms/token, 9.6% of the token.
        self.assertIn("at most 20.944 ms/token of its 43.006", out)
        self.assertIn("9.6% of the token", out)

    def test_the_absolute_bound_is_every_millisecond_that_is_not_a_blocked_wait(self):
        _, out = run(LOG)
        # Even the loosest host-side bound -- every per-token millisecond that
        # is not the router wait or the miss pread, inside the loop or outside
        # it -- is 20.944 + 19.102 = 40.046 ms/token, still 2.960 short of
        # family 2. Nothing left of the host can reach the idle.
        self.assertEqual(cell(out, "host off waits"), 40.046)
        self.assertIn("by 2.960", out)

    def test_it_names_the_arm_worth_pricing_when_only_the_absolute_bound_reaches(self):
        # family 2 at 27.000 sits above in-body bookkeeping (20.944) and below
        # the absolute bound (40.046), so the host can still reach it -- but the
        # work that would pay is off the layer loop, not inside it. Reporting
        # this as "no host-path lever" because bookkeeping alone is short would
        # close the seam on the wrong evidence.
        text = LOG.replace("per_token_ms=27.001", "per_token_ms=22.000").replace(
            "per_token_ms=16.005", "per_token_ms=5.000"
        )
        code, out = run(text)
        self.assertEqual(code, 1, out)
        self.assertIn("but not inside the loop", out)

    def test_it_names_the_seam_real_when_family_two_drops_below_bookkeeping(self):
        # The falsifier, in the other direction: a page whose GPU idle behind
        # the shared expert is smaller than the host work that is not a blocked
        # wait does have a host-side lever, and the screen must say so with
        # exit 1 rather than close the seam it was built to close. (The first
        # draft of this test reached the branch only by inflating
        # rdadvise_ms/body_ms, which made body longer than the token and so is
        # now a refusal, not a verdict.)
        text = LOG.replace("per_token_ms=27.001", "per_token_ms=5.000").replace(
            "per_token_ms=16.005", "per_token_ms=3.000"
        )
        code, out = run(text)
        self.assertEqual(code, 1, out)
        self.assertIn("host-path arm is worth pricing", out)


class RefusesAPageItCannotRead(unittest.TestCase):
    def test_a_log_with_no_runner_line_is_not_measured_as_zero_bookkeeping(self):
        code, out = run("\n".join(GAPS + [WALL]))
        self.assertEqual(code, 2, out)
        self.assertIn("no `TinyTitan runner` line", out)
        self.assertNotIn("0.000 ms/token", out)

    def test_a_log_with_no_shared_expert_gap_cannot_price_family_two(self):
        text = "\n".join([RUNNER, GAPS[0], GAPS[3], WALL])
        code, out = run(text)
        self.assertEqual(code, 2, out)
        self.assertIn("no `shared_expert->` gap line", out)

    def test_a_runner_line_missing_an_envelope_term_is_not_split(self):
        # The screen's whole argument is the identity, so a footer that stopped
        # printing one of its three terms cannot be split at all.
        text = LOG.replace("body_ms=199.960 ", "")
        code, out = run(text)
        self.assertEqual(code, 2, out)
        self.assertIn("body_ms", out)

    def test_a_page_with_no_decode_footer_cannot_be_bounded(self):
        # `off the loop` is wall - body, so a page whose footer was not captured
        # has no second envelope. The screen must refuse rather than quietly
        # price family 2 against the in-body half and report the narrower bound.
        text = "\n".join([RUNNER] + GAPS)
        code, out = run(text)
        self.assertEqual(code, 2, out)
        self.assertIn("footer", out)

    def test_a_bookkeeping_envelope_that_comes_out_negative_is_a_refusal(self):
        # body < wait + io breaks the first identity, so every number
        # downstream of it is unreadable.
        text = LOG.replace("body_ms=199.960", "body_ms=100.000")
        code, out = run(text)
        self.assertEqual(code, 2, out)
        self.assertIn("nesting", out)

    def test_a_layer_loop_longer_than_the_token_it_belongs_to_is_a_refusal(self):
        # 48 bodies cannot span more than the token that contains them: at
        # body_ms=250 against a 219.062 ms/token wall, the second identity fails
        # and `wall - body` would be negative.
        text = LOG.replace("body_ms=199.960", "body_ms=250.000")
        code, out = run(text)
        self.assertEqual(code, 2, out)
        self.assertIn("wall", out)

    def test_a_counter_the_screen_has_not_assigned_is_reported_not_dropped(self):
        # AUD-276's shape: a footer that gains a term would otherwise make the
        # split silently wrong.
        text = LOG.replace("cb1_ms=4.976", "cb1_ms=4.976 brand_new_ms=9.999")
        code, out = run(text)
        self.assertEqual(code, 0, out)
        self.assertIn("UNASSIGNED", out)
        self.assertIn("brand_new_ms", out)


if __name__ == "__main__":
    unittest.main()
