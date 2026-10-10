"""The release notes budget is a size, and `0` is not a size — it is a switch.

`tools/release.sh` reads the operator's `TINYTITAN_RELEASE_NOTES_MAX_CHARS` and
hands it to `tools/compact-release-notes.py --max-chars`. That tool documents its
own zero as "no limit" (`:115-120`, and the check at `:315` is
`if args.max_chars and ...`), which is right for a CLI and wrong one level up:
the release path has no shape check on the value, so `0` silently removes the gate
the script's own comment calls "the part that actually keeps notes short" (`:605`),
and the budget is never printed, so nothing says it was off. Measured on the real
compactor over a 3,003-character note:

    --max-chars 0     ->  exit 0, published        (the budget is gone)
    --max-chars 100   ->  exit 1, over budget      (works as intended)
    --max-chars -1    ->  exit 1, "over the -1 budget"
    --max-chars abc   ->  exit 2, argparse usage error

`-1` and `abc` are the second defect, which is a message rather than a status: the
release script wraps any non-zero exit in `die "the notes did not survive
compaction, or are over the ${NOTES_MAX_CHARS}-character budget"` (`:624`), so a
value that was never a budget gets reported as a note that is too long, and an
argument-parsing failure gets reported as content. Both arrive after the archives
are built and hashed.

The other deliberate narrowings on this path are held to a stricter bar than the
budget is: `TINYTITAN_RELEASE_ALLOW_RED_CI` and `TINYTITAN_RELEASE_SKIP_GOLDENS`
each *die without a companion reason* (`:165`, `:194`) and each name themselves in
the published notes. So the repair is the same principle, applied as the sibling
port guard was: check the value where it is read, refuse the shapes that are not
budgets, and leave the tool's own documented zero alone.

    cd benchmark && python3 -m unittest test_release_notes_budget_gate -v
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile
import unittest

from test_release_ci_green import run_notes_block

ROOT = pathlib.Path(__file__).resolve().parents[1]
COMPACTOR = ROOT / "tools/compact-release-notes.py"
GREEN = "result\tgreen\t-"
VARIABLE = "TINYTITAN_RELEASE_NOTES_MAX_CHARS"


def budget_arg(argv: list[str]) -> str | None:
    """The value the compaction step was handed for --max-chars, if it was reached."""
    for index, token in enumerate(argv):
        if token == "--max-chars":
            return argv[index + 1] if index + 1 < len(argv) else None
    return None


class TheBudgetIsCheckedWhereItIsRead(unittest.TestCase):
    """The release path refuses a value that is not a size before it uses it."""

    def test_a_zero_budget_is_refused_before_the_compactor_runs(self) -> None:
        result = run_notes_block(GREEN, {VARIABLE: "0"})
        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0, "0 published the notes with no budget")
        self.assertIn("0", output)
        self.assertEqual(budget_arg(result.argv), None, f"the compactor was reached: {result.argv}")

    def test_a_zero_is_not_reported_as_a_note_that_is_too_long(self) -> None:
        result = run_notes_block(GREEN, {VARIABLE: "0"})
        self.assertNotIn("did not survive compaction", result.stdout + result.stderr)

    def test_a_negative_budget_is_refused_at_the_read(self) -> None:
        result = run_notes_block(GREEN, {VARIABLE: "-1"})
        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0, "a negative budget reached the compactor")
        self.assertIn("-1", output)
        self.assertEqual(budget_arg(result.argv), None, f"the compactor was reached: {result.argv}")

    def test_a_non_numeric_budget_is_refused_at_the_read(self) -> None:
        result = run_notes_block(GREEN, {VARIABLE: "abc"})
        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0, "a word reached the compactor")
        self.assertIn("abc", output)
        self.assertNotIn("did not survive compaction", output)

    def test_the_default_budget_still_reaches_the_compactor(self) -> None:
        result = run_notes_block(GREEN)
        self.assertEqual(result.returncode, 0, f"{result.stdout}\n{result.stderr}")
        self.assertEqual(budget_arg(result.argv), "12000")

    def test_a_raised_budget_reaches_the_compactor_as_written(self) -> None:
        result = run_notes_block(GREEN, {VARIABLE: "50000"})
        self.assertEqual(result.returncode, 0, f"{result.stdout}\n{result.stderr}")
        self.assertEqual(budget_arg(result.argv), "50000")

    def test_an_unset_value_is_the_documented_default_not_an_empty_argument(self) -> None:
        # `:-12000` is what makes a blank assignment mean "use the default"; a
        # guard that reads the raw variable would put `--max-chars ""` on the
        # compactor's command line instead.
        result = run_notes_block(GREEN, {VARIABLE: ""})
        self.assertEqual(result.returncode, 0, f"{result.stdout}\n{result.stderr}")
        self.assertEqual(budget_arg(result.argv), "12000")


class TheToolsOwnDocumentedSemanticsStay(unittest.TestCase):
    """The compactor keeps its own no-limit, because its help promises it."""

    def compact(self, max_chars: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as work:
            notes = pathlib.Path(work) / "notes.md"
            notes.write_text("- " + ("x" * 3000) + "\n", encoding="utf-8")
            out = pathlib.Path(work) / "out.md"
            return subprocess.run(
                [
                    sys.executable,
                    str(COMPACTOR),
                    str(notes),
                    "--out",
                    str(out),
                    "--max-chars",
                    max_chars,
                ],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )

    def test_the_compactors_own_zero_still_means_no_limit(self) -> None:
        result = self.compact("0")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_a_real_over_budget_note_names_both_numbers(self) -> None:
        result = self.compact("100")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("3003", result.stderr)
        self.assertIn("100", result.stderr)

    def test_a_value_that_is_not_a_number_fails_the_tools_own_parse(self) -> None:
        # Why the release path must not pass this through: exit 2 from argparse
        # is not a statement about the notes, and the caller cannot tell.
        result = self.compact("abc")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("usage:", result.stderr)


if __name__ == "__main__":
    unittest.main()
