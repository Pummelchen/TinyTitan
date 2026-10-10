"""Gates what `tools/release.sh --publish` puts on the Release page.

`tools/compact-release-notes.py` is the only author of the published notes: the
full record stays in `docs/release-notes-vX.Y.md` and the compact form is what
GitHub shows. Nothing tested it -- no test file in this repository names the
script at all -- and its docstring promise "nothing is reworded, only
re-laid-out" is false for every block whose markdown syntax IS its content. All
three shapes below are measured on the real notes, and the corpus test runs the
script over every `docs/release-notes-v*.md`:

1. **a fenced code block collapses into one bullet.** The v5.18 notes carry a
   two-line example of the progress line format inside an indented fence and the
   compact form is one line, `- ``` converting 38.7/91.6 GB 42% 4/13 shards eta
   12m installing 12.3/38.1 GB 32% eta 4m10s ``` ` -- an example whose whole
   content is the line breaks.
2. **a markdown table collapses into one bullet** (`- | Model | 4-bit | 8-bit |
   | --- | ---: | ---: | | Qwen 3.6 35B-A3B | +1.8% ...` in release-notes-v5.1),
   which GitHub then renders as a sentence, not a table.
3. **a blockquote collapses the same way**, and its `>` markers survive inside
   the merged text (`- > **Superseded.** The eight per-model scripts ... > since
   been ...`), so the reader sees stray angle brackets mid-sentence.

`--require` cannot catch any of this: it greps for tokens and every token did
survive, only glued together.

The fourth defect is the lower-case fragment fold, which works in neither of its
two cases. It tests `out[-1].startswith("- ")` *before* walking back over the
wrapped continuation lines, so when the previous bullet wrapped -- the normal
case at the 100-column default -- the fragment gets its own bullet, and when it
did not wrap, the code re-wraps a line that already carries its marker and
publishes `- - The reader coalesces adjacent ranges. e.g. ...`. Both measured on
/tmp/aud263-probe.md and /tmp/aud263b.md.

The rest of the suite pins the guards the release path relies on and that were
equally untested: `### Checksum` emitted byte-for-byte because release.sh
substitutes placeholders into it and greps the result, `--require` failing when a
token a grep depends on does not survive, and `--max-chars` failing over budget.
Two more pins came out of the mutation sweep of the new branches: the fold is
conditional on the fragment starting lower-case -- folding a capitalised sentence
merges two claims into one bullet -- and opening a fence flushes the paragraph
that introduces it, which otherwise gets published *below* its own example.

    cd benchmark && python3 -m unittest test_release_notes_compact -v
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "compact-release-notes.py"

FENCE_IN_A_BULLET = """### Progress lines

- **The installer prints one line per 10%.** It used to print nothing between
  finished shards, so a fetch that runs for hours looked like a hang:

  ```
  converting  38.7/91.6 GB  42%  4/13 shards  eta 12m
  installing  12.3/38.1 GB  32%  eta 4m10s
  ```

  **Checked by** `benchmark/test_progress.py` (twelve tests).
"""

FENCE_AT_THE_MARGIN = """### Launcher

```bash
swift run -c release TinyTitanCLI --model models/qwen3.5_4B_4Bit
```
"""

TABLE = """### What that is worth, per install

| Model | 4-bit | 8-bit |
| --- | ---: | ---: |
| Qwen 3.6 35B-A3B | +1.8% | +11.3% |
| Ornith 1.5 35B-A3B | +1.8% | +12.6% |
"""

BLOCKQUOTE = """### A start script per model

> **Superseded.** The eight per-model scripts have since been retired in favour
> of one launcher, which asks what to launch and serves every installed model on
> one port.
"""

# Prose, not a bullet: this is the shape that reaches the fold, because the
# bullet path never splits a paragraph into sentences.
LONG_PROSE_THEN_FRAGMENT = """### Streaming

The reader coalesces adjacent expert ranges before the disk sees them, which is
the reason decode latency fell by a third on the 125B install. e.g. the 4-bit
build reads nine ranges where it used to read twenty-seven.
"""

SHORT_PROSE_THEN_FRAGMENT = """### Streaming

The reader coalesces adjacent ranges. e.g. the 4-bit build reads nine ranges.
"""

# A new sentence is a new claim, so it must not fold into the bullet above it.
BULLET_THEN_CAPITALISED = """### Streaming

- The reader coalesces adjacent expert ranges before the disk sees them.

The same holds for the writer, which batches every shard it opens.
"""

# A fence interrupts a paragraph, so the sentence introducing it stays above it.
PROSE_THEN_FENCE = """### Launcher

Run this before anything else.
```bash
swift run -c release TinyTitanCLI
```
"""

CHECKSUM_NOTES = """### Checksum

| Artifact | sha256 |
| --- | --- |
| tinytitan-9.9-macos-arm64.tar.gz | SHA256_PENDING |
"""

BLOCK_SHAPES = (FENCE_IN_A_BULLET, FENCE_AT_THE_MARGIN, TABLE, BLOCKQUOTE)
FOLD_SHAPES = (
    LONG_PROSE_THEN_FRAGMENT,
    SHORT_PROSE_THEN_FRAGMENT,
    BULLET_THEN_CAPITALISED,
    PROSE_THEN_FENCE,
)


def compact(text: str, *args: str) -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory() as tmp:
        src = pathlib.Path(tmp) / "notes.md"
        dst = pathlib.Path(tmp) / "compact.md"
        src.write_text(text, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), str(src), "--out", str(dst), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        proc.compact_out = dst.read_text(encoding="utf-8") if dst.exists() else ""
    return proc


def blocks(text: str) -> list[tuple[str, list[str]]]:
    """Each fenced block, table and blockquote, as the run of lines it is.

    Joined rather than a bag of lines on purpose: a compactor that keeps every
    line but puts a blank between two table rows produces output this test must
    fail, because a table separated line by line is not a table.
    """
    found: list[tuple[str, list[str]]] = []
    for block in re.findall(r"^[ \t]*```[^\n]*\n(.*?)^[ \t]*```", text, re.M | re.S):
        found.append(("fence", [line.strip() for line in block.rstrip("\n").splitlines()]))
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        marker = stripped[:1]
        if marker in ("|", ">"):
            run = []
            while index < len(lines) and lines[index].strip().startswith(marker):
                run.append(lines[index].strip())
                index += 1
            found.append(("table" if marker == "|" else "quote", run))
            continue
        index += 1
    return found


def assert_block(
    result: subprocess.CompletedProcess,
    run: list[str],
    note: str = "",
    fence: bool = False,
) -> None:
    """`run` travels as one contiguous block.

    A compactor that keeps every line but drops the blank between two rows
    publishes a table that is not a table, so the block's neighbours are part of
    the assertion: a fenced block keeps its ``` markers, and a table or quote is
    delimited from the text around it by blank lines.
    """
    out = result.compact_out
    joined = "\n".join(run)
    at = out.find(joined)
    if at == -1:
        raise AssertionError(f"{note}block merged or lost:\n{joined}\nin\n{out}")
    before = out[:at].splitlines()
    after = out[at + len(joined) :].split("\n")
    prev_line = before[-1] if before else ""
    next_line = after[1] if len(after) > 1 else ""
    if fence:
        if not prev_line.lstrip().startswith("```") or not next_line.lstrip().startswith("```"):
            raise AssertionError(
                f"{note}the fence markers did not travel:\n{prev_line}\n{joined}\n{next_line}"
            )
        return
    if prev_line.strip() or next_line.strip():
        raise AssertionError(
            f"{note}block is not blank-delimited:\n{prev_line}\n{joined}\n{next_line}"
        )


class VerbatimBlockTests(unittest.TestCase):
    def test_a_fenced_block_inside_a_bullet_keeps_its_line_breaks(self) -> None:
        result = compact(FENCE_IN_A_BULLET)
        self.assertEqual(result.returncode, 0, result.stderr)
        for kind, run in blocks(FENCE_IN_A_BULLET):
            assert_block(result, run, f"{kind} inside a bullet: ", fence=kind == "fence")

    def test_a_fenced_block_at_the_margin_is_not_turned_into_prose(self) -> None:
        result = compact(FENCE_AT_THE_MARGIN)
        self.assertEqual(result.returncode, 0, result.stderr)
        for kind, run in blocks(FENCE_AT_THE_MARGIN):
            assert_block(result, run, f"{kind} at the margin: ", fence=kind == "fence")

    def test_a_table_keeps_one_row_per_line(self) -> None:
        result = compact(TABLE)
        self.assertEqual(result.returncode, 0, result.stderr)
        for kind, run in blocks(TABLE):
            self.assertEqual((kind, len(run)), ("table", 4), run)
            assert_block(result, run, "table: ")

    def test_a_blockquote_keeps_its_markers_and_its_lines(self) -> None:
        result = compact(BLOCKQUOTE)
        self.assertEqual(result.returncode, 0, result.stderr)
        for kind, run in blocks(BLOCKQUOTE):
            self.assertEqual((kind, len(run)), ("quote", 3), run)
            assert_block(result, run, "blockquote: ")
        merged = [
            line
            for line in result.compact_out.splitlines()
            if line.startswith("- ") and "> " in line
        ]
        self.assertEqual(merged, [], f"quote markers left inside a bullet:\n{merged}")

    def test_the_shipped_notes_keep_their_tables_examples_and_quotes_as_blocks(self) -> None:
        """The corpus, not a fixture: this is what the Release page shows."""
        checked = 0
        for notes in sorted(ROOT.glob("docs/release-notes-v*.md")):
            text = notes.read_text(encoding="utf-8")
            runs = blocks(text)
            if not runs:
                continue
            result = compact(text)
            self.assertEqual(result.returncode, 0, f"{notes}: {result.stderr}")
            for kind, run in runs:
                checked += 1
                assert_block(result, run, f"{notes.name} {kind}: ", fence=kind == "fence")
        self.assertGreater(checked, 0, "no notes file has a table, fence or quote -- fix the probe")

    def test_a_fence_stays_below_the_sentence_that_introduces_it(self) -> None:
        """A fence interrupts a paragraph, so opening one must flush it first."""
        result = compact(PROSE_THEN_FENCE)
        self.assertEqual(result.returncode, 0, result.stderr)
        out = result.compact_out
        self.assertLess(
            out.index("- Run this before anything else."),
            out.index("```bash"),
            f"the example was published above the line that introduces it:\n{out}",
        )
        for kind, run in blocks(PROSE_THEN_FENCE):
            assert_block(result, run, f"{kind} after a paragraph: ", fence=kind == "fence")


class FragmentFoldTests(unittest.TestCase):
    def test_a_fragment_folds_into_a_bullet_that_wrapped(self) -> None:
        result = compact(LONG_PROSE_THEN_FRAGMENT)
        self.assertEqual(result.returncode, 0, result.stderr)
        bullets = [line for line in result.compact_out.splitlines() if line.startswith("- ")]
        self.assertEqual(len(bullets), 1, f"the fragment got its own bullet:\n{result.compact_out}")
        self.assertIn("e.g. the 4-bit", result.compact_out, result.compact_out)

    def test_the_fold_does_not_double_the_list_marker(self) -> None:
        result = compact(SHORT_PROSE_THEN_FRAGMENT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("- - ", result.compact_out, result.compact_out)
        self.assertIn(
            "The reader coalesces adjacent ranges. e.g. the 4-bit build reads nine ranges.",
            result.compact_out,
            result.compact_out,
        )

    def test_a_capitalised_sentence_after_a_bullet_stays_its_own_bullet(self) -> None:
        """Only a lower-case fragment folds, and folding is opt-in per sentence.

        Dropping the case test merges two claims into one bullet -- measured on
        this input, the publisher's own sentence arrives inside the first one.
        """
        result = compact(BULLET_THEN_CAPITALISED)
        self.assertEqual(result.returncode, 0, result.stderr)
        bullets = [line for line in result.compact_out.splitlines() if line.startswith("- ")]
        self.assertEqual(
            len(bullets),
            2,
            f"a new sentence was folded into the bullet above:\n{result.compact_out}",
        )
        self.assertTrue(
            bullets[1].startswith("- The same holds for the writer"),
            f"the second claim did not survive as its own bullet:\n{result.compact_out}",
        )

    def test_compacting_its_own_output_changes_nothing(self) -> None:
        """The docstring's idempotency promise, on every shape it must preserve."""
        for text in BLOCK_SHAPES + FOLD_SHAPES:
            first = compact(text)
            self.assertEqual(first.returncode, 0, first.stderr)
            second = compact(first.compact_out)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(
                first.compact_out, second.compact_out, "the compact form is not a fixed point"
            )


class ChecksumAndRefusalTests(unittest.TestCase):
    def test_the_checksum_section_is_emitted_byte_for_byte(self) -> None:
        result = compact(CHECKSUM_NOTES)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("| tinytitan-9.9-macos-arm64.tar.gz | SHA256_PENDING |", result.compact_out)

    def test_a_required_token_that_did_not_survive_refuses_the_notes(self) -> None:
        result = compact(FENCE_IN_A_BULLET, "--require", "eta 4m10s", "--require", "ornith-4")
        self.assertNotEqual(result.returncode, 0, "the compactor accepted dropped content")
        self.assertIn("ornith-4", result.stderr, result.stderr)
        self.assertNotIn("'eta 4m10s'", result.stderr, result.stderr)

    def test_a_required_token_that_survived_is_silent(self) -> None:
        result = compact(FENCE_IN_A_BULLET, "--require", "Checked by")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_notes_over_the_budget_are_refused_with_the_count(self) -> None:
        result = compact(FENCE_IN_A_BULLET, "--max-chars", "10")
        self.assertNotEqual(result.returncode, 0, "over-budget notes were accepted")
        self.assertIn("budget", result.stderr, result.stderr)


if __name__ == "__main__":
    unittest.main()
