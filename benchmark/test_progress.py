#!/usr/bin/env python3.13
"""Tests for `tools/lib/progress.py`, the installer's progress line.

The line is cosmetic until it is wrong: a download that reports a percentage
nobody can act on, an ETA that says `eta 0s` for an hour, or a closing newline
that mangles the install summary are all small, and all are the kind of thing
that survives forever because nothing pins them. These cases pin the arithmetic
and both output modes (a terminal rewrites one line; a pipe prints steps).

    cd benchmark && python3 -m unittest test_progress -v
"""

from __future__ import annotations

import io
import os
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "lib"))

import progress  # noqa: E402


class FakeStream(io.StringIO):
    """A stream that claims to be a terminal, or does not."""

    def __init__(self, tty: bool) -> None:
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


class HumanFormattingTests(unittest.TestCase):
    def test_bytes_use_decimal_units_like_the_install_menu(self) -> None:
        self.assertEqual(progress.human_bytes(0), "0 B")
        self.assertEqual(progress.human_bytes(512), "512 B")
        self.assertEqual(progress.human_bytes(1_500), "2 kB")
        self.assertEqual(progress.human_bytes(2_500_000), "2.5 MB")
        self.assertEqual(progress.human_bytes(91_600_000_000), "91.6 GB")

    def test_seconds_read_as_a_person_would_say_them(self) -> None:
        self.assertEqual(progress.human_seconds(0), "0s")
        self.assertEqual(progress.human_seconds(45.9), "45s")
        self.assertEqual(progress.human_seconds(250), "4m10s")
        self.assertEqual(progress.human_seconds(3900), "1h05m")

    def test_a_share_names_its_unit_once_like_the_repack_binary(self) -> None:
        # "12.3/38.1 GB", not "12.3 GB/38.1 GB": the repack binary renders its
        # copy phase this way, and a whole install has to read as one story.
        self.assertEqual(progress.share(12_300_000_000, 38_100_000_000), "12.3/38.1 GB")
        self.assertEqual(progress.share(0, 100_000), "0.0/0.1 MB")
        self.assertEqual(progress.share(2_500_000, 5_000_000), "2.5/5.0 MB")


class ModeTests(unittest.TestCase):
    def test_a_disabled_line_writes_nothing(self) -> None:
        stream = FakeStream(tty=True)
        line = progress.Progress("downloading", 100, stream=stream, enabled=False)
        line.add(50)
        line.note("still nothing")
        line.finish("done")
        self.assertEqual(stream.getvalue(), "")

    def test_the_environment_can_turn_it_off(self) -> None:
        previous = os.environ.get(progress.DISABLED_ENV)
        os.environ[progress.DISABLED_ENV] = "1"
        try:
            self.assertFalse(progress.Progress("x", 10, stream=FakeStream(True)).enabled)
        finally:
            if previous is None:
                del os.environ[progress.DISABLED_ENV]
            else:
                os.environ[progress.DISABLED_ENV] = previous

    def test_a_terminal_rewrites_one_line_and_finish_ends_it(self) -> None:
        stream = FakeStream(tty=True)
        line = progress.Progress("downloading", 100, stream=stream, interval=0.0)
        line.add(10)
        line.add(50)
        self.assertEqual(stream.getvalue().count("\n"), 0, "no newline before the end")
        self.assertTrue(stream.getvalue().startswith("\r"), "rewritten in place")
        self.assertIn("downloading", stream.getvalue())
        self.assertIn("60%", stream.getvalue())
        line.finish()
        self.assertTrue(stream.getvalue().endswith("\n"), "the end breaks the line")

    def test_a_pipe_gets_one_line_per_step_not_per_chunk(self) -> None:
        stream = FakeStream(tty=False)
        line = progress.Progress("downloading", 100, stream=stream, interval=0.0)
        for _ in range(100):
            line.add(1)
        lines = stream.getvalue().splitlines()
        # A line at the start and one per 10% step -- not one per chunk, which
        # would be 100 lines for a 100-chunk download.
        self.assertLessEqual(len(lines), 12)
        self.assertGreaterEqual(len(lines), 10)
        self.assertIn("100%", lines[-1])
        line.finish()
        self.assertEqual(
            len(stream.getvalue().splitlines()), len(lines), "finish does not repeat the last step"
        )

    def test_an_unknown_total_still_says_how_much_arrived(self) -> None:
        stream = FakeStream(tty=False)
        line = progress.Progress("downloading", None, stream=stream)
        line.add(2_500_000)
        line.finish("finished")
        text = stream.getvalue()
        self.assertIn("2.5 MB", text)
        self.assertNotIn("%", text)
        self.assertTrue(text.endswith("finished\n"))

    def test_note_clears_the_line_before_it_prints(self) -> None:
        stream = FakeStream(tty=True)
        line = progress.Progress("downloading", 100, stream=stream, interval=0.0)
        line.add(10)
        line.note("    shard-1 @0: chunk failed, waiting 5 s")
        text = stream.getvalue()
        self.assertIn("\r    shard-1 @0: chunk failed, waiting 5 s\n", text)

    def test_item_progress_counts_units_not_bytes(self) -> None:
        stream = FakeStream(tty=True)
        line = progress.Progress("converting", stream=stream, interval=0.0)
        line.item(4, 13, "model-00003.safetensors")
        text = stream.getvalue()
        self.assertIn("converting  4/13", text)
        self.assertIn("model-00003.safetensors", text)
        self.assertIn("31%", text)
        line.finish()
        self.assertIn("4/13", stream.getvalue())

    def test_finish_is_idempotent(self) -> None:
        stream = FakeStream(tty=True)
        line = progress.Progress("downloading", 100, stream=stream, interval=0.0)
        line.finish()
        once = stream.getvalue()
        line.finish()
        self.assertEqual(stream.getvalue(), once)

    def test_a_closed_pipe_is_not_an_install_failure(self) -> None:
        class Broken(io.StringIO):
            def isatty(self) -> bool:
                return False

            def write(self, _text: str) -> int:
                raise BrokenPipeError("gone")

        line = progress.Progress("downloading", 100, stream=Broken())
        line.add(50)
        line.finish()
        self.assertFalse(line.enabled, "a stream that went away turns the line off")


if __name__ == "__main__":
    unittest.main()
