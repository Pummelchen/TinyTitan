#!/usr/bin/env python3.13
"""One-line progress for the long installer steps: a download, a conversion.

A Qwen3.5-MoE install fetches ~70 GB and a Qwen3.8-Flash-Next install ~360 GB,
one shard at a time, and the converters used to print nothing between finished
shards -- so a terminal sat on the same line for half an hour and the honest
reading was "it hung". This module draws the line that says otherwise.

The rendering contract is deliberately the same one the repack binary uses
(`sources/TinyTitanRepack/Command/InstallProgressDisplay.swift`), so a whole
install reads as one continuous story:

    downloading  38.7/91.6 GB  42%  eta 12m
    converting  4/13  30%  model-00003.safetensors

Two output modes, because both are read by people:

* a terminal gets one line rewritten in place (`\\r`), with an ETA computed
  from the observed rate;
* a pipe or a log file gets a plain line at each 10% step, so
  `install_models.sh > log` still shows that progress happened and a captured
  install is diagnosable after the fact.

`TINYTITAN_NO_PROGRESS=1` (or `on`/`true`) turns every line off, which is what
the test suite uses so output assertions stay about the thing under test.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import TextIO

DISABLED_ENV = "TINYTITAN_NO_PROGRESS"

#: How often a terminal redraw happens, in seconds. A redraw is cheap, but a
#: chunked download can add a chunk every few hundred milliseconds and a
#: flickering line is worse than a calm one.
DEFAULT_INTERVAL = 1.0

#: The step between lines when there is no terminal to rewrite.
LOG_STEP_PERCENT = 10


def human_bytes(count: float) -> str:
    """Bytes as a person reads them: GB with one decimal, then MB.

    Decimal units on purpose: the catalogue, the install menu and the server's
    own disk checks all speak GB = 1e9, and a progress line that disagreed with
    the size the launcher just printed would look like a bug.
    """
    if count >= 1e9:
        return f"{count / 1e9:.1f} GB"
    if count >= 1e6:
        return f"{count / 1e6:.1f} MB"
    if count >= 1e3:
        return f"{count / 1e3:.0f} kB"
    return f"{int(count)} B"


def human_seconds(seconds: float) -> str:
    """A duration as `12s`, `4m10s` or `1h04m`, for an ETA."""
    total = int(max(0.0, seconds))
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _scaled(count: float) -> tuple[float, str, int]:
    """A byte count as `(value, unit, decimals)` in the unit it is read in."""
    if count >= 1e9:
        return count / 1e9, "GB", 1
    if count >= 1e6:
        return count / 1e6, "MB", 1
    if count >= 1e3:
        return count / 1e3, "kB", 0
    return float(int(count)), "B", 0


def share(done: float, total: float) -> str:
    """`12.3/38.1 GB`: a done/total pair with the unit written once.

    The repack binary renders its copy phase the same way and with the same
    unit rule (`sources/TinyTitanRepack/Command/InstallProgressDisplay.swift`:
    GB above 1e9, MB below), so a whole install -- download, convert, install --
    reads as one story rather than three dialects. Both numbers take the
    total's unit, because that is the number the launcher printed when the
    person agreed to the download.
    """
    if total >= 1e9:
        unit = "GB"
        scale = 1e9
    else:
        unit = "MB"
        scale = 1e6
    return f"{done / scale:.1f}/{total / scale:.1f} {unit}"


def _disabled_by_environment() -> bool:
    return os.environ.get(DISABLED_ENV, "").strip().lower() in ("1", "on", "true", "yes")


class Progress:
    """One progress line, for bytes or for finished items.

    Bytes (`add`/`set`) are for a download or a copy; items (`item`) are for a
    pass that finishes whole units, like one checkpoint shard converted. Both
    render through the same line so a caller can switch from fetching to
    converting without the display changing shape.

    Nothing here raises: a progress line is never worth failing an install
    over, so a stream that has gone away (a closed pipe) is swallowed.
    """

    def __init__(
        self,
        label: str,
        total: float | None = None,
        *,
        stream: TextIO | None = None,
        interval: float = DEFAULT_INTERVAL,
        enabled: bool | None = None,
    ) -> None:
        self.label = label
        self.stream: TextIO = stream if stream is not None else sys.stdout
        self.total = float(total) if isinstance(total, (int, float)) and total > 0 else None
        self.interval = interval
        self.done = 0.0
        self.started = time.monotonic()
        #: A download and a conversion run at the same time (a small fetch pool
        #: feeds a converter), so every mutation and every draw is serialised:
        #: two threads writing one line is how a progress bar becomes garbage.
        self._lock = threading.Lock()
        self._last_draw = 0.0
        self._last_step = -1
        self._on_screen = False
        self._finished = False
        #: `bytes` for a download or a copy, `items` for a pass that finishes
        #: whole units. `item()` switches it; the two never mix on one line.
        self.unit = "bytes"
        if enabled is None:
            enabled = not _disabled_by_environment()
        self.enabled = enabled
        try:
            self.is_tty = bool(self.stream.isatty())
        except Exception:  # noqa: BLE001  # a stream without isatty is not a terminal
            self.is_tty = False

    # ------------------------------------------------------------------ state

    @property
    def percent(self) -> float | None:
        """Completion in 0..100, or `None` when the total is unknown."""
        if self.total is None:
            return None
        return min(100.0, max(0.0, self.done / self.total * 100.0))

    def _eta(self) -> str:
        """`eta 4m10s`, or empty until the rate means something."""
        elapsed = time.monotonic() - self.started
        if self.total is None or self.done <= 0 or elapsed < 2.0:
            return ""
        rate = self.done / elapsed
        if rate <= 0:
            return ""
        return f"  eta {human_seconds((self.total - self.done) / rate)}"

    def _line(self, detail: str) -> str:
        percent = self.percent
        head = f"{self.label}  {detail}"
        if percent is None:
            return head
        return f"{head}  {percent:.0f}%{self._eta()}"

    # -------------------------------------------------------------- rendering

    def _write(self, text: str) -> None:
        if not self.enabled:
            return
        try:
            self.stream.write(text)
            self.stream.flush()
        except Exception:  # noqa: BLE001  # a closed pipe must not fail an install
            self.enabled = False

    def _draw(self, detail: str, *, force: bool = False) -> None:
        if not self.enabled:
            return
        percent = self.percent
        if self.is_tty:
            now = time.monotonic()
            if not force and now - self._last_draw < self.interval:
                return
            self._last_draw = now
            self._write("\r" + self._line(detail).ljust(self._width()))
            self._on_screen = True
            return
        # No terminal: one plain line per step, so a log shows progress without
        # a line per chunk. An unknown total prints on every forced draw only.
        if percent is None:
            if force:
                self._write(self._line(detail) + "\n")
            return
        step = int(percent // LOG_STEP_PERCENT)
        if force and step == self._last_step:
            # `finish()` and `item()` force a draw; without this the closing
            # 100% line would repeat the step the last update already printed.
            return
        if force or step > self._last_step:
            self._last_step = step
            self._write(self._line(detail) + "\n")

    def _width(self) -> int:
        try:
            return max(20, os.get_terminal_size().columns - 1)
        except Exception:  # noqa: BLE001  # no terminal size: any width will do
            return 100

    # ---------------------------------------------------------------- updates

    def set(self, done: float, detail: str | None = None) -> None:
        """Set the absolute amount done (bytes), for a resumed download."""
        with self._lock:
            self.done = float(done)
            self._draw(detail if detail is not None else self.amount)

    def add(self, count: float, detail: str | None = None) -> None:
        """Add to the amount done (bytes), for one finished chunk."""
        with self._lock:
            self.done += float(count)
            self._draw(detail if detail is not None else self.amount)

    def show(self, detail: str) -> None:
        """Redraw now with a caller's own text, without touching the counter.

        The pipeline fetches and converts at once, so the useful line carries
        both: `converting  8/13  42%  38.7/91.6 GB`. The byte count is the
        counter; the shard count is the caller's detail.
        """
        with self._lock:
            self._draw(detail, force=True)

    def item(self, done: int, total: int, name: str | None = None) -> None:
        """A whole-unit pass: `converting  4/13  30%  model-00003.safetensors`."""
        self.unit = "items"
        self.total = float(total) if total > 0 else None
        with self._lock:
            self.done = float(done)
            detail = f"{done}/{total}"
            if name:
                detail = f"{detail}  {name}"
            self._draw(detail, force=True)

    @property
    def amount(self) -> str:
        """The done/total text for the mode this line is in."""
        if self.unit == "items":
            total = "" if self.total is None else f"/{int(self.total)}"
            return f"{int(self.done)}{total}"
        if self.total is None:
            return human_bytes(self.done)
        return share(self.done, self.total)

    def note(self, text: str) -> None:
        """Print a message without the progress line eating it.

        A retry, a resume, an error: anything a caller would otherwise `print`
        while a line is on screen. The line is cleared, the message is written
        whole, and the next update redraws the line beneath it.
        """
        with self._lock:
            if not self.enabled:
                return
            if self._on_screen:
                self._write("\r" + " " * self._width() + "\r")
                self._on_screen = False
            self._last_draw = 0.0
            self._write(text + "\n")

    def finish(self, message: str | None = None) -> None:
        """End the line: draw 100%, break it, and print a closing message.

        Idempotent, because a caller's `finally` and its success path both call
        it and a doubled newline is a cosmetic bug that never gets fixed.
        """
        with self._lock:
            if self._finished:
                return
            self._finished = True
            self._draw(self.amount, force=True)
            if self._on_screen:
                self._write("\n")
                self._on_screen = False
            if message:
                self._write(message + "\n")
