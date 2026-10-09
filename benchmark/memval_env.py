#!/usr/bin/env python3
"""The one reading of the environment that names a memory run, shared by the drivers.

Every `memory_*` driver binds its results tree from this variable at import, and
the drivers that write and the drivers that report must land on the same tree or
the report reads back a directory the run never wrote to. A value that was set to
empty is the failure mode `Path(os.environ.get(...))` cannot express: it is not
"unset", and it resolves to the directory the command happens to run in.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV = "TINYTITAN_MEMVAL_RESULTS"


def results_tree(default: Path) -> Path:
    """The tree the environment names, or `default` when the variable is unset.

    A set-but-blank value is refused rather than resolved: a run record dropped
    in the launcher's cwd is invisible to every later report, and a report over
    that directory would print a path that identifies nothing.
    """
    named = os.environ.get(ENV)
    if named is not None and not named.strip():
        raise SystemExit(
            f"ABORT: {ENV} is set to {named!r}, which names no directory. Left as it is "
            f"the tree becomes the current directory, so this run would write its records "
            f"wherever it happened to be launched and no report would find them. Unset "
            f"{ENV} to use {default}, or name a directory."
        )
    return Path(named) if named else default


def publish_results_tree(default: Path) -> Path:
    """Set the environment to the tree this process uses, and return it.

    For a driver that imports the others and needs them to agree on one tree.
    """
    tree = results_tree(default)
    os.environ[ENV] = str(tree)
    return tree


RUN_ENV = "TINYTITAN_MEMVAL_RUN"


def run_token(default: str = "1") -> str:
    """Which repeat of the arm this is, from `TINYTITAN_MEMVAL_RUN`.

    The token is part of every result file's name, so a set-but-blank one is not
    a cosmetic slip: all the repeats of an arm write the same file, each
    overwriting the last, and the report counts the files it finds. Measured over
    three launches: with the token `1`, `2`, `3` the tree keeps three records and
    the report prints three rows; blank on all three it keeps one and prints one,
    with exit 0 either way.
    """
    named = os.environ.get(RUN_ENV)
    if named is not None and not named.strip():
        raise SystemExit(
            f"ABORT: {RUN_ENV} is set to {named!r}, which names no run. Every repeat of an "
            f"arm would be written to the same result file and overwrite the last, so a "
            f"report over them could not tell three runs from one. Unset {RUN_ENV} to use "
            f"run {default!r}, or name the repeat."
        )
    return named if named else default
