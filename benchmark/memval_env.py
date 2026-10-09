#!/usr/bin/env python3
"""The one reading of `TINYTITAN_MEMVAL_RESULTS`, shared by the memory drivers.

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
