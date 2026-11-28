"""Argument arithmetic for the ANE sidecar exporter, testable without Core ML.

`export_ane_prefill.py` imports `coremltools` at module scope, so nothing in it
can be unit-tested on a machine without that stack. The one piece of its
argument handling that can silently produce the wrong export is the history
list: `range(0, max_history + 1, chunk)` is empty for a negative max-history
that happens to be a multiple of the chunk, and the exporter then builds a
multi-function package with no functions — an export whose failure names the
conversion rather than the argument.
"""

from __future__ import annotations


def history_variants(max_history: int, chunk: int) -> list[int]:
    """The KV histories a sidecar carries, or a refusal naming the argument.

    `max-history` is the largest history variant, so it is a count of tokens,
    not a signed offset: a negative value is refused before the exporter touches
    the model. The multiple-of-chunk rule is the graph's own: every variant ends
    on a chunk boundary.
    """
    if max_history < 0:
        raise ValueError(f"--max-history must be 0 or more, got {max_history}")
    if chunk < 1:
        raise ValueError(f"--chunk must be at least 1, got {chunk}")
    if max_history % chunk:
        raise ValueError(
            f"--max-history must be a multiple of --chunk ({chunk}), got {max_history}"
        )
    return list(range(0, max_history + 1, chunk))
