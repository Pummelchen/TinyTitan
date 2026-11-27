"""Bounded reads of an install's metadata documents.

The Swift side reads a manifest through `BoundedMetadataRead`, which refuses a
document that is over its ceiling before allocating anything: a 2 GiB sparse
`manifest.json` cost 2,049 MB of footprint when it was read whole, measured
2026-11-26. These tools open the same installs, so they inherit the same rule --
`read_text()`/`read_bytes()` on a metadata file allocates whatever the file
claims to be, and the claim is the untrusted part.

`safe_open`-style model data is streamed elsewhere in these tools; this module is
only for the small documents beside it.
"""

from __future__ import annotations

import os

# A ceiling chosen here, the way the engine's `BoundedMetadataRead` takes one
# from each caller: manifests and layout documents are kilobytes, so 64 MiB
# leaves every real document room and still refuses a multi-gigabyte claim
# before it allocates.
DEFAULT_LIMIT = 64 * 1024 * 1024


def read_metadata(path, limit=DEFAULT_LIMIT):
    """The bytes of `path`, refused over `limit` before the read allocates.

    Raises `ValueError` with the path, the size and the ceiling -- a message an
    operator can act on -- rather than reading whatever is there.
    """
    with open(path, "rb") as handle:
        size = os.fstat(handle.fileno()).st_size
        if size > limit:
            raise ValueError(f"{path} is {size} bytes, over the {limit}-byte metadata limit")
        return handle.read()


def read_metadata_json(path, limit=DEFAULT_LIMIT):
    """`json.loads` of `read_metadata`, for the documents that are JSON."""
    import json

    return json.loads(read_metadata(path, limit))
