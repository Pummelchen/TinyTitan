#!/usr/bin/env python3
"""The install readers refuse an oversized metadata document before reading it.

The Swift engine reads a manifest through `BoundedMetadataRead`, which `fstat`s
the descriptor and refuses over its cap *before* allocating: a 2 GiB sparse
`manifest.json` cost 2,049 MB of `phys_footprint` when it was read whole
(measured 2026-11-26). These tools open the same installs, so a hostile or
corrupt document is the same trust decision -- `read_text()` allocates whatever
the file claims to be, and the claim is the untrusted part (AUD-0103).

The tests are sized cheaply: an oversized document is a sparse file, so its
real length is what the reader sees and no disk is spent. The read-before-alloc
test fakes the descriptor's size instead, which is the only way to prove the
bound is applied to `fstat` rather than to the bytes already in memory.
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import quantize_ngram_table  # noqa: E402
import ssdai_metadata  # noqa: E402
import ssdai_reader  # noqa: E402
import tiered_repack  # noqa: E402

try:
    import verify_ane_sidecar
    import export_ane_prefill

    VERIFY_ERROR = ""
except Exception as exc:  # coremltools absent
    verify_ane_sidecar = None
    export_ane_prefill = None
    VERIFY_ERROR = str(exc)

LIMIT = ssdai_metadata.DEFAULT_LIMIT


def oversized(path: pathlib.Path, document: str = "{}") -> pathlib.Path:
    """A document one byte over the ceiling, with almost no disk spent on it.

    The file starts as real JSON and then has a hole, so an *unbounded* reader
    gets a `JSONDecodeError` -- which is a `ValueError`, and would satisfy a
    bare `assertRaises(ValueError)` on the very code these tests exist to fail.
    That is why every call site asserts the message names the metadata limit:
    only `read_metadata` can say that, and only before it allocates.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(document.encode())
        handle.seek(LIMIT)
        handle.write(b"\n")
    return path


class BoundedMetadataReadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = pathlib.Path(self.tmp.name)

    def test_a_document_under_the_ceiling_is_returned(self):
        path = self.dir / "manifest.json"
        path.write_text(json.dumps({"files": {"a": 1}}))
        self.assertEqual(ssdai_metadata.read_metadata_json(path), {"files": {"a": 1}})
        self.assertEqual(ssdai_metadata.read_metadata(path), path.read_bytes())

    def test_a_document_over_the_ceiling_is_refused(self):
        path = oversized(self.dir / "manifest.json")
        with self.assertRaises(ValueError) as raised:
            ssdai_metadata.read_metadata_json(path)
        self.assertIn("metadata limit", str(raised.exception))
        self.assertIn(str(LIMIT), str(raised.exception))

    def test_the_bound_is_applied_to_the_descriptor_not_the_bytes(self):
        # The file on disk is one byte, but the descriptor claims to be over
        # the cap: the read must not happen at all. A bound applied after the
        # read would allocate the claim and then refuse it.
        handle = mock.MagicMock()
        handle.fileno.return_value = 7
        opened = mock.MagicMock()
        opened.__enter__.return_value = handle
        with (
            mock.patch("builtins.open", return_value=opened),
            mock.patch(
                "ssdai_metadata.os.fstat",
                return_value=SimpleNamespace(st_size=LIMIT + 1),
            ),
        ):
            with self.assertRaises(ValueError):
                ssdai_metadata.read_metadata("claimed.json")
        handle.read.assert_not_called()


class ToolReadsAreBoundedTests(unittest.TestCase):
    """Each reader that opens an install refuses the oversized document."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = pathlib.Path(self.tmp.name)

    def test_the_resident_weights_reader_refuses_an_oversized_manifest(self):
        oversized(self.dir / "manifest.json")
        with self.assertRaisesRegex(ValueError, "metadata limit"):
            ssdai_reader.SSDAIWeights(self.dir)

    def test_the_expert_reader_refuses_an_oversized_layout(self):
        (self.dir / "manifest.json").write_text("{}")
        oversized(self.dir / "packed_experts" / "layout.json", '{"layers": []}')
        with self.assertRaisesRegex(ValueError, "metadata limit"):
            ssdai_reader.PackedExperts(self.dir)

    def test_tiered_repack_refuses_an_oversized_manifest_before_writing(self):
        model = self.dir / "model"
        oversized(model / "manifest.json")
        out = self.dir / "out"
        argv = [
            "tiered_repack.py",
            "--model",
            str(model),
            "--hot-profile",
            str(self.dir / "hot.json"),
            "--out",
            str(out),
        ]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaisesRegex(ValueError, "metadata limit"):
                tiered_repack.main()
        self.assertFalse(out.exists(), "the refused run created its output directory")

    def test_quantize_ngram_table_refuses_an_oversized_constants(self):
        install = self.dir / "install"
        oversized(install / "ple_constants.json")
        table = install / "ngram_table.q4"
        table.write_bytes(b"table")
        with self.assertRaisesRegex(ValueError, "metadata limit"):
            quantize_ngram_table.update_metadata(install, table)
        # Refused before the rewrite, so the document the reader refused is
        # still the one on disk.
        self.assertEqual((install / "ple_constants.json").stat().st_size, LIMIT + 1)

    @unittest.skipIf(
        verify_ane_sidecar is None, f"sidecar verifier needs coremltools: {VERIFY_ERROR}"
    )
    def test_the_sidecar_verifier_refuses_an_oversized_manifest(self):
        model = self.dir / "model"
        chunk = export_ane_prefill.CHUNK
        directory = model / export_ane_prefill.sidecar_directory(chunk)
        directory.mkdir(parents=True)
        (directory / "ane_prefill.json").write_text(json.dumps({"chunkTokens": chunk}))
        oversized(model / "manifest.json")
        argv = ["verify_ane_sidecar.py", "--model", str(model), "--chunk", str(chunk)]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaisesRegex(ValueError, "metadata limit"):
                verify_ane_sidecar.main()
