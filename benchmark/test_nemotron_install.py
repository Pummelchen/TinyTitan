"""The Nemotron installer's format gate, run as part of the Python suite.

`tools/test_nemotron_writer.py` builds a three-layer Nemotron-shaped `.ssdai`
from scratch and hands it to the real `TinyTitanRepack --verify-install` for
both widths. That is the check that has to pass before a 149.77 GB download is
started, and it costs seconds. This wrapper puts it in the suite CI runs, and
pins the two manifest invariants the format enforces on this family:

  * `arch.numExperts == expertsPerLayer` -- one install-wide count, with the
    smaller layers expressed per layer in `layout.json`;
  * every one of the model's layers has a layout entry, including the ones
    that carry no experts.

No model, no network: the toy is generated from a fixed seed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPACK = Path(os.environ.get("TINYTITAN_REPACK", str(ROOT / ".build/release/TinyTitanRepack")))
TOOL = ROOT / "tools/test_nemotron_writer.py"


@unittest.skipUnless(REPACK.exists(), "TinyTitanRepack is not built")
class NemotronInstallFormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        environment = dict(os.environ)
        environment["TINYTITAN_REPACK"] = str(REPACK)
        result = subprocess.run(
            [sys.executable, str(TOOL)],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        cls.result = result

    def test_both_widths_verify(self) -> None:
        self.assertEqual(self.result.returncode, 0, self.result.stdout + self.result.stderr)
        for width in (4, 8):
            manifest = json.loads(
                (ROOT / f".build/nemotron-toy-{width}bit/manifest.json").read_text()
            )
            self.assertEqual(manifest["magic"], "SSDAI")
            self.assertEqual(manifest["arch"]["family"], "nemotron_h")
            self.assertEqual(manifest["arch"]["numExperts"], manifest["expertsPerLayer"])
            self.assertEqual(manifest["numLayers"], 3)
            self.assertEqual(
                len(manifest["files"]),
                # model + layout + the one expert layer (mamba/attention have none)
                # + the three tokenizer stubs + the resident payload
                1 + 1 + 1 + 3,
            )

    def test_layout_covers_every_layer(self) -> None:
        for width in (4, 8):
            layout = json.loads(
                (ROOT / f".build/nemotron-toy-{width}bit/packed_experts/layout.json").read_text()
            )
            self.assertEqual(len(layout["layers"]), layout["numLayers"])
            kinds = {entry["layer"]: entry["expertsPerLayer"] for entry in layout["layers"]}
            self.assertEqual(kinds, {0: 0, 1: 8, 2: 0})
            self.assertEqual(layout["expertsPerLayer"], 8)

    def test_experts_are_two_roles(self) -> None:
        layout = json.loads(
            (ROOT / ".build/nemotron-toy-4bit/packed_experts/layout.json").read_text()
        )
        moe = next(entry for entry in layout["layers"] if entry["layer"] == 1)
        roles = set(moe["experts"][0]["tensors"])
        self.assertEqual(
            roles,
            {
                "up",
                "up_scales",
                "up_biases",
                "down",
                "down_scales",
                "down_biases",
            },
        )


if __name__ == "__main__":
    unittest.main()
