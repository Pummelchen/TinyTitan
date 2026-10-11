"""Pins what `docs/paper/build.sh` is allowed to call a successful build.

The script's last line prints a page count read off whatever PDF is on disk:

    "/Applications/Google Chrome.app/…/Google Chrome" --headless=new … \\
      --print-to-pdf="$PWD/continuitycore-paper.pdf" "file://$PWD/paper.html" 2>/dev/null
    python3 -c "… open('continuitycore-paper.pdf','rb').read() … print('pages:', …)"

Two shapes follow, and both are the class this audit keeps finding -- a verdict
computed from a run that did not happen. Measured on this machine's
/bin/bash 3.2.57 against a copy of the script whose launch is pointed at a stub:

  * the `2>/dev/null` swallows the failure. A binary that cannot be executed
    makes bash print its own "No such file or directory" on the *same* redirected
    stderr, so the operator gets a non-zero status and no text at all.
  * a launch that exits 0 without writing the PDF -- Chrome does this when
    another instance owns the profile -- leaves the previous paper on disk, and
    the page count prints over it. A stale 11-page PDF reports `pages: 11` and
    the build says it built a paper it never rendered.

`figures.py` is stubbed here on purpose: AUD-308 pinned its input refusals, and
this file is about the launch and the verdict. The template is a real one and the
inlining step runs for real, so `paper.html` exists when the launch is reached.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO / "docs" / "paper" / "build.sh"

PDF_MAGIC = b"%PDF-1.4\n1 0 obj << /Type /Page >> endobj\ntrailer << /Root 1 0 R >>\n%%EOF\n"

STUB_FAILS = """#!/usr/bin/env bash
echo "STUB CHROME: failed to launch (profile is locked)" >&2
exit 3
"""

STUB_SILENT = """#!/usr/bin/env bash
# Exits 0 and writes nothing: the shape a Chrome leaves behind when another
# instance already owns the user data directory.
exit 0
"""

STUB_WRITES = """#!/usr/bin/env bash
printf '%%PDF-1.4\\n1 0 obj << /Type /Page >> endobj\\ntrailer << /Root 1 0 R >>\\n%%%%EOF\\n' \\
  > "${TT_STUB_TARGET:?}"
exit 0
"""

STUB_WRITES_EMPTY = """#!/usr/bin/env bash
printf '%%PDF-1.4\\nno pages\\n' > "${TT_STUB_TARGET:?}"
exit 0
"""

FIGURES_STUB = """#!/usr/bin/env python3
print("stub figures")
"""


class PaperBuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="paper-build-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        paper = self.tmp / "docs" / "paper"
        paper.mkdir(parents=True)
        shutil.copy(SCRIPT, paper / "build.sh")
        (paper / "paper.template.html").write_text("<h1>{{FIG:nothing}}</h1>\n")
        (paper / "figures.py").write_text(FIGURES_STUB)
        # A real figure for the template's one placeholder.
        fig = paper / "fig"
        fig.mkdir()
        (fig / "nothing.svg").write_text("<svg/>")
        self.paper = paper
        self.pdf = paper / "continuitycore-paper.pdf"

    def stub(self, body: str) -> pathlib.Path:
        bin_dir = self.tmp / "bin"
        bin_dir.mkdir(exist_ok=True)
        chrome = bin_dir / "chrome"
        chrome.write_text(body)
        chrome.chmod(0o755)
        return chrome

    def run_build(self, *, chrome: str | None = None) -> subprocess.CompletedProcess:
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("TINYTITAN_", "TT_"))
        }
        env["PATH"] = os.environ["PATH"]
        if chrome is not None:
            env["TINYTITAN_PAPER_CHROME"] = chrome
        env["TT_STUB_TARGET"] = str(self.pdf)
        return subprocess.run(
            ["/bin/bash", str(self.paper / "build.sh")],
            capture_output=True,
            text=True,
            cwd=self.paper,
            env=env,
            timeout=120,
            check=False,
        )

    def test_a_failing_launch_is_not_silenced(self):
        """A build that cannot render has to say so: the redirect currently eats
        the child's stderr and bash's own message about the missing binary too."""
        proc = self.run_build(chrome=str(self.stub(STUB_FAILS)))
        output = proc.stdout + proc.stderr
        self.assertNotEqual(proc.returncode, 0, f"the build claimed success: {output}")
        self.assertIn(
            "profile is locked", output, f"the launch's own error was swallowed: {output}"
        )

    def test_the_default_browser_is_still_the_absolute_path(self):
        """The override exists for the tests; unset it and the script still names
        Google Chrome where it is installed, which is what an operator types."""
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", text)

    def test_a_missing_binary_says_which_one(self):
        """A browser that cannot be executed has to be named: the redirect this
        script used to set also swallowed bash's own message about it."""
        missing = self.tmp / "no-such-chrome"
        proc = self.run_build(chrome=str(missing))
        output = proc.stdout + proc.stderr
        self.assertNotEqual(proc.returncode, 0, f"the build claimed success: {output}")
        self.assertIn(
            "No such file or directory", output, f"bash's own message was swallowed: {output}"
        )
        self.assertIn(str(missing), output, f"no message names the binary it tried: {output}")
        self.assertIn("ERROR", output, f"the script adds no refusal of its own: {output}")

    def test_a_launch_that_wrote_nothing_refuses_over_a_stale_pdf(self):
        """The page count reads the disk, not the run: an exit-0 launch that wrote
        nothing has to be a refusal, not a report of the previous paper."""
        self.pdf.write_bytes(PDF_MAGIC)
        stale = self.pdf.stat().st_mtime_ns
        proc = self.run_build(chrome=str(self.stub(STUB_SILENT)))
        output = proc.stdout + proc.stderr
        self.assertNotEqual(proc.returncode, 0, f"a stale PDF was certified as the build: {output}")
        self.assertNotIn("pages: 1", output, f"the old paper's count was printed: {output}")
        self.assertIn("pdf", output.lower(), f"the refusal has to name the artifact: {output}")
        self.assertEqual(self.pdf.stat().st_mtime_ns, stale, "the refusal rewrote the file anyway")

    def test_a_build_that_renders_is_accepted(self):
        """The guard must not be a refusal of every build: a launch that writes the
        PDF still prints its page count and exits 0."""
        proc = self.run_build(chrome=str(self.stub(STUB_WRITES)))
        output = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 0, output)
        self.assertIn("pages: 1", output, output)
        self.assertTrue(self.pdf.exists(), "the accepted build left no PDF")

    def test_a_pdf_of_no_pages_is_refused(self):
        """The count is the only figure the script reports, so a zero has to be an
        error rather than a printed measurement over an empty document."""
        proc = self.run_build(chrome=str(self.stub(STUB_WRITES_EMPTY)))
        output = proc.stdout + proc.stderr
        self.assertNotEqual(
            proc.returncode, 0, f"a zero-page document was reported as a result: {output}"
        )
        self.assertNotIn("pages: 0", output, output)


if __name__ == "__main__":
    unittest.main()
