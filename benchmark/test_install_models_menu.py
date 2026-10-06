"""The install tool's contract with the person running it.

Two things `tools/install_models.sh` must not get wrong:

- `--choose` starts a real download on a single keystroke, so "the caller pressed
  Enter" and "the caller's stdin ended" have to mean different things. It was
  measured the wrong way round on 2026-09-24 — closing stdin started the
  recommended 36.9 GB download — so this file pins both directions.
- Every download and conversion is staged under one absolute work root. The
  staging paths were bare `.build/...` until 2026-09-24, which put a factory-new
  install's tens-to-hundreds of GB in `~/.build` — outside the install root —
  because a relative path follows the caller's working directory and the
  installer never changes it.

`install_one` is replaced with a stub before the menu is called: a regression
there prints `INSTALLED <model>` instead of downloading one. The path tests read
the script. Nothing here touches the network or `models/`.

    cd benchmark && python3 -m unittest test_install_models_menu -v
"""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/install_models.sh"

# `source` also runs the script's own dispatcher, which with no arguments prints
# the status table; it is silenced, and `install_one` is stubbed straight after.
HARNESS = f"""
set -uo pipefail
set --
source "{SCRIPT}" >/dev/null 2>&1 || true
install_one() {{ echo "INSTALLED $1"; }}
choose_model
echo "rc=$?"
"""


def run_menu(stdin: str | None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", HARNESS],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


class InstallMenuAnswerTests(unittest.TestCase):
    def test_an_empty_line_takes_the_default(self) -> None:
        result = run_menu("\n")
        self.assertIn("INSTALLED ornith15-8bit", result.stdout + result.stderr)

    def test_a_number_installs_that_row(self) -> None:
        result = run_menu("13\n")
        self.assertIn("INSTALLED qwen35-2b", result.stdout + result.stderr)

    def test_end_of_input_installs_nothing(self) -> None:
        # The bug: `read` failing left the reply empty, and the empty reply fell
        # through to the default. stdin closed must refuse, not choose for the
        # caller.
        result = run_menu("")
        output = result.stdout + result.stderr
        self.assertNotIn("INSTALLED", output)
        self.assertIn("rc=2", result.stdout)
        self.assertIn("nothing was installed", output)

    def test_a_choice_outside_the_list_installs_nothing(self) -> None:
        result = run_menu("0\n")
        output = result.stdout + result.stderr
        self.assertNotIn("INSTALLED", output)
        self.assertIn("not a choice", output)


class StagingPathTests(unittest.TestCase):
    """Downloads stage under one absolute root, whatever the caller's cwd is."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.script = SCRIPT.read_text()

    def test_the_root_is_derived_absolutely_from_the_script_path(self) -> None:
        self.assertIn('ROOT="$(cd "$(dirname "$0")/.." && pwd)"', self.script)

    def test_the_work_root_defaults_under_the_install_root(self) -> None:
        self.assertIn('WORK="${TINYTITAN_WORK_DIR:-$ROOT/.build}"', self.script)

    def test_no_staging_path_is_relative_to_the_working_directory(self) -> None:
        # Comments may name `.build/release` (the repack binary default) or the
        # old shape; what matters is that no *code* argument or variable is a
        # bare relative `.build/...` path any more.
        for number, line in enumerate(self.script.splitlines(), start=1):
            if line.lstrip().startswith("#") or ".build/release" in line:
                continue
            with self.subTest(line=number):
                self.assertNotIn(".build/", line)

    def test_download_and_conversion_arguments_use_the_work_root(self) -> None:
        for argument in (
            '--work "$WORK/${preset}-shards"',
            '--output "$WORK/qwen38-affine-${width}bit"',
            '--work "$WORK/qwen38-shards"',
            '--output "$WORK/${preset}-affine"',
        ):
            with self.subTest(argument=argument):
                self.assertIn(argument, self.script)

    def test_relative_staging_paths_are_not_quoted_in_a_command(self) -> None:
        # Belt and braces for a future edit that reintroduces the old spelling.
        self.assertIsNone(re.search(r'"\.build/(?!release)', self.script))


class DiskSpaceAndCleanupTests(unittest.TestCase):
    """A download starts only if it can finish; staging leaves when it cannot help.

    Everything runs against temp volumes with a stubbed `df`, a stubbed repack
    binary and `/usr/bin/true` as the interpreter, so no test here can fetch a
    byte. The stub repack prints `FAKE-REPACK …`, which is how a test sees that
    an install would have started.
    """

    ORNITH_4 = "ornith-1.5_35B_A3B_4Bit"
    ORNITH_8 = "ornith-1.5_35B_A3B_8Bit"

    def setUp(self) -> None:
        self.root = pathlib.Path(tempfile.mkdtemp(prefix="tt-install-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.work = self.root / "work"
        self.models = self.root / "models"
        self.bin = self.root / "bin"
        for path in (self.work, self.models, self.bin):
            path.mkdir(parents=True)
        repack = self.bin / "TinyTitanRepack"
        repack.write_text('#!/bin/sh\necho "FAKE-REPACK $*"\n')
        repack.chmod(0o755)
        self.df_dir = self.root / "dfbin"
        self.df_dir.mkdir()

    def stub_df(self, available_gb: int) -> None:
        df = self.df_dir / "df"
        df.write_text(
            f'#!/bin/sh\necho "Filesystem 1G-blocks Used Available Capacity Mounted on"\n'
            f'echo "/dev/test 10000 1 {available_gb} 1% /"\n'
        )
        df.chmod(0o755)

    def run_tool(self, *args: str, available_gb: int | None = None):
        env = dict(
            os.environ,
            TINYTITAN_WORK_DIR=str(self.work),
            TINYTITAN_MODELS_DIR=str(self.models),
            TINYTITAN_BIN_DIR=str(self.bin),
            TINYTITAN_PYTHON="/usr/bin/true",
        )
        if available_gb is not None:
            self.stub_df(available_gb)
            env["PATH"] = f"{self.df_dir}:{env['PATH']}"
        return subprocess.run(
            ["bash", str(SCRIPT), *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    def install(self, name: str) -> None:
        (self.models / name).mkdir(parents=True, exist_ok=True)

    def stage(self, relative: str, megabytes: int) -> pathlib.Path:
        path = self.work / relative
        path.mkdir(parents=True, exist_ok=True)
        (path / "blob").write_bytes(b"x" * (megabytes * 1024))
        return path

    def test_a_short_volume_refuses_before_the_download_starts(self) -> None:
        result = self.run_tool("ornith15", available_gb=3)
        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("but about", output)
        self.assertIn("Not starting ornith15", output)
        self.assertIn("TINYTITAN_WORK_DIR", output)
        self.assertNotIn("FAKE-REPACK", output)
        self.assertFalse((self.models / self.ORNITH_4).exists())

    def test_enough_space_lets_the_install_proceed(self) -> None:
        result = self.run_tool("ornith15", available_gb=9999)
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertIn("installing ornith15", output)
        self.assertIn("FAKE-REPACK", output)

    def test_the_precheck_can_be_skipped_explicitly(self) -> None:
        env = dict(
            os.environ,
            TINYTITAN_WORK_DIR=str(self.work),
            TINYTITAN_MODELS_DIR=str(self.models),
            TINYTITAN_BIN_DIR=str(self.bin),
            TINYTITAN_PYTHON="/usr/bin/true",
            TINYTITAN_SKIP_DISK_CHECK="1",
        )
        self.stub_df(3)
        env["PATH"] = f"{self.df_dir}:{env['PATH']}"
        result = subprocess.run(
            ["bash", str(SCRIPT), "ornith15"],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        self.assertIn("FAKE-REPACK", result.stdout + result.stderr)

    def test_staging_for_a_missing_width_is_kept_and_explained(self) -> None:
        self.install(self.ORNITH_4)
        snapshot = self.stage("ornith15-affine-4bit", 1)
        result = self.run_tool("ornith15")
        output = result.stdout + result.stderr
        self.assertTrue(snapshot.exists(), "the reusable snapshot was removed")
        self.assertIn("kept", output)
        self.assertIn("clean", output)

    def test_staging_goes_once_both_widths_are_installed(self) -> None:
        for name in (self.ORNITH_4, self.ORNITH_8):
            self.install(name)
        snapshot = self.stage("ornith15-affine-4bit", 1)
        result = self.run_tool("ornith15")
        self.assertFalse(snapshot.exists())
        self.assertIn("staging cleaned", result.stdout + result.stderr)

    def test_draft_head_staging_is_removed_immediately(self) -> None:
        # Nothing reuses a draft head's source shard, installed or not.
        self.install("ornith-1.5_35B_A3B_MTP_4Bit")
        source = self.stage("ornith-mtp-src", 1)
        result = self.run_tool("ornith15-mtp")
        self.assertFalse(source.exists())
        self.assertIn("staging cleaned", result.stdout + result.stderr)

    def test_clean_keeps_staging_an_uninstalled_width_still_needs(self) -> None:
        self.install(self.ORNITH_4)
        snapshot = self.stage("ornith15-affine-4bit", 1)
        result = self.run_tool("clean")
        self.assertTrue(snapshot.exists())
        self.assertIn("still staged", result.stdout + result.stderr)

        self.install(self.ORNITH_8)
        result = self.run_tool("clean")
        self.assertFalse(snapshot.exists())
        self.assertIn("nothing left staged", result.stdout + result.stderr)

    def test_status_shows_what_is_staged(self) -> None:
        self.stage("ornith15-affine-4bit", 1)
        result = self.run_tool()
        self.assertIn("staging:", result.stdout)


if __name__ == "__main__":
    unittest.main()
