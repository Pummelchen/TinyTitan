"""The release install verifies *both* artifacts it then runs.

`tools/install_tinytitan.sh` downloads the engine binaries and a tools tree, and
then executes the launcher out of that tools tree. Until 2026-10-06 the engine
tarball was checksummed only if the digest download happened to succeed — a 404
was a warning — and the tools archive was never verified at all, fetched from a
source-archive URL rather than the release asset. Both are fixed here, and this
file is what keeps them fixed: every branch that cannot prove the bytes must stop
the install, and the name the installer asks for must be the name
`tools/release.sh` publishes.

Nothing is fetched. `curl` is a stub that serves files out of a fixture directory
named in `TT_FIXTURE`, so a missing digest is a missing file and a tampered
artifact is a file whose bytes no longer match its published `.sha256`. The
digests themselves are real SHA-256 sums and the real `shasum` does the checking,
so these tests exercise the verification rather than a picture of it. Each run
happens under a temporary `HOME` and `TINYTITAN_ROOT`, with `--no-model --no-web`,
and `df` is stubbed so the disk question is never asked.

    cd benchmark && python3 -m unittest test_release_installer_verification -v
"""

from __future__ import annotations

import hashlib
import io
import os
import pathlib
import re
import shutil
import subprocess
import tarfile
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "tools/install_tinytitan.sh"
RELEASE = ROOT / "tools/release.sh"

VERSION = "9.9.9"
TAG = f"v{VERSION}"
ENGINE_ASSET = f"tinytitan-{VERSION}-macos-arm64.tar.gz"
DOWNLOAD_BASE = f"https://github.com/Pummelchen/TinyTitan/releases/download/{TAG}"

CURL_STUB = """#!/usr/bin/env bash
set -u
out=""
url=""
while [ $# -gt 0 ]; do
  case "$1" in
    -o) shift; out="${1:-}" ;;
    -o*) out="${1#-o}" ;;
    -*) ;;
    *) url="$1" ;;
  esac
  shift
done
name="${url##*/}"
printf '%s\\n' "$url" >> "$TT_REQUESTS"
if [ ! -e "$TT_FIXTURE/$name" ]; then
  echo "stub-curl: nothing served for $url" >&2
  exit 22
fi
if [ -n "$out" ]; then
  cp "$TT_FIXTURE/$name" "$out"
else
  cat "$TT_FIXTURE/$name"
fi
"""

# A volume with plenty of room, so the installer's disk question is never asked
# and no test depends on the free space of the machine running it.
DF_STUB = """#!/usr/bin/env bash
echo "Filesystem 1024-blocks Used Available Capacity Mounted on"
echo "/dev/disk1s2 1000000000 400000000 560000000 42% /System/Volumes/Data"
"""


def sha256_of(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_file(archive: pathlib.Path) -> str:
    """One line in the exact `shasum -a 256` output a release publishes."""
    return f"{sha256_of(archive)}  {archive.name}\n"


def tar_gz(target: pathlib.Path, prefix: str, files: dict[str, tuple[int, bytes]]) -> None:
    with tarfile.open(target, "w:gz") as archive:
        for name, (mode, payload) in sorted(files.items()):
            info = tarfile.TarInfo(f"{prefix}/{name}")
            info.size = len(payload)
            info.mode = mode
            archive.addfile(info, io.BytesIO(payload))


def release_tools_asset() -> str:
    """The tools asset name `tools/release.sh` stages, read from its own source.

    The installer and the release script build this name in two places, and a
    rename in either one breaks every install from the next tag. Taking it from
    the release script here, rather than spelling it out again, is what turns
    that drift into a failing test instead of a 404 on someone's Mac.
    """
    if 'TOOLS_STAGE_PREFIX="tinytitan-$VERSION-tools"' not in RELEASE.read_text():
        raise AssertionError(
            "tools/release.sh no longer stages a tinytitan-$VERSION-tools archive, "
            "so the installer's tools_asset has nothing to check against"
        )
    return f"tinytitan-{VERSION}-tools.tar.gz"


class InstallerRun:
    """One release install, against a fixture directory the stub `curl` serves."""

    def __init__(self, home: pathlib.Path, *, with_farm: bool = False) -> None:
        self.root = home
        self.home = home / "home"
        self.install_root = home / "tinytitan"
        self.fixture = home / "fixture"
        self.stub = home / "stub"
        self.requests = home / "requests.log"
        for directory in (self.home, self.install_root, self.fixture, self.stub):
            directory.mkdir(parents=True)
        self.requests.write_text("")
        for name, body in (("curl", CURL_STUB), ("df", DF_STUB)):
            script = self.stub / name
            script.write_text(body)
            script.chmod(0o755)
        # A PATH with every system tool *except* the checksum tool, for the case
        # where there is nothing to verify against.
        self.farm = home / "farm"
        if with_farm:
            self.farm.mkdir()
            for directory in ("/usr/bin", "/bin", "/usr/sbin"):
                for entry in pathlib.Path(directory).iterdir():
                    link = self.farm / entry.name
                    if not link.exists():
                        os.symlink(entry, link)
            for removed in list(self.farm.glob("shasum*")) + [self.farm / "sha256sum"]:
                if removed.exists():
                    removed.unlink()
        self.bin_path = self.install_root / "bin"
        self.src_path = self.install_root / "src"

    def serve(self, name: str, body: bytes) -> None:
        (self.fixture / name).write_bytes(body)

    def serve_release(
        self, *, with_engine_digest: bool = True, with_tools_digest: bool = True
    ) -> None:
        """The two archives a release carries, plus their published checksums."""
        engine = self.fixture / ENGINE_ASSET
        tar_gz(
            engine,
            f"tinytitan-{VERSION}-macos-arm64",
            {
                "TinyTitanServer": (0o755, b"#!/bin/sh\necho fake-server\n"),
                "TinyTitanCLI": (0o755, b"#!/bin/sh\necho fake-cli\n"),
                "LICENSE": (0o644, b"Apache 2.0\n"),
            },
        )
        if with_engine_digest:
            (self.fixture / f"{ENGINE_ASSET}.sha256").write_text(digest_file(engine))

        tools_name = release_tools_asset()
        tools = self.fixture / tools_name
        tar_gz(
            tools,
            f"tinytitan-{VERSION}-tools",
            {
                "tools/install_tinytitan.sh": (0o755, INSTALLER.read_bytes()),
                "tools/server_launcher.sh": (0o755, b"#!/bin/sh\n# launcher\n"),
                "tools/install_models.sh": (0o755, b"#!/bin/sh\n# models\n"),
                "Package.swift": (0o644, b"// swift-tools-version\n"),
            },
        )
        if with_tools_digest:
            (self.fixture / f"{tools_name}.sha256").write_text(digest_file(tools))

    def tamper(self, name: str) -> None:
        """Change the served bytes, leaving the published checksum as it was."""
        path = self.fixture / name
        path.write_bytes(path.read_bytes() + b"tampered")

    def run(self) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env["HOME"] = str(self.home)
        env["TINYTITAN_ROOT"] = str(self.install_root)
        env["TT_FIXTURE"] = str(self.fixture)
        env["TT_REQUESTS"] = str(self.requests)
        # With a farm, the system directories are off PATH entirely: the point of
        # that run is that no `shasum` can be found anywhere.
        if self.farm.exists():
            env["PATH"] = f"{self.stub}:{self.farm}"
        else:
            env["PATH"] = f"{self.stub}:{env['PATH']}"
        return subprocess.run(
            [
                "bash",
                str(INSTALLER),
                "--yes",
                "--no-model",
                "--no-web",
                "--version",
                TAG,
            ],
            input="",
            text=True,
            capture_output=True,
            check=False,
            env=env,
            timeout=180,
        )

    @property
    def urls(self) -> list[str]:
        return self.requests.read_text().split()


class VerifiedInstallTests(unittest.TestCase):
    """The path where both checksums exist and both match."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.home = pathlib.Path(tempfile.mkdtemp(prefix="tt-install-ok-"))
        cls.env = InstallerRun(cls.home)
        cls.env.serve_release()
        cls.result = cls.env.run()
        cls.output = cls.result.stdout + cls.result.stderr

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.home, ignore_errors=True)

    def test_the_release_installs(self) -> None:
        self.assertEqual(self.result.returncode, 0, self.output)

    def test_both_artifacts_say_their_checksum_was_verified(self) -> None:
        self.assertIn("engine checksum verified", self.output)
        self.assertIn("tools checksum verified", self.output)

    def test_the_engine_and_the_tools_land_in_the_install_root(self) -> None:
        self.assertTrue((self.env.bin_path / "TinyTitanServer").is_file())
        self.assertTrue((self.env.src_path / "tools/server_launcher.sh").is_file())

    def test_the_tools_come_from_the_release_asset_not_a_source_archive(self) -> None:
        self.assertIn(f"{DOWNLOAD_BASE}/{release_tools_asset()}", self.env.urls)
        self.assertNotIn("archive/refs/tags", " ".join(self.env.urls))

    def test_every_downloaded_archive_has_its_digest_fetched(self) -> None:
        for asset in (ENGINE_ASSET, release_tools_asset()):
            with self.subTest(asset=asset):
                self.assertIn(f"{DOWNLOAD_BASE}/{asset}.sha256", self.env.urls)


class EngineDigestTests(unittest.TestCase):
    """A release the engine cannot be checked against is not installed."""

    def setUp(self) -> None:
        home = pathlib.Path(tempfile.mkdtemp(prefix="tt-install-engine-"))
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        self.env = InstallerRun(home)

    def test_a_missing_engine_checksum_stops_the_install(self) -> None:
        self.env.serve_release(with_engine_digest=False)
        result = self.env.run()
        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0, output)
        self.assertIn("No checksum published for the engine download", output)
        self.assertFalse((self.env.bin_path / "TinyTitanServer").exists())

    def test_a_tampered_engine_stops_the_install(self) -> None:
        self.env.serve_release()
        self.env.tamper(ENGINE_ASSET)
        result = self.env.run()
        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0, output)
        self.assertIn("engine download does not match its published checksum", output)
        self.assertFalse((self.env.bin_path / "TinyTitanServer").exists())


class ToolsDigestTests(unittest.TestCase):
    """The tree the installer goes on to execute is verified like the engine."""

    def setUp(self) -> None:
        home = pathlib.Path(tempfile.mkdtemp(prefix="tt-install-tools-"))
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        self.env = InstallerRun(home)

    def test_a_missing_tools_checksum_stops_the_install(self) -> None:
        self.env.serve_release(with_tools_digest=False)
        result = self.env.run()
        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0, output)
        self.assertIn("No checksum published for the tools download", output)
        self.assertFalse((self.env.src_path / "tools/server_launcher.sh").exists())

    def test_a_tampered_tools_tree_stops_the_install(self) -> None:
        self.env.serve_release()
        self.env.tamper(release_tools_asset())
        result = self.env.run()
        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0, output)
        self.assertIn("tools download does not match its published checksum", output)
        self.assertFalse((self.env.src_path / "tools/server_launcher.sh").exists())


class MissingShasumTests(unittest.TestCase):
    """No checksum tool means no install, not an unchecked one.

    `shasum` ships with macOS, so this is a corner rather than the road. It is
    still the corner that matters: with a softer guard, a machine missing the tool
    would either fall into the mismatch branch and blame the download, or install
    bytes nobody verified.
    """

    def test_the_install_refuses_when_no_checksum_tool_is_available(self) -> None:
        home = pathlib.Path(tempfile.mkdtemp(prefix="tt-install-noshasum-"))
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        env = InstallerRun(home, with_farm=True)
        env.serve_release()
        result = env.run()
        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0, output)
        self.assertIn("shasum is missing", output)
        self.assertFalse((env.bin_path / "TinyTitanServer").exists())


class ReleaseSideTests(unittest.TestCase):
    """What `tools/release.sh` must keep publishing for those checks to work.

    The release script cannot run inside a test suite — it needs a tag, a clean
    tree and the full gates — so these pin the parts the installer depends on by
    name, including the digest-substitution order that once published a hash
    belonging to the wrong archive.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.script = RELEASE.read_text()

    def test_the_tools_archive_is_checksummed_like_the_other_two(self) -> None:
        self.assertIn(
            'shasum -a 256 "$TOOLS_ARCHIVE" | sed "s|$STAGE_ROOT/||" > "$TOOLS_ARCHIVE.sha256"',
            self.script,
        )

    def test_both_files_are_uploaded_with_the_release(self) -> None:
        create = re.search(r"gh release create.*?--latest", self.script, re.DOTALL)
        self.assertIsNotNone(create, "gh release create no longer found")
        for asset in ("$TOOLS_ARCHIVE", "$TOOLS_ARCHIVE.sha256"):
            with self.subTest(asset=asset):
                self.assertIn(asset, create.group(0))

    def test_the_notes_must_quote_the_tools_digest_and_size(self) -> None:
        for token in ("TOOLS_SHA256_PENDING", "TOOLS_BYTES_PENDING"):
            with self.subTest(token=token):
                self.assertIn(token, self.script)
        self.assertIn("nor quote the tools archive's sha256", self.script)

    def test_the_specific_digest_tokens_are_substituted_before_the_engine_one(self) -> None:
        # `SHA256_PENDING` is a substring of both `LIBRARY_SHA256_PENDING` and
        # `TOOLS_SHA256_PENDING`, so replacing the engine's token first rewrites
        # the other two names and their rules match nothing.
        start = self.script.index("sed -e ")
        block = self.script[start : self.script.index('"$RENDERED_NOTES"', start)]
        self.assertIn("sed", block)
        for specific in ("LIBRARY_SHA256_PENDING", "TOOLS_SHA256_PENDING"):
            with self.subTest(token=specific):
                self.assertLess(block.index(f"s/{specific}/"), block.index("s/SHA256_PENDING/"))
        for specific in ("LIBRARY_BYTES_PENDING", "TOOLS_BYTES_PENDING"):
            with self.subTest(token=specific):
                self.assertLess(
                    block.index(f"s/{specific}/"), block.index("s/ARCHIVE_BYTES_PENDING/")
                )

    def test_the_tools_digest_survives_compaction(self) -> None:
        # Every string the notes greps rely on is handed to the compactor as
        # --require, so a compaction that would drop the tools digest fails
        # before the Release exists rather than publishing one that lacks it.
        require = re.search(r"for required in (.*?); do", self.script, re.DOTALL)
        self.assertIsNotNone(require, "the --require loop changed shape")
        for variable in ("$TOOLS_SHA", "$TOOLS_BYTES"):
            with self.subTest(variable=variable):
                self.assertIn(variable, require.group(1))


class InstallerScriptTests(unittest.TestCase):
    """The fail-closed shape, read from the script the stubbed runs exercise."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.script = INSTALLER.read_text()

    def test_no_verification_branch_falls_through_to_an_unchecked_install(self) -> None:
        body = re.search(r"verify_release_artifact\(\) \{(.*?)\n\}", self.script, re.DOTALL)
        self.assertIsNotNone(body, "verify_release_artifact no longer found")
        block = body.group(1)
        self.assertEqual(block.count("die "), 3, "a branch was added or removed")
        self.assertEqual(block.count('rm -rf "$dir"'), 3)
        self.assertNotIn("warn ", block, "a verification failure must not be a warning")

    def test_both_downloads_go_through_it(self) -> None:
        self.assertIn('verify_release_artifact "$tmp" "$asset"', self.script)
        self.assertIn('verify_release_artifact "$tmp" "$tools_asset"', self.script)

    def test_neither_tree_is_unpacked_before_its_checksum_is_verified(self) -> None:
        for call, unpack in (
            ('verify_release_artifact "$tmp" "$asset"', 'tar -xzf "$tmp/$asset"'),
            ('verify_release_artifact "$tmp" "$tools_asset"', 'tar -xzf "$tmp/$tools_asset"'),
        ):
            with self.subTest(call=call):
                self.assertLess(self.script.index(call), self.script.index(unpack))


if __name__ == "__main__":
    unittest.main()
