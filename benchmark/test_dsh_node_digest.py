"""The Node that `tools/dsh_local.sh` fetches is proven before it is unpacked.

A Mac with no Node gets one installed into the private root, and the harness runs
on it seconds later. Until 2026-10-06 the tarball was downloaded and handed
straight to `tar -xzf`, with no digest consulted: a mirror or a proxy that
changed the bytes would have produced a working-looking `node` that was not the
one nodejs.org published. This is the same defect `verify_release_artifact` was
fixed for on the engine and tools archives (AUD-109), on the one surface that was
left out.

What is pinned here is the fail-closed shape, not the happy path alone: every
branch that cannot prove the bytes must stop the setup, must say *which* cause it
was — this machine has no `shasum`, nodejs.org published nothing, the published
list does not name this file, the bytes do not match the line that does — and must
leave the private root empty. The name-matching case matters as much as the
mismatch case: SHASUMS256.txt lists the whole release, so a check that takes any
line is not a check on this tarball.

Nothing is fetched and nothing is installed for real. `curl` is a stub serving a
fixture directory, the tarball is a synthetic one, the digests are computed here,
and the PATH is a farm of symlinks to the system tools so that the run sees no
node and no npm whatever the machine testing it has.

    cd benchmark && python3 -m unittest test_dsh_node_digest -v
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
DSH_LOCAL = ROOT / "tools/dsh_local.sh"

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


def pinned_node_version() -> str:
    """The Node version `tools/dsh_local.sh` fetches, read from the script.

    The digest check is per-version — the file consulted is the one published for
    exactly that pin — so spelling the version out again here would test a tarball
    the script never asks for.
    """
    match = re.search(
        r'NODE_VERSION="\$\{TINYTITAN_DSH_NODE_VERSION:-([^}]+)\}"',
        DSH_LOCAL.read_text(),
    )
    if match is None:
        raise AssertionError(
            "tools/dsh_local.sh no longer pins Node with a "
            "TINYTITAN_DSH_NODE_VERSION default, so this harness has no version "
            "to build a published digest for"
        )
    return match.group(1)


NODE_VERSION = pinned_node_version()
TARBALL = f"node-v{NODE_VERSION}-darwin-arm64.tar.gz"
# The other artifacts of the same release, named here so the fixture looks like
# the real file rather than like a test: one of these in particular is the line a
# check that does not match on the filename would take.
DECOY = f"node-v{NODE_VERSION}-darwin-x64.tar.gz"
SOURCE_DECOY = f"node-v{NODE_VERSION}.tar.gz"

# A farm of symlinks to the system tools, built once and shared: the whole point
# is that the run sees a plain macOS install, so a development machine's Homebrew
# node, npm or curl cannot decide which branch of install_node is exercised.
FARM_DIRECTORIES = ("/usr/bin", "/bin", "/usr/sbin")
MODULE_ROOT: pathlib.Path | None = None
FARM: pathlib.Path | None = None
FARM_WITHOUT_SHASUM: pathlib.Path | None = None


def build_farm(directory: pathlib.Path, *, without_checksum_tool: bool = False) -> pathlib.Path:
    directory.mkdir(parents=True)
    for system in FARM_DIRECTORIES:
        for entry in pathlib.Path(system).iterdir():
            link = directory / entry.name
            if not link.exists():
                os.symlink(entry, link)
    if without_checksum_tool:
        for removed in list(directory.glob("shasum*")) + [directory / "sha256sum"]:
            if removed.exists():
                removed.unlink()
    return directory


def setUpModule() -> None:
    global MODULE_ROOT, FARM, FARM_WITHOUT_SHASUM
    MODULE_ROOT = pathlib.Path(tempfile.mkdtemp(prefix="tt-dsh-node-"))
    FARM = build_farm(MODULE_ROOT / "farm")
    FARM_WITHOUT_SHASUM = build_farm(MODULE_ROOT / "farm-no-shasum", without_checksum_tool=True)
    for name in ("shasum", "sha256sum"):
        if (FARM_WITHOUT_SHASUM / name).exists():
            raise AssertionError(
                f"the farm still carries {name}, so the missing-tool case tests nothing"
            )


def tearDownModule() -> None:
    if MODULE_ROOT is not None and MODULE_ROOT.exists():
        shutil.rmtree(MODULE_ROOT, ignore_errors=True)


def digest_of(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def node_tarball_bytes() -> bytes:
    """A synthetic darwin-arm64 Node: just enough to unpack and be executable.

    `bin/node` is a script that answers `--version`, because install_node checks
    the file is there and the caller's next line reports it. `bin/npm` is a stub
    that announces itself and fails: the accept case then proves it got past the
    digest by reaching the harness install, rather than by exiting zero for some
    reason that has nothing to do with the check.
    """
    buffer = io.BytesIO()
    prefix = TARBALL.removesuffix(".tar.gz")
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, mode, payload in (
            ("bin/node", 0o755, f'#!/bin/sh\necho "v{NODE_VERSION}"\n'.encode()),
            ("bin/npm", 0o755, b'#!/bin/sh\necho "stub-npm-invoked"\nexit 1\n'),
            ("bin/corepack", 0o755, b"#!/bin/sh\nexit 0\n"),
            ("README.md", 0o644, b"synthetic node\n"),
        ):
            info = tarfile.TarInfo(f"{prefix}/{name}")
            info.size = len(payload)
            info.mode = mode
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def shasums_text(lines: list[tuple[str, str]]) -> str:
    """A SHASUMS256.txt-shaped file: `<hash>  <name>`, one entry per artifact."""
    return "".join(f"{digest}  {name}\n" for digest, name in lines)


class DshRun:
    """One `ensure`, against a fixture the stub `curl` serves."""

    def __init__(self, name: str, *, farm: pathlib.Path) -> None:
        if MODULE_ROOT is None:
            raise RuntimeError("setUpModule did not build the shared fixture root")
        self.root = MODULE_ROOT / name
        self.home = self.root / "home"
        self.dsh_root = self.root / "dsh"
        self.fixture = self.root / "fixture"
        self.stub = self.root / "stub"
        self.requests = self.root / "requests.log"
        for directory in (self.home, self.dsh_root, self.fixture, self.stub):
            directory.mkdir(parents=True)
        self.requests.write_text("")
        self.result: subprocess.CompletedProcess[str] | None = None
        script = self.stub / "curl"
        script.write_text(CURL_STUB)
        script.chmod(0o755)
        self.farm = farm

    # --- what the fake nodejs.org serves ------------------------------------

    def serve_tarball(self) -> bytes:
        payload = node_tarball_bytes()
        (self.fixture / TARBALL).write_bytes(payload)
        return payload

    def serve_checksums(self, *, arm64_digest: str, with_arm64: bool = True) -> None:
        # The decoys come first, ours last: a check that took the first line of
        # the file, or handed the whole file to `shasum -c`, fails here instead of
        # passing on our digest by luck.
        lines = [("f" * 64, DECOY), ("0" * 64, SOURCE_DECOY)]
        if with_arm64:
            lines.append((arm64_digest, TARBALL))
        (self.fixture / "SHASUMS256.txt").write_text(shasums_text(lines))

    def tamper(self) -> None:
        """Change the served bytes, leaving the published digest as it was."""
        path = self.fixture / TARBALL
        path.write_bytes(path.read_bytes() + b"tampered")

    # --- running it ---------------------------------------------------------

    def ensure(self, *, dry_run: bool = False) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env["HOME"] = str(self.home)
        env["PATH"] = f"{self.stub}:{self.farm}"
        env["TT_FIXTURE"] = str(self.fixture)
        env["TT_REQUESTS"] = str(self.requests)
        env["TINYTITAN_DSH_ROOT"] = str(self.dsh_root)
        # The farm has no node, so nothing here can quietly take the
        # "already on this Mac" early return instead of the fetch path.
        for variable in ("TINYTITAN_DSH_NODE_VERSION", "TINYTITAN_DSH_DRY_RUN"):
            env.pop(variable, None)
        if dry_run:
            env["TINYTITAN_DSH_DRY_RUN"] = "1"
        self.result = subprocess.run(
            ["/bin/bash", str(DSH_LOCAL), "ensure"],
            input="",
            text=True,
            capture_output=True,
            check=False,
            env=env,
            timeout=180,
        )
        return self.result

    @property
    def urls(self) -> list[str]:
        return self.requests.read_text().split()

    @property
    def output(self) -> str:
        return f"{self.result.stdout}{self.result.stderr}"

    @property
    def installed_node(self) -> pathlib.Path:
        return self.dsh_root / "node" / "bin" / "node"


class RefusalTestCase(unittest.TestCase):
    """Assertions every 'this must stop the setup' case shares."""

    env: DshRun

    def assert_refused(self, phrase: str) -> None:
        self.assertNotEqual(self.env.result.returncode, 0, self.env.output)
        self.assertIn(phrase, self.env.output)
        # The promise is that nothing was unpacked, not merely that a message was
        # printed: a guard that runs after `tar -xzf` would satisfy the message
        # and still leave the unproven bytes on disk.
        self.assertFalse(self.env.installed_node.exists(), self.env.output)
        self.assertFalse((self.env.dsh_root / "node").exists(), self.env.output)


class VerifiedNodeDownloadTests(unittest.TestCase):
    """The published digest names this tarball and the bytes match it."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.env = DshRun("ok", farm=FARM)
        payload = cls.env.serve_tarball()
        cls.env.serve_checksums(arm64_digest=digest_of(payload))
        cls.result = cls.env.ensure()

    def test_the_digest_is_fetched_for_the_pinned_version(self) -> None:
        self.assertIn(f"https://nodejs.org/dist/v{NODE_VERSION}/SHASUMS256.txt", self.env.urls)
        self.assertIn(f"https://nodejs.org/dist/v{NODE_VERSION}/{TARBALL}", self.env.urls)

    def test_the_tarball_is_unpacked_once_the_digest_matches(self) -> None:
        # The check passing must be visible as progress, not as silence: the run
        # gets all the way to the harness install, and the stub npm in the
        # synthetic tarball is what shows it. A guard that died early would never
        # reach that invocation.
        self.assertIn("verified against nodejs.org", self.env.output)
        self.assertTrue(self.env.installed_node.exists(), self.env.output)
        self.assertIn("stub-npm-invoked", self.env.output)

    def test_other_artifacts_in_the_same_checksum_file_are_not_consulted(self) -> None:
        # The fixture puts two entries with impossible digests *after* ours and
        # the arm64 line last among the darwin builds. A check handed the whole
        # file would fail on those missing files; a check that took the first line
        # would fail on the wrong digest. Either way the install would stop here.
        self.assertNotIn("does not match", self.env.output)
        self.assertTrue(self.env.installed_node.exists())


class TamperedTarballTests(RefusalTestCase):
    """The digest matches the published line, but the bytes are not those bytes."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.env = DshRun("tampered", farm=FARM)
        payload = cls.env.serve_tarball()
        cls.env.serve_checksums(arm64_digest=digest_of(payload))
        cls.env.tamper()
        cls.result = cls.env.ensure()

    def test_the_install_stops_and_names_the_mismatch(self) -> None:
        self.assert_refused("does not match the checksum")

    def test_nothing_is_unpacked(self) -> None:
        self.assertNotIn("Private Node ready", self.env.output)


class MissingChecksumFileTests(RefusalTestCase):
    """nodejs.org answers for the tarball and nothing for its checksums."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.env = DshRun("no-shasums", farm=FARM)
        cls.env.serve_tarball()
        cls.result = cls.env.ensure()

    def test_the_install_stops_and_names_the_absent_checksums(self) -> None:
        # The distinction matters to whoever reads it: a 404 on the checksum file
        # is "there is nothing to check against", not "these bytes are wrong".
        self.assert_refused("published no checksums")
        self.assertIn("TINYTITAN_DSH_NODE_VERSION", self.env.output)


class UnlistedTarballTests(RefusalTestCase):
    """Checksums are published, and none of them is for the darwin-arm64 build."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.env = DshRun("unlisted", farm=FARM)
        payload = cls.env.serve_tarball()
        cls.env.serve_checksums(arm64_digest=digest_of(payload), with_arm64=False)
        cls.result = cls.env.ensure()

    def test_the_install_stops_naming_the_file_that_is_missing(self) -> None:
        # SHASUMS256.txt is non-empty and well-formed here, so a check that did
        # not match on the filename would either take a decoy line and report a
        # mismatch — the wrong cause — or find nothing at all.
        self.assert_refused(f"do not name {TARBALL}")

    def test_the_mismatch_message_is_not_used(self) -> None:
        self.assertNotIn("does not match", self.env.output)


class MissingShasumToolTests(RefusalTestCase):
    """A Mac whose PATH carries every system tool except the checksum tool."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.env = DshRun("no-shasum", farm=FARM_WITHOUT_SHASUM)
        payload = cls.env.serve_tarball()
        cls.env.serve_checksums(arm64_digest=digest_of(payload))
        cls.result = cls.env.ensure()

    def test_the_install_refuses_rather_than_installing_unverified_bytes(self) -> None:
        self.assert_refused("shasum is missing")

    def test_it_offers_the_route_that_actually_works(self) -> None:
        # An existing node is used in preference to a fetch, so installing Node by
        # another route is the fix — not a flag that skips the check.
        self.assertIn("install Node yourself", self.env.output)


class DryRunTests(unittest.TestCase):
    """The plan shows both fetches and writes nothing."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.env = DshRun("dry", farm=FARM)
        payload = cls.env.serve_tarball()
        cls.env.serve_checksums(arm64_digest=digest_of(payload))
        cls.result = cls.env.ensure(dry_run=True)

    def test_both_fetches_are_in_the_transcript(self) -> None:
        self.assertIn("would download Node from nodejs.org", self.env.output)
        self.assertIn("would fetch nodejs.org's published checksums", self.env.output)

    def test_the_dry_run_fetches_nothing_and_writes_no_node(self) -> None:
        self.assertEqual(self.env.urls, [])
        self.assertFalse((self.env.dsh_root / "node").exists(), self.env.output)
        self.assertEqual(
            sorted(path.name for path in self.env.dsh_root.iterdir()),
            [],
            self.env.output,
        )


if __name__ == "__main__":
    unittest.main()
