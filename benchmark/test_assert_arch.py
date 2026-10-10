"""AUD-265: the arm64 assertion must not pass over an artifact it could not read.

`tools/assert-arch.sh` is RELEASE.md rule 2 made executable: every Mach-O in a
released archive reports exactly `arm64` from `lipo -archs`, and a fat or x86_64
binary is a release defect rather than a build option. Its own docstring promises
more than that -- "candidates come from the file's magic number, not from a list of
expected names, so an artifact nobody thought to assert on is still checked" -- and
the read it uses to keep that promise cannot fail:

    magic="$(od -An -tx1 -N4 "$f" 2>/dev/null | tr -d ' \\n')"

The status of that assignment is `tr`'s, never `od`'s, so a file the gate is not
allowed to open yields an empty magic and takes the same `continue` as a README
does. Measured on this host: a directory holding one real arm64 binary and a
second binary chmodded to 000 exits **0** and prints "1 Mach-O artifact(s), each
exactly arm64" -- a claim about every artifact in a set the gate read only half
of. A 1-3 byte truncation of a Mach-O header is skipped the same way, which is
what a `cp` that filled the disk mid-write actually leaves behind, and this
repository runs with 163 GB of models on the volume.

So the two shapes the gate has to tell apart are "this is not an artifact" (skip,
and the docstring's `.swiftmodule` case is exactly why) and "I could not read this
artifact" (refuse). The tests below put each of them next to a real arm64 binary so
a refusal can only come from its own branch and not from the vacuity guard, and
pin the branches the fix must not disturb: the fat and x86_64 refusals, the
non-artifact skip, the empty directory, the archive, and the fact that `tt_die`
`exit`s the script that *sources* it -- release.sh and build_library.sh both run
without `set -e`, so that exit is the only thing that stops either.

No model, no network, no build: the arm64 fixture is taken from the toolchain
already on the runner and the fat fixture is `/bin/ls`, which on Apple Silicon
reports three arches. If no exactly-arm64 Mach-O is found, the suite fails rather
than skips -- a gate proven on nothing is the defect this file is about.

Run from `benchmark/`:

    python3 -m unittest test_assert_arch
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
GATE = ROOT / "tools" / "assert-arch.sh"
ARM64_CANDIDATES = (
    "/Applications/Xcode.app/Contents/Developer/usr/bin/xcodebuild",
    "/Applications/Xcode.app/Contents/Developer/usr/bin/instruments",
)


def find_arm64_binary() -> pathlib.Path:
    """A real Mach-O that `lipo -archs` reports as exactly `arm64`."""
    for candidate in ARM64_CANDIDATES:
        path = pathlib.Path(candidate)
        if path.is_file() and lipo_archs(path) == "arm64":
            return path
    raise AssertionError(
        "no exactly-arm64 Mach-O on this host (tried "
        + ", ".join(ARM64_CANDIDATES)
        + ") — the fixtures cannot stand in for a staged artifact"
    )


def lipo_archs(path: pathlib.Path) -> str:
    proc = subprocess.run(
        ["lipo", "-archs", str(path)], capture_output=True, text=True, check=False
    )
    return proc.stdout.strip() if proc.returncode == 0 else f"error: {proc.stderr.strip()}"


class AssertArchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arm64 = find_arm64_binary()

    def setUp(self):
        stage = tempfile.TemporaryDirectory(prefix="assert-arch-")
        self.addCleanup(stage.cleanup)
        self.root = pathlib.Path(stage.name)
        self.restore = []

    def dir_with(self, *names: str) -> pathlib.Path:
        """A directory holding a clean arm64 artifact plus the named subjects."""
        target = self.root / "stage"
        target.mkdir(exist_ok=True)
        shutil.copy(self.arm64, target / "arm64_bin")
        for name in names:
            getattr(self, f"make_{name}")(target)
        return target

    def make_fat(self, target: pathlib.Path) -> pathlib.Path:
        # /bin/ls on Apple Silicon is multi-arch, which is precisely what the rule forbids.
        return shutil.copy("/bin/ls", target / "zzz_fat")

    def make_unreadable(self, target: pathlib.Path) -> pathlib.Path:
        path = shutil.copy(self.arm64, target / "zzz_unreadable")
        path.chmod(0o000)
        self.restore.append(path)
        return path

    def make_truncated(self, target: pathlib.Path) -> pathlib.Path:
        """The first three bytes of a Mach-O header — a cp that died mid-write."""
        head = self.arm64.read_bytes()[:3]
        path = target / "zzz_truncated"
        path.write_bytes(head)
        return path

    def tearDown(self):
        for path in self.restore:
            path.chmod(0o644)

    def run_gate(self, *args: str) -> tuple[int, str]:
        proc = subprocess.run(
            ["/bin/bash", str(GATE), *args], capture_output=True, text=True, check=False
        )
        return proc.returncode, proc.stdout + proc.stderr

    def test_a_clean_directory_passes_and_says_what_it_counted(self):
        rc, out = self.run_gate(str(self.dir_with()))
        self.assertEqual(0, rc, out)
        self.assertIn("1 Mach-O artifact(s), each exactly arm64", out)

    def test_a_multi_arch_artifact_is_refused_by_name(self):
        rc, out = self.run_gate(str(self.dir_with("fat")))
        self.assertEqual(1, rc, out)
        self.assertIn("zzz_fat", out)
        self.assertIn("expected exactly 'arm64'", out)

    def test_a_non_mach_o_file_is_skipped_and_still_leaves_the_count_honest(self):
        target = self.dir_with()
        (target / "README.md").write_text("# notes\n", encoding="utf-8")
        rc, out = self.run_gate(str(target))
        self.assertEqual(0, rc, out)
        self.assertIn("1 Mach-O artifact(s)", out)

    def test_an_artifact_whose_bytes_cannot_be_read_is_refused_not_skipped(self):
        # The bug: od's status is hidden behind the pipe, so EACCES looks like a
        # non-Mach-O and the gate passes while claiming to have checked everything.
        rc, out = self.run_gate(str(self.dir_with("unreadable")))
        self.assertEqual(1, rc, f"the gate passed over an artifact it could not read:\n{out}")
        self.assertIn("zzz_unreadable", out)
        self.assertIn("cannot read", out)

    def test_a_truncated_mach_o_header_is_refused_not_skipped(self):
        rc, out = self.run_gate(str(self.dir_with("truncated")))
        self.assertEqual(1, rc, f"the gate passed over a truncated header:\n{out}")
        self.assertIn("zzz_truncated", out)
        self.assertIn("truncated", out)

    def test_an_empty_directory_is_refused_as_a_run_that_checked_nothing(self):
        empty = self.root / "empty"
        empty.mkdir()
        rc, out = self.run_gate(str(empty))
        self.assertEqual(1, rc, out)
        self.assertIn("no Mach-O found", out)

    def test_a_missing_target_is_refused_rather_than_called_a_non_artifact(self):
        rc, out = self.run_gate(str(self.root / "never-staged"))
        self.assertEqual(1, rc, out)
        self.assertIn("cannot read", out)

    def test_a_64_bit_fat_header_is_a_candidate_and_not_a_skip(self):
        # The list its own comment calls "fat 32/64" carried only fat 32, so a
        # 64-bit universal header took the skip and a directory scan over it
        # printed "each exactly arm64". The fixture is a four-byte header stub:
        # what is pinned is that the gate consults `lipo` for this magic instead
        # of passing it by.
        target = self.dir_with()
        (target / "zzz_fat64").write_bytes(b"\xfe\xed\xfa\xcb")
        rc, out = self.run_gate(str(target))
        self.assertEqual(1, rc, f"a 64-bit fat header was skipped:\n{out}")
        self.assertIn("zzz_fat64", out)
        self.assertNotIn("each exactly arm64", out)

    def test_the_archive_of_a_multi_arch_artifact_is_refused(self):
        stage = self.dir_with("fat")
        archive = self.root / "tinytitan-test.tar.gz"
        subprocess.run(
            ["tar", "czf", str(archive), "-C", str(self.root), stage.name],
            check=True,
            capture_output=True,
        )
        rc, out = self.run_gate(str(archive))
        self.assertEqual(1, rc, out)
        self.assertIn("zzz_fat", out)

    def test_sourcing_it_still_stops_the_script_that_calls_it(self):
        # release.sh and build_library.sh run without `set -e`; tt_die's `exit` is
        # the only thing that keeps a refused artifact from being published.
        script = f'. "{GATE}"; assert_arm64_dir "{self.root}/empty" probe; echo NOT_REACHED'
        (self.root / "empty").mkdir()
        proc = subprocess.run(
            ["/bin/bash", "-c", script], capture_output=True, text=True, check=False
        )
        self.assertEqual(1, proc.returncode, proc.stdout + proc.stderr)
        self.assertNotIn("NOT_REACHED", proc.stdout)
        self.assertIn("no Mach-O found", proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main()
