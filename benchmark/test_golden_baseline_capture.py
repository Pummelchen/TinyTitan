"""`tools/golden-baseline.sh` is the only check that exercises real inference.

The row these pin (AUD-208) is the capture path accepting a run that produced
nothing. `--check` compares the stored baseline's body — everything after its
`---` — against what the CLI prints, so a baseline with an empty body passes
against an empty run: "ok — output identical to baseline", exit 0, forever. The
gate that exists to catch a runtime regression had stopped being able to fail,
and the capture arm that wrote it never looked at the size of what it captured,
never checked the write it performed, and printed the temporary file's byte
count as though it were the size of the baseline.

The fixture is a tree of stubs, not a model. The script derives its repository
root from its own location, so it is copied into a temporary checkout with a
fake CLI whose output each test chooses. No model, no GPU, no network, and no
install under `models/` is touched.

Run from `benchmark/`:

    python3 -m unittest test_golden_baseline_capture
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "golden-baseline.sh"
MODEL_DIR = "ornith-1.5_35B_A3B_8Bit"
BASELINE_NAME = "ornith-1.5-35b-a3b-8bit.txt"

# Prints exactly what the test asks and exits with the status it asks, so both
# the empty generation and a refusal are reproducible without a model.
CLI_STUB = """#!/bin/sh
printf '%s' "${GOLDEN_FIXTURE_OUTPUT-}"
exit "${GOLDEN_FIXTURE_RC:-0}"
"""

# The script refuses to start beside a model process, and that guard is correct:
# it is what stops this gate racing someone else's GPU. Nothing here touches a
# GPU — the CLI is a `printf` — so the fixture answers the guard the same way
# `test_release_ci_green.py` answers `gh`: a stub earlier on `PATH`. The real
# `pgrep` is untouched for every other caller, and no production behavior
# changes; a run against a real install still refuses while a server is up.
PGREP_STUB = """#!/bin/sh
exit 1
"""


@contextlib.contextmanager
def tree(*, golden_is_file: bool = False, baseline_is_dir: bool = False):
    """A checkout with nothing in it but a stub CLI and one verified install.

    `golden_is_file` makes the whole output directory a plain file, which the
    script has to notice before it starts. `baseline_is_dir` leaves the
    directory sound and puts a directory where this target's baseline file
    belongs, which only the write itself can notice.
    """
    with tempfile.TemporaryDirectory() as work:
        root = pathlib.Path(work)
        (root / "tools").mkdir()
        script = root / "tools" / "golden-baseline.sh"
        shutil.copy(SCRIPT, script)
        script.chmod(0o755)
        build = root / ".build" / "release"
        build.mkdir(parents=True)
        cli = build / "TinyTitanCLI"
        cli.write_text(CLI_STUB, encoding="utf-8")
        cli.chmod(0o755)
        stubs = root / "stubs"
        stubs.mkdir()
        pgrep = stubs / "pgrep"
        pgrep.write_text(PGREP_STUB, encoding="utf-8")
        pgrep.chmod(0o755)
        model = root / "models" / MODEL_DIR
        model.mkdir(parents=True)
        (model / "verified-install.json").write_text("{}\n", encoding="utf-8")
        golden = root / "benchmark" / "golden"
        if golden_is_file:
            golden.parent.mkdir(parents=True)
            golden.write_text("not a directory\n", encoding="utf-8")
        else:
            golden.mkdir(parents=True)
            if baseline_is_dir:
                (golden / BASELINE_NAME).mkdir()
        yield root


def run(root: pathlib.Path, output: str, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PATH"] = f"{root / 'stubs'}{os.pathsep}{env['PATH']}"
    env["GOLDEN_FIXTURE_OUTPUT"] = output
    result = subprocess.run(
        ["/bin/bash", str(root / "tools" / "golden-baseline.sh"), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    return result


def baseline(root: pathlib.Path) -> pathlib.Path:
    return root / "benchmark" / "golden" / BASELINE_NAME


def body(path: pathlib.Path) -> str:
    """The part `--check` compares: everything after the header's `---`."""
    text = path.read_text(encoding="utf-8")
    return text.split("---\n", 1)[1] if "---\n" in text else ""


SAMPLE = "A mutex is a lock. Use it when two threads write the same value.\n"


class GoldenCaptureTests(unittest.TestCase):
    def test_capture_refuses_a_generation_that_produced_nothing(self) -> None:
        with tree() as root:
            result = run(root, "", "ornith-8")
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            # Refusing is only half of it: a vacuous baseline must not be left
            # behind for the next `--check` to certify.
            path = baseline(root)
            if path.exists():
                self.fail(f"a baseline was stored over an empty run: {path.read_text()!r}")

    def test_check_refuses_a_baseline_with_an_empty_body(self) -> None:
        with tree() as root:
            captured = run(root, SAMPLE, "ornith-8")
            self.assertEqual(captured.returncode, 0, captured.stdout + captured.stderr)
            path = baseline(root)
            path.write_text(
                path.read_text(encoding="utf-8").split("---\n", 1)[0] + "---\n",
                encoding="utf-8",
            )
            result = run(root, "", "--check", "ornith-8")
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn("identical to baseline", result.stdout)

    def test_capture_reports_the_size_of_the_file_it_wrote(self) -> None:
        with tree() as root:
            result = run(root, SAMPLE, "ornith-8")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            reported = re.search(r"captured -> \S+ \((\d+) bytes\)", result.stdout)
            self.assertIsNotNone(reported, result.stdout)
            self.assertEqual(
                int(reported.group(1)),
                baseline(root).stat().st_size,
                f"the line says {reported.group(1)} bytes, the file is "
                f"{baseline(root).stat().st_size}",
            )

    def test_capture_fails_when_the_output_dir_cannot_be_made(self) -> None:
        with tree(golden_is_file=True) as root:
            result = run(root, SAMPLE, "ornith-8")
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn("captured ->", result.stdout)
            # The message is asserted, not just the exit code: the write guard
            # below would also refuse this shape, so without it a broken `mkdir`
            # guard would be invisible.
            self.assertIn("cannot create", result.stderr, result.stdout + result.stderr)

    def test_capture_fails_when_the_baseline_file_cannot_be_written(self) -> None:
        # The output directory is sound here, so the only thing that can refuse
        # this capture is the status of the write itself.
        with tree(baseline_is_dir=True) as root:
            result = run(root, SAMPLE, "ornith-8")
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn("captured ->", result.stdout)
            self.assertIn("FAILED to write", result.stdout, result.stdout + result.stderr)

    def test_a_normal_capture_stays_checkable(self) -> None:
        with tree() as root:
            captured = run(root, SAMPLE, "ornith-8")
            self.assertEqual(captured.returncode, 0, captured.stdout + captured.stderr)
            self.assertEqual(baseline(root).read_text(encoding="utf-8").count("---\n"), 1)
            self.assertEqual(body(baseline(root)), SAMPLE)
            checked = run(root, SAMPLE, "--check", "ornith-8")
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
            self.assertIn("identical to baseline", checked.stdout)

    def test_a_real_mismatch_still_fails_the_check(self) -> None:
        with tree() as root:
            captured = run(root, SAMPLE, "ornith-8")
            self.assertEqual(captured.returncode, 0, captured.stdout + captured.stderr)
            result = run(root, "Something else came out.\n", "--check", "ornith-8")
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("MISMATCH", result.stdout)


if __name__ == "__main__":
    unittest.main()
