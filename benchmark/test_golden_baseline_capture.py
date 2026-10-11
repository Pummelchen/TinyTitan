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
#
# AUD-268 made the answer choosable so the busy and cannot-ask branches could be
# tested at all (`test_model_process_guard.py` drives them). The defaults are the
# answers these seven tests always had: exit 1, print nothing.
PGREP_STUB = """#!/bin/sh
if [ -n "${STUB_PGREP_ERR-}" ]; then printf '%s\\n' "$STUB_PGREP_ERR" >&2; fi
if [ -n "${STUB_PGREP_OUT-}" ]; then printf '%s' "$STUB_PGREP_OUT"; fi
exit "${STUB_PGREP_RC:-1}"
"""

# The guard is a second file since AUD-268, sourced next to the script, so the
# copied checkout has to carry it too or the script dies at the source line.
GUARD = ROOT / "tools" / "model-guard.sh"


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
        shutil.copy(GUARD, root / "tools" / "model-guard.sh")
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


def run(
    root: pathlib.Path, output: str, *args: str, **operator_values: str
) -> subprocess.CompletedProcess[str]:
    """Run the copied script, optionally with the operator's own numbers set.

    `MAX_NEW`, `SEED`, `PORT` and `READY_TIMEOUT` are environment defaults the
    script reads, so a test names the one it is about; anything left out stays at
    the script's own default.
    """
    env = dict(os.environ)
    env["PATH"] = f"{root / 'stubs'}{os.pathsep}{env['PATH']}"
    env["GOLDEN_FIXTURE_OUTPUT"] = output
    for key in ("STUB_PGREP_RC", "STUB_PGREP_OUT", "STUB_PGREP_ERR"):
        env.pop(key, None)
    for key in ("MAX_NEW", "SEED", "PORT", "READY_TIMEOUT"):
        if key in operator_values:
            env[key] = operator_values[key]
        else:
            env.pop(key, None)
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


class GoldenOperatorNumbers(unittest.TestCase):
    """The four numbers the operator can set are read far from where they mislead."""

    # MAX_NEW, SEED, PORT and READY_TIMEOUT are environment defaults
    # (tools/golden-baseline.sh:71-76). Where each one is actually read, and what
    # refuses a wrong one, measured on /bin/bash 3.2.57 and python3:
    #
    #   number          readers                                  guarded by
    #   MAX_NEW         CLI --max-new (Args.swift:254), the body  the CLI only,
    #                   builder's int() (:244-252)                after the load
    #   SEED            same two                                 the CLI only
    #   PORT            the launcher (:1230-1234), two curl URLs  the launcher
    #   READY_TIMEOUT   [ "$waited" -lt "$READY_TIMEOUT" ] (:223)  nothing
    #
    # READY_TIMEOUT is the reason this suite exists here. A word makes `test`
    # error and take the false branch -- `[ 0 -lt 1800s ]` prints
    # "integer expression expected" and is false -- so the poll loop never runs,
    # `ready` stays 0, and :236 prints "server: FAILED - not ready after 1800s":
    # the operator's own typo, quoted back as a model that took too long to
    # load. The other three cost the same class of thing one step later: the
    # script spends minutes loading a model, then hands the bad word to the CLI
    # or to `int()` and reports the failure as the run's.
    #
    # So the guard refuses at the read, in this file's own `exit 2` style, before
    # anything expensive runs. It deliberately refuses less than a sanitizer
    # could:
    #
    #  - Leading zeros stay legal for MAX_NEW, SEED and READY_TIMEOUT because
    #    `test` reads them as decimal (`[ 0 -lt 0900 ]` is true) and so does
    #    `int()`. A refusal there would be a rule no reader has.
    #  - A port is the exception, and measured: the launcher's range check is
    #    `(( PORT < 1 || PORT > 65535 ))`, and `(( ))` does read octal, so
    #    `08757` prints "value too great for base", evaluates to neither branch,
    #    and is accepted unchecked. Refusing it here is what keeps a port from
    #    reaching a guard that silently does nothing.
    #  - Zero is refused only where zero means "run nothing": MAX_NEW and
    #    READY_TIMEOUT. SEED=0 is a seed and still runs.
    #  - Out-of-range-but-numeric values (a 90,000-token MAX_NEW) stay the CLI's
    #    named error; this script does not restate the engine's bounds.
    #
    # The `:NNNN` numbers above are the pre-fix file; the guard is what shifted
    # them. The sweep killed seventeen of seventeen shapes, including the four
    # "sanitiser instead of refusal" mutants (each default line rewritten to
    # ignore the operator), the string-compare twins of the two zero checks, a
    # port ceiling off by one, a refusal moved to stdout, a guard widened to
    # refuse `SEED=0` and leading zeros, and the guard removed. The last one is
    # the placement test: `test_the_guard_refuses_before_anything_expensive`
    # compares the guard's position against the first `"$CLI" --model` and the
    # readiness poll, so a guard that moves below the cost it exists to avoid is
    # caught even though no model-free test can feel the wait.

    def assert_refused(
        self, root: pathlib.Path, result: subprocess.CompletedProcess, name: str
    ) -> None:
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(name, result.stderr, result.stdout + result.stderr)
        # Nothing downstream ran, so none of the shapes a bad number used to
        # produce -- a false timeout, an empty completion, a stored baseline --
        # may appear.
        combined = result.stdout + result.stderr
        for marker in ("not ready after", "no completion returned", "captured ->"):
            self.assertNotIn(marker, combined)
        self.assertFalse(baseline(root).exists(), combined)

    def refuse(self, **operator_values: str) -> None:
        name = next(iter(operator_values))
        with tree() as root:
            result = run(root, SAMPLE, "ornith-8", **operator_values)
            self.assert_refused(root, result, name)

    def test_a_word_for_max_new_refuses(self) -> None:
        self.refuse(MAX_NEW="96s")

    def test_a_negative_max_new_refuses(self) -> None:
        # The CLI refuses it, but only after the model load; `int('-5')` in the
        # body builder does not, so the second reader would have accepted it.
        self.refuse(MAX_NEW="-5")

    def test_a_zero_max_new_refuses(self) -> None:
        # The spelling is the point as much as the value: the guard compares with
        # `-eq`, which reads "00" as the zero it is. A string compare would let it
        # through to a `[ 0 -lt ... ]` that never polls.
        for value in ("0", "00"):
            with self.subTest(value=value):
                self.refuse(MAX_NEW=value)

    def test_a_word_for_seed_refuses(self) -> None:
        self.refuse(SEED="12a4")

    def test_a_zero_seed_is_a_seed_and_still_runs(self) -> None:
        # The twin of the AUD-301 lesson: do not invent a refusal for a value the
        # reader accepts. `0` is a switch to nothing here; it is a seed.
        with tree() as root:
            result = run(root, SAMPLE, "ornith-8", SEED="0")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("captured ->", result.stdout)
            self.assertIn("# seed:        0", baseline(root).read_text(encoding="utf-8"))

    def test_a_word_for_ready_timeout_refuses(self) -> None:
        # The defect itself: before the guard this reached :223, made `test`
        # error, skipped the poll, and printed "not ready after 1800s".
        self.refuse(READY_TIMEOUT="1800s")

    def test_a_zero_ready_timeout_refuses(self) -> None:
        # `[ 0 -lt 0 ]` is false without erroring, so a zero skips the poll the
        # same way a word does -- and reports the same timeout.
        for value in ("0", "00"):
            with self.subTest(value=value):
                self.refuse(READY_TIMEOUT=value)

    def test_a_port_that_is_not_a_port_refuses(self) -> None:
        for value in ("8080a", "1e4", "0"):
            with self.subTest(value=value):
                self.refuse(PORT=value)

    def test_a_port_above_the_port_space_refuses(self) -> None:
        for value in ("65536", "99999"):
            with self.subTest(value=value):
                self.refuse(PORT=value)

    def test_a_leading_zero_port_refuses(self) -> None:
        self.refuse(PORT="08757")

    def test_a_leading_zero_number_is_the_number_it_reads_as(self) -> None:
        # `test` and `int()` both read these as decimal, so a guard that refused
        # them would be stricter than any reader here. Pinned so the fix cannot
        # grow that rule later.
        cases = (
            {"MAX_NEW": "096"},
            {"SEED": "01234"},
            {"READY_TIMEOUT": "0900"},
        )
        for operator_values in cases:
            with self.subTest(**operator_values):
                with tree() as root:
                    result = run(root, SAMPLE, "ornith-8", **operator_values)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("captured ->", result.stdout)

    def test_the_guard_refuses_before_anything_expensive(self) -> None:
        # The whole point of the placement: a bad number has to be refused while
        # refusing is cheap, not after a model has been loaded or a server polled.
        # Model-free, the only observable is order, so the order is asserted
        # against the file itself.
        text = SCRIPT.read_text(encoding="utf-8")
        guard = text.index('for name in MAX_NEW SEED PORT READY_TIMEOUT')
        self.assertLess(guard, text.index('"$CLI" --model'))
        self.assertLess(guard, text.index('while [ "$waited" -lt "$READY_TIMEOUT" ]'))

    def test_the_documented_default_run_is_unchanged(self) -> None:
        # The guard must sit over the refusals only: the plain documented call
        # still captures, and still checks.
        with tree() as root:
            result = run(root, SAMPLE, "ornith-8")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("captured ->", result.stdout)
            checked = run(root, SAMPLE, "--check", "ornith-8")
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
            self.assertIn("identical to baseline", checked.stdout)


if __name__ == "__main__":
    unittest.main()
