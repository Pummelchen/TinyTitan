"""Gates `benchmark/tinytitan_overlap_measure.py`'s verdict and exit status.

The driver's whole product is the server log it captures. Until AUD-225 a run
in which the capture held nothing — a server that never started, a binary that
no longer prints `cb1_ms=`, an env var the release CLI dropped — printed three
`=== ... ===` section headers over three empty bodies and exited **0**. The
reader could not tell "the instrument measured zero" from "the instrument
measured nothing", and the headers asserted that three sections existed.

Like AUD-213 and AUD-224, the distinction held here is that a bad reading is a
result and a run that captured nothing is not: every section the driver claims
must have at least one captured line, or the run says which section is empty
and fails.

Two of these tests also pin the refactor that made the rest possible. The module
used to open its log and `Popen` a `TinyTitanServer` at *import*, so `import
tinytitan_overlap_measure` started a model process, and it cast
`int(sys.argv[1])` at module level against the importer's argv. Neither is
re-measured against the real engine here: no model, no server and no GPU is
involved, because `Popen`, the health wait and the request loop are all faked.
The overlap numbers the driver produces remain the operator's to schedule.

    cd benchmark && python3 -m unittest test_tinytitan_overlap_measure -v
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "tinytitan_overlap_measure", ROOT / "benchmark" / "tinytitan_overlap_measure.py"
)
om = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(om)

GEN = "TinyTitan generation prefill_s=4.912 decode_s=11.900 decode_tok_s=43.210"
RUNNER = "TinyTitan runner cb1_ms=1.210 io_ms=0.440 cb2_ms=0.980"
KERNEL_ROLE = "TinyTitan kernel role=gemv gpu_ms=0.310 per_token_ms=0.021 count=64"
KERNEL_TOTAL = "TinyTitan kernel total_gpu_ms=12.700 gpu_share_of_decode=10.7%"
BUSY = "TinyTitan kernel busy_ms=9.800 span_ms=11.200"

FULL_LOG = [GEN, GEN, RUNNER, RUNNER, KERNEL_ROLE, KERNEL_TOTAL]


class CaptureTests(unittest.TestCase):
    """The classification the refactor must not have changed."""

    def test_each_line_lands_in_the_section_the_prose_names(self):
        gen, runner, kernels = om.capture(FULL_LOG)
        self.assertEqual(gen, [GEN, GEN])
        self.assertEqual(runner, [RUNNER, RUNNER])
        self.assertEqual(kernels, [KERNEL_ROLE, KERNEL_TOTAL])

    def test_a_generation_line_without_a_decode_rate_is_not_a_footer(self):
        """`decode_tok_s=` is the half that carries the measurement; a bare
        `TinyTitan generation` line is a progress print."""
        gen, _, _ = om.capture(["TinyTitan generation started"])
        self.assertEqual(gen, [])

    def test_a_runner_line_without_the_stage_split_is_not_a_stage_split(self):
        _, runner, _ = om.capture(["TinyTitan runner warm"])
        self.assertEqual(runner, [])

    def test_whitespace_is_stripped_from_every_captured_line(self):
        gen, _, _ = om.capture([f"  {GEN}  \n"])
        self.assertEqual(gen, [GEN])

    def test_the_busy_and_span_channel_reaches_no_section(self):
        """Pinned as the driver's scope, not as its ideal: the engine also prints
        `TinyTitan kernel busy_ms=… span_ms=…`, which is the one line that names
        overlap directly, and no section reads it. Filed as a follow-up, not
        folded into this fix."""
        _, _, kernels = om.capture([BUSY])
        self.assertEqual(kernels, [])


class VerdictTests(unittest.TestCase):
    def verdict(self, gen, runner, kernels):
        """(printed text, exit status) -- `verdict()` returns lines, and the
        assertions below read the page a user sees."""
        rendered, status = om.verdict(gen, runner, kernels)
        return "\n".join(rendered), status

    def test_an_empty_capture_is_not_reported_as_three_empty_sections(self):
        """The old output: three headers, nothing under them, exit 0. A section
        header is a claim that the section was measured."""
        rendered, status = self.verdict([], [], [])
        self.assertEqual(status, 1, "a run that captured nothing must not exit 0")
        self.assertEqual(rendered.count("NOT MEASURED"), 3, rendered)

    def test_an_empty_section_names_which_one_it_was(self):
        """The runner channel is the one the overlap analysis exists for, so a
        missing `cb1_ms=` line has to say so rather than print an empty body."""
        rendered, status = self.verdict([GEN], [], [KERNEL_ROLE])
        self.assertEqual(status, 1)
        self.assertIn("NOT MEASURED", rendered)
        self.assertIn("runner stage splits", rendered)
        self.assertIn("cb1_ms=", rendered, "say what the line had to contain")

    def test_a_full_capture_of_every_section_exits_zero(self):
        """The guard must not swallow the case it exists to report: three
        captured sections are a measurement."""
        rendered, status = self.verdict([GEN, GEN], [RUNNER], [KERNEL_ROLE, KERNEL_TOTAL])
        self.assertEqual(status, 0, rendered)
        self.assertNotIn("NOT MEASURED", rendered)
        self.assertIn(GEN, rendered)
        self.assertIn(KERNEL_TOTAL, rendered)

    def test_every_section_prints_how_many_lines_were_captured(self):
        """The number a reader could not see before: two requests were sent, so
        a section with one line under it is a partial run, and says so."""
        rendered, _ = self.verdict([GEN, GEN], [RUNNER], [KERNEL_ROLE])
        self.assertIn("2 lines", rendered)
        self.assertIn("1 line", rendered)


class ArgumentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="overlap-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.log = self.tmp / "overlap.log"

    def run_main(self, argv, log_lines=None, healthy=True):
        """The real `main()`, with the server, the health wait and the request
        loop replaced by fakes that write `log_lines` into the log the driver opens.

        Returns (status, output, request lengths, child env). No HTTP and no
        process is involved, so the port the driver would use is never touched.
        """
        sent, seen = [], {}

        class FakeProc:
            def __init__(self, *args, **kwargs):
                self.stream = kwargs["stdout"]
                seen["args"] = args
                seen["env"] = kwargs["env"]
                self.stopped = False

            def poll(self):
                return None

            def terminate(self):
                if self.stopped:
                    return
                self.stopped = True
                for line in log_lines or []:
                    self.stream.write(line + "\n")
                self.stream.flush()

            def wait(self, timeout=None):
                return 0

            def kill(self):
                self.terminate()

        def fake_health(proc, *args, **kwargs):
            return healthy

        def fake_request(*args, **kwargs):
            sent.append(args)

        def fake_command(*args, **kwargs):
            return ["fake-server"]

        def fake_log_path(name):
            return str(self.log)

        buffer = io.StringIO()
        original_argv = sys.argv
        sys.argv = ["tinytitan_overlap_measure.py", *argv]
        try:
            with (
                mock.patch.object(om.subprocess, "Popen", FakeProc),
                mock.patch.object(om, "server_command", new=fake_command),
                mock.patch.object(om, "wait_for_health", new=fake_health),
                mock.patch.object(om, "request_twice", new=fake_request),
                mock.patch.object(om, "benchmark_log_path", new=fake_log_path),
                contextlib.redirect_stdout(buffer),
                contextlib.redirect_stderr(buffer),
            ):
                status = om.main()
        finally:
            sys.argv = original_argv
        return status, buffer.getvalue(), sent, seen

    def test_a_run_that_captured_nothing_exits_non_zero(self):
        status, output, _, _ = self.run_main([])
        self.assertEqual(status, 1, output)
        self.assertIn("NOT MEASURED", output)

    def test_a_run_that_captured_everything_exits_zero(self):
        status, output, _, _ = self.run_main([], FULL_LOG)
        self.assertEqual(status, 0, output)
        self.assertNotIn("NOT MEASURED", output)

    def test_the_output_is_the_captured_sections(self):
        """The driver's product is the log lines, and they must survive the verdict."""
        _, output, _, _ = self.run_main([], FULL_LOG)
        self.assertIn(GEN, output)
        self.assertIn(KERNEL_TOTAL, output)

    def test_the_driver_asks_for_the_stats_it_reads_back(self):
        """The docstring's claim is an env var. If a future edit stops setting
        `TINYTITAN_KERNEL_STATS`, the kernel section goes empty and this is the
        test that says which half of the instrument was unplugged."""
        _, _, _, seen = self.run_main([], FULL_LOG)
        self.assertEqual(seen["env"].get("TINYTITAN_RUNNER_STATS"), "1")
        self.assertEqual(seen["env"].get("TINYTITAN_KERNEL_STATS"), "1")

    def test_a_non_integer_length_is_refused_before_the_server_starts(self):
        """The module used to `int(sys.argv[1])` at import and raise a Traceback;
        a refusal needs the value quoted and no process started."""
        status, output, _, seen = self.run_main(["not-a-number"])
        self.assertEqual(status, 2)
        self.assertIn("not-a-number", output)
        self.assertNotIn("Traceback", output)
        self.assertNotIn("env", seen, "a refused argument must not start a server")

    def test_a_zero_or_negative_length_is_refused(self):
        for value in ("0", "-5"):
            status, output, _, _ = self.run_main([value])
            self.assertEqual(status, 2, f"{value}: {output}")
            self.assertIn(value, output)

    def test_the_default_length_is_the_one_the_payload_asks_for(self):
        _, _, sent, _ = self.run_main([])
        self.assertEqual(sent, [(om.PROMPT, 512, om.PORT)])

    def test_an_explicit_length_reaches_the_payload(self):
        _, _, sent, _ = self.run_main(["1024"])
        self.assertEqual(sent, [(om.PROMPT, 1024, om.PORT)])

    def test_a_server_that_died_is_still_refused(self):
        """The pre-existing early-exit guard, pinned so the new verdict does not
        replace it: a dead server is not a measurement of anything."""
        status, output, _, _ = self.run_main([], FULL_LOG, healthy=False)
        self.assertEqual(status, 1)
        self.assertIn("server exited early", output)


class ImportTests(unittest.TestCase):
    def test_importing_the_module_starts_no_server_and_opens_no_log(self):
        """The defect that made every other test here impossible to write: the
        driver `Popen`ed `TinyTitanServer` at import, so `import` = a model run.
        The guard is installed inside the child, so a surviving import raises there."""
        program = (
            "import sys, subprocess\n"
            "sys.argv = ['tinytitan_overlap_measure.py', 'not-a-number']\n"
            "def boom(*a, **k):\n"
            "    raise RuntimeError('Popen ran at import')\n"
            "subprocess.Popen = boom\n"
            "import tinytitan_overlap_measure\n"
            "print('imported clean')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", program],
            cwd=ROOT / "benchmark",
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "PYTHONPATH": str(ROOT / "benchmark")},
        )
        self.assertEqual(result.returncode, 0, result.stderr[-400:])
        self.assertIn("imported clean", result.stdout)


if __name__ == "__main__":
    unittest.main()
