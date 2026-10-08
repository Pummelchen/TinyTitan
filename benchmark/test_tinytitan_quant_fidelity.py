"""Gates `benchmark/tinytitan_quant_fidelity.py`'s verdict and exit status.

The driver answers one question -- does this community 4-bit build match the
official bf16 weights -- and its answer line is the thing a precision promotion
gets quoted. Until AUD-226 that line read `VERDICT: quantization is faithful`
whenever `bad` was zero, and `bad` is incremented only by a tensor that was
*compared and disagreed*. A target missing from either index, a fetch that
failed, or a build whose `.scales` entry is absent all `continue` without
touching it, so a run in which not one tensor was measured printed "faithful" and
exited **0**. That is the same shape AUD-212 left in the memory matrix, AUD-218
in tsan-storm and AUD-224 in the determinism probe: the instrument that measured
nothing reports a pass.

The second half of the finding is why no test existed. The module loaded three
Hugging Face index files at *import*, from the current working directory, so
`import tinytitan_quant_fidelity` raised `FileNotFoundError` before a caller could
assert anything about it.

No download and no model is involved: `fetch()` is answered from in-memory arrays
and the index files are written into a temporary directory, so the real
`dequant()`, the real error arithmetic and the real verdict all run over synthetic
tensors. `numpy` is imported by the driver itself, which is why this suite is
registered in the CI step that installs `benchmark/requirements.txt` rather than
the stdlib-only one.

    cd benchmark && python3 -m unittest test_tinytitan_quant_fidelity -v
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "tinytitan_quant_fidelity", ROOT / "benchmark" / "tinytitan_quant_fidelity.py"
)
qf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qf)

# One 64-wide row: eight u32 holding eight 4-bit codes each, all zero, so
# `dequant()` returns the bias broadcast across the row. Group size is 64, so
# scales and biases are one value each.
OUT, WIDTH, GROUP = 1, 64, 64
PACKED = np.zeros((OUT, WIDTH // 8), dtype=np.uint32)
SCALES = np.ones((OUT, 1), dtype=np.float32)
BIASES = np.ones((OUT, 1), dtype=np.float32)
FAITHFUL_REF = np.broadcast_to(BIASES, (OUT, WIDTH)).astype(np.float32)
WRONG_REF = np.zeros((OUT, WIDTH), dtype=np.float32)


class VerdictTests(unittest.TestCase):
    def test_zero_compared_tensors_is_not_a_pass(self):
        """The broken-instrument case: `bad` is zero because nothing was measured,
        and `bad == 0` was the whole test for faithfulness."""
        lines, status = qf.verdict(checked=0, bad=0, skipped=6, total=6)
        rendered = "\n".join(lines)
        self.assertEqual(status, 1, "a run that compared nothing must not exit 0")
        self.assertNotIn("faithful", rendered, "the word the reader would quote")
        self.assertIn("NOT MEASURED", rendered)

    def test_the_refusal_prints_the_denominator_it_refused(self):
        """Six targets were asked for; saying only "nothing measured" leaves the
        reader to guess whether that is six or one."""
        rendered = "\n".join(qf.verdict(checked=0, bad=0, skipped=6, total=6)[0])
        self.assertIn("0 of 6", rendered)
        self.assertIn("6", rendered)

    def test_a_partial_run_says_what_it_left_out(self):
        """A comparison that measured three of six is a result, so it exits 0 --
        but the verdict line must carry the denominator it actually covered."""
        lines, status = qf.verdict(checked=3, bad=0, skipped=3, total=6)
        rendered = "\n".join(lines)
        self.assertEqual(status, 0, "a measured subset is still a measurement")
        self.assertIn("3 of 6", rendered)
        self.assertIn("faithful", rendered)

    def test_a_measured_disagreement_still_fails(self):
        """The guard must not swallow the finding the driver exists to produce."""
        lines, status = qf.verdict(checked=6, bad=2, skipped=0, total=6)
        self.assertEqual(status, 1)
        self.assertIn("2 tensor(s) SUSPECT", "\n".join(lines))

    def test_a_complete_faithful_run_exits_zero(self):
        lines, status = qf.verdict(checked=6, bad=0, skipped=0, total=6)
        self.assertEqual(status, 0)
        rendered = "\n".join(lines)
        self.assertIn("6 of 6", rendered)
        self.assertNotIn("NOT MEASURED", rendered)


class DriverTests(unittest.TestCase):
    """The real `main()` over synthetic indices, with `fetch()` answering from
    arrays instead of HTTP range requests."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="quant-fidelity-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def write_indices(self, off_names, mlx_names):
        (self.tmp / "off_index.json").write_text(
            json.dumps({"weight_map": {n: "official.safetensors" for n in off_names}}),
            encoding="utf-8",
        )
        (self.tmp / "rt_index.json").write_text(
            json.dumps({"weight_map": {n: "community.safetensors" for n in mlx_names}}),
            encoding="utf-8",
        )
        (self.tmp / "rt_config.json").write_text(
            json.dumps({"quantization_config": {"group_size": GROUP}}), encoding="utf-8"
        )

    def run_main(self, off_names, mlx_names, ref):
        """`main()` with the network half replaced: every tensor the indices name
        comes back from `ref` (official side) or the packed community triple."""

        def fake_fetch(repo, rev, shard, name):
            if name.endswith(".scales"):
                return SCALES
            if name.endswith(".biases"):
                return BIASES
            if "language_model" not in name:
                return None
            return ref if repo == qf.OFF else PACKED

        buffer = io.StringIO()
        original_cwd, original_fetch, original_argv = os.getcwd(), qf.fetch, sys.argv
        os.chdir(self.tmp)
        qf.fetch = fake_fetch
        sys.argv = ["tinytitan_quant_fidelity.py"]
        try:
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                status = qf.main()
        finally:
            os.chdir(original_cwd)
            qf.fetch = original_fetch
            sys.argv = original_argv
        return status, buffer.getvalue()

    def test_indices_are_read_where_the_driver_runs(self):
        """The three index files were opened at import, from the current working
        directory, which is what made the module impossible to import or test.
        Empty indices are the cheapest honest input: every target is unavailable."""
        self.write_indices([], [])
        status, output = self.run_main([], [], FAITHFUL_REF)
        self.assertEqual(status, 1, output)
        self.assertIn("NOT MEASURED", output)

    def test_a_faithful_tensor_passes_and_says_so(self):
        names = [qf.TARGETS[0]]
        stem = names[0][: -len(".weight")]
        indexed = names + [stem + ".scales", stem + ".biases"]
        self.write_indices(indexed, indexed)
        status, output = self.run_main(indexed, indexed, FAITHFUL_REF)
        self.assertEqual(status, 0, output)
        self.assertIn("1 of 6", output)
        self.assertIn("faithful", output)

    def test_a_partial_verdict_carries_the_plan_it_did_not_cover(self):
        """One of six compared is a measurement, but not the measurement the
        reader assumes -- so the same line says which."""
        names = [qf.TARGETS[0]]
        stem = names[0][: -len(".weight")]
        indexed = names + [stem + ".scales", stem + ".biases"]
        self.write_indices(indexed, indexed)
        _, output = self.run_main(indexed, indexed, FAITHFUL_REF)
        self.assertIn("5 target(s) were unavailable", output)

    def test_absent_index_files_are_refused_as_a_missing_input_not_a_crash(self):
        """The three files used to be opened at import, so the failure was a
        Traceback from wherever a caller happened to import it."""
        buffer = io.StringIO()
        original_cwd = os.getcwd()
        os.chdir(self.tmp)
        try:
            with contextlib.redirect_stderr(buffer):
                status = qf.main()
        finally:
            os.chdir(original_cwd)
        self.assertEqual(status, 2)
        self.assertIn("off_index.json", buffer.getvalue())
        self.assertNotIn("Traceback", buffer.getvalue())

    def test_a_suspect_tensor_fails(self):
        names = [qf.TARGETS[0]]
        stem = names[0][: -len(".weight")]
        indexed = names + [stem + ".scales", stem + ".biases"]
        self.write_indices(indexed, indexed)
        status, output = self.run_main(indexed, indexed, WRONG_REF)
        self.assertEqual(status, 1, output)
        self.assertIn("SUSPECT", output)

    def test_a_missing_scales_entry_is_skipped_not_a_traceback(self):
        """`mlx_idx[stem + ".scales"]` was a bare subscript while the weight name
        beside it was guarded, so a build that packs scales differently crashed the
        driver instead of reporting the target unavailable."""
        names = [qf.TARGETS[0]]
        self.write_indices(names, names)
        status, output = self.run_main(names, names, FAITHFUL_REF)
        self.assertEqual(status, 1, output)
        self.assertNotIn("Traceback", output)
        self.assertIn("NOT MEASURED", output)

    def test_the_target_list_is_printed_with_its_own_length(self):
        """The denominator comes from `TARGETS`, so a target renamed in the index
        shows up as unavailable rather than silently shrinking the run."""
        self.assertTrue(qf.TARGETS)
        self.write_indices([], [])
        _, output = self.run_main([], [], FAITHFUL_REF)
        self.assertIn(f"{len(qf.TARGETS)} target", output)


class ImportTests(unittest.TestCase):
    def test_importing_the_module_opens_no_file_and_downloads_nothing(self):
        """The regression this whole suite depends on: a module-scope `open()` or
        `urlopen()` means no caller can import the driver to check it."""
        program = (
            "import builtins, io, json, pathlib, sys, urllib.request\n"
            "sys.argv = ['tinytitan_quant_fidelity.py']\n"
            "real_open = builtins.open\n"
            "scratch = pathlib.Path(sys.argv[0] + '.guard')\n"
            "def guarded(path, *a, **k):\n"
            "    name = str(path)\n"
            "    if name.endswith(('.safetensors', 'index.json', 'config.json')):\n"
            "        raise RuntimeError('opened ' + name + ' at import')\n"
            "    return real_open(path, *a, **k)\n"
            "builtins.open = guarded\n"
            "def boom(*a, **k):\n"
            "    raise RuntimeError('downloaded at import')\n"
            "urllib.request.urlopen = boom\n"
            "import tinytitan_quant_fidelity\n"
            "print('imported clean')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", program],
            cwd=ROOT / "benchmark",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env={**os.environ, "PYTHONPATH": str(ROOT / "benchmark")},
        )
        self.assertEqual(result.returncode, 0, result.stdout[-500:])
        self.assertIn("imported clean", result.stdout)


if __name__ == "__main__":
    unittest.main()
