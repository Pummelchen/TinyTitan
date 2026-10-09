#!/usr/bin/env python3
"""Tests for benchmark/tinytitan_3bit_probe.py, driven model-free.

The probe decides "is 3-bit worth a repacker, a kernel and a re-download?", and
every number it printed to decide that was unreachable or absent. These tests
pin five defects in the driver as it stood, plus the five a mutation check then
caught in the first version of the fix:

1. The headline ratio divided by zero. `e4 = rel_error(w, requantize(w, 4))`
   where `w` IS the stored 4-bit tensor, so requantizing it to 4 bits is exact
   by construction -- `test_the_stored_grid_is_its_own_four_bit_reference`
   measures 0.0 -- and the ratio is `e3 / max(0.0, 1e-12)`. The pre-fix driver
   prints `median 3-bit/4-bit error ratio: 205600894987.58x` and a `4-bit`
   column of `0.0000`, so the docstring's rule ("an error several times 4-bit's
   is a strong signal") can never fire against any input. A ratio needs weights
   finer than the one under test, so the driver now takes a reference install
   through `TINYTITAN_BENCH_REFERENCE` and, without one, says `NOT MEASURED` and
   names the variable instead of dividing by its own floor.
2. The sample contained none of the weights the decision is about. The
   docstring says it quantizes "the real routed-expert and attention tensors";
   the pre-fix file read `model_weights.bin` and named six tensors, none of them
   a routed expert -- and on this repository's own MoE format the routed experts
   are not in that file at all (measured on an installed 125B-A6B MTP build:
   `model_weights.bin` is 51,681,440 bytes with 30 entries, of which the only
   ones named `expert` are the four `mlp.shared_expert*` projections, while 512
   experts per layer sit in `packed_experts/layer_00.bin` at 1,417,674,752
   bytes). The header comment compounds it: "Attention + shared-expert tensors
   ... plus the router", and no router tensor is sampled.
3. The width was assumed. `dequant4` treats every entry as two-nibbles-per-byte
   while the install's own manifest declares the router 8-bit. The repo's reader
   derives the width from the entry size and is what the fix reads through.
4. The second deciding question was never measured. The docstring asks "how much
   extra unpack work each weight costs"; the pre-fix table printed bytes only.
5. The status said nothing: `return 0` was the only return in `main()`, the
   guard discarded it, and the install path was a hardcoded constant, so an
   absent install was a `FileNotFoundError` traceback.

6. Five things the mutation check caught in the fix itself, each now pinned: a
   reference install stored at 4 bits has no 4-bit error of its own, and the
   first version of the message blamed a tensor that was there for one that was
   not; nothing asserted the `PROBE INCOMPLETE` line, so a run could return 1
   and still print the OK verdict; the dense-install test asserted only the
   words `NOT MEASURED`, which the always-declared router row satisfies on its
   own, so deleting the expert gap entirely still passed; a sample of zero
   tensors had no test at all; and `open_weights` named
   `TINYTITAN_BENCH_MODEL` for a bad `TINYTITAN_BENCH_REFERENCE`, sending the
   reader reinstalling the install that was fine.

No model is loaded and no network is touched: the driver reads synthetic
installs built in a temp directory in the format `tools/ssdai_reader.py`
documents.
"""

import io
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmark"
TOOLS = ROOT / "tools"
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(TOOLS))

import tinytitan_3bit_probe as probe  # noqa: E402

GROUP = 64
ROWS, COLS = 8, 256
HEADER_BYTES = 24
ENTRY_BYTES = 72
ENTRY = "<IHBBQQIIIIQQQQ"


def bf16_round(x):
    v = np.asarray(x, dtype=np.float32).view(np.int32)
    bits = ((v + 0x8000 + ((v >> 16) & 1)) >> 16).astype(np.uint32)
    return (bits << 16).view(np.float32)


def bf16_bytes(x):
    v = np.asarray(x, dtype=np.float32).view(np.uint32)
    bits = ((v + 0x8000 + ((v >> 16) & 1)) >> 16).astype(np.uint16)
    return bits.tobytes()


def grid_weights(seed, bits=4):
    """Float32 matrix that IS the dequantization of an exact `bits`-bit code."""
    rng = np.random.default_rng(seed)
    levels = (1 << bits) - 1
    groups = COLS // GROUP
    g = rng.normal(0.0, 0.05, (ROWS, groups, GROUP)).astype(np.float32)
    lo = g.min(-1, keepdims=True)
    hi = g.max(-1, keepdims=True)
    scale = bf16_round((hi - lo) / levels)
    bias = bf16_round(lo)
    q = np.clip(np.rint((g - bias) / np.where(scale == 0, 1.0, scale)), 0, levels)
    dense = (q * scale + bias).reshape(ROWS, COLS)
    flat = q.reshape(ROWS, COLS).astype(np.uint8)
    if bits == 4:
        payload = ((flat[:, 1::2] << 4) | flat[:, 0::2]).tobytes()
    else:
        payload = flat.tobytes()
    return dense, payload, scale.reshape(ROWS, groups), bias.reshape(ROWS, groups)


def write_install(
    directory: Path,
    names,
    bits=4,
    experts=0,
    model_id="synthetic-4bit",
    with_router=False,
    streaming=True,
):
    """A whole .ssdai install: resident index, manifest, packed experts."""
    directory.mkdir(parents=True, exist_ok=True)
    listed = list(names)
    if with_router:
        listed = listed + ["model.language_model.layers.0.mlp.gate.weight"]
    tensors = []
    for i, name in enumerate(listed):
        # the router is stored one byte per value while the 4-bit tensors pack
        # two, which is what the manifest declares and what the entry size says
        tensor_bits = 8 if name.endswith("mlp.gate.weight") else bits
        _dense, payload, scale, bias = grid_weights(seed=100 + i, bits=tensor_bits)
        tensors.append((name, payload, bf16_bytes(scale), bf16_bytes(bias)))

    entries = HEADER_BYTES + len(tensors) * ENTRY_BYTES
    strings = b""
    table = {}
    for name in [t[0] for t in tensors]:
        raw = name.encode()
        table[name] = (entries + len(strings), len(raw))
        strings += raw + b"\x00"
    index_size = entries + len(strings)

    cursor = index_size
    body = bytearray()
    blob = bytearray()
    for name, payload, scale_bytes, bias_bytes in tensors:
        file_off = cursor
        body += payload
        cursor += len(payload)
        scale_off = cursor
        body += scale_bytes
        cursor += len(scale_bytes)
        bias_off = cursor
        body += bias_bytes
        cursor += len(bias_bytes)
        name_off, name_len = table[name]
        blob += struct.pack(
            ENTRY,
            name_off,
            name_len,
            0,
            0,
            file_off,
            len(payload),
            ROWS,
            COLS,
            0,
            0,
            scale_off,
            len(scale_bytes),
            bias_off,
            len(bias_bytes),
        )
    out = bytearray()
    out += struct.pack("<QQQ", index_size, len(body), len(tensors))
    out += blob
    out += strings
    out += body
    (directory / "model_weights.bin").write_bytes(bytes(out))

    if experts:
        root = directory / "packed_experts"
        root.mkdir(parents=True, exist_ok=True)
        blob = bytearray()
        records = []
        for e in range(experts):
            base = len(blob)
            per = {}
            for j, tname in enumerate(("gate", "up", "down")):
                _dense, payload, scale, bias = grid_weights(seed=400 + e * 10 + j, bits=bits)
                per[tname] = {
                    "bits": bits,
                    "dtype": "U32",
                    "offset": len(blob) - base,
                    "size": len(payload),
                    "shape": [ROWS, COLS],
                }
                blob += payload
                for key, arr in ((f"{tname}_scales", scale), (f"{tname}_biases", bias)):
                    raw = bf16_bytes(arr)
                    per[key] = {
                        "dtype": "BF16",
                        "offset": len(blob) - base,
                        "size": len(raw),
                        "shape": [ROWS, COLS // GROUP],
                    }
                    blob += raw
            records.append({"expert": e, "offset": base, "size": len(blob) - base, "tensors": per})
        (root / "layer_00.bin").write_bytes(bytes(blob))
        (root / "layout.json").write_text(
            json.dumps(
                {
                    "expertsPerLayer": experts,
                    "expertStride": records[0]["size"],
                    "layers": [{"experts": records}],
                },
                indent=2,
            )
        )

    widths = {
        role: (8 if role == "router" else bits)
        for role in ("attention", "embedding", "routedExpert", "router", "sharedExpert")
    }
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "modelID": model_id,
                "numLayers": 20,
                "expertsPerLayer": experts or 1,
                "quant": {
                    role: {
                        "weightBits": w,
                        "groupSize": GROUP,
                        "scheme": "affine",
                        "scaleType": "BF16",
                        "biasType": "BF16",
                    }
                    for role, w in widths.items()
                },
                "flags": {"streamingPresent": bool(experts) and streaming},
            },
            indent=2,
        )
    )
    return directory


ATTN = "model.language_model.layers.3.self_attn.q_proj.weight"
ATTN2 = "model.language_model.layers.3.self_attn.o_proj.weight"
SHARED = "model.language_model.layers.3.mlp.shared_expert.gate_proj.weight"
SHARED2 = "model.language_model.layers.3.mlp.shared_expert.down_proj.weight"
RESIDENT = [ATTN, ATTN2, SHARED, SHARED2]


def run_probe(argv=(), *, env=None, model=None, reference=None, out=None, cwd=None):
    """Drive the real `main()` against synthetic installs."""
    calls = dict(env or {})
    base = {
        probe.BENCH_MODEL_ENV: str(model if model is not None else "/models/not-installed"),
        probe.OUT_ENV: str(out if out is not None else Path(tempfile.mkdtemp()) / "results"),
    }
    if reference is not None:
        base[probe.REFERENCE_ENV] = str(reference)
    elif probe.REFERENCE_ENV in calls:
        base[probe.REFERENCE_ENV] = calls.pop(probe.REFERENCE_ENV)
    base.update(calls)
    argv_backup = list(sys.argv)
    sys.argv = ["tinytitan_3bit_probe.py", *argv]
    buffer = io.StringIO()
    status = None
    escaped = None
    here = Path.cwd()
    try:
        if cwd is not None:
            os.chdir(cwd)
        with mock.patch.dict(os.environ, base, clear=False), redirect_stdout(buffer):
            status = probe.main()
    except SystemExit as exit_error:
        escaped = exit_error.code
        status = exit_error.code if isinstance(exit_error.code, int) else None
    finally:
        os.chdir(here)
        sys.argv = argv_backup
    return status, buffer.getvalue(), escaped


class FixtureTests(unittest.TestCase):
    """The installs the tests read are in the format the repo reader parses."""

    def test_the_repo_reader_reads_a_synthetic_install(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_install(Path(tmp) / "m", RESIDENT, experts=2)
            from ssdai_reader import PackedExperts, SSDAIWeights

            weights = SSDAIWeights(directory)
            self.assertIn(ATTN, weights.entries)
            tensor = weights.get(ATTN)
            self.assertEqual(tensor.shape, (ROWS, COLS))
            experts = PackedExperts(directory)
            self.assertEqual(experts.tensor(0, 0, "down").shape, (ROWS, COLS))

    def test_the_stored_grid_is_its_own_four_bit_reference(self):
        """Why the pre-fix ratio could only be `e3 / 1e-12`.

        A tensor already on the 4-bit grid requantizes to 4 bits exactly, so the
        denominator the pre-fix driver divided by is 0.0 for every tensor in
        every install, and `max(0.0, 1e-12)` is what the ratio really was.
        """
        dense, _payload, _scale, _bias = grid_weights(seed=17, bits=4)
        self.assertEqual(probe.requantize_error(dense, 4), 0.0)
        self.assertGreater(probe.requantize_error(dense, 3), 0.05)


class NamingTests(unittest.TestCase):
    def test_the_env_names_the_install_and_is_read_at_call_time(self):
        self.assertTrue(hasattr(probe, "probe_model"))
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_install(Path(tmp) / "m", RESIDENT)
            with mock.patch.dict(os.environ, {probe.BENCH_MODEL_ENV: str(directory)}, clear=False):
                self.assertEqual(probe.probe_model(), str(directory))

    def test_a_blank_variable_uses_the_default_install_when_it_is_there(self):
        """The fallback is the behavior; pin it without depending on this Mac.

        `DEFAULT_MODEL` names an install a development machine may or may not
        have, so the test puts a synthetic one there instead of hoping the real
        one is absent.
        """
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_install(
                Path(tmp) / "default", RESIDENT, experts=2, model_id="synthetic-default"
            )
            with (
                mock.patch.object(probe, "DEFAULT_MODEL", directory),
                mock.patch.dict(os.environ, {probe.BENCH_MODEL_ENV: ""}, clear=False),
            ):
                self.assertEqual(probe.probe_model(), str(directory))

    def test_a_blank_variable_with_no_default_installs_refuses_naming_the_variable(self):
        with mock.patch.object(probe, "DEFAULT_MODEL", Path("/models/nowhere-on-this-machine")):
            status, output, _ = run_probe(env={probe.BENCH_MODEL_ENV: ""}, model="")
            self.assertEqual(status, 2, output)
            self.assertIn("REFUSED", output)
            self.assertIn(probe.BENCH_MODEL_ENV, output)

    def test_a_missing_install_refuses_naming_the_variable_and_the_path(self):
        status, output, _ = run_probe(model="/models/no-such-install")
        self.assertEqual(status, 2)
        self.assertIn("REFUSED", output)
        self.assertIn(probe.BENCH_MODEL_ENV, output)
        self.assertIn("/models/no-such-install", output)

    def test_a_missing_reference_refuses_naming_the_variable_that_holds_it(self):
        """A wrong `TINYTITAN_BENCH_REFERENCE` must not be reported as a wrong
        model: the reader would go reinstalling the thing that was already fine."""
        with tempfile.TemporaryDirectory() as tmp:
            four = write_install(Path(tmp) / "four", RESIDENT, experts=2)
            status, output, _ = run_probe(model=four, reference=Path(tmp) / "no-such-reference")
            self.assertEqual(status, 2, output)
            self.assertIn("REFUSED", output)
            self.assertIn(probe.REFERENCE_ENV, output)
            self.assertNotIn(f"{probe.BENCH_MODEL_ENV}=", output)

    def test_an_install_without_a_manifest_refuses_with_the_file_it_needed(self):
        """Name the missing file, not whatever the reader crashed on.

        Without this check the refusal still comes back as a wrapped
        `FileNotFoundError`, whose message happens to contain the string, so the
        assertion has to be the driver's own sentence rather than a substring
        that a stack trace also carries.
        """
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_install(Path(tmp) / "m", RESIDENT)
            (directory / "manifest.json").unlink()
            status, output, _ = run_probe(model=directory)
            self.assertEqual(status, 2)
            self.assertIn("has no manifest.json", output)


class SampleTests(unittest.TestCase):
    def test_routed_experts_are_sampled_when_the_install_streams_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_install(Path(tmp) / "m", RESIDENT, experts=3)
            counts = probe.sample_counts(probe.sample_tensors(directory))
            self.assertGreaterEqual(counts.get("routedExpert", 0), 1)
            self.assertGreaterEqual(counts.get("attention", 0), 1)
            self.assertGreaterEqual(counts.get("sharedExpert", 0), 1)

    def test_the_sample_labels_each_tensor_with_the_role_the_manifest_declares(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_install(Path(tmp) / "m", RESIDENT, experts=2)
            roles = {t.role for t in probe.sample_tensors(directory)}
            self.assertIn("routedExpert", roles)
            self.assertIn("attention", roles)

    def test_a_router_tensor_is_read_at_the_width_the_manifest_declares(self):
        """The pre-fix `dequant4` assumed two nibbles per byte for every entry.

        The install declares the router 8-bit, so its payload is one byte per
        value; reading it as 4-bit is not a crash, it is a wrong shape, and the
        width has to come from the entry.
        """
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_install(Path(tmp) / "m", RESIDENT, with_router=True)
            tensors = probe.sample_tensors(directory)
            router = [t for t in tensors if t.role == "router"]
            self.assertEqual(len(router), 1)
            self.assertEqual(router[0].array.shape, (ROWS, COLS))

    def test_a_dense_install_reports_the_expert_role_as_not_measured(self):
        """The named gap has to be the role that was missing.

        Every synthetic manifest declares the router too, so a run without a
        router tensor also prints `NOT MEASURED: router`; an assertion that only
        looks for the words would pass on that line and say nothing about the
        experts. The install here carries a router, so the only gap left is the
        streamed one.
        """
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_install(Path(tmp) / "m", RESIDENT, experts=0, with_router=True)
            status, output, _ = run_probe(model=directory)
            self.assertEqual(status, 1, output)
            self.assertIn("NOT MEASURED: routedExpert", output)

    def test_an_install_whose_tensors_match_no_role_measures_nothing(self):
        """A sample of zero tensors is not a run that measured every role."""
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_install(Path(tmp) / "m", ["model.language_model.embed_tokens.weight"])
            self.assertEqual(probe.sample_tensors(directory), [])
            status, output, _ = run_probe(model=directory)
            self.assertEqual(status, 1, output)
            self.assertIn("NOT MEASURED: every role", output)

    def test_the_header_names_the_install_and_the_declared_widths(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_install(
                Path(tmp) / "m", RESIDENT, experts=2, model_id="synthetic-probe-8"
            )
            status, output, _ = run_probe(model=directory)
            self.assertIn("synthetic-probe-8", output)
            self.assertIn("routedExpert", output)
            self.assertNotEqual(status, 2)


class RatioTests(unittest.TestCase):
    def test_no_ratio_is_printed_without_a_wider_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_install(Path(tmp) / "m", RESIDENT, experts=2)
            status, output, _ = run_probe(model=directory)
            # every ratio the run prints must be a number of its stated size,
            # not `e3 / 1e-12`: the pre-fix driver printed 207836508750.92x
            printed = [float(m.group(1)) for m in re.finditer(r"(\d{1,6}\.\d+)x", output)]
            for value in printed:
                self.assertLess(value, 10.0, output)
            self.assertIn("NOT MEASURED", output)
            self.assertIn(probe.REFERENCE_ENV, output)
            self.assertEqual(status, 1)

    def test_the_added_error_is_still_reported_without_a_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_install(Path(tmp) / "m", RESIDENT, experts=2)
            _status, output, _ = run_probe(model=directory)
            self.assertIn("3-bit added", output)
            self.assertIn("2-bit added", output)

    def test_a_reference_install_gives_a_ratio_that_is_a_number_of_its_size(self):
        """3-bit against 4-bit on weights finer than either: about 2x, not 2e11x."""
        with tempfile.TemporaryDirectory() as tmp:
            four = write_install(Path(tmp) / "four", RESIDENT, experts=2, bits=4, with_router=True)
            eight = write_install(
                Path(tmp) / "eight",
                RESIDENT,
                experts=2,
                bits=8,
                model_id="synthetic-8bit",
                with_router=True,
            )
            status, output, _ = run_probe(model=four, reference=eight)
            self.assertEqual(status, 0, output)
            ratios = [
                float(m.group(1))
                for m in re.finditer(
                    r"^(?:routedExpert|sharedExpert|attention|router)\b.*?(\d+\.\d+)x$",
                    output,
                    re.M,
                )
            ]
            self.assertTrue(ratios, output)
            for ratio in ratios:
                self.assertGreater(ratio, 1.2, output)
                self.assertLess(ratio, 6.0, output)

    def test_a_reference_stored_at_the_same_width_offers_no_ratio(self):
        """The exact shape that produced the pre-fix 207836508750.92x headline.

        `TINYTITAN_BENCH_REFERENCE` pointing at another 4-bit install finds a
        twin for every tensor, so nothing is missing -- but a 4-bit twin's own
        requantize-to-4 error is 0.0, and dividing by it is the bug. The run has
        to say the reference is not finer instead of printing the quotient.
        """
        with tempfile.TemporaryDirectory() as tmp:
            four = write_install(Path(tmp) / "four", RESIDENT, experts=2, with_router=True)
            also_four = write_install(
                Path(tmp) / "other",
                RESIDENT,
                experts=2,
                bits=4,
                model_id="synthetic-4bit-2",
                with_router=True,
            )
            status, output, _ = run_probe(model=four, reference=also_four)
            printed = [float(m.group(1)) for m in re.finditer(r"(\d{1,6}\.\d+)x", output)]
            for value in printed:
                self.assertLess(value, 10.0, output)
            self.assertIn("NOT MEASURED", output)
            # name the real gap. Both readings end in "...that had a finer twin",
            # so the word `finer` alone does not distinguish a tensor that is
            # absent from one that is there at the wrong width.
            self.assertIn("no finer than its own 4-bit grid", output)
            self.assertNotIn("no twin there at all", output)
            self.assertEqual(status, 1, output)

    def test_a_reference_that_names_no_common_tensor_says_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            four = write_install(Path(tmp) / "four", RESIDENT, experts=2)
            other = write_install(
                Path(tmp) / "other",
                ["model.language_model.layers.7.self_attn.q_proj.weight"],
                bits=8,
                model_id="different",
            )
            status, output, _ = run_probe(model=four, reference=other)
            self.assertEqual(status, 1, output)
            self.assertIn("NOT MEASURED", output)
            self.assertIn("no twin there at all", output)
            self.assertNotIn("no finer than its own 4-bit grid", output)


class PackingTests(unittest.TestCase):
    def test_the_packing_section_answers_the_unpack_question(self):
        rows = {r["bits"]: r for r in probe.packing_analysis()}
        self.assertGreater(rows[3]["crossing_values"], 0)
        self.assertEqual(rows[4]["crossing_values"], 0)
        self.assertEqual(rows[2]["crossing_values"], 0)
        self.assertGreater(rows[3]["extra_loads_per_weight"], 0.0)

    def test_the_byte_arithmetic_the_docstring_quotes_is_unchanged(self):
        rows = {r["bits"]: r for r in probe.packing_analysis()}
        self.assertEqual(rows[3]["group_bytes_stream"], 28.0)
        self.assertEqual(rows[3]["group_bytes_padded"], 32.0)
        self.assertAlmostEqual(rows[3]["vs_4bit_stream"], 28.0 / 36.0, places=6)

    def test_the_printed_table_carries_the_unpack_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_install(Path(tmp) / "m", RESIDENT, experts=2)
            _status, output, _ = run_probe(model=directory)
            self.assertIn("cross", output.lower())
            self.assertIn("loads", output.lower())


class RoleReportingTests(unittest.TestCase):
    def test_a_role_the_manifest_does_not_declare_is_reported_without_a_penalty(self):
        """A dense checkpoint is not a measurement gap about routed experts."""
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_install(
                Path(tmp) / "m", [ATTN, SHARED], experts=0, model_id="dense", with_router=True
            )
            manifest = json.loads((directory / "manifest.json").read_text())
            del manifest["quant"]["routedExpert"]
            manifest["expertsPerLayer"] = 1
            (directory / "manifest.json").write_text(json.dumps(manifest))
            status, output, _ = run_probe(model=directory)
            self.assertIn("not in this checkpoint", output)
            self.assertNotIn("NOT MEASURED: routedExpert", output)
            self.assertNotEqual(status, 2)

    def test_a_layout_that_names_a_missing_layer_file_is_refused_by_name(self):
        """Half an expert store is an install fault, not a shorter table."""
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_install(Path(tmp) / "m", [ATTN, SHARED], experts=4)
            (directory / "packed_experts" / "layer_00.bin").unlink()
            status, output, _ = run_probe(model=directory)
            self.assertEqual(status, 2, output)
            self.assertIn("REFUSED", output)
            self.assertIn("layer_00.bin", output)


class ProseTests(unittest.TestCase):
    def test_the_docstring_promises_only_what_the_run_prints(self):
        doc = probe.__doc__ or ""
        self.assertIn("3-bit", doc)
        # the docstring's two deciding questions must both name a thing the run
        # measures, and must not claim a comparison it cannot make
        self.assertNotIn("against 4-bit on the same tensors", doc)
        for claim in ("routed expert", "attention"):
            self.assertIn(claim, doc.lower())

    def test_the_module_carries_no_stale_sample_comment(self):
        source = (BENCH / "tinytitan_3bit_probe.py").read_text()
        self.assertNotIn("plus the router", source)


class StatusTests(unittest.TestCase):
    def test_the_guard_exits_with_the_status(self):
        source = (BENCH / "tinytitan_3bit_probe.py").read_text()
        self.assertIn("sys.exit(main())", source)
        self.assertNotIn("\n    main()\n", source)

    def test_main_is_declared_to_return_a_status(self):
        source = (BENCH / "tinytitan_3bit_probe.py").read_text()
        self.assertIn("def main() -> int:", source)
        self.assertNotIn("    return 0\n\n\nif __name__", source)

    def test_a_run_that_measured_everything_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            four = write_install(Path(tmp) / "four", RESIDENT, experts=2, with_router=True)
            eight = write_install(
                Path(tmp) / "eight", RESIDENT, experts=2, bits=8, with_router=True
            )
            status, output, _ = run_probe(model=four, reference=eight)
            self.assertEqual(status, 0, output)
            self.assertIn("PROBE OK", output)
            self.assertNotIn("NOT MEASURED", output)

    def test_a_run_that_missed_a_number_prints_the_incomplete_verdict(self):
        """The line and the status are one claim, not two."""
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_install(Path(tmp) / "m", RESIDENT, experts=2)
            status, output, _ = run_probe(model=directory)
            self.assertEqual(status, 1, output)
            self.assertIn("PROBE INCOMPLETE", output)
            self.assertNotIn("PROBE OK", output)

    def test_nothing_is_written_into_the_benchmark_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_install(Path(tmp) / "m", RESIDENT, experts=2)
            out = Path(tmp) / "results"
            before = {p.name for p in BENCH.iterdir()}
            run_probe(model=directory, out=out, cwd=str(tmp))
            self.assertEqual({p.name for p in BENCH.iterdir()}, before)
            self.assertTrue(out.is_dir(), "the artifact directory was never created")
            self.assertEqual([p.name for p in out.iterdir()], ["3bit-probe.json"])

    def test_the_artifact_names_the_install_it_measured(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_install(
                Path(tmp) / "m", RESIDENT, experts=2, model_id="synthetic-labeled"
            )
            out = Path(tmp) / "results"
            run_probe(model=directory, out=out)
            payload = json.loads((out / "3bit-probe.json").read_text())
            self.assertEqual(payload["model_id"], "synthetic-labeled")
            self.assertIn("routedExpert", payload["roles"])
            self.assertIn("packing", payload)


class ImportTests(unittest.TestCase):
    def test_importing_the_module_opens_no_file_and_starts_nothing(self):
        child = r"""
import builtins, sys, pathlib
from unittest import mock

def refuse(*a, **k):
    raise AssertionError("import touched the filesystem: %r" % (a[:1],))

sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
with mock.patch.object(builtins, "open", new=refuse):
    import tinytitan_3bit_probe
print("imported clean")
"""
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "child.py"
            script.write_text(child)
            result = subprocess.run(
                [sys.executable, str(script), str(BENCH), str(TOOLS)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("imported clean", result.stdout)

    def test_the_configured_constants_are_not_read_at_import(self):
        source = (BENCH / "tinytitan_3bit_probe.py").read_text()
        for line in source.splitlines():
            if "os.environ" in line and not line.startswith((" ", "\t", "#")):
                if "ENV" in line and "=" in line and '"' in line and "os.environ.get" not in line:
                    continue
                self.fail(f"environment read at import scope: {line!r}")


if __name__ == "__main__":
    unittest.main()
