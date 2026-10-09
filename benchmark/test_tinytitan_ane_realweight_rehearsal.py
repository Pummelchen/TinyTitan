#!/usr/bin/env python3
"""Tests for benchmark/tinytitan_ane_realweight_rehearsal.py, driven model-free.

The rehearsal is the step between the probe ("is one full-attention block fast on
the ANE?") and integration ("does a real 6,103-token prefill get faster, on the
real weights?"). It could not answer that, and nothing it printed could tell you.
These tests pin what the pre-fix driver did.

1. The verdict could not fail. `def main() -> int` at :105 ends on `return 0`
   (:197) as its only return and the guard calls bare `main()` at :201, while the
   two things it checks -- `worst per-layer rel err` at :183 and `total nan/inf`
   at :183 -- are computed, printed and never compared to anything. A run whose
   every block output is NaN prints the count and exits 0.
2. The layer list is a hardcoded guess about somebody else's model. `FULL_LAYERS =
   list(range(3, 40, 4))` at :31 restates a rule the install's own manifest
   records: `arch.fullAttentionLayerMask` (0 = sliding window, 1 = full, 2 =
   linear, per `sources/TinyTitanFormat/SSDAIManifestV1.swift:34`), which the CPU
   engine itself reads back into an interval at
   `AffineSnapshot+Loading.swift:211`. Measured on the only 4-bit install on this
   host: the mask is 48 long with 12 ones, at layers 3, 7 ... 47 -- so the
   hardcoded ten skip two layers, while the footer divides a reference recorded
   over ten into the total and calls the run "10 full-attention layers".
3. The install is a frozen constant and it is not here. `MODEL_BIN` at :30 names
   `models/ornith-1.5_35B_A3B_4Bit/model_weights.bin` at import, no
   `TINYTITAN_BENCH_MODEL` is read anywhere, and `ls models/` on this host holds
   the two qwen3.8 installs only -- the shipped command on the pinned venv ends
   `FileNotFoundError: .../ornith-1.5_35B_A3B_4Bit/model_weights.bin` (measured).
4. The tensor names are a guess about a different family. The prefix at :90 is
   `language_model.model.layers.N.self_attn.` and `tools/export_ane_prefill.py:141`
   records which family uses which stem -- that one for qwen36 and the dense
   builds, `model.language_model.layers.N.self_attn.` for the 3.8 one. Re-pointed
   at the installed 3.8 build by hand the driver dies `KeyError:
   'language_model.model.layers.3.self_attn.q_proj.weight'` (measured): a traceback
   naming neither the prefix it searched nor the install it searched. The same
   install holds no `q_norm`/`k_norm` at all (measured: 0 entries of 1,079), so a
   missing role is a bare KeyError too.
5. The width is assumed. `load_tensor` at :67 branches on `dtype == 1` (bf16) and
   parses every other entry as two nibbles per byte, while the repository's own
   reader infers width from the payload size (`tools/ssdai_reader.py:106` -- "Bit
   width is not recorded per tensor; the payload size states it") and this build's
   manifest declares `router` and `embedding` at 8 bits. Measured: 98 of 1,079
   entries say 8 bits by their own size, and feeding one -- the layer-0 router,
   shape (512, 2560), 1,310,720 bytes -- to the pre-fix `load_tensor` gives
   `ValueError: cannot reshape array of size 1310720 into shape (512, 1280)`. The
   fix reads through `ssdai_reader.SSDAIWeights`, so the entry layout, the bf16
   path and the width inference live in one place again.
6. One raise loses everything: nothing wraps `build_block`, `predict` or
   `reference`, and the artifact is written once after the whole sweep (:184), so
   the layer that fails takes the artifact, and the rows already measured, with it.
7. The block output is picked by an inline `next()` over shapes (:131) -- the code
   AUD-237 replaced in the probe with `block_output_name()`. Here a program with no
   `(chunk, D)` output raises `StopIteration` and says nothing about the shapes it
   did find.
8. The artifact path is `ROOT/.build/benchmark-results/...` with no `mkdir` (:184),
   so a fresh checkout measures the whole rehearsal and then dies with
   FileNotFoundError, and `json.dump` writes a literal `NaN` for a non-finite
   `rel_err`, which is not RFC 8259 JSON.
9. `GPU_REFERENCE_S = 84.3` at :33 restates a number the probe already names
   (`probe.GPU_LAYER_SECONDS`), and `remainder = 133.2 - GPU_REFERENCE_S` at :176
   prints a projected end-to-end multiplier off two figures whose only source is a
   comment -- while the probe's docstring says the go/no-go has to clear the
   per-chunk cost "by enough to survive integration overheads", so a projection
   with no overhead in it is a ceiling, not a result.

No Core ML program is converted and no model is loaded: `coremltools` is faked in
`sys.modules` before the driver is imported, `build_block` / `reference` /
`rope_tables` / `causal_mask` are patched, and the installs are synthetic `.ssdai`
trees written in a temp directory in the format `tools/ssdai_reader.py` documents.
The probe's geometry constants are patched down to a small set whose column
counts are multiples of 64, so group-64 affine quantization is exercised for real
without a 16 MB tensor per role.
"""

import io
import itertools
import json
import pathlib
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmark"
TOOLS = ROOT / "tools"
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(TOOLS))

FAKED = [
    "coremltools",
    "coremltools.converters",
    "coremltools.converters.mil",
    "coremltools.converters.mil.mil",
    "coremltools.converters.mil.mil.types",
]
for _name in FAKED:
    sys.modules.setdefault(_name, mock.MagicMock(name=_name))

import tinytitan_ane_attention_probe as probe  # noqa: E402
import tinytitan_ane_realweight_rehearsal as rehearsal  # noqa: E402
from ssdai_reader import SSDAIWeights  # noqa: E402

GROUP = 64
HEADER_BYTES = 24
ENTRY_BYTES = 72
ENTRY = "<IHBBQQIIIIQQQQ"
DTYPE_U32, DTYPE_BF16 = 0, 1

# A geometry small enough to hold real tensors, related exactly as the probe's
# constants are: Q_PROJ_ROWS = 2 * N_Q_HEADS * HEAD_DIM, KV_DIM = N_KV_HEADS *
# HEAD_DIM, and every quantized column count a multiple of 64.
G = {
    "D": 64,
    "N_Q_HEADS": 2,
    "N_KV_HEADS": 2,
    "HEAD_DIM": 32,
    "Q_DIM": 64,
    "Q_PROJ_ROWS": 128,
    "KV_DIM": 64,
}
PROJECTION = {"wq": (G["Q_PROJ_ROWS"], G["D"]), "wk": (G["KV_DIM"], G["D"])}
ROLE_SHAPE = {
    "q_proj.weight": (G["Q_PROJ_ROWS"], G["D"]),
    "k_proj.weight": (G["KV_DIM"], G["D"]),
    "v_proj.weight": (G["KV_DIM"], G["D"]),
    "o_proj.weight": (G["D"], G["Q_DIM"]),
}
NORM_SHAPE = (G["HEAD_DIM"],)
STEM38 = "model.language_model.layers"
STEM36 = "language_model.model.layers"
ROLES = tuple(ROLE_SHAPE) + ("q_norm.weight", "k_norm.weight")
# `run_main` patches CHUNKS down to these two rows, so a block that echoes has to
# declare an output at both shapes.
RUN_CHUNKS = (4, 2)


def bf16_round(values) -> np.ndarray:
    """The float32 values a bf16 field actually holds."""
    raw = bf16_bytes(values)
    wide = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    return wide.view(np.float32)


def bf16_bytes(values) -> bytes:
    v = np.asarray(values, dtype=np.float32).view(np.uint32)
    bits = ((v + 0x8000 + ((v >> 16) & 1)) >> 16).astype(np.uint16)
    return bits.tobytes()


def temp_dir() -> pathlib.Path:
    return pathlib.Path(tempfile.mkdtemp())


class Install:
    """A synthetic .ssdai: resident index, string table, payload, manifest.json."""

    def __init__(self, directory: pathlib.Path):
        self.directory = directory
        self.tensors = []
        self.values = {}

    def quant(self, name, shape, bits=4, seed=0):
        rows, cols = shape
        if cols % GROUP:
            # A fixture authoring error rather than a test assertion: a width
            # that is not whole groups has no quantization to write.
            raise ValueError(f"{name} shape {shape}: {cols} is not a multiple of {GROUP}")
        levels = (1 << bits) - 1
        codes = np.random.default_rng(seed).integers(0, levels + 1, (rows, cols)).astype(np.uint8)
        # scale 1.0 and bias 0.0 per group, so the dequantized tensor is the
        # code grid itself and a test can assert the loaded values
        ones = np.ones((rows, cols // GROUP), dtype=np.float32)
        zeros = np.zeros((rows, cols // GROUP), dtype=np.float32)
        if bits == 4:
            payload = ((codes[..., 1::2] << 4) | codes[..., 0::2]).tobytes()
        else:
            payload = codes.tobytes()
        self.tensors.append((name, shape, DTYPE_U32, payload, bf16_bytes(ones), bf16_bytes(zeros)))
        self.values[name] = codes.astype(np.float32)

    def bf16(self, name, shape, seed=0):
        values = np.random.default_rng(seed).standard_normal(int(np.prod(shape))).astype(np.float32)
        held = bf16_round(values).reshape(shape)
        self.tensors.append((name, shape, DTYPE_BF16, bf16_bytes(values), b"", b""))
        self.values[name] = held

    def write(self, mask, model_id="synthetic-35b-a3b"):
        num_layers, full = mask
        tensors = self.tensors
        entries_at = HEADER_BYTES + len(tensors) * ENTRY_BYTES
        strings = b""
        table = {}
        for name, *_rest in tensors:
            raw = name.encode()
            table[name] = (entries_at + len(strings), len(raw))
            strings += raw + b"\0"
        index_size = entries_at + len(strings)
        cursor = index_size
        body = bytearray()
        blob = bytearray()
        for name, shape, dtype, payload, scale_bytes, bias_bytes in tensors:
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
            dims = list(shape) + [0] * (4 - len(shape))
            blob += struct.pack(
                ENTRY,
                name_off,
                name_len,
                dtype,
                0,
                file_off,
                len(payload),
                *dims,
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
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "model_weights.bin").write_bytes(bytes(out))
        arch = {
            "numLayers": num_layers,
            "fullAttentionLayerMask": full,
            "hiddenSize": G["D"],
            "numHeads": G["N_Q_HEADS"],
            "numKVHeads": G["N_KV_HEADS"],
            "headDim": G["HEAD_DIM"],
        }
        (self.directory / "manifest.json").write_text(
            json.dumps({"modelID": model_id, "numLayers": num_layers, "arch": arch})
        )
        return self.directory


def build_install(
    directory: pathlib.Path,
    stem=STEM38,
    layers=(3, 7),
    num_layers=None,
    roles=ROLES,
    bits=4,
    full=None,
) -> Install:
    """Every role of every full-attention layer, plus the mask that declares them."""
    install = Install(directory)
    num_layers = num_layers or (max(layers) + 1)
    if full is None:
        full = [0] * num_layers
        for layer in layers:
            full[layer] = 1
    for layer in layers:
        for role in roles:
            name = f"{stem}.{layer}.self_attn.{role}"
            if role.endswith("_norm.weight"):
                install.bf16(name, NORM_SHAPE, seed=layer * 7)
            else:
                install.quant(name, ROLE_SHAPE[role], bits=bits, seed=layer * 11)
    install.write((num_layers, full))
    return install


class FakeModel:
    """A converted block: named outputs with shapes, and an answering predict."""

    def __init__(self, out_name="block_output", array=None, shapes=None):
        self.out_name = out_name
        self.array = array
        self.shapes = shapes if shapes is not None else self._defaults()
        self.calls = 0

    def _defaults(self):
        if self.array is not None:
            return [(self.out_name, tuple(self.array.shape))]
        # An echoing block answers at every chunk shape the rehearsal runs, so
        # the driver's lookup by shape finds one at each row.
        named = [(self.out_name, RUN_CHUNKS[0])]
        named += [(f"{self.out_name}_{chunk}", chunk) for chunk in RUN_CHUNKS[1:]]
        return [(name, (chunk, G["D"])) for name, chunk in named]

    def get_spec(self):
        outputs = [
            types.SimpleNamespace(
                name=name,
                type=types.SimpleNamespace(multiArrayType=types.SimpleNamespace(shape=list(shape))),
            )
            for name, shape in self.shapes
        ]
        return types.SimpleNamespace(description=types.SimpleNamespace(output=outputs))

    def answer(self, feed):
        hidden = np.asarray(feed["hidden"], dtype=np.float32)
        return {name: (hidden if self.array is None else self.array) for name, _ in self.shapes}

    def predict(self, feed, *args, **kwargs):
        self.calls += 1
        return self.answer(feed)


def echo_reference(hidden, k_hist, v_hist, cos_t, sin_t, mask, weights):
    return np.asarray(hidden, dtype=np.float32)


def scaled_reference(factor):
    def reference(hidden, k_hist, v_hist, cos_t, sin_t, mask, weights):
        return np.asarray(hidden, dtype=np.float32) * factor

    return reference


def small_rope(start, count):
    half = G["HEAD_DIM"] // 4
    return (
        np.zeros((count, half), dtype=np.float16),
        np.zeros((count, half), dtype=np.float16),
    )


def small_mask(t, history):
    return np.zeros((1, 1, t, history + t), dtype=np.float16)


def clock(deltas):
    """A `perf_counter` that yields exactly these per-predict deltas.

    Each pair starts at zero so the subtraction returns the delta the test named
    rather than its float drift, and a list describes one row (its warm-up plus
    its timed repeats), so it cycles: a short list still covers a run of several
    rows.
    """
    values = []
    for delta in deltas:
        values += [0.0, delta]
    ticks = itertools.cycle(values)

    def counter():
        return next(ticks)

    return counter


def geometry_patches():
    """Patch the probe's geometry down to the one these installs are built at."""
    return [mock.patch.object(probe, name, value) for name, value in G.items()]


def run_main(
    directory, model=None, reference=echo_reference, deltas=None, env=None, build_error=None
):
    """Drive the real main() with Core ML stubbed and the artifact in a temp dir."""
    harness = model if model is not None else FakeModel()
    out_dir = temp_dir() / "results"
    # `clear=True` empties the environment for the run, so the output variable is
    # absent unless a test names it -- patch.dict cannot delete a key it is given.
    mapping = {"TINYTITAN_BENCH_MODEL": str(directory)}
    if env is not None:
        mapping = {k: v for k, v in env.items() if v is not None}
    patches = geometry_patches()
    if build_error is None:
        patches.append(mock.patch.object(probe, "build_block", return_value=harness))
    else:
        # a test that wants the conversion to fail has to ask here: patching
        # build_block around run_main would just be shadowed by this one
        patches.append(
            mock.patch.object(probe, "build_block", side_effect=RuntimeError(build_error))
        )
    patches += [
        mock.patch.object(probe, "reference", side_effect=reference),
        mock.patch.object(probe, "rope_tables", side_effect=small_rope),
        mock.patch.object(probe, "causal_mask", side_effect=small_mask),
        mock.patch.object(rehearsal, "CHUNKS", [(4, 0), (2, 4)]),
        mock.patch.object(rehearsal, "DEFAULT_OUT", out_dir / rehearsal.ARTIFACT),
    ]
    if deltas is not None:
        patches.append(mock.patch.object(probe.time, "perf_counter", clock(deltas)))
    buffer = io.StringIO()
    with mock.patch.dict(rehearsal.os.environ, mapping, clear=True):
        for patcher in patches:
            patcher.start()
        try:
            with redirect_stdout(buffer):
                status = rehearsal.main()
        finally:
            for patcher in patches:
                patcher.stop()
    printed = buffer.getvalue()
    artifact = out_dir / rehearsal.ARTIFACT
    payload = json.loads(artifact.read_text()) if artifact.exists() else None
    return status, printed, harness, payload


class CleanRun(unittest.TestCase):
    """One harness run shared by the assertions about a rehearsal that worked."""

    def setUp(self):
        self.dir = temp_dir()
        build_install(self.dir)
        (self.status, self.printed, self.model, self.payload) = run_main(
            self.dir, deltas=[5.0, 0.10, 0.20, 0.30]
        )


class ModelDirectoryTests(unittest.TestCase):
    def test_the_install_is_named_at_call_time_not_at_import(self):
        directory = temp_dir() / "some-install"
        with mock.patch.dict(rehearsal.os.environ, {"TINYTITAN_BENCH_MODEL": str(directory)}):
            self.assertEqual(rehearsal.model_directory(), directory)

    def test_a_blank_variable_falls_back_to_the_documented_default(self):
        with mock.patch.dict(rehearsal.os.environ, {"TINYTITAN_BENCH_MODEL": "   "}):
            self.assertEqual(rehearsal.model_directory().name, rehearsal.DEFAULT_MODEL.name)

    def test_the_default_names_an_install_directory_not_a_file_inside_it(self):
        # the reader takes a directory; the pre-fix constant took the .bin inside
        # one, which is why nothing could say which install it was looking for
        self.assertFalse(str(rehearsal.DEFAULT_MODEL).endswith(".bin"))
        self.assertEqual(rehearsal.DEFAULT_MODEL.name, "ornith-1.5_35B_A3B_4Bit")

    def test_a_missing_install_is_a_refusal_naming_the_variable_and_the_path(self):
        absent = temp_dir() / "no-such-install"
        with mock.patch.dict(rehearsal.os.environ, {"TINYTITAN_BENCH_MODEL": str(absent)}):
            with self.assertRaises(rehearsal.RehearsalError) as caught:
                rehearsal.open_weights(rehearsal.model_directory())
        message = str(caught.exception)
        self.assertIn("TINYTITAN_BENCH_MODEL", message)
        self.assertIn(str(absent), message)

    def test_an_absent_install_exits_2_without_a_traceback(self):
        absent = temp_dir() / "no-such-install"
        status, printed, _model, _payload = run_main(absent)
        self.assertEqual(status, 2)
        self.assertIn("REFUSED", printed)
        self.assertNotIn("Traceback", printed)

    def test_a_directory_with_no_resident_index_refuses_naming_the_file(self):
        hollow = temp_dir() / "hollow"
        hollow.mkdir()
        with mock.patch.dict(rehearsal.os.environ, {"TINYTITAN_BENCH_MODEL": str(hollow)}):
            with self.assertRaises(rehearsal.RehearsalError) as caught:
                rehearsal.open_weights(rehearsal.model_directory())
        self.assertIn("model_weights.bin", str(caught.exception))


class FullAttentionLayerTests(unittest.TestCase):
    def test_the_layers_come_from_the_manifest_mask_not_a_hardcoded_range(self):
        directory = temp_dir()
        build_install(directory, layers=[3, 7])
        self.assertEqual(rehearsal.full_attention_layers(SSDAIWeights(directory)), [3, 7])

    def test_a_mask_with_twelve_ones_gives_twelve_layers(self):
        # measured on the only 4-bit install here: 48 entries, ones at 3..47 step
        # 4, which is 12 full-attention layers where the pre-fix list held ten
        directory = temp_dir()
        build_install(directory, layers=list(range(3, 48, 4)), num_layers=48)
        self.assertEqual(len(rehearsal.full_attention_layers(SSDAIWeights(directory))), 12)

    def test_a_mask_of_only_linear_layers_is_refused_not_an_empty_sweep(self):
        directory = temp_dir()
        build_install(directory, layers=[3], full=[2, 2, 2, 2])
        with self.assertRaises(rehearsal.RehearsalError) as caught:
            rehearsal.full_attention_layers(SSDAIWeights(directory))
        self.assertIn("fullAttentionLayerMask", str(caught.exception))

    def test_a_missing_mask_key_names_the_key_it_wanted(self):
        directory = temp_dir()
        build_install(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        del manifest["arch"]["fullAttentionLayerMask"]
        (directory / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaises(rehearsal.RehearsalError) as caught:
            rehearsal.full_attention_layers(SSDAIWeights(directory))
        self.assertIn("fullAttentionLayerMask", str(caught.exception))

    def test_a_mask_shorter_than_num_layers_names_both_counts(self):
        directory = temp_dir()
        build_install(directory, layers=[3], num_layers=8)
        manifest = json.loads((directory / "manifest.json").read_text())
        manifest["arch"]["fullAttentionLayerMask"] = [0, 0, 0, 1]
        (directory / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaises(rehearsal.RehearsalError) as caught:
            rehearsal.full_attention_layers(SSDAIWeights(directory))
        message = str(caught.exception)
        self.assertIn("4", message)
        self.assertIn("8", message)


class LayerNameTests(unittest.TestCase):
    def test_the_3_8_family_stem_is_found(self):
        directory = temp_dir()
        build_install(directory, stem=STEM38)
        self.assertEqual(rehearsal.layer_stem(SSDAIWeights(directory), 3), f"{STEM38}.3")

    def test_the_qwen36_stem_is_found_too(self):
        directory = temp_dir()
        build_install(directory, stem=STEM36)
        self.assertEqual(rehearsal.layer_stem(SSDAIWeights(directory), 3), f"{STEM36}.3")

    def test_a_layer_with_no_attention_tensors_names_the_layer_and_the_suffix(self):
        directory = temp_dir()
        build_install(directory, layers=[3, 7])
        with self.assertRaises(rehearsal.RehearsalError) as caught:
            rehearsal.layer_stem(SSDAIWeights(directory), 11)
        message = str(caught.exception)
        self.assertIn("11", message)
        self.assertIn("self_attn.q_proj.weight", message)
        self.assertNotIn("Traceback", message)

    def test_a_missing_role_names_the_role_and_the_stem(self):
        directory = temp_dir()
        build_install(directory, roles=("q_proj.weight", "k_proj.weight", "v_proj.weight"))
        with self.assertRaises(rehearsal.RehearsalError) as caught:
            rehearsal.load_layer_weights(SSDAIWeights(directory), 3)
        message = str(caught.exception)
        self.assertIn("o_proj.weight", message)
        self.assertIn(STEM38, message)

    def test_a_missing_norm_is_refused_rather_than_replaced_with_ones(self):
        # measured: the installed 3.8 build holds no q_norm/k_norm at all, so a
        # rehearsal that substituted an identity would time a different block
        directory = temp_dir()
        build_install(directory, roles=tuple(ROLE_SHAPE))
        with self.assertRaises(rehearsal.RehearsalError) as caught:
            rehearsal.load_layer_weights(SSDAIWeights(directory), 3)
        self.assertIn("q_norm.weight", str(caught.exception))


class GeometryTests(unittest.TestCase):
    def setUp(self):
        # these tests call check_shapes() directly, so the probe geometry it
        # compares against has to be the one the synthetic install was built at
        self.patches = geometry_patches()
        for patcher in self.patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_matching_shapes_pass(self):
        directory = temp_dir()
        build_install(directory)
        self.assertIsNone(
            rehearsal.check_shapes(rehearsal.found_shapes(SSDAIWeights(directory), 3))
        )

    def test_a_wider_hidden_size_names_the_tensor_and_both_widths(self):
        directory = temp_dir()
        build_install(directory)
        shapes = rehearsal.found_shapes(SSDAIWeights(directory), 3)
        shapes["wq"] = (G["Q_PROJ_ROWS"], 2560)  # the installed 3.8 build's real shape
        with self.assertRaises(rehearsal.RehearsalError) as caught:
            rehearsal.check_shapes(shapes)
        message = str(caught.exception)
        self.assertIn("wq", message)
        self.assertIn("2560", message)
        self.assertIn(str(G["D"]), message)

    def test_a_norm_of_the_wrong_length_is_named_as_that_norm(self):
        directory = temp_dir()
        build_install(directory, layers=[3], roles=tuple(ROLE_SHAPE) + ("k_norm.weight",))
        install = Install(directory)
        install.quant("x_unused", (64, 64))
        shapes = rehearsal.found_shapes(SSDAIWeights(directory), 3)
        shapes["q_norm"] = (16,)
        with self.assertRaises(rehearsal.RehearsalError) as caught:
            rehearsal.check_shapes(shapes)
        self.assertIn("q_norm", str(caught.exception))

    def test_the_wrong_family_is_refused_before_any_block_is_built(self):
        directory = temp_dir()
        build_install(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        manifest["arch"]["hiddenSize"] = 2560
        (directory / "manifest.json").write_text(json.dumps(manifest))
        # a role's real shape, not the manifest, is what the check reads
        install = Install(directory)
        install.quant(f"{STEM38}.3.self_attn.q_proj.weight", (G["Q_PROJ_ROWS"], 2560), seed=1)
        install.write((4, [0, 0, 0, 1]))
        status, printed, model, _payload = run_main(directory)
        self.assertEqual(status, 2)
        self.assertIn("REFUSED", printed)
        self.assertEqual(model.calls, 0)


class WidthTests(unittest.TestCase):
    def test_a_role_stored_at_8_bits_is_read_at_8_bits(self):
        # the pre-fix load_tensor nibble-parsed anything that was not bf16 and
        # raised `cannot reshape array of size 1310720 into shape (512, 1280)`
        # on the installed build's layer-0 router
        directory = temp_dir()
        build_install(directory, bits=8)
        loaded = rehearsal.load_layer_weights(SSDAIWeights(directory), 3)
        self.assertEqual(loaded["wq"].shape, (G["Q_PROJ_ROWS"], G["D"]))

    def test_a_group_64_tensor_dequantizes_to_the_codes_written(self):
        directory = temp_dir()
        install = build_install(directory)
        weights = SSDAIWeights(directory)
        loaded = rehearsal.load_layer_weights(weights, 3)
        self.assertEqual(loaded["wq"].dtype, np.float16)
        expected = install.values[f"{STEM38}.3.self_attn.q_proj.weight"]
        self.assertTrue(np.array_equal(loaded["wq"].astype(np.float32), expected))

    def test_the_norm_is_read_as_the_bf16_values_stored(self):
        directory = temp_dir()
        install = build_install(directory)
        loaded = rehearsal.load_layer_weights(SSDAIWeights(directory), 3)
        expected = install.values[f"{STEM38}.3.self_attn.q_norm.weight"]
        self.assertEqual(int(loaded["q_norm"].size), G["HEAD_DIM"])
        self.assertTrue(np.allclose(loaded["q_norm"].astype(np.float32), expected.ravel()))

    def test_the_format_is_read_through_the_repository_reader(self):
        source = (BENCH / "tinytitan_ane_realweight_rehearsal.py").read_text()
        self.assertIn("ssdai_reader", source)
        self.assertNotIn("ENTRY_BYTES", source)
        self.assertNotIn("def read_index", source)


class MeasurementTests(CleanRun):
    def test_a_row_carries_its_layer_chunk_history_time_gap_and_state(self):
        first = self.payload["rows"][0]
        for key in ("layer", "chunk", "history", "ane_ms", "rel_err", "nan_inf", "status"):
            self.assertIn(key, first)
        self.assertEqual(first["status"], "measured")

    def test_the_median_of_the_timed_samples_is_what_the_row_reports(self):
        # one warm-up then three timed, scripted so the median is unambiguous
        self.assertEqual(self.payload["rows"][0]["ane_ms"], 200.0)
        self.assertIn("200.0 ms", self.printed)

    def test_the_warm_up_is_not_timed_but_it_is_counted(self):
        self.assertEqual(self.model.calls, 4 * (1 + 3))  # 4 chunks, 1 warm + 3 timed

    def test_a_median_of_zero_milliseconds_is_still_reported_as_measured(self):
        directory = temp_dir()
        build_install(directory)
        status, printed, _model, payload = run_main(directory, deltas=[0.0] * 4)
        self.assertEqual(payload["rows"][0]["status"], "measured")
        self.assertIn("0.0 ms", printed)
        self.assertNotIn("NOT MEASURED", printed)

    def test_a_block_with_no_matching_output_names_the_shapes_it_found(self):
        model = FakeModel(shapes=[("kv_cache", (1, G["N_KV_HEADS"], 4, G["HEAD_DIM"]))])
        directory = temp_dir()
        build_install(directory)
        status, printed, _model, payload = run_main(directory, model=model)
        self.assertEqual(status, 1)
        self.assertIn("kv_cache", printed)

    def test_a_predict_that_raises_is_an_error_row_and_the_sweep_continues(self):
        model = FakeModel()
        calls = {"n": 0}

        def explode(feed, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] <= 4:  # both chunks of the first layer, warm-ups included
                raise RuntimeError("the ANE refused this program")
            return model.answer(feed)

        model.predict = explode
        directory = temp_dir()
        build_install(directory)
        status, printed, _model, payload = run_main(directory, model=model)
        self.assertEqual(status, 1)
        self.assertIn("layer  3", printed)
        self.assertIn("layer  7", printed)
        self.assertIn("the ANE refused this program", printed)
        self.assertEqual(len(payload["rows"]), 4)

    def test_a_build_that_raises_is_an_error_row_naming_the_layer(self):
        directory = temp_dir()
        build_install(directory)
        status, printed, _model, _payload = run_main(directory, build_error="convert failed")
        self.assertEqual(status, 1)
        self.assertIn("convert failed", printed)
        self.assertIn("layer  3", printed)


class NumericsTests(unittest.TestCase):
    def test_the_ceiling_is_the_probe_s_own_number(self):
        self.assertEqual(rehearsal.MAX_MEAN_REL_ERROR, probe.MAX_MEAN_REL_ERROR)

    def test_a_gap_over_the_ceiling_fails_the_run_and_names_both_numbers(self):
        directory = temp_dir()
        build_install(directory)
        status, printed, _model, _payload = run_main(directory, reference=scaled_reference(0.5))
        self.assertEqual(status, 1)
        self.assertIn("numerics FAILED", printed)
        self.assertIn(str(rehearsal.MAX_MEAN_REL_ERROR), printed)

    def test_a_block_output_that_is_all_nan_costs_the_run(self):
        model = FakeModel(array=np.full((4, G["D"]), np.nan, dtype=np.float32))
        directory = temp_dir()
        build_install(directory)
        status, printed, _model, _payload = run_main(directory, model=model)
        self.assertEqual(status, 1)
        self.assertIn("nan/inf", printed)
        self.assertNotIn("REHEARSAL COMPLETE", printed)

    def test_a_non_finite_gap_never_passes_the_ceiling(self):
        self.assertFalse(rehearsal.passes_ceiling(float("nan")))
        self.assertFalse(rehearsal.passes_ceiling(float("inf")))
        self.assertTrue(rehearsal.passes_ceiling(0.0))
        self.assertTrue(rehearsal.passes_ceiling(rehearsal.MAX_MEAN_REL_ERROR))
        self.assertFalse(rehearsal.passes_ceiling(rehearsal.MAX_MEAN_REL_ERROR + 0.01))

    def test_a_reference_that_answers_nan_is_not_a_measured_row(self):
        # the output is clean here, so nothing but the gap can fail the row: a
        # row status that keys off the output alone would call this a pass
        directory = temp_dir()
        build_install(directory)
        status, printed, _model, payload = run_main(
            directory, reference=scaled_reference(float("nan"))
        )
        self.assertEqual(status, 1)
        for row in payload["rows"]:
            self.assertNotEqual(row["status"], "measured")
            self.assertEqual(row["nan_inf"], 0)
            self.assertIsNone(row["rel_err"])
        self.assertIn("numerics FAILED", printed)

    def test_an_infinite_output_is_counted_as_a_bad_value(self):
        # isnan() alone reads +inf as clean, which is the failure mode this
        # rehearsal exists to catch
        model = FakeModel(array=np.full((4, G["D"]), np.inf, dtype=np.float32))
        directory = temp_dir()
        build_install(directory)
        status, _printed, _model, payload = run_main(directory, model=model)
        self.assertEqual(status, 1)
        self.assertEqual(payload["rows"][0]["nan_inf"], 4 * G["D"])
        self.assertIsNone(payload["rows"][0]["rel_err"])

    def test_the_worst_gap_line_states_the_ceiling_it_was_checked_against(self):
        directory = temp_dir()
        build_install(directory)
        _status, printed, _model, _payload = run_main(directory)
        self.assertIn(f"ceiling {rehearsal.MAX_MEAN_REL_ERROR}", printed)


class StatusTests(unittest.TestCase):
    def test_everything_measured_and_within_the_ceiling_is_0(self):
        directory = temp_dir()
        build_install(directory)
        status, printed, _model, payload = run_main(directory)
        self.assertEqual(status, 0)
        self.assertIn("REHEARSAL COMPLETE", printed)
        self.assertEqual(payload["status"], 0)

    def test_the_incomplete_line_counts_failed_rows_out_of_the_total(self):
        model = FakeModel(array=np.full((4, G["D"]), np.nan, dtype=np.float32))
        directory = temp_dir()
        build_install(directory)
        status, printed, _model, _payload = run_main(directory, model=model)
        self.assertEqual(status, 1)
        self.assertIn("REHEARSAL INCOMPLETE", printed)
        self.assertIn("of 4", printed)

    def test_no_verdict_is_printed_over_a_refusal(self):
        status, printed, _model, _payload = run_main(temp_dir() / "gone")
        self.assertEqual(status, 2)
        self.assertNotIn("COMPLETE", printed)
        self.assertNotIn("INCOMPLETE", printed)

    def test_a_run_that_measured_nothing_does_not_divide_by_zero(self):
        directory = temp_dir()
        build_install(directory)
        status, printed, _model, _payload = run_main(directory, build_error="nope")
        self.assertEqual(status, 1)
        self.assertIn("NOT MEASURED", printed)
        self.assertNotIn("Traceback", printed)
        self.assertNotIn("inf", printed)

    def test_main_returns_an_int_on_every_path(self):
        source = (BENCH / "tinytitan_ane_realweight_rehearsal.py").read_text()
        body = source.split("def main() -> int:")[1].split('if __name__ == "__main__":')[0]
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("return "):
                value = stripped[len("return ") :].strip()
                self.assertTrue(
                    value.isdigit() or value == "status",
                    f"`{stripped}` hands the guard nothing usable",
                )

    def test_the_guard_propagates_the_status(self):
        source = (BENCH / "tinytitan_ane_realweight_rehearsal.py").read_text()
        self.assertIn("sys.exit(main())", source)
        self.assertNotIn("    main()\n", source)


class FooterTests(unittest.TestCase):
    def test_the_reference_is_the_probe_s_constant_not_a_restated_literal(self):
        self.assertEqual(rehearsal.GPU_REFERENCE_S, probe.GPU_LAYER_SECONDS)

    def test_the_header_names_the_layer_count_the_manifest_declared(self):
        directory = temp_dir()
        build_install(directory, layers=[3, 7])
        _status, printed, _model, _payload = run_main(directory)
        self.assertIn("2 full-attention layer(s)", printed)
        self.assertIn("[3, 7]", printed)

    def test_the_comparison_is_per_layer_chunk_against_the_recorded_reference(self):
        directory = temp_dir()
        build_install(directory)
        _status, printed, _model, _payload = run_main(directory, deltas=[5.0, 0.10, 0.20, 0.30])
        self.assertIn("ms per layer-chunk", printed)
        self.assertIn(f"{probe.GPU_MS_PER_LAYER_CHUNK:,.0f}", printed)
        # the recorded reference over the fixture's 200 ms median (4,215 / 200 is
        # 21.075, whose double rounds to 21.07), so the ratio is named by both of
        # its inputs rather than by a hand-written rounding of one
        self.assertIn(f"{probe.GPU_MS_PER_LAYER_CHUNK / 200.0:.2f}x", printed)

    def test_a_layer_count_the_reference_does_not_cover_gets_no_projection(self):
        # the recorded 84.3 s / 133.2 s cover 10 full-attention layers; this
        # synthetic install declares 2, so the total is not comparable
        directory = temp_dir()
        build_install(directory, layers=[3, 7])
        _status, printed, _model, _payload = run_main(directory)
        self.assertIn("PROJECTION NOT AVAILABLE", printed)
        self.assertIn(str(probe.FULL_ATTENTION_LAYERS), printed)

    def test_the_projection_is_printed_only_for_the_layer_count_it_covers(self):
        directory = temp_dir()
        build_install(directory, layers=list(range(3, 43, 4)), num_layers=40)
        deltas = [5.0, 0.10, 0.20, 0.30] * 20  # 20 chunks, median 0.2 s each
        _status, printed, _model, _payload = run_main(directory, deltas=deltas)
        self.assertIn("projected end-to-end prefill", printed)
        self.assertIn("no integration overhead", printed)
        self.assertIn(f"{48.9 + 20 * 0.2:.1f}", printed)  # (133.2 - 84.3) + 4.0 s

    def test_the_gpu_reference_and_the_ane_total_are_both_reported(self):
        directory = temp_dir()
        build_install(directory, layers=list(range(3, 43, 4)), num_layers=40)
        deltas = [5.0, 0.10, 0.20, 0.30] * 20
        _status, printed, _model, _payload = run_main(directory, deltas=deltas)
        self.assertIn("4.00 s", printed)
        self.assertIn(f"{rehearsal.GPU_REFERENCE_S:.1f} s", printed)


class ArtifactTests(unittest.TestCase):
    def test_the_default_path_is_anchored_at_the_repository_not_the_cwd(self):
        expected = ROOT / ".build" / "benchmark-results" / rehearsal.ARTIFACT
        self.assertEqual(rehearsal.DEFAULT_OUT, expected)

    def test_the_variable_names_a_file(self):
        target = temp_dir() / "given.json"
        with mock.patch.dict(rehearsal.os.environ, {rehearsal.OUT_ENV: str(target)}):
            self.assertEqual(rehearsal.out_path(), target)

    def test_the_variable_names_a_directory_to_fill(self):
        directory = temp_dir() / "results"
        directory.mkdir()
        with mock.patch.dict(rehearsal.os.environ, {rehearsal.OUT_ENV: str(directory)}):
            self.assertEqual(rehearsal.out_path(), directory / rehearsal.ARTIFACT)

    def test_a_blank_variable_falls_back_to_the_default(self):
        with mock.patch.dict(rehearsal.os.environ, {rehearsal.OUT_ENV: "  "}):
            self.assertEqual(rehearsal.out_path(), rehearsal.DEFAULT_OUT)

    def test_the_parent_directory_is_made_rather_than_crashing_the_run(self):
        target = temp_dir() / "deep" / "nested" / "out.json"
        directory = temp_dir()
        build_install(directory)
        status, printed, _model, _payload = run_main(
            directory, env={"TINYTITAN_BENCH_MODEL": str(directory), rehearsal.OUT_ENV: str(target)}
        )
        self.assertEqual(status, 0)
        self.assertTrue(target.exists())
        self.assertNotIn("FileNotFoundError", printed)

    def test_a_write_that_fails_costs_the_run_and_says_where(self):
        blocked = temp_dir() / "blocked"
        blocked.mkdir()
        (blocked / "not-a-directory").write_text("x")
        target = blocked / "not-a-directory" / "impossible.json"
        directory = temp_dir()
        build_install(directory)
        status, printed, _model, _payload = run_main(
            directory, env={"TINYTITAN_BENCH_MODEL": str(directory), rehearsal.OUT_ENV: str(target)}
        )
        self.assertEqual(status, 1)
        self.assertIn("NOT WRITTEN", printed)
        self.assertIn(str(target), printed)

    def test_the_artifact_holds_no_bare_nan(self):
        model = FakeModel(array=np.full((4, G["D"]), np.nan, dtype=np.float32))
        directory = temp_dir()
        build_install(directory)
        captured = io.StringIO()
        with redirect_stdout(captured):
            _status, _printed, _model, payload = run_main(directory, model=model)
        self.assertIsNotNone(payload)
        raw = json.dumps(payload)

        def refused(name):
            raise AssertionError(f"the artifact wrote a literal {name}")

        json.loads(raw, parse_constant=refused)
        rows = [row for row in payload["rows"] if row["nan_inf"]]
        self.assertTrue(rows)
        self.assertIsNone(rows[0]["rel_err"])
        self.assertIn("note", rows[0])

    def test_the_artifact_records_the_install_the_layers_and_the_ceiling(self):
        directory = temp_dir()
        build_install(directory)
        _status, _printed, _model, payload = run_main(directory)
        self.assertEqual(payload["layers"], [3, 7])
        self.assertEqual(payload["max_mean_rel_error"], rehearsal.MAX_MEAN_REL_ERROR)
        self.assertEqual(payload["gpu_reference_s"], rehearsal.GPU_REFERENCE_S)
        self.assertIn("model", payload)


class ImportSideEffectTests(unittest.TestCase):
    def test_importing_the_driver_opens_nothing_and_maps_no_file(self):
        child = f"""
import sys, builtins, pathlib
from unittest import mock
import numpy
for name in ["coremltools", "coremltools.converters", "coremltools.converters.mil",
             "coremltools.converters.mil.mil", "coremltools.converters.mil.mil.types"]:
    sys.modules.setdefault(name, mock.MagicMock(name=name))
opened = []
real_open = builtins.open
def watched_open(*args, **kwargs):
    target = str(args[0])
    if target.endswith((".bin", ".json")):
        opened.append(target)
    return real_open(*args, **kwargs)
builtins.open = watched_open
def watched_memmap(*args, **kwargs):
    raise AssertionError("a memmap ran at import: " + str(args[0]))
numpy.memmap = watched_memmap
sys.path.insert(0, {str(BENCH)!r})
import tinytitan_ane_realweight_rehearsal as rehearsal
builtins.open = real_open
if opened:
    raise SystemExit("import opened: " + repr(opened))
if getattr(rehearsal, "MODEL_BIN", None):
    raise SystemExit("MODEL_BIN survived the fix")
print("imported clean")
"""
        script = temp_dir() / "child.py"
        script.write_text(child)
        env = dict(rehearsal.os.environ)
        env.pop(rehearsal.MODEL_ENV, None)
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            env=env,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("imported clean", result.stdout)


class ProseTests(unittest.TestCase):
    """Does the prose match what runs -- the second seam question."""

    def setUp(self):
        self.source = (BENCH / "tinytitan_ane_realweight_rehearsal.py").read_text()
        self.doc = self.source.split('"""')[1]

    def test_the_docstring_names_the_variable_that_selects_the_install(self):
        self.assertIn(rehearsal.MODEL_ENV, self.doc)

    def test_the_docstring_does_not_claim_a_layer_count_the_code_cannot_know(self):
        for phrase in ("the 10 real full-attention layers", "10 full-attention layers"):
            self.assertNotIn(phrase, self.doc)

    def test_the_documented_command_is_the_pinned_core_ml_venv(self):
        self.assertIn(".venvs/coreml-py311/bin/python", self.doc)

    def test_the_chunk_list_is_the_recorded_prefill_shape(self):
        self.assertEqual(rehearsal.CHUNKS, [(4096, 0), (2007, 4096)])

    def test_the_recorded_prefill_total_is_named_with_its_source(self):
        self.assertEqual(rehearsal.GPU_PREFILL_TOTAL_S, 133.2)
        self.assertIn("133.2", self.doc)


if __name__ == "__main__":
    unittest.main()
