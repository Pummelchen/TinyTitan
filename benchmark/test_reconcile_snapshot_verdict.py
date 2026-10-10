"""AUD-270: `tools/reconcile_snapshot.py --check` certifies a fold it never looked at.

`tools/reconcile_snapshot.py` is the pre-repack check: `--check` compares an affine
snapshot against what the converter's current policy would produce and, per its own
docstring, "a clean report means the snapshot matches the policy tensor for tensor,
which is a stronger statement than 'the repack did not error'". Four of its five
verdicts are tensor-for-tensor. The fifth is not:

    for name in fold_names[:6]:                      # tools/reconcile_snapshot.py:148
        ...
        if stored.size != original.size:
            continue                                 # :154-155 -- counted as folded

The norm-fold test -- the one that decides whether the `+1` was folded into every
zero-centred RMSNorm, and the failure `prepare_qwen38.py:275-278` describes as
"silent ... the model generates fluent nonsense" -- samples six tensors out of
`fold_names` and calls the rest folded. `fold_names` holds one tensor per norm of each
kind the policy lists, in every layer, so six is a fixed fraction of nothing: measured
here on a snapshot of 24 fold tensors whose last 18 are unfolded, `--check` printed
`snapshot matches the converter policy` and returned 0, because those 18 were never
fetched, never compared and never mentioned. The repair path carries the same verdict,
so the same snapshot also answered `nothing to do`.
The `continue` is the second face: a fold tensor whose stored element count disagrees
with the checkpoint's cannot be compared at all, and it lands in neither `unfolded` nor
any not-checked bucket, so it reaches stdout as `folded`.

This suite builds a synthetic checkpoint with 24 fold tensors and the two widths the
policy distinguishes, serves it from 127.0.0.1 with real `curl` range requests (the
same seam `benchmark/test_prepare_qwen38.py` uses), and drives the real `main()` over
it, so every assertion is about a status the script computed rather than a function it
could be talked into calling. Nothing here touches huggingface.co, loads a model or
needs an install: `tools/patch_snapshot_precision.py`'s module-level `BASE` is pointed
at the local endpoint, which is the only network the file can reach.

The statuses this suite pins are the tree's three, as in `tools/model-guard.sh` and
`tools/qwen35_reference.py:411`:

    0  every tensor was compared and matched
    1  a tensor was compared and drifted -- and it names which
    2  something could not be compared, and the report says so

Run from `benchmark/`:

    python3 -m unittest test_reconcile_snapshot_verdict

Fifteen tests pass and nine of them carry their own evidence. Six were proven RED
against the file before the fix, each naming the status and the report line the old code
claimed. Two -- the stray-tensor pair -- exist because the first mutation sweep survived
a mutant that dropped `extra` from the drifted set: the tool was right about a tensor the
policy refuses to carry and nothing tested it, so they were proven RED against that
mutant and GREEN against the fixed file. One pins a policy invariant (below) and was
proven RED against a mutant of `prepare_qwen38.py`'s own name table. One of the fifteen
is the repair itself: a proven-bad norm used to pull every fold tensor through the
re-fetch, because a six-tensor sample cannot say which of the rest are wrong, so the
fixed tool rewrites the tensors it actually found wanting.

"""

import contextlib
import http.server
import importlib.util
import io
import json
import pathlib
import sys
import tempfile
import threading
import unittest
from unittest import mock

import ml_dtypes
import numpy as np
from safetensors.numpy import save_file

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "reconcile_snapshot.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("reconcile_snapshot", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pq = load_tool().pq

NORMS = [
    f"model.language_model.layers.{i}.self_attn.{kind}_norm.weight"
    for i in range(12)
    for kind in ("q", "k")
]
# One tensor per width the policy cares about: attention at the build width, router
# promoted to 8 inside a 4-bit build.
DENSE = {
    "model.language_model.layers.0.self_attn.q_proj.weight": (256, 256),
    "model.language_model.layers.0.mlp.gate.weight": (8, 256),
}
WIDTH = 4


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    """Serves the synthetic checkpoint under the byte ranges `curl -r` asks for."""

    root = None

    def do_GET(self):  # noqa: N802 - http.server's name
        path = pathlib.Path(self.root) / self.path.lstrip("/")
        try:
            blob = path.read_bytes()
        except OSError:
            self.send_error(404)
            return
        request = self.headers.get("Range")
        start, stop = 0, len(blob) - 1
        if request and request.startswith("bytes="):
            first, _, last = request[6:].partition("-")
            start = int(first)
            if last:
                stop = min(int(last), len(blob) - 1)
        body = blob[start : stop + 1]
        self.send_response(206 if request else 200)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Range", f"bytes {start}-{stop}/{len(blob)}")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def bf16(rows, seed):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(rows) - 0.5).astype(ml_dtypes.bfloat16)


# A buffer the current policy deliberately does not carry (`is_ple_buffer`,
# tools/prepare_qwen38.py:224-238): the repacker has no dtype for it, so a snapshot
# still holding one matches nothing the policy would produce.
PLE_BUFFER = "model.language_model.layers.0.ple.ple_embedding.layer_multipliers"


def snapshot_tensors(source, *, unfolded=(), short=(), dropped=(), at_bits=None, f32=(), stray=()):
    """Build every output tensor the policy asks for, exactly as the converter
    would, then apply the named defects on top."""
    out = {}
    for name, value in source.items():
        shape = list(value.shape)
        for out_name, _shape in pq.outputs_for(name, shape):
            bits = pq.quant_bits(out_name, WIDTH)
            if out_name in dropped:
                continue
            if bits is None:
                stored = np.ascontiguousarray(
                    value if out_name in unfolded else pq.fold_unit_offset(out_name, value)
                )
                if out_name in short:
                    stored = stored[: stored.shape[-1] // 2]
                if out_name in f32:
                    stored = stored.astype(np.float32)
                out[out_name] = stored
                continue
            width = at_bits or bits
            packed, scales, biases = pq.quantize_affine(np.ascontiguousarray(value), width)
            stem = out_name[: -len(".weight")]
            out[out_name], out[stem + ".scales"], out[stem + ".biases"] = packed, scales, biases
    for name in stray:
        out[name] = np.arange(4, dtype=np.int64)
    return out


class ReconcileHarness(unittest.TestCase):
    """One synthetic checkpoint, one synthetic snapshot, the real `main()`."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = pathlib.Path(tempfile.mkdtemp(prefix="aud270-"))
        cls.source = {name: bf16(shape, 30 + i) for i, (name, shape) in enumerate(DENSE.items())}
        cls.source.update({name: bf16((128,), 200 + i) for i, name in enumerate(NORMS)})
        # snapshot names: `rename()` strips `.weight` off the norms it lists
        cls.fold_names = [pq.rename(n) for n in NORMS]
        save_file(cls.source, str(cls.tmp / "ck.safetensors"))
        (cls.tmp / "ck_index.json").write_text(
            json.dumps({"weight_map": {n: "ck.safetensors" for n in cls.source}}, indent=1)
        )
        handler = type("H", (_RangeHandler,), {"root": str(cls.tmp)})
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=5)
        cls.server.server_close()

    def build(self, snapshot, **defects):
        tensors = snapshot_tensors(self.source, **defects)
        shard = snapshot / "model-00001-of-00001.safetensors"
        save_file(tensors, str(shard))
        total = sum(v.nbytes for v in tensors.values())
        (snapshot / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": total},
                    "weight_map": {k: shard.name for k in tensors},
                },
                indent=1,
            )
        )
        return tensors

    def run_main(self, snapshot, extra=()):
        """Call the real `main()` with the tool's only network pointed at the local
        endpoint, and return the status it handed back plus everything it printed."""
        module = load_tool()
        module.patcher.BASE = self.base
        module.patcher._headers.clear()
        out = io.StringIO()
        argv = [
            "reconcile_snapshot.py",
            "--snapshot",
            str(snapshot),
            "--index",
            str(self.tmp / "ck_index.json"),
            "--bits",
            str(WIDTH),
            *extra,
        ]
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(out):
            status = module.main()
        return status, out.getvalue()


class FoldVerdictTests(ReconcileHarness):
    def test_a_fully_folded_snapshot_reports_a_match(self):
        # The pass direction: six tensors or two hundred and four, a snapshot that is
        # right has to answer 0, or the refusal tests would pass against a tool that
        # simply always says no.
        with tempfile.TemporaryDirectory() as raw:
            snap = pathlib.Path(raw)
            self.build(snap)
            status, text = self.run_main(snap, ("--check",))
            self.assertEqual(status, 0, text)
            self.assertIn("matches the converter policy", text)

    def test_every_fold_tensor_is_compared_not_six(self):
        # The defect: norms 0..5 folded, 6..23 stored as the checkpoint's raw offset
        # from one. The tool looks at six and says the whole snapshot agrees.
        with tempfile.TemporaryDirectory() as raw:
            snap = pathlib.Path(raw)
            self.build(snap, unfolded=self.fold_names[6:])
            status, text = self.run_main(snap, ("--check",))
            self.assertEqual(status, 1, text)
            self.assertIn("UNFOLDED", text)

    def test_the_report_says_how_many_fold_tensors_were_compared(self):
        with tempfile.TemporaryDirectory() as raw:
            snap = pathlib.Path(raw)
            self.build(snap)
            _status, text = self.run_main(snap, ("--check",))
            self.assertIn(f"{len(self.fold_names)} of {len(self.fold_names)}", text)

    def test_a_fold_tensor_that_cannot_be_compared_is_not_reported_as_folded(self):
        # A stored norm half the checkpoint's length: nothing about the width or dtype
        # checks can see it, and the fold check `continue`s past it onto the report
        # line that says `folded`.
        with tempfile.TemporaryDirectory() as raw:
            snap = pathlib.Path(raw)
            self.build(snap, short={self.fold_names[7]})
            status, text = self.run_main(snap, ("--check",))
            self.assertEqual(status, 2, text)
            self.assertIn("could not be compared", text)

    def test_the_repair_path_does_not_call_the_same_snapshot_clean(self):
        # Without --check the tool prints `nothing to do` and returns 0 over the
        # identical snapshot, because the verdict it repairs from is the sampled one.
        with tempfile.TemporaryDirectory() as raw:
            snap = pathlib.Path(raw)
            self.build(snap, unfolded=self.fold_names[6:])
            status, text = self.run_main(snap)
            self.assertNotIn("nothing to do", text)
            self.assertEqual(status, 0, text)

    def test_the_repair_path_repairs_a_tensor_it_could_not_compare(self):
        # The status-2 case has a repair too: a stored norm of the wrong length is
        # rewritten from the checkpoint rather than reported and left in place.
        with tempfile.TemporaryDirectory() as raw:
            snap = pathlib.Path(raw)
            self.build(snap, short={self.fold_names[7]})
            status, text = self.run_main(snap)
            self.assertEqual(status, 0, text)
            self.assertNotIn("nothing to do", text)
            self.assertIn("(1 changes)", text)
            again, check_text = self.run_main(snap, ("--check",))
            self.assertEqual(again, 0, check_text)

    def test_every_folded_tensor_is_written_passthrough(self):
        # The guard for the sibling this sweep looked for and cleared: both repair
        # paths quantise without folding (`reconcile_snapshot.py`'s quantised branch,
        # and all of `patch_snapshot_precision.py`, whose target test requires
        # `.weight`). They cannot reach a fold tensor only because every name in
        # UNIT_OFFSET_NORM_SUFFIXES is also in STRIP_WEIGHT_SUFFIXES, so `rename()`
        # drops the `.weight` that `quant_bits` requires to answer with a width.
        for suffix in pq.UNIT_OFFSET_NORM_SUFFIXES:
            ck_name = f"model.language_model.layers.0{suffix}.weight"
            out_name = pq.rename(ck_name)
            self.assertFalse(out_name.endswith(".weight"), out_name)
            self.assertIsNone(pq.quant_bits(out_name, 4), out_name)
            self.assertIsNone(pq.quant_bits(out_name, 8), out_name)
            folded = pq.fold_unit_offset(out_name, np.ones(2, dtype=ml_dtypes.bfloat16))
            self.assertEqual(float(folded[0]), 2.0, out_name)


class CheckVerdictTests(ReconcileHarness):
    def test_a_missing_tensor_is_refused(self):
        with tempfile.TemporaryDirectory() as raw:
            snap = pathlib.Path(raw)
            self.build(
                snap, dropped={pq.rename("model.language_model.layers.0.self_attn.q_proj.weight")}
            )
            status, text = self.run_main(snap, ("--check",))
            self.assertEqual(status, 1, text)
            self.assertIn("missing  : 1", text)

    def test_a_tensor_at_the_wrong_width_is_refused(self):
        with tempfile.TemporaryDirectory() as raw:
            snap = pathlib.Path(raw)
            self.build(snap, at_bits=8)
            status, text = self.run_main(snap, ("--check",))
            self.assertEqual(status, 1, text)
            self.assertIn("wrong bits: 1", text)
            # the per-item lines print the name after its `language_model.` split
            self.assertIn("layers.0.self_attn.q_proj.weight: have 8, want 4", text)

    def test_a_tensor_at_the_wrong_dtype_is_refused(self):
        # A passthrough norm written as F32 loads far enough to look fine and is then
        # refused as a corrupt index, so the dtype is part of the contract.
        with tempfile.TemporaryDirectory() as raw:
            snap = pathlib.Path(raw)
            name = self.fold_names[0]
            self.build(snap, f32={name})
            status, text = self.run_main(snap, ("--check",))
            self.assertEqual(status, 1, text)
            self.assertIn("wrong dtype: 1", text)
            self.assertIn(name.split("language_model.")[-1] + ": have F32, want BF16", text)

    def test_a_stray_tensor_is_refused(self):
        # The fifth verdict, and the one a mutation sweep found unpinned: a snapshot
        # holding a tensor the current policy refuses to carry is drift, not a match.
        with tempfile.TemporaryDirectory() as raw:
            snap = pathlib.Path(raw)
            self.build(snap, stray=(PLE_BUFFER,))
            status, text = self.run_main(snap, ("--check",))
            self.assertEqual(status, 1, text)
            self.assertIn("extra    : 1", text)
            # the group line is the bare name, with the layer index normalised
            self.assertIn("model.language_model.layers.N.ple.ple_embedding.layer_multipliers", text)


class RepairTests(ReconcileHarness):
    def test_a_clean_snapshot_is_left_alone(self):
        with tempfile.TemporaryDirectory() as raw:
            snap = pathlib.Path(raw)
            self.build(snap)
            status, text = self.run_main(snap)
            self.assertIn("nothing to do", text)
            self.assertEqual(status, 0, text)

    def test_repair_rewrites_only_the_tensors_that_are_wrong(self):
        # Before the fix one proven-bad norm pulled all 24 through the re-fetch,
        # because a six-tensor sample cannot say which of the rest are wrong.
        with tempfile.TemporaryDirectory() as raw:
            snap = pathlib.Path(raw)
            self.build(snap, unfolded={self.fold_names[3]})
            status, text = self.run_main(snap)
            self.assertEqual(status, 0, text)
            self.assertIn("(1 changes)", text)

    def test_the_repair_drops_a_stray_tensor_and_the_check_then_passes(self):
        # The repair face of the same verdict: the stray is removed from the shard and
        # the index, not merely reported, and the index's own total follows.
        with tempfile.TemporaryDirectory() as raw:
            snap = pathlib.Path(raw)
            self.build(snap, stray=(PLE_BUFFER,))
            status, text = self.run_main(snap)
            self.assertEqual(status, 0, text)
            self.assertNotIn("nothing to do", text)
            self.assertIn("drop layers.0.ple.ple_embedding.layer_multipliers", text)
            index = json.loads((snap / "model.safetensors.index.json").read_text())
            self.assertNotIn(PLE_BUFFER, index["weight_map"])
            again, check_text = self.run_main(snap, ("--check",))
            self.assertEqual(again, 0, check_text)

    def test_the_repaired_snapshot_passes_the_check(self):
        with tempfile.TemporaryDirectory() as raw:
            snap = pathlib.Path(raw)
            self.build(snap, unfolded={self.fold_names[3], self.fold_names[9]})
            first, _ = self.run_main(snap)
            self.assertEqual(first, 0)
            status, text = self.run_main(snap, ("--check",))
            self.assertEqual(status, 0, text)


if __name__ == "__main__":
    unittest.main()
