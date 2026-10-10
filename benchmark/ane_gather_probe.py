#!/usr/bin/env python3
"""Size the gather-graph variant: can the ANE attend to only the selected keys?

Today the sidecar computes attention **densely** over the whole context and
folds the QSA selection in as an additive mask: `[chunk, history + chunk]`
scores per layer-chunk. The GPU path instead **gathers** the ~2,051 keys the
indexer kept, per query. So the obvious question is whether the ANE graph could
do the same — gather the selection and score only those keys — which would cut
the arithmetic by the ratio `total / budget` and, if the arena is what the
per-variant load cost tracks, make the sidecar load faster too.

This builds both graphs at one geometry and measures the things that decide it:

  dense   q[T,qDim] k[total,kvDim] v[total,kvDim] mask[1,1,T,total]   -> out[T,qDim]
  gather  q[T,qDim] k[total,kvDim] v[total,kvDim] idx[T,budget] i32  -> out[T,qDim]

The catch the probe is built to expose: a gather has to materialise the selected
keys per query, `[heads, T, budget, headDim]`, where the dense graph materialises
`[heads, T, total]`. Per head that is `budget * headDim` values against `total`
— for the 3.8 geometry, 2,051x256 = 525k against 8,192, about **64x larger** —
because a gathered key is replicated across every query that selects it, which a
GPU hides in registers and a MIL graph cannot. Whether the ANE compiler fuses the
gather into the matmul, refuses it, or runs it on the CPU is exactly what cannot
be reasoned out; it has to be measured.

`--chunk` defaults to a small value because the gathered tensor is the
memory-hungry one: at the real chunk of 4,096 it is ~103 GB, which is the point.
The geometry that matters (heads, headDim, budget, total) is the real 3.8 one.

  ~/.venvs/coreml-py311/bin/python benchmark/ane_gather_probe.py \
      --chunk 32 --record --label v5.6-gather
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import statistics
import sys
import tempfile
import time

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb

from tinytitan_profile import arm_metric, metric_count

try:
    from coremltools.models.compute_plan import MLComputePlan
except Exception:  # noqa: BLE001 — optional
    MLComputePlan = None

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark/ane-prefill"
FP16 = ct.converters.mil.mil.types.fp16
INT32 = ct.converters.mil.mil.types.int32


@dataclasses.dataclass(frozen=True)
class Geometry:
    """The shipped Qwen 3.8 full-attention geometry, at a probe-sized chunk."""

    chunk: int = 32
    total: int = 8_192
    heads: int = 24
    head_dim: int = 256
    budget: int = 2_051

    @property
    def q_dim(self) -> int:
        return self.heads * self.head_dim

    @property
    def dense_values(self) -> int:
        """fp16 values in the dense score matrix, all heads."""
        return self.heads * self.chunk * self.total

    @property
    def gather_values(self) -> int:
        """fp16 values in the gathered keys, all heads."""
        return self.heads * self.chunk * self.budget * self.head_dim

    def at_chunk(self, chunk: int) -> "Geometry":
        return dataclasses.replace(self, chunk=chunk)


def build_dense(geom: Geometry):
    """Today's graph: score every visible key, drop the selection with a mask."""
    t = geom.chunk
    specs = [
        mb.TensorSpec(shape=(t, geom.q_dim), dtype=FP16),
        mb.TensorSpec(shape=(geom.total, geom.q_dim), dtype=FP16),
        mb.TensorSpec(shape=(geom.total, geom.q_dim), dtype=FP16),
        mb.TensorSpec(shape=(1, 1, t, geom.total), dtype=FP16),
    ]

    @mb.program(input_specs=specs, opset_version=ct.target.iOS18)
    def prog(q, k, v, mask):
        q4 = mb.reshape(x=q, shape=[1, geom.heads, t, geom.head_dim])
        k4 = mb.transpose(
            x=mb.reshape(x=k, shape=[geom.total, geom.heads, geom.head_dim]), perm=[1, 0, 2]
        )
        v4 = mb.transpose(
            x=mb.reshape(x=v, shape=[geom.total, geom.heads, geom.head_dim]), perm=[1, 0, 2]
        )
        k4 = mb.expand_dims(x=k4, axes=[0])
        v4 = mb.expand_dims(x=v4, axes=[0])
        scores = mb.matmul(x=q4, y=k4, transpose_y=True)
        scores = mb.add(x=scores, y=mask)
        probs = mb.softmax(x=scores, axis=-1)
        attn = mb.matmul(x=probs, y=v4)
        attn = mb.transpose(x=attn, perm=[0, 2, 1, 3])
        return mb.reshape(x=attn, shape=[t, geom.q_dim])

    return prog


def build_gather(geom: Geometry):
    """The proposed graph: gather the selection, score only those keys."""
    t, budget = geom.chunk, geom.budget
    specs = [
        mb.TensorSpec(shape=(t, geom.q_dim), dtype=FP16),
        mb.TensorSpec(shape=(geom.total, geom.q_dim), dtype=FP16),
        mb.TensorSpec(shape=(geom.total, geom.q_dim), dtype=FP16),
        mb.TensorSpec(shape=(t, budget), dtype=INT32),
    ]

    @mb.program(input_specs=specs, opset_version=ct.target.iOS18)
    def prog(q, k, v, idx):
        # One flattened index list per query row, gathered from the key history.
        flat = mb.reshape(x=idx, shape=[t * budget])
        keys = mb.reshape(
            x=mb.gather(x=k, indices=flat, axis=0), shape=[t, budget, geom.heads, geom.head_dim]
        )
        keys = mb.transpose(x=keys, perm=[2, 0, 1, 3])
        keys = mb.expand_dims(x=keys, axes=[0])
        values = mb.reshape(
            x=mb.gather(x=v, indices=flat, axis=0), shape=[t, budget, geom.heads, geom.head_dim]
        )
        values = mb.transpose(x=values, perm=[2, 0, 1, 3])
        values = mb.expand_dims(x=values, axes=[0])
        # Query gets a singleton key axis so the matmul batches per query:
        # [1, heads, t, 1, hd] x [1, heads, t, budget, hd] -> [1, heads, t, budget].
        q4 = mb.transpose(x=mb.reshape(x=q, shape=[t, geom.heads, geom.head_dim]), perm=[1, 0, 2])
        q5 = mb.expand_dims(x=q4, axes=[0, 3])
        # No mask: the gathered set *is* the selection.
        scores = mb.matmul(x=q5, y=keys, transpose_y=True)
        probs = mb.softmax(x=scores, axis=-1)
        attn = mb.matmul(x=probs, y=values)
        attn = mb.squeeze(x=attn, axes=[3])
        attn = mb.transpose(x=attn, perm=[0, 2, 1, 3])
        return mb.reshape(x=attn, shape=[t, geom.q_dim])

    return prog


def inputs_for(name: str, geom: Geometry, rng):
    t = geom.chunk
    common = {
        "q": rng.standard_normal((t, geom.q_dim)).astype(np.float16),
        "k": rng.standard_normal((geom.total, geom.q_dim)).astype(np.float16),
        "v": rng.standard_normal((geom.total, geom.q_dim)).astype(np.float16),
    }
    if name == "dense":
        mask = np.full((1, 1, t, geom.total), -30000.0, dtype=np.float16)
        for row in range(t):
            mask[0, 0, row, : geom.total - t + row + 1] = 0.0
        common["mask"] = mask
    else:
        # A QSA-shaped selection: ascending keys within the visible window.
        idx = np.empty((t, geom.budget), dtype=np.int32)
        for row in range(t):
            visible = min(geom.total, geom.budget + row)
            keys = np.linspace(0, visible - 1, geom.budget).astype(np.int32)
            idx[row] = keys
        common["idx"] = idx
    return common


def ane_operations(package: pathlib.Path) -> int | None:
    if MLComputePlan is None:
        return None
    compiled = ct.models.utils.compile_model(str(package))
    plan = MLComputePlan.load_from_path(compiled, compute_units=ct.ComputeUnit.CPU_AND_NE)
    program = plan.model_structure.program
    function = program.functions.get("main") or next(iter(program.functions.values()))
    on_ane = 0
    for operation in function.block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
        device = getattr(usage, "preferred_compute_device", None)
        if device is not None and "NeuralEngine" in type(device).__name__:
            on_ane += 1
    return on_ane


def measure(name: str, geom: Geometry, repeats: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    row: dict = {
        "graph": name,
        "chunk": geom.chunk,
        "dense_values": geom.dense_values,
        "gather_values": geom.gather_values,
        "gather_over_dense": geom.gather_values / geom.dense_values,
    }
    directory = pathlib.Path(tempfile.mkdtemp(prefix=f"ane-gather-{name}-"))
    package = directory / f"{name}.mlpackage"
    builder = build_dense(geom) if name == "dense" else build_gather(geom)
    start = time.perf_counter()
    model = ct.convert(
        builder,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    row["convert_seconds"] = time.perf_counter() - start
    model.save(str(package))
    row["package_megabytes"] = (
        sum(f.stat().st_size for f in package.rglob("*") if f.is_file()) / 1e6
    )
    features = inputs_for(name, geom, rng)
    row["operations"] = None
    row["operations_on_ane"] = None
    try:
        row["operations_on_ane"] = ane_operations(package)
    except Exception as exc:  # noqa: BLE001 — reported
        row["plan_error"] = f"{type(exc).__name__}: {exc}"[:200]
    for units, label in ((ct.ComputeUnit.CPU_AND_NE, "ane"), (ct.ComputeUnit.CPU_ONLY, "cpu")):
        try:
            start = time.perf_counter()
            loaded = ct.models.MLModel(str(package), compute_units=units)
            row[f"{label}_load_seconds"] = time.perf_counter() - start
            loaded.predict(features)  # warm
            times = []
            for _ in range(repeats):
                begin = time.perf_counter()
                loaded.predict(features)
                times.append(time.perf_counter() - begin)
            row[f"{label}_predict_seconds"] = statistics.median(times)
            row[f"{label}_predict_runs"] = [round(x, 4) for x in times]
        except Exception as exc:  # noqa: BLE001 — reported
            row[f"{label}_error"] = f"{type(exc).__name__}: {exc}"[:200]
    return row


ARM_KEYS = ("ane_predict_seconds", "ane_load_seconds", "cpu_predict_seconds")
ARM_NAMES = ("dense", "gather")


def verdict(results: list[dict], repeats: int = 1):
    """The two ratios the probe exists to quote, and the status to exit with.

    0 both arms measured every figure the page prints, 1 a ratio printed beside
    an error one arm recorded, 2 a figure the ratios are made of is missing.
    Before this the whole block sat behind four truthy lookups, so a run whose
    arms both errored printed no ratio line at all and exited 0 -- the same page
    shape as a probe that had answered the question.
    """
    reasons: list[str] = []
    lines: list[str] = []
    if repeats < 1:
        reasons.append(
            f"NOT MEASURED: --repeats {repeats} asks for no timed call, so no arm was measured"
        )
    if len(results) != len(ARM_NAMES):
        reasons.append(f"NOT MEASURED: the probe has {len(results)} arm rows, not one per arm")
    by_arm = {name: row for name, row in zip(ARM_NAMES, results, strict=False)}
    for name, row in by_arm.items():
        for key in ARM_KEYS:
            _, counted, total = arm_metric([row], key)
            note = metric_count(name, key, counted, total)
            if note:
                reasons.append(note)
        for key in ("plan_error", "ane_error", "cpu_error"):
            if row.get(key):
                reasons.append(f"REFUSED: the {name} arm recorded {key}: {row[key]}")

    dense, gather = by_arm.get("dense", {}), by_arm.get("gather", {})
    ratio_keys = ARM_KEYS[:2]
    if not all(dense.get(key) is not None and gather.get(key) is not None for key in ratio_keys):
        reasons.append(
            "NOT MEASURED: no gather/dense ratio -- an arm logged neither "
            "ane_predict_seconds nor ane_load_seconds"
        )
    elif not all(dense.get(key) for key in ratio_keys):
        reasons.append(
            "REFUSED: the dense arm logged a 0 s ANE figure, so the gather/dense "
            "ratio would divide by zero"
        )
    else:
        lines.append("verdict inputs:")
        lines.append(
            f"   gather/dense prediction  "
            f"{gather['ane_predict_seconds'] / dense['ane_predict_seconds']:.2f}x"
        )
        lines.append(
            f"   gather/dense load        "
            f"{gather['ane_load_seconds'] / dense['ane_load_seconds']:.2f}x"
        )

    status = (
        2 if any(reason.startswith("NOT MEASURED") for reason in reasons) else 1 if reasons else 0
    )
    lines.extend(f"  {reason}" for reason in reasons)
    lines.append(f"  probe status {status}")
    return lines, status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunk",
        type=int,
        default=32,
        help="probe chunk (default 32). The gathered tensor at "
        "the real 4,096 is ~103 GB, which is the finding; "
        "the geometry that decides the ratio is the real one",
    )
    parser.add_argument("--total", type=int, default=8_192)
    parser.add_argument("--budget", type=int, default=2_051)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--label", default=None)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()

    geom = Geometry(chunk=args.chunk, total=args.total, budget=args.budget)
    print(
        f"geometry: chunk {geom.chunk}, total {geom.total}, "
        f"{geom.heads} heads x {geom.head_dim}, budget {geom.budget}"
    )
    print(f"  dense score matrix  {geom.dense_values * 2 / 1e6:8.1f} MB fp16")
    print(
        f"  gathered keys       {geom.gather_values * 2 / 1e6:8.1f} MB fp16 "
        f"({geom.gather_values / geom.dense_values:.0f}x the dense matrix)"
    )
    print(
        f"  ... at the real chunk 4,096: "
        f"{geom.at_chunk(4096).gather_values * 2 / 1e9:.1f} GB gathered keys "
        f"against {geom.at_chunk(4096).dense_values * 2 / 1e6:.0f} MB dense"
    )
    print()
    if args.repeats < 1:
        lines, status = verdict([], args.repeats)
        print("\n".join(lines))
        return status
    results = []
    for name in ("dense", "gather"):
        row = measure(name, geom, args.repeats, args.seed)
        results.append(row)
        print(f"== {name}", flush=True)
        for key in (
            "convert_seconds",
            "package_megabytes",
            "operations",
            "operations_on_ane",
            "ane_load_seconds",
            "ane_predict_seconds",
            "cpu_load_seconds",
            "cpu_predict_seconds",
            "plan_error",
            "ane_error",
            "cpu_error",
        ):
            if row.get(key) is not None:
                print(f"   {key:<22} {row[key]}")
        print()
    lines, status = verdict(results, args.repeats)
    print("\n".join(lines))
    if args.record:
        RESULTS.mkdir(parents=True, exist_ok=True)
        label = args.label or "unlabelled"
        path = RESULTS / f"ane-gather-probe-{label}.json"
        record = {
            "geometry": dataclasses.asdict(geom),
            "repeats": args.repeats,
            "status": status,
            "results": results,
        }
        path.write_text(json.dumps(record, indent=2) + "\n")
        print(f"\nwrote {path.relative_to(ROOT)}")
    return status


if __name__ == "__main__":
    sys.exit(main())
