#!/usr/bin/env python3
"""Track A rehearsal: the installed model's real full-attention layers on the ANE.

Reads the actual quantized weights of every full-attention layer out of an
installed `.ssdai` — the install is named by `TINYTITAN_BENCH_MODEL`, default
`models/ornith-1.5_35B_A3B_4Bit` — takes the layer list from that install's own
`arch.fullAttentionLayerMask`, dequantizes through the repository's reader
(`tools/ssdai_reader.py`, which infers each tensor's width from its payload
size), and replays the exact layer-chunk sequence of a 6,103-token prefill:
chunk 4096 with no history, then chunk 2007 against 4096 tokens of history, for
each layer.

The GPU side is the measurement recorded in `tinytitan_ane_attention_probe.py`:
the probe's ten full-attention layers cost 84.3 s of a 133.2 s prefill. The
end-to-end projection is printed only for an install that declares the same
layer count that reference was measured over, and it adds no integration
overhead, so it is a bound rather than a result — the go/no-go is to clear the
per-layer-chunk cost by enough to survive the overheads of wiring the block in,
and this rehearsal measures none of them.

Per row: latency (median of `REPEATS` timed runs after `WARM_UPS` warm-up) and
the mean relative gap to the probe's float32 reference of the same math on the
same weights, checked against the probe's ceiling. A row that raises is recorded
and the sweep continues, because the rows already measured are the result.

Exit is 2 when the install, its manifest or its geometry refuse the run, and 1
when any layer-chunk errored, failed the ceiling, or the artifact could not be
written; 0 only when every row measured and passed.

  ~/.venvs/coreml-py311/bin/python benchmark/tinytitan_ane_realweight_rehearsal.py
  TINYTITAN_BENCH_MODEL=models/qwen3.8-flash-next_4-Bit \
    ~/.venvs/coreml-py311/bin/python benchmark/tinytitan_ane_realweight_rehearsal.py
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import tinytitan_ane_attention_probe as probe  # noqa: E402
from ssdai_reader import SSDAIWeights  # noqa: E402

MODEL_ENV = "TINYTITAN_BENCH_MODEL"
DEFAULT_MODEL = ROOT / "models/ornith-1.5_35B_A3B_4Bit"
OUT_ENV = "TINYTITAN_ANE_REHEARSAL_OUT"
ARTIFACT = "ane-realweight-rehearsal.json"
DEFAULT_OUT = ROOT / ".build/benchmark-results" / ARTIFACT

# The 6,103-token prefill: 4,096 tokens cold, then 2,007 against that history.
CHUNKS = [(4096, 0), (2007, 4096)]
GPU_REFERENCE_S = probe.GPU_LAYER_SECONDS
GPU_PREFILL_TOTAL_S = 133.2  # measured in the same run as GPU_REFERENCE_S
MAX_MEAN_REL_ERROR = probe.MAX_MEAN_REL_ERROR
WARM_UPS = 1
REPEATS = 3
SEED = 41

# The probe block's six roles, the names the install keeps them under, and the
# two stems `tools/export_ane_prefill.py` records for the two families.
ROLES = ("wq", "wk", "wv", "wo", "q_norm", "k_norm")
ROLE_SUFFIX = {
    "wq": "q_proj.weight",
    "wk": "k_proj.weight",
    "wv": "v_proj.weight",
    "wo": "o_proj.weight",
    "q_norm": "q_norm.weight",
    "k_norm": "k_norm.weight",
}


class RehearsalError(ValueError):
    """The run cannot start: name the variable, file, layer or shape at fault."""


def model_directory(env=None) -> pathlib.Path:
    """The install to rehearse, read when the run starts, not at import."""
    mapping = os.environ if env is None else env
    value = str(mapping.get(MODEL_ENV, "")).strip()
    return pathlib.Path(value) if value else DEFAULT_MODEL


def open_weights(directory: pathlib.Path) -> SSDAIWeights:
    if not directory.is_dir():
        raise RehearsalError(
            f"{MODEL_ENV}={directory} is not a directory (default {DEFAULT_MODEL}); "
            "point it at an installed .ssdai"
        )
    for name in ("model_weights.bin", "manifest.json"):
        if not (directory / name).is_file():
            raise RehearsalError(f"{directory} has no {name} — not a complete install")
    return SSDAIWeights(directory)


def full_attention_layers(weights: SSDAIWeights) -> list[int]:
    """The layers this install declares as full-attention, from its own mask."""
    arch = weights.manifest.get("arch") or {}
    mask = arch.get("fullAttentionLayerMask")
    if mask is None:
        raise RehearsalError(
            f"{weights.root} manifest arch has no fullAttentionLayerMask, so nothing here "
            "can say which layers attend in full"
        )
    num_layers = int(arch.get("numLayers") or weights.manifest.get("numLayers") or 0)
    if len(mask) != num_layers:
        raise RehearsalError(
            f"fullAttentionLayerMask holds {len(mask)} entries for a {num_layers}-layer model"
        )
    layers = [index for index, kind in enumerate(mask) if int(kind) == 1]
    if not layers:
        raise RehearsalError(
            f"fullAttentionLayerMask declares no full-attention layer among {num_layers} "
            "(1 is full, 0 sliding window, 2 linear) — there is nothing to rehearse"
        )
    return layers


def layer_stem(weights: SSDAIWeights, layer: int) -> str:
    """The name prefix this install keeps that layer's attention tensors under."""
    wanted = f".{layer}.self_attn.{ROLE_SUFFIX['wq']}"
    for name in weights.entries:
        if name.endswith(wanted):
            return name[: -len(f".self_attn.{ROLE_SUFFIX['wq']}")]
    raise RehearsalError(
        f"layer {layer} has no *{wanted} in {weights.root}; the two stems in use here are "
        f"'language_model.model.layers{wanted}' (qwen36 and dense) and "
        f"'model.language_model.layers{wanted}' (the 3.8 family)"
    )


def expected_shapes() -> dict[str, tuple[int, ...]]:
    """The shapes the probe block is built at, read from the probe at call time."""
    return {
        "wq": (probe.Q_PROJ_ROWS, probe.D),
        "wk": (probe.KV_DIM, probe.D),
        "wv": (probe.KV_DIM, probe.D),
        "wo": (probe.D, probe.Q_DIM),
        "q_norm": (probe.HEAD_DIM,),
        "k_norm": (probe.HEAD_DIM,),
    }


def found_shapes(weights: SSDAIWeights, layer: int) -> dict[str, tuple[int, ...] | None]:
    prefix = layer_stem(weights, layer)
    shapes = {}
    for role in ROLES:
        entry = weights.entries.get(f"{prefix}.self_attn.{ROLE_SUFFIX[role]}")
        shapes[role] = tuple(entry["shape"]) if entry else None
    return shapes


def check_shapes(shapes: dict[str, tuple[int, ...] | None]) -> None:
    wanted = expected_shapes()
    for role, expected in wanted.items():
        found = shapes.get(role)
        if found is None:
            raise RehearsalError(f"{role} ({ROLE_SUFFIX[role]}) is not in this install")
        if found != expected:
            raise RehearsalError(
                f"{role} is {found} but the probe block builds {expected} — this rehearsal "
                "times the probe's geometry, so an install shaped otherwise is not it"
            )


def load_layer_weights(weights: SSDAIWeights, layer: int) -> dict[str, np.ndarray]:
    """The six roles of one layer, dequantized to fp16 as the block wants them."""
    prefix = layer_stem(weights, layer)
    loaded = {}
    for role in ROLES:
        name = f"{prefix}.self_attn.{ROLE_SUFFIX[role]}"
        if name not in weights.entries:
            raise RehearsalError(f"{name} is missing from {weights.root} under {prefix}")
        loaded[role] = weights.get(name).astype(np.float16)
    return loaded


def passes_ceiling(rel) -> bool:
    """A non-finite gap is never within the ceiling, whatever the ceiling is."""
    return rel is not None and math.isfinite(rel) and rel <= MAX_MEAN_REL_ERROR


def new_row(layer: int, chunk: int, history: int) -> dict:
    return {
        "layer": layer,
        "chunk": chunk,
        "history": history,
        "ane_ms": None,
        "rel_err": None,
        "nan_inf": 0,
        "status": "error",
        "note": "not run",
    }


def measure_chunk(layer, chunk, history, layer_weights, rng) -> dict:
    """One layer-chunk: the row of what was actually measured, or why not."""
    row = new_row(layer, chunk, history)
    try:
        model = probe.build_block(chunk, history, layer_weights)
        hidden = (rng.standard_normal((chunk, probe.D)) * 0.5).astype(np.float16)
        k_hist = (rng.standard_normal((1, probe.N_KV_HEADS, history, probe.HEAD_DIM)) * 0.5).astype(
            np.float16
        )
        v_hist = (rng.standard_normal((1, probe.N_KV_HEADS, history, probe.HEAD_DIM)) * 0.5).astype(
            np.float16
        )
        cos_t, sin_t = probe.rope_tables(history, chunk)
        mask = probe.causal_mask(chunk, history)
        feed = {"hidden": hidden, "cos_t": cos_t, "sin_t": sin_t, "mask": mask}
        if history > 0:
            feed.update(k_hist=k_hist, v_hist=v_hist)

        name = probe.block_output_name(model, chunk)
        samples, out = probe.timed_predictions(model, feed, REPEATS, WARM_UPS)
        row["ane_ms"] = probe.median_ms(probe.timed_samples(samples, WARM_UPS))

        got = np.asarray(out[name], dtype=np.float32)
        ref = probe.reference(hidden, k_hist, v_hist, cos_t, sin_t, mask, layer_weights)
        row["nan_inf"] = int(np.count_nonzero(~np.isfinite(got)))
        rel = float(np.abs(got - ref).mean() / max(np.abs(ref).mean(), 1e-9))
        row["rel_err"] = rel if math.isfinite(rel) else None
        if not math.isfinite(rel):
            row["status"] = "non-finite"
            row["note"] = "the gap to the float32 reference is not a finite number"
        elif not passes_ceiling(rel):
            row["status"] = "numerics-failed"
            row["note"] = f"the gap {rel:.4f} is over the {MAX_MEAN_REL_ERROR} ceiling"
        else:
            row["status"] = "measured"
            row["note"] = ""
    except Exception as error:  # one layer-chunk failing does not end the sweep
        row["status"] = "error"
        row["note"] = f"{type(error).__name__}: {error}"
    return row


def row_report(row: dict) -> str:
    label = f"layer {row['layer']:2d} chunk {row['chunk']}:{row['history']}"
    if row["ane_ms"] is None:
        return f"  {label}  ERROR -- {row['note']}"
    gap = "NOT FINITE" if row["rel_err"] is None else f"{row['rel_err']:.4f}"
    return (
        f"  {label}  {row['ane_ms']:8.1f} ms  rel {gap}  nan/inf {row['nan_inf']}  {row['status']}"
    )


def out_path(env=None) -> pathlib.Path:
    """Where the artifact goes: the variable names a file, or a directory to fill."""
    mapping = os.environ if env is None else env
    value = str(mapping.get(OUT_ENV, "")).strip()
    if not value:
        return DEFAULT_OUT
    path = pathlib.Path(value)
    return path / ARTIFACT if path.is_dir() else path


def write_artifact(payload: dict, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    try:
        directory = model_directory()
        weights = open_weights(directory)
        layers = full_attention_layers(weights)
        for layer in layers:
            check_shapes(found_shapes(weights, layer))
    except RehearsalError as error:
        print(f"REFUSED: {error}")
        return 2

    print(f"REAL-WEIGHT ANE REHEARSAL -- {directory}")
    print(f"  {len(layers)} full-attention layer(s) from the mask: {layers}")
    print(
        f"  layer-chunks: {', '.join(f'{t}:{h}' for t, h in CHUNKS)} per layer "
        f"= {len(layers) * len(CHUNKS)} program(s), {REPEATS} timed run(s) "
        f"after {WARM_UPS} warm-up"
    )
    print(
        f"  weights: dequantized through tools/ssdai_reader.py into fp16; "
        f"numerics ceiling {MAX_MEAN_REL_ERROR}"
    )

    rng = np.random.default_rng(SEED)
    rows = []
    for layer in layers:
        try:
            layer_weights = load_layer_weights(weights, layer)
        except (RehearsalError, ValueError) as error:
            for chunk, history in CHUNKS:
                row = new_row(layer, chunk, history)
                row["note"] = f"{type(error).__name__}: {error}"
                rows.append(row)
                print(row_report(row), flush=True)
            continue
        for chunk, history in CHUNKS:
            row = measure_chunk(layer, chunk, history, layer_weights, rng)
            rows.append(row)
            print(row_report(row), flush=True)

    statuses = [0 if row["status"] == "measured" else 1 for row in rows]
    measured = [row["ane_ms"] for row in rows if row["ane_ms"] is not None]

    print("\n" + "=" * 66)
    if measured:
        mean_ms = sum(measured) / len(measured)
        total_s = sum(measured) / 1000.0
        head = f"  ANE mean {mean_ms:,.2f} ms per layer-chunk over {len(measured)} of {len(rows)} row(s)"
        reference = f"{probe.GPU_MS_PER_LAYER_CHUNK:,.0f} ms per layer-chunk"
        if mean_ms > 0:
            print(
                f"{head}, against the GPU reference {reference} = "
                f"{probe.GPU_MS_PER_LAYER_CHUNK / mean_ms:.2f}x"
            )
        else:
            print(
                f"{head} -- the clock resolved no time at all, so the GPU reference "
                f"({reference}) has no ratio to print"
            )
        print(f"  ANE, {len(measured)} layer-chunks: {total_s:8.2f} s of prediction wall")
        print(f"  GPU, the same shapes: {GPU_REFERENCE_S:.1f} s (measured, this machine)")
        if len(layers) == probe.FULL_ATTENTION_LAYERS:
            projected = GPU_PREFILL_TOTAL_S - GPU_REFERENCE_S + total_s
            print(
                f"  projected end-to-end prefill: {GPU_PREFILL_TOTAL_S} s -> {projected:.1f} s "
                f"({GPU_PREFILL_TOTAL_S / projected:.2f}x) with no integration overhead in it, "
                "so a bound and not a result"
            )
        else:
            print(
                f"  PROJECTION NOT AVAILABLE: the recorded {GPU_REFERENCE_S} s and "
                f"{GPU_PREFILL_TOTAL_S} s cover the probe's {probe.FULL_ATTENTION_LAYERS} "
                f"layers, this install declares {len(layers)}, so the totals are not "
                "comparable and no projection is printed"
            )
    else:
        print("  NOT MEASURED: no layer-chunk completed, so this run has no timing at all")

    gaps = [row["rel_err"] for row in rows if row["rel_err"] is not None]
    if gaps:
        print(f"  worst per-layer mean rel err {max(gaps):.4f}   ceiling {MAX_MEAN_REL_ERROR}")
    else:
        print(f"  numerics NOT MEASURED: no row has a finite gap; ceiling {MAX_MEAN_REL_ERROR}")
    bad = [row for row in rows if row["status"] != "measured"]
    if bad:
        print(
            f"  numerics FAILED: {len(bad)} row(s) errored or are outside the "
            f"{MAX_MEAN_REL_ERROR} ceiling"
        )

    payload = {
        "model": str(directory),
        "layers": layers,
        "max_mean_rel_error": MAX_MEAN_REL_ERROR,
        "gpu_reference_s": GPU_REFERENCE_S,
        "rows": rows,
        "status": 1 if any(statuses) else 0,
    }
    path = out_path()
    try:
        write_artifact(payload, path)
    except OSError as error:
        print(f"NOT WRITTEN: {path} -- {type(error).__name__}: {error}")
        statuses.append(1)
    else:
        print(f"wrote {path}")

    failed = sum(statuses)
    if failed:
        print(
            f"REHEARSAL INCOMPLETE: {failed} of {len(statuses)} check(s) failed -- "
            "no go/no-go can be read off this run."
        )
        return 1
    print(
        f"REHEARSAL COMPLETE: {len(rows)} layer-chunk(s) measured and within the "
        f"{MAX_MEAN_REL_ERROR} numerics ceiling."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
