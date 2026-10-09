#!/usr/bin/env python3
"""Item 8 gate: is 3-bit worth a repacker, a kernel, and a re-download?

Two questions decide it, and both are answerable from the installed weights
without writing any runtime code:

  1. QUALITY. Quantize the real routed expert, shared expert, attention and
     router tensors to 3-bit affine (group 64, the format TinyTitan already
     uses) and measure the error that adds on top of the weights the model runs
     today. The routed experts are the streamed part of an MoE checkpoint and
     most of its bytes, so they are sampled from `packed_experts/`, not only
     from the resident file.

  2. PACKING. 3 bits does not divide a 32-bit word. Measure how much of the
     nominal 25% byte saving survives realistic packing, and how many weights
     straddle a word boundary and so cost an extra load. TinyTitan's own 6-bit
     experience is the precedent: non-power-of-two packing measured 46.8 GB/s
     against 60 for both 4-bit and 8-bit, and 6-bit was withdrawn.

Decode is bandwidth-bound, so 3-bit only pays if the byte saving is real AND
the unpack does not push the kernel into being ALU-bound instead.

What this file cannot decide alone: "3-bit is N times worse than 4-bit" needs
weights finer than either. A 4-bit install's values ARE their own 4-bit
reference -- requantizing them to 4 bits is exact -- so no ratio to 4-bit
exists inside it. Point `TINYTITAN_BENCH_REFERENCE` at an 8-bit install of the
same architecture to get that half; without it the run reports the added error
it can measure and marks the comparison NOT MEASURED, and exits 1.

    TINYTITAN_BENCH_MODEL=models/ornith-1.5_35B_A3B_4Bit \
    TINYTITAN_BENCH_REFERENCE=models/ornith-1.5_35B_A3B_8Bit \
      python benchmark/tinytitan_3bit_probe.py
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
from dataclasses import dataclass

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

# The repo's own reader: it takes each tensor's width from the entry rather
# than assuming 4-bit, which matters because the manifest gives the router a
# different width from the attention.
from ssdai_reader import GROUP_SIZE, PackedExperts, SSDAIWeights  # noqa: E402

GROUP = GROUP_SIZE
BENCH_MODEL_ENV = "TINYTITAN_BENCH_MODEL"
REFERENCE_ENV = "TINYTITAN_BENCH_REFERENCE"
OUT_ENV = "TINYTITAN_PROBE_OUT"
DEFAULT_MODEL = ROOT / "models/ornith-1.5_35B_A3B_4Bit"
DEFAULT_OUT = ROOT / ".build/benchmark-results"
ARTIFACT = "3bit-probe.json"

ADDED_BITS = (2, 3)  # against the weights the model runs today
PACKING_BITS = (2, 3, 4, 6, 8)
SAMPLES_PER_ROLE = 2
ROUTED_EXPERT_SAMPLE = 4
ROUTED_EXPERT_LAYERS = 1
ROLES = ("routedExpert", "sharedExpert", "attention", "router")

RESIDENT_ROLES = {
    "attention": "self_attn.",
    "sharedExpert": "mlp.shared_expert.",
    "router": "mlp.gate.weight",
}
EXPERT_TENSORS = ("gate", "up", "down")


class ConfigError(ValueError):
    """The run cannot start: say which variable or file is wrong."""


@dataclass
class Tensor:
    """One sampled weight, and where the reference install keeps its twin."""

    label: str
    role: str
    array: np.ndarray
    name: str | None = None  # resident index key
    locator: tuple[int, int, str] | None = None  # (layer, expert, tensor)


def probe_model() -> str:
    """The install to probe, named when the run starts.

    Read at call time, not at import: `TINYTITAN_BENCH_MODEL` is how every
    other harness here names its model, and an import-frozen path makes the
    variable a no-op for anyone who exports it after the module is loaded.
    """
    return os.environ.get(BENCH_MODEL_ENV, "").strip() or str(DEFAULT_MODEL)


def reference_model() -> str | None:
    """A wider install of the same architecture, or None.

    A blank value is not a directory: `Path("")` is `PosixPath('.')`, which
    exists, so the blank case has to be named before it becomes a path.
    """
    return os.environ.get(REFERENCE_ENV, "").strip() or None


def out_dir() -> pathlib.Path:
    value = os.environ.get(OUT_ENV, "").strip()
    return pathlib.Path(value) if value else DEFAULT_OUT


def open_weights(directory: str, env: str = BENCH_MODEL_ENV) -> SSDAIWeights:
    """Read an install, naming the variable that named it.

    `env` is not decoration: the same routine opens both the probed install and
    the reference, and a refusal that blames `TINYTITAN_BENCH_MODEL` for a bad
    `TINYTITAN_BENCH_REFERENCE` sends the reader reinstalling the wrong thing.
    """
    path = pathlib.Path(str(directory))
    if not path.is_dir():
        raise ConfigError(f"{env}='{path}' is not a directory; install one or name it")
    for needed in ("model_weights.bin", "manifest.json"):
        if not (path / needed).is_file():
            raise ConfigError(f"{path} has no {needed}; point {env} at a .ssdai install")
    try:
        return SSDAIWeights(path)
    except ConfigError:
        raise
    except Exception as error:  # a truncated or foreign index is this install's fault to name
        raise ConfigError(
            f"{path / 'model_weights.bin'} cannot be read: {type(error).__name__}: {error}"
        ) from error


def open_experts(directory: str, manifest: dict) -> PackedExperts | None:
    """The streamed experts, or None when this checkpoint has none."""
    root = pathlib.Path(str(directory)) / "packed_experts"
    if not root.is_dir():
        # No streamed experts is a measurement gap the report names, not a
        # refusal: a dense checkpoint has no `packed_experts/` at all, and the
        # reader still needs the run to say which roles it did not reach.
        return None
    if not (root / "layout.json").is_file():
        raise ConfigError(f"{root} has no layout.json; the expert offsets cannot be read")
    experts = PackedExperts(pathlib.Path(str(directory)))
    for layer in range(len(experts.layout.get("layers") or [])):
        blob = root / f"layer_{layer:02d}.bin"
        if not blob.is_file():
            raise ConfigError(f"{root / 'layout.json'} describes {blob.name}, which is not there")
    return experts


def sample_tensors(directory: str) -> list[Tensor]:
    """One representative slice per role, routed experts included.

    Roles are found in the install rather than listed by hardcoded name: the
    names a checkpoint uses are its own, and a sample taken by pattern cannot
    silently stop containing a role the way a stale name list does.
    """
    weights = open_weights(directory)
    tensors: list[Tensor] = []
    for role in ("attention", "sharedExpert", "router"):
        pattern = RESIDENT_ROLES[role]
        matches = [n for n in weights.names() if pattern in n or n.endswith(pattern)]
        for name in matches[:SAMPLES_PER_ROLE]:
            tensors.append(Tensor(label=name, role=role, array=weights.get(name), name=name))
    experts = open_experts(directory, weights.manifest)
    if experts is not None:
        available = experts.layout.get("layers") or []
        count = int(experts.layout.get("expertsPerLayer") or 0)
        for layer in range(min(ROUTED_EXPERT_LAYERS, len(available))):
            usable = min(ROUTED_EXPERT_SAMPLE, len(available[layer].get("experts") or []), count)
            for expert in range(usable):
                for tname in EXPERT_TENSORS:
                    label = f"packed_experts layer {layer} expert {expert} {tname}"
                    tensors.append(
                        Tensor(
                            label=label,
                            role="routedExpert",
                            array=experts.tensor(layer, expert, tname),
                            locator=(layer, expert, tname),
                        )
                    )
    return tensors


def sample_counts(tensors: list[Tensor]) -> dict[str, int]:
    counts = dict.fromkeys(ROLES, 0)
    for tensor in tensors:
        counts[tensor.role] += 1
    return counts


def requantize(w: np.ndarray, bits: int) -> np.ndarray:
    """Quantize to `bits` affine per group of 64 and dequantize, as the format does."""
    rows, cols = w.shape
    g = w.reshape(rows, cols // GROUP, GROUP)
    lo = g.min(-1, keepdims=True)
    hi = g.max(-1, keepdims=True)
    levels = (1 << bits) - 1
    scale = (hi - lo) / levels
    scale = np.where(scale == 0, 1.0, scale)
    q = np.clip(np.rint((g - lo) / scale), 0, levels)
    return (q * scale + lo).reshape(rows, cols)


def requantize_error(w: np.ndarray, bits: int) -> float:
    """Mean absolute error of `bits` against `w`, relative to `w`'s own scale."""
    error = float(np.abs(requantize(w, bits) - w).mean())
    return error / max(float(np.abs(w).mean()), 1e-12)


def ratio_against(reference: np.ndarray) -> float | None:
    """3-bit's error over 4-bit's, both against weights finer than either.

    None when the reference is no finer than 4-bit: its own requantize-to-4
    error is 0.0 by construction, and dividing by that is what made the old
    headline print 207836508750.92x.
    """
    e4 = requantize_error(reference, 4)
    if e4 <= 0.0:
        return None
    return requantize_error(reference, 3) / e4


def reference_array(
    tensor: Tensor, weights: SSDAIWeights | None, experts: PackedExperts | None
) -> np.ndarray | None:
    """The same weight from the reference install, or None if it is not there."""
    if weights is None:
        return None
    if tensor.locator is not None:
        if experts is None:
            return None
        layer, expert, tname = tensor.locator
        record = (experts.layout.get("layers") or [{}])[layer].get("experts") or []
        if expert >= len(record) or tname not in record[expert].get("tensors", {}):
            return None
        return experts.tensor(layer, expert, tname)
    if tensor.name not in weights.entries:
        return None
    return weights.get(tensor.name)


def word_crossings(bits: int) -> tuple[int, float]:
    """Weights in one group that straddle a 32-bit word, and the extra loads.

    A group is word-aligned, so value `i` occupies bits `[i*bits, (i+1)*bits)`;
    a boundary strictly inside that span needs a second load and a shift.
    """
    if bits and 32 % bits == 0:
        return 0, 0.0
    boundaries = (GROUP * bits) // 32
    return boundaries, boundaries / GROUP


def packing_analysis() -> list[dict]:
    """Bytes per 64-weight group and unpack cost, for the packings a real
    kernel could use."""
    rows = []
    for bits in PACKING_BITS:
        payload_bits = GROUP * bits
        # Scheme A: bit-exact stream (what TinyTitan's affine_quant_value does).
        stream_bytes = payload_bits / 8
        # Scheme B: whole values per 32-bit word, wasting the remainder.
        per_word = 32 // bits
        words = -(-GROUP // per_word)
        padded_bytes = words * 4
        # Metadata: bf16 scale + bias per group.
        meta = 4
        crossings, extra_loads = word_crossings(bits)
        rows.append(
            {
                "bits": bits,
                "group_bytes_stream": stream_bytes + meta,
                "group_bytes_padded": padded_bytes + meta,
                "vs_4bit_stream": (stream_bytes + meta) / (GROUP * 4 / 8 + meta),
                "vs_4bit_padded": (padded_bytes + meta) / (GROUP * 4 / 8 + meta),
                "power_of_two": crossings == 0,
                "crossing_values": crossings,
                "extra_loads_per_weight": extra_loads,
            }
        )
    return rows


def header_lines(weights: SSDAIWeights, directory: str, counts: dict[str, int]) -> list[str]:
    quant = weights.manifest.get("quant", {})
    widths = ", ".join(
        f"{role} {spec.get('weightBits')}bit" for role, spec in sorted(quant.items())
    )
    sampled = ", ".join(f"{role} {counts[role]}" for role in ROLES)
    return [
        f"# install: {directory} ({weights.manifest.get('modelID', 'unknown modelID')})",
        f"# declared widths: {widths or 'the manifest names none'}",
        f"# tensors sampled: {sampled}",
    ]


def quality_report(
    tensors: list[Tensor], directory: str, reference: str | None, declared: set[str]
) -> tuple[list[str], int]:
    if not tensors:
        return (["NOT MEASURED: every role -- no tensor in this install matched one"], 1)
    ref_weights = open_weights(reference, REFERENCE_ENV) if reference else None
    ref_experts = open_experts(reference, ref_weights.manifest) if reference else None
    per_role: dict[str, dict[str, list[float]]] = {}
    missing = 0
    not_finer = 0
    for tensor in tensors:
        bucket = per_role.setdefault(tensor.role, {"added": {}, "ratio": []})
        for bits in ADDED_BITS:
            bucket["added"].setdefault(bits, []).append(requantize_error(tensor.array, bits))
        if reference:
            twin = reference_array(tensor, ref_weights, ref_experts)
            if twin is None:
                missing += 1
                continue
            value = ratio_against(twin)
            if value is None:
                not_finer += 1
                continue
            bucket["ratio"].append(value)

    lines = ["== quality: error 3-bit adds to the weights the model runs today =="]
    lines.append(
        f"{'role':<14} {'n':>3} {'2-bit added':>12} {'3-bit added':>12} {'3-vs-4 ratio':>13}"
    )
    status = 0
    for role in ROLES:
        bucket = per_role.get(role)
        if not bucket:
            # A role the manifest declares and the sample never found is a gap
            # in this run; a role the checkpoint does not have is not.
            if role in declared:
                lines.append(
                    f"NOT MEASURED: {role} -- the manifest declares it and no tensor matched"
                )
                status = 1
            else:
                lines.append(
                    f"{role:<14} {'0':>3} {'-':>12} {'-':>12} {'not in this checkpoint':>13}"
                )
            continue
        added = bucket["added"]
        ratios = bucket["ratio"]
        ratio_text = f"{float(np.median(ratios)):.2f}x" if ratios else "not measured"
        lines.append(
            f"{role:<14} {len(added[3]):>3} {float(np.median(added[2])):12.4f} {float(np.median(added[3])):12.4f} {ratio_text:>13}"
        )
    if not reference:
        lines.append(
            "NOT MEASURED: the 3-vs-4-bit ratio. A 4-bit install is its own 4-bit reference, "
            f"so its requantize-to-4-bit error is 0.0 and the ratio has no denominator; set "
            f"{REFERENCE_ENV} to a wider install of the same architecture to measure it."
        )
        status = 1
    elif missing or not_finer:
        gaps = []
        if missing:
            gaps.append(f"{missing} had no twin there at all")
        if not_finer:
            gaps.append(
                f"{not_finer} had a twin stored no finer than its own 4-bit grid, so it has no "
                "4-bit error to divide by -- point the reference at an 8-bit install"
            )
        lines.append(
            f"NOT MEASURED: the 3-vs-4-bit ratio for {missing + not_finer} of {len(tensors)} "
            f"sampled tensor(s) against '{reference}': " + "; ".join(gaps) + ". "
            f"The medians above cover the {len(tensors) - missing - not_finer} that had a finer twin."
        )
        status = 1
    return lines, status


def packing_report() -> tuple[list[str], int]:
    lines = [
        "",
        "== packing: bytes per 64-weight group (incl. bf16 scale+bias) and unpack cost ==",
        (
            f"{'bits':>5} {'stream B':>9} {'padded B':>9} {'vs 4-bit (stream)':>18} "
            f"{'vs 4-bit (padded)':>18} {'PoT':>5} {'cross/group':>12} {'extra loads/weight':>19}"
        ),
    ]
    rows = packing_analysis()
    for r in rows:
        lines.append(
            f"{r['bits']:>5} {r['group_bytes_stream']:>9.1f} {r['group_bytes_padded']:>9.1f} "
            f"{r['vs_4bit_stream']:>17.3f}x {r['vs_4bit_padded']:>17.3f}x "
            f"{'yes' if r['power_of_two'] else 'NO':>5} {r['crossing_values']:>12d} "
            f"{r['extra_loads_per_weight']:>19.4f}"
        )
    three = next(r for r in rows if r["bits"] == 3)
    six = next(r for r in rows if r["bits"] == 6)
    lines.append(
        f"\n  3-bit byte saving vs 4-bit: {(1 - three['vs_4bit_stream']) * 100:.1f}% (bit-exact stream), "
        f"{(1 - three['vs_4bit_padded']) * 100:.1f}% (word-padded)"
    )
    lines.append(
        f"  3-bit unpack cost: {three['crossing_values']} of {GROUP} weights cross a word boundary, "
        f"{three['extra_loads_per_weight']:.4f} extra loads per weight"
    )
    lines.append(
        f"  6-bit, for reference (withdrawn after measuring 46.8 GB/s against 60): "
        f"{(1 - six['vs_4bit_stream']) * 100:.1f}% / {(1 - six['vs_4bit_padded']) * 100:.1f}%, "
        f"{six['crossing_values']} crossings per group"
    )
    return lines, 0


def artifact_payload(
    weights: SSDAIWeights,
    directory: str,
    counts: dict[str, int],
    quality: list[str],
    packing: list[str],
    reference: str | None,
) -> dict:
    return {
        "install": directory,
        "model_id": weights.manifest.get("modelID"),
        "reference": reference,
        "roles": counts,
        "quality": quality,
        "packing": packing,
    }


def run() -> int:
    directory = probe_model()
    reference = reference_model()
    weights = open_weights(directory)
    tensors = sample_tensors(directory)
    counts = sample_counts(tensors)
    print("\n".join(header_lines(weights, directory, counts)))
    declared = set(weights.manifest.get("quant", {}))
    quality, quality_status = quality_report(tensors, directory, reference, declared)
    print("\n" + "\n".join(quality))
    packing, packing_status = packing_report()
    print("\n".join(packing))
    status = max(quality_status, packing_status)

    out = out_dir()
    out.mkdir(parents=True, exist_ok=True)
    target = out / ARTIFACT
    target.write_text(
        json.dumps(
            artifact_payload(weights, directory, counts, quality, packing, reference), indent=2
        )
        + "\n"
    )
    print(f"\nwrote {target}")
    if status:
        print("PROBE INCOMPLETE: at least one deciding number was not measured.")
    else:
        print(
            "PROBE OK: added error measured for every role, and the 3-vs-4-bit ratio against a wider reference."
        )
    return status


def main() -> int:
    try:
        return run()
    except ConfigError as error:
        print(f"REFUSED: {error} No measurement was taken.")
        return 2


if __name__ == "__main__":
    sys.exit(main())
