#!/usr/bin/env python3
"""Model-free logit noise-floor gate for TinyTitan experiment captures.

The gate reads per-token logit captures written by another tool and compares
them element by element; it never loads a model.

Noise floor
-----------
    floor = 2 x (max absolute delta between two runs of the same build)

Supply the reference-vs-reference delta with ``--reference-delta`` (before
doubling) or measure it from ``--reference-repeat``, a second capture from the
reference build. With neither, the reference is compared with itself: a zero
delta and a zero floor, the strictest reading, not a determinism claim. A
candidate passes only when its global maximum absolute delta is at most the
floor (inclusive) and, for every prompt, its top-1 token at every position
equals the reference's. Top-1 is the first index carrying the maximum, so ties
break towards the lower token id.

Capture format
--------------
JSONL (one record per line; empty lines ignored) or JSON (an array of records,
one record object, or an object with a ``captures`` array). A record is:

    {"prompt_id": "basic-fact", "position": 0, "logits": [0.125, -2.5, ...]}

- ``prompt_id`` names the prompt and the exact token sequence scored; identical
  text tokenised differently is not the same prompt.
- ``position`` is a non-negative integer, contiguous from 0 for each prompt.
- ``logits`` is a non-empty array of finite numbers, one per vocabulary entry;
  every record in one capture has the same vocabulary length.

Precision: numbers are parsed as IEEE-754 binary64 and never rounded. A
producer that stores fp16 (for example the activation dumper's ``.f16`` slices)
records the fp16 grid, so a smaller delta is not observable in the file; measure
the floor from the same precision class and do not truncate digits more coarsely
than they were measured. Integers convert exactly.

Prompt sets, per-prompt lengths and vocabulary lengths must match across every
capture. Duplicate ``(prompt_id, position)`` records, position gaps, empty
captures, NaN/Infinity and non-numeric logits are refusals.

Statuses
--------
    0  every candidate is within the floor and no top-1 sequence diverges
    1  the comparison ran and at least one candidate failed a condition
    2  nothing was measured: a missing/empty/incompatible capture, an invalid
       floor, or a reference not top-1 stable against its repeat. The reason is
       printed as ``NOT MEASURED: ...``; no traceback is raised.

Usage
-----
    python3 benchmark/quality_floor.py REFERENCE.jsonl CANDIDATE.jsonl
    python3 benchmark/quality_floor.py REFERENCE.jsonl CANDIDATE.jsonl \\
        --reference-repeat REFERENCE_REPEAT.jsonl
    python3 benchmark/quality_floor.py REFERENCE.jsonl CANDIDATE.jsonl \\
        --reference-delta 0.000122
    python3 benchmark/quality_floor.py REFERENCE.jsonl CANDIDATE.jsonl --json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path


class Refused(Exception):
    """A capture or argument the gate cannot compare; reported as status 2."""


@dataclass(frozen=True)
class Capture:
    path: str
    prompts: dict[str, tuple[tuple[float, ...], ...]]
    vocab: int


@dataclass(frozen=True)
class Divergence:
    prompt_id: str
    position: int
    reference: int
    candidate: int

    def as_dict(self) -> dict[str, object]:
        return {
            "prompt_id": self.prompt_id,
            "position": self.position,
            "reference": self.reference,
            "candidate": self.candidate,
        }


@dataclass(frozen=True)
class Comparison:
    path: str
    max_abs_delta: float
    mean_abs_delta: float
    top1_divergences: tuple[Divergence, ...]

    @property
    def top1_identical(self) -> bool:
        return not self.top1_divergences


@dataclass(frozen=True)
class Candidate:
    comparison: Comparison
    within_floor: bool

    @property
    def passed(self) -> bool:
        return self.within_floor and self.comparison.top1_identical


@dataclass(frozen=True)
class Report:
    reference: Capture
    reference_delta: float
    floor: float
    floor_source: str
    reference_repeat: Comparison | None
    candidates: tuple[Candidate, ...]

    @property
    def passed(self) -> bool:
        return all(candidate.passed for candidate in self.candidates)


def _read_text(path: Path) -> str:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise Refused(f"cannot read capture {path}: {exc}") from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Refused(f"capture {path} is not UTF-8: {exc}") from exc
    if not text.strip():
        raise Refused(f"capture {path} is empty")
    return text


def _records(path: Path, text: str) -> list[object]:
    if path.suffix == ".jsonl":
        records: list[object] = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise Refused(
                    f"capture {path} line {lineno} is not JSON: {exc.msg} (column {exc.colno})"
                ) from exc
        return records
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise Refused(
            f"capture {path} is not valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})"
        ) from exc
    if isinstance(document, list):
        return document
    if isinstance(document, dict):
        captures = document.get("captures")
        return captures if isinstance(captures, list) else [document]
    raise Refused(
        f"capture {path}: JSON must be an array of records, a record object, "
        "or an object with a 'captures' array"
    )


def _logit_values(path: Path, record_number: int, logits: object) -> tuple[float, ...]:
    if not isinstance(logits, list) or not logits:
        raise Refused(f"capture {path} record {record_number}: 'logits' must be a non-empty array")
    values: list[float] = []
    for offset, value in enumerate(logits):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise Refused(
                f"capture {path} record {record_number}: logit {offset} is "
                f"{type(value).__name__}, not a number"
            )
        try:
            converted = float(value)
        except OverflowError as exc:
            raise Refused(f"logit {offset} in {path} record {record_number} is too large") from exc
        if not math.isfinite(converted):
            raise Refused(
                f"capture {path} record {record_number}: logit {offset} is not finite ({value!r})"
            )
        values.append(converted)
    return tuple(values)


def _parse_capture(path_text: str) -> Capture:
    path = Path(path_text)
    records = _records(path, _read_text(path))
    if not records:
        raise Refused(f"capture {path} holds no records")

    positions: dict[str, dict[int, tuple[float, ...]]] = {}
    seen: set[tuple[str, int]] = set()
    vocab: int | None = None
    for record_number, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise Refused(f"capture {path} record {record_number} is not a JSON object")
        prompt_id = record.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise Refused(
                f"capture {path} record {record_number}: 'prompt_id' must be a non-empty string"
            )
        position = record.get("position")
        if isinstance(position, bool) or not isinstance(position, int) or position < 0:
            raise Refused(
                f"capture {path} record {record_number}: 'position' must be a non-negative integer"
            )
        values = _logit_values(path, record_number, record.get("logits"))
        if vocab is None:
            vocab = len(values)
        elif len(values) != vocab:
            raise Refused(
                f"capture {path} record {record_number}: vocabulary length {len(values)} "
                f"differs from the capture's {vocab}"
            )
        key = (prompt_id, position)
        if key in seen:
            raise Refused(f"capture {path} repeats prompt {prompt_id!r} at position {position}")
        seen.add(key)
        positions.setdefault(prompt_id, {})[position] = values

    if not positions:
        raise Refused(f"capture {path} holds no records")
    prompts: dict[str, tuple[tuple[float, ...], ...]] = {}
    for prompt_id in sorted(positions):
        by_position = positions[prompt_id]
        ordered = sorted(by_position)
        if ordered != list(range(len(ordered))):
            missing = sorted(set(range(max(ordered) + 1)) - set(ordered))
            raise Refused(
                f"capture {path}: positions for prompt {prompt_id!r} are not "
                f"contiguous from 0 (missing {missing[0]}..{missing[-1]})"
            )
        prompts[prompt_id] = tuple(by_position[position] for position in ordered)
    return Capture(path=str(path), prompts=prompts, vocab=vocab or 0)


def _require_compatible(reference: Capture, other: Capture, role: str) -> None:
    reference_ids = set(reference.prompts)
    other_ids = set(other.prompts)
    if reference_ids != other_ids:
        detail = []
        missing = sorted(reference_ids - other_ids)
        extra = sorted(other_ids - reference_ids)
        if missing:
            detail.append(f"missing {', '.join(missing)}")
        if extra:
            detail.append(f"extra {', '.join(extra)}")
        raise Refused(
            f"{role} {other.path} does not cover the reference's prompt set: " + "; ".join(detail)
        )
    if reference.vocab != other.vocab:
        raise Refused(
            f"{role} {other.path} has vocabulary length {other.vocab}; the reference "
            f"has {reference.vocab}"
        )
    for prompt_id in sorted(reference.prompts):
        if len(reference.prompts[prompt_id]) != len(other.prompts[prompt_id]):
            raise Refused(
                f"{role} {other.path} scores prompt {prompt_id!r} at "
                f"{len(other.prompts[prompt_id])} positions; the reference scores "
                f"{len(reference.prompts[prompt_id])}"
            )


def _top1(logits: tuple[float, ...]) -> int:
    """First index carrying the maximum, so ties break towards the lower id."""
    return max(range(len(logits)), key=logits.__getitem__)


def _compare(reference: Capture, candidate: Capture) -> Comparison:
    max_abs_delta = 0.0
    total_abs_delta = 0.0
    compared = 0
    divergences: list[Divergence] = []
    for prompt_id in sorted(reference.prompts):
        for position, (reference_row, candidate_row) in enumerate(
            zip(reference.prompts[prompt_id], candidate.prompts[prompt_id], strict=True)
        ):
            for reference_value, candidate_value in zip(reference_row, candidate_row, strict=True):
                delta = abs(reference_value - candidate_value)
                max_abs_delta = max(max_abs_delta, delta)
                total_abs_delta += delta
                compared += 1
            reference_top1 = _top1(reference_row)
            candidate_top1 = _top1(candidate_row)
            if reference_top1 != candidate_top1:
                divergences.append(Divergence(prompt_id, position, reference_top1, candidate_top1))
    return Comparison(
        path=candidate.path,
        max_abs_delta=max_abs_delta,
        mean_abs_delta=total_abs_delta / compared,
        top1_divergences=tuple(divergences),
    )


def _floor(delta: float) -> float:
    if not math.isfinite(delta) or delta < 0:
        raise Refused(f"the reference-vs-reference delta must be finite and >= 0, got {delta!r}")
    return 2.0 * delta


def evaluate(
    reference_path: str,
    candidate_paths: list[str] | tuple[str, ...],
    *,
    reference_repeat: str | None = None,
    reference_delta: float | None = None,
) -> Report:
    """Parse every capture, measure the floor, and judge every candidate."""
    if not candidate_paths:
        raise Refused("at least one candidate capture is required")
    if reference_delta is not None and reference_repeat is not None:
        raise Refused("supply --reference-delta or --reference-repeat, not both")

    reference = _parse_capture(reference_path)
    repeat_comparison: Comparison | None = None
    if reference_delta is not None:
        floor, source = _floor(reference_delta), "supplied"
    elif reference_repeat is not None:
        repeat = _parse_capture(reference_repeat)
        _require_compatible(reference, repeat, "reference repeat")
        repeat_comparison = _compare(reference, repeat)
        if not repeat_comparison.top1_identical:
            first = repeat_comparison.top1_divergences[0]
            raise Refused(
                "the reference repeat is not top-1 stable "
                f"({first.prompt_id!r} position {first.position}: reference "
                f"{first.reference}, repeat {first.candidate}), so no deterministic "
                "floor can be measured"
            )
        floor = _floor(repeat_comparison.max_abs_delta)
        source = f"reference repeat {repeat.path}"
    else:
        floor, source = _floor(0.0), "self-comparison (no same-build repeat supplied)"

    candidates: list[Candidate] = []
    for path in candidate_paths:
        captured = _parse_capture(path)
        _require_compatible(reference, captured, "candidate")
        comparison = _compare(reference, captured)
        candidates.append(
            Candidate(comparison=comparison, within_floor=comparison.max_abs_delta <= floor)
        )
    return Report(
        reference=reference,
        reference_delta=floor / 2.0,
        floor=floor,
        floor_source=source,
        reference_repeat=repeat_comparison,
        candidates=tuple(candidates),
    )


def _number(value: float) -> str:
    """Nine significant digits keep a small delta readable and a zero a zero."""
    return f"{value:.9g}"


def render(report: Report) -> str:
    """The stable human-readable report."""
    reference = report.reference
    lines = [
        "Logit quality floor",
        f"reference: {reference.path} ({len(reference.prompts)} prompt(s), "
        f"{sum(len(rows) for rows in reference.prompts.values())} token(s))",
        f"noise floor: {_number(report.floor)} "
        f"(2 x {_number(report.reference_delta)}, {report.floor_source})",
    ]
    if report.reference_repeat is not None:
        lines.append(
            f"reference repeat: {report.reference_repeat.path} "
            f"(max |delta| {_number(report.reference_repeat.max_abs_delta)})"
        )
    for candidate in report.candidates:
        comparison = candidate.comparison
        lines.append("")
        lines.append(f"candidate: {comparison.path}")
        lines.append(f"  max |delta|: {_number(comparison.max_abs_delta)}")
        lines.append(f"  mean |delta|: {_number(comparison.mean_abs_delta)}")
        lines.append(f"  top-1 divergences: {len(comparison.top1_divergences)}")
        for divergence in comparison.top1_divergences:
            lines.append(
                f"    {divergence.prompt_id} position {divergence.position}: "
                f"reference {divergence.reference}, candidate {divergence.candidate}"
            )
        reasons: list[str] = []
        if not candidate.within_floor:
            reasons.append(
                f"max |delta| {_number(comparison.max_abs_delta)} exceeds the floor "
                f"{_number(report.floor)}"
            )
        if not comparison.top1_identical:
            reasons.append(f"{len(comparison.top1_divergences)} top-1 divergence(s)")
        if not reasons:
            reasons.append("within the floor and every prompt's top-1 sequence is identical")
        lines.append(f"  verdict: {'PASS' if candidate.passed else 'FAIL'} ({'; '.join(reasons)})")
    lines.append("")
    lines.append(f"OVERALL: {'PASS' if report.passed else 'FAIL'}")
    return "\n".join(lines)


def as_dict(report: Report) -> dict[str, object]:
    """The stable JSON report."""
    repeat = report.reference_repeat
    return {
        "reference": report.reference.path,
        "reference_prompts": len(report.reference.prompts),
        "reference_tokens": sum(len(rows) for rows in report.reference.prompts.values()),
        "noise_floor": {
            "reference_delta": report.reference_delta,
            "floor": report.floor,
            "source": report.floor_source,
        },
        "reference_repeat": (
            {
                "path": repeat.path,
                "max_abs_delta": repeat.max_abs_delta,
                "mean_abs_delta": repeat.mean_abs_delta,
            }
            if repeat is not None
            else None
        ),
        "candidates": [
            {
                "path": candidate.comparison.path,
                "max_abs_delta": candidate.comparison.max_abs_delta,
                "mean_abs_delta": candidate.comparison.mean_abs_delta,
                "top1_divergences": [
                    divergence.as_dict() for divergence in candidate.comparison.top1_divergences
                ],
                "within_floor": candidate.within_floor,
                "top1_identical": candidate.comparison.top1_identical,
                "passed": candidate.passed,
            }
            for candidate in report.candidates
        ],
        "passed": report.passed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare per-token logit captures against a same-build noise floor. "
            "The first capture is the reference; every later capture is a candidate."
        ),
        epilog=(
            "Capture format: JSONL or JSON records of "
            '{"prompt_id": str, "position": int, "logits": [finite numbers]}. '
            "Status: 0 pass, 1 fail, 2 not measured."
        ),
    )
    parser.add_argument("captures", nargs="+", metavar="CAPTURE")
    parser.add_argument(
        "--reference-repeat",
        help="second capture from the reference build; its max delta sets the floor",
    )
    parser.add_argument(
        "--reference-delta",
        type=float,
        help="reference-vs-reference max absolute delta to double into the floor",
    )
    parser.add_argument("--json", action="store_true", help="print the JSON report")
    args = parser.parse_args(argv)

    if len(args.captures) < 2:
        print("NOT MEASURED: the gate needs a reference and at least one candidate capture")
        return 2
    try:
        report = evaluate(
            args.captures[0],
            args.captures[1:],
            reference_repeat=args.reference_repeat,
            reference_delta=args.reference_delta,
        )
    except Refused as exc:
        print(f"NOT MEASURED: {exc}")
        return 2

    print(json.dumps(as_dict(report), indent=2) if args.json else render(report))
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
