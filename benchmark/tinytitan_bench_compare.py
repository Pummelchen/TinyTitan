#!/usr/bin/env python3
"""Compare TinyTitan benchmark configurations from their saved result files.

The comparator is file-based on purpose: ``tinytitan_benchmark.py`` loads a
model, so a runner mode here would need an install and a server. It ingests the
``aggregate.json`` files that driver writes and refuses to compare results that
were not recorded over the same workload.

Inputs
------

Each positional argument is either a path to an ``aggregate.json`` or a
directory that holds one. A result file is the object written by
``tinytitan_benchmark.py``: ``config``, ``cache_mode``, ``mtp_config``,
``runtime_profile``, ``results`` (one row per prompt, each carrying
``capability`` and ``prompt_tokens``) and ``summary`` (``measured``,
``prompts_configured``, ``total_requests``, ``avg_decode_tok_s``,
``avg_footer_decode_tok_s``, ``verified``).

The exact commands that produce those inputs, from the repository root:

    cd benchmark
    python3 tinytitan_benchmark.py models/<install>

The driver writes ``benchmark/benchmark-results/bench-<config>-<stamp>/
aggregate.json``. Produce one file per configuration or model, then:

    python3 benchmark/tinytitan_bench_compare.py \\
        benchmark/benchmark-results/bench-cache_off_mtp_off_4bit-.../aggregate.json \\
        benchmark/benchmark-results/bench-cache_on_mtp_off_4bit-.../aggregate.json

Workload comparability
----------------------

Two results are comparable only when they recorded the same number of configured
prompts, the same set of prompt capability names, and the same ``prompt_tokens``
for every one of those prompts. Completion tokens may differ: two models that
decode different lengths still ran the same input workload. A differing prompt
set or prompt-token count is a refusal, never a padded or averaged table.

Provenance
----------

Older aggregates that omit ``summary.measured`` or ``prompts_configured`` are
accepted: completeness is derived from the recorded rows, with a ``Notes:`` line.

The table carries git commit, chip/hw.model, macOS, flags/config and repeats for every row.
A result file may record them in a ``provenance`` object:

    "provenance": {"git_commit": "...", "chip": "...", "macos": "...",
                   "flags": "...", "repeats": 3, "model": "..."}

Legacy top-level ``commit``, ``chip``, ``macos`` and ``model`` are accepted.
``flags`` is derived from ``cache_mode``, ``mtp_config`` and
``runtime_profile`` when the file does not record it, and ``repeats`` is read
from ``provenance.repeats`` or ``summary.repeats``. A field the file does not
record is printed as ``not recorded`` and is ``null`` in ``--json``: the
comparator never fills it from the machine it happens to run on.

Output and statuses
-------------------

The default output is a stable Markdown report; ``--json`` prints one JSON object.
Statuses:

    0  the report was produced and every included run is complete
    1  the report was produced, but a run is incomplete or failed (the named
       reasons are printed and the JSON carries them)
    2  no comparison exists: fewer than two inputs, a missing/unreadable file,
       a schema the comparator cannot read, or workloads that are not
       comparable. The reason is printed as ``NOT MEASURED: ...``; no traceback.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path


class Refused(Exception):
    """An input the comparator cannot compare; reported as status 2."""


@dataclass(frozen=True)
class Provenance:
    git_commit: str | None
    chip: str | None
    macos: str | None
    flags: str
    repeats: str | None
    model: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "git_commit": self.git_commit,
            "chip": self.chip,
            "macos": self.macos,
            "flags": self.flags,
            "repeats": self.repeats,
            "model": self.model,
        }


@dataclass(frozen=True)
class Measurement:
    path: str
    config: str
    cache_mode: str
    mtp_config: str
    prompts: tuple[str, ...]
    prompt_tokens: dict[str, int]
    configured: int
    avg_decode: float | None
    verified: int
    incomplete: tuple[str, ...]
    notes: tuple[str, ...]
    provenance: Provenance

    @property
    def complete(self) -> bool:
        return not self.incomplete


def _text(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _scalar_text(value: object) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return str(value)
    return _text(value)


def _require_text(value: object, label: str, path: Path) -> str:
    text = _text(value)
    if text is None:
        raise Refused(f"result {path} has no non-empty {label}")
    return text


def _require_int(value: object, label: str, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Refused(f"result {path} has a {label} that is not a non-negative integer")
    return value


def _derived_flags(data: dict[str, object], cache_mode: str, mtp_config: str) -> str:
    parts = [f"cache={cache_mode}", f"mtp={mtp_config}"]
    profile = data.get("runtime_profile")
    if isinstance(profile, dict):
        for key in sorted(profile):
            value = profile[key]
            rendered = json.dumps(value) if isinstance(value, bool) or value is None else str(value)
            parts.append(f"{key}={rendered}")
    return " ".join(parts)


def _provenance(data: dict[str, object], summary: dict[str, object], path: Path) -> Provenance:
    block = data.get("provenance")
    if block is None:
        block = {}
    if not isinstance(block, dict):
        raise Refused(f"result {path} has a non-object 'provenance'")
    record: dict[str, object] = block
    raw_flags = record.get("flags")
    if isinstance(raw_flags, dict):
        flags = json.dumps(raw_flags, sort_keys=True, separators=(",", ":"))
    else:
        flags = _text(raw_flags) or ""
    repeats = _scalar_text(record.get("repeats")) or _scalar_text(summary.get("repeats"))
    return Provenance(
        git_commit=(
            _text(record.get("git_commit"))
            or _text(record.get("commit"))
            or _text(data.get("commit"))
        ),
        chip=(
            _text(record.get("chip")) or _text(record.get("hw_model")) or _text(data.get("chip"))
        ),
        macos=_text(record.get("macos")) or _text(data.get("macos")),
        flags=flags,
        repeats=repeats,
        model=_text(record.get("model")) or _text(data.get("model")),
    )


def _measurement(data: dict[str, object], path: Path) -> Measurement:
    config = _require_text(data.get("config"), "'config'", path)
    cache_mode = _require_text(data.get("cache_mode"), "'cache_mode'", path)
    mtp_config = _require_text(data.get("mtp_config"), "'mtp_config'", path)

    results = data.get("results")
    if not isinstance(results, list) or not results:
        raise Refused(f"result {path} has no non-empty 'results' array")
    prompts: list[str] = []
    prompt_tokens: dict[str, int] = {}
    for index, row in enumerate(results, start=1):
        if not isinstance(row, dict):
            raise Refused(f"result {path} row {index} is not a JSON object")
        capability = _require_text(row.get("capability"), f"capability in row {index}", path)
        if capability in prompt_tokens:
            raise Refused(f"result {path} repeats prompt {capability!r}")
        prompts.append(capability)
        prompt_tokens[capability] = _require_int(
            row.get("prompt_tokens"), f"prompt_tokens in row {index}", path
        )

    summary = data.get("summary")
    if not isinstance(summary, dict):
        raise Refused(f"result {path} has no 'summary' object")
    notes: list[str] = []
    configured_raw = summary.get("prompts_configured")
    if configured_raw is None:
        configured = len(results)
        notes.append("prompts_configured not recorded; using the row count")
    else:
        configured = _require_int(configured_raw, "'prompts_configured'", path)
        if configured == 0:
            raise Refused(f"result {path} configured zero prompts")

    measured_raw = summary.get("measured")
    if measured_raw is not None and not isinstance(measured_raw, bool):
        raise Refused(f"result {path} has a 'measured' that is not a boolean")
    total_requests = summary.get("total_requests")
    total: int | None = None
    if total_requests is not None:
        total = _require_int(total_requests, "'total_requests'", path)

    avg: float | None = None
    raw_avg = summary.get("avg_decode_tok_s")
    if raw_avg is None and measured_raw is not False:
        raw_avg = summary.get("avg_footer_decode_tok_s")
    if raw_avg is None:
        if measured_raw is True:
            notes.append("no average decode rate recorded")
    elif isinstance(raw_avg, bool) or not isinstance(raw_avg, (int, float)):
        raise Refused(f"result {path} has a non-numeric average decode rate")
    else:
        try:
            avg = float(raw_avg)
        except OverflowError as exc:
            raise Refused(f"result {path} average decode rate is too large") from exc
        if not math.isfinite(avg):
            notes.append("the recorded average decode rate is not finite")
            avg = None

    if measured_raw is None:
        measured = len(results) == configured and total in (None, len(results)) and avg is not None
        notes.append("measured not recorded; completeness derived from the recorded rows")
    else:
        measured = measured_raw

    incomplete: list[str] = []
    if not measured:
        incomplete.append(
            "measured not recorded and the rows do not establish a complete run"
            if measured_raw is None
            else "the run reports measured=false"
        )
    if len(results) != configured:
        incomplete.append(f"{len(results)} of {configured} prompts recorded")
    if total is not None and total != len(results):
        incomplete.append(f"total_requests={total} but {len(results)} rows recorded")
    if measured and avg is None:
        incomplete.append("no average decode rate recorded")

    verified_raw = summary.get("verified")
    if verified_raw is None:
        verified = sum(1 for row in results if row.get("verified") is True)
    else:
        verified = _require_int(verified_raw, "'verified'", path)
        if verified > len(results):
            raise Refused(
                f"result {path} verified {verified} answers but recorded {len(results)} rows"
            )

    provenance = _provenance(data, summary, path)
    if not provenance.flags:
        provenance = Provenance(
            git_commit=provenance.git_commit,
            chip=provenance.chip,
            macos=provenance.macos,
            flags=_derived_flags(data, cache_mode, mtp_config),
            repeats=provenance.repeats,
            model=provenance.model,
        )
    return Measurement(
        path=str(path),
        config=config,
        cache_mode=cache_mode,
        mtp_config=mtp_config,
        prompts=tuple(prompts),
        prompt_tokens=prompt_tokens,
        configured=configured,
        avg_decode=avg,
        verified=verified,
        incomplete=tuple(incomplete),
        notes=tuple(notes),
        provenance=provenance,
    )


def _load(path_text: str) -> Measurement:
    path = Path(path_text)
    if path.is_dir():
        path = path / "aggregate.json"
    if not path.is_file():
        raise Refused(f"result file {path} does not exist or is not a readable file")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise Refused(f"cannot read result file {path}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise Refused(
            f"result file {path} is not valid JSON: {exc.msg} "
            f"(line {exc.lineno}, column {exc.colno})"
        ) from exc
    if not isinstance(data, dict):
        raise Refused(f"result file {path} is not a JSON object")
    return _measurement(data, path)


def _require_comparable(measurements: list[Measurement]) -> None:
    first = measurements[0]
    for other in measurements[1:]:
        if other.configured != first.configured:
            raise Refused(
                f"{other.config!r} configured {other.configured} prompts; "
                f"{first.config!r} configured {first.configured}"
            )
        missing = sorted(set(first.prompts) - set(other.prompts))
        extra = sorted(set(other.prompts) - set(first.prompts))
        if missing or extra:
            detail = []
            if missing:
                detail.append(f"missing {', '.join(missing)}")
            if extra:
                detail.append(f"extra {', '.join(extra)}")
            raise Refused(
                f"{other.config!r} did not run the same workload as {first.config!r}: "
                + "; ".join(detail)
            )
        for prompt in first.prompts:
            if other.prompt_tokens[prompt] != first.prompt_tokens[prompt]:
                raise Refused(
                    f"{other.config!r} scored {other.prompt_tokens[prompt]} prompt "
                    f"tokens for {prompt!r}; {first.config!r} scored "
                    f"{first.prompt_tokens[prompt]}"
                )


def _escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _cell(value: str | None) -> str:
    return _escape(value) if value is not None else "not recorded"


def _metric(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "n/a"


def render(measurements: list[Measurement]) -> str:
    """The stable Markdown report."""
    first = measurements[0]
    total_tokens = sum(first.prompt_tokens.values())
    lines = [
        "# TinyTitan bench comparison",
        "",
        f"Workload: {len(first.prompts)} prompts, {total_tokens} prompt tokens, "
        f"shared by {len(measurements)} result(s).",
        "",
        "| Config | Model | Cache | MTP | Prompts | Avg decode tok/s | Verified |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for measurement in measurements:
        lines.append(
            f"| {_escape(measurement.config)} "
            f"| {_cell(measurement.provenance.model)} "
            f"| {_escape(measurement.cache_mode)} "
            f"| {_escape(measurement.mtp_config)} "
            f"| {len(measurement.prompts)}/{measurement.configured} "
            f"| {_metric(measurement.avg_decode)} "
            f"| {measurement.verified}/{len(measurement.prompts)} |"
        )
    incomplete = [measurement for measurement in measurements if not measurement.complete]
    if incomplete:
        lines.extend(["", "Incomplete runs:"])
        for measurement in incomplete:
            lines.append(f"- {measurement.config}: {'; '.join(measurement.incomplete)}")
    noted = [measurement for measurement in measurements if measurement.notes]
    if noted:
        lines.extend(["", "Notes:"])
        for measurement in noted:
            lines.append(f"- {measurement.config}: {'; '.join(measurement.notes)}")
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            "| Config | Git commit | Chip/hw.model | macOS | Flags/config | Repeats |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for measurement in measurements:
        provenance = measurement.provenance
        lines.append(
            f"| {_escape(measurement.config)} "
            f"| {_cell(provenance.git_commit)} "
            f"| {_cell(provenance.chip)} "
            f"| {_cell(provenance.macos)} "
            f"| {_escape(provenance.flags)} "
            f"| {_cell(provenance.repeats)} |"
        )
    return "\n".join(lines)


def as_dict(measurements: list[Measurement]) -> dict[str, object]:
    """The stable JSON report."""
    first = measurements[0]
    return {
        "workload": {
            "prompts": list(first.prompts),
            "prompt_tokens": {
                prompt: first.prompt_tokens[prompt] for prompt in sorted(first.prompt_tokens)
            },
            "prompt_tokens_total": sum(first.prompt_tokens.values()),
        },
        "results": [
            {
                "path": measurement.path,
                "config": measurement.config,
                "model": measurement.provenance.model,
                "cache_mode": measurement.cache_mode,
                "mtp_config": measurement.mtp_config,
                "prompts_recorded": len(measurement.prompts),
                "prompts_configured": measurement.configured,
                "avg_decode_tok_s": measurement.avg_decode,
                "verified": measurement.verified,
                "complete": measurement.complete,
                "incomplete": list(measurement.incomplete),
                "notes": list(measurement.notes),
                "provenance": measurement.provenance.as_dict(),
            }
            for measurement in measurements
        ],
        "complete": all(measurement.complete for measurement in measurements),
    }


def refuse(reason: str) -> int:
    print(f"NOT MEASURED: {reason}")
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare saved TinyTitan benchmark result files over a shared workload. "
            "Each input is an aggregate.json or a directory holding one."
        ),
        epilog=(
            "Status: 0 report produced, 1 report produced but a run is incomplete, "
            "2 no comparison exists (reason printed)."
        ),
    )
    parser.add_argument("results", nargs="+", metavar="RESULT")
    parser.add_argument("--json", action="store_true", help="print the JSON report")
    args = parser.parse_args(argv)

    if len(args.results) < 2:
        return refuse("the comparator needs at least two result files")
    try:
        measurements = [_load(path) for path in args.results]
        _require_comparable(measurements)
    except Refused as exc:
        return refuse(str(exc))

    payload = as_dict(measurements)
    print(json.dumps(payload, indent=2) if args.json else render(measurements))
    return 0 if payload["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
