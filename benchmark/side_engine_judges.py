#!/usr/bin/env python3.13
"""Who should judge: the resident 4B, or the big model already on the engine?

The side-engine's tasks are one-word decisions (`docs/side-engine-tasks.md`).
The default judge is a dense 4B on the CPU, chosen because it never takes the
main engine and never evicts the person's prompt cache — but the served model
is already loaded, is smarter, and costs no extra RAM. This runs the same
prepared cases through either and reports accuracy, the two answer halves, and
seconds per judgement, so the choice is a measurement rather than a belief.

    python3.13 benchmark/side_engine_tasks.py --prepare /tmp/tasks.jsonl
    python3.13 benchmark/side_engine_judges.py --jobs /tmp/tasks.jsonl \
        --judge cpu:models/qwen3.5_4B_4Bit \
        --judge server:http://127.0.0.1:8096/v1:qwen3.6_35B_A3B_4-Bit

`cpu:<install>` drives the release bench (`cpu35batch`), which is the dense
engine and therefore the 2B/4B/9B only. `server:<url>:<model>` posts the same
system+user prompt to a running TinyTitan server, which is how a 35B MoE or a
125B is reached. The jobs file is the one `side_engine_tasks.py --prepare`
writes; scoring is that script's, imported rather than copied.

A request that never answered is not a judgement that answered wrongly. A
refused case records its reason in the done file's `error` field, is counted
apart, and stays out of every denominator; a blank answer with no reason is
counted the same way, as a judgement that was never written.

`main()` answers a status, and the percentages it prints are only as wide as the
jobs that earned them:

    0  every job wrote a row, every row carried an answer, and the table covers them
    1  measured and contested: a row refused, an answer was never written, the run
       wrote more or fewer rows than there are jobs
    2  NOT MEASURED -- no jobs file, no done file, no rows in it, or a row that
       names no task, prompt or truth; the reason prints and no table does
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / ".build/release/TinyTitanBench"


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tasks = _load("side_engine_tasks", "benchmark/side_engine_tasks.py")


def run_cpu(install: str, jobs: list[dict], out: Path) -> float:
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "jobs.jsonl"
        source.write_text("\n".join(json.dumps(j) for j in jobs) + "\n", encoding="utf-8")
        started = time.time()
        result = subprocess.run(
            [str(BENCH), "cpu35batch", install, str(source), str(out)],
            capture_output=True,
            text=True,
            timeout=14_400,
            check=False,
        )
        elapsed = time.time() - started
    if result.returncode != 0:
        raise SystemExit(f"cpu35batch exited {result.returncode}\n{result.stderr[-2000:]}")
    return elapsed


def run_server(url: str, model: str, jobs: list[dict], out: Path) -> float:
    started = time.time()
    lines = []
    for job in jobs:
        body = json.dumps(
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": job["system"]},
                    {"role": "user", "content": job["prompt"]},
                ],
                "max_completion_tokens": job.get("max", 8),
                "temperature": 0,
            }
        ).encode()
        request = urllib.request.Request(
            f"{url.rstrip('/')}/chat/completions",
            data=body,
            headers={"content-type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as reply:
                payload = json.loads(reply.read())
            completion = payload["choices"][0]["message"]["content"]
            refusal = None
        except Exception as error:  # recorded, not fatal: one bad case is data
            completion, refusal = "", f"{type(error).__name__}: {error}"
        record = dict(job)
        record["completion"] = completion
        if refusal:
            record["error"] = refusal
        record.pop("system", None)
        lines.append(json.dumps(record))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return time.time() - started


def refuse(reason: str) -> int:
    """Say why there is no comparison, print no table, and answer 2."""
    print(f"\nNOT MEASURED: {reason}")
    return 2


def read_rows(path: Path, jobs: int = 0) -> tuple[list[dict], str]:
    """Every row of a done file, or the reason no comparison can be read from it."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        return [], f"{path} cannot be read: {type(error).__name__}: {error}"
    rows: list[dict] = []
    for number, text in enumerate(lines, start=1):
        if not text.strip():
            continue
        try:
            row = json.loads(text)
        except ValueError as error:
            return [], f"line {number} of {path.name} is not json: {text[:60]!r} ({error})"
        if not isinstance(row, dict) or not {"task", "prompt", "truth"} <= set(row):
            return [], (
                f"line {number} of {path.name} names no task, prompt and truth, so it is "
                "not one of the jobs this driver was handed"
            )
        rows.append(row)
    if not rows:
        return [], f"{path} holds no rows, so this judge answered none of the {jobs} jobs"
    return rows, ""


def tally(rows: list[dict]) -> dict:
    per_task: dict[str, dict] = {}
    for row in rows:
        entry = per_task.setdefault(
            row["task"], {"correct": 0, "total": 0, "refused": 0, "silent": 0, "halves": {}}
        )
        if row.get("error"):
            entry["refused"] += 1
            continue
        answer = (row.get("completion") or "").strip().upper()
        if not answer:
            entry["silent"] += 1
            continue
        expected = tasks.truth_of(row)
        answer = answer.split()[0].strip(".,:;\"'")
        legal = {"YES", "NO"} if expected in ("YES", "NO") else {"UPDATE", "CONFLICT"}
        entry["total"] += 1
        half = entry["halves"].setdefault(expected, [0, 0])
        half[1] += 1
        if answer in legal and answer == expected:
            entry["correct"] += 1
            half[0] += 1
    return per_task


def summarize(path: Path) -> dict:
    return tally(read_rows(path)[0])


def line(label: str, per_task: dict, seconds: float, count: int) -> None:
    print(f"\n== {label}  ({seconds:.0f}s, {seconds / max(1, count):.1f}s/judgement)")
    print(f"{'task':5s} {'n':>3s} {'correct':>8s}  halves")
    ready = 0
    for task in sorted(per_task):
        entry = per_task[task]
        halves = "  ".join(f"{k}: {v[0]}/{v[1]}" for k, v in sorted(entry["halves"].items()))
        scored = [v for v in entry["halves"].values() if v[1]]
        worst = min(v[0] / v[1] for v in scored) if scored else None
        good = worst is not None and worst >= 0.7
        ready += good
        score = f"{100 * entry['correct'] / entry['total']:7.0f}%" if scored else "        -"
        flag = "" if good else "   *"
        refused = f"   refused {entry['refused']}" if entry["refused"] else ""
        print(f"{task:5s} {entry['total']:3d} {score}  {halves}{flag}{refused}")
    measured = sum(entry["total"] for entry in per_task.values())
    refused = sum(entry["refused"] for entry in per_task.values())
    print(f"  tasks good on both halves: {ready}/{len(per_task)}")
    if refused:
        print(
            f"  judgements refused: {refused}; every percentage covers only the {measured} that answered"
        )
    if not measured:
        silent = sum(entry.get("silent", 0) for entry in per_task.values())
        if refused:
            print("  measured nothing: every request refused, so none of the above is a score")
        else:
            print(
                f"  measured nothing: {silent} rows carried no answer, "
                "so none of the above is a score"
            )


def parse_judge(spec: str) -> tuple[str, str, str | None]:
    """`cpu:<install>` or `server:<url>:<model>` -> (kind, target, model).

    The URL carries colons (scheme and port) and the model id does not, so the
    last colon separates the two halves of a server spec.
    """
    if spec.startswith("cpu:"):
        return ("cpu", spec[len("cpu:") :], None)
    if spec.startswith("server:"):
        url, model = spec[len("server:") :].rsplit(":", 1)
        return ("server", url, model)
    raise ValueError(f"unknown judge spec: {spec}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument(
        "--judge", action="append", required=True, help="cpu:<install> or server:<url>:<model>"
    )
    ap.add_argument(
        "--out", type=Path, help="directory for the done files (default: beside the jobs file)"
    )
    args = ap.parse_args()
    try:
        source = args.jobs.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        return refuse(f"{args.jobs} cannot be read: {type(error).__name__}: {error}")
    jobs = []
    for number, text in enumerate(source, start=1):
        if not text.strip():
            continue
        try:
            jobs.append(json.loads(text))
        except ValueError as error:
            return refuse(f"line {number} of {args.jobs.name} is not json ({error})")
    if not jobs:
        return refuse(f"{args.jobs} holds no jobs, so there is nothing to judge")
    out_dir = args.out or args.jobs.parent
    # The jobs file's stem is part of the done name: the same judge over two
    # case files must not overwrite the first result with the second.
    contested = False
    for spec in args.judge:
        kind, target, model = parse_judge(spec)
        if kind == "cpu":
            label = spec
            safe = target.replace("/", "-")
        else:
            label = f"server:{model}"
            safe = str(model).replace("/", "-")
        done = out_dir / f"{args.jobs.stem}.{safe}.done.jsonl"
        if kind == "cpu":
            seconds = run_cpu(target, jobs, done)
        else:
            seconds = run_server(target, str(model), jobs, done)
        rows, refusal = read_rows(done, len(jobs))
        if refusal:
            return refuse(refusal)
        per_task = tally(rows)
        line(label, per_task, seconds, len(jobs))
        judged = sum(entry["total"] for entry in per_task.values())
        refused = sum(entry["refused"] for entry in per_task.values())
        silent = sum(entry["silent"] for entry in per_task.values())
        print(f"  judged {judged} of {len(jobs)} jobs, {refused} refused, {silent} with no answer")
        if len(rows) != len(jobs):
            gap = len(jobs) - len(rows)
            detail = (
                f"{gap} of the jobs wrote no row"
                if gap > 0
                else f"{-gap} row repeats a job, so one verdict is counted twice"
            )
            print(f"  the run wrote {len(rows)} rows for {len(jobs)} jobs, so {detail}")
        if judged != len(jobs):
            contested = True
        print(f"  done file: {done}")
    return 1 if contested else 0


if __name__ == "__main__":
    raise SystemExit(main())
