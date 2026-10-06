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
        except Exception as error:  # recorded, not fatal: one bad case is data
            completion = f"<error {error}>"
        record = dict(job)
        record["completion"] = completion
        record.pop("system", None)
        lines.append(json.dumps(record))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return time.time() - started


def summarize(path: Path) -> dict:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    per_task: dict[str, dict] = {}
    for row in rows:
        expected = tasks.truth_of(row)
        answer = (row.get("completion") or "").strip().upper()
        answer = answer.split()[0].strip(".,:;\"'") if answer else ""
        legal = {"YES", "NO"} if expected in ("YES", "NO") else {"UPDATE", "CONFLICT"}
        entry = per_task.setdefault(row["task"], {"correct": 0, "total": 0, "halves": {}})
        entry["total"] += 1
        half = entry["halves"].setdefault(expected, [0, 0])
        half[1] += 1
        if answer in legal and answer == expected:
            entry["correct"] += 1
            half[0] += 1
    return per_task


def line(label: str, per_task: dict, seconds: float, count: int) -> None:
    print(f"\n== {label}  ({seconds:.0f}s, {seconds / max(1, count):.1f}s/judgement)")
    print(f"{'task':5s} {'n':>3s} {'correct':>8s}  halves")
    ready = 0
    for task in sorted(per_task):
        entry = per_task[task]
        halves = "  ".join(f"{k}: {v[0]}/{v[1]}" for k, v in sorted(entry["halves"].items()))
        worst = min((v[0] / v[1]) for v in entry["halves"].values() if v[1])
        ready += worst >= 0.7
        print(
            f"{task:5s} {entry['total']:3d} "
            f"{100 * entry['correct'] / entry['total']:7.0f}%  {halves}"
            f"{'' if worst >= 0.7 else '   *'}"
        )
    print(f"  tasks good on both halves: {ready}/{len(per_task)}")


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
    jobs = [json.loads(line) for line in args.jobs.read_text().splitlines() if line.strip()]
    out_dir = args.out or args.jobs.parent
    # The jobs file's stem is part of the done name: the same judge over two
    # case files must not overwrite the first result with the second.
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
        line(label, summarize(done), seconds, len(jobs))
        print(f"  done file: {done}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
