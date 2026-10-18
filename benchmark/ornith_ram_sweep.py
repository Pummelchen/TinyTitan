#!/usr/bin/env python3
"""Decode sweep across RAM budgets for one or more installs, with a text rubric.

For each (model, budget, run) it starts the server with `--ram-budget`, sends one
chat request, reads the engine's own `decode_tok_s` line from the server log, and
stops the server. The request is greedy so the three runs of a cell differ only in
timing — `--ram-budget` is a residency knob and the cached slot count never
changes the produced text (measured byte-identical elsewhere in this tree).

`--score` applies a transparent keyword/structure rubric to each distinct answer
and prints it as a percentage. The rubric is a proxy, not a judgement: it rewards
coverage of the dimensions a Swift-vs-C++ comparison should touch, structure and
length, and penalises repetition.

    python3 benchmark/ornith_ram_sweep.py \
        --model 4bit=models/ornith-1.5_35B_A3B_4Bit \
        --model 8bit=models/ornith-1.5_35B_A3B_8Bit \
        --budgets 8G 10G 12G --runs 3 --port 8092 \
        --out .build/ornith-ram-sweep.json
"""

from __future__ import annotations

import argparse
import http.client
import json
import pathlib
import re
import subprocess
import time

PROMPT = "difference swift vs c++ in detail"
ROOT = pathlib.Path(__file__).resolve().parents[1]

# Each dimension a Swift/C++ comparison is expected to cover, with the spellings
# that count as covering it.
CRITERIA: dict[str, tuple[str, ...]] = {
    "memory": ("memory management", "arc", "manual memory", "raii", "ownership", "garbage"),
    "performance": ("performance", "faster", "overhead", "zero-cost", "speed"),
    "types": ("type system", "static typing", "strongly typed", "type inference"),
    "generics": ("generic", "template", "protocol", "concept"),
    "concurrency": ("concurrency", "async", "await", "thread", "actor", "gcd", "dispatch"),
    "tooling": ("xcode", "swift package", "swiftpm", "cmake", "build system", "toolchain"),
    "errors": ("error handling", "exception", "optional", "throws", "result type", "try/catch"),
    "syntax": ("syntax", "verbose", "readab", "semicolon", "concise"),
    "safety": ("memory safe", "safety", "undefined behavior", "bounds", "null safety"),
    "ecosystem": ("ios", "macos", "apple", "linux", "embedded", "game", "systems programming"),
    "interop": ("interop", "c interop", "bridging", "objective-c", 'extern "c"'),
    "use_cases": ("use case", "when to use", "choose", "appropriate", "best for"),
}


def score(text: str) -> tuple[float, dict[str, int]]:
    lower = text.lower()
    parts: dict[str, int] = {}
    for name, needles in CRITERIA.items():
        parts[name] = 8 if any(n in lower for n in needles) else 0
    words = len(re.findall(r"\w+", text))
    parts["length_300"] = 5 if words >= 300 else 0
    parts["length_600"] = 5 if words >= 600 else 0
    headings = len(re.findall(r"^#{1,4} ", text, re.M))
    parts["structure"] = 10 if headings >= 4 else (5 if headings >= 2 else 0)
    fences = len(re.findall(r"```", text)) // 2
    parts["examples"] = 10 if fences >= 2 else (5 if fences >= 1 else 0)
    # Only substantial lines can be repetition: "}" and "```" repeat in any
    # code sample and penalising them punishes code blocks, not rambling.
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) >= 24]
    repeated = sum(1 for line in set(lines) if lines.count(line) >= 3)
    parts["repetition_penalty"] = -10 if repeated else 0
    raw = sum(parts.values())
    return raw / 116 * 100, parts


def wait_ready(port: int, process: subprocess.Popen, log: pathlib.Path, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            return False
        try:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            connection.request("GET", "/v1/models")
            response = connection.getresponse()
            response.read()
            connection.close()
            if response.status == 200:
                return True
        except OSError:
            pass
        time.sleep(0.5)
    return log.exists()


def decode_rate(log: pathlib.Path) -> dict[str, float] | None:
    text = log.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"generation prefill_s=([\d.]+) decode_s=([\d.]+) decode_tok_s=([\d.]+)", text
    )
    if not matches:
        return None
    prefill, decode, rate = matches[-1]
    return {"prefill_s": float(prefill), "decode_s": float(decode), "tok_s": float(rate)}


def served_model_id(port: int) -> str:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.request("GET", "/v1/models")
    payload = json.loads(connection.getresponse().read())
    connection.close()
    return payload["data"][0]["id"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="label=path")
    parser.add_argument("--budgets", nargs="+", default=["8G", "10G", "12G"])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()

    models = []
    for spec in arguments.model:
        label, path = spec.split("=", 1)
        models.append((label, path))

    results = []
    for label, path in models:
        for budget in arguments.budgets:
            for run in range(1, arguments.runs + 1):
                log = ROOT / f".build/sweep-{label}-{budget}-{run}.log"
                log.unlink(missing_ok=True)
                with log.open("w") as handle:
                    process = subprocess.Popen(
                        [
                            str(ROOT / ".build/release/TinyTitanServer"),
                            "--model",
                            path,
                            "--ram-budget",
                            budget,
                            "--thinking",
                            "off",
                            "--port",
                            str(arguments.port),
                        ],
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        cwd=ROOT,
                    )
                try:
                    if not wait_ready(arguments.port, process, log, 180):
                        results.append(
                            {"label": label, "budget": budget, "run": run, "error": "server failed"}
                        )
                        print(f"{label:<5} {budget:<4} r{run}  SERVER FAILED")
                        continue
                    model_id = served_model_id(arguments.port)
                    body = json.dumps(
                        {
                            "model": model_id,
                            "messages": [{"role": "user", "content": arguments.prompt}],
                            "max_tokens": arguments.max_tokens,
                            "temperature": 0,
                        }
                    )
                    connection = http.client.HTTPConnection(
                        "127.0.0.1", arguments.port, timeout=900
                    )
                    connection.request(
                        "POST", "/v1/chat/completions", body, {"Content-Type": "application/json"}
                    )
                    payload = json.loads(connection.getresponse().read())
                    connection.close()
                    if "choices" not in payload:
                        results.append(
                            {"label": label, "budget": budget, "run": run, "error": payload}
                        )
                        print(f"{label:<5} {budget:<4} r{run}  ERROR")
                        continue
                    text = payload["choices"][0]["message"].get("content") or ""
                    usage = payload.get("usage") or {}
                    rate = decode_rate(log) or {}
                    percent, parts = score(text)
                    results.append(
                        {
                            "label": label,
                            "budget": budget,
                            "run": run,
                            "text": text,
                            "usage": usage,
                            "decode": rate,
                            "score": percent,
                            "score_parts": parts,
                        }
                    )
                    print(
                        f"{label:<5} {budget:<4} r{run}  "
                        f"{rate.get('tok_s', 0):6.2f} tok/s  "
                        f"{usage.get('completion_tokens', 0):3d} tok  score {percent:5.1f}%"
                    )
                finally:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    time.sleep(1)

    pathlib.Path(arguments.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\n{len(results)} rows -> {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
