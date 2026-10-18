#!/usr/bin/env python3
"""Quantization quality on bug-fix tasks: 4-bit vs 8-bit, scored by unit tests.

Each task ships a broken function and a symptom. The model is asked to return the
fixed function only. The harness extracts the code block, runs the task's own
unit tests against it in a subprocess, and counts passes — so the score is
behaviour, not a similarity to a reference answer.

The tasks are deliberately small and independent of this repository, because the
question is about the weights, not about the engine: an 8-bit build is compared
with its 4-bit sibling on the same prompts, same sampling and same tests.

Usage (a server is started per model by the caller, or by --cli):

    python3 benchmark/quant_bugfix_eval.py --model models/ornith-1.5_35B_A3B_4Bit \
        --port 8091 --out /tmp/ornith-4bit.json

The result is one JSON object per model: per-task pass/fail, the extracted code,
and the token counts the server reported.
"""

from __future__ import annotations

import argparse
import http.client
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import time

TASKS: list[dict[str, str]] = [
    {
        "name": "average_off_by_one",
        "buggy": (
            "def average(values):\n"
            "    if not values:\n"
            "        return 0\n"
            "    total = 0\n"
            "    for v in values:\n"
            "        total += v\n"
            "    return total / (len(values) - 1)\n"
        ),
        "symptom": "average([1, 2, 3]) returns 3.0 instead of 2.0, and a single-element list raises ZeroDivisionError.",
        "tests": (
            "assert average([1, 2, 3]) == 2.0, average([1, 2, 3])\n"
            "assert average([5]) == 5.0, average([5])\n"
            "assert average([]) == 0, average([])\n"
            "assert average([2, 4]) == 3.0, average([2, 4])\n"
        ),
    },
    {
        "name": "binary_search_high",
        "buggy": (
            "def binary_search(items, target):\n"
            "    low, high = 0, len(items) - 1\n"
            "    while low < high:\n"
            "        mid = (low + high) // 2\n"
            "        if items[mid] == target:\n"
            "            return mid\n"
            "        if items[mid] < target:\n"
            "            low = mid + 1\n"
            "        else:\n"
            "            high = mid - 1\n"
            "    return -1\n"
        ),
        "symptom": "binary_search([1, 2, 3, 4, 5], 5) and the search for 1 both return -1; it finds elements only in the middle.",
        "tests": (
            "xs = [1, 2, 3, 4, 5]\n"
            "assert binary_search(xs, 1) == 0, binary_search(xs, 1)\n"
            "assert binary_search(xs, 5) == 4, binary_search(xs, 5)\n"
            "assert binary_search(xs, 3) == 2, binary_search(xs, 3)\n"
            "assert binary_search(xs, 9) == -1, binary_search(xs, 9)\n"
            "assert binary_search([], 1) == -1\n"
        ),
    },
    {
        "name": "dedupe_keeps_first",
        "buggy": (
            "def dedupe(items):\n"
            "    seen = set()\n"
            "    out = []\n"
            "    for x in items:\n"
            "        if x in seen:\n"
            "            out.append(x)\n"
            "        seen.add(x)\n"
            "    return out\n"
        ),
        "symptom": "dedupe([1, 2, 1, 3, 2]) returns [1, 2] instead of [1, 2, 3]: it returns the duplicates and drops the unique values.",
        "tests": (
            "assert dedupe([1, 2, 1, 3, 2]) == [1, 2, 3], dedupe([1, 2, 1, 3, 2])\n"
            "assert dedupe([]) == []\n"
            "assert dedupe(['a', 'a']) == ['a']\n"
            "assert dedupe([3, 1, 3]) == [3, 1]\n"
        ),
    },
    {
        "name": "parse_duration_units",
        "buggy": (
            "def parse_duration(text):\n"
            "    units = {'s': 1, 'm': 60, 'h': 3600}\n"
            "    number, unit = text[:-1], text[-1]\n"
            "    return int(number) * units.get(unit, 60)\n"
        ),
        "symptom": "parse_duration('5x') silently returns 300 instead of raising ValueError for an unknown unit.",
        "tests": (
            "assert parse_duration('30s') == 30, parse_duration('30s')\n"
            "assert parse_duration('2m') == 120, parse_duration('2m')\n"
            "assert parse_duration('1h') == 3600, parse_duration('1h')\n"
            "try:\n"
            "    parse_duration('5x')\n"
            "except ValueError:\n"
            "    pass\n"
            "else:\n"
            "    raise AssertionError('unknown unit must raise ValueError')\n"
        ),
    },
    {
        "name": "flatten_nested",
        "buggy": (
            "def flatten(items):\n"
            "    out = []\n"
            "    for x in items:\n"
            "        if isinstance(x, list):\n"
            "            out.extend(x)\n"
            "        else:\n"
            "            out.append(x)\n"
            "    return out\n"
        ),
        "symptom": "flatten([1, [2, [3, 4]]]) returns [1, 2, [3, 4]] instead of [1, 2, 3, 4]: nested lists are only one level deep.",
        "tests": (
            "assert flatten([1, [2, [3, 4]]]) == [1, 2, 3, 4], flatten([1, [2, [3, 4]]])\n"
            "assert flatten([]) == []\n"
            "assert flatten([[[1]]]) == [1]\n"
            "assert flatten([1, 2]) == [1, 2]\n"
        ),
    },
    {
        "name": "paginate_last_page",
        "buggy": (
            "def paginate(items, page, per_page):\n"
            "    start = page * per_page\n"
            "    end = start + per_page\n"
            "    return items[start:end]\n"
        ),
        "symptom": "paginate(list(range(10)), 1, 4) returns [4, 5, 6, 7] instead of [0, 1, 2, 3]: every page is shifted forward by one.",
        "tests": (
            "assert paginate(list(range(10)), 1, 4) == [0, 1, 2, 3], paginate(list(range(10)), 1, 4)\n"
            "assert paginate(list(range(10)), 2, 4) == [4, 5, 6, 7]\n"
            "assert paginate(list(range(10)), 3, 4) == [8, 9], paginate(list(range(10)), 3, 4)\n"
            "assert paginate([], 1, 4) == []\n"
            "assert paginate(list(range(10)), 4, 4) == []\n"
        ),
        "fix": (
            "def paginate(items, page, per_page):\n"
            "    start = (page - 1) * per_page\n"
            "    end = start + per_page\n"
            "    return items[start:end]\n"
        ),
    },
    {
        "name": "merge_intervals_adjacent",
        "buggy": (
            "def merge_intervals(intervals):\n"
            "    if not intervals:\n"
            "        return []\n"
            "    intervals = sorted(intervals, key=lambda p: p[0])\n"
            "    out = [list(intervals[0])]\n"
            "    for start, end in intervals[1:]:\n"
            "        if start < out[-1][1]:\n"
            "            out[-1][1] = max(out[-1][1], end)\n"
            "        else:\n"
            "            out.append([start, end])\n"
            "    return [tuple(p) for p in out]\n"
        ),
        "symptom": "merge_intervals([(1, 2), (2, 3)]) returns [(1, 2), (2, 3)] instead of [(1, 3)]: touching intervals are treated as disjoint.",
        "tests": (
            "assert merge_intervals([(1, 2), (2, 3)]) == [(1, 3)], merge_intervals([(1, 2), (2, 3)])\n"
            "assert merge_intervals([(1, 5), (2, 3)]) == [(1, 5)], merge_intervals([(1, 5), (2, 3)])\n"
            "assert merge_intervals([(1, 2), (4, 5)]) == [(1, 2), (4, 5)]\n"
            "assert merge_intervals([]) == []\n"
        ),
        "fix": (
            "def merge_intervals(intervals):\n"
            "    if not intervals:\n"
            "        return []\n"
            "    intervals = sorted(intervals, key=lambda p: p[0])\n"
            "    out = [list(intervals[0])]\n"
            "    for start, end in intervals[1:]:\n"
            "        if start <= out[-1][1]:\n"
            "            out[-1][1] = max(out[-1][1], end)\n"
            "        else:\n"
            "            out.append([start, end])\n"
            "    return [tuple(p) for p in out]\n"
        ),
    },
    {
        "name": "moving_average_window",
        "buggy": (
            "def moving_average(xs, k):\n"
            "    out = []\n"
            "    for i in range(len(xs)):\n"
            "        window = xs[i - k + 1:i + 1]\n"
            "        out.append(sum(window) / k)\n"
            "    return out\n"
        ),
        "symptom": "moving_average([1, 2, 3, 4, 5], 3) returns five values whose first two divide a short window by 3; only complete windows should be returned.",
        "tests": (
            "assert moving_average([1, 2, 3, 4, 5], 3) == [2.0, 3.0, 4.0], moving_average([1, 2, 3, 4, 5], 3)\n"
            "assert moving_average([1, 2], 3) == []\n"
            "assert moving_average([4, 6], 1) == [4.0, 6.0]\n"
            "assert moving_average([], 2) == []\n"
        ),
        "fix": (
            "def moving_average(xs, k):\n"
            "    out = []\n"
            "    for i in range(len(xs) - k + 1):\n"
            "        window = xs[i:i + k]\n"
            "        out.append(sum(window) / k)\n"
            "    return out\n"
        ),
    },
    {
        "name": "copy_config_aliasing",
        "buggy": ("def copy_config(config):\n    return dict(config)\n"),
        "symptom": "After copy = copy_config(cfg), copy['db']['host'] = 'x' also changes cfg['db']['host']: nested values are shared.",
        "tests": (
            "cfg = {'db': {'host': 'a'}, 'tags': ['x']}\n"
            "copy = copy_config(cfg)\n"
            "copy['db']['host'] = 'b'\n"
            "copy['tags'].append('y')\n"
            "assert cfg['db']['host'] == 'a', cfg\n"
            "assert cfg['tags'] == ['x'], cfg\n"
        ),
        "fix": (
            "import copy as _copy\n\ndef copy_config(config):\n    return _copy.deepcopy(config)\n"
        ),
    },
    {
        "name": "backoff_doubles",
        "buggy": (
            "def backoff_delays(attempts, base=1.0, cap=30.0):\n"
            "    return [base * 2 * i for i in range(1, attempts + 1)]\n"
        ),
        "symptom": "backoff_delays(4) returns [2.0, 4.0, 6.0, 8.0]: the delay grows linearly and the cap is never applied. It should double from base and stop at cap.",
        "tests": (
            "assert backoff_delays(4) == [1.0, 2.0, 4.0, 8.0], backoff_delays(4)\n"
            "assert backoff_delays(6) == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0], backoff_delays(6)\n"
            "assert backoff_delays(0) == []\n"
            "assert backoff_delays(2, base=0.5) == [0.5, 1.0]\n"
        ),
        "fix": (
            "def backoff_delays(attempts, base=1.0, cap=30.0):\n"
            "    return [min(base * (2 ** i), cap) for i in range(attempts)]\n"
        ),
    },
    {
        "name": "ipv4_range",
        "buggy": (
            "def valid_ipv4(text):\n"
            "    parts = text.split('.')\n"
            "    if len(parts) != 4:\n"
            "        return False\n"
            "    return all(p.isdigit() for p in parts)\n"
        ),
        "symptom": "valid_ipv4('999.1.1.1') returns True: octets are never range-checked, and leading zeros are accepted.",
        "tests": (
            "assert valid_ipv4('192.168.0.1') is True\n"
            "assert valid_ipv4('255.255.255.255') is True\n"
            "assert valid_ipv4('999.1.1.1') is False\n"
            "assert valid_ipv4('1.2.3') is False\n"
            "assert valid_ipv4('1.2.3.4.5') is False\n"
            "assert valid_ipv4('01.2.3.4') is False\n"
            "assert valid_ipv4('0.0.0.0') is True\n"
        ),
        "fix": (
            "def valid_ipv4(text):\n"
            "    parts = text.split('.')\n"
            "    if len(parts) != 4:\n"
            "        return False\n"
            "    for part in parts:\n"
            "        if not part.isdigit():\n"
            "            return False\n"
            "        if len(part) > 1 and part[0] == '0':\n"
            "            return False\n"
            "        if int(part) > 255:\n"
            "            return False\n"
            "    return True\n"
        ),
    },
    {
        "name": "roman_subtractive",
        "buggy": (
            "def roman_to_int(s):\n"
            "    values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n"
            "    total = 0\n"
            "    for ch in s:\n"
            "        total += values[ch]\n"
            "    return total\n"
        ),
        "symptom": "roman_to_int('IV') returns 6 instead of 4 and roman_to_int('IX') returns 11 instead of 9: subtractive pairs are ignored.",
        "tests": (
            "assert roman_to_int('IV') == 4, roman_to_int('IV')\n"
            "assert roman_to_int('IX') == 9, roman_to_int('IX')\n"
            "assert roman_to_int('XL') == 40, roman_to_int('XL')\n"
            "assert roman_to_int('III') == 3\n"
            "assert roman_to_int('MCMXCIV') == 1994, roman_to_int('MCMXCIV')\n"
        ),
        "fix": (
            "def roman_to_int(s):\n"
            "    values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n"
            "    total = 0\n"
            "    for i, ch in enumerate(s):\n"
            "        value = values[ch]\n"
            "        nxt = values[s[i + 1]] if i + 1 < len(s) else 0\n"
            "        total += -value if value < nxt else value\n"
            "    return total\n"
        ),
    },
]

PROMPT = """Fix the bug in this function.

Symptom: {symptom}

```python
{buggy}```

Return ONLY the corrected function inside a single ```python code block. No
explanation, no tests, no example usage."""


def ask(port: int, prompt: str, max_tokens: int, model_id: str) -> dict:
    body = json.dumps(
        {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    )
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    started = time.time()
    try:
        connection.request(
            "POST", "/v1/chat/completions", body, {"Content-Type": "application/json"}
        )
        response = connection.getresponse()
        payload = json.loads(response.read())
    finally:
        connection.close()
    if "choices" not in payload:
        return {"error": payload, "elapsed_s": time.time() - started}
    choice = payload["choices"][0]
    return {
        "text": choice["message"].get("content") or "",
        "finish": choice.get("finish_reason"),
        "usage": payload.get("usage"),
        "elapsed_s": time.time() - started,
    }


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    if blocks:
        return max(blocks, key=len)
    return text


def run_tests(task: dict[str, str], code: str) -> tuple[bool, str]:
    source = code + "\n\n" + task["tests"]
    with tempfile.TemporaryDirectory(prefix="quant-bugfix-") as directory:
        path = pathlib.Path(directory) / "candidate.py"
        path.write_text(source, encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, str(path)], capture_output=True, text=True, timeout=60, check=False
        )
    detail = (completed.stderr or completed.stdout or "").strip().splitlines()
    return completed.returncode == 0, detail[-1] if detail else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--model-id", default="local")
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", action="append", default=None)

    parser.add_argument("--verify-tasks", action="store_true")
    arguments = parser.parse_args()

    tasks = [t for t in TASKS if arguments.only is None or t["name"] in arguments.only]
    if arguments.verify_tasks:
        failures = 0
        for task in [t for t in tasks if "fix" in t]:
            passed, detail = run_tests(task, task["fix"])
            print(f"{task['name']:<26} {'ok' if passed else 'BROKEN TESTS'}  {detail[:80]}")
            failures += 0 if passed else 1
        return 0 if failures == 0 else 1

    rows = []
    for task in tasks:
        result = ask(
            arguments.port, PROMPT.format(**task), arguments.max_tokens, arguments.model_id
        )
        if "error" in result:
            rows.append({"task": task["name"], "passed": False, "error": result["error"]})
            print(f"{task['name']:<26} ERROR {json.dumps(result['error'])[:120]}")
            continue
        code = extract_code(result["text"])
        passed, detail = run_tests(task, code)
        rows.append(
            {
                "task": task["name"],
                "passed": passed,
                "detail": detail,
                "code": code,
                "finish": result["finish"],
                "usage": result["usage"],
                "elapsed_s": result["elapsed_s"],
            }
        )
        mark = "PASS" if passed else "FAIL"
        print(f"{task['name']:<26} {mark}  {detail[:90]}")

    passed = sum(1 for r in rows if r["passed"])
    summary = {
        "label": arguments.label,
        "passed": passed,
        "total": len(rows),
        "rows": rows,
    }
    pathlib.Path(arguments.out).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\n{arguments.label}: {passed}/{len(rows)} passed -> {arguments.out}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
