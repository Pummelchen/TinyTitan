#!/usr/bin/env python3
"""Reproducible barrier vs hit/fixup decode A/B for the production profile.

Each mode gets a fresh server. Every fixed prompt is sent twice with the same
seed: the first request observes colder routed-expert slots and the second a
warmer working set. The server's TINYTITAN_RUNNER_STATS footer supplies cache and
I/O measurements; response text must match across modes before results pass, and
both arms must have answered and logged a figure the other can be compared
against -- AUD-280 measured that text equality alone was the whole verdict, so
two empty answers, `decode_tok_s=nan`, a run that claimed 128 completion tokens
at 0.00 tok/s, an answer beside a footer that read 0.00 tok/s, and a response
that logged no token count at all all exited 0.

    0  every case compared, both arms answered, and every published figure came
       from a footer that carried it
    1  it measured, and the page is contested by a named disagreement or by cases
       it could not compare
    2  the A/B did not run: the driver's own refusal, no case answered, or no
       case published a usable figure, with the reason printed and no artifact
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import pathlib
import platform
import re
import subprocess
import sys
import time

from tinytitan_profile import (
    DEFAULT_MODEL_PATH,
    benchmark_log_path,
    logged,
    resolve_api_model,
    server_command,
    server_environment,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]
SERVER = ROOT / ".build/release/TinyTitanServer"
PORT = 8112
PROCESS_PATTERN = (
    "TinyTitanServer|TinyTitanCLI|TinyTitanPackageTests|"
    "swiftpm-testing-helper|mlx_lm|mlx-lm|llama-server|unsloth-studio"
)
PROMPTS = {
    "short": "Explain why a mutex protects shared state in one short paragraph.",
    "medium": "Explain how an SSD-backed sparse mixture-of-experts cache can bound memory "
    "while preserving inference correctness. Include hits, misses, and eviction.",
    "long": (ROOT / "AGENTS.md").read_text(),
}
MODES = ("barrier", "hit-fixup")
WARMTH = ("cold", "warm")


def command_output(command: list[str]) -> str:
    return subprocess.run(command, text=True, capture_output=True, check=True).stdout.strip()


def preflight() -> dict[str, object]:
    if not SERVER.is_file():
        raise RuntimeError(f"release server missing: {SERVER}; run swift build -c release")
    for required in ("manifest.json", "verified-install.json"):
        if not (DEFAULT_MODEL_PATH / required).is_file():
            raise RuntimeError(f"incomplete model installation: missing {required}")
    process_listing = command_output(["ps", "-axo", "pid=,command="])
    process_pattern = re.compile(PROCESS_PATTERN)
    processes = [
        line.strip() for line in process_listing.splitlines() if process_pattern.search(line)
    ]
    if processes:
        raise RuntimeError(
            "model process already running; refusing to benchmark:\n" + "\n".join(processes)
        )
    pressure = command_output(["memory_pressure", "-Q"])
    free = re.search(r"free percentage:\s*(\d+)%", pressure)
    if not free or int(free.group(1)) < 10:
        raise RuntimeError("memory pressure is not acceptable: " + pressure)
    return {
        "commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short"]),
        "hardware": command_output(["system_profiler", "SPHardwareDataType"]),
        "macos": platform.mac_ver()[0],
        "swift": command_output(["swift", "--version"]),
        "model": str(DEFAULT_MODEL_PATH),
    }


def wait_until_healthy(process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited early with {process.returncode}")
        try:
            connection = http.client.HTTPConnection("127.0.0.1", PORT, timeout=1)
            connection.request("GET", "/health")
            healthy = connection.getresponse().read().decode()
            connection.close()
            if "ok" in healthy:
                return
        except OSError:
            pass
        time.sleep(0.05)
    raise RuntimeError("server health check timed out")


def memory_snapshot(process_id: int) -> dict[str, object]:
    rss = command_output(["ps", "-o", "rss=", "-p", str(process_id)])
    pressure = command_output(["memory_pressure", "-Q"])
    free = re.search(r"free percentage:\s*(\d+)%", pressure)
    return {
        "server_rss_kib": int(rss),
        "machine_free_percent": int(free.group(1)) if free else None,
    }


def request(prompt: str) -> dict[str, object]:
    payload = json.dumps(
        {
            "model": resolve_api_model(PORT),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 0.0,
            "seed": 41,
            "max_completion_tokens": 128,
            "stream": False,
        }
    ).encode()
    started = time.monotonic()
    connection = http.client.HTTPConnection("127.0.0.1", PORT, timeout=1800)
    connection.request(
        "POST", "/v1/chat/completions", body=payload, headers={"Content-Type": "application/json"}
    )
    response = connection.getresponse()
    body = response.read()
    connection.close()
    if response.status != 200:
        raise RuntimeError(f"request failed HTTP {response.status}: {body.decode()}")
    decoded = json.loads(body)
    return {
        "wall_seconds": time.monotonic() - started,
        "content": decoded["choices"][0]["message"]["content"],
        "usage": decoded.get("usage", {}),
    }


def parse_footers(log_path: pathlib.Path) -> tuple[list[str], list[str]]:
    generation: list[str] = []
    runner: list[str] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if "TinyTitan generation" in line and "decode_tok_s=" in line:
            generation.append(line.strip())
        if "TinyTitan runner" in line and "expert_hit_rate=" in line:
            runner.append(line.strip())
    return generation, runner


def run_case(
    mode: str,
    prompt_name: str,
    prompt: str,
    io_backend: str = "pread",
    io_sync: str = "host",
    io_submission: str = "deferred",
) -> tuple[list[dict[str, object]], pathlib.Path]:
    environment = server_environment()
    environment["TINYTITAN_DECODE_EXPERT_EXECUTION"] = mode
    environment["TINYTITAN_EXPERT_IO_BACKEND"] = io_backend
    environment["TINYTITAN_EXPERT_IO_SYNC"] = io_sync
    environment["TINYTITAN_EXPERT_IO_SUBMISSION"] = io_submission
    environment["TINYTITAN_RUNNER_STATS"] = "1"
    environment["TINYTITAN_KERNEL_STATS"] = "1"
    log_path = pathlib.Path(
        benchmark_log_path(
            f"expert-ab-{mode}-{io_backend}-{io_sync}-{io_submission}-{prompt_name}.log"
        )
    )
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            server_command(SERVER, PORT),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            wait_until_healthy(process)
            baseline_memory = memory_snapshot(process.pid)
            results = []
            for warmth in WARMTH:
                print(f"{mode:9s} {prompt_name:6s} {warmth}", flush=True)
                results.append(
                    {
                        "mode": mode,
                        "io_backend": io_backend,
                        "io_sync": io_sync,
                        "io_submission": io_submission,
                        "prompt": prompt_name,
                        "warmth": warmth,
                        "baseline_memory": baseline_memory,
                        **request(prompt),
                        "memory_after": memory_snapshot(process.pid),
                    }
                )
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    generation, runner = parse_footers(log_path)
    if len(generation) != len(results) or len(runner) != len(results):
        raise RuntimeError(
            f"missing benchmark footers in {log_path}: "
            f"generation={len(generation)} runner={len(runner)} expected={len(results)}"
        )
    for index, result in enumerate(results):
        result["generation_footer"] = generation[index]
        result["runner_footer"] = runner[index]
    return results, log_path


def footer_raw(line: str, key: str) -> str | None:
    """The token one footer printed for one key, exactly as printed, or None."""
    match = re.search(rf"(?:^|\s){re.escape(key)}=(\S+)", line)
    return match.group(1) if match else None


def footer_value(line: str, key: str) -> float | None:
    """One figure from a server footer, or None when the footer cannot answer.

    AUD-280: the A/B required a footer by line count and read none of its values,
    so a log that printed `decode_tok_s=nan` exited 0 beside one that printed a
    rate. A footer figure is a measurement only when it parses to a finite number;
    the raw token is what the refusal quotes, which is why the two live apart.
    """
    raw = footer_raw(line, key)
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _claimed_tokens(row: dict) -> int | None:
    usage = row.get("usage") or {}
    return usage.get("completion_tokens")


def _case_state(case: dict) -> str:
    """Classify one case and record the cells, detail lines and names it publishes."""
    case["cells"] = {}
    case["details"] = []
    case["answered"] = {}
    case["tokens"] = {}
    case["empty"] = []
    case["no_claim"] = False
    prompt, warmth = case["prompt"], case["warmth"]
    arms = case["arms"]
    arm_names = case["arm_names"]

    unusable = False
    unlogged = False
    for mode in arm_names:
        row = arms[mode]
        generation = str(row.get("generation_footer") or "")
        runner = str(row.get("runner_footer") or "")
        decode = footer_value(generation, "decode_tok_s")
        hit = footer_value(runner, "expert_hit_rate")
        claimed = _claimed_tokens(row)
        case["answered"][mode] = bool(str(row.get("content") or "").strip())
        case["tokens"][mode] = claimed
        case["cells"][mode] = (
            logged(decode, ".2f", " tok/s", "-- tok/s"),
            logged(None if hit is None else 100 * hit, ".1f", "% hit", "-- hit"),
            logged(claimed, "d", " claimed", "-- claimed"),
        )
        for label, value, footer in (
            ("decode_tok_s", decode, generation),
            ("expert_hit_rate", hit, runner),
        ):
            raw = footer_raw(footer, label)
            if raw is None:
                unlogged = True
                case["details"].append(f"{mode} {prompt}/{warmth} logged no {label}")
            elif value is None:
                unusable = True
                case["details"].append(f"{mode} {prompt}/{warmth} logged {label}={raw}")
        if claimed is None:
            unlogged = True
            case["details"].append(f"{mode} {prompt}/{warmth} claimed no completion token count")
            case["no_claim"] = True
        if decode == 0.0 and (claimed or case["answered"][mode]):
            unusable = True
            reason = (
                f"while it reported {claimed} completion tokens"
                if claimed
                else "while the response carried text"
            )
            rate = logged(decode, ".2f", " tok/s", "--")
            case["details"].append(f"{mode} {prompt}/{warmth} measured {rate} {reason}")
    if unusable:
        return "unusable"
    if unlogged:
        return "unlogged"

    empty = [mode for mode in arm_names if not case["answered"][mode]]
    case["empty"] = empty
    if len(empty) == len(arm_names):
        return "empty-case"
    if empty:
        return "empty-arm"
    if arms[arm_names[0]]["content"] != arms[arm_names[1]]["content"]:
        case["details"].append(f"{prompt}/{warmth} text differs")
        return "differ"
    left, right = case["tokens"][arm_names[0]], case["tokens"][arm_names[1]]
    if left is not None and right is not None and left != right:
        case["details"].append(f"{prompt}/{warmth} completion length differs: {left} vs {right}")
        return "differ"
    return "compared"


def _cases(rows: list[dict], prompts, arm_key="mode", arms=MODES) -> list[dict]:
    """One record per (prompt, warmth) case, in page order. Pure.

    `arm_key` names the record field that carries the arm and `arms` the two names
    it takes, so a sibling A/B over a different pair -- the I/O backends, say --
    runs the same classifier instead of writing its own. AUD-285.
    """
    by_case = {(row["prompt"], row["warmth"], row[arm_key]): row for row in rows}
    cases = []
    for prompt in prompts:
        for warmth in WARMTH:
            case = {
                "prompt": prompt,
                "warmth": warmth,
                "arm_names": list(arms),
                "arms": {mode: by_case[(prompt, warmth, mode)] for mode in arms},
            }
            case["state"] = _case_state(case)
            cases.append(case)
    return cases


def verdict(rows: list[dict], prompts, arm_key="mode", arms=MODES) -> tuple[list[str], int]:
    """The page and the status.

    AUD-280: text equality between the two arms used to be the whole verdict, so a
    sweep where every response was empty was an agreement about nothing, arms that
    agreed on the words but not on the tokens they took were a pass, and the
    footers this A/B exists to read were required by count and parsed by nothing.
    """
    if not rows or not prompts:
        return ["\nNOT MEASURED: the sweep produced no case rows to compare"], 2
    cases = _cases(rows, prompts, arm_key=arm_key, arms=arms)
    lines = []
    for case in cases:
        lines.append(
            f"{case['prompt']} {case['warmth']:5s} "
            + "   ".join(
                f"{mode:9s} " + " ".join(case["cells"][mode]) for mode in case["arm_names"]
            )
            + f"   {case['state']}"
        )
        lines.extend(case["details"])
    total = len(cases)
    compared = sum(1 for case in cases if case["state"] == "compared")
    unusable = [case for case in cases if case["state"] == "unusable"]
    empty_cases = [case for case in cases if case["state"] == "empty-case"]

    if len(unusable) == total:
        lines.append(
            "\nNOT MEASURED: no case published a usable figure; the lines above quote each one"
        )
        return lines, 2
    if len(empty_cases) == total:
        lines.append(
            "\nNOT MEASURED: every case answered nothing, so the A/B compared two "
            "empty answers instead of text"
        )
        return lines, 2

    lines.append(f"\ncases compared {compared} of {total}")
    notes = []
    if unusable:
        notes.append("a published figure could not be read as a number")
    if any(case["state"] == "unlogged" for case in cases):
        notes.append("a cell above prints -- because no footer logged it")
    if any(case["no_claim"] for case in cases):
        notes.append(
            "the arms' completion length was never compared: a response claimed no token count"
        )
    if empty_cases:
        names = ", ".join(f"{case['prompt']}/{case['warmth']}" for case in empty_cases)
        notes.append(f"{names} answered nothing, so the arms there are not comparable")
    arm_names = []
    for case in cases:
        if case["state"] != "empty-arm":
            continue
        for mode in case["empty"]:
            name = f"{mode} {case['prompt']}"
            if name not in arm_names:
                arm_names.append(name)
    if arm_names:
        notes.append(
            f"{', '.join(arm_names)} answered nothing, so the arms there are not comparable"
        )
    if any(case["state"] == "differ" for case in cases):
        notes.append("the arms disagree as named above")
    for note in notes:
        lines.append(f"CONTESTED: {note}")
    return lines, 0 if compared == total else 1


def refuse(reason: str, ran: list[dict]) -> int:
    """Say what stopped the A/B, name what did run, and write no artifact.

    A refused preflight, a dead server and a log with no footers each used to
    escape as a traceback at exit 1 -- the status a measured disagreement carries --
    and left no record of the cases that had already run.
    """
    print(f"\nNOT MEASURED: {reason}")
    if ran:
        names = ", ".join(sorted({f"{r['mode']} {r['prompt']}/{r['warmth']}" for r in ran}))
        print(f"cases that ran before the refusal: {names}")
    else:
        print("no case ran before the refusal")
    return 2


def main() -> int:
    argparse.ArgumentParser(
        description="Compare barrier and hit/fixup decode with fixed production settings"
    ).parse_args()
    try:
        metadata = preflight()
    except (RuntimeError, subprocess.SubprocessError, OSError) as error:
        return refuse(str(error), [])

    all_results: list[dict[str, object]] = []
    logs: dict[str, str] = {}
    try:
        for mode in MODES:
            for prompt_name, prompt in PROMPTS.items():
                results, log = run_case(mode, prompt_name, prompt)
                all_results.extend(results)
                logs[f"{mode}/{prompt_name}"] = str(log)
    except (RuntimeError, subprocess.SubprocessError, OSError, ValueError, KeyError) as error:
        return refuse(str(error), all_results)

    lines, status = verdict(all_results, tuple(PROMPTS))
    for line in lines:
        print(line)
    if status == 2:
        return 2

    cases = _cases(all_results, tuple(PROMPTS))
    mismatches = [
        f"{case['prompt']}/{case['warmth']}" for case in cases if case["state"] == "differ"
    ]
    output = ROOT / ".build/benchmark-results/hit-fixup-ab.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "configuration": {
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "top_k": 20,
                    "presence_penalty": 0.0,
                    "seed": 41,
                    "max_completion_tokens": 128,
                },
                "logs": logs,
                "results": all_results,
                "response_mismatches": mismatches,
                "uncompared_cases": [
                    f"{case['prompt']}/{case['warmth']} {case['state']}"
                    for case in cases
                    if case["state"] != "compared"
                ],
                "status": status,
                "cases_compared": sum(1 for case in cases if case["state"] == "compared"),
                "cases_total": len(cases),
                "passed": status == 0,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"results: {output}")
    if status == 1:
        print("ERROR: the A/B is contested; the lines above name why", file=sys.stderr)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
