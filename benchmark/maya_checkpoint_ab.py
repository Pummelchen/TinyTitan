#!/usr/bin/env python3
"""T24: edited-turn interior-checkpoint A/B on a long conversation.

One server per arm. Each arm sends a full conversation once (which captures
the arm's checkpoints), then sends the same conversation with an edited last
user turn three times. The edited request can only reuse the prefix before
that turn, so it prices exactly the thing the checkpoint tier exists for.

Arms:
  none      TINYTITAN_PROMPT_CHECKPOINTS=0 (full prefill of the edit)
  message   message-boundary checkpoints (system end, last-turn start)
  16k/32k   message boundaries plus a token grid, the placement T20 rejected
            as a default and T24 prices

The reported number is the server's own `prefill_s` for the edited request,
plus wall time and the checkpoint hit line. Prompts carry no nonce: every
measured request is the same edited prompt, which is what a real edit retry
is. The first request is a separate prompt (the unedited turn), so it cannot
warm the cache the measured requests use.

    python3 benchmark/maya_checkpoint_ab.py \
        --model ornith4=models/ornith-1.5_35B_A3B_4Bit \
        --target-tokens 100000 --arms none message 16384 32768 \
        --runs 3 --out .build/t24-checkpoints-ornith4.json
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import http.client
import json
import os
import pathlib
import re
import signal
import statistics
import subprocess
import time
import uuid

ROOT = pathlib.Path(__file__).resolve().parents[1]
HARNESS_LOCK = ROOT / ".build/maya-checkpoint.lock"

PARAGRAPH = (
    "The scheduler keeps a run queue per core, and a thread that becomes "
    "runnable joins the queue of the core it last ran on unless its affinity "
    "says otherwise. The kernel balances queues lazily: it moves work only "
    "when the imbalance exceeds a threshold, because every migration throws "
    "away the cache warmth the thread had accumulated. Priority affects the "
    "order of dequeue, not the length of the quantum, and a thread that "
    "blocks on I/O leaves the queue immediately so the core can pick up the "
    "next item without waiting for the timer. "
)


def running_servers() -> list[str]:
    try:
        found = subprocess.run(
            ["pgrep", "-fl", "TinyTitanServer"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return []
    # `pgrep -f` also matches a shell command that merely mentions the
    # binary (this harness's own `pgrep`, or a wrapper). Only a line whose
    # executable is the server counts.
    return [
        line
        for line in found.stdout.splitlines()
        if line.strip()
        and "TinyTitanServer" in line
        and "bash -c" not in line
        and "pgrep" not in line
        and "python" not in line
    ]


def require_no_server(port: int) -> None:
    existing = running_servers()
    if existing:
        raise RuntimeError(
            "refusing to start: a TinyTitanServer is already running; "
            "running two can crash the machine:\n" + "\n".join(existing)
        )
    probe = http.client.HTTPConnection("127.0.0.1", port, timeout=0.5)
    try:
        probe.request("GET", "/v1/models")
        probe.getresponse().read()
    except OSError:
        return
    finally:
        probe.close()
    raise RuntimeError(f"refusing to start: port {port} already answers")


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        process.terminate()
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()
        process.wait(timeout=30)
    deadline = time.time() + 30
    while time.time() < deadline:
        if not running_servers():
            return
        time.sleep(0.5)
    raise RuntimeError(
        "a TinyTitanServer survived cleanup; refusing to start another:\n"
        + "\n".join(running_servers())
    )


def wait_ready(port: int, process: subprocess.Popen, timeout: float) -> bool:
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
    return False


def served_model_id(port: int) -> str:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.request("GET", "/v1/models")
    payload = json.loads(connection.getresponse().read())
    connection.close()
    return payload["data"][0]["id"]


def request(port: int, model: str, messages: list[dict], max_tokens: int) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        }
    )
    started = time.time()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=14400)
    try:
        connection.request(
            "POST", "/v1/chat/completions", body, {"Content-Type": "application/json"}
        )
        response = connection.getresponse()
        raw = response.read()
    finally:
        connection.close()
    elapsed = time.time() - started
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status}: {raw.decode('utf-8', 'replace')[:2000]}")
    payload = json.loads(raw)
    choices = payload.get("choices") or [{}]
    text = (choices[0].get("message") or {}).get("content", "")
    return {
        "elapsed_s": elapsed,
        "usage": payload.get("usage") or {},
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text_prefix": text[:120],
    }


def build_conversation(target_tokens: int) -> list[dict]:
    """A long alternating history plus one short last user turn.

    Characters per token is about 3.1 for this prose and the ChatML wrapper:
    the 2K smoke run asked for 8,000 characters and the server reported 2,563
    prompt tokens. The actual count is read back and recorded, so the arm's
    real size is evidence rather than an assumption.
    """
    target_chars = int(target_tokens * 3.1)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful engineering assistant. Answer from the "
                "conversation and keep the final answer short. " + PARAGRAPH * 2
            ),
        }
    ]
    written = len(messages[0]["content"])
    turn = 0
    while written < target_chars:
        user = f"Notes for section {turn + 1}.\n" + PARAGRAPH * 6
        assistant = f"Section {turn + 1} summary.\n" + PARAGRAPH * 5
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
        written += len(user) + len(assistant)
        turn += 1
    messages.append(
        {
            "role": "user",
            "content": "Summarise the final section in one sentence.",
        }
    )
    return messages


def edited_conversation(messages: list[dict], variant: int) -> list[dict]:
    edited = [dict(message) for message in messages]
    edited[-1]["content"] = f"Summarise the final section in one sentence, variant {variant}."
    return edited


def log_text(log: pathlib.Path) -> str:
    try:
        return log.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text))


def wait_for_checkpoints(log: pathlib.Path, expected: int, timeout: float) -> int:
    """Wait until the detached save tasks stop adding `stored` lines."""
    deadline = time.time() + timeout
    stable = 0
    previous = -1
    while time.time() < deadline:
        stored = count(r"prompt_checkpoint stored", log_text(log))
        if stored >= expected and stored == previous:
            stable += 1
            if stable >= 3:
                return stored
        else:
            stable = 0
        previous = stored
        time.sleep(2)
    return count(r"prompt_checkpoint stored", log_text(log))


def run_arm(
    label: str,
    model_path: str,
    arm: str,
    target_tokens: int,
    runs: int,
    port: int,
    max_context: int,
) -> dict:
    log = ROOT / f".build/t24-{label}-{arm.replace('/', '-')}.log"
    cache_dir = ROOT / f".build/t24-{label}-{arm.replace('/', '-')}-cache-{uuid.uuid4().hex[:8]}"
    log.unlink(missing_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    environment["TINYTITAN_PROMPT_CHECKPOINTS"] = "0" if arm == "none" else "1"
    if arm in {"16k", "32k"}:
        # Grid-only: the arms must price the placement, not inherit the
        # message boundaries and then restore from last-turn-start anyway.
        environment["TINYTITAN_CHECKPOINT_GRID"] = str(int(arm[:-1]) * 1024)
        environment["TINYTITAN_CHECKPOINT_GRID_ONLY"] = "1"
    else:
        environment.pop("TINYTITAN_CHECKPOINT_GRID", None)
        environment.pop("TINYTITAN_CHECKPOINT_GRID_ONLY", None)
    require_no_server(port)
    with log.open("w") as handle:
        process = subprocess.Popen(
            [
                str(ROOT / ".build/release/TinyTitanServer"),
                "--model",
                model_path,
                "--port",
                str(port),
                "--max-context",
                str(max_context),
                "--thinking",
                "off",
                "--prompt-cache-mode",
                "multi-prefix",
                "--prompt-cache-disk",
                str(cache_dir),
                "--prompt-cache-disk-mib",
                "8192",
            ],
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=ROOT,
            env=environment,
            start_new_session=True,
        )
    try:
        if not wait_ready(port, process, 600):
            return {"arm": arm, "error": "server failed", "log": str(log)}
        model = served_model_id(port)
        messages = build_conversation(target_tokens)
        first = request(port, model, messages, max_tokens=1)
        prompt_tokens = int(first["usage"].get("prompt_tokens") or 0)
        first_prefill = re.findall(r"generation prefill_s=([\d.]+)", log_text(log))
        stored = wait_for_checkpoints(
            log,
            expected=0 if arm == "none" else 1,
            timeout=600 if arm != "none" else 5,
        )
        edited = []
        for variant in range(runs):
            measured = request(
                port,
                model,
                edited_conversation(messages, variant),
                max_tokens=1,
            )
            text = log_text(log)
            rates = re.findall(
                r"generation prefill_s=([\d.]+) decode_s=([\d.]+) decode_tok_s=([\d.]+)",
                text,
            )
            hits = re.findall(
                r"prompt_checkpoint hit kind=([a-z-]+) cached_tokens=(\d+)",
                text,
            )
            measured["server_prefill_s"] = float(rates[-1][0]) if rates else None
            measured["checkpoint_hit"] = (
                {"kind": hits[-1][0], "cached_tokens": int(hits[-1][1])} if hits else None
            )
            edited.append(measured)
        text = log_text(log)
        return {
            "arm": arm,
            "grid_tokens": (int(arm[:-1]) * 1024 if arm in {"16k", "32k"} else None),
            "prompt_tokens": prompt_tokens,
            "capture_request": {
                "wall_s": first["elapsed_s"],
                "server_prefill_s": (float(first_prefill[0]) if first_prefill else None),
            },
            "checkpoint_stored": stored,
            "edited": edited,
            "edited_prefill_median_s": (
                statistics.median(
                    [
                        row["server_prefill_s"]
                        for row in edited
                        if row["server_prefill_s"] is not None
                    ]
                )
                if any(row["server_prefill_s"] is not None for row in edited)
                else None
            ),
            "edited_wall_median_s": statistics.median([row["elapsed_s"] for row in edited]),
            "speedup_vs_capture_prefill": (
                float(first_prefill[0])
                / statistics.median(
                    [
                        row["server_prefill_s"]
                        for row in edited
                        if row["server_prefill_s"] is not None
                    ]
                )
                if first_prefill and any(row["server_prefill_s"] is not None for row in edited)
                else None
            ),
            "log": str(log),
        }
    finally:
        stop_server(process)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="label=path")
    parser.add_argument(
        "--arms", nargs="+", default=["none", "message"], help="none|message|16k|32k"
    )
    parser.add_argument("--target-tokens", type=int, default=100_000)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--port", type=int, default=8095)
    parser.add_argument("--max-context", type=int, default=131_072)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    HARNESS_LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock = HARNESS_LOCK.open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another maya_checkpoint_ab.py is running") from None
    require_no_server(args.port)
    label, path = args.model.split("=", 1)
    payload = {
        "model": label,
        "path": path,
        "target_tokens": args.target_tokens,
        "runs": args.runs,
        "hardware": {
            "physical_memory": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
            "macos": subprocess.run(
                ["sw_vers", "-productVersion"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip(),
        },
        "server": str(ROOT / ".build/release/TinyTitanServer"),
        "results": [],
    }
    for arm in args.arms:
        print(f"arm={arm}", flush=True)
        result = run_arm(
            label, path, arm, args.target_tokens, args.runs, args.port, args.max_context
        )
        payload["results"].append(result)
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    baseline = next((row for row in payload["results"] if row["arm"] == "none"), None)
    for row in payload["results"]:
        if (
            baseline
            and row["arm"] != "none"
            and baseline.get("edited_prefill_median_s")
            and row.get("edited_prefill_median_s")
        ):
            row["speedup_vs_none"] = (
                baseline["edited_prefill_median_s"] / row["edited_prefill_median_s"]
            )
    out = pathlib.Path(args.out)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
    print(json.dumps({"out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
