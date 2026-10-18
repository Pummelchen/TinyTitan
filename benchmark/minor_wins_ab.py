#!/usr/bin/env python3
"""Paired, interleaved A/B for the small decode knobs that were never settled.

The history in `docs/minor-wins-accounting.md` shows why these were left alone:
each measured +0.4 to +2.2%, and every one of those effects is smaller than this
host's round-to-round swing. Testing them one at a time against that floor is a
design that rejects true wins. This driver therefore does two things the earlier
sweeps did not:

* **interleaves and pairs** every arm with the base arm inside each round, so a
  slow hour cannot be mistaken for a slow arm;
* supports a **stack** arm, because five 1% effects are a 5% effect and that *is*
  measurable where each alone is not.

Each run is a fresh server (the knobs are read at start), one greedy request, and
the engine's own `decode_tok_s` line. Text is compared across arms: a knob that
changes the output is not a performance win.

    python3 benchmark/minor_wins_ab.py \
        --model ornith4=models/ornith-1.5_35B_A3B_4Bit \
        --rounds 3 --max-tokens 128 --out .build/minor-wins-ornith4.json
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import pathlib
import re
import statistics
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROMPT = "Write a detailed technical explanation of how a compiler turns source code into a binary."

# Every arm is environment-only: these are the switches that still exist in the
# tree. `EARLY_HITS` and `PARALLEL_IO` were deleted from the engine, and the
# eviction policy has no environment reader any more (default `.lfu`), so those
# three are not testable here without restoring code.
ARMS: list[tuple[str, dict[str, str]]] = [
    ("base", {}),
    ("pagecache", {"TINYTITAN_BOUNDED_IO": "0"}),
    ("advice-off", {"TINYTITAN_RDADVISE_POLICY": "off"}),
    ("immediate", {"TINYTITAN_EXPERT_IO_SUBMISSION": "immediate"}),
    # Advice stays on here on purpose: Lever 4 measured that advice earns its
    # keep exactly where the reads can consume it, so `advice-off` is a
    # substitute for the page cache rather than an addition to it.
    (
        "stack",
        {"TINYTITAN_BOUNDED_IO": "0", "TINYTITAN_EXPERT_IO_SUBMISSION": "immediate"},
    ),
]

COUNTERS = (
    "expert_hit_rate",
    "expert_read_mib",
    "wait_ms",
    "io_ms",
    "io_hidden_pct",
    "cache_plan_ms",
    "gather_ms",
)


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
    return False


def served_model_id(port: int) -> str:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.request("GET", "/v1/models")
    payload = json.loads(connection.getresponse().read())
    connection.close()
    return payload["data"][0]["id"]


def read_metrics(log: pathlib.Path) -> dict[str, float]:
    text = log.read_text(encoding="utf-8", errors="replace")
    out: dict[str, float] = {}
    match = re.findall(
        r"generation prefill_s=([\d.]+) decode_s=([\d.]+) decode_tok_s=([\d.]+)", text
    )
    if match:
        prefill, decode, rate = match[-1]
        out["prefill_s"] = float(prefill)
        out["decode_s"] = float(decode)
        out["tok_s"] = float(rate)
    runner = text.rfind("TinyTitan runner ")
    if runner >= 0:
        line = text[runner : text.find("\n", runner)]
        for name in COUNTERS:
            found = re.search(rf"\b{name}=([\d.]+)", line)
            if found:
                out[name] = float(found.group(1))
    return out


def run_arm(
    label: str,
    prompt_text: str,
    path: str,
    env_extra: dict[str, str],
    port: int,
    max_tokens: int,
    tag: str,
    ram_budget: str | None = None,
    extra_args: list[str] | None = None,
):
    log = ROOT / f".build/minor-wins-{tag}.log"
    log.unlink(missing_ok=True)
    env = dict(os.environ)
    env.update(env_extra)
    env["TINYTITAN_RUNNER_STATS"] = "1"
    with log.open("w") as handle:
        process = subprocess.Popen(
            [
                str(ROOT / ".build/release/TinyTitanServer"),
                "--model",
                path,
                "--thinking",
                "off",
                "--port",
                str(port),
            ]
            + (["--ram-budget", ram_budget] if ram_budget else [])
            + (extra_args or []),
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=ROOT,
            env=env,
        )
    try:
        if not wait_ready(port, process, log, 300):
            return {"error": "server failed"}
        body = json.dumps(
            {
                "model": served_model_id(port),
                "messages": [{"role": "user", "content": prompt_text}],
                "max_tokens": max_tokens,
                "temperature": 0,
            }
        )
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1800)
        connection.request(
            "POST", "/v1/chat/completions", body, {"Content-Type": "application/json"}
        )
        payload = json.loads(connection.getresponse().read())
        connection.close()
        if "choices" not in payload:
            return {"error": payload}
        text = payload["choices"][0]["message"].get("content") or ""
        metrics = read_metrics(log)
        # sha256, not `hash()`: the builtin is salted per process, so it would
        # differ between arms even for identical text and the check would never
        # fire.
        metrics["text_sha"] = hashlib.sha256(text.encode()).hexdigest()[:16]
        metrics["chars"] = len(text)
        return metrics
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
        time.sleep(2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="label=path")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--port", type=int, default=8093)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="use this text as the prompt instead of the built-in one; with "
        "--repeat it reaches a long-prefill workload, which is the condition "
        "the pressure-dependent knobs pay in",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--server-arg",
        action="append",
        default=[],
        help="extra server argument, repeatable and passed verbatim (e.g. "
        "--server-arg=--kv-bits --server-arg=16)",
    )
    parser.add_argument(
        "--arm",
        action="append",
        default=None,
        metavar="NAME=K=V[,K=V]",
        help="an arm to test, repeatable; replaces the built-in list (base is "
        "always included as the reference). Example: --arm pc=TINYTITAN_BOUNDED_IO=0",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="restrict to these arms (base is always kept, it is the reference)",
    )
    parser.add_argument(
        "--ram-budget",
        default=None,
        help="forwarded to the server; a tight budget raises expert-miss pressure, "
        "which is the condition the knobs in the history actually paid in",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="skip the discarded first run; it exists because the first server in "
        "a session pays for a cold page cache over the weights, and that is "
        "larger than every effect measured here",
    )
    arguments = parser.parse_args()

    label, path = arguments.model.split("=", 1)
    prompt_text = PROMPT
    if arguments.prompt_file:
        prompt_text = pathlib.Path(arguments.prompt_file).read_text(encoding="utf-8")
    prompt_text = (prompt_text + "\n\n") * max(1, arguments.repeat)
    arms = ARMS
    if arguments.arm:
        custom = [("base", {}, [])]
        for spec in arguments.arm:
            name, _, pairs = spec.partition("=")
            env: dict[str, str] = {}
            sargs: list[str] = []
            for pair in filter(None, pairs.split(",")):
                key, _, value = pair.partition("=")
                if key.startswith("@"):
                    # A server flag rather than an environment variable, so a
                    # testable arm can be `@kv-bits=16` as well as an env pair.
                    sargs += [key[1:], value]
                else:
                    env[key] = value
            custom.append((name, env, sargs))
        arms = custom
    elif arguments.only:
        wanted = set(arguments.only) | {"base"}
        arms = [(n, e, []) for n, e in ARMS if n in wanted]
    else:
        arms = [(n, e, []) for n, e in ARMS]
    rows: list[dict] = []
    if not arguments.no_warmup:
        print("warm-up (discarded) ...", flush=True)
        run_arm(
            "base",
            prompt_text,
            path,
            {},
            arguments.port,
            arguments.max_tokens,
            f"{label}-warmup",
            arguments.ram_budget,
            arguments.server_arg,
        )
    for round_index in range(1, arguments.rounds + 1):
        # Rotate so that "first arm of the round" is not always the base: the
        # machine warms through a round as much as between rounds.
        order = arms[round_index % len(arms) :] + arms[: round_index % len(arms)]
        for position, (arm, env_extra, sargs) in enumerate(order):
            tag = f"{label}-r{round_index}-{arm}"
            metrics = run_arm(
                arm,
                prompt_text,
                path,
                env_extra,
                arguments.port,
                arguments.max_tokens,
                tag,
                arguments.ram_budget,
                arguments.server_arg + sargs,
            )
            row = {
                "round": round_index,
                "arm": arm,
                "position": position,
                **env_extra,
                **metrics,
            }
            rows.append(row)
            if "error" in metrics:
                print(f"r{round_index} {arm:<11} ERROR {metrics['error']}")
            else:
                print(
                    f"r{round_index} {arm:<11} {metrics.get('tok_s', 0):6.3f} tok/s  "
                    f"hit {metrics.get('expert_hit_rate', 0):.4f}  "
                    f"reads {metrics.get('expert_read_mib', 0):7.0f} MiB  "
                    f"wait {metrics.get('wait_ms', 0):6.1f}  chars {metrics.get('chars', 0)}"
                )

    print(f"\n== paired deltas vs base, per round ({label}) ==")
    for arm, _, _ in arms:
        if arm == "base":
            continue
        deltas, signs, counter_notes = [], 0, set()
        for round_index in range(1, arguments.rounds + 1):
            base = next((r for r in rows if r["round"] == round_index and r["arm"] == "base"), None)
            arm_row = next((r for r in rows if r["round"] == round_index and r["arm"] == arm), None)
            if not base or not arm_row or "tok_s" not in base or "tok_s" not in arm_row:
                continue
            delta = (arm_row["tok_s"] - base["tok_s"]) / base["tok_s"] * 100
            deltas.append(delta)
            signs += delta > 0
            if base.get("text_sha") != arm_row.get("text_sha"):
                counter_notes.add("OUTPUT DIFFERS")
            if arm_row.get("expert_read_mib") != base.get("expert_read_mib"):
                counter_notes.add("reads changed")
        if deltas:
            print(
                f"{arm:<11} mean {statistics.mean(deltas):+6.2f}%  "
                f"median {statistics.median(deltas):+6.2f}%  "
                f"{signs}/{len(deltas)} positive  "
                f"{' '.join(sorted(counter_notes)) or 'counters unchanged'}"
            )

    pathlib.Path(arguments.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\n{len(rows)} rows -> {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
