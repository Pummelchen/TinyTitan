#!/usr/bin/env python3
"""Every served model, one or more prompts, TTFT and tok/s from the SSE stream.

Reads its whole configuration from the environment inside `main()` and serves
nothing until then: importing this module starts no run and opens no results
file. For each (model, prompt, repeat) it sends a warm-up request on a *model
change* (loads or switches it, discarded) and one measured streaming request to
a server the operator already started. A repeat of the model already resident is
measured back to back, and the row says which it was, so a cold load can be told
apart from the steady state. One JSON object per run is appended to `$RESULTS`
as it goes, so progress is visible while it works.

`main()` returns the exit status and counts the rows it planned:

    0   every planned row came back `ok`
    1   a row failed (the line names the error), or the matrix planned nothing
    2   a run variable was refused before the first request -- named, not a
        traceback

The run behind the wiki's `Capital-of-Paris-Smartness` page:

    .build/release/TinyTitanServer --models-dir models \
        --model qwen3.5-2b_4-Bit --port 8091 --reasoning off

    PORT=8091 MAXTOK=128 REPEATS=3 \
        PROMPTS='["Capital of Paris","What is the capital of France? Answer in one short sentence."]' \
        RESULTS=/tmp/smartness_results.jsonl RUNS="$(cat runs.json)" \
        python3 benchmark/capital_of_paris_smartness.py

`RUNS` is `[[id, engine, label, quant], ...]`; the dense installs are named
`<id>@cpu` for the CPU engine and by their bare id for the GPU. `PROMPTS` is a
JSON array of prompts; a bare `PROMPT` string is still accepted and is one.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

DEFAULT_PORT = 8091
DEFAULT_PROMPT = "Capital of Paris"
DEFAULT_MAXTOK = 128
DEFAULT_RESULTS = "/tmp/smartness_results.jsonl"

# The one piece of configuration `post()` needs, and `main()` sets it from the
# validated environment, so a caller that imports this module has a base URL but
# no run.
BASE = f"http://127.0.0.1:{DEFAULT_PORT}"


class ConfigError(ValueError):
    """A run variable the driver refuses with its name instead of crashing."""


def parse_int(env, key, default=None, minimum=1):
    raw = env.get(key)
    if raw is None:
        if default is None:
            raise ConfigError(f"{key} is not set; it must be an integer of at least {minimum}")
        raw = default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ConfigError(f"{key}={raw!r} is not an integer") from None
    if value < minimum:
        raise ConfigError(f"{key}={value} must be at least {minimum}")
    return value


def parse_prompts(env):
    """`PROMPTS` as a JSON array, `PROMPT` as one bare string, or the default."""
    raw = env.get("PROMPTS")
    if raw is None:
        bare = env.get("PROMPT")
        return [bare] if bare is not None else [DEFAULT_PROMPT]
    try:
        value = json.loads(raw)
    except ValueError:
        raise ConfigError(f"PROMPTS={raw!r} is not valid JSON") from None
    if not isinstance(value, list) or not all(isinstance(p, str) for p in value):
        raise ConfigError("PROMPTS must be a JSON array of strings")
    return value


def parse_runs(env):
    """`RUNS` as a list of (id, engine, label, quant) tuples, every row checked."""
    raw = env.get("RUNS")
    if raw is None:
        raise ConfigError("RUNS is not set; it is a JSON array of [id, engine, label, quant] rows")
    try:
        value = json.loads(raw)
    except ValueError:
        raise ConfigError(f"RUNS={raw[:60]!r} is not valid JSON") from None
    if not isinstance(value, list):
        raise ConfigError("RUNS must be a JSON array of [id, engine, label, quant] rows")
    runs = []
    for index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != 4:
            raise ConfigError(
                f"RUNS row {index}={row!r} is not [id, engine, label, quant] -- four fields"
            )
        runs.append(tuple(row))
    return runs


def parse_config(env):
    port = parse_int(env, "PORT", str(DEFAULT_PORT), 1)
    if port > 65535:
        raise ConfigError(f"PORT={port} is not a TCP port")
    prompts = parse_prompts(env)
    maxtok = parse_int(env, "MAXTOK", str(DEFAULT_MAXTOK))
    repeats = parse_int(env, "REPEATS", "1")
    results = env.get("RESULTS", DEFAULT_RESULTS)
    runs = parse_runs(env)
    return {
        "port": port,
        "prompts": prompts,
        "maxtok": maxtok,
        "repeats": repeats,
        "results": results,
        "runs": runs,
    }


def post(payload, timeout=1800):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions", data=body, headers={"content-type": "application/json"}
    )
    return urllib.request.urlopen(req, timeout=timeout)


def warm(model, prompt):
    """Load or switch to the model, with the request that is then measured.

    The prompt is the one under test, not a placeholder: prefill for a long
    prompt is part of what the request costs, and warming with a different
    prompt would leave the KV cache holding the wrong prefix.
    """
    t0 = time.monotonic()
    with post(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
            "temperature": 0,
        }
    ) as r:
        r.read()
    return time.monotonic() - t0


def measure(model, prompt, maxtok=DEFAULT_MAXTOK):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": maxtok,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t_send = time.monotonic()
    ttft = None
    t_last = t_send
    content, reasoning, usage, finish = "", "", None, None
    with post(payload) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except ValueError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                delta = ch.get("delta") or {}
                piece_c = delta.get("content") or ""
                piece_r = delta.get("reasoning_content") or ""
                if (piece_c or piece_r) and ttft is None:
                    ttft = time.monotonic() - t_send
                if piece_c or piece_r:
                    t_last = time.monotonic()
                content += piece_c
                reasoning += piece_r
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    total = time.monotonic() - t_send
    tokens = (usage or {}).get("completion_tokens")
    decode = None
    if tokens and ttft is not None and t_last > t_send + ttft:
        decode = (tokens - 1) / (t_last - (t_send + ttft))
    return {
        "ttft_s": ttft,
        "total_s": total,
        "completion_tokens": tokens,
        "decode_tok_s": decode,
        "e2e_tok_s": (tokens / total) if tokens else None,
        "finish": finish,
        "content": content,
        "reasoning": reasoning,
        "content_chars": len(content),
        "reasoning_chars": len(reasoning),
    }


def shown(row, key, unit=""):
    """A row value for the line: `n/a` where the run never reached one."""
    value = row.get(key)
    return "n/a" if value is None else f"{value}{unit}"


def main():
    try:
        cfg = parse_config(os.environ)
    except ConfigError as e:
        print(f"REFUSED: {e}", flush=True)
        return 2
    global BASE
    BASE = f"http://127.0.0.1:{cfg['port']}"
    runs, prompts, repeats = cfg["runs"], cfg["prompts"], cfg["repeats"]
    planned = len(runs) * len(prompts) * repeats
    if planned == 0:
        print(
            f"NOT MEASURED: the matrix planned no request "
            f"({len(runs)} model x {len(prompts)} prompt x {repeats} repeat)",
            flush=True,
        )
        return 1
    print(
        f"{planned} rows planned: {len(runs)} model x {len(prompts)} prompt x "
        f"{repeats} repeat against {BASE}, results to {cfg['results']}",
        flush=True,
    )

    resident = None
    ok = failed = 0
    for model, engine, label, quant in runs:
        for prompt in prompts:
            for repeat in range(1, repeats + 1):
                row = {
                    "model": model,
                    "engine": engine,
                    "label": label,
                    "quant": quant,
                    "prompt": prompt,
                    "repeat": repeat,
                    "load_s": None,
                    "cold": None,
                }
                switched = model != resident
                try:
                    # Warm only on a switch, so a repeat measures the resident
                    # model rather than another load.
                    row["load_s"] = round(warm(model, prompt), 2) if switched else 0.0
                    row["cold"] = switched
                    if switched:
                        resident = model
                    row.update(measure(model, prompt, cfg["maxtok"]))
                    row["status"] = "ok"
                    ok += 1
                except urllib.error.HTTPError as e:
                    row["status"] = "http_error"
                    row["error"] = f"{e.code} {e.read()[:300].decode('utf-8', 'replace')}"
                    failed += 1
                except Exception as e:  # noqa: BLE001
                    row["status"] = "error"
                    row["error"] = f"{type(e).__name__}: {e}"
                    failed += 1
                for k in ("ttft_s", "total_s", "decode_tok_s", "e2e_tok_s"):
                    if isinstance(row.get(k), float):
                        row[k] = round(row[k], 3)
                with open(cfg["results"], "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(
                    f"{row['status']:10s} {label:26s} {quant}-bit {engine:3s} "
                    f"repeat={repeat} cold={str(row['cold']):5s} "
                    f"load={shown(row, 'load_s', 's')} ttft={shown(row, 'ttft_s', 's')} "
                    f"tok/s={shown(row, 'decode_tok_s')} tokens={shown(row, 'completion_tokens')} "
                    f"content={row.get('content_chars', 0)}ch "
                    f"reasoning={row.get('reasoning_chars', 0)}ch "
                    f"prompt={prompt[:28]!r}",
                    flush=True,
                )
                if row["status"] != "ok":
                    print("           " + str(row.get("error"))[:300], flush=True)

    print(f"{ok} ok, {failed} failed -- {ok + failed} of {planned} rows written", flush=True)
    return 1 if failed or ok + failed != planned else 0


if __name__ == "__main__":
    sys.exit(main())
