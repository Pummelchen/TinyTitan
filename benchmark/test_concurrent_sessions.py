#!/usr/bin/env python3
"""Four concurrent sessions on one server: does every answer belong to its own user?

Every request carries a unique marker that only that session may ever see. A response
that contains another session's marker is a cross-session leak; a response missing its
own marker is a fidelity failure. The marker ledger is global for the model run, so a
stale slot can also be caught leaking a marker from an earlier round or wave.

Three rounds, each run twice with freshly randomised content:

  1. echo+math       copy a code exactly, then answer an addition
  2. marker+capital  include an exact rare phrase, then name a capital
  3. recall          two interleaved turns: remember a code, then return it

Round 3 is the sharpest: each turn-2 request carries only its own history, so any
other user's code appearing in its answer can only have come from server-side state.

Three controls keep a clean result meaningful, rather than the silence of a detector
that never fires:

  * determinism -- four identical greedy requests must come back identical
  * canary      -- a prompt deliberately carrying another session's marker must be
                   flagged, which is the check that the detector works at all
  * cancellation -- one client vanishes mid-generation; the other three must be
                   untouched and the server must still answer afterwards

This is a model run: it starts one server at a time, at `--max-concurrent-sequences 4`
(above 1 the session-wide prompt cache is off by design), and stops it before the next
model starts. Models that are not installed are skipped rather than fetched.

Exit status covers the safety properties and the controls that make them mean
something: a foreign marker, an HTTP failure, a canary the detector did not
catch, or survivors the cancellation did not leave intact. Answer quality
(own_missed, task_wrong, attribution_unproven) is printed beside the verdict
and does not move it, because an incoherent answer is not a mixed session.

    python3 benchmark/test_concurrent_sessions.py
    python3 benchmark/test_concurrent_sessions.py models/qwen3.5_2B_4Bit
    SEED=12345 python3 benchmark/test_concurrent_sessions.py --context 32768

Against a server you started yourself, `BASE=http://127.0.0.1:8090` runs the checks
without launching anything. Exit status is non-zero if any model leaks or misroutes.

A run on the 2B reports `task_wrong` and `own_missed`: that model genuinely
mishandles arithmetic and two-part instructions (measured; see the 2B note on
https://github.com/Pummelchen/TinyTitan/wiki/Getting-Started), which is answer
quality rather than a mixed session and is deliberately not part of the
separation verdict. `TOKEN_SCALE`,
`TEMPERATURE`, `TOP_P`/`TOP_K` and `SERVER_ARGS` exist for thinking runs, which
need their family's own sampling and a budget big enough for the reasoning block.
"""

from __future__ import annotations

import http.client
import json
import os
import pathlib
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[1]
BINARY = ROOT / ".build/release/TinyTitanServer"
# The two dense installs both engines serve, small enough to run this quickly.
DEFAULT_MODELS = ["models/qwen3.5_2B_4Bit", "models/qwen3.5_4B_4Bit"]

# Set per model by `run_one_model`. An empty BASE means an already-running server
# named by the environment, in which case this script launches nothing.
BASE = os.environ.get("BASE", "")
PORT = 0
MODEL = ""
SEED = os.environ.get("SEED") or str(int(time.time()))
# A thinking model spends tokens before it answers, so the per-round budgets can
# be scaled up: a truncated answer is a harness artifact, not a model failure.
# `SERVER_ARGS` passes extra server flags through (for example `--thinking on`).
#
# Sampling matters as much as the budget: a reasoning model decoded greedily can
# loop in its thinking block until the budget runs out and never answer, so a
# thinking run has to use the model's own sampling (temperature 0.6, top_p 0.95,
# top_k 20 for the Qwen 3.5 family). TEMPERATURE also decides whether the
# determinism control means anything.
TOKEN_SCALE = float(os.environ.get("TOKEN_SCALE", "1"))
SERVER_ARGS = os.environ.get("SERVER_ARGS", "").split()
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0"))
TOP_P = float(os.environ.get("TOP_P", "0"))
TOP_K = int(os.environ.get("TOP_K", "0"))
random.seed(SEED)


def served_model() -> str:
    with urllib.request.urlopen(BASE + "/v1/models", timeout=30) as response:
        ids = [m["id"] for m in json.load(response)["data"]]
    base = [i for i in ids if not i.endswith("-fast")]
    return (base or ids)[0]


def chat(messages: list[dict], max_tokens: int, temperature: float | None = None) -> dict:
    max_tokens = max(8, int(max_tokens * TOKEN_SCALE))
    payload = {
        "model": MODEL,
        "temperature": TEMPERATURE if temperature is None else temperature,
        "max_completion_tokens": max_tokens,
        "messages": messages,
    }
    if TOP_P > 0:
        payload["top_p"] = TOP_P
    if TOP_K > 0:
        payload["top_k"] = TOP_K
    request = urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(payload).encode(), method="POST"
    )
    request.add_header("content-type", "application/json")
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            body = json.loads(response.read())
            choice = body["choices"][0]
            return {
                "status": response.status,
                "text": choice["message"].get("content") or "",
                "finish": choice.get("finish_reason"),
                "usage": body.get("usage"),
                "elapsed": time.monotonic() - started,
                "error": None,
            }
    except urllib.error.HTTPError as error:
        return {
            "status": error.code,
            "text": "",
            "finish": None,
            "usage": None,
            "elapsed": time.monotonic() - started,
            "error": error.read().decode("utf-8", "replace"),
        }


# ---------------------------------------------------------------- markers

CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
RARE_WORDS = [
    "obsidian",
    "quokka",
    "abacus",
    "velvet",
    "lantern",
    "meridian",
    "cobalt",
    "thistle",
    "zephyr",
    "garnet",
    "walrus",
    "juniper",
]
CAPITALS = {
    "France": "Paris",
    "Japan": "Tokyo",
    "Kenya": "Nairobi",
    "Peru": "Lima",
    "Norway": "Oslo",
    "Thailand": "Bangkok",
    "Chile": "Santiago",
    "Portugal": "Lisbon",
    "Canada": "Ottawa",
    "Egypt": "Cairo",
}

LEDGER: list[dict] = []


def new_code() -> str:
    return "".join(random.choice(CODE_ALPHABET) for _ in range(6))


def issue(label: str, marker: str) -> None:
    LEDGER.append({"label": label, "marker": marker})


def normalize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", text).upper()


def foreign_hits(text: str, own: list[str]) -> list[str]:
    """Ledger markers that appear in this answer but belong to another session."""
    normalized = normalize(text)
    own_normalized = {normalize(m) for m in own}
    return [
        entry["label"]
        for entry in LEDGER
        if normalize(entry["marker"]) not in own_normalized
        and normalize(entry["marker"]) in normalized
    ]


# ---------------------------------------------------------------- rounds


def round_echo_math(wave: int) -> list[dict]:
    users = []
    for index in range(4):
        marker, left, right = new_code(), random.randint(120, 980), random.randint(120, 980)
        label = f"r1 wave{wave} user{index + 1}"
        issue(label, marker)
        users.append(
            {
                "label": label,
                "own": [marker],
                "kind": "echo+math",
                "expect": f"code {marker} + {left}+{right}={left + right}",
                "sum": left + right,
                "operands": [str(left), str(right)],
                "messages": [
                    {
                        "role": "user",
                        "content": "Follow these instructions exactly.\n"
                        f"Line 1: write the code {marker} exactly as written.\n"
                        f"Line 2: write only the result of {left} + {right}.\n"
                        "Write nothing else.",
                    }
                ],
                "max_tokens": 96,
            }
        )
    return users


def round_marker_capital(wave: int) -> list[dict]:
    users = []
    for index in range(4):
        marker = f"{random.choice(RARE_WORDS)}-{random.randint(1000, 9999)}"
        country = random.choice(list(CAPITALS))
        label = f"r2 wave{wave} user{index + 1}"
        issue(label, marker)
        users.append(
            {
                "label": label,
                "own": [marker],
                "kind": "marker+capital",
                "expect": f"phrase {marker} + capital of {country}",
                "capital": CAPITALS[country],
                "messages": [
                    {
                        "role": "user",
                        "content": f"Write one short sentence that contains the exact phrase {marker}.\n"
                        f"Then, on a new line, write the capital city of {country}.",
                    }
                ],
                "max_tokens": 96,
            }
        )
    return users


def round_recall(wave: int) -> list[dict]:
    users = []
    for index in range(4):
        marker = new_code()
        label = f"r3 wave{wave} user{index + 1}"
        issue(label, marker)
        users.append(
            {
                "label": label,
                "own": [marker],
                "kind": "recall",
                "code": marker,
                "expect": f"return {marker}",
            }
        )
    return users


def fire(users: list[dict]) -> list[dict]:
    """Submit together, with a small random stagger so slot order varies."""

    def one(index: int) -> tuple[int, dict]:
        time.sleep(random.uniform(0.0, 0.15))
        return index, chat(users[index]["messages"], users[index].get("max_tokens", 96))

    results: list[dict | None] = [None] * len(users)
    with ThreadPoolExecutor(max_workers=len(users)) as pool:
        for index, result in pool.map(one, range(len(users))):
            results[index] = result
    return [r for r in results if r is not None]  # type: ignore[return-value]


# ---------------------------------------------------------------- reporting

TOTALS = {
    "responses": 0,
    "own_missed": 0,
    "foreign": 0,
    "http_error": 0,
    "task_wrong": 0,
    "cache_on": 0,
    "attribution_unproven": 0,
}


def reset_state() -> None:
    """Begin a run with no ledger and no totals, so a later run cannot inherit them.

    Both entry points need this: the launched path clears them per model, and the
    `BASE=` path used not to, so an in-process second run judged its own rounds on
    totals the first one left behind.
    """
    LEDGER.clear()
    for key in list(TOTALS):
        TOTALS[key] = 0


def separation_state() -> tuple[int, bool, bool, bool]:
    """(leaked, canary_ok, cancel_ok, separation_bad) for the totals as they stand.

    Separation is gated on the safety properties only: a leak (with the canary
    proving the detector fires), the HTTP boundary, and the other slots surviving a
    cancelled neighbour. Answer quality is the model's, reported beside it -- an
    incoherent answer is not a mixed session. The two entry points must agree, or the
    one an operator runs against their own server is the one that can report a blind
    detector as a clean result (AUD-252).
    """
    leaked = TOTALS["foreign"] > 0
    canary_ok = TOTALS.get("canary_detected", 0) == 1
    cancel_ok = (
        TOTALS.get("cancel_survivors_intact", 0) == 1 and TOTALS.get("post_cancel_alive", 0) == 1
    )
    separation_bad = leaked or TOTALS["http_error"] > 0 or not canary_ok or not cancel_ok
    return TOTALS["foreign"], canary_ok, cancel_ok, separation_bad


def print_verdict(label: str) -> bool:
    """Print the totals, the controls and the verdict; return whether it passed."""
    foreign, canary_ok, cancel_ok, separation_bad = separation_state()
    unproven = TOTALS.get("attribution_unproven", 0)
    print("\n================ summary ================")
    for key, value in TOTALS.items():
        print(f"  {key:18} {value}")
    print(f"  markers issued     {len(LEDGER)}")
    print(
        f"  controls: canary={'caught' if canary_ok else 'BLIND'} "
        f"cancel={'intact' if cancel_ok else 'BROKEN'} "
        f"identical_greedy_answers={TOTALS.get('det_distinct')}"
    )
    print(
        f"VERDICT {label} separation={'FAIL' if separation_bad else 'PASS'} "
        f"(leaks={foreign}, http_error={TOTALS['http_error']}, "
        f"canary={'caught' if canary_ok else 'BLIND'})"
    )
    print(
        f"  not separation: own_missed={TOTALS['own_missed']} "
        f"task_wrong={TOTALS['task_wrong']} attribution_unproven={unproven} "
        f"cached_tokens_nonzero={TOTALS['cache_on']}"
    )
    return not separation_bad


def report(users: list[dict], results: list[dict], phase: str) -> None:
    print(f"\n--- {phase} ---")
    for user, result in zip(users, results, strict=False):
        TOTALS["responses"] += 1
        # A "recall" row is the turn-1 acknowledgement: it is not supposed to
        # echo the code, so only the foreign-marker check applies to it.
        enforce_own = user["kind"] != "recall"
        own = all(normalize(m) in normalize(result["text"]) for m in user["own"])
        foreign = foreign_hits(result["text"], user["own"])
        own_hit = enforce_own and not own
        TOTALS["own_missed"] += int(own_hit)
        # The canary is the detector's own self-test: its foreign hit is the
        # expected result, so it must not be counted as a real leak.
        if user["kind"] != "canary":
            TOTALS["foreign"] += len(foreign)
        if result["error"] is not None or result["status"] != 200:
            TOTALS["http_error"] += 1

        task = "n/a"
        if user["kind"] == "echo+math":
            found = re.search(rf"\b{user['sum']}\b", result["text"]) is not None
            task = "sum ok" if found else "SUM WRONG"
            if not found:
                TOTALS["task_wrong"] += 1
            # Whether this user's own prompt can be shown to have reached this
            # slot: the code alone settles it, and the right sum or this user's
            # own operands settle it for an answer that dropped the code line. An
            # answer with none of the three proves nothing either way, which is a
            # fidelity gap rather than a leak -- it is counted and reported, not
            # used to fail the run.
            operands = all(normalize(o) in normalize(result["text"]) for o in user["operands"])
            routed = own or found or operands
            if not routed:
                TOTALS["attribution_unproven"] += 1
            task += f", own material {'seen' if operands else ('sum only' if found else 'MISSING')}"
        elif user["kind"] == "marker+capital":
            found = user["capital"].lower() in result["text"].lower()
            task = "capital ok" if found else "CAPITAL WRONG"
            if not found:
                TOTALS["task_wrong"] += 1
        elif user["kind"] == "recall-answer":
            exact = normalize(result["text"]) == normalize(user["code"])
            task = "recall EXACT" if exact else "recall approximate"
            if not exact:
                TOTALS["task_wrong"] += 1

        usage = result["usage"] or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        if cached:
            TOTALS["cache_on"] += 1
        flag = "LEAK" if foreign else ("own-marker MISSING" if own_hit else "clean")
        print(
            f"  {user['label']:16} HTTP {result['status']} "
            f"{result['elapsed']:5.2f}s tok={usage.get('completion_tokens')} "
            f"prompt={usage.get('prompt_tokens')} cached={cached} "
            f"finish={result['finish']} "
            f"| own={'yes' if own else 'NO'} foreign={foreign or 'none'} "
            f"| {task} | {flag}"
        )
        print(f"      expect: {user['expect']}")
        print(
            f"      got   : {' '.join(result['text'].split())[:200]}"
            + (f"   [error: {result['error'][:120]}]" if result["error"] else "")
        )


def run_round(title: str, maker) -> None:
    for wave in (1, 2):
        users = maker(wave)
        started = time.monotonic()
        results = fire(users)
        spread = max(r["elapsed"] for r in results) - min(r["elapsed"] for r in results)
        print(
            f"\n===== {title} / wave {wave} "
            f"(4 concurrent, wall {time.monotonic() - started:.2f}s, "
            f"completion spread {spread:.2f}s) ====="
        )
        report(users, results, f"{title} wave{wave}")


def run_recall_round() -> None:
    for wave in (1, 2):
        users = round_recall(wave)
        first = fire(
            [
                {
                    "label": u["label"],
                    "own": u["own"],
                    "kind": "recall-setup",
                    "expect": f"ack {u['code']}",
                    "max_tokens": 8,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Remember this code: {u['code']}. Reply with just: OK",
                        }
                    ],
                }
                for u in users
            ]
        )
        print(f"\n===== recall / wave {wave} — turn 1 (set the code) =====")
        report(users, first, f"recall wave{wave} turn1")

        turn2 = []
        for user, ack in zip(users, first, strict=False):
            turn2.append(
                {
                    "label": user["label"],
                    "own": user["own"],
                    "kind": "recall-answer",
                    "expect": user["expect"],
                    "code": user["code"],
                    "max_tokens": 16,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Remember this code: {user['code']}. Reply with just: OK",
                        },
                        {"role": "assistant", "content": ack["text"] or "OK"},
                        {
                            "role": "user",
                            "content": "What code did I ask you to remember? Reply with only the code, "
                            "nothing else.",
                        },
                    ],
                }
            )
        results = fire(turn2)
        print(f"\n===== recall / wave {wave} — turn 2 (return the code) =====")
        report(turn2, results, f"recall wave{wave} turn2")


def run_determinism_control() -> None:
    """Four identical greedy requests: variation would mean cross-slot bleed.

    The task is a high-margin copy, so a correct run is identical four times.
    """
    marker = new_code()
    issue("control determinism", marker)
    prompt = f"Reply with exactly this code and nothing else: {marker}"
    users = [
        {
            "label": f"det u{i + 1}",
            "own": [marker],
            "kind": "determinism",
            "expect": f"echo {marker}",
            "max_tokens": 24,
            "messages": [{"role": "user", "content": prompt}],
        }
        for i in range(4)
    ]
    results = fire(users)
    print("\n===== control 1: determinism (4 identical greedy requests) =====")
    report(users, results, "control determinism")
    distinct = {normalize(r["text"]) for r in results}
    TOTALS["det_distinct"] = len(distinct)
    if TEMPERATURE == 0:
        print(
            f"  distinct answers: {len(distinct)} "
            f"({'deterministic' if len(distinct) == 1 else 'VARIATION at temperature 0'})"
        )
    else:
        print(
            f"  distinct answers: {len(distinct)} (sampling at temperature "
            f"{TEMPERATURE}; variation is expected, determinism is not a control here)"
        )


def run_canary_control() -> None:
    """A prompt that deliberately carries another session's marker must be caught.

    Without this the rest of the run proves nothing: a detector that never fires
    looks exactly like a clean run.
    """
    victim = LEDGER[0]
    marker = new_code()
    issue("control canary", marker)
    users = [
        {
            "label": "canary",
            "own": [marker],
            "kind": "canary",
            "expect": f"must reveal {victim['label']} ({victim['marker']})",
            "max_tokens": 40,
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with exactly these two codes separated by a comma, "
                    f"and nothing else: {victim['marker']}, {marker}",
                }
            ],
        }
    ]
    results = fire(users)
    print("\n===== control 2: canary (the detector must fire) =====")
    report(users, results, "control canary")
    detected = foreign_hits(results[0]["text"], users[0]["own"])
    TOTALS["canary_detected"] = int(victim["label"] in detected)
    print(f"  detector flagged: {detected or 'NOTHING — the detector is blind'}")


def abandon(messages: list[dict], max_tokens: int, after: float) -> dict:
    """Send a request, then close the connection mid-generation."""
    payload = {
        "model": MODEL,
        "temperature": 0.0,
        "max_completion_tokens": max_tokens,
        "messages": messages,
    }
    connection = http.client.HTTPConnection("127.0.0.1", PORT, timeout=60)
    connection.request(
        "POST",
        "/v1/chat/completions",
        body=json.dumps(payload),
        headers={"content-type": "application/json"},
    )
    time.sleep(after)
    connection.close()
    return {
        "status": "abandoned",
        "text": "",
        "elapsed": after,
        "usage": None,
        "finish": None,
        "error": None,
    }


def run_cancellation_control() -> None:
    """One client vanishes mid-generation; the other three must be untouched.

    A failed generation once reset every slot rather than its own, so a dropped
    client could corrupt live sequences. This is that failure mode under load.
    """
    users = round_echo_math(3)
    victim, survivors = users[0], users[1:]
    with ThreadPoolExecutor(max_workers=4) as pool:
        dropped = pool.submit(abandon, victim["messages"], victim.get("max_tokens", 96), 1.0)
        time.sleep(0.4)
        results = list(pool.map(lambda u: chat(u["messages"], u.get("max_tokens", 96)), survivors))
        dropped.result()
    print("\n===== control 3: cancellation (one client vanishes mid-generation) =====")
    report(survivors, results, "control cancellation survivors")
    intact = all(
        not foreign_hits(r["text"], u["own"]) and normalize(u["own"][0]) in normalize(r["text"])
        for u, r in zip(survivors, results, strict=False)
    )
    follow = chat([{"role": "user", "content": "Reply with exactly: ALIVE"}], 8)
    alive = "ALIVE" in follow["text"].upper()
    TOTALS["cancel_survivors_intact"] = int(intact)
    TOTALS["post_cancel_alive"] = int(alive)
    print(
        f"  abandoned {victim['label']}; survivors intact: {intact}; "
        f"server answers afterwards: {alive}"
    )


def start_server(model_dir: pathlib.Path, port: int, context: int, concurrency: int):
    process = subprocess.Popen(
        [
            str(BINARY),
            "--model",
            str(model_dir),
            "--port",
            str(port),
            "--max-context",
            str(context),
            "--max-concurrent-sequences",
            str(concurrency),
            *SERVER_ARGS,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"server exited before becoming healthy:\n{process.stdout.read()}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
            return process
        except Exception:
            time.sleep(1)
    process.terminate()
    raise SystemExit("server did not become healthy in time")


def stop_server(process) -> str:
    process.terminate()
    try:
        return process.communicate(timeout=60)[0] or ""
    except subprocess.TimeoutExpired:
        process.kill()
        return process.communicate()[0] or ""


def run_one_model(
    label: str, model_dir: pathlib.Path, port: int, context: int, concurrency: int
) -> bool:
    """One model, all three rounds and all three controls. True when separated."""
    global BASE, PORT, MODEL
    print(f"\n################ {label} ({model_dir}) ################")
    process = start_server(model_dir, port, context, concurrency)
    try:
        BASE, PORT = f"http://127.0.0.1:{port}", port
        MODEL = served_model()
        reset_state()
        print(f"model={MODEL} seed={SEED} base={BASE}")

        run_round("round 1: echo+math", round_echo_math)
        run_round("round 2: marker+capital", round_marker_capital)
        run_recall_round()
        run_determinism_control()
        run_canary_control()
        run_cancellation_control()

        return print_verdict(MODEL)
    finally:
        log = stop_server(process)
        for line in log.splitlines():
            if "ready at" in line:
                print(f"  banner: {line}")
        clamp = [line for line in log.splitlines() if "batch width" in line]
        print(f"  width clamped: {'yes — ' + clamp[0] if clamp else 'no (width held)'}")
        shed = sum(1 for line in log.splitlines() if "status=429" in line)
        print(f"  429s: {shed}")


def parse_args() -> tuple[list[str], int, int, int]:
    """Positional model directories, then --context/--concurrency/--port."""
    models: list[str] = []
    seen: dict[str, int] = {}
    argv = sys.argv[1:]
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument in ("--context", "--concurrency", "--port"):
            seen[argument] = int(argv[index + 1])
            index += 2
            continue
        models.append(argument)
        index += 1
    context = seen.get("--context", int(os.environ.get("CONTEXT", 32768)))
    concurrency = seen.get("--concurrency", int(os.environ.get("CONCURRENCY", 4)))
    port = seen.get("--port", int(os.environ.get("PORT", 8090)))
    return models, context, concurrency, port


def main() -> int:
    argv, context, concurrency, port = parse_args()

    # A server the caller started: check it and launch nothing.
    if BASE:
        global MODEL
        reset_state()
        MODEL = served_model()
        print(f"model={MODEL} seed={SEED} base={BASE} (external server)")
        run_round("round 1: echo+math", round_echo_math)
        run_round("round 2: marker+capital", round_marker_capital)
        run_recall_round()
        run_determinism_control()
        run_canary_control()
        run_cancellation_control()
        # The same predicate the launched path uses: this branch is the one the
        # docstring tells an operator to run, so it may not drop the two controls
        # that make "no foreign marker" mean something.
        return 0 if print_verdict(MODEL) else 1

    if not BINARY.is_file():
        raise SystemExit(f"no server built at {BINARY}; run: swift build -c release")

    candidates = argv or DEFAULT_MODELS
    present = [m for m in candidates if (ROOT / m).is_dir()]
    for missing in [m for m in candidates if m not in present]:
        print(f"SKIP {missing}: not installed (this test never fetches a model)")
    if not present:
        raise SystemExit("no installed model to test")

    results = {}
    for index, model in enumerate(present):
        results[model] = run_one_model(
            pathlib.Path(model).name, ROOT / model, port + index, context, concurrency
        )

    print("\n================ overall ================")
    for model, ok in results.items():
        print(f"  {model:36} {'PASS' if ok else 'FAIL'}")
    print(f"seed={SEED}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
