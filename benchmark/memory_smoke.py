"""Memory against a real model, in three requests.

Not a benchmark: a check that the wiring does what the unit tests say it
does when a 35B model is on the other end. A novel session stores a fact,
a codebase session must not see it, a second novel session must. Placement
is by the working directory the client declares, exactly as Claude Code
and Codex declare it, so this is the book-versus-git case end to book.

Every section reports one of three ways, because "nothing was checked" and
"nothing was wrong" print the same verdict otherwise:

- a check that ran and passed prints its line and the run exits 0,
- a check that could not run -- a request that did not answer, a log that is
  not there, a memory directory that was never configured -- prints
  `NOT MEASURED` or `NOT CHECKED` and the run exits 1,
- a server that does not answer `/models` at all prints `NOT RUN` and sends
  no request.

The environment is read when the run starts, not when this file is imported:
`TINYTITAN_PORT` (default 8096), `TINYTITAN_MEMVAL_MEMDIR` (the memory
directory the journals are read from) and `TINYTITAN_MEMVAL_SERVER_LOG` (the
server log the consolidation line is read from). An unset log or memory
directory is reported as unconfigured; it is never read as the model failing
to write, and the launch directory is never read as the memory directory.

    benchmark/memval_run.sh smoke      # memory on, no tools: the engine must do the writing
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PORT_ENV = "TINYTITAN_PORT"
MEMDIR_ENV = "TINYTITAN_MEMVAL_MEMDIR"
LOG_ENV = "TINYTITAN_MEMVAL_SERVER_LOG"
DEFAULT_PORT = 8096

# Memory on with no tools: the fragment is ~90 tokens on top of the ~60-token
# request, which measured 149 prompt tokens answered. Under this floor the
# fragment is not in the prompt at all, so the recall checks below measure the
# conversation rather than memory.
PROMPT_FLOOR_TOKENS = 120
CONSOLIDATION_POLLS = 90
CONSOLIDATION_POLL_SECONDS = 2
JOURNAL_SETTLE_SECONDS = 3
REQUEST_TIMEOUT_SECONDS = 1800
MODELS_TIMEOUT_SECONDS = 30
FACT_MARKER = "ashgrove"
WORKSPACES = ("photograph", "widget")

NOVEL = (
    "You are a coding assistant.\n\n# Environment\n"
    " - Primary working directory: /Users/ada/novels/photograph\n"
)
CODE = "<environment_context>\n  <cwd>/Users/ada/src/widget</cwd>\n</environment_context>"
STORE_USER = (
    "Store this in memory for later sessions: in this novel the town is called "
    "Ashgrove and it never rains there. Then confirm in one sentence."
)
RECALL_CODE_USER = (
    "What do you already know about this project from memory? One sentence; "
    "say 'nothing' if nothing."
)
RECALL_NOVEL_USER = "What do you already know about this novel from memory? One sentence."


class ConfigError(ValueError):
    """The operator named something the driver cannot use."""


def port() -> int:
    raw = os.environ.get(PORT_ENV, str(DEFAULT_PORT)).strip()
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{PORT_ENV}={raw!r} is not a port number") from None
    if not 1024 <= value <= 65535:
        raise ConfigError(f"{PORT_ENV}={value} is not a port a server can bind")
    return value


def base_url() -> str:
    return f"http://127.0.0.1:{port()}/v1"


def memdir() -> Path | None:
    raw = os.environ.get(MEMDIR_ENV, "").strip()
    return Path(raw) if raw else None


def log_path() -> Path | None:
    raw = os.environ.get(LOG_ENV, "").strip()
    return Path(raw) if raw else None


def consolidation_lines(where: Path | None) -> list[str]:
    """The `consolidated session=` lines in the server log, newest last."""
    if where is None or not where.exists():
        return []
    with open(where, encoding="utf-8", errors="replace") as handle:
        return [line.strip() for line in handle if "consolidated session=" in line]


def journal_files(where: Path | None) -> list[str]:
    """Every `.ndjson` under the memory directory, named relative to it."""
    if where is None or not where.is_dir():
        return []
    return sorted(str(path.relative_to(where)) for path in where.rglob("*.ndjson"))


def model_id() -> str:
    with urllib.request.urlopen(f"{base_url()}/models", timeout=MODELS_TIMEOUT_SECONDS) as response:
        return json.load(response)["data"][0]["id"]


def ask(model: str, system: str, user: str, label: str) -> dict:
    """One chat request as a row: what came back, or why nothing did."""
    body = json.dumps(
        {
            "model": model,
            "temperature": 0,
            "max_completion_tokens": 400,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
    ).encode()
    request = urllib.request.Request(
        f"{base_url()}/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    started = time.monotonic()
    status = None
    error = None
    payload: dict = {}
    raw = b""
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as exc:
        status, raw = exc.code, exc.read() or b"{}"
    except OSError as exc:
        error = f"the request raised {type(exc).__name__}: {exc}"
    elapsed = time.monotonic() - started
    if error is None:
        try:
            decoded = json.loads(raw or b"{}")
            if not isinstance(decoded, dict):
                raise ValueError("not an object")
            payload = decoded
        except (ValueError, UnicodeDecodeError):
            error = f"HTTP {status} answered a body that is not a JSON object: {raw[:120]!r}"
    message = (payload.get("choices") or [{}])[0].get("message") or {}
    content = (message.get("content") or "").strip()
    usage = payload.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens") or 0
    completion_tokens = usage.get("completion_tokens") or 0
    print(
        f"[{label}] HTTP {status}, {elapsed:.0f}s, prompt {prompt_tokens} "
        f"completion {completion_tokens}",
        flush=True,
    )
    print(f"   {content[:300]}", flush=True)
    if error:
        print(f"   ERROR: {error}", flush=True)
    return {
        "label": label,
        "status": status,
        "content": content,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "error": error,
    }


def why_not(row: dict) -> str | None:
    """Why this row cannot carry its check, or None when it can."""
    if row["error"]:
        return row["error"]
    if row["status"] != 200:
        return f"HTTP {row['status']} answered instead of a completion"
    if not row["content"]:
        return "the answer held no content"
    return None


def prompt_floor_report(row: dict, floor: int = PROMPT_FLOOR_TOKENS) -> tuple[list[str], int]:
    """(lines, status) for whether the memory fragment reached the prompt."""
    blocked = why_not(row)
    if blocked:
        return (
            [
                f"NOT MEASURED: {row['label']} -- {blocked}, so the prompt-token floor that says "
                "the memory fragment is in the prompt did not run."
            ],
            1,
        )
    if row["prompt_tokens"] < floor:
        return (
            [
                f"FAILED: {row['label']} -- the answered prompt was {row['prompt_tokens']} tokens, "
                f"under the {floor}-token floor, so the memory fragment is not in the prompt."
            ],
            1,
        )
    return (
        [
            f"  [{row['label']}] prompt {row['prompt_tokens']} tokens, at or over the {floor}-token "
            "floor: the memory fragment is in the prompt"
        ],
        0,
    )


def consolidation_report(where: Path | None, lines: list[str]) -> tuple[list[str], int]:
    """(lines, status) for the write the engine was measured not making."""
    if where is None:
        return (
            [
                f"NOT MEASURED: consolidation -- {LOG_ENV} is unset, so no server log was read and "
                "the write the engine may or may not have made is unknown."
            ],
            1,
        )
    if not where.exists():
        return (
            [
                f"NOT MEASURED: consolidation -- no server log at {where}, so the write the engine "
                "may or may not have made is unknown."
            ],
            1,
        )
    if not lines:
        waited = CONSOLIDATION_POLLS * CONSOLIDATION_POLL_SECONDS
        return (
            [
                f"FAILED: consolidation -- no consolidation line in {where} within {waited}s: the "
                "novel session was never distilled."
            ],
            1,
        )
    if "facts=0" in lines[-1]:
        return (
            [f"FAILED: consolidation -- it ran and wrote no facts: {lines[-1]}"],
            1,
        )
    return ([f"  consolidation: {lines[-1]}"], 0)


def isolation_report(row: dict, saw_fact: bool) -> tuple[list[str], int]:
    """(lines, status) for the codebase session that must not know the novel."""
    blocked = why_not(row)
    if blocked:
        return (
            [
                f"NOT MEASURED: {row['label']} -- {blocked}, so the placement isolation check did "
                "not run; nothing here says the fact stayed out of the other workspace."
            ],
            1,
        )
    if saw_fact:
        return (
            [f"FAILED: {row['label']} -- the codebase session saw the novel's fact."],
            1,
        )
    return (
        [
            f"  [{row['label']}] isolation held: the codebase session did not repeat the novel's fact"
        ],
        0,
    )


def recall_report(row: dict, saw_fact: bool) -> tuple[list[str], int]:
    """(lines, status) for the second novel session that must know the fact."""
    blocked = why_not(row)
    if blocked:
        return (
            [
                f"NOT MEASURED: {row['label']} -- {blocked}, so the recall check did not run; this "
                "is not the model failing to remember."
            ],
            1,
        )
    if not saw_fact:
        return (
            [f"FAILED: {row['label']} -- the second novel session did not recall the fact."],
            1,
        )
    return ([f"  [{row['label']}] recall: the second novel session repeated the fact"], 0)


def journal_report(root: Path | None, files: list[str]) -> tuple[list[str], int]:
    """(lines, status) for the journal files the two sessions should have left."""
    if root is None:
        return (
            [
                f"NOT CHECKED: journals -- {MEMDIR_ENV} is unset, so nothing was scanned. The "
                "directory this runs from is not the memory directory."
            ],
            1,
        )
    if not root.is_dir():
        return (
            [f"NOT CHECKED: journals -- no memory directory at {root}."],
            1,
        )
    missing = [name for name in WORKSPACES if not any(name in item for item in files)]
    if missing:
        return (
            [
                f"FAILED: journals -- {root} holds {len(files)} .ndjson file(s) and none for "
                f"{', '.join(missing)}."
            ],
            1,
        )
    return ([f"  journals: {len(files)} file(s) under {root}, one workspace per session"], 0)


def saw_fact(row: dict) -> bool:
    """Whether this answer repeats the stored fact."""
    return FACT_MARKER in row["content"].lower()


def run() -> int:
    print(f"# Base: {base_url()}", flush=True)
    print(
        "# Checks: novel-1 stores, the prompt floor, consolidation, code-1 isolation, "
        "novel-2 recall, journals",
        flush=True,
    )
    try:
        model = model_id()
    except OSError as exc:
        print(
            f"NOT RUN: no model list answered at {base_url()}/models -- "
            f"{type(exc).__name__}: {exc}. No request was sent.",
            flush=True,
        )
        return 1
    print(f"# Model: {model}", flush=True)

    reports: list[tuple[list[str], int]] = []
    rows: dict[str, dict] = {}

    rows["novel-1"] = ask(model, NOVEL, STORE_USER, "novel-1")

    # The novel session is over. With the runner's short idle the engine should
    # distil it before the next request; a real person's pause does the same.
    # This is the write the model was measured not making, and it is only waited
    # for when there is a log to read.
    where = log_path()
    lines: list[str] = []
    if where is not None and where.exists():
        for _ in range(CONSOLIDATION_POLLS):
            lines = consolidation_lines(where)
            if lines:
                break
            time.sleep(CONSOLIDATION_POLL_SECONDS)
        lines = consolidation_lines(where)

    reports.append(prompt_floor_report(rows["novel-1"]))
    reports.append(consolidation_report(where, lines))

    rows["code-1"] = ask(model, CODE, RECALL_CODE_USER, "code-1")
    reports.append(isolation_report(rows["code-1"], saw_fact(rows["code-1"])))

    rows["novel-2"] = ask(model, NOVEL, RECALL_NOVEL_USER, "novel-2")
    reports.append(recall_report(rows["novel-2"], saw_fact(rows["novel-2"])))

    root = memdir()
    if root is not None and root.is_dir():
        time.sleep(JOURNAL_SETTLE_SECONDS)
    reports.append(journal_report(root, journal_files(root)))

    for lines, _status in reports:
        for line in lines:
            print(line, flush=True)

    status = 1 if any(report_status for _lines, report_status in reports) else 0
    if status:
        print("\nSMOKE FAILED: at least one check did not pass, see the lines above.")
    else:
        print(
            "\nSMOKE OK: placement by declared directory, fact carried within the "
            "project and not across it."
        )
    return status


def main() -> int:
    try:
        status = run()
    except ConfigError as exc:
        print(f"REFUSED: {exc}. No request was sent.", flush=True)
        return 2
    if status == 0:
        print("\n" + "=" * 78 + "\nSMOKE COMPLETE\n" + "=" * 78, flush=True)
    else:
        print("\n" + "=" * 78 + "\nSMOKE STOPPED\n" + "=" * 78, flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
