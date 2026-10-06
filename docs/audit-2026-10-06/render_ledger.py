#!/usr/bin/env python3
"""Render the audit ledger.

`ledger.json` is the single source of truth (§9); `ledger.md` is generated output and is
never edited by hand. Run from the repository root:

    python3 docs/audit-2026-10-06/render_ledger.py
"""

from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
LEDGER = HERE / "ledger.json"
RENDERED = HERE / "ledger.md"

SEVERITY_ORDER = ("S0", "S1", "S2", "S3")
STATUS_ORDER = ("START", "PROGRESS", "TEST", "AUDIT", "SWEPT", "OPEN", "BLOCKED", "DONE")


def sort_key(task: dict) -> tuple[int, int, int, str]:
    """Severity first, then the gate a task is stuck at, then its id."""
    severity = task["severity"]
    if severity not in SEVERITY_ORDER:
        raise SystemExit(f"{task['id']}: unknown severity {severity}")
    status = task["status"]
    if status not in STATUS_ORDER:
        raise SystemExit(f"{task['id']}: unknown status {status}")
    return (SEVERITY_ORDER.index(severity), STATUS_ORDER.index(status), 0, task["id"])


def counts(tasks: list[dict]) -> dict[str, int]:
    terminal = {"DONE", "BLOCKED"}
    return {
        "total": len(tasks),
        "done": sum(1 for task in tasks if task["status"] == "DONE"),
        "open": sum(1 for task in tasks if task["status"] not in terminal),
        "blocked": sum(1 for task in tasks if task["status"] == "BLOCKED"),
    }


def row(task: dict) -> str:
    return (
        f"| {task['id']} | {task['severity']} | {task['tier']} | {task['project']} | "
        f"`{task['file_line']}` | {task['title']} | {task['category']} | {task['status']} | "
        f"{task['host']} |"
    )


def detail(task: dict) -> list[str]:
    lines = [f"### {task['id']} — {task['title']}", ""]
    facts = (
        ("Severity / tier", f"{task['severity']} / Tier {task['tier']}"),
        ("Project", task["project"]),
        ("Location", f"`{task['file_line']}`"),
        ("Category", task["category"]),
        ("Status", task["status"]),
        ("Host", task["host"]),
        ("Discovered by", task["discovered-by"]),
    )
    lines += [f"- **{label}:** {value}" for label, value in facts]
    lines += ["", f"**Evidence before.** {task['evidence-before']}", ""]
    if task["fix-summary"]:
        lines += [f"**Fix.** {task['fix-summary']}", ""]
    if task["evidence-after"]:
        lines += [f"**Evidence after.** {task['evidence-after']}", ""]
    if task["commit"]:
        lines += [f"**Commit.** `{task['commit']}`", ""]
    if task["blocked-reason"]:
        lines += [
            f"**Blocked.** owner `{task.get('owner') or 'UNASSIGNED — a blocked row needs one'}` "
            f"— {task['blocked-reason']}",
            "",
        ]
    return lines


def main() -> int:
    data = json.loads(LEDGER.read_text(encoding="utf-8"))
    tasks = sorted(data["tasks"], key=sort_key)
    tally = counts(tasks)
    expected = data.get("counts")
    if expected is not None and expected != tally:
        raise SystemExit(f"ledger.json counts are stale: {expected} != rendered {tally}")

    out = [
        f"# {data['audit']}",
        "",
        f"Branch `{data['branch']}` on {data['host']}. This page is generated from "
        "`ledger.json` by `render_ledger.py` in this directory; edit the JSON, not the Markdown.",
        "",
        f"**Open:{tally['open']}  Done:{tally['done']}  Blocked:{tally['blocked']}  "
        f"Total:{tally['total']}**",
        "",
        "## Table",
        "",
        "| ID | Sev | Tier | Project | Location | Title | Category | Status | Host |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    out += [row(task) for task in tasks]
    out += ["", "## Detail", ""]
    for task in tasks:
        out += detail(task)
    RENDERED.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
    print(
        "open:{} done:{} blocked:{} total:{} -> {}".format(
            tally["open"], tally["done"], tally["blocked"], tally["total"], RENDERED
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
