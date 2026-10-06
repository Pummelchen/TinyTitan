#!/usr/bin/env python3
"""AUD-160 mutation driver for `tools/lint.sh docs`.

Each arm edits one tracked file in place, runs the gate, asserts the exact verdict,
and restores. The last arm proves the restore: byte-identical files and the gate
green again. `AGENTS.md` is deliberately never snapshotted, mutated or restored --
it is the maintainer's file and is edited outside this session, so writing it back
from a stale copy could clobber a live edit; the arm that shows it is reported and
not enforced is the baseline run, which prints its OWNER note and still exits 0.

Run it with `python3 docs/audit-2026-10-06/AUD-160-docs-probe.py` from anywhere;
it needs a clean checkout of this repository two directories up.
"""

import hashlib
import os
import subprocess
import sys

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
)
FILES = [
    "CONTRIBUTING.md",
    "RELEASE.md",
    "docs/handover-tinytitan.md",
    "tools/lint.sh",
    "tools/docs-facts.py",
]


def run(env=None):
    proc = subprocess.run(
        [sys.executable, "tools/docs-facts.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    return proc.returncode, proc.stdout + proc.stderr


def read(rel):
    with open(f"{ROOT}/{rel}", encoding="utf-8") as handle:
        return handle.read()


def write(rel, text):
    with open(f"{ROOT}/{rel}", "w", encoding="utf-8") as handle:
        handle.write(text)


SNAP = {f: read(f) for f in FILES}
failures = []


def arm(name, rel, old, new, expect_rc, needles, absent=()):
    text = SNAP[rel]
    if old not in text:
        failures.append(f"{name}: anchor not found in {rel}")
        print(f"FAIL {name}: anchor missing")
        return
    write(rel, text.replace(old, new, 1))
    try:
        rc, out = run()
    finally:
        write(rel, text)
    ok = rc == expect_rc and all(n in out for n in needles)
    ok = ok and not any(a in out for a in absent)
    print(f"{'ok  ' if ok else 'FAIL'} {name}: rc={rc}")
    if not ok:
        for line in out.splitlines():
            print(f"     | {line}")
        failures.append(f"{name}: rc={rc} want {expect_rc} out={out[:400]}")


# M0 — the baseline is green with the owner note printed and not enforced.
rc, out = run()
ok = rc == 0 and "OWNER AGENTS.md" in out and "FAIL" not in out
print(f"{'ok  ' if ok else 'FAIL'} M0 baseline green, owner note printed")
if not ok:
    print(out)
    failures.append("M0")

# M1 — a documented count one off the derived number.
arm(
    "M1 wrong count",
    "CONTRIBUTING.md",
    "enforces seventeen checks",
    "enforces sixteen checks",
    1,
    ["CONTRIBUTING.md:", "says “sixteen checks”", "runs 17"],
)

# M2 — a gate name the script does not answer to, inside a list of real ones.
arm(
    "M2 stale mode name",
    "CONTRIBUTING.md",
    "## Pull requests",
    "Run the modes `force-cast`, `func-length`, `library-facade` and "
    "`audit-coverage`.\n\n## Pull requests",
    1,
    ["names “audit-coverage”"],
)

# M3 — a restated ledger count that disagrees with ledger.json.
arm(
    "M3 wrong ledger counts",
    "docs/handover-tinytitan.md",
    "**61 rows / 57 closed / 3 open / 1 blocked**",
    "**60 rows / 57 closed / 3 open / 1 blocked**",
    1,
    ["says “60 rows / 57 closed / 3 open / 1 blocked”"],
)

# M4 — a tag citation that names the tag object instead of the tagged commit.
arm(
    "M4 tag-object sha",
    "docs/handover-tinytitan.md",
    "## What has landed",
    "The release `v5.18` -> `d30de4433811d2f691e693aed8b8a28ecc23b4b5` is the "
    "one to watch.\n\n## What has landed",
    1,
    ["cites `v5.18`", "that is the *tag object*"],
)

# M5 — a tag citation that is simply not the commit the tag points at.
arm(
    "M5 wrong sha",
    "docs/handover-tinytitan.md",
    "## What has landed",
    "The release `v5.18` -> `1111111111111111111111111111111111111111` is it.\n"
    "\n## What has landed",
    1,
    ["cites `v5.18`", "points at commit `ea5de8c"],
    absent=["tag object"],
)

# M6 — a mode name in a doc that is not a mode; and the script disagreeing
# with itself: a usage header label with no case arm behind it.
arm(
    "M6 header advertises a dead mode",
    "tools/lint.sh",
    "#   docs                a documented count",
    "#   docs-audit          a documented count",
    1,
    ["usage header advertises `docs-audit`"],
)

# M7 — the all chain runs a check no mode invokes.
arm(
    "M7 chain without an arm",
    "tools/lint.sh",
    "check_library_facade; ",
    "check_library_facade; check_docs_typo; ",
    1,
    ["no mode invokes it"],
)

# M8 — the unknown-check message omits a real mode: --help would lie.
arm(
    "M8 usage string omits a mode",
    "tools/lint.sh",
    "library-facade|docs|",
    "library-facade|",
    1,
    ["unknown-check message omits docs"],
)

# M9 — a history section stays wrong on purpose: must NOT fail.
arm(
    "M9 history exempt",
    "docs/handover-tinytitan.md",
    "## What has landed",
    "## What has landed\n\ntools/lint.sh ran fifteen checks, CI ran thirteen gates there.",
    0,
    ["ok ("],
)

# M10 — a run record is a dated fact, not a live claim: must NOT fail.
arm(
    "M10 run record exempt",
    "CONTRIBUTING.md",
    "## Pull requests",
    "Verification: tools/lint.sh ran sixteen gates, all clean.\n\n## Pull requests",
    0,
    ["ok ("],
)

# M11 — a sentence about something else entirely is not a count claim.
arm(
    "M11 unanchored number exempt",
    "CONTRIBUTING.md",
    "## Pull requests",
    "Keep each pull request narrow: nine checks were written by hand.\n\n## Pull requests",
    0,
    ["ok ("],
)

# M12 — a quoted former value is a mention, not a claim (RELEASE.md carries one).
arm(
    "M13 quote exempt",
    "RELEASE.md",
    'used to keep a second copy with a count ("fifteen checks as of 2026-10-07")',
    'used to keep a second copy with a count ("eleven checks as of 2026-10-07")',
    0,
    ["ok ("],
)

# M14 — the owner file is reported and never enforced: proven by M0, which sees
# "thirteen checks" in AGENTS.md, prints OWNER and still exits 0. AGENTS.md is
# deliberately not mutated or restored here -- it is the maintainer's file and is
# being edited outside this session, so snapshotting it could clobber a live edit.

# M13 — no repository to compare against must fail loudly, not pass vacuously.
proc = subprocess.run(
    [sys.executable, f"{ROOT}/tools/docs-facts.py"],
    cwd="/tmp",
    capture_output=True,
    text=True,
    env={"ROOT": "/tmp", "PATH": "/usr/bin:/bin"},
    check=False,
)
out = proc.stdout + proc.stderr
ok = proc.returncode == 1 and ("no tracked documents" in out or "cannot list tracked" in out)
print(f"{'ok  ' if ok else 'FAIL'} M13 empty root fails loudly: rc={proc.returncode}")
if not ok:
    print(out)
    failures.append("M13 empty root")

# M14 — missing lint.sh must fail, not report zero gates.
proc = subprocess.run(
    [sys.executable, f"{ROOT}/tools/docs-facts.py"],
    cwd="/tmp",
    capture_output=True,
    text=True,
    env={"ROOT": "/tmp/definitely-not-a-repo", "PATH": "/usr/bin:/bin"},
    check=False,
)
out = proc.stdout + proc.stderr
ok = proc.returncode == 1 and "cannot read" in out and "lint.sh" in out
print(f"{'ok  ' if ok else 'FAIL'} M14 missing lint.sh fails: rc={proc.returncode}")
if not ok:
    print(out)
    failures.append("M14 missing lint.sh")

# Restore integrity + green again.
status = subprocess.run(
    ["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True, check=False
).stdout
drift = [
    f
    for f in FILES
    if hashlib.sha256(read(f).encode()).hexdigest() != hashlib.sha256(SNAP[f].encode()).hexdigest()
]
rc, out = run()
ok = not drift and rc == 0
print(f"{'ok  ' if ok else 'FAIL'} M15 restored byte-identical and green: rc={rc}")
if drift:
    print(f"     drifted: {drift}")
    failures.append("M15 drift")
print("git status:\n" + status)
print(f"\n{16 - len(failures)}/16 arms ok" if not failures else f"\nFAILURES: {failures}")
sys.exit(1 if failures else 0)
