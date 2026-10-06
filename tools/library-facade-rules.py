#!/usr/bin/env python3
"""Rule 1 of AGENTS.md's "Two products, one repository", as a scan.

`sources/TinyTitanLib/` is the surface an embedder depends on, and two of its
promises are absolute rather than ratcheted: it imports no NIO (no HTTP and no
server concept may enter the library), and it writes nothing to stdout, because
stdout belongs to the embedding program -- diagnostics go to stderr through
`ServerLog.diagnostic()`.

Prints one line per violation and exits 0, prints nothing and exits 0 when
clean, exits nonzero when it could not scan at all: a gate that did not run must
not read as a pass.
"""

import os
import re
import sys

ROOT = os.environ.get("ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
LIB = os.path.join(ROOT, "sources", "TinyTitanLib")

IMPORT_NIO = re.compile(r"^\s*import\s+\S*NIO")
WRITES_STDOUT = re.compile(r"(^|[^.\w])(print|printf|puts)\(|FileHandle\.standardOutput|stdout")
COMMENT = ("//", "/*", "*", "*/")


def main() -> int:
    if not os.path.isdir(LIB):
        print(f"cannot scan {LIB}: not a directory", file=sys.stderr)
        return 1
    rows = []
    scanned = 0
    for dirpath, _dirs, names in os.walk(LIB):
        for name in sorted(names):
            if not name.endswith(".swift"):
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, ROOT)
            scanned += 1
            with open(path, encoding="utf-8") as handle:
                for number, line in enumerate(handle, 1):
                    text = line.strip()
                    if text.startswith(COMMENT):
                        continue
                    if IMPORT_NIO.search(line):
                        rows.append(f"NIO    {rel}:{number}: {text}")
                    elif WRITES_STDOUT.search(line):
                        rows.append(f"STDOUT {rel}:{number}: {text}")
    if scanned == 0:
        print(f"cannot scan {LIB}: no Swift files", file=sys.stderr)
        return 1
    for row in rows:
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
