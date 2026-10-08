#!/usr/bin/env python3
"""The lifetime of the server the launcher starts.

`tools/server_launcher.sh` installs `trap cleanup EXIT INT TERM` at the moment it puts
the server in the background, so the model it loaded dies with the script — including
when the script ends because the *client* exited. That is the lifetime this project's
own precondition requires: an orphaned model process makes the next gate or golden run
refuse to start. It is also what the launcher cannot currently be shown to do, because
the only sentence written down about the client path says the model keeps running.

All three are pinned here: where the trap sits, what the prose at the client path
promises, and the real thing — a stub server that must be gone when the launcher
returns.

Nothing here loads a model. `TINYTITAN_BIN_DIR` is this project's own seam for the
server binary (the release-tarball layout sets it), so the launcher starts a python
stub that answers `/v1/models` and records its pid, and `CODEX` is the client override
`client_bin` reads before PATH.

    cd benchmark && python3 -m unittest test_launcher_lifetime -v
"""

from __future__ import annotations

import os
import pathlib
import re
import signal
import socket
import subprocess
import tempfile
import textwrap
import unittest

import launcher_fixture

ROOT = pathlib.Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "tools/server_launcher.sh"

SERVER_STUB = """#!/usr/bin/env python3
import http.server, json, os, sys

args = sys.argv[1:]
port = int(args[args.index("--port") + 1]) if "--port" in args else 8080
with open(os.environ["STUB_PIDFILE"], "w") as handle:
    handle.write(str(os.getpid()))


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps(
            {"object": "list", "data": [{"id": "stub-model", "object": "model"}]}
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


http.server.HTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""

CLIENT_STUB = """#!/bin/bash
echo "fake client ran"
exit 0
"""


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class LauncherLifetimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text(encoding="utf-8")
        lines = cls.source.splitlines()
        # Four `case "$CLIENT"` blocks exist; the dispatch is the one that runs a
        # client binary, so anchor on its first arm rather than on the keyword.
        codex_arm = next(
            i for i, line in enumerate(lines) if line.startswith('  codex)    "$BIN" ;;')
        )
        cls.case_at = max(
            i for i, line in enumerate(lines[:codex_arm]) if line.startswith('case "$CLIENT" in')
        )
        cls.case_offset = len("\n".join(lines[: cls.case_at])) + 1
        comment: list[str] = []
        for line in reversed(lines[: cls.case_at]):
            stripped = line.strip()
            if not stripped:
                continue
            if not stripped.startswith("#"):
                break
            comment.append(stripped)
        cls.client_comment = "\n".join(reversed(comment))

    def test_the_trap_is_installed_before_the_client_runs(self) -> None:
        """The trap is what ends the model, so its position is the contract.

        Placed with the server rather than only on the client path, it also covers the
        harness-signalled exit that orphaned a model before it existed.
        """
        trap = self.source.find("trap cleanup EXIT INT TERM")
        self.assertNotEqual(trap, -1, "no EXIT/INT/TERM cleanup trap in the launcher")
        self.assertLess(
            trap,
            self.case_offset,
            "the cleanup trap is installed after the client runs, so a client that "
            "exits normally leaves the server behind",
        )

    def test_the_client_path_describes_the_lifetime_the_code_gives(self) -> None:
        """The comment above the client dispatch is the only written promise about
        when the model stops, so it has to match the trap above it."""
        block = self.client_comment
        self.assertIn("client", block.lower(), f"no client-path comment found: {block}")
        stale = re.search(
            r"keeps? running after the client|stays up after the client",
            block,
            re.IGNORECASE,
        )
        self.assertIsNone(
            stale,
            textwrap.dedent(
                f"""\
                {LAUNCHER.name} promises a lifetime the trap above it does not give:
                {stale.group(0) if stale else ""}
                The launcher's EXIT trap kills the server it started, so the model is
                gone when the client exits. Say that, or change the trap — not both.
                """
            ),
        )
        self.assertRegex(
            block,
            r"stops?|kills?|does not outlive",
            f"the comment at the client path never says what happens to the server: {block}",
        )


class LauncherServerProcessTests(unittest.TestCase):
    """The same contract, measured by running the launcher."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="tinytitan-launcher-lifetime-")
        self.addCleanup(tmp.cleanup)
        self.work = pathlib.Path(tmp.name)
        self.bin = self.work / "bin"
        self.bin.mkdir(exist_ok=True)
        self.server = self.bin / "TinyTitanServer"
        self.server.write_text(SERVER_STUB, encoding="utf-8")
        self.server.chmod(0o755)
        self.client = self.work / "fake-client"
        self.client.write_text(CLIENT_STUB, encoding="utf-8")
        self.client.chmod(0o755)
        self.pidfile = self.work / "server.pid"
        self.installs = launcher_fixture.SyntheticInstalls().create()
        self.addCleanup(self.installs.destroy)

    def tearDown(self) -> None:
        if self.pidfile.exists():
            pid = int(self.pidfile.read_text())
            if alive(pid):
                os.kill(pid, signal.SIGTERM)

    def test_the_server_is_gone_when_the_launcher_exits_after_the_client(self) -> None:
        port = str(free_port())
        env = self.installs.env(
            TINYTITAN_BIN_DIR=str(self.bin),
            STUB_PIDFILE=str(self.pidfile),
            CODEX=str(self.client),
            TINYTITAN_PORT=port,
        )
        env.pop("TINYTITAN_LAUNCHER_DRY_RUN", None)
        run = subprocess.run(
            [
                "bash",
                str(LAUNCHER),
                "--answers",
                "default",
                "--thinking",
                "off",
                "--ram",
                "4",
                "--engine",
                "gpu",
                "--concurrency",
                "1",
                "--port",
                port,
                "--client",
                "codex",
            ],
            input="2\n",
            text=True,
            capture_output=True,
            env=env,
            timeout=180,
            check=False,
        )
        self.assertEqual(
            run.returncode,
            0,
            f"launcher failed:\nstdout\n{run.stdout[-800:]}\nstderr\n{run.stderr[-800:]}",
        )
        self.assertIn("fake client ran", run.stdout, "the client never ran")
        self.assertTrue(
            self.pidfile.exists(),
            "the launcher started no server: it did not honour TINYTITAN_BIN_DIR",
        )
        pid = int(self.pidfile.read_text())
        self.assertFalse(
            alive(pid),
            f"the server the launcher started (pid {pid}) outlived it — the EXIT trap "
            "is not what this file says it is",
        )


if __name__ == "__main__":
    unittest.main()
