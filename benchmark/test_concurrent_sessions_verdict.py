#!/usr/bin/env python3
"""Gates the verdict of `benchmark/test_concurrent_sessions.py` in its BASE mode.

That suite checks one server with four concurrent sessions and calls an answer that
carries another session's marker a leak. Its docstring promises that "three controls
keep a clean result meaningful, rather than the silence of a detector that never
fires", and the launched-server path (`run_one_model`) gates on exactly that:

    separation_bad = leaked or TOTALS["http_error"] > 0 or not canary_ok or not cancel_ok

The other path does not. `main()`'s `BASE=` branch -- the one the docstring tells an
operator to use against a server they started themselves, which is the supported way
to run this -- returns `0 if TOTALS["foreign"] == 0 and TOTALS["http_error"] == 0`,
dropping the canary and the cancellation controls, and prints no summary and no
controls line either. So a server that answers every request with an empty completion
has no foreign marker in it because it has no text at all, the canary is never caught,
the detector state is never shown, and the run exits 0: the silence of a detector that
never fires, reported as "no leaks" (AUD-252).

These tests drive that branch against a loopback stub -- real HTTP, `http.server` on
127.0.0.1, no model, no server binary, nothing fetched. `echo` answers each request
with its own prompt, so every session sees its own marker and only the canary sees a
second one, which is the shape a clean run has; `blind` answers with empty text, which
is the shape a dead-but-answering backend has.

"""

from __future__ import annotations

import contextlib
import io
import json
import os
import pathlib
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benchmark"))
import test_concurrent_sessions as sessions  # noqa: E402


class _Stub(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _handler(mode: str):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # the stub is not the subject
            pass

        def _send(self, payload: dict):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/v1/models":
                self._send({"data": [{"id": "stub-model"}]})
                return
            self.send_error(404)

        def do_POST(self):
            raw = int(self.headers.get("content-length") or 0)
            request = json.loads(self.rfile.read(raw) or b"{}")
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            text = ""
            if mode == "echo":
                text = request["messages"][-1]["content"]
            self._send(
                {
                    "choices": [
                        {"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
                    ],
                    "usage": {"total_tokens": 1},
                }
            )

    return Handler


@contextlib.contextmanager
def stub_server(mode: str):
    server = _Stub(("127.0.0.1", 0), _handler(mode))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def run_base_mode(mode: str) -> tuple[int, str]:
    """`main()` against the stub, the way an operator points it at their own server."""
    with (
        stub_server(mode) as (base, port),
        mock.patch.object(sessions, "BASE", base),
        mock.patch.object(sessions, "PORT", port),
        mock.patch.object(sessions, "LEDGER", sessions.LEDGER.__class__()),
        mock.patch.dict(os.environ, {"BASE": base}),
        mock.patch.object(sys, "argv", ["test_concurrent_sessions.py"]),
    ):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = sessions.main()
    return status, out.getvalue()


class TestTheBaseModeVerdict(unittest.TestCase):
    """Two of these three fail against HEAD and discriminate against the mutant;
    the echo case passes before the fix too, and is here as the shape the repair must
    keep -- a gate that refused every backend would be as useless as the one it
    replaces -- so it is labelled rather than claimed as a guard.
    """

    def test_a_backend_that_answers_nothing_is_not_a_clean_run(self):
        """The defect: empty completions are 'no leaks' to the dropped predicate."""
        status, output = run_base_mode("blind")
        self.assertNotEqual(status, 0, output)

    def test_the_base_mode_reader_is_told_what_the_detector_saw(self):
        """The same run must not be silent about the controls it just ran."""
        status, output = run_base_mode("blind")
        self.assertIn(
            "canary=", output, "a BASE run prints no summary, so a blind detector is invisible"
        )

    def test_a_server_that_answers_each_session_from_its_own_prompt_passes(self):
        """The fix must refuse the blind backend, not every backend."""
        status, output = run_base_mode("echo")
        self.assertEqual(status, 0, output)


if __name__ == "__main__":
    unittest.main()
