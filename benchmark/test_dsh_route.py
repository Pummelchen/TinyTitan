"""The DeepSeek Harness route writer: one generated block, no hand-written YAML.

`tools/dsh_route.sh` turns the installs under `models/` into the `llm-pi-ai`
route the harness reads, so its model picker follows the catalog instead of a
list someone typed. These tests pin what matters: the ids and effort ladders it
derives, the three switches that are easy to get wrong by hand, the settings-file
surgery (replace the section, keep everything else, back it up), its refusal to
describe nothing, and the fence on an id it reads off a running server.

They run against `tools/testdata/catalog-example.json` through
`TINYTITAN_CATALOG_JSON`, so they need no model, no built server and no network.

Run from this directory, like the other benchmark tests:

    cd benchmark && python3 -m unittest test_dsh_route -v
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/dsh_route.sh"
CATALOG = ROOT / "tools/testdata/catalog-example.json"

try:
    import yaml
except ImportError:  # pragma: no cover - the parse check is a bonus
    yaml = None


def catalog_models() -> list[dict]:
    return json.loads(CATALOG.read_text())["models"]


def run_route(
    *args: str, catalog: pathlib.Path | None = CATALOG, expect: int = 0
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    if catalog is None:
        environment.pop("TINYTITAN_CATALOG_JSON", None)
    else:
        environment["TINYTITAN_CATALOG_JSON"] = str(catalog)
    environment.pop("TINYTITAN_MODELS_DIR", None)
    run = subprocess.run(
        ["bash", str(SCRIPT), *args], text=True, capture_output=True, check=False, env=environment
    )
    if expect is not None:
        if run.returncode != expect:
            raise AssertionError(run.stderr)
    return run


def declared_ids(block: str) -> list[str]:
    return re.findall(r"^        - id: (.+)$", block, re.MULTILINE)


def parsed_levels(model_block: str) -> dict[str, str | None]:
    """The reasoningEfforts map of one model, keyed as the harness reads it.

    Only the keys matter here: a level's *value* is its wire spelling, which the
    block writes as the level's own name (or `on` for a binary family). PyYAML
    is not used for this because it follows YAML 1.1, where a bare `off` key
    parses as the boolean false, while the harness's parser (YAML 1.2) reads the
    string `off`.
    """
    section = re.search(r"reasoningEfforts:\n((?:            .*\n)+)", model_block)
    if section is None:
        raise AssertionError(model_block)
    keys = [line.strip().split(":")[0].strip('"') for line in section.group(1).splitlines()]
    return {key: None for key in keys}


class RouteBlockTests(unittest.TestCase):
    def test_declares_every_model_in_the_catalog(self) -> None:
        run = run_route()
        ids = declared_ids(run.stdout)
        self.assertEqual(ids, [model["id"] for model in catalog_models()])

    def test_effort_ladders_follow_each_template(self) -> None:
        run = run_route("--models", "qwen38")
        blocks = run.stdout.split("        - id: ")[1:]
        self.assertEqual(len(blocks), 2)
        for block in blocks:
            self.assertEqual(sorted(parsed_levels(block)), ["low", "medium", "off", "xhigh"])
            self.assertIn("            xhigh: xhigh", block)
            self.assertNotIn("on:", block)

        # A binary-thinking family has one thinking mode: `off`, plus that mode
        # offered as `medium` with the wire value `on`, because pi-ai's level
        # vocabulary has no `on` of its own.
        binary = run_route("--models", "qwen3.6").stdout
        self.assertIn("            off:\n            medium: on", binary)
        self.assertNotIn("            on:", binary)

    def test_the_three_switches_and_the_usage_contract_are_set(self) -> None:
        block = run_route("--models", "qwen38").stdout
        self.assertIn("      baseURL: http://127.0.0.1:8080/v1", block)
        self.assertIn("        authorization: Bearer tinytitan-local", block)
        self.assertIn("      streamIdleTimeoutMs: 3600000", block)
        self.assertIn("            thinkingFormat: chat-template", block)
        self.assertIn("              enable_thinking: { $var: thinking.enabled }", block)
        self.assertIn("              reasoning_effort: { $var: thinking.effort }", block)
        self.assertIn("            maxTokensField: max_tokens", block)
        self.assertIn("            supportsUsageInStreaming: true", block)

    def test_options_reach_the_block(self) -> None:
        block = run_route(
            "--models",
            "qwen38",
            "--port",
            "8096",
            "--reasoning",
            "off",
            "--context",
            "131072",
            "--max-tokens",
            "4096",
        ).stdout
        self.assertIn("      baseURL: http://127.0.0.1:8096/v1", block)
        self.assertIn("      reasoning: off", block)
        self.assertIn("          contextWindow: 131072", block)
        self.assertIn("          maxTokens: 4096", block)

    def test_filters_match_ids_keys_and_families(self) -> None:
        by_key = declared_ids(run_route("--models", "qwen38").stdout)
        self.assertEqual(by_key, ["qwen3.8-flash-next_4-Bit", "qwen3.8-flash-next_8-Bit"])
        by_id = declared_ids(run_route("--models", "qwen3.5-4b_4-Bit").stdout)
        self.assertEqual(by_id, ["qwen3.5-4b_4-Bit"])
        by_family = declared_ids(run_route("--models", "qwen3_5_dense").stdout)
        self.assertEqual(len(by_family), 4)


class SettingsSurgeryTests(unittest.TestCase):
    def settings(self, body: str) -> pathlib.Path:
        directory = pathlib.Path(tempfile.mkdtemp(prefix="dsh-route-test-"))
        path = directory / "settings.yaml"
        path.write_text(body)
        return path

    def test_write_replaces_the_section_and_keeps_everything_else(self) -> None:
        path = self.settings(
            "ui-theme:\n  preference: dark\nagent-presets:\n  default: qwen38\n"
            "llm-pi-ai:\n  providers:\n    stale-route:\n      displayName: old\n"
        )
        run = run_route("--models", "qwen38", "--write", "--settings", str(path))
        self.assertIn("replaced", run.stdout)
        text = path.read_text()
        self.assertIn("ui-theme:", text)
        self.assertIn("  preference: dark", text)
        self.assertIn("  default: qwen38", text)
        self.assertNotIn("stale-route", text)
        self.assertEqual(
            declared_ids(text), ["qwen3.8-flash-next_4-Bit", "qwen3.8-flash-next_8-Bit"]
        )
        backups = list(path.parent.glob("settings.yaml.bak-*"))
        self.assertEqual(len(backups), 1)
        self.assertIn("stale-route", backups[0].read_text())

        if yaml is not None:
            parsed = yaml.safe_load(text)
            route = parsed["llm-pi-ai"]["providers"]["tinytitan"]
            self.assertEqual(route["api"], "openai-completions")
            self.assertEqual(route["baseURL"], "http://127.0.0.1:8080/v1")
            self.assertEqual(route["reasoning"], "medium")
            self.assertEqual(len(route["models"]), 2)

    def test_a_rewrite_replaces_the_previous_generated_header(self) -> None:
        # The plugin refreshes the route at every harness boot, and the header
        # sits above `llm-pi-ai:` — so a naive section replacement orphans it
        # and each refresh adds three more stale comment lines.
        path = self.settings("ui-theme:\n  preference: dark\n")
        run_route("--models", "qwen38", "--write", "--settings", str(path))
        first = path.read_text()
        run_route("--models", "qwen38", "--write", "--settings", str(path))
        second = path.read_text()
        self.assertEqual(second.count("# Generated by tools/dsh_route.sh"), 1)
        self.assertEqual(second.count("settings.yaml is hot-reloaded."), 1)
        self.assertEqual(first, second)

    def test_write_appends_when_there_is_no_section(self) -> None:
        path = self.settings("ui-theme:\n  preference: dark\n")
        run = run_route("--models", "qwen38", "--write", "--settings", str(path))
        self.assertIn("appended", run.stdout)
        text = path.read_text()
        self.assertTrue(text.startswith("ui-theme:\n  preference: dark\n"))
        self.assertIn("llm-pi-ai:", text)


class RefusalTests(unittest.TestCase):
    def test_bad_arguments_exit_two(self) -> None:
        run_route("--reasoning", "bogus", expect=2)
        run_route("--port", "eighty", expect=2)
        run_route("--models", "no-such-install", expect=2)

    def test_no_catalog_and_no_server_is_an_error(self) -> None:
        run_route("--from-server", "--port", "59999", expect=2)

    def test_write_without_a_settings_file_is_an_error(self) -> None:
        missing = pathlib.Path(tempfile.mkdtemp()) / "absent.yaml"
        run = run_route("--models", "qwen38", "--write", "--settings", str(missing), expect=2)
        self.assertIn("no DSH settings file", run.stderr)


class ServedIdTests(unittest.TestCase):
    """``--from-server`` takes its ids from a running server, and validates them.

    The listing is printed straight into the harness's settings block, so one id
    carrying a newline writes two rows and one carrying YAML punctuation changes what
    the block declares. The catalogue path refuses both shapes in its parser and the
    plugin writer has its own validator; this third source asked neither question.
    A real loopback server is required because the fence sits after ``curl``, in the
    pipeline that turns the answer into rows - stubbing it would test the stub.
    """

    def serve(self, ids: list[str]) -> int:
        body = json.dumps({"object": "list", "data": [{"id": one} for one in ids]}).encode()

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_: object) -> None:
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return int(server.server_address[1])

    def test_plain_served_ids_reach_the_block(self) -> None:
        port = self.serve(["qwen3.5_4B_4Bit", "ornith-1.5_35B_A3B_8Bit"])
        run = run_route("--from-server", "--port", str(port), catalog=None)
        self.assertEqual(declared_ids(run.stdout), ["qwen3.5_4B_4Bit", "ornith-1.5_35B_A3B_8Bit"])

    def test_an_id_with_a_newline_is_refused(self) -> None:
        port = self.serve(["qwen3.5_4B_4Bit\n        - id: injected"])
        run = run_route("--from-server", "--port", str(port), catalog=None, expect=2)
        self.assertIn("not plain tokens", run.stderr)
        self.assertNotIn("- id: injected", run.stdout)

    def test_an_id_with_yaml_punctuation_is_refused(self) -> None:
        port = self.serve(["a: http://evil.example/v1"])
        run = run_route("--from-server", "--port", str(port), catalog=None, expect=2)
        self.assertIn("not plain tokens", run.stderr)

    def test_one_bad_id_refuses_the_whole_listing(self) -> None:
        # Partial acceptance would be the worse outcome: the picker would show the
        # models that passed and silently drop the one the operator launched.
        port = self.serve(["qwen3.5_4B_4Bit", "two\nrows"])
        run = run_route("--from-server", "--port", str(port), catalog=None, expect=2)
        self.assertIn("not plain tokens", run.stderr)
        self.assertEqual(declared_ids(run.stdout), [])


class InstalledLayoutTests(unittest.TestCase):
    """It finds the engine and the models where an *installed* copy keeps them.

    A checkout builds into `.build/release` and models into `models/`; an install
    from the release tarball keeps both one level above the tools (`bin/`,
    `models/`) and normally has `TINYTITAN_BIN_DIR`/`TINYTITAN_MODELS_DIR` set by
    the launcher it wrote. The installer's own advice after a model lands is to
    re-run `tools/dsh_local.sh ensure` from a shell, where neither variable is
    set: before 2026-09-25 that run died with "no server binary at
    …/.build/release/TinyTitanServer", which a user reads as "no model".
    """

    def setUp(self) -> None:
        self.root = pathlib.Path(tempfile.mkdtemp(prefix="tt-installed-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        tools = self.root / "src" / "tools"
        tools.mkdir(parents=True)
        shutil.copytree(ROOT / "tools", tools, dirs_exist_ok=True)
        (self.root / "bin").mkdir()
        (self.root / "models").mkdir()
        self.stub_engine(CATALOG.read_text())

    def stub_engine(self, catalog_body: str) -> None:
        """An engine that records the arguments it was called with."""
        catalog = self.root / "stub-catalog.json"
        catalog.write_text(catalog_body)
        stub = self.root / "bin" / "TinyTitanServer"
        stub.write_text(
            f'#!/bin/sh\nprintf "%s\\n" "$@" > "{self.root}/engine-args.txt"\ncat "{catalog}"\n'
        )
        stub.chmod(0o755)

    def engine_arguments(self) -> list[str]:
        return (self.root / "engine-args.txt").read_text().splitlines()

    def run_installed(self, *args: str) -> subprocess.CompletedProcess[str]:
        # A deliberately bare environment: no TINYTITAN_* variables at all.
        environment = {"PATH": os.environ["PATH"], "HOME": str(self.root)}
        return subprocess.run(
            ["bash", str(self.root / "src" / "tools" / "dsh_route.sh"), *args],
            text=True,
            capture_output=True,
            check=False,
            env=environment,
        )

    def test_it_finds_both_the_engine_and_the_models_directory(self) -> None:
        run = self.run_installed("--models", "qwen38")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertTrue(declared_ids(run.stdout))
        # The engine came from the installed `bin/` and was pointed at the
        # installed `models/`: neither came from an environment variable.
        arguments = self.engine_arguments()
        self.assertIn("--models-dir", arguments)
        self.assertEqual(arguments[arguments.index("--models-dir") + 1], str(self.root / "models"))

    def test_an_empty_catalog_is_not_reported_as_a_missing_binary(self) -> None:
        self.stub_engine('{"models":[]}')
        run = self.run_installed("--models", "qwen38")
        self.assertEqual(run.returncode, 2)
        self.assertIn("no catalog", run.stderr)
        self.assertNotIn("no server binary", run.stderr)


if __name__ == "__main__":
    unittest.main()
