"""The launcher and the coder harness have to name the same clients.

The list lives once, in `tools/tinytitan_models.sh` (`TINYTITAN_CLIENTS`). These tests
pin both consumers to it: the launcher's help, its menu and its `--client`
validation, and this harness's `--clients` choices, binary search and command
builders. A client added to the catalogue without a command builder — or a coder
client the harness can run and the launcher cannot start — fails here instead of
during a benchmark run.

The model they are asked to start comes from `launcher_fixture`, so these run on a
checkout with no install and no built server.

Run from this directory, like the other benchmark tests:

    cd benchmark && python3 -m unittest test_coder_clients -v
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import unittest

import coder_cli_benchmark as harness
import launcher_fixture

ROOT = pathlib.Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "tools/server_launcher.sh"


def run_launcher(
    installs: launcher_fixture.SyntheticInstalls, *args: str, stdin: str | None = None
) -> subprocess.CompletedProcess[str]:
    environment = installs.env()
    if stdin is not None:
        environment["TINYTITAN_LAUNCHER_ASSUME_TTY"] = "1"
    return subprocess.run(
        ["bash", str(LAUNCHER), *args],
        input=stdin,
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )


def run_harness(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "benchmark/coder_cli_benchmark.py"), *args],
        text=True,
        capture_output=True,
        check=False,
    )


class ClientCatalogueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        launcher_fixture.install_fixture(cls)

    def test_entries_are_well_formed(self) -> None:
        self.assertTrue(harness.CLIENTS)
        for client in harness.CLIENTS:
            with self.subTest(client=client["id"]):
                self.assertTrue(client["id"])
                self.assertTrue(client["label"])
                self.assertIn(client["kind"], ("coder", "editor"))
                self.assertTrue(client["binaries"])
        ids = [client["id"] for client in harness.CLIENTS]
        self.assertEqual(len(ids), len(set(ids)), "duplicate client id")
        self.assertEqual(set(harness.CODER_CLIENTS) | set(harness.EDITOR_CLIENTS), set(ids))

    def test_launcher_help_lists_exactly_the_catalogue(self) -> None:
        run = run_launcher(self.installs, "--help")
        self.assertEqual(run.returncode, 0, run.stderr)
        line = next(line for line in run.stdout.splitlines() if "--client" in line)
        self.assertEqual(
            line.split()[-1].removeprefix("server|").split("|"),
            [client["id"] for client in harness.CLIENTS],
        )

    def test_launcher_menu_numbers_every_client(self) -> None:
        run = run_launcher(self.installs, "--dry-run", stdin="\n")
        self.assertIn("What do you want to launch?", run.stdout)
        for index, client in enumerate(harness.CLIENTS, start=2):
            with self.subTest(client=client["id"]):
                self.assertIn(f"  {index}) {client['label']}", run.stdout)

    def test_unknown_client_is_refused_with_the_catalogue(self) -> None:
        run = run_launcher(self.installs, "--client", "not-a-client", "--dry-run")
        self.assertEqual(run.returncode, 2)
        self.assertIn("unknown client: not-a-client", run.stderr)
        for client in harness.CLIENTS:
            self.assertIn(client["id"], run.stderr)

    def test_launcher_accepts_every_client(self) -> None:
        model = self.installs.first_gpu()
        for client in harness.CLIENTS:
            with self.subTest(client=client["id"]):
                run = run_launcher(
                    self.installs, "--client", client["id"], "--model", model, "--dry-run"
                )
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertIn(f"Client:     {client['label']}", run.stdout)

    def test_every_coder_client_has_a_command_builder(self) -> None:
        source = (ROOT / "benchmark/coder_cli_benchmark.py").read_text()
        for client in harness.CODER_CLIENTS:
            with self.subTest(client=client):
                self.assertIn(
                    f'client == "{client}"',
                    source,
                    f"{client} is a coder client with no run_client branch",
                )

    def test_editor_client_is_refused_as_a_benchmark_client(self) -> None:
        for client in harness.EDITOR_CLIENTS:
            with self.subTest(client=client):
                run = run_harness("--round", "coder", "--clients", client)
                self.assertNotEqual(run.returncode, 0)
                self.assertIn("editor client", run.stderr)
                self.assertIn("--round clients", run.stderr)

    def test_dense_family_cannot_run_the_features_round(self) -> None:
        dense = sorted(harness.DENSE_FAMILIES)[0]
        run = run_harness("--round", "features", "--model", dense)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("dense install", run.stderr)

    def test_binaries_cover_every_catalogue_entry(self) -> None:
        found = harness.client_binaries()
        self.assertEqual(set(found), {client["id"] for client in harness.CLIENTS})


if __name__ == "__main__":
    unittest.main()
