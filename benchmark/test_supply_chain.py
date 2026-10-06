"""Every dependency surface in this repository must be watched by an advisory feed.

AUD-120: pip-audit covers `benchmark/requirements.txt` and `npm audit` covers both
plugin packages, but Swift had no scanner and no `.github/dependabot.yml`, so a
published CVE against `swift-nio` or `swift-transformers` would have gone
unnoticed however long the pin held. The feed for SwiftPM does exist (the GitHub
dependency graph reads `Package.resolved`), so the gap was the config file.

This pins the class rather than the one row: the ecosystems are derived from the
dependency manifests actually tracked, so adding a package directory, a
requirements file or a toolchain without naming it in the config fails here.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG = ROOT / ".github/dependabot.yml"


def tracked(*suffixes: str) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", *suffixes],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return sorted(result.stdout.split())


def update_blocks(text: str) -> list[dict[str, str]]:
    """Each `- package-ecosystem:` entry, its keys, and its own text.

    Parsed line by line rather than with a YAML library because this suite runs on
    the system `python3`, where PyYAML is not one of the pins in
    `benchmark/requirements.txt` — a test that dies on a missing import is a test
    that did not run.

    The block keeps its raw text because a key asserted against the whole file is
    a key asserted against whichever entry happens to carry it.
    """
    blocks: list[dict[str, str]] = []
    for chunk in re.split(r"^\s*- package-ecosystem:", text, flags=re.M)[1:]:
        entry = {"raw": chunk}
        for line in chunk.splitlines():
            stripped = line.strip()
            if stripped.startswith("- "):
                break
            if stripped.startswith("directory:"):
                entry["directory"] = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("schedule:"):
                entry["schedule"] = "yes"
            elif stripped.startswith("open-pull-requests-limit:"):
                entry["limit"] = stripped.split(":", 1)[1].strip()
        first = chunk.splitlines()[0].strip() if chunk.strip() else ""
        entry["ecosystem"] = first.strip("'\"")
        blocks.append(entry)
    return blocks


class AdvisoryCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = CONFIG.read_text()
        cls.blocks = update_blocks(cls.text)
        cls.entries = {(b["ecosystem"], b.get("directory", "")) for b in cls.blocks}

    def test_the_config_exists_and_every_entry_is_scheduled(self) -> None:
        self.assertTrue(CONFIG.is_file(), f"{CONFIG} is absent: no advisory feed at all")
        self.assertGreaterEqual(len(self.blocks), 4, "expected an entry per dependency surface")
        for block in self.blocks:
            with self.subTest(ecosystem=block["ecosystem"], directory=block.get("directory")):
                self.assertIn("schedule", block, "an entry with no schedule never runs")

    def test_every_dependency_surface_present_is_declared(self) -> None:
        # The repository's own manifests decide the list, not a literal kept in
        # this test: a new one has to be named in the config or it fails here.
        if (ROOT / "Package.resolved").is_file():
            self.assertIn(("swift", "/"), self.entries, "Package.resolved has no advisory feed")
        for lock in tracked("plugins/*/package-lock.json", "*/package-lock.json"):
            directory = "/" + str(pathlib.Path(lock).parent)
            self.assertIn(
                ("npm", directory),
                self.entries,
                f"{lock} has no advisory feed; add an npm entry for {directory}",
            )
        for req in tracked("requirements.txt", "*/requirements.txt"):
            parent = str(pathlib.Path(req).parent)
            directory = "/" if parent == "." else f"/{parent}"
            self.assertIn(
                ("pip", directory),
                self.entries,
                f"{req} has no advisory feed; add a pip entry for {directory}",
            )
        if tracked(".github/workflows/*.yml"):
            self.assertIn(
                ("github-actions", "/"),
                self.entries,
                "the pinned actions have no advisory feed",
            )

    def test_no_entry_claims_coverage_for_a_directory_that_is_not_there(self) -> None:
        # A stale entry reads as coverage and gives none: the graph cannot resolve
        # a path that has been renamed away.
        for block in self.blocks:
            directory = block.get("directory", "/")
            with self.subTest(ecosystem=block["ecosystem"], directory=directory):
                self.assertTrue(
                    (ROOT / directory.lstrip("/")).is_dir(),
                    f"{directory} does not exist, so nothing is being watched there",
                )

    def test_swift_and_package_entries_do_not_open_bump_pull_requests(self) -> None:
        # Version updates are security-only by choice: every gate in this tree is
        # measured on the pin it names, so an unreviewed bump is a numerics change
        # waiting to happen. Actions are the exception — a bump there is a
        # supply-chain event that should arrive as a PR rather than as drift.
        watched = [b for b in self.blocks if b["ecosystem"] in {"swift", "npm", "pip"}]
        self.assertEqual(len(watched), 4, "expected the swift, two npm and pip entries")
        for block in watched:
            with self.subTest(ecosystem=block["ecosystem"], directory=block.get("directory")):
                self.assertEqual(
                    block.get("limit"),
                    "0",
                    "a pinned ecosystem may open bump PRs against the gate it is measured on",
                )
        actions = [b for b in self.blocks if b["ecosystem"] == "github-actions"]
        self.assertEqual(len(actions), 1)
        self.assertNotEqual(
            actions[0].get("limit"),
            "0",
            "actions bumps must arrive as PRs, not as silent re-pins",
        )


if __name__ == "__main__":
    unittest.main()
