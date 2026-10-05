"""The launcher's expert-cache rule: 30% of physical memory, warned not capped.

The rule belongs to the launcher, and it is a recommendation rather than a
limit: a larger `--ram` is warned about in red and passed on, because the
machine is the operator's. These tests pin the arithmetic (three tenths,
floored to whole GB, which is what `--ram` takes) and both sides of the
boundary.

`TINYTITAN_PHYSICAL_RAM_BYTES` is the launcher's seam for exactly this: the mapping
has to be checkable on a machine of any size.

Run from this directory, like the other benchmark tests:

    cd benchmark && python3 -m unittest test_launcher_ram -v
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "tools/server_launcher.sh"
SERVER = ROOT / ".build/release/TinyTitanServer"
MODELS = ROOT / "models"

# Installed memory in bytes, and the launcher's rule for it: 30%, floored.
MACHINES = {
    8 * 2**30: 2,
    16 * 2**30: 4,
    24 * 2**30: 7,
    32 * 2**30: 9,
    64 * 2**30: 19,
}


def installed_model() -> str | None:
    """The first served id under models/, or None when nothing is installed.

    The launcher's dry run needs a model that is really there — its catalog is
    the only place a served id comes from — and this test has to be runnable on
    a checkout whose models/ is empty.
    """
    if not SERVER.is_file():
        return None
    try:
        listing = subprocess.run(
            [str(SERVER), "--catalog", "--models-dir", str(MODELS)],
            text=True,
            capture_output=True,
            check=True,
            timeout=120,
        ).stdout
        models = json.loads(listing)["models"]
    except Exception:
        return None
    return models[0]["id"] if models else None


def dry_run(*args: str, physical_bytes: int) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["TINYTITAN_PHYSICAL_RAM_BYTES"] = str(physical_bytes)
    return subprocess.run(
        ["bash", str(LAUNCHER), "--client", "server", *args, "--dry-run"],
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )


def answer_ram(model: str, answer: str, *, physical_bytes: int) -> subprocess.CompletedProcess[str]:
    """Answer the interactive RAM question and nothing else.

    Every other question is pre-answered by a flag, so the piped input is the
    RAM choice; without that the answers would be consumed in some other order
    and this would be testing the question order, not the tiers.
    """
    environment = dict(os.environ)
    environment["TINYTITAN_PHYSICAL_RAM_BYTES"] = str(physical_bytes)
    environment["TINYTITAN_LAUNCHER_DRY_RUN"] = "1"
    environment["TINYTITAN_LAUNCHER_ASSUME_TTY"] = "1"
    return subprocess.run(
        [
            "bash",
            str(LAUNCHER),
            "--client",
            "server",
            "--dry-run",
            "--model",
            model,
            "--thinking",
            "off",
            "--answers",
            "default",
            "--engine",
            "gpu",
            "--concurrency",
            "1",
            "--port",
            "8080",
        ],
        input=answer,
        text=True,
        capture_output=True,
        check=False,
        env=environment,
        timeout=180,
    )


def target_of(run: subprocess.CompletedProcess[str]) -> str:
    """The `--ram-budget <n>G` the launcher would pass, or an empty string."""
    match = re.search(r"--ram-budget (\d+)G", run.stdout)
    return match.group(1) if match else ""


class RamMenuTests(unittest.TestCase):
    """The interactive tiers: 4, 6, 8, 10, 12, 14, 16, Custom, model default."""

    # 24 GB of physical memory, so the 30% rule is 7 GB and the Custom default
    # is a number the tier list does not contain -- a value that happens to be
    # a tier could come from either the default or the list.
    MEMORY = 24 * 2**30
    TIERS = {"1": "4", "2": "6", "3": "8", "4": "10", "5": "12", "6": "14", "7": "16"}

    def setUp(self) -> None:
        model = installed_model()
        if model is None:
            self.skipTest("no install under models/ and no built server to list one")
        self.model = model

    def menu(self, answer: str) -> subprocess.CompletedProcess[str]:
        return answer_ram(self.model, answer, physical_bytes=self.MEMORY)

    def test_the_menu_offers_the_seven_tiers_and_a_custom_value(self) -> None:
        run = self.menu("\n")
        menu = run.stdout
        for label in ("4 GB", "6 GB", "8 GB", "10 GB", "12 GB", "14 GB", "16 GB"):
            self.assertIn(label, menu, f"{label} is not offered")
        self.assertIn("Custom", menu)
        self.assertIn("Model default", menu)
        self.assertIn("Choice [1-9] (default 9)", menu)

    def test_each_tier_passes_its_own_budget(self) -> None:
        for choice, gb in self.TIERS.items():
            with self.subTest(choice=choice, gb=gb):
                self.assertEqual(target_of(self.menu(f"{choice}\n")), gb)

    def test_custom_takes_any_whole_number_of_gb(self) -> None:
        # 20 is above every tier and above the 7 GB rule, so it also exercises
        # the warning path rather than a silent accept.
        run = self.menu("8\n20\n")
        self.assertEqual(target_of(run), "20")

    def test_custom_refuses_below_the_floor(self) -> None:
        run = self.menu("8\n3\n")
        self.assertEqual(run.returncode, 2)
        self.assertIn("invalid choice: 3", run.stderr)
        self.assertEqual(target_of(run), "")

    def test_enter_takes_the_model_default_and_passes_no_budget(self) -> None:
        run = self.menu("\n")
        self.assertEqual(target_of(run), "", "the default must not pin a budget")
        self.assertIn("model default", run.stdout)


class RamRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        model = installed_model()
        if model is None:
            self.skipTest("no install under models/ and no built server to list one")
        self.model = model

    def test_rule_is_thirty_percent_rounded_down(self) -> None:
        for memory, expected in MACHINES.items():
            with self.subTest(memory_gb=memory // 2**30):
                run = dry_run("--model", self.model, physical_bytes=memory)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertIn(
                    f"RAM: model default (measured; 30% of this Mac is {expected} GB)",
                    run.stdout,
                )

    def test_at_the_rule_is_silent_and_above_it_warns(self) -> None:
        memory = 24 * 2**30
        at_rule = dry_run("--model", self.model, "--ram", "7", physical_bytes=memory)
        self.assertEqual(at_rule.returncode, 0, at_rule.stderr)
        self.assertNotIn("WARNING", at_rule.stderr)
        self.assertIn("--ram-budget 7G", at_rule.stdout)

        above = dry_run("--model", self.model, "--ram", "8", physical_bytes=memory)
        self.assertEqual(above.returncode, 0, above.stderr)
        self.assertIn("WARNING: the server would hold about 8 GB", above.stderr)
        self.assertIn("30% of this Mac's 24 GB", above.stderr)
        self.assertIn("Starting anyway with 8 GB", above.stderr)
        # Warned, not capped: the requested size is what the server is given.
        self.assertIn("--ram-budget 8G", above.stdout)
        self.assertIn("over 30% of this Mac's RAM", above.stdout)

    def test_default_path_warns_about_nothing(self) -> None:
        run = dry_run("--model", self.model, physical_bytes=24 * 2**30)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertNotIn("WARNING", run.stderr)
        self.assertNotIn("--ram-budget", run.stdout)

    def test_unreadable_memory_means_no_rule(self) -> None:
        # No rule, so nothing to warn about even for a large explicit size…
        run = dry_run("--model", self.model, "--ram", "32", physical_bytes=0)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertNotIn("WARNING", run.stderr)
        self.assertNotIn("30% of this Mac", run.stdout)
        self.assertIn("--ram-budget 32G", run.stdout)
        # …and the default's note claims no percentage either.
        default = dry_run("--model", self.model, physical_bytes=0)
        self.assertIn("RAM: model default (measured) |", default.stdout)

    def test_boundary_scales_with_the_machine(self) -> None:
        # The rule is 2 GB on an 8 GB Mac, so the smallest accepted target (4)
        # is above it there and below it on a 24 GB Mac (rule 7).
        small = dry_run("--model", self.model, "--ram", "4", physical_bytes=8 * 2**30)
        self.assertEqual(small.returncode, 0, small.stderr)
        self.assertIn("WARNING: the server would hold about 4 GB", small.stderr)
        self.assertIn("--ram-budget 4G", small.stdout)

        large = dry_run("--model", self.model, "--ram", "4", physical_bytes=24 * 2**30)
        self.assertEqual(large.returncode, 0, large.stderr)
        self.assertNotIn("WARNING", large.stderr)
        self.assertIn("--ram-budget 4G", large.stdout)

    def test_targets_below_the_floor_are_refused(self) -> None:
        # A streaming install cannot stay under 4 GB: the weights plus the
        # 8-slot minimum cache are ~4.7 GB. The launcher refuses rather than
        # accepting a number the runtime would silently overshoot, and a bare
        # number is refused too instead of being ignored as an unknown word.
        for ram in ("1", "2", "3", "2G"):
            with self.subTest(placement="flag", ram=ram):
                run = dry_run("--model", self.model, "--ram", ram, physical_bytes=24 * 2**30)
                self.assertEqual(run.returncode, 2, run.stdout)
                self.assertIn("unknown RAM target", run.stderr)
                self.assertNotIn("--ram-budget", run.stdout)
        # A bare number is refused too rather than ignored as an unknown word.
        # 1 and 0 are not tested positionally: they are thinking levels ("on",
        # "off") and the optional-value loop reads those first.
        for ram in ("2", "3", "2G"):
            with self.subTest(placement="positional", ram=ram):
                run = dry_run("--model", self.model, ram, physical_bytes=24 * 2**30)
                self.assertEqual(run.returncode, 2, run.stdout)
                self.assertIn("unknown RAM target", run.stderr)
                self.assertNotIn("--ram-budget", run.stdout)


if __name__ == "__main__":
    unittest.main()
