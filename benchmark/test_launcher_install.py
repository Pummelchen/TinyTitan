#!/usr/bin/env python3
"""The launcher fetching a model that is not on disk yet.

Asking for a model this checkout does not have used to end in an error and a
pointer at another tool. It now ends in the model: the launcher asks once, runs
`tools/install_models.sh` with the catalogue's own key, and carries on with the
freshly read catalog. These tests pin the parts that must not drift or surprise:

- the stem -> install-key map, and that every key it returns is a key the
  installer's `CATALOGUE` actually carries (the two lists cannot disagree);
- that a piped or `--dry-run` invocation never downloads, and says the exact
  command instead — a launcher test harness runs every case with `--dry-run`,
  so an install that ignored it would fetch 70 GB during the suite;
- that a model which is present still takes the normal path.

Run from this directory, like the other benchmark tests:

    cd benchmark && python3 -m unittest test_launcher_install -v
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "tools/server_launcher.sh"
MODELS_SH = ROOT / "tools/tinytitan_models.sh"
INSTALLER = ROOT / "tools/install_models.sh"
MODELS = ROOT / "models"

# The runtime stems the launcher can name, and the catalogue key each width
# installs under. Spelled out here on purpose: this is the assertion, so reading
# it from the script under test would assert nothing.
STEM_KEYS = {
    "ornith-1.5_35B_A3B": ("ornith15", "ornith15-8bit"),
    "qwen3.6_35B_A3B": ("qwen36", "qwen36-8bit"),
    "qwen-agentworld_35B_A3B": ("agentworld", "agentworld-8bit"),
    "kat-coder-v2.5_35B_A3B": ("katcoder", "katcoder-8bit"),
    "qwen3.8-flash-next_125B_A6B": ("qwen38flash", "qwen38flash-8bit"),
    "qwen3.5_2B": ("qwen35-2b", "qwen35-2b-8bit"),
    "qwen3.5_4B": ("qwen35-4b", "qwen35-4b-8bit"),
    "qwen3.5_9B": ("qwen35-9b", "qwen35-9b-8bit"),
}

# The names the launcher's own `tinytitan_resolve_model` accepts, with the
# catalogue key each one's 4-bit install uses. A missing model is chosen from
# these, because `--model qwen38flash` is not a name the launcher knows.
LAUNCHER_NAMES = {
    "ornith": "ornith15",
    "qwen36": "qwen36",
    "agentworld": "agentworld",
    "katcoder": "katcoder",
    "qwen38": "qwen38flash",
    "qwen35-2b": "qwen35-2b",
    "qwen35-4b": "qwen35-4b",
    "qwen35-9b": "qwen35-9b",
}


def catalogue() -> dict[str, str]:
    """`install_models.sh`'s CATALOGUE as `key -> install directory`."""
    text = INSTALLER.read_text()
    block = re.search(r"^CATALOGUE=\((.*?)^\)", text, re.MULTILINE | re.DOTALL)
    if block is None:
        raise AssertionError("install_models.sh no longer declares CATALOGUE")
    rows = {}
    for line in block.group(1).splitlines():
        match = re.match(r'\s*"([^|]+)\|([^|]+)\|', line)
        if match:
            rows[match.group(1)] = match.group(2)
    if not rows:
        raise AssertionError("CATALOGUE parsed as empty")
    return rows


def shell_model_function(function: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run one `tinytitan_models.sh` function in bash and hand back the result."""
    script = f'source "{MODELS_SH}"; {function} {args[0]} {args[1]}'
    return subprocess.run(
        ["bash", "-c", script], text=True, capture_output=True, check=False, timeout=60
    )


def run_launcher(
    *args: str, models_dir: pathlib.Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Run the launcher in a dry run, optionally against another `models/`."""
    environment = dict(os.environ)
    environment["TINYTITAN_LAUNCHER_DRY_RUN"] = "1"
    environment.pop("TINYTITAN_LAUNCHER_ASSUME_TTY", None)
    if models_dir is not None:
        environment["TINYTITAN_MODELS_DIR"] = str(models_dir)
    return subprocess.run(
        ["bash", str(LAUNCHER), "--dry-run", *args],
        input="",
        text=True,
        capture_output=True,
        check=False,
        env=environment,
        timeout=120,
    )


def first_missing() -> tuple[str, str, str] | None:
    """A (launcher name, bits, install key) this checkout does not have.

    Used for skipping: on a machine that has everything, there is no
    uninstalled row to test, and a test that pretended otherwise would assert
    against a menu shape that cannot occur.
    """
    known = catalogue()
    for name, key in LAUNCHER_NAMES.items():
        for bits in ("4", "8"):
            catalogue_key = key if bits == "4" else f"{key}-8bit"
            directory = known.get(catalogue_key)
            if directory is None:
                continue
            if not (MODELS / directory).is_dir():
                return name, bits, catalogue_key
    return None


def first_catalogue_model() -> tuple[str, str, str]:
    """A (launcher name, bits, install key) both lists agree on, disk ignored.

    `first_missing` depends on what this machine happens to have installed, so a
    test built on it runs or skips by accident. This one is the same lookup with
    the filesystem taken out of it, for a case that must always run.
    """
    known = catalogue()
    for name, key in LAUNCHER_NAMES.items():
        for bits in ("4", "8"):
            catalogue_key = key if bits == "4" else f"{key}-8bit"
            if catalogue_key in known:
                return name, bits, catalogue_key
    raise AssertionError("no launcher name maps to an installer catalogue key")


class InstallKeyTests(unittest.TestCase):
    def test_every_stem_and_width_maps_to_the_catalogue_key(self) -> None:
        for stem, (four, eight) in STEM_KEYS.items():
            for bits, expected in (("4", four), ("8", eight)):
                with self.subTest(stem=stem, bits=bits):
                    result = shell_model_function("tinytitan_install_key", stem, bits)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.strip(), expected)

    def test_every_mapped_key_is_one_the_installer_carries(self) -> None:
        # The map and the installer are two lists; this is the assertion that
        # keeps a renamed or added model from silently having no fetch path.
        known = catalogue()
        for stem, keys in STEM_KEYS.items():
            for expected in keys:
                with self.subTest(key=expected):
                    self.assertIn(expected, known, f"{stem} maps to a key CATALOGUE lacks")

    def test_an_unknown_stem_has_no_key(self) -> None:
        result = shell_model_function("tinytitan_install_key", "not-a-stem", "4")
        self.assertNotEqual(result.returncode, 0)

    def test_the_install_size_comes_from_the_menu_entry(self) -> None:
        result = subprocess.run(
            [
                "bash",
                "-c",
                f'source "{MODELS_SH}"; tinytitan_install_size_gb katcoder',
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(result.stdout.strip(), r"^\d+(\.\d+)?$")


class MissingModelTests(unittest.TestCase):
    def test_a_pipe_is_told_the_command_and_downloads_nothing(self) -> None:
        candidate = first_missing()
        if candidate is None:
            self.skipTest("every catalogue model is installed here")
        name, bits, key = candidate
        before = sorted(p.name for p in MODELS.iterdir()) if MODELS.is_dir() else []
        run = run_launcher("--client", "server", "--model", name, "--bits", bits)
        self.assertEqual(run.returncode, 1, run.stdout + run.stderr)
        self.assertIn(f"tools/install_models.sh {key}", run.stderr + run.stdout)
        self.assertNotIn("Starting TinyTitanServer", run.stdout)
        after = sorted(p.name for p in MODELS.iterdir()) if MODELS.is_dir() else []
        self.assertEqual(before, after, "a piped launcher installed something")

    def test_install_flag_is_documented_and_refuses_to_start_a_download_in_a_dry_run(self) -> None:
        help_run = subprocess.run(
            ["bash", str(LAUNCHER), "--help"],
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
        self.assertEqual(help_run.returncode, 0)
        self.assertIn("--install", help_run.stdout)

        candidate = first_missing()
        if candidate is None:
            self.skipTest("every catalogue model is installed here")
        name, bits, key = candidate
        before = sorted(p.name for p in MODELS.iterdir()) if MODELS.is_dir() else []
        run = run_launcher("--client", "server", "--model", name, "--bits", bits, "--install")
        self.assertEqual(run.returncode, 1, run.stdout + run.stderr)
        # `--install` answers the question; the dry run still starts nothing,
        # and it says the command it would have run.
        self.assertIn(f"tools/install_models.sh {key}", run.stdout + run.stderr)
        after = sorted(p.name for p in MODELS.iterdir()) if MODELS.is_dir() else []
        self.assertEqual(before, after, "a dry run installed something")


class MissingOffersTests(unittest.TestCase):
    """`tinytitan_missing_offers`: what the menu appends, row by row."""

    def offers(self, models_dir: pathlib.Path) -> list[list[str]]:
        result = subprocess.run(
            [
                "bash",
                "-c",
                f'source "{MODELS_SH}"; tinytitan_missing_offers "{models_dir}"',
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [line.split("|") for line in result.stdout.splitlines() if line]
        for row in rows:
            self.assertEqual(
                len(row),
                7,
                f"offer row is not name|bits|key|label|gb|engines|thinking: {row}",
            )
        return rows

    def test_an_empty_install_directory_offers_every_supported_width(self) -> None:
        with tempfile.TemporaryDirectory(prefix="missing-") as directory:
            rows = self.offers(pathlib.Path(directory))
        names = [row[0] for row in rows]
        self.assertIn("ornith", names)
        self.assertIn("katcoder", names)
        self.assertIn("qwen38", names)
        self.assertEqual(len(rows), 16, "eight models, two widths each")
        for row in rows:
            self.assertIn(row[1], ("4", "8"))
            self.assertTrue(row[3], f"offer row has no label: {row}")
            self.assertRegex(row[4], r"^\d+(\.\d+)?$", f"size missing on {row}")
            # Engine and levels are the columns a person compares rows by, so a
            # row without them would be the one row they cannot judge.
            self.assertIn(row[5], ("gpu", "cpu", "gpu,cpu", "cpu,gpu"), f"bad engines: {row}")
            self.assertTrue(row[6], f"offer row has no thinking levels: {row}")

    def test_every_offered_key_is_one_the_installer_carries(self) -> None:
        # The rows are what a pick is installed with, so a key the installer
        # does not know would be a menu entry that cannot work.
        known = catalogue()
        with tempfile.TemporaryDirectory(prefix="missing-") as directory:
            rows = self.offers(pathlib.Path(directory))
        for row in rows:
            self.assertIn(row[2], known, f"{row[0]}({row[1]}) offers a key CATALOGUE lacks")

    def test_the_widths_that_are_there_are_not_offered(self) -> None:
        with tempfile.TemporaryDirectory(prefix="missing-") as directory:
            root = pathlib.Path(directory)
            (root / "ornith-1.5_35B_A3B_8Bit").mkdir()
            (root / "kat-coder-v2.5_35B_A3B_4Bit").mkdir()
            rows = self.offers(root)
        offered = {(row[0], row[1]) for row in rows}
        self.assertNotIn(("ornith", "8"), offered)
        self.assertNotIn(("katcoder", "4"), offered)
        self.assertIn(("ornith", "4"), offered, "the other width is still missing")

    def test_nothing_missing_offers_nothing(self) -> None:
        with tempfile.TemporaryDirectory(prefix="missing-") as directory:
            root = pathlib.Path(directory)
            for stem in (
                "ornith-1.5_35B_A3B",
                "qwen3.6_35B_A3B",
                "qwen-agentworld_35B_A3B",
                "kat-coder-v2.5_35B_A3B",
                "qwen3.8-flash-next_125B_A6B",
                "qwen3.5_2B",
                "qwen3.5_4B",
                "qwen3.5_9B",
            ):
                for bits in ("4Bit", "8Bit"):
                    (root / f"{stem}_{bits}").mkdir()
            rows = self.offers(root)
        self.assertEqual(rows, [])


class MenuSelectionTests(unittest.TestCase):
    """Picking an uninstalled row from the menu fetches it, then continues."""

    def menu(
        self, answers: str, *, extra: tuple[str, ...] = ()
    ) -> subprocess.CompletedProcess[str]:
        environment = dict(os.environ)
        environment["TINYTITAN_LAUNCHER_DRY_RUN"] = "1"
        environment["TINYTITAN_LAUNCHER_ASSUME_TTY"] = "1"
        return subprocess.run(
            [
                "bash",
                str(LAUNCHER),
                "--client",
                "server",
                "--answers",
                "default",
                "--thinking",
                "off",
                "--ram",
                "9",
                "--engine",
                "gpu",
                "--concurrency",
                "1",
                *extra,
            ],
            input=answers,
            text=True,
            capture_output=True,
            check=False,
            env=environment,
            timeout=120,
        )

    def menu_text(self) -> str:
        return self.menu("\n").stdout

    def first_missing_row(self, text: str) -> tuple[int, str]:
        """The first not-installed row's number and its label."""
        for line in text.splitlines():
            if "not installed" in line and "fetches it first" in line:
                number = int(line.strip().split(")")[0])
                label = line.split(")")[1].strip().split("  ")[0].strip()
                return number, label
        raise AssertionError(f"no uninstalled row in the menu:\n{text}")

    def test_the_menu_offers_the_uninstalled_models_as_rows(self) -> None:
        if first_missing() is None:
            self.skipTest("every catalogue model is installed here")
        text = self.menu_text()
        number, _label = self.first_missing_row(text)
        choice = number
        self.assertIn("fetches it first", text)
        self.assertRegex(text, rf"Rows {choice}-\d+ are not installed yet")

    def test_choosing_one_offers_the_download_and_says_what_it_would_run(self) -> None:
        candidate = first_missing()
        if candidate is None:
            self.skipTest("every catalogue model is installed here")
        number, label = self.first_missing_row(self.menu_text())
        run = self.menu(f"{number}\ny\n")
        self.assertIn(label, run.stdout)
        self.assertIn("Would install:  tools/install_models.sh ", run.stdout)
        self.assertNotIn("Starting TinyTitanServer", run.stdout, "a dry run started a server")

    def test_declining_one_installs_nothing_and_names_the_command(self) -> None:
        if first_missing() is None:
            self.skipTest("every catalogue model is installed here")
        number, _label = self.first_missing_row(self.menu_text())
        before = sorted(p.name for p in MODELS.iterdir()) if MODELS.is_dir() else []
        run = self.menu(f"{number}\nn\n")
        self.assertIn("Nothing was installed", run.stdout)
        self.assertIn("tools/install_models.sh ", run.stdout)
        after = sorted(p.name for p in MODELS.iterdir()) if MODELS.is_dir() else []
        self.assertEqual(before, after, "declining installed something")

    def test_the_default_row_still_launches_an_installed_model(self) -> None:
        # The offers array must not disturb the path the menu took before it
        # existed: Enter takes the default installed row and the run goes on.
        run = self.menu("\n")
        self.assertRegex(run.stdout, r"Choice \[1-\d+\] \(default \d+\)")
        self.assertNotIn("ERROR", run.stdout)


class EmptyModelsDirTests(unittest.TestCase):
    """A checkout with nothing installed reaches the fetch instead of stopping.

    `models/` is gitignored, so this is the state a fresh clone and every CI run
    are in. The launcher's empty-catalog guard used to answer it with `exit 2`
    before the offer and fetch code, which made the feature these other tests pin
    unreachable on exactly the machines that need it. The directory is created
    here rather than inherited, so the case runs identically on a checkout with
    eight installs and one with none.
    """

    def test_a_named_model_is_offered_not_refused(self) -> None:
        name, bits, key = first_catalogue_model()
        with tempfile.TemporaryDirectory() as empty:
            models = pathlib.Path(empty)
            run = run_launcher(
                "--client", "server", "--model", name, "--bits", bits, models_dir=models
            )
            self.assertEqual(run.returncode, 1, run.stdout + run.stderr)
            combined = run.stdout + run.stderr
            self.assertIn(f"tools/install_models.sh {key}", combined)
            # The old refusal must not be what produced that line.
            self.assertNotIn("no install under", combined)
            self.assertNotIn("Starting TinyTitanServer", run.stdout)
            self.assertEqual(sorted(p.name for p in models.iterdir()), [])

    def test_the_menu_draws_the_fetch_rows(self) -> None:
        with tempfile.TemporaryDirectory() as empty:
            run = run_launcher("--client", "server", models_dir=pathlib.Path(empty))
        self.assertEqual(run.returncode, 1, run.stdout + run.stderr)
        self.assertIn("are not installed yet", run.stdout)
        self.assertIn("fetches it first", run.stdout)


class MenuRowPaddingTests(unittest.TestCase):
    """A zero-padded reply names the row its digits spell, not its octal value.

    The pick is guarded with `(( pick >= 1 && pick <= count ))` and indexed with
    `$((pick - 1))`, and bash 3.2 reads a leading zero as octal in both. Measured
    on 2026-10-10 against an empty catalogue, where the menu's 16 rows are the
    catalogue in order, before the fix:

    - `010` picked row 8 -- `katcoder-8bit`, 38.0 GB -- for a person who pointed
      at row 10, `qwen38flash-8bit` at 220.0 GB. Nothing errors.
    - `013` picked row 11 (`qwen35-2b`) for row 13 (`qwen35-4b`), and `016`
      picked row 14 (`qwen35-4b-8bit`) for row 16 (`qwen35-9b-8bit`).
    - `017` and `020` were *accepted* as rows 15 and 16, where the plain replies
      `17` and `20` are refused: padding moved a reply past the end of the list.
    - `08` was refused with `value too great for base` -- the right outcome for
      the wrong reason, and the neighbour `01`-`07` pass unchanged because their
      octal reading equals their decimal one.

    The rows run against a temporary empty `models/` rather than this checkout's,
    so every catalogue row is on the menu in a fixed order on any machine, and the
    run declines the fetch, so nothing is downloaded here.

        cd benchmark && python3 -m unittest test_launcher_install.MenuRowPaddingTests -v
    """

    def pick(self, reply: str, models: pathlib.Path) -> subprocess.CompletedProcess[str]:
        environment = dict(os.environ)
        environment["TINYTITAN_LAUNCHER_DRY_RUN"] = "1"
        environment["TINYTITAN_LAUNCHER_ASSUME_TTY"] = "1"
        environment["TINYTITAN_MODELS_DIR"] = str(models)
        return subprocess.run(
            [
                "bash",
                str(LAUNCHER),
                "--dry-run",
                "--client",
                "server",
                "--answers",
                "default",
                "--thinking",
                "off",
                "--ram",
                "9",
                "--engine",
                "gpu",
                "--concurrency",
                "1",
            ],
            input=f"{reply}\nn\n",
            text=True,
            capture_output=True,
            check=False,
            env=environment,
            timeout=120,
        )

    @staticmethod
    def named_key(run: subprocess.CompletedProcess[str]) -> str | None:
        """The catalogue key the pick resolved to, from the command it prints."""
        match = re.search(r"Install it with:\s+tools/install_models\.sh (\S+)", run.stdout)
        return match.group(1) if match else None

    @staticmethod
    def row_label(menu_output: str, number: str) -> str:
        """The label the menu itself prints for a row number.

        Reading the row back out of the drawn menu is what makes the comparison
        below independent of which models the catalogue happens to carry: the
        claim is that the row you pointed at is the row that got picked.
        """
        pattern = rf"^\s+{number}\)\s+(.+?)\s+(4|8)-bit\b"
        match = re.search(pattern, menu_output, re.MULTILINE)
        if match is None:
            raise AssertionError(f"row {number} is not on the drawn menu")
        return f"{match.group(1)} {match.group(2)}-bit"

    def test_a_padded_reply_names_the_row_it_points_at(self) -> None:
        # Each padded reply must reach the same row as its plain digits, and must
        # not do it by an arithmetic error the guard happens to swallow.
        with tempfile.TemporaryDirectory() as empty:
            models = pathlib.Path(empty)
            drawn = self.pick("1", models)
            for plain, padded in (("8", "08"), ("10", "010"), ("13", "013"), ("16", "016")):
                with self.subTest(row=plain, reply=padded):
                    bare = self.pick(plain, models)
                    pad = self.pick(padded, models)
                    combined = pad.stdout + pad.stderr
                    self.assertNotIn(
                        "value too great for base",
                        combined,
                        f"reply {padded} reached the arithmetic guard as an error",
                    )
                    label = self.row_label(drawn.stdout, plain)
                    self.assertIn(
                        f"{label} is not installed",
                        pad.stdout,
                        f"reply {padded} did not pick the row the menu draws as {label}",
                    )
                    self.assertEqual(self.named_key(bare), self.named_key(pad))

    def test_a_padded_reply_past_the_end_still_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as empty:
            models = pathlib.Path(empty)
            for padded in ("017", "020"):
                with self.subTest(reply=padded):
                    pad = self.pick(padded, models)
                    combined = pad.stdout + pad.stderr
                    self.assertIsNone(self.named_key(pad), f"{padded} reached a row")
                    self.assertIn("invalid choice", combined)
                    self.assertEqual(pad.returncode, 2)


if __name__ == "__main__":
    unittest.main()
