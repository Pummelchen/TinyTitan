"""Synthetic installs: the launcher's choices, checkable with no model on disk.

`tools/testdata/catalog-example.json` is the shipped catalog fixture — twelve
rows, one per family and width, pointing at `/Users/example/TinyTitan/models/…`.
That path is a claim about a machine that does not exist, and the launcher
answers it: `tinytitan_catalog_keep_installed` drops every entry whose install
directory is gone, so feeding the fixture straight to `TINYTITAN_CATALOG_JSON`
leaves the launcher with nothing to present.

This module makes the fixture survive that filter: the same rows, each `path`
rewritten to an empty directory under a temporary `models/`, with a
`manifest.json` in each so a catalog id can be read back out of an install the
way `tinytitan_profile.catalog_id_for` reads it. Nothing here is a model and
nothing here is loaded; a suite that uses it needs neither `models/` nor a built
server. That is the point — without it the launcher's RAM, port, concurrency,
client and profile arithmetic is skipped on any checkout with no install, a clean
clone included, and so is checked nowhere.

    cd benchmark && python3 -m unittest launcher_fixture -v
"""

from __future__ import annotations

import json
import os
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CATALOG = ROOT / "tools/testdata/catalog-example.json"


class SyntheticInstalls:
    """The shipped catalog, materialised as empty directories in a temp tree.

    `create()` and `destroy()` are the pair for a test class (call
    `addClassCleanup(destroy)` right after creating one); `__enter__`/`__exit__`
    are the same pair for a `with` block. The tree has to outlive every launcher
    run it is handed to, because the launcher re-checks it on each one.
    """

    def __init__(self, catalog: pathlib.Path = CATALOG) -> None:
        self._catalog = catalog
        self._tmp: tempfile.TemporaryDirectory | None = None
        self.rows: list[dict] = []

    def create(self) -> "SyntheticInstalls":
        self._tmp = tempfile.TemporaryDirectory(prefix="tinytitan-launcher-fixture-")
        self.models_dir = pathlib.Path(self._tmp.name) / "models"
        self.models_dir.mkdir()
        for row in json.loads(self._catalog.read_text())["models"]:
            directory = self.models_dir / pathlib.Path(row["path"]).name
            directory.mkdir()
            # The catalog id is `<modelID>_<bits>-Bit`, so a manifest written
            # from the row's own fields reads back as that id: the round trip
            # install directory -> catalog id is assertable here rather than
            # only against a real install.
            manifest = {
                "modelID": row["id"].rsplit("_", 1)[0],
                "quant": {"routedExpert": {"weightBits": row["quant"]}},
            }
            (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            installed = dict(row)
            installed["path"] = str(directory)
            self.rows.append(installed)
        self.catalog_path = self.models_dir.parent / "catalog.json"
        self.catalog_path.write_text(json.dumps({"models": self.rows}, indent=2) + "\n")
        return self

    def destroy(self) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None

    def __enter__(self) -> "SyntheticInstalls":
        return self.create()

    def __exit__(self, *exc: object) -> None:
        self.destroy()

    @property
    def ids(self) -> list[str]:
        return [row["id"] for row in self.rows]

    def path_of(self, model_id: str) -> pathlib.Path:
        for row in self.rows:
            if row["id"] == model_id:
                return pathlib.Path(row["path"])
        raise KeyError(f"{model_id} is not a row of {CATALOG}")

    def first_gpu(self) -> str:
        """The first gpu-backed id, which is the one the launcher lists first."""
        for row in self.rows:
            if row["backend"] == "gpu":
                return row["id"]
        raise AssertionError(f"{CATALOG} lists no gpu-backed model")

    def dense(self) -> str:
        """A dense 4-bit id: the width the CPU engine serves as well as the GPU.

        The concurrency tests need one model they can name to either engine, and
        a dense row is the only kind that qualifies.
        """
        for row in self.rows:
            if "dense" in row["family"] and row["quant"] == 4:
                return row["id"]
        raise AssertionError(f"{CATALOG} lists no dense 4-bit model")

    def env(self, **extra: str) -> dict[str, str]:
        """A complete environment that points the launcher at this tree.

        It sets both variables the launcher reads, so a caller cannot leave a
        real `models/` in play by naming only one of them.
        """
        environment = dict(os.environ)
        environment["TINYTITAN_CATALOG_JSON"] = str(self.catalog_path)
        environment["TINYTITAN_MODELS_DIR"] = str(self.models_dir)
        environment.update(extra)
        return environment


def install_fixture(case: type) -> None:
    """Give a test class its fixture tree, removed when the class is done.

    Call it from `setUpClass`: the tree has to outlive every launcher run in the
    class (each run re-checks the directories it points at), and
    `addClassCleanup` is what makes it go away even when a test inside the class
    fails.
    """
    case.installs = SyntheticInstalls().create()
    case.addClassCleanup(case.installs.destroy)


class SyntheticInstallsTests(unittest.TestCase):
    """The fixture's own invariants: it stands in for an install, or it is noise.

    These fail if the shipped catalog changes shape or the launcher's filter
    starts checking something this tree does not provide — which is exactly the
    moment the four suites that depend on it would start testing nothing.
    """

    def setUp(self) -> None:
        self.installs = SyntheticInstalls().create()
        self.addCleanup(self.installs.destroy)

    def test_every_row_points_at_a_directory_that_exists(self) -> None:
        self.assertGreater(len(self.installs.rows), 1)
        for row in self.installs.rows:
            with self.subTest(model=row["id"]):
                self.assertTrue(pathlib.Path(row["path"]).is_dir(), row["path"])

    def test_a_manifest_reads_back_as_its_own_catalog_id(self) -> None:
        from tinytitan_profile import catalog_id_for

        for row in self.installs.rows:
            with self.subTest(model=row["id"]):
                self.assertEqual(catalog_id_for(row["path"]), row["id"])

    def test_the_environment_names_the_fixture_and_nothing_else(self) -> None:
        environment = self.installs.env(TINYTITAN_PORT="9999")
        self.assertEqual(environment["TINYTITAN_CATALOG_JSON"], str(self.installs.catalog_path))
        self.assertEqual(environment["TINYTITAN_MODELS_DIR"], str(self.installs.models_dir))
        self.assertEqual(environment["TINYTITAN_PORT"], "9999")
        self.assertNotEqual(self.installs.models_dir, ROOT / "models")

    def test_the_selectors_the_suites_ask_for_exist(self) -> None:
        # A suite that silently lost its row would test a different model than it
        # says it does, so both selectors are pinned to the shipped fixture.
        gpu = self.installs.first_gpu()
        dense = self.installs.dense()
        self.assertIn(gpu, self.installs.ids)
        self.assertIn(dense, self.installs.ids)
        self.assertEqual(pathlib.Path(self.installs.path_of(dense)).name, "qwen3.5_2B_4Bit")
        with self.assertRaises(KeyError):
            self.installs.path_of("no-such-model")


if __name__ == "__main__":
    unittest.main()
