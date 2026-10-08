"""Gates what the memory drivers *tell* their operator against what they do.

Two defects found by reading a driver's own documentation after AUD-221, both of
which send a person to do the wrong thing with a live model run:

`benchmark/memory_book.py` documents "Three arms" and lists `summary`, `minimal`
and `full`, while its `ARMS` is `("summary", "auto", "minimal", "full")` — the
`auto` arm (memory on with no tools, the bootstrap fragment alone) exists in the
code, has prompt-token floors written for it at `:235`, and is named nowhere a
human would read. An operator comparing configurations from the docstring cannot
run the one arm that separates the memory fragment's cost from the tools' cost.

`benchmark/memory_value.py` carries `compiles()` and `code_block()`, which nothing
in the tree calls: the only matches for either name across `benchmark/`, `tools/`,
`tests/` and `docs/` are the two `def` lines. `compiles()` answers `False` from a
bare `except Exception`, and what actually raises there is the environment — a
`swiftc` that is not on `PATH` (`FileNotFoundError`), a result directory that
cannot be written (`OSError`), a typecheck slower than 180 s (`TimeoutExpired`) —
so the day anyone wires that column up, an uninstalled toolchain scores every
Swift arm as code the model wrote badly. Its docstring had also documented
`memory_value.py memory`, a command that has never existed, which is how AUD-221's
silent report fallback was reachable without a typo.

The first two tests are the general seam: a documented command word must be one
the driver accepts, and an arm named in the prose must be in `ARMS` and the other
way round. No model, no server, no compiler is run here — the drivers are read as
text and imported, and the report path reads JSON from a temporary directory.

    cd benchmark && python3 -m unittest test_driver_documented_commands -v
"""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import pathlib
import re
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "benchmark"))

DRIVERS = (
    "memory_book",
    "memory_correct",
    "memory_master",
    "memory_mini",
    "memory_projects",
    "memory_sim",
    "memory_small_model",
    "memory_volume",
    "memory_value",
)

WORDS = re.compile(r"python3 benchmark/(memory_\w+)\.py ([a-z][a-z-]*)")
COUNT = re.compile(r"\b(One|Two|Three|Four|Five|Six) arms\b")
NUMERALS = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5, "Six": 6}


def accepted_words(module) -> set[str]:
    """Every command the driver's own `__main__` dispatches on."""
    source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
    words = set(re.findall(r'(?:elif|if) (?:command|which) == "([a-z-]+)"', source))
    words.update(module.ARMS) if hasattr(module, "ARMS") else None
    return words


class DocumentedCommandTests(unittest.TestCase):
    def test_every_documented_command_is_one_the_driver_accepts(self):
        for name in DRIVERS:
            with self.subTest(driver=name):
                module = importlib.import_module(name)
                known = accepted_words(module)
                for driver, word in WORDS.findall(module.__doc__ or ""):
                    self.assertEqual(driver, name, f"{name} documents another driver")
                    self.assertIn(
                        word,
                        known,
                        f"{name}'s docstring offers `{word}` but its dispatch refuses it",
                    )

    def test_the_arms_the_prose_lists_are_the_arms_the_code_has(self):
        for name in DRIVERS:
            module = importlib.import_module(name)
            arms = set(getattr(module, "ARMS", ()))
            doc = module.__doc__ or ""
            prose = COUNT.search(doc)
            if not arms or prose is None:
                continue
            with self.subTest(driver=name):
                for arm in sorted(arms):
                    self.assertRegex(
                        doc,
                        rf"(?m)^\s{{4}}{arm}\s+",
                        f"{name} runs an `{arm}` arm its docstring never names",
                    )
                self.assertEqual(
                    NUMERALS[prose.group(1)],
                    len(arms),
                    f"{name} says `{prose.group(1)} arms` while ARMS holds "
                    f"{len(arms)}: {sorted(arms)}",
                )


class DeadHelperTests(unittest.TestCase):
    def test_memory_value_ships_no_helper_it_never_calls(self):
        module = importlib.import_module("memory_value")
        for name in ("compiles", "code_block"):
            self.assertFalse(
                hasattr(module, name),
                f"{name}() is uncalled anywhere and scores an absent toolchain as "
                f"the model's failure the moment it is wired up",
            )

    def test_deleting_them_leaves_the_report_the_same(self):
        """The removal is pure: the printed measurement does not change shape."""
        module = importlib.import_module("memory_value")
        with tempfile.TemporaryDirectory() as temp:
            results = pathlib.Path(temp)
            module.OUT = results
            row = {
                "stage": "swift",
                "prompt_tokens": 210,
                "completion_tokens": 90,
                "seconds": 1.0,
                "consolidation_wait": 0.0,
                "content": '```json\n{"field_w": 800}\n```',
            }
            (results / "control-r1.json").write_text(json.dumps([row]), encoding="utf-8")
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                module.report()
        text = printed.getvalue()
        self.assertIn("arm      run stage     prompt  completion  seconds", text)
        self.assertIn("Carry-over per run:", text)
        self.assertIn("Cost per run", text)
        self.assertIn("control", text)


if __name__ == "__main__":
    unittest.main()
