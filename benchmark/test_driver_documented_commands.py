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
import os
import pathlib
import re
import sys
import tempfile
import unittest
from unittest import mock

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


def import_driver(name: str):
    """The one door this suite uses to import a driver: a hermetic one.

    Reloading rather than importing is what makes the door worth having --
    `importlib.import_module` answers from `sys.modules` after the first call, so
    a driver's module-level code runs once per process and a leak would hide
    behind the cache from whichever test came second.
    """
    with mock.patch.dict(os.environ):
        module = importlib.import_module(name)
        return importlib.reload(module)


class DocumentedCommandTests(unittest.TestCase):
    def test_every_documented_command_is_one_the_driver_accepts(self):
        for name in DRIVERS:
            with self.subTest(driver=name):
                module = import_driver(name)
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
            module = import_driver(name)
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


class ImportHygieneTests(unittest.TestCase):
    def test_reading_a_driver_does_not_change_the_process_environment(self):
        """AUD-250: importing a driver to read it must not retarget the run.

        `memory_small_model.py` publishes its results tree and its port into
        `os.environ` at import, because the harness it then imports has to write
        where it does. That is right for a run and wrong for an inspection: this
        suite imports nine drivers in-process, so the process leaves the suite
        with `TINYTITAN_MEMVAL_RESULTS` set to
        `.build/benchmark-logs/memory-small-qwen2b`. Measured over
        `python3 -m unittest test_driver_documented_commands test_memval_master_exit`:
        the stubbed master run wrote nothing, its report walked the tree the
        *leaked* variable named -- the repository's real recorded runs -- and
        printed `MASTER DONE ... all 10 scenarios exited 0` over photograph's
        116/135 carryable and 2743 s of someone else's model time, exit 0.
        AUD-244's refusal guard is the test that failed, so the leak does not
        merely reorder a suite: it silently disarms the gate that catches a run
        which measured nothing.

        The two knobs are named and cleared first because a leaked value already
        in the environment would make the comparison pass without anything being
        proven: the first draft of this test ran last in the module's own order,
        after the drivers had been imported, and reported ok over a process that
        was already retargeted.
        """
        with mock.patch.dict(os.environ):
            for key in ("TINYTITAN_MEMVAL_RESULTS", "TINYTITAN_PORT"):
                os.environ.pop(key, None)
            before = dict(os.environ)
            for name in DRIVERS:
                import_driver(name)
            changed = {
                key: (before.get(key), os.environ.get(key))
                for key in set(before) | set(os.environ)
                if before.get(key) != os.environ.get(key)
            }
            self.assertEqual(
                changed,
                {},
                "importing a driver for inspection changed environment variables "
                "every later test in this process inherits: (before, after) per key",
            )


class DeadHelperTests(unittest.TestCase):
    def test_memory_value_ships_no_helper_it_never_calls(self):
        module = import_driver("memory_value")
        for name in ("compiles", "code_block"):
            self.assertFalse(
                hasattr(module, name),
                f"{name}() is uncalled anywhere and scores an absent toolchain as "
                f"the model's failure the moment it is wired up",
            )

    def test_deleting_them_leaves_the_report_the_same(self):
        """The removal is pure: the printed measurement does not change shape."""
        module = import_driver("memory_value")
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
