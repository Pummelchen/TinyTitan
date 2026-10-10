"""AUD-272: a markdown table is one paragraph to `check_names`, so one row's gate names draft its neighbours.

`tools/docs-facts.py`'s names rule is good: "inside a paragraph that lists modes, a
hyphenated name that is not a mode is a name the script no longer answers to." Its
unit is the bug. `paragraphs_of()` splits on blank lines, and a GFM table has none
between its rows, so the eleven-row `Where the work stands` table in
`docs/handover-tinytitan.md` is a single paragraph. Any row that names three real
gates therefore puts every hyphenated token in *every other row* on trial, and a row
about model goldens has no idea it is being read as a list of gates.

Measured on the tree that made it happen: `docs/handover-tinytitan.md:120` reported
four refusals -- `ornith-8`, `qwen38-4`, `qwen38-8`, `thread-sanitizer` -- none of
which is a gate, a mode, or mentioned anywhere near one. Each is a real name in this
repository (three golden targets in the Goldens row, a sanitizer run in another), and
the row that triggered the check was the Audit row, several rows below.

The other direction is why the threshold matters and why the fix keeps it: a
paragraph that names fewer than three modes is not read as a gate list at all, so a
bogus gate name hiding beside one or two real ones is never flagged. Splitting the
table by row does not change that rule, and a row that genuinely names three gates
still answers for its own invented fourth -- which is the second test, and the reason
the first one cannot be satisfied by making the check do nothing.

Run from `benchmark/`:

    python3 -m unittest test_docs_facts_gate_names
"""

from __future__ import annotations

import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
GATE = ROOT / "tools" / "docs-facts.py"

TABLE = [
    "| Piece | State |",
    "| --- | --- |",
    "| Goldens | the default `ornith-8` is among them |",
    "| Audit | the gates `force-cast`, `arch-path`, `shell-portability` all pass |",
]
BOGUS_ROW = "| Audit | `force-cast`, `arch-path`, `shell-portability`, `cast-length` |"
BOGUS_PROSE = (
    "The gates `force-cast`, `arch-path`, `shell-portability` and `cast-length` all run in CI."
)


def gate_module():
    spec = importlib.util.spec_from_file_location("docs_facts_names_gate", GATE)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {GATE} as a module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GateNameParagraphUnit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gate = gate_module()
        cls.modes, cls.checks, problems = cls.gate.gate_set()
        if problems:
            raise AssertionError(f"tools/lint.sh modes did not derive: {problems}")
        for token in ("force-cast", "arch-path", "shell-portability"):
            if token not in cls.modes:
                raise AssertionError(f"the fixture names {token}, which lint.sh no longer has")

    def rows(self, lines):
        return self.gate.check_names("doc.md", self.gate.paragraphs_of(lines), self.modes)

    def test_a_neighbouring_row_is_not_read_as_this_rows_gate_list(self):
        self.assertEqual([], self.rows(TABLE))

    def test_a_row_naming_three_gates_still_answers_for_a_fourth_it_invented(self):
        found = self.rows(TABLE[:3] + [BOGUS_ROW])
        self.assertEqual(1, len(found), found)
        self.assertIn("cast-length", found[0])
        # The message has to point at the row that invented the name, not at the
        # table's header: `doc.md:1` sent the reader to `| Piece | State |`.
        self.assertIn("doc.md:4:", found[0])

    def test_a_paragraph_that_wraps_across_lines_still_groups(self):
        # Row granularity is for table rows only. Prose wraps, and a claim that
        # names its gates on one line and its invention on the next is one claim;
        # a fix that split every line into its own paragraph would unread it.
        wrapped = [
            "The gates `force-cast`, `arch-path`, `shell-portability` are pinned,",
            "and so is `cast-length`, which is not one of them.",
        ]
        found = self.rows(wrapped)
        self.assertEqual(1, len(found), found)
        self.assertIn("cast-length", found[0])

    def test_prose_that_lists_gates_is_still_checked(self):
        found = self.rows([BOGUS_PROSE])
        self.assertEqual(1, len(found), found)
        self.assertIn("cast-length", found[0])

    def test_the_shipped_handover_table_names_no_gate_that_is_not_one(self):
        text = (ROOT / "docs" / "handover-tinytitan.md").read_text(encoding="utf-8")
        self.assertEqual([], self.rows(text.splitlines()))


if __name__ == "__main__":
    unittest.main()
