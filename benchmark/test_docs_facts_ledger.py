"""AUD-209: the audit ledger's `commit` field is evidence, and nothing checked it.

The row that started this is real and still in the file: AUD-108 records its fix as
`14710ba`, and `git cat-file -e 14710ba^{commit}` says there is no such object --
the depth cap landed at `14a4e06`. Six more rows name a commit as prose
(`this commit`, `see the audit(AUD-101) commit`), which was true the moment they
were written and refers to nothing afterwards; two record nothing although their
fix is committed. A reader who cannot check a sha cannot check the fix, so a ledger
row whose commit field does not resolve is a row that has lost its evidence.

`tools/docs-facts.py` is the gate that already refuses a documented number that
disagrees with the repository, so the same rule is extended to the commit
references. It reads a *fixture* git repository here rather than this one, so every
branch is reachable: a sha that does not exist at all, a sha that exists on a branch
HEAD cannot reach, prose where a sha belongs, and the one deliberate exception --
a blank field, which is honest for a row that recorded no fix (AUD-132 refuted its
finding; AUD-139 is blocked). The two shapes are tested apart, because an
implementation that only ever ran the ancestry check would pass the first test for
the wrong reason.

Run from `benchmark/`:

    python3 -m unittest test_docs_facts_ledger
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
GATE = ROOT / "tools" / "docs-facts.py"
LEDGER_REL = "docs/audit-2026-10-06/ledger.json"
BOGUS = "14710ba"  # the sha the ledger actually got wrong


def gate_module():
    spec = importlib.util.spec_from_file_location("docs_facts_gate", GATE)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {GATE} as a module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(task_id: str, commit: str) -> dict:
    return {
        "id": task_id,
        "severity": "S2",
        "tier": "A",
        "project": "docs-gates",
        "file_line": "docs/audit-2026-10-06/ledger.json",
        "title": f"fixture row {task_id}",
        "category": "missing gate",
        "status": "DONE",
        "host": "fixture",
        "discovered-by": "fixture",
        "evidence-before": "before",
        "evidence-after": "after",
        "fix-summary": "summary",
        "commit": commit,
    }


class LedgerCommitEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = gate_module()

    def setUp(self):
        stage = tempfile.TemporaryDirectory(prefix="docs-facts-ledger-")
        self.addCleanup(stage.cleanup)
        self.repo = pathlib.Path(stage.name)
        git(self.repo, "init", "-q")
        git(self.repo, "config", "user.email", "fixture@example.invalid")
        git(self.repo, "config", "user.name", "Fixture")
        (self.repo / "a.txt").write_text("one\n", encoding="utf-8")
        git(self.repo, "add", "a.txt")
        git(self.repo, "commit", "-q", "-m", "first")
        self.on_main = git(self.repo, "rev-parse", "--short", "HEAD")
        # A commit HEAD cannot reach: it exists, its object is valid, and naming it
        # as the commit that fixed something is still wrong.
        git(self.repo, "checkout", "-q", "-b", "side")
        (self.repo / "b.txt").write_text("two\n", encoding="utf-8")
        git(self.repo, "add", "b.txt")
        git(self.repo, "commit", "-q", "-m", "unmerged")
        self.unmerged = git(self.repo, "rev-parse", "--short", "HEAD")
        git(self.repo, "checkout", "-q", "-")

    def check(self, tasks):
        ledger = self.repo / LEDGER_REL
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_text(
            json.dumps({"audit": "fixture", "counts": {}, "tasks": tasks}) + "\n",
            encoding="utf-8",
        )
        lines, checked = self.module.ledger_commit_evidence([LEDGER_REL], root=self.repo)
        return lines, checked

    def test_a_real_ancestor_sha_passes(self):
        lines, checked = self.check([row("AUD-1", self.on_main)])
        self.assertEqual([], lines)
        self.assertEqual(1, checked)

    def test_a_sha_with_no_object_fails_by_name(self):
        lines, _ = self.check([row("AUD-9", BOGUS)])
        self.assertEqual(1, len(lines), lines)
        self.assertIn("FAIL", lines[0])
        self.assertIn("AUD-9", lines[0])
        self.assertIn(BOGUS, lines[0])
        self.assertIn("no such commit", lines[0])

    def test_a_sha_that_exists_but_head_cannot_reach_fails_differently(self):
        lines, _ = self.check([row("AUD-8", self.unmerged)])
        self.assertEqual(1, len(lines), lines)
        self.assertIn("AUD-8", lines[0])
        self.assertIn("not reachable from HEAD", lines[0])
        self.assertNotIn("no such commit", lines[0])

    def test_prose_where_a_sha_belongs_fails(self):
        lines, _ = self.check([row("AUD-7", "this commit")])
        self.assertEqual(1, len(lines), lines)
        self.assertIn("AUD-7", lines[0])
        self.assertIn("names no commit", lines[0])

    def test_a_blank_field_is_the_allowed_exception(self):
        # A row that recorded no fix has no commit to name; that is the one honest
        # use of an empty field, and the gate must not push a real sha into it.
        lines, checked = self.check([row("AUD-6", ""), row("AUD-5", "")])
        self.assertEqual([], lines)
        self.assertEqual(0, checked)

    def test_every_sha_in_a_multi_commit_field_is_checked(self):
        lines, checked = self.check([row("AUD-4", f"{self.on_main}, {BOGUS}")])
        self.assertEqual(1, len(lines), lines)
        self.assertIn(BOGUS, lines[0])
        self.assertNotIn(self.on_main, lines[0])
        self.assertEqual(2, checked)

    def test_a_ledger_that_yielded_no_rows_fails_instead_of_passing(self):
        # The vacuity rule this gate already uses elsewhere: a scan that read
        # nothing must not report a pass.
        lines, _ = self.check([])
        self.assertEqual(1, len(lines), lines)
        self.assertIn("no task rows", lines[0])

    def test_an_unreadable_ledger_fails(self):
        ledger = self.repo / LEDGER_REL
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_text("{not json\n", encoding="utf-8")
        lines, _ = self.module.ledger_commit_evidence([LEDGER_REL], root=self.repo)
        self.assertEqual(1, len(lines), lines)
        self.assertIn("cannot read", lines[0])

    def test_no_ledger_at_all_fails_rather_than_passing(self):
        # The list comes from `git ls-files`; an emptied or moved audit directory
        # must read as a gate that could not check, not as a ledger with nothing
        # wrong in it.
        lines, checked = self.module.ledger_commit_evidence([], root=self.repo)
        self.assertEqual(1, len(lines), lines)
        self.assertIn("no docs/audit-*/ledger.json", lines[0])
        self.assertEqual(0, checked)

    def test_a_missing_ledger_fails_rather_than_being_skipped(self):
        lines, _ = self.module.ledger_commit_evidence(
            ["docs/audit-1999-01-01/ledger.json"], root=self.repo
        )
        self.assertEqual(1, len(lines), lines)
        self.assertIn("cannot read", lines[0])

    def test_the_gate_checks_this_repository_and_finds_it_clean(self):
        # Reachability: the published count only appears if main() calls the check,
        # so a real run over this repository's own ledger is the proof it is wired.
        result = subprocess.run(
            ["python3", str(GATE)], capture_output=True, text=True, check=False, cwd=ROOT
        )
        self.assertGreaterEqual(shas_in_real_ledger(), 100)
        self.assertIn("ledger commit reference(s)", result.stdout)
        self.assertNotIn("no such commit", result.stdout)
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)


def shas_in_real_ledger() -> int:
    ledger = json.loads((ROOT / LEDGER_REL).read_text(encoding="utf-8"))
    hexes = 0
    for task in ledger["tasks"]:
        for token in str(task.get("commit", "")).replace(",", " ").split():
            if len(token) >= 7 and all(c in "0123456789abcdef" for c in token):
                hexes += 1
    return hexes


def git(repo: pathlib.Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)
    return proc.stdout.strip()


if __name__ == "__main__":
    unittest.main()
