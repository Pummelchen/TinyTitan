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
HEAD cannot reach, prose where a sha belongs, a shallow clone that carries no history
to resolve a sha against, and the one deliberate exception --
a blank field on a row that recorded no fix (AUD-132 refuted its finding; AUD-139
is blocked). A blank on a DONE row is the opposite of that: it claims a fix and
names no pointer, which is how the row that closed AUD-262 lost its evidence. The
two shapes are tested apart, because an implementation that only ever ran the
ancestry check would pass the first test for the wrong reason.

AUD2-049: the list of ledgers was a hardcoded `docs/audit-ledger.json`, so this
audit's own `AUDIT/ledger.json` was opened by nothing while the gate printed its
reference count as if it had. Extending the path alone would have traded one
silence for a wrong number: the two ledgers describe two different audits, this
audit's ledger publishes no `counts` block, and the handover restates the *prior*
ledger's 210/210/0/0. Two fixes are therefore pinned here -- every tracked
`*ledger.json` is discovered and counted from its own `tasks` statuses (a declared
`counts` block is checked against them, not trusted), and a restated count is
judged against the ledger its sentence names rather than against a sum. The commit
check is exercised on a copy of this audit's real ledger with one DONE row's sha
made unresolvable, which is the failure the handover could not see before.

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
LEDGER_REL = "docs/audit-ledger.json"
BOGUS = "14710ba"  # the sha the ledger actually got wrong


def gate_module():
    spec = importlib.util.spec_from_file_location("docs_facts_gate", GATE)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {GATE} as a module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(task_id: str, commit: str, status: str = "DONE") -> dict:
    return {
        "id": task_id,
        "severity": "S2",
        "tier": "A",
        "project": "docs-gates",
        "file_line": "docs/audit-ledger.json",
        "title": f"fixture row {task_id}",
        "category": "missing gate",
        "status": status,
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
        # The shallow clone the last test makes goes in its own stage, not beside
        # the fixture repo: `repo.parent` is the system temp dir, so writing there
        # would collide with the next run and leave a directory behind.
        self.dest = tempfile.TemporaryDirectory(prefix="docs-facts-clone-")
        self.addCleanup(self.dest.cleanup)
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

    def test_a_shallow_clone_names_the_history_it_does_not_carry(self):
        # CI ran this gate on `fe19203` and printed 211 `no such commit` lines, one
        # per ledger reference, every one of them false: the commits exist, the
        # checkout was `actions/checkout`'s default depth of 1. A refusal whose
        # stated cause did not happen is the defect this audit keeps finding, so a
        # clone without history is refused as a clone without history.
        (self.repo / "c.txt").write_text("three\n", encoding="utf-8")
        git(self.repo, "add", "c.txt")
        git(self.repo, "commit", "-q", "-m", "second")
        clone = pathlib.Path(self.dest.name) / "shallow"
        subprocess.run(
            ["git", "clone", "-q", "--depth", "1", f"file://{self.repo}", str(clone)],
            capture_output=True,
            text=True,
            check=True,
        )
        ledger = clone / LEDGER_REL
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_text(
            json.dumps({"audit": "fixture", "counts": {}, "tasks": [row("AUD-4", self.on_main)]})
            + "\n",
            encoding="utf-8",
        )
        self.assertEqual("true", git(clone, "rev-parse", "--is-shallow-repository"))
        lines, checked = self.module.ledger_commit_evidence([LEDGER_REL], root=clone)
        self.assertEqual(1, len(lines), lines)
        self.assertIn("AUD-4", lines[0])
        self.assertIn("shallow", lines[0])
        self.assertNotIn("no such commit", lines[0])
        self.assertEqual(0, checked, "nothing was resolved, so nothing is claimed")

    def test_prose_where_a_sha_belongs_fails(self):
        lines, _ = self.check([row("AUD-7", "this commit")])
        self.assertEqual(1, len(lines), lines)
        self.assertIn("AUD-7", lines[0])
        self.assertIn("names no commit", lines[0])

    def test_a_blank_field_is_allowed_for_a_row_that_recorded_no_fix(self) -> None:
        # A row that refuted its finding, was blocked, or is still open has no
        # commit to name; the gate must not push a real sha into it.
        lines, checked = self.check(
            [row("AUD-6", "", status="BLOCKED"), row("AUD-5", "", status="OPEN")]
        )
        self.assertEqual([], lines)
        self.assertEqual(0, checked)

    def test_a_done_row_with_a_blank_field_fails(self) -> None:
        """A blank field is honest for a row that fixed nothing, not for a DONE one.

        The exception above is justified by rows that have no fix to point at, but
        the gate applied it to every status -- which is how this audit closed
        AUD-262 with a fix on `main`, an empty `commit` field, and a report that
        said the pointer had been kept. A reader cannot check what is not named.
        """
        lines, checked = self.check([row("AUD-3", "")])
        self.assertEqual(1, len(lines), lines)
        self.assertIn("AUD-3", lines[0])
        self.assertIn("DONE", lines[0])
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
        self.assertIn("no audit ledger is tracked", lines[0])
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


LEDGER_PRIOR = "docs/audit-ledger.json"
LEDGER_THIS = "AUDIT/ledger.json"


def fixture_tasks(*statuses: str) -> list[dict]:
    """Task rows carrying only what the count is derived from."""
    return [{"id": f"T-{index}", "status": status} for index, status in enumerate(statuses)]


def fixture_ledger(*statuses: str, counts: dict | None = None) -> dict:
    body = {"audit": "fixture", "tasks": fixture_tasks(*statuses)}
    if counts is not None:
        body["counts"] = counts
    return body


class LedgerCountTests(unittest.TestCase):
    """AUD2-049: a ledger's counts are its own task rows, never a cross-ledger total."""

    @classmethod
    def setUpClass(cls):
        cls.module = gate_module()

    def test_every_status_is_counted_and_open_is_everything_but_closed_and_blocked(self):
        derived = self.module.counts_from_tasks(
            fixture_tasks("DONE", "OPEN", "PROGRESS", "AUDIT", "BLOCKED")
        )
        self.assertEqual((5, 1, 3, 1), derived)

    def test_counts_are_derived_when_the_ledger_publishes_no_counts_block(self):
        # This is AUDIT/ledger.json's shape: DONE, OPEN, PROGRESS, AUDIT, BLOCKED
        # with no `counts` block at all. An implementation that reads only the
        # block sees nothing here.
        self.assertEqual(
            (5, 1, 3, 1),
            self.module.counts_from_tasks(
                fixture_tasks("DONE", "OPEN", "OPEN", "PROGRESS", "BLOCKED")
            ),
        )


class LedgerDiscoveryTests(unittest.TestCase):
    """AUD2-049: `git ls-files` has to yield every ledger, counted separately.

    The fixture repository carries two ledgers with different totals, which is the
    shape that breaks when a hardcoded path is widened naively: summing them gives
    a number neither file states, and the handover restates one file's counts.
    """

    def setUp(self):
        self.module = gate_module()
        stage = tempfile.TemporaryDirectory(prefix="docs-facts-counts-")
        self.addCleanup(stage.cleanup)
        self.repo = pathlib.Path(stage.name)
        git(self.repo, "init", "-q")
        git(self.repo, "config", "user.email", "fixture@example.invalid")
        git(self.repo, "config", "user.name", "Fixture")
        (self.repo / "a.txt").write_text("one\n", encoding="utf-8")
        git(self.repo, "add", "a.txt")
        git(self.repo, "commit", "-q", "-m", "first")

    def write(self, rel: str, body: object) -> None:
        path = self.repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(body) + "\n", encoding="utf-8")
        git(self.repo, "add", rel)

    def test_every_tracked_ledger_is_discovered_and_counted_on_its_own(self):
        self.write(
            LEDGER_PRIOR,
            fixture_ledger(
                "DONE", "DONE", "DONE", counts={"total": 3, "done": 3, "open": 0, "blocked": 0}
            ),
        )
        self.write(LEDGER_THIS, fixture_ledger("DONE", "OPEN"))
        git(self.repo, "commit", "-q", "-m", "ledgers")

        counts, paths, errors = self.module.ledger_counts(root=self.repo)
        self.assertEqual([], errors)
        self.assertEqual([LEDGER_THIS, LEDGER_PRIOR], paths)
        self.assertEqual((3, 3, 0, 0), counts[LEDGER_PRIOR])
        self.assertEqual((2, 1, 1, 0), counts[LEDGER_THIS])
        # The sum -- what a widened hardcoded pattern would have produced -- is in
        # neither ledger and must not become the gate's idea of either.
        self.assertNotIn((5, 4, 1, 0), counts.values())

    def test_a_declared_counts_block_that_disagrees_with_the_tasks_fails(self):
        self.write(
            LEDGER_THIS,
            fixture_ledger("DONE", "OPEN", counts={"total": 3, "done": 2, "open": 1, "blocked": 0}),
        )
        git(self.repo, "commit", "-q", "-m", "ledger")
        counts, _, errors = self.module.ledger_counts(root=self.repo)
        self.assertEqual(1, len(errors), errors)
        self.assertIn("counts block", errors[0])
        self.assertIn("2 task rows", errors[0])
        self.assertEqual((2, 1, 1, 0), counts[LEDGER_THIS], "the rows still count")

    def test_a_ledger_with_no_task_rows_fails_rather_than_counting_zero(self):
        self.write(LEDGER_THIS, {})
        git(self.repo, "commit", "-q", "-m", "empty ledger")
        _, _, errors = self.module.ledger_counts(root=self.repo)
        self.assertEqual(1, len(errors), errors)
        self.assertIn("no task rows", errors[0])

    def test_a_ledger_that_is_not_an_object_fails_rather_than_crashing(self):
        self.write(LEDGER_THIS, [])
        git(self.repo, "commit", "-q", "-m", "list-shaped ledger")
        _, _, errors = self.module.ledger_counts(root=self.repo)
        self.assertEqual(1, len(errors), errors)
        self.assertIn("not a JSON object", errors[0])


class LedgerRestatementTests(unittest.TestCase):
    """AUD2-049: "N rows / N closed / ..." is judged against one ledger, not a sum."""

    COUNTS = {LEDGER_PRIOR: (210, 210, 0, 0), LEDGER_THIS: (51, 36, 14, 1)}

    def setUp(self):
        self.module = gate_module()

    def judge(self, text: str) -> list[str]:
        paragraphs = self.module.paragraphs_of(text.splitlines())
        return self.module.check_ledger_counts(
            "docs/handover-tinytitan.md", paragraphs, self.COUNTS
        )

    def test_the_prior_ledgers_restated_counts_still_pass_beside_a_second_ledger(self):
        text = (
            "| Audit | `docs/audit-ledger.json` is **210 rows / 210 closed / 0 open / 0 blocked** |"
        )
        self.assertEqual([], self.judge(text))

    def test_this_audits_own_counts_pass_without_a_counts_block(self):
        text = (
            "| Audit | `AUDIT/ledger.json` stands at "
            "**51 rows / 36 closed / 14 open / 1 blocked** |"
        )
        self.assertEqual([], self.judge(text))

    def test_the_sum_of_two_audits_is_not_a_ledger_count(self):
        rows = self.judge("The audit stands at 261 rows / 246 closed / 14 open / 1 blocked.")
        self.assertEqual(1, len(rows), rows)
        self.assertIn("261 rows / 246 closed / 14 open / 1 blocked", rows[0])
        self.assertIn(f"{LEDGER_PRIOR} has 210 rows / 210 closed / 0 open / 0 blocked", rows[0])
        self.assertIn(f"{LEDGER_THIS} has 51 rows / 36 closed / 14 open / 1 blocked", rows[0])

    def test_a_count_that_names_one_ledger_is_checked_against_that_ledger(self):
        rows = self.judge(
            "`docs/audit-ledger.json` holds 51 rows / 36 closed / 14 open / 1 blocked."
        )
        self.assertEqual(1, len(rows), rows)
        self.assertIn(f"{LEDGER_PRIOR} has 210 rows", rows[0])


class ThisAuditsLedgerTests(unittest.TestCase):
    """AUD2-049: the gate has to read `AUDIT/ledger.json`, and would see its rot."""

    @classmethod
    def setUpClass(cls):
        cls.module = gate_module()

    def test_discovery_reads_both_ledgers_and_derives_each_ones_counts(self):
        counts, paths, errors = self.module.ledger_counts()
        self.assertEqual([], errors)
        self.assertIn(LEDGER_PRIOR, paths)
        self.assertIn(LEDGER_THIS, paths)
        for rel in (LEDGER_PRIOR, LEDGER_THIS):
            body = json.loads((ROOT / rel).read_text(encoding="utf-8"))
            self.assertEqual(
                self.module.counts_from_tasks(body["tasks"]),
                counts[rel],
                f"{rel} must be counted from its own task rows",
            )
            declared = body.get("counts")
            if declared is not None:
                self.assertEqual(
                    tuple(declared[key] for key in ("total", "done", "open", "blocked")),
                    counts[rel],
                    f"{rel}'s declared counts must agree with its rows",
                )
        self.assertNotEqual(counts[LEDGER_PRIOR], counts[LEDGER_THIS])

    def test_the_gate_resolves_the_commit_references_of_both_ledgers(self):
        result = subprocess.run(
            ["python3", str(GATE)], capture_output=True, text=True, check=False, cwd=ROOT
        )
        expected = sum(
            len(self.module.COMMIT_REF.findall(str(task.get("commit") or "")))
            for rel in (LEDGER_PRIOR, LEDGER_THIS)
            for task in json.loads((ROOT / rel).read_text(encoding="utf-8"))["tasks"]
        )
        self.assertIn(f"{expected} ledger commit reference(s) resolved", result.stdout)
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)

    def test_an_unresolvable_sha_in_a_copy_of_this_ledger_fails(self):
        body = json.loads((ROOT / LEDGER_THIS).read_text(encoding="utf-8"))
        done = next(task for task in body["tasks"] if task["status"] == "DONE")
        done["commit"] = BOGUS
        expected = sum(
            len(self.module.COMMIT_REF.findall(str(task.get("commit") or "")))
            for task in body["tasks"]
        )
        with tempfile.TemporaryDirectory(prefix="docs-facts-copy-") as stage:
            copy = pathlib.Path(stage) / "ledger.json"
            copy.write_text(json.dumps(body) + "\n", encoding="utf-8")
            lines, checked = self.module.ledger_commit_evidence([str(copy)], root=ROOT)
        self.assertEqual(1, len(lines), lines)
        self.assertIn(done["id"], lines[0])
        self.assertIn(BOGUS, lines[0])
        self.assertIn("no such commit", lines[0])
        self.assertEqual(expected, checked)

    def test_a_null_commit_field_is_blank_not_the_word_none(self):
        # AUDIT/ledger.json writes `null` for the rows that recorded no fix. That is
        # the blank-field exception, not a prose pointer named "None".
        lines, checked = self.module.ledger_commit_evidence([str(self.null_ledger())], root=ROOT)
        self.assertEqual([], lines)
        self.assertEqual(0, checked)

    def null_ledger(self) -> pathlib.Path:
        stage = tempfile.TemporaryDirectory(prefix="docs-facts-null-")
        self.addCleanup(stage.cleanup)
        path = pathlib.Path(stage.name) / "ledger.json"
        path.write_text(
            json.dumps(
                {
                    "audit": "fixture",
                    "tasks": [
                        row("AUD-6", None, status="OPEN"),
                        row("AUD-5", None, status="BLOCKED"),
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return path


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
