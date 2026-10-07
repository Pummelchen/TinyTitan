#!/usr/bin/env python3
"""AUD-160: compare what the documents claim to what the repository computes.

Two kinds of fact get restated in prose here and can be derived instead: how many
gates `tools/lint.sh` runs and what they are called, and which commit a release
tag points at. Both have already drifted. `AGENTS.md` says "thirteen checks"
while the script runs seventeen; `CONTRIBUTING.md` said eleven; `RELEASE.md` and
`ci.yml` each kept their own number; four names in the script's own usage header
(`unchecked-sendable`, `silent-test-skip`, `shell-portability`, `shell-lint`)
were not runnable modes at all, so `tools/lint.sh shell-portability` -- the name
`AGENTS.md` teaches -- exited 2; and the release brief cited three *tag-object*
shas where every consumer of a sha (`tools/ci-green.sh`, `tools/release.sh`,
`gh run list --commit`) needs the tagged commit. Prose does not fail when the
computation changes, so none of that was visible until someone read it.

The gate reads the gate set out of `tools/lint.sh` itself -- the `all)` chain,
its `case` arms, its usage header and its unknown-check message, which must agree
with each other before any document is judged against them -- then scans the
tracked Markdown and workflow YAML for count claims, mode names that no longer
exist, the other restated counts the repository can compute (a ratchet baseline's
rows, a directory's tracked files, the golden target list), `<tag> -> <sha>`
citations, and tables whose rows do not all have the
header's column count. That last one is not a style rule: a row with too many or
too few cells renders as a table that silently drops or pads them, so the table
looks right and says something else, and it was introduced while this gate was
being written.

Excluded on purpose: this audit's own ledger and tool-coverage notes. The ledger
quotes wrong numbers and wrong shas *verbatim* as the thing it later refutes, and
it is corrected forward, so a gate that failed on a quotation would demand an
edit to the record rather than an addition to it.

Output: one line per finding. `FAIL` fails the gate. `OWNER` is the same
mismatch inside `AGENTS.md`, which is the maintainer's file: it is printed on
every run and does not fail, because a red CI held hostage by a file the person
writing it is mid-edit on teaches nobody anything. Everything else is enforced.
"""

import json
import os
import re
import subprocess
import sys

ROOT = os.environ.get("ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
LINT = os.path.join(ROOT, "tools", "lint.sh")
OWNER_FILES = ("AGENTS.md",)
# The audit's own record and the dated release notes: both quote wrong numbers
# and shas verbatim as the thing they later refute, and both are corrected
# forward, so a gate that failed on a quotation would demand an edit to history
# rather than an addition to it.
EXCLUDED = re.compile(r"^(docs/audit-[0-9-]+/|docs/release-notes-)")

WORD_NUMBER = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "twenty-one": 21,
    "twenty-two": 22,
}

# A claim about the size of the gate set: "runs **fifteen** pinned gates",
# "the thirteen checks CI enforces". Three things keep this from reading every
# plural as a claim. Only the plural noun counts, so "0 warnings gate" and "every
# one it could not check" are not claims about this. At most two modifier words
# may sit between the number and the noun, so "a 25 MB tarball. It now: checks"
# and "step 0 is one of the gates" are not either. And the sentence must name the
# thing being counted -- `tools/lint.sh` or CI -- within a short window, because
# that is what separates "the fifteen gates CI runs" (a live claim, checkable now)
# from "Verification: eleven gates clean" (a dated record of a run, which is
# history and stays wrong on purpose).
COUNT_CLAIM = re.compile(
    r"(?<![-\w.])\b("
    + "|".join(sorted(WORD_NUMBER, key=len, reverse=True))
    + r"|\d{1,2})\b(?:[\s*]+[a-z-]+){0,2}[\s*]*\b(checks|gates)\b"
)
COUNT_ANCHOR = re.compile(r"tools/lint\.sh|`lint\.sh`|\bCI\b")
# …and a sentence that reports a run ("all four gates ok", "eleven gates clean")
# is a record of one machine on one day, not a claim about the current tree.
RUN_RECORD = re.compile(r"\b(ok|okay|clean|passing|passed|green|verified|failures?)\b", re.I)
# …and a number inside quotation marks is a *mention*, not a claim: "This line
# used to keep a count ("fifteen checks")" is the shape a document uses to record
# a value it has just retired, and a gate that failed on it would demand the
# record be rewritten.
QUOTED = re.compile(r'["“][^"”]*["”]')
# AUD-162: the same drift in three other restated counts. Each demands the file
# or directory its number is about, because `N rows|files|targets` on its own
# matches n-gram tables, git diffs and locale catalogues.
BASELINE_ROWS = re.compile(r"(?<![\w.,\-/])(\d{1,3})\s+rows\b")
BASELINE_PATH = re.compile(r"\b(tools/[A-Za-z0-9._-]+\.txt)\b")
FILES_UNDER = re.compile(r"(?<![\w.,\-/])(\d{1,4})\s+files?\s+under\s+`([^`\n]+)`")
GOLDEN_TARGETS = re.compile(
    r"(?<![\w.,\-/])(\d{1,3})\s+targets?\s+in\s+`tools/golden-baseline\.sh`"
)

BACKTICK = re.compile(r"`([^`\n]+)`")
BOLD = re.compile(r"\*\*([^*\n]+)\*\*")
HEX_SHA = re.compile(r"\b([0-9a-f]{7,40})\b")
TAG = re.compile(r"`(v\d+(?:\.\d+)*)`")
NAME_SHAPE = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)+$")
# A section that records what happened rather than what is true. Its numbers are
# dated and are left dated; correcting the record means adding to it.
HISTORY_HEADING = re.compile(
    r"\b(landed|history|historical|archive|archived|changelog|log|verification|"
    r"measurements?|results?)\b",
    re.I,
)


def sh(*args):
    try:
        proc = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=False)
    except OSError as error:
        # Missing git, or a ROOT that is not a directory: report and fail, rather
        # than letting a traceback read as a gate that ran.
        return 1, "", str(error)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def kind_for(path):
    return "OWNER" if path in OWNER_FILES else "FAIL"


def gate_set():
    """The gate set derived from the script, plus its own internal disagreements."""
    problems = []
    try:
        with open(LINT, encoding="utf-8") as handle:
            text = handle.read()
    except OSError as error:
        # A gate that cannot read its own source must say so, not crash or report
        # zero gates against zero documents.
        return set(), set(), [f"FAIL cannot read {LINT}: {error}"]

    chain = re.search(r"^  all\)[ \t]+(.*?);;$", text, re.M | re.S)
    if not chain:
        return set(), set(), ["FAIL tools/lint.sh: no `all)` chain found"]
    chain_fns = [c.strip() for c in chain.group(1).split(";") if c.strip()]
    if len(set(chain_fns)) != len(chain_fns):
        problems.append("FAIL tools/lint.sh: the all chain runs a check twice")

    arms = dict(
        re.findall(r"^  (?!all\))([a-z][a-z-]*)\)[ \t]+(check_[a-z0-9_]+)[ \t]*;;$", text, re.M)
    )
    labels = [
        m.group(1) for m in re.finditer(r"^#[ \t]+([a-z][a-z-]+)[ \t]{2,}\S[^\n]*$", text, re.M)
    ]
    usage = re.search(r"unknown check: \$want \(([^)]*)\)", text)
    listed = set(usage.group(1).split("|")) if usage else set()

    for fn in chain_fns:
        if fn not in arms.values():
            problems.append(
                f"FAIL tools/lint.sh: {fn} runs in the all chain and no mode invokes it"
            )
    for label in labels:
        if label not in arms:
            problems.append(
                f"FAIL tools/lint.sh: the usage header advertises `{label}`, "
                f"which is not a runnable mode"
            )
    for mode, fn in arms.items():
        if fn not in chain_fns:
            problems.append(
                f"FAIL tools/lint.sh: mode `{mode}` calls {fn}, which the all chain never runs"
            )
    documented = {arms[label] for label in labels if label in arms}
    for fn in chain_fns:
        if fn not in documented:
            problems.append(
                f"FAIL tools/lint.sh: {fn} runs and the usage header never "
                f"describes it under any name"
            )
    missing_usage = sorted({m for m, f in arms.items() if f in chain_fns} - set(listed) - {"all"})
    if missing_usage:
        problems.append(
            "FAIL tools/lint.sh: the unknown-check message omits "
            f"{', '.join(missing_usage)}, so `tools/lint.sh --help` lies"
        )
    if len(labels) != len(set(labels)):
        problems.append("FAIL tools/lint.sh: the usage header names a check twice")

    names = {m for m, f in arms.items() if f in chain_fns and m != "all"}
    return names | set(labels), set(chain_fns), problems


def tracked_docs():
    """(documents the number checks may read, every tracked document, errors).

    The second list is for `check_tables` only: a row with the wrong number of
    cells is a rendering defect, not a quotation of history, so the audit's own
    ledger and these notes are exempt from the number checks and not from that
    one. It found one -- the `library-facade` row in `tool-coverage.md` carried a
    fourth column in a three-column table, and said so on its own line as if it
    were a row of its own."""
    rc, out, err = sh("git", "ls-files", "*.md", "*.yml", "*.yaml")
    if rc != 0:
        return [], [], [f"FAIL cannot list tracked documents: {err}"]
    everything = [p for p in out.splitlines() if os.path.isfile(os.path.join(ROOT, p))]
    paths = [p for p in everything if not EXCLUDED.match(p)]
    if not paths:
        return [], everything, ["FAIL no tracked documents to compare"]
    return paths, everything, []


def quoted_at(line, offset):
    """Is the match starting at `offset` inside a double-quoted span of `line`?"""
    return any(span.start() < offset < span.end() - 1 for span in QUOTED.finditer(line))


def check_counts(path, paragraphs, want):
    rows = []
    for start, text, heading in paragraphs:
        if is_history(heading):
            continue
        for match in COUNT_CLAIM.finditer(text):
            window = text[max(0, match.start() - 300) : match.end() + 300]
            if not COUNT_ANCHOR.search(window):
                continue
            prefix = text[: match.start()].rsplit("\n", 1)[-1]
            line = prefix + text[match.end() :].split("\n", 1)[0]
            if RUN_RECORD.search(line):
                continue
            if quoted_at(line, len(prefix)):
                continue
            word = match.group(1).lower()
            got = WORD_NUMBER.get(word)
            if got is None:
                if not word.isdigit():
                    continue
                got = int(word)
            if got == want:
                continue
            number = start + text[: match.start()].count("\n")
            rows.append(
                f"{kind_for(path)} {path}:{number}: says "
                f"“{match.group(0).replace('*', '')}”; tools/lint.sh runs {want}"
            )
    return rows


def ledger_totals():
    """The audit ledgers' own `counts`, summed. Returns (rows, done, open,
    blocked, paths); an empty path list means there is no ledger to compare to."""
    rc, out, _ = sh("git", "ls-files", "docs/audit-*/ledger.json")
    paths = sorted(out.splitlines()) if rc == 0 else []
    total = [0, 0, 0, 0]
    errors = []
    for rel in paths:
        try:
            with open(os.path.join(ROOT, rel), encoding="utf-8") as handle:
                counts = json.load(handle).get("counts", {})
        except (OSError, ValueError) as error:
            errors.append(f"FAIL cannot read {rel}: {error}")
            continue
        for index, key in enumerate(("total", "done", "open", "blocked")):
            total[index] += counts.get(key, 0)
    return tuple(total), paths, errors


LEDGER_COUNTS = re.compile(r"(\d+) rows / (\d+) closed / (\d+) open / (\d+) blocked")


def check_ledger_counts(path, paragraphs, totals, ledgers):
    """The handover restates the audit ledger's `counts`. That table is derived,
    and this audit has already shipped a commit whose printed counts disagreed
    with the file they described, so the restatement is checked against it."""
    rows = []
    if not ledgers:
        return rows
    want = list(totals)
    for start, text, heading in paragraphs:
        if is_history(heading):
            continue
        for match in LEDGER_COUNTS.finditer(text):
            prefix = text[: match.start()].rsplit("\n", 1)[-1]
            line = prefix + text[match.end() :].split("\n", 1)[0]
            if quoted_at(line, len(prefix)):
                continue
            if [int(group) for group in match.groups()] == want:
                continue
            line = start + text[: match.start()].count("\n")
            rows.append(
                f"{kind_for(path)} {path}:{line}: says “{match.group(0)}”; the "
                f"ledger(s) {', '.join(ledgers)} total {want[0]} rows / "
                f"{want[1]} closed / {want[2]} open / {want[3]} blocked"
            )
    return rows


def baseline_rows(rel):
    """A ratchet file's row count, blank lines ignored."""
    try:
        with open(os.path.join(ROOT, rel), encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return None


def tracked_file_count(pattern):
    rc, out, _ = sh("git", "ls-files", pattern)
    if rc != 0:
        return None
    return len([path for path in out.splitlines() if path])


def golden_target_count():
    """How many targets `tools/golden-baseline.sh` answers to, read from the
    `unknown target:` message -- the same list a user sees when they typo one."""
    try:
        with open(os.path.join(ROOT, "tools", "golden-baseline.sh"), encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return None
    match = re.search(r"unknown target: \$t \(([^)]*)\)", text)
    if not match:
        return None
    return len([name for name in match.group(1).split(",") if name.strip()])


def find_baseline(text, match):
    """The ratchet file named within 300 characters of a `N rows` claim, taking
    the closest one. A row count with no file beside it is prose about something
    else -- an n-gram table, a git diff -- and is not this gate's business; the
    window rather than the whole paragraph matters because a bulleted list is one
    paragraph, and a claim in the last bullet would otherwise inherit the first
    bullet's file."""
    start = max(0, match.start() - 300)
    before, after = text[start : match.start()], text[match.end() : match.end() + 300]
    hits = [(len(before) - h.end(), h.group(1)) for h in BASELINE_PATH.finditer(before)]
    hits += [(h.end(), h.group(1)) for h in BASELINE_PATH.finditer(after)]
    if not hits:
        return None
    return min(hits)[1]


LEDGER_TAIL = re.compile(r"\s*/\s*\d+ closed")


def check_derived_counts(path, paragraphs):
    """AUD-162: three more restated counts, each computable from the tree -- a
    ratchet baseline's rows, a directory's tracked files, the golden target
    list. Matched by phrase rather than by number: a corpus scan of `N
    rows|files|targets` turned up 23 hits and most describe n-gram tables, git
    diffs and locales rather than anything here, so each pattern demands the
    file or directory its number is a claim about."""
    rows = []
    for start, text, heading in paragraphs:
        if is_history(heading):
            continue
        for phrase, kind in (
            (BASELINE_ROWS, "rows"),
            (FILES_UNDER, "files"),
            (GOLDEN_TARGETS, "targets"),
        ):
            for match in phrase.finditer(text):
                prefix = text[: match.start()].rsplit("\n", 1)[-1]
                line = prefix + text[match.end() :].split("\n", 1)[0]
                if quoted_at(line, len(prefix)):
                    continue
                if kind == "rows":
                    # "62 rows / 58 closed" is the ledger count, and
                    # check_ledger_counts already judges it against ledger.json.
                    if LEDGER_TAIL.match(text[match.end() :]):
                        continue
                    source = find_baseline(text, match)
                    if source is None:
                        continue
                    got, truth = baseline_rows(source), f"{source} holds"
                elif kind == "files":
                    source = match.group(2).strip().strip("`")
                    got, truth = tracked_file_count(source), f"git ls-files {source} lists"
                else:
                    source = "tools/golden-baseline.sh"
                    got, truth = golden_target_count(), "tools/golden-baseline.sh lists"
                if got is None:
                    rows.append(
                        f"FAIL {path}: cannot count the {kind} a claim of "
                        f"“{match.group(0)}” rests on ({source})"
                    )
                    continue
                if int(match.group(1)) == got:
                    continue
                number = start + text[: match.start()].count("\n")
                rows.append(
                    f"{kind_for(path)} {path}:{number}: says "
                    f"“{match.group(0).replace('*', '')}”; {truth} {got} {kind}"
                )
    return rows


def check_names(path, paragraphs, modes):
    """Inside a paragraph that lists modes, a hyphenated name that is not a mode
    is a name the script no longer answers to."""
    rows = []
    for start, text, heading in paragraphs:
        if is_history(heading):
            continue
        tokens = {t.strip() for t in BACKTICK.findall(text)}
        tokens |= {t.strip() for t in BOLD.findall(text)}
        named = tokens & modes
        if len(named) < 3:
            continue
        for token in sorted(tokens - modes):
            if not NAME_SHAPE.match(token):
                continue
            rows.append(
                f"{kind_for(path)} {path}:{start}: names “{token}”, which is "
                f"neither a mode nor a header label of tools/lint.sh; the modes "
                f"are {', '.join(sorted(modes))}"
            )
    return rows


def check_tag_citations(path, paragraphs):
    rows = []
    for start, text, heading in paragraphs:
        if is_history(heading):
            continue
        for match in TAG.finditer(text):
            tag = match.group(1)
            cited = HEX_SHA.search(text[match.end() : match.end() + 60])
            if not cited:
                continue
            sha = cited.group(1)
            rc, commit, _ = sh("git", "rev-parse", "--verify", "--quiet", f"{tag}^{{commit}}")
            if rc != 0:
                continue  # a tag this clone lacks is not a stale citation
            if commit.startswith(sha) or sha.startswith(commit):
                continue
            rc, obj, _ = sh("git", "rev-parse", "--verify", "--quiet", tag)
            hint = ""
            if rc == 0 and obj == sha:
                hint = " -- that is the *tag object*, and the tools key on the "
                hint += "commit"
            number = start + text[: match.start()].count("\n")
            rows.append(
                f"{kind_for(path)} {path}:{number}: cites `{tag}` -> `{sha}`, "
                f"but {tag} points at commit `{commit}`{hint}"
            )
    return rows


def paragraphs_of(lines):
    """Blank-line separated blocks, each tagged with the heading it sits under.

    A document's history section ("What has landed", "Verification", an archive
    note) records what was true on a day, and it stays wrong on purpose for the
    same reason the ledger does. The heading tells the two apart without guessing
    at the prose.
    """
    out, buf, start, heading = [], [], None, ""
    for number, line in enumerate(lines, 1):
        if line.startswith("#"):
            if buf:
                out.append((start, "\n".join(buf), heading))
                buf, start = [], None
            heading = line.lstrip("#").strip()
            continue
        if line.strip():
            if start is None:
                start = number
            buf.append(line)
        elif buf:
            out.append((start, "\n".join(buf), heading))
            buf, start = [], None
    if buf:
        out.append((start, "\n".join(buf), heading))
    return out


def is_history(heading):
    return bool(HISTORY_HEADING.search(heading or ""))


def delimiters(line):
    """Unescaped `|` in a line. GFM splits a table row on these only, and a pipe
    inside an inline code span splits it too unless it is written `\\|` -- which is
    why `docs/release-process.md`'s `pgrep` row has nine of them and is correct."""
    count, escaped = 0, False
    for char in line:
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "|":
            count += 1
    return count


def check_tables(path, text):
    """A row with the wrong number of cells does not fail anything: the renderer
    silently drops the extra columns or pads the short ones, so the table looks
    fine and says something else. Written while fixing a document, this is the
    defect that was introduced and caught only by reading the line back."""
    rows = []
    lines = text.splitlines()
    block = []
    for number, line in enumerate(lines + [""], 1):
        if line.lstrip().startswith("|"):
            block.append((number, line))
            continue
        # A table is a header, a separator row, then body rows.
        if len(block) >= 2 and re.fullmatch(r" *\|[\s:|-]+\| *", block[1][1]):
            want = delimiters(block[0][1])
            if want >= 2:
                for row_number, row_line in block:
                    got = delimiters(row_line)
                    if got != want:
                        rows.append(
                            f"{kind_for(path)} {path}:{row_number}: a "
                            f"{want - 1}-column table has a row with {got - 1} "
                            "columns; the renderer will drop or pad it in silence"
                        )
        block = []
    return rows


def shared_group_key():
    """AUD-170: the shipped group key is one value written in two languages.

    The fleet manager is Swift, the harness plugin is JavaScript, and one Mac runs
    both against the same LAN -- so the default that lets an unconfigured install
    join the group has to be the *same string* in each. It lives at two definitional
    sites, `FleetGroupKey.shippedDefault` and `DEFAULT_GROUP_KEY`, and only prose
    said so. Move one of them and every other member reads a peer reporting a group
    that is not ours: a silently empty peer table, not an error.

    Derived rather than asserted, and it fails when it cannot read. A check that
    found nothing and reported a pass is the exact defect this gate hunts.
    """
    sites = (
        (
            "sources/TinyTitanFleet/Core/FleetGroupKey.swift",
            'static let shippedDefault = "([^"]+)"',
        ),
        (
            "plugins/dsh-lan-manager/src/config.js",
            'export const DEFAULT_GROUP_KEY = "([^"]+)"',
        ),
    )
    rows = []
    found = {}
    for rel, pattern in sites:
        full = os.path.join(ROOT, rel)
        try:
            with open(full, encoding="utf-8") as handle:
                body = handle.read()
        except (OSError, UnicodeDecodeError) as error:
            rows.append(f"FAIL cannot read {rel} for the group key: {error}")
            continue
        match = re.search(pattern, body)
        if match is None:
            rows.append(f"FAIL {rel}: no shipped default group key matched {pattern}")
            continue
        found[rel] = match.group(1)
    if len(found) == len(sites) and len(set(found.values())) > 1:
        joined = "; ".join(f"{rel} = {value!r}" for rel, value in sorted(found.items()))
        rows.append(f"FAIL the shipped group key has two values: {joined}")
    return rows


def main():
    modes, checks, problems = gate_set()
    docs, everything, listing = tracked_docs()
    rows = list(problems) + list(listing)
    totals, ledgers, ledger_errors = ledger_totals()
    rows += ledger_errors
    rows += shared_group_key()
    if not checks:
        rows.append("FAIL tools/lint.sh: derived no checks from the all chain")

    scored = set(docs)
    for path in everything:
        full = os.path.join(ROOT, path)
        try:
            with open(full, encoding="utf-8") as handle:
                text = handle.read()
        except (OSError, UnicodeDecodeError) as error:
            rows.append(f"FAIL cannot read {path}: {error}")
            continue
        rows += check_tables(path, text)
        if path not in scored:
            continue
        paragraphs = paragraphs_of(text.splitlines())
        rows += check_counts(path, paragraphs, len(checks))
        rows += check_names(path, paragraphs, modes)
        rows += check_ledger_counts(path, paragraphs, totals, ledgers)
        rows += check_derived_counts(path, paragraphs)
        rows += check_tag_citations(path, paragraphs)

    for row in rows:
        print(row)
    fails = [r for r in rows if r.startswith("FAIL")]
    owner = [r for r in rows if r.startswith("OWNER")]
    summary = (
        f"{len(docs)} documents against {len(checks)} gates derived "
        f"from tools/lint.sh, table shape in all {len(everything)}; "
        f"{len(owner)} owner-file note(s) reported and not enforced"
    )
    if fails:
        print(f"  FAIL: {len(fails)} documented fact(s) disagree with the repository ({summary})")
        return 1
    print(f"  ok ({summary})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
