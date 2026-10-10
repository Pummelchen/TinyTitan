#!/usr/bin/env python3
"""Turn a release-notes file into its compact form.

The full notes in `docs/release-notes-vX.Y.md` are the record: they explain why
each change exists and carry the verification detail. The Release page is read
far more casually, so `tools/release.sh --publish` puts THIS form there instead:
the same claims, as bullets, one sentence each, wrapped narrow.

    python3 tools/compact-release-notes.py docs/release-notes-v5.5.md
    python3 tools/compact-release-notes.py in.md --out out.md --require ornith-4

What it does, per section:

  * the `## ` title and every `### ` heading are kept verbatim;
  * the lead block stays a paragraph -- the notes shape asks for exactly one
    there -- unless it is already bullets, which are then kept as bullets;
  * prose paragraphs become one bullet per sentence, so nothing is reworded,
    only re-laid-out. A fragment that starts lower-case is folded back into the
    bullet above it, because the sentence rule splits "e.g. the author" apart;
  * a bullet's wrapped continuation lines stay part of that bullet -- a
    2-space-indented line is not a new item;
  * fences, table rows and blockquotes travel line for line, like `###
    Checksum`. Their line breaks ARE their content: an example of a line format
    that arrives as one sentence, or a table that arrives as a paragraph, is not
    the same claim re-laid-out -- it is the claim destroyed. release.sh's
    `--require` cannot see this, because every token survives the merge.
  * a numbered item keeps its enumerator inside the bullet (`- 1. Verify the
    checksum ...`): the number IS the order the claim makes, and the sentence
    rule used to cut right after it, publishing `- 1.` as an item of its own.
    A nested item is a new bullet rather than more of its parent's sentence --
    the second level is flattened, the claim stays separate.

`--require TOKEN` (repeatable) makes the script fail when TOKEN is absent from
its own output. release.sh passes every string `--publish` greps the notes for,
so a compaction that would drop one dies here rather than at publish time. That
check is the point: the alternative is discovering the omission after the
Release is public.

Idempotent: running it on its own output changes nothing. `--max-chars` fails
when the compact form is still over budget, which is how "compact" stays true
for the next release rather than only the one it was written for.
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap

WRAP = 100
CHECKSUM_HEADING = "### Checksum"
BULLET = re.compile(r"^[-*]\s+")
ORDERED = re.compile(r"^\d+\.\s+")

# Splitting on (?<=[.!?])\s+ alone breaks these apart. Only abbreviations this
# repo's notes actually use are listed.
ABBREVIATIONS = (
    "e.g.",
    "i.e.",
    "cf.",
    "vs.",
    "etc.",
    "resp.",
    "approx.",
    "no.",
    "fig.",
    "al.",
    "Dr.",
    "Mr.",
    "Ms.",
    "St.",
)


def split_sentences(paragraph: str) -> list[str]:
    """Split prose into sentences, respecting known abbreviations and versions."""
    protected = paragraph
    holes: list[str] = []

    for token in ABBREVIATIONS:
        while token in protected:
            holes.append(token)
            protected = protected.replace(token, f"\x00{len(holes) - 1}\x00", 1)
    while True:
        found = re.search(r"\d\.\d", protected)
        if not found:
            break
        holes.append(found.group(0))
        protected = (
            protected[: found.start()] + f"\x00{len(holes) - 1}\x00" + protected[found.end() :]
        )

    sentences = []
    for part in re.split(r"(?<=[.!?])\s+", protected):
        for index, hole in enumerate(holes):
            part = part.replace(f"\x00{index}\x00", hole)
        part = " ".join(part.split())
        if part:
            sentences.append(part)
    return sentences


def wrap_bullet(text: str, width: int) -> list[str]:
    return textwrap.wrap(
        text, width=width, initial_indent="- ", subsequent_indent="  ", break_long_words=False
    ) or [""]


def main() -> int:
    ap = argparse.ArgumentParser(description="Compact a release-notes file.")
    ap.add_argument("notes", help="the full release-notes markdown")
    ap.add_argument("--out", help="write here instead of stdout")
    ap.add_argument("--width", type=int, default=WRAP, help=f"wrap column (default {WRAP})")
    ap.add_argument(
        "--max-chars",
        type=int,
        default=0,
        help="fail when the compact form exceeds this many characters (0 = no limit)",
    )
    ap.add_argument(
        "--require",
        action="append",
        default=[],
        metavar="TOKEN",
        help="fail unless TOKEN survives into the compact output (repeatable)",
    )
    args = ap.parse_args()

    try:
        with open(args.notes, encoding="utf-8") as handle:
            lines = handle.read().split("\n")
    except OSError as exc:
        print(f"error: cannot read {args.notes}: {exc}", file=sys.stderr)
        return 2

    out: list[str] = []
    mode = "lead"  # lead | section | checksum
    lead: list[str] = []
    prose: list[str] = []
    bullet: list[str] = []

    def blank() -> None:
        if out and out[-1] != "":
            out.append("")

    def flush_bullet() -> None:
        if not bullet:
            return
        text = " ".join(" ".join(bullet).split())
        bullet.clear()
        out.extend(wrap_bullet(text, args.width))

    def flush_prose() -> None:
        if not prose:
            return
        text = " ".join(" ".join(prose).split())
        prose.clear()
        for sentence in split_sentences(text):
            # Fold a fragment back into the bullet above rather than give it a
            # bullet of its own.
            if sentence[:1].islower() and fold_into_previous_bullet(sentence):
                continue
            out.extend(wrap_bullet(sentence, args.width))

    def flush_lead() -> None:
        """The lead block: a paragraph, or bullets when it already is one."""
        if not lead:
            return
        items = list(lead)
        lead.clear()
        if any(BULLET.match(item) for item in items):
            for item in items:
                out.extend(wrap_bullet(BULLET.sub("", item).strip(), args.width))
        else:
            text = " ".join(" ".join(items).split())
            out.extend(textwrap.wrap(text, width=args.width, break_long_words=False))

    def flush_all() -> None:
        flush_bullet()
        flush_prose()

    def fold_into_previous_bullet(sentence: str) -> bool:
        """Append a lower-case fragment to the bullet above it, marker included once.

        The bullet's own head is not the last line whenever the bullet wrapped,
        which is the usual case at the default width -- so walk back over the
        continuation lines first, then take the head's text, not its marker.
        """
        head = len(out) - 1
        while head >= 0 and out[head].startswith("  "):
            head -= 1
        if head < 0 or not out[head].startswith("- "):
            return False
        lines = out[head:]
        del out[head:]
        text = " ".join([lines[0][2:]] + [line.strip() for line in lines[1:]])
        out.extend(wrap_bullet(f"{text} {sentence}", args.width))
        return True

    def open_block(marker: str) -> None:
        """Start or continue a block whose line breaks are its content."""
        nonlocal verbatim
        flush_lead() if mode == "lead" else flush_all()
        if not (verbatim and out and out[-1].startswith(marker)):
            blank()
        verbatim = True

    def close_block() -> None:
        nonlocal verbatim
        if verbatim:
            blank()
            verbatim = False

    in_fence = False
    verbatim = False
    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()

        if in_fence:
            out.append(stripped)
            if stripped.startswith("```"):
                in_fence = False
                close_block()
            continue

        if stripped.startswith("```"):
            flush_lead() if mode == "lead" else flush_all()
            blank()
            out.append(stripped)
            in_fence = True
            verbatim = True
            continue

        if stripped.startswith(("|", ">")):
            open_block(stripped[0])
            out.append(stripped)
            continue

        indented = line[:1] in (" ", "\t")
        close_block()

        if stripped.startswith("### "):
            flush_lead() if mode == "lead" else flush_all()
            blank()
            mode = "checksum" if stripped == CHECKSUM_HEADING else "section"
            out.append(stripped)
            blank()
            continue

        if stripped.startswith("## "):
            flush_lead() if mode == "lead" else flush_all()
            blank()
            mode = "lead"
            out.append(stripped)
            blank()
            continue

        if mode == "checksum":
            out.append(line)
            continue

        if not stripped:
            flush_lead() if mode == "lead" else flush_all()
            continue

        if mode == "lead":
            lead.append(stripped)
            continue

        # A marker decides whether this is a new item, whatever its indentation: an
        # indented `- ` line is a second level, not more of the parent's sentence.
        if BULLET.match(stripped) or ORDERED.match(stripped):
            flush_prose()
            flush_bullet()
            # A numbered item keeps its enumerator in the text: the number IS the
            # order the claim makes, and the sentence rule cuts right after it.
            bullet.append(BULLET.sub("", stripped) if BULLET.match(stripped) else stripped)
            continue

        if indented:
            # A continuation of whatever is open: a bullet's wrapped remainder,
            # or a paragraph's.
            if bullet:
                bullet.append(stripped)
            else:
                prose.append(stripped)
            continue

        flush_bullet()
        prose.append(stripped)

    flush_lead() if mode == "lead" else flush_all()

    compacted: list[str] = []
    for line in out:
        if line == "" and compacted and compacted[-1] == "":
            continue
        compacted.append(line)
    while compacted and compacted[-1] == "":
        compacted.pop()
    body = "\n".join(compacted) + "\n"

    missing = [token for token in args.require if token not in body]
    if missing:
        for token in missing:
            print(
                f"error: compaction dropped required token {token!r}; "
                "these notes cannot be published from this form",
                file=sys.stderr,
            )
        return 1

    if args.max_chars and len(body) > args.max_chars:
        print(
            f"error: the compact notes are {len(body)} characters, over the "
            f"{args.max_chars} budget; tighten the notes rather than the wrap width",
            file=sys.stderr,
        )
        return 1

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(body)
        print(
            f"compacted {args.notes} -> {args.out} "
            f"({len(lines)} -> {len(body.splitlines())} lines, {len(body)} chars)"
        )
    else:
        sys.stdout.write(body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
