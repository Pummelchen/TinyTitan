#!/usr/bin/env python3
"""Expert-cache slot sweep: hit rate, bytes/token and memory (TT-011).

The proposal was to raise the Qwen3.8-Flash-Next slot budget from the shipped
96 slots (12 GiB) to 128 (15.8 GiB), on the strength of a *simulated* LRU trace
that put the hit rate at 65% for 64 slots and 78% at 128. The runtime's own
counters tell a different story -- 81.3% at 96 slots on a short prompt -- so the
decision needs the runtime's numbers, not the trace's.

`TinyTitanCLI --expert-cache-slots N` sets the count directly (bypassing the
half-of-RAM affordability cap in `decodeTuning`), and the CLI prints
`[decode expert io] hits H misses M (X% hit) B GiB = MiB/token`. Memory comes
from `/usr/bin/time -l` (maximum resident set size) wrapped around the process.

Arms run ascending then descending (64, 96, 128, 128, 96, 64) so drift lands on
both ends. The prompt is a fixed-length slice of a repository document, and the
run's own `prefill=NNNtok` is the length that gets recorded, not the character
count.

  python3 benchmark/expert_cache_slots.py --slots 64,96,128 --rounds 2 --record
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLI = ROOT / ".build/release/TinyTitanCLI"
RESULTS = ROOT / "benchmark/expert-cache"

DEFAULT_SOURCE = "docs/qwen38-flash-next-port.md"
INSTRUCTION = (
    "\n\nContinue this document, writing at least 300 more words in the same "
    "style. Do not summarize and do not stop early.\n"
)

IO = re.compile(
    r"\[decode expert io\] hits (\d+) misses (\d+) \(([\d.]+)% hit\) "
    r"([\d.]+) GiB(?: = ([\d.]+) MiB/token)?"
)
FOOTER = re.compile(
    r"\[stop=(\S+) prefill=(\d+)tok/([\d.]+)s new=(\d+)tok "
    r"decode=([\d.]+)s tok/s=([\d.]+)\]"
)
MAXRSS = re.compile(r"(\d+)\s+maximum resident set size")


def build_prompt(source: pathlib.Path, characters: int) -> tuple[str, str]:
    text = source.read_text(encoding="utf-8")
    body = text[:characters]
    digest = hashlib.sha256(body.encode()).hexdigest()[:16]
    return body + INSTRUCTION, digest


def swap_used_gib() -> float:
    out = subprocess.run(
        ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=False
    ).stdout
    match = re.search(r"used = ([\d.]+)M", out)
    return float(match.group(1)) / 1024.0 if match else float("nan")


def run_once(model: pathlib.Path, messages: pathlib.Path, slots: int, max_new: int) -> dict:
    before = swap_used_gib()
    # `TINYTITAN_DECODE_IO_TRACE` is what makes the runner take a statistics
    # baseline; without it `decodeExpertIO()` returns nil and the CLI prints no
    # `[decode expert io]` line at all.
    env = dict(os.environ)
    env["TINYTITAN_DECODE_IO_TRACE"] = "1"
    proc = subprocess.run(
        [
            "/usr/bin/time",
            "-l",
            str(CLI),
            "--model",
            str(model),
            "--messages-file",
            str(messages),
            "--expert-cache-slots",
            str(slots),
            "--max-new",
            str(max_new),
            "--temperature",
            "0",
        ],
        env=env,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )
    after = swap_used_gib()
    err = proc.stderr
    row: dict = {
        "slots": slots,
        "swap_before_gib": round(before, 2),
        "swap_after_gib": round(after, 2),
        "swap_delta_gib": round(after - before, 2),
    }
    io = IO.search(err)
    if io:
        hits, misses, hit_pct, gib, per_token = io.groups()
        row.update(
            hits=int(hits),
            misses=int(misses),
            hit_pct=float(hit_pct),
            read_gib=float(gib),
            mib_per_token=float(per_token) if per_token else None,
        )
    footer = FOOTER.search(err)
    if footer:
        stop, prefill_tokens, prefill_s, new, decode_s, rate = footer.groups()
        row.update(
            stop=stop,
            prompt_tokens=int(prefill_tokens),
            prefill_s=float(prefill_s),
            new_tokens=int(new),
            decode_s=float(decode_s),
            decode_tok_s=float(rate),
        )
    rss = MAXRSS.search(err)
    if rss:
        row["max_rss_gib"] = round(int(rss.group(1)) / 1_073_741_824, 2)
    if not footer:
        row["failed"] = err[-400:]
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/qwen3.8-flash-next_125B_A6B_4Bit")
    parser.add_argument("--slots", default="64,96,128")
    parser.add_argument(
        "--rounds", type=int, default=2, help="each round is the sweep ascending then descending"
    )
    parser.add_argument(
        "--characters",
        type=int,
        default=24_000,
        help="prompt body slice; ~4,500 tokens at ~5.3 chars/token",
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--label", default="qwen38-4bit")
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()

    model = (ROOT / args.model).resolve()
    if not (model / "verified-install.json").exists():
        raise SystemExit(f"not an installed model: {model}")
    busy = subprocess.run(
        ["pgrep", "-fl", "TinyTitanCLI|TinyTitanServer"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if busy:
        raise SystemExit(f"a model process is already running:\n{busy}")

    slots = [int(s) for s in args.slots.split(",")]
    manifest = json.loads((model / "manifest.json").read_text())
    stride = int(manifest["expertStride"])
    layers = int(manifest["arch"]["numLayers"])
    footprint = {s: stride * layers * s / 1_073_741_824 for s in slots}
    prompt, digest = build_prompt(ROOT / args.source, args.characters)
    messages = ROOT / ".build/tt011-messages.json"
    messages.parent.mkdir(parents=True, exist_ok=True)
    messages.write_text(json.dumps([{"role": "user", "content": prompt}]))

    rows: list[dict] = []
    print(
        f"[tt011] {model.name}: slots {slots}, prompt {len(prompt)} chars "
        f"(sha {digest}), {args.max_new} new tokens, swap now {swap_used_gib():.2f} GiB",
        flush=True,
    )
    for round_index in range(args.rounds):
        order = slots if round_index % 2 == 0 else list(reversed(slots))
        for slot_count in order:
            row = run_once(model, messages, slot_count, args.max_new)
            rows.append(row)
            if row.get("failed"):
                print(f"[{slot_count:>3} slots] FAILED: {row['failed']}", flush=True)
                continue
            print(
                f"[{slot_count:>3} slots] cache {footprint[slot_count]:5.2f} GiB  "
                f"hit {row.get('hit_pct', 0):5.1f}%  "
                f"{row.get('decode_tok_s', 0):5.2f} tok/s  "
                f"io {row.get('mib_per_token', 0):6.1f} MiB/tok  "
                f"rss {row.get('max_rss_gib', 0):5.2f} GiB  "
                f"swap {row.get('swap_delta_gib', 0):+5.2f} GiB",
                flush=True,
            )

    ok = [r for r in rows if not r.get("failed")]
    print("\n" + "=" * 78)
    print(
        f"EXPERT-CACHE SLOT SWEEP — {model.name}, {ok[0]['prompt_tokens'] if ok else '?'} "
        f"prompt tokens, {args.max_new} new tokens, greedy"
    )
    print("=" * 78)
    print(
        f"  {'slots':>5} {'cache GiB':>9} {'hit %':>7} {'tok/s':>7} "
        f"{'MiB/token':>10} {'max RSS':>8} {'swap Δ':>7}"
    )
    for slot_count in slots:
        group = [r for r in ok if r["slots"] == slot_count]
        if not group:
            continue

        def med(key, group=group):
            return statistics.median([r[key] for r in group if r.get(key) is not None])

        print(
            f"  {slot_count:>5} {footprint[slot_count]:>9.2f} {med('hit_pct'):>7.1f} "
            f"{med('decode_tok_s'):>7.2f} {med('mib_per_token'):>10.1f} "
            f"{med('max_rss_gib'):>8.2f} {med('swap_delta_gib'):>+7.2f}"
        )
    by_slot = {s: [r for r in ok if r["slots"] == s] for s in slots}
    for lower, higher in zip(slots, slots[1:], strict=False):
        lo = [r["hit_pct"] for r in by_slot[lower] if "hit_pct" in r]
        hi = [r["hit_pct"] for r in by_slot[higher] if "hit_pct" in r]
        if lo and hi:
            print(
                f"  {higher} slots vs {lower}: hit "
                f"{statistics.median(hi) - statistics.median(lo):+.1f} points, "
                f"tok/s {statistics.median([r['decode_tok_s'] for r in by_slot[higher]]) - statistics.median([r['decode_tok_s'] for r in by_slot[lower]]):+.2f}"
            )

    if args.record:
        RESULTS.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M")
        out = RESULTS / f"slots-{args.label}-{stamp}.json"
        out.write_text(
            json.dumps(
                {
                    "model": model.name,
                    "prompt_sha": digest,
                    "source": args.source,
                    "characters": args.characters,
                    "max_new": args.max_new,
                    "slots": slots,
                    "rows": rows,
                },
                indent=2,
            )
        )
        print(f"\nwrote {out.relative_to(ROOT)}")
    try:
        messages.unlink()
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
