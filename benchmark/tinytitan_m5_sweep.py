#!/usr/bin/env python3
"""TinyTitan tuning sweep for M5-class hardware.

Measures prefill seconds and decode tok/s across expert-cache slot counts and
prefill chunk sizes, using the CLI's timing footer. The two sweeps answer the
questions raised in the M5 feedback:

  slots  — does decode still climb past 32/64 on higher-bandwidth hardware?
  chunk  — is smaller or larger prefill better at your prompt length?

Each run loads the model fresh (cold page cache), so runs are minutes long:
the default sweeps are 6 slots + 5 chunks = 11 runs. Pick one sweep to halve
that, or trim the lists with --slots / --chunks.

Usage:
  python3 benchmark/tinytitan_m5_sweep.py
  python3 benchmark/tinytitan_m5_sweep.py --model ... --sweep slots
  python3 benchmark/tinytitan_m5_sweep.py --model ... --sweep chunk --prompt-tokens 22800
  python3 benchmark/tinytitan_m5_sweep.py --model ... --slots 32,64,128 --chunks 512,4096

Requires a release build: swift build -c release

The exit status is part of the result: 0 only when every planned run measured and
the CSV was written, 1 when a run errored or the sweep was interrupted or the file
could not be written, and 2 for a refusal that costs no run at all. Set
TINYTITAN_M5_SWEEP_OUT to a directory, or to a path ending in .csv, to move the
results; the default is benchmark/benchmark-results/.
"""

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time

from tinytitan_profile import DEFAULT_CONTEXT_TOKENS, DEFAULT_MODEL_PATH

DEFAULT_SLOTS = [16, 24, 32, 64, 96, 128]
DEFAULT_CHUNKS = [256, 512, 1024, 2048, 4096]

OUT_ENV = "TINYTITAN_M5_SWEEP_OUT"
ARTIFACT = "m5_sweep_results.csv"

# Each sweep varies one control and pins the other, so the two headers can name
# the pinned value rather than leaving the reader to infer it from the rows.
SLOT_SWEEP_CHUNK = 4096
CHUNK_SWEEP_SLOTS = 64

# Mirrors RuntimeConfiguration.supportedContextTokens. The CLI rejects any
# other value, so the sweep picks from this list rather than passing an
# arbitrary size.
SUPPORTED_CONTEXTS = [4096, 8192, 16384, 32768, 65536, 131072, 262144]

# Headroom over prompt + generated tokens, covering the ChatML wrapper and
# special tokens that make_prompt cannot account for.
CONTEXT_HEADROOM_TOKENS = 512

# A run that produces no output in this long is wedged; fail it and continue
# rather than hanging the whole sweep. Generous: a cold 128k-token prefill on
# slow hardware is legitimately many minutes.
RUN_TIMEOUT_S = 3600

# A repeatable ~50-word paragraph; tokens/word is roughly 1.3 for English, so
# 40 words ≈ 50 tokens. The script repeats it to reach the target token count.
PARAGRAPH = (
    "The server streams routed expert weights from the model store on demand, "
    "keeping only a resident working window, which is what lets it run within a "
    "bounded memory budget while retaining the full model quality at decode time. "
    "The chunked prefill path amortizes per-chunk route readbacks across long "
    "prompts, and the expert cache raises the hit rate for repeated generations. "
)


def make_prompt(target_tokens: int) -> str:
    words_per_block = len(PARAGRAPH.split())
    approx_tokens_per_block = int(words_per_block * 1.3)
    blocks = max(1, (target_tokens + approx_tokens_per_block - 1) // approx_tokens_per_block)
    return (PARAGRAPH * blocks).strip()


class ConfigError(ValueError):
    """A refusal the operator gets before any of the minute-long runs starts."""


def parse_sizes(value: str, flag: str) -> list:
    """A comma-separated list of positive sizes, or a refusal naming the flag."""
    try:
        values = [int(v) for v in value.split(",")]
    except ValueError as error:
        raise ConfigError(
            "%s wants comma-separated positive integers, got %r" % (flag, value)
        ) from error
    if min(values) < 1:
        raise ConfigError("%s wants positive integers, got %r" % (flag, value))
    return values


def plan_runs(sweep: str, slots: list, chunks: list) -> list:
    """Every run the sweep will execute, before any of them has run.

    The count is what the status and the footer are measured against; deriving it
    from the rows that survived would let a sweep that ran nothing report 0 of 0.
    """
    planned = []
    if sweep in ("slots", "both"):
        planned += [("slot", s, SLOT_SWEEP_CHUNK) for s in slots]
    if sweep in ("chunk", "both"):
        planned += [("chunk", CHUNK_SWEEP_SLOTS, c) for c in chunks]
    return planned


def validate_context(prompt_tokens: int, max_new: int, max_context: int) -> int:
    """Reject a selected context that cannot hold the estimated request."""
    need = prompt_tokens + max_new + CONTEXT_HEADROOM_TOKENS
    if max_context < need:
        raise ConfigError(
            "--prompt-tokens %d + --max-new %d needs ~%d tokens of context, "
            "above --max-context %d" % (prompt_tokens, max_new, need, max_context)
        )
    return max_context


def run_once(
    cli: str, model: str, slots: int, chunk: int, prompt_file: str, max_new: int, max_context: int
) -> dict:
    cmd = [
        cli,
        "--model",
        model,
        "--expert-cache-slots",
        str(slots),
        "--prefill-chunk",
        str(chunk),
        "--messages-file",
        prompt_file,
        "--max-context",
        str(max_context),
        "--rope-scaling",
        "none",
        "--kv-bits",
        "8",
        "--max-new",
        str(max_new),
    ]
    # Every return below carries slots/chunk, so a failed run is still
    # attributable to its configuration in the CSV.
    tags = {"slots": slots, "chunk": chunk}
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=RUN_TIMEOUT_S, check=False
        )
    except subprocess.TimeoutExpired:
        return {**tags, "error": "timed out after %ds" % RUN_TIMEOUT_S}
    elapsed = time.time() - t0
    if proc.returncode != 0:
        detail = (
            proc.stderr.strip().splitlines()[-1] if proc.stderr else "exit %d" % proc.returncode
        )
        return {**tags, "error": detail}
    footer = proc.stderr
    m = re.search(
        r"prefill=(\d+)tok/([\d.]+)s new=(\d+)tok decode=([\d.]+)s tok/s=([\d.]+)", footer
    )
    if not m:
        return {**tags, "error": "no timing footer in output"}
    return {
        "slots": slots,
        "chunk": chunk,
        "prefill_tokens": int(m.group(1)),
        "prefill_s": float(m.group(2)),
        "new_tokens": int(m.group(3)),
        "decode_s": float(m.group(4)),
        "tok_per_s": float(m.group(5)),
        "wall_s": round(elapsed, 1),
    }


def repository_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def results_path(env=None) -> pathlib.Path:
    """Where the CSV goes: TINYTITAN_M5_SWEEP_OUT, else the repository's results dir.

    A relative value resolves against the repository, not the working directory, so
    a sweep launched from elsewhere still lands where the operator said and not in
    whatever directory happened to be current.
    """
    mapping = os.environ if env is None else env
    value = str(mapping.get(OUT_ENV, "")).strip()
    root = repository_root()
    if not value:
        return pathlib.Path(root, "benchmark", "benchmark-results", ARTIFACT)
    path = pathlib.Path(value).expanduser()
    if not path.is_absolute():
        path = pathlib.Path(root, path)
    if path.suffix.lower() != ".csv":
        path = path / ARTIFACT
    return path


def results_csv(results: list) -> str:
    header = "slots,chunk,prefill_s,prefill_tokens,tok_per_s,new_tokens,decode_s,wall_s,error\n"
    rows = []
    for r in results:
        if "error" in r:
            rows.append(
                ",".join(
                    [
                        str(r.get("slots", "")),
                        str(r.get("chunk", "")),
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        r["error"],
                    ]
                )
            )
        else:
            rows.append(
                "%d,%d,%.2f,%d,%.2f,%d,%.2f,%.1f,"
                % (
                    r["slots"],
                    r["chunk"],
                    r["prefill_s"],
                    r["prefill_tokens"],
                    r["tok_per_s"],
                    r["new_tokens"],
                    r["decode_s"],
                    r["wall_s"],
                )
            )
    return header + "".join(row + "\n" for row in rows)


def write_results(results: list, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(results_csv(results), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    ap.add_argument(
        "--cli", default=None, help="path to TinyTitanCLI (default: .build/release/TinyTitanCLI)"
    )
    ap.add_argument("--sweep", choices=["slots", "chunk", "both"], default="both")
    ap.add_argument("--slots", default=",".join(map(str, DEFAULT_SLOTS)))
    ap.add_argument("--chunks", default=",".join(map(str, DEFAULT_CHUNKS)))
    ap.add_argument("--prompt-tokens", type=int, default=8192)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument(
        "--max-context", type=int, choices=SUPPORTED_CONTEXTS, default=DEFAULT_CONTEXT_TOKENS
    )
    args = ap.parse_args()

    cli = args.cli or os.path.join(repository_root(), ".build", "release", "TinyTitanCLI")
    if not os.path.isfile(cli):
        print("REFUSED: TinyTitanCLI not found at %s — run `swift build -c release` first" % cli)
        return 2

    try:
        slots = parse_sizes(args.slots, "--slots")
        chunks = parse_sizes(args.chunks, "--chunks")
        max_context = validate_context(args.prompt_tokens, args.max_new, args.max_context)
    except ConfigError as error:
        print("REFUSED: %s" % error)
        return 2

    planned = plan_runs(args.sweep, slots, chunks)
    prompt_directory = os.path.join(repository_root(), ".build", "benchmark-prompts")
    os.makedirs(prompt_directory, exist_ok=True)
    prompt_file = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", suffix=".json", delete=False, dir=prompt_directory
    ).name
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(json.dumps([{"role": "user", "content": make_prompt(args.prompt_tokens)}]))

    print("TinyTitan M5 sweep")
    print("  model: %s" % args.model)
    print(
        "  prompt: ~%d tokens | max-new: %d | context: %d | first run is cold"
        % (args.prompt_tokens, args.max_new, max_context)
    )
    print("  planned runs: %d" % len(planned))
    print()

    results = []
    interrupted = False
    shown = None
    try:
        for kind, s, c in planned:
            if kind != shown:
                if shown is not None:
                    print()
                if kind == "slot":
                    print("== slot sweep (chunk %d) ==" % SLOT_SWEEP_CHUNK)
                else:
                    print("== chunk sweep (slots %d) ==" % CHUNK_SWEEP_SLOTS)
                shown = kind
            r = run_once(cli, args.model, s, c, prompt_file, args.max_new, max_context)
            results.append(r)
            label = "slots=%3d" % s if kind == "slot" else "chunk=%4d" % c
            if "error" in r:
                print("  %s  ERROR: %s" % (label, r["error"]))
            else:
                print(
                    "  %s  prefill %6.1fs (%4d tok)  decode %6.2f tok/s (%3d tok)  wall %5.1fs"
                    % (
                        label,
                        r["prefill_s"],
                        r["prefill_tokens"],
                        r["tok_per_s"],
                        r["new_tokens"],
                        r["wall_s"],
                    )
                )
        print()
    except KeyboardInterrupt:
        # A full sweep is 11 cold runs. Keep whatever completed rather than
        # discarding an hour of measurements on Ctrl-C.
        interrupted = True
        print("\ninterrupted — writing partial results")
    finally:
        try:
            os.unlink(prompt_file)
        except OSError:
            pass

    # make_prompt sizes the prompt by a words-to-tokens estimate; the footer
    # reports what the tokenizer actually produced. Surface the gap so a sweep
    # is never silently run at a different length than requested.
    measured = [r for r in results if "prefill_tokens" in r]
    if measured:
        actual = measured[0]["prefill_tokens"]
        drift = abs(actual - args.prompt_tokens) / max(args.prompt_tokens, 1)
        note = "  (estimate off by %.0f%%)" % (drift * 100) if drift > 0.05 else ""
        print(
            "prompt: requested ~%d tokens, tokenizer produced %d%s"
            % (args.prompt_tokens, actual, note)
        )

    path = results_path()
    write_error = None
    try:
        write_results(results, path)
    except OSError as error:
        write_error = str(error)

    status = 0
    missing = len(planned) - len(measured)
    if interrupted:
        print("interrupted: %d run(s) never attempted" % (len(planned) - len(results)))
        status = 1
    if missing or write_error is not None:
        status = 1
    verdict = "SWEEP COMPLETE" if status == 0 else "NOT MEASURED"
    if write_error is None:
        print(
            "%s: %d of %d run(s) measured, written to %s"
            % (verdict, len(measured), len(planned), path)
        )
    else:
        print(
            "%s: %d of %d run(s) measured, results NOT WRITTEN: %s"
            % (verdict, len(measured), len(planned), write_error)
        )
    return status


if __name__ == "__main__":
    sys.exit(main())
