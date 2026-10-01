# TinyTitan

<!-- agent-harnesses:begin -->
> **One instruction file.** This is it. Codex, DeepSeek Harness, OpenCode,
> Qwen Code, Qoder and Zed read `AGENTS.md` directly, and Claude Code reads it
> through the committed `CLAUDE.md`, which contains nothing but `@AGENTS.md`.
> **Edit only this file** — do not add a second set of instructions anywhere.
>
> Do **not** add `.rules`, `.cursorrules`, `.windsurfrules`, `.clinerules`,
> `.github/copilot-instructions.md` or `AGENT.md`. Zed takes the *first match*
> from that list, **ahead of `AGENTS.md`**, so any one of them silently
> replaces this file for every Zed user.
<!-- agent-harnesses:end -->

Swift and Metal inference for Qwen-family MoE and dense text models on Apple
Silicon, streaming routed experts from SSD so a model larger than RAM still runs.
Supports 4-bit and 8-bit builds of Qwen3.8-Flash-Next 125B-A6B,
KAT-Coder-V2.5-Dev 35B-A3B, Qwen-AgentWorld 35B-A3B, Ornith 1.5 35B-A3B and
Qwen 3.6 35B-A3B, plus the dense Qwen 3.5 2B/4B/9B on either engine. **The README
is the authority on which checkpoints and widths are supported** — it changes more
often than this file, so do not treat any list here as the current matrix.
`Qwen 3.5 <size> <bits>` (keys `qwen35-2b/4b/9b`) are dense: no routed experts,
nothing streamed from SSD. Ornith 1.5 8-bit is the default install and the default
golden target; 6-bit is a withdrawn legacy format.

## Working here

This is the development repository — changing source is the point, subject to the
gates below, and a change is finished when it is tested, not when it compiles.
What needs a reason is *optimization*: do not start performance work, change
runtime defaults, or alter numerics unless the user asked. Report results as
measurements, not as ceilings, and treat an unexplained slowdown as a finding.

**Scope: an LLM engine and its local server, and nothing else.** The supported way
to use a model is the loopback OpenAI-compatible server with a client the user
already has — Zed, Codex, Claude Code, DeepSeek Harness, `curl`. There is
deliberately no desktop, GUI or bundled app: a second front end is a second
surface to build, keep in step with every engine feature, and support, and it was
removed for exactly that reason. Do not add one back, and do not take on work that
only a GUI needs.

The **one** sanctioned convenience is a client we merely install and configure, not
build: `tools/dsh_local.sh` sets up a pinned, private DeepSeek Harness under
`~/.tinytitan` and the launcher's `--web` opens it in the browser, so a user who
wants a window gets one already pointed at their model. It is upstream's code with
our plugin, isolated from any DeepSeek Harness the user runs, and it is a client of
the loopback server like Zed or `curl` — not a front end this repository maintains.
Keep it that way: no forking the harness, no GUI code in this tree, and nothing
here may depend on a window existing.

## Layout and commands

`sources/` holds one directory per SwiftPM target. `sources/TinyTitan/` is the
runtime; `sources/TinyTitanFormat/` plus `sources/TinyTitanKernelsC/` are its
format types and C kernels. `sources/TinyTitanRepack/`, `sources/TinyTitanCLI/` and
`sources/TinyTitanServer/` hold the installer, CLI and loopback server; each of
those three is a thin executable over a `*Core` library target
(`TinyTitanRepackCore`, `TinyTitanCLICore`, `TinyTitanServerCore`).
`sources/TinyTitanMemory/` and `sources/ContinuityCore/` are persistent
agent memory, `sources/TinyTitanMemoryTool/` inspects it (executable
`tinytitan-memory`), `sources/ContinuityDemo/` is the memory demo executable,
`sources/TinyTitanFleet/` is the LAN manager (executable `ttlanmanager`), and
`sources/TinyTitanBench/` plus `sources/TinyTitanValidation/` are the benchmark
driver and the validation/reference target. An executable target keeps its
top-level or `@main` entry in `Command/`; `plugins/dsh-tinytitan/` is the DeepSeek
Harness bundle (route writer + quiet compaction). `docs/repository-layout.md` has
the conventions, `tests/` mirrors `sources/` path for path and never loads a model,
**user documentation lives in the
[GitHub Wiki](https://github.com/Pummelchen/TinyTitan/wiki) and engineering
documentation in this repository's `docs/`** — plans, measurements, design notes
and the handover, which the wiki does not carry. Open work is tracked
in **one** table; the standard, and the reusable prompt for other projects, is
`docs/task-table-standard.md`.

A file under `sources/` stays at 500 physical lines or fewer — comments and
blank lines included. The rule is scoped to the production sources: `tests/`
is organised by the suite each file covers and is not held to it, so an
oversized test file is split for readability, not to satisfy a number. A
production file over the limit is split along a cohesive seam (one type, one
phase, or one cluster of helpers) as pure code motion, with the public API and
import paths preserved; `docs/repository-layout.md` records the splits and the
`lint:allow-long` exemptions that a straight-line sequence keeps.

The wiki is a **separate repository** (`TinyTitan.wiki.git`, branch `master`,
usually cloned at the gitignored `.qwen/wiki`), and it is the project's user
documentation. It is not deployed from here: publishing is **two pushes** —
`main` to this repository and `master` to the wiki — and a request to push or
publish a change means both, whether or not the change touched the wiki.

```bash
swift build -c release
swift run -c release TinyTitanCLI \
  --model models/qwen3.5_4B_4Bit \
  --prompt "The capital of France is" \
  --max-new 64
```

`swift build -c release` is the portable build and works in a fresh clone.
`swift run … TinyTitanCLI` needs a model install under `models/`, which is
gitignored — a new checkout has none, so install one first
(`docs/adding-a-model.md`). The same holds for the repack and `--verify-install`
examples below.

That is the **portable** build and what `tools/release.sh` ships: the target is
`arm64-apple-macos26.0` with no CPU flag, so the codegen baseline is clang's
default for the triple — **apple-m1** — and an M3's BF16/I8MM go unused.

The C kernels are built `-O2`: `Package.swift` sets it for `TinyTitanKernelsC`
with `.unsafeFlags`, because SwiftPM's `swiftbuild` system otherwise compiles C
at `-Os` in release. That is the measured win — 2.35 -> 1.94 ms per pass on the
CPU int8 GEMV, six interleaved rounds, checksum unchanged, **1.21x** (1.24x in a
quieter run), and the internal-speeds re-record confirms it costs nothing else
(decode 4.653 against 4.538, prefill +6.5%).

Naming this Mac's CPU as well (`-target-cpu apple-mN` for Swift, `-mcpu` for C)
measured about 1%, inside the noise, because the kernel is float-based rather than
an integer dot product — so there is **no native-build script**; the flags are
here for anyone who wants them on a machine they will not ship from:

```bash
swift build -c release -Xswiftc -target-cpu -Xswiftc apple-m3 -Xcc -mcpu=apple-m3
```

That artifact is not portable (it may use this core's instructions), so never ship
it from `tools/release.sh` or hand it to another Mac. The `.unsafeFlags` is also why
this package cannot be consumed as a dependency; it is an application package and
nothing depends on it.

## Models

**Installing a model is a separate, operator-requested job** — never run it to
satisfy a check, a gate, a benchmark or a release. The 4-bit download is about
19.5 GB and the 8-bit about 36.9 GB. The installer streams the pinned checkpoint
without staging the full source, so it needs `HF_TOKEN` only if one is requested;
cancellation preserves verified completed ranges, which `--resume` continues and
`--discard-partial --output <model.ssdai>` removes.

```bash
swift run -c release TinyTitanRepack --model ornith15-8bit --output models/ornith-1.5_35B_A3B_8Bit
swift run -c release TinyTitanRepack --model ornith15-8bit --output models/ornith-1.5_35B_A3B_8Bit --resume
```

An installed model's `verified-install.json` receipt is bound to the absolute path
it was installed to, so **moving or renaming a model directory makes it fail to
load** with `trusted receipt invalid: model directory mismatch`. This is not
corruption and does not need a re-download — re-issue the receipt in place
(re-hashes the payload against the manifest and rebinds it to the current path):

```bash
swift run -c release TinyTitanRepack --verify-install --input-ssdai models/qwen3.5_4B_4Bit
```

Never hand-edit the receipt to match the new path: the path binding is what detects
a moved or swapped directory, so editing it forges the attestation instead of
re-establishing it. Adding a model is the other runbook, `docs/adding-a-model.md`:
it lists the eight places a new checkpoint has to be wired — the last being its ANE
prefill sidecar — the disk each width needs, the verification bar before it may be
called supported, and how to re-issue install receipts after the checkout moves.

## Test rules

Before a model run, require macOS 26+, Swift 6.4+, enough disk, acceptable
`memory_pressure -Q`, a completed selected `.ssdai` installation, and no process
from `pgrep -fl 'TinyTitanServer|TinyTitanCLI|TinyTitanPackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'`.
If a check fails, inform the user and stop; do not terminate apps or delete or
reinstall the model.

Run package tests serially (`swift test --no-parallel`), passing extra arguments
like `--filter` through. Run only one CLI or model-using test at a time.

**Ad-hoc model runs use the 4B or 9B, not the 2B, unless the task says
otherwise.** That means any run that exercises a code path against a live model —
a feature check, a live end-to-end, a hand-driven probe:
`models/qwen3.5_4B_4Bit` or `models/qwen3.5_9B_4Bit`, the 4B first when only one
is needed. The 2B loads in seconds and is the wrong instrument: it fails
instruction-following, arithmetic and summarisation in ways that read as defects
in the code under test, and a session spent chasing one is a session wasted. Reach
for it only when the small model *is* the subject — its own limits, its own
behaviour — and say in the report that it was deliberate. This does not change the
golden-baseline targets below, which are what they are.

`tools/lint.sh` runs the eleven checks CI enforces beyond the compiler — the
first six are project-specific probes, the last five are pinned third-party
linters:

- `force-cast` — no `as!` / `try!` under `sources/` without a
  `lint:allow-force <reason>` comment above it.
- `func-length` — no function over 120 lines without an inline `lint:allow-long
  <reason>`. The ratchet file `tools/func-length-baseline.txt` carries the
  audited exemptions (14 rows: the formatter sweep's expansions), and the gate
  fails on a stale exemption row as well as on a new offender.
- `unchecked-sendable` — every `@unchecked Sendable` carries an
  `unchecked-invariant:` note.
- `converter` — a probe that files routed experts by index rather than arrival
  order.
- `arch-path` — no hardcoded SwiftPM target triple in a build path, which points
  at nothing on a newer toolchain or at a stale binary on this one.
- `shell-portability` — every shell script parses and runs under `/bin/bash`,
  which is 3.2.57 on a factory Mac, not the Homebrew 5.x a development machine
  puts first on `PATH`. That one is not academic: a single-quoted heredoc holding
  an apostrophe inside `$( )` stops 3.2 parsing the file at all; `${v^^}` or
  `mapfile` parses and then dies mid-menu; and a whole-array expansion
  `"${a[@]}"` on an **empty** array is `a[@]: unbound variable` under the
  `set -u` these scripts set, which 5.x accepts silently. Write it
  `${a[@]+"${a[@]}"}` (likewise `[*]`), which means the same thing for a
  non-empty array on both shells — every script here does, and the gate fails on
  a bare one.
- `shellcheck` (pinned 0.11.0), `swiftlint` (pinned 0.65.1, `--strict`),
  `swift-format` (the committed `.swift-format`), `javascript` (each plugin
  package's own eslint + prettier) and `python` (pinned ruff, plus a parse at the
  declared 3.13 floor). Each pins its tool version and fails when another is on
  `PATH`.

`tools/lint.sh <mode>` runs a single check (`force-cast`, `func-length`,
`sendable`, `converter`, `arch-path`, `shell`, `shellcheck`, `swiftlint`,
`swift-format`, `javascript`, `python`; `format` and `js` are aliases).

In a fresh checkout the `javascript` check needs the plugin packages'
dependencies first — `npm ci` in `plugins/dsh-lan-manager/` and
`plugins/dsh-tinytitan/` (all CI installs before the gate); without it the check
fails with that command in its message and the other ten still run.

`tools/golden-baseline.sh --check <target>` compares greedy, fixed-seed generation
against `benchmark/golden/`. It is the only check that exercises real inference, so
run it for any change to the runtime or the model-load path — the unit tests never
load a model. Ornith 1.5 8-bit (`8`) is the default target; bare `4` still means
Ornith 1.5 4-bit. It counts as a model run: apply the preconditions above first. A
baseline is valid for one (machine, build, model) triple; re-capture only for a
deliberate numerics change, never to make a mismatch go away.

The converter's gate is two python suites, run together:

    cd benchmark && python3 -m unittest test_prepare_qwen38 test_qwen38_resume_e2e

`test_prepare_qwen38` pins the pieces (shard validation, the constants gate, the
retry loop) in a second. `test_qwen38_resume_e2e` runs `main()` end to end against a
synthetic checkpoint served from 127.0.0.1 — real `curl` downloads into a scratch
directory with transfers dropped, truncated, stalled, 404'd and range-refused, a real
`SIGKILL` mid-conversion, a truncated adopted shard, a table reused in place or
copied across a mounted disk image — and asserts that every recovery ends with the
same snapshot a clean run produces. It never fetches a real shard, so it is safe in
CI and takes about half a minute. Run both after any change to the converter, the
resume path, the n-gram table or the download loop; CI runs them too, in a venv
because the runner does not carry `numpy`, `safetensors` or `ml_dtypes`.

**Verification uses only the models already installed under `models/`.** `models/`
is deliberately kept smaller than the full supported set to save disk, so a golden
target with no install there is *reported as not checked* — by `tools/release.sh` and in
the release notes — and never "fixed" by downloading, converting, repacking or
re-installing it. No gate, benchmark or release step may fetch a model to satisfy
itself. Do not download a full checkpoint, duplicate the `.ssdai` model, create a
worktree, or purge caches just to run tests, a gate or a release.

For performance results, build release once and follow the
[community benchmark guide](https://github.com/Pummelchen/TinyTitan/wiki/Benchmarking-Guide)
exactly. Do not enable experimental controls or profiling. Launch helpers live in
`benchmark/`; start the server before running any benchmark script.

Report the commit, hardware and RAM, macOS, Swift version, exact command, exit code,
complete timing footer or error, and every protocol deviation.

## Issues

An issue report is a claim until it is checked. Work it in this order, and skip
none of it because the report looks obviously right or obviously wrong:

1. **Verify against the code**, not against the reporter's summary. Reproduce
   their command where the machine allows it, and say plainly what was and was not
   reproduced. Check whether the defect is already fixed on `main`: a report can
   be true for the commit it names and stale against the current tree, and that is
   the common case — it changes the whole reply, so establish it first.
2. **Fix only what is true and unfixed.** When the report is already fixed, the
   fix *is* the commit that did it and the reply names it. When part of it is
   true, fix that part, and say which part was not.
3. **Test the fix** — a unit test wherever one is possible, and a real run
   wherever the defect is only visible in one (a model run for inference, an
   export for a sidecar). A guard that exists to catch the defect and has no test
   is half a fix. Pin any arithmetic the conclusion rests on.
4. **Verify again** on current `main`, using the reporter's own reproduction where
   it can run, and record the command, the exit code and the output.
5. **Audit for the sibling defect** before closing, and report what you found: the
   same mistake elsewhere, the guard that stops it recurring, and anywhere the fix
   is not reachable.
6. **Reply politely and with evidence**: what was verified, the commit, what to do
   next, what could not be reproduced, and an invitation to reopen. No blame, and
   never "works for me".
7. **Close it** once a fix is on `main`, even while the release lags. The closing
   comment names the commit and says that the next release is where to confirm it;
   an issue left open because no release carries the fix yet becomes a stale list
   nobody reads. If it survives the release for the reporter, it reopens with new
   information.

## Local server

Follow the [server guide](https://github.com/Pummelchen/TinyTitan/wiki/OpenAI-Compatible-Server)
for launch commands, health checks, client setup, prompt reuse, tool loops, and
supported API behavior. Apply the model-process checks above first; never start a
second model process or terminate an existing one.

Keep the server on `127.0.0.1`; it has no remote authentication or TLS, so do not
proxy, tunnel, or expose it. A tool call from the local model never bypasses the
client's normal permission policy. Keep the execution session alive while the
server is needed, and stop only a server you launched.

## Releases and handover

Cutting a release is a runbook, not improvisation: `docs/release-process.md` holds
the order (notes, changelog, version, tag, dry run, publish), the machine
preconditions, and what `tools/release.sh`'s failure messages actually mean — including a
golden gate that reports a *refused* start as a "mismatch". The cross-repository
standard is [`RELEASE.md`](RELEASE.md).

Work in flight is handed over in `docs/handover-<name>.md`;
`docs/handover-tinytitan.md` is the current one and starts with the prompt for the
next session. Read it before installing, converting or moving anything: it names
what is open, and the traps that have already cost a session.

## Releasing

**Read [`RELEASE.md`](RELEASE.md) before cutting a release.** It is this repository's
own release standard — edited here, not deployed from anywhere — and it carries both
the general rules and this repository's own section. Do not improvise a release.

The non-negotiables:

- **Apple Silicon only** — build native `arm64` (M1–M6). Never `--arch x86_64`,
  never `ARCHS=arm64 x86_64`, and never `lipo -create`, which is how a universal
  binary gets made.
- **Assert it** — `lipo -archs <binary>` must report exactly `arm64`. A build that
  silently produced a fat binary is a release defect, not a build option.
- **Every release carries the artifacts.** A tag alone is not a release.
- **Identity is single-sourced and enforced** — never bump one declaration of the
  version or build number on its own; the build or CI must fail on a mismatch.
- **Dry run first**; publish only on an explicit flag.
- **Never fetch a model, dataset or dependency to make a gate pass.** A check that
  cannot run is reported *not checked*, and the release notes must name it.
