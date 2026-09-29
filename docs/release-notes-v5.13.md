## TinyTitan 5.13 — a wrong model path names itself, and a wiki written for users

This release is one diagnostic fix and a documentation rewrite. No kernel, model
format, default or API surface was retargeted, so the engine's speeds and its
outputs are unchanged; the large file-size refactor that came with the work is
pure code motion, verified as such.

### What is fixed

- **A `--model` directory that is not there is named, instead of being reported
  as a tokenizer problem.** Any path mistake used to fail with
  `error: installed tokenizer is missing chat_template.jinja; reinstall the
  model`, because the tokenizer is the CLI's first step and it reports a missing
  `tokenizer/tokenizer.json` with the template's error — sending the reader to
  reinstall a model that is simply not at that path. It now fails with
  `error: model directory not found: <absolute path>`. The same check runs at
  every entry point that resolves a model path: the CLI's tokenizer load, the
  server's session load and the router's token counter. The template error keeps
  its meaning for a directory that exists without the file.
  **Checked by** three tests: `CLIModelDirectoryTests` drives `run(args:)` with a
  missing `--model` and requires the path in stderr with no mention of
  `chat_template.jinja` — against the pre-fix tree at `7535ebc` it fails with
  exactly the old message — plus two tokenizer-layer cases covering a missing
  directory and a path that is a regular file.

### What is new

- **The wiki is a user guide.** 21 pages → 16. The engineering record, the closed
  audit findings and two measurement studies were removed, and every remaining
  page was rewritten around using TinyTitan. New: **Installation and
  Configuration** — every model key, where installs and staging live, the
  launcher's flags and all the environment variables in one place. Rewritten:
  **Home** (the guide map and a three-command quickstart), **Runtime Controls**
  (every CLI, server and API parameter with its default and meaning), **Local
  Server and API** (endpoints, request examples, structured output, function
  tools, model ids, client wiring, error codes), **Cookbook** (more recipes, each
  with the output to expect: a messages file, reproducible sampling, streaming, a
  named JSON schema, function calling, a Python client) and **Benchmarks** (what
  to expect on an M3, without the version-by-version history).
- **The README is a verified quickstart.** What TinyTitan is and who it is for, a
  quickstart with the commands that were actually run, prerequisites, install,
  configuration, a complete worked example (server, `/health`, `/v1/models`,
  Chat Completions, `json_object`), common workflows, troubleshooting and where
  to get help — with a `Last verified` line naming what was run and what was not.

### Also in this release

- **50 source files split under the 500-line rule**, with no behaviour change.
  Every moved block was checked byte-for-byte against the revision it came from,
  and the full suite was re-run after each split. No file under `sources/` is
  now above the limit (353 files).
- **Instruction-document drift corrected.** `AGENTS.md`, `RELEASE.md` and
  `CONTRIBUTING.md` said six lint gates (there are eleven), claimed the
  function-length baseline was empty (it holds 14 rows), and their structure map
  omitted two targets; the `--verify-install` example named a model no install
  has.
- **`docs/repository-layout.md`** records the splits, the `lint:allow-long`
  exemptions a straight-line sequence keeps, and the production-only scope of the
  size rule.

### Verification

- `tools/lint.sh` — all eleven gates, clean.
- `swift test --no-parallel` — 1,496 tests in 224 suites.
- **7 of the 16 stored golden baselines compared byte-identical**
  (`qwen36-{4,8}`, `qwen38-4`, `qwen35-{4b,9b}-{4,8}`) — every installed model
  that has a golden target — then a clean scratch release build with a clean
  warning scan.
- **Not checked, because their install is not under `models/` and nothing may be
  fetched to change that**: `ornith-4`, `ornith-8`, `qwen38-8`, `agentworld-4`,
  `agentworld-8`, `katcoder-4`, `katcoder-8`, `qwen35-2b-4`, `qwen35-2b-8`.
- The internal-speed record (`benchmark/internal-speeds/v5.13.json`, the 4B dense
  install) against `v5.12.json`: **all thirteen metrics inside the 10% gate** —
  decode 27.96 tok/s (baseline 27.81), prefill 28.0 tok/s (23.3 in the baseline's
  noisy first pass, 28.0 in its repeat), TTFT 0.25 s, GPU QKV 77.3 GB/s, routed
  MoE 46.2 GB/s, GDN in-projection 83.2 GB/s, CPU affine 60.8 GB/s, ANE prefill
  52.7 tok/s.
- **Two earlier passes on this same tree were discarded, and are named here
  rather than hidden**: the release machine had a 3D game holding about 78% of a
  core and the GPU, and those passes read 24–69% below the entire historical band
  of records (GPU QKV 48.8 then 24.9 GB/s against 67–80 across `v5.10`–`v5.12`,
  decode 16.8 then 19.5 tok/s against 26–28.6). Contention, not the build: the two
  passes disagreed with each other, which a code regression cannot do, and this
  release moves no kernel arithmetic or memory layout — the seven golden
  baselines above are byte-identical.

### Checksum

`tinytitan-5.13-macos-arm64.tar.gz` sha256: `SHA256_PENDING`
`tinytitan-5.13-macos-arm64.tar.gz` size: `ARCHIVE_BYTES_PENDING` bytes
