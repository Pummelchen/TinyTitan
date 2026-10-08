# Handover: after release 5.18, the pre-production audit

**Paste this into the next session:**

> Continue the TinyTitan work in this checkout. Read `AGENTS.md`, then
> `docs/handover-tinytitan.md`, then the audit ledger
> `docs/audit-2026-10-06/ledger.md`, then the wiki `Project-Tracker`.
> **5.18 is cut and published** (`v5.18` → commit `ea5de8c`, 2026-10-05: the
> launcher fetches a model that is not on disk, the menu offers the ones the
> checkout lacks, and every long download, conversion and repack draws a real
> percentage). Before it, **5.17** (`v5.17` → commit `701bb2e`, 2026-10-04) fixed
> the converter guard that keyed the per-expert duplicate check on the tensor name
> instead of name + width, so `tools/install_models.sh katcoder [both]`
> installs again (issue #19), and moved the user documentation into the wiki;
> **5.16** (`v5.16` → commit `551522a`, 2026-10-03) made the engine a library —
> `TinyTitanLib` is a shipped product, the CLI and the server both generate
> through it, and the toolchain is pinned **exactly**: Xcode 27 / Swift 6.4,
> nothing else. These three citations name the **tagged commit**, not the tag
> object — `git rev-parse v5.18` is `d30de44`, which is a tag object and is what
> this brief said until 2026-10-06 — because CI, `tools/ci-green.sh` and
> `tools/release.sh` all key on the commit, and `git show <tag-object-sha>` does
> not print a diff.
> `tools/lint.sh` runs **eighteen** pinned gates (measured 2026-10-08: the modes
> in its own usage header), and every one
> of them **fails** when its dependency is missing rather than reporting a
> skip — a check that did not run is not a pass. Release notes in the tree keep
> `SHA256_PENDING`; `tools/release.sh` fills them in the staged copy at publish
> time, so a placeholder there is the house pattern, not a defect.
>
> The open work is the **pre-production audit** in
> `docs/audit-2026-10-06/` — `ledger.json` is the single source of truth and
> `ledger.md` is rendered from it (recompute `counts` by hand before
> `render_ledger.py`, which refuses stale ones). Row discipline: verify against
> the code before fixing anything, fix only what is true and unfixed, test the
> fix, re-verify on current `main`, then audit for the sibling defect and close
> with the measured evidence and the commit hash. **Do not read a list of open
> rows out of this brief** — name the severity you want and ask the ledger:
> `python3 -c "import json;[print(t['id'],t['severity'],t['title']) for t in
> json.load(open('docs/audit-2026-10-06/ledger.json'))['tasks']
> if t['status']=='OPEN']"`. This block used to point at **AUD-143** as the
> hardest open row; it closed at `07e867b`, and a session that trusted the prose
> would have re-learned that for nothing. One row needs a human and not a fix:
> **AUD-139** (S1) is blocked on the repository owner, because `v5.18` published
> no `tinytitan-5.18-tools.tar.gz` (confirmed against the release assets on
> 2026-10-06), so the installer's closed check refuses the newest release until
> someone re-publishes it.
>
> The product is the engine plus its loopback server — the Mac app is gone — and
> `tools/install_tinytitan.sh` downloads a built release instead of compiling
> one; the browser chat window arrives the same way, from the release tag's
> source archive, with no registry account on either side. The installs under
> `models/` have receipts **valid for this folder**, because a rename invalidates
> them; re-issue in place with `--verify-install` if the folder moves again, and
> never hand-edit a receipt. **Verification uses only the installs already under
> `models/`** — never download, convert, repack or re-install a model to make a
> gate pass, and never fetch one of the installs the operator deleted; a gate
> that cannot run is reported *not checked*. The open model row is still
> **TT-048, a fresh KAT-Coder install**: the install-path bug is fixed and its
> failure point verified on the real checkpoint (2,075 per-expert tensors, both
> widths, fused `[256, …]` stacks), while the ~69 GB fetch and both repacks
> remain. Report measurements, not assurances.

This is the only current brief; the 5.12 handover it replaces is superseded. The
traps that one named still bite and are folded in below.

> **The product shape changed: the GUI is gone.** The Mac app, the out-of-process
> decode service, the app's library and test targets and `tools/make_app_icon.py`
> were all removed. TinyTitan is an **LLM engine plus its loopback server**:
> `tools/install_tinytitan.sh` downloads the published executables, optionally
> downloads a model, installs `~/.local/bin/tinytitan`, and offers to start the
> server; that command runs `tools/server_launcher.sh`, which prints the base URL a
> client is pointed at. **Do not add a GUI, a desktop front end, or any work that
> only one needs** — `AGENTS.md` states the policy and the reason, and this is not
> a pause. The app's `Info.plist` was also the tree's only version literal; that
> literal now lives in `ServerVersion.current`
> (`sources/TinyTitanServer/Core/ServerVersion.swift`), is printed in the server's
> ready banner, and is checked against the release tag by `tools/release.sh`.

> **The replacement window is DeepSeek Harness, installed not built.**
> `tools/dsh_local.sh` installs a **pinned** `@deepseek-ai/dsh` (`0.2.0-rc.2`,
> into `~/.tinytitan/dsh`, with our `plugins/dsh-tinytitan` bundle from this
> checkout, and the launcher's `--web` starts the server and opens it in the
> browser. Two things are load-bearing. **Isolation:** our copy uses its own
> `DSH_HOME`, npm prefix, pnpm store, and port (7788, stepping up when taken), so a
> DeepSeek Harness the user already runs — their `~/.dsh`, their `dsh` on PATH,
> their 3080 UI — is never read, written or stopped. **No fork:** it is upstream's
> code plus our plugin; the harness is in developer preview and says it will break
> compatibility, which is exactly why the version is pinned and why nothing here
> may start depending on a window existing. Two traps cost time and are recorded in
> the script: a fresh `DSH_HOME` has no `settings.yaml`, so we create it before
> `tools/dsh_route.sh --write` will touch it — 0.2.0 then imports that file into the
> profile patch at boot and renames it to `settings.yaml.imported`, so the plugin's
> own refresh goes through the `settings` service instead; and pnpm's npm-installed
> shim has no shebang, which macOS refuses to `exec` (`spawnSync pnpm ENOEXEC`) —
> the private shim execs `@pnpm/exe.darwin-arm64` instead. `tools/dsh_local.sh
> status` says what is installed.
>
> **Since 5.7 the pin is enforced rather than declared.** Both plugins support
> exactly `0.2.0-rc.2` — `dsh-tinytitan`'s peers are exact, not ranges — and
> **refuse to run** on any other harness, including one whose version cannot be
> read. A refusal never throws: it writes one line to stderr and returns, so DSH
> boots, every other plugin loads, and removing ours leaves nothing to undo. stderr
> is not a preference — the harness prints a plugin's log records only when the boot
> itself fails, so a host-logger line would be invisible (tracker, plugins section).

## Where the work stands

Every measured cell below carries the date it was measured and the command that
re-measures it. That is the point of the format: this table was a hand-copied
snapshot, and six of its eleven rows had drifted by 2026-10-06 — it named 5.15 as
the current release after 5.16, 5.17 and 5.18 shipped, "8 installs, 244 GB"
after the model set was pruned to two, "7 checked" goldens when one target has
an install here, "level with `origin/main`" on a tree with work no remote has
seen, "28 findings, all closed" beside an open audit, and "level with
`origin/master`" for a wiki with an unpushed commit. **When a fact here changes,
re-run the command and rewrite the cell with the new value and today's date** —
do not edit the number to match what you expect, and do not trust it as current
if the date is old.

| Piece | State |
| --- | --- |
| Repository | `Pummelchen/TinyTitan` (renamed 2026-09-14; the old URL redirects) |
| Checkout folder | `~/Downloads/TinyTitan` — **renamed from `~/Downloads/NVMAI`**, which invalidated every receipt and `.build`'s debug half |
| `main` | as of 2026-10-08, **151 commits ahead of `origin/main` and nothing pushed** (`git rev-list --count @{u}..HEAD` — 149 measured at the AUD-208 close commit `4faccda` and 150 at the AUD-209 fix commit `2e0af50`, and the close commit this row lands in is the 151st; the commit that edits this row makes it one more, which is what the command is for, not a mistake to fix); newest tag `v5.18` (tagged commit `ea5de8c`), HEAD `2e0af50` plus this commit. Consequence, stated plainly because it bites at release time: CI runs on push, so **no CI run covers any of that work** — the local gates and the serial suite are the only evidence, and `tools/release.sh` will refuse to tag until `tools/ci-green.sh` sees a run on the commit |
| Release | **5.18 published** 2026-10-05 (`gh release list` — it is the latest), assets `tinytitan-5.18-macos-arm64.tar.gz` + `.sha256` and `tinytitan-lib-5.18-macos-arm64.tar.gz` + `.sha256`; **no `tinytitan-5.18-tools.tar.gz`**, which is what blocks AUD-139 on the repository owner. `ServerVersion.current` is `5.18`, and `tools/release.sh:118` refuses a tag that disagrees with it |
| Models | as of 2026-10-06, **2 installs, 163 GB** (`du -sh models/*`): `qwen3.8-flash-next_125B_A6B_4Bit` (162 GB) and `qwen3.8-flash-next_125B_A6B_MTP_4Bit` (1.4 GB). The rest were pruned for disk and **must not be re-fetched** to satisfy a gate; every receipt here is bound to this path, so both load |
| Goldens stored | 16 files under `benchmark/golden/`, 16 targets in `tools/golden-baseline.sh`; as of 2026-10-06 **1 is checkable** on this host — `qwen38-4`, the only target whose directory exists under `models/`. The other 15 (`ornith-{4,8}`, `qwen38-8`, `qwen36-{4,8}`, `agentworld-{4,8}`, `katcoder-{4,8}`, `qwen35-{2b,4b,9b}-{4,8}`) are reported *not checked* and named in the notes; the default `ornith-8` is among them. The MTP install maps to no golden target at all |
| Audit | **this audit**: `docs/audit-2026-10-06/` — as of 2026-10-08, `counts` in `ledger.json` is **106 rows / 105 closed / 0 open / 1 blocked** (`python3 -c "import json;print(json.load(open('docs/audit-2026-10-06/ledger.json'))['counts'])"`), and the eighteen gates in `tools/lint.sh` are partly what it left behind. That command is not a suggestion here: `tools/lint.sh docs` compares this very string against `ledger.json` and fails, so any ledger close that moves a count edits this line in the same commit — which is the point, because a restated count is how AUD-122 shipped a commit message its own file contradicted. **the 2026-09 audit**: 28 findings, all closed. The wiki's archive page was removed on 2026-09-29 when the wiki became user-only — the record is in the wiki repository's history at `6acaa8f` |
| `.build` | release build of current `main` (`swift build -c release`, 2026-10-06); a clean scratch release build is part of each dry run |
| Wiki | `.qwen/wiki`, remote `TinyTitan.wiki.git`, **1 commit ahead of `origin/master`** as of 2026-10-06 (`git -C .qwen/wiki status -sb`) — the wiki half of the last change is unpushed, exactly as the code half is; publishing is **two pushes**. User-facing only since 2026-09-29 |
| DeepSeek Harness | pinned `0.2.0-rc.2` and **enforced**; both plugins refuse any other version; the global harness runs the gate, the private one is refreshed but idle until its next start. The private bundle is isolated down to the caches: npm's cache/logs/user config, pnpm's home and the XDG cache/state all live under `~/.tinytitan/dsh`, so a run adds nothing to `~/.npm`, `~/Library/pnpm`, `~/.cache` or `~/.local/state` (`benchmark/test_dsh_isolation.py` pins it; verified in a simulated factory-new HOME). Since 5.11 the bundle is the delivery — the installer's source archive carries `plugins/`, and the route writer and the launcher both resolve the installed layout (`../bin`, `../models`) instead of a checkout's. 0.2.0 removed `settings.yaml` and the preset files: the harness imports a legacy `settings.yaml` into the profile patch at boot (and renames it `.imported`), the plugin writes the route through the `settings` service and registers its preset with the preset registry, and the default preset is set only while the profile names none |
| CI | every `main` push runs the `test` job (eighteen gates, plugin/ converter/ installer/ release gates, the embedded-dependency check, serial tests, Markdown links) and the `thread-sanitizer` job, plus CodeQL (`languages: swift`); the release commit's push is the run to watch (`gh run list`). As of 2026-10-08 nothing has been pushed, so **no run covers the 120 commits above** (`git rev-list --count origin/main..HEAD`, measured at `b5e0c4e` plus this commit — the row is wrong the moment anything lands, and the `main` row above carries the live count) — that is the `main` row, not a secret. Four CI arms have been run by hand on this tree because no push can cover them: `tools/embedded-dependency-check.sh` is exit 0 at `87ace2c` (`Build complete! (29.43 sec)`, `EmbeddedDemo: TinyTitanLib linked (Engine)`, `== ok: the package is consumable as a dependency ==`, `/tmp/embedded188.log`); the plugin suites pass through `tools/lint.sh javascript` on the pinned 10.11.0/3.9.9, and each package's own suite passes run directly at `c7e2159` — `npm test` in `plugins/dsh-tinytitan` 154/154 and in `plugins/dsh-lan-manager` 141/141, both exit 0 (`/tmp/js_tinytitan.log`, `/tmp/js_lan.log`); and the `thread-sanitizer` job's exact command (`swift test --no-parallel --sanitize=thread`, `ci.yml:259-273`) completed at `c3d8d58` with **exit 0 and zero `WARNING: ThreadSanitizer` lines** over all 7 targets, 1711 tests in 253 suites — the same counts as the unsanitized run, so nothing was skipped (`/tmp/tsan_all.log`; a `--scratch-path /tmp/tt-tsan` keeps `.build` warm). ASan has since been run over the C-kernel call sites (`swift test --no-parallel --sanitize=address --scratch-path /tmp/tt-asan`, scoped to the CPU suites that reach the kernels — `Int8AffineGEMVTests`, `CPUExpertFFNTests`, `CPUEngineTests`, `CPUServingTests`): **exit 0, 55 tests in 5 suites passed, zero `ERROR: AddressSanitizer` lines** (`/tmp/asan191.log`). UBSan has **not** been run through SwiftPM: `swift test --no-parallel --sanitize=undefined --scratch-path /tmp/tt-ubsan` fails before any test executes, because linking `TinyTitanBench` under that scratch path dies with `ld: symbol(s) not found for architecture arm64` on `_tinytitan_expert_reader_create in TinyTitanKernelsC.o` (`/tmp/ubsan191.log`) — an artifact of the sanitizer's separate build, not a defect in the kernels. It has since been run **directly**, which is how all three C files are now covered: `/tmp/probe_wide.c` (both affine GEMVs, 257 groups past the hoisted-sum table) and `/tmp/probe_io.c` (the reader: EINVAL shapes, thread clamping, exact-stride destinations, an offset past EOF, a `UINT32_MAX` expert id, two concurrent submitters) compiled with `-fsanitize=address,undefined,bounds,alignment,shift,signed-integer-overflow -fno-sanitize-recover=all` against the real kernel sources — **both exit 0, zero `runtime error` and zero `ERROR: AddressSanitizer` lines** (`/tmp/gemv_asan.log`). The reader driver is not vacuous: replacing `read_one`'s short-file `return EIO` with `return 0` flips three of its assertions and exits 1, and `expert_io.c` was restored from `/tmp/expert_io_backup.c` with `git diff` on `sources/TinyTitanKernelsC/` empty afterwards. Neither is evidence about the *pushed* tree; they are evidence that this tree would pass |.

## What has landed

- **5.15** (`4ff6041`) — `.gturbo` → `.ssdai`: the format's **name only**.
  `manifest.json`'s magic is written as `"SSDAI"`, and reads accept the legacy
  `"GTURBO"` for one release (`SSDAIFormatV1.isSupportedMagic`), so every install
  built before the rename keeps loading untouched: verified with
  `TinyTitanRepack --verify-install --input-ssdai models/qwen3.5_4B_4Bit`
  ("Verified 7 files", legacy manifest) and by loading that install through the
  CLI. Nothing rewrites an existing manifest — the receipt
  (`verified-install.json`) binds the manifest's digest and the directory path,
  so editing the magic would invalidate every receipt for a string. The Swift
  vocabulary (`SSDAIFormatV1`, `SSDAIDirectoryAccess`, `SSDAIBinary`, …), the
  `--input-ssdai` flag (`--input-gturbo` stays a deprecated alias),
  `tools/ssdai_reader.py`, `tools/ssdai_diff_snapshot.py` and
  `docs/ssdai-format.md` carry the new name; the directory suffix is a
  convention the reader never checks. Dated records — release notes, the v4.x
  plans, `plan-dense-gturbo-installs.md` — keep the old word on purpose, with
  their links to the spec updated. Verification: eleven gates clean, 1,496 tests
  in 224 suites, 45/45 local links, plugin suites green.
- **5.14** (`10e5d0f`) — the browser window and both plugins on DeepSeek Harness
  **0.2.0-rc.2**. The `llm-pi-ai` route is applied through the harness's `settings`
  service (0.2.0 imports a legacy `settings.yaml` once and renames it
  `.imported`), the `tinytitan` preset is registered with the preset registry and
  becomes the default only while the profile names no selection, the `models/`
  watcher shares the boot refresher so a mid-session install reaches the picker,
  and the optional `compactionHeadroomTokens` sets the compaction trigger's
  headroom (the stock 65,536 holds it near 62% of a 262,144 window and stops
  compaction entirely below roughly `cap + 65,536`). `tools/dsh_local.sh status`
  now reports the installed harness version rather than the pin. Also: the
  internal-speed record became optional in `docs/release-process.md` §4b and
  `RELEASE.md` (owner decision 2026-10-01 — this machine's timings swing past the
  10% threshold under a browser or a game), so 5.14 carries no timing record.
  Verification: eleven gates clean, **1,496 tests in 224 suites**, **7 of 16
  goldens byte-identical**, the nine absent baselines named in the notes, a
  warning-free clean scratch build, both plugin suites green (84 tests: 83 pass,
  1 skipped; the LAN manager's 107), and CI green on `f77f786` and `bc3136e`.
- **5.13** (`89d3317`) — one diagnostic plus a documentation rewrite. A `--model`
  path that does not exist now fails with `model directory not found: <absolute
  path>` instead of `installed tokenizer is missing chat_template.jinja`, at the
  CLI, the server's session load and the router's token counter; three tests pin
  it, and the CLI test fails on the pre-fix tree at `7535ebc`. The wiki became a
  user guide (21 → 16 pages: new Installation and Configuration, rewritten
  Runtime Controls and Local Server and API, a larger Cookbook) and the README a
  verified quickstart. 50 source files were split under the 500-line rule as pure
  code motion. Verification: eleven gates clean, **1,496 tests in 224 suites**,
  7 goldens byte-identical, a warning-free clean scratch build, and the 4B
  internal-speed record inside the 10% gate on all thirteen metrics — after two
  contended passes were discarded because a 3D game held ~78% of a core and the
  GPU.
- **5.12** (`fc23691`) — the pre-production audit, drained to zero open findings:
  every force unwrap gone (171 in `sources/`, 139 in tests and benchmarks), the
  chunked-prefill force cast now the existing `chunkedUnsupported` error, a
  fleet-scanner test that could never fail made real, Swift SAST (CodeQL) running
  again after failing before it compiled anything, and the Markdown-link and
  Python-suite CI regressions closed. `tools/lint.sh` now runs **eleven** pinned
  gates (SwiftLint `--strict` and swift-format joined it, plus eslint/prettier for
  the plugin packages with committed lockfiles), the tree is at 0 SwiftLint and 0
  swift-format findings after a 442-file sweep, and all 28 audit findings are closed
  (archived in the wiki's history at `6acaa8f`; the wiki page itself was removed on
  2026-09-29). Verification: eleven gates clean, **1,493 tests
  in 223 suites**, 7 goldens byte-identical, a warning-free clean scratch build,
  and a 4B speed record whose first pass read prefill low (28.0 → 23.3 tok/s) and
  whose repeat read the baseline exactly with all thirteen metrics inside the gate
  — both records and the reason are in `docs/release-notes-v5.12.md`.
- **5.11** (`6e7cfc3`) — the DSH bundle is the delivery: the installer's source
  archive carries `plugins/`, the bundle is added from there with a `file:`
  install and no registry account exists on either side, the private harness
  keeps npm's cache/logs/user config, pnpm's home and the XDG cache/state inside
  `~/.tinytitan/dsh`, and both installed-layout resolutions are fixed
  (`dsh_route.sh` checks `../bin`/`../models`; `dsh_local.sh` resolves the models
  directory once and prints it in `paths`). The model installer refuses a download
  that cannot finish (staging: size × 1.25 + 12 GB; models: size + 3 GB, both
  printed, `TINYTITAN_SKIP_DISK_CHECK=1` to override), stages under the install
  root rather than the caller's cwd, reclaims staging as widths complete, and an
  EOF at the model menu no longer installs anything. Memory distillation is
  chained per scope, so a later session reads memory only after the earlier one
  wrote (TT-035). `--ram` help now says it accepts any whole GB from 4.
  Gates: six lint gates clean (2,031 functions, 20 scripts), **1,493 tests in 223
  suites**, 7 goldens byte-identical, a warning-free scratch build, and a 4B
  speed record with every metric inside the 10% gate and none regressed
  (`docs/release-notes-v5.11.md`).
- **5.10** (`a89255e`) — `--ram` is a target for the whole server process rather
  than the expert cache alone (4 GB floor, printed estimate; `--ram 8` now buys 32
  slots and `--ram 12` reproduces the old 64), Qwen3.8's two sampling rows are
  implemented (`--presence-penalty`, the row chosen from the request's thinking
  mode), the C kernels compile at `-O2`, the expert-cache ceiling is a third of
  physical memory, twelve decode switches that measured a wash or a loss are gone,
  and converting Qwen3.8 resumes and works through mirrors
  (`docs/release-notes-v5.10.md`). Gates: six lint gates clean (2,030 functions,
  20 scripts), **1,491 tests in 223 suites**, 7 goldens byte-identical, a
  warning-free scratch build, and a 4B speed record with every metric inside the
  gate — `gpu.routed_moe` read low on the first run under residual background load
  and both values are in the notes.
  `in_proj_a`/`in_proj_b` pair at the attention slot's own width loads and serves
  again (issue #16, a qwen38flash 4-bit install); the ten master prompts are
  runnable end to end, with a client's own summary as the baseline memory has to
  beat; and which judge runs the side-engine's tasks is a measurement. Beside
  those, one ordering fix: a search queues its T7 question before it answers
  (`docs/release-notes-v5.9.md`). Gates: six lint gates clean (2,035 functions,
  20 scripts), **1,484 tests in 222 suites**, 11 goldens byte-identical, a
  warning-free scratch build, and a 4B speed record with every generation metric
  inside the 10% gate — the two synthetic kernel metrics read low under system
  load and the notes carry both values and the reason.
- **5.8** (`4fc0726`) — the memory side-engine (a 4B on the CPU decides
  durability, duplication, contradiction and supersession, six questions a
  consolidation), the rule that holds a write back, IDF retrieval plus the
  background T7 caller, shared n-gram tables, per-tensor bit widths in the
  resident index, the DSH LAN manager, and one pinned harness release. Record:
  `docs/release-notes-v5.8.md`. Gates: six lint gates clean (2,035 functions, 19
  scripts), **1,482 tests in 222 suites**, 11 goldens byte-identical, a
  warning-free scratch build, and speeds inside the 10% gate against 5.7.
- **5.7** (`44e1ae9`) — one command installs a built engine, the app is gone, and
  every script runs on `/bin/bash` 3.2.57. `docs/release-notes-v5.7.md`.

## What is open

The [Project Tracker](https://github.com/Pummelchen/TinyTitan/wiki/Project-Tracker)
is the authority. On 2026-10-04 it holds one Open row:

1. **TT-048 — re-verify a fresh KAT-Coder-V2.5-Dev install.** Issue #19: the
   converter's per-expert duplicate guard was keyed on the source tensor name
   alone, while `convert_shard` adds one tensor once per requested width and
   `install_models.sh` always converts this family with `--bits 4 8`, so every
   fresh `katcoder` install stopped at the first routed expert with
   `duplicate source tensor …`. Fixed in `025dacb` (guard keyed by
   `(width, name)`), along with two shutdown defects the verification exposed
   (`14cc6e7`, `f1c70b7`). Verified on the real checkpoint: shard
   `model-00000`'s 2,075 per-expert routed tensors convert at both widths and
   write fused `switch_mlp.*` stacks with a 256-expert axis. What remains is the
   rest of the 13-shard fetch, both repacks, and the `docs/adding-a-model.md` §4
   checklist before the model is called supported again; the installs already
   under `models/` are unaffected. TT-038 (an embeddable engine) closed on
   2026-10-02 — the facade shipped as `TinyTitanLib`.
2. **Carried forward, not tracked as tasks:** the Qwen 3.8 port items — QSA
   indexer selections to the GPU, a higher expert slot budget, the n-gram gather a
   token ahead. TT-021–TT-023 were closed on 2026-09-19 (no other machines; no disk
   for the ~360 GB bf16 reference), so the M1–M6 claim stays a design intent and
   Qwen 3.8 long-context stays verified only at a lowered budget.
3. **Three audit follow-ups that measurement could not close.**
   (a) `try? await flush()` at `ContinuityEngine.swift:128` and `:147`, and the
   deferred barrier at `Journal.swift:145`, swallow a failure of the durability
   barrier without recording it in `journalFailure` — so a workspace whose `fsync`
   fails goes on reporting itself durable, which is the one promise the memory
   subsystem is written to keep honestly. It was not filed because it could not be
   reproduced here rather than because it looks fine: `flush()` casts its journal to
   `FileJournal` and returns for anything else, so no injected journal can fail a
   barrier through any seam, and nothing on this Mac makes `fsync` fail on a file
   whose `write` calls all succeeded. **Measured 2026-10-08, and it still cannot be
   produced** — see `docs/audit-2026-10-06/AUD-204-barrier-probe.c` for the run and
   its numbers. A descriptor opened on a mounted 16 MB APFS disk image, then the
   image force-ejected underneath it: `fcntl(fd, F_FULLFSYNC)` returns -1 with
   `EBADF`, `fsync(fd)` returns **0**, and the next `write(2)` returns -1 with
   `EIO`. So the barrier's only failure signal is an errno the kernel does not set
   for a dead volume, while the append does fail — through `writeFully`, into the
   observer, onto `journalFailure`, where `isDurable` already reads it. The two
   `try?` remain an error with no caller and no trace in the source, and no machine
   here turns them into a lost record. Do not spend another session trying `hdiutil`
   variants; the next thing that could answer it is a volume that answers `fsync`
   with `EIO`, which is a failing drive rather than an experiment.
   (b) AUD-189's fix repairs a dangling journal line when the file is next *opened*;
   an append that fails partway inside a running process leaves the same line until
   then, and its next record would fuse onto it. The writer there already has an
   error in hand and records the durability failure, so the loss is reported — what
   is not pinned is the recovery. It has no test seam: a partial `write(2)` needs a
   disk that fills mid-call, and no gate here may fill one. Closing it needs either a
   volume that can refuse a write at a chosen byte count or a decision to accept the
   next open as the repair point.
   (c) AUD-190's sibling sweep left one site of the same shape unfilled:
   `ServerPromptStateStore.swift:314-321` and `:382-395` subtract a record's
   `payloadBytes` from `diskBytes` *before* the `try? removeItem` that has to free
   the directory, so an eviction the volume refused leaves the bytes counted as
   spent — the disk cap then believes it has room and stops evicting, and
   `ServerPromptStateSaveResult.diskBytes` reports the same fiction through the
   server's own status. Not filed, because the failure cannot be reached on a store
   root that will not take unlinks: `writeDisk` uses `try` and rethrows
   (`:447`, `:464`), so a volume that refuses the directory refuses the file first
   and says so through the `diskError` channel the type already carries. A next
   attempt needs a directory the process can create into and cannot unlink from
   (an `uchg` flag, or a mode change made under it), and the fix is small once that
   is reachable: subtract on success, and put the refusal in `diskError`.

TT-018 (the plugin's delivery) and TT-020 (reaching the LAN manager from another
machine) were both closed on 2026-10-01: the `awesome-dsh-plugin` fork is gone,
and the LAN capability is verified across hosts with the direct interface bind
left as an upstream ask in `docs/dsh-upstream-asks.md`.

## Traps worth carrying forward

- **A hand-set `baseURL` can 404 every model call.** `dsh-llm-deepseek` defaults to
  `protocol: messages`, whose root is `https://api.deepseek.com/anthropic`;
  `https://api.deepseek.com/v1` is the chat-completions root, and every request then
  goes to a path DeepSeek does not serve. All four fleet nodes were failing every
  turn this way until 2026-09-18. The generated route (`tools/dsh_route.sh`) does
  not make this mistake; config typed by hand does. Full entry under *Traps that
  have already cost time* in the engineering notes, which now live in the wiki
  repository's history (`git -C .qwen/wiki show 6acaa8f:Engineering-Notes.md`)
  rather than in the user wiki.
- **A version gate must be visible, not merely correct.** The harness collects a
  plugin's log records and prints them **only when the boot itself fails**, so a
  refusal reported through the host logger is invisible on a healthy boot. Both
  plugins write to stderr for that reason. Found by booting a throwaway harness in a
  temporary `DSH_HOME`, not by reading — do that again for any boot-time claim.
- **Renaming the checkout invalidates every install receipt and `.build`'s debug
  half.** Receipts bind absolute paths; re-issue with
  `swift run -c release TinyTitanRepack --verify-install --input-ssdai <dir>`
  and never hand-edit one. The debug tree is compiled against absolute paths too
  — after the rename, 7,312 files named the old path and `swift test` died with
  `precompiled file …_Builtin_stdbool….pcm was compiled with module cache path
  '/Users/andreborchert/Downloads/NVMAI/…'` **before a single test ran**;
  `release.sh` reports that as `swift test did not report a passing run`, which
  reads like a failing test. Remove `.build/debug` and let it rebuild.
- **A release tag that is not yet published may be force-moved.** The dry run's
  numbers must be in the notes, so the sequence is: commit prep → tag → dry run →
  fill in `### Verification` → commit → `git tag -f` → `git push --force origin
  vX.Y` → `--publish`. `--publish` re-runs every gate and rebuilds the archive,
  so **the published digest and size are never the dry run's** (5.11: 15,370,277
  bytes dry, 15,370,253 published; 5.10: 15,367,456 dry, 15,367,433 published;
  5.9: 15,437,771 dry, 15,437,857 published; 5.8: 15,436,743 dry, 15,436,730
  published; 5.5: 26,093,424 dry, 26,094,346 published) — that is what the
  placeholders are for.
- **A fire-and-forget registration can race the observer that awaits it.** The
  T7 schedule closure in `MemoryService` handed the question to an unstructured
  `Task { await hinter.register(…) }` and returned, so `waitForRetrievalHints()`
  could return before the question was queued; the plain suite passed and only
  `--sanitize=thread` failed, on the 5.9 release commit. Fixed by awaiting the
  queueing hop, which never runs the engine. The general form: when something is
  described as background, the seam that observes it must wait for the
  *hand-over*, not merely for already-started work.
- **`swift build -c release --build-tests` cannot pass on this tree, and it is not
  a manifest defect.** It fails `unable to resolve Swift module dependency to a
  compatible module` for the libraries the tests `@testable import` — measured 2 of
  2 on 2026-10-06, seven modules named once and one the next time, because the build
  stops at the first module it cannot bind. `swift test -c release` on the same tree
  builds and runs (`Build complete! (148.88 sec)`), and adding `-Xswiftc
  -enable-testing` to the failing command makes it succeed too (`Build complete!
  (147.46 sec)`), which is the mechanism: a release `--build-tests` compiles the
  product libraries without `-enable-testing`, so a `@testable` import has no
  compatible module to bind. Ledger AUD-152, filed first as an unreproduced flake.
  So: run release-config tests with `swift test -c release`, and do not purge
  `.build` or edit `Package.swift` when this error appears.
- **A test that awaits a parked task cannot time out, so it hangs instead of
  failing.** Two probe harnesses written to catch a lost wake produced no output at
  all and had to be killed after 7 and 10 minutes: a task group joins *every* child,
  so the child blocked on `await parkedTask.value` outlives the timeout child and
  `group.cancelAll()`, and cancelling the waiter does not cancel the unstructured
  task it is awaiting. The deadline has to be a polling loop over a counter the work
  writes on its way out — which is how `concurrentSwitchesWakeEveryQueuedCaller` and
  `RoutingGateTests` are written, and both then fail in 30.0 s and 10.1 s instead of
  hanging. A second defect was hiding behind the first: the fixture gate stored one
  continuation for any number of waiters, so the second concurrent waiter overwrote
  the first.
- **A pinned tool can be shadowed before the gate's remedy can reach it.** This host
  has Homebrew `ruff` 0.16.10 in `/opt/homebrew/bin`, which precedes pipx's
  `~/.local/bin` on `PATH`, so `pipx install --force ruff==0.16.7` — the remedy the
  gate printed — installs the pin where nothing will use it. The gate now names the
  binary it resolved. Until that PATH entry is removed the `python` gate is *not
  checked* on this host; the check itself passes with the pinned ruff first on
  `PATH` (`ok (ruff 0.16.7, check and format clean, parses under 3.13)`), so the
  repository's python is clean and only the environment is not.
- **The synthetic kernel metrics swing with the machine, not the code.** QKV GEMV
  and GDN in-projection have read 55.4–78.6 and 66.8–77.4 GB/s across the
  v5.5–v5.8 records on this machine, and 5.9 measured 63.5/67.9 while macOS's
  `dasd` held a core at ~95% and Chrome was active. 5.10 hit the same on
  `gpu.routed_moe`: **37.1 GB/s against 43.6** on the first run with Chrome
  helpers and `mediaanalysisd` still active, **41.4** on the re-run — and that
  counter has ranged 36.8–56.1 across the stored records. One re-run, with both
  values and the reason in `### Verification`, is the treatment; a search for the
  best of many is not. The speed gate compares
  against the previous release's record, which can be the series' high-water
  mark. Cross-check the generation metrics and the greedy response hash (an
  unchanged hash means no arithmetic moved), then record both values and the
  reason in `### Verification` rather than re-rolling the number into a
  flattering record.
- **A `file:` plugin install is a copy.** After editing
  `plugins/dsh-tinytitan/`, re-install it
  (`dsh plugin --profile web remove dsh-tinytitan`, then `add
  file:<checkout>/plugins/dsh-tinytitan`) or the harness keeps running the copy.
  This bit on 2026-09-18: both installed copies were a week of edits behind, and
  only a re-install plus a harness restart put the current code in service.
- **The wiki is a second repository with its own history.** Pull `.qwen/wiki`
  before editing it; the tracker and the Changelog are separate commits; the
  fine-grained PAT can read it but was rejected for push, so use the `gh`
  credential helper (`gh auth setup-git`). A push means **both** repositories.
- **Verification is what is installed.** `models/` is pruned for disk on purpose;
  a target with no install is reported *not checked* and named in the notes, and
  nothing is fetched to change that. A stored baseline is never deleted because
  its model is currently absent.
- **`release.sh` needs `HEAD` to *be* the tag** and a release build at
  `.build/release/TinyTitanCLI` to exist **before** it starts.
  The golden phase refuses to run beside any model process.
- **The `xcode-27` runner is an x64 Actions agent on arm64 hardware.** Inside its
  jobs `uname -m` prints `x86_64` while `swift --version` targets arm64, and
  `/usr/bin/sandbox-exec` ships only arm64e slices. Anything that injects a native
  arm64 dylib into spawned processes therefore dies on that binary: CodeQL's tracer
  failed with `posix_spawn error: Bad CPU type in executable (86)` before compiling
  a single file, for months, and the workaround is `swift build --disable-sandbox`
  in `.github/workflows/codeql.yml`. The durable fix is an `osx-arm64` Actions
  runner; after that the flag can go and the manifest sandbox return. Do not
  re-diagnose it as a tool bug.
- **Installing a toolchain can change what a scanner scans.** The first `npm ci` in
  CI made the Markdown link check read `plugins/*/node_modules/**` and report 461
  broken links from vendored READMEs. Build output, vendored dependencies and
  model stores are not project source: exclude them explicitly, and re-run any
  whole-tree scanner after adding an install step.
- **Opening a journal for writing now truncates its last incomplete line.**
  `FileJournal` was append-only, except at compaction, until 2026-10-08; AUD-189 put
  `dropDanglingTail` in the initializer, because the unterminated tail a kill leaves is
  also the line `O_APPEND` writes the next record onto, and the fused line decodes as
  neither. The bytes removed are ones `replay` already dropped, so no readable record
  is lost — but a journal copied out of a workspace after a crash is shorter than the
  same file was a minute before the server opened it, and that is the repair, not a
  second defect.
- **Say what a guard actually reads, not what it intends**, and **test a claim
  rather than trusting it** — both defects that reached a release in this project
  were claims broader or more specific than the code.
