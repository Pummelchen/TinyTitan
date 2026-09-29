# Handover: after release 5.13, the engine and its loopback server

**Paste this into the next session:**

> Continue the TinyTitan work in this checkout. Read `AGENTS.md`, then
> `docs/handover-tinytitan.md`, then the wiki `Project-Tracker`. **5.13 is cut and
> published** (`v5.13` → `89d3317`; `tinytitan-5.13-macos-arm64.tar.gz`, 15,415,573
> bytes, sha256 `e5359f1881d2dffa3ad8599c912eb44ca199eb42831ce8af4222e1b2a8dfc6e3`,
> 2026-09-29) and **`main` sits one commit past it** — this brief; the release notes
> are `docs/release-notes-v5.13.md`. 5.13 carries one diagnostic — a `--model` path
> that does not exist now names the path instead of blaming the tokenizer — plus a
> wiki rewritten as a user guide; no engine behaviour changed, and the 50 file
> splits it carries are pure code motion. `tools/lint.sh` runs **eleven** pinned
> gates. The product is the
> engine plus its loopback server — the Mac app is gone — and
> `tools/install_tinytitan.sh` downloads a built release instead of compiling one;
> the browser chat window arrives the same way, from the release tag's source
> archive, with no registry account on either side. The eight installs under
> `models/` have
> receipts **valid for this folder**, because a rename invalidates them; re-issue
> with `--verify-install` if the folder moves again. **Verification uses only the
> installs already under `models/`** — never download, convert, repack or re-install
> a model to make a gate pass, and never fetch one of the installs the operator
> deleted. Report measurements, not assurances.

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
> `tools/dsh_local.sh` installs a **pinned** `@deepseek-ai/dsh` (`0.1.6-alpha.2`,
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
> `tools/dsh_route.sh --write` will touch it; and pnpm's npm-installed shim has no
> shebang, which macOS refuses to `exec` (`spawnSync pnpm ENOEXEC`) — the private
> shim execs `@pnpm/exe.darwin-arm64` instead. `tools/dsh_local.sh status` says what
> is installed.
>
> **Since 5.7 the pin is enforced rather than declared.** Both plugins support
> exactly `0.1.6-alpha.2` — `dsh-tinytitan`'s peers are exact, not ranges — and
> **refuse to run** on any other harness, including one whose version cannot be
> read. A refusal never throws: it writes one line to stderr and returns, so DSH
> boots, every other plugin loads, and removing ours leaves nothing to undo. stderr
> is not a preference — the harness prints a plugin's log records only when the boot
> itself fails, so a host-logger line would be invisible (tracker, plugins section).

## Where the work stands

| Piece | State |
| --- | --- |
| Repository | `Pummelchen/TinyTitan` (renamed 2026-09-14; the old URL redirects) |
| Checkout folder | `~/Downloads/TinyTitan` — **renamed from `~/Downloads/NVMAI`**, which invalidated every receipt and `.build`'s debug half |
| `main` | level with `origin/main`, one commit past `v5.13` (this brief); the release commit is `89d3317` |
| Release | **5.13 published** 2026-09-29 — `tinytitan-5.13-macos-arm64.tar.gz`, 15,415,573 bytes, sha256 `e5359f18…` with its `.sha256` beside it |
| Models | **8 installs, 244 GB**; every receipt bound to this path, so all load |
| Goldens stored | 16; **7 checked** here (qwen38-125b-4bit, qwen36-{4,8}, qwen35-{4b,9b}-{4,8}); the nine with no install — `ornith-{4,8}`, `qwen38-8`, `agentworld-{4,8}`, `katcoder-{4,8}`, `qwen35-2b-{4,8}` — are reported *not checked* and named in the notes |
| Audit | **28 findings, all closed**; the eleven gates in `tools/lint.sh` are what the audit left behind. The wiki's archive page was removed on 2026-09-29 when the wiki became user-only — the record is in the wiki repository's history at `6acaa8f` |
| `.build` | release rebuilt for 5.13; a clean scratch release build is part of each dry run |
| Wiki | `.qwen/wiki`, remote `TinyTitan.wiki.git`, level with `origin/master` (Changelog carries the 5.13 section); user-facing only since 2026-09-29 |
| DeepSeek Harness | pinned `0.1.6-alpha.2` and **enforced**; both plugins refuse any other version; the global harness runs the gate, the private one is refreshed but idle until its next start. The private bundle is isolated down to the caches: npm's cache/logs/user config, pnpm's home and the XDG cache/state all live under `~/.tinytitan/dsh`, so a run adds nothing to `~/.npm`, `~/Library/pnpm`, `~/.cache` or `~/.local/state` (`benchmark/test_dsh_isolation.py` pins it; verified in a simulated factory-new HOME). Since 5.11 the bundle is the delivery — the installer's source archive carries `plugins/`, and the route writer and the launcher both resolve the installed layout (`../bin`, `../models`) instead of a checkout's |
| CI | every `main` push runs CI (eleven gates, serial tests, thread-sanitizer, converter and installer gates, Markdown links) and CodeQL (Swift, 249 files); the 5.13 push is the run to watch (`gh run list`) |

## What has landed

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
is the authority, and it holds one table with no Open row. Two items are Blocked
on other people:

1. **TT-018 — the plugin's delivery is settled, one chore is not.** `dsh-tinytitan`
   **is not published to npm and will not be** (decided 2026-09-25): the installer
   fetches this project's source archive for the release tag, which carries
   `plugins/`, and installs the bundle from there, so the code reaches a user from
   the web with no registry account on either side. The catalogue half is done
   ([#5396](https://github.com/awesome-dsh-plugin/awesome-dsh-plugin/pull/5396)
   merged 2026-09-19, `dsh-tinytitan` listed, and the listing points at the repo).
   What is left of the row is the inert `Pummelchen/awesome-dsh-plugin` fork: it
   **cannot be deleted from this checkout** because the token lacks `delete_repo`,
   so remove it in the web UI or `gh auth refresh -h github.com -s delete_repo`.
2. **TT-020 — reaching the LAN manager from another machine.**
   [Discussion #7111](https://github.com/deepseek-ai/deepseek-harness/discussions/7111)
   was **answered 2026-09-19 by `PerryLink`**: the gate is the webserver schema,
   which admits exactly `127.0.0.1` and `0.0.0.0`, not the CLI guard; a
   specific-interface bind would also need its address folded into
   `resolveLanTrust`'s `trustedHosts`. Upstream's call, so the plugin works and
   nothing reaches it remotely yet. `docs/dsh-upstream-asks.md` holds all three
   asks and the replies.
3. **Carried forward, not tracked as tasks:** the Qwen 3.8 port items — QSA
   indexer selections to the GPU, a higher expert slot budget, the n-gram gather a
   token ahead. TT-021–TT-023 were closed on 2026-09-19 (no other machines; no disk
   for the ~360 GB bf16 reference), so the M1–M6 claim stays a design intent and
   Qwen 3.8 long-context stays verified only at a lowered budget.

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
  `swift run -c release TinyTitanRepack --verify-install --input-gturbo <dir>`
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
- **Say what a guard actually reads, not what it intends**, and **test a claim
  rather than trusting it** — both defects that reached a release in this project
  were claims broader or more specific than the code.
