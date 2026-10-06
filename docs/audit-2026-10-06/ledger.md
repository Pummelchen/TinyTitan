# 2026-10-06 pre-production audit

Branch `audit/2026-10-06` on MacBook Pro (M3, 24 GB, macOS 27.0) — primary and only Apple-silicon host. This page is generated from `ledger.json` by `render_ledger.py` in this directory; edit the JSON, not the Markdown.

**Open:27  Done:16  Blocked:1  Total:44**

## Table

| ID | Sev | Tier | Project | Location | Title | Category | Status | Host |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AUD-143 | S1 | A | server | `sources/TinyTitanServer/Core/ModelRouterError.swift:244-263 (acquire's barrier: the check at :245, the switch's await boundary at :250, the waiter's decrement at :258), :272-274 (switching claimed only inside switchTo), :286 (the loader's unstructured Task); tests/TinyTitanServer/ModelRouterTests.swift:57-78 (the test that measures it)` | A pending model switch can be overtaken by new work for the resident model, and in a release build the same test aborts the server bundle with signal 6 on this host | residency fairness bug, plus a signal-6 abort of the server bundle on this host (found while verifying AUD-110 and AUD-111) | OPEN | Mac (primary) |
| AUD-139 | S1 | A | release | `GitHub Release v5.18 (assets), tools/install_tinytitan.sh:272-276` | v5.18 publishes no tinytitan-5.18-tools.tar.gz, so the closed installer check refuses the newest release until the asset is backfilled | release artifact gap created by a fix; needs an action on a published release | BLOCKED | Mac (primary) |
| AUD-101 | S1 | A | launcher | `tools/server_launcher.sh:517-522` | A fresh checkout cannot fetch the model the launcher advertises: the empty-models guard exits before the install path | unreachable-fix / broken first-run path | DONE | Mac (primary) |
| AUD-102 | S1 | B | tests | `benchmark/test_launcher_install.py:104-120,162-197` | The launcher-install suite is not model-free: it fails in CI and passes locally, so CI has been red on main for three pushes | test correctness / CI gate | DONE | Mac (primary) + GitHub Actions |
| AUD-107 | S1 | A | engine | `sources/TinyTitan/Runtime/Family/PLEConstants.swift:39-44 (before); :56-101 (after)` | tableRowCount traps on a negative sidecar value: validate() never checks sign or offset order, so the corrupt-sidecar guard misses its own stated purpose | silent truncation/overflow, unchecked input | DONE | Mac (primary) |
| AUD-108 | S1 | A | server | `sources/TinyTitanLib/OpenAIRequestValidator.swift:359-385, sources/TinyTitan/Runtime/Generation/JSONSchemaNode.swift:97` | Request JSON is walked with unbounded recursion: a 1 MiB body affords ~10^5 nesting levels and no depth cap exists in the file | missing error handling, unbounded resource, network-facing input | DONE | Mac (primary) |
| AUD-109 | S1 | A | installer | `tools/install_tinytitan.sh:202-236 (verify_release_artifact), :246-258, :272-276; tools/release.sh:355-385` | The install verifies the engine tarball only if the checksum happens to download, and never verifies the tools tree it then executes | integrity / download-and-execute, fail-open check | DONE | Mac (primary) |
| AUD-121 | S1 | A | converter-gates | `benchmark/test_prepare_qwen38.py:633 FinishedOutputGuardTests` | A run without the converter's three pinned dependencies FAILS instead of skipping: 75 tests, 74 skipped, 1 failure | test that cannot distinguish 'environment missing' from 'code broken' | DONE | Mac (primary) |
| AUD-124 | S1 | A | repack | `sources/TinyTitanRepack/Core/Verification/VerifiedInstallTool.swift:213-224, :256, :368-390 (before); Core/Verification/PackedExpertLayoutVerification.swift:34-141, :143-314 (after)` | A MoE install's declared routed-expert width is never checked against its payload: the resident check skips experts and the layout check never compares bytes to bits | integrity verifier has no coverage on the shipped shapes | DONE | Mac (primary) |
| AUD-103 | S2 | B | python-tooling | `pyproject.toml:20-24` | Five of the nine Python pitfalls the audit standard names have no rule behind them, and the config comment claims they do | check coverage gap | START | Mac (primary) |
| AUD-104 | S2 | A | lint-gates | `tools/lint.sh:319-330` | The converter expert-order gate reports nothing when the converter dependencies are missing | gate fails open | START | Mac (primary) |
| AUD-105 | S2 | B | release | `tools/release.sh, docs/release-process.md:3` | Nothing in the release runbook requires CI to be green on the tag, and v5.18 was published while its commit's CI was failing | missing gate | START | Mac (primary) + GitHub |
| AUD-141 | S2 | A | runtime | `sources/TinyTitan/Runtime/Inference/Model+SchemaValidation.swift:270-272` | The load path cross-checks only one expert per layer, so a width or shape lie confined to any later expert loads and answers wrongly | incomplete validation on the load path (found by the AUD-124 fix, not fixed by it) | START | Mac (primary) |
| AUD-106 | S2 | C | docs | `docs/handover-tinytitan.md:1-37` | The handover brief describes release 5.15 as current while 5.16, 5.17 and 5.18 are published | documentation drift | OPEN | Mac (primary) |
| AUD-115 | S2 | A | contract | `sources/TinyTitanRepack/Core/Format/SSDAIJSON.swift:121 → sources/TinyTitanFormat/SSDAIManifestV1.swift:321` | bitWidthOverridesHonored is written and decoded but never reaches the runtime Manifest: a documented contract field no consumer can read | dead contract field | OPEN | Mac (primary) |
| AUD-116 | S2 | A | repack | `sources/TinyTitanRepack/Core/Format/SSDAIJSON.swift:149` | A per-tensor quant override whose stem equals a slot name is silently skipped — the failure class the hand-written decoder exists to fix | silent drop on a numerics-affecting field | OPEN | Mac (primary) |
| AUD-117 | S2 | A | contract | `sources/TinyTitanRepack/Core/Format/SSDAIJSON.swift:79-96 → sources/TinyTitan/Infrastructure/ModelIO/ManifestReader.swift:322-327` | hc/indexer/ple geometry is emitted for one family only, and the reader treats every absent optional field as fine, so a family that needs them loads unvalidated | unvalidated external input / contract drift | OPEN | Mac (primary) |
| AUD-119 | S2 | C | docs | `AGENTS.md (Test rules, 'The converter's gate is two python suites') and benchmark/requirements.txt:4-5` | Documented converter command names two suites where CI runs three, and the requirements comment states a test count the baseline does not reproduce | documentation drift / stale measurement quoted as fact | OPEN | Mac (primary) |
| AUD-120 | S2 | B | ci | `.github/dependabot.yml (absent) and Package.resolved` | No dependency vulnerability feed for Swift: swift-nio 2.100.0 and swift-transformers are pinned but never checked against a CVE source | missing CVE coverage on one of three languages | OPEN | Mac (primary) |
| AUD-122 | S2 | B | tests | `sources/TinyTitan/Infrastructure/ModelIO/ArchConfig+Manifest.swift and 6 more` | Seven production files have zero covered lines with no model gate explaining it | coverage gap on non-gated code | OPEN | Mac (primary) |
| AUD-123 | S2 | A | fleet | `plugins/dsh-lan-manager/src/net.js:38 and src/config.js:38` | The LAN manager admits link-local peers by default and its default group key is a published literal | permissive default on a network-facing surface | OPEN | Mac (primary) |
| AUD-126 | S2 | B | tests | `benchmark/test_launcher_ram.py:135, :194 and 5 more` | Six benchmark suites skip when models/ has no install, so their gate is a no-op on the host that runs it most | coverage gap on a CI gate | OPEN | Mac (primary) |
| AUD-127 | S2 | C | tests | `tests/TinyTitanRepack/Core/Format/Qwen4ExpArchInfoTests.swift:141-143` | A guard-else-return inside a test body passes green when its env var is unset, and is not recorded as a skip | test that asserts nothing on the path it did not take | OPEN | Mac (primary) |
| AUD-130 | S2 | B | plugins | `plugins/dsh-tinytitan/src/route.js:45-46, :99-100 and src/config.js:294-351` | context and maxTokens are plumbed into the route writer but resolveConfig never emits them, so every route write uses ROUTE_DEFAULTS | surface wired to nothing (§5) | OPEN | Mac (primary) |
| AUD-132 | S2 | A | fleet | `plugins/dsh-lan-manager/src/router.js:186-192` | The origin guard only rejects an Origin that is present: a mutating request with no Origin header passes it outright | CSRF guard with an absent-header hole | OPEN | Mac (primary) |
| AUD-133 | S2 | A | fleet | `plugins/dsh-lan-manager/src/discovery.js:238-248` | A failed tailscale or Bonjour probe is swallowed by a per-source catch, so /peers is quietly short rather than reporting a degraded probe | silent failure on a network path | OPEN | Mac (primary) |
| AUD-134 | S2 | A | memory | `sources/TinyTitanMemory/MemoryService+Sessions.swift:39, :42, :202; MemoryService+Consolidation.swift:29-31, :72-73; sources/TinyTitanMemory/ContinuityJournalStore.swift:52` | Journal and store reads fall back to `?? []` / `.empty` on a thrown error, and that fallback is not covered by journalFailed, so a broken memory answers 'there is nothing' | silent failure, error swallowed into an empty answer | OPEN | Mac (primary) |
| AUD-135 | S2 | B | memory | `sources/TinyTitanMemory/MemoryService+Maintenance.swift:77-79` | expireSessionLog returns true after a try?-wrapped compactJournal, so a failed compaction reads as an expired log | silent failure | OPEN | Mac (primary) |
| AUD-136 | S2 | B | server | `sources/TinyTitanLib/OpenAIRequestValidator.swift:88-92, :140-146` | reasoning_budget_tokens and parallel_tool_calls are accepted from the wire and not enforced | surface wired to nothing, publicly disclosed (§5) | OPEN | Mac (primary) |
| AUD-140 | S2 | A | installer | `tools/dsh_local.sh:256-275` | The sanctioned browser client downloads a Node tarball and runs what it extracts without checking any digest | integrity / download-and-execute, fail-open check | OPEN | Mac (primary) |
| AUD-142 | S2 | A | engine | `sources/TinyTitan/CPUEngine/AffineSnapshot.swift:131, sources/TinyTitan/Tokenization/Detokenizer.swift:51, sources/TinyTitanLib/ServerModelSession+Loading.swift:293, sources/TinyTitanRepack/Core/Format/ArchInfo.swift:168, sources/TinyTitanRepack/Core/Format/SSDAILayoutValidator.swift:5, sources/TinyTitanBench/CPUCommands.swift:229, sources/TinyTitanFleet/Command/main.swift:248-250` | Seven metadata reads still have no size bound, and they sit at four different trust boundaries, so they do not all want the same cap | unbounded memory on an input file (found by the AUD-113 fix, not fixed by it) | OPEN | Mac (primary) |
| AUD-144 | S2 | A | memory | `sources/ContinuityCore/Persistence/Journal.swift:116, :137, :382, :416` | Four journal openers create files without O_NOFOLLOW, the pattern AUD-110 just closed in the installer | symlink following on a predicted path | OPEN | Mac (primary) |
| AUD-110 | S2 | A | repack | `sources/TinyTitanRepack/Core/System/Posix.swift:32 (before); :29-44 (after)` | openCreateRW is the only opener without O_NOFOLLOW, and it is used for weight outputs | symlink following / TOCTOU on a predicted path | DONE | Mac (primary) |
| AUD-111 | S2 | A | engine | `sources/TinyTitan/Runtime/Inference/RealForwardRunner.swift:478, :489 (before); RealForwardRunner+Diagnostics.swift:9-28 (after)` | Two env-named trace files open 0o644 with no O_NOFOLLOW: world-readable routing traces | permissive file mode + symlink following | DONE | Mac (primary) |
| AUD-112 | S2 | A | repack | `sources/TinyTitan/Infrastructure/ModelIO/Sha256Verifier.swift:34, 50, 57, 65, 68, 74 (before); :1, :25-62 (after)` | Six CommonCrypto SHA-256 return values are discarded on the integrity-hash path | unchecked return value | DONE | Mac (primary) |
| AUD-113 | S2 | A | engine | `sources/TinyTitan/Runtime/Family/PLEConstants.swift:33 (before); PLEConstants.swift:35-46, sources/TinyTitan/Runtime/Prefill/ANEPrefillAttention.swift:172-189, sources/TinyTitanServer/Core/ModelCatalog.swift:278-293 (after)` | ple_constants.json is read with an uncapped Data(contentsOf:) although its sibling receipts cap the same file class | unbounded memory on a model-supplied file | DONE | Mac (primary) |
| AUD-114 | S2 | B | build-config | `Package.swift:81-83` | A build-config comment still claims the package cannot be consumed as a dependency, which was measured false on 2026-10-02 | contract drift / stale documentation on a load-bearing rule | DONE | Mac (primary) |
| AUD-118 | S2 | A | contract | `sources/TinyTitanRepack/Core/Verification/VerifiedInstallReceiptWriter.swift:4 and sources/TinyTitan/Infrastructure/ModelIO/VerifiedInstallReceipt.swift:76` | The receipt file name is declared twice, once per side of the contract | duplication on a cross-target constant | DONE | Mac (primary) |
| AUD-125 | S2 | A | memory | `sources/TinyTitanMemory/ContinuityStore.swift:110` | memory_delete swallows every non-notPersisted archive error in an empty catch (reclassified from S1: the dominant failure path was already rethrowing) | silent failure, wrong result reported to the model | DONE | Mac (primary) |
| AUD-129 | S2 | B | docs | `docs/agent-memory.md:141 and docs/side-engine-tasks.md:328` | Two documents tell the reader to pass --models-directory; the parser's flag is --models-dir, so the documented flag cannot work | surface wired to nothing (§5) | DONE | Mac (primary) |
| AUD-128 | S3 | C | tests | `tests/ (18 sites, see evidence)` | Test bodies that cannot fail: preconditions recorded as expressions, one self-referential digest assertion, and non-throw-only bodies | tests that assert nothing | OPEN | Mac (primary) |
| AUD-137 | S3 | B | server | `sources/TinyTitanLib/ServerInference.swift:101 and OpenAIRequestValidator.swift:32-33` | An unreachable ?? 262_144 fallback on a non-empty constant array | defensive code for a case that cannot happen | OPEN | Mac (primary) |
| AUD-138 | S3 | C | memory | `sources/TinyTitanMemory/MemoryRetrieval.swift:52, :233; sources/TinyTitanMemory/ContinuityJournalStore.swift:71, :92` | Four try?-to-empty reads split off AUD-134: recall quality on a background path, and two protocol methods with no production caller | error swallowed into an empty answer (low reach) | OPEN | Mac (primary) |
| AUD-131 | S3 | C | docs | `docs/release-notes-v5.8.md:132` | release-notes-v5.8.md still advertises TINYTITAN_KEEP_WIRED as a live tri-state although the knob was deleted by 3eb11cf and the repo has a Superseded-banner convention for exactly this | stale documentation, documented switch with no consumer (L0/§6) | DONE | Mac (primary) |

## Detail

### AUD-143 — A pending model switch can be overtaken by new work for the resident model, and in a release build the same test aborts the server bundle with signal 6 on this host

- **Severity / tier:** S1 / Tier A
- **Project:** server
- **Location:** `sources/TinyTitanServer/Core/ModelRouterError.swift:244-263 (acquire's barrier: the check at :245, the switch's await boundary at :250, the waiter's decrement at :258), :272-274 (switching claimed only inside switchTo), :286 (the loader's unstructured Task); tests/TinyTitanServer/ModelRouterTests.swift:57-78 (the test that measures it)`
- **Category:** residency fairness bug, plus a signal-6 abort of the server bundle on this host (found while verifying AUD-110 and AUD-111)
- **Status:** OPEN
- **Host:** Mac (primary)
- **Discovered by:** AUD-110/AUD-111 verification, full serial suite; root-caused by the diagnosis pass on 2026-10-06

**Evidence before.** Two manifestations on this host, and the split between them was measured after the first cut at this row, which had conflated them. (1) The ordering defect is deterministic: every run in which the assertion was evaluated records `log.loads == ["load alpha_4-Bit", "load small-2b"]` (4 of 4: two filtered full-suite runs and two single-test runs), so the third request for the resident model never queues behind the pending switch -- it acquires alpha while alpha is still resident and runs before the switch commits. The actor's own comment (ModelRouterError.swift:74-76) states the opposite guarantee: while a switcher waits, new work for the resident model queues behind it. So the test's expectation is legitimate and the defect is in the router, not in the assertion. The candidate hole is visible in the code: the waiter decrements `pendingSwitches` at :258 when it wakes, while the switch it was waiting for only reaches `switching = true` at :272 across the await boundary at :250, which leaves a window in which a newly arriving caller takes branch :245 (`resident.id == target, !switching, pendingSwitches == 0`). Root cause is *not* established beyond that: the trap and the window were both read in the source, and no experiment removing the third task could be run without editing `tests/`. (2) The signal-6 abort is flaky, not certain: `--filter 'ModelRouterTests'` aborted 1 of 3 runs and the single test aborted 4 of 6 (the other 2 failed the assertion alone); the sibling two-task gate test `aSwitchWaitsForTheInFlightGenerationToDrain` ran clean 4 of 4, so the trap is specific to the three-task/two-waiter shape. It is not repo misuse of an executor -- `serialExecutor`, `withTaskExecutorPreference`, `NonisolatedNaggingExecutor` and `TaskExecutor` appear nowhere in `sources/`, and `ModelRouter`, `RoutingGate` and `RoutedStubModel` are plain actors; the closest candidate is the unstructured `Task { loader }` inside the actor at :286 inheriting the router's executor preference, and the trap itself is raised inside the closed concurrency runtime.
    It is machine-flipped, not a local-only curiosity: on GitHub CI the same test PASSES with 3 loads (run 37265257867, both the `test` and `thread-sanitizer` jobs), which is the other half of the same finding -- the ordering is timing-dependent and this host loses it. CI does not currently see it either way, because the recent `main` runs fail before `swift test` (python `test_launcher_install.py`, 6 failures). The earlier wording of this row said the filtered run aborted in 20 consecutive runs: what was 20 of 20 was the *exit code*, and half of those were assertion failures; the count above replaces that claim. `main` carried both halves before this branch existed -- neither `ModelRouter` nor `ModelRouterTests.swift` is touched by any commit on it, and they were last changed by 0cdba76 and 715df2c, both on `main` -- so this is a red-on-main finding. It is S1 for two reasons: a fairness guarantee the server documents and does not keep, and a bundle that can abort and then reports nothing, which is how a lost count reads as a small suite to the release gate.

**Evidence after.** Not fixed, and two candidate fixes were measured and reverted, so this row's next step is a different one. Controls first: the unmodified tree runs green in debug (6 of 6 filtered runs exit 0), so the debug config never shows this defect at all -- which also means a debug-green run proves nothing about a change to it. In release the same unmodified tree fails 7 of 7, splitting 3 runs that report the ordering failure (`log.loads == ["load alpha_4-Bit", "load small-2b"]`, i.e. the third request acquires the still-resident model instead of queueing behind the pending switch) and 4 that abort with the signal-6 trap before the assertion is evaluated.
    Attempt A, the hole as read: hold the barrier until the switch is committed -- claim `switching = true` in `acquire` before the await at :250, drop `pendingSwitches` only at that claim, make the cancellation check at the loop top synchronous so no suspension sits between the wake and the claim, guard the re-queue against counting the same caller twice, and release the held barrier on the cancellation exit. It builds clean and keeps debug green (5 of 5), but debug was green without it. In release it aborts 7 of 7 -- the bundle stops reporting *more* reliably, not less, so the change cannot even be measured there. Reverted.
    Attempt B, the minimal form of the same idea: replace only the loop-top `try Task.checkCancellation()` at :244 with `if Task.isCancelled { throw CancellationError() }`, which removes the single await between the waiter's decrement at :258 and the claim, and leaves the rest of the loop untouched. Release then aborts 6 of 7, and the one run that reports records the *same* two-load sequence as the unmodified tree. So the window that the code reading identified is not the defect, or not the whole of it: the resident-model fast path at :245 is still reached while a switch is pending by a route this analysis has not found.
    Probe, on the trap's own hypothesis: the loader hop at :286 was changed from the unstructured `Task {}` inherited from the actor to `Task.detached`, on the reading that an unstructured task created inside an actor carries that actor's executor assumption. It aborted 3 of 3, so that hop is not the trap's cause by itself.
    What has to be true for this to close: a build in which the release-config bundle reports rather than aborts, because until then the ordering defect can only be sampled, not measured, and any change to `acquire` is unverifiable. The mechanism to find next is why release alone traps on this host while GitHub CI runs the same release suite green (run 37265257867 records the three-load sequence), and what else lets a caller pass the `!switching, pendingSwitches == 0` test at :245 while a switcher is awake. Nothing here is landed; the tree is at the state these controls describe.

### AUD-139 — v5.18 publishes no tinytitan-5.18-tools.tar.gz, so the closed installer check refuses the newest release until the asset is backfilled

- **Severity / tier:** S1 / Tier A
- **Project:** release
- **Location:** `GitHub Release v5.18 (assets), tools/install_tinytitan.sh:272-276`
- **Category:** release artifact gap created by a fix; needs an action on a published release
- **Status:** BLOCKED
- **Host:** Mac (primary)
- **Discovered by:** AUD-109 fix, sibling consequence

**Evidence before.** AUD-109 made the tools tree a verified release asset. `gh release view v5.18 --repo Pummelchen/TinyTitan --json assets` lists the engine archive, its .sha256, the library archive and its .sha256 -- and no tools asset, because no release script before 8057a76 ever published one. The installer at :272 therefore dies on 'No checksum published for the tools download' for every tag up to and including v5.18, including the default newest-release path.

**Evidence after.** Expected: `gh release upload v5.18 <tools archive> <tools archive>.sha256 --clobber` puts the asset and its digest on the existing Release, the archive is `git archive --prefix=tinytitan-5.18-tools/ v5.18`, and a `bash tools/install_tinytitan.sh --yes --no-model --version v5.18` in a clean VM then reports 'tools checksum verified'. Alternatively cut v5.19, which publishes it by construction, and merge AUD-109's installer after that release. Not taken by the auditor: uploading to, or otherwise mutating, a published release is an action on a shared surface, and the audit's own rule forbids it.

**Blocked.** owner `repository owner (Pummelchen)` — Requires writing to the published v5.18 Release (or cutting v5.19), which is the repository owner's action, not an auditor's. The audit branch does not touch releases; the merge of AUD-109 should be sequenced after one of the two options above.

### AUD-101 — A fresh checkout cannot fetch the model the launcher advertises: the empty-models guard exits before the install path

- **Severity / tier:** S1 / Tier A
- **Project:** launcher
- **Location:** `tools/server_launcher.sh:517-522`
- **Category:** unreachable-fix / broken first-run path
- **Status:** DONE
- **Host:** Mac (primary)
- **Discovered by:** CI run 37329361684 + local reproduction

**Evidence before.** 5.18's headline is 'a model that is not on disk can be fetched from the launcher'. Reproduced on this host: `TINYTITAN_MODELS_DIR=<empty> TINYTITAN_LAUNCHER_DRY_RUN=1 bash tools/server_launcher.sh --dry-run --client server --model katcoder --bits 4` prints 'ERROR: no install under <dir> matches the built-in list' and exits 2. tinytitan_static_catalog (tools/tinytitan_models.sh:458-462) returns 1 when models/ holds none of the built-in models, and server_launcher.sh exits 2 at :521, i.e. before the fetch/offer code at :569-616 and before --model handling at :620. The offer helper itself is fine — tinytitan_missing_offers on an empty directory is tested and passes (benchmark/test_launcher_install.py:MissingOffersTests) — so the defect is the composed path, not the helper. Expected-correct: a --model/--bits request for a width that is not on disk reaches install_model_key and, piped or --dry-run, prints the tools/install_models.sh command and downloads nothing.

**Fix.** server_launcher.sh: the empty-static-catalog branch stops only when there is also nothing to fetch (TINYTITAN_CATALOG_MISSING empty). With offers available it says so in one line and falls through to the menu and the --model path, which already re-read the catalog after an install. The block comment above it is corrected too: it asserted the menu never offers what is not on disk, which stopped being true when the fetch rows were added.

**Evidence after.** Before: `TINYTITAN_MODELS_DIR=<empty> bash tools/server_launcher.sh --dry-run --client server --model katcoder --bits 4` -> exit 2, 'ERROR: no install under … matches the built-in list'. After: exit 1 and 'Install it with:  tools/install_models.sh katcoder'. Menu with an empty dir: rows 1-16 drawn, 'Rows 1-16 are not installed yet', choice 9 -> exit 1 + 'Install it with: tools/install_models.sh qwen38flash'; EOF at the menu -> exit 1, nothing downloaded. The empty dir stayed empty in all four cases. New EmptyModelsDirTests fail (2 failures) against HEAD's launcher and pass against the fixed one; test_launcher_install 16/16 OK.

**Commit.** `see the audit(AUD-101) commit`

### AUD-102 — The launcher-install suite is not model-free: it fails in CI and passes locally, so CI has been red on main for three pushes

- **Severity / tier:** S1 / Tier B
- **Project:** tests
- **Location:** `benchmark/test_launcher_install.py:104-120,162-197`
- **Category:** test correctness / CI gate
- **Status:** DONE
- **Host:** Mac (primary) + GitHub Actions
- **Discovered by:** gh run list + gh run view --log-failed

**Evidence before.** CI on main: ea5de8c FAIL, 2d510fc FAIL, 9bb9051 FAIL (last green f5f1204). The failing step is 'Installer gates', `FAILED (failures=6)` in test_launcher_install. first_missing() returns a candidate when a catalogue model is NOT installed, so on a CI runner with an empty models/ nothing skips, the six menu/missing-model tests run against the launcher's exit-2 path, and assertEqual(returncode, 1) fails with '2 != 1'. Locally the same 14 tests pass (`python3 -m unittest benchmark.test_launcher_install` -> OK) because this checkout has two installs. Its sibling suites do guard this case — test_launcher_ram.py:135 and test_launcher_port.py:108 skipTest('no install under models/ and no built server to list one'). AGENTS.md states tests never load a model and no gate may fetch one, so the suite must be correct against an empty models/.

**Fix.** The suite's expectation was right and is kept: returncode 1 plus the exact install_models.sh command. What was missing was coverage of the state CI is actually in, so the same assertion only ran against whatever this machine happens to have. EmptyModelsDirTests creates an empty models dir and points TINYTITAN_MODELS_DIR at it, and first_catalogue_model() picks a launcher name without consulting the disk. No assertion was loosened and no test was skipped: 14 tests before, 16 after.

**Evidence after.** Pre-fix launcher: `Ran 16 tests … FAILED (failures=2)`. Fixed launcher: `Ran 16 tests in 1.357s OK`. Sibling suites still green on this host: test_launcher_ram, test_launcher_port, test_progress, test_coder_clients -> Ran 45 tests, OK.

**Commit.** `see the audit(AUD-101) commit`

### AUD-107 — tableRowCount traps on a negative sidecar value: validate() never checks sign or offset order, so the corrupt-sidecar guard misses its own stated purpose

- **Severity / tier:** S1 / Tier A
- **Project:** engine
- **Location:** `sources/TinyTitan/Runtime/Family/PLEConstants.swift:39-44 (before); :56-101 (after)`
- **Category:** silent truncation/overflow, unchecked input
- **Status:** DONE
- **Host:** Mac (primary)
- **Discovered by:** L3 line pass

**Evidence before.** Read at the cited lines: ngramHeadsOffsets/ngramHeadsVocabSizes are [Int64] decoded straight from ple_constants.json (:11-12, JSONDecoder :34) and widened with UInt64(offset) + UInt64(vocab) (:43). UInt64() of a negative Int64 is a fatal trap, not a throw. validate() (:56-89) compares counts and the headCount*pleHeadDim product but never sign, and never checks that offsets ascend. The docstring at :48-55 says the check exists 'to turn a corrupt sidecar into a report rather than a trap' — it does not cover the trap this path can hit.

**Fix.** `tableRowCount` is a throwing function and the only way to ask for the count, so no call order can reach the conversion unchecked: it walks the head tables the way the producer builds them, requiring offset[i] == sum(vocab[0..i]), vocab > 0 everywhere, paired arrays, a non-empty table, and `addingReportingOverflow` rather than `+=` for the running total. `validate()` calls it, so the load-time gate refuses the addressing as well as the width. RealForwardRunner+BuildCore.swift:333 passes `try constants.tableRowCount()`. Note for the record: the commit message says eleven tests; the suite is twelve with eight added -- the ledger number is the checked one.

**Evidence after.** Before: a standalone `UInt64(negative Int64)` under this toolchain dies with `Fatal error: Negative value is not representable`, exit 133 -- the exact failure `ple_constants.json` could produce at RealForwardRunner+BuildCore.swift:333. After: `swift test --no-parallel --filter PLE` -> 26 tests in 4 suites passed, PLEConstantsGeometryTests 4 -> 12 tests, including `acceptsProductionConstants`, which feeds the checkpoint's own constants (the ple_golden fixture, whose offsets equal the installed model's ple_constants.json) through validate() and pins 320001446 rows, so the new invariants are the checkpoint's rules and not a refusal waiting for a working model. Full `swift test --no-parallel` exit 0; `swift build -c release` exit 0 (583 steps, 121.52 s); swift-format, swiftlint, force-cast, func-length all ok. Sibling audit: PLEConstants was the only model-supplied signed-to-unsigned conversion in sources/ (grep for `[Int64]` decodables finds only its three arrays; CPUEngine/SafeTensors.swift:119 already guards `offsets[0] >= 0`; ModelCatalog.sizeBytes is our own number, used only for display), and the guard against a repeat is that the count now cannot be obtained without the checks running.

**Commit.** `fa1ca79`

### AUD-108 — Request JSON is walked with unbounded recursion: a 1 MiB body affords ~10^5 nesting levels and no depth cap exists in the file

- **Severity / tier:** S1 / Tier A
- **Project:** server
- **Location:** `sources/TinyTitanLib/OpenAIRequestValidator.swift:359-385, sources/TinyTitan/Runtime/Generation/JSONSchemaNode.swift:97`
- **Category:** missing error handling, unbounded resource, network-facing input
- **Status:** DONE
- **Host:** Mac (primary)
- **Discovered by:** L4 security pass

**Evidence before.** Paths on the original row were wrong and are corrected above: JSONValue lives at sources/TinyTitan/Tokenization/JSONValue.swift and JSONSchemaNode at sources/TinyTitan/Runtime/Generation/JSONSchemaNode.swift, not under TinyTitanLib. What was checked: validateSchemaKeys recursed into every object value and array element with no depth parameter; JSONSchemaNode.compile recursed through properties (:156) and items (:173) with none; jinjaSendableValue and JSONValue.init(from:) recurse over the same shape; grep for 'depth|maxDepth|nesting' found no bound in any of the four files. The claim that a 1 MiB body therefore affords ~10^5 levels was measured and is FALSE: on this toolchain Foundation's parser accepts 512 nesting levels and throws at 513 (probe: 512 -> decoded, 513 -> "Too many nested arrays or dictionaries"), and the handler maps that to a 400 invalid_json (HTTPServerHandler+Chat.swift:129-135). No stack overflow was reachable from a request body.

**Fix.** The bound is moved from the parser into the code that walks: JSONSchemaNode.maximumNestingDepth = 64, checked by compile on each level it descends through properties or items, and by validateSchemaKeys over the same document. validateSchemaKeys no longer counts a scalar as a step, which is what made the two walkers disagree (a schema that compiled at 64 was refused as a tool at 32). The wire answer for a refused depth is a 400 with code invalid_tool_schema, param tools; a compiled-schema refusal is malformed(...) mapped to unsupported_value. jinjaSendableValue is bounded transitively on both request paths (validateSchemaKeys runs first for a tool schema; historical arguments are bounded by the parse). foundationObject() and the encode/jinja walks over free-form metadata stay bounded by the parse rather than by the schema cap: they echo data the same request already carried, and refusing them at 64 would reject valid requests for no gain. Recorded as a decision, not a scope cut -- the row asked for a stated maximum depth, a 400 past it, and no reachable overflow, and each is now true of this repository rather than of Foundation.

**Evidence after.** New tests, all green: JSONSchemaCompileTests.theNestingCapIsExactlyWhereItSaysItIs (64 accepted, 65 refused, via properties and via items), .aRefusedDepthIsAMalformedSchemaErrorThatNamesTheCap, .theDecoderBoundsWhatTheCompilerCanBeHanded (513 still throws -- the tripwire if the platform bound moves); OpenAIValidationTests.aToolSchemaPastTheNestingCapIsRefusedAsBadRequest asserts the 400 envelope shape through the real wire types. Affected suites: 111 tests in 6 suites passed. Full package suite `swift test --no-parallel`: exit 0, 1540 tests in 7 Swift Testing targets, 0 failures. swift-format, swiftlint and func-length clean. Regression surface measured before choosing 64: the deepest schema in any client config cached on this host nests 11 levels, and no JSON owned by this repository comes within 50 of the cap.

**Commit.** `14710ba`

### AUD-109 — The install verifies the engine tarball only if the checksum happens to download, and never verifies the tools tree it then executes

- **Severity / tier:** S1 / Tier A
- **Project:** installer
- **Location:** `tools/install_tinytitan.sh:202-236 (verify_release_artifact), :246-258, :272-276; tools/release.sh:355-385`
- **Category:** integrity / download-and-execute, fail-open check
- **Status:** DONE
- **Host:** Mac (primary)
- **Discovered by:** L4 security pass

**Evidence before.** install_tinytitan.sh:214 fetches $asset.sha256 under `if curl -fsSL`; on failure the else branch at :227-229 only warns 'No checksum published for $tag; continuing without verification' and installs. tools/release.sh:320/349 always generate the .sha256 and :451 uploads it as a release asset, so the branch is a fail-open on a path that always has a checksum. The second artifact — src_url at :246, the tag archive holding tools/, extracted to $SRC_PATH and executed later by tools/server_launcher.sh — has no checksum code at all. Severity rationale, recorded so it is not later re-litigated: both downloads come over HTTPS from the same github.com origin as the digest itself, so the digest is corruption and rename detection, not publisher identity; that is S1, not a supply-chain S0.

**Fix.** Both downloads now go through one helper with no fall-through: a missing `shasum`, an unfetchable `.sha256`, or a mismatch each removes the staging directory and dies BEFORE anything is unpacked, so the engine binaries and the tools tree are each installed only on proof. The warning branch is gone. The tools stopped being a source archive (`archive/refs/tags/$tag.tar.gz`) and became a release asset, `tinytitan-<version>-tools.tar.gz`, which `tools/release.sh` now stages with `git archive` of the tag, asserts member by member (install_tinytitan.sh, server_launcher.sh, install_models.sh, tinytitan_models.sh, dsh_local.sh, plugins/dsh-tinytitan/, Package.swift, sources/TinyTitanLib/), checksums, uploads beside the other two, and enforces in the release notes through TOOLS_SHA256_PENDING / TOOLS_BYTES_PENDING -- which is why the digest substitution order comment and docs/release-process.md changed shape as well. The installer's refusal messages name what to do (a newer --version, --from-source, retry, or report a damaged release) rather than only what failed.

**Evidence after.** benchmark/test_release_installer_verification.py: 18 tests, OK in 1.4s. Five behavioural cases run the REAL installer (temp HOME, TINYTITAN_ROOT, a stub `curl` serving fixture archives, real SHA-256 digests, the real `shasum`) and assert both 'engine checksum verified' and 'tools checksum verified', the release-asset URL rather than archive/refs/tags, and that every digest was fetched. Four refusal cases -- engine digest missing, engine tampered, tools digest missing, tools tampered -- each assert a non-zero exit, the specific message, and that NOTHING landed in the install root afterwards. One case builds a PATH containing every system tool except the checksum tool and asserts the install refuses instead of installing unverified bytes. Negative control: the same harness pointed at HEAD~1's installer fails 12 of the 18, so they test the change and not their own fixtures. Gates: bash -n parses; tools/lint.sh shell ok (24 scripts, system bash 3.2.57); shellcheck 0.11.0 ok with no warnings; tools/lint.sh python ok (ruff 0.16.7, the pinned version installed with `pipx install --force ruff==0.16.7`; homebrew carries 0.16.10). Converter gate test_prepare_qwen38 + test_qwen38_resume_e2e: 75 tests OK. Sibling sweep: install_models.sh:620-623 checks content-length only, but the repacker hashes every payload against the manifest and the verified-install receipt is what the runtime demands, so the bytes are proven before use -- no row opened. tools/dsh_local.sh:264 downloads a Node tarball and executes what it extracts with no digest at all: same defect class, different surface, opened as AUD-140. Sequencing consequence, stated in the commit and in the runbook: the closed check requires an asset no published release carries, so this installer refuses v5.18 until it is backfilled or v5.19 is cut -- AUD-139.

**Commit.** `8057a76`

### AUD-121 — A run without the converter's three pinned dependencies FAILS instead of skipping: 75 tests, 74 skipped, 1 failure

- **Severity / tier:** S1 / Tier A
- **Project:** converter-gates
- **Location:** `benchmark/test_prepare_qwen38.py:633 FinishedOutputGuardTests`
- **Category:** test that cannot distinguish 'environment missing' from 'code broken'
- **Status:** DONE
- **Host:** Mac (primary)
- **Discovered by:** baseline run, this host

**Evidence before.** Reproduced during the §3 baseline: `/tmp/tt-audit/venv/bin/python -m unittest test_prepare_qwen38 test_qwen38_resume_e2e` in a venv that lacks numpy/safetensors/ml_dtypes → `AssertionError: 'already holds a finished snapshot' not found in "missing dependency: No module named 'ml_dtypes' …"`, `Ran 75 tests`, `FAILED (failures=1, skipped=74)`, exit 1. (The row originally recorded `Ran 87 tests … OK` for the with-deps run; re-measured on this host the two converter suites are 75 tests with and without the pins — the delta is skip vs run, not count.)

**Fix.** FinishedOutputGuardTests was the only class in the file without the module-availability skip its nine siblings carry. It spawns tools/prepare_qwen38.py, which exits at import time (line 70) when a pin is missing — before argument handling — so the guard it asserts can never be reached in that environment. One `@unittest.skipIf(prepare is None, …)` decorator added. Not an assertion change: the class body is untouched and still fails if the guard regresses.

**Evidence after.** Dep-less venv: `Ran 75 tests in 0.001s / OK (skipped=75)`, exit 0 (was failures=1, skipped=74, exit 1). Pinned interpreter 3.13.16 with numpy/safetensors/ml_dtypes: `Ran 75 tests in 34.609s / OK`, exit 0, and `-v test_prepare_qwen38.FinishedOutputGuardTests` shows the single test running and passing, so the guard still bites. `tools/lint.sh python` clean under the pinned ruff 0.16.7.

**Commit.** `fc68d72`

### AUD-124 — A MoE install's declared routed-expert width is never checked against its payload: the resident check skips experts and the layout check never compares bytes to bits

- **Severity / tier:** S1 / Tier A
- **Project:** repack
- **Location:** `sources/TinyTitanRepack/Core/Verification/VerifiedInstallTool.swift:213-224, :256, :368-390 (before); Core/Verification/PackedExpertLayoutVerification.swift:34-141, :143-314 (after)`
- **Category:** integrity verifier has no coverage on the shipped shapes
- **Status:** DONE
- **Host:** Mac (primary)
- **Discovered by:** L2 module pass + §5 facade sweep

**Evidence before.** Read at the cited lines and confirmed by the auditor. validateQuantAgainstResident `continue`s every u32 entry once expertsPerLayer != 0 (:221-224, with a comment saying the widths live in packed_experts/layout.json), and the dominant-width guard returns for the same condition (:256). validatePackedExpertLayout (:368-390) compares expertStride/numLayers/expertsPerLayer against the manifest and checks counts, offsets and alignment — but PackedExpertsLayout (VerifiedInstallManifest.swift:28-33) carries no width field, and grep for weightBits across the whole file finds only :197, :198, :211 and :258, all inside the resident path. So quant.routedExpert.weightBits is compared against nothing for exactly the MoE installs the product ships (35B-A3B, 125B-A6B), while the runtime dequantizes with it and ManifestIdentity turns it into the `_<bits>-Bit` id. The writer's own comment names the failure mode (SSDAIJSON.swift:140-144): 'the word count changes, the strides still divide evenly, every shape check passes, and the model answers fluently and wrongly'.

**Fix.** The packed layout is made to prove its own widths. `validatePackedExpertLayout` now reads the `quant` block it used to ignore and, for every expert of every layer, runs a new `PackedExpertBytes.validate`: a U32 slice's byte extent divided by its element count implies a width, that implied width must equal the slice's own `bits` annotation where it carries one, and it must equal the declared routed-expert slot for every slice; a BF16 slice must be exactly two bytes an element; and the slices must sum to the blob that rounds up to the declared `expertStride`. Every multiply and add is `addingReportingOverflow`/`multipliedReportingOverflow`, so a `UInt32.max` shape reports `too many bf16 values to count in bytes` instead of wrapping into a passing sum. The derived width is bounded by the format layer's own 1...32, not by the 4-or-8 the resident check insists on: an install whose bytes match its description is a truthful install even in a withdrawn width, and refusing it here would make the verifier stricter than the load path with a message telling the user to re-download weights that are fine. The check runs BEFORE the file-hash loop, and the layout file is now read once into a decoded value that both the check and the index-consistency pass use, so the attestation cannot report `all 104 files hash correctly` over bytes whose description already failed. Tensors are walked in key order so the first complaint is the same on two machines. The dense case is the sibling risk and is pinned: the verifier's `expectedLayerSize == 0` shortcut runs before the `quant` block is asked for (PackedExpertLayoutVerification.swift:101), and the load path's `config.numExperts > 0` guard runs before its routed cross-check, so a dense Qwen 3.5 install stays verifiable either way. The layout validator moved to `PackedExpertLayoutVerification.swift` (314 lines) as pure code motion for the 500-line rule, which is what the new arithmetic then went into.

**Evidence after.** `tests/TinyTitanRepack/Core/Verification/PackedExpertWidthTests.swift`: 22 tests in 1 suite, all green, no model and no network. 16 go straight through `PackedExpertBytes.validate` on hand-built tables and pin the arithmetic: the real 125B-A6B expert record at its own width (2,764,800 bytes -> 2,768,896 stride at 4-bit) and at 8-bit; a declared width that disagrees with the bytes; an annotation that disagrees with its own slice; the truthful withdrawn 6-bit accepted while a mislabelled one is refused; a width derived with no annotation at all; a missing slice; a duplicated slice; an expert that fills its stride exactly; a byte extent that does not divide; a tensor with no elements; an implied width outside the format's 1...32; a scale slice the wrong size; an unknown dtype; overflow refused rather than wrapped (element count, byte sum, and expert-index multiply); and the same complaint printed from ten shuffled dictionaries. 6 go through the real `VerifiedInstallTool.validatePackedExpertLayout` on a temporary install, three of which repack the synthetic MoE snapshot: a fresh 8-bit install verifies, a manifest that misdeclares the routed width is refused, and a layout that misannotates an expert is refused before the hashes. The two dense-row cases are the sibling guards: `aDenseLayoutHasNoWidthToDeclare` pins that the empty-layer shortcut runs before the `quant` block is asked for, and `everyExpertInTheLayerIsCheckedNotOnlyTheFirst` pins that a lie in expert 1 is caught when the load path only ever looks at expert 0. Both controls re-measured at closure, not inherited from the fix run: commenting the `PackedExpertBytes.validate` call out makes three end-to-end tests fail, two of them with `expected a refusal, and the check passed`; moving the layout check to after the file-hash loop makes exactly one fail -- `aLayoutThatMisannotatesAnExpertIsRefusedBeforeTheHashes`, whose `annotated 8-bit` expectation goes false because the digest complaint now arrives first. Full suite `swift test --no-parallel` exit 0: 1562 tests in seven bundles, 0 failures (the closure commit first wrote `1518 in six`, which is what the pipe hid rather than what the run printed -- corrected here against the saved log, and the seven bundle summaries are 731 + 382 + 126 + 162 + 28 + 44 + 89). `swift build -c release` clean. All eleven `tools/lint.sh` gates ok, the python one run with the pinned ruff 0.16.7 ahead of Homebrew's 0.16.10 on `PATH` -- see environment.md.

**Commit.** `53b64d5`

### AUD-103 — Five of the nine Python pitfalls the audit standard names have no rule behind them, and the config comment claims they do

- **Severity / tier:** S2 / Tier B
- **Project:** python-tooling
- **Location:** `pyproject.toml:20-24`
- **Category:** check coverage gap
- **Status:** START
- **Host:** Mac (primary)
- **Discovered by:** tool-coverage proof (docs/audit-2026-10-06/tool-coverage.md)

**Evidence before.** Proven: E722, B006, S101, F632 each fire on a probe. Silent on the same probe: subprocess.run(check=False), time.sleep() used to synchronise, open() without encoding=, datetime.datetime.now() without a timezone, and a test body that asserts nothing. pyproject.toml says 'the rest are the families the audit standard names', which is a claim broader than select = [E4,E7,E9,F,W,B,E722,S101,PT].

### AUD-104 — The converter expert-order gate reports nothing when the converter dependencies are missing

- **Severity / tier:** S2 / Tier A
- **Project:** lint-gates
- **Location:** `tools/lint.sh:319-330`
- **Category:** gate fails open
- **Status:** START
- **Host:** Mac (primary)
- **Discovered by:** gate reading + probe

**Evidence before.** The probe imports numpy and prepare_agentworld; on ImportError it prints 'SKIP: ... (converter deps unavailable)' and exits 0, so lint.sh reports the check line and keeps status 0. This is the defect class the gate exists for: experts filed by arrival order are invisible to every downstream check (the docstring in tools/prepare_agentworld.py:325-334 says exactly that). Verified the gate does catch the real sabotage when numpy is present: filing by len(target['experts']) produced 'FAIL: experts landed by arrival order: [3, 0, 7, 1, 5, 2, 6, 4]'. Expected-correct: missing converter deps FAIL the gate with the install command, unless an explicit documented opt-out is passed.

### AUD-105 — Nothing in the release runbook requires CI to be green on the tag, and v5.18 was published while its commit's CI was failing

- **Severity / tier:** S2 / Tier B
- **Project:** release
- **Location:** `tools/release.sh, docs/release-process.md:3`
- **Category:** missing gate
- **Status:** START
- **Host:** Mac (primary) + GitHub
- **Discovered by:** gh release v5.18 published 2026-10-05T06:17:34Z vs CI on ea5de8c (the tag) failing at 05:59:23Z

**Evidence before.** docs/release-process.md opens 'This is the runbook for turning a green main into a tagged, published release', but the green-main precondition is a sentence, not a check: tools/release.sh gates the gates it runs locally and nothing queries the Actions run for HEAD. v5.18 is a live instance: published an hour after its own tag commit failed the Installer gates job.

### AUD-141 — The load path cross-checks only one expert per layer, so a width or shape lie confined to any later expert loads and answers wrongly

- **Severity / tier:** S2 / Tier A
- **Project:** runtime
- **Location:** `sources/TinyTitan/Runtime/Inference/Model+SchemaValidation.swift:270-272`
- **Category:** incomplete validation on the load path (found by the AUD-124 fix, not fixed by it)
- **Status:** START
- **Host:** Mac (primary)
- **Discovered by:** AUD-124 fix, sibling sweep

**Evidence before.** `validateRoutedExpertLayout` takes `layer.experts.first` as the reference for the whole layer (Model+SchemaValidation.swift:271) and compares the nine expected role records (gate/up/down and their `_scales` and `_biases`) against that one expert only. A 125B-A6B layer carries 512 experts, so 511 of them are unchecked at load: a layout can shift an offset, change a shape or annotate a different width in expert 1 and the model still opens. The runtime then dequantizes those bytes with the word count taken from the declaration, which is exactly the failure the writer's own comment describes (sources/TinyTitanRepack/Core/Format/SSDAIJSON.swift:140-144). AUD-124 closed the same hole on the verifier side -- `PackedExpertBytes.validate` loops every expert, pinned by PackedExpertWidthTests.everyExpertInTheLayerIsCheckedNotOnlyTheFirst -- and this row records that the load path is still the one that runs when a receipt was already issued. Not fixed with AUD-124 because it is load-path work the user has not asked for, it needs a golden-baseline model run to close, and checking 512 experts x 9 records x every layer at load is a startup-cost change that needs a measurement and a decision, not an assertion dropped into a loop.

**Evidence after.** Expected: opening a packed MoE install validates the role records of every expert in every layer, or the load path says in one line which experts it did not check and why. Either outcome needs the load cost measured before and after.

### AUD-106 — The handover brief describes release 5.15 as current while 5.16, 5.17 and 5.18 are published

- **Severity / tier:** S2 / Tier C
- **Project:** docs
- **Location:** `docs/handover-tinytitan.md:1-37`
- **Category:** documentation drift
- **Status:** OPEN
- **Host:** Mac (primary)
- **Discovered by:** recon of main vs tags/releases

**Evidence before.** AGENTS.md: 'Work in flight is handed over in docs/handover-<name>.md; docs/handover-tinytitan.md is the current one and starts with the prompt for the next session. Read it before installing, converting or moving anything.' The file's title and pasted prompt are 5.15, its 'Where the work stands' table says main is level with the v5.15 tag and models/ holds 8 installs at 244 GB, while v5.18 is published (2026-10-05) and models/ holds 2 installs (163 GB). A next session that follows it acts on the wrong release and the wrong install set.

### AUD-115 — bitWidthOverridesHonored is written and decoded but never reaches the runtime Manifest: a documented contract field no consumer can read

- **Severity / tier:** S2 / Tier A
- **Project:** contract
- **Location:** `sources/TinyTitanRepack/Core/Format/SSDAIJSON.swift:121 → sources/TinyTitanFormat/SSDAIManifestV1.swift:321`
- **Category:** dead contract field
- **Status:** OPEN
- **Host:** Mac (primary)
- **Discovered by:** L1 architecture pass

**Evidence before.** Written at SSDAIJSON.swift:121 from plan.bitsOverrideCount; decoded into the wire struct at SSDAIManifestV1.swift:321/:332/:346; asserted in tests/TinyTitan/Infrastructure/ModelIO/ManifestReaderTests.swift:236 and :291. ManifestReader.swift:448-465 maps wire→Manifest (including quant.overrides→quantOverrides at :459) and does not carry it; grep over sources/ finds no runtime reader. The repack side keeps its own copy in RepackAudit.swift:12/:86, so the value exists twice and is consumed zero times.

**Evidence after.** None yet.

### AUD-116 — A per-tensor quant override whose stem equals a slot name is silently skipped — the failure class the hand-written decoder exists to fix

- **Severity / tier:** S2 / Tier A
- **Project:** repack
- **Location:** `sources/TinyTitanRepack/Core/Format/SSDAIJSON.swift:149`
- **Category:** silent drop on a numerics-affecting field
- **Status:** OPEN
- **Host:** Mac (primary)
- **Discovered by:** L1 architecture pass

**Evidence before.** `guard dict[stem] == nil else { continue }` at :149 inside quantObject; the doc comment at :147-148 states the intent ('a stem that collides with a slot name would overwrite a slot, so it is skipped'). Skipping protects the slot but drops the override, and the reader's decoder comment (SSDAIManifestV1.swift:222-226) records what a dropped override did historically: a 4-bit install's 8-bit K/V read back as 4-bit — 'the model answers fluently and wrongly' (SSDAIJSON.swift:140-144). Rated S2 not S1 because the five slot keys are camelCase while tensor stems are dotted Hugging Face names, so a real collision is implausible; the guard should refuse rather than drop.

**Evidence after.** None yet.

### AUD-117 — hc/indexer/ple geometry is emitted for one family only, and the reader treats every absent optional field as fine, so a family that needs them loads unvalidated

- **Severity / tier:** S2 / Tier A
- **Project:** contract
- **Location:** `sources/TinyTitanRepack/Core/Format/SSDAIJSON.swift:79-96 → sources/TinyTitan/Infrastructure/ModelIO/ManifestReader.swift:322-327`
- **Category:** unvalidated external input / contract drift
- **Status:** OPEN
- **Host:** Mac (primary)
- **Discovered by:** L1 architecture pass

**Evidence before.** SSDAIJSON.swift:79 gates the extension fields on `.qwen38flash`. ManifestReader.swift:322-327 `checkOptional` skips whatever is absent, so a second family that requires the same geometry would produce no manifest fields and no error. PLEConstants.validate() (:56-89) is what catches the ple half at load — but only if something calls it, and nothing in the manifest layer requires it.

**Evidence after.** None yet.

### AUD-119 — Documented converter command names two suites where CI runs three, and the requirements comment states a test count the baseline does not reproduce

- **Severity / tier:** S2 / Tier C
- **Project:** docs
- **Location:** `AGENTS.md (Test rules, 'The converter's gate is two python suites') and benchmark/requirements.txt:4-5`
- **Category:** documentation drift / stale measurement quoted as fact
- **Status:** OPEN
- **Host:** Mac (primary)
- **Discovered by:** baseline + L0 repository pass

**Evidence before.** AGENTS.md gives `cd benchmark && python3 -m unittest test_prepare_qwen38 test_qwen38_resume_e2e`; .github/workflows/ci.yml:129-130 runs those plus test_prepare_agentworld, whose own comment (ci.yml:118-120) says it covers the Qwen3.5-MoE fusion at both widths. requirements.txt:4-5 asserts the baseline ran '299 tests, 52 skipped'; measured here with the pins installed: Ran 87 tests, OK, 0 skipped. Nothing in the build compares the docs to the run, so both claims are unfalsifiable as written.

**Evidence after.** None yet.

### AUD-120 — No dependency vulnerability feed for Swift: swift-nio 2.100.0 and swift-transformers are pinned but never checked against a CVE source

- **Severity / tier:** S2 / Tier B
- **Project:** ci
- **Location:** `.github/dependabot.yml (absent) and Package.resolved`
- **Category:** missing CVE coverage on one of three languages
- **Status:** OPEN
- **Host:** Mac (primary)
- **Discovered by:** §3 baseline

**Evidence before.** pip-audit 2.10.1 covers benchmark/requirements.txt and `npm audit` covers both plugin packages, both clean at baseline (see baseline.md). For Swift there is no scanner in the toolchain, no .github/dependabot.yml in the repo, and ci.yml installs nothing that would consult an advisory feed. Package.resolved carries exact pins, which is reproducibility, not vulnerability coverage.

**Evidence after.** None yet.

### AUD-122 — Seven production files have zero covered lines with no model gate explaining it

- **Severity / tier:** S2 / Tier B
- **Project:** tests
- **Location:** `sources/TinyTitan/Infrastructure/ModelIO/ArchConfig+Manifest.swift and 6 more`
- **Category:** coverage gap on non-gated code
- **Status:** OPEN
- **Host:** Mac (primary)
- **Discovered by:** §3 coverage run

**Evidence before.** From the baseline llvm-cov aggregation over sources/ (42182/54206 lines = 77.8%): ten files have 0% covered lines. Three are the ServerModelSession Generation/Loading/PromptCache trio, whose 0% is the documented model-free unit run. The other seven have no such excuse: ArchConfig+Manifest.swift, ProcessMemory.swift, Kernels/MoE/SharedExpertAffineQuant.swift, Kernels/Primitives/PLEBlock.swift, Runtime/Family/Qwen35DenseFamily.swift, Runtime/Inference/RealForwardRunner+ANE.swift, TinyTitanLib/Watchdogs/WatchdogSupervisor.swift.

**Evidence after.** None yet.

### AUD-123 — The LAN manager admits link-local peers by default and its default group key is a published literal

- **Severity / tier:** S2 / Tier A
- **Project:** fleet
- **Location:** `plugins/dsh-lan-manager/src/net.js:38 and src/config.js:38`
- **Category:** permissive default on a network-facing surface
- **Status:** OPEN
- **Host:** Mac (primary)
- **Discovered by:** L4 security pass

**Evidence before.** DEFAULT_IPV4_NETWORKS (net.js:33-40) includes ['169.254.0.0', 16] beside loopback/RFC1918/CGNAT, and DEFAULT_GROUP_KEY is the string 'tinytitan-lan' (config.js:38), which config.js:132 always resolves to when nothing is configured — so the token guard is nominal against anyone reading the source. Two facts bound the severity: the harness webserver binds loopback only (index.js:232-240 documents and verifies it), and checkAddress compares the Origin hostname as a string against an IPv4 pattern (net.js:27) and never resolves DNS, so rebinding cannot smuggle a name through. S2 is the honest rating on those facts.

**Evidence after.** None yet.

### AUD-126 — Six benchmark suites skip when models/ has no install, so their gate is a no-op on the host that runs it most

- **Severity / tier:** S2 / Tier B
- **Project:** tests
- **Location:** `benchmark/test_launcher_ram.py:135, :194 and 5 more`
- **Category:** coverage gap on a CI gate
- **Status:** OPEN
- **Host:** Mac (primary)
- **Discovered by:** sibling scan while closing AUD-102

**Evidence before.** Read at the cited lines and confirmed by the auditor. test_launcher_ram.py:135 and :194, test_launcher_port.py:108 and :155, test_coder_clients.py:117, test_tinytitan_profile.py:37 each skipTest on 'no install under models/ and no built server to list one' (or a local variant). CI has no models/ by policy and none may be fetched to satisfy a gate, so these never run there — the same shape AUD-102 was, found by looking at the siblings rather than assuming they were clean. EmptyModelsDirTests demonstrates the fix: a synthetic directory plus TINYTITAN_MODELS_DIR exercises the launcher's choices with no model.

**Evidence after.** None yet.

### AUD-127 — A guard-else-return inside a test body passes green when its env var is unset, and is not recorded as a skip

- **Severity / tier:** S2 / Tier C
- **Project:** tests
- **Location:** `tests/TinyTitanRepack/Core/Format/Qwen4ExpArchInfoTests.swift:141-143`
- **Category:** test that asserts nothing on the path it did not take
- **Status:** OPEN
- **Host:** Mac (primary)
- **Discovered by:** §5 facade sweep

**Evidence before.** Reported by the discovery sweep at these lines; NOT yet re-read by the auditor — verification precedes any fix. Reported as `guard let path, fileExists(...) else { return }` in the test body. Swift Testing's `try #require` or `.enabled(if:)` would record the skip instead of hiding it.

**Evidence after.** None yet.

### AUD-130 — context and maxTokens are plumbed into the route writer but resolveConfig never emits them, so every route write uses ROUTE_DEFAULTS

- **Severity / tier:** S2 / Tier B
- **Project:** plugins
- **Location:** `plugins/dsh-tinytitan/src/route.js:45-46, :99-100 and src/config.js:294-351`
- **Category:** surface wired to nothing (§5)
- **Status:** OPEN
- **Host:** Mac (primary)
- **Discovered by:** §5 facade sweep

**Evidence before.** Reported by the discovery sweep at these lines; NOT yet re-read by the auditor — verification precedes any fix. Reported at route.js:45-46/:99-100 and generate.js:193-194, with the shell branch (route.js:66-72) said to forward neither --context nor --max-tokens.

**Evidence after.** None yet.

### AUD-132 — The origin guard only rejects an Origin that is present: a mutating request with no Origin header passes it outright

- **Severity / tier:** S2 / Tier A
- **Project:** fleet
- **Location:** `plugins/dsh-lan-manager/src/router.js:186-192`
- **Category:** CSRF guard with an absent-header hole
- **Status:** OPEN
- **Host:** Mac (primary)
- **Discovered by:** §5 facade sweep

**Evidence before.** Reported by the discovery sweep at these lines; NOT yet re-read by the auditor — verification precedes any fix. Reported as `if (origin && !isAllowedOrigin(...))`. Two mitigating facts to confirm before scoping the fix: the harness binds loopback only, and the JSON routes require content-type: application/json, which a cross-origin simple form POST cannot set.

**Evidence after.** None yet.

### AUD-133 — A failed tailscale or Bonjour probe is swallowed by a per-source catch, so /peers is quietly short rather than reporting a degraded probe

- **Severity / tier:** S2 / Tier A
- **Project:** fleet
- **Location:** `plugins/dsh-lan-manager/src/discovery.js:238-248`
- **Category:** silent failure on a network path
- **Status:** OPEN
- **Host:** Mac (primary)
- **Discovered by:** §5 facade sweep

**Evidence before.** Reported by the discovery sweep at these lines; NOT yet re-read by the auditor — verification precedes any fix. Reported as `catch {}` around each discovery source.

**Evidence after.** None yet.

### AUD-134 — Journal and store reads fall back to `?? []` / `.empty` on a thrown error, and that fallback is not covered by journalFailed, so a broken memory answers 'there is nothing'

- **Severity / tier:** S2 / Tier A
- **Project:** memory
- **Location:** `sources/TinyTitanMemory/MemoryService+Sessions.swift:39, :42, :202; MemoryService+Consolidation.swift:29-31, :72-73; sources/TinyTitanMemory/ContinuityJournalStore.swift:52`
- **Category:** silent failure, error swallowed into an empty answer
- **Status:** OPEN
- **Host:** Mac (primary)
- **Discovered by:** §5 facade sweep, verified and narrowed by the auditor on this branch

**Evidence before.** Every site re-read on this branch; the sweep's list was wrong in both directions. `journalFailed(in:)` (MemoryService+Maintenance.swift:128-136) reports a *write* refusal once per scope through `log(.degraded(operation: "journal"))`, and MemoryService+Sessions.swift:108 reads it -- so open and write failures ARE surfaced. None of these *reads* set that flag, and each returns a value indistinguishable from an empty workspace: (1) :39/:42 `(try? await localStore.sessionInit(...)) ?? .empty` -- on the :42 branch (no workspace at all) `isDegraded` stays false, so a throwing local store yields an empty bootstrap and a prompt that says the model has no memories; (2) :202 recordedFacts `(try? await store.search(...)) ?? []` -- the tool-facing fact list and, at MemoryBackend+Consolidation.swift:111, the consolidation prompt's `existing:` facts, so consolidation is told there is nothing on file and re-adds what is already there; (3) Consolidation.swift:29-31 and :72-73 `(try? await store.search(limit: 400)) ?? []` -- the duplicate/conflict candidate pool, so a failed read silently disables the dedup that T2/T4/T5 exist to do; (4) ContinuityJournalStore.turns():52 `guard let taskID = try? await store.taskID(...)` else `[]` -> MemoryBackend+Consolidation.swift:97 logs 'consolidation skipped ... no new turns' when the truth is 'the journal could not be read'. The last is the actively misleading one: the log names the wrong cause.

**Evidence after.** None yet -- first: a test that makes the read throw and asserts the caller says 'unavailable' rather than 'empty'. Scope corrections, recorded here per §0 rather than by editing the finding away: MemoryService+Sessions.swift:78 and MemoryService+Consolidation.swift:75 are not `try?` sites at all (sweep false positives, :78 is `guard configuration.isEnabled else { return [] }`, :75 a plain `pool = sharedCandidates ?? []` over an already-read value); MemoryRetrieval.swift:256 is `try? await Task.sleep`; MemoryRetrieval.swift:52/:233 and ContinuityJournalStore.sessions():71/search():92 are real `try?`-to-empty but benign -- the first two cost recall on a background path, the last two have no production caller (the only consumer of journalStore(for:) is consolidation, which calls turns()), and MemoryRetrieval.swift:289 is documented intent ('a failure is never a hint', :285-287). Those four moved to AUD-138 so this row can be closed on what is actually model-visible.

### AUD-135 — expireSessionLog returns true after a try?-wrapped compactJournal, so a failed compaction reads as an expired log

- **Severity / tier:** S2 / Tier B
- **Project:** memory
- **Location:** `sources/TinyTitanMemory/MemoryService+Maintenance.swift:77-79`
- **Category:** silent failure
- **Status:** OPEN
- **Host:** Mac (primary)
- **Discovered by:** §5 facade sweep

**Evidence before.** Read at the cited lines and confirmed by the auditor. `try? await engine.compactJournal()` then `await engine.shutDown()` then `return true` (:77-79). The open-failure path is handled honestly (`guard let journal = try? FileJournal(url: url) else { return false }` at :70), so the asymmetry is only the compaction. journalFailed(in:) (:128) and reportedJournalFailures (MemoryService.swift:63) report open failures once per scope; nothing reports this one. Consequence is disk growth, not lost data, hence S2.

**Evidence after.** None yet.

### AUD-136 — reasoning_budget_tokens and parallel_tool_calls are accepted from the wire and not enforced

- **Severity / tier:** S2 / Tier B
- **Project:** server
- **Location:** `sources/TinyTitanLib/OpenAIRequestValidator.swift:88-92, :140-146`
- **Category:** surface wired to nothing, publicly disclosed (§5)
- **Status:** OPEN
- **Host:** Mac (primary)
- **Discovered by:** §5 facade sweep

**Evidence before.** Reported by the discovery sweep at these lines; NOT yet re-read by the auditor — verification precedes any fix. Reported as accepted-and-ignored with disclosure at plugins/dsh-tinytitan/README.md (~:227), so it is not a silent facade. Either enforce or reject with a clear error: a field that is silently ignored changes what the client believes it asked for.

**Evidence after.** None yet.

### AUD-140 — The sanctioned browser client downloads a Node tarball and runs what it extracts without checking any digest

- **Severity / tier:** S2 / Tier A
- **Project:** installer
- **Location:** `tools/dsh_local.sh:256-275`
- **Category:** integrity / download-and-execute, fail-open check
- **Status:** OPEN
- **Host:** Mac (primary)
- **Discovered by:** AUD-109 sibling sweep

**Evidence before.** dsh_local.sh:264 `run "download Node from nodejs.org" curl -fsSL "$url" -o "$tmp/$tarball"`, then unpacks it and uses the resulting node/npm to install packages that are then executed by the chat window. No SHASUMS256.txt is fetched and nothing hashes the tarball; nodejs.org publishes the digest beside every download, so unlike AUD-109 the verified path exists and is simply not taken. The npm dependencies are a separate surface -- npm verifies its own tarballs against the lockfile's integrity field, so the gap is the runtime that runs npm, not the packages.

**Evidence after.** Expected: the Node download is checked against nodejs.org's published SHASUMS256.txt for that exact version, and a missing or non-matching digest stops the setup -- the same fail-closed shape AUD-109 gave the engine and tools archives, with a harness that stubs curl and proves the refusal.

### AUD-142 — Seven metadata reads still have no size bound, and they sit at four different trust boundaries, so they do not all want the same cap

- **Severity / tier:** S2 / Tier A
- **Project:** engine
- **Location:** `sources/TinyTitan/CPUEngine/AffineSnapshot.swift:131, sources/TinyTitan/Tokenization/Detokenizer.swift:51, sources/TinyTitanLib/ServerModelSession+Loading.swift:293, sources/TinyTitanRepack/Core/Format/ArchInfo.swift:168, sources/TinyTitanRepack/Core/Format/SSDAILayoutValidator.swift:5, sources/TinyTitanBench/CPUCommands.swift:229, sources/TinyTitanFleet/Command/main.swift:248-250`
- **Category:** unbounded memory on an input file (found by the AUD-113 fix, not fixed by it)
- **Status:** OPEN
- **Host:** Mac (primary)
- **Discovered by:** AUD-113 fix, sibling sweep

**Evidence before.** Swept from `grep -rn 'Data(contentsOf:' sources/`: fourteen code lines remain in the tree after AUD-113. Six are already bounded — `VerifiedInstallReceipt.swift:95` is the K17 read-then-check AUD-113 cites as its model, `ModelCatalog.swift:287` and `ANEPrefillAttention.swift:182` are AUD-113's own, and three (`ServerPromptStateStore.swift:121`, `RemoteSnapshotLoader.swift:88` and `:113`) are bounded by a check that runs before or beside the read — and one (`AffineSnapshot.swift:263`) is a mapped weight file rather than a metadata document. That leaves seven uncapped, and they split by who supplied the bytes, which is why this is a row and not a search-and-replace of `maxBytes`:
    (1) Model directory, engine-side — same attacker model as AUD-113 (§2.3: an install may have been copied off another machine). `AffineSnapshot.swift:131` reads `config.json` uncapped on every CPU-engine path (CPUModelBackend.swift:97, SideEngine.swift:364, and the bench at CPUCommands.swift:207); `Detokenizer.swift:51` (`GFByteLevelDecoderConfiguration.load`, reached from `Tokenizer+Helpers.swift:45` during `GFTokenizer.load`) reads the whole `tokenizer.json`, which measures 12,809,320 bytes in the installed 125B tokenizer on this host — a real document, not a rounding error; `ServerModelSession+Loading.swift:293` reads `chat_template.jinja` (8,952 bytes installed) into `SHA256.hash` for the runtime identity with no bound. For these three the measured files sit under the manifest's 64 MiB, so `ManifestReader.defaultMaxBytes` is defensible.
    (2) Operator-supplied HF checkpoint, converter-side — `ArchInfo.swift:168` reads the snapshot's `config.json`. The operator chose that directory, and a cap here refuses a legitimate checkpoint rather than an attacker's, so the right bound is a stated limit, not a borrowed one.
    (3) A file the tool was handed on the command line — `CPUCommands.swift:229` (`vocab.json`) and `TinyTitanFleet/Command/main.swift:250`, where the same function's `path == "-"` branch reads stdin to end-of-file at :248, so a bound on the file and none on the pipe protects neither. Both are dev/CLI surfaces over the operator's own input.
    (4) Own output, re-read as a round-trip check — `SSDAILayoutValidator.swift:5` is called only at RemoteStreamingRepacker+Local.swift:211 and +Remote.swift:308, each immediately after `writeSmall` wrote that very file into the partial directory. Uncapped, but there is no adversary on that path; it is a self-check, and belongs to the §6 unused/low-tier classification, not to AUD-113's.
    Bounded already, recorded so the next sweep does not re-count them: `RemoteSnapshotLoader.swift:88` reads an 8-byte range we requested and then asserts `prefixData.count == 8` (:89); `RemoteSnapshotLoader.swift:113` reads a header whose size was checked against `Safetensors.maxHeaderBytes` *before* the range was requested (:100-105), so the network cannot make that allocation grow; `ServerPromptStateStore.swift:121` checks `metadataSize <= maximumMetadataBytes` via `resourceValues` first — bounded, but by a stat followed by a re-read, which is precisely the K17 window, so it is the one of the three where AUD-113's convention should be applied.
    Excluded from this row: `AffineSnapshot.swift:263` — `Data(contentsOf:options:.alwaysMapped)` over `model_weights.bin` maps lazily and is held for the snapshot's life by design (ResidentWeights), so bounding it is a different decision about resident weight size, not a metadata read.
    Not fixed with AUD-113 because none of these seven is the file that row named, three of the four groups need a cap chosen for their boundary rather than the manifest's, and the three engine-side sites are load-path changes that each need a golden-baseline run to close.

**Evidence after.** Expected: group (1) reads under a stated bound and says so on refusal, with a test per site in the shape AUD-113's `anOversizedSidecarIsRefusedByTheBoundNotByTheDecoder` uses (the bound fires, not the decoder); `ServerPromptStateStore.swift:121` moves to read-then-check so its bound stops being a TOCTOU; groups (2) and (3) get a limit written down where the operator can see it, or an explicit note that the operator's own checkpoint is out of scope; group (4) classified as a self-check rather than left to look like an unfixed hole. Load-path sites need `tools/golden-baseline.sh --check qwen38-4` before and after.

### AUD-144 — Four journal openers create files without O_NOFOLLOW, the pattern AUD-110 just closed in the installer

- **Severity / tier:** S2 / Tier A
- **Project:** memory
- **Location:** `sources/ContinuityCore/Persistence/Journal.swift:116, :137, :382, :416`
- **Category:** symlink following on a predicted path
- **Status:** OPEN
- **Host:** Mac (primary)
- **Discovered by:** AUD-110/AUD-111 fix, sibling sweep

**Evidence before.** `grep -rn O_CREAT sources/` after AUD-110: every creating opener in `TinyTitan` and `TinyTitanRepack` now refuses links, and the four left are here -- :116 `open(lockURL.path, O_RDWR | O_CREAT | O_CLOEXEC, 0o600)` (the `flock` anchor), :137 the append handle, and :382 and :416 both `O_WRONLY | O_CREAT | O_TRUNC`, the compaction temp and `truncate()`. The mode and O_CLOEXEC are already right; only the link guard is missing, and `grep -rn 'isSymbolicLink|symlink' sources/ContinuityCore/` returns nothing, so there is no equivalent of the repack's `InstallLockTests.symlinkedLockIsRejectedWithoutFollowingIt` anywhere in this target either. The two O_TRUNC sites are the sharp pair: a link planted at a predicted journal or checkpoint name turns a memory-store write into a destructive write elsewhere, reachable by anything that can create a file in the store directory. Weaker boundary than AUD-110 -- the store lives under the operator's own `~/.tinytitan`, so it needs local write access to that directory, which is why this is a sibling row and not the same severity -- but the same primitive, and the installer's precedent says the fix costs nothing at the open.
    Not fixed with AUD-110 because it is a different target (Tier A per `docs/audit-2026-10-06/inventory.md:120`, `TinyTitanMemory` + `ContinuityCore`), it needs the memory suite (`tests/ContinuityCore/JournalTests.swift`, `JournalFailureTests.swift`) rather than the repack's, and `:116` deserves thought the sweep did not have time for: refusing a link at the lock path changes who can contend for the lock, which is the one behaviour here that is load-bearing.

**Evidence after.** Expected: the four opens carry O_NOFOLLOW, or `:116` is argued out with the reason recorded; a test per site in the shape `PosixOpenTests.aSymlinkAtTheOutputPathIsRefusedAndItsTargetSurvives` asserting the victim's bytes survive, not merely that an error was thrown; and the memory suites green in the full serial run.

### AUD-110 — openCreateRW is the only opener without O_NOFOLLOW, and it is used for weight outputs

- **Severity / tier:** S2 / Tier A
- **Project:** repack
- **Location:** `sources/TinyTitanRepack/Core/System/Posix.swift:32 (before); :29-44 (after)`
- **Category:** symlink following / TOCTOU on a predicted path
- **Status:** DONE
- **Host:** Mac (primary)
- **Discovered by:** L4 security pass

**Evidence before.** Posix.swift:33 `open(path, O_RDWR | O_CREAT | O_TRUNC, 0o600)` — no O_NOFOLLOW, no O_EXCL, no O_CLOEXEC, while :38 openExistingRW, :44 openDirectory, :50 openLock and :212 atomic-temp all set O_NOFOLLOW. Callers: ResidentWriter.swift:16, RemoteStreamingRepacker.swift:172, :183 — the .partial weight outputs. A symlink planted at a predicted output path is followed and truncated.

**Fix.** O_NOFOLLOW | O_CLOEXEC added, so `openCreateRW` now carries the same flags as `openLock` (Posix.swift:59) and the atomic-temp opener (:221) in the very same file. No O_EXCL: the resume path reopens a payload that already exists, and a test pins that reopening still truncates. The comment records why the guard exists rather than just the mode, including the part that decides whether this is safe for operators: O_NOFOLLOW applies to the last path component only, so symlinking a whole model directory -- the ordinary way to keep weights on a second disk -- keeps working, and only a link standing where a payload file is about to be written is refused.

**Evidence after.** Five new tests in `tests/TinyTitanRepack/Core/System/PosixOpenTests.swift`, all green. The control is what makes them evidence: with `\| O_NOFOLLOW` deleted from the flags and everything else unchanged, the suite fails 2 issues -- `#expect(throws:)` sees no error at all AND the victim's canary bytes are gone, i.e. the open was followed and O_TRUNC destroyed the file behind it -- while the other four stay green, so the tests pin the guard and not something incidental. With the fix in place all five pass, including the two that prove the guard did not become a wall: the directory-symlink case still installs, and an existing regular file is still reopened and truncated to 0 for the resume path. The mode is pinned at 0600 and FD_CLOEXEC is pinned through `fcntl(F_GETFD)`, because the installer shells out to curl and an unconditionally-inheritable weight descriptor is the same leak in a different shape. Restored from the control afterwards; `git diff` shows only the intended change. Same verification as AUD-111 (one commit, one fix): build clean, nine new tests green, eleven lint gates ok, golden `qwen38-4` identical, and the full serial suite red ONLY in `TinyTitanServerTests`, which aborts at a routing test this change does not touch and which reproduces at clean HEAD -- see AUD-143. Commit 7896ad5.

**Commit.** `7896ad5`

### AUD-111 — Two env-named trace files open 0o644 with no O_NOFOLLOW: world-readable routing traces

- **Severity / tier:** S2 / Tier A
- **Project:** engine
- **Location:** `sources/TinyTitan/Runtime/Inference/RealForwardRunner.swift:478, :489 (before); RealForwardRunner+Diagnostics.swift:9-28 (after)`
- **Category:** permissive file mode + symlink following
- **Status:** DONE
- **Host:** Mac (primary)
- **Discovered by:** L4 security pass

**Evidence before.** Both `open(path, O_WRONLY | O_CREAT | O_TRUNC, 0o644)` for TINYTITAN_ROUTE_TRACE (:478) and TINYTITAN_PREFETCH_TRACE (:489). Contrast Posix.swift:31-33, which documents 0600 because 'model, partial and temp files are never shared with other users of the machine'. The path is operator-chosen, so the exposure is a local information leak and a symlink write, not a remote hole.

**Fix.** The two opens were one duplicated expression, so they are now one `RealForwardRunner.openTraceFile(path:)` in the diagnostics extension, where the writers that consume the descriptor already live: 0600 (the installer's own rule, quoted in the doc comment), O_NOFOLLOW, O_CLOEXEC, and the same -1-on-failure answer the writers already guard on. The helper reports a refusal on stderr rather than returning -1 in silence -- an operator who sets TINYTITAN_ROUTE_TRACE and gets no file is otherwise looking at a runner that appears not to route, and this runner routes silently by design, so silence is indistinguishable from a bug. Chosen over guarding inline because `RealForwardRunner.swift` sits at 496 of the 500-line production budget, and an untestable two-line change in a file that cannot grow is not the fix.

**Evidence after.** Four new tests in `tests/TinyTitan/Runtime/TraceFileOpenTests.swift`, all green: the fresh file is 0600 by `posixPermissions`; a link at the trace path returns -1 AND the victim's canary survives (the truncation is the defect, so the test asserts the bytes, not just the refusal); an unopenable path returns -1 rather than trapping, which is the contract `recordRouteTrace`'s `guard routeTraceFD >= 0` depends on; and a symlinked directory still opens. Control measured: with the helper's flags restored to the pre-fix `O_WRONLY | O_CREAT | O_TRUNC, 0o644`, the suite fails 3 issues -- the mode test plus both halves of the symlink case -- and the other two stay green. Verified in the same run as AUD-110: `swift build -c release` clean; eleven lint gates ok; this is a change to the runner's stored-property initialisation, so the golden gate ran: `tools/golden-baseline.sh --check qwen38-4` exit 0, `ok — output identical to baseline`, on the qwen3.8-flash-next _125B_A6B_4Bit install, with the preconditions checked immediately before (macOS 27.0.1, Swift 6.4, 79% memory free, 332 GB free, no competing model process); Ornith 1.5 8-bit not checked, nothing fetched. Sibling sweep: every `O_CREAT` open in `sources/` now carries O_NOFOLLOW except four in `ContinuityCore/Persistence/Journal.swift` (:116 lock, :137 append, :382 and :416 both O_TRUNC), which are Tier A per the inventory and got their own row -- AUD-144 -- because that store lives under the operator's own home directory and a fix there needs the memory suite, not this commit. Commit 7896ad5.

**Commit.** `7896ad5`

### AUD-112 — Six CommonCrypto SHA-256 return values are discarded on the integrity-hash path

- **Severity / tier:** S2 / Tier A
- **Project:** repack
- **Location:** `sources/TinyTitan/Infrastructure/ModelIO/Sha256Verifier.swift:34, 50, 57, 65, 68, 74 (before); :1, :25-62 (after)`
- **Category:** unchecked return value
- **Status:** DONE
- **Host:** Mac (primary)
- **Discovered by:** L3 line pass

**Evidence before.** CC_SHA256_Init is called bare at :34 and :65; CC_SHA256_Update and CC_SHA256_Final are `_ =` at :50, :57, :68, :74. The read() return three lines above (:41-47) IS checked and thrown on, so the asymmetry is in the same function. Impact is bounded: a partial digest fails against VerifiedInstallReceipt.swift:134, so it fails closed. Recorded at S2 rather than S1 for that reason, and because CommonCrypto's one-shot context calls do not fail in practice for a non-NULL ctx.

**Fix.** The hasher now wraps an API that has no error state, so there is nothing left to discard. `import CommonCrypto` became `import CryptoKit`: `hashFile(fileDescriptor:)` streams through `SHA256().update(bufferPointer:)` and `finalize()`, and `hashData` is `SHA256.hash(data:)`. Six guards would have been the smaller diff and the worse fix -- `hashData` is non-throwing, so checking its statuses either meant crashing on a path the caller cannot recover from or changing a public signature through `Model+Loading`'s two call sites, and either way the failable calls would still have been there to be re-inspected next audit. Public and package API unchanged, `hashData` still non-throwing, the loop's `read()`/`EINTR`/`posixFailed` handling untouched. The `raw.baseAddress` guard and `got > 0` test in the update path went with the calls they were guarding, since `chunkBytes > 0` is checked at the top and `got` is therefore in 1...chunkBytes when it slices. This also makes the two halves of the integrity path one implementation: the install side already hashed with CryptoKit (`Sha256Stream`, and `SSDAIDirectoryAccess.hash` routes through it), so the verifier and the runtime no longer run different SHA-256 code against the same manifest.

**Evidence after.** Digests pinned to FIPS 180-4 rather than to the implementation, in `tests/TinyTitan/Infrastructure/ModelIO/Sha256VerifierTests.swift` (4 tests before, 9 after, all green): SHA-256("") and SHA-256("abc") worked examples; one million `a`s streamed at 64 KiB chunks against `cdc76e5c...`, which a digest covering only the first chunk cannot pass; `hashFile` at five chunk sizes (1, 4096, 64 KiB, 1 MiB, 8 MiB) against `hashData` on the same 3 MiB, because the two paths feed one manifest and may not drift; and an unopenable path throwing rather than returning the digest of nothing. The empty-buffer case is the one the discarded statuses reached most easily -- `Update` never runs, so only `Init` and `Final` decide it -- and it is pinned directly. Cost measured, because this runs inside a load: nine interleaved rounds over 64 MiB gave 1861 MiB/s CommonCrypto against 1847 MiB/s CryptoKit at the median, 0.8% and inside both sides' min/max spread of 0.034-0.037 s, with the digests equal on the same buffer. Sibling sweep: `CC_SHA`/`CommonCrypto` now appear in no Swift source at all, only in two comments that explain what this replaced; the repack's `Sha256Stream` and `RangeCopyPlanner`'s private `hashData` were already CryptoKit, so every SHA-256 in the tree is computed by a hasher that cannot silently half-finish. `swift build -c release` clean; full `swift test --no-parallel` exit 0, 1567 tests in seven bundles, 0 failures; all eleven `tools/lint.sh` gates ok with pinned ruff 0.16.7 ahead of Homebrew's 0.16.10 on `PATH`. Golden gate for a model-load-path change: `tools/golden-baseline.sh --check qwen38-4` exit 0, `== qwen38-4 ==` / `ok -- output identical to baseline`, on the qwen3.8-flash-next _125B_A6B_4Bit install (greedy, seed 1234, max-new 96, Mac15,3/24 GB/macOS 27.0). Preconditions checked before the run: Swift 6.4, 46% memory free, 336 GB free, no competing model process. Not checked: Ornith 1.5 8-bit, the default target, has no install on this host, and no model was fetched to satisfy the gate.

**Commit.** `0f7e82f`

### AUD-113 — ple_constants.json is read with an uncapped Data(contentsOf:) although its sibling receipts cap the same file class

- **Severity / tier:** S2 / Tier A
- **Project:** engine
- **Location:** `sources/TinyTitan/Runtime/Family/PLEConstants.swift:33 (before); PLEConstants.swift:35-46, sources/TinyTitan/Runtime/Prefill/ANEPrefillAttention.swift:172-189, sources/TinyTitanServer/Core/ModelCatalog.swift:278-293 (after)`
- **Category:** unbounded memory on a model-supplied file
- **Status:** DONE
- **Host:** Mac (primary)
- **Discovered by:** L4 security pass

**Evidence before.** `let data = try Data(contentsOf: url)` with no size bound. Compare VerifiedInstallReceipt.swift:81/:95-98, which carries defaultMaxBytes = ManifestReader.defaultMaxBytes (64 MiB) and checks it after the read with the K17 comment at :92-94 explaining why it is not a stat-then-read TOCTOU; and ModelCatalog.swift:281, :345 and ANEPrefillAttention.swift:270, which read the same way uncapped. The model directory is operator-supplied but can be copied from another machine, which is the attacker model §2.3 names.

**Fix.** All four sites named in the row now carry the bound the manifest already carries, `ManifestReader.defaultMaxBytes` (64 MiB, ManifestReader.swift:20), threaded as a defaulted `maxBytes:` parameter exactly as `ManifestReader.load`, `VerifiedInstallReceiptReader.load`, `PackedExpertsLayoutReader.load` and `ResidentIndex.load` each do. Two mechanisms, because the two kinds of reader differ: `PLEConstants.load` goes through `SSDAIModelDirectory.readMetadata`, so it is root-anchored, O_NOFOLLOW at every level and capped *before* the allocation; the ANE prefill sidecar and the catalog's `config.json` are read by URL, so they use the K17 convention the receipt already documents — read, then check the size of the bytes actually read, because stat-then-re-read is a TOCTOU window. The sidecar's read moved out of `init` into `ANEPrefillAttention.loadSidecarMetadata(at:maxBytes:)`, which is what makes it testable at all; the catalog's two reads (`probeSnapshot` and `snapshotFamily`) moved into one shared `ModelCatalog.readSnapshotConfig(_:maxBytes:)`, because two copies of a bound is two that can drift. Error types are unchanged and stay in their own domain: `ModelError`, `PrefillError.chunkedUnsupported` (which now says the byte count, the cap and to re-export the sidecar), and `CPUBackendError.unsupported` for the catalog — the last of these matters, because `probeSnapshot` catches it and turns it into a `ProbeFailure` reason, so an oversized file in a scanned directory makes the catalog report that directory as refused instead of taking the server down. Public API gains only defaulted parameters; no call site changed.

**Evidence after.** Six new tests, all green, and the pair that matters is the one that pins *which* check fires: `anOversizedSidecarIsRefusedByTheBoundNotByTheDecoder` pads a real sidecar JSON and refuses it at `maxBytes: 64`, well below anything a decoder could misparse, then loads the same file under the default bound — so the guard is proven to be the bound and not a JSON error wearing its clothes. `aScannedConfigOverTheBoundIsRefusedAndSaysSo` pins the catalog's user-visible half (the refusal reaches `ProbeFailure.reason` with the size and the cap in it) and `theFamilyProbeStillReadsAnOrdinaryConfig` pins that the shared helper did not break the family probe. `PLEHashTests` goes 20 -> 23 (`loadsFromDirectory`, `refusesOverBoundSidecar`, `refusesMissingSidecar`), `ANEPrefillAttentionTests` 13 -> 14, `ModelCatalogTests` 10 -> 12. `swift build -c release` clean; full `swift test --no-parallel` exit 0, 1573 tests in seven bundles, 0 failures (740 + 384 + 126 + 162 + 28 + 44 + 89, read from the saved log rather than a pipe); `tools/lint.sh` **was not clean**: ten of the eleven gates passed with the pinned ruff 0.16.7 ahead of Homebrew's 0.16.10 on `PATH`, but `swift-format lint --strict` flagged two files this row's own fix added -- `tests/TinyTitanServer/ModelCatalogTests.swift:343` and `tests/TinyTitan/Runtime/PLEHashTests.swift:392`, one AddLines break each. The claim as first written here ('all eleven gates ok') was wrong and is corrected forward rather than deleted: the drift was formatted away with the toolchain's own `swift-format` in commit 7896ad5, which is where the gate is genuinely eleven-for-eleven. The lesson is recorded in the runbook: a gate claim needs the failing output re-read after the last test file is written, because a formatter complaint in a test added late is exactly what an earlier clean run misses. Golden gate because this is the model-load path: `tools/golden-baseline.sh --check qwen38-4` exit 0, `ok — output identical to baseline`, on the qwen3.8-flash-next _125B_A6B_4Bit install — which is also the model that reads this sidecar, so the new bound ran for real. Ornith 1.5 8-bit, the default target, has no install on this host and is reported not checked; nothing was fetched.
    Sibling sweep, and it is recorded rather than fixed (AUD-142). Fourteen `Data(contentsOf:)` code lines remain in the tree after this fix (the fourth site this row named no longer appears at all, because `PLEConstants.load` reads through `SSDAIModelDirectory.readMetadata`), and they were measured against their trust boundaries rather than counted: seven are genuinely uncapped, six are already bounded — three of them by this very commit, one being the receipt whose K17 convention this fix copied, and two by protocol caps that hold before the allocation — and one is a mapped weight file of a different class. The commit message for this fix said "10 `Data(contentsOf:)` reads of non-model-directory documents remain unbounded": that number included three of the already-bounded sites and left the mapped file out of the list entirely, so it is wrong in both directions and AUD-142 carries the corrected seven with the reason each one is or is not the same threat as this row's.

**Commit.** `7b614a8`

### AUD-114 — A build-config comment still claims the package cannot be consumed as a dependency, which was measured false on 2026-10-02

- **Severity / tier:** S2 / Tier B
- **Project:** build-config
- **Location:** `Package.swift:81-83`
- **Category:** contract drift / stale documentation on a load-bearing rule
- **Status:** DONE
- **Host:** Mac (primary)
- **Discovered by:** L1 architecture pass

**Evidence before.** Package.swift:81-83 says `.unsafeFlags` 'is the only way to set it, which is why this package cannot be consumed as a dependency'. AGENTS.md:140-147 records the opposite as measured on 2026-10-02 in Swift 6.4, and tools/embedded-dependency-check.sh builds examples/embedded (Package.swift:27 depends on the TinyTitanLib product) precisely to keep that property true. Two authoritative files disagree, and the compiler does not notice.

**Fix.** The Package.swift comment now states what was measured: `.unsafeFlags` is the only way to set -O2 on a C target, and it does not stop the package being consumed as a dependency; the toolchain's "contains unsafe build flags" diagnostic is noise, not a refusal. The comment names tools/embedded-dependency-check.sh as the thing that keeps the property true, so a future reader has the check, not just the claim.

**Evidence after.** No behaviour to test -- the manifest, the build and the dependency edge are unchanged, and `swift build -c debug` is clean after the edit. The property the comment asserts is checked by tools/embedded-dependency-check.sh (task #23), which resolves the released tag from another package and builds examples/embedded against it; the defect here was two authoritative files disagreeing, and they now agree.

**Commit.** `this commit`

### AUD-118 — The receipt file name is declared twice, once per side of the contract

- **Severity / tier:** S2 / Tier A
- **Project:** contract
- **Location:** `sources/TinyTitanRepack/Core/Verification/VerifiedInstallReceiptWriter.swift:4 and sources/TinyTitan/Infrastructure/ModelIO/VerifiedInstallReceipt.swift:76`
- **Category:** duplication on a cross-target constant
- **Status:** DONE
- **Host:** Mac (primary)
- **Discovered by:** L1 architecture pass

**Evidence before.** Both declare `verified-install.json` independently. Equal today. The receipt is the trust anchor AGENTS.md documents (path binding, re-issue with --verify-install), so a one-sided edit is a silent contract break the tests would only catch by accident.

**Fix.** The receipt's name is declared once, in TinyTitanFormat, as `SSDAIInstallFileNames` (`manifest`, `verifiedInstallReceipt`, and the `reserved` set built from both). The three sites that each carried their own copy -- `VerifiedInstallReceiptWriter.fileName` (the repacker writes), `VerifiedInstallReceiptReader.fileName` (the engine reads) and `SSDAIManifestV1.validate`'s `reservedFiles` (a checkpoint may not smuggle either name in as a payload file) -- now reference it, so the two sides of the contract cannot be edited apart. TinyTitanFormat is the only target all three already depend on, which is why the constant lives there and not in the engine.

**Evidence after.** New test `aReservedArtifactNameIsRefusedAsPayload` repacks the synthetic Qwen snapshot and, for each name in `SSDAIInstallFileNames.reserved`, adds it to the manifest's `files` table and asserts the validator throws `.invalid(field: "manifest.files.<name>", reason: "reserved artifact filename")`. It reads the names from the shared constant rather than hardcoding them, so the reserved-file rule follows a rename instead of stranding one side. `swift test --filter QuantManifestPayloadAgreementTests` -> 13 tests in 1 suite passed (12 before). The single-sourcing itself is proved by the compiler -- there is now one literal to get wrong, not three -- and the test is what keeps the rule it feeds honest.

**Commit.** `this commit`

### AUD-125 — memory_delete swallows every non-notPersisted archive error in an empty catch (reclassified from S1: the dominant failure path was already rethrowing)

- **Severity / tier:** S2 / Tier A
- **Project:** memory
- **Location:** `sources/TinyTitanMemory/ContinuityStore.swift:110`
- **Category:** silent failure, wrong result reported to the model
- **Status:** DONE
- **Host:** Mac (primary)
- **Discovered by:** L2 module pass + §5 facade sweep

**Evidence before.** Read at the cited lines and confirmed by the auditor. The function recalls the fact, then `do { try await engine.archive(...) } catch ContinuityError.notPersisted(let detail) { throw MemoryError.notPersisted(detail) } catch {}` and `return true` (:106-111). The notPersisted branch carries the right reasoning — 'Answering "deleted" would bring the fact back after a restart with the model believing it gone' — and the very next catch discards that reasoning for every other failure (journal write error, lock loss, decode error). So the tool reports success on the same condition the code says must not be reported as success.

**Fix.** ContinuityStore.delete now uses the same `catch let error as ContinuityError { throw Self.translate(error) }` the store path already uses, so no archive failure can be swallowed and answer true. translate(.notPersisted) returns .notPersisted (ContinuityStore+KeyMapping.swift:132-133), so the message the RAM-only case earns is byte-identical to before. The empty catch is gone, which §0 names as an anti-pattern regardless of reachability.

**Evidence after.** RECLASSIFICATION, recorded rather than quietly narrowed: the sweep's claim that memory_delete 'answers deleted when nothing was archived' is not reachable by the main failure path. ContinuityEngine+Internals.swift:42-47 wraps EVERY journal append failure as ContinuityError.notPersisted, which is the branch the old code already rethrew. The only errors that could reach the empty catch were non-notPersisted ContinuityError cases, and archive has no reachable throw of those (the key is validated before the call). So this is hardening, not a live wrong answer: S1 -> S2, and no failing-before test exists because no observable behaviour changed. The behaviour IS already pinned by MemoryJournalFailureTests.aDeleteTheJournalRefusesIsAFailure and by ContinuityCore's failedWritesAreReportedNotSwallowed. Evidence run: swift build clean, those suites 57 + 4 tests green.

**Commit.** `see the audit(AUD-125) commit`

### AUD-129 — Two documents tell the reader to pass --models-directory; the parser's flag is --models-dir, so the documented flag cannot work

- **Severity / tier:** S2 / Tier B
- **Project:** docs
- **Location:** `docs/agent-memory.md:141 and docs/side-engine-tasks.md:328`
- **Category:** surface wired to nothing (§5)
- **Status:** DONE
- **Host:** Mac (primary)
- **Discovered by:** §5 facade sweep

**Evidence before.** Reported by the discovery sweep at these lines; NOT yet re-read by the auditor — verification precedes any fix. Reported against ServerArguments.swift:305 (the real flag) and :421 (unknown flags throw), so the failure is loud rather than silent — but the documented spelling can never work, and a reader following the doc gets an error.

**Fix.** Both documents now name `--models-dir`, which is the flag the parser accepts (ServerArguments.swift:305): docs/agent-memory.md's TINYTITAN_SIDE_ENGINE row and docs/side-engine-tasks.md's side-engine paragraph. No other document in the tree, README or the wiki spells it the other way.

**Evidence after.** Verified before fixing, as the row required: `grep -rn models-directory sources/ tests/` returns nothing while `--models-dir` is the parsed case at ServerArguments.swift:305 and the unknown-flag throw means a reader following the old spelling got a loud error, not a silent default -- which is why this was S2 and not S1. After the edit `grep -rn models-directory docs/ README.md` returns only the ledger rows that record this finding. No code change, so no new test.

**Commit.** `this commit`

### AUD-128 — Test bodies that cannot fail: preconditions recorded as expressions, one self-referential digest assertion, and non-throw-only bodies

- **Severity / tier:** S3 / Tier C
- **Project:** tests
- **Location:** `tests/ (18 sites, see evidence)`
- **Category:** tests that assert nothing
- **Status:** OPEN
- **Host:** Mac (primary)
- **Discovered by:** §5 facade sweep

**Evidence before.** Reported by the discovery sweep at these lines; NOT yet re-read by the auditor — verification precedes any fix. RMSNormReferenceTests.swift:55 (`_ = (RmsNormRef.apply, "precondition…")`, comment says documentation-only); Sha256VerifierTests.swift:32 (asserts against a hex produced by the function under test, so it passes by construction); ManifestReaderTests.swift:625/:633; RouterTopKTests.swift:220/:228 (comment concedes the precondition traps); non-throw-only bodies at HyperConnectionTests:31, SampleTopK64Tests:16, PLEHashTests:167, Qwen38FlashSchemaTests:191/:210, ReasoningControlTests:20/:118, RoleUniformityTests:21/:94/:100, PrefillGroupedRoutedMoETests+Binding:144, HTTPServerTests:607, QuantManifestPayloadAgreementTests:269/:324; ClientCLITests.swift:162-168 whose own comment reads 'Not a test of anything'. The framework is Swift Testing throughout (209 files import Testing, no XCTest).

**Evidence after.** None yet — S3 by rule sweep; Sha256VerifierTests:32 is behavioural and reclassifies S2 once confirmed.

### AUD-137 — An unreachable ?? 262_144 fallback on a non-empty constant array

- **Severity / tier:** S3 / Tier B
- **Project:** server
- **Location:** `sources/TinyTitanLib/ServerInference.swift:101 and OpenAIRequestValidator.swift:32-33`
- **Category:** defensive code for a case that cannot happen
- **Status:** OPEN
- **Host:** Mac (primary)
- **Discovered by:** §5 facade sweep

**Evidence before.** Reported by the discovery sweep at these lines; NOT yet re-read by the auditor — verification precedes any fix. Reported as `supportedContextTokens.max() ?? 262_144` where the array is a non-empty constant, so the fallback can never run — a magic number standing in for an impossible branch.

**Evidence after.** None yet.

### AUD-138 — Four try?-to-empty reads split off AUD-134: recall quality on a background path, and two protocol methods with no production caller

- **Severity / tier:** S3 / Tier C
- **Project:** memory
- **Location:** `sources/TinyTitanMemory/MemoryRetrieval.swift:52, :233; sources/TinyTitanMemory/ContinuityJournalStore.swift:71, :92`
- **Category:** error swallowed into an empty answer (low reach)
- **Status:** OPEN
- **Host:** Mac (primary)
- **Discovered by:** split from AUD-134 during its verification

**Evidence before.** Read and confirmed while narrowing AUD-134. MemoryRetrieval.swift:52 skips one promoted fact and :233 caches a question with an empty candidate set when the store read throws -- the cost is recall on the background retrieval path, never a wrong answer shown as right. ContinuityJournalStore.sessions():71 and search():92 both `guard let taskID = try? await store.taskID(...) else { return [] }`; they satisfy the SessionJournal protocol (SessionJournal.swift:25-28) and the only production consumer of `journalStore(for:)` is consolidation, which calls `turns()` (MemoryBackend+Consolidation.swift:88-91). Benign only while that stays true, which is why this is a row and not a dismissal.

**Fix.** Decide once: either give these reads the same report AUD-134 adds, or note in the protocol that a journal read cannot distinguish empty from failed and leave them.

**Evidence after.** None yet.

### AUD-131 — release-notes-v5.8.md still advertises TINYTITAN_KEEP_WIRED as a live tri-state although the knob was deleted by 3eb11cf and the repo has a Superseded-banner convention for exactly this

- **Severity / tier:** S3 / Tier C
- **Project:** docs
- **Location:** `docs/release-notes-v5.8.md:132`
- **Category:** stale documentation, documented switch with no consumer (L0/§6)
- **Status:** DONE
- **Host:** Mac (primary)
- **Discovered by:** L0 repository pass + §6 unused-code sweep (row re-added: the first append script aborted before writing it)

**Evidence before.** Re-derived on this branch rather than reconstructed from the aborted run, and the wider hypothesis was tested and rejected first: a scan of every TINYTITAN_* token in README.md and the non-historical docs, and in tools/ and plugins/, against actual readers (`${VAR}` in bash, a string literal in Swift, `os.environ`, `process.env`) found NO documented-but-unread switch -- TINYTITAN_ALL_MODELS (tools/tinytitan_models.sh:296 sets it, :440 reads it), TINYTITAN_STUB_SERVER_SECONDS (tests/TinyTitanServer/ClientCLITests.swift:168) and TINYTITAN_RELEASE_{NOTES_MAX_CHARS,SKIP_GOLDENS_REASON} (tools/release.sh:422, :71) are all live, so that whole class is clean and only the knob-deletion case survives. `git show --stat 3eb11cf` ('runtime: remove every decode knob that measured a wash or a loss') deletes 18 tokens; 13 of them have no reader anywhere in sources/, tools/ or plugins/. Of the docs that name a dead token, all carry a superseded note except three dated measurement records (docs/qwen38-prefetch-predictor-study.md, docs/v4.2-experiments.md, docs/v4.3-predictive-prefetch-plan.md -- accurate as records of what was measured then, not claims about the current engine) and docs/release-notes-v5.8.md:132, which says in the present tense that '`TINYTITAN_KEEP_WIRED` is a tri-state so `=0` pages the expert cache out'. docs/release-notes-v5.1.md:57 and docs/qwen38-decode-profile-2026-09-05.md:3-13 establish the house convention for exactly this: a `> **Superseded.**` block that names what was retired and keeps the text as the record of what that release shipped. The real model sidecar and the tests confirm the knob is gone: tests/TinyTitan/Runtime/Configuration/ModelProfileTests.swift:119-122 asserts 'The tri-state `TINYTITAN_KEEP_WIRED` override is gone' and that the profile row decides.

**Fix.** Add the Superseded banner to the 5.8 note's 'Also in this release' item, in the form release-notes-v5.1.md:57 already uses, naming the deletion commit and where the surviving decision lives (the profile row). Historical measurement records are left alone.

**Evidence after.** docs/release-notes-v5.8.md carries the Superseded block in the form release-notes-v5.1.md:57 already uses: it names 3eb11cf, points at ModelProfile as the thing that decides now, cites ModelProfileTests:119-122 as the record that the override is gone, and keeps the bullet as what 5.8 shipped. Verified each claim against the code before writing it (`git show --stat 3eb11cf`, the test comment, and the absence of the token anywhere under sources/). Markdown has no lint gate here, so the check is factual rather than mechanical: no other doc claims a retired knob as current except three dated measurement records, which are left alone deliberately because they describe what was measured then, not what the engine does now.

**Commit.** `dbb8bf6`
