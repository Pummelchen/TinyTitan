## TinyTitan 5.18 — the launcher installs what you ask for

This release makes the first run the easy one: a model that is not on disk can be
fetched from the launcher, the menu offers the models this checkout supports but
does not have, and every long download, conversion and repack draws a real
percentage instead of a silent terminal.

### What is new

- **A model you ask for is downloaded and converted, not refused.**
  `tools/server_launcher.sh --model katcoder --bits 4` on a checkout without it
  used to end in an error and a pointer at another tool. It now asks once (a
  20–220 GB download is not something to start unannounced), runs
  `tools/install_models.sh` with the catalogue key the installer is addressed by,
  re-reads the catalog and starts normally. `--install` answers ahead of time. A
  **piped or `--dry-run` invocation never downloads**: it prints the
  `tools/install_models.sh <key>` command instead, which is what keeps CI and
  scripted use safe. **Checked by** `benchmark/test_launcher_install.py` —
  fourteen tests, including that neither a pipe nor a dry run changes `models/`,
  and that every key the new stem→key map returns is one the installer's
  `CATALOGUE` carries — and by a live dry run on this checkout.
- **The menu offers what you do not have yet.** Run the launcher with no
  arguments and it lists the installed models, then the supported ones that are
  absent, marked `not installed` in the same columns (engine, size, thinking
  levels) so they can be compared rather than guessed at. Picking one fetches it
  and then takes the ordinary path, so nothing downstream knows where the bytes
  came from. **Checked by** the same suite: the offer rows, the
  `Rows N-M are not installed yet` hint, choosing one, declining one, and the
  regression guard that Enter on the default row still launches an installed
  model.
- **Downloads, conversions and repacks report a percentage.** A ~70 GB MoE fetch
  or a ~360 GB 125B fetch runs for hours, and the converters used to print
  nothing between finished shards. One line now covers it — rewritten in place on
  a terminal (with an ETA from the observed rate), one plain line per 10% in a
  log or a pipe, `TINYTITAN_NO_PROGRESS=1` to silence it — and the same format is
  used by `TinyTitanRepack`, whose `ModelInstallProgress` stream existed but was
  consumed by nobody:

  ```
  converting  38.7/91.6 GB  42%  4/13 shards  eta 12m
  installing  12.3/38.1 GB  32%  eta 4m10s
  ```

  **Checked by** `benchmark/test_progress.py` (twelve tests: both output modes,
  the ETA, the shared `12.3/38.1 GB` unit rule, a closed pipe, a disabled line),
  the converter suites (51 tests together), and `RepackCLITests` (six, including
  a new local-import case that asserts a real CLI run reports progress to a
  pipe).
- **The DeepSeek Harness plugin keeps a long session going.** `dsh-tinytitan`
  gains three switches a profile opts into: `autoGoal` (a direct prompt arms a
  goal, so the harness keeps working until the model completes it), `autonomy`
  (the preset policy plus the fresh-agent `ralph` loop) and `handoff` (an
  unfinished objective moves to a fresh child context when the budget is spent,
  with a wall-recovery path for a turn that ends on `max-tokens`). The handoff
  budget is derived from the routed model's declared window — `min(0.6 × window,
  window − 131,072)`, which is 131,072 on 256K, 600,000 on a 1,000,000-token
  route and 629,145 on 1,048,576 — rather than a constant. The shipped bundle
  patch writes all three as `false`: each starts work nobody asked for, so a
  profile names the field it wants. **Checked by** the plugin suite (130 tests,
  129 pass and 1 skip) and the live chain recorded in
  `plugins/dsh-tinytitan/README.md`.

### Also in this release

- **Two memory defaults in the documentation were wrong and are fixed**: the
  bootstrap cap is 60 records (not "forty"), and the guard ships **on**
  (`TINYTITAN_MEMORY_GUARD=1`) rather than off. The wiki's agent-memory table
  gained the guard and consolidation rows and lost two stale cells.
- **Two engineering analyses landed** as repository documents:
  `docs/clickhouse-long-session-verdict.md` (a columnar store does not belong in
  the memory or prompt path; it belongs in the evaluation/telemetry plane) and
  `docs/plan-fact-keeping-over-sessions.md` (the seven-step program to keep facts
  over long sessions, each step tied to the metric it must move).
- **CI runs the new suites**: `test_launcher_install` and `test_progress` joined
  the dependency-free Python step.

### Performance

No timing was measured on this commit. Nothing here touches the engine's hot
path: the Swift change is confined to the `TinyTitanRepack` tool's output, and
the rest is the launcher, two Python converters and the DSH plugin.

### Verification

- `tools/lint.sh` — all eleven gates clean.
- `swift test --no-parallel` — 1,528 tests in 234 suites.
- **7 of the 16 stored golden baselines compared byte-identical**
  (`qwen35-4b-4`, `qwen35-4b-8`, `qwen35-9b-4`, `qwen35-9b-8`, `qwen36-4`,
  `qwen36-8`, `qwen38-4`), then a clean scratch release build with a clean
  warning scan.
- **Not checked, because their install is not under `models/` and nothing may be fetched to change that**:
  `ornith-4`, `ornith-8`, `qwen38-8`, `agentworld-4`, `agentworld-8`,
  `katcoder-4`, `katcoder-8`, `qwen35-2b-4`, `qwen35-2b-8`.

### Checksum

`tinytitan-5.18-macos-arm64.tar.gz` sha256: `SHA256_PENDING`
`tinytitan-5.18-macos-arm64.tar.gz` size: `ARCHIVE_BYTES_PENDING` bytes
`tinytitan-lib-5.18-macos-arm64.tar.gz` sha256: `LIBRARY_SHA256_PENDING`
`tinytitan-lib-5.18-macos-arm64.tar.gz` size: `LIBRARY_BYTES_PENDING` bytes
