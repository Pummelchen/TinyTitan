## TinyTitan 5.14 — the browser window moves to DeepSeek Harness 0.2.0-rc.2

This release is about the chat window, not the engine: the `dsh-tinytitan`
bundle and both TinyTitan plugins now run on DeepSeek Harness **0.2.0-rc.2**,
the release that removed the settings file and the preset files the bundle used
to write. No engine behaviour changed — sampling, kernels, the model format and
the API are untouched — and every installed golden baseline is byte-identical.

### What is new

- **The browser window runs on DeepSeek Harness 0.2.0-rc.2.**
  `tools/dsh_local.sh` installs that release, and
  `tools/server_launcher.sh --web` opens it with the TinyTitan route in place.
  Two things moved with the harness: the `llm-pi-ai` route is applied through
  the harness's own `settings` service instead of a `settings.yaml` file (0.2.0
  imports a legacy file once, then renames it `settings.yaml.imported`), and the
  agent preset is registered with the preset registry instead of being written
  as a file. The preset keeps its id (`tinytitan`), so sessions already using it
  keep working; it becomes the default for a new session only while nothing else
  is selected, and a choice made on the Agent presets page is never overwritten.
  **Checked by** booting a throwaway harness whose route had been edited: the
  boot rewrote it from the catalog, the registry gained `selectedDefault:
  tinytitan`, and a `models/` change mid-session rewrote it again — with no
  `settings.yaml` anywhere. Both plugin suites pass (84 tests: 83 pass, 1
  skipped; the LAN manager's 107).
- **A model installed while the window is open reaches the picker again.** The
  `models/` watcher refreshed through the old file path, which 0.2.0 cannot
  satisfy; boot and the watcher now share one refresher, so installing a model
  and using it in the same sitting needs no harness restart.
- **`compactionHeadroomTokens`** (optional) sets the compaction trigger's
  headroom in the generated preset. The engine compacts at
  `min(window × ratio, window − maxTokens − headroom)`, and its stock
  65,536-token headroom holds the trigger near 62% of a 262,144-token window
  instead of the documented 80%; on a narrower declared window it leaves no
  pressure budget at all and compaction stops. Leave it unset for the harness's
  policy, set `0` to let the ratio govern, or the cap's quarter as a guard.

### Also in this release

- **`tools/dsh_local.sh status` reports the installed harness version**, not the
  pin. A private harness a release behind used to read as `ok`, which is how the
  0.2.0 move left this machine's private copy on `0.1.6-alpha.2` unnoticed. The
  route check also accepts the profile patch, where 0.2.0 keeps the route after
  the first boot.
- **The bundle's documentation matches what it writes.** The plugin README and
  the wiki's plugin page describe the settings-service route, the registered
  preset and the upgrade path; `docs/handover-tinytitan.md` carries the new pin,
  and `docs/dsh-upstream-asks.md` says plainly that its three asks were read on
  `0.1.6-alpha.2` and have not been re-read since.
- **The release process no longer gates on the internal-speed record.** The
  owner removed the mandatory timing step on 2026-10-01; the tool and its
  records stay, and `docs/release-process.md` §4b says when taking one is still
  worth it. The reason is the machine: its timings swing past the 10% threshold
  whenever a browser, WindowServer or a game holds the GPU.
- **CI's first plugin step no longer needs the JS toolchain.** A test that parses
  with the harness's `js-yaml` now skips with a reason on a checkout where the
  toolchain is not installed yet, which is the state CI runs it in — the suite
  was red on `main` from the port commit until that was fixed.

### Verification

- `tools/lint.sh` — all eleven gates, clean.
- `swift test --no-parallel` — 1,496 tests in 224 suites.
- **7 of the 16 stored golden baselines compared byte-identical** (`qwen36-{4,8}`,
  `qwen38-4`, `qwen35-{4b,9b}-{4,8}`) — every installed model that has a golden
  target — then a clean scratch release build with a clean warning scan.
- **Not checked, because their install is not under `models/` and nothing may be
  fetched to change that**: `ornith-4`, `ornith-8`, `qwen38-8`, `agentworld-4`,
  `agentworld-8`, `katcoder-4`, `katcoder-8`, `qwen35-2b-4`, `qwen35-2b-8`.
- **No internal-speed record this release**, by the 2026-10-01 decision above:
  the release path no longer gates on timing, and this machine was carrying a
  game and a browser during the cut.

### Checksum

`tinytitan-5.14-macos-arm64.tar.gz` sha256: `SHA256_PENDING`
`tinytitan-5.14-macos-arm64.tar.gz` size: `ARCHIVE_BYTES_PENDING` bytes
