## TinyTitan 5.15 — the model format is renamed `.ssdai`

This release renames the install format: `.gturbo` is now `.ssdai`, end to end —
manifest magic, Swift vocabulary, CLI, tooling and docs. **It moves a name, not a
byte of payload.** Every install built before this release keeps loading, and not
one of them is modified. No engine behaviour changed, and the seven installed
golden baselines are byte-identical; the golden phase itself exercises the legacy
read, because every existing install still carries the old magic.

### What is new

- **The format is `.ssdai`.** `manifest.json`'s magic is written as `"SSDAI"` by
  the only writer, and the Swift vocabulary follows: `SSDAIFormatV1`,
  `SSDAIBinary`, `SSDAIDirectoryAccess`, `SSDAIManifestV1` and the rest — 28
  types across 73 files, with the format's own files renamed. The tools are
  `tools/ssdai_reader.py` and `tools/ssdai_diff_snapshot.py`, and the spec is
  `docs/ssdai-format.md`.
- **Every existing install keeps working, untouched.** Reads accept the legacy
  magic `"GTURBO"` for one release (`SSDAIFormatV1.isSupportedMagic` gates both
  manifest readers and the repacker's validator), and nothing rewrites a
  manifest in place: the receipt (`verified-install.json`) binds the manifest's
  digest *and* the directory path, so editing the magic would invalidate the
  receipt of every install in existence — 244 GB of them on this machine alone.
  An install moves to the new magic the next time it is rebuilt.
  **Checked by** two live runs on a real install whose manifest still says
  `GTURBO`: `TinyTitanRepack --verify-install --input-ssdai models/qwen3.5_4B_4Bit`
  reports "Verified 7 files (2,724,545,695 bytes)", and the CLI loads that same
  install and generates from it. Two tests pin the contract so it cannot drift:
  `legacyGTURBOMagicStillLoads` on the reader, and an assertion on the emitted
  magic in the repack encoder's test.
- **`--input-ssdai` is the flag; `--input-gturbo` stays accepted** as a
  deprecated alias for one release, so existing scripts keep working.
- **The directory's `.ssdai` suffix is a convention, not a check.** Nothing in
  the reader looks at the extension, so a directory still named
  `something.gturbo` works exactly as well.
- **Dated records keep the old word on purpose** — release notes before this
  one, the v4.x design and plan records, and `plan-dense-gturbo-installs.md` —
  because they record what the format was called when they were written. Their
  links to the spec were updated, and `docs/ssdai-format.md` gained a "Naming"
  section that states the rename, the compatibility window and this policy.

### Verification

- `tools/lint.sh` — all eleven gates, clean.
- `swift test --no-parallel` — 1,496 tests in 224 suites.
- **7 of the 16 stored golden baselines compared byte-identical** (`qwen36-{4,8}`,
  `qwen38-4`, `qwen35-{4b,9b}-{4,8}`) — every installed model that has a golden
  target, each of them loading a manifest that still carries the legacy magic —
  then a clean scratch release build with a clean warning scan.
- **Not checked, because their install is not under `models/` and nothing may be
  fetched to change that**: `ornith-4`, `ornith-8`, `qwen38-8`, `agentworld-4`,
  `agentworld-8`, `katcoder-4`, `katcoder-8`, `qwen35-2b-4`, `qwen35-2b-8`.
- The plugin suites are unaffected and green (84 tests: 83 pass, 1 skipped; the
  LAN manager's 107), and the 45 local documentation links resolve.
- **No internal-speed record accompanies this release.** The owner removed that
  step from the release path on 2026-10-01 (see `docs/release-process.md` §4b):
  this machine's timings swing past the 10% threshold whenever a browser,
  WindowServer or a game holds the GPU. The tool and its records remain for a
  quiet machine.

### Checksum

`tinytitan-5.15-macos-arm64.tar.gz` sha256: `SHA256_PENDING`
`tinytitan-5.15-macos-arm64.tar.gz` size: `ARCHIVE_BYTES_PENDING` bytes
