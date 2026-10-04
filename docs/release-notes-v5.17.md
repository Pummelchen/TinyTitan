## TinyTitan 5.17 — the katcoder install works again

This release fixes the converter defect that made every fresh
KAT-Coder-V2.5-Dev install fail (issue #19), makes an interrupted conversion
stop promptly and cleanly instead of ending in a crash report, and moves the
user documentation into the wiki.

### What is new

- **`tools/install_models.sh katcoder` and `katcoder both` install again.** KAT
  is the one checkpoint in this family that ships its routed experts per tensor,
  and the fusion that stacks them guarded against a repeated tensor by the
  tensor's *name* alone — while the converter adds one tensor once per requested
  width, and this family is always converted with `--bits 4 8`. The second width
  therefore looked like a duplicate, and the conversion stopped on the first
  routed expert with
  `duplicate source tensor model.language_model.layers.N.mlp.experts.0.down_proj.weight`.
  The guard is keyed by `(width, name)` now. **Checked by** converting the
  pinned checkpoint for real: shard `model-00000`, which carries 2,075
  per-expert routed tensors, converts at both widths and writes the fused
  `switch_mlp.{gate,up,down}_proj` stacks with a 256-expert axis (1.75 GB at
  4-bit, 2.82 GB at 8-bit, 27 tensors each), then continues to the next shard
  with no error. Installs already under `models/` were never affected: the
  defect was in conversion, not in the runtime. The full install and the §4
  verification of `docs/adding-a-model.md` remain open as TT-048.
- **An interrupted conversion stops when you stop it.** Ctrl-C, a `pkill` or a
  dying parent during the ~69 GB fetch now ends in under a second, with no
  orphan `curl` and no crash report. Two defects were involved: the fetcher
  threads were daemon threads left alive at interpreter finalization (CPython
  3.14 aborts there and files a macOS crash report; 3.13 only printed the fatal
  error), and a chunk retry could sit in a wait of up to two minutes without
  noticing the stop, so even a pool that was joined took ~20 s to die.
  **Checked by** `benchmark/test_prepare_agentworld.py` — twelve tests, of which
  the shutdown test reports three live fetcher threads against the previous
  commit — and by interrupting the real download: the converter is gone in 0 s,
  no `curl` is left, the log carries no fatal error and no crash report is
  written. A stopped download keeps its shards and resumes from them.
- **The user documentation is in the wiki.** The README is the front page again:
  what the repository delivers, one worked example, the benchmark table, the
  supported models and the links. The guides — getting started, cookbook,
  installation and configuration, runtime controls, the OpenAI-compatible
  server, the FAQ — are wiki pages, so each is found in one place instead of
  two. Wiki links to repository files are absolute GitHub URLs, because the wiki
  is a separate repository.

### Also in this release

- **The Qwen3.5-MoE converter has a test suite for the first time**:
  `benchmark/test_prepare_agentworld.py` pins the per-expert naming contract,
  the both-widths fusion, expert placement by index rather than by arrival,
  per-width emission, the incomplete-layer refusal and the shutdown behaviour.
  It runs in the converter gate in CI, from a synthetic shard, with no
  checkpoint and no network.
- **`docs/t6-reply-check-offline.md`** records the side-engine reply-check task
  measured on 417 real recorded replies: 66.7% precision and 38.5% recall
  against a pre-registered 95%/50% gate, so the task is **not wired**. The
  harness, the hand-audited labels and the tests that pin the label traps ship
  with it as the record.
- **The wiki's install page states the breakage honestly**: the `katcoder`
  command was broken in 5.4–5.16, there is no installer workaround on those
  builds (convert one width at a time with `prepare_agentworld.py --bits 4`),
  and a fresh full install is tracked as TT-048.

### Performance

No performance change and no new timings. This release touches the converter and
the documentation, not the runtime; the README's benchmark table is unchanged
and was not re-run.

### Verification

- `tools/lint.sh` — all eleven gates clean.
- `swift test --no-parallel` — 1,527 tests in 234 suites.
- **7 of the 16 stored golden baselines compared byte-identical**, then a clean
  scratch release build with a clean warning scan.
- **Not checked, because their install is not under `models/` and nothing may be
  fetched to change that**: `ornith-4`, `ornith-8`, `qwen38-8`, `agentworld-4`,
  `agentworld-8`, `katcoder-4`, `katcoder-8`, `qwen35-2b-4`, `qwen35-2b-8`.

### Checksum

`tinytitan-5.17-macos-arm64.tar.gz` sha256: `SHA256_PENDING`
`tinytitan-5.17-macos-arm64.tar.gz` size: `ARCHIVE_BYTES_PENDING` bytes
`tinytitan-lib-5.17-macos-arm64.tar.gz` sha256: `LIBRARY_SHA256_PENDING`
`tinytitan-lib-5.17-macos-arm64.tar.gz` size: `LIBRARY_BYTES_PENDING` bytes
