## TinyTitan 5.21 — the prefill gather stops queueing

This release is mostly about waiting: a prefill used to issue its n-gram table
reads one after another, and now it does not. Around it, the context window gets
a 2M YaRN rung for hosts that can hold it, three decode levers arrive as opt-in
knobs that change nothing when unset, and the installer and converter stop
losing a long download to one bad request. The generation path is otherwise
untouched: no default moved.

### The prefill n-gram gather reads on eight workers

- **The reads are concurrent** (`93de0c7`). A token needs
  `headsPerNgram * (ngramSize - 1) = 16` rows of 320 B, and
  `gatherPLERowsPrefill` issued those `pread`s serially on the prefill critical
  path — 6,592 of them for a 412-token prompt. At ~100 us per read the cost is
  round-trips, not bandwidth, so independent positions can be fetched at once.
  The chunk's rows are still resolved in token order (the hash window is
  stateful); only the fetching is fanned out, through the new
  `NgramTableReader.gatherConcurrently`, which follows the existing
  `Int8AffineGEMV` idiom: workers write disjoint regions, so no byte is shared
  and `pread` needs no lock.
- **Eight is the measured knee** on the 125B install, 412-token prompt,
  `gather_ms`, two runs per arm:

  | readers | `gather_ms` |
  | --- | ---: |
  | serial (previous) | 647, 673 |
  | 4 | 176, 192 |
  | **8 (shipped)** | 101, 117, 109, 125 |
  | 16 | 109, 113 |

  That is 660 ms down to ~110 ms, 2.3% of prefill, and it scales with prompt
  length.
- **`gather_ms` now covers prefill too.** It timed only the decode gather, so
  the prefill cost was invisible and could only be extrapolated. The end-to-end
  total is not usable as evidence — `prefill_s` swung 23.0-25.5 s across arms —
  so the counter is where the numbers above come from.
- **Decode is deliberately still serial.** Its 16 reads are usually warm (4.4 ms
  across 65 generated tokens, against ~1.6 ms/token cold), so parallelising them
  would add dispatch to the hot loop for at most ~1% while competing with expert
  I/O for SSD queue slots.

### A 2M context rung, still gated by memory

- **`--rope-scaling yarn --max-context 2097152` is accepted** (`f36a7c7`).
  262144 stays the native default and 1048576 the YaRN default. 2097152 over
  262144 is exactly the 8x LongRoPE (arXiv 2402.13753) reports without
  fine-tuning, and `YaRNRoPEParameters` derives factor 8 with no new code path.
  The KV cost at 2M is 12 KB/token — 24.6 GB at 8-bit — and the loader's memory
  guard, not the parser, is what refuses on a host that cannot hold it.
- **The rungs cannot drift from the help text.** Three hand-written literals
  ("524288 or 1048576") in the server usage, the CLI usage and the server's
  refusal now spell from `RuntimeConfiguration.supportedYaRNContextTokens`, so a
  new rung arrives documented or not at all.

### Three decode levers arrive inert, and stay that way unless asked

Each of these is an environment variable that changes nothing when unset. That
is deliberate: this repository does not move a default on the strength of a bench
instrument.

- **GPU keep-warm** (`248f795`) — `TINYTITAN_GPU_KEEPWARM_US`, unset means
  inert, and a non-integer or a value outside 100...20000 us refuses the request
  rather than quietly running a control that reads as the arm. §5bk measured
  -5.88% of pure device-execution time on the head GEMV when a second command
  queue filled the launch gaps; this is the engine arm that can be priced against
  a real decode. It reports its *achieved* period, because §5bk's achieved
  period was 616 us against a 200 us ask.
- **Per-layer expert I/O timeline** (`c262da7`) —
  `TINYTITAN_EXPERT_IO_TIMELINE=per-layer` gives each layer streamer its own
  coordinator; the shipped default stays shared. §5bh priced the coarser `.event`
  arm at -14.27%, and the per-model event's contiguous-prefix rule — one slow
  read holding every later release across all 48 layers — is the granularity that
  cost it.
- **Saturating expert use-count cap** (`9c858cd`) — `TINYTITAN_EXPERT_USE_CAP`
  clamps only the eviction price; the count keeps accumulating and the tie still
  falls to `slotLastUse`. §5y's offline replay kept a 2-7% placement win, and it
  is the only lever that survived §5ab's closures. Unset, the 96-slot policy is
  byte-for-byte v5.20's.

### A bad network day no longer ends an install

- **The shard URL resolve retries** (`59cffab`). `resolve_url` ran `curl -sIL`
  with no retry and `check=True`, unlike the chunked range fetches beside it. On
  2026-10-17 one transient HEAD failure escaped as `CalledProcessError` and killed
  the Ornith 1.5 install at 19 GB, after the range fetches had already survived
  two network drops. It now passes the same 5-attempt flags. Nothing else
  changes.
- **`install_models` works from any directory** (`1c12a0a`). The converter calls
  were the literal `tools/prepare_*.py`, which resolved against the caller's
  working directory: `./install_models.sh` run from `tools/` opened
  `tools/tools/prepare_agentworld.py` and died before fetching a byte. All six
  now use `$ROOT/tools/...`, and `ConverterPathTests` runs from `tools/` so a
  stub that ignores its argv can no longer hide a missing file. The fetch sites,
  status table and menu now say what is downloaded and which number is the
  installed size.
- **`prepare_agentworld` streams the shards with twenty fetchers** (`9dcade9`),
  measured at 2.47 MB/s on the live install, +47% over the old rate. The host caps
  each connection at 70-365 KB/s, so the transfer was latency-bound rather than
  CPU-bound: the converter used 0.8 s of CPU in 9 minutes while quantizing a real
  1.07 GB tensor measured 138 MB/s at 4-bit.

### The DSH bundles move to harness 0.2.1-rc.2

`dsh-tinytitan` and `dsh-lan-manager`, plus the launcher, now support exactly
harness 0.2.1-rc.2 (`65e3500`). The exact-version gate was doing its job: rc.2 is
not the release they named, so profile startup denied the plugin rather than
letting it write into a harness home it was not written against. Four pin sites,
both READMEs and the lockfile moved together; the previous release moved into
`OTHER_VERSIONS` in both suites, because that list is what "one supported
release" means — the old one becomes a refusal, not a permission.

### Also in this release

- **`docs:` the 2M window is scoped and blocked by KV memory on a 24 GB host**,
  so the rung above ships with the constraint written down next to it.
- **`docs:` the 5 tok/s question answered with the 125B's own numbers**, and the
  throughput decisions become checklist items with the co-activation gain marked
  for an A/B rather than assumed.
- **`ci:` the expert use-count A/B suite runs**, which was tracked and named
  nowhere; gitleaks 8.30.1's one counter-value false positive is allowlisted.
- **V4.1-Flash port documentation** now plans 1017 and records what the revision
  changes; the port's container decisions are chosen for decode throughput.

### Performance

Measured on this commit, release build, 125B 4-bit install:

- **Prefill gather: 660 ms → ~110 ms** on a 412-token prompt (table above), the
  only end-to-end movement this release claims.
- **No decode default changed.** The three levers above are inert unless set, and
  no table from a previous release was re-run here — the v5.20 decode figures are
  not restated, because they were measured on v5.20's commit, not this one.

### Golden baselines

Three installed targets were checked and produced output identical to their
stored baselines: `ornith-4`, `ornith-8`, `qwen38-4`.

Thirteen were **not checked**, because there is no install under `models/` and
none may be fetched to make a gate pass: `qwen36-4`, `qwen36-8`, `qwen38-8`,
`agentworld-4`, `agentworld-8`, `katcoder-4`, `katcoder-8`, `qwen35-2b-4`,
`qwen35-2b-8`, `qwen35-4b-4`, `qwen35-4b-8`, `qwen35-9b-4`, `qwen35-9b-8`.

### Checks

- `swift build`, all products, clean.
- `swift test --no-parallel`: 1,787 tests in 256 suites across all seven bundles,
  exit 0.
- `tools/lint.sh`: every check green. ruff ran at the pinned 0.16.7 from pipx
  (a Homebrew 0.17.1 earlier in `PATH` shadows it, so the run pins the path);
  the converter gate reports `SKIPPED by ALLOW_MISSING_CONVERTER_DEPS=1` — this
  host has no `numpy` — and is named here rather than forced, as the release
  standard requires. CI's clean clone, which installs the converter stack, is the
  authority for that probe.
- **CI green on this commit** (`c7e882b`, run 39139662120: `test`,
  `thread-sanitizer` and `supply-chain` all success), which is the gate
  `tools/ci-green.sh` asks for and what a local gate run cannot be.
- Clean scratch release build with a clean warning scan; every staged executable
  asserted exactly `arm64`.

### Checksum

`tinytitan-5.21-macos-arm64.tar.gz` sha256: `SHA256_PENDING`
`tinytitan-5.21-macos-arm64.tar.gz` size: `ARCHIVE_BYTES_PENDING` bytes
`tinytitan-lib-5.21-macos-arm64.tar.gz` sha256: `LIBRARY_SHA256_PENDING`
`tinytitan-lib-5.21-macos-arm64.tar.gz` size: `LIBRARY_BYTES_PENDING` bytes
`tinytitan-5.21-tools.tar.gz` sha256: `TOOLS_SHA256_PENDING`
`tinytitan-5.21-tools.tar.gz` size: `TOOLS_BYTES_PENDING` bytes
