## TinyTitan 5.19 — the audit release

This release is cut at the close of an independent audit of the whole tree,
run 2026-10-06 to 2026-10-11: 210 findings filed in a public ledger
(`docs/audit-2026-10-06/`), 209 fixed or closed with evidence during the
audit, and the 210th — this release itself — closed by it. The headline for a
user is the installer: it verifies every archive it downloads, v5.18
published only two of the three, and the closed installer therefore refused
the newest release. 5.19 publishes all three and installs again. The
generation path did not move: the stored golden baseline for the one model
installed here compares byte-identical.

### The installer installs again

- **The installer verifies all three archives, and this release carries all
  three.** `tools/install_tinytitan.sh` used to verify the engine tarball
  only when its checksum happened to download, and never verified the tools
  tree it extracts into `~/.tinytitan/src` and then runs. Both downloads now
  go through one helper with no fall-through: a missing `shasum`, an
  unfetchable `.sha256`, or a mismatch removes the staging directory and dies
  before anything is unpacked (AUD-109). The tools tree stopped being a
  generic source archive and became a release asset,
  `tinytitan-5.19-tools.tar.gz`, which `tools/release.sh` stages from
  `git archive` of the tag, asserts member by member, and publishes with its
  own digest (AUD-139). v5.18 shipped two of the three assets, so the closed
  installer refused it on "No checksum published for the tools download" —
  **v5.19 is the first release that installer accepts.**

### Sampling knobs are validated and applied on both engines

- **The CPU serving path applies `repetition_penalty` and
  `presence_penalty`.** It used to accept `seed`, `presence_penalty` and
  `repetition_penalty`, apply none of them, and answer 200 — a different
  completion than the same request on the GPU engine, silently (AUD-176).
  The CPU sampler now carries both penalties and applies them the way the
  GPU sampler does, before the argmax, with the prompt ids seeding the
  history; an id outside the row is skipped rather than trapping.
- **The wire's bounds are the engine's bounds.** `repetition_penalty` below
  one is refused with 400 `invalid_request_error` and the parameter named,
  so one request gets one answer on both engines and no penalty is ever
  served inverted (AUD-179); HTTP bounds `presence_penalty` like the CLI
  does instead of accepting −2.5 and applying it (AUD-254); and a
  wrong-typed field in a well-formed JSON body is answered with the
  parameter named rather than "malformed JSON request" (AUD-193).

### The server closes the doors the audit found open

- **Request bodies are parsed under a depth cap.** A 1 MiB body afforded
  ~10⁵ nesting levels with no cap; the schema walker now refuses at 64 with
  400 `invalid_tool_schema`, and the two walkers agree on what a step is
  (AUD-108).
- **A corrupted sidecar is refused at validation** instead of trapping
  `tableRowCount` on a negative value at load (AUD-107).
- **A pending model switch can no longer be overtaken** by new work for the
  resident model — in release builds the same race could run the old engine
  after its teardown. The fix is structural: waiters register through a
  lock-guarded `SuspensionSlot` in the same synchronous actor step as the
  check that queued them (AUD-143).
- **`agent-default-model` gets repaired when the model it names stops being
  served**, by the boot-time route refresh and by the settings-file route
  writers (AUD-163, AUD-172). The repair is deliberately narrow: a default
  naming a served model is kept even when it is not first, a default another
  provider owns is kept, and a read that throws skips the repair.
- **The LAN surface holds its guards on every route.** An absent Origin no
  longer passes the mutating-route guard, and matching `Host` no longer
  makes an origin "same-origin" (AUD-132, AUD-148); `readJsonBody` checks
  content-type (AUD-147); `POST /v1/models/unload` sits behind the
  browser-form gate like every other mutating route (AUD-167); the two peer
  readers cap what a hostile host can buffer into them (AUD-168); a failed
  tailscale or Bonjour probe no longer silently shortens `/peers` (AUD-133);
  the shipped group key can no longer buy a mutation from off-loopback — it
  answers 403 and names `groupKey`/`DSH_LAN_KEY` (AUD-123); the IPv4
  allowlist no longer admits 169.254.0.0/16 by default (AUD-155); and the
  Swift and JavaScript copies of the shipped default key are pinned equal by
  the docs gate (AUD-175).
- **A chat stream whose finish frame is refused by the backpressure cap no
  longer ends with `data: [DONE]`**: both terminal frames go through the
  same checkpoint as every other frame, and a refused finish fails the
  stream, cancels the generation and closes the outbox (AUD-194).

### The load path checks what it trusts

- Metadata reads at the trust boundaries are bounded before they allocate:
  the profile constants, the seven unbounded metadata reads, the CPU
  snapshot's shard names (contained, `O_NOFOLLOW`), the receipt file name
  (declared once instead of twice), and `manifest.json` reads that still
  used the 4 MiB ceiling the load path moved off two years ago (AUD-113,
  AUD-142, AUD-150, AUD-164, AUD-165, AUD-192, AUD-118).
- A quantized weight width outside [4, 8] is refused by the CPU snapshot
  reader (AUD-161), the sidecar geometry reader no longer treats every
  absent optional field as fatal for one family (AUD-117), the six
  CommonCrypto SHA-256 statuses on the integrity-hash path are checked
  (AUD-112), and the env-named trace files open `0o600` with `O_NOFOLLOW`
  (AUD-111).

### The launcher keeps its promises

- **Ports and replies are read as what they are.** `--port`/`PORT` values
  that bash arithmetic would read as octal or evaluate as expressions —
  `-1`, a blank, `08757` — are refused by the port guard with its range
  rules actually evaluated (AUD-297, AUD-304), and a zero-padded menu choice
  (`010`) takes the row the person meant, not row 8 (AUD-305).
- **The knobs it reads are read once, validated, and honored.** A bad
  `TINYTITAN_PHYSICAL_RAM_BYTES` is refused instead of printing three
  arithmetic errors and finishing the plan anyway (AUD-300); `TINYTITAN_MEMORY=1`
  — the spelling the docs name — now turns the memory switch on the same way
  `true` and `on` do, across all three readers (AUD-301, AUD-302); the two
  warm-up numbers are checked where they are read, so a word in
  `TINYTITAN_WARM_TOKENS` fails before the model loads instead of after
  (AUD-299); `TINYTITAN_PYTHON` is tested like the interpreters the search
  finds (AUD-295); and `install_models.sh --all-8bit` exits on what it
  installed rather than on its catalogue's last row (AUD-253).
- **The launcher's lifetime promise now says what the code does**: the
  server does not outlive the script, the trap fires when the launcher
  returns, and the stale-server stop keeps its documented reason — behaviour
  deliberately unchanged (AUD-186).

### The DeepSeek Harness plugins move with the harness

- Both bundles (`dsh-tinytitan`, `dsh-lan-manager`) and the launcher are
  updated for **deepseek-harness 0.2.1-alpha.2**, with the plugin suites
  pinned against both the package's peer range and the launcher's harness
  pin, so the next drift fails CI in seconds instead of at a user's boot.

### Also in this release

- **The audit itself is part of the tree**: `docs/audit-2026-10-06/` — the
  210-row ledger (`ledger.json`, rendered `ledger.md`), the environment
  record, the per-finding evidence and the handover. CI now carries a
  **supply-chain job** (gitleaks 8.30.1 over the full history, pip-audit
  2.10.1 over the converter's requirements, `npm audit` over both plugin
  lockfiles), uploads the **llvm-cov coverage JSON** as an artifact from the
  serial test run, and releases are gated on **CI green on the tagged
  commit** (`tools/ci-green.sh` — v5.18 was published over a red run, which
  is what started the audit's release track, AUD-105).
- **The release script's own gates are verdict functions now**: the test
  gate reads every target's summary line and the run status (AUD-262), the
  compiler-warning gate cannot pass over a log it cannot read (AUD-266), the
  `gh` preconditions distinguish an invisible repository from a missing
  release (AUD-267), and `assert-arch.sh` no longer reads an artifact whose
  magic it cannot open as a pass (AUD-265).
- **About forty benchmark and evaluation drivers answer a status.** The
  audit's largest single class was drivers that printed a verdict and
  returned 0 regardless: import-time servers over empty arms, medians
  computed over survivors, denominator-zero cells reported COMPLETE, run
  counts read as octal. Every one now refuses the run it did not have, and
  the exit-code convention is pinned by suites (AUD-210 through AUD-293,
  plus AUD-306 and AUD-307).
- **Dead code deleted, standards enforced**: the unused prefill-diagnostics
  types, two uncalled setters (AUD-259, AUD-310), an unreachable fallback
  (AUD-137, AUD-145), stdout-trace writes from the library (AUD-180); the
  500-line production-file rule and the library-facade allowlist are gates
  now, not review folklore (AUD-151, AUD-158).
- Documentation: the wiki's agent-memory table, the server guide and the
  launcher docs were corrected where the audit found them describing
  behaviour the tree does not have; `docs/repository-layout.md` records the
  file splits; the engineering paper's figures refuse a run they did not
  read (AUD-308, AUD-309).

### Performance

No timing was measured on this commit. The audit's mandate excluded
performance and numerics work, and nothing here changes the generation path:
the stored golden baseline for the installed model compares byte-identical,
which is the direct evidence.

### Verification

- `tools/lint.sh` — all eighteen gates clean.
- `swift test --no-parallel` — measured on the tag during the release dry
  run; the count lands here when it is seen, not before.
- **1 of the 16 stored golden baselines compared byte-identical**
  (`qwen38-4`, the 125B install under `models/`), and the library contract
  suite ran against the same install: 4 tests in 68.9 s — two engines on one
  device, two generations on one session, mid-decode cancellation,
  `unload()` shutdown.
- **Not checked, because their install is not under `models/` and nothing
  may be fetched to change that**: `ornith-4`, `ornith-8`, `qwen38-8`,
  `agentworld-4`, `agentworld-8`, `katcoder-4`, `katcoder-8`, `qwen35-2b-4`,
  `qwen35-2b-8`, `qwen35-4b-4`, `qwen35-4b-8`, `qwen35-9b-4`, `qwen35-9b-8`,
  `qwen36-4`, `qwen36-8`.
- Clean scratch release build with a clean warning scan; every staged
  executable asserted exactly `arm64`.

### Checksum

`tinytitan-5.19-macos-arm64.tar.gz` sha256: `SHA256_PENDING`
`tinytitan-5.19-macos-arm64.tar.gz` size: `ARCHIVE_BYTES_PENDING` bytes
`tinytitan-lib-5.19-macos-arm64.tar.gz` sha256: `LIBRARY_SHA256_PENDING`
`tinytitan-lib-5.19-macos-arm64.tar.gz` size: `LIBRARY_BYTES_PENDING` bytes
`tinytitan-5.19-tools.tar.gz` sha256: `TOOLS_SHA256_PENDING`
`tinytitan-5.19-tools.tar.gz` size: `TOOLS_BYTES_PENDING` bytes
