## TinyTitan 5.20 — the 125B's tuned cache becomes the default

This release carries what the Qwen3.8 decode campaign delivered: the expert-cache
ceiling is floored at the 24 GiB the budgets were tuned on, the QSA indexer no
longer reserves its full-context buffers up front, expert blob reads are split so
the miss fixup runs on the GPU while the tail fetch is still in flight, both
resident tiers scale with the machine, and the load-time RAM lines report the
cache that actually ran. The generation path did not move: greedy text is
byte-identical across every arm that measured it, and the stored `qwen38-4`
golden baseline compares identical.

### The shipped default on a 24 GiB Mac is the tuned cache

- **At or above the tune, the tuned budget is kept** (`3ed7102`).
  `affordableExpertCacheBudget` used to clamp every tuned budget to a third of
  physical memory, so a 24 GiB machine was handed 8 GiB for a model tuned at 12;
  the third still applies below 24 GiB. Qwen3.8-Flash-Next 4-bit now derives 96
  expert slots instead of the clamped 64, with no operator flag: the load line
  reads `ram profile cache=11.88G slots=96 resident_floor=3.72G
  estimate=15.60G`.
- **Measured** (§5p, at `3ed7102`, release build, greedy, `--max-new 128`):
  4.811 tok/s on the old build at 64 slots, 4.720 on the new build at 64, **5.156
  at 96 slots at the full 262144 context**, **5.045 with no flags at all**. The
  text is byte-identical across all four arms.
- **Confirmed at 384 tokens on mains power, interleaved and paired** (§5r, at
  `0242f65`): the 96−64 paired deltas are **+0.776** and **+0.384 tok/s**, two
  rounds, both positive. The counters move with the clock — 76 GB of expert reads
  against 112 GB (−32 %), hit rate 0.845 against 0.771, evictions 31,293 against
  48,582, `io_ms` 69–72 against 100–105.
- **Effect by family at ≥24 GiB**: qwen3.6 35B-A3B 4-bit gets its tuned 10 GiB
  back (the ~+1.3 % that `808cea3` priced); Ornith 1.5 8-bit is unchanged, because
  its 8 GiB is exactly a third of 24 GiB so floor and clamp agree; machines under
  24 GiB are unchanged — the 8 GB evidence for the clamp still stands there.
- **This is not 6 tok/s.** The shipped default measures 5.02–5.07 in §5r's paired
  batch. The campaign's remaining gap is not attributable to a single knob, and
  this release does not claim otherwise.

### The QSA indexer's buffers grow on demand

- **The reservation is gone** (`45f55bb`). The indexer used to allocate its
  full-context key buffers on a layer's first use: 84 MB a layer at 262144,
  **0.94 GiB across qwen38's twelve full-attention layers**, wired from the first
  token of the shortest conversation. They now start at 8192 cells and double to
  the configured capacity, copying raw keys and pooled blocks bit for bit — the
  pattern `KVCacheManager.reserve` already uses, at the same call points. At short
  contexts the same cost is 0.12 GiB.
- **Why it mattered**: with the reservation in place, forcing the tuned 96 slots
  at full context paged 12.15 GiB of cache and measured 3.052/3.201 tok/s; the
  same 96 slots without it hold and measure 5.156. The floor above is only safe
  because this landed first.
- **Numerics cannot move**: selection runs over visible keys only, so the result
  is storage-independent. `QSAIndexerGrowthTests` (model-free) pins the start
  size, the doubling, the cap and that contents survive a grow.

### Expert blob reads are split, and the miss fixup overlaps the tail fetch

- **Four steps** (`87d072c`, `af26fc5`, `c4a431c`, `53d7b8d`): the parallel reader
  takes per-item byte counts (`tinytitan_expert_reader_fetch_ranges`); the
  streamer splits a blob read at the manifest's `down`-tensor offset;
  `ExpertLoadOperation` carries a first-stage publication signal; and the runner
  encodes and commits phase 1 over the miss slots as soon as the front (gate+up)
  range is resident, on the GPU, **while the tail (down) fetch is still in
  flight**. Residency and slot bytes are unchanged. Whole-region layouts, the
  event path and miss-free plans are untouched.
- **Measured +2.5 %** (§5c, at `53d7b8d`, two back-to-back pairs at
  `--max-new 384` with runner stats): 4.180 → 4.286 tok/s mean, ≈ +0.11 tok/s,
  with per-layer wait down 11.5 % (86.2 → 76.4 ms) and hit rate held equal
  (74.8 % → 74.2 %). That is below the plan's +0.3–0.5 tok/s projection, and the
  record says so: the fixup fills only the GPU-idle slice of the tail window.
- **Golden gate on the full chain**: `tools/golden-baseline.sh --check qwen38-4`
  reports output identical to the recorded baseline.

### Larger Macs get their resident tiers sized to them

- **Both tiers derive from the machine now** (`0242f65`). The third of RAM is a
  floor as well as a ceiling at or above the tune, so 48 GiB derives 16 GiB / 128
  slots, 64 GiB → 21.3 GiB / 160, and 128 GiB → the 256-slot top rung; a 64 GiB
  Mac previously got the same 96 slots as a 24 GiB one and left ~40 GiB of the
  tier idle. The prompt cache's RAM tier is derived the same way — 2 GiB at
  48 GiB and above.
- **Those rows are arithmetic pinned by test, not measurements**: this machine is
  24 GiB and §5q labels them that way. What was measured here is that the tune
  class does not move — the no-flag run banners `expert_slots=96
  prompt_cache_memory_mib=256` and golden `qwen38-4` is unchanged.
- **Why the prompt tier was flat-and-wrong**: prompt state measures 0.74 MiB per
  prefix token, so a fixed 256 MiB holds ~350 tokens and pushed a long tool-loop
  prefix onto the SSD tier beside the expert reads.
- **The launcher stopped overriding it**: `server_launcher.sh` wrote
  `--prompt-cache-memory-mib 256` into every GPU command, which hid the new
  derivation from exactly the users it was added for. It names the mode only now.

### The RAM log lines name the cache that actually ran

- **An override used to be invisible in the log** (`2c2a30b`). An operator who
  passed `--expert-cache-slots` changed the allocation while the load-time line
  kept printing the budget-derived rung, so the record described a configuration
  that never ran. Both diagnostic lines print the applied count now and name the
  derived one only when it was replaced; the flag-then-environment-then-derivation
  precedence is defined once and shared, which is what keeps the printed number
  from drifting from the allocated one. The text lives in `ExpertCacheDiagnostics`
  with six tests pinning it.
- **The warning used to blame the wrong thing** (`0355d2c`): `--ram-budget 12G`
  with `--expert-cache-slots 96` called the operator's own 96 slots "the minimum
  cache" and said "the cache is already at its smallest". Both are false when an
  override pinned the count — the target alone allows 64, and the gap is closed by
  lowering the flag. The line now branches on whether the derivation or the
  override set the count.

### Also in this release

- **`8832ca8`** split the DecodeMoE phase-1 encodes into their own file to satisfy
  the 500-line production rule. No behavior change.
- **A model-free expert-cache simulator** (`ab80d5e`) runs a route trace captured
  off the real install (`9a784af`) through uniform, flat-pair and heat policies,
  and is registered in CI. Its first verdict is a rejection: the flat
  per-(layer, expert)-pair cache is dead as modelled, because a 16-token working
  set exceeds a 12 GiB cache.
- **`TinyTitanBench` gained a decode-cadence mode** (`2c3fc36`):
  `TINYTITAN_BENCH_HEAD_CADENCE_MS` launches one command buffer per token
  interval, mimicking decode's once-per-token cadence. On the identical qwen38
  kernel, 9.42 ms back-to-back against 12.73/15.49 ms at 50/200 ms gaps brackets
  the in-engine 13.49 ms — the head's in-engine cost is launch-cadence state, not
  kernel inefficiency.
- **CI action pins moved**: `actions/setup-node` 7.1.0, `actions/setup-python`
  7.0.0, `github/codeql-action` 4.38.3.
- **The decode campaign is closed** at the delivered numbers. B1–B7 each closed by
  measurement, arithmetic or an operator decision — including B5, `lm_head` at
  4-bit, which the operator declined because the offline gate did not show it
  near-lossless: quality is not traded for tokens per second. The record is
  `docs/qwen38-decode-12gb-measurements.md` §5b–§5s.
- **Documentation**: the wiki Benchmarks page carries a provenance note on its
  decode table — four of its five medians sit *above* the peak the same page
  quotes for the same model and width, which cannot both hold for one build. The
  v4.7 `6.82 tok/s` row is left exactly as it is: refreshing it needs that
  protocol's own arm (TT-049 is open). `docs/ngram-table-sharing-plan.md`'s 8-bit
  projection was retargeted off the same stale anchor.

### Performance

Every figure above names the commit and the section it was measured at (§5c at
`53d7b8d`, §5p at `3ed7102`, §5r at `0242f65`), on this 24 GiB M3 (`hw.memsize`
25769803776), macOS 27.0.1, Swift 6.4 (swiftlang-6.4.0.34.1 clang-2100.3.34.1),
release build, mains power with `lowpowermode 0`, one model process at a time.
Nothing was re-measured at the tag commit itself: the tag adds only the version
literal and these notes, and the golden baseline is the check that speaks for the
generation path. **The README's benchmark matrix was not re-run for this
release.**

One methodological finding from §5r, because it changes how to read any single
number above: **the first arm after a model load is systematically slow**. Three
identical arms in one batch rose 4.075 → 4.406 → 4.632 tok/s, and a server arm's
first request decoded 4.099 where its second reached 5.431. Paired, interleaved
deltas are the only figures here that carry a magnitude; single-arm levels are
reported as ranges.

### Verification

Seen before the tag; the dry run repeats all of it on the tagged tree.

- `tools/lint.sh` — exit 0, all eighteen gates clean.
- `swift test --no-parallel` — exit 0, 1,771 tests in 255 suites over seven
  target runs, every one reporting passed and none failed.
- `tools/golden-baseline.sh --check qwen38-4` — exit 0, `ok — output identical
  to baseline`. **1 of the 16 stored golden baselines compared byte-identical**,
  which is the only one this machine has an install for.
- **Not checked, because their install is not under `models/` and nothing
  may be fetched to change that**: `ornith-4`, `ornith-8`, `qwen36-4`,
  `qwen36-8`, `qwen38-8`, `agentworld-4`, `agentworld-8`, `katcoder-4`,
  `katcoder-8`, `qwen35-2b-4`, `qwen35-2b-8`, `qwen35-4b-4`, `qwen35-4b-8`,
  `qwen35-9b-4`, `qwen35-9b-8`.
- No internal-speed record was taken. It has been optional since 2026-10-01 and
  the numbers above already name their commit, section and pairing.
- Clean scratch release build with a clean warning scan; every staged
  executable asserted exactly `arm64`.

### Checksum

`tinytitan-5.20-macos-arm64.tar.gz` sha256: `SHA256_PENDING`
`tinytitan-5.20-macos-arm64.tar.gz` size: `ARCHIVE_BYTES_PENDING` bytes
`tinytitan-lib-5.20-macos-arm64.tar.gz` sha256: `LIBRARY_SHA256_PENDING`
`tinytitan-lib-5.20-macos-arm64.tar.gz` size: `LIBRARY_BYTES_PENDING` bytes
`tinytitan-5.20-tools.tar.gz` sha256: `TOOLS_SHA256_PENDING`
`tinytitan-5.20-tools.tar.gz` size: `TOOLS_BYTES_PENDING` bytes
