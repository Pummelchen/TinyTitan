# TinyTitan 6.1 — hardening, shared prefixes, and the first Kapitsa adoptions

This release is mostly durability. A second source audit closed 40 concrete
defects, and the first ideas taken from Kapitsa are the ones that improve the
product without changing its answers: a shared system prefix, a machine report,
and benchmark tooling that can reject a speed claim on quality.

### Hardening: 40 audit findings fixed

The second audit (`AUD2-*`) found and fixed real failures: caller cancellation
was not forwarded into one-shot generation; a stalled request body could stay in
flight; the PLE prefill gather could run past its block; checkpoint metadata was
not validated and corrupt records were not quarantined; a failed deletion could
stop charging its bytes; the two prompt stores did not share one disk budget;
calibration profiles were not bound to what they measured; rANS geometry and the
LZFSE final wave could publish early; a port was validated only after the model
loaded. The audit ledger is `docs/audit-ledger.json` (the old audit directory is
gone; the live ledger moved out of it so the docs gate can still resolve every
commit it cites).

### A new session reuses the system prefix

Interior checkpoints already made an edited turn cheap. They are now also shared
across sessions: the system-end snapshot is captured as `.sharedSystem`, keyed by
`SHA-256(domain + system token IDs)`, and restored by a new session with the same
system prompt, so only the user suffix is prefilled. Only the system/developer
prefix is shared; generic interior restore is gated on the conversation key, and
a mismatch, corrupt record or missing Qwen3.8 QSA/PLE payload falls back to a full
prefill. A hit logs `shared_system_hit tier=… prefix_tokens=… suffix_tokens=…`.
Verified on the real session orchestrator with a byte-identical output control.

### `tinytitan doctor`

One command reports the machine and what it can run: chip, RAM, macOS, disk,
GPU/Apple10/Metal4/ANE, the chosen tensor-ops path, installed models with a fit
estimate, the page-cache decision, calibration status, and other running model
processes. Human and stable-key `--json` output; it loads no model.

### Benchmark tooling that can say no

`benchmark/quality_floor.py` compares per-token logit captures against a measured
2x reference-vs-reference noise floor and rejects a candidate on any top-1
divergence. `benchmark/tinytitan_bench_compare.py` compares benchmark results only
when they share a workload and prints provenance from the files, never from the
comparison host. Both are registered in the model-free CI group.

### Tensor-op prefill for M4 and later, gated

The MPP tensor path that already covered prefill attention now also covers the
QKV/input projections and the dense/shared-expert FFN, selected through
`HardwareCapabilities` and `TINYTITAN_TENSOROPS=auto|off|on`. On M3 `auto` remains
scalar and the golden is byte-identical in `auto`, `off` and `on`; the M4/M5/M6
win is **unverified** on this host, and the 8-bit projection route is new.

### Rejected and reverted

Tree speculation is rejected on the cost model: 0.830x measured, 0.894x ceiling
at any tree width, and KFlash at its own eight rows projects to 0.243x. The
2-bit low-memory profile was implemented and then **reverted**: its throughput was
measured on an install requantized from 4-bit whose text was degraded. The rule is
now in `AGENTS.md`: a performance improvement may not be bought with model quality,
and a commit that does is rejected, experimental flag or not.

### DSH 0.2.1-rc.3 plugins

The TinyTitan 6.0 plugin line moves to DeepSeek Harness `0.2.1-rc.3`, and the
bundled private harness is removed. The plugin version probe now resolves the
`dsh` launcher symlink, so the bundle mounts on a normal install.

### Checks not run

Golden baselines are checked only against installs present under `models/`. On this
host the checked targets are `ornith-4`, `ornith-8`, `qwen38-4`, `qwen35-2b-4`,
`qwen35-2b-8`, `qwen35-4b-4`, `qwen35-4b-8`, `qwen35-9b-4` and `qwen35-9b-8`.
Baselines with no install were **not checked**: `qwen36-4`, `qwen36-8`,
`qwen38-8`, `agentworld-4`, `agentworld-8`, `katcoder-4` and `katcoder-8`.
No M4/M5/M6 hardware was available, so the tensor-op projections/FFN path is
**unverified**; no 2-bit install quality run exists because the profile was
reverted.

### Checksum

`tinytitan-6.1-macos-arm64.tar.gz` sha256: `SHA256_PENDING`
`tinytitan-6.1-macos-arm64.tar.gz` size: `ARCHIVE_BYTES_PENDING` bytes
`tinytitan-lib-6.1-macos-arm64.tar.gz` sha256: `LIBRARY_SHA256_PENDING`
`tinytitan-lib-6.1-macos-arm64.tar.gz` size: `LIBRARY_BYTES_PENDING` bytes
`tinytitan-6.1-tools.tar.gz` sha256: `TOOLS_SHA256_PENDING`
`tinytitan-6.1-tools.tar.gz` size: `TOOLS_BYTES_PENDING` bytes
