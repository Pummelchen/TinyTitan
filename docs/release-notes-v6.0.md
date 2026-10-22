## TinyTitan 6.0 — checkpoints, calibration, and a machine-size cache policy

This release closes the Project Maya transfer and is the first to persist
conversation state that is not just the completed turn. It is a major version
because the cache policy changes by machine size, the checkpoint format gains a
Qwen3.8 auxiliary payload, and `tinytitan calibrate` is a new command surface.

### An edited last turn no longer re-prefills the conversation

The prompt cache used to have one snapshot per completed assistant turn. Editing
the last user turn invalidated it, so the whole conversation was prefilled again.

The engine now captures interior checkpoints at the end of the system message
and at the start of the last user turn, keyed by a hash of the exact prefix. On
an edited turn it restores the longest matching checkpoint and prefills only the
suffix. A corrupt or mismatched record is deleted and the request falls back to
a full prefill.

- Ornith 1.5 4-bit, 24,551-token conversation, edited turn: **1,119.3 s →
  3.07 s prefill (365×)**, byte-identical text.
- Qwen3.8-Flash-Next 4-bit, 1,407-token conversation: **44.38 s → 2.76 s
  (16.1×)**. QSA pooled blocks/raw tail and the PLE convolution window travel in
  an auxiliary payload beside the snapshot.
- The turn-end cache carries the same Qwen3.8 payload, so an append after a
  server restart restores from SSD: **14.33 s → 6.22 s prefill (2.30×)**, with
  identical text.
- Capture is not free: the first request pays ~6% (Ornith) or ~33% (Qwen3.8) for
  the extra snapshot work; the break-even is the first edit.

### Qwen3.8-Flash-Next 4-bit gets its own tuning

The 4-bit profile now uses a **2,048-token prefill chunk** instead of 4,096.
The shorter reservation is held through decode and measured 5–16% faster decode
at the same prefill on 200- and 851-token prompts; the 8-bit row keeps 4,096 for
long-prefill use, and `--prefill-chunk 4096` remains the override.

Qwen3.8 also refuses more than one concurrent sequence: its QSA indexer and PLE
window are per-sequence, not slot-keyed, so a second slot produced different
text. The server now queues instead of emitting the wrong answer and says so in
its log.

### Calibration is a product command

`tinytitan calibrate` writes and validates a per-machine profile, measures
interleaved arms with a 3% floor and a measured noise band, and applies the
result at the next launch. Explicit flags always win; a profile from another
machine, context or model is refused. The first Qwen3.8 run kept no knobs
(chunk-128 +4.0% inside a 4.7% band; slots-96 −3.9%).

### Page cache follows the machine

Routed-expert reads stay bounded on a host that cannot hold the install, and
use the page cache only when RAM is at least **2× the install size**. The rule
fails closed when memory or install size is unreadable, and
`TINYTITAN_BOUNDED_IO=0|1` still overrides it. On the 24 GB development host
every install selects bounded, so the shipped behaviour there is unchanged.

### Also in this release

- The Ornith MTP sidecar was installed from the pinned Hugging Face source and
  benchmarked: MTP at 64/128 tokens measured **−5.1% / −12.0%** aggregate with
  identical text, so the controller remains off.
- Machine-size and checkpoint policy are covered by new tests: policy
  precedence, prefix/domain/store limits, corruption, concurrency, QSA auxiliary
  round-trip, and turn-store auxiliary persistence.
- The T10/T34 fused loop remains opt-in and unwired; it is rejected on wall
  clock and, on Qwen3.8, on correctness.
- `docs/qwen38-4bit-tuning-2026-10-22.md`,
  `docs/qwen38-checkpoint-auxiliary.md`,
  `docs/page-cache-policy.md` and
  `docs/speculation-cost-model-verdict.md` carry the measurements.

### Checks not run

Golden baselines are checked only against installs present under `models/`.
On the cut host the checked targets are **ornith-4**, **ornith-8** and
**qwen38-4**. Every other stored baseline has no install and was **not checked**:
`qwen36-4`, `qwen36-8`, `qwen38-8`, `agentworld-4`, `agentworld-8`,
`katcoder-4`, `katcoder-8`, `qwen35-2b-4`, `qwen35-2b-8`, `qwen35-4b-4`,
`qwen35-4b-8`, `qwen35-9b-4` and `qwen35-9b-8`.

The host-KV parity/timing tasks at 128K/1M (T11–T14) and the multi-Mac and
64/128 GB page-cache tasks (T19/T27) are not runnable on this 24 GB Mac; their
ledger entries stay blocked rather than passed. The literal 100K checkpoint read
(T24) is recorded at 24.5K because the full prefill exceeded the request timeout.

### Performance

Measured on this 24 GB M3, on the commits in this release:

| workload | before | 6.0 |
| --- | ---: | ---: |
| Ornith 4-bit, 24.5K edited turn | 1,119.3 s prefill | **3.07 s (365×)** |
| Qwen3.8 4-bit, 1.4K edited turn | 44.38 s prefill | **2.76 s (16.1×)** |
| Qwen3.8 4-bit, turn cache after restart | 14.33 s prefill | **6.22 s (2.30×)** |
| Qwen3.8 4-bit, 851-token chat decode | 4.77 / 4.65 tok/s | **5.51 / 4.90** |
| Ornith 4-bit, width-2 aggregate | 9.02 tok/s (64) | **9.45** |

The checkpoint numbers are prefill only; the capture cost and its break-even are
stated above. Qwen3.8 batching, 4-bit KV, MTP and the fused loop were measured
and are not enabled: their losses are recorded in the tuning and speculation
documents rather than hidden in a default.

### Checksum

`tinytitan-6.0-macos-arm64.tar.gz` sha256: `SHA256_PENDING`
`tinytitan-6.0-macos-arm64.tar.gz` size: `ARCHIVE_BYTES_PENDING` bytes
`tinytitan-lib-6.0-macos-arm64.tar.gz` sha256: `LIBRARY_SHA256_PENDING`
`tinytitan-lib-6.0-macos-arm64.tar.gz` size: `LIBRARY_BYTES_PENDING` bytes
`tinytitan-6.0-tools.tar.gz` sha256: `TOOLS_SHA256_PENDING`
`tinytitan-6.0-tools.tar.gz` size: `TOOLS_BYTES_PENDING` bytes
