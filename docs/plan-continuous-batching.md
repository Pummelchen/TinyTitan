# Continuous batching for the server (4 sessions at once)

Status: **design; no engine change yet**. Chosen over a session pool (option A/B
in the decision record) because the batching win is not only concurrency: one
forward pass that carries B tokens from B conversations amortises the routed-
expert reads that dominate decode, which a pool of independent runners cannot.

> **Status update: the width is opt-in, and its ceiling is the engine's, not 4.**
> The shipped default is 1: a plain launch serves one generation at a time, and
> `--max-concurrent-sequences` raises it. The argument is now **any power of two
> up to 256** — an agentic workload may want many, and what a machine can actually
> hold is decided per load by `BatchedMemoryBudget`, which clamps the width to the
> worst-case per-slot stores and logs what it built. `KVCacheManager.maximumSlots`
> (mirrored by `GDNStateManager`) was raised from 8 to 256 for that: it is the
> sanity bound on a typed argument, never the practical limit. The launcher offers
> 1 / 2 / 4 / 8 / 16 or a custom power of two and keeps its red warning above 1.
> The phases below landed with the default at 4 and a cap of 4 and are recorded as
> they landed; read those numbers as the values at the time.

Goal: `POST /v1/responses` and `POST /v1/chat/completions` accept up to four
generations at once through **one loaded model**, with the fifth and later
queued exactly as today (`--queue-limit`). Batch size 1 must stay
**byte-identical** — the golden baseline is the gate.

## Why this is a project and not an admission change

`ServerCoordinator.run` admits one operation at a time (`active: Bool`,
`ServerCoordinator.swift:47-71`). That is load-bearing, not incidental:

- `RealForwardRunner` is `@unchecked Sendable` whose safety argument is
  *exclusive ownership*: "two concurrent callers would corrupt them — the
  ownership is the whole safety argument" (`RealForwardRunner.swift:128-134`).
- `ServerModelSession` is an actor, but `generate` suspends at `await`s, so
  actor isolation alone does **not** serialise it; the coordinator's single
  active slot is what does.
- One runner means one `KVCacheManager` (single `position`, one K/V buffer per
  layer), one `GDNStateManager` (one recurrent state per linear layer), one
  `RawCompletionScratch`, one prompt cache, one MTP decoder.

So "allow 4" is not a wider gate; it is a batched forward path plus a
scheduler.

## What already exists that we can reuse

The prefill path already runs **T tokens through one forward pass**:

- `executePrefillChunk` / `runPrefillLayer` drive `t`-row projections
  (`encodeAffineProjection(..., tokenCount: t)`), so the quantised GEMM kernels
  for multi-row input exist.
- `encodeRoutedMoEPrefill` groups routed experts over T tokens
  (`RealForwardRunner+Prefill.swift:1268`), and `encodeDenseFFNPrefill`
  (`:1206`) covers the dense family.
- `encodeLinearAttentionPrefill` (`:736`) and `encodeFullAttentionPrefill`
  (`:880`) are chunk-aware and take `tokenCount`/`startPosition`.

Decode (`produceToken`, `RealForwardRunner+Decode.swift:64`) is the single-token
special case. A continuous-batch decode step of B sequences is structurally a
chunk of B rows — the work is making the chunk's rows **independent sequences**
instead of one contiguous prefix.

## Target architecture

**Batch slot.** A running sequence owns: its KV slot, its GDN state slot, its
token cursor, its sampler state, its structured-output decoder, its stop
matcher, and its watchdog counters. `B` of these are grouped per step.

1. **Slot-aware KV.** `KVCacheManager` gains `slots` (default 1). Each layer's
   buffer holds `slots` regions of `capacityTokens * stride`; slot `s` at
   logical position `p` lives at `(s * capacity + p % capacity) * stride`.
   `position` / `advance()` / `reserve(tokens:)` stay the slot-0 convenience
   path so every existing call site is unchanged at `slots == 1`. New
   `position(slot:)`, `advance(slot:by:)`, `reserve(tokens:slot:)`,
   `kSlot(layer:position:slot:)` and a slot-aware `KVView` (`offset` becomes the
   region base and is no longer always 0).
2. **Slot-aware GDN.** `GDNStateManager` gains `slots`; the delta-rule state and
   conv tail are `[slots, ...]` with per-slot base offsets. A chunk whose rows
   belong to different slots must **segment** the sequential scan — the current
   chunk kernel assumes one recurrence, so this is the kernel that needs the
   most care.
3. **Per-row attention.** A batch descriptor carries, per row: KV region base,
   valid length, ring capacity, and (for the linear path) GDN region base. The
   full-attention kernel reads `[0, len_row)` from that row's region and applies
   no cross-row causal mask; the linear kernel runs one segmented scan per slot.
   Rows of the same slot (a prefill chunk) keep today's behaviour.
4. **Per-row head and sampler.** The head writes B rows of logits; the sampler
   selects one token per row. Greedy/top-k/top-p state and RNG are per slot.
   `RawCompletionScratch` becomes per slot.
5. **Per-slot decoding.** `StructuredAssistantDecoder`, `StreamingStopMatcher`,
   the tool-call marker counter, and watchdogs move behind a per-slot
   accumulator. A slot that finishes is retired from the batch and its result
   published at once; new work is admitted into the freed slot (this is what
   "continuous" buys over static batching).
6. **Scheduler.** `ServerCoordinator` gains a width `maxConcurrentSequences`
   (default 4, argument `--max-concurrent-sequences`, validated 1...4 for now).
   Admission still bounds `admitted ≤ width + queueLimit`, so the queue
   semantics the operator knows are preserved; only the active set grows.
   `ManagedModelBackend` keeps one resident session, so residency, unload,
   reaper and idle timeout are unchanged.
7. **Per-slot prompt cache.** Publishing or restoring a KV prefix is only valid
   for the slot whose range it is; while a batch is running, cache publish is
   deferred to slot retirement, and a restore that would land in a busy slot
   waits. `activePromptCacheEntryID` becomes per slot.

## Invariants

- **Batch 1 is byte-identical.** At `B == 1` every code path must produce the
  same tokens, the same streaming events, and the same usage as today. The
  golden baseline (`tools/golden-baseline.sh --check ornith-8`) is run on the
  real model before and after each phase.
- **Loopback, single process, one model.** No second server, no second model
  load, one `MetalContext` and one resident session.
- **Peak KV is budgeted, not assumed.** Width is capped by measured headroom
  (`memory_pressure`), not by hope: `B × (KV + GDN + scratch)` must fit beside
  the weights and the wired expert cache. `--max-concurrent-sequences` is
  clamped down when the budget cannot take it, and the refusal is logged.
- **No new force-casts, no over-long functions** (`tools/lint.sh`), and every
  new type that is `@unchecked Sendable` carries its invariant note.

## Phases

| Phase | Deliverable | Files | Gate |
| --- | --- | --- | --- |
| 0 | this design | `docs/plan-continuous-batching.md` | — |
| 1 | slot-aware KV (**done**); GDN storage + encoder offsets (with Phase 2) | `Runtime/KVCache/*` | KV unit tests + golden |
| 2 | per-row full/linear attention, GEMM rows, GDN slot offsets | `Runtime/Inference/RealForwardRunner+*`, `Kernels/Attention/*`, `Kernels/GDN/*` | kernel parity tests (B=1 vs today; B>1 vs B sequential) |
| 3 | per-row head/sampler/decoders | `Runtime/Generation/*` | sampler parity + structured-output tests |
| 4 | width, admission, slot scheduling, streaming multiplex | `TinyTitanServer/Core/ServerCoordinator.swift`, `ServerModelSession+Generation.swift`, `HTTPServer*.swift`, `ServerArguments.swift` | server concurrency tests, no 429 below width+queue |
| 5 | memory-budget admission | `ServerModelSession+Loading.swift`, startup banner | budget unit tests |
| 6 | end-to-end exactness: B concurrent == B sequential | server tests + Open Responses suite | suite; golden |

## Progress

**Phase 1 (KV half) landed.** `KVCacheManager` takes `slots` (default 1, hard
cap 8). Each layer buffer holds `slots` contiguous regions one capacity apart;
`kSlot`/`vSlot`/`kRange`/`vRange`/`keyView`/`valueView`/`keyRangeView`/
`valueRangeView` take an optional `slot` (default 0), `position(slot:)` /
`advance(slot:by:)` / `reset(slot:)` / `rewind(slot:to:)` / `reserve(tokens:slot:)`
give each sequence its own cursor, and growth copies every slot's live rows to
its new region. At `slots == 1` every offset and buffer length is exactly what it
was before the change, and `oneSlotKeepsTheOriginalLayout` asserts that rather
than assuming it.

- `swift test --no-parallel --filter KVCache`: 25 tests, 0 failures.
- `swift test --no-parallel`: 1528 tests / 234 suites, 0 failures.
- `tools/lint.sh`: all four gates ok.
- `tools/golden-baseline.sh --check qwen35-2b-4`: output identical to baseline.

GDN slot storage and the encoder offsets (`encodeConvDecode`,
`encodeDeltaStepDecode` and the prefill variants all bind `tail`/`state` at
offset 0) move to Phase 2, where the per-row attention work already has to touch
those encoders. Splitting them keeps this increment at "no kernel signature
changed".

**Phase 2 (GDN half) landed.** `GDNStateManager` takes `slots` (default 1, cap
8). Each linear layer's delta-rule state and conv tail hold `slots` contiguous
regions, `stateOffset`/`convTailOffset` and `stateSlot`/`convTailSlot` name a
slot's region, and `reset(slot:)` zeroes one slot only. The six state-carrying
encoders (`encodeConvDecode`, `encodeConvPrefill`, `encodeConvTailUpdate`,
`encodeConvTailCheckpoint`, `encodeDeltaStepDecode`, `encodeDeltaStepPrefill`)
take an optional offset (default 0) and bind it instead of hard-coding 0. The
prefill path's disabled-checkpoint binding now reuses the state's own offset,
which would otherwise have read another slot.

- `swift test --no-parallel --filter GDN`: 12 tests, 0 failures, including the
  new `decodeStepHonoursSlotOffsets`: one decode step run in slot 1 of a
  two-slot state/tail buffer produces exactly the output and state of a lone
  slot, and leaves slot 0 untouched. That is the test that would catch a kernel
  silently writing offset 0 — the failure mode where two sequences share one
  recurrence and both answers are quietly wrong.
- `swift test --no-parallel`: 1532 tests / 235 suites, 0 failures.
- `tools/lint.sh`: all four gates ok.
- `tools/golden-baseline.sh --check katcoder-4`: output identical to baseline
  (the target that exercises linear attention and routed experts).

The snapshot and MTP-checkpoint paths remain slot 0 by construction (they cover
a single sequence's prefix and the speculative draft), which is correct until
the runner becomes slot-aware in Phase 4.

**Recon for the attention half.** The decode full-attention call already binds
`keyView.offset` and reads `keyView` as its `kvFormat`
(`RealForwardRunner+Decode.swift:951-969`), so the slot region base added in Phase 1
is honoured by `encodeFull` without a kernel change. The SWA branch binds the
same offsets (`:976-989`); Qwen3.5/3.6/3.8 carry only full and linear
layers, so no shipped family reaches it. The remaining decode work is therefore
structural — a batched step that carries B rows through the token-wise stages
(the chunk kernels already take `tokenCount`) and loops attention and the GDN
step per row — rather than a new attention kernel.

**Phase 2 (slot-aware decode) landed.** `RealForwardRunner` takes `slots`
(default 1) and builds its KV and GDN stores at that width. `produce(_:slot:)`
and the internal `produceToken`, `encodeDecodeAttention`,
`encodeLinearAttentionDecode` and `encodeGatedFullAttentionDecode` carry the slot
through to every KV region, GDN state/tail offset and `encodeQuantizedKV`
destination; the SWA decode branch's hard-coded `kOffset: 0` became
`keyView.offset`. Slot 0 is the default everywhere, so the single-sequence path
is untouched. `produceBatch(rows:logits:)` is the batched entry point; it still
runs each row's token-wise stages in turn (the fusion is the next step), so it
is correct before it is fast.

- `swift test --no-parallel --filter QwenRunnerTests`: 13 tests, 0 failures,
  including `batchedDecodeKeepsSlotsIndependent`: sequences `[11, 7, 5]` and
  `[3, 9, 4]` interleaved token by token through one two-slot runner reproduce,
  to 1e-3, the logits each produces alone in a one-slot runner. This runs the
  real hybrid graph (full attention + GDN) end to end, so a dropped slot offset
  in KV, GDN state or conv tail shows up as divergence.
- `swift test --no-parallel`: 1533 tests / 235 suites, 0 failures.
- `tools/lint.sh`: all four gates ok.
- `tools/golden-baseline.sh --check qwen35-2b-4` and `--check katcoder-4`:
  output identical to baseline.

What is **not** yet batched: the per-layer token-wise stages (projections, FFN,
routed MoE) still run once per row, so `B > 1` gives four sequences progressing
through one runner and one KV, not four rows in one forward pass. That fusion —
carrying B rows through the chunk kernels that already take `tokenCount` — is
the remaining Phase 2 work, and it is where the expert-read amortisation comes
from. The exactness test above is the harness that will gate it.

**Phase 3 (generation loop) started.** `LogitProducer` gained a required
slot-aware `produce(token:position:slot:into:)` (a requirement, not just an
extension method: an extension method on an existential dispatches statically and
would have silently run every slot through slot 0). `runRawCompletion` takes
`slot:` and passes it to both the decode step and the sequential prefill. Chunked
prefill stays slot-0-only — it writes slot 0's KV region — so a sequence in
another slot prefills through the slot-aware decode step; slot 0 keeps the
chunked fast path unchanged. The MTP path rejects `slot != 0` rather than
drafting into the wrong sequence.

`ForwardStepGate` (an actor) now serializes `produce` and `prefillChunked`: the
slots are independent, but the runner's scratch, cursors and counters are not,
so two sequences must never be inside a step at once.

- New tests: `nonZeroSlotPrefillsThroughTheSlotAwareDecodeStep` (slot 2 reaches
  the producer, no chunked prefill), `slotZeroKeepsTheChunkedPrefillPath` (slot 0
  unchanged), and `concurrentSlotsSerializeThroughTheStepGate` (two sequences
  driven by concurrent tasks still match their solo logits).
- `swift test --no-parallel`: 1536 tests / 235 suites, 0 failures.
- `tools/lint.sh`: all four gates ok.
- `tools/golden-baseline.sh --check katcoder-4` and `--check qwen35-2b-4`:
  output identical to baseline.

Remaining for the objective: session-level per-slot scratch and slot assignment,
the `ServerCoordinator` width, admission/queueing, and the memory-budget rule.

**Phase 4 (server admission and slots) landed.** `ServerCoordinator` takes a
`width`: up to `width` operations run at once, `queueLimit` more queue behind
them, and the rest are shed with 429. `width == 1` reproduces the original
`queueLimit + 1` bound exactly. `ServerModelSession` builds its runner with
`slots` equal to that width, holds one `RawCompletionScratch` per slot, and
hands each generation a slot from a pool (with a waiter queue as a safety net);
`runRawCompletion` receives the slot, so each sequence prefills and decodes into
its own KV/GDN region. A failed generation resets only its own slot
(`RealForwardRunner.reset(slot:)`) instead of wiping the other live sequences.

Consequences chosen deliberately for the first cut:

- **The prompt cache is off when `slots > 1`** (`effectivePromptCacheMode`).
  Its snapshot, restore and `activePromptCacheEntryID` are session-wide, and a
  prefix restored into the wrong slot is plausible wrong output rather than an
  error; batching re-prefills until the cache is slot-keyed.
- **MTP caps the width to 1.** Its draft stream is one sequence; the server
  clamps and `runRawCompletion` rejects `slot != 0` rather than mis-drafting.

New `--max-concurrent-sequences <1...4>`, default 4.

- `swift test --no-parallel`: 1540 tests / 235 suites, 0 failures, including
  `widthRunsSeveralAtOnceAndQueuesTheRest` (coordinator),
  `fourGenerationsRunAtOnceAndTheSixthIsShed` (HTTP: four running, one queued,
  the sixth 429, peak concurrency exactly four), the argument-bound and
  prompt-cache-mode tests, plus the round-4 engine tests.
- `tools/lint.sh`: all four gates ok.
- `tools/golden-baseline.sh --check katcoder-4` and `--check qwen35-2b-4`:
  output identical to baseline.

Still open: the token-wise stages are not fused across rows (Phase 2c), and the
memory-budget admission rule (Phase 5).

**Real-model end-to-end.** Against `qwen3.5-2b_4-Bit` on the GPU with
`--max-concurrent-sequences 4` (queue limit 4), nine concurrent
`/v1/chat/completions` clients returned **8×200 and 1×429**, and the server log
shows four `generating` at once, the ninth `status=429 error=queueFull`, then
the queued requests starting as slots freed. All eight completions carried real
content. The prompt cache reported itself off, as the batching rule requires.

**Phase 5 (memory budget) landed.** `KVCacheManager.worstCaseBytes` and
`GDNStateManager.worstCaseBytes` compute the per-slot ceiling from the
architecture without allocating, and `BatchedMemoryBudget` turns that into a
width: the slots must fit in half of physical memory less the wired routed-expert
cache, and the width is clamped down when they do not. `ServerModelSession.load`
applies it per load (so a catalog switch re-evaluates it against the model being
loaded) and logs the clamp; the single-sequence path is always available, so the
floor is one.

The worst-case formulas are checked against what the managers actually allocate
(`kvWorstCaseMatchesTheAllocatedLayout`, `gdnWorstCaseMatchesTheAllocatedLayout`)
rather than against a second copy of the arithmetic, so a storage-format change
cannot leave the budget quietly wrong. The test also caught a real ambiguity:
a zero budget means "no headroom, one at a time", not "unknown size", and the
two are now distinguished.

On the installed 2B at the advertised 262144 context the clamp does not fire —
re-running the nine-client check after adding it still gave 8×200 and 1×429 with
no clamp warning — which is the point: it guards the models that do not fit,
not the common case.

- `swift test --no-parallel`: 1545 tests / 236 suites, 0 failures.
- `tools/lint.sh`: all four gates ok.
- `tools/golden-baseline.sh --check katcoder-4` and `--check qwen35-2b-4`:
  output identical to baseline.

Still open: the token-wise stages are not fused across rows (Phase 2c), which is
the throughput win rather than a correctness or safety gap.

## Multi-slot output corruption: found and fixed

Width > 1 produced degenerate output on the real models while width 1 was
correct. Four bugs, found in this order, all now fixed:

1. **The catalog loader dropped `slots`**, so `--models-dir` sessions ran one
   sequence while the coordinator admitted four (`dbbffff`).
2. **The KV budget charged dense models for an expert cache** they never
   allocate, clamping a width that fits (`dbbffff`).
3. **Slots 1... fell back to sequential `.off` prefill.** Chunked prefill was
   slot-0-only, so the other slots fed their prompt through the decode step,
   which does not match the chunked path numerically — on 4-bit it diverges
   badly ("The capital of France is" → `oriarianum.org …`). Chunked prefill is
   now slot-aware: `prefillChunked(slot:)`, `executePrefillChunk(slot:)`,
   `runPrefillLayer(slot:)`, `copyPrefillKVToCache(slot:)`, the GDN slot
   offsets, and `encodeFullAttentionPrefill(slot:)` all carry the slot, and
   `runRawCompletion` uses the chunked path for every slot.
4. **The prefill encoders were called without the slot** in `runPrefillLayer`,
   so `encodeFullAttentionPrefill` and `encodeLinearAttentionPrefill` defaulted
   to 0 and the prefill attention read slot 0's KV for every slot. Also
   `prefillAttention.encodeCausal` was never passed `kOffset`/`vOffset`, so it
   read the KV buffers at their base.

Two concurrency defects surfaced once slots were live:

- `produce` checked the runner-wide `prefillChunkState` **before** taking the
  step gate, so it saw another slot's in-flight prefill and returned 500
  (`chunkedRunnerDirty`). The check moved inside the gate.
- `runRawCompletion`'s `start == .reset` called `producer.reset()`, which wipes
  **every** slot and the runner's transient state, outside the gate. Replaced
  by a gated, slot-scoped `resetSequence(slot:)`.

Verified on `qwen3.5-2B_4Bit` at width 4: a single request that lands in slot 3
and four concurrent distinct prompts all produce coherent text, and
`chunkedPrefillFillsTheRequestedSlot` checks every slot 0...3 against a
single-slot reference at 4-bit and 8-bit.

## Risks and open questions

- **GDN segmented scan** is the sharp edge: the recurrent state makes rows
  order-dependent, so unlike attention they cannot simply read their own
  region in parallel. If segmentation proves too slow, the fallback is to run
  linear layers per slot in a loop while the full-attention/MoE layers batch —
  still a win for the MoE-heavy models.
- **MTP** drafts several tokens for one sequence; a batched step must either
  disable MTP while `B > 1` or verify drafts per slot. Disabling it for the
  batched path (and keeping it for `B == 1`) is the conservative first cut.
- **Routed-expert streaming** at `B > 1` reads a union of experts per layer;
  the slot cache must not thrash between slots' expert sets. This is why the
  budget rule has to be decided with the transport, per the 5.5 notes.
- **Prompt cache** is the feature most at risk of silent misuse: a prefix
  restored into the wrong slot yields plausible wrong output. Phase 1 must make
  the slot part of the cache key.

## Rejected alternative: session pool (options A/B)

Four independent `ServerModelSession`s, or one `Model` with four runners, would
reach four-way concurrency with no kernel work. It was rejected because the
routed-expert cache is per `Model` and wired, so four sessions quadruple wired
memory and split SSD bandwidth — on a 24 GB machine the 35B/125B streaming
installs cannot take it, and even the dense 2B pays four prompt caches. It also
does not amortise expert reads, which is the reason to batch at all. Recorded
here because the memory numbers are the fallback if a batching phase stalls.
