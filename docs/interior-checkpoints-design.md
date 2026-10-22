# Interior checkpoints: granularity and payload (T20)

Status: T20 design; T21 format and store; T22 session capture and edited-turn
restore; T23 correctness matrix implemented. The content-addressed store
(`PromptCheckpointTypes.swift`, `PromptCheckpointStore.swift`) saves and
restores an `InferenceStateSnapshot` keyed by domain + prefix tokens, with
position/hash/size validation, memory/disk LRU and SHA-verified loads. A real
runner round-trip is pinned by `InteriorCheckpointRestoreTests`; the split
prefill at a real boundary and the edited-turn resume are pinned by
`EditedTurnCheckpointTests`, and the boundary planner by
`PromptCheckpointPlannerTests`.

T24's placement benchmark is the only part not closed here; see
`docs/maya-task-status.md` for its measured result.

## The gap this closes

The prompt cache stores one snapshot per **completed assistant turn**: the KV
and Gated-DeltaNet state at the end of the turn, the token prefix it covers and
the assistant message. A new request is matched against those turn endpoints;
a hit restores the snapshot and replays only the uncommitted boundary token.

That is the right shape for an appended turn, and the wrong shape for an
**edited last turn**. Editing the last user message invalidates the assistant
turn that followed it, so the cache has no prefix that covers the new turn
start: the engine re-prefills the whole prompt (system message, tools and all
earlier turns) even though everything before the edited turn is unchanged.

An interior checkpoint is a snapshot at a boundary *inside* the prompt that
survives the edit: at minimum the end of the system prompt (+ tool schema),
and the start of the last user turn.

## Points, not a grid

Proposed points per request:

| Point | When | Survives | Cost |
| --- | --- | --- | --- |
| `systemEnd` | after the system message and tool schema | every later turn | small: system + tools |
| `lastTurnStart` | at the first token of the latest user turn | an edit to that turn | close to one full snapshot of the history before it |
| `turnEnd` | existing completed-turn snapshot | appends | one full snapshot of the whole prompt |

There is **no per-16K (or per-N-token) default**. A fixed grid pays for
snapshots at positions no request can resume from: a turn boundary is the only
place a request can diverge, and every *completed* turn already leaves a
`turnEnd`. A grid would multiply snapshot bytes and write traffic for edits
that land mid-prompt, which the tokenizer/request model does not produce: a
chat request is a list of messages, so its possible prefixes are message
boundaries.

An operator may request an interior interval for a workload that edits inside
one long message (an agent rewriting a tool result, for example), but it is
opt-in and the default is the coarse set above.

## Payload

`InferenceStateSnapshot` already carries exactly what a prefix needs:
`position`, per-layer KV segment lengths, GDN segment lengths and a payload of
the K and V segments followed by the GDN recurrent state. It is versioned and
validated against the live runtime layout on restore.

An interior checkpoint adds only:

- the **prefix token IDs** the snapshot covers (the state's logical position
  must equal their count);
- the **domain hash** the existing cache uses (model ID, source snapshot hash,
  runtime profile hash, max context, KV storage, template hash);
- a **content hash** over the token IDs plus the domain, used as the lookup
  key so two different prefixes can never collide;
- a `kind` (`systemEnd`, `lastTurnStart`, `turnEnd`) and a `conversationKey`
  (the stable hash of the conversation's earlier messages) for eviction
  grouping.

No logits are stored. A prefix checkpoint resumes by prefilling the remaining
tokens; logits are only meaningful for the token that produced them, and the
existing turn-end entry keeps its one uncommitted boundary token for that
purpose.

## Space at 100K

For the 35B-A3B family the full-attention KV is 10 layers × 2 (K,V) ×
(2 heads × 256) elements; at int8 that is 544 bytes per segment-token, so
**~10.9 KB per token**, or ~1.09 GB for a 100K prefix. FP16 doubles it.

| Checkpoint set | Tokens captured | Bytes (int8) |
| --- | ---: | ---: |
| `systemEnd` (~2K) + `lastTurnStart` (~95K) | ~97K | ~1.06 GB |
| the above + `turnEnd` (100K) | ~197K | ~2.15 GB |
| per-16K grid (16K…100K) | 336K | ~3.65 GB |

So the coarse set costs about what the endpoint already costs, while a 16K
grid costs 3.4× more writes and disk for no additional resumable boundary on
message-shaped requests. With the shipped 8 GiB disk budget a 16K grid fits one
100K conversation with little room left; the coarse set leaves room for a
second conversation or a longer one.

## Storage and eviction

Reuse `ServerPromptStateStore` (memory limit plus optional disk directory,
bounded payload reads, SHA-256 verification). A checkpoint is stored under its
content hash rather than the entry UUID, so the same prefix can be referenced
by several entries without a second copy.

Eviction, per conversation:

1. keep `systemEnd` until the conversation is evicted (it is the smallest and
   the most reused);
2. keep the latest `lastTurnStart` and the latest `turnEnd`;
3. drop older interior points first, by kind, then by age.

The store's global memory and disk limits still bound the total; the
per-conversation rule only orders the eviction within them. A checkpoint whose
payload fails verification is deleted and never retried.

## Restore

For a new request:

1. tokenize and compute the prefix hashes at the candidate points;
2. look up the longest prefix whose content hash matches (the domain must
   match exactly; a model/config/template change invalidates every point);
3. load the snapshot, validate position and payload against the live layout,
   restore KV/GDN, set the cursor to the prefix length;
4. prefill from the checkpoint to the end of the prompt;
5. if no checkpoint matches, full prefill as today.

The edited-turn case is step 2 finding `lastTurnStart` when the `turnEnd` no
longer matches; the append case keeps hitting `turnEnd` and never pays for the
interior lookup.

## Correctness gates (for T23)

- A checkpoint is never restored when the domain hash differs (stale model,
  different KV precision, different template or runtime profile).
- Content hash is over token IDs plus domain; a one-token difference is a
  miss, never a wrong prefix.
- Position in the snapshot must equal the prefix length; a mismatch is a
  corrupt record, not a guess.
- Corrupt or truncated payload is deleted and the request falls back to full
  prefill.
- Concurrency: a checkpoint is captured from a completed command buffer, and
  an entry referenced by an in-flight restore is not evicted until the restore
  releases it.

## Non-goals

- No per-16K default and no mid-generation checkpoints.
- No logits payload for interior points.
- No change to the turn-end entry format; interior points are additive.
- No cross-machine reuse: the domain hash includes the runtime profile and
  KV storage, and the payload is device/runtime-layout specific.

## Implementation notes (T22)

- **Boundary detection is on the rendered ids.** The ChatML template renders
  the last assistant turn differently depending on whether a user turn follows
  it, so encoding `messages.dropLast()` is not always a prefix of the full
  render. `PromptCheckpointPlanner` locates `<|im_start|>` in the ids that were
  actually rendered instead: the second message start is `systemEnd`, the last
  message start before the generation prompt is `lastTurnStart`.
- **Capture splits prefill at the boundary.** `runRawCompletion` takes
  `checkpointPositions` and `onPrefillCheckpoint`; it prefills a segment, calls
  the hook with the KV cursor exactly at the boundary, then continues. The
  callback copies the snapshot and spawns the store save, so a multi-gigabyte
  write does not stall decode. A failed capture logs and is ignored: the next
  request can always full-prefill.
- **Restore is tried only after a turn-cache miss.** An append still uses the
  longer `turnEnd` entry. On a miss, candidates are tried longest first; a
  content-hash miss is a miss, a corrupt record is deleted, and the next
  candidate (or full prefill) follows. A QSA/PLE install refuses restore before
  reading the payload, because those subsystems are not in the snapshot.
- **Switch.** `TINYTITAN_PROMPT_CHECKPOINTS=0` turns the tier off for A/B.
  `TINYTITAN_CHECKPOINT_GRID=<tokens>` adds an opt-in grid (default off), and
  `TINYTITAN_CHECKPOINT_GRID_ONLY=1` removes the message boundaries so a
  benchmark can price the grid on its own. Both are used only by T24; the
  shipped default remains the two message boundaries.
