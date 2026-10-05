# Fact keeping over sessions: a plan

The goal: a long session — a novel, a repo — should keep working at the same
quality on chapter forty as on chapter one, without losing decisions, current
state or detail. Stated honestly, **zero forgetting is impossible**: the target
is *bounded, measured loss*, with a hard guarantee on the things that must never
drift — decisions, current state, and what the person said — and graceful loss of
everything else.

This plan works from what is already measured in this repository, not from
general practice. Read `docs/agent-memory.md` first; it documents the machine
this builds on.

## What the measurements say

`docs/master-benchmark-results.md`, ten multi-session worlds, memory arm against
a forced 200-word summary note:

| arm | carryable | foundation | stale | model cost |
| --- | ---: | ---: | ---: | ---: |
| summary | **294/431 = 68.2%** | 251/347 = 72.3% | **44** | **235.1 min** |
| memory | 256/431 = 59.4% | 234/347 = 67.4% | **87** | 399.6 min |

Pooled, memory loses on changing facts and carries **twice the stale facts**;
unweighted it wins +2.2 pp carryable and **+24.8 pp foundation**. And
`docs/memory-database-v2-comparison-2026-09-07.md` shows the same shape for the
book: the summary arm moves 90–96% carry-over, auto memory 95–96%, with v2's
consolidation costing **15% more prompt tokens** for carry-over equal within
noise. Three separate failure modes hide in those numbers.

## The three failure modes

1. **Missing facts (recall).** Ranking is lexical. On the authored recall set,
   token match scores **recall@1 1/4** against the side-engine hint's **4/4**
   (`docs/side-engine-tasks.md`) — the misses are "no term in common", which no
   amount of ranking polish fixes.
2. **Stale facts (time).** A later state coexists with, or loses to, an earlier
   one. Key reuse is supposed to make the newer value win, but the bootstrap's
   documented tie-break prefers the *older* fact "so a foundation outranks last
   session's state" — which is exactly backwards for `state/*`. Photograph is
   the canonical loss: memory 51.1% against the summary's 85.9%.
3. **Budget competition (what is in the window).** The bootstrap is one bounded
   set (config: 60 records / 16 KiB, values clipped to 200 chars) and with
   `TINYTITAN_MEMORY_TOOLS=off` — the default — whatever it omits is simply not
   there. Measured prefill on the 125B-A6B/24 GiB M3 is **651.8 s for 10k
   tokens** (`docs/qwen38-prefill-profile.md`), so the answer cannot be "inject
   more"; it has to be "inject the right tier".

The benchmark's deepest lesson: the arm that wins on changing facts is a
**forced, compact, freshly written note**, and memory's advantage is
**foundation**. The design should stop choosing between them.

## What is already right (build on it, do not rebuild)

- The engine **forces** consolidation: 30 s quiet or a rollover, tool-free, one
  generation per session, incremental (only turns since the last distillation),
  addressed facts, reusing a key so a later state supersedes the earlier one.
- The extraction is shown existing keys rather than all values, drops
  placeholders, and logs empty results.
- The **guard** stops a model-derived fact from silently superseding one the
  person asserted (`MemoryRecord.isUserAsserted`, `isDisputed`).
- The record already carries `importance`, `confidence`, `tags`,
  `sourceSession`, `createdAt`, `updatedAt`, `isDisputed`, `isGlobal`.
- The side engine supplies near-duplicate, disagreement and "could this answer
  it" judgements on otherwise-idle CPU.
- The semantic layer is already specified and the interface shaped for it
  (`docs/agent-memory.md`, "A future semantic layer").

## The program

Each step names the metric it must move. Nothing ships without moving one
without regressing another, and the benchmark's own noise floor (~6 points on
one run) means pooled or repeated runs only.

### 1. Classify keys, and let freshness win where it must

Give every key a **state class** — `state/*` (a value that changes), `rules/*`
and `decisions/*` (a value that is superseded deliberately), `characters/*` and
world facts (a value that is stable). Rank the bootstrap by class first, then
importance, then freshness: for `state/*` the **newest** value wins outright; for
foundation keys the current tie-break stays. This is a small, local change to
`MemoryRetrieval`/bootstrap selection and it attacks the largest measured hole.

*Gate:* pooled `stale` down from 87; carryable up on `photograph`, `pigeon`,
`contract`; bootstrap bytes unchanged.

### 2. Make supersession a record, not an overwrite (wire T4)

When a consolidation writes an existing key with a different value, store the
transition: `supersedes` (fingerprint of the old value), `validFrom`, and a short
`because` taken from the turn that changed it. Keep the address history that
already exists (32 versions). Today `isDisputed` flags a disagreement but nothing
distinguishes *deliberately superseded* from *never contradicted*; T4 is measured
**perfect when the rule is supplied and the rule source is unwired**
(`docs/side-engine-tasks.md`). This is that rule source.

*Gate:* `stale` again; plus a new mechanical counter — "reply used a superseded
value" — computed offline from stored answers.

### 3. Tier the prompt: a live-state digest above the bootstrap

Regenerate a bounded digest (~200–400 tokens) whenever a fact changes, and inject
it **every turn**: the objective, the live `state/*` values, the decisions in
force, and one line per recent supersession ("inn: standing → burned, session 4").
Below it, keep the existing bootstrap (keys + one-line summaries) and keep tools
off by default. This is the summary arm's winning property — compact and fresh —
expressed in the store's own vocabulary instead of prose.

*Gate:* carryable on changing-fact worlds, at a measured **tokens-per-turn** cost;
prefill is the budget, so the digest has a hard byte cap like the bootstrap's.

### 4. Catch contradictions where they are cheap

T6 (reply checker) is 100% on the served 35B at ~2.5 s per judgement and 62% on
the 4B (`docs/side-engine-tasks.md`, `docs/t6-reply-check-offline.md`). Do not run
it on every reply. Run it **only on replies that mention a key superseded in the
last N sessions** — few calls, high yield — and keep the expensive direction
(false positives, "claiming a contradiction") protected by the existing
"silence is not a contradiction" prompt rule. The guard stays structural for
user-asserted facts.

*Gate:* the existing 417 hand-audited cases in `benchmark/t6_prose_score.py`;
precision ≥ the 95% gate or it stays offline.

### 5. Add the semantic layer, embedded

Build the documented layer: embed on write, cosine rank behind the existing
prefix/tag/importance filters, lexical fallback when no embedding exists. Measure
it against the **T7 hint**, which is the current best (4/4) and costs CPU time,
not prompt tokens. An embedded index is the requirement — the store is 96–121 KB
for a novel, and `ContinuityStore.swift` records that an external store was
deliberately removed.

*Gate:* `benchmark/side_engine_recall.py` — recall@1 at 4/4 at lower cost than the
hint, or a strict improvement on a larger authored set.

### 6. Keep consolidation cost flat as the store grows

v2's extraction prompt grew with the key list and wrote more facts per session
(29–41 against 23) for equal carry-over. Show namespace totals and prefix
rollups instead of every key by name once a namespace passes a threshold, and
refuse a write whose value memory already holds (already done) — the goal is a
consolidation prompt that does not grow with the store.

*Gate:* consolidation prompt tokens per session, flat across a ten-session run,
at equal carry-over.

### 7. On the harness side: structured compaction and a recap-carrying hop

Two changes in `dsh-tinytitan`, both about not losing the thread when the window
fills:

- **Compaction note shape.** The server's compaction already produces a summary
  (`ServerCompaction.swift`); make the *plugin's* compaction preset ask for the
  same addressed shape memory uses — objective, decisions in force, live state,
  open questions — so the note is a digest a reader can act on, not prose.
- **The hop carries the recap.** A `fork` child inherits the parent's entire
  transcript and starts at the parent's size; the recorded fix is a `spawn` child
  with a bounded recap (clean objective + the live-state digest + "the workspace
  is the authority"). That is the single largest lever for long sessions: a fresh
  window that is *told* what matters instead of one that re-reads everything.

*Gate:* a handoff chain on a scratch profile with a long objective — the child's
first prompt must be the bounded recap, and the objective must be identical
across hops.

## Order

1 → 3 (both cheap and aimed at the measured loss), 2 (the mechanism that keeps
them honest), 6 (cost hygiene), 4 (targeted checking), 5 (the biggest build,
gated on 1–3 being in place), 7 (harness side, independent of the engine work).

## What not to do

- **Do not add a database.** `docs/clickhouse-long-session-verdict.md` has the
  measurements: the store is kilobytes, retrieval speed is not the constraint.
- **Do not inject more context.** Prefill dominates; the summary arm already
  proves a small fresh note beats a large stale one.
- **Do not write more facts.** v2 did, and paid 15% more for carry-over equal
  within noise.
- **Do not rely on the model to write memory.** Measured: the model made zero
  writes given the bible in its prompt, then invented one. Forcing is what works.

## Measurement

Minimum metric set per run, all already produced by the benchmark:
`carryable`, `foundation`, `stale`, `tokens per turn` (prompt), `model-busy`,
recall@1/@3 on the authored recall set, and — new — "reply used a superseded
value". Extend the suite to **repo coding** before claiming coding quality: the
current evidence is book, pong and multi-world prose. For code the facts are
decisions, interfaces, invariants and file ownership; retrieval of *code* should
stay with the agent's own grep/read loop until measured otherwise.

## Risks

- Freshness ranking could starve foundation facts on a store near the bootstrap
  cap — the byte cap has to be split, not shared, between the digest and the
  bootstrap.
- Supersession records grow the journal; the 32-version address bound and the
  retention settings already cap it.
- Semantic recall adds a resident encoder, which competes with a model already
  streaming experts from SSD — measure memory and busy-time, not just recall.
- The benchmark's noise floor is real: single-run wins under ~6 points are not
  evidence.
