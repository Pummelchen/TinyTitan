# ClickHouse for long sessions: a verdict

The question: would putting ClickHouse behind either the TinyTitan engine's
memory or the `dsh-tinytitan` plugin improve prompt→reply quality over long
sessions — a hundred-chapter novel, a long repo-coding run?

**Verdict: no, in both serving paths.** Long-session quality here is limited by
which facts survive, how fresh they are, and how many tokens can be prefilled —
none of which a columnar query engine decides. ClickHouse has a real place in
this project, but it is the *analysis plane* (benchmark and telemetry), not the
prompt path.

## What the store actually holds

From `docs/agent-memory.md` ("Store size"), measured by replaying the benchmark
journals with the store's own accounting:

| task | facts held | resident (facts, history, log) | journal on disk |
| --- | ---: | ---: | ---: |
| hundred-chapter novel, ten sessions | 93–134 | **96–121 KB** | 194–249 KB |
| pong in three languages, three sessions | 10–22 | **10–20 KB** | 25–45 KB |

ClickHouse's value curve starts where scans dominate — billions of rows, terabytes
on disk. The whole working set here is a rounding error next to one KV-cache
block, it already lives in RAM, and a query is a bounded scan of at most 2,000
records inside the engine (`MemoryRetrieval.swift`, `MemoryStore.swift`). There
is no scan cost to remove.

## What the measurements say quality is limited by

**Selection and staleness, not speed.** `docs/master-benchmark-results.md`, ten
multi-session worlds, three runs on the starred ones, memory arm against the
200-word-note summary arm:

| arm | carryable | foundation | stale | model cost |
| --- | ---: | ---: | ---: | ---: |
| summary | **294/431 = 68.2%** | 251/347 = 72.3% | **44** | **235.1 min** |
| memory | 256/431 = 59.4% | 234/347 = 67.4% | **87** | 399.6 min |

Pooled, memory is behind on changing facts and roughly twice as stale, for ~1.7×
the model time; unweighted it leads by +2.2 pp carryable and +24.8 pp foundation.
A faster store does not turn a stale hit into a fresh one, and it has no opinion
about which of two conflicting facts is current.

**The one real retrieval defect is semantic, and its fix is a model.** On the
authored recall set (`docs/side-engine-tasks.md`, `benchmark/side_engine_recall.py`),
token-match figures re-measured with `--baseline` on 2026-10-10 — the token ranking runs
no model, so its row is one figure shown for both installs):

| ranking | 4B recall@1 | 4B recall@3 | 9B recall@1 | 9B recall@3 |
| --- | ---: | ---: | ---: | ---: |
| token match | 3/4 | 3/4 | 3/4 | 3/4 |
| side-engine hint | **4/4** | **4/4** | **4/4** | **4/4** |

The lexical miss is the "no term in common" case. What fixes it is an
embedding or a model judgement about relevance — a representation and compute
problem. `docs/agent-memory.md` ("A future semantic layer") already prices this
correctly: *an embedding model resident alongside the LLM*, worth measuring
against the lexical ranking, with the interface (`MemoryQuery` / `MemoryRanking`)
shaped so the model-facing tools do not change. Storage engine choice is
orthogonal to it.

**On this hardware, injected context tokens are the cost.** Measured prefill,
Qwen3.8-Flash-Next 4-bit on the 24 GiB M3 (`docs/qwen38-prefill-profile.md`):

| prompt tokens | prefill |
| ---: | ---: |
| 320 | 19.4 s |
| 1,244 | 47.9 s |
| 5,006 | 298.1 s |
| 10,022 | **651.8 s** |

Eleven minutes to first token on a 10k prompt. Every kilobyte of retrieved
context costs seconds; a retrieval hop that saves 100 ms is invisible. The lever
is *how few, how fresh, how right* the injected facts are — and ClickHouse, by
making more context easy to fetch, pushes the wrong way.

## Why the engine should not take it

`ContinuityStore.swift` opens with the decision this would reverse:

> This replaced a Valkey client. Nothing above it changed: the server still talks
> to `MemoryStore` … What went away is a second process, a wire protocol, a
> connection to lose and a cache to size. The store now lives in the same binary
> as the model that reads it.

The documented invariants are the opposite of a shared OLAP server: RAM-primary
with a journal file, one writer per workspace enforced by an exclusive file lock,
and a README that sells "the library and the engine" as the two products.
ClickHouse would re-introduce the second process and the wire protocol for a
100 KB store, and add the third storage story after Valkey and Continuity.

## Why the plugin should not take it

`dsh-tinytitan` owns no data to put in it. The harness owns sessions (compressed
JSONL on disk) and memory belongs to a different plugin (markdown); the
`dsh-tinytitan` surface is route, compaction preset, `autoGoal`, `autonomy` and
`handoff`. Its contract is zero-config — one npm package, no daemon, no port —
and a refusal gate that keeps an unsupported harness from writing anything
(`src/index.js`). A database in the boot path of every profile buys no reply
quality: whatever retrieval it served would still have to be selected, ranked
and injected by something else.

## Where ClickHouse does earn its place

The analysis plane. The master benchmark is currently recomputed from stored
answers by Python; every turn, judgement, score, staleness flag and cost lands
per run across worlds, models and versions. That is exactly a columnar workload —
"did this change help, on which worlds, at what cost" — and it keeps history
instead of recomputing it. `clickhouse-local` (a single binary, no daemon) is the
low-risk form: it never enters the product, only the tooling.

The same holds if the raw transcript corpus ever needs analytic queries —
millions of turns across repos and years, full-text and ANN over the event
stream. Feed it *from* the session logs; never gate inference on it.

## What would change the verdict

1. **Scale.** Records reaching millions rather than hundreds — e.g. every turn of
   every session embedded and searchable, not just facts.
2. **Sharing.** Several servers needing one memory. Today that is deliberately
   refused (`docs/agent-memory.md`, "One writer per workspace").
3. **Vector volume.** An ANN index over 10^8+ vectors where a dedicated index
   cannot be embedded, and query latency dominates.
4. **Analytic queries over raw transcripts**, as a requirement rather than a
   convenience.

None of those is true today, and (1)–(3) are the ones a semantic layer would have
to reach before an external store is worth its process.

## The cheaper path, with gates

1. **Measure the hole first.** Extend the master benchmark's `stale` count and the
   recall@1/@3 harness to the book and coding worlds, so any change is judged on
   the two numbers that matter. Gate: no change ships without moving recall or
   staleness.
2. **Build the semantic layer embedded**, exactly as documented: embed on write,
   cosine rank behind the existing prefix/tag/importance filters, lexical
   fallback. Compare against the T7 hint on the same recall set before wiring it.
   Gate: recall@1 at 4/4 at lower cost than the hint, or a strict improvement.
3. **Attack staleness and consolidation cost** — 87 vs 44 stale and +15%
   consolidation prompt for carry-over equal within noise are the largest measured
   *fixable* quality drags, and no store fixes them.
4. **For long repo coding**, measure repo-level retrieval against the agent's own
   grep/read loop on the existing repo tasks before adding any index at all.

## Reproduce

```bash
sed -n '/## Store size/,/## One writer/p' docs/agent-memory.md
sed -n '/## Results/,/## What the benchmark says/p' docs/master-benchmark-results.md
sed -n '/T7 — retrieval/,/## Where it is wired/p' docs/side-engine-tasks.md
sed -n '1,20p' docs/qwen38-prefill-profile.md
python3.13 benchmark/side_engine_recall.py --prepare /tmp/recall.jsonl
.build/release/TinyTitanBench cpu35batch models/qwen3.5_4B_4Bit /tmp/recall.jsonl /tmp/done.jsonl
python3.13 benchmark/side_engine_recall.py --score /tmp/done.jsonl
```
