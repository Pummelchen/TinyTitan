## TinyTitan 5.8 — a resident side-engine for memory, and a LAN manager for a fleet

Agent memory grew a second, small model: a 4B on the CPU that answers one closed
question at a time — is this fact worth keeping, do these two say the same
thing, do they disagree, which kind of change is this, could this fact answer
this question — while the main model keeps the GPU and the person keeps their
turn. Around it, memory learned to hold a write a stored rule forbids, to rank a
search by how rare a term is rather than how often it appears, and to let the
retrieval question work in the background instead of on the request path. The
repository also gained `dsh-lan-manager` and `ttlanmanager`, a LAN-scoped
control plane for a fleet of DeepSeek Harness instances. Everything here was
already on `main`.

### Agent memory has a resident side-engine

A small Qwen 3.5 runs on the CPU as a resident helper and decides one thing at a
time, because a small model composes badly but decides well
(`docs/side-engine-tasks.md`). The 4B 4-bit install is the default and the
verification instrument; the 9B is optional and buys the reply check while being
*worse* at durability; the 2B is not used.

Only the tasks with a caller are on the port, and the caller is consolidation.
A model-derived fact is asked first whether it is **worth keeping** (T2), and one
the engine rejects is not stored at all. A changed value the store already holds
is asked which kind of change it is (T4) once a stored rule fixes it. New keys
are checked for **duplication** (T5) and **contradiction** (T3) against the
session's own scope and the shared workspace. `nil` is "no decision", so an
absent, shut-down or confused engine leaves the deterministic path exactly as it
was.

The questions are budgeted like the model calls they are: measured at 15.2 s on
the 4B, so one consolidation puts at most six questions in total and three to
any one fact — about a minute and a half in the pause consolidation already
runs in.

Verified against the real model: the release-only test drives durability,
duplication and supersession on the 4B (133.1 s for six judgements) and the 9B
(255.4 s), and the task matrix is in `docs/side-engine-tasks.md`.

### A stored rule can hold a write back

T4 was measured ready but unreachable: it decides 100% once the stored rule is
supplied, and nothing supplied one. `MemoryRuleLookup` now finds it by key — the
last segment of the changed key, so `characters/marcus/eyes` looks for
`rules/eyes` — which is free, because it is a key match and not a model call. A
`.conflict` keeps the old value and logs only the key, never the rule or either
value; `.update` changes nothing. Only a model-derived fact is asked about, so
the person always overrules a rule, and a change with no rule is not asked about
at all.

### Retrieval ranks by rarity, and T7 works in the background

The token ranking scaled every term the same, so a question naming a *common*
word won: "does it ever rain in this town?" matched the town's key for 3 and the
rain rule's value for 2. `MemoryRanking` now weights each term by its inverse
document frequency over the candidates, which takes the authored paraphrase set
from recall@1 1 of 4 to **3 of 4** while the ten questions phrased in the store's
own words stay 10 of 10 (`benchmark/side_engine_recall.py --baseline`).

The one miss left shares no term in any form — "How often does the boat cross
the water?" against `rules/ferry = runs only on Sundays` — so no weighting
reaches it. T7 does, and it now has a caller that can afford it:
`MemoryRetrievalHinter` registers the question a search just asked and walks the
scope's facts **only in the idle window** (`ServerCoordinator.generating` read
as `isIdle`), never on the request path. Each YES becomes a ranking hint keyed
by the fact plus an FNV-1a fingerprint of the value it judged, so the next
search for that question puts the hinted facts first, including one the token
ranking never returned. It covers at most 64 facts a question and keeps 16
questions, least recently asked evicted; a value that changed cannot be promoted
on an answer about the old one. Without an engine, or while a client is
generating, the path is byte-for-byte the token ranking it was.

### The n-gram table is shared between builds instead of copied

Qwen3.8-Flash-Next's `ngram_table.bin` is 102 GB, and every quantization of the
model stored its own copy. `TinyTitanRepack --share-ngram-table` hardlinks it
from the source snapshot instead, and `prepare_qwen38.py --reuse-ngram-table`
links an existing table into a new build. The table is a hash table with no
self-description, so a wrong one would read silently as garbage ids: the reuse
gate refuses a table whose PLE constants differ, is pinned by five cases in
`benchmark/test_prepare_qwen38.py`, and the link-and-size path has a Swift test.

### Per-tensor bit widths resolve in the resident index

`HyperConnection`, `PLEBlock` and `QSAIndexer` read weights whose width came from
the attention slot, and the loader never checked those tensors' widths — so a
per-tensor override on them was unhonoured *and* unguarded: packed at one width,
read at another, silently wrong. The three families now resolve their width
through the manifest's overrides with the slot as fallback and validate the
resolved width against the kernel, with the GDN a/b pair refused a quantized
override by name. This is the enabler for the ~10 MB promotion
`tools/precision_plan_qwen35.py` proposed.

Its quality case is measured twice and is not there: the 4B's own 16 MB
`k_proj`/`v_proj` promotion scores **18/20 against a control's 18/20** on a
twenty-prompt suite, and on a paired held-out perplexity A/B it is
**0.0097 ± 0.0067 nats** ahead of the control (t −1.46), inside the
instrument's ~0.013-nat floor. The converter policy is therefore deliberately
not built (`benchmark/quant_perplexity_ab.py`).

### The DSH LAN Manager: a control plane for a DSH fleet

`plugins/dsh-lan-manager` gives a DeepSeek Harness host a LAN-scoped management
API — list active workspaces and their sessions, prompt one or all of them, read
a session's messages back, archive or delete, start a session through the
harness's own controller, and an aggregate inventory of every member of the
group. Members find each other over tailnet peers, Bonjour, configured peers and
an optional subnet sweep, and every gossiped address is validated against the
same allowlist the request fence uses before anything is dialled.

`sources/TinyTitanFleet` builds `ttlanmanager`, a live terminal dashboard over
that API with a pure renderer and key map (so layout and keys are tested with no
terminal). It is an operator tool and deliberately not part of the installed
engine. Verified by 44 tests in 11 suites; reaching it from another machine is
blocked upstream, because the harness refuses any bind but loopback (TT-020).

### Both DSH plugins pin exactly one harness release

`dsh-tinytitan` and `dsh-lan-manager` support **`0.1.6-alpha.2`** and refuse any
other release — older, newer, a build from `main`, or one whose version cannot
be read — with one line on stderr and no writes into the harness home. Neither
throws, so DSH still boots and removing ours leaves nothing to undo. The pin
lives in the peer dependencies, the launcher and each plugin's constant, and CI
now runs both plugin suites, which nothing did before.

### Also in this release

- **Steady-state decode past the ANE handover costs nothing** — +0.7% on
  AgentWorld 35B-A3B 4-bit, measured after the old figures turned out to be a
  ~60-token window (TT-007).
- **The Qwen3.8 expert-cache budget stays at 96 slots** (TT-011), and
  `TINYTITAN_KEEP_WIRED` is a tri-state so `=0` pages the expert cache out
  (TT-008).

  > **Superseded.** `TINYTITAN_KEEP_WIRED` is gone, along with the rest of the
  > opt-in decode switches that tuning round measured: `3eb11cf` removed each
  > one after it turned out a wash or a loss on the current build. Whether the
  > expert cache stays wired is now the profile row's decision alone
  > (`ModelProfile`), and `ModelProfileTests` records that the override is gone.
  > The bullet stands as the record of what 5.8 shipped.
- **The MTP verify pass is attributed on the installed pair** (TT-006), the ANE
  re-warm is separated from drift and the pin removes it (TT-005), the E5RT
  arenas are returned (TT-004), and the top-2 logits of both sampling paths are
  traced (TT-002).
- **The QSA host prefill selection is 2.75% of a long prefill**, measured before
  building a GPU path, and the GPU path is not built (TT-010).
- **The ThreadSanitizer gate carries one top-frame suppression** (TT-001), and
  the project has a written standard for its single open-task table
  (`docs/task-table-standard.md`).

### Performance

The README's benchmark table was **not** re-measured for this release; its rows
are quoted as they stand. Measured on this commit for this release:

- the engine's own speeds against the 5.7 record
  (`benchmark/internal-speeds/v5.8.json`): QKV GEMV **78.6 GB/s** (+4.4%),
  routed MoE **44.2 GB/s** (+4.0%), GDN in-projection **77.4 GB/s** (0.0%), CPU
  int8 GEMV **49.2 GB/s** (+1.7%), prefill **25.9 tok/s** (0.0%), decode
  **26.7 tok/s** (+5.2%), first token **0.27 s**, ANE prefill **49.3 tok/s**
  (−2.6%). No metric is past the 10% gate and the response hash did not change;
- the side-engine's wired judgements, 15.2 s each on the 4B;
- the held-out perplexity A/B, 1,023 paired token positions per install, about
  three minutes per install on the CPU.

### Verification

Measured on this commit by the release dry run:

- six lint gates clean, **2,035 functions** scanned, the shell gate over 19
  scripts on bash 3.2.57;
- **1,482 tests in 222 suites**, all passing;
- **11 golden baselines byte-identical**: the 125B at 4-bit, AgentWorld 4- and
  8-bit, qwen36 4- and 8-bit, and the dense 2B/4B/9B at both widths;
- a clean scratch release build with the compiler-warning scan clean, and the
  archive staged and packaged from that tree;
- the engine's speeds recorded against the 5.7 baseline and committed
  (`benchmark/internal-speeds/v5.8.json`), every metric inside the gate.

**Five golden targets are not checked**, because their install is not under
`models/` and nothing may be fetched to change that: `ornith-8`, `ornith-4`,
`qwen38-8`, `katcoder-4`, `katcoder-8`.

### Checksum

`tinytitan-5.8-macos-arm64.tar.gz` sha256: `SHA256_PENDING`
`tinytitan-5.8-macos-arm64.tar.gz` size: `ARCHIVE_BYTES_PENDING` bytes
