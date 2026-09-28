# Open Responses conformance

What the [Open Responses](https://www.openresponses.org/specification) acceptance
suite says about this server's `/v1/responses` surface: first run as a scorecard on
2026-09-14, then **re-run serially on 2026-09-15** after the sampling echo fix. Open
Responses is an open, vendor-neutral formalisation of the OpenAI Responses API —
schema, items, semantic streaming events, state machines — plus things this server
does not have (`/v1/responses/compact`, `allowed_tools`) and one it has decided not
to implement at all: the WebSocket transport, which the specification makes optional
for servers (see [Out of scope](#out-of-scope-websocket-transport)).

The point of running it was to turn "we implement the Responses API" into a
number with named failures. It did, and it corrected a guess: the most valuable
fix is **not** one of the missing features.

**Score on the HTTP/SSE surface: 7 passed, 3 failed, 0 skipped — 10 total**,
measured 2026-09-15. The suite runs 17 tests and seven of them exercise the
WebSocket transport, which this project will not implement; they are excluded here
rather than counted as failures
([Out of scope](#out-of-scope-websocket-transport)). Two of the three in-scope
failures needed `/v1/responses/compact`, which has since been implemented
([below](#what-was-missing-and-where-it-stands)); the third is the deliberate
text-only `image-input` refusal.

## First run (2026-09-14)

**1 passed, 16 failed, 0 skipped — 17 total.**

| Test | Result | Cause |
| --- | --- | --- |
| `response-output-phase-schema` | **pass** | — |
| `basic-response` | fail | required sampling fields |
| `assistant-phase` | fail | required sampling fields |
| `streaming-response` | fail | required sampling fields |
| `system-prompt` | fail | required sampling fields |
| `tool-calling` | fail | required sampling fields |
| `compact-response` | fail | no `/v1/responses/compact` (405) |
| `compact-missing-model` | fail | no `/v1/responses/compact` (405) |
| `websocket-compact-new-chain` | fail | no `/v1/responses/compact` (405) |
| `websocket-response` | fail | no WebSocket transport |
| `websocket-sequential-responses` | fail | no WebSocket transport |
| `websocket-continuation` | fail | no WebSocket transport |
| `websocket-reconnect-store-false-recovery` | fail | no WebSocket transport |
| `websocket-previous-response-not-found` | fail | no WebSocket transport |
| `websocket-failed-continuation-evicts-cache` | fail | no WebSocket transport |
| `image-input` | fail | **by design** — this server is text-only |
| `multi-turn` | fail | **suite artifact** — 429 `queueFull` |

### What is a real conformance failure, and what is not

| Cause | Tests | Nature |
| --- | --- | --- |
| Required sampling fields absent | **5** | real, and cheap |
| No compaction endpoint | 3 | missing feature |
| No WebSocket transport | 6 (+1 that also needs compaction) | **out of scope by decision** — see below |
| Text-only refuses `input_image` | 1 | deliberate limitation, not a bug |
| 429 `queueFull` | 1 | artifact of the suite, not the server |

## Re-run (2026-09-15): 7 passed

The sampling echo is implemented and the suite was re-run with the parallelism
removed — one `--filter` invocation per test, so the single-model server is never
asked to admit 17 generations at once. The upstream entrypoint is unmodified
(no suite patch): only its own `--filter` flag separates the runs.

**7 passed, 10 failed, 0 skipped — 17 total.**

| Test | First run | Re-run | Cause now |
| --- | --- | --- | --- |
| `basic-response` | fail | **pass** | — |
| `assistant-phase` | fail | **pass** | — |
| `streaming-response` | fail | **pass** | — |
| `system-prompt` | fail | **pass** | — |
| `tool-calling` | fail | **pass** | — |
| `response-output-phase-schema` | pass | **pass** | — |
| `multi-turn` | fail | **pass** | serial run removes the 429 |
| `compact-response` | fail | fail | no `/v1/responses/compact` (405) |
| `compact-missing-model` | fail | fail | no `/v1/responses/compact` (405) |
| `websocket-compact-new-chain` | fail | fail | no WebSocket, then no compact (405) |
| `websocket-response` | fail | fail | no WebSocket transport |
| `websocket-sequential-responses` | fail | fail | no WebSocket transport |
| `websocket-continuation` | fail | fail | no WebSocket transport |
| `websocket-reconnect-store-false-recovery` | fail | fail | no WebSocket transport |
| `websocket-previous-response-not-found` | fail | fail | no WebSocket transport |
| `websocket-failed-continuation-evicts-cache` | fail | fail | no WebSocket transport |
| `image-input` | fail | fail | **by design** — text-only refusal (`unsupported_content`) |

Two runs, two separate causes, and the prediction was one test low: the five
sampling-field tests flip, and `multi-turn` flips too — its only failure was the
429, so removing the concurrency removes it. `response-output-phase-schema` never
failed. **0 of the 17 results now report a sampling-field error.**

The five flips are the echo fix alone. `temperature`, `top_p`,
`presence_penalty` and `frequency_penalty` are now numbers in every `Response`
object — resolved from the same `GenerationConfig` the sampler is handed, while
the request-side fields stay nil so the served model's profile still supplies
them (audit C11). The remaining 10 are the two absent optional surfaces —
`/v1/responses/compact`, which is being built
([below](#the-two-missing-features-judged-on-merit)), and the WebSocket transport,
out of scope by decision — plus the deliberate text-only limit; none is a schema
or object-shape defect.

## The finding that mattered: five tests, three fields (fixed)

Five tests — every test that validates the `Response` object — failed on the same
three fields:

```
top_p: Expected number, received null
presence_penalty: Required
frequency_penalty: Required
```

`responseResourceSchema` marks all three **required**. This server emits `top_p`
as `null` and omits the two penalties, so nothing that reads a `Response` object
can validate it.

This is worth separating into two concerns that are currently one:

- **Validation** wants the request's fields to stay nil when the client named
  none, so the served model's own profile supplies them. That is what the mapper
  does today, and the mapper tests pin it.
- **Echo** wants the `Response` object to report the values the server actually
  used. A client cannot tell what `top_p` was applied otherwise, and the spec
  requires the number.

Keeping the fields nil satisfies the first and breaks the second. Three fields in
the `Response` builder — resolved from the same profile the sampler uses — was the
single highest-value change on this page: **5 of the 16 failures**, for a change
far smaller than either missing feature.

**Fixed on 2026-09-15.** `ResponsesAPIEcho` now takes the validated
`GenerationConfig` and echoes `temperature`, `top_p`, `presence_penalty` and
`frequency_penalty` as numbers; the mapper still leaves an omitted request field
nil (C11), so the resolved value comes from the served model's profile. The
mapper test `responseEchoesTheResolvedSampling` pins the echo against a
non-house profile, beside the existing C11 test. All five tests pass in the
re-run above.

## `[DONE]`: a specification MUST this server should not follow

The specification says, under *Streaming HTTP Responses*:

> "The terminal event **MUST** be the literal string `[DONE]`."

This server deliberately does not send it, and says so in two places:

- `HTTPServerHandler+Responses.swift` — *"The Responses API has no `[DONE]`
  terminator; the final event is it."*
- `HTTPServerHandler+Chat.swift` — *"chat sends an error object then `[DONE]`; the
  Responses API and the Messages API send a typed `error` event and no
  terminator."*

**OpenAI's Responses API ends with `response.completed`, not `[DONE]`.** Following
the MUST would make this server *less* compatible with the API it imitates.

The suite agrees with this server, not with its own specification:

- `src/lib/sse-parser.ts` treats `[DONE]` on HTTP as a sentinel to **skip** —
  `// Skip the [DONE] sentinel - it's not a real event` — so it is neither
  required nor rejected there.
- On WebSocket — a transport this project does not implement — the same suite
  **rejects** it: `"Received [DONE] before a terminal WebSocket event"` is
  recorded as an error.

So the normative MUST is unenforced on one transport and contradicted on the
other. That belongs upstream as a specification bug, not in this server as a
compliance fix.

## What was missing, and where it stands

**`/v1/responses/compact` — implemented.** It takes a conversation and returns the
compacted input window the spec describes: the caller's instructions verbatim,
then one `compaction` item carrying the note, which the next response takes as its
base `input`. The two in-scope failures above are the tests it was written
against. The pipeline — thinking off, a tokenizer-verified budget, a second pass
that compresses rather than truncates, and guards that drop instruction echoes and
recognise a repetition loop — is documented in `docs/server-api.md`.

**This record has not been re-run against it yet.** The score above still measures
the tree before that work; the next run is expected to show 9 passed and 1 failed,
the by-design text-only refusal. A third test, `websocket-compact-new-chain`, needs
this *and* the WebSocket transport, so it stays out of scope with the transport.

**`allowed_tools`** — the other gap, and *not* exercised by this
suite (no test covers it). Its stated purpose is to narrow the executable tool set
without changing `tools`, because mutating `tools` invalidates prompt and schema
caches. This server has a real prompt cache (`--prompt-cache-mode multi-prefix`,
256 MiB), which is exactly the deployment the field was designed for. Not urgent,
but the fit is unusually good.

## Out of scope: WebSocket transport

**Not implemented, and not planned.** The specification makes it optional for
servers, and the reasons here are concrete: the server is loopback-only with one
resident model, so a persistent socket buys no reach; the same event model is
already carried over HTTP and SSE; and the one thing the transport would really
add — connection-local continuation, so a client can carry on without resending
and re-prefilling the whole context — is served here by the prompt cache
(`--prompt-cache-mode multi-prefix`), which is the mechanism this project has
actually measured.

Seven suite tests exercise the transport: `websocket-response`,
`websocket-sequential-responses`, `websocket-continuation`,
`websocket-reconnect-store-false-recovery`, `websocket-previous-response-not-found`,
`websocket-failed-continuation-evicts-cache` and `websocket-compact-new-chain`.
They stay in the tables above exactly as the suite reported them, because that is
what the run produced, and they are excluded from the score. A failure against a
transport this project has decided not to build is not a conformance gap, and
nothing in the suite requires the transport of a server that speaks HTTP and SSE.

## One caveat about the scorecard itself

The suite runs **all 17 tests in parallel** (`runAllTests` maps over
`testTemplates` with `Promise.all`). This server admits one generation at a time
with four queued — `queueLimit + 1`, default 5 — and sheds the rest with 429
(`ServerCoordinator.swift`; the first run's log showed
`status=429 error=ServerRequestError.queueFull`).

So at least one failure is the suite's concurrency colliding with a
single-model server, not a conformance defect. **Confirmed by the 2026-09-15
re-run**: `multi-turn` and every other generation test pass when the tests are
run one at a time, with no server change (`--queue-limit` was not raised). A
re-run that separates the two:

```bash
tools/server_launcher.sh --client server --model qwen35-2b --bits 4
# then, against the running server, one test per invocation:
bun run test:compliance -u http://127.0.0.1:8080/v1 -k any -m qwen3.5-2b_4-Bit \
  --filter basic-response
```

`image-input` will still fail whatever the concurrency: the models are text-only
and the Responses mapper refuses image parts by design
(`ResponsesAPIModels.swift`: *"Text-only: image and …"*).

## Reproducing

```bash
# 1. the server, on an installed model (2B 4-bit is enough and loads in seconds)
tools/server_launcher.sh --client server --model qwen35-2b --bits 4 --dry-run   # prints the exact command
# ... run the printed command

# 2. the suite, from a checkout of openresponses/openresponses
git clone --depth 1 https://github.com/openresponses/openresponses.git
cd openresponses
node -e "const d=require('./package.json');require('fs').writeFileSync('package.json',JSON.stringify({name:'or',private:true,dependencies:{zod:d.dependencies.zod}},null,2))"
npm install                      # zod only; the generated schemas are committed under src/generated
npx tsx bin/compliance-test.ts -u http://127.0.0.1:8080/v1 -k any -m qwen3.5-2b_4-Bit --json
```

For a **serial** run, invoke the same entrypoint once per test id so the server
never sees more than one generation at a time (no suite patch, no raised
`--queue-limit`):

```bash
for id in basic-response assistant-phase response-output-phase-schema \
          streaming-response system-prompt tool-calling image-input \
          multi-turn compact-response compact-missing-model; do
  npx tsx bin/compliance-test.ts -u http://127.0.0.1:8080/v1 -k any \
    -m qwen3.5-2b_4-Bit --filter "$id" --json
done
```

These are the ten tests of the surface this server ships. The seven WebSocket test
ids are omitted on purpose: the transport is out of scope by decision (above), so
running them would only reproduce a known non-result.

Node alone cannot run the suite: its resolver rejects the extensionless relative
imports (`from "../src/lib/compliance-tests"`), so `tsx` (or the repo's own Bun)
is required. The generated zod schemas are committed, so `kubb generate` is not
needed — only `zod` is.

## Verified

- Server: `qwen3.5-2b_4-Bit` on the GPU, `--reasoning off`, port 8080, macOS
  26.6.2, Swift 6.3.3, Apple M3. No performance claim here — this is a protocol
  check, not a measurement.
- Suite: `openresponses/openresponses` at `master`, 17 tests. First run
  2026-09-14 (all 17 in parallel); re-run 2026-09-15 (one test per invocation)
  against a release build of the working tree containing the echo fix.
- `swift test --no-parallel` on that tree: 1524 tests in 234 suites, 0 failures.
- The scorecard is a point-in-time measurement, not a release claim; the
  `[DONE]` item is an upstream report, not a work item. The compaction endpoint is
  not in the tree yet — it is the feature being built — so those two failures are
  the current expectation, and the third in-scope failure
  is the by-design image refusal. WebSocket transport is out of scope by decision,
  so its seven tests are not counted against the surface this server ships.
