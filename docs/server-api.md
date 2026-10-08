# The server's APIs

TinyTitanServer speaks three client protocols over one generation path. A request
on any of them becomes the same validated chat request, runs through the same
queue, prompt cache, memory decorator and decoder, and is answered in the
protocol's own object and event shapes. What the model can do is identical on
all three; what differs is the wire format.

| Protocol | Routes | Spoken by |
| --- | --- | --- |
| OpenAI Chat Completions | `POST /v1/chat/completions` | most clients, Aider, Continue, OpenCode |
| OpenAI Responses | `POST /v1/responses`, `GET/DELETE /v1/responses/{id}`, `POST /v1/responses/{id}/cancel`, `GET /v1/responses/{id}/input_items` | Codex CLI, the OpenAI SDKs' `responses` namespace |
| Anthropic Messages | `POST /v1/messages`, `POST /v1/messages/count_tokens` | Claude Code, the Anthropic SDKs |

Shared: `GET /health`, `GET /v1/models`, `GET /v1/models/{id}` (in the
Anthropic list shape when the request carries `anthropic-version`),
`POST /v1/models/unload`.

There is no authentication: `Authorization` and `x-api-key` are accepted and
ignored, because the server binds to 127.0.0.1 and the machine is the trust
boundary. Every response carries `openai-version: 2020-10-01`; Anthropic
responses also carry `request-id`.

## The model name

What a request may name depends on how the server was started. Both modes refuse
an unknown name rather than answering it with whatever is loaded — that would let
a misconfigured client believe it was talking to a model it was not.

- **One model** (`--model <dir>`, no `--models-dir`): the served model or its
  `<id>-fast` alias. Any other name is refused: 404 `model_not_found` on the
  OpenAI paths, 404 `not_found_error` on the Anthropic path. That includes the
  `claude-*` names Claude Code sends by default, so point it at the served id:

  ```bash
  ANTHROPIC_BASE_URL=http://127.0.0.1:8096 ANTHROPIC_MODEL=$(curl -s 127.0.0.1:8096/v1/models | jq -r '.data[0].id') claude
  ```

- **A catalog** (`--models-dir <dir>`): `GET /v1/models` lists every install the
  catalog serves — plus `<id>@cpu` where the install has a second engine that is
  a real choice — and a request naming **any** of them is served. The router
  waits for in-flight generations, unloads the resident model and loads the named
  one, so one model is resident at a time and a client may switch between turns
  of a session. Measured on this machine's dense installs (2B/4B at 4- and
  8-bit): a switch costs about 1.5–5 s of load, and `GET /v1/models/{id}` answers
  for a model that is not resident without loading it.

`<id>-fast` is accepted in either mode for any served id, but it is **not** listed
by `GET /v1/models` — it names the same weights with the CLI-strip heuristic
switched on for that request (see [Chat Completions](#chat-completions)), so
listing it would put two entries for one install in front of every client. One
deviation worth knowing: a response to a `-fast` request reports the base id in
its `model` field rather than the alias that was asked for.

## What every protocol shares

- **Text only.** Image, file, audio and document inputs are refused with the
  protocol's error envelope naming the offending part
  (`unsupported_content` / `invalid_request_error`), never silently dropped.
- **Function tools.** Tools are the client's to run; the server returns the
  call. The server runs exactly one kind of tool itself: its own `memory_*`
  functions, when memory tools are on (see `agent-memory.md`). Hosted tool
  types on the Responses API (`web_search`, `file_search`,
  `code_interpreter`, `mcp`, `image_generation`, computer use, shell) have
  nothing here to run them and are left out of the prompt — the model never
  sees them, never calls them, and the request goes through, because Codex
  sends `web_search` on every turn. Anthropic's built-in tool types
  (`bash_*`, `text_editor_*`, `web_search_*`, ...) are refused by type;
  Claude Code sends only custom tools. A Responses `namespace` tool is
  flattened into its function tools, and a call to one carries `namespace`
  back, as the API's does.
- **Tool choice.** `auto` and `none` are honoured. Forcing a call —
  `required`, a named function, Anthropic's `any` / `tool` — is refused,
  because the decoder cannot guarantee one. `parallel_tool_calls` is
  **accepted and not enforced**, on either value: the decoder emits the calls
  the model produces, so one-at-a-time cannot be promised — and refusing the
  field would fail every client that sends it defensively (the OpenAI SDKs
  default it, Codex sends `false` on every turn) for a preference it cannot
  verify anyway. The Responses object echoes what it was given; Chat
  Completions has no field to echo into. Anthropic's
  `disable_parallel_tool_use` is refused.
- **Reasoning levels.** The server's own `--thinking` / `--reasoning-effort`
  decide what a model renders by default. A request may ask for a different
  level and the nearest one the served model actually renders is applied — the
  substitution is logged, never turned into a refusal, because coding agents
  send vocabularies this project never defined (`xhigh` on a binary-thinking
  model, `ultra`, `none`, `extra-high`) and failing one breaks that agent for
  the rest of the session. A mid-session switch is real rather than a log line:
  the level is carried into generation, which resolves the tokenizer for it.
  The controls are read from **either dialect**: `reasoning_effort` (this
  project's own spelling) and, as llama.cpp, vLLM and TabbyAPI clients send
  them, a `chat_template_kwargs` object carrying `enable_thinking` and
  `reasoning_effort`. Precedence is explicit — an explicit top-level
  `reasoning_effort` wins, then `chat_template_kwargs.reasoning_effort`, then
  `chat_template_kwargs.enable_thinking` (`false` means thinking off). Reading
  only the top-level field used to honour the level beside that object and drop
  the switch, which is the case that matters: a client forcing thinking off for
  its summarization calls, so the model's own thinking cannot eat the output cap
  and truncate the summary, had the fix silently lost. llama.cpp's
  `reasoning_budget_tokens` is **accepted and not enforced** — this runtime
  bounds thinking by the level a template renders, not by a token count — and
  the log says so. The Messages API asks for the level per request too, through
  `thinking`: `disabled` is a real off, `enabled` maps its Anthropic
  `budget_tokens` onto the same ladder (under 4k is `low`, under 16k `medium`,
  else `xhigh`) and is refused only for the budget rules Anthropic itself
  sets — required, at least 1024, below `max_tokens` — while
  `thinking: {type: adaptive}` and `output_config.effort` are accepted whatever
  the server runs (Claude Code sends the former on every request, meaning "you
  decide", so it must not force a level). A client may therefore turn thinking
  on, off or to another effort between turns of one session, whichever API it
  speaks. The model's thoughts come back as `reasoning_content` on the OpenAI
  surfaces and as a `thinking` block on the Messages API, carrying the empty
  signature string: an Anthropic signature is an attestation this server cannot
  produce, and a made-up token would only pretend to be verifiable. Thinking
  blocks a client sends back are not rendered into the prompt, so Anthropic's
  `context_management` edits (clearing old thinking) are accepted and have
  nothing to do.
- **Structured output** (`response_format`, `text.format`,
  `output_config.format`) is **enforced**, not requested: a byte-level JSON
  grammar masks the sampler, so the model can only emit a document the schema
  allows. `{"type": "json_object"}` means an object at the top level;
  `{"type": "json_schema", ...}` compiles the schema to that grammar and
  refuses, by name, every keyword it cannot promise (the subset is listed in
  [Structured output](structured-output.md)). `{"type": "text"}` and an
  unrecognized shape stay plain text. This is the one place the same rule has
  three spellings — `response_format` (Chat Completions), `text.format`
  (Responses) and `output_config.format` (Messages) — and all three are
  normalized into the Chat Completions spelling before one validator parses
  them. Thinking is **off** for such a request: the grammar constrains every
  token, so a thought would have to be written as part of the document. The
  server says so in its reasoning note rather than ignoring the level a client
  asked for.
- **Logprobs**, `n > 1`, `background: true`, prompt templates, hosted
  conversations, containers and MCP servers are refused by name.
- **Output cap.** Omitting `max_tokens` / `max_output_tokens` lets the model
  run to the context window. The Messages API requires `max_tokens`, as the
  real one does, and clamps it to the server's context window: Claude Code
  sends 32,000 on every request, and a server started with a smaller window
  serves what it has rather than refusing every turn.
- **Usage** is exact: prompt and generated tokens, with the prompt-cache hit
  reported as `cached_tokens` (OpenAI) or `cache_read_input_tokens`
  (Anthropic, where `input_tokens` excludes it), and the generated tokens the
  model spent thinking reported as `completion_tokens_details.reasoning_tokens`
  (OpenAI; `output_tokens_details.reasoning_tokens` on the Responses API). That
  count is a subset of `completion_tokens`, not an addition to it, and it comes
  from the channel each generated token landed in rather than from measuring the
  thought text — detokenizing and re-tokenizing is not an identity.

## Chat Completions

`messages` with `system`/`developer`/`user`/`assistant`/`tool` roles, `tools`,
`stream` with `stream_options.include_usage`, `stop` (up to four strings),
`seed`, `temperature`, `top_p`, `top_k`, `repetition_penalty`. Streams end with
`data: [DONE]`; a failure mid-stream sends an error object first.

A `developer` message is the OpenAI role that replaced `system`, and it is
served as one: it is validated as leading guidance before the conversation and
then rendered as the system turn it stands for, because the installed Qwen
templates define only `system`/`user`/`assistant`/`tool`. Handing the role name
to a template raised `Jinja.TemplateException("Unexpected message role.")` — an
HTTP 500 for a role the API defines and clients such as pi-ai send on every
reasoning request — so no client switch is needed any more. The Responses
surface still merges `developer` items into the leading system message, which is
the same turn by a different road.

## Responses

`input` is a string or a list of items. Item kinds: `message` (roles
`user`, `system`, `developer`, `assistant`; content a string or parts of type
`input_text`, `output_text`, `refusal`), `function_call`, `function_call_output`
(output a string or `input_text` parts), `reasoning` (accepted and skipped: a
client replaying an earlier turn returns what it was given) and
`item_reference` (resolved against stored responses). `instructions` and any
`system`/`developer` items merge into the single leading system message the
chat template accepts.

**Storage.** `store` defaults to true, as in the API. A finished response is
kept in memory (the newest 256) so that `previous_response_id` continues it —
the stored conversation and its output are prepended to the new input — and
`GET /v1/responses/{id}`, `GET .../input_items` and `DELETE` work on it.
`POST .../cancel` answers 400: nothing this server produces is a background
response. `store: false` keeps nothing, and a `previous_response_id` that
names nothing is 404 `previous_response_not_found`-style (`not_found`).

**The object** has every field the API defines (`background`,
`completed_at`, `incomplete_details`, `max_tool_calls`, `prompt_cache_key`,
`safety_identifier`, `service_tier`, `text.verbosity`, `top_logprobs`,
`truncation`, `usage.input_tokens_details.cache_write_tokens`, ...). Fields the
server cannot act on are echoed, not invented. A generation the output cap
ended is `status: "incomplete"` with `incomplete_details.reason:
"max_output_tokens"`.

**Streaming** follows the API's event grammar exactly, every event carrying
`sequence_number`:

```
response.created → response.in_progress
→ response.output_item.added (message) → response.content_part.added
→ response.output_text.delta … → response.output_text.done
→ response.content_part.done → response.output_item.done
→ [per call] response.output_item.added (function_call)
             → response.function_call_arguments.delta …
             → response.function_call_arguments.done → response.output_item.done
→ response.completed | response.incomplete
```

A failure after the stream opened ends it with `response.failed` carrying the
error inside the response object. There is no `[DONE]`; the terminal event is
the end.

## Compaction (`POST /v1/responses/compact`)

Compaction takes a conversation and returns a **compacted input window**, not a
response: nothing is stored and no session begins. The client sends the returned
`output` back as the base `input` of its next response — dropping
`previous_response_id` — and this server decodes the note into the leading system
block on the way in, so the model continues from the compacted state rather than
from the transcript.

```
POST /v1/responses/compact
{ "model": "…", "instructions": "…", "input": [ …items… ],
  "prompt_cache_key": "…", "max_compaction_tokens": 4096 }

→ { "id": "resp_cmp_…", "object": "response.compaction", "created_at": …,
    "output": [ {message with `instructions` verbatim}, {compaction item} ],
    "usage": { …the whole cost, both passes… } }
```

`model` is required (400 naming the parameter without it). The `compaction` item
carries the note in `encrypted_content`; a response's `output` is non-empty and
contains exactly one such item, as the spec requires. A `compaction` item sent
back as input is decoded in `ResponsesAPIMapper.chatMessages`, and one this server
cannot read is refused by name (`compaction_payload_invalid`) rather than dropped.

**What `encrypted_content` actually holds.** The field is provider-opaque, not
necessarily encrypted, and this server does not pretend otherwise: the payload is
base64 JSON — `{v, model, createdAt, mode, summary}` — that the client is not
expected to inspect and this server reads back on the next turn. It is versioned,
so a payload from another build is refused by name instead of being misread. Real
encryption would buy nothing here: the server is loopback-only and is summarising
the caller's own session.

**How the note is produced.** The conversation is rendered as a role-labelled
transcript and summarised under a handover instruction that keeps requirements,
decisions (with the reason and any option rejected), facts, and the next step,
quoting paths and identifiers exactly. The call runs through the normal queue
with **thinking off** — a model that reasons inside its own output cap returns an
empty note — and at temperature 0, so the same session compacts the same way.

The note is then measured with this server's own tokenizer. Over the budget
(`max_compaction_tokens` when present, else one eighth of the context, capped at
4096) it is
**compressed by a second pass rather than truncated**, because truncation drops
the end of the session, which a continuation needs most. A budget that is present
is used as sent, clamped only to half the context window; one that is zero or
negative is `400 invalid_value` on `max_compaction_tokens`, because the default is
for the caller that asked for none. Three guards keep a bad
pass out of the caller's history: lines copied from the instruction are dropped,
a repetition loop is recognised as a failed pass, and a pass that still produces
nothing usable falls back to the newest text trimmed to the budget. `mode` in the
payload — `model`, `compressed` or `extractive` — says which path produced the
note, and the server logs it with the note's token count.

Measured on the 4B and 9B installs at 4-bit, an eight-turn session compacted in
16.5 s and 39.7 s to notes that kept all four load-bearing facts (the decision and
its rejected alternative, the measurement, the path, the next step) and that the
model could then answer questions from. A 2B is not a summariser: it repeated the
transcript instead of condensing it, which the repetition guard now catches.

## Messages (Anthropic)

`messages` with `user` and `assistant` roles, content a string or blocks of
type `text`, `tool_use` (assistant), `tool_result` (user; `is_error` prefixes
the result with `Error:`), `thinking` and `redacted_thinking` (accepted and
skipped). `system` is a string or text blocks; a `system` message inside
`messages` (Claude Code's mid-conversation guidance) joins the leading system
block, because the chat template renders exactly one. Consecutive user turns
combine into one, as the API documents. A trailing assistant message (prefill) is
refused. `tools` are `{name, description, input_schema}`; `cache_control`,
`strict`, `input_examples` and the other tool decorations are accepted and
ignored. `stop_sequences`, `temperature` (0–1), `top_p`, `top_k`, `metadata`
and `service_tier` are honoured or accepted as the real API does.

**The object**: `{id: "msg_…", type: "message", role: "assistant", model,
content: [text | tool_use …], stop_reason, stop_sequence, usage}` with
`stop_reason` one of `end_turn`, `max_tokens`, `stop_sequence` (and the
string in `stop_sequence`), `tool_use`.

**Streaming**:

```
message_start → content_block_start (text) → content_block_delta (text_delta) …
→ content_block_stop → [per call] content_block_start (tool_use)
→ content_block_delta (input_json_delta) … → content_block_stop
→ message_delta (stop_reason, stop_sequence, usage) → message_stop
```

`ping` events keep a slow first token alive. A failure sends
`event: error` with `{"type":"error","error":{...}}`. Errors on every
Anthropic route use that envelope with the API's types and codes:
`invalid_request_error` 400, `not_found_error` 404, `overloaded_error` 529
when the queue is full, `api_error` 5xx.

`POST /v1/messages/count_tokens` returns `{"input_tokens": N}` for the prompt
as the chat template renders it, from the model's own tokenizer. A backend
without a tokenizer (a test double) answers 501.

## Testing

```bash
swift test --filter "ResponsesAPI|AnthropicMapper|AnthropicMessages|HTTPServerTests|OpenAIValidation"
swift test --filter ClientCLITests     # the installed codex and claude binaries, end to end
```

The HTTP suites drive real sockets against scripted backends and check the
event grammars event by event; the mapper suites cover the request grammars
and the refusals. `ClientCLITests` runs the real Codex CLI and Claude Code
against the server with the model replaced by a scripted backend, which is
what caught the fields the unit tests never wrote (Codex's `namespace` and
`web_search` tools, Claude Code's `context_management` and `adaptive`
thinking); it skips when a CLI is not installed. None of them needs a model.
For hand runs, `TINYTITAN_STUB_SERVER_SECONDS=600 swift test --filter
StubServerForManualRuns` keeps such a server up and writes its port to
`/tmp/tinytitan-stub-port`.
