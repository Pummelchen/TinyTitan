# `dsh-lan-manager`

A LAN-scoped management API for a DeepSeek Harness (`dsh`) host. It answers the
question _"what is running on that Mac, and can I drive it from here?"_ for a
cluster of harness instances, without a GUI and without a third-party gateway.

```bash
curl -H "x-dsh-token: $DSH_LAN_KEY" http://127.0.0.1:3080/dsh-lan/workspaces
curl -X POST http://127.0.0.1:3080/dsh-lan/prompt-all \
     -H "x-dsh-token: $DSH_LAN_KEY" -H 'content-type: application/json' \
     -d '{"prompt":"report your current goal"}'
```

## What it does

| #   | Capability                                                                   | Endpoint                                                         |
| --- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| 1   | List active workspaces — the ones the web page shows                         | `GET /dsh-lan/workspaces`                                        |
| 2   | List the visible sessions of a workspace                                     | `GET /dsh-lan/workspaces/:id/sessions` · `GET /dsh-lan/sessions` |
| 3   | Prompt one session                                                           | `POST /dsh-lan/prompt`                                           |
| 3a  | Prompt **every** active session                                              | `POST /dsh-lan/prompt-all`                                       |
| 3b  | Read a session's messages back — the answers, not just the questions         | `GET /dsh-lan/sessions/:id/messages`                             |
| 4   | Delete a workspace (archiving its sessions first)                            | `POST /dsh-lan/workspaces/:id/delete`                            |
| 5   | Archive a session                                                            | `POST /dsh-lan/sessions/:id/archive`                             |
| 6   | Register an existing folder as a workspace                                   | `POST /dsh-lan/workspaces`                                       |
| 6a  | Start a session on a workspace, through the harness's own session controller | `POST /dsh-lan/sessions`                                         |
| 7   | The group: who else is running, and what they hold                           | `GET /dsh-lan/peers` · `GET /dsh-lan/peers/:id`                  |
| 8   | One aggregate for a manager: this Mac **and** every member                   | `GET /dsh-lan/inventory`                                         |
| —   | Liveness and the caller's fence verdict                                      | `GET /dsh-lan/health`                                            |

**The group.** Every instance shares one **group key** (a string, default
`tinytitan-lan`, changeable) and discovers the others without being told where
they are — **online tailnet peers that have an IPv4 address**, whatever continent
they are on (mobile platforms and IPv6-only peers are skipped), Bonjour
(`_dsh-lan._tcp`) on the local network, configured `peers`, and optionally
a local-subnet sweep — on a timer (60 s default). Note that Bonjour does not cross
a tailnet, and a peer behind an ACL that blocks the port stays invisible: the mesh
gossip is what carries discovery from one member to the rest. Each member keeps
the others' active workspaces and sessions, and members trade address lists with
each other so one found Mac is enough to find the rest. Every gossiped address is
validated against the same LAN/Tailscale allowlist the request fence uses before
anything is dialled, and a member must answer with our group key to be listed.

**Bonjour is browse-only, and on the pinned harness it is idle.** Nothing in this
project registers `_dsh-lan._tcp`, and registering it would be a _false beacon_
today: Bonjour advertises this host's **LAN** address, where nothing is listening,
because the harness refuses to bind anything but loopback (see _Reaching it from
another machine_). A peer that is discovered and then cannot be probed is worse than
no peer, so the browser stays — it is correct, and it will find a third-party
advertiser — and registration waits for a reachable bind. The plugin says so in the
log at every mount.

**The mesh knows; a manager acts.** A plugin never sends a prompt to another
instance and never modifies one. Prompting and mutating is the job of the
external manager — `ttlanmanager`, the **TinyTitan DSH LAN Manager**, in this
repository's `sources/` — which reads the group from any one member's
`/inventory` and then talks to the member that owns the thing being acted on.

**"Active" means what the web page shows.** Active workspaces are derived from the
session projection: a workspace is listed when at least one non-archived session
lives in it, and registry metadata is layered on where the registry knows it — so a
workspace the registry never saw is still listed, with `registered: false` and a
null `id`. A session is visible when it is not in the registry-global
`archivedSessionIds` set. Archived sessions keep their slot and their history —
archiving hides a row, it does not delete anything.

**Archive ≠ delete.** `archive` hides a session. `delete` removes a workspace from
the registry and, by default, archives its sessions on the way out so a stray call
cannot silently drop history. Neither touches the folder on disk or the session
logs; pass `{"archiveSessions": false}` to delete a workspace without archiving.

## Security

The API mutates workspaces and enqueues model prompts, so access is fenced in three
layers, checked in this order:

1. **Source address**, before the body is read and before any handler runs. Allowed:
   loopback, RFC 1918 (`10/8`, `172.16/12`, `192.168/16`), link-local, and
   **`100.64.0.0/10`** — the Tailscale/CGNAT block. Everything else, including any
   unparseable address, is refused. The peer address is taken from the socket;
   `X-Forwarded-For` is deliberately **not** trusted, because a forwarded header is
   attacker-controlled and trusting it would let any caller claim loopback.
2. **The group key** (`groupKey` / `token`, `DSH_LAN_KEY` / `DSH_LAN_TOKEN`),
   compared in constant time. It is one string with two jobs: the door key every
   request presents, and the tag that decides which instances are one fleet.
   **It ships with a default (`tinytitan-lan`), so by default it groups rather
   than protects** — every Mac that installs this plugin joins the same group
   with no setup, and any host on the allowlist that knows the default can call
   it. Change it on every Mac when the network is not entirely yours.
3. **Origin**, on mutating verbs only: a foreign site in an allowlisted browser must
   not be usable as a confused deputy.

### Reaching it from another machine

**It cannot, on the pinned harness.** The API is registered on the harness's own
web server, so it answers only where that server listens — and
`@deepseek-ai/dsh-host-webserver` accepts just two bind addresses:

```
$.host expected "127.0.0.1" | "0.0.0.0" but got "192.168.18.73" (at host)
```

`127.0.0.1` is the default and reaches this machine alone. `0.0.0.0`, which would
put the API on the network, is refused by the harness before it binds:

```
error: --host 0.0.0.0 is intentionally not supported yet for safety:
it would expose remote code execution to the network; use 127.0.0.1 instead
```

A specific interface does not work either: the webserver plugin's schema rejects
it and the profile fails to load. So the source fence, the group key and the
fleet-wide prompt are all built and idle until upstream allows one of them — a
third ask alongside the two in `docs/dsh-upstream-asks.md`.

Call this API from the machine running the harness:

```bash
curl -H "x-dsh-token: $DSH_LAN_KEY" http://127.0.0.1:3080/dsh-lan/health
```

The fence and the key still do their jobs on that path — they are what keep the
routes from being reachable by anything the harness is later bound to expose.

## Supported harness version

This plugin supports **exactly one DeepSeek Harness release: `0.2.0-rc.2`** —
not older, not newer, and not a build from `main`. It registers a route into the
harness web server and drives the harness through host services, so a release it
has not been verified against is not something to guess at. It is the same release
`dsh-tinytitan` supports and the one `tools/dsh_local.sh` installs, and
`test/support.test.js` asserts the constants cannot drift apart.

On any other release it **refuses to run**: one line naming both versions, and
nothing else — no route registered, no discovery started. It never throws, so DSH
boots normally, every other plugin loads, and removing this plugin leaves nothing
to undo. A version that cannot be read at all is refused the same way.

The refusal goes to **stderr**, and that is not a style choice: the harness collects
a plugin's log records and prints them **only when the boot itself fails**, and its
startup exporter takes level ≥ 2, so a host-logger line is invisible on a healthy
boot. The refusal is therefore not subject to `logToHost` either — it is not
routine chatter. Found by booting a throwaway harness, not by reading.

## Install

```bash
dsh plugin --profile web add ./plugins/dsh-lan-manager        # from the checkout
dsh plugin --profile web add github:Pummelchen/TinyTitan#<commit>   # pinned
```

The bundle patch mounts the row; the next `dsh web` picks it up. No `--patch` flag
is needed.

## API

### `GET /dsh-lan/workspaces`

```json
{
  "ok": true,
  "archivedCount": 29,
  "totalWorkspaces": 12,
  "workspaces": [
    {
      "id": "a9832cc9-…",
      "path": "/Users/me/ProjectA",
      "title": "ProjectA",
      "createdAt": "2026-09-01T…",
      "updatedAt": "2026-09-02T…",
      "sessionCount": 2,
      "hiddenSessionCount": 1,
      "sessionIds": ["s-a1", "s-a2"]
    }
  ]
}
```

`sessionIds` holds only visible sessions; `hiddenSessionCount` is how many are
archived inside that workspace. Pass `?includeEmpty=true`-equivalent config
(`includeEmptyWorkspaces`) to keep workspaces the page would not show.

### `GET /dsh-lan/sessions`

Every visible session across every active workspace, de-duplicated, each tagged
with its workspace.

### `GET /dsh-lan/sessions/:id/messages`

The session's message history, which is what makes a fleet audit an audit:
`prompt-all` delivers the question, and this collects the answers.

```json
{
  "ok": true,
  "sessionId": "s-a1",
  "total": 42,
  "returned": 40,
  "truncated": true,
  "messages": [
    {
      "id": "m-40",
      "role": "assistant",
      "text": "all green",
      "textTruncated": false,
      "reasoningChars": 1180,
      "otherBlocks": 2
    }
  ]
}
```

`?limit=N` caps how many of the **newest** messages come back (default 40, hard
ceiling 200), because an audit wants the end of the conversation. Per message:
`text` is the concatenated text blocks, capped so one dumped tool result cannot
dominate a reply; `reasoningChars` reports a thinking block's size without
inlining it, since reasoning is not the answer; and `otherBlocks` counts blocks the
plugin does not render, so nothing is dropped silently.

The history is the harness's own derivation — `deriveMessages()`, the same
projection `POST /prompt` feeds — so the plugin holds no opinion about session
format. A **live** agent is preferred because its session is already in memory;
with none, the stored log is read through the harness's cold-read service
(`sessionQuery.readSession`) and handed back to `sessions.prepare()` as a
**detached** session, which is the same call that service makes internally. An idle
or archived session therefore still answers. A session that exists in neither is a
`404`, not an empty conversation, which would read as "this session said nothing".

### `POST /dsh-lan/sessions`

Start a live session on an existing workspace:

```json
{ "workspaceId": "ws-1" }
```

`path` (or `cwd`) works instead of `workspaceId`, and `agentPreset` names a preset.
The reply is `{ "sessionId": "session-…", "agentPreset": "standard" }`.

Starting is **delegated to the harness's session controller** — the service named
`sessionController`, _not_ `sessions` (that is the raw `dsh-session` store, whose
`create(id, options)` mints a bare session and rejects this request shape) — and not
reimplemented: that one call composes the agent's world from the preset, resolves the
default model, creates the working directory, mints the id and attaches the session
to the workspace — rebuilding it here would be a second copy of harness logic,
drifting at every release. A profile that composes no session controller (headless,
SDK-only) replies `501` and says so, because answering `ok` for a session that was
not created is worse than refusing.

`POST /dsh-lan/workspaces` also accepts `"startSession": true`: it creates the
workspace and then starts a session on it, returning it as `session`.

### `POST /dsh-lan/prompt`

```json
{ "sessionId": "s-a1", "prompt": "run the tests and summarise failures" }
```

`prompt` may also be an array of content blocks. The target session must have a
**live agent** — one the UI has open or is currently running — otherwise the call is
a `404`, because there is nothing to enqueue onto. Delivery uses the same
`followup` entry point the SDK server uses, so a prompt sent here is an ordinary
turn, not a side channel; `followup` always wakes the agent, so the receipt reports
`wakeup: true` as a fact. The message is a complete user message — content **and**
`source: { kind: "user" }` — because upstream's `createUserMessage` mints the id but
does not invent a source, and the agent loop reads `source.kind`.

### `POST /dsh-lan/prompt-all`

```json
{ "prompt": "report status", "sessionIds": ["s-a1"], "limit": 10 }
```

`sessionIds` and `limit` are optional. Delivery is per-session and never
all-or-nothing: one session without a live agent is reported in `failed[]` while the
rest are delivered, so one stale session cannot stall a fleet-wide prompt.

```json
{ "ok": true, "delivered": [ … ], "failed": [ { "sessionId": "s-x", "code": "not-found" } ],
  "total": 3, "considered": 3 }
```

### `POST /dsh-lan/sessions/:id/archive`

Hides the session in the UI. Reversible; history and slot are kept.

### `POST /dsh-lan/workspaces/:id/delete`

```json
{ "archiveSessions": true }
```

Removes the workspace from the registry. With `archiveSessions` (the default) its
sessions are archived first; the response reports `archivedSessionIds` and any
`archiveFailures` individually, so a partial archive is visible rather than silent.

## Managing a cluster

The plugin discovers the group itself, so this is only the fallback for a
network with neither Tailscale nor Bonjour. A loop over addresses you already
know:

```bash
for host in 192.168.18.27 192.168.18.25 192.168.18.29 192.168.18.26; do
  printf '%s: ' "$host"
  curl -fsS --max-time 5 -H "x-dsh-token: $DSH_LAN_TOKEN" \
    "http://$host:3080/dsh-lan/workspaces" \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print(len(d["workspaces"]), "workspaces", sum(w["sessionCount"] for w in d["workspaces"]), "sessions")' \
    || echo unreachable
done
```

Broadcast a prompt across every active session on every instance:

```bash
for host in 192.168.18.27 192.168.18.25; do
  curl -fsS -X POST "http://$host:3080/dsh-lan/prompt-all" \
    -H 'content-type: application/json' -H "x-dsh-token: $DSH_LAN_TOKEN" \
    -d '{"prompt":"checkpoint: summarise your state in one line"}' | head -c 400; echo
done
```

## Configuration

| Key                        | Env                            | Default                              | Meaning                                                                                                                                            |
| -------------------------- | ------------------------------ | ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `basePath`                 | `DSH_LAN_BASE_PATH`            | `/dsh-lan`                           | Route prefix                                                                                                                                       |
| `groupKey` / `token`       | `DSH_LAN_KEY`, `DSH_LAN_TOKEN` | `tinytitan-lan`                      | Group tag **and** the secret every request presents                                                                                                |
| `peers`                    | `DSH_LAN_PEERS`                | `[]`                                 | Seed addresses (`host` or `host:port`) to try even when discovery finds nothing                                                                    |
| `discoveryIntervalSeconds` | `DSH_LAN_DISCOVERY_SECONDS`    | `60`                                 | How often the group is refreshed (minimum 5)                                                                                                       |
| `discoverTailscale`        | —                              | `true`                               | Enumerate online tailnet peers that have an IPv4 address — macOS, Linux, Windows — from the Tailscale CLI                                          |
| `discoverBonjour`          | —                              | `true`                               | Browse `_dsh-lan._tcp` through macOS `dns-sd` — browse only, and idle until something registers the service; the plugin registers none, on purpose |
| `discoverSubnet`           | —                              | `false`                              | Sweep each local `/24` on the peer port — the only source that touches hosts which never opted in                                                  |
| `peerPort`                 | —                              | `3080`                               | The port other members answer on                                                                                                                   |
| `probeTimeoutMs`           | `DSH_LAN_PROBE_TIMEOUT`        | `3000`                               | How long a peer probe waits — three seconds because a member may be on another continent                                                           |
| `discoveryConcurrency`     | —                              | `24`                                 | How many peers are probed at once (socket connects)                                                                                                |
| `resolveConcurrency`       | `DSH_LAN_RESOLVE_CONCURRENCY`  | `4`                                  | How many hostnames are resolved at once — a stale name takes the full mDNS timeout (~5 s)                                                          |
| `resolveTtlMs`             | —                              | `300000`                             | How long an answer is reused, **including a failure**: a stale Bonjour name otherwise costs ~5 s every cycle                                       |
| `allowAddresses`           | `DSH_LAN_ALLOW`                | `[]`                                 | Extra single hosts or CIDRs to admit                                                                                                               |
| `ipv4Networks`             | —                              | loopback, RFC1918, link-local, CGNAT | Replace the IPv4 allowlist                                                                                                                         |
| `ipv6Networks`             | —                              | `::1/128`, `fc00::/7`, `fe80::/10`   | Replace the IPv6 allowlist                                                                                                                         |
| `trustedOrigins`           | —                              | `[]`                                 | Extra Origins accepted                                                                                                                             |
| `allowPrivateOrigins`      | —                              | `true`                               | Accept LAN Origins on mutations                                                                                                                    |
| `enforceOrigin`            | —                              | `true`                               | Check Origin on mutations at all                                                                                                                   |
| `includeEmptyWorkspaces`   | —                              | `false`                              | Show workspaces with no visible session                                                                                                            |
| `maxBodyBytes`             | —                              | `262144`                             | Request body cap                                                                                                                                   |

## How it hangs together

| File               | Role                                                                                                    |
| ------------------ | ------------------------------------------------------------------------------------------------------- |
| `src/index.js`     | `apply()` — config, route registration, disposal, banner                                                |
| `src/router.js`    | the three guards, routing, JSON bodies and responses                                                    |
| `src/api.js`       | the operations against `workspaceRegistry` / `agents` / `sessionController` / `sessionQuery`+`sessions` |
| `src/net.js`       | the address fence (pure, no I/O)                                                                        |
| `src/discovery.js` | Tailscale, Bonjour, seeds and subnet candidates (best-effort, never throws)                             |
| `src/peers.js`     | the peer table: validate, probe, gossip, expire, on a timer                                             |
| `src/config.js`    | config and environment resolution                                                                       |

It reads harness services lazily through `ctx.get(...)` and imports no harness
internals, so a harness upgrade cannot desynchronise it. The single dynamic import
is `@deepseek-ai/dsh-llm`'s `createUserMessage`, used to build a prompt exactly as
the SDK server does; if that export moves, the plugin falls back to the equivalent
literal shape and reports which path it took in `/health`.

## Tests

```bash
npm test        # node --test 'test/*.test.js' — 107 cases
```

`test/net.test.js` is the important one: it pins every allowed range and, more to
the point, the addresses just outside each one, plus the spoofed-header case.
