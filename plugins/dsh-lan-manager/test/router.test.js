/**
 * Router and operations tests.
 *
 * These follow the **real** derivation, which is the part that is easy to get
 * wrong: the web page's workspace groups come from the session projection (each
 * session's `cwd`), *not* from `workspaceRegistry` — on the author's machine the
 * registry holds two workspaces while the page shows twelve. So the fixture is a
 * session cache directory plus a registry that only supplies metadata, and the
 * assertions pin that the page-visible set is what comes back.
 *
 * The router's source-address fence is exercised through the real handler, since
 * a fence only tested in isolation is a fence nobody proved is wired up.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { DEFAULT_GROUP_KEY, groupDigest, resolveConfig } from "../src/config.js";
import { resolveMessageFactory, setContextOverrides } from "../src/api.js";
import {
  createHandler,
  isAllowedOrigin,
  isKnownHost,
  readJsonBody,
  subPath,
} from "../src/router.js";

/**
 * Write a session projection the way the harness does: one JSON per session with
 * `record.identity.cwd` carrying the workspace and `record.rows` carrying rows.
 */
function writeSession(dir, { id, cwd, title, createdAt, turns = 1, steps = 1 }) {
  const doc = {
    version: 7,
    record: {
      identity: { formatVersion: 3, createdAt, cwd, isSeeded: false, inheritedEventCount: 0 },
      rows: {
        title: { ver: 1, seq: 1, val: title },
        sessionStats: { ver: 1, seq: 1, val: { turns, steps } },
      },
    },
  };
  writeFileSync(join(dir, `${id}.json`), JSON.stringify(doc));
}

/** A session store with two workspaces, one archived session, one empty workspace. */
function makeStore() {
  const root = mkdtempSync(join(tmpdir(), "lan-manager-"));
  const dir = join(root, "sessions");
  mkdirSync(dir, { recursive: true });

  writeSession(dir, {
    id: "s-a1",
    cwd: "/Users/me/ProjectA",
    title: "alpha one",
    createdAt: 1000,
    turns: 3,
    steps: 9,
  });
  writeSession(dir, {
    id: "s-hidden",
    cwd: "/Users/me/ProjectA",
    title: "archived",
    createdAt: 1500,
  });
  writeSession(dir, {
    id: "s-a2",
    cwd: "/Users/me/ProjectA",
    title: "alpha two",
    createdAt: 2000,
    turns: 1,
    steps: 2,
  });
  writeSession(dir, { id: "s-b1", cwd: "/Users/me/ProjectB", title: "beta one", createdAt: 3000 });
  return { root, dir, cleanup: () => rmSync(root, { recursive: true, force: true }) };
}

/** A registry supplying metadata for only ONE of the two page-visible workspaces. */
function fakeRegistry(archiveInto) {
  const workspaces = [
    {
      id: "ws-1",
      path: "/Users/me/ProjectA",
      title: "ProjectA (pinned)",
      createdAt: "2026-09-01T00:00:00Z",
      updatedAt: "2026-09-02T00:00:00Z",
      // Accounting for the sessions that live in this path, as the real registry
      // does; an archive of an unaccounted session is refused.
      sessionIds: ["s-a1", "s-hidden", "s-a2"],
    },
    {
      id: "ws-empty",
      path: "/Users/me/RegisteredOnly",
      title: "RegisteredOnly",
      createdAt: "2026-09-05T00:00:00Z",
      updatedAt: "2026-09-05T00:00:00Z",
      sessionIds: [],
    },
  ];
  return {
    deleted: [],
    archiveCalls: [],
    list: () => workspaces.filter((w) => !w.__deleted),
    get: (id) => workspaces.find((w) => w.id === id && !w.__deleted),
    // Must mutate the same set the ctx reports, or an archive would appear
    // to do nothing — which is exactly the bug this pins.
    archiveSession: async (id) => {
      archiveInto?.add(id);
    },
    delete: async (id) => {
      const found = workspaces.find((w) => w.id === id);
      if (!found) return false;
      found.__deleted = true;
      return true;
    },
    _workspaces: workspaces,
  };
}

/** Fake agents: live sessions with delivery recorded. */
function fakeAgents(live = ["s-a1", "s-a2", "s-b1"]) {
  const delivered = [];
  const set = new Set(live);
  return {
    delivered,
    get: (id) =>
      set.has(id) ? { id, followup: (message) => delivered.push({ id, message }) } : undefined,
    add: (id) => set.add(id),
  };
}

/** Build a ctx wired to the fixture store and fakes. */
function fakeCtx({
  store,
  registry,
  agents = fakeAgents(),
  sessionController,
  sessions,
  sessionQuery,
  archived = ["s-hidden"],
}) {
  const archivedSet = new Set(archived);
  const reg = registry ?? fakeRegistry(archivedSet);
  const serviceReads = [];
  const ctx = {
    _registry: reg,
    _agents: agents,
    _archived: archivedSet,
    // Recorded because the service **name** is load-bearing: `sessions` is the raw
    // session store, and asking it to create a session is the defect TT-014 found
    // against a real harness. The name is pinned by a test below.
    _serviceReads: serviceReads,
    get(name) {
      serviceReads.push(name);
      if (name === "workspaceRegistry") return reg;
      if (name === "agents") return agents;
      if (name === "sessionController") return sessionController;
      if (name === "sessions") return sessions;
      if (name === "sessionQuery") return sessionQuery;
      return undefined;
    },
  };
  // Overrides live outside the context object: reading an undeclared property off
  // a real Cordis context throws, so the tests must not teach that pattern.
  return setContextOverrides(ctx, { sessionCacheDir: store.dir, archivedSessionIds: archivedSet });
}

/** Minimal request. */
function makeReq({
  method = "GET",
  url = "/dsh-lan/health",
  body,
  raw,
  contentType,
  headers = {},
  remote = "127.0.0.1",
}) {
  const payload =
    raw !== undefined
      ? [Buffer.from(raw)]
      : body === undefined
        ? []
        : [Buffer.from(JSON.stringify(body))];
  // Every real caller presents the group key, so tests do too — except the
  // ones testing the door itself, which pass their own header and win here.
  // A body is presented as JSON for the same reason: `FleetClient.swift:89`
  // sets the header for every body it writes, and a bodiless POST sends none.
  const typed = contentType ?? (payload.length > 0 ? "application/json" : undefined);
  return {
    method,
    url,
    headers: {
      ...(typed ? { "content-type": typed } : {}),
      "x-dsh-token": DEFAULT_GROUP_KEY,
      ...headers,
    },
    socket: { remoteAddress: remote },
    async *[Symbol.asyncIterator]() {
      for (const chunk of payload) yield chunk;
    },
  };
}

/** Minimal response capturing status and JSON. */
function makeRes() {
  const res = {
    status: undefined,
    chunks: [],
    writeHead(status) {
      res.status = status;
    },
    end(value) {
      if (value !== undefined) res.chunks.push(Buffer.from(String(value)));
    },
    get body() {
      const text = Buffer.concat(res.chunks).toString("utf8");
      try {
        return JSON.parse(text);
      } catch {
        return text;
      }
    },
  };
  return res;
}

async function call(handler, reqOptions) {
  const res = makeRes();
  await handler(makeReq(reqOptions), res);
  return res;
}

/** A handler over the fixture store with an immediately-resolved factory. */
async function setup(overrides = {}) {
  const store = overrides.store ?? makeStore();
  const ctx = fakeCtx({ store, ...overrides });
  const config = resolveConfig(overrides.config ?? {}, {});
  // `overrides.llm` stands in for the `@deepseek-ai/dsh-llm` module, so the
  // primary factory path — the one a real profile takes — is exercised too.
  const factory = await resolveMessageFactory(
    overrides.llm
      ? async () => overrides.llm
      : async () => {
          throw new Error("no dsh-llm here");
        },
  );
  const handler = createHandler({
    ctx,
    config,
    messageFactory: factory,
    log: () => {},
    peers: overrides.peers,
    self: overrides.self,
  });
  return { ctx, config, factory, handler, store };
}

test("subPath strips the prefix and tolerates a trailing slash", () => {
  assert.equal(subPath("/dsh-lan/workspaces", "/dsh-lan"), "/workspaces");
  assert.equal(subPath("/dsh-lan/workspaces/", "/dsh-lan"), "/workspaces");
  assert.equal(subPath("/dsh-lan", "/dsh-lan"), "/");
  assert.equal(subPath("/dsh-lan?x=1", "/dsh-lan"), "/");
  assert.equal(subPath("/other", "/dsh-lan"), undefined);
});

test("health reports the plugin and the caller's fence verdict", async () => {
  const { handler, store } = await setup();
  try {
    const res = await call(handler, { url: "/dsh-lan/health" });
    assert.equal(res.status, 200);
    assert.equal(res.body.ok, true);
    assert.equal(res.body.plugin, "dsh-lan-manager");
    assert.equal(res.body.source.address, "127.0.0.1");
  } finally {
    store.cleanup();
  }
});

test("a public source is refused before any work happens", async () => {
  const { handler, store } = await setup();
  try {
    const res = await call(handler, { url: "/dsh-lan/workspaces", remote: "8.8.8.8" });
    assert.equal(res.status, 403);
    assert.equal(res.body.error, "source-not-allowed");
    assert.equal(res.body.reason, "source-not-in-allowlist");
  } finally {
    store.cleanup();
  }
});

test("a Tailscale source is allowed", async () => {
  const { handler, store } = await setup();
  try {
    const res = await call(handler, { url: "/dsh-lan/workspaces", remote: "100.100.16.45" });
    assert.equal(res.status, 200);
  } finally {
    store.cleanup();
  }
});

test("workspaces come from the session projection, not the registry", async () => {
  const { handler, store } = await setup();
  try {
    const res = await call(handler, { url: "/dsh-lan/workspaces" });
    assert.equal(res.status, 200);
    assert.equal(res.body.strategy, "session-projection");
    // ProjectA and ProjectB are visible because sessions live there — even though
    // the registry pins only ProjectA.
    assert.deepEqual(res.body.workspaces.map((w) => w.path).sort(), [
      "/Users/me/ProjectA",
      "/Users/me/ProjectB",
    ]);
  } finally {
    store.cleanup();
  }
});

test("the archived session is hidden, and its count reported", async () => {
  const { handler, store } = await setup();
  try {
    const res = await call(handler, { url: "/dsh-lan/workspaces" });
    const a = res.body.workspaces.find((w) => w.path === "/Users/me/ProjectA");
    assert.deepEqual(a.sessionIds.sort(), ["s-a1", "s-a2"], "s-hidden is not visible");
    assert.equal(a.sessionCount, 2);
    assert.equal(res.body.archivedCount, 1);
  } finally {
    store.cleanup();
  }
});

test("registry metadata is layered on when the path matches", async () => {
  const { handler, store } = await setup();
  try {
    const res = await call(handler, { url: "/dsh-lan/workspaces" });
    const a = res.body.workspaces.find((w) => w.path === "/Users/me/ProjectA");
    const b = res.body.workspaces.find((w) => w.path === "/Users/me/ProjectB");
    assert.equal(a.title, "ProjectA (pinned)", "pinned title wins");
    assert.equal(a.id, "ws-1");
    assert.equal(a.registered, true);
    assert.equal(b.registered, false, "a session-only workspace is still listed");
    assert.equal(b.id, null);
    assert.equal(b.title, "ProjectB", "falls back to the folder name");
  } finally {
    store.cleanup();
  }
});

test("newest session wins the per-workspace ordering", async () => {
  const { handler, store } = await setup();
  try {
    const res = await call(handler, { url: "/dsh-lan/workspaces" });
    const a = res.body.workspaces.find((w) => w.path === "/Users/me/ProjectA");
    assert.deepEqual(a.sessionIds, ["s-a2", "s-a1"], "newest first");
  } finally {
    store.cleanup();
  }
});

test("includeEmptyWorkspaces adds a registered workspace with no session", async () => {
  const { handler, store } = await setup({ config: { includeEmptyWorkspaces: true } });
  try {
    const res = await call(handler, { url: "/dsh-lan/workspaces" });
    const paths = res.body.workspaces.map((w) => w.path);
    assert.ok(paths.includes("/Users/me/RegisteredOnly"), "empty registered workspace appears");
    const empty = res.body.workspaces.find((w) => w.path === "/Users/me/RegisteredOnly");
    assert.equal(empty.sessionCount, 0);
  } finally {
    store.cleanup();
  }
});

test("all active sessions span workspaces, archived excluded", async () => {
  const { handler, store } = await setup();
  try {
    const res = await call(handler, { url: "/dsh-lan/sessions" });
    assert.equal(res.status, 200);
    assert.equal(res.body.count, 3);
    assert.deepEqual(res.body.sessions.map((s) => s.sessionId).sort(), ["s-a1", "s-a2", "s-b1"]);
    assert.equal(res.body.sessions[0].workspacePath.startsWith("/Users/me/"), true);
  } finally {
    store.cleanup();
  }
});

test("a workspace's sessions can be listed by registry id", async () => {
  const { handler, store } = await setup();
  try {
    const res = await call(handler, { url: "/dsh-lan/workspaces/ws-1/sessions" });
    assert.equal(res.status, 200);
    assert.deepEqual(res.body.workspace.sessionIds.sort(), ["s-a1", "s-a2"]);
  } finally {
    store.cleanup();
  }
});

test("a workspace's sessions can be listed by path suffix when unregistered", async () => {
  const { handler, store } = await setup();
  try {
    const res = await call(handler, { url: "/dsh-lan/workspaces/ProjectB/sessions" });
    assert.equal(res.status, 200);
    assert.equal(res.body.workspace.path, "/Users/me/ProjectB");
    assert.deepEqual(res.body.workspace.sessionIds, ["s-b1"]);
  } finally {
    store.cleanup();
  }
});

test("an unknown workspace is a 404, not an empty list", async () => {
  const { handler, store } = await setup();
  try {
    const res = await call(handler, { url: "/dsh-lan/workspaces/nope/sessions" });
    assert.equal(res.status, 404);
    assert.equal(res.body.error, "not-found");
  } finally {
    store.cleanup();
  }
});

test("prompting a session enqueues a follow-up", async () => {
  const { handler, ctx, store } = await setup();
  try {
    const res = await call(handler, {
      method: "POST",
      url: "/dsh-lan/prompt",
      body: { sessionId: "s-a1", prompt: "run the tests" },
    });
    assert.equal(res.status, 200);
    assert.equal(res.body.delivered, true);
    assert.equal(ctx._agents.delivered.length, 1);
    const sent = ctx._agents.delivered[0].message;
    assert.equal(sent.role, "user");
    assert.equal(sent.content[0].text, "run the tests");
    assert.equal(sent.source.kind, "user", "the agent loop reads source.kind");
  } finally {
    store.cleanup();
  }
});

test("a prompt carries the user source on both factory paths", async () => {
  // Upstream's `createUserMessage` mints the id but does **not** invent a source,
  // so the caller supplies one. A prompt delivered without it dies as
  // `Cannot read properties of undefined (reading 'kind')` before any model call —
  // found by driving the route against a real harness, not by a unit test.
  const seen = [];
  const llm = {
    createUserMessage: (input) => {
      seen.push(input);
      return { id: "m-1", ...input };
    },
  };
  const primary = await setup({ llm });
  try {
    assert.equal(primary.factory.strategy, "dsh-llm:createUserMessage");
    const res = await call(primary.handler, {
      method: "POST",
      url: "/dsh-lan/prompt",
      body: { sessionId: "s-a1", prompt: "run the tests" },
    });
    assert.equal(res.status, 200);
    assert.deepEqual(seen[0].source, { kind: "user" });
    assert.deepEqual(seen[0].content, [{ type: "text", text: "run the tests" }]);
  } finally {
    primary.store.cleanup();
  }

  const fallback = await setup();
  try {
    assert.equal(fallback.factory.strategy, "inline-user-message");
    const res = await call(fallback.handler, {
      method: "POST",
      url: "/dsh-lan/prompt",
      body: { sessionId: "s-a1", prompt: "hi" },
    });
    assert.equal(res.status, 200);
    assert.equal(fallback.ctx._agents.delivered[0].message.source.kind, "user");
  } finally {
    fallback.store.cleanup();
  }
});

test("prompting a session with no live agent is a 404, and nothing is delivered", async () => {
  const { handler, ctx, store } = await setup();
  try {
    const res = await call(handler, {
      method: "POST",
      url: "/dsh-lan/prompt",
      body: { sessionId: "s-ghost", prompt: "hello" },
    });
    assert.equal(res.status, 404);
    assert.equal(ctx._agents.delivered.length, 0);
  } finally {
    store.cleanup();
  }
});

test("an empty prompt is refused", async () => {
  const { handler, store } = await setup();
  try {
    const res = await call(handler, {
      method: "POST",
      url: "/dsh-lan/prompt",
      body: { sessionId: "s-a1", prompt: "   " },
    });
    assert.equal(res.status, 400);
    assert.equal(res.body.error, "bad-request");
  } finally {
    store.cleanup();
  }
});

test("prompt-all fans out across every visible session", async () => {
  const { handler, ctx, store } = await setup();
  try {
    const res = await call(handler, {
      method: "POST",
      url: "/dsh-lan/prompt-all",
      body: { prompt: "status?" },
    });
    assert.equal(res.status, 200);
    assert.equal(res.body.delivered.length, 3);
    assert.deepEqual(res.body.failed, []);
    assert.deepEqual(ctx._agents.delivered.map((d) => d.id).sort(), ["s-a1", "s-a2", "s-b1"]);
  } finally {
    store.cleanup();
  }
});

test("prompt-all reports per-session failure instead of aborting the fan-out", async () => {
  const agents = fakeAgents(["s-a1", "s-b1"]); // s-a2 lists but has no agent
  const { handler, store } = await setup({ agents });
  try {
    const res = await call(handler, {
      method: "POST",
      url: "/dsh-lan/prompt-all",
      body: { prompt: "status?" },
    });
    assert.equal(res.status, 200, "partial success is not an HTTP error");
    assert.equal(res.body.delivered.length, 2);
    assert.equal(res.body.failed.length, 1);
    assert.equal(res.body.failed[0].sessionId, "s-a2");
    assert.equal(res.body.failed[0].code, "not-found");
  } finally {
    store.cleanup();
  }
});

test("prompt-all can be narrowed to specific sessions and capped", async () => {
  const { handler, ctx, store } = await setup();
  try {
    const narrowed = await call(handler, {
      method: "POST",
      url: "/dsh-lan/prompt-all",
      body: { prompt: "only b", sessionIds: ["s-b1"] },
    });
    assert.equal(narrowed.body.delivered.length, 1);
    ctx._agents.delivered.length = 0;
    const capped = await call(handler, {
      method: "POST",
      url: "/dsh-lan/prompt-all",
      body: { prompt: "first two", limit: 2 },
    });
    assert.equal(capped.body.delivered.length, 2);
  } finally {
    store.cleanup();
  }
});

test("archiving a session hides it without a delete", async () => {
  const { handler, store } = await setup();
  try {
    const res = await call(handler, { method: "POST", url: "/dsh-lan/sessions/s-a2/archive" });
    assert.equal(res.status, 200);
    assert.equal(res.body.archived, true);
    const after = await call(handler, { url: "/dsh-lan/workspaces" });
    const a = after.body.workspaces.find((w) => w.path === "/Users/me/ProjectA");
    assert.deepEqual(a.sessionIds, ["s-a1"], "archived row is now hidden");
    assert.equal(after.body.archivedCount, 2, "the archive set grew by one");
  } finally {
    store.cleanup();
  }
});

test("archiving an unknown session is a 404", async () => {
  const { handler, store } = await setup();
  try {
    const res = await call(handler, { method: "POST", url: "/dsh-lan/sessions/s-ghost/archive" });
    assert.equal(res.status, 404);
  } finally {
    store.cleanup();
  }
});

test("deleting a workspace archives its sessions first by default", async () => {
  const { handler, ctx, store } = await setup();
  try {
    const res = await call(handler, { method: "POST", url: "/dsh-lan/workspaces/ws-1/delete" });
    assert.equal(res.status, 200);
    assert.equal(res.body.deleted, true);
    assert.deepEqual(
      res.body.archivedSessionIds.sort(),
      ["s-a1", "s-a2", "s-hidden"],
      "every session the workspace accounted for is archived before the delete",
    );
    assert.equal(ctx._registry.get("ws-1"), undefined, "workspace is gone");
  } finally {
    store.cleanup();
  }
});

test("deleting an unknown workspace is a 404", async () => {
  const { handler, store } = await setup();
  try {
    const res = await call(handler, { method: "POST", url: "/dsh-lan/workspaces/nope/delete" });
    assert.equal(res.status, 404);
  } finally {
    store.cleanup();
  }
});

test("a configured token is required and compared exactly", async () => {
  const { handler, store } = await setup({ config: { token: "s3cret" } });
  try {
    assert.equal((await call(handler, { url: "/dsh-lan/health" })).status, 401);
    assert.equal(
      (await call(handler, { url: "/dsh-lan/health", headers: { "x-dsh-token": "nope" } })).status,
      401,
    );
    assert.equal(
      (await call(handler, { url: "/dsh-lan/health", headers: { "x-dsh-token": "s3cret" } }))
        .status,
      200,
    );
  } finally {
    store.cleanup();
  }
});

test("a foreign Origin cannot drive a mutating request", async () => {
  const { handler, store } = await setup();
  try {
    const res = await call(handler, {
      method: "POST",
      url: "/dsh-lan/prompt",
      body: { sessionId: "s-a1", prompt: "x" },
      headers: { origin: "https://evil.example", host: "192.168.1.5:3080" },
    });
    assert.equal(res.status, 403);
    assert.equal(res.body.error, "origin-not-allowed");
  } finally {
    store.cleanup();
  }
});

test("a mutating request with no Origin is the CLI's contract, and the other guards still bite", async () => {
  // Guard 3 checks an Origin only when one is present, because a browser never
  // omits it on POST (same-origin, cross-origin and `<form>` posts all carry it)
  // while `ttlanmanager` posts through URLSession, which sends no Origin at all
  // (FleetClient.swift:198). So the header-less caller is a program, and what
  // authorises it is Guard 1 and Guard 2 — which is what the two halves below
  // pin, rather than trusting the missing header to be harmless.
  const { handler, store } = await setup();
  try {
    const res = await call(handler, {
      method: "POST",
      url: "/dsh-lan/sessions/s-a1/archive",
      headers: { host: "127.0.0.1:3080" },
    });
    assert.equal(res.status, 200);
    assert.equal(res.body.ok, true);
    assert.notEqual(res.body.error, "origin-not-allowed");
  } finally {
    store.cleanup();
  }

  // The same request, from a source the address fence rejects, is still refused
  // with no Origin in play.
  const fenced = await setup();
  try {
    const res = await call(fenced.handler, {
      method: "POST",
      url: "/dsh-lan/sessions/s-a1/archive",
      headers: { host: "127.0.0.1:3080" },
      remote: "203.0.113.9",
    });
    assert.equal(res.status, 403);
    assert.equal(res.body.error, "source-not-allowed");
  } finally {
    fenced.store.cleanup();
  }

  // And the same request with a token configured is refused until the token
  // arrives, so an absent Origin never doubles as an authentication bypass.
  const tokened = await setup({ config: { token: "s3cret" } });
  try {
    assert.equal(
      (
        await call(tokened.handler, {
          method: "POST",
          url: "/dsh-lan/sessions/s-a1/archive",
          headers: { host: "127.0.0.1:3080", "x-dsh-token": "" },
        })
      ).status,
      401,
    );
    assert.equal(
      (
        await call(tokened.handler, {
          method: "POST",
          url: "/dsh-lan/sessions/s-a1/archive",
          headers: { host: "127.0.0.1:3080", "x-dsh-token": "s3cret" },
        })
      ).status,
      200,
    );
  } finally {
    tokened.store.cleanup();
  }
});

test("a same-host Origin is accepted", async () => {
  const { handler, store } = await setup();
  try {
    const res = await call(handler, {
      method: "POST",
      url: "/dsh-lan/prompt",
      body: { sessionId: "s-a1", prompt: "x" },
      headers: { origin: "http://192.168.1.5:3080", host: "192.168.1.5:3080" },
    });
    assert.equal(res.status, 200);
  } finally {
    store.cleanup();
  }
});

test("isAllowedOrigin accepts loopback and rejects public origins", () => {
  assert.equal(isAllowedOrigin("http://localhost:3080", "localhost:3080", {}), true);
  assert.equal(isAllowedOrigin("http://127.0.0.1:3080", "127.0.0.1:3080", {}), true);
  assert.equal(isAllowedOrigin("http://192.168.1.5:3080", "192.168.1.5:3080", {}), true);
  assert.equal(isAllowedOrigin("https://evil.example", "192.168.1.5:3080", {}), false);
  assert.equal(isAllowedOrigin("not a url", "x", {}), false);
  assert.equal(isAllowedOrigin("https://1.1.1.1", "192.168.1.5:3080", {}), false);
});

test("an oversized body is refused rather than buffered", async () => {
  const { handler, store } = await setup({ config: { maxBodyBytes: 64 } });
  try {
    const res = await call(handler, {
      method: "POST",
      url: "/dsh-lan/prompt",
      body: { sessionId: "s-a1", prompt: "x".repeat(500) },
    });
    assert.equal(res.status, 413);
  } finally {
    store.cleanup();
  }
});

test("malformed JSON is a 400 with a stable code", async () => {
  const { handler, store } = await setup();
  try {
    // Announced as JSON and not parseable: that is the 400 half. Announcing some
    // other media type is the 415 half, and it is refused before the bytes are
    // parsed, so a body that is not offered as JSON never reaches `JSON.parse`.
    const res = await call(handler, {
      method: "POST",
      url: "/dsh-lan/prompt",
      raw: "{not json",
      headers: { origin: "http://127.0.0.1:3080", host: "127.0.0.1:3080" },
    });
    assert.equal(res.status, 400);
    assert.equal(res.body.error, "bad-json");
  } finally {
    store.cleanup();
  }
});

test("an unknown route is a 404", async () => {
  const { handler, store } = await setup();
  try {
    assert.equal((await call(handler, { url: "/dsh-lan/nope" })).status, 404);
  } finally {
    store.cleanup();
  }
});

test("readJsonBody returns an empty object for an empty body", async () => {
  assert.deepEqual(await readJsonBody(makeReq({ method: "POST" })), {});
});

test("a missing workspaceRegistry still serves the session-derived list", async () => {
  // The registry is metadata only; its absence must not take the route down.
  const store = makeStore();
  try {
    const ctx = setContextOverrides(
      { get: () => undefined },
      { sessionCacheDir: store.dir, archivedSessionIds: new Set(["s-hidden"]) },
    );
    const config = resolveConfig({}, {});
    const factory = await resolveMessageFactory(async () => {
      throw new Error("x");
    });
    const handler = createHandler({ ctx, config, messageFactory: factory, log: () => {} });
    const res = await call(handler, { url: "/dsh-lan/workspaces" });
    assert.equal(res.status, 200);
    assert.equal(res.body.workspaces.length, 2, "still grouped from sessions");
    assert.equal(res.body.workspaces[0].registered, false);
  } finally {
    store.cleanup();
  }
});

/** A PeerTable stand-in: the router only ever reads these three things. */
function fakePeers({ peers = [] } = {}) {
  return {
    lastRefresh: 1234,
    list: () => peers,
    get: (selector) =>
      peers.find((p) => p.id === selector || p.address === selector || p.name === selector),
  };
}

test("GET /peers is the light list, with no inventories embedded", async () => {
  const peers = fakePeers({
    peers: [
      {
        id: "192.168.18.25:3080",
        address: "192.168.18.25",
        port: 3080,
        name: "node-a",
        source: "seed",
        version: "0.1.0",
        lastSeen: 99,
        rttMs: 4,
        workspaceCount: 1,
        sessionCount: 2,
        workspaces: [{ id: "w1" }],
        sessions: [{ sessionId: "s1" }],
      },
    ],
  });
  const { handler, store } = await setup({
    peers,
    self: { id: "me:3080", name: "me", addresses: ["127.0.0.1"] },
  });
  try {
    const res = await call(handler, { url: "/dsh-lan/peers" });
    assert.equal(res.status, 200);
    assert.equal(res.body.group, groupDigest(DEFAULT_GROUP_KEY));
    assert.equal(res.body.groupLabel, DEFAULT_GROUP_KEY);
    assert.equal(res.body.peers.length, 1);
    assert.equal(res.body.peers[0].sessionCount, 2);
    assert.equal(
      res.body.peers[0].sessions,
      undefined,
      "the light list carries counts, not inventories",
    );
  } finally {
    store.cleanup();
  }
});

test("GET /inventory aggregates this instance and every member", async () => {
  const peers = fakePeers({
    peers: [
      {
        id: "192.168.18.25:3080",
        address: "192.168.18.25",
        port: 3080,
        name: "node-a",
        workspaceCount: 1,
        sessionCount: 1,
        workspaces: [{ id: "ws-remote", path: "/Users/node3/ProjectX" }],
        sessions: [{ sessionId: "s-remote", workspaceId: "ws-remote" }],
      },
    ],
  });
  const { handler, store } = await setup({
    peers,
    self: { id: "me:3080", name: "me", addresses: ["127.0.0.1"] },
  });
  try {
    const res = await call(handler, { url: "/dsh-lan/inventory" });
    assert.equal(res.status, 200);
    assert.equal(res.body.self.name, "me");
    assert.ok(res.body.workspaces.length >= 2, "this instance's own workspaces are present");
    assert.ok(res.body.sessions.length >= 1);
    assert.equal(
      res.body.peers[0].workspaces[0].id,
      "ws-remote",
      "a member's inventory rides along",
    );
  } finally {
    store.cleanup();
  }
});

test("GET /peers/:id resolves a member, and 404s an unknown one", async () => {
  const peers = fakePeers({
    peers: [
      {
        id: "192.168.18.25:3080",
        address: "192.168.18.25",
        name: "node-a",
        workspaceCount: 0,
        sessionCount: 0,
      },
    ],
  });
  const { handler, store } = await setup({ peers });
  try {
    assert.equal((await call(handler, { url: "/dsh-lan/peers/192.168.18.25" })).status, 200);
    assert.equal((await call(handler, { url: "/dsh-lan/peers/192.168.18.25:3080" })).status, 200);
    assert.equal((await call(handler, { url: "/dsh-lan/peers/nobody" })).status, 404);
  } finally {
    store.cleanup();
  }
});

/** An `agents` fake whose live agent exposes a session with a derived history. */
function fakeSessionAgents(messages) {
  return {
    get: (id) => ({ id, session: { deriveMessages: () => messages }, followup: () => {} }),
  };
}

test("GET /sessions/:id/messages returns the derived history", async () => {
  const history = [
    { id: "m1", role: "user", content: [{ type: "text", text: "what is your status?" }] },
    {
      id: "m2",
      role: "assistant",
      content: [
        { type: "thinking", text: "weighing it up" },
        { type: "text", text: "all good" },
        { type: "tool-call", name: "read_file" },
      ],
    },
  ];
  const { handler, store } = await setup({ agents: fakeSessionAgents(history) });
  try {
    const res = await call(handler, { method: "GET", url: "/dsh-lan/sessions/s-1/messages" });
    assert.equal(res.status, 200);
    assert.deepEqual(
      res.body.messages.map((m) => m.role),
      ["user", "assistant"],
    );
    assert.equal(res.body.messages[0].text, "what is your status?");
    assert.equal(res.body.messages[1].text, "all good", "the answer is the text blocks");
    assert.equal(
      res.body.messages[1].reasoningChars,
      "weighing it up".length,
      "reasoning is measured, not inlined",
    );
    assert.equal(
      res.body.messages[1].otherBlocks,
      1,
      "an unrecognised block is counted, never dropped silently",
    );
    assert.deepEqual(
      { total: res.body.total, returned: res.body.returned, truncated: res.body.truncated },
      { total: 2, returned: 2, truncated: false },
    );
  } finally {
    store.cleanup();
  }
});

test("GET /sessions/:id/messages keeps the newest, and says it truncated", async () => {
  const history = Array.from({ length: 5 }, (_, i) => ({
    id: `m${i}`,
    role: "user",
    content: [{ type: "text", text: `t${i}` }],
  }));
  const { handler, store } = await setup({ agents: fakeSessionAgents(history) });
  try {
    const res = await call(handler, {
      method: "GET",
      url: "/dsh-lan/sessions/s-1/messages?limit=2",
    });
    assert.equal(res.status, 200);
    assert.deepEqual(
      res.body.messages.map((m) => m.text),
      ["t3", "t4"],
      "an audit wants the end of the conversation",
    );
    assert.deepEqual(
      { total: res.body.total, returned: res.body.returned, truncated: res.body.truncated },
      { total: 5, returned: 2, truncated: true },
    );
  } finally {
    store.cleanup();
  }
});

/**
 * A cold-read pair: `sessionQuery.readSession` returns one loaded log, and
 * `sessions.prepare` records what the plugin handed it and returns a detached
 * session deriving the given messages.
 */
function fakeColdReader(
  messages,
  loaded = { session: { id: "s-cold" }, inheritedEventCount: 0, events: [] },
) {
  const prepared = [];
  return {
    prepared,
    sessionQuery: { readSession: async () => loaded },
    sessions: {
      prepare(id, options) {
        prepared.push({ id, options });
        return { id, deriveMessages: () => messages };
      },
    },
  };
}

test("GET /sessions/:id/messages reads a cold session the store still holds", async () => {
  // TT-030: no live agent — the history comes from storage instead, through the
  // harness's own cold reader and the same deriveMessages() the live path uses.
  const loaded = {
    session: { id: "s-cold" },
    inheritedEventCount: 0,
    events: [{ type: "user/message", seq: 0 }],
  };
  const cold = fakeColdReader(
    [{ id: "m1", role: "assistant", content: [{ type: "text", text: "from storage" }] }],
    loaded,
  );
  const { handler, store } = await setup({ agents: { get: () => undefined }, ...cold });
  try {
    const res = await call(handler, { method: "GET", url: "/dsh-lan/sessions/s-cold/messages" });
    assert.equal(res.status, 200);
    assert.equal(res.body.messages[0].text, "from storage");
    assert.deepEqual(
      cold.prepared,
      [
        {
          id: "s-cold",
          options: {
            seed: loaded.events,
            meta: loaded.session,
            inheritedEventCount: 0,
            eventState: "detached",
          },
        },
      ],
      "the validated log is handed back as a detached session",
    );
  } finally {
    store.cleanup();
  }
});

test("GET /sessions/:id/messages is a 404 when storage does not know the session", async () => {
  const missing = Object.assign(new Error('session "gone" not found'), {
    code: "SESSION_QUERY_SESSION_NOT_FOUND",
  });
  const { handler, store } = await setup({
    agents: { get: () => undefined },
    sessionQuery: {
      readSession: async () => {
        throw missing;
      },
    },
    sessions: {
      prepare: () => {
        throw new Error("must not prepare a session that does not exist");
      },
    },
  });
  try {
    const res = await call(handler, { method: "GET", url: "/dsh-lan/sessions/gone/messages" });
    assert.equal(res.status, 404);
    assert.match(String(res.body.message), /no session gone/);
    assert.equal(res.body.error, "not-found");
  } finally {
    store.cleanup();
  }
});

test("GET /sessions/:id/messages says so when the profile composes no cold reader", async () => {
  const { handler, store } = await setup({ agents: { get: () => undefined } });
  try {
    const res = await call(handler, { method: "GET", url: "/dsh-lan/sessions/s-1/messages" });
    assert.equal(res.status, 503);
    assert.equal(res.body.error, "agent-service-unavailable");
    assert.match(String(res.body.message), /cold session reader/);
  } finally {
    store.cleanup();
  }
});

test("GET /sessions/:id/messages reports an agent that cannot derive history", async () => {
  const { handler, store } = await setup({ agents: { get: (id) => ({ id }) } });
  try {
    const res = await call(handler, { method: "GET", url: "/dsh-lan/sessions/s-1/messages" });
    assert.equal(res.status, 503);
    assert.equal(res.body.error, "agent-service-unavailable");
  } finally {
    store.cleanup();
  }
});

test("POST /workspaces registers a folder, and refuses startSession honestly", async () => {
  const store = makeStore();
  try {
    const created = [];
    const ctx = setContextOverrides(
      {
        get: (name) =>
          name === "workspaceRegistry"
            ? {
                list: () => [],
                create: async (path, title) => {
                  created.push({ path, title });
                  return { id: "ws-new", path, title: title ?? "New" };
                },
              }
            : undefined,
      },
      { sessionCacheDir: store.dir, archivedSessionIds: new Set() },
    );
    const config = resolveConfig({}, {});
    const factory = await resolveMessageFactory(async () => {
      throw new Error("x");
    });
    const handler = createHandler({ ctx, config, messageFactory: factory, log: () => {} });

    const ok = await call(handler, {
      method: "POST",
      url: "/dsh-lan/workspaces",
      body: { path: "/Users/me/New", title: "New" },
    });
    assert.equal(ok.status, 200);
    assert.equal(ok.body.workspaceId, "ws-new");
    assert.deepEqual(created, [{ path: "/Users/me/New", title: "New" }]);

    const missing = await call(handler, { method: "POST", url: "/dsh-lan/workspaces", body: {} });
    assert.equal(missing.status, 400, "a path is required");

    const notWired = await call(handler, {
      method: "POST",
      url: "/dsh-lan/workspaces",
      body: { path: "/Users/me/New", startSession: true },
    });
    assert.equal(notWired.status, 501, "no session controller in this context");
    assert.equal(notWired.body.error, "session-controller-unavailable");
    assert.match(notWired.body.message, /session controller/);
  } finally {
    store.cleanup();
  }
});

/** A `sessions` fake that records what the plugin delegated. */
function fakeSessions(reply) {
  const calls = [];
  return {
    calls,
    async create(request) {
      calls.push(request);
      if (reply instanceof Error) throw reply;
      return reply ?? { sessionId: "session-new", agentPreset: "standard" };
    },
  };
}

test("POST /sessions delegates to the harness session controller", async () => {
  const sessionController = fakeSessions();
  const { handler, store, ctx } = await setup({ sessionController });
  try {
    const res = await call(handler, {
      method: "POST",
      url: "/dsh-lan/sessions",
      body: { workspaceId: "ws-1", agentPreset: "qwen38" },
    });
    assert.equal(res.status, 200);
    assert.equal(res.body.sessionId, "session-new");
    assert.equal(res.body.agentPreset, "standard");
    assert.deepEqual(sessionController.calls, [{ workspaceId: "ws-1", agentPreset: "qwen38" }]);
    assert.ok(
      ctx._serviceReads.includes("sessionController"),
      "the controller service was asked for",
    );
    assert.ok(
      !ctx._serviceReads.includes("sessions"),
      "and never the raw session store of the same name",
    );
  } finally {
    store.cleanup();
  }
});

test("POST /sessions takes a path instead of an id, and needs one of them", async () => {
  const sessionController = fakeSessions();
  const { handler, store } = await setup({ sessionController });
  try {
    const byPath = await call(handler, {
      method: "POST",
      url: "/dsh-lan/sessions",
      body: { path: "/Users/me/Repo" },
    });
    assert.equal(byPath.status, 200);
    assert.deepEqual(
      sessionController.calls,
      [{ cwd: "/Users/me/Repo" }],
      "a path is handed to the controller as cwd",
    );

    const neither = await call(handler, { method: "POST", url: "/dsh-lan/sessions", body: {} });
    assert.equal(neither.status, 400);
    assert.equal(neither.body.error, "bad-request");
    assert.equal(
      sessionController.calls.length,
      1,
      "nothing was delegated for a request with no target",
    );
  } finally {
    store.cleanup();
  }
});

test("POST /sessions keeps the controller's own failure code and status", async () => {
  const failure = Object.assign(new Error('workspace "ws-x" not found'), {
    code: "workspace/not-found",
  });
  const sessionController = fakeSessions(failure);
  const { handler, store } = await setup({ sessionController });
  try {
    const res = await call(handler, {
      method: "POST",
      url: "/dsh-lan/sessions",
      body: { workspaceId: "ws-x" },
    });
    assert.equal(res.status, 404, "a missing workspace is not the caller's bad request");
    assert.equal(res.body.error, "workspace/not-found");
  } finally {
    store.cleanup();
  }
});

test("POST /workspaces with startSession returns the workspace and its session", async () => {
  const sessionController = fakeSessions({ sessionId: "session-on-ws", agentPreset: "qwen38" });
  const registry = {
    list: () => [],
    get: () => undefined,
    create: async (path, title) => ({ id: "ws-new", path, title: title ?? "New" }),
  };
  const { handler, store } = await setup({ sessionController, registry });
  try {
    const res = await call(handler, {
      method: "POST",
      url: "/dsh-lan/workspaces",
      body: { path: "/Users/me/New", startSession: true },
    });
    assert.equal(res.status, 200);
    assert.equal(res.body.workspaceId, "ws-new");
    assert.deepEqual(res.body.session, { sessionId: "session-on-ws", agentPreset: "qwen38" });
    assert.deepEqual(
      sessionController.calls,
      [{ workspaceId: "ws-new" }],
      "started on the workspace just created",
    );
  } finally {
    store.cleanup();
  }
});

// --- AUD-147: the content-type layer is real ----------------------------------
//
// AUD-132's note recorded two mitigations for cross-origin writes: the Origin
// guard and `content-type: application/json`, which a simple form post cannot
// set. Only the first existed. These pin the second, and they set a *passing*
// Origin so the refusal cannot be credited to Guard 3 — the point of the layer is
// that it holds when the Origin guard is switched off.

test("a form post with JSON in it is refused before anything is parsed", async () => {
  const { handler, store } = await setup();
  try {
    const res = await call(handler, {
      method: "POST",
      url: "/dsh-lan/prompt",
      // Exactly what `<form enctype="text/plain">` sends: the body verbatim, and
      // a content type a cross-origin page may set without a preflight.
      raw: '{"sessionId":"s-a1","prompt":"x"}',
      contentType: "text/plain;charset=UTF-8",
      headers: { origin: "http://127.0.0.1:3080", host: "127.0.0.1:3080" },
    });
    assert.equal(res.status, 415);
    assert.equal(res.body.error, "unsupported-media-type");
  } finally {
    store.cleanup();
  }
});

test("a urlencoded post is refused the same way", async () => {
  const { handler, store } = await setup();
  try {
    const res = await call(handler, {
      method: "POST",
      url: "/dsh-lan/prompt",
      raw: "sessionId=s-a1&prompt=x",
      contentType: "application/x-www-form-urlencoded",
      headers: { origin: "http://127.0.0.1:3080", host: "127.0.0.1:3080" },
    });
    assert.equal(res.status, 415);
  } finally {
    store.cleanup();
  }
});

test("json with parameters on the media type is accepted", async () => {
  const { handler, store } = await setup();
  try {
    const res = await call(handler, {
      method: "POST",
      url: "/dsh-lan/prompt",
      raw: '{"sessionId":"s-a1","prompt":"x"}',
      contentType: "application/json; charset=utf-8",
      headers: { origin: "http://127.0.0.1:3080", host: "127.0.0.1:3080" },
    });
    assert.equal(res.status, 200);
    assert.equal(res.body.ok, true);
  } finally {
    store.cleanup();
  }
});

// --- AUD-148: same-origin is a host the server knows, not an equality ----------

test("a rebinding same-host Origin is refused", async () => {
  const { handler, store } = await setup();
  try {
    // The classic shape: `attacker.example` resolves to this machine, so the
    // browser sends an Origin that equals the Host it was told to use. Equality
    // alone called that same-origin and opened every mutating route.
    const res = await call(handler, {
      method: "POST",
      url: "/dsh-lan/prompt",
      body: { sessionId: "s-a1", prompt: "x" },
      headers: {
        origin: "http://attacker.example:3080",
        host: "attacker.example:3080",
      },
    });
    assert.equal(res.status, 403);
    assert.equal(res.body.error, "origin-not-allowed");
  } finally {
    store.cleanup();
  }
});

test("a host the operator names is trusted, and only that name", async () => {
  const { handler, store } = await setup({ config: { trustedHosts: ["mordor.local"] } });
  try {
    const allowed = await call(handler, {
      method: "POST",
      url: "/dsh-lan/prompt",
      body: { sessionId: "s-a1", prompt: "x" },
      headers: { origin: "http://mordor.local:3080", host: "mordor.local:3080" },
    });
    assert.equal(allowed.status, 200);
    const other = await call(handler, {
      method: "POST",
      url: "/dsh-lan/prompt",
      body: { sessionId: "s-a1", prompt: "x" },
      headers: { origin: "http://elsewhere.example:3080", host: "elsewhere.example:3080" },
    });
    assert.equal(other.status, 403);
  } finally {
    store.cleanup();
  }
});

test("isKnownHost names loopback and the fence, and nothing that merely repeats itself", () => {
  assert.equal(isKnownHost("127.0.0.1:3080", {}), true);
  assert.equal(isKnownHost("localhost:3080", {}), true);
  assert.equal(isKnownHost("[::1]:3080", {}), true);
  assert.equal(isKnownHost("192.168.1.5:3080", {}), true);
  assert.equal(isKnownHost("169.254.9.9", {}), true);
  assert.equal(isKnownHost("attacker.example:3080", {}), false);
  assert.equal(isKnownHost("", {}), false);
  assert.equal(isKnownHost(":3080", {}), false);
  assert.equal(isKnownHost("attacker.example:3080", { trustedHosts: ["attacker.example"] }), true);
  // The flag that narrows origins narrows the host set with it, or the two
  // predicates would disagree about what "my own host" means.
  assert.equal(isKnownHost("192.168.1.5:3080", { allowPrivateOrigins: false }), false);
  assert.equal(isAllowedOrigin("http://attacker.example:3080", "attacker.example:3080", {}), false);
  assert.equal(
    isAllowedOrigin("http://mordor.local:3080", "mordor.local:3080", {
      trustedHosts: ["mordor.local"],
    }),
    true,
  );
});

// --- AUD-123: the shipped default key is a tag, not a door ---------------------

test("the published default group key does not open a mutating route beyond loopback", async () => {
  const { handler, store } = await setup();
  try {
    // Inside the fence (link-local is in the defaults) and carrying the shipped
    // key, so Guard 1 and Guard 2 both pass — and both are nominal here: the
    // address range is one the caller may share, and the key is in the source.
    const res = await call(handler, {
      method: "POST",
      url: "/dsh-lan/sessions/s-a1/archive",
      headers: { host: "169.254.9.9:3080" },
      remote: "169.254.9.9",
    });
    assert.equal(res.status, 403);
    assert.equal(res.body.error, "default-group-key-outside-loopback");

    // A read of the group from the same caller is unaffected: an unconfigured
    // fleet still has to be listed before anyone can decide what to do with it.
    const listed = await call(handler, {
      url: "/dsh-lan/peers",
      headers: { host: "169.254.9.9:3080" },
      remote: "169.254.9.9",
    });
    assert.equal(listed.status, 200);
  } finally {
    store.cleanup();
  }

  // The same request succeeds once the key is one an operator chose.
  const keyed = await setup({ config: { token: "s3cret" } });
  try {
    const res = await call(keyed.handler, {
      method: "POST",
      url: "/dsh-lan/sessions/s-a1/archive",
      headers: { host: "169.254.9.9:3080", "x-dsh-token": "s3cret" },
      remote: "169.254.9.9",
    });
    assert.equal(res.status, 200);
  } finally {
    keyed.store.cleanup();
  }

  // And on loopback the documented single-user trade still holds.
  const local = await setup();
  try {
    const res = await call(local.handler, {
      method: "POST",
      url: "/dsh-lan/sessions/s-a1/archive",
      headers: { host: "127.0.0.1:3080" },
    });
    assert.equal(res.status, 200);
  } finally {
    local.store.cleanup();
  }
});

// --- AUD-155: a self-assigned source with the shipped key gets the group ------
//
// `169.254/16` and `fe80::/10` are inside the fence because a fleet is meant to
// need no setup, and they are also where a host lands when nothing assigned it an
// address. Membership there is claimed, so with the published key it must not
// extend to another machine's prompts.
test("the shipped default key buys a link-local source membership, not content", async () => {
  const membership = ["/dsh-lan/health", "/dsh-lan/peers"];
  const content = [
    "/dsh-lan/inventory",
    "/dsh-lan/workspaces",
    "/dsh-lan/sessions",
    "/dsh-lan/sessions/s-a1/messages",
    "/dsh-lan/workspaces/w-1/sessions",
    "/dsh-lan/peers/192.168.18.25%3A3080",
  ];
  const { handler, store } = await setup();
  try {
    for (const remote of ["169.254.9.9", "fe80::1%en0"]) {
      for (const url of membership) {
        const listed = await call(handler, { url, remote, headers: { host: `${remote}:3080` } });
        assert.equal(listed.status, 200, `${url} from ${remote}: this is how a member is found`);
      }
      for (const url of content) {
        const denied = await call(handler, { url, remote, headers: { host: `${remote}:3080` } });
        assert.equal(denied.status, 403, `${url} from ${remote} is content`);
        assert.equal(denied.body.error, "default-group-key-link-local-source");
      }
    }
  } finally {
    store.cleanup();
  }

  // A source in a range the operator's network actually owns is untouched: this is
  // the tailnet manager's inventory read, still working with the shipped key.
  const tailnet = await setup();
  try {
    const res = await call(tailnet.handler, {
      url: "/dsh-lan/inventory",
      remote: "100.64.9.9",
      headers: { host: "100.64.9.9:3080" },
    });
    assert.equal(res.status, 200);
  } finally {
    tailnet.store.cleanup();
  }

  // And the directly-coupled pair reads the same content once the key is one an
  // operator chose — which is the whole suggestion the README makes anyway.
  const keyed = await setup({ config: { token: "s3cret" } });
  try {
    const res = await call(keyed.handler, {
      url: "/dsh-lan/inventory",
      remote: "169.254.9.9",
      headers: { host: "169.254.9.9:3080", "x-dsh-token": "s3cret" },
    });
    assert.equal(res.status, 200);
  } finally {
    keyed.store.cleanup();
  }
});

// --- AUD-154: the key that opens the API is not what the API says out loud ----
test("a group-facing body names the group by digest, never by the key", async () => {
  const secret = "a-key-nobody-published";
  const { handler, store } = await setup({ config: { groupKey: secret } });
  try {
    for (const url of ["/dsh-lan/health", "/dsh-lan/peers", "/dsh-lan/inventory"]) {
      const res = await call(handler, { url, headers: { "x-dsh-token": secret } });
      assert.equal(res.status, 200);
      assert.equal(res.body.group, groupDigest(secret), `${url} names the group by digest`);
      assert.equal(res.body.groupLabel, undefined, `${url} has no public label to give`);
      assert.equal(
        JSON.stringify(res.body).includes(secret),
        false,
        `${url} repeated the group key in its body`,
      );
    }
  } finally {
    store.cleanup();
  }

  // The shipped default keeps its readable name, because it is a published
  // literal and an unconfigured fleet identifies itself by it on screen.
  const fresh = await setup();
  try {
    const res = await call(fresh.handler, { url: "/dsh-lan/health" });
    assert.equal(res.body.group, groupDigest(DEFAULT_GROUP_KEY));
    assert.equal(res.body.groupLabel, DEFAULT_GROUP_KEY);
  } finally {
    fresh.store.cleanup();
  }
});

// The digest is the group's public name, so its two boring properties are load
// bearing: every instance must arrive at the same one, and two keys must not.
test("groupDigest is stable, key-specific, and absent with no key", () => {
  assert.equal(groupDigest("tinytitan-lan"), groupDigest("tinytitan-lan"));
  assert.notEqual(groupDigest("tinytitan-lan"), groupDigest("tinytitan-lam"));
  assert.equal(groupDigest("tinytitan-lan").length, 16);
  assert.equal(/^[0-9a-f]{16}$/.test(groupDigest("anything")), true);
  for (const empty of [null, undefined, ""]) {
    assert.equal(groupDigest(empty), null, "no key means no group to name");
  }
  // Resolution carries it alongside the key, so a body can name the group without
  // the caller having to know the secret to render a banner.
  const resolved = resolveConfig({ groupKey: "chosen-by-operator" }, {});
  assert.equal(resolved.groupDigest, groupDigest("chosen-by-operator"));
  assert.equal(resolved.groupLabel, null);
  const shipped = resolveConfig({}, {});
  assert.equal(shipped.groupDigest, groupDigest(DEFAULT_GROUP_KEY));
  assert.equal(shipped.groupLabel, DEFAULT_GROUP_KEY);
});

// AUD-133, the reader's half: the reason has to reach whoever lists the fleet.
test("a degraded discovery shows up on the peer list and on health", async () => {
  const errors = [{ source: "tailscale", message: "spawn Tailscale ENOENT" }];
  const peers = { list: () => [], lastRefresh: 1234, lastDiscoveryErrors: errors };
  const { handler, store } = await setup({ peers });
  try {
    const listed = await call(handler, { url: "/dsh-lan/peers" });
    assert.deepEqual(listed.body.discoveryErrors, errors);
    const health = await call(handler, { url: "/dsh-lan/health" });
    assert.deepEqual(health.body.discoveryErrors, errors);
  } finally {
    store.cleanup();
  }

  // With no table at all the field is an empty list, not a missing key: "no
  // failures" and "I did not look" read differently to a manager.
  const bare = await setup();
  try {
    const listed = await call(bare.handler, { url: "/dsh-lan/peers" });
    assert.deepEqual(listed.body.discoveryErrors, []);
  } finally {
    bare.store.cleanup();
  }
});
