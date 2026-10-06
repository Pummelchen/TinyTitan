/**
 * Tests for the peer table. The two properties that matter are pinned here:
 * nothing is dialled before it validates against the fence's allowlist, and a
 * peer's gossip can only ever add *candidates* for the next cycle.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { createServer } from "node:http";

import { groupDigest } from "../src/config.js";
import {
  MAX_GOSSIP_ENTRIES,
  MAX_PEER_RESPONSE_BYTES,
  PeerTable,
  discoveryDelayMs,
  httpJson,
  mapLimit,
  peerKey,
  validateCandidate,
} from "../src/peers.js";

const CONFIG = {
  basePath: "/dsh-lan",
  peerPort: 3080,
  groupKey: "tinytitan-lan",
  token: "tinytitan-lan",
};

/** A resolve() stub mapping names to addresses. */
const resolver = (table) => async (name) => {
  const address = table[name];
  if (!address) throw new Error(`ENOTFOUND ${name}`);
  return { address };
};

test("validateCandidate admits an address inside the allowlist", async () => {
  const ok = await validateCandidate(
    { address: "192.168.18.25", port: 3080, source: "seed" },
    { config: CONFIG },
  );
  assert.deepEqual(ok, {
    address: "192.168.18.25",
    port: 3080,
    name: "192.168.18.25",
    source: "seed",
  });
});

test("validateCandidate judges a hostname by what it resolves to, not by its name", async () => {
  const resolve = resolver({ "Node3.local": "100.114.69.128", "evil.example": "8.8.8.8" });
  const mine = await validateCandidate(
    { address: "Node3.local", port: 3080, source: "bonjour" },
    { config: CONFIG, resolve },
  );
  assert.equal(mine.address, "100.114.69.128", "a Tailscale peer resolves into the allowlist");
  const hostile = await validateCandidate(
    { address: "evil.example", port: 3080, source: "gossip:x" },
    { config: CONFIG, resolve },
  );
  assert.equal(hostile, undefined, "a name pointing at the public internet is dropped");
});

test("validateCandidate rejects unresolvable names, bad ports and unparseable addresses", async () => {
  const resolve = resolver({});
  for (const candidate of [
    { address: "nowhere.local", port: 3080 },
    { address: "10.0.0.5", port: 0 },
    { address: "10.0.0.5", port: 99999 },
    { address: "", port: 3080 },
  ]) {
    assert.equal(
      await validateCandidate(candidate, { config: CONFIG, resolve }),
      undefined,
      JSON.stringify(candidate),
    );
  }
});

test("mapLimit keeps input order and never exceeds its bound", async () => {
  let active = 0;
  let peak = 0;
  const items = Array.from({ length: 40 }, (_, i) => i);
  const results = await mapLimit(items, 8, async (item) => {
    active += 1;
    peak = Math.max(peak, active);
    await new Promise((resolve) => setTimeout(resolve, 1));
    active -= 1;
    return item * 2;
  });
  assert.deepEqual(
    results,
    items.map((i) => i * 2),
  );
  assert.ok(peak <= 8, `peak concurrency ${peak} must not exceed 8`);
});

test("a peer's gossip only adds candidates, and never this machine", () => {
  const table = new PeerTable({ config: CONFIG, self: { id: "me", addresses: ["192.168.18.27"] } });
  const kept = table.mergeGossip(
    [
      { address: "192.168.18.25", port: 3080 },
      { address: "192.168.18.27", port: 3080 }, // ourselves — ignored
      { address: "192.168.18.25", port: 3080 }, // duplicate — counted once
      { address: "", port: 3080 },
    ],
    { from: "192.168.18.30:3080" },
  );
  assert.equal(kept, 1);
  assert.equal(table.gossip.size, 1);
  assert.equal(table.peers.size, 0, "gossip alone never creates a member");
  assert.equal(
    table.gossip.get(peerKey("192.168.18.25", 3080)).source,
    "gossip:192.168.18.30:3080",
  );
});

test("gossip is capped so a hostile member cannot grow the table without bound", () => {
  const table = new PeerTable({ config: CONFIG });
  const flood = Array.from({ length: MAX_GOSSIP_ENTRIES + 50 }, (_, i) => ({
    address: `10.0.${Math.floor(i / 254)}.${(i % 254) + 1}`,
    port: 3080,
  }));
  table.mergeGossip(flood, { from: "hostile" });
  assert.equal(table.gossip.size, MAX_GOSSIP_ENTRIES);
});

test("refresh records a member's inventory, merges its gossip, and prunes the stale", async () => {
  let clock = 1_000_000;
  let cycles = 0;
  const table = new PeerTable({
    config: CONFIG,
    self: { id: "me", addresses: ["192.168.18.27"] },
    now: () => clock,
    ttlMs: 1000,
    // The member answers only on the first cycle, so the second one can show
    // what happens to a member discovery no longer offers.
    discovery: async () =>
      cycles++ === 0
        ? [
            { address: "192.168.18.25", port: 3080, name: "node-a", source: "seed" },
            { address: "192.168.18.26", port: 3080, name: "node-b", source: "seed" },
            { address: "192.168.18.27", port: 3080, name: "me", source: "seed" },
            { address: "8.8.8.8", port: 3080, name: "public", source: "seed" },
          ]
        : [],
    fetch: async ({ address }) => {
      if (address === "192.168.18.26") return { status: 401, body: { error: "unauthorized" } };
      if (address !== "192.168.18.25") return { status: 0, body: undefined };
      return {
        status: 200,
        body: {
          ok: true,
          group: "tinytitan-lan",
          self: { name: "node-a", addresses: ["192.168.18.25"], version: "0.1.0" },
          workspaces: [{ id: "w1", path: "/Users/me/ProjectA" }],
          sessions: [{ sessionId: "s1", workspaceId: "w1" }],
          peers: [{ address: "100.114.69.128", port: 3080, name: "Node3" }],
        },
      };
    },
  });

  const list = await table.refresh();
  assert.equal(list.length, 1, "one member: the 401 and the public address are not members");
  assert.equal(list[0].workspaceCount, 1);
  assert.equal(list[0].sessionCount, 1);
  assert.equal(list[0].name, "node-a");
  assert.equal(table.gossip.size, 1, "the member's gossip became a candidate for the next cycle");
  assert.equal(table.gossip.has(peerKey("100.114.69.128", 3080)), true);

  clock += 5000;
  const after = await table.refresh();
  assert.equal(after.length, 0, "a member discovery no longer offers, past the TTL, drops out");
});

test("a member reporting another group is refused even when it answers 200", async () => {
  const table = new PeerTable({
    config: CONFIG,
    discovery: async () => [{ address: "192.168.18.25", port: 3080 }],
    fetch: async () => ({
      status: 200,
      body: { ok: true, group: "someone-elses-group", self: { name: "x" } },
    }),
  });
  assert.equal((await table.refresh()).length, 0);
});

// AUD-154: a member now names its group by digest. Every spelling that a peer on
// an older build can still send stays accepted, because a rollout that emptied the
// peer table would be a worse outage than the plaintext it removed.
test("the group matches by digest, by the old literal, and by the default label", async () => {
  const digest = groupDigest("tinytitan-lan");
  const cases = [
    ["the digest", { group: digest }],
    ["a peer still sending the literal", { group: "tinytitan-lan" }],
    ["a digest with the published label", { group: digest, groupLabel: "tinytitan-lan" }],
    ["no group field at all", {}],
  ];
  for (const [name, group] of cases) {
    const table = new PeerTable({
      config: { ...CONFIG, groupDigest: digest },
      discovery: async () => [{ address: "192.168.18.25", port: 3080 }],
      fetch: async () => ({
        status: 200,
        body: { ok: true, self: { name: "node-a" }, workspaces: [], sessions: [], ...group },
      }),
    });
    assert.equal((await table.refresh()).length, 1, `${name} is one of ours`);
  }
});

test("a foreign group string is refused without being copied into our log", async () => {
  const lines = [];
  const table = new PeerTable({
    config: { ...CONFIG, groupKey: "ours", token: "ours", groupDigest: groupDigest("ours") },
    log: (line) => lines.push(line),
    discovery: async () => [{ address: "192.168.18.25", port: 3080 }],
    fetch: async () => ({
      status: 200,
      body: { ok: true, group: "their-secret-literal", self: { name: "x" } },
    }),
  });
  assert.equal((await table.refresh()).length, 0);
  assert.match(lines.join("\n"), /reports a group that is not ours/);
  assert.equal(
    lines.join("\n").includes("their-secret-literal"),
    false,
    "a peer's group string is that peer's credential, not a diagnostic",
  );
});

test("get resolves by id, address or name", async () => {
  const table = new PeerTable({
    config: CONFIG,
    discovery: async () => [
      { address: "192.168.18.25", port: 3080, name: "node-a", source: "seed" },
    ],
    fetch: async () => ({
      status: 200,
      body: {
        ok: true,
        group: "tinytitan-lan",
        self: { name: "node-a" },
        workspaces: [],
        sessions: [],
        peers: [],
      },
    }),
  });
  await table.refresh();
  assert.equal(table.get("192.168.18.25:3080").name, "node-a");
  assert.equal(table.get("192.168.18.25").name, "node-a");
  assert.equal(table.get("node-a").name, "node-a");
  assert.equal(table.get("nobody"), undefined);
});

test("a refresh cycle that throws in discovery still returns the table", async () => {
  const table = new PeerTable({
    config: CONFIG,
    discovery: async () => {
      throw new Error("tailscale exploded");
    },
  });
  assert.deepEqual(await table.refresh(), []);
});

/** A group read that finds nothing, so only validation is exercised. */
const nullFetch = async () => ({ status: 0, body: undefined });

/** Hostname candidates, so every one of them needs a resolve. */
function hostCandidates(count) {
  return Array.from({ length: count }, (_, i) => ({
    address: `node-${i}.local`,
    port: 3080,
    name: `node-${i}`,
    source: "bonjour",
  }));
}

test("the jittered interval stays within 10% and never goes non-positive", () => {
  assert.equal(discoveryDelayMs(60, 0.5), 60_000, "the middle of the range is the interval");
  assert.equal(discoveryDelayMs(60, 0), 54_000, "the low edge is -10%");
  assert.equal(discoveryDelayMs(60, 1), 66_000, "the high edge is +10%");
  for (const random of [0, 0.1, 0.37, 0.9, 1]) {
    const delay = discoveryDelayMs(5, random);
    assert.ok(delay >= 4_500 && delay <= 5_500, `${delay}ms for random=${random}`);
    assert.ok(delay > 0);
  }
  assert.equal(discoveryDelayMs(0, 0.5), 60_000, "a nonsense interval falls back to a minute");
});

test("a hostname is resolved once and reused, so the threadpool is not re-hit", async () => {
  let calls = 0;
  let clock = 1_000_000;
  const table = new PeerTable({
    config: { ...CONFIG, resolveConcurrency: 4, resolveTtlMs: 60_000 },
    now: () => clock,
    discovery: async () => hostCandidates(3),
    resolve: async (name) => {
      calls += 1;
      return { address: "192.168.18.9", name };
    },
    fetch: nullFetch,
  });
  await table.refresh();
  assert.equal(calls, 3, "one lookup per unique name");
  await table.refresh();
  assert.equal(calls, 3, "the second cycle reuses the cached answers");
  clock += 120_000;
  await table.refresh();
  assert.equal(calls, 6, "and re-resolves once the TTL has passed");
});

test("hostname resolution runs on its own, shallower limit", async () => {
  let active = 0;
  let peak = 0;
  const table = new PeerTable({
    config: { ...CONFIG, resolveConcurrency: 2 },
    discovery: async () => hostCandidates(12),
    resolve: async (name) => {
      active += 1;
      peak = Math.max(peak, active);
      await new Promise((resolve) => setTimeout(resolve, 2));
      active -= 1;
      return { address: "192.168.18.9", name };
    },
    fetch: nullFetch,
  });
  await table.refresh();
  assert.ok(
    peak <= 2,
    `peak concurrent resolutions ${peak} must not exceed resolveConcurrency (2)`,
  );
  assert.ok(peak >= 1);
});

test("a name that does not resolve is cached too, so it is not retried every cycle", async () => {
  // Measured: a stale Bonjour name burns the full mDNS timeout (~5 s) on every
  // attempt. Caching the failure is what stops that repeating every cycle.
  let calls = 0;
  let clock = 1_000_000;
  const table = new PeerTable({
    config: { ...CONFIG, resolveConcurrency: 4, resolveTtlMs: 60_000 },
    now: () => clock,
    discovery: async () => [{ address: "gone.local", port: 3080, name: "gone", source: "bonjour" }],
    resolve: async () => {
      calls += 1;
      throw new Error("ENOTFOUND gone.local");
    },
    fetch: nullFetch,
  });
  assert.deepEqual(await table.refresh(), [], "a name that does not resolve is not a member");
  assert.equal(calls, 1);
  await table.refresh();
  assert.equal(calls, 1, "the second cycle reuses the failure instead of waiting again");
  clock += 120_000;
  await table.refresh();
  assert.equal(calls, 2, "and tries again once the TTL has passed");
});

test("a failed discovery source is kept on the table, not only in the log", async () => {
  const lines = [];
  const table = new PeerTable({
    config: CONFIG,
    log: (line) => lines.push(line),
    discovery: async ({ onSourceError }) => {
      onSourceError({ source: "tailscale", message: "spawn Tailscale ENOENT" });
      return [];
    },
  });
  await table.refresh();
  assert.deepEqual(table.lastDiscoveryErrors, [
    { source: "tailscale", message: "spawn Tailscale ENOENT" },
  ]);
  assert.ok(
    lines.some((line) => line.includes("discovery source tailscale failed")),
    `the failure was not logged: ${lines.join(" | ")}`,
  );
});

test("a discovery call that throws is recorded as the whole cycle failing", async () => {
  const table = new PeerTable({
    config: CONFIG,
    log: () => {},
    discovery: async () => {
      throw new Error("resolver wedged");
    },
  });
  await table.refresh();
  assert.deepEqual(table.lastDiscoveryErrors, [
    { source: "discovery", message: "resolver wedged" },
  ]);
});

// AUD-168: `probe` dials every candidate the fence admits and buffered whatever
// came back, so a host inside it -- and the default fence includes the
// self-assigned range -- could size this process's memory by streaming at it.
// `{ok: true}` is all it takes to be believed a member, because the group field is
// how a peer says which fleet it is in, not a credential. A real loopback server is
// required: the guard is inside the response reader, which a stubbed `fetch` cannot
// reach.
test("an answer over the cap is dropped, and one under it is parsed", async (t) => {
  const server = createServer((req, res) => {
    res.on("error", () => {});
    req.socket.on("error", () => {});
    if (req.url === "/under") {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ ok: true }));
      return;
    }
    // Never `end` the body: the point is that the reader stops it, and a peer that
    // finished first would resolve through the parse path instead of the cap.
    res.writeHead(200, { "content-type": "application/json" });
    const chunk = Buffer.alloc(256 * 1024, 0x61);
    const pump = () => {
      if (res.destroyed || res.writableEnded) return;
      res.write(chunk);
      setTimeout(pump, 2);
    };
    pump();
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const port = server.address().port;
  t.after(
    () =>
      new Promise((resolve) => {
        server.closeAllConnections?.();
        server.close(resolve);
      }),
  );

  const under = await httpJson({ address: "127.0.0.1", port, path: "/under" });
  assert.equal(under.status, 200);
  assert.deepEqual(under.body, { ok: true }, "a normal inventory still parses");

  // The deadline is part of the proof, not decoration. `httpJson`'s own `timeout`
  // is a socket *inactivity* timeout, and a peer that writes every two
  // milliseconds is never inactive, so with the byte bound removed nothing in the
  // reader ends this request: measured on the unfixed code it simply hangs, which
  // is the second half of the defect (a peer inside the fence can hold a probe open
  // for as long as it likes). Racing it against a clock turns that into a failure.
  let arm;
  const deadline = new Promise((resolve) => {
    arm = () =>
      resolve({
        status: -1,
        error: "the reader never stopped an answer that does not end",
      });
  });
  const clock = setTimeout(arm, 8000);
  // `unref` so a passed test is not held open for the rest of the deadline, and
  // cleared below so the timer cannot outlive the request it guards.
  clock.unref();
  const over = await Promise.race([
    httpJson({ address: "127.0.0.1", port, path: "/over" }),
    deadline,
  ]);
  clearTimeout(clock);
  assert.equal(over.status, 0, "an over-cap answer is not a peer");
  assert.match(
    String(over.error),
    new RegExp(`exceeds ${MAX_PEER_RESPONSE_BYTES} bytes`),
    "and it says which rule cut it, naming the cap that is exported",
  );
});

test("a clean cycle clears the previous cycle's failures", async () => {
  let broken = true;
  const table = new PeerTable({
    config: CONFIG,
    log: () => {},
    discovery: async ({ onSourceError }) => {
      if (broken) {
        onSourceError({ source: "bonjour", message: "dns-sd exited 1" });
      }
      broken = false;
      return [];
    },
  });
  await table.refresh();
  assert.equal(table.lastDiscoveryErrors.length, 1);
  await table.refresh();
  assert.deepEqual(table.lastDiscoveryErrors, []);
});
