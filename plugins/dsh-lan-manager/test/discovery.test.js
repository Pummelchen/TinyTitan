/**
 * Tests for LAN discovery. Every source is driven from canned command output, so
 * the parsing — the part that breaks when a tool changes its formatting — is
 * pinned without needing Tailscale or Bonjour to be present.
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  bonjourPeers,
  discoverCandidates,
  parseBonjourBrowse,
  parseBonjourResolve,
  parseTailscalePeers,
  seedPeers,
  subnetHosts,
  tailscalePeers,
} from "../src/discovery.js";

const TAILSCALE_JSON = JSON.stringify({
  Self: { HostName: "macbook-ab", OS: "macOS", TailscaleIPs: ["100.114.69.1"] },
  Peer: {
    a: {
      HostName: "Node3",
      OS: "macOS",
      Online: true,
      TailscaleIPs: ["100.114.69.128", "fd7a::1"],
    },
    b: { HostName: "Ternak", OS: "macOS", Online: false, TailscaleIPs: ["100.75.83.5"] },
    c: { HostName: "windows-box", OS: "windows", Online: true, TailscaleIPs: ["100.75.83.9"] },
    d: {
      HostName: "Maria",
      OS: "macOS",
      Online: true,
      DNSName: "maria.tail.ts.net.",
      TailscaleIPs: ["100.80.66.66"],
    },
    e: { HostName: "v6only", OS: "macOS", Online: true, TailscaleIPs: ["fd7a:115c:a1e0::5"] },
    f: { HostName: "fra-dc-01", OS: "linux", Online: true, TailscaleIPs: ["100.101.5.9"] },
    g: { HostName: "phone", OS: "iOS", Online: true, TailscaleIPs: ["100.99.9.9"] },
  },
});

test("tailscale peers: every online host, whatever continent or OS it runs", () => {
  // A tailnet spans datacenters, so a Linux box in Frankfurt is a member just
  // like the Mac next to you. Filtering to macOS here once hid every one of them.
  const peers = parseTailscalePeers(TAILSCALE_JSON);
  assert.deepEqual(
    peers.map((p) => p.address),
    [
      "100.114.69.128", // macOS, online
      "100.75.83.9", // windows, online
      "100.80.66.66", // macOS, reached by DNSName
      "100.101.5.9", // linux in a datacenter — the case that was missed
    ],
  );
  assert.equal(peers[0].name, "Node3");
  assert.ok(peers.every((p) => p.source === "tailscale"));
  assert.ok(!peers.some((p) => p.name === "Ternak"), "offline peers are skipped");
  assert.ok(!peers.some((p) => p.name === "v6only"), "a peer with no IPv4 is skipped");
  assert.ok(!peers.some((p) => p.name === "phone"), "iOS/Android never host a harness");
});

test("geography does not matter to the fence: every tailnet address is admitted", async () => {
  const { checkAddress } = await import("../src/net.js");
  // Tailscale addresses are CGNAT (100.64/10) or ULA (fd7a::), both inside the
  // allowlist, so a peer in another datacenter passes exactly like the Mac here.
  for (const address of [
    "100.101.5.9",
    "100.88.7.7",
    "100.114.69.128",
    "fd7a:115c:a1e0::7f01:45af",
  ]) {
    assert.equal(checkAddress(address).allowed, true, address);
  }
});

test("tailscale peers: garbage in, empty out — never a throw", () => {
  for (const input of ["", "not json", "{}", "[]", "{bad"]) {
    assert.deepEqual(parseTailscalePeers(input), []);
  }
});

test("bonjour browse parses instance names", () => {
  const text = [
    "Browsing for _dsh-lan._tcp.local.",
    "DATE: ---Wed 17 Sep 2026---",
    "10:41:02.123  ...STARTING...",
    "10:41:05.123  Add        3  4 local.               _dsh-lan._tcp.       Node3",
    "10:41:06.123  Add        3  4 local.               _dsh-lan._tcp.       macbook-ab",
    "10:41:07.123  Rmv        3  4 local.               _dsh-lan._tcp.       gone",
  ].join("\n");
  assert.deepEqual(parseBonjourBrowse(text), ["Node3", "macbook-ab"]);
});

test("bonjour resolve parses host and port", () => {
  const text = [
    "Lookup Node3._dsh-lan._tcp.local.",
    "10:41:10.123  Node3._dsh-lan._tcp.local. can be reached at Node3.local.:3080 (interface 4)",
  ].join("\n");
  assert.deepEqual(parseBonjourResolve(text), { host: "Node3.local", port: 3080 });
  assert.equal(parseBonjourResolve("nothing here"), undefined);
});

test("subnet hosts are the local /24 minus this machine", () => {
  const nets = {
    en0: [{ address: "192.168.18.27", family: "IPv4", internal: false }],
    lo0: [{ address: "127.0.0.1", family: "IPv4", internal: true }],
    utun3: [{ address: "100.114.69.1", family: "IPv4", internal: false }],
  };
  const hosts = subnetHosts(nets);
  assert.equal(hosts.length, 506, "two /24s of 254, minus each interface's own address");
  assert.ok(!hosts.includes("192.168.18.27"), "never probes itself");
  assert.ok(!hosts.includes("127.0.0.1"), "loopback is internal");
  assert.ok(hosts.includes("100.114.69.128"), "the Tailscale /24 is included too");
});

test("seeds accept host and host:port", () => {
  assert.deepEqual(seedPeers(["192.168.18.25", "node3.local:3080"], 3080), [
    { address: "192.168.18.25", port: 3080, name: "192.168.18.25", source: "seed" },
    { address: "node3.local", port: 3080, name: "node3.local", source: "seed" },
  ]);
});

test("tailscale discovery falls back to the app bundle path", async () => {
  const calls = [];
  const exec = async (file) => {
    calls.push(file);
    if (file === "tailscale") return { stdout: "", ok: false };
    return { stdout: TAILSCALE_JSON, ok: true };
  };
  const peers = await tailscalePeers({ exec });
  assert.equal(peers.length, 4);
  assert.equal(calls.length, 2, "tried PATH first, then the bundle");
});

test("bonjour discovery resolves every browsed instance", async () => {
  const exec = async (file, args) => {
    assert.equal(file, "dns-sd");
    if (args[0] === "-B") {
      return {
        stdout: "10:41:05.123  Add        3  4 local.               _dsh-lan._tcp.       Node3",
        ok: false,
      };
    }
    return {
      stdout: "Node3._dsh-lan._tcp.local. can be reached at Node3.local.:3080 (interface 4)",
      ok: false,
    };
  };
  const peers = await bonjourPeers({ exec });
  assert.deepEqual(peers, [
    { address: "Node3.local", port: 3080, name: "Node3", source: "bonjour" },
  ]);
});

test("discoverCandidates unions the enabled sources and de-duplicates", async () => {
  const exec = async (file, args) => {
    if (file === "tailscale") return { stdout: TAILSCALE_JSON, ok: true };
    if (args[0] === "-B") return { stdout: "", ok: true };
    return { stdout: "", ok: true };
  };
  const config = {
    peerPort: 3080,
    peers: ["100.114.69.128:3080", "192.168.18.25"],
    discoverTailscale: true,
    discoverBonjour: false,
    discoverSubnet: false,
  };
  const found = await discoverCandidates({ config, exec });
  const keys = found.map((c) => `${c.address}:${c.port}`);
  assert.deepEqual(keys, [
    "100.114.69.128:3080", // the seed wins the de-duplication
    "192.168.18.25:3080",
    "100.75.83.9:3080",
    "100.80.66.66:3080",
    "100.101.5.9:3080",
  ]);
});

test("discoverCandidates survives a source that blows up", async () => {
  const exec = async () => {
    throw new Error("command not found");
  };
  const found = await discoverCandidates({
    config: { peerPort: 3080, peers: ["10.0.0.5"], discoverTailscale: true, discoverBonjour: true },
    exec,
  });
  assert.deepEqual(
    found.map((c) => c.address),
    ["10.0.0.5"],
  );
});

// AUD-133: a source that dies must say so. The union is still allowed to come up
// without Tailscale installed — that is the whole reason the catch exists — but a
// caller holding the list has to be able to tell a small fleet from a dead probe.
test("a failing source is reported and contributes nothing", async () => {
  const failures = [];
  const candidates = await discoverCandidates({
    config: { peers: [], peerPort: 3080, discoverSubnet: false },
    exec: async () => {
      throw new Error("spawn Tailscale ENOENT");
    },
    onSourceError: (failure) => failures.push(failure),
  });
  assert.deepEqual(candidates, []);
  assert.deepEqual(failures.map((failure) => failure.source).sort(), ["bonjour", "tailscale"]);
  assert.ok(failures.every((failure) => failure.message.includes("ENOENT")));
});

test("a healthy run reports no source failures", async () => {
  const failures = [];
  await discoverCandidates({
    config: { peers: ["10.0.0.5"], peerPort: 3080, discoverBonjour: false, discoverSubnet: false },
    exec: async () => ({ stdout: JSON.stringify({ peers: [] }), code: 0 }),
    onSourceError: (failure) => failures.push(failure),
  });
  assert.deepEqual(failures, []);
});
