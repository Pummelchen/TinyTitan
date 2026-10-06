/**
 * LAN discovery: find candidate DSH hosts without being told where to look.
 *
 * Four sources, unioned and de-duplicated. Each is best-effort — a source that
 * is missing, slow or broken contributes nothing and never throws into the
 * caller, because discovery runs on a timer for the life of the process.
 *
 * | Source | What it gives | Needs |
 * |---|---|---|
 * | `tailscale` | every online macOS peer's `100.x` address | the Tailscale CLI |
 * | `bonjour` | `_dsh-lan._tcp` instances, as hostnames | `/usr/bin/dns-sd` |
 * | `seed` | whatever the operator configured | nothing |
 * | `subnet` | the local `/24` of each interface, probed | nothing (off by default) |
 *
 * Addresses from `bonjour` are **hostnames**, not literals; the peer table
 * resolves them and validates the resolved address against the same allowlist
 * the request fence uses, so a name that resolves somewhere unexpected is
 * dropped rather than dialled.
 *
 * @module dsh-lan-manager/discovery
 */

import { execFile } from "node:child_process";
import { networkInterfaces } from "node:os";

/** The Bonjour service this plugin browses. Nothing here registers it — see TT-029. */
export const BONJOUR_SERVICE = "_dsh-lan._tcp";

/** Where the Tailscale CLI lives when it is not on `PATH`. */
export const TAILSCALE_FALLBACK = "/Applications/Tailscale.app/Contents/MacOS/Tailscale";

/** Bound on a child process, so one hung source cannot stall a cycle. */
const DEFAULT_TIMEOUT_MS = 5000;

/**
 * Run a command, resolving with whatever it printed — including when the
 * timeout killed it, which is the normal case for the streaming `dns-sd` tools.
 * @param command - executable.
 * @param args - arguments.
 * @param options - `{ timeoutMs }`.
 * @returns `{ stdout, ok }`.
 */
export function exec(file, args, { timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
  return new Promise((resolve) => {
    execFile(file, args, { timeout: timeoutMs, maxBuffer: 4 * 1024 * 1024 }, (error, stdout) => {
      resolve({ stdout: String(stdout ?? ""), ok: !error });
    });
  });
}

/**
 * Mobile platforms that never host a harness. Everything else — macOS, Linux,
 * Windows, the BSDs — is kept, because a tailnet can span continents and
 * datacenters and the harness runs on all of them. Filtering to macOS here once
 * hid every Linux box on the tailnet, which is exactly the datacenter case.
 */
const MOBILE_PLATFORMS = new Set(["iOS", "Android"]);

/**
 * Pull the online peers out of `tailscale status --json`.
 *
 * Geography is irrelevant: every tailnet peer carries a `100.x` (or `fd7a::`)
 * address whatever continent it sits on, and both are inside the fence the rest
 * of the plugin enforces. What this can *not* see is a peer whose tailnet ACL
 * blocks the peer port, or one running the harness on a different port than
 * `peerPort` — the list carries no port, so that case needs an explicit seed.
 *
 * @param source - the JSON text, or the parsed object.
 * @returns `[{address, name, source}]` with one IPv4 address per peer.
 */
export function parseTailscalePeers(source) {
  let data;
  try {
    data = typeof source === "string" ? JSON.parse(source) : source;
  } catch {
    return [];
  }
  const peers = data?.Peer;
  if (!peers || typeof peers !== "object") return [];
  const out = [];
  for (const peer of Object.values(peers)) {
    if (!peer || typeof peer !== "object") continue;
    if (MOBILE_PLATFORMS.has(String(peer.OS ?? ""))) continue;
    if (peer.Online === false) continue;
    const addresses = Array.isArray(peer.TailscaleIPs) ? peer.TailscaleIPs : [];
    const ipv4 = addresses.find((entry) => typeof entry === "string" && !entry.includes(":"));
    if (!ipv4) continue;
    out.push({
      address: ipv4,
      name: String(peer.HostName ?? peer.DNSName ?? ipv4),
      source: "tailscale",
    });
  }
  return out;
}

/**
 * Pull instance names out of `dns-sd -B` output.
 * @param text - captured stdout.
 * @returns instance names, de-duplicated.
 */
export function parseBonjourBrowse(text) {
  const names = [];
  for (const line of String(text ?? "").split("\n")) {
    //  10:41:05.123  Add        3  4 local.               _dsh-lan._tcp.       Node3
    const match = /^\s*\d[\d:.]*\s+Add\s+\S+\s+\S+\s+\S+\s+\S+\s+(.+?)\s*$/.exec(line);
    if (!match) continue;
    names.push(match[1]);
  }
  return [...new Set(names)];
}

/**
 * Pull the host and port out of `dns-sd -L` output.
 * @param text - captured stdout.
 * @returns `{host, port}` or `undefined`.
 */
export function parseBonjourResolve(text) {
  const match = /can be reached at\s+(\S+?):(\d+)/.exec(String(text ?? ""));
  if (!match) return undefined;
  const port = Number(match[2]);
  if (!Number.isInteger(port) || port <= 0) return undefined;
  return { host: match[1].replace(/\.$/, ""), port };
}

/**
 * The hosts to probe in each local `/24`.
 *
 * The scan is deliberately clamped to a /24 whatever the interface's real mask
 * is: a `/16` would be 65k connections on every cycle, and the mesh gossip means
 * one found peer is enough to learn the rest.
 *
 * @param nets - `os.networkInterfaces()` output (injectable for tests).
 * @returns dotted-quad host addresses, excluding this machine's own.
 */
export function subnetHosts(nets = networkInterfaces()) {
  const mine = new Set();
  const bases = new Set();
  for (const entries of Object.values(nets ?? {})) {
    for (const entry of entries ?? []) {
      if (!entry?.address) continue;
      if (entry.family !== "IPv4" && entry.family !== 4) continue;
      if (entry.internal) continue;
      mine.add(entry.address);
      const parts = String(entry.address).split(".").map(Number);
      if (parts.length !== 4 || parts.some((n) => !Number.isInteger(n))) continue;
      bases.add(`${parts[0]}.${parts[1]}.${parts[2]}`);
    }
  }
  const hosts = [];
  for (const base of bases) {
    for (let last = 1; last <= 254; last += 1) {
      const host = `${base}.${last}`;
      if (!mine.has(host)) hosts.push(host);
    }
  }
  return hosts;
}

/**
 * Every online macOS Tailscale peer.
 * @param options - `{ exec, timeoutMs }`.
 * @returns candidate peers, or `[]` when Tailscale is absent.
 */
export async function tailscalePeers({ exec: run = exec, timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
  const attempts = [
    ["tailscale", ["status", "--json"]],
    [TAILSCALE_FALLBACK, ["status", "--json"]],
  ];
  for (const [file, args] of attempts) {
    const { stdout } = await run(file, args, { timeoutMs });
    const peers = parseTailscalePeers(stdout);
    if (peers.length > 0) return peers;
    // An empty parse is ambiguous — no Tailscale, or a fleet with no macOS
    // peers. Only try the fallback path when the command itself failed.
    if (/[{[]/.test(stdout)) return peers;
  }
  return [];
}

/**
 * Browse for `_dsh-lan._tcp` instances and resolve each to host and port.
 *
 * `dns-sd` never exits — it streams — so each call is bounded by a timeout and
 * whatever it printed is what we parse.
 *
 * @param options - `{ exec, browseMs, resolveMs, service }`.
 * @returns `[{address, port, name, source}]`, addresses as hostnames.
 */
export async function bonjourPeers({
  exec: run = exec,
  browseMs = 2500,
  resolveMs = 1500,
  service = BONJOUR_SERVICE,
} = {}) {
  const browsed = await run("dns-sd", ["-B", service, "local."], { timeoutMs: browseMs });
  const names = parseBonjourBrowse(browsed.stdout);
  const out = [];
  for (const name of names) {
    const lookup = await run("dns-sd", ["-L", name, service, "local."], { timeoutMs: resolveMs });
    const resolved = parseBonjourResolve(lookup.stdout);
    if (!resolved || resolved.host === "localhost") continue;
    out.push({ address: resolved.host, port: resolved.port, name, source: "bonjour" });
  }
  return out;
}

/**
 * Merge the configured seeds, which may be `host` or `host:port`.
 * @param seeds - configured entries.
 * @param defaultPort - port for entries that do not name one.
 * @returns candidates.
 */
export function seedPeers(seeds = [], defaultPort) {
  const out = [];
  for (const raw of seeds) {
    const text = String(raw ?? "").trim();
    if (!text) continue;
    const match = /^\[?([^\]]+?)\]?(?::(\d+))?$/.exec(text);
    if (!match) continue;
    const port = match[2] ? Number(match[2]) : defaultPort;
    if (!Number.isInteger(port) || port <= 0) continue;
    out.push({ address: match[1], port, name: match[1], source: "seed" });
  }
  return out;
}

/**
 * Run every enabled source and union the results.
 *
 * A source that throws contributes nothing to the list — and says so, through
 * `onSourceError`, to whoever is holding the list. Both halves matter: the fleet
 * must still come up when the Tailscale binary is absent, and a manager reading
 * `/peers` has to be able to tell ten members from three members plus a probe
 * that died (AUD-133).
 *
 * @param options - `{ config, exec, interfaces, onSourceError }`.
 * @returns `[{address, port, name, source}]`, de-duplicated by address+port.
 */
export async function discoverCandidates({
  config = {},
  exec: run = exec,
  interfaces,
  onSourceError = () => {},
} = {}) {
  const port = config.peerPort;
  const found = [...seedPeers(config.peers ?? [], port)];
  const report = (source, error) =>
    onSourceError({ source, message: String(error?.message ?? error) });
  if (config.discoverTailscale !== false) {
    try {
      found.push(...(await tailscalePeers({ exec: run })).map((p) => ({ ...p, port })));
    } catch (error) {
      report("tailscale", error);
    }
  }
  if (config.discoverBonjour !== false) {
    try {
      found.push(...(await bonjourPeers({ exec: run })));
    } catch (error) {
      report("bonjour", error);
    }
  }
  if (config.discoverSubnet === true) {
    for (const address of subnetHosts(interfaces)) {
      found.push({ address, port, name: address, source: "subnet" });
    }
  }
  const seen = new Set();
  return found.filter((candidate) => {
    const key = `${candidate.address}:${candidate.port}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}
