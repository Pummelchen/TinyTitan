/**
 * The peer table: the group's member list, with each member's inventory.
 *
 * Discovery finds *candidates*; this table decides which of them are really
 * group members, remembers what each one holds, and keeps the set fresh on a
 * timer. Two rules shape it:
 *
 * 1. **Nothing is dialled before it is validated.** A candidate address — from a
 *    seed, from Bonjour (a hostname), or from another member's gossip — must
 *    resolve to an address inside the same LAN/Tailscale allowlist the request
 *    fence uses, and must answer naming our group (by digest since AUD-154, by
 *    the key itself from a peer still on the previous build), or it is dropped.
 *    Without that, gossip would let one member make the others knock on arbitrary
 *    doors. One consequence is worth naming: the probe reads `/inventory`, so a
 *    member that the router's Guard 2c will not give content to — a link-local
 *    source while the key is still the shipped default — never becomes a peer. It
 *    is not unreachable, it is unconfigured, and setting a key is the fix.
 * 2. **A peer's list is a hint, never authority.** Gossiped addresses join the
 *    candidate set for the *next* cycle and are validated then like any other;
 *    nothing a peer says bypasses rule 1.
 *
 * @module dsh-lan-manager/peers
 */

import { lookup } from "node:dns/promises";
import { request } from "node:http";

import { DEFAULT_RESOLVE_TTL_MS } from "./config.js";
import { discoverCandidates } from "./discovery.js";
import { checkAddress } from "./net.js";

/** A peer unseen for this long drops out of the table. */
export const DEFAULT_TTL_MS = 5 * 60 * 1000;

/** How many candidates are dialled at once (the subnet source can offer 254). */
export const DEFAULT_CONCURRENCY = 24;

/** Cap on addresses learned by gossip, so a hostile peer cannot grow the table. */
export const MAX_GOSSIP_ENTRIES = 512;

/**
 * Cap on one peer's answer, in bytes.
 *
 * The gossip cap and the 3 s probe timeout bound the *table*, and nothing bounded
 * the *buffer*: `probe` dialled every candidate each cycle and appended whatever
 * came back, so any host the fence accepts -- and the default fence accepts the
 * self-assigned range, so a segment with no DHCP hands it to everyone -- could
 * size this process's memory by streaming at it. `{ok: true}` is enough to be
 * believed a member, because a peer's group field is how it says which fleet it is
 * in, not a credential: an intruder simply leaves it out.
 *
 * Same 4 MiB the `dns-sd` reader passes as `maxBuffer`, and the same shape the
 * loopback server applies to a request body: count while reading and drop the
 * connection past the cap, rather than reading it all and checking after.
 */
export const MAX_PEER_RESPONSE_BYTES = 4 * 1024 * 1024;

/**
 * Key one peer by where it answers.
 * @param address - host or IP.
 * @param port - port.
 * @returns the table key.
 */
export function peerKey(address, port) {
  return `${address}:${port}`;
}

/**
 * One JSON request against a peer, resolving instead of throwing so a dead or
 * hostile host costs a timeout, not a crash.
 * @param options - `{address, port, path, method, token, body, timeoutMs}`.
 * @returns `{status, body}`; `status` is 0 when the connection failed.
 */
export function httpJson({ address, port, path, method = "GET", token, body, timeoutMs = 2000 }) {
  return new Promise((resolve) => {
    const payload = body === undefined ? undefined : Buffer.from(JSON.stringify(body));
    let settled = false;
    const done = (value) => {
      if (!settled) {
        settled = true;
        resolve(value);
      }
    };
    const req = request(
      {
        host: address,
        port,
        path,
        method,
        headers: {
          ...(token ? { "x-dsh-token": token } : {}),
          ...(payload
            ? { "content-type": "application/json", "content-length": payload.length }
            : {}),
        },
        timeout: timeoutMs,
      },
      (res) => {
        const chunks = [];
        let received = 0;
        res.on("data", (chunk) => {
          received += chunk.length;
          if (received > MAX_PEER_RESPONSE_BYTES) {
            // The answer is worthless past the cap, so it is dropped rather than
            // kept: the destroy lands in the `error` handler below, which resolves
            // with the reason instead of buffering what the peer chose to send.
            req.destroy(new Error(`peer response exceeds ${MAX_PEER_RESPONSE_BYTES} bytes`));
            return;
          }
          chunks.push(chunk);
        });
        res.on("end", () => {
          const text = Buffer.concat(chunks).toString("utf8");
          let parsed;
          try {
            parsed = JSON.parse(text);
          } catch {
            parsed = undefined;
          }
          done({ status: Number(res.statusCode ?? 0), body: parsed });
        });
      },
    );
    req.on("timeout", () => {
      req.destroy();
      done({ status: 0, body: undefined, error: "timeout" });
    });
    req.on("error", (error) =>
      done({ status: 0, body: undefined, error: String(error?.message ?? error) }),
    );
    if (payload) req.write(payload);
    req.end();
  });
}

/**
 * Resolve a candidate to an address that passes the fence's allowlist.
 * @param candidate - `{address, port, name, source}`.
 * @param options - `{config, resolve}`.
 * @returns `{address, port, name, source}` with a validated IP, or `undefined`.
 */
export async function validateCandidate(candidate, { config = {}, resolve = lookup } = {}) {
  const raw = String(candidate?.address ?? "").trim();
  if (!raw) return undefined;
  let address = raw;
  if (checkAddress(address).family === "unknown") {
    // A hostname (Bonjour hands us one). Resolve first, then judge the address
    // we would actually connect to — never the name.
    try {
      const resolved = await resolve(raw);
      const first = Array.isArray(resolved) ? resolved[0] : resolved;
      address = String(first?.address ?? "");
    } catch {
      return undefined;
    }
  }
  const verdict = checkAddress(address, {
    ipv4Networks: config.ipv4Networks,
    ipv6Networks: config.ipv6Networks,
    allow: config.allowAddresses,
  });
  if (!verdict.allowed) return undefined;
  const port = Number(candidate.port ?? config.peerPort);
  if (!Number.isInteger(port) || port <= 0 || port > 65535) return undefined;
  return {
    address: verdict.address,
    port,
    name: String(candidate.name ?? raw),
    source: String(candidate.source ?? "unknown"),
  };
}

/**
 * Run `worker` over `items`, at most `limit` at a time.
 * @param items - work items.
 * @param limit - concurrency.
 * @param worker - async function of one item.
 * @returns results in input order, with `undefined` where the worker threw.
 */
export async function mapLimit(items, limit, worker) {
  const results = new Array(items.length);
  let next = 0;
  const runners = Array.from({ length: Math.max(1, Math.min(limit, items.length)) }, async () => {
    for (;;) {
      const index = next;
      next += 1;
      if (index >= items.length) return;
      try {
        results[index] = await worker(items[index], index);
      } catch {
        results[index] = undefined;
      }
    }
  });
  await Promise.all(runners);
  return results;
}

/**
 * The delay before the next discovery cycle: the interval, plus or minus up to
 * 10 %. Pure, so the bounds are testable without waiting a minute.
 * @param seconds - the configured interval.
 * @param random - a value in [0, 1).
 * @returns milliseconds.
 */
export function discoveryDelayMs(seconds, random = Math.random()) {
  const base = Math.max(1, Number(seconds) || 60) * 1000;
  const spread = base * 0.1;
  return Math.round(base + (random * 2 - 1) * spread);
}

/**
 * The group's member list, refreshed on a timer.
 */
export class PeerTable {
  /**
   * @param options - `{config, log, self, discovery, validate, fetch, now, ttlMs}`.
   *   `self` is `{id, name, addresses}`, used to skip this machine.
   */
  constructor({
    config = {},
    log = () => {},
    self = {},
    discovery = discoverCandidates,
    validate = validateCandidate,
    fetch = httpJson,
    resolve = lookup,
    now = () => Date.now(),
    ttlMs = DEFAULT_TTL_MS,
  } = {}) {
    this.config = config;
    this.log = log;
    this.self = self;
    this.discovery = discovery;
    this.validate = validate;
    this.fetch = fetch;
    this.resolve = resolve;
    this.now = now;
    this.ttlMs = ttlMs;
    this.peers = new Map();
    this.gossip = new Map();
    // A Bonjour name resolves to the same address cycle after cycle, and one
    // that has gone away burns the full mDNS timeout on every attempt. The cache
    // holds successes and failures alike: measured, twelve stale names cost 30 s
    // per cycle uncached, and nothing at all on the second cycle when cached.
    this.resolveCache = new Map();
    this.timer = undefined;
    this.refreshing = undefined;
    this.lastRefresh = undefined;
    // The last cycle's failed discovery sources, `[{source, message}]`. Kept on
    // the table rather than only in the log so `/peers` can say *why* it is short,
    // instead of a dead Tailscale probe reading as a small fleet.
    this.lastDiscoveryErrors = [];
  }

  /**
   * Resolve a hostname, reusing an answer until its TTL expires.
   *
   * **Failures are cached too**, and that is the point of the cache. A Bonjour
   * name that has gone away does not fail fast: `getaddrinfo` gives it the full
   * mDNS timeout, measured at ~5 s here. Re-resolving it every cycle means a LAN
   * with one stale advertisement spends five seconds of every cycle on a machine
   * that is not there. Caching the failure costs that once per TTL instead.
   *
   * @param name - the hostname.
   * @returns the resolver's result.
   */
  async resolveHost(name) {
    const ttl = this.config.resolveTtlMs ?? DEFAULT_RESOLVE_TTL_MS;
    const cached = this.resolveCache.get(name);
    if (cached && cached.expiresAt > this.now()) {
      if (cached.error) throw cached.error;
      return cached.value;
    }
    const expiresAt = this.now() + ttl;
    try {
      const value = await this.resolve(name);
      this.resolveCache.set(name, { value, expiresAt });
      return value;
    } catch (error) {
      this.resolveCache.set(name, { error, expiresAt });
      throw error;
    }
  }

  /** Addresses this machine owns, so it never lists itself as a peer. */
  selfAddresses() {
    return new Set(this.self?.addresses ?? []);
  }

  /**
   * Record addresses another member told us about. They are candidates only:
   * they are validated on the next refresh exactly like anything else, so a
   * poisoned entry never becomes a connection by itself.
   * @param entries - `[{address, port}]` from a peer's inventory.
   * @param options - `{from}` for the log line.
   * @returns how many new addresses were kept.
   */
  mergeGossip(entries, { from = "peer" } = {}) {
    if (!Array.isArray(entries)) return 0;
    const mine = this.selfAddresses();
    let kept = 0;
    for (const entry of entries) {
      const address = String(entry?.address ?? "").trim();
      if (!address || mine.has(address)) continue;
      const port = Number(entry?.port ?? this.config.peerPort);
      if (!Number.isInteger(port) || port <= 0) continue;
      const key = peerKey(address, port);
      if (this.peers.has(key) || this.gossip.has(key)) continue;
      if (this.gossip.size >= MAX_GOSSIP_ENTRIES) break;
      this.gossip.set(key, {
        address,
        port,
        name: String(entry?.name ?? address),
        source: `gossip:${from}`,
        seenAt: this.now(),
      });
      kept += 1;
    }
    return kept;
  }

  /**
   * Ask one candidate what it holds.
   * @param candidate - a validated candidate.
   * @returns a peer record, or `undefined` when it is not one of ours.
   */
  async probe(candidate) {
    const started = this.now();
    const path = `${this.config.basePath}/inventory`;
    const { status, body } = await this.fetch({
      address: candidate.address,
      port: candidate.port,
      path,
      token: this.config.token,
      timeoutMs: this.config.probeTimeoutMs ?? 3000,
    });
    if (status !== 200 || !body || body.ok !== true) {
      // A 401 means it answered but with another group key: reachable, not ours.
      if (status === 401)
        this.log(`peer ${candidate.address}:${candidate.port} is not in our group (401)`);
      return undefined;
    }
    // AUD-154: a member names its group by digest. The literal spellings stay
    // accepted, because a peer running the previous build sends the key itself
    // and an update that emptied the peer table would be a worse outage than the
    // plaintext it removed; they go once no member in the wild answers that way.
    const digest = this.config.groupDigest;
    const reported = body.group === undefined ? undefined : String(body.group);
    const literal = this.config.groupKey === undefined ? undefined : String(this.config.groupKey);
    const ours =
      literal === undefined ||
      reported === undefined ||
      (digest !== undefined && reported === String(digest)) ||
      reported === literal ||
      (body.groupLabel !== undefined && String(body.groupLabel) === literal);
    if (!ours) {
      // The value is not interpolated: a foreign peer's group string is that
      // peer's credential, and logging it puts someone else's secret in our logs.
      this.log(`peer ${candidate.address}:${candidate.port} reports a group that is not ours`);
      return undefined;
    }
    const address = candidate.address;
    return {
      id: peerKey(address, candidate.port),
      address,
      port: candidate.port,
      name: String(body.self?.name ?? candidate.name ?? address),
      source: candidate.source,
      version: body.self?.version ?? null,
      addresses: Array.isArray(body.self?.addresses) ? body.self.addresses : [],
      workspaces: Array.isArray(body.workspaces) ? body.workspaces : [],
      sessions: Array.isArray(body.sessions) ? body.sessions : [],
      peerCount: Array.isArray(body.peers) ? body.peers.length : 0,
      gossip: Array.isArray(body.peers) ? body.peers : [],
      rttMs: this.now() - started,
      lastSeen: this.now(),
    };
  }

  /**
   * One discovery cycle: union discovery with gossip, validate, probe, prune.
   * Concurrent calls share the in-flight pass rather than starting a second one.
   * @returns the table as a list.
   */
  refresh() {
    if (this.refreshing) return this.refreshing;
    this.refreshing = this.#refresh().finally(() => {
      this.refreshing = undefined;
    });
    return this.refreshing;
  }

  async #refresh() {
    const candidates = [];
    const sourceErrors = [];
    const noteSourceError = (failure) => {
      sourceErrors.push(failure);
      this.log(`discovery source ${failure.source} failed: ${failure.message}`);
    };
    try {
      candidates.push(
        ...(await this.discovery({ config: this.config, onSourceError: noteSourceError })),
      );
    } catch (error) {
      const message = String(error?.message ?? error);
      this.log(`discovery failed: ${message}`);
      sourceErrors.push({ source: "discovery", message });
    }
    this.lastDiscoveryErrors = sourceErrors;
    candidates.push(...this.gossip.values());

    const mine = this.selfAddresses();
    const seen = new Set();
    // Validation may resolve a hostname (threadpool), so it runs on its own,
    // shallower limit than the socket probes that follow.
    const checked = await mapLimit(candidates, this.config.resolveConcurrency ?? 4, (candidate) =>
      this.validate(candidate, {
        config: this.config,
        resolve: (name) => this.resolveHost(name),
      }),
    );
    const validated = [];
    for (const check of checked) {
      if (!check) continue;
      if (mine.has(check.address)) continue;
      const key = peerKey(check.address, check.port);
      if (seen.has(key)) continue;
      seen.add(key);
      validated.push(check);
    }

    const probed = await mapLimit(
      validated,
      this.config.discoveryConcurrency ?? DEFAULT_CONCURRENCY,
      (candidate) => this.probe(candidate),
    );

    let added = 0;
    for (const record of probed) {
      if (!record) continue;
      if (!this.peers.has(record.id)) added += 1;
      this.peers.set(record.id, record);
      this.gossip.delete(record.id);
      // A member's own view of the group becomes candidate addresses for the
      // next cycle — the mesh shortcut.
      this.mergeGossip(record.gossip, { from: record.id });
    }

    const cutoff = this.now() - this.ttlMs;
    for (const [id, record] of this.peers) {
      if (record.lastSeen < cutoff) this.peers.delete(id);
    }
    this.lastRefresh = this.now();
    if (added > 0) this.log(`discovery: ${added} new peer(s), ${this.peers.size} in the group`);
    return this.list();
  }

  /**
   * The table as plain JSON, newest first.
   * @returns peer records without the raw gossip payload.
   */
  list() {
    return [...this.peers.values()]
      .sort((a, b) => b.lastSeen - a.lastSeen)
      .map(({ gossip, ...rest }) => ({
        ...rest,
        sessionCount: rest.sessions.length,
        workspaceCount: rest.workspaces.length,
      }));
  }

  /**
   * What this instance knows, in the shape the manager consumes: itself plus
   * every member's workspaces and sessions.
   * @returns `{self, peers}`.
   */
  inventory() {
    return { self: this.self, peers: this.list() };
  }

  /**
   * Resolve a peer by id, address, or name.
   * @param selector - any of those.
   * @returns the record, or `undefined`.
   */
  get(selector) {
    const wanted = String(selector ?? "");
    if (!wanted) return undefined;
    return (
      this.peers.get(wanted) ??
      [...this.peers.values()].find((peer) => peer.address === wanted || peer.name === wanted)
    );
  }

  /**
   * Start the discovery timer, jittered.
   *
   * A fleet that all installed at once would otherwise run its cycles in
   * lockstep — every member probing every other member in the same second, for
   * ever. Each cycle is scheduled from the last, with up to ±10 % jitter, which
   * also means a slow cycle can never overlap the next one.
   */
  start() {
    if (this.timer) return;
    const seconds = this.config.discoveryIntervalSeconds ?? 60;
    const schedule = () => {
      const delay = discoveryDelayMs(seconds, Math.random());
      this.timer = setTimeout(() => {
        this.refresh()
          .catch((error) => this.log(`discovery cycle failed: ${error?.message ?? error}`))
          .finally(schedule);
      }, delay);
      this.timer.unref?.();
    };
    schedule();
  }

  /** Stop the timer. */
  stop() {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = undefined;
    }
  }
}
