/**
 * Configuration resolution for `dsh-lan-manager`.
 *
 * Every field is optional and resolved once at apply time. Environment variables
 * are read as a fallback so a deployment can inject the token without writing it
 * into a patch file that ends up in a repository.
 *
 * | Key | Env fallback | Default |
 * |---|---|---|
 * | `basePath` | `DSH_LAN_BASE_PATH` | `/dsh-lan` |
 * | `groupKey` / `token` | `DSH_LAN_KEY`, `DSH_LAN_TOKEN` | {@link DEFAULT_GROUP_KEY} |
 * | `peers` | `DSH_LAN_PEERS` (comma separated) | `[]` |
 * | `discoveryIntervalSeconds` | `DSH_LAN_DISCOVERY_SECONDS` | `60` |
 * | `resolveConcurrency` | `DSH_LAN_RESOLVE_CONCURRENCY` | `4` |
 * | `allowAddresses` | `DSH_LAN_ALLOW` (comma separated) | `[]` |
 * | `trustedHosts` | `DSH_LAN_TRUSTED_HOSTS` (comma separated) | `[]` |
 * | `includeEmptyWorkspaces` | — | `false` |
 * | `enforceOrigin` | — | `true` |
 *
 * **One key, two jobs.** The same string is the group tag (instances carrying it
 * are one fleet) and the door key every request presents in `x-dsh-token`. The
 * default is public — it ships in this file — so it *groups* rather than
 * *protects*, which is the intended single-user-LAN trade. Because it protects
 * nothing until it is changed, the router refuses a **mutating** request that
 * arrives from outside loopback while the key is still the shipped default
 * (AUD-123): the trade is only ever meant to be made with yourself, and a peer
 * address the fence also admits — link-local and CGNAT included — is not you.
 *
 * **The key never goes on the wire.** The group-facing bodies name the group by
 * {@link groupDigest}, not by the key (AUD-154): a value that authorises writes
 * has no business appearing in a `/health` response, which is the one thing
 * every operator curls, pastes into an issue, and leaves in a log file. The
 * digest is not a second secret — with a low-entropy key it is offline-testable
 * against one — so it buys the case that actually happens, which is the secret
 * being *copied*, not the secret being *attacked*.
 *
 * @module dsh-lan-manager/config
 */

import { createHash } from "node:crypto";
import { networkInterfaces } from "node:os";

/** The route prefix an unconfigured install mounts. */
export const DEFAULT_BASE_PATH = "/dsh-lan";

/**
 * The group key an unconfigured install shares. Public by design: it keeps a
 * fleet of freshly-installed Macs talking to each other with no setup, and it is
 * the first thing to change on a network you do not solely own.
 *
 * The CLI's `FleetGroupKey.shippedDefault` is the same string, and
 * `tools/docs-facts.py` fails the docs gate if the two drift (AUD-175).
 */
export const DEFAULT_GROUP_KEY = "tinytitan-lan";

/**
 * Name the group without naming the key that opens it.
 *
 * Sixteen hex characters, domain-separated so a digest of this key is not a
 * digest of anything else the operator hashed, and truncated so it fits on a
 * terminal line. Truncation costs nothing here: an attacker who can reach a
 * group route has already presented the key, so the digest is not the thing
 * standing between them and the API — it only has to be stable enough to match
 * a peer's and short enough to read.
 *
 * @param key - the resolved group key.
 * @returns the digest, or `null` when no key is configured.
 */
export function groupDigest(key) {
  if (key === null || key === undefined || key === "") return null;
  return createHash("sha256").update(`dsh-lan-group:${key}`).digest("hex").slice(0, 16);
}

/** How often discovery runs, in seconds, before an operator changes it. */
export const DEFAULT_DISCOVERY_SECONDS = 60;

/** The port a peer's management API answers on when nothing else is known. */
export const DEFAULT_PEER_PORT = 3080;

/**
 * How long a peer probe waits. Three seconds rather than a LAN's one: a member
 * reached across a tailnet may be on another continent, and a relayed path must
 * not read as "not running a harness".
 */
export const DEFAULT_PROBE_TIMEOUT_MS = 3000;

/** How many peers are probed at once during a discovery cycle. */
export const DEFAULT_DISCOVERY_CONCURRENCY = 24;

/**
 * How many candidates are *resolved* at once.
 *
 * A socket connect costs the event loop nothing, so the probes run wide; a
 * `dns.lookup` is different, because a name that has gone away does not fail
 * fast — it takes the full mDNS timeout, measured at ~5 s on this machine. This
 * bounds how many of those are in flight, and therefore how long one cycle can
 * spend resolving.
 *
 * It is **not** a fix for threadpool starvation, tempting as that story is:
 * measured here, unrelated file reads stayed under 2 ms while twelve 5-second
 * lookups ran, so the shared pool is not the constraint it appears to be. What
 * keeps a Bonjour-heavy LAN cheap is caching the answers *and the failures* —
 * see `resolveTtlMs`.
 */
export const DEFAULT_RESOLVE_CONCURRENCY = 4;

/**
 * How long a hostname's answer — including a *failed* answer — is reused.
 *
 * The failures are the important half. A stale Bonjour advertisement costs the
 * full mDNS timeout on every cycle it is retried; measured with twelve stale
 * names, a cycle took 30 s with this at 0 and 0 s on the second cycle with it at
 * five minutes.
 */
export const DEFAULT_RESOLVE_TTL_MS = 5 * 60 * 1000;

/**
 * Describe this host's own candidate addresses, for the `/health` banner.
 * @returns `[{ iface, address, family }]` excluding loopback; the caller drops
 * link-local before printing.
 */
export function localAddresses() {
  const out = [];
  const nets = networkInterfaces();
  for (const [iface, entries] of Object.entries(nets)) {
    for (const entry of entries ?? []) {
      if (!entry?.address) continue;
      const family = entry.family === "IPv4" || entry.family === 4 ? "ipv4" : "ipv6";
      if (family === "ipv4" && entry.address.startsWith("127.")) continue;
      if (family === "ipv6" && entry.address === "::1") continue;
      out.push({ iface, address: entry.address, family });
    }
  }
  return out;
}

/**
 * Split a comma/space separated list, dropping empties.
 * @param value - raw string.
 * @returns trimmed entries.
 */
export function parseList(value) {
  if (Array.isArray(value)) return value.map((v) => String(v).trim()).filter(Boolean);
  if (typeof value !== "string") return [];
  return value
    .split(/[,\s]+/)
    .map((v) => v.trim())
    .filter(Boolean);
}

/**
 * Resolve plugin configuration.
 * @param raw - the row's `config` object.
 * @param env - environment source (injectable for tests).
 * @returns the resolved config consumed by the router and plugin.
 */
export function resolveConfig(raw = {}, env = process.env) {
  const basePathRaw = raw.basePath ?? env.DSH_LAN_BASE_PATH ?? DEFAULT_BASE_PATH;
  let basePath = String(basePathRaw).trim();
  if (!basePath.startsWith("/")) basePath = `/${basePath}`;
  basePath = basePath.replace(/\/+$/, "") || DEFAULT_BASE_PATH;

  // The group key and the door key are the same string; `token` is kept as the
  // name the router reads so existing callers do not have to know both.
  const groupRaw =
    raw.groupKey ?? raw.token ?? env.DSH_LAN_KEY ?? env.DSH_LAN_TOKEN ?? DEFAULT_GROUP_KEY;
  const groupKey = groupRaw === null || groupRaw === undefined ? "" : String(groupRaw);

  const intervalRaw = raw.discoveryIntervalSeconds ?? env.DSH_LAN_DISCOVERY_SECONDS;
  const interval = Number(intervalRaw);
  const discoveryIntervalSeconds =
    Number.isFinite(interval) && interval >= 5 ? Math.floor(interval) : DEFAULT_DISCOVERY_SECONDS;

  const allowAddresses = [...parseList(raw.allowAddresses ?? env.DSH_LAN_ALLOW ?? "")];

  return {
    version: raw.version ?? null,
    basePath,
    groupKey,
    token: groupKey,
    // What the API says out loud about the group (AUD-154). The literal label
    // survives only for the shipped default, which is public anyway and which an
    // unconfigured fleet reads on its screen to recognise itself; anything an
    // operator chose is named only by its digest.
    groupDigest: groupDigest(groupKey),
    groupLabel: groupKey === DEFAULT_GROUP_KEY ? groupKey : null,
    allowAddresses,
    ipv4Networks: Array.isArray(raw.ipv4Networks) ? raw.ipv4Networks : undefined,
    ipv6Networks: Array.isArray(raw.ipv6Networks) ? raw.ipv6Networks : undefined,
    originNetworks: Array.isArray(raw.originNetworks) ? raw.originNetworks : undefined,
    trustedOrigins: parseList(raw.trustedOrigins ?? ""),
    // Hostnames the operator declares this server is reached by. A same-origin
    // claim in an Origin header is only equal to the request's own Host, and
    // equality the attacker's DNS chose is the rebinding shape (AUD-148), so the
    // router grants same-origin only for a host named here or already inside the
    // address fence.
    trustedHosts: parseList(raw.trustedHosts ?? env.DSH_LAN_TRUSTED_HOSTS ?? ""),
    allowPrivateOrigins: raw.allowPrivateOrigins !== false,
    enforceOrigin: raw.enforceOrigin !== false,
    includeEmptyWorkspaces: raw.includeEmptyWorkspaces === true,
    maxBodyBytes:
      Number.isInteger(raw.maxBodyBytes) && raw.maxBodyBytes > 0 ? raw.maxBodyBytes : undefined,
    logToHost: raw.logToHost !== false,
    // --- fleet discovery -----------------------------------------------------
    // Seeded peers always count; the sources below add to them. A subnet sweep
    // is off by default because it is the only one that touches hosts that never
    // opted in — one config flag away when a LAN has no Bonjour or Tailscale.
    peers: parseList(raw.peers ?? env.DSH_LAN_PEERS ?? ""),
    discoveryIntervalSeconds,
    discoverTailscale: raw.discoverTailscale !== false,
    discoverBonjour: raw.discoverBonjour !== false,
    discoverSubnet: raw.discoverSubnet === true,
    peerPort: Number.isInteger(raw.peerPort) && raw.peerPort > 0 ? raw.peerPort : DEFAULT_PEER_PORT,
    // A tailnet peer can be on another continent. On a direct WireGuard path it
    // answers in tens of milliseconds, but a relayed (DERP) or busy one is
    // slower than the 2 s that would do on a LAN, and a timed-out probe looks
    // exactly like a machine that is not running a harness.
    probeTimeoutMs: probeTimeout(env, raw),
    discoveryConcurrency:
      Number.isInteger(raw.discoveryConcurrency) && raw.discoveryConcurrency > 0
        ? raw.discoveryConcurrency
        : DEFAULT_DISCOVERY_CONCURRENCY,
    resolveConcurrency:
      positive(env.DSH_LAN_RESOLVE_CONCURRENCY, raw.resolveConcurrency) ??
      DEFAULT_RESOLVE_CONCURRENCY,
    resolveTtlMs:
      Number.isInteger(raw.resolveTtlMs) && raw.resolveTtlMs >= 0
        ? raw.resolveTtlMs
        : DEFAULT_RESOLVE_TTL_MS,
  };
}

/**
 * An explicit positive integer from the row, else from the environment.
 * @param fromEnv - the environment value.
 * @param fromRaw - the row value.
 * @returns the number, or `undefined`.
 */
function positive(fromEnv, fromRaw) {
  if (Number.isInteger(fromRaw) && fromRaw > 0) return fromRaw;
  const parsed = Number(fromEnv);
  return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed) : undefined;
}

/**
 * Resolve the peer-probe timeout: an explicit option, then the environment, then
 * the default chosen for a fleet that is not all on one desk.
 * @param env - environment source.
 * @param raw - the row config.
 * @returns milliseconds.
 */
function probeTimeout(env, raw) {
  if (Number.isInteger(raw.probeTimeoutMs) && raw.probeTimeoutMs > 0) return raw.probeTimeoutMs;
  const fromEnv = Number(env.DSH_LAN_PROBE_TIMEOUT);
  if (Number.isFinite(fromEnv) && fromEnv > 0) return Math.floor(fromEnv);
  return DEFAULT_PROBE_TIMEOUT_MS;
}
