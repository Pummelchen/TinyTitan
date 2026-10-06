/**
 * Source-address fencing for the LAN management API.
 *
 * The API mutates workspaces and sessions and enqueues model prompts, so who may
 * call it is the plugin's primary security property. The fence is a **source-IP
 * allowlist** evaluated on every request, before any handler runs:
 *
 * | Range | Why |
 * |---|---|
 * | `127.0.0.0/8`, `::1` | loopback — the CLI on the same machine |
 * | `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16` | RFC 1918 private LAN |
 * | `169.254.0.0/16` | link-local |
 * | `100.64.0.0/10` | Tailscale / carrier-grade NAT (100.64–100.127) |
 * | `fc00::/7`, `fe80::/10` | IPv6 unique-local and link-local |
 *
 * `100.64.0.0/10` is included deliberately: Tailscale hands peers `100.x.y.z`
 * addresses out of the CGNAT block, and a fleet of Macs reached over Tailscale
 * is the shape this plugin is built for. It is *not* a general "100.*" match —
 * the mask is /10, so only 100.64.0.0–100.127.255.255 pass.
 *
 * Everything else — public addresses, and any address the parser cannot make
 * sense of — is refused. The default is deny.
 *
 * @module dsh-lan-manager/net
 */

/** Minimal IPv4 pattern check; the numeric work is done on the 32-bit value. */
const IPV4 = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/;

/**
 * Networks allowed by default, as `[network, prefixLength]` IPv4 tuples plus
 * IPv6 prefixes handled separately.
 */
export const DEFAULT_IPV4_NETWORKS = Object.freeze([
  ["127.0.0.0", 8], // loopback
  ["10.0.0.0", 8], // RFC 1918
  ["172.16.0.0", 12], // RFC 1918
  ["192.168.0.0", 16], // RFC 1918
  ["169.254.0.0", 16], // link-local
  ["100.64.0.0", 10], // CGNAT / Tailscale 100.x
]);

/** IPv6 prefixes allowed by default, as `[network, prefixLength]`. */
export const DEFAULT_IPV6_NETWORKS = Object.freeze([
  ["::1", 128], // loopback
  ["fc00::", 7], // unique local (fc00::/7)
  ["fe80::", 10], // link-local (fe80::/10)
]);

/**
 * Parse a dotted-quad IPv4 address into an unsigned 32-bit integer.
 * @param value - candidate address.
 * @returns the numeric value, or `undefined` when it is not a valid IPv4 literal.
 */
export function ipv4ToInt(value) {
  const match = IPV4.exec(String(value ?? "").trim());
  if (!match) return undefined;
  const octets = match.slice(1).map((part) => Number(part));
  if (octets.some((n) => !Number.isInteger(n) || n < 0 || n > 255)) return undefined;
  // `<< 24` on the first octet would go negative for values >= 128; multiply.
  return ((octets[0] * 256 + octets[1]) * 256 + octets[2]) * 256 + octets[3];
}

/**
 * Does an IPv4 address fall inside a `network/prefix`?
 * @param address - dotted-quad address.
 * @param network - dotted-quad network base.
 * @param prefix - prefix length in bits (0-32).
 * @returns true when the address is inside the network.
 */
export function ipv4InNetwork(address, network, prefix) {
  const a = ipv4ToInt(address);
  const n = ipv4ToInt(network);
  if (a === undefined || n === undefined) return false;
  const bits = Number(prefix);
  if (!Number.isInteger(bits) || bits < 0 || bits > 32) return false;
  if (bits === 0) return true;
  // >>> keeps the shift unsigned; << 32 is not representable, so guard bits===0.
  const mask = (0xffffffff << (32 - bits)) >>> 0;
  return (a & mask) >>> 0 === (n & mask) >>> 0;
}

/**
 * Normalize an IPv6 literal for prefix comparison: lowercase, strip brackets,
 * drop a zone id (`%en0`), rewrite an IPv4-embedded tail into hex, and shorten
 * the longest run of zero groups to `::`.
 *
 * Malformed input is **rejected** (`undefined`) rather than repaired. The fence's
 * contract is deny-by-default, so a literal the parser cannot make sense of must
 * not be rewritten into some other address — `fc00::1::2` (two `::`) previously
 * normalized to `fc00:0:0:0:0:0:1:2` and was admitted as `fc00::/7`, and a
 * trailing colon produced an empty group that `ipv6ToBytes` read as a zero.
 *
 * @param value - candidate address.
 * @returns the normalized form, or `undefined` when it is not IPv6.
 */
export function normalizeIpv6(value) {
  let text = String(value ?? "")
    .trim()
    .toLowerCase();
  if (!text) return undefined;
  if (text.startsWith("[") && text.endsWith("]")) text = text.slice(1, -1);
  const zone = text.indexOf("%");
  if (zone !== -1) text = text.slice(0, zone);
  if (!text.includes(":")) return undefined;

  // An IPv4-embedded tail (`::ffff:192.168.1.5`) is legal IPv6. Rewrite the
  // dotted quad as the two hex groups it stands for before the group logic, so
  // that form parses instead of being refused as "ten groups".
  const lastColon = text.lastIndexOf(":");
  const dottedTail = text.slice(lastColon + 1);
  if (dottedTail.includes(".")) {
    const octets = dottedTail.split(".");
    if (octets.length !== 4) return undefined;
    const values = octets.map((octet) => (/^\d{1,3}$/.test(octet) ? Number(octet) : NaN));
    if (values.some((n) => !Number.isInteger(n) || n > 255)) return undefined;
    const high = ((values[0] << 8) | values[1]).toString(16);
    const low = ((values[2] << 8) | values[3]).toString(16);
    text = `${text.slice(0, lastColon)}:${high}:${low}`;
  }

  const isGroup = (g) => /^[0-9a-f]{1,4}$/.test(g);

  // Expand `::` in place: head groups, then the zeros the gap stands for, then
  // the tail. Putting the padding at the front would move the network bits and
  // make a correct `fe80::/10` test fail on an expanded address.
  if (text.includes("::")) {
    // At most one `::`, and it must stand for at least one zero group.
    if (text.split("::").length > 2) return undefined;
    const at = text.indexOf("::");
    const head = text.slice(0, at);
    const tail = text.slice(at + 2);
    const headGroups = head === "" ? [] : head.split(":");
    const tailGroups = tail === "" ? [] : tail.split(":");
    const groups = [...headGroups, ...tailGroups];
    if (groups.some((g) => !isGroup(g))) return undefined;
    const missing = 8 - groups.length;
    if (missing < 1) return undefined;
    return [...headGroups, ...Array(missing).fill("0"), ...tailGroups]
      .map((g) => g.replace(/^0+(?=.)/, ""))
      .join(":");
  }
  const groups = text.split(":");
  if (groups.length !== 8 || groups.some((g) => !isGroup(g))) return undefined;
  return groups.map((g) => g.replace(/^0+(?=.)/, "")).join(":");
}

/**
 * Parse an IPv6 literal into its 16 bytes.
 * @param value - candidate address (zone id and brackets tolerated).
 * @returns a 16-byte `Uint8Array`, or `undefined` when it is not IPv6.
 */
export function ipv6ToBytes(value) {
  const normalized = normalizeIpv6(value);
  if (normalized === undefined) return undefined;
  const groups = normalized.split(":");
  if (groups.length !== 8) return undefined;
  const bytes = new Uint8Array(16);
  for (let i = 0; i < 8; i += 1) {
    const group = groups[i] === "" ? "0" : groups[i];
    if (!/^[0-9a-f]{1,4}$/.test(group)) return undefined;
    const value16 = Number.parseInt(group, 16);
    bytes[i * 2] = (value16 >> 8) & 0xff;
    bytes[i * 2 + 1] = value16 & 0xff;
  }
  return bytes;
}

/**
 * Is an IPv6 address inside a `network/prefix`?
 * @param address - candidate address.
 * @param network - network base.
 * @param prefix - prefix length in bits (0-128).
 * @returns true when inside.
 */
export function ipv6InNetwork(address, network, prefix) {
  const a = ipv6ToBytes(address);
  const n = ipv6ToBytes(network);
  if (!a || !n) return false;
  const bits = Number(prefix);
  if (!Number.isInteger(bits) || bits < 0 || bits > 128) return false;
  const wholeBytes = Math.floor(bits / 8);
  for (let i = 0; i < wholeBytes; i += 1) {
    if (a[i] !== n[i]) return false;
  }
  const remainder = bits % 8;
  if (remainder === 0) return true;
  // Compare only the high `remainder` bits of the next byte.
  const mask = (0xff << (8 - remainder)) & 0xff;
  return (a[wholeBytes] & mask) === (n[wholeBytes] & mask);
}

/**
 * Strip the IPv4-mapped IPv6 prefix so `::ffff:192.168.1.5` is judged as IPv4.
 *
 * All three spellings of the mapped prefix name the same IPv4 address — dotted
 * (`::ffff:192.168.1.5`), hex (`::ffff:c0a8:105`) and fully expanded — so they
 * must reach the same verdict; before this, the hex spelling was refused while
 * the dotted one was admitted, which made a caller's standing depend on how its
 * stack happened to print the address.
 *
 * @param address - remote address from the socket.
 * @returns `{family, address}` with the mapped form unwrapped.
 */
export function unwrapAddress(address) {
  const text = String(address ?? "").trim();
  const mapped = /^::ffff:(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})$/i.exec(text);
  if (mapped) return { family: "ipv4", address: mapped[1] };
  if (ipv4ToInt(text) !== undefined) return { family: "ipv4", address: text };
  const v6 = normalizeIpv6(text);
  if (v6 === undefined) return { family: "unknown", address: text };
  const groups = v6.split(":");
  if (
    groups.length === 8 &&
    groups[0] === "0" &&
    groups[1] === "0" &&
    groups[2] === "0" &&
    groups[3] === "0" &&
    groups[4] === "0" &&
    groups[5] === "ffff"
  ) {
    const high = Number.parseInt(groups[6], 16);
    const low = Number.parseInt(groups[7], 16);
    return { family: "ipv4", address: `${high >> 8}.${high & 0xff}.${low >> 8}.${low & 0xff}` };
  }
  return { family: "ipv6", address: v6 };
}

/**
 * Is an address loopback — the machine talking to itself?
 *
 * This is deliberately **not** a narrower `checkAddress`. The fence answers
 * "may this source reach the API at all", and it is widened on purpose to a LAN,
 * a tailnet and link-local so a fleet works with no setup. Loopback answers a
 * different question — "is there anyone else on this network who could have sent
 * it" — and only that one can support a rule that rests on the caller being this
 * machine (AUD-123: the shipped default group key is public, so beyond loopback
 * it proves nothing).
 * @param address - dotted quad, IPv6 literal, or a socket address.
 * @returns true for `127.0.0.0/8` and `::1`.
 */
export function isLoopback(address) {
  const { family, address: normalized } = unwrapAddress(address);
  if (family === "ipv4") return ipv4InNetwork(normalized, "127.0.0.0", 8);
  // `normalizeIpv6` expands, so `::1` arrives as its full group form.
  if (family === "ipv6") return normalized === "0:0:0:0:0:0:0:1";
  return false;
}

/**
 * Is one source address permitted?
 * @param address - the socket's `remoteAddress`.
 * @param options - optional `ipv4Networks` / `ipv6Networks` / `allow` overrides.
 * @returns a decision with the reason, suitable for logging and for the 403 body.
 */
export function checkAddress(address, options = {}) {
  const networks = options.ipv4Networks ?? DEFAULT_IPV4_NETWORKS;
  const prefixes = options.ipv6Networks ?? DEFAULT_IPV6_NETWORKS;
  const extra = options.allow ?? [];
  const { family, address: normalized } = unwrapAddress(address);

  if (family === "unknown") {
    // `family` is carried on every verdict, including this one: callers switch on
    // it (a hostname needs resolving before it can be judged), and an omitted
    // field here reads as "not unknown" to an `=== "unknown"` test.
    return {
      allowed: false,
      reason: "unparseable-source-address",
      address: String(address ?? ""),
      family,
    };
  }

  // Explicit extra allowances run first, so an operator can open one host without
  // widening a whole range.
  for (const entry of extra) {
    const { family: ef, address: ea } = unwrapAddress(String(entry).split("/")[0]);
    if (ef === family && ea === normalized) {
      return { allowed: true, reason: "explicit-allow", address: normalized, family };
    }
    const slash = String(entry).indexOf("/");
    if (ef === "ipv4" && slash !== -1) {
      const [base, bits] = String(entry).split("/");
      if (ipv4InNetwork(normalized, base, bits)) {
        return { allowed: true, reason: "explicit-allow-network", address: normalized, family };
      }
    }
  }

  if (family === "ipv4") {
    for (const [network, prefix] of networks) {
      if (ipv4InNetwork(normalized, network, prefix)) {
        return {
          allowed: true,
          reason: `ipv4 ${network}/${prefix}`,
          address: normalized,
          family,
        };
      }
    }
    return { allowed: false, reason: "source-not-in-allowlist", address: normalized, family };
  }

  for (const [network, prefix] of prefixes) {
    if (ipv6InNetwork(normalized, network, prefix)) {
      return {
        allowed: true,
        reason: `ipv6 ${network}/${prefix}`,
        address: normalized,
        family,
      };
    }
  }
  return { allowed: false, reason: "source-not-in-allowlist", address: normalized, family };
}

/**
 * Resolve the peer address for a request. Uses the socket address, deliberately
 * **not** `X-Forwarded-For`: a forwarded header is attacker-controlled unless a
 * trusted proxy rewrote it, and trusting it here would let any caller claim
 * loopback.
 * @param req - the Node request.
 * @returns the peer address string.
 */
export function peerAddress(req) {
  return req?.socket?.remoteAddress ?? req?.connection?.remoteAddress ?? "";
}
