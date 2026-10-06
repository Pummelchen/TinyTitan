/**
 * Tests for the source-address fence. This is the plugin's security boundary, so
 * the cases are exhaustive rather than illustrative: every allowed range, the
 * boundary just outside each one, and the headers an attacker controls.
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  DEFAULT_IPV4_NETWORKS,
  checkAddress,
  ipv4InNetwork,
  ipv4ToInt,
  ipv6InNetwork,
  isLinkLocal,
  isLoopback,
  normalizeIpv6,
  peerAddress,
  unwrapAddress,
} from "../src/net.js";

// AUD-123: loopback is a different question from "inside the fence", and the
// fence is deliberately wider. These pin the narrow one, because a rule that
// rests on "this is the machine talking to itself" is only as good as this test.
test("isLoopback admits only the machine itself", () => {
  assert.equal(isLoopback("127.0.0.1"), true);
  assert.equal(isLoopback("127.5.6.7"), true);
  assert.equal(isLoopback("::1"), true);
  assert.equal(isLoopback("[::1]"), true);
  assert.equal(isLoopback("::ffff:127.0.0.1"), true);
  assert.equal(isLoopback("10.0.0.1"), false);
  assert.equal(isLoopback("169.254.9.9"), false);
  assert.equal(isLoopback("100.64.0.1"), false);
  assert.equal(isLoopback("fe80::1%en0"), false);
  assert.equal(isLoopback("mordor.local"), false);
  assert.equal(isLoopback(""), false);
  // Every one of the false cases above is *inside* the default fence: the two
  // predicates must disagree, or the rule they both feed is not the one claimed.
  for (const address of ["10.0.0.1", "169.254.9.9", "100.64.0.1", "fe80::1%en0"]) {
    assert.equal(checkAddress(address).allowed, true);
    assert.equal(isLoopback(address), false);
  }
});

test("isLinkLocal names the ranges a host gives itself", () => {
  // Both halves of the boundary: the range a host lands in when nothing assigned
  // it an address, and the addresses an operator did assign.
  for (const address of ["169.254.1.1", "169.254.255.255", "fe80::1", "febf::9", "fe80::1%en0"]) {
    assert.equal(isLinkLocal(address), true, address);
  }
  // The IPv4-mapped spelling of the same address has to reach the same verdict.
  assert.equal(isLinkLocal("::ffff:169.254.5.6"), true);
  for (const address of [
    "10.0.0.1",
    "192.168.1.9",
    "127.0.0.1",
    "100.64.0.1",
    "fec0::1",
    "fd00::1",
  ]) {
    assert.equal(isLinkLocal(address), false, address);
  }
  // `fe80::/10` ends at febf, so the /10 has to be the mask rather than a prefix
  // match on the string; `fec0::/10` was the old site-local range and is refused.
  assert.equal(isLinkLocal("febf::1"), true);
  assert.equal(isLinkLocal("fec0::1"), false);
  assert.equal(isLinkLocal("mordor.local"), false);
  assert.equal(isLinkLocal(""), false);
  // Everything this says yes to is also inside the default fence — the guard that
  // uses it narrows a decision the fence already made, it does not replace one.
  for (const address of ["169.254.9.9", "fe80::1%en0"]) {
    assert.equal(checkAddress(address).allowed, true);
    assert.equal(isLinkLocal(address), true);
    assert.equal(isLoopback(address), false);
  }
});

test("ipv4ToInt parses and rejects", () => {
  assert.equal(ipv4ToInt("0.0.0.0"), 0);
  assert.equal(ipv4ToInt("255.255.255.255"), 4294967295);
  assert.equal(ipv4ToInt("192.168.1.5"), 3232235781);
  assert.equal(ipv4ToInt(" 10.0.0.1 "), ipv4ToInt("10.0.0.1"));
  assert.equal(ipv4ToInt("256.1.1.1"), undefined);
  assert.equal(ipv4ToInt("1.2.3"), undefined);
  assert.equal(ipv4ToInt("1.2.3.4.5"), undefined);
  assert.equal(ipv4ToInt("::1"), undefined);
  assert.equal(ipv4ToInt(""), undefined);
  assert.equal(ipv4ToInt(undefined), undefined);
});

test("high first octets do not go negative (no << 24 sign bug)", () => {
  // 192.168.x and above would be negative under a signed left shift.
  for (const address of ["128.0.0.0", "192.168.1.1", "200.100.50.25", "255.0.0.1"]) {
    assert.ok(ipv4ToInt(address) > 0, `${address} should be positive`);
  }
});

test("ipv4InNetwork masks correctly, including /0 and /32", () => {
  assert.equal(ipv4InNetwork("192.168.1.5", "192.168.0.0", 16), true);
  assert.equal(ipv4InNetwork("192.169.1.5", "192.168.0.0", 16), false);
  assert.equal(ipv4InNetwork("10.1.2.3", "10.0.0.0", 8), true);
  assert.equal(ipv4InNetwork("11.1.2.3", "10.0.0.0", 8), false);
  assert.equal(ipv4InNetwork("1.2.3.4", "0.0.0.0", 0), true, "/0 allows everything");
  assert.equal(ipv4InNetwork("10.0.0.1", "10.0.0.1", 32), true, "exact /32");
  assert.equal(ipv4InNetwork("10.0.0.2", "10.0.0.1", 32), false);
  assert.equal(ipv4InNetwork("bogus", "10.0.0.0", 8), false);
  assert.equal(ipv4InNetwork("10.0.0.1", "10.0.0.0", 99), false, "invalid prefix");
});

test("loopback is allowed", () => {
  for (const address of ["127.0.0.1", "127.1.2.3", "127.255.255.255"]) {
    assert.equal(checkAddress(address).allowed, true, address);
  }
});

test("RFC 1918 private ranges are allowed", () => {
  const allowed = [
    "10.0.0.1",
    "10.255.255.255",
    "172.16.0.1",
    "172.31.255.254",
    "192.168.0.1",
    "192.168.18.27",
    "192.168.255.255",
    "169.254.1.1",
  ];
  for (const address of allowed) {
    assert.equal(checkAddress(address).allowed, true, address);
  }
});

test("Tailscale / CGNAT 100.64.0.0/10 is allowed, and only that window", () => {
  // The whole point: a fleet over Tailscale uses 100.x addresses.
  assert.equal(checkAddress("100.64.0.1").allowed, true, "start of CGNAT block");
  assert.equal(checkAddress("100.100.16.45").allowed, true, "typical Tailscale peer");
  assert.equal(checkAddress("100.127.255.255").allowed, true, "end of CGNAT block");
  // Just outside on both sides must not pass.
  assert.equal(checkAddress("100.63.255.255").allowed, false, "below the /10");
  assert.equal(checkAddress("100.128.0.0").allowed, false, "above the /10");
});

test("public addresses are refused", () => {
  const denied = ["8.8.8.8", "1.1.1.1", "93.184.216.34", "11.0.0.1", "172.32.0.1", "192.169.0.1"];
  for (const address of denied) {
    const verdict = checkAddress(address);
    assert.equal(verdict.allowed, false, address);
    assert.equal(verdict.reason, "source-not-in-allowlist");
  }
});

test("unparseable and empty sources are refused, never defaulted open", () => {
  for (const address of ["", undefined, null, "not-an-ip", "10.0.0.300", "1.2.3.4; rm -rf /"]) {
    const verdict = checkAddress(address);
    assert.equal(verdict.allowed, false, String(address));
  }
});

test("IPv6 loopback and unique-local are allowed", () => {
  for (const address of ["::1", "fd00::1", "fc00::abcd", "fe80::1%en0"]) {
    assert.equal(checkAddress(address).allowed, true, address);
  }
});

test("IPv6 global unicast is refused", () => {
  for (const address of ["2001:4860:4860::8888", "2606:4700::1111"]) {
    assert.equal(checkAddress(address).allowed, false, address);
  }
});

test("IPv4-mapped IPv6 is judged as IPv4", () => {
  assert.equal(checkAddress("::ffff:192.168.1.5").allowed, true, "private mapped");
  assert.equal(checkAddress("::ffff:8.8.8.8").allowed, false, "public mapped");
  assert.deepEqual(unwrapAddress("::ffff:10.0.0.1"), { family: "ipv4", address: "10.0.0.1" });
});

test("an IPv4-mapped address is judged as IPv4 in every spelling", () => {
  // Dotted, hex and fully expanded all name 192.168.1.5. Before the fix the hex
  // spelling was refused while the dotted one was admitted, so a caller's
  // standing depended on how its stack printed the address.
  const spellings = ["::ffff:192.168.1.5", "::ffff:c0a8:105", "0:0:0:0:0:ffff:192.168.1.5"];
  for (const address of spellings) {
    assert.deepEqual(unwrapAddress(address), { family: "ipv4", address: "192.168.1.5" }, address);
    assert.equal(checkAddress(address).allowed, true, address);
  }
  // The public mapping must stay refused whichever way it is spelled.
  for (const address of ["::ffff:8.8.8.8", "::ffff:808:808", "0:0:0:0:0:ffff:8.8.8.8"]) {
    assert.equal(checkAddress(address).allowed, false, address);
  }
});

test("normalizeIpv6 makes equivalent forms compare equal", () => {
  // The normalized form is fully expanded on purpose: matching happens on bytes,
  // so `fe80::1` and `fe80:0:0:0:0:0:0:1` must collapse to the same string.
  assert.equal(normalizeIpv6("::1"), normalizeIpv6("0:0:0:0:0:0:0:1"));
  assert.equal(normalizeIpv6("fe80::1"), normalizeIpv6("fe80:0:0:0:0:0:0:1"));
  assert.equal(normalizeIpv6("fe80::1"), "fe80:0:0:0:0:0:0:1", "gap filled in place, not prefixed");
  assert.equal(normalizeIpv6("[::1]"), normalizeIpv6("::1"), "brackets stripped");
  assert.equal(normalizeIpv6("fe80::1%en0"), normalizeIpv6("fe80::1"), "zone id stripped");
  assert.equal(normalizeIpv6("10.0.0.1"), undefined, "IPv4 is not IPv6");
});

test("malformed IPv6 is refused, never repaired into an allowed address", () => {
  // Each of these was previously rewritten by the parser into a *different*,
  // valid address — `fc00::1::2` (two `::`) landed inside the allowed fc00::/7,
  // and a trailing colon produced an empty group that was read as a zero.
  const malformed = [
    "1::2::3",
    "fc00::1::2",
    "1:2:3:4:5:6:7:",
    "1:2:3:4:5:6:7:8::",
    ":::",
    "fe80:::1",
  ];
  for (const address of malformed) {
    assert.equal(normalizeIpv6(address), undefined, `${address} must not parse`);
    assert.equal(checkAddress(address).allowed, false, `${address} must be refused`);
  }
});

test("ipv6InNetwork matches on bytes, not string prefixes", () => {
  assert.equal(ipv6InNetwork("fe80::1", "fe80::", 10), true);
  assert.equal(ipv6InNetwork("febf::1", "fe80::", 10), true, "end of fe80::/10");
  assert.equal(ipv6InNetwork("fec0::1", "fe80::", 10), false, "outside fe80::/10");
  assert.equal(ipv6InNetwork("fd00::1", "fc00::", 7), true);
  assert.equal(ipv6InNetwork("fe00::1", "fc00::", 7), false);
  assert.equal(ipv6InNetwork("::1", "::1", 128), true);
  assert.equal(ipv6InNetwork("::2", "::1", 128), false);
});

test("explicit allowances open one host without widening a range", () => {
  const options = { allow: ["203.0.113.7"] };
  assert.equal(checkAddress("203.0.113.7", options).allowed, true);
  assert.equal(checkAddress("203.0.113.8", options).allowed, false, "neighbour stays closed");
});

test("explicit allowance accepts a network form", () => {
  const options = { allow: ["203.0.113.0/24"] };
  assert.equal(checkAddress("203.0.113.7", options).allowed, true);
  assert.equal(checkAddress("203.0.114.1", options).allowed, false);
});

test("a narrowed network list is honoured", () => {
  const narrow = { ipv4Networks: [["127.0.0.0", 8]] };
  assert.equal(checkAddress("127.0.0.1", narrow).allowed, true);
  assert.equal(checkAddress("192.168.1.1", narrow).allowed, false, "LAN no longer allowed");
});

test("the default list is exactly loopback, RFC1918, link-local and CGNAT", () => {
  assert.deepEqual(
    DEFAULT_IPV4_NETWORKS.map(([net, bits]) => `${net}/${bits}`),
    [
      "127.0.0.0/8",
      "10.0.0.0/8",
      "172.16.0.0/12",
      "192.168.0.0/16",
      "169.254.0.0/16",
      "100.64.0.0/10",
    ],
  );
});

test("peerAddress reads the socket, never a forwarded header", () => {
  assert.equal(peerAddress({ socket: { remoteAddress: "10.0.0.9" } }), "10.0.0.9");
  assert.equal(peerAddress({ connection: { remoteAddress: "10.0.0.8" } }), "10.0.0.8");
  assert.equal(peerAddress({}), "");
  // A spoofed header must not influence the resolved peer.
  const spoofed = {
    socket: { remoteAddress: "203.0.113.9" },
    headers: { "x-forwarded-for": "127.0.0.1" },
  };
  assert.equal(peerAddress(spoofed), "203.0.113.9");
  assert.equal(
    checkAddress(peerAddress(spoofed)).allowed,
    false,
    "spoofed loopback is still denied",
  );
});
