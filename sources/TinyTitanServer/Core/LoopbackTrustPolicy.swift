//
//  LoopbackTrustPolicy.swift
//  TinyTitanServer
//
//  The Host/Origin/Referer half of the loopback trust boundary.
//

import Foundation
import NIOHTTP1

/// A request the loopback trust boundary refuses.
struct LoopbackTrustRefusal: Equatable, Sendable {
    let status: HTTPResponseStatus
    let code: String
    let message: String
}

/// Who this server is willing to answer.
///
/// The listener binds `127.0.0.1`, there is no authentication, and the machine
/// is the documented trust boundary (`docs/server-api.md`, `SECURITY.md`). That
/// makes two attackers relevant and one not:
///
/// - Other processes on this machine can reach the port and are **accepted**:
///   they already have the operator's privileges, so a token in this server
///   would not keep them out. That is the residual, documented risk.
/// - A web page the operator visits is **not** trusted. DNS rebinding points a
///   page's own hostname at `127.0.0.1`; the browser then treats this server as
///   same-origin, so CORS never runs and a `fetch` with `Content-Type:
///   application/json` needs no preflight. The `Host` header still carries the
///   attacker's name, which is what this policy reads.
///
/// The policy is deliberately narrow and does not add authentication:
///
/// - `Host` must be IPv4 loopback (`127.0.0.1`), `localhost`, or IPv6 loopback
///   (`::1`), with an optional valid port. HTTP/1.1 still requires the header;
///   HTTP/1.0 may omit it (the pre-1.1 contract), but a present one is checked.
/// - Duplicate `Host`, `Origin` or `Referer` fields are ambiguous and refused.
/// - An `Origin` or `Referer`, when present, must name a loopback host. Ports
///   are not compared: a local page on another loopback port is still inside
///   the machine boundary. CLI clients send neither header and are unaffected.
///
/// CORS is not enabled anywhere: no preflight is answered and no
/// `Access-Control-Allow-Origin` is written, so a cross-origin browser fetch
/// that needs JSON cannot even send its request. This policy covers the
/// requests that do not need a preflight (simple form/text posts, GETs and the
/// rebinding case).
enum LoopbackTrustPolicy {
    /// Hosts that name the loopback interface. `::1` and its long form are both
    /// accepted because a client may send either; the server itself listens on
    /// IPv4 only.
    private static let loopbackHosts: Set<String> = [
        "127.0.0.1", "localhost", "::1", "0:0:0:0:0:0:0:1",
    ]

    static func refusal(for head: HTTPRequestHead) -> LoopbackTrustRefusal? {
        if let refusal = hostRefusal(head) { return refusal }
        if let refusal = headerRefusal(
            name: "origin", values: head.headers["origin"],
            untrustedCode: "untrusted_origin", untrustedMessage: "Origin is not a loopback origin")
        {
            return refusal
        }
        return headerRefusal(
            name: "referer", values: head.headers["referer"],
            untrustedCode: "untrusted_referer",
            untrustedMessage: "Referer is not a loopback URL")
    }

    private static func hostRefusal(_ head: HTTPRequestHead) -> LoopbackTrustRefusal? {
        let hosts = head.headers["host"]
        if hosts.isEmpty {
            guard head.version == .http1_1 else { return nil }
            return LoopbackTrustRefusal(
                status: .badRequest, code: "missing_host", message: "missing Host header")
        }
        guard hosts.count == 1 else {
            return LoopbackTrustRefusal(
                status: .badRequest, code: "duplicate_host",
                message: "more than one Host header is ambiguous")
        }
        guard let host = hostName(fromAuthority: hosts[0]) else {
            return LoopbackTrustRefusal(
                status: .badRequest, code: "invalid_host", message: "invalid Host header")
        }
        guard loopbackHosts.contains(host) else {
            return LoopbackTrustRefusal(
                status: .forbidden, code: "untrusted_host",
                message: "Host must name the loopback interface")
        }
        return nil
    }

    private static func headerRefusal(
        name: String, values: [String],
        untrustedCode: String, untrustedMessage: String
    ) -> LoopbackTrustRefusal? {
        guard !values.isEmpty else { return nil }
        guard values.count == 1 else {
            return LoopbackTrustRefusal(
                status: .badRequest, code: "duplicate_\(name)",
                message: "more than one \(name) header is ambiguous")
        }
        guard let url = URLComponents(string: values[0].trimmingCharacters(in: .whitespaces)),
            let scheme = url.scheme?.lowercased(), scheme == "http" || scheme == "https",
            let rawHost = url.host?.lowercased(), loopbackHosts.contains(unbracketed(rawHost))
        else {
            return LoopbackTrustRefusal(
                status: .forbidden, code: untrustedCode, message: untrustedMessage)
        }
        return nil
    }

    /// The host name from an RFC 7230 `Host` authority:
    /// `host [":" port]` or `"[" IPv6 "]" [":" port]`. Returns nil for a
    /// malformed authority, an unbracketed IPv6 address, or a bad port.
    private static func hostName(fromAuthority raw: String) -> String? {
        let authority = raw.trimmingCharacters(in: .whitespaces)
        guard !authority.isEmpty else { return nil }

        if authority.hasPrefix("[") {
            guard let close = authority.firstIndex(of: "]") else { return nil }
            let host = String(authority[authority.index(after: authority.startIndex)..<close])
            let rest = authority[authority.index(after: close)...]
            if !rest.isEmpty {
                guard rest.hasPrefix(":"), validPort(rest.dropFirst()) else { return nil }
            }
            return host.lowercased()
        }

        guard let colon = authority.lastIndex(of: ":") else {
            return authority.lowercased()
        }
        // A colon before the port separator means an unbracketed IPv6 literal,
        // which is not a valid Host authority.
        guard !authority[..<colon].contains(":") else { return nil }
        guard validPort(authority[authority.index(after: colon)...]) else { return nil }
        return String(authority[..<colon]).lowercased()
    }

    private static func validPort(_ digits: Substring) -> Bool {
        guard !digits.isEmpty, digits.utf8.allSatisfy({ $0 >= 48 && $0 <= 57 }),
            let port = Int(digits)
        else { return false }
        return (1...65_535).contains(port)
    }

    /// `URLComponents.host` keeps the brackets around an IPv6 literal on this
    /// platform, while `Host` parsing above strips them. Compare the same form.
    private static func unbracketed(_ host: String) -> String {
        guard host.hasPrefix("["), host.hasSuffix("]") else { return host }
        return String(host.dropFirst().dropLast())
    }
}
