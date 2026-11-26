//
//  HTTPMediaType.swift
//  TinyTitanServer
//
//  Exact `Content-Type` parsing for the JSON-body admission gate.
//

import Foundation
import NIOHTTP1
import TinyTitanCore

/// A media type parsed as RFC 7231 defines it: `type "/" subtype` followed by
/// optional `; name=value` parameters.
///
/// The admission gate used to ask `lowercased().hasPrefix("application/json")`,
/// which is not a media-type test: `application/jsonx`, `application/json-foo`
/// and every vendor `+json` type passed it. This parser accepts exactly one
/// media type, with case-insensitive type, subtype and parameter names, and
/// valid optional parameters (`charset=UTF-8`, `boundary="a;b"`).
///
/// `+json` is deliberately **not** accepted. The supported contract is the
/// literal `application/json` this server's error message and documentation
/// name; a structured suffix is a different media type, and recognising it by
/// suffix would repeat the prefix mistake one level down. A client that sends
/// `application/vnd.api+json` is refused like any other non-JSON type.
struct HTTPMediaType: Equatable, Sendable {
    let type: String
    let subtype: String
    /// Lower-cased parameter names; values unquoted and unescaped. Duplicate
    /// parameter names are invalid, so this cannot silently pick one.
    let parameters: [String: String]

    var isJSON: Bool { type == "application" && subtype == "json" }

    /// Parses `raw`, or returns nil when it is not a valid media type. The
    /// caller decides what an invalid value means; this type never guesses.
    static func parse(_ raw: String) -> HTTPMediaType? {
        var parser = Parser(Array(raw.utf8))
        parser.skipOptionalWhitespace()
        guard let type = parser.token(), !type.isEmpty else { return nil }
        guard parser.consume(UInt8(ascii: "/")) else { return nil }
        guard let subtype = parser.token(), !subtype.isEmpty else { return nil }

        var parameters: [String: String] = [:]
        parser.skipOptionalWhitespace()
        while !parser.atEnd {
            guard parser.consume(UInt8(ascii: ";")) else { return nil }
            parser.skipOptionalWhitespace()
            guard let name = parser.token(), !name.isEmpty else { return nil }
            parser.skipOptionalWhitespace()
            guard parser.consume(UInt8(ascii: "=")) else { return nil }
            parser.skipOptionalWhitespace()
            guard let value = parser.parameterValue() else { return nil }
            parser.skipOptionalWhitespace()

            let key = name.lowercased()
            guard parameters[key] == nil else { return nil }
            parameters[key] = value
        }

        return HTTPMediaType(
            type: type.lowercased(), subtype: subtype.lowercased(), parameters: parameters)
    }

    /// RFC 7230 `tchar`, the characters an unquoted token may contain.
    private static func isTokenByte(_ byte: UInt8) -> Bool {
        switch byte {
        case UInt8(ascii: "a")...UInt8(ascii: "z"),
            UInt8(ascii: "A")...UInt8(ascii: "Z"),
            UInt8(ascii: "0")...UInt8(ascii: "9"):
            return true
        case UInt8(ascii: "!"), UInt8(ascii: "#"), UInt8(ascii: "$"),
            UInt8(ascii: "%"), UInt8(ascii: "&"), UInt8(ascii: "'"),
            UInt8(ascii: "*"), UInt8(ascii: "+"), UInt8(ascii: "-"),
            UInt8(ascii: "."), UInt8(ascii: "^"), UInt8(ascii: "_"),
            UInt8(ascii: "`"), UInt8(ascii: "|"), UInt8(ascii: "~"):
            return true
        default:
            return false
        }
    }

    private struct Parser {
        let bytes: [UInt8]
        var index = 0

        var atEnd: Bool { index >= bytes.count }

        init(_ bytes: [UInt8]) {
            self.bytes = bytes
        }

        mutating func skipOptionalWhitespace() {
            while index < bytes.count, bytes[index] == 0x20 || bytes[index] == 0x09 {
                index += 1
            }
        }

        mutating func consume(_ byte: UInt8) -> Bool {
            guard index < bytes.count, bytes[index] == byte else { return false }
            index += 1
            return true
        }

        /// One run of `tchar`, or nil at a non-token byte.
        mutating func token() -> String? {
            let start = index
            while index < bytes.count, HTTPMediaType.isTokenByte(bytes[index]) {
                index += 1
            }
            guard index > start else { return nil }
            return bytes[start..<index].lossyUTF8String
        }

        /// A token, or an RFC 7230 quoted-string with backslash escapes.
        mutating func parameterValue() -> String? {
            if index < bytes.count, bytes[index] == UInt8(ascii: "\"") {
                index += 1
                var value: [UInt8] = []
                while index < bytes.count {
                    let byte = bytes[index]
                    index += 1
                    switch byte {
                    case UInt8(ascii: "\""):
                        return value.lossyUTF8String
                    case UInt8(ascii: "\\"):
                        guard index < bytes.count else { return nil }
                        let escaped = bytes[index]
                        index += 1
                        // A quoted-pair may quote any visible byte; a bare CR or
                        // LF is never legal in a header value.
                        guard escaped != 0x0D, escaped != 0x0A else { return nil }
                        value.append(escaped)
                    case 0x0D, 0x0A:
                        return nil
                    default:
                        value.append(byte)
                    }
                }
                // Unterminated quoted-string.
                return nil
            }
            return token()
        }
    }
}

/// What a request's `Content-Type` headers amount to.
enum RequestContentType: Equatable, Sendable {
    /// No `Content-Type` at all. Only the documented bodyless routes may treat
    /// this as acceptable; a JSON body route still refuses it.
    case absent
    case mediaType(HTTPMediaType)
    /// Present but not a valid media type.
    case invalid
    /// More than one `Content-Type` header. HTTP does not define how two media
    /// types combine, so this is ambiguous rather than a merge.
    case duplicate

    /// Exactly `application/json`, with or without valid parameters.
    var isJSON: Bool {
        if case .mediaType(let type) = self { return type.isJSON }
        return false
    }

    var isAbsent: Bool { self == .absent }
}

extension HTTPHeaders {
    /// Classifies the request's raw `Content-Type` headers. Uses the
    /// case-insensitive field-name lookup and keeps every value, because
    /// `first(name:)` silently chose one of a duplicate pair.
    func requestContentType() -> RequestContentType {
        let values = self["content-type"]
        guard !values.isEmpty else { return .absent }
        guard values.count == 1 else { return .duplicate }
        guard let parsed = HTTPMediaType.parse(values[0]) else { return .invalid }
        return .mediaType(parsed)
    }
}
