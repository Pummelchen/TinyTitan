import NIOHTTP1
import Testing

@testable import TinyTitanServerCore

/// The media-type grammar behind the JSON-body admission gate.
///
/// The old gate was `lowercased().hasPrefix("application/json")`, which is a
/// string test, not a media-type test: every case below starting with that
/// prefix was admitted. These pin the boundary the HTTP tests exercise through
/// the server, where a wrong admission is harder to read.
@Suite("HTTP media type")
struct HTTPMediaTypeTests {
    @Test func exactJsonInAnyCaseIsJSON() throws {
        for raw in ["application/json", "Application/JSON", "APPLICATION/JSON"] {
            let type = try #require(HTTPMediaType.parse(raw), "\(raw) did not parse")
            #expect(type.isJSON, "\(raw) is application/json")
        }
    }

    @Test func validParametersAreParsedAndDoNotChangeTheType() throws {
        let type = try #require(
            HTTPMediaType.parse("application/json ; charset=\"UTF-8\"; boundary=\"a;b\""))
        #expect(type.isJSON)
        #expect(type.parameters["charset"] == "UTF-8")
        #expect(type.parameters["boundary"] == "a;b")
    }

    @Test func parameterNamesAreCaseInsensitive() throws {
        let type = try #require(HTTPMediaType.parse("application/json; Charset=utf-8"))
        #expect(type.parameters["charset"] == "utf-8")
    }

    @Test func aQuotedPairIsUnescaped() throws {
        let type = try #require(HTTPMediaType.parse(#"application/json; boundary="a\"b""#))
        #expect(type.parameters["boundary"] == #"a"b"#)
    }

    @Test func leadingAndTrailingWhitespaceIsOptional() throws {
        #expect(HTTPMediaType.parse("  application/json\t")?.isJSON == true)
        #expect(HTTPMediaType.parse("application/json ; charset=utf-8 ")?.isJSON == true)
    }

    @Test func aDifferentSubtypeIsNotJSON() {
        for raw in [
            "application/jsonx", "application/json-foo", "application/vnd.api+json",
            "application/xml", "text/json", "application/*",
        ] {
            #expect(HTTPMediaType.parse(raw)?.isJSON == false, "\(raw) is not application/json")
        }
    }

    @Test func malformedMediaTypesDoNotParse() {
        for raw in [
            "", "application", "application/", "/json", "application /json",
            "application/ json", "application/json, text/plain", "application/json;",
            "application/json; charset", "application/json; charset=",
            "application/json; charset=utf-8; charset=utf-8",
            "application/json; =utf-8", "application/json extra",
            "application/json; charset=\"utf-8", "application/json; charset=a\"b",
        ] {
            #expect(HTTPMediaType.parse(raw) == nil, "\(raw) must not parse")
        }
    }

    @Test func headersClassifyAbsentDuplicateInvalidAndExact() {
        #expect(HTTPHeaders().requestContentType() == .absent)

        var duplicate = HTTPHeaders()
        duplicate.add(name: "content-type", value: "application/json")
        duplicate.add(name: "content-type", value: "text/plain")
        #expect(duplicate.requestContentType() == .duplicate)

        var nonJSON = HTTPHeaders()
        nonJSON.add(name: "content-type", value: "application/jsonx")
        #expect(nonJSON.requestContentType().isJSON == false)
        #expect(nonJSON.requestContentType() != .invalid, "jsonx is a media type, just not ours")

        var invalid = HTTPHeaders()
        invalid.add(name: "content-type", value: "application/json; charset")
        #expect(invalid.requestContentType() == .invalid)

        var exact = HTTPHeaders()
        exact.add(name: "Content-Type", value: "Application/JSON; charset=utf-8")
        #expect(exact.requestContentType().isJSON)
    }
}
