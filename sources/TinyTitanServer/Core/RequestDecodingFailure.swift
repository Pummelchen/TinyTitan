import Foundation
import NIOCore
import TinyTitanLib

/// The answer a body that failed to decode deserves, from the error the decoder
/// raised plus the bytes it was raised over.
///
/// Two different client bugs arrive here. A payload that is not JSON at all, and
/// a payload whose syntax is fine but whose shape is not — and the second one
/// comes with the failing parameter already named by Foundation's `codingPath`.
/// Answering both "malformed JSON request" threw that name away, so a client
/// whose `max_tokens` was the string `"512"` was sent to check its JSON
/// serializer instead of the one field it got wrong.
///
/// The syntax question is decided by re-parsing, which is affordable because
/// this only runs on a request that is already being refused; a decoder that
/// reports a syntax failure as a `DecodingError` cannot otherwise tell the two
/// apart.
package enum RequestDecodingFailure {
    package static func serverError(_ error: Error, body: ByteBuffer) -> ServerRequestError {
        let data = Data(body.readableBytesView)
        guard error is DecodingError else {
            return malformedJSON()
        }
        // A JSON document may legitimately be any fragment, so validity here
        // means "an object, array, string, number, true, false or null" and
        // nothing more.
        if (try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])) == nil {
            return malformedJSON()
        }
        let param = parameterPath(of: error)
        return .invalid(
            message: param.map { "invalid value for \($0)" }
                ?? "request body does not match this endpoint",
            param: param,
            code: "invalid_value")
    }

    private static func malformedJSON() -> ServerRequestError {
        .invalid(message: "malformed JSON request", param: nil, code: "invalid_json")
    }

    /// The decoder's own path to the offending value, in the dotted form the
    /// request validators use for their `param` (`messages.0.content`).
    private static func parameterPath(of error: Error) -> String? {
        guard let decoding = error as? DecodingError else { return nil }
        let codingPath: [CodingKey]
        switch decoding {
        case .typeMismatch(_, let context), .valueNotFound(_, let context),
            .keyNotFound(_, let context), .dataCorrupted(let context):
            codingPath = context.codingPath
        @unknown default:
            // A case this toolchain does not know about gets the nameless
            // wording rather than a guess at its path.
            return nil
        }
        guard !codingPath.isEmpty else { return nil }
        return codingPath.map { key in
            key.intValue.map { "\($0)" } ?? key.stringValue
        }.joined(separator: ".")
    }
}
