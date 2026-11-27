import Foundation
import Testing

@testable import TinyTitanCore

/// The parser's incremental decoder box, against the one-shot decode.
///
/// `DetokenizerTextDecoder` is what `StructuredAssistantDecoder` hands the
/// Inkling parser so a segment is decoded once per token instead of once per
/// segment per token (AUD-0100). It must produce exactly the text the
/// whole-answer decode does, including a scalar split across a token boundary.
@Suite struct DetokenizerTextDecoderTests {
    private static func tokenizer() async throws -> GFTokenizer {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        return try await GFTokenizer.load(
            from: root.appendingPathComponent("TinyTitanServer/Fixtures/ChatMLTokenizer"))
    }

    @Test func theIncrementalTextEqualsTheWholeDecode() async throws {
        let tokenizer = try await Self.tokenizer()
        let ids = tokenizer.encode(
            "héllo 世界 🚀 an answer, with multi-byte scalars and 123", addBOS: false
        ).map { Int32($0) }
        let decoder = DetokenizerTextDecoder(tokenizer)
        var streamed = ""
        for id in ids { streamed += decoder.push(Int(id)) }
        streamed += decoder.flush()
        #expect(streamed == tokenizer.decode(ids, skipSpecialTokens: true))
    }
}
