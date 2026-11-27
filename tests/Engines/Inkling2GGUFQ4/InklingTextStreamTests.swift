import Foundation
import Testing

@testable import Inkling2GGUFQ4
@testable import TinyTitanCore

/// The streaming text of an answer must equal the whole answer's decode.
///
/// `InklingQ4TextStream` used to keep every id and re-decode the answer per
/// token — O(n^2) lookups to emit the last delta (AUD-0100's class). It now
/// decodes each token once through the engine's detokenizer, which holds back a
/// byte sequence a token boundary splits. This pins that the incremental text
/// and the one-shot decode agree, including for multi-byte scalars.
@Suite struct InklingTextStreamTests {
    /// The ChatML fixture tokenizer (no model): the Inkling stream only needs a
    /// `GFTokenizer`, and this one has a byte-level BPE that splits scalars.
    private static func tokenizer() async throws -> GFTokenizer {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        return try await GFTokenizer.load(
            from: root.appendingPathComponent("TinyTitanServer/Fixtures/ChatMLTokenizer"))
    }

    @Test func theStreamedDeltasJoinToTheWholeDecode() async throws {
        let tokenizer = try await Self.tokenizer()
        let text = "héllo 世界 🚀 — an answer with multi-byte scalars, 123"
        let ids = tokenizer.encode(text, addBOS: false).map { Int32($0) }
        #expect(!ids.isEmpty)

        var stream = InklingQ4TextStream(tokenizer: tokenizer)
        var pieces: [String] = []
        for id in ids { pieces.append(stream.push(id)) }
        pieces.append(stream.flush())
        let streamed = pieces.joined()

        let whole = tokenizer.decode(ids, skipSpecialTokens: true)
        #expect(streamed == whole)
    }

    /// Every push after the first must be a suffix of what follows the text
    /// already emitted: no character is emitted twice and none is dropped.
    @Test func everyDeltaExtendsThePreviousText() async throws {
        let tokenizer = try await Self.tokenizer()
        let ids = tokenizer.encode(
            "one two three four five six seven eight nine ten", addBOS: false
        ).map { Int32($0) }
        var stream = InklingQ4TextStream(tokenizer: tokenizer)
        var emitted = ""
        for id in ids {
            let piece = stream.push(id)
            guard !piece.isEmpty else { continue }
            #expect(
                (emitted + piece)
                    == String(
                        tokenizer.decode(ids, skipSpecialTokens: true).prefix(
                            emitted.count + piece.count)),
                "delta \(piece.debugDescription) does not extend \(emitted.debugDescription)")
            emitted += piece
        }
        emitted += stream.flush()
        #expect(emitted == tokenizer.decode(ids, skipSpecialTokens: true))
    }
}
