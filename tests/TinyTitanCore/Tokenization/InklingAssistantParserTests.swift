import Foundation
import Testing
import TinyTitanCore

/// Model-free (TT-122): the Inkling dialect's output parser, now in Core where
/// the shared session's `StructuredAssistantDecoder` runs it. The same cases
/// that pinned the Inkling route's parser, against the moved copy.
@Suite struct InklingAssistantParserTests {
    /// A toy vocabulary: control ids 100-106, words 1-9.
    static let vocabulary = InklingAssistantParser.Vocabulary(
        messageModel: 100, contentText: 101, contentThinking: 102, invokeToolJSON: 103,
        invokeToolText: 104, endMessage: 105, foreignTurn: [106],
        control: Set(100...106))
    static let words: [Int: String] = [
        1: "Hello", 2: " world", 3: "let me think", 4: "get_weather",
        5: #"{"name":"get_weather","args":{"city":"Köln"}}"#, 6: "rm_rf",
        7: #"{"name":"rm_rf","args":{}}"#, 8: "\u{FFFD}",
    ]

    func parser(tools: Set<String> = ["get_weather"], startInText: Bool = false)
        -> InklingAssistantParser
    {
        InklingAssistantParser(
            vocabulary: Self.vocabulary,
            decode: { $0.compactMap { Self.words[$0] }.joined() },
            offeredTools: tools, startInText: startInText)
    }

    func run(_ tokens: [Int], _ parser: inout InklingAssistantParser)
        -> [StructuredAssistantEvent]
    {
        tokens.flatMap { parser.consume($0) } + parser.finish()
    }

    @Test func splitsThinkingTextAndToolCalls() throws {
        var parser = parser()
        let events = run([100, 102, 3, 105, 100, 101, 1, 2, 105, 100, 4, 103, 5, 105], &parser)
        #expect(events.count == 4)
        #expect(events[0] == .reasoning("let me think"))
        #expect(events[1] == .content("Hello"))
        #expect(events[2] == .content(" world"))
        guard case .toolCall(let call) = events[3] else {
            Issue.record("expected a tool call, got \(events[3])")
            return
        }
        #expect(call.name == "get_weather")
        #expect(call.arguments == .object(["city": .string("Köln")]))
        #expect(call.argumentsJSON == #"{"city":"Köln"}"#)
        #expect(call.id.hasPrefix("call_"))
    }

    @Test func aCallToAToolNobodyOfferedStaysVisibleAsText() {
        var parser = parser()
        let events = run([100, 6, 103, 7, 105], &parser)
        #expect(events == [.content(#"{"name":"rm_rf","args":{}}"#)])
    }

    @Test func aPromptThatOpenedTextStreamsFromTheFirstToken() {
        var parser = parser(startInText: true)
        #expect(run([1, 2, 105], &parser) == [.content("Hello"), .content(" world")])
    }

    @Test func aForeignTurnEndsTheAnswer() {
        var parser = parser(startInText: true)
        #expect(parser.consume(1) == [.content("Hello")])
        #expect(parser.consume(106).isEmpty)
        #expect(parser.ended)
    }

    @Test func aHalfSpelledCharacterWaitsForTheNextToken() {
        var parser = parser(startInText: true)
        #expect(parser.consume(8).isEmpty)
        #expect(parser.finish() == [.content("\u{FFFD}")])
    }

    // MARK: - The incremental decode (AUD-0100)

    /// A decoder that appends each id's word, standing in for the detokenizer
    /// so the two parser paths can be compared on the same token stream.
    private final class WordsDecoder: IncrementalTextDecoder {
        func push(_ id: Int) -> String { Self.words[id] ?? "" }
        func flush() -> String { "" }
        static let words = InklingAssistantParserTests.words
    }

    /// The live parser decodes each token once instead of re-decoding the
    /// segment per token. Both paths must produce the same events.
    @Test func theIncrementalAndWholeSegmentParsersAgree() throws {
        let streams: [[Int]] = [
            [100, 102, 3, 105, 100, 101, 1, 2, 105],
            [100, 101, 1, 105, 100, 4, 103, 5, 105],
            [1, 2, 3],
            [100, 101, 2, 1, 105],
        ]
        for tokens in streams {
            var whole = parser()
            var incremental = InklingAssistantParser(
                vocabulary: Self.vocabulary,
                decode: { $0.compactMap { Self.words[$0] }.joined() },
                incremental: WordsDecoder(),
                offeredTools: ["get_weather"], startInText: false)
            // A generated call carries a fresh random id per parser, so the
            // comparison names the call by its content.
            func normalized(_ events: [StructuredAssistantEvent]) -> [String] {
                events.map { event in
                    switch event {
                    case .content(let text): return "content:\(text)"
                    case .reasoning(let text): return "reasoning:\(text)"
                    case .toolCall(let call):
                        return "call:\(call.name):\(call.argumentsJSON)"
                    }
                }
            }
            #expect(
                normalized(run(tokens, &whole)) == normalized(run(tokens, &incremental)),
                "stream \(tokens) disagreed")
        }
    }
}
