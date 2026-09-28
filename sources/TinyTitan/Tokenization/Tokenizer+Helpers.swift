import Foundation
import Tokenizers

// Tokenizer introspection helpers: the Jinja context, the resolved-special-token
// type, the streaming-decoder check and the special-token resolution.
//
// Split out of `Tokenizer.swift` (2026-09-28) under the 500-line-per-file rule
// (Task 8 of the cleanup runbook) as pure code motion. Four members widened from
// `private` to internal because the initializer that calls them stays in
// Tokenizer.swift: `templateContext`, `ResolvedSpecialTokens`,
// `validateStreamingDecoder` and `resolveChatMLTokens`.
extension GFTokenizer {

    /// The Jinja context shared by every bundled-template render: the binary
    /// switch always, plus the effort override only when one is active (an
    /// absent key selects the template's own default level).
    static func templateContext(
        thinkingEnabled: Bool,
        reasoningEffort: ModelReasoningEffort?
    ) -> [String: any Sendable] {
        var context: [String: any Sendable] = ["enable_thinking": thinkingEnabled]
        if let reasoningEffort {
            context["reasoning_effort"] = reasoningEffort.rawValue
        }
        return context
    }

    struct ResolvedSpecialTokens {
        let bosID: Int32
        let eosID: Int32
        let padID: Int32
        let endOfTurnID: Int32
        let toolCallStartID: Int32
        let toolCallEndID: Int32
        let toolResponseID: Int32
        let toolResponseEndID: Int32
        let channelStartID: Int32
        let channelEndID: Int32
        let thinkStartID: Int32?
        let thinkEndID: Int32?
        let stopTokenIDs: Set<Int32>
        let vocabSize: Int
    }

    static func validateStreamingDecoder(
        _ decoder: GFByteLevelDecoderConfiguration,
        tokenizer: any Tokenizer,
        resolved: ResolvedSpecialTokens
    ) throws {
        let literalMarkers: [(Int32, String)] = [
            (resolved.toolCallStartID, "<tool_call>"),
            (resolved.toolCallEndID, "</tool_call>"),
            (resolved.toolResponseID, "<tool_response>"),
            (resolved.toolResponseEndID, "</tool_response>"),
            (resolved.channelStartID, "<think>"),
            (resolved.channelEndID, "</think>"),
        ]
        for (id, content) in literalMarkers {
            guard let added = decoder.addedTokens[id],
                added.content == content, !added.special
            else {
                throw GFTokenizerError.unsupportedForDialect(
                    "ChatML control token \(content) must be a literal ByteLevel barrier")
            }
        }

        let filteredMarkers = [resolved.eosID, resolved.endOfTurnID]
        for id in filteredMarkers {
            guard decoder.addedTokens[id]?.special == true else {
                let token = tokenizer.convertIdToToken(Int(id)) ?? "id \(id)"
                throw GFTokenizerError.unsupportedForDialect(
                    "ChatML stop token \(token) must be marked special")
            }
        }
    }

    /// Resolves a token string to its ID, rejecting the unk-token fallback
    /// some tokenizers substitute for out-of-vocabulary strings.
    private static func specialTokenID(_ tokenizer: any Tokenizer, _ token: String) -> Int? {
        guard let id = tokenizer.convertTokenToId(token),
            tokenizer.convertIdToToken(id) == token
        else { return nil }
        return id
    }

    /// The model's padded embedding/lm_head row count. The tokenizer's own
    /// vocab (248 077 for Qwen) is smaller; logits buffers and the
    /// embedding/lm_head are sized to the padded rows, and `vocabSize`
    /// reports at least this many.
    private static let paddedLogitsVocabSize = 248_320

    /// Derive the tokenizer's actual vocab by probing `convertIdToToken` for
    /// the first invalid id. Standard vocab files keep ids dense from 0, so
    /// the first nil is the vocab count. Bounded so a pathological tokenizer
    /// cannot make init scan forever; nil means "no reliable derivation".
    private static func derivedVocabSize(_ tokenizer: any Tokenizer) -> Int? {
        // 2,097,152 — far above any shipping vocab.
        let upperBound = 1 << 21
        for id in 0..<upperBound where tokenizer.convertIdToToken(id) == nil {
            return id
        }
        return nil
    }

    static func resolveChatMLTokens(
        _ tokenizer: any Tokenizer
    ) throws -> ResolvedSpecialTokens {
        func id(_ token: String) throws -> Int32 {
            guard let value = specialTokenID(tokenizer, token) else {
                throw GFTokenizerError.missingSpecialToken(token)
            }
            return Int32(value)
        }
        // `<|im_start|>` is required even though no stored property holds it;
        // template rendering relies on the tokenizer recognizing its text.
        _ = try id(Self.imStartMark)
        let imEnd = try id(Self.imEndMark)
        let endOfText = try id("<|endoftext|>")
        let toolCallStart = try id("<tool_call>")
        let toolCallEnd = try id("</tool_call>")
        let toolResponse = try id("<tool_response>")
        let toolResponseEnd = try id("</tool_response>")
        let thinkStart = try id("<think>")
        let thinkEnd = try id("</think>")
        return ResolvedSpecialTokens(
            bosID: endOfText,
            eosID: endOfText,
            padID: endOfText,
            endOfTurnID: imEnd,
            toolCallStartID: toolCallStart,
            toolCallEndID: toolCallEnd,
            toolResponseID: toolResponse,
            toolResponseEndID: toolResponseEnd,
            channelStartID: thinkStart,
            channelEndID: thinkEnd,
            thinkStartID: thinkStart,
            thinkEndID: thinkEnd,
            stopTokenIDs: [imEnd, endOfText],
            // At least the model's padded embedding/lm_head rows; larger when
            // the tokenizer's own vocab (derived from `convertIdToToken`)
            // exceeds them.
            vocabSize: max(
                Self.derivedVocabSize(tokenizer) ?? 0,
                Self.paddedLogitsVocabSize))
    }
}
