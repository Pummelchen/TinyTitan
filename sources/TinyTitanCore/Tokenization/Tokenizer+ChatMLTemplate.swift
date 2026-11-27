import Foundation
import Tokenizers

// The prompt renderer: the installed tokenizer's template path and the manual
// ChatML renderer the Qwen families use.
//
// Split out of `Tokenizer.swift` (2026-10-25) as pure code motion under the
// 500-line-per-file rule. `generationSuffix` and `effortSystemInstruction`
// widened from private to internal so the extension can read them.
extension GFTokenizer {
    public func applyChatTemplate(_ messages: [Message]) throws -> String {
        if dialect == .inkling {
            return inklingPromptText(messages: messages, tools: [], structuredAnswer: false)
        }
        return try chatMLChatTemplate(messages)
    }

    private func chatMLChatTemplate(_ messages: [Message]) throws -> String {
        var s = ""
        // Effort-aware templates open the conversation with the derived
        // instruction: inside the leading system block when the chat has one,
        // otherwise as a synthetic system block of its own. A leading
        // `developer` message *is* that system block (`Role.templateRole`), so
        // it must not also get a synthetic one in front of it.
        let opensWithGuidance =
            messages.first.map {
                $0.role == .system || $0.role == .developer
            } ?? false
        if let instruction = effortSystemInstruction, !opensWithGuidance {
            s += Self.imStartMark + "system\n" + instruction + Self.imEndMark + "\n"
        }
        // The bundled template's `ns.last_query_index`: the last user turn that
        // is not a `<tool_response>` echo. Everything after it is the turn being
        // *continued* rather than history, which is the distinction the assistant
        // branch below turns on. Default is the final index, so a chat ending on a
        // user turn has nothing after it.
        var lastQueryIndex = messages.count - 1
        for index in stride(from: messages.count - 1, through: 0, by: -1)
        where messages[index].role == .user {
            let trimmed = (messages[index].content ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.hasPrefix("<tool_response>"), trimmed.hasSuffix("</tool_response>") {
                continue
            }
            lastQueryIndex = index
            break
        }
        for (index, message) in messages.enumerated() {
            guard let rawContent = message.content else {
                throw GFTokenizerError.invalidChatTemplate("text-only messages require content")
            }
            // The bundled Jinja template trims every message's rendered
            // content (`render_content(...)|trim`); the manual renderer
            // mirrors that exactly so both paths agree byte-for-byte.
            var content = rawContent.trimmingCharacters(in: .whitespacesAndNewlines)
            // `developer` is `system` in everything but name here
            // (`Role.templateRole`), so it takes the same place rule and the
            // same effort-instruction fold.
            let isGuidance = message.role == .system || message.role == .developer
            if isGuidance && index != 0 {
                throw GFTokenizerError.invalidChatTemplate("system message must be first")
            }
            if index == 0, isGuidance,
                let instruction = effortSystemInstruction
            {
                content = content.isEmpty ? instruction : instruction + "\n\n" + content
            }
            if message.role == .assistant {
                // Historical assistant turns carry their reasoning *stripped*:
                // the template takes everything after the last `</think>` as the
                // answer and wraps it back in `<think>…</think>` only when the
                // turn comes after the last real user query (the turn being
                // continued). This renderer emitted the raw content, so a
                // previous turn's whole chain of thought stayed in every prompt
                // — context the template exists to remove, and a prompt shape
                // the model was not trained on. `preserve_thinking`, which the
                // template also honours, has no equivalent here.
                let (reasoning, answer) = Self.splitThinking(content)
                content =
                    index > lastQueryIndex
                    ? "<think>\n" + reasoning + "\n</think>\n\n" + answer
                    : answer
            }
            s +=
                Self.imStartMark + message.role.templateRole + "\n" + content + Self.imEndMark
                + "\n"
        }
        s += generationSuffix
        return s
    }
}
