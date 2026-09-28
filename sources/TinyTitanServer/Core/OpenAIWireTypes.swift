import Foundation
import TinyTitan

/// The OpenAI-compatible wire types: the error envelope, message content and
/// its parts, tool and function definitions, stop sequences, stream options and
/// the template kwargs block.
///
/// Split out of `OpenAIModels.swift` (2026-09-28) under the 500-line-per-file
/// rule (Task 8 of the cleanup runbook) as pure code motion: the declarations
/// are unchanged and remain top-level types of this module.
public struct OpenAIErrorEnvelope: Codable, Equatable, Sendable {
    public struct Detail: Codable, Equatable, Sendable {
        public let message: String
        public let type: String
        public let param: String?
        public let code: String
    }

    public let error: Detail

    public init(
        message: String, param: String? = nil, code: String, type: String = "invalid_request_error"
    ) {
        error = Detail(
            message: message,
            type: type,
            param: param,
            code: code)
    }
}

public struct OpenAITextPart: Codable, Equatable, Sendable {
    public let type: String
    public let text: String?
}

public enum OpenAIMessageContent: Codable, Equatable, Sendable {
    case text(String)
    case parts([OpenAITextPart])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let text = try? container.decode(String.self) {
            self = .text(text)
        } else {
            self = .parts(try container.decode([OpenAITextPart].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .text(let text): try container.encode(text)
        case .parts(let parts): try container.encode(parts)
        }
    }

    func textValue() throws -> String {
        switch self {
        case .text(let text):
            return text
        case .parts(let parts):
            guard parts.allSatisfy({ $0.type == "text" && $0.text != nil }) else {
                throw ServerRequestError.invalid(
                    message: "only text content parts are supported",
                    param: "messages",
                    code: "unsupported_content")
            }
            return parts.compactMap(\.text).joined()
        }
    }
}

public struct OpenAIFunctionCall: Codable, Equatable, Sendable {
    public let name: String
    public let arguments: String
}

public struct OpenAIToolCall: Codable, Equatable, Sendable {
    public let id: String
    public let type: String
    public let function: OpenAIFunctionCall
}

public struct OpenAIChatMessage: Codable, Equatable, Sendable {
    public let role: String
    public let content: OpenAIMessageContent?
    public let toolCalls: [OpenAIToolCall]?
    public let toolCallID: String?
    public let name: String?

    enum CodingKeys: String, CodingKey {
        case role, content, name
        case toolCalls = "tool_calls"
        case toolCallID = "tool_call_id"
    }
}

public struct OpenAIFunctionDefinition: Codable, Equatable, Sendable {
    public let name: String
    public let description: String?
    public let parameters: JSONValue
}

public struct OpenAITool: Codable, Equatable, Sendable {
    public let type: String
    public let function: OpenAIFunctionDefinition
}

public enum OpenAIStop: Codable, Equatable, Sendable {
    case one(String)
    case many([String])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let one = try? container.decode(String.self) {
            self = .one(one)
        } else {
            self = .many(try container.decode([String].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .one(let value): try container.encode(value)
        case .many(let value): try container.encode(value)
        }
    }

    var values: [String] {
        switch self {
        case .one(let value): [value]
        case .many(let value): value
        }
    }
}

public struct OpenAIStreamOptions: Codable, Equatable, Sendable {
    public let includeUsage: Bool?

    enum CodingKeys: String, CodingKey {
        case includeUsage = "include_usage"
    }
}

/// The thinking controls as Qwen's chat templates take them.
///
/// llama.cpp, vLLM and TabbyAPI clients send these inside a
/// `chat_template_kwargs` object rather than as the top-level
/// `reasoning_effort`, and `enable_thinking: false` there is the only way those
/// clients turn thinking off for one request. A server that read only the
/// top-level field honoured the *level* beside this object and dropped the
/// switch, which is the case that matters: a client that forces thinking off
/// for its summarization calls (so the model's own thinking cannot eat the
/// output cap and truncate the summary) had the fix silently lost.
public struct OpenAIChatTemplateKwargs: Codable, Equatable, Sendable {
    public let enableThinking: Bool?
    public let reasoningEffort: String?

    enum CodingKeys: String, CodingKey {
        case enableThinking = "enable_thinking"
        case reasoningEffort = "reasoning_effort"
    }

    public init(enableThinking: Bool? = nil, reasoningEffort: String? = nil) {
        self.enableThinking = enableThinking
        self.reasoningEffort = reasoningEffort
    }
}
