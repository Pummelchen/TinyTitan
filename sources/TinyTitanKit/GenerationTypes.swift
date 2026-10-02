// The generation surface an embedder calls: messages in, events and a summary
// out.
//
// Part of the supported facade (phase A1 of `docs/plan-embedded-library.md`,
// §4). Every type here is kit-native: the shapes are the engine's, but no
// engine type appears in a signature. The wrapper in `Session` maps to and
// from `OpenAIChatRequest`/`ServerInferenceEvent` internally.

/// One turn of a conversation, in the shape the chat template expects.
public struct ChatMessage: Sendable, Equatable {
    public enum Role: Sendable, Equatable {
        case system
        case user
        case assistant

        /// The wire spelling the validator's role table is keyed on.
        var openAIRole: String {
            switch self {
            case .system: "system"
            case .user: "user"
            case .assistant: "assistant"
            }
        }
    }

    public var role: Role
    public var content: String

    public init(role: Role, content: String) {
        self.role = role
        self.content = content
    }
}

/// Sampling and length controls for one generation.
///
/// Every field is sent explicitly, so the served model's profile only fills
/// what this type cannot express (the penalties and the reasoning controls).
public struct GenerationOptions: Sendable {
    public var maxTokens: Int
    public var temperature: Double
    public var topP: Double
    public var topK: Int
    public var repetitionPenalty: Double
    /// `nil` draws from system entropy, as the server does.
    public var seed: UInt64?
    /// Strings that end the answer when they appear in the visible text.
    public var stop: [String]

    public init(
        maxTokens: Int = 512,
        temperature: Double = 0.6,
        topP: Double = 0.95,
        topK: Int = 20,
        repetitionPenalty: Double = 1.0,
        seed: UInt64? = nil,
        stop: [String] = []
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
        self.stop = stop
    }
}

/// Why generation ended, named rather than spelled as the wire's string.
///
/// `toolCalls` is carried because the orchestrator produces it (the model
/// asked for a tool); the §4 sketch listed only the first three cases plus
/// cancellation, and dropping the distinction would have reported a tool call
/// as an ordinary end of turn.
public enum GenerationStopReason: Sendable, Equatable {
    case endOfText
    case length
    case stopString
    case toolCalls
    case cancelled
}

/// What a finished generation produced.
public struct GenerationSummary: Sendable, Equatable {
    /// The visible answer, with any client stop string removed.
    public let text: String
    public let promptTokens: Int
    public let completionTokens: Int
    public let stopReason: GenerationStopReason
}

/// One step of a streaming generation.
///
/// `promptProcessed` is declared for the contract the facade is meant to
/// offer, but the moved orchestrator has no prompt event: its callback carries
/// visible text, reasoning and tool calls only. It is therefore never emitted
/// today -- see the phase A1 report.
public enum GenerationEvent: Sendable {
    case promptProcessed(tokens: Int, cachedTokens: Int)
    case token(String)
    case finished(GenerationSummary)
}

extension GenerationSummary {
    /// The server reports a wire reason string; this is the one place it
    /// becomes a case.
    init(completion: ServerCompletion, cancelled: Bool) {
        self.init(
            text: completion.content,
            promptTokens: completion.usage.promptTokens,
            completionTokens: completion.usage.completionTokens,
            stopReason: cancelled
                ? .cancelled
                : GenerationStopReason(
                    finishReason: completion.finishReason,
                    stopSequence: completion.stopSequence))
    }
}

extension GenerationStopReason {
    init(finishReason: String, stopSequence: String?) {
        switch finishReason {
        case "length": self = .length
        case "tool_calls": self = .toolCalls
        default: self = stopSequence == nil ? .endOfText : .stopString
        }
    }
}
