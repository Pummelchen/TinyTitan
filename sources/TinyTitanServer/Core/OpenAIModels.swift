import Foundation
import TinyTitan

public struct OpenAIChatRequest: Codable, Equatable, Sendable {
    public let model: String
    public let messages: [OpenAIChatMessage]
    public let stream: Bool?
    public let streamOptions: OpenAIStreamOptions?
    public let temperature: Float?
    public let topP: Float?
    public let maxTokens: Int?
    public let maxCompletionTokens: Int?
    public let stop: OpenAIStop?
    public let seed: UInt64?
    public let tools: [OpenAITool]?
    public let toolChoice: JSONValue?
    public let parallelToolCalls: Bool?
    public let topK: Int?
    public let repetitionPenalty: Float?
    public let n: Int?
    public let logprobs: Bool?
    public let presencePenalty: Float?
    public let frequencyPenalty: Float?
    /// Requested reasoning-effort level. Validated against the served model
    /// family's chat template and the server's load-time profile.
    public let reasoningEffort: String?
    /// The same controls in the template-kwargs dialect (see above).
    public let chatTemplateKwargs: OpenAIChatTemplateKwargs?
    /// llama.cpp's hard per-request thinking-token budget.
    ///
    /// Decoded, never refused, and not enforced: this runtime bounds thinking by
    /// the level a template renders rather than by a token count, and a client
    /// that sends this also sends the level it wants. Refusing a field the
    /// runtime does not implement would break that client for the rest of the
    /// session, which is the failure the effort mapping already exists to avoid.
    public let reasoningBudgetTokens: Int?
    /// The requested output format. Decoded so a structured-output request can
    /// be *refused* rather than silently answered as prose: a client that asks
    /// for JSON and gets unconstrained text is worse off than one told no.
    /// `{"type": "text"}`, the API's own default, is accepted.
    public let responseFormat: JSONValue?

    /// Explicit, with the two thinking-control extras defaulted, so the protocol
    /// mappers that build a chat request from their own shapes keep compiling
    /// unchanged. A `let` with an inline default would have been skipped by the
    /// synthesised decoder, which is how the budget silently decoded to nil.
    public init(
        model: String,
        messages: [OpenAIChatMessage],
        stream: Bool? = nil,
        streamOptions: OpenAIStreamOptions? = nil,
        temperature: Float? = nil,
        topP: Float? = nil,
        maxTokens: Int? = nil,
        maxCompletionTokens: Int? = nil,
        stop: OpenAIStop? = nil,
        seed: UInt64? = nil,
        tools: [OpenAITool]? = nil,
        toolChoice: JSONValue? = nil,
        parallelToolCalls: Bool? = nil,
        topK: Int? = nil,
        repetitionPenalty: Float? = nil,
        n: Int? = nil,
        logprobs: Bool? = nil,
        presencePenalty: Float? = nil,
        frequencyPenalty: Float? = nil,
        reasoningEffort: String? = nil,
        chatTemplateKwargs: OpenAIChatTemplateKwargs? = nil,
        reasoningBudgetTokens: Int? = nil,
        responseFormat: JSONValue? = nil
    ) {
        self.model = model
        self.messages = messages
        self.stream = stream
        self.streamOptions = streamOptions
        self.temperature = temperature
        self.topP = topP
        self.maxTokens = maxTokens
        self.maxCompletionTokens = maxCompletionTokens
        self.stop = stop
        self.seed = seed
        self.tools = tools
        self.toolChoice = toolChoice
        self.parallelToolCalls = parallelToolCalls
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
        self.n = n
        self.logprobs = logprobs
        self.presencePenalty = presencePenalty
        self.frequencyPenalty = frequencyPenalty
        self.reasoningEffort = reasoningEffort
        self.chatTemplateKwargs = chatTemplateKwargs
        self.reasoningBudgetTokens = reasoningBudgetTokens
        self.responseFormat = responseFormat
    }

    enum CodingKeys: String, CodingKey {
        case model, messages, stream, temperature, stop, seed, tools, n, logprobs
        case streamOptions = "stream_options"
        case topP = "top_p"
        case maxTokens = "max_tokens"
        case maxCompletionTokens = "max_completion_tokens"
        case toolChoice = "tool_choice"
        case parallelToolCalls = "parallel_tool_calls"
        case topK = "top_k"
        case repetitionPenalty = "repetition_penalty"
        case presencePenalty = "presence_penalty"
        case frequencyPenalty = "frequency_penalty"
        case reasoningEffort = "reasoning_effort"
        case chatTemplateKwargs = "chat_template_kwargs"
        case reasoningBudgetTokens = "reasoning_budget_tokens"
        case responseFormat = "response_format"
    }
}

/// The load-time reasoning configuration the HTTP layer validates requests
/// against: the served family's template capability plus the flags the model
/// was loaded with. Effort is a load-time control because it changes the
/// rendered prompt, so a request may only confirm the active level, never
/// switch it.
public struct ServerReasoningProfile: Sendable, Equatable {
    public let family: ModelFamily
    public let thinkingMode: ModelThinkingMode
    public let reasoningEffort: ModelReasoningEffort?

    public init(
        family: ModelFamily,
        thinkingMode: ModelThinkingMode,
        reasoningEffort: ModelReasoningEffort?
    ) {
        self.family = family
        self.thinkingMode = thinkingMode
        self.reasoningEffort = reasoningEffort
    }

    /// The compatible Qwen3.5-MoE baseline: binary thinking, off.
    public static let `default` = ServerReasoningProfile(
        family: .qwen36, thinkingMode: .off, reasoningEffort: nil)

    /// The effort the template actually applies under this profile; nil for
    /// binary families and while thinking is off.
    public var effectiveEffort: ModelReasoningEffort? {
        family.effectiveReasoningEffort(
            thinkingMode: thinkingMode,
            effort: reasoningEffort)
    }
}

public struct OpenAIUsage: Codable, Equatable, Sendable {
    public struct PromptTokensDetails: Codable, Equatable, Sendable {
        public let cachedTokens: Int

        enum CodingKeys: String, CodingKey {
            case cachedTokens = "cached_tokens"
        }

        public init(cachedTokens: Int) {
            self.cachedTokens = cachedTokens
        }
    }

    /// How many of `completion_tokens` the model spent thinking.
    ///
    /// The clients that route reasoning separately (llama.cpp's patched servers,
    /// the coding harnesses built on them) read this to bill and to budget; the
    /// runtime already knows the split, because the decoder puts every token in
    /// one channel or the other. Kept as its own object rather than folded into
    /// `completion_tokens` so a client can see both.
    public struct CompletionTokensDetails: Codable, Equatable, Sendable {
        public let reasoningTokens: Int

        enum CodingKeys: String, CodingKey {
            case reasoningTokens = "reasoning_tokens"
        }

        public init(reasoningTokens: Int) {
            self.reasoningTokens = reasoningTokens
        }
    }

    public let promptTokens: Int
    public let completionTokens: Int
    public let totalTokens: Int
    public let promptTokensDetails: PromptTokensDetails
    public let completionTokensDetails: CompletionTokensDetails

    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
        case promptTokensDetails = "prompt_tokens_details"
        case completionTokensDetails = "completion_tokens_details"
    }

    public init(
        promptTokens: Int,
        completionTokens: Int,
        totalTokens: Int,
        cachedTokens: Int = 0,
        reasoningTokens: Int = 0
    ) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.totalTokens = totalTokens
        self.promptTokensDetails = PromptTokensDetails(cachedTokens: cachedTokens)
        self.completionTokensDetails = CompletionTokensDetails(
            reasoningTokens: reasoningTokens)
    }
}

public struct OpenAIModelList: Codable, Equatable, Sendable {
    public struct Model: Codable, Equatable, Sendable {
        public let id: String
        public let object: String
        /// Model creation time. Omitted when unknown rather than lying with a
        /// fabricated epoch (S30).
        public let created: Int?
        public let ownedBy: String

        enum CodingKeys: String, CodingKey {
            case id, object, created
            case ownedBy = "owned_by"
        }

        public init(id: String, object: String, created: Int?, ownedBy: String) {
            self.id = id
            self.object = object
            self.created = created
            self.ownedBy = ownedBy
        }
    }

    public let object: String
    public let data: [Model]
}
