// One conversation over a loaded model.
//
// Part of the supported facade (phase A1 of `docs/plan-embedded-library.md`,
// §4). It is a thin, honest wrapper: `respond` builds the same validated
// request the chat surface builds, hands it to the moved orchestrator
// (`ServerModelSession.generate`), and maps the events and the completion back
// into kit-native shapes. Nothing about prompt rendering, caching or sampling
// is reimplemented here.
//
// Phase A1 gaps, reported rather than papered over:
//
//   * The orchestrator's callback carries reasoning text and tool calls that
//     this surface's `GenerationEvent`/`GenerationSummary` cannot express yet,
//     so they are dropped. Closing that is additive.
//   * Request validation failures are rethrown as the validator's own error;
//     `TinyTitanError` has no case for a malformed request.
import TinyTitan

public actor Session {
    private let system: String?
    /// Weak on purpose: `Engine` owns the model, so `unload()` really releases
    /// it and this session then reports `engineShutDown`.
    private weak var modelSession: ServerModelSession?
    private let descriptor: ModelDescriptor
    private let reasoningProfile: ServerReasoningProfile
    private var currentTask: Task<GenerationSummary, any Error>?

    init(
        system: String?,
        modelSession: ServerModelSession?,
        descriptor: ModelDescriptor,
        reasoningProfile: ServerReasoningProfile
    ) {
        self.system = system
        self.modelSession = modelSession
        self.descriptor = descriptor
        self.reasoningProfile = reasoningProfile
    }

    /// Generates one answer and streams its visible text through `onEvent`.
    ///
    /// The returned summary is the same one the `finished` event carries. One
    /// generation per session is the contract; a second call waits on the
    /// orchestrator's slot pool.
    public func respond(
        to messages: [ChatMessage],
        options: GenerationOptions = .init(),
        onEvent: @escaping @Sendable (GenerationEvent) -> Void
    ) async throws -> GenerationSummary {
        // The work runs in its own task so `cancel()` has something to cancel:
        // the actor is free while this call awaits the result, which is what
        // lets `cancel()` in. The orchestrator already stops on task
        // cancellation, the same way the server stops a disconnected client.
        let work = Task {
            try await self.generate(messages: messages, options: options, onEvent: onEvent)
        }
        currentTask = work
        defer { currentTask = nil }
        do {
            return try await work.value
        } catch is CancellationError {
            throw TinyTitanError.cancelled
        }
    }

    /// Stops the generation in flight at its next token boundary.
    ///
    /// Cancels the task that is running `generate`, which is how the server
    /// stops a generation whose client has gone; the KV and expert state stay
    /// usable, as they do there. A no-op when nothing is running.
    public func cancel() async {
        currentTask?.cancel()
    }

    private func generate(
        messages: [ChatMessage],
        options: GenerationOptions,
        onEvent: @escaping @Sendable (GenerationEvent) -> Void
    ) async throws -> GenerationSummary {
        guard let modelSession else { throw TinyTitanError.engineShutDown }
        let validated = try validate(
            messages: messages, options: options, modelSession: modelSession)
        let completion: ServerCompletion
        do {
            completion = try await modelSession.generate(validated) { event in
                switch event {
                case .content(let text):
                    onEvent(.token(text))
                case .reasoning, .toolCall:
                    // Not expressible on the A1 surface yet; see the header.
                    break
                }
            }
        } catch let error as GeneratorError {
            switch error {
            case .contextOverflow(let prompt, _, let maxContext):
                throw TinyTitanError.contextWindowExceeded(
                    prompt: prompt, window: maxContext)
            default:
                throw error
            }
        } catch is CancellationError {
            throw TinyTitanError.cancelled
        }
        let summary = GenerationSummary(
            completion: completion, cancelled: Task.isCancelled)
        onEvent(.finished(summary))
        return summary
    }

    private func validate(
        messages: [ChatMessage],
        options: GenerationOptions,
        modelSession: ServerModelSession
    ) throws -> ValidatedChatRequest {
        let request = OpenAIChatRequest(
            model: descriptor.id,
            messages: wireMessages(messages),
            temperature: Float(options.temperature),
            topP: Float(options.topP),
            maxTokens: options.maxTokens,
            stop: options.stop.isEmpty ? nil : .many(options.stop),
            seed: options.seed,
            topK: options.topK,
            repetitionPenalty: Float(options.repetitionPenalty))
        return try OpenAIRequestValidator.validate(
            request,
            modelID: descriptor.id,
            maxContext: modelSession.maximumContext,
            reasoningProfile: reasoningProfile,
            sampling: modelSession.samplingDefaults
        )
        .withModel(descriptor.id)
    }

    private func wireMessages(_ messages: [ChatMessage]) -> [OpenAIChatMessage] {
        var result: [OpenAIChatMessage] = []
        if let system, !system.isEmpty {
            result.append(wireMessage(role: "system", content: system))
        }
        result.append(
            contentsOf: messages.map {
                wireMessage(role: $0.role.openAIRole, content: $0.content)
            })
        return result
    }

    private func wireMessage(role: String, content: String) -> OpenAIChatMessage {
        OpenAIChatMessage(
            role: role, content: .text(content),
            toolCalls: nil, toolCallID: nil, name: nil)
    }
}
