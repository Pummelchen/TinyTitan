import Foundation
import Metal
import TinyTitan
import TinyTitanLib

/// Serving K2-Horizon from the GPU probe, through the same HTTP surface as the
/// other families.
///
/// The probe (`K2GraphProbe`) is the only graph that runs these weights today:
/// `RealForwardRunner` sizes a Qwen-shaped pipeline set and refuses the family.
/// Rather than pretend the runner serves it, this backend is the probe behind
/// the `ServerInferenceBackend` protocol, the way `CPUModelBackend` is the CPU
/// engine. Everything above it — the OpenAI and Responses surfaces, the
/// memory subsystem, the watchdogs — is unchanged.
///
/// **What it does not do, deliberately.** No prompt cache: each request builds
/// a fresh graph, because the probe's KV caches are sized per sequence and its
/// layer instances are not built for reuse across requests. No batching, no
/// ANE prefill, no runner statistics. Those belong to the runner graph, not to
/// this adapter.
///
/// **Memory.** The KV caches are plain FP16 and per layer, about 0.2 MB per
/// position across the 48 layers; 8,192 positions is ~1.6 GB on top of the
/// expert cache the model already holds, which is what this 24 GB host can
/// carry. The checkpoint's 524,288-token context is a property of the weights,
/// not something this path can afford to allocate.
public actor K2ProbeBackend: ServerInferenceBackend, PromptCacheDescribing {
    /// The context this path will serve, however long the checkpoint says it
    /// can. See the memory note above.
    public static let contextCeiling = 8_192
    /// The graph's initial sequence capacity. K/V costs ~0.2 MB per position
    /// across the 48 layers, so this reserves ~0.4 GB and covers a normal
    /// multi-turn conversation without a rebuild.
    static let minimumGraphCapacity = 2_048

    private let model: Model
    private let context: MetalContext
    private let tokenizer: GFTokenizer
    private let tokenizerFolder: URL
    private let loadedReasoning: RequestReasoning
    private let contextLimit: Int
    private let defaults: GenerationDefaults.Sampling
    private let slotCount: Int
    private var jsonTokenTable: JSONTokenTable?
    /// The graph and the tokens whose K/V it currently holds. A request whose
    /// prompt shares a prefix with the last one continues from that prefix
    /// instead of re-forwarding it: the layers read the cache by position, so
    /// identical tokens at identical positions have identical K/V.
    ///
    /// `nonisolated(unsafe)` because the graph is a non-Sendable class the
    /// generation loop awaits; every access is inside this actor's `generate`,
    /// which the actor serializes, so there is no second task to race.
    nonisolated(unsafe) private var graph: K2GraphProbe?
    private var graphCapacity = 0
    private var cachedTokens: [Int32] = []

    public nonisolated var maximumContext: Int { contextLimit }
    public nonisolated var samplingDefaults: GenerationDefaults.Sampling { defaults }
    /// The probe keeps no prompt cache and says so rather than borrowing the
    /// mode the server was asked for.
    package nonisolated var promptCacheMode: ServerPromptCacheMode { .off }
    /// The routed-expert slots per layer in force, for the launch banner.
    package nonisolated var expertCacheSlots: Int { slotCount }

    public enum K2BackendError: Error, CustomStringConvertible {
        case unsupported(String)
        case promptTooLong(Int, Int)
        case emptyPrompt

        public var description: String {
            switch self {
            case .unsupported(let detail): detail
            case .promptTooLong(let count, let limit):
                "prompt is \(count) tokens and the context is \(limit)"
            case .emptyPrompt: "the prompt tokenized to nothing"
            }
        }
    }

    /// Loads the install and the tokenizer. The receipt path matches the
    /// bench: a valid receipt avoids re-hashing 15 GB of expert layers on
    /// every start, and an invalid one falls back to the full verification
    /// rather than loading unchecked.
    public init(
        modelDirectory: URL,
        maximumContext: Int,
        thinking: ModelThinkingMode,
        reasoningEffort: ModelReasoningEffort?,
        slots: Int? = nil
    ) async throws {
        let config = try ArchConfig.resolved(
            forFamily: .k2Horizon, directoryURL: modelDirectory)
        let identity = try ManifestReader.peekIdentity(directoryURL: modelDirectory)
        let resolvedSlots = slots ?? (identity.weightBits == 4 ? 64 : 16)
        guard resolvedSlots > 0 else {
            throw K2BackendError.unsupported("expert cache slots must be positive")
        }
        let metal = try MetalContext()
        let loaded: Model
        do {
            loaded = try Model.load(
                directoryURL: modelDirectory, device: metal.device,
                expecting: config, streamingMode: .pread(slotCount: resolvedSlots),
                integrityPolicy: .sizeCheckTrustedReceipt)
        } catch let error as ModelError {
            guard case .trustedReceiptInvalid = error else { throw error }
            loaded = try Model.load(
                directoryURL: modelDirectory, device: metal.device,
                expecting: config, streamingMode: .pread(slotCount: resolvedSlots))
        }
        guard
            let folder = GFTokenizer.resolvedTokenizerFolder(
                forModelDirectory: modelDirectory)
        else {
            throw K2BackendError.unsupported(
                "no tokenizer in \(modelDirectory.lastPathComponent)")
        }
        self.model = loaded
        self.context = metal
        self.slotCount = resolvedSlots
        self.tokenizerFolder = folder
        self.tokenizer = try await GFTokenizer.load(
            from: folder, thinkingMode: thinking, reasoningEffort: reasoningEffort)
        self.loadedReasoning = RequestReasoning(
            thinkingMode: thinking, effort: reasoningEffort)
        self.contextLimit = min(maximumContext, Self.contextCeiling)
        self.defaults = GenerationDefaults.forFamily(.k2Horizon)
    }

    package func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        let renderTokenizer = try await resolvedTokenizer(for: request.reasoning)
        let promptIDs = try encodePrompt(
            renderTokenizer, messages: request.messages, tools: request.tools)
        guard !promptIDs.isEmpty else { throw K2BackendError.emptyPrompt }
        guard promptIDs.count < contextLimit else {
            throw K2BackendError.promptTooLong(promptIDs.count, contextLimit)
        }
        let budget = min(
            request.maximumCompletionTokens, contextLimit - promptIDs.count)
        let configuration = structuredConfiguration(request)
        let sampler = CPUSampler(
            temperature: configuration.temperature,
            topP: configuration.topP ?? 1,
            topK: configuration.topK ?? 0,
            presencePenalty: configuration.presencePenalty,
            repetitionPenalty: configuration.repetitionPenalty,
            seed: configuration.seed)
        let generator = sampler.makeGenerator()

        try ensureGraph(needed: promptIDs.count + budget)
        guard let graph else {
            throw K2BackendError.unsupported("the probe graph is not resident")
        }
        // At least one token is always forwarded: its logits are what the
        // sampler reads. Everything before it is K/V the previous request
        // already computed from the same ids.
        let reused = min(
            Self.commonPrefixLength(promptIDs, cachedTokens),
            max(0, promptIDs.count - 1))
        if reused > 0,
            ProcessInfo.processInfo.environment["TINYTITAN_K2_TIMING"] == "1"
        {
            FileHandle.standardError.write(
                Data(
                    ("k2 server: reused \(reused)/\(promptIDs.count) prompt tokens "
                        + "(capacity \(graphCapacity))\n").utf8))
        }
        var logits: [Float] = []
        for position in reused..<promptIDs.count {
            logits = try await graph.forward(
                tokenID: UInt32(bitPattern: promptIDs[position]),
                position: UInt32(position))
        }
        cachedTokens = Array(promptIDs.prefix(promptIDs.count))

        let allowedTools =
            request.tools.isEmpty ? nil : Set(request.tools.map(\.name))
        let decoder = StructuredAssistantDecoder.forGeneration(
            tokenizer: renderTokenizer, promptIDs: promptIDs,
            allowedTools: allowedTools)
        var detokenizer = GFDetokenizer(tokenizer: renderTokenizer)
        var output = AssistantOutput(
            stops: configuration.stopStrings, onEvent: onEvent)
        var history = promptIDs
        var produced = 0
        var position = promptIDs.count
        var reason = "length"
        let stops: Set<Int> = [
            Int(renderTokenizer.eosID), Int(renderTokenizer.endOfTurnID),
        ]
        while produced < budget {
            if let constraint = configuration.constraint {
                let mask = constraint.allowedMask()
                guard !mask.isEmpty else {
                    throw GeneratorError.constrainedDecodeStalled
                }
                mask.apply(toLogits: &logits)
            }
            let next = sampler.pick(logits, history: history, using: generator)
            if let constraint = configuration.constraint,
                !constraint.observe(Int32(next))
            {
                throw GeneratorError.constrainedDecodeViolation(id: Int32(next))
            }
            if stops.contains(next) {
                reason = "stop"
                break
            }
            produced += 1
            history.append(Int32(next))
            output.publish(
                try decoder.consume(
                    tokenID: Int32(next), delta: detokenizer.push(Int32(next))))
            if output.isStopped {
                reason = "stop"
                break
            }
            if produced >= budget { break }
            logits = try await graph.forward(
                tokenID: UInt32(bitPattern: Int32(next)),
                position: UInt32(position))
            cachedTokens.append(Int32(next))
            position += 1
        }
        output.publish(
            try decoder.consumeTail(detokenizer.flush()), isToken: false)
        try decoder.finish()
        output.finish()
        return completion(
            promptTokens: promptIDs.count, produced: produced,
            reason: reason, output: output)
    }

    /// The request's generation config with the structured-output constraint,
    /// over a token table built once per backend.
    private func structuredConfiguration(
        _ request: ValidatedChatRequest
    ) -> GenerationConfig {
        var configuration = request.generationConfig
        guard let node = request.jsonSchema else { return configuration }
        let table: JSONTokenTable
        if let existing = jsonTokenTable {
            table = existing
        } else {
            let made = JSONTokenTable(tokenizer: tokenizer)
            jsonTokenTable = made
            table = made
        }
        configuration.constraint = JSONConstraint(
            table: table, node: node, vocab: tokenizer.vocabSize)
        return configuration
    }

    /// The resident graph, grown when a request needs more sequence positions.
    /// Growth starts from `minimumGraphCapacity` so a normal conversation does
    /// not rebuild on its second turn; a rebuild drops the K/V cache, which is
    /// why the capacity is not this request's exact size.
    private func ensureGraph(needed: Int) throws {
        let wanted = min(contextLimit, max(needed, Self.minimumGraphCapacity))
        guard graph == nil || needed > graphCapacity else { return }
        let graphStart = Date()
        graph = try K2GraphProbe(
            model: model, context: context, maxSequence: wanted)
        graphCapacity = wanted
        cachedTokens = []
        if ProcessInfo.processInfo.environment["TINYTITAN_K2_TIMING"] == "1" {
            FileHandle.standardError.write(
                Data(
                    ("k2 server: graph built in "
                        + String(format: "%.2f", Date().timeIntervalSince(graphStart))
                        + "s (maxSequence \(wanted))\n").utf8))
        }
    }

    /// The completion object, from the accumulated stream.
    private func completion(
        promptTokens: Int, produced: Int, reason: String,
        output: AssistantOutput
    ) -> ServerCompletion {
        ServerCompletion(
            content: output.content,
            toolCalls: output.calls,
            finishReason: reason,
            usage: OpenAIUsage(
                promptTokens: promptTokens,
                completionTokens: produced,
                totalTokens: promptTokens + produced,
                cachedTokens: 0,
                reasoningTokens: output.reasoningTokens),
            stopSequence: output.matchedStop,
            reasoning: output.reasoning,
            // K2 is effort-only: a render always opens a thought block, so any
            // reasoning is requested.
            unrequestedReasoning: 0)
    }

    /// The number of leading token ids the two sequences share.
    private static func commonPrefixLength(_ lhs: [Int32], _ rhs: [Int32]) -> Int {
        var count = 0
        while count < lhs.count, count < rhs.count, lhs[count] == rhs[count] {
            count += 1
        }
        return count
    }

    private func resolvedTokenizer(
        for reasoning: RequestReasoning?
    ) async throws -> GFTokenizer {
        guard let reasoning, !reasoning.matches(loadedReasoning) else {
            return tokenizer
        }
        return try await GFTokenizer.load(
            from: tokenizerFolder,
            thinkingMode: reasoning.thinkingMode,
            reasoningEffort: reasoning.effort)
    }

    /// The same render the GPU path uses: the tool template when the request
    /// carries tools, the plain chat template otherwise.
    private func encodePrompt(
        _ renderTokenizer: GFTokenizer,
        messages: [GFTokenizer.Message],
        tools: [GFTokenizer.FunctionDefinition]
    ) throws -> [Int32] {
        if tools.isEmpty {
            let rendered = try renderTokenizer.applyChatTemplate(messages)
            return renderTokenizer.encode(rendered, addBOS: false)
        }
        return try renderTokenizer.encodeToolChat(messages: messages, tools: tools)
    }
}

/// The Messages API's count_tokens, from the same rendering `generate` uses.
extension K2ProbeBackend: PromptTokenCounting {
    package func countPromptTokens(_ request: ValidatedChatRequest) async throws -> Int {
        let renderTokenizer = try await resolvedTokenizer(for: request.reasoning)
        return try encodePrompt(
            renderTokenizer, messages: request.messages, tools: request.tools
        ).count
    }
}
