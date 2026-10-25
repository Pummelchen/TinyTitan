import Foundation
import TinyTitan
import TinyTitanLib
import Tokenizers

/// Serves a K2-Horizon snapshot through the same HTTP surface as every other
/// model.
///
/// The model itself is the CPU text path in `K2Model`; this type is the thin
/// request layer around it: render the checkpoint's own chat template (through
/// the generic swift-transformers tokenizer, because K2's special tokens are
/// not Qwen's), prefill, sample with the request's settings, and stream the
/// answer.
///
/// What it deliberately does not do yet: tool-call parsing (`k2_horizon`'s
/// three tool formats), a separate reasoning channel for
/// `<ifm|think>` text, and a prompt cache. Those are named in
/// `docs/plan-k2-horizon.md` rather than pretended here.
public actor K2ModelBackend: ServerInferenceBackend {

    private let model: K2Model
    private let tokenizer: Tokenizer
    private let template: String
    private let contextLimit: Int

    public nonisolated let maximumContext: Int
    public nonisolated let samplingDefaults: GenerationDefaults.Sampling

    public init(
        snapshotDirectory: URL,
        maximumContext: Int
    ) async throws {
        let snapshot = try K2Snapshot(directory: snapshotDirectory)
        model = try K2Model(snapshot: snapshot)
        guard let folder = GFTokenizer.resolvedTokenizerFolder(forModelDirectory: snapshotDirectory)
        else {
            throw CPUModelBackend.CPUBackendError.unsupported(
                "\(snapshotDirectory.path) has no tokenizer.json")
        }
        tokenizer = try await AutoTokenizer.from(modelFolder: folder)
        // lint:allow-unbounded-read the checkpoint's own template: 52 KB,
        // written by the converter, and read once per process.
        template = try String(
            contentsOf: folder.appendingPathComponent("chat_template.jinja"),
            encoding: .utf8)
        let limit = max(256, min(maximumContext, snapshot.configuration.maxPositions))
        contextLimit = limit
        self.maximumContext = limit
        // The card's recommendation: temperature 1.0, top-p 0.95, high effort.
        samplingDefaults = GenerationDefaults.Sampling(
            temperature: 1.0, topK: GenerationDefaults.topK, topP: 0.95)
    }

    package func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        let messages: [Message] = request.messages.map { message in
            ["role": message.role.rawValue, "content": message.content ?? ""]
        }
        let prompt = try tokenizer.applyChatTemplate(
            messages: messages, chatTemplate: template)
        guard !prompt.isEmpty else {
            throw CPUModelBackend.CPUBackendError.unsupported(
                "the chat template rendered no prompt")
        }
        guard prompt.count < contextLimit else {
            throw CPUModelBackend.CPUBackendError.promptTooLong(prompt.count, contextLimit)
        }
        let budget = min(
            min(request.maximumCompletionTokens, request.generationConfig.maxNewTokens),
            contextLimit - prompt.count)

        model.reset()
        var logits: [Float] = []
        for (index, token) in prompt.enumerated() {
            logits = try model.step(token: token, needsLogits: index == prompt.count - 1)
        }
        onEvent(.promptProcessed(tokens: prompt.count, cachedTokens: 0))

        let sampler = CPUModelBackend.cpuSampler(for: request.generationConfig)
        let generator = sampler.makeGenerator()
        var history = prompt.map(Int32.init)
        var output = AssistantOutput(
            stops: request.generationConfig.stopStrings, onEvent: onEvent)
        var tokenizerState = IncrementalDetokenizer(tokenizer: tokenizer)
        var produced = 0
        var finishReason = "length"
        let eos = tokenizer.eosTokenId

        while produced < budget {
            let next = sampler.pick(logits, history: history, using: generator)
            if next == eos || request.generationConfig.extraStopTokens.contains(Int32(next)) {
                finishReason = "stop"
                break
            }
            produced += 1
            history.append(Int32(next))
            let delta = tokenizerState.push(Int32(next))
            output.publish([.content(delta)], isToken: true)
            if output.isStopped {
                finishReason = "stop"
                break
            }
            logits = try model.step(token: next)
        }
        output.publish([.content(tokenizerState.flush())], isToken: false)
        output.finish()

        return ServerCompletion(
            content: output.content,
            toolCalls: [],
            finishReason: finishReason,
            usage: OpenAIUsage(
                promptTokens: prompt.count,
                completionTokens: produced,
                totalTokens: prompt.count + produced,
                cachedTokens: 0,
                reasoningTokens: 0),
            stopSequence: nil,
            reasoning: "",
            unrequestedReasoning: 0)
    }
}

/// Token-by-token decoding: decode the ids seen so far and emit the suffix
/// that has not been sent.
///
/// The generic tokenizer has no streaming detokenizer, and a per-token decode
/// that skipped the previous text would split multi-byte characters; decoding
/// the prefix and taking the new suffix keeps them whole. Answers here are
/// short enough that the quadratic rescan is free next to a token's weight
/// reads.
private struct IncrementalDetokenizer {
    let tokenizer: Tokenizer
    private var ids: [Int] = []
    private var emitted = ""

    init(tokenizer: Tokenizer) {
        self.tokenizer = tokenizer
    }

    mutating func push(_ id: Int32) -> String {
        ids.append(Int(id))
        let text = tokenizer.decode(tokens: ids, skipSpecialTokens: true)
        guard text.count >= emitted.count, text.hasPrefix(emitted) else {
            emitted = text
            return text
        }
        let delta = String(text.dropFirst(emitted.count))
        emitted = text
        return delta
    }

    mutating func flush() -> String {
        ""
    }
}
