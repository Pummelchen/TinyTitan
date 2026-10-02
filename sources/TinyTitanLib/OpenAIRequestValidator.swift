import Foundation
import TinyTitan

/// Which request limits apply.
///
/// The structural rules are the same either way — a message needs content,
/// guidance precedes the conversation, a tool result names an open call. What
/// differs is the wire's *caps*, and they exist to protect the engine from a
/// third party's request rather than from its own embedder: four stop strings
/// and a thousand messages are OpenAI's numbers, and the CLI never had them
/// before it moved onto the facade.
package enum RequestRules: Sendable, Equatable {
    /// What arrived over HTTP.
    case wire
    /// A caller inside this process.
    case local
}

/// The OpenAI-compatible request validator: model resolution, sampling bounds,
/// message, tool and schema-key validation.
///
/// Split out of `OpenAIModels.swift` (2026-09-28) under the 500-line-per-file
/// rule (Task 8 of the cleanup runbook) as pure code motion; its `private`
/// helpers moved with the enum, so no access widened.
package enum OpenAIRequestValidator {
    /// lint:allow-long a straight-line validation cascade: each guard
    /// rejects one malformed field with its own error. Grouping them into
    /// sub-validators would add indirection without removing a single check.
    package static func validate(
        _ request: OpenAIChatRequest,
        modelID: String,
        maxContext: Int = RuntimeConfiguration
            .supportedContextTokens.max() ?? 262_144,
        reasoningProfile: ServerReasoningProfile = .default,
        // Filled in for a request that omits the value.
        // Defaults to the house settings so callers that
        // do not know the family keep today's behaviour.
        sampling: GenerationDefaults.Sampling = GenerationDefaults.house,
        rules: RequestRules = .wire
    ) throws -> ValidatedChatRequest {
        // The "<model>-fast" alias selects the same weights as the base model
        // but enables the CLI-strip heuristic per request (chat-only speed),
        // so tool-using clients keep the base model and chat users opt in.
        let fastModelID = modelID + "-fast"
        let stripCLIPrompt = request.model == fastModelID
        guard request.model == modelID || stripCLIPrompt else {
            throw ServerRequestError.unknownModel
        }
        guard request.n == nil || request.n == 1 else {
            throw invalid("only n=1 is supported", "n", "unsupported_value")
        }
        guard request.logprobs != true else {
            throw invalid("logprobs are not supported", "logprobs", "unsupported_value")
        }
        // A non-zero presence penalty is supported now: the sampler subtracts it
        // once per distinct id already in the history. This guard used to reject
        // every non-zero value, which is why Qwen3.8's published instruct row
        // (presence 1.5) could not be expressed.
        guard request.frequencyPenalty == nil || request.frequencyPenalty == 0 else {
            throw invalid(
                "frequency_penalty must be zero", "frequency_penalty", "unsupported_value")
        }
        // Reasoning effort is defined per family and fixed at model load
        // because it changes the rendered prompt. A request may not be able
        // to switch it, but it must never be refused for asking: coding
        // agents send vocabularies this project never defined (`xhigh` on a
        // model with no effort levels, `ultra`, `none`, `extra-high`), and
        // failing those breaks the agent for the rest of the session. So the
        // request is mapped to the nearest level the served model renders,
        // and the difference is recorded below rather than turned into an
        // error.
        var reasoningNotes: [String] = []
        var reasoning = RequestReasoning(
            thinkingMode: reasoningProfile.thinkingMode,
            effort: reasoningProfile.effectiveEffort)
        // The controls arrive in two dialects. `reasoning_effort` is this
        // project's own spelling; llama.cpp, vLLM and TabbyAPI clients put the
        // same controls in a `chat_template_kwargs` object, where
        // `enable_thinking: false` is the only way they turn thinking off.
        // Reading only the top-level field honoured the level beside that
        // object and dropped the switch. Precedence is explicit: an explicit
        // top-level effort wins, then the template-kwargs effort, then the
        // template switch (true -> on, false -> off).
        let requestedEffortRaw =
            request.reasoningEffort
            ?? request.chatTemplateKwargs?.reasoningEffort
            ?? request.chatTemplateKwargs?.enableThinking.map { $0 ? "on" : "off" }
        if request.reasoningBudgetTokens != nil {
            reasoningNotes.append(
                "reasoning_budget_tokens is accepted but not enforced; this runtime "
                    + "bounds thinking by the requested level, not by a token count")
        }
        if let effortRaw = requestedEffortRaw {
            let control = reasoningProfile.family.reasoningControl
            let supported = control.supportedLevels
            if let requested = ReasoningLevel.requested(effortRaw) {
                // The same mapping the server-wide level goes through, so a
                // request and `--reasoning` cannot disagree about what a
                // model does with a level it lacks.
                let efforts = supported.filter { $0 != .off && $0 != .on }
                let applied = ReasoningFallback.effectiveLevel(
                    requested, supported: supported,
                    whenOn: efforts.last)
                // This is what makes a mid-session switch real: the level is
                // carried into generation, which resolves a tokenizer for it
                // rather than re-rendering with the loaded one.
                reasoning = RequestReasoning(
                    thinkingMode: applied == .off ? .off : .on,
                    effort: applied == .off
                        ? nil
                        : ModelReasoningEffort(rawValue: applied.rawValue))
                if applied != requested {
                    reasoningNotes.append(
                        "reasoning level '\(effortRaw)' is not supported by this model; "
                            + "applied \(applied.displayName) instead (supports: "
                            + supported.map(\.displayName).joined(separator: ", ") + ")")
                }
            } else {
                // Unintelligible, not impossible: keep what the model was
                // loaded with rather than guessing at a level.
                reasoningNotes.append(
                    "reasoning level '\(effortRaw)' was not recognised; "
                        + "the model's own default applies (supports: "
                        + supported.map(\.displayName).joined(separator: ", ") + ")")
            }
        }
        // Structured output. The grammar constrains *every* token, so a
        // thought cannot be generated beside the document -- the template's
        // think block would have to be written as part of the JSON. Thinking
        // is therefore off for a request that names a format, and the note
        // says so rather than letting a client wonder why its level was
        // ignored.
        let jsonSchema = try structuredOutputSchema(request.responseFormat)
        if jsonSchema != nil {
            reasoning = RequestReasoning(thinkingMode: .off, effort: nil)
            reasoningNotes.append(
                "a JSON response format constrains every token, so thinking is off "
                    + "for this request")
        }
        // `parallel_tool_calls` is accepted and not enforced, on either value.
        // The decoder emits the calls the model produces, so the server cannot
        // promise one at a time; refusing the field would fail every client
        // that sends it defensively (the OpenAI SDKs default it, Codex sends
        // `false` on every turn) for a preference it cannot verify anyway. The
        // Responses object echoes what it was given; Chat Completions has no
        // field to echo into, which is why this is documented instead.
        // S17: include_usage is a streaming option; silently ignoring it on a
        // non-stream request hides a client bug.
        if request.streamOptions?.includeUsage == true, request.stream != true {
            throw invalid(
                "stream_options.include_usage requires stream=true",
                "stream_options", "invalid_value")
        }
        // S16: OpenAI forbids setting both bounds in one request.
        guard request.maxCompletionTokens == nil || request.maxTokens == nil else {
            throw invalid(
                "max_tokens and max_completion_tokens cannot both be set",
                "max_tokens", "invalid_value")
        }

        // Qwen3.8 publishes a different sampling row inside and outside thinking
        // mode, so the row is chosen from the mode the request actually runs in
        // rather than from the one the model was loaded with.
        let effectiveSampling: GenerationDefaults.Sampling
        switch reasoningProfile.family {
        case .qwen38flash, .qwen38flashMTP:
            effectiveSampling = GenerationDefaults.forFamily(
                reasoningProfile.family, thinking: reasoning.thinkingMode == .on)
        default:
            effectiveSampling = sampling
        }
        let temperature = request.temperature ?? effectiveSampling.temperature
        guard temperature >= 0, temperature <= 2 else {
            throw invalid(
                "temperature must be between 0 and 2",
                "temperature", "invalid_value")
        }
        let topP = request.topP ?? effectiveSampling.topP
        guard topP > 0, topP <= 1 else {
            throw invalid(
                "top_p must be greater than 0 and at most 1",
                "top_p", "invalid_value")
        }
        let topK = request.topK ?? effectiveSampling.topK
        guard (1...256).contains(topK) else {
            throw invalid("top_k must be between 1 and 256", "top_k", "invalid_value")
        }
        let repetitionPenalty = request.repetitionPenalty ?? 1
        guard repetitionPenalty > 0 else {
            throw invalid(
                "repetition_penalty must be positive",
                "repetition_penalty", "invalid_value")
        }
        // No artificial output cap: when the client omits max_tokens /
        // max_completion_tokens, generation is bounded only by the session's
        // configured context window (further clamped to the available context
        // at inference time), so the model replies until it is done.
        let maximum = request.maxCompletionTokens ?? request.maxTokens ?? maxContext
        guard maximum > 0 else {
            throw invalid(
                "maximum completion tokens must be positive",
                request.maxCompletionTokens != nil ? "max_completion_tokens" : "max_tokens",
                "invalid_value")
        }
        // S11: validate against the session's configured context window, not
        // the hard architectural ceiling.
        let cappedMaximum = min(maximum, maxContext)
        guard cappedMaximum == maximum else {
            throw invalid(
                "maximum completion tokens exceeds the configured context window (\(maxContext))",
                request.maxCompletionTokens != nil ? "max_completion_tokens" : "max_tokens",
                "value_too_large")
        }

        // S18: stop strings must be non-empty and unique. The count and total
        // length are the wire's caps, so a local caller is not held to them.
        let stopValues = request.stop?.values ?? []
        var stopStrings: [String] = []
        if !stopValues.isEmpty {
            guard stopValues.allSatisfy({ !$0.isEmpty }) else {
                throw invalid("stop strings must not be empty", "stop", "invalid_value")
            }
            if rules == .wire {
                guard stopValues.count <= 4 else {
                    throw invalid(
                        "at most 4 stop strings are supported", "stop", "value_too_large")
                }
                let totalLength = stopValues.reduce(0) { $0 + $1.utf8.count }
                guard totalLength <= 256 else {
                    throw invalid(
                        "stop strings must total at most 256 bytes", "stop", "value_too_large")
                }
            }
            var seen: Set<String> = []
            stopStrings = stopValues.filter { seen.insert($0).inserted }
        }

        let includeTools: Bool
        switch request.toolChoice {
        case nil, .some(.string("auto")):
            includeTools = true
        case .some(.string("none")):
            includeTools = false
        case .some(.string("required")):
            throw invalid(
                "tool_choice=required is not supported",
                "tool_choice", "unsupported_value")
        case .some(.bool(true)):
            // Legacy boolean form of "auto" (S31).
            includeTools = true
        case .some(.bool(false)):
            // Legacy boolean form of "none" (S31).
            includeTools = false
        default:
            throw invalid(
                "named tool choices are not supported",
                "tool_choice", "unsupported_value")
        }

        let tools = try (includeTools ? request.tools ?? [] : []).map {
            try validateTool($0)
        }
        let messages = try validateMessages(request.messages, rules: rules)
        // A client-supplied seed makes sampling deterministic.
        let config = GenerationConfig(
            maxNewTokens: maximum,
            temperature: temperature,
            topK: topK,
            topP: topP,
            presencePenalty: request.presencePenalty
                ?? effectiveSampling.presencePenalty,
            minP: effectiveSampling.minP,
            repetitionPenalty: repetitionPenalty,
            seed: request.seed,
            stopStrings: stopStrings)
        return ValidatedChatRequest(
            messages: messages,
            tools: tools,
            stream: request.stream ?? false,
            includeUsage: request.streamOptions?.includeUsage ?? false,
            generationConfig: config,
            maximumCompletionTokens: maximum,
            stripCLIPrompt: stripCLIPrompt,
            reasoningNotes: reasoningNotes,
            reasoning: reasoning,
            jsonSchema: jsonSchema)
    }

    /// The compiled schema a `response_format` asks for, or nil for plain text.
    ///
    /// Chat Completions' own spelling is the one parsed. The Responses surface
    /// and the Messages API reshape theirs into it before validation, so the
    /// rule lives in exactly one place: `{"type": "text"}` (the API's own
    /// default) and an unrecognized *shape* stay accepted, a `json_object`
    /// means an object at the top level, and a `json_schema` is compiled by
    /// `JSONSchemaNode` -- which refuses, by name, every keyword a byte-level
    /// grammar cannot promise.
    static func structuredOutputSchema(_ format: JSONValue?) throws -> JSONSchemaNode? {
        guard let format, case .object(let dict) = format,
            case .string(let type)? = dict["type"]
        else {
            return nil
        }
        switch type {
        case "text":
            return nil
        case "json_object":
            return .object(properties: [:], required: [], additional: true)
        case "json_schema":
            guard case .object(let wrapper)? = dict["json_schema"],
                let schema = wrapper["schema"]
            else {
                throw invalid(
                    "json_schema requires json_schema.schema",
                    "response_format.json_schema.schema", "invalid_value")
            }
            do {
                return try JSONSchemaNode.compile(schema)
            } catch let error as JSONSchemaCompileError {
                throw invalid(
                    error.description, "response_format.json_schema.schema",
                    "unsupported_value")
            }
        default:
            throw invalid(
                "response_format \(type) is not supported; use text, json_object or json_schema",
                "response_format", "unsupported_value")
        }
    }

    private static func validateTool(_ tool: OpenAITool) throws -> GFTokenizer.FunctionDefinition {
        guard tool.type == "function" else {
            throw invalid("only function tools are supported", "tools", "unsupported_tool")
        }
        let name = tool.function.name
        guard name.range(of: #"^[A-Za-z0-9_-]{1,64}$"#, options: .regularExpression) != nil else {
            throw invalid(
                "tool name must match [A-Za-z0-9_-]{1,64}",
                "tools", "invalid_tool_name")
        }
        guard tool.function.parameters.objectValue != nil else {
            throw invalid(
                "tool parameters must be an object schema",
                "tools", "invalid_tool_schema")
        }
        try validateSchemaKeys(tool.function.parameters)
        let parameters = tool.function.parameters
        guard (try? parameters.jinjaSendableValue()) != nil else {
            throw invalid(
                "tool schema contains a number that cannot be represented exactly",
                "tools", "invalid_tool_schema")
        }
        return GFTokenizer.FunctionDefinition(
            name: name,
            description: tool.function.description ?? "",
            parameters: parameters)
    }

    private static func validateSchemaKeys(_ schema: JSONValue) throws {
        switch schema {
        case .object(let object):
            for (schemaKey, value) in object {
                if schemaKey == "properties" {
                    guard case .object(let definitions) = value else {
                        throw invalid(
                            "tool schema properties must be an object",
                            "tools", "invalid_tool_schema")
                    }
                    for (_, definition) in definitions {
                        // ChatML tool-call parameter names are free-form;
                        // only the schema structure itself is validated.
                        try validateSchemaKeys(definition)
                    }
                } else {
                    try validateSchemaKeys(value)
                }
            }
        case .array(let values):
            for value in values {
                try validateSchemaKeys(value)
            }
        default:
            break
        }
    }

    private static func validateMessages(
        _ input: [OpenAIChatMessage],
        rules: RequestRules
    ) throws -> [GFTokenizer
        .Message]
    {
        guard !input.isEmpty else {
            throw invalid("messages must not be empty", "messages", "invalid_message")
        }
        // The thousand-message ceiling is the wire's; a local caller's history
        // is its own business.
        if rules == .wire {
            guard input.count <= 1000 else {
                throw invalid(
                    "message count exceeds maximum of 1000",
                    "messages", "value_too_large")
            }
        }
        var knownCalls: [String: (name: String, resolved: Bool)] = [:]
        var result: [GFTokenizer.Message] = []
        var sawConversationMessage = false
        for message in input {
            guard let role = GFTokenizer.Role(rawValue: message.role) else {
                throw invalid(
                    "unsupported message role \(message.role)",
                    "messages", "invalid_message")
            }
            if role == .system || role == .developer {
                guard !sawConversationMessage else {
                    throw invalid(
                        "system or developer guidance must precede the conversation",
                        "messages", "invalid_message")
                }
            } else {
                sawConversationMessage = true
            }
            let content = try message.content?.textValue()
            let calls: [GFTokenizer.HistoricalToolCall] = try (message.toolCalls ?? []).map {
                call in
                guard role == .assistant, call.type == "function",
                    !call.id.isEmpty, knownCalls[call.id] == nil,
                    call.function.name.range(
                        of: #"^[A-Za-z0-9_-]{1,64}$"#,
                        options: .regularExpression) != nil
                else {
                    throw invalid(
                        "invalid or duplicate historical tool call",
                        "messages", "invalid_tool_call")
                }
                let data = Data(call.function.arguments.utf8)
                let arguments = try JSONDecoder().decode(JSONValue.self, from: data)
                guard arguments.objectValue != nil else {
                    throw invalid(
                        "historical tool arguments must be a JSON object",
                        "messages", "invalid_tool_arguments")
                }
                guard (try? arguments.jinjaSendableValue()) != nil else {
                    throw invalid(
                        "historical tool arguments cannot be represented exactly",
                        "messages",
                        "invalid_tool_arguments")
                }
                knownCalls[call.id] = (call.function.name, false)
                return GFTokenizer.HistoricalToolCall(
                    id: call.id, name: call.function.name, arguments: arguments)
            }
            if role == .tool {
                guard let id = message.toolCallID,
                    let call = knownCalls[id], !call.resolved
                else {
                    throw invalid(
                        "tool result must reference one unresolved call",
                        "messages", "invalid_tool_result")
                }
                knownCalls[id] = (call.name, true)
                guard content != nil else {
                    throw invalid(
                        "tool result content is required",
                        "messages", "invalid_tool_result")
                }
            } else if content == nil && calls.isEmpty {
                throw invalid(
                    "message content is required",
                    "messages", "invalid_message")
            }
            result.append(
                GFTokenizer.Message(
                    role: role,
                    content: content,
                    toolCalls: calls,
                    toolCallID: message.toolCallID,
                    name: message.name))
        }
        // S19: a conversation that ends with an assistant tool call that is
        // never answered by a tool result would resume from an unanswerable
        // state; reject it instead of generating tool-response markup.
        if knownCalls.contains(where: { !$0.value.resolved }) {
            throw invalid(
                "conversation ends with an unresolved tool call",
                "messages", "invalid_tool_call")
        }
        return result
    }

    private static func invalid(
        _ message: String,
        _ param: String?,
        _ code: String
    ) -> ServerRequestError {
        .invalid(message: message, param: param, code: code)
    }
}
