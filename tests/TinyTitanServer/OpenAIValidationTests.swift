import Foundation
import NIOHTTP1
import Testing

@testable import TinyTitan
@testable import TinyTitanKit
@testable import TinyTitanServerCore

@Suite("OpenAI request validation")
struct OpenAIValidationTests {
    @Test func omittedSamplingControlsUseProductionDefaults() throws {
        let data = Data(#"{"model":"m","messages":[{"role":"user","content":"x"}]}"#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)

        let validated = try OpenAIRequestValidator.validate(request, modelID: "m")

        #expect(validated.generationConfig.temperature == 0.6)
        #expect(validated.generationConfig.topK == 20)
        #expect(validated.generationConfig.topP == 0.95)
        #expect(validated.generationConfig.presencePenalty == 0)
    }

    @Test func qwen38SelectsItsSamplingRowFromTheThinkingMode() throws {
        let data = Data(#"{"model":"m","messages":[{"role":"user","content":"x"}]}"#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)

        let on = try OpenAIRequestValidator.validate(
            request, modelID: "m",
            reasoningProfile: ServerReasoningProfile(
                family: .qwen38flash,
                thinkingMode: .on,
                reasoningEffort: nil),
            sampling: GenerationDefaults.qwen38Thinking)
        #expect(on.generationConfig.temperature == 1.0)
        #expect(on.generationConfig.topP == 0.95)
        #expect(on.generationConfig.topK == 20)
        #expect(on.generationConfig.presencePenalty == 0)

        // The same served model, asked for non-thinking, must take the instruct
        // row even though the model was loaded with the thinking one.
        let off = try OpenAIRequestValidator.validate(
            request, modelID: "m",
            reasoningProfile: ServerReasoningProfile(
                family: .qwen38flash,
                thinkingMode: .off,
                reasoningEffort: nil),
            sampling: GenerationDefaults.qwen38Thinking)
        #expect(off.generationConfig.temperature == 0.7)
        #expect(off.generationConfig.topP == 0.80)
        #expect(off.generationConfig.topK == 20)
        #expect(off.generationConfig.presencePenalty == 1.5)
    }

    @Test func requiredToolChoiceIsRejected() throws {
        let data = Data(
            #"""
            {"model":"m","messages":[{"role":"user","content":"x"}],"tool_choice":"required"}
            """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        #expect(throws: ServerRequestError.self) {
            try OpenAIRequestValidator.validate(request, modelID: "m")
        }
    }

    @Test func acceptsLeadingSystemAndDeveloperGuidance() throws {
        let data = Data(
            #"""
            {"model":"m","messages":[
              {"role":"system","content":"system"},
              {"role":"developer","content":"developer"},
              {"role":"user","content":"hello"}
            ]}
            """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let validated = try OpenAIRequestValidator.validate(request, modelID: "m")
        #expect(validated.messages.map(\.role) == [.system, .developer, .user])
    }

    @Test func rejectsLateDeveloperGuidance() throws {
        let data = Data(
            #"""
            {"model":"m","messages":[
              {"role":"user","content":"hello"},
              {"role":"developer","content":"late"}
            ]}
            """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        #expect(throws: ServerRequestError.self) {
            try OpenAIRequestValidator.validate(request, modelID: "m")
        }
    }

    @Test func fastAliasSelectsStripButBaseModelDoesNot() throws {
        let base = try JSONDecoder().decode(
            OpenAIChatRequest.self,
            from: Data(#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#.utf8))
        let baseValidated = try OpenAIRequestValidator.validate(base, modelID: "m")
        #expect(baseValidated.stripCLIPrompt == false)

        let fast = try JSONDecoder().decode(
            OpenAIChatRequest.self,
            from: Data(#"{"model":"m-fast","messages":[{"role":"user","content":"hi"}]}"#.utf8))
        let fastValidated = try OpenAIRequestValidator.validate(fast, modelID: "m")
        #expect(fastValidated.stripCLIPrompt == true)
    }

    @Test func fastAliasOfUnservedModelIsRejected() throws {
        let data = Data(
            #"""
            {"model":"nope-fast","messages":[{"role":"user","content":"hi"}]}
            """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        #expect(throws: ServerRequestError.self) {
            try OpenAIRequestValidator.validate(request, modelID: "m")
        }
    }

    @Test func wideIntegerToolArgumentsRoundTripExactly() async throws {
        let expected = "9007199254740993"
        let parsed = try QwenToolCallParser().parse(
            "\n<function=lookup>\n<parameter=id>\n\(expected)\n</parameter>\n</function>\n",
            allowedTools: ["lookup"],
            id: "call_0123456789abcdef01234567")
        #expect(parsed.argumentsJSON.contains(#""id":\#(expected)"#))
        let signedMinimum = String(Int64.min)
        let signedMaximum = String(Int64.max)
        let unsignedMaximum = String(UInt64.max)
        let edges = try QwenToolCallParser().parse(
            "\n<function=lookup>\n<parameter=minimum>\n\(signedMinimum)\n</parameter>\n"
                + "<parameter=maximum>\n\(signedMaximum)\n</parameter>\n"
                + "<parameter=unsigned>\n\(unsignedMaximum)\n</parameter>\n</function>\n",
            allowedTools: ["lookup"],
            id: "call_0123456789abcdef01234568")
        #expect(edges.arguments.objectValue?["minimum"] == .integer(.min))
        #expect(edges.arguments.objectValue?["maximum"] == .integer(.max))
        #expect(edges.arguments.objectValue?["unsigned"] == .unsignedInteger(.max))
        let encodedEdges = try edges.arguments.encoded()
        #expect(encodedEdges.contains(signedMinimum))
        #expect(encodedEdges.contains(signedMaximum))
        #expect(encodedEdges.contains(unsignedMaximum))
        #expect(
            try JSONDecoder().decode(
                JSONValue.self,
                from: Data(encodedEdges.utf8)) == edges.arguments)
        // The Qwen parser keeps non-JSON parameter values as raw strings
        // (no strict numeric grammar); malformed-number rejection lives in
        // QwenToolCallParserTests via the JSONValue decode path.

        let data = Data(
            #"""
            {
              "model":"m",
              "messages":[
                {"role":"user","content":"lookup"},
                {"role":"assistant","tool_calls":[{
                  "id":"call_0123456789abcdef01234567",
                  "type":"function",
                  "function":{"name":"lookup","arguments":"{\"id\":9007199254740993}"}
                }]},
                {"role":"tool","tool_call_id":"call_0123456789abcdef01234567","content":"ok"}
              ],
              "tools":[{
                "type":"function",
                "function":{
                  "name":"lookup",
                  "parameters":{"type":"object","properties":{"id":{"type":"integer"}}}
                }
              }]
            }
            """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let validated = try OpenAIRequestValidator.validate(request, modelID: "m")
        let call = try #require(validated.messages[1].toolCalls.first)
        #expect(try call.arguments.encoded().contains(#""id":\#(expected)"#))
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let rendered = tokenizer.decode(
            try tokenizer.encodeToolChat(
                messages: validated.messages,
                tools: validated.tools),
            skipSpecialTokens: false)
        #expect(rendered.contains(expected))

        // 18446744073709551615 (UInt64.max) cannot be represented exactly as
        // an Int64 for the jinja tool renderer. The request shape is
        // otherwise valid (the tool call IS answered by a tool result), so
        // the rejection below is specifically about the unrepresentable
        // number, not the S19 unresolved-tool-call check.
        let unrepresentableHistory = Data(
            #"""
            {
              "model":"m",
              "messages":[
                {"role":"user","content":"lookup"},
                {"role":"assistant","tool_calls":[{
                  "id":"call_0123456789abcdef01234569",
                  "type":"function",
                  "function":{"name":"lookup","arguments":"{\"id\":18446744073709551615}"}
                }]},
                {"role":"tool","tool_call_id":"call_0123456789abcdef01234569","content":"ok"}
              ],
              "tools":[{
                "type":"function",
                "function":{
                  "name":"lookup",
                  "parameters":{"type":"object","properties":{"id":{"type":"integer"}}}
                }
              }]
            }
            """#.utf8)
        let rejected = try JSONDecoder().decode(
            OpenAIChatRequest.self,
            from: unrepresentableHistory)
        #expect(throws: ServerRequestError.self) {
            try OpenAIRequestValidator.validate(rejected, modelID: "m")
        }
    }

    @Test func acceptedNonIdentifierParameterKeysParseAndRender() async throws {
        let data = Data(
            #"""
            {
              "model":"m",
              "messages":[{"role":"user","content":"lookup"}],
              "tools":[{
                "type":"function",
                "function":{
                  "name":"lookup",
                  "parameters":{
                    "type":"object",
                    "properties":{
                      "$id":{"type":"string"},
                      "file-path":{"type":"string"},
                      "nested":{"type":"object","properties":{"child-key":{"type":"integer"}}}
                    }
                  }
                }
              }]
            }
            """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let validated = try OpenAIRequestValidator.validate(request, modelID: "m")
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        _ = try tokenizer.encodeToolChat(
            messages: validated.messages,
            tools: validated.tools)
        let parsed = try QwenToolCallParser().parse(
            #"""
            <function=lookup>
            <parameter=$id>
            item
            </parameter>
            <parameter=file-path>
            /tmp/x
            </parameter>
            <parameter=nested>
            {"child-key":7}
            </parameter>
            </function>
            """#,
            allowedTools: ["lookup"],
            id: "call_0123456789abcdef01234567")
        #expect(parsed.arguments.objectValue?["$id"] == .string("item"))
        #expect(parsed.arguments.objectValue?["file-path"] == .string("/tmp/x"))
        #expect(
            parsed.arguments.objectValue?["nested"]
                == .object(["child-key": .integer(7)]))
    }

    @Test func freeFormParameterNamesAreAccepted() throws {
        let data = Data(
            #"""
            {
              "model":"m",
              "messages":[{"role":"user","content":"lookup"}],
              "tools":[{
                "type":"function",
                "function":{
                  "name":"lookup",
                  "parameters":{
                    "type":"object",
                    "allOf":[{
                      "type":"object",
                      "properties":{"bad:key":{"type":"string"}}
                    }]
                  }
                }
              }]
            }
            """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        // S-audit: tool parameter names are deliberately free-form (only the
        // schema structure is validated), even inside allOf compositions.
        let validated = try OpenAIRequestValidator.validate(request, modelID: "m")
        #expect(validated.tools.count == 1)
    }

    /// The Chat Completions spelling of the Responses API's `text.format`.
    /// A named JSON format compiles into a schema the sampler is masked with;
    /// the API's own default and an unrecognized shape stay free text, and a
    /// *named* format that is not JSON is still refused rather than answered
    /// with prose. The grammar itself is covered by `StructuredOutputTests`.
    @Test func carriesStructuredOutputFormatsAndRefusesUnknownOnes() throws {
        func validate(_ format: String) throws -> ValidatedChatRequest {
            let data = Data(
                #"""
                {"model":"m","messages":[{"role":"user","content":"hi"}],"response_format":\#(format)}
                """#.utf8)
            let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
            return try OpenAIRequestValidator.validate(request, modelID: "m")
        }
        #expect(
            try validate(#"{"type":"json_object"}"#).jsonSchema
                == .object(properties: [:], required: [], additional: true))
        // An empty schema says nothing, so it compiles to "any JSON value" --
        // the document grammar without a schema layer.
        #expect(
            try validate(#"{"type":"json_schema","json_schema":{"name":"x","schema":{}}}"#)
                .jsonSchema == .any)
        #expect(try validate(#"{"type":"text"}"#).jsonSchema == nil)
        #expect(throws: ServerRequestError.self) { _ = try validate(#"{"type":"xml"}"#) }
    }

    @Test func acceptsPlainTextResponseFormat() throws {
        let data = Data(
            #"""
            {"model":"m","messages":[{"role":"user","content":"hi"}],
             "response_format":{"type":"text"}}
            """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let validated = try OpenAIRequestValidator.validate(request, modelID: "m")
        #expect(validated.messages.count == 1)
    }

    /// Accepted, not enforced: the decoder emits the calls the model produces,
    /// so a client that asks for one-at-a-time cannot be promised it — and
    /// refusing the field would fail every client that sends it defensively
    /// (the OpenAI SDKs default it; Codex sends `false` on every turn).
    @Test func acceptsParallelToolCallsEitherWay() throws {
        for value in ["true", "false"] {
            let data = Data(
                #"""
                {"model":"m","messages":[{"role":"user","content":"lookup"}],
                 "parallel_tool_calls":\#(value),
                 "tools":[{"type":"function","function":{"name":"lookup",
                           "parameters":{"type":"object"}}}]}
                """#.utf8)
            let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
            let validated = try OpenAIRequestValidator.validate(request, modelID: "m")
            #expect(validated.tools.count == 1, "parallel_tool_calls=\(value) dropped the tools")
        }
    }
}

@Suite("Streaming stop matcher")
struct StreamingStopMatcherTests {
    @Test func withholdsCrossChunkStop() {
        var matcher = StreamingStopMatcher(stops: ["END"])
        #expect(matcher.push("hello E") == "hello ")
        #expect(matcher.push("N") == "")
        #expect(matcher.push("D ignored") == "")
        #expect(matcher.isStopped)
    }

    @Test func flushesUnicodeTail() {
        var matcher = StreamingStopMatcher(stops: ["🌳stop"])
        #expect(matcher.push("hello 🌳") == "hello ")
        #expect(matcher.finish() == "🌳")
    }
}

@Suite("Server arguments")
struct ServerArgumentTests {
    @Test func defaults() throws {
        let arguments = try ServerArguments.parse(
            ["--model", "model.ssdai"], environment: [:])
        #expect(arguments.mtpModel == nil)
        #expect(arguments.mtpMemoryMiB == 384)
        #expect(arguments.port == 8080)
        #expect(arguments.maxContext == 262_144)
        #expect(arguments.queueLimit == 4)
        #expect(arguments.promptCacheMode == .multiPrefix)
        #expect(arguments.promptCacheMaximumEntries == 4)
        #expect(arguments.promptCacheMemoryMiB == 256)
        #expect(arguments.promptCacheDiskDirectory == nil)
        #expect(arguments.promptCacheDiskMiB == 8_192)
        #expect(arguments.prefillChunkTokens == nil)
        #expect(arguments.kvCachePrecision == .int8)
        #expect(arguments.ropeScalingMode == .none)
        #expect(arguments.thinkingMode == .off)
    }

    @Test func parsesOnlyBinaryThinkingModes() throws {
        let on = try ServerArguments.parse([
            "--model", "model.ssdai", "--thinking", "on",
        ])
        #expect(on.thinkingMode == .on)
        let environmentOn = try ServerArguments.parse(
            ["--model", "model.ssdai"],
            environment: ["TINYTITAN_THINKING_MODE": "true"])
        #expect(environmentOn.thinkingMode == .on)
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse([
                "--model", "model.ssdai", "--thinking", "high",
            ])
        }
    }

    @Test func parsesKVPrecisionAndYaRNContexts() throws {
        let defaults = try ServerArguments.parse([
            "--model", "model.ssdai", "--kv-bits", "16",
            "--rope-scaling", "yarn",
        ])
        #expect(defaults.kvCachePrecision == .fp16)
        #expect(defaults.ropeScalingMode == .yarn)
        #expect(defaults.maxContext == 1_048_576)
        let halfMillion = try ServerArguments.parse([
            "--model", "model.ssdai", "--rope-scaling", "yarn",
            "--max-context", "524288",
        ])
        #expect(halfMillion.maxContext == 524_288)
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse([
                "--model", "model.ssdai", "--max-context", "524288",
            ])
        }
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse([
                "--model", "model.ssdai", "--mtp-model", "mtp.ssdai",
                "--rope-scaling", "yarn",
            ])
        }
    }

    @Test func acceptsPublicPrefillChunksAndRejectsUnsupportedValues() throws {
        let arguments = try ServerArguments.parse([
            "--model", "model.ssdai",
            "--prefill-chunk", "4096",
        ])
        #expect(arguments.prefillChunkTokens == 4_096)
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse([
                "--model", "model.ssdai",
                "--prefill-chunk", "8192",
            ])
        }
    }

    @Test func parsesBoundedMTPOptions() throws {
        let arguments = try ServerArguments.parse([
            "--model", "qwen.ssdai",
            "--mtp-model", "qwen-mtp.ssdai",
            "--mtp-memory-mib", "512",
        ])
        #expect(arguments.mtpModel == "qwen-mtp.ssdai")
        #expect(arguments.mtpMemoryMiB == 512)
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse([
                "--model", "qwen.ssdai",
                "--mtp-memory-mib", "1024",
            ])
        }
    }

    @Test func mtpForcesPromptCacheOff() {
        #expect(
            ServerModelSession.effectivePromptCacheMode(
                requested: .multiPrefix,
                mtpEnabled: true) == .off)
        #expect(
            ServerModelSession.effectivePromptCacheMode(
                requested: .multiPrefix,
                mtpEnabled: false) == .multiPrefix)
    }

    @Test func parsesSinglePrefixModeAndRejectsUnknownMode() throws {
        let arguments = try ServerArguments.parse([
            "--model", "model.ssdai",
            "--prompt-cache-mode", "single-prefix",
        ])
        #expect(arguments.promptCacheMode == .singlePrefix)
        let multi = try ServerArguments.parse([
            "--model", "model.ssdai",
            "--prompt-cache-mode", "multi-prefix",
            "--prompt-cache-entries", "8",
            "--prompt-cache-memory-mib", "512",
            "--prompt-cache-disk", "/tmp/tinytitan-cache",
            "--prompt-cache-disk-mib", "16384",
        ])
        #expect(multi.promptCacheMode == .multiPrefix)
        #expect(multi.promptCacheMaximumEntries == 8)
        #expect(multi.promptCacheMemoryMiB == 512)
        #expect(multi.promptCacheDiskDirectory == "/tmp/tinytitan-cache")
        #expect(multi.promptCacheDiskMiB == 16_384)
        let rollback = try ServerArguments.parse([
            "--model", "model.ssdai",
            "--prompt-cache-mode", "off",
        ])
        #expect(rollback.promptCacheMode == .off)
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse([
                "--model", "model.ssdai",
                "--prompt-cache-mode", "many",
            ])
        }
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse([
                "--model", "model.ssdai",
                "--prompt-cache-entries", "0",
            ])
        }
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse([
                "--model", "model.ssdai",
                "--prompt-cache-memory-mib", "4097",
            ])
        }
    }

    @Test func accepts256KContextAndRejectsUnsupportedValues() throws {
        let arguments = try ServerArguments.parse([
            "--model", "model.ssdai",
            "--max-context", "262144",
        ])
        #expect(arguments.maxContext == 262_144)
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse([
                "--model", "model.ssdai",
                "--max-context", "100000",
            ])
        }
    }

    private static func effortRequest(_ effort: String) throws -> OpenAIChatRequest {
        try JSONDecoder().decode(
            OpenAIChatRequest.self,
            from: Data(
                """
                {"model":"m","messages":[{"role":"user","content":"x"}],\
                "reasoning_effort":"\(effort)"}
                """.utf8))
    }

    /// A coding agent that names a level this model cannot render must keep
    /// working. Before this policy, `reasoning_effort` on a binary-thinking
    /// model was a 400, which broke OpenCode, Zed and Qoder for the rest of
    /// the session; now the nearest supported level is applied and reported.
    @Test func reasoningEffortIsClampedForBinaryFamilies() throws {
        let request = try Self.effortRequest("low")
        // The default profile is the compatible Qwen3.5-MoE baseline, whose
        // template defines only the binary switch.
        let validated = try OpenAIRequestValidator.validate(request, modelID: "m")
        #expect(
            validated.reasoningNotes.count == 1,
            "the clamp is reported rather than silent")
        #expect(validated.reasoningNotes[0].contains("low"))
    }

    /// The agent keeps its work: the request validates, and the note says
    /// what the model will actually do instead.
    @Test func reasoningEffortReportsWhatWasApplied() throws {
        let profile = ServerReasoningProfile(
            family: .qwen38flash,
            thinkingMode: .on,
            reasoningEffort: nil)
        // With no override the template default xhigh is the active level, so
        // asking for it is exact and silent.
        let exact = try OpenAIRequestValidator.validate(
            try Self.effortRequest("xhigh"), modelID: "m",
            reasoningProfile: profile)
        #expect(exact.reasoningNotes.isEmpty)

        // The template defines low, medium and xhigh. It does not define
        // `high`; it must land on the nearest it has, not fail. `high` sits
        // between `medium` and `xhigh`, and the router's rule sends an
        // equidistant level to the cheaper one, so this is `medium`.
        let high = try OpenAIRequestValidator.validate(
            try Self.effortRequest("high"), modelID: "m",
            reasoningProfile: profile)
        #expect(!high.reasoningNotes.isEmpty)
        #expect(high.reasoningNotes[0].contains("medium"))

        // `max` is above every effort the template defines, so it is not a
        // tie: the top of the ladder is the nearest thing to it.
        let max = try OpenAIRequestValidator.validate(
            try Self.effortRequest("max"), modelID: "m",
            reasoningProfile: profile)
        #expect(!max.reasoningNotes.isEmpty)
        #expect(max.reasoningNotes[0].contains("extra high"))

        let lowProfile = ServerReasoningProfile(
            family: .qwen38flash,
            thinkingMode: .on,
            reasoningEffort: .low)
        let low = try OpenAIRequestValidator.validate(
            try Self.effortRequest("low"), modelID: "m",
            reasoningProfile: lowProfile)
        #expect(low.reasoningNotes.isEmpty)
    }

    /// The request carries the level the session must render at, which is
    /// what makes a mid-session switch real rather than a log line.
    @Test func reasoningEffortIsCarriedForTheSession() throws {
        // Loaded on/off and thinking off: a request naming an effort level
        // must come back wanting thinking on.
        let binary = ServerReasoningProfile(
            family: .qwen36,
            thinkingMode: .off,
            reasoningEffort: nil)
        let wantsThinking = try OpenAIRequestValidator.validate(
            try Self.effortRequest("xhigh"), modelID: "m",
            reasoningProfile: binary)
        #expect(wantsThinking.reasoning?.thinkingMode == .on)

        // And naming "off" on a server loaded with thinking on must come back
        // wanting it off -- the switch a coding agent reaches for.
        let effortOn = ServerReasoningProfile(
            family: .qwen38flash,
            thinkingMode: .on,
            reasoningEffort: .xhigh)
        let wantsOff = try OpenAIRequestValidator.validate(
            try Self.effortRequest("off"), modelID: "m",
            reasoningProfile: effortOn)
        #expect(wantsOff.reasoning?.thinkingMode == .off)
        #expect(wantsOff.reasoning?.effort == nil)

        // An effort the model defines is carried verbatim.
        let wantsLow = try OpenAIRequestValidator.validate(
            try Self.effortRequest("low"), modelID: "m",
            reasoningProfile: effortOn)
        #expect(wantsLow.reasoning?.thinkingMode == .on)
        #expect(wantsLow.reasoning?.effort == .low)

        // Omitting the field means "what the model was loaded with", so the
        // session keeps its own tokenizer.
        let omitted = try JSONDecoder().decode(
            OpenAIChatRequest.self,
            from: Data(#"{"model":"m","messages":[{"role":"user","content":"x"}]}"#.utf8))
        let loaded = try OpenAIRequestValidator.validate(
            omitted, modelID: "m", reasoningProfile: effortOn)
        #expect(loaded.reasoning?.thinkingMode == .on)
        #expect(loaded.reasoning?.effort == .xhigh)
        #expect(loaded.reasoningNotes.isEmpty)
    }

    /// A request carrying `extra` beside the usual fields.
    private static func request(withExtra extra: String) throws -> OpenAIChatRequest {
        let json = #"{"model":"m","messages":[{"role":"user","content":"x"}],"# + extra + "}"
        return try JSONDecoder().decode(OpenAIChatRequest.self, from: Data(json.utf8))
    }

    /// The template-kwargs dialect is how llama.cpp, vLLM and TabbyAPI clients
    /// carry the thinking controls, and `enable_thinking: false` there is the
    /// only way they turn thinking off. Reading just the top-level field
    /// honoured the level beside that object and dropped the switch. That is the
    /// case that matters: a client which forces thinking off for its
    /// summarization calls -- so the model's own thinking cannot eat the output
    /// cap and truncate the summary -- had the fix silently lost.
    @Test func templateKwargsTurnThinkingOff() throws {
        let loadedOn = ServerReasoningProfile(
            family: .qwen38flash,
            thinkingMode: .on,
            reasoningEffort: .xhigh)
        let off = try OpenAIRequestValidator.validate(
            try Self.request(withExtra: #""chat_template_kwargs":{"enable_thinking":false}"#),
            modelID: "m", reasoningProfile: loadedOn)
        #expect(off.reasoning?.thinkingMode == .off)
        #expect(off.reasoning?.effort == nil)

        // `true` is the switch the other way: it must turn a thinking-off
        // server on rather than being ignored as a no-op.
        let loadedOff = ServerReasoningProfile(
            family: .qwen36,
            thinkingMode: .off,
            reasoningEffort: nil)
        let on = try OpenAIRequestValidator.validate(
            try Self.request(withExtra: #""chat_template_kwargs":{"enable_thinking":true}"#),
            modelID: "m", reasoningProfile: loadedOff)
        #expect(on.reasoning?.thinkingMode == .on)
    }

    /// The object carries the level too, and an explicit top-level
    /// `reasoning_effort` wins over it. The precedence is pinned so a client
    /// that sends both cannot be surprised by which one applied.
    @Test func templateKwargsEffortIsHonouredAndTopLevelWins() throws {
        let profile = ServerReasoningProfile(
            family: .qwen38flash,
            thinkingMode: .on,
            reasoningEffort: .xhigh)
        let fromKwargs = try OpenAIRequestValidator.validate(
            try Self.request(withExtra: #""chat_template_kwargs":{"reasoning_effort":"low"}"#),
            modelID: "m", reasoningProfile: profile)
        #expect(fromKwargs.reasoning?.thinkingMode == .on)
        #expect(fromKwargs.reasoning?.effort == .low)

        let both = try OpenAIRequestValidator.validate(
            try Self.request(
                withExtra: #""reasoning_effort":"medium","#
                    + #""chat_template_kwargs":{"reasoning_effort":"low"}"#),
            modelID: "m", reasoningProfile: profile)
        #expect(
            both.reasoning?.effort == .medium,
            "the top-level field is this project's own spelling and takes precedence")
    }

    /// llama.cpp's per-request thinking budget is accepted and reported as
    /// unenforced rather than refused or dropped in silence: refusing a field
    /// the runtime does not implement would break the client for the rest of the
    /// session, and ignoring it without a word would hide the deviation.
    @Test func reasoningBudgetTokensIsAcceptedAndReported() throws {
        let validated = try OpenAIRequestValidator.validate(
            try Self.request(withExtra: #""reasoning_budget_tokens":4096"#), modelID: "m")
        #expect(
            validated.reasoningNotes.contains { $0.contains("reasoning_budget_tokens") },
            "an accepted-but-unenforced field must say so")
    }

    /// Every spelling a coding agent might send means something on the
    /// ladder, so none of them is a failure.
    @Test func unfamiliarReasoningVocabularyIsAccepted() throws {
        let profile = ServerReasoningProfile(
            family: .qwen38flash,
            thinkingMode: .on,
            reasoningEffort: nil)
        for word in [
            "ultra", "none", "extra-high", "thinking", "auto",
            "max", "minimal", "EXTRA HIGH", "highest",
        ] {
            let validated = try OpenAIRequestValidator.validate(
                try Self.effortRequest(word), modelID: "m",
                reasoningProfile: profile)
            // Either it mapped exactly (silent) or it was clamped (noted);
            // what it must never do is throw.
            #expect(validated.reasoningNotes.count <= 1)
        }
        // A word with no meaning on any ladder is still not fatal: the
        // model's own default applies.
        let unknown = try OpenAIRequestValidator.validate(
            try Self.effortRequest("galaxy-brain"), modelID: "m",
            reasoningProfile: profile)
        #expect(unknown.reasoningNotes.count == 1)
        #expect(unknown.reasoningNotes[0].contains("not recognised"))
    }

    @Test func responsesReasoningEffortMapsIntoTheChatRequest() throws {
        let decoded = try JSONDecoder().decode(
            ResponsesAPIRequest.self,
            from: Data(
                """
                {"model":"m","input":[{"type":"message","role":"user","content":"x"}],\
                "reasoning":{"effort":"medium"}}
                """.utf8))
        let chatRequest = try ResponsesAPIMapper.chatRequest(decoded)
        #expect(chatRequest.reasoningEffort == "medium")
    }

    @Test func serverArgumentsGateEffortOnThinking() throws {
        let parsed = try ServerArguments.parse([
            "--model", "model.ssdai",
            "--thinking", "on",
            "--reasoning-effort", "medium",
        ])
        #expect(parsed.reasoningEffort == .medium)
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse([
                "--model", "model.ssdai",
                "--reasoning-effort", "medium",
            ])
        }
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse([
                "--model", "model.ssdai",
                "--thinking", "on",
                "--reasoning-effort", "high",
            ])
        }
    }

    /// Header parsing, on its own. Trimming and the empty case matter
    /// because a proxy that adds `X-TinyTitan-Workspace:` with nothing after it
    /// must not create a workspace called "".
    @Test func workspaceHeaderIsReadAndTrimmed() {
        func head(_ value: String?) -> HTTPRequestHead {
            var headers = HTTPHeaders()
            if let value { headers.add(name: "X-TinyTitan-Workspace", value: value) }
            return HTTPRequestHead(
                version: .http1_1, method: .POST,
                uri: "/v1/chat/completions", headers: headers)
        }
        #expect(WorkspaceHeader.value(in: head("proj-alpha")) == "proj-alpha")
        #expect(WorkspaceHeader.value(in: head("  proj-beta  ")) == "proj-beta")
        #expect(WorkspaceHeader.value(in: head("   ")) == nil)
        #expect(WorkspaceHeader.value(in: head(nil)) == nil)
        #expect(WorkspaceHeader.value(in: nil) == nil)
    }
}
