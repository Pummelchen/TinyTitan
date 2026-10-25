import Foundation
import TinyTitan
import Tokenizers

/// `TinyTitanBench k2 <snapshot> "<prompt>" [n]`: the K2-Horizon CPU model,
/// tokenizer and timings in one command.
///
/// The model is a plain decode loop (no prompt cache, no batching yet), so the
/// two numbers it prints are TTFT for the prompt and decode tokens per second.
extension TinyTitanBench {
    static func runK2(snapshot path: String, prompt: String, limit: Int) throws {
        let directory = URL(fileURLWithPath: path)
        let snapshot = try K2Snapshot(directory: directory)
        let model = try K2Model(snapshot: snapshot)
        // A comma-separated list is taken as token ids, which is how the
        // tiny fixture (no tokenizer) is exercised.
        let numeric = prompt.split(separator: ",").allSatisfy { Int($0) != nil }
        let tokenizer: Tokenizer?
        let ids: [Int]
        if numeric {
            tokenizer = nil
            ids = prompt.split(separator: ",").compactMap { Int($0) }
        } else {
            guard let loaded = try loadK2Tokenizer(directory) else {
                FileHandle.standardError.write(Data("no tokenizer in \(path)\n".utf8))
                exit(2)
            }
            tokenizer = loaded
            if ProcessInfo.processInfo.environment["TINYTITAN_K2_CHAT"] == "1" {
                // lint:allow-unbounded-read the checkpoint's own template: 52 KB,
                // written by the converter, and read once per process.
                let template = try String(
                    contentsOf: directory.appendingPathComponent("tokenizer/chat_template.jinja"),
                    encoding: .utf8)
                let messages: [Message] = [["role": "user", "content": prompt]]
                ids = try loaded.applyChatTemplate(messages: messages, chatTemplate: template)
            } else {
                ids = loaded.encode(text: prompt, addSpecialTokens: false)
            }
        }
        guard !ids.isEmpty else {
            FileHandle.standardError.write(Data("the prompt tokenized to nothing\n".utf8))
            exit(2)
        }

        let prefillStart = Date()
        var logits: [Float] = []
        for (index, token) in ids.enumerated() {
            logits = try model.step(token: token, needsLogits: index == ids.count - 1)
        }
        let prefillSeconds = Date().timeIntervalSince(prefillStart)

        var produced: [Int] = []
        let decodeStart = Date()
        for _ in 0..<limit {
            var best = 0
            for index in logits.indices where logits[index] > logits[best] { best = index }
            if snapshot.configuration.eosTokens.contains(best) { break }
            produced.append(best)
            logits = try model.step(token: best)
        }
        let decodeSeconds = Date().timeIntervalSince(decodeStart)
        let text: String
        if let tokenizer {
            text = tokenizer.decode(tokens: produced, skipSpecialTokens: true)
        } else {
            text = produced.map { String($0) }.joined(separator: ",")
        }
        print(
            "prompt \(ids.count) tokens: \(String(format: "%.2f", prefillSeconds))s "
                + "(\(String(format: "%.1f", Double(ids.count) / prefillSeconds)) tok/s)")
        if decodeSeconds > 0, !produced.isEmpty {
            print(
                "decode \(produced.count) tokens: \(String(format: "%.2f", decodeSeconds))s "
                    + "(\(String(format: "%.2f", Double(produced.count) / decodeSeconds)) tok/s)")
        } else {
            print("decode 0 tokens")
        }
        print("output: \(text)")
        if ProcessInfo.processInfo.environment["TINYTITAN_K2_LOGITS"] == "1" {
            print("logits: " + logits.map { String(format: "%.6f", $0) }.joined(separator: ","))
        }
    }

    /// The generic swift-transformers tokenizer, for a checkpoint whose special
    /// tokens are not Qwen's.
    static func loadK2Tokenizer(_ directory: URL) throws -> Tokenizer? {
        guard let folder = GFTokenizer.resolvedTokenizerFolder(forModelDirectory: directory)
        else { return nil }
        // unchecked-invariant: written only inside the Task below and read only
        // after `semaphore.wait()` returns, which the signal orders after the
        // last write. There is no concurrent access.
        /// unchecked-invariant: written only inside the Task below and read
        /// only after `semaphore.wait()` returns, which orders the read after
        /// the write; no concurrent access.
        final class Box: @unchecked Sendable {
            var value: Tokenizer?
            var error: Error?
        }
        let box = Box()
        let semaphore = DispatchSemaphore(value: 0)
        Task {
            do {
                box.value = try await AutoTokenizer.from(modelFolder: folder)
            } catch {
                box.error = error
            }
            semaphore.signal()
        }
        semaphore.wait()
        if let error = box.error { throw error }
        return box.value
    }
}

/// `TinyTitanBench k2gpu <install> "<prompt>" [n]`: the K2 install through the
/// probe graph on Metal, greedy.
///
/// This is the real 36B path minus the runner: one token at a time, the
/// per-layer probes with their plain K/V caches, and the fetched routed
/// experts. It exists to put a number and a text on the graph before it is
/// wired to the server.
extension TinyTitanBench {
    static func runK2GPU(install path: String, prompt: String, limit: Int) throws {
        /// unchecked-invariant: written only inside the Task below and read
        /// only after `semaphore.wait()` returns, which orders the read after
        /// the write; no concurrent access.
        final class Box: @unchecked Sendable {
            var error: Error?
        }
        let box = Box()
        let semaphore = DispatchSemaphore(value: 0)
        Task {
            do {
                try await runK2GPUAsync(
                    install: URL(fileURLWithPath: path), prompt: prompt,
                    limit: limit)
            } catch {
                box.error = error
            }
            semaphore.signal()
        }
        semaphore.wait()
        if let error = box.error { throw error }
    }

    private static func runK2GPUAsync(
        install: URL, prompt: String, limit: Int
    ) async throws {
        let config = try ArchConfig.resolved(
            forFamily: .k2Horizon, directoryURL: install)
        let context = try MetalContext()
        // Expert-cache slots per layer. Eight covers exactly one top-8 set, so
        // a changing selection misses; the measured default trades RAM for
        // hits (4-bit experts are 3.3 MB, 8-bit 6.6 MB per layer-slot).
        let identity = try ManifestReader.peekIdentity(directoryURL: install)
        let defaultSlots = identity.weightBits == 4 ? 64 : 16
        let slots =
            Int(ProcessInfo.processInfo.environment["TINYTITAN_K2_SLOTS"] ?? "")
            ?? defaultSlots
        // Full SHA-256 over the payload is ~7.6 s of the first token's 8 s:
        // the per-layer opens re-hash 15 GB before any expert is read. The
        // receipt the installer writes is a complete alternative -- it is
        // bound to the manifest hash, the file list and the directory path --
        // so the fast path validates it first and falls back to the full hash
        // (loudly) when it is missing, moved or stale. An invalid receipt
        // still ends in the full verification, never in an unchecked load.
        let forceFullVerify =
            ProcessInfo.processInfo.environment["TINYTITAN_K2_FULL_VERIFY"] == "1"
        let model: Model
        let integrityNote: String
        if forceFullVerify {
            model = try Model.load(
                directoryURL: install, device: context.device, expecting: config,
                streamingMode: .pread(slotCount: slots))
            integrityNote = "integrity: full SHA-256 (TINYTITAN_K2_FULL_VERIFY=1)"
        } else {
            do {
                model = try Model.load(
                    directoryURL: install, device: context.device, expecting: config,
                    streamingMode: .pread(slotCount: slots),
                    integrityPolicy: .sizeCheckTrustedReceipt)
                integrityNote = "integrity: trusted receipt"
            } catch let error as ModelError {
                guard case .trustedReceiptInvalid(let detail) = error else { throw error }
                model = try Model.load(
                    directoryURL: install, device: context.device, expecting: config,
                    streamingMode: .pread(slotCount: slots))
                integrityNote = "integrity: full SHA-256 (receipt rejected: \(detail))"
            }
        }
        let effort = ModelReasoningEffort.resolved() ?? .high
        let tokenizer = try await GFTokenizer.load(
            forModelDirectory: install, thinkingMode: .on,
            reasoningEffort: effort)
        let chat = ProcessInfo.processInfo.environment["TINYTITAN_K2_CHAT"] == "1"
        let ids: [Int32]
        let decoderCheck =
            ProcessInfo.processInfo.environment["TINYTITAN_K2_DECODER"] == "1"
        if chat {
            // The decoder check renders the prompt with a tool so the model
            // sees the same surface a server request would.
            ids = try tokenizer.encodeToolChat(
                messages: [GFTokenizer.Message(role: .user, content: prompt)],
                tools: decoderCheck ? PromptTools.sample : [])
        } else {
            ids = tokenizer.encode(prompt, addBOS: false)
        }
        guard !ids.isEmpty else {
            FileHandle.standardError.write(Data("the prompt tokenized to nothing\n".utf8))
            exit(2)
        }
        let graph = try K2GraphProbe(
            model: model, context: context, maxSequence: ids.count + limit + 1)

        // Repeat the whole prompt+decode loop in-process so a measurement can
        // report a median instead of a single sample. Run 1 pays the cold
        // expert cache; later runs are the steady state a server sees, and the
        // machine may be busy with unrelated work, which single runs cannot
        // distinguish from a regression.
        let repeatCount = max(
            1,
            Int(ProcessInfo.processInfo.environment["TINYTITAN_K2_REPEAT"] ?? "") ?? 1)
        let stops: Set<Int32> = [tokenizer.eosID, tokenizer.endOfTurnID]
        var produced: [Int32] = []
        var prefillTimes: [Double] = []
        var decodeTimes: [Double] = []
        for run in 0..<repeatCount {
            let start = Date()
            var logits: [Float] = []
            for (position, token) in ids.enumerated() {
                logits = try await graph.forward(
                    tokenID: UInt32(bitPattern: token),
                    position: UInt32(position))
            }
            prefillTimes.append(Date().timeIntervalSince(start))

            var runTokens: [Int32] = []
            let decodeStart = Date()
            var position = ids.count
            for _ in 0..<limit {
                var best = 0
                for index in logits.indices where logits[index] > logits[best] {
                    best = index
                }
                let token = Int32(best)
                if stops.contains(token) { break }
                runTokens.append(token)
                logits = try await graph.forward(
                    tokenID: UInt32(bitPattern: token),
                    position: UInt32(position))
                position += 1
            }
            decodeTimes.append(Date().timeIntervalSince(decodeStart))
            if repeatCount > 1 {
                let elapsed = decodeTimes[run]
                print(
                    "run \(run): prompt \(String(format: "%.2f", prefillTimes[run]))s, "
                        + "decode \(String(format: "%.2f", elapsed))s "
                        + "(\(String(format: "%.2f", Double(runTokens.count) / max(elapsed, 0.001))) tok/s)"
                )
            }
            if run == 0 {
                produced = runTokens
            } else if runTokens != produced {
                print(
                    "warning: run \(run) produced different tokens: "
                        + runTokens.prefix(8).map(String.init).joined(separator: ","))
            }
        }
        let prefill = prefillTimes[0]
        let decode = decodeTimes[0]
        print(integrityNote)
        print(
            "prompt \(ids.count) tokens: \(String(format: "%.2f", prefill))s "
                + "(\(String(format: "%.2f", Double(ids.count) / prefill)) tok/s)")
        print(
            "decode \(produced.count) tokens: \(String(format: "%.2f", decode))s "
                + "(\(String(format: "%.2f", Double(produced.count) / max(decode, 0.001))) tok/s)")
        if repeatCount > 1 {
            // Run 0 pays the cold expert cache; the steady-state figure is the
            // later runs. With only two runs there is no steady state to
            // separate, so the median covers all of them.
            let steady = repeatCount > 2 ? Array(decodeTimes.dropFirst()) : decodeTimes
            let steadyPrefill =
                repeatCount > 2 ? Array(prefillTimes.dropFirst()) : prefillTimes
            func median(_ values: [Double]) -> Double {
                let sorted = values.sorted()
                guard !sorted.isEmpty else { return 0 }
                if sorted.count % 2 == 1 { return sorted[sorted.count / 2] }
                return (sorted[sorted.count / 2 - 1] + sorted[sorted.count / 2]) / 2
            }
            let steadyMedian = median(steady)
            let best = decodeTimes.min() ?? steadyMedian
            let window = repeatCount > 2 ? "runs 1..\(repeatCount - 1)" : "all runs"
            print(
                "steady state (\(window)): prompt \(String(format: "%.2f", median(steadyPrefill)))s, "
                    + "decode \(String(format: "%.2f", steadyMedian))s "
                    + "(\(String(format: "%.2f", Double(produced.count) / max(steadyMedian, 0.001))) tok/s), "
                    + "best \(String(format: "%.2f", best))s")
        }
        print(
            "tokens: "
                + produced.map { String(Int($0)) }.joined(separator: ","))
        print("output: \(tokenizer.decode(produced, skipSpecialTokens: true))")
        print(
            "raw: \(tokenizer.decode(produced, skipSpecialTokens: false))")
        if ProcessInfo.processInfo.environment["TINYTITAN_K2_DECODER_SELFTEST"] == "1" {
            try k2DecoderSelfTest(tokenizer: tokenizer)
        }
        if decoderCheck {
            let allowed = chat ? Set(PromptTools.sample.map(\.name)) : nil
            print(
                "decoder: "
                    + (try k2DecoderReport(
                        tokenizer: tokenizer, promptIDs: ids,
                        tokens: produced, allowedTools: allowed)))
        }
    }

    /// A sample tool set for the chat path, so a tool-templated prompt can be
    /// rendered and parsed without a server request.
    private enum PromptTools {
        static let sample: [GFTokenizer.FunctionDefinition] = [
            GFTokenizer.FunctionDefinition(
                name: "get_weather",
                description: "Current weather for a city",
                parameters: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "city": .object(["type": .string("string")])
                    ]),
                    "required": .array([.string("city")]),
                ]))
        ]
    }

    /// Run a token stream through the same structured decoder the server uses
    /// and render its events, so the K2 think/tool surfaces are checked on the
    /// real tokenizer instead of only in unit tests with a fixture.
    private static func k2DecoderReport(
        tokenizer: GFTokenizer,
        promptIDs: [Int32],
        tokens: [Int32],
        allowedTools: Set<String>?
    ) throws -> String {
        let decoder = StructuredAssistantDecoder.forGeneration(
            tokenizer: tokenizer, promptIDs: promptIDs,
            allowedTools: allowedTools)
        var detokenizer = GFDetokenizer(tokenizer: tokenizer)
        var events: [StructuredAssistantEvent] = []
        for id in tokens {
            events += try decoder.consume(
                tokenID: id, delta: detokenizer.push(id))
        }
        events += try decoder.consumeTail(detokenizer.flush())
        try decoder.finish()
        var reasoning = ""
        var content = ""
        var calls: [ParsedToolCall] = []
        for event in events {
            switch event {
            case .reasoning(let text): reasoning += text
            case .content(let text): content += text
            case .toolCall(let call): calls.append(call)
            }
        }
        var line = "reasoning=\(reasoning.debugDescription) content=\(content.debugDescription)"
        if !calls.isEmpty {
            line += " calls=" + calls.map { "\($0.name)(\($0.arguments))" }.joined(separator: ",")
        }
        return line
    }

    /// Feed two canned streams through the decoder with the real tokenizer:
    /// one closes a thought and answers, one emits an xml tool call. This is
    /// install-dependent, so it lives behind TINYTITAN_K2_DECODER_SELFTEST.
    private static func k2DecoderSelfTest(tokenizer: GFTokenizer) throws {
        let effort = ModelReasoningEffort.resolved() ?? .high
        var tag = "think"
        switch effort {
        case .medium: tag = "think_fast"
        case .low: tag = "think_faster"
        case .high, .xhigh: tag = "think"
        }
        let promptIDs = tokenizer.encode(
            "<|ifm|im_start|>assistant\n<ifm|\(tag)>\n", addBOS: false)

        let answer = "Simple.\n</ifm|\(tag)>\nThe capital is Paris."
        let tool =
            "<ifm|tool_call>get_weather\n"
            + "<ifm|arg_key>city</ifm|arg_key>\n"
            + "<ifm|arg_value>Paris</ifm|arg_value>\n"
            + "</ifm|tool_call>"
        for (label, text) in [("answer", answer), ("tool", tool)] {
            let tokens = tokenizer.encode(text, addBOS: false)
            let report = try k2DecoderReport(
                tokenizer: tokenizer, promptIDs: promptIDs, tokens: tokens,
                allowedTools: label == "tool" ? ["get_weather"] : nil)
            print("selftest \(label): \(report)")
        }
    }
}
