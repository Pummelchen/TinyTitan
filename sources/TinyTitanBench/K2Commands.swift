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
        let model = try Model.load(
            directoryURL: install, device: context.device, expecting: config,
            streamingMode: .pread(slotCount: 8))
        let tokenizer = try await GFTokenizer.load(
            forModelDirectory: install, thinkingMode: .on,
            reasoningEffort: .high)
        let ids = tokenizer.encode(prompt, addBOS: false)
        guard !ids.isEmpty else {
            FileHandle.standardError.write(Data("the prompt tokenized to nothing\n".utf8))
            exit(2)
        }
        let graph = try K2GraphProbe(
            model: model, context: context, maxSequence: ids.count + limit + 1)

        let start = Date()
        var logits: [Float] = []
        for (position, token) in ids.enumerated() {
            logits = try await graph.forward(
                tokenID: UInt32(bitPattern: token),
                position: UInt32(position))
        }
        let prefill = Date().timeIntervalSince(start)

        var produced: [Int32] = []
        let decodeStart = Date()
        var position = ids.count
        for _ in 0..<limit {
            var best = 0
            for index in logits.indices where logits[index] > logits[best] {
                best = index
            }
            let token = Int32(best)
            if token == tokenizer.eosID { break }
            produced.append(token)
            logits = try await graph.forward(
                tokenID: UInt32(bitPattern: token),
                position: UInt32(position))
            position += 1
        }
        let decode = Date().timeIntervalSince(decodeStart)
        print(
            "prompt \(ids.count) tokens: \(String(format: "%.2f", prefill))s "
                + "(\(String(format: "%.2f", Double(ids.count) / prefill)) tok/s)")
        print(
            "decode \(produced.count) tokens: \(String(format: "%.2f", decode))s "
                + "(\(String(format: "%.2f", Double(produced.count) / max(decode, 0.001))) tok/s)")
        print("output: \(tokenizer.decode(produced, skipSpecialTokens: true))")
    }
}
