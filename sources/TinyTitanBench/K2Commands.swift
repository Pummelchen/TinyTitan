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
            ids = loaded.encode(text: prompt, addSpecialTokens: false)
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
