import Foundation
import TinyTitan
import Tokenizers

/// `TinyTitanBench lagunagpu <install> "<prompt>" [n]`: the Laguna-S-2.1
/// install through the probe graph on Metal, greedy.
///
/// This is the real 118B weights minus the serving runner: one token at a
/// time, per-layer probes with plain K/V caches, and the fetched routed
/// experts. It exists to put a number and a text on the graph before the
/// server is wired to it.
extension TinyTitanBench {
    static func runLagunaGPU(install path: String, prompt: String, limit: Int) throws {
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
                try await runLagunaGPUAsync(
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

    private static func runLagunaGPUAsync(
        install: URL, prompt: String, limit: Int
    ) async throws {
        let config = try ArchConfig.resolved(
            forFamily: .laguna, directoryURL: install)
        let context = try MetalContext()
        let model = try Model.load(
            directoryURL: install, device: context.device, expecting: config,
            streamingMode: .pread(slotCount: 8))
        guard let tokenizer = try loadK2Tokenizer(install) else {
            FileHandle.standardError.write(Data("no tokenizer in \(install.path)\n".utf8))
            exit(2)
        }
        let ids = tokenizer.encode(text: prompt, addSpecialTokens: false)
        guard !ids.isEmpty else {
            FileHandle.standardError.write(Data("the prompt tokenized to nothing\n".utf8))
            exit(2)
        }
        let graph = try LagunaGraphProbe(
            model: model, context: context,
            maxSequence: ids.count + limit + 2)

        let start = Date()
        var logits: [Float] = []
        for (position, token) in ids.enumerated() {
            logits = try await graph.forward(
                tokenID: UInt32(token),
                position: UInt32(position))
        }
        let prefill = Date().timeIntervalSince(start)

        var produced: [Int] = []
        let decodeStart = Date()
        var position = ids.count
        for _ in 0..<limit {
            var best = 0
            for index in logits.indices where logits[index] > logits[best] {
                best = index
            }
            let token = best
            if LagunaTensors.eosTokens.contains(token) { break }
            produced.append(token)
            logits = try await graph.forward(
                tokenID: UInt32(token),
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
        print(
            "tokens: "
                + produced.map { String($0) }.joined(separator: ","))
        print("output: \(tokenizer.decode(tokens: produced, skipSpecialTokens: true))")
        print(
            "raw: \(tokenizer.decode(tokens: produced, skipSpecialTokens: false))")
        if ProcessInfo.processInfo.environment["TINYTITAN_LAGUNA_LOGITS"] == "1" {
            print(
                "logits: "
                    + logits.prefix(8).map { String(format: "%.5f", $0) }
                    .joined(separator: ","))
        }
    }
}
