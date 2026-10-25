import Foundation
import Metal
import TinyTitan

/// `TinyTitanBench nemotron <install> "<prompt>" [n]`: the Nemotron 3.5 Bolt
/// text graph, its tokenizer and timings in one command.
///
/// The 74B-A9B checkpoint is not a Qwen shape, so it does not go through
/// `RealForwardRunner`; it loads its `.ssdai` install directly and runs the
/// Mamba-2 / LatentMoE / attention blocks in `NemotronModel`. Prefill is
/// token-by-token, so the first number is the real TTFT and the second is
/// steady-state decode.
extension TinyTitanBench {
    static func runNemotron(
        install path: String, prompt: String, limit: Int, gpu: Bool = false
    ) throws {
        /// unchecked-invariant: written only inside the Task and read after
        /// `semaphore.wait()` orders it; no concurrent access.
        final class Box: @unchecked Sendable {
            var error: Error?
        }
        let box = Box()
        let semaphore = DispatchSemaphore(value: 0)
        Task {
            do {
                try await runNemotronAsync(
                    install: URL(fileURLWithPath: path), prompt: prompt,
                    limit: limit, gpu: gpu)
            } catch {
                box.error = error
            }
            semaphore.signal()
        }
        semaphore.wait()
        if let error = box.error { throw error }
    }

    private static func runNemotronAsync(
        install: URL, prompt: String, limit: Int, gpu: Bool
    ) async throws {
        let loaded = try NemotronInstall(directory: install)
        let backend: NemotronMatvec?
        if gpu {
            let context = try MetalContext()
            print("device: \(context.device.name)")
            backend = try NemotronMetal(install: loaded, context: context)
        } else {
            backend = nil
        }
        let tokenizer = try await GFTokenizer.load(
            forModelDirectory: install, thinkingMode: .off)
        let chat = ProcessInfo.processInfo.environment["TINYTITAN_NEMOTRON_CHAT"] == "1"
        let ids: [Int32]
        if chat {
            ids = try tokenizer.encodeToolChat(
                messages: [GFTokenizer.Message(role: .user, content: prompt)],
                tools: [])
        } else {
            ids = tokenizer.encode(prompt, addBOS: false)
        }
        guard !ids.isEmpty else {
            FileHandle.standardError.write(Data("the prompt tokenized to nothing\n".utf8))
            exit(2)
        }
        let model = try NemotronModel(
            install: loaded, maxSequence: ids.count + limit + 8, matvec: backend)

        let prefillStart = Date()
        var logits: [Float] = []
        for (index, token) in ids.enumerated() {
            logits = try model.step(token: Int(token))
            _ = index
        }
        let prefill = Date().timeIntervalSince(prefillStart)

        var produced: [Int32] = []
        let decodeStart = Date()
        for _ in 0..<limit {
            guard let best = greedy(logits) else { break }
            if loaded.geometry.eosTokens.contains(best) { break }
            produced.append(Int32(best))
            logits = try model.step(token: best)
        }
        let decode = Date().timeIntervalSince(decodeStart)

        print(
            "prompt \(ids.count) tokens: \(String(format: "%.2f", prefill))s "
                + "(\(String(format: "%.2f", Double(ids.count) / prefill)) tok/s)")
        if produced.isEmpty {
            print("decode 0 tokens")
        } else {
            print(
                "decode \(produced.count) tokens: \(String(format: "%.2f", decode))s "
                    + "(\(String(format: "%.2f", Double(produced.count) / decode)) tok/s)")
        }
        print("tokens: " + produced.map { String(Int($0)) }.joined(separator: ","))
        print("output: \(tokenizer.decode(produced, skipSpecialTokens: true))")
    }

    /// `TinyTitanBench nemotronlogits <install> <token,token,...> [--gpu]`:
    /// the logits after a token prefix, with no tokenizer in the way. This is
    /// the instrument the NumPy reference is compared against.
    static func runNemotronLogits(install path: String, tokens: String, gpu: Bool) throws {
        let loaded = try NemotronInstall(directory: URL(fileURLWithPath: path))
        let backend: NemotronMatvec?
        if gpu {
            let context = try MetalContext()
            backend = try NemotronMetal(install: loaded, context: context)
        } else {
            backend = nil
        }
        let ids = tokens.split(separator: ",").compactMap { Int($0) }
        guard !ids.isEmpty else {
            FileHandle.standardError.write(Data("no token ids\n".utf8))
            exit(2)
        }
        let model = try NemotronModel(
            install: loaded, maxSequence: ids.count + 8, matvec: backend)
        var logits: [Float] = []
        for token in ids { logits = try model.step(token: token) }
        print(logits.map { String(format: "%.6f", $0) }.joined(separator: ","))
        var best = 0
        for index in logits.indices where logits[index] > logits[best] { best = index }
        print("argmax \(best)")
    }

    /// `TinyTitanBench nemotroninfo <install>`: what the install declares and
    /// what the reader found, before a token is run.
    static func runNemotronInfo(install path: String) throws {
        let loaded = try NemotronInstall(directory: URL(fileURLWithPath: path))
        let g = loaded.geometry
        print("family: nemotron_h")
        print(
            "layers \(g.numLayers): "
                + "\(g.layerKinds.filter { $0 == .mamba }.count) mamba, "
                + "\(g.layerKinds.filter { $0 == .moe }.count) moe, "
                + "\(g.layerKinds.filter { $0 == .attention }.count) attention")
        let carried = g.layerKinds.indices
            .filter { g.layerKinds[$0] == .moe }
            .map { g.expertCounts[$0] }
        print(
            "hidden \(g.hiddenSize), vocab \(g.vocabSize), latent \(g.moeLatentSize), "
                + "experts \(carried.min() ?? 0)...\(carried.max() ?? 0) "
                + "top-\(g.topKExperts)")
        print(
            "mamba \(g.mambaHeads)x\(g.mambaHeadDim), state \(g.ssmStateSize), "
                + "groups \(g.mambaGroups), conv \(g.convKernel)")
        print("attention \(g.attentionHeads)/\(g.attentionKVHeads)x\(g.attentionHeadDim)")
        print("resident tensors: \(loaded.resident.count)")
        print("expert layers: \(loaded.experts.count)")
        print("eos: \(g.eosTokens.map(String.init).joined(separator: ","))")
    }

    private static func greedy(_ logits: [Float]) -> Int? {
        guard !logits.isEmpty else { return nil }
        var best = 0
        for index in logits.indices where logits[index] > logits[best] { best = index }
        return best
    }
}
