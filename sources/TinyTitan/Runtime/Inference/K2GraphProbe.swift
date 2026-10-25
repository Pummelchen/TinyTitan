import Foundation
import Metal

/// A K2 decode step over a mixed model: embedding, the dense and sparse layers
/// in order, then the final norm and head.
///
/// The sparse probe drives its own command buffers (its router ids have to
/// cross to the CPU between phases), so this graph sequences each layer one at
/// a time and carries the hidden state as halfs between them. That is the probe
/// form of the loop, not the batched serving graph.
package final class K2GraphProbe {
    private let config: ArchConfig
    private let head: K2Head
    private let dense: [Int: K2DenseLayer]
    private let sparse: [Int: K2SparseLayerProbe]
    private let context: MetalContext
    private let queue: MTLCommandQueue
    private let hiddenA: MTLBuffer
    private let hiddenB: MTLBuffer

    package init(
        model: Model, context: MetalContext, maxSequence: Int
    ) throws {
        let config = model.config
        self.config = config
        self.context = context
        self.queue = context.queue
        self.head = try K2Head(model: model, context: context)
        var dense: [Int: K2DenseLayer] = [:]
        var sparse: [Int: K2SparseLayerProbe] = [:]
        for layer in 0..<config.numLayers {
            if config.denseLayerIndices.contains(layer) {
                dense[layer] = try K2DenseLayer(
                    model: model, context: context, layer: layer,
                    maxSequence: maxSequence)
            } else {
                sparse[layer] = try K2SparseLayerProbe(
                    model: model, context: context, layer: layer,
                    maxSequence: maxSequence)
            }
        }
        self.dense = dense
        self.sparse = sparse
        let hidden = config.hiddenSize
        func buffer(_ label: String) throws -> MTLBuffer {
            guard
                let buffer = context.device.makeBuffer(
                    length: hidden * MemoryLayout<UInt16>.stride,
                    options: .storageModeShared)
            else {
                throw MetalError.bufferAllocationFailed(label)
            }
            return buffer
        }
        self.hiddenA = try buffer("k2.graph.hiddenA")
        self.hiddenB = try buffer("k2.graph.hiddenB")
    }

    package func forward(tokenID: UInt32, position: UInt32) async throws -> [Float] {
        let started = Date()
        if ProcessInfo.processInfo.environment["TINYTITAN_K2_TIMING"] == "1" {
            for layer in sparse.values { layer.resetTimings() }
        }
        var hidden = try await embed(tokenID: tokenID)
        for layer in 0..<config.numLayers {
            if let dense = dense[layer] {
                hidden = try await runDense(
                    dense, hidden: hidden, position: position)
            } else if let sparse = sparse[layer] {
                hidden = try await sparse.run(
                    queue: queue, hidden: hidden, position: Int(position)
                ).map { Float16($0) }
            } else {
                throw ModelError.internalInconsistency(
                    detail: "layer \(layer) is neither dense nor sparse")
            }
        }
        let result = try await logits(hidden: hidden)
        if ProcessInfo.processInfo.environment["TINYTITAN_K2_TIMING"] == "1" {
            let fetch = sparse.values.reduce(0.0) { $0 + $1.fetchSeconds }
            let encode = sparse.values.reduce(0.0) { $0 + $1.encodeSeconds }
            let gpu = sparse.values.reduce(0.0) { $0 + $1.gpuSeconds }
            let busy = sparse.values.reduce(0.0) { $0 + $1.gpuBusySeconds }
            let entry = sparse.values.reduce(0.0) { $0 + $1.entryBusySeconds }
            let attention = sparse.values.reduce(0.0) { $0 + $1.attentionBusySeconds }
            let experts = sparse.values.reduce(0.0) { $0 + $1.expertBusySeconds }
            let total = Date().timeIntervalSince(started)
            let rest = total - fetch - encode - gpu
            FileHandle.standardError.write(
                Data(
                    String(
                        format:
                            "k2 timing: total %.3fs fetch %.3fs gpu %.3fs busy %.3fs (entry %.3fs attn %.3fs experts %.3fs) encode %.3fs other %.3fs\n",
                        total, fetch, gpu, busy, entry, attention, experts, encode, rest
                    ).utf8))
        }
        return result
    }

    private func embed(tokenID: UInt32) async throws -> [Float16] {
        guard let cb = queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        try head.encodeEmbedding(
            commandBuffer: cb, tokenID: tokenID, out: hiddenA)
        cb.commit()
        await cb.completed()
        return halves(from: hiddenA)
    }

    private func runDense(
        _ layer: K2DenseLayer, hidden: [Float16], position: UInt32
    ) async throws -> [Float16] {
        guard let cb = queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        copy(hidden, into: hiddenA)
        try layer.encode(
            commandBuffer: cb, hidden: hiddenA, out: hiddenB,
            position: position)
        cb.commit()
        await cb.completed()
        return halves(from: hiddenB)
    }

    private func logits(hidden: [Float16]) async throws -> [Float] {
        guard let cb = queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        copy(hidden, into: hiddenA)
        try head.encodeLogits(commandBuffer: cb, hidden: hiddenA)
        cb.commit()
        await cb.completed()
        return head.readLogits()
    }

    private func copy(_ values: [Float16], into buffer: MTLBuffer) {
        values.withUnsafeBufferPointer { source in
            if let base = source.baseAddress {
                memcpy(
                    buffer.contents(), base,
                    values.count * MemoryLayout<Float16>.stride)
            }
        }
    }

    private func halves(from buffer: MTLBuffer) -> [Float16] {
        let base = buffer.contents()
        return (0..<config.hiddenSize).map {
            base.load(
                fromByteOffset: $0 * MemoryLayout<Float16>.stride,
                as: Float16.self)
        }
    }
}
