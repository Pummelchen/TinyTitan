import Foundation
import Metal

/// A K2 decode step over a mixed model: embedding, the dense and sparse layers
/// in order, then the final norm and head.
///
/// The sparse probe drives its own command buffers (its router ids have to
/// cross to the CPU to address the expert blobs), so this graph sequences each
/// layer one at a time and carries the hidden state as halfs between them. That is the probe
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
        // The hidden state stays on the GPU for the whole token. A dense layer
        // writes the other ping-pong buffer; a sparse layer writes its own
        // output, which the next layer reads. Only the router ids and the final
        // logits come back to the CPU.
        guard let embed = queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        try head.encodeEmbedding(commandBuffer: embed, tokenID: tokenID, out: hiddenA)
        embed.commit()
        var current = hiddenA
        for layer in 0..<config.numLayers {
            if let dense = dense[layer] {
                let out = current === hiddenA ? hiddenB : hiddenA
                guard let cb = queue.makeCommandBuffer() else {
                    throw MetalError.commandEncoderFailed
                }
                try dense.encode(
                    commandBuffer: cb, hidden: current, out: out, position: position)
                cb.commit()
                current = out
            } else if let sparse = sparse[layer] {
                try await sparse.prefetch(
                    queue: queue, input: current, position: Int(position))
                try await sparse.submitExperts(queue: queue)
                current = sparse.output
            } else {
                throw ModelError.internalInconsistency(
                    detail: "layer \(layer) is neither dense nor sparse")
            }
        }
        guard let headCB = queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        try head.encodeLogits(commandBuffer: headCB, hidden: current)
        headCB.commit()
        await headCB.completed()
        // The expert phases are submitted without waits; surface any failure
        // they reported before the token's logits are trusted.
        for layer in sparse.values { try layer.checkPendingError() }
        if ProcessInfo.processInfo.environment["TINYTITAN_K2_TIMING"] == "1" {
            let fetch = sparse.values.reduce(0.0) { $0 + $1.fetchSeconds }
            let encode = sparse.values.reduce(0.0) { $0 + $1.encodeSeconds }
            let busy = sparse.values.reduce(0.0) { $0 + $1.gpuBusySeconds }
            let prefetch = sparse.values.reduce(0.0) { $0 + $1.prefetchBusySeconds }
            let experts = sparse.values.reduce(0.0) { $0 + $1.expertBusySeconds }
            let total = Date().timeIntervalSince(started)
            let rest = total - fetch - encode - busy
            FileHandle.standardError.write(
                Data(
                    String(
                        format:
                            "k2 timing: total %.3fs fetch %.3fs busy %.3fs (pre %.3fs experts %.3fs) encode %.3fs other %.3fs\n",
                        total, fetch, busy, prefetch, experts, encode, rest
                    ).utf8))
        }
        return head.readLogits()
    }

}
