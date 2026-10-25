import Foundation
import Metal

/// A Laguna-S-2.1 decode step: embedding, all 48 layers in order, then the
/// final norm and the untied head.
///
/// The sparse layers drive their own command buffers because the router ids
/// have to cross to the CPU between phases, so this graph sequences one layer
/// at a time and carries the hidden state as halfs between them. That is the
/// probe form of the serving loop, not the batched one.
package final class LagunaGraphProbe {
    private let config: ArchConfig
    private let head: K2Head
    private let layers: [LagunaLayer]
    private let context: MetalContext
    private let queue: MTLCommandQueue
    private let hiddenA: MTLBuffer
    private let hiddenB: MTLBuffer

    package init(model: Model, context: MetalContext, maxSequence: Int) throws {
        self.config = model.config
        self.context = context
        self.queue = context.queue
        self.head = try K2Head(model: model, context: context)
        self.layers = try (0..<model.config.numLayers).map { layer in
            try LagunaLayer(
                model: model, context: context, layer: layer,
                maxSequence: maxSequence)
        }
        let hidden = model.config.hiddenSize
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
        self.hiddenA = try buffer("laguna.graph.hiddenA")
        self.hiddenB = try buffer("laguna.graph.hiddenB")
    }

    package func forward(tokenID: UInt32, position: UInt32) async throws -> [Float] {
        guard let embed = queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        try head.encodeEmbedding(
            commandBuffer: embed, tokenID: tokenID, out: hiddenA)
        embed.commit()
        await embed.completed()

        var input = hiddenA
        var output = hiddenB
        for layer in layers {
            try await layer.run(
                queue: queue, hidden: input, out: output, position: Int(position))
            swap(&input, &output)
        }
        guard let final = queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        try head.encodeLogits(commandBuffer: final, hidden: input)
        final.commit()
        await final.completed()
        return head.readLogits()
    }
}
