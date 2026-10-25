import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

/// The mixed graph on the sparse toy: embedding, three dense layers, the sparse
/// layer, final norm and head. Numeric layer correctness is pinned by the
/// per-layer tests; this checks the graph runs the layers in order, at the right
/// positions, with each cache intact.
@Suite struct K2GraphProbeTests {
    private static func halves(_ buffer: MTLBuffer, count: Int) -> [Float16] {
        let base = buffer.contents()
        return (0..<count).map {
            base.load(
                fromByteOffset: $0 * MemoryLayout<Float16>.stride,
                as: Float16.self)
        }
    }

    private static func copy(_ values: [Float16], into buffer: MTLBuffer) {
        values.withUnsafeBufferPointer { source in
            if let base = source.baseAddress {
                memcpy(buffer.contents(), base, values.count * 2)
            }
        }
    }

    @Test func graphMatchesTheManualLayerSequence() async throws {
        let directory = try K2ToySynthetic.write(sparse: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let config = try ArchConfig.resolved(
            forFamily: .k2Horizon, directoryURL: directory)
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: directory, device: ctx.device, expecting: config,
            integrityPolicy: .fullSha256)
        let graph = try K2GraphProbe(model: model, context: ctx, maxSequence: 2)

        let head = try K2Head(model: model, context: ctx)
        let dense = try Dictionary(
            uniqueKeysWithValues: config.denseLayerIndices.map {
                (
                    $0,
                    try K2DenseLayer(
                        model: model, context: ctx, layer: $0, maxSequence: 2)
                )
            })
        let sparseLayer = config.numLayers - 1
        let sparse = try K2SparseLayerProbe(
            model: model, context: ctx, layer: sparseLayer, maxSequence: 2)
        let hiddenA = try #require(
            ctx.device.makeBuffer(
                length: config.hiddenSize * 2, options: .storageModeShared))
        let hiddenB = try #require(
            ctx.device.makeBuffer(
                length: config.hiddenSize * 2, options: .storageModeShared))

        func manual(tokenID: UInt32, position: UInt32) async throws -> [Float] {
            guard let cb = ctx.queue.makeCommandBuffer() else {
                throw MetalError.commandEncoderFailed
            }
            try head.encodeEmbedding(commandBuffer: cb, tokenID: tokenID, out: hiddenA)
            cb.commit()
            await cb.completed()
            var hidden = Self.halves(hiddenA, count: config.hiddenSize)
            for layer in 0..<config.numLayers {
                if let layerKernel = dense[layer] {
                    Self.copy(hidden, into: hiddenA)
                    guard let step = ctx.queue.makeCommandBuffer() else {
                        throw MetalError.commandEncoderFailed
                    }
                    try layerKernel.encode(
                        commandBuffer: step, hidden: hiddenA, out: hiddenB,
                        position: position)
                    step.commit()
                    await step.completed()
                    hidden = Self.halves(hiddenB, count: config.hiddenSize)
                } else {
                    let out = try await sparse.run(
                        queue: ctx.queue, hidden: hidden, position: Int(position))
                    hidden = out.map { Float16($0) }
                }
            }
            Self.copy(hidden, into: hiddenA)
            guard let headCB = ctx.queue.makeCommandBuffer() else {
                throw MetalError.commandEncoderFailed
            }
            try head.encodeLogits(commandBuffer: headCB, hidden: hiddenA)
            headCB.commit()
            await headCB.completed()
            return head.readLogits()
        }

        let graph0 = try await graph.forward(tokenID: 7, position: 0)
        let manual0 = try await manual(tokenID: 7, position: 0)
        #expect(RelError.compute(actual: graph0, reference: manual0) < Tolerance.fp16Reduction)

        let graph1 = try await graph.forward(tokenID: 8, position: 1)
        let manual1 = try await manual(tokenID: 8, position: 1)
        #expect(RelError.compute(actual: graph1, reference: manual1) < Tolerance.fp16Reduction)
    }
}
