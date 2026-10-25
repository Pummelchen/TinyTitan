import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

/// The chunked dense layer against the token path: same weights, same K/V
/// cache, same positions. The toy geometry is too short for the register-
/// resident GEMM (projections fall back to GEMV), so this pins the batching
/// orchestration -- norms over rows, per-token RoPE and causal attention,
/// contiguous K/V writes, flat elementwise stages, residuals -- while the GEMM
/// itself is covered by `AffineQuantGEMMTests`.
@Suite struct K2DenseBatchTests {
    private static func halves(_ buffer: MTLBuffer, offset: Int, count: Int) -> [Float16] {
        let base = buffer.contents()
        return (0..<count).map {
            base.load(
                fromByteOffset: offset + $0 * MemoryLayout<Float16>.stride,
                as: Float16.self)
        }
    }

    private static func copy(_ values: [Float16], into buffer: MTLBuffer, offset: Int = 0) {
        values.withUnsafeBufferPointer { source in
            if let base = source.baseAddress {
                memcpy(
                    buffer.contents() + offset, base,
                    values.count * MemoryLayout<Float16>.stride)
            }
        }
    }

    private static func values(_ count: Int, salt: Int) -> [Float16] {
        (0..<count).map { index in
            Float16(Float((index * 37 + salt * 11) % 29) / 14 - 1)
        }
    }

    @Test func chunkMatchesTheTokenPathAcrossTwoChunks() async throws {
        let directory = try K2ToySynthetic.write(sparse: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let config = try ArchConfig.resolved(
            forFamily: .k2Horizon, directoryURL: directory)
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: directory, device: ctx.device, expecting: config,
            integrityPolicy: .fullSha256)
        let d = config.hiddenSize
        let chunk = 4
        let batch = try K2DenseBatch(
            model: model, context: ctx, layer: 0,
            maxSequence: 8, maxTokens: chunk)
        let token = try K2DenseLayer(
            model: model, context: ctx, layer: 0, maxSequence: 8)

        guard
            let batchIn = ctx.device.makeBuffer(
                length: chunk * d * 2, options: .storageModeShared),
            let batchOut = ctx.device.makeBuffer(
                length: chunk * d * 2, options: .storageModeShared),
            let tokenIn = ctx.device.makeBuffer(
                length: d * 2, options: .storageModeShared),
            let tokenOut = ctx.device.makeBuffer(
                length: d * 2, options: .storageModeShared)
        else {
            Issue.record("could not allocate the layer buffers")
            return
        }

        for base in stride(from: 0, to: 8, by: chunk) {
            let input = Self.values(chunk * d, salt: base)
            Self.copy(input, into: batchIn)

            let batchCB = try #require(ctx.queue.makeCommandBuffer())
            try batch.encode(
                commandBuffer: batchCB, hidden: batchIn, out: batchOut,
                basePosition: base, tokenCount: chunk)
            batchCB.commit()
            await batchCB.completed()
            #expect(batchCB.status == .completed)

            var expected: [Float16] = []
            for index in 0..<chunk {
                let row = Array(
                    input[index * d..<(index + 1) * d])
                Self.copy(row, into: tokenIn)
                let cb = try #require(ctx.queue.makeCommandBuffer())
                try token.encode(
                    commandBuffer: cb, hidden: tokenIn, out: tokenOut,
                    position: UInt32(base + index))
                cb.commit()
                await cb.completed()
                #expect(cb.status == .completed)
                expected += Self.halves(tokenOut, offset: 0, count: d)
            }

            let actual = Self.halves(
                batchOut, offset: 0, count: chunk * d
            ).map { Float($0) }
            let reference = expected.map { Float($0) }
            let rel = RelError.compute(actual: actual, reference: reference)
            #expect(rel < Tolerance.fp16Reduction, "relErr=\(rel) base=\(base)")
        }
    }
}
