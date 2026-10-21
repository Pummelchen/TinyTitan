import Foundation
import Metal
import Testing

@testable import TinyTitan

/// T5: the batched QKV/o_proj stage must produce each row's attention output
/// through that row's own KV slot, matching the per-row gated attention path.
@Suite struct BatchAttentionTests {
    private let layer = 3
    private let maxContext = 64

    private func makeRunner(
        _ ctx: MetalContext
    ) throws -> (URL, RealForwardRunner) {
        let dir = try QwenToySynthetic.write(weightBits: 4)
        let model = try Model.load(
            directoryURL: dir, device: ctx.device, expecting: .qwenToy())
        let runtime = try RuntimeConfiguration(
            forceLogitsHead: true, kvCachePrecision: .int8)
        let runner = try RealForwardRunner(
            model: model, context: ctx, maxContext: maxContext,
            slots: 2, runtimeConfiguration: runtime)
        return (dir, runner)
    }

    private func upload(
        _ ctx: MetalContext, values: [Float16], to destination: MTLBuffer
    ) throws {
        let bytes = values.count * 2
        let source = try #require(
            ctx.device.makeBuffer(length: bytes, options: .storageModeShared))
        values.withUnsafeBytes { raw in
            if let base = raw.baseAddress { memcpy(source.contents(), base, bytes) }
        }
        guard let cb = ctx.queue.makeCommandBuffer(),
            let blit = cb.makeBlitCommandEncoder()
        else { throw MetalError.commandEncoderFailed }
        blit.copy(
            from: source, sourceOffset: 0, to: destination,
            destinationOffset: 0, size: bytes)
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }

    private func download(
        _ ctx: MetalContext, from source: MTLBuffer, bytes: Int
    ) throws -> [Float16] {
        let destination = try #require(
            ctx.device.makeBuffer(length: bytes, options: .storageModeShared))
        guard let cb = ctx.queue.makeCommandBuffer(),
            let blit = cb.makeBlitCommandEncoder()
        else { throw MetalError.commandEncoderFailed }
        blit.copy(
            from: source, sourceOffset: 0, to: destination,
            destinationOffset: 0, size: bytes)
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        let p = destination.contents().bindMemory(to: Float16.self, capacity: bytes / 2)
        return (0..<(bytes / 2)).map { p[$0] }
    }

    @Test func batchMatchesPerRowGatedAttention() throws {
        let ctx = try MetalContext()
        let (dir, runner) = try makeRunner(ctx)
        defer { try? FileManager.default.removeItem(at: dir) }
        let d = runner.cfg.hiddenSize
        let scratch = try runner.ensureBatchedDecodeScratch(rows: 2)
        var hidden = [Float16](repeating: 0, count: 2 * d)
        for row in 0..<2 {
            for i in 0..<d {
                hidden[row * d + i] = Float16(
                    sin(Float(row * 1000 + i) * 0.017) * 0.3)
            }
        }
        try upload(ctx, values: hidden, to: scratch.hidden)
        let rows = [(position: 0, slot: 0), (position: 0, slot: 1)]
        guard var cb = ctx.queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        try runner.encodeBatchGatedAttention(
            layer: layer, rows: rows, scratch: scratch, commandBuffer: &cb)
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.error == nil)
        let batchOut = try download(
            ctx, from: scratch.o, bytes: 2 * d * 2)

        // Reference: the same two rows through the normal per-row path.
        runner.reset()
        var reference: [[Float16]] = []
        for row in 0..<2 {
            let staging = try #require(
                ctx.device.makeBuffer(
                    length: d * 2, options: .storageModeShared))
            hidden.withUnsafeBytes { raw in
                if let base = raw.baseAddress {
                    memcpy(
                        staging.contents(),
                        base.advanced(by: row * d * 2), d * 2)
                }
            }
            guard let copy = ctx.queue.makeCommandBuffer(),
                let blit = copy.makeBlitCommandEncoder()
            else { throw MetalError.commandEncoderFailed }
            blit.copy(
                from: staging, sourceOffset: 0, to: runner.hidden,
                destinationOffset: 0, size: d * 2)
            blit.endEncoding()
            copy.commit()
            copy.waitUntilCompleted()
            let norm = try runner.model.inputNorm(layer: layer)
            guard var refCB = ctx.queue.makeCommandBuffer() else {
                throw MetalError.commandEncoderFailed
            }
            try runner.encodeResidualEntryDecode(
                commandBuffer: refCB, hidden: runner.hidden,
                norm: norm, out: runner.normed,
                sublayer: .attention, layer: layer, eps: 1e-6)
            try runner.encodeGatedFullAttentionDecode(
                &refCB, layer: layer, position: 0, slot: row,
                seqLen: 1)
            refCB.commit()
            refCB.waitUntilCompleted()
            #expect(refCB.error == nil)
            reference.append(try download(ctx, from: runner.oOut, bytes: d * 2))
        }
        for row in 0..<2 {
            for i in 0..<d {
                let got = batchOut[row * d + i]
                let want = reference[row][i]
                #expect(
                    abs(Float(got) - Float(want)) < 1e-3,
                    "row \(row) element \(i): \(got) vs \(want)")
            }
        }
    }
}
