import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

/// The first K2 layer arithmetic on the GPU: grouped norm, q/k/v/gate
/// projections, single-token GQA (which is just the value vector), the
/// softplus gate, o_proj, and the dense SwiGLU MLP with both residuals. The
/// toy install's weights are the known 0.01 constant, so the reference is a
/// direct transcription of the CPU formulas.
@Suite struct K2DenseLayerProbeTests {
    private static let eps: Float = 1e-6

    private static func groupNorm(_ x: [Float], weight: [Float], groups: Int) -> [Float] {
        let groupDim = x.count / groups
        var out = [Float](repeating: 0, count: x.count)
        for group in 0..<groups {
            let range = (group * groupDim)..<((group + 1) * groupDim)
            let mean = range.reduce(Float(0)) { $0 + x[$1] * x[$1] } / Float(groupDim)
            let inv = 1 / (mean + eps).squareRoot()
            for i in range {
                out[i] = Float(Float16(x[i] * inv * weight[i]))
            }
        }
        return out
    }

    private static func gemv(_ x: [Float], rows: Int, weight: Float) -> [Float] {
        let dot = x.reduce(Float(0), +)
        return (0..<rows).map { _ in Float(Float16(dot * weight)) }
    }

    private static func softplus(_ x: Float) -> Float {
        let beta = Float(log(2.0))
        let s = beta * x
        let value = s > 0 ? s + log(1 + exp(-s)) : log(1 + exp(s))
        return value / beta
    }

    private static func reference(_ hidden: [Float], config: ArchConfig) -> [Float] {
        let d = config.hiddenSize
        let groups = config.normGroups
        let queryDim = config.numHeads * config.fullHeadDim
        let kvDim = config.numFullKVHeads * config.fullHeadDim
        let ffn = config.intermediateSize
        let weight: Float = 0.01  // toy fixture: packed code 1, scale 0.01
        let normWeight = [Float](repeating: 1, count: d)

        let normed = groupNorm(hidden, weight: normWeight, groups: groups)
        _ = gemv(normed, rows: queryDim, weight: weight)
        _ = gemv(normed, rows: kvDim, weight: weight)
        let v = gemv(normed, rows: kvDim, weight: weight)
        let gate = gemv(normed, rows: queryDim, weight: weight)
        // One token: every query head reads its GQA value head.
        let headsPerKV = config.numHeads / config.numFullKVHeads
        var attn = [Float](repeating: 0, count: queryDim)
        for head in 0..<config.numHeads {
            let kvHead = head / headsPerKV
            for i in 0..<config.fullHeadDim {
                attn[head * config.fullHeadDim + i] =
                    v[kvHead * config.fullHeadDim + i]
                    * softplus(gate[head * config.fullHeadDim + i])
            }
        }
        let o = gemv(attn, rows: d, weight: weight)
        var h1 = [Float](repeating: 0, count: d)
        for i in 0..<d { h1[i] = Float(Float16(hidden[i] + o[i])) }

        let normed2 = groupNorm(h1, weight: normWeight, groups: groups)
        let up = gemv(normed2, rows: ffn, weight: weight)
        let gateMlp = gemv(normed2, rows: ffn, weight: weight)
        var act = [Float](repeating: 0, count: ffn)
        for i in 0..<ffn {
            let g = gateMlp[i]
            act[i] = Float(Float16((g / (1 + exp(-g))) * up[i]))
        }
        let down = gemv(act, rows: d, weight: weight)
        return (0..<d).map { Float(Float16(h1[$0] + down[$0])) }
    }

    @Test func denseLayerMatchesTheReference() throws {
        let directory = try K2ToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: directory) }
        let config = try ArchConfig.resolved(
            forFamily: .k2Horizon, directoryURL: directory)
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: directory, device: ctx.device, expecting: config,
            integrityPolicy: .fullSha256)
        let probe = try K2DenseLayerProbe(model: model, context: ctx, layer: 0)

        var rng = SeedTree(0x2DE_5EED).key("k2-dense-probe")
        let hidden = (0..<config.hiddenSize).map { _ in Float16(rng.uniform(-1, 1)) }
        let hiddenBuf = try #require(Fp16Buffer.make(ctx.device, halves: hidden))
        let outBuf = try #require(Fp16Buffer.make(ctx.device, count: config.hiddenSize))
        let cb = try #require(ctx.queue.makeCommandBuffer())
        try probe.encode(
            commandBuffer: cb, hidden: hiddenBuf, out: outBuf, position: 0)
        cb.commit()
        cb.waitUntilCompleted()

        let actual = Fp16Buffer.read(outBuf, count: config.hiddenSize)
        let expected = Self.reference(hidden.map(Float.init), config: config)
        let rel = RelError.compute(actual: actual, reference: expected)
        #expect(rel < Tolerance.fp16Reduction, "relErr=\(rel)")
    }
}
