import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

/// A full dense-only token through the GPU graph on the synthetic install:
/// embedding → three K2 layers (each with its own cached attention) → final
/// norm → untied head. The reference chains the same formulas used by the
/// single-layer tests.
@Suite struct K2DenseGraphTests {
    private static let eps: Float = 1e-6
    private static let weight: Float = 0.01

    private static func groupNorm(_ x: [Float], groups: Int) -> [Float] {
        let groupDim = x.count / groups
        var out = [Float](repeating: 0, count: x.count)
        for group in 0..<groups {
            let range = (group * groupDim)..<((group + 1) * groupDim)
            let mean = range.reduce(Float(0)) { $0 + x[$1] * x[$1] } / Float(groupDim)
            let inv = 1 / (mean + eps).squareRoot()
            for i in range { out[i] = Float(Float16(x[i] * inv)) }
        }
        return out
    }

    private static func gemv(_ x: [Float], rows: Int) -> [Float] {
        let dot = x.reduce(Float(0), +)
        return (0..<rows).map { _ in Float(Float16(dot * weight)) }
    }

    private static func softplus(_ x: Float) -> Float {
        let beta = Float(log(2.0))
        let s = beta * x
        return (s > 0 ? s + log(1 + exp(-s)) : log(1 + exp(s))) / beta
    }

    private static func layer(_ hidden: [Float], config: ArchConfig) -> [Float] {
        let d = config.hiddenSize
        let queryDim = config.numHeads * config.fullHeadDim
        let kvDim = config.numFullKVHeads * config.fullHeadDim
        let ffn = config.intermediateSize
        let normed = groupNorm(hidden, groups: config.normGroups)
        let q = gemv(normed, rows: queryDim)
        let v = gemv(normed, rows: kvDim)
        let gate = gemv(normed, rows: queryDim)
        var attn = [Float](repeating: 0, count: queryDim)
        let headsPerKV = config.numHeads / config.numFullKVHeads
        for head in 0..<config.numHeads {
            let kvHead = head / headsPerKV
            for i in 0..<config.fullHeadDim {
                let index = head * config.fullHeadDim + i
                let value = v[kvHead * config.fullHeadDim + i]
                // One token: attention is the value, then the gate.
                attn[index] = Float(Float16(value * softplus(gate[index])))
            }
        }
        _ = q
        let o = gemv(attn, rows: d)
        var h1 = [Float](repeating: 0, count: d)
        for i in 0..<d { h1[i] = Float(Float16(hidden[i] + o[i])) }
        let normed2 = groupNorm(h1, groups: config.normGroups)
        let up = gemv(normed2, rows: ffn)
        let mlpGate = gemv(normed2, rows: ffn)
        var act = [Float](repeating: 0, count: ffn)
        for i in 0..<ffn {
            let g = mlpGate[i]
            act[i] = Float(Float16((g / (1 + exp(-g))) * up[i]))
        }
        let down = gemv(act, rows: d)
        return (0..<d).map { Float(Float16(h1[$0] + down[$0])) }
    }

    @Test func oneTokenThroughTheWholeDenseGraph() throws {
        let directory = try K2ToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: directory) }
        let config = try ArchConfig.resolved(
            forFamily: .k2Horizon, directoryURL: directory)
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: directory, device: ctx.device, expecting: config,
            integrityPolicy: .fullSha256)
        let graph = try K2DenseGraph(model: model, context: ctx, maxSequence: 2)
        let cb = try #require(ctx.queue.makeCommandBuffer())
        try graph.encode(commandBuffer: cb, tokenID: 7, position: 0)
        cb.commit()
        cb.waitUntilCompleted()

        // Embedding table: packed code 1, scale 0.01.
        var hidden = [Float](repeating: 0.01, count: config.hiddenSize)
        for _ in 0..<config.numLayers { hidden = Self.layer(hidden, config: config) }
        let normed = Self.groupNorm(hidden, groups: config.normGroups)
        let dot = normed.reduce(Float(0), +) * Self.weight
        let expected = [Float](repeating: Float(Float16(dot)), count: config.vocabSize)
        let actual = graph.readLogits()
        let rel = RelError.compute(actual: actual, reference: expected)
        #expect(rel < Tolerance.fp16Reduction, "relErr=\(rel)")
    }
}
