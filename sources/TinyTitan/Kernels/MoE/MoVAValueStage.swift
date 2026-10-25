import Foundation
import Metal

/// K2-Horizon's MoVA value stage: a bf16 value router selects 4 of 64 experts,
/// each expert projects `hidden -> kvHeads * headDim` from 4-bit resident
/// weights, and the four outputs are SiLU-activated, weighted and summed.
///
/// The selection runs on the GPU, but the four expert GEMVs are separate
/// dispatches over different resident weights, so their ids have to come back
/// to the CPU between the two steps. `encodeRouter` and `encodeValues` are
/// therefore two encodes with a completed command buffer between them -- the
/// same split the routed MoE already uses for its expert fetches.
///
/// The value experts stay resident in the install (top-4 of 64 is far smaller
/// than the routed MoE), so nothing here streams from disk.
final class MoVAValueStage {
    /// Four is the kernel's fixed slot count and K2's `top_k_experts` for the
    /// value router. A different count is new kernel work, not a parameter.
    static let topK = 4
    static let maximumExperts = 128

    /// One expert's resident weights. The GEMV reads them in place, so the
    /// stage never copies a checkpoint tensor.
    struct ExpertWeights {
        let weights: MTLBuffer
        let weightsOffset: Int
        let scales: MTLBuffer
        let scalesOffset: Int
        let biases: MTLBuffer
        let biasesOffset: Int

        init(
            weights: MTLBuffer, weightsOffset: Int = 0,
            scales: MTLBuffer, scalesOffset: Int = 0,
            biases: MTLBuffer, biasesOffset: Int = 0
        ) {
            self.weights = weights
            self.weightsOffset = weightsOffset
            self.scales = scales
            self.scalesOffset = scalesOffset
            self.biases = biases
            self.biasesOffset = biasesOffset
        }
    }

    private let bf16: BF16GEMV
    private let gemv: AffineQuantGEMV
    private let selection: K2RouterSelection
    private let combinePSO: MTLComputePipelineState
    private let hiddenSize: Int
    private let valueDim: Int
    private let maxExperts: Int
    /// Float logits: the selector needs the unrounded scores.
    private let logits: MTLBuffer
    private let indices: MTLBuffer
    private let weights: MTLBuffer
    private let expertOutputs: MTLBuffer

    init(
        context: MetalContext,
        weightBits: Int = 4,
        hiddenSize: Int,
        valueDim: Int,
        maxExperts: Int = MoVAValueStage.maximumExperts
    ) throws {
        precondition(
            hiddenSize > 0 && hiddenSize % 64 == 0,
            "MoVA hidden size must be a multiple of 64")
        precondition(valueDim > 0, "MoVA value dimension must be positive")
        precondition(
            maxExperts > 0 && maxExperts <= Self.maximumExperts,
            "maxExperts \(maxExperts) is outside 1...\(Self.maximumExperts)")
        self.hiddenSize = hiddenSize
        self.valueDim = valueDim
        self.maxExperts = maxExperts
        self.bf16 = try BF16GEMV(context: context)
        self.gemv = try AffineQuantGEMV(context: context, weightBits: weightBits)
        self.selection = try K2RouterSelection(context: context)
        self.combinePSO = try context.pipeline("mova_combine_fp16")

        let device = context.device
        func make(_ length: Int, _ label: String) throws -> MTLBuffer {
            guard
                let buffer = device.makeBuffer(
                    length: length, options: .storageModeShared)
            else {
                throw MetalError.bufferAllocationFailed(label)
            }
            return buffer
        }
        self.logits = try make(
            maxExperts * MemoryLayout<Float>.stride, "mova.routerLogits")
        self.indices = try make(
            Self.topK * MemoryLayout<UInt32>.stride, "mova.indices")
        self.weights = try make(
            Self.topK * MemoryLayout<UInt16>.stride, "mova.weights")
        self.expertOutputs = try make(
            Self.topK * valueDim * MemoryLayout<UInt16>.stride,
            "mova.expertOutputs")
    }

    /// Router logits and selection. `hidden` is the post-attention normed
    /// input; `routerWeight` is `[numExperts, hiddenSize]` bf16 and
    /// `routerBias` is `[numExperts]` bf16.
    func encodeRouter(
        commandBuffer: MTLCommandBuffer,
        hidden: MTLBuffer, hiddenOffset: Int = 0,
        routerWeight: MTLBuffer, routerWeightOffset: Int = 0,
        routerBias: MTLBuffer, routerBiasOffset: Int = 0,
        numExperts: Int,
        scaling: Float
    ) throws {
        precondition(
            (Self.topK...maxExperts).contains(numExperts),
            "MoVA has \(numExperts) experts; the stage holds \(maxExperts) at top-\(Self.topK)"
        )
        try bf16.encodeFloat(
            commandBuffer: commandBuffer,
            weights: routerWeight, weightsOffset: routerWeightOffset,
            x: hidden, xOffset: hiddenOffset,
            y: logits, m: UInt32(numExperts), n: UInt32(hiddenSize))
        try selection.encode(
            commandBuffer: commandBuffer,
            logits: logits,
            bias: routerBias, biasOffset: routerBiasOffset,
            outIndices: indices, outWeights: weights,
            numExperts: numExperts, topK: Self.topK, scaling: scaling)
    }

    /// The four selected expert ids, in selection order. Meaningful only after
    /// the command buffer that encoded `encodeRouter` has completed.
    func selectedExpertIDs() -> [Int] {
        let base = indices.contents()
        return (0..<Self.topK).map {
            Int(
                base.load(
                    fromByteOffset: $0 * MemoryLayout<UInt32>.stride, as: UInt32.self))
        }
    }

    /// The four value GEMVs and the SiLU-weighted sum. `experts` is indexed by
    /// the router's expert id. The command buffer that encoded `encodeRouter`
    /// must have completed before this is encoded.
    func encodeValues(
        commandBuffer: MTLCommandBuffer,
        hidden: MTLBuffer, hiddenOffset: Int = 0,
        experts: [ExpertWeights],
        out: MTLBuffer, outOffset: Int = 0
    ) throws {
        let ids = selectedExpertIDs()
        for (slot, id) in ids.enumerated() {
            guard experts.indices.contains(id) else {
                throw ModelError.internalInconsistency(
                    detail: "MoVA selected expert \(id), but \(experts.count) are bound")
            }
            let expert = experts[id]
            try gemv.encode(
                commandBuffer: commandBuffer,
                weights: expert.weights, weightsOffset: expert.weightsOffset,
                scales: expert.scales, scalesOffset: expert.scalesOffset,
                biases: expert.biases, biasesOffset: expert.biasesOffset,
                x: hidden, xOffset: hiddenOffset,
                y: expertOutputs,
                yOffset: slot * valueDim * MemoryLayout<UInt16>.stride,
                m: UInt32(valueDim), n: UInt32(hiddenSize))
        }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(combinePSO)
        encoder.setBuffer(expertOutputs, offset: 0, index: 0)
        encoder.setBuffer(weights, offset: 0, index: 1)
        encoder.setBuffer(out, offset: outOffset, index: 2)
        var count = UInt32(valueDim)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 3)
        let width = min(combinePSO.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: valueDim, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
