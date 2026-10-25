import Foundation
import Metal

/// K2-Horizon's sparse expert block: bf16 router, sigmoid selection, then the
/// selected SwiGLU experts and the always-on shared expert.
///
/// The routed experts are separate GEMV dispatches over different blobs, so
/// the stage owns one gate/up/act/down scratch slot per selected expert and a
/// final pass applies the router weights and adds the result to the shared
/// expert's output. That is the resident form; a streaming caller hands the
/// same `ExpertWeights` descriptors for whichever blobs are in memory.
final class K2MoEStage {
    struct ExpertWeights {
        let gate: TensorView
        let up: TensorView
        let down: TensorView
    }

    private let hiddenSize: Int
    private let expertWidth: Int
    private let numExperts: Int
    private let topK: Int
    private let bf16: BF16GEMV
    private let gemv: AffineQuantGEMV
    private let selection: K2RouterSelection
    private let elementwise: Elementwise
    private let combinePSO: MTLComputePipelineState
    private let logits: MTLBuffer
    private let indices: MTLBuffer
    private let weights: MTLBuffer
    private let gateScratch: MTLBuffer
    private let upScratch: MTLBuffer
    private let actScratch: MTLBuffer
    private let downScratch: MTLBuffer
    private let sharedGate: MTLBuffer
    private let sharedUp: MTLBuffer
    private let sharedAct: MTLBuffer

    init(
        context: MetalContext,
        weightBits: Int,
        hiddenSize: Int,
        expertWidth: Int,
        numExperts: Int,
        topK: Int
    ) throws {
        precondition(hiddenSize > 0 && hiddenSize % 64 == 0)
        precondition(expertWidth > 0)
        precondition(numExperts > 0 && topK > 0 && topK <= min(16, numExperts))
        self.hiddenSize = hiddenSize
        self.expertWidth = expertWidth
        self.numExperts = numExperts
        self.topK = topK
        self.bf16 = try BF16GEMV(context: context)
        self.gemv = try AffineQuantGEMV(context: context, weightBits: weightBits)
        self.selection = try K2RouterSelection(context: context)
        self.elementwise = try Elementwise(context: context)
        self.combinePSO = try context.pipeline("k2_moe_accumulate_fp16")

        func buffer(_ count: Int, _ label: String) throws -> MTLBuffer {
            guard
                let buffer = context.device.makeBuffer(
                    length: max(count * MemoryLayout<UInt16>.stride, 4),
                    options: .storageModeShared)
            else {
                throw MetalError.bufferAllocationFailed(label)
            }
            return buffer
        }
        func rawBuffer(_ length: Int, _ label: String) throws -> MTLBuffer {
            guard
                let buffer = context.device.makeBuffer(
                    length: max(length, 4), options: .storageModeShared)
            else {
                throw MetalError.bufferAllocationFailed(label)
            }
            return buffer
        }
        self.logits = try rawBuffer(
            numExperts * MemoryLayout<Float>.stride, "k2.moe.logits")
        self.indices = try rawBuffer(
            topK * MemoryLayout<UInt32>.stride, "k2.moe.indices")
        self.weights = try buffer(topK, "k2.moe.weights")
        self.gateScratch = try buffer(topK * expertWidth, "k2.moe.gate")
        self.upScratch = try buffer(topK * expertWidth, "k2.moe.up")
        self.actScratch = try buffer(topK * expertWidth, "k2.moe.act")
        self.downScratch = try buffer(topK * hiddenSize, "k2.moe.down")
        self.sharedGate = try buffer(expertWidth, "k2.moe.sharedGate")
        self.sharedUp = try buffer(expertWidth, "k2.moe.sharedUp")
        self.sharedAct = try buffer(expertWidth, "k2.moe.sharedAct")
    }

    /// The router GEMV (float logits) and selection. `hidden` is the post-norm
    /// block input.
    func encodeRouter(
        commandBuffer: MTLCommandBuffer,
        hidden: MTLBuffer, hiddenOffset: Int = 0,
        routerWeight: MTLBuffer, routerWeightOffset: Int = 0,
        routerBias: MTLBuffer, routerBiasOffset: Int = 0,
        scaling: Float
    ) throws {
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
            numExperts: numExperts, topK: topK, scaling: scaling)
    }

    /// Meaningful after the router command buffer completed.
    func selectedExpertIDs() -> [Int] {
        let base = indices.contents()
        return (0..<topK).map {
            Int(
                base.load(
                    fromByteOffset: $0 * MemoryLayout<UInt32>.stride,
                    as: UInt32.self))
        }
    }

    /// The shared expert then the selected routed experts, summed into `out`.
    /// The command buffer that encoded `encodeRouter` must have completed.
    func encodeExperts(
        commandBuffer: MTLCommandBuffer,
        hidden: MTLBuffer, hiddenOffset: Int = 0,
        experts: [ExpertWeights],
        shared: ExpertWeights?,
        out: MTLBuffer, outOffset: Int = 0
    ) throws {
        if let shared {
            try encodeSwiGLU(
                commandBuffer: commandBuffer,
                weights: shared,
                x: hidden, xOffset: hiddenOffset,
                gate: sharedGate, up: sharedUp, act: sharedAct,
                y: out, yOffset: outOffset,
                rows: hiddenSize, width: expertWidth)
        }

        let ids = selectedExpertIDs()
        for (slot, id) in ids.enumerated() {
            guard experts.indices.contains(id) else {
                throw ModelError.internalInconsistency(
                    detail: "K2 MoE selected expert \(id), but \(experts.count) are bound")
            }
            try encodeSwiGLU(
                commandBuffer: commandBuffer,
                weights: experts[id],
                x: hidden, xOffset: hiddenOffset,
                gate: gateScratch, up: upScratch, act: actScratch,
                y: downScratch,
                yOffset: slot * hiddenSize * MemoryLayout<UInt16>.stride,
                rows: hiddenSize, width: expertWidth)
        }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(combinePSO)
        encoder.setBuffer(downScratch, offset: 0, index: 0)
        encoder.setBuffer(weights, offset: 0, index: 1)
        encoder.setBuffer(out, offset: outOffset, index: 2)
        var count = UInt32(hiddenSize)
        var slots = UInt32(topK)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&slots, length: MemoryLayout<UInt32>.stride, index: 4)
        let width = min(combinePSO.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: hiddenSize, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeSwiGLU(
        commandBuffer: MTLCommandBuffer,
        weights: ExpertWeights,
        x: MTLBuffer, xOffset: Int,
        gate: MTLBuffer, up: MTLBuffer, act: MTLBuffer,
        y: MTLBuffer, yOffset: Int,
        rows: Int, width: Int
    ) throws {
        try gemv(
            weights.gate, commandBuffer: commandBuffer, x: x, xOffset: xOffset, y: gate,
            rows: width, columns: hiddenSize)
        try gemv(
            weights.up, commandBuffer: commandBuffer, x: x, xOffset: xOffset, y: up, rows: width,
            columns: hiddenSize)
        try elementwise.encodeSiluMul(
            commandBuffer: commandBuffer,
            gate: gate, up: up, out: act, count: width)
        try gemv(
            weights.down, commandBuffer: commandBuffer, x: act, y: y, yOffset: yOffset, rows: rows,
            columns: width)
    }

    private func gemv(
        _ view: TensorView,
        commandBuffer: MTLCommandBuffer,
        x: MTLBuffer, xOffset: Int = 0,
        y: MTLBuffer, yOffset: Int = 0,
        rows: Int, columns: Int
    ) throws {
        if view.dtype == 1 {
            throw ModelError.internalInconsistency(
                detail: "K2 MoE expects quantized expert projections")
        }
        try gemv.encode(
            commandBuffer: commandBuffer,
            weights: view.buffer, weightsOffset: Int(view.offset),
            scales: view.buffer, scalesOffset: Int(view.scaleOffset),
            biases: view.buffer, biasesOffset: Int(view.biasOffset),
            x: x, xOffset: xOffset, y: y, yOffset: yOffset,
            m: UInt32(rows), n: UInt32(columns))
    }
}
