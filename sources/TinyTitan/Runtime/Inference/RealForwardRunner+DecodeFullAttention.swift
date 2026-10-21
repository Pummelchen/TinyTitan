import Foundation
import Metal

/// Split-out pieces of the gated full-attention decode path: the pre-attention
/// projection/norm/RoPE block and the host-KV gather both branches share.
extension RealForwardRunner {
    /// The windowed indirection for one decode step, or nil when this layer's
    /// rows still live contiguously in VRAM.
    func decodeWindowSelection(
        kv: KVCacheManager,
        layer: Int,
        slot: Int,
        position: Int,
        seqLen: UInt32,
        selection: DecodeKeySelection?
    ) throws -> KVWindowSelection? {
        guard kv.streamsLayer(layer) else { return nil }
        return try kv.prepareDecodeWindow(
            layer: layer, slot: slot, position: position,
            indices: selection?.indices,
            count: selection?.indices == nil
                ? Int(seqLen) : (selection?.count ?? Int(seqLen)))
    }

    /// Packed q_proj split, per-head q/k norms, NeoX sub-dim RoPE, and the
    /// quantized-KV write of one gated full-attention layer.
    func encodeGatedFullPreAttention(
        _ cb: inout MTLCommandBuffer,
        layer L: Int,
        position: Int,
        kWrite: (buffer: MTLBuffer, offset: Int),
        vWrite: (buffer: MTLBuffer, offset: Int),
        qDim: UInt32,
        kvDim: UInt32,
        headDim: Int,
        numKV: Int,
        eps: Float
    ) throws {
        guard let elementwise, let rope, let qPackedScratch, let attnGateScratch
        else {
            throw ModelError.internalInconsistency(
                detail:
                    "attn_output_gate layer \(L) without gate kernels (arch mask misconfiguration)")
        }
        let qNormW = try model.qNorm(layer: L)
        let kNormW = try model.kNorm(layer: L)
        let rotaryDim = UInt32(Double(headDim) * cfg.partialRotaryFactor)
        try encodeGatedFullQKVProjection(
            cb, layer: L, qOutput: qPackedScratch,
            kOutput: kWrite, vOutput: vWrite,
            qDimension: qDim, kvDimension: kvDim)
        try rotate(&cb, role: "qsa.qkv")
        try elementwise.encodeSplitQGate(
            commandBuffer: cb,
            packed: qPackedScratch,
            q: qScratch,
            gate: attnGateScratch,
            heads: cfg.numHeads,
            dim: headDim)
        try rms.encodeBF16WPerHead(
            commandBuffer: cb,
            x: qScratch,
            weight: qNormW.buffer,
            weightOffset: Int(qNormW.offset),
            out: qScratch,
            headDim: UInt32(headDim),
            numHeads: cfg.numHeads,
            eps: eps)
        try rms.encodeBF16WPerHead(
            commandBuffer: cb,
            x: kWrite.buffer, xOffset: kWrite.offset,
            weight: kNormW.buffer,
            weightOffset: Int(kNormW.offset),
            out: kWrite.buffer, outOffset: kWrite.offset,
            headDim: UInt32(headDim),
            numHeads: numKV,
            eps: eps)
        try rope.encodeNeoxSubdim(
            commandBuffer: cb,
            data: qScratch,
            position: UInt32(position),
            headDim: UInt32(headDim),
            numHeads: UInt32(cfg.numHeads),
            rotaryDim: rotaryDim,
            theta: Float(cfg.fullRopeTheta))
        try rotate(&cb, role: "qsa.gate_norm_rope1")
        try rope.encodeNeoxSubdim(
            commandBuffer: cb,
            data: kWrite.buffer,
            dataOffset: kWrite.offset,
            position: UInt32(position),
            headDim: UInt32(headDim),
            numHeads: UInt32(numKV),
            rotaryDim: rotaryDim,
            theta: Float(cfg.fullRopeTheta))
    }
}
