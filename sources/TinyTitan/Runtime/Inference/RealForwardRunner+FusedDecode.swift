import Foundation
import Metal

// T34: the fused width-2 decode loop.
//
// Attention is batched through `encodeBatchAttentionBlock` (entry norm, one
// QKV/o_proj pair of GEMVs, per-row epilogue/attention/KV write, residual and
// post-attention norm). GDN layers stay per row with the row-sliced linear
// decoder. The FFN runs the production per-row decode stage on each row's
// slice, so its arithmetic is the per-row path's rather than a second
// implementation. The loop is opt-in until the T34 gates pass; T7 flips the
// default.
extension RealForwardRunner {
    /// Opt-in switch for the fused loop. T7 replaces this with the server's
    /// width selection once the gates are green.
    static var fusedDecodeOptIn: Bool {
        ProcessInfo.processInfo.environment["TINYTITAN_FUSED_DECODE"] == "1"
    }

    /// The families and layouts the fused loop implements today. Anything the
    /// loop does not model -- PLE, QSA, hyper-connections, host-KV streaming,
    /// activation dumps, widths other than two -- falls back to the per-row
    /// path rather than running with a missing stage.
    func canUseFusedBatchDecode(
        _ rows: [(token: Int32, position: Int, slot: Int)]
    ) -> Bool {
        guard fusedDecodeOverride ?? Self.fusedDecodeOptIn else { return false }
        guard rows.count == 2, slots >= 2 else { return false }
        guard cfg.family == .qwen36 else { return false }
        guard cfg.ple.layerIndices.isEmpty else { return false }
        guard !cfg.hyperConnections.enabled else { return false }
        guard qsaIndexer == nil else { return false }
        guard activationDumpDirectory == nil else { return false }
        guard let kv, !kv.streamsKVFromHost else { return false }
        guard cfg.numExperts > 0, cfg.topKExperts > 0 else { return false }
        guard
            model.qoProjectionWeightBits == 4 || model.qoProjectionWeightBits == 8
        else { return false }
        return true
    }

    private func encodeEmbeddingRow(
        token: Int32, into out: MTLBuffer, outOffset: Int,
        commandBuffer cb: MTLCommandBuffer
    ) throws {
        let D = UInt32(cfg.hiddenSize)
        let scale =
            cfg.embeddingScaledBySqrtHidden
            ? Float(cfg.hiddenSize).squareRoot() : 1.0
        let emb = try model.embedding()
        if let affineEmbed {
            try affineEmbed.encode(
                commandBuffer: cb,
                table: emb.buffer, tableOffset: Int(emb.offset),
                scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                out: out, outOffset: outOffset,
                tokenId: UInt32(bitPattern: token),
                d: D, outScale: scale, vocab: UInt32(cfg.vocabSize))
        } else {
            try embedInt4.encode(
                commandBuffer: cb,
                table: emb.buffer, tableOffset: Int(emb.offset),
                scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                out: out, outOffset: outOffset,
                tokenId: UInt32(bitPattern: token),
                d: D, outScale: scale, vocab: UInt32(cfg.vocabSize))
        }
    }

    /// Runs one width-2 decode step with the fused loop. Returns false when the
    /// batch is not eligible, so `produceBatch` can fall back without having
    /// touched any state.
    ///
    /// lint:allow-long one decode step is a single ordered pass -- embedding,
    /// forty layers of batch attention plus per-row FFN, final norm, head --
    /// whose stages share the row cursor, the command-buffer chain and the
    /// row buffers; splitting it would thread a dozen live locals through
    /// helpers that are not independently testable.
    func produceBatchFused(
        _ rows: [(token: Int32, position: Int, slot: Int)],
        logits: [MTLBuffer]
    ) async throws -> Bool {
        guard canUseFusedBatchDecode(rows) else { return false }
        guard let kv else { return false }
        fusedBatchDecodeLaunches += 1
        // Same handover the per-row decode does: wire the routed-expert slot
        // cache for the whole generation before its slots can be reclaimed
        // between the plan and the GPU's read of them.
        model.setExpertCachePinned(true)
        let count = rows.count
        let d = cfg.hiddenSize
        let half = MemoryLayout<Float16>.stride
        let D = UInt32(d)
        let eps: Float = 1e-6

        for row in rows {
            let cursor = kv.position(slot: row.slot)
            guard cursor == row.position else {
                throw PrefillError.prefillCursorMismatch(
                    "fused cursor \(cursor) != position \(row.position) for slot \(row.slot)")
            }
            guard row.position < maxContext else {
                throw PrefillError.prefillCursorMismatch(
                    "fused position \(row.position) exceeds maxContext \(maxContext)")
            }
            try kv.reserve(tokens: row.position + 1, slot: row.slot)
            try requireQSAExact(visibleKeys: row.position + 1)
        }

        let scratch = try ensureBatchedDecodeScratch(rows: count)
        let views = try makeLayerPrefillViews()
        let fNorm = try model.finalNorm()
        let (normed, batchLogits) = try ensureBatchScratch(rows: count)

        guard var cb = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        for (index, row) in rows.enumerated() {
            try encodeEmbeddingRow(
                token: row.token, into: scratch.hidden,
                outOffset: index * d * half, commandBuffer: cb)
        }
        cb.commit()
        try waitForCompletion(cb)

        for L in 0..<cfg.numLayers {
            guard var layerCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            if cfg.layerIsLinear(L) {
                for (index, row) in rows.enumerated() {
                    let offset = index * d * half
                    try encodeResidualEntryDecode(
                        commandBuffer: layerCB,
                        hidden: scratch.hidden, hiddenOffset: offset,
                        norm: views[L].inputNorm,
                        out: scratch.normed, outOffset: offset,
                        sublayer: .attention, layer: L, eps: eps)
                    try encodeLinearAttentionDecode(
                        &layerCB, layer: L, slot: row.slot,
                        input: scratch.normed, inputOffset: offset,
                        output: scratch.o, outputOffset: offset)
                    try encodeResidualExitDecode(
                        commandBuffer: layerCB,
                        hidden: scratch.hidden, hiddenOffset: offset,
                        delta: scratch.o, deltaOffset: offset,
                        sublayer: .attention, layer: L)
                    try encodeResidualEntryDecode(
                        commandBuffer: layerCB,
                        hidden: scratch.hidden, hiddenOffset: offset,
                        norm: views[L].postAttention,
                        out: scratch.routedX, outOffset: offset,
                        sublayer: .mlp, layer: L, eps: eps)
                }
            } else {
                try encodeBatchAttentionBlock(
                    layer: L,
                    rows: rows.map { (position: $0.position, slot: $0.slot) },
                    scratch: scratch,
                    postAttnNorm: views[L].postAttention,
                    commandBuffer: &layerCB)
            }

            // Attention and the post-attention norm for the batch are encoded;
            // commit them before each row's router reads that row's slice.
            layerCB.commit()
            try waitForCompletion(layerCB)
            let sharedProj = sharedExpertProjections[L]
            guard let routerView = views[L].router else {
                throw ModelError.internalInconsistency(
                    detail: "fused loop reached a routed layer without a router")
            }
            let perExpertScale = try requireOnesPerExpertScale()
            for (index, row) in rows.enumerated() {
                let offset = index * d * half
                guard let routerCB = ctx.queue.makeCommandBuffer() else {
                    throw ModelError.residentBufferWrapFailed
                }
                try moe.encodeRouter(
                    commandBuffer: routerCB,
                    weights: routerView.buffer,
                    weightsOffset: Int(routerView.offset),
                    scales: routerView.buffer,
                    scalesOffset: Int(routerView.scaleOffset),
                    biases: routerView.buffer,
                    biasesOffset: Int(routerView.biasOffset),
                    hidden: scratch.routedX, hiddenOffset: offset,
                    effectiveScale: effectiveScaleBuffers[L],
                    perExpertScale: perExpertScale,
                    perExpertScaleOffset: 0,
                    outIndices: outIndices, outWeights: outWeights,
                    numExperts: UInt32(cfg.numExperts), d: D,
                    topK: UInt32(cfg.topKExperts))
                routerCB.commit()
                guard let sharedCB = ctx.queue.makeCommandBuffer() else {
                    throw ModelError.residentBufferWrapFailed
                }
                try encodeSharedExpert(
                    layer: L,
                    x: scratch.routedX, xOffset: offset,
                    y: h1Buf, yOffset: 0,
                    commandBuffer: sharedCB)
                sharedCB.commit()
                try waitForCompletion(routerCB)
                // The phase-1 kernels read the runner-wide `routedX`; stage
                // this row's slice there. The copy is ordered on the same
                // queue before the routed command that reads it.
                guard let copyCB = ctx.queue.makeCommandBuffer() else {
                    throw ModelError.residentBufferWrapFailed
                }
                guard let blit = copyCB.makeBlitCommandEncoder() else {
                    throw MetalError.commandEncoderFailed
                }
                blit.copy(
                    from: scratch.routedX, sourceOffset: offset,
                    to: self.routedX, destinationOffset: 0, size: d * half)
                blit.endEncoding()
                copyCB.commit()
                var pending: PendingRoutedCommand?
                try await encodeDecodeRoutedMoE(
                    layer: L, position: row.position,
                    sharedProj: sharedProj,
                    attnCB: routerCB, tailCB: routerCB, sharedCB: sharedCB,
                    overlapCompletionClock: nil,
                    pending: &pending,
                    bodyStart: 0, cb1Start: 0, waitMark: 0, waitNanos: 0,
                    previousRoutedMicros: 0, predictedNextLayer: [],
                    hidden: scratch.hidden, hiddenOffset: offset)
                if let pending {
                    try finishPendingRoutedCommand(pending, waitIfNeeded: true)
                }
            }
            guard let nextCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            cb = nextCB
        }

        for index in rows.indices {
            // `encodeFinalNorm` normalizes the runner-wide `hidden`; the batch
            // rows live in `scratch.hidden`, so dispatch the same RMSNorm on
            // the row slice here.
            try rms.encodeBF16W(
                commandBuffer: cb,
                x: scratch.hidden, xOffset: index * d * half,
                weight: fNorm.buffer, weightOffset: Int(fNorm.offset),
                out: normed, outOffset: index * d * half,
                d: D, eps: eps)
        }
        cb.commit()
        try waitForCompletion(cb)
        try encodeBatchHead(
            rows: count, logits: logits,
            normed: normed, batchLogits: batchLogits)
        for row in rows {
            kv.advance(slot: row.slot, by: 1)
        }
        return true
    }
}
