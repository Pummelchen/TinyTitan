import Foundation
import Metal

/// Single-token decode: the token entry points and the layer loop.
///
/// Split from RealForwardRunner.swift in the modularity refactor
/// (docs/modularity-refactor.md) as pure code motion: one concern
/// per file, no signature or behavior changes. On 2026-09-28 the stage code
/// moved to its own files on the same terms — the routed MoE to
/// `RealForwardRunner+DecodeMoE.swift`, the attention dispatch to
/// `RealForwardRunner+DecodeAttention.swift`, the GEMV dispatch and its
/// diagnostics to `RealForwardRunner+DecodeGEMV.swift` — leaving one file for
/// the loop that calls them.
extension RealForwardRunner {
    public func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
        try await produce(token: token, position: position, slot: 0, into: logits)
    }

    /// Decode one token for sequence `slot`. Slot 0 is the single-sequence path
    /// every existing caller takes; another slot reads and writes that slot's
    /// own KV region and GDN state, so a batched step can advance several
    /// sequences through one runner without aliasing.
    public func produce(
        token: Int32, position: Int, slot: Int,
        into logits: MTLBuffer
    ) async throws {
        try await forwardStepGate.acquire()
        do {
            // Checked under the gate: the commit state is runner-wide, so
            // another slot's in-flight prefill is only visible here, not before
            // the gate.
            try prefillChunkState.requireClean(operation: "produce")
            try await produceToken(
                token: token,
                position: position,
                slot: slot,
                into: logits,
                emitHead: true,
                outputMode: .greedyIfAvailable)
        } catch {
            await forwardStepGate.release()
            throw error
        }
        await forwardStepGate.release()
    }

    /// Decode one row per slot in `rows`, each row's logits landing in the
    /// matching buffer of `logits`. The rows advance in call order; the
    /// token-wise stages are still run per row (not yet fused across the
    /// batch), so this is the correctness-first batched entry point.
    public func produceBatch(
        _ rows: [(token: Int32, position: Int, slot: Int)],
        logits: [MTLBuffer]
    ) async throws {
        precondition(
            rows.count == logits.count,
            "one logits buffer per batch row")
        for (row, buffer) in zip(rows, logits) {
            try await produce(
                token: row.token, position: row.position,
                slot: row.slot, into: buffer)
        }
    }

    /// lint:allow-long the orchestrator for one decode step, in the same
    /// shape as executePrefillChunk: embed, the per-layer dispatch, the head.
    func produceToken(
        token: Int32,
        position: Int,
        slot: Int,
        into logits: MTLBuffer,
        emitHead: Bool,
        outputMode: PrefillOutputMode
    ) async throws {
        let tPreamble = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let kvPosition = kv?.position(slot: slot) ?? 0
        guard kvPosition == position else {
            throw PrefillError.prefillCursorMismatch(
                "produce cursor \(kvPosition) != position \(position) for slot \(slot)")
        }
        // Decode must not share RAM with an idle ANE context (Track A):
        // prompts that end exactly on a chunk boundary reach here with the
        // last model still resident. No-op when ANE prefill is off or empty.
        let handoverStart =
            PreadExpertStreamer.wireTraceEnabled
            ? clock_gettime_nsec_np(CLOCK_UPTIME_RAW) : 0
        let tRelease = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        anePrefill?.releaseModels()
        totalPreambleReleaseNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tRelease
        if PreadExpertStreamer.wireTraceEnabled, handoverStart != 0 {
            let ms =
                Double(
                    clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                        - handoverStart) / 1e6
            if ms > 1 {
                FileHandle.standardError.write(
                    Data(
                        "[wire] releaseModels \(String(format: "%.1f", ms)) ms\n".utf8))
            }
        }
        // Snapshot expert I/O at the handover so decode's share can be
        // separated from prefill's. The two phases stream through the same
        // cache, so a whole-request total cannot answer whether ANE prefill
        // leaves decode re-reading experts -- which is the standing claim
        // ("94% more expert I/O after ANE prefill") that has never been
        // tested directly.
        if Self.decodeIOTraceEnabled, decodeIOBaseline == nil {
            decodeIOBaseline = model.routedExpertStatistics()
        }
        // Wire the slot cache for decode. Unwired, unrelated memory churn can
        // reclaim the budget and decode then re-reads routed experts from SSD
        // for the rest of the request -- measured at 94% more expert I/O after
        // ANE prefill.
        //
        // This is the only place the cache is ever wired: the release at
        // prefill start is a no-op (see there). Wiring costs ~136 ms for
        // 4.2 GiB across 40 layers, so it is not itself a decode cost --
        // wiring at allocation instead measured identically (-28.5% against
        // -27.9% for 4-bit ANE decode), which is why no wiring policy ships.
        let tPin = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        // Wire once per generation, retrying only while a wire was partial.
        // Calling into the model every token measured 21-125 ms per token
        // on Qwen3.8 4-bit under memory pressure (pre_pin_ms), with no
        // mlock and no queue wait inside it.
        // The model returns early once every opened layer is wired; the walk
        // only runs after an unpin, a partial wire, or a newly opened layer.
        model.setExpertCachePinned(true)
        let tReserve = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        totalPreamblePinNanos &+= tReserve - tPin
        totalPreambleReleaseNanos = model.expertCachePinQueueWaitNanos
        try kv?.reserve(tokens: position + 1, slot: slot)
        totalPreambleReserveNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tReserve
        guard position < maxContext else {
            throw PrefillError.prefillCursorMismatch(
                "produce position \(position) exceeds maxContext \(maxContext)")
        }
        // A decode step at `position` makes position + 1 keys visible.
        try requireQSAExact(visibleKeys: position + 1)
        let D = UInt32(cfg.hiddenSize)
        let eps: Float = 1e-6
        let embedOutScale =
            cfg.embeddingScaledBySqrtHidden
            ? Float(cfg.hiddenSize).squareRoot()
            : 1.0
        var pendingRoutedCommand: PendingRoutedCommand?

        /// Drain a routed layer's command buffers, surfacing any `.error`
        /// (R1/R2): the routed-CB failure must fail the generation rather than
        /// print-and-continue into silently corrupt output. The per-layer call
        /// (waitIfNeeded: false) runs right after the next layer's tailCB
        /// wait, so the routed CBs have completed on the GPU and their spans
        /// are valid — recording them here (not only in the waitIfNeeded
        /// drain) makes TINYTITAN_KERNEL_STATS cover every layer instead of just
        /// the final layer of each token.

        totalPreambleNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tPreamble
        let tEmbed = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        // Embed lookup + sqrt(H) fused.
        let emb = try model.embedding()
        let embedCB = try runSync { cb in
            if let affineEmbed {
                try affineEmbed.encode(
                    commandBuffer: cb,
                    table: emb.buffer, tableOffset: Int(emb.offset),
                    scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                    biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                    out: hidden, tokenId: UInt32(bitPattern: token),
                    d: D, outScale: embedOutScale,
                    vocab: UInt32(cfg.vocabSize))
            } else {
                try embedInt4.encode(
                    commandBuffer: cb,
                    table: emb.buffer, tableOffset: Int(emb.offset),
                    scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                    biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                    out: hidden,
                    tokenId: UInt32(bitPattern: token),
                    d: D,
                    outScale: embedOutScale,
                    vocab: UInt32(cfg.vocabSize))
            }
        }
        guard embedCB != nil else {
            throw ModelError.residentBufferWrapFailed
        }
        // Entry to a hyper-connection stack: every stream starts from the
        // token embedding. The embed kernel wrote stream 0; replicate it.
        if cfg.hyperConnections.enabled {
            _ = try runSync { cb in
                try requireElementwise().encodeHCBroadcast(
                    commandBuffer: cb, streams: hidden,
                    dim: cfg.hiddenSize,
                    streamCount: cfg.hyperConnections.count)
            }
        }
        if let embedCB { recordKernelGPU(role: "embed", embedCB) }
        totalEmbedNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tEmbed
        // The n-gram rows depend only on this token and its predecessors, so
        // the gather can run here, before any layer needs it.
        predictivePrefetch?.beginToken()
        let tGather = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        try gatherPLERows(token: token)
        totalGatherNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tGather
        if activationDumpActive(position: position) {
            dumpActivationToken(token, position: position)
            dumpActivation("embed", hidden, count: residualWidth, position: position)
            if let ple = pleBlock {
                dumpActivation(
                    "ple_embedding", ple.embedding,
                    count: cfg.ple.embedDim, position: position)
            }
        }

        for L in 0..<cfg.numLayers {
            // Dumping drains the previous layer's routed command first. The
            // residual is only settled once that has landed, and a dump taken
            // at encode time would read whatever the buffer held before the
            // GPU ran -- which reads exactly like a wrong answer.
            if activationDumpActive(position: position) {
                if let pending = pendingRoutedCommand {
                    try finishPendingRoutedCommand(pending, waitIfNeeded: true)
                    pendingRoutedCommand = nil
                    if L <= dumpLayerLimit {
                        dumpActivation(
                            "L\(L - 1)_mlp_out", h2Buf,
                            count: cfg.hiddenSize, position: position)
                        dumpActivation(
                            "L\(L - 1)_moe_acts", moeActs,
                            count: cfg.topKExperts
                                * cfg.moeIntermediateSize,
                            position: position)
                    }
                }
                dumpActivation("L\(L)_entry", hidden, count: residualWidth, position: position)
            }
            let tBodyStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let isLinear = cfg.layerIsLinear(L)
            // A dense model has no routed mixture at all: its FFN is the
            // shared-expert stage (whose projections the dense schema points at
            // `mlp.*`), so there is no router to read, no expert to fetch and
            // nothing to classify. Every MoE-only binding and stage below is
            // skipped rather than fed a zero-expert placeholder -- the router
            // role resolves to a name that cannot exist for this family, and
            // reading it would fail the layer.
            let denseFFN = cfg.numExperts == 0

            let inNorm = try model.inputNorm(layer: L)
            let postAttn = try model.postAttnNorm(layer: L)
            let sharedProj = sharedExpertProjections[L]
            let nextRouterW: TensorView?
            if !denseFFN, nextLayerPredictionEnabled, L + 1 < cfg.numLayers {
                nextRouterW = try model.router(layer: L + 1)
            } else {
                nextRouterW = nil
            }
            let next2RouterW: TensorView? =
                (!denseFFN && Self.probe2TraceEnabled && L + 2 < cfg.numLayers)
                ? try model.router(layer: L + 2) : nil
            let residencyResources =
                (!denseFFN && decodeExpertExecution == .gpuResidency)
                ? try model.routedExpertResidency(layer: L) : nil
            let perExpertScale: (buffer: any MTLBuffer, offset: Int) =
                (try requireOnesPerExpertScale(), 0)

            let tCb1Start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            // Attention+router split into measured sub-command-buffers
            // (TINYTITAN_KERNEL_STATS): attnCB = input norm + QKV + epilogue
            // (or the linear/gated attention), softmaxCB = the softmax
            // attention pass on full layers, tailCB = O-proj + residual +
            // post-norm + router. Same queue, same order, one wait on the
            // last CB; only the router readback forces the barrier.
            guard var attnCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            // The PLE block rewrites the wide residual before the attention
            // read gate sees it, so it is encoded ahead of the entry.
            try encodePLEDecode(
                commandBuffer: attnCB, layer: L,
                position: position, eps: eps)
            // Full-attention layers of a sparse-attention family run their
            // entry ahead of the rest, because the indexer's key cache and
            // its selection both hang off the block input.
            //
            // That entry can run on its own command buffer, committed before
            // this one, so a PLE block encoded here would rewrite the
            // residual *after* the entry had already read it. The two never
            // coincide in this architecture -- the n-gram layer is a linear
            // one -- and the guard is here so that stays a fact rather than
            // an assumption.
            precondition(
                !(qsaIndexer != nil
                    && cfg.fullAttentionLayerMask[L] == 1
                    && cfg.ple.layerIndices.contains(L)),
                "layer \(L) is both a PLE layer and a sparse-attention "
                    + "layer; the residual entry would be reordered "
                    + "around the n-gram block")
            var keepMask: MTLBuffer?
            if qsaIndexer != nil, cfg.fullAttentionLayerMask[L] == 1 {
                keepMask = try encodeQSAEntryAndSelect(
                    passthrough: &attnCB,
                    hidden: hidden, norm: inNorm, out: normed,
                    layer: L, position: position, eps: eps)
            } else {
                try encodeResidualEntryDecode(
                    commandBuffer: attnCB,
                    hidden: hidden, norm: inNorm,
                    out: normed, sublayer: .attention,
                    layer: L, eps: eps)
                try rotate(&attnCB, role: "glue.entry_attn")
            }
            var softmaxCB: MTLCommandBuffer?
            guard var tailCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }

            try encodeDecodeAttention(
                attnCB: &attnCB, tailCB: tailCB,
                softmaxCB: &softmaxCB,
                layer: L, position: position,
                slot: slot,
                isLinear: isLinear, rmsEps: eps,
                keepMask: keepMask)
            try encodeResidualExitDecode(
                commandBuffer: tailCB,
                hidden: hidden, delta: oOut,
                sublayer: .attention, layer: L)
            try rotate(&tailCB, role: "glue.exit_attn")
            try encodeResidualEntryDecode(
                commandBuffer: tailCB,
                hidden: hidden, norm: postAttn,
                out: routedX, sublayer: .mlp,
                layer: L, eps: eps)
            try rotate(&tailCB, role: "glue.entry_mlp")

            if !denseFFN {
                let routerW = try model.router(layer: L)
                try moe.encodeRouter(
                    commandBuffer: tailCB,
                    weights: routerW.buffer, weightsOffset: Int(routerW.offset),
                    scales: routerW.buffer, scalesOffset: Int(routerW.scaleOffset),
                    biases: routerW.buffer, biasesOffset: Int(routerW.biasOffset),
                    hidden: routedX,
                    effectiveScale: effectiveScaleBuffers[L],
                    perExpertScale: perExpertScale.buffer,
                    perExpertScaleOffset: perExpertScale.offset,
                    outIndices: outIndices, outWeights: outWeights,
                    numExperts: UInt32(cfg.numExperts), d: D, topK: UInt32(cfg.topKExperts))
                try rotate(&tailCB, role: "glue.router")
                if let nextRouterW {
                    // Probe only: score the next router against the current
                    // post-attention normalized residual. The exact router above
                    // remains authoritative; this result never selects experts for
                    // this layer. It drives TINYTITAN_PREFETCH_TRACE and, when
                    // TINYTITAN_PREDICTIVE_PREFETCH is set, the speculative ring.
                    try moe.encodeRouter(
                        commandBuffer: tailCB,
                        weights: nextRouterW.buffer, weightsOffset: Int(nextRouterW.offset),
                        scales: nextRouterW.buffer, scalesOffset: Int(nextRouterW.scaleOffset),
                        biases: nextRouterW.buffer, biasesOffset: Int(nextRouterW.biasOffset),
                        hidden: routedX,
                        effectiveScale: effectiveScaleBuffers[L + 1],
                        perExpertScale: perExpertScale.buffer,
                        perExpertScaleOffset: perExpertScale.offset,
                        outIndices: prefetchPredictionIndices,
                        outWeights: prefetchPredictionWeights,
                        numExperts: UInt32(cfg.numExperts), d: D,
                        topK: UInt32(cfg.topKExperts))
                }
                if let next2RouterW {
                    // Trace-only: layer L+2's router on layer L's residual.
                    try moe.encodeRouter(
                        commandBuffer: tailCB,
                        weights: next2RouterW.buffer, weightsOffset: Int(next2RouterW.offset),
                        scales: next2RouterW.buffer, scalesOffset: Int(next2RouterW.scaleOffset),
                        biases: next2RouterW.buffer, biasesOffset: Int(next2RouterW.biasOffset),
                        hidden: routedX,
                        effectiveScale: effectiveScaleBuffers[L + 2],
                        perExpertScale: perExpertScale.buffer,
                        perExpertScaleOffset: perExpertScale.offset,
                        outIndices: prefetchPrediction2Indices,
                        outWeights: prefetchPrediction2Weights,
                        numExperts: UInt32(cfg.numExperts), d: D,
                        topK: UInt32(cfg.topKExperts))
                }
                if let residencyResources {
                    try moe.encodeResidencyClassification(
                        commandBuffer: tailCB,
                        topKIndices: outIndices,
                        residencyTable: residencyResources.table,
                        hitCount: residencyHitCount,
                        hitPositions: residencyHitPositions,
                        missCount: residencyMissCount,
                        missPositions: residencyMissPositions,
                        missExperts: residencyMissExperts,
                        resolvedSlots: residencyResolvedSlots,
                        resolvedGenerations: residencyResolvedGenerations,
                        topK: UInt32(cfg.topKExperts),
                        numExperts: UInt32(cfg.numExperts))
                }
            }
            attnCB.commit()
            if let attentionCB = softmaxCB {
                attentionCB.commit()
            }
            tailCB.commit()
            // Queued before the wait below, not after: the GPU runs the shared
            // MLP while the CPU blocks on tailCB for the routing.
            let overlapCompletionClock = runnerStatsEnabled ? CommandCompletionClock() : nil
            let sharedCB = try encodeAndCommitSharedExpert(
                layer: L,
                completionClock: overlapCompletionClock)
            let tWait = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            try waitForCompletion(tailCB)
            // After the tail wait every attention-side buffer is settled and
            // nothing overwrites them until the next layer, so the dumps cost
            // no extra synchronization.
            if activationDumpActive(position: position) && L <= dumpLayerLimit {
                if let ple = pleBlock, cfg.ple.layerIndices.contains(L) {
                    let wide = residualWidth
                    dumpActivation("ple_key", ple.keyBuf, count: wide, position: position)
                    dumpActivation(
                        "ple_value", ple.valueBuf, count: cfg.hiddenSize, position: position)
                    dumpActivation("ple_key_normed", ple.keyNormed, count: wide, position: position)
                    dumpActivation("ple_query", ple.queryNormed, count: wide, position: position)
                    dumpActivation(
                        "ple_score", ple.scoreBuf,
                        count: cfg.hyperConnections.count, position: position)
                    dumpActivation(
                        "ple_gate", ple.gateBuf,
                        count: cfg.hyperConnections.count, position: position)
                    dumpActivation("ple_gated", ple.gatedBuf, count: wide, position: position)
                    dumpActivation("ple_conv", ple.convOut, count: wide, position: position)
                }
                dumpActivationPrivate(
                    "L\(L)_attn_in", normed, count: cfg.hiddenSize, position: position)
                dumpActivation("L\(L)_attn_out", oOut, count: cfg.hiddenSize, position: position)
                dumpActivation("L\(L)_mlp_in", routedX, count: cfg.hiddenSize, position: position)
                dumpActivation(
                    "L\(L)_hidden_post_attn", hidden,
                    count: residualWidth, position: position)
            }
            recordKernelGPU(role: "attn_norm_qkv", attnCB)
            // Split-mode kernels were committed ahead of attnCB on the same
            // queue, so they completed before the tail wait above.
            for (role, cb) in splitTimedBuffers { recordKernelGPU(role: role, cb) }
            splitTimedBuffers.removeAll(keepingCapacity: true)
            if let attentionCB = softmaxCB {
                recordKernelGPU(role: "attn_softmax", attentionCB)
            }
            recordKernelGPU(role: "attn_tail_router", tailCB)
            let waitNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tWait
            totalWaitNanos &+= waitNanos
            var prevRoutedUs: Double = 0
            if let pending = pendingRoutedCommand {
                prevRoutedUs = (pending.cb.gpuEndTime - pending.cb.gpuStartTime) * 1_000_000
                try finishPendingRoutedCommand(pending, waitIfNeeded: false)
                pendingRoutedCommand = nil
            }
            totalCb1Nanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tCb1Start - waitNanos
            let predictedNextLayer: [Int]
            var predictedNextLayerWeights: [Float] = []
            if nextLayerPredictionEnabled, L + 1 < cfg.numLayers {
                let ptr = prefetchPredictionIndices.contents().bindMemory(
                    to: UInt32.self, capacity: cfg.topKExperts)
                predictedNextLayer = (0..<cfg.topKExperts).map {
                    min(Int(ptr[$0]), cfg.numExperts - 1)
                }
                if prefetchTraceFD >= 0 {
                    // The probe's routing weights, for the trace only: the
                    // prefetch gate that used them (TINYTITAN_PREFETCH_MIN_MARGIN)
                    // measured -4.2% and is gone. Width follows the buffer:
                    // fp32 at 4 bytes per entry, fp16 otherwise.
                    let k = cfg.topKExperts
                    if prefetchPredictionWeights.length >= k * MemoryLayout<Float>.stride {
                        let w = prefetchPredictionWeights.contents().bindMemory(
                            to: Float.self, capacity: k)
                        predictedNextLayerWeights = (0..<k).map { w[$0] }
                    } else {
                        let w = prefetchPredictionWeights.contents().bindMemory(
                            to: Float16.self, capacity: k)
                        predictedNextLayerWeights = (0..<k).map { Float(w[$0]) }
                    }
                }
            } else {
                predictedNextLayer = []
            }
            let predictedNext2Layer: [Int]
            if Self.probe2TraceEnabled, L + 2 < cfg.numLayers {
                let ptr = prefetchPrediction2Indices.contents().bindMemory(
                    to: UInt32.self, capacity: cfg.topKExperts)
                predictedNext2Layer = (0..<cfg.topKExperts).map {
                    min(Int(ptr[$0]), cfg.numExperts - 1)
                }
            } else {
                predictedNext2Layer = []
            }
            self.lastPredictedNext2Layer = predictedNext2Layer

            // CPU readback to fetch routed-expert blobs from disk. The expert
            // id list is reused host scratch (R16); the runner is single-flight
            // per generation, so it never aliases concurrent decode work.
            if denseFFN {
                // The shared-expert stage *is* this family's FFN: its output is
                // the layer's MLP contribution, added to the residual exactly
                // as the routed stage's phase-2 reduce adds a mixture's. The
                // buffer is submitted after `sharedCB` on the same queue, so the
                // read of `h1Buf` is ordered behind its write.
                guard let denseCB = ctx.queue.makeCommandBuffer() else {
                    throw ModelError.residentBufferWrapFailed
                }
                try encodeResidualExitDecode(
                    commandBuffer: denseCB,
                    hidden: hidden, delta: h1Buf,
                    sublayer: .mlp, layer: L)
                recordKernelGPU(role: "dense_ffn_exit", denseCB)
                denseCB.commit()
            } else {
                try await encodeDecodeRoutedMoE(
                    layer: L, position: position, sharedProj: sharedProj,
                    attnCB: attnCB, tailCB: tailCB,
                    sharedCB: sharedCB,
                    overlapCompletionClock: overlapCompletionClock,
                    pending: &pendingRoutedCommand,
                    bodyStart: tBodyStart, cb1Start: tCb1Start,
                    waitMark: tWait, waitNanos: waitNanos,
                    previousRoutedMicros: prevRoutedUs,
                    predictedNextLayer: predictedNextLayer,
                    predictedNextLayerWeights: predictedNextLayerWeights)
            }
        }
        if let pending = pendingRoutedCommand {
            try finishPendingRoutedCommand(pending, waitIfNeeded: true)
            pendingRoutedCommand = nil
        }

        // The fused head skips the vocab buffer and leaves a greedy token in
        // greedyTokenBuf; the logits path writes the complete vector.
        let fNorm = try model.finalNorm()
        let lm = try model.lmHead()
        let gFinalNorm: (MTLCommandBuffer) throws -> Void = { cb in
            if let hc = self.hyperConnection {
                // The stack ends by collapsing the streams through the
                // model-level mixer: the same gated read a sublayer uses, with
                // no inject, and its hc_norm serving as the final norm.
                try hc.encodeRead(
                    commandBuffer: cb,
                    streamsBuffer: self.hidden,
                    hcNorm: fNorm.buffer,
                    hcNormOffset: Int(fNorm.offset),
                    down: self.gateWeightsPublic(try self.model.hcMixerDown()),
                    up: self.gateWeightsPublic(try self.model.hcMixerUp()),
                    blockInput: self.normed, eps: eps)
            } else {
                try self.rms.encodeBF16W(
                    commandBuffer: cb, x: self.hidden,
                    weight: fNorm.buffer, weightOffset: Int(fNorm.offset),
                    out: self.normed, d: D, eps: eps)
            }
        }
        let gLmHead: (MTLCommandBuffer) throws -> Void = { cb in
            try self.encodeHeadGEMV(
                commandBuffer: cb,
                weights: lm.buffer, weightsOffset: Int(lm.offset),
                scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                x: self.normed, y: logits, m: UInt32(self.cfg.vocabSize), n: D)
        }
        let gFusionHead: (MTLCommandBuffer) throws -> Void = { cb in
            try self.fusionHead.encodeGreedyDecode(
                commandBuffer: cb,
                hidden: self.hidden,
                normWeight: fNorm.buffer, normOffset: Int(fNorm.offset),
                weights: lm.buffer, weightsOffset: Int(lm.offset),
                scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                outToken: self.greedyTokenBuf,
                d: D, vocab: UInt32(self.cfg.vocabSize),
                rmsEps: eps)
        }
        if activationDumpActive(position: position) {
            if let pending = pendingRoutedCommand {
                try finishPendingRoutedCommand(pending, waitIfNeeded: true)
                pendingRoutedCommand = nil
            }
            dumpActivation("stack_out", hidden, count: residualWidth, position: position)
        }
        if emitHead {
            let useFusedHeadForThisToken = useFusedGreedyHead && outputMode == .greedyIfAvailable
            let tHead = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            if useFusedHeadForThisToken {
                if let headCB = try runSync(gFusionHead) {
                    recordKernelGPU(role: "head_fused", headCB)
                }
                totalHeadFusedNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tHead
                lastGreedyToken = greedyTokenBuf.contents().load(as: UInt32.self)
            } else {
                guard
                    let headCB = try runSync({ cb in
                        try gFinalNorm(cb)
                        try gLmHead(cb)
                    })
                else {
                    throw ModelError.residentBufferWrapFailed
                }
                recordKernelGPU(role: "head_logits", headCB)
                if activationDumpActive(position: position) {
                    dumpActivation(
                        "mixer_out", normed, count: cfg.hiddenSize,
                        position: position)
                    dumpActivation(
                        "logits", logits, count: cfg.vocabSize,
                        position: position)
                }
                // The last prompt position's logits, however prefill produced
                // them: the one place a batched path and the sequential
                // oracle can be compared as numbers rather than as text.
                if activationDumpDirectory != nil, emitHead {
                    dumpActivation(
                        "prefill_logits", logits,
                        count: cfg.vocabSize, position: 0)
                }
                totalHeadNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tHead
            }
        }

        kv?.advance(slot: slot, by: 1)
    }

}
