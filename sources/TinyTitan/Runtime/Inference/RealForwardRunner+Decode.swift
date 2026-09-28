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

        pendingRoutedCommand = try await decodeLayerLoop(
            position: position, slot: slot, into: logits, emitHead: emitHead,
            outputMode: outputMode, D: D, eps: eps,
            pendingRoutedCommand: pendingRoutedCommand)
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
