import Foundation
import Metal

// Phase one of RealForwardRunner's initializer: configuration, kernels and the
// optional subsystems. Moved verbatim out of the class body (2026-09-28) under the
// 500-line-per-file rule; the statements and their order are unchanged, and the
// only edits are `self.` -> `bp.` plus the optional unwraps the staging type needs.

extension RealForwardRunner {
    /// lint:allow-long a straight-line construction sequence: the staged values are
    /// assigned in dependency order, and moving any group out needs the locals it
    /// reads hoisted into the builder first.
    static func buildCore(
        _ bp: Builder, model: Model, context: MetalContext, maxContext: Int, slots: Int,
        runtimeConfiguration: RuntimeConfiguration, enableSpeculativeGDN: Bool
    ) throws -> ModelProfile {
        bp.model = model
        bp.ctx = context
        let cfg = model.config
        bp.cfg = cfg
        bp.maxContext = maxContext
        precondition(
            slots > 0 && slots <= KVCacheManager.maximumSlots,
            "slots must be between 1 and \(KVCacheManager.maximumSlots)")
        bp.slots = slots
        try runtimeConfiguration.validate(maxContext: maxContext)
        let yarnParameters: YaRNRoPEParameters?
        if runtimeConfiguration.ropeScalingMode == .yarn {
            guard model.config.ropeNeoxSubdim else {
                throw RuntimeConfigurationError.yaRNUnsupportedArchitecture
            }
            yarnParameters = YaRNRoPEParameters(
                headDim: model.config.fullHeadDim,
                partialRotaryFactor: model.config.partialRotaryFactor,
                theta: model.config.fullRopeTheta,
                targetContextTokens: runtimeConfiguration.yarnContextTokens)
        } else {
            yarnParameters = nil
        }
        // The fused greedy head folds a plain RMSNorm into the vocabulary
        // GEMV. A hyper-connection model does not end in an RMSNorm: it ends
        // in the gated mixer that collapses the residual streams, so the
        // fused path would normalize the wide residual and read stream 0 as
        // if it were the whole hidden state. Correct output beats one fused
        // dispatch; the family takes the two-step head.
        bp.useFusedGreedyHead =
            runtimeConfiguration.headPath == .fusedRows
            && !cfg.hyperConnections.enabled
            && model.lmHeadWeightBits == 4
            && model.attentionWeightBits == 4
        bp.prefillAttentionPath = runtimeConfiguration.prefillAttentionPath
        // The family's measured optimum is the default; an explicit
        // TINYTITAN_PREDICTIVE_PREFETCH still wins either way, so a probe can turn
        // it on where it ships off and off where it ships on.
        let profile = ModelProfile.resolve(
            modelID: model.modelID, family: cfg.family,
            weightBits: model.routedExpertWeightBits)
        bp.profile = profile
        bp.decodeExpertExecution = runtimeConfiguration.decodeExpertExecution
        let expertIOSynchronization = runtimeConfiguration.expertIOSynchronization
        bp.expertIOSynchronization = expertIOSynchronization
        bp.expertIOSubmission = runtimeConfiguration.expertIOSubmission
        bp.expertIOBackend = try ExpertIOBackend.environmentValue()
        if ProcessInfo.processInfo.environment["TINYTITAN_RUNNER_STATS"] != nil
            || ProcessInfo.processInfo.environment["TINYTITAN_KERNEL_STATS"] != nil
        {
            FileHandle.standardError.write(Data(("TinyTitan \(profile.summary)\n").utf8))
        }
        // A model with no routed experts has nothing to prefetch -- the ring
        // reads expert blobs, and `topKExperts`/`prefetchDepth` both describe a
        // mixture. `(1...0)` is not a range, so the guard below trapped the
        // runner's construction for a dense install instead of refusing it:
        // a value from a manifest reaching a range that requires a mixture.
        let denseFFN = cfg.numExperts == 0
        let rawPrefetchEnabled = !denseFFN && profile.prefetchDepth > 0
        // One read deep, not four. The ring depth is a bandwidth decision, not
        // a coverage one: the SSD is saturated while it reads, so a speculative
        // read that misses its layer has stolen service from a demand read that
        // did not. One read has the whole inter-layer window to itself and
        // lands in time; deeper rings contend with the demand traffic and
        // arrive too late to be adopted, which shows up as a *lower* hit rate.
        // Measured on qwen38 4-bit, interleaved against prefetch off:
        // M=1 +12.2% (hit 78.7%), M=2 +6.0% (77.9%), M=4 -9.8% (77.1%).
        // The depth is the profile's, one everywhere it ships: an env override
        // for it was measured negative at every value but 1 and has been
        // removed, so the ring is one read a layer by construction.
        let ringDepth = max(1, profile.prefetchDepth)
        if !denseFFN {
            guard (1...cfg.topKExperts).contains(ringDepth) else {
                throw ModelError.internalInconsistency(
                    detail: "profile prefetch depth \(ringDepth) must be 1...\(cfg.topKExperts)")
            }
        }
        bp.predictivePrefetch =
            rawPrefetchEnabled
            ? try ExpertPrefetchRing(
                device: context.device,
                expertStride: model.routedExpertByteStride(layer: 0),
                slotCount: ringDepth)
            : nil
        // Track A: the ANE prefill sidecar, opt-in. Only the qwen36 target
        // family qualifies (the one-layer MTP draft has no exported sidecar
        // and must stay silently on the GPU); with the switch on and the
        // sidecar missing, construction fails closed with the export command.
        // Track A: ANE prefill, on by default for the one family that has an
        // exported sidecar. Measured end to end on an idle machine with a
        // 9,316-token prompt: 313.6 s against 99.9 s at 4-bit, a 3.14x
        // request, and 1.91x at 8-bit. It costs about 0.023 s per generated
        // token in decode, so it does not break even against the ~215 s
        // prefill saving until roughly 9,200 generated tokens.
        //
        // It is not bit-identical to the GPU path: the ANE reduces attention
        // in fp16 in a different order, so a prompt long enough to reach the
        // sidecar (>= one full 4,096-token chunk) can decode to different --
        // equally valid -- greedy text. Shorter prompts never reach it and
        // are unaffected, which is why the golden baselines still hold.
        // Any family may carry a sidecar now: the graph is built from the
        // model's own geometry and `ANEPrefillAttention.init` refuses one that
        // does not match this model, so the gate is the sidecar itself rather
        // than a family name. A family that *selects* keys rather than changing
        // the arithmetic — Qwen 3.8's QSA indexer — is served by folding that
        // selection into the additive mask the graph already takes, which the
        // sidecar records as `selectionFolded`. A family the exporter does not
        // build for (the one-layer MTP draft) simply has none — and 3.8, whose
        // fold is wired and verified, is measured *slower* on the ANE
        // (benchmark/ane-prefill/README.md), so no sidecar is installed for it
        // and the chunk stays here on the GPU.
        let wantsANE = try RuntimePrefillANE.environmentValue() == .on
        if wantsANE {
            do {
                bp.anePrefill = try ANEPrefillAttention(
                    modelDirectory: model.directoryURL,
                    device: context.device,
                    hiddenSize: model.config.hiddenSize,
                    kvDim: model.config.numFullKVHeads * model.config.fullHeadDim,
                    weightsSha256: model.weightsDigestFromManifest,
                    family: model.config.family,
                    fullAttentionLayerMask: model.config.fullAttentionLayerMask,
                    sparseIndexer: model.config.sparseIndexer,
                    configChunkTokens: runtimeConfiguration.prefillChunkTokens)
            } catch {
                // An explicit request must fail loudly with the export
                // command; the default must degrade to the GPU, because a
                // model without a sidecar is the normal case.
                guard !RuntimePrefillANE.wasRequestedExplicitly() else { throw error }
                bp.anePrefill = .some(nil)
            }
        } else {
            bp.anePrefill = .some(nil)
        }
        let useFP16Ring = runtimeConfiguration.fp16RingEnabled
        bp.rdadvisePolicyMode = runtimeConfiguration.rdadvisePolicy
        bp.rdadviseAdaptiveState = RDAdviceAdaptivePolicyState(
            config: RDAdviceAdaptivePolicyConfig(
                missCap: Self.rdadviseAdaptiveMissCap,
                byteCap: Self.rdadviseAdaptiveByteCap,
                slowCallNanos: Self.rdadviseAdaptiveSlowCallNanos))
        bp.rdadviseEnabled = runtimeConfiguration.rdadviseEnabled
        bp.kv = try KVCacheManager(
            device: context.device,
            config: cfg,
            maxContext: maxContext,
            slots: slots,
            fp16RingEnabled: useFP16Ring,
            precision: runtimeConfiguration.kvCachePrecision,
            slidingWindow: cfg.slidingWindow,
            maxPrefillChunkTokens: runtimeConfiguration.prefillChunkTokens)

        let silu = cfg.hiddenActivation == "silu"
        bp.embedInt4 = try EmbedLookupInt4(context: context)
        bp.affineEmbed =
            model.embeddingWeightBits == 4
            ? nil
            : try AffineQuantEmbeddingLookup(
                context: context,
                weightBits: model.embeddingWeightBits)
        bp.rms = try RMSNorm(context: context)
        bp.int4 = try DequantInt4GEMV(
            context: context,
            additionalShapes: cfg.decodeInt4GEMVShapes)
        // One affine dispatcher per width the model's *roles* actually use.
        // Most families need one (their roles share the attention slot); the
        // dense Qwen 3.5 installs need two at once, because their full-attention
        // `k_proj`/`v_proj` are 8-bit while `q_proj`/`o_proj` are 4-bit. Asking
        // the roles rather than the slot is also what keeps an 8-bit tensor
        // from being read as nibbles.
        var affineByWidth: [Int: AffineQuantGEMV] = [:]
        for width in Set([
            model.attentionWeightBits, model.qoProjectionWeightBits,
            model.kvProjectionWeightBits,
            model.gdnProjectionWeightBits,
            // The three families that read the attention slot
            // until a per-tensor override promotes them.
            model.hyperConnectionWeightBits, model.pleKeyWeightBits,
            model.qsaIndexerWeightBits,
        ]).sorted() where width != 4 {
            affineByWidth[width] = try AffineQuantGEMV(context: context, weightBits: width)
        }
        bp.affineByWidth = affineByWidth
        bp.affine = affineByWidth[model.attentionWeightBits]
        bp.affineKV = affineByWidth[model.kvProjectionWeightBits]
        // The vocabulary head carries its own bit width. It matched the
        // attention slot in every earlier family, so the head simply reused
        // the attention GEMV -- which reads an 8-bit head as packed 4-bit the
        // moment a model quantizes the two differently, as this one does
        // (4-bit attention, 8-bit embedding and head). The result is a full
        // logit vector of confident nonsense, so the width is selected here
        // rather than inherited.
        bp.affineHead =
            model.lmHeadWeightBits == 4
            ? nil
            : try AffineQuantGEMV(
                context: context,
                weightBits: model.lmHeadWeightBits)
        bp.attention = try Attention(context: context)
        bp.attention?.simdPartialOverride = profile.attentionSimdPartial
        bp.kvQuantizer =
            runtimeConfiguration.kvCachePrecision.isQuantized
            ? try KVCacheQuantizer(context: context) : nil
        bp.shared = try SharedExpertRuntime(
            context: context,
            weightBits: model.ffnWeightBits,
            siluActivation: silu)
        if ProcessInfo.processInfo.environment["TINYTITAN_DEBUG_PROMOTION"] != nil {
            FileHandle.standardError.write(
                Data(
                    ("promotion: routerBits=\(model.effectiveRouterWeightBits) "
                        + "slotRouterBits=\(model.routerWeightBits) "
                        + "gdnAB_bf16=\(model.gdnABIsBF16)\n").utf8))
        }
        bp.moe = try MoE(
            context: context,
            routerTopKSimd: profile.routerTopKSimd,
            siluActivation: silu,
            routedWeightBits: model.routedExpertWeightBits,
            routerWeightBits: model.effectiveRouterWeightBits,
            eventGatedIO: expertIOSynchronization == .event,
            specializedD: UInt32(cfg.hiddenSize),
            specializedF: UInt32(cfg.moeIntermediateSize),
            specializedNumExperts: UInt32(cfg.numExperts),
            topKExperts: cfg.topKExperts)
        bp.fusionHead = try LMHeadChainInt4(
            context: context,
            maxD: cfg.hiddenSize,
            maxVocab: cfg.vocabSize)
        bp.fusedQKVGEMV = try FusedQKVGEMV(context: context)
        bp.fusedQKVEpilogue = try FusedQKVEpilogue(context: context)
        bp.prefillEmbed = try PrefillEmbedLookupInt4(
            context: context,
            weightBits: model.embeddingWeightBits)
        bp.prefillRMS = try PrefillRMSNorm(context: context)
        bp.prefillQMM = try PrefillInt4QMM(
            context: context,
            weightBits: model.attentionWeightBits)
        bp.prefillMPPAffineInt4 = MPPPrefillInt4QMM(
            context: context,
            weightBits: model.attentionWeightBits)
        bp.prefillQKVEpilogue = try PrefillQKVEpilogue(
            context: context,
            yarn: yarnParameters)
        bp.prefillAttention = try PrefillAttention(context: context)
        bp.prefillRouter = try PrefillRouter(
            context: context,
            weightBits: model.effectiveRouterWeightBits)
        bp.prefillSharedExpert = try PrefillSharedExpert(
            context: context,
            weightBits: model.ffnWeightBits,
            siluActivation: silu)
        bp.prefillGroupedMoE = try PrefillGroupedRoutedMoE(
            context: context,
            siluActivation: silu,
            weightBits: model.routedExpertWeightBits)
        bp.prefillMoE = try PrefillMoE(context: context)
        bp.prefillFinalRowHead = try PrefillFinalRowHeadInt4(
            context: context,
            maxD: cfg.hiddenSize,
            weightBits: model.lmHeadWeightBits)

        // Qwen 3.6 kernels, keyed off the data flags so architectures that
        // never dispatch them pay no PSO compile cost.
        let needsElementwise =
            cfg.attnOutputGate
            || cfg.sharedExpertGated
            || cfg.hasLinearAttentionLayers
        bp.elementwise = .some(needsElementwise ? try Elementwise(context: context) : nil)
        bp.activationDumpDirectory = .some(
            ProcessInfo.processInfo.environment["TINYTITAN_ACT_DUMP"]
                .map { URL(fileURLWithPath: $0) })
        // Both gated blocks keep a whole prefill chunk resident: the write
        // gate consumes what its matching read produced, and the block runs
        // in between, so the rows cannot be streamed one at a time.
        let gateRows = max(1, runtimeConfiguration.prefillChunkTokens)
        bp.hyperConnection =
            cfg.hyperConnections.enabled
            ? try HyperConnection(
                context: context,
                dim: cfg.hiddenSize,
                streams: cfg.hyperConnections.count,
                lowRank: cfg.hyperConnections.lowRank,
                maxRows: gateRows,
                weightBits: model.hyperConnectionWeightBits)
            : nil
        bp.qsaIndexer =
            cfg.sparseIndexer.enabled
            ? try QSAIndexer(
                context: context,
                config: cfg.sparseIndexer,
                budget: Self.qsaBudget(cfg.sparseIndexer),
                ropeTheta: Float(cfg.fullRopeTheta),
                capacity: maxContext,
                weightBits: model.qsaIndexerWeightBits)
            : nil
        if cfg.ple.enabled {
            let constants = try PLEConstants.load(
                directoryURL: model.directoryURL)
            // Before anything derives buffer sizes or row addressing from the
            // sidecar. Its values are geometry, and `PLEBlock`'s embedding
            // buffer is sized from `cfg.ple.embedDim` while the gather width
            // comes from this file -- a disagreement writes past that buffer
            // (host heap, not a GPU fault) or feeds the block wrong-width rows,
            // silently. `PLEHash`'s own checks are preconditions, so this has
            // to run first to make a corrupt sidecar a report rather than a trap.
            try constants.validate(
                embedDim: cfg.ple.embedDim,
                ngramSize: cfg.ple.ngramSize,
                headsPerNgram: cfg.ple.headsPerNgram)
            bp.pleHash = constants.makeHash()
            bp.ngramTable = try NgramTableReader(
                path: model.directoryURL.appendingPathComponent(
                    Qwen38FlashTensors.ngramTableFile
                ).path,
                rowDim: constants.pleHeadDim,
                rowCount: constants.tableRowCount)
            bp.pleBlock = try PLEBlock(
                context: context,
                dim: cfg.hiddenSize,
                streams: cfg.hyperConnections.count,
                embedDim: cfg.ple.embedDim,
                kernelSize: cfg.ple.convKernelSize,
                // The dilation is the n-gram size, not a constant of its own.
                dilation: cfg.ple.ngramSize,
                maxRows: gateRows,
                weightBits: model.pleKeyWeightBits)
        } else {
            bp.pleHash = .some(nil)
            bp.ngramTable = .some(nil)
            bp.pleBlock = .some(nil)
        }
        if cfg.hasLinearAttentionLayers {
            bp.gdn = try GDN(
                context: context, config: cfg.linearAttention,
                specializedHiddenSize: cfg.hiddenSize,
                abBF16: model.gdnABIsBF16)
            bp.gdnState = try GDNStateManager(
                device: context.device,
                config: cfg,
                slots: slots,
                enableSpeculativeCheckpoint: enableSpeculativeGDN)
        } else {
            bp.gdn = .some(nil)
            bp.gdnState = .some(nil)
        }
        bp.rope =
            cfg.ropeNeoxSubdim
            ? try RoPE(context: context, yarn: yarnParameters) : nil
        return profile
    }
}
