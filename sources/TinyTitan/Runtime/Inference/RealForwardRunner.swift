import Foundation
import Metal

/// Compatible Qwen3.5-MoE real-forward decode pass.
///
/// Composes the production kernels against the `.gturbo` model:
///
///   embed_lookup_int4(token) * sqrt(H)
///   for L in 0..<40:
///     a = rmsnorm_bf16w(h, input_layernorm)
///     Q = q_proj(a)    K = k_proj(a)    V = v_proj(a)
///     per-head q/k_norm (bf16w)
///     NeoX RoPE on Q + K (full-attention layers; linear layers use
///     Gated-DeltaNet recurrent state instead of K/V slots)
///     write K and V into the cache slots
///     attn = attention(scale, full causal)
///     attn = o_proj(attn)
///     h = h + rmsnorm_bf16w(attn, post_attention_layernorm)
///     h1 = rmsnorm_bf16w(h, pre_feedforward_layernorm)
///     h1 = SharedExpertInt8(h1)  // sigmoid-gated by shared_expert_gate
///     xr = rmsnorm_no_scale(h)
///     idx, w = router_topk(xr, effective_scale[L], per_expert_scale[L])
///     h2 = moe_fused_ffn_streamed_routed(h2, residual=h1, routedBlobs=fetch(idx), w)
///     h = h + h2
///     h = h * layer_scalar[L]
///   logits = DequantInt4GEMV(rmsnorm_bf16w(h, model.norm), lm_head^T)
///   // final softmax happens in the Sampler.
///
/// Direct against `Model`; this is the only production decode forward path.
/// unchecked-invariant: exclusively owned by one caller for its lifetime and
/// never shared. In the server it is a `private let` on the `ServerModelSession`
/// actor, so every entry point is already actor-isolated; the CLI and each
/// server model session drive one runner from a single task. Its ~19 mutable
/// properties are decode cursors and scratch handles with no internal locking,
/// so two concurrent callers would corrupt them -- the ownership is the whole
/// safety argument, not an implementation detail.
public final class RealForwardRunner: ChunkedPrefillRunner, ContextWindowReporting,
    ContinuableLogitProducer, @unchecked Sendable
{
    struct LayerSharedExpertProjections {
        let gate: SharedExpertInt8Proj
        let up: SharedExpertInt8Proj
        let down: SharedExpertInt8Proj
        /// Qwen3.5-MoE [1, hidden] scalar gate on the shared expert branch.
        let scalarGate: TensorView?
    }

    let model: Model
    let ctx: MetalContext
    let kv: KVCacheManager?
    let cfg: ArchConfig
    /// How many independent sequences this runner's KV and GDN stores hold.
    /// One unless the server batches; `produce(…slot:)` selects the sequence.
    public let slots: Int
    /// Serializes forward steps so concurrent slots never share the runner's
    /// scratch. Held around each `produce`/`prefillChunked`, not across a whole
    /// generation.
    let forwardStepGate = ForwardStepGate()

    // Kernels
    let embedInt4: EmbedLookupInt4
    let affineEmbed: AffineQuantEmbeddingLookup?
    let rms: RMSNorm
    let int4: DequantInt4GEMV
    let affine: AffineQuantGEMV?
    /// The k/v role's dispatcher when its width differs from the attention slot's.
    let affineKV: AffineQuantGEMV?
    /// Every affine dispatcher this model's roles need, by width. `int4` covers
    /// 4, which needs no entry here.
    let affineByWidth: [Int: AffineQuantGEMV]
    /// Vocabulary head GEMV, keyed off `lmHeadWeightBits` rather than the
    /// attention slot. Nil when the head is 4-bit.
    let affineHead: AffineQuantGEMV?
    let attention: Attention
    let kvQuantizer: KVCacheQuantizer?
    let shared: SharedExpertRuntime
    let moe: MoE
    let fusionHead: LMHeadChainInt4
    let fusedQKVGEMV: FusedQKVGEMV
    let fusedQKVEpilogue: FusedQKVEpilogue

    // Qwen 3.6 kernels. Nil on architectures that never dispatch them.
    let elementwise: Elementwise?

    /// The Gated Residual, for families that carry one. Owns its own scratch,
    /// so a family without hyper-connections allocates nothing.
    /// Set by `TINYTITAN_ACT_DUMP`; nil disables every dump call site.
    let activationDumpDirectory: URL?
    /// Dumps recorded mid-layer, performed once the layer's work is awaited.
    /// The runner is single-flight per generation, so this needs no lock; it is
    /// only ever touched from the encoding task.
    var pendingDumps: [PendingActivationDump] = []
    let hyperConnection: HyperConnection?
    /// The n-gram (PLE) block, its row addressing, and the table it reads.
    /// All three or none: a family without PLE layers leaves them nil.
    /// The sparse-attention indexer, for families that have one. Nil leaves
    /// the runtime on dense attention, which is exact only inside
    /// `QSAExactness`'s window.
    let qsaIndexer: QSAIndexer?
    let pleBlock: PLEBlock?
    let pleHash: PLEHash?
    let ngramTable: NgramTableReader?
    /// The current token and its predecessors, nearest first, as `PLEHash`
    /// wants them. Only `ngramSize` entries are ever consulted.
    var pleContext: [Int32] = []
    let gdn: GDN?
    let gdnState: GDNStateManager?
    let rope: RoPE?
    let int8ScalarGate: DequantInt8GEMV?
    /// Used instead of `int8ScalarGate` when the scalar gate was promoted to
    /// the checkpoint's bf16. Kept as a separate member rather than folded
    /// into SlotGEMV because the quantized one is built with decode-specific
    /// shapes that the generic wrapper does not carry.
    let bf16ScalarGate: BF16GEMV?
    /// Used for any resident projection whose tensor was promoted to the
    /// checkpoint's bf16. Held unconditionally: promotion is a property of the
    /// install, not of the family, and the pipeline costs nothing unused.
    let bf16Projection: BF16GEMV

    // Prefill kernels. These are initialized once per runner so the chunk path
    // cannot accidentally rebuild PSOs inside a per-layer loop.
    let prefillEmbed: PrefillEmbedLookupInt4
    let prefillRMS: PrefillRMSNorm
    let prefillQMM: PrefillInt4QMM
    let prefillMPPAffineInt4: MPPPrefillInt4QMM?
    let prefillQKVEpilogue: PrefillQKVEpilogue
    let prefillAttention: PrefillAttention
    let prefillRouter: PrefillRouter
    let prefillSharedExpert: PrefillSharedExpert
    let prefillGroupedMoE: PrefillGroupedRoutedMoE
    let prefillMoE: PrefillMoE
    let prefillFinalRowHead: PrefillFinalRowHeadInt4

    // Scratch — preallocated per spec'd D / F / vocab.
    let hidden: MTLBuffer  // [D] FP16
    let normed: MTLBuffer  // [D] FP16
    let attnOut: MTLBuffer  // [N_HEADS * head_dim] FP16
    let qScratch: MTLBuffer  // [N_HEADS * head_dim] FP16
    let kStage: MTLBuffer  // [max KV heads * head_dim] FP16, current token
    let vStage: MTLBuffer  // [max KV heads * head_dim] FP16, current token
    let oOut: MTLBuffer  // [D] FP16
    let h1Buf: MTLBuffer  // [D] FP16 (dense MLP output)
    let h2Buf: MTLBuffer  // [D] FP16 (routed output)
    let routedX: MTLBuffer  // [D] FP16 (pre_feedforward_layernorm_2 output)
    let denseX: MTLBuffer  // [D] FP16 (pre_feedforward_layernorm output)
    let denseScratchGate: MTLBuffer  // [F=2112] FP16
    let denseScratchUp: MTLBuffer  // [F=2112] FP16
    let denseScratchAct: MTLBuffer  // [F=2112] FP16
    let routerInput: MTLBuffer  // [D] FP16 (rmsnorm_no_scale(h))
    let zeroResidual: MTLBuffer  // [D] FP16 zeros — for routed branch base
    let outIndices: MTLBuffer  // [topK] UInt32
    let outWeights: MTLBuffer  // [topK] FP16
    /// Trace-only next-layer router result. It is never read by inference.
    let prefetchPredictionIndices: MTLBuffer
    let prefetchPredictionWeights: MTLBuffer
    /// TINYTITAN_PROBE2_TRACE=1: a second probe scores layer L+2's router on the
    /// current residual, trace-only, to measure whether a two-layer-ahead
    /// prediction is accurate enough to widen the prefetch window.
    static let probe2TraceEnabled =
        ProcessInfo.processInfo.environment["TINYTITAN_PROBE2_TRACE"] == "1"
    let prefetchPrediction2Indices: MTLBuffer
    let prefetchPrediction2Weights: MTLBuffer
    var lastPredictedNext2Layer: [Int] = []
    // Persistent MoE scratch, allocated once; about 56 KiB at production shape.
    let moeActs: MTLBuffer  // [topK * FmoE] FP16
    /// Width-2 MTP verify scratch (B2 pair schedule): per-row activation and
    /// output buffers plus two persistent routed argument buffers, created on
    /// first verify. Per-row buffers are deliberately *separate allocations*,
    /// not offsets into one: Metal hazard tracking is whole-buffer, so a
    /// shared acts buffer would falsely serialize row 1's phase 1 behind
    /// row 0's phase 2 and cost real GPU concurrency. The rewrite-per-layer
    /// hazard on the argument buffers is safe because the pair schedule waits
    /// on each layer's routed command before the next layer re-encodes them.
    var verifyPairActs: [MTLBuffer] = []
    var verifyPairY: [MTLBuffer] = []
    var verifyPairArgBuffers: [MTLBuffer] = []
    let moeHitActiveSlots: MTLBuffer  // [topK] UInt32
    let moeMissActiveSlots: MTLBuffer  // [topK] UInt32
    let residencyHitCount: MTLBuffer
    let residencyHitPositions: MTLBuffer
    let residencyMissCount: MTLBuffer
    let residencyMissPositions: MTLBuffer
    let residencyMissExperts: MTLBuffer
    let residencyResolvedSlots: MTLBuffer
    let residencyResolvedGenerations: MTLBuffer
    let greedyTokenBuf: MTLBuffer  // 4 B UInt32 fused-head output
    let verificationHidden: MTLBuffer  // [2, D] FP16 shared readback
    let verificationLogits: MTLBuffer  // [2, vocab] FP16 shared readback
    // Qwen 3.6 decode scratch (nil on architectures that never use it).
    let qPackedScratch: MTLBuffer?  // [2 * N_HEADS * head_dim] packed [q ; gate]
    let attnGateScratch: MTLBuffer?  // [N_HEADS * head_dim]
    let gdnQKVRaw: MTLBuffer?  // [qkvDim] raw in_proj_qkv output
    let gdnConvOut: MTLBuffer?  // [qkvDim] conv + SiLU output
    let gdnZ: MTLBuffer?  // [valueDim]
    let gdnA: MTLBuffer?  // [numVHeads]
    let gdnB: MTLBuffer?  // [numVHeads]
    let gdnY: MTLBuffer?  // [valueDim] delta-rule output
    let gdnOut: MTLBuffer?  // [valueDim] gated-norm output
    let sharedScalarGateBuf: MTLBuffer?  // [1] shared-expert gate logit
    /// BF16 ones over [numExperts]; neutral per_expert_scale when the router
    /// has no auxiliary scale tensors.
    let onesPerExpertScale: MTLBuffer?
    var prefillChunkState = PrefillChunkCommitState()
    var prefillScratch: PrefillChunkScratchBuffers?
    static let mtpChunkCapacity = 32
    let mtpTokenBlock: MTLBuffer?
    let mtpEmbeddingBlock: MTLBuffer?
    let mtpNormalizedEmbeddingBlock: MTLBuffer?
    let mtpNormalizedHiddenBlock: MTLBuffer?
    let mtpConcatBlock: MTLBuffer?
    let mtpProjectedBlock: MTLBuffer?
    let mtpTargetHiddenBlock: MTLBuffer?
    var mtpPrefillReadback: MTLBuffer?
    /// Reusable UInt32 token-ID buffer for chunked prefill (R23): sized to the
    /// largest chunk seen so far and grown on demand, so the prefill hot path
    /// never allocates an MTLBuffer per chunk.
    var prefillTokenBuffer: MTLBuffer?

    /// Host scratch reused across prefill chunks (R38) and decode layers (R16).
    /// The runner is single-flight per generation (guarded by
    /// `prefillChunkState` and the callers' serial decode loop), so these
    /// never alias concurrent work.
    var routeIDScratch: [UInt32] = []
    var routeWeightScratch: [Float16] = []
    var decodeExpertsScratch: [Int] = []
    var decodeHitSlotsScratch: [UInt32] = []
    var decodeMissSlotsScratch: [UInt32] = []
    var decodeHitSplitRoutedBufsScratch: [MTLBuffer] = []
    var decodeHitSplitRoutedOffsetsScratch: [Int] = []
    var decodeRoutedBufsScratch: [MTLBuffer] = []
    var decodeRoutedOffsetsScratch: [Int] = []

    static let rdadviseBoundedMissCap = 12
    static let rdadviseBoundedMaxCallNanos: UInt64 = 250_000
    static let rdadviseAdaptiveMissCap = 12
    static let rdadviseAdaptiveByteCap: UInt64 = 384 * 1_048_576
    static let rdadviseAdaptiveSlowCallNanos: UInt64 = 1_000_000
    static let prefillRoutedTileSchedulerConfig = PrefillRoutedTileSchedulerConfig()

    /// Per-layer `router.scale * D^-0.5` pre-folded into one BF16 buffer
    /// allocation per layer. ~168 KB total at 30 layers × 2816 BF16 — bounded
    /// host work done once at init.
    let effectiveScaleBuffers: [MTLBuffer]
    /// The (model, width) tuning this runner was built with.
    let profile: ModelProfile
    let sharedExpertProjections: [LayerSharedExpertProjections]

    public let maxContext: Int

    /// Per-instance head and RDADVISE modes. The fused head (default) skips the
    /// 512 KB logits write and leaves a greedy argmax in `lastGreedyToken`;
    /// callers that sample from the logits buffer (non-greedy configs) must pass
    /// `forceLogitsHead: true` or they read a never-written buffer.
    let useFusedGreedyHead: Bool
    let prefillAttentionPath: RuntimePrefillAttentionPath
    let decodeExpertExecution: RuntimeDecodeExpertExecution
    let expertIOSynchronization: RuntimeExpertIOSynchronization
    let expertIOSubmission: RuntimeExpertIOSubmission
    let expertIOBackend: ExpertIOBackend
    let predictivePrefetch: ExpertPrefetchRing?
    let anePrefill: ANEPrefillAttention?
    public let rdadviseEnabled: Bool
    public let rdadvisePolicyMode: RDAdvicePolicyMode
    var rdadviseSkipUntilPosition: Int = -1
    var rdadviseAdaptiveState: RDAdviceAdaptivePolicyState
    var rdadviseAdaptivePosition: Int = -1
    var rdadviseAdaptivePositionBytes: UInt64 = 0
    public init(
        model: Model, context: MetalContext, maxContext: Int,
        slots: Int = 1,
        runtimeConfiguration: RuntimeConfiguration = .production,
        enableSpeculativeGDN: Bool = false
    ) throws {
        self.model = model
        self.ctx = context
        self.cfg = model.config
        self.maxContext = maxContext
        precondition(
            slots > 0 && slots <= KVCacheManager.maximumSlots,
            "slots must be between 1 and \(KVCacheManager.maximumSlots)")
        self.slots = slots
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
        self.useFusedGreedyHead =
            runtimeConfiguration.headPath == .fusedRows
            && !cfg.hyperConnections.enabled
            && model.lmHeadWeightBits == 4
            && model.attentionWeightBits == 4
        self.prefillAttentionPath = runtimeConfiguration.prefillAttentionPath
        // The family's measured optimum is the default; an explicit
        // TINYTITAN_PREDICTIVE_PREFETCH still wins either way, so a probe can turn
        // it on where it ships off and off where it ships on.
        let profile = ModelProfile.resolve(
            modelID: model.modelID, family: cfg.family,
            weightBits: model.routedExpertWeightBits)
        self.profile = profile
        self.decodeExpertExecution = runtimeConfiguration.decodeExpertExecution
        self.expertIOSynchronization = runtimeConfiguration.expertIOSynchronization
        self.expertIOSubmission = runtimeConfiguration.expertIOSubmission
        self.expertIOBackend = try ExpertIOBackend.environmentValue()
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
        self.predictivePrefetch =
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
                self.anePrefill = try ANEPrefillAttention(
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
                self.anePrefill = nil
            }
        } else {
            self.anePrefill = nil
        }
        let useFP16Ring = runtimeConfiguration.fp16RingEnabled
        self.rdadvisePolicyMode = runtimeConfiguration.rdadvisePolicy
        self.rdadviseAdaptiveState = RDAdviceAdaptivePolicyState(
            config: RDAdviceAdaptivePolicyConfig(
                missCap: Self.rdadviseAdaptiveMissCap,
                byteCap: Self.rdadviseAdaptiveByteCap,
                slowCallNanos: Self.rdadviseAdaptiveSlowCallNanos))
        self.rdadviseEnabled = runtimeConfiguration.rdadviseEnabled
        self.kv = try KVCacheManager(
            device: context.device,
            config: cfg,
            maxContext: maxContext,
            slots: slots,
            fp16RingEnabled: useFP16Ring,
            precision: runtimeConfiguration.kvCachePrecision,
            slidingWindow: cfg.slidingWindow,
            maxPrefillChunkTokens: runtimeConfiguration.prefillChunkTokens)

        let silu = cfg.hiddenActivation == "silu"
        self.embedInt4 = try EmbedLookupInt4(context: context)
        self.affineEmbed =
            model.embeddingWeightBits == 4
            ? nil
            : try AffineQuantEmbeddingLookup(
                context: context,
                weightBits: model.embeddingWeightBits)
        self.rms = try RMSNorm(context: context)
        self.int4 = try DequantInt4GEMV(
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
        self.affineByWidth = affineByWidth
        self.affine = affineByWidth[model.attentionWeightBits]
        self.affineKV = affineByWidth[model.kvProjectionWeightBits]
        // The vocabulary head carries its own bit width. It matched the
        // attention slot in every earlier family, so the head simply reused
        // the attention GEMV -- which reads an 8-bit head as packed 4-bit the
        // moment a model quantizes the two differently, as this one does
        // (4-bit attention, 8-bit embedding and head). The result is a full
        // logit vector of confident nonsense, so the width is selected here
        // rather than inherited.
        self.affineHead =
            model.lmHeadWeightBits == 4
            ? nil
            : try AffineQuantGEMV(
                context: context,
                weightBits: model.lmHeadWeightBits)
        self.attention = try Attention(context: context)
        self.attention.simdPartialOverride = profile.attentionSimdPartial
        self.kvQuantizer =
            runtimeConfiguration.kvCachePrecision.isQuantized
            ? try KVCacheQuantizer(context: context) : nil
        self.shared = try SharedExpertRuntime(
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
        self.moe = try MoE(
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
        self.fusionHead = try LMHeadChainInt4(
            context: context,
            maxD: cfg.hiddenSize,
            maxVocab: cfg.vocabSize)
        self.fusedQKVGEMV = try FusedQKVGEMV(context: context)
        self.fusedQKVEpilogue = try FusedQKVEpilogue(context: context)
        self.prefillEmbed = try PrefillEmbedLookupInt4(
            context: context,
            weightBits: model.embeddingWeightBits)
        self.prefillRMS = try PrefillRMSNorm(context: context)
        self.prefillQMM = try PrefillInt4QMM(
            context: context,
            weightBits: model.attentionWeightBits)
        self.prefillMPPAffineInt4 = MPPPrefillInt4QMM(
            context: context,
            weightBits: model.attentionWeightBits)
        self.prefillQKVEpilogue = try PrefillQKVEpilogue(
            context: context,
            yarn: yarnParameters)
        self.prefillAttention = try PrefillAttention(context: context)
        self.prefillRouter = try PrefillRouter(
            context: context,
            weightBits: model.effectiveRouterWeightBits)
        self.prefillSharedExpert = try PrefillSharedExpert(
            context: context,
            weightBits: model.ffnWeightBits,
            siluActivation: silu)
        self.prefillGroupedMoE = try PrefillGroupedRoutedMoE(
            context: context,
            siluActivation: silu,
            weightBits: model.routedExpertWeightBits)
        self.prefillMoE = try PrefillMoE(context: context)
        self.prefillFinalRowHead = try PrefillFinalRowHeadInt4(
            context: context,
            maxD: cfg.hiddenSize,
            weightBits: model.lmHeadWeightBits)

        // Qwen 3.6 kernels, keyed off the data flags so architectures that
        // never dispatch them pay no PSO compile cost.
        let needsElementwise =
            cfg.attnOutputGate
            || cfg.sharedExpertGated
            || cfg.hasLinearAttentionLayers
        self.elementwise = needsElementwise ? try Elementwise(context: context) : nil
        self.activationDumpDirectory = ProcessInfo.processInfo
            .environment["TINYTITAN_ACT_DUMP"].map { URL(fileURLWithPath: $0) }
        // Both gated blocks keep a whole prefill chunk resident: the write
        // gate consumes what its matching read produced, and the block runs
        // in between, so the rows cannot be streamed one at a time.
        let gateRows = max(1, runtimeConfiguration.prefillChunkTokens)
        self.hyperConnection =
            cfg.hyperConnections.enabled
            ? try HyperConnection(
                context: context,
                dim: cfg.hiddenSize,
                streams: cfg.hyperConnections.count,
                lowRank: cfg.hyperConnections.lowRank,
                maxRows: gateRows,
                weightBits: model.hyperConnectionWeightBits)
            : nil
        self.qsaIndexer =
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
            self.pleHash = constants.makeHash()
            self.ngramTable = try NgramTableReader(
                path: model.directoryURL.appendingPathComponent(
                    Qwen38FlashTensors.ngramTableFile
                ).path,
                rowDim: constants.pleHeadDim,
                rowCount: constants.tableRowCount)
            self.pleBlock = try PLEBlock(
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
            self.pleHash = nil
            self.ngramTable = nil
            self.pleBlock = nil
        }
        if cfg.hasLinearAttentionLayers {
            self.gdn = try GDN(
                context: context, config: cfg.linearAttention,
                specializedHiddenSize: cfg.hiddenSize,
                abBF16: model.gdnABIsBF16)
            self.gdnState = try GDNStateManager(
                device: context.device,
                config: cfg,
                slots: slots,
                enableSpeculativeCheckpoint: enableSpeculativeGDN)
        } else {
            self.gdn = nil
            self.gdnState = nil
        }
        self.rope =
            cfg.ropeNeoxSubdim
            ? try RoPE(context: context, yarn: yarnParameters) : nil
        self.int8ScalarGate =
            cfg.sharedExpertGated
            ? try DequantInt8GEMV(
                context: context,
                additionalShapes: cfg.decodeInt8GEMVShapes)
            : nil
        self.bf16ScalarGate =
            cfg.sharedExpertGated
            ? try BF16GEMV(context: context) : nil
        self.bf16Projection = try BF16GEMV(context: context)

        let device = context.device
        let D = cfg.hiddenSize
        let F = cfg.intermediateSize
        let maxQ = cfg.numHeads * max(cfg.headDim, cfg.fullHeadDim)

        func buf(
            _ count: Int,
            _ stride: Int = MemoryLayout<Float16>.size,
            label: String
        ) throws -> MTLBuffer {
            guard
                let b = device.makeBuffer(
                    length: max(count, 1) * stride,
                    options: .storageModeShared)
            else {
                throw ModelError.residentBufferWrapFailed
            }
            b.label = label
            return b
        }
        // The residual is the one scratch buffer whose width is not D. A
        // hyper-connection family carries `hc_count` parallel streams of D and
        // reads a single D-wide vector out of them per sublayer, so only this
        // allocation widens -- every downstream buffer stays D. Families
        // without them get exactly D, as before.
        let residualElements =
            cfg.hyperConnections.enabled
            ? D * cfg.hyperConnections.count
            : D
        self.hidden = try buf(residualElements, label: "decode.hidden")
        self.normed = try buf(D, label: "decode.normed")
        self.attnOut = try buf(maxQ, label: "decode.attnOut")
        self.qScratch = try buf(maxQ, label: "decode.qScratch")
        self.kStage = try buf(
            max(
                cfg.numKVHeads * cfg.headDim,
                cfg.numFullKVHeads * cfg.fullHeadDim), label: "decode.kStage")
        self.vStage = try buf(
            max(
                cfg.numKVHeads * cfg.headDim,
                cfg.numFullKVHeads * cfg.fullHeadDim), label: "decode.vStage")
        self.oOut = try buf(D, label: "decode.oOut")
        self.h1Buf = try buf(D, label: "decode.h1")
        self.h2Buf = try buf(D, label: "decode.h2")
        self.routedX = try buf(D, label: "decode.routedX")
        self.denseX = try buf(D, label: "decode.denseX")
        self.denseScratchGate = try buf(F, label: "decode.denseScratchGate")
        self.denseScratchUp = try buf(F, label: "decode.denseScratchUp")
        self.denseScratchAct = try buf(F, label: "decode.denseScratchAct")
        self.routerInput = try buf(D, label: "decode.routerInput")
        self.zeroResidual = try buf(D, label: "decode.zeroResidual")
        // The routed MoE kernel seeds y[d] = residual[d]; pinning this buffer
        // to zero once at init makes the routed branch's residual contribution
        // exactly zero (it's combined with the dense MLP downstream).
        memset(self.zeroResidual.contents(), 0, self.zeroResidual.length)
        self.outIndices = try buf(
            cfg.topKExperts, MemoryLayout<UInt32>.size, label: "decode.outIndices")
        self.outWeights = try buf(cfg.topKExperts, label: "decode.outWeights")
        self.prefetchPredictionIndices = try buf(
            cfg.topKExperts, MemoryLayout<UInt32>.size, label: "decode.prefetchPredictionIndices")
        self.prefetchPrediction2Indices = try buf(
            cfg.topKExperts, MemoryLayout<UInt32>.size, label: "decode.prefetchPrediction2Indices")
        self.prefetchPrediction2Weights = try buf(
            cfg.topKExperts, MemoryLayout<Float16>.size, label: "decode.prefetchPrediction2Weights")
        self.prefetchPredictionWeights = try buf(
            cfg.topKExperts, label: "decode.prefetchPredictionWeights")
        self.moeActs = try buf(cfg.topKExperts * cfg.moeIntermediateSize, label: "decode.moeActs")
        self.moeHitActiveSlots = try buf(
            cfg.topKExperts, MemoryLayout<UInt32>.size, label: "decode.moeHitActiveSlots")
        self.moeMissActiveSlots = try buf(
            cfg.topKExperts, MemoryLayout<UInt32>.size, label: "decode.moeMissActiveSlots")
        self.residencyHitCount = try buf(
            1, MemoryLayout<UInt32>.size,
            label: "decode.residencyHitCount")
        self.residencyHitPositions = try buf(
            cfg.topKExperts, MemoryLayout<UInt32>.size,
            label: "decode.residencyHitPositions")
        self.residencyMissCount = try buf(
            1, MemoryLayout<UInt32>.size,
            label: "decode.residencyMissCount")
        self.residencyMissPositions = try buf(
            cfg.topKExperts, MemoryLayout<UInt32>.size,
            label: "decode.residencyMissPositions")
        self.residencyMissExperts = try buf(
            cfg.topKExperts, MemoryLayout<UInt32>.size,
            label: "decode.residencyMissExperts")
        self.residencyResolvedSlots = try buf(
            cfg.topKExperts, MemoryLayout<UInt32>.size,
            label: "decode.residencyResolvedSlots")
        self.residencyResolvedGenerations = try buf(
            cfg.topKExperts,
            MemoryLayout<UInt64>.size,
            label: "decode.residencyResolvedGenerations")
        guard
            let tok = device.makeBuffer(
                length: MemoryLayout<UInt32>.size,
                options: .storageModeShared)
        else {
            throw ModelError.residentBufferWrapFailed
        }
        tok.label = "decode.greedyToken"
        self.greedyTokenBuf = tok
        // Two rows of the residual as this family carries it: wide for a
        // hyper-connection model, because the draft's fusion reads all four
        // streams rather than a collapsed one.
        self.verificationHidden = try buf(
            2 * Self.residualWidthFor(cfg),
            label: "decode.verificationHidden")
        self.verificationLogits = try buf(2 * cfg.vocabSize, label: "decode.verificationLogits")

        // Qwen 3.6 decode scratch — allocated once here, never in the hot path.
        if cfg.attnOutputGate {
            self.qPackedScratch = try buf(2 * maxQ, label: "decode.qPackedScratch")
            self.attnGateScratch = try buf(maxQ, label: "decode.attnGateScratch")
        } else {
            self.qPackedScratch = nil
            self.attnGateScratch = nil
        }
        if cfg.hasLinearAttentionLayers {
            let la = cfg.linearAttention
            self.gdnQKVRaw = try buf(la.qkvDim, label: "decode.gdnQKVRaw")
            self.gdnConvOut = try buf(la.qkvDim, label: "decode.gdnConvOut")
            self.gdnZ = try buf(la.valueDim, label: "decode.gdnZ")
            self.gdnA = try buf(la.numVHeads, label: "decode.gdnA")
            self.gdnB = try buf(la.numVHeads, label: "decode.gdnB")
            self.gdnY = try buf(la.valueDim, label: "decode.gdnY")
            self.gdnOut = try buf(la.valueDim, label: "decode.gdnOut")
        } else {
            self.gdnQKVRaw = nil
            self.gdnConvOut = nil
            self.gdnZ = nil
            self.gdnA = nil
            self.gdnB = nil
            self.gdnY = nil
            self.gdnOut = nil
        }
        self.sharedScalarGateBuf =
            cfg.sharedExpertGated ? try buf(1, label: "decode.sharedScalarGate") : nil
        if cfg.family == .qwen36MTP || cfg.family == .qwen38flashMTP {
            guard
                let tokenBlock = ctx.device.makeBuffer(
                    length: Self.mtpChunkCapacity * MemoryLayout<UInt32>.stride,
                    options: .storageModeShared)
            else {
                throw ModelError.residentBufferWrapFailed
            }
            self.mtpTokenBlock = tokenBlock
            self.mtpEmbeddingBlock = try buf(Self.mtpChunkCapacity * D, label: "mtp.embedding")
            self.mtpNormalizedEmbeddingBlock = try buf(
                Self.mtpChunkCapacity * D, label: "mtp.normalizedEmbedding")
            self.mtpNormalizedHiddenBlock = try buf(
                Self.mtpChunkCapacity * D, label: "mtp.normalizedHidden")
            // Qwen 3.6 concatenates the two normalized branches and runs one
            // projection; Qwen3.8-Flash-Next projects each separately and adds.
            // The wider block covers the concatenation the first needs and the
            // wide residual the second reads.
            let fuseWidth = max(2 * D, Self.residualWidthFor(cfg))
            self.mtpConcatBlock = try buf(
                Self.mtpChunkCapacity * fuseWidth,
                label: "mtp.concat")
            self.mtpProjectedBlock = try buf(Self.mtpChunkCapacity * D, label: "mtp.projected")
            self.mtpTargetHiddenBlock = try buf(
                Self.mtpChunkCapacity * Self.residualWidthFor(cfg),
                label: "mtp.targetHidden")
        } else {
            self.mtpTokenBlock = nil
            self.mtpEmbeddingBlock = nil
            self.mtpNormalizedEmbeddingBlock = nil
            self.mtpNormalizedHiddenBlock = nil
            self.mtpConcatBlock = nil
            self.mtpProjectedBlock = nil
            self.mtpTargetHiddenBlock = nil
        }
        self.mtpPrefillReadback = nil

        func sharedProj(_ view: TensorView, rows: UInt32, cols: UInt32) -> SharedExpertProjection {
            SharedExpertProjection(
                weights: view.buffer,
                scales: view.buffer,
                biases: view.buffer,
                weightsOffset: Int(view.offset),
                scalesOffset: Int(view.scaleOffset),
                biasesOffset: Int(view.biasOffset),
                rows: rows,
                cols: cols)
        }
        var sharedViews: [LayerSharedExpertProjections] = []
        sharedViews.reserveCapacity(cfg.numLayers)
        for L in 0..<cfg.numLayers {
            let gate = try model.sharedExpertGate(layer: L)
            let up = try model.sharedExpertUp(layer: L)
            let down = try model.sharedExpertDown(layer: L)
            sharedViews.append(
                LayerSharedExpertProjections(
                    gate: sharedProj(gate, rows: UInt32(F), cols: UInt32(D)),
                    up: sharedProj(up, rows: UInt32(F), cols: UInt32(D)),
                    down: sharedProj(down, rows: UInt32(D), cols: UInt32(F)),
                    scalarGate: cfg.sharedExpertGated
                        ? try model.sharedExpertScalarGate(layer: L) : nil))
        }
        self.sharedExpertProjections = sharedViews

        func bf16OnesBuffer(count: Int, label: String) throws -> MTLBuffer {
            // `max(count, 1)`: a dense model has no experts, so its per-expert
            // scale holds nothing -- and Metal needs a non-empty allocation.
            // Nothing reads it on that path (the router stage is skipped), so
            // the one element is a placeholder, not a value.
            guard
                let buf = device.makeBuffer(
                    length: max(count, 1) * MemoryLayout<UInt16>.size,
                    options: .storageModeShared)
            else {
                throw ModelError.residentBufferWrapFailed
            }
            let dst = buf.contents().assumingMemoryBound(to: UInt16.self)
            for i in 0..<count { dst[i] = 0x3F80 }  // BF16 1.0
            buf.label = label
            return buf
        }

        // Plain linear router (Qwen): one shared BF16 ones buffer keeps
        // the router kernel's effective_scale multiply neutral, and a ones
        // per_expert_scale keeps the top-k weights untouched. (Softmax
        // over top-k then renormalize equals Qwen's softmax over all
        // experts then renormalize the selected top-k.)
        let ones = try bf16OnesBuffer(count: D, label: "effective_scale.ones")
        self.effectiveScaleBuffers = [MTLBuffer](
            repeating: ones,
            count: cfg.numLayers)
        self.onesPerExpertScale = try bf16OnesBuffer(
            count: cfg.numExperts,
            label: "per_expert_scale.ones")
        if profile.keepExpertCacheWired {
            model.setKeepExpertCacheWired(true)
            model.setExpertCachePinned(true)
        }
    }

    var decodeIOBaseline: ExpertStreamingStatistics?

    public internal(set) var totalIoNanos: UInt64 = 0
    public internal(set) var totalCb1Nanos: UInt64 = 0
    public internal(set) var totalCb2Nanos: UInt64 = 0
    public internal(set) var totalHeadNanos: UInt64 = 0
    public internal(set) var totalHeadFusedNanos: UInt64 = 0
    // Overlap-analysis counters (TINYTITAN_RUNNER_STATS): the per-layer wall spent
    // waiting on the attention+router command buffer (covers the previous
    // layer's routed CB plus this layer's cb1 on the GPU) and the per-layer
    // loop-body wall. body = cb1 + wait + readback/plan + rdadvise + io + cb2.
    public internal(set) var totalWaitNanos: UInt64 = 0
    public internal(set) var totalBodyNanos: UInt64 = 0
    // Between-token segments (TINYTITAN_RUNNER_STATS): what a decode step spends
    // before its layer loop starts -- the produce preamble (cache pin, KV
    // reserve, QSA check), the embed dispatches, the n-gram gather -- and,
    // filled in by the completion loop, the sampler wait, the progress
    // callback and the loop's own remainder. Together with `body` and `head`
    // these account for the whole token, so a GPU gap can be placed.
    public internal(set) var totalPrefetchAdopted: UInt64 = 0
    public internal(set) var totalPreambleNanos: UInt64 = 0
    public internal(set) var totalPreambleReleaseNanos: UInt64 = 0
    public internal(set) var totalPreamblePinNanos: UInt64 = 0
    public internal(set) var totalPreambleReserveNanos: UInt64 = 0
    public internal(set) var totalEmbedNanos: UInt64 = 0
    public internal(set) var totalGatherNanos: UInt64 = 0
    public var totalLoopSampleNanos: UInt64 = 0
    public var totalLoopProgressNanos: UInt64 = 0
    public var totalLoopOtherNanos: UInt64 = 0
    public internal(set) var totalMissIoNanos: UInt64 = 0
    public internal(set) var totalExposedIoNanos: UInt64 = 0
    public internal(set) var totalHitFixupLayers: UInt64 = 0
    public internal(set) var totalRouterReadbackNanos: UInt64 = 0
    public internal(set) var totalCachePlanNanos: UInt64 = 0
    public internal(set) var totalIOQueueNanos: UInt64 = 0
    public internal(set) var totalIOCompletionToFixupSubmitNanos: UInt64 = 0
    public internal(set) var totalExpertIOHostWaits: UInt64 = 0
    public internal(set) var totalExpertIOHostWaitsAvoided: UInt64 = 0
    public internal(set) var totalGPUClassifiedHits: UInt64 = 0
    public internal(set) var totalGPUClassifiedMisses: UInt64 = 0
    public internal(set) var totalGPUResidencyAllHitLayers: UInt64 = 0
    public internal(set) var lastGreedyToken: UInt32 = 0
    public internal(set) var totalRDAdviseNanos: UInt64 = 0
    public internal(set) var totalRDAdviseCalls: UInt64 = 0
    public internal(set) var totalRDAdviseBytes: UInt64 = 0
    public internal(set) var totalRDAdviseFailures: UInt64 = 0
    public internal(set) var totalRDAdviseSkipped: UInt64 = 0
    var kernelGPUTimings: [KernelGPUTiming] = []
    /// Command buffers split out per kernel under TINYTITAN_KERNEL_SPLIT, awaiting
    /// their GPU timestamps; drained into `kernelGPUTimings` after the layer's
    /// tail wait.
    var splitTimedBuffers: [(role: String, cb: MTLCommandBuffer)] = []
    let kernelGPUTimingsEnabled =
        ProcessInfo.processInfo.environment["TINYTITAN_KERNEL_STATS"] != nil
    let runnerStatsEnabled =
        ProcessInfo.processInfo.environment["TINYTITAN_RUNNER_STATS"] != nil

    /// Open file descriptor for TINYTITAN_ROUTE_TRACE, or -1. Opened once and
    /// never closed: the runner lives as long as the process, and a decode
    /// loop is the wrong place to manage a diagnostic file's lifetime.
    let routeTraceFD: Int32 = {
        guard let path = ProcessInfo.processInfo.environment["TINYTITAN_ROUTE_TRACE"],
            !path.isEmpty
        else { return -1 }
        return open(path, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
    }()

    /// JSONL trace for the v4.3 predictive-prefetch qualification probe.
    /// It deliberately records only exact routing and authoritative cache
    /// residency before planning; enabling it cannot submit I/O or alter cache
    /// decisions. Kept separate from TINYTITAN_ROUTE_TRACE for compatibility.
    let prefetchTraceFD: Int32 = {
        guard let path = ProcessInfo.processInfo.environment["TINYTITAN_PREFETCH_TRACE"],
            !path.isEmpty
        else { return -1 }
        return open(path, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
    }()

    // MARK: - Chunked prefill helpers

    // MARK: - Decode routed-expert helpers

}
