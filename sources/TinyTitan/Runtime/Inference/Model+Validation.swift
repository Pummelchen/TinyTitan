//
//  Model+Validation.swift
//  TinyTitan
//
//  The runtime schema validation: the geometry and schema checks a load runs,
//  plus the receipt layer-layout check and the threadgroup tile bounds.
//
import Darwin
import Foundation
import Metal
import TinyTitanFormat

extension Model {

    static func validateTrustedReceiptLayerLayout(
        modelDirectory: SSDAIModelDirectory,
        manifest: Manifest,
        layout: PackedExpertsLayout
    ) throws {
        // A dense install packs no experts: its layout names one file per layer
        // with an empty expert list, and the repacker wrote none of them, so
        // there is no per-layer payload for the receipt to attest. That is a
        // property of the install rather than a missing file -- anything that
        // does pack experts still goes through the loop below, and a zero-expert
        // layout that names experts was already refused when the layout decoded.
        guard layout.expertsPerLayer > 0, layout.expertStride > 0 else { return }
        for layer in layout.layers {
            let relativePath = "packed_experts/\(layer.file)"
            guard let manifestEntry = manifest.files[relativePath] else {
                throw ModelError.trustedReceiptInvalid(
                    detail: "manifest missing \(relativePath)")
            }
            let actualSize: UInt64
            do {
                let fd = try modelDirectory.openFile(relativePath)
                defer { close(fd) }
                actualSize = try modelDirectory.fileSize(
                    fileDescriptor: fd, relativePath: relativePath)
            }
            guard actualSize == manifestEntry.size else {
                throw ModelError.trustedReceiptInvalid(
                    detail: "\(relativePath) size \(actualSize) != \(manifestEntry.size)")
            }
        }
    }

    /// Width of the fixed threadgroup tiles the MoE and GDN kernels stage
    /// activations into: `kMoEXMaxD` in `moe.metal` and
    /// `kGDNActivationMaxD` in `gdn.metal`. A hidden size above this writes
    /// past the tile, so `validateRuntimeSchema` refuses it rather than letting
    /// the kernel do it.
    ///
    /// The bound is per kernel family because a model dispatches one or the
    /// other: the MoE tiles are 2816 wide and the dense Qwen 3.5 9B (4096) never
    /// reaches them, while its Gated-DeltaNet layers do reach the staging tile,
    /// which is sized for 4096.
    static let maximumThreadgroupTileWidth = 2816
    static let maximumDenseThreadgroupTileWidth = 4096

    /// Refuses model geometry the compiled kernels cannot serve.
    ///
    /// Split out of `validateRuntimeSchema` to keep it inside the project's
    /// function-length gate. Both checks are about the *shape* of the model
    /// rather than its tensors, and both exist because the failure they prevent
    /// is silent: the kernels index threadgroup memory and pick attention
    /// geometry from these values, so a shape they were not compiled for does not
    /// fail, it produces wrong numbers.
    static func validateExecutableGeometry(_ config: ArchConfig) throws {
        // A sliding-window layer (mask 0) is a valid `ArchConfig` value and the
        // CPU engine implements it, but the GPU path does not: the gated
        // attention branch runs every non-linear layer as *full* attention with
        // `fullHeadDim`, `numFullKVHeads` and `fullRopeTheta`, and prefill's neox
        // branch uses `fullRopeTheta` unconditionally too. A mask-0 layer would
        // attend to the whole context with the wrong row geometry, silently. No
        // shipped preset declares one (they use 1 and 2), so this refuses rather
        // than guessing — implementing windowed handling in that branch is the
        // alternative, and this guard is what says which one is missing.
        guard !config.fullAttentionLayerMask.contains(0) else {
            throw ModelError.unsupportedArchitecture(
                detail: "a sliding-window layer (fullAttentionLayerMask == 0) is not "
                    + "implemented on the GPU path: the gated attention branch runs "
                    + "every non-linear layer as full attention")
        }
        // A model with no routed experts never dispatches the MoE kernels, so
        // only the Gated-DeltaNet tile bounds it -- and that tile is wider.
        let tileWidth =
            config.numExperts == 0
            ? Self.maximumDenseThreadgroupTileWidth
            : Self.maximumThreadgroupTileWidth
        guard config.hiddenSize <= tileWidth else {
            throw ModelError.unsupportedArchitecture(
                detail: "hiddenSize \(config.hiddenSize) exceeds the "
                    + "\(tileWidth)-element threadgroup tiles the "
                    + (config.numExperts == 0 ? "Gated-DeltaNet" : "MoE and Gated-DeltaNet")
                    + " kernels are compiled with")
        }
        // The attention kernels size their scratch from two compile-time
        // ceilings: `Attention.maxQHeads` (the host-side split-KV reduction
        // buffer) and `kAttnMaxHeadDim` in `attention.metal`, which declares
        // `q_smem` and the per-thread row from it. Nothing bounded the config
        // against either. A manifest with more query heads than the ceiling
        // reaches `Attention.encode`'s own precondition and **traps** -- an abort
        // on install-derived data -- and a head dimension above the kernel's
        // constant overruns threadgroup memory, silently. Refused at load, for
        // the same reason as the hidden-size bound above.
        guard config.numHeads <= Attention.maxQHeads else {
            throw ModelError.unsupportedArchitecture(
                detail: "numHeads \(config.numHeads) exceeds the \(Attention.maxQHeads)-head "
                    + "split-KV scratch the attention kernels are built with")
        }
        guard config.fullHeadDim <= Attention.maxHeadDim,
            config.headDim <= Attention.maxHeadDim
        else {
            throw ModelError.unsupportedArchitecture(
                detail: "head dimension \(max(config.fullHeadDim, config.headDim)) exceeds "
                    + "the \(Attention.maxHeadDim)-element attention threadgroup tile")
        }

    }

    static func validateRuntimeSchema(
        residentIndex: ResidentIndex,
        layout: PackedExpertsLayout,
        manifest: Manifest,
        config: ArchConfig
    ) throws {
        guard let quant = manifest.quant else {
            throw ModelError.indexCorrupt(
                detail: "manifest.quant is required by the executable runtime schema")
        }
        // The MoE and GDN kernels stage activations into fixed threadgroup tiles
        // of 2816 elements (`kMoEXMaxD` in moe.metal, `xt[2816]` in gdn.metal),
        // and their staging loops are bounded by the configured hidden size. A
        // model wider than that writes past the tile into whatever shares the
        // threadgroup's memory -- undefined behaviour rather than a caught
        // error, and reachable only by a config that has never shipped (every
        // preset here is 2048, 2560 or 2816). This is the guard for the next
        // family, and it is what lets the tile stay a compile-time constant.
        try Self.validateExecutableGeometry(config)

        let checks = RuntimeSchemaChecks(residentIndex: residentIndex, quant: quant)

        switch config.family {
        case .qwen35Dense:
            try Self.validateDenseSchema(
                checks: checks, config: config,
                quant: quant, overrides: manifest.quantOverrides)
        case .qwen38flash:
            // Embedding and head are 8-bit in this checkpoint while the body
            // is 4-bit, so both are validated against the embedding slot the
            // manifest declares rather than an assumed width.
            try checks.requireAffine(
                "model.language_model.embed_tokens.weight",
                rows: config.vocabSize,
                columns: config.hiddenSize,
                slot: quant.embedding)
            // `lm_head` sits at the archive root in this family, not under the
            // language-model prefix.
            try checks.requireAffine(
                "lm_head.weight",
                rows: config.vocabSize,
                columns: config.hiddenSize,
                slot: quant.embedding)
            // The hyper-connection residual is the family's defining feature
            // and the one thing whose absence would let a mis-repacked payload
            // load and then compute a plain-residual model. Check the
            // model-level mixer and one layer's worth of both sublayer gates.
            let hcDim = config.hiddenSize * config.hyperConnections.count
            try checks.requireBF16(
                "model.language_model.hyper_connection_mixer.hc_norm",
                count: hcDim)
            for layer in 0..<config.numLayers {
                try checks.requireBF16(
                    "model.language_model.layers.\(layer)."
                        + "attn_hyper_connection.hc_norm", count: hcDim)
                try checks.requireBF16(
                    "model.language_model.layers.\(layer)."
                        + "mlp_hyper_connection.hc_norm", count: hcDim)
            }
            // The PLE block exists on exactly the configured layers, and its
            // constants and table are passthrough files rather than tensors --
            // their presence is the manifest's business, checked below.
            for layer in config.ple.layerIndices {
                try checks.requireBF16(
                    "model.language_model.layers.\(layer).ple.conv1d",
                    count: hcDim * config.ple.convKernelSize)
            }
            guard manifest.files[Qwen38FlashTensors.pleConstantsFile] != nil else {
                throw ModelError.missingFile(
                    name: Qwen38FlashTensors.pleConstantsFile)
            }
        case .qwen38flashMTP:
            try Self.validateQwen38DraftSchema(
                checks: checks, quant: quant,
                config: config)
        case .qwen36:
            try checks.requireAffine(
                "language_model.model.embed_tokens.weight",
                rows: config.vocabSize,
                columns: config.hiddenSize,
                slot: quant.embedding)
            // The untied lm_head is quantized with the embedding slot layout
            // (padded to the same vocab rows). `Model.lmHeadWeightBits` falls
            // back to that slot, so the coupling is validated here — the
            // fallback is only reachable when this check already passed.
            try checks.requireAffine(
                "language_model.lm_head.weight",
                rows: config.vocabSize,
                columns: config.hiddenSize,
                slot: quant.embedding)
        case .qwen36MTP:
            // The MTP sidecar shares the target's embedding and lm_head; it
            // carries only the 2D->D projection and its two input norms.
            try checks.requireAffine(
                "fc.weight",
                rows: config.hiddenSize,
                columns: 2 * config.hiddenSize,
                slot: quant.attention)
            try checks.requireBF16("pre_fc_norm_embedding.weight", count: config.hiddenSize)
            try checks.requireBF16("pre_fc_norm_hidden.weight", count: config.hiddenSize)
        }
        // Resolved through the family's schema: this norm is not always
        // `model.norm`, and not always `hiddenSize` wide. A hyper-connection
        // family collapses its streams through a mixer whose norm spans the
        // full residual.
        try checks.requireBF16(
            TensorSchema.schema(for: config.family).finalNorm,
            count: config.hyperConnections.enabled
                ? config.hiddenSize * config.hyperConnections.count
                : config.hiddenSize)

        try validateLayerSchema(
            checks: checks, layout: layout, config: config,
            quant: quant, overrides: manifest.quantOverrides)

    }

    /// The dense Qwen 3.5 family's own tensors.
    ///
    /// No router and no shared expert, an MLP that *is* the FFN, and per-tensor
    /// widths that differ from the slots: `mlp.*` is 4-bit against an 8-bit
    /// `sharedExpert` slot, and the full-attention `k_proj`/`v_proj` are 8-bit
    /// against a 4-bit `attention` slot. Every check resolves the tensor's own
    /// slot, which is why this family has a branch of its own rather than
    /// reusing the qwen36 one.
    private static func validateDenseSchema(
        checks: RuntimeSchemaChecks,
        config: ArchConfig,
        quant: ManifestQuant,
        overrides: [String: Int]
    ) throws {
        let dense = TensorSchema.schema(for: .qwen35Dense)
        try checks.requireAffine(
            dense.embedding, rows: config.vocabSize, columns: config.hiddenSize,
            slot: quant.slot(
                forTensorNamed: dense.embedding,
                overrides: overrides, fallback: quant.embedding))
        if !config.tieWordEmbeddings {
            // The 9B. The 2B and 4B tie the embedding and ship no head tensor
            // at all, so requiring one there would refuse a correct install.
            try checks.requireAffine(
                dense.lmHead, rows: config.vocabSize, columns: config.hiddenSize,
                slot: quant.slot(
                    forTensorNamed: dense.lmHead,
                    overrides: overrides, fallback: quant.embedding))
        }
    }

    /// Per-layer tensor schema: shapes, dtypes and quant layouts for every
    /// transformer layer, plus the packed-expert layout cross-check.
    private static func validateLayerSchema(
        checks: RuntimeSchemaChecks,
        layout: PackedExpertsLayout,
        config: ArchConfig,
        quant: ManifestQuant,
        overrides: [String: Int]
    ) throws {
        // Qwen 3.6 schema, verified against the installed checkpoints:
        // every layer carries the layer norms, the router and the gated
        // shared expert; full-attention layers carry the gate-packed
        // [query; gate] q_proj, and gated-DeltaNet layers carry the
        // linear_attn bundle. The Qwen checkpoints keep no auxiliary
        // sandwich/scale tensors.
        try validateFamilyQuantSupport(
            config: config, quant: quant,
            overrides: overrides)
        try validateRoleUniformity(
            overrides: overrides, family: config.family,
            attentionBits: quant.attention.weightBits)
        try validateLayerTensors(
            checks: checks, config: config, quant: quant,
            overrides: overrides)
        // A dense install packs no experts at all (`expertsPerLayer: 0` and an
        // empty layout), so the routed cross-check has nothing to cross-check
        // and would divide by zero experts.
        if config.numExperts > 0 {
            try validateRoutedExpertLayout(
                checks: checks, layout: layout,
                config: config, quant: quant)
        }
    }
}
