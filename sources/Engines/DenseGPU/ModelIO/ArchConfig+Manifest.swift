import Foundation
import TinyTitanCore

/// Resolving a model's architecture: the family's preset, or the manifest's own
/// declaration when the family has no preset.
///
/// `ArchConfig.knownArchitectures` holds one preset per family, which works
/// while a family has a single geometry. The dense Qwen 3.5 family does not:
/// the 2B and 4B are hidden 2048 with a 6144-wide FFN and the 9B is 12288, all
/// under one `ModelFamily` case. Rather than split the enum or carry three
/// presets, the architecture comes from the manifest, which already declares
/// every field the presets spell -- geometry, the full-attention layer mask,
/// the Gated-DeltaNet widths, and the layer conventions.
///
/// The conventions are *required*, not defaulted. The presets state them per
/// family and the kernels depend on them structurally: reading a gated
/// attention output as ungated, or a NeoX sub-dimension RoPE as the interleaved
/// one, yields fluent nonsense rather than a shape error, so a manifest that
/// omits one is refused by name instead of being run under an assumption.
public enum ArchResolutionError: Error, CustomStringConvertible {
    case unsupportedFamily(String)
    case manifestOmitsField(family: String, field: String)
    case manifestDeclaresInvalidValue(family: String, field: String, value: String)

    public var description: String {
        switch self {
        case .unsupportedFamily(let family):
            return "model declares family \(family), which this runtime does not implement"
        case .manifestOmitsField(let family, let field):
            return "the \(family) manifest does not declare \(field); this family's "
                + "architecture is read from the manifest, and that field cannot be "
                + "assumed without risking a silently wrong model"
        case .manifestDeclaresInvalidValue(let family, let field, let value):
            return "the \(family) manifest declares \(field)=\(value), which this "
                + "runtime does not implement"
        }
    }
}

extension ArchConfig {

    /// The architecture to load a model at: its family's preset when there is
    /// one, otherwise the manifest's own declaration.
    public static func resolved(
        forFamily family: ModelFamily,
        directoryURL: URL
    ) throws -> ArchConfig {
        if let preset = knownArchitectures[family] {
            // `qwen36` is no longer one geometry. It began as the 35B-A3B
            // (hidden 2048, 40 layers), and Ornith-1.5-122B is the same
            // architecture at hidden 3072 and 48 layers. Returning the 35B
            // preset for the 122B install would validate the manifest against
            // the wrong contract and refuse it; returning a *manifest-declared*
            // ArchConfig for it keeps one family while allowing the second
            // shape. The read is best-effort: a directory with no readable
            // manifest falls through to the preset exactly as before (and the
            // real load then reports why), so the preset path still answers
            // without a manifest.
            if family == .qwen36,
                let manifest = try? ManifestReader.read(directoryURL: directoryURL),
                manifest.arch.hiddenSize != preset.hiddenSize
                    || manifest.arch.numLayers != preset.numLayers
            {
                // Every convention the kernels depend on is `required` by
                // `from`, so a wider install cannot load with an assumed
                // gate, attention scale or layer mask.
                return try from(manifest: manifest.arch, family: family)
            }
            // The 4.0 checkpoint is not published yet, so its preset is the
            // 3.8 preview's geometry. If the real model is a different width
            // (the leaked name says 120B against the preview's 125B), its own
            // manifest is the authority and `from` rebuilds every convention
            // the kernels depend on -- including the hyper-connection, indexer
            // and PLE blocks that the preset path would otherwise impose.
            if family == .qwen4flash || family == .qwen4flashMTP,
                let manifest = try? ManifestReader.read(directoryURL: directoryURL),
                manifest.arch.hiddenSize != preset.hiddenSize
                    || manifest.arch.numLayers != preset.numLayers
            {
                return try from(manifest: manifest.arch, family: family)
            }
            return preset
        }
        // The families without a preset whose architecture is fully declared by
        // the manifest. Anything else keeps the previous behaviour: refused by
        // name, because there is no preset to load it at.
        guard family == .qwen35Dense else {
            throw ArchResolutionError.unsupportedFamily(family.rawValue)
        }
        // `read`, not `load`: this is precisely the case `load` cannot serve --
        // there is no GPU config to validate the manifest against yet.
        let manifest = try ManifestReader.read(directoryURL: directoryURL)
        return try from(manifest: manifest.arch, family: family)
    }

    /// The architecture a manifest declares.
    static func from(manifest arch: ManifestArch, family: ModelFamily) throws -> ArchConfig {
        func required<T>(_ value: T?, _ field: String) throws -> T {
            guard let value else {
                throw ArchResolutionError.manifestOmitsField(
                    family: family.rawValue,
                    field: field)
            }
            return value
        }
        // The Gated-DeltaNet geometry, on the same terms: the CPU engine reads
        // it from the snapshot config and the GPU path must read it from the
        // manifest rather than from a preset that describes another model.
        let isFlash = family == .qwen4flash || family == .qwen4flashMTP
        let isDraftFlash = family == .qwen4flashMTP
        let linear = LinearAttentionConfig(
            numKHeads: try required(arch.linearNumKHeads, "linearNumKHeads"),
            numVHeads: try required(arch.linearNumVHeads, "linearNumVHeads"),
            keyHeadDim: try required(arch.linearKeyHeadDim, "linearKeyHeadDim"),
            valueHeadDim: try required(arch.linearValueHeadDim, "linearValueHeadDim"),
            convKernelSize: try required(arch.linearConvKernelSize, "linearConvKernelSize"),
            // Qwen 3.5 and 3.6 are the same lineage; the Flash families state
            // `sigmoid`. Declared here rather than defaulted from the other
            // one, which is the point the config type makes.
            outputGate: isFlash ? .sigmoid : .silu)
        // The Flash extensions are required on this path: a 4.0 install whose
        // manifest omits its hyper-connection, indexer or PLE geometry would
        // otherwise load as a plain-residual model with no error. The draft
        // carries the first two but has no n-gram block of its own.
        let hyperConnections =
            isFlash
            ? HyperConnectionConfig(
                count: try required(arch.hcCount, "hcCount"),
                lowRank: try required(arch.hcLowRank, "hcLowRank"))
            : HyperConnectionConfig(
                count: arch.hcCount ?? 0, lowRank: arch.hcLowRank ?? 0)
        let sparseIndexer =
            isFlash
            ? SparseIndexerConfig(
                numHeads: try required(arch.indexerNumHeads, "indexerNumHeads"),
                numKVHeads: try required(arch.indexerNumKVHeads, "indexerNumKVHeads"),
                headDim: try required(arch.indexerHeadDim, "indexerHeadDim"),
                budget: try required(arch.indexerBudget, "indexerBudget"),
                compressRatio: try required(
                    arch.indexerCompressRatio, "indexerCompressRatio"))
            : (arch.indexerBudget.map {
                SparseIndexerConfig(
                    numHeads: arch.indexerNumHeads ?? 0,
                    numKVHeads: arch.indexerNumKVHeads ?? 0,
                    headDim: arch.indexerHeadDim ?? 0,
                    budget: $0,
                    compressRatio: arch.indexerCompressRatio ?? 0)
            } ?? .none)
        let ple: PLEConfig =
            isFlash && !isDraftFlash
            ? PLEConfig(
                layerIndices: try required(arch.pleLayerIndices, "pleLayerIndices"),
                embedDim: try required(arch.pleEmbedDim, "pleEmbedDim"),
                convKernelSize: try required(
                    arch.pleConvKernelSize, "pleConvKernelSize"),
                ngramSize: try required(arch.pleNgramSize, "pleNgramSize"),
                vocabSizeBase: try required(
                    arch.pleVocabSizeBase, "pleVocabSizeBase"),
                headsPerNgram: try required(
                    arch.pleHeadsPerNgram, "pleHeadsPerNgram"),
                vocabDivisor: try required(
                    arch.pleVocabDivisor, "pleVocabDivisor"),
                // The seed is not a manifest field; it is recorded for
                // validation only and the shipped 4.0 table is read, not
                // re-derived.
                seed: 1234)
            : .none
        let shape = try layerShape(manifest: arch, family: family)
        return ArchConfig(
            hiddenSize: arch.hiddenSize,
            intermediateSize: arch.ffnIntermediate,
            moeIntermediateSize: arch.moeIntermediateSize,
            numHeads: arch.numHeads,
            numKVHeads: arch.numKVHeads,
            numFullKVHeads: arch.numFullKVHeads,
            headDim: arch.headDim,
            fullHeadDim: arch.fullHeadDim,
            vocabSize: arch.vocabSize,
            slidingWindow: arch.slidingWindow,
            finalLogitSoftcap: arch.finalLogitSoftcap,
            ropeTheta: arch.ropeTheta,
            fullRopeTheta: arch.fullRopeTheta,
            partialRotaryFactor: arch.partialRotaryFactor,
            numLayers: arch.numLayers,
            numExperts: arch.numExperts,
            topKExperts: arch.topKExperts,
            tieWordEmbeddings: arch.tieWordEmbeddings,
            attentionKEqV: arch.attentionKEqV,
            fullAttentionLayerMask: arch.fullAttentionLayerMask.map { UInt8($0) },
            hiddenActivation: arch.hiddenActivation,
            family: family.rawValue,
            attnOutputGate: try required(arch.attnOutputGate, "attnOutputGate"),
            attentionScale: try required(arch.attentionScale, "attentionScale"),
            embeddingScaledBySqrtHidden: try required(
                arch.embeddingScaledBySqrtHidden,
                "embeddingScaledBySqrtHidden"),
            routerScaled: try required(arch.routerScaled, "routerScaled"),
            ffnSandwichNorms: try required(arch.ffnSandwichNorms, "ffnSandwichNorms"),
            sharedExpertGated: try required(arch.sharedExpertGated, "sharedExpertGated"),
            ropeNeoxSubdim: try required(arch.ropeNeoxSubdim, "ropeNeoxSubdim"),
            linearAttention: linear,
            hyperConnections: hyperConnections,
            sparseIndexer: sparseIndexer,
            ple: ple,
            // Whether the router normalizes over the top-k is declared per
            // install (`norm_topk_prob`). See `layerShape`.
            routerNormTopK: shape.routerNormTopK,
            quantGroupSize: arch.quantGroupSize ?? 64,
            normGroups: shape.normGroups,
            attentionGate: shape.attentionGate,
            routerHasBias: shape.routerHasBias,
            routerScaling: shape.routerScaling,
            denseLayerIndices: shape.denseLayerIndices,
            mova: shape.mova)
    }

    /// The family-dependent layer shape a manifest declares.
    ///
    /// Split out of `from` under the 120-line gate. A family that declares
    /// these reads them by name; one that does not reads what it declares and
    /// defaults the rest, which keeps the dense manifests loading unchanged.
    private struct ManifestLayerShape {
        let attentionGate: AttentionGateFunction
        let normGroups: Int
        let denseLayerIndices: [Int]
        let mova: MoVAConfig
        let routerHasBias: Bool
        let routerScaling: Double
        let routerNormTopK: Bool
    }

    private static func layerShape(
        manifest arch: ManifestArch,
        family: ModelFamily
    ) throws -> ManifestLayerShape {
        let isFlash = family == .qwen4flash || family == .qwen4flashMTP
        func extensionValue<T>(
            _ value: T?, default fallback: @autoclosure () -> T
        ) -> T {
            value ?? fallback()
        }
        let gateName = extensionValue(
            arch.attentionGate, default: "sigmoid")
        guard let attentionGate = AttentionGateFunction(rawValue: gateName) else {
            throw ArchResolutionError.manifestDeclaresInvalidValue(
                family: family.rawValue, field: "attentionGate", value: gateName)
        }
        let normGroups = extensionValue(
            arch.normGroups, default: 1)
        guard normGroups > 0, arch.hiddenSize % normGroups == 0 else {
            throw ArchResolutionError.manifestDeclaresInvalidValue(
                family: family.rawValue, field: "normGroups",
                value: "\(normGroups) (hiddenSize \(arch.hiddenSize))")
        }
        let denseLayerIndices = extensionValue(
            arch.denseLayerIndices, default: [])
        guard denseLayerIndices.allSatisfy({ $0 >= 0 && $0 < arch.numLayers }),
            Set(denseLayerIndices).count == denseLayerIndices.count,
            denseLayerIndices == denseLayerIndices.sorted()
        else {
            throw ArchResolutionError.manifestDeclaresInvalidValue(
                family: family.rawValue, field: "denseLayerIndices",
                value: "\(denseLayerIndices)")
        }
        let mova = MoVAConfig(
            numExperts: extensionValue(
                arch.movaNumExperts, default: 0),
            topKExperts: extensionValue(
                arch.movaTopKExperts, default: 0),
            routerScaling: extensionValue(
                arch.movaRouterScaling, default: 1),
            routerHasBias: extensionValue(
                arch.movaRouterHasBias, default: false),
            activation: extensionValue(
                arch.movaActivation, default: "silu"))
        return ManifestLayerShape(
            attentionGate: attentionGate,
            normGroups: normGroups,
            denseLayerIndices: denseLayerIndices,
            mova: mova,
            routerHasBias: extensionValue(
                arch.routerHasBias, default: false),
            routerScaling: extensionValue(
                arch.routerScaling, default: 1),
            // The norm-over-top-k field is declared per install; when it is
            // absent only the flash families default it on. An explicit
            // declaration always wins.
            routerNormTopK: arch.routerNormTopK ?? isFlash)
    }
}
