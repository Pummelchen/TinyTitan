import Foundation

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
        if let preset = knownArchitectures[family] { return preset }
        // The families without a preset whose architecture is fully declared by
        // the manifest. Anything else keeps the previous behaviour: refused by
        // name, because there is no preset to load it at.
        guard family == .qwen35Dense || family == .k2Horizon else {
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
        let linear = LinearAttentionConfig(
            numKHeads: try required(arch.linearNumKHeads, "linearNumKHeads"),
            numVHeads: try required(arch.linearNumVHeads, "linearNumVHeads"),
            keyHeadDim: try required(arch.linearKeyHeadDim, "linearKeyHeadDim"),
            valueHeadDim: try required(arch.linearValueHeadDim, "linearValueHeadDim"),
            convKernelSize: try required(arch.linearConvKernelSize, "linearConvKernelSize"),
            // Qwen 3.5 and 3.6 are the same lineage; Qwen3.8 is the family that
            // states `sigmoid`. Declared here rather than defaulted from the
            // other one, which is the point the config type makes.
            outputGate: .silu)
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
            family: family,
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
            // K2 normalizes the selected sigmoid scores over the top-k
            // (`norm_topk_prob` is true in its config). See `layerShape`.
            routerNormTopK: shape.routerNormTopK,
            quantGroupSize: arch.quantGroupSize ?? 64,
            normGroups: shape.normGroups,
            attentionGate: shape.attentionGate,
            routerHasBias: shape.routerHasBias,
            routerScaling: shape.routerScaling,
            denseLayerIndices: shape.denseLayerIndices,
            mova: shape.mova)
    }

    /// The family-dependent layer shape a manifest declares (K2's block).
    ///
    /// Split out of `from` under the 120-line gate. K2 requires every field by
    /// name; a family without them reads what it declares and defaults the
    /// rest, which keeps the dense manifests loading unchanged.
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
        func required<T>(_ value: T?, _ field: String) throws -> T {
            guard let value else {
                throw ArchResolutionError.manifestOmitsField(
                    family: family.rawValue, field: field)
            }
            return value
        }
        let isK2 = family == .k2Horizon
        func extensionValue<T>(
            _ value: T?, _ field: String, default fallback: @autoclosure () -> T
        ) throws -> T {
            if isK2 { return try required(value, field) }
            return value ?? fallback()
        }
        let gateName = try extensionValue(
            arch.attentionGate, "attentionGate", default: "sigmoid")
        guard let attentionGate = AttentionGateFunction(rawValue: gateName) else {
            throw ArchResolutionError.manifestDeclaresInvalidValue(
                family: family.rawValue, field: "attentionGate", value: gateName)
        }
        let normGroups = try extensionValue(
            arch.normGroups, "normGroups", default: 1)
        guard normGroups > 0, arch.hiddenSize % normGroups == 0 else {
            throw ArchResolutionError.manifestDeclaresInvalidValue(
                family: family.rawValue, field: "normGroups",
                value: "\(normGroups) (hiddenSize \(arch.hiddenSize))")
        }
        let denseLayerIndices = try extensionValue(
            arch.denseLayerIndices, "denseLayerIndices", default: [])
        guard denseLayerIndices.allSatisfy({ $0 >= 0 && $0 < arch.numLayers }),
            Set(denseLayerIndices).count == denseLayerIndices.count,
            denseLayerIndices == denseLayerIndices.sorted()
        else {
            throw ArchResolutionError.manifestDeclaresInvalidValue(
                family: family.rawValue, field: "denseLayerIndices",
                value: "\(denseLayerIndices)")
        }
        let mova = MoVAConfig(
            numExperts: try extensionValue(
                arch.movaNumExperts, "movaNumExperts", default: 0),
            topKExperts: try extensionValue(
                arch.movaTopKExperts, "movaTopKExperts", default: 0),
            routerScaling: try extensionValue(
                arch.movaRouterScaling, "movaRouterScaling", default: 1),
            routerHasBias: try extensionValue(
                arch.movaRouterHasBias, "movaRouterHasBias", default: false),
            activation: try extensionValue(
                arch.movaActivation, "movaActivation", default: "silu"))
        if isK2 {
            guard mova.numExperts > 0, mova.topKExperts > 0,
                mova.topKExperts <= mova.numExperts
            else {
                throw ArchResolutionError.manifestDeclaresInvalidValue(
                    family: family.rawValue, field: "movaNumExperts/topK",
                    value: "\(mova.numExperts)/\(mova.topKExperts)")
            }
        }
        return ManifestLayerShape(
            attentionGate: attentionGate,
            normGroups: normGroups,
            denseLayerIndices: denseLayerIndices,
            mova: mova,
            routerHasBias: try extensionValue(
                arch.routerHasBias, "routerHasBias", default: false),
            routerScaling: try extensionValue(
                arch.routerScaling, "routerScaling", default: 1),
            // K2 normalizes the selected sigmoid scores over the top-k
            // (`norm_topk_prob` is true in its config). The field was not
            // written into older K2 manifests, and defaulting it to false would
            // hand the router unnormalized scores -- a fluent, wrong mixture
            // rather than an error. An explicit declaration still wins.
            routerNormTopK: arch.routerNormTopK ?? isK2)
    }
}
