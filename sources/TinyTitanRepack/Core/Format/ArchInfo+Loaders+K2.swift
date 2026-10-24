import Foundation

// K2-Horizon's ArchInfo loader, split out of ArchInfo+Loaders.swift under the
// 500-line-per-file rule (file-length gate).

extension ArchInfo {
    /// K2-Horizon (`model_type == "k2_horizon"`): IFM's 36B-A4B sparse model.
    ///
    /// Everything that differs from the Qwen MoE contract is recorded here:
    /// 100 routed experts at top-8 with a selection bias and a 2.5 router
    /// scale, a 2-way grouped RMS norm, a softplus attention gate, and 64
    /// MoVA value experts at top-4 inside every sparse layer. The dense Qwen
    /// fields stay at their inert values, because this family has no
    /// gated-DeltaNet layers and no MTP head.
    static func loadK2Horizon(
        configPath: String,
        tc: [String: Any]
    ) throws -> ArchInfo {
        func i(_ k: String) throws -> Int {
            guard let n = (tc[k] as? Int) ?? (tc[k] as? NSNumber)?.intValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return n
        }
        func d(_ k: String) throws -> Double {
            guard let n = (tc[k] as? Double) ?? (tc[k] as? NSNumber)?.doubleValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return n
        }
        let headDim = try i("head_dim")
        let ropeHeadDim = (tc["rope_head_dim"] as? Int) ?? headDim
        guard ropeHeadDim > 0, ropeHeadDim <= headDim else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "rope_head_dim \(ropeHeadDim) is outside 1...\(headDim)")
        }
        let layers = try i("num_hidden_layers")
        let groups = (tc["layernorm_num_groups"] as? Int) ?? 1
        let hidden = try i("hidden_size")
        guard groups > 0, hidden % groups == 0 else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "hidden_size \(hidden) is not divisible by layernorm groups \(groups)")
        }
        let rope = (tc["rope_parameters"] as? [String: Any]) ?? [:]
        let theta =
            (rope["rope_theta"] as? Double)
            ?? (rope["rope_theta"] as? NSNumber)?.doubleValue
            ?? 1_000_000
        let gate = (tc["attention_gate_func"] as? String) ?? "none"
        guard ["none", "sigmoid", "silu", "softplus"].contains(gate) else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "unknown attention_gate_func \(gate)")
        }
        let routerScaling = (tc["router_scaling_factor"] as? Double) ?? 1
        let routerBias = (tc["moe_gate_bias"] as? Bool) ?? false
        let denseLayers = (tc["mlp_only_layers"] as? [Int]) ?? []
        return ArchInfo(
            hiddenSize: hidden,
            intermediateSize: try i("intermediate_size"),
            moeIntermediateSize: try i("moe_intermediate_size"),
            numHeads: try i("num_attention_heads"),
            numKVHeads: try i("num_key_value_heads"),
            numFullKVHeads: try i("num_key_value_heads"),
            headDim: headDim,
            fullHeadDim: headDim,
            vocabSize: try i("vocab_size"),
            slidingWindow: 0,
            finalLogitSoftcap: 0,
            ropeTheta: theta,
            fullRopeTheta: theta,
            partialRotaryFactor: Double(ropeHeadDim) / Double(headDim),
            numLayers: layers,
            numExperts: try i("num_experts"),
            topKExperts: try i("num_experts_per_tok"),
            tieWordEmbeddings: (tc["tie_word_embeddings"] as? Bool) ?? false,
            attentionKEqV: false,
            fullAttentionLayerMask: [UInt8](repeating: 1, count: layers),
            hiddenActivation: (tc["hidden_act"] as? String) ?? "silu",
            family: .k2Horizon,
            attnOutputGate: gate != "none",
            attentionScale: 1.0 / Double(headDim).squareRoot(),
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: true,
            ropeNeoxSubdim: true,
            linearNumKHeads: 0,
            linearNumVHeads: 0,
            linearKeyHeadDim: 0,
            linearValueHeadDim: 0,
            linearConvKernelSize: 0,
            routerNormTopK: (tc["norm_topk_prob"] as? Bool) ?? false,
            quantGroupSize: 64,
            movaNumExperts: try i("mova_num_experts"),
            movaTopKExperts: try i("mova_num_experts_per_tok"),
            normGroups: groups,
            attentionGate: gate,
            routerScaling: routerScaling,
            routerHasBias: routerBias,
            movaRouterScaling: routerScaling,
            movaRouterHasBias: routerBias,
            movaActivation: "silu",
            denseLayerIndices: denseLayers)
    }
}
