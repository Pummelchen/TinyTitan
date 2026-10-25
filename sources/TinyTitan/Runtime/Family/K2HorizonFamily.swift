import Foundation

/// K2-Horizon's tensor-name contract.
///
/// The checkpoint is written without the `language_model.` prefix that the
/// Qwen repacks carry: the attention, the routed-experts router and the shared
/// expert live at `model.layers.N.*`, the embedding at `model.embed_tokens` and
/// the untied head at `lm_head`. The shared expert is `shared_experts` (plural)
/// and K2 has no per-head q/k norms and no Gated-DeltaNet, so those roles point
/// at names that cannot exist -- a lookup failure is the diagnosis, where a
/// plausible name would silently feed the wrong tensor.
///
/// Sparse layers replace `v_proj` with MoVA: a `v_router` over 64 value
/// experts. Those tensors are not in this schema because `TensorSchema` has no
/// role for them yet; the GPU family adds the roles when the MoVA stage exists.
extension TensorSchema {
    private static func k2Layer(_ layer: Int, _ suffix: String) -> String {
        "model.layers.\(layer).\(suffix)"
    }

    static let k2Horizon = TensorSchema(
        embedding: "model.embed_tokens.weight",
        lmHead: "lm_head.weight",
        finalNorm: "model.norm.weight",
        qProj: { k2Layer($0, "self_attn.q_proj.weight") },
        kProj: { k2Layer($0, "self_attn.k_proj.weight") },
        // Only the three dense layers carry a v_proj; sparse layers use MoVA.
        vProj: { k2Layer($0, "self_attn.v_proj.weight") },
        oProj: { k2Layer($0, "self_attn.o_proj.weight") },
        router: { k2Layer($0, "mlp.gate.weight") },
        sharedExpertGate: { k2Layer($0, "mlp.shared_experts.gate_proj.weight") },
        sharedExpertUp: { k2Layer($0, "mlp.shared_experts.up_proj.weight") },
        sharedExpertDown: { k2Layer($0, "mlp.shared_experts.down_proj.weight") },
        // K2's shared expert has no sigmoid scalar gate; the role cannot exist.
        sharedExpertScalarGate: { _ in "k2_horizon.has_no_shared_expert_gate" },
        inputNorm: { k2Layer($0, "input_layernorm.weight") },
        postAttnNorm: { k2Layer($0, "post_attention_layernorm.weight") },
        qNorm: { _ in "k2_horizon.has_no_q_norm" },
        kNorm: { _ in "k2_horizon.has_no_k_norm" },
        gdnQKV: { _ in "k2_horizon.has_no_gdn" },
        gdnZ: { _ in "k2_horizon.has_no_gdn" },
        gdnA: { _ in "k2_horizon.has_no_gdn" },
        gdnB: { _ in "k2_horizon.has_no_gdn" },
        gdnOut: { _ in "k2_horizon.has_no_gdn" },
        gdnConv: { _ in "k2_horizon.has_no_gdn" },
        gdnALog: { _ in "k2_horizon.has_no_gdn" },
        gdnDtBias: { _ in "k2_horizon.has_no_gdn" },
        gdnNorm: { _ in "k2_horizon.has_no_gdn" })
}
