import Foundation
import TinyTitanFormat

/// The `ArchInfo` of an Inkling 2 GGUF, read from the file's own `inkling2.*`
/// metadata rather than from a preset (TT-121); `Inkling2Arch` turns the
/// constants into the manifest's shape.
enum Inkling2GGUFArch {
    static func archInfo(_ header: GGUFHeaderV3) throws -> ArchInfo {
        func number(_ key: String) throws -> Double {
            guard let value = header.number("inkling2." + key) else {
                throw RepackError.configurationInvalid(
                    detail: "Inkling 2 GGUF lacks inkling2.\(key)")
            }
            return value
        }
        func int(_ key: String) throws -> Int {
            try Self.integer(try number(key), field: "inkling2.\(key)")
        }
        func firstOf(_ key: String) throws -> Int {
            if let list = header.numbers("inkling2." + key), let first = list.first {
                return try Self.integer(first, field: "inkling2.\(key)[0]")
            }
            return try int(key)
        }
        guard try int("rope.dimension_count") == 0 else {
            throw RepackError.configurationInvalid(
                detail: "Inkling 2 GGUF declares rotary dimensions; this install format has none")
        }
        let loopStart = try int("loop.core_start")
        let loopEnd = try int("loop.core_end")
        let passes = try int("num_loops")
        let unrolled = try int("block_count")
        guard loopStart >= 0, loopEnd > loopStart, passes >= 1 else {
            throw RepackError.configurationInvalid(
                detail: "Inkling 2 loop geometry \(loopStart)..<\(loopEnd) x \(passes) "
                    + "does not fit \(unrolled) layer visits")
        }
        let stored = try Inkling2Arch.storedLayers(
            unrolled: unrolled, loopStart: loopStart, loopEnd: loopEnd, passes: passes,
            field: "Inkling 2 loop geometry")
        guard stored > 0, loopEnd <= stored else {
            throw RepackError.configurationInvalid(
                detail: "Inkling 2 loop geometry \(loopStart)..<\(loopEnd) x \(passes) "
                    + "does not fit \(unrolled) layer visits")
        }
        return try Inkling2Arch.archInfo(
            Inkling2Arch.Parameters(
                hiddenSize: try int("embedding_length"),
                intermediateSize: try int("feed_forward_length"),
                expertIntermediateSize: try int("expert_feed_forward_length"),
                heads: try firstOf("attention.head_count"),
                kvHeads: try firstOf("attention.head_count_kv"),
                headDim: try int("attention.key_length"),
                paddedVocabulary: try int("vocab_size"),
                vocabulary: try int("vocab_size_unpadded"),
                slidingWindow: try int("attention.sliding_window"),
                finalLogitSoftcap: try number("final_logit_softcapping"),
                storedLayers: stored,
                experts: try int("expert_count"),
                expertsPerToken: try int("expert_used_count"),
                localLayers: try slidingLayers(
                    header, stored: stored, loopStart: loopStart, loopEnd: loopEnd,
                    passes: passes),
                tiedEmbeddings: !header.tensors.contains { $0.name == "output.weight" },
                loopStart: loopStart, loopEnd: loopEnd, loopPasses: passes,
                loopGateRank: try number("loop.gate_rank"),
                attentionResidualBlock: try number("attn_res.block_size"),
                relativeLength: try number("attention.relative_length"),
                relativeExtent: try number("attention.relative_extent"),
                relativeClip: try number("attention.relative_clip"),
                convolutionTaps: try number("shortconv.l_cache"),
                rmsEpsilon: try number("attention.layer_norm_rms_epsilon"),
                qkClip: try number("attention.qk_clip"),
                logScaleFloor: try number("attention.log_scale_floor"),
                logScaleAlpha: try number("attention.log_scale_alpha"),
                expertWeightsScale: try number("expert_weights_scale"),
                situBeta: try number("activation.situ_beta"),
                situLinearBeta: try number("activation.situ_linear_beta"),
                contextLength: try number("context_length"),
                quantGroupSize: 32))
    }

    /// A metadata value as an exact `Int`, or a typed error.
    ///
    /// `Int(exactly:)` rejects NaN, ±infinity, a fractional value and anything
    /// outside `Int`; a plain `Int(_: Double)` traps on all of those, and the
    /// values come from a downloaded GGUF's metadata.
    static func integer(_ value: Double, field: String) throws -> Int {
        guard let exact = Int(exactly: value) else {
            throw RepackError.configurationInvalid(
                detail: "Inkling 2 GGUF \(field) is \(value), not an exact integer")
        }
        return exact
    }

    /// The stored layers that use sliding-window attention. The GGUF's pattern
    /// lists one flag per layer *visit* (48); a stored layer's flag is the one
    /// at its first visit — the loop repeats layers, it does not change them.
    static func slidingLayers(
        _ header: GGUFHeaderV3, stored: Int, loopStart: Int, loopEnd: Int, passes: Int
    ) throws -> [Int] {
        guard let pattern = header.numbers("inkling2.attention.sliding_window_pattern") else {
            throw RepackError.configurationInvalid(
                detail: "Inkling 2 GGUF lacks inkling2.attention.sliding_window_pattern")
        }
        return try slidingLayers(
            pattern: pattern, stored: stored, loopStart: loopStart, loopEnd: loopEnd,
            passes: passes)
    }

    /// The same mapping over an explicit pattern, so the loop arithmetic can be
    /// tested without a header.
    static func slidingLayers(
        pattern: [Double], stored: Int, loopStart: Int, loopEnd: Int, passes: Int
    ) throws -> [Int] {
        let extraLayers = try Inkling2Arch.extraLoopLayers(
            loopStart: loopStart, loopEnd: loopEnd, passes: passes,
            field: "sliding-window loop geometry")
        return try (0..<stored).filter { layer in
            // Layers before the loop's end first appear at their own index; a
            // layer after it follows the core's extra passes.
            let visit: Int
            if layer < loopEnd {
                visit = layer
            } else {
                visit = try Inkling2Arch.checkedAdd(
                    layer, extraLayers, field: "sliding-window visit")
            }
            guard visit < pattern.count else {
                throw RepackError.configurationInvalid(
                    detail: "sliding_window_pattern has \(pattern.count) entries, no visit \(visit)"
                )
            }
            return pattern[visit] != 0
        }
    }
}
