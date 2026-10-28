import Foundation

// The manifest quant types: the five width slots plus the per-tensor overrides.
//
// Split out of `SSDAIManifestV1.swift` (2026-10-25) as pure code motion under
// the 500-line-per-file rule.

package struct SSDAIManifestQuantSlotV1: Codable, Equatable, Sendable {
    /// Weight widths a reader in this project implements.
    ///
    /// The kernel arithmetic is `32 / bits` lanes per u32 word and `columns / 8`
    /// packed words per row at the 4-bit end, so a value outside this set is not
    /// a wider or narrower packing — it is a number nothing can decode. 6-bit
    /// was withdrawn (non-power-of-two packing measured 46.8 GB/s against 60 for
    /// 4-bit and 8-bit).
    package static let supportedWeightBits: Set<Int> = [4, 8]

    package let weightBits: Int
    package let scheme: String
    package let scaleType: String
    package let biasType: String
    package let groupSize: Int

    package init(
        weightBits: Int, scheme: String, scaleType: String,
        biasType: String, groupSize: Int
    ) {
        self.weightBits = weightBits
        self.scheme = scheme
        self.scaleType = scaleType
        self.biasType = biasType
        self.groupSize = groupSize
    }
}

package struct SSDAIManifestQuantV1: Codable, Equatable, Sendable {
    package let embedding: SSDAIManifestQuantSlotV1
    package let attention: SSDAIManifestQuantSlotV1
    package let router: SSDAIManifestQuantSlotV1
    package let sharedExpert: SSDAIManifestQuantSlotV1
    package let routedExpert: SSDAIManifestQuantSlotV1
    /// Per-tensor width overrides, keyed by tensor stem.
    ///
    /// The writer emits these beside the five slots and the decoder used to
    /// drop them, which mattered as soon as a reader needed them: a 4-bit
    /// build keeps its embedding and every attention K/V at 8 bits, and
    /// dequantizing those as 4-bit unpacks the same bytes wrongly. Absent in
    /// manifests written before this field existed, and in builds that have no
    /// overrides at all.
    package let overrides: [String: SSDAIManifestQuantSlotV1]?

    /// Tensor stems stored at the checkpoint's own bf16 inside a quantized
    /// slot, sorted.
    ///
    /// A separate list rather than `weightBits: 16`, because every reader that
    /// takes a width as arithmetic (`32 / bits` lanes, packed-row extents)
    /// would unpack such a tensor wrongly instead of failing; a width means a
    /// packing, and bf16 is the absence of one. The list says which tensors the
    /// slot describes but does not pack, and a reader that needs a kernel width
    /// resolves those to 16 by name.
    package let promoted: [String]?

    private enum CodingKeys: String, CodingKey {
        case embedding, attention, router, sharedExpert, routedExpert, promoted
    }

    /// Hand-written because the object mixes five fixed slots with an open set
    /// of per-tensor overrides keyed by tensor stem. A synthesised `Codable`
    /// silently dropped the open set, which is how a 4-bit install's 8-bit
    /// K/V got read back as 4-bit.
    package init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        embedding = try container.decode(SSDAIManifestQuantSlotV1.self, forKey: .embedding)
        attention = try container.decode(SSDAIManifestQuantSlotV1.self, forKey: .attention)
        router = try container.decode(SSDAIManifestQuantSlotV1.self, forKey: .router)
        sharedExpert = try container.decode(
            SSDAIManifestQuantSlotV1.self,
            forKey: .sharedExpert)
        routedExpert = try container.decode(
            SSDAIManifestQuantSlotV1.self,
            forKey: .routedExpert)
        self.promoted = try container.decodeIfPresent([String].self, forKey: .promoted)
        let dynamic = try decoder.container(keyedBy: AnyKey.self)
        var overrides: [String: SSDAIManifestQuantSlotV1] = [:]
        for key in dynamic.allKeys {
            guard CodingKeys(stringValue: key.stringValue) == nil else { continue }
            // `try`, not `try?`. A malformed override used to be dropped in
            // silence, which is precisely how a width goes missing: the reader
            // then falls back to a slot, and a slot that disagrees with the
            // payload unpacks the same bytes wrongly. Every key in this object
            // is one the writer emitted, so a key that does not decode is
            // corruption rather than an extension this build should tolerate.
            let slot = try dynamic.decode(SSDAIManifestQuantSlotV1.self, forKey: key)
            // A width no kernel implements would be read by arithmetic that
            // assumes 4 or 8 bits per value (`32 / bits` lanes, `columns / 8`
            // packed words). Refusing here is the difference between a load
            // error and fluent nonsense.
            guard SSDAIManifestQuantSlotV1.supportedWeightBits.contains(slot.weightBits) else {
                throw DecodingError.dataCorruptedError(
                    forKey: key, in: dynamic,
                    debugDescription: "quant override for \(key.stringValue) declares "
                        + "\(slot.weightBits) bits; supported: "
                        + "\(SSDAIManifestQuantSlotV1.supportedWeightBits.sorted())")
            }
            overrides[key.stringValue] = slot
        }
        self.overrides = overrides.isEmpty ? nil : overrides
    }

    package func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(embedding, forKey: .embedding)
        try container.encode(attention, forKey: .attention)
        try container.encode(router, forKey: .router)
        try container.encode(sharedExpert, forKey: .sharedExpert)
        try container.encode(routedExpert, forKey: .routedExpert)
        if let promoted {
            try container.encode(promoted, forKey: .promoted)
        }
        if let overrides {
            var dynamic = encoder.container(keyedBy: AnyKey.self)
            for (stem, slot) in overrides {
                guard let key = AnyKey(stringValue: stem) else {
                    throw TinyTitanFormatError.invalid(
                        field: "manifest.overrides.\(stem)", reason: "not encodable as a key")
                }
                try dynamic.encode(slot, forKey: key)
            }
        }
    }

    /// A decoder/encoder key for the arbitrary tensor stems in the object.
    private struct AnyKey: CodingKey {
        let stringValue: String
        var intValue: Int? { nil }
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { nil }
    }

    package init(
        embedding: SSDAIManifestQuantSlotV1,
        attention: SSDAIManifestQuantSlotV1,
        router: SSDAIManifestQuantSlotV1,
        sharedExpert: SSDAIManifestQuantSlotV1,
        routedExpert: SSDAIManifestQuantSlotV1,
        overrides: [String: SSDAIManifestQuantSlotV1]? = nil,
        promoted: [String]? = nil
    ) {
        self.overrides = overrides
        self.promoted = promoted
        self.embedding = embedding
        self.attention = attention
        self.router = router
        self.sharedExpert = sharedExpert
        self.routedExpert = routedExpert
    }
}
