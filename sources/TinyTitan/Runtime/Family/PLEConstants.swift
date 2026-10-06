import Foundation

/// The hash constants that accompany Qwen3.8-Flash-Next's n-gram table.
///
/// These ship as a sidecar JSON rather than tensors because they are integer
/// parameters of the row addressing, not weights: `PLEHash` needs them before
/// any GPU work, and re-deriving them from a seed would be a second
/// implementation of something the checkpoint already states.
public struct PLEConstants: Decodable, Sendable {
    public let layerMultipliers: [Int64]
    public let ngramHeadsOffsets: [Int64]
    public let ngramHeadsVocabSizes: [Int64]
    public let eosTokenID: Int32
    public let ngramSize: Int
    public let headsPerNgram: Int
    public let pleNumHeads: Int
    public let pleHeadDim: Int

    enum CodingKeys: String, CodingKey {
        case layerMultipliers = "layer_multipliers"
        case ngramHeadsOffsets = "ngram_heads_offsets"
        case ngramHeadsVocabSizes = "ngram_heads_vocab_sizes"
        case eosTokenID = "eos_token_id"
        case ngramSize = "ngram_size"
        case headsPerNgram = "heads_per_ngram"
        case pleNumHeads = "ple_n_heads"
        case pleHeadDim = "ple_head_dim"
    }

    /// The sidecar shares the manifest's bound rather than carrying a second,
    /// unrelated one that can drift below it, as `VerifiedInstallReceiptReader`
    /// does for the same reason. See `ManifestReader.defaultMaxBytes`.
    public static let defaultMaxBytes: UInt64 = ManifestReader.defaultMaxBytes

    public static func load(
        directoryURL: URL,
        maxBytes: UInt64 = defaultMaxBytes
    ) throws -> PLEConstants {
        let name = Qwen38FlashTensors.pleConstantsFile
        // Root-anchored and bounded before the allocation. `Data(contentsOf:)`
        // grows a buffer as it reads, so an uncapped read of a directory that
        // may have been copied off another machine is an unbounded allocation,
        // and this file is a few hundred KB in a real install.
        let data = try SSDAIModelDirectory(rootURL: directoryURL)
            .readMetadata(name, maxBytes: maxBytes)
        return try JSONDecoder().decode(PLEConstants.self, from: data)
    }

    /// Row count of the table these constants address: the last head's offset
    /// plus its own vocabulary, which is every head's vocabulary summed.
    ///
    /// Throws rather than traps, and is the only way to ask. Both halves of the
    /// sum are `Int64`s read out of `ple_constants.json`, and `UInt64(-1)` is a
    /// fatal error, so a corrupt sidecar has to be a report on this path too --
    /// `validate()` cannot be the only guard, because a `public` property cannot
    /// insist that some other method ran before it.
    ///
    /// The layout is what the producer states, not an invention:
    /// `tools/prepare_qwen38.py:148-152` appends the running total before adding
    /// each head's size, so the offsets ascend by exactly the preceding
    /// vocabulary from a zero base. A negative value, a zero vocabulary or a gap
    /// therefore means the file does not describe the table it claims to, and
    /// each reaches `PLEHash` differently: a zero trips its `vocabSizes > 0`
    /// precondition, and a negative survives that check as the bit-pattern giant
    /// `UInt64(bitPattern:)` makes of it, so every row of every head hashes into
    /// the wrong place with no error anywhere.
    ///
    /// - Throws: `ModelError.archMismatch` naming the head whose addressing is
    ///     wrong, or a head table that does not pair up or has no entries.
    public func tableRowCount() throws -> UInt64 {
        guard ngramHeadsOffsets.count == ngramHeadsVocabSizes.count else {
            throw ModelError.archMismatch(
                field: "ple_constants.json head tables",
                expected: "one vocab size per offset",
                actual: "\(ngramHeadsOffsets.count) offsets, "
                    + "\(ngramHeadsVocabSizes.count) vocab sizes")
        }
        guard !ngramHeadsVocabSizes.isEmpty else {
            throw ModelError.archMismatch(
                field: "ple_constants.json head tables",
                expected: "at least one head set",
                actual: "no entries")
        }
        var rows: Int64 = 0
        for index in ngramHeadsOffsets.indices {
            let offset = ngramHeadsOffsets[index]
            let vocab = ngramHeadsVocabSizes[index]
            guard offset == rows else {
                throw ModelError.archMismatch(
                    field: "ple_constants.json offset[\(index)]",
                    expected: "\(rows)",
                    actual: "\(offset)")
            }
            guard vocab > 0 else {
                throw ModelError.archMismatch(
                    field: "ple_constants.json vocab_size[\(index)]",
                    expected: "> 0",
                    actual: "\(vocab)")
            }
            // Checked rather than `+=`: two legal-looking Int64 sizes can pass a
            // sign test and still overflow, and the wrapped value would address
            // a table that does not exist.
            let (next, overflow) = rows.addingReportingOverflow(vocab)
            guard !overflow else {
                throw ModelError.archMismatch(
                    field: "ple_constants.json table size",
                    expected: "<= Int64.max rows",
                    actual: "overflow at head[\(index)]")
            }
            rows = next
        }
        return UInt64(rows)
    }

    /// Check the sidecar's geometry against the architecture that will use it.
    ///
    /// `headCount * pleHeadDim` is the number of fp16 values one token gathers
    /// (`PLEHash.headCount` rows of the table's row width), and the PLE block's
    /// embedding buffer is sized from `cfg.ple.embedDim`. Nothing else compares
    /// the two: a sidecar from another model would have the gather write past
    /// that buffer -- host heap corruption, not a GPU fault -- or feed the block
    /// rows of the wrong width, silently. `PLEHash`'s own consistency checks are
    /// preconditions, so this must run *before* `makeHash()` to turn a corrupt
    /// sidecar into a report rather than a trap.
    ///
    /// It also runs `tableRowCount()`, because the row addressing is part of the
    /// same geometry: a negative offset or a zero vocabulary passes every count
    /// and product check above, and reaches either a trap or a wrong table.
    public func validate(
        embedDim: Int,
        ngramSize: Int,
        headsPerNgram: Int
    ) throws {
        let headCount = headsPerNgram * (self.ngramSize - 1)
        guard self.ngramSize == ngramSize else {
            throw ModelError.archMismatch(
                field: "ple.ngramSize",
                expected: "\(ngramSize)",
                actual: "\(self.ngramSize)")
        }
        guard self.headsPerNgram == headsPerNgram else {
            throw ModelError.archMismatch(
                field: "ple.headsPerNgram",
                expected: "\(headsPerNgram)",
                actual: "\(self.headsPerNgram)")
        }
        guard ngramHeadsOffsets.count == headCount,
            ngramHeadsVocabSizes.count == headCount
        else {
            throw ModelError.archMismatch(
                field: "ple_constants.json head tables",
                expected: "\(headCount) entries each",
                actual: "\(ngramHeadsOffsets.count) offsets, "
                    + "\(ngramHeadsVocabSizes.count) vocab sizes")
        }
        guard headCount * pleHeadDim == embedDim else {
            throw ModelError.archMismatch(
                field: "ple geometry (headCount * pleHeadDim)",
                expected: "\(embedDim) values per token",
                actual: "\(headCount) * \(pleHeadDim) = \(headCount * pleHeadDim)")
        }
        _ = try tableRowCount()
    }

    public func makeHash() -> PLEHash {
        PLEHash(
            multipliers: layerMultipliers.map { UInt64(bitPattern: $0) },
            offsets: ngramHeadsOffsets.map { UInt64(bitPattern: $0) },
            vocabSizes: ngramHeadsVocabSizes.map { UInt64(bitPattern: $0) },
            ngramSize: ngramSize,
            headsPerNgram: headsPerNgram,
            eosTokenID: eosTokenID)
    }
}
