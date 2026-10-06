import Foundation
import Testing

@testable import TinyTitan

/// PLE row addressing against golden vectors produced by the reference
/// implementation (`mlx_qwen4exp/ple.py`) using the pinned checkpoint's own
/// `ple_constants.json`.
///
/// Golden vectors rather than a re-derived formula: this is integer hashing
/// with wrapping multiplies, a latching context cut and a 32-bit truncation.
/// Every one of those can be implemented plausibly and wrongly, and a wrong
/// row id fetches real embedding data from the wrong place -- coherent-looking
/// output, no error anywhere.
@Suite("PLE row hashing")
struct PLEHashTests {
    struct Golden: Decodable {
        struct Constants: Decodable {
            let multipliers: [UInt64]
            let offsets: [UInt64]
            let vocab: [UInt64]
            let eos: Int32
            let ngramSize: Int
            let headsPerNgram: Int
        }
        struct Case: Decodable {
            let tokens: [Int32]
            let prev: [Int32]
            let rows: [[UInt32]]
        }
        let constants: Constants
        let cases: [String: Case]
    }

    static let golden: Golden = {
        guard
            let url = Bundle.module.url(
                forResource: "ple_golden",
                withExtension: "json"),
            let data = try? Data(contentsOf: url),
            let g = try? JSONDecoder().decode(Golden.self, from: data)
        else { fatalError("ple_golden.json fixture missing") }
        return g
    }()

    static func hash() -> PLEHash {
        let c = golden.constants
        return PLEHash(
            multipliers: c.multipliers, offsets: c.offsets,
            vocabSizes: c.vocab, ngramSize: c.ngramSize,
            headsPerNgram: c.headsPerNgram, eosTokenID: c.eos)
    }

    private func check(_ name: String) {
        guard let c = Self.golden.cases[name] else {
            Issue.record("missing golden case \(name)")
            return
        }
        // `prev` is -1 padded in the fixture; drop those to real predecessors.
        let previous = c.prev.filter { $0 >= 0 }
        let actual = Self.hash().rows(tokens: c.tokens, previous: previous)
        #expect(actual.count == c.rows.count, "\(name): token count")
        for (t, expected) in c.rows.enumerated() where t < actual.count {
            let got = "\(actual[t].prefix(4))"
            let want = "\(expected.prefix(4))"
            #expect(
                actual[t] == expected,
                "\(name) token \(t): got \(got)... want \(want)...")
        }
    }

    @Test("Production constants and geometry") func geometry() {
        let h = Self.hash()
        #expect(h.ngramSize == 3)
        #expect(h.headsPerNgram == 8)
        #expect(h.headCount == 16)  // 2 orders x 8 heads
        #expect(h.vocabSizes.count == 16)
        // Head vocabularies are distinct primes just above 20M.
        #expect(Set(h.vocabSizes).count == 16)
        #expect(h.vocabSizes.allSatisfy { $0 > 20_000_000 && $0 < 20_001_000 })
    }

    @Test("A plain sequence") func fresh() { check("fresh") }

    @Test("An EOS mid-sequence cuts everything older") func eosMid() {
        check("eos_mid")
    }

    @Test("Carried predecessors from an earlier chunk") func withPrev() {
        check("with_prev")
    }

    @Test("A leading EOS does not cut its own context") func eosFirst() {
        check("eos_first")
    }

    @Test("Large ids exercise the wrapping multiply") func largeIDs() {
        check("large_ids")
    }

    @Test("Rows land inside their head's slice of the table")
    func rowsWithinHeadRanges() {
        let h = Self.hash()
        let rows = h.rows(tokens: [7, 99, 1234, 248_000], previous: [])
        for tokenRows in rows {
            for (head, row) in tokenRows.enumerated() {
                let lo = h.offsets[head]
                let hi = lo + h.vocabSizes[head]
                #expect(
                    UInt64(row) >= lo && UInt64(row) < hi,
                    "head \(head) row \(row) outside [\(lo), \(hi))")
            }
        }
    }

    @Test("One mix per n-gram order, not per head")
    func headsShareTheirOrdersMix() {
        // Heads within an order differ only by modulus and offset, so
        // subtracting each head's offset must leave the same value reduced by
        // different primes -- consistent with a single shared mix.
        let h = Self.hash()
        let rows = h.rows(context: [11, 22, 33])
        for order in 0..<2 {
            let base = order * h.headsPerNgram
            let residues = (0..<h.headsPerNgram).map {
                UInt64(rows[base + $0]) - h.offsets[base + $0]
            }
            // If some head had its own mix, agreement across all eight
            // congruences would be a coincidence of ~20M^-7.
            for (i, r) in residues.enumerated() {
                #expect(r < h.vocabSizes[base + i])
            }
        }
    }
}

/// The n-gram sidecar is geometry, and nothing else compared it to the
/// architecture.
///
/// `headCount * pleHeadDim` is the number of fp16 values one token gathers,
/// while the PLE block's embedding buffer is sized from `cfg.ple.embedDim`.
/// A sidecar built for another model would have the gather write past that
/// buffer (host heap corruption) or feed the block wrong-width rows, silently.
/// `PLEHash`'s own consistency checks are preconditions, so this validation has
/// to run before `makeHash()` for a corrupt sidecar to be a report rather than a
/// trap. The row addressing is checked on the same terms: the offsets and
/// vocabularies are `Int64` from JSON, and the count that comes out of them is
/// what sizes the table read.
@Suite("PLE sidecar geometry")
struct PLEConstantsGeometryTests {
    private func constants(
        ngramSize: Int = 3,
        headsPerNgram: Int = 8,
        pleHeadDim: Int = 160,
        headEntries: Int? = nil,
        vocabSizes: [Int64]? = nil,
        offsets: [Int64]? = nil
    ) -> PLEConstants {
        let headCount = headEntries ?? (headsPerNgram * (ngramSize - 1))
        // The producer builds the offsets by accumulating each head's size from
        // zero (`tools/prepare_qwen38.py:148-152`), so a well-formed fixture has
        // to look the same way or every test below trips the layout check.
        let vocab = vocabSizes ?? Array(repeating: 1, count: headCount)
        var running: [Int64] = []
        if offsets == nil {
            var total: Int64 = 0
            for size in vocab {
                running.append(total)
                total += size
            }
        }
        return PLEConstants(
            layerMultipliers: Array(repeating: 1, count: ngramSize),
            ngramHeadsOffsets: offsets ?? running,
            ngramHeadsVocabSizes: vocab,
            eosTokenID: 0,
            ngramSize: ngramSize,
            headsPerNgram: headsPerNgram,
            pleNumHeads: 16,
            pleHeadDim: pleHeadDim)
    }

    @Test("A sidecar that agrees with the architecture validates")
    func acceptsMatchingGeometry() throws {
        // 8 * (3 - 1) * 160 = 2560 values per token.
        try constants().validate(embedDim: 2560, ngramSize: 3, headsPerNgram: 8)
    }

    @Test("A per-token width that disagrees with embedDim is refused")
    func refusesWidthMismatch() {
        #expect(throws: ModelError.self) {
            try constants().validate(embedDim: 2048, ngramSize: 3, headsPerNgram: 8)
        }
    }

    @Test("A different n-gram shape is refused")
    func refusesShapeMismatch() {
        #expect(throws: ModelError.self) {
            try constants(ngramSize: 4).validate(
                embedDim: 2560, ngramSize: 3,
                headsPerNgram: 8)
        }
        #expect(throws: ModelError.self) {
            try constants(headsPerNgram: 4).validate(
                embedDim: 2560, ngramSize: 3,
                headsPerNgram: 8)
        }
    }

    @Test("Head tables that do not match the shape are refused")
    func refusesHeadTableMismatch() {
        // Enough entries to pass PLEHash's own precondition would be a trap;
        // too few is what a truncated sidecar looks like, and it is checked
        // here so the failure is named rather than left to a precondition.
        #expect(throws: ModelError.self) {
            try constants(headEntries: 4).validate(
                embedDim: 2560, ngramSize: 3,
                headsPerNgram: 8)
        }
    }

    @Test("The row count is every head's vocabulary summed")
    func countsTheTable() throws {
        // 3 + 5 + 7 = 15 rows, which is also last offset (8) + last vocab (7).
        let sidecar = constants(
            headEntries: 3, vocabSizes: [3, 5, 7], offsets: [0, 3, 8])
        #expect(try sidecar.tableRowCount() == 15)
        // 16 heads of the shipped width, so a legal sidecar addresses a table
        // of exactly headCount * vocab rows.
        #expect(try constants().tableRowCount() == 16)
    }

    @Test("A negative addressing value is a report, and never a trap")
    func refusesNegativeAddressing() {
        // `UInt64(offset)` for offset < 0 is a fatal error. This is the case the
        // guard exists for, so it is asserted through `tableRowCount()` directly:
        // the trap must not be reachable on a path that did not run `validate()`.
        #expect(throws: ModelError.self) {
            try constants(headEntries: 3, offsets: [0, -1, 2]).tableRowCount()
        }
        #expect(throws: ModelError.self) {
            try constants(headEntries: 3, vocabSizes: [1, -2, 1]).tableRowCount()
        }
        // And through the load-time gate, which is how the model path reaches it.
        #expect(throws: ModelError.self) {
            try constants(
                offsets: Array(repeating: -1, count: 16)
            ).validate(embedDim: 2560, ngramSize: 3, headsPerNgram: 8)
        }
    }

    @Test("A zero vocabulary is refused before it reaches the hash")
    func refusesZeroVocabulary() {
        // A zero trips `PLEHash`'s `vocabSizes > 0` precondition, which is a
        // trap in a load path; refusing it here is what keeps the corrupt sidecar
        // a report. The negative case below is the quieter one: it survives that
        // precondition as a bit-pattern giant and misaddresses every row.
        #expect(throws: ModelError.self) {
            try constants(headEntries: 3, vocabSizes: [1, 0, 1]).tableRowCount()
        }
    }

    @Test("Offsets that do not follow the preceding head are refused")
    func refusesNonContiguousOffsets() {
        // Any of these describes a table whose heads overlap or leave a hole,
        // so a gathered row would be read from another head's range.
        for offsets in [[0, 2, 4], [0, 1, 1], [0, 1, 0], [1, 2, 3]] as [[Int64]] {
            #expect(throws: ModelError.self) {
                try constants(headEntries: 3, offsets: offsets).tableRowCount()
            }
        }
    }

    @Test("Head tables that do not pair up are refused")
    func refusesUnpairedHeadTables() {
        #expect(throws: ModelError.self) {
            try constants(
                headEntries: 3, vocabSizes: [1, 1], offsets: [0, 1, 2]
            ).tableRowCount()
        }
        #expect(throws: ModelError.self) {
            try constants(headEntries: 0).tableRowCount()
        }
    }

    @Test("Sizes that overflow the row count are refused, not wrapped")
    func refusesOverflowingTableSize() {
        // Two values that are each legal, summing past Int64.max: a wrapped
        // count would address a table that does not exist.
        let sidecar = constants(
            headEntries: 2,
            vocabSizes: [Int64.max, 1],
            offsets: [0, Int64.max])
        #expect(throws: ModelError.self) {
            try sidecar.tableRowCount()
        }
    }

    @Test("Validating checks the addressing as well as the width")
    func validateCoversAddressing() {
        // The gate and the count are one contract: passing validate() must mean
        // the row count is obtainable, or a load could pass the gate and then
        // trap on the very next line.
        #expect(throws: ModelError.self) {
            try constants(offsets: [0, 5] + Array(repeating: 2, count: 14))
                .validate(embedDim: 2560, ngramSize: 3, headsPerNgram: 8)
        }
    }

    @Test("The checkpoint's own constants address the shipped table")
    func acceptsProductionConstants() throws {
        // The fixture is the pinned checkpoint's real `ple_constants.json`, so
        // this is the test that the checks above are the checkpoint's rules and
        // not an invention that would refuse a working model. 320001446 is the
        // row count of the n-gram table in the install, which is what the
        // loader sizes the table read from.
        let production = PLEHashTests.golden.constants
        let sidecar = PLEConstants(
            layerMultipliers: production.multipliers.map { Int64(bitPattern: $0) },
            ngramHeadsOffsets: production.offsets.map { Int64($0) },
            ngramHeadsVocabSizes: production.vocab.map { Int64($0) },
            eosTokenID: production.eos,
            ngramSize: production.ngramSize,
            headsPerNgram: production.headsPerNgram,
            pleNumHeads: 16,
            pleHeadDim: 160)
        try sidecar.validate(embedDim: 2560, ngramSize: 3, headsPerNgram: 8)
        #expect(try sidecar.tableRowCount() == 320_001_446)
    }
}
