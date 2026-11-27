import Foundation
import Testing

@testable import TinyTitanCore

/// The indexer is enabled only when both halves of its geometry are present.
///
/// `enabled` used to be `budget > 0` alone, while the pooling divides by
/// `compressRatio` — so a manifest naming a budget and no ratio produced an
/// enabled indexer with a zero divisor: a trap in `QSAIndexer`, and an
/// integer divide in `qsa.metal` (AUD-0093). A missing half now means the
/// dense attention path, which is slower and correct.
@Suite struct SparseIndexerConfigTests {
    @Test func aBudgetWithoutARatioIsDisabled() {
        let half = SparseIndexerConfig(
            numHeads: 8, numKVHeads: 2, headDim: 128, budget: 2_048, compressRatio: 0)
        #expect(!half.enabled)
        #expect(SparseIndexerConfig.none.enabled == false)
    }

    @Test func anOrdinaryConfigIsEnabled() {
        let full = SparseIndexerConfig(
            numHeads: 8, numKVHeads: 2, headDim: 128, budget: 2_048, compressRatio: 4)
        #expect(full.enabled)
    }
}
