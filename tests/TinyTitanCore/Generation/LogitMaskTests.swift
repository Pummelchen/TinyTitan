import Testing

@testable import TinyTitanCore

/// The mask floors every id it does not allow — including padded rows.
///
/// The logits buffer can be wider than the tokenizer's vocabulary (the model's
/// padded embedding rows). `apply` looped to `min(count, vocab)`, so those tail
/// ids kept their model logits: the sampler could draw one, and the decode loop
/// then reported `samplerReturnedOutOfRangeToken` for an id the mask was
/// documented to have excluded (AUD-0119).
@Suite struct LogitMaskTests {
    @Test func allowedIdsAreUntouchedAndDisallowedIdsAreFloored() {
        var mask = LogitMask(vocab: 4)
        mask.insert(1)
        var half = [Float16](repeating: 0.5, count: 4)
        half.withUnsafeMutableBufferPointer { buffer in
            mask.apply(toLogits: buffer.baseAddress!, count: 4)
        }
        #expect(half[1] == 0.5)
        #expect(half[0] == LogitMaskFloor.half)
        #expect(half[2] == LogitMaskFloor.half)
        #expect(half[3] == LogitMaskFloor.half)

        var single = [Float](repeating: 0.5, count: 4)
        mask.apply(toLogits: &single)
        #expect(single[1] == 0.5)
        #expect(single[3] == LogitMaskFloor.single)
    }

    @Test func paddedRowsBeyondTheVocabularyAreFloored() {
        var mask = LogitMask(vocab: 4)
        mask.insert(1)
        var half = [Float16](repeating: 0.5, count: 6)
        half.withUnsafeMutableBufferPointer { buffer in
            mask.apply(toLogits: buffer.baseAddress!, count: 6)
        }
        #expect(half[4] == LogitMaskFloor.half)
        #expect(half[5] == LogitMaskFloor.half)

        var single = [Float](repeating: 0.5, count: 6)
        mask.apply(toLogits: &single)
        #expect(single[4] == LogitMaskFloor.single)
        #expect(single[5] == LogitMaskFloor.single)
    }

    @Test func theMaskRefusesIdsOutsideItsVocabulary() {
        let mask = LogitMask(vocab: 4)
        #expect(!mask.contains(4))
        #expect(!mask.contains(-1))
        var mutable = mask
        mutable.insert(4)
        #expect(mutable.isEmpty)
    }
}
