import Testing

import TinyTitanValidationSupport

/// The chunked-prefill reference over a chunk with padded query positions.
///
/// The reference builds its key range as `first..<last` from the query position,
/// the sliding window and `kvValid`. A trailing query whose window has already
/// passed the last valid key makes `first > last`, and that range is a runtime
/// trap rather than a row of zeros — while `AttentionRef` already treats an
/// empty score row as a zero output (AUD-0115).
@Suite struct PrefillAttentionRefTests {
    @Test func aQueryPastTheWindowedKeyRangeIsAZeroRow() {
        let inputs = PrefillAttentionRef.Inputs(
            q: [Float](repeating: 1, count: 4),
            k: [1, 0],
            v: [1, 0],
            qStride: 2,
            kvStride: 2,
            oStride: 2,
            headDim: 2,
            qHeads: 1,
            kvHeads: 1,
            start: 10,
            chunk: 2,
            kvValid: 1,
            window: 1,
            scale: 0.5
        )
        let out = PrefillAttentionRef.apply(inputs)
        #expect(out == [Float](repeating: 0, count: 4))
    }

    @Test func aNormalChunkStillAttends() {
        // One query at position 1 over two valid keys, window 0 (unbounded):
        // the softmax picks the matching key, so the output is that V row.
        let inputs = PrefillAttentionRef.Inputs(
            q: [1, 0],
            k: [1, 0, 1, 0],
            v: [0.5, 0.25, 0.5, 0.25],
            qStride: 2,
            kvStride: 2,
            oStride: 2,
            headDim: 2,
            qHeads: 1,
            kvHeads: 1,
            start: 1,
            chunk: 1,
            kvValid: 2,
            window: 0,
            scale: 1
        )
        let out = PrefillAttentionRef.apply(inputs)
        #expect(out == [0.5, 0.25])
    }
}
