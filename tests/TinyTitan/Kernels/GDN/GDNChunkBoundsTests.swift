import Testing
import TinyTitanKernelsC

/// The chunked DeltaNet entry point takes its geometry from a C caller that it
/// cannot see the buffers of, and it narrows every BLAS extent to `int`. These
/// tests pin what it refuses before allocating or writing, and that valid
/// geometry still agrees with the scalar recurrence.
///
/// The allocation-failure paths are reached through
/// `tinytitan_delta_rule_chunk_with_allocator`; a real OOM here would need
/// gigabytes, which is exactly what the guards exist to avoid.
@Suite struct GDNChunkBoundsTests {

    private static let ok: Int32 = 0
    private static let fellBack: Int32 = 1
    private static let invalidGeometry: Int32 = 2
    private static let allocationFailed: Int32 = 3

    // MARK: - Helpers

    private func call(
        queries: [Float], queryStride: Int,
        keys: [Float], keyStride: Int,
        values: [Float], valueStride: Int,
        alphas: [Float], alphaStride: Int,
        betas: [Float], betaStride: Int,
        tokens: Int, dk: Int, dv: Int,
        state: inout [Float], readout: inout [Float], readoutStride: Int
    ) -> Int32 {
        queries.withUnsafeBufferPointer { q in
            keys.withUnsafeBufferPointer { k in
                values.withUnsafeBufferPointer { v in
                    alphas.withUnsafeBufferPointer { a in
                        betas.withUnsafeBufferPointer { b in
                            state.withUnsafeMutableBufferPointer { s in
                                readout.withUnsafeMutableBufferPointer { o in
                                    tinytitan_delta_rule_chunk(
                                        q.baseAddress, queryStride,
                                        k.baseAddress, keyStride,
                                        v.baseAddress, valueStride,
                                        a.baseAddress, alphaStride,
                                        b.baseAddress, betaStride,
                                        tokens, dk, dv,
                                        s.baseAddress, o.baseAddress, readoutStride)
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private func callWithAllocator(
        _ allocate: @convention(c) (Int) -> UnsafeMutableRawPointer?,
        queries: [Float], queryStride: Int,
        keys: [Float], keyStride: Int,
        values: [Float], valueStride: Int,
        alphas: [Float], alphaStride: Int,
        betas: [Float], betaStride: Int,
        tokens: Int, dk: Int, dv: Int,
        state: inout [Float], readout: inout [Float], readoutStride: Int
    ) -> Int32 {
        queries.withUnsafeBufferPointer { q in
            keys.withUnsafeBufferPointer { k in
                values.withUnsafeBufferPointer { v in
                    alphas.withUnsafeBufferPointer { a in
                        betas.withUnsafeBufferPointer { b in
                            state.withUnsafeMutableBufferPointer { s in
                                readout.withUnsafeMutableBufferPointer { o in
                                    tinytitan_delta_rule_chunk_with_allocator(
                                        q.baseAddress, queryStride,
                                        k.baseAddress, keyStride,
                                        v.baseAddress, valueStride,
                                        a.baseAddress, alphaStride,
                                        b.baseAddress, betaStride,
                                        tokens, dk, dv,
                                        s.baseAddress, o.baseAddress, readoutStride,
                                        allocate)
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// The recurrence the unrolled kernel must agree with, evaluated in the
    /// token order of the specification.
    private func scalarReference(
        tokens: Int, dk: Int, dv: Int,
        queries: [Float], keys: [Float], values: [Float],
        alphas: [Float], betas: [Float],
        state: inout [Float]
    ) -> [Float] {
        var readout = [Float](repeating: 0, count: tokens * dv)
        for t in 0..<tokens {
            let alpha = alphas[t]
            let beta = betas[t]
            for i in 0..<dv {
                var stored: Float = 0
                for column in 0..<dk {
                    state[i * dk + column] *= alpha
                    stored += state[i * dk + column] * keys[t * dk + column]
                }
                let correction = beta * (values[t * dv + i] - stored)
                for column in 0..<dk {
                    state[i * dk + column] += keys[t * dk + column] * correction
                }
                var read: Float = 0
                for column in 0..<dk {
                    read += state[i * dk + column] * queries[t * dk + column]
                }
                readout[t * dv + i] = read
            }
        }
        return readout
    }

    /// Deterministic, non-degenerate inputs for a small valid geometry.
    private func fixture(
        tokens: Int, dk: Int, dv: Int
    ) -> (queries: [Float], keys: [Float], values: [Float], alphas: [Float], betas: [Float]) {
        let queries = (0..<(tokens * dk)).map { Float($0 % 7) * 0.25 - 0.5 }
        let keys = (0..<(tokens * dk)).map { Float($0 % 5) * 0.2 - 0.3 }
        let values = (0..<(tokens * dv)).map { Float($0 % 11) * 0.125 - 0.5 }
        let alphas = (0..<tokens).map { 0.5 + Float($0 % 3) * 0.1 }
        let betas = (0..<tokens).map { 0.2 + Float($0 % 4) * 0.05 }
        return (queries, keys, values, alphas, betas)
    }

    // MARK: - Refusal

    @Test func zeroGeometryIsANoOp() {
        // The pointers are nil on purpose: the zero-dimension contract is that
        // nothing is dereferenced, not merely that nothing is written.
        #expect(
            tinytitan_delta_rule_chunk(
                nil, 0, nil, 0, nil, 0, nil, 0, nil, 0,
                0, 1, 1, nil, nil, 0) == Self.ok)
        #expect(
            tinytitan_delta_rule_chunk(
                nil, 0, nil, 0, nil, 0, nil, 0, nil, 0,
                1, 0, 1, nil, nil, 0) == Self.ok)
        #expect(
            tinytitan_delta_rule_chunk(
                nil, 0, nil, 0, nil, 0, nil, 0, nil, 0,
                1, 1, 0, nil, nil, 0) == Self.ok)
    }

    /// `2 * tokens` above INT_MAX cannot be the shared BLAS extent.
    @Test func twoTokensAboveIntMaxIsRejected() {
        var state: [Float] = [.nan, .nan, .nan, .nan]
        var readout: [Float] = [.nan, .nan, .nan, .nan]
        let status = call(
            queries: [0], queryStride: 1,
            keys: [0], keyStride: 1,
            values: [0], valueStride: 1,
            alphas: [1], alphaStride: 1,
            betas: [1], betaStride: 1,
            tokens: Int(Int32.max) / 2 + 1, dk: 1, dv: 1,
            state: &state, readout: &readout, readoutStride: 1)
        #expect(status == Self.invalidGeometry)
        #expect(state.allSatisfy { $0.isNaN })
        #expect(readout.allSatisfy { $0.isNaN })
    }

    /// A scratch product that wraps `size_t` is refused before the allocator is
    /// asked for anything, even though the individual dimensions are only
    /// `size_t`-sized.
    @Test func scratchProductOverflowIsRejected() {
        var state: [Float] = [.nan, .nan, .nan, .nan]
        var readout: [Float] = [.nan, .nan, .nan, .nan]
        let wide = 1 << 62
        let status = call(
            queries: [0], queryStride: wide,
            keys: [0], keyStride: wide,
            values: [0], valueStride: 1,
            alphas: [1], alphaStride: 1,
            betas: [1], betaStride: 1,
            tokens: 1, dk: wide, dv: 1,
            state: &state, readout: &readout, readoutStride: 1)
        #expect(status == Self.invalidGeometry)
        #expect(state.allSatisfy { $0.isNaN })
        #expect(readout.allSatisfy { $0.isNaN })
    }

    /// Every BLAS extent is narrowed to `int`; one past `INT_MAX` must be
    /// refused rather than silently becoming a negative or truncated extent.
    @Test func blasExtentAboveIntMaxIsRejected() {
        let tooWide = Int(Int32.max) + 1
        var state: [Float] = [.nan, .nan]
        var readout: [Float] = [.nan, .nan]

        func status(tokens: Int, dk: Int, dv: Int) -> Int32 {
            call(
                queries: [0], queryStride: max(dk, 1),
                keys: [0], keyStride: max(dk, 1),
                values: [0], valueStride: max(dv, 1),
                alphas: [1], alphaStride: 1,
                betas: [1], betaStride: 1,
                tokens: tokens, dk: dk, dv: dv,
                state: &state, readout: &readout, readoutStride: max(dv, 1))
        }

        // `2 * tokens` above INT_MAX is the first narrowing failure.
        #expect(status(tokens: tooWide / 2, dk: 1, dv: 1) == Self.invalidGeometry)
        #expect(status(tokens: 1, dk: tooWide, dv: 1) == Self.invalidGeometry)
        #expect(status(tokens: 1, dk: 1, dv: tooWide) == Self.invalidGeometry)
        #expect(state.allSatisfy { $0.isNaN })
        #expect(readout.allSatisfy { $0.isNaN })
    }

    /// A stride below the width it indexes overlaps the next row; a zero
    /// alpha/beta stride collapses every token onto one. Both are refused.
    @Test func badStridesAreRejected() {
        var state: [Float] = [.nan, .nan, .nan, .nan]
        var readout: [Float] = [.nan, .nan, .nan, .nan]

        func status(
            queryStride: Int, keyStride: Int, valueStride: Int,
            alphaStride: Int, betaStride: Int, readoutStride: Int
        ) -> Int32 {
            call(
                queries: [0], queryStride: queryStride,
                keys: [0], keyStride: keyStride,
                values: [0], valueStride: valueStride,
                alphas: [1], alphaStride: alphaStride,
                betas: [1], betaStride: betaStride,
                tokens: 2, dk: 2, dv: 2,
                state: &state, readout: &readout, readoutStride: readoutStride)
        }

        #expect(
            status(
                queryStride: 1, keyStride: 2, valueStride: 2,
                alphaStride: 1, betaStride: 1, readoutStride: 2) == Self.invalidGeometry)
        #expect(
            status(
                queryStride: 2, keyStride: 1, valueStride: 2,
                alphaStride: 1, betaStride: 1, readoutStride: 2) == Self.invalidGeometry)
        #expect(
            status(
                queryStride: 2, keyStride: 2, valueStride: 1,
                alphaStride: 1, betaStride: 1, readoutStride: 2) == Self.invalidGeometry)
        #expect(
            status(
                queryStride: 2, keyStride: 2, valueStride: 2,
                alphaStride: 0, betaStride: 1, readoutStride: 2) == Self.invalidGeometry)
        #expect(
            status(
                queryStride: 2, keyStride: 2, valueStride: 2,
                alphaStride: 1, betaStride: 0, readoutStride: 2) == Self.invalidGeometry)
        #expect(
            status(
                queryStride: 2, keyStride: 2, valueStride: 2,
                alphaStride: 1, betaStride: 1, readoutStride: 1) == Self.invalidGeometry)
        #expect(state.allSatisfy { $0.isNaN })
        #expect(readout.allSatisfy { $0.isNaN })
    }

    // MARK: - Valid geometry

    /// The guard must not perturb the accepted path: the chunked result agrees
    /// with the token-ordered recurrence, and padded strides still work.
    @Test func validGeometryMatchesTheScalarReference() {
        let tokens = 5
        let dk = 3
        let dv = 2
        let input = fixture(tokens: tokens, dk: dk, dv: dv)
        var state = (0..<(dv * dk)).map { Float($0 % 4) * 0.1 }
        var expectedState = state
        let expected = scalarReference(
            tokens: tokens, dk: dk, dv: dv,
            queries: input.queries, keys: input.keys, values: input.values,
            alphas: input.alphas, betas: input.betas, state: &expectedState)
        var readout = [Float](repeating: .nan, count: tokens * dv)

        let status = call(
            queries: input.queries, queryStride: dk,
            keys: input.keys, keyStride: dk,
            values: input.values, valueStride: dv,
            alphas: input.alphas, alphaStride: 1,
            betas: input.betas, betaStride: 1,
            tokens: tokens, dk: dk, dv: dv,
            state: &state, readout: &readout, readoutStride: dv)
        #expect(status == Self.ok)
        for index in 0..<(tokens * dv) {
            #expect(
                abs(readout[index] - expected[index]) < 1e-5,
                "readout \(index): \(readout[index]) vs \(expected[index])")
        }
        for index in 0..<(dv * dk) {
            #expect(abs(state[index] - expectedState[index]) < 1e-5)
        }
    }

    /// A stride wider than the width it indexes is padding, not an error.
    @Test func paddedStridesAreAccepted() {
        let tokens = 2
        let dk = 2
        let dv = 2
        let queryStride = 4
        let keyStride = 3
        let valueStride = 5
        let readoutStride = 3
        var queries = [Float](repeating: 0, count: tokens * queryStride)
        var keys = [Float](repeating: 0, count: tokens * keyStride)
        var values = [Float](repeating: 0, count: tokens * valueStride)
        for t in 0..<tokens {
            for c in 0..<dk {
                queries[t * queryStride + c] = Float(t + c + 1) * 0.1
                keys[t * keyStride + c] = Float(t - c) * 0.2
            }
            for c in 0..<dv {
                values[t * valueStride + c] = Float(t + c) * 0.3
            }
        }
        var state = [Float](repeating: 0, count: dv * dk)
        var readout = [Float](repeating: .nan, count: tokens * readoutStride)
        let status = call(
            queries: queries, queryStride: queryStride,
            keys: keys, keyStride: keyStride,
            values: values, valueStride: valueStride,
            alphas: [0.9, 0.8], alphaStride: 1,
            betas: [0.5, 0.4], betaStride: 1,
            tokens: tokens, dk: dk, dv: dv,
            state: &state, readout: &readout, readoutStride: readoutStride)
        #expect(status == Self.ok)
        #expect(readout[0..<dv].allSatisfy { $0.isFinite })
        #expect(readout[readoutStride..<(readoutStride + dv)].allSatisfy { $0.isFinite })
        // The padding columns are never written.
        #expect(readout[dv].isNaN)
        #expect(readout[readoutStride + dv].isNaN)
    }

    // MARK: - Allocation failure

    /// With `dv` inside the fallback's fixed buffer, a failed scratch allocation
    /// still produces the chunk sequentially and reports that it fell back.
    @Test func failedAllocationFallsBackWhenDvFits() {
        let tokens = 4
        let dk = 3
        let dv = 2
        let input = fixture(tokens: tokens, dk: dk, dv: dv)
        var expectedState = (0..<(dv * dk)).map { Float($0 % 4) * 0.1 }
        let expected = scalarReference(
            tokens: tokens, dk: dk, dv: dv,
            queries: input.queries, keys: input.keys, values: input.values,
            alphas: input.alphas, betas: input.betas, state: &expectedState)
        var state = (0..<(dv * dk)).map { Float($0 % 4) * 0.1 }
        var readout = [Float](repeating: .nan, count: tokens * dv)

        let status = callWithAllocator(
            { _ in nil },
            queries: input.queries, queryStride: dk,
            keys: input.keys, keyStride: dk,
            values: input.values, valueStride: dv,
            alphas: input.alphas, alphaStride: 1,
            betas: input.betas, betaStride: 1,
            tokens: tokens, dk: dk, dv: dv,
            state: &state, readout: &readout, readoutStride: dv)
        #expect(status == Self.fellBack)
        for index in 0..<(tokens * dv) {
            #expect(abs(readout[index] - expected[index]) < 1e-5)
        }
        for index in 0..<(dv * dk) {
            #expect(abs(state[index] - expectedState[index]) < 1e-5)
        }
    }

    /// Past the fallback's fixed buffer there is no result to produce, so the
    /// call must say so and leave the caller's buffers untouched.
    @Test func failedAllocationProducesNoOutputForLargeDv() {
        let tokens = 2
        let dk = 2
        let dv = 257
        var queries = [Float](repeating: 0, count: tokens * dk)
        var keys = [Float](repeating: 0, count: tokens * dk)
        var values = [Float](repeating: 0, count: tokens * dv)
        for index in queries.indices { queries[index] = Float(index % 5) * 0.1 - 0.2 }
        for index in keys.indices { keys[index] = Float(index % 3) * 0.2 - 0.1 }
        for index in values.indices { values[index] = Float(index % 7) * 0.05 }
        var state = [Float](repeating: .nan, count: dv * dk)
        var readout = [Float](repeating: .nan, count: tokens * dv)

        let status = callWithAllocator(
            { _ in nil },
            queries: queries, queryStride: dk,
            keys: keys, keyStride: dk,
            values: values, valueStride: dv,
            alphas: [0.9, 0.8], alphaStride: 1,
            betas: [0.5, 0.4], betaStride: 1,
            tokens: tokens, dk: dk, dv: dv,
            state: &state, readout: &readout, readoutStride: dv)
        #expect(status == Self.allocationFailed)
        #expect(state.allSatisfy { $0.isNaN })
        #expect(readout.allSatisfy { $0.isNaN })
    }

    @Test func nullAllocatorIsRejected() {
        // The NULL-allocator refusal is ahead of geometry, so every other
        // pointer is nil on purpose: nothing may be dereferenced.
        let status = tinytitan_delta_rule_chunk_with_allocator(
            nil, 0, nil, 0, nil, 0, nil, 0, nil, 0,
            1, 1, 1, nil, nil, 1, nil)
        #expect(status == Self.invalidGeometry)
    }
}
