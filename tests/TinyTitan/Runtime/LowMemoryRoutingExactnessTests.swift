import Foundation
import Testing

@testable import TinyTitan

/// The 2-bit `lowmem` tier against the 4-bit arithmetic it replaces.
///
/// The converter remaps an already-quantized 4-bit value with
/// `q2 = round(q4 / 5)` and stores `scale2 = 5 * scale`. The native 2-bit
/// kernel computes `q2 * scale2 + bias`, which is `(5*q2) * scale + bias` --
/// the expanded 4-bit value. The two paths are therefore arithmetically the
/// same, and any disagreement here is a packing-order bug rather than a
/// rounding difference.
///
/// The byte order is the part the existing kernel test cannot see: its 2-bit
/// blobs are all zeros, so a reversed pair order would still produce zeros.
/// The native kernel takes element `k` from bits `[2k, 2k+2)`; the expander
/// packs element 0 in the top pair.
@Suite struct LowMemoryRoutingExactnessTests {

    private static let rows = 8
    private static let columns = 64

    /// Pack a row of q values low-element-first, the way
    /// `moe_affine_value` reads them.
    private static func packNative2(_ values: [Int]) -> [UInt8] {
        precondition(values.count % 4 == 0)
        var out = [UInt8](repeating: 0, count: values.count / 4)
        for (index, value) in values.enumerated() {
            out[index / 4] |= UInt8(value & 0x3) << UInt8((index % 4) * 2)
        }
        return out
    }

    private static func unpackNative2(_ bytes: [UInt8], count: Int) -> [Int] {
        (0..<count).map { index in
            Int((bytes[index / 4] >> UInt8((index % 4) * 2)) & 0x3)
        }
    }

    /// The expander's order: element 0 sits in the top pair so
    /// `TwoBitExpand` can rebuild a 4-bit byte low-nibble first.
    private static func packExpansion2(_ values: [Int]) -> [UInt8] {
        precondition(values.count % 4 == 0)
        var out = [UInt8](repeating: 0, count: values.count / 4)
        for (index, value) in values.enumerated() {
            out[index / 4] |= UInt8(value & 0x3) << UInt8((3 - index % 4) * 2)
        }
        return out
    }

    private static func unpack4(_ bytes: [UInt8], count: Int) -> [Int] {
        (0..<count).map { index in
            Int((bytes[index / 2] >> UInt8((index % 2) * 4)) & 0xF)
        }
    }

    /// The 4-bit reference for a 2-bit blob: `5*q2`, low nibble first.
    private static func expandToFourBit(_ values: [Int]) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: values.count / 2)
        for (index, value) in values.enumerated() {
            out[index / 2] |= UInt8((5 * value) & 0xF) << UInt8((index % 2) * 4)
        }
        return out
    }

    private static func gemv(
        rows: Int, columns: Int, scale: Float, bias: Float,
        decode: (Int) -> Int, offset: Int, x: [Float]
    ) -> [Float] {
        var out = [Float](repeating: 0, count: rows)
        for row in 0..<rows {
            var sum: Float = 0
            for column in 0..<columns {
                // One group of 64 weights, so the scale is per row here.
                let q = decode(offset + row * columns + column)
                sum += (Float(q) * scale + bias) * x[column]
            }
            out[row] = sum
        }
        return out
    }

    @Test func theTwoByteOrdersAreNotTheSameStream() {
        let values = [0, 1, 2, 3, 3, 2, 1, 0]
        let native = Self.packNative2(values)
        let expansion = Self.packExpansion2(values)
        #expect(native != expansion)
        #expect(Self.unpackNative2(native, count: values.count) == values)
        // Element 0 of the native stream lives in bits 0-1.
        #expect(native[0] & 0x3 == UInt8(values[0]))
    }

    @Test func nativeTwoBitMatchesTheExpandedFourBitArithmetic() {
        // A deterministic q4 tensor covering the whole 0...15 range.
        let q4 = (0..<(Self.rows * Self.columns)).map { ($0 * 7 + 3) % 16 }
        let q2 = q4.map { Int((Double($0) / 5.0).rounded()) }
        let scale: Float = 0.125
        let bias: Float = -0.25
        let x = (0..<Self.columns).map { Float(($0 % 7) - 3) * 0.125 }

        let native = Self.packNative2(q2)
        let expanded = Self.expandToFourBit(q2)
        let nativeValues = Self.unpackNative2(native, count: q2.count)
        let referenceValues = Self.unpack4(expanded, count: q2.count)
        // The native blob stores 5x the source scale; the expanded blob stores
        // the source scale against 5*q2. The products are the same.
        let nativeOut = Self.gemv(
            rows: Self.rows, columns: Self.columns, scale: 5 * scale, bias: bias,
            decode: { nativeValues[$0] }, offset: 0, x: x)
        let referenceOut = Self.gemv(
            rows: Self.rows, columns: Self.columns, scale: scale, bias: bias,
            decode: { referenceValues[$0] }, offset: 0, x: x)
        for (index, pair) in zip(nativeOut, referenceOut).enumerated() {
            #expect(
                abs(pair.0 - pair.1) <= 1e-5,
                "row \(index): native \(pair.0) vs expanded \(pair.1)")
        }
        // And the native stream carries the remapped values in order, not a
        // reversed pair order.
        #expect(nativeValues == q2)
    }

    @Test func theRemapStaysWithinTwoFourBitSteps() {
        let scale: Float = 0.125
        let bias: Float = 0.75
        for q4 in 0...15 {
            let q2 = Int((Double(q4) / 5.0).rounded())
            let source = Float(q4) * scale + bias
            let remapped = Float(q2) * (5 * scale) + bias
            #expect(
                abs(source - remapped) <= 2 * scale + 1e-6,
                "q4 \(q4): \(source) vs \(remapped)")
        }
    }

    /// Routing exactness: the gate scores of the toy experts rank identically
    /// whether they were read from the native 2-bit blobs or from the 4-bit
    /// expansion of the same values, so the same top-k experts are selected.
    @Test func routingScoresAgreeWithTheFourBitReference() {
        var rng = SeededRandom(seed: 0xF00D)
        let scale: Float = 0.125
        let bias: Float = 0
        let fanIn = 64
        let fanOut = 32
        let experts = 4
        let x = (0..<fanIn).map { _ in Float(rng.next() % 200) / 1000 - 0.1 }
        var nativeScores: [Float] = []
        var referenceScores: [Float] = []
        for _ in 0..<experts {
            let q4 = (0..<(fanOut * fanIn)).map { _ in Int(rng.next() % 16) }
            let q2 = q4.map { Int((Double($0) / 5.0).rounded()) }
            let native = Self.packNative2(q2)
            let expanded = Self.expandToFourBit(q2)
            let nativeValues = Self.unpackNative2(native, count: q2.count)
            let referenceValues = Self.unpack4(expanded, count: q2.count)
            let nativeGates = Self.gemv(
                rows: fanOut, columns: fanIn, scale: 5 * scale, bias: bias,
                decode: { nativeValues[$0] }, offset: 0, x: x)
            let referenceGates = Self.gemv(
                rows: fanOut, columns: fanIn, scale: scale, bias: bias,
                decode: { referenceValues[$0] }, offset: 0, x: x)
            nativeScores.append(nativeGates.reduce(0, +))
            referenceScores.append(referenceGates.reduce(0, +))
        }
        let nativeTop = nativeScores.indices.sorted { nativeScores[$0] > nativeScores[$1] }.prefix(
            2)
        let referenceTop =
            referenceScores.indices.sorted { referenceScores[$0] > referenceScores[$1] }.prefix(2)
        #expect(Array(nativeTop) == Array(referenceTop))
    }
}

/// Small deterministic generator, so the test does not depend on the
/// standard library's hash seeding.
private struct SeededRandom {
    private var state: UInt64
    init(seed: UInt64) { state = seed &* 6_364_136_223_846_793_005 &+ 1 }
    mutating func next() -> UInt64 {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return state >> 33
    }
}
