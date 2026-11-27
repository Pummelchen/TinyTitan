import Foundation
import Testing
import TinyTitanKernelsC

/// The affine kernels are written for whole 64-element groups, and until now
/// that contract lived only in Swift (`Int8AffineGEMV.requireWholeGroups`).
///
/// In C a width that is not a multiple of the group size is not out of bounds —
/// it is worse: `groups = n / 64` and the row stride `n / 2` disagree, so every
/// row after the first is read from the wrong byte and the tail is dropped.
/// The kernels now refuse such a width at the boundary; these pin that they
/// leave the destination untouched instead of filling it with wrong numbers.
@Suite struct AffineKernelWidthGuardTests {
    private static let sentinel: Float = 42

    /// Two rows, so the mis-addressed second row is part of the shape.
    @Test func aWidthThatIsNotWholeGroupsLeavesTheOutputUntouched() {
        let n = 96  // 1.5 groups
        let rows = 2
        let weights = [UInt8](repeating: 0, count: rows * n)
        let scales = [UInt16](repeating: 0, count: rows * 2)
        let biases = [UInt16](repeating: 0, count: rows * 2)
        let x = [Float](repeating: 0, count: n)
        var out = [Float](repeating: Self.sentinel, count: rows)

        weights.withUnsafeBufferPointer { w in
            scales.withUnsafeBufferPointer { s in
                biases.withUnsafeBufferPointer { b in
                    x.withUnsafeBufferPointer { xp in
                        out.withUnsafeMutableBufferPointer { o in
                            tinytitan_int8_affine_gemv(
                                w.baseAddress, s.baseAddress, b.baseAddress, xp.baseAddress,
                                rows, n, o.baseAddress)
                        }
                    }
                }
            }
        }
        #expect(
            out.allSatisfy { $0 == Self.sentinel },
            "the 8-bit GEMV wrote with a width that is not whole groups")
    }

    @Test func aDequantWidthThatIsNotWholeGroupsLeavesTheOutputUntouched() {
        let n = 96
        let rows = 2
        let weights = [UInt8](repeating: 0, count: rows * n)
        let scales = [UInt16](repeating: 0, count: rows * 2)
        let biases = [UInt16](repeating: 0, count: rows * 2)
        var out8 = [Float](repeating: Self.sentinel, count: rows)
        var out4 = [Float](repeating: Self.sentinel, count: rows)

        weights.withUnsafeBufferPointer { w in
            scales.withUnsafeBufferPointer { s in
                biases.withUnsafeBufferPointer { b in
                    out8.withUnsafeMutableBufferPointer { o in
                        tinytitan_dequant_affine8(
                            w.baseAddress, s.baseAddress, b.baseAddress, rows, n, o.baseAddress)
                    }
                    out4.withUnsafeMutableBufferPointer { o in
                        tinytitan_dequant_affine4(
                            w.baseAddress, s.baseAddress, b.baseAddress, rows, n, o.baseAddress)
                    }
                }
            }
        }
        #expect(out8.allSatisfy { $0 == Self.sentinel }, "dequant8 wrote a partial group")
        #expect(out4.allSatisfy { $0 == Self.sentinel }, "dequant4 wrote a partial group")
    }

    /// And the whole-group path still computes, so the guard is not a refusal
    /// of the ordinary call.
    @Test func aWholeGroupWidthStillWrites() {
        let n = 64
        let rows = 1
        let weights = [UInt8](repeating: 0, count: rows * n)
        let scales = [UInt16](repeating: 0, count: rows)
        let biases = [UInt16](repeating: 0, count: rows)
        let x = [Float](repeating: 1, count: n)
        var out = [Float](repeating: Self.sentinel, count: rows)
        weights.withUnsafeBufferPointer { w in
            scales.withUnsafeBufferPointer { s in
                biases.withUnsafeBufferPointer { b in
                    x.withUnsafeBufferPointer { xp in
                        out.withUnsafeMutableBufferPointer { o in
                            tinytitan_int8_affine_gemv(
                                w.baseAddress, s.baseAddress, b.baseAddress, xp.baseAddress,
                                rows, n, o.baseAddress)
                        }
                    }
                }
            }
        }
        #expect(
            out.allSatisfy { $0 == 0 },
            "a zero weight row must dequantise to zero; got \(out.prefix(4))")
    }
}
