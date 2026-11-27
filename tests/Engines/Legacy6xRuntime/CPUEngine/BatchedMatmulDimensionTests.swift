import Foundation
import Testing

@testable import Legacy6xRuntime
@testable import Qwen35Dense2B8Bit
@testable import Qwen35Dense4B4Bit
@testable import Qwen35Dense4B8Bit
@testable import Qwen35Dense9B4Bit
@testable import Qwen35Dense9B8Bit
@testable import TinyTitanCore

/// A matrix's row count reaches the C kernel as an `Int32`, and the shape it
/// came from is manifest input. A count above `Int32.max` is refused with an
/// error instead of trapping in `Int32(...)` (AUD-0084).
@Suite("batched matmul dimension refusal")
struct BatchedMatmulDimensionTests {
    private static let tokens = 16
    private static let columns = 64
    private static let rowsBeyondInt32 = Int(Int32.max) + 1

    /// Builds a matrix describing far more rows than can be allocated; the
    /// buffers are four bytes because the dimension guard must run before
    /// anything reads them.
    private static func withOversizedMatrix<Matrix>(
        make: (
            UnsafeRawBufferPointer, UnsafePointer<UInt16>, UnsafePointer<UInt16>, Int, Int, Int
        ) -> Matrix,
        _ body: (Matrix) throws -> Void
    ) throws {
        let weights = UnsafeMutablePointer<UInt8>.allocate(capacity: 4)
        let scales = UnsafeMutablePointer<UInt16>.allocate(capacity: 1)
        let biases = UnsafeMutablePointer<UInt16>.allocate(capacity: 1)
        defer {
            weights.deallocate()
            scales.deallocate()
            biases.deallocate()
        }
        weights.initialize(repeating: 0, count: 4)
        scales.initialize(repeating: 0, count: 1)
        biases.initialize(repeating: 0, count: 1)
        let matrix = make(
            UnsafeRawBufferPointer(start: weights, count: 4), scales, biases,
            rowsBeyondInt32, columns, 4)
        try body(matrix)
    }

    private static func refuses<Matrix>(
        make: (
            UnsafeRawBufferPointer, UnsafePointer<UInt16>, UnsafePointer<UInt16>, Int, Int, Int
        ) -> Matrix,
        multiply: (Matrix, [Float], Int, Int) throws -> [Float]
    ) throws {
        let x = [Float](repeating: 0, count: tokens * columns)
        try withOversizedMatrix(make: make) { matrix in
            #expect(throws: ModelError.self) {
                _ = try multiply(matrix, x, Self.tokens, 1)
            }
        }
    }

    @Test func everyDenseVariantRefusesARowCountBeyondInt32() throws {
        try Self.refuses(
            make: { weights, scales, biases, rows, columns, bits in
                Legacy6xRuntime.AffineSnapshot.Matrix(
                    weights: weights, scales: scales, biases: biases,
                    rows: rows, columns: columns, bits: bits, groupSize: 64)
            },
            multiply: Legacy6xRuntime.CPUBatchedMatmul.multiply)
        try Self.refuses(
            make: { weights, scales, biases, rows, columns, bits in
                Qwen35Dense2B8Bit.AffineSnapshot.Matrix(
                    weights: weights, scales: scales, biases: biases,
                    rows: rows, columns: columns, bits: bits, groupSize: 64)
            },
            multiply: Qwen35Dense2B8Bit.CPUBatchedMatmul.multiply)
        try Self.refuses(
            make: { weights, scales, biases, rows, columns, bits in
                Qwen35Dense4B4Bit.AffineSnapshot.Matrix(
                    weights: weights, scales: scales, biases: biases,
                    rows: rows, columns: columns, bits: bits, groupSize: 64)
            },
            multiply: Qwen35Dense4B4Bit.CPUBatchedMatmul.multiply)
        try Self.refuses(
            make: { weights, scales, biases, rows, columns, bits in
                Qwen35Dense4B8Bit.AffineSnapshot.Matrix(
                    weights: weights, scales: scales, biases: biases,
                    rows: rows, columns: columns, bits: bits, groupSize: 64)
            },
            multiply: Qwen35Dense4B8Bit.CPUBatchedMatmul.multiply)
        try Self.refuses(
            make: { weights, scales, biases, rows, columns, bits in
                Qwen35Dense9B4Bit.AffineSnapshot.Matrix(
                    weights: weights, scales: scales, biases: biases,
                    rows: rows, columns: columns, bits: bits, groupSize: 64)
            },
            multiply: Qwen35Dense9B4Bit.CPUBatchedMatmul.multiply)
        try Self.refuses(
            make: { weights, scales, biases, rows, columns, bits in
                Qwen35Dense9B8Bit.AffineSnapshot.Matrix(
                    weights: weights, scales: scales, biases: biases,
                    rows: rows, columns: columns, bits: bits, groupSize: 64)
            },
            multiply: Qwen35Dense9B8Bit.CPUBatchedMatmul.multiply)
    }
}
