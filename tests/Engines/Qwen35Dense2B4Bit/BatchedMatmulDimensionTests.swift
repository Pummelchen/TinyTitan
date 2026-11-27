import Foundation
import Testing

@testable import Qwen35Dense2B4Bit
@testable import TinyTitanCore

/// A matrix's row count reaches the C kernel as an `Int32`, and the shape it
/// came from is manifest input. A count above `Int32.max` is refused with an
/// error instead of trapping in `Int32(...)` (AUD-0084).
@Suite("2B4Bit batched matmul dimension refusal")
struct BatchedMatmulDimensionTests {
    private static let tokens = 16
    private static let columns = 64

    @Test func aRowCountBeyondInt32IsRefused() throws {
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
        // The buffers are four bytes because the dimension guard must run
        // before the matrix is dequantized.
        let matrix = AffineSnapshot.Matrix(
            weights: UnsafeRawBufferPointer(start: weights, count: 4),
            scales: scales, biases: biases,
            rows: Int(Int32.max) + 1, columns: Self.columns, bits: 4, groupSize: 64)
        let x = [Float](repeating: 0, count: Self.tokens * Self.columns)
        #expect(throws: ModelError.self) {
            _ = try CPUBatchedMatmul.multiply(matrix, x: x, tokenCount: Self.tokens, threads: 1)
        }
    }
}
