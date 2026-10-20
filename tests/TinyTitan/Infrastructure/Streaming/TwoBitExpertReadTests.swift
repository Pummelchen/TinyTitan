import Foundation
import Testing

@testable import TinyTitan

/// The cold tier reads a 2-bit expert that is half a slot long, then expands it
/// to the 4-bit slot the existing kernels expect. This pins both halves on a
/// real file: the offset/byte-count read returns exactly the expert's bytes,
/// and the expansion reproduces the 4-bit values the 2-bit ones stood for.
@Suite struct TwoBitExpertReadTests {

    /// Writes one layer file: a 4-bit expert at 0..<stride and a 2-bit expert
    /// at the next stride. Returns the file URL and the two payloads.
    private static func writeLayerFile(
        stride: Int
    ) throws -> (url: URL, short: [UInt8], shortOffset: UInt64) {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ssdai-two-bit-read-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("layer_00.bin")

        // The short expert is half a slot: 2-bit values, four per byte, derived
        // from a deterministic pattern so the expansion can be checked exactly.
        var short = [UInt8](repeating: 0, count: stride / 2)
        for i in short.indices { short[i] = UInt8(truncatingIfNeeded: i &* 29 &+ 7) }
        var payload = [UInt8](repeating: 0xAA, count: stride)  // hot expert filler
        payload.append(contentsOf: short)
        try Data(payload).write(to: url)
        return (url, short, UInt64(stride))
    }

    @Test func aShortExpertIsReadByOffsetAndCount() throws {
        let stride = 16_384
        let (url, short, offset) = try Self.writeLayerFile(stride: stride)
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }

        let reader = try ParallelExpertReader(
            path: url.path, expertStride: stride, threads: 2, bypassCache: true)
        var destination = [UInt8](repeating: 0, count: stride)
        destination.withUnsafeMutableBytes { raw in
            let pointer = raw.baseAddress!
            #expect(throws: Never.self) {
                try reader.fetch(
                    offsets: [offset], byteCounts: [short.count], into: [pointer])
            }
        }
        #expect(destination.prefix(short.count).elementsEqual(short))
        // Bytes past the payload are untouched, not zeroed by the read.
        #expect(destination[short.count] == 0x00)
    }

    @Test func expansionReproducesTheFourBitValues() throws {
        let stride = 16_384
        let (url, short, offset) = try Self.writeLayerFile(stride: stride)
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }

        let reader = try ParallelExpertReader(
            path: url.path, expertStride: stride, threads: 2, bypassCache: true)
        var staged = [UInt8](repeating: 0, count: short.count)
        staged.withUnsafeMutableBytes { raw in
            let pointer = raw.baseAddress!
            try? reader.fetch(offsets: [offset], byteCounts: [short.count], into: [pointer])
        }

        var expanded = [UInt8](repeating: 0, count: short.count * 2)
        staged.withUnsafeBufferPointer { src in
            expanded.withUnsafeMutableBufferPointer { dst in
                TwoBitExpand.expand(
                    source: src.baseAddress!, destination: dst.baseAddress!,
                    byteCount: short.count)
            }
        }
        // Every 2-bit value in the source becomes 5x in the 4-bit stream, so
        // decoding both back to values must agree.
        for index in 0..<short.count {
            let byte = short[index]
            let values = [
                (byte >> 6) & 0x3, (byte >> 4) & 0x3,
                (byte >> 2) & 0x3, byte & 0x3,
            ]
            let out0 = expanded[2 * index]
            let out1 = expanded[2 * index + 1]
            #expect((out0 >> 4) == values[0] * 5)
            #expect((out0 & 0xF) == values[1] * 5)
            #expect((out1 >> 4) == values[2] * 5)
            #expect((out1 & 0xF) == values[3] * 5)
        }
    }

    /// The expansion is only useful if the arithmetic survives it: an expanded
    /// 4-bit value against the divided scale must reproduce the 2-bit product,
    /// which is what lets the existing 4-bit kernels run a cold expert.
    @Test func expandedValuesReproduceTheTwoBitProduct() {
        let scales: [Float] = [0.03125, 0.1, 1.0, 7.5]
        let biases: [Float] = [0, -0.5, 2.25]
        for q2 in 0...3 {
            let q4 = Float(q2 * 5)
            for scale in scales {
                for bias in biases {
                    let original = Float(q2) * scale + bias
                    let expanded = q4 * TwoBitExpand.adjustedScale(scale) + bias
                    #expect(abs(original - expanded) <= max(1e-6, abs(original) * 1e-5))
                }
            }
        }
    }
}
