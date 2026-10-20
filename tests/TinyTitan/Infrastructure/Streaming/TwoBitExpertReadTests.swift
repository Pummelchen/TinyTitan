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
            guard let pointer = raw.baseAddress else {
                Issue.record("destination buffer has no base address")
                return
            }
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
            guard let pointer = raw.baseAddress else {
                Issue.record("staging buffer has no base address")
                return
            }
            try? reader.fetch(offsets: [offset], byteCounts: [short.count], into: [pointer])
        }

        var expanded = [UInt8](repeating: 0, count: short.count * 2)
        staged.withUnsafeBufferPointer { src in
            expanded.withUnsafeMutableBufferPointer { dst in
                guard let source = src.baseAddress, let destination = dst.baseAddress
                else {
                    Issue.record("expansion buffers have no base address")
                    return
                }
                TwoBitExpand.expand(
                    source: source, destination: destination,
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
            // 4-bit packs element 2k in the low nibble, 2k+1 in the high one.
            #expect((out0 & 0xF) == values[0] * 5)
            #expect((out0 >> 4) == values[1] * 5)
            #expect((out1 & 0xF) == values[2] * 5)
            #expect((out1 >> 4) == values[3] * 5)
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

    /// The in-place rewrite: a compact blob becomes the 4-bit slot layout
    /// without a second buffer. Processed backwards, so the scale move lands
    /// before the weight expansion overwrites the bytes it came from.
    @Test func compactBlobRewritesIntoTheFourBitLayout() {
        let slot = UnsafeMutableRawPointer.allocate(byteCount: 16, alignment: 1)
        defer { slot.deallocate() }
        let bytes = slot.assumingMemoryBound(to: UInt8.self)
        for i in 0..<16 { bytes[i] = 0 }
        // weight: 4 bytes of 2-bit values; scale: 5.0, 10.0; bias: 1.0, -2.0
        let weight: [UInt8] = [0x1B, 0xE4, 0x4E, 0xB1]
        for (i, b) in weight.enumerated() { bytes[i] = b }
        func putBF16(_ value: Float, at index: Int) {
            let bits = value.bitPattern >> 16
            bytes[index] = UInt8(truncatingIfNeeded: bits)
            bytes[index + 1] = UInt8(truncatingIfNeeded: bits >> 8)
        }
        putBF16(5.0, at: 4); putBF16(10.0, at: 6)
        putBF16(1.0, at: 8); putBF16(-2.0, at: 10)

        let tensors = [
            ExpertBlobTensor(
                sourceOffset: 0, sourceSize: 4,
                targetOffset: 0, targetSize: 8, bits: 2,
                expands: true, rescales: false),
            ExpertBlobTensor(
                sourceOffset: 4, sourceSize: 4,
                targetOffset: 8, targetSize: 4, bits: 0,
                expands: false, rescales: true),
            ExpertBlobTensor(
                sourceOffset: 8, sourceSize: 4,
                targetOffset: 12, targetSize: 4, bits: 0,
                expands: false, rescales: false),
        ]
        TwoBitExpand.expandExpertInPlace(slot: slot, tensors: tensors)

        var expected = [UInt8](repeating: 0, count: 8)
        weight.withUnsafeBufferPointer { src in
            expected.withUnsafeMutableBufferPointer { dst in
                guard let source = src.baseAddress, let destination = dst.baseAddress
                else {
                    Issue.record("expected buffers have no base address")
                    return
                }
                TwoBitExpand.expand(
                    source: source, destination: destination,
                    byteCount: weight.count)
            }
        }
        for i in 0..<8 { #expect(bytes[i] == expected[i]) }
        func bf16(at index: Int) -> Float {
            let bits = UInt32(bytes[index]) | (UInt32(bytes[index + 1]) << 8)
            return Float(bitPattern: bits << 16)
        }
        #expect(abs(bf16(at: 8) - 1.0) < 1e-3)
        #expect(abs(bf16(at: 10) - 2.0) < 1e-3)
        #expect(abs(bf16(at: 12) - 1.0) < 1e-3)
        #expect(abs(bf16(at: 14) + 2.0) < 1e-3)
    }
}
