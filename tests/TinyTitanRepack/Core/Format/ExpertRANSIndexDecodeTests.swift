import Foundation
import Testing
import TinyTitanFormat

/// The sidecar decoder's contract with the GPU kernel.
///
/// The kernel is compiled for one scale — a 16,384-entry slot table, `& 16383`,
/// `>> 14`, renormalisation at 2^23 — so a sidecar that declares another
/// `scaleBits` is not a valid input for it: the writer only ever emits 14
/// (`RANSExpertLayerWriter.scaleBits`), and a foreign file with a matching
/// frequency total would otherwise decode to garbage rather than being
/// refused.
@Suite("expert rANS index decode")
struct ExpertRANSIndexDecodeTests {
    /// One minimal sidecar: a single block, a flat frequency model whose total
    /// is exactly 2^scaleBits, one expert whose payload run sits at 0.
    private static func sidecar(scaleBits: UInt8) -> Data {
        var data = Data()
        data.append(contentsOf: Array("TTRX".utf8))
        var version = UInt16(ExpertRANSIndex.currentVersion).littleEndian
        withUnsafeBytes(of: &version) { data.append(contentsOf: $0) }
        data.append(scaleBits)
        data.append(1)  // blockCount
        var expertCount = UInt32(1).littleEndian
        withUnsafeBytes(of: &expertCount) { data.append(contentsOf: $0) }
        data.append(contentsOf: [UInt8](repeating: 0, count: 4))  // reserved
        let perSymbol = (1 << Int(scaleBits)) / 256
        for _ in 0..<256 {
            var frequency = UInt16(perSymbol).littleEndian
            withUnsafeBytes(of: &frequency) { data.append(contentsOf: $0) }
        }
        var payloadOffset = UInt64(0).littleEndian
        withUnsafeBytes(of: &payloadOffset) { data.append(contentsOf: $0) }
        var payloadLength = UInt32(4).littleEndian
        withUnsafeBytes(of: &payloadLength) { data.append(contentsOf: $0) }
        var rawOffset = UInt32(0).littleEndian
        withUnsafeBytes(of: &rawOffset) { data.append(contentsOf: $0) }
        var symbolCount = UInt32(4).littleEndian
        withUnsafeBytes(of: &symbolCount) { data.append(contentsOf: $0) }
        data.append(1)  // stride
        data.append(0)  // role
        data.append(contentsOf: [UInt8](repeating: 0, count: 2))  // reserved
        return data
    }

    @Test("A sidecar declaring a scale the kernel does not implement is refused")
    func refusesUnimplementedScale() {
        #expect(throws: ExpertRANSIndex.Failure.self) {
            _ = try ExpertRANSIndex.decode(
                Self.sidecar(scaleBits: 13),
                expectedExperts: 1,
                expertStride: 4096,
                payloadSize: 4)
        }
        #expect(throws: ExpertRANSIndex.Failure.self) {
            _ = try ExpertRANSIndex.decode(
                Self.sidecar(scaleBits: 15),
                expectedExperts: 1,
                expertStride: 4096,
                payloadSize: 4)
        }
    }

    @Test func refusesOverflowingPayloadRange() {
        var data = Self.sidecar(scaleBits: 14)
        let offset = ExpertRANSIndex.headerBytes + 512
        data.replaceSubrange(offset..<(offset + 8), with: [UInt8](repeating: 255, count: 8))
        #expect(throws: ExpertRANSIndex.Failure.self) {
            _ = try ExpertRANSIndex.decode(
                data, expectedExperts: 1, expertStride: 4096, payloadSize: 4)
        }
    }

    @Test(arguments: [UInt64(0), UInt64(2)])
    func refusesOverlappingExpertPayloads(secondOffset: UInt64) {
        let model = ExpertRANSIndex.Model(frequencies: [UInt16](repeating: 64, count: 256))
        let blocks = [UInt64(0), secondOffset].map { offset in
            [
                ExpertRANSIndex.Block(
                    payloadOffset: offset, payloadLength: 4,
                    rawOffset: 0, symbolCount: 4, stride: 1, role: 0)
            ]
        }
        let data = ExpertRANSIndex.encode(
            .init(scaleBits: 14, models: [model], blocks: blocks))
        #expect(throws: ExpertRANSIndex.Failure.self) {
            _ = try ExpertRANSIndex.decode(
                data, expectedExperts: 2, expertStride: 4096, payloadSize: 8)
        }
    }

    @Test("The implemented scale decodes")
    func acceptsImplementedScale() throws {
        let layout = try ExpertRANSIndex.decode(
            Self.sidecar(scaleBits: 14),
            expectedExperts: 1,
            expertStride: 4096,
            payloadSize: 4)
        #expect(layout.scaleBits == 14)
        #expect(layout.models.count == 1)
        #expect(layout.blocks.count == 1)
    }
}
