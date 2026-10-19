import Foundation
import Metal
import Testing

@testable import TinyTitan

/// A streamer reading a *compressed* layer.
///
/// The index and codec suites prove the pieces in isolation; this proves the
/// consumer: the streamer opens a layer whose blobs are LZFSE frames, reads one
/// by index, decodes it into the slot at the right offset, and every byte of the
/// slot is the expert that was asked for. A mistake here is silent weight
/// corruption, not a slow number, which is why it is checked byte for byte.
@Suite("Compressed expert streamer")
struct CompressedExpertStreamerTests {
    private static let experts = 4
    private static let stride = 1_000
    private static let split = 670

    private static func blob(_ expert: Int, _ count: Int) -> [UInt8] {
        [UInt8](repeating: UInt8(expert & 0xFF), count: count)
    }

    /// Writes the frame layout the repacker will write: gate|up and the tail as
    /// separate frames when `split` is set, one whole frame when it is not.
    private static func writeCompressedLayer(
        split: Int?
    ) throws -> (url: URL, frames: [[ExpertBlobIndex.Blob]]) {
        var file = Data()
        var frames: [[ExpertBlobIndex.Blob]] = []
        for expert in 0..<experts {
            var blobs: [ExpertBlobIndex.Blob] = []
            let halves: [Range<Int>] =
                if let split {
                    [0..<split, split..<stride]
                } else {
                    [0..<stride]
                }
            for range in halves {
                let raw = blob(expert, stride)[range]
                let frame = try raw.withUnsafeBytes { buffer in
                    try ExpertBlobCodec.compress(
                        source: try #require(buffer.baseAddress), count: buffer.count)
                }
                blobs.append(
                    .init(offset: UInt64(file.count), length: UInt32(frame.count)))
                file.append(frame)
            }
            frames.append(blobs)
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("compressed-layer-\(UUID().uuidString).bin")
        try file.write(to: url)
        return (url, frames)
    }

    @Test("A split compressed layer decodes both halves into the slot")
    func splitLayerLoads() throws {
        let (url, frames) = try Self.writeCompressedLayer(split: Self.split)
        defer { try? FileManager.default.removeItem(at: url) }
        let device = try MetalContext().device
        let layout = StreamLayout(
            path: url.path,
            streamOffset: 0,
            streamSize: try FileManager.default
                .attributesOfItem(atPath: url.path)[.size] as! UInt64,
            expertsPerLayer: Self.experts,
            expertStride: UInt64(Self.stride),
            phase1Bytes: UInt64(Self.split),
            compressedFrames: frames)
        let streamer = try PreadExpertStreamer(
            layout: layout, device: device, slotCount: 2)
        for expert in 0..<Self.experts {
            let slot = try streamer.loadExpert(layer: 0, expert: expert)
            let bytes = Self.bytes(of: slot.buffer, offset: slot.offset, count: Self.stride)
            #expect(
                bytes == Self.blob(expert, Self.stride),
                "expert \(expert) did not decode to its own bytes")
        }
    }

    @Test("A whole-expert compressed layer decodes in one frame")
    func wholeLayerLoads() throws {
        let (url, frames) = try Self.writeCompressedLayer(split: nil)
        defer { try? FileManager.default.removeItem(at: url) }
        let device = try MetalContext().device
        let layout = StreamLayout(
            path: url.path,
            streamOffset: 0,
            streamSize: try FileManager.default
                .attributesOfItem(atPath: url.path)[.size] as! UInt64,
            expertsPerLayer: Self.experts,
            expertStride: UInt64(Self.stride),
            phase1Bytes: nil,
            compressedFrames: frames)
        let streamer = try PreadExpertStreamer(
            layout: layout, device: device, slotCount: 2)
        let slot = try streamer.loadExpert(layer: 0, expert: 2)
        let bytes = Self.bytes(of: slot.buffer, offset: slot.offset, count: Self.stride)
        #expect(bytes == Self.blob(2, Self.stride))
    }

    /// An index whose expert count disagrees with the layer's geometry would
    /// address plausible but wrong bytes; the streamer refuses at open rather
    /// than decoding the wrong expert into a slot.
    @Test("An index that disagrees with the geometry is refused at open")
    func refusesMismatchedIndex() throws {
        let (url, frames) = try Self.writeCompressedLayer(split: Self.split)
        defer { try? FileManager.default.removeItem(at: url) }
        let device = try MetalContext().device
        let layout = StreamLayout(
            path: url.path,
            streamOffset: 0,
            streamSize: try FileManager.default
                .attributesOfItem(atPath: url.path)[.size] as! UInt64,
            expertsPerLayer: Self.experts + 1,
            expertStride: UInt64(Self.stride),
            phase1Bytes: UInt64(Self.split),
            compressedFrames: frames)
        #expect(throws: StreamerError.self) {
            _ = try PreadExpertStreamer(layout: layout, device: device, slotCount: 1)
        }
    }

    private static func bytes(of buffer: MTLBuffer, offset: UInt64, count: Int) -> [UInt8] {
        let base = buffer.contents().advanced(by: Int(offset))
        return [UInt8](UnsafeRawBufferPointer(start: base, count: count))
    }
}
