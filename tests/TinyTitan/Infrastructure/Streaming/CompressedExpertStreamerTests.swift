import Foundation
import Metal
import Testing

@testable import TinyTitan
import TinyTitanFormat

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
            streamSize: try Self.fileSize(url),
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
            streamSize: try Self.fileSize(url),
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
            streamSize: try Self.fileSize(url),
            expertsPerLayer: Self.experts + 1,
            expertStride: UInt64(Self.stride),
            phase1Bytes: UInt64(Self.split),
            compressedFrames: frames)
        #expect(throws: StreamerError.self) {
            _ = try PreadExpertStreamer(layout: layout, device: device, slotCount: 1)
        }
    }

    /// The prefetch ring reads ahead into buffers that are not cache slots, and
    /// adoption copies from them. For a compressed layer the ring therefore has
    /// to hold *decoded* weights, or adoption would copy frame bytes into a slot.
    @Test("Prefetch decodes a compressed expert into its destination")
    func prefetchDecodes() throws {
        let (url, frames) = try Self.writeCompressedLayer(split: Self.split)
        defer { try? FileManager.default.removeItem(at: url) }
        let device = try MetalContext().device
        let layout = StreamLayout(
            path: url.path,
            streamOffset: 0,
            streamSize: try Self.fileSize(url),
            expertsPerLayer: Self.experts,
            expertStride: UInt64(Self.stride),
            phase1Bytes: UInt64(Self.split),
            compressedFrames: frames)
        let streamer = try PreadExpertStreamer(
            layout: layout, device: device, slotCount: 2)

        let destination = UnsafeMutableRawPointer.allocate(
            byteCount: Self.stride, alignment: 4096)
        defer { destination.deallocate() }
        // Poisoned, so a prefetch that did nothing cannot pass by accident.
        destination.initializeMemory(as: UInt8.self, repeating: 0xAA, count: Self.stride)
        let operation = try streamer.beginPrefetch(
            experts: [2], destinations: [destination])
        try operation.wait()
        let bytes = [UInt8](
            UnsafeRawBufferPointer(start: destination, count: Self.stride))
        #expect(bytes == Self.blob(2, Self.stride))
    }

    /// The batched decode path is what the runner actually uses for a token's
    /// experts; the per-expert path above is the one-shot API. They read through
    /// different code (`executeCompressedMisses` versus `loadExpertUnlocked`),
    /// and a plan that took the raw backend was a live failure on a real install
    /// -- it read `n * stride` from a 92%-sized file. So both are pinned.
    @Test("A compressed plan decodes every miss into its own slot")
    func planDecodesCompressedExperts() throws {
        let (url, frames) = try Self.writeCompressedLayer(split: Self.split)
        defer { try? FileManager.default.removeItem(at: url) }
        let device = try MetalContext().device
        let layout = StreamLayout(
            path: url.path,
            streamOffset: 0,
            streamSize: try Self.fileSize(url),
            expertsPerLayer: Self.experts,
            expertStride: UInt64(Self.stride),
            phase1Bytes: UInt64(Self.split),
            compressedFrames: frames)
        let streamer = try PreadExpertStreamer(
            layout: layout, device: device, slotCount: 4)
        let experts = [0, 2, 3]
        let results = try streamer.executeExpertCachePlan(
            try streamer.planExpertsCached(experts: experts))
        #expect(results.count == experts.count)
        for (result, expert) in zip(results, experts) {
            let bytes = Self.bytes(
                of: result.buffer, offset: result.offset, count: Self.stride)
            #expect(bytes == Self.blob(expert, Self.stride))
        }
    }

    @Test("The asynchronous plan path decodes a compressed layer too")
    func asyncPlanDecodesCompressedExperts() throws {
        let (url, frames) = try Self.writeCompressedLayer(split: Self.split)
        defer { try? FileManager.default.removeItem(at: url) }
        let device = try MetalContext().device
        let layout = StreamLayout(
            path: url.path,
            streamOffset: 0,
            streamSize: try Self.fileSize(url),
            expertsPerLayer: Self.experts,
            expertStride: UInt64(Self.stride),
            phase1Bytes: UInt64(Self.split),
            compressedFrames: frames)
        let streamer = try PreadExpertStreamer(
            layout: layout, device: device, slotCount: 2)
        let operation = try streamer.beginExpertCachePlan(
            try streamer.planExpertsCached(experts: [1]))
        try operation.wait()
        let slot = try streamer.loadExpert(layer: 0, expert: 1)
        let bytes = Self.bytes(of: slot.buffer, offset: slot.offset, count: Self.stride)
        #expect(bytes == Self.blob(1, Self.stride))
    }

    private static func fileSize(_ url: URL) throws -> UInt64 {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        return try #require((attributes[.size] as? NSNumber)?.uint64Value)
    }

    private static func bytes(of buffer: MTLBuffer, offset: UInt64, count: Int) -> [UInt8] {
        let base = buffer.contents().advanced(by: Int(offset))
        return [UInt8](UnsafeRawBufferPointer(start: base, count: count))
    }
}
