import Foundation
import Testing
import TinyTitanFormat

@testable import TinyTitanRepackCore

/// The writer that makes a compressed install out of a raw one.
///
/// The streamer's suite proves it can *read* frames; this proves the bytes it
/// reads are the ones the writer produced, byte for byte. A mistake here is
/// silent weight corruption at load, so every expert is compared whole.
@Suite("Compressed expert layer writer")
struct CompressedExpertLayerWriterTests {
    private static let experts = 4
    private static let stride = 4_096
    private static let split = 2_700

    /// Compressible enough that LZFSE actually shrinks it (real expert blobs do,
    /// at ~91.7%), deterministic so a failure is reproducible.
    private static func expertBytes(_ expert: Int) -> [UInt8] {
        (0..<stride).map { index in
            UInt8(truncatingIfNeeded: (index / 8) &+ expert &* 17)
        }
    }

    private static func rawLayer() -> Data {
        var data = Data()
        for expert in 0..<experts {
            data.append(contentsOf: expertBytes(expert))
        }
        return data
    }

    private static func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("compressed-expert-writer-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    @Test("A split layer round-trips through the writer, the index and the codec")
    func splitLayerRoundTrips() throws {
        let directory = try Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = directory.appendingPathComponent("layer_00.bin")
        try Self.rawLayer().write(to: source)
        let destination = directory.appendingPathComponent("layer_00.bin")

        let result = try CompressedExpertLayerWriter.write(
            sourcePath: source.path, destinationPath: destination.path,
            geometry: CompressedExpertLayerGeometry(
                expertStride: UInt64(Self.stride),
                expertsPerLayer: Self.experts,
                splitBytes: UInt64(Self.split)))
        #expect(result.framesPerExpert == 2)
        #expect(result.compressedBytes < UInt64(Self.experts * Self.stride))

        let packed = try Data(contentsOf: destination)
        #expect(UInt64(packed.count) == result.compressedBytes)
        let indexURL = directory.appendingPathComponent("layer_00.idx")
        let indexData = try Data(contentsOf: indexURL)
        #expect(UInt64(indexData.count) == result.indexBytes)
        let entries = try ExpertBlobIndex.decode(
            indexData, expectedCount: Self.experts, expectedFrames: 2)
        #expect(ExpertBlobIndex.maximumEnd(entries) == result.compressedBytes)

        for expert in 0..<Self.experts {
            var restored = [UInt8](repeating: 0, count: Self.stride)
            let halves = [0..<Self.split, Self.split..<Self.stride]
            var cursor = 0
            for (frame, range) in zip(entries[expert].frames, halves) {
                let start = Int(frame.offset)
                let frameData = packed.subdata(in: start..<start + Int(frame.length))
                try restored.withUnsafeMutableBytes { buffer in
                    try ExpertBlobCodec.decompress(
                        frameData,
                        into: try #require(buffer.baseAddress).advanced(by: cursor),
                        expected: range.count)
                }
                cursor += range.count
            }
            #expect(restored == Self.expertBytes(expert))
        }
    }

    @Test("A layer without a split is one frame per expert")
    func wholeLayerIsOneFrame() throws {
        let directory = try Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = directory.appendingPathComponent("layer_01.bin")
        try Self.rawLayer().write(to: source)
        let destination = directory.appendingPathComponent("layer_01.bin")
        let result = try CompressedExpertLayerWriter.write(
            sourcePath: source.path, destinationPath: destination.path,
            geometry: CompressedExpertLayerGeometry(
                expertStride: UInt64(Self.stride),
                expertsPerLayer: Self.experts,
                splitBytes: nil))
        #expect(result.framesPerExpert == 1)
        let entries = try ExpertBlobIndex.decode(
            try Data(contentsOf: directory.appendingPathComponent("layer_01.idx")),
            expectedCount: Self.experts, expectedFrames: 1)
        #expect(entries.allSatisfy { $0.frames.count == 1 })
    }

    @Test("A source that is not the layout's shape is refused")
    func refusesWrongSourceSize() throws {
        let directory = try Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = directory.appendingPathComponent("layer_00.bin")
        try Data(count: 100).write(to: source)
        #expect(throws: RepackError.self) {
            _ = try CompressedExpertLayerWriter.write(
                sourcePath: source.path,
                destinationPath: directory.appendingPathComponent("out.bin").path,
                geometry: CompressedExpertLayerGeometry(
                    expertStride: UInt64(Self.stride),
                    expertsPerLayer: Self.experts, splitBytes: nil))
        }
    }

    @Test("A split outside the stride is refused")
    func refusesBadSplit() {
        #expect(throws: RepackError.self) {
            try CompressedExpertLayerGeometry(
                expertStride: 1_000, expertsPerLayer: 2, splitBytes: 1_000
            ).validate()
        }
        #expect(throws: RepackError.self) {
            try CompressedExpertLayerGeometry(
                expertStride: 1_000, expertsPerLayer: 2, splitBytes: 0
            ).validate()
        }
    }

    @Test("The sidecar is named after the layer, not after its extension")
    func sidecarNaming() {
        #expect(
            CompressedExpertLayerWriter.indexPath(forLayerFile: "a/layer_07.bin")
                == "a/layer_07.idx")
        #expect(
            ExpertBlobIndex.sidecarName(forLayerFile: "layer_07.bin") == "layer_07.idx")
    }
}
