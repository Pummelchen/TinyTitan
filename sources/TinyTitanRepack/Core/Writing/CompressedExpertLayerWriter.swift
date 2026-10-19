import Foundation
import TinyTitanFormat

/// The geometry one raw expert layer is compressed against.
///
/// It is the layer's own contract, not a whole-install one: every layer of an
/// install shares `expertStride` and `expertsPerLayer`, but `splitBytes` comes
/// from that layer's `down` tensor offset and can differ between layers.
public struct CompressedExpertLayerGeometry: Sendable, Equatable {
    public let expertStride: UInt64
    public let expertsPerLayer: Int
    /// Bytes of the expert blob that decode's phase 1 (gate|up) consumes.
    ///
    /// `nil` means the whole expert is one frame. A split has to be strictly
    /// interior, exactly as the streamer requires it.
    public let splitBytes: UInt64?
    /// Independent block size every phase is cut into.
    ///
    /// A compressed stream decodes on one core, so one big frame per phase
    /// becomes the layer's latency floor (1.2 ms for Ornith's 1.18 MB front).
    /// 256 KiB measured 0.1% *better* in ratio than whole-phase frames on
    /// Ornith layer 20, and gives each expert 8 independently decodable blocks.
    public let blockBytes: UInt64

    public init(
        expertStride: UInt64, expertsPerLayer: Int, splitBytes: UInt64?,
        blockBytes: UInt64 = 256 * 1024
    ) {
        self.expertStride = expertStride
        self.expertsPerLayer = expertsPerLayer
        self.splitBytes = splitBytes
        self.blockBytes = blockBytes
    }

    /// One or two, matching `ExpertBlobIndex`'s own limit.
    public var framesPerExpert: Int { splitBytes == nil ? 1 : 2 }

    /// Bytes one expert occupies in the raw layer.
    public var rawBytes: UInt64 { expertStride * UInt64(expertsPerLayer) }

    /// The halves of one expert, in decode order.
    public var frameRanges: [Range<UInt64>] {
        guard let splitBytes else { return [0..<expertStride] }
        return [0..<splitBytes, splitBytes..<expertStride]
    }

    /// Refuses a shape that would produce a file the streamer cannot read.
    public func validate() throws {
        guard expertsPerLayer > 0 else {
            throw RepackError.configurationInvalid(
                detail: "a compressed layer needs at least one expert")
        }
        guard expertStride > 0, expertStride <= UInt64(Int.max) else {
            throw RepackError.configurationInvalid(
                detail: "expert stride \(expertStride) is not addressable")
        }
        // The streamer's own rule: an interior boundary is the only kind that
        // lets phase 1 start while the tail is in flight. Anything else is a
        // whole-expert read, so the writer must not write two frames for it.
        if let splitBytes {
            guard splitBytes > 0, splitBytes < expertStride else {
                throw RepackError.configurationInvalid(
                    detail: "expert split \(splitBytes) is not inside the "
                        + "\(expertStride)-byte stride")
            }
        }
        guard blockBytes >= 4096, blockBytes <= (1 << 20),
            blockBytes & (blockBytes - 1) == 0
        else {
            throw RepackError.configurationInvalid(
                detail: "block size \(blockBytes) must be a power of two in 4 KiB...1 MiB")
        }
    }
}

public struct CompressedExpertLayerResult: Sendable, Equatable {
    public let compressedBytes: UInt64
    public let indexBytes: UInt64
    public let sha256: String
    public let indexSha256: String
    public let blocksPerExpert: Int
    public let blockBytes: UInt64
}

/// Writes a raw expert layer as LZFSE frames, plus the index that addresses them.
///
/// The output is what the streamer's compressed path reads: one frame per
/// expert, or two when the layer splits (gate|up and the tail), laid back to
/// back, with `layer_NN.idx` naming each frame's `(offset, length)`. The index is
/// a separate file for the reason the format gives -- an install can be
/// recompressed without rewriting its layer files.
///
/// The raw layer is not modified: this reads it and writes new files beside the
/// requested destination, so a caller can keep both formats (which is what a
/// paired A/B needs).
public enum CompressedExpertLayerWriter {
    /// Compresses `sourcePath` into `destinationPath` and writes the sidecar.
    ///
    /// Both files are written to temporary names and renamed into place, so an
    /// interrupted run leaves no half layer that a reader could open: a layer
    /// without its index is refused at load instead of being read as raw.
    public static func write(
        sourcePath: String,
        destinationPath: String,
        geometry: CompressedExpertLayerGeometry
    ) throws -> CompressedExpertLayerResult {
        try geometry.validate()
        let fd = try Posix.openRead(sourcePath)
        defer { close(fd) }
        let sourceSize = try Posix.fileSize(fd: fd, path: sourcePath)
        guard sourceSize == geometry.rawBytes else {
            throw RepackError.configurationInvalid(
                detail: "\(sourcePath) is \(sourceSize) bytes; the layout says "
                    + "\(geometry.rawBytes) for \(geometry.expertsPerLayer) experts "
                    + "of \(geometry.expertStride) bytes")
        }
        let temporaryPath = destinationPath + ".compressing"
        let temporaryIndexPath = indexPath(forLayerFile: destinationPath) + ".compressing"
        let result: CompressedExpertLayerResult
        do {
            result = try writeFrames(
                fd: fd, sourcePath: sourcePath,
                destinationPath: temporaryPath, indexPath: temporaryIndexPath,
                geometry: geometry)
            try Posix.rename(from: temporaryPath, to: destinationPath)
            try Posix.rename(
                from: temporaryIndexPath, to: indexPath(forLayerFile: destinationPath))
        } catch {
            try? FileManager.default.removeItem(atPath: temporaryPath)
            try? FileManager.default.removeItem(atPath: temporaryIndexPath)
            throw error
        }
        try Posix.fsyncDirectory((destinationPath as NSString).deletingLastPathComponent)
        return result
    }

    /// The sidecar path this writer produces for a layer file.
    public static func indexPath(forLayerFile path: String) -> String {
        let name = ExpertBlobIndex.sidecarName(forLayerFile: (path as NSString).lastPathComponent)
        return ((path as NSString).deletingLastPathComponent as NSString)
            .appendingPathComponent(name)
    }

    /// Compress every expert into the temporary payload, then write the index.
    ///
    /// One expert-sized buffer serves the whole layer: the stride is a couple of
    /// megabytes, and a per-expert allocation would put 256 of them on the heap
    /// for every layer of a 40-layer model.
    private static func writeFrames(
        fd: Int32,
        sourcePath: String,
        destinationPath: String,
        indexPath: String,
        geometry: CompressedExpertLayerGeometry
    ) throws -> CompressedExpertLayerResult {
        let outFD = try Posix.openCreateRW(destinationPath)
        defer { close(outFD) }
        let stride = Int(geometry.expertStride)
        let buffer = UnsafeMutableRawPointer.allocate(byteCount: stride, alignment: 4096)
        defer { buffer.deallocate() }
        var hasher = Sha256Stream()
        var entries: [ExpertBlobIndex.Entry] = []
        entries.reserveCapacity(geometry.expertsPerLayer)
        var written: UInt64 = 0
        let ranges = geometry.frameRanges.map { Int($0.lowerBound)..<Int($0.upperBound) }
        for expert in 0..<geometry.expertsPerLayer {
            let blocks = try compressExpert(
                fd: fd, sourcePath: sourcePath, expert: expert,
                ranges: ranges, blockBytes: Int(geometry.blockBytes),
                buffer: buffer, outFD: outFD,
                destinationPath: destinationPath, written: &written, hasher: &hasher)
            entries.append(.init(frames: blocks))
        }
        try Posix.fsync(outFD, path: destinationPath)
        let indexData = ExpertBlobIndex.encode(
            entries, blockBytes: geometry.blockBytes)
        try writeIndex(indexData, to: indexPath)
        return CompressedExpertLayerResult(
            compressedBytes: written,
            indexBytes: UInt64(indexData.count),
            sha256: hasher.finalizeHexString(),
            indexSha256: Self.sha256(of: indexData),
            blocksPerExpert: entries.first?.frames.count ?? 0,
            blockBytes: geometry.blockBytes)
    }

    /// Reads, compresses and writes the blocks of one expert.
    ///
    /// Each phase is read once and then cut into `blockBytes` blocks, so the
    /// source read stays the sequential phase it was and only the compressed
    /// output gains the parallelism.
    private static func compressExpert(
        fd: Int32,
        sourcePath: String,
        expert: Int,
        ranges: [Range<Int>],
        blockBytes: Int,
        buffer: UnsafeMutableRawPointer,
        outFD: Int32,
        destinationPath: String,
        written: inout UInt64,
        hasher: inout Sha256Stream
    ) throws -> [ExpertBlobIndex.Blob] {
        var blocks: [ExpertBlobIndex.Blob] = []
        let base = UInt64(expert) * UInt64(ranges.reduce(0) { $0 + $1.count })
        for range in ranges {
            try Posix.preadAll(
                fd: fd, path: sourcePath, buf: buffer, count: range.count,
                offset: base + UInt64(range.lowerBound))
            var offset = 0
            while offset < range.count {
                let count = min(blockBytes, range.count - offset)
                let packed = try ExpertBlobCodec.compress(
                    source: UnsafeRawPointer(buffer).advanced(by: offset), count: count)
                guard packed.count > 0, packed.count <= Int(UInt32.max) else {
                    throw RepackError.configurationInvalid(
                        detail: "expert \(expert) block compressed to \(packed.count) bytes")
                }
                try packed.withUnsafeBytes { raw in
                    guard let baseAddress = raw.baseAddress else {
                        throw RepackError.configurationInvalid(
                            detail: "expert \(expert) block has no bytes")
                    }
                    try Posix.pwriteAll(
                        fd: outFD, path: destinationPath, buf: baseAddress,
                        count: raw.count, offset: written)
                    hasher.update(raw)
                }
                blocks.append(.init(offset: written, length: UInt32(packed.count)))
                written += UInt64(packed.count)
                offset += count
            }
        }
        return blocks
    }

    private static func writeIndex(_ data: Data, to path: String) throws {
        try Posix.atomicWrite(
            data, to: path, durableIn: (path as NSString).deletingLastPathComponent)
    }

    private static func sha256(of data: Data) -> String {
        var hasher = Sha256Stream()
        data.withUnsafeBytes { hasher.update($0) }
        return hasher.finalizeHexString()
    }
}
