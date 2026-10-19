import Foundation

/// The seek index a compressed expert layer needs.
///
/// An uncompressed layer is addressable by arithmetic: expert `n` is at
/// `n * expertStride`. A compressed one is not — every blob has its own length —
/// so the streamer needs (offset, length) per blob before it can read a specific
/// expert instead of decompressing everything in front of it.
///
/// **Blocks, not whole phases.** A compressed stream is serial: one LZFSE frame
/// decodes on one core no matter how many threads are free, so a single frame per
/// phase puts that frame's decode latency (1.2 ms for gate|up's 1.18 MB) on the
/// layer's critical path. Version 3 therefore splits every phase into independent
/// blocks (`blockBytes`, 256 KiB by default) and addresses each. Measured on
/// Ornith 4-bit layer 20: 256 KiB blocks compress to 0.9151 of the raw bytes
/// against 0.9162 for whole-phase frames -- the split costs nothing, and encoding
/// is slightly faster.
///
/// The phase split itself is still the reason for more than one block group:
/// `down` sits at 67% of the stride on both installed 35B families, so gate|up
/// and the tail decode as separate groups and the GPU can start on phase 1 while
/// the tail is still in flight.
///
/// The file is written beside the layer, one per layer, and is small: 512
/// experts with 8 blocks is 16 bytes of header plus 48 KiB of entries. It is
/// deliberately not embedded in the layer file, so an install can be recompressed
/// without rewriting 1.4 GiB.
///
/// Layout, little-endian throughout:
///
///     magic     4 bytes  "TTXI"
///     version   u16      3 (2 is still readable: 1 or 2 whole-phase frames)
///     exponent  u8       log2(blockBytes), 12...20
///     reserved  u8       zero
///     count     u32      number of experts
///     then per expert: one (offset u64, length u32) per block, in decode order:
///     phase 0's blocks, then phase 1's
package enum ExpertBlobIndex {
    package static let magic: [UInt8] = Array("TTXI".utf8)
    package static let currentVersion: UInt16 = 3
    /// Version 2 stored one or two whole-phase frames per expert; it is still
    /// read so an install converted before the block format keeps working.
    package static let legacyVersion: UInt16 = 2
    package static let headerBytes = 12
    package static let frameBytes = 12
    package static let maximumFramesPerExpert = 2
    /// A v3 expert's blocks: two phases of at most 64 KiB blocks on a 4 MiB
    /// phase. The cap only bounds the allocation a corrupt index can ask for.
    package static let maximumBlocksPerExpert = 128
    package static let minimumBlockExponent = 12
    package static let maximumBlockExponent = 20

    /// One decoded index: the blocks themselves plus the block size that gives
    /// them meaning. `blockBytes == 0` is a legacy (v2) whole-phase file.
    package struct Layout: Sendable, Equatable {
        package let entries: [Entry]
        package let blockBytes: UInt64
    }

    /// The number of blocks one phase is cut into.
    package static func blockCount(phaseBytes: UInt64, blockBytes: UInt64) -> Int {
        guard blockBytes > 0, phaseBytes > 0 else { return 0 }
        return Int((phaseBytes + blockBytes - 1) / blockBytes)
    }

    /// The decoded size of block `index` in a phase of `phaseBytes`.
    package static func blockOutputBytes(
        phaseBytes: UInt64, blockBytes: UInt64, index: Int
    ) -> UInt64 {
        let start = UInt64(index) * blockBytes
        return min(blockBytes, phaseBytes - start)
    }

    /// One compressed blob: a frame that decodes to a known number of bytes.
    package struct Blob: Sendable, Equatable {
        package let offset: UInt64
        package let length: UInt32

        package init(offset: UInt64, length: UInt32) {
            self.offset = offset
            self.length = length
        }
    }

    /// One expert's frames, in decode order: gate|up first when split, the tail
    /// second.
    package struct Entry: Sendable, Equatable {
        package let frames: [Blob]

        package init(frames: [Blob]) {
            self.frames = frames
        }
    }

    package enum Failure: Error, CustomStringConvertible {
        case badMagic
        case unsupportedVersion(UInt16)
        case badFrameCount(UInt8)
        case truncated(expected: Int, got: Int)
        case countMismatch(expected: Int, got: Int)
        case frameMismatch(expert: Int, expected: Int, got: Int)
        case emptyFrame(expert: Int, frame: Int)
        case notMonotonic(expert: Int, frame: Int)
        case frameOverflow(expert: Int, frame: Int)
        case badBlockExponent(UInt8)
        case badBlockCount(Int)

        package var description: String {
            switch self {
            case .badMagic:
                return "expert blob index is not a TTXI file"
            case .unsupportedVersion(let version):
                return "expert blob index version \(version) is not readable here"
            case .badFrameCount(let frames):
                return "expert blob index declares \(frames) frames per expert; 1 or 2 are readable"
            case .truncated(let expected, let got):
                return "expert blob index is \(got) bytes; \(expected) are needed"
            case .countMismatch(let expected, let got):
                return "expert blob index declares \(got) experts; the layer has \(expected)"
            case .frameMismatch(let expert, let expected, let got):
                return "expert \(expert) has \(got) frames; the layer splits into \(expected)"
            case .emptyFrame(let expert, let frame):
                return "expert \(expert) frame \(frame) has zero length"
            case .notMonotonic(let expert, let frame):
                return "expert \(expert) frame \(frame) does not start after the previous one"
            case .frameOverflow(let expert, let frame):
                return "expert \(expert) frame \(frame) ends past the addressable file"
            case .badBlockExponent(let exponent):
                return "expert blob index block exponent \(exponent) is outside 12...20"
            case .badBlockCount(let frames):
                return
                    "expert blob index declares \(frames) blocks per expert; 1...\(maximumBlocksPerExpert) are readable"
            }
        }
    }

    /// Writes a legacy (v2) index: one or two whole-phase frames per expert.
    ///
    /// Kept for tests and for anything that still wants the old shape; the
    /// converter and the writer emit `encode(_:blockBytes:)`.
    package static func encode(_ entries: [Entry], framesPerExpert: Int) -> Data {
        precondition(
            (1...maximumFramesPerExpert).contains(framesPerExpert),
            "framesPerExpert must be 1 or 2")
        precondition(
            entries.allSatisfy { $0.frames.count == framesPerExpert },
            "every entry must carry exactly framesPerExpert frames")
        var data = Data(
            capacity: headerBytes + entries.count * framesPerExpert * frameBytes)
        data.append(contentsOf: magic)
        append(legacyVersion, to: &data)
        data.append(UInt8(framesPerExpert))
        data.append(0)
        append(UInt32(entries.count), to: &data)
        for entry in entries {
            for frame in entry.frames {
                append(frame.offset, to: &data)
                append(frame.length, to: &data)
            }
        }
        return data
    }

    /// Writes a v3 index: every phase split into independent `blockBytes` blocks.
    package static func encode(_ entries: [Entry], blockBytes: UInt64) -> Data {
        precondition(
            blockBytes > 0 && blockBytes & (blockBytes - 1) == 0,
            "blockBytes must be a power of two")
        let exponent = UInt8(blockBytes.trailingZeroBitCount)
        precondition(
            (minimumBlockExponent...maximumBlockExponent).contains(Int(exponent)),
            "blockBytes is outside the readable range")
        let frames = entries.first?.frames.count ?? 0
        precondition(
            entries.allSatisfy { $0.frames.count == frames },
            "every entry must carry the same number of blocks")
        var data = Data(capacity: headerBytes + entries.count * frames * frameBytes)
        data.append(contentsOf: magic)
        append(currentVersion, to: &data)
        data.append(exponent)
        data.append(0)
        append(UInt32(entries.count), to: &data)
        for entry in entries {
            for frame in entry.frames {
                append(frame.offset, to: &data)
                append(frame.length, to: &data)
            }
        }
        return data
    }

    /// Decodes a legacy (v2) index and checks the properties the reader relies on.
    ///
    /// `expectedCount` and `expectedFrames` come from the install's geometry, not
    /// from the file: an index written against another layer or another split
    /// would otherwise address plausible but wrong bytes, which is the failure
    /// this file exists to prevent.
    package static func decode(
        _ data: Data,
        expectedCount: Int,
        expectedFrames: Int
    ) throws -> [Entry] {
        guard data.count >= headerBytes else {
            throw Failure.truncated(expected: headerBytes, got: data.count)
        }
        let bytes = [UInt8](data)
        guard Array(bytes[0..<4]) == magic else { throw Failure.badMagic }
        let version = read(UInt16.self, bytes, at: 4)
        guard version == legacyVersion else {
            throw Failure.unsupportedVersion(version)
        }
        let frames = Int(bytes[6])
        guard (1...maximumFramesPerExpert).contains(frames) else {
            throw Failure.badFrameCount(UInt8(frames))
        }
        let count = Int(read(UInt32.self, bytes, at: 8))
        guard count == expectedCount else {
            throw Failure.countMismatch(expected: expectedCount, got: count)
        }
        guard frames == expectedFrames else {
            throw Failure.frameMismatch(
                expert: 0, expected: expectedFrames, got: frames)
        }
        let needed = headerBytes + count * frames * frameBytes
        guard data.count >= needed else {
            throw Failure.truncated(expected: needed, got: data.count)
        }
        return try parseEntries(
            bytes, count: count, framesPerExpert: frames, expectedTotal: needed)
    }

    /// Decodes a v2 or v3 index into blocks the reader can schedule.
    ///
    /// The phase sizes come from the install's geometry, not from the file, and
    /// they decide how many blocks each expert must carry: an index written for
    /// another layer, another split, or another block size is refused instead of
    /// decoding plausible bytes into the wrong offsets.
    package static func decodeLayout(
        _ data: Data,
        expectedCount: Int,
        phase0Bytes: UInt64,
        phase1Bytes: UInt64?
    ) throws -> Layout {
        guard data.count >= headerBytes else {
            throw Failure.truncated(expected: headerBytes, got: data.count)
        }
        let bytes = [UInt8](data)
        guard Array(bytes[0..<4]) == magic else { throw Failure.badMagic }
        let version = read(UInt16.self, bytes, at: 4)
        if version == legacyVersion {
            let entries = try decode(
                data, expectedCount: expectedCount,
                expectedFrames: phase1Bytes == nil ? 1 : 2)
            return Layout(entries: entries, blockBytes: 0)
        }
        guard version == currentVersion else {
            throw Failure.unsupportedVersion(version)
        }
        let exponent = Int(bytes[6])
        guard (minimumBlockExponent...maximumBlockExponent).contains(exponent) else {
            throw Failure.badBlockExponent(bytes[6])
        }
        let blockBytes = UInt64(1) << exponent
        var frames = blockCount(phaseBytes: phase0Bytes, blockBytes: blockBytes)
        if let phase1Bytes {
            frames += blockCount(phaseBytes: phase1Bytes, blockBytes: blockBytes)
        }
        guard frames > 0, frames <= maximumBlocksPerExpert else {
            throw Failure.badBlockCount(frames)
        }
        let count = Int(read(UInt32.self, bytes, at: 8))
        guard count == expectedCount else {
            throw Failure.countMismatch(expected: expectedCount, got: count)
        }
        let needed = headerBytes + count * frames * frameBytes
        guard data.count >= needed else {
            throw Failure.truncated(expected: needed, got: data.count)
        }
        let entries = try parseEntries(
            bytes, count: count, framesPerExpert: frames, expectedTotal: needed)
        return Layout(entries: entries, blockBytes: blockBytes)
    }

    /// Parses `count` experts of `framesPerExpert` blobs each, in order.
    private static func parseEntries(
        _ bytes: [UInt8],
        count: Int,
        framesPerExpert: Int,
        expectedTotal: Int
    ) throws -> [Entry] {
        var entries: [Entry] = []
        entries.reserveCapacity(count)
        var cursor = headerBytes
        var previousEnd: UInt64 = 0
        for expert in 0..<count {
            var blobs: [Blob] = []
            blobs.reserveCapacity(framesPerExpert)
            for frame in 0..<framesPerExpert {
                let offset = read(UInt64.self, bytes, at: cursor)
                let length = read(UInt32.self, bytes, at: cursor + 8)
                guard length > 0 else {
                    throw Failure.emptyFrame(expert: expert, frame: frame)
                }
                // Overlap would decode one frame's bytes into another's slot,
                // which is silent weight corruption rather than a read error.
                guard offset >= previousEnd else {
                    throw Failure.notMonotonic(expert: expert, frame: frame)
                }
                // A wrapping end would otherwise read as "inside the file" to
                // every later `end <= streamSize` guard.
                let (frameEnd, overflow) = offset.addingReportingOverflow(UInt64(length))
                guard !overflow else {
                    throw Failure.frameOverflow(expert: expert, frame: frame)
                }
                blobs.append(Blob(offset: offset, length: length))
                previousEnd = frameEnd
                cursor += frameBytes
            }
            entries.append(Entry(frames: blobs))
        }
        return entries
    }

    /// Where block `index` decodes to, how many bytes it must produce, and
    /// which phase it belongs to.
    ///
    /// `blockBytes == 0` is the legacy (v2) shape: index 0 is the whole of phase
    /// 0 and index 1 (when the layer splits) is the tail. A v3 index cuts both
    /// phases into `blockBytes` blocks, phase 0 first.
    package static func blockDestination(
        index: Int,
        phase0Bytes: UInt64,
        phase1Bytes: UInt64?,
        blockBytes: UInt64
    ) -> (offset: UInt64, expected: UInt64, phase: Int)? {
        if blockBytes == 0 {
            switch index {
            case 0:
                return (0, phase0Bytes, 0)
            case 1 where phase1Bytes != nil:
                return (phase0Bytes, phase1Bytes ?? 0, 1)
            default:
                return nil
            }
        }
        let phase0Blocks = blockCount(phaseBytes: phase0Bytes, blockBytes: blockBytes)
        if index < phase0Blocks {
            return (
                UInt64(index) * blockBytes,
                blockOutputBytes(
                    phaseBytes: phase0Bytes, blockBytes: blockBytes, index: index),
                0
            )
        }
        guard let phase1Bytes else { return nil }
        let tailIndex = index - phase0Blocks
        guard tailIndex < blockCount(phaseBytes: phase1Bytes, blockBytes: blockBytes) else {
            return nil
        }
        return (
            phase0Bytes + UInt64(tailIndex) * blockBytes,
            blockOutputBytes(
                phaseBytes: phase1Bytes, blockBytes: blockBytes, index: tailIndex),
            1
        )
    }

    /// The byte after the last frame's end: the compressed layer's stream size.
    ///
    /// A compressed layer is not `expertsPerLayer * expertStride` bytes long,
    /// and the streamer validates its file size against this instead. The
    /// returned value assumes entries that came from `decode`, which already
    /// refused a wrapping frame end.
    package static func maximumEnd(_ entries: [Entry]) -> UInt64 {
        maximumEnd(entries.map(\.frames))
    }

    /// The same for raw frame lists, which is what the streamer holds.
    package static func maximumEnd(_ entries: [[Blob]]) -> UInt64 {
        var end: UInt64 = 0
        for entry in entries {
            for frame in entry {
                end = max(end, frame.offset + UInt64(frame.length))
            }
        }
        return end
    }

    /// The largest index a layer of `expertsPerLayer` experts can have.
    ///
    /// Used as the allocation cap when a loader reads a sidecar: the file is
    /// 12 bytes of header plus 24 bytes per expert for a split layer, so a
    /// document claiming more is not one this format wrote. nil means the
    /// expert count is too large to size at all.
    package static func maximumIndexBytes(expertsPerLayer: Int) -> UInt64? {
        guard expertsPerLayer > 0 else { return UInt64(headerBytes) }
        let (frames, framesOverflow) = UInt64(expertsPerLayer)
            .multipliedReportingOverflow(by: UInt64(maximumBlocksPerExpert))
        guard !framesOverflow else { return nil }
        let (body, bodyOverflow) = frames.multipliedReportingOverflow(by: UInt64(frameBytes))
        guard !bodyOverflow,
            body <= UInt64.max - UInt64(headerBytes)
        else { return nil }
        return UInt64(headerBytes) + body
    }

    /// Where a layer file's index lives: `layer_00.bin` -> `layer_00.idx`.
    ///
    /// One rule for the writer, the loader and the verifier; spelling it three
    /// times is how a sidecar ends up named `.bin.idx` in one of them.
    package static func sidecarName(forLayerFile file: String) -> String {
        (file as NSString).deletingPathExtension + ".idx"
    }

    /// Where a compressed expert's phase-1 frame ends, or nil for a whole blob.
    ///
    /// The rule is the reader's, not the writer's: an interior `down` offset
    /// lets the GPU start on gate|up while the tail is still in flight, and
    /// anything else (no `down` entry, or padding that pushed it to the end) is
    /// a whole-expert read. Both sides call this so a layer can never be
    /// written with a split the streamer refuses.
    package static func phase1Bytes(downOffset: UInt64?, expertStride: UInt64) -> UInt64? {
        guard let downOffset, downOffset > 0, downOffset < expertStride else {
            return nil
        }
        return downOffset
    }

    private static func append<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
        withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
    }

    private static func read<T: FixedWidthInteger>(
        _ type: T.Type, _ bytes: [UInt8], at offset: Int
    ) -> T {
        var value: T = 0
        withUnsafeMutableBytes(of: &value) { destination in
            destination.copyBytes(from: bytes[offset..<offset + MemoryLayout<T>.size])
        }
        return T(littleEndian: value)
    }
}
