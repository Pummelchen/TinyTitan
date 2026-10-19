import Foundation

/// The seek index a compressed expert layer needs.
///
/// An uncompressed layer is addressable by arithmetic: expert `n` is at
/// `n * expertStride`. A compressed one is not — every blob has its own length —
/// so the streamer needs (offset, length) per blob before it can read a specific
/// expert instead of decompressing everything in front of it.
///
/// **Two frames per expert when the layer splits.** The streamer can split an
/// expert read at the `down` tensor's offset, letting the GPU start on gate|up
/// while the tail is still in flight; both installed models split at 67% of the
/// stride. A compressed frame cannot be split once written, so a split layer
/// stores gate|up and the tail as *separate* frames and this index addresses each
/// one. A layer without a split stores one frame per expert.
///
/// The file is written beside the layer, one per layer, and is small: 512
/// experts with two frames is 16 bytes of header plus 24 bytes an expert, 12 KiB.
/// It is deliberately not embedded in the layer file, so an install can be
/// recompressed without rewriting 1.4 GiB.
///
/// Layout, little-endian throughout:
///
///     magic     4 bytes  "TTXI"
///     version   u16      currently 2
///     frames    u8       1 or 2, frames per expert
///     reserved  u8       zero
///     count     u32      number of experts
///     then per expert: `frames` x (offset u64, length u32)
public enum ExpertBlobIndex {
    public static let magic: [UInt8] = Array("TTXI".utf8)
    public static let currentVersion: UInt16 = 2
    public static let headerBytes = 12
    public static let frameBytes = 12
    public static let maximumFramesPerExpert = 2

    /// One compressed blob: a frame that decodes to a known number of bytes.
    public struct Blob: Sendable, Equatable {
        public let offset: UInt64
        public let length: UInt32

        public init(offset: UInt64, length: UInt32) {
            self.offset = offset
            self.length = length
        }
    }

    /// One expert's frames, in decode order: gate|up first when split, the tail
    /// second.
    public struct Entry: Sendable, Equatable {
        public let frames: [Blob]

        public init(frames: [Blob]) {
            self.frames = frames
        }
    }

    public enum Failure: Error, CustomStringConvertible {
        case badMagic
        case unsupportedVersion(UInt16)
        case badFrameCount(UInt8)
        case truncated(expected: Int, got: Int)
        case countMismatch(expected: Int, got: Int)
        case frameMismatch(expert: Int, expected: Int, got: Int)
        case emptyFrame(expert: Int, frame: Int)
        case notMonotonic(expert: Int, frame: Int)

        public var description: String {
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
            }
        }
    }

    public static func encode(_ entries: [Entry], framesPerExpert: Int) -> Data {
        precondition(
            (1...maximumFramesPerExpert).contains(framesPerExpert),
            "framesPerExpert must be 1 or 2")
        precondition(
            entries.allSatisfy { $0.frames.count == framesPerExpert },
            "every entry must carry exactly framesPerExpert frames")
        var data = Data(
            capacity: headerBytes + entries.count * framesPerExpert * frameBytes)
        data.append(contentsOf: magic)
        append(UInt16(currentVersion), to: &data)
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

    /// Decodes the index and checks the properties the reader relies on.
    ///
    /// `expectedCount` and `expectedFrames` come from the install's geometry, not
    /// from the file: an index written against another layer or another split
    /// would otherwise address plausible but wrong bytes, which is the failure
    /// this file exists to prevent.
    public static func decode(
        _ data: Data,
        expectedCount: Int,
        expectedFrames: Int
    ) throws -> [Entry] {
        let needed = headerBytes + expectedCount * expectedFrames * frameBytes
        guard data.count >= headerBytes else {
            throw Failure.truncated(expected: needed, got: data.count)
        }
        let bytes = [UInt8](data)
        guard Array(bytes[0..<4]) == magic else { throw Failure.badMagic }
        let version = read(UInt16.self, bytes, at: 4)
        guard version == currentVersion else {
            throw Failure.unsupportedVersion(version)
        }
        let frames = bytes[6]
        guard (1...UInt8(maximumFramesPerExpert)).contains(frames) else {
            throw Failure.badFrameCount(frames)
        }
        let count = Int(read(UInt32.self, bytes, at: 8))
        guard count == expectedCount else {
            throw Failure.countMismatch(expected: expectedCount, got: count)
        }
        guard Int(frames) == expectedFrames else {
            throw Failure.frameMismatch(
                expert: 0, expected: expectedFrames, got: Int(frames))
        }
        guard data.count >= needed else {
            throw Failure.truncated(expected: needed, got: data.count)
        }
        var entries: [Entry] = []
        entries.reserveCapacity(count)
        var cursor = headerBytes
        var previousEnd: UInt64 = 0
        for expert in 0..<count {
            var blobs: [Blob] = []
            for frame in 0..<Int(frames) {
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
                blobs.append(Blob(offset: offset, length: length))
                previousEnd = offset + UInt64(length)
                cursor += frameBytes
            }
            entries.append(Entry(frames: blobs))
        }
        return entries
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
