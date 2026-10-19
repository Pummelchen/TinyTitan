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
package enum ExpertBlobIndex {
    package static let magic: [UInt8] = Array("TTXI".utf8)
    package static let currentVersion: UInt16 = 2
    package static let headerBytes = 12
    package static let frameBytes = 12
    package static let maximumFramesPerExpert = 2

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
            }
        }
    }

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
    package static func decode(
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
            .multipliedReportingOverflow(by: UInt64(maximumFramesPerExpert))
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
