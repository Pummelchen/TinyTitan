import Foundation

/// The seek index a compressed expert layer needs.
///
/// An uncompressed layer is addressable by arithmetic: expert `n` is at
/// `n * expertStride`. A compressed one is not — every blob has its own length —
/// so the streamer needs (offset, length) per expert before it can issue a read
/// for a specific expert instead of decompressing everything in front of it.
///
/// The file is written beside the layer, one per layer, and is small: a layer of
/// 512 experts costs 16 bytes of header plus 12 bytes an expert, 6 KiB. It is
/// deliberately not embedded in the layer file, so an install can be recompressed
/// — or, later, recompressed with a different codec — without rewriting 1.4 GiB.
///
/// Layout, little-endian throughout:
///
///     magic  4 bytes  "TTXI"
///     version u16     currently 1
///     count   u32     number of experts
///     then per expert: offset u64, length u32
public enum ExpertBlobIndex {
    public static let magic: [UInt8] = Array("TTXI".utf8)
    public static let currentVersion: UInt16 = 1
    public static let headerBytes = 10
    public static let entryBytes = 12

    public struct Entry: Sendable, Equatable {
        public let offset: UInt64
        public let length: UInt32

        public init(offset: UInt64, length: UInt32) {
            self.offset = offset
            self.length = length
        }
    }

    public enum Failure: Error, CustomStringConvertible {
        case badMagic
        case unsupportedVersion(UInt16)
        case truncated(expected: Int, got: Int)
        case countMismatch(expected: Int, got: Int)
        case emptyEntry(expert: Int)
        case notMonotonic(expert: Int)

        public var description: String {
            switch self {
            case .badMagic:
                return "expert blob index is not a TTXI file"
            case .unsupportedVersion(let version):
                return "expert blob index version \(version) is not readable here"
            case .truncated(let expected, let got):
                return "expert blob index is \(got) bytes; \(expected) are needed"
            case .countMismatch(let expected, let got):
                return "expert blob index declares \(got) experts; the layer has \(expected)"
            case .emptyEntry(let expert):
                return "expert blob index entry \(expert) has zero length"
            case .notMonotonic(let expert):
                return "expert blob index entry \(expert) does not start after the previous one"
            }
        }
    }

    public static func encode(_ entries: [Entry]) -> Data {
        var data = Data(capacity: headerBytes + entries.count * entryBytes)
        data.append(contentsOf: magic)
        append(UInt16(currentVersion), to: &data)
        append(UInt32(entries.count), to: &data)
        for entry in entries {
            append(entry.offset, to: &data)
            append(entry.length, to: &data)
        }
        return data
    }

    /// Decodes the index and checks the properties the reader relies on.
    ///
    /// `expectedCount` is the layer's expert count from the install's geometry:
    /// an index written against a different model would otherwise address
    /// plausible but wrong bytes, which is the failure mode this whole file
    /// exists to prevent.
    public static func decode(
        _ data: Data,
        expectedCount: Int
    ) throws -> [Entry] {
        let needed = headerBytes + expectedCount * entryBytes
        guard data.count >= headerBytes else {
            throw Failure.truncated(expected: needed, got: data.count)
        }
        let bytes = [UInt8](data)
        guard Array(bytes[0..<4]) == magic else { throw Failure.badMagic }
        let version = read(UInt16.self, bytes, at: 4)
        guard version == currentVersion else {
            throw Failure.unsupportedVersion(version)
        }
        let count = Int(read(UInt32.self, bytes, at: 6))
        guard count == expectedCount else {
            throw Failure.countMismatch(expected: expectedCount, got: count)
        }
        guard data.count >= needed else {
            throw Failure.truncated(expected: needed, got: data.count)
        }
        var entries: [Entry] = []
        entries.reserveCapacity(count)
        var cursor = headerBytes
        var previousEnd: UInt64 = 0
        for expert in 0..<count {
            let offset = read(UInt64.self, bytes, at: cursor)
            let length = read(UInt32.self, bytes, at: cursor + 8)
            guard length > 0 else { throw Failure.emptyEntry(expert: expert) }
            // Overlap would decode one expert's bytes into another's slot, which
            // is silent weight corruption rather than a read error.
            guard offset >= previousEnd else {
                throw Failure.notMonotonic(expert: expert)
            }
            entries.append(Entry(offset: offset, length: length))
            previousEnd = offset + UInt64(length)
            cursor += entryBytes
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
