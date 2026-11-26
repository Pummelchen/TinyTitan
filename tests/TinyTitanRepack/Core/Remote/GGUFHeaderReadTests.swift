import Foundation
import Testing
import TinyTitanFormat

@testable import TinyTitanRepackCore

/// The local GGUF install's header reader: it starts at a fixed read size,
/// widens on `.needMoreBytes`, and must never grow past the parser's budget or
/// trap on a declared alignment. Synthetic headers only.
@Suite struct GGUFHeaderReadTests {
    private static func temporaryFile(_ bytes: [UInt8]) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gguf-header-\(UUID().uuidString).gguf")
        try Data(bytes).write(to: url)
        return url
    }

    private static func u32(_ value: UInt32) -> [UInt8] {
        withUnsafeBytes(of: value.littleEndian, Array.init)
    }

    private static func u64(_ value: UInt64) -> [UInt8] {
        withUnsafeBytes(of: value.littleEndian, Array.init)
    }

    /// An empty v3 header: 24 bytes, padded to the default 32-byte alignment.
    private static func emptyHeader() -> [UInt8] {
        var bytes = Array("GGUF".utf8)
        bytes += u32(3)
        bytes += u64(0)
        bytes += u64(0)
        bytes += [UInt8](repeating: 0, count: 8)
        return bytes
    }

    @Test func readsAValidHeaderAndItsExactBytes() throws {
        let bytes = Self.emptyHeader()
        let url = try Self.temporaryFile(bytes)
        defer { try? FileManager.default.removeItem(at: url) }

        let (header, headerBytes, fileSize) = try RemoteStreamingRepacker.readGGUFHeader(
            at: url.path, initialBytes: 16)
        #expect(header.dataStart == 32)
        #expect(headerBytes == bytes)
        #expect(fileSize == UInt64(bytes.count))
    }

    /// A header that ends past the first read grows it, and the grown read is
    /// still the exact header (no truncation, no padding carried into the
    /// sidecar beyond `dataStart`).
    @Test func growsPastTheInitialRead() throws {
        var bytes = Array("GGUF".utf8)
        bytes += Self.u32(3)
        bytes += Self.u64(0)
        bytes += Self.u64(1)
        // "general.alignment" = 64, as a uint64.
        let key = Array("general.alignment".utf8)
        bytes += Self.u64(UInt64(key.count))
        bytes += key
        bytes += Self.u32(10)
        bytes += Self.u64(64)
        let headerEnd = bytes.count
        let dataStart = (headerEnd + 63) / 64 * 64
        while bytes.count < dataStart { bytes.append(0) }

        let url = try Self.temporaryFile(bytes)
        defer { try? FileManager.default.removeItem(at: url) }
        let (header, headerBytes, _) = try RemoteStreamingRepacker.readGGUFHeader(
            at: url.path, initialBytes: 8)
        #expect(header.alignment == 64)
        #expect(header.dataStart == UInt64(dataStart))
        #expect(headerBytes.count == dataStart)
    }

    /// Before the parser bounded `dataStart`, this header made the caller
    /// evaluate `Int(1 << 63)`, which traps. It must be a typed error now.
    @Test func refusesAnAbsurdAlignmentInsteadOfTrapping() throws {
        var bytes = Array("GGUF".utf8)
        bytes += Self.u32(3)
        bytes += Self.u64(0)
        bytes += Self.u64(1)
        let key = Array("general.alignment".utf8)
        bytes += Self.u64(UInt64(key.count))
        bytes += key
        bytes += Self.u32(10)
        bytes += Self.u64(1 << 63)

        let url = try Self.temporaryFile(bytes)
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: GGUFHeaderV3.ReadError.self) {
            _ = try RemoteStreamingRepacker.readGGUFHeader(at: url.path, initialBytes: 16)
        }
    }

    @Test func refusesATruncatedHeader() throws {
        let bytes = Array(Self.emptyHeader().prefix(12))
        let url = try Self.temporaryFile(bytes)
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: GGUFHeaderV3.ReadError.self) {
            _ = try RemoteStreamingRepacker.readGGUFHeader(at: url.path, initialBytes: 8)
        }
    }
}
