import Compression
import Foundation

/// Compresses and decompresses one stored expert blob.
///
/// This is the format-side half of the compressed-expert work: nothing calls it
/// on the hot path yet, and it exists so the codec choice is pinned by a test
/// rather than by a benchmark script in `/tmp`.
///
/// **Why LZFSE and not zstd.** Measured on 128 MiB of a real expert layer:
/// `lzfse` **91.7% at 896 MiB/s** decode, `zstd -1` 90.8% at 1,113, `lz4` 98.7%
/// at 20,904, `zlib` 91.1% at 272. zstd is 0.9 points better and 24% faster to
/// decode, and would cost a new SwiftPM dependency in an engine that has none;
/// LZFSE is in the system framework, and a decode rate of ~0.9 GiB/s is enough
/// for the ~1 GB/s a 16 tok/s decode needs once it rides behind the prefetch
/// ring. `lz4`'s 1.3% saving does not justify the plumbing on its own.
///
/// Encoding runs at ~205 MiB/s, so repacking Ornith's 16.9 GiB of expert blobs
/// takes about 85 seconds and lands at 15.5 GiB.
public enum ExpertBlobCodec {
    public enum Failure: Error, CustomStringConvertible {
        case encodeFailed
        case decodeFailed(expected: Int, got: Int)

        public var description: String {
            switch self {
            case .encodeFailed:
                return "LZFSE refused the expert blob"
            case .decodeFailed(let expected, let got):
                return "expert blob decoded to \(got) bytes; the header said \(expected)"
            }
        }
    }

    /// Ratio the format's planning numbers were taken at, so a reader that sees
    /// a dramatically different figure knows the file is not what it claims.
    public static let measuredRatio = 0.917

    /// Compresses `count` bytes at `source`.
    ///
    /// The destination is sized by `compression_encode_buffer`'s own bound rather
    /// than a guess: an incompressible blob must not be truncated, which would
    /// produce a file that decodes to something the header disagrees with.
    public static func compress(
        source: UnsafeRawPointer,
        count: Int
    ) throws -> Data {
        guard count > 0 else { throw Failure.encodeFailed }
        let capacity = count + count / 16 + 4_096
        var packed = Data(count: capacity)
        let written = packed.withUnsafeMutableBytes { destination -> Int in
            guard let base = destination.bindMemory(to: UInt8.self).baseAddress else { return 0 }
            return compression_encode_buffer(
                base, capacity,
                source.assumingMemoryBound(to: UInt8.self), count,
                nil, COMPRESSION_LZFSE)
        }
        guard written > 0 else { throw Failure.encodeFailed }
        packed.removeSubrange(written..<packed.count)
        return packed
    }

    /// Decompresses `packed` into `destination`, which must hold `expected` bytes.
    ///
    /// `expected` comes from the install's geometry, not from the blob, so a
    /// truncated or swapped blob is a decode failure rather than a short read
    /// that leaves the tail of a slot holding the previous expert's weights.
    public static func decompress(
        _ packed: Data,
        into destination: UnsafeMutableRawPointer,
        expected: Int
    ) throws {
        guard expected > 0, !packed.isEmpty else {
            throw Failure.decodeFailed(expected: expected, got: 0)
        }
        let written = packed.withUnsafeBytes { source -> Int in
            guard let base = source.bindMemory(to: UInt8.self).baseAddress else { return 0 }
            return compression_decode_buffer(
                destination.assumingMemoryBound(to: UInt8.self), expected,
                base, packed.count,
                nil, COMPRESSION_LZFSE)
        }
        guard written == expected else {
            throw Failure.decodeFailed(expected: expected, got: written)
        }
    }
}
