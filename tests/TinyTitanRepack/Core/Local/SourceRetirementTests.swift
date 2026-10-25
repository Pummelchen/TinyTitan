import Foundation
import Testing

@testable import TinyTitanRepackCore

/// The 122B local import retires a source shard when its last range is copied.
/// The accounting is what stands between this feature and deleting a shard
/// that a later range still needs, so the tests are about counts, not about
/// whether a file happened to disappear.
@Suite struct SourceRetirementTests {
    private static func temporaryRoot(_ name: String) -> String {
        let root = NSTemporaryDirectory() + "tt-retire-\(name)-\(UUID().uuidString)"
        try? FileManager.default.createDirectory(
            atPath: root, withIntermediateDirectories: true)
        return root
    }

    private static func copy(
        _ shard: String, offset: UInt64, size: UInt64
    ) -> CoalescedRangeCopy {
        CoalescedRangeCopy(
            id: "\(shard)@\(offset)",
            shardID: shard,
            sourceOffset: offset,
            size: size,
            destinations: [
                RangeCopy(
                    shardID: shard,
                    sourceOffset: offset,
                    size: size,
                    destinationPath: "/tmp/out.bin",
                    destinationOffset: 0)
            ])
    }

    @Test func keepsAShardUntilItsLastRangeIsConsumed() throws {
        let root = Self.temporaryRoot("last-range")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let shard = (root as NSString)
            .appendingPathComponent("model-00001-of-00002.safetensors")
        try Data(repeating: 0, count: 64).write(to: URL(fileURLWithPath: shard))
        let copies = [
            Self.copy("model-00001-of-00002.safetensors", offset: 0, size: 32),
            Self.copy("model-00001-of-00002.safetensors", offset: 32, size: 32),
        ]
        var retirement = SourceRetirement(copies: copies, enabled: true)

        retirement.consume(copies[0]) { shard }
        #expect(FileManager.default.fileExists(atPath: shard))
        retirement.consume(copies[1]) { shard }
        #expect(!FileManager.default.fileExists(atPath: shard))
    }

    @Test func retiresOnlyTheShardWhoseBytesAreExhausted() throws {
        let root = Self.temporaryRoot("two-shards")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let one = (root as NSString)
            .appendingPathComponent("model-00001-of-00002.safetensors")
        let two = (root as NSString)
            .appendingPathComponent("model-00002-of-00002.safetensors")
        for path in [one, two] {
            try Data(repeating: 0, count: 32).write(to: URL(fileURLWithPath: path))
        }
        let copies = [
            Self.copy("model-00001-of-00002.safetensors", offset: 0, size: 32),
            Self.copy("model-00002-of-00002.safetensors", offset: 0, size: 32),
        ]
        var retirement = SourceRetirement(copies: copies, enabled: true)

        retirement.consume(copies[0]) { one }
        #expect(!FileManager.default.fileExists(atPath: one))
        #expect(FileManager.default.fileExists(atPath: two))
    }

    @Test func disabledLeavesEveryShardInPlace() throws {
        let root = Self.temporaryRoot("disabled")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let shard = (root as NSString)
            .appendingPathComponent("model-00001-of-00002.safetensors")
        try Data(repeating: 0, count: 8).write(to: URL(fileURLWithPath: shard))
        let copies = [Self.copy("model-00001-of-00002.safetensors", offset: 0, size: 8)]
        var retirement = SourceRetirement(copies: copies, enabled: false)

        retirement.consume(copies[0]) { shard }
        #expect(FileManager.default.fileExists(atPath: shard))
    }

    /// The path guard, independent of the count: only a safetensors shard may
    /// be unlinked, even if a malformed plan named something else.
    @Test func refusesToUnlinkANonShard() throws {
        let root = Self.temporaryRoot("non-shard")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let index = (root as NSString).appendingPathComponent("model.safetensors.index.json")
        try Data("{}".utf8).write(to: URL(fileURLWithPath: index))
        #expect(!SourceRetirement.isSnapshotShard(index))
        #expect(
            SourceRetirement.isSnapshotShard(
                (root as NSString)
                    .appendingPathComponent("model-00001-of-00002.safetensors")))
    }
}
