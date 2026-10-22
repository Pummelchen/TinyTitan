import Foundation
import Testing
import TinyTitan

@testable import TinyTitanLib

@Suite struct PromptCheckpointStoreTests {
    private let domain = ServerPromptCacheDomain(
        modelID: "ornith",
        sourceSnapshotHash: "source",
        runtimeProfileHash: "profile",
        maximumContext: 8192,
        kvStorage: "int8",
        fp16RingEnabled: false,
        templateSHA256: "template")

    private func descriptor(
        position: Int, payloadBytes: Int
    ) -> InferenceStateSnapshotDescriptor {
        InferenceStateSnapshotDescriptor(
            position: position,
            kvSegmentLengths: [payloadBytes],
            gdnSegmentLengths: [],
            payloadBytes: payloadBytes)
    }

    private func metadata(
        tokens: [Int32],
        kind: PromptCheckpointKind = .systemEnd,
        conversation: String = "conversation",
        position: Int? = nil,
        key: PromptCheckpointKey? = nil,
        createdAt: Date = Date(),
        domain: ServerPromptCacheDomain? = nil
    ) -> PromptCheckpointMetadata {
        let resolvedDomain = domain ?? self.domain
        return PromptCheckpointMetadata(
            key: key
                ?? PromptCheckpointKey.make(
                    domain: resolvedDomain, prefixTokenIDs: tokens),
            domain: resolvedDomain,
            kind: kind,
            conversationKey: conversation,
            prefixTokenIDs: tokens,
            descriptor: descriptor(
                position: position ?? tokens.count,
                payloadBytes: 16),
            createdAt: createdAt)
    }

    private func snapshot(
        for metadata: PromptCheckpointMetadata, byte: UInt8 = 7
    ) -> InferenceStateSnapshot {
        InferenceStateSnapshot(
            descriptor: metadata.descriptor,
            payload: Data(repeating: byte, count: metadata.descriptor.payloadBytes))
    }

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("checkpoints-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: url, withIntermediateDirectories: true)
        return url
    }

    private func memoryStore(limit: Int = 1 << 20) throws -> PromptCheckpointStore {
        try PromptCheckpointStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: limit, diskDirectory: nil,
                diskLimitBytes: 0))
    }

    @Test func memoryRoundTrip() async throws {
        let store = try memoryStore()
        let meta = metadata(tokens: [1, 2, 3])
        try await store.save(meta, snapshot: snapshot(for: meta))
        let present = await store.contains(meta.key)
        #expect(present)
        let loaded = try await store.load(meta.key, domain: domain)
        #expect(loaded.metadata == meta)
        #expect(loaded.snapshot.payload == snapshot(for: meta).payload)
    }

    @Test func contentHashIsPrefixAndDomainSensitive() {
        let base = PromptCheckpointKey.make(
            domain: domain, prefixTokenIDs: [1, 2, 3])
        #expect(
            base
                == PromptCheckpointKey.make(
                    domain: domain, prefixTokenIDs: [1, 2, 3]))
        #expect(
            base
                != PromptCheckpointKey.make(
                    domain: domain, prefixTokenIDs: [1, 2, 4]))
        #expect(
            base
                != PromptCheckpointKey.make(
                    domain: domain, prefixTokenIDs: [1, 2]))
        var other = domain
        other = ServerPromptCacheDomain(
            modelID: "other", sourceSnapshotHash: other.sourceSnapshotHash,
            runtimeProfileHash: other.runtimeProfileHash,
            maximumContext: other.maximumContext, kvStorage: other.kvStorage,
            fp16RingEnabled: other.fp16RingEnabled,
            templateSHA256: other.templateSHA256)
        #expect(
            base
                != PromptCheckpointKey.make(
                    domain: other, prefixTokenIDs: [1, 2, 3]))
    }

    @Test func positionAndKeyMismatchesAreRefused() async throws {
        let store = try memoryStore()
        let wrongPosition = metadata(tokens: [1, 2], position: 3)
        await #expect(throws: PromptCheckpointStoreError.self) {
            try await store.save(
                wrongPosition,
                snapshot: self.snapshot(
                    for: wrongPosition))
        }
        let foreignKey = PromptCheckpointKey.make(
            domain: domain, prefixTokenIDs: [9, 9])
        let wrongKey = metadata(tokens: [1, 2], key: foreignKey)
        await #expect(throws: PromptCheckpointStoreError.self) {
            try await store.save(wrongKey, snapshot: self.snapshot(for: wrongKey))
        }
    }

    @Test func payloadSizeMismatchIsRefused() async throws {
        let store = try memoryStore()
        let meta = metadata(tokens: [1, 2, 3])
        let bad = InferenceStateSnapshot(
            descriptor: meta.descriptor,
            payload: Data(repeating: 1, count: 8))
        await #expect(throws: PromptCheckpointStoreError.self) {
            try await store.save(meta, snapshot: bad)
        }
    }

    @Test func domainMismatchOnLoadIsRefused() async throws {
        let store = try memoryStore()
        let meta = metadata(tokens: [1, 2, 3])
        try await store.save(meta, snapshot: snapshot(for: meta))
        let other = ServerPromptCacheDomain(
            modelID: "other", sourceSnapshotHash: domain.sourceSnapshotHash,
            runtimeProfileHash: domain.runtimeProfileHash,
            maximumContext: domain.maximumContext, kvStorage: domain.kvStorage,
            fp16RingEnabled: domain.fp16RingEnabled,
            templateSHA256: domain.templateSHA256)
        await #expect(throws: PromptCheckpointStoreError.self) {
            try await store.load(meta.key, domain: other)
        }
        let stillThere = try await store.load(meta.key, domain: domain)
        #expect(stillThere.metadata == meta)
    }

    @Test func memoryLimitEvictsTheOldest() async throws {
        let store = try memoryStore(limit: 16)
        let first = metadata(
            tokens: [1, 2, 3], createdAt: Date(timeIntervalSince1970: 1))
        let second = metadata(
            tokens: [4, 5, 6], createdAt: Date(timeIntervalSince1970: 2))
        try await store.save(first, snapshot: snapshot(for: first))
        try await store.save(second, snapshot: snapshot(for: second))
        let hasSecond = await store.contains(second.key)
        let hasFirst = await store.contains(first.key)
        #expect(hasSecond)
        #expect(!hasFirst)
    }

    @Test func diskRoundTripsAcrossStores() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let configuration = ServerPromptCacheStorageConfiguration(
            memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 1 << 20)
        let meta = metadata(tokens: [1, 2, 3])
        do {
            let first = try PromptCheckpointStore(configuration: configuration)
            try await first.save(meta, snapshot: snapshot(for: meta))
        }
        let second = try PromptCheckpointStore(configuration: configuration)
        let loaded = try await second.load(meta.key, domain: domain)
        #expect(loaded.metadata == meta)
        #expect(loaded.snapshot.payload.count == meta.descriptor.payloadBytes)
        let listed = await second.metadata(for: domain)
        #expect(listed == [meta])
    }

    @Test func corruptDiskPayloadIsDeletedAndReported() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let configuration = ServerPromptCacheStorageConfiguration(
            memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 1 << 20)
        let meta = metadata(tokens: [1, 2, 3])
        do {
            let first = try PromptCheckpointStore(configuration: configuration)
            try await first.save(meta, snapshot: snapshot(for: meta))
        }
        let payload = root.appendingPathComponent(meta.key.contentHash)
            .appendingPathComponent("state.bin")
        var bytes = try Data(contentsOf: payload)
        bytes[0] ^= 0xFF
        try bytes.write(to: payload)
        let second = try PromptCheckpointStore(configuration: configuration)
        await #expect(throws: PromptCheckpointStoreError.self) {
            try await second.load(meta.key, domain: domain)
        }
        #expect(!FileManager.default.fileExists(atPath: payload.path))
    }

    @Test func diskLimitEvictsTheOldestDirectory() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let configuration = ServerPromptCacheStorageConfiguration(
            memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 20)
        let store = try PromptCheckpointStore(configuration: configuration)
        let first = metadata(
            tokens: [1, 2, 3], createdAt: Date(timeIntervalSince1970: 1))
        let second = metadata(
            tokens: [4, 5, 6], createdAt: Date(timeIntervalSince1970: 2))
        try await store.save(first, snapshot: snapshot(for: first))
        try await store.save(second, snapshot: snapshot(for: second))
        let hasSecond = await store.contains(second.key)
        let hasFirst = await store.contains(first.key)
        #expect(hasSecond)
        #expect(!hasFirst)
        let firstDirectory = root.appendingPathComponent(
            first.key.contentHash)
        #expect(!FileManager.default.fileExists(atPath: firstDirectory.path))
    }

    @Test func oversizedPayloadIsRefused() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let configuration = ServerPromptCacheStorageConfiguration(
            memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 8)
        let store = try PromptCheckpointStore(configuration: configuration)
        let meta = metadata(tokens: [1, 2, 3])
        await #expect(throws: PromptCheckpointStoreError.self) {
            try await store.save(meta, snapshot: self.snapshot(for: meta))
        }
    }
}
