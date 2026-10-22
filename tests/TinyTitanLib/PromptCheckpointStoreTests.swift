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

    @Test func aDiskOnlyRecordIsFoundWithoutLoadingIt() async throws {
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
        // `contains` must see the record the first process wrote, or the
        // capture path rewrites the same multi-gigabyte payload every turn.
        #expect(await second.contains(meta.key))
        let absent = PromptCheckpointKey.make(
            domain: domain, prefixTokenIDs: [7])
        #expect(!(await second.contains(absent)))
    }

    @Test func rewritingTheSameDiskRecordChargesItsBytesOnce() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let configuration = ServerPromptCacheStorageConfiguration(
            memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 1 << 20)
        let store = try PromptCheckpointStore(configuration: configuration)
        let meta = metadata(tokens: [1, 2, 3])
        try await store.save(meta, snapshot: snapshot(for: meta))
        let afterFirst = await store.diskBytes
        try await store.save(meta, snapshot: snapshot(for: meta, byte: 8))
        let afterSecond = await store.diskBytes
        #expect(afterFirst == meta.descriptor.payloadBytes)
        #expect(afterSecond == meta.descriptor.payloadBytes)
    }

    @Test func metadataListsOneConversationOldestFirst() async throws {
        let store = try memoryStore()
        let first = metadata(
            tokens: [1, 2], conversation: "a",
            createdAt: Date(timeIntervalSince1970: 1))
        let second = metadata(
            tokens: [3, 4], conversation: "a",
            createdAt: Date(timeIntervalSince1970: 2))
        let other = metadata(
            tokens: [5, 6], conversation: "b",
            createdAt: Date(timeIntervalSince1970: 3))
        try await store.save(first, snapshot: snapshot(for: first))
        try await store.save(second, snapshot: snapshot(for: second))
        try await store.save(other, snapshot: snapshot(for: other))
        let listed = await store.metadata(for: domain)
        #expect(listed.map(\.conversationKey) == ["a", "a", "b"])
        #expect(listed.map(\.position) == [2, 2, 2])
    }

    @Test func concurrentSavesAndLoadsSeeWholeRecords() async throws {
        let store = try memoryStore(limit: 1 << 22)
        let domain = self.domain
        try await withThrowingTaskGroup(of: Void.self) { group in
            for index in 0..<16 {
                group.addTask {
                    let tokens = [Int32(index), 40, 41]
                    let key = PromptCheckpointKey.make(
                        domain: domain, prefixTokenIDs: tokens)
                    let descriptor = InferenceStateSnapshotDescriptor(
                        position: tokens.count,
                        kvSegmentLengths: [16],
                        gdnSegmentLengths: [],
                        payloadBytes: 16)
                    let metadata = PromptCheckpointMetadata(
                        key: key,
                        domain: domain,
                        kind: .systemEnd,
                        conversationKey: "conversation",
                        prefixTokenIDs: tokens,
                        descriptor: descriptor)
                    let snapshot = InferenceStateSnapshot(
                        descriptor: descriptor,
                        payload: Data(repeating: UInt8(index), count: 16))
                    try await store.save(metadata, snapshot: snapshot)
                    let loaded = try await store.load(key, domain: domain)
                    #expect(loaded.metadata == metadata)
                    #expect(
                        loaded.snapshot.payload
                            == Data(repeating: UInt8(index), count: 16))
                }
            }
            try await group.waitForAll()
        }
        #expect(await store.memoryBytes == 16 * 16)
    }

    @Test func aLoadedDiskRecordIsListedOnceAndOldestFirst() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let configuration = ServerPromptCacheStorageConfiguration(
            memoryLimitBytes: 1 << 20, diskDirectory: root,
            diskLimitBytes: 1 << 20)
        let first = metadata(
            tokens: [1, 2], createdAt: Date(timeIntervalSince1970: 1))
        let second = metadata(
            tokens: [3, 4], createdAt: Date(timeIntervalSince1970: 2))
        do {
            // Force both onto disk by saving through a memory-less store, so
            // the reading store has to index them from the filesystem.
            let writer = try PromptCheckpointStore(
                configuration: ServerPromptCacheStorageConfiguration(
                    memoryLimitBytes: 0, diskDirectory: root,
                    diskLimitBytes: 1 << 20))
            try await writer.save(first, snapshot: snapshot(for: first))
            try await writer.save(second, snapshot: snapshot(for: second))
        }
        let reader = try PromptCheckpointStore(configuration: configuration)
        _ = try await reader.load(second.key, domain: domain)
        let listed = await reader.metadata(for: domain)
        #expect(listed.count == 2)
        #expect(listed.map(\.key) == [first.key, second.key])
    }

    @Test func auxiliaryQSAStateTravelsWithTheCheckpoint() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let configuration = ServerPromptCacheStorageConfiguration(
            memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 1 << 20)
        let meta = metadata(tokens: [1, 2, 3])
        let auxiliary = InferenceAuxiliaryState(
            position: 3, payload: Data(repeating: 9, count: 64))
        do {
            let writer = try PromptCheckpointStore(configuration: configuration)
            try await writer.save(
                meta, snapshot: snapshot(for: meta), auxiliary: auxiliary)
        }
        let reader = try PromptCheckpointStore(configuration: configuration)
        let loaded = try await reader.load(meta.key, domain: domain)
        #expect(loaded.auxiliary == auxiliary)
        #expect(loaded.snapshot.payload == snapshot(for: meta).payload)
    }

    @Test func corruptAuxiliaryStateIsDeletedRatherThanRestored() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let configuration = ServerPromptCacheStorageConfiguration(
            memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 1 << 20)
        let meta = metadata(tokens: [1, 2, 3])
        let auxiliary = InferenceAuxiliaryState(
            position: 3, payload: Data(repeating: 4, count: 32))
        do {
            let writer = try PromptCheckpointStore(configuration: configuration)
            try await writer.save(
                meta, snapshot: snapshot(for: meta), auxiliary: auxiliary)
        }
        let path = root.appendingPathComponent(meta.key.contentHash)
            .appendingPathComponent("auxiliary.bin")
        var bytes = try Data(contentsOf: path)
        bytes[0] ^= 0xFF
        try bytes.write(to: path)
        let reader = try PromptCheckpointStore(configuration: configuration)
        await #expect(throws: PromptCheckpointStoreError.self) {
            try await reader.load(meta.key, domain: domain)
        }
        #expect(!FileManager.default.fileExists(atPath: path.path))
    }

    @Test func rewritingWithoutAuxiliaryDropsTheOldPayload() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let configuration = ServerPromptCacheStorageConfiguration(
            memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 1 << 20)
        let store = try PromptCheckpointStore(configuration: configuration)
        let meta = metadata(tokens: [1, 2, 3])
        try await store.save(
            meta, snapshot: snapshot(for: meta),
            auxiliary: InferenceAuxiliaryState(
                position: 3, payload: Data(repeating: 1, count: 32)))
        let firstBytes = await store.diskBytes
        try await store.save(meta, snapshot: snapshot(for: meta))
        let secondBytes = await store.diskBytes
        #expect(firstBytes == meta.descriptor.payloadBytes + 32)
        #expect(secondBytes == meta.descriptor.payloadBytes)
        let loaded = try await store.load(meta.key, domain: domain)
        #expect(loaded.auxiliary == nil)
    }

    @Test func auxiliaryStateCountsAgainstTheMemoryLimit() async throws {
        let store = try PromptCheckpointStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: 20, diskDirectory: nil, diskLimitBytes: 0))
        let meta = metadata(tokens: [1, 2, 3])
        await #expect(throws: PromptCheckpointStoreError.self) {
            try await store.save(
                meta, snapshot: self.snapshot(for: meta),
                auxiliary: InferenceAuxiliaryState(
                    position: 3, payload: Data(repeating: 1, count: 8)))
        }
    }
}
