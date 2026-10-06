import Foundation
import Testing

@testable import TinyTitan
@testable import TinyTitanLib
@testable import TinyTitanServerCore

@Suite("Server prompt state store")
struct ServerPromptStateStoreTests {
    private let domain = ServerPromptCacheDomain(
        modelID: "model",
        sourceSnapshotHash: "snapshot",
        runtimeProfileHash: "profile",
        maximumContext: 16_384,
        kvStorage: "fp16",
        fp16RingEnabled: true,
        templateSHA256: "template")

    @Test func diskSnapshotSurvivesStoreRecreation() async throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let entry = makeEntry(tokens: [1, 2])
        let snapshot = makeSnapshot(position: 2, payload: Data([1, 2, 3, 4, 5, 6, 7]))
        let configuration = ServerPromptCacheStorageConfiguration(
            memoryLimitBytes: 0,
            diskDirectory: root,
            diskLimitBytes: 1_024)

        let writer = try ServerPromptStateStore(configuration: configuration)
        let saved = await writer.save(entry: entry, snapshot: snapshot)
        #expect(saved.diskError == nil)
        #expect(saved.diskBytes == snapshot.payload.count)
        let entryDirectory = root.appendingPathComponent(
            entry.id.uuidString.lowercased())
        #expect(permissions(of: root) & 0o077 == 0)
        #expect(permissions(of: entryDirectory) & 0o077 == 0)
        #expect(
            permissions(of: entryDirectory.appendingPathComponent("metadata.json"))
                & 0o077 == 0)
        #expect(
            permissions(of: entryDirectory.appendingPathComponent("state.bin"))
                & 0o077 == 0)

        let reader = try ServerPromptStateStore(configuration: configuration)
        #expect(reader.loadEntries(domain: domain) == [entry])
        #expect(reader.loadEntries(domain: domain) == [entry])
        let restored = try reader.loadSnapshot(entryID: entry.id)
        #expect(restored.tier == "ssd")
        #expect(restored.snapshot == snapshot)
    }

    @Test func memoryLRUEvictsOldestUnbackedSnapshot() async throws {
        let store = try ServerPromptStateStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: 10,
                diskDirectory: nil,
                diskLimitBytes: 0))
        let first = makeEntry(tokens: [1])
        let second = makeEntry(tokens: [2])
        let firstSnapshot = makeSnapshot(position: 1, payload: Data(repeating: 1, count: 6))
        let secondSnapshot = makeSnapshot(position: 1, payload: Data(repeating: 2, count: 6))

        _ = await store.save(entry: first, snapshot: firstSnapshot)
        let saved = await store.save(entry: second, snapshot: secondSnapshot)

        #expect(saved.unbackedEntryIDs.contains(first.id))
        #expect(!store.contains(first.id))
        #expect(store.contains(second.id))
        #expect(try store.loadSnapshot(entryID: second.id).snapshot == secondSnapshot)
    }

    @Test func diskLRUEvictsOldestUnbackedSnapshot() async throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try ServerPromptStateStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: 0,
                diskDirectory: root,
                diskLimitBytes: 10))
        let first = makeEntry(tokens: [1])
        let second = makeEntry(tokens: [2])

        _ = await store.save(
            entry: first,
            snapshot: makeSnapshot(
                position: 1,
                payload: Data(repeating: 1, count: 6)))
        let saved = await store.save(
            entry: second,
            snapshot: makeSnapshot(
                position: 1,
                payload: Data(repeating: 2, count: 6)))

        #expect(saved.unbackedEntryIDs.contains(first.id))
        #expect(!store.contains(first.id))
        #expect(store.contains(second.id))
        #expect(saved.diskBytes == 6)
    }

    @Test func corruptDiskPayloadFailsClosedAndIsRemoved() async throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let entry = makeEntry(tokens: [1, 2])
        let snapshot = makeSnapshot(position: 2, payload: Data([1, 2, 3, 4, 5, 6, 7]))
        let configuration = ServerPromptCacheStorageConfiguration(
            memoryLimitBytes: 0,
            diskDirectory: root,
            diskLimitBytes: 1_024)
        let writer = try ServerPromptStateStore(configuration: configuration)
        _ = await writer.save(entry: entry, snapshot: snapshot)
        let payload =
            root
            .appendingPathComponent(entry.id.uuidString.lowercased())
            .appendingPathComponent("state.bin")
        try Data([9, 2, 3, 4, 5, 6, 7]).write(to: payload)

        let reader = try ServerPromptStateStore(configuration: configuration)
        #expect(reader.loadEntries(domain: domain) == [entry])
        #expect(throws: ServerPromptStateStoreError.self) {
            try reader.loadSnapshot(entryID: entry.id)
        }
        #expect(
            !FileManager.default.fileExists(
                atPath: payload.deletingLastPathComponent().path))
    }

    /// AUD-142: a record whose `metadata.json` crosses the store's bound is
    /// skipped, and skipped *without being read*. The shape this replaced stat'd
    /// the path, checked the size, then re-read the path — so the bytes checked
    /// were not the bytes used (K17), and the whole document was allocated before
    /// the refusal. The fixture is a sparse tail for that reason: it is 16 MiB and
    /// one byte of *size* with almost nothing behind it, which is exactly what a
    /// planted cache entry looks like and what an unbounded read would put in
    /// memory.
    @Test func oversizedDiskMetadataIsSkippedWithoutBeingRead() async throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let entry = makeEntry(tokens: [1, 2])
        let snapshot = makeSnapshot(position: 2, payload: Data([1, 2, 3, 4, 5, 6, 7]))
        let configuration = ServerPromptCacheStorageConfiguration(
            memoryLimitBytes: 0,
            diskDirectory: root,
            diskLimitBytes: 1_024)
        let writer = try ServerPromptStateStore(configuration: configuration)
        _ = await writer.save(entry: entry, snapshot: snapshot)

        let metadata =
            root
            .appendingPathComponent(entry.id.uuidString.lowercased())
            .appendingPathComponent("metadata.json")
        let handle = try FileHandle(forWritingTo: metadata)
        try handle.truncate(atOffset: 16 * 1_024 * 1_024 + 1)
        try handle.close()

        let reader = try ServerPromptStateStore(configuration: configuration)
        #expect(reader.loadEntries(domain: domain).isEmpty)
    }

    /// A link standing in for a record's metadata is not a record. The check that
    /// covers this used to be a `resourceValues` look at the path before the read;
    /// it now lives in the read, on the descriptor that is actually used.
    @Test func symlinkedDiskMetadataIsSkipped() async throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let entry = makeEntry(tokens: [1, 2])
        let snapshot = makeSnapshot(position: 2, payload: Data([1, 2, 3, 4, 5, 6, 7]))
        let configuration = ServerPromptCacheStorageConfiguration(
            memoryLimitBytes: 0,
            diskDirectory: root,
            diskLimitBytes: 1_024)
        let writer = try ServerPromptStateStore(configuration: configuration)
        _ = await writer.save(entry: entry, snapshot: snapshot)

        let record = root.appendingPathComponent(entry.id.uuidString.lowercased())
        let metadata = record.appendingPathComponent("metadata.json")
        let real = root.appendingPathComponent("elsewhere.json")
        try FileManager.default.moveItem(at: metadata, to: real)
        try FileManager.default.createSymbolicLink(
            atPath: metadata.path, withDestinationPath: real.path)

        let reader = try ServerPromptStateStore(configuration: configuration)
        #expect(reader.loadEntries(domain: domain).isEmpty)
    }

    private func makeEntry(tokens: [Int32]) -> ServerPromptCacheEntry {
        ServerPromptCacheEntry(
            id: UUID(),
            domain: domain,
            inputMessages: [GFTokenizer.Message(role: .user, content: "prompt")],
            tools: [],
            assistantTurn: CachedAssistantTurn(
                message: GFTokenizer.Message(role: .assistant, content: "answer"),
                rawStopReason: .endOfTurn),
            kvBackedTokenIDs: tokens,
            uncommittedBoundaryTokenIDs: [99],
            kvPosition: tokens.count)
    }

    private func makeSnapshot(position: Int, payload: Data) -> InferenceStateSnapshot {
        InferenceStateSnapshot(
            descriptor: InferenceStateSnapshotDescriptor(
                position: position,
                kvSegmentLengths: [payload.count - 3],
                gdnSegmentLengths: [3],
                payloadBytes: payload.count),
            payload: payload)
    }

    private func temporaryDirectory() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("tinytitan-prompt-cache-tests-\(UUID().uuidString)")
    }

    private func permissions(of url: URL) -> Int {
        let attributes = try? FileManager.default.attributesOfItem(atPath: url.path)
        return attributes?[.posixPermissions] as? Int ?? 0o777
    }
}
