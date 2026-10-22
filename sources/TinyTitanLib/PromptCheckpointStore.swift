import CryptoKit
import Foundation
import TinyTitan

/// Content-addressed store for interior checkpoints.
///
/// The turn-end prompt cache keeps one snapshot per completed assistant turn.
/// This store holds the points *inside* a prompt (system end, last-turn start)
/// that survive an edit to the last turn. Payloads are the same
/// `InferenceStateSnapshot` the turn cache already captures; the metadata adds
/// the exact prefix token IDs and a content hash, so a restore can never use
/// a snapshot for a different prefix.
///
/// The store is an actor: capture, restore and eviction are serialized, and
/// disk writes happen inside it so two saves cannot interleave. The same
/// memory/disk accounting as `ServerPromptStateStore` applies: a payload goes
/// to memory when it fits the memory budget, otherwise to disk, otherwise it
/// is refused rather than silently dropped.
actor PromptCheckpointStore {
    /// A single captured snapshot is capped, like the turn store, so a huge
    /// context cannot allocate an unbounded payload regardless of the budget.
    static let maximumCaptureBytes = 4 * 1_024 * 1_048_576

    private struct DiskMetadata: Codable {
        static let currentVersion = 1

        let version: Int
        let metadata: PromptCheckpointMetadata
        let payloadSHA256: String
    }

    private struct DiskRecord {
        let metadata: PromptCheckpointMetadata
        let directory: URL
        let payload: URL
        var modificationDate: Date
    }

    private struct MemoryRecord {
        let metadata: PromptCheckpointMetadata
        let snapshot: InferenceStateSnapshot
    }

    private struct State {
        var memory: [PromptCheckpointKey: MemoryRecord] = [:]
        var memoryLRU: [PromptCheckpointKey] = []
        var memoryBytes = 0
        var disk: [PromptCheckpointKey: DiskRecord] = [:]
        var diskLRU: [PromptCheckpointKey] = []
        var diskBytes = 0
    }

    private static let metadataName = "metadata.json"
    private static let payloadName = "state.bin"

    private let configuration: ServerPromptCacheStorageConfiguration
    private let fileManager: FileManager
    private var state = State()
    private var diskIndexed = false

    init(
        configuration: ServerPromptCacheStorageConfiguration,
        fileManager: FileManager = .default
    ) throws {
        self.configuration = configuration
        self.fileManager = fileManager
        if let root = configuration.diskDirectory {
            try fileManager.createDirectory(
                at: root, withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700])
            try fileManager.setAttributes(
                [.posixPermissions: 0o700], ofItemAtPath: root.path)
        }
    }

    /// Saves one checkpoint. The metadata is validated against the payload
    /// before anything is written.
    func save(
        _ metadata: PromptCheckpointMetadata,
        snapshot: InferenceStateSnapshot
    ) throws {
        try metadata.validate()
        guard snapshot.descriptor == metadata.descriptor else {
            throw PromptCheckpointStoreError.invalidMetadata(
                "snapshot descriptor does not match the metadata")
        }
        guard snapshot.payload.count == metadata.descriptor.payloadBytes else {
            throw PromptCheckpointStoreError.invalidMetadata(
                "payload is \(snapshot.payload.count) bytes, descriptor says "
                    + "\(metadata.descriptor.payloadBytes)")
        }
        let bytes = snapshot.payload.count
        guard bytes <= Self.maximumCaptureBytes else {
            throw PromptCheckpointStoreError.exceedsLimit(
                bytes: bytes, limit: Self.maximumCaptureBytes)
        }
        if bytes <= configuration.memoryLimitBytes {
            insertMemory(metadata: metadata, snapshot: snapshot)
            return
        }
        guard configuration.diskDirectory != nil else {
            throw PromptCheckpointStoreError.exceedsLimit(
                bytes: bytes, limit: configuration.memoryLimitBytes)
        }
        guard bytes <= configuration.diskLimitBytes else {
            throw PromptCheckpointStoreError.exceedsLimit(
                bytes: bytes, limit: configuration.diskLimitBytes)
        }
        try writeDisk(metadata: metadata, snapshot: snapshot)
    }

    /// Loads a checkpoint and verifies it against `domain` and its content
    /// hash. A corrupt record is deleted and reported, never returned.
    func load(
        _ key: PromptCheckpointKey,
        domain: ServerPromptCacheDomain
    ) throws -> (metadata: PromptCheckpointMetadata, snapshot: InferenceStateSnapshot) {
        if let record = state.memory[key] {
            guard record.metadata.domain == domain else {
                throw PromptCheckpointStoreError.domainMismatch
            }
            try Self.verifyKey(record.metadata)
            touchMemory(key)
            return (record.metadata, record.snapshot)
        }
        if state.disk[key] == nil,
            let restored = try? diskRecordFromFileSystem(for: key)
        {
            state.disk[key] = restored
            state.diskLRU.append(key)
            state.diskBytes += restored.metadata.descriptor.payloadBytes
        }
        guard let diskRecord = state.disk[key] else {
            throw PromptCheckpointStoreError.missing(key)
        }
        guard diskRecord.metadata.domain == domain else {
            throw PromptCheckpointStoreError.domainMismatch
        }
        do {
            let metadataData = try BoundedMetadataRead.read(
                fileAt: diskRecord.directory
                    .appendingPathComponent(Self.metadataName),
                maxBytes: 16 * 1_048_576)
            let diskMetadata = try JSONDecoder().decode(
                DiskMetadata.self, from: metadataData)
            guard diskMetadata.version == DiskMetadata.currentVersion else {
                throw PromptCheckpointStoreError.corrupt(
                    key, "unsupported metadata version \(diskMetadata.version)")
            }
            guard diskMetadata.metadata.key == key else {
                throw PromptCheckpointStoreError.keyMismatch
            }
            try diskMetadata.metadata.validate()
            let payload = try readPayload(
                at: diskRecord.payload,
                expectedBytes: diskMetadata.metadata.descriptor.payloadBytes)
            guard Sha256Verifier.hashData(payload) == diskMetadata.payloadSHA256
            else {
                throw PromptCheckpointStoreError.corrupt(
                    key, "payload hash does not match")
            }
            touchDisk(key)
            let snapshot = InferenceStateSnapshot(
                descriptor: diskMetadata.metadata.descriptor, payload: payload)
            if payload.count <= configuration.memoryLimitBytes {
                insertMemory(metadata: diskMetadata.metadata, snapshot: snapshot)
            }
            return (diskMetadata.metadata, snapshot)
        } catch let error as PromptCheckpointStoreError {
            remove(key)
            throw error
        } catch {
            remove(key)
            throw PromptCheckpointStoreError.corrupt(
                key, String(describing: error))
        }
    }

    func contains(_ key: PromptCheckpointKey) -> Bool {
        indexDiskIfNeeded()
        if state.memory[key] != nil || state.disk[key] != nil { return true }
        // A checkpoint written by an earlier process is on disk but not in
        // either in-memory index until someone loads it. The capture path asks
        // first so a repeated request does not rewrite a multi-gigabyte
        // payload that is already there.
        guard configuration.diskDirectory != nil else { return false }
        return (try? diskRecordFromFileSystem(for: key)) != nil
    }

    func remove(_ key: PromptCheckpointKey) {
        state.memory.removeValue(forKey: key)
        state.memoryLRU.removeAll { $0 == key }
        if let record = state.disk.removeValue(forKey: key) {
            state.diskLRU.removeAll { $0 == key }
            state.diskBytes -= record.metadata.descriptor.payloadBytes
            try? fileManager.removeItem(at: record.directory)
        }
    }

    func removeAll() {
        for record in state.disk.values {
            try? fileManager.removeItem(at: record.directory)
        }
        state = State()
    }

    /// Every checkpoint recorded for a domain, oldest first.
    ///
    /// A record loaded from disk also lands in memory (when it fits), so the
    /// two indexes can name the same key; the key is the identity and the
    /// memory copy is the newer one.
    func metadata(for domain: ServerPromptCacheDomain) -> [PromptCheckpointMetadata] {
        indexDiskIfNeeded()
        var byKey: [PromptCheckpointKey: PromptCheckpointMetadata] = [:]
        for record in state.disk.values {
            byKey[record.metadata.key] = record.metadata
        }
        for record in state.memory.values {
            byKey[record.metadata.key] = record.metadata
        }
        return byKey.values.filter { $0.domain == domain }
            .sorted { $0.createdAt < $1.createdAt }
    }

    var memoryBytes: Int { state.memoryBytes }
    var diskBytes: Int { state.diskBytes }

    private func indexDiskIfNeeded() {
        guard !diskIndexed else { return }
        diskIndexed = true
        guard let root = configuration.diskDirectory else { return }
        let entries =
            (try? fileManager.contentsOfDirectory(
                at: root, includingPropertiesForKeys: [.isDirectoryKey]))
            ?? []
        for directory in entries {
            guard
                (try? directory.resourceValues(forKeys: [.isDirectoryKey]))?
                    .isDirectory == true
            else { continue }
            let metadataURL = directory.appendingPathComponent(
                Self.metadataName)
            guard
                let data = try? BoundedMetadataRead.read(
                    fileAt: metadataURL, maxBytes: 16 * 1_048_576),
                let diskMetadata = try? JSONDecoder().decode(
                    DiskMetadata.self, from: data),
                diskMetadata.version == DiskMetadata.currentVersion,
                (try? diskMetadata.metadata.validate()) != nil
            else { continue }
            let key = diskMetadata.metadata.key
            guard state.disk[key] == nil else { continue }
            let payloadURL = directory.appendingPathComponent(Self.payloadName)
            state.disk[key] = DiskRecord(
                metadata: diskMetadata.metadata, directory: directory,
                payload: payloadURL, modificationDate: Date())
            state.diskLRU.append(key)
            state.diskBytes += diskMetadata.metadata.descriptor.payloadBytes
        }
    }

    private func diskRecordFromFileSystem(for key: PromptCheckpointKey)
        throws -> DiskRecord?
    {
        guard let root = configuration.diskDirectory else { return nil }
        let directory = root.appendingPathComponent(
            key.contentHash, isDirectory: true)
        let metadataURL = directory.appendingPathComponent(Self.metadataName)
        guard fileManager.fileExists(atPath: metadataURL.path) else {
            return nil
        }
        let data = try BoundedMetadataRead.read(
            fileAt: metadataURL, maxBytes: 16 * 1_048_576)
        let diskMetadata = try JSONDecoder().decode(
            DiskMetadata.self, from: data)
        guard diskMetadata.version == DiskMetadata.currentVersion,
            diskMetadata.metadata.key == key
        else { return nil }
        try diskMetadata.metadata.validate()
        return DiskRecord(
            metadata: diskMetadata.metadata, directory: directory,
            payload: directory.appendingPathComponent(Self.payloadName),
            modificationDate: Date())
    }

    private func insertMemory(
        metadata: PromptCheckpointMetadata,
        snapshot: InferenceStateSnapshot
    ) {
        if let existing = state.memory[metadata.key] {
            state.memoryBytes -= existing.snapshot.payload.count
        }
        state.memory[metadata.key] = MemoryRecord(
            metadata: metadata, snapshot: snapshot)
        state.memoryLRU.removeAll { $0 == metadata.key }
        state.memoryLRU.append(metadata.key)
        state.memoryBytes += snapshot.payload.count
        while state.memoryBytes > configuration.memoryLimitBytes,
            let oldest = state.memoryLRU.first
        {
            if let record = state.memory.removeValue(forKey: oldest) {
                state.memoryBytes -= record.snapshot.payload.count
            }
            state.memoryLRU.removeFirst()
        }
    }

    private func touchMemory(_ key: PromptCheckpointKey) {
        state.memoryLRU.removeAll { $0 == key }
        state.memoryLRU.append(key)
    }

    private func touchDisk(_ key: PromptCheckpointKey) {
        state.diskLRU.removeAll { $0 == key }
        state.diskLRU.append(key)
        state.disk[key]?.modificationDate = Date()
    }

    private func writeDisk(
        metadata: PromptCheckpointMetadata,
        snapshot: InferenceStateSnapshot
    ) throws {
        guard let root = configuration.diskDirectory else {
            throw PromptCheckpointStoreError.diskUnavailable
        }
        let directory = root.appendingPathComponent(
            metadata.key.contentHash, isDirectory: true)
        try fileManager.createDirectory(
            at: directory, withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700])
        let payloadURL = directory.appendingPathComponent(Self.payloadName)
        try snapshot.payload.write(to: payloadURL, options: .atomic)
        try fileManager.setAttributes(
            [.posixPermissions: 0o600], ofItemAtPath: payloadURL.path)
        let diskMetadata = DiskMetadata(
            version: DiskMetadata.currentVersion,
            metadata: metadata,
            payloadSHA256: Sha256Verifier.hashData(snapshot.payload))
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(diskMetadata).write(
            to: directory.appendingPathComponent(Self.metadataName),
            options: .atomic)
        // Replacing a record charges the payload once, not twice: the old
        // entry's bytes are still in `diskBytes` until they are subtracted.
        if let replaced = state.disk[metadata.key] {
            state.diskBytes -= replaced.metadata.descriptor.payloadBytes
            state.diskLRU.removeAll { $0 == metadata.key }
        }
        state.disk[metadata.key] = DiskRecord(
            metadata: metadata, directory: directory, payload: payloadURL,
            modificationDate: Date())
        state.diskLRU.append(metadata.key)
        state.diskBytes += snapshot.payload.count
        while state.diskBytes > configuration.diskLimitBytes,
            let oldest = state.diskLRU.first
        {
            guard let record = state.disk.removeValue(forKey: oldest) else {
                state.diskLRU.removeFirst()
                continue
            }
            state.diskLRU.removeFirst()
            state.diskBytes -= record.metadata.descriptor.payloadBytes
            try? fileManager.removeItem(at: record.directory)
        }
    }

    private func readPayload(
        at url: URL, expectedBytes: Int
    ) throws -> Data {
        guard expectedBytes > 0, expectedBytes <= Self.maximumCaptureBytes else {
            throw PromptCheckpointStoreError.invalidMetadata(
                "payload size \(expectedBytes) is not a valid capture size")
        }
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let data = try handle.read(upToCount: expectedBytes + 1) ?? Data()
        guard data.count == expectedBytes else {
            throw PromptCheckpointStoreError.corrupt(
                PromptCheckpointKey(contentHash: url.deletingLastPathComponent().lastPathComponent),
                "payload is \(data.count) bytes, expected \(expectedBytes)")
        }
        return data
    }

    private static func verifyKey(_ metadata: PromptCheckpointMetadata) throws {
        let recomputed = PromptCheckpointKey.make(
            domain: metadata.domain, prefixTokenIDs: metadata.prefixTokenIDs)
        guard recomputed == metadata.key else {
            throw PromptCheckpointStoreError.keyMismatch
        }
    }
}
