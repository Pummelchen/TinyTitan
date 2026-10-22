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
    /// QSA pooled blocks plus the PLE window. At 8K this is tens of MiB; the
    /// cap keeps a pathological context from turning an auxiliary file into a
    /// second unbounded snapshot.
    static let maximumAuxiliaryBytes = 1_024 * 1_048_576

    private struct DiskMetadata: Codable {
        static let currentVersion = 1

        let version: Int
        let metadata: PromptCheckpointMetadata
        let payloadSHA256: String
        // Additive in v1: an old record has no auxiliary state, a new Qwen3.8
        // record carries the QSA/PLE payload and its hash.
        let auxiliaryBytes: Int?
        let auxiliarySHA256: String?
    }

    private struct DiskRecord {
        let metadata: PromptCheckpointMetadata
        let directory: URL
        let payload: URL
        let auxiliaryBytes: Int
        var modificationDate: Date
    }

    private struct MemoryRecord {
        let metadata: PromptCheckpointMetadata
        let snapshot: InferenceStateSnapshot
        let auxiliary: InferenceAuxiliaryState?
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
    private static let auxiliaryName = "auxiliary.bin"

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
        snapshot: InferenceStateSnapshot,
        auxiliary: InferenceAuxiliaryState? = nil
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
        let auxiliaryBytes = auxiliary?.payload.count ?? 0
        guard auxiliaryBytes <= Self.maximumAuxiliaryBytes else {
            throw PromptCheckpointStoreError.exceedsLimit(
                bytes: auxiliaryBytes, limit: Self.maximumAuxiliaryBytes)
        }
        let total = bytes + auxiliaryBytes
        if total <= configuration.memoryLimitBytes {
            insertMemory(
                metadata: metadata, snapshot: snapshot, auxiliary: auxiliary)
            return
        }
        guard configuration.diskDirectory != nil else {
            throw PromptCheckpointStoreError.exceedsLimit(
                bytes: total, limit: configuration.memoryLimitBytes)
        }
        guard total <= configuration.diskLimitBytes else {
            throw PromptCheckpointStoreError.exceedsLimit(
                bytes: total, limit: configuration.diskLimitBytes)
        }
        try writeDisk(
            metadata: metadata, snapshot: snapshot, auxiliary: auxiliary)
    }

    /// Loads a checkpoint and verifies it against `domain` and its content
    /// hash. A corrupt record is deleted and reported, never returned.
    func load(
        _ key: PromptCheckpointKey,
        domain: ServerPromptCacheDomain
    ) throws -> (
        metadata: PromptCheckpointMetadata,
        snapshot: InferenceStateSnapshot,
        auxiliary: InferenceAuxiliaryState?
    ) {
        if let record = state.memory[key] {
            guard record.metadata.domain == domain else {
                throw PromptCheckpointStoreError.domainMismatch
            }
            try Self.verifyKey(record.metadata)
            touchMemory(key)
            return (record.metadata, record.snapshot, record.auxiliary)
        }
        if state.disk[key] == nil,
            let restored = try? diskRecordFromFileSystem(for: key)
        {
            state.disk[key] = restored
            state.diskLRU.append(key)
            state.diskBytes +=
                restored.metadata.descriptor.payloadBytes + restored.auxiliaryBytes
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
            let auxiliary: InferenceAuxiliaryState?
            if let auxiliaryBytes = diskMetadata.auxiliaryBytes,
                let auxiliarySHA256 = diskMetadata.auxiliarySHA256
            {
                let auxiliaryData = try readPayload(
                    at: diskRecord.directory.appendingPathComponent(
                        Self.auxiliaryName),
                    expectedBytes: auxiliaryBytes)
                guard Sha256Verifier.hashData(auxiliaryData) == auxiliarySHA256
                else {
                    throw PromptCheckpointStoreError.corrupt(
                        key, "auxiliary payload hash does not match")
                }
                auxiliary = InferenceAuxiliaryState(
                    position: diskMetadata.metadata.position,
                    payload: auxiliaryData)
            } else {
                auxiliary = nil
            }
            touchDisk(key)
            let snapshot = InferenceStateSnapshot(
                descriptor: diskMetadata.metadata.descriptor, payload: payload)
            if payload.count + (auxiliary?.payload.count ?? 0)
                <= configuration.memoryLimitBytes
            {
                insertMemory(
                    metadata: diskMetadata.metadata, snapshot: snapshot,
                    auxiliary: auxiliary)
            }
            return (diskMetadata.metadata, snapshot, auxiliary)
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
        if let removed = state.memory.removeValue(forKey: key) {
            state.memoryBytes -=
                removed.snapshot.payload.count
                + (removed.auxiliary?.payload.count ?? 0)
        }
        state.memoryLRU.removeAll { $0 == key }
        if let record = state.disk.removeValue(forKey: key) {
            state.diskLRU.removeAll { $0 == key }
            state.diskBytes -=
                record.metadata.descriptor.payloadBytes + record.auxiliaryBytes
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
            let auxiliaryBytes = diskMetadata.auxiliaryBytes ?? 0
            state.disk[key] = DiskRecord(
                metadata: diskMetadata.metadata, directory: directory,
                payload: payloadURL, auxiliaryBytes: auxiliaryBytes,
                modificationDate: Date())
            state.diskLRU.append(key)
            state.diskBytes +=
                diskMetadata.metadata.descriptor.payloadBytes + auxiliaryBytes
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
            auxiliaryBytes: diskMetadata.auxiliaryBytes ?? 0,
            modificationDate: Date())
    }

    private func insertMemory(
        metadata: PromptCheckpointMetadata,
        snapshot: InferenceStateSnapshot,
        auxiliary: InferenceAuxiliaryState?
    ) {
        func bytes(_ record: MemoryRecord) -> Int {
            record.snapshot.payload.count + (record.auxiliary?.payload.count ?? 0)
        }
        if let existing = state.memory[metadata.key] {
            state.memoryBytes -= bytes(existing)
        }
        state.memory[metadata.key] = MemoryRecord(
            metadata: metadata, snapshot: snapshot, auxiliary: auxiliary)
        state.memoryLRU.removeAll { $0 == metadata.key }
        state.memoryLRU.append(metadata.key)
        state.memoryBytes += snapshot.payload.count + (auxiliary?.payload.count ?? 0)
        while state.memoryBytes > configuration.memoryLimitBytes,
            let oldest = state.memoryLRU.first
        {
            if let record = state.memory.removeValue(forKey: oldest) {
                state.memoryBytes -= bytes(record)
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
        snapshot: InferenceStateSnapshot,
        auxiliary: InferenceAuxiliaryState?
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
        let auxiliaryURL = directory.appendingPathComponent(Self.auxiliaryName)
        let auxiliaryBytes = auxiliary?.payload.count ?? 0
        let auxiliarySHA256: String?
        if let auxiliary {
            try auxiliary.payload.write(to: auxiliaryURL, options: .atomic)
            try fileManager.setAttributes(
                [.posixPermissions: 0o600], ofItemAtPath: auxiliaryURL.path)
            auxiliarySHA256 = Sha256Verifier.hashData(auxiliary.payload)
        } else {
            // A rewrite with no auxiliary state must not leave the previous
            // record's QSA/PLE payload behind a new metadata file.
            try? fileManager.removeItem(at: auxiliaryURL)
            auxiliarySHA256 = nil
        }
        let diskMetadata = DiskMetadata(
            version: DiskMetadata.currentVersion,
            metadata: metadata,
            payloadSHA256: Sha256Verifier.hashData(snapshot.payload),
            auxiliaryBytes: auxiliary == nil ? nil : auxiliaryBytes,
            auxiliarySHA256: auxiliarySHA256)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(diskMetadata).write(
            to: directory.appendingPathComponent(Self.metadataName),
            options: .atomic)
        // Replacing a record charges the payload once, not twice: the old
        // entry's bytes are still in `diskBytes` until they are subtracted.
        if let replaced = state.disk[metadata.key] {
            state.diskBytes -=
                replaced.metadata.descriptor.payloadBytes
                + replaced.auxiliaryBytes
            state.diskLRU.removeAll { $0 == metadata.key }
        }
        state.disk[metadata.key] = DiskRecord(
            metadata: metadata, directory: directory, payload: payloadURL,
            auxiliaryBytes: auxiliaryBytes,
            modificationDate: Date())
        state.diskLRU.append(metadata.key)
        state.diskBytes += snapshot.payload.count + auxiliaryBytes
        while state.diskBytes > configuration.diskLimitBytes,
            let oldest = state.diskLRU.first
        {
            guard let record = state.disk.removeValue(forKey: oldest) else {
                state.diskLRU.removeFirst()
                continue
            }
            state.diskLRU.removeFirst()
            state.diskBytes -=
                record.metadata.descriptor.payloadBytes + record.auxiliaryBytes
            try? fileManager.removeItem(at: record.directory)
        }
    }

    private func readPayload(
        at url: URL, expectedBytes: Int
    ) throws -> Data {
        guard expectedBytes > 0,
            expectedBytes <= max(Self.maximumCaptureBytes, Self.maximumAuxiliaryBytes)
        else {
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
