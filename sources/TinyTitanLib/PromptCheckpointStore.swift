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

    struct DiskMetadata: Codable {
        static let currentVersion = 1

        let version: Int
        let metadata: PromptCheckpointMetadata
        let payloadSHA256: String
        // Additive in v1: an old record has no auxiliary state, a new Qwen3.8
        // record carries the QSA/PLE payload and its hash.
        let auxiliaryBytes: Int?
        let auxiliarySHA256: String?

        func validate() throws {
            try metadata.validate()
            let payloadBytes = metadata.descriptor.payloadBytes
            guard payloadBytes > 0, payloadBytes <= PromptCheckpointStore.maximumCaptureBytes else {
                throw PromptCheckpointStoreError.invalidMetadata(
                    "payload size \(payloadBytes) is out of range")
            }
            let auxiliaryBytes = auxiliaryBytes ?? 0
            guard auxiliaryBytes >= 0,
                auxiliaryBytes <= PromptCheckpointStore.maximumAuxiliaryBytes
            else {
                throw PromptCheckpointStoreError.invalidMetadata(
                    "auxiliary size \(auxiliaryBytes) is out of range")
            }
        }
    }

    struct DiskRecord {
        let metadata: PromptCheckpointMetadata
        let directory: URL
        let payload: URL
        let auxiliaryBytes: Int
        var modificationDate: Date
    }

    struct MemoryRecord {
        let metadata: PromptCheckpointMetadata
        let snapshot: InferenceStateSnapshot
        let auxiliary: InferenceAuxiliaryState?
    }

    struct State {
        var memory: [PromptCheckpointKey: MemoryRecord] = [:]
        var memoryLRU: [PromptCheckpointKey] = []
        var memoryBytes = 0
        var disk: [PromptCheckpointKey: DiskRecord] = [:]
        var diskLRU: [PromptCheckpointKey] = []
        var diskBytes = 0
    }

    static let metadataName = "metadata.json"
    static let payloadName = "state.bin"
    static let auxiliaryName = "auxiliary.bin"

    let configuration: ServerPromptCacheStorageConfiguration
    let fileManager: FileManager
    var state = State()
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
        if state.disk[key] == nil {
            // A corrupt record is deleted and reported (see load's catch),
            // never surfaced as `missing`: `missing` means the space is free,
            // and an uncounted directory left behind is neither evictable
            // nor reclaimed.
            if let restored = try diskRecordFromFileSystem(for: key) {
                state.disk[key] = restored
                state.diskLRU.append(key)
                state.diskBytes +=
                    restored.metadata.descriptor.payloadBytes
                    + restored.auxiliaryBytes
            }
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
            try diskMetadata.validate()
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
        // Index first: without it, `state.disk` is empty and nothing is
        // unlinked, so the directories survive a removeAll on a fresh store.
        indexDiskIfNeeded()
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

    func indexDiskIfNeeded() {
        guard !diskIndexed else { return }
        guard let root = configuration.diskDirectory else {
            diskIndexed = true
            return
        }
        let keys: Set<URLResourceKey> = [
            .isDirectoryKey, .contentModificationDateKey,
        ]
        // A failed listing is not "nothing to index": leaving the flag clear
        // lets a later call retry, where latching it would strand every
        // record on disk for the store's lifetime.
        guard
            let entries = try? fileManager.contentsOfDirectory(
                at: root, includingPropertiesForKeys: Array(keys))
        else { return }
        diskIndexed = true
        // Restart LRU order is the directories' modification dates, not the
        // volume's listing order — the turn store applies the same rule, and
        // a listing order would evict an arbitrary record first.
        var loaded: [(record: DiskRecord, date: Date)] = []
        for directory in entries {
            guard
                let values = try? directory.resourceValues(forKeys: keys),
                values.isDirectory == true
            else { continue }
            let metadataURL = directory.appendingPathComponent(
                Self.metadataName)
            guard
                let data = try? BoundedMetadataRead.read(
                    fileAt: metadataURL, maxBytes: 16 * 1_048_576),
                let diskMetadata = try? JSONDecoder().decode(
                    DiskMetadata.self, from: data),
                diskMetadata.version == DiskMetadata.currentVersion,
                (try? diskMetadata.validate()) != nil
            else { continue }
            let key = diskMetadata.metadata.key
            guard state.disk[key] == nil else { continue }
            let payloadURL = directory.appendingPathComponent(Self.payloadName)
            let auxiliaryBytes = diskMetadata.auxiliaryBytes ?? 0
            loaded.append(
                (
                    DiskRecord(
                        metadata: diskMetadata.metadata, directory: directory,
                        payload: payloadURL, auxiliaryBytes: auxiliaryBytes,
                        modificationDate:
                            values.contentModificationDate ?? .distantPast),
                    values.contentModificationDate ?? .distantPast
                ))
        }
        loaded.sort { $0.date < $1.date }
        // A record `load` found before this index is already in `state.disk`;
        // appending the listing behind it would make the most recently loaded
        // record the first eviction candidate. Both sides carry a date -- the
        // filesystem's for indexed records, the write or touch time for the
        // in-process ones -- so one merged order is well defined.
        var order: [(key: PromptCheckpointKey, date: Date)] =
            state.diskLRU.compactMap { key in
                state.disk[key].map { (key, $0.modificationDate) }
            }
        order.append(
            contentsOf: loaded.map {
                (key: $0.record.metadata.key, date: $0.record.modificationDate)
            })
        order.sort { $0.date < $1.date }
        state.diskLRU = order.map(\.key)
        for entry in loaded {
            state.disk[entry.record.metadata.key] = entry.record
            state.diskBytes +=
                entry.record.metadata.descriptor.payloadBytes
                + entry.record.auxiliaryBytes
        }
        // A restart can index more than the budget allows (the limit may have
        // been lowered); evict immediately rather than waiting for a save
        // that may never come — the turn store does the same on load.
        evictDiskIfNeeded()
    }

    func evictDiskIfNeeded() {
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
        // A directory named for this key whose metadata cannot be read,
        // decoded or validated is corrupt, not absent: quarantine it (delete,
        // so its bytes are reclaimed) and say so. Returning nil here would
        // surface as `missing` and leave the directory uncounted forever.
        do {
            let data = try BoundedMetadataRead.read(
                fileAt: metadataURL, maxBytes: 16 * 1_048_576)
            let diskMetadata = try JSONDecoder().decode(
                DiskMetadata.self, from: data)
            guard diskMetadata.version == DiskMetadata.currentVersion,
                diskMetadata.metadata.key == key
            else {
                throw PromptCheckpointStoreError.corrupt(
                    key, "metadata does not match the record directory")
            }
            try diskMetadata.validate()
            return DiskRecord(
                metadata: diskMetadata.metadata, directory: directory,
                payload: directory.appendingPathComponent(Self.payloadName),
                auxiliaryBytes: diskMetadata.auxiliaryBytes ?? 0,
                modificationDate: Date())
        } catch {
            try? fileManager.removeItem(at: directory)
            ServerLog.diagnostic(
                "prompt checkpoints: quarantined corrupt record "
                    + "\(key.contentHash): \(error)")
            throw PromptCheckpointStoreError.corrupt(
                key, String(describing: error))
        }
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

    private static func verifyKey(_ metadata: PromptCheckpointMetadata) throws {
        let recomputed = PromptCheckpointKey.make(
            domain: metadata.domain, prefixTokenIDs: metadata.prefixTokenIDs)
        guard recomputed == metadata.key else {
            throw PromptCheckpointStoreError.keyMismatch
        }
    }
}
