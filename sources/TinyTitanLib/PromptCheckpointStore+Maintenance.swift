import Foundation
import TinyTitan

extension PromptCheckpointStore {
    /// Indexes the disk directories once, and only once the listing succeeds.
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
            do {
                try fileManager.removeItem(at: record.directory)
            } catch {
                // Keep the charge: the directory still occupies the disk, and
                // retrying it in this loop cannot succeed.
                state.disk[oldest] = record
                ServerLog.diagnostic(
                    "prompt checkpoints: could not evict "
                        + "\(oldest.contentHash): \(error)")
                return
            }
            state.diskLRU.removeFirst()
            state.diskBytes -=
                record.metadata.descriptor.payloadBytes + record.auxiliaryBytes
        }
    }

    func remove(_ key: PromptCheckpointKey) {
        if let removed = state.memory.removeValue(forKey: key) {
            state.memoryBytes -=
                removed.snapshot.payload.count
                + (removed.auxiliary?.payload.count ?? 0)
        }
        state.memoryLRU.removeAll { $0 == key }
        guard let record = state.disk[key] else { return }
        do {
            try fileManager.removeItem(at: record.directory)
        } catch {
            // The directory still occupies disk. Dropping the entry would
            // free bytes that are not free and leave nothing to retry.
            ServerLog.diagnostic(
                "prompt checkpoints: could not delete "
                    + "\(key.contentHash): \(error)")
            return
        }
        state.disk.removeValue(forKey: key)
        state.diskLRU.removeAll { $0 == key }
        state.diskBytes -=
            record.metadata.descriptor.payloadBytes + record.auxiliaryBytes
    }

    func removeAll() {
        // Index first: without it, `state.disk` is empty and nothing is
        // unlinked, so the directories survive a removeAll on a fresh store.
        indexDiskIfNeeded()
        var survivors = State()
        for record in state.disk.values {
            do {
                try fileManager.removeItem(at: record.directory)
            } catch {
                ServerLog.diagnostic(
                    "prompt checkpoints: could not delete "
                        + "\(record.metadata.key.contentHash) during removeAll: "
                        + "\(error)")
                survivors.disk[record.metadata.key] = record
                survivors.diskBytes +=
                    record.metadata.descriptor.payloadBytes
                    + record.auxiliaryBytes
                survivors.diskLRU.append(record.metadata.key)
            }
        }
        survivors.diskLRU.sort {
            (survivors.disk[$0]?.modificationDate ?? .distantPast)
                < (survivors.disk[$1]?.modificationDate ?? .distantPast)
        }
        // Memory is process-local and always reclaimable; only the disk
        // records that refused removal survive.
        state = survivors
    }
}
