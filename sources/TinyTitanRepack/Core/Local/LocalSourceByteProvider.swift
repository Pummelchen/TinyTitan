import Darwin
import Foundation

/// Copies planned safetensors ranges from a completed local snapshot. This is
/// used for derived sidecars whose source tensor conversion must happen
/// locally, while preserving the same bounded-copy and destination-digest
/// contracts as the remote installer.
final class LocalSourceByteProvider: SourceByteProvider {
    private let snapshotDirectory: String
    private let writeTileBytes: Int
    private let retireSourceShards: Bool

    init(
        snapshotDirectory: String,
        writeTileBytes: Int = WriterCore.tileBytes,
        retireSourceShards: Bool = false
    ) {
        self.snapshotDirectory = snapshotDirectory
        self.writeTileBytes = writeTileBytes
        self.retireSourceShards = retireSourceShards
    }

    func copyBatch(
        _ copies: [CoalescedRangeCopy],
        completedRangeIDs: Set<String>,
        partialDirectory: String,
        temporaryPath: String,
        audit: RepackAudit,
        progress: @escaping @Sendable (UInt64) -> Void,
        commit: (RemoteCompletedRange) throws -> Void
    ) async throws {
        let scratch = UnsafeMutableRawBufferPointer.allocate(
            byteCount: writeTileBytes,
            alignment: 16_384)
        defer { scratch.deallocate() }
        audit.largestScratchBytes = max(audit.largestScratchBytes, scratch.count)

        var outputDescriptors: [String: Int32] = [:]
        defer {
            for descriptor in outputDescriptors.values { close(descriptor) }
        }
        var copiedBytes: UInt64 = 0
        var retirement = SourceRetirement(
            copies: copies, enabled: retireSourceShards)

        for copy in copies {
            // A range completed by an earlier attempt still counts against its
            // shard's remaining bytes: the shard is retired only when every
            // range that reads it has been copied, whether in this run or a
            // previous one.
            if completedRangeIDs.contains(copy.id) {
                retirement.consume(copy) { try? resolvedSourcePath(copy.shardID) }
                continue
            }
            try Task.checkCancellation()
            let sourcePath = try resolvedSourcePath(copy.shardID)
            let sourceDescriptor = try Posix.openReadNoFollow(sourcePath)
            var touched = Set<String>()
            do {
                let sourceSize = try Posix.fileSize(fd: sourceDescriptor, path: sourcePath)
                guard copy.sourceOffset <= sourceSize,
                    copy.size <= sourceSize - copy.sourceOffset
                else {
                    throw RepackError.safetensorsTensorOutOfRange(
                        path: sourcePath,
                        name: copy.id,
                        end: copy.sourceOffset + copy.size,
                        fileSize: sourceSize)
                }
                for destination in copy.destinations {
                    let descriptor = try outputDescriptor(
                        path: destination.destinationPath,
                        cache: &outputDescriptors)
                    touched.insert(destination.destinationPath)
                    try copyBytes(
                        sourceDescriptor: sourceDescriptor,
                        sourcePath: sourcePath,
                        destinationDescriptor: descriptor,
                        destinationPath: destination.destinationPath,
                        sourceOffset: destination.sourceOffset,
                        destinationOffset: destination.destinationOffset,
                        size: destination.size,
                        scratch: scratch,
                        audit: audit)
                }
                close(sourceDescriptor)
            } catch {
                close(sourceDescriptor)
                throw error
            }
            for path in touched {
                if let descriptor = outputDescriptors[path] {
                    try Posix.fsync(descriptor, path: path)
                }
            }
            let digest = try HTTPRangeSourceByteProvider.destinationDigest(
                copy,
                partialDirectory: partialDirectory,
                scratch: scratch)
            try commit(
                RemoteCompletedRange(
                    id: copy.id,
                    destinationDigest: digest,
                    sourceBytes: copy.size,
                    destinationBytes: copy.destinations.reduce(0) { $0 + $1.size }))
            copiedBytes += copy.size
            progress(copiedBytes)
            // Only after the range is committed: the shard that supplied it is
            // no longer read by this plan.
            retirement.consume(copy) { sourcePath }
        }
    }

    private func resolvedSourcePath(_ shard: String) throws -> String {
        guard !shard.isEmpty,
            !shard.hasPrefix("/"),
            !shard.contains(".."),
            !shard.contains("/"),
            !shard.contains("\\")
        else {
            throw RepackError.configurationInvalid(
                detail: "unsafe local snapshot shard path \(shard)")
        }
        let root = URL(fileURLWithPath: snapshotDirectory).standardizedFileURL.path
        let path = URL(
            fileURLWithPath: shard,
            relativeTo:
                URL(fileURLWithPath: root, isDirectory: true)
        ).standardizedFileURL.path
        guard path.hasPrefix(root + "/") else {
            throw RepackError.configurationInvalid(
                detail: "local snapshot shard escapes its directory: \(shard)")
        }
        return path
    }

    private func outputDescriptor(
        path: String,
        cache: inout [String: Int32]
    ) throws -> Int32 {
        if let existing = cache[path],
            try Posix.descriptorMatchesPath(existing, path: path)
        {
            return existing
        }
        if let existing = cache.removeValue(forKey: path) {
            close(existing)
        }
        let descriptor = try Posix.openExistingRW(path)
        cache[path] = descriptor
        return descriptor
    }

    private func copyBytes(
        sourceDescriptor: Int32,
        sourcePath: String,
        destinationDescriptor: Int32,
        destinationPath: String,
        sourceOffset: UInt64,
        destinationOffset: UInt64,
        size: UInt64,
        scratch: UnsafeMutableRawBufferPointer,
        audit: RepackAudit
    ) throws {
        // `scratch` is the caller's tile buffer; without storage there is no
        // destination for a read, so this is refused rather than trapping.
        guard let scratchBase = scratch.baseAddress else {
            throw RepackError.configurationInvalid(
                detail: "the copy scratch buffer has no storage")
        }
        var remaining = size
        var source = sourceOffset
        var destination = destinationOffset
        while remaining > 0 {
            try Task.checkCancellation()
            let count = min(Int(remaining), scratch.count)
            try Posix.preadAll(
                fd: sourceDescriptor,
                path: sourcePath,
                buf: scratchBase,
                count: count,
                offset: source)
            try Posix.pwriteAll(
                fd: destinationDescriptor,
                path: destinationPath,
                buf: scratchBase,
                count: count,
                offset: destination)
            audit.recordTile(bytes: count)
            audit.recordRead(bytes: count)
            audit.recordWrite(bytes: count)
            remaining -= UInt64(count)
            source += UInt64(count)
            destination += UInt64(count)
        }
    }
}

/// Retires a local snapshot's source shards as their last range is committed.
///
/// A 122B snapshot and its install cannot both fit on a 1 TB Mac (129.8 GB +
/// ~125 GB). The copy plan names every range a shard supplies, so once the last
/// of them is committed nothing later reads that shard. Deleting it then turns
/// the two-copy peak into roughly one copy: the space a deleted shard frees is
/// the space the growing install needs.
///
/// The accounting is over the whole plan, not one batch: `copies` is the
/// complete coalesced plan, and a range already completed by an earlier
/// attempt still counts, because the shard must not be retired while a later
/// range still names it. A shard is unlinked only after that count reaches
/// zero, and only a safetensors shard (`model-*.safetensors`, or the
/// single-file `model.safetensors`) can be unlinked -- an index, config or
/// tokenizer is never a shard ID.
///
/// This is opt-in (`--retire-source-shards`) because it consumes the snapshot:
/// if the repack fails after a shard has gone, the snapshot has to be
/// regenerated before another attempt. That is the trade the 122B build makes,
/// and the reason the install is verified before the marker is trusted.
struct SourceRetirement {
    private var remaining: [String: UInt64]
    private let enabled: Bool

    init(copies: [CoalescedRangeCopy], enabled: Bool) {
        self.enabled = enabled
        var bytes: [String: UInt64] = [:]
        for copy in copies {
            bytes[copy.shardID, default: 0] += copy.size
        }
        self.remaining = bytes
    }

    /// Account for one copied range; unlink its shard when nothing later reads
    /// it. `sourcePath` is evaluated only on retirement, so the common path
    /// does no filesystem work.
    mutating func consume(_ copy: CoalescedRangeCopy, sourcePath: () -> String?) {
        guard enabled, let left = remaining[copy.shardID] else { return }
        let after = left > copy.size ? left - copy.size : 0
        guard after == 0 else {
            remaining[copy.shardID] = after
            return
        }
        remaining.removeValue(forKey: copy.shardID)
        guard let path = sourcePath(), Self.isSnapshotShard(path) else { return }
        try? FileManager.default.removeItem(atPath: path)
    }

    static func isSnapshotShard(_ path: String) -> Bool {
        let name = URL(fileURLWithPath: path).lastPathComponent
        // The converter's shards are `model-00001-of-00062.safetensors`; a
        // tiny snapshot is one plain `model.safetensors`. The index and the
        // config are neither, which is what keeps them alive.
        guard name.hasSuffix(".safetensors") else { return false }
        return name.hasPrefix("model-") || name == "model.safetensors"
    }
}
