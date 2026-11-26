import Foundation

// Rewriting a journal: the checkpoint replacement and the truncation, each of
// which replaces the file the append descriptor points at.
//
// Split out of `Journal.swift` under the 500-line-per-file rule (audit 0010).
// Every step that decides whether the replacement may happen — the data
// barrier, the rename, the directory barrier — is here, because the answer to
// "did this succeed" is not one bit.

extension FileJournal {

    /// Replace the journal with a single checkpoint.
    ///
    /// Three things can go wrong, and they are not the same thing:
    ///
    /// - The replacement's own bytes never reached the platter. The rename
    ///   must not happen, because it would replace the only durable copy with
    ///   an unproven one. The old file stays, the descriptor still writes to
    ///   it, and the error is the caller's.
    /// - The rename failed. Same answer: nothing was replaced.
    /// - The rename landed but the parent directory was not made durable.
    ///   The checkpoint is what the path names *now*, so the descriptor has to
    ///   follow it; a crash may still see the old directory entry, which is a
    ///   durability uncertainty the caller has to be told about rather than a
    ///   rollback this type could perform.
    public func compact(sessionLog: SessionLogSnapshot, memory: MemorySnapshot) async throws {
        await acquireRewrite()
        defer { releaseRewrite() }
        try refuseDamagedReplacement()
        await settleBarrier()
        var data = try encoder.encode(JournalRecord.checkpoint(sessionLog, memory))
        data.append(0x0A)
        let temporary = url.appendingPathExtension("compacting")
        let target = url
        let payload = data
        let fileSystem = self.fileSystem
        // Write, barrier, rename and directory barrier all happen on the
        // blocking queue: compaction is the largest write this type makes and
        // the one most likely to be big enough to notice.
        let outcome: CheckpointOutcome = await withCheckedContinuation { continuation in
            Self.blockingQueue.async {
                continuation.resume(
                    returning: Self.writeCheckpoint(
                        payload, to: temporary,
                        replacing: target, using: fileSystem))
            }
        }
        switch outcome {
        case .failed(let error):
            // Nothing was renamed. The old journal and the append descriptor
            // are untouched, which is the whole point of checking the barrier
            // before committing.
            throw error
        case .committed, .committedWithUncertainDurability:
            // The lock lives on a sidecar, so swapping this file never gives
            // it up. Reopen the append descriptor on the new inode.
            //
            // Forgetting the number before reopening is the point: if
            // `openForAppend` throws, the field must not still hold the
            // descriptor just closed, or the `descriptor >= 0` guard in
            // `writeFully` and the barrier passes and a later append writes
            // into whatever object the kernel has since handed that number
            // to. The generation moves with it for the same reason: a barrier
            // still in flight on the old descriptor must not settle the new
            // file's accounting.
            if descriptor >= 0 { fileSystem.close(descriptor) }
            descriptor = -1
            descriptorGeneration &+= 1
            appendedRecords = 0
            durableRecords = 0
            lastBarrier = ContinuousClock.now
            descriptor = try Self.openForAppend(url, using: fileSystem)
            if case .committedWithUncertainDurability(let error) = outcome {
                throw error
            }
        }
    }

    /// A replay that skipped complete records must not be followed by a
    /// rewrite that would omit them. The records that did decode are in RAM;
    /// the ones that did not are only in this file.
    private func refuseDamagedReplacement() throws {
        if let damage = replayDamage {
            throw JournalError.damagedReplay(url, lines: damage.lineNumbers)
        }
    }

    /// Forget the damage a replay reported, so a compaction may replace the
    /// file after the damaged bytes have been copied somewhere else.
    ///
    /// This is the explicit operator-approved recovery step; nothing in the
    /// package calls it, because only a person can decide that the bytes are
    /// recoverable or no longer wanted. Without it the journal keeps working
    /// and keeps growing — a safe state, not a silent one.
    public func discardDamagedRecords() {
        replayDamage = nil
    }

    /// The checkpoint write, in full, for the blocking queue.
    ///
    /// Replace only once the new file is complete on disk, so a crash during
    /// compaction leaves the old journal intact, and sync the directory so
    /// the rename itself is durable rather than only the bytes it points at.
    private static func writeCheckpoint(
        _ data: Data, to temporary: URL,
        replacing target: URL, using fileSystem: any JournalFileSystem
    ) -> CheckpointOutcome {
        // A leftover temp from a crash is ours to remove. A failure here is
        // not fatal: `O_NOFOLLOW` and the type check below are what stop the
        // create from going through a planted link, and the unlink cannot be
        // relied on as a guard anyway.
        _ = fileSystem.unlink(temporary.path)
        // The temp name is predicted rather than unique, so the unlink above
        // is not a guard: it unlinks a planted link along with its anchor.
        // O_NOFOLLOW is what stops the create writing through one.
        let handle = fileSystem.open(
            path: temporary.path,
            flags: O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK,
            mode: 0o600)
        guard handle >= 0 else {
            return .failed(
                .cannotOpen(temporary, underlying: String(cString: strerror(-handle))))
        }
        do {
            try validateRegularFile(
                descriptor: handle, url: temporary, using: fileSystem,
                ownerAndMode: false)
        } catch {
            fileSystem.close(handle)
            _ = fileSystem.unlink(temporary.path)
            return .failed(error)
        }
        if let error = writeFully(data, to: handle, url: temporary, using: fileSystem) {
            fileSystem.close(handle)
            _ = fileSystem.unlink(temporary.path)
            return .failed(error)
        }
        // The strong barrier, with the portable one as a checked fallback.
        // The old code discarded the fallback's answer, which is how a
        // replacement whose bytes never reached the platter was renamed over
        // the only durable copy.
        var barrier = fileSystem.fullSync(handle)
        if barrier != 0 { barrier = fileSystem.fsync(handle) }
        fileSystem.close(handle)
        if barrier != 0 {
            _ = fileSystem.unlink(temporary.path)
            return .failed(.writeFailed(temporary, errno: -barrier))
        }

        let renamed = fileSystem.rename(from: temporary.path, to: target.path)
        guard renamed == 0 else {
            let code = -renamed
            _ = fileSystem.unlink(temporary.path)
            return .failed(.writeFailed(target, errno: code))
        }
        // From here the replacement is what the path names. A failure to make
        // the directory entry durable is reported, not rolled back: the new
        // file is complete and deleting it would leave less than there is.
        let directoryURL = target.deletingLastPathComponent()
        let directory = fileSystem.open(
            path: directoryURL.path,
            flags: O_RDONLY | O_CLOEXEC | O_DIRECTORY,
            mode: 0)
        guard directory >= 0 else {
            return .committedWithUncertainDurability(
                .cannotOpen(directoryURL, underlying: String(cString: strerror(-directory))))
        }
        let synced = fileSystem.fsync(directory)
        fileSystem.close(directory)
        if synced != 0 {
            return .committedWithUncertainDurability(
                .writeFailed(directoryURL, errno: -synced))
        }
        return .committed
    }

    public func truncate() async throws {
        await acquireRewrite()
        defer { releaseRewrite() }
        try refuseDamagedReplacement()
        await settleBarrier()
        if descriptor >= 0 { fileSystem.close(descriptor) }
        descriptor = -1
        descriptorGeneration &+= 1
        let emptied = fileSystem.open(
            path: url.path,
            flags: O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK,
            mode: 0o600)
        guard emptied >= 0 else {
            throw JournalError.cannotOpen(url, underlying: String(cString: strerror(-emptied)))
        }
        fileSystem.close(emptied)
        appendedRecords = 0
        durableRecords = 0
        descriptor = try Self.openForAppend(url, using: fileSystem)
    }
}

/// What a checkpoint replacement actually achieved.
enum CheckpointOutcome {
    /// Renamed, and the directory entry made durable.
    case committed
    /// Renamed, so the path names the new checkpoint, but the directory was
    /// not opened or synced: the replacement may not survive a power cut.
    case committedWithUncertainDurability(JournalError)
    /// Nothing was renamed. The old journal is still the file at the path.
    case failed(JournalError)
}
