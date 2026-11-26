import Foundation
import Testing

@testable import ContinuityCore

/// A checkpoint is the only operation that replaces the one durable copy of
/// the journal, and "it worked" is three separate facts: the new bytes reached
/// the platter, the rename happened, and the directory entry is durable.
///
/// These tests fail each of those in turn and pin what the caller is told and
/// what the file holds afterwards. The filesystem is injected; the records,
/// the temp file, the lock and the bytes read back are real.
@Suite struct JournalDurabilityTests {
    private func temporaryDirectory() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("journal-durability-\(UUID().uuidString)")
    }

    private func makeJournal(
        at url: URL, _ fileSystem: JournalFaultFileSystem,
        idleDelay: Duration = .seconds(60)
    ) throws -> FileJournal {
        try FileJournal(
            url: url, fileSystem: fileSystem, idleDelay: idleDelay)
    }

    private func records(at url: URL) throws -> [JournalRecord] {
        try FileJournal.read(contentsOf: url)
    }

    private func task(_ title: String) -> JournalRecord {
        .task(ContinuityTask(title: title))
    }

    /// Records thrown away by a compaction that claimed to be replaceable.
    /// Every one of these tests starts from a file with two durable records.
    private func journalWithTwoRecords(
        _ directory: URL
    ) async throws -> (url: URL, journal: FileJournal, fileSystem: JournalFaultFileSystem) {
        let url = directory.appendingPathComponent("journal.ndjson")
        let fileSystem = JournalFaultFileSystem()
        let journal = try makeJournal(at: url, fileSystem)
        try await journal.append(task("first"))
        try await journal.append(task("second"))
        try await journal.sync()
        return (url, journal, fileSystem)
    }

    // MARK: - Before the rename

    /// The old code discarded the fallback barrier's answer and renamed
    /// regardless: a replacement whose bytes never reached the platter could
    /// replace the only durable copy. Both barriers failing must leave the
    /// file, its bytes and the append descriptor exactly as they were.
    @Test func aFailedDataBarrierLeavesTheJournalAndTheDescriptorUntouched() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let (url, journal, fileSystem) = try await journalWithTwoRecords(directory)
        let before = try Data(contentsOf: url)

        fileSystem.failFullSync(EIO)
        fileSystem.failFsync(EIO)
        do {
            try await journal.compact(sessionLog: SessionLogSnapshot(), memory: MemorySnapshot())
            Issue.record("a checkpoint whose data barrier failed was reported as committed")
        } catch let error as JournalError {
            guard case .writeFailed(let failedURL, let code) = error else {
                Issue.record("expected writeFailed for the temp, got \(error)")
                return
            }
            #expect(failedURL.path == url.appendingPathExtension("compacting").path)
            #expect(code == EIO)
        } catch {
            Issue.record("unexpected error type: \(error)")
        }

        // Not one byte of the old journal moved, and the half-written
        // replacement is gone rather than left as a trap for the next caller.
        #expect(try Data(contentsOf: url) == before)
        #expect(
            !FileManager.default.fileExists(
                atPath: url.appendingPathExtension("compacting").path))

        // And the descriptor still names that file: an append after the failed
        // compaction is a third record, not an EBADF or a write somewhere else.
        fileSystem.failFullSync(nil)
        fileSystem.failFsync(nil)
        try await journal.append(task("after"))
        try await journal.sync()
        #expect(try await journal.replay().count == 3)
        try await journal.shutDown()
        #expect(try records(at: url).count == 3)
    }

    /// The strong Darwin barrier is not implemented by every filesystem; the
    /// portable one is the fallback. A fallback that *succeeds* has to be
    /// enough, or every non-APFS store would stop compacting forever.
    @Test func aFallbackBarrierThatSucceedsStillReplacesTheCheckpoint() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let (url, journal, fileSystem) = try await journalWithTwoRecords(directory)

        fileSystem.failFullSync(EINVAL)
        try await journal.compact(sessionLog: SessionLogSnapshot(), memory: MemorySnapshot())
        #expect(fileSystem.fsyncs >= 1, "the fallback barrier was never called")
        #expect(try await journal.replay().count == 1)

        fileSystem.failFullSync(nil)
        try await journal.append(task("after"))
        #expect(try await journal.replay().count == 2)
        try await journal.shutDown()
        #expect(try records(at: url).count == 2)
    }

    @Test func aRenameThatFailsLeavesTheJournalInPlace() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let (url, journal, fileSystem) = try await journalWithTwoRecords(directory)
        let before = try Data(contentsOf: url)

        fileSystem.failRename(EACCES)
        do {
            try await journal.compact(sessionLog: SessionLogSnapshot(), memory: MemorySnapshot())
            Issue.record("a refused rename was reported as a committed checkpoint")
        } catch let error as JournalError {
            guard case .writeFailed(let failedURL, let code) = error else {
                Issue.record("expected writeFailed for the target, got \(error)")
                return
            }
            #expect(failedURL.path == url.path)
            #expect(code == EACCES)
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
        #expect(try Data(contentsOf: url) == before)
        #expect(
            !FileManager.default.fileExists(
                atPath: url.appendingPathExtension("compacting").path))

        fileSystem.failRename(nil)
        try await journal.append(task("after"))
        #expect(try await journal.replay().count == 3)
        try await journal.shutDown()
    }

    /// A cleanup that fails must not replace the failure that caused it: the
    /// caller needs to know the data barrier failed, because that is the one
    /// that says the old journal was preserved.
    @Test func aCleanupFailureDoesNotMaskTheBarrierFailure() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let (url, journal, fileSystem) = try await journalWithTwoRecords(directory)
        let before = try Data(contentsOf: url)

        fileSystem.configure { state in
            state.unlinkFailure = EACCES
        }
        fileSystem.failFullSync(EIO)
        fileSystem.failFsync(EIO)
        do {
            try await journal.compact(sessionLog: SessionLogSnapshot(), memory: MemorySnapshot())
            Issue.record("a failed checkpoint was reported as committed")
        } catch let error as JournalError {
            guard case .writeFailed(let failedURL, let code) = error else {
                Issue.record("expected the barrier failure, got \(error)")
                return
            }
            #expect(failedURL.path == url.appendingPathExtension("compacting").path)
            #expect(code == EIO)
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
        #expect(try Data(contentsOf: url) == before)
        fileSystem.configure { state in
            state.unlinkFailure = nil
        }
        fileSystem.failFullSync(nil)
        fileSystem.failFsync(nil)
        try await journal.shutDown()
    }

    // MARK: - After the rename

    /// The rename landed, so the path names the checkpoint. A directory that
    /// cannot be opened for its own barrier is a durability uncertainty, not a
    /// rollback: the caller is told, the descriptor follows the visible file,
    /// and what it appends is in the file that is there.
    @Test func aParentDirectoryThatCannotBeOpenedIsReportedAfterTheCheckpointLands() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let (url, journal, fileSystem) = try await journalWithTwoRecords(directory)

        fileSystem.failOpen(at: directory.path, EACCES)
        do {
            try await journal.compact(sessionLog: SessionLogSnapshot(), memory: MemorySnapshot())
            Issue.record("an unsynced directory was reported as a durable checkpoint")
        } catch let error as JournalError {
            guard case .cannotOpen(let failedURL, _) = error else {
                Issue.record("expected cannotOpen for the directory, got \(error)")
                return
            }
            #expect(failedURL.path == directory.path)
        } catch {
            Issue.record("unexpected error type: \(error)")
        }

        // The checkpoint is what the path names; nothing pretended it was not.
        #expect(try records(at: url).count == 1)
        #expect(
            !FileManager.default.fileExists(
                atPath: url.appendingPathExtension("compacting").path))

        // The descriptor was reopened on the visible inode, not left on the
        // unlinked old one: an append is durable and replays after it.
        fileSystem.configure { $0.openFailure.removeAll() }
        try await journal.append(task("after the uncertain commit"))
        try await journal.sync()
        try await journal.shutDown()
        let reopened = try FileJournal(url: url)
        let after = try await reopened.replay()
        #expect(after.count == 2)
        guard case .task(let appended)? = after.last else {
            Issue.record("the append after the uncertain commit is missing")
            try await reopened.shutDown()
            return
        }
        #expect(appended.title == "after the uncertain commit")
        try await reopened.shutDown()
    }

    @Test func aParentDirectoryThatCannotBeSyncedIsReportedAfterTheCheckpointLands() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let (url, journal, fileSystem) = try await journalWithTwoRecords(directory)

        fileSystem.failFsync(at: directory.path, EIO)
        do {
            try await journal.compact(sessionLog: SessionLogSnapshot(), memory: MemorySnapshot())
            Issue.record("an unsynced directory was reported as a durable checkpoint")
        } catch let error as JournalError {
            guard case .writeFailed(let failedURL, let code) = error else {
                Issue.record("expected writeFailed for the directory, got \(error)")
                return
            }
            #expect(failedURL.path == directory.path)
            #expect(code == EIO)
        } catch {
            Issue.record("unexpected error type: \(error)")
        }

        #expect(try records(at: url).count == 1)
        // The lock is on a sidecar, so it survived the replacement.
        #expect(throws: JournalError.self) { _ = try FileJournal(url: url) }
        fileSystem.configure { $0.fsyncFailurePaths.removeAll() }
        try await journal.append(task("after"))
        #expect(try await journal.replay().count == 2)
        try await journal.shutDown()
        let next = try FileJournal(url: url)
        #expect(try await next.replay().count == 2)
        try await next.shutDown()
    }

    /// The rename happened but the new file cannot be opened for appending.
    /// There is no old inode to fall back to, so the only safe answer is to
    /// report it and refuse to append — a number left in the field would name
    /// a closed descriptor the kernel may already have reused.
    @Test func aCheckpointThatCannotBeReopenedIsReportedAndNothingAppendsThroughIt() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let (url, journal, fileSystem) = try await journalWithTwoRecords(directory)

        fileSystem.failReopen(at: url.path, EMFILE)
        do {
            try await journal.compact(sessionLog: SessionLogSnapshot(), memory: MemorySnapshot())
            Issue.record("a checkpoint whose reopen failed was reported as committed")
        } catch let error as JournalError {
            guard case .cannotOpen(let failedURL, _) = error else {
                Issue.record("expected cannotOpen for the journal, got \(error)")
                return
            }
            #expect(failedURL.path == url.path)
        } catch {
            Issue.record("unexpected error type: \(error)")
        }

        // The checkpoint is on disk; the actor just cannot append to it.
        #expect(try records(at: url).count == 1)
        do {
            try await journal.append(task("lost"))
            Issue.record("an append after a failed reopen reported success")
        } catch let error as JournalError {
            guard case .writeFailed(_, let code) = error else {
                Issue.record("expected writeFailed, got \(error)")
                return
            }
            #expect(code == EBADF)
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
        // Shutdown still releases the workspace, so the file can be reopened.
        try await journal.shutDown()
        let next = try FileJournal(url: url)
        #expect(try await next.replay().count == 1)
        try await next.shutDown()
    }

    @Test func aSuccessfulCheckpointIsOneRecordAndStillWritable() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let (url, journal, fileSystem) = try await journalWithTwoRecords(directory)

        try await journal.compact(sessionLog: SessionLogSnapshot(), memory: MemorySnapshot())
        #expect(fileSystem.renames == 1)
        #expect(try await journal.replay().count == 1)
        try await journal.append(task("after"))
        #expect(try await journal.replay().count == 2)
        try await journal.shutDown()
        #expect(try records(at: url).count == 2)
    }
}
