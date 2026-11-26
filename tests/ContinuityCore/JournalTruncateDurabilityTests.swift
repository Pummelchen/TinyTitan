import Foundation
import Testing

@testable import ContinuityCore

/// Truncation is a durability operation, not a file-length assignment.
///
/// `O_TRUNC` makes the file empty in the page cache. Without a barrier the
/// drive can still hold the old length, so a power cut right after a `truncate`
/// brings the old bytes back — while the records appended after it, which were
/// barriered, stay. The journal would then be newer than its own beginning.
/// These pin the barrier, its failure and that the failure is not lost behind a
/// cleanup error.
@Suite struct JournalTruncateDurabilityTests {
    private func temporaryDirectory() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("journal-truncate-\(UUID().uuidString)")
    }

    private func task(_ title: String) -> JournalRecord {
        .task(ContinuityTask(title: title))
    }

    private func records(at url: URL) throws -> [JournalRecord] {
        try FileJournal.read(contentsOf: url)
    }

    /// A fresh journal with two durable records, closed. The caller reopens it
    /// with a fault-injecting filesystem.
    private func journalWithTwoRecords(_ directory: URL) async throws -> URL {
        let url = directory.appendingPathComponent("journal.ndjson")
        let journal = try FileJournal(url: url, idleDelay: .seconds(60))
        try await journal.append(task("first"))
        try await journal.append(task("second"))
        try await journal.sync()
        try await journal.shutDown()
        return url
    }

    @Test func aTruncationIsBarrieredBeforeItReturns() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = try await journalWithTwoRecords(directory)

        let fileSystem = JournalFaultFileSystem()
        let journal = try FileJournal(
            url: url, fileSystem: fileSystem, idleDelay: .seconds(60))
        let barriersBefore = fileSystem.fullSyncs
        try await journal.truncate()
        #expect(
            fileSystem.fullSyncs == barriersBefore + 1,
            "truncate returned without a barrier on the truncated file")
        #expect(try Data(contentsOf: url).isEmpty)
        #expect(await journal.pendingRecords == 0)

        // The file is a usable journal again, and the next record is durable.
        try await journal.append(task("after"))
        try await journal.sync()
        try await journal.shutDown()
        #expect(try records(at: url).count == 1)
        let reopened = try FileJournal(url: url)
        #expect(try await reopened.replay().count == 1)
        try await reopened.shutDown()
    }

    /// Both barriers fail: the truncation is visible but its durability is not
    /// proven. That is reported, the workspace stays held while the journal is
    /// open, and the file is still a usable journal afterwards.
    @Test func aFailedTruncationBarrierIsReportedAndTheJournalStaysUsable() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = try await journalWithTwoRecords(directory)

        let fileSystem = JournalFaultFileSystem()
        let journal = try FileJournal(
            url: url, fileSystem: fileSystem, idleDelay: .seconds(60))
        fileSystem.failFullSync(EIO)
        fileSystem.failFsync(EIO)
        do {
            try await journal.truncate()
            Issue.record("a truncation whose barrier failed was reported as durable")
        } catch let error as JournalError {
            guard case .writeFailed(let failedURL, let code) = error else {
                Issue.record("expected writeFailed, got \(error)")
                return
            }
            #expect(failedURL.path == url.path)
            #expect(code == EIO)
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
        // The truncation is what the file names now, and the actor follows it.
        #expect(try Data(contentsOf: url).isEmpty)
        #expect(await journal.pendingRecords == 0)
        #expect(throws: JournalError.self) { _ = try FileJournal(url: url) }

        // The journal is still usable: the failure was reported, not turned
        // into a dead handle.
        fileSystem.failFullSync(nil)
        fileSystem.failFsync(nil)
        try await journal.append(task("after"))
        try await journal.sync()
        #expect(try await journal.replay().count == 1)
        try await journal.shutDown()
        let reopened = try FileJournal(url: url)
        #expect(try await reopened.replay().count == 1)
        try await reopened.shutDown()
    }

    /// The reopen after the truncation can fail too. The barrier failure is the
    /// one the operation is about, and it must not be replaced by a cleanup
    /// error from the descriptor recovery.
    @Test func aReopenFailureDoesNotMaskTheBarrierFailure() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = try await journalWithTwoRecords(directory)

        let fileSystem = JournalFaultFileSystem()
        let journal = try FileJournal(
            url: url, fileSystem: fileSystem, idleDelay: .seconds(60))
        fileSystem.failFullSync(EIO)
        fileSystem.failFsync(EIO)
        // Opens of the journal path: 1 was the initial open in this test's
        // constructor. 2 is `truncate`'s O_TRUNC and 3 is its reopen.
        fileSystem.failOpen(at: url.path, call: 3, EMFILE)
        do {
            try await journal.truncate()
            Issue.record("a truncation whose barrier failed was reported as durable")
        } catch let error as JournalError {
            guard case .writeFailed(_, let code) = error else {
                Issue.record("the reopen error replaced the barrier error: \(error)")
                return
            }
            #expect(code == EIO)
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
        #expect(try Data(contentsOf: url).isEmpty)
        // No descriptor was left behind: an append reports EBADF rather than
        // writing into whatever the kernel handed that number to.
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
        try await journal.shutDown()
        let reopened = try FileJournal(url: url)
        #expect(try await reopened.replay().isEmpty)
        try await reopened.shutDown()
    }
}
