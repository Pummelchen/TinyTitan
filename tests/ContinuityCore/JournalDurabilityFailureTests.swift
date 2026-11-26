import Foundation
import Testing

@testable import ContinuityCore

/// A durability failure that happens with no caller holding the throw.
///
/// The scheduled barrier runs in a task nobody awaits, and the old
/// `shutDown` wrapped both of its flushes in `try?` even though it is declared
/// `async throws`. A drive that stopped accepting writes could therefore leave
/// records unsynced, the store reporting itself durable, and the process exit
/// look clean. These pin the two channels the caller actually has: the
/// actor's recorded failure, and the throw from `shutDown`.
@Suite struct JournalDurabilityFailureTests {
    private func temporaryURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("journal-failure-\(UUID().uuidString)")
            .appendingPathComponent("journal.ndjson")
    }

    private func task(_ title: String) -> JournalRecord {
        .task(ContinuityTask(title: title))
    }

    /// The scheduled barrier has no caller, so its failure is only useful if
    /// it is recorded on the actor. The record must stay pending: the flush
    /// that failed covered nothing.
    @Test func aBackgroundBarrierFailureIsVisibleAndKeepsTheRecordPending() async throws {
        let url = temporaryURL()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let fileSystem = JournalFaultFileSystem()
        let journal = try FileJournal(
            url: url, fileSystem: fileSystem, idleDelay: .zero)
        fileSystem.failFullSync(EIO)
        fileSystem.failFsync(EIO)

        try await journal.append(task("in the page cache"))
        // Nothing sleeps here: the scheduled task is awaited, and it records
        // its outcome on the actor before completing.
        await journal.awaitScheduledBarrier()
        #expect(
            await journal.durabilityFailure != nil,
            "a failed background barrier left no trace")
        #expect(await journal.pendingRecords == 1)

        // An explicit sync retries; a success is what clears the diagnostic,
        // because the records are durable at that point.
        fileSystem.failFullSync(nil)
        fileSystem.failFsync(nil)
        try await journal.sync()
        #expect(await journal.pendingRecords == 0)
        #expect(await journal.durabilityFailure == nil)
        try await journal.shutDown()
    }

    /// Shutdown is a flush point. A flush that fails must be thrown after the
    /// resources are released, not swallowed before them.
    @Test func shutdownThrowsAFlushFailureAndStillReleasesTheWorkspace() async throws {
        let url = temporaryURL()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let fileSystem = JournalFaultFileSystem()
        let journal = try FileJournal(
            url: url, fileSystem: fileSystem, idleDelay: .seconds(60))
        try await journal.append(task("unsynced"))
        fileSystem.failFullSync(EIO)
        fileSystem.failFsync(EIO)

        do {
            try await journal.shutDown()
            Issue.record("a shutdown whose flush failed reported success")
        } catch let error as JournalError {
            guard case .writeFailed(_, let code) = error else {
                Issue.record("expected writeFailed, got \(error)")
                return
            }
            #expect(code == EIO)
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
        #expect(await journal.pendingRecords == 1)

        // The resources went back even though the flush failed: no append can
        // reach a closed descriptor, and the workspace lock is free.
        do {
            try await journal.append(task("after shutdown"))
            Issue.record("an append after shutdown reached a descriptor")
        } catch let error as JournalError {
            guard case .writeFailed(_, let code) = error else {
                Issue.record("expected writeFailed, got \(error)")
                return
            }
            #expect(code == EBADF)
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
        let next = try FileJournal(url: url, fileSystem: JournalFaultFileSystem())
        // The bytes are in the file (the write reached the kernel); only the
        // barrier failed, which is exactly what the error said.
        #expect(try await next.replay().count == 1)
        try await next.shutDown()
    }

    /// A shutdown with nothing left to flush is not a failure, and calling it
    /// twice must not close a descriptor number twice or throw a second time.
    @Test func repeatedShutdownIsSafe() async throws {
        let url = temporaryURL()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let journal = try FileJournal(
            url: url, fileSystem: JournalFaultFileSystem(), idleDelay: .seconds(60))
        try await journal.append(task("synced"))
        try await journal.sync()
        try await journal.shutDown()
        try await journal.shutDown()
        #expect(await journal.pendingRecords == 0)
    }

    /// Cancellation of the scheduled sleep is expected — a newer append
    /// replaced the barrier — and must not be recorded as a storage failure.
    @Test func aReplacedScheduledBarrierIsNotAFailure() async throws {
        let url = temporaryURL()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let journal = try FileJournal(
            url: url, fileSystem: JournalFaultFileSystem(), idleDelay: .seconds(60))
        try await journal.append(task("first"))
        try await journal.append(task("second"))
        #expect(await journal.durabilityFailure == nil)
        #expect(await journal.pendingRecords == 2)
        try await journal.sync()
        #expect(await journal.durabilityFailure == nil)
        try await journal.shutDown()
    }

    /// The engine is the caller that owns the journal's lifetime. Its
    /// `shutDown` used to discard both failures; now the last flush and the
    /// close are both reported on the engine's durability channel, which the
    /// memory service reads.
    @Test func theEngineRecordsAFlushFailureAtShutdown() async throws {
        let url = temporaryURL()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let fileSystem = JournalFaultFileSystem()
        let journal = try FileJournal(
            url: url, fileSystem: fileSystem, idleDelay: .seconds(60))
        let engine = ContinuityEngine(journal: journal)
        try await engine.start()
        try await engine.createTask(title: "work")
        fileSystem.failFullSync(EIO)
        fileSystem.failFsync(EIO)

        await engine.shutDown()
        #expect(
            await engine.journalFailure != nil,
            "the engine reported a clean shutdown after its flush failed")

        // The workspace is free even though the flush failed.
        let next = try FileJournal(url: url, fileSystem: JournalFaultFileSystem())
        try await next.shutDown()
    }
}
