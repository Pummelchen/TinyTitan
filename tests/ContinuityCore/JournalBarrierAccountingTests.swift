import Foundation
import Testing

@testable import ContinuityCore

/// What a durability barrier covers.
///
/// A barrier is dispatched, then it waits on the drive. The actor admits
/// appends while it waits — that is what an `await` is — so a record can reach
/// the file after the barrier was sent and before it reports back. That record
/// was not what the barrier was asked to make durable. Clearing the whole
/// count when the barrier returns would mark it covered, and a run that then
/// stopped would lose it with no pending record left to force another flush.
@Suite struct JournalBarrierAccountingTests {
    private func temporaryURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("barrier-accounting-\(UUID().uuidString)")
            .appendingPathComponent("journal.ndjson")
    }

    private func task(_ title: String) -> JournalRecord {
        .task(ContinuityTask(title: title))
    }

    /// Append A, start the barrier, hold it inside the syscall, append B, then
    /// let the barrier finish. B must still be pending; a second barrier must
    /// clear it.
    ///
    /// The hold is a semaphore the stub signals from the barrier's own queue,
    /// so the interleaving is the one under test rather than one a sleep
    /// happened to produce.
    @Test func aRecordWrittenDuringABarrierIsNotCoveredByIt() async throws {
        let url = temporaryURL()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let fileSystem = JournalFaultFileSystem()
        let entered = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        fileSystem.holdFullSync(entered: entered, release: release)
        let journal = try FileJournal(
            url: url, fileSystem: fileSystem, idleDelay: .seconds(60))

        try await journal.append(task("before the barrier"))
        #expect(await journal.pendingRecords == 1)

        let syncing = Task { try await journal.sync() }
        #expect(
            await waitForSignal(entered),
            "the barrier never reached the syscall")

        // The actor is suspended inside the barrier, so this append is
        // admitted now, before the barrier reports back.
        try await journal.append(task("during the barrier"))
        #expect(await journal.pendingRecords == 2)

        release.signal()
        try await syncing.value
        #expect(
            await journal.pendingRecords == 1,
            "a record written after the barrier was dispatched was marked durable")

        // The record is a record, not a leak: the next barrier covers it.
        try await journal.sync()
        #expect(await journal.pendingRecords == 0)
        try await journal.shutDown()
        #expect(try FileJournal.read(contentsOf: url).count == 2)
    }

    /// A barrier that fails covers nothing. The old code left the count alone
    /// on the failure path but returned success from the scheduled path; the
    /// count here is the part that has to survive either way.
    @Test func aFailedBarrierKeepsThePendingRecords() async throws {
        let url = temporaryURL()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let fileSystem = JournalFaultFileSystem()
        let journal = try FileJournal(
            url: url, fileSystem: fileSystem, idleDelay: .seconds(60))
        try await journal.append(task("one"))
        try await journal.append(task("two"))

        fileSystem.failFullSync(EIO)
        fileSystem.failFsync(EIO)
        do {
            try await journal.sync()
            Issue.record("a barrier that failed was reported as successful")
        } catch let error as JournalError {
            guard case .writeFailed(_, let code) = error else {
                Issue.record("expected writeFailed, got \(error)")
                return
            }
            #expect(code == EIO)
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
        #expect(await journal.pendingRecords == 2)
        #expect(await journal.durabilityFailure != nil)

        fileSystem.failFullSync(nil)
        fileSystem.failFsync(nil)
        try await journal.sync()
        #expect(await journal.pendingRecords == 0)
        #expect(await journal.durabilityFailure == nil)
        try await journal.shutDown()
    }

    /// Two barriers may overlap; both are asked about the same watermark and
    /// the account must settle once, not go negative or clear a later record.
    @Test func overlappingBarriersSettleTheWatermarkOnce() async throws {
        let url = temporaryURL()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let fileSystem = JournalFaultFileSystem()
        let journal = try FileJournal(
            url: url, fileSystem: fileSystem, idleDelay: .seconds(60))
        for index in 0..<8 {
            try await journal.append(task("t\(index)"))
        }
        let first = Task { try await journal.sync() }
        let second = Task { try await journal.sync() }
        try await first.value
        try await second.value
        #expect(await journal.pendingRecords == 0)
        #expect(await journal.durabilityFailure == nil)
        try await journal.shutDown()
        #expect(try FileJournal.read(contentsOf: url).count == 8)
    }

    /// A batch is one write but several records, and the watermark counts
    /// records: a later append must not be covered by a barrier that only saw
    /// the batch.
    @Test func aBatchAdvancesTheWatermarkByItsRecordCount() async throws {
        let url = temporaryURL()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let fileSystem = JournalFaultFileSystem()
        let entered = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        fileSystem.holdFullSync(entered: entered, release: release)
        let journal = try FileJournal(
            url: url, fileSystem: fileSystem, idleDelay: .seconds(60))

        try await journal.append([task("a"), task("b"), task("c")])
        #expect(await journal.pendingRecords == 3)
        let syncing = Task { try await journal.sync() }
        #expect(await waitForSignal(entered))
        try await journal.append(task("d"))
        #expect(await journal.pendingRecords == 4)
        release.signal()
        try await syncing.value
        #expect(await journal.pendingRecords == 1)
        try await journal.sync()
        #expect(await journal.pendingRecords == 0)
        try await journal.shutDown()
        #expect(try FileJournal.read(contentsOf: url).count == 4)
    }
}
