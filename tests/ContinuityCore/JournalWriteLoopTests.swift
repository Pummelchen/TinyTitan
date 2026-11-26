import Foundation
import Testing

@testable import ContinuityCore

/// Progress in the two write loops.
///
/// `write(2)` on a regular file advances by a positive number or fails; a
/// zero return with bytes still to write is not a short write, it is a call
/// that did nothing. The loops used to advance by whatever they were given,
/// so zero left the offset where it was and spun. These inject every outcome a
/// loop has to survive and pin the bytes that actually reached the file.
@Suite struct JournalWriteLoopTests {
    private func temporaryDirectory() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("journal-write-\(UUID().uuidString)")
    }

    private func records(at url: URL) throws -> [JournalRecord] {
        try FileJournal.read(contentsOf: url)
    }

    /// The bytes a record encodes to, from a journal that is not the one
    /// under test. Same record value, same encoder, same bytes.
    private func encoded(_ record: JournalRecord, in directory: URL) async throws -> Data {
        let url = directory.appendingPathComponent("control-\(UUID().uuidString).ndjson")
        let journal = try FileJournal(url: url)
        try await journal.append(record)
        try await journal.sync()
        try await journal.shutDown()
        return try Data(contentsOf: url)
    }

    // MARK: - Append

    /// The failure this prevents: a `write` that reports zero with bytes
    /// remaining makes the loop retry the same bytes forever. It must be a
    /// reported I/O failure, and it must be reported from the first one — not
    /// after an unbounded number of attempts.
    @Test func aZeroProgressWriteFailsTheAppendInsteadOfSpinning() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let fileSystem = JournalFaultFileSystem()
        // If the loop does not stop on the zero, this next outcome terminates
        // it: the test fails on the count rather than hanging the runner.
        fileSystem.planWrites([.zero, .failure(EPIPE)])
        let journal = try FileJournal(
            url: url, fileSystem: fileSystem, idleDelay: .seconds(60))

        do {
            try await journal.append(.task(ContinuityTask(title: "never lands")))
            Issue.record("a write that made no progress was reported as success")
        } catch let error as JournalError {
            guard case .writeFailed(_, let code) = error else {
                Issue.record("expected writeFailed, got \(error)")
                return
            }
            #expect(code == EIO)
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
        #expect(fileSystem.writes == 1, "the loop retried a zero-progress write")
        #expect(await journal.pendingRecords == 0)
        #expect(try records(at: url).isEmpty)
        try await journal.shutDown()
    }

    /// A short write is normal and must be continued from where it stopped.
    /// The bytes on disk after a failure are the exact prefix that was
    /// written, and the next append repairs the torn tail instead of fusing a
    /// new record onto it.
    @Test func aShortWriteThenAZeroKeepsTheExactPrefixAndRepairsTheTail() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let task = ContinuityTask(title: "first")
        let fullEncoding = try await encoded(.task(task), in: directory)

        let fileSystem = JournalFaultFileSystem()
        fileSystem.planWrites([.bytes(5), .zero, .failure(EPIPE)])
        let journal = try FileJournal(
            url: url, fileSystem: fileSystem, idleDelay: .seconds(60))
        do {
            try await journal.append(.task(task))
            Issue.record("a write that made no progress was reported as success")
        } catch let error as JournalError {
            guard case .writeFailed(_, let code) = error else {
                Issue.record("expected writeFailed, got \(error)")
                return
            }
            #expect(code == EIO)
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
        #expect(fileSystem.writes == 2, "the loop did not stop at the zero")
        let partial = try Data(contentsOf: url)
        #expect(partial == Data(fullEncoding.prefix(5)), "the written prefix is not the record's")

        // The next append cuts the partial record off first. Appending onto it
        // would fuse the two lines and lose the successful record too.
        fileSystem.planWrites([])
        try await journal.append(.task(ContinuityTask(title: "second")))
        try await journal.sync()
        let replayed = try await journal.replay()
        #expect(replayed.count == 1, "the torn record fused with the next one")
        guard case .task(let only)? = replayed.first else {
            Issue.record("expected the second record")
            try await journal.shutDown()
            return
        }
        #expect(only.title == "second")
        try await journal.shutDown()
        #expect(try records(at: url).count == 1)
    }

    @Test func anInterruptedWriteIsRetried() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let task = ContinuityTask(title: "interrupted")
        let fullEncoding = try await encoded(.task(task), in: directory)

        let fileSystem = JournalFaultFileSystem()
        fileSystem.planWrites([.interrupted, .bytes(4), .all])
        let journal = try FileJournal(
            url: url, fileSystem: fileSystem, idleDelay: .seconds(60))
        try await journal.append(.task(task))
        try await journal.sync()
        #expect(fileSystem.writes == 3)
        #expect(try Data(contentsOf: url) == fullEncoding)
        try await journal.shutDown()
        #expect(try records(at: url).count == 1)
    }

    @Test func manyShortWritesAssembleTheWholeRecord() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let task = ContinuityTask(title: "in pieces")
        let fullEncoding = try await encoded(.task(task), in: directory)

        let fileSystem = JournalFaultFileSystem()
        fileSystem.planWrites([.bytes(1), .bytes(2), .bytes(3), .all])
        let journal = try FileJournal(
            url: url, fileSystem: fileSystem, idleDelay: .seconds(60))
        try await journal.append(.task(task))
        try await journal.sync()
        #expect(fileSystem.writes == 4)
        #expect(try Data(contentsOf: url) == fullEncoding)
        try await journal.shutDown()
    }

    @Test func aHardWriteFailureIsReportedWithItsErrno() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let fileSystem = JournalFaultFileSystem()
        fileSystem.planWrites([.failure(ENOSPC)])
        let journal = try FileJournal(
            url: url, fileSystem: fileSystem, idleDelay: .seconds(60))

        do {
            try await journal.append(.task(ContinuityTask(title: "full disk")))
            Issue.record("a full disk was reported as a successful append")
        } catch let error as JournalError {
            guard case .writeFailed(_, let code) = error else {
                Issue.record("expected writeFailed, got \(error)")
                return
            }
            #expect(code == ENOSPC)
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
        #expect(fileSystem.writes == 1)
        #expect(try Data(contentsOf: url).isEmpty)
        try await journal.shutDown()
    }

    /// An empty batch is a no-op, not an empty `write(2)` that could report
    /// zero progress.
    @Test func anEmptyAppendWritesNothing() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let fileSystem = JournalFaultFileSystem()
        let journal = try FileJournal(
            url: url, fileSystem: fileSystem, idleDelay: .seconds(60))
        try await journal.append([JournalRecord]())
        #expect(fileSystem.writes == 0)
        #expect(await journal.pendingRecords == 0)
        try await journal.shutDown()
        #expect(try Data(contentsOf: url).isEmpty)
    }

    // MARK: - The checkpoint write

    /// The checkpoint helper had the same loop. A zero there must fail the
    /// compaction — never rename a file whose bytes did not all go out.
    @Test func theCheckpointWriteHasTheSameZeroProgressGuard() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let fileSystem = JournalFaultFileSystem()
        fileSystem.planWrites([.zero, .failure(EPIPE)])
        let journal = try FileJournal(
            url: url, fileSystem: fileSystem, idleDelay: .seconds(60))
        let before = try Data(contentsOf: url)

        do {
            try await journal.compact(sessionLog: SessionLogSnapshot(), memory: MemorySnapshot())
            Issue.record("a checkpoint write that made no progress was renamed")
        } catch let error as JournalError {
            guard case .writeFailed(let failedURL, let code) = error else {
                Issue.record("expected writeFailed, got \(error)")
                return
            }
            #expect(failedURL.path == url.appendingPathExtension("compacting").path)
            #expect(code == EIO)
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
        #expect(fileSystem.writes == 1, "the checkpoint loop retried a zero-progress write")
        #expect(fileSystem.renames == 0, "an incomplete checkpoint was renamed")
        #expect(try Data(contentsOf: url) == before)
        try await journal.shutDown()
    }
}
