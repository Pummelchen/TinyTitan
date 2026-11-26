import Foundation
import Testing

@testable import ContinuityCore

/// A journal record's timestamps are part of the record.
///
/// `Date` holds sub-second time; `.iso8601` without a fractional part throws it
/// away on replay, so a restored record is not the one that was written and two
/// events in the same second can come back indistinguishable. The journal now
/// writes milliseconds and still reads the whole-second form every earlier
/// build wrote: an existing install must replay unchanged.
@Suite struct JournalTimestampTests {
    private func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("journal-timestamp-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        return directory
    }

    private func line(_ url: URL) throws -> String {
        let data = try Data(contentsOf: url)
        return try #require(String(bytes: data, encoding: .utf8))
    }

    @Test func aSubSecondTaskRecordRoundTripsExactly() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let createdAt = Date(timeIntervalSince1970: 1_700_000_000.123)
        let updatedAt = Date(timeIntervalSince1970: 1_700_000_060.987)
        let task = ContinuityTask(
            title: "precise", objective: "milliseconds",
            createdAt: createdAt, updatedAt: updatedAt)
        let journal = try FileJournal(url: url, idleDelay: .seconds(60))
        try await journal.append(.task(task))
        try await journal.sync()
        // The precision is in the bytes, not only in memory.
        #expect(try line(url).contains("2023-11-14T22:13:20.123Z"))
        #expect(try line(url).contains("2023-11-14T22:14:20.987Z"))
        try await journal.shutDown()

        let reopened = try FileJournal(url: url)
        let replayed = try await reopened.replay()
        #expect(replayed == [.task(task)], "the replayed record is not the written one")
        try await reopened.shutDown()
    }

    /// A value with more precision than the format carries loses the excess,
    /// at a documented place: milliseconds survive, nothing below them does.
    @Test func theDocumentedPrecisionIsMilliseconds() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let written = Date(timeIntervalSince1970: 1_700_000_000.123456)
        let task = ContinuityTask(title: "sub-millisecond", createdAt: written, updatedAt: written)
        let journal = try FileJournal(url: url, idleDelay: .seconds(60))
        try await journal.append(.task(task))
        try await journal.sync()
        #expect(try line(url).contains("2023-11-14T22:13:20.123Z"))
        try await journal.shutDown()

        let reopened = try FileJournal(url: url)
        let replayed = try await reopened.replay()
        guard case .task(let restored)? = replayed.first else {
            Issue.record("the record did not survive as a task")
            try await reopened.shutDown()
            return
        }
        let expected = Date(timeIntervalSince1970: 1_700_000_000.123)
        #expect(restored.createdAt == expected)
        #expect(
            abs(restored.createdAt.timeIntervalSince1970 - written.timeIntervalSince1970) < 0.001)
        try await reopened.shutDown()
    }

    /// The fixture an install written before this fix holds: whole seconds,
    /// unsorted keys, no fractional part. It must decode to the same instant.
    @Test func anOldWholeSecondTimestampStillDecodes() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let instant = Date(timeIntervalSince1970: 1_700_000_000)
        let task = ContinuityTask(title: "legacy", createdAt: instant, updatedAt: instant)

        let legacy = JSONEncoder()
        legacy.dateEncodingStrategy = .iso8601
        legacy.outputFormatting = [.withoutEscapingSlashes]
        var data = try legacy.encode(JournalRecord.task(task))
        data.append(0x0A)
        // Pin the old encoding rather than only producing it: if Foundation's
        // `.iso8601` ever gained a fractional part, this fixture would stop
        // being the old format.
        let text = try #require(String(bytes: data, encoding: .utf8))
        #expect(text.contains("\"createdAt\":\"2023-11-14T22:13:20Z\""))
        try data.write(to: url)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o600], ofItemAtPath: url.path)

        let detailed = try FileJournal.readDetailed(contentsOf: url)
        #expect(detailed.damage == nil)
        #expect(detailed.records == [.task(task)])
    }

    /// Every other record kind carries dates too; a checkpoint is the one a
    /// restart actually reads. Engine-created `Date()` values are not aligned
    /// to the journal's milliseconds, so the assertion is the documented
    /// precision, not identity.
    @Test func aCheckpointRoundTripsItsTimestamps() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let journal = try FileJournal(url: url, idleDelay: .seconds(60))
        let engine = ContinuityEngine(journal: journal)
        try await engine.start()
        let task = try await engine.createTask(title: "checkpoint", objective: "dates")
        let session = try await engine.beginSession(taskID: task.id)
        try await engine.remember(
            sessionID: session.id, namespace: "n", key: "k", value: "v")
        let item = try #require(
            await engine.recall(taskID: task.id, namespace: "n", key: "k"))
        try await engine.compactJournal()
        await engine.shutDown()

        let reopened = try FileJournal(url: url)
        let records = try await reopened.replay()
        #expect(records.count == 1)
        guard case .checkpoint(let log, let memory)? = records.first else {
            Issue.record("the checkpoint did not decode")
            try await reopened.shutDown()
            return
        }
        func sameMillisecond(_ first: Date, _ second: Date) -> Bool {
            abs(first.timeIntervalSince(second)) < 0.001
        }
        let restoredTask = try #require(log.tasks.first { $0.id == task.id })
        #expect(sameMillisecond(restoredTask.createdAt, task.createdAt))
        #expect(sameMillisecond(restoredTask.updatedAt, task.updatedAt))
        let restoredSession = try #require(log.sessions.first { $0.id == session.id })
        #expect(sameMillisecond(restoredSession.startedAt, session.startedAt))
        let restoredItem = try #require(memory.items.first { $0.key == item.key })
        #expect(sameMillisecond(restoredItem.createdAt, item.createdAt))
        #expect(sameMillisecond(restoredItem.updatedAt, item.updatedAt))
        try await reopened.shutDown()
    }
}
