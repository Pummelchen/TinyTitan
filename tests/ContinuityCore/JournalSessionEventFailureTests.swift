import Foundation
import Testing

@testable import ContinuityCore

/// Two session-log writes are best-effort, and that is now said out loud.
///
/// The memory event and the context-assembled event are part of the session
/// log, not the fact being written: the fact is already durable when the first
/// is attempted, and the context snapshot has already been assembled when the
/// second is. Failing the caller would report an operation that succeeded as
/// one that did not. Discarding the error, as `try?` did, left the engine
/// claiming the file matched its state when a record was missing from it.
/// The chosen policy is: best-effort, recorded on `journalFailure`, never with
/// record contents.
@Suite struct JournalSessionEventFailureTests {
    private func startedEngine() async throws -> (engine: ContinuityEngine, task: ContinuityTask) {
        let engine = ContinuityEngine()
        try await engine.start()
        let task = try await engine.createTask(title: "session events")
        return (engine, task)
    }

    @Test func anUnknownSessionAtContextAssemblyIsReported() async throws {
        let (engine, task) = try await startedEngine()
        let known = try await engine.beginSession(taskID: task.id)
        try await engine.remember(
            sessionID: known.id, namespace: "n", key: "k", value: "SENTINEL-VALUE")

        // An id that is not a session of this engine: the event cannot be
        // recorded, and the call that produced the snapshot still answers.
        let snapshot = try await engine.assembleContext(
            taskID: task.id, sessionID: UUID())
        #expect(!snapshot.renderedContext.isEmpty)
        let failure = try #require(await engine.journalFailure)
        #expect(failure.contains("no such session"), "unexpected failure: \(failure)")
        #expect(
            !failure.contains("SENTINEL-VALUE"),
            "the failure quoted a record's contents: \(failure)")
        await engine.shutDown()
    }

    @Test func aSuccessfulSessionEventLeavesNoFailure() async throws {
        let (engine, task) = try await startedEngine()
        let session = try await engine.beginSession(taskID: task.id)
        _ = try await engine.assembleContext(taskID: task.id, sessionID: session.id)
        try await engine.remember(
            sessionID: session.id, namespace: "n", key: "k", value: "v")
        #expect(await engine.journalFailure == nil)
        let events = await engine.events(sessionID: session.id)
        #expect(events.contains { $0.kind == .contextAssembled })
        #expect(events.contains { $0.kind == .memoryWritten })
        await engine.shutDown()
    }

    /// The memory event's own path, with the session pruned while the fact is
    /// being journaled: the lookup at the top of `remember` succeeded, the
    /// append that makes the fact durable is held in the filesystem seam, and
    /// the prune runs before the event is recorded. The write must still
    /// succeed, and the event it could not record must be reported.
    @Test func aSessionPrunedDuringTheWriteIsReportedNotFatal() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("journal-session-event-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let fileSystem = JournalFaultFileSystem()
        let journal = try FileJournal(
            url: url, fileSystem: fileSystem, idleDelay: .seconds(60))
        let engine = ContinuityEngine(journal: journal)
        try await engine.start()
        let task = try await engine.createTask(title: "pruned")
        let session = try await engine.beginSession(taskID: task.id)

        let entered = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        fileSystem.holdWrite(entered: entered, release: release)
        let writing = Task {
            try await engine.remember(
                sessionID: session.id, namespace: "n", key: "k", value: "kept")
        }
        #expect(await waitForSignal(entered), "the fact's append never reached the drive")
        // The engine is suspended inside `memory.write`'s observer, so this is
        // admitted now.
        await engine.pruneSessions(taskID: task.id, keeping: 0)
        release.signal()
        let result = try await writing.value
        #expect(result.item.value == "kept")
        #expect(
            await engine.journalFailure != nil,
            "a memory event the log could not take left no trace")
        #expect(
            await engine.recall(taskID: task.id, namespace: "n", key: "k")?.value == "kept")
        await engine.shutDown()
    }

    @Test func aSessionThatVanishesDoesNotUndoTheFact() async throws {
        let (engine, task) = try await startedEngine()
        let session = try await engine.beginSession(taskID: task.id)
        let result = try await engine.remember(
            sessionID: session.id, namespace: "n", key: "before", value: "kept")
        #expect(result.item.value == "kept")
        #expect(await engine.journalFailure == nil)

        // The end of a session is not a prune, and a pruned session is the one
        // the memory event can meet. Exercise the same reporting path the
        // memory event uses: a context event for a session the log no longer
        // knows.
        await engine.pruneSessions(taskID: task.id, keeping: 0)
        _ = try await engine.assembleContext(
            taskID: task.id, sessionID: session.id)
        #expect(
            await engine.journalFailure != nil,
            "an event the log could not take left no trace")
        #expect(
            await engine.recall(taskID: task.id, namespace: "n", key: "before")?.value == "kept")
        await engine.shutDown()
    }
}
