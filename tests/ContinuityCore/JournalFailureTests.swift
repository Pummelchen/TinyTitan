import Foundation
import Testing

@testable import ContinuityCore

/// A journal that refuses writes on demand, standing in for a full disk, an
/// I/O error or a descriptor closed underneath it.
private actor RefusingJournal: ContinuityJournal {
    private(set) var records: [JournalRecord] = []
    private var refusing = false

    func refuse(_ value: Bool) { refusing = value }

    func append(_ record: JournalRecord) async throws {
        try check()
        records.append(record)
    }

    func replay() async throws -> [JournalRecord] { records }

    func compact(sessionLog: SessionLogSnapshot, memory: MemorySnapshot) async throws {
        try check()
        records = [.checkpoint(sessionLog, memory)]
    }

    func truncate() async throws { records = [] }

    private func check() throws {
        guard !refusing else {
            throw JournalError.writeFailed(URL(fileURLWithPath: "/journal.ndjson"), errno: ENOSPC)
        }
    }
}

/// A journal whose appends land and whose compaction does not: the disk fills
/// between one write and the next, or the directory refuses the rename that
/// replaces the file. Separated from the journal above because refusing
/// everything cannot tell the two failures apart, and they owe the caller
/// different answers — a refused append loses a record, a refused compaction
/// loses nothing and only stops the file from collapsing.
private actor CompactionRefusingJournal: ContinuityJournal {
    private(set) var records: [JournalRecord] = []
    private var refusing = false

    func refuseCompaction(_ value: Bool) { refusing = value }

    func append(_ record: JournalRecord) async throws { records.append(record) }

    func replay() async throws -> [JournalRecord] { records }

    func compact(sessionLog: SessionLogSnapshot, memory: MemorySnapshot) async throws {
        guard !refusing else {
            throw JournalError.writeFailed(URL(fileURLWithPath: "/journal.ndjson"), errno: ENOSPC)
        }
        records = [.checkpoint(sessionLog, memory)]
    }

    func truncate() async throws { records = [] }
}

/// RAM is the source of truth during a run, so a refused journal write
/// never loses the change in-process. What these pin down is who gets told:
/// the writer of a fact, always; the author of a session event, never.
@Suite struct JournalFailureTests {
    private func started(_ journal: RefusingJournal) async throws
        -> (engine: ContinuityEngine, task: ContinuityTask, session: Session)
    {
        let engine = ContinuityEngine(journal: journal)
        try await engine.start()
        let task = try await engine.createTask(title: "Disk")
        let session = try await engine.beginSession(taskID: task.id)
        return (engine, task, session)
    }

    @Test func aMemoryWriteTheJournalRefusesFailsItsCaller() async throws {
        let journal = RefusingJournal()
        let (engine, task, session) = try await started(journal)
        await journal.refuse(true)

        do {
            try await engine.remember(sessionID: session.id, namespace: "n", key: "k", value: "v")
            Issue.record("a write the journal refused was reported as saved")
        } catch ContinuityError.notPersisted {}

        // Kept for the session, and the engine no longer claims the file
        // matches it.
        #expect(await engine.recall(taskID: task.id, namespace: "n", key: "k")?.value == "v")
        #expect(await engine.journalFailure != nil)

        // A later write landing does not bring back the one that did not.
        await journal.refuse(false)
        try await engine.remember(sessionID: session.id, namespace: "n", key: "k2", value: "v2")
        #expect(await engine.journalFailure != nil)
    }

    @Test func anArchiveTheJournalRefusesFailsItsCaller() async throws {
        let journal = RefusingJournal()
        let (engine, task, session) = try await started(journal)
        try await engine.remember(sessionID: session.id, namespace: "n", key: "k", value: "v")
        await journal.refuse(true)

        // An archive that does not reach the file is a delete that undoes
        // itself on restart.
        await #expect(throws: ContinuityError.self) {
            try await engine.archive(taskID: task.id, namespace: "n", key: "k")
        }
        #expect(await engine.journalFailure != nil)
    }

    @Test func aSessionEventTheJournalRefusesDoesNotFailTheTurn() async throws {
        let journal = RefusingJournal()
        let (engine, task, session) = try await started(journal)
        await journal.refuse(true)

        try await engine.recordUserPrompt(sessionID: session.id, text: "hello")
        try await engine.recordAssistantResponse(sessionID: session.id, text: "hi")

        #expect(await engine.turns(taskID: task.id).count == 1)
        #expect(await engine.journalFailure != nil)
    }

    @Test func writesThatLandLeaveNoFailure() async throws {
        let journal = RefusingJournal()
        let (engine, _, session) = try await started(journal)
        try await engine.remember(sessionID: session.id, namespace: "n", key: "k", value: "v")
        try await engine.recordUserPrompt(sessionID: session.id, text: "hello")

        #expect(await engine.journalFailure == nil)
        let records = await journal.records
        #expect(
            records.contains { record in
                if case .memory(let item) = record { return item.value == "v" }
                return false
            })
    }

    /// Automatic compaction fires from `countRecord()`, after the write that
    /// tripped it has already landed, and its error goes through `try?`. So a
    /// journal that can no longer collapse keeps every prompt and reply on the
    /// disk forever while `journalFailure` — the engine's only channel for "the
    /// file is not what I think" — stays nil, and the store above it goes on
    /// reporting itself healthy. The failure needs a trace of its own, and not
    /// a durability flip: these writes did reach the file.
    @Test func aCompactionTheJournalRefusesLeavesItsOwnTrace() async throws {
        let journal = CompactionRefusingJournal()
        let engine = ContinuityEngine(
            configuration: ContinuityConfiguration(compactionThreshold: 2), journal: journal)
        try await engine.start()
        let task = try await engine.createTask(title: "Disk")
        let session = try await engine.beginSession(taskID: task.id)
        await journal.refuseCompaction(true)

        for index in 0..<4 {
            try await engine.remember(
                sessionID: session.id, namespace: "n", key: "k\(index)", value: "v\(index)")
        }

        #expect(await engine.compactionFailure != nil, "a swallowed compaction left no trace")
        // The refused checkpoint is not a refused record: every fact is still
        // in the file, so durability stands.
        #expect(await engine.journalFailure == nil)
        let written = await journal.records
        let memories = written.filter { record in
            if case .memory = record { return true }
            return false
        }
        #expect(memories.count == 4, "the refused compaction cost a write: \(written.count)")
        #expect(
            await engine.recall(taskID: task.id, namespace: "n", key: "k3")?.value == "v3")

        // And the trace has to clear when the cause does. A sticky warning would
        // outlive a journal that has since collapsed, and say the file is still
        // growing when it is not.
        await journal.refuseCompaction(false)
        try await engine.remember(
            sessionID: session.id, namespace: "n", key: "k9", value: "v9")
        #expect(await engine.compactionFailure == nil, "the retry left the warning standing")
        // The shape of the file, not its size: one write is a fact and the
        // session event that says it was written, so a collapsed journal is
        // a checkpoint plus whatever landed after it.
        let collapsed = await journal.records
        let stillLoose = collapsed.filter { record in
            if case .memory = record { return true }
            return false
        }
        #expect(stillLoose.isEmpty, "the retry left \(stillLoose.count) facts uncollapsed")
        #expect(
            collapsed.contains { record in
                if case .checkpoint = record { return true }
                return false
            },
            "the retry did not write a checkpoint")
    }
}
