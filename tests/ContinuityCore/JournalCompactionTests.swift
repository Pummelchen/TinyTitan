import Foundation
import Testing

@testable import ContinuityCore

/// A journal that pauses in the middle of a compaction and records what
/// happened to the file while it was there. The pause is a continuation the
/// test resumes, not a sleep.
private actor PausingJournal: ContinuityJournal {
    private(set) var records: [JournalRecord] = []
    private(set) var appendsDuringCompact = 0
    private var compacting = false
    private var enteredContinuations: [CheckedContinuation<Void, Never>] = []
    private var finishContinuation: CheckedContinuation<Void, Never>?
    private var appendSignal: DispatchSemaphore?

    func signalAppends(_ semaphore: DispatchSemaphore?) {
        appendSignal = semaphore
    }

    func append(_ record: JournalRecord) async throws {
        if compacting { appendsDuringCompact += 1 }
        records.append(record)
        appendSignal?.signal()
    }

    func replay() async throws -> [JournalRecord] { records }

    func compact(sessionLog: SessionLogSnapshot, memory: MemorySnapshot) async throws {
        compacting = true
        let entered = enteredContinuations
        enteredContinuations.removeAll()
        for continuation in entered { continuation.resume() }
        await withCheckedContinuation { finishContinuation = $0 }
        // The replacement: everything in the file becomes the checkpoint,
        // which is what a record appended during the pause would be lost to.
        records = [.checkpoint(sessionLog, memory)]
        compacting = false
    }

    func truncate() async throws { records = [] }

    func waitUntilCompacting() async {
        if compacting { return }
        await withCheckedContinuation { enteredContinuations.append($0) }
    }

    func finishCompact() {
        finishContinuation?.resume()
        finishContinuation = nil
    }
}

/// Appending while the file is being replaced.
///
/// The journal's snapshot is taken before the replacement is written, so a
/// record appended during either window — after the snapshot and before the
/// rename — is not in the checkpoint and would disappear with the old file.
/// An actor does not serialize across an `await`; something has to.
@Suite struct JournalCompactionTests {
    private func temporaryDirectory() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("journal-compaction-\(UUID().uuidString)")
    }

    private func task(_ title: String) -> JournalRecord {
        .task(ContinuityTask(title: title))
    }

    /// The journal-level half: an append asked for while a replacement is in
    /// flight is admitted only once the replacement has finished, and lands in
    /// the file that survives.
    @Test func anAppendWaitsForACompactionToFinish() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let fileSystem = JournalFaultFileSystem()
        let entered = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        fileSystem.holdRename(entered: entered, release: release)
        let journal = try FileJournal(
            url: url, fileSystem: fileSystem, idleDelay: .seconds(60))
        try await journal.append(task("before"))
        try await journal.sync()

        let compacting = Task {
            try await journal.compact(
                sessionLog: SessionLogSnapshot(), memory: MemorySnapshot())
        }
        #expect(await waitForSignal(entered), "the compaction never reached the rename")

        let appending = Task { try await journal.append(task("during")) }
        // Waiter registration is observable, so this does not sleep and hope:
        // the yield loop only lets the append task run far enough to be
        // counted.
        var waited = false
        for _ in 0..<100_000 {
            if await journal.rewriteWaiterCount == 1 {
                waited = true
                break
            }
            await Task.yield()
        }
        #expect(waited, "the append was admitted while the file was being replaced")

        release.signal()
        try await compacting.value
        try await appending.value

        let replayed = try await journal.replay()
        #expect(replayed.count == 2, "the record appended during the replace was lost")
        guard case .task(let appended)? = replayed.last else {
            Issue.record("the appended record is not last")
            try await journal.shutDown()
            return
        }
        #expect(appended.title == "during")
        try await journal.shutDown()
        #expect(try FileJournal.read(contentsOf: url).count == 2)
    }

    /// The behavioural half, on the same pause: the record is written while
    /// the rename is held, i.e. exactly the interleaving the guard rules out.
    /// The wait only establishes that interleaving on the broken code; what
    /// the test asserts is the bytes that survive.
    @Test func aRecordWrittenDuringTheReplaceIsNotLost() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let fileSystem = JournalFaultFileSystem()
        let entered = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        let wrote = DispatchSemaphore(value: 0)
        fileSystem.holdRename(entered: entered, release: release)
        let journal = try FileJournal(
            url: url, fileSystem: fileSystem, idleDelay: .seconds(60))
        try await journal.append(task("before"))
        try await journal.sync()
        // Armed after setup, so the next write signalled is the one under
        // test.
        fileSystem.signalWrites(at: url.path, wrote)

        let compacting = Task {
            try await journal.compact(
                sessionLog: SessionLogSnapshot(), memory: MemorySnapshot())
        }
        #expect(await waitForSignal(entered), "the compaction never reached the rename")
        let appending = Task { try await journal.append(task("during")) }
        // Bounded: on the broken code the append writes now; on the fixed code
        // it is waiting for the gate and this expires.
        let wroteDuring = await waitForSignal(wrote, timeout: .now() + 1)
        #expect(!wroteDuring, "the append was written while the file was being replaced")
        release.signal()
        try await compacting.value
        try await appending.value

        let replayed = try await journal.replay()
        #expect(replayed.count == 2, "the record written during the replace was lost")
        try await journal.shutDown()
        #expect(try FileJournal.read(contentsOf: url).count == 2)
    }

    /// Two compactions, one after the other: the second replaces the first's
    /// checkpoint rather than racing it, and neither deadlocks the other.
    @Test func competingCompactionsSerialise() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let fileSystem = JournalFaultFileSystem()
        let journal = try FileJournal(
            url: url, fileSystem: fileSystem, idleDelay: .seconds(60))
        try await journal.append(task("before"))
        let first = Task {
            try await journal.compact(
                sessionLog: SessionLogSnapshot(), memory: MemorySnapshot())
        }
        let second = Task {
            try await journal.compact(
                sessionLog: SessionLogSnapshot(), memory: MemorySnapshot())
        }
        try await first.value
        try await second.value
        #expect(fileSystem.renames == 2)
        #expect(try await journal.replay().count == 1)
        try await journal.append(task("after"))
        #expect(try await journal.replay().count == 2)
        try await journal.shutDown()
        #expect(try FileJournal.read(contentsOf: url).count == 2)
    }

    /// Shutdown waits for a replacement in flight instead of closing the
    /// descriptor under it, and still releases the workspace afterwards.
    @Test func shutdownWaitsForACompactionInFlight() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let fileSystem = JournalFaultFileSystem()
        let entered = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        fileSystem.holdRename(entered: entered, release: release)
        let journal = try FileJournal(
            url: url, fileSystem: fileSystem, idleDelay: .seconds(60))
        try await journal.append(task("before"))
        let compacting = Task {
            try await journal.compact(
                sessionLog: SessionLogSnapshot(), memory: MemorySnapshot())
        }
        #expect(await waitForSignal(entered), "the compaction never reached the rename")
        let closing = Task { try await journal.shutDown() }
        release.signal()
        try await compacting.value
        try await closing.value
        // The workspace is free, and the checkpoint is what it names.
        let next = try FileJournal(url: url, fileSystem: JournalFaultFileSystem())
        #expect(try await next.replay().count == 1)
        try await next.shutDown()
    }

    /// The engine-level half. The journal can only preserve what is written
    /// after it starts replacing; a record appended before that, from a
    /// snapshot the engine took earlier, is outside its reach. The engine
    /// therefore holds its own admission across snapshot capture *and*
    /// replacement.
    @Test func aMemoryWriteDuringCompactionSurvives() async throws {
        let appendSignal = DispatchSemaphore(value: 0)
        let journal = PausingJournal()
        let engine = ContinuityEngine(journal: journal)
        try await engine.start()
        let task = try await engine.createTask(title: "work")
        let session = try await engine.beginSession(taskID: task.id)
        // Armed after setup, so the next append signalled is the one under
        // test.
        await journal.signalAppends(appendSignal)

        let compacting = Task { try await engine.compactJournal() }
        await journal.waitUntilCompacting()

        let writing = Task {
            _ = try await engine.remember(
                sessionID: session.id, namespace: "n", key: "k", value: "survives")
        }
        // Bounded: the write proceeds during the pause on the broken code and
        // waits at the engine gate on the fixed one.
        let wroteDuring = await waitForSignal(appendSignal, timeout: .now() + 1)
        await journal.finishCompact()
        try await compacting.value
        try await writing.value

        #expect(!wroteDuring, "a fact was appended while the checkpoint was being replaced")
        let during = await journal.appendsDuringCompact
        #expect(during == 0, "a fact reached the journal during the replacement")
        let records = await journal.records
        #expect(
            records.contains { record in
                if case .memory(let item) = record { return item.value == "survives" }
                return false
            },
            "the fact outlived the compaction only in RAM")
        #expect(
            records.contains { record in
                if case .checkpoint = record { return true }
                return false
            })
        await engine.shutDown()
    }
}
