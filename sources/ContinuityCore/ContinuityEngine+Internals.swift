import Foundation

// The engine's internals: observer installation, journal failure handling and
// the journal record/restore path.
//
// Split out of `ContinuityEngine.swift` (2026-09-28) under the 500-line-per-
// file rule (Task 8 of the cleanup runbook) as pure code motion. The stored
// properties and helpers widened from `private` to internal because the actor
// methods that call them stay behind.
extension ContinuityEngine {

    // MARK: - Internals

    /// Complete lines the journal could not decode on the last replay, or nil
    /// when every complete line was a record.
    ///
    /// The good records around them are replayed and the engine runs; what
    /// this changes is that the file cannot be replaced until someone handles
    /// the damaged bytes, and the memory service reports the store as
    /// incomplete rather than healthy.
    ///
    /// Read from the journal rather than cached: the explicit recovery step
    /// clears the journal's damage, and a cached string would keep reporting a
    /// file that has since been handled as damaged.
    public var journalDamage: String? {
        get async {
            guard let fileJournal = journal as? FileJournal,
                let damage = await fileJournal.replayDamage
            else { return nil }
            // Only the positions: the line did not decode, so there is
            // nothing safe to quote, and a damaged line may hold a prompt.
            let lines = damage.lineNumbers.map(String.init).joined(separator: ", ")
            return "\(damage.lines.count) complete record(s) did not decode (lines \(lines))"
        }
    }

    func installObservers() async {
        let journal = self.journal
        let journalsContent = configuration.journalsSessionContent

        // A session event that fails to journal does not fail the turn that
        // produced it: the event is already in the log and the person is owed
        // their reply. It is recorded as a durability failure instead.
        await sessionLog.setObserver { [weak self] event in
            guard journalsContent || !Self.carriesContent(event) else { return }
            // A compaction must not replace the file underneath this write.
            await self?.awaitJournalRewrite()
            do {
                try await journal.append(.event(event))
            } catch {
                await self?.journalWriteFailed(error)
                return
            }
            await self?.countRecord()
        }
        // A memory mutation that fails to journal does fail its caller. The
        // value stays in RAM, but whoever wrote it would otherwise believe it
        // saved, and a fact reported as stored that ends with the process is
        // the one answer memory must never give.
        await memory.setObserver { [weak self] mutation in
            let entry: JournalRecord
            switch mutation {
            case .versioned(let version): entry = .memoryVersion(version)
            case .written(let result): entry = .memory(result.item)
            case .statusChanged(let item): entry = .memory(item)
            }
            // The fact is already in RAM; this is the write that carries it
            // to the file. It waits for any compaction to finish, because the
            // checkpoint being written was taken from a snapshot that cannot
            // contain a record not yet appended.
            await self?.awaitJournalRewrite()
            do {
                try await journal.append(entry)
            } catch {
                await self?.journalWriteFailed(error)
                throw ContinuityError.notPersisted(String(describing: error))
            }
            await self?.countRecord()
        }
    }

    /// Whether an event carries what a person or the model actually wrote, as
    /// opposed to structure and measurements.
    static func carriesContent(_ event: SessionEvent) -> Bool {
        switch event.kind {
        case .userPrompt, .assistantResponse, .assistantResponseChunk,
            .assistantResponseCompleted:
            return true
        case .sessionStarted, .sessionEnded, .assistantResponseStarted,
            .memoryWritten, .contextAssembled:
            return false
        }
    }

    func countRecord() async {
        journaledRecords += 1
        guard configuration.compactionThreshold > 0,
            journaledRecords > configuration.compactionThreshold
        else { return }
        // Every other journal call is wrapped and reported; this one went
        // through `try?`, so a journal that can no longer rewrite itself kept
        // every prompt and reply on the disk forever while the engine's only
        // failure channel stayed nil and the workspace went on reporting
        // itself healthy. The records landed, so this is not a durability
        // failure — it is a growth failure, and it gets its own trace.
        do {
            try await compactJournal()
        } catch {
            compactionFailure = String(describing: error)
        }
    }

    func record(_ entry: JournalRecord) async throws {
        await awaitJournalRewrite()
        do {
            try await journal.append(entry)
        } catch {
            journalWriteFailed(error)
            throw error
        }
        journaledRecords += 1
    }

    /// Wait until no compaction owns the snapshot.
    func awaitJournalRewrite() async {
        while rewritingJournal {
            await withCheckedContinuation { journalRewriteWaiters.append($0) }
        }
    }

    /// Take the snapshot for a compaction. Every journal append waits until
    /// `endJournalRewrite` resumes it.
    func beginJournalRewrite() async {
        await awaitJournalRewrite()
        rewritingJournal = true
    }

    func endJournalRewrite() {
        rewritingJournal = false
        let waiters = journalRewriteWaiters
        journalRewriteWaiters.removeAll()
        for waiter in waiters { waiter.resume() }
    }

    func restoreFromJournal() async throws {
        let records = try await journal.replay()
        guard !records.isEmpty else { return }
        var log = SessionLogSnapshot()
        var store = MemorySnapshot()
        var itemsByAddress: [String: MemoryItem] = [:]
        var tasksByID: [UUID: ContinuityTask] = [:]
        var sessionsByID: [UUID: Session] = [:]

        for entry in records {
            switch entry {
            case .checkpoint(let logSnapshot, let memorySnapshot):
                log = logSnapshot
                store = memorySnapshot
                tasksByID = Dictionary(uniqueKeysWithValues: logSnapshot.tasks.map { ($0.id, $0) })
                sessionsByID = Dictionary(
                    uniqueKeysWithValues:
                        logSnapshot.sessions.map { ($0.id, $0) })
                itemsByAddress = [:]
                for item in memorySnapshot.items {
                    itemsByAddress["\(item.taskID)/\(item.address)"] = item
                }
            case .task(let task):
                tasksByID[task.id] = task
            case .session(let session):
                sessionsByID[session.id] = session
            case .event(let event):
                log.events.append(event)
            case .memory(let item):
                itemsByAddress["\(item.taskID)/\(item.address)"] = item
            case .memoryVersion(let version):
                store.versions.append(version)
            }
        }
        log.tasks = Array(tasksByID.values)
        log.sessions = Array(sessionsByID.values)
        store.items = Array(itemsByAddress.values)
        await sessionLog.restore(log)
        await memory.restore(store)
        journaledRecords = records.count
    }
}
