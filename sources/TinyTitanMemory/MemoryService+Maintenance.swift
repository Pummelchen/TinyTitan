import ContinuityCore
import Foundation

// Lock checks, stale-session expiry, journal access and shutdown.
//
// Split out of `MemoryService.swift` (2026-09-28) under the 500-line-per-file
// rule (Task 8 of the cleanup runbook) as pure code motion. Members whose
// callers stay behind widened from `private` to internal.
extension MemoryService {

    /// Whether another process holds this project's workspace lock.
    ///
    /// `FileJournal` takes `flock(LOCK_EX | LOCK_NB)` on `<journal>.lock`, so the
    /// probe is the same call: it succeeds only when nobody holds the workspace.
    /// Deleting a file another process is appending to takes its `.lock` with it
    /// -- the file that process's `flock` is attached to -- and leaves it
    /// writing to a deleted inode, which its next compaction then rewrites into
    /// nothing. The retention pass already behaves this way by opening the file
    /// through the ordinary engine; this is the same rule for the cap.
    ///
    /// Everything unreadable, and every unexpected `flock` failure, counts as
    /// held: skipping a deletable file costs disk, deleting a live one costs
    /// data.
    static func isLockHeld(at journalURL: URL) -> Bool {
        let descriptor = open(
            journalURL.appendingPathExtension("lock").path,
            O_RDWR | O_NOFOLLOW | O_CLOEXEC)
        if descriptor < 0 {
            // No lock file at all means no journal has opened this workspace, so
            // there is nothing that could be holding it. Any other errno is not
            // ours to read in favour of deleting.
            //
            // O_NOFOLLOW is part of that "any other errno", not an extra: a link
            // planted at the lock path would otherwise be locked *instead of* the
            // real anchor, this would answer "not held", and the retention pass
            // would delete a journal another process is appending to -- exactly
            // the data loss the comment above exists to prevent. ELOOP is a
            // failure to probe, so it counts as held.
            return errno != ENOENT
        }
        defer { close(descriptor) }
        if flock(descriptor, LOCK_EX | LOCK_NB) == 0 {
            _ = flock(descriptor, LOCK_UN)
            return false
        }
        return true
    }

    /// Every project file under the directory with its last-write time.
    /// Synchronous on purpose: a directory enumerator cannot be iterated
    /// from an async context.
    static func projectFiles(under directory: URL) -> [(url: URL, modified: Date)] {
        guard
            let walker = FileManager.default.enumerator(
                at: directory, includingPropertiesForKeys: [.contentModificationDateKey],
                options: [.skipsHiddenFiles])
        else { return [] }
        var files: [(url: URL, modified: Date)] = []
        for case let url as URL in walker where url.pathExtension == "ndjson" {
            let modified =
                (try? url.resourceValues(forKeys: [.contentModificationDateKey]))?
                .contentModificationDate ?? .distantPast
            files.append((url, modified))
        }
        return files
    }

    /// Rewrites a project file as a checkpoint of its facts alone.
    ///
    /// Opens it with the ordinary engine, which takes the workspace lock, so
    /// a file another server holds is left alone. Every session is pruned
    /// and the journal compacted; the facts, their history and the task
    /// survive.
    ///
    /// - Returns: `nil` when the file was rewritten, otherwise the reason it
    ///   was not, for the caller to log. Every step's failure is a return,
    ///   including the compaction: wrapping it in `try?` and answering `true`
    ///   anyway reported a file whose session log was still on disk in full as
    ///   expired, which is the one outcome the retention pass must not claim —
    ///   the disk it exists to reclaim keeps growing and the log says nothing
    ///   about why.
    static func expireSessionLog(at url: URL) async -> String? {
        let journal: FileJournal
        do {
            journal = try FileJournal(url: url)
        } catch {
            return "could not be opened (\(error))"
        }
        let engine = ContinuityEngine(journal: journal)
        do {
            try await engine.start()
        } catch {
            await engine.shutDown()
            return "could not be replayed (\(error))"
        }
        for task in await engine.tasks() {
            await engine.pruneSessions(taskID: task.id, keeping: 0)
        }
        do {
            try await engine.compactJournal()
        } catch {
            await engine.shutDown()
            return "compaction failed (\(error))"
        }
        await engine.shutDown()
        return nil
    }

    /// Close every workspace, flushing and releasing the workspace locks.
    ///
    /// A workspace's journal holds an exclusive lock for as long as it is
    /// open, so a process that is finished with a workspace has to say so.
    /// Leaving it to deallocation would make the moment another server can
    /// take over depend on when ARC happens to release an actor.
    public func shutDown() async {
        // The background sweep goes first: it drives the same engine, and a
        // judgement must not be in flight while the weights are released.
        await retrievalHinter?.shutdown()
        for workspace in workspaces.values {
            await workspace.engine?.shutDown()
        }
        workspaces.removeAll()
        lastUsed.removeAll()
        reportedJournalFailures.removeAll()
        reportedExpiries.removeAll()
        reportedSweeps.removeAll()
        reportedReadFailures.removeAll()
        // The side-engine is a second resident model, so it is released on the
        // same shutdown that releases the stores rather than at process exit.
        await sideEngine?.shutdown()
    }

    /// The journal, for a caller that wants to read it back. Never used to
    /// build a prompt.
    public func journalStore(for scope: MemoryScope? = nil) async -> (any SessionJournal)? {
        guard let resolved = scope ?? configuration.scope() else { return nil }
        return await workspace(for: resolved)?.journal
    }

    public var isEnabled: Bool { configuration.isEnabled }

    /// Whether writes reach durable storage in a scope. False once a durable
    /// operation has failed, false when the journal could not be opened at
    /// all, and false once the journal has refused a write.
    public func isDurable(in scope: MemoryScope) async -> Bool {
        guard !isDegraded, let workspace = await workspace(for: scope),
            workspace.persists
        else { return false }
        return !(await journalFailed(in: scope))
    }

    /// Whether a workspace's journal has refused a write, logged the first
    /// time it is seen.
    ///
    /// Read from the engine rather than inferred from a tool result: a turn's
    /// prompt and reply are journaled on a path where no caller sees the
    /// write fail.
    func journalFailed(in scope: MemoryScope) async -> Bool {
        guard let store = workspaces[scope]?.store as? ContinuityStore,
            let failure = await store.journalFailure
        else { return false }
        if reportedJournalFailures.insert(scope).inserted {
            log(.degraded(operation: "journal", detail: failure))
        }
        return true
    }

    /// A memory *read* that threw, said once per operation and workspace.
    ///
    /// This is `journalFailed(in:)` for the read side, and it exists because
    /// an empty answer and a failed read are not the same answer. A caller
    /// that swallows the throw into `[]` or `.empty` reports "this session has
    /// no memories", "this workspace has no facts on file", "there were no new
    /// turns" — each of which sends someone off to act on evidence that was
    /// never gathered. Sites that still have to produce a value call this with
    /// what they are about to return, so the log says which of the two it was;
    /// sites that can propagate instead do, and this is for the rest.
    ///
    /// Deliberately does not set `isDegraded`: that flag means *writes are not
    /// reaching storage* and makes the prompt say so. A read that failed says
    /// nothing about a write, and claiming it did would tell the model its
    /// memory is not being saved when it is.
    func reportReadFailure(_ operation: String, _ error: Error, in scope: MemoryScope) {
        guard reportedReadFailures.insert("\(operation)@\(scope.workspace)").inserted else {
            return
        }
        log(
            .degraded(
                operation: operation,
                detail: "read failed in workspace '\(scope.workspace)', "
                    + "answering as unknown: \(error)"))
    }

    /// Whether the configuration's own scope is persisting.
    public var isDurable: Bool {
        get async {
            guard let scope = configuration.scope() else { return false }
            return await isDurable(in: scope)
        }
    }

    /// Starts a session and returns what the engine needs to install.
    ///
    /// A failure here degrades rather than propagates: the session continues
    /// with local memory when that is allowed, and with none when it is not.
    /// - Parameter tag: what the session is about, when the caller could
    ///   tell. Recorded on the session, shown in the log; not a scope.
}
