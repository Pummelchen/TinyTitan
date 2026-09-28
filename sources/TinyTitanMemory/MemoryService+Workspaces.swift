import ContinuityCore
import Foundation

// Workspace resolution, the residency budget and engine construction.
//
// Split out of `MemoryService.swift` (2026-09-28) under the 500-line-per-file
// rule (Task 8 of the cleanup runbook) as pure code motion. Members whose
// callers stay behind widened from `private` to internal.
extension MemoryService {

    /// The engine, store and journal for a scope, built on first use.
    ///
    /// Deferred rather than done in `init` because opening a journal is I/O
    /// that can fail, and an initializer cannot report that to the caller who
    /// will actually be affected by it.
    func workspace(for scope: MemoryScope) async -> Workspace? {
        if let existing = workspaces[scope] {
            lastUsed[scope] = Date()
            return existing
        }
        guard configuration.isEnabled else { return nil }

        if let injectedStore {
            let workspace = Workspace(
                store: injectedStore, journal: injectedJournal,
                engine: nil, persists: true)
            workspaces[scope] = workspace
            return workspace
        }

        let (engine, persists) = Self.makeEngine(
            configuration: configuration,
            scope: scope, log: log)
        do {
            try await engine.start()
        } catch {
            isDegraded = true
            log(.degraded(operation: "start", detail: "\(error)"))
        }
        let store = ContinuityStore(engine: engine, limits: configuration.limits)
        var journal: (any SessionJournal)?
        if let injectedJournal {
            journal = injectedJournal
        } else if configuration.journalEnabled {
            journal = ContinuityJournalStore(
                engine: engine, store: store,
                limits: configuration.journalLimits)
        }
        let workspace = Workspace(
            store: store, journal: journal, engine: engine,
            persists: persists)
        workspaces[scope] = workspace
        lastUsed[scope] = Date()
        await enforceResidencyBudget(keeping: scope)
        // A new project file may be the one that pushes the count past the
        // cap; the sweep skips every workspace this process holds open.
        await sweepStaleWorkspaces()
        return workspace
    }

    /// Keeps the whole subsystem inside the ceiling, when one is set.
    ///
    /// A ceiling is a total, not a per-workspace allowance: per-workspace
    /// limits alone would multiply it by the number of workspaces a session
    /// has touched. With no ceiling, the default, this does nothing.
    ///
    /// Over the ceiling, the least recently used workspace is closed. Nothing
    /// is lost: everything it held is in its journal, and touching that
    /// workspace again replays it. The workspace in use is never closed.
    func enforceResidencyBudget(keeping scope: MemoryScope) async {
        guard let ceiling = configuration.storage.maximumMemoryBytes, ceiling > 0 else { return }
        while workspaces.count > 1, await residentBytes() > ceiling {
            let candidates =
                lastUsed
                .filter { $0.key != scope && workspaces[$0.key] != nil }
                .sorted { $0.value < $1.value }
            guard let oldest = candidates.first?.key else { return }
            await workspaces[oldest]?.engine?.shutDown()
            workspaces[oldest] = nil
            lastUsed[oldest] = nil
            // Reopening replays the file into a fresh engine, so a failure
            // there later is a new one and worth its own line.
            reportedJournalFailures.remove(oldest)
            log(
                .degraded(
                    operation: "residency",
                    detail: "closed workspace \(oldest.workspace) to stay inside "
                        + "\(ceiling >> 20) MiB"))
        }
    }

    /// Bytes memory is holding in this process, across every open workspace.
    public func residentBytes() async -> Int {
        var total = 0
        for workspace in workspaces.values {
            total += await workspace.engine?.residentBytes() ?? 0
        }
        return total
    }

    /// Builds the engine, with a journal file when one can be opened.
    ///
    /// A directory that cannot be written is not fatal: the engine still runs
    /// in memory for the session. It is logged, and `isDurable` reports false,
    /// so the prompt tells the model its writes will not outlive the session
    /// rather than letting it assume they will.
    ///
    /// - Returns: the engine, and whether it is actually writing to a file.
    ///   The flag is not cosmetic: without it a session whose journal could
    ///   not be opened would tell the model its writes persist, which is the
    ///   one thing memory must never get wrong.
    static func makeEngine(
        configuration: MemoryConfiguration,
        scope: MemoryScope,
        log: @Sendable (MemoryLogEvent) -> Void
    ) -> (engine: ContinuityEngine, persists: Bool) {
        let budget = configuration.storage.budget
        let limits = ContinuityCore.MemoryLimits(
            maxValueBytes: configuration.limits.maximumValueBytes,
            maxBytesPerTask: budget.factBytes)
        let engineConfiguration = ContinuityConfiguration(
            memoryLimits: limits,
            sessionLogOptions: SessionLogOptions(maxBytesPerTask: budget.logBytes),
            journalsSessionContent: configuration.journalEnabled)
        do {
            let journal = try FileJournal(
                url: configuration.storage.journalURL(for: scope),
                synchronizesEveryWrite: configuration.storage.synchronizesEveryWrite)
            return (ContinuityEngine(configuration: engineConfiguration, journal: journal), true)
        } catch {
            // A journal held by another server is the expected case here, not
            // a broken install. Either way the session runs without
            // persistence and says so rather than writing into a file someone
            // else is also writing.
            log(.degraded(operation: "openJournal", detail: "\(error)"))
            return (ContinuityEngine(configuration: engineConfiguration), false)
        }
    }

    /// Records a completed turn. Content is filtered to substance here, so no
    /// caller can accidentally journal a tool result or a file dump.
}
