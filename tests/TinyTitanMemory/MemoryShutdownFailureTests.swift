import ContinuityCore
import Foundation
import Synchronization
import Testing

@testable import TinyTitanMemory

/// A journal that refuses every append. Used to put a durability failure on an
/// engine without the service being the caller that caused it — the shape of a
/// failure that happens behind the service's back.
private actor RefusingJournal: ContinuityJournal {
    func append(_ record: JournalRecord) async throws {
        throw JournalError.writeFailed(
            URL(fileURLWithPath: "/journal.ndjson"), errno: ENOSPC)
    }

    func replay() async throws -> [JournalRecord] { [] }
    func compact(sessionLog: SessionLogSnapshot, memory: MemorySnapshot) async throws {}
    func truncate() async throws {}
}

/// Collects log events from the service's `@Sendable` callback.
private final class LogCollector: Sendable {
    private let events = Mutex<[MemoryLogEvent]>([])

    func append(_ event: MemoryLogEvent) {
        events.withLock { $0.append(event) }
    }

    func messages() -> [String] {
        events.withLock { $0.map(\.message) }
    }
}

/// The service is the last place a journal failure can be said out loud:
/// shutdown is a flush point, and after it the engine is gone.
@Suite struct MemoryShutdownFailureTests {
    @Test func shutdownReportsAJournalFailureNobodySaw() async throws {
        var configuration = MemoryConfiguration()
        configuration.isEnabled = true
        configuration.workspace = "repo-a"
        configuration.user = "local"
        configuration.toolSurface = .full
        let engine = ContinuityEngine(journal: RefusingJournal())
        try await engine.start()
        let store = ContinuityStore(engine: engine)
        let events = LogCollector()
        let service = MemoryService(
            configuration: configuration,
            durableStore: store,
            log: { events.append($0) })
        _ = try #require(await service.beginSession(id: "s1"))
        // Forces the workspace into existence, which is what shutdown walks,
        // without going through `isDurable` (which would report the failure
        // itself and leave shutdown with nothing new to say).
        _ = await service.journalStore(for: nil)
        // The injected store backs both this workspace and the shared one, so
        // both name the same engine; production gives each its own file.
        let scopes = await service.workspaces.keys.map(\.workspace).sorted()
        #expect(scopes == ["_global", "repo-a"], "workspaces: \(scopes)")
        #expect(!events.messages().contains { $0.contains("degraded during journal") })

        // A write fails on the engine, which records it; the service has not
        // been asked anything since, so the failure has not been said yet.
        await #expect(throws: JournalError.self) {
            _ = try await engine.createTask(title: "work")
        }
        #expect(await store.journalFailure != nil)
        #expect(
            !events.messages().contains { $0.contains("degraded during journal") },
            "the failure was reported before shutdown, so this test proves nothing")

        await service.shutDown()
        let journalLines = events.messages().filter { $0.contains("degraded during journal") }
        #expect(
            journalLines.count == scopes.count,
            "a durability failure present at shutdown went unsaid: \(journalLines)")
    }
}
