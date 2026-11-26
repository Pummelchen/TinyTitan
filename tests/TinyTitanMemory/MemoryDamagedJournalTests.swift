import ContinuityCore
import Foundation
import Testing

@testable import TinyTitanMemory

/// A journal that replayed with complete lines missing.
///
/// The store recovers every record that decoded and keeps working; what it
/// must not do is tell the model its memory is complete, or let a compaction
/// replace the file with a checkpoint that omits the damaged lines. The file
/// lives in a scratch directory and is written with the real syscalls: this is
/// the public path, not the fault seam.
@Suite struct MemoryDamagedJournalTests {
    private func temporaryDirectory() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("memory-damage-\(UUID().uuidString)")
    }

    private func records(_ directory: URL, _ titles: [String]) async throws -> Data {
        let url = directory.appendingPathComponent("control-\(UUID().uuidString).ndjson")
        let journal = try FileJournal(url: url)
        for title in titles {
            try await journal.append(.task(ContinuityTask(title: title)))
        }
        try await journal.shutDown()
        return try Data(contentsOf: url)
    }

    @Test func aDamagedReplayIsReportedAndTheFileIsNotReplaced() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let first = try await records(directory, ["before"])
        let second = try await records(directory, ["after"])
        var contents = first
        contents.append(Data(#"{"somethingElse":{"version":9}}"#.utf8))
        contents.append(0x0A)
        contents.append(second)
        try contents.write(to: url)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o600], ofItemAtPath: url.path)

        let journal = try FileJournal(url: url)
        let engine = ContinuityEngine(journal: journal)
        try await engine.start()
        let store = ContinuityStore(engine: engine)
        var configuration = MemoryConfiguration()
        configuration.isEnabled = true
        configuration.workspace = "repo-a"
        configuration.user = "local"
        configuration.toolSurface = .full
        let events = TestLogCollector()
        let service = MemoryService(
            configuration: configuration,
            durableStore: store,
            log: { events.append($0) })

        #expect(await store.journalDamage != nil)
        #expect(await service.isDurable == false)
        let lines = events.messages().filter { $0.contains("degraded during journal-damage") }
        #expect(lines.count == 1, "a damaged replay was not said: \(lines)")

        // The good records are there, and the file is not rewritten over.
        let before = try Data(contentsOf: url)
        await #expect(throws: JournalError.self) {
            try await engine.compactJournal()
        }
        #expect(try Data(contentsOf: url) == before)

        // Explicit recovery is what allows the rewrite, and the warning clears
        // with the condition.
        await journal.discardDamagedRecords()
        try await engine.compactJournal()
        #expect(try FileJournal.read(contentsOf: url).count == 1)
        #expect(await service.isDurable)
        await service.shutDown()
    }
}
