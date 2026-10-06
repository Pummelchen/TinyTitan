import ContinuityCore
import Foundation
import Testing

@testable import TinyTitanMemory

/// Retention's expiry pass: what it claims, and what the log says when it
/// cannot do it.
///
/// Before AUD-135 the compaction step was wrapped in `try?` and the function
/// answered `true` anyway, so a file whose session log was still on disk in
/// full was reported as expired. The disk it exists to reclaim kept growing
/// and the log said nothing about why. These tests hold both halves of the
/// contract: the reason comes back when a step fails, and nothing is reported
/// as expired that is still there.
@Suite struct ExpireSessionLogTests {

    private func makeDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("tinytitan-expire-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func scope(_ workspace: String = "repo-a") throws -> MemoryScope {
        try MemoryScope(namespace: "tinytitan", user: "local", workspace: workspace)
    }

    /// A real project file: one fact and one journaled turn, closed the way a
    /// server closes one at exit so the next opener can take the workspace
    /// lock.
    private func writtenJournal(
        in directory: URL,
        named name: String = "repo-a.ndjson"
    ) async throws -> URL {
        let url = directory.appendingPathComponent("tinytitan/local", isDirectory: true)
            .appendingPathComponent(name)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let journal = try FileJournal(url: url)
        let engine = ContinuityEngine(journal: journal)
        try await engine.start()
        let store = ContinuityStore(engine: engine)
        let scope = try scope()
        _ = try await store.sessionInit(MemorySession(id: "s1"), in: scope)
        try await store.set(
            MemoryRecord(
                key: try MemoryKey(validating: "decisions/sync"),
                value: "FooManager stays."),
            in: scope)
        let turns = ContinuityJournalStore(engine: engine, store: store)
        await turns.record(
            JournalTurn(
                session: "s1",
                workspace: "repo-a",
                index: 0,
                prompt: "Write chapter 34.",
                reply: "The tide came in, and the inn burned down in chapter 34."),
            in: scope)
        await engine.shutDown()
        return url
    }

    /// A directory holding a file, planted at the predicted compaction temp
    /// name: `writeCheckpoint` removes the path first, and a recursive remove
    /// of a directory the process cannot write into fails, so the rename
    /// target cannot be created.
    private func blockCompaction(of url: URL) throws {
        let blocker = url.appendingPathExtension("compacting")
        try FileManager.default.createDirectory(at: blocker, withIntermediateDirectories: true)
        try Data("holding the name".utf8)
            .write(to: blocker.appendingPathComponent("child"))
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o500], ofItemAtPath: blocker.path)
    }

    private func unblockCompaction(of url: URL) {
        try? FileManager.default.setAttributes(
            [.posixPermissions: 0o700],
            ofItemAtPath: url.appendingPathExtension("compacting").path)
    }

    private func transcript(_ url: URL) throws -> String {
        String(bytes: try Data(contentsOf: url), encoding: .utf8) ?? ""
    }

    /// The honest outcome: the transcript goes, the fact stays.
    @Test func expiryDropsTheTranscriptAndKeepsTheFact() async throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = try await writtenJournal(in: directory)
        let before = try transcript(url)
        #expect(before.contains("burned down in chapter 34"))

        let reason = await MemoryService.expireSessionLog(at: url)
        #expect(reason == nil, "a clean expiry returns no reason, got \(reason ?? "nil")")

        let after = try transcript(url)
        #expect(!after.contains("burned down in chapter 34"), "the session log survived expiry")
        #expect(after.count < before.count)
        // Replayed, the workspace still knows its fact and has no sessions.
        let engine = ContinuityEngine(journal: try FileJournal(url: url))
        try await engine.start()
        let store = ContinuityStore(engine: engine)
        let records = try await store.search(MemoryQuery(limit: 10), in: try scope())
        #expect(records.map(\.key.rawValue) == ["decisions/sync"])
        for task in await engine.tasks() {
            #expect(
                await engine.sessions(taskID: task.id).isEmpty,
                "expiry kept the sessions it exists to remove")
        }
        await engine.shutDown()
    }

    /// The defect: a failed compaction used to be swallowed and reported as
    /// an expiry. The reason now comes back, and the transcript is still on
    /// disk because nothing removed it.
    @Test func aFailedCompactionIsReportedAndTheFileStays() async throws {
        let directory = try makeDirectory()
        let url = try await writtenJournal(in: directory, named: "stale.ndjson")
        try blockCompaction(of: url)
        defer {
            unblockCompaction(of: url)
            try? FileManager.default.removeItem(at: directory)
        }

        let reason = await MemoryService.expireSessionLog(at: url)
        #expect(
            reason?.contains("compaction failed") == true,
            "expected a compaction reason, got \(reason ?? "nil")")
        #expect(
            try transcript(url).contains("burned down in chapter 34"),
            "a refused expiry must leave the file exactly as it was")
    }

    /// A journal that is not a file cannot be opened, and that is a reason
    /// rather than an expiry.
    @Test func anUnopenableJournalIsReportedNotClaimedExpired() async throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let folder =
            directory
            .appendingPathComponent("tinytitan/local", isDirectory: true)
            .appendingPathComponent("folder.ndjson", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)

        let reason = await MemoryService.expireSessionLog(at: folder)
        #expect(
            reason?.contains("could not be opened") == true,
            "expected an open reason, got \(reason ?? "nil")")
    }

    /// The sweep's half: one line per stuck file rather than one per sweep,
    /// and no `.expired` event naming a file that was not expired.
    @Test func theSweepReportsARefusalOnceAndNeverSaysExpired() async throws {
        let directory = try makeDirectory()
        let url = try await writtenJournal(in: directory, named: "stale.ndjson")
        try blockCompaction(of: url)
        // Past the one-day cutoff, which is what makes it a candidate.
        try FileManager.default.setAttributes(
            [.modificationDate: Date().addingTimeInterval(-3 * 86_400)],
            ofItemAtPath: url.path)
        defer {
            unblockCompaction(of: url)
            try? FileManager.default.removeItem(at: directory)
        }

        var configuration = MemoryConfiguration()
        configuration.isEnabled = true
        configuration.workspace = "repo-a"
        configuration.user = "local"
        configuration.storage.directory = directory
        configuration.storage.retentionDays = 1
        configuration.storage.maximumWorkspaces = 0

        let events = EventBox()
        let service = MemoryService(configuration: configuration, log: { events.append($0) })
        await service.sweepStaleWorkspaces(now: Date())
        await service.sweepStaleWorkspaces(now: Date())

        let messages = events.messages()
        let refusals = messages.filter { $0.contains("degraded during expire") }
        #expect(refusals.count == 1, "one refusal per file, got \(refusals)")
        #expect(
            refusals.first?.contains("stale.ndjson: compaction failed") == true,
            "the refusal should name the file and the step: \(refusals)")
        #expect(
            messages.filter { $0.contains("expired the session log") }.isEmpty,
            "a file that was not expired was reported as expired")
    }

    /// unchecked-invariant: `events` is only ever touched under `lock`.
    private final class EventBox: @unchecked Sendable {
        private let lock = NSLock()
        private var events: [MemoryLogEvent] = []
        func append(_ event: MemoryLogEvent) { lock.withLock { events.append(event) } }
        func messages() -> [String] { lock.withLock { events.map(\.message) } }
    }
}
