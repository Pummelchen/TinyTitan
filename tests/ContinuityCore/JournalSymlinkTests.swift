import Foundation
import Testing

@testable import ContinuityCore

/// Every path `FileJournal` writes to is one it derives from the store
/// directory — `<journal>.lock`, `<journal>.compacting`, the journal itself —
/// so anything with write access to that directory can plant a link at a
/// predicted name. These tests are the two halves of the guard, per site: the
/// open is refused, and the file behind the link still holds its bytes. The
/// second half is the one that matters; an error that arrives after `O_TRUNC`
/// has already gone through the link protects nothing.
@Suite struct JournalSymlinkTests {
    private func temporaryDirectory() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("journal-symlink-\(UUID().uuidString)")
    }

    /// Created, because a journal's own directory only appears when the
    /// journal opens it and these tests write a victim file first.
    private func madeDirectory() throws -> URL {
        let directory = temporaryDirectory()
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        return directory
    }

    /// A directory holding `victim`, plus a link at `linkName` pointing at it.
    /// Returns the directory, the link URL and the canary.
    private func plantedLink(
        named linkName: String,
        tag: String
    ) throws -> (directory: URL, link: URL, canary: Data) {
        let directory = temporaryDirectory()
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let victim = directory.appendingPathComponent("victim-\(tag)")
        let canary = Data("do not truncate me".utf8)
        try canary.write(to: victim)
        let link = directory.appendingPathComponent(linkName)
        try FileManager.default.createSymbolicLink(
            at: link, withDestinationURL: victim)
        return (directory, link, canary)
    }

    private func victimSurvives(_ victim: URL, as canary: Data) throws {
        #expect(try Data(contentsOf: victim) == canary)
    }

    // MARK: - The append handle

    @Test func aSymlinkAtTheJournalPathIsRefusedAndItsTargetSurvives() async throws {
        let planted = try plantedLink(named: "journal.ndjson", tag: "append")
        defer { try? FileManager.default.removeItem(at: planted.directory) }

        #expect(throws: JournalError.self) {
            _ = try FileJournal(url: planted.link)
        }
        try victimSurvives(
            planted.directory.appendingPathComponent("victim-append"), as: planted.canary)
    }

    // MARK: - The workspace lock

    @Test func aSymlinkAtTheLockPathIsRefusedAndItsTargetSurvives() async throws {
        let planted = try plantedLink(named: "journal.ndjson.lock", tag: "lock")
        defer { try? FileManager.default.removeItem(at: planted.directory) }
        let journalURL = planted.directory.appendingPathComponent("journal.ndjson")

        #expect(throws: JournalError.self) {
            _ = try FileJournal(url: journalURL)
        }
        // Refused, and not opened for the flock either: the anchor's bytes are
        // untouched and no journal file was created behind it.
        try victimSurvives(
            planted.directory.appendingPathComponent("victim-lock"), as: planted.canary)
        #expect(!FileManager.default.fileExists(atPath: journalURL.path))
    }

    // MARK: - The compaction temp

    @Test func aSymlinkAtTheCompactionTempWritesNothingThroughItAndKeepsTheJournal() async throws {
        let directory = try madeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let journalURL = directory.appendingPathComponent("journal.ndjson")
        let victim = directory.appendingPathComponent("victim-compacting")
        let canary = Data("do not truncate me".utf8)
        try canary.write(to: victim)

        let journal = try FileJournal(url: journalURL)
        try await journal.append(.task(ContinuityTask(title: "kept")))
        // Compaction unlinks its temp name before creating it, so at this one
        // site the link is removed and the create lands on a fresh file: the
        // observable contract here is the damage property, not a thrown error.
        // `O_NOFOLLOW` is what covers the window the unlink cannot -- a temp
        // path the process fails to remove, where the create would otherwise
        // truncate a victim chosen by whoever planted the link.
        let temporary = journalURL.appendingPathExtension("compacting")
        try FileManager.default.createSymbolicLink(
            at: temporary, withDestinationURL: victim)

        try await journal.compact(sessionLog: SessionLogSnapshot(), memory: MemorySnapshot())
        try victimSurvives(victim, as: canary)
        #expect(!FileManager.default.fileExists(atPath: temporary.path))
        // The journal is the checkpoint, and the checkpoint is readable.
        #expect(try FileJournal.read(contentsOf: journalURL).count == 1)
        try await journal.shutDown()
    }

    // MARK: - truncate()

    @Test func aSymlinkPlantedAfterOpeningIsRefusedByTruncate() async throws {
        let directory = try madeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let journalURL = directory.appendingPathComponent("journal.ndjson")
        let victim = directory.appendingPathComponent("victim-truncate")
        let canary = Data("do not truncate me".utf8)
        try canary.write(to: victim)

        let journal = try FileJournal(url: journalURL)
        try await journal.append(.task(ContinuityTask(title: "before")))
        // Swap the file out from under the open descriptor, which is what a
        // process that can write the directory can do between two calls.
        try FileManager.default.removeItem(at: journalURL)
        try FileManager.default.createSymbolicLink(
            at: journalURL, withDestinationURL: victim)

        await #expect(throws: JournalError.self) {
            try await journal.truncate()
        }
        try victimSurvives(victim, as: canary)
        try await journal.shutDown()
    }

    // MARK: - What must keep working

    @Test func aJournalUnderANamedDirectorySymlinkStillOpens() async throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let real = root.appendingPathComponent("real-store")
        try FileManager.default.createDirectory(at: real, withIntermediateDirectories: true)
        // The supported way to keep the memory store on a second disk.
        // `O_NOFOLLOW` covers the final component only, so this must be
        // unaffected; a guard that broke it would be a regression.
        let alias = root.appendingPathComponent("alias-store")
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: real)

        let journalURL = alias.appendingPathComponent("journal.ndjson")
        let journal = try FileJournal(url: journalURL)
        try await journal.append(.task(ContinuityTask(title: "through the alias")))
        try await journal.shutDown()
        #expect(try FileJournal.read(contentsOf: journalURL).count == 1)
        #expect(
            FileManager.default.fileExists(
                atPath: real.appendingPathComponent("journal.ndjson").path))
    }
}
