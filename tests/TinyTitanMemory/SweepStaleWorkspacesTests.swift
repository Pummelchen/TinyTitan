import Foundation
import Testing

@testable import TinyTitanMemory

/// The workspace cap is the only retention rule that *removes facts*, so what it
/// is allowed to delete matters more than what it does delete.
@Suite struct SweepStaleWorkspacesTests {

    private func makeDirectory() throws -> URL {
        let url = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("tinytitan-sweep-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func write(_ url: URL, modified: Date) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try Data("{}\n".utf8).write(to: url)
        try FileManager.default.setAttributes(
            [.modificationDate: modified],
            ofItemAtPath: url.path)
    }

    /// Cap only: retention rewrites survivors through the journal, which is a
    /// different path (and takes the lock itself).
    private func configuration(directory: URL, cap: Int) -> MemoryConfiguration {
        var configuration = MemoryConfiguration()
        configuration.storage.directory = directory
        configuration.storage.maximumWorkspaces = cap
        configuration.storage.retentionDays = 0
        return configuration
    }

    /// Holds a workspace's lock the way `FileJournal` does. `flock` conflicts
    /// between open-file-descriptions even inside one process, so this stands in
    /// for another process exactly as far as the probe is concerned.
    private func lock(_ journalURL: URL) -> Int32 {
        let path = journalURL.appendingPathExtension("lock").path
        let descriptor = open(path, O_RDWR | O_CREAT | O_CLOEXEC, 0o600)
        guard descriptor >= 0, flock(descriptor, LOCK_EX | LOCK_NB) == 0 else {
            Issue.record("could not take the workspace lock at \(path)")
            if descriptor >= 0 { close(descriptor) }
            return -1
        }
        return descriptor
    }

    /// Three workspaces and a cap of one: the two oldest are doomed, and the
    /// oldest of all is held by "another process". Before the lock probe both
    /// were deleted, taking the holder's `.lock` with them.
    @Test func theCapLeavesALockedWorkspaceAlone() async throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let now = Date()
        let locked = directory.appendingPathComponent("tinytitan/local/locked.ndjson")
        let free = directory.appendingPathComponent("tinytitan/local/free.ndjson")
        let kept = directory.appendingPathComponent("tinytitan/local/kept.ndjson")
        try write(locked, modified: now.addingTimeInterval(-300))
        try write(free, modified: now.addingTimeInterval(-200))
        try write(kept, modified: now.addingTimeInterval(-100))
        let descriptor = lock(locked)
        defer { if descriptor >= 0 { close(descriptor) } }

        let service = MemoryService(configuration: configuration(directory: directory, cap: 1))
        await service.sweepStaleWorkspaces()

        let manager = FileManager.default
        #expect(
            manager.fileExists(atPath: locked.path),
            "a workspace another process holds was deleted")
        #expect(
            manager.fileExists(atPath: locked.appendingPathExtension("lock").path),
            "the holder's lock file was deleted with it")
        #expect(
            manager.fileExists(atPath: kept.path),
            "the newest workspace is inside the cap and must survive")
        #expect(
            !manager.fileExists(atPath: free.path),
            "an unlocked workspace past the cap was not deleted")
    }

    /// The probe must not amount to disabling the cap: with no lock held, the
    /// same sweep deletes the same file.
    @Test func theCapStillDeletesAnUnlockedWorkspace() async throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let now = Date()
        let older = directory.appendingPathComponent("tinytitan/local/older.ndjson")
        let newer = directory.appendingPathComponent("tinytitan/local/newer.ndjson")
        try write(older, modified: now.addingTimeInterval(-300))
        try write(newer, modified: now.addingTimeInterval(-100))

        let service = MemoryService(configuration: configuration(directory: directory, cap: 1))
        await service.sweepStaleWorkspaces()

        let manager = FileManager.default
        #expect(!manager.fileExists(atPath: older.path))
        #expect(manager.fileExists(atPath: newer.path))
    }

    /// The cap is the one retention rule that removes facts, so a file it could
    /// not delete has to stay out of the list it reports as deleted. The expiry
    /// rule already refuses to report a file it could not expire
    /// (`theSweepReportsARefusalOnceAndNeverSaysExpired`); the delete rule's own
    /// `try?` swallowed the failure, named the file as swept anyway, and left a
    /// read-only volume with the disk still growing and the log saying it had
    /// been reclaimed.
    @Test func theCapReportsOnlyFilesItActuallyDeleted() async throws {
        let directory = try makeDirectory()
        let now = Date()
        let stuck = directory.appendingPathComponent("tinytitan/local/stuck.ndjson")
        let kept = directory.appendingPathComponent("tinytitan/local/kept.ndjson")
        try write(stuck, modified: now.addingTimeInterval(-300))
        try write(kept, modified: now.addingTimeInterval(-100))
        // Nothing inside this directory can be unlinked, so the doomed journal
        // survives its removal. The lock probe still answers "not held" (it
        // opens without `O_CREAT`, and there is no lock file), so the sweep
        // really does attempt the delete rather than skipping the file.
        let parent = stuck.deletingLastPathComponent()
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o500], ofItemAtPath: parent.path)
        defer {
            try? FileManager.default.setAttributes(
                [.posixPermissions: 0o700], ofItemAtPath: parent.path)
            try? FileManager.default.removeItem(at: directory)
        }

        let events = EventBox()
        let service = MemoryService(
            configuration: configuration(directory: directory, cap: 1),
            log: { events.append($0) })
        await service.sweepStaleWorkspaces()
        await service.sweepStaleWorkspaces()

        let manager = FileManager.default
        #expect(
            manager.fileExists(atPath: stuck.path),
            "the probe failed: the read-only directory did not refuse the removal")
        let messages = events.messages()
        #expect(
            !messages.filter { $0.contains("memory swept") }
                .contains { $0.contains("stuck.ndjson") },
            "a file that is still on disk was reported as deleted: \(messages)")
        let refusals = messages.filter { $0.contains("degraded during sweep") }
        #expect(refusals.count == 1, "one refusal per stuck file, got \(refusals)")
        #expect(
            refusals.first?.contains("sweep: stuck.ndjson: ") == true,
            "the refusal should name the operation, the file and the step: \(refusals)")
    }

    /// A workspace with no `.lock` file at all was never opened by a journal, so
    /// nothing can be holding it and the probe must say so rather than treating
    /// the missing file as "in use" (which would disable the cap for every
    /// workspace that has ever been idle).
    @Test func aWorkspaceWithoutALockFileIsNotHeld() throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let journal = directory.appendingPathComponent("tinytitan/local/never-opened.ndjson")
        try write(journal, modified: Date())
        #expect(!MemoryService.isLockHeld(at: journal))
    }

    /// A real, regular, unheld lock must still read as free: the `O_NOFOLLOW`
    /// probe below is only allowed to reject links, not to turn every idle
    /// workspace into a held one and switch the cap off.
    @Test func aRegularUnheldLockIsNotHeld() throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let journal = directory.appendingPathComponent("tinytitan/local/idle.ndjson")
        try write(journal, modified: Date())
        try Data().write(to: journal.appendingPathExtension("lock"))
        #expect(!MemoryService.isLockHeld(at: journal))
    }

    /// The sibling of AUD-110's symlink guard, at the probe that protects
    /// deletion. A link planted at the predicted `<journal>.lock` name would
    /// otherwise be locked *instead of* the real anchor: the probe answers
    /// "not held", the sweep deletes a journal another process is appending
    /// to, and that process's next compaction rewrites the workspace from its
    /// own stale memory. An unreadable probe is a held probe.
    @Test func aSymlinkedLockCountsAsHeldAndItsTargetSurvives() throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let journal = directory.appendingPathComponent("tinytitan/local/linked.ndjson")
        try write(journal, modified: Date())
        let victim = directory.appendingPathComponent("someone-elses-file")
        let canary = Data("do not lock me".utf8)
        try canary.write(to: victim)
        try FileManager.default.createSymbolicLink(
            at: journal.appendingPathExtension("lock"), withDestinationURL: victim)

        #expect(MemoryService.isLockHeld(at: journal))
        #expect(try Data(contentsOf: victim) == canary)
    }

    /// unchecked-invariant: `events` is only ever touched under `lock`.
    private final class EventBox: @unchecked Sendable {
        private let lock = NSLock()
        private var events: [MemoryLogEvent] = []
        func append(_ event: MemoryLogEvent) { lock.withLock { events.append(event) } }
        func messages() -> [String] { lock.withLock { events.map(\.message) } }
    }
}
