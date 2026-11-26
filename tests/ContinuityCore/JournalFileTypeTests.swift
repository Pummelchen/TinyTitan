import Foundation
import Testing

@testable import ContinuityCore

/// What the journal path is allowed to be.
///
/// `O_NOFOLLOW` refuses a link standing at the final component, and the create
/// mode `0600` governs a file this process makes. Neither says anything about
/// a file that already exists: a FIFO, a device, a directory, a file another
/// user owns, or a journal the rest of the machine can read all pass those two
/// checks. The store directory is the operator's and may be a symlink on
/// purpose; the journal file inside it has to be exactly one regular,
/// owner-only file, and every opener checks the descriptor it actually got.
@Suite struct JournalFileTypeTests {
    private func temporaryDirectory() -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("journal-type-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        return directory
    }

    private func makeFIFO(_ url: URL) throws {
        let made = url.path.withCString { mkfifo($0, 0o600) }
        guard made == 0 else {
            throw JournalError.cannotOpen(url, underlying: "mkfifo failed")
        }
    }

    private func mode(_ url: URL) throws -> mode_t {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        return (attributes[.posixPermissions] as? NSNumber)?.uint16Value ?? 0
    }

    // MARK: - The read path

    /// A read is not a way around the guard just because it looks harmless.
    /// `O_NONBLOCK` makes the open return; the type check is what refuses it
    /// before anything reads from it.
    @Test func aFIFOAtTheJournalPathIsRefusedByTheReadPath() throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        try makeFIFO(url)

        do {
            _ = try FileJournal.read(contentsOf: url)
            Issue.record("a FIFO was read as if it were a journal")
        } catch {
            guard case .notAFile(let failedURL) = error else {
                Issue.record("expected notAFile, got \(error)")
                return
            }
            #expect(failedURL.path == url.path)
        }
    }

    @Test func aSymlinkAtTheJournalPathIsRefusedByTheReadPath() throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let victim = directory.appendingPathComponent("victim")
        let canary = Data("do not read me".utf8)
        try canary.write(to: victim)
        let link = directory.appendingPathComponent("journal.ndjson")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: victim)

        #expect(throws: JournalError.self) {
            _ = try FileJournal.read(contentsOf: link)
        }
        #expect(try Data(contentsOf: victim) == canary)
    }

    @Test func aDirectoryAtTheJournalPathIsRefusedByTheReadPath() throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)

        do {
            _ = try FileJournal.read(contentsOf: url)
            Issue.record("a directory was read as if it were a journal")
        } catch {
            guard case .notAFile = error else {
                Issue.record("expected notAFile, got \(error)")
                return
            }
        }
    }

    // MARK: - The write path

    @Test func aFIFOAtTheJournalPathIsRefusedByTheWritePath() throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        try makeFIFO(url)

        do {
            _ = try FileJournal(url: url)
            Issue.record("a FIFO was opened as an append journal")
        } catch let error as JournalError {
            guard case .notAFile(let failedURL) = error else {
                Issue.record("expected notAFile, got \(error)")
                return
            }
            #expect(failedURL.path == url.path)
        }
        // Still a FIFO, not replaced by a regular file.
        var status = stat()
        #expect(lstat(url.path, &status) == 0)
        #expect(status.st_mode & S_IFMT == S_IFIFO)
    }

    @Test func aFIFOAtTheLockPathIsRefusedAndCreatesNoJournal() throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        try makeFIFO(url.appendingPathExtension("lock"))

        #expect(throws: JournalError.self) {
            _ = try FileJournal(url: url)
        }
        #expect(!FileManager.default.fileExists(atPath: url.path))
    }

    /// A journal written by an older build or by hand at `0644` holds prompts
    /// and replies. It is refused, not silently rewritten or `chmod`ed: the
    /// operator is the one who can decide the file is theirs to repair.
    @Test func aPermissiveExistingJournalIsRefusedAndLeftAlone() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let writer = try FileJournal(url: url)
        try await writer.append(.task(ContinuityTask(title: "private")))
        try await writer.shutDown()
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o644], ofItemAtPath: url.path)

        do {
            _ = try FileJournal(url: url)
            Issue.record("a world-readable journal was opened for writing")
        } catch let error as JournalError {
            guard case .insecurePermissions(let failedURL, let fileMode) = error else {
                Issue.record("expected insecurePermissions, got \(error)")
                return
            }
            #expect(failedURL.path == url.path)
            #expect(fileMode & 0o077 != 0)
            #expect(error.description.contains("chmod 600"))
        }
        #expect(try mode(url) == 0o644, "the refused open changed the file's mode")

        // The documented repair is the only thing that makes it openable.
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o600], ofItemAtPath: url.path)
        let reopened = try FileJournal(url: url)
        #expect(try await reopened.replay().count == 1)
        try await reopened.shutDown()
    }

    @Test func aJournalOwnedByAnotherUserIsRefused() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let writer = try FileJournal(url: url)
        try await writer.append(.task(ContinuityTask(title: "kept")))
        try await writer.shutDown()

        let fileSystem = JournalFaultFileSystem()
        // Another uid, without needing privileges: the check reads the
        // descriptor's own `fstat`, and this only overrides what it reports
        // for the journal.
        fileSystem.overrideOwner(geteuid() == 0 ? 501 : geteuid() + 1, onlyFor: url.path)
        do {
            _ = try FileJournal(url: url, fileSystem: fileSystem)
            Issue.record("a journal owned by another user was opened for writing")
        } catch let error as JournalError {
            guard case .wrongOwner(let failedURL, _) = error else {
                Issue.record("expected wrongOwner, got \(error)")
                return
            }
            #expect(failedURL.path == url.path)
        }
    }

    @Test func aLockOwnedByAnotherUserIsRefusedAndCreatesNoJournal() throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let lock = url.appendingPathExtension("lock")
        try Data().write(to: lock)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o600], ofItemAtPath: lock.path)

        let fileSystem = JournalFaultFileSystem()
        fileSystem.overrideOwner(geteuid() == 0 ? 501 : geteuid() + 1, onlyFor: lock.path)
        #expect(throws: JournalError.self) {
            _ = try FileJournal(url: url, fileSystem: fileSystem)
        }
        #expect(!FileManager.default.fileExists(atPath: url.path))
    }

    /// A directory at the journal path is refused before anything is opened.
    @Test func aDirectoryAtTheJournalPathIsRefusedByTheWritePath() throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)
        #expect(throws: JournalError.self) {
            _ = try FileJournal(url: url)
        }
    }

    // MARK: - What must keep working

    /// The supported way to keep the store on another disk: a symlink on the
    /// directory. That is covered by `JournalSymlinkTests`; this pins that the
    /// new checks did not add anything a normal owner-only journal trips.
    @Test func anOwnerOnlyRegularJournalStillOpens() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let journal = try FileJournal(url: url)
        try await journal.append(.task(ContinuityTask(title: "ordinary")))
        try await journal.shutDown()
        #expect(try mode(url) == 0o600)

        let reopened = try FileJournal(url: url)
        #expect(try await reopened.replay().count == 1)
        try await reopened.shutDown()
    }
}
