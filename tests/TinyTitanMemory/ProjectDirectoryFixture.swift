import ContinuityCore
import Foundation

@testable import TinyTitanMemory

/// A scratch memory directory with projects that load, and names that cannot
/// be loaded.
///
/// Projects are written by the real engine, so the files are the ones a server
/// would leave behind. The unreadable cases are the cheap POSIX ones a
/// non-privileged test can create: a mode with no read bit, a directory where
/// a file is expected, and a FIFO.
struct ProjectDirectoryFixture {
    let directory: URL

    init() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("memory-discovery-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        self.directory = directory
    }

    /// A readable project with one fact, and the value that fact holds.
    ///
    /// `parent` places it below the root, the layout the scan also descends
    /// into; the top-level store is one file per workspace.
    @discardableResult
    func makeProject(
        _ workspace: String, in parent: URL? = nil
    ) async throws -> (url: URL, value: String) {
        let url = (parent ?? directory).appendingPathComponent("\(workspace).ndjson")
        let value = "value-for-\(workspace)"
        let journal = try FileJournal(url: url, idleDelay: .seconds(60))
        let engine = ContinuityEngine(journal: journal)
        try await engine.start()
        let task = try await engine.createTask(title: workspace)
        let session = try await engine.beginSession(taskID: task.id)
        try await engine.remember(
            sessionID: session.id, namespace: "notes", key: "kept", value: value)
        await engine.shutDown()
        return (url, value)
    }

    /// Remove every permission bit: the owner can no longer open it.
    @discardableResult
    func makeUnreadable(_ workspace: String) throws -> URL {
        let url = directory.appendingPathComponent("\(workspace).ndjson")
        try Data("{}\n".utf8).write(to: url)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o600], ofItemAtPath: url.path)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o000], ofItemAtPath: url.path)
        return url
    }

    /// A directory at the path a project file would occupy.
    @discardableResult
    func makeDirectoryAtProjectPath(_ workspace: String) throws -> URL {
        let url = directory.appendingPathComponent("\(workspace).ndjson")
        try FileManager.default.createDirectory(
            at: url, withIntermediateDirectories: false)
        return url
    }

    /// A FIFO at the path a project file would occupy.
    @discardableResult
    func makeFIFOAtProjectPath(_ workspace: String) throws -> URL {
        let url = directory.appendingPathComponent("\(workspace).ndjson")
        let made = url.path.withCString { mkfifo($0, 0o600) }
        guard made == 0 else {
            throw JournalError.cannotOpen(url, underlying: "mkfifo failed")
        }
        return url
    }

    /// Remove the read bit from the directory itself, so it cannot be listed.
    func makeUnlistable() throws {
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o000], ofItemAtPath: directory.path)
    }

    /// A subdirectory of the store, readable.
    @discardableResult
    func makeNestedDirectory(_ name: String) throws -> URL {
        let url = directory.appendingPathComponent(name)
        try FileManager.default.createDirectory(
            at: url, withIntermediateDirectories: false)
        return url
    }

    /// A subdirectory that can be seen but not entered, so a walk of the root
    /// cannot descend into it.
    @discardableResult
    func makeUnlistableSubdirectory(_ name: String) throws -> URL {
        let url = try makeNestedDirectory(name)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o000], ofItemAtPath: url.path)
        return url
    }

    /// Put a plain file where a project directory is expected. Inside the
    /// fixture so its cleanup owns it; the name has no `.ndjson` extension, so
    /// it does not disturb a scan of the directory itself.
    func makeFileWhereADirectoryIsExpected() throws -> URL {
        let url = directory.appendingPathComponent("not-a-directory")
        try Data("{}\n".utf8).write(to: url)
        return url
    }

    func cleanup() {
        // Restore the root's mode first, or the enumerator below cannot walk
        // into a directory this fixture made unlistable.
        _ = try? FileManager.default.setAttributes(
            [.posixPermissions: 0o700], ofItemAtPath: directory.path)
        if let walker = FileManager.default.enumerator(
            at: directory, includingPropertiesForKeys: nil)
        {
            for case let url as URL in walker {
                _ = try? FileManager.default.setAttributes(
                    [.posixPermissions: 0o700], ofItemAtPath: url.path)
            }
        }
        try? FileManager.default.removeItem(at: directory)
    }
}
