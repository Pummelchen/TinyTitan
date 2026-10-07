import Foundation
import Testing

@testable import TinyTitanMemory

/// `forget` is the one command in `tinytitan-memory` that removes a whole
/// project, and both its own usage header (`main.swift:15-16`) and
/// `docs/agent-memory.md` promise it "refuses the workspace while a server
/// holds it". It did not: the branch unlinked the journal and then the `.lock`
/// sidecar — the file that server's `flock` is attached to — and reported
/// `forgot <project>` over exit 0. `MemoryService.isLockHeld` states what
/// follows: the server goes on appending to a deleted inode, and its next
/// compaction republishes that inode as the workspace, so the disk silently
/// becomes whatever one process holds in memory.
///
/// The tool is an executable target with no library behind it, so the code
/// under test is the built binary. The lock is taken in *this* process, which
/// is enough: `flock` conflicts between open file descriptions even inside one
/// process, the same stand-in `SweepStaleWorkspacesTests` relies on.
@Suite(.serialized)
struct MemoryToolForgetTests {

    /// A project file the tool can discover: `{}` is a line no record decodes
    /// from, so the journal is empty but readable — enough for a workspace to
    /// exist, which is all `forget` asks of it.
    private func makeWorkspace(_ name: String) throws -> (directory: URL, journal: URL) {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("tinytitan-forget-\(UUID().uuidString)", isDirectory: true)
        let journal = directory.appendingPathComponent("\(name).ndjson")
        try FileManager.default.createDirectory(
            at: journal.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("{}\n".utf8).write(to: journal)
        return (directory, journal)
    }

    /// Holds `journal`'s workspace lock the way a running server does.
    private func lock(_ journal: URL) throws -> Int32 {
        let path = journal.appendingPathExtension("lock").path
        let descriptor = open(path, O_RDWR | O_CREAT | O_CLOEXEC, 0o600)
        try #require(descriptor >= 0)
        guard flock(descriptor, LOCK_EX | LOCK_NB) == 0 else {
            close(descriptor)
            throw POSIXError(POSIXError.Code(rawValue: errno) ?? .EIO)
        }
        return descriptor
    }

    private func run(_ directory: URL, _ arguments: [String]) throws
        -> (status: Int32, stdout: String, stderr: String)
    {
        let executable = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(".build/debug/tinytitan-memory")
        try #require(
            FileManager.default.fileExists(atPath: executable.path),
            "the suite runs the built tool, so \(executable.path) must exist")
        let process = Process()
        let stdout = Pipe()
        let stderr = Pipe()
        process.executableURL = executable
        process.arguments = ["--dir", directory.path] + arguments
        process.standardOutput = stdout
        process.standardError = stderr
        try process.run()
        process.waitUntilExit()
        let out = stdout.fileHandleForReading.readDataToEndOfFile()
        let err = stderr.fileHandleForReading.readDataToEndOfFile()
        return (
            process.terminationStatus,
            try #require(String(bytes: out, encoding: .utf8)),
            try #require(String(bytes: err, encoding: .utf8))
        )
    }

    /// The refusal: a held workspace is left exactly as it was, and the failure
    /// is loud — a nonzero exit and the name of the thing to do about it.
    @Test func forgetRefusesAWorkspaceAnotherProcessHolds() throws {
        let (directory, journal) = try makeWorkspace("held-project")
        defer { try? FileManager.default.removeItem(at: directory) }
        let descriptor = try lock(journal)
        defer { close(descriptor) }

        let result = try run(directory, ["forget", "held-project", "--yes"])

        #expect(result.status == 1, "stdout: \(result.stdout) stderr: \(result.stderr)")
        #expect(
            result.stderr.contains("Stop the server that has this project open"),
            "stderr: \(result.stderr)")
        #expect(
            !result.stdout.contains("forgot"),
            "the tool reported success over a refusal: \(result.stdout)")
        #expect(
            FileManager.default.fileExists(atPath: journal.path),
            "the held journal was deleted anyway")
        #expect(
            FileManager.default.fileExists(
                atPath: journal.appendingPathExtension("lock").path),
            "the holder's .lock was unlinked out from under its flock")
    }

    /// The counter-test, and the reason the one above is not trivially green:
    /// nothing may refuse when the workspace *is* free. Both removals happen —
    /// the journal and the now-unowned sidecar.
    @Test func forgetStillDeletesAFreeWorkspace() throws {
        let (directory, journal) = try makeWorkspace("free-project")
        defer { try? FileManager.default.removeItem(at: directory) }

        let result = try run(directory, ["forget", "free-project", "--yes"])

        #expect(result.status == 0, "stderr: \(result.stderr)")
        #expect(result.stdout.contains("forgot free-project"), "stdout: \(result.stdout)")
        #expect(
            !FileManager.default.fileExists(atPath: journal.path),
            "a free workspace was not forgotten")
    }

    /// `--yes` is the whole guard on a command that erases a project, so the
    /// lock probe must not come before it: without `--yes` nothing is opened,
    /// and the prompt the operator has to answer stays the destructive one.
    @Test func forgetWithoutConfirmationStillRefusesToAct() throws {
        let (directory, journal) = try makeWorkspace("unconfirmed-project")
        defer { try? FileManager.default.removeItem(at: directory) }

        let result = try run(directory, ["forget", "unconfirmed-project"])

        #expect(result.status == 1)
        #expect(result.stderr.contains("repeat with --yes"), "stderr: \(result.stderr)")
        #expect(FileManager.default.fileExists(atPath: journal.path))
    }
}
