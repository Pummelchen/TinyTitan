import Foundation
import Testing

@testable import TinyTitanMemory

/// `tinytitan-memory` against a `--dir` it cannot list.
///
/// The command fails with the path and the reason, rather than reporting an
/// empty directory. A listed directory containing a broken file is unchanged:
/// that is a warning on stderr and exit 0 (see `MemoryToolBrokenProjectTests`).
@Suite(.serialized)
struct MemoryToolListingFailureTests {
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

    @Test func projectsOnAnUnlistableDirectoryFails() async throws {
        let fixture = try ProjectDirectoryFixture()
        defer { fixture.cleanup() }
        try await fixture.makeProject("repo-a")
        try fixture.makeUnlistable()

        let result = try run(fixture.directory, ["projects"])
        #expect(result.status == 1, "stdout: \(result.stdout)")
        #expect(result.stdout.isEmpty, "stdout: \(result.stdout)")
        #expect(result.stderr.contains("could not be listed"), "stderr: \(result.stderr)")
        #expect(
            result.stderr.contains(
                fixture.directory.resolvingSymlinksInPath().path),
            "stderr: \(result.stderr)")
        #expect(!result.stdout.contains("no project files"))
    }

    @Test func listOnAnUnlistableDirectoryFailsWithTheReason() async throws {
        let fixture = try ProjectDirectoryFixture()
        defer { fixture.cleanup() }
        try fixture.makeUnlistable()

        let result = try run(fixture.directory, ["list", "repo-a"])
        #expect(result.status == 1, "stdout: \(result.stdout)")
        #expect(result.stderr.contains("could not be listed"), "stderr: \(result.stderr)")
        #expect(!result.stderr.contains("no project matches"))
    }

    @Test func projectsOnAMissingDirectoryFails() throws {
        let fixture = try ProjectDirectoryFixture()
        defer { fixture.cleanup() }
        let missing = fixture.directory.appendingPathComponent("absent")

        let result = try run(missing, ["projects"])
        #expect(result.status == 1, "stdout: \(result.stdout)")
        #expect(result.stderr.contains("could not be listed"), "stderr: \(result.stderr)")
        #expect(result.stderr.contains("does not exist"), "stderr: \(result.stderr)")
    }

    /// The clean directory's answer is untouched.
    @Test func projectsOnACleanDirectoryStillAnswers() async throws {
        let fixture = try ProjectDirectoryFixture()
        defer { fixture.cleanup() }
        try await fixture.makeProject("repo-a")

        let result = try run(fixture.directory, ["projects"])
        #expect(result.status == 0, "stderr: \(result.stderr)")
        #expect(result.stdout.contains("repo-a"), "stdout: \(result.stdout)")
        #expect(!result.stderr.contains("could not be listed"), "stderr: \(result.stderr)")
    }
}
