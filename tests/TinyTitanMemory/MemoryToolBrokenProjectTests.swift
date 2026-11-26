import Foundation
import Testing

@testable import TinyTitanMemory

/// `tinytitan-memory` over a directory that holds a project it cannot open.
///
/// The readable projects are listed and answered exactly as before; the one
/// that cannot be read is named on stderr with the reason, and asking for it
/// says it is unreadable rather than pretending no such project exists.
@Suite(.serialized)
struct MemoryToolBrokenProjectTests {
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

    @Test func projectsListsTheReadableOnesAndNamesTheBrokenOne() async throws {
        let fixture = try ProjectDirectoryFixture()
        defer { fixture.cleanup() }
        try await fixture.makeProject("readable-project")
        try fixture.makeUnreadable("locked-project")

        let result = try run(fixture.directory, ["projects"])
        #expect(result.status == 0, "stderr: \(result.stderr)")
        #expect(result.stdout.contains("readable-project"), "stdout: \(result.stdout)")
        // The clean row is unchanged and the broken file is not on stdout...
        #expect(!result.stdout.contains("locked-project"), "stdout: \(result.stdout)")
        // ...it is named as broken, with a reason, on stderr.
        #expect(result.stderr.contains("locked-project"), "stderr: \(result.stderr)")
        #expect(result.stderr.contains("could not be read"), "stderr: \(result.stderr)")
        #expect(result.stderr.contains("Repair the file"), "stderr: \(result.stderr)")
    }

    @Test func listOfAnUnreadableProjectSaysSo() async throws {
        let fixture = try ProjectDirectoryFixture()
        defer { fixture.cleanup() }
        try fixture.makeUnreadable("locked-project")

        let result = try run(fixture.directory, ["list", "locked-project"])
        #expect(result.status == 1, "stdout: \(result.stdout) stderr: \(result.stderr)")
        #expect(result.stderr.contains("is present but could not be read"))
        #expect(result.stderr.contains("locked-project"))
        #expect(!result.stderr.contains("no project matches"))
        #expect(result.stdout.isEmpty)
    }

    @Test func aReadableProjectIsUnaffectedByItsBrokenNeighbour() async throws {
        let fixture = try ProjectDirectoryFixture()
        defer { fixture.cleanup() }
        let project = try await fixture.makeProject("readable-project")
        try fixture.makeUnreadable("locked-project")

        let result = try run(fixture.directory, ["list", "readable-project"])
        #expect(result.status == 0, "stderr: \(result.stderr)")
        #expect(result.stdout.contains(project.value), "stdout: \(result.stdout)")
        #expect(!result.stderr.contains("could not be read"), "stderr: \(result.stderr)")
    }

    /// A directory whose only project is broken is not an empty directory.
    @Test func aDirectoryOfOnlyBrokenProjectsStillAnswers() async throws {
        let fixture = try ProjectDirectoryFixture()
        defer { fixture.cleanup() }
        try fixture.makeUnreadable("locked-project")

        let result = try run(fixture.directory, ["projects"])
        #expect(result.status == 0, "stderr: \(result.stderr)")
        #expect(result.stdout.contains(fixture.directory.path), "stdout: \(result.stdout)")
        #expect(!result.stdout.contains("no project files"))
        #expect(result.stderr.contains("locked-project"))
    }

    @Test func anEmptyDirectoryStillSaysThereAreNoProjects() throws {
        let fixture = try ProjectDirectoryFixture()
        defer { fixture.cleanup() }

        let result = try run(fixture.directory, ["projects"])
        #expect(result.status == 0, "stderr: \(result.stderr)")
        #expect(result.stdout.contains("no project files"))
        #expect(result.stderr.isEmpty)
    }
}
