import Foundation
import Testing

@testable import TinyTitanMemory

/// The `tinytitan-memory` commands over a damaged project file.
///
/// The listing still answers — the records that decoded are real facts — and
/// the operator is told, on stderr, that the answer is incomplete. stdout
/// keeps its shape, because it is the part a script may read.
@Suite(.serialized)
struct MemoryToolDamageTests {
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

    @Test func listStillAnswersAndSaysTheFileIsIncomplete() async throws {
        let fixture = try await DamagedProjectFixture()
        defer { try? FileManager.default.removeItem(at: fixture.directory) }

        let result = try run(fixture.directory, ["list", "damaged-project"])
        #expect(result.status == 0, "stderr: \(result.stderr)")
        // The fact that decoded is shown.
        #expect(result.stdout.contains(fixture.factValue), "stdout: \(result.stdout)")
        // And the answer is qualified, on stderr.
        #expect(result.stderr.contains("is incomplete"), "stderr: \(result.stderr)")
        #expect(result.stderr.contains("did not decode"), "stderr: \(result.stderr)")
        #expect(
            result.stderr.contains(String(DamagedProjectFixture.damagedLine)),
            "the rejected line was not named: \(result.stderr)")
    }

    @Test func projectsListsTheFileAndWarnsAboutIt() async throws {
        let fixture = try await DamagedProjectFixture()
        defer { try? FileManager.default.removeItem(at: fixture.directory) }

        let result = try run(fixture.directory, ["projects"])
        #expect(result.status == 0, "stderr: \(result.stderr)")
        #expect(result.stdout.contains("damaged-project"), "stdout: \(result.stdout)")
        #expect(result.stderr.contains("is incomplete"), "stderr: \(result.stderr)")
    }

    @Test func aCleanProjectIsNotAccused() async throws {
        let fixture = try await DamagedProjectFixture(damaged: false)
        defer { try? FileManager.default.removeItem(at: fixture.directory) }

        let result = try run(fixture.directory, ["list", "damaged-project"])
        #expect(result.status == 0, "stderr: \(result.stderr)")
        #expect(result.stderr.contains("incomplete") == false, "stderr: \(result.stderr)")
        #expect(result.stdout.contains(fixture.factValue), "stdout: \(result.stdout)")
    }
}
