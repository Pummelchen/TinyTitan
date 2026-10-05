import Foundation
import Testing

@Suite(.serialized)
struct RepackCLITests {
    @Test func resumeAndDiscardAreMutuallyExclusive() throws {
        let output = temporaryOutput("exclusive")
        defer { clean(output) }
        let result = try run([
            "--output", output,
            "--resume",
            "--discard-partial",
        ])

        #expect(result.status == 2)
        #expect(result.stderr.contains("mutually exclusive"))
    }

    @Test func resumeWithoutStateFailsBeforeNetwork() throws {
        let output = temporaryOutput("missing-resume")
        defer { clean(output) }
        let result = try run([
            "--output", output,
            "--resume",
        ])

        #expect(result.status == 1)
        #expect(result.stderr.contains("no resumable install state exists"))
    }

    @Test func discardWithoutStateReportsAnError() throws {
        let output = temporaryOutput("missing-discard")
        defer { clean(output) }
        let result = try run([
            "--discard-partial",
            "--output", output,
        ])

        #expect(result.status == 1)
        #expect(result.stderr.contains("no resumable install state exists"))
    }

    @Test func unknownModelSelectorIsRejected() throws {
        let output = temporaryOutput("bad-model")
        defer { clean(output) }
        let result = try run([
            "--model", "bogus",
            "--output", output,
        ])

        #expect(result.status == 2)
        #expect(result.stderr.contains("unknown model"))
        #expect(result.stderr.contains("qwen36"))
    }

    @Test func qwenModelSelectorIsAccepted() throws {
        let output = temporaryOutput("qwen-model")
        defer { clean(output) }
        // --resume without saved state fails fast after argument parsing,
        // proving the selector itself is accepted without touching the network.
        let result = try run([
            "--model", "qwen36",
            "--output", output,
            "--resume",
        ])

        #expect(result.status == 1)
        #expect(result.stderr.contains("no resumable install state exists"))
    }

    /// A local snapshot import reaches the copy phase without any network, so
    /// the CLI must report progress even when stdout is a pipe (no tty), which
    /// is how an install redirected into a log file is seen.
    @Test func localImportReportsProgressToAPipe() throws {
        let root = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("tinytitanrepack-progress-\(UUID().uuidString)")
        let snapshot = (root as NSString).appendingPathComponent("snapshot")
        let output = (root as NSString).appendingPathComponent("model.ssdai")
        defer { try? FileManager.default.removeItem(atPath: root) }
        _ = try SyntheticSnapshot.buildQwenMTP(at: snapshot)

        let result = try run([
            "--input-snapshot", snapshot,
            "--model-id", "cli-progress-fixture",
            "--output", output,
        ])

        #expect(result.status == 0, "stderr: \(result.stderr)")
        #expect(
            result.stdout.contains("installing") || result.stdout.contains("%"),
            "stdout: \(result.stdout)")
        // The final line must be terminated: the summary the CLI prints next
        // must not be appended to the progress line.
        #expect(
            result.stdout.contains("100%\n"),
            "stdout: \(result.stdout)")
    }

    private func run(_ arguments: [String]) throws
        -> (status: Int32, stdout: String, stderr: String)
    {
        let executable = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(".build/debug/TinyTitanRepack")
        let process = Process()
        let stdout = Pipe()
        let stderr = Pipe()
        process.executableURL = executable
        process.arguments = arguments
        process.standardOutput = stdout
        process.standardError = stderr
        // The progress assertion needs the display on: a developer running the
        // suite with TINYTITAN_NO_PROGRESS exported must not flip it off.
        var environment = ProcessInfo.processInfo.environment
        environment.removeValue(forKey: "TINYTITAN_NO_PROGRESS")
        process.environment = environment
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

    private func temporaryOutput(_ tag: String) -> String {
        (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("tinytitanrepack-\(tag)-\(UUID().uuidString).ssdai")
    }

    private func clean(_ output: String) {
        for path in [
            output,
            output + ".partial",
            output + ".install-state",
            output + ".install-state.cleanup",
            output + ".install.lock",
        ] {
            try? FileManager.default.removeItem(atPath: path)
        }
    }
}
