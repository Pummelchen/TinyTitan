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

    // MARK: - --draft-head and --share-ngram-table need --input-snapshot

    /// Both flags are fields of `LocalSnapshotRepackOptions`: one takes only the
    /// draft head's tensors, the other hardlinks the snapshot's 102 GB n-gram
    /// table instead of rebuilding it. The download route has no parameter for
    /// either — `SupportedModelSource.installOptions(outputDirectory:overwrite:
    /// token:resume:)` — so it used to accept them, install the default shape,
    /// and exit 0.
    ///
    /// `--input-ssdai` is in these arguments so no tree can reach the network:
    /// the guard answers before dispatch, and the download it would otherwise
    /// start is the 36.9 GB one the rules here forbid fetching to satisfy a test.
    @Test func draftHeadWithoutASnapshotIsRefused() throws {
        let output = temporaryOutput("draft-head")
        defer { clean(output) }
        let result = try run([
            "--draft-head",
            "--output", output,
            "--input-ssdai", output,
        ])

        #expect(result.status == 2, "stderr: \(result.stderr)")
        #expect(
            result.stderr.contains("apply only to --input-snapshot"),
            "stderr: \(result.stderr)")
    }

    /// `--discard-partial` returns from `parse` before the import branch, so
    /// this is the route that shows the guard is not simply another clause of
    /// the local import's own checks. Before the fix the flag was set, never
    /// read, and the discard ran as if nothing had been asked.
    @Test func shareNgramTableWithDiscardPartialIsRefused() throws {
        let output = temporaryOutput("share-discard")
        defer { clean(output) }
        let result = try run([
            "--discard-partial",
            "--output", output,
            "--share-ngram-table",
        ])

        #expect(result.status == 2, "stderr: \(result.stderr)")
        #expect(
            result.stderr.contains("apply only to --input-snapshot"),
            "stderr: \(result.stderr)")
    }

    /// The receipt path is the third branch that ignored them.
    @Test func draftHeadWithVerifyInstallIsRefused() throws {
        let output = temporaryOutput("draft-head-verify")
        defer { clean(output) }
        let result = try run([
            "--verify-install",
            "--input-ssdai", output,
            "--draft-head",
        ])

        #expect(result.status == 2, "stderr: \(result.stderr)")
        #expect(
            result.stderr.contains("apply only to --input-snapshot"),
            "stderr: \(result.stderr)")
    }

    /// The counter-tests, and the reason the three above are not a guard that
    /// refuses everything: with a snapshot to act on, each flag is still
    /// accepted. This one is a real import — a synthetic Qwen3.8-MTP snapshot,
    /// no network — so the flag is not merely parsed, it reaches the copy phase
    /// and the install completes.
    @Test func shareNgramTableWithASnapshotStillImports() throws {
        let root = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("tinytitanrepack-shareflag-\(UUID().uuidString)")
        let snapshot = (root as NSString).appendingPathComponent("snapshot")
        let output = (root as NSString).appendingPathComponent("model.ssdai")
        defer { try? FileManager.default.removeItem(atPath: root) }
        _ = try SyntheticSnapshot.buildQwenMTP(at: snapshot)

        let result = try run([
            "--input-snapshot", snapshot,
            "--model-id", "cli-shareflag-fixture",
            "--output", output,
            "--share-ngram-table",
        ])

        #expect(result.status == 0, "stderr: \(result.stderr)")
        #expect(
            result.stdout.contains("Imported local snapshot"),
            "stdout: \(result.stdout)")
    }

    /// `--draft-head` needs a Qwen3.8-Flash-Next *target* config to derive the
    /// draft arch from (`ArchInfo.qwen38FlashNextMTP`, via
    /// `LocalSnapshotLoader.load(directory:draftHead:)`), and every fixture here
    /// is qwen3_5_moe or qwen3_5_mtp shaped, so the import cannot be run to
    /// completion without a target-sized synthetic snapshot. What this pins is
    /// the part the guard could break: the flag is accepted when a snapshot is
    /// named, and the CLI fails later and elsewhere -- measured as
    /// `MTP draft requires a Qwen3.8-Flash-Next target`, with exit 1 and no
    /// refusal from the parse stage.
    @Test func draftHeadWithASnapshotReachesTheImporter() throws {
        let root = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("tinytitanrepack-drafthead-\(UUID().uuidString)")
        let snapshot = (root as NSString).appendingPathComponent("snapshot")
        let output = (root as NSString).appendingPathComponent("model.ssdai")
        defer { try? FileManager.default.removeItem(atPath: root) }
        _ = try SyntheticSnapshot.buildQwenMTP(at: snapshot)

        let result = try run([
            "--input-snapshot", snapshot,
            "--model-id", "cli-drafthead-fixture",
            "--output", output,
            "--draft-head",
        ])

        #expect(
            !result.stderr.contains("apply only to --input-snapshot"),
            "the guard fired on a command that did name a snapshot: \(result.stderr)")
        #expect(result.status != 2, "stderr: \(result.stderr)")
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
