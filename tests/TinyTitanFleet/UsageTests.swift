import Foundation
import Testing
import TinyTitanFleetCore

/// The command line's own contract: `--help` names every option the parser
/// reads, a value that is not a number is refused instead of discarded, and an
/// option nobody reads is an error rather than a silence.
///
/// These spawn the built executable because the parser lives in the executable
/// target, which a test cannot import. That is also why `--help` itself lives in
/// `FleetUsage` in the core library: a string inside an unimportable target is a
/// string nothing can assert about.
@Suite struct FleetUsageTests {

    private func run(_ arguments: [String]) throws
        -> (status: Int32, stdout: String, stderr: String)
    {
        let executable = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(".build/debug/ttlanmanager")
        let process = Process()
        let stdout = Pipe()
        let stderr = Pipe()
        process.executableURL = executable
        process.arguments = arguments
        process.standardOutput = stdout
        process.standardError = stderr
        // No key at all: the CLI then resolves the plugin's shipped default, so a
        // run carries no `--key puts the key in argv` warning to read as an error.
        try process.run()
        process.waitUntilExit()
        let out = stdout.fileHandleForReading.readDataToEndOfFile()
        let err = stderr.fileHandleForReading.readDataToEndOfFile()
        return (
            process.terminationStatus,
            try #require(String(bytes: out, encoding: .utf8)),
            try #require(String(bytes: err, encoding: .utf8)),
        )
    }

    /// Every `--long-form` option named in `text`, scanned the same way on both
    /// sides of the comparison so the two sets mean the same thing.
    private func optionTokens(in text: String) -> Set<String> {
        let name = Set("abcdefghijklmnopqrstuvwxyz0123456789-_")
        var found: Set<String> = []
        var index = text.startIndex
        while index < text.endIndex {
            let next = text.index(after: index)
            guard text[index] == "-", next < text.endIndex, text[next] == "-" else {
                index = text.index(after: index)
                continue
            }
            var end = text.index(next, offsetBy: 1)
            while end < text.endIndex, name.contains(text[end]) {
                end = text.index(after: end)
            }
            if end > text.index(next, offsetBy: 1) {
                found.insert(String(text[index..<end]))
            }
            index = end
        }
        return found
    }

    /// Only the options the parser *reads*: a source literal that is exactly an
    /// option name, as passed to `takeOption`, `takeFlag`, `numberOption` and the
    /// `arguments.contains` checks. Prose that names a flag — an error message, or
    /// a comment about a typo — is not a parse site, and counting it would make the
    /// assertion below measure the wrong thing.
    private func parsedOptions(in source: String) -> Set<String> {
        var found: Set<String> = []
        for match in optionTokens(in: source) where source.contains("\"\(match)\"") {
            found.insert(match)
        }
        return found
    }

    /// The parser reads an option `--help` never mentions, so nobody finds it; and
    /// `--help` promises an option the parser never reads, so it is ignored. Both
    /// are the same defect seen from either side, so both are asserted.
    @Test func helpNamesEveryOptionTheParserReads() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // TinyTitanFleet
            .deletingLastPathComponent()  // tests
            .deletingLastPathComponent()  // <root>
        let parser = root.appendingPathComponent("sources/TinyTitanFleet/Command/main.swift")
        let source = try String(contentsOf: parser, encoding: .utf8)
        let parsed = parsedOptions(in: source)
        #expect(!parsed.isEmpty, "the scan read no options out of \(parser.path)")

        let help = try run(["--help"]).stdout
        #expect(!help.isEmpty, "the built executable printed no help")
        let documented = optionTokens(in: help)
        let undocumented = parsed.subtracting(documented).sorted()
        let phantom = documented.subtracting(parsed).sorted()
        #expect(undocumented.isEmpty, "--help does not name: \(undocumented)")
        #expect(phantom.isEmpty, "--help names options the parser never reads: \(phantom)")
    }

    /// `--timeout ten` used to run at ten seconds and say nothing about the value
    /// it threw away. A number the operator meant is either honored or refused.
    @Test func anUnreadableNumberIsRefusedNotSilentlyDefaulted() throws {
        let bad = try run(["--timeout", "ten", "list", "--peer", "127.0.0.1:1"])
        #expect(bad.status != 0, "stdout: \(bad.stdout), stderr: \(bad.stderr)")
        #expect(bad.stderr.contains("--timeout"), "stderr: \(bad.stderr)")
        #expect(bad.stderr.contains("ten"), "stderr: \(bad.stderr)")

        let interval = try run(["--interval", "soon", "list", "--peer", "127.0.0.1:1"])
        #expect(interval.status != 0, "stderr: \(interval.stderr)")
        #expect(interval.stderr.contains("--interval"), "stderr: \(interval.stderr)")

        // The control: a value that *is* a number is honored, which makes the
        // refusal above about the text rather than about the flag. Port 1 has
        // nothing listening, so this can only report the peer it was given —
        // never a live fleet.
        let good = try run(["--timeout", "3", "list", "--peer", "127.0.0.1:1"])
        #expect(
            good.stderr.contains("unreachable"),
            "a valid timeout should reach the peer and fail on it: \(good.stderr)")
    }

    /// `workspace delete --workspace ID --keep-session` (one "s" short) used to
    /// delete the workspace *and* archive its sessions, because an option the
    /// parser does not know is dropped on the floor.
    @Test func anOptionNobodyReadsIsAnErrorNotASilence() throws {
        let typo = try run([
            "workspace", "delete", "--workspace", "w-1", "--keep-session",
            "--peer", "127.0.0.1:1",
        ])
        #expect(typo.status != 0, "stdout: \(typo.stdout), stderr: \(typo.stderr)")
        #expect(typo.stderr.contains("--keep-session"), "stderr: \(typo.stderr)")

        let misplaced = try run(["list", "--peer", "127.0.0.1:1", "--not-an-option"])
        #expect(misplaced.status != 0, "stderr: \(misplaced.stderr)")
        #expect(
            misplaced.stderr.contains("--not-an-option"),
            "stderr: \(misplaced.stderr)")
    }
}
