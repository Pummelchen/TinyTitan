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

    /// A number that parses but cannot mean what the flag is for. `--limit 0` used
    /// to widen the fan-out to every session in the group — the opposite of a cap —
    /// `--width 0` used to print an empty frame and exit 0, and `--timeout 0` handed
    /// `URLSession` a timeout of zero, which means no timeout at all: measured
    /// against a member that accepts and never answers, the run was still hanging at
    /// 14 s while `--timeout 2` reported the timeout in 3 s. A count, a size and a
    /// duration are positive or the run refuses.
    @Test func aNumberThatCannotMeanWhatTheFlagSaysIsRefused() throws {
        let cases: [(flag: String, value: String, arguments: [String])] = [
            ("--limit", "0", ["prompt-all", "--text", "hi", "--peer", "127.0.0.1:1"]),
            ("--limit", "-1", ["prompt-all", "--text", "hi", "--peer", "127.0.0.1:1"]),
            ("--concurrency", "0", ["prompt-all", "--text", "hi", "--peer", "127.0.0.1:1"]),
            ("--width", "0", ["top", "--once", "--peer", "127.0.0.1:1"]),
            ("--height", "-5", ["top", "--once", "--peer", "127.0.0.1:1"]),
            ("--timeout", "0", ["list", "--peer", "127.0.0.1:1"]),
            ("--timeout", "-5", ["list", "--peer", "127.0.0.1:1"]),
        ]
        for each in cases {
            let bad = try run(each.arguments + [each.flag, each.value])
            #expect(
                bad.status != 0,
                "\(each.flag) \(each.value) exited 0: stdout \(bad.stdout) stderr \(bad.stderr)")
            #expect(
                bad.stderr.contains(each.flag),
                "\(each.flag) \(each.value) stderr: \(bad.stderr)")
            #expect(
                bad.stderr.contains("positive"),
                "\(each.flag) \(each.value) stderr: \(bad.stderr)")
            #expect(
                bad.stderr.contains(each.value),
                "\(each.flag) \(each.value) stderr: \(bad.stderr)")
        }

        // The control is the same command with a value that *can* mean a cap: it
        // reaches the peer, so the refusals above are about the number, not the flag.
        let bounded = try run([
            "prompt-all", "--text", "hi", "--peer", "127.0.0.1:1", "--limit", "1",
            "--concurrency", "1",
        ])
        #expect(
            bounded.stderr.contains("unreachable"),
            "a valid limit should reach the peer and fail on it: \(bounded.stderr)")
    }

    /// `--from` is the option that says "read this inventory instead of dialing a
    /// fleet", and `--help` promises it with no fleet running. Two arms honor it;
    /// the rest call `runner.read` and never look at the value, so a snapshot the
    /// operator named is dropped and the run polls the peer they asked it not to
    /// contact. The stray-option guard cannot catch this, because `--from` is a
    /// known option — it is only unread.
    @Test func anInventoryIsOnlyAcceptedWhereAnInventoryIsRead() throws {
        let fixture = FileManager.default.temporaryDirectory
            .appendingPathComponent("ttlanmanager-inventory-\(UUID().uuidString).json")
        try Data(#"{"ok":true}"#.utf8).write(to: fixture)
        defer { try? FileManager.default.removeItem(at: fixture) }
        let path = fixture.path

        // The two arms that read it, asserted as working so the rule below cannot
        // be satisfied by refusing `--from` everywhere.
        let listed = try run(["list", "--from", path, "--peer", "127.0.0.1:1"])
        #expect(listed.status == 0, "list --from: \(listed.stderr)")
        let frame = try run(["top", "--once", "--from", path, "--peer", "127.0.0.1:1"])
        #expect(frame.status == 0, "top --once --from: \(frame.stderr)")

        for arguments in [
            ["prompt", "--session", "s-1", "--text", "hi"],
            ["prompt-all", "--text", "hi"],
            ["workspace", "delete", "--workspace", "w-1"],
            ["session", "archive", "--session", "s-1"],
        ] {
            let ignored = try run(arguments + ["--from", path, "--peer", "127.0.0.1:1"])
            #expect(
                ignored.status != 0,
                "\(arguments) read the peer, not the file: stdout \(ignored.stdout)")
            #expect(
                ignored.stderr.contains("--from"),
                "\(arguments) stderr named no option: \(ignored.stderr)")
        }

        // The live dashboard is the case the help text is closest to: `top` draws the
        // group, and a file named beside it was never opened. A spawned `top` has no
        // terminal, so before the fix it died at the guard inside `runDashboard` —
        // which is why this needs no deadline, and why the name of the option in
        // stderr is the whole assertion: the refusal sits ahead of that guard.
        let live = try run(["top", "--from", path, "--peer", "127.0.0.1:1"])
        #expect(live.status != 0, "stdout: \(live.stdout), stderr: \(live.stderr)")
        #expect(live.stderr.contains("--from"), "stderr: \(live.stderr)")
    }
}
