import Darwin
import Foundation
import NIOCore
import Testing
@testable import TinyTitanCore

@testable import TinyTitan
@testable import TinyTitanLib
@testable import TinyTitanServerCore

// The real coding CLIs against the server, with the model replaced by a
// scripted backend. This is the check the unit suites cannot make: that
// Codex's Responses client and Claude Code's Messages client accept what
// this server sends, end to end, including every request field they add
// that the unit tests never wrote.
//
// A CLI is present at a fixed install path or it is not. Presence alone is not
// runnability: a half-installed Homebrew `codex` could not load its node
// dependency, and the old print/return reported the test *passed* with zero
// assertions when absent and failed the whole suite when broken. The tests are
// gated with `.enabled(if:)` on a bounded `--version` probe, so the framework
// records a real skip; a working CLI still runs the full exercise.

private actor GreetingBackend: ServerInferenceBackend, PromptTokenCounting {
    private let lock = NSLock()
    nonisolated(unsafe) private var seen: [ValidatedChatRequest] = []

    nonisolated var requests: [ValidatedChatRequest] { lock.withLock { seen } }

    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        lock.withLock { seen.append(request) }
        onEvent(.content("hello from tinytitan"))
        return ServerCompletion(
            content: "hello from tinytitan", toolCalls: [], finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 20, completionTokens: 4, totalTokens: 24))
    }

    func countPromptTokens(_ request: ValidatedChatRequest) async throws -> Int { 20 }
}

private func binary(_ name: String) -> String? {
    let candidates = [
        "/opt/homebrew/bin/\(name)",
        "/usr/local/bin/\(name)",
        NSHomeDirectory() + "/.local/bin/\(name)",
    ]
    return candidates.first { FileManager.default.isExecutableFile(atPath: $0) }
}

/// Whether `path` runs at all: a bounded `--version` probe, with no output kept
/// and no inherited stdin. A half-installed CLI fails here instead of taking the
/// suite down at its first real turn.
func cliIsUsable(at path: String, timeout: TimeInterval = 5) -> Bool {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: path)
    process.arguments = ["--version"]
    process.standardOutput = FileHandle.nullDevice
    process.standardError = FileHandle.nullDevice
    process.standardInput = FileHandle.nullDevice
    do {
        try process.run()
    } catch {
        return false
    }
    let deadline = Date().addingTimeInterval(timeout)
    while process.isRunning, Date() < deadline {
        usleep(50_000)
    }
    if process.isRunning {
        process.terminate()
        return false
    }
    return process.terminationStatus == 0
}

/// The predicate the `.enabled(if:)` traits use: installed *and* runnable.
func usableCLI(_ name: String) -> Bool {
    guard let path = binary(name) else { return false }
    return cliIsUsable(at: path)
}

/// Run a CLI with a timeout; returns stdout+stderr and the exit status.
private func run(
    _ executable: String, _ arguments: [String],
    environment: [String: String], directory: URL,
    timeout: TimeInterval = 180
) async throws -> (output: String, status: Int32) {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: executable)
    process.arguments = arguments
    process.environment = ProcessInfo.processInfo.environment.merging(environment) { $1 }
    process.currentDirectoryURL = directory
    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = pipe
    process.standardInput = FileHandle.nullDevice
    try process.run()
    let reader = Task.detached { pipe.fileHandleForReading.readDataToEndOfFile() }
    let deadline = Date().addingTimeInterval(timeout)
    while process.isRunning, Date() < deadline {
        try await Task.sleep(for: .milliseconds(200))
    }
    if process.isRunning { process.terminate() }
    let data = await reader.value
    return (data.lossyUTF8String, process.terminationStatus)
}

@Suite("Coding CLIs against the server", .serialized)
struct ClientCLITests {
    @Test(.enabled(if: usableCLI("codex"))) func codexSpeaksTheResponsesAPI() async throws {
        let codex = try #require(binary("codex"), "codex disappeared after the probe")
        let backend = GreetingBackend()
        let server = TinyTitanHTTPServer(modelID: "test-model", queueLimit: 2, backend: backend)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("tinytitan-codex-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let result = try await run(
            codex,
            [
                "exec", "--skip-git-repo-check", "-s", "read-only",
                "-c", "model_providers.tinytitan.name=TinyTitan",
                "-c", "model_providers.tinytitan.base_url=http://127.0.0.1:\(port)/v1",
                "-c", "model_providers.tinytitan.wire_api=responses",
                "-c", "model_provider=tinytitan",
                "-m", "test-model",
                "Reply with one word.",
            ], environment: ["OPENAI_API_KEY": "unused", "CODEX_HOME": directory.path],
            directory: directory)
        try await server.shutdown()

        let requests = backend.requests
        #expect(
            !requests.isEmpty,
            "codex never reached the server (exit \(result.status)): \(result.output)")
        #expect(
            result.output.contains("hello from tinytitan"),
            result.output.isEmpty ? "no output" : Comment(rawValue: result.output))
        if let first = requests.first {
            // Codex sends its instructions as a leading system message and
            // its tool suite as function tools; both must have mapped.
            #expect(first.messages.first?.role == .system)
            #expect(first.messages.contains { $0.role == .user })
        }
    }

    @Test(.enabled(if: usableCLI("claude"))) func claudeCodeSpeaksTheMessagesAPI() async throws {
        let claude = try #require(binary("claude"), "claude disappeared after the probe")
        let backend = GreetingBackend()
        let server = TinyTitanHTTPServer(modelID: "test-model", queueLimit: 2, backend: backend)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("tinytitan-claude-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let result = try await run(
            claude,
            [
                "-p", "Reply with one word.", "--output-format", "text",
                "--model", "test-model",
            ],
            environment: [
                "ANTHROPIC_BASE_URL": "http://127.0.0.1:\(port)",
                "ANTHROPIC_API_KEY": "unused",
                "ANTHROPIC_MODEL": "test-model",
                "ANTHROPIC_SMALL_FAST_MODEL": "test-model",
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": "test-model",
                "ANTHROPIC_DEFAULT_SONNET_MODEL": "test-model",
                "ANTHROPIC_DEFAULT_OPUS_MODEL": "test-model",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "DISABLE_TELEMETRY": "1",
            ], directory: directory)
        try await server.shutdown()

        let requests = backend.requests
        #expect(
            !requests.isEmpty,
            "claude never reached the server (exit \(result.status)): \(result.output)")
        #expect(
            result.output.contains("hello from tinytitan"),
            result.output.isEmpty ? "no output" : Comment(rawValue: result.output))
        if let first = requests.first {
            #expect(first.messages.first?.role == .system)
            #expect(first.messages.contains { $0.role == .user })
        }
    }

    /// The gate itself, without depending on which CLIs this machine has: a
    /// working probe is usable, a failing one is not, a missing path is not, and
    /// a CLI that never answers `--version` is terminated at the deadline rather
    /// than hanging test discovery.
    @Test func theCLIProbeDistinguishesBrokenFromWorking() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("tinytitan-cli-probe-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        func script(_ name: String, _ body: String) throws -> String {
            let path = directory.appendingPathComponent(name).path
            try body.write(toFile: path, atomically: true, encoding: .utf8)
            try FileManager.default.setAttributes(
                [.posixPermissions: 0o755], ofItemAtPath: path)
            return path
        }

        let working = try script("working", "#!/bin/sh\nexit 0\n")
        let failing = try script("failing", "#!/bin/sh\nexit 1\n")
        let stalled = try script("stalled", "#!/bin/sh\nexec sleep 30\n")

        #expect(FileManager.default.isExecutableFile(atPath: working))
        #expect(cliIsUsable(at: working, timeout: 2))
        #expect(!cliIsUsable(at: failing, timeout: 2))
        #expect(!cliIsUsable(at: directory.appendingPathComponent("missing").path))
        #expect(!cliIsUsable(at: stalled, timeout: 0.2))
    }
}

/// The requested lifetime of the manual stub server, or `nil` when the ordinary
/// run should not start one.
private func stubServerSeconds() -> Double? {
    ProcessInfo.processInfo.environment["TINYTITAN_STUB_SERVER_SECONDS"]
        .flatMap(Double.init)
}

/// Not a test of anything: with TINYTITAN_STUB_SERVER_SECONDS set to a number of
/// seconds, keeps a scripted server up for that long and prints its port, so a
/// CLI can be run against it by hand while debugging a client's request grammar.
/// Gated rather than guarded in the body: the ordinary run has to *record* that
/// this did not run, and an early return there reports a passed test that did
/// nothing.
@Suite struct StubServerForManualRuns {
    @Test(.enabled(if: stubServerSeconds() != nil))
    func stubServer() async throws {
        let seconds = try #require(stubServerSeconds())
        let backend = GreetingBackend()
        let server = TinyTitanHTTPServer(modelID: "test-model", queueLimit: 2, backend: backend)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        let file =
            ProcessInfo.processInfo.environment["TINYTITAN_STUB_SERVER_PORT_FILE"]
            ?? "/tmp/tinytitan-stub-port"
        try String(port).write(toFile: file, atomically: true, encoding: .utf8)
        print("stub server on port \(port) for \(seconds)s")
        try await Task.sleep(for: .seconds(seconds))
        for (index, request) in backend.requests.enumerated() {
            print(
                "request \(index): roles \(request.messages.map(\.role)) tools \(request.tools.count) max \(request.maximumCompletionTokens)"
            )
        }
        try await server.shutdown()
    }
}
