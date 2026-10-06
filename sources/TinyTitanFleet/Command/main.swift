import Foundation
import TinyTitanFleetCore

/// `ttlanmanager` — TinyTitan DSH LAN Manager: drive a group of DeepSeek Harness
/// instances from one place.
///
/// The discovery mesh belongs to the `dsh-lan-manager` plugin on each Mac; this
/// tool reads the group from any one of them and then talks to the member that
/// owns the thing being acted on. It never asks one instance to relay a prompt
/// to another — the plugin is a receiver, and this is the manager.
///
///   ttlanmanager list
///   ttlanmanager prompt --session ID --text "run the tests"
///   ttlanmanager prompt-all --text "checkpoint: summarise your state"
///   ttlanmanager workspace create --on node3 --path /Users/me/Project
///   ttlanmanager session archive --session ID
///   ttlanmanager workspace delete --workspace ID

let usage = """
    \(FleetBrand.name) (\(FleetBrand.command)) — a control plane for a group of DSH hosts.

    usage: \(FleetBrand.command) [--peer HOST[:PORT]] [--key KEY] [--json] [--timeout SECONDS] <command>

      top                                              live dashboard: every member, what it holds, act on it
      list                                             every Mac in the group, with its workspaces and sessions
      prompt --session ID --text TEXT                  prompt one session, on the Mac that owns it
      prompt-all --text TEXT [--limit N] [--concurrency N]
                                                       prompt every active session in the group
      workspace create --on NAME --path DIR [--title TITLE]
                                                       register a folder as a workspace on one Mac
      session archive --session ID                     hide a session (reversible; history kept)
      workspace delete --workspace ID [--keep-sessions]
                                                       remove a workspace from the registry

    `top` draws the group live; a scanner on its own task polls the fleet every 30 s
    by default — half the plugin's discovery period, so its polling adds at most half
    a cycle of latency — while the window keeps drawing. It resizes with the window
    and never needs more than 44x6. `--once` prints a single frame instead
    (useful in a pipe, and with --width/--height for a fixed size). `--from FILE`
    renders an inventory JSON taken earlier — or from stdin with `-` — with no fleet
    running.

    --peer is the member the group is *read* from (default 127.0.0.1:3080); --on is
    the member an action is sent to. Every action then goes directly to the Mac that
    owns it — nothing is relayed through another instance. --json prints the raw
    answer. --version prints the name. Keys resolve in this order: --key,
    DSH_LAN_KEY, DSH_LAN_TOKEN, the plugin's shipped default. Prefer the
    environment forms: --key puts the key in argv, where every other local
    account can read it with ps.
    """

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("\(FleetBrand.command): \(message)\n".utf8))
    exit(1)
}

var arguments = Array(CommandLine.arguments.dropFirst())

@MainActor func takeOption(_ name: String) -> String? {
    guard let index = arguments.firstIndex(of: name) else { return nil }
    guard index + 1 < arguments.count else { fail("\(name) needs a value") }
    let value = arguments[index + 1]
    arguments.removeSubrange(index...(index + 1))
    return value
}

@MainActor func takeFlag(_ name: String) -> Bool {
    guard let index = arguments.firstIndex(of: name) else { return false }
    arguments.remove(at: index)
    return true
}

/// Outcomes as JSON, built here rather than by hand in the middle of a `print`.
func outcomesJSON(_ outcomes: [FleetOutcome]) -> String {
    let rows = outcomes.map { outcome in
        let detail = outcome.detail.replacingOccurrences(of: "\"", with: "\\\"")
        return
            #"{"ok":\#(outcome.ok),"node":"\#(outcome.node)","sessionId":"\#(outcome.sessionId)","detail":"\#(detail)"}"#
    }
    let delivered = outcomes.filter(\.ok).count
    return
        #"{"ok":\#(delivered == outcomes.count),"delivered":\#(delivered),"considered":\#(outcomes.count),"results":[\#(rows.joined(separator: ","))]}"#
}

if arguments.contains("--help") || arguments.contains("-h") {
    print(usage)
    exit(0)
}
if arguments.contains("--version") {
    print("\(FleetBrand.name) (\(FleetBrand.command))")
    exit(0)
}

let peerText = takeOption("--peer")
let onMember = takeOption("--on")
let keyOption = takeOption("--key")
let basePath = takeOption("--base-path") ?? "/dsh-lan"
let timeout = takeOption("--timeout").flatMap(Double.init) ?? 10
let sessionOption = takeOption("--session")
let workspaceOption = takeOption("--workspace")
let textOption = takeOption("--text")
let pathOption = takeOption("--path")
let titleOption = takeOption("--title")
let limitOption = takeOption("--limit").flatMap(Int.init)
let concurrencyOption = takeOption("--concurrency").flatMap(Int.init)
let asJSON = takeFlag("--json")
let keepSessions = takeFlag("--keep-sessions")
let once = takeFlag("--once")
let widthOption = takeOption("--width").flatMap(Int.init)
let heightOption = takeOption("--height").flatMap(Int.init)
let intervalOption = takeOption("--interval").flatMap(Int.init)
let fromOption = takeOption("--from")

let environment = ProcessInfo.processInfo.environment
let keyResolution = FleetGroupKey.resolve(option: keyOption, environment: environment)
let key = keyResolution.key
if let warning = keyResolution.warning {
    FileHandle.standardError.write(Data("\(FleetBrand.command): \(warning)\n".utf8))
}

guard let command = arguments.first else {
    print(usage)
    exit(2)
}
arguments.removeFirst()

let seedText = peerText ?? "127.0.0.1:3080"
guard let seed = FleetTarget(text: seedText, defaultPort: 3080) else {
    fail("--peer does not name a usable host: \(seedText)")
}

let runner = FleetRunner(
    client: FleetClient(
        token: key,
        basePath: basePath,
        transport: URLSessionTransport(timeout: timeout)
    ))

@MainActor func report(_ ack: FleetAck) {
    print(asJSON ? ack.raw : (ack.ok ? "ok" : "refused"))
}

@MainActor func requireSession() -> String {
    guard let sessionOption else { fail("\(command) needs --session ID") }
    return sessionOption
}

@MainActor func requireText() -> String {
    guard let textOption, !textOption.isEmpty else { fail("\(command) needs --text TEXT") }
    return textOption
}

/// Notes the UI has to pick up from work it did not wait for.
///
/// An action's network calls run on their own task so the dashboard keeps
/// drawing; the result lands here and the next frame shows it. An actor rather
/// than a shared variable, because the UI and the action are different tasks.
actor FleetNotes {
    private var pending: String?

    func put(_ message: String) { pending = message }

    func take() -> String? {
        defer { pending = nil }
        return pending
    }
}

/// The live dashboard: draw, take keys, and let a scanner keep the group fresh.
///
/// Nothing here waits on the network. The scanner polls on its own task and each
/// frame reads its latest snapshot, so an unreachable member costs a timeout in
/// the background rather than a frozen window — which is what the earlier
/// synchronous refresh got wrong.
@MainActor
func runDashboard(runner: FleetRunner, seed: FleetTarget, scanSeconds: Int) async {
    guard let terminal = FleetTerminal() else {
        fail("top needs an interactive terminal — use `list` when piping, or `top --once`")
    }
    let scanner = FleetScanner(runner: runner, seed: seed, intervalSeconds: scanSeconds)
    let notes = FleetNotes()
    await scanner.start()

    var dashboard = FleetDashboard(seed: seed)
    var appliedScan: Date?
    var running = true

    while running {
        let snapshot = await scanner.current()
        if snapshot.scannedAt != appliedScan {
            dashboard.apply(group: snapshot.group)
            if !snapshot.newMembers.isEmpty {
                dashboard.setStatus("joined: \(snapshot.newMembers.joined(separator: ", "))")
            }
            appliedScan = snapshot.scannedAt
        }
        if let failure = snapshot.failure { dashboard.apply(failure: failure) }
        if let note = await notes.take() { dashboard.setStatus(note) }

        let size = terminal.size()
        terminal.draw(FleetDashboardView.render(dashboard, width: size.columns, height: size.rows))

        for key in terminal.readKeys(timeoutMs: 200) {
            dashboard.press(key)
        }

        guard let action = dashboard.takePendingAction() else { continue }
        if case .quit = action {
            running = false
            continue
        }
        if case .refresh = action {
            await scanner.requestScan()
            dashboard.setStatus("scanning…")
            continue
        }
        // Mutating and prompting run off the frame: the fleet can be slow and the
        // window should not be.
        let group = dashboard.group
        dashboard.setStatus("working…")
        Task {
            let message = await runner.perform(action, in: group)
            await notes.put(message)
            await scanner.requestScan()
        }
    }
    await scanner.stop()
    terminal.restore()
}

/// Read the group from a fixture when one was given, otherwise over the network.
///
/// An explicit branch rather than `??`: `a ?? b` evaluates both sides here, which
/// would dial the fleet even when a fixture was supplied.
func loadGroup(runner: FleetRunner, seed: FleetTarget, from path: String?) async throws -> FleetRead
{
    if let path {
        return try localRead(path: path, seed: seed)
    }
    return try await runner.read(seed: seed)
}

/// Read an inventory from a file, or stdin with `-`.
///
/// The same JSON a member answers `/inventory` with, so a snapshot can be
/// rendered, reviewed or diffed with no fleet running.
func localRead(path: String, seed: FleetTarget) throws -> FleetRead {
    let data =
        path == "-"
        ? try FleetInventorySource.readStandardInput()
        : try FleetInventorySource.read(fileAt: URL(fileURLWithPath: path))
    let inventory = try JSONDecoder().decode(FleetInventory.self, from: data)
    return FleetRead(group: FleetRunner.assemble(inventory: inventory, seed: seed), raw: data)
}

do {
    switch command {
    case "top", "ui", "dashboard":
        if once {
            let read = try await loadGroup(runner: runner, seed: seed, from: fromOption)
            var dashboard = FleetDashboard(seed: seed)
            dashboard.apply(group: read.group)
            let frame = FleetDashboardView.render(
                dashboard,
                width: widthOption ?? 100,
                height: heightOption ?? 30
            )
            print(frame.lines.joined(separator: "\n"))
        } else {
            await runDashboard(
                runner: runner, seed: seed, scanSeconds: max(2, intervalOption ?? 30))
        }

    case "list":
        let read = try await loadGroup(runner: runner, seed: seed, from: fromOption)
        print(
            asJSON
                ? String(data: read.raw, encoding: .utf8) ?? "{}"
                : FleetRenderer.text(read.group))

    case "prompt":
        let read = try await runner.read(seed: seed)
        let outcome = try await runner.prompt(
            group: read.group, sessionId: requireSession(), text: requireText())
        print(asJSON ? outcomesJSON([outcome]) : FleetRenderer.outcomes([outcome]))
        exit(outcome.ok ? 0 : 1)

    case "prompt-all":
        let read = try await runner.read(seed: seed)
        let outcomes = await runner.promptAll(
            group: read.group,
            text: requireText(),
            limit: limitOption,
            concurrency: concurrencyOption ?? 4
        )
        print(asJSON ? outcomesJSON(outcomes) : FleetRenderer.outcomes(outcomes))
        exit(outcomes.contains { !$0.ok } ? 1 : 0)

    case "workspace":
        guard let sub = arguments.first else { fail("workspace needs create or delete") }
        arguments.removeFirst()
        let read = try await runner.read(seed: seed)
        switch sub {
        case "create":
            guard let pathOption else { fail("workspace create needs --path DIR") }
            guard let onMember else {
                fail("workspace create needs --on NAME (the Mac to register it on)")
            }
            report(
                try await runner.createWorkspace(
                    group: read.group, node: onMember, path: pathOption, title: titleOption))
        case "delete":
            guard let workspaceOption else { fail("workspace delete needs --workspace ID") }
            report(
                try await runner.deleteWorkspace(
                    group: read.group, workspaceId: workspaceOption, archiveSessions: !keepSessions)
            )
        default:
            fail("workspace needs create or delete, not \(sub)")
        }

    case "session":
        guard let sub = arguments.first else { fail("session needs archive") }
        arguments.removeFirst()
        guard sub == "archive" else { fail("session needs archive, not \(sub)") }
        let read = try await runner.read(seed: seed)
        report(try await runner.archive(group: read.group, sessionId: requireSession()))

    default:
        fail("unknown command: \(command)")
    }
} catch {
    fail((error as? FleetError)?.description ?? "\(error)")
}
