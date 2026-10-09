import Foundation

/// What the scanner currently knows.
public struct FleetSnapshot: Sendable, Equatable {
    public let group: FleetGroup
    public let scannedAt: Date?
    public let scanning: Bool
    public let failure: String?
    /// Members that appeared in the most recent scan, by name.
    public let newMembers: [String]
    /// Members that answered when asked directly, out of those polled.
    public let answered: Int
    public let polled: Int
    public let intervalSeconds: Int

    public init(
        group: FleetGroup = FleetGroup(), scannedAt: Date? = nil, scanning: Bool = false,
        failure: String? = nil, newMembers: [String] = [], answered: Int = 0,
        polled: Int = 0, intervalSeconds: Int = 30
    ) {
        self.group = group
        self.scannedAt = scannedAt
        self.scanning = scanning
        self.failure = failure
        self.newMembers = newMembers
        self.answered = answered
        self.polled = polled
        self.intervalSeconds = intervalSeconds
    }
}

/// Polls the fleet for members on its own task, so the dashboard never waits.
///
/// Two things this does that a single poll of the seed cannot:
///
/// 1. **It runs off the UI.** The dashboard draws and takes keys while a scan is
///    in flight; an unreachable member costs the scan a timeout, not a frame.
/// 2. **It asks every member directly.** One member's answer about another is a
///    cached view; the member's own answer is the truth, and it may name members
///    the seed has not learned yet. That is what makes a new machine appear
///    within one scan interval rather than waiting for the whole mesh.
///
/// The interval defaults to **half** the plugin's discovery period: the manager
/// reads no faster than members are found, but its own polling adds no more than
/// half a discovery cycle of latency on top.
public actor FleetScanner {
    private let runner: FleetRunner
    private let seed: FleetTarget
    private let intervalSeconds: Int
    private let concurrency: Int
    private let sleep: @Sendable (Duration) async -> Void
    private var state: FleetSnapshot
    private var task: Task<Void, Never>?
    private var inFlight: Task<Void, Never>?
    /// Set by `stop()` and never cleared. It refuses `requestScan()`, which is
    /// the one entry point something other than the owner can reach after the
    /// teardown; `start()` is the owner speaking again and is not refused.
    private var stopped = false

    public init(
        runner: FleetRunner,
        seed: FleetTarget,
        intervalSeconds: Int = 30,
        concurrency: Int = 4,
        sleep: @escaping @Sendable (Duration) async -> Void = { try? await Task.sleep(for: $0) }
    ) {
        self.runner = runner
        self.seed = seed
        self.intervalSeconds = max(2, intervalSeconds)
        self.concurrency = max(1, concurrency)
        self.sleep = sleep
        self.state = FleetSnapshot(intervalSeconds: max(2, intervalSeconds))
    }

    /// The latest snapshot. A cheap actor hop — the dashboard calls it every frame.
    public func current() -> FleetSnapshot { state }

    /// Begin polling. Idempotent.
    public func start() {
        guard task == nil else { return }
        task = Task { [intervalSeconds, sleep] in
            while !Task.isCancelled {
                await self.scan()
                await sleep(.seconds(intervalSeconds))
            }
        }
    }

    /// Stop polling and cancel any scan in flight.
    ///
    /// Cancels rather than waits, and a synchronous actor method could not wait
    /// if it meant to: the one caller is the dashboard's teardown, and a scan
    /// that asked a dead peer sits out the full timeout, so waiting here would
    /// hold the terminal before `restore()`. What that costs is that an
    /// in-flight scan can still publish a snapshot afterwards, which nothing
    /// reads at that point.
    ///
    /// Stopping is the last word for work the owner did not ask for. `start()`
    /// after it polls again, because that is the owner asking; `requestScan()`
    /// does not, because the dashboard runs a mutating action on its own task
    /// and refreshes when it finishes, so one can land after the teardown and
    /// dial the fleet behind a restored terminal.
    public func stop() {
        stopped = true
        task?.cancel()
        task = nil
        inFlight?.cancel()
        inFlight = nil
    }

    /// Scan now and wait, on the caller's task.
    ///
    /// No production caller: the dashboard either polls or kicks, and
    /// `top --once` renders a file rather than dialling. This is the
    /// synchronous seam the scanner's own tests drive.
    public func scanNow() async {
        await scan()
    }

    /// Kick a scan without waiting: after a mutation, the next frame picks up the
    /// result instead of the UI sitting on a spinner.
    public func requestScan() {
        guard !stopped, inFlight == nil else { return }
        inFlight = Task { [weak self] in
            await self?.scan()
            await self?.clearInFlight()
        }
    }

    private func clearInFlight() { inFlight = nil }

    /// One full scan: read the seed, then ask every member it names.
    private func scan() async {
        state = FleetSnapshot(
            group: state.group, scannedAt: state.scannedAt, scanning: true,
            failure: state.failure, newMembers: [], answered: 0, polled: 0,
            intervalSeconds: intervalSeconds
        )
        do {
            let seedRead = try await runner.read(seed: seed)
            let members = seedRead.group.nodes
            let reports = await askEveryMember(members)
            let merged = Self.merge(seed: seedRead.group, reports: reports.compactMap { $0 })
            let known = Set(state.group.nodes.map(\.name))
            let appeared = merged.nodes.filter {
                !known.contains($0.name) && state.scannedAt != nil
            }
            state = FleetSnapshot(
                group: merged,
                scannedAt: Date(),
                scanning: false,
                failure: nil,
                newMembers: appeared.map(\.name),
                answered: reports.compactMap { $0 }.count,
                polled: members.count,
                intervalSeconds: intervalSeconds
            )
        } catch {
            // The previous group stays: a fleet view that empties when one member
            // is unreachable is worse than one that says so.
            state = FleetSnapshot(
                group: state.group, scannedAt: state.scannedAt, scanning: false,
                failure: (error as? FleetError)?.description ?? "\(error)",
                newMembers: [], answered: 0, polled: 0, intervalSeconds: intervalSeconds
            )
        }
    }

    /// Ask each member for its own inventory, in bounded batches.
    private func askEveryMember(_ members: [FleetNode]) async -> [FleetGroup?] {
        let batchSize = max(1, concurrency)
        var reports: [FleetGroup?] = []
        for start in stride(from: 0, to: members.count, by: batchSize) {
            let batch = Array(members[start..<min(start + batchSize, members.count)])
            let runner = self.runner
            let answered = await withTaskGroup(of: FleetGroup?.self) { group in
                for node in batch {
                    group.addTask {
                        // A member that does not answer is simply not merged; the
                        // seed's cached view of it stays.
                        try? await runner.read(node: node)
                    }
                }
                var collected: [FleetGroup?] = []
                for await report in group { collected.append(report) }
                return collected
            }
            reports.append(contentsOf: answered)
        }
        return reports
    }

    /// Fold the seed's view together with what each member said about itself.
    ///
    /// Members are matched **by machine name and port**, not by id: the same
    /// machine is `macbook-ab:3080` in its own answer and `100.114.69.128:3080`
    /// in a peer's view of it, and keying on the id would list one Mac twice.
    /// A member's own report replaces the cached view of it; its peers are added
    /// only if nothing else already names them.
    public static func merge(seed: FleetGroup, reports: [FleetGroup]) -> FleetGroup {
        var byKey: [String: FleetNode] = [:]
        var order: [String] = []

        func key(_ node: FleetNode) -> String {
            "\(node.name.lowercased()):\(node.port)"
        }
        func add(_ node: FleetNode, replace: Bool) {
            let identifier = key(node)
            if byKey[identifier] == nil {
                byKey[identifier] = node
                order.append(identifier)
            } else if replace {
                byKey[identifier] = node
            }
        }

        for node in seed.nodes { add(node, replace: false) }
        for report in reports {
            for node in report.nodes { add(node, replace: node.isSelf) }
        }
        return FleetGroup(group: seed.group, nodes: order.compactMap { byKey[$0] })
    }
}
