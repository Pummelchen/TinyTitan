import Foundation
import Testing

@testable import TinyTitanFleetCore

/// A transport that answers per host, so the scanner's fan-out is exercised for
/// real: the seed has one view of a member and the member has another.
/** Counts how many times the scanner waited between cycles. */
private actor TickCounter {
    private var ticks = 0
    func bump() { ticks += 1 }
    func count() -> Int { ticks }
}

private actor HostTransport: FleetTransport {
    private var inventories: [String: String]
    private(set) var requested: [String] = []
    var failNext = false

    init(inventories: [String: String]) {
        self.inventories = inventories
    }

    func send(_ request: FleetRequest) async throws -> FleetResponse {
        requested.append("\(request.target.host):\(request.target.port)\(request.path)")
        if failNext {
            throw FleetError.unreachable(target: "\(request.target)", reason: "test")
        }
        let key = "\(request.target.host):\(request.target.port)"
        guard request.path.hasSuffix("/inventory"), let body = inventories[key] else {
            return FleetResponse(status: 404, body: Data(#"{"error":"not-found"}"#.utf8))
        }
        return FleetResponse(status: 200, body: Data(body.utf8))
    }

    func setFailNext(_ value: Bool) { failNext = value }
    func setInventory(_ key: String, _ body: String) { inventories[key] = body }
    func count() -> Int { requested.count }
    func clear() { requested.removeAll() }
}

private func inventory(
    self name: String, id: String, address: String,
    workspaces: String = "[]", sessions: String = "[]",
    peers: String = "[]"
) -> String {
    """
    { "ok": true, "group": "tinytitan-lan",
      "self": { "id": "\(id)", "name": "\(name)", "port": 3080, "addresses": ["\(address)"], "dshVersion": "0.1.5-rc.2" },
      "workspaces": \(workspaces), "sessions": \(sessions), "peers": \(peers) }
    """
}

private func peer(
    id: String, name: String, address: String,
    workspaces: String = "[]", sessions: String = "[]"
) -> String {
    """
    { "id": "\(id)", "address": "\(address)", "port": 3080, "name": "\(name)",
      "source": "tailscale", "dshVersion": "0.1.5-rc.2", "lastSeen": 1758100000000,
      "workspaceCount": 0, "sessionCount": 0, "workspaces": \(workspaces), "sessions": \(sessions) }
    """
}

/// The seed knows Node3 only as an address; Node3's own answer names a machine
/// the seed has never heard of, and reports richer detail about itself.
private func hosts() -> [String: String] {
    [
        "127.0.0.1:3080": inventory(
            self: "macbook-ab", id: "macbook-ab:3080", address: "127.0.0.1",
            workspaces:
                #"[{ "id": "w-ab", "path": "/Users/me/Project", "title": "Project", "sessionCount": 1, "hiddenSessionCount": 0, "sessionIds": ["s-ab"] }]"#,
            sessions:
                #"[{ "sessionId": "s-ab", "workspaceId": "w-ab", "workspaceTitle": "Project", "title": "local", "turns": 1 }]"#,
            peers:
                "[\(peer(id: "100.114.69.128:3080", name: "Node3", address: "100.114.69.128", sessions: #"[{ "sessionId": "stale", "title": "a cached view" }]"#))]"
        ),
        "100.114.69.128:3080": inventory(
            self: "Node3", id: "Node3.local:3080", address: "100.114.69.128",
            workspaces:
                #"[{ "id": "w-n3", "path": "/Users/node3/TT_Test", "title": "TT_Test", "sessionCount": 1, "hiddenSessionCount": 0, "sessionIds": ["s-n3"] }]"#,
            sessions:
                #"[{ "sessionId": "s-n3", "workspaceId": "w-n3", "workspaceTitle": "TT_Test", "title": "fresh", "turns": 9 }]"#,
            peers: "[\(peer(id: "100.101.5.9:3080", name: "fra-dc-01", address: "100.101.5.9"))]"
        ),
    ]
}

private func makeScanner(_ transport: HostTransport, interval: Int = 30) -> FleetScanner {
    let client = FleetClient(token: "tinytitan-lan", transport: transport)
    return FleetScanner(
        runner: FleetRunner(client: client),
        seed: FleetTarget(host: "127.0.0.1", port: 3080),
        intervalSeconds: interval
    )
}

@Suite struct FleetScannerMergeTests {
    @Test func aMemberIsNotListedTwiceUnderItsTwoIdentities() {
        // The seed calls it 100.114.69.128:3080; the machine calls itself
        // Node3.local:3080. Keyed by id, that is two Macs.
        let seed = FleetGroup(
            group: "g",
            nodes: [
                FleetNode(
                    id: "100.114.69.128:3080", name: "Node3", host: "100.114.69.128", port: 3080,
                    isSelf: false, source: "tailscale", addresses: ["100.114.69.128"],
                    dshVersion: "0.1.5-rc.2",
                    sessions: [FleetSession(sessionId: "stale", title: "a cached view")])
            ])
        let direct = FleetGroup(
            group: "g",
            nodes: [
                FleetNode(
                    id: "Node3.local:3080", name: "Node3", host: "100.114.69.128", port: 3080,
                    isSelf: true, source: "local", addresses: ["100.114.69.128"],
                    dshVersion: "0.1.5-rc.2",
                    sessions: [FleetSession(sessionId: "s-n3", title: "fresh", turns: 9)])
            ])
        let merged = FleetScanner.merge(seed: seed, reports: [direct])
        #expect(merged.nodes.count == 1, "one machine, not two")
        #expect(
            merged.nodes[0].sessions.map(\.sessionId) == ["s-n3"],
            "the member's own answer wins over the cache")
    }

    @Test func aMemberOnlyOnePeerKnowsStillAppears() {
        let seed = FleetGroup(
            group: "g",
            nodes: [
                FleetNode(id: "a:3080", name: "A", host: "10.0.0.1", port: 3080, isSelf: true)
            ])
        let report = FleetGroup(
            group: "g",
            nodes: [
                FleetNode(id: "b:3080", name: "B", host: "10.0.0.2", port: 3080, isSelf: true),
                FleetNode(
                    id: "c:3080", name: "C", host: "10.0.0.3", port: 3080, isSelf: false,
                    source: "gossip:b"),
            ])
        let merged = FleetScanner.merge(seed: seed, reports: [report])
        #expect(
            merged.nodes.map(\.name) == ["A", "B", "C"],
            "discovery reaches what the seed has not learned")
    }
}

@Suite struct FleetScannerScanTests {
    @Test func aScanAsksEveryMemberAndMergesTheirAnswers() async throws {
        let transport = HostTransport(inventories: hosts())
        let scanner = makeScanner(transport)
        await scanner.scanNow()
        let snapshot = await scanner.current()

        #expect(snapshot.failure == nil)
        #expect(snapshot.scannedAt != nil)
        #expect(snapshot.polled == 2, "the seed plus Node3 were polled")
        #expect(snapshot.answered == 2)
        // The seed, Node3 once, and fra-dc-01 which only Node3 knew about.
        #expect(snapshot.group.nodes.map(\.name).sorted() == ["Node3", "fra-dc-01", "macbook-ab"])
        let done = await transport.requested
        #expect(
            done.contains("100.114.69.128:3080/dsh-lan/inventory"), "members are asked directly")
    }

    @Test func aSecondScanReportsWhatJoined() async throws {
        // One host first: the group has to grow before there is news to report.
        let transport = HostTransport(inventories: [
            "127.0.0.1:3080": inventory(
                self: "macbook-ab", id: "macbook-ab:3080", address: "127.0.0.1")
        ])
        let scanner = makeScanner(transport)

        await scanner.scanNow()
        // The first scan is a baseline: with no previous view to compare
        // against, nothing can be new yet.
        #expect(await scanner.current().group.nodes.map(\.name) == ["macbook-ab"])
        #expect(await scanner.current().newMembers.isEmpty)

        // Node3 joins: the seed now names it as a peer, and answers itself.
        await transport.setInventory(
            "127.0.0.1:3080",
            inventory(
                self: "macbook-ab", id: "macbook-ab:3080", address: "127.0.0.1",
                peers:
                    "[\(peer(id: "100.114.69.128:3080", name: "Node3", address: "100.114.69.128"))]"
            ))
        await transport.setInventory(
            "100.114.69.128:3080",
            inventory(self: "Node3", id: "Node3.local:3080", address: "100.114.69.128"))

        await scanner.scanNow()
        #expect(await scanner.current().newMembers == ["Node3"])

        await scanner.scanNow()
        #expect(
            await scanner.current().newMembers.isEmpty, "nothing joined between two identical scans"
        )
    }

    @Test func anUnreachableSeedKeepsTheLastGoodViewAndSaysSo() async throws {
        let transport = HostTransport(inventories: hosts())
        let scanner = makeScanner(transport)
        await scanner.scanNow()
        let before = await scanner.current().group.nodes.count
        #expect(before == 3)

        await transport.setFailNext(true)
        await scanner.scanNow()
        let after = await scanner.current()
        #expect(after.group.nodes.count == before, "the fleet stays on screen")
        #expect(after.failure != nil)
        #expect(after.scanning == false)
    }

    @Test func aMemberThatDoesNotAnswerIsLeftAsTheSeedDescribedIt() async throws {
        // Only the seed answers: Node3's own read 404s, so the seed's cached view
        // of it is what remains.
        let transport = HostTransport(inventories: [
            "127.0.0.1:3080": hosts()["127.0.0.1:3080"] ?? ""
        ])
        let scanner = makeScanner(transport)
        await scanner.scanNow()
        let snapshot = await scanner.current()
        #expect(snapshot.answered == 1)
        #expect(snapshot.polled == 2)
        #expect(
            snapshot.group.nodes.contains { $0.name == "Node3" },
            "the cached view survives a silent member")
    }
}

@Suite struct FleetScannerScheduleTests {
    @Test func itPollsRepeatedlyOnItsOwnTaskUntilStopped() async throws {
        let transport = HostTransport(inventories: hosts())
        // The wait between cycles is injected, so the schedule is testable
        // without spending the interval: scan, wait, scan, wait.
        let ticks = TickCounter()
        let scanner = FleetScanner(
            runner: FleetRunner(client: FleetClient(token: "k", transport: transport)),
            seed: FleetTarget(host: "127.0.0.1", port: 3080),
            intervalSeconds: 2,
            sleep: { _ in
                await ticks.bump()
                await Task.yield()
            }
        )
        await scanner.start()
        try await Task.sleep(for: .milliseconds(150))
        await scanner.stop()
        let scans = await ticks.count()
        let calls = await transport.count()
        #expect(scans >= 2, "it should keep cycling on its own task, saw \(scans) waits")
        #expect(calls >= scans * 2, "each cycle asks the seed and every member")
    }

    @Test func theDefaultIntervalIsHalfThePluginsDiscoveryPeriod() async {
        let transport = HostTransport(inventories: hosts())
        let scanner = FleetScanner(
            runner: FleetRunner(client: FleetClient(token: "k", transport: transport)),
            seed: FleetTarget(host: "127.0.0.1", port: 3080)
        )
        #expect(
            await scanner.current().intervalSeconds == 30,
            "the plugin discovers every 60 s; the manager polls at half that")
    }

    @Test func requestScanDoesNotBlockAndStillHappens() async throws {
        let transport = HostTransport(inventories: hosts())
        let scanner = makeScanner(transport)
        await scanner.requestScan()
        try await Task.sleep(for: .milliseconds(150))
        #expect(await transport.count() >= 2, "the requested scan ran in the background")
        #expect(await scanner.current().scannedAt != nil)
    }

    /// `stop()` is the owner's last word. The dashboard runs a mutating action
    /// on its own untracked task so a slow peer cannot freeze a frame, and that
    /// task asks for a refresh when it finishes -- so pressing `q` while an
    /// action is in flight lands a `requestScan()` after the scanner has been
    /// stopped. Before this, `stop()` cleared the in-flight slot, the guard
    /// passed, and the manager dialled the fleet again after the terminal had
    /// already been restored.
    @Test func aScanRequestedAfterStopDoesNotDialTheFleetAgain() async throws {
        let transport = HostTransport(inventories: hosts())
        let scanner = makeScanner(transport)
        await scanner.requestScan()
        try await Task.sleep(for: .milliseconds(150))
        let before = await transport.count()
        await scanner.stop()

        await scanner.requestScan()
        try await Task.sleep(for: .milliseconds(150))
        #expect(
            await transport.count() == before,
            "a stopped scanner must not start another scan")
    }
}

@Suite struct FleetBrandTests {
    @Test func theCommandAndTheProductNameAreBothUsedWhereTheyBelong() {
        #expect(FleetBrand.command == "ttlanmanager")
        #expect(FleetBrand.name == "TinyTitan DSH LAN Manager")

        var state = FleetDashboard(seed: FleetTarget(host: "127.0.0.1", port: 3080))
        state.apply(group: sampleGroupFixture())

        // Wide enough for the product's name...
        let wide = FleetDashboardView.render(state, width: 120, height: 10)
        #expect(wide.lines[0].contains("TinyTitan DSH LAN Manager"))
        // ...and narrow enough that the group and counts matter more.
        let narrow = FleetDashboardView.render(state, width: 60, height: 10)
        #expect(narrow.lines[0].contains("ttlanmanager"))
        #expect(!narrow.lines[0].contains("TinyTitan DSH LAN Manager"))

        // The help screen is never width-constrained, so it always carries the name.
        state.press(.character("?"))
        let help = FleetDashboardView.render(state, width: 60, height: 24)
        #expect(help.lines[0].contains("TinyTitan DSH LAN Manager"))
    }
}

/// A group with one member, for the brand assertions.
private func sampleGroupFixture() -> FleetGroup {
    FleetGroup(
        group: "tinytitan-lan",
        nodes: [
            FleetNode(
                id: "me:3080", name: "macbook-ab", host: "127.0.0.1", port: 3080, isSelf: true,
                source: "local", addresses: [], dshVersion: "0.1.5-rc.2")
        ])
}
