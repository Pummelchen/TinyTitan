import Foundation
import Testing
import TinyTitanFleetCore

/// A transport that answers from a table and remembers every request, so the
/// manager's routing — *which* Mac a prompt is sent to — is what gets asserted,
/// without a network or a live fleet.
actor FakeTransport: FleetTransport {
    struct Reply: Sendable {
        let status: Int
        let body: Data
    }

    private let replies: [String: Reply]
    private(set) var requests: [FleetRequest] = []

    init(replies: [String: Reply] = [:]) {
        self.replies = replies
    }

    func send(_ request: FleetRequest) async throws -> FleetResponse {
        requests.append(request)
        let key = "\(request.method) \(request.target)\(request.path)"
        if let reply = replies[key] ?? replies[request.path] {
            return FleetResponse(status: reply.status, body: reply.body)
        }
        return FleetResponse(status: 404, body: Data(#"{"error":"not-found"}"#.utf8))
    }

    /// `"POST 100.114.69.128:3080/dsh-lan/prompt"` for each request, in order.
    func called() -> [String] {
        requests.map { "\($0.method) \($0.target)\($0.path)" }
    }

    /// Everything except the group read, which every command makes first — so a
    /// routing assertion is about the action and not about the read.
    func actions() -> [String] {
        requests.filter { !$0.path.hasSuffix("/inventory") }
            .map { "\($0.method) \($0.target)\($0.path)" }
    }

    /// The body of the nth action (not the nth request).
    func actionBodyText(of index: Int) -> String? {
        let filtered = requests.filter { !$0.path.hasSuffix("/inventory") }
        guard index < filtered.count, let data = filtered[index].body else { return nil }
        return String(data: data, encoding: .utf8)
    }

}

/// The group one member answers with: itself, one local workspace and session,
/// and one peer on the Tailscale address with its own.
let inventoryJSON = """
    {
      "ok": true,
      "group": "tinytitan-lan",
      "self": { "id": "macbook-ab:3080", "name": "macbook-ab", "port": 3080,
                "addresses": ["127.0.0.1", "192.168.18.27"], "version": "0.1.0" },
      "workspaces": [ { "id": "w-local", "path": "/Users/me/Local", "title": "Local",
                        "sessionCount": 1, "hiddenSessionCount": 2, "sessionIds": ["s-local"] } ],
      "sessions": [ { "sessionId": "s-local", "workspaceId": "w-local",
                      "workspacePath": "/Users/me/Local", "workspaceTitle": "Local",
                      "title": "local session", "turns": 3 } ],
      "peers": [ { "id": "100.114.69.128:3080", "address": "100.114.69.128", "port": 3080,
                   "name": "Node3", "version": "0.1.0", "lastSeen": 1758000000000,
                   "workspaceCount": 1, "sessionCount": 1,
                   "workspaces": [ { "id": "w-remote", "path": "/Users/node3/Remote",
                                     "title": "Remote", "sessionCount": 1,
                                     "hiddenSessionCount": 0, "sessionIds": ["s-remote"] } ],
                   "sessions": [ { "sessionId": "s-remote", "workspaceId": "w-remote",
                                   "workspacePath": "/Users/node3/Remote",
                                   "workspaceTitle": "Remote", "title": "remote session",
                                   "turns": 1 } ] } ]
    }
    """

/// Parse a request body locally, outside the actor.
private func json(_ text: String?) -> [String: Any] {
    guard let text, let data = text.data(using: .utf8),
        let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { return [:] }
    return object
}

private func loadedGroup(
    replies: [String: FakeTransport.Reply] = [:]
) async throws -> (FleetRunner, FakeTransport, FleetGroup) {
    var table = replies
    table["/dsh-lan/inventory"] = FakeTransport.Reply(status: 200, body: Data(inventoryJSON.utf8))
    let transport = FakeTransport(replies: table)
    let runner = FleetRunner(client: FleetClient(token: "tinytitan-lan", transport: transport))
    let seed = try #require(FleetTarget(text: "127.0.0.1:3080", defaultPort: 3080))
    let read = try await runner.read(seed: seed)
    return (runner, transport, read.group)
}

@Suite struct FleetTargetTests {
    @Test func parsesHostPortAndIPv6() throws {
        #expect(
            FleetTarget(text: "192.168.18.25", defaultPort: 3080)
                == FleetTarget(host: "192.168.18.25", port: 3080))
        #expect(FleetTarget(text: "node3.local:3081", defaultPort: 3080)?.port == 3081)
        #expect(
            FleetTarget(text: "[fd7a:115c:a1e0::5]:3080", defaultPort: 3080)?.host
                == "fd7a:115c:a1e0::5")
        #expect(
            FleetTarget(text: "100.114.69.128:3080", defaultPort: 3080)?.host == "100.114.69.128")
    }

    @Test func refusesWhatItCannotDial() {
        #expect(FleetTarget(text: "", defaultPort: 3080) == nil)
        #expect(FleetTarget(text: "host:not-a-port", defaultPort: 3080) == nil)
        #expect(FleetTarget(text: "a:b:c", defaultPort: 3080) == nil)
        #expect(FleetTarget(text: "host:0", defaultPort: 3080) == nil)
        #expect(FleetTarget(text: "host:99999", defaultPort: 3080) == nil)
    }
}

@Suite struct FleetGroupTests {
    @Test func decodesTheReservedSelfKey() async throws {
        let (_, _, group) = try await loadedGroup()
        #expect(group.group == "tinytitan-lan")
        #expect(group.nodes.count == 2)
        #expect(group.nodes.first?.name == "macbook-ab")
        #expect(group.nodes.first?.isSelf == true)
    }

    @Test func foldsSelfAndPeersIntoOneView() async throws {
        let (_, _, group) = try await loadedGroup()
        #expect(group.workspaces == 2)
        #expect(group.sessions == 2)
        let remote = try #require(group.nodes.first { !$0.isSelf })
        #expect(remote.name == "Node3")
        #expect(remote.target == FleetTarget(host: "100.114.69.128", port: 3080))
        #expect(remote.workspaces.first?.path == "/Users/node3/Remote")
    }

    @Test func resolvesTheOwnerOfASessionOrWorkspace() async throws {
        let (_, _, group) = try await loadedGroup()
        #expect(group.owner(ofSession: "s-local")?.isSelf == true)
        #expect(group.owner(ofSession: "s-remote")?.name == "Node3")
        #expect(group.owner(ofWorkspace: "w-remote")?.name == "Node3")
        #expect(group.owner(ofSession: "nope") == nil)
    }
}

@Suite struct FleetRoutingTests {
    @Test func aPromptGoesToTheOwnerAndNeverToTheSeed() async throws {
        let (runner, transport, group) = try await loadedGroup(
            replies: [
                "/dsh-lan/prompt": FakeTransport.Reply(
                    status: 200, body: Data(#"{"ok":true}"#.utf8))
            ]
        )
        let outcome = try await runner.prompt(
            group: group, sessionId: "s-remote", text: "checkpoint")
        #expect(outcome.ok)
        let calls = await transport.actions()
        #expect(
            calls == ["POST 100.114.69.128:3080/dsh-lan/prompt"],
            "the remote session's prompt goes to Node3, not to the member we read from")
        let body = json(await transport.actionBodyText(of: 0))
        #expect(body["sessionId"] as? String == "s-remote")
        #expect(body["prompt"] as? String == "checkpoint")
    }

    @Test func anUnknownSessionIsNotFoundRatherThanSentAnywhere() async throws {
        let (runner, transport, group) = try await loadedGroup()
        await #expect(throws: FleetError.notFound("session ghost")) {
            try await runner.prompt(group: group, sessionId: "ghost", text: "x")
        }
        #expect(await transport.actions().isEmpty, "nothing is dialled for a session nobody holds")
    }

    @Test func promptAllReportsEverySessionAndSurvivesAFailure() async throws {
        let (runner, transport, group) = try await loadedGroup(
            replies: [
                "/dsh-lan/prompt": FakeTransport.Reply(
                    status: 404,
                    body: Data(#"{"error":"not-found","message":"session has no live agent"}"#.utf8)
                )
            ]
        )
        let outcomes = await runner.promptAll(group: group, text: "status?", concurrency: 2)
        #expect(
            outcomes.count == 2, "every active session is attempted, including the one that fails")
        #expect(outcomes.allSatisfy { !$0.ok })
        #expect(outcomes.contains { $0.detail.contains("no live agent") })
        #expect(await transport.actions().count == 2)
    }

    /// `limit` is a cap. A cap of zero used to be read as "no cap", so asking for
    /// no sessions prompted every session on every Mac in the group.
    @Test func aLimitOfZeroPromptsNoSessionsRatherThanAllOfThem() async throws {
        let (runner, transport, group) = try await loadedGroup(
            replies: [
                "/dsh-lan/prompt": FakeTransport.Reply(
                    status: 200, body: Data(#"{"ok":true}"#.utf8))
            ]
        )
        let outcomes = await runner.promptAll(group: group, text: "status?", limit: 0)
        let calls = await transport.actions()
        #expect(outcomes.isEmpty, "a cap of zero is zero sessions, got \(outcomes.count)")
        #expect(calls.isEmpty, "a bounded broadcast dialled \(calls.count) sessions")
    }

    /// The same cap below zero. `Array.prefix(-1)` traps, so the only thing standing
    /// between a negative limit and a crash was the guard that inverted its meaning.
    @Test func aNegativeLimitPromptsNoSessionsAndDoesNotTrap() async throws {
        let (runner, transport, group) = try await loadedGroup(
            replies: [
                "/dsh-lan/prompt": FakeTransport.Reply(
                    status: 200, body: Data(#"{"ok":true}"#.utf8))
            ]
        )
        let outcomes = await runner.promptAll(group: group, text: "status?", limit: -1)
        let calls = await transport.actions()
        #expect(outcomes.isEmpty, "a negative cap asks for no sessions, got \(outcomes.count)")
        #expect(calls.isEmpty, "a negative cap dialled \(calls.count) sessions")
    }

    @Test func mutatingPathsAreEncodedAndTargeted() async throws {
        // An id is one path component on the wire, whatever it contains: the
        // plugin's routes reject a slash inside one.
        #expect(FleetClient.encodePathComponent("s/1") == "s%2F1")
        #expect(FleetClient.encodePathComponent("plain-id") == "plain-id")

        let (runner, transport, group) = try await loadedGroup(
            replies: [
                "/dsh-lan/sessions/s-local/archive": FakeTransport.Reply(
                    status: 200, body: Data(#"{"ok":true}"#.utf8)),
                "/dsh-lan/workspaces/w-remote/delete": FakeTransport.Reply(
                    status: 200, body: Data(#"{"ok":true,"deleted":true}"#.utf8)),
            ]
        )
        _ = try await runner.archive(group: group, sessionId: "s-local")
        _ = try await runner.deleteWorkspace(
            group: group, workspaceId: "w-remote", archiveSessions: false)
        let calls = await transport.actions()
        #expect(
            calls.contains("POST 127.0.0.1:3080/dsh-lan/sessions/s-local/archive"),
            "a local session is archived on the member that holds it")
        #expect(
            calls.contains("POST 100.114.69.128:3080/dsh-lan/workspaces/w-remote/delete"),
            "the workspace is deleted on the member that holds it")
        let body = json(await transport.actionBodyText(of: 1))
        #expect(
            body["archiveSessions"] as? Bool == false,
            "--keep-sessions is passed through, not assumed")
    }

    @Test func createWorkspaceGoesToTheNamedMember() async throws {
        let (runner, transport, group) = try await loadedGroup(
            replies: [
                "/dsh-lan/workspaces": FakeTransport.Reply(
                    status: 200, body: Data(#"{"ok":true,"workspaceId":"ws-new"}"#.utf8))
            ]
        )
        let ack = try await runner.createWorkspace(
            group: group, node: "Node3", path: "/Users/node3/New", title: "New")
        #expect(ack.ok)
        #expect(await transport.actions() == ["POST 100.114.69.128:3080/dsh-lan/workspaces"])
        let body = json(await transport.actionBodyText(of: 0))
        #expect(body["path"] as? String == "/Users/node3/New")
        #expect(body["title"] as? String == "New")
    }

    @Test func anAckThatCarriesNoBooleanOkIsNotAnAck() async throws {
        // Every 2xx from the plugin's router has carried `ok: true` since its
        // first commit (0ddce77), so a 2xx without one has not confirmed the
        // action. `?? true` used to call that success.
        let (runner, _, group) = try await loadedGroup(
            replies: [
                "/dsh-lan/workspaces": FakeTransport.Reply(
                    status: 200, body: Data(#"{"workspaceId":"ws-new"}"#.utf8))
            ]
        )
        await #expect(
            throws: FleetError.decoding(
                target: "100.114.69.128:3080",
                reason: "a 2xx answer carries no boolean `ok`")
        ) {
            _ = try await runner.createWorkspace(
                group: group, node: "Node3", path: "/Users/node3/New", title: nil)
        }
    }

    @Test func anAckWhoseOkIsAStringIsNotAnAck() async throws {
        let (runner, _, group) = try await loadedGroup(
            replies: [
                "/dsh-lan/sessions/s-local/archive": FakeTransport.Reply(
                    status: 200, body: Data(#"{"ok":"true"}"#.utf8))
            ]
        )
        await #expect(
            throws: FleetError.decoding(
                target: "127.0.0.1:3080",
                reason: "a 2xx answer carries no boolean `ok`")
        ) {
            _ = try await runner.archive(group: group, sessionId: "s-local")
        }
    }

    @Test func aTwoHundredPageThatIsNotJSONIsNotAnAck() async throws {
        // A transparent proxy or a captive portal answers 200 with a page. The
        // manager used to print the page and exit 0 as if the member had
        // confirmed the delete — which is the arm that archives sessions.
        let (runner, _, group) = try await loadedGroup(
            replies: [
                "/dsh-lan/workspaces/w-remote/delete": FakeTransport.Reply(
                    status: 200, body: Data("<html><body>intercepted</body></html>".utf8))
            ]
        )
        await #expect(
            throws: FleetError.decoding(
                target: "100.114.69.128:3080",
                reason: "a 2xx answer carries no boolean `ok`")
        ) {
            _ = try await runner.deleteWorkspace(
                group: group, workspaceId: "w-remote", archiveSessions: true)
        }
    }

    @Test func anHTTPFailureCarriesTheMembersOwnMessage() async throws {
        let transport = FakeTransport(replies: [
            "/dsh-lan/inventory": FakeTransport.Reply(
                status: 401,
                body: Data(#"{"error":"unauthorized","hint":"send the shared token"}"#.utf8))
        ])
        let runner = FleetRunner(client: FleetClient(token: "wrong", transport: transport))
        let seed = try #require(FleetTarget(text: "127.0.0.1:3080", defaultPort: 3080))
        await #expect(
            throws: FleetError.http(target: "127.0.0.1:3080", status: 401, message: "unauthorized")
        ) {
            try await runner.read(seed: seed)
        }
    }
}

@Suite struct FleetRendererTests {
    @Test func textNamesTheGroupAndEveryNode() async throws {
        let (_, _, group) = try await loadedGroup()
        let text = FleetRenderer.text(group)
        #expect(text.contains("group tinytitan-lan"))
        #expect(text.contains("2 Mac(s), 2 workspace(s), 2 session(s)"))
        #expect(text.contains("macbook-ab"))
        #expect(text.contains("Node3"))
        #expect(text.contains("/Users/node3/Remote"))
        #expect(text.contains("s-remote"))
    }

    @Test func outcomesSummariseAndMarkFailures() {
        let text = FleetRenderer.outcomes([
            FleetOutcome(node: "Node3", sessionId: "s-1", ok: true, detail: "delivered"),
            FleetOutcome(node: "Maria", sessionId: "s-2", ok: false, detail: "unreachable"),
        ])
        #expect(text.contains("FAIL Maria"))
        #expect(text.contains("1 delivered, 1 failed, of 2 considered"))
    }
}
