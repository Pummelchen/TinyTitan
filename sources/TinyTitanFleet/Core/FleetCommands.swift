import Foundation

/// One member's answer, parsed and raw.
///
/// The raw bytes are kept so `--json` prints verbatim what the member said,
/// rather than a re-encoding that could quietly differ from the wire format.
public struct FleetRead: Sendable {
    public let group: FleetGroup
    public let raw: Data

    public init(group: FleetGroup, raw: Data) {
        self.group = group
        self.raw = raw
    }
}

/// An inventory snapshot the operator handed the tool, read from a file or a pipe.
public enum FleetInventoryError: Error, Equatable, CustomStringConvertible {
    case unreadable(path: String)
    case oversized(path: String, bytes: Int, cap: Int)

    public var description: String {
        switch self {
        case .unreadable(let path):
            return "cannot read \(path)"
        case .oversized(let path, let bytes, let cap):
            return "\(path) is over the \(cap)-byte inventory bound (\(bytes) bytes read)"
        }
    }
}

/// Where `--from` reads its inventory from.
///
/// Both branches are capped, which is the point: `--from -` reads stdin to
/// end-of-file, so bounding the file and not the pipe would protect neither —
/// the pipe is the same allocation under a different name.
///
/// The bound is checked as the bytes accumulate rather than from an `fstat` size
/// before the read, because a pipe has no size to consult. That still bounds the
/// allocation — an over-bound input costs at most one chunk more than the ceiling
/// instead of its full size — but it is a weaker shape than the engine's
/// `BoundedMetadataRead`, and it is on a weaker boundary: the operator named this
/// file or this pipe themselves, so a link at that path is their choice, not an
/// attacker's, which is why the read follows links that the model-directory reads
/// refuse.
public enum FleetInventorySource {
    /// A rendered fleet is one entry per workspace, session and peer, so a real
    /// snapshot is hundreds of kilobytes; 32 MiB is three orders of magnitude of
    /// headroom, and exists to bound the allocation, not to police the format.
    public static let maxBytes = 32 * 1024 * 1024

    public static func read(fileAt url: URL) throws -> Data {
        let handle: FileHandle
        do {
            handle = try FileHandle(forReadingFrom: url)
        } catch {
            throw FleetInventoryError.unreadable(path: url.path)
        }
        defer { try? handle.close() }
        return try readLimited(handle, path: url.path)
    }

    public static func readStandardInput() throws -> Data {
        try readLimited(.standardInput, path: "<stdin>")
    }

    /// Read to end-of-file, refusing as soon as `maxBytes` is crossed.
    public static func readLimited(
        _ handle: FileHandle,
        path: String = "<stream>",
        maxBytes: Int = maxBytes
    ) throws -> Data {
        var data = Data()
        while true {
            let chunk = handle.availableData
            if chunk.isEmpty { break }
            data.append(chunk)
            guard data.count <= maxBytes else {
                throw FleetInventoryError.oversized(path: path, bytes: data.count, cap: maxBytes)
            }
        }
        return data
    }
}

/// What happened to one prompt addressed to one session.
public struct FleetOutcome: Sendable, Equatable {
    public let node: String
    public let sessionId: String
    public let ok: Bool
    public let detail: String

    public init(node: String, sessionId: String, ok: Bool, detail: String) {
        self.node = node
        self.sessionId = sessionId
        self.ok = ok
        self.detail = detail
    }
}

/// The manager's operations. Every one of them resolves the owning member from
/// the group first and then talks to *that* member directly — a prompt is never
/// handed to one instance for another to run.
public struct FleetRunner: Sendable {
    public let client: FleetClient

    public init(client: FleetClient) {
        self.client = client
    }

    /// Read the group from one member and fold its own inventory in as a node,
    /// so the group is complete without walking the fleet.
    public func read(seed: FleetTarget) async throws -> FleetRead {
        let response = try await client.inventoryData(from: seed)
        let inventory: FleetInventory
        do {
            inventory = try JSONDecoder().decode(FleetInventory.self, from: response)
        } catch {
            throw FleetError.decoding(target: "\(seed)", reason: "\(error)")
        }
        return FleetRead(group: Self.assemble(inventory: inventory, seed: seed), raw: response)
    }

    /// Read one member's own view, asking *it* rather than a peer's cache of it.
    /// - Parameter node: the member to ask.
    /// - Returns: that member's group, assembled with itself as the seed.
    public func read(node: FleetNode) async throws -> FleetGroup {
        let inventory = try await client.inventory(from: node.target)
        return Self.assemble(inventory: inventory, seed: node.target)
    }

    /// Fold a member's aggregate answer into the group.
    public static func assemble(inventory: FleetInventory, seed: FleetTarget) -> FleetGroup {
        var nodes: [FleetNode] = []
        if let me = inventory.node {
            nodes.append(
                FleetNode(
                    id: me.id ?? "\(seed)",
                    name: me.name ?? seed.host,
                    host: seed.host,
                    port: me.port ?? seed.port,
                    isSelf: true,
                    source: "local",
                    addresses: me.addresses ?? [],
                    dshVersion: me.dshVersion,
                    workspaces: inventory.workspaces ?? [],
                    sessions: inventory.sessions ?? []
                ))
        }
        for peer in inventory.peers ?? [] {
            nodes.append(
                FleetNode(
                    id: peer.id,
                    name: peer.name ?? peer.address,
                    host: peer.address,
                    port: peer.port,
                    isSelf: false,
                    source: peer.source ?? "unknown",
                    addresses: peer.addresses ?? [],
                    dshVersion: peer.dshVersion,
                    lastSeen: peer.lastSeen,
                    workspaces: peer.workspaces ?? [],
                    sessions: peer.sessions ?? []
                ))
        }
        return FleetGroup(group: inventory.group, nodes: nodes)
    }

    /// Prompt one session on the member that owns it.
    public func prompt(group: FleetGroup, sessionId: String, text: String) async throws
        -> FleetOutcome
    {
        guard let owner = group.owner(ofSession: sessionId) else {
            throw FleetError.notFound("session \(sessionId)")
        }
        return await Self.deliver(client: client, node: owner, sessionId: sessionId, text: text)
    }

    /// Prompt every active session in the group, in bounded batches. One member
    /// being down never stops the rest: each delivery reports its own result.
    public func promptAll(
        group: FleetGroup,
        text: String,
        limit: Int? = nil,
        concurrency: Int = 4
    ) async -> [FleetOutcome] {
        var work = group.sessionOwners
        if let limit, limit > 0 {
            work = Array(work.prefix(limit))
        }
        let batchSize = max(1, concurrency)
        var outcomes: [FleetOutcome] = []
        for start in stride(from: 0, to: work.count, by: batchSize) {
            let batch = Array(work[start..<min(start + batchSize, work.count)])
            let client = self.client
            let delivered = await withTaskGroup(of: FleetOutcome.self) { group in
                for item in batch {
                    group.addTask {
                        await Self.deliver(
                            client: client, node: item.node, sessionId: item.session.sessionId,
                            text: text)
                    }
                }
                var collected: [FleetOutcome] = []
                for await outcome in group { collected.append(outcome) }
                return collected
            }
            outcomes.append(contentsOf: delivered)
        }
        return outcomes
    }

    /// Deliver one prompt, turning a transport failure into a reported outcome
    /// rather than an aborted fan-out.
    private static func deliver(
        client: FleetClient,
        node: FleetNode,
        sessionId: String,
        text: String
    ) async -> FleetOutcome {
        do {
            let ack = try await client.prompt(to: node.target, sessionId: sessionId, text: text)
            return FleetOutcome(
                node: node.name, sessionId: sessionId, ok: ack.ok,
                detail: ack.ok ? "delivered" : "refused")
        } catch {
            let detail = (error as? FleetError)?.description ?? "\(error)"
            return FleetOutcome(node: node.name, sessionId: sessionId, ok: false, detail: detail)
        }
    }

    /// Register a folder as a workspace on a named member.
    public func createWorkspace(
        group: FleetGroup, node selector: String, path: String, title: String?
    ) async throws -> FleetAck {
        guard let node = Self.node(group, matching: selector) else {
            throw FleetError.notFound("member \(selector)")
        }
        return try await client.createWorkspace(on: node.target, path: path, title: title)
    }

    /// Archive a session on the member that owns it.
    public func archive(group: FleetGroup, sessionId: String) async throws -> FleetAck {
        guard let owner = group.owner(ofSession: sessionId) else {
            throw FleetError.notFound("session \(sessionId)")
        }
        return try await client.archiveSession(on: owner.target, sessionId: sessionId)
    }

    /// Delete a workspace from the member that owns it.
    public func deleteWorkspace(group: FleetGroup, workspaceId: String, archiveSessions: Bool)
        async throws -> FleetAck
    {
        guard let owner = group.owner(ofWorkspace: workspaceId) else {
            throw FleetError.notFound("workspace \(workspaceId)")
        }
        return try await client.deleteWorkspace(
            on: owner.target, workspaceId: workspaceId, archiveSessions: archiveSessions)
    }

    /// Resolve a member by id, name or address.
    public static func node(_ group: FleetGroup, matching selector: String) -> FleetNode? {
        group.nodes.first {
            $0.id == selector || $0.name == selector || "\($0.target)" == selector
                || $0.host == selector
        }
    }

    /// Carry out one dashboard action and describe what happened.
    ///
    /// Every branch resolves the owning member first and talks to it directly —
    /// a workspace or node target is expanded into that member's own sessions,
    /// which the manager prompts one by one rather than asking a peer to fan out.
    ///
    /// @param action - what the user asked for.
    /// @param group - the group the action was chosen from.
    /// @returns a one-line status for the dashboard.
    public func perform(_ action: FleetAction, in group: FleetGroup) async -> String {
        switch action {
        case .quit, .refresh:
            return "refreshing…"

        case .prompt(let target, let text):
            return await performPrompt(target: target, text: text, in: group)

        case .archiveSession(let nodeID, let sessionID):
            return await performMutation(label: "archived \(sessionID)") {
                guard let node = Self.node(group, matching: nodeID) else {
                    throw FleetError.notFound("member \(nodeID)")
                }
                return try await client.archiveSession(on: node.target, sessionId: sessionID)
            }

        case .deleteWorkspace(let nodeID, let workspaceID):
            return await performMutation(label: "deleted \(workspaceID)") {
                guard let node = Self.node(group, matching: nodeID) else {
                    throw FleetError.notFound("member \(nodeID)")
                }
                return try await client.deleteWorkspace(
                    on: node.target, workspaceId: workspaceID, archiveSessions: true)
            }

        case .createWorkspace(let nodeID, let path, let title):
            return await performMutation(label: "registered \(path)") {
                guard let node = Self.node(group, matching: nodeID) else {
                    throw FleetError.notFound("member \(nodeID)")
                }
                return try await client.createWorkspace(on: node.target, path: path, title: title)
            }
        }
    }

    private func performPrompt(target: FleetPromptTarget, text: String, in group: FleetGroup) async
        -> String
    {
        switch target {
        case .group:
            let outcomes = await promptAll(group: group, text: text)
            let delivered = outcomes.filter(\.ok).count
            return "prompted the group: \(delivered)/\(outcomes.count) delivered"

        case .session(_, let sessionID):
            do {
                let outcome = try await prompt(group: group, sessionId: sessionID, text: text)
                return outcome.ok
                    ? "prompted \(sessionID)" : "refused \(sessionID): \(outcome.detail)"
            } catch {
                return "failed: \((error as? FleetError)?.description ?? "\(error)")"
            }

        case .node(let nodeID):
            guard let node = Self.node(group, matching: nodeID) else {
                return "no member \(nodeID)"
            }
            return await deliver(
                text, to: node, sessions: node.sessions.map(\.sessionId), label: node.name)

        case .workspace(_, let workspaceID):
            guard let node = group.owner(ofWorkspace: workspaceID),
                let workspace = node.workspaces.first(where: { $0.id == workspaceID })
            else {
                return "no member holds workspace \(workspaceID)"
            }
            let ids =
                workspace.sessionIds
                ?? node.sessions
                .filter { $0.workspaceId == workspaceID }
                .map(\.sessionId)
            return await deliver(text, to: node, sessions: ids, label: workspaceID)
        }
    }

    /// Prompt a list of sessions on one member, reporting how many took it.
    private func deliver(_ text: String, to node: FleetNode, sessions: [String], label: String)
        async -> String
    {
        guard !sessions.isEmpty else { return "\(label) has no active sessions" }
        var delivered = 0
        for sessionId in sessions {
            let outcome = await Self.deliver(
                client: client, node: node, sessionId: sessionId, text: text)
            if outcome.ok { delivered += 1 }
        }
        return "prompted \(label): \(delivered)/\(sessions.count) delivered"
    }

    private func performMutation(label: String, _ work: () async throws -> FleetAck) async -> String
    {
        do {
            let ack = try await work()
            return ack.ok ? label : "\(label) — refused: \(ack.message ?? "no message")"
        } catch {
            return "failed: \((error as? FleetError)?.description ?? "\(error)")"
        }
    }
}

/// Text rendering. Kept apart from the operations so `--json` can bypass it.
public enum FleetRenderer {
    /// The whole group as a readable tree.
    public static func text(_ group: FleetGroup) -> String {
        var lines: [String] = []
        lines.append(
            "group \(group.group ?? "(unknown)") — \(group.nodes.count) Mac(s), "
                + "\(group.workspaces) workspace(s), \(group.sessions) session(s)")
        for node in group.nodes {
            lines.append("")
            lines.append("\(node.name)\(node.isSelf ? "  (this instance)" : "")  \(node.target)")
            lines.append(contentsOf: workspaceLines(node))
            lines.append(contentsOf: sessionLines(node))
        }
        return lines.joined(separator: "\n")
    }

    private static func workspaceLines(_ node: FleetNode) -> [String] {
        guard !node.workspaces.isEmpty else { return ["  no active workspaces"] }
        var lines = ["  workspaces:"]
        for workspace in node.workspaces {
            let count = workspace.sessionCount ?? workspace.sessionIds?.count ?? 0
            let hidden = workspace.hiddenSessionCount ?? 0
            let title = workspace.title.map { "  \($0)" } ?? ""
            lines.append(
                "    \(workspace.id)\(title)  \(workspace.path)  [\(count) visible\(hidden > 0 ? ", \(hidden) archived" : "")]"
            )
        }
        return lines
    }

    private static func sessionLines(_ node: FleetNode) -> [String] {
        guard !node.sessions.isEmpty else { return ["  no active sessions"] }
        var lines = ["  sessions:"]
        for session in node.sessions {
            let title = session.title ?? "(untitled)"
            let where_ = session.workspaceTitle ?? session.workspacePath ?? "?"
            lines.append("    \(session.sessionId)  \(title)  — \(where_)")
        }
        return lines
    }

    /// One line per delivered prompt.
    public static func outcomes(_ outcomes: [FleetOutcome]) -> String {
        guard !outcomes.isEmpty else { return "no active sessions to prompt" }
        var lines: [String] = []
        for outcome in outcomes {
            lines.append(
                "\(outcome.ok ? "ok  " : "FAIL") \(outcome.node)  \(outcome.sessionId)  \(outcome.detail)"
            )
        }
        let ok = outcomes.filter(\.ok).count
        lines.append("")
        lines.append(
            "\(ok) delivered, \(outcomes.count - ok) failed, of \(outcomes.count) considered")
        return lines.joined(separator: "\n")
    }
}
