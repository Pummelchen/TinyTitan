import Foundation

/// One rendered frame. Lines are plain text; which line is selected is reported
/// separately so the terminal layer can highlight it and the tests need no ANSI.
public struct FleetFrame: Sendable, Equatable {
    public let lines: [String]
    public let selectedLine: Int?

    public init(lines: [String], selectedLine: Int? = nil) {
        self.lines = lines
        self.selectedLine = selectedLine
    }
}

/// Draws the fleet dashboard at whatever size the terminal currently is.
///
/// Nothing here touches the terminal, so every layout question — which columns
/// survive at 60 columns, what a long machine name does, what a 40×8 window
/// shows — is answerable in a test.
public enum FleetDashboardView {
    /// Below this there is no honest way to draw a table.
    public static let minimumWidth = 44
    public static let minimumHeight = 6

    /// Render one frame of exactly `height` lines of at most `width` columns.
    public static func render(_ state: FleetDashboard, width: Int, height: Int) -> FleetFrame {
        guard width >= minimumWidth, height >= minimumHeight else {
            return tooSmall(width: width, height: height)
        }
        if case .help = state.mode { return help(width: width, height: height) }
        return table(state, width: width, height: height)
    }

    // MARK: - screens

    private static func tooSmall(width: Int, height: Int) -> FleetFrame {
        let message =
            "terminal too small — need \(minimumWidth)×\(minimumHeight), have \(width)×\(height)"
        // This frame is the only answer a too-small window gets, so it is the one
        // place that prints past the width it was handed: fitted to a window of one
        // column the message was `…`, and at zero or less it was nothing at all.
        let line = width >= message.count ? fit(center(message, width), width) : message
        // A one-line window keeps the message rather than the blank above it.
        let lines = height >= 2 ? [fit("", width), line] : [line]
        return FleetFrame(lines: Array(lines.prefix(max(1, height))), selectedLine: nil)
    }

    private static func help(width: Int, height: Int) -> FleetFrame {
        let body = [
            "\(FleetBrand.name) — keys",
            "",
            "  ↑ ↓ / k j     move through members, workspaces and sessions",
            "  → / Enter     expand the selected member",
            "  ←             collapse it",
            "  p             prompt the selected session (or every session in a",
            "                selected workspace)",
            "  P             prompt every session on the selected member",
            "  A             prompt every session in the whole group",
            "  c             register a folder as a workspace on the selected member",
            "  a             archive the selected session",
            "  d             delete the selected workspace (its sessions are archived first)",
            "  r             refresh now",
            "  ?             this help",
            "  q / Esc       quit",
            "",
            "Members are found by the plugin on each machine: Tailscale (anywhere),",
            "Bonjour (local network), configured seeds, and address gossip between",
            "members. The manager only ever talks to the member that owns the thing",
            "it is acting on.",
        ]
        var lines = body.map { fit($0, width) }
        while lines.count < height - 1 { lines.append(fit("", width)) }
        lines.append(fit("  press any key to go back", width))
        return FleetFrame(lines: Array(lines.prefix(height)), selectedLine: nil)
    }

    // MARK: - the table

    private static func table(_ state: FleetDashboard, width: Int, height: Int) -> FleetFrame {
        let columns = Columns(width: width)
        var lines: [String] = []
        lines.append(fit(header(state, width: width), width))
        lines.append(fit(columns.headings(), width))

        let rows = state.rows
        let visible = max(1, height - 4)
        let offset = scrollOffset(selection: state.selection, count: rows.count, visible: visible)
        var selectedLine: Int?
        for index in offset..<min(rows.count, offset + visible) {
            let line = describe(rows[index], state: state, columns: columns)
            let text = columns.format(line)
            if index == state.selection { selectedLine = lines.count }
            lines.append(fit(text, width))
        }
        if rows.isEmpty {
            lines.append(
                fit("  no members yet — each machine discovers the others on its own timer", width))
        }
        while lines.count < height - 2 { lines.append(fit("", width)) }

        lines.append(fit(statusLine(state, width: width), width))
        lines.append(fit(footer(state, width: width), width))
        return FleetFrame(lines: Array(lines.prefix(height)), selectedLine: selectedLine)
    }

    private static func header(_ state: FleetDashboard, width: Int) -> String {
        let group = state.group.group ?? "unknown group"
        let age = state.refreshedAt.map { "refreshed \(Self.age(since: $0))" } ?? "reading…"
        let tail =
            "  ·  \(group)  ·  \(state.group.nodes.count) members"
            + "  ·  \(state.group.workspaces) ws  ·  \(state.group.sessions) sess  ·  \(age)"
        // The product's name where it fits, the command where it does not: the
        // group and the counts are what a narrow window needs to keep.
        let named = " \(FleetBrand.name)\(tail)"
        return named.count <= width ? named : " \(FleetBrand.command)\(tail)"
    }

    private static func statusLine(_ state: FleetDashboard, width: Int) -> String {
        if let failure = state.failure { return " ! \(failure)" }
        return " " + state.status
    }

    private static func footer(_ state: FleetDashboard, width: Int) -> String {
        switch state.mode {
        case .browse:
            let prompt =
                state.selectedRow.flatMap { state.promptTarget(for: $0)?.label }
                ?? "nothing selected"
            let keys =
                "↑↓ move · → expand · p prompt · P node · A all · c create · a archive · d delete · r refresh · ? help · q quit"
            let line = " \(prompt)  |  \(keys)"
            return line.count <= width ? line : " " + keys
        case .input(.promptText(let target)):
            return " prompt \(target.label) ▸ \(state.input)▏"
        case .input(.workspacePath):
            return " folder to register ▸ \(state.input)▏"
        case .input(.workspaceTitle):
            return " title (Enter for the folder name) ▸ \(state.input)▏"
        case .confirm(let question, _):
            return " \(question) [y/N]"
        case .help:
            return " ? "
        }
    }

    private static func scrollOffset(selection: Int, count: Int, visible: Int) -> Int {
        guard count > visible else { return 0 }
        let half = visible / 2
        return max(0, min(count - visible, selection - half))
    }

    /// One row as fields, before any width is applied.
    static func describe(_ row: FleetRow, state: FleetDashboard, columns: Columns) -> FleetLine {
        switch row {
        case .node(let index):
            guard state.group.nodes.indices.contains(index) else {
                return FleetLine(row: row, name: "?")
            }
            let node = state.group.nodes[index]
            let expanded = state.expanded.contains(node.id)
            let name = "\(expanded ? "▾" : "▸") \(node.name)"
            let address = "\(node.host):\(node.port)"
            return FleetLine(
                row: row,
                name: name,
                kind: node.sourceLabel,
                address: address,
                version: node.dshVersion ?? (node.isSelf ? "unknown" : "unknown"),
                workspaces: "\(node.workspaces.count)",
                sessions: "\(node.sessions.count)",
                lastSeen: node.isSelf ? "—" : age(milliseconds: node.lastSeen),
                ipv6: node.ipv6 ?? ""
            )
        case .workspace(let nodeIndex, let index):
            guard state.group.nodes.indices.contains(nodeIndex),
                state.group.nodes[nodeIndex].workspaces.indices.contains(index)
            else {
                return FleetLine(row: row, name: "?")
            }
            let workspace = state.group.nodes[nodeIndex].workspaces[index]
            return FleetLine(
                row: row,
                name: "    \(workspace.path)",
                kind: "workspace",
                address: workspace.id.isEmpty ? "(by path)" : workspace.id,
                workspaces: "\(workspace.sessionCount ?? workspace.sessionIds?.count ?? 0)",
                sessions: ""
            )
        case .session(let nodeIndex, let index):
            guard state.group.nodes.indices.contains(nodeIndex),
                state.group.nodes[nodeIndex].sessions.indices.contains(index)
            else {
                return FleetLine(row: row, name: "?")
            }
            let session = state.group.nodes[nodeIndex].sessions[index]
            return FleetLine(
                row: row,
                name: "      \(session.title ?? "(untitled)")",
                kind: "session",
                address: session.sessionId
            )
        }
    }

    static func age(milliseconds: Double?) -> String {
        guard let milliseconds, milliseconds > 0 else { return "—" }
        let secondsAgo = Date().timeIntervalSince1970 - milliseconds / 1000
        return age(seconds: secondsAgo)
    }

    static func age(since date: Date, now: Date = Date()) -> String {
        age(seconds: now.timeIntervalSince(date))
    }

    private static func age(seconds: TimeInterval) -> String {
        let seconds = max(0, Int(seconds.rounded()))
        if seconds < 60 { return "\(seconds)s" }
        if seconds < 3600 { return "\(seconds / 60)m" }
        if seconds < 86_400 { return "\(seconds / 3600)h" }
        return "\(seconds / 86_400)d"
    }
}

/// Column widths for one terminal size. Optional columns are dropped in a fixed
/// order as the window narrows, so what disappears is predictable.
public struct Columns: Sendable {
    public let name: Int
    public let kind: Int
    public let address: Int
    public let version: Int
    public let workspaces: Int
    public let sessions: Int
    public let lastSeen: Int
    public let ipv6: Int
    public let showKind: Bool
    public let showVersion: Bool
    public let showLastSeen: Bool
    public let showIPv6: Bool

    public init(width: Int) {
        // Give ground in a fixed order, so what a narrowing window costs is
        // predictable: first the slack in the address and the name (down to
        // widths that still read), then the optional columns from the least to
        // the most useful, and only at the very end the two columns that carry
        // the answer at all.
        var nameWidth = 24
        var addressWidth = 34
        var showIPv6 = true
        var showLastSeen = true
        var showVersion = true
        var showKind = true

        func total() -> Int {
            let widths = [
                nameWidth, showKind ? 10 : 0, addressWidth, showVersion ? 13 : 0,
                3, 4, showLastSeen ? 5 : 0, showIPv6 ? 24 : 0,
            ].filter { $0 > 0 }
            return 2 + widths.reduce(0, +) + 2 * (widths.count - 1)
        }

        while total() > width, addressWidth > 22 { addressWidth -= 1 }
        while total() > width, nameWidth > 12 { nameWidth -= 1 }
        while total() > width {
            if showIPv6 {
                showIPv6 = false
            } else if showLastSeen {
                showLastSeen = false
            } else if showVersion {
                showVersion = false
            } else if showKind {
                showKind = false
            } else {
                break
            }
        }
        while total() > width, addressWidth > 12 { addressWidth -= 1 }
        while total() > width, nameWidth > 10 { nameWidth -= 1 }

        // Spend whatever is left on where machines answer, then on their names.
        var slack = width - total()
        if slack > 0 {
            let growth = min(slack, max(0, 46 - addressWidth))
            addressWidth += growth
            slack -= growth
        }
        if slack > 0 {
            let growth = min(slack, max(0, 26 - nameWidth))
            nameWidth += growth
        }

        self.name = nameWidth
        self.kind = showKind ? 10 : 0
        self.address = addressWidth
        self.version = showVersion ? 13 : 0
        self.workspaces = 3
        self.sessions = 4
        self.lastSeen = showLastSeen ? 5 : 0
        self.ipv6 = showIPv6 ? 24 : 0
        self.showKind = showKind
        self.showVersion = showVersion
        self.showLastSeen = showLastSeen
        self.showIPv6 = showIPv6
    }

    public func headings() -> String {
        var parts = ["  " + pad("NAME", name)]
        if showKind { parts.append(pad("TYPE", kind)) }
        parts.append(pad("ADDRESS", address))
        if showVersion { parts.append(pad("DSH", version)) }
        parts.append(pad("WS", workspaces))
        parts.append(pad("SESS", sessions))
        if showLastSeen { parts.append(pad("SEEN", lastSeen)) }
        if showIPv6 { parts.append(pad("IPV6", ipv6)) }
        return parts.joined(separator: "  ")
    }

    public func format(_ line: FleetLine) -> String {
        var parts = ["  " + pad(line.name, name)]
        if showKind { parts.append(pad(line.kind, kind)) }
        parts.append(pad(line.address, address))
        if showVersion { parts.append(pad(line.version, version)) }
        parts.append(padLeft(line.workspaces, workspaces))
        parts.append(padLeft(line.sessions, sessions))
        if showLastSeen { parts.append(pad(line.lastSeen, lastSeen)) }
        if showIPv6 { parts.append(pad(line.ipv6, ipv6)) }
        return parts.joined(separator: "  ")
    }
}

/// Cut or pad to exactly `width` display columns.
public func fit(_ text: String, _ width: Int) -> String {
    guard width > 0 else { return "" }
    if text.count == width { return text }
    if text.count < width { return text + String(repeating: " ", count: width - text.count) }
    guard width > 1 else { return "…" }
    return String(text.prefix(width - 1)) + "…"
}

func pad(_ text: String, _ width: Int) -> String {
    guard width > 0 else { return "" }
    if text.count == width { return text }
    if text.count < width { return text + String(repeating: " ", count: width - text.count) }
    guard width > 1 else { return "…" }
    return String(text.prefix(width - 1)) + "…"
}

func padLeft(_ text: String, _ width: Int) -> String {
    guard width > 0 else { return "" }
    if text.count >= width { return pad(text, width) }
    return String(repeating: " ", count: width - text.count) + text
}

func center(_ text: String, _ width: Int) -> String {
    guard text.count < width else { return fit(text, width) }
    let left = (width - text.count) / 2
    return String(repeating: " ", count: left) + text
}
