import Foundation
import Testing

@testable import TinyTitanFleetCore

/// A group of three members: one local with a session, one Tailscale peer with a
/// workspace and two sessions, and one peer with nothing open.
private func sampleGroup() -> FleetGroup {
    let local = FleetNode(
        id: "macbook-ab:3080", name: "macbook-ab", host: "127.0.0.1", port: 3080, isSelf: true,
        source: "local", addresses: ["192.168.18.27", "fd7a:115c:a1e0::1"],
        dshVersion: "0.1.5-rc.2",
        workspaces: [
            FleetWorkspace(
                id: "w-ab", path: "/Users/me/Project", title: "Project", sessionCount: 1,
                hiddenSessionCount: 0, sessionIds: ["s-ab"])
        ],
        sessions: [
            FleetSession(sessionId: "s-ab", workspaceId: "w-ab", title: "local work", turns: 3)
        ]
    )
    let remote = FleetNode(
        id: "100.114.69.128:3080", name: "Node3", host: "100.114.69.128", port: 3080, isSelf: false,
        source: "tailscale", addresses: ["100.114.69.128", "fd7a:115c:a1e0::7f01:45af"],
        dshVersion: "0.1.4",
        lastSeen: Date().timeIntervalSince1970 * 1000 - 4000,
        workspaces: [
            FleetWorkspace(
                id: "w-n3", path: "/Users/node3/Downloads/TT_Test", title: "TT_Test",
                sessionCount: 2, hiddenSessionCount: 0, sessionIds: ["s-n3a", "s-n3b"])
        ],
        sessions: [
            FleetSession(
                sessionId: "s-n3a", workspaceId: "w-n3", title: "story benchmark", turns: 12),
            FleetSession(sessionId: "s-n3b", workspaceId: "w-n3", title: "fleet smoke", turns: 3),
        ]
    )
    let idle = FleetNode(
        id: "100.101.5.9:3080", name: "fra-dc-01", host: "100.101.5.9", port: 3080, isSelf: false,
        source: "tailscale", addresses: ["100.101.5.9"], dshVersion: nil,
        lastSeen: Date().timeIntervalSince1970 * 1000 - 90_000
    )
    return FleetGroup(group: "tinytitan-lan", nodes: [local, remote, idle])
}

private func dashboard() -> FleetDashboard {
    var state = FleetDashboard(seed: FleetTarget(host: "127.0.0.1", port: 3080))
    state.apply(group: sampleGroup())
    return state
}

@Suite struct FleetDashboardLayoutTests {
    @Test func aFrameIsExactlyTheRequestedSize() {
        let state = dashboard()
        for (width, height) in [(120, 30), (80, 12), (60, 8), (46, 6)] {
            let frame = FleetDashboardView.render(state, width: width, height: height)
            #expect(frame.lines.count == height, "\(width)x\(height) must fill the height")
            for line in frame.lines {
                #expect(line.count <= width, "\(width)x\(height): '\(line)' is too wide")
            }
        }
    }

    @Test func columnsGiveGroundInAFixedOrderAsTheWindowNarrows() {
        let state = dashboard()
        func headings(_ width: Int) -> String {
            FleetDashboardView.render(state, width: width, height: 12).lines[1]
        }
        #expect(headings(120).contains("IPV6"), "a wide window shows everything")
        #expect(!headings(96).contains("IPV6"), "IPv6 goes first")
        #expect(headings(96).contains("SEEN"))
        #expect(!headings(76).contains("SEEN"), "then the age")
        #expect(headings(76).contains("DSH"))
        #expect(!headings(60).contains("DSH"), "then the harness version")
        #expect(!headings(60).contains("TYPE"))
        #expect(
            headings(46).contains("NAME") && headings(46).contains("ADDRESS"),
            "the member and its address are the last to go")
    }

    @Test func aTinyWindowSaysSoRatherThanDrawingRubbish() {
        let frame = FleetDashboardView.render(dashboard(), width: 30, height: 4)
        #expect(frame.lines.contains { $0.contains("too small") })
    }

    /// The message that tells the operator the window is too small must survive
    /// the window. It used to be fitted to the width that triggered it, so one
    /// column printed `…`, zero or less printed nothing at all, and a one-line
    /// height kept only the blank line above it — each with exit 0.
    @Test func theTooSmallMessageSurvivesASizeThatCannotHoldIt() {
        let sizes = [
            (width: 0, height: 30), (width: -5, height: 30), (width: 1, height: 30),
            (width: 100, height: 1), (width: 100, height: 0), (width: 100, height: -5),
        ]
        for size in sizes {
            let frame = FleetDashboardView.render(
                dashboard(), width: size.width, height: size.height)
            let readable = frame.lines.contains { $0.contains("too small") }
            #expect(readable, "\(size.width)×\(size.height) printed \(frame.lines)")
        }
    }

    @Test func longNamesAndPathsAreElidedNotOverflowed() {
        let group = FleetGroup(
            group: "g",
            nodes: [
                FleetNode(
                    id: "x:1", name: "a-very-long-machine-name-that-will-not-fit", host: "10.0.0.1",
                    port: 3080, isSelf: false,
                    source: "seed", addresses: [], dshVersion: "0.1.0",
                    workspaces: [
                        FleetWorkspace(
                            id: "w", path: "/Users/someone/an/extremely/long/path/that/keeps/going",
                            sessionCount: 0, hiddenSessionCount: 0, sessionIds: [])
                    ],
                    sessions: [])
            ])
        var state = FleetDashboard(seed: FleetTarget(host: "127.0.0.1", port: 3080))
        state.apply(group: group)
        let frame = FleetDashboardView.render(state, width: 60, height: 10)
        #expect(frame.lines.contains { $0.contains("…") }, "something was elided")
        for line in frame.lines { #expect(line.count <= 60) }
    }

    @Test func theSelectedRowIsReportedForHighlighting() {
        var state = dashboard()
        let frame = FleetDashboardView.render(state, width: 100, height: 20)
        // `selectedLine` is `Int?`; `try? #require(...)` on it was reported as a
        // redundant require and `try?` then doubled the optional. Reading it
        // directly keeps the assertion identical: a nil selection fails the
        // comparison below.
        let selected = frame.selectedLine
        #expect(selected == 2, "the first row after the title and the headings")

        state.move(by: 1)
        let moved = FleetDashboardView.render(state, width: 100, height: 20)
        #expect(moved.selectedLine == selected.map { $0 + 1 })
    }

    @Test func theSelectionStaysVisibleWhileScrolling() {
        var state = FleetDashboard(seed: FleetTarget(host: "127.0.0.1", port: 3080))
        let many = (0..<40).map { index in
            FleetNode(
                id: "n\(index):3080", name: "node-\(index)", host: "10.0.0.\(index + 1)",
                port: 3080,
                isSelf: false, source: "seed", addresses: [], dshVersion: "0.1.0")
        }
        state.apply(group: FleetGroup(group: "g", nodes: many))
        for _ in 0..<35 { state.move(by: 1) }
        let frame = FleetDashboardView.render(state, width: 100, height: 10)
        // The old shape bound `try? #require(...)` and then shadowed it in an
        // `if let` whose binding was never used. The checks below are the same
        // ones, with the presence check written once.
        #expect(frame.selectedLine != nil)
        #expect((frame.lines.count - 4) > 0)
        if let index = frame.selectedLine {
            #expect(
                index > 1 && index < frame.lines.count - 2,
                "the cursor stays inside the table, not in the footer")
            #expect(frame.lines[index].contains("node-35"))
        }
    }

    @Test func anEmptyGroupSaysWhatIsHappening() {
        let state = FleetDashboard(seed: FleetTarget(host: "127.0.0.1", port: 3080))
        let frame = FleetDashboardView.render(state, width: 100, height: 10)
        #expect(frame.lines.contains { $0.contains("no members yet") })
    }
}

@Suite struct FleetDashboardKeyTests {
    @Test func theTreeExpandsAndCollapsesPerMember() {
        var state = dashboard()
        #expect(state.rows.count == 8, "3 members + 2 workspaces + 3 sessions, all open")
        state.move(by: 3)  // Node3
        state.press(.left)  // collapse it
        #expect(state.rows.count == 5, "its workspace and two sessions go with it")
        state.press(.right)
        #expect(state.rows.count == 8)
    }

    @Test func pOnASessionAsksForTextAndBuildsTheRightAction() {
        var state = dashboard()
        state.move(by: 5)  // the first session row under Node3
        guard case .session(let node, _) = state.selectedRow else {
            Issue.record("expected a session row, got \(String(describing: state.selectedRow))")
            return
        }
        #expect(node == 1)

        state.press(.character("p"))
        guard case .input(.promptText(let target)) = state.mode else {
            Issue.record("p should open the prompt line")
            return
        }
        #expect(target == .session(nodeID: "100.114.69.128:3080", sessionID: "s-n3a"))

        state.press(.character("h"))
        state.press(.character("i"))
        state.press(.backspace)
        state.press(.character("!"))
        state.press(.enter)
        let action = state.takePendingAction()
        #expect(action == .prompt(target: target, text: "h!"))
        #expect(state.takePendingAction() == nil, "an action is taken once")
    }

    @Test func capitalPAndATargetTheMemberAndTheGroup() {
        var state = dashboard()
        state.press(.character("P"))
        #expect(state.mode == .input(field: .promptText(target: .node(nodeID: "macbook-ab:3080"))))
        state.press(.escape)

        state.press(.character("A"))
        #expect(state.mode == .input(field: .promptText(target: .group)))
    }

    @Test func pOnAWorkspaceTargetsThatWorkspace() {
        var state = dashboard()
        state.move(by: 4)  // the workspace row under Node3
        guard case .workspace = state.selectedRow else {
            Issue.record("expected a workspace row")
            return
        }
        state.press(.character("p"))
        #expect(
            state.mode
                == .input(
                    field: .promptText(
                        target: .workspace(nodeID: "100.114.69.128:3080", workspaceID: "w-n3"))))
    }

    @Test func archivingNeedsAConfirmation() {
        var state = dashboard()
        state.move(by: 5)  // a session
        state.press(.character("a"))
        guard case .confirm(_, let action) = state.mode else {
            Issue.record("a should confirm before archiving")
            return
        }
        #expect(action == .archiveSession(nodeID: "100.114.69.128:3080", sessionID: "s-n3a"))
        state.press(.character("n"))
        #expect(state.mode == .browse)
        #expect(state.takePendingAction() == nil, "declining performs nothing")
    }

    @Test func deletingAWorkspaceConfirmsAndThenActs() {
        var state = dashboard()
        state.move(by: 4)
        state.press(.character("d"))
        guard case .confirm(let question, let action) = state.mode else {
            Issue.record("d should confirm")
            return
        }
        #expect(question.contains("w-n3"))
        #expect(action == .deleteWorkspace(nodeID: "100.114.69.128:3080", workspaceID: "w-n3"))
        state.press(.character("y"))
        #expect(state.takePendingAction() == action)
    }

    @Test func creatingAWorkspaceAsksForAFolderThenATitle() {
        var state = dashboard()
        state.press(.character("c"))
        guard case .input(.workspacePath(let nodeID)) = state.mode else {
            Issue.record("c should ask for a folder")
            return
        }
        #expect(nodeID == "macbook-ab:3080")
        for character in "/tmp/x" { state.press(.character(String(character))) }
        state.press(.enter)
        guard case .input(.workspaceTitle(let sameNode, let path)) = state.mode else {
            Issue.record("then for a title")
            return
        }
        #expect(sameNode == nodeID && path == "/tmp/x")
        state.press(.enter)  // Enter takes the folder name as the title
        #expect(
            state.takePendingAction()
                == .createWorkspace(nodeID: nodeID, path: "/tmp/x", title: nil))
    }

    @Test func destructiveKeysOnTheWrongRowExplainThemselves() {
        var state = dashboard()
        state.press(.character("a"))  // a member row, not a session
        #expect(state.mode == .browse)
        #expect(state.status.contains("select a session"))
        state.press(.character("d"))
        #expect(state.status.contains("select a workspace"))
        #expect(state.takePendingAction() == nil)
    }

    @Test func helpIsAFullScreenAndAnyKeyReturns() {
        var state = dashboard()
        state.press(.character("?"))
        #expect(state.mode == .help)
        let frame = FleetDashboardView.render(state, width: 100, height: 24)
        #expect(frame.lines.contains { $0.contains("prompt every session in the whole group") })
        state.press(.character("x"))
        #expect(state.mode == .browse)
    }

    @Test func qAsksToQuitAndRefreshAsksToReload() {
        var state = dashboard()
        state.press(.character("r"))
        #expect(state.takePendingAction() == .refresh)
        state.press(.character("q"))
        #expect(state.takePendingAction() == .quit)
    }

    @Test func aFailedRefreshKeepsTheLastGoodViewAndSaysWhatHappened() {
        var state = dashboard()
        let before = state.rows.count
        state.apply(failure: "100.114.69.128:3080 is unreachable: timeout")
        #expect(state.rows.count == before, "the fleet stays on screen")
        #expect(state.status.contains("unreachable"))
        let frame = FleetDashboardView.render(state, width: 100, height: 12)
        #expect(frame.lines.contains { $0.contains("! 100.114.69.128:3080 is unreachable") })
    }

    @Test func aRefreshKeepsTheCursorOnTheSameRow() {
        var state = dashboard()
        state.move(by: 5)
        let before = state.selectedRow?.id
        state.apply(group: sampleGroup())
        #expect(state.selectedRow?.id == before, "a refresh does not move the cursor")
    }
}

@Suite struct FleetKeyDecodingTests {
    @Test func arrowsEnterBackspaceAndEscapeDecode() {
        #expect(FleetTerminal.decode([0x1B, 0x5B, 0x41]) == [.up])
        #expect(FleetTerminal.decode([0x1B, 0x5B, 0x42]) == [.down])
        #expect(FleetTerminal.decode([0x1B, 0x5B, 0x43]) == [.right])
        #expect(FleetTerminal.decode([0x1B, 0x5B, 0x44]) == [.left])
        #expect(FleetTerminal.decode([0x0D]) == [.enter])
        #expect(FleetTerminal.decode([0x7F]) == [.backspace])
        #expect(FleetTerminal.decode([0x1B]) == [.escape])
        #expect(FleetTerminal.decode([0x03]) == [.escape], "Ctrl-C quits")
    }

    @Test func typingDecodesIncludingMultiByteCharacters() {
        #expect(
            FleetTerminal.decode(Array("abc".utf8)) == [
                .character("a"), .character("b"), .character("c"),
            ])
        #expect(FleetTerminal.decode(Array("é".utf8)) == [.character("é")], "two bytes, one key")
        #expect(FleetTerminal.decode(Array("✓".utf8)) == [.character("✓")])
        #expect(
            FleetTerminal.decode([0x1B, 0x5B, 0x41] + Array("x".utf8)) == [.up, .character("x")])
    }
}
