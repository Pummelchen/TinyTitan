import Foundation
import Testing

@testable import TinyTitanLib

/// What the facade reports from the orchestrator's event stream.
///
/// One mapper serves both entry points — a conversation and a raw completion —
/// so they cannot drift apart in what they tell a caller. Reasoning and tool
/// calls are dropped here rather than by accident: the A1 surface cannot express
/// them yet, and that is recorded rather than silent.
@Suite struct FacadeEventMappingTests {
    @Test func contentBecomesAToken() {
        guard case .token(let text)? = Session.facadeEvent(.content("hello")) else {
            Issue.record("content should map to a token event")
            return
        }
        #expect(text == "hello")
    }

    @Test func thePromptEventIsForwardedWithItsCounts() {
        guard
            case .promptProcessed(let tokens, let cachedTokens)? = Session.facadeEvent(
                .promptProcessed(tokens: 19, cachedTokens: 4))
        else {
            Issue.record("the prompt event should survive the mapping")
            return
        }
        #expect(tokens == 19)
        #expect(cachedTokens == 4)
    }

    @Test func reasoningIsDroppedUntilTheSurfaceCanCarryIt() {
        #expect(Session.facadeEvent(.reasoning("thinking…")) == nil)
    }
}
