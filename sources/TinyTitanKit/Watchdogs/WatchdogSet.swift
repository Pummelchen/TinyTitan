import Foundation

/// The one place a watchdog's report becomes a decision.
///
/// Detectors observe; this decides. A concern is logged always and stops the
/// generation only when its kind is named in `TINYTITAN_WATCHDOG_ACT`, so the
/// blast radius of a mis-calibrated threshold is a log line until an
/// operator deliberately widens it.
///
/// Three further rules, each of which has a test:
///
///   * **Off is free.** With watchdogs disabled nothing is allocated and no
///     detector is consulted, so a feature nobody asked for costs nothing on
///     the per-chunk path.
///   * **One report per kind.** A loop reports once, not once per byte.
///   * **Never fail a completion.** There is no throwing path here at all;
///     the worst a watchdog can do is end a generation early and say so.
package struct WatchdogSet: Sendable {
    package struct Trip: Sendable, Equatable {
        package let kind: WatchdogKind
        package let message: String
        package let acted: Bool
    }

    package let configuration: WatchdogConfiguration
    private var loop: LoopWatchdog
    /// A second loop detector, for the model's reasoning. Its own window,
    /// because an answer that restates the end of its thought is not a loop,
    /// and one shared window would see it as one.
    private var reasoningLoop: LoopWatchdog
    private var stall: StallWatchdog
    private var stub: StubWatchdog
    package private(set) var trips: [Trip] = []
    /// Set when a watchdog that is allowed to act has tripped mid-stream.
    package private(set) var stopMessage: String?

    package init(configuration: WatchdogConfiguration) {
        self.configuration = configuration
        loop = LoopWatchdog(configuration: configuration)
        reasoningLoop = LoopWatchdog(configuration: configuration)
        stall = StallWatchdog(configuration: configuration)
        stub = StubWatchdog(configuration: configuration)
    }

    /// B6: the engine's own generations are not watched. Memory
    /// consolidation is a server-internal call with a deliberately
    /// repetitive prompt and a deliberately terse answer -- exactly the
    /// shape the loop and stub detectors look for -- and the person never
    /// sees it, so stopping it would be a cost with no benefit.
    package static let inert = WatchdogSet(configuration: .off)

    package var isActive: Bool { configuration.isEnabled }

    /// True when a watchdog allowed to act has decided this generation
    /// should end.
    package var wantsStop: Bool { stopMessage != nil }

    package mutating func observe(
        _ chunk: String,
        at instant: ContinuousClock.Instant = .now
    ) {
        guard configuration.isEnabled else { return }
        record(LoopWatchdog.kind, loop.observe(chunk, at: instant))
        record(StallWatchdog.kind, stall.observe(chunk, at: instant))
    }

    /// Reasoning is watched for loops only. A model can think in circles
    /// until its token budget is gone, and since thinking left the answer
    /// channel nothing saw it. Stall and stub stay on the answer: a long
    /// thought before a short reply is the model working, not stalling.
    package mutating func observeReasoning(
        _ chunk: String,
        at instant: ContinuousClock.Instant = .now
    ) {
        guard configuration.isEnabled else { return }
        record(
            LoopWatchdog.kind, reasoningLoop.observe(chunk, at: instant),
            where: "in reasoning")
    }

    package mutating func check(at instant: ContinuousClock.Instant = .now) {
        guard configuration.isEnabled else { return }
        record(StallWatchdog.kind, stall.check(at: instant))
    }

    package mutating func finish(
        visibleBytes: Int, requestBytes: Int,
        finishReason: String
    ) {
        guard configuration.isEnabled else { return }
        record(
            StubWatchdog.kind,
            stub.finish(
                visibleBytes: visibleBytes, requestBytes: requestBytes,
                finishReason: finishReason))
    }

    /// A ping-pong report from the incoming request (B2), folded in so
    /// every watchdog result reaches the log by one path.
    package mutating func record(pingPong verdict: WatchdogVerdict) {
        guard configuration.isEnabled, let message = verdict.message else { return }
        // `acted` is always false: ping-pong has no safe intervention, and
        // `WatchdogKind.canAct` records why.
        trips.append(Trip(kind: PingPongWatchdog.kind, message: message, acted: false))
    }

    private mutating func record(
        _ kind: WatchdogKind, _ verdict: WatchdogVerdict,
        where place: String? = nil
    ) {
        guard let found = verdict.message else { return }
        let message = place.map { "\($0): \(found)" } ?? found
        let acts = configuration.acts(kind)
        trips.append(Trip(kind: kind, message: message, acted: acts))
        if acts, stopMessage == nil {
            stopMessage = "\(kind.rawValue): \(message)"
        }
    }

    /// What the client is told, once the generation is over.
    ///
    /// This is the whole of the user-visible policy, in one place so it can
    /// be read and tested without a model: the note is appended to the
    /// content and the finish reason is mapped. Nothing else about the
    /// completion changes.
    package struct Outcome: Sendable, Equatable {
        package let content: String
        package let finishReason: String
        /// The text appended, or nil when nothing was.
        package let note: String?
    }

    package func resolve(content: String, finishReason: String) -> Outcome {
        guard let explanation else {
            return Outcome(content: content, finishReason: finishReason, note: nil)
        }
        // B4: `length` is the nearest honest reason either protocol offers.
        let reason = stopMessage == nil ? finishReason : "length"
        return Outcome(
            content: content + explanation,
            finishReason: reason,
            note: explanation)
    }

    /// B4: a stopped generation must say so in its content. Neither the
    /// OpenAI nor the Anthropic protocol has an honest finish reason for
    /// "the server stopped this", and inventing one breaks clients -- so the
    /// reason is mapped to the nearest existing value and the truth is told
    /// in the one place that cannot break a client, the text itself.
    package var explanation: String? {
        var notes: [String] = []
        if let stopMessage {
            notes.append("[TinyTitan stopped this generation: \(stopMessage).]")
        }
        guard !notes.isEmpty else { return nil }
        return "\n\n" + notes.joined(separator: " ")
    }
}
