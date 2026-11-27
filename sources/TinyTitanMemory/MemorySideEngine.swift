import Foundation

/// One fact as the side-engine sees it.
///
/// The whole fact, key included: the key carries the claim as often as the
/// value does, and withholding it halved accuracy when it was measured
/// (`docs/side-engine-tasks.md`).
public struct MemoryFact: Sendable, Equatable {
    public let key: String
    public let value: String

    public init(key: String, value: String) {
        self.key = key
        self.value = value
    }
}

/// What kind of change two versions of one fact are.
public enum MemorySupersession: Sendable, Equatable {
    /// The world moved on: both values can be true, one after the other.
    case update
    /// A rule fixes this value, or the two are about the same moment, so both
    /// cannot be true and one is wrong.
    case conflict
}

/// The judgements a resident side-engine is asked, in memory's own terms.
///
/// The engine itself lives in a runtime target, which this one cannot import —
/// `TinyTitanMemory` depends on `ContinuityCore` and nothing else — so a host
/// adapts its own judge to this port. The shipped server binds none: its only
/// implementation was the dense CPU side engine, and that left with TT-142
/// slice (b) (2026-11-28). Two rules make an optional, fallible engine safe to
/// consult:
///
/// - **`nil` is "no decision"**, and the caller then behaves exactly as it did
///   before the engine existed. A shut-down engine, a completion the parser
///   refused, a timeout — each falls back to the deterministic behaviour
///   instead of blocking a write.
/// - **Only the tasks a caller actually asks appear here.** Contradiction
///   works from the smallest model up; durability, duplication and
///   supersession from the 4B; the reply check from the 9B. Retrieval is
///   measured and now has its caller: it is worth 4 of 4 recall@1 against the
///   token ranking on the fair cases, but a judgement is 15 s, so the only
///   caller is `MemoryRetrievalHinter` — a background pass in the idle window,
///   never a request (`docs/side-engine-tasks.md`).
public protocol MemorySideEngine: Sendable {
    /// T2: is this fact worth keeping after the session ends? False means the
    /// store should not hold it: a line of the story, a remark about the
    /// writing, an acknowledgement.
    ///
    /// A caller must not drop a fact the person asserted on this answer alone —
    /// the guard's whole premise is that their words are not the model's to
    /// discard — so the memory path asks only about model-derived facts.
    func isDurable(_ fact: MemoryFact) async -> Bool?

    /// T5: do these two facts say the same thing? True means a reader learns
    /// nothing from `new` that `stored` did not already tell them.
    ///
    /// **`stored` comes first.** The order is not cosmetic: the prompts were
    /// measured with the fact already in the store as `A` and the incoming one
    /// as `B`, and on the 4B the same pair answers YES in that order and NO
    /// reversed (`docs/side-engine-tasks.md`). A caller that swaps them gets a
    /// silent no.
    func duplicates(_ stored: MemoryFact, _ new: MemoryFact) async -> Bool?

    /// T3: do these two statements disagree? `stored` first, for the same
    /// reason.
    ///
    /// Advisory. Disagreement is not supersession — a state that legitimately
    /// moved on also "disagrees" — so a caller must not refuse a write on this
    /// answer alone.
    func contradicts(_ stored: MemoryFact, _ new: MemoryFact) async -> Bool?

    /// T4: which kind of change is this? `stored` first, as above.
    ///
    /// `rule` is the stored rule that fixes this value, found by the caller —
    /// see `MemoryRuleLookup`. Without one the answer is meaningless for a
    /// value a rule fixes (an eye colour changing is only a conflict if
    /// something says it never may), so a caller with no rule must not pass
    /// `nil` and then act on the answer.
    func supersedes(
        _ stored: MemoryFact, _ new: MemoryFact,
        rule: String?
    ) async -> MemorySupersession?

    /// T7: could this stored fact answer this question? True means the fact
    /// contains the answer, or part of it.
    ///
    /// The caller is `MemoryRetrievalHinter`, which asks only while no client
    /// is generating and stores the answer as a ranking hint for a later
    /// search. A request path must not await this: one judgement is 15.2 s on
    /// the 4B (`docs/side-engine-tasks.md`).
    func couldAnswer(_ question: String, _ fact: MemoryFact) async -> Bool?

    /// Release the engine. The memory service calls this on shutdown, because
    /// the engine is a second resident model and should not wait for process
    /// exit to let its weights go.
    func shutdown() async
}

extension MemorySideEngine {
    /// A value with nothing resident to release says nothing.
    public func shutdown() async {}
}
