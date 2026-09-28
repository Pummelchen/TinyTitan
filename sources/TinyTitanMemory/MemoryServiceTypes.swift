import ContinuityCore
import Foundation

// The memory service's session context and log-event value types.
//
// Split out of `MemoryService.swift` (2026-09-28) under the 500-line-per-file
// rule (Task 8 of the cleanup runbook) as pure code motion. Members whose
// callers stay behind widened from `private` to internal.
/// Everything a session needs to carry once memory has started.
public struct MemorySessionContext: Sendable, Equatable {
    public let session: MemorySession
    public let scope: MemoryScope
    public let bootstrap: MemoryBootstrap
    /// False when the session is running on the local fallback, which the
    /// prompt tells the model so it does not promise persistence.
    public let isDurable: Bool

    public init(
        session: MemorySession,
        scope: MemoryScope,
        bootstrap: MemoryBootstrap,
        isDurable: Bool
    ) {
        self.session = session
        self.scope = scope
        self.bootstrap = bootstrap
        self.isDurable = isDurable
    }
}

/// Observable memory events. The engine maps these onto its own log; keeping
/// them as values means this module prints nothing itself and stays testable.
public enum MemoryLogEvent: Sendable, Equatable {
    case sessionStarted(
        session: String, scope: MemoryScope, bootstrapRecords: Int,
        bootstrapBytes: Int)
    case sessionEnded(session: String, scope: MemoryScope)
    case toolSucceeded(tool: String)
    case toolFailed(tool: String, detail: String)
    case degraded(operation: String, detail: String)
    case rejectedScope(String)
    case consolidated(session: String, records: Int)
    case journaled(session: String, index: Int, bytes: Int)
    /// Project files removed by retention, and why.
    case swept(removed: [String], reason: String)
    /// A consolidation wrote a value the key had before; it is now disputed.
    case reversionFlagged(key: String)
    /// The guard refused a model-derived write over what the person asserted.
    /// The person's value stays active and the address is disputed.
    case guardHeld(key: String)
    /// Facts a consolidation returned that already held the same value.
    case unchangedSkipped(session: String, count: Int)
    /// A fact the side-engine judged not worth keeping. It is not stored at
    /// all, and only the key is logged.
    case notDurableStopped(key: String)
    /// How many facts the side-engine's durability check dropped from one
    /// consolidation.
    case notDurablesStopped(session: String, count: Int)
    /// A change to a value a stored rule fixes. The old value stays and the
    /// change is not written; only the key is logged, never the rule or either
    /// value.
    case ruleConflictStopped(key: String)
    /// How many rule conflicts stopped one consolidation's writes.
    case ruleConflictsStopped(session: String, count: Int)
    /// A new key whose content an existing key already carried. The store
    /// keeps one address instead of two, and both keys are named — never a
    /// value.
    case nearDuplicateStopped(key: String, kept: String)
    /// How many facts the side-engine's near-duplicate check stopped in one
    /// consolidation.
    case nearDuplicatesStopped(session: String, count: Int)
    /// A new key the side-engine judged unable to be true at the same time as
    /// an existing one. Advisory: the write is not changed, because
    /// disagreement is not supersession and T4, which would tell them apart,
    /// is not ready.
    case contradictionFound(key: String, conflictsWith: String)
    /// How many possible contradictions one consolidation recorded.
    case contradictionsFound(session: String, count: Int)
    /// A consolidation wrote a fact about the person to the shared workspace.
    case sharedFactWritten(key: String)
    /// One background T7 sweep finished. Neither the question nor a value is
    /// logged — a question is the person's words and memory holds anything the
    /// model wrote — only how much work was done.
    case retrievalHints(judged: Int, answered: Int)
    /// Project files whose session log was expired by retention; facts kept.
    case expired(files: [String])

    /// One log line. Never contains a memory's contents or a credential: the
    /// log is operational, and memory can hold anything the model wrote.
    public var message: String {
        switch self {
        case .sessionStarted(let session, let scope, let records, let bytes):
            return "memory session=\(session) scope=\(scope.namespace)/\(scope.user)/"
                + "\(scope.workspace) bootstrap=\(records) records \(bytes)B"
        case .sessionEnded(let session, _):
            return "memory session=\(session) ended"
        case .toolSucceeded(let tool):
            return "memory tool=\(tool) ok"
        case .toolFailed(let tool, let detail):
            return "memory tool=\(tool) failed: \(detail)"
        case .degraded(let operation, let detail):
            return "memory degraded during \(operation): \(detail)"
        case .rejectedScope(let workspace):
            return "memory disabled for this session: unusable workspace '\(workspace)'"
        case .consolidated(let session, let records):
            return "memory session=\(session) consolidated \(records) records"
        case .journaled(let session, let index, let bytes):
            return "journal session=\(session) turn=\(index) \(bytes)B"
        case .reversionFlagged(let key):
            return "memory reversion flagged as disputed: \(key)"
        case .guardHeld(let key):
            return "memory guard kept the user's fact, marked disputed: \(key)"
        case .sharedFactWritten(let key):
            return "memory shared fact written for every project: \(key)"
        case .retrievalHints(let judged, let answered):
            return "memory retrieval hints: judged \(judged) fact(s), "
                + "\(answered) could answer"
        case .unchangedSkipped(let session, let count):
            return "memory session=\(session) consolidation skipped \(count) unchanged fact(s)"
        case .nearDuplicateStopped(let key, let kept):
            return "memory near-duplicate stopped: \(key) is already \(kept)"
        case .notDurableStopped(let key):
            return "memory not worth keeping, not stored: \(key)"
        case .notDurablesStopped(let session, let count):
            return "memory session=\(session) consolidation dropped \(count) fact(s) not "
                + "worth keeping"
        case .ruleConflictStopped(let key):
            return "memory rule conflict, change not stored: \(key)"
        case .ruleConflictsStopped(let session, let count):
            return "memory session=\(session) consolidation stopped \(count) change(s) a "
                + "rule fixes"
        case .nearDuplicatesStopped(let session, let count):
            return "memory session=\(session) consolidation stopped \(count) near-duplicate(s)"
        case .contradictionFound(let key, let conflictsWith):
            return "memory possible conflict: \(key) may disagree with \(conflictsWith)"
        case .contradictionsFound(let session, let count):
            return "memory session=\(session) consolidation recorded \(count) possible conflict(s)"
        case .expired(let files):
            return "memory expired the session log of \(files.count) project file(s), facts kept: "
                + files.joined(separator: ", ")
        case .swept(let removed, let reason):
            return "memory swept \(removed.count) project file(s) (\(reason)): "
                + removed.joined(separator: ", ")
        }
    }
}
