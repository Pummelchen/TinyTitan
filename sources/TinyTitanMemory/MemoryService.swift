import ContinuityCore
import Foundation

/// What the serving engine talks to.
///
/// It owns the choice of backend, the fallback when durable storage cannot
/// be written, and the session lifecycle. The engine calls four things: `beginSession`,
/// `instructions`, `execute` and `endSession`. Everything else stays here.
///
/// The service never throws at the engine. Memory is optional by design, so
/// a failure produces a logged event and a degraded mode, not a failed
/// completion. The one thing it will not do is report a write as successful
/// when it was not.
public actor MemoryService {
    public private(set) var configuration: MemoryConfiguration
    /// A store supplied by the caller, used for every scope. Nil in normal
    /// operation; this is how the tests drive the service.
    let injectedStore: (any MemoryStore)?
    let injectedJournal: (any SessionJournal)?
    /// The resident side-engine, when one is wired. Nil in every test and in a
    /// deployment that has not asked for one, which is what keeps the
    /// callers' no-decision fallback the ordinary path rather than a special
    /// case.
    let sideEngine: (any MemorySideEngine)?
    /// The background T7 pass and the ranking hints it leaves. Nil exactly
    /// when no side-engine is wired, so the deterministic search path is the
    /// only path a deployment without an engine ever takes.
    let retrievalHinter: MemoryRetrievalHinter?
    /// How many questions one consolidation may put to the side-engine, in
    /// total.
    ///
    /// A judgement is a full generation on a CPU model, not a lookup:
    /// measured over the wired cases, 15.2 s per case on the 4B and 29.8 s on
    /// the 9B (`docs/side-engine-tasks.md`). A per-fact candidate loop would
    /// therefore cost minutes, so the loop is bounded by a budget for the
    /// whole consolidation instead. Six questions is about a minute and a half
    /// on the 4B, which is the pause consolidation already runs in.
    static let maximumSideEngineQuestions = 6
    /// And how many of them one fact may use — durability, then a duplicate
    /// and a contradiction against one candidate — so the first fact cannot
    /// spend the whole budget.
    static let maximumQuestionsPerFact = 3
    /// One engine, one journal file and one workspace lock per scope.
    ///
    /// Not one engine for the whole service: a request that names another
    /// workspace would otherwise have its facts written into the default
    /// workspace's file, so deleting one project's memory would delete
    /// another's, and the "one file per workspace" the documentation promises
    /// would be false.
    var workspaces: [MemoryScope: Workspace] = [:]
    /// When each workspace was last used, for deciding which to let go of.
    var lastUsed: [MemoryScope: Date] = [:]
    let localStore: InMemoryStore
    /// The engine-authored journal. Separate store, separate key space,
    /// separate trim policy: a busy week of sessions must never evict the
    /// facts the model wrote deliberately.
    let journalFilter: JournalFilter
    /// Set once a durable operation has failed, so the session prompt can say
    /// memory is not persisting instead of the model assuming it is.
    var isDegraded = false
    /// Workspaces whose journal failure has been logged, so a disk that stays
    /// full produces one line rather than one per tool call.
    var reportedJournalFailures: Set<MemoryScope> = []
    /// Workspaces whose stalled compaction has been logged, for the same
    /// reason, and with the same reach: the journal keeps every byte until it
    /// collapses again, so one line per workspace is the whole report.
    var reportedCompactionStalls: Set<MemoryScope> = []
    /// Project files whose failed expiry has been logged, for the same reason:
    /// retention offers an unexpirable file again on every sweep.
    var reportedExpiries: Set<String> = []
    /// Project files whose failed deletion has been logged, for the same
    /// reason: a file the cap cannot remove stays a candidate on every sweep.
    var reportedSweeps: Set<String> = []
    /// Reads whose failure has been reported, keyed by operation and
    /// workspace, so a store that stays broken produces one line per
    /// operation rather than one per call.
    var reportedReadFailures: Set<String> = []
    var log: @Sendable (MemoryLogEvent) -> Void

    /// Everything one scope needs, created on first use.
    struct Workspace {
        let store: any MemoryStore
        let journal: (any SessionJournal)?
        /// Nil when the caller injected its own store, in which case this
        /// service owns no engine to close.
        let engine: ContinuityEngine?
        /// False when the engine has no journal behind it, so writes last
        /// only as long as the process.
        let persists: Bool
    }

    public init(
        configuration: MemoryConfiguration,
        durableStore: (any MemoryStore)? = nil,
        journal: (any SessionJournal)? = nil,
        sideEngine: (any MemorySideEngine)? = nil,
        isIdle: (@Sendable () -> Bool)? = nil,
        log: @escaping @Sendable (MemoryLogEvent) -> Void = { _ in }
    ) {
        self.configuration = configuration
        self.localStore = InMemoryStore(limits: configuration.limits)
        self.journalFilter = configuration.journalLimits.filter
        self.log = log
        self.injectedStore = durableStore
        self.injectedJournal = journal
        self.sideEngine = sideEngine
        // A caller with no idle signal — a benchmark, a test — reads as
        // always idle, which is what makes the pass finish deterministically
        // off the server.
        self.retrievalHinter = sideEngine.map {
            MemoryRetrievalHinter(engine: $0, isIdle: isIdle ?? { true }, log: log)
        }
    }

    /// Records a completed turn. Content is filtered to substance here, so no
    /// caller can accidentally journal a tool result or a file dump.
    public func recordTurn(
        session: MemorySessionContext,
        index: Int,
        prompt: String,
        reply: String,
        model: String?,
        promptTokens: Int,
        completionTokens: Int,
        latencyMilliseconds: Int,
        stopReason: String?
    ) async {
        guard let journal = await workspace(for: session.scope)?.journal else { return }
        let filteredPrompt = journalFilter.filter(prompt)
        let filteredReply = journalFilter.filter(reply)
        let turn = JournalTurn(
            session: session.session.id,
            workspace: session.scope.workspace,
            index: index,
            prompt: filteredPrompt.kept,
            reply: filteredReply.kept,
            model: model,
            promptTokens: promptTokens,
            completionTokens: completionTokens,
            latencyMilliseconds: latencyMilliseconds,
            stopReason: stopReason,
            droppedBytes: filteredPrompt.dropped + filteredReply.dropped)
        await journal.record(turn, in: session.scope)
        // Never fails the turn: the reply has already been given. A journal
        // that refused it stops the workspace reporting itself durable.
        _ = await journalFailed(in: session.scope)
        _ = await compactionStalled(in: session.scope)
        log(.journaled(session: session.session.id, index: index, bytes: turn.byteCount))
        await enforceResidencyBudget(keeping: session.scope)
    }

    /// Open the configured workspace now, so its journal is replayed at boot
    /// rather than on the first request.
    ///
    /// Replay is the one bulk read the store ever does. Paying it at start,
    /// while nothing is being generated, keeps it off the same disk the
    /// expert streamer is about to saturate and off the first user's
    /// latency. Safe to call more than once and safe with memory disabled.
    public func warmUp() async {
        await sweepStaleWorkspaces()
        guard let scope = configuration.scope() else { return }
        _ = await workspace(for: scope)
    }

    /// Keeps project files from piling up, without losing what they know.
    ///
    /// One journal per project means one per directory a client ever ran
    /// from, and nothing else removes them. Two rules. A file untouched for
    /// `retentionDays` has its session log expired -- the transcript, which
    /// is the bulk of it -- and keeps its facts, because a novel paused for
    /// six weeks must not come back without its bible. Beyond
    /// `maximumWorkspaces`, the oldest by last write are deleted outright;
    /// that cap is the only thing that removes facts. A workspace this
    /// process has open is never touched -- its lock is held, and it was
    /// written moments ago in any case. Runs at start and whenever a new
    /// project file is created.
    public func sweepStaleWorkspaces(now: Date = Date()) async {
        let storage = configuration.storage
        guard storage.retentionDays > 0 || storage.maximumWorkspaces > 0 else { return }
        let manager = FileManager.default
        let open = Set(workspaces.keys.map { storage.journalURL(for: $0).standardizedFileURL.path })
        let candidates = Self.projectFiles(under: storage.directory).filter {
            !open.contains($0.url.standardizedFileURL.path)
                && $0.url.deletingPathExtension().lastPathComponent
                    != MemoryConfiguration.sharedWorkspace
        }
        // The cap deletes; it is the only rule that removes facts.
        var doomed: [URL] = []
        if storage.maximumWorkspaces > 0 {
            let ordered = candidates.sorted { $0.modified > $1.modified }
            // The open workspaces count against the cap too.
            let keep = max(0, storage.maximumWorkspaces - open.count)
            doomed = ordered.dropFirst(keep).map(\.url)
        }
        if !doomed.isEmpty {
            var removed: [String] = []
            var refused: [(path: String, file: String, reason: String)] = []
            for url in doomed {
                // Never remove a workspace another process is inside. The cap is
                // the one rule here that deletes facts, and `open` only knows
                // *this* process's workspaces.
                guard !Self.isLockHeld(at: url) else { continue }
                // The journal going is what makes the removal a fact. A volume
                // that refuses it leaves the workspace on disk, so the sweep
                // must neither claim it was deleted nor take the `.lock` that
                // still protects it -- the sibling expiry rule says the same
                // thing about a file it could not rewrite.
                do {
                    try manager.removeItem(at: url)
                } catch {
                    refused.append((url.path, url.lastPathComponent, String(describing: error)))
                    continue
                }
                try? manager.removeItem(at: url.appendingPathExtension("lock"))
                try? manager.removeItem(at: url.appendingPathExtension("compacting"))
                removed.append(url.lastPathComponent)
            }
            if !removed.isEmpty {
                log(
                    .swept(
                        removed: removed,
                        reason: "more than \(storage.maximumWorkspaces) projects"))
            }
            // Once per file, on the precedent of `reportedExpiries`: a workspace
            // the volume will not give up is offered to every later sweep.
            for refusal in refused where reportedSweeps.insert(refusal.path).inserted {
                log(.degraded(operation: "sweep", detail: "\(refusal.file): \(refusal.reason)"))
            }
        }
        // Retention expires the session log and keeps the facts.
        guard storage.retentionDays > 0 else { return }
        let cutoff = now.addingTimeInterval(-Double(storage.retentionDays) * 86_400)
        let deleted = Set(doomed.map(\.path))
        var expired: [String] = []
        var refused: [(path: String, file: String, reason: String)] = []
        for candidate in candidates
        where candidate.modified < cutoff && !deleted.contains(candidate.url.path) {
            if let reason = await Self.expireSessionLog(at: candidate.url) {
                refused.append((candidate.url.path, candidate.url.lastPathComponent, reason))
            } else {
                expired.append(candidate.url.lastPathComponent)
            }
        }
        if !expired.isEmpty { log(.expired(files: expired)) }
        // Once per file, on the precedent of `reportedJournalFailures`: a file
        // that cannot be expired is offered to every later sweep, and one stuck
        // workspace should not produce a line per sweep. Keyed by path because
        // project names are only unique inside their directory.
        for refusal in refused where reportedExpiries.insert(refusal.path).inserted {
            log(.degraded(operation: "expire", detail: "\(refusal.file): \(refusal.reason)"))
        }
    }

}
