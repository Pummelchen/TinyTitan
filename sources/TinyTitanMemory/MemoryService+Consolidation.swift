import ContinuityCore
import Foundation

// Storing a consolidation and the summary it logs.
//
// Split out of `MemoryService.swift` (2026-09-28) under the 500-line-per-file
// rule (Task 8 of the cleanup runbook) as pure code motion. Members whose
// callers stay behind widened from `private` to internal.
extension MemoryService {

    /// Stores a consolidation the engine produced at session end.
    public func storeConsolidation(
        _ records: [MemoryRecord],
        in context: MemorySessionContext
    ) async -> Int {
        let store = await activeStore(for: context.scope)
        var written = 0
        var held = 0
        var unchanged = 0
        var duplicates = 0
        var conflicts = 0
        var dropped = 0
        var ruleConflicts = 0
        // Read once, and only when an engine is wired: the deterministic path
        // pays nothing for the check it cannot make.
        let candidates: [MemoryRecord]
        if sideEngine != nil {
            do {
                candidates = try await store.search(
                    MemoryQuery(limit: 400),
                    in: context.scope)
            } catch {
                // The write still happens: a fact the model distilled is worth
                // more than the duplicate the missing pool cannot rule out, and
                // refusing here would drop the fact for good, because the
                // caller has already spent its session on the extraction. What
                // must not happen is the pool-based dedup going quiet — hence
                // the report, which is what makes this an unknown pool rather
                // than an empty one.
                reportReadFailure("dedup-pool", error, in: context.scope)
                candidates = []
            }
        } else {
            candidates = []
        }
        /// The shared workspace's pool, read the first time one is needed.
        /// Three states, because "this workspace has no facts" and "the
        /// workspace would not answer" are different answers, and `nil` for
        /// both would let a failed read look like the second.
        enum SharedPool {
            case unread
            case unknown
            case known([MemoryRecord])
        }
        var sharedPool = SharedPool.unread
        var questionsLeft = Self.maximumSideEngineQuestions
        for record in records {
            // A fact about the person rather than the project goes to the
            // shared workspace, where every project's bootstrap reads it. Its
            // conventions and preferences are the most user-asserted category
            // in the store, so the guard reaches there too.
            let scope: MemoryScope
            let destination: any MemoryStore
            let isShared: Bool
            if record.isGlobal, let sharedScope = configuration.sharedScope,
                context.scope != sharedScope
            {
                scope = sharedScope
                destination = await activeStore(for: sharedScope)
                isShared = true
            } else {
                scope = context.scope
                destination = store
                isShared = false
            }
            // The extraction is told to write only what changed and still
            // restates unchanged facts: every eye colour in a novel got a v2
            // and a v3 with the identical value. A write that changes nothing
            // is version churn and completion tokens for no fact.
            let current: MemoryRecord?
            do {
                current = try await destination.get(record.key, in: scope)
            } catch {
                // Unknown, not absent: the unchanged-fold below cannot run, so
                // this may cost a version rather than skip one.
                reportReadFailure("current-value", error, in: scope)
                current = nil
            }
            if let current, Self.fold(current.value) == Self.fold(record.value) {
                unchanged += 1
                continue
            }
            // Read once per scope, and only with an engine wired: the
            // deterministic path pays nothing for a check it cannot make.
            var pool = candidates
            if isShared, sideEngine != nil {
                if case .unread = sharedPool {
                    do {
                        sharedPool = .known(
                            try await destination.search(
                                MemoryQuery(limit: 400), in: scope))
                    } catch {
                        // Remembered as unknown, and not retried per record: a
                        // store that threw once on this scope throws again, and
                        // the price of the failure is the dedup it stops doing,
                        // which is exactly what the report is for.
                        reportReadFailure("dedup-pool", error, in: scope)
                        sharedPool = .unknown
                    }
                }
                switch sharedPool {
                case .known(let records): pool = records
                // Not `candidates`: that pool is the project's, and a shared
                // fact deduped against another workspace's records would be
                // judged against the wrong store. An empty pool here is the
                // honest reading of "unknown" — the rules that need it are
                // reported, not silently passed.
                case .unknown, .unread: pool = []
                }
            }
            // T2, T4, T5 and T3 over one budget for the whole consolidation.
            // Durability comes first, then the rule that fixes a value the
            // same key already holds, then the comparison against other keys.
            // The contradiction is advisory; the rule conflict is not, because
            // a rule says the new value cannot be right.
            if let sideEngine, questionsLeft > 0 {
                let outcome = await inspectForWrite(
                    record, current: current, pool: pool, using: sideEngine,
                    budget: min(questionsLeft, Self.maximumQuestionsPerFact))
                questionsLeft -= outcome.asked
                switch outcome.inspection {
                case .dropped:
                    log(.notDurableStopped(key: record.key.rawValue))
                    dropped += 1
                    continue
                case .ruleConflict:
                    // The stored rule fixes this value and the rule wins: the
                    // old value stays and the change is not written.
                    log(.ruleConflictStopped(key: record.key.rawValue))
                    ruleConflicts += 1
                    continue
                case .duplicate(let kept):
                    log(.nearDuplicateStopped(key: record.key.rawValue, kept: kept))
                    duplicates += 1
                    continue
                case .conflict(let conflictsWith):
                    log(
                        .contradictionFound(
                            key: record.key.rawValue,
                            conflictsWith: conflictsWith))
                    conflicts += 1
                case .none:
                    break
                }
            }
            switch await write(
                record, to: destination, scope: scope,
                session: context.session.id,
                flaggingReversions: !isShared)
            {
            case .stored:
                written += 1
                if isShared { log(.sharedFactWritten(key: record.key.rawValue)) }
            case .reverted:
                written += 1
                log(.reversionFlagged(key: record.key.rawValue))
            case .held:
                // Not written: the person said otherwise and the model did
                // not. The address is disputed, so the next session is shown
                // both rather than one of them. No value is logged, ever.
                log(.guardHeld(key: record.key.rawValue))
                held += 1
            case .failed:
                break
            }
        }
        logConsolidationSummary(
            session: context.session.id, written: written,
            unchanged: unchanged, duplicates: duplicates,
            dropped: dropped, ruleConflicts: ruleConflicts,
            conflicts: conflicts)
        return written
    }

    /// One line per counter that fired, then the total.
    func logConsolidationSummary(
        session: String, written: Int, unchanged: Int,
        duplicates: Int, dropped: Int,
        ruleConflicts: Int, conflicts: Int
    ) {
        if unchanged > 0 { log(.unchangedSkipped(session: session, count: unchanged)) }
        if duplicates > 0 { log(.nearDuplicatesStopped(session: session, count: duplicates)) }
        if dropped > 0 { log(.notDurablesStopped(session: session, count: dropped)) }
        if ruleConflicts > 0 {
            log(.ruleConflictsStopped(session: session, count: ruleConflicts))
        }
        if conflicts > 0 { log(.contradictionsFound(session: session, count: conflicts)) }
        log(.consolidated(session: session, records: written))
    }

    /// Where one record ended up, so the caller keeps the counting and the
    /// logging in one place instead of at each destination.
    enum WriteOutcome {
        case stored
        case reverted
        case held
        case failed
    }

    /// One fact into one store.
    ///
    /// `flaggingReversions` is the consolidation heuristic, not the protocol's
    /// rule: it is on for the project's own store and off for the shared
    /// workspace and for every deliberate tool call.
}
