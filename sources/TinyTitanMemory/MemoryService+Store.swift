import ContinuityCore
import Foundation

// The store internals: write, inspect, inspect-for-write and the active store.
//
// Split out of `MemoryService.swift` (2026-09-28) under the 500-line-per-file
// rule (Task 8 of the cleanup runbook) as pure code motion. Members whose
// callers stay behind widened from `private` to internal.
extension MemoryService {

    /// One fact into one store.
    ///
    /// `flaggingReversions` is the consolidation heuristic, not the protocol's
    /// rule: it is on for the project's own store and off for the shared
    /// workspace and for every deliberate tool call.
    func write(
        _ record: MemoryRecord,
        to store: any MemoryStore,
        scope: MemoryScope,
        session: String,
        flaggingReversions: Bool
    ) async -> WriteOutcome {
        var stamped = record
        stamped.sourceSession = session
        do {
            if let continuity = store as? ContinuityStore {
                switch try await continuity.set(
                    stamped, in: scope,
                    guarding: configuration.guardsUserFacts,
                    flaggingReversions: flaggingReversions)
                {
                case .stored: return .stored
                case .reverted: return .reverted
                case .heldByGuard: return .held
                }
            }
            // Degraded to process-local storage: there is no provenance to
            // enforce precedence with, and the protocol says so rather than
            // pretending.
            _ = try await store.set(
                stamped, in: scope,
                guarding: configuration.guardsUserFacts)
            return .stored
        } catch {
            log(.toolFailed(tool: "consolidation", detail: "\(error)"))
            return .failed
        }
    }

    /// What the side-engine said about one new fact, and what it cost.
    struct SideEngineVerdict {
        /// `false` means the fact is not worth keeping; `nil` is no answer.
        let durable: Bool?
        /// `.conflict` means a stored rule fixes this value; `nil` is no
        /// answer, and `.update` leaves the write alone.
        let supersession: MemorySupersession?
        let duplicate: String?
        let conflict: String?
        let asked: Int
    }

    /// T2, T4, T5 and T3 over one new fact, stopping as soon as the budget is
    /// gone.
    ///
    /// Durability comes first and ends the check when the answer is no — a fact
    /// that is not worth keeping needs no comparison. The rule check needs both
    /// the stored value (`current`) and a rule the caller found. Candidates are
    /// facts in the same leading segment with a different key; the duplicate is
    /// looked for before the contradiction, and finding one ends the search
    /// because there is nothing to add about a fact already stored.
    ///
    /// `isModelDerived` is false for a fact the person asserted, which is not
    /// the engine's to discard or to hold back behind a rule.
    func inspect(
        _ record: MemoryRecord,
        current: MemoryRecord?,
        rule: String?,
        among candidates: [MemoryRecord],
        using engine: any MemorySideEngine,
        budget: Int,
        isModelDerived: Bool
    ) async -> SideEngineVerdict {
        let fact = MemoryFact(key: record.key.rawValue, value: record.value)
        var asked = 0
        var durable: Bool?
        if isModelDerived, budget > 0 {
            asked += 1
            durable = await engine.isDurable(fact)
            if durable == false {
                return SideEngineVerdict(
                    durable: durable, supersession: nil,
                    duplicate: nil, conflict: nil, asked: asked)
            }
        }
        var supersession: MemorySupersession?
        if isModelDerived, let current, let rule, asked < budget {
            asked += 1
            supersession = await engine.supersedes(
                MemoryFact(key: current.key.rawValue, value: current.value),
                fact, rule: rule)
        }
        var conflict: String?
        for candidate in candidates {
            guard candidate.key != record.key,
                candidate.key.category == record.key.category
            else { continue }
            guard asked < budget else { break }
            let existing = MemoryFact(
                key: candidate.key.rawValue,
                value: candidate.value)
            asked += 1
            // Stored first: the prompts answer YES in that order and NO
            // reversed, so the order is part of the contract.
            if await engine.duplicates(existing, fact) == true {
                return SideEngineVerdict(
                    durable: durable, supersession: supersession,
                    duplicate: candidate.key.rawValue,
                    conflict: nil, asked: asked)
            }
            guard asked < budget else { break }
            asked += 1
            if conflict == nil, await engine.contradicts(existing, fact) == true {
                conflict = candidate.key.rawValue
            }
        }
        return SideEngineVerdict(
            durable: durable, supersession: supersession,
            duplicate: nil, conflict: conflict, asked: asked)
    }

    /// What the engine's answers mean for this write.
    enum Inspection {
        case none
        case dropped
        case ruleConflict
        case duplicate(String)
        case conflict(String)
    }

    /// Runs the questions and reduces them to one outcome, so the write path
    /// reads as one decision rather than four.
    func inspectForWrite(
        _ record: MemoryRecord,
        current: MemoryRecord?,
        pool: [MemoryRecord],
        using engine: any MemorySideEngine,
        budget: Int
    ) async -> (inspection: Inspection, asked: Int) {
        let rule = MemoryRuleLookup.rule(for: record.key, among: pool)
        let verdict = await inspect(
            record, current: current, rule: rule, among: pool,
            using: engine, budget: budget,
            isModelDerived: !record.isUserAsserted)
        let inspection: Inspection
        if verdict.durable == false {
            inspection = .dropped
        } else if verdict.supersession == .conflict {
            inspection = .ruleConflict
        } else if let kept = verdict.duplicate {
            inspection = .duplicate(kept)
        } else if let conflictsWith = verdict.conflict {
            inspection = .conflict(conflictsWith)
        } else {
            inspection = .none
        }
        return (inspection, verdict.asked)
    }

    static func fold(_ value: String) -> String {
        value.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: ".\"'"))
    }

    func activeStore(for scope: MemoryScope) async -> any MemoryStore {
        guard !isDegraded, let workspace = await workspace(for: scope) else { return localStore }
        return workspace.store
    }
}
