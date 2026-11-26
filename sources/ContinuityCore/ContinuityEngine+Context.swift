import Foundation

// Context assembly: the block of text the model is given for a task.
//
// Split out of `ContinuityEngine.swift` under the 500-line-per-file rule
// (audit 0050) as pure code motion. The actor's stored state is internal to
// the module, so the extension reads it exactly as the method did; no
// signature changed.

extension ContinuityEngine {

    // MARK: - Context

    /// Build the block of text to put in front of the model.
    ///
    /// Records a `contextAssembled` event when a session is given, so the log
    /// can later show what the model was looking at when it answered.
    public func assembleContext(
        taskID: UUID,
        sessionID: UUID? = nil,
        focus: String? = nil,
        budget: ContextBudget? = nil,
        query: MemoryQuery = .active
    ) async throws -> ContextSnapshot {
        guard let task = await sessionLog.task(taskID) else {
            throw ContinuityError.unknownTask(taskID)
        }
        let effectiveBudget = budget ?? configuration.defaultBudget
        let candidates = await memory.query(taskID: taskID, query)
        let all = await memory.query(
            taskID: taskID,
            MemoryQuery(
                statuses: [.active, .disputed],
                order: .address))
        var index: [String: MemoryItem] = [:]
        for item in all { index[item.address] = item }
        let turns =
            effectiveBudget.recentTurnCount > 0
            ? await sessionLog.turns(taskID: taskID, limit: effectiveBudget.recentTurnCount)
            : []
        let request = ContextRequest(
            task: task,
            sessionID: sessionID,
            items: candidates,
            index: index,
            turns: turns,
            budget: effectiveBudget,
            focus: focus)
        let snapshot = try assembler.assemble(request)
        if let sessionID {
            // Best-effort on the same terms as the memory event in
            // `remember`: the
            // snapshot has been assembled and returned, and failing this call
            // would answer a question that was answered. A refusal is still
            // reported on the durability channel so an unknown or pruned
            // session cannot silently cost the context record.
            do {
                _ = try await sessionLog.recordContextAssembled(
                    sessionID: sessionID,
                    snapshot: snapshot)
            } catch {
                journalWriteFailed(error)
            }
        }
        return snapshot
    }

}
