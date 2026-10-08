import ContinuityCore
import Foundation

/// Session identity: the mapping from a caller's own session name to the
/// continuity session the engine holds under it.
///
/// This is the seam that answers *which session does this write belong to*, so
/// it lives apart from the store's record operations rather than inside them.
extension ContinuityStore {
    /// The live continuity session for a memory session name, or nil when the
    /// caller named none.
    ///
    /// `sessionIDs` remembers ids `begin(session:)` handed out, and nothing
    /// invalidates it when retention drops the session behind one —
    /// `pruneSessions` reports what it removed and `enforceByteBudget` does not.
    /// A stale entry was enough to fail a write: `remember(sessionID:)` rejects
    /// a session the log no longer holds, and `translate` renders that as
    /// "memory backend unavailable", so `memory_set` blamed the backend for a
    /// cache miss and the fact the model asked to keep was not kept.
    ///
    /// The name, not the id, is what the conversation carries, so a dropped
    /// session is resolved again by name the same way `begin(session:)` resolves
    /// one. If the log holds no session under the name, a new one is opened. If
    /// that fails too the write goes through unattributed: a fact with no author
    /// beats a fact that never landed, and the only reason an open fails is an
    /// append that has already recorded itself on the engine.
    func liveSession(for name: String?, taskID: UUID) async -> UUID? {
        guard let name, let cached = sessionIDs[name] else { return nil }
        if await engine.sessions(taskID: taskID).contains(where: { $0.id == cached }) {
            return cached
        }
        if let known = await engine.session(externalID: name, taskID: taskID) {
            remember(known, as: name)
            return known.id
        }
        guard
            let opened = try? await engine.beginSession(
                taskID: taskID, model: nil, externalID: name)
        else { return nil }
        remember(opened, as: name)
        return opened.id
    }

    private func remember(_ session: Session, as name: String) {
        sessionIDs[name] = session.id
        // The label map is filled once per task, by walking the sessions that
        // exist at that moment, so a session resolved after the first read has
        // to be labelled here or its records come back saying nobody wrote them.
        sessionLabels[session.id] = name
    }
}
