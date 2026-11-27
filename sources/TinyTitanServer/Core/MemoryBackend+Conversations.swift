import TinyTitanLib
import TinyTitanMemory

// How many conversations the decorator keeps state for, and what leaves with
// one when it is dropped.
//
// Split out of `MemoryBackend.swift` (2026-11-27) under the 500-line-per-file
// rule. The API has no end-of-conversation signal: a client that is finished
// simply stops sending. Without a bound, every distinct (workspace, first user
// message) left a context, a frozen instruction, a turn index and a
// consolidated position in the actor for the life of the process, so a
// long-lived server grew without limit (AUD-0067).

extension MemoryBackend {
    /// Conversations kept at once.
    ///
    /// A conversation is identified by its workspace and its first user
    /// message, so a server sees one per chat a person has ever started. The
    /// most recent are the ones a client can still be resuming; older ones
    /// are history whose only live value is the frozen prompt prefix, and
    /// resuming one of those after this many newer chats rebuilds the prefix
    /// once and costs one prefill. A hundred and twenty-eight is far more
    /// than one person works with in a day and, at a bounded bootstrap each,
    /// keeps the maps in the low megabytes.
    static let maximumConversations = 128

    /// Marks a conversation as the most recently used, for the LRU order.
    func noteConversationUse(_ id: String) {
        if let index = conversationRecency.firstIndex(of: id) {
            conversationRecency.remove(at: index)
        }
        conversationRecency.append(id)
    }

    /// Drops the oldest conversations until the map is within capacity.
    ///
    /// Everything keyed by the conversation goes with it: the frozen
    /// instruction, the journal turn counter and the consolidated position
    /// describe that conversation and nothing else. A pending consolidation
    /// is unaffected: it holds its own reference to the context in
    /// `unconsolidated`, and the consolidated position it advances is
    /// re-derived from the journal if the conversation ever returns.
    func evictConversationsIfOverCapacity() {
        while conversationRecency.count > Self.maximumConversations {
            let oldest = conversationRecency[0]
            forgetConversation(oldest)
            ServerLog.memory(
                "conversation evicted id=\(oldest) reason=capacity "
                    + "capacity=\(Self.maximumConversations)")
        }
    }

    /// Removes every per-conversation entry for one id.
    func forgetConversation(_ id: String) {
        if let index = conversationRecency.firstIndex(of: id) {
            conversationRecency.remove(at: index)
        }
        contexts.removeValue(forKey: id)
        installedInstructions.removeValue(forKey: id)
        turnIndex.removeValue(forKey: id)
        consolidatedThrough.removeValue(forKey: id)
    }

    /// Runs the session-end hook, if consolidation is on. The engine calls
    /// this when a conversation is finished with; nothing calls it
    /// automatically, because the API has no end-of-conversation signal.
    public func endSession(conversation id: String) async {
        guard let context = contexts[id] else { return }
        forgetConversation(id)
        await service.endSession(context)
    }
}
