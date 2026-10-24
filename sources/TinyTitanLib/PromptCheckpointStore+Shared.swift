import Foundation
import Synchronization

extension PromptCheckpointStore {
    /// Process-wide stores, keyed by their storage configuration.
    ///
    /// A model session gets a fresh store today, so with the disk tier off a
    /// new session cannot see a checkpoint the previous one captured and the
    /// system segment is prefilled again. Keeping the actor, not just the disk
    /// directory, lets a checkpoint survive a model reload while the server
    /// runs. Interior checkpoints are content-addressed, so two sessions can
    /// share one store without sharing a conversation: only an entry whose
    /// exact prefix and domain match can be found.
    ///
    /// One instance per configuration also means one memory/disk ledger, so
    /// the existing budget and LRU rules apply unchanged rather than being
    /// multiplied by the number of live sessions.
    private static let registry =
        Mutex<[ServerPromptCacheStorageConfiguration: PromptCheckpointStore]>([:])

    /// The store for `configuration`, creating it on first use.
    ///
    /// Two concurrent loads race to create the actor; the loser's instance is
    /// discarded inside the same lock, so both callers observe one store and
    /// no bytes are charged twice.
    static func sharedInstance(
        configuration: ServerPromptCacheStorageConfiguration
    ) throws -> PromptCheckpointStore {
        if let existing = registry.withLock({ $0[configuration] }) {
            return existing
        }
        let created = try PromptCheckpointStore(configuration: configuration)
        return registry.withLock { registry in
            if let existing = registry[configuration] { return existing }
            registry[configuration] = created
            return created
        }
    }
}
