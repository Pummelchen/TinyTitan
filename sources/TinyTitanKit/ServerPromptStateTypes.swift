import Foundation

// The prompt-state store's value types: its storage configuration, the save
// result and the store error.
//
// Split out of `ServerPromptStateStore.swift` (2026-09-28) under the
// 500-line-per-file rule (Task 8 of the cleanup runbook) as pure code motion.
struct ServerPromptCacheStorageConfiguration: Sendable, Equatable {
    let memoryLimitBytes: Int
    let diskDirectory: URL?
    let diskLimitBytes: Int

    init(
        memoryLimitBytes: Int,
        diskDirectory: URL?,
        diskLimitBytes: Int
    ) {
        precondition(memoryLimitBytes >= 0)
        precondition(diskLimitBytes >= 0)
        self.memoryLimitBytes = memoryLimitBytes
        self.diskDirectory = diskDirectory
        self.diskLimitBytes = diskLimitBytes
    }
}

struct ServerPromptStateSaveResult: Sendable, Equatable {
    let unbackedEntryIDs: [UUID]
    let diskError: String?
    let memoryBytes: Int
    let diskBytes: Int
}

enum ServerPromptStateStoreError: Error, CustomStringConvertible {
    case missing(UUID)
    case corrupt(UUID, String)

    var description: String {
        switch self {
        case .missing(let id):
            "prompt-cache state \(id.uuidString.lowercased()) is unavailable"
        case .corrupt(let id, let reason):
            "prompt-cache state \(id.uuidString.lowercased()) is corrupt: \(reason)"
        }
    }
}
