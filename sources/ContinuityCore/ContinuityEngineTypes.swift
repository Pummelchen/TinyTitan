import Foundation

// The engine's configuration and statistics types.
//
// Split out of `ContinuityEngine.swift` (2026-09-28) under the 500-line-per-
// file rule (Task 8 of the cleanup runbook) as pure code motion.
/// How the engine behaves.
public struct ContinuityConfiguration: Sendable {
    public var memoryLimits: MemoryLimits
    public var sessionLogOptions: SessionLogOptions
    public var defaultBudget: ContextBudget
    /// Write prompts and replies to the journal.
    ///
    /// Separate from journalling memory because the two differ in
    /// sensitivity: memory holds distilled facts, the session log holds
    /// everything a person typed. A caller who wants durable continuity
    /// without a transcript on disk turns this off and keeps the rest.
    public var journalsSessionContent: Bool
    /// Compact the journal once it holds more than this many records.
    /// Zero disables automatic compaction.
    public var compactionThreshold: Int

    public init(
        memoryLimits: MemoryLimits = .default,
        sessionLogOptions: SessionLogOptions = .init(),
        defaultBudget: ContextBudget = ContextBudget(),
        journalsSessionContent: Bool = true,
        compactionThreshold: Int = 20_000
    ) {
        self.memoryLimits = memoryLimits
        self.sessionLogOptions = sessionLogOptions
        self.defaultBudget = defaultBudget
        self.journalsSessionContent = journalsSessionContent
        self.compactionThreshold = compactionThreshold
    }
}

public struct ContinuityStatistics: Sendable, Equatable {
    public let taskCount: Int
    public let sessionCount: Int
    public let eventCount: Int
    public let memoryItemCount: Int
    /// Bytes of task memory held in this process.
    public let memoryBytes: Int
    /// Bytes of session log held in this process. The journal file may hold
    /// more; this is what is resident.
    public let logBytes: Int
    public let journaledRecords: Int
}
