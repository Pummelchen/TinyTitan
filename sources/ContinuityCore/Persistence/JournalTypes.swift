import Foundation

// The journal's value types and protocol: the record enum, the journal
// interface with its no-op defaults, the null journal, and the error type.
//
// Split out of `Journal.swift` (2026-09-28) under the 500-line-per-file rule
// (Task 8 of the cleanup runbook) as pure code motion.
/// One durable line of the record.
///
/// The journal is a log of facts, not of commands. Replaying it applies the
/// same values again in the same order, so a replay of a replay is identical
/// and a partially written tail can be dropped without corrupting what came
/// before.
public enum JournalRecord: Codable, Sendable, Equatable {
    case task(ContinuityTask)
    case session(Session)
    case event(SessionEvent)
    case memory(MemoryItem)
    case memoryVersion(MemoryVersion)
    /// A compaction point. Everything before it in the file is redundant.
    case checkpoint(SessionLogSnapshot, MemorySnapshot)
}

/// Durable storage for the engine's state.
///
/// RAM stays the source of truth during a run. The journal exists so that
/// state survives a restart, which is the whole point of a system built for
/// work that spans months.
public protocol ContinuityJournal: Sendable {
    func append(_ record: JournalRecord) async throws
    func append(_ records: [JournalRecord]) async throws
    /// Every record in write order. A truncated final line is discarded.
    func replay() async throws -> [JournalRecord]
    /// Replace the file with a single checkpoint.
    func compact(sessionLog: SessionLogSnapshot, memory: MemorySnapshot) async throws
    /// Discard everything.
    func truncate() async throws
}

extension ContinuityJournal {
    public func append(_ records: [JournalRecord]) async throws {
        for record in records { try await append(record) }
    }
}

/// Keeps nothing. The default, so an engine that was never given a file does
/// not quietly write one.
public struct NullJournal: ContinuityJournal {
    public init() {}
    public func append(_ record: JournalRecord) async throws {}
    public func replay() async throws -> [JournalRecord] { [] }
    public func compact(sessionLog: SessionLogSnapshot, memory: MemorySnapshot) async throws {}
    public func truncate() async throws {}
}

public enum JournalError: Error, CustomStringConvertible {
    case cannotOpen(URL, underlying: String)
    case notAFile(URL)
    /// Another process already has this journal open for writing.
    case locked(URL)
    case writeFailed(URL, errno: Int32)
    /// The journal exists but could not be read.
    ///
    /// Deliberately not the same as an empty journal, and not `cannotOpen`
    /// either: this is the case that has to stop a compaction. The engine
    /// replays before it rewrites, so a read that reported "no records" for a
    /// file it merely failed to read would checkpoint over the only copy.
    case readFailed(URL, errno: Int32)

    public var description: String {
        switch self {
        case .cannotOpen(let url, let underlying):
            return "cannot open journal at \(url.path): \(underlying)"
        case .notAFile(let url):
            return "journal path \(url.path) is not a regular file"
        case .locked(let url):
            return "another process is already writing the journal at \(url.path)"
        case .writeFailed(let url, let code):
            return "writing \(url.path) failed: \(String(cString: strerror(code)))"
        case .readFailed(let url, let code):
            return "reading \(url.path) failed: \(String(cString: strerror(code)))"
        }
    }
}
