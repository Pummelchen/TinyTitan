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

/// Complete lines of a journal that did not decode on replay.
///
/// A final line with no terminator is a torn write and is dropped without
/// appearing here: the append that would have completed it never reported
/// success. A *complete* line that will not decode is a different thing — a
/// bit flip, a truncated-then-reused file, or a record from a build that
/// writes a case this one does not know — and the good records around it are
/// still worth replaying. Recording the position lets a caller say which line
/// was rejected, and it is what stops a compaction from replacing the file
/// with a checkpoint that does not contain it.
///
/// No record contents are kept: the line was undecodable, so there is nothing
/// safe to quote, and a damaged line may hold a prompt.
public struct JournalReplayDamage: Sendable, Equatable {
    public struct Line: Sendable, Equatable {
        /// 1-based, counting blank lines, so it matches what an editor shows.
        public let number: Int
        /// Byte offset of the line's first byte in the journal.
        public let byteOffset: Int

        public init(number: Int, byteOffset: Int) {
            self.number = number
            self.byteOffset = byteOffset
        }
    }

    public let lines: [Line]

    public init(lines: [Line]) {
        self.lines = lines
    }

    /// Only the line numbers, for a message that names them without offsets.
    public var lineNumbers: [Int] { lines.map(\.number) }
}

/// One read of a journal: the records that decoded and the complete lines
/// that did not. A caller that only wants records uses
/// `FileJournal.read(contentsOf:)`; a caller deciding whether the file can be
/// rewritten wants both.
public struct JournalReplay: Sendable {
    public let records: [JournalRecord]
    public let damage: JournalReplayDamage?

    public init(records: [JournalRecord], damage: JournalReplayDamage?) {
        self.records = records
        self.damage = damage
    }
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
    /// A replay skipped complete lines that did not decode.
    ///
    /// Rewriting the journal now would replace those bytes with a checkpoint
    /// that does not contain them, which turns a recoverable file into a
    /// permanently damaged one. The records that did decode are still
    /// returned; this is what refuses the rewrite until someone decides.
    case damagedReplay(URL, lines: [Int])
    /// The journal belongs to another user, so appending to it would write
    /// through a file this process does not own.
    case wrongOwner(URL, uid: uid_t)
    /// The journal is readable or writable by someone other than its owner.
    /// It holds prompts and replies, so a permissive mode is refused rather
    /// than silently chmod'ed: the operator is the one who knows whether the
    /// file predates this policy.
    case insecurePermissions(URL, mode: mode_t)

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
        case .damagedReplay(let url, let lines):
            let named = lines.map(String.init).joined(separator: ", ")
            return
                "journal at \(url.path) has \(lines.count) complete record(s) that did not "
                + "decode (lines \(named)); refusing to replace the file"
        case .wrongOwner(let url, let uid):
            return
                "journal at \(url.path) is owned by uid \(uid), not this process; "
                + "refusing to write it"
        case .insecurePermissions(let url, let mode):
            return
                "journal at \(url.path) has mode \(String(mode, radix: 8)) instead of 600; "
                + "refusing to write a file other users can read (chmod 600 it to proceed)"
        }
    }
}
