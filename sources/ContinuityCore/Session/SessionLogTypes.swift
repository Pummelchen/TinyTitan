import Foundation

// The session log's value types: options, a rendered turn, the snapshot and
// the response-id accessor.
//
// Split out of `SessionLog.swift` (2026-09-28) under the 500-line-per-file rule
// (Task 8 of the cleanup runbook) as pure code motion.
public struct SessionLogOptions: Sendable, Equatable {
    /// Bytes of log a task may hold in memory. Zero disables the bound.
    ///
    /// The oldest whole sessions are dropped from memory when it is exceeded.
    /// They remain in the journal file: this bounds what the process holds,
    /// not what was recorded.
    public var maxBytesPerTask: Int

    /// Write one event per streamed chunk in addition to the completed reply.
    ///
    /// Off by default. It exists for callers who need a partial reply to
    /// survive a crash mid-stream; it costs one event per chunk and readers
    /// must fold, which `transcript` and `turns` already do.
    public var persistsChunks: Bool

    public init(
        persistsChunks: Bool = false,
        maxBytesPerTask: Int = 64 << 20
    ) {
        self.persistsChunks = persistsChunks
        self.maxBytesPerTask = maxBytesPerTask
    }
}

/// One exchange. The response is absent when a reply never arrived, which is
/// itself worth seeing.
public struct SessionTurn: Sendable, Equatable {
    public let sessionID: UUID
    public let promptEventID: UUID?
    public let prompt: String
    public let response: String?
    /// The reply's measurements, when the reply carried any.
    public let responseRecord: ResponseRecord?
    /// When the reply landed. Nil while the turn is unanswered.
    public let completedAt: Date?
    /// When the prompt arrived.
    public let timestamp: Date

    public init(
        sessionID: UUID,
        promptEventID: UUID?,
        prompt: String,
        response: String?,
        responseRecord: ResponseRecord? = nil,
        completedAt: Date? = nil,
        timestamp: Date
    ) {
        self.sessionID = sessionID
        self.promptEventID = promptEventID
        self.prompt = prompt
        self.response = response
        self.responseRecord = responseRecord
        self.completedAt = completedAt
        self.timestamp = timestamp
    }
}

public struct SessionLogSnapshot: Codable, Sendable, Equatable {
    public var tasks: [ContinuityTask]
    public var sessions: [Session]
    public var events: [SessionEvent]

    public init(
        tasks: [ContinuityTask] = [],
        sessions: [Session] = [],
        events: [SessionEvent] = []
    ) {
        self.tasks = tasks
        self.sessions = sessions
        self.events = events
    }
}

extension SessionEvent {
    func withResponseID(_ id: UUID) -> SessionEvent {
        SessionEvent(
            id: self.id, sessionID: sessionID, taskID: taskID,
            timestamp: timestamp, kind: kind, payload: payload, responseID: id)
    }
}
