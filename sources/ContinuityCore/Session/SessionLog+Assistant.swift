import Foundation

// Assistant-response recording: the completed reply, the streaming begin /
// chunk / complete path and their byte-budget bookkeeping.
//
// Split out of `SessionLog.swift` (2026-09-28) under the 500-line-per-file rule
// (Task 8 of the cleanup runbook) as pure code motion. The actor's stored
// properties and helpers widened from `private` to internal because the
// methods that stay call them.
extension SessionLog {

    /// Record a reply that is already complete.
    @discardableResult
    public func recordAssistantResponse(
        sessionID: UUID,
        _ record: ResponseRecord,
        now: Date = Date()
    ) async throws -> SessionEvent {
        let session = try requireOpenSession(sessionID)
        let event = SessionEvent(
            sessionID: sessionID, taskID: session.taskID,
            timestamp: now, kind: .assistantResponse,
            payload: .response(record))
        await append(event)
        return event
    }

    // MARK: - Streaming replies

    /// Open a streamed reply and return the identifier its chunks belong to.
    @discardableResult
    public func beginAssistantResponse(
        sessionID: UUID,
        model: String? = nil,
        requestID: String? = nil,
        now: Date = Date()
    ) async throws -> UUID {
        let session = try requireOpenSession(sessionID)
        let event = SessionEvent(
            sessionID: sessionID, taskID: session.taskID,
            timestamp: now, kind: .assistantResponseStarted,
            payload: .none)
        openResponses[event.id] = OpenResponse(
            sessionID: sessionID,
            taskID: session.taskID,
            model: model,
            requestID: requestID,
            startedAt: now)
        await append(event.withResponseID(event.id))
        return event.id
    }

    /// Add a piece of a streamed reply.
    ///
    /// The text is buffered, not appended as an event, so the completed reply
    /// appears exactly once in the log. A chunk event is written only when
    /// `SessionLogOptions.persistsChunks` is set, and readers then ignore
    /// chunks for any response that also has a completion.
    public func appendAssistantChunk(
        responseID: UUID,
        text: String,
        now: Date = Date()
    ) async throws {
        guard var open = openResponses[responseID] else {
            throw ContinuityError.unknownSession(responseID)
        }
        open.buffer += text
        open.chunkCount += 1
        openResponses[responseID] = open
        guard options.persistsChunks else { return }
        await append(
            SessionEvent(
                sessionID: open.sessionID, taskID: open.taskID,
                timestamp: now, kind: .assistantResponseChunk,
                payload: .text(text), responseID: responseID))
    }

    /// Close a streamed reply, writing the assembled text as one event.
    @discardableResult
    public func completeAssistantResponse(
        responseID: UUID,
        inputTokens: Int? = nil,
        outputTokens: Int? = nil,
        finishReason: String? = nil,
        responseIdentifier: String? = nil,
        now: Date = Date()
    ) async throws -> SessionEvent {
        guard let open = openResponses.removeValue(forKey: responseID) else {
            throw ContinuityError.unknownSession(responseID)
        }
        let latency = Int(now.timeIntervalSince(open.startedAt) * 1000)
        let record = ResponseRecord(
            text: open.buffer,
            model: open.model,
            requestID: open.requestID,
            responseID: responseIdentifier,
            inputTokens: inputTokens,
            outputTokens: outputTokens,
            latencyMilliseconds: latency,
            finishReason: finishReason)
        let event = SessionEvent(
            sessionID: open.sessionID, taskID: open.taskID,
            timestamp: now, kind: .assistantResponseCompleted,
            payload: .response(record), responseID: responseID)
        await append(event)
        return event
    }
}
