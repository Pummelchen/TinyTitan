//
//  HTTPServerHandler+Responses.swift
//  TinyTitanServer
//
//  The OpenAI Responses API surface.
//

import Foundation
import NIOCore
import NIOHTTP1
import NIOPosix
import Synchronization
import TinyTitanCore
import TinyTitanLib

extension ServerHTTPHandler {
    func resolveReferences(_ items: [ResponsesAPIRequest.Item]) -> [ResponsesAPIRequest.Item] {
        items.map { item in
            guard item.resolvedType == "item_reference", let id = item.id,
                let stored = responseStore.item(withID: id)
            else { return item }
            return stored
        }
    }

    /// The API reports a failed generation as a response object in state
    /// "failed" carrying the error, then ends the stream.
    func responsesFailureFrames(
        id: String, created: Int, echo: ResponsesAPIEcho, itemState: ResponsesStreamState
    ) -> @Sendable (OpenAIErrorEnvelope) -> [Data] {
        { envelope in
            let failed = ResponsesAPIBuilder.responseObject(
                id: id, created: created, model: self.responseModelID,
                status: "failed", output: [], usage: nil, echo: echo,
                error: (envelope.error.code, envelope.error.message))
            let event = ResponsesAPIBuilder.event(
                "response.failed", sequence: itemState.nextSequence(), ["response": failed])
            return Self.eventFrame(name: "response.failed", object: event).map { [$0] } ?? []
        }
    }

    /// The stored conversation `previous_response_id` continues, or none.
    func priorConversation(
        _ request: ResponsesAPIRequest
    ) throws -> [ResponsesAPIRequest.Item] {
        guard let previous = request.previousResponseID else { return [] }
        guard let entry = responseStore.get(previous) else {
            throw ServerRequestError.notFound(
                message: "Previous response with id '\(previous)' not found.",
                param: "previous_response_id")
        }
        return entry.conversation
    }

    /// OpenAI Responses API endpoint (`POST /v1/responses`). The request is
    /// mapped onto the chat-completions path (see ResponsesAPIMapper) and the
    /// generation is streamed back as Responses-API SSE events (or returned
    /// as a single response object when stream is false). A finished response
    /// is stored (unless store=false) so a later request can continue it by
    /// previous_response_id and the retrieval endpoints can serve it.
    func handleResponses(
        body: ByteBuffer,
        context: ChannelHandlerContext,
        workspace: String? = nil
    ) {
        do {
            let decoded = try JSONDecoder().decode(
                ResponsesAPIRequest.self, from: Data(body.readableBytesView))
            let target = try servedModel(named: decoded.model)
            let prior = try priorConversation(decoded)
            let inputItems = resolveReferences(decoded.inputItems)
            let chatRequest = try ResponsesAPIMapper.chatRequest(
                decoded, priorItems: prior, inputItems: inputItems)
            let request = try validate(chatRequest, for: target)
                .withWorkspace(workspace)
            let echo = ResponsesAPIEcho(
                request: decoded,
                effectiveEffort: target.reasoningProfile.effectiveEffort,
                applied: request.generationConfig)
            let responseID = ResponsesAPIBuilder.responseID()
            let created = Int(Date().timeIntervalSince1970)
            let contextBox = SendableContext(context)
            let streamState = StreamState()
            let itemState = ResponsesStreamState()
            let phaseState = requestPhaseState
            let storedInput = prior + inputItems
            let stores = decoded.stores
            // The flattened chain is what a stored response retains. Refuse an
            // over-budget one before generation rather than running the model
            // and only then discovering the entry cannot be stored.
            if stores {
                let inputBytes = ResponseStore.retainedByteCount(of: storedInput) ?? Int.max
                guard inputBytes <= responseStore.maximumEntryBytes else {
                    throw ServerRequestError.invalid(
                        message:
                            "the stored conversation exceeds the server's "
                            + "\(responseStore.maximumEntryBytes)-byte budget",
                        param: decoded.previousResponseID == nil
                            ? "input" : "previous_response_id",
                        code: "stored_response_too_large")
                }
            }
            let startStream: @Sendable () -> Void = {
                guard request.stream,
                    streamState.start(
                        eventLoop: contextBox.value.eventLoop,
                        interval: self.heartbeatInterval,
                        ping: {
                            self.writeHeartbeat(contextBox.value)
                        })
                else { return }
                let future = self.beginResponsesStream(
                    contextBox.value, id: responseID, created: created,
                    echo: echo, itemState: itemState)
                streamState.setStartFuture(future)
            }
            let onQueued: @Sendable () -> Void = {
                phaseState.set("queued")
                ServerLog.queued(id: responseID)
                startStream()
            }
            activeTask = childChannels.startTask {
                defer { streamState.stop() }
                let started = ContinuousClock.now
                ServerLog.accepted(id: responseID, streaming: request.stream)
                let outbox: SSEOutbox? =
                    request.stream
                    ? SSEOutbox(capacity: Self.maximumPendingStreamChunks)
                    : nil
                let drainer = outbox.map { outbox in
                    Task { [self] in
                        await self.drainOutbox(
                            contextBox.value, outbox: outbox,
                            streamState: streamState)
                    }
                }
                do {
                    let completion = try await self.coordinator.run(onQueued: onQueued) {
                        try Task.checkCancellation()
                        startStream()
                        try await streamState.waitUntilStarted()
                        try Task.checkCancellation()
                        phaseState.set("generating")
                        ServerLog.generating(id: responseID)
                        return try await self.backend.generate(request) { event in
                            guard request.stream, let outbox else { return }
                            self.enqueueResponsesEvent(
                                event, id: responseID, echo: echo,
                                itemState: itemState,
                                outbox: outbox, context: contextBox.value)
                        }
                    }
                    ServerLog.completed(
                        id: responseID,
                        duration: started.duration(to: .now),
                        completion: completion)
                    if request.stream, let outbox {
                        streamState.stop()
                        _ = self.finishResponsesStream(
                            contextBox.value, id: responseID, created: created, echo: echo,
                            completion: completion, itemState: itemState, outbox: outbox,
                            stores: stores, storedInput: storedInput,
                            namespaces: echo.namespaces)
                    } else {
                        var object = self.finalResponsesObject(
                            id: responseID, created: created, echo: echo,
                            completion: completion, itemState: nil)
                        // Attempt the store *before* writing, so a refused entry
                        // is reported as `store: false` instead of telling the
                        // client it can be retrieved by an id that is not there.
                        if stores,
                            !self.storeResponse(
                                id: responseID, object: object,
                                input: storedInput, completion: completion,
                                namespaces: echo.namespaces)
                        {
                            object["store"] = false
                        }
                        self.writeJSON(contextBox.value, status: .ok, object: object)
                    }
                } catch {
                    streamState.stop()
                    self.handleAsyncFailure(
                        error, context: contextBox.value, id: responseID,
                        phase: phaseState.value, stream: request.stream, outbox: outbox,
                        streamState: streamState,
                        surface: .responses,
                        failureFrames: self.responsesFailureFrames(
                            id: responseID, created: created, echo: echo, itemState: itemState))
                }
                if let drainer {
                    await Self.awaitDrainer(drainer)
                }
            }
        } catch let error as ServerRequestError {
            writeRequestError(
                context, error,
                status: HTTPResponseStatus(statusCode: error.httpStatus),
                surface: .responses)
        } catch {
            writeRequestError(
                context, RequestDecodingFailure.serverError(error, body: body),
                status: .badRequest, surface: .responses)
        }
    }

    /// The response object of a finished generation. In a stream the output
    /// order is the order items were announced; otherwise message first.
    func finalResponsesObject(
        id: String,
        created: Int,
        echo: ResponsesAPIEcho,
        completion: ServerCompletion,
        itemState: ResponsesStreamState?
    ) -> [String: Any] {
        let ids = ResponsesAPIBuilder.itemIDs(responseID: id, completion: completion)
        var output: [[String: Any]]
        if let itemState {
            var slots: [Int: [String: Any]] = [:]
            for (ordinal, item) in itemState.reasoningItems.enumerated() {
                slots[item.index] = ResponsesAPIBuilder.reasoningItem(
                    id: ResponsesAPIBuilder.reasoningItemID(responseID: id, index: ordinal),
                    text: item.text)
            }
            if let index = itemState.messageIndex {
                slots[index] = ResponsesAPIBuilder.messageItem(
                    id: ids.message, role: "assistant", text: completion.content,
                    status: "completed")
            }
            for (ordinal, index) in itemState.callIndices.enumerated()
            where ordinal < completion.toolCalls.count {
                let call = completion.toolCalls[ordinal]
                slots[index] = ResponsesAPIBuilder.functionCallItem(
                    id: ids.calls[ordinal], name: call.name, arguments: call.argumentsJSON,
                    callID: call.id, status: "completed", namespace: echo.namespaces[call.name])
            }
            output = slots.keys.sorted().compactMap { slots[$0] }
        } else {
            output = ResponsesAPIBuilder.outputItems(
                completion: completion, responseID: id,
                namespaces: echo.namespaces)
        }
        if output.isEmpty {
            output = [
                ResponsesAPIBuilder.messageItem(
                    id: ids.message, role: "assistant", text: "", status: "completed")
            ]
        }
        let terminal = ResponsesAPIBuilder.terminalStatus(for: completion)
        return ResponsesAPIBuilder.responseObject(
            id: id, created: created, model: responseModelID, status: terminal.status,
            output: output, usage: completion.usage, echo: echo,
            incompleteReason: terminal.reason,
            completedAt: Int(Date().timeIntervalSince1970))
    }

    /// Stores a finished response, reporting whether it can actually be
    /// retrieved. `false` means the entry was over budget (or unencodable) and
    /// the caller must report `store: false`; it is never silently dropped
    /// while the response claims otherwise.
    @discardableResult
    func storeResponse(
        id: String,
        object: [String: Any],
        input: [ResponsesAPIRequest.Item],
        completion: ServerCompletion,
        namespaces: [String: String]
    ) -> Bool {
        guard let data = try? JSONSerialization.data(withJSONObject: object) else {
            ServerLog.diagnostic("responses store: response \(id) could not be encoded; not stored")
            return false
        }
        let outcome = responseStore.put(
            id: id,
            entry: ResponseStore.Entry(
                responseJSON: data, inputItems: input,
                outputItems: ResponsesAPIMapper.outputAsInput(
                    completion: completion, responseID: id, namespaces: namespaces),
                created: Date()))
        switch outcome {
        case .stored:
            return true
        case .rejectedEntryTooLarge, .rejectedNoCapacity, .rejectedUnmeasurableEntry:
            ServerLog.diagnostic(
                "responses store: response \(id) not stored (\(outcome)); "
                    + "budget \(responseStore.maximumEntryBytes) bytes per entry, "
                    + "\(responseStore.maximumTotalBytes) bytes total")
            return false
        }
    }

}
