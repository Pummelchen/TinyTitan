import Foundation
import NIOCore
import NIOEmbedded
import NIOHTTP1
import Testing

@testable import TinyTitanCore
@testable import TinyTitanLib
@testable import TinyTitanServerCore

/// One request contributes exactly one increment to `inFlightRequests` and must
/// contribute exactly one decrement (AUD-0066).
///
/// Three paths used to end a request twice — an outbox abandoned before its
/// stream head existed (the ordinary writer also counted it), and a refused
/// header block answered *before* the increment. The `> 0` guard hid both by
/// eating the next pipelined request's count, which is how a generating request
/// lost the never-close-mid-generation rule and how `errorCaught` could cancel
/// the wrong task. The cancellation path had the opposite defect: it wrote
/// nothing and decremented nothing, leaking a phantom request per cancellation.
/// A backend that is never asked to generate in these tests: the accounting
/// paths under test run before a request reaches it.
private actor IdleBackend: ServerInferenceBackend {
    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        ServerCompletion(
            content: "", toolCalls: [], finishReason: "stop",
            usage: OpenAIUsage(
                promptTokens: 0, completionTokens: 0, totalTokens: 0,
                cachedTokens: 0),
            stopSequence: nil)
    }
}

@Suite struct InFlightAccountingTests {
    private func makeHandler() -> (EmbeddedChannel, ServerHTTPHandler) {
        let channel = EmbeddedChannel()
        let handler = ServerHTTPHandler(
            modelID: "test-model",
            backend: IdleBackend(),
            coordinator: ServerCoordinator(queueLimit: 1),
            heartbeatInterval: .seconds(5),
            reasoningProfile: .default,
            router: nil,
            childChannels: ChildChannelRegistry(maximumChannels: 2),
            responseStore: ResponseStore(capacity: 2),
            idleReadTimeout: .seconds(30))
        return (channel, handler)
    }

    /// A neighbour's generation is in flight; an oversized header block is
    /// answered before it is counted, so it must leave the count alone.
    @Test func aRefusedHeaderBlockDoesNotConsumeAnotherRequestsCount() async throws {
        let (channel, handler) = makeHandler()
        try await channel.pipeline.addHandler(handler).get()
        handler.inFlightRequests = 1  // the pipelined neighbour, still generating

        var head = HTTPRequestHead(
            version: .http1_1, method: .POST, uri: "/v1/chat/completions")
        for index in 0..<(TinyTitanHTTPServer.maximumRequestHeaderFields + 1) {
            head.headers.add(name: "x-\(index)", value: "v")
        }
        try channel.writeInbound(HTTPServerRequestPart.head(head))
        channel.embeddedEventLoop.run()

        #expect(
            handler.inFlightRequests == 1,
            "the refusal took the neighbour's in-flight count")
        _ = try? channel.finish()
    }

    /// An outbox abandoned before the stream head existed is ended by the
    /// ordinary response writer; the drainer must not count it again.
    @Test func anAbandonedOutboxIsNotCountedByTheDrainer() async throws {
        let (channel, handler) = makeHandler()
        try await channel.pipeline.addHandler(handler).get()
        let handlerContext = try channel.pipeline.syncOperations.context(handler: handler)

        let outbox = SSEOutbox(capacity: 8)
        let streamState = StreamState()
        outbox.abandon()
        handler.inFlightRequests = 1

        await handler.drainOutbox(
            handlerContext, outbox: outbox, streamState: streamState)
        channel.embeddedEventLoop.run()

        #expect(
            handler.inFlightRequests == 1,
            "the drainer decremented a request the writer already ended")
        _ = try? channel.finish()
    }

    /// A cancellation that writes nothing still ends the request, or the
    /// connection carries a phantom in-flight request for ever.
    @Test func aCancellationWithoutAResponseStillEndsTheRequest() async throws {
        let (channel, handler) = makeHandler()
        try await channel.pipeline.addHandler(handler).get()
        let handlerContext = try channel.pipeline.syncOperations.context(handler: handler)

        handler.inFlightRequests = 1
        handler.handleAsyncFailure(
            CancellationError(), context: handlerContext, id: "req", phase: "generate",
            stream: false, outbox: nil, streamState: StreamState(), surface: .chat)
        channel.embeddedEventLoop.run()

        #expect(
            handler.inFlightRequests == 0,
            "a cancelled request that wrote nothing kept its in-flight count")
        _ = try? channel.finish()
    }
}
