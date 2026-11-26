import Foundation
import NIOCore
import NIOHTTP1
import NIOPosix
import Synchronization
import TinyTitanCore
import TinyTitanLib

public actor TinyTitanHTTPServer {
    public static let maximumBodyBytes = 1_048_576

    /// S1: close a connection that has been idle (no request in flight) for
    /// this long. Also used as the pipeline read timeout, bounding slowloris
    /// style partial-request stalls.
    public static let idleReadTimeout: TimeAmount = .seconds(120)

    /// S1: reject connections beyond this cap to bound FD/memory usage.
    public static let maximumConcurrentConnections = 64

    /// Ceiling on the *aggregate* request header block, checked when the head
    /// arrives.
    ///
    /// SwiftNIO 2.100's decoder does have an aggregate bound
    /// (`NIOHTTPDecoderLimitConfiguration.maxHeaderListSize`), and
    /// `decoderLimitConfiguration` below wires it; this application check is
    /// the tighter, documented policy and is what answers 431 for a head the
    /// decoder admitted. The decoder's bound is the outer backstop that stops
    /// buffering for a head that never completes. Before the decoder limit was
    /// wired, a client could send an unbounded number of small headers and the
    /// decoder accumulated every one before this handler saw a head at all.
    public static let maximumRequestHeaderBytes = 16 * 1024

    /// Ceiling on the number of header fields, for the same reason. A field
    /// count is what a hostile client can inflate without inflating bytes.
    public static let maximumRequestHeaderFields = 128

    /// The pre-decode hard bound on a request's header list (field names and
    /// values, including the URI and trailers as NIO counts them).
    ///
    /// Twice the application byte budget so a head the application would
    /// accept can never be refused by the decoder for byte count alone, while
    /// a stream that never terminates its head is still bounded to a few tens
    /// of KiB per connection. A request over this bound gets NIO's protocol
    /// error 400 before any `HTTPRequestHead` is emitted; a completed head
    /// between this bound and the 16 KiB policy gets the documented 431.
    public static let maximumDecoderHeaderListBytes = 32 * 1024

    /// Decoder limits for the server pipeline: one field may not exceed the
    /// application's whole byte budget, the list is bounded by
    /// `maximumDecoderHeaderListBytes`, and the field count stays at NIO's
    /// bound of twice the application policy. Keeping the decoder count above
    /// 128 is deliberate: a 129-field head is answered by the application's
    /// `request_headers_too_large` (431), and the decoder only catches a stream
    /// that is still inflating fields without completing the head.
    static var decoderLimitConfiguration: NIOHTTPDecoderLimitConfiguration {
        var limits = NIOHTTPDecoderLimitConfiguration()
        limits.maxHeaderFieldSize = maximumRequestHeaderBytes
        limits.maxHeaderListSize = maximumDecoderHeaderListBytes
        limits.maxHeaderFieldCount = maximumRequestHeaderFields * 2
        return limits
    }

    /// How much of a refused request's body is read and discarded before the
    /// connection is closed.
    ///
    /// Answering and closing immediately is the tempting version and it is
    /// wrong: a client mid-upload often sees a connection reset instead of the
    /// response it was just sent, which turns "your body is too large" into "the
    /// connection broke". Draining a bounded remainder lets the client finish
    /// writing, read the refusal, and shut down in order — while still bounding
    /// what a multi-gigabyte body can make the server consume to this much.
    public static let maximumDrainedBytesAfterReject = 4 * 1024 * 1024

    private let group: MultiThreadedEventLoopGroup
    private let modelID: String
    private let backend: any ServerInferenceBackend
    private let coordinator: ServerCoordinator
    private let heartbeatInterval: TimeAmount
    private let reasoningProfile: ServerReasoningProfile
    /// Set when the server routes between catalog models; nil keeps the
    /// single-model server exactly as it was.
    private let router: (any ModelRouting)?
    private let childChannels = ChildChannelRegistry(
        maximumChannels: maximumConcurrentConnections)
    /// Finished /v1/responses kept for previous_response_id and retrieval.
    private let responseStore: ResponseStore
    private var channel: Channel?
    private var shutdownTask: Task<Void, any Error>?

    package init(
        modelID: String,
        queueLimit: Int,
        maxConcurrentSequences: Int = 1,
        backend: any ServerInferenceBackend,
        heartbeatInterval: TimeAmount = .seconds(5),
        reasoningProfile: ServerReasoningProfile = .default,
        group: MultiThreadedEventLoopGroup = .init(numberOfThreads: 1),
        router: (any ModelRouting)? = nil,
        coordinator: ServerCoordinator? = nil,
        responseStore: ResponseStore? = nil
    ) {
        self.group = group
        self.modelID = modelID
        self.backend = backend
        // Injectable so a resident side-engine can read the same
        // `generating` signal the coordinator raises for every client
        // generation, which is how it chooses its width.
        self.coordinator =
            coordinator
            ?? ServerCoordinator(
                queueLimit: queueLimit,
                width: maxConcurrentSequences)
        self.heartbeatInterval = heartbeatInterval
        self.reasoningProfile = reasoningProfile
        self.router = router
        self.responseStore = responseStore ?? ResponseStore()
    }

    public func start(port: Int) async throws -> Channel {
        let modelID = self.modelID
        let backend = self.backend
        let coordinator = self.coordinator
        let heartbeatInterval = self.heartbeatInterval
        let reasoningProfile = self.reasoningProfile
        let router = self.router
        let childChannels = self.childChannels
        let responseStore = self.responseStore
        let bootstrap = ServerBootstrap(group: group)
            .serverChannelOption(ChannelOptions.backlog, value: 16)
            .serverChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
            .childChannelInitializer { channel in
                childChannels.insert(channel)
                return channel.pipeline.configureHTTPServerPipeline(
                    withPipeliningAssistance: true,
                    withErrorHandling: true,
                    withDecoderLimitConfiguration: Self.decoderLimitConfiguration
                ).flatMap {
                    channel.pipeline.addHandler(
                        ServerHTTPHandler(
                            modelID: modelID,
                            backend: backend,
                            coordinator: coordinator,
                            heartbeatInterval: heartbeatInterval,
                            reasoningProfile: reasoningProfile,
                            router: router,
                            childChannels: childChannels,
                            responseStore: responseStore))
                }
            }
        // S29: so_reuseaddr belongs on the listening socket only, not on
        // accepted sockets.
        let channel = try await bootstrap.bind(host: "127.0.0.1", port: port).get()
        self.channel = channel
        return channel
    }

    public func shutdown() async throws {
        if let shutdownTask {
            try await shutdownTask.value
            return
        }

        let listeningChannel = channel
        channel = nil
        let childChannels = self.childChannels
        let coordinator = self.coordinator
        let group = self.group
        let task = Task { @Sendable in
            var firstError: (any Error)?
            await coordinator.shutdown()
            if let listeningChannel {
                do {
                    try await listeningChannel.close().get()
                } catch ChannelError.alreadyClosed {
                } catch {
                    firstError = error
                }
            }
            await childChannels.closeAll()
            do {
                try await group.shutdownGracefully()
            } catch {
                if firstError == nil {
                    firstError = error
                }
            }
            if let firstError {
                throw firstError
            }
        }
        shutdownTask = task
        try await task.value
    }

    var queuedRequestCount: Int {
        get async { await coordinator.queuedCount }
    }

    var hasActiveRequest: Bool {
        get async { await coordinator.isActive }
    }

    var acceptedConnectionCount: Int {
        childChannels.count
    }
}

/// The workspace a request names, from `X-TinyTitan-Workspace`.
///
/// This is how one server serves several checkouts, and it was documented in
/// three places and read in none: `ValidatedChatRequest.workspace` had no
/// caller at all, so two projects sharing a server silently shared a memory
/// store — the exact cross-project mixing the workspace design exists to
/// prevent.
///
/// Only the shape is checked here. Whether the name is *allowed* — the
/// reserved shared workspace, characters a scope forbids — belongs to the
/// memory layer, which already refuses those and disables memory for the
/// request rather than failing it. A bad workspace must never cost someone
/// their answer.
enum WorkspaceHeader {
    static let name = "x-tinytitan-workspace"

    /// The header's value, or the reason it cannot be used.
    ///
    /// A duplicate is not a merge and not a first-wins pick: the workspace
    /// selects which project's persistent memory a request reads and writes, so
    /// choosing one by wire order would let a proxy or a retry append a second
    /// value and silently move the request to another project. Admission refuses
    /// it before routing.
    enum Value: Equatable {
        case absent
        case named(String)
        case duplicate

        var value: String? {
            if case .named(let value) = self { return value }
            return nil
        }
    }

    static func resolve(in head: HTTPRequestHead?) -> Value {
        guard let head else { return .absent }
        let values = head.headers[name]
        guard !values.isEmpty else { return .absent }
        guard values.count == 1 else { return .duplicate }
        let trimmed = values[0].trimmingCharacters(in: .whitespaces)
        // A proxy that adds the header with nothing after it must not create
        // a workspace called "".
        return trimmed.isEmpty ? .absent : .named(trimmed)
    }

    /// The named workspace, or nil when absent or duplicated. Routed requests
    /// have already refused `.duplicate`; this spelling stays for callers that
    /// only want the value.
    static func value(in head: HTTPRequestHead?) -> String? {
        resolve(in: head).value
    }
}
