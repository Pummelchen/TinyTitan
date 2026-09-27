// The admission queue and the per-runner counters the server infers from.
//
// Split out of ServerInference.swift (2026-09-27) under the 500-line-per-file
// rule; declarations and their doc comments are unchanged, except that
// `private` became internal where a declaration now lives beside an extension
// in another file.
import CryptoKit
import Foundation
import TinyTitan

public actor ServerCoordinator {
    struct Waiter {
        let id: UUID
        let continuation: CheckedContinuation<Void, Error>
    }

    let queueLimit: Int
    /// How many generations may run at once. One is the historical
    /// single-generation server; more lets the batched slots through while the
    /// excess still queues. The engine's `ForwardStepGate` keeps their forward
    /// passes from interleaving.
    let width: Int
    var admittedCount = 0
    var activeCount = 0
    var waiters: [Waiter] = []
    var shuttingDown = false
    /// Raised for the duration of every client generation. The side-engine
    /// reads it to choose its width: one thread while a person is waiting,
    /// four in the gaps.
    public nonisolated let generating = GenerationSignal()

    public init(queueLimit: Int, width: Int = 1) {
        self.queueLimit = queueLimit
        self.width = max(1, width)
    }

    public func run<T: Sendable>(
        onQueued: @escaping @Sendable () -> Void = {},
        _ operation: @escaping @Sendable () async throws -> T
    ) async throws -> T {
        try await runPreparing(
            onQueued: onQueued,
            prepare: { () },
            operation: { _ in try await operation() })
    }

    func runPreparing<Prepared: Sendable, T: Sendable>(
        onQueued: @escaping @Sendable () -> Void = {},
        prepare: @escaping @Sendable () async throws -> Prepared,
        operation: @escaping @Sendable (Prepared) async throws -> T
    ) async throws -> T {
        try Task.checkCancellation()
        guard !shuttingDown else { throw CancellationError() }
        // S6: at most `width` running and `queueLimit` queued behind them, so
        // `width + queueLimit` admitted. Width 1 reproduces the original
        // single-generation bound exactly.
        guard admittedCount < width + queueLimit else {
            // Shed load rather than queue without bound.
            throw ServerRequestError.queueFull
        }
        admittedCount += 1
        defer { admittedCount -= 1 }

        let prepared = try await prepare()
        try Task.checkCancellation()
        try await acquire(onQueued: onQueued)
        defer { release() }
        generating.enter()
        defer { generating.leave() }
        return try await operation(prepared)
    }

    func acquire(onQueued: @escaping @Sendable () -> Void) async throws {
        try Task.checkCancellation()
        guard !shuttingDown else { throw CancellationError() }
        if activeCount < width {
            activeCount += 1
            return
        }
        guard waiters.count < queueLimit else { throw ServerRequestError.queueFull }
        onQueued()
        let id = UUID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                waiters.append(Waiter(id: id, continuation: continuation))
            }
        } onCancel: {
            Task { await self.cancelWaiter(id) }
        }
        if Task.isCancelled {
            release()
            throw CancellationError()
        }
    }

    func cancelWaiter(_ id: UUID) {
        guard let index = waiters.firstIndex(where: { $0.id == id }) else { return }
        let waiter = waiters.remove(at: index)
        waiter.continuation.resume(throwing: CancellationError())
    }

    func release() {
        if waiters.isEmpty {
            activeCount = max(0, activeCount - 1)
        } else {
            waiters.removeFirst().continuation.resume()
        }
    }

    public func shutdown() {
        shuttingDown = true
        let queued = waiters
        waiters.removeAll()
        for waiter in queued {
            waiter.continuation.resume(throwing: CancellationError())
        }
    }

    public var queuedCount: Int { waiters.count }
    public var isActive: Bool { activeCount > 0 }
    /// Running generations, for tests and the readiness view.
    public var runningCount: Int { activeCount }
    /// The admission width this coordinator was built with.
    public var concurrencyWidth: Int { width }
}

/// Snapshot of the runner's lifetime stage counters at request start, so the
/// TINYTITAN_RUNNER_STATS footer can report this request's per-stage deltas.
struct RunnerCounterSnapshot {
    let cb1: UInt64
    let io: UInt64
    let cb2: UInt64
    let head: UInt64
    let headFused: UInt64
    let rdadvise: UInt64
    let rdadviseCalls: UInt64
    let rdadviseBytes: UInt64
    let wait: UInt64
    let body: UInt64
    let prefetchIssued: UInt64
    let prefetchAdopted: UInt64
    let preamble: UInt64
    let preambleRelease: UInt64
    let preamblePin: UInt64
    let preambleReserve: UInt64
    let embed: UInt64
    let gather: UInt64
    let loopSample: UInt64
    let loopProgress: UInt64
    let loopOther: UInt64
    let missIo: UInt64
    let exposedIo: UInt64
    let hitFixupLayers: UInt64
    let routerReadback: UInt64
    let cachePlan: UInt64
    let ioQueue: UInt64
    let ioCompletionToFixup: UInt64
    let ioHostWaits: UInt64
    let ioHostWaitsAvoided: UInt64
    let gpuClassifiedHits: UInt64
    let gpuClassifiedMisses: UInt64
    let gpuAllHitLayers: UInt64
    let expertStreaming: ExpertStreamingStatistics
}
