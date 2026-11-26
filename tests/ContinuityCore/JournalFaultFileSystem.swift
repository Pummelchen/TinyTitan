import Foundation
import Synchronization

@testable import ContinuityCore

/// A `JournalFileSystem` that forwards to the real syscalls and can be told to
/// fail one chosen call, or to perform only part of one.
///
/// The point is that a journal guarantee is about a fault — a barrier that
/// returns an error, a write that advances nothing, a `pread` that answers
/// short — and every one of those is a state a real drive refuses to enter on
/// request. The production implementation stays the only one the package
/// ships; this is handed to the internal initializer from a test only.
///
/// State lives behind `Mutex`, not an `NSLock` in an `@unchecked Sendable`
/// class: the type has to be `Sendable` because the blocking queue captures it,
/// and the compiler can check a `Mutex`.
final class JournalFaultFileSystem: JournalFileSystem, Sendable {
    enum WriteOutcome: Sendable, Equatable {
        case all
        /// Write this many bytes at most, then report it. A short write.
        case bytes(Int)
        /// Report zero bytes written and write nothing.
        case zero
        case interrupted
        case failure(Int32)
    }

    enum ReadOutcome: Sendable, Equatable {
        case all
        /// Read this many bytes at most.
        case bytes(Int)
        case zero
        case interrupted
        case failure(Int32)
    }

    struct State {
        var fullSyncFailure: Int32?
        var fullSyncFailurePaths: [String: Int32] = [:]
        var fsyncFailure: Int32?
        var fsyncFailurePaths: [String: Int32] = [:]
        var openFailure: [String: Int32] = [:]
        /// Fail this path once a rename has happened; the checkpoint's reopen.
        var reopenFailurePath: String?
        var reopenFailureCode: Int32 = 0
        var renameFailure: Int32?
        var unlinkFailure: Int32?
        var writePlan: [WriteOutcome] = []
        var readPlan: [ReadOutcome] = []
        var fstatModeOverride: mode_t?
        var fstatUIDOverride: uid_t?
        var opens = 0
        var writes = 0
        var reads = 0
        var fullSyncs = 0
        var fsyncs = 0
        var renames = 0
        var unlinks = 0
        var descriptorPaths: [Int32: String] = [:]
        /// The checkpoint pause: signalled when `rename` is entered, released
        /// by the test. A semaphore, not a sleep, so the ordering under test is
        /// the ordering that happens.
        var renameGate: (entered: DispatchSemaphore, release: DispatchSemaphore)?
        /// The barrier pause: signalled when `fullSync` is entered and released
        /// by the test. Consumed by the first call so a later barrier (a
        /// compaction's data barrier, say) is not blocked by the same hold.
        var fullSyncGate: (entered: DispatchSemaphore, release: DispatchSemaphore)?
    }

    let state: Mutex<State>
    private let real = SystemJournalFileSystem()

    init() {
        self.state = Mutex(State())
    }

    // MARK: - Configuration

    func configure(_ change: (inout sending State) -> sending Void) {
        state.withLock(change)
    }

    func failFullSync(_ code: Int32?) {
        state.withLock { $0.fullSyncFailure = code.map { -$0 } }
    }

    func failFullSync(at path: String, _ code: Int32) {
        state.withLock { $0.fullSyncFailurePaths[path] = -code }
    }

    func failFsync(_ code: Int32?) {
        state.withLock { $0.fsyncFailure = code.map { -$0 } }
    }

    func failFsync(at path: String, _ code: Int32) {
        state.withLock { $0.fsyncFailurePaths[path] = -code }
    }

    func failOpen(at path: String, _ code: Int32) {
        state.withLock { $0.openFailure[path] = -code }
    }

    func failReopen(at path: String, _ code: Int32) {
        state.withLock {
            $0.reopenFailurePath = path
            $0.reopenFailureCode = code
        }
    }

    func failRename(_ code: Int32?) {
        state.withLock { $0.renameFailure = code.map { -$0 } }
    }

    func planWrites(_ outcomes: [WriteOutcome]) {
        state.withLock { $0.writePlan = outcomes }
    }

    func planReads(_ outcomes: [ReadOutcome]) {
        state.withLock { $0.readPlan = outcomes }
    }

    func overrideMode(_ mode: mode_t?) {
        state.withLock { $0.fstatModeOverride = mode }
    }

    func overrideOwner(_ uid: uid_t?) {
        state.withLock { $0.fstatUIDOverride = uid }
    }

    func holdRename(entered: DispatchSemaphore, release: DispatchSemaphore) {
        state.withLock {
            $0.renameGate = (entered: entered, release: release)
        }
    }

    func holdFullSync(entered: DispatchSemaphore, release: DispatchSemaphore) {
        state.withLock {
            $0.fullSyncGate = (entered: entered, release: release)
        }
    }

    var writes: Int { state.withLock { $0.writes } }
    var reads: Int { state.withLock { $0.reads } }
    var renames: Int { state.withLock { $0.renames } }
    var fullSyncs: Int { state.withLock { $0.fullSyncs } }
    var fsyncs: Int { state.withLock { $0.fsyncs } }

    // MARK: - JournalFileSystem

    func open(path: String, flags: Int32, mode: mode_t) -> Int32 {
        let failure = state.withLock { state -> Int32? in
            state.opens += 1
            if let code = state.openFailure[path] { return code }
            if state.renames > 0, state.reopenFailurePath == path {
                return -state.reopenFailureCode
            }
            return nil
        }
        if let failure { return failure }
        let descriptor = real.open(path: path, flags: flags, mode: mode)
        if descriptor >= 0 {
            state.withLock { $0.descriptorPaths[descriptor] = path }
        }
        return descriptor
    }

    func close(_ descriptor: Int32) {
        state.withLock { _ = $0.descriptorPaths.removeValue(forKey: descriptor) }
        real.close(descriptor)
    }

    func write(_ descriptor: Int32, bytes: UnsafeRawBufferPointer) -> Int {
        let outcome = state.withLock { state -> WriteOutcome in
            state.writes += 1
            return state.writePlan.isEmpty ? .all : state.writePlan.removeFirst()
        }
        switch outcome {
        case .all:
            return real.write(descriptor, bytes: bytes)
        case .bytes(let count):
            let limit = min(count, bytes.count)
            return real.write(
                descriptor, bytes: UnsafeRawBufferPointer(rebasing: bytes[0..<limit]))
        case .zero:
            return 0
        case .interrupted:
            return Int(-EINTR)
        case .failure(let code):
            return Int(-code)
        }
    }

    func read(_ descriptor: Int32, into buffer: UnsafeMutableRawBufferPointer) -> Int {
        real.read(descriptor, into: buffer)
    }

    func pread(
        _ descriptor: Int32, into buffer: UnsafeMutableRawBufferPointer, offset: off_t
    ) -> Int {
        let outcome = state.withLock { state -> ReadOutcome in
            state.reads += 1
            return state.readPlan.isEmpty ? .all : state.readPlan.removeFirst()
        }
        switch outcome {
        case .all:
            return real.pread(descriptor, into: buffer, offset: offset)
        case .bytes(let count):
            let limit = min(count, buffer.count)
            return real.pread(
                descriptor, into: UnsafeMutableRawBufferPointer(rebasing: buffer[0..<limit]),
                offset: offset)
        case .zero:
            return 0
        case .interrupted:
            return Int(-EINTR)
        case .failure(let code):
            return Int(-code)
        }
    }

    func fsync(_ descriptor: Int32) -> Int32 {
        let failure = state.withLock { state -> Int32? in
            state.fsyncs += 1
            if let path = state.descriptorPaths[descriptor],
                let code = state.fsyncFailurePaths[path]
            {
                return code
            }
            return state.fsyncFailure
        }
        if let failure { return failure }
        return real.fsync(descriptor)
    }

    func fullSync(_ descriptor: Int32) -> Int32 {
        let decision = state.withLock {
            state -> (failure: Int32?, gate: State.PendingGate?) in
            state.fullSyncs += 1
            let gate = state.fullSyncGate.map(State.PendingGate.init)
            // One hold: a later barrier in the same test is not blocked by it.
            state.fullSyncGate = nil
            if let path = state.descriptorPaths[descriptor],
                let code = state.fullSyncFailurePaths[path]
            {
                return (code, gate)
            }
            return (state.fullSyncFailure, gate)
        }
        if let gate = decision.gate {
            gate.entered.signal()
            gate.release.wait()
        }
        if let failure = decision.failure { return failure }
        return real.fullSync(descriptor)
    }

    func rename(from: String, to: String) -> Int32 {
        let decision = state.withLock { state -> (failure: Int32?, gate: State.PendingGate?) in
            state.renames += 1
            return (state.renameFailure, state.renameGate.map(State.PendingGate.init))
        }
        if let gate = decision.gate {
            gate.entered.signal()
            gate.release.wait()
        }
        if let failure = decision.failure { return failure }
        return real.rename(from: from, to: to)
    }

    func unlink(_ path: String) -> Int32 {
        var failure: Int32?
        state.withLock { state in
            state.unlinks += 1
            failure = state.unlinkFailure
        }
        // `configure` stores the positive errno an operator would name.
        if let failure { return -failure }
        return real.unlink(path)
    }

    func ftruncate(_ descriptor: Int32, offset: off_t) -> Int32 {
        real.ftruncate(descriptor, offset: offset)
    }

    func lock(_ descriptor: Int32, operation: Int32) -> Int32 {
        real.lock(descriptor, operation: operation)
    }

    func fstat(_ descriptor: Int32) -> (status: stat, error: Int32) {
        var (status, error) = real.fstat(descriptor)
        guard error == 0 else { return (status, error) }
        let override = state.withLock { ($0.fstatModeOverride, $0.fstatUIDOverride) }
        if let mode = override.0 { status.st_mode = mode }
        if let uid = override.1 { status.st_uid = uid }
        return (status, 0)
    }
}

/// Wait for a signal without blocking a cooperative thread.
///
/// The wait runs on a global queue; the timeout only guards against a hang in
/// the code under test, it is not the synchronisation.
func waitForSignal(
    _ semaphore: DispatchSemaphore, timeout: DispatchTime = .now() + 10
) async -> Bool {
    await withCheckedContinuation { continuation in
        DispatchQueue.global().async {
            continuation.resume(returning: semaphore.wait(timeout: timeout) == .success)
        }
    }
}

extension JournalFaultFileSystem.State {
    /// The semaphore pair, copied out so `rename` does not hold the lock while
    /// it waits.
    struct PendingGate {
        let entered: DispatchSemaphore
        let release: DispatchSemaphore

        init(_ gate: (entered: DispatchSemaphore, release: DispatchSemaphore)) {
            self.entered = gate.entered
            self.release = gate.release
        }
    }
}
