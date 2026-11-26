import Foundation

/// An append-only file of JSON lines.
///
/// The file holds complete user prompts and model replies. It is created
/// with owner-only permissions and never leaves the machine: nothing in this
/// package opens a socket, and no caller should hand this file to one without
/// the user deciding to.
///
/// Exactly one process may write a given journal. That is enforced with an
/// exclusive advisory lock on a sidecar `.lock` file, taken for the life of
/// this object. Without it two servers launched from the same directory would
/// each hold their own copy of the state in memory, see none of the other's
/// writes, and interleave their appends into one file that replays as a
/// braid of two divergent histories. The lock is on a sidecar rather than on
/// the journal itself so that compaction can replace the journal without ever
/// letting go of it.
///
/// The lock is released by the kernel when the process exits, however it
/// exits, so a crash never leaves a journal that cannot be reopened.
public actor FileJournal: ContinuityJournal {
    public let url: URL
    let lockURL: URL
    /// Every blocking syscall runs through this, so a failure the drive will
    /// not produce on request can be produced in a test. Production is
    /// `SystemJournalFileSystem`.
    let fileSystem: any JournalFileSystem
    var lockDescriptor: Int32 = -1
    var descriptor: Int32 = -1
    let encoder: JSONEncoder
    let decoder: JSONDecoder
    private let synchronizesEveryWrite: Bool
    /// Identifies the descriptor, not its number. `compact` closes one
    /// descriptor and opens another, and the kernel may hand back the same
    /// number; a barrier that resumed holding only the number would then
    /// settle a watermark against a file that descriptor never wrote.
    var descriptorGeneration: UInt64 = 0
    /// Records accepted by `append`, and records a completed barrier covers.
    /// The difference is what `pendingRecords` reports, and it is the reason a
    /// barrier settles the watermark it captured rather than zeroing the
    /// count: an append admitted while the barrier was in flight was not
    /// covered by it, even if the drive happened to flush it too.
    var appendedRecords: UInt64 = 0
    var durableRecords: UInt64 = 0
    /// The last barrier that failed, or nil once one succeeds again. This is
    /// the scheduled barrier's only way to tell anyone: it has no caller.
    public private(set) var durabilityFailure: JournalError?
    /// Complete lines the last `replay` could not decode, or nil when every
    /// complete line was a record. A final unterminated line is not damage; it
    /// is a torn write that never reported success.
    public internal(set) var replayDamage: JournalReplayDamage?
    /// Delay after the last append before the durability barrier is taken.
    private let idleDelay: Duration
    /// Longest a record may wait for a barrier while appends keep arriving.
    private let maximumLatency: Duration
    private var barrier: Task<Void, Never>?
    var lastBarrier = ContinuousClock.now
    /// A compaction or truncation owns the descriptor while it replaces the
    /// file, and a shutdown owns it while it closes it. Appends wait for that
    /// to finish rather than writing into an inode nobody will read again.
    private var rewriting = false
    private var rewriteWaiters: [CheckedContinuation<Void, Never>] = []
    /// Every blocking syscall this type makes runs here, not on the
    /// cooperative pool. A barrier is tens of milliseconds of the drive
    /// doing nothing else; on a Mac with a two-thread pool that would be
    /// half of every actor in the process stalled behind a memory write.
    static let blockingQueue = DispatchQueue(
        label: "ContinuityCore.journal",
        qos: .utility)

    /// - Parameters:
    ///   - synchronizesEveryWrite: take the barrier on every append, inline.
    ///     Correct across a power cut, and slow enough that it is off by
    ///     default; a crash of the process alone loses nothing either way,
    ///     because the write has already reached the kernel.
    ///   - idleDelay: how long after the last append the barrier is taken.
    ///     The point of waiting is that an append happens while a model is
    ///     answering, and the moment after the answer is the one moment the
    ///     drive is not being asked for expert weights.
    ///   - maximumLatency: the longest a record waits for a barrier while
    ///     appends keep arriving, so a busy tool loop cannot postpone
    ///     durability indefinitely.
    /// - Throws: `JournalError.locked` when another process holds this
    ///   journal. Callers should treat that as "run without persistence and
    ///   say so", never as a reason to write anyway.
    public init(
        url: URL,
        synchronizesEveryWrite: Bool = false,
        idleDelay: Duration = .seconds(2),
        maximumLatency: Duration = .seconds(30)
    ) throws {
        try self.init(
            url: url,
            fileSystem: SystemJournalFileSystem(),
            synchronizesEveryWrite: synchronizesEveryWrite,
            idleDelay: idleDelay,
            maximumLatency: maximumLatency)
    }

    /// The seam, internal so a test can hand in a failing filesystem. The
    /// public initializer above is the only one production code uses.
    init(
        url: URL,
        fileSystem: any JournalFileSystem,
        synchronizesEveryWrite: Bool = false,
        idleDelay: Duration = .seconds(2),
        maximumLatency: Duration = .seconds(30)
    ) throws {
        self.url = url
        self.lockURL = url.appendingPathExtension("lock")
        self.fileSystem = fileSystem
        self.synchronizesEveryWrite = synchronizesEveryWrite
        self.idleDelay = idleDelay
        self.maximumLatency = maximumLatency
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.withoutEscapingSlashes]
        self.encoder = encoder
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        self.decoder = decoder

        try Self.prepareDirectory(for: url)
        self.lockDescriptor = try Self.acquireLock(at: lockURL, journal: url, using: fileSystem)
        do {
            self.descriptor = try Self.openForAppend(url, using: fileSystem)
            self.descriptorGeneration = 1
            try Self.dropDanglingTail(
                descriptor: self.descriptor, url: url, using: fileSystem)
        } catch {
            // A thrown initializer never runs `deinit`, so both descriptors
            // this call opened have to go back here or the workspace leaks an
            // open file and a lock the process no longer owns.
            if descriptor >= 0 { fileSystem.close(descriptor) }
            descriptor = -1
            fileSystem.close(lockDescriptor)
            lockDescriptor = -1
            throw error
        }
    }

    deinit {
        if descriptor >= 0 { fileSystem.close(descriptor) }
        // Closing releases the flock. Doing it explicitly rather than relying
        // on process exit means a journal dropped mid-run frees its workspace
        // for another server immediately.
        if lockDescriptor >= 0 { fileSystem.close(lockDescriptor) }
    }

    // MARK: - Writing

    public func append(_ record: JournalRecord) async throws {
        var data = try encoder.encode(record)
        data.append(0x0A)
        await awaitRewrite()
        if let error = writeFully(data) { throw error }
        appendedRecords &+= 1
        try await afterAppend()
    }

    public func append(_ records: [JournalRecord]) async throws {
        guard !records.isEmpty else { return }
        // One write for the batch: a partial batch on disk is a torn tail the
        // replay would drop anyway, and one syscall is cheaper than many.
        var data = Data()
        for record in records {
            data.append(try encoder.encode(record))
            data.append(0x0A)
        }
        await awaitRewrite()
        if let error = writeFully(data) { throw error }
        appendedRecords &+= UInt64(records.count)
        try await afterAppend()
    }

    /// Decides when the barrier happens. Never inline unless asked for.
    ///
    /// `write(2)` to the page cache is microseconds and is all the request
    /// path ever pays. The barrier is scheduled for when appends stop, and
    /// forced only when a record has been waiting longer than the maximum.
    private func afterAppend() async throws {
        if synchronizesEveryWrite {
            if let error = await performBarrier() { throw error }
            return
        }
        let overdue = ContinuousClock.now - lastBarrier > maximumLatency
        barrier?.cancel()
        let delay = overdue ? Duration.zero : idleDelay
        barrier = Task { [weak self] in
            if delay > .zero {
                guard await Self.sleepUnlessCancelled(delay) else { return }
            }
            guard !Task.isCancelled, let self else { return }
            // The failure is recorded on the actor by `performBarrier`. A
            // scheduled barrier has no caller to throw to, which is why the
            // old `try?` here left a store reporting itself durable after its
            // drive had stopped accepting writes.
            _ = await self.performBarrier()
        }
    }

    /// True unless the task was cancelled first. `Task.sleep` has exactly one
    /// error, cancellation, and a cancelled barrier is replaced by the append
    /// that cancelled it — not a storage failure.
    private static func sleepUnlessCancelled(_ delay: Duration) async -> Bool {
        do {
            try await Task.sleep(for: delay)
            return true
        } catch {
            return false
        }
    }

    /// Records written but not yet behind a barrier. For diagnostics.
    public var pendingRecords: Int {
        let pending = appendedRecords &- durableRecords
        return pending > UInt64(Int.max) ? Int.max : Int(pending)
    }

    /// Writes every byte, or says why not. A short write is normal for
    /// `write(2)` on a large buffer and silently dropping the remainder would
    /// corrupt the record that followed it; a zero write with bytes remaining
    /// would loop forever, so it is reported instead of retried.
    func writeFully(_ data: Data) -> JournalError? {
        guard descriptor >= 0 else { return .writeFailed(url, errno: EBADF) }
        return Self.writeFully(data, to: descriptor, url: url, using: fileSystem)
    }

    /// The same loop for a descriptor this actor does not own — the checkpoint
    /// temp is written through one.
    static func writeFully(
        _ data: Data, to descriptor: Int32, url: URL,
        using fileSystem: any JournalFileSystem
    ) -> JournalError? {
        data.withUnsafeBytes { buffer -> JournalError? in
            // An empty write has no bytes to write and nothing to fail.
            guard !buffer.isEmpty else { return nil }
            var offset = 0
            while offset < buffer.count {
                let written = fileSystem.write(
                    descriptor, bytes: UnsafeRawBufferPointer(rebasing: buffer[offset...]))
                if written == 0 {
                    // The only write that may advance nothing is the one with
                    // nothing to write, and an empty buffer never enters this
                    // loop. Continuing would spin at full CPU until something
                    // else kills the process, with the record half-written.
                    return .writeFailed(url, errno: EIO)
                }
                if written < 0 {
                    if written == -EINTR { continue }
                    return .writeFailed(url, errno: Int32(-written))
                }
                if written > buffer.count - offset {
                    return .writeFailed(url, errno: EIO)
                }
                offset += written
            }
            return nil
        }
    }

    /// Force everything written so far to the platter, now.
    ///
    /// `F_FULLFSYNC` rather than `fsync` because on Darwin `fsync` only
    /// promises the write reached the drive's cache, which a power cut can
    /// still lose. The stronger barrier is the point of calling this at all.
    public func sync() async throws {
        barrier?.cancel()
        barrier = nil
        if let error = await performBarrier() { throw error }
    }

    /// The barrier itself, on the blocking queue. The actor suspends until
    /// it is done, so appends queue behind it in order, but no pool thread is
    /// held while the drive works.
    ///
    /// Answers the failure rather than throwing it: the scheduled barrier has
    /// no caller to receive a throw, and it must record the same failure the
    /// explicit `sync` reports. Both read the actor-owned
    /// `durabilityFailure`.
    private func performBarrier() async -> JournalError? {
        guard descriptor >= 0, appendedRecords != durableRecords else { return nil }
        let target = descriptor
        let generation = descriptorGeneration
        // The watermark belongs to the moment the barrier is dispatched. A
        // record appended while the drive works is not covered by this
        // barrier, whatever the kernel does with it.
        let watermark = appendedRecords
        let fileSystem = self.fileSystem
        let code: Int32 = await withCheckedContinuation { continuation in
            Self.blockingQueue.async {
                var result = fileSystem.fullSync(target)
                if result != 0 {
                    // Not every filesystem implements it; fall back rather
                    // than fail, but only to a fallback that is checked.
                    result = fileSystem.fsync(target)
                }
                continuation.resume(returning: result)
            }
        }
        // A compaction may have swapped the descriptor while the barrier was
        // in flight. The old file is not the journal any more, and the new
        // one has its own accounting.
        guard generation == descriptorGeneration else { return nil }
        if code != 0 {
            let error = JournalError.writeFailed(url, errno: -code)
            durabilityFailure = error
            return error
        }
        if watermark > durableRecords { durableRecords = watermark }
        durabilityFailure = nil
        lastBarrier = ContinuousClock.now
        return nil
    }

    // MARK: - Descriptor ownership

    /// Wait until no compaction or truncation is replacing the file.
    func awaitRewrite() async {
        while rewriting {
            await withCheckedContinuation { rewriteWaiters.append($0) }
        }
    }

    /// Take exclusive ownership of the descriptor. The caller must release it
    /// on every path, including a throw.
    func acquireRewrite() async {
        await awaitRewrite()
        rewriting = true
    }

    func releaseRewrite() {
        rewriting = false
        let waiters = rewriteWaiters
        rewriteWaiters.removeAll()
        for waiter in waiters { waiter.resume() }
    }

    /// Cancels a scheduled barrier and waits for one already running, so a
    /// descriptor is never closed or replaced underneath the drive.
    func settleBarrier() async {
        barrier?.cancel()
        await awaitScheduledBarrier()
        barrier = nil
    }

    /// Waits for the barrier that is currently scheduled, without cancelling
    /// it.
    ///
    /// `settleBarrier` cancels first, which is right for a caller about to
    /// replace the file and wrong for a test or a diagnostic that wants to
    /// observe what the scheduled barrier actually did. Awaiting the task is
    /// the only deterministic way to see that: the task records its outcome on
    /// this actor before it completes.
    func awaitScheduledBarrier() async {
        if let running = barrier { await running.value }
    }

    /// Flush, close the journal and release the workspace lock.
    ///
    /// Named `shutDown` rather than `close` so it cannot be confused with
    /// `close(2)`, which this type calls throughout.
    ///
    /// A flush that fails still closes the descriptors and releases the lock —
    /// a caller that cannot shut down cleanly must still be able to shut down
    /// once — and the failure is thrown afterwards, so a storage failure is
    /// never reported as a clean shutdown.
    public func shutDown() async throws {
        await acquireRewrite()
        defer { releaseRewrite() }
        await settleBarrier()
        let failure = await performBarrier()
        if descriptor >= 0 { fileSystem.close(descriptor) }
        descriptor = -1
        descriptorGeneration &+= 1
        if lockDescriptor >= 0 { fileSystem.close(lockDescriptor) }
        lockDescriptor = -1
        if let failure { throw failure }
    }
}
