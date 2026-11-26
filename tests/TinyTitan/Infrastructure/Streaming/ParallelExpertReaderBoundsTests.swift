import Foundation
import Testing

@testable import TinyTitan
import TinyTitanKernelsC

/// The C reader's offset contract and its lifecycle, checked without a model.
///
/// The streamer validates offsets before it calls the reader, but the reader is
/// also a C entry point and the Swift wrapper takes raw `[UInt64]`, so the
/// refusal has to live in C: an offset above `off_t`'s range or a wrapped
/// `id * stride` used to reach `pread` as a negative or truncated position and
/// come back `EINVAL` instead of a typed `EOVERFLOW`.
@Suite struct ParallelExpertReaderBoundsTests {
    private static let stride = 4096

    nonisolated(unsafe) fileprivate static var injectedReadIndex = 0
    nonisolated(unsafe) fileprivate static var gateEntered: DispatchSemaphore?
    nonisolated(unsafe) fileprivate static var gateRelease: DispatchSemaphore?
    nonisolated(unsafe) fileprivate static var gateCalls = 0
    nonisolated(unsafe) fileprivate static var firstStatus: Int32 = -1
    nonisolated(unsafe) fileprivate static var secondStatus: Int32 = -1
    nonisolated(unsafe) fileprivate static var pausedHandle: OpaquePointer?
    nonisolated(unsafe) fileprivate static var pausedFirst: UnsafeMutableRawPointer?
    nonisolated(unsafe) fileprivate static var pausedSecond: UnsafeMutableRawPointer?
    nonisolated(unsafe) fileprivate static var pausedFirstDone: DispatchSemaphore?
    nonisolated(unsafe) fileprivate static var pausedSecondDone: DispatchSemaphore?

    private static func makeFixture(blocks: Int) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("tinytitan-io-bounds-\(UUID().uuidString).bin")
        var data = Data(capacity: blocks * stride)
        for block in 0..<blocks {
            data.append(Data(repeating: UInt8(block % 251), count: stride))
        }
        try data.write(to: url)
        return url
    }

    private static func withReader<T>(
        path: String, stride: Int, threads: Int = 1,
        _ body: (OpaquePointer) throws -> T
    ) throws -> T {
        var failure: Int32 = 0
        let handle = path.withCString { pointer in
            tinytitan_expert_reader_create(pointer, stride, Int32(threads), 0, &failure)
        }
        guard let handle else {
            throw ParallelExpertReader.Failure.openFailed(path: path, errno: failure)
        }
        defer { tinytitan_expert_reader_destroy(handle) }
        return try body(handle)
    }

    private static func withDestinations<T>(
        _ count: Int, bytes: Int, _ body: ([UnsafeMutableRawPointer?]) throws -> T
    ) rethrows -> T {
        let buffers = (0..<count).map { _ in
            UnsafeMutableRawPointer.allocate(byteCount: bytes, alignment: 8)
        }
        defer { for buffer in buffers { buffer.deallocate() } }
        return try body(buffers.map { Optional($0) })
    }

    private static func fetchOffsets(
        _ handle: OpaquePointer, offsets: [UInt64], destinations: [UnsafeMutableRawPointer?]
    ) -> Int32 {
        offsets.withUnsafeBufferPointer { offsetBuffer in
            destinations.withUnsafeBufferPointer { destinationBuffer in
                tinytitan_expert_reader_fetch_offsets(
                    handle, offsetBuffer.baseAddress, destinationBuffer.baseAddress,
                    offsets.count)
            }
        }
    }

    private static func fetchRanges(
        _ handle: OpaquePointer, offsets: [UInt64], counts: [UInt32],
        destinations: [UnsafeMutableRawPointer?]
    ) -> Int32 {
        offsets.withUnsafeBufferPointer { offsetBuffer in
            counts.withUnsafeBufferPointer { countBuffer in
                destinations.withUnsafeBufferPointer { destinationBuffer in
                    tinytitan_expert_reader_fetch_ranges(
                        handle, offsetBuffer.baseAddress, countBuffer.baseAddress,
                        destinationBuffer.baseAddress, offsets.count)
                }
            }
        }
    }

    private static func fetchIds(
        _ handle: OpaquePointer, ids: [UInt32], destinations: [UnsafeMutableRawPointer?]
    ) -> Int32 {
        ids.withUnsafeBufferPointer { idBuffer in
            destinations.withUnsafeBufferPointer { destinationBuffer in
                tinytitan_expert_reader_fetch(
                    handle, idBuffer.baseAddress, destinationBuffer.baseAddress, ids.count)
            }
        }
    }

    // MARK: - Valid geometry

    @Test func validFullStrideBatchReadsEveryExpert() throws {
        let url = try Self.makeFixture(blocks: 2)
        defer { try? FileManager.default.removeItem(at: url) }
        let first = UnsafeMutableRawPointer.allocate(byteCount: Self.stride, alignment: 8)
        let second = UnsafeMutableRawPointer.allocate(byteCount: Self.stride, alignment: 8)
        defer {
            first.deallocate()
            second.deallocate()
        }
        try Self.withReader(path: url.path, stride: Self.stride, threads: 2) { handle in
            let status = Self.fetchOffsets(
                handle, offsets: [0, UInt64(Self.stride)],
                destinations: [first, second])
            #expect(status == 0)
            #expect(first.assumingMemoryBound(to: UInt8.self)[0] == 0)
            #expect(first.assumingMemoryBound(to: UInt8.self)[Self.stride - 1] == 0)
            #expect(second.assumingMemoryBound(to: UInt8.self)[0] == 1)
            #expect(second.assumingMemoryBound(to: UInt8.self)[Self.stride - 1] == 1)
        }
    }

    /// The last byte of the file is addressable, and the range ends there.
    @Test func rangeAtTheLastByteIsValid() throws {
        let url = try Self.makeFixture(blocks: 1)
        defer { try? FileManager.default.removeItem(at: url) }
        let destination = UnsafeMutableRawPointer.allocate(byteCount: 1, alignment: 1)
        defer { destination.deallocate() }
        try Self.withReader(path: url.path, stride: Self.stride) { handle in
            let status = Self.fetchRanges(
                handle, offsets: [UInt64(Self.stride - 1)], counts: [1],
                destinations: [destination])
            #expect(status == 0)
            #expect(destination.assumingMemoryBound(to: UInt8.self)[0] == 0)
        }
    }

    // MARK: - Refused offsets

    @Test func rangeStartingAtEndOfFileIsShort() throws {
        let url = try Self.makeFixture(blocks: 1)
        defer { try? FileManager.default.removeItem(at: url) }
        let destination = UnsafeMutableRawPointer.allocate(byteCount: 8, alignment: 8)
        defer { destination.deallocate() }
        destination.storeBytes(of: UInt64(0xA5A5_A5A5_A5A5_A5A5), as: UInt64.self)
        try Self.withReader(path: url.path, stride: Self.stride) { handle in
            let status = Self.fetchRanges(
                handle, offsets: [UInt64(Self.stride)], counts: [8],
                destinations: [destination])
            #expect(status == EIO)
            #expect(destination.load(as: UInt64.self) == 0xA5A5_A5A5_A5A5_A5A5)
        }
    }

    @Test func offsetAboveOffTMaxIsRejected() throws {
        let url = try Self.makeFixture(blocks: 1)
        defer { try? FileManager.default.removeItem(at: url) }
        let destination = UnsafeMutableRawPointer.allocate(byteCount: 8, alignment: 8)
        defer { destination.deallocate() }
        destination.storeBytes(of: UInt64(0xA5A5_A5A5_A5A5_A5A5), as: UInt64.self)
        try Self.withReader(path: url.path, stride: Self.stride) { handle in
            let beyond = UInt64(Int64.max) + 1
            let status = Self.fetchRanges(
                handle, offsets: [beyond], counts: [1],
                destinations: [destination])
            #expect(status == EOVERFLOW)
            #expect(destination.load(as: UInt64.self) == 0xA5A5_A5A5_A5A5_A5A5)
        }
    }

    @Test func offsetPlusLengthOverflowIsRejected() throws {
        let url = try Self.makeFixture(blocks: 1)
        defer { try? FileManager.default.removeItem(at: url) }
        let destination = UnsafeMutableRawPointer.allocate(byteCount: 8, alignment: 8)
        defer { destination.deallocate() }
        destination.storeBytes(of: UInt64(0xA5A5_A5A5_A5A5_A5A5), as: UInt64.self)
        try Self.withReader(path: url.path, stride: Self.stride) { handle in
            let status = Self.fetchRanges(
                handle, offsets: [UInt64.max - 1], counts: [8],
                destinations: [destination])
            #expect(status == EOVERFLOW)
            #expect(destination.load(as: UInt64.self) == 0xA5A5_A5A5_A5A5_A5A5)
        }
    }

    /// `offset` alone fits `off_t`, but the last byte of the range does not.
    @Test func rangePastOffTMaxIsRejected() throws {
        let url = try Self.makeFixture(blocks: 1)
        defer { try? FileManager.default.removeItem(at: url) }
        let destination = UnsafeMutableRawPointer.allocate(byteCount: 8, alignment: 8)
        defer { destination.deallocate() }
        destination.storeBytes(of: UInt64(0xA5A5_A5A5_A5A5_A5A5), as: UInt64.self)
        try Self.withReader(path: url.path, stride: Self.stride) { handle in
            let status = Self.fetchRanges(
                handle, offsets: [UInt64(Int64.max)], counts: [2],
                destinations: [destination])
            #expect(status == EOVERFLOW)
            #expect(destination.load(as: UInt64.self) == 0xA5A5_A5A5_A5A5_A5A5)
        }
    }

    @Test func rangeLongerThanPreadAcceptsIsRejected() throws {
        let url = try Self.makeFixture(blocks: 1)
        defer { try? FileManager.default.removeItem(at: url) }
        let destination = UnsafeMutableRawPointer.allocate(byteCount: 1, alignment: 1)
        defer { destination.deallocate() }
        destination.storeBytes(of: UInt8(0xA5), as: UInt8.self)
        try Self.withReader(path: url.path, stride: Self.stride) { handle in
            // Longer than INT_MAX: Darwin's pread answers EINVAL, so the reader
            // refuses it as a range overflow before the kernel sees it.
            let status = Self.fetchRanges(
                handle, offsets: [0], counts: [UInt32(0x8000_0000)],
                destinations: [destination])
            #expect(status == EOVERFLOW)
            #expect(destination.load(as: UInt8.self) == 0xA5)
        }
    }

    /// `expert_id * expert_stride` wraps for a wide stride before any read.
    @Test func expertIdTimesStrideOverflowIsRejected() throws {
        let url = try Self.makeFixture(blocks: 1)
        defer { try? FileManager.default.removeItem(at: url) }
        let destination = UnsafeMutableRawPointer.allocate(byteCount: 1, alignment: 1)
        defer { destination.deallocate() }
        destination.storeBytes(of: UInt8(0xA5), as: UInt8.self)
        try Self.withReader(path: url.path, stride: 1 << 40) { handle in
            let status = Self.fetchIds(
                handle, ids: [UInt32.max], destinations: [destination])
            #expect(status == EOVERFLOW)
            #expect(destination.load(as: UInt8.self) == 0xA5)
        }
    }

    @Test func zeroRangeLengthIsRejected() throws {
        let url = try Self.makeFixture(blocks: 1)
        defer { try? FileManager.default.removeItem(at: url) }
        try Self.withReader(path: url.path, stride: Self.stride) { handle in
            Self.withDestinations(1, bytes: 1) { destinations in
                #expect(
                    Self.fetchRanges(
                        handle, offsets: [0], counts: [0], destinations: destinations) == EINVAL)
            }
            #expect(Self.fetchOffsets(handle, offsets: [], destinations: []) == 0)
        }
    }

    @Test func nullBatchArgumentsAreRejected() throws {
        let url = try Self.makeFixture(blocks: 1)
        defer { try? FileManager.default.removeItem(at: url) }
        try Self.withReader(path: url.path, stride: Self.stride) { handle in
            let ids: [UInt32] = [0]
            let status = ids.withUnsafeBufferPointer { buffer in
                tinytitan_expert_reader_fetch(handle, buffer.baseAddress, nil, 1)
            }
            #expect(status == EINVAL)
            let offsets: [UInt64] = [0]
            let offsetStatus = offsets.withUnsafeBufferPointer { buffer in
                tinytitan_expert_reader_fetch_offsets(handle, buffer.baseAddress, nil, 1)
            }
            #expect(offsetStatus == EINVAL)
        }
    }

    // MARK: - Short read and EINTR

    /// The per-item loop must keep the bytes it already read across a short
    /// `pread` and retry after `EINTR`. A real APFS read never produces either
    /// on a small file, so the `pread` is injected.
    @Test func shortReadAndEINTRAreRetried() throws {
        let url = try Self.makeFixture(blocks: 1)
        defer { try? FileManager.default.removeItem(at: url) }
        let fd = open(url.path, O_RDONLY)
        try #require(fd >= 0)
        defer { close(fd) }
        Self.injectedReadIndex = 0
        let destination = UnsafeMutableRawPointer.allocate(byteCount: 4, alignment: 1)
        defer { destination.deallocate() }
        destination.storeBytes(of: UInt32(0xA5A5_A5A5), as: UInt32.self)
        let status = tinytitan_expert_reader_testing_read_one(
            fd, destination, 4, 0, shortThenInterruptedThenPread)
        #expect(status == 0)
        #expect(destination.load(as: UInt32.self) == 0)
        #expect(Self.injectedReadIndex >= 3)
    }

    // MARK: - Lifecycle

    /// A worker paused mid-batch, a second submit queued behind it, and a
    /// bounded wait for both. The queue path is deterministic here because the
    /// worker is parked at a known point instead of racing a sleep.
    ///
    /// Destroy is not attempted here: the C contract (and the Swift wrapper's
    /// ARC lifetime) says no destroy while a fetch is active or queued, so the
    /// test proves the queued handshake completes and then destroys.
    @Test func pausedWorkerSerialisesAQueuedBatch() throws {
        let url = try Self.makeFixture(blocks: 2)
        defer { try? FileManager.default.removeItem(at: url) }
        let stride = Self.stride
        let first = UnsafeMutableRawPointer.allocate(byteCount: stride, alignment: 8)
        let second = UnsafeMutableRawPointer.allocate(byteCount: stride, alignment: 8)
        defer {
            first.deallocate()
            second.deallocate()
        }
        Self.gateEntered = DispatchSemaphore(value: 0)
        Self.gateRelease = DispatchSemaphore(value: 0)
        Self.gateCalls = 0
        Self.firstStatus = -1
        Self.secondStatus = -1
        Self.pausedFirstDone = DispatchSemaphore(value: 0)
        Self.pausedSecondDone = DispatchSemaphore(value: 0)
        tinytitan_expert_reader_testing_set_gate(pauseOnFirstItem)

        try Self.withReader(path: url.path, stride: stride, threads: 1) { handle in
            Self.pausedHandle = handle
            Self.pausedFirst = first
            Self.pausedSecond = second
            // Only Sendable synchronisation is captured: the handle and the
            // destination pointers are single-assignment statics set before the
            // threads start.
            Thread.detachNewThread {
                guard let handle = Self.pausedHandle, let first = Self.pausedFirst else {
                    return
                }
                Self.firstStatus = Self.withUnsafeFetch(handle, offset: 0, into: first)
                Self.pausedFirstDone?.signal()
            }
            // The worker is parked before it reads the first item, so the batch
            // is published and `batch_idle` is not signalled yet.
            let parked = Self.gateEntered?.wait(timeout: .now() + 20)
            Thread.detachNewThread {
                guard let handle = Self.pausedHandle, let second = Self.pausedSecond else {
                    return
                }
                Self.secondStatus = Self.withUnsafeFetch(
                    handle, offset: UInt64(stride), into: second)
                Self.pausedSecondDone?.signal()
            }
            Self.gateRelease?.signal()
            let finishedFirst = Self.pausedFirstDone?.wait(timeout: .now() + 30)
            let finishedSecond = Self.pausedSecondDone?.wait(timeout: .now() + 30)
            #expect(parked == .success, "the worker never reached the gate")
            #expect(finishedFirst == .success, "the paused batch did not finish")
            #expect(finishedSecond == .success, "the queued batch did not finish")
            #expect(Self.firstStatus == 0)
            #expect(Self.secondStatus == 0)
            #expect(first.assumingMemoryBound(to: UInt8.self)[0] == 0)
            #expect(second.assumingMemoryBound(to: UInt8.self)[0] == 1)
        }
        tinytitan_expert_reader_testing_set_gate(nil)
    }

    /// A one-item full-stride fetch for the paused-worker test.
    private static func withUnsafeFetch(
        _ handle: OpaquePointer, offset: UInt64, into destination: UnsafeMutableRawPointer
    ) -> Int32 {
        var offsets = [offset]
        var destinations: [UnsafeMutableRawPointer?] = [destination]
        return offsets.withUnsafeMutableBufferPointer { offsetBuffer in
            destinations.withUnsafeMutableBufferPointer { destinationBuffer in
                tinytitan_expert_reader_fetch_offsets(
                    handle, offsetBuffer.baseAddress, destinationBuffer.baseAddress, 1)
            }
        }
    }
}

private func shortThenInterruptedThenPread(
    _ fd: Int32, _ buffer: UnsafeMutableRawPointer?, _ count: Int, _ offset: Int64
) -> Int {
    let index = ParallelExpertReaderBoundsTests.injectedReadIndex
    ParallelExpertReaderBoundsTests.injectedReadIndex += 1
    if index == 0 {
        return pread(fd, buffer, 1, offset)
    }
    if index == 1 {
        errno = EINTR
        return -1
    }
    return pread(fd, buffer, count, offset)
}

private func pauseOnFirstItem() {
    ParallelExpertReaderBoundsTests.gateCalls += 1
    if ParallelExpertReaderBoundsTests.gateCalls == 1 {
        ParallelExpertReaderBoundsTests.gateEntered?.signal()
        ParallelExpertReaderBoundsTests.gateRelease?.wait()
    }
}
