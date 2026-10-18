import Foundation

/// Row gather over Qwen3.8-Flash-Next's n-gram embedding table.
///
/// The table is ~320M rows of `embedDim / heads` fp16 values — 102 GB at the
/// production geometry, far past anything that could be resident. It is read
/// the way routed experts are: bypassing the page cache, so the declared
/// memory budget stays the real footprint.
///
/// The gather is much friendlier than an expert miss, though, and the
/// difference is worth stating because it shapes how this should be scheduled:
/// a row id depends only on token ids (see `PLEHash`), never on router output
/// or any activation. The rows for token N+1 are therefore knowable the moment
/// token N is sampled, so this read can be issued entirely off the decode
/// critical path — where an expert miss cannot be, because routing is not
/// known until the layer runs.
///
/// unchecked-invariant: immutable after `init`. The descriptor and geometry
/// are never mutated afterwards, and `pread` does not move a shared file
/// offset, so concurrent gathers only read.
public final class NgramTableReader: @unchecked Sendable {
    public enum Failure: Error, CustomStringConvertible {
        case openFailed(path: String, errno: Int32)
        case sizeMismatch(path: String, expected: UInt64, actual: UInt64)
        case invalidGeometry(rowDim: Int, rowCount: UInt64)
        case rowOutOfRange(row: UInt32, rowCount: UInt64)
        case readFailed(row: UInt32, errno: Int32)
        case shortRead(row: UInt32, expected: Int, got: Int)

        public var description: String {
            switch self {
            case .openFailed(let p, let e):
                return "n-gram table open(\(p)) failed: \(String(cString: strerror(e)))"
            case .sizeMismatch(let p, let expected, let actual):
                return "n-gram table \(p) is \(actual) bytes; expected at least "
                    + "\(expected) and at most 1 MiB of alignment padding past "
                    + "it. The table does not match the manifest's geometry"
            case .invalidGeometry(let rowDim, let count):
                return "n-gram table geometry is not usable: rowDim \(rowDim), "
                    + "rowCount \(count). Both come from ple_constants.json, so "
                    + "the sidecar is corrupt or built for another model"
            case .rowOutOfRange(let row, let count):
                return "n-gram row \(row) is outside the table's \(count) rows"
            case .readFailed(let row, let e):
                return "n-gram row \(row) read failed: \(String(cString: strerror(e)))"
            case .shortRead(let row, let expected, let got):
                return "n-gram row \(row) short read: expected \(expected), got \(got)"
            }
        }
    }

    private let fd: Int32
    public let path: String
    /// Values per row: `embedDim / totalHeads` (160 in production).
    public let rowDim: Int
    public let rowBytes: Int
    public let rowCount: UInt64

    /// - Parameters:
    ///   - rowCount: expected rows, from the manifest's PLE geometry. The file
    ///     size is checked against it: a table that disagrees would otherwise
    ///     be read at plausible but wrong offsets, returning real embedding
    ///     data from the wrong rows.
    ///   - bypassCache: keep the gather out of the page cache. Default on,
    ///     because a bounded footprint is the point of streaming.
    public init(
        path: String, rowDim: Int, rowCount: UInt64,
        bypassCache: Bool = true
    ) throws {
        // Thrown, not a precondition: both values come from `ple_constants.json`,
        // which is a file on disk rather than a fact of the build. A corrupt or
        // foreign sidecar used to abort the process here instead of reporting
        // itself.
        guard rowDim > 0, rowCount > 0 else {
            throw Failure.invalidGeometry(rowDim: rowDim, rowCount: rowCount)
        }
        let opened = open(path, O_RDONLY)
        guard opened >= 0 else {
            throw Failure.openFailed(path: path, errno: errno)
        }
        if bypassCache { _ = fcntl(opened, F_NOCACHE, 1) }
        var st = stat()
        guard fstat(opened, &st) == 0 else {
            let code = errno
            close(opened)
            throw Failure.openFailed(path: path, errno: code)
        }
        // Checked, not wrapping. `rowCount &* bytes` could wrap to a small
        // number, which collapses `expected` and lets the size guard below
        // accept a table built for a completely different geometry.
        let (rowBytes, byteOverflow) = rowDim.multipliedReportingOverflow(
            by: MemoryLayout<Float16>.stride)
        guard !byteOverflow, rowBytes > 0 else {
            throw Failure.invalidGeometry(rowDim: rowDim, rowCount: rowCount)
        }
        let bytes = rowBytes
        let (expected, sizeOverflow) = rowCount.multipliedReportingOverflow(
            by: UInt64(bytes))
        guard !sizeOverflow else {
            throw Failure.invalidGeometry(rowDim: rowDim, rowCount: rowCount)
        }
        // The checkpoint pads the table past its last addressable row (the
        // shipped one rounds up to a 512-row boundary), so an exact size is
        // the wrong check. What this must still catch is a table built for a
        // different geometry, and that differs by a whole head -- 20M rows,
        // 6.4 GB -- not by alignment. Allowing at most 1 MiB of surplus
        // separates the two by three orders of magnitude.
        let size = UInt64(st.st_size)
        let surplusLimit: UInt64 = 1 << 20
        guard size >= expected, size - expected <= surplusLimit,
            size % UInt64(bytes) == 0
        else {
            close(opened)
            throw Failure.sizeMismatch(
                path: path, expected: expected,
                actual: size)
        }
        self.fd = opened
        self.path = path
        self.rowDim = rowDim
        self.rowBytes = bytes
        self.rowCount = rowCount
    }

    deinit { close(fd) }

    /// Gathers `rows` into `destination`, which must hold
    /// `rows.count * rowDim` fp16 values. Rows land in the order given, which
    /// is head order — the caller concatenates them into the PLE input.
    public func gather(
        rows: [UInt32],
        into destination: UnsafeMutableRawPointer
    ) throws {
        for (i, row) in rows.enumerated() {
            guard UInt64(row) < rowCount else {
                throw Failure.rowOutOfRange(row: row, rowCount: rowCount)
            }
            let offset = UInt64(row) &* UInt64(rowBytes)
            let target = destination.advanced(by: i * rowBytes)
            var moved = 0
            while moved < rowBytes {
                let n = pread(
                    fd, target.advanced(by: moved),
                    rowBytes - moved, off_t(offset) + off_t(moved))
                if n < 0 {
                    if errno == EINTR { continue }
                    throw Failure.readFailed(row: row, errno: errno)
                }
                if n == 0 {
                    throw Failure.shortRead(
                        row: row, expected: rowBytes,
                        got: moved)
                }
                moved += n
            }
        }
    }

    /// Gathers independent batches, fanning the reads across `threads` callers.
    ///
    /// Batch `i` lands at `destination + i * chunkStride`, which is how a caller
    /// laying one row block per sequence position expects it.
    ///
    /// One row is a 320 B random read, so a gather's cost is round-trips rather
    /// than bandwidth. On a 412-token prefill of the 125B install the reads
    /// took 660 ms serially, 184 ms on four readers, 109 ms on eight and 111 ms
    /// on sixteen -- so eight is the knee, and the default. A single-threaded
    /// prefill does those reads one after another before layer 0, so the latency
    /// is exposed rather than hidden behind GPU work.
    ///
    /// `pread` does not move the shared file offset, so concurrent readers need
    /// no lock. The workers write disjoint regions of `destination`, so no byte
    /// is shared. The first failure is recorded and rethrown after every worker
    /// has stopped; the remaining rows of a failed batch are abandoned, which is
    /// the same partial state a serial gather leaves.
    public func gatherConcurrently(
        batches: [[UInt32]],
        chunkStride: Int,
        into destination: UnsafeMutableRawPointer,
        threads: Int = 8
    ) throws {
        let usable = max(1, min(threads, batches.count))
        guard usable > 1 else {
            for (index, rows) in batches.enumerated() {
                try gather(
                    rows: rows,
                    into: destination.advanced(by: index * chunkStride))
            }
            return
        }
        let chunk = (batches.count + usable - 1) / usable
        // `concurrentPerform` returns only after every iteration has finished,
        // so the buffer outlives the closure; Swift cannot see that through the
        // call and unsafe pointers are not `Sendable`, which is what this
        // binding records.
        nonisolated(unsafe) let destination = destination
        let failure = GatherFailure()
        DispatchQueue.concurrentPerform(iterations: usable) { worker in
            let lower = worker * chunk
            let upper = min(lower + chunk, batches.count)
            guard lower < upper else { return }
            for index in lower..<upper where !failure.isSet {
                do {
                    try gather(
                        rows: batches[index],
                        into: destination.advanced(by: index * chunkStride))
                } catch {
                    failure.record(error)
                }
            }
        }
        if let error = failure.error { throw error }
    }

    /// Bytes one token's gather moves, for budgeting. At the production
    /// geometry this is 16 x 320 B = 5 KiB per token — three orders of
    /// magnitude under a routed-expert token's traffic.
    public func gatherBytes(headCount: Int) -> Int { headCount * rowBytes }

    /// First failure across the gather workers, if any.
    ///
    /// unchecked-invariant: `stored` is the only mutable state, and every read
    /// and write of it happens under `lock`. `lock` is a `let`, so no caller can
    /// swap it out from under a concurrent `record`.
    private final class GatherFailure: @unchecked Sendable {
        private let lock = NSLock()
        private var stored: Error?

        var isSet: Bool {
            lock.lock()
            defer { lock.unlock() }
            return stored != nil
        }

        var error: Error? {
            lock.lock()
            defer { lock.unlock() }
            return stored
        }

        func record(_ error: Error) {
            lock.lock()
            if stored == nil { stored = error }
            lock.unlock()
        }
    }
}
