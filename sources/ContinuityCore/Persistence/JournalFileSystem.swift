import Foundation

/// The blocking filesystem calls `FileJournal` makes, behind one seam.
///
/// Every journal guarantee this package claims — that a checkpoint is not
/// renamed over the only durable copy before its data reached the platter,
/// that a short `pread` cannot truncate an unread record, that a zero-progress
/// `write` cannot spin — is only observable on a real fault, and a real drive
/// does not fail on request. The production implementation is
/// `SystemJournalFileSystem`, which makes exactly the syscalls the type made
/// before this seam existed; the tests inject an implementation that returns a
/// chosen outcome for a chosen call and records what was asked of it.
///
/// Every method answers a non-negative value on success and `-errno` on
/// failure, never touching the caller's `errno`. A seam that answered "failed"
/// with `-1` would force the test double to write thread-local `errno` from
/// whatever queue the call happened on, which is a different value on the
/// caller's thread.
protocol JournalFileSystem: Sendable {
    /// A descriptor, or `-errno`.
    func open(path: String, flags: Int32, mode: mode_t) -> Int32
    func close(_ descriptor: Int32)
    /// Bytes written (short writes are legal), or `-errno`.
    func write(_ descriptor: Int32, bytes: UnsafeRawBufferPointer) -> Int
    func read(_ descriptor: Int32, into buffer: UnsafeMutableRawBufferPointer) -> Int
    func pread(
        _ descriptor: Int32, into buffer: UnsafeMutableRawBufferPointer, offset: off_t
    ) -> Int
    func fsync(_ descriptor: Int32) -> Int32
    /// The Darwin power-cut barrier (`F_FULLFSYNC`); 0 or `-errno`.
    func fullSync(_ descriptor: Int32) -> Int32
    func rename(from: String, to: String) -> Int32
    func unlink(_ path: String) -> Int32
    func ftruncate(_ descriptor: Int32, offset: off_t) -> Int32
    func lock(_ descriptor: Int32, operation: Int32) -> Int32
    func fstat(_ descriptor: Int32) -> (status: stat, error: Int32)
}

/// The real syscalls, with `errno` captured on the line that set it.
struct SystemJournalFileSystem: JournalFileSystem {
    func open(path: String, flags: Int32, mode: mode_t) -> Int32 {
        let descriptor = Darwin.open(path, flags, mode)
        return descriptor >= 0 ? descriptor : -errno
    }

    func close(_ descriptor: Int32) {
        // Closing is the last thing done to a descriptor on every path, so
        // there is no caller left to tell. A close that fails has nothing to
        // retry: the descriptor is gone either way.
        _ = Darwin.close(descriptor)
    }

    func write(_ descriptor: Int32, bytes: UnsafeRawBufferPointer) -> Int {
        guard let base = bytes.baseAddress, !bytes.isEmpty else { return 0 }
        let written = Darwin.write(descriptor, base, bytes.count)
        return written >= 0 ? Int(written) : -Int(errno)
    }

    func read(_ descriptor: Int32, into buffer: UnsafeMutableRawBufferPointer) -> Int {
        guard let base = buffer.baseAddress, !buffer.isEmpty else { return 0 }
        let count = Darwin.read(descriptor, base, buffer.count)
        return count >= 0 ? Int(count) : -Int(errno)
    }

    func pread(
        _ descriptor: Int32, into buffer: UnsafeMutableRawBufferPointer, offset: off_t
    ) -> Int {
        guard let base = buffer.baseAddress, !buffer.isEmpty else { return 0 }
        let count = Darwin.pread(descriptor, base, buffer.count, offset)
        return count >= 0 ? Int(count) : -Int(errno)
    }

    func fsync(_ descriptor: Int32) -> Int32 {
        Darwin.fsync(descriptor) == 0 ? 0 : -errno
    }

    func fullSync(_ descriptor: Int32) -> Int32 {
        // Not every filesystem implements it; the caller decides whether the
        // `fsync` fallback is good enough for the write it is covering.
        fcntl(descriptor, F_FULLFSYNC) == 0 ? 0 : -errno
    }

    func rename(from: String, to: String) -> Int32 {
        Darwin.rename(from, to) == 0 ? 0 : -errno
    }

    func unlink(_ path: String) -> Int32 {
        Darwin.unlink(path) == 0 ? 0 : -errno
    }

    func ftruncate(_ descriptor: Int32, offset: off_t) -> Int32 {
        Darwin.ftruncate(descriptor, offset) == 0 ? 0 : -errno
    }

    func lock(_ descriptor: Int32, operation: Int32) -> Int32 {
        // Unqualified: `flock` also names a struct, and `Darwin.flock` picks
        // the type. The two-argument call is the C function.
        return flock(descriptor, operation) == 0 ? 0 : -errno
    }

    func fstat(_ descriptor: Int32) -> (status: stat, error: Int32) {
        var status = stat()
        guard Darwin.fstat(descriptor, &status) == 0 else { return (stat(), -errno) }
        return (status, 0)
    }
}
