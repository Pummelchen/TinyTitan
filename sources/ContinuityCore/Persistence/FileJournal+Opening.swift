import Foundation

/// Opening a journal for writing: its directory, its lock, its append
/// descriptor, and the line discipline of a file this process did not create.
///
/// Split out of `Journal.swift` (2026-10-08) under the 500-line-per-file rule
/// as pure code motion, plus `dropDanglingTail` for the damage it governs. The
/// three moved helpers widened from `private` to internal because the
/// initializer that calls them stays behind.
extension FileJournal {

    /// Every opener below passes `O_NOFOLLOW`.
    ///
    /// The journal, its lock and the compaction temp all live at paths this
    /// type derives from the store directory rather than paths it was handed,
    /// so anything that can create a file there can plant a link at one of
    /// them. Without the flag that turns a memory-store write into a write
    /// somewhere else entirely -- and at the two `O_TRUNC` sites, into a
    /// destructive one. `O_NOFOLLOW` applies to the final component only, so
    /// the operator symlinking the store *directory* onto another disk still
    /// works; only a link standing where a journal file is expected is
    /// refused, and it is refused rather than followed. Same rule the
    /// installer's `Posix.openCreateRW` enforces.
    static func prepareDirectory(for url: URL) throws {
        let manager = FileManager.default
        let directory = url.deletingLastPathComponent()
        if !manager.fileExists(atPath: directory.path) {
            try manager.createDirectory(
                at: directory, withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700])
        }
        var isDirectory: ObjCBool = false
        if manager.fileExists(atPath: url.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        {
            throw JournalError.notAFile(url)
        }
    }

    static func acquireLock(at lockURL: URL, journal: URL) throws -> Int32 {
        // Refusing a link here is what makes the lock guard the *path* the
        // journal is written under. Following one would put the flock on an
        // arbitrary inode, so two journals at two paths could both believe
        // they owned the same one -- which is the exact failure the lock
        // exists to prevent, and it would be silent.
        let descriptor = open(
            lockURL.path, O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard descriptor >= 0 else {
            throw JournalError.cannotOpen(
                lockURL,
                underlying: String(cString: strerror(errno)))
        }
        // flock is per open-file-description, so a second FileJournal on the
        // same path inside this process conflicts too. fcntl locks would not,
        // which is exactly why they are the wrong tool here.
        guard flock(descriptor, LOCK_EX | LOCK_NB) == 0 else {
            let code = errno
            close(descriptor)
            if code == EWOULDBLOCK { throw JournalError.locked(journal) }
            throw JournalError.cannotOpen(lockURL, underlying: String(cString: strerror(code)))
        }
        return descriptor
    }

    /// `O_APPEND` so every write lands at the end without a seek, which is
    /// what keeps a record from being written into the middle of another.
    ///
    /// `O_RDWR`, not `O_WRONLY`: the tail scan below reads through the same
    /// descriptor the appends write through, so the bytes it decides on and the
    /// truncate that follows cannot be separated by another writer's open. The
    /// file is owner-only and this process created it, so the read permission
    /// the flag asks for is the one it already has.
    static func openForAppend(_ url: URL) throws -> Int32 {
        let descriptor = open(
            url.path, O_RDWR | O_APPEND | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard descriptor >= 0 else {
            throw JournalError.cannotOpen(url, underlying: String(cString: strerror(errno)))
        }
        return descriptor
    }

    /// Cut the file back to its last complete line, before anything is appended.
    ///
    /// A record and its newline go out in one write, so a file that ends without
    /// a newline ends mid-record: a process killed during an append leaves
    /// exactly that, and the journal is reopened before anything else is written
    /// to it. That dangling line is what `replay` already drops, and dropping it
    /// is right. Leaving it in place is not: the next append goes to the end of
    /// the file, which is the end of that partial line, so the two fuse into one
    /// line that decodes as neither -- and the record lost that way is one whose
    /// writer got a successful `append`, in this process, with no failure
    /// anywhere for the degraded report to find. The cost of one torn write would
    /// otherwise be two records, silently, and the first record after a crash is
    /// exactly the one a session is most likely to still owe someone.
    ///
    /// Only the bytes after the final newline are removed, so every complete
    /// record survives. A file with no newline at all is one partial record and
    /// empties, which is what replay of it would have returned anyway.
    ///
    /// An append that fails partway *inside* a running process leaves the same
    /// dangling line, and it is repaired at the next open rather than at the
    /// failure: the writer there already has an error in hand and records the
    /// durability failure itself, and a partial `write(2)` has no seam to test
    /// through without a full disk.
    static func dropDanglingTail(descriptor: Int32, url: URL) throws {
        var status = stat()
        guard fstat(descriptor, &status) == 0 else {
            throw JournalError.readFailed(url, errno: errno)
        }
        var end = status.st_size
        var buffer = [UInt8](repeating: 0, count: 64 * 1024)
        var keep: off_t = 0
        while end > 0 {
            let size = min(off_t(buffer.count), end)
            let offset = end - size
            let read = pread(descriptor, &buffer, Int(size), offset)
            guard read >= 0 else {
                throw JournalError.readFailed(url, errno: errno)
            }
            if let index = buffer[0..<Int(read)].lastIndex(of: 0x0A) {
                keep = offset + off_t(index) + 1
                break
            }
            end = offset
        }
        guard keep != status.st_size else { return }
        guard ftruncate(descriptor, keep) == 0 else {
            throw JournalError.writeFailed(url, errno: errno)
        }
    }
}
