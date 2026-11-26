import Foundation

/// Opening a journal for writing: its directory, its lock, its append
/// descriptor, and the line discipline of a file this process did not create.
///
/// Split out of `Journal.swift` (2026-10-08) under the 500-line-per-file rule
/// as pure code motion, plus `dropDanglingTail` for the damage it governs. The
/// three moved helpers widened from `private` to internal because the
/// initializer that calls them stays behind.
///
/// Every opener here now validates the descriptor it got back as well as the
/// flags it asked for: `O_NOFOLLOW` refuses a link at the final component, but
/// it does not refuse a FIFO, a device, a directory, a file owned by someone
/// else or a journal the rest of the machine can read. The store directory is
/// operator-controlled and may be a symlink on purpose; the journal file
/// inside it is the part that must be exactly one regular, owner-only file.
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

    static func acquireLock(
        at lockURL: URL, journal: URL, using fileSystem: any JournalFileSystem
    ) throws(JournalError) -> Int32 {
        // Refusing a link here is what makes the lock guard the *path* the
        // journal is written under. Following one would put the flock on an
        // arbitrary inode, so two journals at two paths could both believe
        // they owned the same one -- which is the exact failure the lock
        // exists to prevent, and it would be silent.
        let descriptor = fileSystem.open(
            path: lockURL.path,
            flags: O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK,
            mode: 0o600)
        guard descriptor >= 0 else {
            throw .cannotOpen(lockURL, underlying: String(cString: strerror(-descriptor)))
        }
        // A FIFO or device at the lock path would make `flock` meaningless (or
        // blocking); only a regular file is a lock this process can own.
        do {
            try validateRegularFile(
                descriptor: descriptor, url: lockURL, using: fileSystem,
                checkOwner: true, checkMode: false)
        } catch {
            fileSystem.close(descriptor)
            throw error
        }
        // flock is per open-file-description, so a second FileJournal on the
        // same path inside this process also conflicts too. fcntl locks would
        // not, which is exactly why they are the wrong tool here.
        let locked = fileSystem.lock(descriptor, operation: LOCK_EX | LOCK_NB)
        guard locked == 0 else {
            fileSystem.close(descriptor)
            if -locked == EWOULDBLOCK { throw .locked(journal) }
            throw .cannotOpen(lockURL, underlying: String(cString: strerror(-locked)))
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
    ///
    /// The descriptor is checked, not assumed: a journal that another user
    /// owns, that the group can read, or that is not a regular file at all is
    /// refused before a byte is appended to it.
    static func openForAppend(
        _ url: URL, using fileSystem: any JournalFileSystem
    ) throws(JournalError) -> Int32 {
        let descriptor = fileSystem.open(
            path: url.path,
            flags: O_RDWR | O_APPEND | O_CREAT | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK,
            mode: 0o600)
        guard descriptor >= 0 else {
            if -descriptor == ELOOP { throw .notAFile(url) }
            throw .cannotOpen(url, underlying: String(cString: strerror(-descriptor)))
        }
        do {
            try validateRegularFile(
                descriptor: descriptor, url: url, using: fileSystem,
                checkOwner: true, checkMode: true)
        } catch {
            fileSystem.close(descriptor)
            throw error
        }
        return descriptor
    }

    /// Refuse anything a journal may not be, from the descriptor that was
    /// actually opened rather than from the path that was opened a moment
    /// earlier.
    ///
    /// The path check in `prepareDirectory` is a courtesy that produces a
    /// better message; this is the check that cannot be raced, because there
    /// is no second lookup between it and the first write.
    static func validateRegularFile(
        descriptor: Int32, url: URL, using fileSystem: any JournalFileSystem,
        checkOwner: Bool, checkMode: Bool
    ) throws(JournalError) {
        let (status, error) = fileSystem.fstat(descriptor)
        guard error == 0 else { throw .readFailed(url, errno: -error) }
        guard status.st_mode & S_IFMT == S_IFREG else {
            throw .notAFile(url)
        }
        if checkOwner {
            guard status.st_uid == geteuid() else {
                throw .wrongOwner(url, uid: status.st_uid)
            }
        }
        if checkMode, status.st_mode & 0o077 != 0 {
            throw .insecurePermissions(url, mode: status.st_mode)
        }
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
    static func dropDanglingTail(
        descriptor: Int32, url: URL, using fileSystem: any JournalFileSystem
    ) throws(JournalError) {
        let (status, error) = fileSystem.fstat(descriptor)
        guard error == 0 else { throw .readFailed(url, errno: -error) }
        var end = status.st_size
        var buffer = [UInt8](repeating: 0, count: 64 * 1024)
        var keep: off_t = 0
        while end > 0 {
            let size = min(off_t(buffer.count), end)
            let offset = end - size
            // A chunk is searched only once all of it has been read. `pread`
            // is allowed to answer with fewer bytes than asked for; searching
            // the prefix and then stepping over the rest, as this did, can
            // step over the last newline and truncate a complete record.
            try readExactly(
                descriptor, into: &buffer, count: Int(size), offset: offset,
                url: url, using: fileSystem)
            if let index = buffer[0..<Int(size)].lastIndex(of: 0x0A) {
                keep = offset + off_t(index) + 1
                break
            }
            end = offset
        }
        guard keep != status.st_size else { return }
        let truncated = fileSystem.ftruncate(descriptor, offset: keep)
        guard truncated == 0 else {
            throw .writeFailed(url, errno: -truncated)
        }
    }

    /// Fill `buffer[0..<count]` from `offset`, retrying an interrupted read
    /// and refusing to make progress out of nothing.
    ///
    /// The file's size came from `fstat` a moment ago, so a read that returns
    /// zero before the requested range is filled is not a short read to work
    /// around: it means the bytes this scan is deciding on are not there any
    /// more. Reporting that leaves the file untouched, which is the only safe
    /// answer when the range being scanned cannot be proven.
    static func readExactly(
        _ descriptor: Int32, into buffer: inout [UInt8], count: Int, offset: off_t,
        url: URL, using fileSystem: any JournalFileSystem
    ) throws(JournalError) {
        var filled = 0
        while filled < count {
            let read = buffer.withUnsafeMutableBytes { raw -> Int in
                fileSystem.pread(
                    descriptor,
                    into: UnsafeMutableRawBufferPointer(rebasing: raw[filled..<count]),
                    offset: offset + off_t(filled))
            }
            if read == 0 {
                throw .readFailed(url, errno: EIO)
            }
            if read < 0 {
                if read == -EINTR { continue }
                throw .readFailed(url, errno: Int32(-read))
            }
            if read > count - filled {
                throw .readFailed(url, errno: EIO)
            }
            filled += read
        }
    }
}
