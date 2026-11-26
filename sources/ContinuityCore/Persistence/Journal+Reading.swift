import Foundation

// Reading a journal: replay, the read-only inspector, and the decoder that
// separates a torn tail from a complete record that did not decode.
//
// Split out of `Journal.swift` under the 500-line-per-file rule (audit 0014).
// The members the reading path needs stay internal so an extension in another
// file can reach them; the public surface is unchanged.

extension FileJournal {

    public func replay() async throws -> [JournalRecord] {
        // Reading and decoding a journal is the one bulk operation this type
        // does, and it is done once, at start. It still goes to the blocking
        // queue: a large file on a slow disk must not pin a pool thread.
        let path = url.path
        let decoder = self.decoder
        let decoded: DecodedJournal = try await withCheckedThrowingContinuation { continuation in
            Self.blockingQueue.async {
                do {
                    let contents = try Self.readContents(atPath: path) ?? Data()
                    continuation.resume(returning: Self.decodeJournal(contents, decoder: decoder))
                } catch {
                    // Forwarded, not swallowed: a read that failed must not
                    // look like a journal that was empty. The caller decides
                    // whether to start without persistence.
                    continuation.resume(throwing: error)
                }
            }
        }
        // Recorded before returning so a compaction requested after this
        // replay can refuse to replace bytes the caller has not seen.
        replayDamage = decoded.damage
        return decoded.records
    }

    /// Read a journal without opening it for writing.
    ///
    /// Takes no lock, so it is safe to point at the file of a running server.
    /// A store only its own process can look at is a store nobody can debug.
    ///
    /// **This is the lossy compatibility read, by contract.** It returns the
    /// records that decoded and cannot tell the caller that a complete line
    /// did not; that distinction is `readDetailed`. It stays because callers
    /// that only need decodable records (and tests that assert bytes) should
    /// not have to handle a damage value they do not act on. A caller that
    /// decides anything *from* the file — lists a history, reports a project,
    /// rewrites the store — must use `readDetailed` and handle `damage`, or it
    /// will present an incomplete file as a whole one. The package's own
    /// callers do: `MemoryProjectFile.load` and `ContinuityDemo.inspect`.
    public static func read(contentsOf url: URL) throws -> [JournalRecord] {
        try readDetailed(contentsOf: url).records
    }

    /// The same read, with the positions of complete lines that did not
    /// decode. `read(contentsOf:)` keeps its old shape for the callers that
    /// only want the records; a tool that is diagnosing a file wants this.
    public static func readDetailed(contentsOf url: URL) throws -> JournalReplay {
        let decoder = Self.makeDecoder()
        // `nil` means the file is not there; anything else that goes wrong is
        // reported, so a caller listing workspaces does not show a file it could
        // not read as an empty one.
        guard let contents = try readContents(atPath: url.path) else {
            return JournalReplay(records: [], damage: nil)
        }
        let decoded = decodeJournal(contents, decoder: decoder)
        return JournalReplay(records: decoded.records, damage: decoded.damage)
    }

    /// The journal's bytes, or `nil` when there is no file at that path.
    ///
    /// The two cases have to be distinguishable: this is what the engine replays
    /// before it compacts, so reading a file that exists but cannot be read as
    /// "no records" writes a checkpoint over the only copy of those records.
    /// `FileManager.contents` answers `nil` for both and drops the errno with
    /// it.
    ///
    /// `O_NOFOLLOW` and the type check are what keep a read-only inspection
    /// from following a link planted at the journal path, or blocking on a
    /// FIFO and reading `/dev/zero` until memory runs out. A read may not be a
    /// way around the write path's guard just because it looks harmless.
    static func readContents(atPath path: String) throws -> Data? {
        let url = URL(fileURLWithPath: path)
        let descriptor = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK)
        if descriptor < 0 {
            if errno == ENOENT { return nil }
            // A link where the journal should be is refused, not resolved.
            if errno == ELOOP { throw JournalError.notAFile(url) }
            throw JournalError.readFailed(url, errno: errno)
        }
        defer { close(descriptor) }
        // Type only: a legacy file with a permissive mode is exactly the file
        // an operator runs this inspection to diagnose, and refusing to read
        // it would hide the problem. The write path is where a mode is
        // refused.
        try validateRegularFile(
            descriptor: descriptor, url: url,
            using: SystemJournalFileSystem(), checkOwner: false, checkMode: false)
        var contents = Data()
        var buffer = [UInt8](repeating: 0, count: 64 * 1024)
        while true {
            // `Darwin.read`, not a member: this type has a `read` of its own
            // and the unqualified name resolves to it.
            let count = Darwin.read(descriptor, &buffer, buffer.count)
            if count < 0 {
                if errno == EINTR { continue }
                throw JournalError.readFailed(url, errno: errno)
            }
            if count == 0 { return contents }
            contents.append(contentsOf: buffer[0..<count])
        }
    }

    /// Splits a journal into records, and says where a *complete* line did not
    /// decode.
    ///
    /// The distinction matters on the way back out. A final line with no
    /// terminator is a torn write: the append that would have completed it
    /// never reported success, so dropping it loses nothing that was
    /// acknowledged. A complete line that will not decode was acknowledged by
    /// some writer — a bit flip, a file reused after a truncation, or a case
    /// from an incompatible build — and the records around it are still worth
    /// replaying. Silently dropping it, as this did, let a later compaction
    /// replace the file with a checkpoint that did not contain it, which is
    /// the one way a replay can turn a damaged file into a permanently
    /// damaged one.
    static func decodeJournal(_ contents: Data, decoder: JSONDecoder) -> DecodedJournal {
        var records: [JournalRecord] = []
        var damaged: [JournalReplayDamage.Line] = []
        var start = contents.startIndex
        var lineNumber = 0
        while let newline = contents[start...].firstIndex(of: 0x0A) {
            let line = contents[start..<newline]
            lineNumber += 1
            let offset = contents.distance(from: contents.startIndex, to: start)
            start = contents.index(after: newline)
            guard !line.isEmpty else { continue }
            do {
                records.append(try decoder.decode(JournalRecord.self, from: Data(line)))
            } catch {
                // The record is untrusted bytes, so any decoding failure —
                // including a schema this build does not know — is exactly
                // the condition being classified. The contents are not kept:
                // the line did not decode, and it may hold a prompt.
                damaged.append(
                    JournalReplayDamage.Line(number: lineNumber, byteOffset: offset))
            }
        }
        return DecodedJournal(
            records: records,
            damage: damaged.isEmpty ? nil : JournalReplayDamage(lines: damaged))
    }
}

/// The decoder's two answers: what could be replayed, and what could not be
/// decoded even though it was complete.
struct DecodedJournal {
    let records: [JournalRecord]
    let damage: JournalReplayDamage?
}
