import Foundation
import Testing

@testable import ContinuityCore

/// `pread` may answer with fewer bytes than asked for; the tail scan searched
/// whatever came back and then stepped over the rest of the chunk.
///
/// The scan walks backwards in 64 KiB chunks looking for the last newline, so
/// the bytes it steps over are the *end of the file* — exactly where the last
/// newline and the torn suffix are. Stepping over them sets `keep` to a chunk
/// boundary instead of the proven newline, and the `ftruncate` that follows
/// removes complete records. These fixtures put the last newline in the part a
/// short read would have skipped, and check the bytes of the file before and
/// after.
@Suite struct JournalTailRepairTests {
    private func temporaryDirectory() -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("journal-tail-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        return directory
    }

    /// A file with one complete record (no newline until the very end) and a
    /// four-byte torn suffix. Larger than one 64 KiB chunk.
    private func longRecordWithTornSuffix() -> (bytes: Data, valid: Data) {
        var bytes = Data(repeating: 0x61, count: 64 * 1024)
        bytes.append(contentsOf: Array("b\n".utf8))
        let valid = bytes
        bytes.append(contentsOf: Array("junk".utf8))
        return (bytes, valid)
    }

    private func writeJournal(_ contents: Data, to url: URL) throws {
        try contents.write(to: url)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o600], ofItemAtPath: url.path)
    }

    // MARK: - Short reads

    @Test func aShortReadCannotSkipTheLastNewline() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let fixture = longRecordWithTornSuffix()
        try writeJournal(fixture.bytes, to: url)

        let fileSystem = JournalFaultFileSystem()
        // The first scan of the final chunk answers with 100 bytes; the
        // newline is 65 531 bytes into that chunk.
        fileSystem.planReads([.bytes(100)])
        let journal = try FileJournal(url: url, fileSystem: fileSystem)
        try await journal.shutDown()

        #expect(fileSystem.reads >= 2, "the scan did not finish the chunk in pieces")
        let after = try Data(contentsOf: url)
        #expect(after == fixture.valid, "a short read truncated a complete record")
        #expect(after.count == 64 * 1024 + 2)
    }

    @Test func anInterruptedReadIsRetriedAndTheNewlineStillFound() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let fixture = longRecordWithTornSuffix()
        try writeJournal(fixture.bytes, to: url)

        let fileSystem = JournalFaultFileSystem()
        fileSystem.planReads([.interrupted, .bytes(100)])
        let journal = try FileJournal(url: url, fileSystem: fileSystem)
        try await journal.shutDown()

        #expect(try Data(contentsOf: url) == fixture.valid)
        #expect(fileSystem.reads >= 3)
    }

    /// Zero bytes while the scan knows the file is that long means the bytes it
    /// is deciding on are gone. Refusing leaves the file untouched; guessing
    /// truncates.
    @Test func aZeroReadLeavesTheFileAlone() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let fixture = longRecordWithTornSuffix()
        try writeJournal(fixture.bytes, to: url)

        let fileSystem = JournalFaultFileSystem()
        fileSystem.planReads([.zero])
        do {
            _ = try FileJournal(url: url, fileSystem: fileSystem)
            Issue.record("a scan that read nothing accepted the file")
        } catch let error as JournalError {
            guard case .readFailed(let failedURL, let code) = error else {
                Issue.record("expected readFailed, got \(error)")
                return
            }
            #expect(failedURL.path == url.path)
            #expect(code == EIO)
        }
        #expect(try Data(contentsOf: url) == fixture.bytes)
        #expect(fileSystem.ftruncates == 0)
    }

    @Test func aHardReadErrorLeavesTheFileAlone() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let fixture = longRecordWithTornSuffix()
        try writeJournal(fixture.bytes, to: url)

        let fileSystem = JournalFaultFileSystem()
        fileSystem.planReads([.failure(EIO)])
        do {
            _ = try FileJournal(url: url, fileSystem: fileSystem)
            Issue.record("a scan that failed accepted the file")
        } catch let error as JournalError {
            guard case .readFailed(_, let code) = error else {
                Issue.record("expected readFailed, got \(error)")
                return
            }
            #expect(code == EIO)
        }
        #expect(try Data(contentsOf: url) == fixture.bytes)
        #expect(fileSystem.ftruncates == 0)
    }

    // MARK: - The shapes that must keep working

    @Test func aFileEndingInANewlineIsNotTruncated() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        // Exactly one 64 KiB chunk, ending on the newline.
        let bytes = Data(repeating: 0x61, count: 64 * 1024 - 1) + Data([0x0A])
        try writeJournal(bytes, to: url)

        // Opening performs the scan; a journal object is not needed to keep it
        // alive for the assertion.
        let journal = try FileJournal(url: url)
        #expect(try Data(contentsOf: url) == bytes)
        #expect(try FileJournal.read(contentsOf: url).isEmpty)
        try await journal.shutDown()
        #expect(try Data(contentsOf: url) == bytes)
    }

    @Test func anEmptyFileIsNotTruncated() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        try writeJournal(Data(), to: url)
        let fileSystem = JournalFaultFileSystem()
        let journal = try FileJournal(url: url, fileSystem: fileSystem)
        #expect(fileSystem.reads == 0)
        #expect(fileSystem.ftruncates == 0)
        #expect(try Data(contentsOf: url).isEmpty)
        try await journal.shutDown()
    }

    @Test func anUnterminatedRecordWithNoNewlineIsEmptied() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let bytes = Data(#"{"task":{"_0":{"tit"#.utf8)
        try writeJournal(bytes, to: url)
        let journal = try FileJournal(url: url)
        #expect(try Data(contentsOf: url).isEmpty)
        try await journal.shutDown()
    }

    @Test func aTornSuffixIsCutToTheExactByte() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let valid = Data("{\"line\":1}\n{\"line\":2}\n".utf8)
        try writeJournal(valid + Data("partial".utf8), to: url)
        let journal = try FileJournal(url: url)
        #expect(try Data(contentsOf: url) == valid)
        try await journal.shutDown()
    }

    @Test func aMultiChunkHistoryWithoutATornTailIsUntouched() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        // Fill past two 64 KiB chunks with newline-terminated lines.
        var bytes = Data()
        while bytes.count < 3 * 64 * 1024 {
            bytes.append(Data("{\"line\":\(bytes.count)}\n".utf8))
        }
        try writeJournal(bytes, to: url)
        let journal = try FileJournal(url: url)
        #expect(try Data(contentsOf: url) == bytes)
        // The lines decode as records? They are not `JournalRecord` shapes, so
        // replay reports damage; what this test is about is the bytes.
        try await journal.shutDown()
        #expect(try Data(contentsOf: url) == bytes)
    }
}
