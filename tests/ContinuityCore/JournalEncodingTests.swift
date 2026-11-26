import Foundation
import Testing

@testable import ContinuityCore

/// The journal's bytes are a contract.
///
/// `JSONEncoder` does not promise an object's key order, and in practice it
/// varies it between two encodes of the same value in one process. A journal
/// is compared byte for byte (`before == after` in recovery tests, a
/// crash-replay fixture, a golden) and is the thing an operator inspects after
/// an incident; an ordering that moves on its own makes a harmless re-encode
/// look like damage. The encoder therefore sorts keys, and these pin that.
@Suite struct JournalEncodingTests {
    private func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("journal-encoding-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        return directory
    }

    private func lines(_ url: URL) throws -> [String] {
        let data = try Data(contentsOf: url)
        return try data.split(separator: 0x0A, omittingEmptySubsequences: true)
            .map { line in
                try #require(String(bytes: line, encoding: .utf8))
            }
    }

    /// The same record, written several times, is the same bytes every time.
    @Test func theSameRecordEncodesToTheSameBytesEveryTime() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let journal = try FileJournal(url: url, idleDelay: .seconds(60))
        let record = JournalRecord.task(
            ContinuityTask(title: "stable", objective: "same bytes"))
        for _ in 0..<8 {
            try await journal.append(record)
        }
        try await journal.sync()
        try await journal.shutDown()

        let written = try lines(url)
        #expect(written.count == 8)
        #expect(
            Set(written).count == 1,
            "\(Set(written).count) different encodings of one record")
    }

    /// Keys are sorted at every level, so the shape of a line is predictable
    /// without decoding it. `createdAt` is the first task field
    /// lexicographically, which the old hash order would choose only by
    /// accident.
    @Test func everyObjectInALineHasItsKeysSorted() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let journal = try FileJournal(url: url, idleDelay: .seconds(60))
        let task = ContinuityTask(title: "sorted", objective: "keys")
        try await journal.append(.task(task))
        try await journal.sync()
        try await journal.shutDown()

        let line = try #require(try lines(url).first)
        #expect(
            line.hasPrefix(#"{"task":{"_0":{"createdAt":"#),
            "the first task key is not the lexicographically first one: \(line.prefix(60))")
        let keys = ["\"createdAt\"", "\"id\"", "\"objective\"", "\"title\"", "\"updatedAt\""]
        var previous: String.Index?
        for key in keys {
            let position = try #require(line.range(of: key)?.lowerBound)
            if let previous {
                #expect(previous < position, "\(key) is out of order")
            }
            previous = position
        }
    }

    /// A record written before the ordering was fixed — unsorted keys, which
    /// is what an existing install holds — still decodes. JSON objects are
    /// unordered; the decoder keys by name.
    @Test func aRecordWithUnsortedKeysStillDecodes() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let task = ContinuityTask(title: "old build", objective: "unsorted keys")

        // The encoder this type used before the fix: ISO8601 dates, no
        // sortedKeys.
        let legacy = JSONEncoder()
        legacy.dateEncodingStrategy = .iso8601
        legacy.outputFormatting = [.withoutEscapingSlashes]
        var data = try legacy.encode(JournalRecord.task(task))
        data.append(0x0A)
        try data.write(to: url)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o600], ofItemAtPath: url.path)

        let detailed = try FileJournal.readDetailed(contentsOf: url)
        #expect(detailed.damage == nil)
        #expect(detailed.records.count == 1)
        guard case .task(let restored)? = detailed.records.first else {
            Issue.record("the legacy record did not decode as a task")
            return
        }
        #expect(restored.id == task.id)
        #expect(restored.title == task.title)

        let journal = try FileJournal(url: url, idleDelay: .seconds(60))
        #expect(try await journal.replay().count == 1)
        // Appending to it writes the new, sorted form beside the old one.
        try await journal.append(.task(ContinuityTask(title: "new build")))
        try await journal.sync()
        try await journal.shutDown()
        #expect(try FileJournal.read(contentsOf: url).count == 2)
    }
}
