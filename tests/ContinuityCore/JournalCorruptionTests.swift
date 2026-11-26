import Foundation
import Testing

@testable import ContinuityCore

/// A complete line that does not decode is not a torn tail.
///
/// A torn tail is a write that never reported success, so dropping it loses
/// nothing anyone was told was stored. A complete line was acknowledged by its
/// writer; it can be a bit flip, a file reused after truncation, or a record
/// from a build that writes a case this one does not know. Replaying the good
/// records around it is right. Treating the file as whole and compacting over
/// it is not: the checkpoint would not contain that line, and the only copy of
/// it would be gone.
@Suite struct JournalCorruptionTests {
    private func temporaryDirectory() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("journal-corruption-\(UUID().uuidString)")
    }

    private func task(_ title: String) -> JournalRecord {
        .task(ContinuityTask(title: title))
    }

    /// One record's bytes, from a journal of its own.
    private func encoded(_ record: JournalRecord, in directory: URL) async throws -> Data {
        let url = directory.appendingPathComponent("control-\(UUID().uuidString).ndjson")
        let journal = try FileJournal(url: url)
        try await journal.append(record)
        try await journal.shutDown()
        return try Data(contentsOf: url)
    }

    /// A journal file assembled from whole lines plus whatever damage the test
    /// wants, with the owner-only mode the journal requires of an existing
    /// file.
    @discardableResult
    private func writeJournal(
        _ contents: Data, to url: URL
    ) throws -> URL {
        try contents.write(to: url)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o600], ofItemAtPath: url.path)
        return url
    }

    private var unreadable: Data {
        // Complete JSON object, but not a record this build knows: the shape a
        // future case would have to an older reader.
        Data(#"{"somethingElse":{"version":9}}"#.utf8) + Data([0x0A])
    }

    // MARK: - What is damage and what is not

    @Test func aCompleteRecordThatDoesNotDecodeIsReportedWithItsLine() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let first = try await encoded(task("before"), in: directory)
        let second = try await encoded(task("after"), in: directory)
        try writeJournal(first + unreadable + second, to: url)

        let detailed = try FileJournal.readDetailed(contentsOf: url)
        #expect(detailed.records.count == 2)
        let damage = try #require(detailed.damage)
        #expect(damage.lines.count == 1)
        #expect(damage.lines[0].number == 2)
        #expect(damage.lines[0].byteOffset == first.count)
        // The plain reader keeps its old shape: good records, no throw.
        #expect(try FileJournal.read(contentsOf: url).count == 2)
    }

    @Test func anUnknownRecordInTheMiddleIsReported() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let first = try await encoded(task("before"), in: directory)
        let second = try await encoded(task("after"), in: directory)
        try writeJournal(first + unreadable + second, to: url)

        let journal = try FileJournal(url: url)
        let records = try await journal.replay()
        #expect(records.count == 2)
        let damage = try #require(await journal.replayDamage)
        #expect(damage.lineNumbers == [2])
        try await journal.shutDown()
    }

    /// The tail that never reported success: dropped, and not damage.
    @Test func aTornFinalLineIsNotDamage() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let first = try await encoded(task("before"), in: directory)
        try writeJournal(first + Data(#"{"task":{"_0":{"tit"#.utf8), to: url)

        let detailed = try FileJournal.readDetailed(contentsOf: url)
        #expect(detailed.records.count == 1)
        #expect(detailed.damage == nil)

        let journal = try FileJournal(url: url)
        #expect(await journal.replayDamage == nil)
        #expect(try await journal.replay().count == 1)
        // Opening cut the partial line; the complete record is untouched.
        #expect(try Data(contentsOf: url) == first)
        try await journal.shutDown()
    }

    @Test func aBlankLineIsNotDamage() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let first = try await encoded(task("before"), in: directory)
        let second = try await encoded(task("after"), in: directory)
        try writeJournal(first + Data([0x0A]) + second, to: url)

        let detailed = try FileJournal.readDetailed(contentsOf: url)
        #expect(detailed.records.count == 2)
        #expect(detailed.damage == nil)
    }

    // MARK: - What damage stops

    @Test func aDamagedReplayRefusesCompactionAndPreservesTheFile() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let first = try await encoded(task("before"), in: directory)
        let second = try await encoded(task("after"), in: directory)
        try writeJournal(first + unreadable + second, to: url)
        let before = try Data(contentsOf: url)

        let journal = try FileJournal(url: url)
        #expect(try await journal.replay().count == 2)
        do {
            try await journal.compact(sessionLog: SessionLogSnapshot(), memory: MemorySnapshot())
            Issue.record("a damaged replay was compacted over")
        } catch let error as JournalError {
            guard case .damagedReplay(let failedURL, let lines) = error else {
                Issue.record("expected damagedReplay, got \(error)")
                return
            }
            #expect(failedURL.path == url.path)
            #expect(lines == [2])
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
        // Truncation is the other way to destroy the bytes.
        await #expect(throws: JournalError.self) {
            try await journal.truncate()
        }
        #expect(try Data(contentsOf: url) == before)

        // The explicit recovery step is the only thing that allows a rewrite.
        await journal.discardDamagedRecords()
        try await journal.compact(sessionLog: SessionLogSnapshot(), memory: MemorySnapshot())
        #expect(try await journal.replay().count == 1)
        try await journal.shutDown()
        #expect(try FileJournal.read(contentsOf: url).count == 1)
    }

    /// The older read is lossy by contract and its replacement reports the
    /// same records plus where the damage is. A caller that decides anything
    /// from the file uses the second one.
    @Test func theCompatibilityReadIsLossyByContract() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let first = try await encoded(task("before"), in: directory)
        let second = try await encoded(task("after"), in: directory)
        try writeJournal(first + unreadable + second, to: url)

        let detailed = try FileJournal.readDetailed(contentsOf: url)
        let lossy = try FileJournal.read(contentsOf: url)
        #expect(lossy == detailed.records)
        #expect(detailed.damage?.lineNumbers == [2])
    }

    @Test func aReadOnlyInspectionDoesNotChangeADamagedFile() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let first = try await encoded(task("before"), in: directory)
        let second = try await encoded(task("after"), in: directory)
        try writeJournal(first + unreadable + second, to: url)
        let before = try Data(contentsOf: url)

        for _ in 0..<3 {
            let detailed = try FileJournal.readDetailed(contentsOf: url)
            #expect(detailed.records.count == 2)
            #expect(detailed.damage?.lineNumbers == [2])
            #expect(try FileJournal.read(contentsOf: url).count == 2)
        }
        #expect(try Data(contentsOf: url) == before)
    }

    @Test func theEngineReportsDamageAndRefusesToReplaceTheFile() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let first = try await encoded(task("before"), in: directory)
        let second = try await encoded(task("after"), in: directory)
        try writeJournal(first + unreadable + second, to: url)
        let before = try Data(contentsOf: url)

        let journal = try FileJournal(url: url)
        let engine = ContinuityEngine(journal: journal)
        try await engine.start()
        // Availability first: the good records are restored.
        #expect(await engine.tasks().count == 2)
        #expect(
            await engine.journalDamage != nil,
            "a replay that skipped a complete record reported itself whole")
        await #expect(throws: JournalError.self) {
            try await engine.compactJournal()
        }
        #expect(try Data(contentsOf: url) == before)

        await journal.discardDamagedRecords()
        try await engine.compactJournal()
        #expect(try FileJournal.read(contentsOf: url).count == 1)
        await engine.shutDown()
    }
}
