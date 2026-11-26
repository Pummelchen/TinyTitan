import ContinuityCore
import Foundation
import Testing

@testable import TinyTitanMemory

/// The compatibility read that a person's `tinytitan-memory` commands use.
///
/// A complete line that does not decode is skipped and the rest is readable,
/// which is the right availability trade — but the facts then shown are not
/// the whole file. `MemoryProjectFile` loads through `readDetailed` and keeps
/// the rejected line numbers, so no caller has to present damage as clean.
@Suite struct MemoryProjectFileDamageTests {
    @Test func aDamagedProjectKeepsItsReadableFactsAndNamesTheLine() async throws {
        let fixture = try await DamagedProjectFixture()
        defer { try? FileManager.default.removeItem(at: fixture.directory) }

        let file = try MemoryProjectFile.load(fixture.journal)
        #expect(file.damagedRecordLines == [DamagedProjectFixture.damagedLine])
        let fact = try #require(file.facts.first { $0.value == fixture.factValue })
        #expect(fact.key.hasSuffix("kept"))
        #expect(file.sessionCount == 1)
    }

    @Test func aCleanProjectReportsNoDamage() async throws {
        let fixture = try await DamagedProjectFixture(damaged: false)
        defer { try? FileManager.default.removeItem(at: fixture.directory) }

        let file = try MemoryProjectFile.load(fixture.journal)
        #expect(file.damagedRecordLines.isEmpty)
        #expect(file.facts.contains { $0.value == fixture.factValue })
    }

    /// Discovery is the path a person uses to find the file at all. A damaged
    /// file must still be listed, with its damage, rather than vanishing or
    /// looking clean.
    @Test func discoveryListsADamagedProjectWithItsLines() async throws {
        let fixture = try await DamagedProjectFixture()
        defer { try? FileManager.default.removeItem(at: fixture.directory) }

        let files = MemoryProjectFile.discover(in: fixture.directory)
        #expect(files.count == 1)
        #expect(files.first?.damagedRecordLines == [DamagedProjectFixture.damagedLine])
    }

    /// A file whose *only* content is damaged is not an empty project: the
    /// facts are empty and the damage is what says why.
    @Test func aFullyDamagedProjectIsNotEmptyItIsDamaged() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("memory-damage-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("only-damaged.ndjson")
        try Data(#"{"somethingElse":{"version":9}}"#.utf8 + [0x0A]).write(to: url)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o600], ofItemAtPath: url.path)

        let file = try MemoryProjectFile.load(url)
        #expect(file.facts.isEmpty)
        #expect(file.damagedRecordLines == [1])
    }
}
