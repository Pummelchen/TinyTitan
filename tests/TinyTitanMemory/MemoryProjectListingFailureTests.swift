import ContinuityCore
import Foundation
import Testing

@testable import TinyTitanMemory

/// A directory that cannot be listed is not an empty directory.
///
/// `FileManager.enumerator` answers nil for a path that is missing, is not a
/// directory, or cannot be read. `discoverAll` used to turn all three into an
/// empty result, so `tinytitan-memory projects` said "no project files" over an
/// unreadable directory — a claim about the contents that nothing established.
/// The scan now carries its own failure, separate from the per-file `broken`
/// list, and the tool refuses the command instead of answering.
@Suite struct MemoryProjectListingFailureTests {
    @Test func anUnlistableDirectoryIsReported() async throws {
        let fixture = try ProjectDirectoryFixture()
        defer { fixture.cleanup() }
        try await fixture.makeProject("repo-a")
        try fixture.makeUnlistable()

        let discovery = MemoryProjectFile.discoverAll(in: fixture.directory)
        let failure = try #require(discovery.listingFailure)
        #expect(
            failure.url.resolvingSymlinksInPath().path
                == fixture.directory.resolvingSymlinksInPath().path)
        #expect(
            failure.reason.lowercased().contains("permission denied"),
            "unexpected reason: \(failure.reason)")
        // The scan did not happen, so the lists say nothing about the files.
        #expect(discovery.projects.isEmpty)
        #expect(discovery.broken.isEmpty)
        #expect(!discovery.isEmpty, "an unlistable directory read as empty")
    }

    @Test func aMissingDirectoryIsReported() throws {
        let fixture = try ProjectDirectoryFixture()
        defer { fixture.cleanup() }
        let missing = fixture.directory.appendingPathComponent("no-such-project-dir")

        let discovery = MemoryProjectFile.discoverAll(in: missing)
        let failure = try #require(discovery.listingFailure)
        #expect(failure.url.path == missing.path)
        #expect(failure.reason == "does not exist")
        #expect(!discovery.isEmpty)
    }

    @Test func aFileWhereADirectoryIsExpectedIsReported() throws {
        let fixture = try ProjectDirectoryFixture()
        defer { fixture.cleanup() }
        let file = try fixture.makeFileWhereADirectoryIsExpected()

        let discovery = MemoryProjectFile.discoverAll(in: file)
        let failure = try #require(discovery.listingFailure)
        #expect(failure.url.path == file.path)
        #expect(failure.reason == "is not a directory")
    }

    /// The compatibility scan has no channel for this, by its contract; what
    /// matters is that the detailed one does.
    @Test func theCompatibilityDiscoverAnswersAnEmptyList() throws {
        let fixture = try ProjectDirectoryFixture()
        defer { fixture.cleanup() }
        let missing = fixture.directory.appendingPathComponent("no-such-project-dir")
        #expect(MemoryProjectFile.discover(in: missing).isEmpty)
        #expect(MemoryProjectFile.discoverAll(in: missing).listingFailure != nil)
    }

    /// The clean and broken-file cases from 0061 are unchanged: a listed
    /// directory has no listing failure, whether or not a file is broken.
    @Test func aListedDirectoryHasNoListingFailure() async throws {
        let fixture = try ProjectDirectoryFixture()
        defer { fixture.cleanup() }
        try await fixture.makeProject("readable-project")
        try fixture.makeUnreadable("locked-project")

        let discovery = MemoryProjectFile.discoverAll(in: fixture.directory)
        #expect(discovery.listingFailure == nil)
        #expect(discovery.projects.count == 1)
        #expect(discovery.broken.count == 1)
        #expect(!discovery.isEmpty)
    }

    /// AUD-0064: the walk descends, so a directory it cannot enter one level
    /// down is the same defect as an unlistable root. The old enumerator
    /// yielded nothing for it and offered no error, so the answer read as
    /// complete.
    @Test func anUnlistableNestedDirectoryIsReported() async throws {
        let fixture = try ProjectDirectoryFixture()
        defer { fixture.cleanup() }
        try await fixture.makeProject("repo-a")
        let nested = try fixture.makeUnlistableSubdirectory("archive")

        let discovery = MemoryProjectFile.discoverAll(in: fixture.directory)
        #expect(discovery.listingFailure == nil)
        let failure = try #require(discovery.unlistableDirectories.first)
        #expect(
            failure.url.resolvingSymlinksInPath().path
                == nested.resolvingSymlinksInPath().path)
        #expect(
            failure.reason.lowercased().contains("permission denied"),
            "unexpected reason: \(failure.reason)")
        // What could be read is still read, and the scan is not "empty".
        #expect(discovery.projects.map(\.workspace) == ["repo-a"])
        #expect(!discovery.isEmpty, "an unreadable nested directory read as empty")
    }

    /// The walk rewrite must not lose what the enumerator found: a project in
    /// a readable subdirectory is still discovered, and reported as clean.
    @Test func aProjectInAReadableSubdirectoryIsStillFound() async throws {
        let fixture = try ProjectDirectoryFixture()
        defer { fixture.cleanup() }
        try await fixture.makeProject("repo-a")
        let archive = try fixture.makeNestedDirectory("archive")
        try await fixture.makeProject("repo-b", in: archive)

        let discovery = MemoryProjectFile.discoverAll(in: fixture.directory)
        #expect(discovery.unlistableDirectories.isEmpty)
        #expect(discovery.listingFailure == nil)
        #expect(
            Set(discovery.projects.map(\.workspace)) == ["repo-a", "repo-b"],
            "found: \(discovery.projects.map(\.workspace))")
    }
}
