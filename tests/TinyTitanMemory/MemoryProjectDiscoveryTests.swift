import ContinuityCore
import Foundation
import Testing

@testable import TinyTitanMemory

/// Finding project files, including the ones that cannot be opened.
///
/// A file that is present but unreadable used to disappear from the listing:
/// `discover` skipped it with `try?`, so a project an operator could see in the
/// directory was missing from `tinytitan-memory projects` and `list` answered
/// "no project matches" for a file that was right there. It is now reported as
/// broken, with a reason that is a path and an errno — never record contents.
@Suite struct MemoryProjectDiscoveryTests {
    @Test func anUnreadableProjectIsReportedInsteadOfDropped() async throws {
        let fixture = try ProjectDirectoryFixture()
        defer { fixture.cleanup() }
        try await fixture.makeProject("readable-project")
        let unreadable = try fixture.makeUnreadable("locked-project")

        let discovery = MemoryProjectFile.discoverAll(in: fixture.directory)
        #expect(discovery.projects.map(\.workspace) == ["readable-project"])
        #expect(discovery.broken.map(\.workspace) == ["locked-project"])
        let broken = try #require(discovery.broken.first)
        // The enumerator answers with the resolved `/private/var` form while
        // the fixture holds the `/var` alias of the same file.
        #expect(
            broken.url.resolvingSymlinksInPath().path
                == unreadable.resolvingSymlinksInPath().path)
        #expect(
            broken.reason.lowercased().contains("permission denied"),
            "unexpected reason: \(broken.reason)")
        // The reason names the file and the errno, not its contents.
        #expect(!broken.reason.contains("value-for-"))
    }

    @Test func aDirectoryAtAProjectPathIsReported() async throws {
        let fixture = try ProjectDirectoryFixture()
        defer { fixture.cleanup() }
        try fixture.makeDirectoryAtProjectPath("blocked-project")

        let discovery = MemoryProjectFile.discoverAll(in: fixture.directory)
        #expect(discovery.projects.isEmpty)
        #expect(discovery.broken.map(\.workspace) == ["blocked-project"])
        let broken = try #require(discovery.broken.first)
        #expect(
            broken.reason.contains("not a regular file"),
            "unexpected reason: \(broken.reason)")
    }

    @Test func aFIFOAtAProjectPathIsReportedWithoutBlocking() async throws {
        let fixture = try ProjectDirectoryFixture()
        defer { fixture.cleanup() }
        try fixture.makeFIFOAtProjectPath("pipe-project")

        let discovery = MemoryProjectFile.discoverAll(in: fixture.directory)
        #expect(discovery.projects.isEmpty)
        #expect(discovery.broken.map(\.workspace) == ["pipe-project"])
        #expect(discovery.broken.first?.reason.contains("not a regular file") == true)
    }

    /// The compatibility list is the readable subset of the same scan.
    @Test func discoverIsTheReadableSubsetOfDiscoverAll() async throws {
        let fixture = try ProjectDirectoryFixture()
        defer { fixture.cleanup() }
        try await fixture.makeProject("readable-project")
        try fixture.makeUnreadable("locked-project")

        let discovery = MemoryProjectFile.discoverAll(in: fixture.directory)
        let compatibility = MemoryProjectFile.discover(in: fixture.directory)
        #expect(compatibility.map(\.workspace) == discovery.projects.map(\.workspace))
        #expect(discovery.broken.count == 1)
    }

    @Test func aCleanDirectoryHasNoBrokenEntries() async throws {
        let fixture = try ProjectDirectoryFixture()
        defer { fixture.cleanup() }
        try await fixture.makeProject("one")
        try await fixture.makeProject("two")

        let discovery = MemoryProjectFile.discoverAll(in: fixture.directory)
        #expect(discovery.broken.isEmpty)
        #expect(discovery.projects.count == 2)
        #expect(discovery.projects.allSatisfy { $0.damagedRecordLines.isEmpty })
        #expect(!discovery.isEmpty)
    }

    @Test func resolveSaysUnreadableRatherThanNotFound() async throws {
        let fixture = try ProjectDirectoryFixture()
        defer { fixture.cleanup() }
        try await fixture.makeProject("readable-project")
        try fixture.makeUnreadable("locked-project")

        let discovery = MemoryProjectFile.discoverAll(in: fixture.directory)
        let result = MemoryProjectFile.resolve(
            "locked-project", among: discovery.projects, broken: discovery.broken)
        guard case .failure(let error) = result else {
            Issue.record("a broken project resolved as readable")
            return
        }
        #expect(error.description.contains("is present but could not be read"))
        #expect(error.description.contains("locked-project"))
        #expect(!error.description.contains("no project matches"))
    }

    @Test func resolveStillResolvesACleanProjectByPrefix() async throws {
        let fixture = try ProjectDirectoryFixture()
        defer { fixture.cleanup() }
        try await fixture.makeProject("photograph-851a1c1a")

        let discovery = MemoryProjectFile.discoverAll(in: fixture.directory)
        let result = MemoryProjectFile.resolve(
            "photograph", among: discovery.projects, broken: discovery.broken)
        guard case .success(let file) = result else {
            Issue.record("a clean project did not resolve")
            return
        }
        #expect(file.workspace == "photograph-851a1c1a")
    }

    @Test func aNameMatchingBothKindsIsAmbiguous() async throws {
        let fixture = try ProjectDirectoryFixture()
        defer { fixture.cleanup() }
        try await fixture.makeProject("photograph-aaaa1111")
        try fixture.makeUnreadable("photograph-bbbb2222")

        let discovery = MemoryProjectFile.discoverAll(in: fixture.directory)
        let result = MemoryProjectFile.resolve(
            "photograph", among: discovery.projects, broken: discovery.broken)
        guard case .failure(let error) = result else {
            Issue.record("two matching projects resolved to one")
            return
        }
        #expect(error.description.contains("photograph-aaaa1111"))
        #expect(error.description.contains("photograph-bbbb2222"))
    }
}
