import Foundation
import Testing

@testable import TinyTitanCore

/// A receipt is bound to the install's physical path, not the operator's
/// spelling of it (AUD-0072).
///
/// Keeping models on a second disk by symlinking the directory is a supported
/// layout. Recording the typed path made an untouched install refuse to load
/// with "model directory mismatch … Moving or renaming an installed model
/// invalidates its receipt" — a false diagnosis — and `--verify-install`
/// re-issued the same spelling.
@Suite struct InstallPathBindingTests {
    struct Tree {
        let root: URL
        let real: URL
        let alias: URL
    }

    func makeTree() throws -> Tree {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("install-binding-\(UUID().uuidString)")
        let real = root.appendingPathComponent("volumes/models")
        try FileManager.default.createDirectory(at: real, withIntermediateDirectories: true)
        let alias = root.appendingPathComponent("models")
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: real)
        return Tree(root: root, real: real, alias: alias)
    }

    @Test func theCanonicalFormResolvesASymlinkedParent() throws {
        let tree = try makeTree()
        defer { try? FileManager.default.removeItem(at: tree.root) }
        let install = tree.alias.appendingPathComponent("ornith")
        #expect(
            InstallPathBinding.canonical(install)
                == tree.real.appendingPathComponent("ornith").standardizedFileURL.path,
            "the leaf does not exist yet; the parent still has to resolve")
    }

    @Test func aReceiptMatchesTheCanonicalAndTheSpelledForm() throws {
        let tree = try makeTree()
        defer { try? FileManager.default.removeItem(at: tree.root) }
        let install = tree.alias.appendingPathComponent("ornith")
        try FileManager.default.createDirectory(at: install, withIntermediateDirectories: true)
        let spelled = install.standardizedFileURL.path
        let canonical = InstallPathBinding.canonical(install)
        #expect(spelled != canonical, "the fixture must actually be a symlinked parent")
        #expect(InstallPathBinding.matches(spelled, directory: install))
        #expect(InstallPathBinding.matches(canonical, directory: install))
        // The real path spelled directly is the same install.
        let realInstall = tree.real.appendingPathComponent("ornith")
        #expect(InstallPathBinding.matches(canonical, directory: realInstall))
    }

    @Test func anotherDirectoryDoesNotMatch() throws {
        let tree = try makeTree()
        defer { try? FileManager.default.removeItem(at: tree.root) }
        let install = tree.alias.appendingPathComponent("ornith")
        let other = tree.alias.appendingPathComponent("qwen")
        try FileManager.default.createDirectory(at: install, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: other, withIntermediateDirectories: true)
        #expect(
            !InstallPathBinding.matches(
                InstallPathBinding.canonical(install), directory: other))
    }

    /// The binding still detects what it exists for: a directory that moved.
    @Test func aMovedDirectoryDoesNotMatch() throws {
        let tree = try makeTree()
        defer { try? FileManager.default.removeItem(at: tree.root) }
        let before = tree.alias.appendingPathComponent("ornith")
        try FileManager.default.createDirectory(at: before, withIntermediateDirectories: true)
        let recorded = InstallPathBinding.canonical(before)
        let after = tree.alias.appendingPathComponent("ornith-moved")
        try FileManager.default.moveItem(at: before, to: after)
        #expect(!InstallPathBinding.matches(recorded, directory: after))
    }
}
