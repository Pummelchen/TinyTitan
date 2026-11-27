import Foundation
import Testing

@testable import TinyTitanRepackCore

/// The receipt records where the install physically is (AUD-0072).
///
/// A symlinked `models/` directory is how people keep installs on a second
/// disk. The payload is placed at the resolved path, so recording the typed
/// spelling produced a receipt the loader refused on a fresh, correct install.
@Suite struct ReceiptPathBindingTests {
    @Test func theReceiptRecordsThePhysicalPath() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("receipt-binding-\(UUID().uuidString)")
        let real = root.appendingPathComponent("volumes/models")
        try FileManager.default.createDirectory(at: real, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let alias = root.appendingPathComponent("models")
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: real)

        let typed = alias.appendingPathComponent("ornith")
        let data = try VerifiedInstallReceiptWriter.encode(
            outputDir: typed.path,
            manifestSha256: String(repeating: "ab", count: 32),
            manifestSize: 10,
            sourceRepoID: nil,
            sourceRevision: nil,
            files: [])
        let receipt = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let recorded = try #require(receipt["modelDirectoryPath"] as? String)
        let physical = real.appendingPathComponent("ornith").standardizedFileURL.path
        #expect(recorded == physical, "recorded \(recorded), expected \(physical)")
        #expect(recorded != typed.path, "the typed spelling must not be what is recorded")
    }
}
