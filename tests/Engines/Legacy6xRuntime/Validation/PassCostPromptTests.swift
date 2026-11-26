import Foundation
import Testing
import TinyTitanValidationSupport

/// A pass-cost measurement must use the prompt it says it used.
///
/// The old helper returned a three-line function whenever the source file could
/// not be read, so a moved file or a wrong working directory produced a
/// plausible-looking measurement on the wrong input. These pin the real read,
/// the 140-line cut and the loud failure.
@Suite struct PassCostPromptTests {
    private func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("tinytitan-pass-cost-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    @Test func aPresentSourceIsReadAndCutAt140Lines() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = directory.appendingPathComponent("Source.swift")
        let lines = (1...160).map { "line\($0)" }.joined(separator: "\n")
        try lines.write(to: source, atomically: true, encoding: .utf8)

        let prompt = try PassCostPrompt.text(promptFile: nil, source: source.path)

        #expect(prompt.hasPrefix("Below is a Swift source file.\n```swift\n"))
        #expect(prompt.contains("\nline1\n"))
        #expect(prompt.contains("\nline140\n"))
        #expect(!prompt.contains("\nline141\n"))
        #expect(prompt.hasSuffix("```swift\n"))
    }

    @Test func aSuppliedPromptIsUsedVerbatim() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let promptFile = directory.appendingPathComponent("prompt.txt")
        let body = "an operator-supplied prompt\nwith two lines"
        try body.write(to: promptFile, atomically: true, encoding: .utf8)

        let prompt = try PassCostPrompt.text(promptFile: promptFile.path)

        #expect(prompt.contains(body))
    }

    @Test func aMissingSourceThrowsInsteadOfSubstituting() {
        #expect(throws: PassCostPrompt.Failure.self) {
            try PassCostPrompt.text(
                promptFile: nil,
                source: "/definitely/not/a/tinytitan/source/StreamingMTP.swift")
        }
    }
}
