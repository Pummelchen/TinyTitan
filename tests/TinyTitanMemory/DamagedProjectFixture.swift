import ContinuityCore
import Foundation

@testable import TinyTitanMemory

/// A scratch project file with a fact that survives a damaged line.
///
/// The journal is written by the real engine (so the record shapes are the
/// ones a restart would see), then one complete, undecodable line is spliced
/// in after the first record. `damaged: false` leaves it clean, for the
/// counter-test.
struct DamagedProjectFixture {
    let directory: URL
    let journal: URL
    /// The value of the fact written before the damage; find its key with
    /// `MemoryProjectFile` rather than assuming the address form.
    let factValue = "still readable"

    init(workspace: String = "damaged-project", damaged: Bool = true) async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("memory-damage-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("\(workspace).ndjson")
        let journal = try FileJournal(url: url, idleDelay: .seconds(60))
        let engine = ContinuityEngine(journal: journal)
        try await engine.start()
        let task = try await engine.createTask(title: "damaged workspace")
        let session = try await engine.beginSession(taskID: task.id)
        try await engine.remember(
            sessionID: session.id, namespace: "notes", key: "kept", value: factValue)
        await engine.shutDown()

        if damaged {
            let raw = try Data(contentsOf: url)
            let lines = raw.split(separator: 0x0A, omittingEmptySubsequences: true)
            guard lines.count > 1 else {
                throw JournalError.readFailed(url, errno: EIO)
            }
            var rebuilt = Data()
            rebuilt.append(contentsOf: lines[0])
            rebuilt.append(0x0A)
            rebuilt.append(Data(#"{"somethingElse":{"version":9}}"#.utf8))
            rebuilt.append(0x0A)
            for line in lines.dropFirst() {
                rebuilt.append(contentsOf: line)
                rebuilt.append(0x0A)
            }
            try rebuilt.write(to: url)
            try FileManager.default.setAttributes(
                [.posixPermissions: 0o600], ofItemAtPath: url.path)
        }
        self.directory = directory
        self.journal = url
    }

    /// The line number the spliced record occupies, which is what a caller is
    /// told about.
    static let damagedLine = 2
}
