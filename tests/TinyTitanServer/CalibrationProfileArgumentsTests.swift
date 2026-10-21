import Foundation
import Testing
import TinyTitanLib

@testable import TinyTitanServerCore

@Suite struct CalibrationProfileArgumentsTests {
    private func profile(
        knobs: CalibrationProfile.Knobs
    ) throws -> URL {
        let profile = CalibrationProfile(
            machine: CalibrationProfile.currentMachine(),
            target: .init(
                modelID: "ornith", family: "qwen36", weightBits: 4,
                contextTokens: 8192),
            knobs: knobs,
            measured: true, measuredAt: "2026-10-22T00:00:00Z",
            source: "calibrate/1")
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("server-calibration-\(UUID().uuidString).json")
        try profile.encodedJSON().write(to: url)
        return url
    }

    @Test func profileFillsKnobsTheFlagsLeftUnset() throws {
        let url = try profile(
            knobs: .init(prefillChunkTokens: 128, expertCacheSlots: 96))
        defer { try? FileManager.default.removeItem(at: url) }
        let arguments = try ServerArguments.parse([
            "--model", "/tmp/model", "--max-context", "8192",
            "--calibration-profile", url.path,
        ])
        #expect(arguments.prefillChunkTokens == 128)
        #expect(arguments.expertCacheSlots == 96)
        #expect(arguments.calibrationProfile == url.path)
        #expect(arguments.calibrationLogLine?.contains("prefillChunkTokens=128") == true)
        #expect(arguments.calibrationLogLine?.contains("expertCacheSlots=96") == true)
    }

    @Test func explicitFlagsWinOverTheProfile() throws {
        let url = try profile(
            knobs: .init(prefillChunkTokens: 128, expertCacheSlots: 96))
        defer { try? FileManager.default.removeItem(at: url) }
        let arguments = try ServerArguments.parse([
            "--model", "/tmp/model", "--max-context", "8192",
            "--calibration-profile", url.path,
            "--prefill-chunk", "256", "--expert-cache-slots", "32",
        ])
        #expect(arguments.prefillChunkTokens == 256)
        #expect(arguments.expertCacheSlots == 32)
    }

    @Test func switchlessKnobsAreReportedNotApplied() throws {
        let url = try profile(knobs: .init(boundedIO: false))
        defer { try? FileManager.default.removeItem(at: url) }
        let arguments = try ServerArguments.parse([
            "--model", "/tmp/model", "--max-context", "8192",
            "--calibration-profile", url.path,
        ])
        #expect(arguments.prefillChunkTokens == nil)
        #expect(arguments.expertCacheSlots == nil)
        #expect(arguments.calibrationLogLine?.contains("not applied: boundedIO") == true)
    }

    @Test func foreignMachineProfileIsRefused() throws {
        var machine = CalibrationProfile.currentMachine()
        machine = .init(
            identifier: "Mac99,9", memoryBytes: machine.memoryBytes,
            osVersion: machine.osVersion)
        let profile = CalibrationProfile(
            machine: machine,
            target: .init(
                modelID: "ornith", family: "qwen36", weightBits: 4,
                contextTokens: 8192),
            measured: true, measuredAt: "2026-10-22T00:00:00Z",
            source: "calibrate/1")
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("server-calibration-foreign-\(UUID().uuidString).json")
        try profile.encodedJSON().write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse([
                "--model", "/tmp/model", "--max-context", "8192",
                "--calibration-profile", url.path,
            ])
        }
    }

    @Test func noProfileLeavesTheFlagsAlone() throws {
        let arguments = try ServerArguments.parse([
            "--model", "/tmp/model", "--max-context", "8192",
        ])
        #expect(arguments.calibrationProfile == nil)
        #expect(arguments.calibrationLogLine == nil)
        #expect(arguments.prefillChunkTokens == nil)
        #expect(arguments.expertCacheSlots == nil)
    }

    @Test func corruptProfileIsRefused() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("server-calibration-bad-\(UUID().uuidString).json")
        try Data(#"{"schemaVersion":99}"#.utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse([
                "--model", "/tmp/model", "--calibration-profile", url.path,
            ])
        }
    }
}
