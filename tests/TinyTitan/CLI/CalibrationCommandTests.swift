import Foundation
import Testing
import TinyTitanLib

@testable import TinyTitanCLICore

@testable import TinyTitan

@Suite struct CalibrationCommandTests {
    @Test func parsesDraftArguments() throws {
        let options = try CalibrationOptions.parse([
            "--model", "/tmp/model", "--max-context", "32768",
            "--out", "/tmp/profile.json",
        ])
        #expect(options.model == "/tmp/model")
        #expect(options.maxContext == 32_768)
        #expect(options.out == "/tmp/profile.json")
        #expect(options.check == nil)
    }

    @Test func parsesMeasureArguments() throws {
        let options = try CalibrationOptions.parse([
            "--measure", "--model", "/tmp/model", "--tokens", "32",
            "--repeats", "2",
        ])
        #expect(options.measure)
        #expect(options.tokens == 32)
        #expect(options.repeats == 2)
    }

    @Test func badMeasureArgumentsAreRefused() {
        #expect(throws: CalibrationCommandError.self) {
            try CalibrationOptions.parse(["--tokens", "4"])
        }
        #expect(throws: CalibrationCommandError.self) {
            try CalibrationOptions.parse(["--repeats", "1"])
        }
    }

    @Test func parsesCheckAndHelp() throws {
        let check = try CalibrationOptions.parse(["--check", "/tmp/p.json"])
        #expect(check.check == "/tmp/p.json")
        let help = try CalibrationOptions.parse(["--help"])
        #expect(help.help)
    }

    @Test func badArgumentsAreRefused() {
        #expect(throws: CalibrationCommandError.self) {
            try CalibrationOptions.parse(["--nope"])
        }
        #expect(throws: CalibrationCommandError.self) {
            try CalibrationOptions.parse(["--model"])
        }
        #expect(throws: CalibrationCommandError.self) {
            try CalibrationOptions.parse(["--max-context", "0"])
        }
    }

    @Test func draftWritesAValidUnmeasuredProfile() async throws {
        let directory = try QwenToySynthetic.write(weightBits: 4)
        defer { try? FileManager.default.removeItem(at: directory) }
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("calibration-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: out) }

        let code = try await CalibrationCommand.run(
            CalibrationOptions(
                model: directory.path, maxContext: 8192, out: out.path))
        #expect(code == 0)

        let data = try Data(contentsOf: out)
        let profile = try CalibrationProfile.decode(from: data)
        #expect(profile.measured == false)
        #expect(profile.source == "unmeasured")
        #expect(!profile.target.modelID.isEmpty)
        #expect(profile.target.family == ModelFamily.qwen36.rawValue)
        #expect(profile.target.weightBits == 4)
        #expect(profile.target.contextTokens == 8192)
    }

    @Test func measurementReportKeepsOnlyMeasuredWinners() async throws {
        let target = CalibrationProfile.Target(
            modelID: "ornith", family: "qwen36", weightBits: 4,
            contextTokens: 8192)
        let report = try await CalibrationCommand.measurementReport(
            target: target, repeats: 2
        ) { arm in
            switch arm.name {
            case "prefill-chunk-128": return 120
            case "cache-slots-96": return 101
            default: return 100
            }
        }
        #expect(report.keptArms.map(\.name) == ["prefill-chunk-128"])
        let profile = report.measuredProfile(
            machine: .init(
                identifier: "Mac15,3", memoryBytes: 24 << 30,
                osVersion: "27.0.1"),
            target: target)
        #expect(profile.measured)
        #expect(profile.source == "calibrate/1")
        #expect(profile.knobs.prefillChunkTokens == 128)
        #expect(profile.knobs.expertCacheSlots == nil)
        try profile.validate()
    }

    @Test func conflictingModesAreRefused() async {
        await #expect(throws: CalibrationCommandError.self) {
            _ = try await CalibrationCommand.run(
                CalibrationOptions(model: "/tmp/m", check: "/tmp/p"))
        }
        await #expect(throws: CalibrationCommandError.self) {
            _ = try await CalibrationCommand.run(CalibrationOptions())
        }
    }

    private func writeProfile(
        _ profile: CalibrationProfile
    ) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("calibration-\(UUID().uuidString).json")
        try profile.encodedJSON().write(to: url)
        return url
    }

    @Test func profileFillsOnlyUnsetKnobs() throws {
        let profile = CalibrationProfile(
            machine: .init(
                identifier: "Mac15,3", memoryBytes: 24 << 30,
                osVersion: "27.0.1"),
            target: .init(
                modelID: "ornith", family: "qwen36", weightBits: 4,
                contextTokens: 8192),
            knobs: .init(prefillChunkTokens: 128, expertCacheSlots: 96),
            measured: true, measuredAt: "2026-10-22T00:00:00Z",
            source: "calibrate/1")
        let url = try writeProfile(profile)
        defer { try? FileManager.default.removeItem(at: url) }

        let parsed = try Args.parse([
            "--model", "/tmp/model", "--prompt", "hi",
            "--calibration-profile", url.path,
        ])
        #expect(parsed.calibrationProfile == url.path)
        #expect(parsed.expertCacheSlots == 96)
        if case .fixed(let chunk)? = parsed.prefillChunk {
            #expect(chunk == 128)
        } else {
            Issue.record("profile chunk was not applied")
        }
        #expect(parsed.calibrationLogLine?.contains("prefillChunkTokens=128") == true)
        #expect(parsed.calibrationLogLine?.contains("expertCacheSlots=96") == true)
    }

    @Test func explicitFlagsWinOverTheProfile() throws {
        let profile = CalibrationProfile(
            machine: .init(
                identifier: "Mac15,3", memoryBytes: 24 << 30,
                osVersion: "27.0.1"),
            target: .init(
                modelID: "ornith", family: "qwen36", weightBits: 4,
                contextTokens: 8192),
            knobs: .init(prefillChunkTokens: 128, expertCacheSlots: 96),
            measured: true, measuredAt: "2026-10-22T00:00:00Z",
            source: "calibrate/1")
        let url = try writeProfile(profile)
        defer { try? FileManager.default.removeItem(at: url) }

        let parsed = try Args.parse([
            "--model", "/tmp/model", "--prompt", "hi",
            "--calibration-profile", url.path,
            "--expert-cache-slots", "32", "--prefill-chunk", "256",
        ])
        #expect(parsed.expertCacheSlots == 32)
        if case .fixed(let chunk)? = parsed.prefillChunk {
            #expect(chunk == 256)
        } else {
            Issue.record("explicit chunk was not kept")
        }
    }

    @Test func profileKnobsWithoutSwitchesAreReported() throws {
        let profile = CalibrationProfile(
            machine: .init(
                identifier: "Mac15,3", memoryBytes: 24 << 30,
                osVersion: "27.0.1"),
            target: .init(
                modelID: "ornith", family: "qwen36", weightBits: 4,
                contextTokens: 8192),
            knobs: .init(boundedIO: false),
            measured: true, measuredAt: "2026-10-22T00:00:00Z",
            source: "calibrate/1")
        let url = try writeProfile(profile)
        defer { try? FileManager.default.removeItem(at: url) }
        let parsed = try Args.parse([
            "--model", "/tmp/model", "--prompt", "hi",
            "--calibration-profile", url.path,
        ])
        #expect(parsed.expertCacheSlots == nil)
        #expect(parsed.calibrationLogLine?.contains("not applied: boundedIO") == true)
    }

    @Test func corruptProfileIsRefused() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("calibration-bad-\(UUID().uuidString).json")
        try Data(#"{"schemaVersion":99}"#.utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: ArgsError.self) {
            try Args.parse([
                "--model", "/tmp/model", "--prompt", "hi",
                "--calibration-profile", url.path,
            ])
        }
    }
}
