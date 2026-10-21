import Foundation
import Testing
import TinyTitanCLICore
import TinyTitanLib

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

    @Test func draftWritesAValidUnmeasuredProfile() throws {
        let directory = try QwenToySynthetic.write(weightBits: 4)
        defer { try? FileManager.default.removeItem(at: directory) }
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("calibration-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: out) }

        let code = try CalibrationCommand.run(
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

    @Test func conflictingModesAreRefused() {
        #expect(throws: CalibrationCommandError.self) {
            _ = try CalibrationCommand.run(
                CalibrationOptions(model: "/tmp/m", check: "/tmp/p"))
        }
        #expect(throws: CalibrationCommandError.self) {
            _ = try CalibrationCommand.run(CalibrationOptions())
        }
    }
}
