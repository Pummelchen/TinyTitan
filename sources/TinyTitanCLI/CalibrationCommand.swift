import Foundation
import TinyTitan
import TinyTitanLib

/// `tinytitan calibrate`: write or validate a `CalibrationProfile`.
///
/// T15 is the interface and the schema; the measurement arms are T16. Until
/// they land the writer produces an unmeasured draft (`measured: false`,
/// every knob nil) from the install's identity, so the file, the validator and
/// the help text exist and a later calibration run only fills the knobs.
public struct CalibrationOptions: Equatable, Sendable {
    public var model: String?
    public var maxContext: Int = 4096
    public var out: String?
    public var check: String?
    public var help: Bool = false

    public init(
        model: String? = nil, maxContext: Int = 4096,
        out: String? = nil, check: String? = nil, help: Bool = false
    ) {
        self.model = model
        self.maxContext = maxContext
        self.out = out
        self.check = check
        self.help = help
    }

    public static let usage = """
        usage: tinytitan calibrate (--model <dir> [--max-context <tokens>] [--out <path>]
                                   | --check <path>)

        options:
          --model <dir>          Path to a .ssdai model directory. Writes an
                                 unmeasured profile draft for this machine,
                                 install and context.
          --max-context <tokens> Context the profile is for; one of 4096, 8192,
                                 16384, 32768, 65536, 131072, 262144.
                                 Default 4096.
          --out <path>           Write the profile here instead of stdout.
          --check <path>         Decode and validate an existing profile, then
                                 print it normalized.
          --help                 Show this help.

        The profile is JSON. A draft has "measured": false and no knobs; a
        calibration run (T16) fills the knobs and sets "measured": true.
        """

    public static func parse(_ arguments: [String]) throws -> CalibrationOptions {
        var options = CalibrationOptions()
        var index = 0
        while index < arguments.count {
            let argument = arguments[index]
            func value(_ name: String) throws -> String {
                index += 1
                guard index < arguments.count else {
                    throw CalibrationCommandError.missingValue(name)
                }
                return arguments[index]
            }
            switch argument {
            case "--help":
                options.help = true
            case "--model":
                options.model = try value("--model")
            case "--out":
                options.out = try value("--out")
            case "--check":
                options.check = try value("--check")
            case "--max-context":
                let raw = try value("--max-context")
                guard let parsed = Int(raw), parsed > 0 else {
                    throw CalibrationCommandError.invalidValue(
                        "--max-context", raw)
                }
                options.maxContext = parsed
            default:
                throw CalibrationCommandError.unknownArgument(argument)
            }
            index += 1
        }
        return options
    }
}

public enum CalibrationCommandError: Error, CustomStringConvertible {
    case unknownArgument(String)
    case missingValue(String)
    case invalidValue(String, String)
    case missingModel
    case conflictingInput
    case profileTooLarge(Int)

    public var description: String {
        switch self {
        case .unknownArgument(let value):
            return "unknown argument \(value)"
        case .missingValue(let name):
            return "\(name) needs a value"
        case .invalidValue(let name, let value):
            return "\(name) got an invalid value \(value)"
        case .missingModel:
            return "calibrate needs --model <dir> or --check <path>"
        case .conflictingInput:
            return "calibrate accepts either --model or --check, not both"
        case .profileTooLarge(let cap):
            return "calibration profile exceeds \(cap) bytes"
        }
    }
}

public enum CalibrationCommand {
    /// Profiles are small metadata documents; bound the read so a stray path
    /// cannot pull a model payload into memory.
    private static func readProfile(at path: String) throws -> Data {
        let cap = 1 << 20
        let handle = try FileHandle(forReadingFrom: URL(fileURLWithPath: path))
        defer { try? handle.close() }
        let data = try handle.read(upToCount: cap + 1) ?? Data()
        guard data.count <= cap else {
            throw CalibrationCommandError.profileTooLarge(cap)
        }
        return data
    }

    private static func writeStdout(_ data: Data) {
        var line = data
        line.append(0x0A)
        FileHandle.standardOutput.write(line)
    }

    /// Runs the subcommand and returns the process exit code.
    public static func run(_ options: CalibrationOptions) throws -> Int32 {
        if options.help {
            print(CalibrationOptions.usage)
            return 0
        }
        if options.check != nil, options.model != nil {
            throw CalibrationCommandError.conflictingInput
        }
        if let path = options.check {
            let data = try readProfile(at: path)
            let profile = try CalibrationProfile.decode(from: data)
            writeStdout(try profile.encodedJSON())
            return 0
        }
        guard let modelPath = options.model else {
            throw CalibrationCommandError.missingModel
        }
        let directory = URL(fileURLWithPath: modelPath)
        let identity = try ManifestReader.peekIdentity(directoryURL: directory)
        let profile = CalibrationProfile(
            machine: CalibrationProfile.currentMachine(),
            target: CalibrationProfile.Target(
                modelID: identity.modelID,
                family: identity.family.rawValue,
                weightBits: identity.weightBits,
                contextTokens: options.maxContext),
            source: "unmeasured")
        try profile.validate()
        let data = try profile.encodedJSON()
        if let out = options.out {
            try data.write(to: URL(fileURLWithPath: out))
            FileHandle.standardError.write(
                Data("wrote \(out)\n".utf8))
        } else {
            writeStdout(data)
        }
        return 0
    }
}
