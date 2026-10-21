import Darwin
import Foundation
import TinyTitan

/// A measured tuning profile for one (machine, model, context) triple.
///
/// This is the T15 schema only: it records what a calibration run decided and
/// validates it, so a profile can be written, read back and refused when it
/// does not describe this machine or this install. The measurement arms that
/// produce a `measured: true` profile are T16; until then the CLI writes an
/// unmeasured draft (`measured: false`, `source: "unmeasured"`) from the
/// install's identity.
public struct CalibrationProfile: Codable, Equatable, Sendable {
    public static let currentSchemaVersion = 1

    public struct Machine: Codable, Equatable, Sendable {
        /// `hw.model`, e.g. `Mac14,5`. Not the host name.
        public let identifier: String
        public let memoryBytes: UInt64
        public let osVersion: String

        public init(identifier: String, memoryBytes: UInt64, osVersion: String) {
            self.identifier = identifier
            self.memoryBytes = memoryBytes
            self.osVersion = osVersion
        }
    }

    public struct Target: Codable, Equatable, Sendable {
        public let modelID: String
        public let family: String
        public let weightBits: Int
        public let contextTokens: Int

        public init(
            modelID: String, family: String, weightBits: Int,
            contextTokens: Int
        ) {
            self.modelID = modelID
            self.family = family
            self.weightBits = weightBits
            self.contextTokens = contextTokens
        }
    }

    /// The knobs a calibration run may decide. Nil means "not measured for
    /// this profile", which is what makes an unmeasured draft distinguishable
    /// from a measured zero/false.
    public struct Knobs: Codable, Equatable, Sendable {
        public var prefillChunkTokens: Int?
        public var expertCacheSlots: Int?
        public var q8Attention: Bool?
        public var readerThreads: Int?
        public var boundedIO: Bool?
        public var prefillTileDepth: Int?

        public init(
            prefillChunkTokens: Int? = nil,
            expertCacheSlots: Int? = nil,
            q8Attention: Bool? = nil,
            readerThreads: Int? = nil,
            boundedIO: Bool? = nil,
            prefillTileDepth: Int? = nil
        ) {
            self.prefillChunkTokens = prefillChunkTokens
            self.expertCacheSlots = expertCacheSlots
            self.q8Attention = q8Attention
            self.readerThreads = readerThreads
            self.boundedIO = boundedIO
            self.prefillTileDepth = prefillTileDepth
        }
    }

    public let schemaVersion: Int
    public let machine: Machine
    public let target: Target
    public let knobs: Knobs
    /// True only when a calibration run measured `knobs` on this machine; a
    /// draft keeps every knob nil and stays false.
    public let measured: Bool
    /// ISO-8601 instant, present only when `measured`.
    public let measuredAt: String?
    /// What produced the profile, e.g. `unmeasured` or `calibrate/1`.
    public let source: String

    public init(
        machine: Machine,
        target: Target,
        knobs: Knobs = Knobs(),
        measured: Bool = false,
        measuredAt: String? = nil,
        source: String = "unmeasured",
        schemaVersion: Int = CalibrationProfile.currentSchemaVersion
    ) {
        self.schemaVersion = schemaVersion
        self.machine = machine
        self.target = target
        self.knobs = knobs
        self.measured = measured
        self.measuredAt = measuredAt
        self.source = source
    }

    /// The machine this process is running on, as the profile records it.
    public static func currentMachine() -> Machine {
        var size = 0
        var identifier = ""
        if sysctlbyname("hw.model", nil, &size, nil, 0) == 0, size > 0 {
            var bytes = [CChar](repeating: 0, count: size)
            if sysctlbyname("hw.model", &bytes, &size, nil, 0) == 0 {
                let scalars = bytes.prefix { $0 != 0 }.map {
                    UInt8(bitPattern: $0)
                }
                identifier = String(bytes: scalars, encoding: .utf8) ?? "unknown"
            }
        }
        let version = ProcessInfo.processInfo.operatingSystemVersion
        return Machine(
            identifier: identifier.isEmpty ? "unknown" : identifier,
            memoryBytes: ProcessInfo.processInfo.physicalMemory,
            osVersion: "\(version.majorVersion).\(version.minorVersion).\(version.patchVersion)")
    }

    /// Rejects a profile that cannot describe a run this engine supports.
    public func validate() throws {
        guard schemaVersion == Self.currentSchemaVersion else {
            throw CalibrationProfileError.unsupportedSchema(schemaVersion)
        }
        guard !machine.identifier.isEmpty else {
            throw CalibrationProfileError.invalidMachine(
                "machine identifier is empty")
        }
        guard machine.memoryBytes > 0 else {
            throw CalibrationProfileError.invalidMachine(
                "machine memoryBytes is zero")
        }
        guard !target.modelID.isEmpty else {
            throw CalibrationProfileError.invalidTarget(
                "modelID is empty")
        }
        guard !target.family.isEmpty else {
            throw CalibrationProfileError.invalidTarget("family is empty")
        }
        guard target.weightBits == 4 || target.weightBits == 8 else {
            throw CalibrationProfileError.invalidTarget(
                "weightBits \(target.weightBits) is not 4 or 8")
        }
        guard
            RuntimeConfiguration.supportedContextTokens.contains(
                target.contextTokens)
        else {
            throw CalibrationProfileError.invalidTarget(
                "contextTokens \(target.contextTokens) is not a supported context")
        }
        if let chunk = knobs.prefillChunkTokens,
            !RuntimeConfiguration.allowedPrefillChunkTokens.contains(chunk)
        {
            throw CalibrationProfileError.invalidKnob(
                "prefillChunkTokens \(chunk) is not an allowed chunk")
        }
        if let slots = knobs.expertCacheSlots,
            !RuntimeConfiguration.allowedExpertCacheSlots.contains(slots)
        {
            throw CalibrationProfileError.invalidKnob(
                "expertCacheSlots \(slots) is not an allowed slot count")
        }
        if let threads = knobs.readerThreads, !(1...64).contains(threads) {
            throw CalibrationProfileError.invalidKnob(
                "readerThreads \(threads) is outside 1...64")
        }
        if let depth = knobs.prefillTileDepth, !(1...32).contains(depth) {
            throw CalibrationProfileError.invalidKnob(
                "prefillTileDepth \(depth) is outside 1...32")
        }
        if let measuredAt {
            guard CalibrationProfile.iso8601(measuredAt) != nil else {
                throw CalibrationProfileError.invalidTimestamp(measuredAt)
            }
        }
        guard !source.isEmpty else {
            throw CalibrationProfileError.invalidSource
        }
    }

    public func encodedJSON() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(self)
    }

    public static func decode(from data: Data) throws -> CalibrationProfile {
        let profile: CalibrationProfile
        do {
            profile = try JSONDecoder().decode(CalibrationProfile.self, from: data)
        } catch {
            throw CalibrationProfileError.malformed(String(describing: error))
        }
        try profile.validate()
        return profile
    }

    static func iso8601(_ value: String) -> Date? {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter.date(from: value)
    }
}

public enum CalibrationProfileError: Error, CustomStringConvertible {
    case unsupportedSchema(Int)
    case invalidMachine(String)
    case invalidTarget(String)
    case invalidKnob(String)
    case invalidTimestamp(String)
    case invalidSource
    case malformed(String)

    public var description: String {
        switch self {
        case .unsupportedSchema(let version):
            return "calibration schema version \(version) is not supported"
        case .invalidMachine(let detail):
            return "invalid calibration machine: \(detail)"
        case .invalidTarget(let detail):
            return "invalid calibration target: \(detail)"
        case .invalidKnob(let detail):
            return "invalid calibration knob: \(detail)"
        case .invalidTimestamp(let value):
            return "invalid calibration timestamp: \(value)"
        case .invalidSource:
            return "calibration source is empty"
        case .malformed(let detail):
            return "malformed calibration profile: \(detail)"
        }
    }
}
