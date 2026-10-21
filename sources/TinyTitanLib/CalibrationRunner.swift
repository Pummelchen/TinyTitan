import Foundation

/// One candidate tuning change for a calibration run.
public struct CalibrationArm: Equatable, Sendable {
    public let name: String
    public let knobs: CalibrationProfile.Knobs
    public let note: String?

    public init(
        name: String, knobs: CalibrationProfile.Knobs,
        note: String? = nil
    ) {
        self.name = name
        self.knobs = knobs
        self.note = note
    }
}

/// What the keep rule decided for one arm.
public struct CalibrationDecision: Equatable, Sendable {
    public let arm: CalibrationArm
    public let baselineMedian: Double
    public let armMedian: Double
    /// Fractional change against the baseline median (0.03 = +3%).
    public let improvement: Double
    /// The larger of the 3% floor and the measured noise band, as a fraction.
    public let threshold: Double
    public let kept: Bool
    public let reason: String
}

/// Result of a calibration run: the samples it decided on and one decision per
/// arm. The measured profile is produced from the kept arms only.
public struct CalibrationReport: Equatable, Sendable {
    public let baseline: CalibrationArm
    public let baselineMedian: Double
    public let decisions: [CalibrationDecision]

    public var keptArms: [CalibrationArm] {
        decisions.filter(\.kept).map(\.arm)
    }

    /// Merges the kept arms onto the baseline knobs in decision order. A later
    /// kept arm wins when two arms set the same knob.
    public func measuredProfile(
        machine: CalibrationProfile.Machine,
        target: CalibrationProfile.Target,
        measuredAt: Date = Date(),
        source: String = "calibrate/1"
    ) -> CalibrationProfile {
        var knobs = baseline.knobs
        for arm in keptArms {
            Self.merge(arm.knobs, into: &knobs)
        }
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return CalibrationProfile(
            machine: machine, target: target, knobs: knobs,
            measured: true, measuredAt: formatter.string(from: measuredAt),
            source: source)
    }

    static func merge(
        _ from: CalibrationProfile.Knobs,
        into to: inout CalibrationProfile.Knobs
    ) {
        if let value = from.prefillChunkTokens { to.prefillChunkTokens = value }
        if let value = from.expertCacheSlots { to.expertCacheSlots = value }
        if let value = from.q8Attention { to.q8Attention = value }
        if let value = from.readerThreads { to.readerThreads = value }
        if let value = from.boundedIO { to.boundedIO = value }
        if let value = from.prefillTileDepth { to.prefillTileDepth = value }
    }
}

/// The 3% keep rule with a noise guard.
///
/// A win is kept only when the arm's median beats the baseline by more than
/// both the 3% floor and three median-absolute-deviations of the samples.
/// That is what keeps a lucky run from turning into a profile: the band is
/// measured on this machine, not assumed.
public struct CalibrationKeepRule: Equatable, Sendable {
    public static let minimumImprovement = 0.03
    /// Multiplier on the median absolute deviation that forms the noise band.
    public let noiseMultiplier: Double

    public init(noiseMultiplier: Double = 3.0) {
        self.noiseMultiplier = noiseMultiplier
    }

    public static func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let middle = sorted.count / 2
        if sorted.count % 2 == 0 {
            return (sorted[middle - 1] + sorted[middle]) / 2
        }
        return sorted[middle]
    }

    /// Median absolute deviation: the robust spread of a sample set.
    public static func medianAbsoluteDeviation(_ values: [Double]) -> Double {
        let centre = median(values)
        return median(values.map { abs($0 - centre) })
    }

    public func decide(
        baselineSamples: [Double],
        armSamples: [Double],
        arm: CalibrationArm
    ) -> CalibrationDecision {
        let baselineMedian = Self.median(baselineSamples)
        let armMedian = Self.median(armSamples)
        guard baselineMedian > 0, armMedian > 0 else {
            return CalibrationDecision(
                arm: arm, baselineMedian: baselineMedian,
                armMedian: armMedian, improvement: 0,
                threshold: Self.minimumImprovement, kept: false,
                reason: "rejected: samples must be positive tok/s")
        }
        let improvement = (armMedian - baselineMedian) / baselineMedian
        let baselineNoise =
            noiseMultiplier * Self.medianAbsoluteDeviation(baselineSamples)
            / baselineMedian
        let armNoise =
            noiseMultiplier * Self.medianAbsoluteDeviation(armSamples)
            / baselineMedian
        let threshold = max(Self.minimumImprovement, max(baselineNoise, armNoise))
        let kept = improvement > threshold
        let percent = String(format: "%.1f", improvement * 100)
        let band = String(format: "%.1f", threshold * 100)
        let reason: String
        if kept {
            reason = "kept: +\(percent)% above the \(band)% band"
        } else if improvement <= 0 {
            reason = "rejected: \(percent)% is not an improvement"
        } else if improvement <= Self.minimumImprovement {
            reason = "rejected: +\(percent)% is under the 3% floor"
        } else {
            reason = "rejected: +\(percent)% is inside the \(band)% noise band"
        }
        return CalibrationDecision(
            arm: arm, baselineMedian: baselineMedian, armMedian: armMedian,
            improvement: improvement, threshold: threshold, kept: kept,
            reason: reason)
    }
}

/// Runs the baseline and every arm once per repetition, interleaved so that a
/// drift in machine load affects every arm equally, then applies the keep rule.
///
/// The runner does not time anything itself: the caller supplies a measurement
/// closure that returns generated tokens per second for one arm. That keeps
/// the decision logic deterministic and testable without wall-clock luck.
public struct CalibrationRunner: Sendable {
    public typealias Measure = @Sendable (CalibrationArm) async throws -> Double

    public let rule: CalibrationKeepRule

    public init(rule: CalibrationKeepRule = CalibrationKeepRule()) {
        self.rule = rule
    }

    public enum RunError: Error, CustomStringConvertible {
        case tooFewRepetitions(Int)
        case noArms

        public var description: String {
            switch self {
            case .tooFewRepetitions(let count):
                return "calibration needs at least 2 repetitions, got \(count)"
            case .noArms:
                return "calibration has no candidate arms"
            }
        }
    }

    public func run(
        baseline: CalibrationArm,
        arms: [CalibrationArm],
        repeats: Int,
        measure: Measure
    ) async throws -> CalibrationReport {
        guard repeats >= 2 else {
            throw RunError.tooFewRepetitions(repeats)
        }
        guard !arms.isEmpty else { throw RunError.noArms }
        var baselineSamples: [Double] = []
        var armSamples = [[Double]](
            repeating: [], count: arms.count)
        for _ in 0..<repeats {
            baselineSamples.append(try await measure(baseline))
            for index in arms.indices {
                armSamples[index].append(try await measure(arms[index]))
            }
        }
        let decisions = arms.indices.map { index in
            rule.decide(
                baselineSamples: baselineSamples,
                armSamples: armSamples[index],
                arm: arms[index])
        }
        return CalibrationReport(
            baseline: baseline,
            baselineMedian: CalibrationKeepRule.median(baselineSamples),
            decisions: decisions)
    }
}

/// The arms the tree can actually switch today, and why the ones the plan
/// listed but the runtime does not expose are not arms yet.
public enum CalibrationPlan {
    public static func baselineArm() -> CalibrationArm {
        CalibrationArm(name: "baseline", knobs: CalibrationProfile.Knobs())
    }

    public static func defaultArms() -> [CalibrationArm] {
        [
            CalibrationArm(
                name: "prefill-chunk-128",
                knobs: CalibrationProfile.Knobs(prefillChunkTokens: 128),
                note: "Shorter prefill chunks trade dispatch overhead for latency."),
            CalibrationArm(
                name: "cache-slots-96",
                knobs: CalibrationProfile.Knobs(expertCacheSlots: 96),
                note: "More routed-expert slots raise the hit rate but use RAM."),
            CalibrationArm(
                name: "bounded-io-off",
                knobs: CalibrationProfile.Knobs(boundedIO: false),
                note: "TINYTITAN_BOUNDED_IO=0; a startup switch."),
        ]
    }

    /// Candidate arms the plan named that have no operator switch today.
    public static let unavailable: [(name: String, reason: String)] = [
        (
            "prefillTileDepth",
            "the prefill kernels pick their tile from head geometry; there is no runtime depth knob"
        ),
        (
            "q8Attention",
            "the q8 attention pipeline is selected from the thread width, not an option"
        ),
        (
            "readerThreads",
            "the C expert reader's thread count is fixed at open; it is not configurable"
        ),
    ]
}
