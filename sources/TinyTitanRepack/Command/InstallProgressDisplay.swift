//
//  InstallProgressDisplay.swift
//  TinyTitanRepack
//
//  The live install progress line for the repack command.
//

import Darwin
import Foundation
import Synchronization
import TinyTitanRepackCore

/// Renders the repacker's `ModelInstallProgress` stream as a live display.
///
/// WHY: installing a shipped model copies, hashes and verifies 20-220 GB, and
/// the repacker used to print nothing until it finished -- minutes to hours of a
/// silent terminal with no way to tell a slow install from a hung one. The
/// progress stream already existed; this is its first consumer.
///
/// The renderer is deliberately plain text. On an interactive terminal it
/// rewrites one line in place with `\r`, at most once a second; on a pipe or a
/// log file it prints a new line only at each phase change and each 10% step,
/// so `TinyTitanRepack ... | tee install.log` still shows where the install is.
/// `TINYTITAN_NO_PROGRESS=1` prints nothing and `TERM=dumb` falls back to the
/// plain-line form. No ANSI codes are emitted anywhere, so `NO_COLOR` is
/// honoured by construction.
final class InstallProgressDisplay: Sendable {

    /// The mutable render state. It lives under `state`'s mutex rather than in
    /// the instance so the `@Sendable` progress closure can call `handle` from
    /// any thread without an unsynchronized race.
    private struct State: Sendable {
        var lastDraw: ContinuousClock.Instant?
        var lastBucket = -1
        var lastLineLength = 0
        var hasOpenLine = false
        var payloadStart: ContinuousClock.Instant?
        var reuseReported = false
    }

    private let state = Mutex(State())
    private let isInteractive: Bool
    private let isEnabled: Bool

    /// Reads the environment once: the display must not re-read it per event.
    init(environment: [String: String] = ProcessInfo.processInfo.environment) {
        self.isEnabled = !Self.disabled(environment)
        self.isInteractive =
            isatty(STDOUT_FILENO) == 1 && environment["TERM"] != "dumb"
    }

    /// The same truthy spellings the Python progress helper accepts, so one
    /// environment value silences the Swift and Python installers together.
    private static func disabled(_ environment: [String: String]) -> Bool {
        let value = (environment["TINYTITAN_NO_PROGRESS"] ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return ["1", "on", "true", "yes"].contains(value)
    }

    /// Consumes one progress event. Safe to call from any thread.
    func handle(_ progress: ModelInstallProgress) {
        guard isEnabled else { return }
        let now = ContinuousClock.now
        state.withLock { state in
            switch progress {
            case .downloadingMetadata:
                drawPhase("downloading metadata", state: &state)
            case .planning(let downloadBytes, let outputBytes):
                drawPhase(
                    "planning  \(Self.humanBytes(outputBytes)) install from a "
                        + "\(Self.humanBytes(downloadBytes)) download",
                    state: &state)
            case .checkingDisk:
                drawPhase("checking disk", state: &state)
            case .reservingOutput:
                drawPhase("reserving output", state: &state)
            case .hashingOutput(let name):
                drawPhase("hashing  \(name)", state: &state)
            case .finalizing:
                drawPhase("finalizing", state: &state)
            case .copyingPayload(let reusedBytes, let downloaded, let total):
                drawPayload(
                    reusedBytes: reusedBytes,
                    downloadedThisRunBytes: downloaded,
                    totalBytes: total,
                    now: now,
                    state: &state)
            }
        }
    }

    /// Draws a phase line. The line is always shown (a phase change is a new
    /// fact, not a redraw), and any open in-place line is finished first so the
    /// two never run together.
    private func drawPhase(_ line: String, state: inout State) {
        guard isInteractive else {
            write(line + "\n")
            return
        }
        if state.hasOpenLine {
            write("\n")
        }
        write(line + "\n")
        state.hasOpenLine = false
        state.lastLineLength = 0
    }

    /// Draws the payload line: rewritten in place on a terminal, or emitted
    /// only when the percentage enters a new decile otherwise.
    private func drawPayload(
        reusedBytes: UInt64,
        downloadedThisRunBytes: UInt64,
        totalBytes: UInt64,
        now: ContinuousClock.Instant,
        state: inout State
    ) {
        if reusedBytes > 0, !state.reuseReported {
            state.reuseReported = true
            drawPhase("reused \(Self.humanBytes(reusedBytes))", state: &state)
        }
        if state.payloadStart == nil {
            state.payloadStart = now
        }
        let sum = reusedBytes.addingReportingOverflow(downloadedThisRunBytes)
        let completed = sum.overflow ? UInt64.max : sum.partialValue
        let capped = min(completed, totalBytes)
        let percent = totalBytes > 0 ? Int(capped * 100 / totalBytes) : 100
        let line =
            "installing  \(Self.progressAmounts(completed: completed, total: totalBytes))"
            + "  \(percent)%"
            + etaSuffix(
                downloadedThisRunBytes: downloadedThisRunBytes,
                remainingBytes: totalBytes > capped ? totalBytes - capped : 0,
                now: now,
                state: state)
        if percent >= 100 {
            // The final line is drawn whatever the mode and is always finished
            // with a newline, so the install summary that follows starts clean.
            if isInteractive {
                drawInline(line, state: &state)
                finishOpenLine(state: &state)
            } else {
                write(line + "\n")
            }
            state.lastDraw = now
            return
        }
        if isInteractive {
            if let last = state.lastDraw, now - last < .seconds(1) { return }
            state.lastDraw = now
            drawInline(line, state: &state)
        } else {
            let bucket = percent / 10
            guard bucket > state.lastBucket else { return }
            state.lastBucket = bucket
            write(line + "\n")
        }
    }

    /// The `  eta 4m10s` segment, or an empty string until the observed rate is
    /// worth trusting: at least two seconds of copying and some new bytes.
    private func etaSuffix(
        downloadedThisRunBytes: UInt64,
        remainingBytes: UInt64,
        now: ContinuousClock.Instant,
        state: State
    ) -> String {
        guard let start = state.payloadStart,
            downloadedThisRunBytes > 0,
            remainingBytes > 0
        else { return "" }
        let elapsed = (now - start) / .seconds(1)
        guard elapsed >= 2 else { return "" }
        let bytesPerSecond = Double(downloadedThisRunBytes) / elapsed
        guard bytesPerSecond > 0 else { return "" }
        return "  eta \(Self.duration(Double(remainingBytes) / bytesPerSecond))"
    }

    /// Rewrites the in-place line, padding over a previously longer one so a
    /// shrinking name cannot leave trailing characters behind.
    private func drawInline(_ line: String, state: inout State) {
        let padding = max(0, state.lastLineLength - line.count)
        write("\r" + line + String(repeating: " ", count: padding))
        state.lastLineLength = line.count
        state.hasOpenLine = true
    }

    /// Ends an open in-place line so whatever is printed next starts clean.
    private func finishOpenLine(state: inout State) {
        guard isInteractive, state.hasOpenLine else { return }
        write("\n")
        state.hasOpenLine = false
        state.lastLineLength = 0
    }

    private func write(_ text: String) {
        FileHandle.standardOutput.write(Data(text.utf8))
    }

    /// The `<copied>/<total> <unit>` pair the progress line shows, with one
    /// unit chosen from the total so both numbers are directly comparable.
    private static func progressAmounts(completed: UInt64, total: UInt64) -> String {
        let gigabytes = total >= 1_000_000_000
        let divisor = gigabytes ? 1e9 : 1e6
        let unit = gigabytes ? "GB" : "MB"
        return String(format: "%.1f", Double(completed) / divisor) + "/"
            + String(format: "%.1f", Double(total) / divisor) + " " + unit
    }

    /// Decimal units (1 GB = 1e9 bytes), which is what the installers' size
    /// claims and the model cards use; gigabytes keep one decimal, smaller
    /// amounts are shown in megabytes.
    private static func humanBytes(_ bytes: UInt64) -> String {
        let gigabytes = Double(bytes) / 1e9
        if gigabytes >= 1 {
            return String(format: "%.1f GB", gigabytes)
        }
        return String(format: "%.1f MB", Double(bytes) / 1e6)
    }

    /// A compact duration: `42s`, `4m10s`, `1h02m`.
    private static func duration(_ seconds: Double) -> String {
        let total = Int(seconds.rounded())
        if total < 60 { return "\(total)s" }
        if total < 3600 { return "\(total / 60)m\(total % 60)s" }
        return "\(total / 3600)h" + String(format: "%02d", (total % 3600) / 60) + "m"
    }
}
