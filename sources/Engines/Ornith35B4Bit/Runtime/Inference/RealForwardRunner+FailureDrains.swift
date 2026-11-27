import Foundation

// A failed decode's bookkeeping, and the seam that makes its failure path
// reachable (AUD-0089).
//
// The decode loop commits a routed command and carries it forward; a throw in
// between used to drop the only reference, releasing the expert lease while the
// GPU might still be reading the pinned slots. The loop's catch now waits the
// command and records it here. The counter is diagnostics — the test asserts a
// failure waited before its lease released.

/// unchecked-invariant: owned by one runner, which is itself single-owner, so
/// the count is only touched by the task driving that runner.
final class DecodeFailureDrains: @unchecked Sendable {
    var count = 0
}

extension RealForwardRunner {
    /// Test seam: throw after layer 0's routed command is committed.
    /// Armed only by `TINYTITAN_FAULT_DECODE=after-layer0`.
    var faultDecodeAfterLayerZero: Bool {
        ProcessInfo.processInfo.environment["TINYTITAN_FAULT_DECODE"] == "after-layer0"
    }

    /// Committed commands a failed decode had to wait for (AUD-0089).
    var decodeFailureDrains: Int { failureDrains.count }

    func noteDecodeFailureDrain() { failureDrains.count += 1 }
}
