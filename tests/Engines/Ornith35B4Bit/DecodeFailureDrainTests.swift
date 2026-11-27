import Foundation
import Metal
import Testing

@testable import Ornith35B4Bit
@testable import TinyTitanCore

/// A thrown decode must wait the routed command it already committed.
///
/// The lease that pins the expert slots releases when the pending command is
/// dropped with the stack; without a wait the cache may evict and overwrite
/// those slots while the GPU is still reading them (AUD-0089). The loop's
/// `catch` now drains the pending command before rethrowing.
///
/// Model-gated: `TINYTITAN_FAULT_DECODE_MODEL` names a shipped 4-bit Ornith
/// install. CI stays model-free (the suite is skipped there), and the seam is
/// armed only by this test.
@Suite struct DecodeFailureDrainTests {
    private static var install: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_FAULT_DECODE_MODEL"]
    }

    private func logits(_ device: MTLDevice) throws -> MTLBuffer {
        try #require(
            device.makeBuffer(
                length: 2048 * MemoryLayout<Float16>.stride, options: .storageModeShared))
    }

    @Test(.enabled(if: install != nil))
    func aThrownDecodeWaitsItsCommittedRoutedCommand() async throws {
        let directory = URL(fileURLWithPath: try #require(Self.install))
        let context = try MetalContext()
        let model = try Model.load(directoryURL: directory, device: context.device)
        let runner = try RealForwardRunner(
            model: model, context: context, maxContext: 64, slots: 1,
            runtimeConfiguration: try RuntimeConfiguration(forceLogitsHead: true))
        let logits = try logits(context.device)

        // Armed: the loop commits layer 0's routed command and then throws, so
        // the catch has something to drain.
        setenv("TINYTITAN_FAULT_DECODE", "after-layer0", 1)
        var threw = false
        do {
            try await runner.produce(token: 1, position: 0, into: logits)
        } catch {
            threw = true
        }
        #expect(threw, "the injected fault did not reach the decode loop")
        #expect(
            runner.decodeFailureDrains == 1,
            "the failed decode did not wait its committed command")

        // Disarmed: a normal decode drains nothing, and the seam leaves the
        // runner usable.
        unsetenv("TINYTITAN_FAULT_DECODE")
        try await runner.produce(token: 1, position: 0, into: logits)
        #expect(runner.decodeFailureDrains == 1)
    }
}
