import Foundation
import Metal
import Testing

@testable import Legacy6xRuntime
@testable import TinyTitanCore

/// A failed chunked prefill may only invalidate the slot it was run for.
///
/// The caller (`ServerModelSession+Generation`) already documents that "one
/// sequence failed; only its slot's KV/GDN is suspect" and resets that slot;
/// the runner used to call the whole-runner `reset()` in its own catch, so a
/// cancelled or failed prefill on a batched session wiped every other
/// sequence's live KV and GDN state and their next decode failed with
/// `prefillCursorMismatch` (AUD-0088).
@Suite struct PrefillSlotIsolationTests {
    private func makeRunner(slots: Int, context: MetalContext) throws -> RealForwardRunner {
        let dir = try QwenToySynthetic.write(weightBits: 8)
        let model = try Model.load(directoryURL: dir, device: context.device, expecting: .qwenToy())
        let runtime = try RuntimeConfiguration(forceLogitsHead: true)
        return try RealForwardRunner(
            model: model, context: context, maxContext: 128, slots: slots,
            runtimeConfiguration: runtime)
    }

    /// unchecked-invariant: the child task is the only writer of this buffer
    /// while it runs, and the test awaits it before reading any state.
    private struct BufferBox: @unchecked Sendable {
        let buffer: any MTLBuffer
    }

    private func logits(_ device: MTLDevice) throws -> MTLBuffer {
        try #require(
            device.makeBuffer(
                length: 1024 * MemoryLayout<Float16>.stride, options: .storageModeShared))
    }

    @Test func aFailedChunkedPrefillLeavesOtherSlotsAlone() async throws {
        let context = try MetalContext()
        let runner = try makeRunner(slots: 2, context: context)
        let logits = try logits(context.device)
        let kv = try #require(runner.kv)

        // Slot 0 is mid-conversation: 64 tokens prefilled, so its cursor is
        // somewhere a whole-runner reset would move to 0.
        let tokens = (0..<64).map { Int32($0 % 17 + 1) }
        _ = try await runner.prefillChunked(
            tokens: tokens[...], startPosition: 0, slot: 0,
            outputMode: .logits, config: .production(chunkTokens: 64), into: logits,
            onProgress: { _ in })
        let slot0Position = kv.position(slot: 0)
        #expect(slot0Position == tokens.count)

        // Slot 1 fails after its first of two chunks: the child cancels itself
        // from `onProgress`, so the loop's next `Task.checkCancellation()`
        // throws after that chunk's KV rows were written.
        let boxed = BufferBox(buffer: logits)
        let failure = await Self.runCancellingAfterFirstChunk {
            _ = try await runner.prefillChunked(
                tokens: tokens[...], startPosition: 0, slot: 1,
                outputMode: .logits, config: .production(chunkTokens: 32),
                into: boxed.buffer,
                onProgress: { _ in withUnsafeCurrentTask { $0?.cancel() } })
        }
        #expect(
            failure is CancellationError,
            "prefill did not fail: \(String(describing: failure))")

        // The other sequence's cursor is untouched; the failing slot is reset.
        #expect(
            kv.position(slot: 0) == slot0Position,
            "slot 0 lost its KV cursor when slot 1 failed")
        #expect(kv.position(slot: 1) == 0)
    }

    /// Runs `body` in a child task and returns the error it threw, if any.
    /// The child is the one cancelled, so the test task itself stays live.
    private static func runCancellingAfterFirstChunk(
        _ body: @escaping @Sendable () async throws -> Void
    ) async -> (any Error)? {
        let task = Task { () -> (any Error)? in
            do {
                try await body()
                return nil
            } catch {
                return error
            }
        }
        return await task.value
    }
}
