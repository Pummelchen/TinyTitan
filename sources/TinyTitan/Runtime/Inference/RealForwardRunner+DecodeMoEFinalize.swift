import Foundation
import Metal

// Decode-time routed-MoE finalisation: the pending-command hand-off, the slot
// publication and the shared-expert commit.
//
// Split out of `RealForwardRunner+DecodeMoE.swift` (2026-09-28) under the
// 500-line-per-file rule (Task 8 of the cleanup runbook) as pure code motion.
extension RealForwardRunner {

    func finishPendingRoutedCommand(
        _ pending: PendingRoutedCommand,
        waitIfNeeded: Bool
    ) throws {
        defer { pending.expertLease?.release() }
        // A staged Metal-I/O batch owns its source buffers until the compute
        // command has completed. If any command/error path exits early, leave
        // the cache entries empty rather than retaining a LOADING slot.
        var finalizedStagingTransfer = false
        defer {
            if let operation = pending.storageOperation {
                if operation.storage.requiresGPUFinalization,
                    !finalizedStagingTransfer
                {
                    model.failRoutedExpertStagingTransfer(plan: operation.plan)
                }
                operation.storage.releaseStagingTransfer()
            }
        }
        if waitIfNeeded {
            if let sharedCB = pending.sharedCB {
                try waitForCompletion(sharedCB)
            }
            if let phase1HitCB = pending.phase1HitCB {
                try waitForCompletion(phase1HitCB)
            }
            try waitForCompletion(pending.cb)
        } else if let err = pending.cb.error {
            throw ModelError.commandBufferFailed(
                detail: "routed layer command buffer: \(err)")
        }
        if let operation = pending.storageOperation {
            // Event-gated commands cannot complete before this operation is
            // terminal, so this is an error check, not a successful-I/O host
            // wait. A failed read is surfaced after safe no-op kernels have
            // prevented incomplete slot bytes from being dereferenced.
            try operation.storage.wait()
            if operation.storage.requiresGPUFinalization {
                try model.finalizeRoutedExpertStagingTransfer(plan: operation.plan)
                finalizedStagingTransfer = true
            }
            totalIOQueueNanos &+= operation.storage.submissionToStartNanos
            totalIoNanos &+= operation.storage.loadNanos
            totalMissIoNanos &+= operation.storage.loadNanos
            if let latest = pending.overlapCompletionClock?.latest(
                expected: pending.expectedOverlapCompletions)
            {
                let completed = operation.storage.completedNanos
                if completed > latest {
                    totalExposedIoNanos &+= completed - latest
                }
            }
        }
        if let sharedCB = pending.sharedCB, let err = sharedCB.error {
            throw ModelError.commandBufferFailed(
                detail: "shared-expert command buffer: \(err)")
        }
        if let phase1HitCB = pending.phase1HitCB, let err = phase1HitCB.error {
            throw ModelError.commandBufferFailed(
                detail: "routed phase-1 hit command buffer: \(err)")
        }
        if let sharedCB = pending.sharedCB {
            recordKernelGPU(role: "shared_expert", sharedCB)
        }
        if let phase1HitCB = pending.phase1HitCB {
            recordKernelGPU(role: "moe_phase1_hit", phase1HitCB)
        }
        recordKernelGPU(role: pending.kernelRole, pending.cb)
        totalCb2Nanos &+= pending.encodeAndCommitNanos
    }

    func writeActiveSlots(_ slots: [UInt32], into buffer: MTLBuffer) {
        let ptr = buffer.contents().assumingMemoryBound(to: UInt32.self)
        for i in 0..<slots.count { ptr[i] = slots[i] }
    }

    /// Encodes the shared dense MLP and commits it immediately.
    ///
    /// It depends only on `routedX`, which `tailCB` produces, so it can be
    /// queued the moment `tailCB` is committed -- before the router readback,
    /// not after it. Both sit on the same queue, so the GPU runs this while the
    /// CPU is blocked waiting for `tailCB` to report the routing.
    ///
    /// That ordering is the whole point. Encoding it after the readback left a
    /// measured 7.88 ms/token of GPU idle in the
    /// `attn_tail_router -> shared_expert` transition -- 0.197 ms per layer of
    /// command-buffer round trip during which the GPU had nothing queued, and
    /// the largest single component of decode's idle time.
    func encodeAndCommitSharedExpert(
        layer L: Int,
        completionClock: CommandCompletionClock?
    ) throws -> MTLCommandBuffer {
        let sharedProj = sharedExpertProjections[L]
        let D = UInt32(cfg.hiddenSize)
        guard let sharedCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        try shared.encode(
            commandBuffer: sharedCB,
            x: routedX,
            gate: sharedProj.gate,
            up: sharedProj.up,
            down: sharedProj.down,
            y: h1Buf,
            scratchGate: denseScratchGate,
            scratchUp: denseScratchUp,
            scratchAct: denseScratchAct)
        if cfg.sharedExpertGated {
            // out = sigmoid(shared_expert_gate(moeX)) * shared_mlp(moeX)
            let gateView = try requireTensorView(sharedProj.scalarGate, "shared-expert scalar gate")
            try encodeScalarGate(
                commandBuffer: sharedCB,
                view: gateView,
                x: routedX,
                y: try requireBuffer(sharedScalarGateBuf, "shared-expert gate buffer"),
                n: D)
            try requireElementwise().encodeSigmoidScalarMul(
                commandBuffer: sharedCB,
                y: h1Buf,
                gate: try requireBuffer(sharedScalarGateBuf, "shared-expert gate buffer"),
                count: cfg.hiddenSize)
        }
        completionClock?.track(sharedCB)
        sharedCB.commit()
        return sharedCB
    }
}
