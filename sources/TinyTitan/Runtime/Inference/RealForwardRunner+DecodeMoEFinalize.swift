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
            if let phase1MissCB = pending.phase1MissCB {
                try waitForCompletion(phase1MissCB)
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
        if let phase1MissCB = pending.phase1MissCB {
            if let err = phase1MissCB.error {
                throw ModelError.commandBufferFailed(
                    detail: "routed early phase-1 miss fixup command buffer: \(err)")
            }
            recordKernelGPU(role: "moe_phase1_miss_fixup", phase1MissCB)
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

    /// Encode one row's shared expert into a caller-owned command buffer.
    ///
    /// `encodeAndCommitSharedExpert` is the decode loop's runner-wide form:
    /// one row, `routedX -> h1Buf`, committed immediately. The fused decode
    /// loop needs the same arithmetic on a row slice of a batch buffer and on
    /// a command buffer it shares with the rest of the layer, so the two row
    /// arguments and the buffer are explicit here. The intermediate scratch is
    /// runner-wide on purpose: rows are encoded in order on one command
    /// buffer, so each row's down projection reads the scratch before the next
    /// row's gate projection overwrites it.
    func encodeSharedExpert(
        layer L: Int,
        x: MTLBuffer, xOffset: Int,
        y: MTLBuffer, yOffset: Int,
        commandBuffer: MTLCommandBuffer
    ) throws {
        let sharedProj = sharedExpertProjections[L]
        let D = UInt32(cfg.hiddenSize)
        try shared.encode(
            commandBuffer: commandBuffer,
            x: x, xOffset: xOffset,
            gate: sharedProj.gate,
            up: sharedProj.up,
            down: sharedProj.down,
            y: y, yOffset: yOffset,
            scratchGate: denseScratchGate,
            scratchUp: denseScratchUp,
            scratchAct: denseScratchAct)
        if cfg.sharedExpertGated {
            // out = sigmoid(shared_expert_gate(moeX)) * shared_mlp(moeX)
            let gateView = try requireTensorView(sharedProj.scalarGate, "shared-expert scalar gate")
            let gateBuf = try requireBuffer(sharedScalarGateBuf, "shared-expert gate buffer")
            try encodeScalarGate(
                commandBuffer: commandBuffer,
                view: gateView,
                x: x, xOffset: xOffset,
                y: gateBuf, yOffset: 0,
                n: D)
            try requireElementwise().encodeSigmoidScalarMul(
                commandBuffer: commandBuffer,
                y: y, yOffset: yOffset,
                gate: gateBuf, gateOffset: 0,
                count: cfg.hiddenSize)
        }
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
        guard let sharedCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        try encodeSharedExpert(
            layer: L,
            x: routedX, xOffset: 0,
            y: h1Buf, yOffset: 0,
            commandBuffer: sharedCB)
        completionClock?.track(sharedCB)
        sharedCB.commit()
        return sharedCB
    }
}
