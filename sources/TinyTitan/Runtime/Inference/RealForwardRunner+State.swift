import Foundation
import Metal

// Reset, continuation and inference-state snapshot/restore for the runner.
// Split out of RealForwardRunner.swift (2026-09-28) under the 500-line-per-file
// rule as pure code motion.

extension RealForwardRunner {
    /// Keep the routed-expert slot cache wired across prefill as well as
    /// decode. The decision is `profile.keepExpertCacheWired`, the row's own
    /// measured value; the tri-state env override that used to force it either
    /// way measured a wash (-0.37%) and is gone.
    ///
    /// Decode reads the same expert bytes either way -- measured identical,
    /// 9.18 against 9.16 GiB at the same 70.5% hit rate -- but with ANE
    /// prefill it waits 3.6x longer on them (3129 ms against 859 ms). Same
    /// reads, far longer awaits, which points at the read *destinations*
    /// faulting rather than at the reads themselves. Holding the cache wired
    /// through prefill is the direct test of that.

    public func reset() {
        kv?.reset()
        gdnState?.reset()
        resetPLEState()
        qsaIndexer?.reset()
        resetTransientState()
    }

    /// Reset one slot's KV rows and GDN state, leaving the other sequences'
    /// live state intact. Used when a single batched sequence fails. It does not
    /// advise pages back to the OS (`KVCacheManager.reset(slot:)` only moves the
    /// cursor), so it is safe while another slot is mid-step.
    ///
    /// PLE and QSA transient state is per-runner, not per-slot, and is left to
    /// the whole-runner `reset()`.
    public func reset(slot: Int) {
        kv?.reset(slot: slot)
        gdnState?.reset(slot: slot)
    }

    /// Clear one sequence's state under the step gate, for starting a batched
    /// request. The KV and GDN regions are per-slot; the PLE latch and the
    /// transient cursors are per-runner, and are cleared here because the gate
    /// means no other step is in flight. The QSA indexer is per-sequence and
    /// only slot 0 may reset it.
    public func resetSequence(slot: Int) async {
        // A cancelled start has nothing to reset; skip rather than trap.
        do { try await forwardStepGate.acquire() } catch { return }
        kv?.reset(slot: slot)
        gdnState?.reset(slot: slot)
        resetPLEState()
        if slot == 0 { qsaIndexer?.reset() }
        resetTransientState()
        await forwardStepGate.release()
    }

    public var continuationPosition: Int {
        kv?.position ?? 0
    }

    public func prepareForContinuation(expectedPosition: Int) throws {
        guard let kv else {
            throw PrefillError.prefillCursorMismatch(
                "continuation requires an initialized KV cache")
        }
        guard expectedPosition > 0, kv.position == expectedPosition else {
            throw PrefillError.prefillCursorMismatch(
                "continuation expected KV position \(expectedPosition), current \(kv.position)")
        }
        resetTransientState()
    }

    public func captureInferenceState(
        maximumBytes: Int? = nil
    ) throws -> InferenceStateSnapshot {
        guard let kv, kv.position > 0 else {
            throw InferenceStateSnapshotError.invalidPosition(kv?.position ?? 0)
        }
        let kvLengths = try kv.snapshotSegmentLengths(at: kv.position)
        let gdnLengths = gdnState?.snapshotSegmentLengths() ?? []
        var payloadBytes = 0
        for length in kvLengths + gdnLengths {
            let (next, overflow) = payloadBytes.addingReportingOverflow(length)
            guard !overflow else { throw InferenceStateSnapshotError.integerOverflow }
            payloadBytes = next
        }
        if let maximumBytes, payloadBytes > maximumBytes {
            throw InferenceStateSnapshotError.exceedsLimit(
                bytes: payloadBytes,
                limit: maximumBytes)
        }
        var payload = Data()
        payload.reserveCapacity(payloadBytes)
        try kv.appendSnapshotPayload(to: &payload, segmentLengths: kvLengths)
        try gdnState?.appendSnapshotPayload(to: &payload, segmentLengths: gdnLengths)
        guard payload.count == payloadBytes else {
            throw InferenceStateSnapshotError.invalidPayloadSize(
                expected: payloadBytes,
                actual: payload.count)
        }
        return InferenceStateSnapshot(
            descriptor: InferenceStateSnapshotDescriptor(
                position: kv.position,
                kvSegmentLengths: kvLengths,
                gdnSegmentLengths: gdnLengths,
                payloadBytes: payloadBytes),
            payload: payload)
    }

    public func restoreInferenceState(_ snapshot: InferenceStateSnapshot) throws {
        do {
            let descriptor = snapshot.descriptor
            guard descriptor.version == InferenceStateSnapshotDescriptor.currentVersion else {
                throw InferenceStateSnapshotError.unsupportedVersion(descriptor.version)
            }
            guard descriptor.position > 0, descriptor.position <= maxContext else {
                throw InferenceStateSnapshotError.invalidPosition(descriptor.position)
            }
            let expectedBytes = try descriptor.validatedPayloadBytes()
            guard snapshot.payload.count == expectedBytes else {
                throw InferenceStateSnapshotError.invalidPayloadSize(
                    expected: expectedBytes,
                    actual: snapshot.payload.count)
            }
            guard let kv else { throw InferenceStateSnapshotError.invalidLayout }
            // Refuse when a subsystem this snapshot does not carry is live.
            //
            // `reset()` clears the sparse indexer and the PLE block;
            // `restoreInferenceState` only calls `resetTransientState()`, which
            // does not. Restoring would therefore leave the indexer ranking
            // blocks from pooled keys that were never rebuilt for this prefix,
            // and the n-gram hashing starting from the previous conversation's
            // predecessors. The model attends to a wrong subset of keys and
            // answers fluently and wrongly, with no error anywhere -- the
            // failure this project refuses. Throwing here costs the caller a
            // re-prefill, not a wrong answer.
            //
            // Both are non-nil only for a family that enables them
            // (Qwen3.8-Flash-Next), so the MoE families keep restoring. Carrying
            // these buffers in the snapshot, or replaying the prefix to rebuild
            // them, is the real fix; it is recorded in
            // the wiki's project tracker.
            if qsaIndexer != nil {
                throw InferenceStateSnapshotError.stateNotInSnapshot("sparse-indexer")
            }
            if pleBlock != nil {
                throw InferenceStateSnapshotError.stateNotInSnapshot("PLE")
            }
            try snapshot.payload.withUnsafeBytes { bytes in
                var offset = 0
                try kv.restoreSnapshot(
                    position: descriptor.position,
                    segmentLengths: descriptor.kvSegmentLengths,
                    bytes: bytes,
                    offset: &offset)
                if let gdnState {
                    try gdnState.restoreSnapshot(
                        segmentLengths: descriptor.gdnSegmentLengths,
                        bytes: bytes,
                        offset: &offset)
                } else if !descriptor.gdnSegmentLengths.isEmpty {
                    throw InferenceStateSnapshotError.invalidLayout
                }
                guard offset == bytes.count else {
                    throw InferenceStateSnapshotError.invalidLayout
                }
            }
            resetTransientState()
        } catch {
            reset()
            throw error
        }
    }

    func resetTransientState() {
        prefillChunkState.reset()
        rdadviseSkipUntilPosition = -1
        rdadviseAdaptiveState.reset()
        rdadviseAdaptivePosition = -1
        rdadviseAdaptivePositionBytes = 0
    }
}
